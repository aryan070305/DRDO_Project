#!/usr/bin/env python3
"""Torch-free runtime helpers for running D-ANC on a Jetson (or any Linux/macOS box).

WHY THIS FILE EXISTS
--------------------
``danc.inference.realtime`` (the live audio app) imports ``danc.inference.runners``, and
``runners.py`` does ``import torch`` at module level (for ``TorchStepRunner``).  The Jetson
runtime is deliberately installed WITHOUT PyTorch (a ~GB dependency that is not needed to run
an ONNX model), so ``python -m danc.inference.realtime`` would stop with
``ModuleNotFoundError: No module named 'torch'`` on a clean Jetson install.  Rather than
modifying ``src/`` this module provides:

  * :class:`ORTStepRunner` - the same math as ``danc.inference.runners.ORTStepRunner``
    (identical session options, state hand-over and outputs), without torch.  The execution
    providers can be chosen with the environment variable ``DANC_ORT_PROVIDERS``
    (e.g. ``CUDAExecutionProvider,CPUExecutionProvider`` with the onnxruntime-gpu wheel).
  * :class:`EngineRunnerAdapter` - lets the TensorRT / ORT runners of ``trt_runner.py``
    (which return ``spp`` as ``[1, 1, F]`` and carry no look-ahead) drive ``HybridEngine``.
  * :func:`install_runners_shim` - registers a torch-free ``danc.inference.runners`` module so
    that ``danc.inference.realtime`` can be imported and run unchanged.
  * :class:`SafeHybridEngine` - ``HybridEngine`` plus input/output sanitisation (non-finite
    samples -> 0, output hard-clipped to [-1, 1]) so that a software glitch can never send
    NaN/Inf to the DAC / radio, and a bounded per-block timing history (the unbounded one makes
    the live statistics, computed inside the audio callback, cost ~25 ms after 1 h of operation).
    For finite input the audio output is bit-identical to ``HybridEngine`` (the limiter already
    keeps it below -1 dBFS).
  * :func:`start_watchdog` - exits the live process (so systemd restarts it) when the audio
    callback stops arriving, e.g. after the USB interface is unplugged; realtime.run_live would
    otherwise wait forever.
  * :func:`pick_rt_cpus` / :func:`apply_rt_settings` - CPU pinning that works in every
    nvpmodel mode (15 W mode has only CPUs 0-3 online, 30 W only 0-7, so a hard-coded
    ``taskset -c 10,11`` fails there) and SCHED_FIFO.

The code paths that touch TensorRT/CUDA are NOT VERIFIED ON HARDWARE IN THIS PROJECT
(development was on macOS).  The ONNX Runtime paths are exercised by verify_install.py and
tests/test_deploy_kit.py on the development machine.
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROVIDERS_ENV = "DANC_ORT_PROVIDERS"
BACKEND_ENV = "DANC_BACKEND"            # "ort" (default) or "trt"
TRT_ENGINE_ENV = "DANC_TRT_ENGINE"      # path of the TensorRT engine when DANC_BACKEND=trt
TRT_CUDA_GRAPH_ENV = "DANC_TRT_CUDA_GRAPH"


def _providers_from_env(providers=None):
    if providers:
        return list(providers)
    env = os.environ.get(PROVIDERS_ENV, "").strip()
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    return ["CPUExecutionProvider"]


class ORTStepRunner:
    """Torch-free copy of ``danc.inference.runners.ORTStepRunner`` (same math, same interface).

        runner.reset()
        spec_out, spp = runner(spec)          # spec [1,1,161,2] float32 -> ([1,1,161,2], [161])
        spec_out, spp = runner(spec, ref)     # two-microphone models (ONNX input 'spec_ref'), same shapes

    Like the original, ``n_mics`` is 2 when the ONNX graph has a ``spec_ref`` input (two-microphone DANCNet,
    exported by danc.inference.export_onnx with in_ch=2); HybridEngine then bypasses the NLMS and feeds the
    reference-mic spectrum to the network.  With ``ref=None`` a two-mic model gets a zero reference (dead mic).
    """

    def __init__(self, onnx_path: str, threads: int = 1, providers=None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.onnx_path = str(onnx_path)
        self.sess = ort.InferenceSession(self.onnx_path, so, providers=_providers_from_env(providers))
        self.providers = self.sess.get_providers()
        self.state_inputs = [i for i in self.sess.get_inputs() if i.name.startswith("state_in_")]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.lookahead = int(meta.get("lookahead", 0))   # written by danc.inference.export_onnx
        self.n_mics = 2 if any(i.name == "spec_ref" for i in self.sess.get_inputs()) else 1
        self.reset()

    def reset(self):
        self.states = {i.name: np.zeros(i.shape, np.float32) for i in self.state_inputs}

    def __call__(self, spec: np.ndarray, ref: np.ndarray | None = None):
        feeds = {"spec": spec.astype(np.float32, copy=False), **self.states}
        if self.n_mics == 2:
            feeds["spec_ref"] = (np.zeros_like(spec, dtype=np.float32) if ref is None
                                 else ref.astype(np.float32, copy=False))
        outs = self.sess.run(self.out_names, feeds)
        d = dict(zip(self.out_names, outs))
        for k in self.states:
            self.states[k] = d[k.replace("state_in_", "state_out_")]
        return d["spec_out"], d["spp"].reshape(-1)


def onnx_n_mics(onnx_path: str) -> int:
    """2 for a two-microphone model (ONNX input 'spec_ref'), else 1."""
    import onnxruntime as ort

    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return 2 if any(i.name == "spec_ref" for i in s.get_inputs()) else 1


class EngineRunnerAdapter:
    """Wrap a ``trt_runner.py`` runner (TRTStepRunner or its ORTStepRunner) for HybridEngine.

    trt_runner returns ``spp`` with shape [1, 1, F]; HybridEngine needs a 1-D [F] vector and a
    ``lookahead`` attribute (TensorRT engines carry no ONNX metadata, so it is passed in).
    """

    n_mics = 1      # trt_runner.py feeds only 'spec': single-microphone models only

    def __init__(self, runner, lookahead: int):
        self.runner = runner
        self.lookahead = int(lookahead)

    def reset(self):
        self.runner.reset()

    def __call__(self, spec: np.ndarray):
        y, spp = self.runner(spec)
        return y, np.asarray(spp).reshape(-1)


def onnx_lookahead(onnx_path: str) -> int:
    import onnxruntime as ort

    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return int(s.get_modelmeta().custom_metadata_map.get("lookahead", 0))


def import_trt_runner():
    """Import deploy/jetson/trt_runner.py as a module (it is a script, not a package member)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import trt_runner  # noqa: E402

    return trt_runner


def make_trt_runner(engine_path: str, onnx_path: str, cuda_graph: bool = False):
    """TensorRT runner usable by HybridEngine.  NOT VERIFIED ON HARDWARE IN THIS PROJECT."""
    if onnx_n_mics(onnx_path) != 1:
        raise NotImplementedError(f"{onnx_path} is a two-microphone model (input 'spec_ref'); trt_runner.py only "
                                  "feeds 'spec'. Use the ONNX Runtime backend (--backend ort) for this model.")
    tr = import_trt_runner()
    return EngineRunnerAdapter(tr.TRTStepRunner(str(engine_path), use_cuda_graph=cuda_graph),
                               onnx_lookahead(onnx_path))


def make_runner(onnx_path: str, threads: int = 1, providers=None):
    """Runner chosen by the environment: DANC_BACKEND=trt + DANC_TRT_ENGINE, else ONNX Runtime."""
    if os.environ.get(BACKEND_ENV, "ort").lower() == "trt":
        eng = os.environ.get(TRT_ENGINE_ENV)
        if not eng:
            raise RuntimeError(f"{BACKEND_ENV}=trt needs {TRT_ENGINE_ENV}=<path to .engine>")
        return make_trt_runner(eng, onnx_path, os.environ.get(TRT_CUDA_GRAPH_ENV, "0") == "1")
    return ORTStepRunner(onnx_path, threads=threads, providers=providers)


def install_runners_shim(force: bool = True) -> types.ModuleType:
    """Register a torch-free ``danc.inference.runners`` so danc.inference.realtime imports without torch.

    With ``force=False`` an already importable original module (torch present) is kept.
    """
    name = "danc.inference.runners"
    if not force:
        try:
            import danc.inference.runners as orig  # noqa: F401

            return orig
        except ImportError:
            pass
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__danc_torchfree__", False):
        return existing
    mod = types.ModuleType(name)
    mod.__doc__ = "Torch-free replacement registered by deploy/jetson/danc_jetson_rt.py"
    mod.__danc_torchfree__ = True

    class _ORTStepRunnerShim:      # signature of danc.inference.runners.ORTStepRunner
        def __new__(cls, onnx_path: str, threads: int = 1, providers=None):
            return make_runner(onnx_path, threads=threads, providers=providers)

    class TorchStepRunner:          # pragma: no cover - only reached if someone asks for torch
        def __init__(self, *a, **k):
            raise RuntimeError("TorchStepRunner needs PyTorch, which is not part of the Jetson runtime; "
                               "use the ONNX models in exports/")

    mod.ORTStepRunner = _ORTStepRunnerShim
    mod.TorchStepRunner = TorchStepRunner
    import danc.inference as pkg

    sys.modules[name] = mod
    pkg.runners = mod
    return mod


def _safe_engine_class():
    from danc.inference.engine import HybridEngine

    class SafeHybridEngine(HybridEngine):
        """HybridEngine with non-finite-input protection, a hard [-1, 1] output clamp and a bounded timing log.

        Bounded timing log (round 1: HybridEngine.timing was an unbounded list and danc.inference.realtime called
        timing_stats() over the WHOLE list inside the audio callback every 500 blocks; since round 2 the engine keeps
        a bounded deque and the statistics run in the logging thread, but this window keeps them short).  Round-1 behaviour measured on an Apple M4: 0.4 ms after 40 s, 25 ms after 1 h, 198 ms after 8 h
        (i.e. a dropout every 5 s) and ~280 MB/day of memory.  Keeping only the last TIMING_WINDOW..2x
        entries makes the live statistics a sliding window (15-30 s) with constant cost.
        """

        TIMING_WINDOW = 1500

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.sanitised_blocks = 0

        def _trim_timing(self):
            # works for a list and for the bounded deque used by danc.inference.engine since round 2
            # (a deque has no slice deletion: `del deque[:n]` raised TypeError after 3 000 blocks = 30 s)
            if len(self.timing) > 2 * self.TIMING_WINDOW:
                keep = list(self.timing)[-self.TIMING_WINDOW:]
                self.timing.clear()
                self.timing.extend(keep)

        def process_block(self, prim, ref=None):
            prim = np.asarray(prim, dtype=np.float64)
            bad = not np.all(np.isfinite(prim))
            if bad:
                prim = np.nan_to_num(prim, nan=0.0, posinf=0.0, neginf=0.0)
            if ref is not None:
                ref = np.asarray(ref, dtype=np.float64)
                if not np.all(np.isfinite(ref)):
                    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
                    bad = True
            y = super().process_block(prim, ref)
            if not np.all(np.isfinite(y)):
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                bad = True
            if bad:
                self.sanitised_blocks += 1
            self._trim_timing()
            HEARTBEAT["t"] = time.monotonic()          # read by start_watchdog()
            HEARTBEAT["blocks"] += 1
            return np.clip(y, -1.0, 1.0)

    return SafeHybridEngine


# ----------------------------------------------------------------------------- live watchdog
HEARTBEAT = {"t": None, "blocks": 0}     # last SafeHybridEngine.process_block() (time.monotonic()) and count


def start_watchdog(timeout_s: float = 3.0, startup_s: float = 20.0, on_stall=None, poll_s: float = 0.5):
    """Stop the process when the audio callback stops arriving (USB interface unplugged, stream stalled).

    danc.inference.realtime.run_live waits forever for the next statistics record; if PortAudio stops calling
    the callback (e.g. the USB interface is unplugged) the live engine would hang silently and systemd would
    never restart it.  This daemon thread calls ``on_stall(message)`` (default: print and ``os._exit(4)``, so
    ``Restart=on-failure`` restarts the service) when no block was processed within ``startup_s`` of the start,
    or when no new block arrived for ``timeout_s`` (300 blocks at 10 ms).  Needs SafeHybridEngine (the default).
    """
    import threading

    HEARTBEAT["t"], HEARTBEAT["blocks"] = None, 0
    t0 = time.monotonic()

    def _exit(msg):
        print(f"[watchdog] {msg} - exiting with code 4 (systemd restarts the service)", file=sys.stderr, flush=True)
        os._exit(4)

    stall = on_stall or _exit

    def loop():
        while True:
            time.sleep(poll_s)
            now, last = time.monotonic(), HEARTBEAT["t"]
            if last is None:
                if now - t0 > startup_s:
                    stall(f"no audio block processed within {startup_s:g} s of start")
                    return
            elif now - last > timeout_s:
                stall(f"no audio block for {now - last:.1f} s after {HEARTBEAT['blocks']} blocks "
                      "(audio device unplugged or stream stalled)")
                return

    th = threading.Thread(target=loop, name="danc-watchdog", daemon=True)
    th.start()
    return th


_SAFE_CLS = None


def __getattr__(name):           # lazy: importing this module must not import danc
    global _SAFE_CLS
    if name == "SafeHybridEngine":
        if _SAFE_CLS is None:
            _SAFE_CLS = _safe_engine_class()
        return _SAFE_CLS
    raise AttributeError(name)


# ----------------------------------------------------------------------------- real-time settings
def online_cpus() -> list[int]:
    """CPUs currently online (Linux sysfs), else the affinity set / os.cpu_count()."""
    p = Path("/sys/devices/system/cpu/online")
    if p.exists():
        cpus: list[int] = []
        for part in p.read_text().strip().split(","):
            if "-" in part:
                a, b = part.split("-")
                cpus.extend(range(int(a), int(b) + 1))
            elif part:
                cpus.append(int(part))
        return sorted(cpus)
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def pick_rt_cpus(spec: str | None, n: int = 2) -> list[int] | None:
    """'auto' -> the last ``n`` online CPUs (CPU0 handles most IRQs); '10,11' -> explicit list."""
    if not spec or spec == "none":
        return None
    online = online_cpus()
    if spec == "auto":
        return online[-n:] if len(online) > n else online[-1:]
    want = [int(c) for c in spec.split(",") if c.strip()]
    ok = [c for c in want if c in online]
    if not ok:
        print(f"[rt] WARNING: requested CPUs {want} are not online (online: {online}); "
              f"falling back to {online[-n:]}. In 15 W mode only CPUs 0-3 are online, in 30 W only 0-7.",
              file=sys.stderr)
        return online[-n:]
    return ok


def apply_rt_settings(cpus: list[int] | None, rt_priority: int | None) -> dict:
    """Pin the process and switch to SCHED_FIFO (Linux only; needs root, CAP_SYS_NICE or rtprio limits)."""
    info = {"cpus": None, "sched": "OTHER"}
    if cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cpus))
            info["cpus"] = sorted(os.sched_getaffinity(0))
        except OSError as e:
            print(f"[rt] WARNING: could not pin to CPUs {cpus}: {e}", file=sys.stderr)
    if rt_priority and hasattr(os, "sched_setscheduler"):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(rt_priority)))
            info["sched"] = f"FIFO/{rt_priority}"
        except (OSError, PermissionError) as e:
            print(f"[rt] WARNING: SCHED_FIFO {rt_priority} refused ({e}); run as root, via the systemd unit, "
                  "or add '@audio - rtprio 95' to /etc/security/limits.d/ and re-login.", file=sys.stderr)
    return info
