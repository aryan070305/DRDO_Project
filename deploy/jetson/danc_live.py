#!/usr/bin/env python3
"""Torch-free launcher for the live D-ANC engine on the Jetson (wraps danc.inference.realtime unchanged).

    NOT VERIFIED ON JETSON HARDWARE IN THIS PROJECT (the simulated mode is tested on macOS).

Why: ``python -m danc.inference.realtime`` imports ``danc.inference.runners``, which imports torch at
module level; the Jetson runtime has no torch.  This launcher registers a torch-free runner
(deploy/jetson/danc_jetson_rt.py, same math) and then calls ``danc.inference.realtime.main`` with all
remaining arguments, so the live code path is exactly the one validated on the Mac.

Examples (run from the project / bundle root, venv activated):
    python3 deploy/jetson/danc_live.py --list-devices
    # HQ (40 ms algorithmic latency), USB interface: input ch0 = primary boom mic, ch1 = outward reference mic
    python3 deploy/jetson/danc_live.py --onnx exports/dancnet_v3hq_cont.onnx --in-device USB --out-device USB --log engine.jsonl
    # LL (20 ms)
    python3 deploy/jetson/danc_live.py --onnx exports/dancnet_ll.onnx --in-device USB --out-device USB
    # no audio hardware: paced simulation with a recorded dual-mic scene
    python3 deploy/jetson/danc_live.py --onnx exports/dancnet_v3hq_cont.onnx --paced \
        --simulate data/testsets/dualmic_v1/primary/0000_stationary_m05.flac \
        --simulate-ref data/testsets/dualmic_v1/reference/0000_stationary_m05.flac --out sim_out.wav

All options of danc.inference.realtime are accepted (--onnx --threads --in-device --out-device --in-channels
--primary-ch --reference-ch --no-lms --bypass --gain-floor-db --limiter-dbfs --simulate --simulate-ref --paced
--out --log --list-devices).  Launcher options:
    --cpus auto|LIST|none   CPU pinning (default auto = the last 2 ONLINE CPUs, valid in every nvpmodel mode)
    --rt-priority N         SCHED_FIFO priority for this process (default: unchanged; the systemd unit sets 80)
    --providers LIST        ONNX Runtime execution providers, comma separated (default CPUExecutionProvider)
    --backend ort|trt       trt = TensorRT engine via trt_runner.py (NOT VERIFIED ON HARDWARE)
    --trt-engine PATH       engine built by build_trt_engine.sh (with --backend trt)
    --cuda-graph            capture TensorRT enqueue into CUDA graphs (with --backend trt)
    --no-preflight          skip the audio-device capability check before going live
    --no-safety             use the plain HybridEngine (no NaN/Inf input guard, no [-1,1] output clamp)
    --watchdog-s S          live mode: exit with code 4 when no audio block arrives for S seconds (USB interface
                            unplugged / stream stalled) so that systemd restarts the service (default 3; 0 = off;
                            needs the safety wrapper)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FS = 16000


def _ensure_paths():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        import danc  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(ROOT / "src"))


def _device(v):
    if v is None:
        return None
    return int(v) if str(v).isdigit() else v


def preflight(in_device, out_device, in_channels: int) -> bool:
    """Check that the interface can do 16 kHz duplex with the requested channels; print remedies if not."""
    try:
        import sounddevice as sd
    except OSError as e:     # libportaudio2 missing
        print(f"[preflight] FAIL: sounddevice cannot load PortAudio ({e}). sudo apt-get install libportaudio2",
              file=sys.stderr)
        return False
    for name, v in (("--in-device", in_device), ("--out-device", out_device)):
        if v is not None and str(v).isdigit():
            print(f"[preflight] WARNING: {name} {v} is numeric. danc.inference.realtime passes it as a NAME "
                  "substring; use a unique part of the name instead, e.g. 'USB Audio' or 'hw:2,0'.", file=sys.stderr)
    ok = True
    try:
        sd.check_input_settings(device=_device(in_device), channels=in_channels, samplerate=FS, dtype="float32")
    except Exception as e:  # noqa: BLE001 - sounddevice raises several types
        ok = False
        print(f"[preflight] FAIL input  {in_device!r}: {in_channels} ch @ {FS} Hz not supported: {e}", file=sys.stderr)
    try:
        sd.check_output_settings(device=_device(out_device), channels=2, samplerate=FS, dtype="float32")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[preflight] FAIL output {out_device!r}: 2 ch @ {FS} Hz not supported: {e}", file=sys.stderr)
    if not ok:
        print("[preflight] Remedies: (1) run --list-devices and pick a unique name substring; (2) if the interface "
              "cannot run at 16 kHz, use an ALSA 'plug' PCM (see QUICKSTART.md, troubleshooting) or the PipeWire "
              "device; (3) check that no other program holds the device (fuser -v /dev/snd/*).", file=sys.stderr)
    else:
        print(f"[preflight] OK: in={in_device!r} ({in_channels} ch), out={out_device!r} (2 ch) at {FS} Hz")
    return ok


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lp = argparse.ArgumentParser(add_help=False)
    lp.add_argument("--cpus", default="auto")
    lp.add_argument("--rt-priority", type=int, default=0)
    lp.add_argument("--providers", default=None)
    lp.add_argument("--backend", choices=["ort", "trt"], default="ort")
    lp.add_argument("--trt-engine", default=None)
    lp.add_argument("--cuda-graph", action="store_true")
    lp.add_argument("--no-preflight", action="store_true")
    lp.add_argument("--no-safety", action="store_true")
    lp.add_argument("--watchdog-s", type=float, default=3.0)
    lp.add_argument("-h", "--help", action="store_true")
    la, rest = lp.parse_known_args(argv)
    if la.help:
        print(__doc__)
        return 0

    # one BLAS/OpenMP thread: the audio path is single-threaded; set before numpy is imported
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _ensure_paths()
    import danc_jetson_rt as rt

    if la.providers:
        os.environ[rt.PROVIDERS_ENV] = la.providers
    if la.backend == "trt":
        if not la.trt_engine:
            print("--backend trt needs --trt-engine <file.engine>", file=sys.stderr)
            return 2
        os.environ[rt.BACKEND_ENV] = "trt"
        os.environ[rt.TRT_ENGINE_ENV] = str(Path(la.trt_engine).resolve())
        os.environ[rt.TRT_CUDA_GRAPH_ENV] = "1" if la.cuda_graph else "0"

    rt.install_runners_shim(force=True)
    import danc.inference.realtime as realtime

    if not la.no_safety:
        realtime.HybridEngine = rt.SafeHybridEngine    # build_engine() looks the class up at call time

    # options that matter for the pre-flight check (parsed again, unchanged, by realtime.main)
    rp = argparse.ArgumentParser(add_help=False)
    rp.add_argument("--in-device", default=None)
    rp.add_argument("--out-device", default=None)
    rp.add_argument("--in-channels", type=int, default=2)
    rp.add_argument("--simulate", default=None)
    rp.add_argument("--list-devices", action="store_true")
    ra, _ = rp.parse_known_args(rest)
    live = not ra.simulate and not ra.list_devices

    info = rt.apply_rt_settings(rt.pick_rt_cpus(la.cpus), la.rt_priority or None)
    if not ra.list_devices:
        print(f"[danc_live] backend={la.backend} providers={os.environ.get(rt.PROVIDERS_ENV, 'CPUExecutionProvider')} "
              f"safety={'off' if la.no_safety else 'on'} cpus={info['cpus']} sched={info['sched']}", flush=True)
    if live and not la.no_preflight and not preflight(ra.in_device, ra.out_device, ra.in_channels):
        return 3
    if live and la.watchdog_s > 0:
        if la.no_safety:
            print("[danc_live] WARNING: --no-safety disables the stall watchdog (it reads the safety wrapper's "
                  "heartbeat)", file=sys.stderr)
        else:
            rt.start_watchdog(timeout_s=la.watchdog_s)
    realtime.main(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
