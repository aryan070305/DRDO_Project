"""Live / simulated-live D-ANC prototype application.

Live (USB interface: ch0 = primary boom mic, ch1 = outward reference mic, stereo headset out):
    python -m danc.inference.realtime --onnx exports/dancnet_hqft.onnx --in-device USB --out-device USB

Simulated live (same per-block callback, paced by a software clock; no audio hardware needed):
    python -m danc.inference.realtime --onnx exports/dancnet_hqft.onnx \
        --simulate data/testsets/dualmic_v1/primary/0100_impulsive_p05.flac \
        --simulate-ref data/testsets/dualmic_v1/reference/0100_impulsive_p05.flac --out reports/audio/sim_out.wav

Other options: --no-lms (DNN only), --bypass (pass-through, for loop-back latency tests),
--gain-floor-db -25 (comfort floor), --list-devices.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time

import numpy as np
import soundfile as sf

from danc.dsp import DEFAULT
from danc.inference.engine import EngineConfig, HybridEngine
from danc.inference.runners import ORTStepRunner

HOP = DEFAULT.hop
FS = DEFAULT.fs


def build_engine(a) -> HybridEngine:
    runner = None if a.bypass else ORTStepRunner(a.onnx, threads=a.threads)
    cfg = EngineConfig(use_lms=not a.no_lms and not a.bypass, spp_gate=True,
                       gain_floor_db=a.gain_floor_db, limiter_dbfs=a.limiter_dbfs)
    if getattr(a, "fallback_onnx", None):
        cfg.ref_floor_dbfs, cfg.ref_rel_dead_db = a.ref_floor_dbfs, a.ref_rel_dead_db
    if getattr(a, "two_mic_cascade", False):
        cfg.lms_two_mic = True
    if getattr(a, "fallback_onnx", None) and runner is not None:
        # two-microphone network + single-mic network in hot standby (reference-mic health monitor, engine.py)
        return HybridEngine(runner, cfg, fallback_runner=ORTStepRunner(a.fallback_onnx, threads=a.threads))
    return HybridEngine(runner, cfg)


def run_live(a):
    import sounddevice as sd

    eng = build_engine(a)
    eng.cfg.timing_window = 6000            # live mode: keep the last 60 s of hop timings only
    eng.reset()
    stats = {"blocks": 0, "xruns": 0, "late": 0}
    log_q: "queue.Queue[dict]" = queue.Queue()

    def callback(indata, outdata, frames, t, status):
        if status:
            stats["xruns"] += 1
        prim = indata[:, a.primary_ch].astype(np.float64)
        ref = indata[:, a.reference_ch].astype(np.float64) if indata.shape[1] > a.reference_ch and not a.no_lms else None
        y = eng.process_block(prim, ref)
        outdata[:] = np.repeat(y[:, None], outdata.shape[1], 1).astype(np.float32)
        stats["blocks"] += 1
        if eng.timing and eng.timing[-1] > HOP / FS:
            stats["late"] += 1
        if stats["blocks"] % 500 == 0:
            rec = {"t": time.time(), **stats}         # cheap; statistics are computed in the logging thread
            if getattr(eng, "fallback", None) is not None:   # reference-health monitor running
                rec.update({"ref_dead": bool(eng.ref_dead), "ref_switches": eng.ref_switches})
            log_q.put(rec)

    with sd.Stream(samplerate=FS, blocksize=HOP, dtype="float32", latency="low",
                   device=(a.in_device, a.out_device), channels=(a.in_channels, 2), callback=callback) as st:
        print(f"[live] running: input latency {st.latency[0]*1000:.1f} ms, output latency {st.latency[1]*1000:.1f} ms "
              f"(+ {eng.latency_ms:.0f} ms algorithmic). Ctrl-C to stop.", flush=True)
        logf = open(a.log, "a") if a.log else None
        try:
            while True:
                rec = log_q.get()
                rec.update(eng.timing_stats())      # outside the audio callback (no real-time stall)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
                if logf:
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
        except KeyboardInterrupt:
            pass


def run_simulated(a):
    """Feed a file block-by-block on a 10 ms software clock (as an audio driver would)."""
    eng = build_engine(a)
    x, fs = sf.read(a.simulate, dtype="float64")
    assert fs == FS, "simulation input must be 16 kHz"
    if x.ndim == 2:
        prim, ref = x[:, a.primary_ch], (x[:, a.reference_ch] if x.shape[1] > 1 else None)
    else:
        prim, ref = x, None
    if a.simulate_ref:
        ref, _ = sf.read(a.simulate_ref, dtype="float64")
    if a.no_lms:
        ref = None
    nb = len(prim) // HOP
    out = np.zeros(nb * HOP)
    misses, t_next = 0, time.perf_counter()
    for k in range(nb):
        sl = slice(k * HOP, (k + 1) * HOP)
        out[sl] = eng.process_block(prim[sl], None if ref is None else ref[sl])
        t_next += HOP / FS
        slack = t_next - time.perf_counter()
        if slack < 0:
            misses += 1
        elif a.paced:
            time.sleep(slack)
    st = eng.timing_stats()
    st.update({"deadline_misses_paced": misses, "audio_s": nb * HOP / FS,
               "lookahead_frames": eng.lookahead, "algorithmic_latency_ms": eng.latency_ms})
    if getattr(eng, "fallback", None) is not None:
        st.update({"ref_switches": eng.ref_switches, "ref_dead_at_end": eng.ref_dead})
    print(json.dumps(st, indent=1))
    if a.out:
        sf.write(a.out, out[DEFAULT.win - DEFAULT.hop + HOP * eng.lookahead:].astype(np.float32), FS)
    return st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="exports/dancnet_v3hq_cont.onnx")   # deployed HQ (round 2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fallback-onnx", default=None,
                    help="two-microphone --onnx only: single-mic model (same look-ahead) kept in hot standby and "
                         "cross-faded in when the reference mic is dead for 0.5 s: digital silence (< --ref-floor-dbfs) "
                         "or >= --ref-rel-dead-db below an active primary")
    ap.add_argument("--ref-floor-dbfs", type=float, default=-100.0)
    ap.add_argument("--two-mic-cascade", action="store_true",
                    help="two-microphone --onnx only: run the gated NLMS first and feed its output plus the reference to "
                         "the network (round-2 recommended two-mic configuration, reports/results/cascade_decision.md)")
    ap.add_argument("--ref-rel-dead-db", type=float, default=40.0)
    ap.add_argument("--in-device", default=None)
    ap.add_argument("--out-device", default=None)
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--primary-ch", type=int, default=0)
    ap.add_argument("--reference-ch", type=int, default=1)
    ap.add_argument("--no-lms", action="store_true")
    ap.add_argument("--bypass", action="store_true")
    ap.add_argument("--gain-floor-db", type=float, default=None)
    ap.add_argument("--limiter-dbfs", type=float, default=-1.0)
    ap.add_argument("--simulate", default=None)
    ap.add_argument("--simulate-ref", default=None)
    ap.add_argument("--paced", action="store_true", help="sleep to real time in simulation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args(argv)
    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if a.simulate:
        run_simulated(a)
    else:
        run_live(a)


if __name__ == "__main__":
    main()
