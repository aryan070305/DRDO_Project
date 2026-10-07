"""Per-frame latency benchmark of the deployed streaming path (run on an otherwise idle machine).

For every exported model it measures, per 10 ms hop (160 samples):
  * network step only (ONNX Runtime, 1 intra-op thread)       -> what TensorRT / ORT must beat on the target
  * full engine block: shared STFT + sub-band NLMS + network + iSTFT + limiter (Python orchestration included)
Statistics over N frames: mean, p50, p99, p99.9, max, real-time factor, deadline misses (> 10 ms).
Results -> reports/results/latency_<host>.json and reports/figures/latency_hist.png

    python scripts/benchmark_latency.py --frames 6000
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from danc.dsp import DEFAULT  # noqa: E402
from danc.inference.engine import EngineConfig, HybridEngine  # noqa: E402
from danc.inference.runners import ORTStepRunner  # noqa: E402


def stats(t_ms: np.ndarray) -> dict:
    hop_ms = 1000 * DEFAULT.hop / DEFAULT.fs
    return {"n": int(len(t_ms)), "mean_ms": float(t_ms.mean()), "p50_ms": float(np.median(t_ms)),
            "p99_ms": float(np.percentile(t_ms, 99)), "p99_9_ms": float(np.percentile(t_ms, 99.9)),
            "max_ms": float(t_ms.max()), "rtf": float(t_ms.mean() / hop_ms),
            "deadline_misses_gt_10ms": int((t_ms > hop_ms).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--models", default="hq,ll,pilot")
    ap.add_argument("--tag", default="", help="suffix for the output files (default: overwrite latency_mac_m4.json)")
    ap.add_argument("--fallback", default="v3hq", help="single-mic model kept in hot standby next to a two-mic model "
                    "for the extra '<model>+fallback' row (reference-health monitor); '' = no such row")
    a = ap.parse_args()
    if a.frames <= 400:
        raise SystemExit("--frames must be > 400 (the first 200 hops are warm-up and are not counted)")
    rng = np.random.default_rng(0)
    # realistic input: a dual-mic test scene looped
    import soundfile as sf
    d = ROOT / "data" / "testsets" / "dualmic_v1"
    uid = sorted(p.stem for p in (d / "primary").glob("*.flac"))[100]
    prim, _ = sf.read(d / "primary" / f"{uid}.flac", dtype="float64")
    ref, _ = sf.read(d / "reference" / f"{uid}.flac", dtype="float64")
    hop = DEFAULT.hop
    reps = int(np.ceil(a.frames * hop / len(prim)))
    prim, ref = np.tile(prim, reps)[: a.frames * hop], np.tile(ref, reps)[: a.frames * hop]
    out = {"host": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
           "python": platform.python_version(), "frames": a.frames, "models": {}}
    hists = {}
    for name in a.models.split(","):
        onnx = ROOT / "exports" / (f"dancnet_{name}.onnx" if name != "pilot" else "dancnet_pilot.onnx")
        if not onnx.exists():
            continue
        r = ORTStepRunner(str(onnx), threads=1)
        spec = (rng.standard_normal((1, 1, DEFAULT.n_bins, 2)) * 0.05).astype(np.float32)
        sref = spec if r.n_mics == 2 else None     # two-microphone network: second input
        for _ in range(200):
            r(spec, sref)
        ts = []
        for _ in range(a.frames):
            t0 = time.perf_counter()
            r(spec, sref)
            ts.append(time.perf_counter() - t0)
        net = stats(np.array(ts) * 1000)
        eng = HybridEngine(ORTStepRunner(str(onnx), threads=1), EngineConfig())
        eng.reset()
        for k in range(a.frames):
            eng.process_block(prim[k * hop:(k + 1) * hop], ref[k * hop:(k + 1) * hop])
        t_eng = np.array(list(eng.timing)[200:]) * 1000      # engine.timing is a bounded deque
        full = stats(t_eng)
        out["models"][name] = {"onnx": str(onnx.relative_to(ROOT)), "lookahead": r.lookahead,
                               "algorithmic_latency_ms": DEFAULT.latency_ms + 10.0 * r.lookahead,
                               "n_mics": r.n_mics, "network_step": net, "full_engine_block": full}
        hists[name] = t_eng
        if r.n_mics == 2:   # NLMS -> two-mic network cascade (EngineConfig.lms_two_mic), with and without fallback
            fbp = ROOT / "exports" / f"dancnet_{a.fallback}.onnx"
            variants = [("+cascade", None)] + ([("+cascade+fallback", fbp)] if a.fallback and fbp.exists() else [])
            for suffix, fpath in variants:
                kw = {"fallback_runner": ORTStepRunner(str(fpath), threads=1)} if fpath else {}
                eng = HybridEngine(ORTStepRunner(str(onnx), threads=1), EngineConfig(lms_two_mic=True), **kw)
                eng.reset()
                for k in range(a.frames):
                    eng.process_block(prim[k * hop:(k + 1) * hop], ref[k * hop:(k + 1) * hop])
                t3 = np.array(list(eng.timing)[200:]) * 1000
                f3 = stats(t3)
                out["models"][name + suffix] = {"onnx": str(onnx.relative_to(ROOT)), "lookahead": r.lookahead, "n_mics": 2,
                                                "cascade": True, "algorithmic_latency_ms": DEFAULT.latency_ms + 10.0 * r.lookahead,
                                                "network_step": None, "full_engine_block": f3,
                                                **({"fallback_onnx": str(fpath.relative_to(ROOT))} if fpath else {})}
                hists[name + suffix] = t3
                print(f"{name}{suffix}: engine mean {f3['mean_ms']:.3f} p99.9 {f3['p99_9_ms']:.3f} max {f3['max_ms']:.2f} ms",
                      flush=True)
        fb = ROOT / "exports" / f"dancnet_{a.fallback}.onnx"
        if r.n_mics == 2 and a.fallback and fb.exists():
            # the shippable two-mic configuration if the network alone is not robust to a dead reference:
            # both networks run every hop (hot standby), so this is the compute that has to fit the deadline
            fr = ORTStepRunner(str(fb), threads=1)
            if fr.lookahead == r.lookahead and fr.n_mics == 1:
                eng = HybridEngine(ORTStepRunner(str(onnx), threads=1), EngineConfig(), fallback_runner=fr)
                eng.reset()
                for k in range(a.frames):
                    eng.process_block(prim[k * hop:(k + 1) * hop], ref[k * hop:(k + 1) * hop])
                t2 = np.array(list(eng.timing)[200:]) * 1000
                f2 = stats(t2)
                out["models"][f"{name}+fallback"] = {"onnx": str(onnx.relative_to(ROOT)), "fallback_onnx": str(fb.relative_to(ROOT)),
                                                     "lookahead": r.lookahead, "n_mics": 2,
                                                     "algorithmic_latency_ms": DEFAULT.latency_ms + 10.0 * r.lookahead,
                                                     "network_step": None, "full_engine_block": f2}
                hists[f"{name}+fallback"] = t2
                print(f"{name}+fallback: engine mean {f2['mean_ms']:.3f} p99.9 {f2['p99_9_ms']:.3f} max {f2['max_ms']:.2f} ms",
                      flush=True)
        print(f"{name}: net mean {net['mean_ms']:.3f} p99.9 {net['p99_9_ms']:.3f} ms | engine mean "
              f"{full['mean_ms']:.3f} p99.9 {full['p99_9_ms']:.3f} max {full['max_ms']:.2f} ms, misses "
              f"{full['deadline_misses_gt_10ms']}", flush=True)
    (ROOT / "reports" / "results").mkdir(parents=True, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""
    (ROOT / "reports" / "results" / f"latency_mac_m4{sfx}.json").write_text(json.dumps(out, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    for name, h in hists.items():
        ax.hist(h, bins=np.linspace(0, 6, 121), histtype="step", lw=1.5, label=f"{name} engine block")
    ax.axvline(10, color="k", ls="--", lw=1)
    ax.set_xlabel("processing time per 10 ms hop [ms] (Apple M4, 1 thread, ONNX Runtime CPU)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Per-hop compute time of the full streaming engine (deadline = 10 ms)")
    fig.savefig(ROOT / "reports" / "figures" / f"latency_hist{sfx}.png", dpi=110)


if __name__ == "__main__":
    main()
