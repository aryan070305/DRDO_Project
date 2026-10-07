"""Evaluate systems on the fixed test sets.

    python -m danc.eval.evaluate --set defence_v1 --systems noisy,specsub,wiener,logmmse,dancnet \
        --ckpt checkpoints/main/best.pt
    python -m danc.eval.evaluate --set dualmic_v1 --systems primary,nlms,dnn,hybrid,hybrid_nogate,logmmse \
        --onnx exports/dancnet_hqft.onnx
    python -m danc.eval.evaluate --set vbdemand --systems noisy,logmmse,dancnet --ckpt ...

Outputs (reports/results/<set>/): per_file.csv, summary_by_condition.csv, summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from danc.dsp import istft, stft
from danc.eval.metrics import batch_metrics, snr
from danc.models import baselines

ROOT = Path(__file__).resolve().parents[3]
TARGETS = {"snr": 15.0, "stoi": 0.85, "pesq_wb": 2.5}


def load_set(name: str):
    if name == "vbdemand":
        d = ROOT / "data" / "raw" / "vbdemand_test"
        ids = sorted(p.stem for p in (d / "clean").glob("*.wav"))
        items = [{"id": i, "category": "vbdemand", "snr_db": np.nan,
                  "noisy": d / "noisy" / f"{i}.wav", "clean": d / "clean" / f"{i}.wav"} for i in ids]
        return items
    d = ROOT / "data" / "testsets" / name
    rows = list(csv.DictReader(open(d / "meta.csv")))
    items = []
    for r in rows:
        it = {"id": r["id"], "category": r["category"], "snr_db": float(r["snr_db"]), "clean": d / "clean" / f"{r['id']}.flac"}
        if name.startswith("dualmic"):
            it["noisy"] = d / "primary" / f"{r['id']}.flac"
            it["reference"] = d / "reference" / f"{r['id']}.flac"
        else:
            it["noisy"] = d / "noisy" / f"{r['id']}.flac"
        items.append(it)
    return items


def _read(p):
    x, fs = sf.read(p, dtype="float32")
    assert fs == 16000
    return x


class DancOffline:
    """Full-sequence DANCNet (numerically identical to the streaming engine; see tests)."""

    def __init__(self, ckpt):
        from danc.inference.export_onnx import load_model
        torch.set_num_threads(4)
        self.m = load_model(ckpt)

    @torch.no_grad()
    def __call__(self, x):
        X = torch.from_numpy(stft(x)).to(torch.complex64)
        Xr = torch.view_as_real(X)[None].float()
        Y, _ = self.m.enhance_spec(Xr)
        Yc = (Y[0, ..., 0] + 1j * Y[0, ..., 1]).numpy()
        return istft(Yc, len(x)).astype(np.float32)


def make_system(name, args):
    if name.startswith("file:"):
        # pre-computed outputs (e.g. the external DeepFilterNet3 reference, see scripts/run_deepfilternet3.py)
        d = ROOT / "reports" / "audio" / f"enhanced_{args.set}_{name[5:]}"
        lookup = {}

        def from_file(x, r=None, _d=d):
            y = _read(_d / f"{make_system.current_id}.flac")
            return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
        return from_file
    if name in ("noisy", "primary"):
        return lambda x, r=None: x
    if name in ("specsub", "wiener", "logmmse"):
        return lambda x, r=None: baselines.enhance(x, name)
    if name == "dancnet" or name.startswith("dancnet:"):
        # "dancnet" uses --ckpt; "dancnet:<run>" uses checkpoints/<run>/best.pt
        if name == "dancnet":
            ck = args.ckpt
        else:   # best.pt, or last.pt for runs that never beat their starting point (fine-tunes)
            run = ROOT / "checkpoints" / name.split(":", 1)[1]
            ck = str(run / "best.pt") if (run / "best.pt").exists() else str(run / "last.pt")
        f = DancOffline(ck)
        return lambda x, r=None: f(x)
    # dual-mic systems (streaming engine)
    from danc.inference.engine import EngineConfig, HybridEngine
    from danc.inference.runners import ORTStepRunner
    if name == "nlms":
        eng = HybridEngine(None, EngineConfig(use_lms=True, spp_gate=False, limiter_dbfs=None))
        return lambda x, r=None: eng.process_signal(x, r)
    base, _, tag = name.partition(":")         # e.g. "hybrid:hq" -> exports/dancnet_hq.onnx
    runner = ORTStepRunner(str(ROOT / "exports" / f"dancnet_{tag}.onnx") if tag else args.onnx)
    name = base
    cfgs = {"dnn2": EngineConfig(use_lms=False, limiter_dbfs=None),   # two-microphone network (P and R as inputs)
            "dnn2_refdead": EngineConfig(use_lms=False, limiter_dbfs=None),   # same, reference mic dead (see below)
            "hybrid2": EngineConfig(use_lms=True, spp_gate=True, limiter_dbfs=None, lms_two_mic=True),  # NLMS -> two-mic net
            "hybrid2_refdead": EngineConfig(use_lms=True, spp_gate=True, limiter_dbfs=None, lms_two_mic=True),
            "dnn": EngineConfig(use_lms=False, limiter_dbfs=None),
            "dnn_stream": EngineConfig(use_lms=False, limiter_dbfs=None),
            "hybrid": EngineConfig(use_lms=True, spp_gate=True, limiter_dbfs=None),
            "hybrid_nogate": EngineConfig(use_lms=True, spp_gate=False, limiter_dbfs=None)}
    eng = HybridEngine(runner, cfgs[name])
    if name in ("dnn2_refdead", "hybrid2_refdead"):   # reference-mic failure: the reference input is -80 dBFS white noise, seeded per file
        import zlib

        def refdead(x, r=None):
            rng = np.random.default_rng(zlib.crc32(str(make_system.current_id).encode()))
            return eng.process_signal(x, (rng.standard_normal(len(x)) * 1e-4).astype(np.float32))
        return refdead
    return lambda x, r=None: eng.process_signal(x, r if (name.startswith("hybrid") or name == "dnn2") else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--ckpt", default="checkpoints/main/best.pt")
    ap.add_argument("--onnx", default="exports/dancnet_hqft.onnx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_audio", default="dancnet,hybrid")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    items = load_set(args.set)
    if args.limit:
        items = items[:: max(1, len(items) // args.limit)][: args.limit]
    out = Path(args.out or ROOT / "reports" / "results" / args.set)
    out.mkdir(parents=True, exist_ok=True)
    per_file_path = out / "per_file.csv"
    rows = []
    cleans = [_read(it["clean"]) for it in items]
    noisys = [_read(it["noisy"]) for it in items]
    refs = [_read(it["reference"]) if "reference" in it else None for it in items]
    in_snr = [snr(c, x) for c, x in zip(cleans, noisys)]
    for sysname in args.systems.split(","):
        t0 = time.time()
        f = make_system(sysname, args)
        ests = []
        for it, x, r in zip(items, noisys, refs):
            make_system.current_id = it["id"]
            ests.append(f(x, r))
        t_proc = time.time() - t0
        if sysname in args.save_audio.split(","):
            ad = ROOT / "reports" / "audio" / f"enhanced_{args.set}_{sysname.replace(':', '_')}"
            ad.mkdir(parents=True, exist_ok=True)
            for it, e in zip(items, ests):
                sf.write(ad / f"{it['id']}.flac", np.clip(e, -1, 1), 16000, subtype="PCM_16", format="FLAC")
        mets = batch_metrics(list(zip(cleans, ests)), workers=args.workers)
        for it, m, s_in in zip(items, mets, in_snr):
            rows.append({"system": sysname, "id": it["id"], "category": it["category"], "snr_db": it["snr_db"],
                         "input_snr_measured": s_in, **m, "delta_snr": m["snr"] - s_in})
        print(f"[eval] {args.set}/{sysname}: {len(items)} files, proc {t_proc:.1f}s, metrics {time.time()-t0-t_proc:.1f}s",
              flush=True)
    df = pd.DataFrame(rows)
    # read-modify-write under an exclusive lock so concurrent evaluations of the same set
    # cannot overwrite each other's rows
    import fcntl
    with open(out / ".per_file.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        old = pd.read_csv(per_file_path) if per_file_path.exists() else None
        if old is not None:
            old = old[~old["system"].isin(df["system"].unique())]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(per_file_path, index=False)
        fcntl.flock(lk, fcntl.LOCK_UN)
    summarize(df, out)


def summarize(df: pd.DataFrame, out: Path):
    mets = ["pesq_wb", "pesq_nb", "stoi", "estoi", "sisdr", "snr", "delta_snr", "segsnr"]
    g = df.groupby(["system", "category", "snr_db"], dropna=False)[mets].mean().reset_index()
    g.to_csv(out / "summary_by_condition.csv", index=False)
    summ = {}
    main = df[(df["snr_db"].isna()) | (df["snr_db"] >= -5)]       # -10 dB is reported separately (stress)
    for sysname, d in main.groupby("system"):
        s = {m: float(d[m].mean()) for m in mets}
        s["n"] = int(len(d))
        s["pass_rate"] = {k: float((d[k] > v).mean()) for k, v in TARGETS.items()}
        s["by_category"] = {c: {m: float(dd[m].mean()) for m in ("pesq_wb", "stoi", "snr")}
                            for c, dd in d.groupby("category")}
        s["by_snr"] = {str(k): {m: float(dd[m].mean()) for m in ("pesq_wb", "stoi", "snr")}
                       for k, dd in d.groupby("snr_db")}
        stress = df[(df["system"] == sysname) & (df["snr_db"] == -10)]
        if len(stress):
            s["stress_m10db"] = {m: float(stress[m].mean()) for m in ("pesq_wb", "stoi", "snr")}
        summ[sysname] = s
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    for k, v in summ.items():
        print(f"  {k:14s} PESQ-WB {v['pesq_wb']:.3f}  STOI {v['stoi']:.3f}  SNR {v['snr']:.2f} dB  "
              f"dSNR {v['delta_snr']:.2f}  SI-SDR {v['sisdr']:.2f}")


if __name__ == "__main__":
    main()
