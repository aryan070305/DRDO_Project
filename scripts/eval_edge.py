"""Evaluate systems on the edge-case set data/testsets/defence_edge_v1 (case-aware metrics).

    python scripts/eval_edge.py --systems noisy,logmmse,dancnet:hq_metric,dancnet:ll [--save_audio dancnet:hq_metric]

Metrics per file: PESQ-WB/NB, STOI, ESTOI, SI-SDR, output SNR (all vs the clean target).  Extra:
  noise_only  -> attenuation_db = 10 log10(sum x^2 / sum y^2) (speech metrics undefined, left empty)
  post_blast  -> recovery_stoi / recovery_pesq_wb on the 2 s window starting at the blast (t = 1.0 s)
Writes reports/results/defence_edge_v1/per_file.csv (file-locked merge) and summary_by_case.csv.
"""
import argparse
import csv
import fcntl
import sys
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from danc.eval.evaluate import make_system  # noqa: E402
from danc.eval.metrics import all_metrics, pesq_wb  # noqa: E402
from pystoi import stoi  # noqa: E402

D = ROOT / "data" / "testsets" / "defence_edge_v1"
OUT = ROOT / "reports" / "results" / "defence_edge_v1"
FS = 16000


def _metrics(args):
    case, x, c, y = args
    if case == "noise_only":
        return {"attenuation_db": float(10 * np.log10(np.sum(x ** 2) / (np.sum(y ** 2) + 1e-12)))}
    m = all_metrics(c, y)
    if case == "post_blast":
        a, b = FS, 3 * FS
        m["recovery_stoi"] = float(stoi(c[a:b], y[a:b], FS))
        m["recovery_pesq_wb"] = pesq_wb(c[a:b], y[a:b])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True)
    ap.add_argument("--save_audio", default="")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(D / "meta.csv")))
    rd = lambda p: sf.read(p, dtype="float32")[0]  # noqa: E731
    noisy = [rd(D / "noisy" / f"{r['id']}.flac") for r in rows]
    clean = [rd(D / "clean" / f"{r['id']}.flac") for r in rows]
    out = []
    args = Namespace(set="defence_edge_v1", ckpt=None, onnx=str(ROOT / "exports" / "dancnet_hqft.onnx"))
    for sysname in a.systems.split(","):
        f = make_system(sysname, args)
        ests = []
        for r, x in zip(rows, noisy):
            make_system.current_id = r["id"]
            ests.append(f(x))
        if sysname in a.save_audio.split(","):
            ad = ROOT / "reports" / "audio" / f"enhanced_defence_edge_v1_{sysname.replace(':', '_')}"
            ad.mkdir(parents=True, exist_ok=True)
            for r, e in zip(rows, ests):
                sf.write(ad / f"{r['id']}.flac", np.clip(e, -1, 1), FS, subtype="PCM_16", format="FLAC")
        with ProcessPoolExecutor(a.workers) as ex:
            mets = list(ex.map(_metrics, [(r["case"], x, c, e) for r, x, c, e in zip(rows, noisy, clean, ests)], chunksize=4))
        for r, mm in zip(rows, mets):
            out.append({"system": sysname, "id": r["id"], "case": r["case"], "snr_db": r["snr_db"], **mm})
        print(f"[edge] {sysname} done", flush=True)
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    pf = OUT / "per_file.csv"
    with open(OUT / ".per_file.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if pf.exists():
            old = pd.read_csv(pf)
            df = pd.concat([old[~old.system.isin(df.system.unique())], df], ignore_index=True)
        df.to_csv(pf, index=False)
        fcntl.flock(lk, fcntl.LOCK_UN)
    cols = [c for c in ["pesq_wb", "pesq_nb", "stoi", "estoi", "sisdr", "snr", "attenuation_db", "recovery_stoi",
                        "recovery_pesq_wb"] if c in df.columns]
    summ = df.groupby(["case", "system"])[cols].mean().round(3)
    summ.to_csv(OUT / "summary_by_case.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
