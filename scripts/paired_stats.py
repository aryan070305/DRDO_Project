"""Paired statistics for the key system comparisons (same test files, so differences are paired per file).

For each comparison and metric: mean difference B - A, 95 % confidence interval from a paired bootstrap
(10 000 resamples of the files, fixed seed), the share of files where B is better, and the two-sided Wilcoxon
signed-rank p-value.  Only comparisons whose two systems are both present are computed.

    python scripts/paired_stats.py      -> reports/results/paired_stats.json and a markdown table on stdout
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports" / "results"
METRICS = ["pesq_wb", "pesq_nb", "stoi", "sisdr"]
PAIRS = [   # (set, A, B, question)
    ("defence_v1", "dancnet:hq_metric", "dancnet:v3hq", "v3 HQ vs v2 HQ (round-1 deployed)"),
    ("defence_v1", "dancnet:ll", "dancnet:v3ll", "v3 LL vs v2 LL (round-1 deployed)"),
    ("defence_v1", "dancnet:v3hq", "dancnet:v3hq_cont", "v3 HQ continued vs v3 HQ"),
    ("defence_v1", "dancnet:hq_metric", "dancnet:v3hq_cont", "round-2 HQ (deployed) vs round-1 HQ"),
    ("vbdemand", "dancnet:hq_metric", "dancnet:v3hq_cont", "round-2 HQ (deployed) vs round-1 HQ (VoiceBank+DEMAND)"),
    ("dualmic_v1", "hybrid:hqft", "hybrid:v3hq_cont", "hybrid with round-2 HQ vs hybrid with round-1 HQ"),
    ("dualmic_v1", "hybrid:v3hq_cont", "hybrid2:v3_2mic", "cascade vs hybrid with round-2 HQ (information)"),
    ("defence_v1", "file:dfn3", "dancnet:v3hq", "v3 HQ vs DeepFilterNet3"),
    ("vbdemand", "dancnet:hq_metric", "dancnet:v3hq", "v3 HQ vs v2 HQ (round-1 deployed; VoiceBank+DEMAND)"),
    ("vbdemand", "dancnet:v3hq", "dancnet:v3hq_cont", "v3 HQ continued vs v3 HQ (VoiceBank+DEMAND)"),
    ("dualmic_v1", "hybrid:hqft", "hybrid:v3hq", "hybrid v3 HQ vs hybrid v2 HQ"),
    ("dualmic_v1", "hybrid:v3hq", "dnn2:v3_2mic", "two-mic network vs hybrid v3 HQ"),
    ("dualmic_v1", "hybrid:v3hq", "hybrid:v3hq_cont", "hybrid v3 HQ continued vs hybrid v3 HQ"),
    ("dualmic_v1", "dnn:v3hq", "dnn2_refdead:v3_2mic", "two-mic network with DEAD reference vs single-mic v3 HQ"),
    ("dualmic_v1", "primary", "dnn2_refdead:v3_2mic", "two-mic network with DEAD reference vs unprocessed"),
    ("dualmic_v1", "hybrid:v3hq", "hybrid2:v3_2mic", "cascade NLMS -> two-mic network vs hybrid v3 HQ"),
    ("dualmic_v1", "hybrid:hqft", "hybrid2:v3_2mic", "cascade vs round-1 deployed hybrid HQ"),
    ("dualmic_v1", "primary", "hybrid2_refdead:v3_2mic", "cascade with DEAD reference vs unprocessed"),
    ("dualmic_babble_v1", "hybrid:v3hq", "hybrid2:v3_2mic", "crew babble: cascade vs hybrid v3 HQ"),
    ("dualmic_babble_v1", "hybrid:v3hq", "dnn2:v3_2mic", "crew babble: two-mic network vs hybrid v3 HQ"),
    ("dualmic_babble_v1", "primary", "dnn2:v3_2mic", "crew babble: two-mic network vs unprocessed"),
    ("dualmic_babble_v1", "primary", "hybrid:v3hq", "crew babble: hybrid v3 HQ vs unprocessed"),
] + [("defence_v1", "dancnet:abl_base", f"dancnet:abl_{a}", f"ablation: {d} vs reference arm")
     for a, d in [("ema", "EMA normaliser"), ("crm", "complex mask"), ("la2", "look-ahead 2"), ("la4", "look-ahead 4"),
                  ("nopmsqe", "no PMSQE")]]


def boot_ci(d: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    out, cache = [], {}
    for st, a, b, q in PAIRS:
        if st not in cache:
            p = R / st / "per_file.csv"
            cache[st] = pd.read_csv(p) if p.exists() else None
        df = cache[st]
        if df is None or a not in set(df.system) or b not in set(df.system):
            continue
        d = df[(df.snr_db.isna()) | (df.snr_db >= -5)] if st != "dualmic_babble_v1" else df
        A = d[d.system == a].set_index("id")
        B = d[d.system == b].set_index("id")
        ids = A.index.intersection(B.index)
        if len(ids) < 2:
            continue
        rec = {"set": st, "A": a, "B": b, "question": q, "n": int(len(ids))}
        for m in METRICS:
            x = (B.loc[ids, m] - A.loc[ids, m]).to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            rec[f"n_{m}"] = int(len(x))          # files with a finite value for this metric
            lo, hi = boot_ci(x)
            p = float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
            rec[m] = {"mean_diff": float(x.mean()), "ci95": [lo, hi], "share_B_better": float((x > 0).mean()), "wilcoxon_p": p}
        out.append(rec)
    (R / "paired_stats.json").write_text(json.dumps(out, indent=1))
    print("| Comparison (B vs A) | set | n | Δ PESQ-WB [95 % CI] | Δ PESQ-NB [95 % CI] | Δ STOI [95 % CI] | Δ SI-SDR dB [95 % CI] | B better (PESQ-WB) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out:
        c = lambda m, f: f"{r[m]['mean_diff']:+{f}} [{r[m]['ci95'][0]:+{f}}, {r[m]['ci95'][1]:+{f}}]"  # noqa: E731
        print(f"| {r['question']} | {r['set']} | {r['n']} | {c('pesq_wb', '.3f')} | {c('pesq_nb', '.3f')} | "
              f"{c('stoi', '.4f')} | {c('sisdr', '.2f')} | {r['pesq_wb']['share_B_better']:.0%} |")


if __name__ == "__main__":
    main()
