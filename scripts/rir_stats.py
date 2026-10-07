"""Reverberation statistics of the RIR bank (data/rir_bank/rirs_v2_*.npz) -> reports/results/rir_stats.json.

RT60 is estimated from the Schroeder backward-integrated energy decay with a T20 fit (linear least-squares
fit of the decay between -5 and -25 dB, extrapolated to -60 dB, ISO 3382-style) and, for comparison, the
simpler two-point crossing estimate.  Note: RIRs are truncated at 1.2*RT60_target + 50 ms, which biases
late-decay estimates low.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FS = 16000


def edc_db(h):
    e = np.cumsum(h[::-1].astype(np.float64) ** 2)[::-1]
    return 10 * np.log10(e / e[0] + 1e-15)


def t20_fit(h):
    d = edc_db(h)
    idx = np.where((d <= -5) & (d >= -25))[0]
    if len(idx) < 10:
        return float("nan")
    t = idx / FS
    slope = np.polyfit(t, d[idx], 1)[0]          # dB/s
    return float(-60.0 / slope) if slope < 0 else float("nan")


def t20_two_point(h):
    d = edc_db(h)
    i5, i25 = np.argmax(d < -5), np.argmax(d < -25)
    return float(3 * (i25 - i5) / FS) if i25 > i5 else float("nan")


out = {}
for split in ("train", "test"):
    z = np.load(ROOT / "data" / "rir_bank" / f"rirs_v2_{split}.npz", allow_pickle=True)
    for fam in ("near", "far"):
        hs = list(z[fam])
        fit = np.array([t20_fit(h) for h in hs])
        tp = np.array([t20_two_point(h) for h in hs])
        out[f"{split}/{fam}"] = {"n": len(hs), "rt60_T20fit_median_s": float(np.nanmedian(fit)),
                                 "rt60_T20fit_p10_p90_s": [float(np.nanpercentile(fit, 10)), float(np.nanpercentile(fit, 90))],
                                 "rt60_twopoint_median_s": float(np.nanmedian(tp)),
                                 "length_median_s": float(np.median([len(h) / FS for h in hs]))}
(ROOT / "reports" / "results" / "rir_stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
