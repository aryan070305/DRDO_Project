"""Per-band error diagnosis on defence_v1 (the analysis behind the round-2 capacity decision, report section 4).

For every 4th file at input SNR -5 / 0 / 5 / 15 dB (40 files per SNR) and 8 frequency bands it computes, from the
saved enhanced audio in reports/audio/enhanced_defence_v1_<system>/:
  speech_share_%   share of the clean-speech energy in the band
  bandSNR_in/out   10 log10(clean energy / complex error energy) of the noisy input / of the enhanced output
  LSD_dB           log-spectral distance between clean and enhanced, over speech-active frames
Writes reports/results/band_error_analysis.csv and prints one pivot table per quantity and system.

    python scripts/band_error_analysis.py [--systems dancnet_hq_metric,dancnet_v3hq]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from danc.dsp import stft  # noqa: E402

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3150), (3150, 4500), (4500, 5500), (5500, 7000), (7000, 8000)]


def analyse(system: str) -> pd.DataFrame:
    d = ROOT / "data" / "testsets" / "defence_v1"
    meta = pd.read_csv(d / "meta.csv")
    f = np.arange(161) * 50
    out = []
    for snr in (-5, 0, 5, 15):
        ids = meta[meta.snr_db == snr].id.tolist()[::4]
        E_in, E_out, S, lsd, cnt = (np.zeros(len(BANDS)) for _ in range(5))
        for uid in ids:
            c, _ = sf.read(d / "clean" / f"{uid}.flac")
            x, _ = sf.read(d / "noisy" / f"{uid}.flac")
            y, _ = sf.read(ROOT / "reports" / "audio" / f"enhanced_defence_v1_{system}" / f"{uid}.flac")
            C, X, Y = stft(c), stft(x), stft(y)
            act = (np.abs(C) ** 2).sum(1) > 1e-6 * (np.abs(C) ** 2).sum(1).max()
            for i, (lo, hi) in enumerate(BANDS):
                m = (f >= lo) & (f < hi)
                S[i] += np.sum(np.abs(C[:, m]) ** 2)
                E_in[i] += np.sum(np.abs(X[:, m] - C[:, m]) ** 2)
                E_out[i] += np.sum(np.abs(Y[:, m] - C[:, m]) ** 2)
                l1 = 10 * np.log10(np.abs(C[act][:, m]) ** 2 + 1e-8)
                l2 = 10 * np.log10(np.abs(Y[act][:, m]) ** 2 + 1e-8)
                lsd[i] += np.sqrt(np.mean((l1 - l2) ** 2, axis=1)).sum()
                cnt[i] += act.sum()
        for i, (lo, hi) in enumerate(BANDS):
            out.append({"system": system, "snr": snr, "band": f"{lo}-{hi}", "speech_share_%": 100 * S[i] / S.sum(),
                        "bandSNR_in": 10 * np.log10(S[i] / E_in[i]), "bandSNR_out": 10 * np.log10(S[i] / E_out[i]),
                        "LSD_dB": lsd[i] / cnt[i]})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="dancnet_hq_metric,dancnet_v3hq")
    a = ap.parse_args()
    df = pd.concat([analyse(s) for s in a.systems.split(",")], ignore_index=True)
    df.to_csv(ROOT / "reports" / "results" / "band_error_analysis.csv", index=False)
    order = [f"{lo}-{hi}" for lo, hi in BANDS]
    for s, g in df.groupby("system", sort=False):
        for k in ("speech_share_%", "bandSNR_in", "bandSNR_out", "LSD_dB"):
            print(f"\n{s}: {k}")
            print(g.pivot(index="band", columns="snr", values=k).round(1).loc[order])


if __name__ == "__main__":
    main()
