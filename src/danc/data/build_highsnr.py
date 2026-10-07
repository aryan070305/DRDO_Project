"""High-SNR extension of the defence test set, to locate where the requested PESQ 3.25 / 3.5 is actually reached.

  data/testsets/defence_highsnr_v1/   4 categories x SNR {20, 25} dB x 20 files (160 files, 6 s)

Built exactly like defence_v1 (danc.data.build_testset.build_defence): LibriSpeech test-clean speakers, test-split
noise, no augmentation, talker at -26 dBFS active level - only the SNRs differ and a separate seed is used.

    python -m danc.data.build_highsnr
"""
from __future__ import annotations

import csv

import numpy as np

from danc.data.build_testset import FS, OUT, SEG_S, _write
from danc.data.mixer import EVAL_CATS, DefenceMixer


def build(n_per: int = 20, snrs=(20, 25), seed: int = 20261005):
    d = OUT / "defence_highsnr_v1"
    if (d / "meta.csv").exists():
        print("[testset] defence_highsnr_v1 exists - skip")
        return
    mixer = DefenceMixer("test", SEG_S, use_synth=True, augment=False)
    rng = np.random.default_rng(seed)
    utts = [u for u in mixer.utts if u["dur"] >= 3.0]
    rows, k = [], 0
    for cat in EVAL_CATS:
        for snr in snrs:
            for _ in range(n_per):
                u = utts[rng.integers(len(utts))]
                ex = mixer.sample(rng, category=cat, snr_db=snr, utt=u, n=int(SEG_S * FS))
                uid = f"{k:04d}_{cat}_p{snr:02d}"
                _write(d / "noisy" / f"{uid}.flac", ex["noisy"])
                _write(d / "clean" / f"{uid}.flac", ex["clean"])
                m = ex["meta"]
                rows.append({"id": uid, "category": cat, "snr_db": snr, "utt": m["utt"], "spk": m["spk"],
                             "noise_sources": "|".join(m.get("noise_sources", [])),
                             "imp_to_bed_db": m.get("imp_to_bed_db", "")})
                k += 1
            print(f"[testset] defence_highsnr {cat} {snr:+d} dB done ({k})", flush=True)
    with open(d / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (d / "README.txt").write_text(__doc__)


if __name__ == "__main__":
    build()
