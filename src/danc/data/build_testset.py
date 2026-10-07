"""Render the fixed evaluation sets to disk (FLAC 16-bit + metadata CSV).

  data/testsets/defence_v1/   single-mic: 4 categories x SNR {-5,0,5,10,15} dB x 40 utts
                              + stress condition SNR -10 dB (4 x 25)
  data/testsets/dualmic_v1/   dual-mic headset scenes (primary, reference, clean):
                              4 categories x SNR {-5,0,5,10} dB x 15 scenes

Speech: LibriSpeech test-clean (40 speakers never used in training/validation).
Noise : test split of the noise bank (NOISEX last 20 %, ESC-50 fold 5, drone hold-out,
        synthetic generators with a seed stream disjoint from training).
No augmentation (no EQ/clipping/random levels); talker level fixed at -26 dBFS active.

    python -m danc.data.build_testset
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from danc.data.dual_mic import DualMicConfig, simulate_scene
from danc.data.mixer import EVAL_CATS, DefenceMixer, active_power

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "testsets"
FS = 16000
SEG_S = 6.0


def _write(path: Path, x: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x, -1, 1), FS, subtype="PCM_16", format="FLAC")


def build_defence(n_per: int = 40, n_stress: int = 25, seed: int = 20261003):
    d = OUT / "defence_v1"
    if (d / "meta.csv").exists():
        print("[testset] defence_v1 exists - skip")
        return
    mixer = DefenceMixer("test", SEG_S, use_synth=True, augment=False)
    rng = np.random.default_rng(seed)
    utts = [u for u in mixer.utts if u["dur"] >= 3.0]
    rows = []
    conds = [(c, s, n_per) for c in EVAL_CATS for s in (-5, 0, 5, 10, 15)] + [(c, -10, n_stress) for c in EVAL_CATS]
    k = 0
    for cat, snr, n in conds:
        for _ in range(n):
            u = utts[rng.integers(len(utts))]
            ex = mixer.sample(rng, category=cat, snr_db=snr, utt=u, n=int(SEG_S * FS))
            uid = f"{k:04d}_{cat}_{'m' if snr < 0 else 'p'}{abs(snr):02d}"
            _write(d / "noisy" / f"{uid}.flac", ex["noisy"])
            _write(d / "clean" / f"{uid}.flac", ex["clean"])
            m = ex["meta"]
            rows.append({"id": uid, "category": cat, "snr_db": snr, "utt": m["utt"], "spk": m["spk"],
                         "noise_sources": "|".join(m.get("noise_sources", [])),
                         "imp_to_bed_db": m.get("imp_to_bed_db", "")})
            k += 1
        print(f"[testset] defence {cat} {snr:+d} dB done ({k})", flush=True)
    with open(d / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (d / "README.txt").write_text(__doc__)


def build_dualmic(n_per: int = 15, seed: int = 777001, name: str = "dualmic_v1", split: str = "test"):
    """split="test": test speakers + test-split noise.  split="val": held-out validation speakers +
    training-split noise (used ONLY to tune the NLMS<->DNN coupling, never for reported results)."""
    d = OUT / name
    if (d / "meta.csv").exists():
        print(f"[testset] {name} exists - skip")
        return
    mixer = DefenceMixer(split, SEG_S, use_synth=True, augment=False)
    rng = np.random.default_rng(seed)
    utts = [u for u in mixer.utts if u["dur"] >= 3.0]
    cfg = DualMicConfig()
    n = int(SEG_S * FS)
    rows, k = [], 0
    for cat in EVAL_CATS:
        for snr in (-5, 0, 5, 10):
            for _ in range(n_per):
                speech, u = mixer._speech(rng, n=n, utt=utts[rng.integers(len(utts))])
                speech = speech * np.sqrt(10 ** (-26 / 10) / active_power(speech))
                meta = {}
                if cat == "mixed":
                    noises = [mixer._component(c, n + FS, rng, meta, c[:3]) for c in
                              (rng.choice(["stationary", "nonstationary"]), "impulsive")]
                else:
                    noises = [mixer._component(cat, n + FS, rng, meta, "src") for _ in range(rng.integers(1, 3))]
                sc = simulate_scene(speech, noises, float(snr), rng, cfg, FS)
                uid = f"{k:04d}_{cat}_{'m' if snr < 0 else 'p'}{abs(snr):02d}"
                _write(d / "primary" / f"{uid}.flac", sc["primary"])
                _write(d / "reference" / f"{uid}.flac", sc["reference"])
                _write(d / "clean" / f"{uid}.flac", sc["clean"])
                sm = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in sc["meta"].items()
                      if isinstance(v, (int, float, str))}
                rows.append({"id": uid, "category": cat, "snr_db": snr, "utt": u["path"],
                             "noise_sources": "|".join(meta.get("noise_sources", [])), "scene": json.dumps(sm)})
                k += 1
        print(f"[testset] dualmic {cat} done ({k})", flush=True)
    with open(d / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    build_defence()
    build_dualmic()
    build_dualmic(n_per=3, seed=424242, name="dualmic_val", split="val")
