"""Dual-microphone crew-babble test set: the hardest single-mic edge case (defence_edge_v1/crew_babble), as a
headset scene so that the reference microphone can be used.

  data/testsets/dualmic_babble_v1/   60 scenes x 6 s (primary, reference, clean, meta.csv)

Scene: the wearer (LibriSpeech test speaker, -26 dBFS active, close-talk at the boom mic, leakage into the
reference -25..-12 dB as in dualmic_v1) + 2-3 OTHER test speakers as directional sources 0.6-2 m away (crew in a
vehicle or shelter, reaching both mics at similar level) + a diffuse vehicle/engine bed 10 dB below the wearer.
Target-to-interferer ratio (TIR) at the primary mic: 0 / 5 / 10 dB, 20 scenes each.  Mic spacing 8-15 cm.
NOTE: the meta.csv column `snr_db` holds the TIR (crew talkers only); with the bed the measured SNR at the primary is
-0.4 / 3.8 / 7.0 dB (per file: `input_snr_measured` in reports/results/dualmic_babble_v1/per_file.csv).
No training speaker or training noise is used.

    python -m danc.data.build_dual_babble
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from danc.data.dual_mic import DualMicConfig, simulate_scene
from danc.data.mixer import DefenceMixer, active_power

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "testsets" / "dualmic_babble_v1"
FS = 16000
N = 6 * FS


def _write(path: Path, x: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x, -1, 1), FS, subtype="PCM_16", format="FLAC")


def build(n_per: int = 20, seed: int = 20261004):
    if (OUT / "meta.csv").exists():
        print(f"[testset] {OUT.name} exists - skip")
        return
    m = DefenceMixer("test", N / FS, use_synth=True, augment=False)
    rng = np.random.default_rng(seed)
    utts = [u for u in m.utts if u["dur"] >= 3.0]
    rows, k = [], 0
    for tir in (0, 5, 10):
        for _ in range(n_per):
            s, u = m._speech(rng, n=N, utt=utts[rng.integers(len(utts))])
            s = s * np.sqrt(10 ** (-26 / 10) / active_power(s))
            talkers, spk = [], []
            while len(talkers) < int(rng.integers(2, 4)):
                o, uo = m._speech(rng, n=N + FS, utt=utts[rng.integers(len(utts))])
                if uo["spk"] != u["spk"] and uo["spk"] not in spk:
                    talkers.append(o / np.sqrt(active_power(o) + 1e-12))
                    spk.append(uo["spk"])
            d = float(rng.uniform(0.08, 0.15))
            base = replace(DualMicConfig(), mic_spacing_m=(d, d), max_peak=None)
            # 1) crew talkers: all directional, 0.6-2 m, no diffuse part; SNR at the primary = TIR
            c1 = replace(base, n_directional=(len(talkers), len(talkers)), source_distance_m=(0.6, 2.0),
                         diffuse_ratio=(0.0, 0.0))
            sc = simulate_scene(s, talkers, float(tir), rng, c1, FS)
            # 2) diffuse vehicle bed, 10 dB below the wearer at the primary (only its noise part is used)
            meta = {}
            bed = m._component("stationary", N + FS, rng, meta, "bed")
            c2 = replace(base, n_directional=(0, 0), diffuse_ratio=(1.0, 1.0))
            sb = simulate_scene(s, [bed], 10.0, rng, c2, FS)
            prim = sc["clean"] + sc["noise_primary"] + sb["noise_primary"]
            ref = sc["speech_reference"] + sc["noise_reference"] + sb["noise_reference"]
            clean = sc["clean"]
            pk = max(np.abs(prim).max(), np.abs(ref).max())
            if pk > 0.99:
                prim, ref, clean = prim * 0.99 / pk, ref * 0.99 / pk, clean * 0.99 / pk
            uid = f"{k:04d}_crew_babble_tir{tir:02d}"
            _write(OUT / "primary" / f"{uid}.flac", prim)
            _write(OUT / "reference" / f"{uid}.flac", ref)
            _write(OUT / "clean" / f"{uid}.flac", clean)
            sm = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in sc["meta"].items()
                  if isinstance(v, (int, float, str))}
            rows.append({"id": uid, "category": "crew_babble", "snr_db": tir, "utt": u["path"],
                         "n_talkers": len(talkers), "interferer_speakers": "|".join(map(str, spk)),
                         "noise_sources": "|".join(meta.get("noise_sources", [])), "scene": json.dumps(sm)})
            k += 1
        print(f"[testset] dualmic_babble TIR {tir} dB done ({k})", flush=True)
    with open(OUT / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    build()
