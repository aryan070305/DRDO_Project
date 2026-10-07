"""Room-impulse-response bank generated with pyroomacoustics (image-source method).

Two families:
  * "near"  - talker 3-12 cm from the boom microphone (close-talk headset); the
              room only adds weak reverberation (high direct-to-reverberant ratio).
  * "far"   - noise source 1-6 m away (vehicle cabin / room / shelter).
RT60 is drawn from 0.1-0.8 s (vehicle cabins are at the short end).
Scientific basis: Allen & Berkley (1979) image-source model, as implemented in
pyroomacoustics (Scheibler, Bezzam, Dokmanic, ICASSP 2018).

The bank is generated once and cached in data/rir_bank/rirs_<split>.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FS = 16000
CACHE = Path(__file__).resolve().parents[3] / "data" / "rir_bank"


def _one_rir(rng: np.random.Generator, kind: str) -> np.ndarray:
    import pyroomacoustics as pra

    L = rng.uniform([2.0, 2.0, 2.2], [9.0, 7.0, 3.5])
    rt60 = rng.uniform(0.1, 0.8)
    try:
        e_abs, max_order = pra.inverse_sabine(rt60, L)
    except ValueError:
        e_abs, max_order = 0.5, 10
    max_order = int(min(max_order, 40))
    room = pra.ShoeBox(L, fs=FS, materials=pra.Material(e_abs), max_order=max_order)
    mic = rng.uniform([0.5, 0.5, 1.0], L - np.array([0.5, 0.5, 0.8]))
    if kind == "near":
        d = rng.uniform(0.03, 0.12)
    else:
        d = rng.uniform(1.0, 6.0)
    for _ in range(50):
        v = rng.standard_normal(3)
        v /= np.linalg.norm(v)
        src = mic + d * v
        if np.all(src > 0.1) and np.all(src < L - 0.1):
            break
    else:
        src = np.clip(mic + 0.05, 0.1, L - 0.1)
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    h = np.asarray(room.rir[0][0], dtype=np.float32)
    h = h[: int(FS * min(1.2, rt60 * 1.2 + 0.05))]
    return h


def build_bank(split: str, n_near: int = 150, n_far: int = 150, seed: int | None = None) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"rirs_v2_{split}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {"near": list(z["near"]), "far": list(z["far"])}
    rng = np.random.default_rng(seed if seed is not None else (11 if split == "train" else 977))
    near = [_one_rir(rng, "near") for _ in range(n_near)]
    far = [_one_rir(rng, "far") for _ in range(n_far)]
    np.savez_compressed(path, near=np.array(near, dtype=object), far=np.array(far, dtype=object))
    return {"near": near, "far": far}


def early_part(h: np.ndarray, ms: float = 50.0) -> np.ndarray:
    """Direct path + early reflections (first `ms` after the direct-path peak) - the
    dereverberation target, following common practice in DNS-style training."""
    k = int(np.argmax(np.abs(h)))
    n = k + int(ms * FS / 1000)
    e = h[:n].copy()
    fade = min(32, len(e))
    e[-fade:] *= np.linspace(1, 0, fade)
    return e
