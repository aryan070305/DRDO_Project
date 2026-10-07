"""Dynamic dual-microphone (primary + reference) training data for the two-microphone DANCNet.

Every example is a new headset scene rendered with danc.data.dual_mic.simulate_scene:
  speech  : LibriSpeech train manifest (speech_train_v2), level U(-40,-15) dBFS, p=.25 near-field reverb
            (target = direct + 50 ms early part, scaled exactly like the primary-mic speech)
  noise   : same categories / sources / split as the single-mic mixer (danc.data.mixer); 1-3 components,
            assigned to directional sources or the diffuse field by simulate_scene
  scene   : mic spacing 8-15 cm, talker leakage into the reference -25..-12 dB, 1-3 directional sources,
            diffuse ratio 0-0.5, self-noise (DualMicConfig defaults)
  SNR     : U(-10, 20) dB at the primary (U(-10, 15) for impulsive / mixed)
  augment : p=.10 clipping of BOTH mics (same drive: a blast saturates the whole headset),
            p=.20 random EQ on primary (+target) and an independent random EQ on the reference,
            p=.2 talker timbre / bandwidth augmentation, p=.03 clean-only, p=.02 noise-only
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from danc.data.dual_mic import DualMicConfig, simulate_scene
from danc.data.mixer import (TRAIN_CAT_P, DefenceMixer, active_power, apply_filters, apply_timbre, clip, power,
                             random_eq, random_timbre)
from danc.data.rir import early_part

FS = 16000


class DualDefenceMixer:
    def __init__(self, split="train", segment_s=3.0, speech_manifest=None, p_timbre=0.2, p_clean_only=0.03,
                 p_noise_only=0.02):
        self.m = DefenceMixer(split, segment_s, use_synth=True, speech_manifest=speech_manifest)
        self.n = int(round(segment_s * FS))
        self.cfg = DualMicConfig()
        self.p_timbre, self.p_clean_only, self.p_noise_only = p_timbre, p_clean_only, p_noise_only

    def _noises(self, category, rng, meta):
        n = self.n + FS
        m = self.m
        if category in ("stationary", "nonstationary"):
            return [m._component(category, n, rng, meta, "src") for _ in range(rng.integers(1, 3))]
        if category == "impulsive":
            comps = [m._component("impulsive", n, rng, meta, "imp")]
            if rng.random() < 0.6:
                bed = m._component(str(rng.choice(["stationary", "nonstationary"])), n, rng, meta, "bed")
                bed *= np.sqrt(power(comps[0]) / power(bed) * 10 ** (rng.uniform(-25, -10) / 10))
                comps.append(bed)
            return comps
        bed = m._component(str(rng.choice(["stationary", "nonstationary"])), n, rng, meta, "bed")
        imp = m._component("impulsive", n, rng, meta, "imp")
        imp *= np.sqrt(power(bed) / power(imp) * 10 ** (rng.uniform(-5, 15) / 10))
        comps = [bed, imp]
        if rng.random() < 0.4:
            ex = m._component("nonstationary", n, rng, meta, "extra")
            ex *= np.sqrt(power(bed) / power(ex) * 10 ** (rng.uniform(-10, 0) / 10))
            comps.append(ex)
        return comps

    def sample(self, rng: np.random.Generator, category=None, snr_db=None) -> dict:
        meta = {}
        speech, u = self.m._speech(rng, n=self.n)
        if category is None:
            cats, ps = zip(*TRAIN_CAT_P.items())
            category = str(rng.choice(cats, p=ps))
        if snr_db is None:
            snr_db = rng.uniform(-10, 20) if category in ("stationary", "nonstationary") else rng.uniform(-10, 15)
        if self.p_timbre > 0 and rng.random() < self.p_timbre:
            speech = apply_timbre(speech, random_timbre(rng))
        speech = speech * np.sqrt(10 ** (rng.uniform(-40, -15) / 10) / active_power(speech))
        target = speech
        if rng.random() < 0.25:
            h = self.m.rirs["near"][rng.integers(len(self.m.rirs["near"]))]
            h = h / (np.abs(h).max() + 1e-9)
            rev = fftconvolve(speech, h)[: self.n]
            g = np.sqrt(active_power(speech) / active_power(rev))
            speech, target = rev * g, fftconvolve(target, early_part(h))[: self.n] * g
        special = rng.random()
        sc = None
        for _attempt in range(5):   # sparse impulsive noise can yield an all-silent window: redraw it
            noises = self._noises(category, rng, meta)
            try:
                sc = simulate_scene(speech, noises, float(snr_db), rng, self.cfg, FS)
                break
            except ValueError:
                continue
        if sc is None:              # last resort: -80 dB floor so the scene is always defined
            noises = [z + rng.standard_normal(len(z)) * 1e-4 for z in self._noises(category, rng, meta)]
            sc = simulate_scene(speech, noises, float(snr_db), rng, self.cfg, FS)
        og = float(sc["meta"].get("output_gain", 1.0))
        prim, ref, tgt = sc["primary"], sc["reference"], target * og
        if special < self.p_noise_only:                    # talker silent: primary/ref contain noise only
            prim, ref, tgt = sc["noise_primary"], sc["noise_reference"], np.zeros_like(tgt)
        elif special < self.p_noise_only + self.p_clean_only:
            prim, tgt = sc["clean"] + rng.standard_normal(self.n) * 1e-4, tgt
            ref = sc["speech_reference"] + rng.standard_normal(self.n) * 1e-4
        if rng.random() < 0.20:
            f1, f2 = random_eq(rng), random_eq(rng)
            prim, tgt, ref = apply_filters(prim, f1), apply_filters(tgt, f1), apply_filters(ref, f2)
        if rng.random() < 0.10:
            drive, soft = rng.uniform(1.2, 4.0), bool(rng.random() < 0.5)
            pk = max(np.abs(prim).max(), np.abs(ref).max()) + 1e-9
            g = drive / pk
            prim = (np.tanh(g * prim) if soft else np.clip(g * prim, -1, 1)) / g
            ref = (np.tanh(g * ref) if soft else np.clip(g * ref, -1, 1)) / g
        pk = max(np.abs(prim).max(), np.abs(ref).max(), np.abs(tgt).max())
        if pk > 0.99:
            prim, ref, tgt = prim * 0.99 / pk, ref * 0.99 / pk, tgt * 0.99 / pk
        meta.update({"category": category, "snr_db": float(snr_db)})
        return {"primary": prim.astype(np.float32), "reference": ref.astype(np.float32),
                "clean": tgt.astype(np.float32), "meta": meta}


try:
    import torch
    from torch.utils.data import Dataset

    from danc.data.mixer import EVAL_CATS

    class DualMixDataset(Dataset):
        def __init__(self, segment_s=3.0, size=20000, seed=0, speech_manifest=None, mix_opts=None):
            self.segment_s, self.size, self.seed = segment_s, size, seed
            self.speech_manifest, self.mix_opts = speech_manifest, mix_opts or {}
            self._m = None

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            if self._m is None:
                self._m = DualDefenceMixer("train", self.segment_s, self.speech_manifest, **self.mix_opts)
            ex = self._m.sample(np.random.default_rng([self.seed, idx]))
            return (torch.from_numpy(ex["primary"]), torch.from_numpy(ex["reference"]),
                    torch.from_numpy(ex["clean"]), torch.tensor(EVAL_CATS.index(ex["meta"]["category"])))
except ImportError:  # pragma: no cover
    pass
