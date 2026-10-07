"""Dynamic noisy/clean pair generator ("dynamic mixing") and torch Dataset.

Each call renders a brand-new example, so the model never sees the same
mixture twice; the fixed evaluation sets are rendered once by
danc.data.build_testset with seeds disjoint from training.

Per-example recipe (train split):
  1. speech  : random LibriSpeech utterance, random 4 s crop, active level
               U(-40, -15) dBFS (P.56-style frame-based active level).
  2. reverb  : p=0.25 the talker is convolved with a near-field RIR (close-talk
               boom mic in a room); target = direct path + 50 ms early part.
  3. noise   : category ~ {stationary .25, nonstationary .35, impulsive .20, mixed .20}
               mixed   = "battlefield" scene: stationary/rotor bed + impulsive events
               impulsive alone gets a weak bed (p=.6) because real fire is never in silence.
               p=0.35 the noise goes through a far-field RIR.
  4. SNR     : U(-10, 20) dB (stationary/non-stationary), U(-10, 15) dB (impulsive/mixed),
               defined as active-speech power / whole-segment noise power at the mic.
  5. clipping: p=0.10 ADC / pre-amp saturation (hard or tanh soft clip, drive 1.2-4x) -
               what a gunshot does to a real headset microphone.
  6. EQ      : p=0.20 random microphone colouration (high-pass + peaking EQ), applied to
               both noisy input and target (the model must not "undo" the mic).
  7. special : p=0.02 noise-only (target = silence), p=0.02 clean-only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, lfilter

from danc.data import speech as speech_mod
from danc.data.noise_bank import NoiseBank
from danc.data.rir import build_bank, early_part

FS = 16000
TRAIN_CAT_P = {"stationary": 0.25, "nonstationary": 0.35, "impulsive": 0.20, "mixed": 0.20}
EVAL_CATS = ("stationary", "nonstationary", "impulsive", "mixed")


# -----------------------------------------------------------------------------
# levels
# -----------------------------------------------------------------------------

def active_power(x: np.ndarray, frame: int = 320, rel_db: float = -40.0) -> float:
    """Mean power over 'active' 20 ms frames (power within rel_db of the loudest
    frame).  An approximation of ITU-T P.56 active speech level - NOT a compliant
    implementation."""
    n = len(x) // frame
    if n == 0:
        return float(np.mean(x ** 2) + 1e-12)
    p = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)
    thr = p.max() * 10 ** (rel_db / 10)
    act = p[p > max(thr, 1e-12)]
    return float(act.mean()) if len(act) else float(np.mean(x ** 2) + 1e-12)


def power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def scale_to_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = active_power(speech)
    pn = power(noise)
    return noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


# -----------------------------------------------------------------------------
# augmentations
# -----------------------------------------------------------------------------

def _biquad_peak(f0, gain_db, q, fs=FS):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    return b / a[0], a / a[0]


def _biquad_hp(fc, q=0.707, fs=FS):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return b / a[0], a / a[0]


def random_eq(rng: np.random.Generator):
    filters = [_biquad_hp(rng.uniform(50, 250))]
    for _ in range(rng.integers(1, 3)):
        filters.append(_biquad_peak(rng.uniform(200, 5000), rng.uniform(-6, 6), rng.uniform(0.5, 2.0)))
    return filters


def apply_filters(x: np.ndarray, filters) -> np.ndarray:
    for b, a in filters:
        x = lfilter(b, a, x)
    return x


def random_timbre(rng: np.random.Generator):
    """Talker-timbre / channel-bandwidth augmentation applied to the CLEAN speech (and hence the target):
    either a low-pass (band-limited channel or dark voice, 3.4-7.5 kHz) or a high shelf (-12..+6 dB).
    Teaches the network that darker or band-limited speech is still speech (found necessary after the
    VoiceBank cross-corpus test, report section 8.3)."""
    from scipy.signal import butter
    if rng.random() < 0.5:
        sos = butter(int(rng.integers(4, 9)), rng.uniform(3400, 7500), "low", fs=FS, output="sos")
        return ("sos", sos)
    f0, g = rng.uniform(1500, 4000), rng.uniform(-12, 6)
    A = 10 ** (g / 40)
    w0 = 2 * np.pi * f0 / FS
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    c = np.cos(w0)
    b = np.array([A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * c),
                  A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * alpha)])
    a = np.array([(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * c),
                  (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * alpha])
    return ("ba", (b / a[0], a / a[0]))


def apply_timbre(x: np.ndarray, t) -> np.ndarray:
    from scipy.signal import sosfilt
    kind, f = t
    return sosfilt(f, x) if kind == "sos" else lfilter(f[0], f[1], x)


def clip(x: np.ndarray, drive: float, soft: bool) -> np.ndarray:
    """Saturate x after amplifying its peak to `drive` x full scale; returns at the original gain."""
    pk = np.max(np.abs(x)) + 1e-9
    g = drive / pk
    y = np.tanh(g * x) if soft else np.clip(g * x, -1.0, 1.0)
    return y / g


# -----------------------------------------------------------------------------
# mixer
# -----------------------------------------------------------------------------

class DefenceMixer:
    def __init__(self, split: str = "train", segment_s: float = 4.0, use_synth: bool = True,
                 augment: bool | None = None, speech_manifest: str | None = None,
                 p_clean_only: float = 0.02, p_noise_only: float = 0.02, p_timbre: float = 0.0):
        assert split in ("train", "val", "test")
        self.split = split
        self.n = int(round(segment_s * FS)) if segment_s else None
        noise_split = "train" if split in ("train", "val") else "test"
        self.bank = NoiseBank(noise_split, use_synth=use_synth)
        self.rirs = build_bank("train" if split in ("train", "val") else "test")
        # speech_manifest (optional, train only) selects e.g. "speech_train_v2.json"
        self.utts = speech_mod.load_manifest(speech_manifest or split)
        self.augment = (split == "train") if augment is None else augment
        self.p_clean_only, self.p_noise_only, self.p_timbre = p_clean_only, p_noise_only, p_timbre

    # ---------------------------------------------------------------- pieces
    def _speech(self, rng, n=None, utt=None):
        n = n or self.n
        u = utt if utt is not None else self.utts[rng.integers(len(self.utts))]
        x = speech_mod.load_utt(u)
        if n is None:
            return x, u
        if len(x) >= n:
            off = rng.integers(0, len(x) - n + 1)
            x = x[off: off + n]
        else:
            off = rng.integers(0, n - len(x) + 1)
            x = np.pad(x, (off, n - len(x) - off))
        return x, u

    def _component(self, category, n, rng, meta, tag):
        src = self.bank.pick_source(category, rng)
        y = self.bank.render(src, n, rng)
        meta.setdefault("noise_sources", []).append(f"{tag}:{src.name}")
        return y

    def _noise(self, category, n, rng, meta):
        if category in ("stationary", "nonstationary"):
            return self._component(category, n, rng, meta, "main")
        if category == "impulsive":
            imp = self._component("impulsive", n, rng, meta, "imp")
            if rng.random() < 0.6:
                bed = self._component(rng.choice(["stationary", "nonstationary"]), n, rng, meta, "bed")
                bed *= np.sqrt(power(imp) / power(bed) * 10 ** (rng.uniform(-25, -10) / 10))
                imp = imp + bed
            return imp
        if category == "mixed":
            bed = self._component(rng.choice(["stationary", "nonstationary"], p=[0.5, 0.5]), n, rng, meta, "bed")
            imp = self._component("impulsive", n, rng, meta, "imp")
            r_db = rng.uniform(-5, 15)            # impulsive energy relative to bed energy
            imp *= np.sqrt(power(bed) / power(imp) * 10 ** (r_db / 10))
            out = bed + imp
            if rng.random() < 0.4:                 # third layer: rotor / drone / siren
                third = self._component("nonstationary", n, rng, meta, "extra")
                third *= np.sqrt(power(bed) / power(third) * 10 ** (rng.uniform(-10, 0) / 10))
                out = out + third
            meta["imp_to_bed_db"] = round(float(r_db), 2)
            return out
        raise ValueError(category)

    # ----------------------------------------------------------------- sample
    def sample(self, rng: np.random.Generator, category: str | None = None, snr_db: float | None = None,
               utt: dict | None = None, n: int | None = None) -> dict:
        aug = self.augment
        meta: dict = {}
        clean, u = self._speech(rng, n=n, utt=utt)
        n = len(clean)
        meta["utt"] = u["path"]
        meta["spk"] = u["spk"]
        if category is None:
            cats, ps = zip(*TRAIN_CAT_P.items())
            category = str(rng.choice(cats, p=ps))
        meta["category"] = category
        if snr_db is None:
            snr_db = rng.uniform(-10, 20) if category in ("stationary", "nonstationary") else rng.uniform(-10, 15)
        meta["snr_db"] = round(float(snr_db), 2)

        # talker level
        level_db = rng.uniform(-40, -15) if aug else -26.0
        if aug and self.p_timbre > 0 and rng.random() < self.p_timbre:
            clean = apply_timbre(clean, random_timbre(rng))
            meta["timbre"] = True
        clean = clean * np.sqrt(10 ** (level_db / 10) / active_power(clean))
        target = clean
        # near-field reverberation
        if aug and rng.random() < 0.25:
            h = self.rirs["near"][rng.integers(len(self.rirs["near"]))]
            h = h / (np.abs(h).max() + 1e-9)
            rev = fftconvolve(clean, h)[:n]
            target = fftconvolve(clean, early_part(h))[:n]
            g = np.sqrt(active_power(clean) / active_power(rev))   # keep talker level
            clean, target = rev * g, target * g
            meta["speech_rir"] = True

        special = rng.random() if aug else 1.0
        pn, pc = self.p_noise_only, self.p_clean_only
        if special < pn:                                       # noise only
            target = np.zeros_like(target)
            clean_in = np.zeros_like(clean)
            meta["special"] = "noise_only"
        else:
            clean_in = clean
        noise = self._noise(category, n, rng, meta)
        if aug and rng.random() < 0.35:
            h = self.rirs["far"][rng.integers(len(self.rirs["far"]))]
            noise = fftconvolve(noise, h / (np.abs(h).max() + 1e-9))[:n]
            meta["noise_rir"] = True
        noise = scale_to_snr(clean, noise, snr_db)
        if pn <= special < pn + pc:                            # clean only (+ mic self-noise)
            noise = rng.standard_normal(n) * np.sqrt(10 ** (rng.uniform(-80, -65) / 10))
            meta["special"] = "clean_only"
        noisy = clean_in + noise

        if aug and rng.random() < 0.20:
            filt = random_eq(rng)
            noisy, target = apply_filters(noisy, filt), apply_filters(target, filt)
            meta["eq"] = True
        if aug and rng.random() < 0.10:
            drive = rng.uniform(1.2, 4.0)
            soft = bool(rng.random() < 0.5)
            noisy = clip(noisy, drive, soft)
            meta["clip_drive"] = round(float(drive), 2)
        pk = max(np.abs(noisy).max(), np.abs(target).max())
        if pk > 0.99:
            noisy, target = noisy * 0.99 / pk, target * 0.99 / pk
        return {"noisy": noisy.astype(np.float32), "clean": target.astype(np.float32), "meta": meta}


# -----------------------------------------------------------------------------
# torch dataset
# -----------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset

    class MixDataset(Dataset):
        """Infinite-style dynamic-mixing dataset; `epoch` reseeds every example."""

        def __init__(self, split="train", segment_s=4.0, size=20000, seed=0, use_synth=True,
                     fixed_category=None, fixed_snr=None, speech_manifest=None, mix_opts=None):
            self.split, self.segment_s, self.size, self.seed = split, segment_s, size, seed
            self.use_synth = use_synth
            self.speech_manifest = speech_manifest
            self.mix_opts = mix_opts or {}
            self.fixed_category, self.fixed_snr = fixed_category, fixed_snr
            self.epoch = 0
            self._mixer = None

        @property
        def mixer(self):
            if self._mixer is None:          # lazily, inside each worker
                self._mixer = DefenceMixer(self.split, self.segment_s, use_synth=self.use_synth,
                                           speech_manifest=self.speech_manifest, **self.mix_opts)
            return self._mixer

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            rng = np.random.default_rng([self.seed, self.epoch, idx])
            ex = self.mixer.sample(rng, category=self.fixed_category, snr_db=self.fixed_snr)
            cat = ex["meta"]["category"]
            return (torch.from_numpy(ex["noisy"]), torch.from_numpy(ex["clean"]),
                    torch.tensor(EVAL_CATS.index(cat)))
except ImportError:  # pragma: no cover
    pass
