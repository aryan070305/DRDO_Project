"""Noise bank: every noise source, its category and its train/test split.

Categories (used for stratified training and evaluation):
  stationary     - quasi-stationary broadband/engine noise (jet cockpits, tanks, ship engine room, vehicles)
  nonstationary  - rapidly changing noise (helicopter, drones, sirens, wind, babble, factory, chainsaw)
  impulsive      - gunfire, artillery, explosions, fireworks, impacts
(a fourth evaluation category, "mixed" battlefield scenes, is composed in danc.data.mixer)

Split policy (no clip is ever shared between train and test):
  NOISEX-92  - one long recording per noise type -> first 75 % train, last 20 % test
               (5 % guard gap).  NB: test noise is *unseen audio* of a *seen recording
               condition*; this is the standard practice for NOISEX but is weaker than
               fully unseen noise types - stated in the report.
  ESC-50     - official folds 1-4 train, fold 5 test.
  Drone      - deterministic 80/20 split on a hash of the file name.
  Synthetic  - generated on the fly from disjoint random-seed streams.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

FS = 16000
ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"

NOISEX_CATEGORY = {
    "white": "stationary", "pink": "stationary", "f16": "stationary", "buccaneer1": "stationary",
    "buccaneer2": "stationary", "destroyerengine": "stationary", "leopard": "stationary", "m109": "stationary",
    "volvo": "stationary", "hfchannel": "stationary",
    "babble": "nonstationary", "factory1": "nonstationary", "factory2": "nonstationary",
    "destroyerops": "nonstationary",
    "machinegun": "impulsive",
}

ESC_CATEGORY = {
    "engine": "stationary", "airplane": "stationary", "train": "stationary", "rain": "stationary",
    "sea_waves": "stationary",
    "helicopter": "nonstationary", "siren": "nonstationary", "wind": "nonstationary", "chainsaw": "nonstationary",
    "hand_saw": "nonstationary", "car_horn": "nonstationary", "thunderstorm": "nonstationary",
    "crackling_fire": "nonstationary",
    "fireworks": "impulsive", "glass_breaking": "impulsive", "door_wood_knock": "impulsive",
}

# synthetic kinds -> category (must match danc.data.synth_noise.KINDS)
SYNTH_CATEGORY = {
    "gunshot_single": "impulsive", "gunfire_burst": "impulsive", "gunfire_sporadic": "impulsive",
    "artillery": "impulsive", "explosion": "impulsive",
    "helicopter": "nonstationary", "drone_multirotor": "nonstationary", "siren_wail": "nonstationary",
    "siren_yelp": "nonstationary", "siren_hilo": "nonstationary", "wind": "nonstationary",
    "jet_flyby": "nonstationary",
    "tracked_vehicle": "stationary",
}

CATEGORIES = ("stationary", "nonstationary", "impulsive")
# Recorded clips are kept in RAM as float16 (11-bit mantissa, ~-66 dB relative quantisation - negligible
# for noise) to halve the memory of every DataLoader worker; rendering is done in float32.
STORE_DTYPE = np.float16


def _trim_silence(x: np.ndarray, thr_db: float = -50.0) -> np.ndarray:
    if len(x) == 0:
        return x
    env = np.abs(x)
    thr = env.max() * 10 ** (thr_db / 20)
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return x[:0]
    return x[idx[0]: idx[-1] + 1]


def _h(name: str) -> float:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


@dataclass
class NoiseSource:
    name: str            # e.g. "noisex/f16", "esc50/helicopter", "drone/bebop", "synth/artillery"
    family: str          # noisex | esc50 | drone | synth
    category: str
    clips: list = field(default_factory=list)   # list of float32 arrays (recorded sources)
    synth_kind: str | None = None

    @property
    def total_s(self) -> float:
        return sum(len(c) for c in self.clips) / FS


class NoiseBank:
    def __init__(self, split: str, root: Path = ROOT, use_synth: bool = True, use_drone: bool = True,
                 verbose: bool = False):
        assert split in ("train", "test")
        self.split = split
        self.sources: list[NoiseSource] = []
        self._load_noisex(root)
        self._load_esc(root)
        if use_drone:
            self._load_drone(root)
        if use_synth:
            try:
                from danc.data import synth_noise  # noqa: F401
                for kind, cat in SYNTH_CATEGORY.items():
                    if kind in synth_noise.KINDS:
                        self.sources.append(NoiseSource(f"synth/{kind}", "synth", cat, synth_kind=kind))
            except ImportError:
                if verbose:
                    print("[noise_bank] synth_noise not available - synthetic sources disabled")
        self.by_cat = {c: [s for s in self.sources if s.category == c] for c in CATEGORIES}
        if verbose:
            for c in CATEGORIES:
                print(f"[noise_bank:{split}] {c}: " + ", ".join(
                    f"{s.name}({s.total_s:.0f}s)" if s.clips else s.name for s in self.by_cat[c]))

    # ------------------------------------------------------------------ loaders
    def _load_noisex(self, root: Path):
        d = root / "noisex92" / "wav16k"
        for name, cat in NOISEX_CATEGORY.items():
            p = d / f"{name}.wav"
            if not p.exists():
                continue
            x, _ = sf.read(p, dtype="float32")
            n = len(x)
            seg = x[: int(0.75 * n)] if self.split == "train" else x[int(0.80 * n):]
            self.sources.append(NoiseSource(f"noisex/{name}", "noisex", cat, clips=[seg.astype(STORE_DTYPE)]))

    def _load_esc(self, root: Path):
        d = root / "esc50"
        meta = d / "esc50.csv"
        if not meta.exists():
            return
        clips: dict[str, list] = {}
        for r in csv.DictReader(open(meta)):
            cat = ESC_CATEGORY.get(r["category"])
            if cat is None:
                continue
            fold = int(r["fold"])
            if (self.split == "train") != (fold <= 4):
                continue
            p = d / "audio16k" / r["filename"]
            if not p.exists():
                continue
            x, _ = sf.read(p, dtype="float32")
            x = _trim_silence(x)
            if len(x) > FS // 10:
                clips.setdefault(r["category"], []).append(x.astype(STORE_DTYPE))
        for c, xs in clips.items():
            self.sources.append(NoiseSource(f"esc50/{c}", "esc50", ESC_CATEGORY[c], clips=xs))

    def _load_drone(self, root: Path):
        d = root / "drone" / "DroneAudioDataset-master"
        groups = {"bebop": d / "Multiclass_Drone_Audio" / "bebop_1",
                  "membo": d / "Multiclass_Drone_Audio" / "membo_1",
                  "mixed": d / "Binary_Drone_Audio" / "yes_drone"}
        for g, gd in groups.items():
            if not gd.exists():
                continue
            xs = []
            for p in sorted(gd.glob("*.wav")):
                is_train = _h(p.name) < 0.8
                if is_train != (self.split == "train"):
                    continue
                x, fs = sf.read(p, dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if fs == FS and len(x) > FS // 4 and np.abs(x).max() > 1e-4:
                    xs.append(x.astype(STORE_DTYPE))
            if xs:
                self.sources.append(NoiseSource(f"drone/{g}", "drone", "nonstationary", clips=xs))

    # ----------------------------------------------------------------- sampling
    def sources_in(self, category: str) -> list[NoiseSource]:
        return self.by_cat[category]

    def pick_source(self, category: str, rng: np.random.Generator) -> NoiseSource:
        """Family-balanced choice so that e.g. 300 drone clips don't swamp 10 NOISEX files."""
        srcs = self.by_cat[category]
        fams = sorted({s.family for s in srcs})
        fam = fams[rng.integers(len(fams))]
        cand = [s for s in srcs if s.family == fam]
        return cand[rng.integers(len(cand))]

    def render(self, src: NoiseSource, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return exactly n samples of noise from `src` (random offset / concatenation)."""
        if src.synth_kind is not None:
            from danc.data import synth_noise
            y = synth_noise.generate(src.synth_kind, n / FS, fs=FS, rng=rng)
            y = np.asarray(y, dtype=np.float32)[:n]
            if len(y) < n:
                y = np.pad(y, (0, n - len(y)))
            return y
        out = np.zeros(n, dtype=np.float32)
        pos = 0
        xf = int(0.01 * FS)  # 10 ms cross-fade between concatenated clips
        ramp = np.linspace(0, 1, xf, dtype=np.float32)
        first = True
        while pos < n:
            clip = src.clips[rng.integers(len(src.clips))]
            if len(clip) > n - pos + xf:
                off = rng.integers(0, len(clip) - (n - pos + xf) + 1)
                clip = clip[off: off + (n - pos) + xf]
            clip = clip.astype(np.float32)          # (copy) clips are stored as float16
            # per-clip gain normalisation so concatenated clips have similar level
            clip /= (np.sqrt(np.mean(clip ** 2)) + 1e-8)
            if not first and len(clip) > xf and pos >= xf:
                clip[:xf] *= ramp
                out[pos - xf: pos] *= ramp[::-1]
                pos -= xf
            m = min(len(clip), n - pos)
            out[pos: pos + m] += clip[:m]
            pos += m
            first = False
        return out
