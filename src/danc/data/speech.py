"""Clean-speech manifests (LibriSpeech, CC BY 4.0) with speaker-disjoint splits.

  train : Mini-LibriSpeech train-clean-5 (28 spk) + LibriSpeech dev-clean minus 6 held-out speakers
  val   : 6 held-out dev-clean speakers (model selection only)
  test  : LibriSpeech test-clean (40 speakers, never seen in training)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
LS = ROOT / "data" / "raw" / "LibriSpeech"
MANIFESTS = ROOT / "data" / "manifests"
N_VAL_SPK = 6


def _scan(subset: str, full_decode: bool = False) -> list[dict]:
    """List utterances; with full_decode=True every file is decoded completely and unreadable or
    truncated files are skipped (reported, never deleted)."""
    out = []
    for p in sorted((LS / subset).rglob("*.flac")):
        try:
            if full_decode:
                x, fs = sf.read(p, dtype="float32")
                dur = len(x) / fs
            else:
                dur = sf.info(p).duration
        except Exception as e:  # noqa: BLE001
            print(f"[speech] skipping unreadable file {p}: {e}")
            continue
        out.append({"path": str(p.relative_to(ROOT)), "spk": p.parts[-3], "dur": round(dur, 3)})
    return out


def build_manifests(force: bool = False) -> dict[str, list[dict]]:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    paths = {s: MANIFESTS / f"speech_{s}.json" for s in ("train", "val", "test")}
    if not force and all(p.exists() for p in paths.values()):
        return {s: json.loads(p.read_text()) for s, p in paths.items()}
    tr5 = _scan("train-clean-5")
    dev = _scan("dev-clean")
    test = _scan("test-clean")
    dev_spk = sorted({u["spk"] for u in dev})
    val_spk = set(np.random.default_rng(1234).choice(dev_spk, N_VAL_SPK, replace=False).tolist())
    man = {
        "train": tr5 + [u for u in dev if u["spk"] not in val_spk],
        "val": [u for u in dev if u["spk"] in val_spk],
        "test": test,
    }
    for s, p in paths.items():
        p.write_text(json.dumps(man[s]))
    return man


def build_train_v2(force: bool = False) -> list[dict]:
    """speech_train_v2.json = speech_train.json + the streamed LibriSpeech train-clean-100 prefix.

    Mini-LibriSpeech train-clean-5 is a subset of train-clean-100, so utterances are de-duplicated
    by ID; speakers are checked to be disjoint from the val and test splits."""
    path = MANIFESTS / "speech_train_v2.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    man = build_manifests()
    seen = {Path(u["path"]).stem for u in man["train"]}
    extra = [u for u in _scan("train-clean-100", full_decode=True) if Path(u["path"]).stem not in seen]
    held_out = {u["spk"] for u in man["val"]} | {u["spk"] for u in man["test"]}
    assert not ({u["spk"] for u in extra} & held_out), "speaker leak into val/test"
    v2 = man["train"] + extra
    path.write_text(json.dumps(v2))
    return v2


def load_manifest(name: str) -> list[dict]:
    """Load a manifest by split name ('train', 'val', 'test') or file name ('speech_train_v2.json')."""
    if name in ("train", "val", "test"):
        return build_manifests()[name]
    if name == "speech_train_v2.json":
        return build_train_v2()
    return json.loads((MANIFESTS / name).read_text())


def load_utt(entry: dict) -> np.ndarray:
    x, fs = sf.read(ROOT / entry["path"], dtype="float32")
    assert fs == 16000
    return x


def summary() -> dict:
    man = build_manifests()
    if (MANIFESTS / "speech_train_v2.json").exists():
        man["train_v2"] = build_train_v2()
    return {s: {"n_utt": len(v), "n_spk": len({u['spk'] for u in v}), "hours": round(sum(u['dur'] for u in v) / 3600, 2)}
            for s, v in man.items()}


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
