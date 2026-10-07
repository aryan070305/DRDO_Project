"""Reproducible downloader for the public corpora used by D-ANC.

Every source, its license and its SHA-256 are written to
data/manifests/provenance.json so a reviewer can re-verify the exact bytes.

Design notes
------------
* Tar archives are streamed straight into the extractor (no archive kept on
  disk) - the development machine had < 9 GB free.
* ESC-50 is fetched per file from the GitHub repository and only the
  defence-relevant classes are kept; clips are resampled 44.1 kHz -> 16 kHz.
* NOISEX-92 .mat files are kept verbatim (provenance) and also converted to
  16 kHz wav (original fs = 19.98 kHz).
* Nothing is ever deleted. Re-running skips sources already present.

Usage:  python -m danc.data.download --root data/raw [--only librispeech,noisex,...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import soxr

FS = 16000

LIBRISPEECH = {
    # name: (url, license)
    "train-clean-5": ("https://www.openslr.org/resources/31/train-clean-5.tar.gz", "CC BY 4.0 (Mini LibriSpeech, OpenSLR SLR31)"),
    "dev-clean": ("https://www.openslr.org/resources/12/dev-clean.tar.gz", "CC BY 4.0 (LibriSpeech, OpenSLR SLR12)"),
    "test-clean": ("https://www.openslr.org/resources/12/test-clean.tar.gz", "CC BY 4.0 (LibriSpeech, OpenSLR SLR12)"),
}

NOISEX_BASE = "http://spib.linse.ufsc.br/data/noise/"
NOISEX_FILES = [
    "white", "pink", "babble", "factory1", "factory2", "buccaneer1", "buccaneer2", "f16",
    "destroyerengine", "destroyerops", "leopard", "m109", "machinegun", "volvo", "hfchannel",
]
NOISEX_FS = 19980
NOISEX_LICENSE = ("NOISE-ROM-0, copyright TNO Soesterberg 1990, distributed under NATO AC243/(Panel 3)/RSG-10 "
                  "and ESPRIT 2589-SAM; mirror: SPIB (UFSC). Redistribution terms NOT stated - UNCLEAR, flag before sharing.")

ESC50_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/"
ESC50_CLASSES = [
    "helicopter", "siren", "engine", "airplane", "wind", "fireworks", "chainsaw", "train",
    "thunderstorm", "rain", "crackling_fire", "hand_saw", "car_horn", "glass_breaking",
    "door_wood_knock", "sea_waves",
]
ESC50_LICENSE = "ESC-50: CC BY-NC 3.0 (collection); individual Freesound clips CC0/CC BY/CC BY-NC - NON-COMMERCIAL"

VBD_PARQUET = "https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k/resolve/main/data/test-00000-of-00001.parquet"
VBD_LICENSE = "CC BY 4.0 (Valentini-Botinhao et al., Edinburgh DataShare 10283/2791; 16 kHz HF mirror JacobLinCool/VoiceBank-DEMAND-16k)"

DRONE_TARBALLS = [
    "https://codeload.github.com/saraalemadi/DroneAudioDataset/tar.gz/refs/heads/master",
    "https://codeload.github.com/saraalemadi/DroneAudioDataset/tar.gz/refs/heads/main",
]
DRONE_LICENSE = ("Al-Emadi et al. DroneAudioDataset (GitHub saraalemadi/DroneAudioDataset): NO LICENSE FILE - "
                 "all rights reserved by default. Used for internal R&D only; DO NOT redistribute.")


class HashingReader(io.RawIOBase):
    """File-like wrapper that hashes and counts bytes while they are read."""

    def __init__(self, raw):
        self.raw = raw
        self.sha = hashlib.sha256()
        self.n = 0

    def readable(self):
        return True

    def readinto(self, b):
        chunk = self.raw.read(len(b))
        if not chunk:
            return 0
        n = len(chunk)
        b[:n] = chunk
        self.sha.update(chunk)
        self.n += n
        return n


def _load_prov(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": {}}


def _save_prov(path: Path, prov: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2))


def _stream_tar(url: str, dest: Path) -> tuple[str, int]:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        hr = HashingReader(r.raw)
        buf = io.BufferedReader(hr, buffer_size=1 << 20)
        with tarfile.open(fileobj=buf, mode="r|gz") as tf:
            tf.extractall(dest, filter="data")
        # drain any remaining bytes so the hash covers the full archive
        while buf.read(1 << 20):
            pass
    return hr.sha.hexdigest(), hr.n


def get_librispeech(root: Path, prov: dict) -> None:
    for name, (url, lic) in LIBRISPEECH.items():
        if (root / "LibriSpeech" / name).exists():
            print(f"[librispeech] {name} present - skip")
            continue
        print(f"[librispeech] streaming {url}")
        sha, n = _stream_tar(url, root)
        prov["sources"][f"librispeech/{name}"] = {"url": url, "license": lic, "sha256_archive": sha, "bytes": n,
                                                   "fetched": time.strftime("%Y-%m-%d")}
        print(f"[librispeech] {name}: {n/1e6:.1f} MB sha256={sha[:16]}...")


LIBRISPEECH_PARTIAL = {
    # name: (url, license, max extracted bytes)
    "train-clean-100": ("https://www.openslr.org/resources/12/train-clean-100.tar.gz",
                        "CC BY 4.0 (LibriSpeech, OpenSLR SLR12) - PARTIAL: archive prefix only", 2_000_000_000),
}


def get_librispeech_partial(root: Path, prov: dict) -> None:
    """Stream a LibriSpeech archive and stop (cleanly, between members) after `max_bytes` of
    extracted audio.  Provenance records the SHA-256 and length of the consumed archive PREFIX,
    so the exact bytes can be re-verified with `curl -r 0-<n-1> <url> | sha256sum`."""
    for name, (url, lic, max_bytes) in LIBRISPEECH_PARTIAL.items():
        key = f"librispeech/{name}-partial"
        if key in prov["sources"]:
            print(f"[librispeech] {name} partial present - skip")
            continue
        extracted, n_files = 0, 0
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            r.raw.decode_content = False
            hr = HashingReader(r.raw)
            buf = io.BufferedReader(hr, buffer_size=1 << 20)
            with tarfile.open(fileobj=buf, mode="r|gz") as tf:
                for m in tf:
                    tf.extract(m, root, filter="data")
                    if m.isfile():
                        extracted += m.size
                        n_files += 1
                    if extracted >= max_bytes and m.isfile() and m.name.endswith(".txt"):
                        break   # stop after a chapter transcript, i.e. at a chapter boundary
        prov["sources"][key] = {"url": url, "license": lic, "partial": True,
                                "archive_prefix_bytes_read": hr.n, "sha256_archive_prefix": hr.sha.hexdigest(),
                                "extracted_bytes": extracted, "n_files": n_files,
                                "fetched": time.strftime("%Y-%m-%d")}
        print(f"[librispeech] {name}: extracted {extracted/1e9:.2f} GB in {n_files} files "
              f"(read {hr.n/1e9:.2f} GB of archive)")


def get_noisex(root: Path, prov: dict) -> None:
    from scipy.io import loadmat

    mdir, wdir = root / "noisex92" / "mat", root / "noisex92" / "wav16k"
    mdir.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)
    for name in NOISEX_FILES:
        mpath = mdir / f"{name}.mat"
        if not mpath.exists():
            r = requests.get(NOISEX_BASE + f"{name}.mat", timeout=120)
            r.raise_for_status()
            mpath.write_bytes(r.content)
        data = mpath.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        m = loadmat(io.BytesIO(data))
        key = [k for k in m if not k.startswith("__")][0]
        x = np.asarray(m[key], dtype=np.float64).squeeze()
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
        y = soxr.resample(x, NOISEX_FS, FS, quality="VHQ").astype(np.float32)
        sf.write(wdir / f"{name}.wav", y, FS, subtype="PCM_16")
        prov["sources"][f"noisex92/{name}"] = {"url": NOISEX_BASE + f"{name}.mat", "license": NOISEX_LICENSE,
                                               "sha256": sha, "bytes": len(data), "orig_fs": NOISEX_FS,
                                               "duration_s": round(len(x) / NOISEX_FS, 2)}
        print(f"[noisex] {name}: {len(x)/NOISEX_FS:.1f}s")


def get_esc50(root: Path, prov: dict) -> None:
    out = root / "esc50"
    (out / "audio16k").mkdir(parents=True, exist_ok=True)
    meta = requests.get(ESC50_RAW + "meta/esc50.csv", timeout=60).text
    (out / "esc50.csv").write_text(meta)
    rows = [r for r in csv.DictReader(io.StringIO(meta)) if r["category"] in ESC50_CLASSES]
    sess = requests.Session()
    files = {}
    for i, r in enumerate(rows):
        dst = out / "audio16k" / r["filename"]
        if not dst.exists():
            for attempt in range(4):
                try:
                    resp = sess.get(ESC50_RAW + "audio/" + r["filename"], timeout=60)
                    resp.raise_for_status()
                    break
                except Exception as e:  # transient network errors
                    print("retry", r["filename"], e)
                    time.sleep(2 + attempt * 2)
            else:
                raise RuntimeError(f"failed {r['filename']}")
            x, fs = sf.read(io.BytesIO(resp.content), dtype="float64")
            if x.ndim > 1:
                x = x.mean(axis=1)
            y = soxr.resample(x, fs, FS, quality="VHQ").astype(np.float32)
            sf.write(dst, y, FS, subtype="PCM_16")
            files[r["filename"]] = hashlib.sha256(resp.content).hexdigest()
        if i % 100 == 0:
            print(f"[esc50] {i}/{len(rows)}")
    prov["sources"]["esc50"] = {"url": ESC50_RAW, "license": ESC50_LICENSE, "classes": ESC50_CLASSES,
                                "n_files": len(rows), "sha256_per_original_file": files or prov["sources"].get("esc50", {}).get("sha256_per_original_file", {})}


def get_vbdemand(root: Path, prov: dict) -> None:
    import pyarrow.parquet as pq

    out = root / "vbdemand_test"
    if (out / "clean").exists() and len(list((out / "clean").glob("*.wav"))) > 800:
        print("[vbdemand] present - skip")
        return
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "noisy").mkdir(parents=True, exist_ok=True)
    r = requests.get(VBD_PARQUET, timeout=300)
    r.raise_for_status()
    sha = hashlib.sha256(r.content).hexdigest()
    tab = pq.read_table(io.BytesIO(r.content))
    cols = tab.column_names
    print("[vbdemand] columns:", cols)
    d = tab.to_pydict()
    n = len(d[cols[0]])
    for i in range(n):
        rec = {c: d[c][i] for c in cols}
        name = None
        for key in ("id", "name", "filename", "file"):
            if key in rec and isinstance(rec[key], str):
                name = Path(rec[key]).stem
        for kind in ("clean", "noisy"):
            a = rec[kind]
            if isinstance(a, dict) and a.get("bytes"):
                x, fs = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                if name is None and a.get("path"):
                    name = Path(a["path"]).stem
            else:
                x, fs = np.asarray(a["array"], dtype=np.float32), a["sampling_rate"]
            if fs != FS:
                x = soxr.resample(x, fs, FS).astype(np.float32)
            sf.write(out / kind / f"{name or i:}.wav", x, FS, subtype="PCM_16")
    prov["sources"]["vbdemand_test"] = {"url": VBD_PARQUET, "license": VBD_LICENSE, "sha256": sha,
                                        "bytes": len(r.content), "n_pairs": n}
    print(f"[vbdemand] wrote {n} pairs")


def get_drone(root: Path, prov: dict) -> None:
    out = root / "drone"
    if out.exists() and any(out.rglob("*.wav")):
        print("[drone] present - skip")
        return
    out.mkdir(parents=True, exist_ok=True)
    for url in DRONE_TARBALLS:
        try:
            sha, n = _stream_tar(url, out)
            prov["sources"]["drone_audio_dataset"] = {"url": url, "license": DRONE_LICENSE, "sha256_archive": sha,
                                                      "bytes": n}
            print(f"[drone] {n/1e6:.1f} MB")
            (out / "LICENSE_NOTICE.txt").write_text(DRONE_LICENSE + "\n")
            return
        except requests.HTTPError as e:
            print("[drone] try next", e)
    raise RuntimeError("drone dataset download failed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--manifest", default="data/manifests/provenance.json")
    ap.add_argument("--only", default="noisex,esc50,vbdemand,drone,librispeech")
    a = ap.parse_args(argv)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    mpath = Path(a.manifest)
    prov = _load_prov(mpath)
    steps = {"librispeech": get_librispeech, "noisex": get_noisex, "esc50": get_esc50,
             "vbdemand": get_vbdemand, "drone": get_drone, "librispeech_partial": get_librispeech_partial}
    for s in a.only.split(","):
        t0 = time.time()
        steps[s](root, prov)
        _save_prov(mpath, prov)
        print(f"== {s} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
