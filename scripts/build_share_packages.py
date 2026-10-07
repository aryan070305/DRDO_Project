"""Build the shared zip packages NEXT TO the project folder (never inside it; nothing in the project is modified).

    python scripts/build_share_packages.py [--out ~/Desktop/DRDO_Project_SHARE_2026-10-05] [--tag 2026-10-05] [--only team|public]

TEAM    every project file except .venv/, __pycache__/, .pytest_cache/, .git/ and .DS_Store, checked by the project's
        own reports/MANIFEST.sha256                                  -> DRDO_Project_TEAM_<tag>.zip
PUBLIC  TEAM minus the restricted third-party material listed in SHARE_README_PUBLIC.md section 1, with its own
        reports/MANIFEST.sha256 covering exactly its files           -> DRDO_Project_PUBLIC_<tag>.zip
Each zip gets <zip>.sha256; SHARE_README.md and SHARE_README_PUBLIC.md are copied next to the zips. Every zip is
re-opened afterwards: member list compared with the planned list and every member's CRC checked (testzip).
Already-compressed formats are stored, everything else deflated; Zip64 is used automatically for > 4 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git"}
STORE_EXT = {".flac", ".zip", ".png", ".jpg", ".jpeg", ".pdf", ".npz", ".pt", ".gz", ".whl", ".so", ".dylib"}
PUBLIC_DEMO_KEEP = "0135_impulsive_p00_"          # synthetic-only scene (explosion); the other three use restricted noise
NOISY_DIR = re.compile(r"^(dist/jetson_bundle/)?data/testsets/[^/]+/(noisy|primary|reference)/")
ONNX = re.compile(r"^(dist/jetson_bundle/)?exports/[^/]+\.onnx$")


def project_files() -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            if f == ".DS_Store":
                continue
            out.append((Path(dirpath) / f).relative_to(ROOT).as_posix())
    return out


def public_excluded(rel: str) -> bool:
    if rel.startswith(("data/raw/drone/", "data/raw/noisex92/", "data/raw/esc50/", "checkpoints/",
                       "docs/workflows/state/", "docs/workflows/journals/")):
        return True
    if NOISY_DIR.match(rel):
        return True
    if rel.startswith("reports/audio/enhanced_") and not rel.startswith("reports/audio/enhanced_vbdemand_dfn3/"):
        return True
    if rel.startswith("reports/audio/demo/"):
        name = rel.rsplit("/", 1)[1]
        return not (name.startswith(PUBLIC_DEMO_KEEP) or "_0_clean_reference" in name)
    if ONNX.match(rel) and not rel.endswith("/dancnet_untrained_test.onnx"):
        return True
    if rel.startswith("docs/references/") and rel.endswith(".pdf"):
        return True
    return rel in ("SHARE_README.md", "reports/MANIFEST.sha256")      # team guide; full-project manifest (replaced)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(zpath: Path, files: list[str], extra: dict[str, bytes]) -> None:
    prefix = ROOT.name
    t0 = time.time()
    with zipfile.ZipFile(zpath, "w", allowZip64=True) as z:
        for i, rel in enumerate(files):
            ext = Path(rel).suffix.lower()
            z.write(ROOT / rel, f"{prefix}/{rel}",
                    compress_type=zipfile.ZIP_STORED if ext in STORE_EXT else zipfile.ZIP_DEFLATED)
            if i % 5000 == 0:
                print(f"  {zpath.name}: {i}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
        for rel, data in extra.items():
            info = zipfile.ZipInfo(f"{prefix}/{rel}", date_time=time.localtime()[:6])
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    # verification: member list and CRC of every member
    with zipfile.ZipFile(zpath) as z:
        names = [n[len(prefix) + 1:] for n in z.namelist()]
        assert sorted(names) == sorted(files + list(extra)), "zip member list differs from the planned list"
        bad = z.testzip()
        assert bad is None, f"CRC error in {bad}"
    digest = sha256(zpath)
    (zpath.parent / f"{zpath.name}.sha256").write_text(f"{digest}  {zpath.name}\n")
    print(f"{zpath.name}: {len(names)} files, {zpath.stat().st_size / 1e9:.2f} GB, CRC OK, sha256 {digest} "
          f"({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="2026-10-05")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", choices=["team", "public"], default=None)
    a = ap.parse_args()
    out = Path(a.out).expanduser() if a.out else ROOT.parent / f"{ROOT.name}_SHARE_{a.tag}"
    if out.resolve() == ROOT.resolve() or ROOT.resolve() in out.resolve().parents:
        raise SystemExit("--out must be outside the project folder")
    out.mkdir(parents=True, exist_ok=True)
    files = project_files()
    print(f"project files to consider: {len(files)}", flush=True)
    if a.only in (None, "team"):
        z = out / f"{ROOT.name}_TEAM_{a.tag}.zip"
        if z.exists():
            raise SystemExit(f"{z} exists; choose another --out or --tag (nothing is overwritten)")
        build(z, files, {})
    if a.only in (None, "public"):
        z = out / f"{ROOT.name}_PUBLIC_{a.tag}.zip"
        if z.exists():
            raise SystemExit(f"{z} exists; choose another --out or --tag (nothing is overwritten)")
        pub = [f for f in files if not public_excluded(f)]
        print(f"public: {len(pub)} files ({len(files) - len(pub)} left out); hashing for its manifest", flush=True)
        manifest = "".join(f"{sha256(ROOT / f)}  {f}\n" for f in pub)
        build(z, pub, {"reports/MANIFEST.sha256": manifest.encode()})
    for readme in ("SHARE_README.md", "SHARE_README_PUBLIC.md"):
        dst = out / readme
        if not dst.exists():
            shutil.copy2(ROOT / readme, dst)
    print(f"share folder: {out}", flush=True)


if __name__ == "__main__":
    main()
