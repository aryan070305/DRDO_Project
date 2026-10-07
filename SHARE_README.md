# D-ANC project: shared copy for the project team (read this first)

**This package is for named members of the project team only. Do not forward it or re-share the link.** It
contains third-party data that may not be redistributed (§4). A licence-safe package for people outside the team is
described in `SHARE_README_PUBLIC.md`.

This archive holds the complete D-ANC project: hybrid AI/ML adaptive noise cancellation for defence voice. It
contains every source file, configuration, dataset used, trained model, result, log, figure, report and deployment
file. It was packaged on 2026-10-05, after round 2 of the work, as `DRDO_Project_TEAM_2026-10-05.zip` (4.9 GB; unpacks to
5.1 GB in 53 701 files).

## 1. What is in it, and what is deliberately not

| Included | Size | Notes |
|---|---|---|
| Code: `src/`, `scripts/`, `tests/`, `configs/`, `deploy/`, `pyproject.toml`, `requirements.txt` | ≈ 1 MB | 103 unit tests |
| Data: `data/raw/` (downloaded corpora), `data/testsets/` (six fixed test sets), `data/rir_bank/`, `data/synthetic_examples/`, `data/manifests/`, `data/ATTRIBUTION.md` | ≈ 3.9 GB | **Licence caveats in §4** |
| Models: `checkpoints/` (PyTorch), `exports/` (ONNX; deployed set in `exports/DEPLOYMENT.json`) | ≈ 0.2 GB | |
| Results: `reports/` (report, tables, per-file results, figures, enhanced audio), `logs/` | ≈ 0.99 GB | |
| Docs: `docs/`, `FILE_INDEX.md` (what every file group is), `README.md` | ≈ 2 MB | |
| Jetson bundle: `dist/jetson_bundle/` (what to copy to the board) | 13 MB | |
| `third_party/`: the DeepFilterNet3 reference model and its isolated packages, built for macOS arm64 | 64 MB | |

**Not included:** `.venv/` (the Python environment; 1.2 GB, works only on the original Mac, rebuilt in §3), Python
caches (`__pycache__/`, `.pytest_cache/`) and macOS `.DS_Store` files. Nothing else was left out. Material that
stayed on the development machine (session records, cached papers) is listed in `FILE_INDEX.md` §15.

## 2. Check the download and the files

The SHA-256 of the zip is in `DRDO_Project_TEAM_2026-10-05.zip.sha256` next to it on Google Drive; ask the sender to
confirm it separately. Free disk needed: about 10 GB (zip plus unpacked copy), plus about 1.5 GB for the Python
environment.

```bash
shasum -a 256 DRDO_Project_TEAM_2026-10-05.zip      # macOS; Linux: sha256sum ...; Windows: certutil -hashfile <zip> SHA256
unzip -q DRDO_Project_TEAM_2026-10-05.zip           # Info-ZIP unzip or 7-Zip (the zip is > 4 GB, Zip64)
cd DRDO_Project
shasum -a 256 -c reports/MANIFEST.sha256 | grep -v ': OK$'   # macOS; prints nothing if every file matches
sha256sum -c reports/MANIFEST.sha256 | grep -v ': OK$'       # Linux, same check
```

Run this check **before** §3, because `pip install -e .` rewrites `src/danc.egg-info/SOURCES.txt`, which is listed.
**On Windows,** unpack with 7-Zip and run the checks in Git Bash or WSL (`sha256sum -c ...`).
`reports/MANIFEST.sha256` lists 52 653 files: every shipped file except `third_party/dfn_pkgs/`, seven empty `*.lock`
files and the manifest itself.

## 3. Rebuild the environment and run the checks

Prerequisites:
- Python **3.11**. The pinned packages need it. The Jetson runtime in `deploy/` uses its own Python 3.10 requirements.
- A C compiler for `pesq`, which installs from source: `xcode-select --install` on macOS, `sudo apt install build-essential python3-dev` on Linux.
- `libportaudio2`, only for live audio.

On a CPU-only Linux machine, `--extra-index-url https://download.pytorch.org/whl/cpu` avoids the large CUDA wheels
(with `uv`, add `--index-strategy unsafe-best-match`). On Windows, `pesq` needs the MSVC Build Tools, and the venv is
activated with `.venv\Scripts\activate`.
The pins come from an Apple-silicon Mac (`uv pip freeze`) and contain no macOS-only package. They were not tested on
Linux.

```bash
# from inside DRDO_Project/
uv venv .venv --python 3.11          # or: python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .    # or: pip install -r requirements.txt && pip install -e .
python -m pytest -q                                           # expected: 103 passed (needs the whole archive unpacked, incl. data/raw)
python deploy/jetson/verify_install.py                       # expected: SUMMARY ... 0 FAIL ... -> PASS (115 PASS on the Mac)
```

The Jetson AGX Orin has its own torch-free install: `deploy/jetson/QUICKSTART.md` (12 numbered steps; the bundle is
already built in `dist/jetson_bundle/`). If you unzip on Windows, the executable bits are lost: run the shell scripts
as `bash <script>`. Nothing in this project was run on Jetson hardware.

## 4. Licences: why this package is team-only

| Material | Licence | What it means |
|---|---|---|
| LibriSpeech, Mini LibriSpeech, VoiceBank+DEMAND test (`data/raw/`) | CC BY 4.0 | Shareable with attribution (`data/ATTRIBUTION.md`) |
| ESC-50, 16 classes (`data/raw/esc50/`) | CC BY-NC 3.0 | **Non-commercial use only** |
| NOISEX-92 (`data/raw/noisex92/`) | © TNO 1990, NATO RSG-10 distribution; **redistribution terms unclear** | Treat as internal until cleared |
| DroneAudioDataset (`data/raw/drone/`) | **No licence file** | **Internal R&D only; do not redistribute** |
| ... its two `unknown/` folders (682 MB, not used by the code) | segments of all 50 ESC-50 classes (CC BY-NC 3.0) and Speech Commands background noise (CC BY 4.0) | as for ESC-50 |
| Test sets (`data/testsets/`), enhanced audio (`reports/audio/`, incl. `demo/`), Jetson bundle test scenes | derived from all of the above (each `meta.csv` names the noise source; `defence_edge_v1` does not) | **Inherit the most restrictive source above** |
| Trained models (`checkpoints/`, `exports/`, `dist/jetson_bundle/exports/`) | trained on LibriSpeech, ESC-50, NOISEX-92, drone and synthetic noise | **Legal review before any use outside the team** (`reports/REPORT.md` risk R9) |
| DeepFilterNet3 weights and packages (`third_party/`) | MIT / Apache-2.0 / BSD; numpy's bundled GCC runtime libraries GPL-3.0-with-runtime-exception / LGPL | Permissive; keep the licence files |
| PMSQE loss (`src/danc/train/third_party/`) | MIT (Asteroid) | Keep `LICENSE_asteroid_MIT.txt` |
| Reference paper (`docs/references/*.pdf`) | © 2026 DESIDOC (Defence Science Journal) | Internal reference copy; cite DOI 10.14429/dsj.21095 outside the team |
| Synthetic noise, RIR bank, project code and documentation | produced by this project | Owner's decision |

About 31 % of this archive (≈ 1.6 GB) is drone or NOISEX-92 material, raw or derived. Defence material may also need
organisation clearance before it leaves the organisation. Details: `docs/SHARING_CHECKLIST.md` §4,
`docs/research/pre_share_review_2026-10-05.md` (a dated review; its findings were acted on, so its counts are older than
§2 here).

The archive also contains development records with the developer's local user path (logs, `docs/workflows/`) and the
owner's original requests word for word (`docs/PROJECT_BRIEF.md`). No credentials or keys were found (pre-share
review).

## 5. Where to start

1. `reports/REPORT.md`, the full technical report: §0 is a two-page summary and §8 holds all results. A web version
   is `reports/site/index.html`; open it in a browser, with the `reports/site/figures/` folder next to it.
2. `README.md`: the repository layout and how to run things.
3. `exports/deployment_selection.json`: the three deployed configurations (HQ, LL, two-mic). Each change was adopted
   under a rule written before its test result was known (`reports/results/*decision*.md`). The cascade's *idea*
   came from an earlier test result, as its rule discloses.
4. `deploy/jetson/QUICKSTART.md`: deployment on the Jetson AGX Orin.
5. `FILE_INDEX.md`: what every file and folder is, and how it was produced.

**Results per test set:** `reports/results/tables.md` (all tables), `reports/results/<set>/summary.json` and
`per_file.csv` (every metric for every file and system).

**Listening:** the outputs of the deployed models are FLAC files with the same file ids as their inputs.
- **Single mic, HQ:** `reports/audio/enhanced_defence_v1_dancnet_v3hq_cont/`. Inputs are in `data/testsets/defence_v1/noisy/`, references in `.../clean/`.
- **Two-mic cascade:** `reports/audio/enhanced_dualmic_v1_hybrid2_v3_2mic/` (inputs `data/testsets/dualmic_v1/primary/`) and `reports/audio/enhanced_dualmic_babble_v1_hybrid2_v3_2mic/` (crew babble).
- **`reports/audio/demo/`** is a small listening set made with the round-1 models.
