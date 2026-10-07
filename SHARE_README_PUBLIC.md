# D-ANC project: licence-safe shared copy (read this first)

This is the **public variant** of the D-ANC project package (`DRDO_Project_PUBLIC_2026-10-05.zip`: 2.6 GB, unpacks to
2.6 GB in 17 435 files). It contains the complete code, configurations, documentation, report, result tables, per-file
metrics, figures and logs. It leaves out every **audio file and trained model** that holds, or was derived from,
third-party audio that may not be redistributed. The full package (`DRDO_Project_TEAM_2026-10-05.zip`, guide
`SHARE_README.md`) is for the project team only.

Defence material: confirm with the project owner that organisation clearance covers you before passing this on.

## 1. What was left out, and why

| Left out | Why |
|---|---|
| `data/raw/drone/` | DroneAudioDataset: no licence file, all rights reserved |
| `data/raw/noisex92/` | NOISEX-92: redistribution terms unclear |
| `data/raw/esc50/` | ESC-50: CC BY-NC 3.0 (non-commercial); left out so this package carries no non-commercial terms |
| `noisy/`, `primary/` and `reference/` folders of every test set in `data/testsets/` and `dist/jetson_bundle/data/testsets/` | mixtures containing the audio above (the `clean/` references and `meta.csv` are kept) |
| `reports/audio/enhanced_*`, except `enhanced_vbdemand_dfn3/` | processed versions of those mixtures |
| the noisy and enhanced files of three of the four `reports/audio/demo/` scenes (their clean references are kept) | mixtures with NOISEX-92 or ESC-50 noise; the fourth scene, `0135_impulsive_p00`, uses only synthetic explosion noise |
| `checkpoints/`, the trained `exports/*.onnx` models, `dist/jetson_bundle/exports/*.onnx` | trained on the audio above; need legal review before release |
| `docs/references/*.pdf` | © DESIDOC; cite DOI 10.14429/dsj.21095 instead |
| `docs/workflows/state/`, `docs/workflows/journals/` | internal records of the AI development tool (token counts, agent ids, time zone) |
| `SHARE_README.md` | the team guide; replaced by this file |
| the full project's `reports/MANIFEST.sha256` | replaced by this package's own manifest (§2) |
| `.venv/`, caches, `.DS_Store` | environment; rebuild it as in §3 |

**Kept:**
- all code, configs, tests and docs;
- LibriSpeech and VoiceBank+DEMAND (CC BY 4.0, `data/ATTRIBUTION.md`);
- the clean references and `meta.csv` of every test set;
- the RIR bank and the synthetic-noise examples (project-generated);
- the DeepFilterNet3 reference model and its packages (permissive licences; the packages are built for macOS arm64:
  elsewhere, install `deepfilternet==0.5.6` into a separate folder);
- every report, table, per-file metric CSV, figure and log.

Non-audio files derived from the restricted mixtures are kept: per-file metrics, `meta.csv` source names, and the
example-spectrogram figure `reports/figures/defence_v1_example_spectrograms.png`.

The development records contain the developer's local user path (logs, `docs/workflows/*.js`, `FILE_INDEX.md` §15),
and `docs/PROJECT_BRIEF.md` reproduces the owner's original requests word for word. No credentials or keys were found
(`docs/research/pre_share_review_2026-10-05.md`).

## 2. Check the download and the files

Free disk needed: about 5.5 GB (zip plus unpacked copy), plus about 1.5 GB for the Python environment.

```bash
shasum -a 256 DRDO_Project_PUBLIC_2026-10-05.zip      # compare with the .sha256 file (Linux: sha256sum; Windows: certutil -hashfile <zip> SHA256)
unzip -q DRDO_Project_PUBLIC_2026-10-05.zip && cd DRDO_Project      # Info-ZIP unzip or 7-Zip (Zip64)
shasum -a 256 -c reports/MANIFEST.sha256 | grep -v ': OK$'   # prints nothing if every file matches (Linux: sha256sum -c)
```

- In this package `reports/MANIFEST.sha256` lists exactly the files it contains (17 434), except itself.
- Run the check before `pip install -e .`, which rewrites `src/danc.egg-info/SOURCES.txt`.
- `dist/jetson_bundle/SHA256SUMS` and `exports/DEPLOYMENT.json` still name the left-out models and mixtures. The
  bundle check therefore reports 28 missing files; that is expected here.
- **On Windows,** use 7-Zip to unpack, and Git Bash or WSL for the checks: `sha256sum -c reports/MANIFEST.sha256 | grep -v ': OK$'`.

## 3. Set up and what you can run

Prerequisites:
- Python 3.11.
- A C compiler for `pesq`: `xcode-select --install` on macOS, `sudo apt install build-essential python3-dev` on
  Linux, the MSVC Build Tools on Windows.
- `libportaudio2`, only for live audio.
- On CPU-only Linux with `uv`, add `--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match`.

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt && uv pip install -e .     # or: pip install -r requirements.txt && pip install -e .
python -m pytest -q                       # measured on this package: 94 passed, 9 skipped (they need the left-out models / mixtures)
python scripts/make_tables.py > /tmp/tables.md && cmp /tmp/tables.md reports/results/tables.md   # identical: every table rebuilds from per_file.csv
python scripts/make_figures_round2.py     # rebuilds the four round-2 figures from the results
```

- **Without the left-out material,** you can read the code and the report and check every reported number against
  `reports/results/*/per_file.csv`.
- **The deployment self-check (`deploy/jetson/verify_install.py`), the evaluations and the round-1 example
  spectrograms need the trained models and the test-set mixtures.** They report missing files in this package.
- **To rebuild what was left out yourself:**
  1. `python -m danc.data.download --root data/raw --only noisex,esc50,drone` fetches the left-out corpora from their
     original sources. You are then responsible for their terms. The DroneAudioDataset has **no licence**: do not use
     it without the authors' permission.
  2. `python -m danc.data.build_testset` (and `build_edgeset`, `build_dual_babble`, `build_highsnr`) re-creates the
     test sets from fixed seeds.
  3. The training commands in `reports/REPORT.md` §10 re-create the models.

## 4. Where to start

1. `reports/REPORT.md`: §0 is the summary, §8 the results.
2. `README.md`.
3. `exports/deployment_selection.json`: which models were deployed.
4. `deploy/jetson/QUICKSTART.md`.
5. `FILE_INDEX.md`, which describes the full (team) project. Files it lists that are absent here are the ones in §1.
   Where it says "start with `SHARE_README.md`", read this file instead.
