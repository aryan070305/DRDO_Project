# Sharing checklist for the D-ANC project folder

How to hand the project over so the recipient gets every file, can check that nothing was lost in transfer,
and can rebuild the environment. It also lists the licence points to clear **before** anything leaves the
team. The file-by-file inventory is in [`../FILE_INDEX.md`](../FILE_INDEX.md).

Updated for **round 2** (2026-10-05): DANCNet-v3, the two-microphone network and cascade, ablations, edge-case,
high-SNR and crew-babble test sets, and the rebuilt Jetson bundle.

---

## 1. What to share

**Share the whole `DRDO_Project/` folder.** Every code, configuration, data, model, result, log and
document file belongs to the project. The only non-content files are the rebuildable caches, the empty
`*.lock` files that the evaluator uses against write races, macOS `.DS_Store` files (never shipped), two empty directories
(`src/x/`, `dist/jetson_bundle/src/x/`; zips carry no empty directories) and the kept-for-the-record duplicates and incident logs that
`FILE_INDEX.md` names. Do not cherry-pick sub-folders: the report, the results and the integrity manifest refer
to files across all of them. Related material that stays outside the folder on purpose (the development
session's transcripts and scratch files, the Python interpreter) is listed in `FILE_INDEX.md` §15.

| Part | Size | Needed by the recipient? | Notes |
|---|---:|---|---|
| `src/`, `configs/`, `scripts/`, `tests/`, `deploy/`, `pyproject.toml`, `requirements.txt` | < 1 MB | Yes | All the code, including the round-2 two-microphone training, cascade engine, queue scripts and Jetson kit |
| `README.md`, `FILE_INDEX.md`, `docs/`, `reports/REPORT.md`, `reports/REPORT_round1.md`, `reports/site/` | ≈ 6 MB | Yes | All the documentation. `REPORT.md` is the round-2 report; `REPORT_round1.md` is the round-1 report, kept. `reports/site/index.html` (round-2 build, 2026-10-05) opens in any browser. `SHARE_README.md` is the recipient's entry point |
| `checkpoints/`, `exports/` | ≈ 215 MB | Yes | Trained and exported models. `exports/deployment_selection.json` chooses the deployed set; `exports/DEPLOYMENT.json` lists it with hashes (round 1: `*_round1.json`) |
| `dist/jetson_bundle/` | 13 MB | Yes, for the Jetson | Run-time bundle for the Jetson (code, the 3 deployed models, 16 verification scenes, `SHA256SUMS`). Copy of project files; rebuildable with `deploy/jetson/package_for_jetson.sh` |
| `reports/results/`, `reports/figures/`, `logs/` | ≈ 20 MB | Yes | Evidence behind every number in the report, including the round-2 decision rules and decisions (`reports/results/*_decision*.md`) |
| `reports/audio/` | 936 MB | Yes, for listening checks | Enhanced audio of the evaluated systems (23 `enhanced_*` folders), plus the `demo/` listening set (round-1 models) |
| `data/testsets/`, `data/manifests/`, `data/rir_bank/`, `data/synthetic_examples/` | ≈ 430 MB | Yes, to re-run the evaluation | Six fixed test sets (round 2 added `defence_edge_v1`, `defence_highsnr_v1`, `dualmic_babble_v1`) and data provenance. Licence caveats in §4 apply |
| `data/raw/` | **3.3 GB** | Only to retrain | Downloaded corpora. Re-downloadable with `danc.data.download`. Licence caveats in §4 apply |
| `third_party/` | 64 MB | Only to re-run the DeepFilterNet3 reference | Permissive: MIT / Apache-2.0 / BSD; numpy's bundled GCC runtime libraries GPL-3.0-with-runtime-exception / LGPL |
| `.venv/` | **1.2 GB** | No (see §3) | Machine-specific; the recipient rebuilds it |
| `__pycache__/`, `.pytest_cache/` | < 1 MB | No | Caches, rebuilt automatically |

Total: about **6.2 GB** with `.venv`; without `.venv`, caches and `.DS_Store` (what is shipped) **5.1 GB** in 53 701 files.

**Round-2 additions that must be in the copy** (all inside the folders above; listed so a reviewer can check):
- Code: `src/danc/data/{build_edgeset,build_dual_babble,build_highsnr,dual_mixer}.py`,
  `src/danc/train/train_dual.py`, `src/danc/inference/postfilter.py` and the extended `engine.py`, `realtime.py`,
  `evaluate.py`; the new scripts in `scripts/` (`choose_batch.py`, `eval_edge.py`, `make_figures_round2.py`,
  `paired_stats.py`, `band_error_analysis.py`, `tune_postfilter.py`, `transparency_test.py`, `rir_stats.py`,
  `run_round2{,b,c,d}_chain.sh`, `eval_v3ll_when_ready.sh`, `eval_babble_when_ready.sh`) and the extended
  `benchmark_latency.py` and `make_tables.py`;
  `tests/test_two_mic.py`, `test_train_resume.py`, `test_deploy_kit.py`.
- Configs: `configs/train_v3_{hq,ll,2mic,hq_cont}.yaml`, `configs/ablations/` (6), `configs/smoke/` (2).
- Models: `checkpoints/{v3hq,v3hq_cont,v3ll,v3_2mic,abl_*,smoke_dual_resume}/`;
  `exports/dancnet_{v3hq,v3hq_cont,v3ll,v3_2mic}.onnx` + `.json`, `deployment_selection*.json`,
  `DEPLOYMENT.json`, `DEPLOYMENT_round1.json`.
- Data: `data/testsets/{defence_edge_v1,defence_highsnr_v1,dualmic_babble_v1}/`.
- Results: `reports/REPORT_round1.md`, the new result folders and files in `reports/results/`, the `round2_*` and
  `latency_hist_round2*` figures, `reports/figures/round1/`, the new `reports/audio/enhanced_*` folders, and about
  65 new logs in `logs/`.
- Deployment: `deploy/jetson/QUICKSTART.md`, `check_channels.py` and the updated kit; `dist/jetson_bundle/`.

**Deployed configurations the recipient gets** (see `exports/deployment_selection.json`, `README.md` and
`deploy/jetson/QUICKSTART.md` step 8):

| Configuration | Files | Decided in |
|---|---|---|
| **HQ** (40 ms), single microphone or behind the NLMS hybrid | `exports/dancnet_v3hq_cont.onnx` (`checkpoints/v3hq_cont/best.pt`) | `reports/results/v3hq_cont_decision.md` |
| **LL** (20 ms), unchanged from round 1 | `exports/dancnet_ll.onnx` (`checkpoints/ll/best.pt`) | `reports/results/finetune_decision_ll.md` |
| **TWO_MIC** (40 ms): gated NLMS → two-microphone network, HQ model in hot standby for a dead reference mic | `exports/dancnet_v3_2mic.onnx` (`checkpoints/v3_2mic/best.pt`), run with `--two-mic-cascade --fallback-onnx exports/dancnet_v3hq_cont.onnx` | `reports/results/cascade_decision.md` |

**About `.venv`:** it is safe to include, but it will **not work on another computer**. Its `pyvenv.cfg` points
to a Python interpreter under the original user's home directory, and its packages are built for macOS on
Apple silicon. Including it costs 1.2 GB and gives the recipient nothing they cannot rebuild in a few minutes
(§3). Recommendation: keep it on the development machine, and share the rest of the folder.

---

## 2. How to package and transfer

Work on a **copy**. Never move or delete files in the original folder. The packages are written **next to** the
project (not inside it), so the project's own `reports/MANIFEST.sha256` stays valid.

**What was built on 2026-10-05** (folder `~/Desktop/DRDO_Project_SHARE_2026-10-05/`, uploaded to Google Drive):

| Package | Contents | For whom |
|---|---|---|
| `DRDO_Project_TEAM_2026-10-05.zip` (+ `.sha256`), 4.9 GB, unpacks to 5.1 GB | Every project file except `.venv/`, `__pycache__/`, `.pytest_cache/` and `.DS_Store` (Python zipfile via `scripts/build_share_packages.py`, Zip64, no `._*` or `__MACOSX/` entries; re-opened and CRC-checked after writing). Checks with the project's own `reports/MANIFEST.sha256` | **Named members of the project team only** (data licences, §4) |
| `DRDO_Project_PUBLIC_2026-10-05.zip` (+ `.sha256`), 2.6 GB | The same minus the restricted material listed in `SHARE_README_PUBLIC.md`: drone and NOISEX-92 audio, ESC-50, the noisy sides of the test sets, enhanced audio derived from them, the trained models and the DSJ PDF. Carries its own `reports/MANIFEST.sha256` | Reviewers outside the team, after organisation clearance |
| `SHARE_README.md`, `SHARE_README_PUBLIC.md` | The recipient guides, readable on Drive without unzipping (also at the top of the zips) | Team / outside reviewers respectively |

The commands are recorded in `scripts/build_share_packages.py`. Send each zip's SHA-256 (from its `.sha256` file)
to the recipients separately from the Drive link. In Google Drive, share the team package as **Restricted** to named
accounts, never "Anyone with the link". **Drive sharing set on a folder applies to every file in it:** keep the
PUBLIC zip, its `.sha256` and `SHARE_README_PUBLIC.md` in a separate Drive folder, and never share the folder that
holds the TEAM zip with anyone outside the named team.

`.DS_Store` files are **not** shipped and **not** listed in `reports/MANIFEST.sha256` (`scripts/make_manifest.py`
skips them, because macOS Finder rewrites them).

**Jetson bundle on its own:** `dist/jetson_bundle/` is a self-contained 13 MB copy for the target. To send it
separately, archive it **outside** the project folder:
`COPYFILE_DISABLE=1 tar -czf ~/Desktop/jetson_bundle.tar.gz -C ~/Desktop/DRDO_Project/dist jetson_bundle`.
The bundle still contains the round-1 HQ model `exports/dancnet_hqft.onnx`, because the packaging script never
deletes; the bundle's `DEPLOYMENT.json` does not use it. Its test scenes and models carry the §4 caveats.

---

## 3. What the recipient does

### 3.1 Check the transfer

```bash
shasum -a 256 -c DRDO_Project_TEAM_2026-10-05.zip.sha256    # macOS; on Linux: sha256sum -c ...; Windows: certutil -hashfile <zip> SHA256
unzip -q DRDO_Project_TEAM_2026-10-05.zip && cd DRDO_Project  # use Info-ZIP unzip or 7-Zip; the zip is > 4 GB (Zip64)
shasum -a 256 -c reports/MANIFEST.sha256 | grep -v ': OK$'  # prints nothing when every file matches
# On Linux: sha256sum -c reports/MANIFEST.sha256 | grep -v ': OK$'
(cd dist/jetson_bundle && shasum -a 256 -c SHA256SUMS | grep -v ': OK$')   # Linux: sha256sum -c SHA256SUMS
```

Run the manifest check **before** setting up the environment (§3.2): `pip install -e .` rewrites
`src/danc.egg-info/SOURCES.txt`, which is listed in the manifest.

`reports/MANIFEST.sha256` covers every shipped project file except `third_party/dfn_pkgs/`, the empty `*.lock`
files and the manifest itself (and, by construction, the excluded `.venv/`, caches and
`.DS_Store`). Any `FAILED` line means a file is missing or damaged. The public package has its own manifest
covering exactly the files it contains.

### 3.2 Rebuild the Python environment (development or evaluation machine)

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/). The original environment was Python 3.11.15 with
`uv` 0.11.8 on macOS (Apple M4).

```bash
cd DRDO_Project
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
python -m pytest -q                      # 103 tests; on the development Mac: "103 passed" (logs/pytest_round2_final.out)
python deploy/jetson/verify_install.py   # deployment self-check of HQ, LL and TWO_MIC, no PyTorch needed;
                                         # on the development Mac: "115 PASS, 0 FAIL" (logs/verify_install_round2.out)
```

Without `uv`: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt &&
pip install -e .`

`verify_install.py` exits with code 0 when there is no FAIL line. Its timing checks depend on the machine; the
quality checks compare 16 scenes per model with the reference rows in `reports/results/`.

Notes:
- `requirements.txt` was frozen on macOS arm64. On Linux x86-64 with an NVIDIA GPU, install the matching
  CUDA build of `torch`/`torchaudio` from pytorch.org if the pinned versions do not resolve.
- `sounddevice` (live audio only) needs the PortAudio system library (`apt install libportaudio2` on Ubuntu).
- **NVIDIA Jetson AGX Orin:** do not use this recipe as is (it installs PyTorch and other development-only
  packages). Follow `deploy/jetson/QUICKSTART.md` (step by step, with `deploy/jetson/README.md` for background):
  copy `dist/jetson_bundle/` (or the whole folder) to `/opt/danc`, check `SHA256SUMS`, run
  `deploy/jetson/install.sh` and `python deploy/jetson/verify_install.py`. The runtime needs no PyTorch, uses the
  CUDA and TensorRT that ship with JetPack, and takes ONNX Runtime from a pip wheel (the CPU build from PyPI, or
  the Jetson-specific `onnxruntime-gpu` wheel matching the JetPack version for the GPU path). JetPack itself
  does not include ONNX Runtime. The TWO_MIC configuration runs on ONNX Runtime only (`trt_runner.py` supports
  single-input models). **None of this was run on Jetson hardware in this project.**
- DeepFilterNet3 reference only: `PYTHONPATH=third_party/dfn_pkgs python scripts/run_deepfilternet3.py ...`
  uses the isolated packages already in `third_party/dfn_pkgs/` (built for macOS arm64; on another platform
  install `deepfilternet==0.5.6` into a separate target directory).

### 3.3 Re-create `data/raw/` (only if it was not shipped)

```bash
python -m danc.data.download --root data/raw                               # NOISEX-92, ESC-50, VB+DEMAND, drone, LibriSpeech dev/test
python -m danc.data.download --root data/raw --only librispeech_partial    # part of train-clean-100
```

The train-clean-100 download in this project was partial (about 1.0 GB; see `data/manifests/provenance.json`).
The exact training files are listed in `data/manifests/speech_train_v2.json`, so a recipient who downloads more
than that still trains on the same list. Compare the hashes in `provenance.json` to confirm identical sources.
The test sets in `data/testsets/` are shipped as rendered files; the round-2 sets can also be rebuilt from
fixed seeds (`python -m danc.data.build_edgeset`, `build_dual_babble`, `build_highsnr`) once `data/raw/` exists.

### 3.4 Where to start reading

`README.md` (round-2 summary and deployed configurations) → `reports/REPORT.md` (or `reports/site/index.html`)
→ `FILE_INDEX.md` → `reports/results/*_decision_rule.md` and `*_decision*.md` (how each model change was
decided) → `docs/PROJECT_BRIEF.md` (what was asked) → `deploy/jetson/QUICKSTART.md` (deployment).
`reports/REPORT_round1.md` is the earlier report, for the history.

---

## 4. Licence caveats: clear these before sharing outside the team

Source of truth: `data/manifests/provenance.json`, `README.md` (data and licences) and `reports/REPORT.md` §9 (R9).
Round 2 used the same corpora; nothing in the licence position changed.

| Material | Where | Licence | What it means for sharing |
|---|---|---|---|
| **NOISEX-92** | `data/raw/noisex92/` | © TNO Soesterberg 1990; SPIB mirror; redistribution terms **not stated** | **Unclear.** Do not redistribute outside the team without legal review |
| **ESC-50** (16 classes) | `data/raw/esc50/` | CC BY-NC 3.0 (individual clips CC0 / CC BY / CC BY-NC) | **Non-commercial only**, with attribution. Not for commercial or product use |
| **DroneAudioDataset** (Al-Emadi et al.) | `data/raw/drone/` | **No licence file**: all rights reserved by default | Internal R&D only. **Do not redistribute** |
| LibriSpeech, Mini LibriSpeech | `data/raw/LibriSpeech/` | CC BY 4.0 | Shareable with attribution |
| VoiceBank+DEMAND test set | `data/raw/vbdemand_test/` | CC BY 4.0 | Shareable with attribution |
| Fixed test sets, including the round-2 sets `defence_edge_v1`, `defence_highsnr_v1`, `dualmic_babble_v1` | `data/testsets/` | Mixtures that contain the noise corpora above | **Inherit** the NOISEX-92, ESC-50 and drone restrictions |
| Enhanced audio | `reports/audio/` | Processed versions of those mixtures | Same caveat as the test sets |
| Trained models, including all round-2 models | `checkpoints/`, `exports/` | Trained on the data above | Legal review before shipping in a product (REPORT R9); retraining on cleared data is the mitigation |
| Jetson bundle | `dist/jetson_bundle/` | Contains the deployed models and 16 test scenes from `defence_v1` / `dualmic_v1` | Same caveats as the models and test sets; clear it before giving it to a hardware partner |
| PMSQE loss | `src/danc/train/third_party/` (and its copy in `dist/jetson_bundle/src/`) | MIT (Asteroid) | Keep `LICENSE_asteroid_MIT.txt` with it |
| DeepFilterNet3 weights and its runtime packages | `third_party/` | MIT / Apache-2.0 (weights); packages MIT / Apache-2.0 / BSD, numpy's GCC runtime libraries GPL-3.0-with-exception / LGPL | Keep the licence files with it (loguru's MIT notice was added on 2026-10-05) |
| Reference paper | `docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf` | Journal copyright (*Defence Science Journal*, DRDO DESIDOC) | Check the journal's terms before passing it outside the team |
| Synthetic noise, project code, results, documents | rest of the folder | Produced by this project | Owner's decision |

**Other sensitivity points before external sharing:**
- `docs/PROJECT_BRIEF.md` reproduces the owner's messages, and `docs/workflows/` contains the orchestration
  scripts and journals. Both contain local paths with the developer's user name, as do many logs in `logs/`,
  `dist/jetson_bundle/BUNDLE_INFO.txt` and `reports/results/verify_install_mac_round2.json`. Review them if the
  recipient is outside the organisation.
- The project is defence-related. Follow your organisation's export-control and classification procedure
  before sending anything externally.

---

## 5. Final checklist

Before sending:
- [x] All long-running jobs have finished (no training or evaluation still writing into the folder); checked 2026-10-05.
- [x] `README.md`, `reports/REPORT.md` and `FILE_INDEX.md` describe the final state (deployed models, results);
      `FILE_INDEX.md` re-checked against `find` (2026-10-05) and updated after the pre-share review.
- [x] `reports/site/` rebuilt with `python scripts/build_report_html.py` (round-2 build, 2026-10-05).
- [x] Workflow records complete: final states in `docs/workflows/state/`, the complete journal of `wf_14dd115a-9e9`
      as `journals/wf_14dd115a-9e9.final.jsonl` (the partial copy is kept), and the 2026-10-05 pre-share review
      `wf_c549189e-2c8` (script, state, journal).
- [x] `exports/deployment_selection.json` names the intended deployed set (HQ = `dancnet_v3hq_cont.onnx`,
      LL = `dancnet_ll.onnx`, TWO_MIC = `dancnet_v3_2mic.onnx` cascade with HQ fallback). `scripts/make_manifest.py`
      reads this file (no hard-coded list any more) and rewrites `exports/DEPLOYMENT.json` from it.
- [x] If `exports/` or the Jetson kit changed after 07:31 on 2026-10-05, `deploy/jetson/package_for_jetson.sh`
      re-run so `dist/jetson_bundle/` matches, and `(cd dist/jetson_bundle && shasum -a 256 -c SHA256SUMS)` is all OK.
- [x] `python -m pytest -q` passes in the project venv (103 tests).
- [x] `python deploy/jetson/verify_install.py` gives 0 FAIL (Mac reference: 115 PASS, 0 FAIL, 0 WARN, 10 INFO).
- [x] `python scripts/make_manifest.py` has been run **last**, after every other file (including this checklist
      and `FILE_INDEX.md`) is final, so `reports/MANIFEST.sha256` and `exports/DEPLOYMENT.json` match the final files.
- [x] `shasum -a 256 -c reports/MANIFEST.sha256 | grep -v ': OK$'` prints nothing.
- [ ] The licence decision is made for NOISEX-92, ESC-50 and the drone dataset, and for the derived test sets
      (including the three round-2 sets), audio, models and the Jetson bundle (§4). For an external recipient, a
      reduced copy has been built if needed. **Owner's decision** (who gets which package).
- [x] Reduced, licence-safe package built (PUBLIC, §2).
- [ ] Organisation clearance obtained for an external recipient (defence material, user paths in `docs/`, logs).
      **Owner's decision.**
- [x] Archives made outside the project folder with `scripts/build_share_packages.py` (not Finder's Compress); `.venv`,
      caches and `.DS_Store` excluded; no `._*` or `__MACOSX/` entries; both zips independently verified
      (workflow `wf_6ba6f06f-920`: member lists, CRCs, manifests, licence screen, stand-alone test runs).
- [ ] Archive hash sent separately from the Drive link. **Owner, at upload time.**
- [ ] On Drive, the PUBLIC package sits in a different folder from the TEAM package. **Owner, at upload time.**
- [x] No file deleted or moved in the original `DRDO_Project/` folder.

After receiving (recipient):
- [ ] Archive hash matches.
- [ ] `shasum -a 256 -c reports/MANIFEST.sha256` shows only OK lines (run it before setting up the venv).
- [ ] Venv rebuilt (§3.2), `python -m pytest -q` passes (103 tests) and `python deploy/jetson/verify_install.py`
      reports 0 FAIL.
- [ ] `reports/site/index.html` opens and the figures show.
- [ ] For Jetson deployment: `dist/jetson_bundle/SHA256SUMS` checks out on the Jetson, then
      `deploy/jetson/QUICKSTART.md` followed from step 1 (HQ, LL or the two-mic cascade in step 8).
