# FILE_INDEX: every directory and file group in the D-ANC project

This index lists every top-level directory and every group of files in the project folder: what each one
is, how it was produced, and what kind of file it is. It is meant for a reviewer who receives the folder and
needs to find things or check that nothing is missing.

- **Final update:** 2026-10-05 (UTC+04), after the pre-share review and the package verification (`docs/research/pre_share_review_2026-10-05.md`):
  files that had produced shipped content but lived outside the folder were copied in (§4, §9, §10, §11), and
  `SHARE_README.md` and `data/ATTRIBUTION.md` were added. Shared as two zip packages (§13).
- **Re-checked:** 2026-10-05 07:40 (UTC+04) against `find` over the real tree, after round 2 (the overnight
  DANCNet-v3, two-microphone, ablation, edge-case, high-SNR, crew-babble and Jetson-kit work of 2026-10-04/05)
  had finished. No training, evaluation or queue script was running. Every directory and file group present at
  that time is listed below; files added in round 2 are marked **(round 2)**.
- **Previous version:** snapshot 2026-10-04 08:55, re-checked 09:00 (round 1 plus the start of round 2). The
  round-1 report is kept as `reports/REPORT_round1.md`.
- **Whole folder:** about **6.2 GB** including `.venv`. Without `.venv`, caches and `.DS_Store` (what is shipped):
  about 5.2 GB and 53 700 files, of which 37 400 are the raw corpora in `data/raw/` and 10 800 are enhanced audio
  in `reports/audio/`.
- **Type legend:**
  **SRC** source code / config · **DATA** input data (downloaded or generated) · **RES** generated result ·
  **MODEL** trained weights or exported model · **LOG** run log · **DOC** documentation ·
  **ENV** environment / cache (rebuildable, not project content) · **3P** third-party code or weights.
- Sizes are from `du -sh` (code folders without `__pycache__`). File counts exclude `__pycache__`.

Start with `SHARE_README.md` (team package) or `SHARE_README_PUBLIC.md` (public package): what was shared and how
to check it. Then `README.md`, then `reports/REPORT.md` (the full technical report), then this file.

**Deployed configurations after round 2** (chosen in `exports/deployment_selection.json`, written out with
hashes and the ONNX contract to `exports/DEPLOYMENT.json`; each decided by a pre-registered rule):

| Configuration | Model / command | Checkpoint | Decision |
|---|---|---|---|
| **HQ** (40 ms), single mic, or behind the NLMS hybrid | `exports/dancnet_v3hq_cont.onnx` | `checkpoints/v3hq_cont/best.pt` | `reports/results/v3hq_cont_decision.md` |
| **LL** (20 ms), unchanged from round 1 | `exports/dancnet_ll.onnx` | `checkpoints/ll/best.pt` | `reports/results/finetune_decision_ll.md` (v3 LL not adopted: `v3_decision_ll.md`) |
| **TWO_MIC** (40 ms): gated NLMS → two-microphone network, HQ model in hot standby | `danc_live.py --onnx exports/dancnet_v3_2mic.onnx --two-mic-cascade --fallback-onnx exports/dancnet_v3hq_cont.onnx` | `checkpoints/v3_2mic/best.pt` | `reports/results/cascade_decision.md` |

---

## 1. Top-level overview

| Path | Size | Files | Type | What it is |
|---|---:|---:|---|---|
| `SHARE_README.md` | 8 KB | 1 | DOC | Recipient guide for the team package: contents, integrity check, setup, licences, where to start (2026-10-05) |
| `SHARE_README_PUBLIC.md` | 7 KB | 1 | DOC | Recipient guide for the licence-safe public package: what was left out and why, checks, what can be run (2026-10-05) |
| `README.md` | 9 KB | 1 | DOC | Project overview, round-2 summary, repository layout, reproduce commands, data licences |
| `FILE_INDEX.md` | – | 1 | DOC | This file |
| `pyproject.toml` | 4 KB | 1 | SRC | Package definition (`danc`, source in `src/`, pytest settings) |
| `requirements.txt` | 4 KB | 1 | SRC | Pinned Python dependencies (`uv pip freeze` of `.venv`) |
| `src/` | 356 KB | 42 | SRC | The `danc` Python package: data pipeline, model, adaptive filters, training, inference, evaluation (+ the empty directory `src/x/`, §2) |
| `configs/` | 88 KB | 22 | SRC | YAML configurations of every training, fine-tuning, ablation and smoke run (14 in `configs/`, 6 in `configs/ablations/`, 2 in `configs/smoke/`) |
| `scripts/` | 304 KB | 27 | SRC | Helper scripts: tuning, benchmarks, statistics, tables, figures, manifests, HTML report, DFN3 reference, run queues |
| `tests/` | 68 KB | 9 | SRC | pytest suite (103 tests) |
| `deploy/` | 224 KB | 17 | SRC + DOC | Jetson AGX Orin deployment kit (16 files) and alternative-platform notes |
| `dist/` | 13 MB | 116 | SRC + MODEL + DATA (generated copy) | `dist/jetson_bundle/`: the Jetson run-time bundle built by `deploy/jetson/package_for_jetson.sh` (§6.1) |
| `checkpoints/` | 196 MB | 55 | MODEL | PyTorch checkpoints and the frozen config of every training run (21 run directories) |
| `exports/` | 17 MB | 23 | MODEL | Streaming ONNX models, their verification sidecars, deployment selection and `DEPLOYMENT.json` (current and round 1) |
| `data/` | 3.7 GB | 41 211 | DATA | Raw corpora, manifests and provenance, attribution (`ATTRIBUTION.md`), RIR bank, synthetic-noise examples, six fixed test sets |
| `reports/` | 967 MB | 10 946 | DOC + RES | `REPORT.md` (round 2), `REPORT_round1.md`, HTML version, figures, per-file results, decisions, enhanced audio, integrity manifest |
| `logs/` | 2.0 MB | 137 | LOG | stdout/stderr of every run, per-step training logs, queue timestamps, final test / self-check / packaging logs |
| `docs/` | 1.6 MB | 45 | DOC | Project brief, sharing checklist, literature research, claim verification, round-2 reviews and audits, reference paper, workflow records |
| `third_party/` | 64 MB | 1 045 | 3P | DeepFilterNet3 reference model and its isolated packages (external SOTA baseline only) |
| `.venv/` | 1.2 GB | 27 092 | ENV | Python 3.11.15 virtual environment made with `uv` 0.11.8. Rebuildable; see `docs/SHARING_CHECKLIST.md` |
| `.pytest_cache/` | 24 KB | 5 | ENV | pytest cache, rebuildable. Its `lastfailed` file still names `test_rotor_harmonic_peaks[drone_multirotor]`, a test id from 2026-10-03 that no longer exists since that test was re-parametrised by seed; the final run passed (§5) |
| `.DS_Store` (4 files) | small | 4 | ENV | macOS Finder metadata. Not project content: **not** in `reports/MANIFEST.sha256` and **not** shipped (§14) |
| `__pycache__/` dirs | small | 7 dirs | ENV | Python bytecode caches: `src/danc/`, `src/danc/data/`, `src/danc/inference/`, `tests/`, `scripts/`, `deploy/jetson/`, and one inside `third_party/dfn_pkgs`. Rebuildable |

---

## 2. Source code: `src/danc/` (SRC, written for this project)

Installed editable into `.venv` with `uv pip install -e .`. `src/danc.egg-info/` (4 files) is generated
packaging metadata from that install.

| File | Purpose |
|---|---|
| `dsp.py` | Shared STFT convention: 16 kHz, 20 ms sqrt-Hann window, 10 ms hop, 161 bins. Offline and streaming |
| `data/download.py` | Reproducible downloader; writes `data/raw/` and `data/manifests/provenance.json` (URLs, licences, SHA-256) |
| `data/speech.py` | LibriSpeech manifests, speaker-disjoint train/val/test (+ v2 train set); writes `data/manifests/speech_*.json` |
| `data/noise_bank.py` | Noise bank: NOISEX-92, ESC-50, drone recordings, synthetic generators; train/test splits |
| `data/synth_noise.py` | Physics-based gunfire, artillery, explosion, helicopter, drone, siren, wind, jet and tracked-vehicle generators |
| `data/rir.py` | Image-source RIR bank (pyroomacoustics); builds and caches `data/rir_bank/rirs_v2_*.npz` on first use |
| `data/mixer.py` | Dynamic mixing and augmentation (SNR, reverb, clipping, EQ, level, timbre) and the torch Dataset |
| `data/dual_mic.py` | Dual-microphone headset scene simulator (directional + diffuse noise, talker leakage) |
| `data/dual_mixer.py` | Dynamic dual-microphone training scenes for the two-microphone DANCNet **(round 2)** |
| `data/build_testset.py` | Renders `defence_v1`, `dualmic_v1` and `dualmic_val` into `data/testsets/` |
| `data/build_edgeset.py` | Renders the defence edge-case set `data/testsets/defence_edge_v1/` (11 cases) **(round 2)** |
| `data/build_dual_babble.py` | Renders the dual-microphone crew-babble set `data/testsets/dualmic_babble_v1/` **(round 2)** |
| `data/build_highsnr.py` | Renders the high-SNR extension `data/testsets/defence_highsnr_v1/` (20 and 25 dB) **(round 2)** |
| `models/dancnet.py` | DANCNet: causal sub-band/full-band complex-domain deep-filtering network, streaming with explicit state. Width, number of dual-path blocks and look-ahead are configurable (v3: width 96, 4 blocks, 609 k parameters); round 2 added `in_ch = 2` (primary + reference microphone as inputs) and `norm_type` (median normaliser, or EMA for the ablation) |
| `models/baselines.py` | Spectral subtraction, Wiener (decision-directed), log-MMSE baselines |
| `adaptive/nlms.py` | Time-domain NLMS and STFT-domain multi-tap sub-band NLMS (SPP-gated, impulse-robust) |
| `train/losses.py` | Compressed complex/magnitude MSE, multi-resolution STFT, SI-SDR, PMSQE, asymmetric and SPP losses |
| `train/train.py` | Training loop (MPS/CPU/CUDA), validation, EMA, warm start, resume. Round 2: `restore_ema` keeps the averaged weights on resume |
| `train/finetune_metric.py` | MetricGAN+-style PESQ-guided fine-tuning, anchored by the full loss |
| `train/train_dual.py` | Training of the two-microphone DANCNet: warm start from a single-mic model (reference-mic filters start at zero), validation on `dualmic_val`, resume with optimizer state **(round 2)** |
| `train/third_party/pmsqe.py`, `bark_matrix_16k.mat`, `LICENSE_asteroid_MIT.txt` | PMSQE loss vendored from Asteroid (MIT licence, 3P) |
| `inference/export_onnx.py` | Single-frame streaming ONNX export with numerical verification (writes `exports/*.onnx` + `*.json`); also two-microphone models |
| `inference/runners.py` | PyTorch and ONNX Runtime step runners with one interface, for single- and two-microphone models |
| `inference/engine.py` | Hybrid real-time engine: shared STFT, sub-band NLMS, DANCNet, limiter. Round 2: two-microphone network, NLMS → two-microphone cascade (`EngineConfig.lms_two_mic`), reference-health monitor with a single-mic network in hot standby (`fallback_runner`, `EngineConfig.ref_*`) |
| `inference/postfilter.py` | Optional SPP-driven residual-noise post-filter, off by default. Tuned on validation data only; **not adopted** (REPORT §8.9) **(round 2)** |
| `inference/realtime.py` | Live duplex audio app (primary + reference mic to headset/radio) and paced simulation. Round 2: default model `exports/dancnet_v3hq_cont.onnx`, options `--two-mic-cascade`, `--fallback-onnx` and the reference-health thresholds |
| `eval/metrics.py` | PESQ-WB/NB, STOI, ESTOI, SI-SDR, output SNR, segmental SNR |
| `eval/evaluate.py` | Runs every system on a test set; writes `reports/results/<set>/` and `reports/audio/enhanced_*`. Round 2 systems: `dnn2` (two-mic network), `hybrid2` (NLMS → two-mic network), `dnn2_refdead` / `hybrid2_refdead` (reference mic replaced by −80 dBFS noise) |

`src/x/` is an **empty directory** created by an automated helper on 2026-10-04 at 09:11. It is not part of
the package and was left in place under the project's no-delete rule (REPORT §10). The same helper created the
empty `dist/jetson_bundle/src/x/` at the same time. Zip archives do not carry empty directories, so neither appears in the shared packages.

## 3. Configurations: `configs/` (SRC)

Each training run copies its config to `checkpoints/<name>/config.yaml` and logs to `logs/<name>.out` and
`logs/train_<name>.jsonl`.

| Config | Run name | Status |
|---|---|---|
| `train_pilot.yaml` | pilot | Pipeline debug run, 164 k parameters, no synthetic noise. Completed |
| `train_main.yaml` | main | **Incident:** stopped at step 1 300 when memory pressure filled the disk with swap. No weights saved |
| `train_v2_hq.yaml` | v2_hq | **Incident:** restarted after 600 steps with a schedule sized to the measured step time (became `hq`) |
| `train_hq.yaml` | hq | v2 HQ model, 340 k parameters, 2-frame look-ahead, 18 000 steps, 5.7 h. Base of the round-1 HQ |
| `train_v2_ll.yaml` | ll | v2 LL model (no look-ahead), warm start from `hq`. **Deployed LL** (rounds 1 and 2) |
| `finetune_hq_metric.yaml` | hq_metric | PESQ-guided fine-tune of `hq`. Adopted in round 1 (round-1 deployed HQ); replaced in round 2 |
| `finetune_ll_metric.yaml` | ll_metric | PESQ-guided fine-tune of `ll`. Not adopted under the pre-registered rule |
| `smoke_ema.yaml` | smoke_ema | Smoke test of the v2 training path (6 steps, CPU). Not a real run |
| `smoke_metric.yaml` | smoke_metric | Smoke test of the fine-tuning path (4 steps, CPU). Not a real run |
| `train_v3_hq.yaml` | v3hq | DANCNet-v3 (width 96, 4 dual-path blocks, 609 k parameters, 2-frame look-ahead), 16 000 steps, 6.9 h on the M4 GPU. Adopted over the round-1 HQ (`v3_decision_hq.md`), then superseded by `v3hq_cont` **(round 2)** |
| `train_v3_hq_cont.yaml` | v3hq_cont | Warm restart of `v3hq` (EMA weights), peak LR 3e-4, new data seed, 10 000 steps, 5.05 h. **Deployed HQ** (`v3hq_cont_decision.md`) **(round 2)** |
| `train_v3_ll.yaml` | v3ll | DANCNet-v3 LL (no look-ahead), warm start from `v3hq`, 4 000 steps, 1.7 h. Not adopted under the rule (`v3_decision_ll.md`; left as an explicit owner decision) **(round 2)** |
| `train_v3_2mic.yaml` | v3_2mic | Two-microphone DANCNet-v3 (`danc.train.train_dual`), warm start from `v3hq`, 5 000 steps. Crashed at step ≈ 2 500 on a data-generator edge case and was resumed from step 2 000 (`--resume`). The network alone was not adopted (`two_mic_decision.md`); behind the gated NLMS it is the **deployed TWO_MIC** configuration (`cascade_decision.md`) **(round 2)** |
| `smoke_dual.yaml` | smoke_dual | Smoke test of the two-microphone training path (3 steps, CPU). Not a real run |
| `ablations/abl_{base,crm,ema,la2,la4,nopmsqe}.yaml` | abl_* | Ablation arms (v2 architecture, 2 500 steps each from scratch, identical data order, seed 101): reference arm, single-tap complex ratio mask, EMA normaliser, look-ahead 2 and 4, no PMSQE. All six completed; results in REPORT Table R13 **(round 2)** |
| `smoke/smoke_dual_resume.yaml`, `smoke/smoke_dual_resume_b.yaml` | smoke_dual_resume | Tiny CPU runs (4 steps, then a time-limited continuation with `--resume`) used by hand to test the resume / EMA-restore fixes. Both write `checkpoints/smoke_dual_resume/`. Test artefacts, not real runs (the unit test is `tests/test_train_resume.py`) **(round 2)** |

## 4. Scripts: `scripts/` (SRC)

| Script | Produces |
|---|---|
| `plot_synth_noise.py` | `data/synthetic_examples/*.wav` and `reports/figures/synth_noise_grid.png` |
| `run_deepfilternet3.py` | `reports/audio/enhanced_<set>_dfn3/` for `defence_v1`, `vbdemand` and (round 2) `defence_highsnr_v1` (DFN3 reference; run with `PYTHONPATH=third_party/dfn_pkgs`) |
| `tune_hybrid.py` | NLMS-to-DNN coupling grid: `reports/results/dualmic_v1/hybrid_tuning.json` (exploratory, test subset) and `reports/results/dualmic_val/hybrid_tuning*.json` (validation) |
| `tune_postfilter.py` | Post-filter tuning on validation data: `reports/results/postfilter_tuning/val_hq_metric.json` (log `logs/tune_postfilter_hq.out`). Outcome: not adopted |
| `eval_edge.py` | Case-aware evaluation on `data/testsets/defence_edge_v1/`: `reports/results/defence_edge_v1/per_file.csv` and `summary_by_case.csv`; with `--save_audio`, `reports/audio/enhanced_defence_edge_v1_<system>/` |
| `transparency_test.py` | `reports/results/transparency_clean_input.json` (clean speech in, compared with output; round-1 models plus `v3hq`, `v3ll`, `v3hq_cont`) |
| `benchmark_latency.py` | `reports/results/latency_mac_m4.json` and `reports/figures/latency_hist.png` (round 1); with `--tag round2` / `--tag round2_final`: `latency_mac_m4_round2*.json` and `latency_hist_round2*.png`, including two-mic, `+cascade` and `+fallback` rows |
| `rir_stats.py` | `reports/results/rir_stats.json` (RT60 statistics of the RIR bank) |
| `choose_batch.py` | Clean-GPU batch-size benchmark: `reports/results/batch_benchmark.json` and the `KEY=VALUE` lines read by `run_round2b_chain.sh` (log `logs/choose_batch.out`; batch 16 kept) **(round 2)** |
| `band_error_analysis.py` | `reports/results/band_error_analysis.csv`: per-band error of the round-1 HQ model, the diagnosis behind the v3 capacity increase **(round 2)** |
| `paired_stats.py` | `reports/results/paired_stats.json`, and on stdout the markdown table saved as `paired_stats.md`: paired bootstrap 95 % CIs and Wilcoxon tests (REPORT Table R15) **(round 2)** |
| `simulated_live_round2.py` | **(round 2)** Paced simulated-live runs of the three deployed configurations through both live launchers → `reports/results/simulated_live_runs_round2.json`, raw output in `logs/simulated_live_round2/` |
| `assemble_report_tables.py` | Copies the table blocks of `reports/results/tables.md` into `reports/REPORT.md` (`--report`), or checks that they are identical (`--check`). Used on 2026-10-05 from the session scratchpad; copied in after the pre-share review |
| `verify_install_jetsonlike.py` | Round-1 review harness: runs `deploy/jetson/verify_install.py` with every package that is not in `requirements-jetson.txt` blocked (→ `logs/verify_install_round1_review_jetsonlike.out`) |
| `build_share_packages.py` | Builds the two shared zips next to the project (§13): the team package and the licence-safe public package with its own manifest |
| `make_tables.py` | `reports/results/tables.md` (every results table, from the per-file CSVs; round 2 adds R11–R15). Earlier outputs are kept as `tables_interim.md`, `tables_round1.md` and `tables_round2_interim.md` |
| `make_figures.py` | `reports/figures/{defence_v1,dualmic_v1}_*.png` and `training_curves.png` (re-run in round 2 with the v3 and two-mic systems) |
| `make_figures_round2.py` | `reports/figures/round2_{pesq_attainment,edge_cases,ablations,training_curves}.png` **(round 2)** |
| `make_manifest.py` | `exports/DEPLOYMENT.json` (from `exports/deployment_selection.json`) and `reports/MANIFEST.sha256` |
| `build_report_html.py` | `reports/site/index.html` and `reports/site/figures/` (shareable HTML version of the report) |
| `run_finetune_chain.sh` | Runs the HQ and LL fine-tunes strictly one after the other (log `logs/chain.out`) |
| `run_round2_chain.sh` | First round-2 queue: waited for `v3hq`, then started `v3_2mic` (log `logs/chain2.out`). Stopped by the operator at 17:02 on 2026-10-04 to insert the batch benchmark; its remaining stages moved to `run_round2b_chain.sh` **(round 2)** |
| `run_round2b_chain.sh` | Waits for `v3_2mic`, then `choose_batch.py`, `v3ll`, the 6 ablation arms and the ablation evaluation on `defence_v1` (log `logs/chain2b.out`) **(round 2)** |
| `run_round2c_chain.sh` | Resumes `v3_2mic` from step 2 000, exports it and evaluates `dnn2:v3_2mic` on `dualmic_v1` (log `logs/chain2c.out`) **(round 2)** |
| `run_round2d_chain.sh` | Idle-machine latency benchmark (`--tag round2`), reference-failure evaluation, `v3hq_cont` training, export and full evaluation (defence, VoiceBank+DEMAND, transparency, dual-mic, edge cases) (log `logs/chain2d.out`) **(round 2)** |
| `eval_v3ll_when_ready.sh` | Waits for "v3ll finished" in `logs/chain2b.out`, then exports and evaluates `v3ll` (defence, dual-mic, VoiceBank+DEMAND, transparency, edge cases); marker `logs/eval_v3ll_chain.done` **(round 2)** |
| `eval_babble_when_ready.sh` | Evaluates the two-mic network and `v3hq_cont` on `dualmic_babble_v1` once `run_round2d_chain.sh` allows it (log `logs/eval_babble_chain.out`) **(round 2)** |

The `v3hq` evaluation (logs `eval_*_v3hq.out`, marker `logs/eval_v3hq_chain.done`), the cascade, crew-babble
single-mic and high-SNR evaluations and the final latency run were started by hand. Their commands are listed in
REPORT §10 ("Round 2, exact sequence"). The `v3hq` ONNX export has no log file.

## 5. Tests: `tests/` (SRC)

`test_dsp.py`, `test_model.py`, `test_engine.py`, `test_adaptive.py`, `test_synth_noise.py`,
`test_jetson_scripts.py`, and, from round 2, `test_two_mic.py` (two-microphone model, cascade, reference-health
fallback switching), `test_train_resume.py` (EMA restore on resume) and `test_deploy_kit.py` (Jetson kit:
scripts parse, ONNX contract of the deployed models, torch-free runner including two-mic models, safety wrapper).
Run with `python -m pytest -q`. Final run on 2026-10-05 07:28 in the project `.venv`: `103 passed, 36 warnings`
(`logs/pytest_round2_final.out`). The round-1 report recorded 68 tests.

## 6. Deployment: `deploy/` (SRC + DOC)

All Jetson files are **not verified on Jetson hardware in this project**; the torch-free parts were tested on
the Mac (`verify_install.py`: 115 PASS, 0 FAIL on 2026-10-05, `logs/verify_install_round2.out`).

| File | Purpose |
|---|---|
| `jetson/QUICKSTART.md` | Numbered copy-paste steps from a fresh Jetson AGX Orin 64 GB (JetPack 6.2.x) to live operation, for HQ, LL and the two-mic cascade, with expected Mac results and troubleshooting **(round 2 update)** |
| `jetson/README.md` | Jetson bring-up guide and design rationale; its header table lists every kit file and the round-2 deployed set |
| `jetson/build_trt_engine.sh` | Builds a TensorRT FP16 engine from an ONNX model (single-input models) |
| `jetson/trt_runner.py` | TensorRT 10 / ONNX Runtime step runner and benchmark for the Jetson |
| `jetson/setup_audio.sh` | ALSA, CPU governor, isolated core and SCHED_FIFO setup |
| `jetson/danc.service` | systemd unit for `python -m danc.inference.realtime` (needs PyTorch; use `danc-live.service` on a torch-free install) |
| `jetson/danc-live.service` | systemd unit that starts the torch-free launcher `danc_live.py` |
| `jetson/log_power.sh` | tegrastats power logging per nvpmodel mode |
| `jetson/measure_latency.py` | Electrical loop-back latency measurement |
| `jetson/danc_jetson_rt.py` | Torch-free runtime: ONNX Runtime runner for single- and two-microphone models, safety wrapper (NaN/Inf guard, bounded statistics), stall watchdog, CPU pinning |
| `jetson/danc_live.py` | Torch-free launcher for the live engine; wraps `danc.inference.realtime` with CPU pinning, SCHED_FIFO, ORT/TensorRT backend choice, audio-device pre-flight; accepts `--two-mic-cascade --fallback-onnx` |
| `jetson/check_channels.py` | Checks that input ch0 = primary (boom) mic and ch1 = reference mic, and the input levels, before going live (added 2026-10-04 09:04) |
| `jetson/verify_install.py` | Deployment self-check without PyTorch: environment, torch-free import, ONNX contract vs `exports/DEPLOYMENT.json`, quality vs the reference rows named in `DEPLOYMENT.json` (`reference_systems`), robustness (incl. reference-mic failure and fallback switch) and timing, for all three deployed configurations |
| `jetson/requirements-jetson.txt` | Runtime package list for the Jetson (no PyTorch) |
| `jetson/install.sh` | Idempotent runtime installer for JetPack 6.2.x: apt packages, venv, pip packages, no PyTorch |
| `jetson/package_for_jetson.sh` | Builds `dist/jetson_bundle/` (and, with `--tar`, `dist/jetson_bundle.tar.gz`, not built here) on the development machine. It never deletes: an existing bundle is updated in place |
| `ALTERNATIVE_PLATFORMS.md` | Notes on DSP and other SoC targets |

### 6.1 Jetson bundle: `dist/jetson_bundle/` (generated copy, 13 MB, 116 files)

Rebuilt on 2026-10-05 at 07:31 (+04) by `deploy/jetson/package_for_jetson.sh` (log
`logs/package_bundle_round2.out`). Everything in it is a copy of project files, except `BUNDLE_INFO.txt`,
`SHA256SUMS` and the two reduced reference CSVs.

| Path | Contents |
|---|---|
| `BUNDLE_INFO.txt` | Creation time, source path, "not verified on Jetson hardware" status, SHA-256 of the three deployed models |
| `SHA256SUMS` | SHA-256 of the other 115 bundle files (`sha256sum -c SHA256SUMS` on the Jetson; all OK on 2026-10-05 07:40) |
| `README.md`, `pyproject.toml`, `requirements.txt` | Copies of the project root files |
| `src/danc/` (38 files), `src/x/` (empty) | Copy of the package; `src/x/` is the empty directory described in §2 |
| `deploy/` (17 files) | Copy of `deploy/` |
| `exports/` (9 files) | `DEPLOYMENT.json`, the three deployed models with sidecars (`dancnet_v3hq_cont`, `dancnet_ll`, `dancnet_v3_2mic`) and the **round-1 HQ `dancnet_hqft.onnx` + `.json`**, left over from the round-1 build because the script never deletes. It is listed in `SHA256SUMS` but not used by `DEPLOYMENT.json` |
| `reports/REPORT.md` | Copy of the round-2 report |
| `reports/results/{defence_v1,dualmic_v1}/per_file.csv` | Reference rows only (16 and 24 rows) used by `verify_install.py` |
| `reports/results/latency_mac_m4.json`, `simulated_live_runs.json` | Copies of the round-1 Mac measurements |
| `data/testsets/{defence_v1,dualmic_v1}/` (42 files) | 16 verification scenes (8 single-mic clean/noisy pairs, 8 dual-mic clean/primary/reference triples) and `meta.csv` |

## 7. Models: `checkpoints/` and `exports/` (MODEL)

`checkpoints/<run>/` holds `config.yaml` (frozen run config) and, where the run saved weights, `best.pt`
(best validation score) and/or `last.pt` (final step).

| Directory | Contents | Produced by |
|---|---|---|
| `pilot/` | best.pt, last.pt (164 k parameters) | `configs/train_pilot.yaml` |
| `main/` | config.yaml only (incident run, no weights) | `configs/train_main.yaml` |
| `v2_hq/` | config.yaml only (restarted as `hq`) | `configs/train_v2_hq.yaml` |
| `hq/` | best.pt, last.pt | `configs/train_hq.yaml` |
| `ll/` | best.pt (**deployed LL**), last.pt | `configs/train_v2_ll.yaml` |
| `hq_metric/` | last.pt (round-1 deployed HQ) | `configs/finetune_hq_metric.yaml` |
| `ll_metric/` | last.pt (not adopted) | `configs/finetune_ll_metric.yaml` |
| `smoke_ema/`, `smoke_metric/` | smoke-test weights (not real models) | `configs/smoke_*.yaml` |
| `smoke_dual/` | best.pt, last.pt: smoke-test weights of the two-microphone path (not a real model) | `configs/smoke_dual.yaml` |
| `v3hq/` | best.pt (step 16 000), last.pt, 19 MB **(round 2)** | `configs/train_v3_hq.yaml` |
| `v3hq_cont/` | best.pt (**deployed HQ**, step 10 000), last.pt **(round 2)** | `configs/train_v3_hq_cont.yaml` |
| `v3ll/` | best.pt, last.pt (not adopted) **(round 2)** | `configs/train_v3_ll.yaml` |
| `v3_2mic/` | best.pt (**deployed TWO_MIC network**, step 5 000), last.pt **(round 2)** | `configs/train_v3_2mic.yaml` (resumed run) |
| `abl_{base,crm,ema,la2,la4,nopmsqe}/` | best.pt, last.pt per ablation arm (short-budget comparison models, not for deployment) **(round 2)** | `configs/ablations/abl_*.yaml` |
| `smoke_dual_resume/` | config.yaml, last.pt: test artefact of the resume fix (not a real model) **(round 2)** | `configs/smoke/smoke_dual_resume*.yaml` |

`exports/` (written by `python -m danc.inference.export_onnx --ckpt ... --out ...`; each `.json` sidecar
holds the I/O shapes, ONNX-vs-PyTorch max error and ORT timing):

| File | Model |
|---|---|
| `dancnet_v3hq_cont.onnx` + `.json` | **Deployed HQ** (40 ms), from `checkpoints/v3hq_cont/best.pt` **(round 2)** |
| `dancnet_ll.onnx` + `.json` | **Deployed LL** (20 ms), from `checkpoints/ll/best.pt` |
| `dancnet_v3_2mic.onnx` + `.json` | **Deployed TWO_MIC network** (40 ms, inputs: primary and reference spectra), from `checkpoints/v3_2mic/best.pt`; run behind the gated NLMS (`--two-mic-cascade`) **(round 2)** |
| `dancnet_v3hq.onnx` + `.json` | v3 HQ before the continuation (adopted, then superseded by `v3hq_cont`), from `checkpoints/v3hq/best.pt` **(round 2)** |
| `dancnet_v3ll.onnx` + `.json` | v3 LL (not adopted; owner decision, `v3_decision_ll.md`), from `checkpoints/v3ll/best.pt` **(round 2)** |
| `dancnet_hqft.onnx` + `.json` | Round-1 deployed HQ (40 ms), from `checkpoints/hq_metric/last.pt` |
| `dancnet_hq.onnx` + `.json` | Base HQ before fine-tuning, from `checkpoints/hq/best.pt` |
| `dancnet_pilot.onnx` | Pilot / LL-small architecture (used in the latency benchmark) |
| `dancnet_untrained_test.onnx` + `.json` | Export test with untrained weights (not a real model) |
| `smoke_ema.onnx` + `.json` | Smoke-test export (not a real model) |
| `deployment_selection.json` | **The single place where the deployed set is chosen:** HQ, LL and TWO_MIC with ONNX file, checkpoint, mode, decision file, reference systems, engine flags, fallback and launch command. Read by `scripts/make_manifest.py` **(round 2)** |
| `deployment_selection_round1.json` | Round-1 selection (HQ = `dancnet_hqft.onnx`, LL = `dancnet_ll.onnx`), kept **(round 2)** |
| `DEPLOYMENT.json` | Deployed models with ONNX metadata, I/O contract, SHA-256 of ONNX and checkpoint, engine defaults. Regenerated on 2026-10-05 07:27 from `deployment_selection.json` (`scripts/make_manifest.py`) |
| `DEPLOYMENT_round1.json` | Round-1 `DEPLOYMENT.json` (HQ = `dancnet_hqft.onnx`), kept **(round 2)** |

## 8. Data: `data/` (DATA, 3.7 GB)

| Path | Size | Files | Contents | Produced by | Licence |
|---|---:|---:|---|---|---|
| `raw/LibriSpeech/` | 2.0 GB | 11 678 | dev-clean (2 703 FLAC, 40 speakers), test-clean (2 620 FLAC, 40 speakers), train-clean-100 **partial** (4 525 FLAC, 42 speakers, ~1.0 GB), Mini LibriSpeech train-clean-5 (1 519 FLAC, 28 speakers), plus the corpus TXT files | `danc.data.download` (and `--only librispeech_partial`) | CC BY 4.0 |
| `raw/noisex92/` | 245 MB | 30 | 15 NOISEX-92 recordings: `mat/` (original 19.98 kHz) and `wav16k/` (resampled) | `danc.data.download` | © TNO 1990, redistribution terms **unclear** |
| `raw/esc50/` | 100 MB | 641 | 640 clips from 16 defence-relevant ESC-50 classes at 16 kHz + `esc50.csv` | `danc.data.download` | CC BY-NC 3.0, **non-commercial** |
| `raw/drone/` | 803 MB | 23 410 | DroneAudioDataset (Al-Emadi et al.), binary and multiclass sets + `LICENSE_NOTICE.txt` | `danc.data.download` | **No licence**, internal R&D only, do not redistribute |
| `raw/vbdemand_test/` | 130 MB | 1 648 | VoiceBank+DEMAND test set, 824 clean/noisy pairs, 16 kHz mirror | `danc.data.download` | CC BY 4.0 |
| `manifests/provenance.json` | 72 KB | 1 | Source URLs, licences, SHA-256 hashes, partial-download note for every corpus | `danc.data.download` | – |
| `manifests/speech_{train,val,test}.json`, `speech_train_v2.json` | 1.6 MB | 4 | Speaker-disjoint speech splits; v2 = larger training set (97 speakers, 25 h) | `python -m danc.data.speech` | – |
| `rir_bank/rirs_v2_{train,test}.npz` | 18 MB | 2 | Image-source RIR bank in use (near-field talker and far-field noise) | `danc.data.rir.build_bank` (on first use) | generated |
| `rir_bank/rirs_{train,test}.npz` | 9 MB | 2 | Earlier RIR bank version, superseded by v2, kept | same, earlier version | generated |
| `synthetic_examples/*.wav` | 2.4 MB | 13 | One example per synthetic defence-noise generator | `scripts/plot_synth_noise.py` | generated |
| `testsets/defence_v1/` | 189 MB | 1 802 | Single-mic test set: 900 clean/noisy pairs (4 categories × SNR −5…15 dB × 40, + 100 at −10 dB), `meta.csv`, `README.txt` | `python -m danc.data.build_testset` | derived from the corpora above |
| `testsets/dualmic_v1/` | 79 MB | 721 | Dual-mic test scenes: 240 × (clean, primary, reference) + `meta.csv` | same | derived |
| `testsets/dualmic_val/` | 16 MB | 145 | Dual-mic **validation** scenes (48, validation speakers, training-split noise) for tuning, two-mic training validation and the exploratory cascade check | same | derived |
| `testsets/defence_edge_v1/` | 64 MB | 610 | Edge-case set: 304 clean/noisy pairs in 11 cases (30 files each: heavy clipping, very low and very high level, cabin reverb, radio band, noise only, post-blast, noise switching, 400 Hz hum, crew babble; plus 4 streams of 60 s), `meta.csv`, `README.txt` **(round 2)** | `python -m danc.data.build_edgeset` | derived |
| `testsets/defence_highsnr_v1/` | 32 MB | 322 | High-SNR extension of `defence_v1`: 160 clean/noisy pairs (4 categories × 20 and 25 dB × 20), built like `defence_v1` with a separate seed, `meta.csv`, `README.txt` **(round 2)** | `python -m danc.data.build_highsnr` | derived |
| `testsets/dualmic_babble_v1/` | 20 MB | 181 | Dual-mic crew-babble scenes: 60 × (clean, primary, reference), wearer + 2–3 other test speakers at TIR 0 / 5 / 10 dB (20 each) + diffuse vehicle bed, `meta.csv` (no README; the scene recipe is in the docstring of `build_dual_babble.py`; the `snr_db` column holds the TIR) **(round 2)** | `python -m danc.data.build_dual_babble` | derived |

Licence details for every source are in `data/ATTRIBUTION.md` (creator, source and licence per corpus, added 2026-10-05), `data/manifests/provenance.json` and `README.md`. The test sets are
derived from the licensed corpora, so the same caveats apply to them. See `docs/SHARING_CHECKLIST.md` before
sharing outside the project.

## 9. Reports and results: `reports/` (DOC + RES)

| Path | Type | Contents | Produced by |
|---|---|---|---|
| `REPORT.md` | DOC | Full technical report, **round-2 version** (2026-10-05): design, evidence, results (§8.6–8.10 are round 2), risks, reproducibility (both rounds' exact command sequences, incidents), references | written by hand from the results; tables from `make_tables.py` |
| `REPORT_round1.md` | DOC | The round-1 report as it stood before round 2, preserved unchanged **(round 2)** | copy of the round-1 `REPORT.md` |
| `site/index.html` + `site/figures/` (12 PNG) | DOC | Shareable HTML version of the report, **round-2 build** (2026-10-05; published privately). Two PNGs (`*_pass_rates.png`) are round-1 leftovers not referenced by the page. Loads Google Fonts when online, falls back to system fonts offline | `scripts/build_report_html.py` |
| `figures/*.png` (14) | RES | Round-1 set (metrics vs SNR, pass rates, example spectrograms, latency histogram, training curves, synthetic-noise grid; the `defence_v1_*`, `dualmic_v1_*` and `training_curves` figures were regenerated in round 2), `latency_hist_round2.png`, `latency_hist_round2_final.png`, `round2_pesq_attainment.png`, `round2_edge_cases.png`, `round2_ablations.png`, `round2_training_curves.png` | `scripts/make_figures.py`, `make_figures_round2.py`, `benchmark_latency.py`, `plot_synth_noise.py` |
| `figures/round1/` (10 PNG + `README.md`) | RES | Copies of the round-1 figures taken on 2026-10-05 before regeneration, so `REPORT_round1.md` can be read against them. A wildcard copy also put the two round-2 latency histograms there; the folder README says so **(round 2)** | copied by hand |
| `results/defence_v1/` | RES | `per_file.csv`: 18 systems × 900 files (round 2 added `dancnet:v3hq`, `v3ll`, `v3hq_cont` and the 6 ablation arms); `summary.json`, `summary_by_condition.csv` | `python -m danc.eval.evaluate` |
| `results/dualmic_v1/` | RES | `per_file.csv`: 23 systems × 240 scenes (round 2 added `dnn`/`hybrid` with `v3hq`, `v3ll`, `v3hq_cont`, and `dnn2`, `hybrid2`, `dnn2_refdead`, `hybrid2_refdead` with `v3_2mic`); `summary.json`, `summary_by_condition.csv`; `hybrid_tuning.json` (exploratory coupling sweep on a test subset, REPORT §8) | `danc.eval.evaluate`, `scripts/tune_hybrid.py` |
| `results/vbdemand/` | RES | `per_file.csv`: 10 systems × 824 files; `summary.json`, `summary_by_condition.csv` | `danc.eval.evaluate` |
| `results/defence_edge_v1/` | RES | `per_file.csv` (10 systems × 304 files) and `summary_by_case.csv` (no `summary.json`) **(round 2)** | `scripts/eval_edge.py` |
| `results/defence_highsnr_v1/` | RES | `per_file.csv` (7 systems × 160), `summary.json`, `summary_by_condition.csv` **(round 2)** | `danc.eval.evaluate` |
| `results/dualmic_babble_v1/` | RES | `per_file.csv` (10 systems × 60), `summary.json`, `summary_by_condition.csv` **(round 2)** | `danc.eval.evaluate` |
| `results/dualmic_val/` | RES | `hybrid_tuning.json`, `hybrid_tuning_with_gate_power.json` (coupling selection, round 1); `per_file.csv`, `summary.json`, `summary_by_condition.csv` (3 systems × 48 validation scenes: the exploratory cascade check, round 2) | `scripts/tune_hybrid.py dualmic_val [--with-gate-power]`, `danc.eval.evaluate` |
| `results/<set>/.per_file.lock` (7) | ENV | Empty lock files used by the evaluator against write races | same |
| `results/postfilter_tuning/val_hq_metric.json` | RES | Post-filter parameter search on validation data for the round-1 HQ (not adopted) | `scripts/tune_postfilter.py` |
| `results/finetune_decision_rule.md`, `finetune_decision_hq.md`, `finetune_decision_ll.md` | RES | Round-1 pre-registered adoption rule for fine-tunes and its application to HQ and LL | written from the results |
| `results/v3_decision_rule.md`, `v3_decision_hq.md`, `v3_decision_ll.md` | RES | Rule for DANCNet-v3 (written before v3 training) and its application: v3 HQ adopted, v3 LL not adopted **(round 2)** | written by hand |
| `results/v3hq_cont_decision_rule.md`, `v3hq_cont_decision.md` | RES | Rule for the v3 HQ warm restart and its application: adopted, deployed HQ **(round 2)** | written by hand |
| `results/two_mic_decision_rule.md`, `two_mic_decision.md` | RES | Rule for the two-mic network alone and its application: not adopted alone **(round 2)** | written by hand |
| `results/cascade_decision_rule.md`, `cascade_decision.md` | RES | Rule for the NLMS → two-mic cascade and its application: adopted with the reference-health fallback (TWO_MIC) **(round 2)** | written by hand |
| `results/tables.md` | RES | All result tables of the round-2 report (R1–R15) | `scripts/make_tables.py` |
| `results/tables_interim.md` | RES | Earlier round-1 snapshot (before the LL fine-tune results), kept | `scripts/make_tables.py` |
| `results/tables_round1.md` | RES | Final round-1 tables, kept to go with `REPORT_round1.md` **(round 2)** | `scripts/make_tables.py` |
| `results/tables_round2_interim.md` | RES | Interim snapshot from 2026-10-04 09:32 (round-1 models plus the first edge-case and PESQ-attainment tables, no v3 rows), kept **(round 2)** | `scripts/make_tables.py` |
| `results/paired_stats.json`, `paired_stats.md` | RES | Paired differences with 95 % bootstrap CIs and Wilcoxon p-values for 27 key comparisons (REPORT Table R15) **(round 2)** | `scripts/paired_stats.py` |
| `results/band_error_analysis.csv` | RES | Per-band SNR and log-spectral distance of the round-1 HQ on 160 `defence_v1` files **(round 2)** | `scripts/band_error_analysis.py` |
| `results/batch_benchmark.json` | RES | MPS batch-size benchmark and the choice (batch 16 kept) **(round 2)** | `scripts/choose_batch.py` |
| `results/latency_mac_m4.json` | RES | Round-1 per-hop latency statistics on the Apple M4 | `scripts/benchmark_latency.py` |
| `results/latency_mac_m4_round2.json`, `latency_mac_m4_round2_final.json` | RES | Round-2 latency runs on the idle M4 (all models; `_final` adds `v3hq_cont` and the shipped cascade + fallback configuration) **(round 2)** | `scripts/benchmark_latency.py --tag round2[_final]` |
| `results/simulated_live_runs.json` | RES | Paced real-time simulation of four dual-mic scenes (round-1 models) | `python -m danc.inference.realtime --simulate ... --paced` |
| `results/simulated_live_runs_round2.json` | RES | **(round 2)** The same four scenes, paced, for HQ, LL and TWO_MIC via `deploy/jetson/danc_live.py`, and TWO_MIC also via `danc.inference.realtime` (16 runs); TWO_MIC had 2 late blocks in 4 720 | `scripts/simulated_live_round2.py` |
| `results/transparency_clean_input.json` | RES | Clean-speech transparency test (round-1 models and, from round 2, `v3hq`, `v3ll`, `v3hq_cont`) | `scripts/transparency_test.py` |
| `results/rir_stats.json` | RES | RT60 statistics of the RIR bank | `scripts/rir_stats.py` |
| `results/verify_install_mac_round2.json` | RES | Full JSON report of the deployment self-check on the Mac (115 PASS, 0 FAIL, 10 INFO) **(round 2)** | `deploy/jetson/verify_install.py --json` |
| `results/verify_install_mac_round1.json`, `results/verify_install_round1_review_jetsonlike.json` | RES | Round-1 self-check reports quoted in `deploy/jetson/QUICKSTART.md` step 5 (83 PASS, 0 FAIL, 1 WARN in 73 s; the Jetson-like re-run in 77 s). Copied in from the session scratchpad on 2026-10-05 | `deploy/jetson/verify_install.py`, `scripts/verify_install_jetsonlike.py` |
| `audio/enhanced_defence_v1_<system>/` (8 dirs × 900 files) | RES | `dancnet_hq`, `dancnet_hq_metric` (round-1 HQ), `dancnet_ll` (deployed LL), `dancnet_ll_metric`, `dfn3`, and from round 2 `dancnet_v3hq`, `dancnet_v3ll`, `dancnet_v3hq_cont` (deployed HQ) | `danc.eval.evaluate` (`--save_audio`) and `run_deepfilternet3.py` |
| `audio/enhanced_dualmic_v1_<system>/` (9 dirs × 240 files) | RES | `dnn_hq`, `hybrid_hq`, `hybrid_hqft`, `hybrid_ll`, and from round 2 `hybrid_v3hq`, `hybrid_v3ll`, `hybrid_v3hq_cont`, `dnn2_v3_2mic`, `hybrid2_v3_2mic` (deployed TWO_MIC cascade, without the fallback) | `danc.eval.evaluate` |
| `audio/enhanced_dualmic_babble_v1_<system>/` (3 dirs × 60 files) | RES | `hybrid_v3hq`, `dnn2_v3_2mic`, `hybrid2_v3_2mic` **(round 2)** | `danc.eval.evaluate` |
| `audio/enhanced_defence_edge_v1_dancnet_hq_metric/` (304 files) | RES | Round-1 HQ on the edge-case set (written by `eval_edge_v2.out` on 2026-10-04; not listed in the previous index) | `scripts/eval_edge.py --save_audio` |
| `audio/enhanced_defence_highsnr_v1_dfn3/` (160 files) | RES | DeepFilterNet3 on the high-SNR set **(round 2)** | `scripts/run_deepfilternet3.py --set defence_highsnr_v1` |
| `audio/enhanced_vbdemand_dfn3/` (824 files) | RES | DeepFilterNet3 on VoiceBank+DEMAND | `scripts/run_deepfilternet3.py --set vbdemand` |
| `audio/demo/` (16 WAV) | RES | Listening set: 4 scenes × (clean reference, noisy primary, HQ hybrid, LL hybrid), made with the round-1 models | paced simulated-live runs (REPORT §8.5) |
| `MANIFEST.sha256` | RES | SHA-256 of every shipped project file except `third_party/dfn_pkgs`, the `*.lock` files and itself (§14) | `scripts/make_manifest.py` |

`reports/audio/` is 936 MB (24 directories, 10 844 files) of the 966 MB; `results/` is 9.5 MB (58 files),
`figures/` 9.4 MB (25 files), `site/` 4.9 MB (13 files).

## 10. Documentation: `docs/` (DOC)

| Path | Contents | Produced by |
|---|---|---|
| `PROJECT_BRIEF.md` | The owner's project brief and every follow-up request, verbatim, including the round-2 requests (PESQ 3.25–3.5, ablations, Jetson deployment, complete shared folder; progress and GPU-use requests) | copied from the owner's messages |
| `SHARING_CHECKLIST.md` | What to share, packaging, how to rebuild the venv, verification steps, licence caveats, final checklist (updated for round 2) | written for sharing |
| `references/Narain_Kant_Singh_DSJ_2026_21095.pdf` | Reference paper the owner supplied (Narain, Kant, Singh, *Def. Sci. J.* 76(3), 2026; report ref. 44). Byte-identical copy of the owner's `21095.pdf` | copied from the owner's Downloads folder |
| `research/T1_architectures.md` … `T7_defence_prior_work.md` | Literature research notes with sources, one per topic (architectures, datasets, impulsive noise, hybrid adaptive filtering, edge deployment, losses/metrics, defence prior work) | research workflow `anc-sota-research` (run `wf_79848172-d66`) |
| `research/claims_verification.md`, `.json` | Adversarial verification of every numeric, licensing and date claim cited in the round-1 report | workflow `verify-report-claims` (run `wf_5597f002-1ea`) |
| `research/report_audit.json` | Read-only audit of every number in the round-1 `REPORT.md` against raw results, logs, code and configs | workflow `audit-report-consistency` (run `wf_c64771c1-4db`) |
| `research/round2_code_review.md`, `research/round2_review_repro/` | Final report of the independent round-2 code review (2026-10-04 23:14–23:28 +04) and its two reproducers (`ema_resume.py`, `refhealth.py`) | exported from the session on 2026-10-05 |
| `research/report_audit_round2.md` | Final report of the adversarial audit of the round-2 `REPORT.md` (2026-10-05); its findings were corrected in the report | exported from the session |
| `research/file_index_review_round2.md` | Hand-back of the agent that refreshed this index and the sharing checklist on 2026-10-05 | exported from the session |
| `research/pre_share_review_2026-10-05.md` | Findings of the four-reviewer pre-share review (secrets, licences, recipient walkthrough, outside material) | workflow `wf_c549189e-2c8` |
| `workflows/*.js` (8) | The multi-agent workflow scripts that orchestrated research, module building, verification, audits and the round-1 packaging step; the two `*-wf_79848172-d66.js` / `*-wf_d657bd8c-717.js` files are the exact scripts of two resumed runs, extracted from their state files (2026-10-05) | copied from the Claude Code session |
| `workflows/state/wf_*.json` (8) + `README.md` | Final state of each workflow run, including the complete final state of `wf_14dd115a-9e9`, whose journal copy is partial | copied from the Claude Code session on 2026-10-05 |
| `workflows/journals/<run-id>.jsonl` (8, + `wf_14dd115a-9e9.final.jsonl`) | Event journals of each workflow run (agent start/finish records); `wf_14dd115a-9e9.jsonl` is a partial copy taken while it ran, `.final.jsonl` the complete one | copied from the Claude Code session |

Workflow runs and their scripts (no workflow was run in round 2; round-2 work was done directly and by the
queue scripts in §4):

| Run id | Script | Notes |
|---|---|---|
| `wf_d812ebf9-67c` | `anc-sota-research-*.js` | First research run; the journal records the agents as failed |
| `wf_79848172-d66` | same script | Re-run; the `docs/research/T*.md` notes date from this run |
| `wf_33c61465-a05` | `danc-parallel-modules-*.js` | First module-build run; the journal records the agents as failed |
| `wf_d657bd8c-717` | same script | Re-run of the module build (synthetic noise, baselines, adaptive/dual-mic, Jetson kit) |
| `wf_5597f002-1ea` | `verify-report-claims-*.js` | Claim verification |
| `wf_c64771c1-4db` | `audit-report-consistency-*.js` | Report audit |
| `wf_14dd115a-9e9` | `danc-deploy-docs-and-completeness-*.js` | Deployment documentation and completeness packaging (round 1). The run ended at about 09:30 on 2026-10-04, but the journal copy in the folder is still the **partial** one from 08:54 (5 040 bytes); copy it again from the session folder (`docs/SHARING_CHECKLIST.md` §5) |

## 11. Logs: `logs/` (LOG, 137 files: 117 at the top level, 17 in `logs/simulated_live_round2/`, 3 in `logs/packager_incident_2026-10-04/`)

Copied in on 2026-10-05 after the pre-share review: `verify_install_round1.out` and
`verify_install_round1_review_jetsonlike.out` (the round-1 self-check runs quoted in QUICKSTART step 5), and
`packager_incident_2026-10-04/` (`tree_before.txt`, `tree_after.txt`, `src_deploy_sums.txt`: the evidence that no
file was deleted when the packager briefly ran with the project root as its output on 2026-10-04).
`package_bundle_round2_share.out` is the final Jetson-bundle rebuild before packaging; `pytest_round2_final.out` was
rewritten by the last test run before packaging (same result, 103 passed).

`logs/simulated_live_round2/` **(round 2)**: one raw output file per paced simulated-live run (`<launcher>_<config>_<scene>.out`, 16 files) and `repeat_two_mic_paced.txt` (12 repeated TWO_MIC runs, 0 late blocks); produced by `scripts/simulated_live_round2.py` and the repeat command recorded at the top of that file.

Training runs (`<name>.out` = console log, `train_<name>.jsonl` = one JSON record per logged step and per
validation):

| Run | Console log | Step log | Outcome |
|---|---|---|---|
| pilot | `pilot.out` | `train_pilot.jsonl` | Completed (pipeline debug) |
| main | `main.out` | `train_main.jsonl` | **Incident:** stopped at step 1 300; the log ends with macOS "out of space" errors (swap filled the disk) |
| v2_hq | `v2_hq.out` | `train_v2_hq.jsonl` | **Incident:** stopped after 600 steps and restarted as `hq` with a re-sized schedule |
| smoke_ema | – | `train_smoke_ema.jsonl` | Smoke test |
| smoke_metric | – | `train_smoke_metric.jsonl` | Smoke test |
| hq | `hq.out` | `train_hq.jsonl` | Completed, 18 000 steps, 5.7 h |
| ll | `ll.out` | `train_ll.jsonl` | Completed, 5 000 steps (warm start from hq) |
| hq_metric | `hq_metric.out` | `train_hq_metric.jsonl` | Completed, 2 000 steps. Adopted in round 1 |
| hq_metric (aborted) | `hq_metric_ABORTED_concurrent_attempt.out` | – | **Incident:** an HQ fine-tune attempt started while LL training was still running. Stopped early (the log ends just after the initial validation and the first training-step warning); the fine-tunes were then run one at a time by `scripts/run_finetune_chain.sh` |
| ll_metric | `ll_metric.out` | `train_ll_metric.jsonl` | Completed, 2 000 steps. Not adopted |
| fine-tune chain | `chain.out` | – | Timestamps of the sequential chain (LL training end, HQ fine-tune end, LL fine-tune end) |
| smoke_dual | – | `train_smoke_dual.jsonl` | Smoke test of two-microphone training |
| v3hq | `v3hq.out` | `train_v3hq.jsonl` | Completed, 16 000 steps, 6.9 h (609 156 parameters) **(round 2)** |
| v3_2mic | `v3_2mic.out`, `v3_2mic_resume.out` | `train_v3_2mic.jsonl` | **Incident:** the first run stopped at step ≈ 2 500 with `ValueError: all noise signals have zero energy` (end of `v3_2mic.out`). After the data-generator fix, `--resume` continued from the step-2 000 checkpoint to step 5 000 (`v3_2mic_resume.out`). The step log holds both runs: steps 2 100–2 500 appear twice **(round 2)** |
| v3ll | `v3ll.out` | `train_v3ll.jsonl` | Completed, 4 000 steps, 1.7 h. Not adopted **(round 2)** |
| abl_base, abl_ema, abl_crm, abl_la2, abl_la4, abl_nopmsqe | `abl_<arm>.out` | `train_abl_<arm>.jsonl` | Completed, 2 500 steps each, 0.8–1.0 h each **(round 2)** |
| smoke_dual_resume | – | `train_smoke_dual_resume.jsonl` | Smoke test of the resume fix (both smoke configs) **(round 2)** |
| v3hq_cont | `v3hq_cont.out` | `train_v3hq_cont.jsonl` | Completed, 10 000 steps, 5.05 h. Adopted (deployed HQ) **(round 2)** |

Round-2 queue logs (timestamps written by the queue scripts in §4):

| Log | Content |
|---|---|
| `chain2.out` | `run_round2_chain.sh`: v3hq end (15:38), then the operator note that the queue was stopped at 17:02 and its stages moved to `run_round2b_chain.sh` |
| `chain2b.out` | `run_round2b_chain.sh`: v3_2mic, batch benchmark, v3ll, each ablation arm, ablation evaluation (ends 00:16 on 2026-10-05). Its first line "v3_2mic finished" only means the process had ended: that run had crashed (see `v3_2mic.out`) |
| `chain2c.out` | `run_round2c_chain.sh`: v3_2mic resume end, two-mic evaluation |
| `chain2d.out` | `run_round2d_chain.sh`: latency benchmark, reference-failure evaluation, v3hq_cont end, "all finished" at 07:12 on 2026-10-05 |
| `choose_batch.out` | `scripts/choose_batch.py` output (step time and memory at batch 16/24/32; chosen configs) |
| `eval_v3hq_chain.done` | Marker "ALLDONE" of the hand-started v3hq evaluation sequence |
| `eval_v3ll_chain.done`, `eval_v3ll_chain.out` | Marker of `eval_v3ll_when_ready.sh` and its (empty) console output; every step wrote its own log |
| `eval_babble_chain.out` | `eval_babble_when_ready.sh` timestamps |

Data, evaluation and tuning logs, round 1:

| Log | Command / content |
|---|---|
| `download.log` | `python -m danc.data.download --root data/raw` (NOISEX-92, ESC-50, VoiceBank+DEMAND, drone, LibriSpeech dev/test) |
| `download_partial.log` | `--only librispeech_partial`; ends with `OSError: [Errno 28] No space left on device` at about 1.0 GB extracted (see the note in `provenance.json`) |
| `eval_defence_baselines.out` | `evaluate --set defence_v1` for noisy, specsub, wiener, logmmse |
| `eval_defence_hq.out`, `eval_defence_hq_metric.out`, `eval_defence_ll.out`, `eval_defence_ll_metric.out` | `evaluate --set defence_v1` for each round-1 DANCNet model |
| `eval_defence_hq_metric_audio.out` | Re-run of the round-1 HQ on defence_v1 that also saved the enhanced audio |
| `eval_dualmic_hq.out` | `evaluate --set dualmic_v1`: primary, logmmse, nlms, dnn:hq, hybrid:hq, hybrid_nogate:hq (untuned coupling) |
| `eval_dualmic_hq_tuned.out`, `eval_dualmic_hq_tuned2.out` | hybrid:hq and hybrid_nogate:hq re-run after coupling tuning |
| `eval_dualmic_hqft.out`, `eval_dualmic_ll.out` | dnn and hybrid systems for the round-1 HQ and LL |
| (note on `dualmic_v1`) | The untuned-coupling rows from `eval_dualmic_hq.out` (`nlms`, `hybrid:hq`, `hybrid_nogate:hq`, μ = 0.3) appear in `reports/results/dualmic_v1/per_file.csv` as `nlms_mu0.3`, `hybrid_mu0.3:hq` and `hybrid_nogate_mu0.3:hq` (relabelled before `eval_dualmic_hq_tuned.out`). The `nlms` row with the tuned μ = 0.1 first appears in the summary of `eval_dualmic_hqft.out`, which has no `[eval] ... nlms` line. Neither the relabelling nor that `nlms` re-run is recorded in a log file; the values are in `per_file.csv` |
| `eval_vbd_baselines.out`, `eval_vbd.out`, `eval_vbd_hq_metric.out`, `eval_vbd_ll.out` | `evaluate --set vbdemand` for baselines, DFN3 and each round-1 DANCNet model |
| `dfn3_defence.out`, `dfn3_vbd.out` | `scripts/run_deepfilternet3.py` on defence_v1 and VoiceBank+DEMAND (the "not a git repository" line is harmless DFN start-up output) |
| `tune_hybrid_on_test_subset_EXPLORATORY.out` | Exploratory coupling sweep on 48 test scenes (REPORT §8, documented as exploratory) |
| `tune_hybrid.out` | Identical copy of the exploratory sweep log (its original name), kept |
| `tune_hybrid_val.out`, `tune_hybrid_val_gatepower.out` | Coupling selection on the validation scenes |
| `tune_postfilter_hq.out` | Post-filter tuning on validation data (not adopted) |
| `eval_edge_v2.out` | `scripts/eval_edge.py --systems noisy,logmmse,dancnet:hq_metric,dancnet:ll --save_audio dancnet:hq_metric` on the edge-case set (completed 09:03 on 2026-10-04) |
| `resources.log` | Free memory and swap sampled during the long HQ training run (memory-safety monitor) |

Export, evaluation, latency and check logs, round 2:

| Log | Command / content |
|---|---|
| `export_v3ll.out`, `export_v3_2mic.out`, `export_v3hq_cont.out` | `danc.inference.export_onnx` for each round-2 model (TorchScript-exporter warnings, then the verification JSON). There is no export log for `v3hq` |
| `eval_defence_v3hq.out`, `eval_defence_v3ll.out`, `eval_defence_v3hq_cont.out` | `evaluate --set defence_v1` for each v3 model (with saved audio) |
| `eval_dualmic_v3hq.out`, `eval_dualmic_v3ll.out`, `eval_dualmic_v3hq_cont.out` | `evaluate --set dualmic_v1`, `dnn:` and `hybrid:` with each v3 model |
| `eval_vbd_v3hq.out`, `eval_vbd_v3ll.out`, `eval_vbd_v3hq_cont.out` | `evaluate --set vbdemand` for each v3 model |
| `transparency_v3hq.out`, `transparency_v3ll.out`, `transparency_v3hq_cont.out` | `scripts/transparency_test.py --runs <model>` |
| `eval_edge_v3hq.out`, `eval_edge_v3ll.out`, `eval_edge_v3hq_cont.out` | `scripts/eval_edge.py` for each v3 model (each log ends with the case table of every system evaluated so far) |
| `eval_ablations.out` | `evaluate --set defence_v1` for the 6 ablation arms (no audio) |
| `eval_edge_ablations.out` | `scripts/eval_edge.py --systems dancnet:abl_base,dancnet:abl_ema,dancnet:abl_crm` (normaliser and output-stage ablation on the edge cases) |
| `eval_dualmic_v3_2mic.out` | `evaluate --set dualmic_v1 --systems dnn2:v3_2mic` (two-mic network alone) |
| `eval_dualmic_refdead.out` | `dnn2_refdead:v3_2mic` (reference-mic failure, criterion 3 of `two_mic_decision_rule.md`) |
| `eval_dualmic_val_cascade_EXPLORATORY.out` | Cascade check on the 48 **validation** scenes (`hybrid:v3hq`, `dnn2:v3_2mic`, `hybrid2:v3_2mic`), cited in `cascade_decision_rule.md` |
| `eval_dualmic_cascade.out` | `hybrid2:v3_2mic` and `hybrid2_refdead:v3_2mic` on `dualmic_v1`, run once after the cascade rule was written |
| `eval_babble_1mic.out`, `eval_babble_2mic.out`, `eval_babble_cascade.out`, `eval_babble_v3hq_cont.out` | `evaluate --set dualmic_babble_v1` for the single-mic systems, the two-mic network, the cascade and `v3hq_cont` |
| `dfn3_highsnr.out` | `scripts/run_deepfilternet3.py --set defence_highsnr_v1` |
| `eval_highsnr_aborted_missing_dfn3.out` | **Incident:** empty log of a first high-SNR evaluation started before the DFN3 outputs existed; stopped before writing results (REPORT §10) |
| `eval_highsnr.out`, `eval_highsnr_v3hq_cont.out` | `evaluate --set defence_highsnr_v1` (noisy, DFN3, ll, hq_metric, v3ll, v3hq; then v3hq_cont) |
| `latency_round2.out`, `latency_round2_final.out` | `scripts/benchmark_latency.py --tag round2` (idle machine, after the two-mic training) and `--tag round2_final` (adds v3hq_cont and the cascade / fallback rows) |
| `verify_install_round2.out` | `python deploy/jetson/verify_install.py` on the Mac, all three configurations: `115 PASS, 0 FAIL, 0 WARN, 10 INFO` (2026-10-05 07:27) |
| `pytest_round2_final.out` | `python -m pytest -q`: `103 passed, 36 warnings` (2026-10-05 07:28) |
| `package_bundle_round2.out` | `deploy/jetson/package_for_jetson.sh`: bundle updated in place, 3 models, 42 data files, 115 checksums, 13 MB (2026-10-05 07:31) |

---

## 12. Third-party material: `third_party/` (3P)

| Path | Size | Contents | Licence |
|---|---:|---|---|
| `dfn_models/DeepFilterNet3.zip` | 7.6 MB | Official DeepFilterNet3 weights archive | MIT / Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE` alongside) |
| `dfn_models/DeepFilterNet3/` | 8.3 MB | Extracted `config.ini` and `checkpoints/model_120.ckpt.best` | same |
| `dfn_pkgs/` | 48 MB | Isolated packages for DFN3 only (deepfilternet 0.5.6, DeepFilterLib 0.5.6, numpy 1.26.4, loguru, appdirs, packaging), installed by `uv` into this separate target directory so the main `.venv` was not changed. Used only via `PYTHONPATH=third_party/dfn_pkgs`. Built for macOS arm64 | per package (dist-info folders): MIT / Apache-2.0 / BSD; numpy's bundled GCC runtime libraries GPL-3.0-with-runtime-exception / LGPL. loguru's MIT notice was missing from its wheel and was added as `loguru-0.7.3.dist-info/LICENSE` (2026-10-05) |

DFN3 is an external reference baseline only; D-ANC does not depend on it. Unchanged in round 2.

---

## 13. Final state and the shared packages (2026-10-05)

- **Nothing is running.** Round 2 is complete; every adoption decision is in `reports/results/*_decision*.md`; the
  deployed set (HQ = `v3hq_cont`, LL = `ll`, TWO_MIC = NLMS → `v3_2mic` with HQ fallback) is in
  `exports/deployment_selection.json` and `exports/DEPLOYMENT.json`.
- **Done before sharing:** HTML report rebuilt (round 2); workflow records completed (`docs/workflows/`, 8 runs);
  pre-share review run and acted on; the Jetson bundle rebuilt from the final files; tests (103) and the
  self-check (115 PASS) passed; `reports/MANIFEST.sha256` regenerated **last**.
- **Shared as** (folder `~/Desktop/DRDO_Project_SHARE_2026-10-05/`, built by `scripts/build_share_packages.py`):
  - `DRDO_Project_TEAM_2026-10-05.zip`: every file listed in this index except the ENV rows (`.venv/`, caches,
    `.DS_Store`). For named team members only, because of the data licences.
  - `DRDO_Project_PUBLIC_2026-10-05.zip`: the same minus restricted third-party material (drone and NOISEX-92 audio,
    ESC-50, test-set noisy sides, enhanced audio, trained models, the DSJ PDF); its own manifest. See
    `SHARE_README_PUBLIC.md` (at the top of both zips and next to them).

---

## 14. How to verify integrity

This section describes the full (team) project. `reports/MANIFEST.sha256` lists the SHA-256 of every shipped
project file except `third_party/dfn_pkgs/`, the empty `*.lock` files and the manifest itself; `.venv/`, `__pycache__/`, `.pytest_cache/`,
`.git/` and `.DS_Store` are not shipped. It was regenerated **last**, with `python scripts/make_manifest.py`, after
every other file (including this index) was final.

`scripts/make_manifest.py` also rewrites `exports/DEPLOYMENT.json`. Since round 2 it reads the deployed models
from `exports/deployment_selection.json` (no hard-coded list) and keeps extra fields such as
`reference_systems`, so regenerating it does not change the deployed set.

```bash
cd DRDO_Project
shasum -a 256 -c reports/MANIFEST.sha256 | grep -v ': OK$'    # prints nothing when every file matches (Linux: sha256sum)
```

Run the check before installing the package: `pip install -e .` rewrites `src/danc.egg-info/SOURCES.txt`, which is
listed. Any `FAILED` line means a file is missing or changed. The deployed models are additionally pinned by the
hashes in `exports/DEPLOYMENT.json` (and, in the Jetson bundle, by `dist/jetson_bundle/SHA256SUMS`), and the
reference paper by the hash in `docs/PROJECT_BRIEF.md`.

---

## 15. Related material outside this folder (not included)

Everything needed to understand, rebuild, evaluate and deploy the project is inside `DRDO_Project/`. On
2026-10-05 the pre-share review checked what lived outside; the items that had produced shipped content were
copied in:
- `scripts/assemble_report_tables.py`, which put the tables into `REPORT.md`;
- the round-1 self-check logs, `logs/verify_install_round1*.out` and `reports/results/verify_install_*round1*.json`;
- `scripts/verify_install_jetsonlike.py`;
- the round-2 code review and report audit, `docs/research/round2_code_review.md`, `report_audit_round2.md`,
  `round2_review_repro/` and `file_index_review_round2.md`;
- the pre-share review, `docs/research/pre_share_review_2026-10-05.md` and workflow `wf_c549189e-2c8`;
- the complete journal of `wf_14dd115a-9e9`;
- the packager-incident evidence, `logs/packager_incident_2026-10-04/`.

What remains outside, deliberately:

| Location | What it is | Why it is not included |
|---|---|---|
| `~/Downloads/21095.pdf` | The owner's original reference paper | Byte-identical copy is in `docs/references/` (same SHA-256) |
| `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/` (≈ 130 MB) | Claude Code session transcripts, sub-agent transcripts, workflow state, tool-result caches, assistant memory notes, background-task outputs (`…/tasks/`, ≈ 0.6 MB, mirrored in `logs/` or reproducible from `reports/results/`) | Development-tool records. The owner's requests are reproduced verbatim in `docs/PROJECT_BRIEF.md`; every workflow's script, final state and journal, and the final reports of the review agents, are in `docs/` |
| `/private/tmp/claude-502/.../scratchpad/` (≈ 130 MB, cleared on reboot) | Cached third-party papers and standards read to verify citations; superseded report drafts and pre-edit script backups; disposable probes and test copies | The papers are third-party copyright (verification outcome in `docs/research/claims_verification.*`); everything that generated shipped content was copied in (above) |
| `https://claude.ai/artifact/Kbp9PUhBQQvhNow44Wz8xu` | The web report, published privately from `reports/site/index.html` | Same content as `reports/site/` |
| `~/Desktop/Screenshot 2026-10-04 at 2.15.56 pm.png` | Screenshot of interim round-2 results in the session | Not needed |
| `~/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/` | The Python interpreter the `.venv` points to | Installed by `uv`; rebuilt with `uv venv --python 3.11` (`SHARE_README.md` §3) |
