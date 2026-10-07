# D-ANC: Hybrid AI/ML Adaptive Noise Cancellation for Defence Voice Communications

D-ANC is a real-time, transmit-path noise-cancellation system for two-way defence voice communications. It
combines a reference-microphone sub-band NLMS adaptive filter with **DANCNet**, a causal
sub-band/full-band, complex-domain deep-filtering network. The full technical report, with evidence,
results, risks and references, is **[`reports/REPORT.md`](reports/REPORT.md)**.

> Development platform: Apple M4 (16 GB, macOS 26), Python 3.11 venv (`.venv`, created with `uv`).
> Target platform: NVIDIA Jetson AGX Orin 64 GB. The Jetson-specific parts are prepared but were **not
> run on hardware** in this project; see `deploy/jetson/README.md`.

## Round 2 at a glance (2026-10-05)

| Deployed configuration | Files | Defence set PESQ-WB / STOI | Dual-mic set PESQ-WB / STOI | M4 per-hop p99.9 |
|---|---|---|---|---|
| **HQ** (40 ms), single mic + NLMS hybrid | `exports/dancnet_v3hq_cont.onnx` | 2.51 / 0.918 | 2.50 / 0.932 | 0.75 ms |
| **LL** (20 ms), unchanged from round 1 | `exports/dancnet_ll.onnx` | 2.33 / 0.904 | 2.37 / 0.923 | 0.60 ms |
| **TWO_MIC** (40 ms): NLMS → two-mic network, HQ in hot standby | `exports/dancnet_v3_2mic.onnx` | (needs two mics) | **2.62** / 0.931; crew babble STOI **0.904** | 1.41 ms |

- **Start here:**
  - `reports/REPORT.md`, the full report (§0 summary; §8.6–8.10 are the round-2 results).
  - `deploy/jetson/QUICKSTART.md`, the Jetson deployment guide.
  - `exports/deployment_selection.json`, which defines what is deployed.
- **Every model change was decided by a pre-registered rule:** `reports/results/*_decision_rule.md` holds each rule, and `*_decision*.md` the matching decision.
- **Self-check:** `python deploy/jetson/verify_install.py`. On the Mac it gives 115 PASS, 0 FAIL for all three configurations.
- **Tests:** `python -m pytest -q` (103 tests).

## Repository layout

```
src/danc/
  dsp.py                    shared STFT convention (16 kHz, 20 ms sqrt-Hann, 10 ms hop), offline + streaming
  data/
    download.py             reproducible downloader + provenance/licence manifest (SHA-256)
    speech.py               LibriSpeech manifests, speaker-disjoint train/val/test (+ v2 train set)
    noise_bank.py           NOISEX-92, ESC-50, drone recordings, synthetic generators; train/test splits
    synth_noise.py          physics-based gunfire/artillery/explosion/helicopter/drone/siren/wind/vehicle noise
    rir.py                  image-source RIR bank (pyroomacoustics)
    mixer.py                dynamic mixing + augmentation (SNR, reverb, clipping, EQ, level) + torch Dataset
    dual_mic.py             dual-microphone headset scene simulator (directional + diffuse noise, speech leak)
    build_testset.py        fixed evaluation sets written to data/testsets/
    build_edgeset.py        round 2: 11 edge cases (clipping, levels, reverb, radio band, blast, babble, 60 s streams)
    build_dual_babble.py    round 2: dual-mic crew-babble scenes
    build_highsnr.py        round 2: defence set at 20 / 25 dB (PESQ attainment)
    dual_mixer.py           round 2: dynamic dual-mic training scenes for the two-mic network
  models/
    dancnet.py              DANCNet (streaming-capable, explicit state, optional look-ahead)
    baselines.py            spectral subtraction, Wiener (DD), log-MMSE (LSA) with causal noise tracking
  adaptive/nlms.py          time-domain NLMS and STFT-domain multi-tap sub-band NLMS (SPP-gated, impulse-robust)
  train/
    losses.py               compressed complex/mag MSE, MR-STFT, SI-SDR, PMSQE, asymmetric, SPP
    train.py                training loop (MPS/CPU), validation, EMA, warm start, resume
    finetune_metric.py      MetricGAN+-style PESQ-guided fine-tuning (anchored by the full loss)
    train_dual.py           round 2: two-microphone network training (warm start, resume)
    third_party/pmsqe.py    PMSQE loss vendored from Asteroid (MIT)
  inference/
    export_onnx.py          single-frame streaming ONNX export + numerical verification
    runners.py              PyTorch and ONNX Runtime step runners (common interface)
    engine.py               hybrid real-time engine: shared STFT -> sub-band NLMS -> DANCNet -> limiter
                            (+ round 2: two-mic network, NLMS->two-mic cascade, reference-health fallback)
    postfilter.py           round 2: optional SPP post-filter (evaluated, not adopted)
    realtime.py             live duplex audio app (primary + reference mic -> headset/radio) and paced simulation
  eval/
    metrics.py              PESQ-WB/NB, STOI, ESTOI, SI-SDR, output SNR, segSNR
    evaluate.py             evaluation of all systems on the test sets -> CSV/JSON
configs/                    training / fine-tuning configurations (YAML)
deploy/jetson/              TensorRT build, TRT runner, ALSA/RT setup, systemd unit, power + latency tools
deploy/ALTERNATIVE_PLATFORMS.md
scripts/                    tables/figures (make_tables, make_figures*, paired_stats, band_error_analysis), latency benchmark, edge-case eval, queue scripts (run_round2*_chain.sh), DeepFilterNet3 reference run
tests/                      pytest suite (DSP, model streaming equivalence, engine, adaptive filters, generators)
data/                       raw corpora, manifests (provenance.json), RIR bank, fixed test sets
checkpoints/ exports/       trained models (PyTorch) and ONNX streaming models
reports/                    REPORT.md, figures/, results/ (per-file CSV + summaries), audio/ (enhanced examples)
docs/research/              literature research notes with sources (T1..T7) + claim verification
third_party/                DeepFilterNet3 reference (official weights MIT/Apache-2.0; isolated macOS-arm64 packages MIT/Apache-2.0/BSD, numpy's GCC runtime GPL-with-exception/LGPL)
```

## Reproduce

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
python -m pytest -q                                            # unit tests

python -m danc.data.download --root data/raw                   # corpora (see licences below)
python -m danc.data.download --root data/raw --only librispeech_partial
python -m danc.data.speech                                     # manifests
python -m danc.data.build_testset                              # fixed test sets

python -m danc.train.train --config configs/train_hq.yaml      # 40 ms "HQ" model (look-ahead 2)
python -m danc.train.train --config configs/train_v2_ll.yaml   # 20 ms "LL" model (warm start from HQ)
python -m danc.train.finetune_metric --config configs/finetune_hq_metric.yaml   # round-1 deployed HQ = this fine-tune (superseded in round 2)
python -m danc.train.finetune_metric --config configs/finetune_ll_metric.yaml

python -m danc.inference.export_onnx --ckpt checkpoints/hq_metric/last.pt --out exports/dancnet_hqft.onnx   # round-1 deployed HQ (superseded in round 2)
python -m danc.inference.export_onnx --ckpt checkpoints/ll/best.pt --out exports/dancnet_ll.onnx           # deployed LL
python -m danc.eval.evaluate --set defence_v1 --systems noisy,specsub,wiener,logmmse,dancnet:hq_metric,dancnet:ll
python -m danc.eval.evaluate --set dualmic_v1 --systems primary,logmmse,nlms,dnn:hqft,hybrid:hqft,dnn:ll,hybrid:ll
PYTHONPATH=third_party/dfn_pkgs python scripts/run_deepfilternet3.py --set defence_v1   # external SOTA reference
python -m danc.eval.evaluate --set defence_v1 --systems file:dfn3
python -m danc.inference.realtime --onnx exports/<name>.onnx --simulate <primary.flac> --simulate-ref <reference.flac> --paced
```

The full command sequence used for every reported number is in `reports/REPORT.md` §10. Deployed models (round 2): `exports/dancnet_v3hq_cont.onnx` (HQ, 40 ms), `exports/dancnet_ll.onnx` (LL, 20 ms) and, for two-microphone headsets, `exports/dancnet_v3_2mic.onnx` with `--two-mic-cascade --fallback-onnx exports/dancnet_v3hq_cont.onnx`; see `exports/DEPLOYMENT.json`. Round 1 deployed `exports/dancnet_hqft.onnx` as HQ (`exports/DEPLOYMENT_round1.json`). The commands above are the round-1 sequence; round 2 is in `reports/REPORT.md` §10.

## Data and licences (details: `data/manifests/provenance.json`, report section 3)

| Source | Use | Licence |
|---|---|---|
| LibriSpeech (dev/test-clean, train-clean-100 partial), Mini LibriSpeech | clean speech | CC BY 4.0 |
| VoiceBank+DEMAND test set (16 kHz mirror) | public benchmark | CC BY 4.0 |
| NOISEX-92 (SPIB mirror) | military vehicle/cockpit/machine-gun noise | © TNO 1990; redistribution terms **unclear** |
| ESC-50 (16 defence-relevant classes) | helicopter, siren, engine, wind, fireworks, … | CC BY-NC 3.0, **non-commercial** |
| DroneAudioDataset (Al-Emadi et al.) | drone noise | **no licence**, internal R&D only, do not redistribute |
| Synthetic generators (`synth_noise.py`) | gunfire, artillery, rotors, drones, sirens, wind | generated by this project |

Before sharing the `data/` folder outside the project, review the NOISEX-92, ESC-50 and drone-dataset terms.
