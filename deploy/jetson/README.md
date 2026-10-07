> ## Start here (deployment kit, added 2026-10-04)
>
> **New to this project? Follow [`QUICKSTART.md`](QUICKSTART.md)**: numbered copy-paste steps from a fresh
> Jetson AGX Orin 64GB (JetPack 6.2.x) to live operation, with troubleshooting and expected (Mac) results.
> Status: **not verified on hardware in this project** (developed on macOS).
>
> **Round 2 (2026-10-05):** the deployed set is HQ = `dancnet_v3hq_cont.onnx`, LL = `dancnet_ll.onnx`, and, for two-microphone headsets, TWO_MIC = NLMS → `dancnet_v3_2mic.onnx` with the HQ model in hot standby (QUICKSTART step 8). `verify_install.py` checks all three by default.
>
> | File | Purpose |
> |---|---|
> | [`QUICKSTART.md`](QUICKSTART.md) | step-by-step deployment, verification, latency/power measurement, systemd, troubleshooting |
> | [`install.sh`](install.sh) | idempotent installer: apt packages, venv (`/opt/danc/.venv`), [`requirements-jetson.txt`](requirements-jetson.txt), `pip install -e .`; no PyTorch; `--gpu`, `--with-trt`, `--offline-wheels` |
> | [`verify_install.py`](verify_install.py) | self-check on any machine (no torch): model contract vs `exports/DEPLOYMENT.json`, PESQ/STOI/SNR on 16 reference scenes, edge cases, 3000-hop timing; exits non-zero on failure |
> | [`package_for_jetson.sh`](package_for_jetson.sh) | builds `dist/jetson_bundle/` (+ `--tar`) with only what the Jetson needs, plus SHA-256 checksums |
> | [`danc_live.py`](danc_live.py) + [`danc-live.service`](danc-live.service) | torch-free live launcher and systemd unit. Use these instead of `python -m danc.inference.realtime` / `danc.service`, which import torch and pin CPUs that are offline in 15W/30W modes |
> | [`check_channels.py`](check_channels.py) | checks that input ch0 = primary (boom) mic and ch1 = reference mic, and the input levels |
> | [`danc_jetson_rt.py`](danc_jetson_rt.py) | shared torch-free runner (single- and two-microphone models), safety wrapper (NaN/Inf guard, bounded live statistics), stall watchdog (exits so systemd restarts the engine if the USB interface disappears), CPU pinning |

# D-ANC on NVIDIA Jetson AGX Orin 64GB — bring-up guide

> **Status: NOT VERIFIED ON HARDWARE IN THIS PROJECT.** All code and model artefacts were developed and
> validated on an Apple M4 (macOS) workstation. Everything that is Jetson-specific, including
> TensorRT engines, CUDA memory handling, ALSA/I2S configuration, power numbers and device latency, is a
> plan and has to be measured on the target before anyone relies on it. The numbers measured on
> the Mac are in `reports/REPORT.md` §6.

## 1. Target platform facts (from NVIDIA, fetched 2026-10-03)

| Item | Jetson AGX Orin 64GB | Source |
|---|---|---|
| AI performance | 275 TOPS (INT8, sparse) | nvidia.com Jetson Orin product page |
| CPU | 12-core Arm Cortex-A78AE v8.2, 2.2 GHz | same |
| GPU | 2048-core Ampere, 64 Tensor Cores, 1.3 GHz | same |
| Memory | 64 GB LPDDR5, 204.8 GB/s | same |
| Power | 15 W – 60 W | same |
| Audio | I2S + DMIC on the dev-kit 40-pin header | same |
| nvpmodel modes (r36.4 docs) | 0 = MAXN (12 CPU @ 2.2 GHz), 1 = 15W (4 CPU @ 1.11 GHz, GPU 408 MHz), 2 = 30W (8 CPU @ 1.73 GHz), 3 = 50W | Jetson Linux r36.4 Developer Guide, *Platform Power and Performance* |
| Software branches | JetPack 6.2.x (TensorRT 10.3, CUDA 12.6, Ubuntu 22.04) or JetPack 7.2 (TensorRT 10.16, CUDA 13.2, Ubuntu 24.04, released 2026-06-02) | developer.nvidia.com JetPack pages |

The NVIDIA documentation notes that hardware throttling engages when module power exceeds the TDP budget. Benchmark in
the power mode you will actually field (15 W or 30 W for battery or vehicle use), not only in MAXN.

## 2. Workload size, and why the CPU may beat the GPU

The deployed DANCNet-v2 streaming step is 340 k parameters and **7.64 M MAC per 10 ms frame (0.76 GMAC/s)**. Its recurrent state is 8
tensors totalling about 48–52 KB. A batch-1 network this small is dominated by per-kernel launch overhead on a GPU. NVIDIA's
TensorRT-RTX documentation quotes about 5–15 µs per kernel launch, and forum reports on AGX Orin describe occasional
launch stalls when a desktop session is running. Recommendation:

1. **Default: ONNX Runtime on 1–2 dedicated Cortex-A78AE cores.** On an Apple M4 core the full engine block takes
   0.5 ms unpaced and 2.1 ms mean (4.3 ms max) in paced real time (see the report, §6.2). An A78AE core is slower, so the
   Jetson figure is an *estimate, not a measurement*. With the performance governor and an isolated core it should stay well
   inside the 10 ms hop, but the paced worst case must be measured. Pin the thread (`taskset`) and run it as SCHED_FIFO (`chrt -f 80`).
2. **GPU path: TensorRT FP16 with CUDA Graphs** (`build_trt_engine.sh`, `trt_runner.py --cuda-graph`). Use it when the
   GPU is needed anyway (for example a larger model or several channels), and disable the desktop
   (`systemctl set-default multi-user.target`).
3. The DLA is not recommended: TensorRT documents that 10.7 was the last release with DLA support, and the DLA targets
   CNN workloads.
4. INT8: TensorRT loops (which is how GRUs are imported) support only FP32 and FP16. INT8 would therefore only be possible on
   the CPU (ORT dynamic quantization for RNNs), and it requires re-running the full PESQ/STOI evaluation.

## 3. Software setup

```bash
# on the Jetson (JetPack 6.2.x assumed)
sudo apt-get install -y python3-venv libsndfile1 libportaudio2 alsa-utils
python3 -m venv /opt/danc/.venv && source /opt/danc/.venv/bin/activate
pip install numpy scipy soundfile sounddevice soxr onnxruntime   # CPU EP (aarch64 wheels on PyPI)
# for the GPU EP / TensorRT: use the Jetson-specific onnxruntime-gpu wheel from the Jetson AI Lab index matching
# your JetPack; TensorRT Python bindings ship with JetPack (python3-libnvinfer).
pip install -e /opt/danc        # this repository (copy src/, exports/, deploy/)
```

Build and validate the engine on the device. The deployed single-mic models are `exports/dancnet_v3hq_cont.onnx` (HQ, 40 ms) and `exports/dancnet_ll.onnx` (LL, 20 ms); see `exports/DEPLOYMENT.json`. Substitute the LL file for the 20 ms mode. The two-microphone configuration (`TWO_MIC`: `exports/dancnet_v3_2mic.onnx --two-mic-cascade --fallback-onnx exports/dancnet_v3hq_cont.onnx`) runs on ONNX Runtime only, because `trt_runner.py` supports single-input models.

```bash
cd deploy/jetson
./build_trt_engine.sh ../../exports/dancnet_v3hq_cont.onnx ./engines
python3 trt_runner.py --onnx ../../exports/dancnet_v3hq_cont.onnx --frames 6000 --json ort_cpu.json   # CPU baseline
python3 trt_runner.py --engine engines/dancnet_v3hq_cont_fp16.engine --onnx ../../exports/dancnet_v3hq_cont.onnx \
        --compare --cuda-graph --json trt_fp16.json
```

Acceptance criteria per frame: p99.9 < 5 ms, no call > 10 ms over 10 minutes, max |TRT − ORT| < 1e-2 on `spec_out`.
After that, re-run `python -m danc.eval.evaluate --set defence_v1 ...` with the TRT runner and confirm that PESQ and STOI
match the Mac results to within 0.02.

## 4. Audio hardware for the prototype

```
 boom mic (primary, 2–3 cm from lips) ──┐
                                        ├─► USB audio interface (2 mic pre-amps, 16/48 kHz) ──USB──► Jetson AGX Orin
 outward reference mic (8–15 cm away) ──┘                         │                                 (danc.inference.realtime)
                                                                  └─► headphone out ─► headset earcups
                                                                  └─► line out ─► radio / intercom MIC input (via pad + isolation transformer)
```

* **Microphones.** Primary: a noise-cancelling dynamic or electret boom mic. Reference: an omni electret or MEMS mic
  on the outside of the earcup or helmet, 8–15 cm from the primary, facing away from the mouth. Pick capsules with a high
  acoustic overload point. Firearm peaks of roughly 140–175 dB SPL, reported in the hearing-conservation literature,
  will still clip most capsules. The network is trained with clipping augmentation, but clipping should be minimised
  in hardware too.
* **USB interface.** Any class-compliant 2-in/2-out interface that supports 16 or 48 kHz. Measure its loop-back latency with
  `measure_latency.py` before and after inserting the engine.
* **I2S MEMS mics on the 40-pin header** are an option for an integrated build. The pinmux and device tree must be configured with
  `jetson-io`, and the ALSA routing (ADMAIF/I2S) is board-specific. This was not attempted here.
* **Radio / communication unit.** Use a line-level output into the radio's microphone input through an attenuator pad,
  with an audio isolation transformer to break ground loops. Key PTT through a GPIO-driven opto-isolator or the radio's
  accessory connector. **Keep the sidetone (own-voice monitoring) path analog and unprocessed.** Own-voice latency
  is far more critical than the 20–30 ms that a transmit-path enhancer adds.
* **Acoustic ANC in the earcup** (anti-noise into the ear) needs microsecond-class latency. It belongs on the headset's
  dedicated ANC codec or DSP, not on the Jetson; see the report, §5.

## 5. Running

```bash
sudo ./setup_audio.sh                    # review first
python3 -m danc.inference.realtime --list-devices
chrt -f 80 taskset -c 10,11 python3 -m danc.inference.realtime --onnx exports/dancnet_v3hq_cont.onnx \
        --in-device USB --out-device USB --log engine.jsonl
./log_power.sh 120 power_15W.csv         # in another shell, for each nvpmodel mode
```

`danc.service` runs the engine as a systemd service with real-time scheduling.

## 6. Latency budget, planned and to be measured

| Stage | Budget |
|---|---|
| ADC + driver buffering (2 × 160-frame periods at 16 kHz) | ≈ 20 ms + converter group delay (measure) |
| Algorithmic (20 ms window, 0 look-ahead) | 20 ms |
| Compute (must fit inside one 10 ms hop) | < 10 ms, absorbed in the hop |
| DAC + driver | ≈ 10–20 ms (measure) |
| **Mouth-to-radio-input total (target)** | **≤ 50–60 ms** |

ITU-T G.114 treats < 150 ms mouth-to-ear as essentially transparent for most applications. The radio vocoder
(for example MELPe framing of 22.5–67.5 ms) and the network consume the rest of that budget.

## 7. Troubleshooting

* *Clicks or xruns:* increase the ALSA periods from 2 to 3, pin the engine to isolated cores, and check `cyclictest`.
* *Occasional 50–100 ms GPU stalls:* disable the desktop, set `CUDA_DEVICE_MAX_CONNECTIONS=32`, and use CUDA Graphs. These
  are user-reported mitigations from the NVIDIA forum and are unverified.
* *Interface refuses 16 kHz:* capture at 48 kHz and resample in software (soxr), which adds about 1–2 ms.
* *The reference mic hears the talker strongly* (speech-leak above −10 dB): move it away from the mouth. The SPP gate
  freezes adaptation during speech, but leakage still limits the achievable noise reduction (Widrow et al., 1975).
