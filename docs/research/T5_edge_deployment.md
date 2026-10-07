# T5 - Edge Deployment of Streaming Speech Enhancement

Research date: 2026-10-03. Status: IN PROGRESS (file updated incrementally).

Scope: Jetson AGX Orin 64GB specs, TensorRT/ONNX Runtime for recurrent/streaming models, published on-device
measurements of SE models, quantization/pruning effects, Linux audio I/O latency, alternative edge platforms.

Evidence convention: `fetched=yes` means the page was fetched and the number was seen on it. `fetched=no` means search
snippet or secondary source only. All SE quality numbers below are on academic benchmarks (VoiceBank+DEMAND / DNS
Challenge), NOT on defence noise (rotor, gunfire, artillery, armoured vehicle). Treat them as relative indicators only.

---

## 1. Jetson AGX Orin 64GB - official specifications

| Item | Value | Source | fetched |
|---|---|---|---|
| AI performance | 275 TOPS (INT8, sparse) | NVIDIA Jetson Orin product page, https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/ | yes |
| GPU | 2048-core NVIDIA Ampere GPU with 64 Tensor Cores, 1.3 GHz | same | yes |
| CPU | 12-core Arm Cortex-A78AE v8.2 64-bit, 3MB L2 + 6MB L3, 2.2 GHz | same | yes |
| DLA | 2x NVDLA v2, 1.6 GHz | same | yes |
| PVA | 1x PVA v2 | same | yes |
| Memory | 64GB 256-bit LPDDR5, 204.8 GB/s | same | yes |
| Storage | 64GB eMMC 5.1 | same | yes |
| Power | 15 W - 60 W | same | yes |
| Audio I/F | I2S and DMIC (via 40-pin header on devkit) | same | yes |
| Industrial variant | Inline DRAM ECC, -40 to 85 C | same | yes |

Note: the WebFetch summary said "32GB & Industrial share identical architecture"; this is a summariser simplification.
The 32GB module has fewer GPU cores/TOPS (200 TOPS per the JetPack 7.2 note below). Verify from the module data sheet
before quoting 32GB numbers.

### 1.1 Power modes (nvpmodel), Jetson AGX Orin 64GB

Source: NVIDIA Jetson Linux Developer Guide r36.4, "Platform Power and Performance - Jetson Orin Nano, NX and AGX Orin",
https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html (fetched=yes)

| Mode | ID | Online CPU | CPU max (MHz) | GPU TPC | GPU max (MHz) |
|---|---|---|---|---|---|
| MAXN | 0 | 12 | 2201.6 | 8 | 1301 |
| 15W | 1 | 4 | 1113.6 | 3 | 408 |
| 30W | 2 | 8 | 1728 | 4 | 612 |
| 50W | 3 | 12 | 1497.6 | 8 | 816 |

- DLA core clock ranges 614.4 MHz (15W) to 1600 MHz (MAXN).
- Doc caveat (verbatim fragment): "hardware throttling is engaged when the total module power exceeds the TDP budget".
  i.e. MAXN is not a guaranteed 60 W operating point; 50 W is the rated TDP mode for the 64GB module.
- Implication for us: in 15W mode only 4 CPU cores at 1.11 GHz and GPU at 408 MHz - a defence handset/vehicle box running
  on battery will likely use 15W/30W; benchmark there, not in MAXN.

### 1.2 Software stack (JetPack / TensorRT) and dates

| Release | Date | Contents | Source | fetched |
|---|---|---|---|---|
| Jetson AGX Orin Developer Kit (32GB memory, "full GPU/CPU/DLA of AGX Orin 64GB") orderable | 2022-03-22 | 275 TOPS INT8 @60W, 32GB LPDDR5 | NVIDIA forum announcement https://forums.developer.nvidia.com/t/introducing-the-jetson-orin-nx-series-and-jetson-agx-orin-developer-kit-availability/209139 | yes |
| Jetson AGX Orin 64GB Developer Kit | ~2023-04 (retailer listing, Switch Science) | 64GB module devkit | https://www.switch-science.com/products/8886 | no (search snippet only) |
| JetPack 6.2.1 | 2025 (forum "now live") | Jetson Linux 36.4.4, kernel 5.15, Ubuntu 22.04, CUDA 12.6, TensorRT 10.3, cuDNN 9.3 | https://docs.nvidia.com/jetson/jetpack/release-notes/index.html | no (snippet) |
| JetPack 6.2.3 | n/a on page | Jetson Linux 36.5.2, kernel 5.15, Ubuntu 22.04, CUDA 12.6.10, cuDNN 9.3.0, TensorRT 10.3.0, DLA 3.14 | https://developer.nvidia.com/embedded/jetpack-sdk-623 | yes |
| JetPack 7.2 | 2026-06-02 | Jetson Linux 39.2, kernel 6.8, Ubuntu 24.04, CUDA 13.2.1, cuDNN 9.20.0, TensorRT 10.16.2; "Adds support for the Jetson Orin product family within JetPack 7 releases"; adds AGX Orin 32GB MAXN_SUPER (200 -> 241 TOPS) | https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.md | yes |

Takeaway: as of 2026-10 there are two maintained branches for Orin: JetPack 6.2.x (TensorRT 10.3, CUDA 12.6, Ubuntu 22.04)
and JetPack 7.2 (TensorRT 10.16, CUDA 13.2, Ubuntu 24.04). Pin one for the programme.

---

## 2. Published runtime measurements of streaming SE models (CPU / embedded)

All quality metrics: academic test sets. Benchmark conditions listed per row.

| Model | Params / MACs | Quality (test set) | Runtime (hardware) | Algorithmic latency | Source | fetched |
|---|---|---|---|---|---|---|
| RNNoise (Valin) | 87,503 weights; ~40 MFLOPS total | n/a in abstract | "1.3% of a single core" Haswell i7-4800MQ; 14% of one core on Raspberry Pi 3 (Cortex-A53 1.2 GHz), 48 kHz | 20 ms window, 10 ms hop | arXiv:1709.08243, J.-M. Valin (Mozilla), 2017; https://ar5iv.labs.arxiv.org/html/1709.08243 | yes |
| DeepFilterNet (v1) | 1.778 M / 0.348 GMACs | PESQ 2.81, STOI 0.942 (VB+DEMAND) | RTF 0.11, i5-8250U | 40 ms | DeepFilterNet2 paper Table 1 | yes |
| DeepFilterNet2 | 2.306 M / 0.356 GMACs | PESQ 3.08, STOI 0.9429, CSIG 4.30, CBAK 3.40, COVL 3.69 (VB+DEMAND test) | RTF 0.04, i5-8250U (avg of 5 runs, single-thread); RTF 0.42 on Raspberry Pi 4 | 20 ms window, 50% overlap, 2-frame look-ahead -> 40 ms | Schroeter et al., "DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio", IWAENC 2022, arXiv:2205.05474 (2022-05-11); https://ar5iv.labs.arxiv.org/html/2205.05474 | yes |
| RNNoise (as baseline in DFN2 paper) | - | - | RTF 0.027, single-thread i5-8250U | - | same Table 1 | yes |
| DeepFilterNet3 | n/r | PESQ 3.17, STOI 0.944, CSIG 4.34, CBAK 3.61, COVL 3.77 (VCTK/DEMAND test; trained on DNS4) | RTF 0.19 single-thread i5-8250U | 40 ms, 2-frame look-ahead | Schroeter et al., "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement", Interspeech 2023 Show&Tell, arXiv:2305.08227 (2023-05-14) | yes |
| GTCRN | 48.2 K / 33.0 MMACs/s | PESQ 2.87, STOI 0.940, SI-SDR 18.83 (VCTK-DEMAND) | RTF 0.07 on i5-12400 (streaming) | n/r on README | Rong et al., ICASSP 2024; GitHub https://github.com/Xiaobin-Rong/gtcrn (MIT) | yes |
| DTLN | <1 M params | PESQ 3.04, STOI 94.76%, SI-SDR 16.34 dB (DNS-Challenge test, per README) | per 8 ms block: RPi 3B+ (A53 1.4 GHz) SavedModel 15.54 ms, TF-lite 9.6 ms, TF-lite quantized 2.2 ms; i5-6600k TF-lite 0.36 ms; ONNX Runtime ~1.13 ms (MacBook Air 2012) | 32 ms block, 8 ms shift, 16 kHz | Westhausen & Meyer, Interspeech 2020, arXiv:2005.07551; GitHub https://github.com/breizhn/DTLN (MIT) | yes |
| FastEnhancer-Medium (48 kHz) via faster-enhancer.c int8 runtime | - | int8 vs fp32: -0.006 PESQ, -0.08 dB SNR | RTF 0.069 Apple M2 core; 0.096 Galaxy S23+ (Snapdragon 8 Gen 2); 3.3x vs fp32 ONNX Runtime | - | G. Kim, arXiv:2607.25350 (2026-07-28), preprint | yes |
| DPDFNet (DeepFilterNet2 + dual-path RNN) | n/r | VB+DEMAND, DNS4 blind, 12-language low-SNR set | "real-time performance on NPN32" (edge NPU) for DPDFNet-4 | - | Rika, Sapir, Gus, arXiv:2512.16420, accepted Speech Communication (v3 2026-06-16) | yes (abstract) |

DeepFilterNet licensing: GitHub README says dual MIT / Apache-2.0 (code). The DFN3 arXiv abstract mentions CC BY-SA 4.0
for framework/weights - check the LICENSE files of the weights before use. https://github.com/Rikorose/DeepFilterNet (fetched=yes).
DeepFilterNet README states only 48 kHz wav input is supported by the CLI; a LADSPA plugin exists for PipeWire.


### 2.1 Published SE measurements on Jetson boards (what exists)

| Work | Board | What was measured | Numbers | Source | fetched |
|---|---|---|---|---|---|
| Kealey et al., "Real-time Audio Video Enhancement with a Microphone Array and Headphones" (Univ. Sherbrooke, F. Grondin group), submitted to IROS 2023, arXiv:2303.00949 (2023-03-02) | Jetson Xavier NX ("installed in a backpack") | Mic-to-headphone loop latency without processing, plus algorithmic latency of beamformer + NN post-filter | "measured to be 80 msec" I/O latency without processing; algorithmic 40 ms; total 120 ms; frame N=512, hop 256 | https://ar5iv.labs.arxiv.org/html/2303.00949 | yes |
| Armandpl/jetson_denoiser (fork of Meta "denoiser" / Demucs, Defossez et al. Interspeech 2020) | Jetson Nano 4GB | README claims TensorRT model + threaded audio I/O; ~3 GB RAM | The only latency line in README ("total lag: 41.3ms ... RTF: 0.8") is copied from upstream desktop-i5 numbers, NOT a Jetson measurement | https://github.com/Armandpl/jetson_denoiser ; upstream https://github.com/facebookresearch/denoiser (CC-BY-NC 4.0) | yes |

Upstream Demucs-denoiser RTF (quad-core i5 @2 GHz): H=48 1 thread 0.8, 4 threads 0.6; H=64 1 thread 1.2, 4 threads 1.0
(README, fetched=yes). License CC-BY-NC 4.0 - not usable commercially / for defence product without permission.

**NOT FOUND (explicit):** No peer-reviewed or vendor-published per-frame latency / RTF for DeepFilterNet, GTCRN, DTLN,
RNNoise, FullSubNet, or Demucs-denoiser on **Jetson AGX Orin** (any power mode) was found. No published TensorRT engine
latency for a GRU-based streaming SE model on Orin was found. No published end-to-end (mic->speaker) audio latency for
Jetson AGX Orin (ALSA/I2S or USB) was found. These must be measured in-house.

---

## 3. TensorRT / ONNX Runtime support for recurrent, stateful streaming models

| Claim | Detail | Source | fetched |
|---|---|---|---|
| IRNNv2Layer removed | "IRNNv2Layer was deprecated in TensorRT 7.2.1 and was removed in TensorRT 10.0." Use the Loop API (ILoopLayer/IRecurrenceLayer) to build LSTM/GRU; see sampleCharRNN | NVIDIA TensorRT 10.x docs, "Working with Loops", https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-with-loops.html | yes |
| Loop precision limit | "The loop API supports only FP32 and FP16 precision." -> no INT8 inside TensorRT loops | same | yes |
| ONNX GRU/LSTM/RNN import | onnx-tensorrt operator table (doc refers to TensorRT 11.3, opsets 9-24): GRU, LSTM, RNN supported in FP32/FP16/BF16; LSTM requires input_forget=0 and layout=0; Scan/Loop: output length cannot be dynamic | https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md | yes |
| DLA deprecation | TensorRT docs: "TensorRT 10.7 was the last release that supported DLA"; DLA not supported in TensorRT 11.0-11.3 | https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html | yes |
| DLA target workload | DLA described as fixed-function accelerator for "offloading CNN processing from the GPU" | same | yes |
| Kernel launch overhead | "most CUDA kernels take the CPU and the driver around 5-15 microseconds to launch per kernel"; small GEMMs / element-wise-heavy nets become "enqueue-bound"; remedy = CUDA Graphs | NVIDIA TensorRT-RTX docs, "Optimizing TensorRT-RTX Performance", https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/performance/optimization.html (desktop RTX doc, not Jetson) | yes |
| ORT quantization guidance | "recommended to use dynamic quantization for RNNs and transformer-based models, and static quantization for CNN models"; ARM CPUs need dot-product instructions for int8 gains; QDQ S8S8 default | ONNX Runtime docs, Quantization, https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html | yes |

Jetson-specific launch-latency anecdotes (NVIDIA developer forum, user reports, low evidence weight):
- AGX Orin thread (2025-03-31 to 2025-04-15): launch latency "~12 us to 1 ms" with periodic ~100 ms stalls; fixed to ~11-13 us
  with `CUDA_DEVICE_MAX_CONNECTIONS=32` and by disabling the desktop (`systemctl set-default multi-user.target`) because
  Xorg/gnome-shell ioctl calls blocked the GPU queue; NVIDIA moderator suggested CUDA Graphs.
  https://forums.developer.nvidia.com/t/jetson-agx-orin-kernel-launch-is-frequently-delayed-after-kernel-launch/328758 (fetched=yes)
- Orin NX thread (2024-09-20): user measured ~513 us total per small kernel (8.98 us of actual kernel time) vs ~110 us on
  Xavier NX; no NVIDIA root cause given. https://forums.developer.nvidia.com/t/very-long-kernel-launch-overhead-on-jetson-orin-nx/307370 (fetched=yes)

Engineering implication (inference, not a sourced fact): a GTCRN/DTLN-class model (tens of kernels, ~0.03-0.4 GMAC/s)
per 8-10 ms hop is likely launch-bound on the Orin GPU; the 12 Cortex-A78AE cores may give lower and more deterministic
latency. Export stateful single-step graphs (hidden state as explicit input/output tensors, seq_len=1) to avoid TRT loops,
use CUDA Graphs if on GPU, and disable the GUI. Must be measured.

---

## 4. Quantization and pruning effects on SE quality (papers)

| Paper | Venue / date | Method | Quality effect | Efficiency | fetched |
|---|---|---|---|---|---|
| Fedorov et al., "TinyLSTMs: Efficient Neural Speech Enhancement for Hearing Aids" (Arm ML Research) | Interspeech 2020, doi 10.21437/Interspeech.2020-1864; arXiv:2005.11138 (2020-05-20) | Pruning + integer quantization of weights/activations | "loss of 0.55dB SDR" | 11.9x model size, 2.9x ops reduction; latency "2.39ms, well within the 10ms target" on MCU | yes |
| Rusci et al., "Accelerating RNN-based Speech Enhancement on a Multi-Core MCU with Mixed FP16-INT8 PTQ" (GreenWaves) | ITEM workshop @ ECML-PKDD 2022; arXiv:2210.07692 (2022-10-14) | PTQ: recurrent layers INT8, rest FP16 | Mixed: PESQ -0.06; uniform 8-bit: PESQ -0.3 average | up to 4x speedup, 1.4-1.7x memory saving, 1+8 RISC-V core MCU, models up to 1.24 M params | yes |
| Cohen, Habi, Netzer, "Towards Fully Quantized Neural Networks For Speech Enhancement" (Sony SSI) | Interspeech 2023, pp. 181-185, doi 10.21437/Interspeech.2023-883 | Full INT8 QAT (FQSE) | Quantization "mainly affects signals with a high input SNR"; quantizing model input/output causes "major performance degradation" (abstract; numeric results not in abstract) | INT8-only devices | yes |
| Kim, "faster-enhancer.c" | arXiv:2607.25350 (2026-07-28), preprint | int8 GEMM runtime, fp16 state | -0.006 PESQ, -0.08 dB SNR vs fp32 | 3.3x vs fp32 ORT; RTF 0.069 M2, 0.096 Snapdragon 8 Gen 2 | yes |
| Itani et al., "Wireless Hearables With Programmable Speech AI Accelerators" (Univ. Washington) | arXiv:2503.18698 (2025-03-24, rev. 2025-10-22) | Mixed-precision quantization + QAT, HW/SW co-design | user study n=28 | 6 ms chunks, 5.54 ms inference, 71.6 mW | yes |
| Chee, Braun, Gopal, Cutler, "Performance optimizations on deep noise suppression models" (Microsoft) | arXiv:2110.04378 (2021-10-08) | Magnitude structured pruning as architecture search | "smooth model performance degradation", non-intrusive quality metric | up to 7.25x inference speedup | yes |
| Wu, Yu, Fu, Liu, Chien, Tsao, "Increasing Compactness of DL-based SE Models with Parameter Pruning and Quantization" | arXiv:1906.01078 (2019) | Channel pruning + weight-clustering quantization | STOI 0.70 -> 0.69 (-1.43%), PESQ 1.85 -> 1.79 (-3.24%) | model size 10.03% of original | yes |

Take-aways: (1) INT8 of recurrent weights is generally cheap in quality, but full INT8 including input/output and high-SNR
regions can degrade; (2) mixed FP16/INT8 is a safe default; (3) on Jetson GPU, TensorRT loops are FP32/FP16 only, so INT8
RNN is mainly a CPU (ORT/XNNPACK/custom) or NPU path.
