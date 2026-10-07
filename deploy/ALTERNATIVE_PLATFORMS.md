# Alternative edge platforms for D-ANC: qualitative, unmeasured

D-ANC's streaming network needs **0.37 to 0.76 GMAC/s** in fp32. That is one 10 ms frame of 3.7 to 7.6 M MAC. Its
recurrent state is under 64 KB and the weights under 1.5 MB. Any platform that can run that load with a
**p99.9 frame time well below 10 ms** and deterministic audio I/O will do. Because the network is batch-1,
sequential and GRU-heavy, single-thread CPU latency and DSP/NPU support for recurrent layers matter more than peak TOPS.

**None of the platforms below was tested in this project.** Do not plan on any figure until it has been taken from the
vendor's current datasheet and measured with `deploy/jetson/trt_runner.py --onnx ...`, or the vendor's
equivalent tool.

| Platform class | Fit for D-ANC | What to verify first |
|---|---|---|
| NVIDIA Jetson AGX Orin 64 GB (primary target) | Ample headroom. CPU path (ORT on Cortex-A78AE) recommended; GPU path via TensorRT FP16 and CUDA Graphs. Verified specs are in `jetson/README.md`. | Per-frame p99.9 in the 15 W and 30 W modes; audio I/O latency |
| Jetson Orin NX / Orin Nano | Smaller and lower-power siblings with the same software stack. The CPU path should still fit **[E]**. | Same as above; thermal behaviour in an enclosure |
| Qualcomm robotics / IoT SoCs (QCS-class) | Arm CPU cores plus Hexagon DSP/NPU. Recurrent-layer support on the NPU needs checking; the CPU path should work **[E]**. | Qualcomm AI Engine support for GRU; audio DSP integration |
| NXP i.MX 8M Plus (Cortex-A53 + NPU) | The A53 is much slower than the A78AE. ULCNet reported RTF 0.127 on one 1.43 GHz Cortex-A53 for a 0.098 GMAC model (ICASSP 2024). DANCNet-v2 needs about 8× more compute, so it would likely need the NPU or a slimmer variant **[E]**. | NPU support for GRU and dynamic state; otherwise use LL-small (0.37 GMAC/s) |
| TI AM62A / C7x DSP | DSP + MMA accelerator with a deterministic audio path. GRU support in the TI deep-learning toolchain must be checked. | TIDL operator coverage for GRU and transposed conv |
| ADI SHARC / audio DSPs | Excellent deterministic latency, typical of headset processing. Would need a hand-optimised GRU or a GRU-free variant (e.g. a TCN) **[E]**. | MAC throughput at fp32/fp16 vs 0.4–0.8 GMAC/s |
| Raspberry Pi 5 (Cortex-A76) | A CPU-only development stand-in. DeepFilterNet2 reported RTF 0.42 on a Pi 4 (IWAENC 2022). | Thermal throttling; audio HAT latency |

Selection rule: choose the smallest platform on which the full `danc.eval.evaluate` run with the target runner reproduces
the Mac results (ΔPESQ, ΔSTOI < 0.02) **and** the 10-minute real-time run has zero deadline misses at p99.9.
