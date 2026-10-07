# Decision: cascade NLMS → two-microphone network (`hybrid2:v3_2mic`) vs `hybrid:v3hq`, applying the pre-registered rule (cascade_decision_rule.md)

Configuration: `EngineConfig(lms_two_mic=True)` (NLMS μ 0.1, 2 taps, gate exponent 4) + `exports/dancnet_v3_2mic.onnx`;
launcher flag `--two-mic-cascade`. Test set `dualmic_v1`, 240 scenes, −5…10 dB, evaluated once
(`logs/eval_dualmic_cascade.out`).

| Criterion | Requirement | Measured (cascade − hybrid v3 HQ) | Pass |
|---|---|---|---|
| 1 Overall | ΔPESQ-WB ≥ +0.03, ΔSTOI ≥ −0.002, ΔSI-SDR ≥ −0.2 dB | ΔPESQ-WB **+0.145** (2.472 → 2.616), ΔPESQ-NB +0.111, ΔSTOI +0.0005 (0.9305 → 0.9310), ΔESTOI +0.0008, ΔSI-SDR +0.21 dB, ΔSNR +0.18 dB | yes |
| 2 Per category | each ΔPESQ-WB ≥ −0.05 and ΔSTOI ≥ −0.01 | stationary +0.016 / −0.0058; non-stationary +0.127 / +0.0007; impulsive +0.237 / +0.0056; mixed +0.199 / +0.0015 | yes |
| 3 Reference failure | dead reference (−80 dBFS): STOI and PESQ-WB ≥ unprocessed primary | STOI 0.8115 vs 0.7982, PESQ-WB 1.309 vs 1.230 | yes, but narrowly (see below) |
| 4 Edge budget | engine p99.9 < 1.5 ms (M4, idle) for the configuration that ships | shipped configuration (cascade + `dancnet_v3hq_cont.onnx` in hot standby): mean 1.271, **p99.9 1.406 ms**, max 1.52 ms; cascade alone: 0.680 / 1.222 / 1.50 ms; 0 misses of the 10 ms deadline (`reports/results/latency_mac_m4_round2_final.json`) | yes |

**Reference failure.** Criterion 3 passes, but only just: with a dead reference the cascade is barely better than
doing nothing (STOI 0.812 vs 0.798), and far below the single-mic network on the same scenes (`dnn:v3hq`: STOI 0.905,
PESQ-WB 2.21). The cascade is therefore recommended **only together with** the reference-health fallback
(`--fallback-onnx exports/dancnet_v3hq_cont.onnx`, the deployed HQ model; any single-mic 40 ms model fits), which switches to the single-mic network in under 1 s
after a reference failure (0.5 s hold, about 0.1 s level smoothing, 0.1 s cross-fade; §4 of the report). Criterion 4 is measured for that configuration.

**Crew babble (reported, not part of the rule; `dualmic_babble_v1`, 60 scenes).**

| System | PESQ-WB | PESQ-NB | STOI | ESTOI | SI-SDR |
|---|---|---|---|---|---|
| unprocessed primary | 1.121 | 1.602 | 0.740 | 0.513 | 3.5 dB |
| hybrid NLMS + v3 HQ | 1.367 | 1.998 | 0.823 | 0.664 | 7.7 dB |
| two-mic network alone | 1.858 | 2.623 | 0.890 | 0.770 | 11.8 dB |
| **cascade** | **2.060** | **2.799** | **0.904** | **0.797** | **12.5 dB** |

**Adopted** as the recommended configuration for two-microphone headsets, **with the reference-health fallback**:
`danc_live.py --onnx exports/dancnet_v3_2mic.onnx --two-mic-cascade --fallback-onnx exports/dancnet_v3hq_cont.onnx`
(deployment entry `TWO_MIC` in `exports/deployment_selection.json`). Single-microphone headsets use the HQ or LL model.

For information, against the hybrid with the final round-2 HQ model (`hybrid:v3hq_cont`, adopted after this rule was
written): PESQ-WB 2.499 → 2.616, STOI 0.9319 → 0.9310, SI-SDR 13.68 → 13.79 dB.
Caveat: the latency margin of the shipped configuration on the M4 is modest (1.41 ms against the 1.5 ms criterion; the
deadline is 10 ms). It runs two networks per hop, so it must be measured on the Jetson (`verify_install.py --strict-timing`).
