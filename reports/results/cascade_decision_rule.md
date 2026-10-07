# Pre-registered adoption rule for the NLMS → two-microphone-network cascade (`hybrid2:v3_2mic`)

Written 2026-10-05 02:00 +04, **before any test-set result of the cascade existed**
(`reports/results/dualmic_v1/per_file.csv` contained 0 `hybrid2` rows).

**Disclosure.** The cascade idea came from the test result of the plain two-mic network (`two_mic_decision.md`): the
network was better on impulsive noise and the gated NLMS on stationary noise. The configuration was then fixed
without retraining: `EngineConfig(lms_two_mic=True)`, with the same NLMS settings as the deployed hybrid (μ 0.1,
2 taps, gate exponent 4) and the same network `exports/dancnet_v3_2mic.onnx`. It was checked on the validation
scenes only (`dualmic_val`, 48 scenes, `logs/eval_dualmic_val_cascade_EXPLORATORY.out`): PESQ-WB 2.653 vs 2.531 for
`hybrid:v3hq`, STOI 0.939 vs 0.939. The test set is evaluated **once**, after this file was written.

The cascade becomes the recommended configuration for two-microphone headsets only if ALL hold, on `dualmic_v1`
(240 scenes) against `hybrid:v3hq`:

1. **Overall:** ΔPESQ-WB ≥ +0.03, ΔSTOI ≥ −0.002, ΔSI-SDR ≥ −0.2 dB.
2. **No category regression:** in each of the 4 categories, ΔPESQ-WB ≥ −0.05 and ΔSTOI ≥ −0.01.
3. **Reference failure:** with the reference dead (−80 dBFS white noise), STOI and PESQ-WB ≥ the unprocessed primary.
   If only this fails, the cascade may be recommended **only together with** the reference-health fallback.
4. **Edge budget:** the full engine p99.9 < 1.5 ms per hop on the M4 (idle, 1 thread, 6 000 hops), measured with
   the configuration that would ship (with the fallback if criterion 3 requires it).

Crew babble (`dualmic_babble_v1`) is reported, not part of the rule.
