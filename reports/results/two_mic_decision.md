# Decision: two-microphone network (`dnn2:v3_2mic`) vs `hybrid:v3hq`, applying the pre-registered rule (two_mic_decision_rule.md)

Model: `checkpoints/v3_2mic/best.pt` (step 5 000, dual-mic validation PESQ-WB 2.558 / STOI 0.927 / SI-SDR 14.08 dB)
→ `exports/dancnet_v3_2mic.onnx`. Test set: `dualmic_v1`, 240 scenes, −5…10 dB.

| Criterion | Requirement | Measured (two-mic − hybrid v3 HQ) | Pass |
|---|---|---|---|
| 1 Overall | ΔPESQ-WB ≥ +0.03, ΔSTOI ≥ −0.002, ΔSI-SDR ≥ −0.2 dB | ΔPESQ-WB **+0.051** (2.472 → 2.522), ΔPESQ-NB +0.072, **ΔSTOI −0.0073** (0.931 → 0.923), ΔESTOI −0.015, ΔSI-SDR +0.44 dB, ΔSNR +0.26 dB | **no**: STOI guard |
| 2 Per category | ΔPESQ-WB ≥ −0.05 and ΔSTOI ≥ −0.01 in each category | stationary **−0.101 / −0.0135**; non-stationary +0.009 / −0.0078; impulsive **+0.219** / +0.0009; mixed +0.075 / −0.0087 | **no**: stationary |
| 3 Reference failure | STOI and PESQ-WB with a dead reference ≥ unprocessed primary | STOI 0.812 vs 0.798, PESQ-WB 1.315 vs 1.230 (single-mic `dnn:v3hq` on the same scenes: 0.905 / 2.206) | yes, narrowly |
| 4 Edge budget | engine p99.9 < 1.5 ms per hop (M4) | network alone 0.635 / **0.697 ms** (mean / p99.9); with the single-mic fallback in hot standby 1.222 / 1.323 ms | yes |

Criteria 3 and 4 pass; criteria 1 and 2 do not.

**Not adopted on its own.** The same network run *behind* the gated NLMS canceller (the cascade, no retraining;
`cascade_decision.md` and its own pre-registered rule) is decided separately. Until that decision is complete,
the single-mic network behind the gated NLMS (`hybrid:v3hq`) remains the recommended two-microphone configuration.

**What the result shows.**
- The network gives the better *perceptual quality and SI-SDR*: +0.05 PESQ-WB and +0.07 PESQ-NB overall, +0.22 PESQ-WB in impulsive scenes, and +0.20 PESQ-WB at 10 dB input.
- The linear NLMS canceller gives the better *intelligibility in stationary and diffuse-plus-coherent noise*. Coherent engine and vehicle noise is exactly what a continuously adapting linear canceller removes without speech distortion. At −5 dB input the hybrid keeps STOI 0.881 vs 0.865.
- The two approaches are complementary. The cascade, NLMS first and then this unchanged network, was therefore tested next (`cascade_decision.md`). A network *trained* on the NLMS output is the natural next step. This network had 5 000 warm-started steps against 16 000 for the single-mic v3 HQ.
- On crew babble (`dualmic_babble_v1`, not part of the rule) the network alone is far ahead of the NLMS hybrid: STOI 0.890 vs 0.823, PESQ-WB 1.86 vs 1.37.
