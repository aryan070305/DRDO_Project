# Decision: v3-HQ (`v3hq`, 40 ms) vs the deployed HQ model (`hq_metric` → `dancnet_hqft.onnx`), applying the pre-registered rule (v3_decision_rule.md)

| Criterion | Requirement | Measured (v3hq − deployed HQ) | Pass |
|---|---|---|---|
| 1 Defence test (defence_v1, −5…15 dB, 800 files) | ΔPESQ-WB ≥ +0.03, ΔSTOI ≥ −0.002, ΔSI-SDR ≥ −0.2 dB | ΔPESQ-WB **+0.043** [95 % CI +0.034, +0.051] (2.440 → 2.483), ΔPESQ-NB +0.044, ΔSTOI +0.0023 (0.914 → 0.916), ΔSI-SDR +0.18 dB (14.02 → 14.20), better on 68 % of files | yes |
| 2 VoiceBank+DEMAND (824 files) | STOI ≥ HQ − 0.005, PESQ-WB ≥ HQ − 0.02 | PESQ-WB 2.425 → 2.472 (+0.047), STOI 0.9187 → 0.9184 (−0.0003), SI-SDR +0.13 dB | yes |
| 3 Clean transparency | VB-clean PESQ-WB ≥ 3.3, Libri-clean ≥ 4.3 | VB-clean 3.573, Libri-clean 4.505 | yes |
| 4 Edge budget | Full engine p99.9 < 1.5 ms per hop (M4, 1 thread, unpaced, 6 000 hops) | mean 0.669 ms, p99.9 **0.751 ms**, max 0.83 ms, 0 misses (`reports/results/latency_mac_m4_round2.json`; deployed HQ: 0.511 / 0.597 ms) | yes |

Also measured, not part of the rule:
- Dual-mic hybrid: PESQ-WB 2.425 → 2.471, STOI 0.928 → 0.931.
- Edge cases: PESQ-WB and STOI equal or better in every case; output SNR is 0.1 dB lower on `hum400` (Table R11).
- Per-SNR: PESQ-WB higher at every input SNR from −10 to 25 dB.

**Adopted:** `checkpoints/v3hq/best.pt` → `exports/dancnet_v3hq.onnx` becomes the HQ (40 ms) model. A warm-restart
continuation of the same architecture (`v3hq_cont`, rule `v3hq_cont_decision_rule.md`) was still training when this
was written. The deployment files (`exports/deployment_selection.json` → `DEPLOYMENT.json`, Jetson bundle,
QUICKSTART) are switched once, after that decision. If the continuation passes its rule, it replaces v3hq;
otherwise v3hq is deployed.
