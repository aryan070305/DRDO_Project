# Decision: v3-HQ continuation (`v3hq_cont`) vs `v3hq`, applying the pre-registered rule (v3hq_cont_decision_rule.md)

Run: warm restart from `checkpoints/v3hq/best.pt` (EMA weights), peak LR 3e-4, 10 000 steps × 16 × 3 s, new data seed,
5.05 h on the M4 GPU (`logs/v3hq_cont.out`). Validation PESQ-WB / STOI / SI-SDR: 2.520 / 0.923 / 14.67 dB (v3hq) →
2.551 / 0.925 / 14.80 dB (`best.pt` = step 10 000). Export max |ONNX − PyTorch| = 1.7e-5.

| Criterion | Requirement | Measured (v3hq_cont − v3hq) | Pass |
|---|---|---|---|
| 1 Defence test (defence_v1, −5…15 dB, 800 files) | ΔPESQ-WB ≥ +0.02, ΔSTOI ≥ −0.002, ΔSI-SDR ≥ −0.2 dB | ΔPESQ-WB **+0.028** [95 % CI +0.024, +0.032] (2.483 → **2.511**), ΔPESQ-NB +0.026, ΔSTOI +0.0019, ΔSI-SDR +0.15 dB, ΔSNR +0.16 dB | yes |
| 2 VoiceBank+DEMAND (824 files) | STOI ≥ v3hq − 0.005, PESQ-WB ≥ v3hq − 0.02 | PESQ-WB 2.472 → 2.485, STOI 0.9184 → 0.9193, SI-SDR +0.14 dB | yes |
| 3 Clean transparency | VB-clean PESQ-WB ≥ 3.3, Libri-clean ≥ 4.3 | VB-clean 3.610, Libri-clean 4.519 | yes |
| 4 Edge budget | identical architecture and ONNX graph size | identical model config, 609 690 parameters + buffers, ONNX 2 542 605 bytes for both, same I/O contract → v3hq's p99.9 0.75 ms applies (re-measured in the final latency run) | yes |

**Adopted.** `checkpoints/v3hq_cont/best.pt` → `exports/dancnet_v3hq_cont.onnx` is the HQ (40 ms) model of round 2.
It replaces v3hq, which `v3_decision_hq.md` had already adopted over the round-1 HQ model. Checked directly against
that rule (vs round-1 HQ `hq_metric`), v3hq_cont gives defence ΔPESQ-WB +0.070, ΔSTOI +0.004 and ΔSI-SDR +0.33 dB;
VoiceBank PESQ-WB 2.485 vs 2.425 and STOI 0.919 vs 0.919. All of these pass. Its defence-set average PESQ-WB of 2.511 over −5…15 dB is the first
single-mic D-ANC model whose average exceeds the brief's PESQ > 2.5 target. Answer to the rule's question:
more training on the M4 GPU still paid, by +0.03 PESQ-WB.
