# Decision: LL fine-tune (ll_metric) vs base (ll), applying the pre-registered rule (finetune_decision_rule.md)

| Criterion | Requirement | Measured | Pass |
|---|---|---|---|
| 1 Defence test (−5..15 dB, 800 files) | ΔPESQ-WB ≥ −0.02, ΔSTOI ≥ −0.003, ΔSI-SDR ≥ −0.2 dB | **ΔPESQ-WB −0.023**, ΔSTOI +0.0001, ΔSI-SDR +0.018 dB (ΔSNR +0.04 dB) | **no**, PESQ threshold missed by 0.003 |
| 2 VoiceBank+DEMAND (824 files) | STOI ↑, or PESQ-WB ≥ +0.05, with ΔSI-SDR ≥ −0.2 | ΔSTOI +0.012, ΔPESQ-WB +0.138, ΔSI-SDR +0.57 dB | yes |
| 3 Clean-input transparency | PESQ-WB, STOI not lower | VB clean 2.66→3.27 / 0.936→0.957; Libri clean 4.41→4.46 / 0.996→0.996 | yes |

**Not adopted.** Under the pre-registered rule, criterion 1 fails, so `checkpoints/ll/best.pt` → `exports/dancnet_ll.onnx`
remains the deployed LL (20 ms) model. `ll_metric` is reported as an ablation. Its generalisation is clearly better
(VoiceBank, clean-speech transparency) at a defence-set cost of 0.023 PESQ-WB, so the choice can be revisited
with field data.
