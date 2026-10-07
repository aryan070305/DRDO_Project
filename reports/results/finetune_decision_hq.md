# Decision: HQ fine-tune (hq_metric) vs base (hq), applying the pre-registered rule (finetune_decision_rule.md)

| Criterion | Requirement | Measured | Pass |
|---|---|---|---|
| 1 Defence test (−5..15 dB, 800 files) | ΔPESQ-WB ≥ −0.02, ΔSTOI ≥ −0.003, ΔSI-SDR ≥ −0.2 dB | ΔPESQ-WB −0.014, ΔSTOI 0.000, ΔSI-SDR +0.002 dB (ΔSNR +0.12 dB) | yes |
| 2 VoiceBank+DEMAND (824 files) | STOI ↑, or PESQ-WB ≥ +0.05, with ΔSI-SDR ≥ −0.2 | ΔSTOI +0.012, ΔPESQ-WB +0.129, ΔSI-SDR +0.37 dB | yes |
| 3 Clean-input transparency | PESQ-WB, STOI not lower | VB clean 2.98→3.56 / 0.940→0.960; Libri clean 4.45→4.47 / 0.996→0.997 | yes |

**Adopted:** `checkpoints/hq_metric/last.pt` → `exports/dancnet_hqft.onnx` is the deployed HQ (40 ms) model.
Attribution note: the stage combines the PESQ-guided adversarial term with timbre/bandwidth + clean-only
augmentation. In-domain PESQ did not rise (validation PESQ-WB 2.482 at the start → 2.461 at the final, deployed step 2000), so the measured benefit is
robustness and generalisation, most plausibly from the augmentation. It is **not** a PESQ gain on defence noise.
