# Pre-registered adoption rule for the final fine-tune stage

Written 2026-10-04 06:10, before any test-set result of the fine-tuned models existed. At that point only the
HQ fine-tune's step-500 in-domain validation was known: PESQ 2.473 vs 2.482 at the start, STOI 0.9205 vs 0.9206.

The fine-tuned model (`hq_metric`, `ll_metric`: PESQ-guided MetricGAN+-style objective plus timbre/bandwidth and
clean-only augmentation) replaces its base model (`hq`, `ll`) as the **deployed** model only if **all** of the
following hold.

1. **Defence test set (defence_v1, input SNR -5..15 dB, primary criterion).** The defence numbers may not get worse
   beyond evaluation noise: ΔPESQ-WB ≥ −0.02, ΔSTOI ≥ −0.003, ΔSI-SDR ≥ −0.2 dB, all relative to the base model.
2. **Generalisation (VoiceBank+DEMAND test).** STOI must improve, or PESQ-WB must improve by ≥ 0.05, while SI-SDR
   stays within −0.2 dB.
3. **Transparency (clean speech in).** On 100 clean VoiceBank and 100 clean LibriSpeech test utterances, PESQ-WB and
   STOI must not decrease.

If the rule is not met, the base model stays deployed. The fine-tuned model is still reported as an ablation.
