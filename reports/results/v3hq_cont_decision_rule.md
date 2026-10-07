# Pre-registered adoption rule for the v3-HQ warm-restart continuation (v3hq_cont)

Written 2026-10-04 22:38 +04, before the continuation run started (no checkpoint, validation or test result of
`v3hq_cont` exists at the time of writing).

**Why this run:** the v3hq validation curve flattened while the cosine learning-rate schedule approached its floor
(PESQ-WB 2.4878 → 2.5066 → 2.5183 → 2.5197 at steps 10k/12k/14k/16k, LR → 2e-5). It is not known whether the
plateau comes from the schedule or from model capacity. A warm restart (same architecture, peak LR 3e-4,
10 000 further steps, new data seed 43) tests whether more training on the M4 GPU still pays.

`v3hq_cont` (`checkpoints/v3hq_cont/best.pt`) replaces `v3hq` as the HQ candidate only if ALL hold:

1. **Defence test set (defence_v1):** ΔPESQ-WB ≥ +0.02, ΔSTOI ≥ −0.002 and ΔSI-SDR ≥ −0.2 dB vs `dancnet:v3hq`.
2. **VoiceBank+DEMAND:** STOI ≥ v3hq − 0.005 and PESQ-WB ≥ v3hq − 0.02.
3. **Clean transparency:** VoiceBank-clean PESQ-WB ≥ 3.3 and LibriSpeech-clean PESQ-WB ≥ 4.3.
4. **Edge budget:** identical architecture and ONNX graph size to v3hq, so v3hq's latency result applies; the
   export is checked for the same parameter count.

Whichever of v3hq / v3hq_cont wins is then subject to `v3_decision_rule.md` against the deployed HQ model.
