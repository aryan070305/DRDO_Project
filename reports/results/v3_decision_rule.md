# Pre-registered adoption rule for DANCNet-v3 (written 2026-10-04, before v3 training started)

v3 (c96, 4 dual-path blocks, deep filter 5 taps, look-ahead 2, 609 k params, 13.75 MMAC/frame) replaces the
deployed HQ model (`exports/dancnet_hqft.onnx`) only if ALL of the following hold.

1. **Defence test set (defence_v1, −5…15 dB)**: ΔPESQ-WB ≥ +0.03, ΔSTOI ≥ −0.002 and ΔSI-SDR ≥ −0.2 dB vs the deployed HQ.
2. **VoiceBank+DEMAND**: STOI ≥ deployed HQ − 0.005, and PESQ-WB ≥ deployed HQ − 0.02.
3. **Clean transparency**: VoiceBank-clean PESQ-WB ≥ 3.3 and LibriSpeech-clean PESQ-WB ≥ 4.3.
4. **Edge budget**: the full streaming engine on the Apple M4 (1 thread, unpaced, 6 000 hops) has p99.9 < 1.5 ms per 10 ms hop.

The same thresholds (1–3) apply to a v3-LL (no look-ahead) model versus the deployed LL model.
