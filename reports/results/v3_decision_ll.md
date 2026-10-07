# Decision: v3-LL (`v3ll`, 20 ms) vs the deployed LL model (`ll`), applying the pre-registered rule (v3_decision_rule.md)

| Criterion | Requirement | Measured (v3ll − ll) | Pass |
|---|---|---|---|
| 1 Defence test (defence_v1, −5…15 dB, 800 files) | ΔPESQ-WB ≥ +0.03, ΔSTOI ≥ −0.002, ΔSI-SDR ≥ −0.2 dB | **ΔPESQ-WB +0.018** (2.326 → 2.344), ΔPESQ-NB +0.017, ΔSTOI +0.002 (0.904 → 0.906), ΔSI-SDR +0.12 dB | **no**: PESQ gain below the +0.03 bar |
| 2 VoiceBank+DEMAND (824 files) | STOI ≥ ll − 0.005, PESQ-WB ≥ ll − 0.02 | PESQ-WB 2.214 → 2.401 (+0.187), STOI 0.904 → 0.916 (+0.012), SI-SDR +0.61 dB | yes |
| 3 Clean transparency | VB-clean PESQ-WB ≥ 3.3, Libri-clean ≥ 4.3 | VB-clean 3.317 (ll: 2.659), Libri-clean 4.471 (ll: 4.414) | yes |

Dual-mic test (dualmic_v1, 240 files, not part of the rule): hybrid:v3ll PESQ-WB 2.373 vs hybrid:ll 2.375, STOI 0.924 vs 0.923.

**Not adopted under the rule.** `checkpoints/ll/best.pt` → `exports/dancnet_ll.onnx` stays the deployed 20 ms model.

**Note for the project owner:** v3ll is better on every metric of the defence test set and VoiceBank+DEMAND, and on clean transparency. It is not better on everything: on the edge set its noise-only attenuation is 1.7 dB lower, and at 25 dB input its PESQ-WB is 0.006 lower. Behind the reference-mic NLMS it is
on par (PESQ-WB −0.002, STOI +0.001). Its defence-set gain (+0.018 PESQ-WB) is
smaller than the pre-registered +0.03 bar. Its robustness gains are large: VoiceBank +0.19 PESQ-WB, and clean-input
transparency 3.32 vs 2.66, so it distorts clean speech much less. The +0.03 bar was set before training to avoid swapping models for noise-level gains.
Switching is a one-line change in `exports/deployment_selection.json`
(`"LL": {"onnx": "exports/dancnet_v3ll.onnx", "checkpoint": "checkpoints/v3ll/best.pt", ...}`) followed by
`python scripts/make_manifest.py`. It is left as an explicit owner decision rather than made silently.
