# Pre-registered adoption rule for the two-microphone DANCNet (v3_2mic)

Written 2026-10-04 22:35 +04, **before any test-set result of the two-microphone network existed**
(`reports/results/dualmic_v1/per_file.csv` contained 0 `dnn2:` rows at the time of writing). Only the
validation scores on `data/testsets/dualmic_val` (used for checkpoint selection) had been seen.

**Comparator:** `hybrid:v3hq` — the single-mic v3 network behind the sub-band NLMS reference canceller. It is the
strongest system on `dualmic_v1` so far (PESQ-WB 2.471, PESQ-NB 3.092, STOI 0.931, SI-SDR 13.58 dB, 240 files).

The two-microphone network (`dnn2:v3_2mic`, `exports/dancnet_v3_2mic.onnx`) becomes the recommended
configuration for two-microphone headsets only if ALL of the following hold:

1. **dualmic_v1, all 240 files:** ΔPESQ-WB ≥ +0.03, ΔSTOI ≥ −0.002 and ΔSI-SDR ≥ −0.2 dB vs `hybrid:v3hq`.
2. **No category regression:** in each of stationary / non-stationary / impulsive / mixed (60 files each),
   ΔPESQ-WB ≥ −0.05 and ΔSTOI ≥ −0.01 vs `hybrid:v3hq`.
3. **Reference-microphone failure (graceful degradation):** with the reference input replaced by −80 dBFS white
   noise (a dead or unplugged reference mic), the network's STOI and PESQ-WB on dualmic_v1 must not be lower than
   those of the unprocessed primary mic. The result is also reported against the single-mic `dnn:v3hq`.
   If criteria 1, 2 and 4 pass but 3 fails, the model may be recommended **only together with** a
   reference-health fallback to the single-mic engine, and the report must say so.
4. **Edge budget:** the full streaming engine on the Apple M4 (1 thread, unpaced, 6 000 hops) has
   p99.9 < 1.5 ms per 10 ms hop.

If the rule is not met, the single-mic hybrid stays the recommended two-microphone configuration and the
two-microphone network is reported as an experimental result, with its numbers.

**Addendum (2026-10-04 22:44 +04, before any two-mic result on it exists):** a dual-microphone crew-babble set
(`data/testsets/dualmic_babble_v1`, 60 scenes, built by `danc.data.build_dual_babble`) was added after the
single-mic edge-case evaluation showed crew babble as the weakest case. Results on it are **reported, not part of
the adoption rule above**.
