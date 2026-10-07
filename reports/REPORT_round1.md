# D-ANC: Hybrid AI/ML Adaptive Noise Cancellation for Defence Voice Communications
### Technical report: research, design, implementation, validation

*Project folder:* `DRDO_Project/` · *Report date:* 2026-10-03/04 · *Development platform:* Apple M4 (MacBook Air, 16 GB) ·
*Target platform:* NVIDIA Jetson AGX Orin 64 GB (prepared, **not measured on hardware**)

Evidence conventions used throughout:
- **[V]** means the claim was re-checked against its primary source by an independent adversarial checker on 2026-10-03. That check confirmed 47 of 54 claims and corrected 7; the corrected wording is used here. See `docs/research/claims_verification.md`.
- **[N]** marks material from the research notes in `docs/research/T1..T7` that was fetched but not re-checked.
- **[M]** marks a measurement made in this project. Its script and raw data are in the repository.
- **[E]** marks an engineering estimate. It has not been measured.
- All published PESQ, STOI and SNR figures quoted here come from civilian benchmarks: VoiceBank+DEMAND (VB-D), DNS Challenge sets and WSJ0 mixtures. **None of them uses defence noise or the SNRs of this brief, so they are not directly comparable with the defence-test-set numbers in §8.**

---

## 0. Executive summary

**What was built.** D-ANC is a complete, tested, real-time hybrid noise-cancellation system for defence voice. It consists of:
- a reproducible data pipeline: public corpora with SHA-256 provenance, and 13 physics-based generators for gunfire, artillery, rotor, drone, siren and wind noise;
- DANCNet, a causal complex-domain sub-band/full-band network with a deep filter, in two modes: **HQ** (40 ms algorithmic latency) and **LL** (20 ms);
- a reference-microphone **sub-band NLMS** coupled to the network through its speech-presence output;
- a training framework with compressed spectral, MR-STFT, SI-SDR and PMSQE perceptual losses, an over-suppression penalty, EMA, and PESQ-guided fine-tuning;
- a streaming ONNX inference engine numerically equivalent to the trained model (≤ 1.3e-5 ONNX vs PyTorch; ≤ 1e-4 streaming vs offline);
- a Jetson AGX Orin deployment kit;
- 68 passing unit tests.

All numbers below were measured in this project on held-out speakers and test-only noise **[M]**.

| Defence test set, 1 mic, input SNR −5…15 dB (800 files) | PESQ-WB | PESQ-NB | STOI | output SNR | SI-SDR |
|---|---|---|---|---|---|
| Noisy input | 1.34 | 1.95 | 0.815 | 3.8 dB | 3.8 dB |
| Best classical method (log-MMSE) | 1.44 | 2.09 | 0.798 | 5.9 dB | 5.1 dB |
| DeepFilterNet3, public causal reference model (40 ms) | 2.04 | 2.75 | 0.885 | 11.8 dB | 11.7 dB |
| **D-ANC LL, 20 ms** | **2.33** | **2.98** | **0.904** | **13.0 dB** | **13.3 dB** |
| **D-ANC HQ, 40 ms** | **2.44** | **3.10** | **0.914** | **13.9 dB** | **14.0 dB** |

| Dual-mic headset scenes, primary + reference mic, −5…10 dB (240 scenes) | PESQ-WB | STOI | output SNR |
|---|---|---|---|
| Primary microphone, unprocessed | 1.23 | 0.798 | 2.5 dB |
| NLMS only | 1.18 | 0.788 | 3.2 dB |
| DNN only (HQ) | 2.15 | 0.901 | 12.1 dB |
| **Hybrid NLMS + DNN, HQ / LL** | **2.43 / 2.38** | **0.928 / 0.923** | **13.4 / 13.4 dB** |

**Against the targets** (SNR > 15 dB, STOI > 0.85, PESQ-WB > 2.5):
- **STOI > 0.85 is met on average** in every evaluation: 0.914 single-mic HQ and 0.928 hybrid HQ. Single-mic, it holds from **0 dB input SNR** upwards. With the reference mic, the hybrid reaches 0.877 even at −5 dB.
- **All three targets are met together from about 10 dB input SNR upwards.** HQ at 10 dB scores PESQ-WB 2.85, STOI 0.951 and 16.5 dB output SNR; at 15 dB, 3.19 / 0.974 / 19.5 dB.
- **They are not met at −5 to 5 dB**, where the literature also does not meet them (§1). As a result, the −5…15 dB averages are PESQ-WB 2.44 (target 2.5) and output SNR 13.9 dB (target 15).
- D-ANC beats the strongest public causal model, DeepFilterNet3.
  - D-ANC HQ beats it in all 96 comparisons (4 noise categories × 6 input SNRs × PESQ-WB, STOI, ESTOI, output SNR).
  - D-ANC LL beats it in 95 of 96 at half its latency. The exception is output SNR in non-stationary noise at −10 dB: 5.5 vs 5.6 dB.

**PESQ > 4 (owner's request): not achieved, and not claimed.**
- No input-SNR condition of the noisy defence set averages PESQ-WB above 3.19 (deployed HQ, 15 dB). The best category × SNR cell is 3.34 (impulsive, 15 dB).
- 8.1 % of HQ test files exceed PESQ-NB 4, and 1.8 % exceed PESQ-WB 4. Only clean, noise-free input passed through the model averages above 4 (4.47, Table R10).
- No published causal or non-causal system averages PESQ-WB > 4, even on the milder VoiceBank+DEMAND benchmark, where the best is 3.55 [V].

**Real time.** The full streaming engine (STFT, NLMS, network, iSTFT and limiter) takes 0.50 ms mean and 0.61 ms p99.9 per 10 ms hop on one Apple-M4 core. In paced real-time simulation it took 2.1 ms mean (2.0–2.2 per scene) and 4.3 ms max, with **0 deadline misses** **[M]**. The network costs 0.76 GMAC/s (340 k parameters). Jetson AGX Orin latency and power were **not measured**; that has to be done on hardware with the scripts in `deploy/jetson/`.

**Main limitations, stated plainly.**
1. Noise is partly synthetic, and NOISEX/ESC-50 test noise is unseen audio of seen recording conditions. Field validation on recorded range and vehicle noise is required (§9, R1).
2. Generalisation to unseen *talkers or channels* needed a robustness fine-tune; see the VoiceBank results in §8.3.
3. Training was limited to about 25 h of speech on a fanless 16 GB laptop.
4. MRT listening tests through the real radio chain are the military acceptance criterion and remain to be done.

---

## 1. Problem framing and how the targets are interpreted

**What "ANC" means here.** The brief mixes two different technologies.

- *Acoustic* active noise control injects anti-noise into the ear. It needs microphone-to-loudspeaker latency of tens to hundreds of **microseconds**. For example, an open-ear ANC prototype reported a 113 µs median latency, and its noise reduction fell from 11.1 dB at 45 µs to 4.4 dB at 726 µs (Yuan et al., arXiv 2604.05519, Apr 2026, preprint) [V]. That work belongs on the headset's dedicated codec or DSP, not on a Jetson-class SoC.
- What this brief needs is **transmit-path adaptive noise cancellation**, the classic Widrow two-microphone configuration (Proc. IEEE 63(12), 1975) [V], combined with DNN speech enhancement. It cleans the talker's microphone signal before it reaches the radio or intercom, and the listener hears the enhanced speech. Its latency budget is measured in **milliseconds**.

D-ANC is a transmit-path system. Earcup acoustic ANC is complementary and out of scope (see §7).

**Targets and their exact meaning in this project.**

| Target | Definition used | Why |
|---|---|---|
| SNR > 15 dB | Output SNR = 10·log10(Σs² / Σ(s−ŝ)²) on the enhanced signal, i.e. non-scale-invariant SDR. ΔSNR and SI-SDR are also reported. | Penalises residual noise **and** speech distortion. SI-SDR (Le Roux et al., ICASSP 2019) [V] is reported alongside. |
| STOI > 0.85 | STOI (Taal et al., IEEE TASLP 19(7), 2011) [V], plus ESTOI (Jensen & Taal, TASLP 24(11), 2016) [V] | ESTOI is more sensitive to modulated maskers such as rotor and gunfire. |
| PESQ > 2.5 | **Wideband** PESQ, ITU-T P.862.2 (MOS-LQO range ≈1.04–4.64) [V]. Narrowband P.862 is also reported. | WB-PESQ is the stricter scale. The narrowband score is closer to an 8 kHz tactical-vocoder channel. ITU-T has marked P.862/P.862.2 out of date and points to P.863 (POLQA) [V], so PESQ is used here as the de-facto research metric the brief requests. |
| Latency | Algorithmic latency (window plus look-ahead), plus measured compute per 10 ms hop, plus I/O buffering | Two-way comms: ITU-T G.114 calls < 150 ms mouth-to-ear "essentially transparent" for most applications [V]. The DNS Challenge real-time track caps algorithmic latency at 40 ms [V]. The AEC Challenge 2023 used algorithmic + buffering ≤ 20 ms [V]. |

**The targets must be read against input SNR.** The literature shows that the targets are reachable near 0 dB input and above, but not at −5 dB with today's causal models:
- **Tan & Wang 2018, causal CRN** [V]: STOI 76.4 % and PESQ 2.04 at −5 dB on babble/cafeteria noise.
- **Tan & Wang 2020, GCRN** [V]: STOI 90.1 % and PESQ 2.65 at 0 dB; STOI 80.8 % and PESQ 2.13 at −5 dB.
- **Xu et al. 2015, unseen NOISEX-92 noise** [V]: PESQ 2.60 and STOI 0.876 at 5 dB, but only 1.89 and 0.688 at −5 dB.
- **Mukhutdinov et al., IEEE Access 2023** [V]: for drone ego-noise at −15 dB, the best of 12 DNNs reaches only PESQ 1.9 and ESTOI 0.4.

D-ANC therefore reports every metric **per input SNR and per noise category**.

**On "PESQ > 4".** The project owner asked for PESQ above 4. On wideband PESQ that has not been achieved on any public benchmark by any causal or non-causal system we could find:
- **Best non-causal systems on the comparatively mild VB-D test set (2.5–17.5 dB SNR)** [V]: CMGAN 3.41, MP-SENet 3.50, SEMamba 3.55 (3.69 with perceptual contrast stretching).
- **Best causal systems** [V]: BASENet-causal 3.44 (Thales, preprint, Jun 2026), FRCRN 3.21, DeepFilterNet3 3.17.

D-ANC reports where PESQ-NB and PESQ-WB exceed 4 by condition, and does not claim an average that the physics and the literature do not support. §8 has the details.

---

## 2. State of the art and the architecture decision

### 2.1 Evidence table: causal or real-time speech enhancement

| Model (venue, date) | Params / compute | Latency | Published quality *(test set — not defence)* | Licence | |
|---|---|---|---|---|---|
| RNNoise (IEEE MMSP 2018) | 87,503 weights | 20 ms window, 10 ms hop, 48 kHz | (DSP/DL hybrid) | BSD-3 | [V] |
| DeepFilterNet2 (IWAENC 2022) | 2.31 M / 0.356 GMAC | 40 ms (2-frame look-ahead) | VB-D PESQ 3.08 (3.03 with post-filter); RTF 0.04 on i5-8250U, 0.42 on Raspberry Pi 4 | MIT / Apache-2.0 | [V] |
| DeepFilterNet3 (Interspeech 2023 show & tell) | ≈2.1 M / 0.35 GMAC (per DPDFNet paper) | 40 ms | VB-D PESQ 3.17, STOI 0.944; RTF 0.19 single-thread i5-8250U | MIT / Apache-2.0 | [V] |
| GTCRN (ICASSP 2024) | 48.2 K / 33 MMAC/s | 0 look-ahead | VB-D PESQ 2.87, STOI 0.940, SI-SNR 18.83 | MIT | [V] |
| ULCNet (ICASSP 2024) | 688 K / 0.098 GMAC | 32 ms window | VB-D PESQ 2.87 (ULCNet_MS); RTF 0.127 on 1 Cortex-A53 core | (no public code found) | [V] |
| LiSenNet (ICASSP 2025) | 37 K / 56 MMAC/s | 0 look-ahead | VB-D PESQ 3.07 (with PESQ loss) | MIT | [V] |
| UL-UNAS (IEEE TASLP, 2026) | 171 K / 35 MMAC/s | 0 look-ahead | VB-D PESQ 2.96, or 3.09 with PESQ loss | MIT | [V] |
| FRCRN (ICASSP 2022) | 6.9 M (16 kHz) | 30 ms | VB-D PESQ 3.21 | Apache-2.0 | [V] |
| BASENet-causal (arXiv Jun 2026, Thales) | 0.81 M / 7.1 GMAC | not stated | VB-D PESQ 3.44 (preprint) | none | [V] |
| DPDFNet-8 (Speech Communication 2026, Ceva) | 3.5 M / 4.4 GMAC | ≈40 ms | own 0/5/10 dB multilingual set: PESQ 3.20 vs DFN3 2.76 vs GTCRN 2.21 (v3 values) | Apache-2.0 | [V] |
| Non-causal ceiling: CMGAN / MP-SENet / SEMamba | 1.8–2.3 M | offline | VB-D 3.41 / 3.50 / 3.55 | MIT | [V] |

### 2.2 Decision

The decision is a **DeepFilterNet/GTCRN/DPCRN-family hybrid, re-designed for 16 kHz defence voice**. It is not a time-domain model and not a generative one. The reasons:

1. **Compute fits the edge with margin.** The two-stage "envelope gain + deep filter" family reaches the best quality per MAC among causal models: DFN2/3 at about 0.35 GMAC/s, and GTCRN/UL-UNAS at tens of MMAC/s [V]. D-ANC's models need 0.37 GMAC/s (LL-small) to 0.76 GMAC/s (v2), counted analytically. The full engine block takes 0.5 ms per 10 ms frame on a single M4 core (2.1 ms mean in paced real time) [M].
2. **Complex domain and phase.** The output is a complex deep filter in the 0–3.15 kHz band, so phase is enhanced, not just copied from the noisy input. This band carries voiced-speech harmonics and also rotor, engine and track-slap lines. The research notes caution that DFN's deep filter was designed for periodic *speech*, and that its effect on periodic *noise* had to be measured [N]. §8 measures it.
3. **Sub-band and full-band processing, as the brief requires.** Sub-band feature unfolding and per-band temporal GRUs model local statistics. An intra-frame bidirectional GRU across frequency models global spectral structure, which is legal because it does not look into the future.
4. **Generative or diffusion enhancers are excluded for safety-critical voice.** The Interspeech 2025 URGENT challenge found that generative enhancers can hallucinate content and that DNSMOS rated hallucinated output highly [N]. A command channel must never invent words.
5. **16 kHz instead of 48 kHz.** Tactical radio vocoders are narrowband (MELPe at 2400 bit/s uses 22.5 ms frames at 8 kHz [V]). 16 kHz keeps wideband intelligibility cues up to 8 kHz at one-third of the compute of full-band models.

---

## 3. Component 1: Scalable dataset pipeline

### 3.1 Sources and licences, all fetched and hashed by `danc.data.download` (except the partial train-clean-100 stream, whose prefix hash was not recorded; see its provenance note)

| Corpus | Content used | Licence | Notes |
|---|---|---|---|
| LibriSpeech dev-clean, test-clean; Mini-LibriSpeech train-clean-5 | Clean speech | CC BY 4.0 [V] | Speaker-disjoint train (62 speakers, 9.9 h), val (6 speakers, 0.8 h), test (40 speakers, 5.4 h) |
| LibriSpeech train-clean-100, **partial** | +35 speakers, 15 h (v2 training set: 97 speakers, 25 h) | CC BY 4.0 [V] | Streaming stopped at about 1.0 GB because of disk space. One truncated file was excluded, not deleted. |
| NOISEX-92, 15 recordings (SPIB mirror) | F-16 and Buccaneer cockpits, Leopard and M109 vehicles, .50 cal machine gun, destroyer engine room and operations room, babble, factory, HF channel, Volvo | © TNO 1990, NATO RSG-10 distribution; **redistribution terms unclear** [V] | 235 s per type at 19.98 kHz, resampled to 16 kHz. Train on the first 75 %, test on the last 20 %. |
| ESC-50, 16 defence-relevant classes | Helicopter, siren, engine, airplane, wind, fireworks, chainsaw, train, thunderstorm, rain, crackling fire, hand saw, car horn, glass breaking, door knock, sea waves | **CC BY-NC 3.0, non-commercial** [V] | Folds 1–4 train, fold 5 test |
| DroneAudioDataset (Al-Emadi et al.) | Bebop and Mambo propeller noise | **No licence file** [V]; internal R&D only, not for redistribution | 80/20 hash split |
| VoiceBank+DEMAND test (16 kHz mirror) | 824 pairs; public benchmark only | CC BY 4.0 [V] | Never used for training |
| Synthetic generators, `danc.data.synth_noise` | 13 defence noise types (§3.2) | Project-generated | Disjoint seed streams for train and test |
| RIR bank (pyroomacoustics image-source method) | 300 train + 300 test RIRs, target RT60 0.1–0.8 s. Far-field median RT60 ≈ 0.50 s by T20 fit (0.37 s two-point estimate); near-field RIRs are dominated by the direct path (`scripts/rir_stats.py` → `reports/results/rir_stats.json`) | Project-generated | Near-field (talker 3–12 cm) and far-field (noise 1–6 m) |

**Data gaps, flagged rather than guessed.**
- No openly licensed corpus was found of artillery, armoured-vehicle or gunfire noise recorded at a headset microphone together with speech [N].
- The best open gunshot corpora are licensed but were **not** used here, because disk space and the user's choice limited downloads: the Kabealo & Wyatt edge-device gunshot set (CC BY 4.0, 1.57 GB) and the Free Firearm Sound Library (CC0) [N].
- DREGON drone data is academic-only.
- DroneAudioSet (NeurIPS 2025, MIT, 42.6 GB) is the most relevant open drone ego-noise corpus but is too large for this machine [N].

### 3.2 Physics-based synthetic defence noise (`synth_noise.py`)

The table below summarises the models. `reports/figures/synth_noise_grid.png` shows the spectrograms; wav examples are in `data/synthetic_examples/`.

| Kind | Model |
|---|---|
| gunshot_single / gunfire_burst / gunfire_sporadic | Modified-Friedlander muzzle blast with positive phase 0.5–3 ms, stretched with distance up to about 5 ms; Maher (IEEE SAFE 2007) notes muzzle blast typically lasts < 3 ms [V]. Optional supersonic N-wave arriving earlier; ground reflection; distance-dependent air-absorption low-pass; open, urban, forest or indoor echo tails. Automatic fire at 600–1000 rounds per minute. |
| artillery / explosion | Friedlander positive phase 5–50 ms (artillery) or 2–10 ms (close explosion); strong rumble below 250 Hz with 1–4 s decay; optional incoming whistle (artillery); debris crackle (explosion). |
| helicopter | Main-rotor blade-passing frequency 10–30 Hz with harmonics; the Black Hawk BPF is about 17 Hz [V]. Blade-vortex-interaction "slap" pulse per blade passage, tail-rotor BPF of 50–120 Hz, turbine tones from 1–6 kHz, rotor-modulated broadband noise, optional Doppler fly-by. |
| drone_multirotor | 4–8 rotors with BPF 80–400 Hz, RPM jitter, manoeuvre drift, 10–30 harmonics, motor whine and turbulence. |
| siren_wail / yelp / hilo | Odd-harmonic sweeps (wail 0.1–0.5 Hz, yelp 2–5 Hz) and two-tone alternation, with Doppler. |
| wind | Low-frequency turbulence with gust envelopes and pops. This is a simplified version of the Nelke & Vary (IWAENC 2014) approach [V]. |
| tracked_vehicle, jet_flyby | Diesel firing harmonics, track-slap impulses, gearbox whine; jet-mixing hump with Doppler fan tones. |

![Synthetic defence-noise generators: spectrograms of 6 s examples](figures/synth_noise_grid.png)

These generators add **parametric variety, not field realism**. Validating on recorded range and vehicle noise is listed as a required next step (§9).

### 3.3 Mixing recipe (`mixer.py`, dynamic mixing, so no mixture is ever reused)

- **Speech.** A 3 s crop for the v2 runs (4 s for the pilot and the validation set), set to an active level of −40 to −15 dBFS using a P.56-*style* frame-based active level. This is not a compliant P.56 implementation.
- **Noise category.** Stationary 25 %, non-stationary 35 %, impulsive 20 %, mixed "battlefield" 20 % (a stationary or non-stationary bed plus impulsive events; with p = 0.4 a third non-stationary layer such as rotor, drone or siren).
  - Within a category, sources are drawn family-balanced across NOISEX, ESC-50, drone and synthetic, so that 300 drone clips do not swamp 10 NOISEX files.
  - Impulsive scenes get a weak background bed with probability 0.6, because real fire never happens in silence.
- **SNR.** Uniform −10…20 dB for stationary and non-stationary noise, −10…15 dB for impulsive and mixed. SNR is active-speech power over whole-segment noise power at the microphone.
- **Augmentation.**
  - Near-field room reverb on speech (p = 0.25; the target keeps the direct path plus the first 50 ms).
  - Far-field reverb on noise (p = 0.35).
  - **ADC/pre-amp clipping** (p = 0.10; drive 1.2–4× full scale, hard or tanh). Firearm peaks of roughly 140–175 dB SPL [V] saturate headset microphones.
  - Microphone EQ (p = 0.2), applied to both input and target.
  - Noise-only examples (p = 0.02, target = silence) and clean-only examples (p = 0.02).
- **Dual-microphone scenes (`dual_mic.py`).** Primary boom mic plus an outward-facing reference mic 8–15 cm away.
  - 1–3 directional sources with fractional-delay propagation, early reflections and head or helmet shadowing.
  - A diffuse field with spherically-isotropic coherence sinc(2πfd/c) (Habets & Gannot 2007).
  - Talker leakage into the reference at −25…−12 dB, plus microphone self-noise.

### 3.4 Fixed evaluation sets (`build_testset.py`; FLAC with metadata, rebuildable from seeds)

- **defence_v1:** 900 mixtures of 6 s. Four categories × SNR {−5, 0, 5, 10, 15} dB × 40, plus a −10 dB stress condition with 4 × 25. Unseen speakers and test-split noise only.
- **dualmic_v1:** 240 headset scenes (primary, reference, clean). Four categories × SNR {−5, 0, 5, 10} dB × 15.
- **VoiceBank+DEMAND test:** 824 pairs, a published civilian benchmark used for comparison only.

---

## 4. Component 2: Model architecture (DANCNet)

```
 STFT frame X(t) [161 bins, 20 ms sqrt-Hann, 10 ms hop]
   │
   ├─ impulse-robust normaliser: divide by sqrt(median of last 64 frame energies)  ← a gunshot cannot move a median
   ├─ features: |X|^0.3 and compressed Re/Im  →  bins 0..63 kept (0–3.15 kHz), bins 64..160 → 32 ERB bands  (96 units)
   ├─ sub-band unfolding (each unit + 2 neighbours)  → 9 channels
   ├─ encoder: Conv(1×5,/2) → CausalConv(2×3,/2) → CausalConv(2×3)        [96 → 48 → 24 units]
   ├─ N × dual-path block:  BiGRU across frequency (full-band)  +  causal GRU across time (per sub-band)
   ├─ decoder: CausalConv + 2 × transposed conv with skip concatenation   [24 → 48 → 96]
   └─ heads (96 units):
        low band 0–3.15 kHz  → complex DEEP FILTER, N taps over frames t-N+1..t, output frame t-L
        high band 3.15–8 kHz → real ERB-band gains (sigmoid)
        all bands            → speech-presence probability (SPP), which gates the reference-mic NLMS
```

| Variant | Width / dual-path blocks | Deep-filter taps / look-ahead L | Algorithmic latency | Params | MAC per frame | GMAC/s |
|---|---|---|---|---|---|---|
| LL-small (pilot, design check) | 64 / 2 | 3 / 0 | 20 ms | 164 K | 3.67 M | 0.37 |
| **v2-HQ** | 80 / 3 | 5 / 2 | **40 ms** | 340 K | 7.64 M | 0.76 |
| **v2-LL** (warm-started from v2-HQ) | 80 / 3 | 5 / 0 | **20 ms** | 340 K | 7.64 M | 0.76 |

Deployed models: **HQ** = v2-HQ plus the robustness/PESQ-guided fine-tune (`exports/dancnet_hqft.onnx`); **LL** = v2-LL (`exports/dancnet_ll.onnx`). See §5 and §8.

**Design choices and how each serves the targets:**
- **Impulse-robust median normaliser.** A gunshot 40 dB above speech lasts a few frames. An exponential-mean normaliser, as in DFN, would be pulled up for hundreds of milliseconds and under-scale the speech that follows. A 0.64 s running median ignores events shorter than about 0.3 s. The state starts all-zero and is valid that way, which TensorRT and ONNX initialisation require.
- **Look-ahead L = 2 (HQ mode).** With look-ahead the network sees a blast onset 20 ms before it must emit that frame, so it can suppress the onset frame itself instead of reacting one frame late. The cost is 20 ms more latency, which is still within the DNS 40 ms real-time limit [V] and far inside G.114's 150 ms [V]. The 20 ms LL mode is kept for latency-critical use.
- **Deep filter (N complex taps) in the low band.** It models periodic components with sub-bin precision: voiced speech, and the rotor, engine and track-slap lines that the 50 Hz bin spacing cannot resolve. It is equivalent to DFN's deep filtering [V], with the trade-offs noted above.
- **SPP head.** Adds about 5 k MACs per frame (< 0.1 % of the network) and gives the adaptive filter a speech-aware step-size control (§6.2).
- **Exact streaming equivalence.** The network is written once, with explicit state tensors: convolution caches, GRU states, normaliser history and deep-filter history. Offline training and frame-by-frame streaming are numerically identical. The tests check this to ≤1e-4, PyTorch and ONNX Runtime both, including look-ahead [M].

---

## 5. Component 3: Training framework

| Item | Value | Basis |
|---|---|---|
| Loss | 1.0·L_cspec + 1.0·L_MR-STFT + 1.0·L_SI-SDR + 1.0·L_PMSQE + 0.5·L_asym + 0.1·L_SPP | See below. L_cspec has internal weights 30/70, and L_asym an internal ×100 scale (`losses.py`) |
| L_cspec | Power-law compressed (c = 0.3) complex and magnitude MSE, weighted 30·(Re/Im) + 70·mag | GTCRN HybridLoss [V]. Braun & Tashev found mixing compressed magnitude and complex losses best at β = 0.3 (with c = 0.3 adopted, not tuned) [V]. |
| L_MR-STFT | Spectral convergence + L1 on compressed magnitudes at n_fft 256/512/1024 | MR-STFT (Yamamoto et al. 2020) [N]; DFN uses a multi-resolution compressed loss [N]. Short windows matter for impulses. |
| L_SI-SDR | −SI-SDR in Bels, with silent targets masked | Le Roux et al. 2019 [V]; same scaling as the GTCRN code [V] |
| L_PMSQE (perceptual) | PESQ-inspired symmetric and asymmetric loudness disturbance on 32 ms frames | Martín-Doñas et al., IEEE SPL 25(11), 2018 [V]; Asteroid implementation vendored (MIT) |
| L_asym | Penalises only target-above-estimate compressed magnitude, i.e. speech removal | Asymmetric losses from VoiceFilter-Lite (Google, 2020) and PLCPA-ASYM (Microsoft, 2021) [N]. Protects STOI at very low SNR. |
| L_SPP | BCE against an IRM-style target, \|S\|²/(\|S\|²+\|N\|²) | Supervises the gating signal for the adaptive filter |
| Optimiser | HQ: AdamW, lr 1e-3, 1000-step warm-up, cosine decay to 2e-5, weight decay 1e-4, gradient clip 5, batch 16 × 3 s (pilot 4 s). LL warm start: lr 3e-4, 200-step warm-up, cosine to 1e-5. Fine-tunes: constant lr 5e-5 (D: 2e-4), gradient clip 3 | Matches DFN2/3 recipes (AdamW 1e-3, warm-up, cosine) [N] |
| EMA | Weight EMA including BatchNorm buffers, decay 0.999 (HQ) or 0.998 (LL and fine-tunes); used for validation, checkpoints and export | Standard practice |
| Model selection | Best validation PESQ + 2·STOI on 200 fixed val mixtures (6 held-out speakers, all categories, −5…15 dB) | |
| Perceptual fine-tune | MetricGAN+-style: a discriminator predicts true PESQ-WB; the generator pushes toward 1 while the full loss above stays as an anchor | MetricGAN+ (Interspeech 2021) reached VB-D 3.15 [V]. Adoption followed a **pre-registered rule** (`reports/results/finetune_decision_rule.md`): on the defence set ΔPESQ-WB ≥ −0.02, ΔSTOI ≥ −0.003 and ΔSI-SDR ≥ −0.2 dB; on VoiceBank, STOI up or PESQ +0.05; clean-input transparency not lower. |

**Runs actually performed** (logs in `logs/`, configs in `configs/`, curves in `reports/figures/training_curves.png`):

| Run | Initialisation | Data | Steps × batch × length | Wall time (M4 GPU) | Best validation PESQ-WB / STOI / SI-SDR |
|---|---|---|---|---|---|
| `pilot` (pipeline check, LL-small 164 k) | random | 10 h speech, no synthetic noise | 1 046 × 16 × 4 s | 0.25 h | 1.83 / 0.871 / 10.9 dB |
| `hq` (v2, 40 ms) | random | 25 h, 97 speakers + all noise incl. synthetic | 18 000 × 16 × 3 s | 5.7 h | 2.482 / 0.921 / 14.5 dB (step 16 000) |
| `ll` (v2, 20 ms) | warm start from `hq` | same | 5 000 × 16 × 3 s | 1.8 h | 2.348 / 0.912 / 13.8 dB |
| `hq_metric` (deployed HQ) | `hq` | + timbre/bandwidth aug. (p = 0.3), clean-only 8 %, PESQ discriminator (w = 2) | 2 000 | 1.1 h | final (deployed `last.pt`): 2.461 / 0.921 / 14.47 dB (start: 2.482) |
| `ll_metric` (not adopted) | `ll` | same | 2 000 | 1.1 h | final: 2.325 / 0.912 / 13.81 dB (start: 2.348) |

![Training loss and validation PESQ-WB / STOI for the HQ run and the LL warm start (LL plotted on its own step axis; the early HQ points reflect the EMA warm-up)](figures/training_curves.png)

**Observations.**
- **EMA lag.** EMA validation lags early in training; at step 2 000, 13 % of the random initial weights are still in the average. It is beneficial later.
- **Diminishing returns.** Validation gains after step 12 000 were small (+0.02 PESQ), so a longer run on this data alone would not change the conclusions.
- **Cost of look-ahead.** Removing look-ahead (LL) costs about 0.13 PESQ-WB and 0.75 dB SNR between the base models on the defence test set, and 0.11 PESQ-WB and 0.9 dB between the deployed models (Table R1).
- **The PESQ-guided stage did not raise in-domain PESQ.** Its measured value is robustness: VoiceBank +0.13 PESQ and +0.012 STOI, and clean-speech transparency 2.98 → 3.56 PESQ (Table R10). The fine-tuned models were adopted or rejected by a **pre-registered rule** (`reports/results/finetune_decision_rule.md`). HQ was adopted. LL was rejected because it missed the defence-PESQ guard by 0.003.

---

## 6. Component 4: Real-time inference engine

### 6.1 Streaming engine (`engine.py`, `runners.py`, `export_onnx.py`)

- **Per 10 ms hop:** shared STFT of both microphones → sub-band NLMS on the reference mic → one DANCNet step (ONNX) → inverse STFT → output peak limiter (−1 dBFS) for hearing protection and DAC headroom.
- **ONNX contract.** Input `spec [1,1,161,2]` plus 8 state tensors for the deployed v2 models (≈48–52 KB; 7 tensors / 32 KB for the 2-block LL-small). Outputs are `spec_out`, `spp` and the updated states. All states start at zero. The look-ahead and STFT parameters are stored in the ONNX metadata.
- **Runners.** PyTorch, ONNX Runtime (CPU EP, with CUDA/TensorRT EPs on Jetson) and TensorRT 10 (`deploy/jetson/trt_runner.py`, untested on hardware) share one interface.

### 6.2 How the learned model and the LMS filter work together

1. **Order: adaptive filter first, DNN second.**
   - A linear canceller fed from an outward reference microphone removes noise that is *coherent* between the two mics: low-frequency engine, rotor and vehicle noise from directional sources. It does so without distorting speech, which raises the SNR the network sees.
   - Coherence collapses for diffuse noise above roughly c/(2d), about 1.1–2 kHz for an 8–15 cm spacing. The DNN removes that diffuse, non-linear and impulsive residual.
   - Putting the DNN first would hand the LMS a non-linearly processed signal whose relation to the reference is no longer a linear transfer function.
2. **Shared STFT, zero extra latency.** The NLMS is per-bin, with 2 complex taps across frames (≈30 ms of reference-to-primary response) and μ = 0.1, working on the network's own analysis frames. The first, untuned version used 4 taps, μ = 0.3 and gate exponent 2 (§8.2).
3. **DNN-gated adaptation.** The step size is μ·(1−SPP_bin)·(1−SPP_global)⁴, where SPP_global is the mean speech presence over bins 6–70 (300–3500 Hz). This is the analogue of double-talk detection in echo cancellers and of DNN-based adaptation control (Haubner et al., ICASSP 2022 [V]; NKF [V]; Meta-AF [N]). Without the gate, talker leakage into the reference makes the LMS cancel speech. Widrow et al. showed that output SNR is then bounded by the inverse of the reference-mic SNR [V].
4. **Impulse robustness.** The update is frozen when reference power jumps by more than 20× (13 dB) over its 2 s average, and stays frozen while the impulse is in the tap line. Divergence guards reset the bin, and a per-bin output limit caps any reference-only transient at +6 dB above the primary in that bin.
5. **Graceful degradation.** The engine runs DNN-only when the reference channel is absent or silent, or with `--no-lms`. Other reference faults (hum, a loose connector) are contained by the divergence reset and output limiter, not detected explicitly. Explicit reference-health monitoring is a listed next step.

**Measured on Apple M4, ONNX Runtime 1.30 CPU, 1 thread, 6 000 frames** (`scripts/benchmark_latency.py`, `reports/results/latency_mac_m4.json`, `reports/figures/latency_hist.png`) **[M]**:

| Model | Algorithmic latency | Network step mean / p99.9 | Full engine block mean / p99.9 / max | RTF | Misses > 10 ms |
|---|---|---|---|---|---|
| HQ deployed (`dancnet_hqft.onnx`) | 40 ms | 0.42 / 0.47 ms | 0.50 / 0.61 / 0.75 ms | 0.050 | 0 |
| LL deployed (`dancnet_ll.onnx`) | 20 ms | 0.43 / 0.45 ms | 0.51 / 0.61 / 0.68 ms | 0.051 | 0 |
| LL-small (pilot architecture, 164 k) | 20 ms | 0.27 / 0.32 ms | 0.37 / 0.44 / 0.51 ms | 0.037 | 0 |

![Per-hop compute time of the full streaming engine, 6 000 hops, Apple M4, one thread](figures/latency_hist.png)

- **Paced real-time simulation.** Four dual-mic scenes were run through the deployed HQ engine with real-time pacing (`danc.inference.realtime --simulate --paced`): 2.0–2.2 ms mean, ≤ 4.3 ms max, 0 deadline misses (`reports/results/simulated_live_runs.json`). Blocks are slower when paced because the CPU idles between them.
- **ONNX parity.** The exported graphs match PyTorch to ≤ 1.3e-5 max absolute error on 300 frames including an impulse, and the streaming engine matches the offline model to ≤ 1e-4 (unit tests).
- **Mouth-to-radio latency budget, with the measured parts marked:**

| Component | Value |
|---|---|
| Algorithmic | 20 ms (LL) or 40 ms (HQ) **[M]** |
| Compute | 0.5 ms mean per hop unpaced; 2.1 ms mean, ≤ 4.3 ms max in paced real time (M4) **[M]**; absorbed in the 10 ms hop |
| Audio I/O | ADC/DAC + 2 × 10 ms ALSA periods, ≈ 25–40 ms **[E]**; to be measured with `measure_latency.py` |
| **Expected total** | **≈ 45–80 ms [E]**, within G.114's 150 ms "transparent" bound [V], before radio/vocoder delay |

### 6.3 Jetson AGX Orin deployment plan (`deploy/jetson/`, **not measured on hardware**)

- **Platform facts** [V]:
  - 275 sparse INT8 TOPS; 12-core Cortex-A78AE at 2.2 GHz; 2048-core Ampere GPU; 15–60 W.
  - nvpmodel modes: 15W (4 CPUs at 1.11 GHz), 30W (8 CPUs at 1.73 GHz, the default), 50W, and MAXN.
- **CPU path (recommended default).** The model is batch-1, single-frame and tiny, so GPU kernel-launch overhead dominates. NVIDIA quotes 5–15 µs per kernel launch [V]. The recommended path is ONNX Runtime on one or two pinned A78AE cores with SCHED_FIFO.
  - **[E]** An A78AE core is assumed 2–3× slower than an M4 performance core.
    - From the measured unpaced 0.5 ms per hop [M], that gives about 1–1.5 ms per hop.
    - The paced figures (2.1 ms mean, 4.3 ms max [M]) reflect idle-CPU clock and core scaling between blocks. Scaled the same way, they give about 4–6.5 ms mean and 8.5–13 ms worst case, and **the worst case could exceed the 10 ms deadline**.
    - Mitigation: performance governor, an isolated core with SCHED_FIFO (`setup_audio.sh`), two ORT threads, the 0.37 GMAC/s LL-small variant, or the TensorRT GPU path.
    - This must be measured with `trt_runner.py --onnx` and a 10-minute real-time run.
- **GPU path.** TensorRT FP16 with CUDA Graphs, using `build_trt_engine.sh`.
  - TensorRT loops (which is how GRUs are imported) support FP32 and FP16 only [V], so INT8 is a CPU-only option. If used, it should be ORT dynamic quantization, which ORT recommends for RNNs [V].
  - Published PTQ results: mixed FP16-INT8 cost 0.06 PESQ, uniform INT8 cost 0.3 [V].
  - DLA is not recommended: TensorRT 10.7 was the last release to support it [V].
- **No published per-frame latency for any speech-enhancement model on Jetson AGX Orin was found** [N]. The only Jetson audio-latency figure found was 80 ms of I/O latency without processing on a Xavier NX, from a robotics prototype [V]. I/O latency has to be measured with `measure_latency.py`.

---

## 7. Component 5: Prototype setup

**Bench prototype as built and validated in this project.** The complete live chain is implemented in `danc.inference.realtime`:
- **Capture.** A two-channel input stream: ch0 is the primary boom mic, ch1 the outward-facing reference mic.
- **Processing.** One `HybridEngine.process_block()` call per 10 ms PortAudio callback.
- **Output.** The same processed signal on both output channels. The headset and the radio line-out are taken from the audio interface's outputs.
- **Monitoring.** Per-block timing, xrun counts and JSON logging.

On this Mac it was validated in **paced simulated-live mode**, where the same callback is driven by a 10 ms software clock from recorded dual-mic test scenes (§8.5). Live microphones were **not** recorded in this session: no capture hardware is attached, and recording the room would raise privacy concerns.

**Hardware for the field prototype** (`deploy/jetson/README.md` §4):

```
 boom mic (primary, 2–3 cm from lips) ──┐
                                        ├─► 2-in/2-out USB audio interface ─USB─► Jetson AGX Orin (danc.inference.realtime, SCHED_FIFO, pinned cores)
 reference mic (outward, 8–15 cm) ──────┘          │
                                                   ├─► headphone out ─► headset earcups (listen-back / receive path)
                                                   └─► line out ─► pad + isolation transformer ─► radio MIC in; PTT via GPIO opto-isolator
 sidetone (own voice) stays analog and unprocessed; the earcup's acoustic ANC stays on the headset's own codec or DSP
```

**Integration points and why they are set up this way:**
- **Reference-mic placement.** It sits on the outside of the earcup or helmet, facing away from the mouth. Widrow's analysis gives output SNR ≤ 1/SNR_ref when the talker leaks into the reference [V], so leakage must be kept low, ideally below −15 dB.
- **Microphones with a high acoustic overload point.** Firearm peaks reach roughly 140–175 dB SPL [V]. Microphone clipping cannot be fully undone, so training includes clipping augmentation (§3.3).
- **Radio.** A level-matched line output, ground isolation, and PTT driven from GPIO or the radio's accessory connector. Tactical vocoders such as MELPe (2400 bit/s, 22.5 ms frames at 8 kHz) [V] add their own framing delay. STANAG 4591 also includes a mandatory noise pre-processor [N], and how D-ANC interacts with it must be tested (§9).
- **Latency test.** Electrical loop-back with `deploy/jetson/measure_latency.py`, first bypassed and then with the engine in the loop.
- **Power test.** `log_power.sh` reads the tegrastats rails in each nvpmodel mode.
- **Acceptance test.** Speech-intelligibility listening tests (Modified Rhyme Test, ANSI/ASA S3.2) through the full radio chain. MIL-STD-1474E §4.3 requires ≥ 80 % MRT (guessing-corrected) in worst-case noise [V]. MIL-STD-1472H Table XVII defines 91 % MRT as "normal acceptable" [V]. STOI and PESQ are engineering proxies; MRT is the military acceptance metric.

---

## 8. Results

All tables in this section are generated automatically from per-file results (`scripts/make_tables.py`), and `reports/results/tables.md` holds the same content. Per-file CSVs, which include every metric for every file and system, are in `reports/results/<set>/per_file.csv`. Enhanced audio is in `reports/audio/enhanced_<set>_<system>/` for the deployed HQ (`dancnet_hq_metric`) and LL (`dancnet_ll`), the base HQ, DeepFilterNet3 and the dual-mic hybrids. A listening set is in `reports/audio/demo/`.

### 8.1 Defence test set (single microphone)

![metrics vs SNR](figures/defence_v1_metrics_vs_snr.png)

#### Table R1: defence_v1, single microphone, mean over input SNR −5…15 dB (800 files per system)

| System | PESQ-WB | PESQ-NB | STOI | ESTOI | SNR out [dB] | ΔSNR [dB] | SI-SDR [dB] | n |
|---|---|---|---|---|---|---|---|---|
| Noisy input | 1.342 | 1.945 | 0.815 | 0.656 | 3.80 | 0.00 | 3.82 | 800 |
| Spectral subtraction | 1.406 | 2.027 | 0.812 | 0.663 | 5.59 | 1.79 | 4.99 | 800 |
| Wiener (DD) | 1.405 | 2.060 | 0.794 | 0.647 | 5.91 | 2.11 | 5.05 | 800 |
| Log-MMSE (LSA) | 1.437 | 2.090 | 0.798 | 0.653 | 5.94 | 2.14 | 5.13 | 800 |
| DeepFilterNet3 (public, 40 ms) | 2.043 | 2.754 | 0.885 | 0.779 | 11.83 | 8.03 | 11.70 | 800 |
| **D-ANC LL deployed (20 ms)** | 2.326 | 2.982 | 0.904 | 0.809 | 13.04 | 9.24 | 13.27 | 800 |
| D-ANC LL fine-tuned (not adopted) | 2.303 | 2.960 | 0.904 | 0.809 | 13.08 | 9.28 | 13.28 | 800 |
| D-ANC HQ base (40 ms) | 2.455 | 3.112 | 0.914 | 0.823 | 13.79 | 9.99 | 14.01 | 800 |
| **D-ANC HQ deployed (40 ms)** | 2.440 | 3.096 | 0.914 | 0.823 | 13.91 | 10.11 | 14.02 | 800 |

#### Table R2: defence_v1 by input SNR (−10 dB = stress condition, 100 files; others 160 files)


*PESQ-WB*

| System | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---|---|---|---|---|
| Noisy input | 1.085 | 1.112 | 1.135 | 1.258 | 1.452 | 1.753 |
| Log-MMSE (LSA) | 1.074 | 1.119 | 1.180 | 1.344 | 1.606 | 1.938 |
| DeepFilterNet3 (public, 40 ms) | 1.299 | 1.475 | 1.706 | 2.024 | 2.332 | 2.676 |
| **D-ANC LL deployed (20 ms)** | 1.416 | 1.627 | 1.885 | 2.293 | 2.723 | 3.101 |
| **D-ANC HQ deployed (40 ms)** | 1.495 | 1.714 | 2.009 | 2.435 | 2.849 | 3.193 |

*PESQ-NB*

| System | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---|---|---|---|---|
| Noisy input | 1.372 | 1.468 | 1.594 | 1.887 | 2.210 | 2.563 |
| Log-MMSE (LSA) | 1.333 | 1.507 | 1.694 | 2.045 | 2.417 | 2.787 |
| DeepFilterNet3 (public, 40 ms) | 1.663 | 2.039 | 2.398 | 2.808 | 3.143 | 3.379 |
| **D-ANC LL deployed (20 ms)** | 1.891 | 2.249 | 2.586 | 3.032 | 3.392 | 3.652 |
| **D-ANC HQ deployed (40 ms)** | 2.006 | 2.371 | 2.728 | 3.160 | 3.493 | 3.729 |

*STOI*

| System | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---|---|---|---|---|
| Noisy input | 0.577 | 0.665 | 0.740 | 0.836 | 0.891 | 0.942 |
| Log-MMSE (LSA) | 0.554 | 0.648 | 0.725 | 0.820 | 0.874 | 0.925 |
| DeepFilterNet3 (public, 40 ms) | 0.662 | 0.772 | 0.847 | 0.910 | 0.936 | 0.962 |
| **D-ANC LL deployed (20 ms)** | 0.724 | 0.808 | 0.870 | 0.924 | 0.946 | 0.971 |
| **D-ANC HQ deployed (40 ms)** | 0.747 | 0.827 | 0.883 | 0.933 | 0.951 | 0.974 |

*output SNR [dB]*

| System | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---|---|---|---|---|
| Noisy input | -11.18 | -6.20 | -1.26 | 3.87 | 8.81 | 13.78 |
| Log-MMSE (LSA) | -7.12 | -2.13 | 2.22 | 6.21 | 10.07 | 13.34 |
| DeepFilterNet3 (public, 40 ms) | 4.71 | 6.92 | 9.33 | 12.21 | 14.22 | 16.50 |
| **D-ANC LL deployed (20 ms)** | 5.39 | 7.78 | 10.18 | 13.35 | 15.62 | 18.26 |
| **D-ANC HQ deployed (40 ms)** | 5.94 | 8.39 | 10.94 | 14.22 | 16.49 | 19.53 |

#### Table R3: defence_v1 by noise category (−5…15 dB)

| System | stationary PESQ-WB / STOI / SNR | nonstationary PESQ-WB / STOI / SNR | impulsive PESQ-WB / STOI / SNR | mixed PESQ-WB / STOI / SNR |
|---|---|---|---|---|
| Noisy input | 1.27 / 0.787 / 3.8 | 1.30 / 0.810 / 3.9 | 1.52 / 0.856 / 3.7 | 1.28 / 0.807 / 3.7 |
| Log-MMSE (LSA) | 1.55 / 0.783 / 8.3 | 1.41 / 0.787 / 6.5 | 1.45 / 0.838 / 3.9 | 1.34 / 0.785 / 5.0 |
| DeepFilterNet3 (public, 40 ms) | 1.96 / 0.865 / 11.5 | 2.03 / 0.874 / 11.9 | 2.26 / 0.921 / 12.7 | 1.92 / 0.881 / 11.2 |
| **D-ANC LL deployed (20 ms)** | 2.18 / 0.883 / 12.1 | 2.26 / 0.890 / 12.5 | 2.69 / 0.940 / 14.7 | 2.17 / 0.902 / 12.8 |
| **D-ANC HQ deployed (40 ms)** | 2.28 / 0.896 / 12.8 | 2.35 / 0.900 / 13.3 | 2.82 / 0.947 / 16.0 | 2.30 / 0.912 / 13.6 |

#### Table R4: share of defence_v1 files meeting each target (−5…15 dB)

| System | SNR>15 dB | STOI>0.85 | PESQ-WB>2.5 | all three | PESQ-NB>4 | PESQ-WB>4 |
|---|---|---|---|---|---|---|
| Noisy input | 0.5% | 47.5% | 2.2% | 0.0% | 1.5% | 0.4% |
| DeepFilterNet3 (public, 40 ms) | 26.0% | 73.5% | 26.6% | 18.1% | 2.6% | 0.6% |
| **D-ANC HQ deployed (40 ms)** | 42.4% | 81.9% | 46.6% | 37.0% | 8.1% | 1.8% |
| **D-ANC LL deployed (20 ms)** | 35.4% | 79.8% | 40.6% | 30.5% | 5.0% | 1.2% |

#### Table R5: target check by input SNR, D-ANC HQ deployed (40 ms)

| input SNR | PESQ-WB | PESQ-NB | STOI | SNR out | meets SNR>15 | STOI>0.85 | PESQ>2.5 |
|---|---|---|---|---|---|---|---|
| -10 dB | 1.50 | 2.01 | 0.747 | 5.9 dB | no | no | no |
| -5 dB | 1.71 | 2.37 | 0.827 | 8.4 dB | no | no | no |
| 0 dB | 2.01 | 2.73 | 0.883 | 10.9 dB | no | yes | no |
| 5 dB | 2.44 | 3.16 | 0.933 | 14.2 dB | no | yes | no |
| 10 dB | 2.85 | 3.49 | 0.951 | 16.5 dB | yes | yes | yes |
| 15 dB | 3.19 | 3.73 | 0.974 | 19.5 dB | yes | yes | yes |

#### Table R5: target check by input SNR, D-ANC LL deployed (20 ms)

| input SNR | PESQ-WB | PESQ-NB | STOI | SNR out | meets SNR>15 | STOI>0.85 | PESQ>2.5 |
|---|---|---|---|---|---|---|---|
| -10 dB | 1.42 | 1.89 | 0.724 | 5.4 dB | no | no | no |
| -5 dB | 1.63 | 2.25 | 0.808 | 7.8 dB | no | no | no |
| 0 dB | 1.88 | 2.59 | 0.870 | 10.2 dB | no | yes | no |
| 5 dB | 2.29 | 3.03 | 0.924 | 13.4 dB | no | yes | no |
| 10 dB | 2.72 | 3.39 | 0.946 | 15.6 dB | yes | yes | yes |
| 15 dB | 3.10 | 3.65 | 0.971 | 18.3 dB | yes | yes | yes |

![examples](figures/defence_v1_example_spectrograms.png)

*Example spectrograms* (gunfire over a tracked vehicle at 0 dB; drone at 0 dB; destroyer-ops room, gunfire and babble at 5 dB; columns: noisy, clean, DeepFilterNet3, D-ANC LL, D-ANC HQ deployed). DeepFilterNet3 deletes whole speech segments under gunfire and under drone noise, at 3.0–3.5 s and 3.4–4.5 s respectively. Both D-ANC modes keep them, leaving faint residual rotor lines. This is consistent with D-ANC's higher STOI and ESTOI.

### 8.2 Dual-microphone hybrid (reference-mic NLMS + DNN)

![dual-mic](figures/dualmic_v1_metrics_vs_snr.png)

#### Table R6: dualmic_v1, primary + reference microphone (240 scenes, −5…10 dB)

| System | PESQ-WB | PESQ-NB | STOI | ESTOI | SNR out [dB] | ΔSNR [dB] | SI-SDR [dB] | n |
|---|---|---|---|---|---|---|---|---|
| Primary mic (unprocessed) | 1.230 | 1.781 | 0.798 | 0.617 | 2.50 | 0.00 | 2.58 | 240 |
| Log-MMSE (LSA) | 1.354 | 1.953 | 0.782 | 0.618 | 5.44 | 2.94 | 4.57 | 240 |
| NLMS only, mu=0.3 (untuned) | 1.141 | 1.567 | 0.755 | 0.565 | 1.96 | -0.54 | -1.37 | 240 |
| NLMS only (ref. mic) | 1.180 | 1.649 | 0.788 | 0.608 | 3.16 | 0.66 | 1.12 | 240 |
| Hybrid, NO SPP gate, untuned (mu=0.3, 4 taps) | 1.596 | 2.250 | 0.842 | 0.708 | 4.39 | 1.89 | 2.35 | 240 |
| Hybrid, NO SPP gate | 1.818 | 2.512 | 0.879 | 0.762 | 6.58 | 4.08 | 5.94 | 240 |
| DNN only (HQ base) | 2.167 | 2.869 | 0.902 | 0.800 | 12.12 | 9.62 | 12.28 | 240 |
| Hybrid, untuned coupling (mu=0.3, 4 taps, gate exp. 2) | 2.308 | 2.959 | 0.920 | 0.832 | 11.09 | 8.59 | 11.08 | 240 |
| Hybrid NLMS+DNN (HQ base) | 2.446 | 3.076 | 0.929 | 0.850 | 13.47 | 10.97 | 13.52 | 240 |
| DNN only (HQ deployed) | 2.151 | 2.846 | 0.901 | 0.800 | 12.07 | 9.57 | 12.16 | 240 |
| **Hybrid NLMS+DNN (HQ deployed)** | 2.425 | 3.051 | 0.928 | 0.849 | 13.38 | 10.88 | 13.44 | 240 |
| DNN only (LL deployed) | 2.047 | 2.736 | 0.891 | 0.784 | 11.44 | 8.94 | 11.58 | 240 |
| **Hybrid NLMS+DNN (LL deployed)** | 2.375 | 2.992 | 0.923 | 0.844 | 13.43 | 10.93 | 13.42 | 240 |

#### Table R7: dualmic_v1 by input SNR


*PESQ-WB*

| System | -5 dB | 0 dB | 5 dB | 10 dB |
|---|---|---|---|---|
| Primary mic (unprocessed) | 1.078 | 1.142 | 1.257 | 1.444 |
| NLMS only (ref. mic) | 1.113 | 1.156 | 1.214 | 1.236 |
| DNN only (HQ deployed) | 1.558 | 1.992 | 2.290 | 2.765 |
| **Hybrid NLMS+DNN (HQ deployed)** | 1.834 | 2.301 | 2.591 | 2.974 |
| DNN only (LL deployed) | 1.471 | 1.878 | 2.154 | 2.685 |
| **Hybrid NLMS+DNN (LL deployed)** | 1.774 | 2.227 | 2.521 | 2.977 |

*STOI*

| System | -5 dB | 0 dB | 5 dB | 10 dB |
|---|---|---|---|---|
| Primary mic (unprocessed) | 0.655 | 0.773 | 0.858 | 0.906 |
| NLMS only (ref. mic) | 0.708 | 0.783 | 0.831 | 0.829 |
| DNN only (HQ deployed) | 0.824 | 0.892 | 0.935 | 0.954 |
| **Hybrid NLMS+DNN (HQ deployed)** | 0.877 | 0.921 | 0.951 | 0.964 |
| DNN only (LL deployed) | 0.805 | 0.881 | 0.926 | 0.951 |
| **Hybrid NLMS+DNN (LL deployed)** | 0.865 | 0.916 | 0.948 | 0.963 |

*output SNR [dB]*

| System | -5 dB | 0 dB | 5 dB | 10 dB |
|---|---|---|---|---|
| Primary mic (unprocessed) | -5.00 | 0.00 | 5.00 | 10.00 |
| NLMS only (ref. mic) | 0.14 | 3.41 | 4.75 | 4.34 |
| DNN only (HQ deployed) | 7.68 | 10.64 | 13.11 | 16.86 |
| **Hybrid NLMS+DNN (HQ deployed)** | 9.96 | 12.67 | 14.51 | 16.38 |
| DNN only (LL deployed) | 7.10 | 10.01 | 12.53 | 16.11 |
| **Hybrid NLMS+DNN (LL deployed)** | 9.57 | 12.54 | 14.63 | 16.97 |

#### Table R8: share of dualmic_v1 scenes meeting each target

| System | SNR>15 dB | STOI>0.85 | PESQ-WB>2.5 | all three | PESQ-NB>4 | PESQ-WB>4 |
|---|---|---|---|---|---|---|
| Primary mic (unprocessed) | 0.0% | 42.9% | 0.4% | 0.0% | 0.4% | 0.0% |
| DNN only (HQ deployed) | 21.2% | 77.9% | 27.9% | 15.4% | 2.9% | 0.4% |
| **Hybrid NLMS+DNN (HQ deployed)** | 32.5% | 88.3% | 45.0% | 28.3% | 2.9% | 0.4% |
| DNN only (LL deployed) | 19.2% | 75.0% | 23.3% | 11.7% | 2.5% | 0.4% |
| **Hybrid NLMS+DNN (LL deployed)** | 31.7% | 87.9% | 41.2% | 26.7% | 3.3% | 0.4% |

**Findings.**
1. **NLMS alone barely helps.** It gains +0.7 dB output SNR on average, but lowers PESQ (1.18 vs 1.23), STOI (0.788 vs 0.798) and SI-SDR, and lowers SNR at ≥ 5 dB input. The talker leaks into the reference mic at −25…−12 dB, and part of the noise field is diffuse. Untuned (μ = 0.3) it is worse than doing nothing (−0.5 dB, SI-SDR −1.4 dB), which is Widrow's speech-cancellation effect [V].
2. **Without the speech-presence gate, the hybrid is far worse than DNN-only** (PESQ 1.82 vs 2.17, both HQ base).
3. **With the gate, and the coupling tuned on a separate validation set** (`dualmic_val`: μ = 0.1, 2 taps, gate exponent 4), the hybrid beats DNN-only on PESQ-WB and STOI at every input SNR, and on output SNR from −5 to 5 dB. At 10 dB the HQ hybrid's output SNR is 0.5 dB below DNN-only (16.4 vs 16.9 dB); the LL hybrid is higher at every SNR. It gains +0.27 PESQ-WB, +0.027 STOI and +1.3 dB SNR overall, and +2.3 dB SNR and +0.05 STOI at −5 dB.
4. With the reference mic, the 20 ms LL hybrid almost matches the 40 ms HQ hybrid (13.4 dB SNR for both).

*Tuning hygiene:*
- **Exploratory sweep.** It was run first on 48 scenes of the test set, before the problem of tuning on test data was recognised (`logs/tune_hybrid_on_test_subset_EXPLORATORY.out`).
- **First validation grid.** The setting was then selected on the validation scenes (`dualmic_val`: validation speakers, training-split noise). This grid varied μ, taps and a hard gate threshold, with the gate exponent fixed at 4 from the exploratory sweep.
- **Second validation grid.** The internal audit flagged the fixed exponent, so a second grid also varied it (2 vs 4).
- **Result.** Both validation grids select the deployed setting: μ = 0.1, 2 taps, exponent 4, no hard threshold (`reports/results/dualmic_val/hybrid_tuning*.json`). The exploratory test-subset optimum differed only by a hard threshold of 0.3, which costs 0.003 PESQ on validation.

### 8.3 Cross-corpus check: VoiceBank+DEMAND (civilian benchmark) and transparency

#### Table R9: VoiceBank+DEMAND test (824 files; civilian benchmark, not defence noise)

| System | PESQ-WB | PESQ-NB | STOI | ESTOI | SNR out [dB] | ΔSNR [dB] | SI-SDR [dB] | n |
|---|---|---|---|---|---|---|---|---|
| Noisy input | 1.971 | 2.946 | 0.921 | 0.787 | 8.45 | 0.00 | 8.45 | 824 |
| Log-MMSE (LSA) | 2.277 | 3.152 | 0.903 | 0.774 | 13.46 | 5.02 | 13.24 | 824 |
| DeepFilterNet3 (public, 40 ms) | 2.705 | 3.494 | 0.925 | 0.840 | 16.33 | 7.88 | 17.25 | 824 |
| D-ANC HQ base (40 ms) | 2.296 | 3.242 | 0.907 | 0.801 | 16.95 | 8.50 | 17.71 | 824 |
| **D-ANC HQ deployed (40 ms)** | 2.425 | 3.382 | 0.919 | 0.821 | 17.59 | 9.14 | 18.08 | 824 |
| **D-ANC LL deployed (20 ms)** | 2.214 | 3.129 | 0.904 | 0.793 | 16.13 | 7.69 | 17.05 | 824 |
| D-ANC LL fine-tuned (not adopted) | 2.352 | 3.287 | 0.917 | 0.815 | 16.77 | 8.33 | 17.62 | 824 |

#### Table R10: transparency (clean speech in → model → compared with the input)

| Model | VoiceBank clean PESQ-WB / STOI / SI-SDR | LibriSpeech clean PESQ-WB / STOI / SI-SDR |
|---|---|---|
| hq | 2.98 / 0.940 / 20.9 | 4.45 / 0.996 / 31.3 |
| hq_metric | 3.56 / 0.960 / 23.9 | 4.47 / 0.997 / 33.9 |
| ll | 2.66 / 0.935 / 19.9 | 4.41 / 0.996 / 28.1 |
| ll_metric | 3.27 / 0.957 / 22.8 | 4.46 / 0.996 / 30.2 |

**Interpretation.**
- On mild civilian noise the base models improved SNR and SI-SDR but *lowered* STOI below the unprocessed input at input SNR above about 5 dB, and were not transparent on clean VoiceBank speech (PESQ 2.98 on clean input).
- Diagnosis: VoiceBank talkers are darker (−24 dB vs −18 dB energy above 4 kHz) and their "clean" references carry a higher noise floor than LibriSpeech.
- The timbre/bandwidth and clean-only augmentation in the final stage fixed most of this: transparency PESQ 3.56 and STOI 0.960, VoiceBank STOI 0.919.
  - The deployed HQ still sits 0.002 below the unprocessed input on VoiceBank STOI overall (0.919 vs 0.921), and below it above 10 dB input.
- DeepFilterNet3 has the best VoiceBank PESQ, STOI and ESTOI; VCTK speech is part of its training data [N]. D-ANC has the best output SNR and SI-SDR there.
- **Lesson for fielding:** collect enrolment or validation speech from the actual headset microphones and the user population (§9, R1).

### 8.4 Ablations and design checks

| Question | Evidence | Answer |
|---|---|---|
| Does look-ahead help? | Table R1: HQ deployed vs LL deployed | +0.11 PESQ-WB, +0.01 STOI, +0.9 dB SNR for 20 ms more latency |
| Is speech-presence gating of the NLMS necessary? | Table R6 | Yes. Ungated hybrid: PESQ 1.82 vs 2.45 gated (HQ base) |
| Does NLMS tuning matter? | Table R6, `hybrid_tuning*.json` | μ 0.3 → 0.1, taps 4 → 2 and gate exponent 2 → 4 together: +0.14 PESQ, +2.4 dB SNR |
| Does the DNN beat classical enhancement? | Tables R1–R3 | Log-MMSE/Wiener/spectral subtraction give +2 dB SNR and *lower* STOI. The DNN gives +10 dB SNR and +0.10 STOI |
| Robustness fine-tune | Tables R9, R10 | Transparency and generalisation up; defence metrics unchanged (HQ adopted) |
| Hard cases (impulsive) | Table R3, figure | Impulsive scenes are the *best* category (HQ: PESQ 2.82, STOI 0.947, SNR 16.0 dB). The median normaliser, the look-ahead and the clipping augmentation target exactly these cases. |

**Not done, flagged:** isolated ablations of the median normaliser vs an exponential-mean normaliser, and of the deep filter vs a plain complex mask. Each needs a separate multi-hour training run. They are listed as next steps rather than claimed.

### 8.5 Prototype run (simulated live)

The deployed HQ engine processed four dual-mic scenes (impulsive 0 dB, non-stationary −5 dB, mixed 5 dB, stationary 0 dB) block by block on a real-time clock, with zero deadline misses. The listening set in `reports/audio/demo/` has, per scene, the clean reference, the noisy primary mic, and the HQ and LL hybrid outputs.

---

## 9. Risks and mitigations

| # | Risk to the targets or latency budget | Likelihood / impact | Mitigation (status) |
|---|---|---|---|
| R1 | **Domain gap.** Synthetic gunfire, artillery and rotor noise and 1990-era NOISEX recordings may not match real fielded noise, so metrics may drop on real recordings. | High / high | Collect a range-and-vehicle noise corpus with the target headset (dual-mic, calibrated SPL), fine-tune on it, and re-validate. Licensed open corpora exist: Kabealo & Wyatt edge gunshot recordings (CC BY 4.0) and DroneAudioSet (MIT) [N]. *Open.* |
| R2 | **Very low SNR (≤ −5 dB)**: targets not met (§8.1); the literature shows the same limit [V]. | Certain / high | Use hardware SNR gain first: a noise-cancelling boom mic (MIL-STD-1472H requires ≥ 10 dBA improvement [V]), a throat or in-ear sensor (Vibravox: 25.6 dB vs 0.1 dB SNR at 90 dB ambient [V]), and the reference-mic NLMS. Specify the targets per input-SNR band. *Partly addressed: hybrid NLMS + DNN implemented.* |
| R3 | **Microphone or ADC clipping on nearby gunfire.** Information is destroyed, not merely masked. | High / medium | High-AOP microphones, analog limiting before the ADC, clipping augmentation in training (implemented); the output limiter protects the listener (implemented). |
| R4 | **Speech distortion and over-suppression.** The STOI or SI-SDR penalty is larger than any PESQ gain. | Medium / high | Asymmetric over-suppression loss, a STOI-weighted model-selection score, and an optional comfort floor `--gain-floor-db` (implemented). Fine-tunes are adopted only under the pre-registered non-inferiority rule (SI-SDR may drop by at most 0.2 dB; §5, §8.3). |
| R5 | **Metric gaming.** PESQ-guided training raises PESQ but not real intelligibility. PESQ is withdrawn by ITU-T [V] and was not designed for DNN artefacts. | Medium / medium | Fine-tuning stays anchored by the full loss; STOI, ESTOI, SI-SDR and SNR are all reported; MRT listening tests are required for acceptance. |
| R6 | **Jetson latency or jitter**: GPU launch stalls, desktop interference, thermal throttling. | Medium / high | CPU path by default; SCHED_FIFO with isolated cores; desktop disabled; CUDA Graphs on the GPU path; p99.9 acceptance criterion. *Must be measured: no published Orin SE latency was found.* |
| R7 | **Reference-mic leakage or failure.** LMS cancels speech, or the reference is lost. | Medium / medium | SPP-gated adaptation, divergence reset and output limiting (implemented); DNN-only fallback (implemented). |
| R8 | **Look-ahead latency.** HQ mode adds 20 ms. | Low / medium | LL mode (20 ms) is kept as the latency-critical alternative; total budget analysis in §6. |
| R9 | **Licences.** ESC-50 is non-commercial, NOISEX-92 terms are unclear, the drone set is unlicensed. | Certain / medium for deployment | Legal review before shipping models trained on these data; retrain on cleared data (R1). The code is the project's own, except the PMSQE (MIT) and DFN3 reference (MIT/Apache-2.0) components. |
| R10 | **Vocoder interaction.** MELPe or CVSD noise pre-processors downstream may double-suppress or create artefacts. | Medium / medium | Evaluate through the actual vocoder chain (narrowband PESQ-NB and STOI after the codec, and MRT). *Open.* |
| R11 | **Development-machine limits.** Training on a fanless 16 GB laptop: limited steps, data and model size. | Certain / medium | A GPU-server run on LibriSpeech-960 + DNS noise + defence recordings is expected to add quality **[E]**. The training code is device-agnostic (set `device: cuda`), but download support and configs for those corpora still have to be written. |

---

## 10. Reproducibility

- **Environment.** Python 3.11.15 venv (`.venv`, created with `uv`). Versions are pinned in `requirements.txt`: torch 2.14.1, onnxruntime 1.30.0, numpy 2.4.6, pesq 0.0.4, pystoi 0.4.1. The DeepFilterNet3 reference uses an isolated package directory (`third_party/dfn_pkgs`, numpy 1.26), so the main venv was not modified.
- **Data provenance.** `data/manifests/provenance.json` records URLs, licences, SHA-256 hashes and the partial-download note. The speech manifests are `data/manifests/speech_*.json`. The test sets are rebuildable from fixed seeds (`danc.data.build_testset`).
- **Exact sequence used in this project:**

```bash
python -m danc.data.download --root data/raw
python -m danc.data.download --root data/raw --only librispeech_partial
python -m danc.data.build_testset                                   # defence_v1, dualmic_v1, dualmic_val
python -m danc.train.train --config configs/train_hq.yaml           # 5.7 h on M4
python -m danc.train.train --config configs/train_v2_ll.yaml        # 1.8 h, warm start from hq
python -m danc.train.finetune_metric --config configs/finetune_hq_metric.yaml   # adopted
python -m danc.train.finetune_metric --config configs/finetune_ll_metric.yaml   # not adopted
python -m danc.inference.export_onnx --ckpt checkpoints/hq/best.pt --out exports/dancnet_hq.onnx
python -m danc.inference.export_onnx --ckpt checkpoints/hq_metric/last.pt --out exports/dancnet_hqft.onnx
python -m danc.inference.export_onnx --ckpt checkpoints/ll/best.pt --out exports/dancnet_ll.onnx
python scripts/tune_hybrid.py dualmic_val && python scripts/tune_hybrid.py dualmic_val --with-gate-power
PYTHONPATH=third_party/dfn_pkgs python scripts/run_deepfilternet3.py --set defence_v1   # and --set vbdemand
python -m danc.eval.evaluate --set defence_v1 --systems noisy,specsub,wiener,logmmse,file:dfn3,dancnet:hq,dancnet:hq_metric,dancnet:ll,dancnet:ll_metric
python -m danc.eval.evaluate --set dualmic_v1 --systems primary,logmmse,nlms,dnn:hq,hybrid:hq,hybrid_nogate:hq,dnn:hqft,hybrid:hqft,dnn:ll,hybrid:ll
python -m danc.eval.evaluate --set vbdemand --systems noisy,logmmse,file:dfn3,dancnet:hq,dancnet:hq_metric,dancnet:ll,dancnet:ll_metric
python scripts/transparency_test.py --runs hq,hq_metric,ll,ll_metric
python scripts/benchmark_latency.py --frames 6000 --models hq,hqft,ll,pilot
python scripts/rir_stats.py
python scripts/make_tables.py > reports/results/tables.md
python scripts/make_figures.py --set defence_v1 --systems noisy,logmmse,file:dfn3,dancnet:ll,dancnet:hq_metric --runs hq,ll
python scripts/make_manifest.py                                     # DEPLOYMENT.json + MANIFEST.sha256
python -m pytest -q                                                 # 68 tests
```

  - The rows labelled "untuned" (μ = 0.3, 4 taps, gate exponent 2) in Table R6 were produced with the engine's earlier defaults, before the validation-tuned defaults were set. Reproduce them with `EngineConfig(lms_mu=0.3, lms_taps=4, gate_power=2.0)`.
  - The ad-hoc commands that computed the transparency figures were folded into `scripts/transparency_test.py` afterwards; it reproduces the same procedure.

- **Deployed models.** `exports/DEPLOYMENT.json` lists the ONNX files, checkpoints, SHA-256 hashes and the I/O contract.
- **Integrity.** `reports/MANIFEST.sha256` holds SHA-256 checksums of every project file except `.venv`, Python caches, `.git` and `third_party/dfn_pkgs`; verify with `shasum -a 256 -c reports/MANIFEST.sha256`.
- **Incidents, for transparency.**
  - The first full training run was stopped at step 1 300 because memory pressure filled the disk with swap. Worker counts were then reduced and noise stored as float16.
  - A first v2 run (`v2_hq`) was restarted after 600 steps with a schedule sized to the measured step time.
  - One evaluation race condition lost CSV rows; those systems were re-run and the evaluator now uses a file lock.
  - All logs are kept in `logs/`. **No file was deleted during the project.**

---

## 11. References (all verified unless marked)

Full verified statements, with URLs and checker comments, are in `docs/research/claims_verification.md`. Only the most relevant are listed here.

1. B. Widrow et al., "Adaptive noise cancelling: Principles and applications," *Proc. IEEE* 63(12):1692–1716, Dec 1975. [V]
2. J.-M. Valin, "A hybrid DSP/deep learning approach to real-time full-band speech enhancement," IEEE MMSP 2018 (arXiv 1709.08243). [V]
3. H. Schröter et al., "DeepFilterNet2: Towards real-time speech enhancement on embedded devices for full-band audio," IWAENC 2022 (arXiv 2205.05474, 11 May 2022). [V]
4. H. Schröter et al., "DeepFilterNet: Perceptually motivated real-time speech enhancement," Interspeech 2023 show & tell (arXiv 2305.08227, 14 May 2023). [V]
5. X. Rong et al., "GTCRN: A speech enhancement model requiring ultralow computational resources," ICASSP 2024, pp. 971–975. [V]
6. N. Shetu et al., "Ultra low complexity deep learning based noise suppression," ICASSP 2024, pp. 466–470 (arXiv 2312.08132). [V]
7. H. Yan et al., "LiSenNet," ICASSP 2025 (arXiv 2409.13285). [V]
8. X. Rong et al., "UL-UNAS," IEEE/ACM TASLP (arXiv 2503.00340, v2 1 Feb 2026). [V]
9. S. Zhao et al., "FRCRN," ICASSP 2022, pp. 9281–9285. [V]
10. R. Cao et al., "CMGAN," Interspeech 2022; Y.-X. Lu et al., "MP-SENet," Interspeech 2023; R. Chao et al., "SEMamba," IEEE SLT 2024. [V]
11. Martins Gomes & Capman (Thales SIX GTS), "BASENet," arXiv 2606.12662, Jun 2026 (preprint). [V]
12. Rika, Sapir, Gus (Ceva), "DPDFNet," Speech Communication 2026 (arXiv 2512.16420 v3). [V]
13. K. Tan & D. Wang, "A convolutional recurrent neural network for real-time speech enhancement," Interspeech 2018, pp. 3229–3233. [V]
14. K. Tan & D. Wang, "Learning complex spectral mapping with gated convolutional recurrent networks," IEEE/ACM TASLP 28:380–390, 2020. [V]
15. K. Tan, X. Zhang, D. Wang, "Deep learning based real-time speech enhancement for dual-microphone mobile phones," IEEE/ACM TASLP 29:1853–1863, 2021. [V]
16. Y. Xu et al., "Multi-objective learning and mask-based post-processing for DNN based speech enhancement," Interspeech 2015. [V]
17. D. Mukhutdinov et al., "Deep learning models for single-channel speech enhancement on drones," IEEE Access 11:22993–23007, 2023. [V]
18. M. Haubner, A. Brendel, W. Kellermann, "End-to-end deep learning-based adaptation control for frequency-domain adaptive system identification," ICASSP 2022. [V]
19. D. Yang et al., "Low-complexity acoustic echo cancellation with neural Kalman filtering," ICASSP 2023 (arXiv 2207.11388). [V]
20. H. Zhang & D. Wang, "Deep ANC: A deep learning approach to active noise control," *Neural Networks* 141:1–10, 2021. [V]
21. Yuan et al., "Active noise cancellation on open-ear smart glasses," arXiv 2604.05519, 2026 (preprint). [V]
22. J. Le Roux et al., "SDR – half-baked or well done?," ICASSP 2019, pp. 626–630. [V]
23. C. Taal et al., STOI, IEEE TASLP 19(7), 2011; J. Jensen & C. Taal, ESTOI, IEEE/ACM TASLP 24(11), 2016. [V]
24. ITU-T P.862 (2001) and P.862.2 (2005/2007), PESQ; marked out of date in 2024 in favour of P.863. [V]
25. J. M. Martín-Doñas et al., "A deep learning loss function based on the perceptual evaluation of the speech quality," IEEE SPL 25(11):1680–1684, 2018. [V]
26. S. Braun & I. Tashev, "A consolidated view of loss functions for supervised deep learning-based speech enhancement," TSP 2021 (arXiv 2009.12286). [V]
27. S.-W. Fu et al., "MetricGAN+," Interspeech 2021, pp. 201–205. [V]
28. C. K. A. Reddy et al., "ICASSP 2021 Deep Noise Suppression Challenge," ICASSP 2021 (40 ms real-time rule). [V]
29. R. Cutler et al., "ICASSP 2023 Acoustic Echo Cancellation Challenge," arXiv 2309.12553. [V]
30. ITU-T G.114 (05/2003), "One-way transmission time." [V]
31. NVIDIA: Jetson Orin product page; Jetson Linux r36.4 Developer Guide (power modes); TensorRT 10.x "Working with Loops"; TensorRT "Working with DLA"; TensorRT-RTX optimisation guide; JetPack 7.2 release page (2 Jun 2026). Accessed 2026-10-03. [V]
32. ONNX Runtime documentation, "Quantize ONNX models." [V]
33. M. Rusci et al., "Accelerating RNN-based speech enhancement on a multi-core MCU with mixed FP16-INT8 post-training quantization," ITEM @ ECML-PKDD 2022. [V]
34. I. Fedorov et al., "TinyLSTMs," Interspeech 2020. [V]
35. Kealey et al., "Real-time audio video enhancement with a microphone array and headphones," arXiv 2303.00949, 2023. [V]
36. R. C. Maher, "Acoustical characterization of gunshots," IEEE SAFE 2007, pp. 109–113. [V]
37. D. K. Meinke et al., "Prevention of noise-induced hearing loss from recreational firearms," *Seminars in Hearing* 38(4):267–281, 2017. [V]
38. F. G. Friedlander, *Proc. R. Soc. Lond. A* 186:322–344, 1946 (Friedlander waveform, as conventionally attributed). [V, corrected]
39. US Patent 5,310,137, "Helicopter active noise control system," 1994. [V]
40. C. M. Nelke & P. Vary, "Measurement, analysis and simulation of wind noise signals for mobile communication devices," IWAENC 2014. [V]
41. J. Hauret et al., "Vibravox," arXiv 2407.11828 (CC BY 4.0). [V]
42. MIL-STD-1474E (15 Apr 2015); MIL-STD-1472H (15 Sep 2020). [V]
43. RFC 8130 (2017), MELPe RTP payload; NATO STANAG 4591. [V]
44. V. Narain, R. Kant, B. P. Singh, "Artificial intelligence driven advances in noise cancellation: a comprehensive review on overcoming noise hazards in military operations," *Def. Sci. J.* 76(3):408–417, May 2026. [V]. Note: its "14–16 dB SNR gains" come from acoustic-ANC papers, and its "PESQ improvements exceeding 7 %" is a relative VB-D gain; neither is a defence-noise speech-enhancement output SNR [N].
45. R. K. Jaiswal et al., "Single-channel speech enhancement using implicit Wiener filter," *Int. J. Speech Technol.* 25(3):745–758, 2022. [V]
46. J. Timmermann, F. Ernst, D. Sachau, "Speech enhancement for helicopter headsets," POMA 52, 2023 (its PESQ figures could only be traced through ref. 44). [V, partial]
47. C. Valentini-Botinhao, VoiceBank+DEMAND, Edinburgh DataShare 10283/2791 (CC BY 4.0). V. Panayotov et al., LibriSpeech (CC BY 4.0). K. Piczak, ESC-50 (CC BY-NC 3.0). A. Varga & H. Steeneken, NOISEX-92 (*Speech Communication* 12(3), 1993). S. Al-Emadi et al., DroneAudioDataset (no licence). [V]
48. R. Scheibler et al., "Pyroomacoustics," ICASSP 2018 [N]; E. A. P. Habets & S. Gannot, "Generating sensor signals in isotropic noise fields," JASA 2007 [N].
