# T7 - Defence-specific prior work, standards and evaluation conditions

Status: IN PROGRESS (incrementally updated). Research date: 2026-10-03.
Scope: speech enhancement (SE) / noise cancellation for military, helicopter, cockpit, armoured-vehicle and firearm
environments; military standards; alternative sensors; DRDO work; realistic metrics at very low input SNR.

Legend: **fetched=Y** means the number was read on the primary source page/PDF during this session.
**fetched=N** means only seen in a snippet/secondary source; treat as unverified.

---

## 1. Narain, Kant, Singh (2026) DSJ review - what it says and what it cites

**Citation.** Vaibhav Narain (Thapar Institute of Engineering and Technology, Patiala), Ravi Kant and Bhanu Pratap Singh
(DRDO-Defence Electronics Applications Laboratory (DEAL), Dehradun). "Artificial Intelligence Driven Advances in Noise
Cancellation: A Comprehensive Review on Overcoming Noise Hazards in Military Operations". *Defence Science Journal*
76(3), May 2026, pp. 408-417. DOI 10.14429/dsj.21095. Received 23 May 2025, revised 30 Oct 2025, accepted 20 Apr 2026,
online 29 Apr 2026. Landing page: https://publicationsdrdo.in/index.php/dsj/article/view/21095 ;
PDF: https://publicationsdrdo.in/index.php/dsj/article/download/21095/8827/94468 (fetched=Y, full text read).

Nature of the paper: a narrative survey (26 references, literature "published from 2019 onward"). It contains no new
experiments. Its headline abstract claim is "SNR gains of up to 14 dB - 16 dB and PESQ improvements exceeding 7 %".

### 1.1 Numerical claims in the review, traced to cited sources

| # | Claim as stated in DSJ review | Ref no. in review | Original source cited | Our assessment (see later sections for verification) |
|---|---|---|---|---|
| R1 | Military aircraft noise can reach 130 dB; jets radiate at 2-5 kHz | [1] | Kuo et al., IJERPH 18(6):2982, 2021 (fighter pilots vs ground staff hearing) | Hearing-health source, not SE. Not re-verified here. |
| R2 | Tinnitus reported by 95 % of affected population; NIHL at 3, 4, 6 kHz | [2] | Lowe & Moore, JASA 150(2):1030-43, 2021 | Hearing-health; not re-verified. |
| R3 | Helicopter cabin noise "can reach over 100 dB"; aircraft "beyond 100 db" | none | uncited | Uncited in the review. Compare with MIL-STD-1474E / measured cabin data (Section 4). |
| R4 | Helicopter headset FPGA SE: PESQ 1.13 -> 2.08; "SNR enhancement of 4.43 dB over -3.88 dB" (Table 1 lists SSNR -3.88 -> 4.43, OQ 1.30 -> 2.62, LLR 1.01 -> 0.70, WSS 93.05 -> 41.42) | [6] | Timmermann, Ernst, Sachau, POMA vol. 52, 4 Dec 2023 (Table 1 labels it "2024") | Review conflates segmental SNR (SSNR) with SNR in the text. Year inconsistent (2023 vs 2024). Verify against original. |
| R5 | LMS extracted helicopter rotor noise ~12 km away; APSA to 8.5 km | [7] | Liu, Liu, Wei, ICICSP 2023, pp. 476-481 | Detection, not SE. |
| R6 | Implicit Wiener filter on Raspberry Pi: PESQ 2.621, LLR 0.677, CD 4.769, WSS 73.173 (helicopter, drone, exhibition noise at 0, 2.5, 5 dB SNR) | [8] | Jaiswal, Yeduri, Cenkeramaddi, Int. J. Speech Technology 25(3):745-758, Sep 2022 | Verify which noise/SNR the 2.621 refers to. |
| R7 | CRN-based ANC enhanced SNR "by 14.05 dB" (Table 2: "SNR: 14.0 dB -> 14.05 dB") | [9] | Biradar & Joshi, ICICT 2022, pp. 249-253 (Table 2 attributes it to "Akarsh et al. (2022)") | Internally inconsistent: Table 2 implies a 0.05 dB change, text says 14.05 dB gain; author attribution differs between table and reference list. ANC, not SE. |
| R8 | DSEGAN: PESQ +7.3 %, SSNR +18.2 % | [10] | Phan et al., "Improving GANs for speech enhancement", IEEE SPL 27:1700-1704, 2020 | Relative improvement on VoiceBank+DEMAND (not defence noise). Verify. This is the source of the abstract's "PESQ improvements exceeding 7 %". |
| R9 | CsNNet ANC on construction sites: average noise attenuation 14.07 dB | [16] | Mostafavi & Cha, Automation in Construction 151:104885, 2023 | ANC attenuation (acoustic), not SE output SNR. Construction-site noise. Likely the "16 dB" upper bound is not explicitly traced in the body. |
| R10 | Hamdan & Punjabi (2024) lightweight GAN, DEMAND, train SNR 15/10/5/0 dB, test 17.5/12.5/7.5/2.5 dB: "Mixed performance in PESQ, CSIG, CBAK, COVL" | none in ref list | Not in reference list | Uncited in the reference list. |
| R11 | Sule et al. 2023 selective noise cancellation: KNN precision 88 %, F1 91 %; LR 68 %/73 % | [18] | Sule et al., APSIT 2023 | Classification metrics, not SE. |

**Key reading of the review.** The abstract's "SNR gains of up to 14-16 dB" are drawn from *active noise control*
(acoustic anti-noise) papers (Biradar & Joshi 2022 CRN ANC; Mostafavi & Cha 2023 construction-site CsNNet), not from
speech-enhancement output-SNR measurements in defence noise. The "PESQ improvements exceeding 7 %" is a *relative*
improvement of DSEGAN over SEGAN on VoiceBank+DEMAND. The only defence-recorded data with SE metrics in the review
are Timmermann et al. (helicopter cabin, PESQ 1.13 -> 2.08 with classical Wiener filtering) and Jaiswal et al.
(helicopter/drone noise at 0-5 dB, IWF). None of the cited works reports STOI. Do not reuse "14-16 dB" as a target
justification for SE output SNR without this caveat.

### 1.2 Verification of the review's primary defence-relevant sources

**Jaiswal, Yeduri, Cenkeramaddi (2022)** - "Single-channel speech enhancement using implicit Wiener filter for
high-quality speech communication". Int. J. Speech Technology 25(3):745-758, 2022. DOI 10.1007/s10772-022-09987-4.
University of Agder (Norway). Open access CC-BY 4.0 (publisher version obtained via NVA/Brage record
https://hdl.handle.net/11250/3013314). **fetched=Y (full PDF read).**

- Data: only **two** clean utterances (one male, one female, ~3 s each), **narrow-band 8 kHz**, 16-bit PCM.
  Noise: AWGN; exhibition and station (NOIZEUS/Aurora); drone (drone audio dataset, Al-Emadi et al. 2019);
  **helicopter and airplane noise taken from ESC-50 (Piczak 2015)** - i.e. *not* recorded cabin noise.
  Input SNRs 0, 2.5, 5 dB only.
- The review's "PESQ 2.621, LLR 0.677, CD 4.769, WSS 73.173" is a single cell: **female speaker, helicopter noise,
  5 dB input SNR, IWF** (Table 9). At 0 dB helicopter the IWF gives PESQ 1.807 (male) / 2.054 (female); at 5 dB male
  2.338. Noisy (unprocessed) PESQ is not tabulated; the comparison is against spectral subtraction (SS).
- "Real-time on Raspberry Pi" is not demonstrated: algorithms were developed in MATLAB, the RPi 4B run is reported
  with "Code execution time 13 s" (Table 1) and CPU utilisation 27-28 %. No latency or real-time factor is given.
- Interpretation: classical IWF gives PESQ ~1.8-2.6 at 0-5 dB on 8 kHz speech; statistically weak (n=2 utterances).

**Timmermann, Ernst, Sachau (2023/2024)** - "Speech enhancement for helicopter headsets: Simulation and
implementation on an FPGA platform". *Proceedings of Meetings on Acoustics* (POMA) vol. 52, art. 055004, 185th Meeting
of the Acoustical Society of America, Sydney, 4-8 Dec 2023; DOI 10.1121/2.0001865; repository record says issued
2024-04-09 (Helmut-Schmidt-University / University of the Federal Armed Forces Hamburg, Mechatronics), companion abstract
DOI 10.1121/10.0023135. Metadata **fetched=Y** (openHSU DSpace API, Crossref). Full text: AIP site blocked (HTTP 403),
**numbers NOT verified at source** - they are known only via the DSJ review (Table 1): PESQ 1.13 -> 2.08, SSNR -3.88 ->
4.43 dB, overall quality (OQ) 1.30 -> 2.62, LLR 1.01 -> 0.70, WSS 93.05 -> 41.42; method = dual-microphone, dual-stage
basic spectral subtraction + Wiener filter with VAD, FPGA real-time; data = in-flight helicopter cabin recordings
(DSJ table says "H120 B") with reference speech from an anechoic chamber. The search-engine abstract confirms
"dual microphone dual stage" SS + Wiener filter, test-flight cabin recordings, FPGA lab validation.
- Interpretation: on real helicopter cabin noise, classical dual-mic processing reached PESQ ~2.1 from ~1.1.
  The input SSNR of -3.88 dB indicates very low-SNR operating conditions typical of helicopter cabins.

---

## 2. Defence noise corpora used in the SE literature

### 2.1 NOISEX-92 (military noise subset)

Source: Rice University SPIB / NOISEX-92 description, mirrored at https://spib.linse.ufsc.br/noise.html
(fetched=Y). Provenance: TNO Institute for Perception (NL) and Speech Research Unit, RSRE (UK), under NATO
AC243/(Panel 3)/RSG-10 and ESPRIT project 2589-SAM. Canonical paper: Varga & Steeneken, "Assessment for automatic
speech recognition: II. NOISEX-92...", *Speech Communication* 12(3):247-251, 1993, DOI 10.1016/0167-6393(93)90095-3.
All recordings 235 s, 19.98 kHz, 16-bit.

| Noise | Recording condition (as stated on SPIB page) | Level at recording |
|---|---|---|
| Buccaneer jet cockpit 1 | 190 knots, 1000 ft | 109 dBA |
| Buccaneer jet cockpit 2 | 450 knots, 300 ft | 116 dBA |
| F-16 cockpit | 500 knots, 300-600 ft (co-pilot seat) | 103 dBA |
| Destroyer engine room | - | 101 dBA |
| Destroyer operations room | - | 70 dBA |
| Leopard (military vehicle/tank) | 70 km/h | 114 dBA |
| M109 (tracked vehicle) | 30 km/h | 100 dBA |
| Machine gun | .50 calibre, repeated fire | not stated |
| Factory 1 / 2, babble (88 dBA), Volvo 340 car, HF radio channel, white, pink | - | - |

Caveats: single-channel, 19.98 kHz (resample to 16 kHz), only ~4 min per noise type, 1990-era platforms. No
helicopter, drone, siren, artillery or wind. Licensing/redistribution status of NOISEX-92 is not documented on the
SPIB page (open question).

---

## 3. Realistic SE performance at very low input SNR (-10..+5 dB)

All numbers below are on *simulated additive mixtures*; none uses real two-way defence radio chains. PESQ
variant (narrowband P.862 vs wideband P.862.2) is often not stated in older papers.

### 3.1 Anchor results (fetched=Y unless marked)

| Source | Noise / condition | Input SNR | Noisy STOI / PESQ | Enhanced STOI / PESQ | Notes |
|---|---|---|---|---|---|
| Xu, Du, Huang, Dai, Lee, Interspeech 2015 (arXiv 1703.07172) | **Unseen NOISEX-92: Buccaneer1, Destroyer engine, HF channel** (avg), TIMIT 16 kHz, DNN trained on 115 noise types, 80 h | -5 dB | 0.541 / 1.235 | 0.688 / 1.887 (proposed); DNN baseline 0.645 / 1.617 | LogMMSE: 0.525 / 1.261 |
| same | same | 0 dB | 0.656 / 1.482 | 0.796 / 2.261 | |
| same | same | 5 dB | 0.772 / 1.793 | 0.876 / 2.597 | First SNR at which both STOI>0.85 and PESQ>2.5 |
| Tan & Wang, Interspeech 2018 (CRN, causal, real-time) | Babble + cafeteria (Auditec), untrained speakers, WSJ0 16 kHz, 500 h training at -5..0 dB | -5 dB | 0.5786 / 1.52 | 0.7642 / 2.04 | |
| same | same | -2 dB | 0.6508 / 1.66 | 0.8331 / 2.33 | |
| Tan & Wang, IEEE/ACM TASLP 28:380-390, 2020 (GCRN) | Babble + cafeteria, untrained speakers | -5 dB | 0.5784 / 1.50 | 0.8075 / 2.13 (G=2) | ΔSNR "more than 12 dB" at -5 dB |
| same | same | 0 dB | 0.6980 / 1.80 | 0.9012 / 2.65 | |
| same | same | 5 dB | 0.8106 / 2.12 | 0.9439 / 3.0x | |

**Implication for targets (STOI>0.85, PESQ>2.5, output SNR>15 dB):**
- At 0 dB input, strong causal DNNs (GCRN 2020) reach STOI ~0.90 and PESQ ~2.6 on non-stationary babble/cafeteria,
  so the STOI and PESQ targets are achievable around 0 dB in matched-domain simulation.
- At -5 dB, published causal DNNs reach STOI ~0.69-0.81 and PESQ ~1.9-2.1. Neither target is met.
- Output SNR: GCRN's ΔSNR of >12 dB at -5 dB input implies an output SNR of roughly +7 dB, not >15 dB. An output SNR
  of 15 dB from a -5 dB input needs ~20 dB improvement; from -10 dB input ~25 dB.
- Therefore the targets should be specified as a function of input SNR (e.g. targets binding for input >= 0 dB,
  relaxed improvement-based targets such as ΔSTOI and ΔSI-SDR for -10..0 dB).

---

## 4. Military standards relevant to evaluation and requirements

### 4.1 MIL-STD-1474E "Design Criteria Standard - Noise Limits" (US DoD, 15 April 2015; supersedes 1474D of 12 Feb 1997)

Source PDF: https://everyspec.com/MIL-STD/MIL-STD-1400-1499/MIL-STD-1474E_52224/ (Distribution A). fetched=Y.

| Clause | Requirement (paraphrased; numbers verbatim) |
|---|---|
| 4.2.1 | Steady-state < 85 dBA and impulsive peak < 140 dBP at the ear (protected or unprotected) during normal operations |
| 4.2.3.1 | Limit A: SPL shall not equal/exceed 85 dBA; Limit B: 8-h TWA 85 dBA; Limit C: 8-h TWA 85 dBA at the ear with hearing protection/headsets |
| 4.2.3.2 | No steady-state exposure > 130 dBA after hearing protection |
| 4.2.3.3 | Whole-body: no steady-state > 150 dB overall SPL regardless of protection |
| **4.3** | **Speech communication shall not be degraded below 80 % correct (guessing-corrected) MRT per ANSI/ASA S3.2, in representative worst-nominal-case ambient noise; applies to headsets** |
| 4.3.1 / Table I | Predictors AI/STI/SII may be used; predicted worst-case intelligibility must be >= the 85 % MRT level else MRT must be run. Table I equivalences: MRT 95 % = AI 0.59 / STI 0.57 / SII 0.66; 90 % = 0.50/0.48/0.56; **85 % = 0.42/0.41/0.49**; 80 % = 0.35/0.36/0.42 |
| Table I note 6 | Predictor-vs-MRT correlation "is variable (sometimes poor) with typical military communications systems in military noise environments" |
| 4.3.2 | Communication requirements based on 1/3-octave bands 160 Hz - 5 kHz |
| 4.3.3.1 | MRT scoring corrected for guessing: % correct = 2.4 x (number correct) - 20 (50-word list) |
| 4.3.3.4 | MRT may use emulated noise if each octave band 125-4000 Hz is within 3 dB and overall within 1 dB; all real comms hardware must be in the path; simulated hardware only if validated |
| App. A Table A-I | Category A: SPL >= 100 dBA, no direct voice comms required; **Category B: < 100 dBA, electrically-aided comms via attenuating helmet or headset required**; C < 90; D < 85; E < 75; F < 65 dBA |

### 4.2 MIL-STD-1472H "Design Criteria Standard - Human Engineering" (US DoD, 15 Sep 2020; supersedes 1472G w/Change 1, 17 Jan 2019)

Source PDF: https://everyspec.com/MIL-STD/MIL-STD-1400-1499/MIL-STD-1472H_57041/ (Distribution A). fetched=Y.

| Clause | Requirement |
|---|---|
| 5.3.6.1 / 5.3.6.2 | Microphones respond optimally 200-6,300 Hz; minimum acceptable band where narrower: 250-4,000 Hz |
| 5.3.6.4 | Predominantly low-frequency noise: 300 Hz high-pass filtering shall be used |
| **5.3.6.5** | In very loud low-frequency noise (100 dBA overall) noise-cancelling microphones shall be used, giving **>= 10 dBA improvement in peak-speech-to-RMS-noise ratio** versus non-noise-cancelling microphones |
| 5.3.6.7 | Peak clipping of 12-20 dB may be used where channel peak-speech-to-RMS-noise < 15 dB |
| 5.3.11.1 | MRT (ANSI/ASA S3.2) "shall be used" to measure most military comms systems; AI or STI as predictors |
| **5.3.11.2 Table XVII** | **MRT / AI criteria: Exceptionally high 97 % / 0.7; Normal acceptable 91 % / 0.5; Minimally acceptable for mechanized-equipment users 80 % / 0.4; Minimally acceptable 75 % / 0.3** |
| 5.3.12.1.2 | Comms system output >= 15 dBA above anticipated ambient noise |
| 5.3.12.1.4 | Output <= 115 dBP voice level at the ear |
| 5.3.12.2 | Receiver/headset response +/-3 dB from 250 to 6,000 Hz |
| 5.3.12.2.2.2 | Headset microphones in high ambient noise shall support active or passive noise reduction as needed to meet intelligibility requirements |

**Implication.** The governing military acceptance metric is a *listening test* (MRT per ANSI/ASA S3.2), not
PESQ/STOI. Our STOI/PESQ targets are engineering proxies; a defence acceptance plan should include MRT (or DRT)
with trained listeners through the full radio/vocoder chain, with 91 % MRT ("normal acceptable") or 80 % (minimum
for mechanised-equipment users / MIL-STD-1474E 4.3) as pass criteria. No published mapping from STOI to MRT for
military noise was found (open question).

### 4.3 ANSI/ASA S3.2 (current edition S3.2-2020, reaffirmed 2025 per ANSI webstore listing)

"Method for Measuring the Intelligibility of Speech over Communication Systems". Specifies talker/listener selection
and training, word lists (MRT, DRT, PB words), test design and scoring. Standard text is paywalled; only catalogue
descriptions seen (fetched=N for content). Referenced normatively by MIL-STD-1474E 4.3 and MIL-STD-1472H 5.3.11.1.1.
- MRT: 50 sets of 6 rhyming monosyllables (6-alternative forced choice), corrected for guessing (MIL-STD-1474E 4.3.3.1).
- DRT: 2-alternative initial-consonant pairs differing in one distinctive feature (secondary description; fetched=N).

### 4.4 STANAG 4591 (MELPe) - NATO 600/1200/2400 bit/s narrow-band voice coder

- RFC 8130 (Demjanenko & Satterlee, VOCAL Technologies, IETF Standards Track, March 2017), "RTP Payload Format for
  the MELPe Codec", https://www.rfc-editor.org/rfc/rfc8130.txt (fetched=Y): MELPe supports 2400, 1200, 600 bps;
  2400 bps uses a **22.5 ms frame of 180 samples at 8 kHz**, 16-bit; 1200 and 600 bps use 3 and 4 frames (67.5 / 90 ms);
  developed by ASPI, TI, SignalCom, Thales with **noise pre-processor contributions from AT&T**, under NSA/DoD contract;
  normative ref: NATO STANAG 4591, January 2006.
- STANAG 4591 ratification draft (Edition Y amendment W) posted by Compandent:
  https://www.compandent.com/wp-content/uploads/2021/01/STANAG4591-2006.pdf (fetched=Y, front matter only): "the
  STANAG contains the design requirements for a mandatory noise pre-processor (NPP)". Related: MIL-STD-3005 (MELP 2400).
  This is a draft posted by a vendor, not an official NATO release.
- **Implications:** (1) Our SE output will usually be band-limited to 8 kHz sampling (narrow-band) before a MELPe/
  CVSD/other tactical vocoder, so evaluate with narrow-band PESQ (P.862) and STOI at 8 kHz *after* the codec, not only
  wideband at 16 kHz. (2) The MELPe NPP is mandatory in-codec; cascading a DNN SE before the NPP should be tested for
  over-suppression and musical-noise interaction. (3) 22.5-67.5 ms codec framing adds to end-to-end latency
  independently of our SE latency.

---

## 5. Alternative sensors (throat, bone-conduction, in-ear) with DNNs

### 5.1 Vibravox dataset (ISL + Cnam)

Hauret, Olivier, Joubaud, Langrenne, Poirée, Zimpfer, Bavu. "Vibravox: A Dataset of French Speech Captured with
Body-conduction Audio Sensors". arXiv 2407.11828 (v1 16 Jul 2024, v4 27 Mar 2025); the authors state it is published
in *Speech Communication* (open access). Affiliations: Cnam LMSSC Paris; **Joubaud and Zimpfer: Department of Acoustics
and Soldier Protection, French-German Research Institute of Saint-Louis (ISL)**. Licence CC BY 4.0 (paper and corpus
on HuggingFace). https://arxiv.org/abs/2407.11828 ; https://vibravox.cnam.fr . fetched=Y.

- 188 participants, >45 h per sensor (45h37), 6 synchronous channels at 48 kHz/32-bit; ambient noise imposed by a
  high-order-ambisonics 3D spatializer.
- **Sensor SNR (Table II): signal-to-external-noise at 90 dB ambient: headset (cardioid boom) mic 0.1 dB; forehead
  accelerometer -3.5 dB; in-ear soft-foam mic 4.0 dB; in-ear rigid earpiece 3.9 dB; throat microphone (laryngophone)
  25.6 dB; temple pickup 8.9 dB.** This quantifies why throat mics are used in military/very-loud settings.
- Bandwidth extension (EBEN) on quiet speech (Table IV), STOI raw -> EBEN: throat 0.677 -> 0.834; rigid in-ear
  0.782 -> 0.877; soft in-ear 0.752 -> 0.868; forehead 0.731 -> 0.855; temple 0.602 -> 0.763. Authors note STOI and
  Noresqa-MOS "plateauing at 0.88 and 4.3".
- Implication: a throat/in-ear reference channel gives ~20-25 dB better SNR than an air mic at 90 dB ambient, but
  its own intelligibility ceiling (STOI ~0.83-0.88 after DNN bandwidth extension) is below clean air-mic speech.

### 5.2 Configurable EBEN (ISL)

Hauret, Joubaud, Zimpfer, Bavu. "Configurable EBEN: Extreme Bandwidth Extension Network to enhance body-conducted
speech capture". IEEE/ACM TASLP 31:3499-3512, 2023, DOI 10.1109/TASLP.2023.3313433; arXiv 2303.10008 (17 Mar 2023).
CC BY 4.0. fetched=Y (abstract): GAN with multiband decomposition + U-Net, "lightweight generator that allows
real-time processing".

(Sections below are being filled in.)
