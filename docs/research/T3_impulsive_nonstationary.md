# T3 - Impulsive and Rapidly-Changing Noise: Gunshots, Artillery, Rotors, Drones, Sirens, Wind

Research notes for the hybrid AI/ML adaptive noise cancellation (speech enhancement) system for defence two-way voice comms.
Compiled 2026-10-03. Status: IN PROGRESS (updated incrementally).

**Evidence convention.** `fetched=yes` means the page was actually retrieved and the number/statement was seen on it.
`fetched=no` means it came from a search snippet or a secondary source and must be re-verified before use.
All benchmark numbers below are from academic/non-defence test conditions unless stated; none of them is a defence-condition result.

---

## 1. Drone / rotor ego-noise (harmonic + non-stationary)

| # | Source | Authors / Org | Date / Venue | Key finding | Conditions | Fetched |
|---|---|---|---|---|---|---|
| D1 | DREGON: Dataset and Methods for UAV-Embedded Sound Source Localization - http://dregon.inria.fr ; https://members.loria.fr/ADeleforge/category/datasets/ | M. Strauss, P. Mordel, V. Miguet, A. Deleforge (Inria) | IEEE/RSJ IROS 2018 | 8-channel mic array embedded in a quadrotor UAV; annotated with 3D source position, motor speeds and IMU data. Licence: "free to use for academic and educational purpose" (dregon.inria.fr). | Localisation dataset, not an SE benchmark | yes (dataset pages); paper PDF on HAL blocked (403) |
| D2 | Audio-Based Search and Rescue with a Drone: Highlights from the IEEE Signal Processing Cup 2019 - https://arxiv.org/abs/1907.04655 (ar5iv HTML) | A. Deleforge, D. Di Carlo, M. Strauss, R. Serizel, L. Marcenaro | arXiv 1907.04655, 2019 (IEEE SP Cup 2019 @ ICASSP 2019) | Ego-noise described as "harmonic but also non-stationary" (motor speeds change rapidly for stabilisation). Cube-shaped 8-mic array, 44.1 kHz. SNRs: about -20 to 5 dB (static task), ~-15 dB (flight task). Rotor speeds were provided to participants. | DREGON-derived competition data | yes (ar5iv HTML) |
| D3 | Deep Learning Models for Single-Channel Speech Enhancement on Drones - https://ieeexplore.ieee.org/document/10061413/ , DOI 10.1109/ACCESS.2023.3253719 (abstract via DOAJ) | D. Mukhutdinov, A. Alex, A. Cavallaro, L. Wang (QMUL) | IEEE Access vol. 11, 2023 | Ego-noise gives "extremely low signal-to-noise ratios (e.g. SNR < -15 dB)" at onboard mics. 12 DNNs compared (TF-magnitude, TF-complex, time-domain; sequential, encoder-decoder, generative). Best: TF-complex UNet; at input SNR -15 dB ESTOI 0.1 -> 0.4, PESQ 1.0 -> 1.9, SI-SDR -15 -> 3.7 dB. | Drone ego-noise mixed with speech at -15 dB input SNR; single channel. Even the best model is far below the project target (PESQ > 2.5, STOI > 0.85). | yes (DOAJ API abstract) |
| D4 | Monaural speech enhancement on drone via Adapter based transfer learning - https://arxiv.org/abs/2405.10022 | X. Chen, H. Bi, W.-T. Lai, F. Ma | arXiv May 2024; IWAENC 2024 (Aalborg) | Frequency-domain bottleneck adapter fine-tunes a frozen pre-trained FRCRN on drone noise, avoiding full fine-tuning per drone type. | No numbers in abstract | yes (abstract) |

## 2. Wind noise

| # | Source | Authors / Org | Date / Venue | Key finding | Fetched |
|---|---|---|---|---|---|
| W1 | Measurement, Analysis and Simulation of Wind Noise Signals for Mobile Communication Devices - https://www.iks.rwth-aachen.de/publikationen/publications-detail//nelke14a/ | C. M. Nelke, P. Vary (IKS, RWTH Aachen) | IWAENC 2014, Sophia Antipolis, Sept 2014, pp. 328-332 | Wind-noise synthesis model: autoregressive model for spectral shape + Markov chain for temporal statistics; plus measured wind-noise database. | yes |
| W2 | IKS Wind Noise Database + Wind Noise Model (MATLAB v1.1) - https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/wind-noise-database/ | IKS, RWTH Aachen | n.d. (web page) | Indoor (constant and varying airflow, mic spacings 2 cm and 10 cm) and outdoor (normal/strong wind, dual-mic) recordings; 16 kHz WAV; MIT licence files in downloads. | yes |

## 3. DNN architectures relevant to periodic/harmonic noise

| # | Source | Authors / Org | Date / Venue | Key finding | Fetched |
|---|---|---|---|---|---|
| A1 | DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering - https://arxiv.org/abs/2110.05588 | H. Schroeter, A. N. Escalante-B., T. Rosenkranz, A. Maier (FAU / WS Audiology) | arXiv Oct 2021; ICASSP 2022 | Stage 1: ERB-scaled gains for the spectral envelope; stage 2: deep filtering "to enhance the periodic components of speech". NOTE: deep filtering in DFN targets periodic SPEECH (pitch harmonics), not periodic NOISE; the claim that it suppresses rotor harmonics must be tested. | yes |

(more sections to follow)

| A2 | A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech (PercepNet) - https://arxiv.org/abs/2008.04259 | J.-M. Valin, U. Isik, N. Phansalkar, R. Giri, K. Helwani, A. Krishnaswamy (Amazon) | arXiv Aug 2020; Proc. INTERSPEECH 2020 | Pitch-controlled acausal comb filter (M=5 periods per side, about -9 dB noise attenuation between harmonics) + 34 ERB-band gains + envelope postfilter. "operates on 10-ms frames with 40 ms of look-ahead" (40 ms includes the 10 ms overlap); 4.1% of an x86 core (5.2% of a 1.8 GHz i7-8565U). Reported PESQ-WB 2.54 vs RNNoise 2.29 and MOS 4.05 vs 3.70 (VCTK-based 48 kHz test); DNS overall MOS 3.52 vs 3.03 baseline. Transients such as stops are treated as part of the stochastic (non-periodic) component. | yes (arXiv PDF text) |

Implication for rotor/drone tonal noise: both DeepFilterNet (deep filtering) and PercepNet (pitch comb filter) are designed to preserve/reconstruct the harmonic structure of SPEECH. Rotor/engine tonal noise is also harmonic; if a rotor harmonic series is mistaken for voiced speech (e.g. tail-rotor or drone BPF ~60-200 Hz overlapping male F0), a pitch-driven comb filter could PASS noise. Untested inference - must be checked empirically.

---

## 4. Helicopter rotor noise - physics and cockpit communication

| # | Source | Authors / Org | Date / Venue | Key finding | Fetched |
|---|---|---|---|---|---|
| H1 | US Patent 5,310,137 "Helicopter active noise control system" - https://patents.google.com/patent/US5310137 | C. A. Yoerkie Jr., W. A. Welsh, T. W. Sheehy; assignee United Technologies (Sikorsky) | filed 16 Apr 1992, published 10 May 1994 | BLACK HAWK (4 main-rotor blades) blade passing frequency "approximately 17 Hz"; S-76 BPF "approximately 19 Hz". Cabin interior noise dominated by transmission gear-mesh tones (bull gear 778 Hz; hydraulic pump ~950 Hz; bevel gear ~1140 Hz; helical gear ~3950 Hz). | yes (Google Patents) |
| H2 | A Subjective Test of Modulated Blade Spacing for Helicopter Main Rotors - https://ntrs.nasa.gov/api/citations/20030015482/downloads/20030015482.pdf | B. M. Sullivan (NASA LaRC), B. D. Edwards (Bell), K. S. Brentner (Penn State), E. R. Booth Jr. (NASA LaRC) | AHS 58th Annual Forum, Montreal, June 2002 (NASA NTRS) | Evenly spaced rotor spectrum is "a series of well-defined tonal components at harmonically related frequencies" with fundamental at the blade passage frequency. Main rotors have "much lower blade passage frequencies" than tail rotors; UAV main-rotor BPFs are similar to helicopter tail rotors. | yes (PDF text) |
| H3 | A New Passive/Active Hybrid Headset for a Helicopter Application - https://iiav.org/ijav/content/volumes/4_1999_226821287074030/vol_2/292_firstpage_2398551287076578.pdf | M. Winberg, S. Johansson, T. Lago, I. Claesson (Univ. Karlskrona/Ronneby, Sweden) | Int. J. Acoustics and Vibration, vol. 4 no. 2, 1999 | Super Puma: noise up to 100 Hz dominated by tonal components, more broadband 100-400 Hz; "Helicopter noise consists of tonal components embedded in broadband noise." Hybrid feedforward digital + analog feedback ANC in headset, plus spectral subtraction on the intercom microphone signal. | yes (PDF text) |

Practical note: helicopter main-rotor BPF (~17-20 Hz) and its low harmonics lie largely below the telephone/wideband speech band and can be removed by a high-pass filter plus a narrow-band adaptive notch/LMS canceller referenced to a tachometer or reference microphone; the harder part for the DNN is the gearbox tones (hundreds of Hz to ~4 kHz, per H1) and blade-vortex interaction (BVI) impulsive noise (not quantified here; see open items).

---

## 5. Impulsive noise: gunshots and blasts - physical acoustics

| # | Source | Authors / Org | Date / Venue | Key finding | Fetched |
|---|---|---|---|---|---|
| G1 | Acoustical Characterization of Gunshots - https://nios.montana.edu/rmaher/publications/maher_ieeesafe_0407_109-113.pdf | R. C. Maher (Montana State Univ.) | IEEE SAFE 2007 Workshop, Washington DC, 11-13 Apr 2007, pp. 109-113 | Muzzle blast "typically lasts for less than 3 milliseconds". Supersonic bullets add a conical shock wave (Mach angle arcsin(1/M)); example .308 Winchester M = 2.54, Mach angle 23.2 deg. 48 kHz (20.8 us interval) adequate for classification, but "extremely rapid rise and fall times of the shock wave require higher sampling rates". Propagation effects can change levels by 20 dB or more over a few hundred metres. | yes (PDF text) |
| G2 | Determining the muzzle blast duration and acoustical energy of quasi-anechoic gunshot recordings - https://fdn.montana.edu/rmaher/publications/routh_maher_aes_0916.pdf | T. Routh, R. C. Maher (Montana State Univ.) | AES 141st Convention, Los Angeles, 29 Sep-2 Oct 2016 (convention paper) | Muzzle blast described as a Friedlander wave (over-pressure interval followed by under-pressure), after Hamernik and Hsueh. Recorded at 500 kHz with 20 dB attenuators at amp inputs to capture amplitude range. In a reverberant room, a Glock 19 shot shows "tens of milliseconds of decaying echoes". Durations vary by azimuth and shot-to-shot. | yes (PDF text) |
| G3 | Prevention of Noise-Induced Hearing Loss from Recreational Firearms - https://pmc.ncbi.nlm.nih.gov/articles/PMC5634813 | D. K. Meinke, D. S. Finan, G. A. Flamme, W. J. Murphy, M. Stewart, J. E. Lankford, S. Tasko | Seminars in Hearing, 2017 (PMC open access) | "Peak sound pressure levels (SPLs) from firearms range from ~140 to 175 dB"; most recreational firearms 150-165 dB peak; rifles 159-174 dB, pistols 148-171 dB, shotguns 152-170 dB. Outdoors the gunshot is "typically less than 10 milliseconds". Electronic level-dependent protectors restore audibility below ~85 dB SPL and limit output to 82-85 dB SPL. 140 dB peak limit used by OSHA/NIOSH/MIL-STD-1474D. | yes |

Engineering implication (inference, not from a source): a 140-175 dB SPL peak at the microphone exceeds typical consumer MEMS acoustic overload points by a wide margin, so the primary mic signal will clip during nearby gunfire; the DNN will see a clipped, broadband, <10 ms burst followed by reverberant tail. See section 6 for clipping/AOP.

