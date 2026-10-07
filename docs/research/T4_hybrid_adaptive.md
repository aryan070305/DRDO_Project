# T4 - Hybrid Adaptive Filtering (LMS/NLMS/FDAF/Kalman) + DNN with a Reference Microphone

Status: IN PROGRESS (incrementally updated). Research date: 2026-10-03.
Project context: defence two-way voice comms, primary + reference mic, Jetson AGX Orin, targets SNR > 15 dB, STOI > 0.85, PESQ > 2.5.

Evidence conventions:
- "Fetched = yes" means the page itself (or an official API record of it) was retrieved during this research and the number/statement was seen there.
- "Fetched = no" means seen only in a search snippet or secondary source; treat as unverified.
- Quotes are verbatim, at most 15 words.
- IMPORTANT: none of the benchmarks below are defence conditions (helicopter, gunfire, armoured vehicle). All metrics are on academic test sets and must be re-measured on project data.

---

## 1. Classical foundation: Widrow et al. 1975 adaptive noise cancelling (ANC in the "transmit-path" sense)

| Field | Value |
|---|---|
| Title | Adaptive noise cancelling: Principles and applications |
| Authors | B. Widrow, J. R. Glover, J. M. McCool, J. Kaunitz, C. S. Williams, R. H. Hearn, J. R. Zeidler, E. Dong Jr., R. C. Goodlin |
| Venue | Proceedings of the IEEE, vol. 63, no. 12, pp. 1692-1716, Dec. 1975, DOI 10.1109/PROC.1975.10036 |
| URL | https://ieeexplore.ieee.org/document/1451965 (IEEE Xplore page did not render for fetch); bibliographic record confirmed via Semantic Scholar API https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/PROC.1975.10036 |
| Fetched | Bibliographic data yes (S2 API: vol 63, pp 1692-1716; S2 "publicationDate 1975-03-24" is likely the receipt date, the issue is Dec 1975). Abstract not available via API. Stanford PDF (www-isl.stanford.edu/~widrow/papers/j1975adaptivenoise.pdf) downloaded but is a scanned image; could not be text-extracted here. |

Principle (from Wikipedia "Adaptive noise cancelling", fetched, secondary): primary input = signal + noise n_p; reference input = noise n_r correlated with n_p via unknown path; LMS filter on reference, output subtracted from primary; error = system output. Requirement stated on that page: target signal and corrupting noise must be uncorrelated with each other for all lags. If target signal leaks into the reference, the filter partially cancels the target (signal cancellation) - see Section 6.

## 2. Deep-learning-assisted adaptive filters (adaptation control, learned optimizers)

### 2.1 Meta-AF (Casebeer, Bryan, Smaragdis)
- arXiv 2204.11942, v1 2022-04-25, v2 2022-11-21; accepted to IEEE/ACM TASLP. https://arxiv.org/abs/2204.11942 (fetched)
- Learns adaptive-filter update rules by meta-learning/self-supervision; tasks: system ID, AEC, blind equalization, multichannel dereverberation, beamforming (GSC). Abstract: filters "operate in real-time".
- Code: https://github.com/adobe-research/metaaf (fetched). Core `metaaf` = University of Illinois Open Source License; `zoo` folder and weights = Adobe Research License (check for commercial/defence use). Related: Auto-DSP (WASPAA 2021), higher-order Meta-AF (IWAENC 2022).

### 2.2 Haubner, Brendel, Kellermann - DNN step-size control for FDAF
- "End-To-End Deep Learning-Based Adaptation Control for Frequency-Domain Adaptive System Identification", arXiv 2106.01262 (v1 2021-06-02, v3 2022-03-04), ICASSP 2022 Singapore. https://arxiv.org/abs/2106.01262 (fetched). DNN maps signal features to step-sizes; trained end-to-end on normalized system distance; robust to non-white, non-stationary high-level noise and abrupt changes.
- "End-To-End Deep Learning-based Adaptation Control for Linear Acoustic Echo Cancellation", arXiv 2306.02450 (2023-06-04), submitted to IEEE/ACM TASLP. https://arxiv.org/abs/2306.02450 (fetched). DNN infers step-size for LMS FDAF; broadband vs narrowband (per-band DNN) vs hybrid; small DNNs.

### 2.3 NKF - Neural Kalman Filtering for AEC
- "Low-Complexity Acoustic Echo Cancellation with Neural Kalman Filtering", D. Yang, F. Jiang, W. Wu, X. Fang, M. Cao. arXiv 2207.11388 (v1 2022-07-23, v2 2022-10-30). ICASSP 2023 (per GitHub README). https://arxiv.org/abs/2207.11388 (fetched); https://github.com/fjiang9/NKF-AEC (fetched; 16 kHz; trained on part of AEC Challenge corpus).
- Model size 5.3 K parameters, RTF 0.09 (abstract; hardware for RTF not stated in abstract).

## 3. Deep ANC (acoustic anti-noise) - Zhang & Wang

- "Deep ANC: A deep learning approach to active noise control", H. Zhang, D. Wang, Neural Networks vol. 141, pp. 1-10, 2021, DOI 10.1016/j.neunet.2021.03.037 (Europe PMC record fetched; PMC8328877).
- CRN estimates real/imag spectrograms of the canceling signal from the reference signal; addresses nonlinear ANC that LMS/FxLMS handles poorly; introduces "a delay-compensated strategy" for the ANC latency problem.
- NOTE: Deep ANC is acoustic ANC (anti-noise from a loudspeaker), NOT a transmit-path enhancement system. See Section 7.
- Follow-up: "A Deep Learning Method to Multi-Channel Active Noise Control", Interspeech 2021, pp. 681-685, DOI 10.21437/Interspeech.2021-1512 (ISCA archive fetched).

## 4. Dual-microphone DNN enhancement (Tan, Zhang, Wang)

- "Deep Learning Based Real-time Speech Enhancement for Dual-microphone Mobile Phones", K. Tan, X. Zhang, D. Wang, IEEE/ACM TASLP vol. 29, pp. 1853-1863, 2021 (first published 2021-05-21), DOI 10.1109/TASLP.2021.3082318. Europe PMC record + NCBI BioC full text fetched (PMC8224499).
- 16 kHz, 20 ms Hamming window, 50% overlap; 2.78 ms average processing per 20-ms frame on Intel Core i7-10510U laptop.
- Params: 290.44 K (causal DC-CRN) -> 103.07 K after structured pruning (P6); MACs 1.97 G -> 502.40 M per 4-s input.
- Mic spacing 0.1 m in main simulation; distances 0.01-0.15 m; head-shadow simulated by attenuating speech at secondary mic by 0 to -10 dB.
- Pruned causal model on simulated diffuse babble (Table I, extracted by fetch summarizer - verify against PDF): PESQ 1.23->2.56 at -5 dB, 1.57->2.74 at 0 dB, 1.88->2.86 at 5 dB, 2.19->2.95 at 10 dB; STOI 54.89->67.77 % at -5 dB, 66.48->76.48 % at 0 dB, 75.90->82.42 % at 5 dB, 83.54->87.50 % at 10 dB.
- Earlier conference version: Tan, Zhang, Wang, "Real-time speech enhancement using an efficient convolutional recurrent network for dual-microphone mobile phones in close-talk scenarios", ICASSP 2019 (fetched = no; seen in search results only).

(Sections 5+ to be added: coherence vs spacing, ANC latency, AEC challenge hybrids, Kalman AF, FDAF/PBFDAF, impulsive noise, recommendations.)

---

## 1b. Widrow et al. 1975 - verified content (OCR text layer of the Stanford-hosted PDF)

Source PDF: https://www-isl.stanford.edu/~widrow/papers/j1975adaptivenoise.pdf (fetched; text extracted from its OCR layer locally). Proc. IEEE, Dec. 1975.

| Point | What the paper says (paraphrase + short verbatim) |
|---|---|
| Reference sensor placement | Reference derived from sensors "located at points in the noise field where the signal is weak or undetectable" |
| Leakage of signal into reference (Sec. V) | Signal components in the reference "will cause some cancellation of the primary input signal". |
| Output SNR with leakage (key result) | With unconstrained Wiener solution and correlated noises, output SNR density "is simply the reciprocal at all frequencies" of the reference-input SNR density. I.e. if the reference mic has SNR of -10 dB, the best achievable output SNR is about +10 dB, regardless of primary SNR. |
| Signal distortion | Low distortion requires high SNR at primary and low SNR at reference (Eq. 44). |
| Uncorrelated noise limit | Noise reduction "is limited by the uncorrelated-to-correlated noise density ratios" at primary and reference inputs. (This is the coherence limit; see Section 5.) |
| Other limits | Finite filter length (App. B) and "misadjustment" caused by gradient-estimation noise. |
| Speech experiment (Sec. D, cockpit simulation) | Person speaks into mic B in a room with strong interference (triangular wave, rich harmonics); reference mic D away from talker; 16 hybrid analog weights; adaptation rate about 5 kHz; convergence after about 5000 adaptations ("one second of real time"); interference output power reduced "by 20 to 25 dB"; "No noticeable distortion"; convergence "on the order of seconds". Periodic interference - NOT broadband diffuse noise. |
| Adaptive line enhancer | When no external reference exists, a delayed copy of the primary can be used to cancel periodic interference (delay must decorrelate broadband signal). Relevant for rotor/engine harmonics. |

Implication for our system: a reference mic that picks up the talker (e.g. a second mic on the same headset boom or helmet) violates the core assumption; the LMS will cancel speech, and the output SNR is bounded by 1/SNR_ref. Adaptation must be frozen during speech (Section 6).

## 7. Acoustic ANC (anti-noise) vs transmit-path noise cancellation - latency regimes

| Regime | Requirement | Source (fetched) |
|---|---|---|
| Acoustic feedforward ANC (anti-noise into ear) | Causality: canceling loudspeaker signal must be generated before the noise reaches it; frequency-domain (STFT) DNN ANC incurs algorithmic delay tied to frame length/shift - "considered as a shortcoming" | Zhang & Wang, "Deep MCANC: A deep learning approach to multi-channel active noise control", Neural Networks 158 (2023) 318-327 (received 13 Sep 2022; online 25 Nov 2022). https://par.nsf.gov/servlets/purl/10396859 (fetched, text extracted locally) |
| Acoustic ANC on wearable (2026 prototype) | Median mic-to-speaker latency 113 us, 45 us fixed compute; noise reduction degrades from 11.1 dB at 45 us to 4.4 dB at 726 us; mic-to-ear acoustic delay "typically less than a few hundred microseconds" | Yuan et al., "Active Noise Cancellation on Open-Ear Smart Glasses", arXiv 2604.05519v1, 2026-04-07 (preprint, not peer reviewed). https://arxiv.org/html/2604.05519v1 (fetched) |
| ANC codec hardware | ADAU1787: "5 us group delay (fS = 768 kHz)" analog-in to analog-out with FastDSP bypass | Analog Devices ADAU1787 product page https://www.analog.com/adau1787 - fetch TIMED OUT; number seen only in search snippet (fetched = no) |
| Transmit-path speech enhancement (DNS Challenge real-time track) | Total algorithmic latency (frame + stride + look-ahead) <= 40 ms; processing per frame < stride on Intel Core i5 quad-core 2.4 GHz | Microsoft, DNS Challenge ICASSP 2021 page https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2021/ (fetched) |
| Transmit-path echo cancellation (AEC Challenge 2023) | "Algorithmic latency + buffering latency <= 20ms" (halved from 40 ms in ICASSP 2022) | Cutler et al., "ICASSP 2023 Acoustic Echo Cancellation Challenge", arXiv 2309.12553 https://arxiv.org/html/2309.12553v1 (fetched) |
| End-to-end conversational delay | 400 ms one-way upper bound for network planning; below 150 ms mouth-to-ear "most applications ... will experience essentially transparent interactivity"; highly interactive tasks may be affected below 100 ms | ITU-T Rec. G.114 (05/2003) "One-way transmission time", in force; https://www.itu.int/rec/T-REC-G.114-200305-I/en (status fetched); text from copy at https://www.cs.columbia.edu/~andreaf/new/documents/other/T-REC-G.114-200305.pdf (fetched, text extracted locally) |

Conclusion: acoustic ANC is a tens-to-hundreds-of-microseconds problem solved on dedicated codec DSP / analog hardware; a Jetson-class GPU pipeline with 10-20 ms frames cannot do anti-noise. Our system is transmit-path noise cancellation (enhance the mic signal before the radio encoder), where a 10-40 ms algorithmic budget is acceptable within the G.114 150 ms mouth-to-ear budget (radio vocoder + network also consume budget). Deep ANC / Deep MCANC / HAD-ANC are acoustic ANC papers and should NOT be used as the architecture for our transmit path, though their FDAF+DNN coupling ideas transfer.
