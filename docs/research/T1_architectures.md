# T1 - Real-time / lightweight speech-enhancement architectures (SOTA through 2026)

Status: IN PROGRESS (written incrementally). Survey date: 2026-10-03.

## 0. Evidence legend and caveats

- **[F]** = number read on the fetched primary page in this session (arXiv abs page, arXiv HTML / ar5iv full text, official GitHub README, ISCA archive, vendor page).
- **[S]** = seen only in a search snippet or a secondary source; NOT verified against the primary source.
- **[U]** = unverified / open; do not build on it without checking.
- ar5iv/arXiv-HTML full texts were read via an automated page-to-text fetch; numbers were taken from the paper body/tables, but small transcription errors are possible. Re-check the PDF before quoting in a deliverable.
- **Defence caveat:** every metric below comes from public benchmarks (VoiceBank+DEMAND "VB-D", DNS Challenge test sets, WSJ0-2mix, etc.). None include helicopter rotor, gunshot, artillery, siren or drone noise at defence SNRs (often < 0 dB). Treat them as relative architecture-ranking signals, not field predictions.
- **PESQ flavour matters:** WB-PESQ (P.862.2) is typically ~0.3-0.5 lower than NB-PESQ (P.862) on the same audio. VB-D papers almost always report WB-PESQ ("PESQ"); DNS papers often report both. Our target "PESQ > 2.5" must be pinned to WB or NB in the spec.
- **Latency convention (DNS Challenge):** algorithmic latency = window (frame) length + any look-ahead (the DNS rule is phrased as frame T + stride Ts + look-ahead <= 40 ms for its real-time track; see C-NSNET-DNS rule below). Paper "latency" figures are not always defined the same way; compare carefully.

## 1. Master comparison table (causal / real-time-capable models)

| Model | Year / venue | Causal | fs | Window / hop | Algorithmic latency (as reported) | Params | Compute | Reported quality (test set) | License / repo | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| RNNoise | 2018, IEEE MMSP 2018 (arXiv 1709.08243) | Yes | 48 kHz | 20 ms window, 10 ms hop, 22 Bark bands | paper: "latency low enough for video-conferencing" (exact ms not quoted in fetched text) | 87,503 weights | ~40 MFLOPS total; ~1.3 % of one x86 core | VB-D WB-PESQ 2.29 (as re-reported in UL-UNAS paper), 2.33 (LiSenNet paper) | BSD-3-Clause, github.com/xiph/rnnoise | [F] |
| PercepNet | 2020, Interspeech 2020 | Yes (40 ms look-ahead) | 48 kHz | 10 ms frames, 34 ERB bands, Vorbis window 50 % overlap | 40 ms look-ahead | 8M weights (8-bit) | ~800 MMACS; 4.1 % of x86 core (5.2 % on i7-8565U mobile) | DNS-2020 P.808 overall MOS 3.52 vs baseline 3.03; ranked 2nd in real-time track; VB-D PESQ 2.73 (re-reported by UL-UNAS) | No official open code from authors [U] | [F] |
| NSNet2 | 2021, ICASSP 2021 (Braun et al.) | Yes | 16 kHz | 20 ms sqrt-Hann, 50 % overlap, 320-pt FFT | depends only on STFT window (20 ms) | not stated in fetched text [U] | FC-GRU-GRU-FC-FC-FC (GRU 400, FC 600) | DNSMOS only (DNS-2020 300 real + 300 synthetic) | DNS-Challenge repo (MIT?) [U] | [F] arch, [U] params |
| DTLN | 2020, Interspeech 2020 | Yes | 16 kHz | 32 ms frame, 8 ms shift | 32 ms input-output delay | < 1M | 0.65 ms/frame i5-6600K; 2.2 ms/frame RPi 3B+ (TF-lite quant) | DNS no-reverb synth: PESQ 3.04, STOI 94.76 %, SI-SDR 16.34 dB (README; PESQ flavour not stated); 8th/17 DNS RT track | MIT, github.com/breizhn/DTLN | [F] |
| DCCRN (-E / -CL) | 2020, Interspeech 2020 | Yes | 16 kHz | 25 ms win, 6.25 ms hop, 512 FFT | 6 frames look-ahead = 37.5 ms (paper) | 3.7M | ~3.12 ms/frame (DCCRN-E); 14.38 GMAC/s (as re-measured in LiSenNet paper) | DNS-2020 no-reverb synth PESQ 3.266 (DCCRN-E); DNS blind MOS 3.42 all / 4.00 no-reverb; 1st in DNS RT track | (official code via authors; check) [U] | [F] |
| DPCRN | 2021, Interspeech 2021 (DNS3) | Yes | 16 kHz | 25 ms win, 12.5 ms hop, 400 FFT | 37.5 ms total | ~0.8M | ~7.45 GFLOPS; 8.9 ms/frame TF on i5-6300HQ | DNS3 blind P.808 MOS 3.57 (vs NSNet2 3.07), 3rd in wide-band track; DNSMOS 3.472 on dev set | [U] | [F] |
| FullSubNet | 2021, ICASSP 2021 | Yes (32 ms look-ahead) | 16 kHz | 32 ms Hann, 16 ms hop | 2-frame (32 ms) look-ahead + 32 ms window | 5.6M | 30.73 G MAC/s (Fast FullSubNet paper) | DNS-2020 synth no-reverb: WB-PESQ 2.777, NB-PESQ 3.305, STOI 96.11, SI-SDR 17.29 | MIT, github.com/Audio-WestlakeU/FullSubNet | [F] |
| FullSubNet+ | 2022, ICASSP 2022 | Yes (32 ms look-ahead) | 16 kHz | 32 ms / 16 ms | 32 ms look-ahead | 8.67M | not stated | DNS-2020 no-reverb: WB-PESQ 2.982, NB 3.504, STOI 96.69, SI-SDR 18.34 | (authors' repo) [U] | [F] |
| Fast FullSubNet | 2022 arXiv (2212.09019) | Yes (32 ms look-ahead) | 16 kHz | 32 ms / 16 ms | 32 ms look-ahead | 6.84M | 4.12 G MAC/s; RTF 0.082 | DNS-2020 no-reverb: WB-PESQ 2.808, NB 3.353, STOI 96.11, SI-SDR 16.98 | MIT (FullSubNet repo) | [F] |
| FRCRN | 2022, ICASSP 2022 (DNS4) | Yes (causal convs) | 16 kHz WB / 48 kHz FB | 20 ms win, 10 ms hop | 30 ms | 6.9M (WB), 10.27M (FB) | 12.30 GMAC/s (FB) | VB-D WB-PESQ 3.21, CSIG 4.23, CBAK 3.64, COVL 3.73; DNS-2020 WB-PESQ 3.23, NB 3.60, STOI 97.69, SI-SNR 19.78; 2nd in DNS4 RT full-band | Apache-2.0 (ClearerVoice-Studio) | [F] |
| DeepFilterNet (v1) | 2022, ICASSP 2022 | Yes | 48 kHz | 20 ms FFT (960), configurable 5-30 ms | min 5 ms (+ look-ahead) | 1.78M | 0.348 G MAC/s | VB-D WB-PESQ 2.81, SI-SDR 16.63 | MIT / Apache-2.0, github.com/Rikorose/DeepFilterNet | [F] |
| DeepFilterNet2 | 2022, IWAENC 2022 | Yes (2-frame look-ahead) | 48 kHz | 20 ms, 50 % overlap | 40 ms overall | 2.31M | 0.356 G MAC/s; RTF 0.04 (i5-8250U), 0.42 (RPi 4) | VB-D PESQ 3.08, STOI 0.943, CSIG 4.30, CBAK 3.40, COVL 3.70; DNS4 blind DNSMOS SIG 4.196 / BAK 4.427 / OVL 3.882 | MIT / Apache-2.0 | [F] |
| DeepFilterNet3 | 2023, Interspeech 2023 (show & tell) | Yes (2-frame look-ahead) | 48 kHz | 20 ms / 10 ms | 40 ms overall | not stated in paper | RTF 0.19 single-thread i5-8250U | VB-D PESQ 3.17, STOI 0.944, CSIG 4.34, CBAK 3.61, COVL 3.77 | MIT / Apache-2.0 | [F] |
| GTCRN | 2024, ICASSP 2024 | Yes | 16 kHz | 32 ms / 16 ms (per UL-UNAS setup; GTCRN paper not fetched) | 0 look-ahead | 48.2K (orig. reported 23.7K) | 33.0 MMAC/s (orig. 39.6 MMAC/s) | VB-D PESQ 2.87, STOI 0.940, SI-SNR 18.83; DNS3 blind DNSMOS OVRL 2.70 | MIT, github.com/Xiaobin-Rong/gtcrn | [F] README |
| ULCNet | 2024, ICASSP 2024 | Yes | 16 kHz | 32 ms / 16 ms, 512 FFT | (window 32 ms) | 688K | 0.098 GMAC/s; RTF 0.127 on 1 core Cortex-A53 1.43 GHz | VB-D PESQ 2.87 (ULCNet-MS), DNS-2020 PESQ 2.64, SI-SDR 16.34 | (Fraunhofer IIS; no public code found) [U] | [F] |
| FSPEN | 2024, ICASSP 2024 (Samsung) | Yes [S] | 16 kHz [S] | [U] | [U] | 79K | 89 MMAC/s | VB-D PESQ 2.97 | No official code found [U] | [F] Samsung blog |
| LiSenNet | 2025, ICASSP 2025 | Yes | 16 kHz | 32 ms / 16 ms | 0 look-ahead | 37K | 56 MMAC/s | VB-D PESQ 3.07, STOI 0.939 (with PESQ loss; 2.95 without per UL-UNAS) | MIT, github.com/hyyan2k/LiSenNet | [F] |
| UL-UNAS | 2025 arXiv -> IEEE TASLP (v2 Feb 2026) | Yes | 16 kHz | 32 ms / 16 ms | 0 ms look-ahead | 171K | 35 MMAC/s | VB-D PESQ 2.96 / 3.09 (w/ PESQ loss), SI-SNR 18.25-18.48; DNS3 PESQ 2.30, DNSMOS OVRL 2.71 | github.com/Xiaobin-Rong/ul-unas | [F] |
| Fast-ULCNet | 2026 arXiv (2601.14925) | Yes | 16 kHz | 32 ms / 16 ms | - | 0.338M | 1.691 M MACs (per frame? see notes) | DNS-2020 no-reverb: PESQ 2.51, SI-SDR 15.99, OVRL 3.09 | [U] | [F] |
| CRN (Tan & Wang) | 2018, Interspeech 2018 | Yes | 16 kHz [U] | [U] | "no or low latency" | "much fewer" than LSTM (17.58M as re-implemented in LiSenNet paper) | 2.56 GMAC/s (LiSenNet re-impl.) | VB-D PESQ 2.54 (LiSenNet re-impl.) | no official code [U] | [F] abstract |
| Demucs (denoiser) | 2020, Interspeech 2020 | Yes (causal variant) | 16 kHz | 37 ms frame, 16 ms stride, 3 ms look-ahead | 40 ms total frame | [U] (H=48/H=64) | RTF 0.6 (H=48, 4 thr), 1.05 (H=64) on quad-core i5 | VB-D causal H=64: PESQ 2.91, STOI 95, CSIG 4.20, CBAK 3.26, COVL 3.51; DNS blind MOS 3.6 (no reverb) | CC-BY-NC 4.0 (non-commercial!), archived 31 Oct 2023 | [F] |
| Conv-TasNet | 2019, IEEE/ACM TASLP | causal variant | 8 kHz (separation) | L=16 samples = 2 ms | very low (2 ms frames) | 5.1M | - | WSJ0-2mix SI-SNRi 15.3 dB non-causal / 10.6 dB causal (separation, not denoising) | (Asteroid / ESPnet impls) | [F] |

## 2. Non-causal quality references ("offline ceiling")

| Model | Year / venue | fs | STFT | Params | VB-D PESQ (WB) | Other | License | Evidence |
|---|---|---|---|---|---|---|---|---|
| CMGAN | 2022, Interspeech 2022 (journal ext. later) | 16 kHz | 25 ms Hamming / 6.25 ms hop (75 % ovl) | 1.83M | 3.41 | CSIG 4.63, CBAK 3.94, COVL 4.12, SSNR 11.10, STOI 0.96 | MIT, github.com/ruizhecao96/CMGAN | [F] |
| MP-SENet | 2023, Interspeech 2023 (long version: Neural Networks vol. 189, 2025) | 16 kHz | 400 / 100 samples | 2.05M | 3.50 | CSIG 4.73, CBAK 3.95, COVL 4.22, STOI 0.96 | MIT, github.com/yxlu-0102/MP-SENet | [F] |
| SEMamba | 2024, IEEE SLT 2024 | 16 kHz (repo: any fs) | - | [U] | 3.55 (paper) / 3.56 (README); 3.69-3.75 with PCS | 4th of 70 in NeurIPS 2024 URGENT | (LICENSE file; type not read) [U], github.com/RoyChao19477/SEMamba | [F] |
| TF-GridNet | 2023, IEEE/ACM TASLP | 8 kHz WSJ0-2mix | - | [U] | n/a (separation) | WSJ0-2mix SI-SDRi 23.5 dB; SOTA on L3DAS22 enhancement | (ESPnet) | [F] abstract |
| MossFormer2 | 2024, ICASSP 2024 (separation) | 8 kHz sep; SE model 48 kHz in ClearerVoice | - | [U] | [U] | SOTA on WSJ0-2/3mix, Libri2Mix, WHAM!/WHAMR! | Apache-2.0 (ClearerVoice-Studio) | [F] |

## 3. Newer (2023-2026) causal / lightweight models

| Model | Date / venue | Org | Causal / latency | fs | Params | Compute | Reported quality (test set) | Code / license | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| BSRNN (online pBSRNN / BSRNN-S) | arXiv 2212.00406; Interspeech 2023 pp. 2483-2487, DOI 10.21437/Interspeech.2023-1433 | Tencent AI Lab | causal + non-causal variants; 20 ms win / 10 ms hop at 48 kHz | 48 kHz (16 kHz baseline) | not reported | online pBSRNN ~14.7 G MAC/s; RTF 0.42 on Intel i5 | DNS-2020 synth no-reverb, causal BSRNN-S+MRSD: WB-PESQ 3.32, NB-PESQ 3.77, SI-SNR 20.5, STOI 97.8; non-causal WB-PESQ 3.53; top-3 in DNS-2023 (both tracks) | [U] | [F] |
| aTENNuate (deep SSM, raw waveform) | arXiv 2409.03377 (v4 20 May 2025); Interspeech 2025 pp. 51-55, DOI 10.21437/Interspeech.2025-19 | BrainChip | causal; 46.5 ms (base), 31.25 ms or 16 ms variants | 16 kHz | 0.84M | 0.33 G MAC/s | VB-D PESQ 3.27 (base), 3.21 (31.25 ms), 3.06 (16 ms); DNS1 PESQ 2.98 / 2.84 / 2.59 | pip "attenuate"; GitHub repo URL returned 404 when fetched [U] | [F] |
| DPDFNet-2/4/8 (DeepFilterNet2 + DPRNN encoder) | arXiv 2512.16420 (18 Dec 2025) | Ceva Inc. | causal; 20 ms Vorbis win / 10 ms hop; README: first output after ~20 ms window, then ~10 ms per block (2-frame look-ahead per paper - exact algorithmic latency definition to verify) | 16 kHz (48 kHz "hr" models too) | 2.49M / 2.84M / 3.54M | 1.35 / 2.36 / 4.37 G MACs | Own multilingual low-SNR set (0/5/10 dB, 12 languages, 13.5 h): DPDFNet-8 PESQ 2.85, STOI 93.4, SI-SNR 14.47 vs DFN2 2.35, DFN3 2.45, GTCRN 2.00; DNS4 blind DNSMOS OVL 3.122 vs DFN3 2.958; DPDFNet-4 RTF 0.97 on Ceva NeuPro-Nano NPN32 (int8 w / int16 act) | Apache-2.0, github.com/ceva-ip/DPDFNet (ONNX + TFLite) | [F] |
| RT-SEMamba | arXiv 2608.12099 (12 Aug 2026, v2 16 Sep 2026); accepted Interspeech 2026 | Chao, Huang, La Quatra, Siniscalchi, Cheng, Fu, Tsao (affiliations not verified) | fully causal, 25 ms algorithmic latency (one 400-sample window) | 16 kHz | 2.74M teacher / 1.05M (1-layer student) | 47.35 / 20.56 G MAC/s; RTF 0.29 / 0.11 on RTX 5090 | VB-D: PESQ 3.32 (8-layer), 3.22 (2-layer), 3.18 (1-layer), CSIG 4.43-4.64, COVL 3.89-4.08, STOI 0.95 | "will be released" at github.com/RoyChao19477/RT-SEMamba [U] | [F] |
| BASENet (causal) | arXiv 2606.12662 (10 Jun 2026) | **Thales SIX GTS, France** | causal variant; nfft 400 / hop 100 (latency not stated) | 16 kHz | 0.81M | 7.1 G MACs | VB-D causal: PESQ 3.44, CSIG 4.58, CBAK 3.85, COVL 4.04, STOI 96 %; non-causal 3.55 | not provided [U] | [F] |
| CleanUMamba | arXiv 2410.11062; ISCAS 2025 | - | causal, raw waveform | [U] | 442K | 468M MACs | PESQ 2.42, STOI 95.1 % (test set not verified) | github (hdkuad/CleanUMamba) [S] | [S] |
| Fast-ULCNet | arXiv 2601.14925 (21 Jan 2026) | - | causal | 16 kHz | 0.338M (ULCNet 0.685M) | 1.691 M MACs (unit as printed; per-frame?) ; RTF 0.657 RPi3 vs 0.976 ULCNet | DNS-2020 no-reverb 10 s clips: PESQ 2.51 vs ULCNet 2.62 | [U] | [F] |
| UL-UNAS (NAS) | arXiv 2503.00340 (v1 1 Mar 2025, v2 1 Feb 2026), accepted IEEE TASLP | Nanjing Univ. / Horizon Robotics | causal, 0 look-ahead, 32/16 ms | 16 kHz | 171K | 35 M MAC/s | VB-D PESQ 3.09 (w/ PESQ loss); DNS3 PESQ 2.30, ESTOI 78.24, DNSMOS OVRL 2.71 | github.com/Xiaobin-Rong/ul-unas | [F] |

### 3.1 Non-causal 2025 reference: xLSTM-SENet
- xLSTM-SENet / xLSTM-SENet2 (Kuhne, Ostergaard, Jensen, Tan; arXiv 2501.06146, Interspeech 2025): bidirectional (non-causal), 16 kHz, nfft/win/hop 400/400/100. VB-D: xLSTM-SENet2 2.27M params, PESQ 3.53, CSIG 4.78, CBAK 3.98, COVL 4.27, STOI 0.96. Re-trained baselines in the same paper: SEMamba 3.49, MP-SENet 3.49, plain LSTM 3.49. Code github.com/NikolaiKyhne/xLSTM-SENet. [F]
- Take-away: at ~2M params on VB-D, Conformer, Mamba, xLSTM and even LSTM blocks land within ~0.05 PESQ of each other when retrained under one recipe. Block type matters less than the TF dual-path magnitude+phase design. [F, same paper]

## 4. Challenges and what their winners tell us

### 4.1 DNS Challenge real-time rules
- ICASSP 2021 DNS (Reddy et al.): Track 1 total algorithmic latency (frame T + stride Ts + look-ahead) must be <= 40 ms; example given is a 20 ms frame with 10 ms stride = 30 ms. Compute rule: process a frame in less than the stride on an Intel Core i5 quad-core at 2.4 GHz. [F] https://ar5iv.labs.arxiv.org/html/2009.06122
- Winners/placings of record: DCCRN 1st real-time track DNS-2020 [F]; PercepNet 2nd DNS-2020 RT [F]; DTLN 8th of 17 DNS-2020 RT [F]; DPCRN 3rd wide-band track DNS3 (Interspeech 2021) [F]; FRCRN 2nd real-time full-band non-personalized track DNS4 (ICASSP 2022) [F]; BSRNN top-3 DNS5 (ICASSP 2023) both tracks [F].
- ICASSP 2023 DNS (5th and last edition): personalized (enrollment) tracks, headset + speakerphone, 48 kHz; score = 0.5[WAcc + 0.25(OVRL-1)]; the same team won both tracks (+0.145 / +0.141 over noisy). Winning team not named in the fetched overview text. [F] https://ar5iv.labs.arxiv.org/html/2303.11510 ; NPU-Elevoc claims tie for 1st in headset track (arXiv 2303.06811) [S].

### 4.2 Interspeech 2025 URGENT challenge (2nd edition)
- Saijo, Zhang, Cornell, Scheibler, Li, Ni, Kumar, Sach, Fu, Wang, Fingscheidt, Watanabe. Interspeech 2025 pp. 858-862, DOI 10.21437/Interspeech.2025-1363; arXiv 2505.23212. [F]
- 32 submissions (22 Track 1 with ~2.5k h speech, 10 Track 2 with ~60k h). Inputs at 8-48 kHz; 7 distortions incl. additive noise, reverb, clipping, bandwidth limitation, codec, packet loss, **wind noise**; up to 5 distortions simultaneously. Baseline TF-GridNet ~8.5M params. [F]
- Rank-1 Track 1 system was purely discriminative (~102M params; TF dual-path variant with fast band processing; Sun et al., "Scaling beyond Denoising", Interspeech 2025 pp. 873-877, DOI 10.21437/Interspeech.2025-795). Most other competitive systems were hybrid (discriminative + adversarial). Purely generative SE showed language dependency; DNSMOS rated hallucinated output highly. [F]
- Relevance: URGENT winners are **not** real-time/low-latency; but the finding "discriminative TF dual-path/band models win; DNSMOS can be fooled by hallucination" is directly relevant for safety-critical comms (avoid generative SE that can hallucinate words).


(more sections being added: Jetson deployment, harmonic/impulsive noise evidence, recommendation)
