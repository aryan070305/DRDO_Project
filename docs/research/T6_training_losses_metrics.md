# T6 - Training framework: losses, hyperparameters, augmentation; metric definitions and latency standards

Research date: 2026-10-03. Status: IN PROGRESS (file is updated incrementally).

Evidence conventions:
- `fetched=Y` means the source page itself was fetched and the number/statement was read there.
- `fetched=N` means seen only in a search snippet or secondary source; treat as unverified.
- All benchmark numbers are on public academic test sets (VoiceBank+DEMAND, DNS Challenge), NOT defence noise (rotor, gunfire, artillery, vehicle). They do not transfer automatically.
- PDF text for some arXiv papers was extracted locally from the arXiv PDF (stdlib zlib parser) because WebFetch could not render the PDF; those are marked "fetched=Y (PDF text)".

---

## 1. Loss functions

### 1.1 SI-SDR / SI-SNR (scale-invariant SDR)

| Item | Value | Source | fetched |
|---|---|---|---|
| Definition | alpha = s_hat^T s / ||s||^2 ; SI-SDR = 10 log10( ||alpha s||^2 / ||alpha s - s_hat||^2 ) | Le Roux, Wisdom, Erdogan, Hershey, "SDR - half-baked or well done?", arXiv:1811.02508 (submitted 6 Nov 2018); published ICASSP 2019 (venue to be confirmed, see open items) https://arxiv.org/abs/1811.02508 ; HTML via https://ar5iv.labs.arxiv.org/html/1811.02508 | Y |
| SI-SNR naming | Paper notes other work called the same quantity "SI-SNR" (same metric) | same | Y |
| SD-SDR | SD-SDR = 10 log10( ||alpha s||^2 / ||s - s_hat||^2 ) = SNR + 10 log10(alpha^2) - penalises rescaling | same | Y |
| Criticism of BSS_eval SDR | bss_eval_sources allows a time-invariant 512-tap filter on the reference, which "completely forgives channel errors" | same | Y |

Practical implication: SI-SDR is blind to gain, so a model trained only with SI-SDR can output arbitrary level; most real-time SE recipes combine SI-SDR with a spectral term (e.g. GTCRN below).

### 1.2 Multi-resolution STFT loss (MR-STFT)

Source: Yamamoto, Song, Kim, "Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram", arXiv:1910.11480 (v1 25 Oct 2019), ICASSP 2020. https://arxiv.org/abs/1910.11480 (HTML: https://ar5iv.labs.arxiv.org/html/1910.11480). fetched=Y.

- Single-resolution loss = spectral convergence L_sc = || |STFT(x)| - |STFT(x_hat)| ||_F / || |STFT(x)| ||_F  plus log-magnitude L_mag = (1/N) || log|STFT(x)| - log|STFT(x_hat)| ||_1.
- M = 3 resolutions, averaged: L_aux = (1/M) sum_m L_s^(m).

| Resolution | FFT size | Window | Hop | (at 24 kHz) |
|---|---|---|---|---|
| 1 | 1024 | 600 (25 ms) | 120 (5 ms) | |
| 2 | 2048 | 1200 (50 ms) | 240 (10 ms) | |
| 3 | 512 | 240 (10 ms) | 50 (~2 ms) | |

Hann window. Originally a vocoder loss (TTS), widely re-used in SE (e.g. DeepFilterNet's multiresspecloss is a variant with power-law compression instead of log, see 1.6).

### 1.3 Power-law compressed (complex) spectral MSE

Source: Braun & Tashev, "A consolidated view of loss functions for supervised deep learning-based speech enhancement", arXiv:2009.12286 (submitted 25 Sep 2020), Microsoft Research. https://arxiv.org/abs/2009.12286 . fetched=Y (abstract page + PDF text).

- magComp: < | |S_hat|^c - |S|^c |^2 > ; cComp: < | |S_hat|^c e^{j phi_S_hat} - |S|^c e^{j phi_S} |^2 >; paper states the compression exponent "is c = 0.3".
- Combined loss = (1 - alpha) * magComp + alpha * cComp; "the combined compressed loss ... achieves the highest performance" with weight 0.3 (PDF text: "highest performance with [alpha] = 0.3").
- Abstract: combining magnitude-only with phase-aware objectives "always leads to improvements"; compressed spectral values give "a significant improvement".
- Training: AdamW, learning rate 1e-4, validated every 10 epochs; CHiME-2 WSJ-20k reverberant data, validation/test SNRs -6 to 9 dB; 2.8 M parameter RNN gain-predictor. Loss weights chosen by grid search on validation PESQ.
- Benchmark condition: CHiME-2 (reverberant, binaural, left channel). NOT defence noise.

### 1.4 Asymmetric losses (penalise speech over-suppression)

| Source | Definition | Value | fetched |
|---|---|---|---|
| Wang, Lopez Moreno, Saglam, Wilson, Chiao, Liu, He, Li, Pelecanos, Nika, Gruenstein (Google), "VoiceFilter-Lite: Streaming Targeted Voice Separation for On-Device Speech Recognition", arXiv:2009.04323 (9 Sep 2020), Interspeech 2020 (venue to confirm) https://arxiv.org/abs/2009.04323 | g_asym(x, alpha) = x if x <= 0; alpha*x if x > 0 ; L_asym = sum_t sum_f ( g_asym(S_cln - S_enh, alpha) )^2 | alpha = 10 in experiments | Y (ar5iv HTML) |
| Eskimez, Yoshioka, Wang, Wang, Chen, Huang (Microsoft), "Personalized Speech Enhancement: New Models and Comprehensive Evaluation", arXiv:2110.09625 (18 Oct 2021) https://arxiv.org/abs/2110.09625 | PLCPA (power-law compressed phase-aware) loss with p = 0.3, alpha = 0.5; PLCPA-ASYM adds an over-suppression term weighted beta = 1.0; introduces TSOS (target speaker over-suppression) metric with threshold gamma = 0.1 | p=0.3, alpha=0.5, beta=1.0 | Y (ar5iv HTML) |
| DeepFilterNet code (Rikorose/DeepFilterNet, df/loss.py) | SpectralLoss multiplies squared magnitude error by `factor_under` where predicted magnitude < target ("Weighting if predicted abs is too low"); MaskLoss has `f_under` similarly | default 1 (i.e. off) in DFN2/DFN3 released configs | Y (raw code) |

Note: the user prompt mentions "Wisdom/Microsoft"; the asymmetric loss found is from Google (VoiceFilter-Lite, 2020) and Microsoft (Eskimez et al. 2021/2022). No Wisdom-authored asymmetric SE loss was located (see open items).

### 1.5 Metric-driven losses (PESQ / PMSQE / MetricGAN)

| Method | Citation | Key facts | fetched |
|---|---|---|---|
| PMSQE | Martin-Donas, Gomez, Gonzalez, Peinado, "A Deep Learning Loss Function Based on the Perceptual Evaluation of the Speech Quality", IEEE Signal Processing Letters, Nov 2018 (vol 25 no 11 pp 1680-1684 per search snippet), DOI 10.1109/LSP.2018.2871419. https://www.ugr.es/~joseangl/publication/martin-donas-deep-2018 | Per-frame PESQ-inspired loss on power spectra; two disturbance terms (symmetric/asymmetric) after auditory masking and threshold effects; amends MSE. Implemented in Asteroid (asteroid.losses.pmsqe). | Y (author page); volume/pages N (snippet) |
| Joint SDR + PESQ | Kim, El-Khamy, Lee (Samsung), "End-to-End Multi-Task Denoising for joint SDR and PESQ Optimization", arXiv:1901.09146 (26 Jan 2019) https://arxiv.org/abs/1901.09146 | Time-domain loss after iSTFT; differentiable PESQ-like loss + SDR loss in multi-task framework | Y (abstract) |
| MetricGAN | Fu, Liao, Tsao, Lin, "MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement", arXiv:1905.04874 (13 May 2019), ICML 2019 https://arxiv.org/abs/1905.04874 | Discriminator learns to mimic a black-box (non-differentiable) metric such as PESQ/STOI; generator optimised against it | Y |
| MetricGAN+ | Fu, Yu, Hsieh, Plantinga, Ravanelli, Lu, Tsao, "MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement", arXiv:2104.03538 (8 Apr 2021), Interspeech 2021 https://arxiv.org/abs/2104.03538 | VoiceBank-DEMAND PESQ 3.15 (+0.3 over MetricGAN) | Y (abstract) |

Caution: optimising PESQ directly risks over-fitting the metric (PESQ is not robust for nonlinear/neural processing - see metric section).

### 1.6 Self-supervised-feature ("perceptual") losses

| Method | Citation | Key facts | fetched |
|---|---|---|---|
| PFPL (phone-fortified perceptual loss) | Hsieh, Yu, Fu, Lu, Tsao, "Improving Perceptual Quality by Phone-Fortified Perceptual Loss using Wasserstein Distance for Speech Enhancement", arXiv:2010.15174 (28 Oct 2020; rev. 27 Apr 2021) https://arxiv.org/abs/2010.15174 | Uses wav2vec latent representations + Wasserstein distance; applied to Deep Complex U-Net on VoiceBank-DEMAND | Y (abstract) |
| SSL-feature loss | Close, Ravenscroft, Hain, Goetze (Sheffield), "Perceive and predict: self-supervised speech representation based loss functions for speech enhancement", arXiv:2301.04388 (11 Jan 2023), ICASSP 2023 https://arxiv.org/abs/2301.04388 | Distances between SSL feature encodings of clean vs noisy speech correlate strongly with PESQ/STOI/MOS; earlier (CNN encoder) features work better than final transformer outputs as a loss | Y (abstract) |

Caution for real-time deployment: these losses are training-only (the SSL model is not run at inference), so they add no runtime cost on the Jetson, but they add GPU memory/compute at training time.

### 1.7 Exact recipes of two reference real-time models

#### 1.7.1 DeepFilterNet family (Schroeter, Escalante-B., Rosenkranz, Maier; FAU Erlangen-Nuernberg + WS Audiology)

Sources:
- DeepFilterNet (v1): arXiv:2110.05588, ICASSP 2022. https://arxiv.org/abs/2110.05588 (HTML https://ar5iv.labs.arxiv.org/html/2110.05588). fetched=Y.
- DeepFilterNet2: arXiv:2205.05474 (11 May 2022), "Submitted to IWAENC 2022" (arXiv comment). https://arxiv.org/abs/2205.05474. fetched=Y.
- DeepFilterNet3 = "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement", arXiv:2305.08227 (14 May 2023), "Accepted as show and tell demo to Interspeech 2023". https://arxiv.org/abs/2305.08227. fetched=Y.
- Official code: https://github.com/Rikorose/DeepFilterNet (dual MIT / Apache-2.0). Loss code: https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/DeepFilterNet/df/loss.py . fetched=Y.
- Released training configs (config.ini). DFN3: third-party Hugging Face mirror https://huggingface.co/fal/DeepFilterNet3/raw/main/config.ini ; DFN2: mirror in https://huggingface.co/spaces/kevinwang676/VALLE/raw/main/DeepFilterNet2/config.ini . fetched=Y, BUT these are mirrors of the config.ini shipped inside the official model zip; verify against the official `models/DeepFilterNet3.zip` before relying on them.

| Item | DFN v1 (paper) | DFN2 (paper + config) | DFN3 (config mirror; paper omits) |
|---|---|---|---|
| Sample rate | 48 kHz | 48 kHz (fft 960 / hop 480 = 20 ms / 10 ms) | 48 kHz (fft 960 / hop 480) |
| Spectral loss | compressed mag + complex, c = 0.6; lambda_spec = 1, lambda_alpha = 0.05 | paper: lambda_spec = 1e3, lambda_MR = 5e2, c = 0.3; config: spectralloss factor_magnitude = 1000, factor_complex = 1000, gamma = 0.3 | spectralloss factors 0 (off) |
| Multi-res loss | - | factor 500, factor_complex 500, gamma 0.3, fft_sizes 256,512,1024 (paper text: windows 5,10,20,40 ms) | factor 500, factor_complex 500, gamma 0.3, fft_sizes 256,512,1024,2048 |
| Local-SNR loss | - | factor 1e-3 | factor 1e-3 |
| Asymmetric factor_under / f_under | - | 1 (off) | 1 (off) |
| Optimizer | Adam, lr 1e-3, decay 0.9 every 3 epochs | AdamW, lr 1e-3, 3 warm-up epochs then cosine decay, weight decay 1e-12 -> 0.05 | AdamW, lr 1e-3, lr_min 1e-6, warmup 3 epochs, weight decay 1e-12 -> 0.01 |
| Batch | 32 | 96 (paper: scheduled 8 -> 96) | 64; batch_size_scheduling 0/16,2/24,5/32,10/64,20/128 |
| Segment length | 3 s | max_sample_len_s 3.0 | max_sample_len_s 3.0 |
| Epochs | 30 | 100 | max 120, early stopping patience 25 |
| Training SNRs | {-5, 0, 5, 10, 20, 40} dB | (config) | dataloader_snrs = -100,-5,0,5,10,20,40 |
| Augmentation | up to 5 noises mixed; 2nd-order filters/EQ; gains {-6,0,6} dB; resampling; 10,000 simulated RIRs (RT60 0.05-1.0 s) | random 2nd-order filtering, gain, EQ, resampling, coloured noise; reverb (p_reverb 0.1); clipping distortion with SNR in [0,20] dB | p_reverb 0.1; clipping/bandwidth-ext/air-absorption/zeroing/interfering-speaker all 0.0 |
| Lookahead / latency | min 5 ms | 2-frame lookahead, 40 ms overall latency | df_lookahead 2, conv_lookahead 2; paper: 40 ms overall latency |
| Data | DNS Challenge (~750 h speech, ~180 h noise) | DNS4 English (~500 h speech, ~150 h noise) | full multilingual DNS4 with PTDB and VCTK oversampled x10 |
| VB-DEMAND result | - | PESQ 3.03, STOI 0.941; 2.306 M params, 0.356 GMACs; RTF 0.04 (notebook Core-i5) | PESQ 3.17, STOI 0.944, CSIG 4.34, CBAK 3.61, COVL 3.77; RTF 0.19 single-thread i5-8250U |

Note the DFN3 "dataloader_snrs" includes -100 dB (noise-only segments, i.e. the model learns to output silence for pure noise).

#### 1.7.2 GTCRN (Rong, Sun, Zhang, Hu, Zhang, Lu; Nanjing University) - ICASSP 2024

Sources: README + loss.py of official repo https://github.com/Xiaobin-Rong/gtcrn (MIT). Paper: IEEE Xplore doc 10448310, DOI 10.1109/ICASSP48485.2024.10448310; SigPort poster page https://sigport.org/node/7390 (posted 6 Jun 2024). fetched=Y (README, loss.py, SigPort page).

- HybridLoss (exact code): `return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr`
  - real/imag: MSE between compressed complex spectra (Re, Im divided by |X|^0.7, i.e. magnitude compressed to |X|^0.3 with phase kept).
  - mag_loss: MSE between |X|^0.3.
  - sisnr: computed on iSTFT waveform (n_fft 512, hop 256, sqrt-Hann window); implemented as `-log10(||s_target||^2/||e||^2)` (i.e. negative SI-SNR in dB / 10).
  - This is the DPCRN-style "compressed complex + magnitude + SI-SNR" hybrid loss.
- Complexity: README updates paper's 23.7 K params / 39.6 MMACs/s to 48.2 K params / 33.0 MMACs/s (ERB module counted).
- VCTK-DEMAND: PESQ 2.87, STOI 0.940, SI-SNR 18.83 dB. DNS3 blind test: DNSMOS P.808 3.44, BAK 3.90, SIG 3.00, OVRL 2.70. Streaming RTF 0.07 on i5-12400.
- Optimizer / lr / batch / segment / epochs for GTCRN itself: NOT found in fetched material (paper body not accessed). A third-party paper (arXiv:2502.14224, "Adaptive Convolution for CNN-based Speech Enhancement Models") reports re-training GTCRN with Adam lr 1e-3, batch 8, cosine annealing over 300 epochs on VCTK-DEMAND (search snippet only, fetched=N). See open items.

