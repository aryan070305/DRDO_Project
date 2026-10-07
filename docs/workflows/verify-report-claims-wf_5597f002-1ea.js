export const meta = {
  name: 'verify-report-claims',
  description: 'Adversarially verify every numeric/licensing/date claim the D-ANC report will cite, against primary sources',
  phases: [{ title: 'Verify', detail: '3 verifiers, ~18 claims each' }],
}

const GROUPS = {
  architectures_benchmarks: [
    'A1 DeepFilterNet2 (Schroeter et al., arXiv 2205.05474, IWAENC 2022): 2.31M params, 0.356 GMAC/s, VoiceBank+DEMAND WB-PESQ 3.08 (some sources 3.03), STOI 0.943, RTF 0.04 on i5-8250U single thread and 0.42 on Raspberry Pi 4, 2-frame look-ahead, 40 ms overall latency',
    'A2 DeepFilterNet3 (arXiv 2305.08227, Interspeech 2023 show&tell): VB+DEMAND PESQ 3.17, STOI 0.944, RTF 0.19 single-thread i5-8250U, 48 kHz',
    'A3 GTCRN (Rong et al., ICASSP 2024, github Xiaobin-Rong/gtcrn, MIT): 48.2K params, 33.0 MMAC/s, VB+DEMAND PESQ 2.87, STOI 0.940, SI-SNR 18.83 dB; HybridLoss = 30*(real+imag)+70*mag+sisnr with power-law compression 0.3',
    'A4 ULCNet (ICASSP 2024): 688K params, 0.098 GMAC/s, VB+DEMAND PESQ 2.87, RTF 0.127 on one Cortex-A53 core',
    'A5 LiSenNet (ICASSP 2025, github hyyan2k/LiSenNet): 37K params, 56 MMAC/s, VB+DEMAND PESQ 3.07',
    'A6 UL-UNAS (arXiv 2503.00340): 171K params, 35 MMAC/s, VB+DEMAND PESQ 2.96 / 3.09 with PESQ loss',
    'A7 FRCRN (ICASSP 2022): VB+DEMAND WB-PESQ 3.21, 6.9M params (16 kHz model)',
    'A8 Non-causal references on VB+DEMAND: CMGAN PESQ 3.41 (Interspeech 2022), MP-SENet 3.50 (Interspeech 2023), SEMamba 3.55 (IEEE SLT 2024)',
    'A9 BASENet (arXiv 2606.12662, June 2026, Thales): causal variant VB+DEMAND PESQ 3.44',
    'A10 DPDFNet (arXiv 2512.16420, Dec 2025, Ceva): on their multilingual low-SNR (0/5/10 dB) test set DFN3 PESQ 2.45, DPDFNet-8 PESQ 2.85, GTCRN 2.00',
    'A11 RNNoise (Valin, arXiv 1709.08243, IEEE MMSP 2018): 87,503 weights, 48 kHz, 20 ms window / 10 ms hop',
    'A12 ICASSP 2021 DNS Challenge real-time track: total algorithmic latency (frame + stride + look-ahead) <= 40 ms; per-frame processing below the stride on an Intel Core i5 quad-core 2.4 GHz',
    'A13 Tan & Wang CRN, Interspeech 2018: at -5 dB (untrained speakers, babble/cafeteria) STOI 0.7642 and PESQ 2.04 enhanced',
    'A14 Xu, Du, Huang, Dai, Lee (Interspeech 2015, arXiv 1703.07172): unseen NOISEX-92 noises (Buccaneer1, Destroyer engine, HF channel) at 5 dB enhanced STOI 0.876, PESQ 2.597; at -5 dB 0.688 / 1.887',
    'A15 Tan & Wang GCRN (IEEE/ACM TASLP 28:380-390, 2020): at 0 dB STOI 0.9012, PESQ 2.65; at -5 dB 0.8075 / 2.13',
    'A16 MetricGAN+ (Fu et al., arXiv 2104.03538, Interspeech 2021): VB+DEMAND PESQ 3.15',
    'A17 Braun & Tashev (arXiv 2009.12286, 2020): compression exponent c = 0.3 and combined compressed loss with weight 0.3 on the complex term performs best',
  ],
  edge_hybrid_standards: [
    'B1 NVIDIA Jetson AGX Orin 64GB: 275 TOPS (INT8 sparse), 12-core Arm Cortex-A78AE at 2.2 GHz, 2048-core Ampere GPU with 64 Tensor Cores, 64 GB LPDDR5 at 204.8 GB/s, power 15 W - 60 W',
    'B2 Jetson Linux r36.4 developer guide nvpmodel modes for AGX Orin 64GB: MAXN (12 CPUs, 2201.6 MHz), 15W (4 CPUs, 1113.6 MHz, GPU 408 MHz), 30W (8 CPUs, 1728 MHz), 50W (12 CPUs, 1497.6 MHz)',
    'B3 TensorRT 10 docs: IRNNv2Layer removed in TensorRT 10.0; loop API supports only FP32 and FP16',
    'B4 TensorRT docs: TensorRT 10.7 was the last release supporting DLA',
    'B5 NVIDIA TensorRT-RTX docs: kernel launch takes about 5-15 microseconds of CPU/driver time per kernel',
    'B6 JetPack 7.2 (released 2026-06-02): TensorRT 10.16.2, CUDA 13.2.1, Ubuntu 24.04, adds Orin support in JetPack 7; JetPack 6.2.x: TensorRT 10.3, CUDA 12.6',
    'B7 Kealey et al. arXiv 2303.00949 (2023): Jetson Xavier NX audio I/O latency 80 ms without processing, 40 ms algorithmic',
    'B8 Widrow et al., Adaptive noise cancelling: principles and applications, Proc. IEEE 63(12):1692-1716, Dec 1975; with signal leakage into the reference the output SNR density equals the reciprocal of the reference SNR density',
    'B9 ITU-T G.114 (05/2003): below 150 ms one-way mouth-to-ear most applications are essentially transparent; 400 ms upper planning bound',
    'B10 ICASSP 2023 AEC Challenge (Cutler et al., arXiv 2309.12553): algorithmic latency + buffering latency <= 20 ms',
    'B11 Tan, Zhang, Wang, Deep learning based real-time speech enhancement for dual-microphone mobile phones, IEEE/ACM TASLP 29:1853-1863, 2021: pruned causal model 103.07K params; 2.78 ms per 20 ms frame on i7-10510U; at 0 dB diffuse babble PESQ 1.57 -> 2.74, STOI 66.48 -> 76.48 %',
    'B12 Haubner, Brendel, Kellermann, End-to-end deep learning-based adaptation control for frequency-domain adaptive system identification, ICASSP 2022, arXiv 2106.01262',
    'B13 NKF neural Kalman filter AEC (Yang et al., arXiv 2207.11388): 5.3K parameters',
    'B14 Zhang & Wang, Deep ANC, Neural Networks vol 141 pp 1-10, 2021',
    'B15 Yuan et al. arXiv 2604.05519 (Apr 2026, preprint): open-ear smart-glasses ANC median mic-to-speaker latency 113 microseconds; noise reduction 11.1 dB at 45 us vs 4.4 dB at 726 us',
    'B16 Rusci et al. arXiv 2210.07692 (2022): mixed FP16-INT8 PTQ PESQ drop 0.06; uniform 8-bit about 0.3',
    'B17 Fedorov et al. TinyLSTMs, Interspeech 2020 (arXiv 2005.11138): 0.55 dB SDR loss after pruning+quantization; 2.39 ms latency on MCU',
    'B18 ONNX Runtime quantization docs: dynamic quantization recommended for RNN/transformer models, static for CNNs',
  ],
  defence_physics_data_metrics: [
    'C1 Narain, Kant, Singh, Defence Science Journal 76(3), May 2026, pp. 408-417, DOI 10.14429/dsj.21095; abstract claims SNR gains 14-16 dB and PESQ improvements exceeding 7 %',
    'C2 Jaiswal, Yeduri, Cenkeramaddi, Int. J. Speech Technology 25(3):745-758, 2022: 8 kHz, two utterances, helicopter/airplane noise from ESC-50, PESQ 2.621 for female speaker helicopter 5 dB with implicit Wiener filter',
    'C3 Timmermann, Ernst, Sachau, Speech enhancement for helicopter headsets: simulation and implementation on an FPGA platform, POMA vol 52, 2023, DOI 10.1121/2.0001865 (metadata; PESQ 1.13 -> 2.08 if accessible)',
    'C4 MIL-STD-1474E (15 April 2015) clause 4.3: speech communication not degraded below 80 % MRT (ANSI/ASA S3.2); impulsive peak < 140 dBP at the ear',
    'C5 MIL-STD-1472H (15 Sep 2020) Table XVII: normal acceptable intelligibility 91 % MRT / AI 0.5; minimally acceptable 75 % / 0.3; clause 5.3.6.5 noise-cancelling mics >= 10 dBA improvement in very loud low-frequency noise',
    'C6 RFC 8130 (March 2017): MELPe 2400 bps uses a 22.5 ms frame of 180 samples at 8 kHz; 1200/600 bps use 3/4 frames; STANAG 4591',
    'C7 R.C. Maher, Acoustical characterization of gunshots, IEEE SAFE workshop 2007: muzzle blast typically lasts less than 3 ms',
    'C8 Meinke et al., Seminars in Hearing 2017 (PMC5634813): peak SPLs from firearms range from about 140 to 175 dB',
    'C9 US Patent 5,310,137 (Sikorsky/United Technologies): BLACK HAWK main rotor blade passing frequency approximately 17 Hz',
    'C10 Mukhutdinov, Alex, Cavallaro, Wang, IEEE Access vol 11, 2023, DOI 10.1109/ACCESS.2023.3253719: drone ego-noise at -15 dB, best model PESQ 1.0 -> 1.9, ESTOI 0.1 -> 0.4',
    'C11 Vibravox (Hauret et al., arXiv 2407.11828): at 90 dB ambient, signal-to-external-noise ratio of throat microphone 25.6 dB vs headset boom mic 0.1 dB',
    'C12 PMSQE: Martin-Donas, Gomez, Gonzalez, Peinado, IEEE Signal Processing Letters 25(11):1680-1684, 2018, DOI 10.1109/LSP.2018.2871419',
    'C13 SI-SDR: Le Roux, Wisdom, Erdogan, Hershey, SDR - half-baked or well done?, ICASSP 2019',
    'C14 STOI: Taal, Hendriks, Heusdens, Jensen, IEEE TASLP 19(7):2125-2136, 2011; ESTOI: Jensen & Taal, IEEE/ACM TASLP 24(11):2009-2022, 2016',
    'C15 ITU-T P.862 (02/2001) PESQ narrowband; ITU-T P.862.2 wideband extension (2005, revised 2007); P.862.2 output range approx 1.04 to 4.64',
    'C16 VoiceBank+DEMAND (Valentini-Botinhao et al., SSW 2016 / Interspeech 2016, Edinburgh DataShare 10283/2791): CC BY 4.0, test SNRs 17.5/12.5/7.5/2.5 dB, 824 test utterances, 2 test speakers',
    'C17 Licenses: ESC-50 CC BY-NC 3.0; LibriSpeech CC BY 4.0; UrbanSound8K CC BY-NC; github saraalemadi/DroneAudioDataset has no license file; NOISEX-92 copyright TNO 1990 distributed under NATO AC243/RSG-10',
    'C18 Nelke & Vary, Measurement, analysis and simulation of wind noise signals for mobile communication devices, IWAENC 2014',
    'C19 Friedlander waveform for blast waves p(t)=Ps(1-t/T)exp(-bt/T) (Friedlander 1946, Proc. R. Soc. A) as used for muzzle blast (e.g. Routh & Maher AES 2016)',
  ],
}

const SCHEMA = {
  type: 'object',
  properties: {
    checks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          status: { type: 'string', enum: ['confirmed', 'corrected', 'unverifiable', 'refuted'] },
          verified_text: { type: 'string', description: 'the claim as it should be stated, with corrected numbers/dates' },
          source_url: { type: 'string' },
          source_date: { type: 'string' },
          comment: { type: 'string' },
        },
        required: ['id', 'status', 'verified_text', 'source_url', 'comment'],
      },
    },
  },
  required: ['checks'],
}

const results = await parallel(Object.entries(GROUPS).map(([g, claims]) => () =>
  agent(`You are an adversarial fact-checker for a defence speech-enhancement engineering report. Today is 2026-10-03.
Load WebSearch and WebFetch with ToolSearch (query "select:WebSearch,WebFetch"). For EACH claim below, fetch a primary source (arXiv abs/HTML page, official GitHub README/code, publisher/DOI landing page, ISCA archive, vendor docs, standards text) and check every number, date, venue and license. Status: confirmed (all details match), corrected (give the right values in verified_text), unverifiable (could not access the evidence yourself - do NOT guess), refuted. Be efficient: one or two authoritative fetches per claim. Background notes with candidate URLs exist in /Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/research/*.md (read them with the Read tool if helpful) but you must re-check against the source itself. Do not write or modify any files. Return verified_text for every claim (a precise, citable sentence including venue and date).

CLAIMS:
${claims.join('\n')}`, { label: `verify:${g}`, phase: 'Verify', schema: SCHEMA }).then(r => r ? { group: g, ...r } : null)))
return results.filter(Boolean)
