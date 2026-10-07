export const meta = {
  name: 'anc-sota-research',
  description: 'Evidence-backed research sweep for a hybrid DNN+LMS defence speech-enhancement system, with per-topic citation verification',
  phases: [
    { title: 'Research', detail: '7 topic researchers using web search + fetch' },
    { title: 'Verify', detail: 'one adversarial citation checker per topic' },
  ],
}

const DATE = (args && args.today) || '2026-10-03'
const OUT = '/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/research'

const COMMON = `You are a senior audio-DSP / deep-learning speech-enhancement researcher. Today is ${DATE}.
Context: we are building a hybrid AI/ML adaptive noise cancellation (speech enhancement) system for defence two-way voice comms. Targets: output SNR > 15 dB, STOI > 0.85, PESQ > 2.5, real-time on NVIDIA Jetson AGX Orin 64GB (or comparable DSP/SoC), latency low enough for live two-way comms. Noise: stationary (engines, vehicles), non-stationary (helicopter rotors, drones, sirens, wind), impulsive (gunshots, artillery). Primary mic + reference mic; optional lightweight LMS adaptive filter combined with the DNN.

TOOLS: Load WebSearch and WebFetch with ToolSearch (query "select:WebSearch,WebFetch") before use. Use many searches. Prefer primary sources: arXiv abstract pages, official GitHub READMEs, conference proceedings (ISCA archive, IEEE Xplore abstract pages), official vendor docs, dataset landing pages (Zenodo, OpenSLR, university pages).

EVIDENCE RULES (critical; the user will build from this and forbids fabrication):
- Every claim must carry: source title, URL, authors/organization, publication or release date (as precise as you can), venue.
- Numeric claims (PESQ, STOI, SI-SDR, params, MACs, latency, dataset size/hours, license) must be read from the source itself (fetch it). Set fetched=true only if you actually fetched the page and saw the number. If you only saw it in a search snippet or secondary source, set fetched=false and say so in notes.
- Note the benchmark conditions for every metric (e.g. VoiceBank+DEMAND test set, SNRs 2.5/7.5/12.5/17.5 dB; DNS Challenge synthetic no-reverb test set; 16 kHz vs 48 kHz; wideband vs narrowband PESQ). These are NOT defence conditions, flag that.
- If you cannot verify something, put it in unverified_or_open, do not guess.
- Evidence quotes: at most 15 words each, verbatim.
- Write a thorough markdown note file (use the Write tool) at the path given below, containing everything you found with full citations, organized with headings and tables. Create the directory if needed with Bash mkdir -p (never delete anything). WRITE THE NOTES FILE EARLY AND UPDATE IT INCREMENTALLY (e.g. after every 5-10 sources) so partial work survives if you are interrupted.
- Be efficient: prefer a few authoritative primary sources per claim over many redundant searches.`

const CLAIM = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    claim: { type: 'string' },
    value: { type: 'string' },
    benchmark_conditions: { type: 'string' },
    source_title: { type: 'string' },
    source_url: { type: 'string' },
    authors_or_org: { type: 'string' },
    date: { type: 'string' },
    venue: { type: 'string' },
    evidence_quote: { type: 'string' },
    fetched: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['id', 'claim', 'source_title', 'source_url', 'date', 'fetched'],
}

const RESEARCH_SCHEMA = {
  type: 'object',
  properties: {
    topic: { type: 'string' },
    summary: { type: 'string' },
    claims: { type: 'array', items: CLAIM },
    recommendations: { type: 'array', items: { type: 'string' } },
    unverified_or_open: { type: 'array', items: { type: 'string' } },
    notes_file: { type: 'string' },
  },
  required: ['topic', 'summary', 'claims', 'recommendations', 'unverified_or_open', 'notes_file'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    topic: { type: 'string' },
    checks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          status: { type: 'string', enum: ['confirmed', 'corrected', 'unverifiable', 'refuted'] },
          corrected_value: { type: 'string' },
          corrected_date: { type: 'string' },
          checked_url: { type: 'string' },
          comment: { type: 'string' },
        },
        required: ['id', 'status', 'comment'],
      },
    },
    missing_important_items: { type: 'array', items: { type: 'string' } },
  },
  required: ['topic', 'checks', 'missing_important_items'],
}

const TOPICS = [
  {
    key: 'T1_architectures',
    title: 'Real-time / lightweight speech-enhancement architectures (SOTA through 2026)',
    prompt: `Survey causal, real-time-capable speech enhancement models and the strongest offline references. Must cover at least: RNNoise (Valin 2018), PercepNet, NSNet2, DTLN, DCCRN, DPCRN, FullSubNet / FullSubNet+ / Fast FullSubNet, FRCRN, DeepFilterNet v1/v2/v3, GTCRN, ULCNet, FSPEN, LiSenNet, CRN (Tan & Wang 2018), Demucs (facebook denoiser), Conv-TasNet, plus non-causal quality references CMGAN, MP-SENet, TF-GridNet, MossFormer2/ClearerVoice, SEMamba. Search for anything newer in 2024-2026 (Mamba / xLSTM / NAS-based / ultra-low-complexity / ICASSP & Interspeech 2024-2026 low-complexity tracks, DNS Challenge winners, the Interspeech 2025 URGENT challenge). For each: year+venue, causal?, sample rate, window/hop and algorithmic latency, params, MACs/FLOPs per second, reported PESQ(WB)/STOI/SI-SDR/DNSMOS and on which test set, license and official repo URL. Conclude which architecture families best fit a 16 kHz, <=20-40 ms algorithmic latency, Jetson-Orin-class budget with impulsive + harmonic (rotor) noise, and why (ERB gains + deep filtering, subband/fullband, complex masking, dual-path).`,
  },
  {
    key: 'T2_datasets',
    title: 'Datasets: clean speech, defence-relevant noise, RIRs; download URLs, sizes, licenses',
    prompt: `Build an accurate catalog of datasets for generating realistic noisy-clean speech pairs for defence comms. Clean speech: LibriSpeech (OpenSLR 12) incl. exact subset sizes (train-clean-100, dev-clean, test-clean), Mini LibriSpeech (OpenSLR 31) sizes, VCTK, VoiceBank+DEMAND (Valentini 2016/2017, Edinburgh DataShare, test-set zip sizes), DNS Challenge data (ICASSP/Interspeech 2020-2023), EARS, LibriTTS-R. Noise: NOISEX-92 (list its military recordings: f16, buccaneer, leopard, m109, machinegun, destroyer..., original sample rate, where it can be downloaded today, and its license status), DEMAND (Zenodo), ESC-50 (classes relevant to defence: helicopter, siren, engine, airplane, wind, fireworks, chainsaw, train, thunderstorm; license CC BY-NC; can individual files be fetched from raw.githubusercontent.com?), UrbanSound8K (gun_shot, siren, engine_idling, jackhammer), FSD50K (Gunshot/gunfire, Explosion, Helicopter, Siren classes; size; licenses), MUSAN, AudioSet (gunshot/explosion/artillery ontology classes), drone/UAV audio datasets (DREGON, Al-Emadi DroneAudioDataset, others on Zenodo), gunshot datasets (e.g. Gunshot Audio Forensics Dataset, any Zenodo/academic gunshot corpora), helicopter/military vehicle datasets, wind-noise datasets, emergency siren datasets, RIR datasets (OpenSLR 26/28, BUT ReverbDB, MIT IR survey, ACE). For each: URL, size, sample rate, number of files/hours, license, release date, and whether redistribution / defence use appears permitted. Explicitly flag datasets whose license is unclear or non-commercial, and state clearly that no openly licensed, curated defence (artillery/armored vehicle) speech-in-noise corpus was found if that is the case.`,
  },
  {
    key: 'T3_impulsive_nonstationary',
    title: 'Impulsive and rapidly-changing noise: gunshots, artillery, rotors, drones, sirens, wind',
    prompt: `Research how DNN and classical methods handle (a) impulsive noise (gunshots, explosions, artillery, transients; microphone clipping/saturation at high SPL; impulse noise in speech enhancement; transient noise suppression papers; level-dependent hearing protection), (b) harmonic/periodic rotor & drone ego-noise (DREGON, drone ego-noise reduction papers, helicopter cockpit speech enhancement, harmonic noise modeling; DeepFilterNet deep filtering for periodic noise), (c) sirens (swept tones), (d) wind noise (DNN wind noise reduction, wind noise generation e.g. Nelke & Vary), (e) rapidly changing noise and its effect on noise-tracking (MCRA/IMCRA, minimum statistics) vs DNN. Also physical acoustics of gunshots (muzzle blast as Friedlander wave, supersonic N-wave, durations, peak SPL 140-170 dB) and artillery, and helicopter blade-passing frequencies. Document risks: speech distortion, musical noise, over-suppression, artifacts, gain-pumping, and techniques to mitigate (gain floors, temporal smoothing, impulse-robust normalization, attack/release, comfort noise, asymmetric losses). Note how PESQ/STOI behave in impulsive-noise conditions if any paper discusses it.`,
  },
  {
    key: 'T4_hybrid_adaptive',
    title: 'Hybrid adaptive filtering (LMS/NLMS/FDAF/Kalman) + DNN with a reference microphone',
    prompt: `Research hybrid systems combining classical adaptive filters with DNNs for noise cancellation using a reference microphone or dual microphones. Cover: Widrow et al. 1975 adaptive noise cancelling (classical two-mic ANC, reference-mic coherence requirement, speech leakage into reference), NLMS, RLS, frequency-domain / partitioned-block / subband adaptive filters, Kalman-filter AF, Deep ANC (Zhang & Wang 2021), DNN-based step-size / adaptation control (e.g. Haubner et al., NeuralKalman/NKF for AEC, Meta-AF by Casebeer et al.), DNN speech-presence probability controlling adaptation (analogous to double-talk detection in AEC), dual-microphone CRN for mobile phones (Tan, Zhang, Wang 2019/2020), dual-mic headsets, coherence of noise fields vs mic spacing (diffuse field coherence sinc(kd)), and residual-echo/noise suppression DNN after a linear AF (AEC challenge architectures). Critically distinguish acoustic ANC (anti-noise into the ear; needs sub-100-microsecond-class latency, done on dedicated DSP/analog) from transmit-path adaptive noise cancellation/speech enhancement — find authoritative sources stating ANC latency requirements. Recommend how a learned model and an LMS-type filter should be ordered and coupled, and what failure modes (speech cancellation, misadjustment, slow convergence in non-stationary noise, impulsive noise breaking LMS) to guard against.`,
  },
  {
    key: 'T5_edge_deployment',
    title: 'Edge deployment: Jetson AGX Orin 64GB, TensorRT/ONNX, quantization, audio I/O latency, alternatives',
    prompt: `Research deployment of streaming speech enhancement on edge hardware. Official NVIDIA specs for Jetson AGX Orin 64GB Developer Kit (AI TOPS, GPU cores, CPU = 12-core Arm Cortex-A78AE, DLA, power modes 15W/30W/50W/MAXN 60W, JetPack 6.x / TensorRT 10.x versions, release dates). TensorRT and ONNX Runtime support for GRU/LSTM and stateful streaming; per-inference kernel-launch overhead for tiny models on GPU vs ARM CPU; any published measurements of speech enhancement models (DeepFilterNet, GTCRN, DTLN, RNNoise, etc.) on Jetson (Nano/Xavier/Orin) or Raspberry Pi/ARM CPUs (real-time factor, ms per frame), and DeepFilterNet's own reported RTF. Quantization (INT8/FP16, QAT) and pruning effects on speech-enhancement quality (papers). Audio I/O latency on Linux (ALSA period sizes, PipeWire/JACK), USB audio interface latency, I2S on Jetson 40-pin header, ReSpeaker/other mic arrays. Comparable alternative platforms (Qualcomm QCS6490/8550, NXP i.MX 8M Plus NPU, TI C7x DSP/AM62A, ADI SHARC, Hailo-8, Syntiant, Ambiq, Raspberry Pi 5) with power. Explicitly state which latency numbers on Jetson AGX Orin could NOT be found.`,
  },
  {
    key: 'T6_training_losses_metrics',
    title: 'Training framework: losses, hyperparameters, augmentation; metric definitions and latency standards',
    prompt: `Research training recipes for real-time SE: loss functions (SI-SNR/SI-SDR (Le Roux 2019 'SDR half-baked or well done'), multi-resolution STFT loss, power-law compressed complex spectral MSE (e.g. exponent 0.3, used by DPCRN/GTCRN/DeepFilterNet), PMSQE, MetricGAN/MetricGAN+, PESQ-based losses, perceptual feature losses (WavLM/HuBERT, 'perceptual loss' papers), phase-aware losses, asymmetric losses penalizing speech over-suppression (e.g. Wisdom/Microsoft), loss combinations used by DeepFilterNet3 and GTCRN with their exact hyperparameters (optimizer, learning rate, schedule, batch size, segment length, epochs, SNR ranges)), data augmentation (dynamic mixing, SNR sampling ranges, RIR convolution, clipping, gain, EQ/bandwidth, codec). Also the metric definitions: PESQ ITU-T P.862 / P.862.2 wideband (dates, score ranges, known limitations), STOI (Taal 2011) and ESTOI (Jensen & Taal 2016), SI-SDR, segmental SNR, DNSMOS P.835 (Reddy 2022), and the python packages that implement them (pesq, pystoi, torchmetrics) with licenses. Latency standards for two-way communication: ITU-T G.114 (one-way delay recommendations), ICASSP/Interspeech DNS Challenge latency limits (e.g. 40 ms algorithmic + buffering), hearing-aid/own-voice latency tolerance (Stone & Moore 1999-2008), military intercom latency if found.`,
  },
  {
    key: 'T7_defence_prior_work',
    title: 'Defence-specific prior work, standards and evaluation conditions',
    prompt: `Research speech enhancement / noise cancellation specifically for military, helicopter, cockpit, armored vehicle, and firearm environments: e.g. Timmermann, Ernst, Sachau (helicopter headset SE on FPGA, 2023), Jaiswal et al. 2022 implicit Wiener filter, NOISEX-92-based DNN SE results at low SNR (e.g. -5/0/5 dB F16, factory, machinegun), papers on SE in tank/armored vehicle noise, helicopter cockpit DNN speech enhancement, drone-based speech capture, military headset/communication standards (MIL-STD-1474E noise limits, MIL-STD-1472 speech intelligibility, STANAG 4591 MELPe codec, ANSI S3.2 intelligibility testing, DRT/MRT), alternative sensors used in military headsets (throat microphones, bone conduction, in-ear mics) and their use with DNNs, and DRDO/Indian defence research on noise cancellation. Also read this review: Narain, Kant, Singh, 'Artificial Intelligence Driven Advances in Noise Cancellation: A Comprehensive Review on Overcoming Noise Hazards in Military Operations', Defence Science Journal 76(3), May 2026, DOI 10.14429/dsj.21095, and extract which numerical claims it cites and their original sources. Report realistic metric levels achieved at very low input SNR (-10..0 dB) in the literature, since defence noise is often at those SNRs, and whether targets SNR>15 dB, STOI>0.85, PESQ>2.5 are realistic there.`,
  },
]

const results = await pipeline(
  TOPICS,
  t => agent(`${COMMON}\n\nTOPIC: ${t.title}\n\n${t.prompt}\n\nWrite your markdown notes to ${OUT}/${t.key}.md . Return the structured result (aim for 25-50 well-sourced claims).`,
    { label: `research:${t.key}`, phase: 'Research', schema: RESEARCH_SCHEMA }),
  (r, t) => {
    if (!r) return null
    const claimsTxt = JSON.stringify(r.claims, null, 1)
    return agent(`You are an adversarial citation checker for a defence speech-enhancement report. Today is ${DATE}. Load WebSearch and WebFetch with ToolSearch ("select:WebSearch,WebFetch").
Topic: ${t.title}
Below are claims produced by another researcher. For EVERY claim that carries a number, a license, a date, a dataset size, or an attribution, fetch the cited source (or a better primary source) and decide: confirmed (number/date/license matches), corrected (give the right value/date), unverifiable (could not access or not stated), refuted (source says otherwise / does not exist). Purely qualitative claims may be marked confirmed if the source supports them. Be skeptical: default to unverifiable when you cannot see the evidence yourself. Watch for common confusions: PESQ narrowband vs wideband, VoiceBank+DEMAND vs DNS test sets, MACs vs FLOPs, causal vs non-causal variants, arXiv date vs conference date, dataset license per-clip vs collection.
Also list important items the researcher missed (missing_important_items).
Then append a section '## Verification (adversarial check)' with a table of your verdicts to the notes file ${OUT}/${t.key}.md using the Edit or Bash tool (append only; never delete content).

CLAIMS:
${claimsTxt}`, { label: `verify:${t.key}`, phase: 'Verify', schema: VERDICT_SCHEMA })
      .then(v => ({ research: r, verification: v }))
  },
)

return results.filter(Boolean)
