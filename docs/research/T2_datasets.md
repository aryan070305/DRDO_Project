# T2 - Datasets for Defence-Comms Speech Enhancement (clean speech, defence-relevant noise, RIRs)

Prepared: 2026-10-03 (research snapshot; WORK IN PROGRESS - updated incrementally)
Scope: catalog of openly available corpora usable to synthesize noisy/clean speech pairs for a hybrid DNN + LMS
noise-suppression system for two-way defence voice comms (Jetson AGX Orin target).

Evidence conventions
- "Fetched = yes" means the page/file was actually retrieved during this session and the value was read from it.
- Quotes are verbatim and <= 15 words.
- Benchmark conditions (VoiceBank+DEMAND, DNS synthetic test sets, etc.) are civilian/VoIP conditions,
  NOT defence conditions (no artillery, armoured vehicle, rotor or gunshot test sets).

---

## 1. Clean speech corpora

### 1.1 LibriSpeech (OpenSLR SLR12)
- URL: https://www.openslr.org/12/  (fetched)
- Org: OpenSLR / Panayotov, Chen, Povey, Khudanpur (ICASSP 2015 paper)
- License: CC BY 4.0 (stated on OpenSLR page)
- Description on page: "Large-scale (1000 hours) corpus of read English speech"
- Sample rate: 16 kHz (FLAC) - NOTE: not stated on the fetched OpenSLR page; from the ICASSP 2015 paper (not fetched here)

| File | Size (OpenSLR page) | Note |
|---|---|---|
| dev-clean.tar.gz | 337M | dev, "clean" |
| dev-other.tar.gz | 314M | dev, "other" |
| test-clean.tar.gz | 346M | test, "clean" |
| test-other.tar.gz | 328M | test, "other" |
| train-clean-100.tar.gz | 6.3G | 100 h clean |
| train-clean-360.tar.gz | 23G | 360 h clean |
| train-other-500.tar.gz | 30G | 500 h other |
| original-mp3.tar.gz | 87G | LibriVox mp3 sources |

### 1.2 Mini LibriSpeech (OpenSLR SLR31)
- URL: https://www.openslr.org/31/ (fetched). License CC BY 4.0.
- Description: "Subset of LibriSpeech corpus for purpose of regression testing."
- Files: dev-clean-2.tar.gz [126M]; train-clean-5.tar.gz [332M]. Hours/speakers NOT stated on the page (open item).

### 1.3 CSTR VCTK Corpus v0.92
- URL: https://datashare.ed.ac.uk/handle/10283/3443 ; DOI https://doi.org/10.7488/ds/2645 (fetched incl. DSpace REST metadata)
- Authors: Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten (University of Edinburgh, CSTR)
- Date available: 2019-11-13
- License (dc.rights): Creative Commons Attribution 4.0 International Public License
- Content: 110 English speakers, ~400 sentences each; recorded 96 kHz/24-bit (DPA 4035 + Sennheiser MKH 800), downsampled to 48 kHz
- Files: VCTK-Corpus-0.92.zip 10.94 GB
- Note: the DNS-Challenge README lists VCTK under ODC-By v1.0 (older release). DataShare v0.92 metadata says CC BY 4.0.

### 1.4 VoiceBank+DEMAND (Valentini-Botinhao 2016/2017)
- Current version: https://datashare.ed.ac.uk/handle/10283/2791 ; DOI https://doi.org/10.7488/ds/2117 ; available 2017-08-21 (fetched)
- Superseded version: https://datashare.ed.ac.uk/handle/10283/1942 ; DOI 10.7488/ds/1356 ; available 2016-03-22 (fetched)
- Creator: Valentini-Botinhao, Cassia (University of Edinburgh, CSTR)
- License (dc.rights, both versions, from DSpace REST API): Creative Commons Attribution 4.0 International Public License
- Abstract: "designed to train and test speech enhancement methods that operate at 48kHz."
- Files (v2017):

| File | Size |
|---|---|
| clean_testset_wav.zip | 147.18 MB |
| noisy_testset_wav.zip | 162.68 MB |
| clean_trainset_28spk_wav.zip | 2.32 GB |
| noisy_trainset_28spk_wav.zip | 2.64 GB |
| clean_trainset_56spk_wav.zip | 4.44 GB |
| noisy_trainset_56spk_wav.zip | 5.24 GB |
| testset_txt.zip | 429.96 KB |
| trainset_28spk_txt.zip / trainset_56spk_txt.zip | 6.19 MB / 12.32 MB |
| logfiles.zip | 113.21 KB |

- Construction (Valentini-Botinhao et al., Interspeech 2016, ISCA archive PDF fetched and text-extracted):
  - Train: 28 speakers (England) [+ 56-speaker set, Scotland/US]; 10 noises (speech-shaped, babble, 8 DEMAND);
    SNRs "15 dB, 10 dB, 5 dB and 0 dB" (40 conditions); levels per ITU-T P.56.
  - Test: 2 other England speakers; 5 other DEMAND noises (living room, office, bus, open-area cafeteria, public square);
    SNRs "17.5 dB, 12.5 dB, 7.5 dB and 2.5 dB" (20 conditions).
  - The widely-quoted 824 test utterances / 11,572 training utterances counts were NOT verified in this session (open).
- Defence relevance: low (domestic/office/transport noise, high SNRs). Most SE papers report PESQ/STOI here; it is
  NOT representative of rotor, gunfire, armoured-vehicle noise or sub-0 dB SNR.

### 1.5 Microsoft DNS Challenge data (2020-2023)
- Repo: https://github.com/microsoft/DNS-Challenge (README fetched via raw.githubusercontent.com)
- Branches (GitHub API, fetched): interspeech2020/master, icassp2021-final, interspeech2021/adddata,
  v4dnschallenge_ICASSP2022, v5dnschallenge_ICASSP2023, master (= DNS5), nsnset_48khz, pdns, ...
- Licenses (README): data/docs CC BY 4.0; code MIT. Per-source: LibriVox public domain; PTDB-TUG ODbL 1.0;
  VCTK ODC-By 1.0; VoxCeleb2 CC BY 4.0; CREMA-D ODbL/DbCL; AudioSet CC BY 4.0; Freesound CC0 only;
  DEMAND CC BY-SA 3.0 (as stated by DNS README); RIRs (OpenSLR26/28) Apache 2.0.
- DNS5 (ICASSP 2023) fullband (48 kHz) sizes, master README:

| Folder | Size |
|---|---|
| clean_fullband (total) | 827G |
| - read_speech | 299G |
| - german_speech | 319G |
| - spanish_speech | 65G |
| - french_speech | 62G |
| - italian_speech | 42G |
| - vctk_wav48_silence_trimmed | 27G |
| - russian_speech | 12G |
| - emotional_speech | 2.4G |
| - VocalSet_48kHz_mono | 974M |
| noise_fullband | 58G |
| impulse_responses | 5.9G |
| Total | "about 1TB" unpacked; "about 550GB" archived |

- DNS1 (Interspeech 2020) paper (Reddy et al., arXiv 2005.13981 / ISCA; PDF text extracted):
  - Clean: "500 hours of speech from 2150 speakers" (top-quartile MOS LibriVox), ~441 h used.
  - Noise: AudioSet+Freesound, "about 150 audio classes and 60,000 clips" + 10,000 Freesound/DEMAND clips.
  - Training mix SNR uniform 0-40 dB; test synthetic SNR uniform 0-25 dB; 4 test categories x 300 clips
    (synthetic no-reverb, synthetic reverb, real MS recordings, real AudioSet).
  - 16 kHz (DNS1-3). Benchmark condition: VoIP/office noises - NOT defence conditions.

### 1.6 EARS (Expressive Anechoic Recordings of Speech)
- Repo: https://github.com/facebookresearch/ears_dataset (fetched); paper arXiv 2406.06185 (fetched), Interspeech 2024
- Authors: Richter, Wu, Krenn, Welker, Lay, Watanabe, Richard, Gerkmann (Univ. Hamburg / Meta Reality Labs / CMU)
- 107 speakers, 100 h, 48 kHz, anechoic chamber; whisper-to-yell dynamic range (useful for shouted comms speech)
- License: README: "released under CC-NC 4.0 International license" -> NON-COMMERCIAL. Flag for defence/commercial use.

### 1.7 LibriTTS-R (OpenSLR SLR141)
- URL: https://www.openslr.org/141/ (fetched); paper arXiv 2305.18802 (fetched), Interspeech 2023, Koizumi et al. (Google)
- License CC BY 4.0; 585 h, 24 kHz, 2,456 speakers; restored (cleaned) version of LibriTTS
- Sizes: train_clean_100 8.1G; train_clean_360 28G; train_other_500 46G; dev_clean 1.3G; test_clean 1.2G

---

## 2. Noise corpora

### 2.1 NOISEX-92 (military noise)
- Original distributor: Speech Research Unit, DRA Malvern, UK; 2 CD-ROMs, GBP 135 (Cambridge comp.speech page, fetched):
  https://mi.eng.cam.ac.uk/comp.speech/Section1/Data/noisex.html
- Underlying source: NATO AC243/(Panel 3)/RSG-10 NOISE-ROM-0; ESPRIT Project 2589-SAM; produced by Institute for
  Perception-TNO (NL) and Speech Research Unit, RSRE (UK). Copyright line on each Rice SPIB page:
  "TNO, Soesterberg, The Netherlands, Feb 1990".
- Rice SPIB sample pages (spib.rice.edu - DNS NOT resolving as of 2026-10-03; archived copies fetched via web.archive.org):
  each file: 19.98 kHz, 16-bit, 235 s, ~9 MB.

| SPIB file | NOISE-ROM-0 signal | Recording detail (archived SPIB page) |
|---|---|---|
| leopard ("Military vehicle noise") | signal.009 | Leopard, 70 km/h, 114 dBA (page says "Leopard 2" then "Leopard 1") |
| m109 ("Tank noise") | signal.007 | M109 at 30 km/h, 100 dBA |
| machinegun | signal.016 | .50 calibre gun fired repeatedly |
| f16 (cockpit) | signal.020 | co-pilot seat, two-seat F-16, 500 knots, 300-600 ft, 103 dBA |
| buccaneer1 (cockpit) | signal.08 | 190 knots, 1000 ft, airbrakes out, 109 dBA |
| buccaneer2 (cockpit) | signal.011 | 450 knots, 300 ft, 116 dBA |
| destroyerengine | signal.015 | engine room, 101 dBA |
| destroyerops | signal.014 | operations room, 70 dBA |
| also: white, pink, hfchannel, babble, factory1, factory2, volvo | | |

- Today's availability: no official distributor found. Unofficial mirror: GitHub panandicoding/Build-SE-Dataset
  data/NoiseX92/*.wav (15 files, 9,399,852 bytes each; repo MIT license covers code only). Redistribution/defence-use
  rights of the TNO-copyrighted audio are UNCLEAR -> treat as research-only, seek permission/legal review.

### 2.2 DEMAND (Diverse Environments Multichannel Acoustic Noise Database)
- Zenodo record 1227121, DOI 10.5281/zenodo.1227121 (Zenodo API fetched); publication date 2013-06-09; v1.0
- Creators: Thiemann, Joachim; Ito, Nobutaka; Vincent, Emmanuel (Inria / Univ. Tokyo)
- 16-channel array recordings; mic spacing 5 cm to 21.8 cm; each environment as 16 single-channel WAVs at 48 kHz and 16 kHz
- 18 environment zips (DKITCHEN, DLIVING, DWASHING, NFIELD, NPARK, NRIVER, OHALLWAY, OMEETING, OOFFICE, PCAFETER,
  PRESTO, PSTATION, SCAFE, SPSQUARE, STRAFFIC, TBUS, TCAR, TMETRO); note: SCAFE_16k.zip absent from Zenodo file list.
- Sizes: 37 files, total 7.36 GB (48 kHz zips 5.60 GB; 16 kHz zips 1.77 GB)
- LICENSE DISCREPANCY: Zenodo metadata field = cc-by-4.0, but the record description says
  "licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License". DNS README also says CC BY-SA 3.0.
  Treat as CC BY-SA 3.0 (share-alike obligations on derived *datasets*) until clarified.
- Defence relevance: low-moderate (TCAR, TBUS, STRAFFIC useful for vehicle-like stationary noise; no military vehicles).

### 2.3 ESC-50
- Repo: https://github.com/karolpiczak/ESC-50 (README fetched); author Karol J. Piczak; v1.0.0 2015-04-15, v2.0.0 2017-12-13
- 2000 clips, 5 s, 44.1 kHz mono WAV; 50 classes x 40 clips; download zip "~600 MB"
- License: CC BY-NC 3.0 (dataset); ESC-10 subset CC BY; per-clip Freesound attributions in LICENSE file.
  GitHub API reports license "Other/NOASSERTION". NON-COMMERCIAL -> flag.
- Defence-relevant classes present (README table): Helicopter, Siren, Engine, Airplane, Wind, Fireworks, Chainsaw, Train,
  Thunderstorm, Hand saw, Rain (40 clips each = 200 s per class only).
- Individual files fetchable from raw.githubusercontent.com: HEAD on
  https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-100032-A-0.wav returned HTTP 200, audio/wav,
  content-length 441,044 bytes (= 5 s x 44.1 kHz x 16-bit + header). Filename pattern {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav;
  class index -> name mapping in meta/esc50.csv.
- Caveat (README): some source clips were preprocessed (band-limited) in a class-dependent way.

### 2.4 UrbanSound8K
- Landing page: https://urbansounddataset.weebly.com/urbansound8k.html (fetched); Zenodo record 1203745 (API fetched)
- Authors: Justin Salamon, Christopher Jacoby, Juan Pablo Bello (NYU); ACM Multimedia 2014; Zenodo date 2014-11-03
- 8732 labelled excerpts, <= 4 s, 10 classes incl. gun_shot, siren, engine_idling, jackhammer, car_horn, drilling
- Sample rate/bit depth: "same as those of the original file uploaded to Freesound" (varies per file)
- Download: UrbanSound8K.tar.gz 6,023,741,708 bytes (~6.0 GB)
- License: site says "free of charge for non-commercial use only" under CC BY-NC 3.0; Zenodo metadata says cc-by-nc-4.0.
  NON-COMMERCIAL -> flag.
- Slicing (ACM MM 2014 paper, PDF text-extracted): 4 s max slices, 2 s hop, max 1000 slices/class; gun_shot is among the
  smallest classes (per-class counts only shown in Fig. 2b - exact gun_shot count NOT verified here).

### 2.5 FSD50K
- Zenodo record 4060432 (API fetched), v1.0, publication date 2020-10-02; MTG, Universitat Pompeu Fabra
- Creators: Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra (paper: IEEE/ACM TASLP)
- 51,197 Freesound clips, 108.3 h, 200 classes (144 leaf, 56 intermediate) from AudioSet Ontology; PCM 16-bit 44.1 kHz mono;
  clip lengths 0.3-30 s
- Download: 11 files, 24.68 GB total (split zips dev_audio.z01-z05 + .zip; eval_audio.z01 + .zip; ground_truth, metadata, doc)
- Licenses: dataset as a whole CC BY (LICENSE-DATASET); per-clip:
  dev 40,966 clips = CC0 14,959 / CC-BY 20,017 / CC-BY-NC 4,616 / CC Sampling+ 1,374;
  eval 10,231 clips = CC0 4,914 / CC-BY 3,489 / CC-BY-NC 1,425 / CC Sampling+ 403.
  -> For commercial/defence deployment, filter to CC0 + CC-BY clips (use per-clip license JSON).
- Defence classes: FSD50K vocabulary is a subset of the AudioSet ontology; presence of "Gunshot_and_gunfire", "Explosion",
  "Helicopter", "Siren" in vocabulary.csv NOT verified in this session (vocabulary file is inside ground_truth zip) -> open item.

### 2.6 MUSAN
- OpenSLR SLR17 https://www.openslr.org/17/ (fetched): musan.tar.gz 11 GB; CC BY 4.0
- Snyder, Chen, Povey, arXiv 1510.08484 (2015-10-28): music, speech (12 languages), "technical and non-technical noises"
- Hours per subset not verified in this session.

### 2.7 AudioSet (labels only; audio = YouTube)
- https://research.google.com/audioset/download.html (fetched): CSVs (YouTube ID, start, end, labels) + 128-d 1 Hz features;
  balanced eval 20,383; balanced train 22,176; unbalanced train 2,042,985 segments; 527 classes; v1 March 2017;
  strong labels May 2021. License: dataset CC BY 4.0; ontology CC BY-SA 4.0.
- Audio itself is NOT distributed: must be scraped from YouTube (availability decays; YouTube ToS and uploader copyright
  apply -> defence/commercial use of the audio is legally UNCLEAR).
- Ontology (github.com/audioset/ontology ontology.json, fetched; 632 nodes): defence-relevant nodes
  Explosion /m/014zdl -> {Gunshot, gunfire /m/032s66 -> {Machine gun /m/04zjc, Fusillade /m/02z32qm (blacklisted),
  Artillery fire /m/0_1c, Cap gun}, Fireworks, Burst/pop, Eruption, Boom -> Sonic boom}; Aircraft -> {Aircraft engine,
  Helicopter /m/09ct_, Fixed-wing aircraft}; Siren /m/03kmc9 -> {Police, Ambulance, Fire engine, Civil defense siren};
  Engine -> {Heavy engine (low frequency), Jet engine, Idling, ...}; Wind -> Wind noise (microphone) /t/dd00092.
- Annotation counts (ontology pages fetched): Gunshot,gunfire 4,221; Machine gun 1,858; Fusillade 1,649;
  Artillery fire 980; Cap gun 671; Explosion 2,274; Fireworks 3,051; Boom 1,650; Helicopter 3,698.

### 2.8 Drone / UAV audio
| Dataset | URL | Content | Size | License | Notes |
|---|---|---|---|---|---|
| DREGON (Strauss, Mordel, Miguet, Deleforge; IROS 2018) | http://dregon.inria.fr/ | 8-ch mic array on quadrotor, ego-noise + sources with 3D positions | not verified | "free to use for academic and educational purpose" | ACADEMIC-ONLY -> flag |
| Al-Emadi DroneAudioDataset (IWCMC 2019) | https://github.com/saraalemadi/DroneAudioDataset | indoor Bebop/Mambo propeller noise + "unknown" (ESC-50 + Speech Commands noise) | Binary: 1,332 drone + 10,372 unknown files (~384 MB) per GitHub tree API | NO LICENSE FILE (GitHub API license=None) | license UNCLEAR; unknown class reuses ESC-50 (CC BY-NC) |
| Drone-detection-dataset (Svanstrom et al. 2020/21) | https://github.com/DroneDetectionThesis/Drone-detection-dataset | 90 audio clips: Drone, Helicopter, Background (+IR/visible video) | repo ~299 MB | CC0-1.0 | small |
| DroneAudioset (Gupta et al., NeurIPS 2025 D&B; arXiv 2510.15383) | https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet | 23.5 h annotated; SNR -57.2 to -2.5 dB; drone ego-noise + human-presence sounds | HF repo 42.6 GB (treesize API) | MIT (HF card + arXiv abstract) | most relevant open drone ego-noise corpus |

### 2.9 Gunshot / impulsive
| Dataset | URL | Content | Size | License | Notes |
|---|---|---|---|---|---|
| Gunshot/Gunfire Audio Dataset (Kabealo & Wyatt; Data in Brief 48:109091, June 2023, doi 10.1016/j.dib.2023.109091) | https://zenodo.org/records/6836032 | multi-firearm, multi-orientation, time-synchronised edge-device recordings at outdoor range; Zenodo text cites IoBT | edge-collected-gunshot-audio.zip 1,567,979,135 B | CC BY 4.0 (Zenodo) | best openly-licensed gunshot corpus found; #firearms/sample rate not verified |
| Free Firearm Sound Library (Jaszczak, Nelson, Heras, Nanney; 2014-03-09) | https://opengameart.org/content/the-free-firearm-sound-library | SFX-grade rifle/shotgun/pistol recordings | Prepared SFX Library.7z 194 MB | CC0 | SFX (close-mic, processed); good for impulsive augmentation |
| Montana State Univ. gunshot recordings (R. C. Maher; NIJ 2014-DN-BX-K034) | https://montana.edu/rmaher/gunshots | anechoic-style multi-azimuth recordings of ~10 firearms | not stated | NO LICENSE STATED | license UNCLEAR |
| Cadre Research Labs NIJ project (Lilien et al.; NCJ 252947, 2018) | https://nij.ojp.gov/library/publications/development-computational-methods-audio-analysis-gunshots | ~20 firearms controlled firings + YouTube-extracted audio | n/a | access/licence not found | UNCLEAR - not verified as downloadable |
| C3GD Certus Caliber Classification Gunshot Dataset (Gurny, Quinn; arXiv 2606.18135, 2026-06-16) | https://arxiv.org/abs/2606.18135 | >8000 field-collected points, 28 firearms, 16 calibers | not verified | CC BY 4.0 shown on arXiv page (may be paper licence, not data) | download URL not verified |
| AudioSet Gunshot/Artillery labels | see 2.7 | YouTube | | CC BY 4.0 labels only | audio rights unclear |
| NOISEX-92 machinegun | see 2.1 | .50 cal, 235 s | 9.4 MB | unclear | |

(sections below in progress)
