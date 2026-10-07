# Data attribution and licences

Every corpus under `data/raw/` was downloaded by `danc.data.download`. (In the licence-safe public package, the
NOISEX-92, ESC-50 and drone folders and every mixture derived from them are left out; see `SHARE_README_PUBLIC.md`.) URLs, licence strings and SHA-256 hashes are in
`data/manifests/provenance.json`. The fixed test sets in `data/testsets/` mix these sources; their `meta.csv` files
name the noise source of every file, except `defence_edge_v1`. The test sets, the enhanced audio in `reports/audio/`
and every trained model in `checkpoints/`, `exports/` and `dist/jetson_bundle/exports/` are derived from these sources
and inherit their terms.

| Corpus (folder) | Creator / citation | Source | Licence |
|---|---|---|---|
| LibriSpeech dev-clean, test-clean, train-clean-100 (partial) (`raw/LibriSpeech/`) | V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015 | https://www.openslr.org/12 | CC BY 4.0 (`raw/LibriSpeech/LICENSE.TXT`) |
| Mini LibriSpeech train-clean-5 (`raw/LibriSpeech/`) | same authors | https://www.openslr.org/31 | CC BY 4.0 |
| VoiceBank+DEMAND test set, 16 kHz (`raw/vbdemand_test/`) | C. Valentini-Botinhao et al., Edinburgh DataShare 10283/2791 (16 kHz mirror: Hugging Face JacobLinCool/VoiceBank-DEMAND-16k) | https://datashare.ed.ac.uk/handle/10283/2791 | CC BY 4.0 |
| ESC-50, 16 classes used (`raw/esc50/`) | K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM MM 2015 | https://github.com/karolpiczak/ESC-50 | CC BY-NC 3.0, **non-commercial** |
| NOISEX-92, 15 recordings (`raw/noisex92/`) | A. Varga, H. J. M. Steeneken, Speech Communication 12(3), 1993; NOISE-ROM-0 © TNO Soesterberg 1990, distributed under NATO AC243/(Panel 3)/RSG-10 and ESPRIT 2589-SAM; obtained from the SPIB mirror | http://spib.linse.ufsc.br/noise.html | **Redistribution terms not stated: treat as internal** |
| DroneAudioDataset (`raw/drone/`) | S. A. Al-Emadi, A. K. Al-Ali, A. Al-Ali, A. Mohamed, "Audio Based Drone Detection and Identification using Deep Learning", IWCMC 2019 (Tangier) — citation as given in the dataset README | https://github.com/saraalemadi/DroneAudioDataset | **No licence file: all rights reserved; internal R&D only; do not redistribute** (`raw/drone/LICENSE_NOTICE.txt`) |
| ... inside it: the two `unknown/` folders (`raw/drone/DroneAudioDataset-master/{Binary,Multiclass}_Drone_Audio/unknown/`, 682 MB, **not used by the code**) | segments of all 50 ESC-50 classes and of the Speech Commands background noise (P. Warden, arXiv:1804.03209, 2018) | as above | ESC-50 parts CC BY-NC 3.0; Speech Commands parts CC BY 4.0 |

Project-generated, no third-party audio: `rir_bank/` (pyroomacoustics image-source RIRs), `synthetic_examples/` (physics-based
noise generators in `src/danc/data/synth_noise.py`), `manifests/`.
