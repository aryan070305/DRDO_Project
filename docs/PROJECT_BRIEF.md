# D-ANC project brief and requirements (verbatim)

This file holds the project owner's brief and every follow-up requirement **verbatim**, so that anyone receiving
the shared folder can check the delivered work against what was actually asked. The text inside the
fenced `text` code blocks has not been edited; each message is placed in such a block only so Markdown does
not reformat it. The headings, the targets table (§1) and the summary (§3.3) outside the blocks are added
for convenience and are not the owner's words.

- Source: the project owner's chat messages to the development assistant (timestamps in UTC; the
  development Mac ran at UTC+04).
- The reference paper the owner attached to the first message (Narain, Kant and Singh, *Defence Science
  Journal* 76(3), May 2026) is stored at
  [`docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf`](references/Narain_Kant_Singh_DSJ_2026_21095.pdf)
  (byte-identical copy of the owner's `21095.pdf`; SHA-256
  `0e52a512a8eeb8c3733dbe7f4ebb946a2815f0ceb69e24e376d93f20e80516d9`).
- How each target is interpreted and measured: [`reports/REPORT.md`](../reports/REPORT.md) §1. Results
  against the targets: §0 and §8.

---

## 1. Project brief (verbatim)

```text
title
To develop an AI/ML-enabled adaptive noise cancellation (ANC) system that effectively suppresses stationary, non-stationary, and impulsive defence noises while maintaining high speech intelligibility and real-time performance on embedded hardware

Description
In defence and mission-critical communication systems, reliable speech transmission is severely affected by diverse acoustic disturbances such as gunshots, artillery fire, helicopter rotor noise, armored vehicle sound and emergency sirens. Traditional signal processing techniques, like spectral subtraction, Wiener filtering, and classical LMS-based ANC, are limited in handling highly dynamic and non-linear noise environments. These methods assume stationary noise characteristics and often introduce artifacts or speech distortion under rapidly changing conditions.

Recent advancements in Artificial Intelligence and Machine Learning (AI/ML) have transformed the field of speech enhancement and ANC. Deep learning models and time-domain architectures are capable of learning complex spectral-temporal patterns directly from data. These models significantly outperform conventional approaches in terms of perceptual quality (PESQ), intelligibility (STOI), and noise suppression (SNR). Additionally, the rise of edge AI platforms enables deployment of such models on embedded systems for real-time applications.

Description:
The proposed system integrates AI/ML-driven noise suppression with adaptive filtering to create a robust ANC pipeline. The development begins with dataset generation, where clean speech data is combined with curated defence noise datasets (gunshots, drones, artillery, vehicle engines, wind, etc.) at varying SNR levels. This synthetic data generation ensures coverage of both stationary and impulsive noise scenarios.

The training pipeline involves transforming audio into time-frequency representations (e.g., STFT spectrograms) or directly using raw waveform inputs. Models process both full-band and sub-band features to capture global and local dependencies, while it also operates in the complex domain to preserve phase information. Training is performed using loss functions such as SI-SNR, L1/L2 loss, and perceptual loss, with evaluation metrics including SNR, STOI, and PESQ. Data augmentation techniques (random noise mixing, reverberation, clipping) are applied to improve generalization.

During inference, the trained model processes incoming noisy audio in real time, estimating a mask or directly reconstructing enhanced speech. The system can optionally include a lightweight adaptive filter (e.g., LMS) for residual noise suppression.

For prototype demonstration, the trained model is deployed on embedded/edge hardware such as DSPs or AI-enabled SoCs (e.g., NVIDIA Jetson AGX Orin 64GB Developer Kit or similar platforms). Optimization techniques like quantization, pruning, and ONNX / TensorRT conversion are applied to meet latency and power constraints. The system is integrated with microphones (primary + reference) and headphones/communication units to validate real-time ANC performance in practical environments

Expected Solution:
The final solution is a hybrid AI-driven ANC system capable of operating in real-time and handling diverse noise environments, including impulsive and highly dynamic defence scenarios. It should include:
* A scalable dataset pipeline for generating realistic noisy-clean speech pairs
* A state-of-the-art AI/ML model trained for robust noise suppression
* A training framework with optimized hyper-parameters and perceptual loss functions
* A real-time inference engine deployable on edge hardware
* A prototype system demonstrating live noise cancellation using microphones / headset integration
The system is expected to achieve significant performance improvements, targeting SNR > 15 dB, STOI > 0.85, and PESQ > 2.5, while maintaining low latency suitable for real-time communication. This solution will enable reliable and intelligible communication in defence, aerospace, and high-noise industrial environments.
```

### Targets stated in the brief

| Target | Value |
|---|---|
| Output SNR | > 15 dB |
| STOI | > 0.85 |
| PESQ | > 2.5 |
| Latency | "low latency suitable for real-time communication" |
| Platform | DSPs or AI-enabled SoCs, e.g. NVIDIA Jetson AGX Orin 64GB Developer Kit |
| Deliverables | dataset pipeline; SOTA model; training framework (hyper-parameters, perceptual losses); real-time edge inference engine; prototype with primary + reference microphones and headset / communication unit |

---

## 2. Original request that framed the brief (verbatim, 2026-10-03 11:29 UTC)

The brief above was pasted inside this request (the brief itself is reproduced in §1 and in Appendix A).

```text
I want a complete, evidence-backed technical solution for a hybrid AI/ML adaptive noise cancellation (ANC) system. It must suppress **stationary, non-stationary, and impulsive defence noise** while preserving speech intelligibility, and it must run in **real time on embedded/edge hardware** (NVIDIA Jetson AGX Orin 64GB Developer Kit or a comparable DSP or AI-enabled SoC). The system must reach **SNR > 15 dB, STOI > 0.85, and PESQ > 2.5** at latency low enough for live two-way communication. Every design choice you make should be justified against these targets and the edge latency and power budget. A model that scores well offline but cannot run in real time on the target hardware does not solve my problem.

Approach this as a senior audio DSP engineer specializing in deep-learning speech enhancement and embedded deployment for defence communications.

**What I need from you**

Research the current state of the art using web search, recent papers, benchmarks, and open-source implementations. Then synthesize a concrete, buildable solution covering the five components in my brief:

1. A scalable dataset pipeline for realistic noisy-clean speech pairs.
2. The AI/ML model architecture for robust noise suppression.
3. The training framework, including hyperparameters and perceptual loss functions.
4. The real-time inference engine for edge hardware.
5. The prototype setup demonstrating live noise cancellation with primary and reference microphones and headset or communication-unit integration.

For each component:

- Recommend a specific approach and explain why it beats the alternatives for this use case.
- Address the hard cases explicitly: impulsive noise (gunshots, artillery), rapidly changing noise (rotors, drones, vehicles, sirens, wind), and the risk of speech distortion or artifacts.
- Explain how the learned model and the optional lightweight adaptive filter (e.g., LMS) on the reference microphone should work together.

**Evidence standard**

I need this to be accurate, with no fabrication, because I will build from it.

- Cite sources for every architecture, dataset, tool, and reported metric. Include publication or release dates.
- Clearly separate published results from your own estimates.
- Note when reported PESQ, STOI, or SNR figures come from benchmarks whose noise conditions differ from defence scenarios, so the numbers are not treated as directly comparable.
- Where you cannot verify a claim, such as a model's latency on Jetson AGX Orin or the availability or licensing of a defence noise dataset, flag it rather than guess.
- Identify realistic risks to hitting the target metrics within the latency budget, and say how you would mitigate them.

**My full project brief (context)**
```

---

## 3. Follow-up requests (verbatim)

### 3.1 PESQ above 4 (2026-10-03 17:39 UTC)

```text
GREAT AND AWESOME WORK. LOTS OF KUDOS. PLEASE MAKE SURE THAT ALL THE REQUIREMENTS ARE MET AND ALSO IMPROVE THE METRICS AS MUCH AS POSSIBLE FOR THE DEPLOYMENT ON THE HARDWARE BOARD, ALSO I WANT THE PESQ TO BE ABOVE 4 TO MAKE THE PROJECT THE BEST AND TOP NOTCH QUALITY, CONTINUE AND FOLLOW THE INSTRUCTIONS THE PREVIOUS PROMPT. 

<pasted_content id="9f55">
Hard constraints that bound everything below (non negotiable):

🚫 DO NOT DELETE ANY FILES — existing files must be preserved
🚫 DO NOT BREAK existing project structure or functionality — all current behavior must continue working
✅ TEST ALL CHANGES in a virtual environment only before applying anything
✅ Proceed step by step, systematically — do not rush implementation.
</pasted_content id="9f55">
```

### 3.2 PESQ-NB and PESQ-WB of at least 3.25-3.5 for all cases, ablations, Jetson deployment, complete shared folder (2026-10-04 04:36 UTC)

```text
AWESOME AND FANTASTIC JOB. LOTS OF KUDOS. improve PESQ further and run the ablations, I want PESQ for both narrow band and wide band above 3.25 TO 3.5 MINIMUM for all the cases, think like a top AI , SPEECH AND ANC DESIGNER EXPERT FOR THE DEFENCE CASES. THINK ON ALL POSSIBLE EDGE CASES TO IMPROVE THE OVERALL PERFORMANCE, IF POSSIBLE UPDATE THE REQUIRED ARCHITECTURE IF NEEDED BASED ON THE REQUIREMENTS GIVEN AS WE CAN'T COMPROMISE ON THE RESULTS AND ALSO MAKE SURE THE DEPLOYMENT STEPS ON JETSON BOARD IS PROPER, SELF EXPLANATORY TO DEPLOY BY ANY ONE AND UPDATE THE FULL DOCUMENTATION, RESULTS ETC BASED ON THE UPDATED RUNS. ALSO PLEASE MAKE SURE THAT THE SHARED FOLDER SHOULD CONTAIN THE FULL PROJECT CODE, DOCUMENTS, RESULTS FILES ETC, ( IT MUST CONTAIN EVERY FILE RELATED TO THIS PROJECT, WE CAN'T MISS A SINGLE FILE FOR THE SHARING IT SO). 

<pasted_content id="9f55">
Hard constraints that bound everything below (non negotiable):

🚫 DO NOT DELETE ANY FILES — existing files must be preserved
🚫 DO NOT BREAK existing project structure or functionality — all current behavior must continue working
✅ TEST ALL CHANGES in a virtual environment only before applying anything
✅ Proceed step by step, systematically — do not rush implementation.
</pasted_content id="9f55">
```

### 3.2a Progress request during round 2 (2026-10-04 10:11 UTC)

```text
how is the progress till now, continue with the previous prompts instructions and i am curious to see the current results status so.
```

### 3.2b GPU use during round 2 (2026-10-04 12:51 UTC)

```text
Are you not using the GPUs to train the model from the current macbook, I guess you can use them to make it faster training, follow the previous prompts instructions and contiue.
```

Answered in `reports/REPORT.md` §5.1. Every training run used the M4 GPU through PyTorch MPS; a clean batch-size
benchmark showed the GPU saturated at batch 16, and jobs run one at a time.

### 3.2c Sharing the project (2026-10-05 04:08 UTC, and a follow-up a few minutes later)

```text
so give me the  full folder i can share with the others
```

```text
I can upload via google drive so, just point me to the folder
```

Answered by the share folder built next to the project: a complete team-only package (`SHARE_README.md`) and a
licence-safe package for outside reviewers (`SHARE_README_PUBLIC.md`), both checked by the pre-share review
(`docs/research/pre_share_review_2026-10-05.md`) and an independent verification of the built zips.

### 3.3 Summary of the follow-up requirements

1. **PESQ above 4** (request of 2026-10-03). How this is handled, and what the literature shows is achievable:
   `reports/REPORT.md` §1 ("On PESQ > 4") and §8.
2. **PESQ narrow-band and wide-band at or above 3.25-3.5 for all cases** (request of 2026-10-04), together with:
   - run the ablations;
   - consider all edge cases relevant to defence use;
   - update the architecture if the requirements need it;
   - make the Jetson deployment steps correct and self-explanatory enough for anyone to follow
     (`deploy/jetson/README.md`);
   - update all documentation and results from the new runs;
   - make the shared folder complete: every code, document and result file of the project
     (see `FILE_INDEX.md` and `docs/SHARING_CHECKLIST.md`).
3. **Hard constraints, repeated with both follow-ups (non-negotiable):**
   - do not delete any file; existing files must be preserved;
   - do not break the existing project structure or functionality; all current behaviour must keep working;
   - test every change in a virtual environment before applying it;
   - proceed step by step and systematically; do not rush the implementation.
4. **Use the MacBook's GPU for training** (2026-10-04): all runs on the M4 GPU (MPS); `reports/REPORT.md` §5.1.
5. **Share the full project folder** (2026-10-05): a complete team package (`SHARE_README.md`) and a licence-safe
   public package (`SHARE_README_PUBLIC.md`), verified before upload.
6. **From the first request:** no file deletion; implement only inside a virtual environment; test and
   validate; write a report; share the full project code, data and reports later for verification.

---

## Appendix A. First message exactly as sent (verbatim, including the second pasted copy of the brief)

The owner pasted the brief twice in the first message. The second copy has the same content but differs in
line breaks, heading markers (`* Description:`), some punctuation (for example `. while its also` instead of
`, while it also`) and a text-encoding artefact (`â€”` where the first copy has a comma). §1 above uses the
first copy. The full message is kept here unchanged for completeness. The leading `@".../21095.pdf"` is the
attached reference paper (now `docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf`).

```text
@"/Users/nsivaprasadnandyala/Downloads/21095.pdf"

<pasted_content id="9f55">
I want a complete, evidence-backed technical solution for a hybrid AI/ML adaptive noise cancellation (ANC) system. It must suppress **stationary, non-stationary, and impulsive defence noise** while preserving speech intelligibility, and it must run in **real time on embedded/edge hardware** (NVIDIA Jetson AGX Orin 64GB Developer Kit or a comparable DSP or AI-enabled SoC). The system must reach **SNR > 15 dB, STOI > 0.85, and PESQ > 2.5** at latency low enough for live two-way communication. Every design choice you make should be justified against these targets and the edge latency and power budget. A model that scores well offline but cannot run in real time on the target hardware does not solve my problem.

Approach this as a senior audio DSP engineer specializing in deep-learning speech enhancement and embedded deployment for defence communications.

**What I need from you**

Research the current state of the art using web search, recent papers, benchmarks, and open-source implementations. Then synthesize a concrete, buildable solution covering the five components in my brief:

1. A scalable dataset pipeline for realistic noisy-clean speech pairs.
2. The AI/ML model architecture for robust noise suppression.
3. The training framework, including hyperparameters and perceptual loss functions.
4. The real-time inference engine for edge hardware.
5. The prototype setup demonstrating live noise cancellation with primary and reference microphones and headset or communication-unit integration.

For each component:

- Recommend a specific approach and explain why it beats the alternatives for this use case.
- Address the hard cases explicitly: impulsive noise (gunshots, artillery), rapidly changing noise (rotors, drones, vehicles, sirens, wind), and the risk of speech distortion or artifacts.
- Explain how the learned model and the optional lightweight adaptive filter (e.g., LMS) on the reference microphone should work together.

**Evidence standard**

I need this to be accurate, with no fabrication, because I will build from it.

- Cite sources for every architecture, dataset, tool, and reported metric. Include publication or release dates.
- Clearly separate published results from your own estimates.
- Note when reported PESQ, STOI, or SNR figures come from benchmarks whose noise conditions differ from defence scenarios, so the numbers are not treated as directly comparable.
- Where you cannot verify a claim, such as a model's latency on Jetson AGX Orin or the availability or licensing of a defence noise dataset, flag it rather than guess.
- Identify realistic risks to hitting the target metrics within the latency budget, and say how you would mitigate them.

**My full project brief (context)**

"""
title
To develop an AI/ML-enabled adaptive noise cancellation (ANC) system that effectively suppresses stationary, non-stationary, and impulsive defence noises while maintaining high speech intelligibility and real-time performance on embedded hardware

Description
In defence and mission-critical communication systems, reliable speech transmission is severely affected by diverse acoustic disturbances such as gunshots, artillery fire, helicopter rotor noise, armored vehicle sound and emergency sirens. Traditional signal processing techniques, like spectral subtraction, Wiener filtering, and classical LMS-based ANC, are limited in handling highly dynamic and non-linear noise environments. These methods assume stationary noise characteristics and often introduce artifacts or speech distortion under rapidly changing conditions.

Recent advancements in Artificial Intelligence and Machine Learning (AI/ML) have transformed the field of speech enhancement and ANC. Deep learning models and time-domain architectures are capable of learning complex spectral-temporal patterns directly from data. These models significantly outperform conventional approaches in terms of perceptual quality (PESQ), intelligibility (STOI), and noise suppression (SNR). Additionally, the rise of edge AI platforms enables deployment of such models on embedded systems for real-time applications.

Description:
The proposed system integrates AI/ML-driven noise suppression with adaptive filtering to create a robust ANC pipeline. The development begins with dataset generation, where clean speech data is combined with curated defence noise datasets (gunshots, drones, artillery, vehicle engines, wind, etc.) at varying SNR levels. This synthetic data generation ensures coverage of both stationary and impulsive noise scenarios.

The training pipeline involves transforming audio into time-frequency representations (e.g., STFT spectrograms) or directly using raw waveform inputs. Models process both full-band and sub-band features to capture global and local dependencies, while it also operates in the complex domain to preserve phase information. Training is performed using loss functions such as SI-SNR, L1/L2 loss, and perceptual loss, with evaluation metrics including SNR, STOI, and PESQ. Data augmentation techniques (random noise mixing, reverberation, clipping) are applied to improve generalization.

During inference, the trained model processes incoming noisy audio in real time, estimating a mask or directly reconstructing enhanced speech. The system can optionally include a lightweight adaptive filter (e.g., LMS) for residual noise suppression.

For prototype demonstration, the trained model is deployed on embedded/edge hardware such as DSPs or AI-enabled SoCs (e.g., NVIDIA Jetson AGX Orin 64GB Developer Kit or similar platforms). Optimization techniques like quantization, pruning, and ONNX / TensorRT conversion are applied to meet latency and power constraints. The system is integrated with microphones (primary + reference) and headphones/communication units to validate real-time ANC performance in practical environments

Expected Solution:
The final solution is a hybrid AI-driven ANC system capable of operating in real-time and handling diverse noise environments, including impulsive and highly dynamic defence scenarios. It should include:
* A scalable dataset pipeline for generating realistic noisy-clean speech pairs
* A state-of-the-art AI/ML model trained for robust noise suppression
* A training framework with optimized hyper-parameters and perceptual loss functions
* A real-time inference engine deployable on edge hardware
* A prototype system demonstrating live noise cancellation using microphones / headset integration
The system is expected to achieve significant performance improvements, targeting SNR > 15 dB, STOI > 0.85, and PESQ > 2.5, while maintaining low latency suitable for real-time communication. This solution will enable reliable and intelligible communication in defence, aerospace, and high-noise industrial environments.
"""
</pasted_content id="9f55">

  

<pasted_content id="9f55">
title
To develop an AI/ML-enabled adaptive noise cancellation (ANC) system that effectively suppresses stationary, non-stationary, and impulsive defence noises while maintaining high speech intelligibility and real-time performance on embedded hardware
Description
In defence and mission-critical communication systems, reliable speech transmission is severely affected by diverse acoustic disturbances such as gunshots, artillery fire, helicopter rotor noise, armored vehicle sound and emergency sirens. Traditional signal processing techniquesâ€”like spectral subtraction, Wiener filtering, and classical LMS-based ANCâ€”are limited in handling highly dynamic and non-linear noise environments. These methods assume stationary noise characteristics and often introduce artifacts or speech distortion under rapidly changing conditions.

Recent advancements in Artificial Intelligence and Machine Learning (AI/ML)

have transformed the field of speech enhancement and ANC. Deep learning models and time-domain architectures are capable of learning complex spectral-temporal patterns directly from data. These models significantly outperform conventional approaches in terms of perceptual quality (PESQ), intelligibility (STOI), and noise suppression (SNR). Additionally, the rise of edge AI platforms enables deployment of such models on embedded systems for real-time applications.

* Description:

The proposed system integrates AI/ML-driven noise suppression with adaptive filtering to create a robust ANC pipeline. The development begins with dataset generation, where clean speech data is combined with curated defence noise datasets (gunshots, drones, artillery, vehicle engines, wind, etc.)

at varying SNR levels. This synthetic data generation ensures coverage of both stationary and impulsive noise scenarios.

The training pipeline involves transforming audio into time-frequency representations (e.g., STFT spectrograms) or directly using raw waveform inputs. Models process both full-band and sub-band features to capture global and local dependencies. while its also operates in the complex domain to preserve phase information. Training is performed using loss functions such as SI-SNR, L1/L2 loss, and perceptual loss, with evaluation metrics including SNR, STOI, and PESQ. Data augmentation techniques (random noise mixing, reverberation, clipping) are applied to improve generalization.

During inference, the trained model processes incoming noisy audio in real time, estimating a mask or directly reconstructing enhanced speech. The system can optionally include a lightweight adaptive filter (e.g., LMS) for residual noise suppression.

For prototype demonstration, the trained model is deployed on embedded/edge hardware such as DSPs or AI-enabled SoCs (e.g., NVIDIA Jetson AGX Orin 64GB Developer Kit or similar platforms). Optimization techniques like quantization, pruning, and ONNX / TensorRT conversion are applied to meet latency and power constraints. The system is integrated with microphones (primary + reference) and headphones/communication units to validate real-time ANC performance in practical environments

* Expected Solution:

The final solution is a hybrid AI-driven ANC system capable of operating in real-time and handling diverse noise environments, including impulsive and highly dynamic defence scenarios. It should include:

* A scalable dataset pipeline for generating realistic noisy-clean speech pairs
* A state-of-the-art AI/ML model trained for robust noise suppression
* A training framework with optimized hyper-parameters and perceptual loss functions
* A real-time inference engine deployable on edge hardware
* A prototype system demonstrating live noise cancellation using microphones / headset integration The system is expected to achieve significant performance improvements, targeting SNR > 15 dB, STOI > 0.85, and PESQ > 2.5, while maintaining low latency suitable for real-time communication. This solution will enable reliable and intelligible communication in defence, aerospace, and high-noise industrial environments.
</pasted_content id="9f55">

 DON'T DELETE ANY FILES, CREATE A VIRTUAL ENVIRONMENT AND IMPLEMENT THE PROJECT IN VIRTUAL ENVIRONMENT ONLY. FOLLOW THE INSTRUCTIONS AND THINK LIKE A TOP SPEECH PROCESSING EXPERT. AFTER COMPLETING THE CODE DEVELOPMENT, TEST AND VALIDATE IT, CREATE A REPORT AND I NEED TO SHARE THE FULL PROJECT CODES, DATA , REPORTS LATER FOR VERIFICATION.
```
