# Round-2 independent code review (read-only)

*Requested 2026-10-04 ~23:14 +04 after the round-2 code changes; final report delivered 2026-10-04T19:28:00.545Z (UTC). Every CONFIRMED item was fixed before the affected code ran (see `reports/REPORT.md` §10, round-2 incidents). Reproducers: `docs/research/round2_review_repro/`.* Exported verbatim from the agent's final report in the Claude Code session transcript.

---

Review findings: D-ANC round-2 changes (read-only review; nothing in the project folder was created or changed)

**Tests:** `python -m pytest -q tests/test_two_mic.py tests/test_deploy_kit.py` gives **30 passed** in about 36 s, exit code 0. On the first run, the known ONNX Runtime teardown abort ("libc++abi … recursive_mutex lock failed") printed after the summary. The rerun was clean.

## CONFIRMED (reproduced, or certain from the code)

**1. `src/danc/train/train_dual.py:99-102` and `:138` — the EMA is not actually restored on `--resume`. This is the most urgent item.**
- **What is wrong:** `AveragedModel` starts with `n_averaged == 0`. Because of that, the first `ema.update_parameters(model)` copies the raw weights over the EMA weights that were just loaded.
- **Reproduced:** on torch 2.14.1, after one update the EMA equals the raw model (scratch `ema_resume.py`).
- **Real impact:** in `checkpoints/v3_2mic/last.pt` (step 2000) the largest gap between EMA and raw weights is 8.9. `run_round2c_chain.sh` runs `train_dual --resume` as soon as `abl_nopmsqe` finishes, and that job is training now. So the resumed two-mic run will throw away its EMA after the first step.
- **Fix:** after loading, call `ema.n_averaged.fill_(max(1, start_step))`, or save `ema.state_dict()` (which includes `n_averaged`) and load that.
- **Same latent bug** in `src/danc/train/train.py` (~lines 133-134, single-mic resume).

**2. `scripts/make_tables.py:151` `snr_needed()` — misleading result at the lowest tested SNR.**
- **What is wrong:** if the lowest tested SNR already meets the PESQ target, it prints that SNR as if it were the exact crossing point.
- **Reproduced:** means of 3.4/3.6/3.8 at −5/0/5 dB give "−5.0 dB" for ≥ 3.25. The true crossing is ≤ −5 dB and unknown.
- **Current tables:** not affected (no system reaches the target at its lowest SNR).
- **Fix:** when `prev is None`, print `≤ {snr_v} dB`.

**3. `src/danc/data/build_dual_babble.py:79` — `snr_db` holds the target-to-interferer ratio (TIR), not the SNR.**
- **What is wrong:** the diffuse bed adds noise 10 dB below the wearer. I measured the built set: the real SNR at the primary is −0.41 / 3.81 / 7.00 dB for TIR 0 / 5 / 10.
- **Effect:** `evaluate.summarize` (`by_snr`) and any per-SNR view label these as 0 / 5 / 10 dB SNR.
- **Fix:** add a `tir_db` column and store the true SNR in `snr_db`, or label the axis "TIR" everywhere.

**4. `src/danc/inference/engine.py:124-139` `_ref_health` — reproduced with stub runners.**
- **Quiet room:** a live reference at −75 dBFS for 1 s, then −40 dBFS for 1 s, repeated, gives **6 switches in 6 s**. `ref_dead=True` whenever the room is quiet. A real headset in a quiet room (around −85 dBFS) with speech pauses over 0.5 s will keep flipping between the two networks, and the live log reports the mic as dead.
- **DC offset:** a dead mic with 5 mV DC is never detected, because the level includes DC.
- **Hiss in the band:** a dead input whose preamp hiss sits at −70…−64 dBFS (inside the 6 dB hysteresis band) is never detected.
- **Fix:**
  - Remove DC first (`ref - ref.mean()`), or measure on the STFT reference bins ≥ 2.
  - Add a relative test, e.g. reference below primary − 35 dB while the primary is active.
  - Smooth the level over about 100 ms.
  - Expose `ref_dead_dbfs` / `ref_dead_s` on the command line.
- **Checked and correct:**
  - With no fallback, behaviour is unchanged (compared against the round-1 copy in `dist/`).
  - Output is bit-exact while mix = 0, including after a dead → alive cycle, because the main runner always gets the same inputs.
  - `reset()` clears the monitor and fallback state.
  - `ref=None` switches at once, with a 10-hop cross-fade.
  - Look-ahead equality is enforced, and `self.spp` is not used on the two-mic path.

**5. `src/danc/inference/realtime.py:64-65` — the live log always shows `ref_dead=false, ref_switches=0`, even with no fallback (monitor not running).**
- **Effect:** for a two-mic model run without `--fallback-onnx`, a really dead reference is reported as alive.
- **Fix:** add these fields only when `eng.fallback is not None`, as `run_simulated` already does.
- **Pass-through works:** `deploy/jetson/danc_live.py --onnx two.onnx --fallback-onnx one.onnx --simulate …` with `SafeHybridEngine` and test models gave `ref_switches 1`, `ref_dead_at_end true`, exit 0. A single-mic main model raises `ValueError` as intended.

## POSSIBLE / latent

**`engine.py:161`**
- At mix = 1 the code computes `0*y + 1*y1`. NaN/Inf from the two-mic network fed a dead reference therefore still reaches the output (plain engine / `--no-safety`). Fix: when `_mix >= 1`, use `y, spp = y1, spp1`.
- An int16 `ref` overflows in `np.square`, the level becomes NaN, and the mic is never declared dead. Callers cast to float today.

**`realtime.py:57,95`** — `--no-lms` silently drops the reference for a two-mic network too (older behaviour). Without a fallback the network then gets zeros; with one it switches at once.

**`scripts/benchmark_latency.py:71`**
- A two-mic model is timed without the hot-standby fallback.
- If the two-mic adoption rule's criterion 3 fails, the model may only ship together with the fallback. Per-hop compute then covers both networks, so criterion 4 and table R14 would be measured on the wrong configuration. Suggest adding a "two-mic + fallback" row.
- Also: `--frames ≤ 200` crashes on an empty array, and the file name and plot label are hard-coded to the Mac M4.

**`train_dual.py` resume, other points**
- The optimizer state is neither saved nor restored (`train.py` does save `opt`). AdamW restarts from fresh state at lr ≈ 2.2e-4 (cosine at step 2000) with no warm-up, so a loss spike is possible.
- The `max_hours` budget restarts at each resume.
- `--resume` without a `last.pt` silently warm-starts, and can then overwrite an existing `best.pt`.
- `init_val` after a resume is the score at the resume step.
- Older behaviour: nothing is saved when the loop stops on `max_hours`. Batches with a non-finite loss use data without advancing `step`, so `step == max_steps` may never be reached and the final validation and save are skipped.
- Fine: the data seed (`[seed*1000+start_step, idx]` gives fresh samples), the LR continuing from `start_step`, the best-score restore, and the `start_step` val log line.

**Labels, `scripts/make_tables.py:33-45`**
- `deployment()` marks `dancnet:<run>` as deployed, but `evaluate.make_system` evaluates `checkpoints/<run>/best.pt` if it exists, else `last.pt`. Consistent today: `hq_metric` has only `last.pt`, and LL uses `ll/best.pt`.
- If a `best.pt` ever appears in `hq_metric`, or the selection points at `v3hq/last.pt` (`v3hq/best.pt` differs), the "deployed" row would show another checkpoint's numbers. Fix: assert the checkpoint file name in `deployment()`.
- The latency in the label comes from the mode name (HQ → 40 ms), not from the model's look-ahead.
- I ran `make_tables` to scratch: the deployed labels match `deployment_selection.json` (hq_metric/hqft and ll/ll).

**`scripts/paired_stats.py`**
- The "deployed v2 HQ/LL" text is hard-coded instead of read from `deployment_selection.json`.
- Ablation names are exact strings; the `_bNN` suffix is not handled as it is in `make_tables` and `make_figures_round2` (the suffix is empty today, so no effect yet).
- An empty id intersection crashes `boot_ci` (`integers(0, 0)`).
- `n` is counted before the NaN filter.
- The maths is fine: pairing by id via index alignment is correct, and the bootstrap CI matches a normal-approximation CI (e.g. ΔPESQ-WB +0.043 [0.034, 0.051]). I ran it against a scratch copy, so the project's `paired_stats.json` was not written.

**`src/danc/eval/evaluate.py:116`** — `dnn2_refdead` shares one RNG across all files, so each file's noise depends on file order and `--limit`. Seed per file instead. The −80 dBFS level (RMS 1e-4) is correct.

**`deploy/jetson/danc_jetson_rt.py` timing (item 3)**
- The `_trim_timing` fix is correct for both list and deque and keeps `maxlen`.
- Nothing else in `deploy/jetson/*.py`, `src/danc/inference/*.py`, `scripts/` or `tests/` slices or deletes from `timing`.
- Nits:
  - The wrapper trims to 15-30 s, so `run_live`'s "last 60 s" (`timing_window = 6000`) never applies.
  - The docstrings at `danc_jetson_rt.py:217-221` and `verify_install.py:410` still say stats run inside the audio callback.
  - `clear()` + `extend()` is not atomic against the logging thread, so a log line can rarely lack stats.
- **`dist/jetson_bundle/` is stale round-1 code** (list timing plus `del self.timing[:…]`). It is consistent with itself, but rebuild it before shipping and never mix the new `src/` with the old `deploy/`.

**`scripts/make_figures_round2.py`** — `edge_cases()` divides by zero if no system is present. The 16000-step offset for `v3hq_cont` is correct (v3hq best is at step 16000).

**`scripts/band_error_analysis.py`**
- The maths is fine: 50 Hz bins, LSD averaged over active frames.
- The top band leaves out the 8 kHz bin.
- The "speech-active" threshold (−60 dB relative to the loudest frame) is very loose.

**`build_dual_babble.py`, physics checked**
- The `replace(...)` fields exist.
- `max_peak=None` in both calls means clean is exactly the speech part of the primary.
- Both calls use the same speech and the same mic spacing, so scaling is consistent.
- The final peak scaling is applied to all three signals together.
- Nits:
  - Self-noise is added twice (+3 dB, negligible).
  - The talker count is re-drawn on every loop pass (still 2-3, 50/50).
  - The talkers' active-power normalisation is overridden by `simulate_scene`.

**`src/danc/data/build_highsnr.py`** — the same recipe as `build_defence`, and its ids (`p20` / `p25`) do not collide with `defence_v1`.

**`src/danc/models/dancnet.py:348` `macs_per_frame`** — correct: only the ERB features and the first encoder conv scale with `in_ch`.

**`deploy/jetson/verify_install.py`**
- The new 30 s check runs through the engine's own deque, which is what it should test.
- Output is flushed and the JSON is written inside `main()` before `os._exit`.
- Bad arguments still exit with code 2, because argparse's `SystemExit` never reaches `os._exit`.
- Nit: the check name has no model prefix, so it appears twice with the same name (HQ and LL).

Scratch files are in `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/review/`: `ema_resume.py`, `refhealth.py`, `mk_models.py`, `ps_run.py`, `tables_now.md` and `pytest.out`.
