# Round-2 file-index refresh: reviewer's hand-back

*The agent that refreshed `FILE_INDEX.md` and `docs/SHARING_CHECKLIST.md` on 2026-10-05; final report delivered 2026-10-05T03:47:36.069Z (UTC). The inconsistencies it listed were fixed afterwards.* Exported verbatim from the agent's final report in the Claude Code session transcript.

---

I updated `FILE_INDEX.md` and `docs/SHARING_CHECKLIST.md`. I changed no other file, deleted or moved nothing, and ran no training, evaluation or tests (read-only shell commands only). Every item on your round-2 list exists in the tree as described.

**What changed in `FILE_INDEX.md`** (now 508 lines, re-check stamped 2026-10-05 07:40 +04, same structure and type legend)
- **Header:** new sizes and counts: about 6.2 GB / 82 800 files with `.venv`, about 4.9 GB / 53 700 without. I added a table of the three deployed configurations: HQ = `v3hq_cont`, LL = `ll`, TWO_MIC = NLMS → `v3_2mic` cascade with HQ fallback.
- **§1:** all sizes and counts updated. New rows for `dist/` (absent from the previous index), the four `.DS_Store` files and `.pytest_cache`.
- **§2 to §5:** every new or extended source file, config, script and test is listed, with outcomes from the decision files. `src/x` and `dist/jetson_bundle/src/x` are listed as empty directories. Tests: 103 passed (`logs/pytest_round2_final.out`).
- **§6:** the Jetson kit table is updated, including `QUICKSTART.md` and `check_channels.py`. A new §6.1 covers `dist/jetson_bundle` (116 files, 13 MB; `SHA256SUMS` all OK when I checked; still contains the round-1 `dancnet_hqft` files).
- **§7:** all 21 checkpoint directories and all 23 export files.
- **§8:** the three new test sets, with file counts and licence inheritance.
- **§9:** `REPORT_round1.md`, all 7 result sets with their system counts, every decision rule and decision, the table snapshots, statistics, latency and self-check JSON files, figures and `figures/round1/`, all 24 audio folders.
- **§10:** docs.
- **§11:** all 113 logs, grouped into training, queue, round-1 and round-2 tables.
- **§13:** replaced "Work in progress" with the actual state: nothing running, round 2 complete, three open items.
- **§14:** manifest status. The current `reports/MANIFEST.sha256` (07:27, 52 589 files) must be regenerated last. `make_manifest.py` now reads `exports/deployment_selection.json`, so the old hard-coded-list warning is gone.

**What changed in `docs/SHARING_CHECKLIST.md`** (same five sections)
- New sizes, a list of round-2 additions to include, and the three deployed configurations.
- How to send the Jetson bundle on its own, without adding files to the original folder.
- Verification steps: `shasum -a 256 -c reports/MANIFEST.sha256`, the bundle's `SHA256SUMS`, `python -m pytest -q` (103 tests) and `python deploy/jetson/verify_install.py` (115 PASS, 0 FAIL on the Mac).
- Licence table now covers the new test sets, the round-2 models and the bundle.
- Final checklist updated with the current status of each item.

**Files I could not classify:** none. Two only have an unknown producer: `logs/eval_v3hq_chain.done` (an "ALLDONE" marker; no script for it is in the project) and the v3hq ONNX export, which has no log. The v3hq commands are in REPORT §10.

**Missing or inconsistent (not fixed):**
1. REPORT.md §10 says the chain log that wrongly recorded "v3_2mic finished" "is annotated". `logs/chain2b.out` has no annotation (first line: `[chain2b] v3_2mic finished Sun Oct 4 17:06:19`).
2. `reports/results/cascade_decision.md`, in its "Reference failure" paragraph, recommends `--fallback-onnx exports/dancnet_v3hq.onnx`. The adopted launch line and `deployment_selection.json` use `dancnet_v3hq_cont.onnx`.
3. `reports/site/index.html` is still the round-1 build (2026-10-04 08:27, no v3 results). `scripts/build_report_html.py` was modified at 07:33 today, so someone may be about to rebuild it. Listed as a to-do.
4. `docs/workflows/journals/wf_14dd115a-9e9.jsonl` is still the partial 5 040-byte copy from 08:54. That workflow's state file in the session folder was last written at 09:30 on 2026-10-04, so the full journal was never copied in.
5. `reports/MANIFEST.sha256` (07:27) lacks 10 files written after it and has stale hashes for README, the docs, the Jetson kit and others. It also lists the `.DS_Store` files, which macOS Finder can rewrite.
6. The `configs/smoke/*.yaml` comments say "see tests/test_train_resume.py", but that test does not use those configs. It tests `restore_ema` in memory; the smoke runs were done by hand.
7. Stale docstrings:
   - `deploy/jetson/verify_install.py` still says "Both deployed ONNX models" with round-1 default reference systems. The actual run checks all three via `DEPLOYMENT.json`.
   - `tests/test_deploy_kit.py` says "both deployed models".
8. `reports/figures/round2_training_curves.png` is generated but not referenced in REPORT.md.
9. Not refreshed for round 2: `reports/audio/demo/`, `reports/results/simulated_live_runs.json` and `latency_mac_m4.json` still use round-1 models. The index says so.
10. Missing from the previous index although they existed then: `reports/audio/enhanced_defence_edge_v1_dancnet_hq_metric/`, `deploy/jetson/check_channels.py` and all of `dist/`. All three are now listed.
11. Three orphaned, idle Python worker processes (PIDs 17983–17985, started 23:00:17 on 2026-10-04, parent launchd) are still alive. Their start time matches the aborted high-SNR evaluation. They write nothing, but you may want to end them before the "no jobs running" check.

Files changed:
- /Users/nsivaprasadnandyala/Desktop/DRDO_Project/FILE_INDEX.md
- /Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/SHARING_CHECKLIST.md
