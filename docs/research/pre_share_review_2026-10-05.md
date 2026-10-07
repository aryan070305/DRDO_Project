# Pre-share review of the project folder (2026-10-05)

> **Dated record.** This review ran before its findings were acted on. Counts and statements here (e.g. 52 629 manifest
> lines, `.DS_Store` in the manifest, building with `zip -r -X`) describe the folder at that moment; the final state is in
> `SHARE_README.md` §2, `SHARE_README_PUBLIC.md` and `FILE_INDEX.md` §13-§14.

*Four independent read-only reviewers (workflow `wf_c549189e-2c8`, script `docs/workflows/pre-share-review-wf_c549189e-2c8.js`) checked the folder before it was packaged for Google Drive. Their findings are reproduced below; what was done about each is in `SHARE_README.md` and `FILE_INDEX.md` §13.*

## Secrets and personal data

I found no credentials, private keys, tokens, credential files or owner contact details in any file that will go into the archive, including the third_party packages, the 34 checkpoints, the 14 ONNX models and the project-generated audio. Two things are worth the owner's attention before sharing. First, docs/workflows/ holds the AI-assistant session's workflow state, journals and scripts, with session and agent IDs, token counts and the owner's time zone. Second, docs/PROJECT_BRIEF.md reproduces the owner's raw chat messages to the assistant word for word. The local home path /Users/nsivaprasadnandyala, which contains the owner's name, appears 193 times in 59 text files, mostly logs and workflow records. It does not appear in any config, checkpoint train_cfg, ONNX metadata, manifest, CSV or binary file.

### [should-fix] AI-assistant session records (workflow state, journals, scripts) are inside the archive

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/wf_14dd115a-9e9.json`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/wf_33c61465-a05.json`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/wf_5597f002-1ea.json`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/wf_79848172-d66.json`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/wf_c64771c1-4db.json`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/wf_d657bd8c-717.json`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/wf_d812ebf9-67c.json`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/README.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/journals/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/*.js`

**Evidence:** docs/workflows/state/README.md says these files were "copied on 2026-10-05 from the Claude Code session folder". All 7 state JSONs are one-line files (line 1). Their scriptPath fields give the Claude Code session UUID and folder: /Users/nsivaprasadnandyala/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/workflows/scripts/... They also contain internal taskIds (e.g. "w23k503yd"), per-agent agentIds (e.g. "a1ee1ab605f2dbced", also in all 7 journals/*.jsonl), defaultModel "claude-opus-5-5", totalTokens (e.g. 1,022,909 and 1,398,570) and totalToolCalls. Four state files (wf_33c61465-a05, wf_79848172-d66, wf_d657bd8c-717, wf_d812ebf9-67c) have logs like "You've hit your session limit · resets 8:20pm (Asia/Dubai)", which gives the owner's location and time zone. The script fields hold full agent prompts, including the owner's "HARD RULES (from the user, non-negotiable)". The result fields hold agent narratives, e.g. wf_14dd115a-9e9.json: "One thing went wrong along the way: my packager briefly ran with the project root as its output". They also reference /private/tmp/claude-502/... scratchpad and tool-results paths (wf_14dd115a-9e9.json and journals/wf_14dd115a-9e9.jsonl). The 22 docs/workflows files hold 82 of the 193 home-path occurrences. They contain no credentials. docs/SHARING_CHECKLIST.md:210 already flags these files for review.

**Recommendation:** For the internal team this is acceptable if the owner confirms it. For reviewers outside the team, leave docs/workflows/state/ and docs/workflows/journals/ (and possibly the *.js scripts) out of the zip; do not delete them, per the owner's rules. reports/MANIFEST.sha256 lists 22 docs/workflows/ entries (8 under state/), so excluding them makes `shasum -c` report missing files. Regenerate the manifest for the shared copy, or say so in SHARE_README.md. SHARE_README.md §1 ("Nothing else was left out") would also need updating.

### [should-fix] docs/PROJECT_BRIEF.md reproduces the owner's raw chat messages to the AI assistant word for word

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/PROJECT_BRIEF.md`

**Evidence:** Lines 8-9 say the file holds "the project owner's chat messages to the development assistant". Line 10 says "the development Mac ran at UTC+04" (FILE_INDEX.md:7 has the same). Line 184 (Appendix A) is the raw attachment line @"/Users/nsivaprasadnandyala/Downloads/21095.pdf". Lines 110-146 (§3.1-3.2b) quote informal prompts word for word, with <pasted_content id="9f55"> markers, e.g. "GREAT AND AWESOME WORK. LOTS OF KUDOS... I WANT THE PESQ TO BE ABOVE 4", "think like a top AI , SPEECH AND ANC DESIGNER EXPERT" and "Are you not using the GPUs to train the model...". Appendix A (lines 175-284) is the whole first prompt, including "Approach this as a senior audio DSP engineer...". There are no secrets or contact details. docs/SHARING_CHECKLIST.md:210 flags this file too.

**Recommendation:** The owner should confirm they want reviewers to see their raw prompts. Otherwise share a version that keeps §1 (the brief), the targets table and the §3.3 summary, and drops Appendix A and the word-for-word informal messages (§2, §3.1-3.2b). Or leave the file out of the zip; do not delete it. If it is changed or excluded, the MANIFEST.sha256 entry for docs/PROJECT_BRIEF.md needs regenerating.

### [note] Absolute home path with the username (= the owner's name) in 59 text files, 193 occurrences; none in configs, checkpoints, ONNX or data

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/logs/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/logs/simulated_live_round2/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_pkgs/bin/f2py`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_pkgs/bin/deepFilter`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_pkgs/bin/deep-filter-py`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/PROJECT_BRIEF.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/FILE_INDEX.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/BUNDLE_INFO.txt`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/results/verify_install_mac_round2.json`

**Evidence:** I grepped every non-audio file in the archive scope, binary files included. Results by folder:
- logs/: 37 files. 21 top-level files (e.g. v3_2mic.out ×12, pytest_round2_final.out ×9, download_partial.log ×9 tracebacks into ~/.local/share/uv/python, export_v3*.out ×6) plus all 16 logs/simulated_live_round2/*.out (line 1 echoes the .venv python path).
- docs/workflows/: 15 files (7 state .json, 7 .js, 1 .jsonl).
- third_party/dfn_pkgs/bin/: 3 shebangs (#!/Users/nsivaprasadnandyala/Desktop/DRDO_Project/.venv/bin/python3). These scripts will not run on a recipient's machine.
- 4 single files: docs/PROJECT_BRIEF.md:184, FILE_INDEX.md:511 (~/.claude/projects/-Users-nsivaprasadnandyala-...), dist/jetson_bundle/BUNDLE_INFO.txt:3 (source: path) and reports/results/verify_install_mac_round2.json:10,107.
The path is absent from: all 43 YAML configs (configs/ and checkpoints/*/config.yaml); checkpoint train_cfg (sampled v3hq_cont/best.pt, v3_2mic/last.pt and abl_base/best.pt; only the relative path 'checkpoints/v3hq/best.pt'; a raw grep of all 34 .pt files found none); the metadata of all 14 ONNX models (only fs, n_fft, hop, win, lookahead, in_ch, algorithmic_latency_ms, model_cfg); data/manifests/*.json, CSVs, PNGs, PDF, npz/npy/pkl/mat; and the audio in reports/audio, data/testsets and data/synthetic_examples. The machine hostname (SIVAPRASADs-MacBook-Air) appears nowhere. The only other home paths are upstream ones inside third_party (/Users/runner, /Users/trentm, /home/guido).

**Recommendation:** Harmless for the team. docs/SHARING_CHECKLIST.md §4 already names these locations. If outside reviewers should not see the name, the simplest fix is in the shared copy only: leave out logs/, or scrub the 59 files with sed in a copy. Never edit the originals.

### [note] macOS extended attributes: the reference PDF records a WhatsApp download origin; Finder 'Compress' would put xattrs into __MACOSX/

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf`

**Evidence:** `xattr -l` and `mdls` show com.apple.metadata:kMDItemWhereFroms = "https://web.whatsapp.com/" and a com.apple.quarantine entry on this PDF. It is the only file in the project with WhereFroms or quarantine. About 54,000 files carry the opaque com.apple.provenance attribute. A plain `zip -r` does not store xattrs. Finder 'Compress' and `ditto -c -k --sequesterRsrc` store them as __MACOSX/._* AppleDouble entries, which would reveal how the paper was received and add clutter.

**Recommendation:** Build the archive with `zip -r -X ...` (excluding .venv, __pycache__, .pytest_cache and .DS_Store as planned) or `ditto -c -k --norsrc --noextattr`. Then check that `unzip -l` shows no __MACOSX/ entries.

### [note] No credentials, keys, tokens or credential files found

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project`

**Evidence:** I scanned all text files, third_party included, and the model and data binaries (.pt, .onnx, .npz, .npy, .pkl, .mat, .png, .pdf, .zip, .fits) for sk-, AKIA/ASIA, gh[pousr]_, github_pat_, hf_, xox*, AIza, -----BEGIN ... PRIVATE KEY, JWTs, ya29., glpat-, sk_live_ and ssh-rsa/ed25519. There were 0 hits. Keyword hits for secret, password, api_key, token and bearer were only LibriSpeech book titles and transcripts (data/raw/LibriSpeech/BOOKS.TXT, CHAPTERS.TXT, *.trans.txt) and workflow prompt text. There are no .env, .netrc, .npmrc, .pypirc, .pem, .key, .p12, cookie or credentials files. The only hidden files are empty 0-byte lock files (third_party/dfn_pkgs/.lock and 7 reports/results/*/.per_file.lock). Neither .service file holds secrets (User=danc, /opt/danc paths). There are no URLs with token, key or signature query parameters, no Google Drive, Dropbox or claude.ai links, and no environment dumps. The only IP-like strings are section numbers such as 5.3.6.5. src/danc.egg-info/PKG-INFO and pyproject.toml have no author fields.

**Recommendation:** None needed.

### [note] Personal data: only cited or upstream author contacts; no owner email, phone or name outside the home path

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/src/danc/train/third_party/pmsqe.py`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/src/danc/train/third_party/pmsqe.py`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_pkgs/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/LibriSpeech/SPEAKERS.TXT`

**Evidence:** Outside third_party there is one email address: pmsqe.py:36 "Implemented by Juan M. Martin. Contact: mdjuamart@ugr.es" (the cited PMSQE author), which also appears in the dist copy. third_party/dfn_pkgs contains 17 distinct upstream maintainer emails in package metadata. LibriSpeech SPEAKERS.TXT lists public-dataset reader names. The owner's email and name strings (siva, nandyala, boardgenix) appear nowhere except inside the /Users/nsivaprasadnandyala path. There are no phone numbers: every 'phone' hit is 'microphone', and the numeric hits in logs/*.jsonl are loss values. Location appears only as the time zone (Asia/Dubai, UTC+04), covered in the two should-fix findings.

**Recommendation:** None needed.

## Material outside the folder

Almost everything the share needs is already inside DRDO_Project. The workflow scripts and states match the ~/.claude copies byte for byte, ~/Downloads/21095.pdf has the same SHA-256 as docs/references, and no code, config or symlink points outside the folder. Four groups of outside files produced or support shipped content and exist only in the session scratchpad (cleared on reboot) or in transcripts: (1) the helper that copied the tables into REPORT.md, (2) the round-1 verify_install logs behind the QUICKSTART claims, (3) the round-2 independent code review and report audit cited by REPORT.md, and (4) the owner's final "share the folder" request, which PROJECT_BRIEF.md lacks. Because of this, FILE_INDEX.md §15 calls the scratchpad "throw-away" and says the owner's requests are all in PROJECT_BRIEF, and both statements are wrong.

### [should-fix] Report-assembly helper that wrote the REPORT.md §8 tables and Table R15 exists only in the scratchpad

**Paths:** `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/assemble_tables.py`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/sec8_round2.md`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/sec8_round2_filled.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/REPORT.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/results/tables.md`

**Evidence:** assemble_tables.py: 2,866 B, 2026-10-05 07:25, sha256 d3d2637b…. Its docstring says it will 'replace the auto-generated table blocks in reports/REPORT.md section 8 with the current blocks of reports/results/tables.md'. `--report` rewrites REPORT.md directly. `--draft` put Tables R11, R12a–d and R13, plus R15 (from paired_stats.md), into sec8_round2.md, giving sec8_round2_filled.md; 92 % of that file's non-blank lines appear verbatim in REPORT.md. I ran replace_blocks() in memory on the current REPORT.md: all 18 table blocks of tables.md matched and the output was byte-identical, so this script is the tool that inserted the tables and the only mechanical check that REPORT §8 equals tables.md. REPORT.md L503 says the tables 'are generated automatically from per-file results (scripts/make_tables.py)', but the step that copies them from tables.md into REPORT.md is in no project file. A grep of the project for 'assemble_tables' or 'scratchpad' finds only FILE_INDEX.md L512.

**Recommendation:** Copy it to scripts/assemble_report_tables.py. Replace the hard-coded ROOT with Path(__file__).resolve().parents[1], keep `--report`, add a `--check` mode that only diffs, and mark `--draft` as historical (its input sec8_round2.md is a scratch draft). Add it to FILE_INDEX §4 and to the REPORT.md §10 command sequence after `make_tables.py`. Do this before the final `make_manifest.py` run.

### [should-fix] Round-1 verify_install evidence quoted in QUICKSTART.md (83 PASS in 73 s; Jetson-like re-run in 77 s) exists only in the scratchpad

**Paths:** `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/verify_mac.log`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/verify_report_mac.json`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/verify_review.txt`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/verify_review.json`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/run_verify_jetsonlike.py`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/deploy/jetson/QUICKSTART.md`

**Evidence:** QUICKSTART.md L180-186 (and its copy in dist/jetson_bundle) states that round 1 gave '83 PASS, 0 FAIL, 1 WARN, 7 INFO -> PASS in 73 s… WARN was LL p99.9 = 5.15 ms'. It also says an independent re-run 'with torch and every other package that is not in requirements-jetson.txt blocked' gave the same result in 77 s with 'HQ p99.9 = 5.01 ms'. verify_mac.log (12,120 B, 2026-10-04 09:09) ends 'SUMMARY: 83 PASS, 0 FAIL, 1 WARN, 7 INFO in 73 s' with '[WARN] LL/ORT p99.9 … 5.15 ms'. verify_review.txt (12,152 B, 09:28) ends '… in 77 s' with 'HQ/ORT p99.9 … 5.01 ms'. verify_report_mac.json (29,381 B) and verify_review.json (29,373 B) are the full reports. run_verify_jetsonlike.py (1,193 B) is the import-blocking harness behind the 'every other package … blocked' claim. A grep for '83 PASS' in the project's logs/ and reports/ finds nothing; the project has only the round-2 equivalents (logs/verify_install_round2.out, reports/results/verify_install_mac_round2.json).

**Recommendation:** Copy them in as logs/verify_install_round1.out, reports/results/verify_install_mac_round1.json, logs/verify_install_round1_review_jetsonlike.out, reports/results/verify_install_round1_review_jetsonlike.json, and scripts/verify_install_jetsonlike.py (with the hard-coded project root replaced by a path relative to the script). List them in FILE_INDEX §9/§11 and cite them in QUICKSTART L180-186.

### [should-fix] Round-2 independent code review and round-2 report audit are cited by REPORT.md but not archived; the round-1 audit is

**Paths:** `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/subagents/agent-a14b9aaf90c53a95e.jsonl`, `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/subagents/agent-af0c274573cb2b314.jsonl`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/review/ema_resume.py`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/review/refhealth.py`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/REPORT.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/research/report_audit.json`

**Evidence:** REPORT.md L84 says 'Three bugs were found by testing and an independent code review'. L1151 says 'EMA lost on resume (found by an independent code review before the resume ran)'. That review ran as a standalone subagent ('Review round-2 code changes', 1.10 MB transcript, 2026-10-04 23:14–23:28), not as a workflow, so nothing of it reached docs/workflows/. Its final report exists only in the transcript. It lists the EMA bug at train_dual.py:99-102, the snr_needed edge case at make_tables.py:151, TIR labelled as SNR at build_dual_babble.py:79, and quiet-room flipping in engine.py:124-139 `_ref_health`. The reproducers are in scratchpad/review/: ema_resume.py and refhealth.py. The round-2 report audit ('Audit round-2 report claims', 1.36 MB, 2026-10-05 07:32–07:48) reported confirmed errors that were then fixed in shipped text: bundle '≈ 8 MB' is now '≈ 13 MB' at REPORT L451 and QUICKSTART L90, and '+0.033 STOI' is now '+0.032' at L68. Its findings are also only in the transcript; its working files are scratchpad/audit/r.txt and t.txt. The round-1 audit, by contrast, is archived as docs/research/report_audit.json (FILE_INDEX L339).

**Recommendation:** Export the final report of each agent (the last tool_use 'message' in each .jsonl) to docs/research/round2_code_review.md and docs/research/report_audit_round2.md. Put ema_resume.py and refhealth.py in docs/research/round2_review_repro/. Add rows to FILE_INDEX §10, and optionally point REPORT §10 to them.

### [should-fix] FILE_INDEX.md §15 is inaccurate and incomplete, and PROJECT_BRIEF.md lacks the owner's final request

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/FILE_INDEX.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/PROJECT_BRIEF.md`, `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09.jsonl`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/tasks/`

**Evidence:** (a) L511 justifies leaving out the transcripts with 'The owner's requests are reproduced verbatim in docs/PROJECT_BRIEF.md', and L334 says the brief has 'every follow-up request'. The main transcript (20.7 MB) has the user message of 2026-10-05T04:08:32Z, 'so give me the  full folder i can share with the others', which is the request behind this share; grepping PROJECT_BRIEF for 'full folder' finds nothing. PROJECT_BRIEF L136 and L142 say 'exact time not recorded', but the transcript records 2026-10-04T10:11:56Z and 12:51:00Z. (b) L512 calls the scratchpad 'throw-away helper scripts… Not referenced by any project file'. Literally true, but REPORT §8 tables and the QUICKSTART round-1 verification claims depend on scratch files (findings 1 and 2). It also lists 'a fresh-bundle test' as round-2 work, but scratchpad/bundle_fresh/ dates from 2026-10-04 09:13 (round 1). (c) L511 says 'the workflow scripts and result journals are in docs/workflows/'. In fact docs/workflows/journals/wf_14dd115a-9e9.jsonl is the 5,040 B partial copy while the complete 26,841 B journal is in ~/.claude, and run wf_c549189e-2c8 is missing. (d) Not listed at all: the sibling folder …/6c56b0b6-…/tasks/ (104 background-task outputs, 612 KB; I checked them and their result lines are either mirrored in logs/ or are ad-hoc deltas reproducible from reports/results/paired_stats.json); the privately published web report (claude.ai artifact Kbp9PUhBQQvhNow44Wz8xu, published from reports/site/index.html, mentioned only as 'published privately' at L469); and the Desktop session screenshot (see the Downloads/Desktop note). (e) The sizes have drifted: ~/.claude is 127 MB, not ≈ 120 MB; the scratchpad is 131 MB, of which 8 MB is this review's share_review/. The rows for 21095.pdf (SHA-256 0e52a512… on both copies) and for the uv interpreter are accurate.

**Recommendation:** Add the 2026-10-05 04:08 UTC request verbatim to PROJECT_BRIEF as §3.4 and fill in the 3.2a/3.2b times. After findings 1–3 are copied in, rewrite the §15 scratchpad and ~/.claude rows: say what was copied in and where, correct the bundle_fresh round, mention the tasks/ folder and the private artifact link, and refresh the sizes. Do this before the final MANIFEST regeneration.

### [should-fix] The pre-share review workflow (wf_c549189e-2c8) is not in docs/workflows/

**Paths:** `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/workflows/scripts/pre-share-review-wf_c549189e-2c8.js`, `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/workflows/wf_c549189e-2c8.json`, `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/subagents/workflows/wf_c549189e-2c8/journal.jsonl`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/state/README.md`

**Evidence:** Comparing ~/.claude workflows/scripts and the state files with docs/workflows/ with cmp: 5 scripts and 7 states are SAME, the 2 resumed-run scripts exist only in the project (as documented), and only wf_c549189e-2c8 is missing (script 8,891 B, state 13,934 B, journal 1,856 B and growing). FILE_INDEX L340-342 and state/README.md describe the archive as 'Final state of each multi-agent workflow run in this project'. This run is the review whose findings will drive the last edits before upload.

**Recommendation:** After this run completes, copy the script to docs/workflows/, the final state to docs/workflows/state/ and the journal to docs/workflows/journals/. Update the counts (7 → 8) in FILE_INDEX §10 and state/README.md, then regenerate the MANIFEST last.

### [note] Complete journal of wf_14dd115a-9e9 is available but only the partial copy is shipped

**Paths:** `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/subagents/workflows/wf_14dd115a-9e9/journal.jsonl`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/workflows/journals/wf_14dd115a-9e9.jsonl`

**Evidence:** The ~/.claude journal has 9 events and 26,841 B, including both Review results. The project copy is 5,040 B, taken at 08:54 while the run was still going. state/README.md acknowledges this, and state/wf_14dd115a-9e9.json already holds the final results, so nothing is lost.

**Recommendation:** Optionally add the full journal as docs/workflows/journals/wf_14dd115a-9e9.final.jsonl, leaving the partial file in place because of the no-delete rule.

### [note] Round-2 in-bundle self-check and packager-incident evidence exist only in the scratchpad

**Paths:** `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/verify_bundle.txt`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/tree_before.txt`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/tree_after.txt`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/src_deploy_sums.txt`

**Evidence:** verify_bundle.txt (16,452 B, 2026-10-05 07:31) is verify_install run with root=dist/jetson_bundle: '115 PASS, 0 FAIL, 0 WARN, 10 INFO in 30 s'. It applies to the 07:31 bundle; logs/package_bundle_round2_final.out shows the bundle was rebuilt at 07:57. The shipped docs make no round-2 in-bundle claim. tree_before.txt and tree_after.txt (484 KB each, identical) plus src_deploy_sums.txt (2026-10-04 09:12–09:13) are the evidence for REPORT.md L1160, 'Nothing was deleted', after the packager ran with the project root as its output.

**Recommendation:** Optional. Copy them as logs/verify_install_bundle_round2_0731.out and logs/packager_incident_2026-10-04/{tree_before,tree_after,src_deploy_sums}.txt, or re-run verify_install inside the final bundle and log that run instead.

### [note] Superseded drafts and earlier script versions in the scratchpad: no need to copy

**Paths:** `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/REPORT_before_tables.md`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/REPORT_before_tables2.md`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/sec10_round2.md`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/sec9_10_round2.md`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/build_report_html_round1.py`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/make_tables_before_round2labels.py`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/verify_install_before_twomic.py`, `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/README.jetson.orig.md`

**Evidence:** Share of each draft's non-blank lines found verbatim in REPORT.md: REPORT_before_tables.md 83 %, REPORT_before_tables2.md 94 %, sec8_round2_filled.md 92 %, sec10_round2.md 88 %; the rest are placeholders that were later filled. tables_now.md, review/tables_now.md, r.md, a, b and c are table snapshots or sorted extracts. build_report_html_round1.py, make_tables_before_round2labels.py and verify_install_before_twomic.py are pre-edit backups of scripts/build_report_html.py, scripts/make_tables.py and deploy/jetson/verify_install.py. README.jetson.orig.md is the deploy/jetson/README.md from before wf_14dd115a. REPORT_round1.md and tables_round1.md are already in the project.

**Recommendation:** Do not copy them. The only possible exception: put make_tables_before_round2labels.py in docs/history/ if someone needs to regenerate tables_round1.md byte for byte with the round-1 labels.

### [note] Remaining scratchpad items, tool-result caches and transcripts are disposable or third-party

**Paths:** `/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/`, `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/tool-results/`, `~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/memory/`

**Evidence:** The scratchpad holds about 100 cached third-party papers and standards (PDF/HTML/TXT such as 1472h.pdf at 12.9 MB, t7/ at 22 MB, vb16.pdf, mil1474e.txt), used only to verify citations; the outcome is in docs/research/claims_verification.*. tool-results/ (61 MB) holds 74 webfetch PDFs, 9 large bash outputs and 3 browser screenshots. Disposable tests and probes: venv/ (18 MB, pypdf only), pesq-0.0.4/ and pesq.tgz, asteroid_pkg/, benchtest/, vroot/, bundle_fresh/, tartest/, nonempty/, figtest/, pytest_review_completeness/, exp*.py (noise-tracker probes), monitor_round2.sh (watcher only), smoke_dual_resume6.yaml (a max_steps 6 variant; the two used configs are already in configs/smoke and differ only in a comment), paired_stats_test.json, man.txt, now.txt, manifest_check.txt, audit/r.txt and t.txt, the size-probe ONNX files, and the *.wav test files. The memory notes (3 files, 12 KB) are assistant notes. The data pipeline does not depend on scratch metadata: data/raw/esc50/esc50.csv is already in the project and download.py fetches its sources itself.

**Recommendation:** Do not copy. The cached papers are third-party copyright, as §15 already says. Copy the scratchpad files from findings 1–3 before the next reboot, because /private/tmp is cleared then.

### [note] ~/Downloads and ~/Desktop: one relevant file, already in the project

**Paths:** `/Users/nsivaprasadnandyala/Downloads/21095.pdf`, `/Users/nsivaprasadnandyala/Desktop/Screenshot 2026-10-04 at 2.15.56 pm.png`, `/Users/nsivaprasadnandyala/Downloads/Quality-and-Reliability-in-DRDO.pptx`, `/Users/nsivaprasadnandyala/Downloads/butler2003.pdf`

**Evidence:** 21095.pdf (617,063 B) has SHA-256 0e52a512a8eeb8c3733dbe7f4ebb946a2815f0ceb69e24e376d93f20e80516d9, the same as docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf, so the §15 row is accurate. The Desktop screenshot (1.56 MB, 2026-10-04 14:16; the file name contains a narrow no-break space) shows this Claude session's interim round-2 results and is not needed. Not project files: Quality-and-Reliability-in-DRDO.pptx (2024-11-17, predates the project), and butler2003.pdf, butler2003-2.pdf and retrieve.pdf (2026-10-04, a gender-studies article by Judith Butler). No other DRDO, danc, 21095 or noise file on Desktop or in Downloads relates to the project.

**Recommendation:** Nothing to copy. Optionally list the screenshot in §15 as 'not needed'.

## Licences and redistribution

No licence that SHARE_README §4 states is wrong, so there are no blockers. The labels for LibriSpeech, Mini LibriSpeech and VoiceBank+DEMAND (CC BY 4.0), ESC-50 (CC BY-NC 3.0), NOISEX-92 (unclear) and the drone set (none) all match provenance.json and the corpora's own files. The archive as built (53,677 files, 5,148 MB once .venv, caches and .DS_Store are left out) is the internal-only version, though. About 897 MB (17%) is DroneAudioDataset or derived from it, about 714 MB (14%) is NOISEX-92 or derived from it, and together that is about 1.59 GB (31%), or about 2.07 GB (40%) counting ESC-50. Every trained model was also trained on all three. Separately, SHARE_README understates what is shipped and leaves out several third-party items: all 50 ESC-50 classes are in the drone folder, and it does not list Speech Commands, the DFN3 weights and packages, the PMSQE code, the copyrighted DSJ PDF or the Jetson bundle.

### [should-fix] The archive as configured gives every recipient the unlicensed DroneAudioDataset, NOISEX-92 and everything derived from them; (a) is what the owner must be told

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/SHARE_README.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/manifests/provenance.json`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/drone/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/noisex92/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/testsets/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/audio/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/checkpoints/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/exports/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/src/danc/data/noise_bank.py`

**Evidence:** provenance.json drone_audio_dataset.license: "NO LICENSE FILE - all rights reserved by default. Used for internal R&D only; DO NOT redistribute." Every noisex92/* entry: "Redistribution terms NOT stated - UNCLEAR, flag before sharing." SHARE_README.md:50 says to clear these "before sharing outside the project team", and :56-57 say "Treat as internal" and "do not redistribute". Yet :19-20 says "Nothing else was left out", and the recipients are described as the owner's team and reviewers.

Quantities, in decimal MB from stat sums over the archive file list:
- DRONE, raw: data/raw/drone/ = 23,410 files, 768.0 MB.
- DRONE, test-set inputs: 199 files, 26.0 MB, across 165 scenes per meta.csv noise_sources (defence_v1 106, defence_highsnr_v1 25, dualmic_v1 29, dualmic_val 5).
- DRONE, enhanced outputs: reports/audio = 1,134 files, 103.1 MB.
- DRONE, Jetson bundle: 2 scenes.
- DRONE total: about 24,745 files, 897 MB, 17.4% of the archive.
- NOISEX, raw: data/raw/noisex92/ = 30 files (15 .mat + 15 wav16k), 257.1 MB.
- NOISEX, test-set inputs: 697 files, 86.0 MB.
- NOISEX, enhanced outputs: 3,984 files, 368.6 MB.
- NOISEX, demo: 6 files, 1.15 MB.
- NOISEX, Jetson bundle: 10 files across 6 scenes.
- NOISEX total: about 4,727 files, 714 MB, 13.9%.
- Drone or NOISEX together: about 1,589 MB (30.9%).
- With ESC-50 added: about 2,068 MB (40%).
- Also possibly restricted: defence_edge_v1 inputs (39.2 MB) and its enhanced outputs (28.8 MB). They have no per-file source record (see the edge-set finding).

Models: every training config goes through DefenceMixer, which calls NoiseBank(noise_split, use_synth=...) at mixer.py:147. NoiseBank defaults to use_drone=True (noise_bank.py:94), and no config sets use_drone. checkpoints/v3hq_cont/config.yaml contains only use_synth: true. So all 55 checkpoint files (198.1 MB), the 9 trained ONNX files in exports/, and the 4 ONNX files in dist/jetson_bundle/exports/ (8.0 MB) were trained on NOISEX, ESC-50 and drone noise. The only exception is exports/dancnet_untrained_test.onnx.

**Recommendation:** (a) Tell the owner before uploading:
1. This archive is the internal-only version. Roughly 31% of it (about 1.6 GB) is drone or NOISEX material, raw or derived, and about 40% if ESC-50 is counted. All trained models were trained on that data.
2. Upload it only if every person who will get the link is inside the project team. In Google Drive, set sharing to Restricted with named accounts, not "Anyone with the link". Tell recipients not to forward the archive, reports/audio/demo/ or dist/jetson_bundle/.
3. If any reviewer is outside the team, send the reduced variant in the exclusion-list finding instead.
4. ESC-50 is CC BY-NC 3.0, so no recipient may use the data, the derived audio or the models commercially, including a hardware partner. The project's own position (REPORT R9) is legal review before any product use.
5. Defence material: get export-control and classification clearance before anything goes outside the organisation (SHARING_CHECKLIST.md:214-215).

Add one line to SHARE_README §1 or §4: "This archive is for the project team only; do not forward it."

### [should-fix] data/raw/drone/ holds 682 MB of unused 'unknown' clips that contain all 50 ESC-50 classes plus Google Speech Commands noise; SHARE_README's 'ESC-50 (16 classes)' understates what is shipped

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/drone/DroneAudioDataset-master/Binary_Drone_Audio/unknown/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/drone/DroneAudioDataset-master/Multiclass_Drone_Audio/unknown/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/drone/DroneAudioDataset-master/README.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/src/danc/data/noise_bank.py`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/SHARE_README.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/SHARING_CHECKLIST.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/research/T2_datasets.md`

**Evidence:** The two unknown/ folders hold 10,372 files each, 341.2 MB each, 682.4 MB in total. Their file lists are identical, and a sampled pair (1-100032-A-00.wav) has the same SHA-256 (8494316b...). Per folder:
- 10,001 files are named after ESC-50 clips. Matched against data/raw/esc50/esc50.csv, they come from all 2,000 ESC-50 source clips and all 50 classes.
- 70 are Speech Commands background-noise segments: doing_the_dishes 20, exercise_bike 13, running_tap 13, white_noise 12, pink_noise 12.
- 301 are 'silence' clips.

The drone README.md (line 5) says the 'unknown' clips are taken from ESC-50 and from Speech Commands. noise_bank.py:153-157 loads only Multiclass_Drone_Audio/bebop_1, membo_1 and Binary_Drone_Audio/yes_drone, so the unknown/ folders are never read (20,744 of the 23,410 drone files).

SHARE_README.md:55 says "ESC-50 (16 classes)", and SHARING_CHECKLIST.md:196 and FILE_INDEX.md:261 say the same. Speech Commands (CC BY 4.0, Warden 2018) is not mentioned in SHARE_README, the checklist or provenance.json; only docs/research/T2_datasets.md:222 notes it. The CC BY-NC label is correct, but the scope is not, and these files sit under a row labelled "No licence file".

**Recommendation:** In SHARE_README §4 (and in SHARING_CHECKLIST §4 at the next manifest run):
- State that data/raw/drone/.../unknown/ contains segmented copies of all 50 ESC-50 classes (CC BY-NC 3.0) and Speech Commands background noise (CC BY 4.0, attribution required), and that the code does not use them.
- In any reduced or staging copy, leave out both unknown/ folders. That saves 682 MB with no effect on code or results. Build the copy with rsync --exclude into a new folder; never delete anything from the original.

### [should-fix] SHARE_README §4 leaves out several third-party items (DFN3 weights and packages, PMSQE code, the DSJ PDF, the Jetson bundle, RIR bank provenance) and contradicts itself on models and results

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/SHARE_README.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_models/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/src/danc/train/third_party/LICENSE_asteroid_MIT.txt`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/src/danc/train/third_party/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/rir_bank/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/SHARING_CHECKLIST.md`

**Evidence:** The SHARE_README.md:52-58 table lists only the corpora plus one row, "Synthetic defence noise, project code, models, results | produced by this project | Owner's decision". It is missing:
(i) DeepFilterNet3. third_party/dfn_models/ holds DeepFilterNet3.zip, model_120.ckpt.best and config.ini; LICENSE-MIT ("Copyright (c) 2021 Hendrik Schröter") and LICENSE-APACHE are present. third_party/dfn_pkgs/ holds 1,039 files, 48.0 MB. Three reports/audio/enhanced_*_dfn3 folders are DFN3 outputs.
(ii) Vendored PMSQE: src/danc/train/third_party/pmsqe.py and bark_matrix_16k.mat (MIT, "Copyright (c) 2019 Pariente Manuel"), with a copy in dist/jetson_bundle/src/danc/train/third_party/. This makes the "project code ... produced by this project" row inaccurate.
(iii) docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf (617 KB). Page 1 carries a "2026, DESIDOC" copyright line (DSJ Vol. 76 No. 3, DOI 10.14429/dsj.21095), and no page has any CC or open-licence text.
(iv) dist/jetson_bundle/ (116 files, 13.4 MB). It contains 16 test scenes, 14 of them with restricted noise, and 4 trained ONNX files.
(v) Speech Commands (see the drone unknown-clips finding).
(vi) The RIR bank: data/rir_bank/ (4 .npz files, 28.0 MB). It is generated by pyroomacoustics' image-source method (src/danc/data/rir.py:1-11), so it is project-made and unrestricted, but it is not named. REPORT §3.1 lists it.

SHARE_README.md:58 calls models and results "Owner's decision", but :60 says the enhanced audio and trained models "inherit their restrictions". SHARING_CHECKLIST.md:203-206 already has rows for the bundle, PMSQE, DFN3 and the PDF, so the gap is in SHARE_README only.

**Recommendation:** Add rows to the SHARE_README §4 table:
- DeepFilterNet3 weights and packages: third_party/; permissive licences (see the third_party licence finding).
- PMSQE: src/danc/train/third_party/; MIT (Asteroid); keep LICENSE_asteroid_MIT.txt.
- Reference paper: docs/references/*.pdf; copyright DESIDOC; internal only unless the journal's terms allow passing it on.
- Jetson bundle: inherits the test-set and model caveats.
- RIR bank and synthetic examples: project-generated.
- Speech Commands: CC BY 4.0.

Change line 58 so models and enhanced audio are not listed as "Owner's decision" without the inheritance caveat. SHARE_README.md is not listed in reports/MANIFEST.sha256, so editing it does not break the manifest. SHARING_CHECKLIST.md is listed, so editing it requires re-running scripts/make_manifest.py.

### [should-fix] third_party/ is labelled 'MIT / Apache-2.0', but dfn_pkgs also contains BSD, GPL-3.0-with-GCC-exception and LGPL components, and loguru ships without its MIT notice

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_pkgs/loguru-0.7.3.dist-info/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_pkgs/numpy-1.26.4.dist-info/LICENSE.txt`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_pkgs/numpy/.dylibs/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/dfn_pkgs/packaging-23.2.dist-info/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/SHARING_CHECKLIST.md`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/FILE_INDEX.md`

**Evidence:** SHARING_CHECKLIST.md:32 and :205, FILE_INDEX.md:451-453 and REPORT R9 call third_party "MIT / Apache-2.0". What is actually there:
- numpy-1.26.4 is BSD-3. Its numpy/.dylibs/ bundles libopenblas64_ (BSD-3), libgfortran.5 and libgcc_s.1.1 (GPL-3.0-with-GCC-exception, LICENSE.txt:166-170) and libquadmath (LGPL, LICENSE.txt:950).
- packaging-23.2 is Apache-2.0 OR BSD-2; LICENSE.APACHE and LICENSE.BSD are present.
- appdirs is MIT (LICENSE.txt present).
- deepfilternet-0.5.6 is MIT (LICENSE present).
- DeepFilterLib-0.5.6 is MIT/Apache, with license_files/ present. Its libdf.cpython-311-darwin.so is a macOS arm64 binary.
- loguru-0.7.3.dist-info has no LICENSE file. Its RECORD lists only INSTALLER, METADATA, RECORD, REQUESTED and WHEEL, METADATA carries only the MIT classifier, and no licence text exists under loguru/. MIT requires the notice to accompany copies.

All of these allow redistribution; the only compliance gap is the missing loguru notice.

**Recommendation:** Correct the label to something like: "permissive: MIT / Apache-2.0 / BSD; numpy's bundled GCC runtime libraries are GPL-3.0 with the runtime exception or LGPL, with texts in numpy-1.26.4.dist-info/LICENSE.txt". In the shared (staging) copy, add loguru's MIT LICENSE text to loguru-0.7.3.dist-info/. dfn_pkgs is outside the manifest, so adding the file does not break verification.

### [note] defence_edge_v1 records no noise source per file, so the whole set and its enhanced outputs must be treated as possibly containing drone, NOISEX and ESC-50 audio

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/testsets/defence_edge_v1/meta.csv`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/src/danc/data/build_edgeset.py`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/audio/enhanced_defence_edge_v1_dancnet_hq_metric/`

**Evidence:** meta.csv columns are id, case, snr_db, utt, info, with no noise_sources column (build_edgeset.py:69). Noise comes from DefenceMixer("test", 6.0, use_synth=True) (build_edgeset.py:54), whose NoiseBank loads NOISEX, ESC-50 and drone by default (noise_bank.py:94-102). Only the 30 hum400 files are provably free of restricted audio (synthetic tracked_vehicle and hum, build_edgeset.py:143-151). post_blast and crew_babble use a stationary bed from the bank. Affected: 304 noisy files (39.2 MB) and 304 enhanced files (28.8 MB).

**Recommendation:** In any reduced copy, leave out defence_edge_v1/noisy/ and the enhanced_defence_edge_v1_* folder in full; the clean/ targets are LibriSpeech-only and can stay. Longer term, have build_edgeset write a noise_sources column, as build_testset already does.

### [note] The parts most likely to be forwarded (demo listening set, Jetson bundle) mostly contain restricted noise

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/audio/demo/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/data/testsets/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/exports/`

**Evidence:** reports/audio/demo/ has 16 WAVs, 3.1 MB, from 4 dualmic_v1 scenes:
- 0015_stationary_p00: src:noisex/destroyerengine
- 0210_mixed_p05: non:noisex/destroyerops
- 0060_nonstationary_m05: src:esc50/hand_saw
- 0135_impulsive_p00: synth/explosion only, so clean.
That is 6 NOISEX files (1.15 MB) and 3 ESC-50 files.

The Jetson bundle has 16 scenes, of which 14 contain restricted noise:
- Drone: defence_v1 0200_nonstationary_m05 (drone/membo) and 0400_impulsive_m05 (bed:drone/membo).
- NOISEX: defence_v1 0360, 0760 and dualmic_v1 0000, 0105, 0120, 0225.
- ESC-50: 8 scenes.
- Synthetic only: just defence_v1 0000_stationary_m05 and 0160_stationary_p15.

The bundle also carries 4 trained ONNX files (8.0 MB), including the unused round-1 dancnet_hqft.onnx.

**Recommendation:** Before the bundle or demo goes to a hardware partner or outside reviewer, rebuild it from synthetic-only scenes. defence_v1 has 199, highsnr 32, dualmic_v1 45, dualmic_val 14 and dualmic_babble_v1 27 such scenes per meta.csv noise_sources. Get the model weights through legal review first (REPORT R9).

### [note] Most CC-licensed corpus folders ship without a licence or attribution file

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/esc50/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/vbdemand_test/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/noisex92/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/LibriSpeech/LICENSE.TXT`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/drone/LICENSE_NOTICE.txt`

**Evidence:** Only data/raw/LibriSpeech/LICENSE.TXT (CC BY 4.0, "(c) 2014 by Vassil Panayotov") and data/raw/drone/LICENSE_NOTICE.txt exist.
- data/raw/esc50/ contains only audio16k/ and esc50.csv. It is CC BY-NC 3.0 and needs attribution plus a licence link.
- data/raw/vbdemand_test/ contains only clean/ and noisy/ (1,648 WAVs). It is CC BY 4.0, Valentini-Botinhao et al.
- data/raw/noisex92/ contains only mat/ and wav16k/, with no TNO notice.

provenance.json and the README.md:106-117 table carry the licence strings, which partly covers attribution. Recipients who use only data/ or the test sets will not see them.

**Recommendation:** In the shared copy, add a single data/ATTRIBUTION.md (or a per-folder NOTICE). It should name the creator, source URL and licence link for LibriSpeech, Mini LibriSpeech, VoiceBank+DEMAND, ESC-50 and Speech Commands, plus the TNO notice for NOISEX and the Al-Emadi citation from the drone README. Note in it that the test sets are derived from these sources.

### [note] Inventory of archive groups that contain or derive from third-party material (sizes, counts, licences)

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/testsets/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/manifests/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/rir_bank/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/synthetic_examples/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/audio/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/checkpoints/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/exports/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/third_party/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/references/`

**Evidence:** Sizes are decimal MB from stat sums; the archive total is 53,677 files, 5,148 MB.

RAW CORPORA
- data/raw/LibriSpeech: 11,678 files, 2,157.6 MB. CC BY 4.0; LICENSE.TXT present.
- data/raw/vbdemand_test: 1,648 files, 132.7 MB. CC BY 4.0.
- data/raw/esc50: 641 files, 102.5 MB, 16 classes. CC BY-NC 3.0.
- data/raw/noisex92: 30 files, 257.1 MB. Unclear terms.
- data/raw/drone: 23,410 files, 768.0 MB. No licence; 682.4 MB of it is the unused ESC-50 and Speech Commands 'unknown' clips.

TEST SETS (data/testsets: 3,781 files, 412.1 MB). Scene counts by noise family, from meta.csv noise_sources:
- defence_v1: 900 scenes. NOISEX 369, ESC 356, drone 106, synthetic 371; 701 restricted, 199 synthetic-only.
- defence_highsnr_v1: 160 scenes. NOISEX 60, ESC 68, drone 25; 128 restricted.
- dualmic_v1: 240 scenes. NOISEX 103, ESC 100, drone 29; 195 restricted.
- dualmic_val: 48 scenes. NOISEX 16, ESC 19, drone 5; 34 restricted.
- dualmic_babble_v1: 60 scenes. NOISEX 15, ESC 18, drone 0; 33 restricted.
- defence_edge_v1: 304 scenes, sources unknown.
- Totals: 1,353 noisy-side files (168.0 MB) contain restricted noise. All clean/ folders are LibriSpeech-only (CC BY 4.0).

METADATA AND GENERATED DATA
- data/manifests: 5 files, 1.7 MB. Paths, durations, hashes and licence strings only; shareable.
- No on-disk noise cache exists. The float16 noise bank is RAM-only (noise_bank.py:60-62).
- data/rir_bank: 4 .npz files, 28.0 MB. Project-generated with pyroomacoustics.
- data/synthetic_examples: 13 WAVs, 2.5 MB. Project-generated; synth_noise.py reads no recordings.

ENHANCED AUDIO (reports/audio: 10,844 files, 958.9 MB)
- 22 enhanced_* folders built from defence and dualmic sets contain restricted noise: 7,590 files (700.4 MB) plus 304 edge files (28.8 MB).
- enhanced_vbdemand_dfn3: 824 files, 29.4 MB. CC BY 4.0-derived, clean.
- demo: 16 files, 3.1 MB.

MODELS
- checkpoints: 55 files, 198.1 MB.
- exports: 23 files, 17.5 MB, with 10 ONNX files (9 trained).
- All trained models used LibriSpeech, NOISEX, ESC-50, drone and synthetic data; VoiceBank+DEMAND was never used for training.

OTHER
- third_party: 1,044 files, 64.7 MB; permissive licences.
- dist/jetson_bundle: 116 files, 13.4 MB, of which data/ is 42 files, 4.8 MB.
- docs/references: 1 PDF, 0.6 MB, copyright DESIDOC.
- reports/figures/*example_spectrograms.png are images of test mixtures only; low risk.

**Recommendation:** Use this table to fill out SHARE_README §4 and as the basis for the exclusion list in the next finding.

### [note] (b) Exact paths to leave out for a publicly redistributable variant (build into a new folder with rsync --exclude; never delete from the original)

**Paths:** `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/drone/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/noisex92/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/raw/esc50/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/data/testsets/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/reports/audio/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/dist/jetson_bundle/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/checkpoints/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/exports/`, `/Users/nsivaprasadnandyala/Desktop/DRDO_Project/docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf`

**Evidence:** Paths below are relative to DRDO_Project/. Leaving out items 1-8 removes 35,606 files (2,434 MB); about 18,071 files (2,714 MB) remain, mostly LibriSpeech.

MUST LEAVE OUT: drone and NOISEX material and their derivatives
1. data/raw/drone/
2. data/raw/noisex92/
3. Test-set noisy sides, i.e. every noisy/, primary/ and reference/ folder (254.3 MB):
   - data/testsets/defence_v1/noisy/
   - data/testsets/defence_highsnr_v1/noisy/
   - data/testsets/defence_edge_v1/noisy/
   - data/testsets/dualmic_v1/primary/ and reference/
   - data/testsets/dualmic_val/primary/ and reference/
   - data/testsets/dualmic_babble_v1/primary/ and reference/
   Keep the clean/ folders and meta.csv.
4. Every reports/audio/enhanced_* folder except enhanced_vbdemand_dfn3/. That is 22 folders:
   - 8 enhanced_defence_v1_* folders
   - enhanced_defence_highsnr_v1_dfn3
   - enhanced_defence_edge_v1_dancnet_hq_metric
   - 9 enhanced_dualmic_v1_* folders
   - 3 enhanced_dualmic_babble_v1_* folders
5. Demo files:
   - reports/audio/demo/0015_stationary_p00_{1_primary_noisy,2_hybrid_HQ_40ms,2_hybrid_LL_20ms}.wav (NOISEX)
   - reports/audio/demo/0210_mixed_p05_{1_primary_noisy,2_hybrid_HQ_40ms,2_hybrid_LL_20ms}.wav (NOISEX)
   - reports/audio/demo/0060_nonstationary_m05_{1_primary_noisy,2_hybrid_HQ_40ms,2_hybrid_LL_20ms}.wav (ESC-50)
6. Jetson bundle test inputs:
   - dist/jetson_bundle/data/testsets/defence_v1/noisy/
   - dist/jetson_bundle/data/testsets/dualmic_v1/primary/ and reference/
7. Model weights, unless legal review clears weights trained on drone and NOISEX data:
   - checkpoints/ (all 55 files)
   - exports/dancnet_{hq,hqft,ll,pilot,v3_2mic,v3hq,v3hq_cont,v3ll}.onnx and exports/smoke_ema.onnx (keep dancnet_untrained_test.onnx and the .json files)
   - dist/jetson_bundle/exports/*.onnx (4 files)
8. docs/references/Narain_Kant_Singh_DSJ_2026_21095.pdf, unless DSJ's terms allow it. Cite DOI 10.14429/dsj.21095 instead.

ALSO LEAVE OUT for a variant that must be free of non-commercial terms
9. data/raw/esc50/ (641 files, 102.5 MB). Mixtures derived from ESC-50 are already removed by items 3-6. If non-commercial use is acceptable, ESC-50 can stay instead, with an attribution file and the variant labelled CC BY-NC.

KEEP
- data/raw/LibriSpeech/
- data/raw/vbdemand_test/
- reports/audio/enhanced_vbdemand_dfn3/
- data/manifests/, data/rir_bank/, data/synthetic_examples/
- third_party/ (add the loguru licence)
- All code, configs and docs

**Recommendation:** If the owner wants the public variant, give recipients this list and warn them of the side effects:
- reports/MANIFEST.sha256 will report FAILED for every excluded path. Generate a separate manifest inside the reduced copy.
- verify_install.py and re-evaluation cannot run without the models and test inputs. Recipients would rebuild the test sets with danc.data.build_testset after downloading the corpora themselves.

A finer-grained option is to keep only the synthetic-only scenes, selected per file from meta.csv noise_sources (counts in the inventory finding). That is not possible for defence_edge_v1.

## Fresh recipient walkthrough

From a recipient's side, the archive verifies and sets up as SHARE_README.md describes, so there are no blockers. Every path it names exists. reports/MANIFEST.sha256 has exactly 52 629 well-formed lines that match the archive file set (53 677 files; the 1 048 unlisted ones are third_party/dfn_pkgs, 7 .lock files, SHARE_README.md and the manifest itself), and 1 593 sampled or recently changed files hash correctly. The other quoted figures also match their sources: 103 passed (logs/pytest_round2_final.out), 115 PASS / 0 FAIL (verify_install_mac_round2.json), the sizes, the deployed set and its hashes (DEPLOYMENT.json, the bundle's SHA256SUMS 115/115), the README headline numbers recomputed from the per_file.csv files, and all 10 images in reports/site. The problems are documents that contradict each other or are out of date. The most visible: FILE_INDEX and SHARING_CHECKLIST still say the .DS_Store files are in the manifest, that the manifest must be regenerated, and that the HTML report is still the round-1 build, and the README labels round-1 models 'deployed HQ'. SHARE_README also leaves out some setup prerequisites (a C compiler for pesq), how to listen to the deployed models' audio, and how the zip is checked.

### [should-fix] Docs say .DS_Store files are in the manifest and must be shipped; they are not in the manifest and the archive leaves them out

**Paths:** `docs/SHARING_CHECKLIST.md`, `FILE_INDEX.md`, `SHARE_README.md`, `scripts/make_manifest.py`, `reports/MANIFEST.sha256`

**Evidence:** docs/SHARING_CHECKLIST.md:89-90 says 'Do not exclude the four .DS_Store files: they are listed in reports/MANIFEST.sha256'. Lines 125 and 252 tell recipients to expect possible .DS_Store FAILED lines. FILE_INDEX.md:58 (§1 row) says 'they are listed in reports/MANIFEST.sha256 (§14)', and FILE_INDEX.md:495 says 'The four .DS_Store files are in the manifest.' In fact `grep -c DS_Store reports/MANIFEST.sha256` = 0, and scripts/make_manifest.py:51 skips them on purpose (`rel.endswith(".DS_Store")`). SHARE_README.md:19-20 says the archive excludes them, which matches the manifest. So the checklist's packaging advice and FILE_INDEX contradict both SHARE_README and the actual manifest.

**Recommendation:** Update SHARING_CHECKLIST §2 (lines 89-90), §3.1 (line 125), §5 (line 252) and FILE_INDEX §1 (line 58) and §14 (lines 495-497) to say that .DS_Store files are not in the manifest and are not shipped. Remove 'and possibly .DS_Store files' from the recipient checks.

### [should-fix] FILE_INDEX §13/§14 still say the manifest is outdated (52 589 files) and must be regenerated

**Paths:** `FILE_INDEX.md`, `reports/MANIFEST.sha256`

**Evidence:** FILE_INDEX.md:471 lists 'regenerate reports/MANIFEST.sha256 last' as still to do. FILE_INDEX.md:481-484 says 'The copy present at this re-check dates from 07:27 ... lists 52 589 files ... so it must be regenerated'. In reality the manifest is dated Oct 5 07:59 (after FILE_INDEX.md at 07:58), has 52 629 lines, and matches the tree. The only file newer than it is SHARE_README.md (08:10). A recipient reading FILE_INDEX would conclude the integrity manifest cannot be trusted.

**Recommendation:** Rewrite FILE_INDEX §13 and §14 to describe the final state: manifest regenerated at 2026-10-05 07:59 with 52 629 files, nothing left to do. Then either regenerate the manifest last, or accept that the FILE_INDEX edit changes its hash (see the manifest-coverage note).

### [should-fix] HTML report is called the round-1 build or 'rebuild first', but it is already the round-2 build

**Paths:** `FILE_INDEX.md`, `docs/SHARING_CHECKLIST.md`, `reports/site/index.html`, `reports/site/figures/`

**Evidence:** FILE_INDEX.md:287 says 'site/figures/ (8 PNG) ... At this re-check it is still the round-1 build (2026-10-04 08:27; it contains no round-2 result); rebuild ... before hand-over'. FILE_INDEX.md:470 (§13) says it was rebuilt, so FILE_INDEX contradicts itself, and its totals line ('site/ 4.5 MB (9 files)') is stale too. docs/SHARING_CHECKLIST.md:25 says '(rebuild it first, §5)', and lines 226-227 say 'at the 07:40 re-check it was still the round-1 build'. Actual state: index.html is dated Oct 5 07:57, mentions v3hq_cont 24 times and TWO_MIC, and site/figures holds 12 PNG files (4.9 MB). The checklist's 'Before sending' boxes in §5 (lines 222-247) are all unchecked, and some carry stale notes, e.g. line 228-229 says the journal is still partial, while FILE_INDEX §13 says the final state is in docs/workflows/state/wf_14dd115a-9e9.json.

**Recommendation:** Update FILE_INDEX §9 (line 287, the totals line), SHARING_CHECKLIST line 25 and lines 226-229 to give the final state. Tick or annotate the §5 'Before sending' boxes so a recipient knows what the sender completed.

### [should-fix] README 'Reproduce' block labels round-1 models as 'deployed HQ'

**Paths:** `README.md`, `exports/deployment_selection.json`

**Evidence:** README.md:92 says `finetune_metric --config configs/finetune_hq_metric.yaml   # deployed HQ = this fine-tune`, and README.md:95 says `export_onnx --ckpt checkpoints/hq_metric/last.pt --out exports/dancnet_hqft.onnx   # deployed HQ`. README.md:16, exports/deployment_selection.json, FILE_INDEX.md:28 and SHARING_CHECKLIST.md:61 all say the deployed HQ is exports/dancnet_v3hq_cont.onnx (checkpoints/v3hq_cont/best.pt). Only README.md:104, after the block, explains that these are round-1 commands.

**Recommendation:** Change the inline comments to '# round-1 deployed HQ (superseded)'. Alternatively add the round-2 HQ commands (configs/train_v3_hq.yaml, then train_v3_hq_cont.yaml, then export to exports/dancnet_v3hq_cont.onnx), or put a heading 'Round-1 sequence' above the block.

### [should-fix] Setup section omits build prerequisites; pins are macOS arm64 freezes; Python range is inconsistent

**Paths:** `SHARE_README.md`, `requirements.txt`, `pyproject.toml`, `docs/SHARING_CHECKLIST.md`, `deploy/jetson/requirements-jetson.txt`

**Evidence:** pesq==0.0.4 (requirements.txt:35) ships as source only: the dev venv's pesq-0.0.4.dist-info has uv_build.json and WHEEL 'Generator: setuptools (84.0.0)', meaning it was compiled locally, and deploy/jetson/requirements-jetson.txt says it is an sdist only and 'needs build-essential + python3-dev'. A recipient therefore needs a C compiler (Xcode Command Line Tools on macOS, build-essential on Linux). SHARE_README §3 and SHARING_CHECKLIST §3.2 do not say so. libportaudio2 is mentioned only in the checklist (line 153). sounddevice is imported lazily, so tests and verify_install still pass without it (verify_install only reports a WARN). Every compiled wheel in the frozen venv has a macosx tag. No package name in requirements.txt is macOS-only (no pyobjc or appnope), so the pins should resolve on Linux x86_64 and arm64 (not tested against PyPI here). On Linux x86_64, torch==2.14.1 from PyPI pulls several GB of unpinned nvidia-* CUDA wheels. pyproject.toml:5 says requires-python '>=3.10', but the pinned numpy 2.4.6, scipy 1.17.1, pandas 3.0.6 and torch 2.14.1 all declare Requires-Python >=3.11, which matches SHARE_README's 'Python 3.11 is required'. pyproject.toml has no [build-system] table; `pip install -e .` and `uv pip install -e .` still work through the setuptools fallback (the dev venv has __editable__.danc-0.1.0.pth).

**Recommendation:** Add a prerequisites line to SHARE_README §3: a C compiler (`xcode-select --install` / `sudo apt install build-essential`) for pesq; libportaudio2 only for live audio; on a CPU-only Linux box optionally `--index-url https://download.pytorch.org/whl/cpu` for torch and torchaudio. Consider setting requires-python = ">=3.11" for the dev install (the Jetson runtime path uses requirements-jetson.txt with Python 3.10), and adding a [build-system] table (setuptools>=64).

### [should-fix] No guidance on listening to the deployed models or finding per-test-set results; the demo set is round-1 only

**Paths:** `SHARE_README.md`, `reports/audio/demo/`, `reports/audio/`, `data/testsets/`, `reports/results/`

**Evidence:** SHARE_README mentions 'enhanced audio' (line 14) but never says how to listen. reports/audio/demo/ (16 WAV files, e.g. 0015_stationary_p00_2_hybrid_HQ_40ms.wav) was 'made with the round-1 models' (FILE_INDEX.md §9). Outputs of the deployed models sit in reports/audio/enhanced_defence_v1_dancnet_v3hq_cont/, enhanced_dualmic_v1_hybrid2_v3_2mic/ and enhanced_dualmic_babble_v1_hybrid2_v3_2mic/. Their files must be matched by id (e.g. 0000_stationary_m05.flac) to data/testsets/<set>/{noisy|primary,clean}/. The edge-case set has enhanced audio only for the round-1 HQ (enhanced_defence_edge_v1_dancnet_hq_metric), and the high-SNR set only for DFN3. Per-set numbers are in reports/results/<set>/summary.json and per_file.csv and reports/results/tables.md, but SHARE_README only says '§8 holds all results'.

**Recommendation:** Add a short 'Listen / look up results' subsection to SHARE_README §5. It should name the 2-3 enhanced folders for the deployed models, the matching noisy and clean folders in data/testsets, and the fact that the files are FLAC with the same file ids. Say that demo/ is round-1. Point to reports/results/tables.md and reports/results/<set>/summary.json for per-set results.

### [should-fix] Transfer and integrity steps assume a tar.gz with a hash file, but the share is a zip on Google Drive

**Paths:** `docs/SHARING_CHECKLIST.md`, `SHARE_README.md`, `reports/MANIFEST.sha256`

**Evidence:** SHARING_CHECKLIST §3.1 (lines 113-118) tells the recipient to run `shasum -a 256 -c DRDO_Project_share.tar.gz.sha256` and `tar -xzf DRDO_Project_share.tar.gz`, and §2 packages with tar. No document mentions a zip or Google Drive. SHARE_README gives no archive hash and no out-of-band anchor for the manifest, so the manifest only proves the files agree with a manifest shipped in the same archive. The manifest's own SHA-256 is e48653028f8463d9eef1cb4611b9867dc912b22979e7b77fad3088c43768ef33. No Windows check command is given, and the bundle check on line 118 uses shasum only. Packaging risks: every file carries the com.apple.provenance xattr, so a Finder 'Compress' zip may add __MACOSX/._* entries; very large (>4 GB) Finder-made zips are known to cause trouble for Linux `unzip`.

**Recommendation:** Build the zip with Info-ZIP, e.g. `cd ~/Desktop && zip -r -X -q DRDO_Project_share.zip DRDO_Project -x 'DRDO_Project/.venv/*' '*/__pycache__/*' 'DRDO_Project/.pytest_cache/*' '*.DS_Store'`, then test it with `unzip -tq`. Send the zip's SHA-256 (or at least the manifest's hash above) separately from the Drive link. Update SHARING_CHECKLIST §3.1 to the zip flow and add `sha256sum` for the bundle check on Linux.

### [should-fix] The archive itself redistributes data that SHARE_README says must not be redistributed

**Paths:** `SHARE_README.md`, `data/raw/drone/`, `data/raw/noisex92/`, `data/raw/esc50/`, `docs/SHARING_CHECKLIST.md`, `docs/PROJECT_BRIEF.md`

**Evidence:** SHARE_README §4 (lines 50-58) says to clear the licences 'before sharing outside the project team' and marks DroneAudioDataset 'Internal R&D only; do not redistribute'. Yet the archive includes data/raw/drone (803 MB), noisex92 (245 MB, terms unclear) and esc50 (100 MB, non-commercial), plus test sets, audio and models derived from them. The intended recipients include 'reviewers'. The SHARING_CHECKLIST §5 licence and organisation-clearance boxes (lines 240-243) are unchecked. SHARE_README does not mention the other sensitivity points listed in the checklist (lines 209-215): the developer's user paths in docs/PROJECT_BRIEF.md, BUNDLE_INFO.txt, verify_install_mac_round2.json and 37 files under logs/, plus export control.

**Recommendation:** Before uploading, the owner should confirm that every reviewer is inside the project team, or build a reduced copy without data/raw/drone and noisex92 (and the derived sets if required). Add one line to SHARE_README §4 stating who the copy is cleared for.

### [note] Manifest coverage statement is incomplete, and editable install changes a manifest-listed file

**Paths:** `SHARE_README.md`, `reports/MANIFEST.sha256`, `src/danc.egg-info/SOURCES.txt`, `src/x`

**Evidence:** SHARE_README.md:31-32 says the manifest 'covers everything except the items above and third_party/dfn_pkgs/'. It also leaves out SHARE_README.md itself (written 08:10, after the manifest at 07:59), the 7 reports/results/*/.per_file.lock files and the manifest. The 1 039 files of third_party/dfn_pkgs cannot be verified at all. src/danc.egg-info/SOURCES.txt (2026-10-03, lists only 13 files) is in the manifest; `pip install -e .` regenerates it, possibly adding the empty src/x as top-level 'x'. A manifest re-check after setup would then show FAILED lines that the checklist (line 126) treats as damage.

**Recommendation:** Make the coverage sentence exact. Note that the integrity check must be run before §3, or exclude src/*.egg-info from the manifest and the archive. Consider adding third_party/dfn_pkgs to the manifest.

### [note] Small size, count and entry-point inconsistencies between the guides

**Paths:** `SHARE_README.md`, `FILE_INDEX.md`, `docs/SHARING_CHECKLIST.md`, `README.md`

**Evidence:** Sizes: SHARE_README.md:17 says third_party is '≈ 0.1 GB'; du gives 64M, as do FILE_INDEX §1 and SHARING_CHECKLIST.md:32. FILE_INDEX §1 counts have drifted: docs 25 files / 1.0 MB, actual 35 / 1.3 MB; scripts 23, actual 24; logs 113, actual 131 including logs/simulated_live_round2; reports 10 939, actual 10 944. Entry point: no document mentions SHARE_README.md, and FILE_INDEX §1, which claims to list every top-level file, omits it. Reading orders differ: SHARE_README §5 starts with REPORT.md, FILE_INDEX header line 23 and SHARING_CHECKLIST §3.4 start with README.md. SHARE_README §3 starts with `cd DRDO_Project` although §2 already runs from inside that folder. SHARE_README §5.3 says every change was decided under a rule 'written before its test result', while REPORT §0 discloses that the cascade idea came from a test result.

**Recommendation:** Change the third_party size to '≈ 64 MB'. Add SHARE_README.md to FILE_INDEX §1 and refresh its counts. Use one reading order everywhere. Drop the extra `cd`. Soften §5.3 to match REPORT §0.

### [note] Stale details in the Jetson QUICKSTART and bundle instructions

**Paths:** `deploy/jetson/QUICKSTART.md`, `deploy/jetson/package_for_jetson.sh`, `dist/jetson_bundle/`

**Evidence:** QUICKSTART.md:91 says the bundle has 'both deployed models'. There are three (HQ, LL, TWO_MIC per dist/jetson_bundle/BUNDLE_INFO.txt), and the bundle also carries the unused round-1 dancnet_hqft.onnx. Line 238 is headed '## 8. Run the engine (HQ or LL)' but the step also covers the two-mic cascade. Step 3 Option A (lines 95-96) tells the user to rebuild with `package_for_jetson.sh --tar`. A recipient already has dist/jetson_bundle/ (verified: SHA256SUMS 115/115 OK, identical to the project files except the reference-subset per_file.csv files). Rebuilding rewrites BUNDLE_INFO.txt and writes dist/jetson_bundle.tar.gz inside the project. Step 5 (line 179) quotes p99.9 of 0.78/0.60/1.46 ms, while README.md:16-18 quotes 0.75/0.60/1.41 ms; both are correct but come from different runs (verify_install vs latency_mac_m4_round2_final.json), and neither is labelled. If the zip is extracted on Windows, executable bits are lost and `./deploy/jetson/install.sh` fails.

**Recommendation:** Change the text to 'the three deployed models'. Rename step 8. In step 3, add 'if you received the shared archive: `tar -czf /tmp/jetson_bundle.tar.gz -C dist jetson_bundle`' instead of rebuilding. Label the two latency sources. Suggest `bash deploy/jetson/install.sh` as an alternative.

### [note] HTML report works offline but loads Google Fonts and ships two unused stale figures

**Paths:** `reports/site/index.html`, `reports/site/figures/defence_v1_pass_rates.png`, `reports/site/figures/dualmic_v1_pass_rates.png`

**Evidence:** All 10 <img src> values (figures/*.png) exist under reports/site/figures/. The page has no scripts and no broken in-page anchors (26 checked). Its only external resources are the Google Fonts stylesheet plus preconnects (fonts.googleapis.com, fonts.gstatic.com), so offline it falls back to system fonts; online, opening it sends a request to Google. defence_v1_pass_rates.png and dualmic_v1_pass_rates.png (dated 2026-10-04 08:27) are not referenced and differ from the current reports/figures versions.

**Recommendation:** Optionally embed the fonts or switch to a system font stack, which suits a defence audience reading offline. Drop the two orphan PNGs from the next site build, or regenerate them.
