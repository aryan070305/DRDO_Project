export const meta = {
  name: 'pre-share-review',
  description: 'Read-only review of the D-ANC project folder before uploading it to Google Drive',
  phases: [
    { title: 'Review', detail: 'four independent read-only reviewers in parallel' },
  ],
}

const ROOT = '/Users/nsivaprasadnandyala/Desktop/DRDO_Project'
const SCRATCH = '/private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/share_review'
const COMMON = `
Project folder: ${ROOT} (a speech-enhancement R&D project "D-ANC"; ~4.9 GB without its venv).
It is about to be zipped and uploaded to Google Drive to share with other people (the owner's team / reviewers).
The archive will contain EVERYTHING in the folder EXCEPT: .venv/, __pycache__/ dirs, .pytest_cache/, .DS_Store files.
(third_party/dfn_pkgs/ IS included.) The recipient guide is ${ROOT}/SHARE_README.md.
STRICT RULES: read-only review. Do NOT create, modify, move or delete any file inside the project folder or anywhere else
except scratch files under ${SCRATCH}/ (create it if needed). Do not run training, evaluation, the test suite, or
anything long-running; quick read-only shell commands (find, grep, ls, du, head, file, shasum on single files) are fine.
Python is available via: source ${ROOT}/.venv/bin/activate (do NOT pip install anything).
Be concrete: every finding must name the exact file path(s) (and line where relevant) and give evidence.`

const FINDINGS = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'two or three sentences' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'should-fix', 'note'] },
          title: { type: 'string' },
          paths: { type: 'array', items: { type: 'string' } },
          evidence: { type: 'string' },
          recommendation: { type: 'string' },
        },
        required: ['severity', 'title', 'paths', 'evidence', 'recommendation'],
      },
    },
  },
  required: ['summary', 'findings'],
}

const SECRETS = COMMON + `

YOUR TASK: secrets and sensitive-information scan of everything that will go into the archive.
Search (excluding .venv, __pycache__, .pytest_cache, binary audio/model files) for:
- credentials: API keys/tokens (e.g. sk-..., AKIA..., ghp_..., hf_..., "api_key", "token", "secret", "password", "Authorization:", "Bearer "),
  private keys ("BEGIN ... PRIVATE KEY"), .env / .netrc / credential files, cookies, OAuth material;
- personal data: email addresses, phone numbers, real names other than cited paper authors, the local username/home path
  (/Users/<name>/...) - count how many files contain absolute home paths and in which folders (logs, json, md, configs, checkpoints' embedded configs);
- anything from the AI-assistant session that should not be shared (e.g. transcripts, internal tool IDs) if present inside the folder.
Also check text files that might embed paths: *.json, *.jsonl, *.yaml, *.csv headers, ONNX metadata (python: onnx is not installed; use
onnxruntime InferenceSession(path).get_modelmeta().custom_metadata_map), and torch checkpoints' train_cfg (python:
torch.load(path, map_location='cpu')['train_cfg'] for 2-3 samples).
Classify: blocker = a real secret/credential or sensitive personal data; should-fix = something the owner would likely not want shared;
note = harmless (e.g. absolute paths with the username - report counts, not every line).`

const LICENCE = COMMON + `

YOUR TASK: licence / redistribution inventory of the archive contents.
Read ${ROOT}/data/manifests/provenance.json, ${ROOT}/docs/SHARING_CHECKLIST.md section 4, ${ROOT}/reports/REPORT.md section 3.1, ${ROOT}/SHARE_README.md section 4.
Then inventory, with sizes (du) and file counts, every folder/file group in the archive that CONTAINS or is DERIVED FROM
third-party data or models, and state for each which licences apply:
- data/raw/* (each corpus folder), data/testsets/* (which noise sources each set's meta.csv says it uses - check noise_sources columns for NOISEX/ESC-50/drone),
  data/manifests, any noise caches (e.g. float16 noise bank files), reports/audio/* (enhanced outputs of test mixtures), reports/audio/demo,
  checkpoints/ and exports/ (trained on what), third_party/ (DeepFilterNet3 model + packages: check licence files present), dist/jetson_bundle (what data it carries),
  data/synthetic_examples, docs/references (third-party PDFs - copyright?).
Specifically quantify how much of the archive involves the DroneAudioDataset (no licence: internal only) and NOISEX-92 (terms unclear).
Check SHARE_README.md section 4 for accuracy and completeness (e.g. missing sources such as RIRs, DFN3 weights licence, copyrighted PDFs in docs/references).
Recommend: (a) what the owner must be told before uploading, (b) the exact list of paths that would have to be left out for a
"publicly redistributable" variant. Severity: blocker only if SHARE_README is factually wrong about a licence; should-fix for missing caveats.`

const RECIPIENT = COMMON + `

YOUR TASK: act as a fresh recipient who just downloaded and unzipped the archive (you have never seen this project).
Read ${ROOT}/SHARE_README.md first, then follow its pointers: README.md, docs/SHARING_CHECKLIST.md section 3, FILE_INDEX.md (header and section 1),
reports/REPORT.md section 0 (summary), deploy/jetson/QUICKSTART.md (steps 1-5 and 8), exports/deployment_selection.json.
Check, WITHOUT running anything long:
- every path and file mentioned in SHARE_README.md exists (relative to the project root) and numbers quoted there match the
  sources (e.g. 52 629 in reports/MANIFEST.sha256 - count its lines; 103 tests - grep logs/pytest_round2_final.out; 115 PASS -
  reports/results/verify_install_mac_round2.json; sizes via du);
- the commands are correct for macOS AND Linux (shasum vs sha256sum; uv vs pip; python version; 'pip install -e .' works with pyproject.toml - read it);
  check requirements.txt pins are installable on Linux x86_64/arm64 as far as you can tell from the package names (flag macOS-only packages);
- README / SHARE_README / SHARING_CHECKLIST / FILE_INDEX do not contradict each other (sizes, counts, what is excluded, deployed models,
  which file is the entry point);
- reports/site/index.html opens offline: every img src it references exists under reports/site/; note any external resources it loads (fonts).
- anything a recipient would need but cannot find (e.g. how to listen to audio examples, where results per test set are, which model is deployed).
Severity: blocker = a recipient cannot verify or set up; should-fix = confusing/contradictory; note = polish.`

const OUTSIDE = COMMON + `

YOUR TASK: completeness critic - is there project-related material that lives OUTSIDE ${ROOT} that should be inside the shared folder
for the share to be complete ("it must contain every file related to this project")?
Look at (read-only):
- /private/tmp/claude-502/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/6c56b0b6-9608-475d-9359-c7e232336d09/scratchpad/ - scripts and drafts used
  to PRODUCE deliverables (e.g. report-assembly helpers like assemble_tables.py, monitor scripts, drafts sec8_round2*.md, smoke configs,
  cached third-party papers/standards used to verify citations, test outputs). Identify which ones were used to generate content that is now
  in the project (the report, tables, results) and therefore matter for reproducibility/audit, vs. disposable scratch.
- ~/.claude/projects/-Users-nsivaprasadnandyala-Desktop-DRDO-Project/ (session transcripts, subagent transcripts, workflow state, memory) - size;
  note: docs/workflows/ in the project already holds workflow scripts, journals and state files; check what is missing vs. that.
- ~/Downloads and ~/Desktop for other files obviously belonging to this project (names containing DRDO, danc, 21095, noise) - list only.
- ${ROOT}/FILE_INDEX.md section 15 ("Related material outside this folder") - is it accurate and complete?
For each item: path, size, what it is, whether it should be copied into the project (where), and why. Do not copy anything yourself.
Severity: should-fix = material that generated shipped content and is needed to reproduce/audit it; note = optional.`

const REVIEWERS = [
  { key: 'secrets', prompt: SECRETS },
  { key: 'licence', prompt: LICENCE },
  { key: 'recipient', prompt: RECIPIENT },
  { key: 'outside', prompt: OUTSIDE },
]

phase('Review')
const results = await parallel(REVIEWERS.map(r => () =>
  agent(r.prompt, { label: 'review:' + r.key, phase: 'Review', schema: FINDINGS })
    .then(res => res ? { reviewer: r.key, ...res } : null)))

const ok = results.filter(Boolean)
const missing = REVIEWERS.map(r => r.key).filter(k => !ok.find(o => o.reviewer === k))
if (missing.length) log('reviewers without a result: ' + missing.join(', '))
return ok
