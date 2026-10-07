export const meta = {
  name: 'audit-report-consistency',
  description: 'Read-only audit: every number/claim in reports/REPORT.md vs raw results, logs, code and configs',
  phases: [{ title: 'Audit', detail: 'numbers-vs-data and claims-vs-code auditors' }],
}
const ROOT = '/Users/nsivaprasadnandyala/Desktop/DRDO_Project'
const SCHEMA = { type: 'object', properties: { issues: { type: 'array', items: { type: 'object', properties: {
  location: { type: 'string', description: 'section / line excerpt in REPORT.md' },
  claim: { type: 'string' }, evidence: { type: 'string', description: 'what the data/code actually shows, with file path' },
  severity: { type: 'string', enum: ['wrong', 'imprecise', 'unsupported', 'ok-but-note'] },
  suggested_fix: { type: 'string' } }, required: ['location', 'claim', 'evidence', 'severity', 'suggested_fix'] } },
  checked_count: { type: 'number' } }, required: ['issues', 'checked_count'] }
const COMMON = `You are a meticulous, skeptical auditor of an engineering report. Project root: ${ROOT}. READ-ONLY: do not create, modify or delete any file. You may run read-only commands (cat, grep, python with pandas reading CSV/JSON) using the venv: "source ${ROOT}/.venv/bin/activate". Report only real discrepancies (wrong numbers, wrong rounding beyond ±1 in the last digit, claims not supported by data/code, internal contradictions between sections). Skip claims marked [V] that cite external literature (already verified separately) unless they contradict the project's own data.`
const tasks = [
  { key: 'numbers', prompt: `${COMMON}
Audit EVERY number in ${ROOT}/reports/REPORT.md sections 0, 5, 6.2 (latency results), 8 against the raw data:
- reports/results/defence_v1/per_file.csv, dualmic_v1/per_file.csv, vbdemand/per_file.csv (means over files; the report uses input SNR >= -5 dB for overall defence numbers, -10 dB is a separate stress condition; dualmic covers -5..10 dB)
- reports/results/transparency_clean_input.json, latency_mac_m4.json, simulated_live_runs.json, dualmic_val/hybrid_tuning.json, finetune_decision_*.md
- logs/hq.out, logs/ll.out, logs/hq_metric.out, logs/ll_metric.out, logs/pilot.out (validation lines "[val]", elapsed_h)
Recompute aggregates yourself with pandas. Also check the system labels (which checkpoint is "deployed") are consistent across the report, exports/ and finetune_decision files.` },
  { key: 'claims', prompt: `${COMMON}
Audit the technical/process claims in ${ROOT}/reports/REPORT.md sections 1-7, 9, 10 and README.md against the code and artefacts:
- architecture description (src/danc/models/dancnet.py: bands, ERB units, deep-filter taps, look-ahead, normaliser window, params/MACs - you may run python to count params and call macs_per_frame for the configs in configs/train_hq.yaml)
- data pipeline claims (src/danc/data/*.py: split policy, SNR ranges, augmentation probabilities, test-set sizes/seeds, speaker counts in data/manifests/*.json)
- loss weights and training hyper-parameters (src/danc/train/*.py, configs/*.yaml)
- engine/coupling claims (src/danc/inference/engine.py defaults, gating formula, impulse handling in src/danc/adaptive/nlms.py)
- test count ("68 tests": run "python -m pytest -q --co -q tests | tail -1" read-only), ONNX contract (exports/DEPLOYMENT.json may not exist yet - check export_onnx.py instead), reproducibility commands (do the referenced configs/scripts exist?).
Flag anything the report states that the code does not do, or does differently.` },
]
const res = await parallel(tasks.map(t => () => agent(t.prompt, { label: `audit:${t.key}`, phase: 'Audit', schema: SCHEMA }).then(r => r ? { key: t.key, ...r } : null)))
return res.filter(Boolean)
