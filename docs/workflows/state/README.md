# Workflow final state files

Final state of each multi-agent workflow run in this project (round 1, 2026-10-03/04), copied on 2026-10-05 from
the Claude Code session folder. They complement `../journals/*.jsonl`. `../journals/wf_14dd115a-9e9.jsonl` is a partial
copy taken while that workflow (Jetson deployment docs and completeness review) was still running;
`wf_14dd115a-9e9.json` here is its complete final state (2026-10-04 09:30 +04). The workflow scripts are in `../*.js`.

`wf_79848172-d66` and `wf_d657bd8c-717` are resumed runs of `wf_d812ebf9-67c` (SOTA research) and `wf_33c61465-a05`
(parallel modules); their state records the same script paths. The exact script text each of them ran is extracted
from the `script` field of its state file into `../anc-sota-research-wf_79848172-d66.js` and
`../danc-parallel-modules-wf_d657bd8c-717.js`.
