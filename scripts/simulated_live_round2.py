"""Paced simulated-live runs of the round-2 deployed configurations through BOTH live launchers.

Same four dual-mic scenes as the round-1 run (reports/results/simulated_live_runs.json), each fed block by block on a
10 ms software clock (--paced) through the real per-block callback path:
  * deploy/jetson/danc_live.py  (torch-free Jetson launcher, safety wrapper) for HQ, LL and TWO_MIC
  * python -m danc.inference.realtime (development launcher) for TWO_MIC
Writes reports/results/simulated_live_runs_round2.json and keeps each run's raw output in logs/simulated_live_round2/.
Run on an otherwise idle machine:

    python scripts/simulated_live_round2.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["0135_impulsive_p00", "0060_nonstationary_m05", "0210_mixed_p05", "0015_stationary_p00"]
CONFIGS = {
    "HQ": ["--onnx", "exports/dancnet_v3hq_cont.onnx"],
    "LL": ["--onnx", "exports/dancnet_ll.onnx"],
    "TWO_MIC": ["--onnx", "exports/dancnet_v3_2mic.onnx", "--two-mic-cascade", "--fallback-onnx", "exports/dancnet_v3hq_cont.onnx"],
}
LAUNCHERS = {"danc_live": [sys.executable, "deploy/jetson/danc_live.py", "--no-preflight"],
             "realtime": [sys.executable, "-m", "danc.inference.realtime"]}


def run(launcher: str, cfg: str, scene: str, logdir: Path) -> dict:
    d = "data/testsets/dualmic_v1"
    cmd = LAUNCHERS[launcher] + CONFIGS[cfg] + ["--paced", "--simulate", f"{d}/primary/{scene}.flac",
                                                "--simulate-ref", f"{d}/reference/{scene}.flac"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"),
                                                                       "PATH": "/usr/bin:/bin"})
    (logdir / f"{launcher}_{cfg}_{scene}.out").write_text(" ".join(cmd) + "\n\n" + p.stdout + "\n" + p.stderr)
    txt = p.stdout
    st = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    st.update({"launcher": launcher, "config": cfg, "command": " ".join(cmd[1:])})
    return st


def main():
    logdir = ROOT / "logs" / "simulated_live_round2"
    logdir.mkdir(parents=True, exist_ok=True)
    out = {}
    jobs = [("danc_live", c, s) for c in CONFIGS for s in SCENES] + [("realtime", "TWO_MIC", s) for s in SCENES]
    for launcher, cfg, scene in jobs:
        st = run(launcher, cfg, scene, logdir)
        out[f"{launcher}/{cfg}/{scene}"] = st
        print(f"{launcher:9s} {cfg:8s} {scene:24s} mean {st['mean_ms']:.2f} p99 {st['p99_ms']:.2f} max {st['max_ms']:.2f} ms "
              f"misses(paced) {st['deadline_misses_paced']} latency {st['algorithmic_latency_ms']:.0f} ms"
              + (f" ref_switches {st['ref_switches']}" if "ref_switches" in st else ""), flush=True)
    (ROOT / "reports" / "results" / "simulated_live_runs_round2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
