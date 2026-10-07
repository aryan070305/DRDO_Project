"""Clean-speech transparency test: clean speech -> model -> compare with the input.
100 VoiceBank clean + 100 LibriSpeech test-clean utterances -> reports/results/transparency_clean_input.json

    python scripts/transparency_test.py --runs hq,hq_metric,ll,ll_metric
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from danc.data import speech  # noqa: E402
from danc.eval.evaluate import DancOffline  # noqa: E402
from danc.eval.metrics import all_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="hq,hq_metric,ll,ll_metric")
    a = ap.parse_args()
    torch.set_num_threads(2)
    vb = [sf.read(f, dtype="float32")[0] for f in sorted(glob.glob(str(ROOT / "data/raw/vbdemand_test/clean/*.wav")))[::8][:100]]
    ls = [speech.load_utt(u)[:16000 * 8] for u in speech.build_manifests()["test"][::26][:100]]
    p = ROOT / "reports" / "results" / "transparency_clean_input.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for run in a.runs.split(","):
        d = ROOT / "checkpoints" / run
        m = DancOffline(str(d / "best.pt") if (d / "best.pt").exists() else str(d / "last.pt"))
        for name, files in (("vbd_clean", vb), ("libri_clean", ls)):
            res = [[r["pesq_wb"], r["stoi"], r["sisdr"]] for r in (all_metrics(x, m(x), nb=False) for x in files)]
            out[f"{run}/{name}"] = dict(zip(["pesq_wb", "stoi", "sisdr"], np.mean(res, 0).round(4).tolist()))
            print(run, name, out[f"{run}/{name}"], flush=True)
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
