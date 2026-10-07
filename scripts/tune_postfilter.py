"""Tune the optional SPP-driven residual-noise post-filter on the VALIDATION set (held-out speakers).

    Y' = Y * g,  g = beta + (1 - beta) * s^gamma,  s = max(spp_bin, spp_frame) smoothed with attack/release
where spp_frame is the mean SPP over 300-3500 Hz.  beta = 1 disables it.  Only validation data is used.
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from danc.dsp import istft, stft  # noqa: E402
from danc.eval.metrics import all_metrics  # noqa: E402
from danc.inference.export_onnx import load_model  # noqa: E402
from danc.inference.postfilter import spp_postfilter  # noqa: E402
from danc.train.train import build_val_set  # noqa: E402


def _m(args):
    c, y = args
    m = all_metrics(c, y)
    return [m["pesq_wb"], m["pesq_nb"], m["stoi"], m["sisdr"], m["snr"]]


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hq_metric/last.pt"
    torch.set_num_threads(3)
    model = load_model(str(ROOT / ck))
    vn, vc, cats = build_val_set(120)
    specs, spps = [], []
    with torch.no_grad():
        for x in vn:
            X = torch.view_as_real(torch.from_numpy(stft(x)).to(torch.complex64))[None].float()
            Y, spp = model.enhance_spec(X)
            specs.append((Y[0, ..., 0] + 1j * Y[0, ..., 1]).numpy())
            spps.append(spp[0].numpy())
    grid = [(1.0, 1.0, 0.0)] + list(itertools.product([0.5, 0.35, 0.25, 0.18], [0.5, 1.0], [0.0, 0.6]))
    res = []
    with ProcessPoolExecutor(3) as ex:
        for beta, gamma, smooth in grid:
            outs = [istft(spp_postfilter(Y, s, beta, gamma, smooth), len(x)) for Y, s, x in zip(specs, spps, vn)]
            m = np.mean(list(ex.map(_m, list(zip(vc, outs)), chunksize=4)), 0)
            res.append({"beta": beta, "gamma": gamma, "smooth": smooth, "pesq_wb": m[0], "pesq_nb": m[1], "stoi": m[2],
                        "sisdr": m[3], "snr": m[4]})
            print({k: round(v, 3) for k, v in res[-1].items()}, flush=True)
    od = ROOT / "reports" / "results" / "postfilter_tuning"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"val_{Path(ck).parent.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
