"""Grid search of the NLMS<->DNN coupling on a stratified subset of dualmic_v1 (tuning only;
final numbers are always reported on the full set with the chosen setting)."""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from danc.eval.evaluate import load_set, _read  # noqa: E402
from danc.eval.metrics import all_metrics  # noqa: E402


def run(cfg_items):
    from danc.inference.engine import EngineConfig, HybridEngine
    from danc.inference.runners import ORTStepRunner
    cfg, items = cfg_items
    eng = HybridEngine(ORTStepRunner(str(ROOT / "exports" / "dancnet_hq.onnx")),
                       EngineConfig(limiter_dbfs=None, **cfg))
    res = []
    for it in items:
        x, r, c = _read(it["noisy"]), _read(it["reference"]), _read(it["clean"])
        y = eng.process_signal(x, r)
        m = all_metrics(c, y, nb=False)
        res.append([m["pesq_wb"], m["stoi"], m["snr"], m["sisdr"]])
    return cfg, np.mean(res, 0).round(4).tolist()


if __name__ == "__main__":
    set_name = sys.argv[1] if len(sys.argv) > 1 else "dualmic_val"
    items = load_set(set_name)                   # tune on the VALIDATION scenes (val speakers, train noise)
    grid = [dict(use_lms=False)]
    mus = [0.03, 0.05, 0.1, 0.3]
    gps = [4.0]
    if len(sys.argv) > 2 and sys.argv[2] == "--with-gate-power":   # audit follow-up: select the exponent on val too
        mus, gps = [0.05, 0.1, 0.3], [2.0, 4.0]
    for mu, taps, thr, gp in itertools.product(mus, [2, 4], [None, 0.3], gps):
        grid.append(dict(use_lms=True, spp_gate=True, lms_mu=mu, lms_taps=taps, gate_threshold=thr, gate_power=gp))
    with ProcessPoolExecutor(3) as ex:
        out = list(ex.map(run, [(g, items) for g in grid]))
    out.sort(key=lambda o: -(o[1][0] + 2 * o[1][1] + o[1][3] / 10))
    for cfg, (pq, st, sn, si) in out:
        print(f"PESQ {pq:.3f} STOI {st:.3f} SNR {sn:6.2f} SI-SDR {si:6.2f}  {cfg}")
    od = ROOT / "reports" / "results" / set_name
    od.mkdir(parents=True, exist_ok=True)
    tag = "_with_gate_power" if len(sys.argv) > 2 and sys.argv[2] == "--with-gate-power" else ""
    (od / f"hybrid_tuning{tag}.json").write_text(json.dumps(out, indent=1))
