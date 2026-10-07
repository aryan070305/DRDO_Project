#!/usr/bin/env python3
"""D-ANC deployment self-check.  Runs on ANY machine with the runtime dependencies - NO PyTorch needed.

    python3 deploy/jetson/verify_install.py                 # full check (about 1-2 min on an Apple M4)
    python3 deploy/jetson/verify_install.py --json verify_report.json
    python3 deploy/jetson/verify_install.py --skip-quality  # contract + robustness + timing only
    sudo -E $VIRTUAL_ENV/bin/python deploy/jetson/verify_install.py --strict-timing --cpus auto --rt-priority 80
                                                            # Jetson acceptance timing with RT settings (root)
    python3 deploy/jetson/verify_install.py --trt-engines deploy/jetson/engines   # also check TensorRT FP16 engines
    python3 deploy/jetson/verify_install.py --list-files    # data files the check needs (stdlib only)

Every check prints one line: PASS / FAIL / WARN / INFO.  Exit code 0 = no FAIL, 1 = at least one FAIL,
2 = files missing / bad arguments.

What is checked
  1. env        Python >= 3.10, runtime packages, ONNX Runtime providers, Jetson release (if present).
  2. torch-free The streaming engine and the live app import with torch BLOCKED (as on the Jetson).
  3. contract   Every deployed ONNX model in exports/DEPLOYMENT.json (round 2: HQ, LL, TWO_MIC; --models to choose):
                sha256, input/output names (the two-mic model also has 'spec_ref'),
                shapes, dtypes, number of recurrent states, look-ahead / STFT metadata, one zero-input step.
  4. quality    The streaming HybridEngine (danc.inference.engine) with a torch-free ONNX Runtime step
                runner (deploy/jetson/danc_jetson_rt.py, same math as danc.inference.runners.ORTStepRunner)
                on 8 dual-mic scenes (reference-mic NLMS + DNN; for TWO_MIC the entry's "engine" options select the
                NLMS -> two-mic cascade) and 8 single-mic defence_v1 files per single-mic model.
                PESQ-WB, PESQ-NB, STOI and SNR are compared per file with the reference rows in
                reports/results/{dualmic_v1,defence_v1}/per_file.csv
                (built-in round-1 defaults HQ: hybrid:hqft / dancnet:hq_metric, LL: hybrid:ll / dancnet:ll; each
                DEPLOYMENT.json entry overrides them with "reference_systems", as all round-2 entries do).
                Tolerances: |dPESQ| < 0.05 (WB and NB), |dSTOI| < 0.01, |dSNR| < 0.25 dB.
                File choice (deterministic): per noise category, the first file at the lowest and the first
                file at the highest main-range input SNR (dual-mic -5/10 dB, defence -5/15 dB).
  5. robust     Edge cases: digital silence, full-scale clipped input on both mics (limiter must hold
                -1 dBFS), reference mic absent / dead, NaN+Inf burst (safety wrapper keeps the output finite),
                reset() determinism (service restart gives identical output), bounded per-block timing
                history (the live stats run inside the audio callback; unbounded they cost ~25 ms after 1 h).
  6. timing     3000 hops (30 s of audio) of the full live engine (NLMS + DNN + limiter + safety wrapper)
                per model: p50 / p99 / p99.9 / max vs the 10 ms hop deadline.  Default (install check):
                FAIL if p99 >= 10 ms; WARN if any hop > 10 ms or p99.9 >= 5 ms.  With --strict-timing
                (field acceptance, deploy/jetson/README.md): FAIL on any hop > 10 ms or p99.9 >= 5 ms.
                The formal acceptance test is the 10-minute live run in QUICKSTART.md.

Reference values were produced on an Apple M4 (macOS, onnxruntime 1.30, CPU EP).  The defence_v1 references
were computed with the offline PyTorch model; the streaming ONNX engine matches it to <= 1e-4 (unit tests),
which is far inside the tolerances above.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]
HOP_MS = 10.0

# reference systems in reports/results/<set>/per_file.csv for each deployed model.  A DEPLOYMENT.json entry may
# override them with  "reference_systems": {"dualmic_v1": "<system>", "defence_v1": "<system>"}  - do that (and
# re-run danc.eval.evaluate for that system) whenever a re-trained model is deployed under the HQ/LL key.
REF_SYSTEMS = {
    "HQ": {"dualmic_v1": "hybrid:hqft", "defence_v1": "dancnet:hq_metric"},
    "LL": {"dualmic_v1": "hybrid:ll", "defence_v1": "dancnet:ll"},
}


def ref_systems(root: Path, model: str) -> dict:
    """Reference system names for one deployed model: DEPLOYMENT.json override, else REF_SYSTEMS."""
    try:
        override = load_deployment(root)["deployed"].get(model, {}).get("reference_systems") or {}
    except (OSError, ValueError, KeyError):
        override = {}
    return {**REF_SYSTEMS.get(model, {}), **override}
# test sets: which SNRs to sample and which audio sub-folders each scene needs
SETS = {
    "dualmic_v1": {"snrs": (-5.0, 10.0), "dirs": ("primary", "reference", "clean"), "hybrid": True},
    "defence_v1": {"snrs": (-5.0, 15.0), "dirs": ("noisy", "clean"), "hybrid": False},
}
TOL = {"pesq_wb": 0.05, "pesq_nb": 0.05, "stoi": 0.01, "snr": 0.25}


# =============================================================================== reporting
class Report:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, group: str, name: str, status: str, detail: str = "", **data):
        self.items.append({"group": group, "name": name, "status": status, "detail": detail, **data})
        print(f"[{status:4s}] {group:10s} {name}" + (f"  -- {detail}" if detail else ""), flush=True)

    def check(self, group, name, ok: bool, detail="", warn=False, **data):
        self.add(group, name, "PASS" if ok else ("WARN" if warn else "FAIL"), detail, **data)
        return ok

    def counts(self):
        c = {"PASS": 0, "FAIL": 0, "WARN": 0, "INFO": 0, "SKIP": 0}
        for it in self.items:
            c[it["status"]] = c.get(it["status"], 0) + 1
        return c


# =============================================================================== stdlib-only helpers
def load_deployment(root: Path) -> dict:
    return json.loads((root / "exports" / "DEPLOYMENT.json").read_text())


def engine_opts(root: Path, model: str) -> dict:
    """Per-model EngineConfig overrides from DEPLOYMENT.json ("engine": {...}), e.g. {"lms_two_mic": true} for the
    two-microphone cascade; empty for the single-mic HQ / LL models."""
    try:
        return dict(load_deployment(root)["deployed"].get(model, {}).get("engine") or {})
    except (OSError, ValueError, KeyError):
        return {}


def select_ids(root: Path, set_name: str) -> list[str]:
    """Deterministic 8-file selection: per category (in meta.csv order) first id at the low and high SNR."""
    meta = list(csv.DictReader(open(root / "data" / "testsets" / set_name / "meta.csv", newline="")))
    lo, hi = SETS[set_name]["snrs"]
    cats: list[str] = []
    for r in meta:
        if r["category"] not in cats:
            cats.append(r["category"])
    ids: list[str] = []
    for c in cats:
        rows = [r for r in meta if r["category"] == c]
        pick = []
        for target in (lo, hi):
            m = [r["id"] for r in rows if float(r["snr_db"]) == target and r["id"] not in pick]
            if m:
                pick.append(m[0])
        for r in rows:                       # fallback if an SNR is missing: next files of the category
            if len(pick) >= 2:
                break
            if r["id"] not in pick:
                pick.append(r["id"])
        ids.extend(pick)
    return ids


def needed_files(root: Path) -> list[str]:
    """Relative paths of every data file the quality check reads (used by package_for_jetson.sh)."""
    out = []
    for s, cfg in SETS.items():
        out.append(f"data/testsets/{s}/meta.csv")
        for i in select_ids(root, s):
            out.extend(f"data/testsets/{s}/{d}/{i}.flac" for d in cfg["dirs"])
    return out


def reference_rows(root: Path, set_name: str, ids: list[str], systems: list[str]) -> list[dict]:
    p = root / "reports" / "results" / set_name / "per_file.csv"
    want_ids, want_sys = set(ids), set(systems)
    with open(p, newline="") as f:
        return [r for r in csv.DictReader(f) if r["system"] in want_sys and r["id"] in want_ids]


def write_reference_subset(root: Path, dest_root: Path) -> list[str]:
    """Write the reference rows (only the selected files / deployed systems) under dest_root, same layout."""
    written = []
    models = list(load_deployment(root)["deployed"])
    for s in SETS:
        systems = sorted({ref_systems(root, m)[s] for m in models if s in ref_systems(root, m)})
        src = root / "reports" / "results" / s / "per_file.csv"
        with open(src, newline="") as f:
            header = next(csv.reader(f))
        rows = reference_rows(root, s, select_ids(root, s), systems)
        out = dest_root / "reports" / "results" / s / "per_file.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        written.append(f"{out} ({len(rows)} rows)")
    return written


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================== contract
def check_contract(onnx_path, spec: dict) -> list[tuple[str, bool, str]]:
    """Check one ONNX model against its exports/DEPLOYMENT.json entry. Returns [(check, ok, detail)].

    Needs only numpy + onnxruntime (no danc, no torch); used by tests/test_deploy_kit.py.
    """
    import numpy as np
    import onnxruntime as ort

    p = Path(onnx_path)
    if not p.exists():
        return [("file exists", False, str(p))]
    res: list[tuple[str, bool, str]] = []
    h = sha256_file(p)
    exp_h = spec.get("sha256_onnx", "")
    res.append(("sha256 matches DEPLOYMENT.json", h == exp_h, f"{h[:16]}... expected {exp_h[:16]}..."))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    ins, outs = s.get_inputs(), s.get_outputs()
    exp_in: dict = spec["inputs"]
    got_names = [i.name for i in ins]
    res.append(("input names/order", got_names == list(exp_in), f"{got_names}"))
    bad_shapes = {i.name: (list(i.shape), exp_in.get(i.name)) for i in ins if list(i.shape) != exp_in.get(i.name)}
    res.append(("input shapes", not bad_shapes, "all static, as in DEPLOYMENT.json" if not bad_shapes else f"{bad_shapes}"))
    res.append(("input dtypes float32", all(i.type == "tensor(float)" for i in ins),
                f"{sorted({i.type for i in ins})}"))
    out_names = [o.name for o in outs]
    res.append(("output names/order", out_names == list(spec["outputs"]), f"{out_names}"))
    n_si = sum(n.startswith("state_in_") for n in got_names)
    n_so = sum(n.startswith("state_out_") for n in out_names)
    n_exp = sum(k.startswith("state_in_") for k in exp_in)
    state_kb = sum(int(np.prod(v)) for k, v in exp_in.items() if k.startswith("state_in_")) * 4 / 1024
    res.append(("number of recurrent states", n_si == n_so == n_exp,
                f"{n_si} in / {n_so} out (expected {n_exp}, {state_kb:.1f} KB)"))

    meta = dict(s.get_modelmeta().custom_metadata_map)
    md_exp = spec.get("metadata", {})
    diff = {k: (meta.get(k), v) for k, v in md_exp.items() if meta.get(k) != v}
    res.append(("metadata == DEPLOYMENT.json", not diff, "all keys equal" if not diff else f"{diff}"))
    try:
        fs, win, hop, nfft = (int(meta[k]) for k in ("fs", "win", "hop", "n_fft"))
        la = int(meta["lookahead"])
        lat = float(meta["algorithmic_latency_ms"])
        stft_ok = (fs, win, hop, nfft) == (16000, 320, 160, 320)
        try:   # also against the package's own STFT definition when it is importable
            from danc.dsp import DEFAULT
            stft_ok = stft_ok and (DEFAULT.fs, DEFAULT.win, DEFAULT.hop, DEFAULT.n_fft) == (fs, win, hop, nfft)
        except ImportError:
            pass
        res.append(("STFT metadata (16 kHz, 20 ms win, 10 ms hop)", stft_ok, f"fs={fs} win={win} hop={hop} n_fft={nfft}"))
        exp_lat = 1000.0 * (win + la * hop) / fs
        cfg_la = json.loads(meta.get("model_cfg", "{}")).get("lookahead", la)
        res.append(("look-ahead metadata consistent", abs(lat - exp_lat) < 1e-6 and cfg_la == la,
                    f"lookahead={la} frames -> {lat:g} ms algorithmic (expected {exp_lat:g})"))
    except (KeyError, ValueError) as e:
        res.append(("STFT / look-ahead metadata present", False, f"missing/invalid: {e}"))

    feeds = {i.name: np.zeros([int(d) for d in i.shape], np.float32) for i in ins}
    try:
        o = dict(zip(out_names, s.run(None, feeds)))
        f_bins = exp_in["spec"][2]
        ok = (o["spec_out"].shape == tuple(exp_in["spec"]) and o["spp"].size == f_bins
              and all(o[k.replace("state_in_", "state_out_")].shape == tuple(v)
                      for k, v in exp_in.items() if k.startswith("state_in_"))
              and all(np.all(np.isfinite(v)) for v in o.values())
              and float(o["spp"].min()) >= 0.0 and float(o["spp"].max()) <= 1.0)
        res.append(("zero-input step (shapes, finite, spp in [0,1])", bool(ok),
                    f"spec_out {list(o['spec_out'].shape)}, spp {list(o['spp'].shape)}"))
    except Exception as e:  # noqa: BLE001
        res.append(("zero-input step", False, repr(e)))
    return res


# =============================================================================== engine helpers
def _read(p: Path):
    import soundfile as sf

    x, fs = sf.read(str(p), dtype="float32")
    if fs != 16000:
        raise ValueError(f"{p}: {fs} Hz, expected 16000")
    return x


def _make_runner(onnx_path: Path, trt_engine: Path | None, threads: int, providers):
    import danc_jetson_rt as rt

    if trt_engine is not None:
        return rt.make_trt_runner(str(trt_engine), str(onnx_path), cuda_graph=False)
    return rt.ORTStepRunner(str(onnx_path), threads=threads, providers=providers)


def run_quality(rep: Report, root: Path, model: str, onnx_path: Path, runner, label: str) -> dict:
    import numpy as np
    from danc.eval.metrics import all_metrics
    from danc.inference.engine import EngineConfig, HybridEngine

    summary = {}
    systems = ref_systems(root, model)
    for s, cfg in SETS.items():
        ids = select_ids(root, s)
        if getattr(runner, "n_mics", 1) == 2 and not cfg["hybrid"]:
            rep.add("quality", f"{label} {s}", "INFO", "skipped: two-microphone model, single-microphone test set")
            continue
        if s not in systems:
            rep.add("quality", f"{label} {s}", "FAIL", f"no reference system for model '{model}' (add "
                    "\"reference_systems\" to its exports/DEPLOYMENT.json entry)")
            continue
        sysname = systems[s]
        refs = {r["id"]: r for r in reference_rows(root, s, ids, [sysname])}
        d = root / "data" / "testsets" / s
        # exactly the evaluation configuration of danc.eval.evaluate (limiter off for the metric runs)
        base = dict(use_lms=True, spp_gate=True, limiter_dbfs=None) if cfg["hybrid"] else dict(use_lms=False, limiter_dbfs=None)
        ecfg = EngineConfig(**{**base, **engine_opts(root, model)})
        eng = HybridEngine(runner, ecfg)
        deltas = {k: [] for k in TOL}
        got_all = {k: [] for k in TOL}
        ref_all = {k: [] for k in TOL}
        n_fail = 0
        t0 = time.time()
        for i in ids:
            if i not in refs:
                rep.add("quality", f"{label} {s}/{i}", "FAIL", f"no reference row for system '{sysname}' in per_file.csv")
                n_fail += 1
                continue
            clean = _read(d / "clean" / f"{i}.flac")
            if cfg["hybrid"]:
                y = eng.process_signal(_read(d / "primary" / f"{i}.flac"), _read(d / "reference" / f"{i}.flac"))
            else:
                y = eng.process_signal(_read(d / "noisy" / f"{i}.flac"), None)
            m = all_metrics(clean, y)
            r = refs[i]
            bad = []
            for k, tol in TOL.items():
                g, rv = float(m[k]), float(r[k])
                dv = g - rv
                deltas[k].append(abs(dv))
                got_all[k].append(g)
                ref_all[k].append(rv)
                if not (abs(dv) < tol) or not np.isfinite(g):
                    bad.append(f"{k} {g:.3f} vs ref {rv:.3f}")
            ok = not bad
            n_fail += 0 if ok else 1
            rep.check("quality", f"{label} {s}/{i}", ok,
                      (f"PESQ-WB {m['pesq_wb']:.3f} (ref {float(r['pesq_wb']):.3f})  PESQ-NB {m['pesq_nb']:.3f} "
                       f"(ref {float(r['pesq_nb']):.3f})  STOI {m['stoi']:.3f} (ref {float(r['stoi']):.3f})  "
                       f"SNR {m['snr']:.2f} dB (ref {float(r['snr']):.2f})") if ok else "; ".join(bad),
                      metrics={k: float(m[k]) for k in ("pesq_wb", "pesq_nb", "stoi", "estoi", "snr")})
        if deltas["pesq_wb"]:                      # at least one file evaluated
            means = {k: float(np.mean(v)) for k, v in got_all.items()}
            rmeans = {k: float(np.mean(v)) for k, v in ref_all.items()}
            maxd = {k: float(np.max(v)) for k, v in deltas.items()}
            rep.check("quality", f"{label} {s} mean of {len(ids)} files", n_fail == 0,
                      f"PESQ-WB {means['pesq_wb']:.3f}/{rmeans['pesq_wb']:.3f}  PESQ-NB {means['pesq_nb']:.3f}/"
                      f"{rmeans['pesq_nb']:.3f}  STOI {means['stoi']:.3f}/{rmeans['stoi']:.3f}  SNR "
                      f"{means['snr']:.2f}/{rmeans['snr']:.2f} dB (got/ref); max |d|: PESQ-WB {maxd['pesq_wb']:.4f} "
                      f"PESQ-NB {maxd['pesq_nb']:.4f} STOI {maxd['stoi']:.5f} SNR {maxd['snr']:.3f}; "
                      f"{time.time() - t0:.1f} s", means=means, ref_means=rmeans, max_abs_delta=maxd)
            summary[s] = {"means": means, "ref_means": rmeans, "max_abs_delta": maxd, "failed_files": n_fail}
    return summary


def run_robustness(rep: Report, root: Path, model: str, runner, fallback=None):
    import numpy as np
    import danc_jetson_rt as rt
    from danc.inference.engine import EngineConfig as _EC, HybridEngine

    opts = engine_opts(root, model)

    def EngineConfig(**kw):   # noqa: N802  (the model's own engine options, e.g. the two-mic cascade)
        return _EC(**{**opts, **kw})

    thr = 10 ** (-1.0 / 20)
    rng = np.random.default_rng(1234)
    Safe = rt.SafeHybridEngine
    hop = 160

    # 1) digital silence must stay (near) silent - no self-generated noise/hiss into the radio
    eng = Safe(runner, EngineConfig())
    y = np.concatenate([eng.process_block(np.zeros(hop), np.zeros(hop)) for _ in range(150)])
    rep.check("robust", f"{model} digital silence in -> silence out", bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e-4),
              f"max |y| = {np.max(np.abs(y)):.2e}")

    # 2) full-scale clipped noise on both mics (gunfire-like overload): limiter must hold -1 dBFS
    eng = Safe(runner, EngineConfig())
    y = np.concatenate([eng.process_block(np.sign(rng.standard_normal(hop)), np.sign(rng.standard_normal(hop)))
                        for _ in range(200)])
    rep.check("robust", f"{model} full-scale clipped input, limiter -1 dBFS",
              bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) <= thr + 1e-9),
              f"peak {20 * np.log10(max(np.max(np.abs(y)), 1e-12)):.2f} dBFS")

    # real scene for the next checks
    sid = select_ids(root, "dualmic_v1")[0]
    d = root / "data" / "testsets" / "dualmic_v1"
    prim = _read(d / "primary" / f"{sid}.flac").astype(np.float64)[: 300 * hop]
    ref = _read(d / "reference" / f"{sid}.flac").astype(np.float64)[: 300 * hop]
    nb = len(prim) // hop

    def run(e, p, r):
        return np.concatenate([e.process_block(p[k * hop:(k + 1) * hop], None if r is None else r[k * hop:(k + 1) * hop])
                               for k in range(nb)])

    # 3) reference mic absent (None) and dead (zeros): graceful DNN-only fallback
    y_none = run(Safe(runner, EngineConfig()), prim, None)
    y_zero = run(Safe(runner, EngineConfig()), prim, np.zeros_like(ref))
    rep.check("robust", f"{model} reference mic absent / dead -> DNN-only",
              bool(np.all(np.isfinite(y_none)) and np.allclose(y_none, y_zero, atol=1e-7) and np.max(np.abs(y_none)) <= thr + 1e-9),
              f"outputs identical: {np.allclose(y_none, y_zero, atol=1e-7)}")

    # 4) NaN / Inf burst on both mics: safety wrapper keeps the output finite and in range
    p_bad, r_bad = prim.copy(), ref.copy()
    p_bad[50 * hop:53 * hop] = np.nan
    r_bad[50 * hop:53 * hop] = np.nan
    p_bad[120 * hop:123 * hop] = np.inf
    r_bad[121 * hop:124 * hop] = -np.inf
    for lms in (True, False):
        e = Safe(runner, EngineConfig(use_lms=lms))
        with np.errstate(all="ignore"):
            y = run(e, p_bad, r_bad)
        rep.check("robust", f"{model} NaN/Inf burst ({'hybrid' if lms else 'DNN-only'}), safety wrapper",
                  bool(np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0 and np.all(np.isfinite(y[-100 * hop:]))),
                  f"output finite throughout; tail RMS {20 * np.log10(np.sqrt(np.mean(y[-100 * hop:] ** 2)) + 1e-12):.1f} dBFS")
    # same burst WITHOUT the wrapper: the plain engine must recover by itself (informational)
    e = HybridEngine(runner, EngineConfig(use_lms=False))
    with np.errstate(all="ignore"):
        y = run(e, p_bad, r_bad)
    nonfin = np.where(~np.isfinite(y.reshape(-1, hop)).all(1))[0]
    rec = bool(np.all(np.isfinite(y[-100 * hop:])))
    if fallback is not None:   # reference-health monitor: a reference that dies mid-stream -> single-mic network
        r_dead = ref.copy()
        r_dead[100 * hop:] = 0.0
        e = Safe(runner, EngineConfig(), fallback_runner=fallback)
        y = run(e, prim, r_dead)
        rep.check("robust", f"{model} reference dies at 1 s -> switches to the single-mic fallback",
                  bool(e.ref_dead and e.ref_switches == 1 and np.all(np.isfinite(y))),
                  f"ref_dead={e.ref_dead}, switches={e.ref_switches}")
    rep.check("robust", f"{model} plain engine recovers after NaN/Inf burst", rec,
              f"{len(nonfin)} non-finite output blocks (the live launcher's safety wrapper zeroes them)", warn=True)

    # 5) reset() determinism: a service restart must reproduce the output bit-exactly
    e = Safe(runner, EngineConfig())
    a = run(e, prim, ref)
    e.reset()
    b = run(e, prim, ref)
    rep.check("robust", f"{model} reset() determinism", bool(np.array_equal(a, b)), f"max |diff| = {np.max(np.abs(a - b)):.1e}")

    # 6) long-run stability: danc.inference.realtime computes timing_stats() INSIDE the audio callback every
    #    500 blocks; with an unbounded history that costs ~25 ms after 1 h (M4).  The wrapper keeps it bounded.
    e = Safe(runner, EngineConfig())
    e.timing = [0.0005] * 360_000                      # = 1 hour of 10 ms blocks
    e.process_block(prim[:hop], ref[:hop])
    t0 = time.perf_counter()
    e.timing_stats()
    dt = (time.perf_counter() - t0) * 1000.0
    rep.check("robust", f"{model} long-run: bounded timing history (1 h of blocks)",
              len(e.timing) <= 2 * e.TIMING_WINDOW and dt < 5.0,
              f"{len(e.timing)} entries kept, timing_stats() {dt:.2f} ms (plain engine after 1 h: ~25 ms on an M4)")
    # 7) the same through the engine's OWN timing container (a bounded deque since round 2): run past the trim point
    #    (2 x TIMING_WINDOW blocks = 30 s) without replacing it - the first wrapper version crashed exactly there
    e = Safe(None, EngineConfig(use_lms=False, limiter_dbfs=None))
    z, ok = np.zeros(hop), True
    try:
        for _ in range(2 * e.TIMING_WINDOW + 50):
            e.process_block(z, z)
    except Exception as ex:   # noqa: BLE001
        ok, msg = False, f"{type(ex).__name__}: {ex}"
    else:
        msg = f"{2 * e.TIMING_WINDOW + 50} blocks, {len(e.timing)} timings kept"
    rep.check("robust", f"{model} live wrapper runs past 30 s (timing-history trim)", ok, msg)


def run_timing(rep: Report, root: Path, model: str, runner, hops: int, label: str, strict: bool = False,
               fallback=None) -> dict:
    import numpy as np
    import danc_jetson_rt as rt
    from danc.inference.engine import EngineConfig

    hop = 160
    sid = [i for i in select_ids(root, "dualmic_v1") if "impulsive" in i][:1] or select_ids(root, "dualmic_v1")[:1]
    d = root / "data" / "testsets" / "dualmic_v1"
    prim = _read(d / "primary" / f"{sid[0]}.flac").astype(np.float64)
    ref = _read(d / "reference" / f"{sid[0]}.flac").astype(np.float64)
    warm = 50
    n = (hops + warm) * hop
    reps = int(np.ceil(n / len(prim)))
    prim, ref = np.tile(prim, reps)[:n], np.tile(ref, reps)[:n]
    kw = {"fallback_runner": fallback} if fallback is not None else {}
    eng = rt.SafeHybridEngine(runner, EngineConfig(**engine_opts(root, model)), **kw)   # live defaults (+ model options)
    t = np.empty(hops)
    finite = True
    for k in range(hops + warm):
        sl = slice(k * hop, (k + 1) * hop)
        t0 = time.perf_counter()
        y = eng.process_block(prim[sl], ref[sl])
        dt = time.perf_counter() - t0
        if k >= warm:
            t[k - warm] = dt * 1000.0
        finite = finite and bool(np.all(np.isfinite(y)))
    st = {"hops": hops, "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
          "p99_ms": float(np.percentile(t, 99)), "p99_9_ms": float(np.percentile(t, 99.9)), "max_ms": float(t.max()),
          "misses_gt_10ms": int(np.sum(t > HOP_MS)), "rtf": float(t.mean() / HOP_MS)}
    txt = (f"mean {st['mean_ms']:.3f}  p50 {st['p50_ms']:.3f}  p99 {st['p99_ms']:.3f}  p99.9 {st['p99_9_ms']:.3f}  "
           f"max {st['max_ms']:.3f} ms; hops > 10 ms: {st['misses_gt_10ms']}/{hops}; RTF {st['rtf']:.3f}")
    st["strict"] = bool(strict)
    if strict:   # field acceptance (deploy/jetson/README.md): no hop over 10 ms, p99.9 < 5 ms
        rep.check("timing", f"{label} {hops} hops STRICT: 0 hops > 10 ms and p99.9 < 5 ms",
                  st["misses_gt_10ms"] == 0 and st["p99_9_ms"] < 5.0 and finite, txt, timing=st)
        return st
    # install check: 99 % of hops must meet the deadline in this unpaced, non-RT micro-benchmark
    rep.check("timing", f"{label} {hops} hops, p99 < 10 ms deadline", st["p99_ms"] < HOP_MS and finite, txt, timing=st)
    if st["misses_gt_10ms"] > 0:
        rep.add("timing", f"{label} hops over the 10 ms deadline", "WARN",
                f"{st['misses_gt_10ms']} hop(s), max {st['max_ms']:.2f} ms - typical of a shared/loaded machine or "
                "missing RT settings; on the Jetson run with --strict-timing --cpus auto --rt-priority 80 (performance "
                "governor), then the 10-minute live test")
    if st["p99_9_ms"] >= 5.0:
        rep.add("timing", f"{label} p99.9 above the 5 ms acceptance target", "WARN", f"p99.9 {st['p99_9_ms']:.2f} ms")
    return st


# =============================================================================== environment
def check_env(rep: Report):
    rep.check("env", "python >= 3.10", sys.version_info >= (3, 10), platform.python_version())
    mods = ["numpy", "scipy", "soundfile", "soxr", "onnxruntime", "pesq", "pystoi"]
    missing = []
    vers = {}
    for m in mods:
        try:
            mod = __import__(m)
            vers[m] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m} ({type(e).__name__})")
    rep.check("env", "runtime packages importable", not missing,
              ", ".join(f"{k} {v}" for k, v in vers.items()) if not missing else "missing: " + ", ".join(missing)
              + " -> run deploy/jetson/install.sh")
    try:
        import sounddevice as sd
        rep.add("env", "sounddevice / PortAudio", "PASS", f"sounddevice {sd.__version__}, {sd.get_portaudio_version()[1]}")
    except Exception as e:  # noqa: BLE001
        rep.add("env", "sounddevice / PortAudio (live audio only)", "WARN",
                f"{type(e).__name__}: {e} -> sudo apt-get install libportaudio2; pip install sounddevice")
    try:
        import onnxruntime as ort
        rep.add("env", "onnxruntime providers", "INFO", ", ".join(ort.get_available_providers()))
    except Exception:  # noqa: BLE001
        pass
    rel = Path("/etc/nv_tegra_release")
    if rel.exists():
        rep.add("env", "Jetson release", "INFO", rel.read_text().strip().splitlines()[0])
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.exists():
        g = gov.read_text().strip()
        rep.add("env", "CPU governor (cpu0)", "INFO" if g == "performance" else "WARN",
                g + ("" if g == "performance" else " -> set 'performance' on every CPU (QUICKSTART.md step 11)"))
    try:
        import danc_jetson_rt as rt
        rep.add("env", "online CPUs", "INFO", str(rt.online_cpus()))
    except Exception:  # noqa: BLE001
        pass
    rep.add("env", "host", "INFO", f"{platform.platform()} ({platform.machine()})")


def check_torch_free(rep: Report):
    import importlib
    import danc_jetson_rt as rt

    blocked = any(isinstance(f, _TorchBlocker) for f in sys.meta_path)
    try:
        import danc
        import danc.adaptive.nlms  # noqa: F401
        import danc.dsp  # noqa: F401
        import danc.eval.metrics  # noqa: F401
        import danc.inference.engine  # noqa: F401
        rep.check("torch-free", "engine modules import without torch", True,
                  f"danc from {Path(danc.__file__).parent}; torch {'blocked' if blocked else 'not blocked (--allow-torch)'}")
    except ImportError as e:
        rep.check("torch-free", "engine modules import without torch", False, repr(e))
        return
    try:
        if "danc.inference.runners" not in sys.modules:
            importlib.import_module("danc.inference.runners")
            rep.add("torch-free", "danc.inference.runners", "INFO", "imported (torch available)")
    except ImportError:
        rep.add("torch-free", "danc.inference.runners needs torch", "INFO",
                "expected: deploy/jetson/danc_live.py registers a torch-free runner instead")
    try:
        rt.install_runners_shim(force=True)
        importlib.import_module("danc.inference.realtime")
        rep.check("torch-free", "live app (danc.inference.realtime) imports via danc_live.py shim", True)
    except Exception as e:  # noqa: BLE001
        rep.check("torch-free", "live app (danc.inference.realtime) imports via danc_live.py shim", False, repr(e))


def check_runner_parity(rep: Report, onnx_path: Path, model: str):
    """danc_jetson_rt.ORTStepRunner == trt_runner.ORTStepRunner (adapter path used for TensorRT)."""
    import numpy as np
    import danc_jetson_rt as rt

    a = rt.ORTStepRunner(str(onnx_path))
    if a.n_mics != 1:
        rep.add("contract", f"{model} runner parity (danc_jetson_rt vs trt_runner adapter)", "INFO",
                "skipped: two-microphone model (input 'spec_ref'); trt_runner.py supports single-mic models only")
        return
    b = rt.EngineRunnerAdapter(rt.import_trt_runner().ORTStepRunner(str(onnx_path)), a.lookahead)
    rng = np.random.default_rng(7)
    md = 0.0
    for _ in range(40):
        x = (0.1 * rng.standard_normal((1, 1, 161, 2))).astype(np.float32)
        (ya, sa), (yb, sb) = a(x), b(x)
        md = max(md, float(np.max(np.abs(ya - yb))), float(np.max(np.abs(sa - sb))))
    rep.check("contract", f"{model} runner parity (danc_jetson_rt vs trt_runner adapter)", md < 1e-6, f"max |diff| {md:.1e}")


# =============================================================================== main
class _TorchBlocker:
    """Meta-path finder that makes 'import torch' / 'import torchaudio' fail, as on a torch-free Jetson."""

    BLOCKED = ("torch", "torchaudio", "torchvision")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(f"No module named '{name}' (blocked by verify_install.py: the Jetson "
                                      "runtime has no PyTorch; pass --allow-torch to disable)", name=name)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="project or bundle root (default: two levels up)")
    ap.add_argument("--models", default="all", help="deployed models to check: keys of DEPLOYMENT.json, or 'all'")
    ap.add_argument("--hops", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads (1 = real-time setting)")
    ap.add_argument("--providers", default=None, help="ORT providers, e.g. CUDAExecutionProvider,CPUExecutionProvider")
    ap.add_argument("--trt-engines", default=None, help="dir with <onnx stem>_fp16.engine from build_trt_engine.sh")
    ap.add_argument("--cpus", default=None, help="pin for the timing test: auto | 10,11 (Linux)")
    ap.add_argument("--rt-priority", type=int, default=0, help="SCHED_FIFO priority for the timing test (Linux, root)")
    ap.add_argument("--strict-timing", action="store_true",
                    help="field acceptance: FAIL on any hop > 10 ms or p99.9 >= 5 ms (use on the Jetson with RT settings)")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--allow-torch", action="store_true", help="do not block 'import torch' (default: blocked)")
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--list-files", action="store_true", help="print the data files needed (relative paths) and exit")
    ap.add_argument("--write-refs", default=None, metavar="DEST_ROOT",
                    help="write the reference rows of per_file.csv for the selected files under DEST_ROOT and exit")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()

    if a.list_files:
        for f in needed_files(root):
            print(f)
        return 0
    if a.write_refs:
        for w in write_reference_subset(root, Path(a.write_refs).resolve()):
            print(w)
        return 0

    # Run exactly like a torch-free Jetson: any 'import torch' raises ImportError from here on.
    # (A meta-path finder, not sys.modules['torch'] = None: scipy probes sys.modules for torch.)
    if not a.allow_torch and "torch" not in sys.modules:
        sys.meta_path.insert(0, _TorchBlocker())
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(HERE))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    providers = [p for p in a.providers.split(",")] if a.providers else None

    print(f"D-ANC verify_install  root={root}")
    print("Reference numbers come from an Apple M4 (macOS). Jetson results are NOT VERIFIED ON HARDWARE IN THIS PROJECT.")
    rep = Report()
    t_start = time.time()
    check_env(rep)
    missing = [f for f in ["exports/DEPLOYMENT.json"] if not (root / f).exists()]
    if missing:
        rep.add("files", "deployment manifest", "FAIL", f"missing {missing}")
        return 2
    dep = load_deployment(root)["deployed"]
    models = list(dep) if a.models.strip() == "all" else [m.strip() for m in a.models.split(",") if m.strip()]
    data_missing = [f for f in needed_files(root) if not (root / f).exists()] if not (a.skip_quality and a.skip_robustness and a.skip_timing) else []
    ref_missing = [f"reports/results/{s}/per_file.csv" for s in SETS if not (root / "reports" / "results" / s / "per_file.csv").exists()]
    rep.check("files", "test audio present (16 scenes + meta)", not data_missing,
              f"{len(needed_files(root))} files" if not data_missing else f"missing {data_missing[:4]}...")
    if not a.skip_quality:
        rep.check("files", "reference per_file.csv present", not ref_missing, ", ".join(ref_missing) or "dualmic_v1, defence_v1")
    check_torch_free(rep)

    results = {"models": {}}
    rt_info = None
    for m in models:
        if m not in dep:
            rep.add("contract", f"{m}", "FAIL", f"not in DEPLOYMENT.json (has {list(dep)})")
            continue
        spec = dep[m]
        onnx = root / spec["onnx"]
        for name, ok, detail in check_contract(onnx, spec):
            rep.check("contract", f"{m} {name}", ok, detail)
        if not onnx.exists():
            continue
        check_runner_parity(rep, onnx, m)
        mres = {"onnx": spec["onnx"]}
        backends = [("ORT", None)]
        if a.trt_engines:
            eng_path = Path(a.trt_engines) / f"{onnx.stem}_fp16.engine"
            if eng_path.exists():
                backends.append(("TRT-FP16", eng_path))
            else:
                rep.add("trt", f"{m} engine", "WARN", f"{eng_path} not found (build with build_trt_engine.sh)")
        fallback = None
        if spec.get("fallback") in dep:          # two-mic entry: single-mic model kept in hot standby
            try:
                fallback = _make_runner(root / dep[spec["fallback"]]["onnx"], None, a.threads, providers)
            except Exception as e:  # noqa: BLE001
                rep.add("runner", f"{m} fallback {spec['fallback']}", "FAIL", repr(e))
        for bname, eng_path in backends:
            label = f"{m}/{bname}"
            try:
                runner = _make_runner(onnx, eng_path, a.threads, providers)
            except Exception as e:  # noqa: BLE001
                rep.add("runner", label, "FAIL", repr(e))
                continue
            if getattr(runner, "providers", None):
                rep.add("runner", label, "INFO", f"providers {runner.providers}, look-ahead {runner.lookahead} frames")
            if not a.skip_quality and not data_missing and not ref_missing:
                mres[f"quality_{bname}"] = run_quality(rep, root, m, onnx, runner, label)
            if not a.skip_robustness and not data_missing and bname == "ORT":
                run_robustness(rep, root, m, runner, fallback)
            if not a.skip_timing and not data_missing:
                if rt_info is None:
                    import danc_jetson_rt as rt
                    rt_info = rt.apply_rt_settings(rt.pick_rt_cpus(a.cpus), a.rt_priority or None)
                    rep.add("timing", "real-time settings", "INFO", f"cpus={rt_info['cpus']} sched={rt_info['sched']}")
                mres[f"timing_{bname}"] = run_timing(rep, root, m, runner, a.hops, label, a.strict_timing, fallback)
        results["models"][m] = mres

    c = rep.counts()
    verdict = "PASS" if c["FAIL"] == 0 else "FAIL"
    print(f"\nSUMMARY: {c['PASS']} PASS, {c['FAIL']} FAIL, {c['WARN']} WARN, {c['INFO']} INFO "
          f"in {time.time() - t_start:.0f} s  ->  {verdict}")
    if a.json:
        out = {"verdict": verdict, "counts": c, "root": str(root), "host": platform.platform(),
               "python": platform.python_version(), "tolerances": TOL,
               "reference_systems": {m: ref_systems(root, m) for m in models},
               "selected_ids": {s: select_ids(root, s) for s in SETS}, "checks": rep.items, **results}
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"report written to {a.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit skips interpreter teardown: on macOS, ONNX Runtime's static destructors can abort at exit
    # ("libc++abi ... recursive_mutex lock failed"), which would turn a PASS into exit status 134
    os._exit(_code)
