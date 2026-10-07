"""Fast checks of the Jetson deployment kit (deploy/jetson/): scripts parse, the ONNX contract check passes for
the single-mic deployed models (HQ, LL), the torch-free runner matches the reference runner, and the safety wrapper behaves.
The full self-check (PESQ/STOI on 16 scenes, timing) is `python deploy/jetson/verify_install.py`."""
import csv
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "deploy" / "jetson"
DEPLOY = json.loads((ROOT / "exports" / "DEPLOYMENT.json").read_text())["deployed"]
# The public (licence-safe) package leaves out the trained models and the test-set mixtures: tests that need them skip
MODELS_PRESENT = all((ROOT / v["onnx"]).exists() for v in DEPLOY.values())
MIXTURES_PRESENT = all((ROOT / "data" / "testsets" / s).is_dir() for s in ("defence_v1/noisy", "dualmic_v1/primary"))
needs_models = pytest.mark.skipif(not MODELS_PRESENT, reason="deployed ONNX models not in this copy (public package)")
needs_mixtures = pytest.mark.skipif(not MIXTURES_PRESENT, reason="test-set mixtures not in this copy (public package)")


def _load(name):
    if str(J) not in sys.path:
        sys.path.insert(0, str(J))
    spec = importlib.util.spec_from_file_location(name, J / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return sys.modules[name]


vi = _load("verify_install")
rt = _load("danc_jetson_rt")


# ------------------------------------------------------------------------------------------ scripts parse
@pytest.mark.parametrize("script", ["install.sh", "package_for_jetson.sh"])
def test_new_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(J / script)], check=True)


@pytest.mark.parametrize("script", ["verify_install.py", "danc_jetson_rt.py", "danc_live.py", "check_channels.py"])
def test_new_python_scripts_compile(script, tmp_path):
    py_compile.compile(str(J / script), cfile=str(tmp_path / (script + "c")), doraise=True)


def test_requirements_are_torch_free():
    req = (J / "requirements-jetson.txt").read_text().lower()
    active = [ln.split("#")[0].strip() for ln in req.splitlines() if ln.split("#")[0].strip()]
    assert not any(ln.startswith(("torch", "torchaudio")) for ln in active)
    for pkg in ("numpy", "scipy", "soundfile", "sounddevice", "soxr", "onnxruntime", "pyyaml", "pandas", "pesq", "pystoi"):
        assert any(ln.startswith(pkg) for ln in active), pkg


# ------------------------------------------------------------------------------------------ ONNX contract
@needs_models
@pytest.mark.parametrize("model", ["HQ", "LL"])
def test_contract_check_passes_for_deployed_models(model):
    spec = DEPLOY[model]
    res = vi.check_contract(ROOT / spec["onnx"], spec)
    failed = [(n, d) for n, ok, d in res if not ok]
    assert not failed, failed
    names = " ".join(n for n, _, _ in res)
    for key in ("sha256", "input names", "output names", "number of recurrent states", "look-ahead", "zero-input"):
        assert key in names
@needs_models
def test_contract_check_detects_a_wrong_manifest():
    spec = json.loads(json.dumps(DEPLOY["HQ"]))
    spec["sha256_onnx"] = "0" * 64
    spec["metadata"]["lookahead"] = "0"
    res = {n: ok for n, ok, _ in vi.check_contract(ROOT / spec["onnx"], spec)}
    assert res["sha256 matches DEPLOYMENT.json"] is False
    assert res["metadata == DEPLOYMENT.json"] is False


# ------------------------------------------------------------------------------------------ file selection
@needs_models
@needs_mixtures
def test_selected_files_exist_and_reference_rows_complete(tmp_path):
    files = vi.needed_files(ROOT)
    assert len(files) == 2 + 8 * 3 + 8 * 2
    assert all((ROOT / f).exists() for f in files)
    vi.write_reference_subset(ROOT, tmp_path)
    models = list(vi.load_deployment(ROOT)["deployed"])
    for s in vi.SETS:
        rows = list(csv.DictReader(open(tmp_path / "reports" / "results" / s / "per_file.csv")))
        systems = {vi.ref_systems(ROOT, m)[s] for m in models if s in vi.ref_systems(ROOT, m)}
        assert len(rows) == 8 * len(systems), s                 # 8 files x each deployed model's reference system


# ------------------------------------------------------------------------------------------ runners / engine
@needs_models
def test_torchfree_runner_matches_reference_runner():
    pytest.importorskip("torch")
    from danc.inference.runners import ORTStepRunner as RefRunner

    onnx = str(ROOT / DEPLOY["HQ"]["onnx"])
    a, b = RefRunner(onnx), rt.ORTStepRunner(onnx)
    assert a.lookahead == b.lookahead == 2
    rng = np.random.default_rng(0)
    for _ in range(30):
        x = (0.1 * rng.standard_normal((1, 1, 161, 2))).astype(np.float32)
        (ya, sa), (yb, sb) = a(x), b(x)
        assert np.array_equal(ya, yb) and np.array_equal(sa, sb)
@needs_models
def test_safe_engine_identical_on_finite_input_and_guards_nan():
    from danc.inference.engine import EngineConfig, HybridEngine

    onnx = str(ROOT / DEPLOY["LL"]["onnx"])
    rng = np.random.default_rng(1)
    p, r = 0.05 * rng.standard_normal(160 * 60), 0.05 * rng.standard_normal(160 * 60)
    # separate runners: a runner carries the recurrent state
    plain, safe = HybridEngine(rt.ORTStepRunner(onnx), EngineConfig()), rt.SafeHybridEngine(rt.ORTStepRunner(onnx), EngineConfig())
    ya = np.concatenate([plain.process_block(p[k * 160:(k + 1) * 160], r[k * 160:(k + 1) * 160]) for k in range(60)])
    yb = np.concatenate([safe.process_block(p[k * 160:(k + 1) * 160], r[k * 160:(k + 1) * 160]) for k in range(60)])
    assert np.array_equal(ya, yb)
    safe.reset()
    bad = p.copy()
    bad[160 * 10:160 * 12] = np.nan
    bad[160 * 20] = np.inf
    with np.errstate(all="ignore"):
        y = np.concatenate([safe.process_block(bad[k * 160:(k + 1) * 160], None) for k in range(60)])
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0 and safe.sanitised_blocks >= 3
    safe.timing = [0.0005] * 100_000                  # long run: history must be trimmed
    safe.process_block(p[:160], r[:160])
    assert len(safe.timing) <= 2 * safe.TIMING_WINDOW


def test_safe_engine_survives_long_run_with_engine_timing_deque():
    """Regression: the engine keeps its timings in a bounded deque (round 2); the Jetson wrapper's trim used slice
    deletion, which a deque does not support, so the live runtime raised TypeError after 3 000 blocks (30 s)."""
    from danc.inference.engine import EngineConfig

    safe = rt.SafeHybridEngine(None, EngineConfig(use_lms=False, limiter_dbfs=None))
    x = np.zeros(160)
    for _ in range(2 * safe.TIMING_WINDOW + 200):
        safe.process_block(x, x)
    assert len(safe.timing) <= 2 * safe.TIMING_WINDOW
    assert list(safe.timing)[-1] >= 0.0


def test_pick_rt_cpus():
    online = rt.online_cpus()
    auto = rt.pick_rt_cpus("auto")
    assert auto and set(auto) <= set(online)
    assert rt.pick_rt_cpus("none") is None and rt.pick_rt_cpus(None) is None
    assert set(rt.pick_rt_cpus("9999")) <= set(online)        # offline CPU -> falls back, never raises
@needs_models
@needs_mixtures
def test_live_launcher_runs_without_torch(tmp_path):
    import soundfile as sf

    d = ROOT / "data" / "testsets" / "dualmic_v1"
    i = vi.select_ids(ROOT, "dualmic_v1")[0]
    p, _ = sf.read(d / "primary" / f"{i}.flac")
    r, _ = sf.read(d / "reference" / f"{i}.flac")
    sf.write(tmp_path / "p.wav", p[:8000], 16000)
    sf.write(tmp_path / "r.wav", r[:8000], 16000)
    code = (
        "import sys, runpy\n"
        "class B:\n"
        "    def find_spec(self, n, p=None, t=None):\n"
        "        if n.split('.')[0] == 'torch': raise ModuleNotFoundError(n, name=n)\n"
        "sys.meta_path.insert(0, B())\n"
        f"sys.argv = ['danc_live.py', '--onnx', {str(ROOT / DEPLOY['LL']['onnx'])!r}, '--simulate', {str(tmp_path / 'p.wav')!r},"
        f" '--simulate-ref', {str(tmp_path / 'r.wav')!r}, '--out', {str(tmp_path / 'o.wav')!r}]\n"
        "try:\n"
        f"    runpy.run_path({str(J / 'danc_live.py')!r}, run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    assert not e.code, e.code\n"
        "assert 'torch' not in sys.modules\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src"))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=str(ROOT), capture_output=True, timeout=120)
    y, fs = sf.read(tmp_path / "o.wav")
    assert fs == 16000 and len(y) > 0 and np.all(np.isfinite(y))


def test_check_channels_detects_swap():
    cc = _load("check_channels")
    rng = np.random.default_rng(2)
    fs, n = 16000, 16000 * 4
    env = (np.sin(2 * np.pi * 2 * np.arange(n) / fs) > 0).astype(float)      # 0.25 s talk / pause bursts
    speech = 0.1 * env * rng.standard_normal(n)
    noise = 0.003 * rng.standard_normal((n, 2))
    x = np.stack([speech, 0.1 * speech], 1) + noise                          # reference hears speech 20 dB lower
    assert cc.analyse(x, fs)["verdict"] == "PASS"
    res = cc.analyse(x[:, ::-1], fs)
    assert res["verdict"] == "FAIL" and "SWAPPED" in " ".join(res["problems"])


# ------------------------------------------------------------------------------------------ docs / units
def test_quickstart_and_unit_reference_existing_files():
    q = (J / "QUICKSTART.md").read_text()
    assert "NOT VERIFIED ON HARDWARE" in q
    import re

    for name in set(re.findall(r"deploy/jetson/([A-Za-z0-9_.-]+\.(?:py|sh|service|txt|md))", q)):
        assert (J / name).exists(), name
    unit = (J / "danc-live.service").read_text()
    assert "/opt/danc/deploy/jetson/danc_live.py" in unit and "--cpus" in unit
    readme = (J / "README.md").read_text()
    assert readme.startswith("> ## Start here") and "# D-ANC on NVIDIA Jetson AGX Orin 64GB" in readme


def test_package_script_refuses_unsafe_output_dirs(tmp_path):
    """The bundle builder must never write into the project tree (except dist/) or a foreign directory."""
    script = str(J / "package_for_jetson.sh")
    nonempty = tmp_path / "foreign"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("keep")
    for out in (".", "src/should_not_exist", "exports/should_not_exist", "dist", "..", str(nonempty)):
        r = subprocess.run(["bash", script, "--out", out], cwd=str(ROOT), capture_output=True, text=True)
        assert r.returncode != 0 and "ERROR" in r.stderr, out
    assert not (ROOT / "src" / "should_not_exist").exists() and not (ROOT / "exports" / "should_not_exist").exists()
    assert (nonempty / "keep.txt").read_text() == "keep" and len(list(nonempty.iterdir())) == 1


# ------------------------------------------------------------------------------------------ added by the review
@needs_models
def test_torchfree_runner_supports_two_mic_models(tmp_path):
    """The shim must follow danc.inference.runners for two-microphone models (ONNX input 'spec_ref')."""
    pytest.importorskip("torch")
    import torch
    from danc.inference.export_onnx import export
    from danc.inference.runners import ORTStepRunner as RefRunner
    from danc.models.dancnet import DANCNet, DANCNetConfig

    torch.manual_seed(3)
    m = DANCNet(DANCNetConfig(in_ch=2, df_order=3, lookahead=0)).eval()
    f = tmp_path / "two_mic.onnx"
    export(m, str(f))
    a, b = RefRunner(str(f)), rt.ORTStepRunner(str(f))
    assert a.n_mics == b.n_mics == 2 and rt.onnx_n_mics(str(f)) == 2
    rng = np.random.default_rng(5)
    for _ in range(10):
        x = (0.1 * rng.standard_normal((1, 1, 161, 2))).astype(np.float32)
        r = (0.1 * rng.standard_normal((1, 1, 161, 2))).astype(np.float32)
        (ya, sa), (yb, sb) = a(x, r), b(x, r)
        assert np.array_equal(ya, yb) and np.array_equal(sa, sb)
    with pytest.raises(NotImplementedError):          # trt_runner.py feeds only 'spec'
        rt.make_trt_runner("unused.engine", str(f))
    assert rt.ORTStepRunner(str(ROOT / DEPLOY["HQ"]["onnx"])).n_mics == 1


def test_stall_watchdog():
    import time

    fired = []
    rt.start_watchdog(timeout_s=0.3, startup_s=5.0, on_stall=fired.append, poll_s=0.05)
    t_end = time.monotonic() + 0.6
    while time.monotonic() < t_end:                      # blocks arriving: must not fire
        rt.HEARTBEAT["t"] = time.monotonic()
        time.sleep(0.02)
    assert not fired
    time.sleep(0.8)                                      # blocks stop: must fire once
    assert len(fired) == 1 and "no audio block for" in fired[0]
    fired.clear()
    rt.start_watchdog(timeout_s=0.3, startup_s=0.2, on_stall=fired.append, poll_s=0.05)
    time.sleep(0.6)                                      # no block at all after start
    assert len(fired) == 1 and "within" in fired[0]
@needs_models
def test_safe_engine_updates_heartbeat():
    from danc.inference.engine import EngineConfig

    rt.HEARTBEAT["t"], n0 = None, rt.HEARTBEAT["blocks"]
    e = rt.SafeHybridEngine(rt.ORTStepRunner(str(ROOT / DEPLOY["LL"]["onnx"])), EngineConfig())
    e.process_block(np.zeros(160), np.zeros(160))
    assert rt.HEARTBEAT["t"] is not None and rt.HEARTBEAT["blocks"] == n0 + 1


def test_reference_systems_override(tmp_path):
    dep = json.loads((ROOT / "exports" / "DEPLOYMENT.json").read_text())
    # current manifest: built-in mapping, overridden by the entry's "reference_systems" where present
    assert vi.ref_systems(ROOT, "HQ") == {**vi.REF_SYSTEMS["HQ"], **(dep["deployed"]["HQ"].get("reference_systems") or {})}
    dep["deployed"]["HQ"]["reference_systems"] = {"dualmic_v1": "hybrid:v3hq", "defence_v1": "dancnet:v3hq"}
    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "DEPLOYMENT.json").write_text(json.dumps(dep))
    assert vi.ref_systems(tmp_path, "HQ") == {"dualmic_v1": "hybrid:v3hq", "defence_v1": "dancnet:v3hq"}
    assert vi.ref_systems(tmp_path, "LL") == {**vi.REF_SYSTEMS["LL"], **(dep["deployed"]["LL"].get("reference_systems") or {})}
