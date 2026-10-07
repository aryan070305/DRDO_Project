"""Write exports/DEPLOYMENT.json (deployed models + contract) and reports/MANIFEST.sha256 (every project file).

    python scripts/make_manifest.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git", "third_party/dfn_pkgs"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import onnxruntime as ort

    sel = ROOT / "exports" / "deployment_selection.json"   # the single place where the deployed models are chosen
    deployed = json.loads(sel.read_text())
    old = ROOT / "exports" / "DEPLOYMENT.json"
    if old.exists():   # keep fields added by other tools (e.g. reference_systems used by deploy/jetson/verify_install.py)
        prev = json.loads(old.read_text()).get("deployed", {})
        for k, v in deployed.items():
            for kk, vv in prev.get(k, {}).items():
                if kk not in v and kk not in ("metadata", "inputs", "outputs", "sha256_onnx", "sha256_checkpoint"):
                    v[kk] = vv
    for k, v in deployed.items():
        sess = ort.InferenceSession(str(ROOT / v["onnx"]), providers=["CPUExecutionProvider"])
        v["metadata"] = sess.get_modelmeta().custom_metadata_map
        v["inputs"] = {i.name: i.shape for i in sess.get_inputs()}
        v["outputs"] = [o.name for o in sess.get_outputs()]
        v["sha256_onnx"] = sha256(ROOT / v["onnx"])
        v["sha256_checkpoint"] = sha256(ROOT / v["checkpoint"])
    eng = {"hybrid_engine_defaults": "danc.inference.engine.EngineConfig (lms_mu=0.1, lms_taps=2, gate_power=4.0, "
                                     "limiter -1 dBFS) - coupling tuned on data/testsets/dualmic_val",
           "stft": "16 kHz, 20 ms sqrt-Hann window, 10 ms hop, n_fft 320 (danc.dsp.DEFAULT)"}
    (ROOT / "exports" / "DEPLOYMENT.json").write_text(json.dumps({"deployed": deployed, **eng}, indent=1))
    lines = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if not p.is_file() or any(rel == d or rel.startswith(d + "/") or f"/{d}/" in f"/{rel}" for d in SKIP_DIRS):
            continue
        if rel.endswith(".lock") or rel.endswith(".DS_Store") or rel == "reports/MANIFEST.sha256":   # Finder rewrites .DS_Store
            continue
        lines.append(f"{sha256(p)}  {rel}")
    (ROOT / "reports" / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(f"DEPLOYMENT.json written; MANIFEST.sha256 lists {len(lines)} files (verify with: shasum -a 256 -c reports/MANIFEST.sha256)")


if __name__ == "__main__":
    main()
