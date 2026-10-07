import py_compile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "deploy" / "jetson"


def test_python_scripts_compile():
    for f in ("trt_runner.py", "measure_latency.py"):
        py_compile.compile(str(J / f), doraise=True)


def test_shell_scripts_parse():
    for f in ("build_trt_engine.sh", "setup_audio.sh", "log_power.sh"):
        subprocess.run(["bash", "-n", str(J / f)], check=True)
