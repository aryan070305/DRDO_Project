"""Run verify_install.py with every package that is in the Mac venv but NOT in requirements-jetson.txt blocked.

Round-1 review harness (2026-10-04): the Jetson-like self-check quoted in deploy/jetson/QUICKSTART.md step 5 was run with it
(logs/verify_install_round1_review_jetsonlike.out). Usage:  python scripts/verify_install_jetsonlike.py [verify_install options]
"""
import sys, runpy
BLOCK = {"torch", "torchaudio", "torchvision", "librosa", "numba", "llvmlite", "matplotlib", "pyroomacoustics",
         "sklearn", "onnx", "tqdm", "pyarrow", "requests", "jinja2", "tabulate", "pooch", "audioread", "joblib",
         "networkx", "pypdf", "PIL", "kiwisolver", "fontTools", "contourpy", "cycler", "threadpoolctl", "msgpack",
         "lazy_loader", "decorator", "Cython", "pytest", "sympy_unused_marker", "fsspec", "filelock", "ml_dtypes"}
seen = set()
class Blocker:
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in BLOCK:
            seen.add(top)
            raise ModuleNotFoundError(f"No module named '{name}' (blocked: not in requirements-jetson.txt)", name=name)
        return None
sys.meta_path.insert(0, Blocker())
root = str(__import__("pathlib").Path(__file__).resolve().parents[1])   # project root (this file is in scripts/)
sys.argv = [root + "/deploy/jetson/verify_install.py"] + sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    print("blocked import attempts:", sorted(seen), file=sys.stderr)
