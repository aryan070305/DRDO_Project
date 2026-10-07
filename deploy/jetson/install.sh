#!/usr/bin/env bash
# D-ANC runtime installer for NVIDIA Jetson AGX Orin with JetPack 6.2.x (Ubuntu 22.04, Python 3.10).
# NOT VERIFIED ON HARDWARE IN THIS PROJECT (development and testing were on macOS).
#
# * Idempotent: safe to re-run; it only adds what is missing (apt packages, venv, pip packages).
# * Does NOT install PyTorch: the Jetson runs the exported ONNX models (exports/*.onnx) with ONNX Runtime.
# * Never deletes project files.  The only removal it can do is 'pip uninstall onnxruntime' inside the venv,
#   and only with --gpu (the CPU and GPU wheels both provide the 'onnxruntime' module and must not coexist).
#
# Usage:   deploy/jetson/install.sh [VENV_DIR] [options]
#   VENV_DIR              venv to create / reuse (default /opt/danc/.venv)
#   --venv DIR            same as the positional argument
#   --python PY           interpreter for the venv (default: python3, the JetPack system Python 3.10)
#   --gpu                 onnxruntime-gpu (CUDA + TensorRT execution providers) from the Jetson AI Lab index
#                         instead of the CPU wheel from PyPI
#   --index URL           Jetson AI Lab pip index (default https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/)
#   --with-trt            TensorRT path for trt_runner.py: venv with --system-site-packages (TensorRT's Python
#                         bindings come from apt python3-libnvinfer), plus cuda-python 12.6 from the index
#   --offline-wheels DIR  air-gapped install: pip --no-index --find-links DIR (build DIR on a connected Jetson
#                         with: pip wheel -r deploy/jetson/requirements-jetson.txt -w DIR)
#   --no-apt              skip apt-get (packages already present / no network)
#   --no-pesq             skip pesq + pystoi (verification metrics; verify_install.py quality check then FAILs)
#   -h, --help            this text
#
# Example: sudo mkdir -p /opt/danc && sudo chown "$USER": /opt/danc && cp -r <bundle>/. /opt/danc/
#          cd /opt/danc && ./deploy/jetson/install.sh
set -euo pipefail

log()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQ="$SCRIPT_DIR/requirements-jetson.txt"
VENV="/opt/danc/.venv"
PY="python3"
GPU=0
TRT=0
APT=1
PESQ=1
WHEELS=""
INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv)           VENV="${2:?--venv needs a path}"; shift 2 ;;
    --python)         PY="${2:?--python needs an interpreter}"; shift 2 ;;
    --gpu)            GPU=1; shift ;;
    --index)          INDEX="${2:?--index needs a URL}"; shift 2 ;;
    --with-trt)       TRT=1; shift ;;
    --offline-wheels) WHEELS="${2:?--offline-wheels needs a directory}"; shift 2 ;;
    --no-apt)         APT=0; shift ;;
    --no-pesq)        PESQ=0; shift ;;
    -h|--help)        usage; exit 0 ;;
    -*)               die "unknown option $1 (see --help)" ;;
    *)                VENV="$1"; shift ;;
  esac
done

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

# ------------------------------------------------------------------------------------------- 0. platform
log "0/8 Platform checks"
uname -a
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
  warn "architecture is $ARCH, not aarch64: this installer targets the Jetson (continuing anyway)"
fi
if [ -f /etc/nv_tegra_release ]; then
  echo "Jetson Linux: $(head -n 1 /etc/nv_tegra_release)"
  echo "  (JetPack 6.2 = R36 REVISION 4.3, 6.2.1 = R36 REVISION 4.4, 6.2.2 = R36 REVISION 5.x)"
else
  warn "/etc/nv_tegra_release not found - this does not look like a Jetson"
fi
command -v "$PY" >/dev/null 2>&1 || die "$PY not found (sudo apt-get install python3)"
PYV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python >= 3.10 needed, $PY is $PYV"
echo "Python for the venv: $PY ($PYV)"
[ -f "$PROJECT/pyproject.toml" ] && [ -d "$PROJECT/src/danc" ] || die "D-ANC project not found at $PROJECT"
[ -f "$REQ" ] || die "missing $REQ"
[ -f "$PROJECT/exports/DEPLOYMENT.json" ] || warn "exports/DEPLOYMENT.json missing: copy exports/ (models) too"
[ -w "$PROJECT" ] || warn "$PROJECT is not writable by $(id -un); 'pip install -e' needs it: sudo chown -R $(id -un): $PROJECT"

# ------------------------------------------------------------------------------------------- 1. apt
APT_PKGS=(python3-venv python3-pip python3-dev build-essential libsndfile1 libportaudio2 alsa-utils)
if [ "$TRT" = 1 ]; then APT_PKGS+=(python3-libnvinfer); fi
if [ "$APT" = 1 ]; then
  log "1/8 System packages (apt)"
  if command -v apt-get >/dev/null 2>&1 && command -v dpkg-query >/dev/null 2>&1; then
    MISSING=()
    for p in "${APT_PKGS[@]}"; do
      if ! dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed"; then MISSING+=("$p"); fi
    done
    if [ "${#MISSING[@]}" -gt 0 ]; then
      echo "installing: ${MISSING[*]}"
      $SUDO apt-get update
      $SUDO apt-get install -y "${MISSING[@]}"
    else
      echo "all present: ${APT_PKGS[*]}"
    fi
  else
    warn "apt-get not available: make sure these are installed: ${APT_PKGS[*]}"
  fi
else
  log "1/8 System packages: skipped (--no-apt); required: ${APT_PKGS[*]}"
fi

# ------------------------------------------------------------------------------------------- 2. venv
log "2/8 Python venv at $VENV"
if [ -x "$VENV/bin/python" ]; then
  echo "venv exists - reusing it"
  if [ "$TRT" = 1 ] && ! grep -q "include-system-site-packages = true" "$VENV/pyvenv.cfg" 2>/dev/null; then
    warn "this venv was created WITHOUT --system-site-packages, so the apt TensorRT bindings are not visible in it."
    warn "use a separate venv for the TensorRT path, e.g.: $0 ${VENV}-trt --with-trt"
  fi
else
  PARENT="$(dirname "$VENV")"
  [ -d "$PARENT" ] || $SUDO mkdir -p "$PARENT"
  [ -w "$PARENT" ] || $SUDO chown "$(id -u):$(id -g)" "$PARENT"
  VENV_ARGS=()
  if [ "$TRT" = 1 ]; then VENV_ARGS+=(--system-site-packages); fi
  "$PY" -m venv ${VENV_ARGS[@]+"${VENV_ARGS[@]}"} "$VENV"
  echo "created $VENV"
fi
VPY="$VENV/bin/python"
PIP=("$VPY" -m pip)
SRC=()
if [ -n "$WHEELS" ]; then
  [ -d "$WHEELS" ] || die "--offline-wheels: $WHEELS is not a directory"
  SRC=(--no-index --find-links "$WHEELS")
fi
pipi() { "${PIP[@]}" install --disable-pip-version-check ${SRC[@]+"${SRC[@]}"} "$@"; }
req_line() { sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$REQ" | grep -iE "^$1([<>=!~ ;\[]|$)" | head -n 1; }

log "3/8 pip, setuptools (>= 64 for editable installs), wheel"
pipi --upgrade pip "setuptools>=64" wheel

# ------------------------------------------------------------------------------------------- 3. requirements
log "4/8 Runtime requirements ($REQ, without torch)"
BASE=()
while IFS= read -r line; do BASE+=("$line"); done < <(
  sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$REQ" | grep -v '^[[:space:]]*$' | grep -viE '^(onnxruntime|pesq|pystoi)')
pipi "${BASE[@]}"

# ------------------------------------------------------------------------------------------- 4. onnxruntime
log "5/8 ONNX Runtime ($( [ "$GPU" = 1 ] && echo 'GPU wheel, Jetson AI Lab index' || echo 'CPU wheel, PyPI'))"
if [ "$GPU" = 1 ]; then
  if "$VPY" -m pip show onnxruntime >/dev/null 2>&1; then
    echo "removing the CPU wheel 'onnxruntime' from the venv (conflicts with onnxruntime-gpu)"
    "${PIP[@]}" uninstall -y onnxruntime
  fi
  # --index-url, not --extra-index-url: PyPI also has generic (server-ARM, "SBSA") onnxruntime-gpu aarch64 wheels
  # for Python >= 3.11 (1.29+, checked 2026-10-04); with an extra index pip would take the higher PyPI version,
  # which is not built for Jetson.  The Jetson AI Lab index redirects packages it does not host (the
  # dependencies) to pypi.org, so they still resolve.
  if [ -n "$WHEELS" ]; then pipi onnxruntime-gpu; else pipi --index-url "$INDEX" onnxruntime-gpu; fi
else
  if "$VPY" -m pip show onnxruntime-gpu >/dev/null 2>&1; then
    warn "onnxruntime-gpu is installed in this venv - keeping it (it includes the CPU execution provider)"
  else
    pipi "$(req_line onnxruntime)"
  fi
fi

# ------------------------------------------------------------------------------------------- 5. metrics
if [ "$PESQ" = 1 ]; then
  log "6/8 Verification metrics (pesq builds a C extension: takes a few minutes the first time)"
  if "$VPY" -c "import pesq" >/dev/null 2>&1; then
    echo "pesq already installed"
  else
    pipi numpy cython pytest-runner
    pipi --no-build-isolation "$(req_line pesq)"
  fi
  pipi "$(req_line pystoi)"
else
  log "6/8 Verification metrics: skipped (--no-pesq)"
fi

# ------------------------------------------------------------------------------------------- 6. TensorRT extras
if [ "$TRT" = 1 ]; then
  log "6b/8 TensorRT path (NOT VERIFIED ON HARDWARE IN THIS PROJECT)"
  "$VPY" -c "import tensorrt as t; print('tensorrt python bindings', t.__version__)" \
    || warn "tensorrt is not importable in the venv: sudo apt-get install python3-libnvinfer (venv needs --system-site-packages)"
  if [ -n "$WHEELS" ]; then pipi "cuda-python==12.6.*"; else pipi --index-url "$INDEX" "cuda-python==12.6.*"; fi
  [ -x /usr/src/tensorrt/bin/trtexec ] || warn "trtexec not found: sudo apt-get install nvidia-jetpack (or libnvinfer-bin)"
fi

# ------------------------------------------------------------------------------------------- 7. project
log "7/8 D-ANC package (editable install of $PROJECT)"
pipi --no-deps --no-build-isolation -e "$PROJECT"

# ------------------------------------------------------------------------------------------- 8. smoke test
log "8/8 Smoke test"
"$VPY" - <<'PYEOF'
import sys
import numpy
import onnxruntime as ort
import danc.dsp, danc.inference.engine  # noqa: F401  (torch-free modules)
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| onnxruntime", ort.__version__,
      "| providers", ort.get_available_providers())
try:
    import torch  # noqa: F401
    print("note: torch is importable in this venv (not needed by the Jetson runtime)")
except ImportError:
    print("torch: not installed (expected) -> use deploy/jetson/danc_live.py for the live engine")
PYEOF

cat <<EOF

D-ANC runtime installed.  Next steps (see deploy/jetson/QUICKSTART.md):
  source $VENV/bin/activate
  cd $PROJECT
  python deploy/jetson/verify_install.py --json verify_report.json      # must end with '-> PASS'
  python deploy/jetson/danc_live.py --list-devices                      # find the USB interface
  python deploy/jetson/danc_live.py --onnx exports/dancnet_v3hq_cont.onnx --in-device USB --out-device USB   # HQ, 40 ms
  python deploy/jetson/danc_live.py --onnx exports/dancnet_ll.onnx   --in-device USB --out-device USB   # LL, 20 ms
EOF
