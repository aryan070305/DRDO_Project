#!/usr/bin/env bash
# Build the D-ANC deployment bundle for the Jetson.  Run it on the DEVELOPMENT machine (macOS or Linux).
#
# The bundle holds only what the Jetson needs at run time, with the same relative layout as the project, so
# every command in deploy/jetson/QUICKSTART.md works unchanged inside it:
#   src/                         the danc package (no caches)
#   exports/                     deployed ONNX models (from exports/DEPLOYMENT.json) + DEPLOYMENT.json
#   deploy/                      Jetson scripts, docs, requirements-jetson.txt, systemd units
#   pyproject.toml, requirements.txt, README.md, reports/REPORT.md
#   data/testsets/...            the 16 verification scenes (8 dual-mic + 8 defence) + clean references + meta.csv
#   reports/results/*/per_file.csv   ONLY the reference rows of those 16 files for the deployed systems
#   reports/results/latency_mac_m4.json, simulated_live_runs.json   Mac reference numbers
#   BUNDLE_INFO.txt, SHA256SUMS  provenance + checksums (verify on the Jetson with: sha256sum -c SHA256SUMS)
#
# Usage:  deploy/jetson/package_for_jetson.sh [--out DIR] [--tar] [--all-models] [--python PY]
#   --out DIR      bundle directory (default: <project>/dist/jetson_bundle)
#   --tar          also write DIR.tar.gz (DIR + .tar.gz) for copying to the Jetson
#   --all-models   copy every exports/*.onnx (+ .json sidecars), e.g. the LL-small pilot model, not only the
#                  deployed HQ/LL models
#   --python PY    python used for the file list / reference rows (stdlib only; default python3)
#
# It never deletes anything: an existing bundle directory is updated in place (files from an older bundle that
# are no longer needed stay there; use a new --out directory for a clean bundle).
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
export COPYFILE_DISABLE=1          # macOS tar: no AppleDouble ._* files in the bundle
TAR_OPTS=()
if tar --version 2>/dev/null | grep -qi bsdtar; then TAR_OPTS=(--no-xattrs); fi   # no macOS xattrs either
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$ROOT/dist/jetson_bundle"
TAR=0
ALL=0
PY="python3"

while [ $# -gt 0 ]; do
  case "$1" in
    --out)        OUT="${2:?--out needs a directory}"; shift 2 ;;
    --tar)        TAR=1; shift ;;
    --all-models) ALL=1; shift ;;
    --python)     PY="${2:?--python needs an interpreter}"; shift 2 ;;
    -h|--help)    sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            die "unknown argument $1 (see --help)" ;;
  esac
done

command -v "$PY" >/dev/null 2>&1 || die "$PY not found"
[ -f "$ROOT/exports/DEPLOYMENT.json" ] || die "$ROOT/exports/DEPLOYMENT.json not found"
# Safety: resolve the output path WITHOUT creating anything, then refuse every location where copying could
# overwrite project files.  Inside the project only dist/... is allowed; outside, never an ancestor of the
# project; an existing non-empty directory only if it is a previous bundle (has BUNDLE_INFO.txt).
realpath_py() { "$PY" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
ROOT_R="$(realpath_py "$ROOT")"
OUT_R="$(realpath_py "$OUT")"
case "$OUT_R/" in
  "$ROOT_R/")       die "--out must not be the project root" ;;
  "$ROOT_R/dist/")  die "--out must be a sub-directory of dist/, e.g. dist/jetson_bundle" ;;
  "$ROOT_R/dist/"*) ;;
  "$ROOT_R/"*)      die "--out inside the project must be under dist/ (got $OUT_R)" ;;
esac
case "$ROOT_R/" in "$OUT_R/"*) die "--out must not be a parent directory of the project (got $OUT_R)" ;; esac
if [ -d "$OUT_R" ]; then
  if [ -n "$(ls -A "$OUT_R")" ] && [ ! -f "$OUT_R/BUNDLE_INFO.txt" ]; then
    die "$OUT_R exists, is not empty and is not a previous D-ANC bundle; choose another --out"
  fi
  echo "bundle directory exists, updating in place: $OUT_R"
elif [ -e "$OUT_R" ]; then
  die "$OUT_R exists and is not a directory"
else
  mkdir -p "$OUT_R"
fi
OUT="$OUT_R"

copy_tree() {   # copy_tree <path relative to ROOT>: regular files only (no caches, build products, empty dirs)
  (cd "$ROOT" && find "$1" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '.DS_Store' ! -path '*.egg-info/*' \
      ! -name '*.engine' ! -path '*/engines/*' -print | LC_ALL=C sort | tar ${TAR_OPTS[@]+"${TAR_OPTS[@]}"} -cf - -T -) \
    | (cd "$OUT" && tar -xf -)
}
copy_file() {   # copy_file <path relative to ROOT>
  [ -f "$ROOT/$1" ] || die "missing $ROOT/$1"
  mkdir -p "$OUT/$(dirname "$1")"
  cp -p "$ROOT/$1" "$OUT/$1"
}

echo "== code, deployment scripts and docs"
copy_tree src
copy_tree deploy
for f in pyproject.toml requirements.txt README.md reports/REPORT.md; do
  if [ -f "$ROOT/$f" ]; then copy_file "$f"; fi
done

echo "== models"
copy_file exports/DEPLOYMENT.json
if [ "$ALL" = 1 ]; then
  MODELS="$(cd "$ROOT" && ls exports/*.onnx)"
else
  MODELS="$("$PY" -c 'import json,sys; print("\n".join(v["onnx"] for v in json.load(open(sys.argv[1]))["deployed"].values()))' \
            "$ROOT/exports/DEPLOYMENT.json")"
fi
for m in $MODELS; do
  copy_file "$m"
  side="${m%.onnx}.json"
  if [ -f "$ROOT/$side" ]; then copy_file "$side"; fi
  echo "   $m"
done

echo "== verification data (16 scenes) and reference rows"
N=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  copy_file "$f"
  N=$((N + 1))
done < <("$PY" "$ROOT/deploy/jetson/verify_install.py" --root "$ROOT" --list-files)
# (a failure of the file-list command inside < <(...) does not stop the script, so check the count)
[ "$N" -gt 0 ] || die "verify_install.py --list-files returned no files (missing data/testsets/*/meta.csv?)"
echo "   $N data files"
"$PY" "$ROOT/deploy/jetson/verify_install.py" --root "$ROOT" --write-refs "$OUT" | sed 's/^/   /'
for f in reports/results/latency_mac_m4.json reports/results/simulated_live_runs.json; do
  if [ -f "$ROOT/$f" ]; then copy_file "$f"; fi
done

echo "== provenance and checksums"
if command -v sha256sum >/dev/null 2>&1; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
{
  echo "D-ANC Jetson deployment bundle"
  echo "created:  $(date -u '+%Y-%m-%dT%H:%M:%SZ') on $(uname -sm) by package_for_jetson.sh"
  echo "source:   $ROOT"
  echo "status:   NOT VERIFIED ON JETSON HARDWARE IN THIS PROJECT (developed and verified on macOS / Apple M4)"
  echo
  echo "models (sha256):"
  for m in $MODELS; do (cd "$OUT" && "${SHA[@]}" "$m"); done
  echo
  echo "On the Jetson:  sha256sum -c SHA256SUMS && ./deploy/jetson/install.sh && python deploy/jetson/verify_install.py"
  echo "Full instructions: deploy/jetson/QUICKSTART.md"
} > "$OUT/BUNDLE_INFO.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name '.DS_Store' -print | LC_ALL=C sort | sed 's|^\./||' \
   | while IFS= read -r f; do "${SHA[@]}" "$f"; done) > "$OUT/SHA256SUMS"
echo "   $(wc -l < "$OUT/SHA256SUMS" | tr -d ' ') files listed in SHA256SUMS"

echo "== bundle: $OUT ($(du -sh "$OUT" | cut -f1))"
if [ "$TAR" = 1 ]; then
  TGZ="$OUT.tar.gz"
  tar ${TAR_OPTS[@]+"${TAR_OPTS[@]}"} -czf "$TGZ" -C "$(dirname "$OUT")" "$(basename "$OUT")"
  echo "== archive: $TGZ ($(du -h "$TGZ" | cut -f1)); sha256: $("${SHA[@]}" "$TGZ" | cut -d' ' -f1)"
  echo "   copy:    scp $TGZ <user>@<jetson>:/tmp/   then on the Jetson: tar -xzf /tmp/$(basename "$TGZ") -C /opt/danc --strip-components=1"
fi
