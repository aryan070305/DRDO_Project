#!/usr/bin/env bash
# Build TensorRT engines for the single-frame streaming DANCNet on a Jetson AGX Orin.
# NOT VERIFIED ON HARDWARE IN THIS PROJECT (developed on macOS where TensorRT is unavailable).
#
# Usage:  ./build_trt_engine.sh [path/to/dancnet_v3hq_cont.onnx] [out_dir]
#
# Notes
#  * Shapes are static (batch 1, one 10 ms frame, explicit recurrent-state tensors), so no
#    optimisation profiles are needed.
#  * GRU/LSTM layers are imported through the ONNX parser into TensorRT loop constructs;
#    TensorRT documents that loops support only FP32/FP16 (no INT8 inside loops), so INT8 is
#    not offered here.  An INT8 path would need calibration AND a full PESQ/STOI re-validation.
#  * --useCudaGraph reduces per-kernel launch overhead, which dominates a tiny batch-1 network.
#  * Build the engine ON the target device (engines are not portable across TensorRT versions/GPUs).
set -euo pipefail

ONNX="${1:-../../exports/dancnet_v3hq_cont.onnx}"
OUT="${2:-./engines}"
NAME="$(basename "$ONNX" .onnx)"      # e.g. dancnet_v3hq_cont (HQ, 40 ms) or dancnet_ll (LL, 20 ms)
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$OUT"

if [ ! -x "$TRTEXEC" ]; then
  echo "trtexec not found at $TRTEXEC (set TRTEXEC=...). It ships with JetPack's TensorRT." >&2
  exit 1
fi

for PREC in fp32 fp16; do
  FLAG=""
  [ "$PREC" = "fp16" ] && FLAG="--fp16"
  echo "=== building $PREC engine ==="
  "$TRTEXEC" --onnx="$ONNX" $FLAG \
      --saveEngine="$OUT/${NAME}_${PREC}.engine" \
      --useCudaGraph --noDataTransfers --separateProfileRun \
      --dumpProfile --exportProfile="$OUT/profile_${PREC}.json" \
      --exportTimes="$OUT/times_${PREC}.json" \
      --iterations=3000 --warmUp=500 --avgRuns=100 \
      2>&1 | tee "$OUT/build_${PREC}.log"
done

echo "Engines in $OUT.  Validate numerically against ONNX Runtime with:"
echo "  python3 trt_runner.py --engine $OUT/${NAME}_fp16.engine --onnx $ONNX --compare --json $OUT/bench_fp16.json"
