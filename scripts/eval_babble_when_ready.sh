#!/usr/bin/env bash
# Dual-mic crew-babble set: evaluate the two-microphone network once round-2d has finished its idle latency benchmark
# (so the benchmark is not disturbed), and the v3-HQ continuation once round-2d is completely done.
cd "$(dirname "$0")/.." && source .venv/bin/activate
until grep -q "latency benchmark finished" logs/chain2d.out 2>/dev/null; do sleep 60; done
if [ -f exports/dancnet_v3_2mic.onnx ]; then
  python -u -m danc.eval.evaluate --set dualmic_babble_v1 --systems dnn2:v3_2mic --save_audio dnn2:v3_2mic --workers 2 > logs/eval_babble_2mic.out 2>&1
fi
echo "[babble] two-mic evaluation finished $(date)"
until grep -q "all finished" logs/chain2d.out 2>/dev/null; do sleep 120; done
[ -f exports/dancnet_v3hq_cont.onnx ] && python -u -m danc.eval.evaluate --set dualmic_babble_v1 --systems dnn:v3hq_cont,hybrid:v3hq_cont --save_audio none --workers 4 > logs/eval_babble_v3hq_cont.out 2>&1
echo "[babble] all finished $(date)"
