#!/usr/bin/env bash
# Round-2d: after round-2c (two-microphone network) finishes, on an otherwise idle machine:
#   1. latency benchmark of every exported model (criterion 4 of v3_decision_rule.md / two_mic_decision_rule.md)
#   2. reference-microphone failure test of the two-microphone network (criterion 3 of two_mic_decision_rule.md)
#   3. warm-restart continuation of v3-HQ on the M4 GPU (rule: reports/results/v3hq_cont_decision_rule.md)
#   4. evaluation of the continuation exactly like v3hq (defence, dual-mic, VoiceBank+DEMAND, transparency, edge)
cd "$(dirname "$0")/.." && source .venv/bin/activate
while pgrep -f "run_round2c_chain.sh" >/dev/null; do sleep 60; done
echo "[chain2d] round-2c finished $(date)"
sleep 30
python -u scripts/benchmark_latency.py --frames 6000 --models hq,hqft,ll,pilot,v3hq,v3ll,v3_2mic --tag round2 > logs/latency_round2.out 2>&1
echo "[chain2d] latency benchmark finished $(date)"
if [ -f exports/dancnet_v3_2mic.onnx ]; then
  python -u -m danc.eval.evaluate --set dualmic_v1 --systems dnn2_refdead:v3_2mic --save_audio none --workers 4 > logs/eval_dualmic_refdead.out 2>&1
fi
echo "[chain2d] reference-failure evaluation finished $(date)"
python -u -m danc.train.train --config configs/train_v3_hq_cont.yaml > logs/v3hq_cont.out 2>&1
echo "[chain2d] v3hq_cont finished $(date)"
python -m danc.inference.export_onnx --ckpt checkpoints/v3hq_cont/best.pt --out exports/dancnet_v3hq_cont.onnx > logs/export_v3hq_cont.out 2>&1
python -u -m danc.eval.evaluate --set defence_v1 --systems dancnet:v3hq_cont --save_audio dancnet:v3hq_cont --workers 4 > logs/eval_defence_v3hq_cont.out 2>&1
python -u -m danc.eval.evaluate --set vbdemand --systems dancnet:v3hq_cont --save_audio none --workers 4 > logs/eval_vbd_v3hq_cont.out 2>&1
python -u scripts/transparency_test.py --runs v3hq_cont > logs/transparency_v3hq_cont.out 2>&1
python -u -m danc.eval.evaluate --set dualmic_v1 --systems dnn:v3hq_cont,hybrid:v3hq_cont --save_audio hybrid:v3hq_cont --workers 4 > logs/eval_dualmic_v3hq_cont.out 2>&1
python -u scripts/eval_edge.py --systems dancnet:v3hq_cont --workers 4 > logs/eval_edge_v3hq_cont.out 2>&1
echo "[chain2d] all finished $(date)"
