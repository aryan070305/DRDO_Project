#!/usr/bin/env bash
# Evaluate the v3 low-latency model as soon as round-2b has trained it (CPU, low worker count).
cd "$(dirname "$0")/.." && source .venv/bin/activate
until grep -q "v3ll finished" logs/chain2b.out 2>/dev/null; do sleep 30; done
python -m danc.inference.export_onnx --ckpt checkpoints/v3ll/best.pt --out exports/dancnet_v3ll.onnx > logs/export_v3ll.out 2>&1
python -u -m danc.eval.evaluate --set defence_v1 --systems dancnet:v3ll --save_audio dancnet:v3ll --workers 2 > logs/eval_defence_v3ll.out 2>&1
python -u -m danc.eval.evaluate --set dualmic_v1 --systems dnn:v3ll,hybrid:v3ll --save_audio hybrid:v3ll --workers 2 > logs/eval_dualmic_v3ll.out 2>&1
python -u -m danc.eval.evaluate --set vbdemand --systems dancnet:v3ll --save_audio none --workers 2 > logs/eval_vbd_v3ll.out 2>&1
python -u scripts/transparency_test.py --runs v3ll > logs/transparency_v3ll.out 2>&1
python -u scripts/eval_edge.py --systems dancnet:v3ll --workers 2 > logs/eval_edge_v3ll.out 2>&1
echo "v3ll evaluation done $(date)" > logs/eval_v3ll_chain.done
