#!/usr/bin/env bash
# Run the final GPU jobs strictly sequentially (memory-safe on the 16 GB development Mac).
cd "$(dirname "$0")/.." && source .venv/bin/activate
while pgrep -f "train_v2_ll.yaml" >/dev/null; do sleep 20; done
echo "[chain] LL training finished $(date)"
python -u -m danc.train.finetune_metric --config configs/finetune_hq_metric.yaml > logs/hq_metric.out 2>&1
echo "[chain] HQ fine-tune finished $(date)"
python -u -m danc.train.finetune_metric --config configs/finetune_ll_metric.yaml > logs/ll_metric.out 2>&1
echo "[chain] LL fine-tune finished $(date)"
