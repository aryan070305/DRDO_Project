#!/usr/bin/env bash
# Round-2 GPU queue, strictly sequential (memory-safe on the 16 GB development Mac):
#   v3hq (already running) -> v3 two-microphone -> v3 low-latency -> 6 ablation arms -> ablation evaluation
cd "$(dirname "$0")/.." && source .venv/bin/activate
while pgrep -f "train_v3_hq.yaml" >/dev/null; do sleep 30; done
echo "[chain2] v3hq finished $(date)"
python -u -m danc.train.train_dual --config configs/train_v3_2mic.yaml > logs/v3_2mic.out 2>&1
echo "[chain2] v3_2mic finished $(date)"
python -u -m danc.train.train --config configs/train_v3_ll.yaml > logs/v3ll.out 2>&1
echo "[chain2] v3ll finished $(date)"
for arm in base ema crm la2 la4 nopmsqe; do
  python -u -m danc.train.train --config configs/ablations/abl_$arm.yaml > logs/abl_$arm.out 2>&1
  echo "[chain2] abl_$arm finished $(date)"
done
python -u -m danc.eval.evaluate --set defence_v1 --systems dancnet:abl_base,dancnet:abl_ema,dancnet:abl_crm,dancnet:abl_la2,dancnet:abl_la4,dancnet:abl_nopmsqe --save_audio none --workers 4 > logs/eval_ablations.out 2>&1
echo "[chain2] ablation evaluation finished $(date)"
