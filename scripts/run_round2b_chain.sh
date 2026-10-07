#!/usr/bin/env bash
# Round-2b GPU queue (replaces the remaining stages of run_round2_chain.sh):
#   wait for v3_2mic -> clean-GPU batch benchmark (choose_batch.py) -> v3 low-latency -> 6 ablations -> evaluation.
# Strictly sequential: nothing else may use the GPU while the benchmark runs.
cd "$(dirname "$0")/.." && source .venv/bin/activate
while pgrep -f "train_v3_2mic.yaml" >/dev/null; do sleep 30; done
echo "[chain2b] v3_2mic finished $(date)"
eval "$(python -u scripts/choose_batch.py | tee logs/choose_batch.out | grep -E '^(LL_CONFIG|ABL_DIR|ABL_SUFFIX)=')"
echo "[chain2b] batch benchmark done: LL=$LL_CONFIG ABL=$ABL_DIR $(date)"
python -u -m danc.train.train --config "$LL_CONFIG" > logs/v3ll.out 2>&1
echo "[chain2b] v3ll finished $(date)"
SYS=""
for arm in base ema crm la2 la4 nopmsqe; do
  python -u -m danc.train.train --config "$ABL_DIR/abl_$arm.yaml" > logs/abl_$arm.out 2>&1
  echo "[chain2b] abl_$arm finished $(date)"
  SYS="$SYS,dancnet:abl_${arm}${ABL_SUFFIX}"
done
python -u -m danc.eval.evaluate --set defence_v1 --systems "${SYS#,}" --save_audio none --workers 4 > logs/eval_ablations.out 2>&1
echo "[chain2b] ablation evaluation finished $(date)"
