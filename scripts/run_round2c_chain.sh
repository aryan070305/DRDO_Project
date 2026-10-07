#!/usr/bin/env bash
# Round-2c: after round-2b (v3 LL + ablations) finishes, resume the two-microphone network from step 2000
# (it stopped on a data-generator edge case, now fixed) and evaluate it on the dual-mic test set.
cd "$(dirname "$0")/.." && source .venv/bin/activate
while pgrep -f "run_round2b_chain.sh" >/dev/null; do sleep 60; done
echo "[chain2c] round-2b finished $(date)"
python -u -m danc.train.train_dual --config configs/train_v3_2mic.yaml --resume > logs/v3_2mic_resume.out 2>&1
echo "[chain2c] v3_2mic resumed run finished $(date)"
python -m danc.inference.export_onnx --ckpt checkpoints/v3_2mic/best.pt --out exports/dancnet_v3_2mic.onnx > logs/export_v3_2mic.out 2>&1
python -u -m danc.eval.evaluate --set dualmic_v1 --systems dnn2:v3_2mic --save_audio dnn2:v3_2mic --workers 4 > logs/eval_dualmic_v3_2mic.out 2>&1
echo "[chain2c] two-mic evaluation finished $(date)"
