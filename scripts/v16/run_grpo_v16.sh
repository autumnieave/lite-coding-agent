#!/usr/bin/env bash
# v16 GRPO retrain: three arms from the restored v16 SFT adapter.
set -u
cd /root/autodl-tmp/lite-coding-agent
PY=/root/miniconda3/bin/python
COMMON=(--prompts private/v16/grpo_v16/prompts/prompts.jsonl
        --init-adapter private/v16/sft_v16/adapter
        --out-dir private/v16/grpo_v16/arms
        --weighting severity --epochs 1 --seed 20261019)

run_arm() {
  local name="$1"; shift
  echo "=== START ${name} $(date -Is) ==="
  "$PY" scripts/v14/train_grpo_v14.py "${COMMON[@]}" --name "${name}" "$@"
  echo "=== END ${name} rc=$? $(date -Is) ==="
}

run_arm v16_base_beta0.01_g8_lr5e-6   --beta 0.01  --group-size 8 --learning-rate 5e-6
run_arm v16_best_beta0.01_g4_lr5e-6   --beta 0.01  --group-size 4 --learning-rate 5e-6
run_arm v16_second_beta0.005_g8_lr5e-6 --beta 0.005 --group-size 8 --learning-rate 5e-6
echo "ALL_ARMS_DONE $(date -Is)"
