#!/usr/bin/env bash
# Evaluate every v16 GRPO arm on the frozen v15 (100) and v14 (72) probe sets.
set -u
cd /root/autodl-tmp/lite-coding-agent
export PYTHONPATH=/root/autodl-tmp/lite-coding-agent/src
PY=/root/miniconda3/bin/python
ROOT=private/v16/grpo_v16

for name in \
  v16_base_beta0.01_g8_lr5e-6 \
  v16_best_beta0.01_g4_lr5e-6 \
  v16_second_beta0.005_g8_lr5e-6; do
  arm="$ROOT/arms/$name/policy"
  if [ ! -d "$arm" ]; then
    echo "=== SKIP $name (missing adapter) ==="
    continue
  fi

  out15="$ROOT/eval_v15/$name"
  mkdir -p "$out15"
  echo "=== EVAL v15 $name $(date -Is) ==="
  "$PY" scripts/v10/run_m2_eval_v10.py --adapter-path "$arm" \
    --eval-file private/v15/eval_v15/eval.jsonl --out-dir "$out15" \
    --layers path,delete,edit,normal
  "$PY" scripts/v14/report_action_eval.py --runs "$out15/runs.jsonl" \
    --eval-file private/v15/eval_v15/eval.jsonl --label "$name" \
    --out-json "$out15/report.json" --out-md "$out15/report.md"

  out14="$ROOT/eval_v14/$name"
  mkdir -p "$out14"
  echo "=== EVAL v14 $name $(date -Is) ==="
  "$PY" scripts/v10/run_m2_eval_v10.py --adapter-path "$arm" \
    --eval-file private/v14/step2_benchmark/eval_set_frozen.jsonl --out-dir "$out14" \
    --layers path,delete,edit,normal
  "$PY" scripts/v14/report_action_eval.py --runs "$out14/runs.jsonl" \
    --eval-file private/v14/step2_benchmark/eval_set_frozen.jsonl --label "$name" \
    --out-json "$out14/report.json" --out-md "$out14/report.md"
  echo "=== DONE $name $(date -Is) ==="
done

"$PY" scripts/v16/summarize_grpo_v16.py
echo "ALL_EVAL_DONE $(date -Is)"
