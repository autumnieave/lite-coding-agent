#!/usr/bin/env bash
set -euo pipefail

# All paths are input/output parameters. No rent-platform path is assumed.
ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-prepare}"                 # prepare | train | all
DRY_RUN="${DRY_RUN:-1}"                 # 1 = parse/check only; 0 = execute
EXPERIMENTS="${EXPERIMENTS:-e5b e1 e2}"

CONTROL_VARS_MD="${CONTROL_VARS_MD:-$ROOT/training/cross_template/CONTROL_VARS.md}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT/training/configs/qlora_7b_t4.json}"
TRAIN_V5="${TRAIN_V5:-$ROOT/private/sft_data_v5/train.jsonl}"
TRAIN_V6="${TRAIN_V6:-$ROOT/private/sft_data_v6/train.jsonl}"
VAL_FILE="${VAL_FILE:-$ROOT/private/sft_data_v6/val.jsonl}"
MIDDLE_TASKS="${MIDDLE_TASKS:-$ROOT/private/final_test_v3/tasks.jsonl}"
BASELINE_PROBES="${BASELINE_PROBES:-$ROOT/private/kaggle_pull_session_constraint64_v1/session_constraint64/baseline.jsonl}"
WORK_DIR="${WORK_DIR:-$ROOT/private/cross_template_runs}"
MODEL_PATH="${MODEL_PATH:-}"
E2_RANKS="${E2_RANKS:-32 64 128}"
E2_ALPHA_2R="${E2_ALPHA_2R:-0}"         # 1 = set E2 lora_alpha = 2 * rank
STRICT_CONTROL="${STRICT_CONTROL:-1}"   # 1 = stop when run_manifest drifts from CONTROL_VARS

control_values() {
  "$PYTHON" - "$CONTROL_VARS_MD" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
if match is None:
    raise SystemExit(f"CONTROL_VARS.md has no fenced JSON block: {path}")
data = json.loads(match.group(1))
keys = (
    "total_train_rows",
    "train_v5_rows",
    "optimizer_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "num_train_epochs",
    "max_seq_length",
    "seed",
    "loss_mode",
    "lora_r_default",
    "e5b_negatives",
)
for key in keys:
    print(f"{key.upper()}={data[key]}")
PY
}

while IFS='=' read -r key value; do
  case "$key" in
    TOTAL_TRAIN_ROWS|TRAIN_V5_ROWS|OPTIMIZER_STEPS|PER_DEVICE_TRAIN_BATCH_SIZE|\
    GRADIENT_ACCUMULATION_STEPS|LEARNING_RATE|NUM_TRAIN_EPOCHS|MAX_SEQ_LENGTH|\
    SEED|LOSS_MODE|LORA_R_DEFAULT|E5B_NEGATIVES)
      export "$key=$value"
      ;;
  esac
done < <(control_values)

has_experiment() {
  case " $EXPERIMENTS " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "missing required path: $1" >&2
    exit 1
  fi
}

assert_trainer_compatible() {
  # training/train_sft.py currently hardcodes these two values. Fail loudly
  # instead of silently training with different controls.
  if [[ "$PER_DEVICE_TRAIN_BATCH_SIZE" != "1" || "$GRADIENT_ACCUMULATION_STEPS" != "8" ]]; then
    echo "train_sft.py hardcodes batch=1 / grad_accum=8, but CONTROL_VARS says" \
      "$PER_DEVICE_TRAIN_BATCH_SIZE / $GRADIENT_ACCUMULATION_STEPS" >&2
    exit 2
  fi
}

make_config() {
  local output="$1"
  local rank="$2"
  local alpha="${3:-}"
  mkdir -p "$(dirname "$output")"
  "$PYTHON" - "$BASE_CONFIG" "$output" "$rank" "$CONTROL_VARS_MD" "$alpha" <<'PY'
import json
import re
import sys
from pathlib import Path

base, output, rank, control_path, alpha = sys.argv[1:]
config = json.loads(Path(base).read_text(encoding="utf-8"))
text = Path(control_path).read_text(encoding="utf-8")
match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
if match is None:
    raise SystemExit("CONTROL_VARS.md has no fenced JSON block")
control = json.loads(match.group(1))
config["lora_r"] = int(rank)
if alpha:
    config["lora_alpha"] = int(alpha)
config["loss_mode"] = str(control["loss_mode"])
config["max_seq_length"] = int(control["max_seq_length"])
config["learning_rate"] = float(control["learning_rate"])
config["num_train_epochs"] = int(control["num_train_epochs"])
config["per_device_train_batch_size"] = int(control["per_device_train_batch_size"])
config["gradient_accumulation_steps"] = int(control["gradient_accumulation_steps"])
config["seed"] = int(control["seed"])
Path(output).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

run_train() {
  local config="$1"
  local train_file="$2"
  local output_dir="$3"
  local -a command=(
    "$PYTHON" "$ROOT/training/train_sft.py"
    --config "$config"
    --model-path "$MODEL_PATH"
    --train-file "$train_file"
    --val-file "$VAL_FILE"
    --output-dir "$output_dir"
  )
  printf 'train: '
  printf '%q ' "${command[@]}"
  printf '\n'
  printf 'seed: %s loss_mode: %s\n' "$SEED" "$LOSS_MODE"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$PYTHON" - "$ROOT" "$config" "$MODEL_PATH" "$train_file" "$VAL_FILE" "$output_dir" \
    "$SEED" "$LOSS_MODE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from training.train_sft import run_sft

run_sft(
    config_path=Path(sys.argv[2]),
    model_path=Path(sys.argv[3]),
    train_file=Path(sys.argv[4]),
    val_file=Path(sys.argv[5]),
    output_dir=Path(sys.argv[6]),
    dry_run=False,
    loss_mode=sys.argv[8],
    seed=int(sys.argv[7]),
)
PY
  check_control_after_run "$output_dir"
}

check_control_after_run() {
  local output_dir="$1"
  local manifest="$output_dir/run_manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "missing run manifest: $manifest" >&2
    exit 3
  fi
  "$PYTHON" - "$manifest" "$OPTIMIZER_STEPS" "$STRICT_CONTROL" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
strict = sys.argv[3] == "1"
actual = int(manifest.get("optimizer_steps", -1))
if actual != expected:
    message = f"optimizer_steps drift: actual={actual} control={expected}"
    if strict:
        raise SystemExit(f"{message}; set STRICT_CONTROL=0 to keep the run")
    print(f"WARNING: {message}", file=sys.stderr)
PY
}

case "$MODE" in
  prepare|train|all) ;;
  *) echo "MODE must be prepare, train, or all" >&2; exit 2 ;;
esac
case "$DRY_RUN" in
  0|1) ;;
  *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

if [[ "$MODE" != "prepare" && "$DRY_RUN" == "0" && -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required for real training" >&2
  exit 2
fi
if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
  assert_trainer_compatible
fi

echo "MODE=$MODE DRY_RUN=$DRY_RUN EXPERIMENTS=$EXPERIMENTS"
echo "ROOT=$ROOT WORK_DIR=$WORK_DIR"
echo "CONTROL_VARS=$CONTROL_VARS_MD"
echo "TOTAL_TRAIN_ROWS=$TOTAL_TRAIN_ROWS OPTIMIZER_STEPS=$OPTIMIZER_STEPS"
echo "SEED=$SEED MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"

if [[ "$MODE" == "prepare" || "$MODE" == "all" ]]; then
  if has_experiment e5b; then
    require_file "$BASELINE_PROBES"
    require_file "$TRAIN_V6"
    builder_args=(
      "$PYTHON" "$ROOT/training/cross_template/e5b_build_negatives.py"
      --baseline "$BASELINE_PROBES"
      --positives "$TRAIN_V6"
      --out-dir "$WORK_DIR/e5b"
      --control-vars "$CONTROL_VARS_MD"
    )
    if [[ "$DRY_RUN" == "1" ]]; then builder_args+=(--dry-run); fi
    "${builder_args[@]}"
  fi
  if has_experiment e1; then
    require_file "$TRAIN_V5"
    require_file "$MIDDLE_TASKS"
    builder_args=(
      "$PYTHON" "$ROOT/training/cross_template/e1_sample_401.py"
      --train-v5 "$TRAIN_V5"
      --middle-tasks "$MIDDLE_TASKS"
      --out-dir "$WORK_DIR/e1"
      --control-vars "$CONTROL_VARS_MD"
    )
    if [[ "$DRY_RUN" == "1" ]]; then builder_args+=(--dry-run); fi
    "${builder_args[@]}"
  fi
fi

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
  require_file "$VAL_FILE"
  if [[ "$DRY_RUN" == "0" ]]; then require_path "$MODEL_PATH"; fi
  if has_experiment e5b; then
    e5b_train="$WORK_DIR/e5b/train.jsonl"
    if [[ "$DRY_RUN" == "0" ]]; then require_file "$e5b_train"; fi
    make_config "$WORK_DIR/configs/e5b.json" "$LORA_R_DEFAULT"
    run_train "$WORK_DIR/configs/e5b.json" "$e5b_train" "$WORK_DIR/e5b/g1"
  fi
  if has_experiment e1; then
    e1_train="$WORK_DIR/e1/train.jsonl"
    if [[ "$DRY_RUN" == "0" ]]; then require_file "$e1_train"; fi
    make_config "$WORK_DIR/configs/e1.json" "$LORA_R_DEFAULT"
    run_train "$WORK_DIR/configs/e1.json" "$e1_train" "$WORK_DIR/e1/g1"
  fi
  if has_experiment e2; then
    require_file "$TRAIN_V6"
    for rank in $E2_RANKS; do
      if [[ "$E2_ALPHA_2R" == "1" ]]; then
        alpha=$((rank * 2))
      else
        alpha=""
      fi
      make_config "$WORK_DIR/configs/e2_r${rank}.json" "$rank" "$alpha"
      run_train "$WORK_DIR/configs/e2_r${rank}.json" "$TRAIN_V6" "$WORK_DIR/e2/r${rank}"
    done
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN complete: no model, GPU, or training data output was written."
fi
