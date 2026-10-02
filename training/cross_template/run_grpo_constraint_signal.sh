#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/lite-coding-agent}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-$REPO/private/adapters/train_v6_37928a28}"
TASKS="${TASKS:-$REPO/private/session_constraint/tasks64.json}"
OUT="${OUT:-$REPO/private/cross_template/grpo_constraint_signal}"
EXPECTED_ADAPTER_SHA256="${EXPECTED_ADAPTER_SHA256:-37928a28d67a5a8350f3b49450fc6945c3ffe05c09da7ea0de36136b4a6a4b96}"
EXPECTED_TASKS_SHA256="${EXPECTED_TASKS_SHA256:-c0aea4c3ed3ccd4a87c3bc5c8ddb719ef636659f396697556374ff051c4b1f54}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-14400}"
ROLLOUTS="${ROLLOUTS:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-20261019}"

export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT"
LOG="$OUT/run.log"
watchdog_log="$OUT/watchdog.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ ! -f "$path" ]]; then
    log "missing $label: $path"
    exit 2
  fi
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    log "$label SHA mismatch: expected=$expected actual=$actual"
    exit 3
  fi
  log "$label SHA ok: $actual"
}

require_sha256 "$ADAPTER/adapter_model.safetensors" "$EXPECTED_ADAPTER_SHA256" "train-v6 adapter"
require_sha256 "$TASKS" "$EXPECTED_TASKS_SHA256" "tasks64"

if [[ ! -f "$MODEL/config.json" ]]; then
  log "missing model: $MODEL"
  exit 4
fi

nohup bash -c "sleep $WATCHDOG_SECONDS && shutdown -h now" >"$watchdog_log" 2>&1 &
echo "$!" >"$OUT/watchdog.pid"
log "START grpo_constraint_signal watchdog_pid=$(cat "$OUT/watchdog.pid")"

set +e
"$PYTHON" "$REPO/training/kaggle/session_constraint_probe.py" \
  --model-path "$MODEL" \
  --adapter-path "$ADAPTER" \
  --tasks "$TASKS" \
  --out-dir "$OUT" \
  --condition v6 \
  --modes untagged \
  --rollouts "$ROLLOUTS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --seed "$SEED" \
  --max-model-len 8192 2>&1 | tee -a "$LOG"
probe_status=${PIPESTATUS[0]}
if [[ "$probe_status" -ne 0 ]]; then
  log "FAIL probe status=$probe_status"
  exit "$probe_status"
fi

"$PYTHON" "$REPO/scripts/analyze_grpo_constraint_signal.py" \
  --input-dir "$OUT" \
  --out-json "$OUT/results.json" \
  --out-md "$OUT/summary.md" 2>&1 | tee -a "$LOG"
analysis_status=${PIPESTATUS[0]}
set -e
if [[ "$analysis_status" -ne 0 ]]; then
  log "FAIL analysis status=$analysis_status"
  exit "$analysis_status"
fi

"$PYTHON" - "$OUT" "$ADAPTER" "$TASKS" "$ROLLOUTS" "$TEMPERATURE" "$TOP_P" "$SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
adapter = Path(sys.argv[2])
tasks = Path(sys.argv[3])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


payload = {
    "version": "grpo-constraint-signal-run-v1",
    "mode": "untagged",
    "rollouts": int(sys.argv[4]),
    "temperature": float(sys.argv[5]),
    "top_p": float(sys.argv[6]),
    "seed": int(sys.argv[7]),
    "decoding": "sampled",
    "sampling_note": "Sampled repeated probe; not the earlier greedy one-shot probe.",
    "adapter_path": str(adapter),
    "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
    "tasks_path": str(tasks),
    "tasks_sha256": sha256(tasks),
}
(out / "run_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

date -Is >"$OUT/DONE"
log "DONE grpo_constraint_signal; watchdog remains active until results are pulled locally"
