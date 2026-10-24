#!/usr/bin/env bash
# System-side 3-group pressure probe runner (G0/G1/G2, Qwen2.5-7B + train-v6).
# Prereg: docs/experiments/system_side_qwen_3group_prereg.md
set -uo pipefail

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/autodl-tmp/lite-coding-agent/private/adapters/train_v6_37928a28}"
EXPECTED_ADAPTER_SHA="${EXPECTED_ADAPTER_SHA:-37928a28d67a5a8350f3b49450fc6945c3ffe05c09da7ea0de36136b4a6a4b96}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/lite-coding-agent/private/cross_template/system_side_qwen_3group}"
SEEDS="${SEEDS:-20261101,20261102,20261103,20261104,20261105,20261106,20261107,20261108,20261109,20261110,20261201,20261202,20261203,20261204,20261205,20261206,20261207,20261208,20261209,20261210}"
MODES="${MODES:-G0,G1,G2}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-7200}"
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/lite-coding-agent}"

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
: > "$LOG"

echo "[runner] model=$MODEL_PATH" | tee -a "$LOG"
echo "[runner] adapter=$ADAPTER_PATH" | tee -a "$LOG"
echo "[runner] seeds=$SEEDS modes=$MODES" | tee -a "$LOG"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[runner] FATAL: model path missing" | tee -a "$LOG"
  exit 2
fi
if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
  echo "[runner] FATAL: adapter missing" | tee -a "$LOG"
  exit 2
fi
ACTUAL_SHA="$(sha256sum "$ADAPTER_PATH/adapter_model.safetensors" | awk '{print $1}')"
echo "[runner] adapter_sha=$ACTUAL_SHA" | tee -a "$LOG"
if [ "$ACTUAL_SHA" != "$EXPECTED_ADAPTER_SHA" ]; then
  echo "[runner] FATAL: adapter sha mismatch" | tee -a "$LOG"
  exit 2
fi

# Watchdog: last-resort shutdown if the main flow hangs.
nohup bash -c "sleep $WATCHDOG_SECONDS && shutdown -h now" >/dev/null 2>&1 &
WATCHDOG_PID=$!
echo "$WATCHDOG_PID" > "$OUT_DIR/watchdog.pid"
cleanup() {
  kill "$WATCHDOG_PID" 2>/dev/null || true
  rm -f "$OUT_DIR/watchdog.pid"
}
trap cleanup EXIT

"$PYTHON" -m training.cross_template.system_side_probe \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA" \
  --out-dir "$OUT_DIR" \
  --seeds "$SEEDS" \
  --modes "$MODES" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
echo "[runner] probe_exit=$STATUS" | tee -a "$LOG"

if [ "$STATUS" -eq 0 ]; then
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT_DIR/DONE"
  echo "[runner] DONE" | tee -a "$LOG"
else
  echo "[runner] FAILED status=$STATUS" | tee -a "$LOG"
fi
exit "$STATUS"
