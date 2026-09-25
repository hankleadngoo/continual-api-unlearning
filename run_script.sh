#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONDA_BIN="${CONDA_EXE:-conda}"
ENV_NAME="${ENV_NAME:-continual-unlearning}"
command -v "$CONDA_BIN" >/dev/null 2>&1 || { echo "Conda not found; set CONDA_EXE to its absolute path." >&2; exit 1; }
export PYTHONUNBUFFERED=1
# Explicit subcommands remain available, e.g. bash run_script.sh fetch-data --family deepseek.
if [[ $# -gt 0 ]]; then
    exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -u algo.py "$@"
fi
MODEL="${MODEL:-deepseek-ai/deepseek-coder-1.3b-instruct}"
FAMILY="${FAMILY:-deepseek}"
OUTPUT="${OUTPUT:-results/${FAMILY}_pipeline_$(date +%Y%m%d_%H%M%S)_$$}"
QUANTIZATION="${QUANTIZATION:-none}"
if [[ "${FETCH_DATA:-1}" == "1" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -u algo.py fetch-data --family "$FAMILY"
fi
args=(pipeline --model "$MODEL" --forget "data/$FAMILY/D_forget.json"
      --test "data/$FAMILY/D_test.json" --output "$OUTPUT"
      --device "${DEVICE:-auto}" --dtype "${DTYPE:-float16}"
      --quantization "$QUANTIZATION" --max-length "${MAX_LENGTH:-1024}"
      --max-new-tokens "${MAX_NEW_TOKENS:-64}" --gate-steps "${GATE_STEPS:-300}"
      --max-samples "${TRAIN_SAMPLES:-0}" --eval-samples "${EVAL_SAMPLES:-0}"
      --strength "${STRENGTH:-1}")
if [[ -n "${STRENGTH_GRID:-}" ]]; then
    args+=(--strength-grid "$STRENGTH_GRID")
fi
echo "Starting pipeline: model=$MODEL family=$FAMILY output=$OUTPUT"
exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -u algo.py "${args[@]}"
