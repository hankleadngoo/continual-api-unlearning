#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-codellama/CodeLlama-7b-hf}"
FAMILY="${FAMILY:-codellama}"

if [[ $# -gt 0 ]]; then
    exec "$PYTHON" algo.py "$@"
fi

"$PYTHON" algo.py fetch-data --family "$FAMILY"
"$PYTHON" algo.py prepare --forget "data/$FAMILY/D_forget.json" --test "data/$FAMILY/D_test.json" --output "data/$FAMILY/prepared.json"
"$PYTHON" algo.py train --model "$MODEL" --data "data/$FAMILY/prepared.json" --output "checkpoints/${FAMILY}_hf"
"$PYTHON" algo.py evaluate-api --model "$MODEL" --forget "data/$FAMILY/D_forget.json" --test "data/$FAMILY/D_test.json" --checkpoints "checkpoints/${FAMILY}_hf" --output "results/${FAMILY}_api_counts.json"
