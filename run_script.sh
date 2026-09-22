#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-codellama/CodeLlama-7b-hf}"

if [[ $# -gt 0 ]]; then
    exec "$PYTHON" algo.py "$@"
fi

"$PYTHON" algo.py prepare
"$PYTHON" algo.py train --model "$MODEL"
"$PYTHON" algo.py evaluate --model "$MODEL" --split validation --output results/validation.json
# Use held-out test only after selecting hyperparameters on validation.
