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
"$PYTHON" algo.py evaluate-api --model "$MODEL" --output results/api_counts.json
