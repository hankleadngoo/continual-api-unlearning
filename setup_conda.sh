#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONDA_BIN="${CONDA_EXE:-conda}"
ENV_NAME="${ENV_NAME:-continual-unlearning}"
command -v "$CONDA_BIN" >/dev/null 2>&1 || { echo "Conda not found. Install Miniconda/Miniforge or set CONDA_EXE." >&2; exit 1; }
# Use UPDATE_ENV=1 when updating an existing environment.
if [[ "${UPDATE_ENV:-0}" == "1" ]]; then
    "$CONDA_BIN" env update --name "$ENV_NAME" --file environment.yml
else
    "$CONDA_BIN" env create --name "$ENV_NAME" --file environment.yml
fi
if [[ "${INSTALL_4BIT:-0}" == "1" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install -r requirements-4bit.txt
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip check
echo "Environment ready: $ENV_NAME. Run: bash run_script.sh"
