#!/usr/bin/env bash
#SBATCH --job-name=api-steering
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
# Submit from the repository root; Slurm copies this script to its spool directory.
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
exec bash ./run_script.sh "$@"
