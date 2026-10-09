#!/usr/bin/env bash
#SBATCH --job-name=music-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# Submit from workspace root. Override resource requests with sbatch flags.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the workspace root}"
MODEL="${1:-songeval}"
if [[ $# -gt 0 ]]; then shift; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
bash scripts/test.sh "$MODEL" "$@"
