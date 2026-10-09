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
BASELINE_ROOT="$PWD"
MODEL="${1:?Usage: sbatch slurm/baseline.sbatch MODEL [runner options]}"
shift
case "$MODEL" in songeval|tunejury|cmi|musecritic) ;; *) echo "Unknown model: $MODEL" >&2; exit 2;; esac
PYTHON="$BASELINE_ROOT/envs/$MODEL/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing environment. Run: bash scripts/setup_baseline.sh $MODEL" >&2
    exit 1
fi
export HF_HOME="$BASELINE_ROOT/cache/huggingface"
export TORCH_HOME="$BASELINE_ROOT/cache/torch"
export XDG_CACHE_HOME="$BASELINE_ROOT/cache/xdg"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="$BASELINE_ROOT/envs/$MODEL/bin:$PATH"
# Do not set CUDA_VISIBLE_DEVICES: Slurm supplies the assigned GPU.
bash "$BASELINE_ROOT/scripts/run_baseline.sh" "$MODEL" \
    --output "$BASELINE_ROOT/results/smoke/${MODEL}-${SLURM_JOB_ID}" "$@"
