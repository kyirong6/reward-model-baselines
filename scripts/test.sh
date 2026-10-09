#!/usr/bin/env bash
# Use on an allocated GPU, or through slurm/baseline.sbatch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL="${1:-songeval}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$MODEL" == all ]]; then
    for model in songeval tunejury cmi musecritic; do bash scripts/test.sh "$model" "$@"; done
    exit 0
fi
case "$MODEL" in songeval|tunejury|cmi|musecritic) ;; *) echo "Unknown model: $MODEL" >&2; exit 2;; esac
PYTHON="$PWD/envs/$MODEL/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Run bash scripts/setup.sh $MODEL first." >&2
    exit 1
fi
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="$PWD/envs/$MODEL/bin:$PATH"
"$PYTHON" scripts/baseline_smoke.py run --model "$MODEL" "$@"
