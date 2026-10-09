#!/usr/bin/env bash
# Run from a node with internet access; inference is submitted separately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASELINE_ROOT="$PWD"
MODEL="${1:?Usage: bash scripts/setup_baseline.sh songeval|tunejury|cmi|musecritic}"
PACKAGE_LOCK="${2:-}"
CONDA="${CONDA_EXE:-$BASELINE_ROOT/tools/miniforge3/bin/conda}"
if [[ ! -x "$CONDA" ]]; then
    CONDA="$(command -v conda || true)"
fi
if [[ ! -x "$CONDA" ]]; then
    echo "Install Conda/Miniforge or set CONDA_EXE to its executable, then retry." >&2
    exit 1
fi
python3 scripts/bootstrap_repos.py
export CONDA_PKGS_DIRS="$BASELINE_ROOT/cache/conda-pkgs"
export PIP_CACHE_DIR="$BASELINE_ROOT/cache/pip"
export HF_HOME="$BASELINE_ROOT/cache/huggingface"
export TORCH_HOME="$BASELINE_ROOT/cache/torch"
export XDG_CACHE_HOME="$BASELINE_ROOT/cache/xdg"
export PYTHONNOUSERSITE=1
case "$MODEL" in
    songeval|tunejury|cmi) PY_VERSION=3.10 ;;
    musecritic) PY_VERSION=3.12 ;;
    *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
ENV_PATH="$BASELINE_ROOT/envs/$MODEL"
mkdir -p logs results checkpoints cache/pip
if [[ ! -x "$ENV_PATH/bin/python" ]]; then
    "$CONDA" create -y -p "$ENV_PATH" "python=$PY_VERSION" pip
fi
"$CONDA" install -y -p "$ENV_PATH" -c conda-forge ffmpeg libsndfile
export PATH="$ENV_PATH/bin:$PATH"
PYTHON="$ENV_PATH/bin/python"
if [[ -n "$PACKAGE_LOCK" ]]; then
    # Replay packages.txt from a successful run on a compatible Linux/CUDA system.
    EXTRA_PIP_ARGS=()
    if [[ "$MODEL" == musecritic ]]; then
        "$PYTHON" -m pip install torch==2.9.1 torchaudio==2.9.1 \
            --index-url https://download.pytorch.org/whl/cu128
        "$PYTHON" -m pip install packaging ninja wheel psutil
        EXTRA_PIP_ARGS+=(--no-build-isolation)
    fi
    "$PYTHON" -m pip install -r "$PACKAGE_LOCK" \
        --extra-index-url "${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
        "${EXTRA_PIP_ARGS[@]}"
else
case "$MODEL" in
    tunejury)
        # Use the upstream environment.yml pins without overwriting an existing env.
        "$PYTHON" -m pip install PyYAML
        "$PYTHON" - <<'PY'
import subprocess, sys, yaml
with open('repos/TuneJury/environment.yml') as f:
    config = yaml.safe_load(f)
requirements = next(d['pip'] for d in config['dependencies'] if isinstance(d, dict) and 'pip' in d)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *requirements])
PY
        ;;
    songeval)
        "$PYTHON" -m pip install -r repos/SongEval/requirements.txt \
            torchaudio==2.7.0 safetensors einops huggingface_hub
        ;;
    cmi)
        "$PYTHON" -m pip install -r repos/CMI-RewardBench/baselines/requirements.txt \
            safetensors huggingface_hub sentencepiece peft
        ;;
    musecritic)
        # Install PyTorch/build dependencies before compiling flash-attn.
        "$PYTHON" -m pip install torch==2.9.1 torchaudio==2.9.1 \
            --index-url https://download.pytorch.org/whl/cu128
        "$PYTHON" -m pip install packaging ninja wheel psutil
        "$PYTHON" -m pip install --no-build-isolation -r repos/MuseCritic/requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu128
        ;;
esac
fi
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "envs/${MODEL}-installed.txt"
"$CONDA" list -p "$ENV_PATH" --explicit > "envs/${MODEL}-conda-explicit.txt"
echo "Environment ready. Next: $PYTHON scripts/download_baseline.py $MODEL"
