#!/usr/bin/env bash
# Run from a node with internet access; inference is submitted separately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASELINE_ROOT="$PWD"
MODEL="${1:-songeval}"
case "$MODEL" in
    songeval) REPO=SongEval; URL=https://github.com/ASLP-lab/SongEval.git; PY_VERSION=3.10 ;;
    tunejury) REPO=TuneJury; URL=https://github.com/yonghyunk1m/TuneJury.git; PY_VERSION=3.10 ;;
    cmi) REPO=CMI-RewardBench; URL=https://github.com/Haiwen-Xia/CMI-RewardBench.git; PY_VERSION=3.10 ;;
    musecritic) REPO=MuseCritic; URL=https://github.com/WuqnEl/MuseCritic.git; PY_VERSION=3.12 ;;
    all)
        for model in songeval tunejury cmi musecritic; do bash scripts/setup.sh "$model"; done
        exit 0 ;;
    *) echo "Usage: bash scripts/setup.sh [songeval|tunejury|cmi|musecritic|all]" >&2; exit 2 ;;
esac
CONDA="${CONDA_EXE:-$BASELINE_ROOT/tools/miniforge3/bin/conda}"
if [[ ! -x "$CONDA" ]]; then
    CONDA="$(command -v conda || true)"
fi
if [[ ! -x "$CONDA" ]]; then
    echo "Install Conda/Miniforge or set CONDA_EXE to its executable, then retry." >&2
    exit 1
fi
mkdir -p repos
if [[ ! -d "repos/$REPO" ]]; then
    git clone "$URL" "repos/$REPO"
fi
export CONDA_PKGS_DIRS="$BASELINE_ROOT/cache/conda-pkgs"
export PIP_CACHE_DIR="$BASELINE_ROOT/cache/pip"
export HF_HOME="$BASELINE_ROOT/cache/huggingface"
export TORCH_HOME="$BASELINE_ROOT/cache/torch"
export XDG_CACHE_HOME="$BASELINE_ROOT/cache/xdg"
export PYTHONNOUSERSITE=1
ENV_PATH="$BASELINE_ROOT/envs/$MODEL"
mkdir -p logs results checkpoints cache/pip
if [[ ! -x "$ENV_PATH/bin/python" ]]; then
    "$CONDA" create -y -p "$ENV_PATH" "python=$PY_VERSION" pip
fi
"$CONDA" install -y -p "$ENV_PATH" -c conda-forge ffmpeg libsndfile
export PATH="$ENV_PATH/bin:$PATH"
PYTHON="$ENV_PATH/bin/python"
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
        # Build flash-attn with a toolkit matching the PyTorch CUDA wheels.
        "$CONDA" install -y -p "$ENV_PATH" --override-channels -c conda-forge gxx_linux-64=13
        NVCC="$(command -v nvcc || true)"
        if [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
            NVCC="$CUDA_HOME/bin/nvcc"
        fi
        if [[ -n "$NVCC" ]] && "$NVCC" --version | grep -q 'release 12.8,'; then
            export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$NVCC")")")"
        else
            "$CONDA" install -y -p "$ENV_PATH" --override-channels -c nvidia -c conda-forge \
                cuda-toolkit=12.8
            export CUDA_HOME="$ENV_PATH"
        fi
        export PATH="$CUDA_HOME/bin:$PATH"
        export CC="$ENV_PATH/bin/x86_64-conda-linux-gnu-cc"
        export CXX="$ENV_PATH/bin/x86_64-conda-linux-gnu-c++"
        export MAX_JOBS="${MAX_JOBS:-4}"
        # Install PyTorch/build dependencies before compiling flash-attn.
        "$PYTHON" -m pip install numpy==2.4.4 packaging ninja wheel psutil
        "$PYTHON" -m pip install torch==2.9.1 torchaudio==2.9.1 \
            --index-url https://download.pytorch.org/whl/cu128
        "$PYTHON" -m pip install --no-build-isolation -r repos/MuseCritic/requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu128
        ;;
esac
"$PYTHON" scripts/download_baseline.py "$MODEL"
echo "Setup complete. Run: bash scripts/test.sh $MODEL"
