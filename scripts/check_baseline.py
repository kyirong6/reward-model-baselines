#!/usr/bin/env python3
"""Check local inputs, imports and optional GPU availability without loading weights."""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from baseline_smoke import ROOT, MODELS, run, downloaded_path
from bootstrap_repos import bootstrap


def checkpoint(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Missing checkpoint: {path}; run the model download command')
    with path.open('rb') as stream:
        if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'{path} is a Git LFS pointer; install Git LFS and run git lfs pull in its repository')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true', help='Use only on an allocated GPU node')
    args = parser.parse_args()
    bootstrap(check=True)
    run(SimpleNamespace(manifest=args.manifest, validate_only=True))
    modules = {
        'songeval': ['torch', 'torchaudio', 'muq', 'librosa', 'safetensors', 'omegaconf'],
        'tunejury': ['torch', 'torchaudio', 'laion_clap', 'transformers'],
        'cmi': ['torch', 'torchaudio', 'transformers', 'safetensors', 'peft'],
        'musecritic': ['torch', 'torchaudio', 'transformers', 'flash_attn'],
    }
    for name in modules[args.model]:
        importlib.import_module(name)
    files = {
        'songeval': ['repos/SongEval/ckpt/model.safetensors'],
        'tunejury': ['repos/TuneJury/checkpoints/tunejury.pt', 'checkpoints/clap.pt'],
        'cmi': ['checkpoints/cmi/model.safetensors', 'checkpoints/cmi/config.yaml'],
        'musecritic': ['checkpoints/musecritic/config.json'],
    }
    for path in files[args.model]:
        checkpoint(ROOT / path)
    record = ROOT / 'checkpoints' / (args.model + '-downloads.json')
    for repo in json.loads(record.read_text()):
        downloaded_path(args.model, repo)
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in this environment/allocation')
        if args.model == 'musecritic' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('MuseCritic requires a GPU supporting bfloat16')
        print(f'GPU: {torch.cuda.get_device_name()}')
    print('Preflight passed. Model loading and inference still require a smoke run.')


if __name__ == '__main__':
    main()
