#!/usr/bin/env python3
"""Prepare paired audio data and run one baseline per process/environment."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('songeval', 'tunejury', 'cmi', 'musecritic')
KEYS = ('Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def resolve_audio(rows, manifest):
    for row in rows:
        for key in ('audio_a', 'audio_b'):
            path = (Path(manifest).resolve().parent / row[key]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            row[key] = str(path)
    return rows


def prepare(args):
    source = args.source.resolve()
    rows = read_jsonl(source)
    for i, row in enumerate(rows):
        row['pair_id'] = f'pair_{i:04d}'
        if row.get('preference') not in ('A', 'B'):
            raise ValueError(f'Invalid preference in row {i}')
        for key in ('audio_a', 'audio_b'):
            path = (source.parent / row[key]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            row[key] = str(path)
    if args.count < 0 or args.count > len(rows):
        raise ValueError(f'count must be between 0 and {len(rows)} (0 means all)')
    # Alternate shuffled vocal/instrumental pools; count=1 selects a vocal pair.
    rng = random.Random(args.seed)
    pools = [[r for r in rows if bool(r.get('is_instrumental')) == flag]
             for flag in (False, True)]
    for pool in pools:
        rng.shuffle(pool)
    selected = []
    while any(pools):
        for pool in pools:
            if pool:
                selected.append(pool.pop())
    selected = selected[:args.count or len(rows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Avoid silently changing a manifest used by an already queued job.
    with args.output.open('x', encoding='utf-8') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Prepared {len(selected)} pairs: {args.output}')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_scorer(args):
    import torch
    import numpy as np
    np.random.seed(42)
    if args.model == 'tunejury':
        # Upstream Scorer handles CLAP/MERT preprocessing and its scalar reward.
        sys.path.insert(0, str(ROOT / 'repos/TuneJury'))
        from tunejury.score import Scorer
        scorer = Scorer.from_pretrained(
            str(ROOT / 'repos/TuneJury/checkpoints/tunejury.pt'),
            clap_ckpt_path=str(ROOT / 'checkpoints/clap.pt'), device=args.device)
        return lambda path, row: {'scores': {'reward': scorer.score(
            path, prompt='' if args.empty_prompt else row['prompt'])}}
    if args.model == 'songeval':
        from muq import MuQ
        from safetensors.torch import load_file
        import librosa
        module = load_module('songeval_model', ROOT / 'repos/SongEval/model.py')
        from omegaconf import OmegaConf
        config = dict(OmegaConf.load(ROOT / 'repos/SongEval/config.yaml').generator)
        config.pop('_target_')
        head = module.Generator(**config).to(args.device).eval()
        head.load_state_dict(load_file(str(ROOT / 'repos/SongEval/ckpt/model.safetensors')))
        encoder = MuQ.from_pretrained('OpenMuQ/MuQ-large-msd-iter').to(args.device).eval()

        def score(path, row):
            # Match SongEval's eval.py: mono 24 kHz -> MuQ layer 6 -> five scores.
            wave, _ = librosa.load(path, sr=24000)
            audio = torch.tensor(wave, device=args.device).unsqueeze(0)
            features = encoder(audio, output_hidden_states=True)['hidden_states'][6]
            values = head(features).squeeze(0).float().cpu().tolist()
            return {'scores': dict(zip(KEYS, values))}
        return score
    if args.model == 'cmi':
        # Use the upstream final mode and its default audio duration limit.
        sys.path.insert(0, str(ROOT / 'repos/CMI-RewardBench'))
        from baselines.inference import RewardModelInference
        scorer = RewardModelInference(str(args.checkpoint or ROOT / 'checkpoints/cmi/model.safetensors'),
                                      device=args.device, mode='final',
                                      bf16=args.device.startswith('cuda') and torch.cuda.is_bf16_supported())
        return lambda path, row: {'scores': scorer.score(
            path, text=row['prompt'], lyrics=row.get('lyrics') or '')}
    repo = ROOT / 'repos/MuseCritic/infer'
    module = load_module('musecritic_inference', repo / 'infer.py')
    module.INFER_CODE_PATH = repo
    checkpoint = (args.checkpoint or ROOT / 'checkpoints/musecritic').resolve()
    _, _, processor_class = module.load_components(checkpoint)
    processor = processor_class.from_pretrained(str(checkpoint), local_files_only=True,
                                                enable_time_marker=False)
    device = torch.device(args.device)
    model = module.load_model(checkpoint, device)
    template = read_jsonl(repo / 'examples/input.jsonl')[0]
    rubric = next(m for m in template['messages'] if m['role'] == 'user')

    def score(path, row):
        # Only the released rubric and audio enter the model; no reference critique/labels.
        result = module.infer_one(model, processor, {'messages': [rubric], 'audios': [path]},
                                  device, args.max_new_tokens)
        return {'scores': result['model_infer'], 'critique': result['infer_critic']}
    return score


def comparison(a, b):
    return 'A' if a > b else 'B' if b > a else 'tie'


def run(args):
    rows = read_jsonl(args.manifest)
    if args.count < 0 or args.count > len(rows):
        raise ValueError(f'count must be between 0 and {len(rows)} (0 means all)')
    rows = resolve_audio(rows[:args.count or len(rows)], args.manifest)
    if not rows:
        raise ValueError('Empty manifest')
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate pair IDs')
    for row in rows:
        if row['preference'] not in ('A', 'B'):
            raise ValueError('Preference must be A or B')
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: verify GPU allocation and PyTorch installation')
    if args.model == 'musecritic' and args.device.startswith('cuda') and not torch.cuda.is_bf16_supported():
        raise RuntimeError('MuseCritic requires a GPU supporting bfloat16')
    random.seed(42)
    torch.manual_seed(42)
    if args.output is None:
        args.output = ROOT / 'results' / f'{args.model}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'pairs.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    metadata = dict(vars(args))
    metadata.update(slurm_job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__,
                    seed=42,
                    audio_policy='no runner cropping; model-specific preprocessing may crop or chunk',
                    primary_score='mean of all returned scores',
                    tie_policy='exact ties count as incorrect',
                    gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None)
    write_json(args.output / 'run.json', {k: str(v) if isinstance(v, Path) else v
                                        for k, v in metadata.items()})
    started = time.monotonic()
    with torch.inference_mode():
        scorer = make_scorer(args)
        load_seconds = time.monotonic() - started
        results = []
        with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                pair_start = time.monotonic()
                outputs = {}
                for side in ('a', 'b'):
                    outputs[side] = scorer(row['audio_' + side], row)
                    scores = outputs[side]['scores']
                    if not scores or not all(math.isfinite(float(v)) for v in scores.values()):
                        raise ValueError(f'Invalid scores: {row["pair_id"]}, side {side}')
                    outputs[side]['mean_score'] = sum(float(v) for v in scores.values()) / len(scores)
                predicted = comparison(outputs['a']['mean_score'], outputs['b']['mean_score'])
                result = dict(pair_id=row['pair_id'], preference=row['preference'],
                              is_instrumental=row['is_instrumental'], prediction=predicted,
                              correct=predicted == row['preference'],
                              seconds=time.monotonic() - pair_start, **outputs)
                if args.model == 'cmi':
                    result['alignment_prediction'] = comparison(
                        outputs['a']['scores']['alignment'], outputs['b']['scores']['alignment'])
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                results.append(result)
                print(f'{row["pair_id"]}: predicted={predicted}, label={row["preference"]}', flush=True)
    summary = {'pairs': len(results), 'accuracy': sum(r['correct'] for r in results) / len(results),
               'ties': sum(r['prediction'] == 'tie' for r in results),
               'load_seconds': load_seconds, 'total_seconds': time.monotonic() - started,
               'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30
               if args.device.startswith('cuda') else None}
    for flag, name in ((False, 'vocal'), (True, 'instrumental')):
        group = [r for r in results if r['is_instrumental'] == flag]
        summary[name] = {'pairs': len(group), 'accuracy': sum(r['correct'] for r in group) / len(group)
                         if group else None}
    if args.model == 'cmi':
        summary['alignment_accuracy'] = sum(r['alignment_prediction'] == r['preference']
                                            for r in results) / len(results)
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print(f'Results: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source', type=Path, default=ROOT / 'data/raw/annotations_with_pref.jsonl')
    prep.add_argument('--output', type=Path, default=ROOT / 'data/processed/smoke/pairs.jsonl')
    prep.add_argument('--count', type=int, default=1, help='0 selects all pairs')
    prep.add_argument('--seed', type=int, default=42)
    runner = commands.add_parser('run')
    runner.add_argument('--model', choices=MODELS, required=True)
    runner.add_argument('--manifest', type=Path, default=ROOT / 'examples/pairs.jsonl')
    runner.add_argument('--count', type=int, default=1, help='Number of pairs; 0 selects all')
    runner.add_argument('--output', type=Path, help='Default: results/MODEL-TIMESTAMP')
    runner.add_argument('--device', default='cuda:0')
    runner.add_argument('--checkpoint', type=Path, help='CMI checkpoint file or MuseCritic directory')
    runner.add_argument('--empty-prompt', action='store_true', help='TuneJury empty-prompt ablation')
    runner.add_argument('--max-new-tokens', type=int, default=4096)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
