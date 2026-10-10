"""Combine completed baseline runs without changing their recorded decisions."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def consolidate(results_dir, output_dir):
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)
    records, models, skipped = [], {}, []
    for directory in sorted(results_dir.iterdir()):
        if not directory.is_dir() or not (directory / 'run.json').is_file():
            continue
        # The runner writes its summary only after all pairs finish.
        if not (directory / 'summary.json').is_file():
            skipped.append(directory.name)
            continue
        metadata = read_json(directory / 'run.json')
        summary = read_json(directory / 'summary.json')
        predictions = read_jsonl(directory / 'predictions.jsonl')
        pairs = read_jsonl(directory / 'pairs.jsonl')
        inputs = {pair['pair_id']: pair for pair in pairs}
        ids = [prediction['pair_id'] for prediction in predictions]
        if (len(inputs) != len(pairs) or len(set(ids)) != len(ids)
                or set(ids) != set(inputs) or len(ids) != summary['pairs']):
            raise ValueError(f'Inconsistent pair counts or IDs in {directory}')
        model, run_id = metadata['model'], directory.name
        for prediction in predictions:
            for side in ('a', 'b'):
                output = prediction[side]
                scores = output['scores']
                if not scores or not all(math.isfinite(float(v)) for v in scores.values()):
                    raise ValueError(f'Invalid scores in {directory}: {prediction["pair_id"]}')
                if 'mean_score' in output:
                    output['mean_score_source'] = 'saved'
                else:
                    output['mean_score'] = sum(float(v) for v in scores.values()) / len(scores)
                    output['mean_score_source'] = 'derived'
            records.append({**prediction, 'model': model, 'run_id': run_id,
                            'source_directory': str(directory),
                            'primary_score': metadata['primary_score'],
                            'tie_policy': metadata['tie_policy'],
                            'pair': inputs[prediction['pair_id']]})
        models.setdefault(model, {})[run_id] = {
            'source_directory': str(directory), 'run': metadata, 'summary': summary}
    combined = {'schema_version': 1, 'results_directory': str(results_dir),
                'run_count': sum(len(runs) for runs in models.values()),
                'prediction_count': len(records), 'skipped_incomplete_runs': skipped,
                'models': models}
    # Validate and serialize everything before replacing either derived artifact.
    prediction_text = ''.join(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
                              for record in records)
    summary_text = json.dumps(combined, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [('combined_predictions.jsonl', prediction_text),
                        ('combined_summary.json', summary_text)]:
        path = output_dir / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(value, encoding='utf-8')
        temporary.replace(path)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--output-dir', type=Path, help='Default: the results directory')
    args = parser.parse_args()
    combined = consolidate(args.results_dir, args.output_dir or args.results_dir)
    print(f'Combined {combined["run_count"]} runs, {combined["prediction_count"]} predictions; '
          f'skipped {len(combined["skipped_incomplete_runs"])} incomplete runs.')
    print(f'Outputs: {(args.output_dir or args.results_dir).resolve()}')


if __name__ == '__main__':
    main()
