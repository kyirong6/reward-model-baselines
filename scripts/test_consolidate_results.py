"""Checks for repeated runs, legacy output, and incomplete/invalid runs."""
import json
from pathlib import Path
import tempfile
import unittest

from consolidate_results import consolidate


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_run(self, name, completed=True, saved_mean=False):
        path = self.root / name
        path.mkdir()
        metadata = {'model': 'songeval', 'primary_score': 'Musicality',
                    'tie_policy': 'exact ties count as incorrect', 'seed': 42}
        (path / 'run.json').write_text(json.dumps(metadata))
        if completed:
            (path / 'summary.json').write_text(json.dumps({'pairs': 1, 'accuracy': 1.0}))
        pair = {'pair_id': 'p1', 'prompt': '音楽', 'audio_a': '/a.mp3', 'audio_b': '/b.mp3'}
        prediction = {'pair_id': 'p1', 'prediction': 'A', 'preference': 'A', 'correct': True,
                      'a': {'scores': {'Musicality': 5, 'Clarity': 1}, 'critique': '音楽'},
                      'b': {'scores': {'Musicality': 4, 'Clarity': 4}}}
        if saved_mean:
            prediction['a']['mean_score'] = 3
            prediction['b']['mean_score'] = 4
        (path / 'pairs.jsonl').write_text(json.dumps(pair) + '\n')
        (path / 'predictions.jsonl').write_text(json.dumps(prediction) + '\n')
        return path

    def test_repeated_runs_legacy_decisions_and_metadata(self):
        old = self.make_run('songeval-old')
        original = (old / 'predictions.jsonl').read_bytes()
        self.make_run('songeval-new', saved_mean=True)
        self.make_run('songeval-pending', completed=False)
        summary = consolidate(self.root, self.root)
        rows = [json.loads(line) for line in
                (self.root / 'combined_predictions.jsonl').read_text().splitlines()]
        self.assertEqual(summary['run_count'], 2)
        self.assertEqual(summary['prediction_count'], 2)
        self.assertEqual(summary['skipped_incomplete_runs'], ['songeval-pending'])
        self.assertEqual(set(summary['models']['songeval']), {'songeval-old', 'songeval-new'})
        legacy = next(row for row in rows if row['run_id'] == 'songeval-old')
        # The derived mean would pick B; preserve the original Musicality decision A.
        self.assertEqual(legacy['prediction'], 'A')
        self.assertEqual(legacy['primary_score'], 'Musicality')
        self.assertEqual(legacy['a']['mean_score'], 3)
        self.assertEqual(legacy['a']['mean_score_source'], 'derived')
        self.assertEqual(legacy['a']['critique'], '音楽')
        self.assertEqual(legacy['pair']['prompt'], '音楽')
        self.assertEqual(rows[0]['a']['mean_score_source'], 'saved')
        self.assertEqual(original, (old / 'predictions.jsonl').read_bytes())
        first = (self.root / 'combined_predictions.jsonl').read_bytes()
        consolidate(self.root, self.root)
        self.assertEqual(first, (self.root / 'combined_predictions.jsonl').read_bytes())

    def test_invalid_completed_run_does_not_replace_outputs(self):
        path = self.make_run('songeval-invalid')
        (path / 'summary.json').write_text(json.dumps({'pairs': 2}))
        output = self.root / 'combined_predictions.jsonl'
        output.write_text('existing output')
        with self.assertRaises(ValueError):
            consolidate(self.root, self.root)
        self.assertEqual(output.read_text(), 'existing output')

    def test_missing_results_directory_rejected(self):
        with self.assertRaises(FileNotFoundError):
            consolidate(self.root / 'missing', self.root)


if __name__ == '__main__':
    unittest.main()
