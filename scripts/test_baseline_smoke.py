"""CPU-only checks for manifest selection and evaluation bookkeeping."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('baseline_smoke', Path(__file__).with_name('baseline_smoke.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'audio.mp3').touch()
        self.source = self.root / 'source.jsonl'
        self.rows = [dict(audio_a='audio.mp3', audio_b='audio.mp3',
                          prompt='multilingual\u2028prompt', preference='A' if i % 2 else 'B',
                          is_instrumental=bool(i % 2)) for i in range(12)]
        self.source.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in self.rows))

    def prepare(self, count=10, name='pairs.jsonl'):
        output = self.root / name
        runner.prepare(SimpleNamespace(source=self.source, output=output, count=count, seed=42))
        return output

    def test_selection_paths_unicode_and_no_overwrite(self):
        path = self.prepare()
        rows = runner.read_jsonl(path)
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(r['is_instrumental'] for r in rows), 5)
        self.assertFalse(rows[0]['is_instrumental'])
        self.assertEqual(rows[0]['prompt'], 'multilingual\u2028prompt')
        self.assertTrue(all(Path(r['audio_a']).is_absolute() for r in rows))
        self.assertEqual(path.read_bytes(), self.prepare(name='repeat.jsonl').read_bytes())
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(len(runner.read_jsonl(self.prepare(0, 'all.jsonl'))), 12)

    def test_missing_audio_rejected(self):
        (self.root / 'audio.mp3').unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_relative_audio_paths(self):
        rows = runner.resolve_audio(runner.read_jsonl(self.source), self.source)
        self.assertTrue(all(Path(row['audio_a']) == self.root / 'audio.mp3' for row in rows))

    def test_pair_scoring_and_tie_accounting(self):
        manifest = self.prepare(2)
        args = SimpleNamespace(manifest=manifest, output=self.root / 'results', model='cmi',
                               device='cpu', count=0)
        fake_torch = SimpleNamespace(manual_seed=lambda seed: None, __version__='test',
                                     inference_mode=contextlib.nullcontext)
        scores = iter([{'scores': {'quality': 2., 'alignment': 1.}},
                       {'scores': {'quality': 1., 'alignment': 2.}},
                       {'scores': {'quality': 1., 'alignment': 2.}},
                       {'scores': {'quality': 1., 'alignment': 1.}}])
        with patch.dict('sys.modules', {'torch': fake_torch}), patch.object(
                runner, 'make_scorer', return_value=lambda path, row: next(scores)):
            runner.run(args)
        predictions = runner.read_jsonl(args.output / 'predictions.jsonl')
        self.assertEqual([r['prediction'] for r in predictions], ['A', 'tie'])
        self.assertEqual([r['alignment_prediction'] for r in predictions], ['B', 'A'])
        summary = json.loads((args.output / 'summary.json').read_text())
        self.assertEqual(summary['ties'], 1)
        self.assertEqual(summary['pairs'], 2)
        self.assertFalse(predictions[1]['correct'])
        self.assertEqual((args.output / 'pairs.jsonl').read_bytes(), manifest.read_bytes())


if __name__ == '__main__':
    unittest.main()
