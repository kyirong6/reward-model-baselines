"""Offline checks for fresh-checkout bootstrap and pinned model downloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bootstrap_repos import bootstrap
from check_baseline import checkpoint


class PortabilityTests(unittest.TestCase):
    def test_bootstrap_pins_commit_and_preserves_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / 'upstream'
            subprocess.run(['git', 'init', str(upstream)], check=True, capture_output=True)
            (upstream / 'source.txt').write_text('original')
            subprocess.run(['git', '-C', str(upstream), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(upstream), '-c', 'user.name=Test',
                            '-c', 'user.email=test@example.invalid', 'commit', '-m', 'initial'],
                           check=True, capture_output=True)
            commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
            (root / 'repos.lock.json').write_text(json.dumps({'example': {'url': str(upstream), 'commit': commit}}))
            with self.assertRaises(RuntimeError):
                bootstrap(root, check=True)
            bootstrap(root)
            bootstrap(root, check=True)
            checkout = root / 'repos/example/source.txt'
            self.assertEqual(checkout.read_text(), 'original')
            checkout.write_text('local changes')
            with self.assertRaises(RuntimeError):
                bootstrap(root)
            self.assertEqual(checkout.read_text(), 'local changes')

    def test_download_uses_existing_revision_and_records_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            script = root / 'scripts/download_baseline.py'
            shutil.copyfile(Path(__file__).with_name('download_baseline.py'), script)
            repo = 'OpenMuQ/MuQ-large-msd-iter'
            (root / 'weights.lock.json').write_text(json.dumps({repo: 'fixed-revision'}))
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            calls = []

            def download(name, **kwargs):
                calls.append((name, kwargs))
                return str(snapshot)

            def unexpected_lookup(*args):
                raise AssertionError('A locked revision must not resolve main again')

            fake = SimpleNamespace(snapshot_download=download, model_info=unexpected_lookup)
            with patch.dict(sys.modules, {'huggingface_hub': fake}), patch.object(
                    sys, 'argv', [str(script), 'songeval']), patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(script), run_name='__main__')
            self.assertEqual(calls, [(repo, {'revision': 'fixed-revision'})])
            record = json.loads((root / 'checkpoints/songeval-downloads.json').read_text())
            self.assertEqual(record[repo]['path'], str(snapshot))
            self.assertEqual(record[repo]['revision'], 'fixed-revision')

    def test_lfs_pointer_is_not_accepted_as_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            path.write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
                checkpoint(path)


if __name__ == '__main__':
    unittest.main()
