#!/usr/bin/env python3
"""Fetch upstream repositories at the commits in repos.lock.json."""
import json
from pathlib import Path
import subprocess
import argparse

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(root=ROOT, check=False):
    for name, entry in json.loads((root / 'repos.lock.json').read_text()).items():
        path = root / 'repos' / name
        if not path.exists():
            if check:
                raise RuntimeError(f'Missing {path}; run python3 scripts/bootstrap_repos.py')
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', entry['url'], str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', entry['commit']], check=True)
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], universal_newlines=True)
        if actual != entry['commit'] or dirty:
            raise RuntimeError(f'{path}: differs from lock or has local changes; resolve manually. Nothing overwritten.')
        if not check:
            subprocess.run(['git', '-C', str(path), 'submodule', 'update', '--init', '--recursive'], check=True)
        print(f'{name}: {actual}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify only; no downloads or changes')
    bootstrap(check=parser.parse_args().check)
