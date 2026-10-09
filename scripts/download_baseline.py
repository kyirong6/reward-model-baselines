#!/usr/bin/env python3
"""Download primary weights; upstream loaders may also fetch backbone assets."""
import argparse
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', choices=['songeval', 'tunejury', 'cmi', 'musecritic'])
parser.add_argument('--lock-only', action='store_true', help='Resolve revisions without downloading weights')
args = parser.parse_args()
(ROOT / 'checkpoints').mkdir(parents=True, exist_ok=True)
from huggingface_hub import snapshot_download, model_info

LOCK = ROOT / 'weights.lock.json'
revisions = json.loads(LOCK.read_text()) if LOCK.exists() else {}
repositories = {
    'songeval': ['OpenMuQ/MuQ-large-msd-iter'],
    'tunejury': ['m-a-p/MERT-v1-330M', 'lukewys/laion_clap'],
    'cmi': ['HaiwenXia/CMI-RM'],
    'musecritic': ['WuqnEl/MuseCritic'],
}[args.model]
# Run downloads serially: this shared lock is updated before downloading.
for repo in repositories:
    if repo not in revisions:
        revisions[repo] = model_info(repo).sha
LOCK.write_text(json.dumps(revisions, indent=2, sort_keys=True) + '\n')
if args.lock_only:
    print(f'Revisions saved to {LOCK}; commit this file before sharing.')
    raise SystemExit(0)

downloaded = {}


def snapshot(repo, **kwargs):
    path = snapshot_download(repo, revision=revisions[repo], **kwargs)
    downloaded[repo] = {'revision': revisions[repo], 'path': str(Path(path).resolve())}
    return path

if args.model == 'songeval':
    snapshot('OpenMuQ/MuQ-large-msd-iter')
elif args.model == 'tunejury':
    snapshot('m-a-p/MERT-v1-330M')
    from huggingface_hub import hf_hub_download
    import shutil
    path = hf_hub_download('lukewys/laion_clap', 'music_audioset_epoch_15_esc_90.14.pt',
                          revision=revisions['lukewys/laion_clap'])
    shutil.copyfile(path, ROOT / 'checkpoints/clap.pt')
    downloaded['lukewys/laion_clap'] = {'revision': revisions['lukewys/laion_clap'],
                                     'path': str(ROOT / 'checkpoints/clap.pt')}
elif args.model == 'cmi':
    snapshot('HaiwenXia/CMI-RM', local_dir=ROOT / 'checkpoints/cmi')
else:
    snapshot('WuqnEl/MuseCritic', local_dir=ROOT / 'checkpoints/musecritic')
(ROOT / 'checkpoints' / (args.model + '-downloads.json')).write_text(
    json.dumps(downloaded, indent=2) + '\n')
print('Primary downloads complete. CMI may fetch additional backbones during model loading.')
