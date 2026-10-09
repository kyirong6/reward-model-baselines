#!/usr/bin/env python3
"""Download primary weights; upstream loaders may also fetch backbone assets."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', choices=['songeval', 'tunejury', 'cmi', 'musecritic'])
args = parser.parse_args()
(ROOT / 'checkpoints').mkdir(parents=True, exist_ok=True)
from huggingface_hub import snapshot_download

if args.model == 'songeval':
    snapshot_download('OpenMuQ/MuQ-large-msd-iter')
elif args.model == 'tunejury':
    snapshot_download('m-a-p/MERT-v1-330M')
    from huggingface_hub import hf_hub_download
    import shutil
    path = hf_hub_download('lukewys/laion_clap', 'music_audioset_epoch_15_esc_90.14.pt')
    shutil.copyfile(path, ROOT / 'checkpoints/clap.pt')
elif args.model == 'cmi':
    snapshot_download('HaiwenXia/CMI-RM', local_dir=ROOT / 'checkpoints/cmi')
else:
    snapshot_download('WuqnEl/MuseCritic', local_dir=ROOT / 'checkpoints/musecritic')
