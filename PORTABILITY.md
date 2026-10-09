# Run on another machine

`repos.lock.json` records exact commits for the four upstream repositories.
`repos/` is ignored by the parent Git repository; the bootstrap script fetches
it. This workflow does not need submodules. Existing changes or different
commits cause an error instead of being overwritten.

## Fresh checkout

Prerequisites: Linux, Git, Python 3 for bootstrap, and Conda/Miniforge.
Inference requires an allocated NVIDIA GPU and a compatible driver.
MuseCritic also requires CUDA build tools (`nvcc`) for flash-attn; load your
cluster's CUDA module if needed. Install/download on an internet-connected node.

```bash
git clone YOUR_REPOSITORY_URL baselines
cd baselines
python3 scripts/bootstrap_repos.py

# Omit this export if conda is on PATH or under tools/miniforge3.
export CONDA_EXE=/path/to/miniforge3/bin/conda
bash scripts/setup_baseline.sh songeval
envs/songeval/bin/python scripts/download_baseline.py songeval
```

Repeat setup/download for `tunejury`, `cmi`, or `musecritic` as needed.
Recreate environments; do not copy `envs/` between machines. If a checkpoint
is a Git LFS pointer, install Git LFS and run `git lfs pull` in that upstream repo.

## Transfer test data

This workspace has prepared bundles at `bundles/smoke-1` and `bundles/smoke-10`.
They contain selected audio, prompts, labels, relative paths, and SHA-256 audio
checksums. Bundles are excluded from Git; transfer one separately, for example:

```bash
rsync -av bundles/smoke-10/ USER@HOST:/path/to/baselines/bundles/smoke-10/
```

To make another bundle on the machine holding the source dataset:

```bash
envs/songeval/bin/python scripts/baseline_smoke.py prepare \
  --source data/raw/annotations_with_pref.jsonl --count 10 --seed 42 \
  --output data/processed/transfer-10.jsonl
envs/songeval/bin/python scripts/baseline_smoke.py bundle \
  --manifest data/processed/transfer-10.jsonl --output bundles/new-smoke-10
```

Choose new output names; existing manifests/bundles are never overwritten.
Selection is reproducible for the same source rows, ordering, count and seed.
A bundle can move anywhere: audio paths resolve relative to its `pairs.jsonl`.
Checksums are verified before inference. Absolute-path manifests still work.

## Check and run

Validate paths and audio checksums without importing model dependencies:

```bash
envs/songeval/bin/python scripts/baseline_smoke.py run --model songeval \
  --manifest bundles/smoke-1/pairs.jsonl --output /tmp/unused --validate-only
```

After setup/download, check imports, checkpoints and upstream commits:

```bash
envs/songeval/bin/python scripts/check_baseline.py --model songeval \
  --manifest bundles/smoke-1/pairs.jsonl
```

Add `--gpu` on an allocated GPU node to check CUDA/GPU support. Preflight does
not load the model or prove that every transitive asset has been cached.

For Slurm, submit from the repository root using your cluster's resource names:

```bash
mkdir -p logs
sbatch --partition=YOUR_PARTITION --gres=gpu:1 \
  slurm/baseline.sbatch songeval --manifest bundles/smoke-1/pairs.jsonl
```

The template requests one generic GPU, 4 CPUs, 64 GB RAM and 2 hours; it does
not select a partition/account. Override these for your cluster.
For an already allocated GPU, including a machine without Slurm:

```bash
bash scripts/run_baseline.sh songeval \
  --manifest bundles/smoke-1/pairs.jsonl --output results/songeval-test-1
```

The wrapper sets cache paths and verifies upstream commits without downloads.
Start with one pair, then use the ten-pair bundle and a new output directory.
Share the entire result directory, including `summary.json`, `predictions.jsonl`,
`packages.txt` and `run.json`. `pairs.jsonl` is the unchanged input manifest;
`resolved-pairs.jsonl` records absolute paths on the execution machine.

## Reproduce an experiment

* Upstream commits are pinned. Runs record actual commits, runner source hashes,
  input/audio hashes, GPU, PyTorch version and installed Python packages.
* `weights.lock.json` pins explicitly downloaded Hugging Face repositories.
  The SongEval encoder is already pinned from the local cache. Other entries
  resolve once before the first download. Run download commands serially and
  **commit/share the updated lock** before another machine downloads. Use
  `download_baseline.py MODEL --lock-only` to resolve without downloading.
  SongEval/TuneJury use downloaded encoder snapshots directly; their head
  checkpoints come from pinned upstream Git commits.
* Some upstream Python requirements remain unpinned. After a successful GPU
  run, save `packages.txt` in a tracked location such as
  `requirements/songeval-tested.txt`. Replay on a compatible system with:

  ```bash
  bash scripts/setup_baseline.sh songeval requirements/songeval-tested.txt
  ```

  Setup also saves `envs/MODEL-installed.txt` and
  `envs/MODEL-conda-explicit.txt`. Conda artifacts are platform-specific.
  If a saved PyTorch wheel uses another CUDA build, set `PYTORCH_INDEX_URL`
  to its matching PyTorch wheel index when replaying the package lock.
  Python/CUDA/compiler compatibility still matters, and identical package
  versions do not guarantee bit-identical GPU outputs. Compiled dependencies
  may still require build-tool preparation when replaying a package lock.
* CMI's upstream loader can fetch additional encoders/tokenizers that the
  primary download command does not pin or prefetch. Warm that loader on an
  internet-connected node (see README) and preserve the resulting cache for
  the experiment. For offline compute, first cache all assets, then set
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so missing assets fail clearly.

The portability changes have not been validated with a fresh dependency
installation or GPU inference. CPU tests cover relocation, checksums, selection
and scoring bookkeeping. Use the first one-pair GPU run to validate compatibility.

## Publish to GitHub

Run from your normal terminal: this agent session exposes the parent `.git`
directory as read-only. After creating an empty GitHub repository:

```bash
git init -b main
git add .gitignore README.md PORTABILITY.md repos.lock.json weights.lock.json scripts/ slurm/
git diff --cached --stat
git commit -m "Add portable baseline evaluation workflow"
git remote add origin git@github.com:YOUR_USERNAME/baselines.git
git push -u origin main
```

Only scripts, documentation and locks are published. The bootstrap fetches
upstream code; transfer test bundles separately. Add tested dependency locks
to the parent repository once you have a successful run.
