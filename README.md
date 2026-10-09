# Music baseline smoke tests on Slurm

**Running on another cluster? Start with [PORTABILITY.md](PORTABILITY.md)** for
fresh-checkout setup, portable test bundles, reproducibility, and GitHub publishing.

Run SongEval, TuneJury, CMI-RM, and MuseCritic on the same local audio pairs.
Start with one pair, then ten, then all 618. A one-pair result checks execution,
not model accuracy. Upstream repositories are left unchanged.

## Current state

`data/processed/smoke/pairs.jsonl` contains one reproducibly selected vocal pair,
with absolute audio paths and its reference preference. All 1,236 paths in the
source dataset were checked. Model environments still need dependencies installed;
CMI-RM and MuseCritic need their checkpoints downloaded. No GPU inference has
been performed by this setup.

Run the commands below from this directory:

```bash
cd /path/to/baselines
```

## 1. Install and download one model at a time

Use a node with internet access where your cluster permits installations.
The scripts use the local Miniforge and keep environments and caches here.
Installation is separate from the timed GPU jobs.

Start with SongEval:

```bash
bash scripts/setup_baseline.sh songeval
envs/songeval/bin/python scripts/download_baseline.py songeval
```

Repeat with `tunejury`, `cmi`, and `musecritic` when ready. Each has its own
`envs/MODEL` environment, and resolved packages are saved to
`envs/MODEL-installed.txt`. The setup script uses upstream dependency pins;
installation compatibility has not yet been verified on the cluster.

MuseCritic's upstream requirements include a compiled `flash_attn` dependency.
Its setup may require your cluster's CUDA toolkit/compiler module and an
appropriate build node. The script installs PyTorch first; if compilation fails,
check `nvcc --version` and the available CUDA modules before retrying.

If a model download requires authentication/access acceptance, authenticate in
the same environment with `HF_HOME` pointing to this workspace's
`cache/huggingface`. Do not put tokens into job scripts.

The download helper fetches primary weights. CMI's configuration can reference
additional encoder/tokenizer models, which its loader downloads on first use.
If compute nodes have no internet access, populate those caches beforehand from
an allowed node (a CPU load can take substantial RAM):

```bash
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
envs/cmi/bin/python - <<'PY'
import sys
sys.path.insert(0, 'repos/CMI-RewardBench')
from baselines.inference import RewardModelInference
RewardModelInference('checkpoints/cmi/model.safetensors', device='cpu', bf16=False)
PY
```

## 2. Submit the first GPU test

Slurm is installed. Local configuration lists L40S GPUs on node028 in `short`
and `medium`, and on node027 in `impa`. This does not verify current availability
or your account permissions. From a terminal able to reach the controller:

```bash
sinfo -o '%P %a %l %G'
```

The job script uses the cluster's default partition, one `gpu:1`, 4 CPUs, 64 GB host RAM,
and 2 hours. These are starting requests, not measured requirements. L40S avoids
the older GPU architectures also present on this cluster, and supports the
bfloat16 path used by MuseCritic. Host RAM (`--mem`) is separate from GPU memory.

```bash
mkdir -p logs
sbatch --job-name=smoke-songeval slurm/baseline.sbatch songeval
```

Override partition/account/QoS if your allocation requires it. For example,
replace `YOUR_ACCOUNT` with your actual account:

```bash
sbatch --partition=medium --account=YOUR_ACCOUNT --time=04:00:00 \
  --job-name=smoke-songeval slurm/baseline.sbatch songeval
```

Once the corresponding installations/downloads are ready:

```bash
sbatch --job-name=smoke-tunejury slurm/baseline.sbatch tunejury
sbatch --job-name=smoke-cmi slurm/baseline.sbatch cmi
sbatch --job-name=smoke-musecritic slurm/baseline.sbatch musecritic
```

Each submission runs one model in one process, loading weights once and scoring
both tracks sequentially. Slurm controls GPU visibility. Model initialization
or inference errors cause a failed job; they are not silently skipped.

```bash
squeue -u "$USER"
# Replace JOB_ID with the ID printed by sbatch:
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Logs go to `logs/smoke-MODEL-JOB_ID.out` and `.err`. Results go to
`results/smoke/MODEL-JOB_ID/`:

- `pairs.jsonl`: exact manifest used, including reference labels.
- `run.json`: settings, PyTorch version, GPU name, and job ID.
- `predictions.jsonl`: A/B scores, prediction, reference preference, timing;
  MuseCritic also includes critiques.
- `summary.json`: accuracy, tie count, vocal/instrumental breakdown, model load
  time, total time, and peak PyTorch allocated GPU memory (not total device usage).

A missing summary means the run did not complete. Completed pairs are flushed
to disk so partial outputs remain available for diagnosis. Output directories
are never overwritten; failed jobs should be resubmitted with a new job ID.

## 3. Scoring protocol

All models receive full tracks with their own native preprocessing. CMI's
default 30-second crop is explicitly disabled. No model receives preference
labels or human feedback. TuneJury receives the original prompt; CMI receives
the original prompt and available lyrics. MuseCritic receives the upstream
evaluation rubric and audio, excluding the example's reference critique/scores.

Primary pairwise predictions use higher TuneJury `reward`, CMI `quality`, or
SongEval/MuseCritic `Musicality`. All score dimensions are retained. CMI alignment
predictions/accuracy are reported separately. The generic human preference may
reflect more than musicality; this protocol is an initial baseline, not a claim
that the dimensions are equivalent. Exact score ties count as incorrect and
are counted separately. Compare preference accuracy across models, not their
raw score magnitudes. The full dataset's majority-B baseline is 344/618 (55.7%).

For a separate TuneJury empty-prompt experiment:

```bash
sbatch --job-name=smoke-tunejury-empty slurm/baseline.sbatch tunejury --empty-prompt
```

## 4. Increase the subset

Selection uses a fixed seed and alternates vocal and instrumental pairs. A
single pair is vocal; ten pairs include five of each. Existing manifests are
never overwritten to protect queued runs.

```bash
envs/tunejury/bin/python scripts/baseline_smoke.py prepare --count 10 \
  --output data/processed/smoke/pairs-10.jsonl
sbatch --job-name=test10-songeval slurm/baseline.sbatch songeval \
  --manifest data/processed/smoke/pairs-10.jsonl
```

Use `--count 0 --output data/processed/pairs-all.jsonl` for the full dataset.
Estimate wall time from the ten-pair job and request an appropriate partition
and time limit before scaling up. The current runner processes pairs serially
and does not resume partial jobs automatically.

## CPU-only validation

These checks need only Python's standard library, so the existing TuneJury
Python interpreter works before dependencies are installed:

```bash
envs/tunejury/bin/python scripts/baseline_smoke.py run --model songeval \
  --output /tmp/unused-baseline-output --validate-only
envs/tunejury/bin/python scripts/test_baseline_smoke.py
bash -n scripts/setup_baseline.sh slurm/baseline.sbatch
```

`--validate-only` checks the manifest and paths; it does not import or load a
model. Inference outside Slurm is blocked unless `--allow-local` is explicitly
provided. Actual dependency, checkpoint, CUDA, and model compatibility must be
verified with the first GPU job.
