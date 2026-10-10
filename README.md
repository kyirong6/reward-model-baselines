# Music reward-model baselines

Test SongEval, TuneJury, CMI-RM, and MuseCritic on the included audio pairs.

Requires Linux, Git, Conda/Miniforge, and an NVIDIA GPU. Setup needs internet.

## Setup and test

```bash
git clone https://github.com/kyirong6/reward-model-baselines.git
cd reward-model-baselines
bash scripts/setup.sh songeval
bash scripts/test.sh songeval
```

Setup installs dependencies and downloads weights. The test runs one audio pair.
Replace `songeval` with `tunejury`, `cmi`, `musecritic`, or `all`.

To test all ten included pairs:

```bash
bash scripts/test.sh songeval --count 10
```

Results are saved in `results/MODEL-TIMESTAMP/`. Open `summary.json` for accuracy
and timing, or `predictions.jsonl` for individual scores.

Each model scores both audios separately. Preference is determined by the higher
unweighted mean of all returned scores (TuneJury's single reward is unchanged).
Exact ties count as incorrect. Predictions include the individual scores and
their mean; CMI alignment accuracy is also reported separately.

To combine all completed runs (CPU only; Python standard library):

```bash
envs/songeval/bin/python scripts/consolidate_results.py
```

This regenerates `results/combined_predictions.jsonl` with one record per
model/run/pair, retaining dimension scores, critiques, decisions, and pair inputs.
Each record includes `model`, `run_id`, `source_directory`, `primary_score`, and
`tie_policy`. Missing legacy means are calculated and marked with
`mean_score_source: "derived"`; existing means are marked `"saved"`. Recorded
predictions and accuracy are preserved even when an older run used another rule.

`results/combined_summary.json` groups run metadata and summaries under
`models[MODEL][RUN_ID]`. Repeated runs are retained separately. Runs without a
`summary.json` are skipped and listed in `skipped_incomplete_runs`. Original run
files are preserved. Use `--results-dir PATH` and optionally `--output-dir PATH`
to read another checkout or write the combined files elsewhere. Regenerate after
new runs finish; compare model accuracy on matching pair sets and scoring rules.

## On a Slurm cluster

After setup, submit from the repository directory:

```bash
sbatch --partition=YOUR_PARTITION --gres=gpu:1 slurm/baseline.sbatch songeval --count 10
```

Adjust the partition and GPU request for your cluster.

If Conda is not on PATH, set `CONDA_EXE=/path/to/miniforge3/bin/conda` before setup.
MuseCritic setup reuses CUDA 12.8 or installs it locally; tests require bfloat16 GPU support.
CMI may need internet on its first run to download additional model components.
