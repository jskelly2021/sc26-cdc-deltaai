# AGENTS.md

## Purpose
This repo supports SC26 / CDC learned image compression profiling. Agents should optimize for reproducible context-only compression measurements, small code changes, and preserving the scientific meaning of timing and bitrate fields.

## Source of Truth
- Priority order:
  1. Current source files and scripts.
  2. Current Slurm scripts and CLI arguments.
  3. `docs/current-experiments.md`, `docs/hpc-usage.md`, and `docs/repo-structure.md`.
  4. Archived notes under `docs/archive/` as historical context only.

## Project Map
- `xparam/profile_compression.py`: active CUDA-only context-side profiler; times `diffusion.context_fn(images)` and writes `compression_profile_results.csv` plus `compression_profile_report.txt`.
- `xparam/plot_compression_profile.py`: active context-only aggregator; writes `compression_profile_summary.csv` plus plots.
- `slurm/run_compression_profile_sweep.sh`: active DeltaAI GH200 context-only compression sweep.
- `slurm/run_compression_profile_sweep_h200.sh`: active Delta H200 context-only compression sweep.
- `xparam/modules/`: core model code, including `ResnetCompressor`, `GaussianDiffusion`, `Unet`, trainer utilities, and network components.
- `xparam/data/`: dataset loaders and dataset-specific helpers.
- `slurm/legacy/`: legacy or secondary Slurm jobs for reconstruction, evaluation, and training.
- `xparam/profile_reconstruction.py`, `xparam/sweep_steps.py`, `xparam/plot_results.py`, `xparam/evaluate_compression.py`: secondary reconstruction/evaluation scripts using `diffusion.compress(...)`; not the default compression timing path.
- `data/`: local data docs/placeholders; actual datasets may live outside the clone.
- `outputs/`, legacy `output/`, `logs/`, `weights/`, checkpoints, reconstructed images, generated plots, and generated CSVs are artifacts, not source.

## Setup and Environment
- Primary environment file: `environment.yml` (`exp_pytorch`, Python 3.9, PyTorch 2.0/CUDA 11.8, CompressAI, LPIPS, EMA, scikit-image, pandas, matplotlib).
- DeltaAI active Slurm script loads `python/miniforge3_pytorch/2.10.0`, activates `base`, and installs missing user-site packages with `python -m pip install --user ...`.
- Delta H200 active Slurm script loads `cudatoolkit/25.3_12.8` and `pytorch-conda/2.8`, then activates `/projects/bfod/jkelly5/envs/cdc-delta/bin/activate`.
- Profiling scripts require CUDA and fail fast when no GPU is visible.
- Model checkpoints are expected as `.pt` files containing an `ema` state dict. Match `--lpips_weight` to the checkpoint family (`0.0` for non-LPIPS, `0.9` for LPIPS checkpoints).

## Common Commands
- Create the conda environment from the tracked file:
  `conda env create -f environment.yml`
- Run a local context-only compression profile inside a GPU allocation:
  `python xparam/profile_compression.py --ckpt /path/to/checkpoint.pt --checkpoint_label b0.2048 --img_dir /path/to/imgs --out_dir outputs/compression_profile/local/b0.2048/repeat_01 --lpips_weight 0.9 --n_images 5 --repeat 1`
- Submit the DeltaAI context-only compression sweep:
  `mkdir -p outputs/slurm && sbatch slurm/run_compression_profile_sweep.sh`
- Submit the Delta H200 context-only compression sweep:
  `mkdir -p outputs/slurm && sbatch slurm/run_compression_profile_sweep_h200.sh`
- Override sweep inputs at submit time:
  `N_IMAGES=10 START_INDEX=5 REPEATS=1 CKPT_DIR=/path/to/x_param IMG_DIR=/path/to/imgs sbatch slurm/run_compression_profile_sweep.sh`
- Summarize/plot context-only compression profile outputs:
  `python xparam/plot_compression_profile.py --profile_dir outputs/compression_profile/<job_id> --out_dir outputs/compression_profile/<job_id>`
- Formal test/lint commands are not currently defined in this repo.

## Coding Rules
- Propose a short implementation plan before editing code.
- Keep diffs small and avoid broad rewrites unless explicitly requested.
- Preserve CLI compatibility and output schemas unless the task asks to change them.
- Avoid changing model architecture, checkpoint loading, sampling schedules, BPP math, or metric definitions casually.
- Prefer explicit paths/arguments over hidden assumptions.
- Keep profiling field names stable, especially CSV/report columns consumed by plot scripts.
- When changing imports, keep scripts runnable from the repo root and from `xparam/` where current scripts support that.

## Experiment / HPC Rules
- Do not commit generated experiment output, logs, reconstructed images, plots, checkpoints, weights, datasets, or large artifacts.
- Respect `.gitignore`: `outputs/`, legacy `output/`, `logs/`, `weights/`, `*.pt`, `*.pth`, `*.ckpt`, `__pycache__/`, and local data folders are ignored for a reason.
- Do not modify cluster/account-specific Slurm settings (`#SBATCH --account`, partitions, modules, hard-coded project paths, email addresses) unless the task explicitly asks for it.
- Slurm writes stdout/stderr before the script body runs; create `outputs/slurm` before `sbatch`.
- Preserve reproducibility fields in CSV/report outputs: checkpoint path/label, repeat, image name, dimensions, LPIPS weight, device/timing/memory, BPP, and status/error fields.
- Distinguish context-only compression profiling (`context_fn`, no diffusion sampling) from end-to-end diffusion reconstruction (`diffusion.compress`, sampling loop).

## Workflow for Agents
- Read the relevant files before editing; prefer scripts and source over prose docs.
- State a short plan before code changes.
- Implement the smallest useful change.
- Run the narrowest relevant check available. If GPU/checkpoint/data are unavailable, say so explicitly.
- Summarize what changed and what was not tested.

## Boundaries

### Always
- Use current source and script CLIs as the authority.
- Preserve generated-output formats unless asked to migrate them.
- Keep paths, checkpoint labels, and LPIPS/checkpoint pairing explicit.
- Check `git status` before and after edits; do not disturb unrelated work.

### Ask First
- Changing scientific semantics, metric formulas, model architecture, checkpoint conventions, or output schemas.
- Editing Slurm accounts, partitions, module stacks, project-root paths, or notification emails.
- Adding new dependencies or replacing the environment strategy.
- Running long GPU jobs, large sweeps, training jobs, or commands that may create large outputs.

### Never
- Commit outputs, logs, checkpoints, weights, datasets, reconstructed images, generated plots, or large artifacts.
- Treat archived docs as authoritative when they conflict with current source or scripts.
- Rewrite broad parts of `xparam/modules/` for style-only reasons.
- Hide fallback paths or silent defaults that make experiments hard to reproduce.
- Revert or overwrite user changes unless explicitly requested.
