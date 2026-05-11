# CDC Compression Profiling

Current focus: reproducible inference-time profiling for CDC learned image compression.

The active compression-side workflow measures:

```text
diffusion.context_fn(images)
```

This is the learned encoder, hyperprior/BPP computation, quantization, and context feature generation path. It is not full reconstruction. `diffusion.compress()` runs reconstruction through `p_sample_loop()` and is treated as a legacy or secondary workflow unless explicitly needed.

Updated: 2026-05-11

## Quick Start

Create the local environment from the tracked file:

```bash
conda env create -f environment.yml
conda activate exp_pytorch
```

Run a small context-only compression profile inside a GPU allocation:

```bash
python xparam/profile_compression.py \
  --ckpt /path/to/checkpoint.pt \
  --checkpoint_label b0.2048 \
  --img_dir /path/to/imgs \
  --out_dir outputs/compression_profile/local/b0.2048/repeat_01 \
  --lpips_weight 0.9 \
  --n_images 5 \
  --repeat 1
```

Summarize a sweep:

```bash
python xparam/plot_compression_profile.py \
  --profile_dir outputs/compression_profile/<job_id> \
  --out_dir outputs/compression_profile/<job_id>
```

## Slurm Entry Points

Active Slurm scripts live in `slurm/`.

DeltaAI GH200:

```bash
mkdir -p outputs/slurm
sbatch slurm/run_compression_profile_sweep.sh
```

Delta H200:

```bash
mkdir -p outputs/slurm
sbatch slurm/run_compression_profile_sweep_h200.sh
```

Common overrides:

```bash
N_IMAGES=10 START_INDEX=5 REPEATS=1 \
CKPT_DIR=/path/to/x_param IMG_DIR=/path/to/imgs \
sbatch slurm/run_compression_profile_sweep.sh
```

By default, active job outputs go under `outputs/compression_profile/<job_id>/`, and Slurm stdout/stderr goes under `outputs/slurm/`.

## Active Files

- `xparam/profile_compression.py`: CUDA-only context-side profiler. Writes `compression_profile_results.csv` and `compression_profile_report.txt`.
- `xparam/plot_compression_profile.py`: aggregates context-only CSVs and writes `compression_profile_summary.csv` plus plots.
- `slurm/run_compression_profile_sweep.sh`: DeltaAI GH200 context-only sweep.
- `slurm/run_compression_profile_sweep_h200.sh`: Delta H200 context-only sweep.

## Documentation

- `docs/current-experiments.md`: current experiment scope and outputs.
- `docs/hpc-usage.md`: cluster usage, submit commands, and path conventions.
- `docs/repo-structure.md`: active vs legacy files.
- `docs/archive/root-readme-legacy.md`: old root README with prior plans, notes, and results.

## Notes

Checkpoint `.pt` files, datasets, generated profiles, plots, logs, and reconstructed images are artifacts and should not be committed. Keep checkpoint labels and `--lpips_weight` paired explicitly with the checkpoint family.
