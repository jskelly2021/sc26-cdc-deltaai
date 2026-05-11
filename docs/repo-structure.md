# Repository Structure

## Active Compression Profiling

- `xparam/profile_compression.py`: active CUDA-only context-side profiler. Times `diffusion.context_fn(images)`.
- `xparam/plot_compression_profile.py`: active aggregator and plotting script for context-only profile CSVs.
- `slurm/run_compression_profile_sweep.sh`: active DeltaAI GH200 Slurm sweep.
- `slurm/run_compression_profile_sweep_h200.sh`: active Delta H200 Slurm sweep.

## Model and Data Code

- `xparam/modules/`: core CDC model components.
- `xparam/data/`: dataset loaders and dataset-specific helpers.
- `environment.yml`: primary local environment definition.

## Secondary or Legacy Python Workflows

These scripts are retained because they may still be useful for comparison or historical reconstruction work, but they are not the default compression timing path:

- `xparam/profile_reconstruction.py`: profiles `diffusion.compress(...)` reconstruction.
- `xparam/sweep_steps.py`: reconstruction step, precision, and batch-size sweep.
- `xparam/plot_results.py`: plots reconstruction sweep CSVs.
- `xparam/evaluate_compression.py`: legacy checkpoint evaluation that saves reconstructed PNGs.
- `xparam/train.py`, `xparam/config.py`, `xparam/config_ae.py`, `xparam/test_xparam.py`: training and older development utilities.

## Legacy Slurm Scripts

Legacy or secondary Slurm jobs live under `slurm/legacy/`:

- `slurm/legacy/run_profiling_sweep.sh`
- `slurm/legacy/run_evaluation.sh`
- `slurm/legacy/run_b02048_resume.sh`
- `slurm/legacy/train.sh`

Review paths and intended outputs before running these. They may use older project roots or reconstruction semantics.

## Generated Artifacts

Generated outputs are ignored by git:

- `outputs/`
- legacy `output/`
- `logs/`
- `weights/`
- checkpoint files such as `*.pt`, `*.pth`, and `*.ckpt`
- local data directories except tracked placeholder docs

The old root README was archived as `docs/archive/root-readme-legacy.md` so previous plans and results remain available without making the root README the current source of truth.
