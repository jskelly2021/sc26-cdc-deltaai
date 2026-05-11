# HPC Usage

Active Slurm scripts live in `slurm/`. Submit from the repository root so relative log paths resolve consistently.

## DeltaAI GH200

```bash
cd /projects/bfod/$USER/sc26-cdc-deltaai
mkdir -p outputs/slurm
sbatch slurm/run_compression_profile_sweep.sh
```

Defaults:

- Repo: `/projects/bfod/$USER/sc26-cdc-deltaai`
- Images: `${REPO_DIR}/data/imgs`
- Checkpoints: `${REPO_DIR}/weights/x_param`
- Outputs: `${REPO_DIR}/outputs/compression_profile/<job_id>`
- Slurm logs: `${REPO_DIR}/outputs/slurm/compression_profile_<job_id>.log`

## Delta H200

```bash
cd /projects/bfod/$USER/sc26-cdc-delta
mkdir -p outputs/slurm
sbatch slurm/run_compression_profile_sweep_h200.sh
```

Defaults:

- Repo: `/projects/bfod/$USER/sc26-cdc-delta`
- Images: `${REPO_DIR}/data/imgs`
- Checkpoints: `${REPO_DIR}/weights/x_param`
- Outputs: `${REPO_DIR}/outputs/compression_profile/<job_id>`
- Slurm logs: `${REPO_DIR}/outputs/slurm/compression_profile_h200_<job_id>.log`

## Overrides

Use environment variables at submit time to keep data, checkpoint, and output paths explicit:

```bash
N_IMAGES=10 START_INDEX=5 REPEATS=1 \
REPO_DIR=/path/to/repo \
IMG_DIR=/path/to/imgs \
CKPT_DIR=/path/to/x_param \
OUT_ROOT=/path/to/outputs/compression_profile/test_run \
sbatch slurm/run_compression_profile_sweep.sh
```

Useful variables:

- `REPO_DIR`: repository clone used by the job.
- `IMG_DIR`: directory of input `.jpg`, `.jpeg`, or `.png` images.
- `WEIGHT_DIR`: parent weights directory.
- `CKPT_DIR`: directory containing checkpoint `.pt` files.
- `OUT_ROOT`: sweep output root.
- `N_IMAGES`: number of images per checkpoint/repeat.
- `START_INDEX`: zero-based starting image index.
- `REPEATS`: number of repeats per checkpoint.

## Manual Local GPU Run

Inside an interactive GPU allocation:

```bash
python xparam/profile_compression.py \
  --ckpt /path/to/checkpoint.pt \
  --checkpoint_label b0.2048 \
  --img_dir /path/to/imgs \
  --out_dir outputs/compression_profile/local/b0.2048/repeat_01 \
  --device 0 \
  --lpips_weight 0.9 \
  --n_images 5 \
  --start_index 0 \
  --repeat 1
```

Then summarize:

```bash
python xparam/plot_compression_profile.py \
  --profile_dir outputs/compression_profile/local \
  --out_dir outputs/compression_profile/local
```

## Cluster Notes

Do not change account names, partitions, module stacks, hard-coded project roots, or notification emails unless that is the requested task. Slurm creates stdout/stderr files before the script body runs, so `outputs/slurm` must exist before `sbatch`.
