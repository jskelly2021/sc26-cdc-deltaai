#!/bin/bash
# =============================================================================
# run_profiling_sweep.sh
# ----------------------
# SLURM job script to run the CDC reconstruction profiling sweep on DeltaAI.
#
# What this does:
#   1. Runs profile_reconstruction.py on 10 images with the baseline config
#      (65 steps, fp32) to get a detailed timing breakdown.
#   2. Runs a small batch-size pilot to test whether batch=2 is stable.
#   3. Runs sweep_steps.py across 7 step counts (5, 10, 20, 30, 50, 65, 100)
#      in both fp32 and fp16 with repeated runs.
#   4. Runs plot_results.py to generate all five plots from the sweep CSV.
#
# Submit with:
#   mkdir -p outputs/slurm
#   sbatch slurm/legacy/run_profiling_sweep.sh
#
# Monitor with:
#   squeue -u $USER
#   tail -f outputs/slurm/profiling_<jobid>.log
# =============================================================================

#SBATCH --job-name=cdc_profile_sweep
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/profiling_%j.log
#SBATCH --error=outputs/slurm/profiling_%j.log

# ── Environment setup ─────────────────────────────────────────────────────────
module load python/miniforge3_pytorch/2.10.0
conda activate base

# DeltaAI's shared conda base does not expose user-site packages by default.
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH}"

# Install the CDC-specific dependencies into the user site if they are missing.
python -m pip install --user scikit-image compressai einops lpips ema-pytorch tqdm matplotlib pandas --quiet

# ── Paths -- edit these to match your DeltaAI setup ──────────────────────────
REPO_DIR="/projects/bfod/$USER/cdc-deltaai/code"
IMG_DIR="/projects/bfod/$USER/cdc-deltaai/data/imgs"
WEIGHT_DIR="/projects/bfod/$USER/cdc-deltaai/weights"

# Checkpoint to use for profiling (choose any one checkpoint for step sweep)
CKPT="${WEIGHT_DIR}/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt"
LPIPS_WEIGHT=0.9

# Experiment structure requested for SC26:
#   - single-image sanity/profile
#   - realistic batch workload
#   - repeated runs for averaged results
FINAL_BATCH_SIZE=1
REPEATS=3

# Output directories
PROFILE_OUT="/projects/bfod/$USER/cdc-deltaai/outputs/profiling"
BATCH_PILOT_OUT="/projects/bfod/$USER/cdc-deltaai/outputs/sweep/batch_pilot"
SWEEP_OUT="/projects/bfod/$USER/cdc-deltaai/outputs/sweep/step_sweep"
PLOT_OUT="/projects/bfod/$USER/cdc-deltaai/outputs/plots"

mkdir -p outputs/slurm "${PROFILE_OUT}" "${BATCH_PILOT_OUT}" "${SWEEP_OUT}" "${PLOT_OUT}"

cd "${REPO_DIR}/xparam"

echo "=========================================="
echo "  CDC Reconstruction Profiling Sweep"
echo "  Job ID : ${SLURM_JOB_ID}"
echo "  Node   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Date   : $(date)"
echo "=========================================="

# ── Step 1: Detailed profiling (baseline = 65 steps, fp32) ───────────────────
echo ""
echo ">>> STEP 1: Profiling baseline (65 steps, fp32, 10 images)"
echo ""

python profile_reconstruction.py \
    --ckpt "${CKPT}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${PROFILE_OUT}/baseline_65steps_fp32" \
    --n_denoise_step 65 \
    --lpips_weight "${LPIPS_WEIGHT}" \
    --n_images 10 \
    --device 0

echo ""
echo ">>> STEP 1 (fp16): Profiling baseline (65 steps, fp16, 10 images)"
echo ""

python profile_reconstruction.py \
    --ckpt "${CKPT}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${PROFILE_OUT}/baseline_65steps_fp16" \
    --n_denoise_step 65 \
    --lpips_weight "${LPIPS_WEIGHT}" \
    --n_images 10 \
    --device 0 \
    --fp16

# ── Step 2: Batch-size pilot ──────────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Batch-size pilot (steps [20 65], batch sizes [1 2], fp32, repeats=2)"
echo ""

python sweep_steps.py \
    --ckpt "${CKPT}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${BATCH_PILOT_OUT}" \
    --lpips_weight "${LPIPS_WEIGHT}" \
    --n_images 4 \
    --device 0 \
    --repeats 2 \
    --batch_sizes 1 2 \
    --steps 20 65

# ── Step 3: Parameter sweep over step counts ──────────────────────────────────
echo ""
echo ">>> STEP 3: Sweeping step counts [5 10 20 30 50 65 100] x [fp32, fp16], batch=${FINAL_BATCH_SIZE}, repeats=${REPEATS}"
echo ""

python sweep_steps.py \
    --ckpt "${CKPT}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${SWEEP_OUT}" \
    --lpips_weight "${LPIPS_WEIGHT}" \
    --n_images 5 \
    --device 0 \
    --batch_size "${FINAL_BATCH_SIZE}" \
    --repeats "${REPEATS}" \
    --test_fp16 \
    --steps 5 10 20 30 50 65 100

# ── Step 4: Generate plots ────────────────────────────────────────────────────
echo ""
echo ">>> STEP 4: Generating plots"
echo ""

python plot_results.py \
    --sweep_csv "${SWEEP_OUT}/sweep_results.csv" \
    --out_dir "${PLOT_OUT}"

echo ""
echo "=========================================="
echo "  All done. Results saved to:"
echo "    Profiling : ${PROFILE_OUT}"
echo "    Batch pilot: ${BATCH_PILOT_OUT}"
echo "    Step sweep : ${SWEEP_OUT}"
echo "    Plots     : ${PLOT_OUT}"
echo "  $(date)"
echo "=========================================="
