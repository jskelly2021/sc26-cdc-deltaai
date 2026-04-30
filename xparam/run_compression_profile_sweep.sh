#!/bin/bash
# =============================================================================
# run_compression_profile_sweep.sh
# --------------------------------
# SLURM job script for a small repeated CDC compression / encoding profiling
# sweep on DeltaAI.
#
# Submit with:
#   cd /projects/bfod/$USER/cdc-deltaai/code
#   mkdir -p xparam/logs
#   sbatch xparam/run_compression_profile_sweep.sh
#
# Optional overrides:
#   N_IMAGES=10 START_INDEX=5 INCLUDE_B00032=1 sbatch xparam/run_compression_profile_sweep.sh
# =============================================================================

#SBATCH --job-name=cdc_compress_profile
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=xparam/logs/compression_profile_%j.log
#SBATCH --error=xparam/logs/compression_profile_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jskelly@tamu.edu

set -euo pipefail

# Environment setup
module load python/miniforge3_pytorch/2.10.0
conda activate base

# DeltaAI's shared conda base does not expose user-site packages by default.
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

# Install the CDC-specific dependencies into the user site if they are missing.
python -m pip install --user compressai einops lpips ema-pytorch tqdm matplotlib pandas --quiet

# Portable DeltaAI paths
REPO_DIR="/projects/bfod/$USER/cdc-deltaai/code"
IMG_DIR="/projects/bfod/$USER/cdc-deltaai/data/imgs"
WEIGHT_DIR="/projects/bfod/$USER/cdc-deltaai/weights"

JOB_ID="${SLURM_JOB_ID:-local}"
OUT_ROOT="/projects/bfod/$USER/cdc-deltaai/output/compression_profile/${JOB_ID}"

N_IMAGES="${N_IMAGES:-5}"
START_INDEX="${START_INDEX:-0}"
REPEATS="${REPEATS:-3}"
N_DENOISE_STEP=65
GAMMA=0.8

mkdir -p "${REPO_DIR}/xparam/logs" "${OUT_ROOT}"

cd "${REPO_DIR}/xparam"

echo "=========================================="
echo "  CDC Compression Profiling Sweep"
echo "  Job ID      : ${JOB_ID}"
echo "  Node        : $(hostname)"
echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Date        : $(date)"
echo "  Images/run  : ${N_IMAGES}"
echo "  Repeats     : ${REPEATS}"
echo "  Steps       : ${N_DENOISE_STEP}"
echo "  Gamma       : ${GAMMA}"
echo "  Output root : ${OUT_ROOT}"
echo "=========================================="

LABELS=("b0.0128" "b0.2048")
LPIPS_WEIGHTS=("0.0" "0.9")
CKPTS=(
    "${WEIGHT_DIR}/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.0128-x-cosine-01-float32-aux0.0_2.pt"
    "${WEIGHT_DIR}/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt"
)

if [[ "${INCLUDE_B00032:-0}" == "1" ]]; then
    LABELS=("b0.0032" "${LABELS[@]}")
    LPIPS_WEIGHTS=("0.0" "${LPIPS_WEIGHTS[@]}")
    CKPTS=(
        "${WEIGHT_DIR}/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt"
        "${CKPTS[@]}"
    )
fi

for idx in "${!LABELS[@]}"; do
    LABEL="${LABELS[$idx]}"
    LPIPS_WEIGHT="${LPIPS_WEIGHTS[$idx]}"
    CKPT="${CKPTS[$idx]}"

    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: checkpoint not found: ${CKPT}" >&2
        exit 1
    fi

    for repeat in $(seq 1 "${REPEATS}"); do
        REPEAT_PADDED=$(printf "%02d" "${repeat}")
        OUT_DIR="${OUT_ROOT}/${LABEL}/repeat_${REPEAT_PADDED}"
        mkdir -p "${OUT_DIR}"

        echo ""
        echo ">>> Profiling ${LABEL}, repeat ${REPEAT_PADDED}/${REPEATS}, lpips_weight=${LPIPS_WEIGHT}"
        echo "    Output: ${OUT_DIR}"

        python profile_compression.py \
            --ckpt "${CKPT}" \
            --checkpoint_label "${LABEL}" \
            --img_dir "${IMG_DIR}" \
            --out_dir "${OUT_DIR}" \
            --gamma "${GAMMA}" \
            --n_denoise_step "${N_DENOISE_STEP}" \
            --device 0 \
            --lpips_weight "${LPIPS_WEIGHT}" \
            --n_images "${N_IMAGES}" \
            --start_index "${START_INDEX}" \
            --repeat "${repeat}"
    done
done

echo ""
echo ">>> Generating compression profile summary and plots"

python plot_compression_profile.py \
    --profile_dir "${OUT_ROOT}" \
    --out_dir "${OUT_ROOT}"

echo ""
echo "=========================================="
echo "  Compression profiling complete."
echo "  Results: ${OUT_ROOT}"
echo "  Summary: ${OUT_ROOT}/compression_profile_summary.csv"
echo "  Plots  : ${OUT_ROOT}/plots"
echo "  $(date)"
echo "=========================================="
