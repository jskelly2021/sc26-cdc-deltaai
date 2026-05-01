#!/bin/bash
# =============================================================================
# run_compression_profile_sweep.sh
# --------------------------------
# SLURM job script for a small repeated CDC context-only compression profiling
# sweep on DeltaAI. This measures diffusion.context_fn(images), not end-to-end
# diffusion reconstruction.
#
# Submit with:
#   cd /projects/bfod/$USER/sc26-cdc-deltaai
#   mkdir -p xparam/logs
#   sbatch xparam/run_compression_profile_sweep.sh
#
# Or, from inside xparam:
#   sbatch run_compression_profile_sweep.sh
#
# Optional overrides:
#   N_IMAGES=10 START_INDEX=5 sbatch xparam/run_compression_profile_sweep.sh
#   N_IMAGES=10 START_INDEX=5 REPEATS=1 sbatch run_compression_profile_sweep.sh
# =============================================================================

#SBATCH --job-name=cdc_compress_profile
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/bfod/%u/sc26-cdc-deltaai/xparam/logs/compression_profile_%j.log
#SBATCH --error=/projects/bfod/%u/sc26-cdc-deltaai/xparam/logs/compression_profile_%j.log

set -euo pipefail

# Environment setup
module load python/miniforge3_pytorch/2.10.0
conda activate base

# DeltaAI's shared conda base does not expose user-site packages by default.
export PYTHONPATH="$HOME/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

# Install the CDC-specific dependencies into the user site if they are missing.
python -m pip install --user compressai einops lpips ema-pytorch tqdm matplotlib pandas --quiet

# Portable DeltaAI paths for the cloned SC26 CDC repo.
# Override IMG_DIR, CKPT_DIR, WEIGHT_DIR, or OUT_ROOT at submit time if your
# data/weights live outside the clone, e.g. CKPT_DIR=/path/to/x_param sbatch ...
REPO_DIR="${REPO_DIR:-/projects/bfod/$USER/sc26-cdc-deltaai}"
IMG_DIR="${IMG_DIR:-${REPO_DIR}/data/imgs}"
WEIGHT_DIR="${WEIGHT_DIR:-${REPO_DIR}/weights}"
CKPT_DIR="${CKPT_DIR:-${WEIGHT_DIR}/x_param}"

if [[ ! -d "${IMG_DIR}" && -d "${REPO_DIR}/imgs" ]]; then
    IMG_DIR="${REPO_DIR}/imgs"
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
    for candidate in \
        "${WEIGHT_DIR}" \
        "${REPO_DIR}/weights/x_param" \
        "${REPO_DIR}/weights" \
        "${REPO_DIR}/results/x_param" \
        "${REPO_DIR}/results"
    do
        if [[ -d "${candidate}" ]] && compgen -G "${candidate}/*.pt" > /dev/null; then
            CKPT_DIR="${candidate}"
            break
        fi
    done
fi

JOB_ID="${SLURM_JOB_ID:-local}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/output/compression_profile/${JOB_ID}}"

N_IMAGES="${N_IMAGES:-5}"
START_INDEX="${START_INDEX:-0}"
REPEATS="${REPEATS:-3}"

mkdir -p "${REPO_DIR}/xparam/logs" "${OUT_ROOT}"

if [[ ! -d "${REPO_DIR}/xparam" ]]; then
    echo "ERROR: xparam directory not found under REPO_DIR=${REPO_DIR}" >&2
    exit 1
fi

if [[ ! -d "${IMG_DIR}" ]]; then
    echo "ERROR: image directory not found: ${IMG_DIR}" >&2
    echo "Set IMG_DIR=/path/to/imgs when submitting if your images live elsewhere." >&2
    exit 1
fi

cd "${REPO_DIR}/xparam"

echo "=========================================="
echo "  CDC Context-Only Compression Profiling Sweep"
echo "  Measures    : diffusion.context_fn(images)"
echo "  Excludes    : diffusion reconstruction / p_sample_loop"
echo "  Job ID      : ${JOB_ID}"
echo "  Node        : $(hostname)"
echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Date        : $(date)"
echo "  Images/run  : ${N_IMAGES}"
echo "  Repeats     : ${REPEATS}"
echo "  Repo        : ${REPO_DIR}"
echo "  Images      : ${IMG_DIR}"
echo "  Checkpoints : ${CKPT_DIR}"
echo "  Output root : ${OUT_ROOT}"
echo "=========================================="

LABELS=("b0.0032" "b0.0064" "b0.0128" "b0.0512" "b0.1024" "b0.2048")
LPIPS_WEIGHTS=("0.0" "0.0" "0.0" "0.9" "0.9" "0.9")
CKPTS=(
    "${CKPT_DIR}/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt"
    "${CKPT_DIR}/image-l2-use_weight5-vimeo-d64-t8193-b0.0064-x-cosine-01-float32-aux0.0_2.pt"
    "${CKPT_DIR}/image-l2-use_weight5-vimeo-d64-t8193-b0.0128-x-cosine-01-float32-aux0.0_2.pt"
    "${CKPT_DIR}/image-l2-use_weight5-vimeo-d64-t8193-b0.0512-x-cosine-01-float32-aux0.9lpips_2.pt"
    "${CKPT_DIR}/image-l2-use_weight5-vimeo-d64-t8193-b0.1024-x-cosine-01-float32-aux0.9lpips_2.pt"
    "${CKPT_DIR}/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt"
)

if [[ "${INCLUDE_B00032:-0}" == "1" ]]; then
    echo "NOTE: INCLUDE_B00032 is deprecated; b0.0032 is included by default."
fi

if [[ "${#LABELS[@]}" -ne "${#LPIPS_WEIGHTS[@]}" || "${#LABELS[@]}" -ne "${#CKPTS[@]}" ]]; then
    echo "ERROR: LABELS, LPIPS_WEIGHTS, and CKPTS must have the same length." >&2
    echo "  LABELS        : ${#LABELS[@]}" >&2
    echo "  LPIPS_WEIGHTS : ${#LPIPS_WEIGHTS[@]}" >&2
    echo "  CKPTS         : ${#CKPTS[@]}" >&2
    exit 1
fi

echo "  Num ckpts   : ${#CKPTS[@]}"

for idx in "${!LABELS[@]}"; do
    LABEL="${LABELS[$idx]}"
    LPIPS_WEIGHT="${LPIPS_WEIGHTS[$idx]}"
    CKPT="${CKPTS[$idx]}"

    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: checkpoint not found: ${CKPT}" >&2
        echo "Set CKPT_DIR=/path/to/x_param when submitting if checkpoints live elsewhere." >&2
        exit 1
    fi

    for repeat in $(seq 1 "${REPEATS}"); do
        REPEAT_PADDED=$(printf "%02d" "${repeat}")
        OUT_DIR="${OUT_ROOT}/${LABEL}/repeat_${REPEAT_PADDED}"
        mkdir -p "${OUT_DIR}"

        echo ""
        echo ">>> Context-only compression profiling ${LABEL}, repeat ${REPEAT_PADDED}/${REPEATS}, lpips_weight=${LPIPS_WEIGHT}"
        echo "    Output: ${OUT_DIR}"

        python profile_compression.py \
            --ckpt "${CKPT}" \
            --checkpoint_label "${LABEL}" \
            --img_dir "${IMG_DIR}" \
            --out_dir "${OUT_DIR}" \
            --device 0 \
            --lpips_weight "${LPIPS_WEIGHT}" \
            --n_images "${N_IMAGES}" \
            --start_index "${START_INDEX}" \
            --repeat "${repeat}"
    done
done

echo ""
echo ">>> Generating context-only compression profile summary and plots"

python plot_compression_profile.py \
    --profile_dir "${OUT_ROOT}" \
    --out_dir "${OUT_ROOT}"

echo ""
echo "=========================================="
echo "  Context-only compression profiling complete."
echo "  Results: ${OUT_ROOT}"
echo "  Summary: ${OUT_ROOT}/compression_profile_summary.csv"
echo "  Plots  : ${OUT_ROOT}/plots"
echo "  $(date)"
echo "=========================================="
