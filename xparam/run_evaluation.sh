#!/bin/bash
#SBATCH --job-name=cdc-xparam-eval
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/xparam_eval_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jskelly@tamu.edu

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE=/projects/bfod/jskelly5/cdc-deltaai
CKPT_DIR=$BASE/weights/x_param
IMG_DIR=$BASE/data/imgs
OUT_BASE=$BASE/output/xparam_eval_${SLURM_JOB_ID}
CODE_DIR=/u/jskelly5/code/xparam

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load default
module load gcc/14.2.0
module load python/miniforge3_pytorch/2.10.0
conda activate base
export PYTHONPATH=/u/jskelly5/.local/lib/python3.12/site-packages:$PYTHONPATH
python -m pip install --user ema-pytorch lpips --quiet

mkdir -p $OUT_BASE
mkdir -p $CODE_DIR/logs

echo "Starting CDC x-param compression evaluation..."
echo "Images:   $IMG_DIR"
echo "Output:   $OUT_BASE"

cd $CODE_DIR

# ── Loop over all x_param checkpoints ─────────────────────────────────────────
for CKPT in \
    "$CKPT_DIR/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt" \
    "$CKPT_DIR/image-l2-use_weight5-vimeo-d64-t8193-b0.0064-x-cosine-01-float32-aux0.0_2.pt" \
    "$CKPT_DIR/image-l2-use_weight5-vimeo-d64-t8193-b0.0128-x-cosine-01-float32-aux0.0_2.pt" \
    "$CKPT_DIR/image-l2-use_weight5-vimeo-d64-t8193-b0.0512-x-cosine-01-float32-aux0.9lpips_2.pt" \
    "$CKPT_DIR/image-l2-use_weight5-vimeo-d64-t8193-b0.1024-x-cosine-01-float32-aux0.9lpips_2.pt" \
    "$CKPT_DIR/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt"
do
    # Extract bitrate label from filename (e.g. b0.0512)
    BNAME=$(basename "$CKPT" .pt)
    BRATE=$(echo "$BNAME" | grep -oP 'b[0-9]+\.[0-9]+')

    # Set lpips_weight based on checkpoint type
    if echo "$BNAME" | grep -q "lpips"; then
        LPIPS=0.9
    else
        LPIPS=0.0
    fi

    OUT_DIR=$OUT_BASE/$BRATE
    mkdir -p $OUT_DIR

    echo ""
    echo "=== Running: $BRATE (lpips_weight=$LPIPS) ==="
    python evaluate_compression.py \
        --ckpt           "$CKPT" \
        --img_dir        "$IMG_DIR" \
        --out_dir        "$OUT_DIR" \
        --gamma          0.8 \
        --n_denoise_step 65 \
        --device         0 \
        --lpips_weight   $LPIPS \
        --n_images       100
done

echo ""
echo "Done. All results in: $OUT_BASE"
