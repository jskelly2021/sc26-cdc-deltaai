#!/bin/bash
#SBATCH --job-name=cdc-xparam-b02048-resume
#SBATCH --account=bfod-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/xparam_b02048_resume_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yyang48@illinois.edu

set -e

BASE=/projects/bfod/yyang48/cdc-deltaai
IMG_DIR=$BASE/data/imgs
CODE_DIR=/u/yyang48/code/xparam
CKPT="$BASE/weights/x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt"
OUT_DIR="$BASE/outputs/xparam_eval_2123265_resume/b0.2048"

module purge
module load default
module load gcc/14.2.0
module load python/miniforge3_pytorch/2.10.0
conda activate base
export PYTHONPATH=/u/yyang48/.local/lib/python3.12/site-packages:$PYTHONPATH
python -m pip install --user ema-pytorch lpips --quiet

mkdir -p outputs/slurm
mkdir -p "$OUT_DIR"
mkdir -p "$CODE_DIR/logs"

cd "$CODE_DIR"

echo "Resuming CDC x-param evaluation for b0.2048..."
echo "Images:   $IMG_DIR"
echo "Output:   $OUT_DIR"
echo "Start index: 76"
echo "Image count: 24"

python evaluate_compression.py \
    --ckpt           "$CKPT" \
    --img_dir        "$IMG_DIR" \
    --out_dir        "$OUT_DIR" \
    --gamma          0.8 \
    --n_denoise_step 65 \
    --device         0 \
    --lpips_weight   0.9 \
    --start_index    76 \
    --n_images       24

echo ""
echo "Done. Resume results in: $OUT_DIR"
