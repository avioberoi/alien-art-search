#!/bin/bash
#SBATCH --job-name=wikiart_clip
#SBATCH --output=logs/wikiart_clip_%j.out
#SBATCH --error=logs/wikiart_clip_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=90G
#SBATCH --time=04:00:00

# ============================================================================
# Build WikiArt CLIP Reference Embeddings
# ============================================================================
# Embeds all 81,444 WikiArt images with CLIP ViT-L-14 for comparison with DINO.
# Output: /project/jevans/avi/wikiart_reference/wikiart_clip_81444.pkl
# ============================================================================

set -e

echo "============================================"
echo "BUILD WIKIART CLIP REFERENCE"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Activate environment
source /software/bin/miniconda3/bin/activate alienart
cd /home/aoberoi1/Project

# Build CLIP embeddings for WikiArt (81,444 images)
# Using same images as DINO reference, just different encoder
python build_wikiart_reference.py \
    --embedding-model clip \
    --num-samples 82000 \
    --output-dir /project/jevans/avi/wikiart_reference

echo ""
echo "============================================"
echo "WikiArt CLIP Reference Complete: $(date)"
echo "============================================"

# Verify output
ls -la /project/jevans/avi/wikiart_reference/wikiart_clip_*.pkl
