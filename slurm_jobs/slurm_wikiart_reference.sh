#!/bin/bash
#SBATCH --job-name=wikiart_ref
#SBATCH --output=logs/wikiart_ref_%j.out
#SBATCH --error=logs/wikiart_ref_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# ============================================================================
# Build WikiArt Reference Cloud (Scaled Up)
# ============================================================================

set -e

echo "============================================"
echo "WikiArt Reference Cloud Builder"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# Load modules
module load python/miniforge-25.3.0
module load cudnn

# Activate environment
source activate alienart

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Environment
export HF_HOME=/project/jevans/avi/hf_cache
export TRANSFORMERS_CACHE=/project/jevans/avi/hf_cache/transformers
export PYTHONUNBUFFERED=1

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ============================================================================
# Step 1: Elbow Analysis (find optimal reference size)
# ============================================================================
echo ""
echo "Step 1: Running elbow analysis..."
python build_wikiart_reference.py --elbow-analysis

# ============================================================================
# Step 2: Build full reference with image caching
# ============================================================================
echo ""
echo "Step 2: Building reference with image cache..."
# Use 20K as default, adjust based on elbow analysis
python build_wikiart_reference.py \
    --num-samples 20000 \
    --cache-images \
    --output-dir /project/jevans/avi/wikiart_reference \
    --cache-dir /project/jevans/avi/wikiart_cache

echo ""
echo "============================================"
echo "Complete: $(date)"
echo "============================================"
