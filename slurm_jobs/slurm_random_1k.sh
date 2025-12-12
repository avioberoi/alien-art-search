#!/bin/bash
#SBATCH --job-name=random_1k
#SBATCH --output=logs/random_1k_%j.out
#SBATCH --error=logs/random_1k_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# ============================================================================
# Random Search - Scaled Up (1000 samples)
# ============================================================================
# Paper experiment: Baseline random search with k-NN novelty metric
# ============================================================================

set -e

echo "============================================"
echo "RANDOM SEARCH - SCALED (1000 samples)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# Setup
module load python/miniforge-25.3.0
module load cudnn
source activate alienart
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

# Environment
export HF_HOME=/project/jevans/avi/hf_cache
export PYTHONUNBUFFERED=1

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ============================================================================
# Configuration
# ============================================================================
NUM_SAMPLES=1000
PROMPT="a painting of a landscape"
OUTPUT_DIR="outputs/paper_random_${SLURM_JOB_ID}"

echo ""
echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Prompt: $PROMPT"
echo "  Output: $OUTPUT_DIR"
echo "  Novelty: k-NN (k=10)"

# ============================================================================
# Run
# ============================================================================
echo ""
echo "Running random search..."
python search.py \
    --num_samples $NUM_SAMPLES \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --embedding_type dino \
    --dino_model facebook/dinov2-base

echo ""
echo "Complete: $(date)"
