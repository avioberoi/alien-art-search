#!/bin/bash
#SBATCH --job-name=cma_1k
#SBATCH --output=logs/cma_1k_%j.out
#SBATCH --error=logs/cma_1k_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# ============================================================================
# CMA-ES Search - Scaled Up (1000 samples)
# ============================================================================
# Paper experiment: CMA-ES on θ = (seed, cfg, steps) with fixed prompt
# Shows optimization can't escape semantic basin
# ============================================================================

set -e

echo "============================================"
echo "CMA-ES SEARCH - SCALED (1000 samples)"
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
ITERATIONS=84       # 84 iterations × 12 population = 1008 samples
POPULATION=12       # Larger batch for better A100 utilization
OUTPUT_DIR="outputs/paper_cma_${SLURM_JOB_ID}"

echo ""
PROMPT="a painting of a landscape"

echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Population: $POPULATION"
echo "  Total samples: $((ITERATIONS * POPULATION))"
echo "  Prompt: $PROMPT"
echo "  Output: $OUTPUT_DIR"
echo "  Novelty: k-NN (k=10)"

# ============================================================================
# Run
# ============================================================================
echo ""
echo "Running CMA-ES search..."
python cma_search.py \
    --iterations $ITERATIONS \
    --population $POPULATION \
    --prompt "$PROMPT" \
    --embedding dino \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Complete: $(date)"
