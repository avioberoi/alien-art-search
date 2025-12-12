#!/bin/bash
#SBATCH --job-name=exp_theta_1k
#SBATCH --output=logs/exp_theta_1k_%j.out
#SBATCH --error=logs/exp_theta_1k_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

# ============================================================================
# Expanded θ Comparison - Scaled Up (1000 samples each)
# ============================================================================
# Paper experiment: Compare basic θ vs expanded θ
#   Basic: seed, cfg, steps (3D)
#   Expanded: + eta, guidance_rescale, clip_skip, prompt_strength (7D)
# 
# Question: Do internal SD controls open new regions of visual space?
# ============================================================================

set -e

echo "============================================"
echo "EXPANDED θ COMPARISON - SCALED (1000 samples)"
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
ITERATIONS=125      # 125 iterations × 8 population = 1000 samples per mode
POPULATION=8        # Batch size for V100 (less VRAM than A100)
OUTPUT_DIR="outputs/paper_expanded_theta_${SLURM_JOB_ID}"

echo ""
echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Population: $POPULATION"
echo "  Total samples per mode: $((ITERATIONS * POPULATION))"
echo "  Output: $OUTPUT_DIR"
echo "  Prompt: 'a painting of a landscape'"
echo "  Mode: compare (basic vs expanded)"

# ============================================================================
# Run Comparison
# ============================================================================
echo ""
echo "Running expanded θ comparison..."
python expanded_theta_v2.py \
    --mode compare \
    --iterations $ITERATIONS \
    --population $POPULATION \
    --output-dir "$OUTPUT_DIR" \
    --prompts "a painting of a landscape"

echo ""
echo "Complete: $(date)"
