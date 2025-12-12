#!/bin/bash
#SBATCH --job-name=expanded_theta
#SBATCH --output=expanded_theta_%j.out
#SBATCH --error=expanded_theta_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=aoberoi1@rcc.uchicago.edu

# ============================================================================
# Expanded θ Search: Basic vs Expanded Comparison
# ============================================================================
# Tests: Do internal diffusion controls (eta, guidance_rescale, clip_skip, 
#        prompt_strength) open regions that external params (seed, cfg, steps)
#        cannot reach?
#
# Basic θ: [seed, cfg, steps] = 3D
# Expanded θ: [seed, cfg, steps, eta, guidance_rescale, clip_skip, prompt_strength] = 7D
#
# Fixed prompts (not evolving) - isolates effect of θ controls
# ============================================================================

echo "============================================"
echo "EXPANDED θ SEARCH: Basic vs Expanded"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo ""

# Load modules
module load python/miniforge-25.3.0
module load cudnn

# Activate environment
source activate alienart

# Navigate to project
cd "$SLURM_SUBMIT_DIR"

# Verify GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Environment
export PYTHONUNBUFFERED=1
export HF_HOME=/project/jevans/avi/hf_cache
export TRANSFORMERS_CACHE=/project/jevans/avi/hf_cache/transformers

# ============================================================================
# Configuration
# ============================================================================

# Reference embeddings (WikiArt DINO - 5000 samples)
REFERENCE="illumination_results/ref_wikiart_dino.pkl"

# Search parameters
ITERATIONS=50  # Per mode (50 basic + 50 expanded = ~600 images total)
POPULATION=6   # CMA-ES population size

# Output
OUTPUT_DIR="outputs/expanded_theta_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Reference: $REFERENCE"
echo "  Iterations per mode: $ITERATIONS"
echo "  Population: $POPULATION"
echo "  Total images: ~$((ITERATIONS * POPULATION * 2))"
echo "  Output: $OUTPUT_DIR"
echo ""

# ============================================================================
# Run Comparison Experiment
# ============================================================================
echo "============================================"
echo "Running Basic θ vs Expanded θ Comparison"
echo "============================================"
echo ""

python expanded_theta_v2.py \
    --mode compare \
    --iterations $ITERATIONS \
    --population $POPULATION \
    --reference "$REFERENCE" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Experiment Complete"
echo "============================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Output files:"
    ls -la "$OUTPUT_DIR/"
    echo ""
    
    if [ -d "$OUTPUT_DIR/basic" ]; then
        echo "Basic θ results:"
        ls -la "$OUTPUT_DIR/basic/" | head -10
    fi
    
    if [ -d "$OUTPUT_DIR/expanded" ]; then
        echo ""
        echo "Expanded θ results:"
        ls -la "$OUTPUT_DIR/expanded/" | head -10
    fi
    
    echo ""
    echo "Key output: $OUTPUT_DIR/basic_vs_expanded_comparison.png"
else
    echo "ERROR: Experiment failed with exit code $EXIT_CODE"
fi

echo "============================================"
