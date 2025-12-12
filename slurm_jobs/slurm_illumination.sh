#!/bin/bash
#SBATCH --job-name=illumination
#SBATCH --output=illumination_%j.out
#SBATCH --error=illumination_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=aoberoi1@rcc.uchicago.edu

# ============================================================================
# Illumination Analysis Script - Analyze Existing Search Results
# ============================================================================
# This script analyzes existing search results against WikiArt reference:
# 1. Builds a reference cloud from HuggingFace WikiArt (DINO embeddings)
# 2. Analyzes search results (MAP-Elites, random, CMA) against reference
# 3. Computes distance-from-manifold metrics
# 4. Generates comparison visualizations
# ============================================================================

echo "============================================"
echo "ALIEN ART - ILLUMINATION ANALYSIS"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Load required modules
module load python/miniforge-25.3.0
module load cudnn

# Set number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate conda environment
source activate /scratch/midway3/aoberoi1/conda_envs/alienart

# Navigate to project directory
cd "$SLURM_SUBMIT_DIR"

# Create output directories
mkdir -p logs
mkdir -p illumination_results

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"
echo ""

# ============================================================================
# Configuration
# ============================================================================
REFERENCE_PKL="illumination_results/ref_wikiart_dino.pkl"
EMBEDDING_TYPE="dino"
N_SAMPLES=5000        # Reference cloud sample size (from 81K WikiArt images)
OUTPUT_DIR="illumination_results"

# Search results to analyze - Update these paths to your search outputs
# MAP-Elites with wow-factor (Job 42708270)
MAPELITES_DIR="outputs/run_42708270/map_elites"
# CMA-ES baseline (Job 42708860)
CMA_DIR="outputs/run_42708860/cma_search"
# Random search DINO (Job 42691366)
RANDOM_DINO_DIR="outputs/run_42691366/search_dino"

# ============================================================================
# Step 1: Build Reference Cloud from HuggingFace WikiArt
# ============================================================================
echo ""
echo "============================================"
echo "Step 1: Building Reference Cloud from HuggingFace WikiArt"
echo "============================================"
echo "Dataset: huggan/wikiart (81,444 artworks)"
echo "Output: $REFERENCE_PKL"
echo "Embedding: $EMBEDDING_TYPE"
echo "Sample Size: $N_SAMPLES"
echo ""

# Check if reference cloud already exists
if [ -f "$REFERENCE_PKL" ]; then
    echo "Reference cloud already exists at $REFERENCE_PKL"
    echo "Skipping build step. Delete the file to rebuild."
else
    python illumination.py build \
        --source wikiart_hf \
        --output "$REFERENCE_PKL" \
        --embedding "$EMBEDDING_TYPE" \
        --num-samples "$N_SAMPLES"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build reference cloud"
        exit 1
    fi
fi

echo ""
echo "Reference cloud ready."

# ============================================================================
# Step 2: Analyze MAP-Elites Results
# ============================================================================
echo ""
echo "============================================"
echo "Step 2: Analyzing MAP-Elites Results"
echo "============================================"

if [ -d "$MAPELITES_DIR" ]; then
    echo "Analyzing: $MAPELITES_DIR"
    
    python illumination.py analyze \
        --search-dir "$MAPELITES_DIR" \
        --reference "$REFERENCE_PKL" \
        --output-dir "$OUTPUT_DIR/mapelites_analysis" \
        --embedding "$EMBEDDING_TYPE"
    
    if [ $? -ne 0 ]; then
        echo "WARNING: MAP-Elites analysis failed"
    fi
else
    echo "MAP-Elites directory not found: $MAPELITES_DIR"
    echo "Skipping..."
fi

# ============================================================================
# Step 3: Analyze CMA-ES Results
# ============================================================================
echo ""
echo "============================================"
echo "Step 3: Analyzing CMA-ES Results"
echo "============================================"

if [ -d "$CMA_DIR" ]; then
    echo "Analyzing: $CMA_DIR"
    
    python illumination.py analyze \
        --search-dir "$CMA_DIR" \
        --reference "$REFERENCE_PKL" \
        --output-dir "$OUTPUT_DIR/cma_analysis" \
        --embedding "$EMBEDDING_TYPE"
    
    if [ $? -ne 0 ]; then
        echo "WARNING: CMA-ES analysis failed"
    fi
else
    echo "CMA-ES directory not found: $CMA_DIR"
    echo "Skipping..."
fi

# ============================================================================
# Step 4: Analyze Random Search Results
# ============================================================================
echo ""
echo "============================================"
echo "Step 4: Analyzing Random Search Results"
echo "============================================"

if [ -d "$RANDOM_DINO_DIR" ]; then
    echo "Analyzing: $RANDOM_DINO_DIR"
    
    python illumination.py analyze \
        --search-dir "$RANDOM_DINO_DIR" \
        --reference "$REFERENCE_PKL" \
        --output-dir "$OUTPUT_DIR/random_analysis" \
        --embedding "$EMBEDDING_TYPE"
    
    if [ $? -ne 0 ]; then
        echo "WARNING: Random search analysis failed"
    fi
else
    echo "Random search directory not found: $RANDOM_DINO_DIR"
    echo "Skipping..."
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Illumination Analysis Complete"
echo "============================================"
echo "End Time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Analyzed:"
echo "  - MAP-Elites: $MAPELITES_DIR"
echo "  - CMA-ES: $CMA_DIR"
echo "  - Random: $RANDOM_DINO_DIR"
echo ""
echo "Each analysis directory contains:"
echo "  - illumination_results.json: Full metrics"
echo "  - illumination_analysis.png: Visualization"
echo "  - top_alien_images/: Highest scoring images"
echo "============================================"
