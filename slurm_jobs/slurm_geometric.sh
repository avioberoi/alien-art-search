#!/bin/bash
#SBATCH --job-name=geometric
#SBATCH --output=geometric_%j.out
#SBATCH --error=geometric_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=aoberoi1@rcc.uchicago.edu

# ============================================================================
# Geometric Off-Manifold Analysis
# ============================================================================
# This script:
# 1. Extracts DINO embeddings from search results (with GPU batch processing)
# 2. Runs geometric analysis comparing methods against WikiArt reference
# 3. Generates comparison visualizations
# ============================================================================

echo "============================================"
echo "ALIEN ART - GEOMETRIC OFF-MANIFOLD ANALYSIS"
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
source activate alienart

# Navigate to project directory
cd "$SLURM_SUBMIT_DIR"

# Create output directory
mkdir -p geometric_data
mkdir -p outputs/geometric

# Print environment info
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ============================================================================
# Configuration
# ============================================================================

# WikiArt reference embeddings (from illumination job)
WIKIART_REF="illumination_results/ref_wikiart_dino.pkl"

# Search result directories
MAPELITES_DIR="outputs/run_42708270/map_elites"
CMA_DIR="outputs/run_42708860/cma_search"
RANDOM_DIR="outputs/run_42691366/search_dino"

# Output paths
MAPELITES_EMB="geometric_data/mapelites_emb.pkl"
CMA_EMB="geometric_data/cma_emb.pkl"
RANDOM_EMB="geometric_data/random_emb.pkl"

EMBEDDING_TYPE="dino"
OUTPUT_DIR="outputs/geometric"

# ============================================================================
# Step 1: Extract Embeddings from Search Results
# ============================================================================
echo ""
echo "============================================"
echo "Step 1: Extracting Embeddings"
echo "============================================"

# MAP-Elites
if [ ! -f "$MAPELITES_EMB" ]; then
    echo "Extracting MAP-Elites embeddings..."
    python extract_embeddings.py \
        --search-dir "$MAPELITES_DIR" \
        --output "$MAPELITES_EMB" \
        --embedding-type "$EMBEDDING_TYPE"
else
    echo "MAP-Elites embeddings already exist: $MAPELITES_EMB"
fi

# CMA-ES
if [ ! -f "$CMA_EMB" ]; then
    echo "Extracting CMA-ES embeddings..."
    python extract_embeddings.py \
        --search-dir "$CMA_DIR" \
        --output "$CMA_EMB" \
        --embedding-type "$EMBEDDING_TYPE"
else
    echo "CMA-ES embeddings already exist: $CMA_EMB"
fi

# Random Search (skip if JSON is incomplete)
if [ ! -f "$RANDOM_EMB" ]; then
    echo "Extracting Random search embeddings..."
    python extract_embeddings.py \
        --search-dir "$RANDOM_DIR" \
        --output "$RANDOM_EMB" \
        --embedding-type "$EMBEDDING_TYPE" || echo "Random extraction failed (may have incomplete JSON)"
else
    echo "Random embeddings already exist: $RANDOM_EMB"
fi

echo ""
echo "Embedding extraction complete."

# ============================================================================
# Step 2: Run Geometric Analysis
# ============================================================================
echo ""
echo "============================================"
echo "Step 2: Running Geometric Analysis"
echo "============================================"

# Check WikiArt reference exists
if [ ! -f "$WIKIART_REF" ]; then
    echo "ERROR: WikiArt reference not found at $WIKIART_REF"
    echo "Please run the illumination job first."
    exit 1
fi

# Build command with available embeddings
CMD="python quick_gerometric.py --wikiart $WIKIART_REF --output-dir $OUTPUT_DIR"

if [ -f "$MAPELITES_EMB" ]; then
    CMD="$CMD --mapelites $MAPELITES_EMB"
fi

if [ -f "$CMA_EMB" ]; then
    CMD="$CMD --cma $CMA_EMB"
fi

if [ -f "$RANDOM_EMB" ]; then
    CMD="$CMD --random $RANDOM_EMB"
fi

echo "Running: $CMD"
$CMD

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "Geometric Analysis Complete"
echo "============================================"
echo "End Time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "  (no output files yet)"
echo ""
echo "============================================"
