#!/bin/bash
#SBATCH --job-name=analyze_me
#SBATCH --output=logs/analyze_me_%j.out
#SBATCH --error=logs/analyze_me_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=02:00:00

# ============================================================================
# Comprehensive MAP-Elites Analysis
# ============================================================================
# Analyzes existing MAP-Elites results for:
# 1. Elite Evolution (when cells were filled, novelty progression)
# 2. Illumination (are we in low-density regions of WikiArt?)
# 3. Off-Manifold (geometric distance from WikiArt distribution)
# 4. Nearest WikiArt neighbors
#
# Usage: sbatch slurm_analyze_mapelites.sh <search_dir>
# Example: sbatch slurm_analyze_mapelites.sh outputs/paper_mapelites_42948155
# ============================================================================

set -e

# Get search directory from argument or use default
SEARCH_DIR="${1:-outputs/paper_mapelites_42948155}"

echo "============================================"
echo "MAP-ELITES COMPREHENSIVE ANALYSIS"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Search Dir: $SEARCH_DIR"

# Setup
module load python/miniforge-25.3.0
module load cudnn
source activate alienart
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Environment
export HF_HOME=/project/jevans/avi/hf_cache
export PYTHONUNBUFFERED=1

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Auto-detect embedding model from archive_data.json
EMBEDDING_MODEL=$(python -c "
import json
with open('$SEARCH_DIR/archive_data.json') as f:
    data = json.load(f)
print(data.get('config', {}).get('embedding_model', 'dino'))
" 2>/dev/null || echo "dino")

echo "Detected embedding model: $EMBEDDING_MODEL"

# Set reference paths based on embedding model
if [ "$EMBEDDING_MODEL" = "clip" ]; then
    REFERENCE="/project/jevans/avi/wikiart_reference/wikiart_clip_81444.pkl"
    FAISS_INDEX="/project/jevans/avi/wikiart_reference/wikiart_clip_81444.faiss"
else
    REFERENCE="/project/jevans/avi/wikiart_reference/wikiart_dino_81444.pkl"
    FAISS_INDEX="/project/jevans/avi/wikiart_reference/wikiart_dino_81444.faiss"
fi

echo "Using reference: $REFERENCE"

# Verify search directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "ERROR: Search directory not found: $SEARCH_DIR"
    exit 1
fi

if [ ! -f "$SEARCH_DIR/archive_data.json" ]; then
    echo "ERROR: archive_data.json not found in $SEARCH_DIR"
    exit 1
fi

# ============================================================================
# 1. Find nearest WikiArt neighbors
# ============================================================================
echo ""
echo "============================================"
echo "1. Finding nearest WikiArt neighbors..."
echo "============================================"
python find_nearest_wikiart.py \
    --search-dir "$SEARCH_DIR" \
    --reference "$REFERENCE" \
    --faiss-index "$FAISS_INDEX" \
    --k 5 \
    --n-examples 10

# ============================================================================
# 2. Comprehensive Analysis (Evolution, Illumination, Off-Manifold)
# ============================================================================
echo ""
echo "============================================"
echo "2. Running comprehensive analysis..."
echo "============================================"
python analyze_mapelites.py \
    --search-dir "$SEARCH_DIR" \
    --reference "$REFERENCE" \
    --output-dir "${SEARCH_DIR}/analysis"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "ANALYSIS COMPLETE"
echo "============================================"
echo "Results saved to:"
echo "  - ${SEARCH_DIR}/wikiart_analysis/"
echo "  - ${SEARCH_DIR}/analysis/"
echo ""
echo "Key outputs:"
echo "  - elite_evolution.png (when cells were filled)"
echo "  - illumination_analysis.png (low-density regions)"
echo "  - offmanifold_analysis.png (geometric metrics)"
echo "  - top_alien_combined.png (best images by combined score)"
echo "  - analysis_results.json (all statistics)"
echo ""
echo "Complete: $(date)"
