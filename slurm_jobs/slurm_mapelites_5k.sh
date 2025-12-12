#!/bin/bash
#SBATCH --job-name=mapelites_5k
#SBATCH --output=logs/mapelites_5k_%j.out
#SBATCH --error=logs/mapelites_5k_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ============================================================================
# MAP-Elites - Scaled Up (5000 samples)
# ============================================================================
# Paper experiment: Quality-Diversity search with prompt evolution
# Shows language is the key to escaping semantic basins
# ============================================================================

set -e

echo "============================================"
echo "MAP-ELITES - SCALED (5000 samples)"
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
ITERATIONS=5000     # One image per iteration
GRID_SIZE=15        # 15x15 = 225 cells
OUTPUT_DIR="outputs/paper_mapelites_${SLURM_JOB_ID}"

echo ""
echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Grid size: ${GRID_SIZE}x${GRID_SIZE} ($((GRID_SIZE * GRID_SIZE)) cells)"
echo "  Output: $OUTPUT_DIR"
echo "  Novelty: k-NN (k=10)"
echo "  Axes: Auto (PCA)"

# ============================================================================
# Run
# ============================================================================
echo ""
echo "Running MAP-Elites search..."
python map_elites.py \
    --iterations $ITERATIONS \
    --grid-size $GRID_SIZE \
    --output-dir "$OUTPUT_DIR" \
    --auto-axes

# ============================================================================
# Post-processing: Find nearest WikiArt neighbors
# ============================================================================
echo ""
echo "Finding nearest WikiArt neighbors..."
python find_nearest_wikiart.py \
    --search-dir "$OUTPUT_DIR" \
    --reference /project/jevans/avi/wikiart_reference/wikiart_dino_81444.pkl \
    --faiss-index /project/jevans/avi/wikiart_reference/wikiart_dino_81444.faiss \
    --k 5 \
    --n-examples 10

# ============================================================================
# Post-processing: Comprehensive Analysis (Evolution, Illumination, Off-Manifold)
# ============================================================================
echo ""
echo "Running comprehensive analysis..."
python analyze_mapelites.py \
    --search-dir "$OUTPUT_DIR" \
    --reference /project/jevans/avi/wikiart_reference/wikiart_dino_81444.pkl \
    --output-dir "${OUTPUT_DIR}/analysis"

echo ""
echo "Complete: $(date)"
