#!/bin/bash
#SBATCH --job-name=mapelites_clip
#SBATCH --output=logs/mapelites_clip_%j.out
#SBATCH --error=logs/mapelites_clip_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00

# ============================================================================
# MAP-Elites with CLIP Embeddings (for comparison with DINO)
# ============================================================================
# This runs the same MAP-Elites search but uses CLIP instead of DINO for:
#   - Novelty computation (cosine distance in CLIP space)
#   - Archive diversity tracking
#
# Hypothesis: CLIP captures semantic novelty, DINO captures visual novelty.
# Comparing results will show which embedding space better identifies "alien art".
# ============================================================================

set -e

# Parse arguments
USE_ART_PROMPTS="${1:-false}"
ITERATIONS="${2:-5000}"

echo "============================================"
echo "MAP-ELITES WITH CLIP EMBEDDINGS"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""
echo "Embedding Model: CLIP (ViT-L-14)"
echo "Art Prompts: $USE_ART_PROMPTS"
echo "Iterations: $ITERATIONS"
echo ""

# Activate environment
source /software/bin/miniconda3/bin/activate alienart
cd /home/aoberoi1/Project

# Set output directory based on prompt mode
if [ "$USE_ART_PROMPTS" = "true" ]; then
    OUTPUT_DIR="outputs/paper_mapelites_clip_art_${SLURM_JOB_ID}"
    ART_FLAG="--use-art-prompts"
else
    OUTPUT_DIR="outputs/paper_mapelites_clip_${SLURM_JOB_ID}"
    ART_FLAG=""
fi

echo "Output: $OUTPUT_DIR"
echo ""

# Run MAP-Elites with CLIP
python map_elites.py \
    --output-dir "$OUTPUT_DIR" \
    --iterations $ITERATIONS \
    --grid-size 15 \
    --auto-axes \
    --embedding-model clip \
    $ART_FLAG

echo ""
echo "============================================"
echo "MAP-Elites (CLIP) Complete: $(date)"
echo "============================================"

# Show results summary
echo ""
echo "Results:"
ls -la "$OUTPUT_DIR/"

# Quick stats from archive
python -c "
import json
with open('$OUTPUT_DIR/archive_data.json') as f:
    data = json.load(f)
elites = data.get('elites', [])
novelties = [e['novelty'] for e in elites]
print(f'Elites: {len(elites)}')
print(f'Coverage: {len(elites) / (15*15) * 100:.1f}%')
print(f'Mean Novelty: {sum(novelties)/len(novelties):.4f}')
print(f'Max Novelty: {max(novelties):.4f}')
"
