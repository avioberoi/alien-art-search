#!/bin/bash
#SBATCH --job-name=analyze_art
#SBATCH --output=logs/analyze_art_%j.out
#SBATCH --error=logs/analyze_art_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# ============================================================================
# MAP-Elites Analysis (jevans-gpu partition)
# ============================================================================
set -e

SEARCH_DIR="${1:-outputs/paper_mapelites_art_42964578}"

echo "============================================"
echo "MAP-ELITES COMPREHENSIVE ANALYSIS"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Search Dir: $SEARCH_DIR"

# Activate environment
source /software/bin/miniconda3/bin/activate alienart
cd /home/aoberoi1/Project

# Run comprehensive analysis (includes image galleries and nearest neighbor visualizations)
echo ""
echo "Running comprehensive analysis..."
python analyze_mapelites.py \
    --search-dir "$SEARCH_DIR" \
    --reference /project/jevans/avi/wikiart_reference/wikiart_dino_81444.pkl

echo ""
echo "============================================"
echo "Analysis Complete: $(date)"
echo "============================================"
echo "Output saved to: $SEARCH_DIR/analysis/"

ls -la "$SEARCH_DIR/analysis/" 2>/dev/null || echo "Check search dir for outputs"
