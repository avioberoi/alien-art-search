#!/bin/bash
#SBATCH --job-name=alienart_dino
#SBATCH --output=alienart_dino_%j.out
#SBATCH --error=alienart_dino_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END              # Email notifications: ALL, BEGIN, END, FAIL, NONE
#SBATCH --mail-user=aoberoi1@rcc.uchicago.edu  # Replace with your CNetID@rcc.uchicago.edu

# ============================================================================
# Alien Art: SLURM Job Script
# ============================================================================
# Submit with: sbatch slurm.sh
# Check status: squeue --user=$USER
# Cancel: scancel <jobid>
# ============================================================================

echo "============================================"
echo "ALIEN ART - SLURM JOB"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load required modules
module load python/miniforge-25.3.0
module load cudnn  # This automatically loads cuda

# Set number of threads to match allocated CPUs (best practice)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate environment
source activate alienart

# Navigate to project directory
cd "$SLURM_SUBMIT_DIR"

# Create output directories
mkdir -p logs outputs

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo "GPU: None available"
fi
echo ""

# ============================================================================
# CONFIGURATION - Edit these for your run
# ============================================================================

NUM_SAMPLES=200
PROMPT="a painting of a landscape"
OUTPUT_DIR="outputs/run_${SLURM_JOB_ID}"

# WikiArt path (comment out if not available)
# WIKIART_DIR="/path/to/wikiart"

# ============================================================================
# RUN PIPELINE
# ============================================================================

# echo "============================================"
# echo "PHASE 0: Demo"
# echo "============================================"
# python demo.py

echo ""
echo "============================================"
echo "PHASE 1: Novelty Search (DINO)"
echo "============================================"
python search.py \
    --num_samples $NUM_SAMPLES \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR/search_dino" \
    --embedding_type dino \
    --dino_model facebook/dinov2-base

# Phase 2B: Art Cloud (uncomment if you have WikiArt)
# echo ""
# echo "============================================"
# echo "PHASE 2B: Build Art Cloud"
# echo "============================================"
# if [ ! -f "embeddings/art_cloud.pkl" ]; then
#     python art_cloud.py build \
#         --wikiart_dir "$WIKIART_DIR" \
#         --output "embeddings/art_cloud.pkl" \
#         --sample_size 3000
# fi

# echo ""
# echo "============================================"
# echo "PHASE 2B: Analyze with Art Cloud"
# echo "============================================"
# python art_cloud.py analyze \
#     --search_dir "$OUTPUT_DIR/search" \
#     --art_cloud "embeddings/art_cloud.pkl" \
#     --output_dir "$OUTPUT_DIR/analysis"

echo ""
echo "============================================"
echo "JOB COMPLETE"
echo "============================================"
echo "End time: $(date)"
echo "Results: $OUTPUT_DIR"