#!/bin/bash
#SBATCH --job-name=alienart_map_elites
#SBATCH --output=alienart_map_elites_%j.out
#SBATCH --error=alienart_map_elites_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=aoberoi1@rcc.uchicago.edu

# ============================================================================
# Alien Art: MAP-Elites Job Script
# ============================================================================

echo "============================================"
echo "ALIEN ART - MAP-ELITES JOB"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load required modules
module load python/miniforge-25.3.0
module load cudnn

# Set number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate environment
source activate /scratch/midway3/aoberoi1/conda_envs/alienart

# Navigate to project directory
cd "$SLURM_SUBMIT_DIR"

# Create output directories
mkdir -p logs outputs

# ============================================================================
# CONFIGURATION
# ============================================================================

ITERATIONS=500
GRID_SIZE=12
OUTPUT_DIR="outputs/run_${SLURM_JOB_ID}/map_elites"

# ============================================================================
# RUN PIPELINE
# ============================================================================

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU Memory: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")')"
fi
echo ""

echo "Running MAP-Elites with Auto-Axes (PCA)..."
python map_elites.py \
    --iterations $ITERATIONS \
    --grid-size $GRID_SIZE \
    --output-dir "$OUTPUT_DIR" \
    --auto-axes

echo ""
echo "============================================"
echo "JOB COMPLETE"
echo "============================================"
echo "End time: $(date)"
