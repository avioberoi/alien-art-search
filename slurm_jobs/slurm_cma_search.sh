#!/bin/bash
#SBATCH --job-name=alienart_cma
#SBATCH --output=alienart_cma_%j.out
#SBATCH --error=alienart_cma_%j.err
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=aoberoi1@rcc.uchicago.edu

# ============================================================================
# Alien Art: CMA-ES Baseline Job Script
# ============================================================================
# This demonstrates that even sophisticated optimization on θ = (seed, cfg, steps)
# cannot escape the semantic basin of a fixed prompt.
# Expected result: ~0.38 mean novelty (similar to random search - proves prompt is key)
# ============================================================================

echo "============================================"
echo "ALIEN ART - CMA-ES BASELINE"
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

ITERATIONS=100
POPULATION=8
EMBEDDING="dino"
OUTPUT_DIR="outputs/run_${SLURM_JOB_ID}/cma_search"

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
fi
echo "CMA-ES version: $(python -c 'import cma; print(cma.__version__)')"
echo ""

echo "Running CMA-ES Baseline Search..."
echo "  Iterations: $ITERATIONS"
echo "  Population size: $POPULATION (batch size)"
echo "  Embedding: $EMBEDDING"
echo ""

python cma_search.py \
    --iterations $ITERATIONS \
    --population $POPULATION \
    --embedding $EMBEDDING \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "JOB COMPLETE"
echo "============================================"
echo "End time: $(date)"
