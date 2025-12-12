#!/bin/bash
# ============================================================================
# Alien Art: Quick Setup Script
# ============================================================================
# Run: bash setup.sh
# ============================================================================

set -e  # Exit on error

echo "Loading required modules..."
module load python/miniforge-25.3.0
module load cudnn  # This automatically loads cuda

# Disable conda cache to save inode quota
export USE_CONDA_CACHE=0

echo "Checking for existing environment..."
if conda env list | grep -q "alienart "; then
    echo "Environment 'alienart' already exists. Removing it..."
    conda env remove -n alienart -y
fi

echo "Creating conda environment 'alienart' in scratch space..."
# Use scratch space to avoid home directory inode limits
ENV_PATH="/scratch/midway3/$USER/conda_envs/alienart"
mkdir -p "/scratch/midway3/$USER/conda_envs"
conda create --prefix "$ENV_PATH" -y

echo "Activating environment..."
source activate "$ENV_PATH"

echo "Installing PyTorch (using existing CUDA installation)..."
conda install pytorch torchvision -c pytorch -y

echo "Installing other dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Creating convenience symlink..."
mkdir -p ~/.conda/envs
ln -sf "$ENV_PATH" ~/.conda/envs/alienart

echo "Cleaning conda cache..."
conda clean --all -y

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Environment installed to: $ENV_PATH"
echo ""
echo "To use the environment, run:"
echo "  module load python/miniforge-25.3.0"
echo "  module load cudnn"
echo "  source activate alienart"
echo ""
echo "Then run:"
echo "  python demo.py          # Quick test"
echo "  python search.py        # Full search"
echo ""