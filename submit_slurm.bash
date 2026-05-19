#!/bin/bash
#SBATCH --job-name=SIR_test_run
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:59:00
#SBATCH --constraint=scratch-node
#SBATCH --output=logs/test_out_%A.txt
#SBATCH --error=logs/test_err_%A.txt

# Purge existing modules and load required software stack
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load Pillow/10.0.0-GCCcore-12.3.0
module load tqdm/4.66.1-GCCcore-12.3.0

# Activate virtual environment
source .venv/bin/activate

# Create output directory for logs
mkdir -p logs

# Execute test script with unbuffered output
python -u test.py