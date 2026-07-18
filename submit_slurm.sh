#!/bin/bash
#SBATCH --job-name=SIR_grid_search
#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=logs/test_out_%A.txt
#SBATCH --error=logs/test_err_%A.txt

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load Pillow/10.0.0-GCCcore-12.3.0
module load tqdm/4.66.1-GCCcore-12.3.0

source .venv/bin/activate

mkdir -p logs

# Slurm dynamically assigns $TMPDIR to an authorized local NVMe path generated for this specific job ID
SCRATCH_DIR="$TMPDIR/dino_db"
mkdir -p "$SCRATCH_DIR"

# Stage the massive HDF5 databases to the local NVMe node concurrently to saturate read/write throughput
# echo ">> Starting NVMe data staging at $(date)"
# cp /projects/prjs2073/SimilarityImageRetrieval/datasets/ILIAS/DINO_data_vit_giant_patch14_dinov2.lvd142m_518.h5 "$SCRATCH_DIR/" &
# cp /projects/prjs2073/SimilarityImageRetrieval/datasets/ILIAS/DINO_data_vit_giant_patch14_reg4_dinov2.lvd142m_518.h5 "$SCRATCH_DIR/" &
# wait
# echo ">> Data staging complete at $(date)"

# Export the temporary path so Python can intercept and reroute disk-backed memory operations
export LOCAL_DB_PATH="$SCRATCH_DIR"

python -u grid_search.py