#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --qos=high
#SBATCH --time=10-00:00:00
#SBATCH --output=%j_%x_out.log
#SBATCH --error=%j_%x_err.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=koddenb@htw-berlin.de

# Log SLURM environment variables
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Nodes: $SLURM_NODELIST"
echo "SLURM GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM Memory per node: $SLURM_MEM_PER_NODE"
echo "SLURM Partition: $SLURM_JOB_PARTITION"
echo "Tuning Configuration Paths: $@"

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

# Exit immediately if a command exits with a non-zero status.
set -e

# Load necessary modules and activate conda environment
source ~/.bashrc
conda activate microtubule_tracking

# Check if conda activation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Conda environment 'microtubule_tracking' could not be activated."
    exit 1
fi

# Print the Python executable path for debugging
which python

# Loop through all provided config files and run optimization
for TUNING_CONFIG_PATH in "$@"; do
    echo "--------------------------------------------------"
    echo "Starting optimization for: $TUNING_CONFIG_PATH"
    python -m scripts.synthetic_data.optimize_synthetic_data --optimize "$TUNING_CONFIG_PATH"
    echo "Finished optimization for: $TUNING_CONFIG_PATH"
    echo "--------------------------------------------------"
done

# Log completion time
echo "All optimizations finished at $(date)"

