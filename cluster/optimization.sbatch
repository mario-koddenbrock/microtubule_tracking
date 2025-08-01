#!/bin/bash
#SBATCH --job-name=microtubule
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --qos=normal
#SBATCH --time=3-12:00:00
#SBATCH --output=%j_microtubule_out.log
#SBATCH --error=%j_microtubule_err.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=koddenb@htw-berlin.de

# Get the tuning config path from the command line argument passed to sbatch
# Slurm automatically passes arguments after the script name to the script's execution environment.
TUNING_CONFIG_PATH="$1"

# Log SLURM environment variables
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Nodes: $SLURM_NODELIST"
echo "SLURM GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM Memory per node: $SLURM_MEM_PER_NODE"
echo "SLURM Partition: $SLURM_JOB_PARTITION"
echo "Tuning Configuration Path: $TUNING_CONFIG_PATH"

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

#pip install -e .
#
## Check if pip install was successful
#if [ $? -ne 0 ]; then
#    echo "ERROR: 'pip install -e .' failed."
#    exit 1
#fi

# Print the Python executable path for debugging
which python

# Run the evaluation script
python scripts/synthetic_data/optimize_synthetic_data.py "$TUNING_CONFIG_PATH"
# Log completion time
echo "Job finished at $(date)"