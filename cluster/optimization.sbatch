#!/bin/bash

#SBATCH --job-name=microtubulus
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # Number of tasks (1 task that uses multiple GPUs)
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G                  # Total memory for the job
#SBATCH --qos=normal
#SBATCH --output=%j_microtubulus_out.log
#SBATCH --error=%j_microtubulus_err.log

# Log SLURM environment variables
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Nodes: $SLURM_NODELIST"
echo "SLURM GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM Memory per node: $SLURM_MEM_PER_NODE"
echo "SLURM Partition: $SLURM_JOB_PARTITION"

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

# Load necessary modules and activate conda environment
source ~/.bashrc
conda activate microtubule_tracking

# Print the Python executable path for debugging
which python

# Run the evaluation script
python ./scripts/optimize_synthetic_data.py

# Log completion time
echo "Job finished at $(date)"