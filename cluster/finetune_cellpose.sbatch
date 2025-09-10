#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --qos=high
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j_%x_out.log
#SBATCH --error=%j_%x_err.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=koddenb@htw-berlin.de
#SBATCH --job-name=finetune_cellpose_mt

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

# Run the CellposeSAM finetuning script
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
torchrun --nproc_per_node=2 --nnodes=1 scripts/finetuning/finetune_cellposesam.py --dataset_dir /scratch/koddenbrock/mt/SynMT/synthetic/full

# Log completion time
echo "Job finished at $(date)"
