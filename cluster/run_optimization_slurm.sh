#!/bin/bash

# This script launches a Slurm job for each tuning configuration file found.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_DIR="config"
SBATCH_SCRIPT="cluster/optimization.sbatch"

# --- Main Logic ---
echo "Starting Slurm job submission process..."
echo "Searching for configuration files in: '$CONFIG_DIR'"
echo "Using sbatch script: '$SBATCH_SCRIPT'"
echo "--------------------------------------------------"

# Check if the sbatch script exists
if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "ERROR: sbatch script not found at '$SBATCH_SCRIPT'"
    exit 1
fi

# Find and loop through all tuning configuration files for type A
for config_file in "$CONFIG_DIR"/tuning_config_B_*.json; do
    # Check if the file exists to handle cases where the glob finds nothing
    if [ -e "$config_file" ]; then
        echo "Found config: '$config_file'"
        echo "Submitting Slurm job..."
        sbatch "$SBATCH_SCRIPT" "$config_file"
        echo "Job for '$config_file' submitted."
        echo "--------------------------------------------------"
    else
        echo "No configuration files matching '$CONFIG_DIR/tuning_config_A_*.json' found."
        break
    fi
done

echo "All jobs have been submitted."