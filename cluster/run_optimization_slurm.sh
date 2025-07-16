#!/bin/bash

# This script finds all tuning configuration files matching 'tuning_config_*.json'
# in the specified directory and submits a separate Slurm job for each using
# the 'cluster/optimization.sbatch' script.

# --- Configuration ---
# Directory where your tuning configuration .json files are located.
CONFIG_DIR="config"
# Path to the sbatch script that will be executed for each job.
SBATCH_SCRIPT="cluster/optimization.sbatch"
# Define how many parallel Optuna workers you want to run
NUM_WORKERS=1
# --- End Configuration ---

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if the sbatch script exists and is executable
if [ ! -x "$SBATCH_SCRIPT" ]; then
    echo "ERROR: SBATCH script '$SBATCH_SCRIPT' not found or not executable."
    echo "Please ensure the path is correct and run 'chmod +x $SBATCH_SCRIPT'."
    exit 1
fi

# Find all configuration files matching the pattern
config_files=("$CONFIG_DIR"/tuning_config_*.json)

# Check if any config files were found
if [ ${#config_files[@]} -eq 0 ] || [ ! -e "${config_files[0]}" ]; then
    echo "No configuration files found in '$CONFIG_DIR' matching 'tuning_config_*.json'."
    exit 1
fi

echo "Found ${#config_files[@]} configuration files. Submitting individual jobs..."

  # --- Job Submission Logic ---
  # Loop to submit multiple Slurm jobs
for i in $(seq 1 $NUM_WORKERS); do
    JOB_NAME="mt_opt_w${i}" # Create a unique job name for each worker
    echo "Submitting worker $i (Job Name: $JOB_NAME)..."

    # Submit one job for each config file found.
    for config_file in "${config_files[@]}"; do
        if [ -f "$config_file" ]; then

          # Extract a descriptive name from the config path, e.g., "A_cosine" from "config/tuning_config_A_cosine.json"
          CONFIG_BASENAME=$(basename "${CONFIG_FILE}" .json) # tuning_config_A_cosine
          CONFIG_PATTERN=${CONFIG_BASENAME#tuning_config_}   # A_cosine
          JOB_NAME="mt_opt_${CONFIG_PATTERN}_w${i}" # Create a unique job name for each worker, e.g., mt_opt_A_cosine_w1

          echo "--------------------------------------------------"
          echo "Submitting job for configuration: $config_file"
          # Pass the config file path as a command-line argument to the sbatch script
          sbatch --job-name="$JOB_NAME" \
                  "$SBATCH_SCRIPT" "$config_file"
          echo "--------------------------------------------------"
        else
          echo "Warning: Found entry '$config_file' is not a file, skipping."
        fi
    done

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to submit worker $i. Aborting."
        exit 1
    fi

done

echo "All jobs have been submitted."