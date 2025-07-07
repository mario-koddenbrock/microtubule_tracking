#!/bin/bash

# Define how many parallel Optuna workers you want to run
NUM_WORKERS=4

# Define the path to your tuning configuration file
# Make sure this path is correct and accessible from the compute nodes!
TUNING_CONFIG_PATH="./config/tuning_config_A.json"

echo "Starting Optuna optimization with $NUM_WORKERS parallel workers."
echo "Using tuning configuration: $TUNING_CONFIG_PATH"

# Check if the tuning config file exists
if [ ! -f "$TUNING_CONFIG_PATH" ]; then
    echo "ERROR: Tuning configuration file not found at '$TUNING_CONFIG_PATH'"
    echo "Please update TUNING_CONFIG_PATH in run_optimization.sh."
    exit 1
fi

# Loop to submit multiple Slurm jobs
for i in $(seq 1 $NUM_WORKERS); do
    JOB_NAME="mt_opt_w${i}" # Create a unique job name for each worker
    echo "Submitting worker $i (Job Name: $JOB_NAME)..."

    # Submit the sbatch script, passing the tuning config path as an argument.
    # The --job-name here will override the one in the sbatch script.
    sbatch --job-name="$JOB_NAME" \
           cluster/optimization.sbatch "$TUNING_CONFIG_PATH"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to submit worker $i. Aborting."
        exit 1
    fi
done

echo "All $NUM_WORKERS Optuna workers submitted."