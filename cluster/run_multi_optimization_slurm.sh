#!/bin/bash

# This script launches a single Slurm job to run optimizations for multiple video files.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_DIR="config"
SBATCH_SCRIPT="cluster/multi_optimization.sbatch"
REFERENCE_DIR="/scratch/koddenbrock/mt/SynMT/real/small/all_frames"
CONFIG_TEMPLATE="config/tuning_config_cluster.json"
FOLDER_START_INDEX=23  # Start index (optional)
FOLDER_END_INDEX=33  # End index (large number for all folders)

# --- Main Logic ---
echo "Starting Slurm job submission process for multiple optimizations..."
echo "Using sbatch script: '$SBATCH_SCRIPT'"
echo "Using template config: '$CONFIG_TEMPLATE'"
echo "Searching for image folders in: '$REFERENCE_DIR'"
echo "Processing folders from index $FOLDER_START_INDEX to $FOLDER_END_INDEX"
echo "--------------------------------------------------"

# Check if the sbatch script and template exist
if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "ERROR: sbatch script not found at '$SBATCH_SCRIPT'"
    exit 1
fi

if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "ERROR: template config not found at '$CONFIG_TEMPLATE'"
    exit 1
fi

# Find all subdirectories containing image files
IMAGE_FOLDERS=()
while IFS= read -r -d $'\0' dir; do
    IMAGE_FOLDERS+=("$dir")
done < <(find "$REFERENCE_DIR" -mindepth 2 \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.tif" -o -iname "*.tiff" \) -printf '%h\0' | sort -zu)

TOTAL_FOLDERS=${#IMAGE_FOLDERS[@]}

echo "Found $TOTAL_FOLDERS image folders."

# Limit to range if specified
if [ $FOLDER_END_INDEX -lt $TOTAL_FOLDERS ]; then
    LIMIT=$FOLDER_END_INDEX
else
    LIMIT=$TOTAL_FOLDERS
fi

INDEX=1
CONFIG_FILES=()

# Loop through all image folders to create config files
for FOLDER_PATH in "${IMAGE_FOLDERS[@]}"; do
    if [ $INDEX -ge $FOLDER_START_INDEX ] && [ $INDEX -le $LIMIT ]; then
        # Create a unique config for this folder
        FOLDER_BASENAME=$(basename "$FOLDER_PATH")
        OUTPUT_CONFIG="$CONFIG_DIR/tuning_config_${FOLDER_BASENAME}.json"

        # Create a custom config file using the template but with updated folder path
        cat "$CONFIG_TEMPLATE" | \
            sed "s|\"reference_images_dir\": \".*\"|\"reference_images_dir\": \"$FOLDER_PATH\"|g" | \
            sed "s|\"output_config_id\": \".*\"|\"output_config_id\": \"${FOLDER_BASENAME}\"|g" > "$OUTPUT_CONFIG"

        echo "Created config: '$OUTPUT_CONFIG' for folder: '$FOLDER_PATH'"
        CONFIG_FILES+=("$OUTPUT_CONFIG")
    fi

    INDEX=$((INDEX + 1))
done

# Submit a single sbatch job with all config files if any were created
if [ ${#CONFIG_FILES[@]} -gt 0 ]; then
    JOB_NAME="mt-multi-opt"
    echo "--------------------------------------------------"
    echo "Submitting a single job for all ${#CONFIG_FILES[@]} configurations."
    sbatch --job-name="$JOB_NAME" "$SBATCH_SCRIPT" "${CONFIG_FILES[@]}"
    echo "Job '$JOB_NAME' submitted with all configurations."
    echo "--------------------------------------------------"
else
    echo "No configurations were generated, so no job was submitted."
fi

echo "Script finished."

