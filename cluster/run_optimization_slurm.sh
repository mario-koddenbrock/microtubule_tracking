#!/bin/bash

# This script launches a Slurm job for each video file in the reference directory

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_DIR="config"
SBATCH_SCRIPT="cluster/optimization.sbatch"
REFERENCE_DIR="/scratch/koddenbrock/mt/data_mpi"
CONFIG_TEMPLATE="config/tuning_config_cluster.json"
VIDEO_START_INDEX=1  # Start index (optional)
VIDEO_END_INDEX=10  # End index (large number for all videos)

# --- Main Logic ---
echo "Starting Slurm job submission process..."
echo "Using sbatch script: '$SBATCH_SCRIPT'"
echo "Using template config: '$CONFIG_TEMPLATE'"
echo "Searching for videos in: '$REFERENCE_DIR'"
echo "Processing videos from index $VIDEO_START_INDEX to $VIDEO_END_INDEX"
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

# Find all video files
VIDEO_FILES=($(find "$REFERENCE_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.tif" -o -name "*.tiff" \) | sort))
TOTAL_VIDEOS=${#VIDEO_FILES[@]}

echo "Found $TOTAL_VIDEOS video files."

# Limit to range if specified
if [ $VIDEO_END_INDEX -lt $TOTAL_VIDEOS ]; then
    LIMIT=$VIDEO_END_INDEX
else
    LIMIT=$TOTAL_VIDEOS
fi

INDEX=1

# Loop through all video files
for VIDEO_PATH in "${VIDEO_FILES[@]}"; do
    if [ $INDEX -ge $VIDEO_START_INDEX ] && [ $INDEX -le $LIMIT ]; then
        # Create a unique config for this video
        VIDEO_BASENAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
        OUTPUT_CONFIG="$CONFIG_DIR/tuning_config_${VIDEO_BASENAME}.json"

        # Create a custom config file using the template but with updated video path
        cat "$CONFIG_TEMPLATE" | \
            sed "s|\"reference_video_path\": \".*\"|\"reference_video_path\": \"$VIDEO_PATH\"|g" | \
            sed "s|\"output_config_id\": \".*\"|\"output_config_id\": \"best_${VIDEO_BASENAME}\"|g" > "$OUTPUT_CONFIG"

        echo "Created config: '$OUTPUT_CONFIG' for video: '$VIDEO_PATH'"
        echo "Submitting Slurm job..."
        sbatch "$SBATCH_SCRIPT" "$OUTPUT_CONFIG"
        echo "Job for '$VIDEO_PATH' submitted."
        echo "--------------------------------------------------"
    fi

    INDEX=$((INDEX + 1))
done

echo "All jobs have been submitted."