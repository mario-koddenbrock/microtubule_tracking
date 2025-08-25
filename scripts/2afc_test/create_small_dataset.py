import os
import shutil
import random

import numpy as np


def create_small_dataset(sample_fraction=0.1):
    """
    Creates a 'small' dataset by randomly sampling a fraction of the 'full' dataset.

    The script ensures that the same random selection of crops is applied to
    both the 'all_frames' (directories) and 'single_frame' (files) data.
    """
    # Define base paths
    base_path = os.path.join('data', 'SynMT', 'real')
    full_path = os.path.join(base_path, 'full')
    small_path = os.path.join(base_path, 'small')

    # Define source paths
    full_all_frames_path = os.path.join(full_path, 'all_frames')
    full_single_frame_path = os.path.join(full_path, 'single_frame')

    # Define destination paths
    small_all_frames_path = os.path.join(small_path, 'all_frames')
    small_single_frame_path = os.path.join(small_path, 'single_frame')

    # --- 1. Validation and Setup ---
    print("--- Starting Dataset Creation ---")

    if not os.path.isdir(full_all_frames_path):
        print(f"Error: Source 'all_frames' directory not found at: {full_all_frames_path}")
        return

    if not os.path.isdir(full_single_frame_path):
        print(f"Error: Source 'single_frame' directory not found at: {full_single_frame_path}")
        return

    # Create destination directories if they don't exist
    print("Creating destination directories...")
    os.makedirs(small_all_frames_path, exist_ok=True)
    os.makedirs(small_single_frame_path, exist_ok=True)
    print(f"  - Created: {small_all_frames_path}")
    print(f"  - Created: {small_single_frame_path}")

    # --- 2. Select a Random Subset of Crops ---

    # The most reliable way to get all crop names is from the 'all_frames' subdirectories
    try:
        all_crops = [
            d for d in os.listdir(full_all_frames_path)
            if os.path.isdir(os.path.join(full_all_frames_path, d))
        ]
    except FileNotFoundError:
        print(f"Error: Could not list directories in {full_all_frames_path}.")
        return

    if not all_crops:
        print("No crop directories found in 'all_frames'. Nothing to do.")
        return

    # Calculate the sample size
    total_crops = len(all_crops)
    sample_size = max(1, int(np.ceil(total_crops * sample_fraction)))  # Ensure at least one is selected

    print(f"\nFound {total_crops} total crops in '{full_all_frames_path}'.")
    print(f"Selecting {sample_fraction * 100:.0f}% ({sample_size} crops) for the 'small' dataset.")

    # Use a fixed seed for reproducible random selection
    random.seed(7)
    selected_crops = random.sample(all_crops, sample_size)

    print("\nSelected crops:")
    for crop in sorted(selected_crops):
        print(f"- {crop}")

    # --- 3. Copy Selected Data ---

    print("\nCopying data for selected crops...")
    copied_single_frames = 0

    # Get all filenames from single_frame once to avoid re-reading the directory in the loop
    all_single_frame_files = os.listdir(full_single_frame_path)

    for i, crop_name in enumerate(selected_crops, 1):
        print(f"[{i}/{sample_size}] Processing '{crop_name}'...")

        # a) Copy the entire folder from all_frames
        src_folder = os.path.join(full_all_frames_path, crop_name)
        dest_folder = os.path.join(small_all_frames_path, crop_name)
        if os.path.exists(src_folder):
            shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
            print(f"  - Copied directory to '{small_all_frames_path}'")
        else:
            print(f"  - Warning: Source directory not found: {src_folder}")

        # b) Copy all corresponding files from single_frame
        files_found = 0
        for filename in all_single_frame_files:
            # A file belongs to a crop if it starts with the crop name followed by an underscore
            if filename.startswith(crop_name + '_'):
                src_file = os.path.join(full_single_frame_path, filename)
                dest_file = os.path.join(small_single_frame_path, filename)
                shutil.copy2(src_file, dest_file)  # copy2 preserves metadata
                files_found += 1

        if files_found > 0:
            print(f"  - Copied {files_found} files to '{small_single_frame_path}'")
            copied_single_frames += files_found
        else:
            print(f"  - Warning: No corresponding files found in '{full_single_frame_path}'")

    # --- 4. Final Summary ---
    print("\n--- Dataset Creation Complete ---")
    print(f"Total crops copied: {len(selected_crops)}")
    print(f"Total single-frame images copied: {copied_single_frames}")
    print(f"New 'small' dataset is ready at: {small_path}")


if __name__ == '__main__':
    # change the fraction here, e.g., 0.2 for 20%
    create_small_dataset(sample_fraction=0.1)