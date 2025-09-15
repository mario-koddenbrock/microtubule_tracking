import os
import shutil
import pandas as pd
import re
from collections import defaultdict

def anonymize_and_copy_files(source_dir, dest_dir):
    """
    Copies files from source_dir to dest_dir, anonymizing filenames and
    creating a mapping CSV.
    """
    # Define subdirectories
    subdirs = ['configs', 'images', 'image_masks', 'videos', 'video_masks']

    # Create destination directories if they don't exist
    for subdir in subdirs:
        os.makedirs(os.path.join(dest_dir, subdir), exist_ok=True)

    # --- 1. Identify unique video base names ---
    # We'll use the video files to get the list of unique videos.
    video_files_path = os.path.join(source_dir, 'videos')
    if not os.path.exists(video_files_path):
        print(f"Error: Source directory '{video_files_path}' not found.")
        return

    video_files = [f for f in os.listdir(video_files_path) if f.endswith('_video.tif')]

    # Extract base names (e.g., 'series_1_9uMporcTub_crop_1_rank_1')
    base_names = sorted([f.replace('_video.tif', '') for f in video_files])

    # --- 2. Process each video and its associated files ---
    mapping_data = []
    video_counter = 1

    for base_name in base_names:
        new_video_id = f"video_{video_counter:04d}"

        # --- Handle video, video_mask, and config files ---
        file_types = {
            'videos': '_video.tif',
            'video_masks': '_video_mask.tif', # Assuming this pattern
            'configs': '_config.json'
        }

        for subdir, suffix in file_types.items():
            original_filename = f"{base_name}{suffix}"
            new_filename = f"{new_video_id}{suffix.replace('_video', '')}"

            source_path = os.path.join(source_dir, subdir, original_filename)
            dest_path = os.path.join(dest_dir, subdir, new_filename)

            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                mapping_data.append({
                    'original_filename': original_filename,
                    'new_filename': new_filename,
                    'type': subdir,
                    'original_base_name': base_name,
                    'new_base_name': new_video_id
                })

        # --- Handle frame-based files (images and image_masks) ---
        frame_subdirs = ['images', 'image_masks']
        for subdir in frame_subdirs:
            frame_files_path = os.path.join(source_dir, subdir)
            if not os.path.exists(frame_files_path):
                continue

            # Find all frames for the current base_name
            frame_pattern = re.compile(f"^{re.escape(base_name)}_frame_(\\d+)\\.png$")

            # Group frames by base name to handle them together
            frames_for_base = []
            for f in os.listdir(frame_files_path):
                match = frame_pattern.match(f)
                if match:
                    frame_number = int(match.group(1))
                    frames_for_base.append((frame_number, f))

            # Sort by frame number to ensure correct order
            frames_for_base.sort()

            for frame_number, original_filename in frames_for_base:
                new_filename = f"{new_video_id}_frame_{frame_number:04d}.png"

                source_path = os.path.join(frame_files_path, original_filename)
                dest_path = os.path.join(dest_dir, subdir, new_filename)

                shutil.copy2(source_path, dest_path)
                mapping_data.append({
                    'original_filename': original_filename,
                    'new_filename': new_filename,
                    'type': subdir,
                    'original_base_name': base_name,
                    'new_base_name': new_video_id
                })

        video_counter += 1

    # --- 3. Create the mapping CSV file ---
    mapping_df = pd.DataFrame(mapping_data)
    csv_path = os.path.join(dest_dir, 'filename_mapping.csv')
    mapping_df.to_csv(csv_path, index=False)

    print(f"Successfully processed {video_counter - 1} videos.")
    print(f"Anonymized files copied to: {dest_dir}")
    print(f"Filename mapping saved to: {csv_path}")


if __name__ == "__main__":
    source_directory = "data/SynMT/synthetic/validation"
    destination_directory = "data/SynMT/public-small"

    # Run the main function
    anonymize_and_copy_files(source_directory, destination_directory)

