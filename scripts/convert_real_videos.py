import os
from glob import glob
from pathlib import Path

import numpy as np

from mt.file_io.utils import extract_frames
from mt.file_io.video import write_video, write_png_frames


def process_video_file(video_path, output_path, num_crops=3, num_frames=10, crop_size=(512, 512)):
    """Extract frames, write video and PNGs for a single video file."""
    frames_list, fps = extract_frames(video_path, num_crops=num_crops, crop_size=crop_size, preprocess=False)
    if not frames_list:
        print(f"  -> Skipping video, no frames were extracted.")
        return

    # Ensure frames_list is always a list of lists
    if isinstance(frames_list[0], np.ndarray):
        frames_list = [frames_list]

    base_name = Path(video_path).stem
    for crop_idx, frames in enumerate(frames_list):
        if not frames:
            print(f"  -> Skipping crop {crop_idx}, no frames.")
            continue

        video_output_path = os.path.join(output_path, f"{base_name}_crop_{crop_idx}.mp4")
        success = write_video(frames, video_output_path, fps, img_size=crop_size)
        if success:
            print(f"  -> MP4 video saved to: {video_output_path}")
        else:
            print(f"  -> Failed to save MP4 video: {video_output_path}")

        frame_output_dir = os.path.join(output_path, base_name)
        write_png_frames(frames, frame_output_dir, num_frames)


def process_all(data_path, output_path, num_crops=3, num_frames=10):
    """Process all .avi and .tif videos in a directory."""
    video_files = glob(os.path.join(data_path, "*.avi")) + glob(os.path.join(data_path, "*.tif"))

    if not video_files:
        print(f"No .avi or .tif videos found in '{data_path}'")
        return

    for video_path in video_files:
        print(f"\nProcessing: {video_path}")
        try:
            process_video_file(video_path, output_path, num_crops=num_crops, num_frames=num_frames)
        except Exception as e:
            print(f"  -> Error processing {video_path}: {e}")


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define paths relative to the script location

    folder_names = [
        '250523 Exemplary IRM Images',
        # '250801 Additional Images from Dominik',
        # 'Simone'
    ]

    for folder_name in folder_names:
        data_path = os.path.join("data", "mpi", folder_name)
        output_path = os.path.join("data", "mpi_converted", folder_name)

        os.makedirs(output_path, exist_ok=True)
        process_all(data_path, output_path)


if __name__ == "__main__":
    main()
