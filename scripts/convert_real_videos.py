import os
from glob import glob
from pathlib import Path
from random import shuffle

import cv2
import numpy as np

from file_io.utils import extract_frames


def process_all(data_path, output_path):
    """
    Converts all .avi and .tif videos in a directory to .mp4 format,
    and also exports each frame as a PNG image into a dedicated subfolder.
    """
    video_files = glob(os.path.join(data_path, "*.avi")) + glob(os.path.join(data_path, "*.tif"))

    if not video_files:
        print(f"No .avi or .tif videos found in '{data_path}'")
        return

    for video_path in video_files:
        print(f"\nProcessing: {video_path}")

        split_and_convert(output_path, video_path)


def split_and_convert(output_path, video_path, num_crops=3, num_frames=10):

    frames_list, fps = extract_frames(video_path, num_crops=num_crops, crop_size=(512, 512))
    if not frames_list:
        print(f"  -> Skipping video, no frames were extracted.")
        return

    is_single_of_videos = isinstance(frames_list[0], np.ndarray)
    if is_single_of_videos:
        frames_list = [frames_list]

    base_name = Path(video_path).stem
    for frames_idx, frames in enumerate(frames_list):
        if not frames:
            print(f"  -> Skipping video, no frames were extracted.")
            return

        # --- 1. Set up MP4 Video Output (existing logic) ---
        video_output_path = os.path.join(output_path, f"{base_name}_crop_{frames_idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Ensure fps is at least 1 to avoid errors with static tiff images
        writer_fps = max(1, fps)
        writer = cv2.VideoWriter(video_output_path, fourcc, writer_fps, (frames[0].shape[1], frames[0].shape[0]))

        # --- 2. Set up PNG Frame Export Directory (new logic) ---
        frame_output_dir = os.path.join(output_path, base_name)
        os.makedirs(frame_output_dir, exist_ok=True)

        # --- 3. Loop Through Frames to Write Both Video and PNGs ---
        # Use enumerate to get a frame index for naming the PNG files.
        for i, frame in enumerate(frames):
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Convert frame to uint8 if necessary
            if frame.dtype != 'uint8':
                frame_uint8 = (255 * frame).astype('uint8')
            else:
                frame_uint8 = frame

            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"  -> MP4 video saved to: {video_output_path}")


        # shuffle the frames to create a random video
        shuffle(frames)
        random_frames = frames[:num_frames]
        for i, frame in enumerate(random_frames):

            # Convert frame to uint8 if necessary
            if frame.dtype != 'uint8':
                frame_uint8 = (255 * frame).astype('uint8')
            else:
                frame_uint8 = frame

            frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

            # Write the frame as a PNG image
            frame_filename = f"{i:05d}.png"
            frame_output_path = os.path.join(frame_output_dir, frame_filename)
            if i < num_frames:
                cv2.imwrite(frame_output_path, frame_uint8)





if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define paths relative to the script location

    folder_names = [
        '250523 Exemplary IRM Images',
        # '250801 Additional Images from Dominik',
        'Simone'
    ]

    for folder_name in folder_names:
        data_path = os.path.join("data", "mpi", folder_name)
        output_path = os.path.join("data", "mpi_converted", folder_name)

        os.makedirs(output_path, exist_ok=True)
        process_all(data_path, output_path)