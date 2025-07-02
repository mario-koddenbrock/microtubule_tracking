import os
from glob import glob
from pathlib import Path

import cv2
from tqdm import tqdm

from file_io.utils import extract_frames


def convert_videos(data_path, output_path):
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

        # Request BGR frames directly, as that's what cv2.imwrite and VideoWriter expect.
        # This simplifies the loop later.
        frames, fps = extract_frames(video_path, color_mode="rgb")

        if not frames:
            print(f"  -> Skipping video, no frames were extracted.")
            continue

        base_name = Path(video_path).stem

        # --- 1. Set up MP4 Video Output (existing logic) ---
        video_output_path = os.path.join(output_path, f"{base_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Ensure fps is at least 1 to avoid errors with static tiff images
        writer_fps = max(1, fps)
        writer = cv2.VideoWriter(video_output_path, fourcc, writer_fps, (frames[0].shape[1], frames[0].shape[0]))

        # --- 2. Set up PNG Frame Export Directory (new logic) ---
        frame_output_dir = os.path.join(output_path, base_name)
        os.makedirs(frame_output_dir, exist_ok=True)
        print(f"  -> Exporting {len(frames)} frames to: {frame_output_dir}")

        # --- 3. Loop Through Frames to Write Both Video and PNGs ---
        # Use enumerate to get a frame index for naming the PNG files.
        for i, frame in enumerate(tqdm(frames, desc="  Exporting")):

            # This check ensures that even if extract_frames returns grayscale,
            # it will be correctly converted before writing.
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Action A: Write the frame to the MP4 video
            writer.write(frame)

            # Action B: Write the frame as a PNG image (new logic)
            # Use zfill or f-string formatting to pad filenames with zeros (e.g., 00001.png)
            # for correct sorting in file explorers.
            frame_filename = f"{i:05d}.png"
            frame_output_path = os.path.join(frame_output_dir, frame_filename)
            cv2.imwrite(frame_output_path, frame)

        writer.release()
        print(f"  -> MP4 video saved to: {video_output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define paths relative to the script location
    data_path_A = os.path.join(script_dir, '..', 'data', 'mpi', 'type_A')
    data_path_B = os.path.join(script_dir, '..', 'data', 'mpi', 'type_B')
    output_path_A = os.path.join(script_dir, '..', 'data', 'mpi_converted', 'type_A')
    output_path_B = os.path.join(script_dir, '..', 'data', 'mpi_converted', 'type_B')

    os.makedirs(output_path_A, exist_ok=True)
    os.makedirs(output_path_B, exist_ok=True)

    # convert_videos(data_path_A, output_path_A)
    convert_videos(data_path_B, output_path_B)