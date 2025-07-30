import os
from glob import glob
from pathlib import Path
from random import shuffle

import cv2

from file_io.utils import extract_frames


def main(data_path: str, output_path: str, num_frames: int = 10):

    video_files = (glob(os.path.join(data_path, "*.avi")) +
                   glob(os.path.join(data_path, "*.tif")) +
                     glob(os.path.join(data_path, "*.mp4")))

    if not video_files:
        print(f"No .avi or .tif videos found in '{data_path}'")
        return

    print(f"Found {len(video_files)} video files in '{data_path}'")
    os.makedirs(output_path, exist_ok=True)

    for video_path in video_files:
        print(f"\nProcessing: {video_path}")

        frames_list, fps = extract_frames(video_path)
        if not frames_list:
            print(f"  -> Skipping video, no frames were extracted.")
            return

        base_name = Path(video_path).stem

        shuffle(frames_list)
        frames = frames_list[:num_frames]

        for frame_idx, frame in enumerate(frames):

            frame_output_path = os.path.join(output_path, f"{base_name}_frame_{frame_idx:02d}.png")
            cv2.imwrite(frame_output_path, frame)



if __name__ == "__main__":

    data_path = os.path.join('data', 'mpi_converted', 'type_B')
    output_path = os.path.join('data', 'real')

    os.makedirs(output_path, exist_ok=True)
    main(data_path, output_path)
