import os
from glob import glob

import cv2

from file_io.utils import extract_frames


def convert_videos(data_path, output_path):
    video_files = glob(os.path.join(data_path, "*.avi")) + glob(os.path.join(data_path, "*.tif"))
    for video_path in video_files:
        print(f"Processing: {video_path}")
        frames, fps = extract_frames(video_path, color_mode="grayscale")

        # write frames to mp4 video
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output = os.path.join(output_path, f"{base_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_output, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            if len(frame.shape) == 2:
                # if frame is grayscale, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                # if frame is single channel, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
        writer.release()
        print(f"Video saved to: {video_output}")


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_path_A = os.path.join(script_dir, '..', 'data', 'mpi', 'type_A')
    data_path_B = os.path.join(script_dir, '..', 'data', 'mpi', 'type_B')
    output_path = os.path.join(script_dir, '..', 'data', 'mpi_converted')
    os.makedirs(output_path, exist_ok=True)

    # convert_videos(data_path_A, output_path)
    convert_videos(data_path_B, output_path)

