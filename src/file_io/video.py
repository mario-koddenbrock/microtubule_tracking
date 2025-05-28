from typing import Generator

import cv2
import numpy as np


def write_video(frames: Generator[np.ndarray, None, None], output_path: str, fps: int, img_size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, img_size[::-1])
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    writer.release()
