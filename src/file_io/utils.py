import json
from typing import List

import cv2
import numpy as np
import tifffile


def extract_frames(video_path):
    frames = []
    if video_path.lower().endswith(".avi"):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif video_path.lower().endswith(".tif"):
        data = tifffile.imread(video_path)
        for i in range(data.shape[0]):
            frame = (data[i, 1, :, :] / 65535.0 * 255).astype(np.uint8)
            frames.append(frame)
    return frames


def save_ground_truth(gt_data: List[dict], output_path: str):
    with open(output_path, "w") as fh:
        json.dump(gt_data, fh, indent=2)