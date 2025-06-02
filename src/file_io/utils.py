import json
from typing import List

import cv2
import numpy as np
import tifffile


def extract_frames(video_path):
    frames = []
    if video_path.lower().endswith(".avi"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif video_path.lower().endswith(".tif"):
        data = tifffile.imread(video_path)
        fps = 5 # Default FPS for TIFF files
        frames = []
        channel = 0  # or 0, depending on which to visualize
        global_max = np.max(data[:, channel])
        for i in range(data.shape[0]):
            frame = data[i, channel]
            frame_8bit = ((frame / global_max) * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)
            frames.append(frame_bgr)

    return frames, fps


def save_ground_truth(gt_data: List[dict], output_path: str):
    with open(output_path, "w") as fh:
        json.dump(gt_data, fh, indent=2)