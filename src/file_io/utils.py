import json
from typing import List

import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def extract_frames(video_path, color_mode: str = "grayscale") -> List[np.ndarray]:
    frames = []
    if video_path.lower().endswith(".avi"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif video_path.lower().endswith(".tif"):
        data = tifffile.imread(video_path)
        fps = 5 # Default FPS for TIFF files
        channel = 0  # or 0, depending on which to use
        global_max = np.max(data[:, channel])
        red_max = np.max(data[:, 0])
        green_max = np.max(data[:, 1])

        for i in range(data.shape[0]):
            # if color_mode == "grayscale":
            #     frame = data[i, channel]
            #     frame = ((frame / global_max) * 255).astype(np.uint8)
            # else:
            red = data[i, 0]
            green = data[i, 1]
            blue = np.zeros(data[i, 0].shape, dtype=np.uint8)
            red = ((red / red_max) * 255).astype(np.uint8)
            green = ((green / green_max) * 255).astype(np.uint8)
            if color_mode == "grayscale":
                if channel == 0:
                    frame = red
                else:
                    frame = green
            else:
                frame = cv2.merge((red, green, blue))

            frames.append(frame)
    else:
        raise ValueError("Unsupported video format. Only AVI and TIFF files are supported.")

    if color_mode == "grayscale":
        if frames[0].ndim == 3:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    elif color_mode == "rgb":
        if frames[0].ndim == 2:
            frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in frames]
    elif color_mode == "bgr":
        if frames[0].ndim == 2:
            frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in frames]

    return frames, fps


def save_ground_truth(gt_data: List[dict], output_path: str):
    with open(output_path, "w") as fh:
        json.dump(gt_data, fh, indent=2)