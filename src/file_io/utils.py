import json
import os
from typing import List, Optional

import cv2
import imageio
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


def save_ground_truth(
        gt_data: List[dict],
        json_output_path: str,
        mask_frames: Optional[List[np.ndarray]] = None,
):
    """
    Saves ground truth data.

    1. Saves annotation data (for detection/tracking) as a JSON file.
    2. If mask_frames are provided, saves the raw integer instance masks
       as a multi-page TIFF file.
    """
    # --- 1. Save the JSON ground truth  ---
    with open(json_output_path, "w") as fh:
        json.dump(gt_data, fh, indent=2)
    print(f"Saved JSON ground truth to: {json_output_path}")

    # --- 2. Save the raw masks to a multipage TIFF ---
    if mask_frames and len(mask_frames) > 0:
        # Generate the TIFF file path from the JSON path
        # e.g., 'series_1_gt.json' -> 'series_1_gt_masks.tif'
        base_path, _ = os.path.splitext(json_output_path)
        tiff_output_path = f"{base_path}_masks.tif"

        print(f"Saving {len(mask_frames)} raw integer masks to: {tiff_output_path}...")

        # Use imageio.mimwrite to save a stack of images into one file
        # The TIFF format will preserve the uint16 data type perfectly.
        imageio.mimwrite(tiff_output_path, mask_frames)

        print("TIFF saving complete.")