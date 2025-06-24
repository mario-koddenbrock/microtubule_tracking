import json
from typing import List

import cv2
import numpy as np
import tifffile


# Custom JSON encoder to handle non-serializable types like sets and numpy arrays.
class CustomJsonEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle common scientific data types.
    - Converts sets to lists.
    - Converts numpy integers and floats to standard Python types.
    - Converts numpy arrays to lists.
    """
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError for other types.
        return super().default(obj)


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
    elif video_path.lower().endswith((".tif", ".tiff")):
        data = tifffile.imread(video_path)
        # Assuming a default FPS for TIFF stacks, as it's not stored in the file.
        fps = 5

        # Handle different TIFF structures (e.g., Time x Channel x H x W or Time x H x W x Channel)
        if data.ndim == 4 and data.shape[1] <= 4: # Likely T x C x H x W
             channel = 0 # or 1, depending on which channel to use
             global_max = np.max(data[:, channel])
             red_max = np.max(data[:, 0]) if data.shape[1] > 0 else 1
             green_max = np.max(data[:, 1]) if data.shape[1] > 1 else 1

             for i in range(data.shape[0]):
                 red = ((data[i, 0] / red_max) * 255).astype(np.uint8) if data.shape[1] > 0 else np.zeros_like(data[i,0])
                 green = ((data[i, 1] / green_max) * 255).astype(np.uint8) if data.shape[1] > 1 else np.zeros_like(data[i,0])
                 blue = np.zeros(data[i, 0].shape, dtype=np.uint8)

                 if color_mode == "grayscale":
                     frame = red if channel == 0 else green
                 else:
                     frame = cv2.merge((blue, green, red)) # OpenCV uses BGR order
                 frames.append(frame)
        elif data.ndim == 3: # Likely T x H x W (grayscale)
            for i in range(data.shape[0]):
                frames.append(data[i])
        else:
            raise ValueError(f"Unsupported TIFF shape: {data.shape}")
    else:
        raise ValueError("Unsupported video format. Only AVI and TIFF/TIF files are supported.")

    # Final color mode conversion
    if frames and frames[0] is not None:
        if color_mode == "grayscale" and frames[0].ndim == 3:
            frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        elif color_mode in ("rgb", "bgr") and frames[0].ndim == 2:
            color_conversion = cv2.COLOR_GRAY2RGB if color_mode == "rgb" else cv2.COLOR_GRAY2BGR
            frames = [cv2.cvtColor(f, color_conversion) for f in frames]

    return frames, fps


def save_ground_truth(gt_data: List[dict], json_output_path: str):
    """
    Saves annotation data (for detection/tracking) as a JSON file.
    Uses a custom encoder to handle special data types like sets and numpy arrays.
    """
    with open(json_output_path, "w") as fh:
        # Pass the custom encoder class to json.dump using the `cls` argument.
        json.dump(gt_data, fh, indent=2, cls=CustomJsonEncoder)
    print(f"Saved JSON ground truth to: {json_output_path}")