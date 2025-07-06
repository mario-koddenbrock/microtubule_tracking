import json
import logging
import os
from typing import List, Tuple, Any

import cv2
import numpy as np
import tifffile

logger = logging.getLogger(f"microtuble_tracking.{__name__}")


# Custom JSON encoder to handle non-serializable types like sets and numpy arrays.
class CustomJsonEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle common scientific data types.
    - Converts sets to lists.
    - Converts numpy integers and floats to standard Python types.
    - Converts numpy arrays to lists.
    """

    def default(self, obj: Any) -> Any:
        try:
            if isinstance(obj, set):
                logger.debug(f"JSONEncoder: Converting set to list.")
                return list(obj)
            if isinstance(obj, np.integer):
                logger.debug(f"JSONEncoder: Converting numpy integer {obj} to int.")
                return int(obj)
            if isinstance(obj, np.floating):
                logger.debug(f"JSONEncoder: Converting numpy float {obj} to float.")
                return float(obj)
            if isinstance(obj, np.ndarray):
                logger.debug(f"JSONEncoder: Converting numpy array of shape {obj.shape} to list.")
                return obj.tolist()
            # Let the base class default method raise the TypeError for other types.
            return super().default(obj)
        except Exception as e:
            logger.error(f"JSONEncoder: Error serializing object of type {type(obj)}: {e}", exc_info=True)
            raise  # Re-raise to prevent partial/corrupted JSON output


def extract_frames(video_path: str, color_mode: str = "grayscale") -> Tuple[List[np.ndarray], int]:
    """
    Extracts frames from video files (.avi, .mp4, .mov, .mkv, .tif/.tiff).

    Args:
        video_path (str): Path to the video file.
        color_mode (str): Desired color mode for output frames ('grayscale', 'rgb', 'bgr').

    Returns:
        Tuple[List[np.ndarray], int]: A tuple containing a list of frames as numpy arrays
                                       and the frames per second (FPS).

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video format is unsupported or TIFF shape is unexpected.
        Exception: For other errors during video reading.
    """
    logger.info(f"Extracting frames from: {video_path} (desired color_mode: '{color_mode}').")

    if not os.path.isfile(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames: List[np.ndarray] = []
    fps: int = 0

    try:
        if video_path.lower().endswith((".avi", ".mp4", ".mov", ".mkv")):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file with OpenCV: {video_path}")
                raise IOError(f"Could not open video file: {video_path}")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            logger.debug(f"Opened video with OpenCV. FPS: {fps}.")

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
            cap.release()
            logger.info(f"Successfully extracted {frame_count} frames from OpenCV video file.")

        elif video_path.lower().endswith((".tif", ".tiff")):
            logger.debug(f"Attempting to read TIFF stack with tifffile: {video_path}")
            data = tifffile.imread(video_path)
            # Assuming a default FPS for TIFF stacks, as it's not stored in the file.
            fps = 5
            logger.info(f"Loaded TIFF stack of shape {data.shape}. Defaulting FPS to {fps}.")

            # Handle different TIFF structures (e.g., Time x Channel x H x W or Time x H x W x Channel)
            if data.ndim == 4:
                # Assuming T x C x H x W (common for multi-channel microscopy)
                if data.shape[1] <= 4:  # Max 4 channels (e.g., RGBA)
                    logger.debug(f"TIFF: Detected 4D data (T x C x H x W), Channels: {data.shape[1]}.")
                    # Dynamically determine max values for scaling per channel
                    red_max = np.max(data[:, 0]) if data.shape[1] > 0 else 1
                    green_max = np.max(data[:, 1]) if data.shape[1] > 1 else 1
                    blue_max = np.max(data[:, 2]) if data.shape[1] > 2 else 1  # Add blue channel if present

                    # Consider global max for consistent scaling, or channel-specific
                    # For simplicity, scaling independently for now.

                    for i in range(data.shape[0]):  # Iterate through time (frames)
                        # Scale channels to 0-255 range and convert to uint8
                        red = ((data[i, 0].astype(np.float32) / red_max) * 255).astype(np.uint8) if data.shape[
                                                                                                        1] > 0 else np.zeros_like(
                            data[i, 0], dtype=np.uint8)
                        green = ((data[i, 1].astype(np.float32) / green_max) * 255).astype(np.uint8) if data.shape[
                                                                                                            1] > 1 else np.zeros_like(
                            data[i, 1], dtype=np.uint8)
                        blue = ((data[i, 2].astype(np.float32) / blue_max) * 255).astype(np.uint8) if data.shape[
                                                                                                          1] > 2 else np.zeros_like(
                            data[i, 0], dtype=np.uint8)  # Check blue channel

                        # Assuming 3-channel output or grayscale conversion from a primary channel
                        if color_mode == "grayscale":
                            # Default to green channel if available, else red, else first
                            if data.shape[1] > 1:
                                frame = green  # Common for GFP in microscopy
                            elif data.shape[1] > 0:
                                frame = red
                            else:
                                frame = np.zeros_like(data[i, 0], dtype=np.uint8)
                            logger.debug(
                                f"TIFF frame {i}: Converted to grayscale from channel {1 if data.shape[1] > 1 else 0}.")
                        else:  # RGB or BGR output
                            frame = cv2.merge((blue, green, red))  # OpenCV uses BGR order
                            logger.debug(f"TIFF frame {i}: Merged to BGR from channels 0,1,2.")
                        frames.append(frame)
                else:  # e.g., T x H x W x C
                    if data.shape[3] == 3:  # Assuming RGB/BGR
                        for i in range(data.shape[0]):
                            # tifffile reads as RGB, OpenCV expects BGR. So, convert if needed.
                            if color_mode in ("bgr", "grayscale"):  # If target is BGR or grayscale, convert from RGB
                                frames.append(cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR))
                                logger.debug(f"TIFF frame {i}: Converted from RGB to BGR.")
                            else:  # If target is RGB, keep as is
                                frames.append(data[i])
                                logger.debug(f"TIFF frame {i}: Retained as RGB.")
                    elif data.shape[3] == 1:  # Grayscale
                        for i in range(data.shape[0]):
                            frames.append(data[i].squeeze())  # Remove channel dimension (H, W, 1) -> (H, W)
                            logger.debug(f"TIFF frame {i}: Squeezed 1-channel grayscale.")
                    else:
                        msg = f"Unsupported 4D TIFF shape (T x H x W x C) with C={data.shape[3]}. Expected C=1 or C=3."
                        logger.error(msg)
                        raise ValueError(msg)

            elif data.ndim == 3:  # Likely T x H x W (grayscale)
                for i in range(data.shape[0]):
                    frames.append(data[i])
                logger.debug(f"TIFF: Detected 3D data (T x H x W), loaded {len(frames)} grayscale frames.")
            else:
                msg = f"Unsupported TIFF shape: {data.shape}. Expected 3D (T x H x W) or 4D (T x C x H x W / T x H x W x C)."
                logger.error(msg)
                raise ValueError(msg)
            logger.info(f"Successfully extracted {len(frames)} frames from TIFF file.")

        else:
            msg = f"Unsupported video format '{os.path.splitext(video_path)[1]}'. Only AVI, MP4, MOV, MKV and TIFF/TIF files are supported."
            logger.error(msg)
            raise ValueError(msg)

    except Exception as e:
        logger.error(f"An error occurred while extracting frames from {video_path}: {e}", exc_info=True)
        # Clear frames to indicate failure
        frames = []
        fps = 0
        raise  # Re-raise the exception after logging

    # Final color mode conversion to ensure requested output format
    if frames and frames[0] is not None:
        first_frame_ndim = frames[0].ndim
        logger.debug(
            f"Performing final color mode conversion. First frame ndim: {first_frame_ndim}, desired: '{color_mode}'.")

        # If output color mode is grayscale but frames are 3-channel
        if color_mode == "grayscale" and first_frame_ndim == 3:
            frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in
                      frames]  # Assuming initial 3-channel is BGR from OpenCV
            logger.debug(f"Converted all frames to grayscale.")
        # If output color mode is color but frames are grayscale
        elif color_mode in ("rgb", "bgr") and first_frame_ndim == 2:
            color_conversion = cv2.COLOR_GRAY2RGB if color_mode == "rgb" else cv2.COLOR_GRAY2BGR
            frames = [cv2.cvtColor(f, color_conversion) for f in frames]
            logger.debug(f"Converted all frames from grayscale to {color_mode.upper()}.")
        elif (color_mode == "rgb" and first_frame_ndim == 3 and frames[0].shape[2] == 3
              and not (video_path.lower().endswith((".tif",
                                                    ".tiff")) and "rgb2bgr_done" in video_path.lower())):  # A crude way to avoid re-converting TIFFs that were already BGR converted
            # If target is RGB but frames might be BGR (from OpenCV read)
            frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            logger.debug(f"Converted all frames from BGR to RGB.")
        elif (color_mode == "bgr" and first_frame_ndim == 3 and frames[0].shape[2] == 3
              and not (video_path.lower().endswith((".tif", ".tiff")) and "rgb2bgr_done" in video_path.lower())):
            # If target is BGR and frames are already BGR (from OpenCV), do nothing.
            # If from tifffile, might need RGB2BGR. Assume tifffile RGB, so convert to BGR.
            if video_path.lower().endswith((".tif", ".tiff")):
                frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
                logger.debug(f"Converted all frames from TIFF (RGB assumed) to BGR.")
            else:
                logger.debug(f"Frames are already in target BGR format (from OpenCV).")
        else:
            logger.debug(
                f"No specific color conversion needed for target '{color_mode}' and source frame dim {first_frame_ndim}.")

    logger.info(f"Finished extracting {len(frames)} frames with FPS {fps}.")
    return frames, fps


def save_ground_truth(gt_data: List[dict], json_output_path: str):
    """
    Saves annotation data (for detection/tracking) as a JSON file.
    Uses a custom encoder to handle special data types like sets and numpy arrays.

    Args:
        gt_data (List[dict]): List of ground truth dictionaries.
        json_output_path (str): Full path to save the JSON file.
    """
    logger.info(f"Attempting to save JSON ground truth to: {json_output_path}")
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(json_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory for ground truth: {output_dir}")

        with open(json_output_path, "w") as fh:
            # Pass the custom encoder class to json.dump using the `cls` argument.
            json.dump(gt_data, fh, indent=2, cls=CustomJsonEncoder)
        logger.info(f"Successfully saved JSON ground truth to: {json_output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON ground truth to {json_output_path}: {e}", exc_info=True)
        # Re-raise the exception to signal failure to the caller
        raise