import json
import logging
import os
from typing import List, Tuple, Any

import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import exposure

logger = logging.getLogger(f"mt.{__name__}")


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


def normalize_contrast_stretch(
        data: np.ndarray,
        min_val: float,
        max_val: float,
        bounds: Tuple[float, float] = (0.0, 0.75)
) -> np.ndarray:
    """
    Normalizes data based on the contrast stretching method from the paper.

    This function implements the piecewise normalization:
    - Values below a lower threshold are set to 0.
    - Values above an upper threshold are set to 1.
    - Values in between are linearly scaled to fit the [0, 1] range.

    Args:
        data (np.ndarray): The input image data (a single channel).
        min_val (float): The global minimum intensity for this channel.
        max_val (float): The global maximum intensity for this channel.
        bounds (Tuple[float, float]): The (t1, t2) parameters, representing the
                                      fractional bounds of the intensity range to stretch.
                                      Defaults to (0.0, 0.75) as per the paper.

    Returns:
        np.ndarray: The normalized data in float format, with values in [0, 1].
    """
    t1, t2 = bounds
    delta_i = max_val - min_val

    t1 = 0.1
    t2 = 95
    p1, p2 = np.percentile(data, (t1, t2))  # Calculate percentiles for rescaling
    img_rescale = exposure.rescale_intensity(data.copy(), in_range=(p1, p2), out_range=(0, 1))

    # # Avoid division by zero for flat images or invalid bounds
    # if delta_i == 0 or t1 >= t2:
    #     return np.zeros_like(data, dtype=np.float32)
    #
    # # Define the lower and upper intensity thresholds
    # lower_bound = min_val + t1 * delta_i
    # upper_bound = min_val + t2 * delta_i
    #
    # # Start with a copy of the data to avoid modifying the original
    # normalized_data = data.astype(np.float32)
    #
    # # Apply the piecewise function using efficient NumPy masking
    #
    # # Condition 1: I <= lower_bound (maps to 0)
    # # Condition 2: I >= upper_bound (maps to 1)
    # # We can use np.clip to handle both of these conditions at once.
    # # It sets all values below `lower_bound` to `lower_bound` and all
    # # values above `upper_bound` to `upper_bound`.
    # clipped_data = np.clip(normalized_data, lower_bound, upper_bound)
    #
    # # Condition 3 (Otherwise): Linearly scale the "in-between" values
    # # The formula is: (I - lower_bound) / (upper_bound - lower_bound)
    # # This works because anything that was clipped to `lower_bound` will become 0,
    # # and anything clipped to `upper_bound` will become 1.
    # img_rescale = (clipped_data - lower_bound) / (upper_bound - lower_bound)

    return img_rescale

def process_tiff_video(
        video_path: str,
        num_crops: int = 3,
        crop_size: Tuple[int, int] = (512, 512),
        norm_bounds: List[Tuple[float, float]] = [(0.0, 0.75), (0.0, 0.75)]
) -> List[List[np.ndarray]]:
    """
    Reads a TIFF video, performs repeated random cropping, applies contrast-stretching
    normalization as described in the paper, and returns processed videos.

    Args:
        video_path (str): The file path to the TIFF video.
        num_crops (int): The number of different random video crops to generate.
        crop_size (Tuple[int, int]): The (height, width) for the random crops.
        norm_bounds (Tuple[float, float]): The (t1, t2) normalization bounds.
                                           Defaults to (0.0, 0.75) from the paper.

    Returns:
        List[List[np.ndarray]]: List of videos (list of frames). Each frame is a
                                NumPy array (crop_h, crop_w, 3) with float values in [0, 1].
    """
    # --- Step 1: Reading and Validation (same as before) ---
    try:
        data = tifffile.imread(video_path)
    except Exception as e:
        logger.error(f"Failed to read TIFF file {video_path}: {e}")
        return []

    if num_crops <= 1:
        logger.warning(f"num_crops is set to {num_crops}. No cropping will be performed.")
        return [[data]]

    logger.debug(f"Loaded TIFF stack of shape {data.shape}.")
    if data.ndim not in [3, 4]:
        logger.error(f"Unsupported TIFF dimension: {data.ndim}.")
        return []

    num_frames = data.shape[0]
    orig_h, orig_w = data.shape[-2:]
    crop_h, crop_w = crop_size
    if orig_h < crop_h or orig_w < crop_w:
        logger.error(f"Crop size {crop_size} is larger than frame size {(orig_h, orig_w)}.")
        return []

    # --- Step 2: Pre-calculate global min/max for consistent normalization ---
    min_vals, max_vals = [], []
    if data.ndim == 4:  # T x C x H x W
        for c in range(data.shape[1]):
            min_vals.append(np.min(data[:, c, :, :]))
            max_vals.append(np.max(data[:, c, :, :]))
    else:  # 3D data: T x H x W (Grayscale)
        min_vals.append(np.min(data))
        max_vals.append(np.max(data))

    all_cropped_videos = []

    # --- Step 3: Cropping Loop (same as before) ---
    for i in range(num_crops):
        top = np.random.randint(0, orig_h - crop_h + 1)
        left = np.random.randint(0, orig_w - crop_w + 1)
        logger.info(f"Generating crop #{i + 1}/{num_crops} at (top={top}, left={left})")

        current_cropped_frames = []
        for t in range(num_frames):
            if data.ndim == 4:
                channels_to_process = [data[t, c, :, :] for c in range(min(3, data.shape[1]))]
            else:
                channels_to_process = [data[t, :, :]]

            # Crop each channel
            cropped_channels = [chan[top:top + crop_h, left:left + crop_w] for chan in channels_to_process]

            # --- Step 4: Normalization (UPDATED) ---
            # Apply the paper's contrast stretching method to each channel
            normalized_channels = [
                normalize_contrast_stretch(chan, min_vals[c_idx], max_vals[c_idx], norm_bounds[c_idx])
                for c_idx, chan in enumerate(cropped_channels)
            ]

            # --- Step 5: Combine to RGB ---
            num_chans = len(normalized_channels)

            if num_chans >= 3:
                # Standard case: We have R, G, and B. Take the first three.
                rgb_frame = np.stack(normalized_channels[:3], axis=-1)

            elif num_chans == 2:

                # Case 1: At least 2 channels (assume R and G are the first two)
                # This handles 2-channel, 3-channel (RGB), 4-channel, etc.
                r_chan_in = normalized_channels[0]
                g_chan_in = normalized_channels[1]

                # The Green channel is the base intensity
                final_g = g_chan_in

                # The Red channel is an overlay. Use np.maximum to merge without clipping.
                # A pixel is red if it was bright in *either* the original red or green.
                final_r = np.maximum(r_chan_in, g_chan_in)

                # The Blue channel is black.
                final_b = final_g

                rgb_frame = np.stack([final_r, final_g, final_b], axis=-1)
                # plt.imshow(rgb_frame, cmap='viridis', vmin=0, vmax=1)
                # plt.axis('off')
                # plt.title(f"Crop #{i + 1} Frame {t + 1} (R+G Overlay)")
                # plt.show()


            elif num_chans == 1:
                # Grayscale case: Duplicate the single channel into R, G, and B.
                gray_chan = normalized_channels[0]
                rgb_frame = np.stack([gray_chan, gray_chan, gray_chan], axis=-1)

            else:  # num_chans == 0 (empty)
                # This is an edge case, create an empty black frame.
                rgb_frame = np.zeros((crop_h, crop_w, 3), dtype=np.float32)

            current_cropped_frames.append(rgb_frame)

        all_cropped_videos.append(current_cropped_frames)

    logger.info("Processing complete.")
    return all_cropped_videos


def extract_frames(video_path: str, num_crops:int = 1, crop_size=(512, 512)) -> Tuple[List[List[np.ndarray]], int]:

    logger.debug(f"Extracting frames from: {video_path}")

    if not os.path.isfile(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    fps: int = 10

    try:
        if video_path.lower().endswith((".avi", ".mp4", ".mov", ".mkv")):
            frames, fps = process_avi_video(fps, video_path)
            ndim = frames[0].ndim

        elif video_path.lower().endswith((".tif", ".tiff")):
            frames = process_tiff_video(
                video_path=video_path,
                num_crops=num_crops,
                crop_size=crop_size,
                norm_bounds=[(0.1, 100), (0.1, 95)]
            )
            # ndim = frames[0][0].ndim

            # plt.imshow(frames[0][0])
            # plt.axis('off')
            # plt.title(f"First frame from {os.path.basename(video_path)}")
            # plt.show()

        else:
            msg = f"Unsupported video format '{os.path.splitext(video_path)[1]}'. Only AVI, MP4, MOV, MKV and TIFF/TIF files are supported."
            logger.error(msg)
            raise ValueError(msg)

    except Exception as e:
        logger.error(f"An error occurred while extracting frames from {video_path}: {e}", exc_info=True)
        raise e

    logger.debug(f"Finished extracting {len(frames)} frames with FPS {fps}.")
    return frames, fps


def process_avi_video(fps, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file with OpenCV: {video_path}")
        raise IOError(f"Could not open video file: {video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.debug(f"Opened video with OpenCV. FPS: {fps}.")
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    cap.release()
    logger.debug(f"Successfully extracted {frame_count} frames from OpenCV video file.")
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
        logger.debug(f"Successfully saved JSON ground truth to: {json_output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON ground truth to {json_output_path}: {e}", exc_info=True)
        # Re-raise the exception to signal failure to the caller
        raise