import collections
import logging
import os
from typing import List

import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from tqdm import tqdm

logger = logging.getLogger(f"microtuble_tracking.{__name__}")


def generate_kymographs(
        frames: List[np.ndarray],
        tracked_masks: List[np.ndarray],
        output_dir: str
):
    """
    Generates and saves a kymograph for each unique tracked object.

    A kymograph is a 2D plot where the Y-axis represents time (frames) and
    the X-axis represents the spatial dimension along the length of the object.
    The pixel intensity at (x, y) corresponds to the brightness of the object
    at position `x` along its length at time `y`.

    Args:
        frames (List[np.ndarray]): The original video frames (in color or grayscale).
        tracked_masks (List[np.ndarray]): The final tracked segmentation masks.
        output_dir (str): The directory where kymograph images will be saved.
    """
    logger.info("--- Starting kymograph generation process ---")
    logger.debug(f"Input frames: {len(frames)}, Input masks: {len(tracked_masks)}, Output dir: {output_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}. Error: {e}", exc_info=True)
        return  # Cannot proceed without output directory

    # --- 1. Pre-process frames to grayscale for intensity sampling ---
    logger.info("Converting input frames to grayscale for intensity sampling.")
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    logger.debug(f"Successfully converted {len(gray_frames)} frames to grayscale.")

    # --- 2. Collect all unique track IDs and their data across all frames ---
    logger.info("Aggregating track data from all frames...")
    tracks_data = collections.defaultdict(list)
    for frame_idx, mask in enumerate(tracked_masks):
        properties = regionprops(mask)
        for prop in properties:
            track_id = prop.label
            # Store the frame index and the object's mask coordinates
            tracks_data[track_id].append({
                "frame_idx": frame_idx,
                "coords": prop.coords
            })

    if not tracks_data:
        logger.info("No tracks found to generate kymographs. Process finished.")
        return

    logger.info(f"Found {len(tracks_data)} unique tracks to process.")

    # --- 3. Generate a kymograph for each track ---
    for track_id, data in tqdm(tracks_data.items(), desc="Generating Kymographs"):
        logger.debug(f"Processing track ID: {track_id} which appears in {len(data)} frames.")

        # --- 3a. Find the maximum length of this track over its lifetime ---
        max_length = 0
        for entry in data:
            # Create a temporary mask just for this one object instance
            instance_mask = np.zeros(frames[0].shape[:2], dtype=bool)
            rows, cols = entry["coords"][:, 0], entry["coords"][:, 1]
            instance_mask[rows, cols] = True
            skeleton = skeletonize(instance_mask)
            length = int(np.sum(skeleton))
            if length > max_length:
                max_length = length

        logger.debug(f"Track {track_id}: Max skeleton length found: {max_length} pixels.")

        if max_length == 0:
            logger.warning(f"Track {track_id} has a maximum length of 0. Skipping kymograph generation for this track.")
            continue

        # --- 3b. Create the kymograph canvas ---
        kymograph = np.zeros((len(data), max_length), dtype=np.uint8)
        logger.debug(f"Track {track_id}: Created kymograph canvas of size {kymograph.shape}.")

        # --- 3c. Fill the kymograph row by row ---
        for row_idx, entry in enumerate(data):
            frame_idx = entry["frame_idx"]
            gray_frame = gray_frames[frame_idx]

            instance_mask = np.zeros(frames[0].shape[:2], dtype=bool)
            rows, cols = entry["coords"][:, 0], entry["coords"][:, 1]
            instance_mask[rows, cols] = True

            skeleton = skeletonize(instance_mask).astype(np.uint8)

            # --- 3d. Find an ordered path along the skeleton ---
            if np.sum(skeleton) > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                neighbor_map = cv2.filter2D(skeleton, -1, kernel)
                endpoints = np.argwhere((skeleton > 0) & (neighbor_map == 2))

                if len(endpoints) > 0:
                    start_node = tuple(endpoints[0])
                    logger.debug(
                        f"Track {track_id}, Frame {frame_idx}: Found endpoint at {start_node} to start traversal.")
                else:  # Fallback for loops or single points
                    fallback_node = tuple(np.argwhere(skeleton)[0])
                    logger.warning(
                        f"Track {track_id}, Frame {frame_idx}: No clear skeleton endpoint found. May be a loop or dot. Using fallback start point: {fallback_node}.")
                    start_node = fallback_node

                queue = collections.deque([start_node])
                visited = {start_node}
                ordered_coords = []
                while queue:
                    r, c = queue.popleft()
                    ordered_coords.append((r, c))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1]
                                    and skeleton[nr, nc] > 0 and (nr, nc) not in visited):
                                visited.add((nr, nc))
                                queue.append((nr, nc))

                # --- 3e. Sample intensities and place them on the kymograph ---
                if ordered_coords:
                    line_intensities = gray_frame[tuple(zip(*ordered_coords))]
                    kymograph[row_idx, :len(line_intensities)] = line_intensities
                    logger.debug(
                        f"Track {track_id}, Frame {frame_idx}: Sampled {len(line_intensities)} intensity values and placed on kymograph row {row_idx}.")

        # --- 3f. Save the final kymograph image ---
        output_path = os.path.join(output_dir, f"kymograph_track_{track_id}.png")
        try:
            cv2.imwrite(output_path, kymograph)
            logger.debug(f"Saved kymograph for track {track_id} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save kymograph for track {track_id} to {output_path}. Error: {e}", exc_info=True)

    logger.info(f"Successfully generated and saved {len(tracks_data)} kymographs to: {output_dir}")