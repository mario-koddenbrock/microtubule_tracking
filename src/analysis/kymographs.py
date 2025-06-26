# Add this import to the top of your script
import collections
import os
from typing import List

import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from tqdm import tqdm


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
    print("\n--- Generating Kymographs ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Pre-process frames to grayscale for intensity sampling ---
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]

    # --- 2. Collect all unique track IDs and their data across all frames ---
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
        print("No tracks found to generate kymographs.")
        return

    # --- 3. Generate a kymograph for each track ---
    for track_id, data in tqdm(tracks_data.items(), desc="Generating Kymographs"):

        # --- 3a. Find the maximum length of this track over its lifetime ---
        max_length = 0
        for entry in data:
            instance_mask = np.zeros(frames[0].shape[:2], dtype=bool)
            rows, cols = entry["coords"][:, 0], entry["coords"][:, 1]
            instance_mask[rows, cols] = True
            skeleton = skeletonize(instance_mask)
            length = int(np.sum(skeleton))
            if length > max_length:
                max_length = length

        if max_length == 0:
            continue  # Skip tracks that have no length (e.g., single pixel noise)

        # --- 3b. Create the kymograph canvas ---
        # Height = number of frames the track appears in
        # Width = the maximum length it ever reached
        kymograph = np.zeros((len(data), max_length), dtype=np.uint8)

        # --- 3c. Fill the kymograph row by row ---
        for row_idx, entry in enumerate(data):
            frame_idx = entry["frame_idx"]
            gray_frame = gray_frames[frame_idx]

            # Recreate the instance mask for the current frame
            instance_mask = np.zeros(frames[0].shape[:2], dtype=bool)
            rows, cols = entry["coords"][:, 0], entry["coords"][:, 1]
            instance_mask[rows, cols] = True

            skeleton = skeletonize(instance_mask).astype(np.uint8)

            # --- 3d. Find an ordered path along the skeleton ---
            # This "unrolls" the curved microtubule into a straight line
            if np.sum(skeleton) > 0:
                # Find an endpoint to start the traversal
                kernel = np.ones((3, 3), dtype=np.uint8)
                neighbor_map = cv2.filter2D(skeleton, -1, kernel)
                endpoints = np.argwhere((skeleton > 0) & (neighbor_map == 2))

                if len(endpoints) > 0:
                    start_node = tuple(endpoints[0])
                else:  # Fallback for loops or single points
                    start_node = tuple(np.argwhere(skeleton)[0])

                # BFS-like traversal to get ordered pixel coordinates along the skeleton
                queue = collections.deque([start_node])
                visited = {start_node}
                ordered_coords = []
                while queue:
                    r, c = queue.popleft()
                    ordered_coords.append((r, c))
                    # Check 8-connectivity neighbors
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

        # --- 3f. Save the final kymograph image ---
        output_path = os.path.join(output_dir, f"kymograph_track_{track_id}.png")
        cv2.imwrite(output_path, kymograph)

    print(f"Saved {len(tracks_data)} kymographs to: {output_dir}")