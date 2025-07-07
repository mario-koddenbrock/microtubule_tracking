import logging
from typing import List, Dict, Tuple

import numpy as np

from scipy.optimize import linear_sum_assignment  
from scipy.spatial.distance import cdist  
from scipy.ndimage import center_of_mass  
from tqdm import tqdm

from .base import BaseTracker


logger = logging.getLogger(f"mt.{__name__}")


class CentroidTracker(BaseTracker):
    """
    Tracks objects across frames by matching the closest centroids.
    """

    def __init__(self, max_distance: int = 50):
        """
        Initializes the Centroid tracker.

        Args:
            max_distance (int): The maximum pixel distance for two centroids
                to be considered a match. Defaults to 50.
        """
        logger.info(f"Initializing CentroidTracker with max_distance: {max_distance}.")
        super().__init__()  # Call the base class constructor
        self.max_distance = max_distance
        logger.debug("CentroidTracker initialization complete.")

    def _get_centroids(self, mask: np.ndarray) -> Dict[int, Tuple[float, ...]]:  # Return type more specific
        """
        Calculates the center of mass for each object in a mask.

        Args:
            mask (np.ndarray): A 2D integer label mask.

        Returns:
            Dict[int, Tuple[float, ...]]: A dictionary mapping object label to its centroid (y, x).
        """
        logger.debug(f"Calculating centroids for mask of shape {mask.shape}.")
        ids = np.unique(mask)
        ids = ids[ids != 0]  # Exclude background

        if len(ids) == 0:
            logger.debug("Mask is empty (no non-zero labels). Returning empty centroids dictionary.")
            return {}

        try:
            # center_of_mass is much faster than manual calculation
            centroids_list: List[Tuple[float, ...]] = center_of_mass(mask, mask, ids)
            centroids_dict = dict(zip(ids, centroids_list))
            logger.debug(f"Extracted {len(centroids_dict)} centroids from mask.")
            return centroids_dict
        except Exception as e:
            logger.error(f"Error calculating centroids for mask of shape {mask.shape}: {e}", exc_info=True)
            return {}  # Return empty on error

    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Processes raw masks to produce tracked masks using the centroid distance algorithm.

        Args:
            raw_masks (List[np.ndarray]): A list of independently segmented masks.
            frames (List[np.ndarray]): A list of original pixel frames (not directly used by centroid tracker, but passed for interface consistency).

        Returns:
            List[np.ndarray]: A list of re-labeled masks with consistent tracking IDs.
        """
        # Call the base class's track_frames for initial validation and logging
        try:
            super().track_frames(raw_masks, frames)
        except Exception as e:
            logger.error(f"BaseTracker validation failed: {e}", exc_info=True)
            raise  # Re-raise, as validation failure is critical

        self.reset()  # Reset tracker state
        if not raw_masks:
            logger.warning("No raw masks provided. Returning empty list of tracked masks.")
            return []

        tracked_masks: List[np.ndarray] = []

        # --- Initialize with the first frame ---
        first_mask = raw_masks[0]
        tracked_first_mask = np.zeros_like(first_mask, dtype=np.uint16)  # Ensure uint16 for labels
        raw_ids_first_frame = np.unique(first_mask)
        raw_ids_first_frame = raw_ids_first_frame[raw_ids_first_frame != 0]  # Exclude background

        logger.info(f"Frame 0: Initializing {len(raw_ids_first_frame)} tracks.")
        for raw_id in raw_ids_first_frame:
            tracked_first_mask[first_mask == raw_id] = self.next_track_id
            logger.debug(f"  Frame 0: Assigned raw ID {raw_id} to new track ID {self.next_track_id}.")
            self.next_track_id += 1
        tracked_masks.append(tracked_first_mask)
        logger.info(f"Frame 0 processed. Next available track ID: {self.next_track_id}.")

        # --- Process subsequent frames ---
        for i in tqdm(range(1, len(raw_masks)), desc="Tracking with Centroids"):
            logger.debug(f"Processing frame {i}...")
            prev_tracked_mask = tracked_masks[i - 1]
            current_raw_mask = raw_masks[i]
            new_tracked_mask = np.zeros_like(current_raw_mask, dtype=np.uint16)

            prev_centroids = self._get_centroids(prev_tracked_mask)
            current_centroids = self._get_centroids(current_raw_mask)

            logger.debug(
                f"  Frame {i}: Found {len(prev_centroids)} centroids in previous frame, {len(current_centroids)} in current raw frame.")

            if not prev_centroids:
                logger.debug(f"  Frame {i}: No objects in previous frame. All current objects are new.")
                # All current objects are new
                for new_raw_id in np.unique(current_raw_mask)[1:]:
                    new_tracked_mask[current_raw_mask == new_raw_id] = self.next_track_id
                    logger.debug(f"    Frame {i}: Raw ID {new_raw_id} assigned to new track ID {self.next_track_id}.")
                    self.next_track_id += 1
                tracked_masks.append(new_tracked_mask)
                continue

            if not current_centroids:
                logger.debug(f"  Frame {i}: No objects in current raw mask. No new tracks or matches.")
                tracked_masks.append(new_tracked_mask)  # Append an empty mask
                continue

            prev_ids = list(prev_centroids.keys())
            current_ids = list(current_centroids.keys())

            prev_coords = np.array(list(prev_centroids.values()))
            current_coords = np.array(list(current_centroids.values()))

            # Calculate the distance matrix between all centroid pairs
            try:
                dist_matrix = cdist(prev_coords, current_coords)
                logger.debug(f"  Frame {i}: Distance matrix shape: {dist_matrix.shape}.")
            except Exception as e:
                logger.error(f"  Frame {i}: Error calculating distance matrix: {e}. Skipping frame.", exc_info=True)
                tracked_masks.append(new_tracked_mask)
                continue

            # Use Hungarian algorithm to find optimal matches
            try:
                prev_indices, current_indices = linear_sum_assignment(dist_matrix)
                logger.debug(f"  Frame {i}: Hungarian algorithm found {len(prev_indices)} potential matches.")
            except Exception as e:
                logger.error(f"  Frame {i}: Error during linear_sum_assignment: {e}. Skipping frame.", exc_info=True)
                tracked_masks.append(new_tracked_mask)
                continue

            matched_current_ids = set()
            matches_count = 0
            for prev_idx_in_matrix, current_idx_in_matrix in zip(prev_indices, current_indices):
                distance = dist_matrix[prev_idx_in_matrix, current_idx_in_matrix]

                if distance <= self.max_distance:
                    prev_track_id = prev_ids[prev_idx_in_matrix]
                    current_raw_id = current_ids[current_idx_in_matrix]

                    new_tracked_mask[current_raw_mask == current_raw_id] = prev_track_id
                    matched_current_ids.add(current_raw_id)
                    matches_count += 1
                    logger.debug(
                        f"    Frame {i}: Matched prev track ID {prev_track_id} to current raw ID {current_raw_id} (distance: {distance:.2f}).")
                else:
                    logger.debug(
                        f"    Frame {i}: Match rejected (distance {distance:.2f} > max_distance {self.max_distance}).")
            logger.debug(f"  Frame {i}: Found {matches_count} matches within max_distance.")

            # Assign new track IDs to unmatched current objects
            unmatched_current_ids_raw = set(current_ids) - matched_current_ids
            for new_raw_id in unmatched_current_ids_raw:
                new_tracked_mask[current_raw_mask == new_raw_id] = self.next_track_id
                logger.debug(f"    Frame {i}: Raw ID {new_raw_id} assigned to new track ID {self.next_track_id}.")
                self.next_track_id += 1
            logger.debug(f"  Frame {i}: Assigned new IDs to {len(unmatched_current_ids_raw)} unmatched objects.")

            tracked_masks.append(new_tracked_mask)
            logger.info(f"Frame {i} processed. Next available track ID: {self.next_track_id}.")

        logger.info(f"Tracking complete. Generated {len(tracked_masks)} tracked masks.")
        return tracked_masks