import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass
from tqdm import tqdm

from tracking.base_tracker import BaseTracker


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
        super().__init__()
        self.max_distance = max_distance
        print(f"CentroidTracker initialized with max_distance: {self.max_distance}")

    def _get_centroids(self, mask: np.ndarray):
        """Calculates the center of mass for each object in a mask."""
        ids = np.unique(mask)[1:]
        # center_of_mass is much faster than manual calculation
        centroids = center_of_mass(mask, mask, ids)
        return dict(zip(ids, centroids))

    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Processes raw masks to produce tracked masks using the centroid distance algorithm.
        """
        self.reset()
        if not raw_masks: return []

        # Initialize with the first frame (same as IoUTracker)
        tracked_masks = []
        first_mask = raw_masks[0]
        tracked_first_mask = np.zeros_like(first_mask)
        raw_ids = np.unique(first_mask)[1:]
        for raw_id in raw_ids:
            tracked_first_mask[first_mask == raw_id] = self.next_track_id
            self.next_track_id += 1
        tracked_masks.append(tracked_first_mask)

        for i in tqdm(range(1, len(raw_masks)), desc="Tracking with Centroids"):
            prev_tracked_mask = tracked_masks[i - 1]
            current_raw_mask = raw_masks[i]
            new_tracked_mask = np.zeros_like(current_raw_mask)

            prev_centroids = self._get_centroids(prev_tracked_mask)
            current_centroids = self._get_centroids(current_raw_mask)

            if not prev_centroids or not current_centroids:
                tracked_masks.append(new_tracked_mask)  # Handle empty frames
                continue

            prev_ids = list(prev_centroids.keys())
            current_ids = list(current_centroids.keys())

            # Calculate the distance matrix between all centroid pairs
            dist_matrix = cdist(list(prev_centroids.values()), list(current_centroids.values()))

            # Use Hungarian algorithm to find optimal matches
            prev_indices, current_indices = linear_sum_assignment(dist_matrix)

            matched_current_ids = set()
            for prev_idx, current_idx in zip(prev_indices, current_indices):
                distance = dist_matrix[prev_idx, current_idx]
                if distance <= self.max_distance:
                    prev_track_id = prev_ids[prev_idx]
                    current_raw_id = current_ids[current_idx]
                    new_tracked_mask[current_raw_mask == current_raw_id] = prev_track_id
                    matched_current_ids.add(current_raw_id)

            unmatched_current_ids = set(current_ids) - matched_current_ids
            for new_raw_id in unmatched_current_ids:
                new_tracked_mask[current_raw_mask == new_raw_id] = self.next_track_id
                self.next_track_id += 1

            tracked_masks.append(new_tracked_mask)
        return tracked_masks