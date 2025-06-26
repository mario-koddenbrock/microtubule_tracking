import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from tracking.base_tracker import BaseTracker


class IoUTracker(BaseTracker):
    """
    Tracks objects across frames using the Intersection over Union (IoU) metric.

    This tracker matches objects by finding the pair with the maximum spatial
    overlap between consecutive frames.
    """
    def __init__(self, iou_threshold: float = 0.3):
        """
        Initializes the IoU tracker.

        Args:
            iou_threshold (float): The minimum IoU for two objects to be
                considered a match. Defaults to 0.3.
        """
        super().__init__() # IMPORTANT: Initialize the parent class
        if not 0 < iou_threshold < 1:
            raise ValueError("iou_threshold must be between 0 and 1.")
        self.iou_threshold = iou_threshold
        print(f"IoUTracker initialized with threshold: {self.iou_threshold}")

    def _calculate_iou_matrix(self, prev_mask: np.ndarray, current_mask: np.ndarray):
        """Calculates a matrix of IoU values between all object pairs."""
        # This implementation is unchanged
        prev_ids = np.unique(prev_mask)[1:]
        current_ids = np.unique(current_mask)[1:]
        iou_matrix = np.zeros((len(prev_ids), len(current_ids)), dtype=np.float32)
        for i, prev_id in enumerate(prev_ids):
            for j, current_id in enumerate(current_ids):
                mask1 = (prev_mask == prev_id)
                mask2 = (current_mask == current_id)
                intersection = np.logical_and(mask1, mask2).sum()
                if intersection == 0: continue
                union = np.logical_or(mask1, mask2).sum()
                iou_matrix[i, j] = intersection / union
        return iou_matrix, prev_ids, current_ids

    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Processes raw masks to produce tracked masks using the IoU algorithm.
        """
        self.reset() # Start fresh for each call
        if not raw_masks: return []

        tracked_masks = []
        # --- The rest of this method's implementation is exactly as you wrote it ---
        # (Copied here for completeness)
        first_mask = raw_masks[0]
        tracked_first_mask = np.zeros_like(first_mask)
        raw_ids = np.unique(first_mask)[1:]
        for raw_id in raw_ids:
            tracked_first_mask[first_mask == raw_id] = self.next_track_id
            self.next_track_id += 1
        tracked_masks.append(tracked_first_mask)

        for i in tqdm(range(1, len(raw_masks)), desc="Tracking with IoU"):
            prev_tracked_mask = tracked_masks[i - 1]
            current_raw_mask = raw_masks[i]
            new_tracked_mask = np.zeros_like(current_raw_mask)
            if np.max(current_raw_mask) == 0 or np.max(prev_tracked_mask) == 0:
                tracked_masks.append(new_tracked_mask)
                continue
            iou_matrix, prev_ids, current_ids = self._calculate_iou_matrix(
                prev_tracked_mask, current_raw_mask)
            cost_matrix = 1 - iou_matrix
            prev_indices, current_indices = linear_sum_assignment(cost_matrix)
            matched_current_ids = set()
            for prev_idx, current_idx in zip(prev_indices, current_indices):
                if iou_matrix[prev_idx, current_idx] >= self.iou_threshold:
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