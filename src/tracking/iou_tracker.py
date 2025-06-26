import numpy as np
from typing import List, Dict
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from tracking.base_tracker import BaseTracker


class IoUTracker(BaseTracker):
    """
    A sophisticated multi-pass tracker designed to handle splits and merges.

    It operates in three stages:
    1. A forward pass identifies tracks and splitting events.
    2. A backward pass identifies merging events (by viewing them as splits).
    3. A reconciliation pass unifies the track IDs from both passes.
    """
    def __init__(self, iou_threshold: float = 0.2, max_lookback: int = 5):
        """
        Initializes the tracker.

        Args:
            iou_threshold (float): Minimum IoU to consider two objects a match.
            max_lookback (int): The maximum number of frames to look back to
                                find a lost object.
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.max_lookback = max_lookback


    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculates IoU for two single-object boolean masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        if intersection == 0: return 0.0
        # Use intersection / area_of_current_object for split detection
        area2 = np.sum(mask2)
        return intersection / float(area2) if area2 > 0 else 0.0

    def _perform_pass(self, raw_masks: List[np.ndarray], desc: str) -> List[np.ndarray]:
        """Performs a single tracking pass (either forward or backward)."""
        self.reset()
        tracked_masks = []
        prev_tracked_mask = np.zeros_like(raw_masks[0])

        for i in tqdm(range(len(raw_masks)), desc=desc):
            current_raw_mask = raw_masks[i]
            new_tracked_mask = np.zeros_like(current_raw_mask)

            # --- 1. Find a candidate ID for each new object using look-back ---
            current_ids = np.unique(current_raw_mask)[1:]
            for raw_id in current_ids:
                instance_mask = (current_raw_mask == raw_id)
                best_match_id = -1
                best_iou = -1

                # Look back through previous frames
                for lookback_idx in range(self.max_lookback):
                    prev_frame_idx = i - 1 - lookback_idx
                    if prev_frame_idx < 0: break

                    comparison_mask = tracked_masks[prev_frame_idx]
                    prev_ids_in_frame = np.unique(comparison_mask)[1:]

                    for p_id in prev_ids_in_frame:
                        iou = self._calculate_iou(comparison_mask == p_id, instance_mask)
                        if iou > self.iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_match_id = p_id

                if best_match_id != -1:
                    new_tracked_mask[instance_mask] = best_match_id
                else:
                    new_tracked_mask[instance_mask] = self.next_track_id
                    self.next_track_id += 1

            # --- 2. Handle splits: If multiple new objects match one old one ---
            prev_ids = np.unique(prev_tracked_mask)[1:]
            for p_id in prev_ids:
                parent_mask = (prev_tracked_mask == p_id)

                overlapping_new_ids = set(np.unique(new_tracked_mask[parent_mask])) - {0}

                if len(overlapping_new_ids) > 1:
                    # This is a split event. Unify the children's IDs.
                    canonical_id = min(overlapping_new_ids)
                    for child_id in overlapping_new_ids:
                        if child_id != canonical_id:
                            new_tracked_mask[new_tracked_mask == child_id] = canonical_id

            tracked_masks.append(new_tracked_mask)
            prev_tracked_mask = new_tracked_mask

        return tracked_masks

    def _reconcile_passes(self, forward_masks: List[np.ndarray], backward_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Unifies tracks by creating a merge map from the backward pass."""

        # --- 1. Build a map of which forward tracks should be merged ---
        merge_map = {}
        for i in range(len(forward_masks)):
            f_mask = forward_masks[i]
            b_mask = backward_masks[i]

            b_ids = np.unique(b_mask)[1:]
            for b_id in b_ids:
                # Find all forward tracks that this backward track overlaps with
                overlapping_f_ids = set(np.unique(f_mask[b_mask == b_id])) - {0}

                if len(overlapping_f_ids) > 1:
                    # The backward pass says these should be one track.
                    # We map all of them to the one with the smallest ID.
                    canonical_id = min(overlapping_f_ids)
                    for f_id in overlapping_f_ids:
                        if f_id != canonical_id:
                            merge_map[f_id] = canonical_id

        # --- 2. Resolve transitive merges (e.g., 5->3 and 3->2 becomes 5->2) ---
        for old_id, new_id in merge_map.items():
            while new_id in merge_map:
                new_id = merge_map[new_id]
            merge_map[old_id] = new_id

        # --- 3. Apply the merge map to create the final tracked masks ---
        final_masks = []
        for f_mask in forward_masks:
            remapped_mask = f_mask.copy()
            for old_id, new_id in merge_map.items():
                remapped_mask[remapped_mask == old_id] = new_id
            final_masks.append(remapped_mask)

        return final_masks

    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray] = None) -> List[np.ndarray]:
        """Orchestrates the full forward, backward, and reconciliation pipeline."""

        # --- STAGE 1: FORWARD PASS ---
        forward_tracked = self._perform_pass(raw_masks, desc="Forward Pass (Splits)")

        # --- STAGE 2: BACKWARD PASS ---
        # The backward pass finds merges by treating them as splits in reverse time.
        backward_tracked_rev = self._perform_pass(raw_masks[::-1], desc="Backward Pass (Merges)")
        backward_tracked = backward_tracked_rev[::-1] # Reverse back to normal order

        # --- STAGE 3: RECONCILIATION ---
        print("Reconciling forward and backward passes...")
        final_tracked_masks = self._reconcile_passes(forward_tracked, backward_tracked)

        return final_tracked_masks
