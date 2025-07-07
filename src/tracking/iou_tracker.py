import logging
from typing import List, Dict, Set, Optional  

import numpy as np
from tqdm import tqdm

from .base import BaseTracker  


logger = logging.getLogger(f"mt.{__name__}")


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
        logger.info(f"Initializing IoUTracker with iou_threshold={iou_threshold}, max_lookback={max_lookback}.")
        super().__init__()  # Call the base class constructor
        if not (0.0 <= iou_threshold <= 1.0):
            msg = f"iou_threshold must be between 0.0 and 1.0, but got {iou_threshold}."
            logger.error(msg)
            raise ValueError(msg)
        if not (max_lookback >= 1):
            msg = f"max_lookback must be at least 1, but got {max_lookback}."
            logger.error(msg)
            raise ValueError(msg)

        self.iou_threshold = iou_threshold
        self.max_lookback = max_lookback
        logger.debug("IoUTracker initialization complete.")

    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculates IoU for two single-object boolean masks."""
        # logger.debug(f"Calculating IoU for masks of shape {mask1.shape} and {mask2.shape}.")
        intersection = np.logical_and(mask1, mask2).sum()
        if intersection == 0:
            return 0.0

        area2 = np.sum(mask2)  # Area of the current object (mask2)
        if area2 == 0:
            logger.warning("Area of current object (mask2) is zero. IoU set to 0.0 to prevent division by zero.")
            return 0.0

        iou = intersection / float(area2)  # IoU is intersection over area of *current* object for split detection
        # logger.debug(f"  Intersection: {intersection}, Area2: {area2}, IoU: {iou:.4f}.")
        return iou

    def _perform_pass(self, raw_masks: List[np.ndarray], desc: str) -> List[np.ndarray]:
        """
        Performs a single tracking pass (either forward or backward).

        Args:
            raw_masks (List[np.ndarray]): List of masks for the pass.
            desc (str): Description for the tqdm progress bar.

        Returns:
            List[np.ndarray]: List of masks with assigned track IDs for this pass.
        """
        logger.info(f"Starting {desc} (IoU pass).")
        self.reset()  # Reset tracker state for this pass

        tracked_masks: List[np.ndarray] = []

        # Initialize the first frame for this pass
        if not raw_masks:
            logger.warning(f"{desc}: No raw masks provided. Returning empty list.")
            return []

        # The first mask always gets new IDs assigned
        first_mask = raw_masks[0]
        tracked_first_mask = np.zeros_like(first_mask, dtype=np.uint16)
        raw_ids_first_frame = np.unique(first_mask)
        raw_ids_first_frame = raw_ids_first_frame[raw_ids_first_frame != 0]  # Exclude background

        logger.debug(f"  {desc}: Frame 0: Initializing {len(raw_ids_first_frame)} tracks.")
        for raw_id in raw_ids_first_frame:
            tracked_first_mask[first_mask == raw_id] = self.next_track_id
            self.next_track_id += 1
        tracked_masks.append(tracked_first_mask)
        logger.debug(f"  {desc}: Frame 0 processed. Next available track ID: {self.next_track_id}.")

        # Iterate from the second frame onwards
        # prev_tracked_mask is the mask from the *previous frame of the current pass*
        prev_tracked_mask = tracked_first_mask

        for i in tqdm(range(1, len(raw_masks)), desc=desc):
            logger.debug(f"  {desc}: Processing frame {i}...")
            current_raw_mask = raw_masks[i]
            new_tracked_mask = np.zeros_like(current_raw_mask, dtype=np.uint16)

            # --- 1. Find a candidate ID for each new object using look-back ---
            current_ids: np.ndarray = np.unique(current_raw_mask)
            current_ids = current_ids[current_ids != 0]  # Exclude background

            if len(current_ids) == 0:
                logger.debug(f"  {desc}: Frame {i} has no objects. Appending empty mask.")
                tracked_masks.append(new_tracked_mask)
                prev_tracked_mask = new_tracked_mask  # Update prev_tracked_mask for next iteration
                continue

            for raw_id in current_ids:
                instance_mask = (current_raw_mask == raw_id)
                best_match_id = -1
                best_iou = -1.0  # Initialize to a value that any valid IoU will beat

                # Look back through previous frames (within max_lookback window)
                for lookback_offset in range(self.max_lookback):
                    prev_frame_idx = i - 1 - lookback_offset
                    if prev_frame_idx < 0:
                        logger.debug(
                            f"    {desc}: Frame {i}, Raw ID {raw_id}: Lookback index {prev_frame_idx} is out of bounds. Stopping lookback.")
                        break

                    comparison_mask = tracked_masks[prev_frame_idx]
                    prev_ids_in_frame: np.ndarray = np.unique(comparison_mask)
                    prev_ids_in_frame = prev_ids_in_frame[prev_ids_in_frame != 0]  # Exclude background

                    if len(prev_ids_in_frame) == 0:
                        logger.debug(
                            f"    {desc}: Frame {i}, Raw ID {raw_id}: No objects in lookback frame {prev_frame_idx}. Skipping.")
                        continue

                    for p_id in prev_ids_in_frame:
                        iou = self._calculate_iou(comparison_mask == p_id, instance_mask)
                        logger.debug(
                            f"    {desc}: Frame {i}, Raw ID {raw_id}: Comparing with prev_id {p_id} from frame {prev_frame_idx}. IoU: {iou:.4f}.")
                        if iou > self.iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_match_id = p_id
                            logger.debug(
                                f"      {desc}: Frame {i}, Raw ID {raw_id}: New best match found: prev_id {p_id} with IoU {iou:.4f}.")

                if best_match_id != -1:
                    new_tracked_mask[instance_mask] = best_match_id
                    logger.debug(
                        f"    {desc}: Frame {i}, Raw ID {raw_id}: Matched to existing track ID {best_match_id} (best IoU: {best_iou:.4f}).")
                else:
                    new_tracked_mask[instance_mask] = self.next_track_id
                    logger.debug(
                        f"    {desc}: Frame {i}, Raw ID {raw_id}: No sufficient match found. Assigning new track ID {self.next_track_id}.")
                    self.next_track_id += 1

            # --- 2. Handle splits: If multiple new objects match one old one ---
            # This logic should operate on objects detected in `new_tracked_mask` that originate from `prev_tracked_mask`

            # Get IDs from the PREVIOUS *tracked* mask that were matched in *current* frame
            # The logic below iterates over prev_tracked_mask IDs and checks their progeny in new_tracked_mask.
            prev_ids_in_prev_frame: np.ndarray = np.unique(prev_tracked_mask)
            prev_ids_in_prev_frame = prev_ids_in_prev_frame[prev_ids_in_prev_frame != 0]  # Exclude background

            for p_id in prev_ids_in_prev_frame:
                parent_mask = (prev_tracked_mask == p_id)
                # Find all unique IDs in the new_tracked_mask that overlap with this parent
                overlapping_new_ids: Set[int] = set(np.unique(new_tracked_mask[parent_mask])) - {
                    0}  # Exclude background

                logger.debug(
                    f"  {desc}: Frame {i}, Parent ID {p_id}: Overlapping new IDs in current frame: {list(overlapping_new_ids)}.")

                if len(overlapping_new_ids) > 1:
                    # This is a split event. Unify the children's IDs.
                    canonical_id = min(overlapping_new_ids)  # Choose the smallest ID as canonical
                    logger.info(
                        f"  {desc}: Frame {i}: Detected split from ID {p_id}. Unifying {list(overlapping_new_ids)} to canonical ID {canonical_id}.")
                    for child_id in overlapping_new_ids:
                        if child_id != canonical_id:
                            # Remap all split children to the same canonical ID in the current mask
                            new_tracked_mask[new_tracked_mask == child_id] = canonical_id
                            logger.debug(
                                f"    {desc}: Frame {i}: Remapped split child ID {child_id} to {canonical_id}.")

            tracked_masks.append(new_tracked_mask)
            prev_tracked_mask = new_tracked_mask  # Update for the next iteration
            logger.info(
                f"  {desc}: Frame {i} processed. Total objects in frame: {len(np.unique(new_tracked_mask)[1:])}. Next track ID: {self.next_track_id}.")

        logger.info(f"{desc} (IoU pass) complete. Generated {len(tracked_masks)} masks.")
        return tracked_masks

    def _reconcile_passes(self, forward_masks: List[np.ndarray], backward_masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Unifies tracks by creating a merge map from the backward pass.

        Args:
            forward_masks (List[np.ndarray]): Masks from the forward pass.
            backward_masks (List[np.ndarray]): Masks from the backward pass (re-ordered to be time-aligned).

        Returns:
            List[np.ndarray]: List of final, reconciled tracked masks.
        """
        logger.info("Starting reconciliation of forward and backward passes...")

        if len(forward_masks) != len(backward_masks):
            logger.error(
                f"Mismatch in number of frames for reconciliation: Forward ({len(forward_masks)}) vs Backward ({len(backward_masks)}). Returning forward masks as fallback.")
            return forward_masks  # Cannot reconcile, return forward pass as best effort

        # --- 1. Build a map of which forward tracks should be merged ---
        # This map will store: {forward_id_to_merge: canonical_forward_id}
        merge_map: Dict[int, int] = {}

        for i in tqdm(range(len(forward_masks)), desc="Building merge map"):
            f_mask = forward_masks[i]
            b_mask = backward_masks[i]

            b_ids: np.ndarray = np.unique(b_mask)
            b_ids = b_ids[b_ids != 0]  # Exclude background

            for b_id in b_ids:
                # Find all forward tracks that this backward track overlaps with
                # An object in the backward pass (which represents a merged track when viewed forward)
                # will overlap with multiple forward track IDs.
                overlapping_f_ids: Set[int] = set(np.unique(f_mask[b_mask == b_id])) - {0}  # Exclude background

                if len(overlapping_f_ids) > 1:
                    # This means multiple forward tracks are identified as one backward track.
                    # This is a merge event from the forward perspective.
                    canonical_id = min(overlapping_f_ids)  # Choose the smallest ID as canonical
                    logger.debug(
                        f"  Frame {i}, B-ID {b_id}: Detected merge of F-IDs {list(overlapping_f_ids)}. Canonical F-ID: {canonical_id}.")
                    for f_id in overlapping_f_ids:
                        if f_id != canonical_id:
                            # Map the larger forward IDs to the canonical smaller ID
                            merge_map[f_id] = canonical_id
                            logger.debug(f"    Mapped F-ID {f_id} to canonical F-ID {canonical_id}.")

        logger.debug(f"Initial merge map built: {merge_map}")

        # --- 2. Resolve transitive merges (e.g., 5->3 and 3->2 becomes 5->2) ---
        resolved_merges_count = 0
        for old_id in list(merge_map.keys()):  # Iterate over a copy of keys as map might change
            current_id = old_id
            path = []  # To detect cycles
            while current_id in merge_map:
                if current_id in path:
                    logger.error(
                        f"Circular merge dependency detected for ID {old_id} via path {path + [current_id]}. This indicates a logic error or complex topology. Breaking cycle for this ID.")
                    break  # Break to avoid infinite loop
                path.append(current_id)
                current_id = merge_map[current_id]
                resolved_merges_count += 1
            if merge_map.get(old_id) != current_id:  # Only update if it actually changed
                merge_map[old_id] = current_id
                logger.debug(f"Resolved transitive merge for {old_id} to {current_id}.")
        logger.debug(f"Transitive merges resolved. Total resolutions: {resolved_merges_count}.")
        logger.debug(f"Final resolved merge map: {merge_map}")

        # --- 3. Apply the merge map to create the final tracked masks ---
        final_masks: List[np.ndarray] = []
        for frame_idx, f_mask in enumerate(tqdm(forward_masks, desc="Applying merge map")):
            remapped_mask = f_mask.copy()
            ids_in_frame = np.unique(remapped_mask)
            ids_in_frame = ids_in_frame[ids_in_frame != 0]  # Exclude background

            remapped_count = 0
            for old_id in ids_in_frame:
                if old_id in merge_map:
                    new_id = merge_map[old_id]
                    remapped_mask[remapped_mask == old_id] = new_id
                    remapped_count += 1
                    logger.debug(f"  Frame {frame_idx}: Remapped old ID {old_id} to new ID {new_id}.")
            final_masks.append(remapped_mask)
            logger.debug(f"  Frame {frame_idx}: Remapped {remapped_count} IDs.")

        logger.info(f"Reconciliation complete. Generated {len(final_masks)} final tracked masks.")
        return final_masks

    def track_frames(self, raw_masks: List[np.ndarray], frames: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Orchestrates the full forward, backward, and reconciliation pipeline.

        Args:
            raw_masks (List[np.ndarray]): A list of independently segmented masks.
            frames (Optional[List[np.ndarray]]): Optional list of original pixel frames.
                                                Not directly used by IoUTracker but passed for interface consistency.

        Returns:
            List[np.ndarray]: A list of re-labeled masks with consistent tracking IDs.
        """
        # Call the base class's track_frames for initial validation and logging
        try:
            super().track_frames(raw_masks, frames if frames is not None else [])
        except Exception as e:
            logger.error(f"BaseTracker validation failed: {e}", exc_info=True)
            raise  # Re-raise, as validation failure is critical

        if not raw_masks:
            logger.warning("No raw masks provided. Returning empty list of tracked masks.")
            return []

        # --- STAGE 1: FORWARD PASS ---
        forward_tracked: List[np.ndarray]
        try:
            forward_tracked = self._perform_pass(raw_masks, desc="Forward Pass (Splits)")
            logger.info(f"Forward pass completed. Total frames: {len(forward_tracked)}.")
        except Exception as e:
            logger.critical(f"Critical error during forward pass: {e}", exc_info=True)
            return [np.zeros_like(mask, dtype=np.uint16) for mask in
                    raw_masks]  # Return empty masks on critical failure

        # --- STAGE 2: BACKWARD PASS ---
        backward_tracked_rev: List[np.ndarray]
        try:
            # The backward pass finds merges by treating them as splits in reverse time.
            backward_tracked_rev = self._perform_pass(raw_masks[::-1], desc="Backward Pass (Merges)")
            backward_tracked = backward_tracked_rev[::-1]  # Reverse back to normal order
            logger.info(f"Backward pass completed. Total frames: {len(backward_tracked)}.")
        except Exception as e:
            logger.critical(f"Critical error during backward pass: {e}", exc_info=True)
            # If backward pass fails, reconciliation is impossible, so return forward pass as best effort.
            return forward_tracked  # Or re-raise, depending on severity requirement

        # --- STAGE 3: RECONCILIATION ---
        final_tracked_masks: List[np.ndarray]
        try:
            final_tracked_masks = self._reconcile_passes(forward_tracked, backward_tracked)
            logger.info(f"Reconciliation completed. Total frames: {len(final_tracked_masks)}.")
        except Exception as e:
            logger.critical(f"Critical error during reconciliation pass: {e}", exc_info=True)
            # If reconciliation fails, return the forward pass as best effort.
            return forward_tracked  # Or re-raise

        logger.info(f"IoUTracker finished. Returning {len(final_tracked_masks)} tracked masks.")
        return final_tracked_masks