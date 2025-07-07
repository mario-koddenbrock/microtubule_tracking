import logging
from typing import List, Tuple, Set

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .base import BaseTracker


logger = logging.getLogger(f"mt.{__name__}")


class ProjectionTracker(BaseTracker):
    """
    Tracks objects across frames, potentially using a global projection step
    (e.g., based on Hough lines) and then an IoU-based matching.
    """

    def __init__(self, iou_threshold: float = 0.3):
        """
        Initializes the ProjectionTracker.

        Args:
            iou_threshold (float): The minimum IoU for two objects to be
                considered a match. Defaults to 0.3.
        """
        logger.info(f"Initializing ProjectionTracker with iou_threshold: {iou_threshold}.")
        super().__init__()  # IMPORTANT: Initialize the parent class

        if not (0.0 < iou_threshold < 1.0):
            msg = f"iou_threshold must be strictly between 0 and 1, but got {iou_threshold}."
            logger.error(msg)
            raise ValueError(msg)

        self.iou_threshold = iou_threshold
        logger.debug("ProjectionTracker initialization complete.")

    def _calculate_iou_matrix(self, prev_mask: np.ndarray, current_mask: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates a matrix of IoU values between all object pairs from two masks.

        Args:
            prev_mask (np.ndarray): The mask from the previous frame (H, W).
            current_mask (np.ndarray): The mask from the current frame (H, W).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - iou_matrix (np.ndarray): (N_prev_objects, N_current_objects) matrix of IoU values.
                - prev_ids (np.ndarray): Array of unique object IDs from prev_mask.
                - current_ids (np.ndarray): Array of unique object IDs from current_mask.
        """
        logger.debug(f"Calculating IoU matrix between masks of shapes {prev_mask.shape} and {current_mask.shape}.")

        prev_ids = np.unique(prev_mask)
        prev_ids = prev_ids[prev_ids != 0]  # Exclude background

        current_ids = np.unique(current_mask)
        current_ids = current_ids[current_ids != 0]  # Exclude background

        iou_matrix = np.zeros((len(prev_ids), len(current_ids)), dtype=np.float32)

        if len(prev_ids) == 0 or len(current_ids) == 0:
            logger.debug("One or both masks are empty for IoU matrix calculation. Returning empty matrix.")
            return iou_matrix, prev_ids, current_ids

        for i, prev_id in enumerate(prev_ids):
            for j, current_id in enumerate(current_ids):
                mask1 = (prev_mask == prev_id)
                mask2 = (current_mask == current_id)

                intersection = np.logical_and(mask1, mask2).sum()
                if intersection == 0:
                    iou_matrix[i, j] = 0.0  # No intersection, IoU is 0
                    continue

                union = np.logical_or(mask1, mask2).sum()
                if union == 0:  # Should not happen if intersection > 0, but as a safeguard
                    logger.warning(f"Union is zero for prev_id {prev_id} and current_id {current_id}. IoU set to 0.0.")
                    iou_matrix[i, j] = 0.0
                else:
                    iou_matrix[i, j] = intersection / union

        logger.debug(f"IoU matrix calculated. Shape: {iou_matrix.shape}.")
        return iou_matrix, prev_ids, current_ids

    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Processes raw masks to produce tracked masks using the IoU algorithm,
        potentially leveraging a global projection step.

        Args:
            raw_masks (List[np.ndarray]): A list of independently segmented masks.
            frames (List[np.ndarray]): A list of original pixel frames (not directly used in core IoU matching, but passed for interface consistency).

        Returns:
            List[np.ndarray]: A list of re-labeled masks with consistent tracking IDs.
        """
        # Call the base class's track_frames for initial validation and logging
        try:
            super().track_frames(raw_masks, frames)
        except Exception as e:
            logger.error(f"BaseTracker validation failed: {e}", exc_info=True)
            raise  # Re-raise, as validation failure is critical

        self.reset()  # Start fresh for each call
        if not raw_masks:
            logger.warning("No raw masks provided. Returning empty list of tracked masks.")
            return []

        tracked_masks: List[np.ndarray] = []

        # --- Projection-based Instance Segmentation (Experimental) ---
        logger.info("Performing global projection for instance mask generation (HoughLinesP)...")
        if not raw_masks:
            logger.warning("No raw masks to create projection. Skipping projection step.")
            projection_mask = np.zeros(raw_masks[0].shape[:2], dtype=np.uint8)  # Create empty placeholder
        else:
            try:
                mask_array = np.array(raw_masks)
                # Max projection across time axis
                projection_mask = np.max(mask_array, axis=0)
                logger.debug(
                    f"Projection mask created from {len(raw_masks)} raw masks. Shape: {projection_mask.shape}.")
                # For HoughLinesP, it usually expects 8-bit single channel image
                if projection_mask.max() > 0:
                    projection_mask_uint8 = (projection_mask > 0).astype(np.uint8) * 255  # Convert to binary 0/255
                else:
                    projection_mask_uint8 = np.zeros_like(projection_mask, dtype=np.uint8)

                # --- Debug Visualization ---
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Displaying projection mask (likely for debug, will block).")
                    plt.figure(figsize=(6, 6))
                    plt.imshow(projection_mask_uint8, cmap='gray')
                    plt.title("Projection Mask (Debug)")
                    plt.axis("off")
                    plt.show()
                # --- End Debug Visualization ---

                lines = cv2.HoughLinesP(projection_mask_uint8,
                                        rho=1,
                                        theta=np.pi / 180,
                                        threshold=150,
                                        minLineLength=30,
                                        maxLineGap=5)

                if lines is None:
                    logger.info("No lines found in projection mask by HoughLinesP.")
                    instance_masks_from_lines = np.zeros_like(projection_mask, dtype=np.uint8)
                else:
                    logger.debug(f"Found {len(lines)} lines using HoughLinesP.")
                    instance_masks_from_lines = np.zeros_like(projection_mask, dtype=np.uint8)
                    for i, line in enumerate(lines.reshape(-1, 4)):
                        x1, y1, x2, y2 = line
                        # Draw each line with a unique ID (i+1 to avoid 0)
                        cv2.line(instance_masks_from_lines, (x1, y1), (x2, y2), i + 1, thickness=5)
                    logger.debug(f"Generated instance mask from {len(lines)} lines.")

                    # --- Debug Visualization ---
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Displaying instances mask from Hough lines (likely for debug, will block).")
                        plt.figure(figsize=(6, 6))
                        plt.imshow(instance_masks_from_lines)  # Should be colorized if many unique IDs
                        plt.title("Instances Mask from Hough Lines (Debug)")
                        plt.axis("off")
                        plt.show()
                    # --- End Debug Visualization ---

                # The `instance_masks_from_lines` can now be potentially used as a "ground truth"
                # for tracking or as a strong prior for initial object assignment.
                # Currently, this code proceeds with the original IoU matching logic,
                # so this projection is informational or a placeholder for future use.
                logger.warning(
                    "HoughLinesP projection result is generated but not yet integrated into the primary IoU tracking logic in this implementation.")

            except Exception as e:
                logger.error(f"Error during HoughLinesP projection step: {e}. Skipping projection part.", exc_info=True)
                instance_masks_from_lines = np.zeros_like(raw_masks[0], dtype=np.uint8)  # Fallback empty mask

        # --- Initialize with the first frame ---
        first_mask = raw_masks[0]
        tracked_first_mask = np.zeros_like(first_mask, dtype=np.uint16)
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
        for i in tqdm(range(1, len(raw_masks)), desc="Tracking with IoU"):
            logger.debug(f"Processing frame {i}...")
            prev_tracked_mask = tracked_masks[i - 1]
            current_raw_mask = raw_masks[i]
            new_tracked_mask = np.zeros_like(current_raw_mask, dtype=np.uint16)

            if np.max(current_raw_mask) == 0 or np.max(prev_tracked_mask) == 0:
                if np.max(current_raw_mask) == 0:
                    logger.debug(f"  Frame {i}: Current raw mask is empty. No objects to track or assign.")
                if np.max(prev_tracked_mask) == 0:
                    logger.debug(f"  Frame {i}: Previous tracked mask is empty. All current objects are new.")
                    for new_raw_id in np.unique(current_raw_mask)[1:]:
                        new_tracked_mask[current_raw_mask == new_raw_id] = self.next_track_id
                        logger.debug(
                            f"    Frame {i}: Raw ID {new_raw_id} assigned to new track ID {self.next_track_id}.")
                        self.next_track_id += 1
                tracked_masks.append(new_tracked_mask)
                continue

            try:
                iou_matrix, prev_ids, current_ids = self._calculate_iou_matrix(
                    prev_tracked_mask, current_raw_mask)

                if iou_matrix.size == 0:
                    logger.debug(f"  Frame {i}: IoU matrix is empty. No matches possible. Assigning new IDs.")
                    # Treat all current objects as new if no matches
                    for new_raw_id in np.unique(current_raw_mask)[1:]:
                        new_tracked_mask[current_raw_mask == new_raw_id] = self.next_track_id
                        logger.debug(
                            f"    Frame {i}: Raw ID {new_raw_id} assigned to new track ID {self.next_track_id}.")
                        self.next_track_id += 1
                    tracked_masks.append(new_tracked_mask)
                    continue

                cost_matrix = 1 - iou_matrix

                prev_indices, current_indices = linear_sum_assignment(cost_matrix)
                logger.debug(f"  Frame {i}: Hungarian algorithm found {len(prev_indices)} potential matches.")
            except Exception as e:
                logger.error(f"  Frame {i}: Error during IoU matrix or Hungarian assignment: {e}. Skipping frame.",
                             exc_info=True)
                tracked_masks.append(new_tracked_mask)
                continue

            matched_current_ids: Set[int] = set()
            matches_count = 0
            for prev_idx_in_matrix, current_idx_in_matrix in zip(prev_indices, current_indices):
                iou = iou_matrix[prev_idx_in_matrix, current_idx_in_matrix]

                if iou >= self.iou_threshold:
                    prev_track_id = prev_ids[prev_idx_in_matrix]
                    current_raw_id = current_ids[current_idx_in_matrix]

                    new_tracked_mask[current_raw_mask == current_raw_id] = prev_track_id
                    matched_current_ids.add(current_raw_id)
                    matches_count += 1
                    logger.debug(
                        f"    Frame {i}: Matched prev track ID {prev_track_id} to current raw ID {current_raw_id} (IoU: {iou:.4f}).")
                else:
                    logger.debug(f"    Frame {i}: Match rejected (IoU {iou:.4f} < iou_threshold {self.iou_threshold}).")
            logger.debug(f"  Frame {i}: Found {matches_count} matches within IoU threshold.")

            unmatched_current_ids_raw = set(current_ids) - matched_current_ids
            for new_raw_id in unmatched_current_ids_raw:
                new_tracked_mask[current_raw_mask == new_raw_id] = self.next_track_id
                logger.debug(f"    Frame {i}: Raw ID {new_raw_id} assigned to new track ID {self.next_track_id}.")
                self.next_track_id += 1
            logger.debug(f"  Frame {i}: Assigned new IDs to {len(unmatched_current_ids_raw)} unmatched objects.")

            tracked_masks.append(new_tracked_mask)
            logger.info(
                f"Frame {i} processed. Total objects in frame: {len(np.unique(new_tracked_mask)[1:])}. Next track ID: {self.next_track_id}.")

        logger.info(f"ProjectionTracker finished. Returning {len(tracked_masks)} tracked masks.")
        return tracked_masks