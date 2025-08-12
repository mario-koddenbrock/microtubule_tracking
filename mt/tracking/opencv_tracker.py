import logging
from typing import List, Dict, Any, Tuple, Set

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .base import BaseTracker

logger = logging.getLogger(f"mt.{__name__}")


# source CSRT https://arxiv.org/abs/1611.08461

class OpenCVTracker(BaseTracker):
    """
    Tracks objects using OpenCV's built-in, stateful tracker algorithms (e.g., CSRT).
    """

    def __init__(self, tracker_type: str = "csrt", iou_threshold: float = 0.2):
        """
        Initializes the OpenCVTracker.

        Args:
            tracker_type (str): The type of OpenCV tracker to use ("csrt", "kcf").
            iou_threshold (float): Minimum IoU to consider two bounding boxes a match.
        """
        logger.info(f"Initializing OpenCVTracker with type: '{tracker_type}' and iou_threshold: {iou_threshold}.")
        super().__init__()  # Call the base class constructor

        if not (0.0 < iou_threshold < 1.0):
            msg = f"iou_threshold must be strictly between 0 and 1, but got {iou_threshold}."
            logger.error(msg)
            raise ValueError(msg)

        self.tracker_type = tracker_type.lower()
        self.iou_threshold = iou_threshold
        self.active_trackers: Dict[int, Any] = {}  # Maps track_id to OpenCV tracker object
        logger.debug("OpenCVTracker initialization complete.")

    def _create_tracker(self) -> Any:
        """Helper to instantiate the chosen OpenCV tracker."""
        logger.debug(f"Creating OpenCV tracker of type: '{self.tracker_type}'.")
        try:
            if self.tracker_type == "csrt": return cv2.TrackerCSRT_create()  # Use _create() method
            if self.tracker_type == "kcf": return cv2.TrackerKCF_create()  # Use _create() method
            msg = f"Unsupported OpenCV tracker type: '{self.tracker_type}'. Supported: 'csrt', 'kcf'."
            logger.error(msg)
            raise ValueError(msg)
        except AttributeError as e:
            logger.error(
                f"Failed to create OpenCV tracker '{self.tracker_type}'. Make sure OpenCV is compiled with contrib modules: {e}",
                exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred creating tracker '{self.tracker_type}': {e}", exc_info=True)
            raise

    def _get_bounding_boxes(self, mask: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Calculates bounding boxes for each object in a mask.

        Returns:
            Dict[int, Tuple[int, int, int, int]]: Maps object label to (x, y, w, h) bounding box.
        """
        logger.debug(f"Getting bounding boxes for mask of shape {mask.shape}.")
        boxes: Dict[int, Tuple[int, int, int, int]] = {}
        if np.max(mask) == 0:
            logger.debug("Mask is empty, no bounding boxes to extract.")
            return boxes

        try:
            properties = regionprops(mask)
            for prop in properties:
                min_r, min_c, max_r, max_c = prop.bbox
                x, y = min_c, min_r
                w, h = max_c - min_c, max_r - min_r

                # Filter out invalid or tiny bounding boxes
                if w > 0 and h > 0:
                    boxes[prop.label] = (x, y, w, h)
                    logger.debug(f"  Object {prop.label}: BBox ({x}, {y}, {w}, {h}).")
                else:
                    logger.warning(f"  Object {prop.label}: Invalid bounding box (w={w}, h={h}). Skipping.")
            logger.debug(f"Extracted {len(boxes)} bounding boxes from mask.")
        except Exception as e:
            logger.error(f"Error extracting bounding boxes from mask of shape {mask.shape}: {e}", exc_info=True)
        return boxes

    def _calculate_bbox_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Calculates IoU for two bounding boxes (x, y, w, h)."""
        # logger.debug(f"Calculating BBox IoU for {boxA} and {boxB}.")

        # Unpack boxes (x, y, w, h) -> (x1, y1, x2, y2)
        boxA_x1, boxA_y1, boxA_w, boxA_h = boxA
        boxA_x2, boxA_y2 = boxA_x1 + boxA_w, boxA_y1 + boxA_h

        boxB_x1, boxB_y1, boxB_w, boxB_h = boxB
        boxB_x2, boxB_y2 = boxB_x1 + boxB_w, boxB_y1 + boxB_h

        # Determine the coordinates of the intersection rectangle
        xA = max(boxA_x1, boxB_x1)
        yA = max(boxA_y1, boxB_y1)
        xB = min(boxA_x2, boxB_x2)
        yB = min(boxA_y2, boxB_y2)

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both bounding boxes
        boxAArea = boxA_w * boxA_h
        boxBArea = boxB_w * boxB_h

        # Compute the union area
        unionArea = float(boxAArea + boxBArea - interArea)
        if unionArea == 0:
            # logger.warning(f"Union area is zero for boxes {boxA}, {boxB}. IoU set to 0.0.")
            return 0.0

        iou = interArea / unionArea
        # logger.debug(f"  BBox IoU: {iou:.4f}.")
        return iou

    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Processes raw masks and frames to produce tracked masks using OpenCV's tracker.

        Args:
            raw_masks (List[np.ndarray]): List of independently segmented masks.
            frames (List[np.ndarray]): List of original pixel frames.

        Returns:
            List[np.ndarray]: List of re-labeled masks with consistent tracking IDs.
        """
        # Call the base class's track_frames for initial validation and logging
        try:
            super().track_frames(raw_masks, frames)
        except Exception as e:
            logger.error(f"BaseTracker validation failed: {e}", exc_info=True)
            raise  # Re-raise, as validation failure is critical

        self.reset()  # Reset tracker state (base class does this, but good to be explicit for clarity)
        if not raw_masks or not frames:
            logger.warning("No raw masks or frames provided. Returning empty list of tracked masks.")
            return []

        tracked_masks: List[np.ndarray] = []

        # --- Frame 0: Initialization ---
        logger.info("Frame 0: Initializing OpenCV trackers for first frame objects.")
        first_frame = frames[0]
        H, W = first_frame.shape[:2]
        first_mask = raw_masks[0]
        tracked_first_mask = np.zeros_like(first_mask, dtype=np.uint16)

        initial_bboxes = self._get_bounding_boxes(first_mask)
        logger.debug(f"Frame 0: Found {len(initial_bboxes)} initial bounding boxes.")

        for raw_id, bbox in initial_bboxes.items():
            x, y, w, h = bbox
            # --- Boundary Check (1) ---
            # Objects too close to boundary (or invalid size) are skipped for initial tracking
            if x < 0 or y < 0 or x + w > W or y + h > H or w <= 0 or h <= 0:
                logger.warning(
                    f"Frame 0, Raw ID {raw_id}: Initial bbox {bbox} is out of image bounds or invalid ({W}x{H}). Skipping object.")
                continue

            try:
                tracker = self._create_tracker()
                # OpenCV tracker init expects BGR images. Ensure frame is BGR.
                init_frame_bgr = first_frame if first_frame.ndim == 3 and first_frame.shape[2] == 3 else cv2.cvtColor(
                    first_frame, cv2.COLOR_GRAY2BGR)
                ok = tracker.init(init_frame_bgr, bbox)
                if not ok:
                    logger.warning(
                        f"Frame 0, Raw ID {raw_id}: Failed to initialize OpenCV tracker for bbox {bbox}. Skipping object.")
                    continue

                track_id = self.next_track_id
                self.active_trackers[track_id] = tracker
                tracked_first_mask[first_mask == raw_id] = track_id
                logger.debug(f"Frame 0, Raw ID {raw_id}: Initialized new track ID {track_id} for bbox {bbox}.")
                self.next_track_id += 1
            except Exception as e:
                logger.error(f"Frame 0, Raw ID {raw_id}: Error during tracker initialization for bbox {bbox}: {e}",
                             exc_info=True)
                continue  # Skip this object if its tracker fails to init

        tracked_masks.append(tracked_first_mask)
        logger.info(
            f"Frame 0 processed. {len(self.active_trackers)} active trackers. Next track ID: {self.next_track_id}.")

        # --- Frames 1 to N: Update, Match, and Add New ---
        for i in tqdm(range(1, len(frames)), desc=f"Tracking with {self.tracker_type.upper()}"):
            logger.debug(f"Processing frame {i}...")
            frame = frames[i]
            H, W = frame.shape[:2]
            raw_mask = raw_masks[i]
            new_tracked_mask = np.zeros_like(raw_mask, dtype=np.uint16)

            # Ensure frame is BGR for OpenCV trackers
            process_frame_bgr = frame if frame.ndim == 3 and frame.shape[2] == 3 else cv2.cvtColor(frame,
                                                                                                   cv2.COLOR_GRAY2BGR)

            # --- 1. Update existing trackers ---
            predicted_boxes: Dict[int, Tuple[int, int, int, int]] = {}  # {track_id: bbox}
            lost_tracks: Set[int] = set()
            logger.debug(f"  Frame {i}: Updating {len(self.active_trackers)} active trackers.")

            for track_id, tracker in list(self.active_trackers.items()):  # Iterate over copy as dict may change
                try:
                    ok, bbox = tracker.update(process_frame_bgr)  # Update tracker with current frame
                    if ok:
                        x, y, w, h = [int(v) for v in bbox]
                        # Validate predicted bbox: must be positive dimensions and within image bounds
                        if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= W and y + h <= H:
                            predicted_boxes[track_id] = (x, y, w, h)
                            logger.debug(f"    Track {track_id}: Predicted bbox ({x}, {y}, {w}, {h}).")
                        else:
                            lost_tracks.add(track_id)
                            logger.info(
                                f"    Track {track_id}: Predicted bbox ({x},{y},{w},{h}) is invalid or out of bounds ({W}x{H}). Marking as lost.")
                    else:
                        lost_tracks.add(track_id)
                        logger.info(f"    Track {track_id}: Tracker reported failure ('ok' is False). Marking as lost.")
                except Exception as e:
                    lost_tracks.add(track_id)
                    logger.error(f"    Track {track_id}: Error updating tracker: {e}. Marking as lost.", exc_info=True)

            for track_id in lost_tracks:
                self.active_trackers.pop(track_id, None)  # Remove lost trackers
                logger.debug(
                    f"  Frame {i}: Removed lost track {track_id}. Active trackers remaining: {len(self.active_trackers)}.")
            logger.debug(f"  Frame {i}: {len(predicted_boxes)} valid predictions from active trackers.")

            # --- 2. Match predictions with current segmentations ---
            current_bboxes = self._get_bounding_boxes(raw_mask)
            logger.debug(f"  Frame {i}: Found {len(current_bboxes)} bounding boxes in current raw mask.")

            if predicted_boxes and current_bboxes:
                pred_ids, pred_bboxes_list = list(predicted_boxes.keys()), list(predicted_boxes.values())
                curr_ids, curr_bboxes_list = list(current_bboxes.keys()), list(current_bboxes.values())

                iou_matrix = np.zeros((len(pred_ids), len(curr_ids)), dtype=np.float32)
                try:
                    for r, pred_box in enumerate(pred_bboxes_list):
                        for c, curr_box in enumerate(curr_bboxes_list):
                            iou_matrix[r, c] = self._calculate_bbox_iou(pred_box, curr_box)
                    logger.debug(f"  Frame {i}: IoU matrix calculated. Shape: {iou_matrix.shape}.")

                    cost_matrix = 1 - iou_matrix
                    pred_indices, curr_indices = linear_sum_assignment(cost_matrix)
                    logger.debug(f"  Frame {i}: Hungarian algorithm found {len(pred_indices)} potential matches.")
                except Exception as e:
                    logger.error(
                        f"  Frame {i}: Error calculating IoU matrix or during linear_sum_assignment: {e}. Skipping matching.",
                        exc_info=True)
                    pred_indices, curr_indices = [], []  # Force no matches if error occurs

                matched_curr_ids: Set[int] = set()  # Raw IDs from current_mask that have been matched
                matches_count = 0
                for p_idx, c_idx in zip(pred_indices, curr_indices):
                    iou = iou_matrix[p_idx, c_idx]
                    if iou >= self.iou_threshold:
                        track_id = pred_ids[p_idx]  # Track ID from prediction
                        raw_id = curr_ids[c_idx]  # Raw ID from current segmentation

                        new_tracked_mask[raw_mask == raw_id] = track_id
                        matched_curr_ids.add(raw_id)
                        matches_count += 1
                        logger.debug(
                            f"    Frame {i}: Matched pred track ID {track_id} to current raw ID {raw_id} (IoU: {iou:.4f}).")
                    else:
                        logger.debug(
                            f"    Frame {i}: Match rejected (IoU {iou:.4f} < iou_threshold {self.iou_threshold}).")
                logger.debug(f"  Frame {i}: Found {matches_count} matches within IoU threshold.")
            else:
                logger.debug(f"  Frame {i}: No predicted or current bounding boxes for matching.")
                matched_curr_ids = set()  # Ensure it's empty

            # --- 3. Add new, unmatched objects as new tracks ---
            unmatched_curr_ids_raw = set(current_bboxes.keys()) - matched_curr_ids
            logger.debug(f"  Frame {i}: Found {len(unmatched_curr_ids_raw)} unmatched objects in current raw mask.")

            for raw_id in unmatched_curr_ids_raw:
                bbox = current_bboxes[raw_id]
                x, y, w, h = bbox
                # Boundary check for new objects too
                if (x < 0 or y < 0 or x + w > W or y + h > H or w <= 0 or h <= 0):
                    logger.warning(
                        f"  Frame {i}, Raw ID {raw_id}: New object bbox {bbox} is out of image bounds or invalid ({W}x{H}). Skipping.")
                    continue

                try:
                    tracker = self._create_tracker()
                    ok = tracker.init(process_frame_bgr, bbox)
                    if not ok:
                        logger.warning(
                            f"  Frame {i}, Raw ID {raw_id}: Failed to initialize new OpenCV tracker for bbox {bbox}. Skipping object.")
                        continue

                    track_id = self.next_track_id
                    self.active_trackers[track_id] = tracker
                    new_tracked_mask[raw_mask == raw_id] = track_id
                    logger.debug(
                        f"    Frame {i}: Raw ID {raw_id} assigned to new track ID {track_id} with bbox {bbox}.")
                    self.next_track_id += 1
                except Exception as e:
                    logger.error(f"  Frame {i}, Raw ID {raw_id}: Error initializing new tracker for bbox {bbox}: {e}",
                                 exc_info=True)
                    continue  # Skip this object if its tracker fails to init

            tracked_masks.append(new_tracked_mask)
            logger.info(
                f"Frame {i} processed. Active trackers: {len(self.active_trackers)}. Next track ID: {self.next_track_id}.")

        logger.info(f"OpenCVTracker finished. Returning {len(tracked_masks)} tracked masks.")
        return tracked_masks