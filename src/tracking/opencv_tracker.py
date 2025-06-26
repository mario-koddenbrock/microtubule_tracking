import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .base_tracker import BaseTracker

# source CSRT https://arxiv.org/abs/1611.08461

class OpenCVTracker(BaseTracker):
    """
    Tracks objects using OpenCV's built-in, stateful tracker algorithms (e.g., CSRT).
    """

    def __init__(self, tracker_type: str = "csrt", iou_threshold: float = 0.2):
        super().__init__()
        self.tracker_type = tracker_type.lower()
        self.iou_threshold = iou_threshold
        self.active_trackers: Dict[int, Any] = {}

    def _create_tracker(self) -> Any:
        # ... (this function is unchanged)
        if self.tracker_type == "csrt": return cv2.TrackerCSRT_create()
        if self.tracker_type == "kcf": return cv2.TrackerKCF_create()
        raise ValueError(f"Unsupported OpenCV tracker type: '{self.tracker_type}'")

    def _get_bounding_boxes(self, mask: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        # ... (this function is unchanged, our previous fix is still correct)
        boxes = {}
        properties = regionprops(mask)
        for prop in properties:
            min_r, min_c, max_r, max_c = prop.bbox
            x, y = min_c, min_r
            w, h = max_c - min_c, max_r - min_r
            if w > 0 and h > 0:
                boxes[prop.label] = (x, y, w, h)
        return boxes

    def _calculate_bbox_iou(self, boxA, boxB):
        # ... (this function is unchanged)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        if boxAArea + boxBArea - interArea == 0: return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray]) -> List[np.ndarray]:
        self.reset()
        if not raw_masks or not frames: return []
        if len(raw_masks) != len(frames):
            raise ValueError("The number of masks must equal the number of frames.")

        tracked_masks = []

        # --- Frame 0: Initialization ---
        first_frame = frames[0]
        H, W = first_frame.shape[:2]
        first_mask = raw_masks[0]
        tracked_first_mask = np.zeros_like(first_mask)

        initial_bboxes = self._get_bounding_boxes(first_mask)
        for raw_id, bbox in initial_bboxes.items():
            # --- NEW BOUNDARY CHECK (1) ---
            x, y, w, h = bbox
            if x + w >= W or y + h >= H:
                continue  # Skip objects that are out of bounds

            tracker = self._create_tracker()
            tracker.init(first_frame, bbox)

            track_id = self.next_track_id
            self.active_trackers[track_id] = tracker
            tracked_first_mask[first_mask == raw_id] = track_id
            self.next_track_id += 1

        tracked_masks.append(tracked_first_mask)

        # --- Frames 1 to N: Update, Match, and Add New ---
        for i in tqdm(range(1, len(frames)), desc=f"Tracking with {self.tracker_type.upper()}"):
            frame = frames[i]
            H, W = frame.shape[:2]  # Get current frame dimensions
            raw_mask = raw_masks[i]
            new_tracked_mask = np.zeros_like(raw_mask)

            # --- 1. Update existing trackers ---
            predicted_boxes: Dict[int, Tuple] = {}
            lost_tracks = set()
            for track_id, tracker in self.active_trackers.items():
                ok, bbox = tracker.update(frame)
                if ok:
                    # Also validate the predicted bbox
                    x, y, w, h = [int(v) for v in bbox]
                    if w > 0 and h > 0 and x + w < W and y + h < H:
                        predicted_boxes[track_id] = (x, y, w, h)
                    else:
                        lost_tracks.add(track_id)  # Treat invalid predictions as lost tracks
                else:
                    lost_tracks.add(track_id)

            for track_id in lost_tracks:
                self.active_trackers.pop(track_id, None)

            # --- 2. Match predictions with current segmentations ---
            current_bboxes = self._get_bounding_boxes(raw_mask)

            # ... (matching logic remains the same) ...
            if predicted_boxes and current_bboxes:
                pred_ids, pred_bboxes_list = list(predicted_boxes.keys()), list(predicted_boxes.values())
                curr_ids, curr_bboxes_list = list(current_bboxes.keys()), list(current_bboxes.values())
                iou_matrix = np.zeros((len(pred_ids), len(curr_ids)), dtype=np.float32)
                for r, pred_box in enumerate(pred_bboxes_list):
                    for c, curr_box in enumerate(curr_bboxes_list):
                        iou_matrix[r, c] = self._calculate_bbox_iou(pred_box, curr_box)
                pred_indices, curr_indices = linear_sum_assignment(1 - iou_matrix)
                matched_curr_ids = set()
                for p_idx, c_idx in zip(pred_indices, curr_indices):
                    if iou_matrix[p_idx, c_idx] >= self.iou_threshold:
                        track_id, raw_id = pred_ids[p_idx], curr_ids[c_idx]
                        new_tracked_mask[raw_mask == raw_id] = track_id
                        matched_curr_ids.add(raw_id)
            else:
                matched_curr_ids = set()

            # --- 3. Add new, unmatched objects as new tracks ---
            unmatched_curr_ids = set(current_bboxes.keys()) - matched_curr_ids
            for raw_id in unmatched_curr_ids:
                bbox = current_bboxes[raw_id]

                x, y, w, h = bbox
                if (x + w >= W) or (y + h >= H) or (w <= 1) or (h <= 1):
                    continue

                tracker = self._create_tracker()
                tracker.init(frame, bbox)

                track_id = self.next_track_id
                self.active_trackers[track_id] = tracker
                new_tracked_mask[raw_mask == raw_id] = track_id
                self.next_track_id += 1

            tracked_masks.append(new_tracked_mask)

        return tracked_masks