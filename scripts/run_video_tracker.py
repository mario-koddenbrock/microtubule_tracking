import argparse
from typing import List, Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from file_io.utils import extract_frames
from segmentation.cellpose import CellposePredictor
from tracking.centroid_tracker import CentroidTracker
from tracking.filters.base import BaseFilter
from tracking.filters.morphology import MorphologyFilter
from tracking.filters.spatial import CornerExclusionFilter
from tracking.iou_tracker import IoUTracker


# --- New Visualization and Pipeline Logic ---
def get_color_map(num_colors: int) -> Dict[int, Tuple[int, int, int]]:
    """Generates a map of track IDs to unique BGR colors."""
    # Use a HSV colormap for better color distinction
    return {
        i: tuple(hsv_to_bgr(i * (180 / num_colors), 255, 255))
        for i in range(1, num_colors + 1)
    }


def hsv_to_bgr(h, s, v):
    """Simple HSV to BGR conversion for color generation."""
    h_i = int(h / 60) % 6
    f = h / 60 - h_i
    p = v * (1 - s / 255)
    q = v * (1 - f * s / 255)
    t = v * (1 - (1 - f) * s / 255)

    v, t, p, q = int(v), int(t), int(p), int(q)

    if h_i == 0: return (p, t, v)
    if h_i == 1: return (p, v, q)
    if h_i == 2: return (q, v, p)
    if h_i == 3: return (v, t, p)
    if h_i == 4: return (v, p, q)
    if h_i == 5: return (t, p, v)


def visualize_tracked_masks(
        frames: List[np.ndarray],
        tracked_masks: List[np.ndarray],
        output_path: str,
        fps: float
):
    """
    Overlays tracked masks on frames and saves the result as an MP4 video.
    """
    if not frames or not tracked_masks:
        print("Cannot visualize, no frames or masks provided.")
        return

    # Determine total number of unique tracks to create a consistent color map
    all_track_ids = set()
    for mask in tracked_masks:
        all_track_ids.update(np.unique(mask))
    all_track_ids.discard(0)  # Remove background

    if not all_track_ids:
        print("No objects were tracked. Saving original video.")
        color_map = {}
    else:
        max_id = max(all_track_ids)
        color_map = get_color_map(max_id)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, max(1, fps), (w, h))

    for frame, mask in tqdm(zip(frames, tracked_masks), total=len(frames), desc="Visualizing & Saving", unit="frame"):
        vis_frame = frame.copy()
        track_ids = np.unique(mask)[1:]  # Get objects in current frame

        for track_id in track_ids:
            color = color_map.get(track_id, (255, 255, 255))  # Default to white if ID is missing
            contours, _ = cv2.findContours((mask == track_id).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_frame, contours, -1, color, 2)

        writer.write(vis_frame)

    writer.release()
    print(f"Successfully saved visualized video to: {output_path}")


def main(args):
    """Main execution function."""

    # 1. Load video into frames
    print(f"Loading video from: {args.video}")
    frames, fps = extract_frames(args.video, color_mode="bgr")
    if not frames:
        return

    # 2. Initialize Segmentation Model
    # For this script, we assume Cellpose, but this could be extended
    model = CellposePredictor(config_path=args.config, device_str=args.device)

    # 3. Initialize Filter Chain
    filters = [CornerExclusionFilter(), MorphologyFilter()]

    # 4. Initialize Tracker
    if args.tracker == 'iou':
        tracker = IoUTracker(iou_threshold=args.iou_threshold)
    elif args.tracker == 'centroid':
        tracker = CentroidTracker(max_distance=args.max_distance)
    else:
        raise ValueError(f"Unknown tracker type: {args.tracker}")
    print(f"Using {tracker.__class__.__name__} for tracking.")

    # 5. Run Segmentation and Filtering (MODIFIED to use the filter chain)
    print("Stage 1/3: Running segmentation and filtering...")
    raw_masks = []
    process_desc = "Segmenting & Filtering" if filters else "Segmenting"
    for frame in tqdm(frames, desc=process_desc, unit="frame"):
        # Step A: Get the raw mask from the model
        mask = model.predict(frame)

        # Step B: Apply each filter in the chain sequentially
        for f in filters:
            mask = f.filter(mask)

        raw_masks.append(mask)

    # 5. Run Tracking
    print("Stage 2/3: Running tracking on segmented masks...")
    tracked_masks = tracker.track_frames(raw_masks)

    # 6. Visualize and Export
    print("Stage 3/3: Visualizing results and exporting video...")
    visualize_tracked_masks(frames, tracked_masks, args.output, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run segmentation and tracking on a video file and export the result."
    )

    # Required arguments
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file (.mp4, .avi, .tif, etc.).")
    parser.add_argument("--config", type=str, required=True, help="Path to the cellpose_config.json file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output MP4 video.")

    # Model and Tracker choice
    parser.add_argument("--tracker", type=str, choices=['iou', 'centroid'], default='iou',
                        help="Tracking algorithm to use.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for model inference (e.g., 'cuda', 'cpu'). Auto-detects if not set.")

    # Tracker-specific parameters
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for the IoU tracker.")
    parser.add_argument("--max-distance", type=int, default=50, help="Max centroid distance for the Centroid tracker.")

    args = parser.parse_args()

    main(args)
