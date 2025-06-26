import argparse

import numpy as np
from tqdm import tqdm

from file_io.utils import extract_frames
from plotting.plotting import visualize_tracked_masks
from segmentation.cellpose import CellposePredictor
from tracking.centroid_tracker import CentroidTracker
from tracking.filters.morphology import MorphologyFilter
from tracking.filters.spatial import CornerExclusionFilter
from tracking.iou_tracker import IoUTracker
from tracking.opencv_tracker import OpenCVTracker
from tracking.projection_tracker import ProjectionTracker


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
    elif args.tracker == 'csrt':
        tracker = OpenCVTracker(tracker_type='csrt', iou_threshold=args.iou_threshold)
    elif args.tracker == 'projection':
        tracker = ProjectionTracker(iou_threshold=args.iou_threshold)
        raise NotImplementedError("ProjectionTracker is not yet implemented.")
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
    tracked_masks = tracker.track_frames(raw_masks, frames)
    print(f"Number of tracked masks: {len(np.unique(tracked_masks))}")


    # 6. Visualize and Export
    print("Stage 3/3: Visualizing results and exporting video...")
    visualize_tracked_masks(
        frames,
        tracked_masks,
        args.video,
        args.output_path,
        fps,
        alpha=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run segmentation and tracking on a video file and export the result."
    )

    # Required arguments
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file (.mp4, .avi, .tif, etc.).")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the cellpose_config.json file.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output video.")

    # Model and Tracker choice
    parser.add_argument("--tracker", type=str, choices=['iou', 'centroid', 'csrt', 'projection'], default='csrt',
                        help="Tracking algorithm to use.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for model inference (e.g., 'cuda', 'cpu'). Auto-detects if not set.")

    # Tracker-specific parameters
    parser.add_argument("--iou-threshold", type=float, default=0.1,
                        help="IoU threshold for the tracker.")
    parser.add_argument("--max-distance", type=int, default=50,
                        help="Max centroid distance for the Centroid tracker.")

    args = parser.parse_args()

    main(args)
