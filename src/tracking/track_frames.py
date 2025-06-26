from typing import List

import numpy as np
from tqdm import tqdm

from segmentation.base_model import BaseSegmentationModel
from tracking.base_tracker import BaseTracker


def track_frames(
        frames: List[np.ndarray],
        model: BaseSegmentationModel,
        tracker: BaseTracker,
) -> List[np.ndarray]:
    """
    Full pipeline: segments all frames and then tracks objects across them.
    This function works with any BaseTracker implementation.
    """
    # Stage 1: Segmentation (unchanged)
    print("Stage 1: Performing segmentation on all frames...")
    raw_masks = [model.predict(frame) for frame in tqdm(frames, desc="Segmenting")]

    # Stage 2: Tracking (works with any tracker)
    print(f"\nStage 2: Handing off to {tracker.__class__.__name__}...")
    tracked_masks = tracker.track_frames(raw_masks)

    return tracked_masks