import logging
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np

logger = logging.getLogger(f"mt.{__name__}")


class BaseTracker(ABC):
    """
    An abstract base class for object trackers.

    This class defines the common interface and state management for tracking
    objects across a sequence of segmentation masks. Any concrete tracker
    implementation should inherit from this class.
    """

    def __init__(self):
        """Initializes the base tracker state."""
        logger.debug(f"Initializing BaseTracker instance of type '{self.__class__.__name__}'.")
        self.next_track_id = 1
        self.tracks: Dict[int, Dict] = {}  # A generic place to store info about active tracks
        logger.debug(
            f"Tracker state initialized: next_track_id={self.next_track_id}, active tracks: {len(self.tracks)}.")

    @abstractmethod
    def track_frames(self, raw_masks: List[np.ndarray], frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Processes a list of raw segmentation masks to produce tracked masks.

        This method must be implemented by all child classes.

        Args:
            raw_masks (List[np.ndarray]): A list of independently segmented masks
                                       where labels are not consistent over time.
            frames (List[np.ndarray]): A list of original pixel frames corresponding

        Returns:
            List[np.ndarray]: A list of re-labeled masks with consistent tracking IDs.
        """
        # Basic input validation and logging for subclasses to build upon
        if not isinstance(raw_masks, list) or not all(isinstance(m, np.ndarray) for m in raw_masks):
            msg = "Input 'raw_masks' must be a list of numpy arrays."
            logger.error(msg)
            raise TypeError(msg)
        if not isinstance(frames, list) or not all(isinstance(f, np.ndarray) for f in frames):
            msg = "Input 'frames' must be a list of numpy arrays."
            logger.error(msg)
            raise TypeError(msg)
        if len(raw_masks) != len(frames):
            msg = f"Number of raw_masks ({len(raw_masks)}) must match number of frames ({len(frames)})."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Starting tracking process for {len(raw_masks)} frames using '{self.__class__.__name__}'.")
        # Subclasses should add detailed logging here for their specific tracking logic.
        # For example:
        # for frame_idx, (raw_mask, frame) in enumerate(zip(raw_masks, frames)):
        #     logger.debug(f"Processing frame {frame_idx} with mask shape {raw_mask.shape}.")
        #     ... tracking logic ...
        #     logger.debug(f"Frame {frame_idx} processed. New track IDs assigned: {newly_assigned_ids}.")

        pass

    def reset(self) -> None:
        """Resets the tracker's state to start a new tracking session."""
        logger.info("Resetting tracker state.")
        self.next_track_id = 1
        self.tracks = {}
        logger.debug(f"Tracker state reset: next_track_id={self.next_track_id}, active tracks: {len(self.tracks)}.")