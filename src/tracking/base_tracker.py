from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

class BaseTracker(ABC):
    """
    An abstract base class for object trackers.

    This class defines the common interface and state management for tracking
    objects across a sequence of segmentation masks. Any concrete tracker
    implementation should inherit from this class.
    """

    def __init__(self):
        """Initializes the base tracker state."""
        self.next_track_id = 1
        self.tracks: Dict[int, Dict] = {} # A generic place to store info about active tracks

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
        pass

    def reset(self) -> None:
        """Resets the tracker's state to start a new tracking session."""
        print("Resetting tracker state.")
        self.next_track_id = 1
        self.tracks = {}