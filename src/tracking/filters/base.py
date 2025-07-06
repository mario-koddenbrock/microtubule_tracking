import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Any  


logger = logging.getLogger(f"microtuble_tracking.{__name__}")


class BaseFilter(ABC):
    """
    An abstract base class for mask filters.

    Defines a common interface for classes that filter objects from a
    segmentation mask based on certain criteria.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initializes the base filter.

        Concrete filter implementations should call `super().__init__(...)`
        and can add their specific configuration parameters here.
        """
        logger.debug(f"Initializing BaseFilter instance of type '{self.__class__.__name__}'.")
        # Subclasses should add their specific configuration logging here
        # For example: logger.info(f"Initialized {self.__class__.__name__} with min_size={self.min_size}.")

    @abstractmethod
    def filter(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies a filtering logic to a segmentation mask.

        Args:
            mask (np.ndarray): A 2D integer label mask where each unique integer
                               corresponds to a distinct object.

        Returns:
            np.ndarray: A new mask containing only the objects that passed
                        the filtering criteria. The original labels are preserved.
        """
        if not isinstance(mask, np.ndarray):
            msg = f"Input mask must be a numpy array, but got {type(mask)}."
            logger.error(msg)
            raise TypeError(msg)
        if mask.ndim != 2:
            msg = f"Input mask must be a 2D array, but got {mask.ndim} dimensions (shape: {mask.shape})."
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(f"Filtering mask of shape {mask.shape} using '{self.__class__.__name__}'. "
                     f"Initial unique labels: {len(np.unique(mask)) - 1} (excluding background 0).")

        # Subclasses should add their specific filtering logic and logging here.
        # For example:
        # num_objects_before = len(np.unique(mask)) - 1
        # filtered_mask = ... actual filtering logic ...
        # num_objects_after = len(np.unique(filtered_mask)) - 1
        # logger.info(f"Filter '{self.__class__.__name__}' reduced objects from {num_objects_before} to {num_objects_after}.")

        pass