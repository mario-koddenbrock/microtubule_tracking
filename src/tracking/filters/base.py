from abc import ABC, abstractmethod
import numpy as np

class BaseFilter(ABC):
    """
    An abstract base class for mask filters.

    Defines a common interface for classes that filter objects from a
    segmentation mask based on certain criteria.
    """

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
        pass