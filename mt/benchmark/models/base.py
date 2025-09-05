# src/benchmark/models/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseModel(ABC):
    """Abstract base class for a segmentation model."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @classmethod
    @abstractmethod
    def get_model_name(cls) -> str:
        """Return the unique name of the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predicts instance masks for a single image.

        Args:
            image: A single input image as a NumPy array (H, W, C).

        Returns:
            A NumPy array of instance masks (num_instances, H, W), where each
            slice along the first axis is a binary mask for one instance.
        """
        pass

    def __str__(self) -> str:
        return self.model_name