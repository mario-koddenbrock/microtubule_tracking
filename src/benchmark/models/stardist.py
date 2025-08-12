# src/benchmark/models/stardist.py
import numpy as np
from .base import BaseModel

class StarDist(BaseModel):
    def __init__(self):
        super().__init__("StarDist")

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Placeholder: returns an empty array of masks
        print(f"Predicting with {self.model_name}...")
        return np.empty((0, *image.shape[:2]), dtype=np.uint16)
