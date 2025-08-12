import cv2
import logging
import numpy as np
from .base import BaseModel

logger = logging.getLogger(__name__)

class FIESTA(BaseModel):
    def __init__(self):
        super().__init__("FIESTA")
        logger.debug("Loading FIESTA model")

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Placeholder: returns an empty array of masks
        logger.debug(f"Predicting image {image.shape}")
        return cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)[1]