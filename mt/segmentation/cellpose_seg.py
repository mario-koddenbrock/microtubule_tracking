import logging
from typing import Optional

import numpy as np
from cellpose_adapt.config.pipeline_config import PipelineConfig
from cellpose_adapt.core import CellposeRunner, initialize_model
from cellpose_adapt.utils import get_device

from .base import BaseSegmentationModel

logger = logging.getLogger(f"mt.{__name__}")


class CellposePredictor(BaseSegmentationModel):
    """
    A concrete implementation of BaseSegmentationModel for Cellpose.

    This class handles the loading of a Cellpose model and running inference.
    """

    def __init__(self, config_path: str, device_str: Optional[str] = None):
        """
        Initializes the CellposePredictor.

        It calls the parent constructor, which in turn calls the `_load_model`
        method to perform the Cellpose-specific setup.
        """
        self.runner: Optional[CellposeRunner] = None  # Will be initialized in _load_model
        logger.debug(f"Calling super().__init__ for CellposePredictor with config_path: {config_path}.")
        super().__init__(config_path, device_str)
        logger.info("CellposePredictor instance created and model loaded.")

    def _load_model(self) -> None:
        """
        Implements the model loading logic for Cellpose.

        This method loads the JSON config, determines the device, and
        instantiates the `CellposeRunner`, which handles the model itself.
        """
        logger.info("Loading Cellpose model and configuration (Cellpose-specific steps)...")

        try:
            # 1. Load pipeline configuration
            logger.debug(f"Loading PipelineConfig from: {self.config_path}")
            self.config = PipelineConfig.from_json(self.config_path)
            logger.debug("PipelineConfig loaded successfully.")

            # 2. Get device
            logger.debug(f"Determining device (CLI specified: {self.device_str}).")
            self.device = get_device(cli_device=self.device_str)
            logger.info(f"Using device for Cellpose: {self.device}")

            # 3. Initialize the CellposeRunner
            # The runner encapsulates the model and other settings
            logger.debug(f"Initializing Cellpose model '{self.config.model_name}' on device '{self.device}'.")
            self.model = initialize_model(self.config.model_name, device=self.device)

            logger.debug(f"Instantiating CellposeRunner with model and config.")
            self.runner = CellposeRunner(model=self.model, config=self.config, device=self.device)
            logger.info("Cellpose model and runner initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Cellpose model or configuration: {e}", exc_info=True)
            # Re-raise the exception to stop initialization as this is a critical failure
            raise

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Runs Cellpose segmentation on a single image.

        Args:
            image (np.ndarray): The input image (H, W) or (H, W, C).

        Returns:
            np.ndarray: A 2D integer array of the predicted segmentation mask.
                        Returns a zero-mask if no objects are found.
        """
        logger.debug(f"CellposePredictor: Running prediction for image of shape {image.shape}.")

        # BaseSegmentationModel's predict method already has a check for self.is_initialized
        # This explicit check here is redundant if super().predict is called first.
        # super().predict(image) # If you want to use the base class's checks
        if not self.is_initialized or self.runner is None:
            msg = "CellposePredictor is not initialized. Cannot run prediction."
            logger.critical(msg)
            raise RuntimeError(msg)
        if not isinstance(image, np.ndarray):
            msg = f"Input image must be a numpy array, but got {type(image)}."
            logger.error(msg)
            raise TypeError(msg)

        pred_mask: Optional[np.ndarray] = None
        try:
            # Use the pre-initialized runner to perform the prediction
            # The runner might return (mask, flows, styles, diams)
            pred_mask, _ = self.runner.run(image)
            logger.debug(
                f"Cellpose runner returned prediction. Raw mask shape: {pred_mask.shape if pred_mask is not None else 'None'}.")
        except Exception as e:
            logger.error(f"An error occurred during Cellpose prediction: {e}", exc_info=True)
            # Fallback to an empty mask on error
            h, w = image.shape[0], image.shape[1]
            pred_mask = np.zeros((h, w), dtype=np.uint16)
            logger.warning(f"Prediction failed, returning empty mask of shape {pred_mask.shape}.")
            return pred_mask  # Return early as prediction itself failed

        # Handle the case where no cells are detected or Cellpose returns None
        if pred_mask is None or pred_mask.size == 0 or np.all(pred_mask == 0):
            if pred_mask is None:
                logger.warning("Cellpose returned None for mask. Creating an empty mask.")
            elif pred_mask.size == 0:
                logger.warning("Cellpose returned an empty mask array. Creating a correctly-sized empty mask.")
            elif np.all(pred_mask == 0):
                logger.info("Cellpose detected no objects (mask is all zeros).")

            h, w = image.shape[0], image.shape[1]
            pred_mask = np.zeros((h, w), dtype=np.uint16)
            logger.debug(f"Returning zero-mask of shape {pred_mask.shape}.")
        else:
            logger.debug(
                f"Prediction successful. Mask shape: {pred_mask.shape}, unique IDs: {np.unique(pred_mask).tolist()}.")

        return pred_mask