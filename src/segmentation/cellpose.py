import numpy as np
from typing import Optional

from cellpose_adapt.config.pipeline_config import PipelineConfig
from cellpose_adapt.core import CellposeRunner, initialize_model
from cellpose_adapt.utils import get_device

from .base_model import BaseSegmentationModel


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
        # The runner will be initialized in _load_model
        self.runner: Optional[CellposeRunner] = None
        super().__init__(config_path, device_str)

    def _load_model(self) -> None:
        """
        Implements the model loading logic for Cellpose.

        This method loads the JSON config, determines the device, and
        instantiates the `CellposeRunner`, which handles the model itself.
        """
        print("Loading Cellpose model and configuration...")

        # 1. Load pipeline configuration
        self.config = PipelineConfig.from_json(self.config_path)

        # 2. Get device
        self.device = get_device(cli_device=self.device_str)
        print(f"Using device: {self.device}")

        # 3. Initialize the CellposeRunner
        # The runner encapsulates the model and other settings
        self.model = initialize_model(self.config.model_name, device=self.device)
        self.runner = CellposeRunner(model=self.model, config=self.config, device=self.device)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Runs Cellpose segmentation on a single image.

        Args:
            image (np.ndarray): The input image (H, W) or (H, W, C).

        Returns:
            np.ndarray: A 2D integer array of the predicted segmentation mask.
                        Returns a zero-mask if no objects are found.
        """
        if not self.is_initialized or self.runner is None:
            raise RuntimeError("Predictor is not initialized. Cannot run prediction.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        # Use the pre-initialized runner to perform the prediction
        pred_mask, _ = self.runner.run(image)

        # Handle the case where no cells are detected
        if pred_mask is None:
            print("Warning: Cellpose returned no mask. Creating an empty mask.")
            h, w = image.shape[0], image.shape[1]
            pred_mask = np.zeros((h, w), dtype=np.uint16)

        return pred_mask