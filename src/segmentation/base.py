import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np


logger = logging.getLogger(f"mt.{__name__}")


class BaseSegmentationModel(ABC):
    """
    An abstract base class for segmentation models.

    This class defines a common interface for initializing a model and running
    predictions. Child classes must implement the `_load_model` and `predict`
    methods.

    The primary benefit of this structure is to ensure that any segmentation
    model (Cellpose, StarDist, etc.) can be used interchangeably in a larger
    application pipeline.
    """

    def __init__(self, config_path: str, device_str: Optional[str] = None):
        """
        Initializes the base model.

        This constructor stores common configuration and triggers the model
        loading process by calling the abstract `_load_model` method, which
        must be implemented by the child class.

        Args:
            config_path (str): Path to the model-specific configuration file.
            device_str (Optional[str]): A string to specify the device (e.g., 'cuda',
                                      'cpu'). If None, the implementation should
                                      handle auto-detection.
        """
        logger.info(f"Initializing BaseSegmentationModel instance of type '{self.__class__.__name__}'...")
        logger.debug(f"Provided config_path: {config_path}, device_str: {device_str}.")

        if not os.path.exists(config_path):
            logger.critical(f"Configuration file not found at: {config_path}. Cannot initialize model.")
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        self.config_path = config_path
        self.device_str = device_str

        # Placeholders for attributes that the child class will populate
        self.model: Optional[Any] = None
        self.config: Optional[Any] = None
        self.device: Optional[Any] = None
        self.is_initialized = False

        # Delegate the heavy, specific loading to the child class
        try:
            logger.debug(f"Calling _load_model() for '{self.__class__.__name__}'.")
            self._load_model()
            self.is_initialized = True
            logger.info(f"{self.__class__.__name__} initialized successfully.")
        except NotImplementedError:
            logger.error(
                f"'_load_model' method is not implemented in '{self.__class__.__name__}'. This is an abstract method and must be overridden.",
                exc_info=False)
            raise  # Re-raise if the abstract method isn't implemented
        except Exception as e:
            self.is_initialized = False  # Mark as not initialized on failure
            logger.critical(f"Failed to initialize '{self.__class__.__name__}' model: {e}", exc_info=True)
            raise  # Re-raise to indicate a critical setup failure

    @abstractmethod
    def _load_model(self) -> None:
        """
        Abstract method for loading the model and its configuration.

        This method should handle all the expensive, one-time setup operations:
        - Loading the configuration file.
        - Determining the compute device.
        - Loading the model weights into memory.
        - Populating `self.model`, `self.config`, and `self.device`.
        """
        # Child classes should log their specific loading steps (e.g., loading config, model weights).
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Abstract method for running segmentation on a single image.

        Args:
            image (np.ndarray): The input image to be segmented.

        Returns:
            np.ndarray: The resulting integer label mask.
        """
        if not self.is_initialized:
            logger.error(f"Cannot call predict on '{self.__class__.__name__}': Model is not initialized.")
            raise RuntimeError("Model must be initialized before prediction.")

        # Child classes should log details about prediction (e.g., input shape, output shape).
        pass