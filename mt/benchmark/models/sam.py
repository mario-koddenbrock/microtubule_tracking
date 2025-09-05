from __future__ import annotations
import numpy as np
from PIL import Image
import torch
from transformers import SamModel, SamProcessor

from .base import BaseModel


class SAM(BaseModel):
    """
    HuggingFace Segment-Anything-Model (SAM) for automatic mask generation.

    Notes:
      - Uses `SamModel` and `SamProcessor` from the `transformers` library.
      - The model automatically finds and masks all objects in an image.
      - Input images are converted to RGB for the model.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-huge",
        use_gpu: bool = True,
    ):
        super().__init__(f"SAM")
        self.device = "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"

        self._model_name = model_name
        self._model: SamModel | None = None
        self._processor: SamProcessor | None = None

    def _load_model(self):
        if self._model is not None:
            return

        self._model = SamModel.from_pretrained(self._model_name).to(self.device)
        self._processor = SamProcessor.from_pretrained(self._model_name)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Generates masks for all objects in the image automatically.

        Parameters
        ----------
        image : np.ndarray
            2D (H, W) or 3D (H, W, C) image.

        Returns
        -------
        np.ndarray
            (N, H, W) uint16 stack of instance masks (N=0 if none found).
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"image must be 2D or 3D (H,W[,C]); got {image.shape}")

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        # SAM expects a PIL image in RGB format.
        pil_image = Image.fromarray(image).convert("RGB")

        # The processor prepares the image for the model.
        # No input_points are provided for automatic mask generation.
        inputs = self._processor(pil_image, return_tensors="pt").to(self.device)

        # Generate masks automatically.
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process the masks to resize them to the original image size.
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        # The output is a list of one tensor (B, N, H, W), we take the first element.
        if not masks or masks[0].shape[1] == 0:
            h, w = image.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        # Squeeze the batch dimension and convert to numpy
        masks = masks[0].squeeze(0).numpy()  # (N, H, W)

        num_masks = masks.shape[0]
        instance_ids = np.arange(1, num_masks + 1, dtype=np.uint16)

        # Use broadcasting to multiply each boolean mask by its ID.
        labeled_masks = masks * instance_ids[:, np.newaxis, np.newaxis]

        return labeled_masks.astype(np.uint16)
