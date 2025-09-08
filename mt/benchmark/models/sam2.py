from __future__ import annotations
import numpy as np
from PIL import Image
import torch
from transformers import pipeline

from .base import BaseModel


class SAM2(BaseModel):
    """
    HuggingFace Segment-Anything-Model 2 (SAM2) for automatic mask generation.

    Notes:
      - Uses the `mask-generation` pipeline from the `transformers` library.
      - The model automatically finds and masks all objects in an image.
      - Input images are converted to RGB for the model.

    For parameters, see
    https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/automatic_mask_generator.py#L35
    """

    def __init__(
        self,
        model_name: str = "facebook/sam2.1-hiera-large",  # the largest one
        use_gpu: bool = True,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 0,
    ):
        """
        Initialize SAM2 model with tunable parameters.

        Using a SAM 2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM 2 with a HieraL backbone.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier for SAM2 variants. "facebook/sam2.1-hiera-large" is the
            largest one available.
        use_gpu : bool
            Whether to use GPU acceleration if available
        points_per_batch : int
            Sets the number of points run simultaneously by the model.
            Higher numbers may be faster but use more GPU memory.
            Default: 64
        pred_iou_thresh : float
            A filtering threshold in [0,1], using the model's predicted mask quality.
            Default: 0.8
        stability_score_thresh : float
            A filtering threshold in [0,1], using the stability of the mask under
            changes to the cutoff used to binarize the model's mask predictions.
            Default: 0.95
        min_mask_region_area : int
            If >0, postprocessing will be applied to remove disconnected regions
            and holes in masks with area smaller than min_mask_region_area.
            Requires opencv.
            Default: 0
        """
        super().__init__(f"SAM2")

        # Determine device
        self.device = -1  # CPU default for pipeline
        if use_gpu:
            if torch.cuda.is_available():
                self.device = 0  # Use first GPU
            elif torch.backends.mps.is_available():
                self.device = "mps"

        self._model_name = model_name
        self._points_per_batch = points_per_batch
        self._pred_iou_thresh = pred_iou_thresh
        self._stability_score_thresh = stability_score_thresh
        self._min_mask_region_area = min_mask_region_area
        self._generator = None

    def _load_model(self):
        if self._generator is not None:
            return

        self._generator = pipeline("mask-generation", model=self._model_name, device=self.device)

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
        assert self._generator is not None

        # Convert to PIL image in RGB format
        if image.ndim == 2:
            # Convert grayscale to RGB
            image = np.stack((image,) * 3, axis=-1)

        # Ensure the image is uint8
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Generate masks using the pipeline
        outputs = self._generator(
            pil_image,
            points_per_batch=self._points_per_batch,
            pred_iou_thresh=self._pred_iou_thresh,
            stability_score_thresh=self._stability_score_thresh,
            min_mask_region_area=self._min_mask_region_area,
        )

        # Extract masks from the pipeline output
        if not outputs or "masks" not in outputs or len(outputs["masks"]) == 0:
            h, w = image.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        # Convert masks to numpy arrays and create instance masks
        masks = []
        for mask_data in outputs["masks"]:
            # The pipeline returns PIL Images for masks, convert to numpy
            if isinstance(mask_data, Image.Image):
                mask = np.array(mask_data)
            else:
                mask = mask_data

            # Ensure mask is boolean
            if mask.dtype != bool:
                mask = mask > 0

            masks.append(mask)

        # Stack masks and create instance IDs
        masks = np.array(masks, dtype=bool)  # Shape: (N, H, W)
        num_masks = masks.shape[0]
        instance_ids = np.arange(1, num_masks + 1, dtype=np.uint16)

        # Use broadcasting to multiply each boolean mask by its ID
        labeled_masks = masks * instance_ids[:, np.newaxis, np.newaxis]

        return labeled_masks.astype(np.uint16)
