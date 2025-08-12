from __future__ import annotations
import numpy as np
from typing import Optional
from .base import BaseModel

def _labels_to_mask_stack(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError("labels must be 2D (H, W)")
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        h, w = labels.shape
        return np.empty((0, h, w), dtype=np.uint16)
    masks = [(labels == i).astype(np.uint16) for i in ids]
    return np.stack(masks, axis=0).astype(np.uint16)

class CellposeSAM(BaseModel):
    """
    Cellpose-SAM (CPSAM) instance segmentation.

    Notes:
      - Uses CellposeModel with pretrained_model='cpsam'.
      - Channels are NOT required/used for CPSAM; it uses the first 3 channels,
        truncating the rest. For grayscale, (H, W) is fine.
      - Diameter is largely invariant for CPSAM (it shouldn’t change which objects are found).
    """

    def __init__(
        self,
        use_gpu: bool = True,
        diameter: Optional[float] = None,          # can be None for CPSAM
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 5,
        tile_overlap: float = 0.1,
        batch_size: int = 8,
        compute_masks: bool = True,
    ):
        super().__init__("Cellpose-SAM")
        self.use_gpu = use_gpu
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self.compute_masks = compute_masks
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from cellpose import models  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cellpose is required. Install with: pip install cellpose"
            ) from e
        # CPSAM is the default pretrained model in recent versions
        self._model = models.CellposeModel(gpu=self.use_gpu, pretrained_model="cpsam")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image : np.ndarray
            2D (H, W) or 3D (H, W, C). CPSAM will use up to the first 3 channels.

        Returns
        -------
        np.ndarray
            (N, H, W) uint16 stack of instance masks (N=0 if none found).
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"image must be 2D or 3D (H,W[,C]); got {image.shape}")

        # Ensure float32 and clip/scale if needed (Cellpose handles normalization internally,
        # but we avoid weird dtypes/ranges)
        img = image.astype(np.float32, copy=False)
        # If values look like 0-255, normalize to 0-1
        vmax = float(img.max()) if img.size else 1.0
        if vmax > 1.0:
            img = img / 255.0 if vmax <= 255.0 else img / vmax

        self._load_model()

        # Call eval; CPSAM ignores channels and uses first 3 channels if present
        masks, flows, styles = self._model.eval(
            img,
            batch_size=self.batch_size,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            min_size=self.min_size,
            tile_overlap=self.tile_overlap,
            compute_masks=self.compute_masks,
        )

        if masks is None:
            h, w = img.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        if masks.dtype != np.uint16:
            masks = masks.astype(np.uint16, copy=False)

        # return _labels_to_mask_stack(masks)
        return masks
