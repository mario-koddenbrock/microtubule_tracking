from __future__ import annotations
import numpy as np
from typing import Optional

from cellpose import models

from .base import BaseModel


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
        diameter: Optional[float] = 15.7, # can be None for CPSAM
        flow_threshold: float = 0.643,
        cellprob_threshold: float = -0.22,
        min_size: int = 22,
        tile_overlap: float = 0.1,
        batch_size: int = 8,
    ):
        super().__init__("Cellpose-SAM")
        self.use_gpu = use_gpu
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.min_size = min_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

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

        normalization_params = {
            "normalize": True,
            "norm3D": False,
            "invert": True,
            "percentile": (2.03, 91.85),
            "sharpen_radius": 0.278,
            "smooth_radius": 0.762,
            "tile_norm_blocksize": 0,
            "tile_norm_smooth3D": 0,
        }

        masks, flows, styles = self._model.eval(
            x=img,
            batch_size=self.batch_size,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            min_size=self.min_size,
            niter=200,
            tile_overlap=self.tile_overlap,
            compute_masks=True,
            max_size_fraction=1.5,
            stitch_threshold=0.0,
            anisotropy=None,
            augment=False,
            bsize=256,
            channel_axis=None,
            do_3D=False,
            flow3D_smooth=0,
            normalize=normalization_params,
            resample=True,
            rescale=None,
            z_axis=None,
        )

        if masks is None:
            h, w = img.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        if masks.dtype != np.uint16:
            masks = masks.astype(np.uint16, copy=False)

        return masks
