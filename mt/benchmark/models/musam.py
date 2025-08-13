from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from micro_sam import util
from micro_sam.automatic_segmentation import get_predictor_and_segmenter
from micro_sam.automatic_segmentation import InstanceSegmentationWithDecoder

from .base import BaseModel


class MuSAM(BaseModel):
    """
    μSAM (micro_sam) automatic instance segmentation for 2D images.

    """

    def __init__(
        self,
        model_type: str = "vit_b_lm",
        checkpoint: Optional[Union[str, Path]] = None,
        is_tiled: bool = False,
        tile_shape: Optional[Tuple[int, int]] = None,
        halo: Optional[Tuple[int, int]] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("MuSAM")
        self.model_type = model_type
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.is_tiled = is_tiled
        self.tile_shape = tile_shape
        self.halo = halo
        self.device = device
        self.batch_size = int(batch_size)
        self.generate_kwargs = dict(generate_kwargs or {})

        self._predictor = None
        self._segmenter = None



    def _load_model(self):
        if self._predictor is not None and self._segmenter is not None:
            return

        self._predictor, self._segmenter = get_predictor_and_segmenter(
            model_type=self.model_type,
            checkpoint=self.checkpoint,   # if you pass a local checkpoint, it's used directly
            device=self.device,
            amg=False,
            is_tiled=self.is_tiled,
        )

    # -------------------------
    # Core API
    # -------------------------
    def predict(self, image: np.ndarray) -> np.ndarray:
        if image.ndim not in (2, 3):
            raise ValueError(f"MuSAM expects 2D images: (H,W) or (H,W,C); got {image.shape}")

        self._load_model()

        image_data = np.asarray(image)
        ndim = 2

        image_embeddings = util.precompute_image_embeddings(
            predictor=self._predictor,
            input_=image_data,
            ndim=ndim,
            tile_shape=self.tile_shape,
            halo=self.halo,
            verbose=False,
            batch_size=self.batch_size,
        )

        initialize_kwargs: Dict[str, Any] = dict(
            image=image_data,
            image_embeddings=image_embeddings,
            verbose=False,
        )

        generate_kwargs = dict(self.generate_kwargs)

        if isinstance(self._segmenter, InstanceSegmentationWithDecoder) and self.tile_shape is not None:
            generate_kwargs.update({"tile_shape": self.tile_shape, "halo": self.halo})
            initialize_kwargs["batch_size"] = self.batch_size

        self._segmenter.initialize(**initialize_kwargs)
        mask_dicts = self._segmenter.generate(**generate_kwargs)

        masks = self._combine_instance_dicts(mask_dicts)

        return masks.astype(np.uint16, copy=False)

    # -------------------------
    # Helper methods
    # -------------------------
    def _combine_instance_dicts(self, mask_dicts):
        """Convert μSAM instance dicts to a (H, W) uint16 label image."""
        if not mask_dicts:
            return np.zeros((0, 0), dtype=np.uint16)

        # Assume all segmentations are same shape
        h, w = mask_dicts[0]["segmentation"].shape
        label_img = np.zeros((h, w), dtype=np.uint16)

        for i, inst in enumerate(mask_dicts, start=1):
            seg = np.asarray(inst["segmentation"], dtype=bool)
            label_img[seg] = i
        return label_img
