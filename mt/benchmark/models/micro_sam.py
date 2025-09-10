from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from micro_sam import util
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    InstanceSegmentationWithDecoder,
)

from .base import BaseModel


class MicroSAM(BaseModel):
    """
    MicroSAM (μSAM) automatic instance segmentation for microscopy images.

    Notes:
      - Uses the μSAM library for automatic instance segmentation (AIS).
      - Specialized for microscopy data with decoder-based segmentation.
      - Since our images are only 512x512, we are not using any tiling.
    """

    def __init__(
        self,
        model_type: str = "vit_l_lm",
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        model_dir: Union[str, Path] = "models/muSAM",
        center_distance_threshold: float = 0.5,
        boundary_distance_threshold: float = 0.5,
        foreground_threshold: float = 0.5,
        foreground_smoothing: float = 1.0,
        distance_smoothing: float = 1.6,
        min_size: int = 0,
    ):
        """
        Initialize μSAM model with tunable segmentation parameters.

        Parameters
        ----------
        model_type : str
            μSAM model type. Available options include "vit_l_lm" (default), "vit_b_lm", etc.
        checkpoint : Optional[Union[str, Path]]
            Path to custom checkpoint. If None, uses pretrained weights.
        device : Optional[str]
            Device to use ("cpu", "cuda", "mps"). If None, auto-detects.
        batch_size : int
            Batch size for processing. Default: 1
        model_dir : Union[str, Path]
            Directory to cache model weights. Default: "models/muSAM"
        center_distance_threshold : float
            Center distance predictions below this value will be used to find seeds
            (intersected with thresholded boundary distance predictions).
            Default: 0.5
        boundary_distance_threshold : float
            Boundary distance predictions below this value will be used to find seeds
            (intersected with thresholded center distance predictions).
            Default: 0.5
        foreground_threshold : float
            Foreground predictions above this value will be used as foreground mask.
            Default: 0.5
        foreground_smoothing : float
            Sigma value for smoothing the foreground predictions, to avoid
            checkerboard artifacts in the prediction.
            Default: 1.0
        distance_smoothing : float
            Sigma value for smoothing the distance predictions.
            Default: 1.6
        min_size : int
            Minimal object size in the segmentation result.
            Default: 0
        """
        super().__init__("MicroSAM")
        self.model_type = model_type
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.device = device
        self.batch_size = int(batch_size)

        # Segmentation parameters
        self.center_distance_threshold = center_distance_threshold
        self.boundary_distance_threshold = boundary_distance_threshold
        self.foreground_threshold = foreground_threshold
        self.foreground_smoothing = foreground_smoothing
        self.distance_smoothing = distance_smoothing
        self.min_size = min_size

        # Use non-tiled AIS for 512x512 images (InstanceSegmentationWithDecoder)
        # Note: For 512x512 images with tile_shape=(512,512), no actual tiling occurs,
        # so we use is_tiled=False to get the simpler InstanceSegmentationWithDecoder
        self.is_tiled = False

        # Set μSAM cache dir for weights
        cache_root = Path(model_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MICROSAM_CACHEDIR", str(cache_root))

        self._predictor = None
        self._segmenter = None

    def _load_model(self):
        if self._predictor is not None and self._segmenter is not None:
            return

        self._predictor, self._segmenter = get_predictor_and_segmenter(
            model_type=self.model_type,
            checkpoint=self.checkpoint,
            device=self.device,
            amg=False,  # AIS mode only
            is_tiled=self.is_tiled,
        )
        if not isinstance(self._segmenter, InstanceSegmentationWithDecoder):
            raise RuntimeError("Loaded segmenter is not AIS (InstanceSegmentationWithDecoder)")

    def predict(self, image: np.ndarray) -> np.ndarray:
        if image.ndim not in (2, 3):
            raise ValueError(f"MicroSAM expects (H,W) or (H,W,C); got {image.shape}")

        self._load_model()

        img = np.asarray(image)
        ndim = 2  # RGB/multi-channel treated as 2D image

        # 1) Compute embeddings
        embeddings = util.precompute_image_embeddings(
            predictor=self._predictor,
            input_=img,
            save_path=None,
            ndim=ndim,
            verbose=False,
            batch_size=self.batch_size,
        )

        # 2) Initialize segmenter
        self._segmenter.initialize(
            image=img,
            image_embeddings=embeddings,
            verbose=False,
        )

        # 3) Generate prediction
        masks = self._segmenter.generate(
            min_size=self.min_size,
            center_distance_threshold=self.center_distance_threshold,
            boundary_distance_threshold=self.boundary_distance_threshold,
            foreground_threshold=self.foreground_threshold,
            foreground_smoothing=self.foreground_smoothing,
            distance_smoothing=self.distance_smoothing,
            output_mode=None,
            n_threads=1,
        )

        # 4) Convert to label image if needed
        return masks

    @staticmethod
    def _to_label_image(result) -> np.ndarray:

        # Case 1: already a label image
        if isinstance(result, np.ndarray) and result.ndim == 2:
            return result

        # Case 2: list with one label image
        if isinstance(result, (list, tuple)) and len(result) == 1:
            first = result[0]
            if isinstance(first, np.ndarray) and first.ndim == 2:
                return first
            if isinstance(first, dict) and "segmentation" in first:
                h, w = first["segmentation"].shape
                lab = np.zeros((h, w), dtype=np.uint16)
                seg = np.asarray(first["segmentation"], dtype=bool)
                lab[seg] = 1
                return lab

        # Case 3: list of dicts (one per instance)
        if (
            isinstance(result, (list, tuple))
            and result
            and isinstance(result[0], dict)
            and "segmentation" in result[0]
        ):
            h, w = result[0]["segmentation"].shape
            lab = np.zeros((h, w), dtype=np.uint16)
            for i, inst in enumerate(result, start=1):
                seg = np.asarray(inst["segmentation"], dtype=bool)
                lab[seg] = i
            return lab

        raise RuntimeError(f"Unexpected AIS output format: {type(result)}")
