from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from micro_sam import util
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, InstanceSegmentationWithDecoder

from .base import BaseModel


class MicroSAM(BaseModel):
    """μSAM automatic instance segmentation (AIS) for 2D images, tuned for microtubules."""

    def __init__(
        self,
        model_type: str = "vit_l_lm",
        checkpoint: Optional[Union[str, Path]] = None,
        tile_shape: Tuple[int, int] = (512, 512),
        halo: Tuple[int, int] = (32, 32),
        device: Optional[str] = None,
        batch_size: int = 1,
        model_dir: Union[str, Path] = "models/muSAM",
    ):
        super().__init__("MicroSAM")
        self.model_type = model_type
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.tile_shape = tile_shape
        self.halo = halo
        self.device = device
        self.batch_size = int(batch_size)

        # Always use tiled AIS if tile_shape is given
        self.is_tiled = True

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
            amg=False,            # AIS mode only
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
            tile_shape=self.tile_shape,
            halo=self.halo,
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
        #         Args:
        #             center_distance_threshold: Center distance predictions below this value will be
        #                 used to find seeds (intersected with thresholded boundary distance predictions).
        #                 By default, set to '0.5'.
        #             boundary_distance_threshold: Boundary distance predictions below this value will be
        #                 used to find seeds (intersected with thresholded center distance predictions).
        #                 By default, set to '0.5'.
        #             foreground_threshold: Foreground predictions above this value will be used as foreground mask.
        #                 By default, set to '0.5'.
        #             foreground_smoothing: Sigma value for smoothing the foreground predictions, to avoid
        #                 checkerboard artifacts in the prediction. By default, set to '1.0'.
        #             distance_smoothing: Sigma value for smoothing the distance predictions.
        #             min_size: Minimal object size in the segmentation result. By default, set to '0'.
        #             output_mode: The form masks are returned in. Pass None to directly return the instance segmentation.
        #                 By default, set to 'binary_mask'.
        #             tile_shape: Tile shape for parallelizing the instance segmentation post-processing.
        #                 This parameter is independent from the tile shape for computing the embeddings.
        #                 If not given then post-processing will not be parallelized.
        #             halo: Halo for parallel post-processing. See also `tile_shape`.
        #             n_threads: Number of threads for parallel post-processing. See also `tile_shape`.
        masks = self._segmenter.generate(
            tile_shape=self.tile_shape,
            halo=self.halo,
            min_size=5,
            center_distance_threshold = 0.05,
            boundary_distance_threshold = 0.05,
            foreground_threshold = 0.005,
            foreground_smoothing = 1.0,
            distance_smoothing = 1.0,
            output_mode=None,
            n_threads = 1,
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
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], dict) and "segmentation" in result[0]:
            h, w = result[0]["segmentation"].shape
            lab = np.zeros((h, w), dtype=np.uint16)
            for i, inst in enumerate(result, start=1):
                seg = np.asarray(inst["segmentation"], dtype=bool)
                lab[seg] = i
            return lab

        raise RuntimeError(f"Unexpected AIS output format: {type(result)}")

