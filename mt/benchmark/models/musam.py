from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from micro_sam.automatic_segmentation import automatic_instance_segmentation
from micro_sam.automatic_segmentation import get_predictor_and_segmenter
import numpy as np

from .base import BaseModel



class MuSAM(BaseModel):
    """
    μSAM (micro_sam) automatic instance segmentation for 2D images.

    This wrapper uses micro_sam's high-level Python API to run either
    Automatic Mask Generation (AMG) or the micro_sam instance-segmentation
    decoder (AIS) depending on the selected model.

    Notes
    -----
    - `model_type` controls which SAM / μSAM model is used. For light microscopy,
      'vit_b_lm' is a good default; see μSAM docs for all options. Weights are
      downloaded to the μSAM cache automatically on first use. :contentReference[oaicite:1]{index=1}
    - If you pass a `checkpoint` path, that file will be used instead.
    - For RGB input (H,W,3) we explicitly set `ndim=2` to segment per-image (not per-channel). :contentReference[oaicite:2]{index=2}

    Parameters
    ----------
    model_type : str
        μSAM model identifier (e.g. 'vit_b_lm', 'vit_l_em_organelles', 'vit_b'). Default: 'vit_b_lm'.
    checkpoint : Optional[Union[str, Path]]
        Optional path to a custom .pth/.pt checkpoint. If None, μSAM fetches weights as needed.
    amg : Optional[bool]
        Force AMG (True) or AIS (False). By default (None) μSAM auto-selects AIS if available
        for the chosen model, otherwise AMG. :contentReference[oaicite:3]{index=3}
    is_tiled : bool
        Return a tiled segmenter (useful for very large images). Default: False.
    tile_shape : Optional[Tuple[int, int]]
        Tile size used during prediction (passed to μSAM when embeddings are computed).
    halo : Optional[Tuple[int, int]]
        Overlap between tiles.
    device : Optional[str]
        Torch device string (e.g. 'cuda', 'cuda:0', 'cpu'). If None μSAM picks automatically.
    batch_size : int
        Batch size used when computing embeddings across tiles / z-planes. Default: 1.
    generate_kwargs : Optional[Dict[str, Any]]
        Extra kwargs forwarded to μSAM's `generate(...)` under the hood, e.g.
        `pred_iou_thresh`, `stability_score_thresh`, `box_nms_thresh`,
        `crop_nms_thresh`, `min_mask_region_area`, etc. (AMG/AIS specific). :contentReference[oaicite:4]{index=4}
    """

    def __init__(
        self,
        model_type: str = "vit_b_lm",
        checkpoint: Optional[Union[str, Path]] = None,
        amg: Optional[bool] = None,
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
        self.amg = amg
        self.is_tiled = is_tiled
        self.tile_shape = tile_shape
        self.halo = halo
        self.device = device
        self.batch_size = int(batch_size)
        self.generate_kwargs = dict(generate_kwargs or {})

        self._predictor = None
        self._segmenter = None

    # -------------------------
    # Lazy loader
    # -------------------------
    def _load_model(self):
        if self._predictor is not None and self._segmenter is not None:
            return

        # Let μSAM choose AIS vs AMG unless the user forces it.
        self._predictor, self._segmenter = get_predictor_and_segmenter(
            model_type=self.model_type,
            checkpoint=self.checkpoint,
            device=self.device,
            amg=self.amg,
            is_tiled=self.is_tiled,
        )
        # (μSAM downloads weights automatically to its cache if needed.) :contentReference[oaicite:5]{index=5}

    # -------------------------
    # Core API
    # -------------------------
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Segment a 2D image and return (N, H, W) uint16 instance masks.

        Accepts grayscale (H,W) or RGB/multi-channel (H,W,C). For RGB, we run 2D segmentation
        (ndim=2) as recommended by μSAM. :contentReference[oaicite:6]{index=6}
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"MuSAM expects 2D images: (H,W) or (H,W,C); got {image.shape}")

        self._load_model()


        # μSAM handles dtype conversion internally; ensure a contiguous array.
        img = np.asarray(image)

        # Important: for RGB inputs, μSAM requires ndim=2 to treat it as a 2D image with channels. :contentReference[oaicite:7]{index=7}
        ndim = 2

        # Run μSAM's high-level automatic instance segmentation on the in-memory array.
        instances = automatic_instance_segmentation(
            predictor=self._predictor,
            segmenter=self._segmenter,
            input_path=img,                 # it accepts ndarray directly
            embedding_path=None,            # no on-disk cache by default
            ndim=ndim,
            tile_shape=self.tile_shape,
            halo=self.halo,
            verbose=False,
            batch_size=self.batch_size,
            **self.generate_kwargs,
        )
        # μSAM returns a 2D label image (uint32) with background=0. Convert to (N,H,W). :contentReference[oaicite:8]{index=8}
        instances = np.asarray(instances)
        if instances.ndim != 2:
            raise RuntimeError(f"μSAM returned unexpected shape {instances.shape}; expected 2D label image.")

        return instances.astype(np.uint16, copy=False)
