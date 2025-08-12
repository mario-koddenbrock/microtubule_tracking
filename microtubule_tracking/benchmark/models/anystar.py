# src/benchmark/models/anystar.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path

from csbdeep.utils import normalize
from stardist.models import StarDist2D

from .base import BaseModel


def _labels_to_mask_stack(labels: np.ndarray) -> np.ndarray:
    """Convert a 2D labeled mask (H,W) to (N,H,W) uint16 stack (background=0 ignored)."""
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D (H,W), got {labels.shape}")
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        h, w = labels.shape
        return np.empty((0, h, w), dtype=np.uint16)
    masks = [(labels == i).astype(np.uint16) for i in ids]
    return np.stack(masks, axis=0).astype(np.uint16)


class AnyStar(BaseModel):
    """
    AnyStar (StarDist-based) instance segmentation for 2D images.

    This wrapper loads a StarDist2D model (either a built-in pretrained name like
    '2D_versatile_fluo' or a custom model directory that contains StarDist weights)
    and returns a stack of instance masks with shape (N, H, W), dtype uint16.

    Parameters
    ----------
    pretrained : Optional[str]
        Name for StarDist2D.from_pretrained(...) (e.g. '2D_versatile_fluo', '2D_versatile_he').
        Ignored if `model_dir` is provided. Default: '2D_versatile_fluo'.
    model_dir : Optional[Union[str, Path]]
        Path to a folder holding a trained StarDist2D model (contains config/weights/thresholds).
        If provided, this is used instead of a built-in pretrained model.
        Example: '/path/to/models/anystar-mix' where that folder contains the files.
    prob_thresh : Optional[float]
        Detection probability threshold for `predict_instances`. If None, use model default.
    nms_thresh : Optional[float]
        Non-maximum suppression threshold for `predict_instances`. If None, use model default.
    normalize : bool
        If True, percentile-normalize the image before prediction.
    norm_percentiles : Tuple[float, float]
        (p_low, p_high) percentiles for normalization (applied per-channel if present).
    channel : Optional[int]
        If the model expects 1 channel but the input is multi-channel (H,W,C), pick this channel.
        If None, the first channel is used.
    n_tiles : Optional[Union[int, Tuple[int, int]]]
        Tiling for large images. StarDist accepts None, an int, or a tuple of tile counts (y, x).
    """

    def __init__(
        self,
        pretrained: Optional[str] = "2D_versatile_fluo",
        model_dir: Optional[Union[str, Path]] = None,
        prob_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        normalize: bool = True,
        norm_percentiles: Tuple[float, float] = (1.0, 99.8),
        channel: Optional[int] = None,
        n_tiles: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        super().__init__("AnyStar")
        self.pretrained = pretrained
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.normalize = normalize
        self.norm_percentiles = norm_percentiles
        self.channel = channel
        self.n_tiles = n_tiles

        self._model = None  # lazy-loaded StarDist2D
        self._model_n_channels = None  # will be set from model.config.n_channel_in

    # -------------------------
    # Lazy loader
    # -------------------------
    def _load_model(self):
        if self._model is not None:
            return

        if self.model_dir is not None:
            # StarDist expects (basedir, name) where basedir contains a subfolder 'name'
            # that holds config/weights/thresholds. If user passed the model directory itself,
            # we split it into parent as basedir and leaf as name.
            model_dir = self.model_dir.resolve()
            if not model_dir.exists():
                raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
            basedir = str(model_dir.parent)
            name = model_dir.name
            self._model = StarDist2D(None, name=name, basedir=basedir)
        else:
            # Built-in pretrained (e.g. '2D_versatile_fluo', '2D_versatile_he', etc.)
            if self.pretrained is None:
                raise ValueError("Either 'pretrained' must be set or 'model_dir' must be provided.")
            self._model = StarDist2D.from_pretrained(self.pretrained)

        # track number of input channels expected by the model
        self._model_n_channels = int(getattr(self._model.config, "n_channel_in", 1))

        # If user didn't override thresholds, use the model's stored defaults
        if self.prob_thresh is None:
            self.prob_thresh = float(getattr(self._model, "thresholds", {}).get("prob", 0.5))
        if self.nms_thresh is None:
            self.nms_thresh = float(getattr(self._model, "thresholds", {}).get("nms", 0.3))

    # -------------------------
    # Core API
    # -------------------------
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run StarDist2D and return (N, H, W) uint16 instance masks.
        Accepts grayscale (H,W) or multi-channel (H,W,C) images.
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"AnyStar expects 2D images: (H,W) or (H,W,C); got {image.shape}")

        self._load_model()

        img = np.asarray(image)
        if img.dtype.kind in "ui":  # integers
            # safe convert to float for normalization
            img = img.astype(np.float32, copy=False)

        # Handle channels vs model expectation
        if img.ndim == 3:
            H, W, C = img.shape
            if self._model_n_channels == 1:
                c = 0 if self.channel is None else int(self.channel)
                if not (0 <= c < C):
                    raise ValueError(f"Requested channel {c} but image has {C} channels.")
                img_in = img[..., c]
            else:
                # If model expects >1 channels, pass as-is (assumes correct order/size)
                img_in = img
        else:
            img_in = img  # (H,W)

        # Optional percentile normalization (per-channel if needed)
        if self.normalize:
            p1, p2 = self.norm_percentiles
            axis = (0, 1) if img_in.ndim == 2 else (0, 1)  # per-channel along H,W
            img_in = normalize(img_in, p1, p2, axis=axis)

        # Predict instances
        labels, _details = self._model.predict_instances(
            img_in,
            prob_thresh=float(self.prob_thresh),
            nms_thresh=float(self.nms_thresh),
            n_tiles=self.n_tiles,
        )

        if labels is None:
            h, w = img.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        # Convert label image to (N,H,W) stack
        return _labels_to_mask_stack(labels)
