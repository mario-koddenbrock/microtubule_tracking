from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D, StarDist3D

from .base import BaseModel


class StarDist(BaseModel):
    """
    Generic StarDist wrapper for 2D and 3D models.
    Returns (N,H,W) uint16 masks.
    For 3D models, it can handle 2D images by treating them as 3D images with a single z-slice.

    Subclasses can override `_prepare_model_dir()` to dynamically provide a
    local folder containing a StarDist model (with config/weights/thresholds).
    If `_prepare_model_dir()` returns None, this class will fall back to
    `StarDist2D.from_pretrained(pretrained)` or `StarDist3D.from_pretrained(pretrained)`.

    Parameters
    ----------
    model_name : str
        Name for logging/identification (BaseModel).
    pretrained : Optional[str]
        Built-in StarDist model name (e.g. '2D_versatile_fluo', '3D_versatile_fluo').
        Ignored if `model_dir` (or subclass `_prepare_model_dir`) provides a folder.
    model_dir : Optional[Union[str, Path]]
        Path to a folder holding a trained StarDist model (contains config/weights/thresholds).
    prob_thresh : Optional[float]
        Probability threshold for predict_instances (None -> use model default).
    nms_thresh : Optional[float]
        NMS threshold for predict_instances (None -> use model default).
    normalize : bool
        If True, percentile-normalize before prediction.
    norm_percentiles : Tuple[float, float]
        (low, high) percentiles for normalization.
    channel : Optional[int]
        If input is (H,W,C), pick this channel (default: 0).
    n_tiles : Optional[Union[int, Tuple[int, ...]]]
        Tiling for large images.
    """

    def __init__(
        self,
        model_name: str = "StarDist",
        *,
        pretrained: Optional[str] = "2D_versatile_fluo",
        model_dir: Optional[Union[str, Path]] = None,
        prob_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        normalize: bool = True,
        norm_percentiles: Tuple[float, float] = (1.0, 99.8),
        channel: Optional[int] = None,
        n_tiles: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        super().__init__(model_name)
        self.pretrained = pretrained
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.normalize = normalize
        self.norm_percentiles = norm_percentiles
        self.channel = channel
        self.n_tiles = n_tiles

        self._model: Optional[Union[StarDist2D, StarDist3D]] = None  # lazy-loaded
        self._is_3d: Optional[bool] = None

    # -------------------------
    # Hooks for subclasses
    # -------------------------
    def _prepare_model_dir(self) -> Optional[Path]:
        """
        Return a directory that contains a StarDist model (with config/weights/thresholds),
        or None to use `pretrained` instead. Subclasses can override to implement
        on-demand downloads or custom discovery.

        Base implementation: return `self.model_dir` if provided; else None.
        """
        return self.model_dir

    # -------------------------
    # Internals
    # -------------------------
    def _load_model(self) -> None:
        if self._model is not None:
            return

        model_dir = self._prepare_model_dir()
        if model_dir is not None:
            model_dir = Path(model_dir).resolve()
            if not model_dir.exists():
                raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

            config_path = model_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"config.json not found in model_dir: {model_dir}")

            with open(config_path) as f:
                config = json.load(f)

            n_dim = config.get("n_dim")
            if n_dim not in [2, 3]:
                raise ValueError(f"Unsupported n_dim in config.json: {n_dim}")
            self._is_3d = n_dim == 3

            basedir = str(model_dir.parent)
            name = model_dir.name
            ModelClass = StarDist3D if self._is_3d else StarDist2D
            self._model = ModelClass(None, name=name, basedir=basedir)
        else:
            if self.pretrained is None:
                raise ValueError("Either provide 'pretrained' or a 'model_dir'.")

            self._is_3d = "3D_" in self.pretrained
            ModelClass = StarDist3D if self._is_3d else StarDist2D
            try:
                self._model = ModelClass.from_pretrained(self.pretrained)
            except Exception as e:
                # Provide a more informative error if the pretrained model is not found
                available_models = [m["name"] for m in ModelClass.get_pretained_models_list()]
                raise ValueError(
                    f"Pretrained model '{self.pretrained}' not found for {ModelClass.__name__}. "
                    f"Available models: {available_models}"
                ) from e

        # adopt stored thresholds if user didn't override
        thr = getattr(self._model, "thresholds", None)
        if self.prob_thresh is None and getattr(thr, "prob", None) is not None:
            self.prob_thresh = float(thr.prob)
        if self.nms_thresh is None and getattr(thr, "nms", None) is not None:
            self.nms_thresh = float(thr.nms)

    # -------------------------
    # Public API
    # -------------------------
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict instances and return (N,H,W) uint16 masks.
        Accepts (H,W) or (H,W,C).
        If the model is 3D, it treats 2D images as a single-slice 3D volume.
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"{self.model_name} expects (H,W) or (H,W,C); got {image.shape}")

        self._load_model()
        assert self._model is not None
        assert self._is_3d is not None

        img = np.asarray(image)
        if img.dtype.kind in "ui":
            img = img.astype(np.float32, copy=False)

        # --- Input image handling ---
        axes = None
        if not self._is_3d:
            # Handle different pretrained 2D models
            if self.pretrained == "2D_versatile_he":
                # This model expects a color image (Y,X,C).
                if img.ndim == 2:
                    # If grayscale, convert to RGB for this model.
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                axes = "YXC"
            else:
                # Other 2D models expect grayscale (Y,X).
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                axes = "YX"
        else:  # 3D model
            if img.ndim == 3 and img.shape[-1] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            axes = "YXZ"

        is_2d_input_for_3d_model = self._is_3d and img.ndim == 2
        if is_2d_input_for_3d_model:
            img = img[..., np.newaxis]  # (H, W) -> (H, W, 1)

        # Normalize if requested
        img_in = normalize(img, *self.norm_percentiles, axis=tuple(range(img.ndim))) if self.normalize else img


        labels, _details = self._model.predict_instances(
            img_in,
            axes=axes,
            prob_thresh=self.prob_thresh,
            nms_thresh=self.nms_thresh,
            n_tiles=self.n_tiles,
            verbose=False,
        )

        if labels is None:
            h, w = image.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        # Squeeze out the z-axis if it was added for a 3D model on a 2D input
        if is_2d_input_for_3d_model:
            labels = labels.squeeze(axis=0) # (H, W, 1) -> (H, W)

        return labels
