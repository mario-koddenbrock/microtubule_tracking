from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D

from .base import BaseModel



class StarDist(BaseModel):
    """
    Generic 2D StarDist wrapper that returns (N,H,W) uint16 masks.

    Subclasses can override `_prepare_model_dir()` to dynamically provide a
    local folder containing a StarDist model (with config/weights/thresholds).
    If `_prepare_model_dir()` returns None, this class will fall back to
    `StarDist2D.from_pretrained(pretrained)`.

    Parameters
    ----------
    model_name : str
        Name for logging/identification (BaseModel).
    pretrained : Optional[str]
        Built-in StarDist 2D model name (e.g. '2D_versatile_fluo').
        Ignored if `model_dir` (or subclass `_prepare_model_dir`) provides a folder.
    model_dir : Optional[Union[str, Path]]
        Path to a folder holding a trained StarDist2D model (contains config/weights/thresholds).
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
    n_tiles : Optional[Union[int, Tuple[int, int]]]
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
        n_tiles: Optional[Union[int, Tuple[int, int]]] = None,
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

        self._model: Optional[StarDist2D] = None  # lazy-loaded

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
    @staticmethod
    def _ensure_is_2d_model_dir(model_dir: Path) -> None:
        """
        Sanity-check that the model in `model_dir` is a 2D StarDist model.
        (Raises if config.json is missing or indicates 3D.)
        """
        cfg_path = model_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json in model directory: {model_dir}")

        # Minimal, dependency-free parse
        import json
        with cfg_path.open("r") as f:
            cfg = json.load(f)

        # 2D StarDist configs typically have grid of length 2
        grid = cfg.get("grid", None)
        if isinstance(grid, (list, tuple)) and len(grid) != 2:
            raise ValueError(
                f"Non-2D StarDist model detected at {model_dir} (grid={grid}). "
                f"This wrapper supports 2D only."
            )
        # Some configs may have n_dim
        n_dim = cfg.get("n_dim", 2)
        if int(n_dim) != 2:
            raise ValueError(
                f"Non-2D StarDist model detected at {model_dir} (n_dim={n_dim}). "
                f"This wrapper supports 2D only."
            )

    def _load_model(self) -> None:
        if self._model is not None:
            return

        model_dir = self._prepare_model_dir()
        if model_dir is not None:
            model_dir = Path(model_dir).resolve()
            if not model_dir.exists():
                raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
            self._ensure_is_2d_model_dir(model_dir)

            basedir = str(model_dir.parent)
            name = model_dir.name
            self._model = StarDist2D(None, name=name, basedir=basedir)
        else:
            if self.pretrained is None:
                raise ValueError("Either provide 'pretrained' or a 'model_dir'.")
            # built-in 2D weights
            self._model = StarDist2D.from_pretrained(self.pretrained)

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
        Predict 2D instances and return (N,H,W) uint16 masks.
        Accepts (H,W) or (H,W,C).
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"{self.model_name} expects (H,W) or (H,W,C); got {image.shape}")

        self._load_model()

        img = np.asarray(image)
        if img.dtype.kind in "ui":
            img = img.astype(np.float32, copy=False)

        # (H,W,C) -> select single channel
        if img.ndim == 3:
            ch = 0 if self.channel is None else int(self.channel)
            if not (0 <= ch < img.shape[2]):
                raise ValueError(f"Requested channel {ch} outside image channels [0..{img.shape[2] - 1}]")
            img = img[..., ch]

        # Ensure strictly 2D contiguous float32 array
        img = np.ascontiguousarray(img.squeeze(), dtype=np.float32)

        # Normalize if requested
        img_in = normalize(img, *self.norm_percentiles, axis=(0, 1)) if self.normalize else img

        labels, _details = self._model.predict_instances(
            img_in,
            prob_thresh=self.prob_thresh,
            nms_thresh=self.nms_thresh,
            n_tiles=self.n_tiles,
        )

        if labels is None:
            h, w = img.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        return labels
