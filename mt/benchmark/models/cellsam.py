# src/benchmark/models/cellsam.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import os
import numpy as np

from dotenv import load_dotenv, set_key  # pip install python-dotenv
from .base import BaseModel
from cellSAM import cellsam_pipeline


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


class CellSAM(BaseModel):
    """
    Wrapper for the CellSAM inference pipeline (2D only) with .env-based token storage.

    Parameters
    ----------
    access_token : Optional[str]
        If provided, set DEEPCELL_ACCESS_TOKEN and store in .env (if persist_token=True).
    env_path : Optional[Union[str, Path]]
        Path to your .env file. Default: repo root / .env.
    persist_token : bool
        Whether to store the provided token into the .env file. Default: True.
    """

    def __init__(
        self,
        *,
        access_token: Optional[str] = None,
        env_path: Optional[Union[str, Path]] = None,
        persist_token: bool = True,
        use_wsi: bool = False,
        bbox_threshold: float = 0.4,
        low_contrast_enhancement: bool = False,
        swap_channels: bool = False,
        gauge_cell_size: bool = False,
        block_size: int = 400,
        overlap: int = 56,
        iou_depth: int = 56,
        iou_threshold: float = 0.5,
        model_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__("CellSAM")
        self.access_token = access_token
        self.persist_token = persist_token
        self.env_path = Path(env_path) if env_path else Path.cwd() / ".env"

        self.use_wsi = bool(use_wsi)
        self.bbox_threshold = float(bbox_threshold)
        self.low_contrast_enhancement = bool(low_contrast_enhancement)
        self.swap_channels = bool(swap_channels)
        self.gauge_cell_size = bool(gauge_cell_size)
        self.block_size = int(block_size)
        self.overlap = int(overlap)
        self.iou_depth = int(iou_depth)
        self.iou_threshold = float(iou_threshold)
        self.model_path = Path(model_path) if model_path is not None else None

        # Ensure DEEPCELL_ACCESS_TOKEN is available
        self._ensure_token()

    def _ensure_token(self):
        # Load .env first
        load_dotenv(dotenv_path=self.env_path)

        if self.access_token:
            os.environ["DEEPCELL_ACCESS_TOKEN"] = self.access_token
            if self.persist_token:
                set_key(str(self.env_path), "DEEPCELL_ACCESS_TOKEN", self.access_token)

        elif "DEEPCELL_ACCESS_TOKEN" not in os.environ:
            raise RuntimeError(
                "DEEPCELL_ACCESS_TOKEN not found. "
                "Provide it via:\n"
                "  • .env file with DEEPCELL_ACCESS_TOKEN=your_token\n"
                "  • env var export DEEPCELL_ACCESS_TOKEN=your_token\n"
                "  • CellSAM(access_token='your_token')"
            )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run CellSAM and return (N, H, W) uint16 instance masks.
        """
        if image.ndim not in (2, 3):
            raise ValueError(f"CellSAM expects 2D images: (H,W) or (H,W,C); got {image.shape}")

        img = np.asarray(image)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]

        mask = cellsam_pipeline(
            img,
            chunks=256,
            model_path=str(self.model_path) if self.model_path is not None else None,
            bbox_threshold=self.bbox_threshold,
            low_contrast_enhancement=self.low_contrast_enhancement,
            swap_channels=self.swap_channels,
            use_wsi=self.use_wsi,
            gauge_cell_size=self.gauge_cell_size,
            block_size=self.block_size,
            overlap=self.overlap,
            iou_depth=self.iou_depth,
            iou_threshold=self.iou_threshold,
        )

        return _labels_to_mask_stack(mask.astype(np.uint32))
