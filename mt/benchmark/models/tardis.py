from __future__ import annotations

import cv2
import numpy as np
import os
import tempfile
import zipfile

import torch

from tardis_em.utils.predictor import Predictor
from tardis_em.utils.aws import get_all_version_aws, get_weights_aws
from .base import BaseModel


class TARDIS(BaseModel):
    VALID_NETWORKS = {"unet", "unet3plus", "fnet_attn", "dist"}
    VALID_SUBTYPES = {"16", "32", "64", "96", "128", "triang", "full"}
    MODEL_NAME = "microtubules_2d"

    def __init__(
        self,
        device: str = "cpu",
        rotate: bool = False,
        network: str = "unet",
        subtype: str = "full",
        model_version: str | None = None,
        sigma: float = 1.0,
        sigmoid: bool = True,
    ):
        super().__init__("TARDIS_TIRF_MT_2D")
        self.device = device
        self.rotate = rotate

        if network not in self.VALID_NETWORKS:
            raise ValueError(f"Invalid network '{network}', must be one of {self.VALID_NETWORKS}")
        if subtype not in self.VALID_SUBTYPES:
            raise ValueError(f"Invalid subtype '{subtype}', must be one of {self.VALID_SUBTYPES}")

        self.network = network
        self.subtype = subtype
        self.model = self.MODEL_NAME
        self.model_version = model_version
        self.sigma = sigma
        self.sigmoid = sigmoid

    def _get_latest_version(self) -> str | None:
        versions = get_all_version_aws(self.network, self.subtype, self.model)
        if not versions:
            return None
        vs = []
        for v in versions:
            try:
                num = int(v.split("_")[1])
                vs.append((num, v))
            except Exception:
                pass
        if vs:
            _, maxv = max(vs, key=lambda x: x[0])
            return maxv
        return None

    def _is_valid_checkpoint(self, path: str) -> bool:
        """
        Basic check: is file a zip archive (as PyTorch expects)?
        And is it not tiny (size threshold)?
        """
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if size < 1_000:  # maybe too small to be legitimate
            return False
        # Check whether it's a zip
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    def predict(self, image: np.ndarray) -> np.ndarray:
        # preprocess
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.ndim != 2:
            raise ValueError(f"image must be 2D (H, W); got {image.shape}")

        h, w = image.shape
        img_size = (h, w)

        version = self.model_version
        if version is None:
            version = self._get_latest_version()

        # fetch checkpoint
        checkpoint_obj = get_weights_aws(
            network=self.network,
            subtype=self.subtype,
            model=self.model,
            version=(None if version is None else int(version.split("_")[1]))
        )

        # write to file if it's buffer
        if hasattr(checkpoint_obj, "read"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
            tmp.write(checkpoint_obj.read())
            tmp.flush()
            tmp.close()
            checkpoint_path = tmp.name
        else:
            checkpoint_path = checkpoint_obj

        # Validate checkpoint
        if not self._is_valid_checkpoint(checkpoint_path):
            # clean up possibly bad file
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass
            raise RuntimeError(f"Checkpoint file is invalid or corrupted: {checkpoint_path}")

        # Now safe to call Predictor
        predictor = Predictor(
            device=self.device,
            network=self.network,
            checkpoint=checkpoint_path,
            subtype=self.subtype,
            model_version=version,
            img_size=img_size,
            model_type="2d",
            sigma=self.sigma,
            sigmoid=self.sigmoid,
            _2d=True,
            logo=False,
        )

        pred_mask = predictor.predict(image, mask=None, rotate=self.rotate)

        if pred_mask is None:
            return np.zeros((h, w), dtype=np.uint16)

        if pred_mask.dtype != np.uint16:
            pred_mask = pred_mask.astype(np.uint16, copy=False)

        return pred_mask
