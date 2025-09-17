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
    # Judging from check_tardis_models.py, there are:
    # ('fnet_attn', '32', 'microtubules_tirf')
    # ('dist', 'triang', 'microtubules')
    # ('dist', 'triang', '2d')

    # I'd also love to see:
    # ('fnet_attn', '32', 'microtubules_2d)

    # Also, I am not sure about the "full" vs. "triang" subtypes (also 32 etc.)

    VALID_COMBINATIONS = {
        ("fnet_attn", "32", "microtubules_tirf"),
        ("dist", "triang", "microtubules"),
        ("dist", "triang", "2d"),
    }

    def __init__(
        self,
        device: str = "cpu",
        network: str = "fnet_attn",
        subtype: str = "full",
        dataset: str = "microtubules_2d",
        model_version: str | None = None,
        sigma: float | None = None,
        sigmoid: bool = True,
        img_size=(512, 512),  # (h,w), fixed for us
    ):
        # Check if given combination is valid
        if (network, subtype, dataset) not in self.VALID_COMBINATIONS:
            raise ValueError(
                f"Invalid combination of network='{network}', subtype='{subtype}', dataset='{dataset}'. "
                f"Valid combinations are: {self.VALID_COMBINATIONS}"
            )

        super().__init__(f"TARDIS-{network}-{subtype}-{dataset}")
        self.device = device
        self.network = network
        self.subtype = subtype
        self.dataset = dataset
        self.model_version = model_version
        self.sigma = sigma
        self.sigmoid = sigmoid
        self.img_size = img_size
        self._predictor = None  # Model weights not loaded yet

    def _get_latest_version(self) -> str | None:
        versions = get_all_version_aws(self.network, self.subtype, self.dataset)
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
            print(f"Checkpoint file too small ({size} bytes): {path}")
            return False
        else:
            print(f"Checkpoint file size: {size} bytes")
        # Check whether it's a zip
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    def _load_model(self):
        if self._predictor is not None:
            return

        version = self.model_version
        if version is None:
            version = self._get_latest_version()

        # fetch checkpoint
        checkpoint_obj = get_weights_aws(
            network=self.network,
            subtype=self.subtype,
            model=self.dataset,
            version=(None if version is None else int(version.split("_")[1])),
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

        self._predictor = (
            Predictor(  # TODO: Maybe we should use GeneralPredcitor to have more hyperparameters?
                device=self.device,
                network=self.network,
                checkpoint=checkpoint_path,
                subtype=self.subtype,
                model_version=version,
                img_size=self.img_size,
                model_type=self.dataset,
                sigma=self.sigma,
                sigmoid=self.sigmoid,
                _2d=True,
                logo=False,
            )
        )

    def predict(self, image: np.ndarray) -> np.ndarray:

        # preprocess
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.ndim != 2:
            raise ValueError(f"image must be 2D (H, W); got {image.shape}")

        h, w = image.shape
        img_size = (h, w)
        assert img_size == self.img_size, f"Expecting image size {self.img_size}, got {img_size}"

        self._load_model()
        assert self._predictor is not None

        pred_mask = self._predictor.predict(image)

        if pred_mask is None:
            return np.zeros((h, w), dtype=np.uint16)

        if pred_mask.dtype != np.uint16:
            pred_mask = pred_mask.astype(np.uint16, copy=False)

        return pred_mask
