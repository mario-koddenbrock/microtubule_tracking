import json
import os
from glob import glob
from typing import List, Tuple, Dict, Any

import imageio
import numpy as np


class BenchmarkDataset:
    """Loads the synthetic dataset for benchmarking."""

    def get_image_path(self, idx: int) -> str:
        """
        Returns the file path of the image at the given index.
        """
        if idx < 0 or idx >= len(self.image_files):
            raise IndexError("Index out of bounds for dataset.")
        return self.image_files[idx]

    def __init__(self, data_path: str, num_samples: int = -1):
        """
        Args:
            data_path: Path to the dataset directory.
            num_samples: Number of samples to load. Default is -1 (load all).
        """
        self.data_path = data_path
        self.image_path = os.path.join(self.data_path, "images")
        self.image_files = sorted(glob(os.path.join(self.image_path, "*.png")))
        if num_samples > 0:
            self.image_files = self.image_files[:num_samples]

        if not self.image_files:
            raise FileNotFoundError(f"Dataset not found or incomplete in {self.image_path}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Returns:
            - image: (H, W, C)
            - gt_mask: (H, W) with instance labels
            - gt_data: dicts for the frame
        """
        image_path = self.image_files[idx]
        mask_path = image_path.replace("images", "image_masks")
        gt_path = image_path.replace("images", "gt")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        try:
            frame_idx_str = os.path.basename(image_path).split("_")[-1].split(".")[0]
            frame_idx = int(frame_idx_str)
        except ValueError:
            raise ValueError(f"Could not extract frame index from image filename: {image_path}")

        gt_path = gt_path.replace("png", "json")
        gt_path = gt_path.replace("frame", "ground")
        gt_path = gt_path.replace(frame_idx_str, "truth")

        image = imageio.imread(image_path)
        mask = imageio.imread(mask_path)

        if os.path.isfile(gt_path):
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
                frame_gt = gt_data[frame_idx]
        else:
            frame_gt = {}


        return image, mask, frame_gt
