import os
import json
from glob import glob
from typing import List, Tuple, Dict, Any
import imageio
import numpy as np


class BenchmarkDataset:
    """Loads the synthetic dataset for benchmarking."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.image_files = sorted(glob(os.path.join(self.data_path, "images", "*.png")))
        self.mask_files = sorted(glob(os.path.join(self.data_path, "image_masks", "*.png")))
        gt_file = glob(os.path.join(self.data_path, "gt", "*.json"))

        if not self.image_files or not self.mask_files or not gt_file:
            raise FileNotFoundError(f"Dataset not found or incomplete in {data_path}")

        with open(gt_file[0], 'r') as f:
            self.gt_data = json.load(f)

        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Returns:
            - image: (H, W, C)
            - gt_mask: (H, W) with instance labels
            - gt_frame_data: list of dicts for the frame
        """
        image = imageio.imread(self.image_files[idx])
        gt_mask = imageio.imread(self.mask_files[idx])

        # The gt_data is a list of lists of dicts, one list per frame
        gt_frame_data = self.gt_data[idx]

        return image, gt_mask, gt_frame_data