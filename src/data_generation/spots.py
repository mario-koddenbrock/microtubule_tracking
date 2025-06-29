from typing import Tuple

import cv2
import numpy as np

from config.spots import SpotConfig


class SpotGenerator:
    """
    Manages the state and drawing of a collection of spots (fixed or moving).
    """

    def __init__(self, spot_cfg: SpotConfig, img_shape: Tuple[int, int], color_mode: str = "dark"):
        """
        Initializes the generator and all its spot properties.
        """
        self.cfg = spot_cfg
        self.img_shape = img_shape
        self.n_spots = self.cfg.count
        self.color_mode = color_mode
        self._initialize_properties()

    def _initialize_properties(self):
        """Generates the initial state for all spot properties."""
        if self.n_spots == 0:
            self.coords, self.intensities, self.radii, self.kernel_sizes = [], [], [], []
            return

        h, w = self.img_shape
        self.coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(self.n_spots)]
        self.intensities = [np.random.uniform(self.cfg.intensity_min, self.cfg.intensity_max) for _ in
                            range(self.n_spots)]
        self.radii = [np.random.randint(self.cfg.radius_min, self.cfg.radius_max + 1) for _ in range(self.n_spots)]
        self.kernel_sizes = [np.random.randint(self.cfg.kernel_size_min, self.cfg.kernel_size_max + 1) for _ in
                             range(self.n_spots)]

    def update(self):
        """
        Updates the state for the next frame. Only moves spots if max_step is defined.
        """
        if self.cfg.max_step is None or self.n_spots == 0:
            return

        h, w = self.img_shape
        step_x = np.random.randint(-self.cfg.max_step, self.cfg.max_step + 1, size=self.n_spots)
        step_y = np.random.randint(-self.cfg.max_step, self.cfg.max_step + 1, size=self.n_spots)

        new_coords = []
        for i, (y, x) in enumerate(self.coords):
            new_y = np.clip(y + step_y[i], 0, h - 1)
            new_x = np.clip(x + step_x[i], 0, w - 1)
            new_coords.append((new_y, new_x))
        self.coords = new_coords

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Draws the current spots onto the given image."""
        if self.n_spots == 0:
            return img

        # The static method `draw_spots` is now RGB-aware.
        return self.draw_spots(
            img,
            self.coords,
            self.intensities,
            self.radii,
            self.kernel_sizes,
            self.cfg.sigma,
            color_mode=self.color_mode
        )

    @staticmethod
    def draw_spots(img, spot_coords, intensity, radii, kernel_size, sigma, color_mode="dark"):
        # CHANGED: Correctly get image dimensions, handling both grayscale and RGB.
        if img.ndim == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape

        if isinstance(intensity, list):
            if len(intensity) != len(spot_coords):
                raise ValueError("Intensity list must match the length of spot coordinates.")
        else:
            intensity = [intensity] * len(spot_coords)

        for idx, (y, x) in enumerate(spot_coords):
            # Create a 2D (grayscale) mask for the spot. This is efficient.
            mask = np.zeros((h, w), dtype=np.float32)
            mask = cv2.circle(mask, (int(x), int(y)), radii[idx], intensity[idx], -1)
            kernel = 2 * kernel_size[idx] + 1
            # Blur the 2D mask
            mask = cv2.GaussianBlur(mask, (kernel, kernel), sigma)

            # Add the mask to the image in an RGB-safe way.
            if img.ndim == 3:
                # Use NumPy broadcasting to add the 2D mask to each of the 3 color channels.
                # `mask[..., np.newaxis]` changes its shape from (H, W) to (H, W, 1),
                # which NumPy then automatically applies to all 3 channels of `img`.
                if color_mode == "dark":
                    img -= mask[..., np.newaxis]
                else:
                    img += mask[..., np.newaxis]
            else:
                # image is grayscale
                if color_mode == "dark":
                    img -= mask
                else:
                    img += mask

        return img

    @staticmethod
    def apply_random_spots(img: np.ndarray, spot_cfg: SpotConfig) -> np.ndarray:
        """
        Adds stateless spots that are regenerated completely on every frame.
        This function is now RGB-aware.
        """
        n_spots = spot_cfg.count
        if n_spots == 0:
            return img

        # CHANGED: Correctly get image dimensions, handling both grayscale and RGB.
        if img.ndim == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape

        # Generate all properties on-the-fly for each frame
        coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_spots)]
        intensities = [np.random.uniform(spot_cfg.intensity_min, spot_cfg.intensity_max) for _ in range(n_spots)]
        radii = [np.random.randint(spot_cfg.radius_min, spot_cfg.radius_max + 1) for _ in range(n_spots)]
        kernel_sizes = [np.random.randint(spot_cfg.kernel_size_min, spot_cfg.kernel_size_max + 1) for _ in
                        range(n_spots)]

        # This call is now safe because SpotGenerator.draw_spots is RGB-aware.
        return SpotGenerator.draw_spots(
            img,
            coords,
            intensities,
            radii,
            kernel_sizes,
            spot_cfg.sigma,
            color_mode=spot_cfg.color_mode,
        )
