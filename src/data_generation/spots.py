import numpy as np
from typing import List, Tuple

from config.synthetic_data import SyntheticDataConfig
from config.spots import SpotConfig
from data_generation.utils import draw_spots


class SpotGenerator:
    """
    Manages the state and drawing of a collection of spots (fixed or moving).

    This class initializes spot properties (coordinates, intensity, etc.) based on a
    SpotConfig and holds onto that state. It can update the state (for moving spots)
    and apply the spots to an image for each frame.
    """

    def __init__(self, spot_cfg: SpotConfig, img_shape: Tuple[int, int]):
        """
        Initializes the generator and all its spot properties.

        Args:
            spot_cfg: The configuration for this type of spot.
            img_shape: The (height, width) of the image to generate spots within.
        """
        self.cfg = spot_cfg
        self.img_shape = img_shape
        self.n_spots = self.cfg.count

        # Initialize all properties immediately
        self._initialize_properties()

    def _initialize_properties(self):
        """Generates the initial state for all spot properties."""
        if self.n_spots == 0:
            self.coords = []
            self.intensities = []
            self.radii = []
            self.kernel_sizes = []
            return

        h, w = self.img_shape
        self.coords = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(self.n_spots)]

        self.intensities = [
            np.random.uniform(self.cfg.intensity_min, self.cfg.intensity_max)
            for _ in range(self.n_spots)
        ]
        self.radii = [
            np.random.randint(self.cfg.radius_min, self.cfg.radius_max + 1)
            for _ in range(self.n_spots)
        ]
        self.kernel_sizes = [
            np.random.randint(self.cfg.kernel_size_min, self.cfg.kernel_size_max + 1)
            for _ in range(self.n_spots)
        ]

    def update(self):
        """
        Updates the state for the next frame. Only moves spots if max_step is defined.
        """
        # This generator only has behavior if it's for moving spots
        if self.cfg.max_step is None or self.n_spots == 0:
            return

        h, w = self.img_shape
        step_x = np.random.randint(-self.cfg.max_step, self.cfg.max_step + 1, size=self.n_spots)
        step_y = np.random.randint(-self.cfg.max_step, self.cfg.max_step + 1, size=self.n_spots)

        new_coords = []
        for i, (y, x) in enumerate(self.coords):
            # Calculate new position and clip it to stay within image bounds
            new_y = np.clip(y + step_y[i], 0, h - 1)
            new_x = np.clip(x + step_x[i], 0, w - 1)
            new_coords.append((new_y, new_x))

        self.coords = new_coords

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Draws the current spots onto the given image."""
        if self.n_spots == 0:
            return img

        # Assuming a draw_spots function exists with this signature
        return draw_spots(
            img,
            self.coords,
            self.intensities,
            self.radii,
            self.kernel_sizes,
            self.cfg.sigma
        )