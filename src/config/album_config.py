from dataclasses import dataclass

from .base import BaseConfig


@dataclass(eq=False)
class AlbumentationsConfig(BaseConfig):
    """Configuration for post-generation image augmentations using Albumentations."""

    # --- Geometric Transforms ---
    p: float = 0.0  # The "master" probability that any augmentation is applied to a frame.

    rotate_limit: int = 15  # Max rotation in degrees. Set to 0 to disable.
    shift_scale_rotate_p: float = 0.5

    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5

    # Simulates tissue stretching. Crucial for microscopy.
    elastic_p: float = 0.3
    elastic_alpha: int = 1
    elastic_sigma: int = 20
    elastic_alpha_affine: int = 20

    # Simulates lens distortion.
    grid_distortion_p: float = 0.2

    # --- Pixel-level & Noise Transforms ---
    brightness_contrast_p: float = 0.5
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1

    gauss_noise_p: float = 0.3
    gauss_noise_mean_range: tuple[float, float] = (-0.1, 0.1)
    gauss_noise_std_range: tuple[int, int] = (0.1, 0.5)

    def validate(self):
        assert 0.0 <= self.p <= 1.0, "Master probability 'p' must be between 0 and 1."