import math
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .album_config import AlbumentationsConfig
from .base import BaseConfig
from .spots import SpotConfig


@dataclass(eq=False)
class SyntheticDataConfig(BaseConfig):
    """
    Configuration for synthetic microtubule video generation.
    """

    # ─── core video info ────────────────────────────────────
    id: int | str = 306
    img_size: Tuple[int, int] = (462, 462)
    fps: int = 5
    num_frames: int = 50
    color_mode: bool = True

    # ─── microtubule kinematics (stochastic model) ─────────────────────────
    growth_speed: float = 2.5
    shrink_speed: float = 5.0  # up to 20x growth_speed
    catastrophe_prob: float = 0.01
    rescue_prob: float = 0.005

    min_base_wagon_length: float = (
        10.0  # base wagon length is fixed and is never undergoing a catastrophe
    )
    max_base_wagon_length: float = 50.0
    max_num_wagons: int = 20
    max_angle: float = 0.1  # Max angle in radians between wagons
    bending_prob: float = 0.01  # Probability of bending at all

    max_angle_sign_changes: int = 0  # 0 for C-shape, 1 for S-shape, etc.
    prob_to_flip_bend: float = 0.001  # Probability to use an available sign change

    min_length_min: int = 50
    min_length_max: int = 80
    max_length_min: int = 100
    max_length_max: int = 200

    min_wagon_length_min: int = 1
    min_wagon_length_max: int = 5
    max_wagon_length_min: int = 10
    max_wagon_length_max: int = 20

    pause_on_max_length: int = 5
    pause_on_min_length: int = 10

    num_tubuli: int = 20
    tubuli_seed_min_dist: int = 50
    margin: int = 5

    # ─── PSF / drawing width ───────────────────────────────
    sigma_x: float = 0.3
    sigma_y: float = 0.8
    width_var_std: float = 0.05

    # ─── new photophysics / camera realism ─────────────────
    background_level: float = 0.8
    tubulus_contrast: float = -0.4
    seed_red_channel_boost: float = 0.5  # Boost the red channel of the seed color

    annotation_color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    tip_brightness_factor: float = 1.0

    quantum_efficiency: float = 50.0  # Higher value = less Poisson noise
    gaussian_noise: float = 0.09  # Camera read noise

    bleach_tau: float = math.inf
    jitter_px: float = 0.5
    vignetting_strength: float = 0.05

    global_blur_sigma: float = 0.9

    fixed_spots: SpotConfig = field(
        default_factory=lambda: SpotConfig(
            count=30, intensity_max=0.1, radius_max=3, kernel_size_max=3, sigma=0.1
        )
    )
    moving_spots: SpotConfig = field(
        default_factory=lambda: SpotConfig(
            count=20, intensity_max=0.08, radius_max=3, kernel_size_max=2, sigma=0.3, max_step=5
        )
    )
    random_spots: SpotConfig = field(
        default_factory=lambda: SpotConfig(
            count=20, intensity_max=0.08, radius_max=5, kernel_size_max=2, sigma=0.5
        )
    )

    albumentations: Optional[AlbumentationsConfig] = field(default_factory=AlbumentationsConfig)

    # ─── annotations ─────────────────────────────────────
    show_time: bool = True
    show_scale: bool = True
    um_per_pixel: float = 0.1
    scale_bar_um: float = 5.0

    # ─── misc ──────────────────────────────────────────────
    generate_tubuli_mask: bool = True
    generate_seed_mask: bool = False

    # ─── validation helper (optional) ─────────────────────
    def validate(self):
        assert 0 <= self.background_level <= 1, "background_level must be 0-1"
        assert 0 <= self.gaussian_noise <= 1, "gaussian_noise must be 0-1"
        assert self.num_frames > 0 and self.fps > 0, "frames & fps must be >0"
        assert self.jitter_px >= 0, "jitter_px must be ≥0"
        assert self.growth_speed > 0 and self.shrink_speed > 0
        assert 0 <= self.catastrophe_prob <= 1 and 0 <= self.rescue_prob <= 1

        # ─── checks for wagons ───────────────────────────
        assert self.max_num_wagons >= 1, "max_num_wagons must be ≥1"
        assert (
            self.min_wagon_length_min < self.min_wagon_length_max
        ), "min_wagon_length_min < min_wagon_length_max"
        assert (
            self.max_wagon_length_min < self.max_wagon_length_max
        ), "max_wagon_length_min < max_wagon_length_max"
