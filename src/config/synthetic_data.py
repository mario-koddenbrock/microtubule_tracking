import math
from dataclasses import dataclass, field
from typing import Tuple

from .base import BaseConfig
from .spots import SpotConfig


@dataclass(eq=False)
class SyntheticDataConfig(BaseConfig):
    """
    Configuration for synthetic microtubule video generation.
    """
    # ─── core video info ────────────────────────────────────
    id: int | str = 0
    img_size: Tuple[int, int] = (462, 462)
    fps: int = 5
    num_frames: int = 100
    color_mode: bool = True

    # ─── microtubule kinematics (stochastic model) ─────────────────────────
    growth_speed: float = 2.5
    shrink_speed: float = 5.0
    catastrophe_prob: float = 0.01
    rescue_prob: float = 0.005

    min_base_wagon_length: float = 10.0
    max_base_wagon_length: float = 60.0
    max_num_wagons: int = 20
    max_angle: float = math.pi / 16
    max_angle_change_prob: float = 0.001

    min_length_min: int = 50
    min_length_max: int = 80
    max_length_min: int = 100
    max_length_max: int = 200

    min_wagon_length_min: int = 5
    min_wagon_length_max: int = 20
    max_wagon_length_min: int = 20
    max_wagon_length_max: int = 100

    pause_on_max_length: int = 5
    pause_on_min_length: int = 10

    num_tubulus: int = 20
    tubuli_min_dist: int = 50
    margin: int = 5

    width_var_std: float = 0.05
    bend_amplitude: float = 5.0
    bend_prob: float = 0.01
    bend_straight_fraction: float = 0.9

    # ─── PSF / drawing width ───────────────────────────────
    sigma_x: float = 0.3
    sigma_y: float = 0.8

    # ─── new photophysics / camera realism ─────────────────
    background_level: float = 0.74
    tubulus_contrast: float = 0.2

    seed_color_rgb: Tuple[float, float, float] = (1.0, 0.2, 0.2)
    tubulus_color_rgb: Tuple[float, float, float] = (0.2, 1.0, 0.2)

    # NEW: Tip-tracking protein simulation
    tip_brightness_factor: float = 1.5  # How much brighter growing tips are

    # NEW: Mixed noise model parameters
    quantum_efficiency: float = 50.0  # Higher value = less Poisson noise
    gaussian_noise: float = 0.09  # Camera read noise

    bleach_tau: float = math.inf
    jitter_px: float = 0.5
    vignetting_strength: float = 0.05
    invert_contrast: bool = True
    global_blur_sigma: float = 0.9

    fixed_spots: SpotConfig = field(default_factory=lambda: SpotConfig(
        count=30, intensity_max=0.1, radius_max=3, kernel_size_max=3, sigma=0.1
    ))
    moving_spots: SpotConfig = field(default_factory=lambda: SpotConfig(
        count=20, intensity_max=0.08, radius_max=3, kernel_size_max=2, sigma=0.3, max_step=5
    ))
    random_spots: SpotConfig = field(default_factory=lambda: SpotConfig(
        count=20, intensity_max=0.08, radius_max=5, kernel_size_max=2, sigma=0.5
    ))

    # ─── annotations ─────────────────────────────────────
    show_time:bool = True
    show_scale:bool = True
    um_per_pixel:float = 0.108
    scale_bar_um:float = 5.0

    # ─── misc ──────────────────────────────────────────────
    generate_mask: bool = True

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
        assert 0 <= self.max_angle_change_prob <= 1, "angle_change_prob must be between 0 and 1"
        assert self.min_wagon_length_min < self.min_wagon_length_max, "min_wagon_length_min < min_wagon_length_max"
        assert self.max_wagon_length_min < self.max_wagon_length_max, "max_wagon_length_min < max_wagon_length_max"