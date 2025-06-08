import math
from dataclasses import dataclass, field
from typing import Tuple

from .base import BaseConfig
from .spots import SpotConfig


@dataclass(eq=False)
class SyntheticDataConfig(BaseConfig):
    """
    Configuration for synthetic microtubule video generation.

    Examples:
        Load default configuration:
            config = SyntheticDataConfig()

        Load from YAML file:
            config = SyntheticDataConfig.load("config.yml")

        Load from JSON file with overrides:
            config = SyntheticDataConfig.load("config.json", overrides={"fps": 30, "snr": 5})

        Create and export config to a file:
            config = SyntheticDataConfig()
            config.to_yml("output_config.yml")
            config.to_json("output_config.json")

    Attributes:
        TODO
    """
    # ─── core video info ────────────────────────────────────
    id: int | str = 0
    img_size: Tuple[int, int] = (462, 462)  # (H, W)
    fps: int = 5
    num_frames: int = 50
    color_mode: bool = False

    # ─── microtubule kinematics ────────────────────────────
    grow_frames: int = 20
    shrink_frames: int = 10
    profile_noise: float = 5

    mmin_base_wagon_length: float = 30.0
    max_base_wagon_length: float = 30.0
    max_num_wagons: int = 5
    max_angle: float = math.pi / 4
    max_angle_change_prob: float = 0.05

    min_length_min: int = 50
    min_length_max: int = 80
    max_length_min: int = 100
    max_length_max: int = 200

    min_wagon_length_min: int = 50
    min_wagon_length_max: int = 80
    max_wagon_length_min: int = 100
    max_wagon_length_max: int = 200
    pause_on_max_length: int = 2
    pause_on_min_length: int = 5

    num_tubulus: int = 20
    tubuli_min_dist:int = 20
    margin: int = 5

    width_var_std: float = 0.05  # std of the width variation (relative to the mean width)
    bend_amplitude: float = 5.0  # max lateral offset
    bend_prob: float = 0.01  # only ~10 % of tubules curved
    bend_straight_fraction: float = 0.9  # fraction of the tubule length that is straight before bending starts

    # ─── PSF / drawing width ───────────────────────────────
    sigma_x: float = 0.3
    sigma_y: float = 0.8

    # ─── new photophysics / camera realism ─────────────────
    background_level: float = 0.74
    tubulus_contrast: float = 0.2

    gaussian_noise: float = 0.09  # 24 / 255
    bleach_tau: float = math.inf  # photobleaching off by default
    jitter_px: float = 0.5
    vignetting_strength: float = 0.05
    invert_contrast: bool = True  # whether to invert the contrast of the image
    global_blur_sigma: float = 0.9  # global blur applied to the whole image

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
    um_per_pixel:float = 0.108  # adjust to match your microscope
    scale_bar_um:float = 5.0  # 5 micrometers bar

    # ─── misc ──────────────────────────────────────────────
    generate_mask: bool = True

    # ─── validation helper (optional) ─────────────────────
    def validate(self):
        assert 0 <= self.background_level <= 1, "background_level must be 0-1"
        assert 0 <= self.gaussian_noise <= 1, "gaussian_noise must be 0-1"
        assert self.num_frames > 0 and self.fps > 0, "frames & fps must be >0"
        assert self.jitter_px >= 0, "jitter_px must be ≥0"
