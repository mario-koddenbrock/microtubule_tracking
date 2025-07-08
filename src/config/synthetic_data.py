import logging
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .album_config import AlbumentationsConfig
from .base import BaseConfig
from .spots import SpotConfig

logger = logging.getLogger(f"mt.{__name__}")


@dataclass(eq=False)
class SyntheticDataConfig(BaseConfig):
    """
    Configuration for synthetic microtubule video generation using a stateful,
    event-driven model for dynamic instability.
    """

    # ─── core video info ────────────────────────────────────
    id: int | str = 309
    img_size: Tuple[int, int] = (462, 462)
    fps: int = 5
    num_frames: int = 100
    color_mode: bool = True

    # ─── microtubule kinematics (stateful model) ─────────────────────────
    growth_speed: float = 2.5  # Pixels per frame
    shrink_speed: float = 5.0  # Pixels per frame
    catastrophe_prob: float = 0.01  # Probability per frame to switch from growing to shrinking
    rescue_prob: float = 0.005  # Probability per frame for a rescue event

    # NEW: Max frames a tubule can be paused at its minimum length before a forced rescue.
    max_pause_at_min_frames: int = 50

    # The "seed" part of the microtubule. Its length is fixed for the entire simulation.
    min_base_wagon_length: float = 10.0
    max_base_wagon_length: float = 50.0

    # The maximum total length a microtubule can reach before it is forced to shrink.
    max_length_min: int = 100
    max_length_max: int = 200

    # Bending is applied to the dynamic "tail" part of the microtubule.
    tail_wagon_length: float = 10.0  # Visual segments for drawing the tail. Does not affect growth speed.
    max_angle: float = 0.1  # Max angle in radians between tail wagons
    bending_prob: float = 0.1  # Probability of the tail being bent at all
    max_angle_sign_changes: int = 1  # 0 for C-shape, 1 for S-shape, etc.
    prob_to_flip_bend: float = 0.01  # Probability to use an available sign change when adding new visual wagons

    # Seeding parameters
    num_tubuli: int = 20
    tubuli_seed_min_dist: int = 50
    margin: int = 5

    # ─── PSF / drawing width ───────────────────────────────
    sigma_x: float = 0.3
    sigma_y: float = 0.8
    tubule_width_variation: float = 0.05

    # ─── photophysics / camera realism ─────────────────
    background_level: float = 0.8
    tubulus_contrast: float = -0.4
    seed_red_channel_boost: float = 0.5
    tip_brightness_factor: float = 1.2  # Growing tips are brighter

    red_channel_noise_std: float = 0.01  # Std dev for red-only noise

    quantum_efficiency: float = 50.0
    gaussian_noise: float = 0.09
    bleach_tau: float = math.inf
    jitter_px: float = 0.5
    vignetting_strength: float = 0.05
    global_blur_sigma: float = 0.9

    fixed_spots: SpotConfig = field(
        default_factory=lambda: SpotConfig(
            count=30, intensity_max=0.1, radius_max=3, kernel_size_max=3, sigma=0.1, polygon_p=0.3
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
    annotation_color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    um_per_pixel: float = 0.1
    scale_bar_um: float = 5.0

    # ─── misc ──────────────────────────────────────────────
    generate_tubuli_mask: bool = True
    generate_seed_mask: bool = False

    def __post_init__(self):
        super().__post_init__()
        logger.info(f"SyntheticDataConfig '{self.id}' initialized. Running initial validation...")
        try:
            self.validate()
        except ValueError as e:
            logger.critical(f"Initial validation of SyntheticDataConfig '{self.id}' failed: {e}", exc_info=False)
            raise

    def validate(self):
        """Validates all configuration parameters."""
        logger.debug(f"Starting validation for SyntheticDataConfig '{self.id}'...")
        errors = []

        if not (isinstance(self.img_size, (list, tuple)) and len(self.img_size) == 2 and all(
                isinstance(x, int) and x > 0 for x in self.img_size)):
            errors.append(f"img_size must be a tuple of two positive integers, but got {self.img_size}.")
        if not (self.growth_speed > 0): errors.append("growth_speed must be positive.")
        if not (self.shrink_speed > 0): errors.append("shrink_speed must be positive.")
        if not (0 <= self.catastrophe_prob <= 1): errors.append("catastrophe_prob must be between 0 and 1.")
        if not (0 <= self.rescue_prob <= 1): errors.append("rescue_prob must be between 0 and 1.")
        if not (self.max_pause_at_min_frames >= 0): errors.append(
            "max_pause_at_min_frames must be a non-negative integer.")
        if not (0 < self.min_base_wagon_length <= self.max_base_wagon_length):
            errors.append("min_base_wagon_length must be positive and <= max_base_wagon_length.")
        if not (0 < self.max_length_min <= self.max_length_max):
            errors.append("max_length_min must be positive and <= max_length_max.")
        if not (self.max_base_wagon_length <= self.max_length_min):
            errors.append(
                f"max_base_wagon_length ({self.max_base_wagon_length}) must be <= max_length_min ({self.max_length_min}).")
        if not (self.tail_wagon_length > 0): errors.append("tail_wagon_length must be positive.")
        if not (self.red_channel_noise_std >= 0): errors.append("red_channel_noise_std must be non-negative.")

        for name, cfg_instance in [("fixed_spots", self.fixed_spots), ("moving_spots", self.moving_spots),
                                   ("random_spots", self.random_spots)]:
            try:
                cfg_instance.validate()
            except ValueError as e:
                errors.append(f"{name} config validation failed: {e}")

        if errors:
            full_msg = f"Validation failed with {len(errors)} error(s):\n" + "\n".join(errors)
            raise ValueError(full_msg)
        logger.info(f"SyntheticDataConfig '{self.id}' validation successful.")