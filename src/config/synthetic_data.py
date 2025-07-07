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

    # ─── post-initialization and validation ─────────────────────
    def __post_init__(self):
        super().__post_init__()  # Call the base class's __post_init__
        logger.info(f"SyntheticDataConfig '{self.id}' initialized. Running initial validation...")
        try:
            self.validate()
        except ValueError as e:
            logger.critical(f"Initial validation of SyntheticDataConfig '{self.id}' failed: {e}", exc_info=False)
            raise  # Re-raise the error as it's a critical configuration issue.

    def validate(self):
        """
        Validates all configuration parameters, including nested configs.
        Raises ValueError if any parameter is invalid.
        """
        logger.debug(f"Starting validation for SyntheticDataConfig '{self.id}'...")
        errors = []

        # --- Core Video Info ---
        if not (isinstance(self.img_size, (list, tuple)) and len(self.img_size) == 2 and all(
                isinstance(x, int) and x > 0 for x in self.img_size)):
            errors.append(f"img_size must be a tuple of two positive integers, but got {self.img_size}.")

        if not (self.fps > 0):
            errors.append(f"fps must be greater than 0, but got {self.fps}.")

        if not (self.num_frames > 0):
            errors.append(f"num_frames must be greater than 0, but got {self.num_frames}.")

        # --- Microtubule Kinematics ---
        if not (self.growth_speed > 0):
            errors.append(f"growth_speed must be positive, but got {self.growth_speed}.")
        if not (self.shrink_speed > 0):
            errors.append(f"shrink_speed must be positive, but got {self.shrink_speed}.")
        if not (0 <= self.catastrophe_prob <= 1):
            errors.append(f"catastrophe_prob must be between 0 and 1, but got {self.catastrophe_prob}.")
        if not (0 <= self.rescue_prob <= 1):
            errors.append(f"rescue_prob must be between 0 and 1, but got {self.rescue_prob}.")

        if not (0 < self.min_base_wagon_length <= self.max_base_wagon_length):
            errors.append(
                f"min_base_wagon_length ({self.min_base_wagon_length}) must be positive and <= max_base_wagon_length ({self.max_base_wagon_length}).")
        if not (self.max_num_wagons >= 1):
            errors.append(f"max_num_wagons must be >= 1, but got {self.max_num_wagons}.")
        if not (self.max_angle >= 0):
            errors.append(f"max_angle must be non-negative, but got {self.max_angle}.")
        if not (0 <= self.bending_prob <= 1):
            errors.append(f"bending_prob must be between 0 and 1, but got {self.bending_prob}.")
        if not (self.max_angle_sign_changes >= 0):
            errors.append(f"max_angle_sign_changes must be non-negative, but got {self.max_angle_sign_changes}.")
        if not (0 <= self.prob_to_flip_bend <= 1):
            errors.append(f"prob_to_flip_bend must be between 0 and 1, but got {self.prob_to_flip_bend}.")

        if not (0 < self.min_length_min <= self.min_length_max):
            errors.append(
                f"min_length_min ({self.min_length_min}) must be positive and <= min_length_max ({self.min_length_max}).")
        if not (0 < self.max_length_min <= self.max_length_max):
            errors.append(
                f"max_length_min ({self.max_length_min}) must be positive and <= max_length_max ({self.max_length_max}).")
        if not (self.min_length_max <= self.max_length_min):
            errors.append(f"min_length_max ({self.min_length_max}) must be <= max_length_min ({self.max_length_min}).")

        if not (0 < self.min_wagon_length_min <= self.min_wagon_length_max):
            errors.append(
                f"min_wagon_length_min ({self.min_wagon_length_min}) must be positive and <= min_wagon_length_max ({self.min_wagon_length_max}).")
        if not (0 < self.max_wagon_length_min <= self.max_wagon_length_max):
            errors.append(
                f"max_wagon_length_min ({self.max_wagon_length_min}) must be positive and <= max_wagon_length_max ({self.max_wagon_length_max}).")

        if not (self.pause_on_max_length >= 0):
            errors.append(f"pause_on_max_length must be non-negative, but got {self.pause_on_max_length}.")
        if not (self.pause_on_min_length >= 0):
            errors.append(f"pause_on_min_length must be non-negative, but got {self.pause_on_min_length}.")

        if not (self.num_tubuli >= 0):
            errors.append(f"num_tubuli must be non-negative, but got {self.num_tubuli}.")
        if not (self.tubuli_seed_min_dist >= 0):
            errors.append(f"tubuli_seed_min_dist must be non-negative, but got {self.tubuli_seed_min_dist}.")
        if not (self.margin >= 0):
            errors.append(f"margin must be non-negative, but got {self.margin}.")
        # Check margin and min_dist vs image size
        if (self.tubuli_seed_min_dist > min(self.img_size) / 2):
            errors.append(
                f"tubuli_seed_min_dist ({self.tubuli_seed_min_dist}) is too large relative to image size {self.img_size}.")
        if (2 * self.margin >= min(self.img_size)):
            errors.append(f"Margin ({self.margin}) is too large, could cover entire image (2*margin >= min(img_size)).")

        # --- PSF / drawing width ---
        if not (self.sigma_x >= 0):
            errors.append(f"sigma_x must be non-negative, but got {self.sigma_x}.")
        if not (self.sigma_y >= 0):
            errors.append(f"sigma_y must be non-negative, but got {self.sigma_y}.")
        if not (self.width_var_std >= 0):
            errors.append(f"width_var_std must be non-negative, but got {self.width_var_std}.")

        # --- Photophysics / Camera Realism ---
        if not (0 <= self.background_level <= 1):
            errors.append(f"background_level must be between 0 and 1, but got {self.background_level}.")
        # tubulus_contrast can be negative for dark tubules, but might have a reasonable range
        # if not (-1.0 <= self.tubulus_contrast <= 1.0): # Example range check
        #     errors.append(f"tubulus_contrast must be between -1.0 and 1.0, but got {self.tubulus_contrast}.")
        if not (self.seed_red_channel_boost >= 0):
            errors.append(f"seed_red_channel_boost must be non-negative, but got {self.seed_red_channel_boost}.")
        if not (all(0.0 <= c <= 1.0 for c in self.annotation_color_rgb) and len(self.annotation_color_rgb) == 3):
            errors.append(
                f"annotation_color_rgb must be a tuple of 3 floats between 0.0 and 1.0, but got {self.annotation_color_rgb}.")
        if not (self.tip_brightness_factor >= 0):
            errors.append(f"tip_brightness_factor must be non-negative, but got {self.tip_brightness_factor}.")
        if not (self.quantum_efficiency > 0):
            errors.append(f"quantum_efficiency must be positive, but got {self.quantum_efficiency}.")
        if not (0 <= self.gaussian_noise <= 1):
            errors.append(f"gaussian_noise must be between 0 and 1, but got {self.gaussian_noise}.")
        if not (self.bleach_tau > 0 or self.bleach_tau == math.inf):
            errors.append(f"bleach_tau must be positive or math.inf, but got {self.bleach_tau}.")
        if not (self.jitter_px >= 0):
            errors.append(f"jitter_px must be non-negative, but got {self.jitter_px}.")
        if not (0 <= self.vignetting_strength <= 1):
            errors.append(f"vignetting_strength must be between 0 and 1, but got {self.vignetting_strength}.")
        if not (self.global_blur_sigma >= 0):
            errors.append(f"global_blur_sigma must be non-negative, but got {self.global_blur_sigma}.")

        # --- Annotations ---
        if not (self.um_per_pixel > 0):
            errors.append(f"um_per_pixel must be positive, but got {self.um_per_pixel}.")
        if not (self.scale_bar_um > 0):
            errors.append(f"scale_bar_um must be positive, but got {self.scale_bar_um}.")

        # --- Recursive Validation for Nested Configs ---
        try:
            self.fixed_spots.validate()
        except ValueError as e:
            errors.append(f"Fixed spots config validation failed: {e}")
            logger.error(f"Nested fixed_spots config invalid for '{self.id}'.")

        try:
            self.moving_spots.validate()
        except ValueError as e:
            errors.append(f"Moving spots config validation failed: {e}")
            logger.error(f"Nested moving_spots config invalid for '{self.id}'.")

        try:
            self.random_spots.validate()
        except ValueError as e:
            errors.append(f"Random spots config validation failed: {e}")
            logger.error(f"Nested random_spots config invalid for '{self.id}'.")

        if self.albumentations:  # Only validate if not None
            try:
                self.albumentations.validate()
            except ValueError as e:
                errors.append(f"Albumentations config validation failed: {e}")
                logger.error(f"Nested albumentations config invalid for '{self.id}'.")
        else:
            logger.debug(f"Albumentations config is None for '{self.id}', skipping validation.")

        # --- Final Error Check ---
        if errors:
            full_msg = f"SyntheticDataConfig '{self.id}' validation failed with {len(errors)} error(s):\n" + "\n".join(
                errors)
            logger.error(full_msg)
            raise ValueError(full_msg)

        logger.info(f"SyntheticDataConfig '{self.id}' validation successful.")