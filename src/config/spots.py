import logging
from dataclasses import dataclass
from typing import Tuple, Optional

from optuna import Trial

from .base import BaseConfig

logger = logging.getLogger(f"mt.{__name__}")


@dataclass(eq=False)
class SpotTuningConfig(BaseConfig):
    """Configuration for tuning the parameters of a single type of spot."""
    count_range: Tuple[int, int] = (0, 50)
    intensity_min_range: Tuple[float, float] = (0.0, 0.2)
    intensity_max_range: Tuple[float, float] = (0.0, 0.3)
    radius_min_range: Tuple[int, int] = (1, 5)
    radius_max_range: Tuple[int, int] = (5, 10)
    kernel_size_min_range: Tuple[int, int] = (0, 5)
    kernel_size_max_range: Tuple[int, int] = (5, 10)
    sigma_range: Tuple[float, float] = (0.1, 5.0)
    # Specific to moving spots
    max_step_range: Optional[Tuple[int, int]] = None

    def validate(self):
        pass

@dataclass(eq=False)
class SpotConfig(BaseConfig):
    """Configuration for a single type of spot in a synthetic video."""
    count: int = 20
    intensity_min: float = 0.005
    intensity_max: float = 0.08
    radius_min: int = 1
    radius_max: int = 3
    kernel_size_min: int = 0
    kernel_size_max: int = 2
    sigma: float = 0.5
    color_mode: str = "dark"
    """Color mode for the spots, either 'dark' or 'bright'."""

    # Specific to moving spots
    max_step: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        logger.debug(f"SpotConfig initialized with count={self.count}, color_mode='{self.color_mode}'.")


    def validate(self):
        logger.debug("Validating SpotConfig parameters...")
        errors = []

        if not (self.count >= 0):
            errors.append(f"Count must be non-negative, but got {self.count}.")

        if not (0.0 <= self.intensity_min <= 1.0):
            errors.append(f"Intensity min must be between 0.0 and 1.0, but got {self.intensity_min}.")
        if not (0.0 <= self.intensity_max <= 1.0):
            errors.append(f"Intensity max must be between 0.0 and 1.0, but got {self.intensity_max}.")
        if not (self.intensity_min <= self.intensity_max):
            errors.append(
                f"Intensity min ({self.intensity_min}) must be less than or equal to intensity max ({self.intensity_max}).")

        if not (self.radius_min >= 0):
            errors.append(f"Radius min must be non-negative, but got {self.radius_min}.")
        if not (self.radius_max >= 0):
            errors.append(f"Radius max must be non-negative, but got {self.radius_max}.")
        if not (self.radius_min <= self.radius_max):
            errors.append(
                f"Radius min ({self.radius_min}) must be less than or equal to radius max ({self.radius_max}).")

        if not (self.kernel_size_min >= 0):
            errors.append(f"Kernel size min must be non-negative, but got {self.kernel_size_min}.")
        if not (self.kernel_size_max >= 0):
            errors.append(f"Kernel size max must be non-negative, but got {self.kernel_size_max}.")
        if not (self.kernel_size_min <= self.kernel_size_max):
            errors.append(
                f"Kernel size min ({self.kernel_size_min}) must be less than or equal to kernel size max ({self.kernel_size_max}).")

        if not (self.sigma > 0.0):
            errors.append(f"Sigma must be positive, but got {self.sigma}.")

        if self.color_mode not in ["dark", "bright"]:
            errors.append(f"Color mode must be 'dark' or 'bright', but got '{self.color_mode}'.")

        if errors:
            full_msg = f"SpotConfig validation failed with {len(errors)} error(s):\n" + "\n".join(errors)
            logger.error(full_msg)
            raise ValueError(full_msg)

        logger.info("SpotConfig validation successful.")

    @staticmethod
    def from_trial(trial: Trial, name: str, tuning: SpotTuningConfig) -> 'SpotConfig':
        """Creates a SpotConfig instance by suggesting parameters from an Optuna trial."""
        logger.info(f"Suggesting SpotConfig parameters for '{name}' using Optuna trial {trial.number}.")

        # Log the ranges being used for suggestion
        logger.debug(
            f"Tuning ranges for '{name}': count={tuning.count_range}, intensity_min={tuning.intensity_min_range}, intensity_max={tuning.intensity_max_range}, radius={tuning.radius_min_range}-{tuning.radius_max_range}, kernel_size={tuning.kernel_size_min_range}-{tuning.kernel_size_max_range}, sigma={tuning.sigma_range}, max_step={tuning.max_step_range}.")

        count = trial.suggest_int(f"{name}_count", *tuning.count_range)
        logger.debug(f"Suggested '{name}_count': {count}")

        intensity_min = trial.suggest_float(f"{name}_intensity_min", *tuning.intensity_min_range)
        logger.debug(f"Suggested '{name}_intensity_min': {intensity_min}")

        # Ensure intensity_max is not less than intensity_min
        intensity_max = trial.suggest_float(
            f"{name}_intensity_max",
            max(intensity_min, tuning.intensity_max_range[0]),
            tuning.intensity_max_range[1])
        logger.debug(f"Suggested '{name}_intensity_max' (constrained by min {intensity_min}): {intensity_max}")

        radius_min = trial.suggest_int(f"{name}_radius_min", *tuning.radius_min_range)
        logger.debug(f"Suggested '{name}_radius_min': {radius_min}")

        # Ensure radius_max is not less than radius_min
        radius_max = trial.suggest_int(
            f"{name}_radius_max",
            max(radius_min, tuning.radius_max_range[0]),
            tuning.radius_max_range[1])
        logger.debug(f"Suggested '{name}_radius_max' (constrained by min {radius_min}): {radius_max}")

        kernel_size_min = trial.suggest_int(f"{name}_kernel_size_min", *tuning.kernel_size_min_range)
        logger.debug(f"Suggested '{name}_kernel_size_min': {kernel_size_min}")

        # Ensure kernel_size_max is not less than kernel_size_min
        kernel_size_max = trial.suggest_int(
            f"{name}_kernel_size_max",
            max(kernel_size_min, tuning.kernel_size_max_range[0]),
            tuning.kernel_size_max_range[1])
        logger.debug(f"Suggested '{name}_kernel_size_max' (constrained by min {kernel_size_min}): {kernel_size_max}")

        sigma = trial.suggest_float(f"{name}_sigma", *tuning.sigma_range)
        logger.debug(f"Suggested '{name}_sigma': {sigma}")

        max_step = None
        if tuning.max_step_range is not None:
            max_step = trial.suggest_int(f"{name}_max_step", *tuning.max_step_range)
            logger.debug(f"Suggested '{name}_max_step': {max_step}")
        else:
            logger.debug(f"'{name}_max_step' range is not defined in tuning config, skipping suggestion.")

        config = SpotConfig(
            count=count,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            radius_min=radius_min,
            radius_max=radius_max,
            kernel_size_min=kernel_size_min,
            kernel_size_max=kernel_size_max,
            sigma=sigma,
            max_step=max_step,
        )
        logger.info(f"Successfully created SpotConfig for '{name}' via Optuna trial. Final config: {config.asdict()}")
        return config