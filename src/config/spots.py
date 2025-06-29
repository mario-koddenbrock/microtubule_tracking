from dataclasses import dataclass, field
from typing import Tuple, Optional

from .base import BaseConfig
from optuna import Trial


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
    color_mode:str = "dark"
    """Color mode for the spots, either 'dark' or 'bright'."""

    # Specific to moving spots
    max_step: Optional[int] = None

    def validate(self):
        assert self.count >= 0, "Count must be non-negative."


    @staticmethod
    def from_trial(trial: Trial, name: str, tuning: SpotTuningConfig) -> 'SpotConfig':
        """Creates a SpotConfig instance by suggesting parameters from an Optuna trial."""
        count = trial.suggest_int(f"{name}_count", *tuning.count_range)
        intensity_min = trial.suggest_float(f"{name}_intensity_min", *tuning.intensity_min_range)
        intensity_max = trial.suggest_float(
            f"{name}_intensity_max",
            max(intensity_min, tuning.intensity_max_range[0]),
            tuning.intensity_max_range[1])

        radius_min = trial.suggest_int(f"{name}_radius_min", *tuning.radius_min_range)
        radius_max = trial.suggest_int(
            f"{name}_radius_max",
            max(radius_min, tuning.radius_max_range[0]),
            tuning.radius_max_range[1])

        kernel_size_min = trial.suggest_int(f"{name}_kernel_size_min", *tuning.kernel_size_min_range)
        kernel_size_max = trial.suggest_int(
            f"{name}_kernel_size_max",
            max(kernel_size_min, tuning.kernel_size_max_range[0]),
            tuning.kernel_size_max_range[1])

        sigma = trial.suggest_float(f"{name}_sigma", *tuning.sigma_range)

        max_step = None
        if tuning.max_step_range is not None:
            max_step = trial.suggest_int(f"{name}_max_step", *tuning.max_step_range)

        return SpotConfig(
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