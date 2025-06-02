import json
import math
import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import yaml


class BaseConfig(ABC):
    @classmethod
    def load(cls, config_path: Optional[str] = None, overrides: Optional[dict] = None):
        if config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            if config_path.endswith(('.yml', '.yaml')):
                config = cls.from_yml(config_path)
            elif config_path.endswith('.json'):
                config = cls.from_json(config_path)
            else:
                raise ValueError("Unsupported config file format. Use .yml, .yaml, or .json")
        else:
            config = cls()

        if overrides:
            config.update(overrides)

        return config

    def __eq__(self, other):
        if isinstance(other, BaseConfig):
            return self.asdict() == other.asdict()
        return NotImplemented

    def get(self, key):
        return getattr(self, key)

    def update(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Invalid configuration key: {key}")

    def __str__(self):
        return yaml.dump(self.asdict(), sort_keys=False)

    def to_yml(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.asdict(), f)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.asdict(), f, indent=2)

    @classmethod
    def from_yml(cls, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def asdict(self):
        raw = asdict(self)
        for k, v in raw.items():
            if isinstance(v, tuple):
                raw[k] = list(v)
        return raw

    @abstractmethod
    def validate(self):
        """
        Optionally implemented by subclasses to validate configuration logic.
        """
        pass


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
        img_size (tuple): Dimensions of the output image (height, width).
        generate_mask (bool): Whether to generate a mask for the microtubules.
        fps (int): Frames per second for the output video.
        num_frames (int): Total number of frames per video (defines video duration).
        snr (int): Signal-to-noise ratio used for Poisson noise. # TODO separately for tubulus and background
        grow_amp (float): Amplitude of the sinusoidal growth signal.
        grow_freq (float): Frequency of the sinusoidal growth signal.
        shrink_amp (float): Amplitude of the sinusoidal shrink signal.
        shrink_freq (float): Frequency of the sinusoidal shrink signal.
        motion (float): Scaling factor to control pixel-wise motion per frame.
        max_length (float): Maximum length of a microtubule in pixels.
        min_length (float): Minimum length of a microtubule in pixels.
        sigma_x (int): Gaussian blur standard deviation for drawing microtubules in the x-direction.
        sigma_y (int): Gaussian blur standard deviation for drawing microtubules in the y-direction.
        margin (int): Number of pixels to leave as a border so microtubules stay in bounds.
        num_tubulus (int): Number of microtubules to generate per series.
    """
    # ─── core video info ────────────────────────────────────
    id: int = 0
    img_size: Tuple[int, int] = (462, 462)   # (H, W)
    fps: int = 25
    num_frames: int = 50  #   167

    # ─── microtubule kinematics ────────────────────────────
    grow_amp: float = 8.0
    grow_freq: float = 0.05
    shrink_amp: float = 4.0
    shrink_freq: float = 0.25
    profile_noise: float = 0.5
    motion:     float = 2.1
    max_length: float = 150
    min_length: float = 20
    num_tubulus:int   = 20
    margin:     int   = 5
    width_var_std: float = 0.05  # std of the width variation (relative to the mean width)
    bend_amp_px: float = 2.0  # max lateral offset
    bend_prob:float = 0.25  # only ~25 % of tubules curved
    bend_phase_rand: float = 0.1   # allow ±1  0 % of the length shift

    # ─── PSF / drawing width ───────────────────────────────
    sigma_x: float = 0.5
    sigma_y: float = 0.5

    # ─── new photophysics / camera realism ─────────────────
    background_level:    float = 0.74
    gaussian_noise:      float = 0.09        # 24 / 255
    bleach_tau:          float = math.inf    # photobleaching off by default
    jitter_px:           float = 0.0
    vignetting_strength: float = 0.05
    invert_contrast: bool = True  # whether to invert the contrast of the image
    fixed_spot_density:   float = 0.0
    fixed_spot_count:     int   = 0
    fixed_spot_strength:  float = 0.05

    moving_spot_density:  float = 0.0
    moving_spot_count_mean: float = 0.0
    moving_spot_strength: float = 0.05
    moving_spot_sigma:    float = 1.0

    # ─── misc ──────────────────────────────────────────────
    generate_mask: bool = True    # still handy for training pipelines


    # ─── validation helper (optional) ─────────────────────
    def validate(self):
        assert 0 <= self.background_level <= 1, "background_level must be 0-1"
        assert 0 <= self.gaussian_noise   <= 1, "gaussian_noise must be 0-1"
        assert self.num_frames > 0 and self.fps > 0, "frames & fps must be >0"
        assert self.max_length > self.min_length > 0, "length range invalid"
        assert self.jitter_px >= 0, "jitter_px must be ≥0"


@dataclass(eq=False)
class TuningConfig(BaseConfig):
    """
    Configuration for hyperparameter tuning of synthetic microtubule data generation.

    Attributes:
        direction (str): Optimization direction ("maximize" or "minimize").
        grow_amp_range (tuple): Range of amplitudes for microtubule growth.
        grow_freq_range (tuple): Range of frequencies for microtubule growth.
        max_length_range (tuple): Range of maximum tubule lengths.
        metric (str): Metric to optimize (e.g., "cosine_similarity").
        min_length_range (tuple): Range of minimum tubule lengths.
        model_name (str): Hugging Face model name for feature extraction.
        motion_range (tuple): Range of motion magnitudes.
        num_compare_frames (int): Number of frames to compare per series.
        num_compare_series (int): Number of synthetic/reference series to compare.
        num_trials (int): Number of optimization trials.
        num_tubulus_range (tuple): Range of number of microtubules per series.
        param_file (str): Path to JSON file containing parameter ranges.
        reference_series_dir (str): Directory path with original reference video series.
        shrink_amp_range (tuple): Range of amplitudes for microtubule shrinkage.
        shrink_freq_range (tuple): Range of frequencies for microtubule shrinkage.
        sigma_range (tuple): Range of Gaussian blur standard deviations.
        snr_range (tuple): Range of signal-to-noise ratios.
    """
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_trials: int = 20
    direction: str = "maximize"
    metric: str = "cosine_similarity"
    param_file: str = "optimize_params.json"
    num_compare_series: int = 3  # Number of synthetic/reference series to compare
    reference_series_dir: str = "reference_data"  # Path to the directory containing reference video series
    num_compare_frames: int = 1  # Number of frames per series to use for comparison
    hf_cache_dir: str = None  # Directory for Hugging Face model cache
    temp_dir: str = "temp_synthetic_data"  # Temporary directory for synthetic data generation

    # Parameter ranges for tuning (only those that are tunable)
    grow_amp_range: tuple[float, float] = (0.5, 5.0)
    grow_freq_range: tuple[float, float] = (0.01, 0.2)
    shrink_amp_range: tuple[float, float] = (0.5, 8.0)
    shrink_freq_range: tuple[float, float] = (0.05, 0.5)
    motion_range: tuple[float, float] = (0.5, 5.0)
    max_length_range: tuple[int, int] = (10, 80)
    min_length_range: tuple[int, int] = (1, 30)
    snr_range: tuple[int, int] = (1, 10)
    sigma_range: tuple[float, float] = (0.5, 3.0)
    num_tubulus_range: tuple[int, int] = (3, 15)

    def validate(self):
        assert self.direction in ["maximize", "minimize"], "Direction must be either 'maximize' or 'minimize'"
        assert self.num_trials > 0, "Number of trials must be positive"
        assert self.metric in ["cosine_similarity"], "Unsupported metric type"

