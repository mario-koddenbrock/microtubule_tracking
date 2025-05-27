# Configuration for microtubule synthetic video generation
import os
from dataclasses import dataclass, asdict
import yaml
import json

from typing import Optional, Union

@dataclass
class SyntheticDataConfig:
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
        sigma (list): Gaussian blur standard deviation for drawing microtubules.
        num_series (int): Number of synthetic video/ground truth pairs to generate.
        margin (int): Number of pixels to leave as a border so microtubules stay in bounds.
        num_tubulus (int): Number of microtubules to generate per series.
    """
    id: int = 0
    img_size: tuple = (512, 512)
    fps: int = 10
    num_frames: int = 60 * 10  # 1-minute duration
    snr: int = 3
    grow_amp: float = 2.0
    grow_freq: float = 0.05
    shrink_amp: float = 4.0
    shrink_freq: float = 0.25
    motion: float = 2.1
    max_length: float = 50
    min_length: float = 5
    sigma: list = (1, 1)
    num_series: int = 3
    margin: int = 5
    num_tubulus: int = 10

    def get(self, key):
        return getattr(self, key)

    def to_yml(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.asdict(), f)

    @classmethod
    def from_yml(cls, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def update(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __str__(self):
        return yaml.dump(self.asdict(), sort_keys=False)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.asdict(), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def asdict(self):
        raw = asdict(self)
        # Convert tuples to lists for serialization
        for k, v in raw.items():
            if isinstance(v, tuple):
                raw[k] = list(v)
        return raw

    @classmethod
    def load(cls, config_path: Optional[str] = None, overrides: Optional[dict] = None):
        if config_path:

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            if config_path.endswith('.yml') or config_path.endswith('.yaml'):
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

