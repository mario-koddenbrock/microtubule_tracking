# Configuration for microtubule synthetic video generation
from dataclasses import dataclass, asdict
import yaml
import json

from typing import Optional, Union

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic microtubule video generation."""
    ID: int = 0
    IMG_SIZE: tuple = (512, 512)
    FPS: int = 10
    NUM_FRAMES: int = 60 * 10  # 1-minute duration
    SNR: int = 3
    GROW_AMP: float = 2.0
    GROW_FREQ: float = 0.05
    SHRINK_AMP: float = 4.0
    SHRINK_FREQ: float = 0.25
    MOTION: float = 2.1
    MAX_LENGTH: float = 50
    MIN_LENGTH: float = 5
    SIGMA: list = (1, 1)
    NUM_SERIES: int = 3
    MARGIN: int = 5
    NUM_TUBULUS: int = 10

    def get(self, key):
        return getattr(self, key)

    def to_yml(self, path):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)

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
        return asdict(self)

    @classmethod
    def load(cls, config_path: Optional[str] = None, overrides: Optional[dict] = None):
        if config_path:
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

CONFIG = SyntheticDataConfig.load()
