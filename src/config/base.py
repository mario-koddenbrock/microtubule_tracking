import json
import os
from abc import abstractmethod, ABC
from dataclasses import asdict
from typing import Optional

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
