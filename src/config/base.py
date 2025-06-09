import json
import os
from abc import abstractmethod, ABC
from dataclasses import asdict, is_dataclass
from typing import Optional, get_type_hints

import yaml


class BaseConfig(ABC):
    @classmethod
    def load(cls, config_path: Optional[str] = None, overrides: Optional[dict] = None):
        """
        Loads a configuration, recursively handling nested BaseConfig objects.
        The process is:
        1. Load data from a file if provided.
        2. Apply overrides to the loaded data.
        3. Recursively construct the configuration object from the data.
        """
        data = {}
        if config_path:
            # Convert to string here to handle both str and Path objects
            config_path_str = str(config_path)

            if not os.path.exists(config_path_str):
                raise FileNotFoundError(f"Config file not found: {config_path_str}")

            with open(config_path_str, 'r') as f:
                # Use the string version for checks
                if config_path_str.endswith(('.yml', '.yaml')):
                    data = yaml.safe_load(f) or {}
                elif config_path_str.endswith('.json'):
                    data = json.load(f) or {}
                else:
                    raise ValueError("Unsupported config file format. Use .yml, .yaml, or .json")

        if overrides:
            # A simple update is fine here, as hydration happens next
            data.update(overrides)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Recursively constructs a config object from a dictionary.
        This is the core of the nested deserialization logic.
        """
        # Get the type hints for the current class
        type_hints = get_type_hints(cls)

        hydrated_data = {}
        for key, value in data.items():
            if key in type_hints:
                field_type = type_hints[key]

                # If the field is a subclass of BaseConfig and the value is a dict,
                # recursively call from_dict on the nested data.
                if isinstance(value, dict) and isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                    hydrated_data[key] = field_type.from_dict(value)
                else:
                    hydrated_data[key] = value
            # Note: Keys in `data` not in `cls` fields will cause an error at `cls(**...)`, which is desired.

        return cls(**hydrated_data)

    def update(self, params: dict):
        """
        Updates the configuration, recursively handling nested objects.
        """
        type_hints = get_type_hints(self.__class__)

        for key, value in params.items():
            if not hasattr(self, key):
                raise KeyError(f"Invalid configuration key: {key}")

            field_type = type_hints.get(key)

            # If we're updating a nested config object with a dictionary,
            # call the nested object's own update method.
            current_attr = getattr(self, key)
            if isinstance(value, dict) and isinstance(current_attr, BaseConfig):
                current_attr.update(value)
            else:
                setattr(self, key, value)

    def asdict(self):
        """
        Converts the config to a dictionary, recursively handling nested objects
        and converting tuples to lists for JSON/YAML compatibility.
        """
        return json.loads(json.dumps(asdict(self)))

    # --- The rest of the methods can be simplified or remain the same ---

    def __eq__(self, other):
        if isinstance(other, BaseConfig):
            return self.asdict() == other.asdict()
        return NotImplemented

    def get(self, key):
        return getattr(self, key)

    def __str__(self):
        return yaml.dump(self.asdict(), sort_keys=False, indent=2)

    def to_yml(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.asdict(), f, indent=2)

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.asdict(), f, indent=2)

    # These classmethods now just become simple wrappers around load()
    @classmethod
    def from_yml(cls, path):
        return cls.load(config_path=path)

    @classmethod
    def from_json(cls, path):
        return cls.load(config_path=path)

    @abstractmethod
    def validate(self):
        """
        Optionally implemented by subclasses to validate configuration logic.
        """
        pass