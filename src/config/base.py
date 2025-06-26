import copy
import json
from abc import abstractmethod, ABC
from dataclasses import asdict
from pathlib import Path
from typing import Optional, get_type_hints

import yaml


class BaseConfig(ABC):
    @classmethod
    def load(cls, config_path: Optional[str | Path] = None, overrides: Optional[dict] = None):
        """
        Loads a configuration, recursively handling nested BaseConfig objects.

        CHANGED: After loading, it stores the source file path on the instance
        so it can be saved back to the same location later.
        """
        data = {}
        if config_path:
            config_path_obj = Path(config_path)  # Use pathlib for robustness
            if not config_path_obj.is_file():
                raise FileNotFoundError(f"Config file not found: {config_path_obj}")

            with config_path_obj.open('r') as f:
                if config_path_obj.suffix.lower() in ('.yml', '.yaml'):
                    data = yaml.safe_load(f) or {}
                elif config_path_obj.suffix.lower() == '.json':
                    data = json.load(f) or {}
                else:
                    raise ValueError("Unsupported config file format. Use .yml, .yaml, or .json")

        if overrides:
            data.update(overrides)

        # Create the instance from the loaded data
        instance = cls.from_dict(data)

        # Store the source path on the instance for the new save() method
        if config_path:
            instance._source_path = config_path_obj.resolve()
        else:
            # This attribute will exist but be None if not loaded from a file
            instance._source_path = None

        return instance

    def save(self, path: Optional[str | Path] = None):
        """
        NEW: Saves the current configuration state to a file.

        If a path is provided, it saves to that path. If no path is provided,
        it attempts to save back to the original file it was loaded from.

        Raises:
            ValueError: If no path is provided and the config was not loaded
                        from a file (i.e., it was created in memory).
        """
        # 1. Determine the target path for saving
        if path:
            target_path = Path(path)
        elif hasattr(self, '_source_path') and self._source_path:
            target_path = self._source_path
        else:
            raise ValueError(
                "Cannot save config: No path was provided and the config was not loaded from a file."
            )

        # 2. Determine the format and call the appropriate writer method
        suffix = target_path.suffix.lower()
        if suffix == '.json':
            self.to_json(target_path)
        elif suffix in ['.yml', '.yaml']:
            self.to_yml(target_path)
        else:
            raise ValueError(f"Unsupported file extension: '{suffix}'. Please use .json, .yml, or .yaml.")

        # 3. After a successful save, update the internal source path to the new location
        self._source_path = target_path.resolve()
        print(f"Configuration saved to {self._source_path} ✓")

    @classmethod
    def from_dict(cls, data: dict):
        """
        Recursively constructs a config object from a dictionary.
        """
        type_hints = get_type_hints(cls)
        hydrated_data = {}
        for key, value in data.items():
            if key in type_hints:
                field_type = type_hints[key]
                if isinstance(value, dict) and isinstance(field_type, type) and issubclass(field_type, BaseConfig):
                    hydrated_data[key] = field_type.from_dict(value)
                else:
                    hydrated_data[key] = value
        return cls(**hydrated_data)

    def update(self, params: dict):
        """
        Updates the configuration, recursively handling nested objects.
        """
        type_hints = get_type_hints(self.__class__)
        for key, value in params.items():
            if not hasattr(self, key):
                raise KeyError(f"Invalid configuration key: {key}")
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

    def __eq__(self, other):
        if isinstance(other, BaseConfig):
            return self.asdict() == other.asdict()
        return NotImplemented

    def get(self, key):
        return getattr(self, key)

    def __str__(self):
        return yaml.dump(self.asdict(), sort_keys=False, indent=2)

    def to_yml(self, path):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open('w') as f:
            yaml.dump(self.asdict(), f, indent=2)

    def to_json(self, path):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open('w') as f:
            json.dump(self.asdict(), f, indent=2)

    def copy(self, deep: bool = True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

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