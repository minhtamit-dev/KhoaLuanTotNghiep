"""YAML Configuration loader and accessor."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class Config(dict):
    """Dictionary subclass that allows dot-notation access to nested dictionaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = Config(value) if isinstance(value, dict) else value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config object back to standard dictionary."""
        result = {}
        for k, v in self.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def load_config(config_path: Union[str, Path]) -> Config:
    """Load a YAML configuration file into a Config object.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config: Populated Config object.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config(data)


def merge_configs(base_config: Config, override_config: Union[Config, Dict[str, Any]]) -> Config:
    """Recursively merge override_config into base_config.

    Args:
        base_config: Base configuration object.
        override_config: Configuration with overriding keys.

    Returns:
        Config: New merged Config object.
    """
    merged = Config(base_config.to_dict())
    
    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], (dict, Config))
            and isinstance(value, (dict, Config))
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = Config(value) if isinstance(value, dict) else value

    return merged
