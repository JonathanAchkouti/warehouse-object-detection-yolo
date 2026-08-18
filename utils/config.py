from pathlib import Path
from typing import Any

import yaml


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an experiment override into a base configuration."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML, resolving an optional relative ``base`` configuration."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    base_path = config.pop("base", None)
    if base_path is not None:
        if not isinstance(base_path, str):
            raise ValueError(f"{path} base must be a relative path string")
        base_config = load_config(path.parent / base_path)
        config = merge_config(base_config, config)
    return config
