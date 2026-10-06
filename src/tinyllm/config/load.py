"""YAML configuration loading with validated dotted overrides."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from tinyllm.config.schema import TinyLLMConfig


def load_config(path: Path, overrides: list[str]) -> TinyLLMConfig:
    """Load a configuration file and apply dotted YAML-scalar overrides."""
    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {path}") from error

    if not isinstance(raw_config, dict):
        raise ValueError("configuration root must be a mapping")

    for override in overrides:
        _apply_override(raw_config, override)

    return TinyLLMConfig.model_validate(raw_config)


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must use key=value syntax: {override}")

    dotted_key, raw_value = override.split("=", maxsplit=1)
    keys = dotted_key.split(".")
    if not dotted_key or any(not key for key in keys):
        raise ValueError(f"override key must be dotted and non-empty: {dotted_key}")

    current: dict[str, Any] = config
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"unknown configuration key: {dotted_key}")
        current = child

    final_key = keys[-1]
    if final_key not in current:
        raise ValueError(f"unknown configuration key: {dotted_key}")

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML override value: {dotted_key}") from error
    if isinstance(value, (Mapping, list)):
        raise ValueError("override values must be YAML scalars")
    current[final_key] = value
