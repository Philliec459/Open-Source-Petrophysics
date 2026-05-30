from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def _load_yaml_file(filename: str) -> Dict[str, Any]:
    path = _config_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")

    return data


def load_curve_families_config() -> Dict[str, list[str]]:
    data = _load_yaml_file("curve_families.yaml")
    families = data.get("families", {})
    if not isinstance(families, dict):
        raise ValueError("curve_families.yaml: 'families' must be a mapping")
    return families


def load_curve_priorities_config() -> Dict[str, list[str]]:
    data = _load_yaml_file("curve_priorities.yaml")
    priorities = data.get("priorities", {})
    if not isinstance(priorities, dict):
        raise ValueError("curve_priorities.yaml: 'priorities' must be a mapping")
    return priorities


def load_curve_sets_config() -> Dict[str, dict]:
    data = _load_yaml_file("curve_sets.yaml")
    sets_cfg = data.get("sets", {})
    if not isinstance(sets_cfg, dict):
        raise ValueError("curve_sets.yaml: 'sets' must be a mapping")
    return sets_cfg