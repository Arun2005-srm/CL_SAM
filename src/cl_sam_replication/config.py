"""Load and validate experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular config inheritance involving {path}")
    seen.add(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = payload.pop("defaults", None)
    if parent:
        payload = _merge(_read((path.parent / parent).resolve(), seen), payload)
    seen.remove(path)
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = _read(config_path, set())
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"experiment", "paths", "tasks", "training", "router", "analysis"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    tasks = config["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    names = [task.get("name") for task in tasks]
    if any(not name for name in names):
        raise ValueError("Every task requires a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("Task names must be unique")

    for task in tasks:
        source = task.get("source", {})
        if not source.get("collections") and not (
            source.get("image_dir") and source.get("mask_dir")
        ):
            raise ValueError(
                f"Task {task['name']!r} requires source.image_dir/mask_dir "
                "or source.collections"
            )
        ratios = task.get("split", {}).get(
            "ratios", {"train": 0.7, "val": 0.15, "test": 0.15}
        )
        if set(ratios) != {"train", "val", "test"}:
            raise ValueError(f"Task {task['name']!r} needs train/val/test ratios")
        if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-8:
            raise ValueError(f"Split ratios for {task['name']!r} must sum to 1")


def task_names(config: dict[str, Any]) -> list[str]:
    """Return task order exactly as declared in the configuration."""
    return [str(task["name"]) for task in config["tasks"]]

