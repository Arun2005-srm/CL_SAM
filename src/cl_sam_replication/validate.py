"""Validate prepared datasets before expensive GPU execution."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import sparse

from .config import load_config


def validate_task(root: Path, task_name: str) -> dict[str, Any]:
    task_root = root / task_name
    manifest_path = task_root / "dataset.json"
    final_path = task_root / "dataset.final_test.json"
    if not manifest_path.is_file() or not final_path.is_file():
        raise FileNotFoundError(f"Missing manifests for {task_name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    train_ids = {item["image"] for item in manifest["training"]}
    val_ids = {item["image"] for item in manifest["test"]}
    test_ids = {item["image"] for item in final["test"]}
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise ValueError(f"Split leakage detected in {task_name}")

    checked = 0
    for split, items in (("train", manifest["training"]), ("val", manifest["test"]), ("test", final["test"])):
        for item in items:
            image_path, label_path = task_root / item["image"], task_root / item["label"]
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing prepared sample in {task_name}/{split}: {item}")
            with Image.open(image_path) as image:
                image.verify()
            shape = ast.literal_eval(label_path.name.split(".")[-2])
            label = sparse.load_npz(label_path).toarray().reshape(shape)
            if label.ndim != 4 or label.shape[-1] != 1 or not label.any():
                raise ValueError(f"Invalid CA-SAM label {label_path}")
            if split == "train":
                pseudo_path = task_root / item.get("imask", "")
                if not pseudo_path.is_file():
                    raise FileNotFoundError(f"Missing pseudo mask for {image_path}")
                pseudo = np.load(pseudo_path)
                if pseudo.shape != label.shape[1:3] or not np.any(pseudo >= 0):
                    raise ValueError(f"Invalid pseudo mask {pseudo_path}")
            checked += 1
    return {
        "task": task_name, "checked": checked,
        "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
    }


def validate_all(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["paths"]["prepared_data_root"])
    reports = [validate_task(root, task["name"]) for task in config["tasks"]]
    destination = Path(config["paths"]["output_root"]) / "data_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_all(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
