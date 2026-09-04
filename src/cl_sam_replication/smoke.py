"""Run a bounded end-to-end GPU smoke test with the official trainer."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from .config import load_config
from .runner import ordered_task_names, run
from .validate import validate_all


def _smoke_view(config, order, samples: int) -> Path:
    source_root = Path(config["paths"]["prepared_data_root"])
    output_root = Path(config["paths"]["output_root"])
    view_root = output_root / "smoke_data"
    for task in order:
        source, destination = source_root / task, view_root / task
        destination.mkdir(parents=True, exist_ok=True)
        for folder in ("image", "label", "imask"):
            link = destination / folder
            if not link.exists():
                try:
                    link.symlink_to(source / folder, target_is_directory=True)
                except OSError:
                    shutil.copytree(source / folder, link, dirs_exist_ok=True)
        manifest = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
        manifest["training"] = manifest["training"][:samples]
        manifest["test"] = manifest["test"][: max(1, min(samples, len(manifest["test"])))]
        manifest["numTraining"] = len(manifest["training"])
        (destination / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return view_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_all(config)
    order = ordered_task_names(config)[: args.tasks]
    smoke_config = copy.deepcopy(config)
    smoke_config["paths"]["prepared_data_root"] = str(_smoke_view(config, order, args.samples))
    smoke_config["paths"]["output_root"] = str(Path(config["paths"]["output_root"]) / "smoke")
    smoke_config["training"].update({"epochs": 1, "train_batch_size": 1, "eval_batch_size": 1, "mask_num": 1})
    smoke_config["router"].update({"epochs": 1, "k_folds": 2})
    run(smoke_config, order, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
