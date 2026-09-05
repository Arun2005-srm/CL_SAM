"""Convert paired 2D masks into the official CA-SAM dataset contract."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage, sparse

from .config import load_config
from .discovery import SamplePair, discover_pairs


def split_pairs(
    pairs: list[SamplePair], split_cfg: dict[str, Any]
) -> dict[str, list[SamplePair]]:
    seed = int(split_cfg.get("seed", 42))
    limit = split_cfg.get("max_samples")
    ratios = split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    rng = random.Random(seed)

    units: list[list[SamplePair]]
    if split_cfg.get("group_aware", False) and any(pair.group for pair in pairs):
        groups: dict[str, list[SamplePair]] = {}
        for pair in pairs:
            groups.setdefault(pair.group or pair.sample_id, []).append(pair)
        units = list(groups.values())
    else:
        units = [[pair] for pair in pairs]
    rng.shuffle(units)
    ordered = [pair for unit in units for pair in unit]
    if limit is not None:
        ordered = ordered[: int(limit)]
        units = [[pair] for pair in ordered]

    total = sum(len(unit) for unit in units)
    targets = {
        "train": round(total * float(ratios["train"])),
        "val": round(total * float(ratios["val"])),
    }
    result: dict[str, list[SamplePair]] = {"train": [], "val": [], "test": []}
    for unit in units:
        if len(result["train"]) < targets["train"]:
            destination = "train"
        elif len(result["val"]) < targets["val"]:
            destination = "val"
        else:
            destination = "test"
        result[destination].extend(unit)
    return result


def _label_tensor(mask: np.ndarray, task: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    if mask.ndim == 3:
        mask = mask[..., 0]
    label_cfg = task.get("labels", {})
    mode = label_cfg.get("mode", "binary")
    if mode == "binary":
        threshold = float(label_cfg.get("threshold", 0))
        tensor = (mask > threshold).astype(np.uint8)[None, ..., None]
        return tensor, [str(label_cfg.get("foreground_name", "foreground"))]
    if mode != "multiclass":
        raise ValueError(f"Unsupported label mode {mode!r}")

    class_map = label_cfg.get("classes")
    if not class_map:
        values = sorted(int(value) for value in np.unique(mask) if int(value) != 0)
        class_map = {str(value): f"class_{value}" for value in values}
    values = [int(value) for value in class_map]
    tensor = np.stack([(mask == value).astype(np.uint8) for value in values], axis=0)[..., None]
    return tensor, [str(class_map[str(value)]) for value in values]


def _pseudo_mask(labels: np.ndarray) -> np.ndarray:
    pseudo = np.full(labels.shape[1:3], -1, dtype=np.int32)
    next_id = 0
    for channel in labels[..., 0]:
        components, count = ndimage.label(channel > 0)
        for component in range(1, count + 1):
            pseudo[components == component] = next_id
            next_id += 1
    # The official loader converts pseudo masks to float32 after loading, so
    # retaining an unnecessarily wide integer dtype only wastes disk space.
    if next_id <= np.iinfo(np.int8).max:
        return pseudo.astype(np.int8)
    if next_id <= np.iinfo(np.int16).max:
        return pseudo.astype(np.int16)
    return pseudo


def _safe_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _materialize_image(source: Path, destination: Path, mode: str = "auto") -> str:
    """Materialize an immutable image without decoding or re-encoding it.

    In auto mode a hard link is preferred. It consumes no second copy of the
    image data, survives deletion of the source path, and is ideal when both
    directories are on Colab's local filesystem. Cross-device and unsupported
    filesystems transparently fall back to a regular metadata-preserving copy.
    """
    if mode not in {"auto", "hardlink", "copy"}:
        raise ValueError(f"Unsupported image_transfer mode: {mode!r}")
    if destination.exists():
        destination.unlink()
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def prepare_task(task: dict[str, Any], output_root: Path) -> dict[str, Any]:
    pairs = discover_pairs(task)
    splits = split_pairs(pairs, task.get("split", {}))
    print(
        f"[{task['name']}] discovered {len(pairs)} pairs; "
        f"split sizes: "
        + ", ".join(f"{name}={len(items)}" for name, items in splits.items()),
        flush=True,
    )
    task_root = output_root / task["name"]
    image_root = task_root / "image"
    label_root = task_root / "label"
    pseudo_root = task_root / "imask"
    for directory in (image_root, label_root, pseudo_root):
        directory.mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict[str, str]]] = {name: [] for name in splits}
    skipped: list[dict[str, str]] = []
    foreground_pixels = 0
    total_pixels = 0
    class_names: list[str] | None = None
    dimensions: Counter[str] = Counter()
    image_transfers: Counter[str] = Counter()

    for split_name, split_pairs_list in splits.items():
        print(f"[{task['name']}] preparing {split_name}...", flush=True)
        for pair_index, pair in enumerate(split_pairs_list, start=1):
            with Image.open(pair.mask) as source_mask:
                mask = np.asarray(source_mask).copy()
            prepare_size = task.get("prepare_size")
            prepared_image = None
            if prepare_size:
                target_height, target_width = (int(value) for value in prepare_size)
                with Image.open(pair.image) as source_image:
                    prepared_image = source_image.convert("RGB").resize(
                        (target_width, target_height), resample=Image.Resampling.NEAREST
                    )
                mask = np.asarray(
                    Image.fromarray(mask).resize(
                        (target_width, target_height), resample=Image.Resampling.NEAREST
                    )
                ).copy()
                image_width, image_height = target_width, target_height
            else:
                with Image.open(pair.image) as source_image:
                    source_image.verify()
                    image_width, image_height = source_image.size
            labels, current_names = _label_tensor(mask, task)
            if class_names is None:
                class_names = current_names
            elif class_names != current_names:
                raise ValueError(f"Inconsistent classes in task {task['name']!r}")
            if not labels.any():
                policy = task.get("empty_mask_policy", "skip")
                if policy == "error":
                    raise ValueError(f"Empty mask: {pair.mask}")
                if policy == "skip":
                    skipped.append({"sample_id": pair.sample_id, "reason": "empty_mask"})
                    continue

            sample_id = _safe_id(pair.sample_id)
            # Preserve the encoded source image. Re-encoding large JPEG datasets
            # such as ISIC as PNG is unnecessarily slow and can multiply storage.
            image_suffix = ".jpg" if prepared_image is not None else (pair.image.suffix.casefold() or ".png")
            image_rel = Path("image") / f"{sample_id}{image_suffix}"
            if prepared_image is not None:
                prepared_image.save(task_root / image_rel, format="JPEG", quality=95)
                transfer = "resized"
            else:
                transfer = _materialize_image(
                    pair.image,
                    task_root / image_rel,
                    mode=str(task.get("image_transfer", "auto")),
                )
            image_transfers[transfer] += 1
            shape = tuple(int(value) for value in labels.shape)
            label_rel = Path("label") / f"{sample_id}.{shape}.npz"
            sparse.save_npz(task_root / label_rel, sparse.csr_matrix(labels.reshape(1, -1)))
            record = {"image": image_rel.as_posix(), "label": label_rel.as_posix()}
            if split_name == "train":
                pseudo_rel = Path("imask") / f"{sample_id}.npy"
                np.save(task_root / pseudo_rel, _pseudo_mask(labels))
                record["imask"] = pseudo_rel.as_posix()
            records[split_name].append(record)
            foreground_pixels += int(labels.any(axis=0).sum())
            total_pixels += int(labels.shape[1] * labels.shape[2])
            dimensions[f"{image_height}x{image_width}"] += 1
            if pair_index % 100 == 0 or pair_index == len(split_pairs_list):
                print(
                    f"[{task['name']}] {split_name}: "
                    f"{pair_index}/{len(split_pairs_list)}",
                    flush=True,
                )

    if not records["train"] or not records["val"] or not records["test"]:
        raise ValueError(
            f"Task {task['name']!r} has an empty split after preparation: "
            f"{ {name: len(items) for name, items in records.items()} }"
        )
    class_names = class_names or ["foreground"]
    labels_json = {"0": "background"}
    labels_json.update({str(index + 1): name for index, name in enumerate(class_names)})
    common = {
        "name": task["name"],
        "description": task.get("description", "Prepared by CL_SAM"),
        "dimension": "2D",
        "modality": {"0": task.get("modality", "unknown")},
        "labels": labels_json,
        "numTraining": len(records["train"]),
        "training": records["train"],
    }
    # The official trainer names this split `test`; we deliberately feed validation here.
    training_manifest = {**common, "test": records["val"]}
    final_manifest = {**common, "test": records["test"]}
    (task_root / "dataset.json").write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
    (task_root / "dataset.final_test.json").write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    stats = {
        "task": task["name"],
        "discovered": len(pairs),
        "prepared": {name: len(items) for name, items in records.items()},
        "skipped": skipped,
        "classes": class_names,
        "foreground_fraction": foreground_pixels / max(total_pixels, 1),
        "image_dimensions": dict(dimensions),
        "image_transfers": dict(image_transfers),
    }
    (task_root / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"[{task['name']}] complete: "
        + ", ".join(f"{name}={len(items)}" for name, items in records.items())
        + f", skipped={len(skipped)}, image_transfer={dict(image_transfers)}",
        flush=True,
    )
    return stats


def prepare_all(config: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = Path(config["paths"]["prepared_data_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [prepare_task(task, output_root) for task in config["tasks"]]
    (output_root / "dataset_statistics.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summaries = prepare_all(load_config(args.config))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
