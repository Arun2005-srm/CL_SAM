"""Configurable image/mask discovery without dataset-specific model code."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SamplePair:
    sample_id: str
    image: Path
    mask: Path
    group: str | None = None


def _key(path: Path, root: Path, rules: dict[str, Any]) -> str:
    mode = rules.get("key", "stem")
    value = path.relative_to(root).with_suffix("").as_posix() if mode == "relative" else path.stem
    for suffix in rules.get("strip_suffixes", []):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    pattern = rules.get("regex")
    if pattern:
        value = re.sub(pattern, rules.get("replacement", ""), value)
    return value.casefold() if rules.get("case_insensitive", True) else value


def _files(root: Path, pattern: str) -> list[Path]:
    return sorted(
        path for path in root.glob(pattern)
        if path.is_file() and path.suffix.casefold() in RASTER_EXTENSIONS
    )


def _discover_collection(
    image_dir: Path,
    mask_dir: Path,
    source: dict[str, Any],
    group: str | None,
) -> list[SamplePair]:
    image_paths = _files(image_dir, source.get("image_glob", "**/*"))
    mask_paths = _files(mask_dir, source.get("mask_glob", "**/*"))
    image_rules = source.get("image_matching", source.get("matching", {}))
    mask_rules = source.get("mask_matching", source.get("matching", {}))

    images = {_key(path, image_dir, image_rules): path for path in image_paths}
    masks = {_key(path, mask_dir, mask_rules): path for path in mask_paths}
    duplicate_images = len(images) != len(image_paths)
    duplicate_masks = len(masks) != len(mask_paths)
    if duplicate_images or duplicate_masks:
        raise ValueError(
            f"Duplicate pairing keys found under {image_dir} or {mask_dir}; "
            "use relative keys or a more specific matching rule"
        )

    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        policy = source.get("unpaired_policy", "error")
        message = (
            f"Unpaired data in {group or image_dir.name}: "
            f"{len(missing_masks)} images lack masks and "
            f"{len(missing_images)} masks lack images. Examples: "
            f"{(missing_masks + missing_images)[:5]}"
        )
        if policy == "error":
            raise ValueError(message)
        if policy != "skip":
            raise ValueError(f"Unsupported unpaired_policy {policy!r}")
        print(f"[WARN] {message} Skipping unmatched files.", flush=True)

    prefix = f"{group}/" if group else ""
    paired_keys = images.keys() & masks.keys()
    return [
        SamplePair(prefix + key, images[key], masks[key], group)
        for key in sorted(paired_keys)
    ]


def discover_pairs(task: dict[str, Any]) -> list[SamplePair]:
    source = task["source"]
    collections = source.get("collections")
    if not collections:
        collections = [{
            "name": None,
            "image_dir": source["image_dir"],
            "mask_dir": source["mask_dir"],
        }]

    pairs: list[SamplePair] = []
    for collection in collections:
        merged = {**source, **collection}
        pairs.extend(
            _discover_collection(
                Path(merged["image_dir"]),
                Path(merged["mask_dir"]),
                merged,
                collection.get("name"),
            )
        )
    if not pairs:
        raise ValueError(f"No paired samples found for task {task['name']!r}")
    return pairs
