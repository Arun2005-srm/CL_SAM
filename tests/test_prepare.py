import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse

from cl_sam_replication.prepare import _materialize_image, prepare_task
from cl_sam_replication.discovery import discover_pairs


def test_generic_folder_preparation(tmp_path: Path):
    images, masks = tmp_path / "source_images", tmp_path / "source_masks"
    images.mkdir()
    masks.mkdir()
    for index in range(20):
        image = np.full((16, 12, 3), index, dtype=np.uint8)
        mask = np.zeros((16, 12), dtype=np.uint8)
        mask[2:8, 3:9] = 255
        Image.fromarray(image).save(images / f"case_{index:02d}.png")
        Image.fromarray(mask).save(masks / f"case_{index:02d}_mask.png")
    task = {
        "name": "synthetic",
        "source": {
            "image_dir": str(images), "mask_dir": str(masks),
            "mask_matching": {"strip_suffixes": ["_mask"]},
        },
        "labels": {"mode": "binary", "foreground_name": "object"},
        "split": {"seed": 2, "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
    }
    stats = prepare_task(task, tmp_path / "prepared")
    root = tmp_path / "prepared" / "synthetic"
    manifest = json.loads((root / "dataset.json").read_text())
    final_manifest = json.loads((root / "dataset.final_test.json").read_text())
    assert stats["prepared"] == {"train": 14, "val": 3, "test": 3}
    assert len(manifest["test"]) == 3
    assert len(final_manifest["test"]) == 3
    label_path = root / manifest["training"][0]["label"]
    shape = eval(label_path.name.split(".")[-2], {"__builtins__": {}})
    assert sparse.load_npz(label_path).toarray().reshape(shape).shape == (1, 16, 12, 1)
    pseudo = np.load(root / manifest["training"][0]["imask"])
    assert pseudo.min() == -1
    assert pseudo.max() >= 0
    assert stats["image_transfers"].get("hardlink", 0) == 20


def test_copy_transfer_mode(tmp_path: Path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"immutable-image")
    assert _materialize_image(source, destination, mode="copy") == "copy"
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino


def test_mask_resize_preserves_source_image_and_compacts_pseudo(tmp_path: Path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for index in range(10):
        Image.fromarray(np.full((30, 40, 3), 100, dtype=np.uint8)).save(images / f"{index}.jpg")
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[4:20, 5:25] = 255
        Image.fromarray(mask).save(masks / f"{index}.png")
    task = {
        "name": "large_source", "mask_prepare_size": [16, 16],
        "source": {"image_dir": str(images), "mask_dir": str(masks)},
        "labels": {"mode": "binary"},
        "split": {"seed": 1, "ratios": {"train": 0.6, "val": 0.2, "test": 0.2}},
    }
    stats = prepare_task(task, tmp_path / "prepared")
    root = tmp_path / "prepared" / "large_source"
    manifest = json.loads((root / "dataset.json").read_text())
    with Image.open(root / manifest["training"][0]["image"]) as prepared:
        assert prepared.size == (40, 30)
    label_path = root / manifest["training"][0]["label"]
    shape = eval(label_path.name.split(".")[-2], {"__builtins__": {}})
    assert shape == (1, 16, 16, 1)
    pseudo = np.load(root / manifest["training"][0]["imask"])
    assert pseudo.shape == (16, 16)
    assert pseudo.dtype == np.int8
    assert stats["image_transfers"] == {"hardlink": 10}


def test_explicit_unpaired_skip_policy(tmp_path: Path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for name in ("paired", "unmatched"):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / f"{name}.png")
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(masks / "paired.png")
    task = {
        "name": "source_with_known_gap",
        "source": {
            "image_dir": str(images), "mask_dir": str(masks),
            "unpaired_policy": "skip",
        },
    }
    pairs = discover_pairs(task)
    assert [pair.sample_id for pair in pairs] == ["paired"]
