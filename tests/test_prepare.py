import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse

from cl_sam_replication.prepare import prepare_task


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

