import csv
from pathlib import Path

import numpy as np
from PIL import Image

from cl_sam_replication.qualitative import sample_scores, save_qualitative_panels


def test_scores_are_perfect_for_identical_masks():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    assert sample_scores(mask, mask) == {
        "accuracy": 1.0, "dice": 1.0, "iou": 1.0, "biou": 1.0,
    }


def test_writes_grid_and_csv(tmp_path: Path):
    root, task = tmp_path / "data", "example"
    (root / task / "image").mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / task / "image" / "case.png")
    (root / task / "dataset.json").write_text(
        '{"test": [{"image": "image/case.png"}]}', encoding="utf-8"
    )
    mask = np.zeros((1, 8, 8), dtype=bool)
    mask[:, 2:6, 2:6] = True
    output = tmp_path / "results"
    save_qualitative_panels(root, output, task, [mask], [mask], 1)
    assert (output / task / "qualitative_grid.png").stat().st_size > 0
    with (output / task / "sample_metrics.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["dice"]) == 1.0
