"""Qualitative segmentation panels built from official evaluator predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion


def sample_scores(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, truth = prediction.astype(bool), target.astype(bool)
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    denominator = pred.sum() + truth.sum()
    pred_boundary = np.logical_xor(pred, binary_erosion(pred, structure=np.ones((3, 3)), border_value=0))
    truth_boundary = np.logical_xor(truth, binary_erosion(truth, structure=np.ones((3, 3)), border_value=0))
    tolerance = max(1, round(0.02 * np.hypot(*truth.shape[-2:])))
    if tolerance % 2 == 0:
        tolerance += 1
    structure = np.ones((tolerance, tolerance), dtype=bool)
    pred_boundary = binary_dilation(pred_boundary, structure=structure)
    truth_boundary = binary_dilation(truth_boundary, structure=structure)
    boundary_union = np.logical_or(pred_boundary, truth_boundary).sum()
    boundary_intersection = np.logical_and(pred_boundary, truth_boundary).sum()
    return {
        "accuracy": float((pred == truth).mean()),
        "dice": float((2 * intersection) / denominator) if denominator else 1.0,
        "iou": float(intersection / union) if union else 1.0,
        "biou": float(boundary_intersection / boundary_union) if boundary_union else 1.0,
    }


def save_qualitative_panels(
    data_root: Path,
    output_root: Path,
    task: str,
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    limit: int,
) -> None:
    manifest = json.loads((data_root / task / "dataset.json").read_text(encoding="utf-8"))
    records = manifest["test"]
    count = min(limit, len(records), len(predictions), len(targets))
    if count <= 0:
        return
    destination = output_root / task
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    figure, axes = plt.subplots(count, 4, figsize=(16, 4 * count), squeeze=False)
    for index in range(count):
        record = records[index]
        image_path = data_root / task / record["image"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        pred = np.squeeze(predictions[index]).astype(bool)
        truth = np.squeeze(targets[index]).astype(bool)
        scores = sample_scores(pred, truth)
        sample_id = image_path.stem
        rows.append({"sample_id": sample_id, **scores})
        display_size = (image.shape[1], image.shape[0])
        gt_view = np.asarray(Image.fromarray(truth.astype(np.uint8) * 255).resize(display_size, Image.Resampling.NEAREST))
        pred_view = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize(display_size, Image.Resampling.NEAREST))
        overlay = image.copy()
        overlay[gt_view > 0] = (0.55 * overlay[gt_view > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        overlay[pred_view > 0] = (0.55 * overlay[pred_view > 0] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        panels = (image, gt_view, pred_view, overlay)
        titles = ("Input image", "Ground truth", "Prediction", "Overlay: GT green, pred red")
        for axis, panel, title in zip(axes[index], panels, titles):
            axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axis.axis("off")
            axis.set_title(title)
        axes[index, 0].set_ylabel(
            f"{sample_id}\nDice {scores['dice']:.4f} | IoU {scores['iou']:.4f}\n"
            f"BIoU {scores['biou']:.4f} | Acc {scores['accuracy']:.4f}",
            fontsize=9,
        )
    figure.suptitle(f"{task}: held-out qualitative segmentation", fontsize=15)
    figure.tight_layout()
    figure.savefig(destination / "qualitative_grid.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    with (destination / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "accuracy", "dice", "iou", "biou"])
        writer.writeheader()
        writer.writerows(rows)
