"""Parse official epoch logs and create reproducible history files and curves."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"\[Epoch (?P<epoch>\d+)\].*?"
    r"Train loss=(?P<train_loss>[0-9.eE+-]+)\s+"
    r"train_accuracy=(?P<train_accuracy>[0-9.eE+-]+)\s+"
    r"train_iou=(?P<train_miou>[0-9.eE+-]+)\s+"
    r"train_dice=(?P<train_dice>[0-9.eE+-]+)\s+"
    r"train_biou=(?P<train_biou>[0-9.eE+-]+)\s+\|\s+"
    r"Test loss=(?P<val_loss>[0-9.eE+-]+)\s+"
    r"test_accuracy=(?P<val_accuracy>[0-9.eE+-]+)\s+"
    r"test_iou=(?P<val_miou>[0-9.eE+-]+)\s+"
    r"test_dice=(?P<val_dice>[0-9.eE+-]+)\s+"
    r"test_biou=(?P<val_biou>[0-9.eE+-]+)"
)


def parse_history(text: str) -> list[dict[str, float | int]]:
    history = []
    for match in EPOCH_PATTERN.finditer(text):
        row: dict[str, float | int] = {"epoch": int(match.group("epoch"))}
        row.update({key: float(value) for key, value in match.groupdict().items() if key != "epoch"})
        history.append(row)
    return history


def save_history(log_path: Path, output_dir: Path) -> list[dict[str, float | int]]:
    history = parse_history(log_path.read_text(encoding="utf-8", errors="replace"))
    if not history:
        raise ValueError(f"No instrumented epoch rows found in {log_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Pixel accuracy", title="Training and validation accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    save_history(Path(args.log), Path(args.output_dir))


if __name__ == "__main__":
    main()

