"""Run official routed evaluation with detached pixel-accuracy reporting."""

from __future__ import annotations

import runpy
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .qualitative import save_qualitative_panels


def main() -> None:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--qualitative_samples", type=int, default=0)
    known, remaining = custom.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    root = Path(__file__).resolve().parents[2]
    upstream = root / "upstream" / "ca_sam_official"
    sys.path.insert(0, str(upstream))
    import metrics as official_metrics  # type: ignore

    original = official_metrics.SegMetrics
    captured_predictions: list[np.ndarray] = []
    captured_targets: list[np.ndarray] = []

    def instrumented(pred, label, metric_names):
        names = [metric_names] if isinstance(metric_names, str) else list(metric_names)
        prediction_array = (pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred))
        target_array = (label.detach().cpu().numpy() if torch.is_tensor(label) else np.asarray(label))
        captured_predictions.extend(list(prediction_array))
        captured_targets.extend(list(target_array))
        values = []
        for name in names:
            if name in {"accuracy", "acc", "pixel_accuracy"}:
                prediction = pred if torch.is_tensor(pred) else torch.as_tensor(np.asarray(pred))
                target = label if torch.is_tensor(label) else torch.as_tensor(np.asarray(label))
                values.append(float((prediction.bool() == target.bool()).float().mean().item()))
            else:
                values.append(float(original(pred, label, [name])[0]))
        return np.asarray(values)

    official_metrics.SegMetrics = instrumented
    runpy.run_path(str(upstream / "eval_vae_router_load_adapter.py"), run_name="__main__")
    if known.qualitative_samples > 0:
        def argument(name: str) -> str:
            index = remaining.index(name)
            return remaining[index + 1]
        data_root = Path(argument("--data_dir"))
        output_root = Path(argument("--work_dir")) / "qualitative"
        order = argument("--all_datasets").split(",")
        current = argument("--dataset_name")
        offset = 0
        for task in order[: order.index(current) + 1]:
            records = json.loads((data_root / task / "dataset.json").read_text(encoding="utf-8"))["test"]
            end = offset + len(records)
            save_qualitative_panels(
                data_root, output_root, task,
                captured_predictions[offset:end], captured_targets[offset:end],
                known.qualitative_samples,
            )
            offset = end


if __name__ == "__main__":
    main()
