"""Run the official trainer with read-only pixel-accuracy instrumentation."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    upstream = repository_root / "upstream" / "ca_sam_official"
    sys.path.insert(0, str(upstream))

    import metrics as official_metrics  # type: ignore

    original = official_metrics.SegMetrics

    def instrumented(pred, label, metric_names):
        names = [metric_names] if isinstance(metric_names, str) else list(metric_names)
        values: list[float] = []
        for name in names:
            if name in {"accuracy", "acc", "pixel_accuracy"}:
                prediction = pred if torch.is_tensor(pred) else torch.as_tensor(np.asarray(pred))
                target = label if torch.is_tensor(label) else torch.as_tensor(np.asarray(label))
                values.append(float((prediction.bool() == target.bool()).float().mean().item()))
            else:
                values.append(float(original(pred, label, [name])[0]))
        return np.asarray(values, dtype=np.float64)

    # The patch adds a detached reporting metric only. Model, losses, gradients,
    # optimization, routing, and checkpoints remain the official implementation.
    official_metrics.SegMetrics = instrumented
    runpy.run_path(str(upstream / "train_align_CL_VAE.py"), run_name="__main__")


if __name__ == "__main__":
    main()

