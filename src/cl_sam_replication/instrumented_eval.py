"""Run official routed evaluation with detached pixel-accuracy reporting."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    upstream = root / "upstream" / "ca_sam_official"
    sys.path.insert(0, str(upstream))
    import metrics as official_metrics  # type: ignore

    original = official_metrics.SegMetrics

    def instrumented(pred, label, metric_names):
        names = [metric_names] if isinstance(metric_names, str) else list(metric_names)
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


if __name__ == "__main__":
    main()

