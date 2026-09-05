"""Run official continual evaluation on the held-out final-test manifests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .extract_latents import _evaluation_view
from .runner import ROOT, ordered_task_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-order")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    order = ordered_task_names(config, args.task_order)
    subprocess.run([sys.executable, str(ROOT / "tools" / "verify_upstream.py")], check=True)
    paths, training, router = config["paths"], config["training"], config["router"]
    qualitative_samples = int(config.get("evaluation", {}).get("qualitative_samples_per_task", 10))
    if qualitative_samples < 0:
        raise ValueError("evaluation.qualitative_samples_per_task cannot be negative")
    if qualitative_samples and int(training.get("eval_batch_size", 1)) != 1:
        raise ValueError("Qualitative evaluation requires training.eval_batch_size: 1")
    evaluation_data = _evaluation_view(config, order)
    output = Path(paths["output_root"])
    official = output / "official"
    evaluation = output / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")

    for stage, task in enumerate(order):
        command = [
            sys.executable, "-m", "cl_sam_replication.instrumented_eval",
            "--work_dir", str(evaluation),
            "--dataset_name", task,
            "--all_datasets", ",".join(order),
            "--data_dir", str(evaluation_data),
            "--sam_checkpoint", str(paths["sam_checkpoint"]),
            "--device", str(training.get("device", "cuda:0")),
            "--model_type", str(training.get("model_type", "vit_b")),
            "--method", str(training.get("method", "cnn")),
            "--num_cnn", str(training.get("num_cnn", 3)),
            "--train_batch_size", str(training.get("train_batch_size", 1)),
            "--eval_batch_size", str(training.get("eval_batch_size", 1)),
            "--num_workers", str(training.get("num_workers", 2)),
            "--image_size", str(training.get("image_size", 1024)),
            "--mask_num", str(training.get("mask_num", 5)),
            "--metrics", "accuracy", "iou", "dice", "biou",
            "--adapters_ckpt_dir", str(official / "adapters_ckpt"),
            "--router_ckpt_dir", str(official / "vaes_ckpt"),
            "--vae_feat", str(router.get("feature", "attn_pool")),
            "--vae_in_dim", str(router.get("input_dimension", 256)),
            "--vae_latent_dim", str(router.get("latent_dimension", 64)),
            "--vae_beta", str(router.get("beta", 16.5)),
            "--router_threshold", str(router.get("threshold", 1.0)),
            "--zero_adapter_mode", str(router.get("zero_adapter_mode", "identity")),
            "--cl_matrix_csv", str(evaluation / "continual_iou.csv"),
            "--cl_matrix_biou_csv", str(evaluation / "continual_biou.csv"),
            "--skip_train_vae",
            "--qualitative_samples", str(qualitative_samples),
        ]
        print(f"=== Final evaluation stage {stage + 1}/{len(order)}: {task} ===")
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT, env=environment)


if __name__ == "__main__":
    main()
