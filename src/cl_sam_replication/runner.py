"""Configuration-driven sequential launcher for the untouched CA-SAM trainer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .history import EPOCH_PATTERN, save_history


ROOT = Path(__file__).resolve().parents[2]


def ordered_task_names(
    config: dict[str, Any], override: str | None = None, shuffle: bool = False, seed: int | None = None
) -> list[str]:
    available = [str(task["name"]) for task in config["tasks"]]
    configured = config["experiment"].get("task_order", available)
    order = [part.strip() for part in override.split(",")] if override else list(configured)
    if len(order) != len(set(order)) or set(order) != set(available):
        raise ValueError(
            "Task order must contain every configured task exactly once. "
            f"Configured={available}, received={order}"
        )
    if shuffle:
        random.Random(seed if seed is not None else config["experiment"].get("seed", 42)).shuffle(order)
    return order


def _value(config: dict[str, Any], section: str, name: str, default: Any) -> Any:
    return config.get(section, {}).get(name, default)


def build_train_command(config: dict[str, Any], task: str, order: list[str], stage: int) -> list[str]:
    paths, training, router = config["paths"], config["training"], config["router"]
    output_root = Path(paths["output_root"])
    work_dir = output_root / "official"
    run_name = f"T{stage + 1:02d}_{task}_cnn{training.get('num_cnn', 3)}"
    command = [
        sys.executable, "-m", "cl_sam_replication.instrumented_train",
        "--work_dir", str(work_dir),
        "--save_root", str(work_dir),
        "--run_name", run_name,
        "--dataset_name", task,
        "--all_datasets", ",".join(order),
        "--data_dir", str(paths["prepared_data_root"]),
        "--sam_checkpoint", str(paths["sam_checkpoint"]),
        "--device", str(training.get("device", "cuda:0")),
        "--model_type", str(training.get("model_type", "vit_b")),
        "--method", str(training.get("method", "cnn")),
        "--num_cnn", str(training.get("num_cnn", 3)),
        "--epochs", str(training.get("epochs", 24)),
        "--lr", str(training.get("learning_rate", 1e-4)),
        "--train_batch_size", str(training.get("train_batch_size", 1)),
        "--eval_batch_size", str(training.get("eval_batch_size", 1)),
        "--num_workers", str(training.get("num_workers", 2)),
        "--image_size", str(training.get("image_size", 1024)),
        "--mask_num", str(training.get("mask_num", 5)),
        "--metrics", "accuracy", "iou", "dice", "biou",
        "--router_type", str(router.get("type", "vae")),
        "--router_threshold", str(router.get("threshold", 1.0)),
        "--zero_adapter_mode", str(router.get("zero_adapter_mode", "identity")),
        "--vae_feat", str(router.get("feature", "attn_pool")),
        "--vae_in_dim", str(router.get("input_dimension", 256)),
        "--vae_latent_dim", str(router.get("latent_dimension", 64)),
        "--vae_beta", str(router.get("beta", 16.5)),
        "--vae_epochs", str(router.get("epochs", 10)),
        "--vae_lr", str(router.get("learning_rate", 5e-4)),
        "--k_folds", str(router.get("k_folds", 5)),
        "--fold_seed", str(router.get("fold_seed", 2025)),
        "--adapters_ckpt_dir", str(work_dir / "adapters_ckpt"),
        "--router_ckpt_dir", str(work_dir / "vaes_ckpt"),
    ]
    if training.get("multimask", False):
        command.append("--multimask")
    return command


def run(config: dict[str, Any], order: list[str], dry_run: bool = False, start_at: int = 0) -> None:
    subprocess.run([sys.executable, str(ROOT / "tools" / "verify_upstream.py")], check=True)
    output_root = Path(config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    source_config = Path(config["_config_path"])
    shutil.copy2(source_config, output_root / "config.snapshot.yaml")
    metadata = {
        "task_order": order,
        "python": sys.version,
        "platform": platform.platform(),
        "upstream_revision": "9a4ee0f71e264343719a42027d71099ea7ccb8d9",
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for stage, task in enumerate(order[start_at:], start=start_at):
        run_name = f"T{stage + 1:02d}_{task}_cnn{config['training'].get('num_cnn', 3)}"
        command = build_train_command(config, task, order, stage)
        print(f"\n=== Stage {stage + 1}/{len(order)}: {task} ===", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not dry_run:
            environment = os.environ.copy()
            source_path = str(ROOT / "src")
            environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                match = EPOCH_PATTERN.search(line)
                if match:
                    values = match.groupdict()
                    print(
                        f"Epoch [{values['epoch']}/{config['training'].get('epochs', 24)}] | "
                        f"train acc={values['train_accuracy']} loss={values['train_loss']} | "
                        f"val acc={values['val_accuracy']} loss={values['val_loss']} | "
                        f"Dice={values['val_dice']} mIoU={values['val_miou']} "
                        f"BIoU={values['val_biou']}",
                        flush=True,
                    )
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
            log_dir = output_root / "official" / "logs"
            logs = sorted(log_dir.glob(f"{run_name}_*.log"), key=lambda path: path.stat().st_mtime)
            if not logs:
                raise FileNotFoundError(f"Official trainer did not produce a log for {run_name}")
            save_history(logs[-1], output_root / "training_history" / run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-order", help="Comma-separated order containing every task")
    parser.add_argument("--shuffle-tasks", action="store_true")
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    order = ordered_task_names(config, args.task_order, args.shuffle_tasks, args.shuffle_seed)
    run(config, order, args.dry_run, args.start_at)


if __name__ == "__main__":
    main()
