"""Extract shared router features, TaskVAE latents, ELBOs, and routes."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .config import load_config
from .runner import ROOT, ordered_task_names


def _evaluation_view(config: dict[str, Any], order: list[str]) -> Path:
    prepared = Path(config["paths"]["prepared_data_root"]).resolve()
    view = Path(config["paths"]["output_root"]) / "evaluation_data"
    view.mkdir(parents=True, exist_ok=True)
    for task in order:
        source = prepared / task
        destination = view / task
        destination.mkdir(parents=True, exist_ok=True)
        for folder in ("image", "label", "imask"):
            link = destination / folder
            target = source / folder
            if not link.exists():
                try:
                    link.symlink_to(target, target_is_directory=True)
                except OSError:
                    shutil.copytree(target, link, dirs_exist_ok=True)
        shutil.copy2(source / "dataset.final_test.json", destination / "dataset.json")
    return view


def extract(config: dict[str, Any], order: list[str]) -> Path:
    seed = int(config["experiment"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    upstream = ROOT / "upstream" / "ca_sam_official"
    sys.path.insert(0, str(upstream))
    from CL.feature_pool import extract_feature_for_vae  # type: ignore
    from CL.vae_router import TaskVAE, score_elbo_all_tasks  # type: ignore
    from data_loader import get_loader  # type: ignore
    from segment_anything import sam_model_registry  # type: ignore

    training, router, paths = config["training"], config["router"], config["paths"]
    device = str(training.get("device", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    evaluation_root = _evaluation_view(config, order)
    args = SimpleNamespace(
        data_dir=str(evaluation_root),
        sam_checkpoint=str(paths["sam_checkpoint"]),
        image_size=int(training.get("image_size", 1024)),
        model_type=str(training.get("model_type", "vit_b")),
        device=device,
        test_mode=True,
        batch_size=int(training.get("eval_batch_size", 1)),
        eval_batch_size=int(training.get("eval_batch_size", 1)),
        train_batch_size=int(training.get("train_batch_size", 1)),
        num_workers=int(training.get("num_workers", 2)),
        mask_num=int(training.get("mask_num", 5)),
        dataset_scale=1.0,
        multimask=False,
        vae_feat=str(router.get("feature", "attn_pool")),
    )
    sam = sam_model_registry[args.model_type](args).to(device).eval()
    for parameter in sam.parameters():
        parameter.requires_grad = False

    checkpoint_root = Path(paths["output_root"]) / "official" / "vaes_ckpt"
    vaes: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for task in order:
        vae = TaskVAE(
            in_dim=int(router.get("input_dimension", 256)),
            latent_dim=int(router.get("latent_dimension", 64)),
        ).to(device)
        vae.load_state_dict(torch.load(checkpoint_root / task / "vae.pth", map_location=device))
        vaes[task] = vae.eval()
        tau_payload = json.loads((checkpoint_root / task / "tau.json").read_text(encoding="utf-8"))
        thresholds[task] = float(tau_payload.get("tau_p97", tau_payload.get("tau_suggested", 1.0)))

    feature_rows, z_rows, mu_rows, logvar_rows, score_rows = [], [], [], [], []
    true_tasks, predicted_tasks, sample_ids, unknown_rows, kl_rows = [], [], [], [], []
    maximum = config["analysis"].get("max_samples_per_task")
    beta = float(router.get("beta", 16.5))

    with torch.no_grad():
        for true_task in order:
            loader = get_loader(args, all_training_sets=[true_task])
            observed = 0
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                feature_map = sam.image_encoder(images)
                features = extract_feature_for_vae(feature_map, mode=args.vae_feat, attn_temp=1.0)
                scores, names = score_elbo_all_tasks(features, vaes, beta=beta, device=device)
                minimum, selected = scores.min(dim=1)
                true_vae = vaes[true_task]
                mu, logvar = true_vae.encode(features)
                # Posterior mean is deterministic and is used as the Z coordinate.
                z = mu
                internal_kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
                for index in range(features.shape[0]):
                    if maximum is not None and observed >= int(maximum):
                        break
                    predicted = names[int(selected[index].item())]
                    feature_rows.append(features[index].cpu().numpy())
                    z_rows.append(z[index].cpu().numpy())
                    mu_rows.append(mu[index].cpu().numpy())
                    logvar_rows.append(logvar[index].cpu().numpy())
                    score_rows.append(scores[index].cpu().numpy())
                    true_tasks.append(true_task)
                    predicted_tasks.append(predicted)
                    unknown_rows.append(bool(minimum[index].item() > thresholds[predicted]))
                    kl_rows.append(float(internal_kl[index].item()))
                    roots = batch.get("image_root", [])
                    sample_ids.append(Path(roots[index] if index < len(roots) else f"{true_task}_{observed}").name)
                    observed += 1
                if maximum is not None and observed >= int(maximum):
                    break

    destination = Path(paths["output_root"]) / "latent_analysis" / "embeddings.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        shared_features=np.asarray(feature_rows),
        z=np.asarray(z_rows),
        mu=np.asarray(mu_rows),
        logvar=np.asarray(logvar_rows),
        elbo_scores=np.asarray(score_rows),
        elbo_task_names=np.asarray(names),
        true_task=np.asarray(true_tasks),
        predicted_task=np.asarray(predicted_tasks),
        is_unknown=np.asarray(unknown_rows),
        internal_kl=np.asarray(kl_rows),
        sample_id=np.asarray(sample_ids),
    )
    print(f"Saved latent data: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-order")
    args = parser.parse_args()
    config = load_config(args.config)
    extract(config, ordered_task_names(config, args.task_order))


if __name__ == "__main__":
    main()

