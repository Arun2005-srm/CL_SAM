"""Joint t-SNE, router diagnostics, and pairwise Gaussian KL analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def diagonal_gaussian_kl(
    mean_p: np.ndarray, variance_p: np.ndarray, mean_q: np.ndarray, variance_q: np.ndarray
) -> float:
    variance_p = np.maximum(variance_p, 1e-8)
    variance_q = np.maximum(variance_q, 1e-8)
    terms = np.log(variance_q / variance_p) + (variance_p + (mean_p - mean_q) ** 2) / variance_q - 1
    return float(0.5 * np.sum(terms))


def _write_matrix(path: Path, names: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *[float(value) for value in row]])


def _plot_tsne(features: np.ndarray, tasks: np.ndarray, output: Path, title: str, seed: int, perplexity: float) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    if len(features) < 3:
        raise ValueError("At least three samples are required for t-SNE")
    effective_perplexity = min(float(perplexity), max(1.0, (len(features) - 1) / 3))
    standardized = StandardScaler().fit_transform(features)
    coordinates = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(standardized)
    figure, axis = plt.subplots(figsize=(10, 8))
    for task in dict.fromkeys(tasks.tolist()):
        selected = tasks == task
        axis.scatter(coordinates[selected, 0], coordinates[selected, 1], s=18, alpha=0.7, label=task)
    axis.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.legend(markerscale=1.5, frameon=True)
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return coordinates


def analyze(input_path: Path, output_dir: Path, seed: int = 42, perplexity: float = 30) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score

    payload = np.load(input_path, allow_pickle=False)
    true = payload["true_task"].astype(str)
    predicted = payload["predicted_task"].astype(str)
    unknown = payload["is_unknown"].astype(bool)
    names = list(dict.fromkeys(true.tolist()))
    output_dir.mkdir(parents=True, exist_ok=True)

    z_coordinates = _plot_tsne(payload["z"], true, output_dir / "joint_tsne_z.png", "Combined TaskVAE Z space", seed, perplexity)
    feature_coordinates = _plot_tsne(
        payload["shared_features"], true, output_dir / "joint_tsne_shared_features.png",
        "Combined shared SAM router-feature space", seed, perplexity,
    )
    with (output_dir / "tsne_coordinates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "true_task", "predicted_task", "is_unknown", "z_x", "z_y", "feature_x", "feature_y"])
        for row in zip(payload["sample_id"], true, predicted, unknown, z_coordinates[:, 0], z_coordinates[:, 1], feature_coordinates[:, 0], feature_coordinates[:, 1]):
            writer.writerow(row)

    confusion = confusion_matrix(true, predicted, labels=names)
    _write_matrix(output_dir / "router_confusion_matrix.csv", names, confusion)
    figure, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(confusion, annot=True, fmt="d", xticklabels=names, yticklabels=names, cmap="Blues", ax=axis)
    axis.set(xlabel="Predicted task", ylabel="True task", title="VAE router confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "router_confusion_matrix.png", dpi=200)
    plt.close(figure)

    with (output_dir / "router_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "true_task", "predicted_task", "is_unknown", "correct", "internal_kl"])
        for row in zip(payload["sample_id"], true, predicted, unknown, true == predicted, payload["internal_kl"]):
            writer.writerow(row)

    means, variances = {}, {}
    for name in names:
        task_z = payload["z"][true == name]
        means[name] = task_z.mean(axis=0)
        variances[name] = task_z.var(axis=0) + 1e-6
    directional = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            directional[i, j] = diagonal_gaussian_kl(means[left], variances[left], means[right], variances[right])
    symmetric = 0.5 * (directional + directional.T)
    _write_matrix(output_dir / "pairwise_kl_directional.csv", names, directional)
    _write_matrix(output_dir / "pairwise_kl_symmetric.csv", names, symmetric)
    figure, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(symmetric, annot=True, fmt=".2f", xticklabels=names, yticklabels=names, cmap="magma", ax=axis)
    axis.set(title="Symmetric pairwise KL divergence in TaskVAE Z space")
    figure.tight_layout()
    figure.savefig(output_dir / "pairwise_kl_heatmap.png", dpi=200)
    plt.close(figure)

    summary = {
        "samples": int(len(true)),
        "router_accuracy": float(accuracy_score(true, predicted)),
        "unknown_rate": float(unknown.mean()),
        "per_task_accuracy": {
            name: float((predicted[true == name] == name).mean()) for name in names
        },
        "mean_internal_kl": {
            name: float(payload["internal_kl"][true == name].mean()) for name in names
        },
        "kl_method": "diagonal Gaussian fitted to deterministic true-task VAE posterior means",
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(analyze(Path(args.input), Path(args.output_dir), args.seed, args.perplexity), indent=2))


if __name__ == "__main__":
    main()

