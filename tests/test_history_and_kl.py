import numpy as np

from cl_sam_replication.history import parse_history
from cl_sam_replication.latent_analysis import analyze, diagonal_gaussian_kl


def test_epoch_log_parser():
    line = (
        "[Epoch 1] lr=0.000100 Train loss=0.4000 train_accuracy=0.9000 "
        "train_iou=0.7000 train_dice=0.8000 train_biou=0.6000 | "
        "Test loss=0.5000 test_accuracy=0.8500 test_iou=0.6500 "
        "test_dice=0.7800 test_biou=0.5500"
    )
    history = parse_history(line)
    assert history[0]["epoch"] == 1
    assert history[0]["val_dice"] == 0.78


def test_diagonal_gaussian_kl_identity_and_direction():
    mean_a, mean_b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    var_a, var_b = np.array([1.0, 1.0]), np.array([2.0, 1.0])
    assert diagonal_gaussian_kl(mean_a, var_a, mean_a, var_a) == 0.0
    assert diagonal_gaussian_kl(mean_a, var_a, mean_b, var_b) != diagonal_gaussian_kl(
        mean_b, var_b, mean_a, var_a
    )


def test_joint_analysis_outputs(tmp_path):
    rng = np.random.default_rng(4)
    tasks = np.repeat(["a", "b"], 12)
    z = np.vstack([rng.normal(0, 0.2, (12, 4)), rng.normal(2, 0.3, (12, 4))])
    source = tmp_path / "embeddings.npz"
    np.savez_compressed(
        source,
        z=z,
        shared_features=np.pad(z, ((0, 0), (0, 2))),
        true_task=tasks,
        predicted_task=tasks,
        is_unknown=np.zeros(len(tasks), dtype=bool),
        internal_kl=np.ones(len(tasks)),
        sample_id=np.asarray([f"sample_{index}" for index in range(len(tasks))]),
    )
    output = tmp_path / "analysis"
    summary = analyze(source, output, seed=3, perplexity=5)
    assert summary["router_accuracy"] == 1.0
    for name in (
        "joint_tsne_z.png", "joint_tsne_shared_features.png",
        "router_confusion_matrix.png", "pairwise_kl_heatmap.png",
        "pairwise_kl_directional.csv", "analysis_summary.json",
    ):
        assert (output / name).is_file()
