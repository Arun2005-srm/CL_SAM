from pathlib import Path

import pytest

from cl_sam_replication.config import load_config
from cl_sam_replication.runner import build_train_command, ordered_task_names


CONFIG = Path(__file__).parents[1] / "configs" / "five_tasks_colab.yaml"


def test_declared_task_order_is_stable():
    config = load_config(CONFIG)
    assert ordered_task_names(config) == [
        "kvasir_seg", "busi", "sts_2d", "isic_2018", "ebhi_seg"
    ]


def test_order_override_must_be_a_permutation():
    config = load_config(CONFIG)
    reversed_order = "ebhi_seg,isic_2018,sts_2d,busi,kvasir_seg"
    assert ordered_task_names(config, reversed_order)[0] == "ebhi_seg"
    with pytest.raises(ValueError):
        ordered_task_names(config, "kvasir_seg,busi")


def test_seeded_shuffle_is_repeatable():
    config = load_config(CONFIG)
    assert ordered_task_names(config, shuffle=True, seed=9) == ordered_task_names(
        config, shuffle=True, seed=9
    )


def test_external_command_requests_all_reporting_metrics():
    config = load_config(CONFIG)
    order = ordered_task_names(config)
    command = build_train_command(config, order[0], order, 0)
    assert "cl_sam_replication.instrumented_train" in command
    metrics_at = command.index("--metrics")
    assert command[metrics_at + 1: metrics_at + 5] == ["accuracy", "iou", "dice", "biou"]
    assert command[command.index("--all_datasets") + 1] == ",".join(order)


def test_server_config_has_dgx_paths():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "five_tasks_server.yaml")
    assert config["paths"] == {
        "prepared_data_root": "/raid/workspace/AI4CV_CLSAM/datasets",
        "sam_checkpoint": "/raid/workspace/AI4CV_CLSAM/models/sam_vit_b_01ec64.pth",
        "output_root": "/raid/workspace/AI4CV_CLSAM/outputs/five_task_default",
    }
    assert config["experiment"]["task_order"] == [
        "kvasir_seg", "busi", "sts_2d", "isic_2018", "ebhi_seg",
    ]
