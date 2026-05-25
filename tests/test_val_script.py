# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from argparse import Namespace

import torch.nn as nn

import val as val_script


class DummyYOLO:
    """Small stand-in for an Ultralytics YOLO object loaded from a checkpoint."""

    def __init__(self, in_channels=3, train_args=None):
        self.model = nn.Sequential(nn.Conv2d(in_channels, 16, 3))
        self.ckpt = {"train_args": train_args or {}}


def test_auto_validation_config_uses_checkpoint_ir_dataset_and_mode():
    """A 3-channel IR checkpoint should not fall back to dual_input or original labels."""
    args = Namespace(
        weights=r"runs\detect\RGBT-3M\ir_YOLOv12_P2_P3aux_fire_person_e100\weights\epoch56.pt",
        data=None,
        input_mode="auto",
    )
    yolo = DummyYOLO(train_args={"data": "ultralytics/cfg/datasets/RGBT-3M-ir-fire-person.yaml"})

    data_path = val_script.resolve_data_path(args, yolo)
    data_cfg = val_script.load_validation_data_cfg(data_path)
    input_mode = val_script.resolve_input_mode(args, yolo, data_cfg)

    assert data_path.name == "RGBT-3M-ir-fire-person.yaml"
    assert data_cfg["label_dir"] == "labels_fire_person"
    assert input_mode == "ir_input"
