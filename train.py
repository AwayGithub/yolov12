from ultralytics import YOLO
from ultralytics.utils import yaml_load
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_mode",
        type=str,
        default="dual_input",
        choices=["dual_input", "rgb_input", "ir_input"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_cfg = yaml_load("ultralytics/cfg/datasets/RGBT-3M.yaml")
    data_cfg["input_mode"] = args.input_mode

    model = YOLO("yolov12n.yaml")
    results = model.train(
        data=data_cfg,
        epochs=300,
        imgsz=256,  # 输入模型的尺寸，也是验证的尺寸
        batch=64,
        workers=0,
        device=0,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        cos_lr=False,
        val_period=5,
    )
