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
    parser.add_argument(
        "--fusion_stage",
        type=str,
        default="middle",
        choices=["early", "middle"],
        help="early: 6ch 直接拼接输入单分支; middle: 双分支中期融合",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="从指定 checkpoint 继续训练，例如 runs/detect/train/weights/last.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_cfg = yaml_load("ultralytics/cfg/datasets/RGBT-3M.yaml")
    data_cfg["input_mode"] = args.input_mode

    if args.resume:
        model = YOLO(args.resume)
    elif args.fusion_stage == "middle" and args.input_mode == "dual_input":
        model = YOLO("yolov12-dual.yaml")  # 双分支中期融合，n scale
    else:
        model = YOLO("yolov12.yaml")       # 单分支（early fusion 或单模态）
    results = model.train(
        data=data_cfg,
        resume=bool(args.resume),
        epochs=300,
        imgsz=[480, 640],  # 输入模型的尺寸，也是验证的尺寸
        batch=16,
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
        val_period=2,
    )
