# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run a single validation pass for a checkpoint and print class-wise metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import yaml_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--project", default="runs/detect/val")
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_cfg = yaml_load(args.data)
    model = YOLO(args.weights)
    metrics = model.val(
        data=data_cfg,
        imgsz=[480, 640],
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=True,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print(f"save_dir: {metrics.save_dir}")
    print("overall:")
    rd = metrics.results_dict
    print(f"  precision: {rd['metrics/precision(B)']:.5f}")
    print(f"  recall: {rd['metrics/recall(B)']:.5f}")
    print(f"  mAP50: {rd['metrics/mAP50(B)']:.5f}")
    print(f"  mAP50-95: {rd['metrics/mAP50-95(B)']:.5f}")
    print("classes:")
    for class_id, name in metrics.names.items():
        p, r, ap50, ap = metrics.class_result(class_id)
        print(f"  {name}: P={p:.5f} R={r:.5f} mAP50={ap50:.5f} mAP50-95={ap:.5f}")


if __name__ == "__main__":
    main()
