"""
Training script for CP-YOLOv11-MF.

Usage:
  conda run -n yolov12 python scripts/train.py
  conda run -n yolov12 python scripts/train.py --batch 8 --epochs 200 --device 0
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import CPYOLOv11MF


def train(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model = CPYOLOv11MF(nc=args.nc).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"CP-YOLOv11-MF: {total:,} params ({total/1e6:.1f}M)")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr0, momentum=0.937, weight_decay=5e-4)

    # placeholder: replace with real dataloader
    print("NOTE: Replace this with actual RGBT-3M dataloader before real training.")
    print(f"Config: batch={args.batch}, epochs={args.epochs}, lr0={args.lr0}, device={device}")

    save_dir = Path(REPO_ROOT) / "runs" / "cp_yolo11mf"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        # TODO: replace with real dataloader loop
        # for batch_idx, (rgb, ir, targets) in enumerate(dataloader):
        #     ...
        print(f"Epoch {epoch+1}/{args.epochs}")

        if (epoch + 1) % args.save_period == 0 or epoch + 1 == args.epochs:
            ckpt = save_dir / f"epoch{epoch+1}.pt"
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()}, ckpt)
            print(f"  Saved: {ckpt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train CP-YOLOv11-MF")
    p.add_argument("--nc", type=int, default=2, help="number of classes")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--save_period", type=int, default=50)
    train(p.parse_args())
