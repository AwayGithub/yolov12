# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Inspect YOLO detection loss scale for the fire/person binary setup."""

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import yaml_load
from ultralytics.utils.tal import make_anchors


ROOT = Path("/data/xwh/dataset/RGBT-3M/RGBT-3M")


def make_trainer(data_yaml: str, model_yaml: str, name: str) -> DetectionTrainer:
    """Build a DetectionTrainer without running full training."""
    data = yaml_load(data_yaml)
    data["path"] = str(ROOT)
    data["train"] = str(ROOT / "train.txt")
    data["val"] = str(ROOT / "val.txt")
    trainer = DetectionTrainer(
        overrides={
            "model": model_yaml,
            "data": data,
            "epochs": 1,
            "batch": 16,
            "imgsz": [480, 640],
            "workers": 0,
            "device": "0",
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 5e-4,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.0,
            "cos_lr": False,
            "seed": 0,
            "deterministic": True,
            "amp": True,
            "plots": False,
            "save": False,
            "project": "/tmp/opencode/yolov12_loss_scale",
            "name": name,
        }
    )
    trainer._setup_train(world_size=1)
    trainer.epoch = trainer.start_epoch
    if hasattr(trainer.model, "use_aux_head"):
        trainer.model.use_aux_head = False
    trainer.model.train()
    return trainer


def inspect_batch(trainer: DetectionTrainer, batch: dict) -> dict:
    """Return loss internals for one batch."""
    batch = trainer.preprocess_batch(batch)
    with torch.amp.autocast("cuda", enabled=trainer.amp):
        preds = trainer.model(batch["img"])
        total_loss, loss_items = trainer.model(batch)

    criterion = trainer.model.criterion.main_criterion if hasattr(trainer.model.criterion, "main_criterion") else trainer.model.criterion
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], criterion.no, -1) for xi in feats], 2).split(
        (criterion.reg_max * 4, criterion.nc), 1
    )
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=criterion.device, dtype=dtype) * criterion.stride[0]
    anchor_points, stride_tensor = make_anchors(feats, criterion.stride, 0.5)
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = criterion.preprocess(targets.to(criterion.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    _, _, target_scores, fg_mask, _ = criterion.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )
    target_scores_sum_raw = target_scores.sum()
    cls_bce_sum = criterion.bce(pred_scores, target_scores.to(dtype)).sum()
    nobj = int(batch["cls"].numel())
    n_images_with_obj = int(batch["batch_idx"].unique().numel()) if nobj else 0
    return {
        "nobj": nobj,
        "empty_images": int(batch_size - n_images_with_obj),
        "target_scores_sum_raw": float(target_scores_sum_raw.detach().float()),
        "fg_mask_sum": int(fg_mask.sum().detach()),
        "cls_bce_sum": float(cls_bce_sum.detach().float()),
        "loss_items": [float(x) for x in loss_items.detach().float().cpu()],
        "total_loss": float(total_loss.detach().float().cpu()),
        "pred_scores_min": float(pred_scores.detach().float().min().cpu()),
        "pred_scores_max": float(pred_scores.detach().float().max().cpu()),
        "finite_loss": bool(torch.isfinite(total_loss).detach().cpu()),
    }


def main() -> None:
    """Print loss-scale diagnostics for binary and 3-class F0."""
    configs = [
        (
            "binary_fp",
            "/data/xwh/code/yolov12/ultralytics/cfg/datasets/RGBT-3M-dual-fire-person.yaml",
            "/data/xwh/code/yolov12/ultralytics/cfg/models/v12/yolov12-dual-f0-no-p2-noaux-fire-person.yaml",
        ),
        (
            "three_class",
            "/data/xwh/code/yolov12/ultralytics/cfg/datasets/RGBT-3M.yaml",
            "/data/xwh/code/yolov12/ultralytics/cfg/models/v12/yolov12-dual-f0-no-p2-noaux.yaml",
        ),
    ]
    for name, data_yaml, model_yaml in configs:
        print(f"=== {name} ===")
        trainer = make_trainer(data_yaml, model_yaml, name)
        for i, batch in zip(range(5), trainer.train_loader):
            print(i, inspect_batch(trainer, batch))
        print()


if __name__ == "__main__":
    main()
