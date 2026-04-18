# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Rank validation images by per-image detection quality (worst → best).

For each image, computes per-image mAP50 (and per-class AP50) by running the
full ultralytics validator, then collecting the per-image tp/conf/cls stats it
already computes.

Usage:
    python tools/rank_val_images.py \
        --ckpt runs/detect/RGBT-3M/dual_MF_ChWP4P5_CMAP5/best.pt \
        --device 0

    # Custom output path
    python tools/rank_val_images.py --ckpt ... --out tools/rank_custom.csv

Outputs:
    CSV with columns: rank, image, n_gt, n_pred, tp, fp, fn, precision, recall, f1, mAP50, ap50_smoke, ap50_fire, ap50_person
    Sorted from worst (lowest mAP50) to best.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DualStreamDetectionModel  # noqa: F401
from ultralytics.utils.metrics import ap_per_class


# ---------------------------------------------------------------------------
# Per-image AP50 computation
# ---------------------------------------------------------------------------

def _per_image_ap50(tp_iou0, conf, pred_cls, target_cls, nc=3):
    """Compute per-image mAP50 and per-class AP50 from single-image stats.

    Args:
        tp_iou0: (N,) bool array — TP flags at IoU=0.5 for each prediction.
        conf:    (N,) float array — confidence scores.
        pred_cls:(N,) int array — predicted class indices.
        target_cls: (M,) int array — ground truth class indices.
        nc: number of classes.

    Returns:
        mAP50 (float), per_class_ap (dict {class_idx: ap50}).
    """
    per_class_ap = {}
    if len(target_cls) == 0:
        # No ground truth — perfect if no predictions, else 0
        mAP50 = 1.0 if len(conf) == 0 else 0.0
        return mAP50, per_class_ap

    if len(conf) == 0:
        # Missed everything
        for c in np.unique(target_cls).astype(int):
            per_class_ap[c] = 0.0
        return 0.0, per_class_ap

    # Sort by confidence (descending)
    order = np.argsort(-conf)
    tp_iou0, conf, pred_cls = tp_iou0[order], conf[order], pred_cls[order]

    aps = []
    for c in range(nc):
        n_gt = (target_cls == c).sum()
        if n_gt == 0:
            continue
        mask = pred_cls == c
        n_pred = mask.sum()
        if n_pred == 0:
            per_class_ap[c] = 0.0
            aps.append(0.0)
            continue

        tp_c = tp_iou0[mask].astype(np.float64)
        tpc = np.cumsum(tp_c)
        fpc = np.cumsum(1 - tp_c)
        recall = tpc / n_gt
        precision = tpc / (tpc + fpc)

        # AP via all-point interpolation (COCO-style)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[1.0], precision, [0.0]])
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

        per_class_ap[c] = float(ap)
        aps.append(ap)

    mAP50 = float(np.mean(aps)) if aps else 0.0
    return mAP50, per_class_ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Rank val images by per-image mAP50")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--data", default="ultralytics/cfg/datasets/RGBT-3M.yaml",
                   help="Dataset config YAML")
    p.add_argument("--imgsz", type=int, nargs="+", default=[480, 640],
                   help="Image size [H, W]")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--device", default="0", help="Device")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: tools/rank_val/<ckpt_dir_name>.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[load] {args.ckpt}")
    model = YOLO(args.ckpt)

    # ── Run validation with per-image tracking ──────────────────────────────
    # We collect per-image stats via a callback on_val_batch_end.
    per_image = []  # list of dicts

    # State shared between callback and main
    _state = {"batch_stats": [], "batch_files": []}

    def _on_batch_end(validator):
        """Called after update_metrics has appended to validator.stats."""
        n_new = len(validator.stats["tp"]) - len(_state["batch_stats"])
        for i in range(n_new):
            idx = len(_state["batch_stats"])
            _state["batch_stats"].append({
                "tp": validator.stats["tp"][idx],
                "conf": validator.stats["conf"][idx],
                "pred_cls": validator.stats["pred_cls"][idx],
                "target_cls": validator.stats["target_cls"][idx],
            })

    # We also need image file paths — hook into update_metrics via a wrapper
    _orig_update_metrics = DetectionValidator.update_metrics

    def _patched_update_metrics(self, preds, batch):
        _orig_update_metrics(self, preds, batch)
        # batch["im_file"] has paths for each image in the batch
        if "im_file" in batch:
            for f in batch["im_file"]:
                if len(_state["batch_files"]) < len(self.stats["tp"]):
                    _state["batch_files"].append(f)

    DetectionValidator.update_metrics = _patched_update_metrics

    try:
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=False,
            save=False,
            workers=0,
            verbose=False,
        )
    finally:
        # Restore original method
        DetectionValidator.update_metrics = _orig_update_metrics

    # ── Compute per-image metrics ───────────────────────────────────────────
    nc = model.model.nc if hasattr(model.model, "nc") else 3
    names = model.model.names if hasattr(model.model, "names") else {i: str(i) for i in range(nc)}
    n_images = len(_state["batch_stats"])
    print(f"\n[rank] Computing per-image mAP50 for {n_images} images ...")

    for i in range(n_images):
        s = _state["batch_stats"][i]
        tp_all = s["tp"].cpu().numpy() if isinstance(s["tp"], torch.Tensor) else np.array(s["tp"])
        conf = s["conf"].cpu().numpy() if isinstance(s["conf"], torch.Tensor) else np.array(s["conf"])
        pred_cls = s["pred_cls"].cpu().numpy() if isinstance(s["pred_cls"], torch.Tensor) else np.array(s["pred_cls"])
        target_cls = s["target_cls"].cpu().numpy() if isinstance(s["target_cls"], torch.Tensor) else np.array(s["target_cls"])

        # tp_all shape: (N_pred, 10) for 10 IoU thresholds; column 0 = IoU=0.5
        tp_iou0 = tp_all[:, 0] if tp_all.ndim == 2 else tp_all

        n_gt = len(target_cls)
        n_pred = len(conf)
        tp_count = int(tp_iou0.sum()) if len(tp_iou0) > 0 else 0
        fp_count = n_pred - tp_count
        fn_count = n_gt - tp_count

        precision = tp_count / max(n_pred, 1)
        recall = tp_count / max(n_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        mAP50, cls_ap = _per_image_ap50(tp_iou0, conf, pred_cls, target_cls, nc=nc)

        im_file = _state["batch_files"][i] if i < len(_state["batch_files"]) else f"image_{i}"

        per_image.append({
            "image": Path(im_file).name,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mAP50": round(mAP50, 4),
            **{f"ap50_{names[c]}": round(cls_ap.get(c, float("nan")), 4) for c in range(nc)},
        })

    # ── Sort worst → best ──────────────────────────────────────────────────
    per_image.sort(key=lambda r: (r["mAP50"], r["f1"]))

    # ── Output ──────────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        ckpt_dir_name = Path(args.ckpt).parent.parent.name  # e.g. "dual_MF_ChWP4P5_CMAP5"
        out_path = Path("tools/rank_val") / f"{ckpt_dir_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["rank", *per_image[0].keys()]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(per_image, 1):
            writer.writerow({"rank": rank, **row})

    print(f"\n[done] Saved ranking to: {out_path.resolve()}")

    # Print top-10 worst
    print(f"\n{'='*90}")
    print(f"  Top-10 worst images (lowest mAP50)")
    print(f"{'='*90}")
    print(f"{'Rank':>5} {'mAP50':>7} {'F1':>6} {'P':>6} {'R':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Image'}")
    print(f"{'-'*90}")
    for rank, row in enumerate(per_image[:10], 1):
        print(f"{rank:>5} {row['mAP50']:>7.4f} {row['f1']:>6.4f} "
              f"{row['precision']:>6.4f} {row['recall']:>6.4f} "
              f"{row['tp']:>4} {row['fp']:>4} {row['fn']:>4} {row['image']}")

    # Print bottom-10 best
    print(f"\n{'='*90}")
    print(f"  Top-10 best images (highest mAP50)")
    print(f"{'='*90}")
    print(f"{'Rank':>5} {'mAP50':>7} {'F1':>6} {'P':>6} {'R':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Image'}")
    print(f"{'-'*90}")
    for rank, row in zip(range(n_images, n_images - 10, -1), reversed(per_image[-10:])):
        print(f"{rank:>5} {row['mAP50']:>7.4f} {row['f1']:>6.4f} "
              f"{row['precision']:>6.4f} {row['recall']:>6.4f} "
              f"{row['tp']:>4} {row['fp']:>4} {row['fn']:>4} {row['image']}")


if __name__ == "__main__":
    main()
