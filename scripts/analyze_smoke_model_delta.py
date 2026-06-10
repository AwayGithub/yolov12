# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Analyze per-instance smoke detection differences between two RGBT-3M checkpoints."""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import yaml_load
from ultralytics.utils.metrics import box_iou


BINS = ("<32", "32-64", "64-128", "128-256", ">=256")


def size_bin(side):
    return "<32" if side < 32 else "32-64" if side < 64 else "64-128" if side < 128 else "128-256" if side < 256 else ">=256"


def patch_legacy_parallel_cross(model):
    if not hasattr(model.model, "_parallel_cross_layer_to_stage"):
        model.model._parallel_cross_layer_to_stage = {}
    if not hasattr(model.model, "parallel_cross_a2c2f_stages"):
        model.model.parallel_cross_a2c2f_stages = set()
    for module in model.model.modules():
        if type(module).__name__ == "FreDFTFusion" and not hasattr(module, "checkpoint_ffn"):
            module.checkpoint_ffn = False
        if type(module).__name__ != "DualParallelCrossA2C2f":
            continue
        if not hasattr(module, "cross_scale_rgb"):
            module.register_buffer("cross_scale_rgb", torch.tensor(1.0))
            module.register_buffer("cross_scale_ir", torch.tensor(1.0))
        if not hasattr(module, "gamma_mode"):
            module.gamma_mode = "free"


class SmokeInstanceValidator(DetectionValidator):
    records = []

    def init_metrics(self, model):
        super().init_metrics(model)
        self.instance_records = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            prepared = self._prepare_batch(si, batch)
            gt_boxes = prepared["bbox"]
            gt_classes = prepared["cls"].long()
            smoke_indices = torch.where(gt_classes == 0)[0]
            if not len(smoke_indices):
                continue
            predn = self._prepare_pred(pred, prepared)
            smoke_preds = predn[predn[:, 5] == 0]
            other_preds = predn[predn[:, 5] != 0]
            image_path = str(batch["im_file"][si])

            smoke_gt_boxes = gt_boxes[smoke_indices]
            matched_smoke_gt = set()
            if len(smoke_preds):
                match_ious = box_iou(smoke_gt_boxes, smoke_preds[:, :4])
                for pred_idx in smoke_preds[:, 4].argsort(descending=True).tolist():
                    if float(smoke_preds[pred_idx, 4]) < 0.25:
                        continue
                    available = [i for i in range(len(smoke_indices)) if i not in matched_smoke_gt]
                    if not available:
                        break
                    available_ious = match_ious[available, pred_idx]
                    best_available = int(available_ious.argmax())
                    if float(available_ious[best_available]) >= 0.5:
                        matched_smoke_gt.add(available[best_available])

            for smoke_gt_idx, gt_idx in enumerate(smoke_indices.tolist()):
                gt = gt_boxes[gt_idx : gt_idx + 1]
                w = float(gt[0, 2] - gt[0, 0])
                h = float(gt[0, 3] - gt[0, 1])
                side = (w * h) ** 0.5

                if len(smoke_preds):
                    smoke_ious = box_iou(gt, smoke_preds[:, :4])[0]
                    best_smoke = int(smoke_ious.argmax())
                    best_iou = float(smoke_ious[best_smoke])
                    best_conf = float(smoke_preds[best_smoke, 4])
                    best_box = smoke_preds[best_smoke, :4].tolist()
                else:
                    best_iou = best_conf = 0.0
                    best_box = None

                if len(other_preds):
                    other_ious = box_iou(gt, other_preds[:, :4])[0]
                    best_other = int(other_ious.argmax())
                    other_iou = float(other_ious[best_other])
                    other_conf = float(other_preds[best_other, 4])
                    other_cls = int(other_preds[best_other, 5])
                else:
                    other_iou = other_conf = 0.0
                    other_cls = -1

                detected = smoke_gt_idx in matched_smoke_gt
                if detected:
                    reason = "detected"
                elif best_iou >= 0.5:
                    reason = "low_confidence"
                elif best_iou >= 0.1:
                    reason = "poor_localization"
                elif other_iou >= 0.5 and other_conf >= 0.25:
                    reason = "class_confusion"
                else:
                    reason = "no_response"

                self.instance_records.append({
                    "image": image_path,
                    "gt_index": gt_idx,
                    "gt_box": [round(float(x), 3) for x in gt[0].tolist()],
                    "width": round(w, 3),
                    "height": round(h, 3),
                    "sqrt_area": round(side, 3),
                    "size_bin": size_bin(side),
                    "detected": detected,
                    "reason": reason,
                    "best_smoke_iou": round(best_iou, 5),
                    "best_smoke_conf": round(best_conf, 5),
                    "best_smoke_box": None if best_box is None else [round(float(x), 3) for x in best_box],
                    "best_other_iou": round(other_iou, 5),
                    "best_other_conf": round(other_conf, 5),
                    "best_other_cls": other_cls,
                })

    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics(*args, **kwargs)
        SmokeInstanceValidator.records = self.instance_records


def summarize(records):
    misses = [x for x in records if not x["detected"]]
    by_bin_total = Counter(x["size_bin"] for x in records)
    by_bin_miss = Counter(x["size_bin"] for x in misses)
    return {
        "instances": len(records),
        "detected": len(records) - len(misses),
        "missed": len(misses),
        "recall_at_conf025_iou050": round((len(records) - len(misses)) / max(len(records), 1), 6),
        "miss_reason": dict(Counter(x["reason"] for x in misses)),
        "size_bins": {
            b: {"total": by_bin_total[b], "missed": by_bin_miss[b], "miss_rate": round(by_bin_miss[b] / max(by_bin_total[b], 1), 6)}
            for b in BINS
        },
        "images_with_misses": len({x["image"] for x in misses}),
    }


def run_checkpoint(weights, output, data_root, device, batch):
    data = yaml_load("ultralytics/cfg/datasets/RGBT-3M.yaml")
    data.update({"path": data_root, "val": str(Path(data_root) / "val.txt"), "input_mode": "dual_input"})
    model = YOLO(weights)
    patch_legacy_parallel_cross(model)
    model.val(
        validator=SmokeInstanceValidator,
        data=data,
        imgsz=[480, 640],
        batch=batch,
        device=device,
        workers=0,
        plots=False,
        save=False,
        verbose=False,
        conf=0.001,
        iou=0.7,
    )
    payload = {"weights": weights, "criteria": {"class": "smoke", "conf": 0.25, "iou": 0.5}, "summary": summarize(SmokeInstanceValidator.records), "records": SmokeInstanceValidator.records}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def compare(a_path, b_path, output):
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    key = lambda x: (x["image"], x["gt_index"])
    a_records = {key(x): x for x in a["records"]}
    b_records = {key(x): x for x in b["records"]}
    lost = [{"a": a_records[k], "b": b_records[k]} for k in a_records.keys() & b_records.keys() if a_records[k]["detected"] and not b_records[k]["detected"]]
    gained = [{"a": a_records[k], "b": b_records[k]} for k in a_records.keys() & b_records.keys() if not a_records[k]["detected"] and b_records[k]["detected"]]
    payload = {
        "a": a["weights"], "b": b["weights"], "a_detected_b_missed": lost, "a_missed_b_detected": gained,
        "summary": {
            "a_detected_b_missed": len(lost), "a_missed_b_detected": len(gained),
            "lost_by_size": dict(Counter(x["b"]["size_bin"] for x in lost)),
            "lost_by_reason": dict(Counter(x["b"]["reason"] for x in lost)),
            "lost_images": len({x["b"]["image"] for x in lost}),
        },
    }
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def visualize(report, output_dir, limit):
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    records = payload.get("a_detected_b_missed") or [{"b": x} for x in payload["records"] if not x["detected"]]
    records.sort(key=lambda x: (x["b"]["size_bin"], -x["b"]["best_smoke_iou"], x["b"]["best_smoke_conf"]))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(records[:limit]):
        rec = pair["b"]
        rgb_path = Path(rec["image"])
        ir_path = Path(str(rgb_path).replace("/RGB/", "/IR/"))
        rgb = cv2.imread(str(rgb_path)); ir = cv2.imread(str(ir_path))
        if rgb is None or ir is None:
            continue
        for image, label in ((rgb, "RGB"), (ir, "IR")):
            x1, y1, x2, y2 = map(int, rec["gt_box"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        canvas = np.concatenate((rgb, ir), axis=1)
        text = f'{rec["size_bin"]} {rec["reason"]} iou={rec["best_smoke_iou"]:.2f} conf={rec["best_smoke_conf"]:.2f}'
        cv2.putText(canvas, text, (8, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imwrite(str(out / f'{index:03d}_{rgb_path.stem}.jpg'), canvas)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("weights"); run.add_argument("output"); run.add_argument("--data-root", default="/data/xwh/dataset/RGBT-3M/RGBT-3M"); run.add_argument("--device", default="0"); run.add_argument("--batch", type=int, default=4)
    comp = sub.add_parser("compare"); comp.add_argument("a"); comp.add_argument("b"); comp.add_argument("output")
    vis = sub.add_parser("visualize"); vis.add_argument("report"); vis.add_argument("output_dir"); vis.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    if args.command == "run": run_checkpoint(args.weights, args.output, args.data_root, args.device, args.batch)
    elif args.command == "compare": compare(args.a, args.b, args.output)
    else: visualize(args.report, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
