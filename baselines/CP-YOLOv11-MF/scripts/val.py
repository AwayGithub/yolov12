"""
CP-YOLOv11-MF validation — matches ultralytics YOLO val output format.
"""
import argparse, sys, os
from pathlib import Path
import torch
import numpy as np
import cv2

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse model/dataset from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_rgbt3m import CPModel, RGBT3MDataset

from torch.utils.data import DataLoader
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou


class Args:
    box, cls, dfl = 7.5, 0.1, 1.5


def decode_preds(feats, strides, nc=2):
    """Decode raw DFL outputs to [B, total_anchors, 4+nc] (xyxy, scores)."""
    reg_max = 16
    all_preds = []

    for feat, stride in zip(feats, strides):
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        pred_reg = feat_flat[:, :, :reg_max * 4]  # [B, HW, 64]
        pred_cls = feat_flat[:, :, reg_max * 4:]  # [B, HW, nc]

        # DFL decode: softmax then weighted sum
        pred_reg = pred_reg.view(B, H * W, 4, reg_max)
        weights = torch.softmax(pred_reg, dim=-1)
        reg_values = torch.arange(reg_max, dtype=feats[0].dtype, device=feats[0].device)
        xywh = (weights * reg_values).sum(dim=-1)  # [B, HW, 4] (ltrb relative to grid)

        # Grid centers
        yv, xv = torch.meshgrid(torch.arange(H, device=feat.device), torch.arange(W, device=feat.device), indexing='ij')
        grid = torch.stack([xv, yv], dim=-1).float().view(-1, 2)  # [HW, 2]

        # Decode: xywh (ltrb) -> xyxy
        cx = grid[:, 0] + 0.5  # grid center x
        cy = grid[:, 1] + 0.5  # grid center y
        l, t, r, b = xywh[:, :, 0], xywh[:, :, 1], xywh[:, :, 2], xywh[:, :, 3]

        x1 = (cx - l) * stride
        y1 = (cy - t) * stride
        x2 = (cx + r) * stride
        y2 = (cy + b) * stride

        scores = torch.sigmoid(pred_cls)  # [B, HW, nc]
        # Append: [x1, y1, x2, y2, score_0, score_1, ...]
        out = torch.cat([x1.unsqueeze(-1), y1.unsqueeze(-1), x2.unsqueeze(-1), y2.unsqueeze(-1), scores], dim=-1)
        all_preds.append(out)

    return torch.cat(all_preds, dim=1)  # [B, total_anchors, 4+nc]


def non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, nc=2, max_det=300):
    """NMS per image. preds: [B, A, 4+nc]."""
    try:
        from torchvision.ops import nms as tv_nms
    except ImportError:
        tv_nms = None

    results = []
    B = preds.shape[0]

    for b in range(B):
        pred = preds[b]  # [A, 4+nc]
        boxes = pred[:, :4]  # [A, 4] xyxy
        scores = pred[:, 4:]  # [A, nc]

        # max class score per prediction
        max_scores, cls_ids = scores.max(dim=1)  # [A]

        # filter by confidence
        mask = max_scores > conf_thres
        boxes, max_scores, cls_ids = boxes[mask], max_scores[mask], cls_ids[mask]

        if boxes.numel() == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        # NMS per class
        if tv_nms is not None:
            keep = []
            for c in cls_ids.unique():
                c_mask = cls_ids == c
                c_keep = tv_nms(boxes[c_mask], max_scores[c_mask], iou_thres)
                # map back to global indices
                global_idx = c_mask.nonzero(as_tuple=False)[c_keep].squeeze(1)
                keep.append(global_idx)
            if keep:
                keep = torch.cat(keep)
            else:
                keep = torch.arange(len(boxes), device=boxes.device)
        else:
            keep = torch.arange(len(boxes), device=boxes.device)

        boxes, max_scores, cls_ids = boxes[keep], max_scores[keep], cls_ids[keep]

        # Sort by score, take top max_det
        order = max_scores.argsort(descending=True)[:max_det]
        boxes, max_scores, cls_ids = boxes[order], max_scores[order], cls_ids[order]

        det = torch.cat([boxes, max_scores.unsqueeze(1), cls_ids.unsqueeze(1).float()], dim=1)  # [N, 6]
        results.append(det.cpu().numpy())

    return results


def run_val(args):
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    root = Path("/data/xwh/dataset/RGBT-3M/RGBT-3M")
    out_dir = Path(REPO_ROOT) / "runs" / "detect" / "val_cp_yolo11mf"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = {0: "fire", 1: "person"}

    print("Loading val dataset...")
    val_ds = RGBT3MDataset(root, "val")
    val_ld = DataLoader(val_ds, 1, False, collate_fn=RGBT3MDataset.collate_fn, num_workers=0)

    ckpt = torch.load(args.weights, map_location=dev, weights_only=False)
    model = CPModel(nc=args.nc).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.inner.neck.detect.stride = torch.tensor([8.0, 16.0, 32.0], device=dev)
    print(f"Loaded: {args.weights} (epoch {ckpt.get('epoch', '?')})")

    strides = [8, 16, 32]

    # Collect all predictions and ground truth
    all_pred_boxes = []   # list of [N_i, 4] (xyxy, pixel coords)
    all_pred_scores = []  # list of [N_i]
    all_pred_cls = []     # list of [N_i]
    all_gt_boxes = []     # list of [M_i, 4] (xyxy, pixel coords, normalized 0-1 -> pixel)
    all_gt_cls = []       # list of [M_i]
    img_shapes = []       # list of (h, w)

    print("Running inference...")
    with torch.no_grad():
        for idx, (imgs, batch) in enumerate(val_ld):
            imgs = imgs.to(dev)
            feats = model(imgs)

            # feats is a tuple: (inference_tensor, raw_list) or (list_of_per_scale,)
            if isinstance(feats, tuple):
                if isinstance(feats[0], torch.Tensor) and feats[0].dim() == 4:
                    raw_feats = feats[0]  # shouldn't happen in train mode
                else:
                    raw_feats = feats[1] if isinstance(feats[1], list) else list(feats[0])
            else:
                raw_feats = feats if isinstance(feats, list) else list(feats)

            # Ensure raw_feats is a list of tensors
            if not isinstance(raw_feats, list):
                raw_feats = list(raw_feats)

            # Decode
            pred = decode_preds(raw_feats, strides, nc=args.nc)  # [1, A, 4+nc]
            det = non_max_suppression(pred, conf_thres=args.conf, iou_thres=args.iou, nc=args.nc)[0]

            all_pred_boxes.append(det[:, :4])
            all_pred_scores.append(det[:, 4])
            all_pred_cls.append(det[:, 5])

            # Ground truth: batch_idx [N,1], cls [N,1], bboxes [N,4]
            b_idx = batch["batch_idx"][:, 0]  # [N]
            cls_flat = batch["cls"][:, 0]      # [N]
            bboxes_flat = batch["bboxes"]       # [N, 4] cx cy w h

            mask = b_idx == 0  # image 0 in batch
            h, w = 480, 640

            if mask.any():
                gt_cls = cls_flat[mask].numpy()
                gt_cxcywh = bboxes_flat[mask].numpy()

                gt_xyxy = np.zeros((len(gt_cxcywh), 4), dtype=np.float32)
                gt_xyxy[:, 0] = (gt_cxcywh[:, 0] - gt_cxcywh[:, 2] / 2) * w
                gt_xyxy[:, 1] = (gt_cxcywh[:, 1] - gt_cxcywh[:, 3] / 2) * h
                gt_xyxy[:, 2] = (gt_cxcywh[:, 0] + gt_cxcywh[:, 2] / 2) * w
                gt_xyxy[:, 3] = (gt_cxcywh[:, 1] + gt_cxcywh[:, 3] / 2) * h
                all_gt_boxes.append(gt_xyxy)
                all_gt_cls.append(gt_cls)
            else:
                all_gt_boxes.append(np.zeros((0, 4), dtype=np.float32))
                all_gt_cls.append(np.zeros((0,), dtype=np.float32))
            img_shapes.append((h, w))

            # Save visualization for first N
            if idx < 20:
                vis = imgs[0, :3].cpu().numpy().transpose(1, 2, 0)
                vis = (vis * 255).astype(np.uint8).copy()
                vis = cv2.resize(vis, (w, h))

                # Draw GT (green)
                for i in range(len(all_gt_boxes[-1])):
                    g = all_gt_boxes[-1][i]
                    vis = cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 2)

                # Draw predictions (red/blue)
                if len(det) > 0:
                    for d in det:
                        x1, y1, x2, y2, sc, ci = d
                        ci = int(ci)
                        color = (0, 0, 255) if ci == 0 else (255, 100, 0)
                        vis = cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{class_names.get(ci, str(ci))} {sc:.2f}"
                        vis = cv2.putText(vis, label, (int(x1), max(int(y1) - 5, 15)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                cv2.imwrite(str(out_dir / f"val_{idx:04d}.jpg"), vis)

            if idx % 500 == 0:
                print(f"  [{idx}/{len(val_ds)}]")

    # ---- Compute metrics ----
    print("\n=== Metrics ===")

    # For AP calculation, we need arrays of (boxes, scores, classes) for predictions
    # and (boxes, classes) for ground truth
    # Use all images' predictions and GT concatenated

    # Compute AP per class using ultralytics ap_per_class
    iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
    seen = 0
    stats = []  # (correct, conf, pred_cls, target_cls)

    for i in range(len(val_ds)):
        pred_boxes = torch.tensor(all_pred_boxes[i], dtype=torch.float32) if len(all_pred_boxes[i]) > 0 else torch.zeros((0, 4))
        pred_scores = torch.tensor(all_pred_scores[i], dtype=torch.float32) if len(all_pred_scores[i]) > 0 else torch.zeros((0,))
        pred_cls = torch.tensor(all_pred_cls[i], dtype=torch.float32) if len(all_pred_cls[i]) > 0 else torch.zeros((0,))

        gt_boxes = torch.tensor(all_gt_boxes[i], dtype=torch.float32) if len(all_gt_boxes[i]) > 0 else torch.zeros((0, 4))
        gt_cls = torch.tensor(all_gt_cls[i], dtype=torch.float32) if len(all_gt_cls[i]) > 0 else torch.zeros((0,))

        nl = len(gt_cls)
        seen += nl

        if nl > 0 and len(pred_boxes) > 0:
            # IoU matrix [nl, np]
            iou_matrix = box_iou(gt_boxes, pred_boxes)
            correct = torch.zeros(iouv.shape[0], len(pred_boxes), dtype=torch.bool)
            # For each GT, find best matching pred
            for j in range(nl):
                # Find best matching pred for this GT
                best_iou = iou_matrix[j].max()
                best_idx = iou_matrix[j].argmax()
                # Assign correct at each IoU threshold
                if gt_cls[j] == pred_cls[best_idx]:
                    for k in range(iouv.shape[0]):
                        if best_iou >= iouv[k]:
                            correct[k, best_idx] = True
        else:
            correct = torch.zeros(iouv.shape[0], len(pred_boxes), dtype=torch.bool)

        stats.append((correct.cpu(), pred_scores.cpu(), pred_cls.cpu(), gt_cls.cpu()))

    # Concatenate
    if stats:
        correct = torch.cat([s[0] for s in stats], 1)  # [10, total_pred]
        conf = torch.cat([s[1] for s in stats])
        pred_cls = torch.cat([s[2] for s in stats])
        target_cls = torch.cat([s[3] for s in stats])

        tp, fp, p, r, f1, ap, _, _, _, _, _, _ = ap_per_class(correct.T.cpu().numpy(), conf.cpu().numpy(), pred_cls.cpu().numpy(), target_cls.cpu().numpy(), plot=False)
        mp, mr, map50, map = p.mean(), r.mean(), ap[:, 0].mean(), ap.mean()
    else:
        mp = mr = map50 = map = 0.0

    print(f"all   P={mp:.5f}  R={mr:.5f}  mAP50={map50:.5f}  mAP50-95={map:.5f}")

    # Per-class
    print("\nPer-class:")
    for c in range(args.nc):
        cls_mask = target_cls == c
        n_gt = cls_mask.sum().item()
        if n_gt == 0:
            print(f"  {class_names[c]:6s}  (no GT)")
            continue
        # filter: only predictions that are this class
        cls_pred_mask = pred_cls == c
        if cls_pred_mask.sum() == 0:
            print(f"  {class_names[c]:6s}  P=0  R=0  mAP50=0  mAP50-95=0  (GT={n_gt})")
            continue
        c_correct = correct[:, cls_pred_mask]
        c_conf = conf[cls_pred_mask]
        c_pred = pred_cls[cls_pred_mask]
        c_gt = target_cls[cls_mask]
        c_tp, c_fp, c_p, c_r, c_f1, c_ap, _, _, _, _, _, _ = ap_per_class(
            c_correct.T.cpu().numpy(), c_conf.cpu().numpy(), c_pred.cpu().numpy(), c_gt.cpu().numpy(), plot=False)
        c_mp, c_mr, c_map50, c_map = c_p.mean(), c_r.mean(), c_ap[:, 0].mean(), c_ap.mean()
        print(f"  {class_names[c]:6s}  P={c_mp:.5f}  R={c_mr:.5f}  mAP50={c_map50:.5f}  mAP50-95={c_map:.5f}  (GT={n_gt})")

    print(f"\nVisualizations saved to: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="runs/detect/train_cp_yolo11mf/last.pt")
    p.add_argument("--nc", type=int, default=2)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", type=str, default="0")
    run_val(p.parse_args())
