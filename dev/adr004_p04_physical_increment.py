# Ultralytics YOLO 🚀, AGPL-3.0 license
"""ADR-004 P04: test whether physical features add information beyond frozen B2 ROI features."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import rankdata

from ultralytics import YOLO
from ultralytics.nn.modules import DMGFusion, DMGFusionInit8d, DMGFusionPosAlpha, FreDFTFusion, M2DLocalIlluminationFusion


CLASS_NAMES = {0: "smoke", 1: "fire", 2: "person"}
IMAGE_SIZE = (480, 640)  # h, w; matches ADR-003 validation convention.
RGB_METRICS = (
    "rgb_gradient_mean",
    "rgb_laplacian_energy",
    "rgb_edge_density",
    "rgb_std",
    "rgb_contrast_abs",
    "rgb_std_ratio",
)
IR_METRICS = (
    "ir_gradient_mean",
    "ir_laplacian_energy",
    "ir_edge_density",
    "ir_contrast_abs",
    "ir_std_ratio",
    "ir_mean",
)
CROSS_METRICS = (
    "cross_edge_iou",
    "cross_gradient_corr",
    "cross_gray_corr",
    "cross_ssim",
    "cross_zmad",
    "cross_contrast_gap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/xwh/dataset/RGBT-3M/RGBT-3M"))
    parser.add_argument(
        "--physical",
        type=Path,
        default=Path("runs/detect/adr004/pilot/P01_physical_profile/instance_physical_metrics.csv"),
    )
    parser.add_argument(
        "--smoke-outcomes",
        type=Path,
        default=Path("runs/detect/adr004/pilot/P02_b2_smoke_failure_physics/smoke_outcome_physics.csv"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4/weights/last.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/detect/adr004/pilot/P04_physical_increment"))
    parser.add_argument("--max-per-class-video", type=int, default=180)
    parser.add_argument("--background-per-image", type=int, default=2)
    parser.add_argument("--max-background-per-video", type=int, default=240)
    parser.add_argument("--batch-images", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def yolo_box_to_xyxy(values: Iterable[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x_center, y_center, width, height = values
    x1 = max(0, min(image_width - 1, math.floor((x_center - width / 2) * image_width)))
    y1 = max(0, min(image_height - 1, math.floor((y_center - height / 2) * image_height)))
    x2 = max(x1 + 1, min(image_width, math.ceil((x_center + width / 2) * image_width)))
    y2 = max(y1 + 1, min(image_height, math.ceil((y_center + height / 2) * image_height)))
    return x1, y1, x2, y2


def read_labels(path: Path, image_width: int, image_height: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 5:
            class_id = int(float(fields[0]))
            labels.append((class_id, yolo_box_to_xyxy(map(float, fields[1:5]), image_width, image_height)))
    return labels


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - inter
    return inter / union if union else 0.0


def safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64, copy=False).ravel()
    second = second.astype(np.float64, copy=False).ravel()
    if first.size < 2 or second.size != first.size or first.std() < 1e-8 or second.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64, copy=False).ravel()
    second = second.astype(np.float64, copy=False).ravel()
    if first.size < 2 or second.size != first.size:
        return float("nan")
    c1, c2 = 0.01**2, 0.03**2
    mean_first, mean_second = first.mean(), second.mean()
    var_first, var_second = first.var(), second.var()
    covariance = np.mean((first - mean_first) * (second - mean_second))
    denominator = (mean_first**2 + mean_second**2 + c1) * (var_first + var_second + c2)
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    return float(numerator / denominator) if denominator > 0 else float("nan")


def load_dense_features(rgb_path: Path, ir_path: Path) -> dict[str, np.ndarray]:
    rgb_bgr = cv2.imread(str(rgb_path))
    ir_bgr = cv2.imread(str(ir_path))
    if rgb_bgr is None or ir_bgr is None:
        raise FileNotFoundError(f"Unable to load image pair: {rgb_path}, {ir_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    rgb_grad_x = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=3)
    rgb_grad_y = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=3)
    ir_grad_x = cv2.Sobel(ir_gray, cv2.CV_32F, 1, 0, ksize=3)
    ir_grad_y = cv2.Sobel(ir_gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_gradient = cv2.magnitude(rgb_grad_x, rgb_grad_y)
    ir_gradient = cv2.magnitude(ir_grad_x, ir_grad_y)
    rgb_laplacian = cv2.Laplacian(rgb_gray, cv2.CV_32F)
    ir_laplacian = cv2.Laplacian(ir_gray, cv2.CV_32F)
    rgb_edges = cv2.Canny((rgb_gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    ir_edges = cv2.Canny((ir_gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    rgb_z = (rgb_gray - rgb_gray.mean()) / (rgb_gray.std() + 1e-8)
    ir_z = (ir_gray - ir_gray.mean()) / (ir_gray.std() + 1e-8)
    return {
        "rgb": rgb,
        "rgb_gray": rgb_gray,
        "rgb_gradient": rgb_gradient,
        "rgb_laplacian": rgb_laplacian,
        "rgb_edges": rgb_edges,
        "rgb_z": rgb_z,
        "ir_gray": ir_gray,
        "ir_gradient": ir_gradient,
        "ir_laplacian": ir_laplacian,
        "ir_edges": ir_edges,
        "ir_z": ir_z,
    }


def make_ring_mask(
    box: tuple[int, int, int, int],
    all_boxes: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
    scale: float = 1.75,
) -> np.ndarray:
    height, width = image_shape
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ex1 = max(0, int(math.floor(cx - (x2 - x1) * scale / 2)))
    ey1 = max(0, int(math.floor(cy - (y2 - y1) * scale / 2)))
    ex2 = min(width, int(math.ceil(cx + (x2 - x1) * scale / 2)))
    ey2 = min(height, int(math.ceil(cy + (y2 - y1) * scale / 2)))
    mask = np.zeros((height, width), dtype=bool)
    mask[ey1:ey2, ex1:ex2] = True
    for bx1, by1, bx2, by2 in all_boxes:
        mask[by1:by2, bx1:bx2] = False
    return mask


def region_metrics(prefix: str, gray: np.ndarray, gradient: np.ndarray, laplacian: np.ndarray, edges: np.ndarray,
                   box: tuple[int, int, int, int], ring_mask: np.ndarray) -> dict[str, float]:
    x1, y1, x2, y2 = box
    object_gray = gray[y1:y2, x1:x2]
    object_gradient = gradient[y1:y2, x1:x2]
    object_laplacian = laplacian[y1:y2, x1:x2]
    object_edges = edges[y1:y2, x1:x2]
    ring_gray = gray[ring_mask]
    ring_gradient = gradient[ring_mask]
    ring_laplacian = laplacian[ring_mask]
    ring_edges = edges[ring_mask]
    values = {
        f"{prefix}_mean": float(object_gray.mean()),
        f"{prefix}_std": float(object_gray.std()),
        f"{prefix}_gradient_mean": float(object_gradient.mean()),
        f"{prefix}_laplacian_energy": float(np.mean(object_laplacian**2)),
        f"{prefix}_edge_density": float(object_edges.mean()),
        f"{prefix}_ring_mean": float(ring_gray.mean()) if ring_gray.size else float("nan"),
        f"{prefix}_ring_std": float(ring_gray.std()) if ring_gray.size else float("nan"),
        f"{prefix}_ring_gradient_mean": float(ring_gradient.mean()) if ring_gradient.size else float("nan"),
        f"{prefix}_ring_laplacian_energy": float(np.mean(ring_laplacian**2)) if ring_laplacian.size else float("nan"),
        f"{prefix}_ring_edge_density": float(ring_edges.mean()) if ring_edges.size else float("nan"),
    }
    values[f"{prefix}_contrast_signed"] = values[f"{prefix}_mean"] - values[f"{prefix}_ring_mean"]
    values[f"{prefix}_contrast_abs"] = abs(values[f"{prefix}_contrast_signed"])
    values[f"{prefix}_std_ratio"] = values[f"{prefix}_std"] / (values[f"{prefix}_ring_std"] + 1e-8)
    return values


def physical_metrics_for_box(root: Path, split: str, image_id: str, box: tuple[int, int, int, int],
                             gt_boxes: list[tuple[int, int, int, int]]) -> dict[str, float]:
    features = load_dense_features(root / "RGB" / split / f"{image_id}.jpg", root / "IR" / split / f"{image_id}.jpg")
    ring_mask = make_ring_mask(box, gt_boxes, features["rgb_gray"].shape)
    x1, y1, x2, y2 = box
    rgb_patch = features["rgb_gray"][y1:y2, x1:x2]
    ir_patch = features["ir_gray"][y1:y2, x1:x2]
    rgb_gradient_patch = features["rgb_gradient"][y1:y2, x1:x2]
    ir_gradient_patch = features["ir_gradient"][y1:y2, x1:x2]
    rgb_edges_patch = features["rgb_edges"][y1:y2, x1:x2] > 0
    ir_edges_patch = features["ir_edges"][y1:y2, x1:x2] > 0
    union_edges = np.logical_or(rgb_edges_patch, ir_edges_patch).sum()
    metrics = {}
    metrics.update(region_metrics("rgb", features["rgb_gray"], features["rgb_gradient"], features["rgb_laplacian"],
                                  features["rgb_edges"], box, ring_mask))
    metrics.update(region_metrics("ir", features["ir_gray"], features["ir_gradient"], features["ir_laplacian"],
                                  features["ir_edges"], box, ring_mask))
    metrics.update({
        "cross_gray_corr": safe_corr(rgb_patch, ir_patch),
        "cross_gradient_corr": safe_corr(rgb_gradient_patch, ir_gradient_patch),
        "cross_ssim": global_ssim(rgb_patch, ir_patch),
        "cross_zmad": float(np.mean(np.abs(features["rgb_z"][y1:y2, x1:x2] - features["ir_z"][y1:y2, x1:x2]))),
        "cross_edge_iou": float(np.logical_and(rgb_edges_patch, ir_edges_patch).sum() / union_edges)
        if union_edges else float("nan"),
        "cross_contrast_gap": abs(float(metrics["rgb_contrast_abs"]) - float(metrics["ir_contrast_abs"])),
    })
    return metrics


def sample_instances(frame: pd.DataFrame, max_per_class_video: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples = []
    for _, group in frame.groupby(["split", "video", "class_name"], sort=False):
        if len(group) > max_per_class_video:
            group = group.iloc[rng.choice(len(group), max_per_class_video, replace=False)]
        samples.append(group)
    return pd.concat(samples, ignore_index=True)


def random_background_rows(args: argparse.Namespace, selected: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    smoke_boxes = selected[selected["class_name"] == "smoke"][["box_width", "box_height"]].to_numpy(dtype=int)
    rows = []
    image_groups = selected.groupby(["split", "image_id", "video"], sort=False)
    for (split, image_id, video), _ in image_groups:
        rgb_path = args.root / "RGB" / split / f"{image_id}.jpg"
        image = cv2.imread(str(rgb_path))
        if image is None or len(smoke_boxes) == 0:
            continue
        height, width = image.shape[:2]
        labels = read_labels(args.root / "labels" / split / f"{image_id}.txt", width, height)
        gt_boxes = [box for _, box in labels]
        made = 0
        for _ in range(60):
            bw, bh = smoke_boxes[rng.integers(0, len(smoke_boxes))]
            bw, bh = int(np.clip(bw, 12, width - 1)), int(np.clip(bh, 12, height - 1))
            x1 = int(rng.integers(0, max(1, width - bw)))
            y1 = int(rng.integers(0, max(1, height - bh)))
            box = (x1, y1, x1 + bw, y1 + bh)
            if gt_boxes and max(box_iou(box, gt) for gt in gt_boxes) > 0.02:
                continue
            metrics = physical_metrics_for_box(args.root, split, image_id, box, gt_boxes)
            rows.append({
                "split": split,
                "image_id": image_id,
                "video": video,
                "object_index": -1 - made,
                "class_id": -1,
                "class_name": "hard_background_candidate",
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "box_width": bw,
                "box_height": bh,
                "box_area": bw * bh,
                **metrics,
            })
            made += 1
            if made >= args.background_per_image:
                break
    if not rows:
        return pd.DataFrame()
    bg = pd.DataFrame(rows)
    metric_cols = [c for c in RGB_METRICS + IR_METRICS + CROSS_METRICS if c in selected.columns and c in bg.columns]
    smoke = selected[selected["class_name"] == "smoke"]
    center = smoke[metric_cols].median(numeric_only=True)
    scale = selected[metric_cols].std(numeric_only=True).replace(0, 1).fillna(1)
    values = bg[metric_cols].replace([np.inf, -np.inf], np.nan).fillna(center)
    bg["smoke_physical_distance"] = (((values - center) / scale)**2).sum(axis=1)
    kept = []
    for _, group in bg.groupby("video", sort=False):
        kept.append(group.nsmallest(args.max_background_per_video, "smoke_physical_distance"))
    bg = pd.concat(kept, ignore_index=True)
    bg["class_name"] = "hard_background"
    return bg.drop(columns=["smoke_physical_distance"])


def build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    physical = pd.read_csv(args.physical)
    physical = physical[physical["split"].isin(["train", "val"]) & physical["class_name"].isin(CLASS_NAMES.values())]
    selected = sample_instances(physical, args.max_per_class_video, args.seed)
    background = random_background_rows(args, selected, args.seed + 1)
    manifest = pd.concat([selected, background], ignore_index=True) if len(background) else selected.copy()
    manifest = manifest.reset_index(drop=True)
    manifest.insert(0, "sample_id", np.arange(len(manifest), dtype=int))
    return manifest


def load_pair_tensor(root: Path, split: str, image_id: str, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    rgb = cv2.imread(str(root / "RGB" / split / f"{image_id}.jpg"))
    ir = cv2.imread(str(root / "IR" / split / f"{image_id}.jpg"))
    if rgb is None or ir is None:
        raise FileNotFoundError(f"Unable to load image pair for {split}/{image_id}")
    height, width = rgb.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0])), cv2.COLOR_BGR2RGB)
    ir = cv2.cvtColor(cv2.resize(ir, (IMAGE_SIZE[1], IMAGE_SIZE[0])), cv2.COLOR_BGR2RGB)
    array = np.concatenate([ir, rgb], axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).to(device), (height, width)


def fused_features(model: torch.nn.Module, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    x_ir = batch[:, :3]
    x_rgb = batch[:, 3:]
    feats_rgb, feats_ir = model._forward_both_backbones(x_rgb, x_ir)
    lif = model.lif_gate(x_rgb) if model.lif_gate is not None else None
    fused = {}
    for stage_name in model.FUSION_LAYER_INDICES:
        r, i = feats_rgb[stage_name], feats_ir[stage_name]
        fc = model.fusion_convs[stage_name]
        if isinstance(fc, M2DLocalIlluminationFusion):
            fused[stage_name] = fc(r, i, lif)
        elif isinstance(fc, (DMGFusion, DMGFusionPosAlpha, DMGFusionInit8d, FreDFTFusion)):
            fused[stage_name] = fc(r, i)
        else:
            fused[stage_name] = fc(torch.cat([r, i], dim=1))
    return fused


def roi_vector(feature: torch.Tensor, box: tuple[float, float, float, float], image_shape: tuple[int, int]) -> np.ndarray:
    _, channels, fmap_h, fmap_w = feature.shape
    image_h, image_w = image_shape
    x1, y1, x2, y2 = box
    fx1 = int(np.floor(x1 / image_w * fmap_w))
    fy1 = int(np.floor(y1 / image_h * fmap_h))
    fx2 = int(np.ceil(x2 / image_w * fmap_w))
    fy2 = int(np.ceil(y2 / image_h * fmap_h))
    fx1, fy1 = max(0, min(fmap_w - 1, fx1)), max(0, min(fmap_h - 1, fy1))
    fx2, fy2 = max(fx1 + 1, min(fmap_w, fx2)), max(fy1 + 1, min(fmap_h, fy2))
    patch = feature[0, :, fy1:fy2, fx1:fx2]
    pooled = torch.cat([F.adaptive_avg_pool2d(patch.unsqueeze(0), 1).flatten(),
                        F.adaptive_max_pool2d(patch.unsqueeze(0), 1).flatten()])
    return pooled.detach().float().cpu().numpy().astype(np.float32)


def extract_roi_features(args: argparse.Namespace, manifest: pd.DataFrame) -> np.ndarray:
    cache = args.output / "b2_roi_features.npy"
    if cache.exists() and not args.force_features:
        return np.load(cache)
    device = select_device(args.device)
    model = YOLO(str(args.weights)).model.to(device).eval()
    ensure_pcross_compat(model, device)
    rows = []
    with torch.no_grad():
        grouped = manifest.groupby(["split", "image_id"], sort=False)
        total = len(grouped)
        for index, ((split, image_id), group) in enumerate(grouped, 1):
            tensor, image_shape = load_pair_tensor(args.root, split, image_id, device)
            fused = fused_features(model, tensor)
            for _, row in group.iterrows():
                box = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
                rows.append(np.concatenate([roi_vector(fused[s], box, image_shape) for s in ("p2", "p3", "p4", "p5")]))
            if index % 50 == 0 or index == total:
                print(f"Extracted B2 ROI features for {index}/{total} images", flush=True)
    features = np.stack(rows).astype(np.float32)
    np.save(cache, features)
    return features


def ensure_pcross_compat(model: torch.nn.Module, device: torch.device) -> None:
    """Fill defaults for old B2 checkpoints trained before cross-scale fields were added."""
    for module in model.modules():
        if type(module).__name__ != "DualParallelCrossA2C2f":
            continue
        if not hasattr(module, "gamma_mode"):
            module.gamma_mode = "free"
        if not hasattr(module, "gamma_max"):
            module.gamma_max = 0.35
        if not hasattr(module, "cross_scale_rgb"):
            module.register_buffer("cross_scale_rgb", torch.tensor(1.0, device=device))
        if not hasattr(module, "cross_scale_ir"):
            module.register_buffer("cross_scale_ir", torch.tensor(1.0, device=device))
        if not hasattr(module, "cross_drop_path"):
            module.cross_drop_path = 0.0


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, _ = torch.cuda.mem_get_info(index)
        free.append((free_bytes, index))
    return torch.device(f"cuda:{max(free)[1]}")


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    valid = np.isfinite(scores)
    labels, scores = labels[valid].astype(int), scores[valid].astype(float)
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    valid = np.isfinite(scores)
    labels, scores = labels[valid].astype(int), scores[valid].astype(float)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels == 1].sum() / positives)


def fit_ridge_logistic(features: np.ndarray, labels: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = parameters[0] + features @ parameters[1:]
        probs = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        loss = -np.mean(labels * np.log(probs + 1e-12) + (1 - labels) * np.log(1 - probs + 1e-12))
        loss += 0.5 * penalty * np.sum(parameters[1:] ** 2)
        residual = probs - labels
        gradient = np.r_[residual.mean(), features.T @ residual / len(labels) + penalty * parameters[1:]]
        return float(loss), gradient

    result = minimize(objective, np.zeros(features.shape[1] + 1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    med = np.nanmedian(train, axis=0)
    train = np.where(np.isfinite(train), train, med)
    test = np.where(np.isfinite(test), test, med)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    labels = labels[order].astype(int)
    scores = scores[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    fn = tp[-1] - tp
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    return float(scores[int(np.nanargmax(f1))])


def evaluate_fold(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float]:
    train_x, test_x = standardize(train_x, test_x)
    weights = fit_ridge_logistic(train_x, train_y)
    train_scores = 1 / (1 + np.exp(-np.clip(weights[0] + train_x @ weights[1:], -40, 40)))
    test_scores = 1 / (1 + np.exp(-np.clip(weights[0] + test_x @ weights[1:], -40, 40)))
    threshold = best_f1_threshold(train_y, train_scores)
    pred = test_scores >= threshold
    tp = int(((pred == 1) & (test_y == 1)).sum())
    fp = int(((pred == 1) & (test_y == 0)).sum())
    fn = int(((pred == 0) & (test_y == 1)).sum())
    return {
        "auc": roc_auc(test_y, test_scores),
        "pr_auc": average_precision(test_y, test_scores),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "threshold": threshold,
        "n_test": int(len(test_y)),
        "n_pos_test": int(test_y.sum()),
        "n_neg_test": int((test_y == 0).sum()),
    }


def build_design_matrices(manifest: pd.DataFrame, roi_features: np.ndarray) -> dict[str, np.ndarray]:
    physical = manifest[list(RGB_METRICS + IR_METRICS + CROSS_METRICS)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    groups = {
        "base": roi_features,
        "base_rgb_structure": np.concatenate([roi_features, manifest[list(RGB_METRICS)].to_numpy(float)], axis=1),
        "base_ir_structure": np.concatenate([roi_features, manifest[list(IR_METRICS)].to_numpy(float)], axis=1),
        "base_cross_edge": np.concatenate([roi_features, manifest[list(CROSS_METRICS)].to_numpy(float)], axis=1),
        "base_combined": np.concatenate([roi_features, physical], axis=1),
    }
    return groups


def run_logo(manifest: pd.DataFrame, matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    tasks = {
        "smoke_vs_hard_background": ("smoke", "hard_background"),
        "smoke_vs_fire": ("smoke", "fire"),
        "smoke_vs_person": ("smoke", "person"),
    }
    rows = []
    for task_name, (positive, negative) in tasks.items():
        mask = manifest["class_name"].isin([positive, negative]).to_numpy()
        labels = (manifest.loc[mask, "class_name"].to_numpy() == positive).astype(int)
        videos = manifest.loc[mask, "video"].to_numpy()
        for feature_set, matrix in matrices.items():
            x = matrix[mask]
            for video in sorted(set(videos)):
                test = videos == video
                train = ~test
                if labels[train].sum() == 0 or (labels[train] == 0).sum() == 0:
                    continue
                if labels[test].sum() == 0 or (labels[test] == 0).sum() == 0:
                    continue
                metrics = evaluate_fold(x[train], labels[train], x[test], labels[test])
                rows.append({"task": task_name, "feature_set": feature_set, "heldout_video": video, **metrics})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    base = results[results["feature_set"] == "base"][["task", "heldout_video", "auc", "pr_auc", "precision", "recall"]]
    base = base.rename(columns={c: f"base_{c}" for c in ("auc", "pr_auc", "precision", "recall")})
    joined = results.merge(base, on=["task", "heldout_video"], how="left")
    for metric in ("auc", "pr_auc", "precision", "recall"):
        joined[f"delta_{metric}"] = joined[metric] - joined[f"base_{metric}"]
    summary = joined.groupby(["task", "feature_set"], as_index=False).agg(
        folds=("heldout_video", "nunique"),
        auc_mean=("auc", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_pr_auc_mean=("delta_pr_auc", "mean"),
        delta_recall_mean=("delta_recall", "mean"),
        improved_auc_folds=("delta_auc", lambda x: int((x > 0).sum())),
        improved_pr_auc_folds=("delta_pr_auc", lambda x: int((x > 0).sum())),
    )
    return summary


def write_report(output: Path, manifest: pd.DataFrame, results: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# ADR-004 P04: physical information beyond frozen B2 ROI features",
        "",
        "## Conclusion",
        "",
    ]
    passed = []
    for _, row in summary[summary["feature_set"] != "base"].iterrows():
        if row.delta_auc_mean > 0.01 and row.improved_auc_folds >= max(1, row.folds - 1):
            passed.append(f"{row.task}/{row.feature_set}")
    if passed:
        lines.append("Some physical feature sets show stable positive increments: " + ", ".join(passed) + ".")
    else:
        lines.append("No physical feature set shows a stable cross-video increment over frozen B2 ROI features.")
    lines += [
        "",
        "## Data",
        "",
        f"- Samples: {len(manifest)}",
        f"- Videos: {manifest['video'].nunique()}",
        f"- Classes: {manifest['class_name'].value_counts().to_dict()}",
        "- Split: leave-one-video-out.",
        "- Base features: B2 frozen P2-P5 fused ROI average+max pooled features.",
        "",
        "## Mean Results",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Per-Video Results",
        "",
        dataframe_to_markdown(results),
        "",
        "## Files",
        "",
        "- `p04_manifest.csv`: sampled instances and hard-background boxes.",
        "- `b2_roi_features.npy`: cached frozen B2 ROI features.",
        "- `p04_logo_results.csv`: per-video metrics.",
        "- `p04_summary.csv`: aggregated metrics and Base deltas.",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.5f}"
        return str(value)

    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "p04_manifest.csv"
    if manifest_path.exists() and not args.force_features:
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_manifest(args)
        manifest.to_csv(manifest_path, index=False, float_format="%.8g")
    roi_features = extract_roi_features(args, manifest)
    matrices = build_design_matrices(manifest, roi_features)
    results = run_logo(manifest, matrices)
    summary = summarize(results)
    results.to_csv(args.output / "p04_logo_results.csv", index=False, float_format="%.8g")
    summary.to_csv(args.output / "p04_summary.csv", index=False, float_format="%.8g")
    settings = vars(args).copy()
    settings.update({"sample_count": len(manifest), "feature_dim": int(roi_features.shape[1])})
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    write_report(args.output, manifest, results, summary)
    print(f"P04 outputs saved to {args.output}")


if __name__ == "__main__":
    main()
