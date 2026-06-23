# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Profile class-specific RGB/IR physical properties for ADR-004 Pilot Study P01."""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


CLASS_NAMES = {0: "smoke", 1: "fire", 2: "person"}
SIZE_BINS = ((0, 32, "<32"), (32, 64, "32-64"), (64, 128, "64-128"), (128, 256, "128-256"),
             (256, float("inf"), ">=256"))
PRIMARY_METRICS = (
    "rgb_brightness_mean",
    "rgb_saturation_mean",
    "rgb_dark_mean",
    "rgb_std",
    "rgb_gradient_mean",
    "rgb_laplacian_energy",
    "rgb_edge_density",
    "rgb_spectral_high_low_log10",
    "rgb_spectral_slope",
    "rgb_contrast_abs",
    "rgb_std_ratio",
    "ir_mean",
    "ir_std",
    "ir_gradient_mean",
    "ir_laplacian_energy",
    "ir_edge_density",
    "ir_contrast_abs",
    "ir_std_ratio",
    "cross_gray_corr",
    "cross_gradient_corr",
    "cross_ssim",
    "cross_zmad",
    "cross_edge_iou",
    "cross_contrast_gap",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/xwh/dataset/RGBT-3M/RGBT-3M"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/detect/adr004/pilot/P01_physical_profile"))
    parser.add_argument("--splits", nargs="+", default=("train", "val"), choices=("train", "val"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ring-scale", type=float, default=1.75)
    parser.add_argument("--clean-overlap", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split image limit for smoke tests.")
    return parser.parse_args()


def size_bin(width: int, height: int) -> str:
    """Return the ADR-004 size bin based on the maximum box side."""
    side = max(width, height)
    return next(name for lower, upper, name in SIZE_BINS if lower <= side < upper)


def yolo_box_to_xyxy(values: Iterable[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Convert a normalized YOLO box to clipped integer xyxy coordinates."""
    x_center, y_center, width, height = values
    x1 = max(0, min(image_width - 1, math.floor((x_center - width / 2) * image_width)))
    y1 = max(0, min(image_height - 1, math.floor((y_center - height / 2) * image_height)))
    x2 = max(x1 + 1, min(image_width, math.ceil((x_center + width / 2) * image_width)))
    y2 = max(y1 + 1, min(image_height, math.ceil((y_center + height / 2) * image_height)))
    return x1, y1, x2, y2


def read_labels(path: Path, image_width: int, image_height: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Read one YOLO label file."""
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(float(fields[0]))
        if class_id not in CLASS_NAMES:
            continue
        labels.append((class_id, yolo_box_to_xyxy(map(float, fields[1:5]), image_width, image_height)))
    return labels


def make_ring_mask(
    box: tuple[int, int, int, int],
    all_boxes: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """Build an expanded background ring while excluding every GT box."""
    height, width = image_shape
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    expanded_width, expanded_height = (x2 - x1) * scale, (y2 - y1) * scale
    ex1 = max(0, int(math.floor(center_x - expanded_width / 2)))
    ey1 = max(0, int(math.floor(center_y - expanded_height / 2)))
    ex2 = min(width, int(math.ceil(center_x + expanded_width / 2)))
    ey2 = min(height, int(math.ceil(center_y + expanded_height / 2)))
    mask = np.zeros((height, width), dtype=bool)
    mask[ey1:ey2, ex1:ex2] = True
    for bx1, by1, bx2, by2 in all_boxes:
        mask[by1:by2, bx1:bx2] = False
    return mask


def overlap_fraction(box: tuple[int, int, int, int], other_boxes: list[tuple[int, int, int, int]]) -> float:
    """Return the fraction of a box covered by the union of other GT boxes."""
    x1, y1, x2, y2 = box
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for ox1, oy1, ox2, oy2 in other_boxes:
        ix1, iy1, ix2, iy2 = max(x1, ox1), max(y1, oy1), min(x2, ox2), min(y2, oy2)
        if ix1 < ix2 and iy1 < iy2:
            mask[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = True
    return float(mask.mean()) if mask.size else float("nan")


def safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate Pearson correlation without emitting warnings for constant arrays."""
    first = first.astype(np.float64, copy=False).ravel()
    second = second.astype(np.float64, copy=False).ravel()
    if first.size < 2 or second.size != first.size or first.std() < 1e-8 or second.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate the single-window SSIM of two aligned normalized patches."""
    first = first.astype(np.float64, copy=False).ravel()
    second = second.astype(np.float64, copy=False).ravel()
    if first.size < 2 or second.size != first.size:
        return float("nan")
    c1, c2 = 0.01**2, 0.03**2
    mean_first, mean_second = first.mean(), second.mean()
    var_first, var_second = first.var(), second.var()
    covariance = np.mean((first - mean_first) * (second - mean_second))
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (mean_first**2 + mean_second**2 + c1) * (var_first + var_second + c2)
    return float(numerator / denominator) if denominator > 0 else float("nan")


def spectral_metrics(patch: np.ndarray, resolution: int = 64) -> tuple[float, float]:
    """Return log high/low spectral-energy ratio and radial log-power slope."""
    if min(patch.shape) < 3:
        return float("nan"), float("nan")
    resized = cv2.resize(patch, (resolution, resolution), interpolation=cv2.INTER_AREA).astype(np.float64)
    resized -= resized.mean()
    window = np.outer(np.hanning(resolution), np.hanning(resolution))
    power = np.abs(np.fft.fftshift(np.fft.fft2(resized * window)))**2
    frequencies = np.fft.fftshift(np.fft.fftfreq(resolution))
    radius = np.sqrt(frequencies[:, None]**2 + frequencies[None, :]**2)
    low = power[(radius >= 0.05) & (radius < 0.20)].mean()
    high = power[(radius >= 0.30) & (radius <= 0.50)].mean()
    high_low_log10 = math.log10((high + 1e-12) / (low + 1e-12))

    centers, radial_power = [], []
    for lower, upper in zip(np.linspace(0.04, 0.50, 13)[:-1], np.linspace(0.04, 0.50, 13)[1:]):
        values = power[(radius >= lower) & (radius < upper)]
        if values.size:
            centers.append((lower + upper) / 2)
            radial_power.append(float(np.median(values)))
    slope = np.polyfit(np.log10(centers), np.log10(np.asarray(radial_power) + 1e-12), 1)[0]
    return float(high_low_log10), float(slope)


def region_metrics(
    prefix: str,
    gray: np.ndarray,
    gradient: np.ndarray,
    laplacian: np.ndarray,
    edges: np.ndarray,
    box: tuple[int, int, int, int],
    ring_mask: np.ndarray,
    dark: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure object and paired background-ring statistics."""
    x1, y1, x2, y2 = box
    object_gray = gray[y1:y2, x1:x2]
    object_gradient = gradient[y1:y2, x1:x2]
    object_laplacian = laplacian[y1:y2, x1:x2]
    object_edges = edges[y1:y2, x1:x2]
    ring_gray = gray[ring_mask]
    ring_gradient = gradient[ring_mask]
    ring_laplacian = laplacian[ring_mask]
    ring_edges = edges[ring_mask]
    metrics = {
        f"{prefix}_mean": float(object_gray.mean()),
        f"{prefix}_std": float(object_gray.std()),
        f"{prefix}_gradient_mean": float(object_gradient.mean()),
        f"{prefix}_laplacian_energy": float(np.mean(object_laplacian**2)),
        f"{prefix}_edge_density": float(object_edges.mean()),
        f"{prefix}_ring_mean": float(ring_gray.mean()) if ring_gray.size else float("nan"),
        f"{prefix}_ring_std": float(ring_gray.std()) if ring_gray.size else float("nan"),
        f"{prefix}_ring_gradient_mean": float(ring_gradient.mean()) if ring_gradient.size else float("nan"),
        f"{prefix}_ring_laplacian_energy": (float(np.mean(ring_laplacian**2))
                                             if ring_laplacian.size else float("nan")),
        f"{prefix}_ring_edge_density": float(ring_edges.mean()) if ring_edges.size else float("nan"),
    }
    metrics[f"{prefix}_contrast_signed"] = metrics[f"{prefix}_mean"] - metrics[f"{prefix}_ring_mean"]
    metrics[f"{prefix}_contrast_abs"] = abs(metrics[f"{prefix}_contrast_signed"])
    metrics[f"{prefix}_std_ratio"] = metrics[f"{prefix}_std"] / (metrics[f"{prefix}_ring_std"] + 1e-8)
    metrics[f"{prefix}_gradient_ratio"] = metrics[f"{prefix}_gradient_mean"] / (
        metrics[f"{prefix}_ring_gradient_mean"] + 1e-8)
    if dark is not None:
        object_dark, ring_dark = dark[y1:y2, x1:x2], dark[ring_mask]
        metrics[f"{prefix}_dark_mean"] = float(object_dark.mean())
        metrics[f"{prefix}_ring_dark_mean"] = float(ring_dark.mean()) if ring_dark.size else float("nan")
        metrics[f"{prefix}_dark_delta"] = metrics[f"{prefix}_dark_mean"] - metrics[f"{prefix}_ring_dark_mean"]
    return metrics


def load_image_features(rgb_path: Path, ir_path: Path) -> dict[str, np.ndarray]:
    """Load aligned RGB/IR images and precompute dense physical maps."""
    rgb_bgr, ir_bgr = cv2.imread(str(rgb_path)), cv2.imread(str(ir_path))
    if rgb_bgr is None or ir_bgr is None:
        raise FileNotFoundError(f"Unable to load aligned pair: {rgb_path}, {ir_path}")
    if rgb_bgr.shape[:2] != ir_bgr.shape[:2]:
        raise ValueError(f"RGB/IR shape mismatch: {rgb_path}={rgb_bgr.shape}, {ir_path}={ir_bgr.shape}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    rgb_hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    rgb_hsv[..., 0] /= 179.0
    rgb_hsv[..., 1:] /= 255.0

    def derivatives(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(sobel_x, sobel_y)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edges = cv2.Canny(np.uint8(np.clip(gray * 255, 0, 255)), 50, 150).astype(np.float32) / 255.0
        return gradient, laplacian, edges

    rgb_gradient, rgb_laplacian, rgb_edges = derivatives(rgb_gray)
    ir_gradient, ir_laplacian, ir_edges = derivatives(ir_gray)
    dark = cv2.erode(rgb.min(axis=2), np.ones((15, 15), dtype=np.uint8))
    rgb_z = (rgb_gray - rgb_gray.mean()) / (rgb_gray.std() + 1e-8)
    ir_z = (ir_gray - ir_gray.mean()) / (ir_gray.std() + 1e-8)
    return {
        "rgb": rgb,
        "rgb_gray": rgb_gray,
        "rgb_hsv": rgb_hsv,
        "rgb_gradient": rgb_gradient,
        "rgb_laplacian": rgb_laplacian,
        "rgb_edges": rgb_edges,
        "rgb_dark": dark,
        "rgb_z": rgb_z,
        "ir_gray": ir_gray,
        "ir_gradient": ir_gradient,
        "ir_laplacian": ir_laplacian,
        "ir_edges": ir_edges,
        "ir_z": ir_z,
    }


def process_image(task: tuple[str, str, str, float]) -> list[dict[str, float | int | str]]:
    """Profile every labeled instance in one aligned image pair."""
    root_value, split, image_id, ring_scale = task
    cv2.setNumThreads(0)
    root = Path(root_value)
    rgb_path = root / "RGB" / split / f"{image_id}.jpg"
    ir_path = root / "IR" / split / f"{image_id}.jpg"
    features = load_image_features(rgb_path, ir_path)
    height, width = features["rgb_gray"].shape
    labels = read_labels(root / "labels" / split / f"{image_id}.txt", width, height)
    boxes = [box for _, box in labels]
    video_match = re.match(r"(video\d+)_", image_id)
    video = video_match.group(1) if video_match else "unknown"
    rows = []

    for object_index, (class_id, box) in enumerate(labels):
        x1, y1, x2, y2 = box
        box_width, box_height = x2 - x1, y2 - y1
        ring_mask = make_ring_mask(box, boxes, (height, width), ring_scale)
        ring_pixels = int(ring_mask.sum())
        rgb_patch = features["rgb_gray"][y1:y2, x1:x2]
        ir_patch = features["ir_gray"][y1:y2, x1:x2]
        rgb_gradient_patch = features["rgb_gradient"][y1:y2, x1:x2]
        ir_gradient_patch = features["ir_gradient"][y1:y2, x1:x2]
        rgb_edges_patch = features["rgb_edges"][y1:y2, x1:x2] > 0
        ir_edges_patch = features["ir_edges"][y1:y2, x1:x2] > 0
        high_low, slope = spectral_metrics(rgb_patch)
        union_edges = np.logical_or(rgb_edges_patch, ir_edges_patch).sum()

        row: dict[str, float | int | str] = {
            "split": split,
            "image_id": image_id,
            "video": video,
            "object_index": object_index,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "box_width": box_width,
            "box_height": box_height,
            "box_area": box_width * box_height,
            "size_bin": size_bin(box_width, box_height),
            "ring_pixels": ring_pixels,
            "ring_to_box_ratio": ring_pixels / max(1, box_width * box_height),
            "overlap_fraction": overlap_fraction(box, [other for index, other in enumerate(boxes)
                                                        if index != object_index]),
            "rgb_spectral_high_low_log10": high_low,
            "rgb_spectral_slope": slope,
        }
        row.update(region_metrics("rgb", features["rgb_gray"], features["rgb_gradient"],
                                  features["rgb_laplacian"], features["rgb_edges"], box, ring_mask,
                                  features["rgb_dark"]))
        row.update(region_metrics("ir", features["ir_gray"], features["ir_gradient"],
                                  features["ir_laplacian"], features["ir_edges"], box, ring_mask))
        rgb_pixels = features["rgb"][y1:y2, x1:x2]
        saturation = features["rgb_hsv"][y1:y2, x1:x2, 1]
        ring_saturation = features["rgb_hsv"][..., 1][ring_mask]
        row.update({
            "rgb_brightness_mean": row["rgb_mean"],
            "rgb_saturation_mean": float(saturation.mean()),
            "rgb_ring_saturation_mean": float(ring_saturation.mean()) if ring_saturation.size else float("nan"),
            "rgb_saturation_delta": (float(saturation.mean() - ring_saturation.mean())
                                     if ring_saturation.size else float("nan")),
            "rgb_channel_min_mean": float(rgb_pixels.min(axis=2).mean()),
            "rgb_channel_mean": float(rgb_pixels.mean()),
            "rgb_channel_max_mean": float(rgb_pixels.max(axis=2).mean()),
            "cross_gray_corr": safe_corr(rgb_patch, ir_patch),
            "cross_gradient_corr": safe_corr(rgb_gradient_patch, ir_gradient_patch),
            "cross_ssim": global_ssim(rgb_patch, ir_patch),
            "cross_zmad": float(np.mean(np.abs(features["rgb_z"][y1:y2, x1:x2]
                                                 - features["ir_z"][y1:y2, x1:x2]))),
            "cross_edge_iou": (float(np.logical_and(rgb_edges_patch, ir_edges_patch).sum() / union_edges)
                               if union_edges else float("nan")),
            "cross_contrast_gap": abs(float(row["rgb_contrast_abs"]) - float(row["ir_contrast_abs"])),
        })
        rows.append(row)
    return rows


def auc_and_cliff(smoke: np.ndarray, other: np.ndarray) -> tuple[float, float, str]:
    """Return orientation-free AUC, signed Cliff's delta, and smoke direction."""
    smoke = smoke[np.isfinite(smoke)]
    other = other[np.isfinite(other)]
    if not smoke.size or not other.size:
        return float("nan"), float("nan"), "unknown"
    combined = np.concatenate((smoke, other))
    ranks = rankdata(combined, method="average")
    raw_auc = (ranks[:smoke.size].sum() - smoke.size * (smoke.size + 1) / 2) / (smoke.size * other.size)
    cliff = 2 * raw_auc - 1
    return float(max(raw_auc, 1 - raw_auc)), float(cliff), "higher" if cliff >= 0 else "lower"


def cluster_bootstrap_mean_ci(
    frame: pd.DataFrame,
    metric: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap the macro mean over videos to reflect between-video variability."""
    video_means = frame.groupby("video", observed=True)[metric].mean().dropna().to_numpy(dtype=float)
    if not video_means.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(video_means, size=(iterations, video_means.size), replace=True).mean(axis=1)
    return float(video_means.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def build_class_summary(
    frame: pd.DataFrame,
    clean_overlap: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize physical metrics by class, split, and overlap sensitivity subset."""
    rows = []
    for subset_name, subset in (("full", frame), ("low_overlap", frame[frame["overlap_fraction"] <= clean_overlap])):
        for split_name in ("all", "train", "val"):
            split_frame = subset if split_name == "all" else subset[subset["split"] == split_name]
            for class_name, class_frame in split_frame.groupby("class_name", observed=True):
                for metric in PRIMARY_METRICS:
                    values = class_frame[metric].dropna().to_numpy(dtype=float)
                    if not values.size:
                        continue
                    macro_mean, ci_low, ci_high = cluster_bootstrap_mean_ci(
                        class_frame, metric, iterations, seed + len(rows))
                    rows.append({
                        "subset": subset_name,
                        "split": split_name,
                        "class_name": class_name,
                        "metric": metric,
                        "n": values.size,
                        "mean": values.mean(),
                        "median": np.median(values),
                        "q1": np.quantile(values, 0.25),
                        "q3": np.quantile(values, 0.75),
                        "std": values.std(),
                        "video_macro_mean": macro_mean,
                        "video_bootstrap_ci95_low": ci_low,
                        "video_bootstrap_ci95_high": ci_high,
                    })
    return pd.DataFrame(rows)


def build_effect_tables(frame: pd.DataFrame, clean_overlap: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build global and video-stratified smoke-vs-other effect tables."""
    effects, video_effects = [], []
    for subset_name, subset in (("full", frame), ("low_overlap", frame[frame["overlap_fraction"] <= clean_overlap])):
        for split_name in ("all", "val"):
            split_frame = subset if split_name == "all" else subset[subset["split"] == split_name]
            for other_class in ("fire", "person"):
                smoke_frame = split_frame[split_frame["class_name"] == "smoke"]
                other_frame = split_frame[split_frame["class_name"] == other_class]
                for metric in PRIMARY_METRICS:
                    smoke = smoke_frame[metric].to_numpy(dtype=float)
                    other = other_frame[metric].to_numpy(dtype=float)
                    auc, cliff, direction = auc_and_cliff(smoke, other)
                    per_video = []
                    for video in sorted(set(smoke_frame["video"]) & set(other_frame["video"])):
                        video_smoke = smoke_frame.loc[smoke_frame["video"] == video, metric].to_numpy(dtype=float)
                        video_other = other_frame.loc[other_frame["video"] == video, metric].to_numpy(dtype=float)
                        if np.isfinite(video_smoke).sum() < 10 or np.isfinite(video_other).sum() < 10:
                            continue
                        video_auc, video_cliff, video_direction = auc_and_cliff(video_smoke, video_other)
                        per_video.append(video_cliff)
                        video_effects.append({
                            "subset": subset_name,
                            "split": split_name,
                            "comparison": f"smoke_vs_{other_class}",
                            "video": video,
                            "metric": metric,
                            "smoke_n": np.isfinite(video_smoke).sum(),
                            "other_n": np.isfinite(video_other).sum(),
                            "auc": video_auc,
                            "cliff_delta": video_cliff,
                            "smoke_direction": video_direction,
                        })
                    consistent = (np.mean(np.sign(per_video) == np.sign(cliff)) if per_video and cliff != 0 else float("nan"))
                    finite_smoke = smoke[np.isfinite(smoke)]
                    finite_other = other[np.isfinite(other)]
                    effects.append({
                        "subset": subset_name,
                        "split": split_name,
                        "comparison": f"smoke_vs_{other_class}",
                        "metric": metric,
                        "smoke_n": np.isfinite(smoke).sum(),
                        "other_n": np.isfinite(other).sum(),
                        "smoke_mean": finite_smoke.mean() if finite_smoke.size else float("nan"),
                        "other_mean": finite_other.mean() if finite_other.size else float("nan"),
                        "auc": auc,
                        "cliff_delta": cliff,
                        "smoke_direction": direction,
                        "valid_videos": len(per_video),
                        "video_direction_consistency": consistent,
                    })
    return pd.DataFrame(effects), pd.DataFrame(video_effects)


def build_size_effects(frame: pd.DataFrame, clean_overlap: float) -> pd.DataFrame:
    """Build smoke-vs-other effect sizes independently within every object-size bin."""
    rows = []
    for subset_name, subset in (("full", frame), ("low_overlap", frame[frame["overlap_fraction"] <= clean_overlap])):
        for split_name in ("all", "val"):
            split_frame = subset if split_name == "all" else subset[subset["split"] == split_name]
            for size in (name for _, _, name in SIZE_BINS):
                size_frame = split_frame[split_frame["size_bin"] == size]
                smoke_frame = size_frame[size_frame["class_name"] == "smoke"]
                for other_class in ("fire", "person"):
                    other_frame = size_frame[size_frame["class_name"] == other_class]
                    for metric in PRIMARY_METRICS:
                        smoke = smoke_frame[metric].to_numpy(dtype=float)
                        other = other_frame[metric].to_numpy(dtype=float)
                        auc, cliff, direction = auc_and_cliff(smoke, other)
                        rows.append({
                            "subset": subset_name,
                            "split": split_name,
                            "size_bin": size,
                            "comparison": f"smoke_vs_{other_class}",
                            "metric": metric,
                            "smoke_n": np.isfinite(smoke).sum(),
                            "other_n": np.isfinite(other).sum(),
                            "auc": auc,
                            "cliff_delta": cliff,
                            "smoke_direction": direction,
                        })
    return pd.DataFrame(rows)


def build_stratified_effects(frame: pd.DataFrame, clean_overlap: float) -> pd.DataFrame:
    """Macro-average Cliff's delta over video and size strata to reduce dataset-composition confounding."""
    rows = []
    for subset_name, subset in (("full", frame), ("low_overlap", frame[frame["overlap_fraction"] <= clean_overlap])):
        for split_name in ("all", "val"):
            split_frame = subset if split_name == "all" else subset[subset["split"] == split_name]
            for other_class in ("fire", "person"):
                for metric in PRIMARY_METRICS:
                    stratum_deltas = []
                    for (video, size), group in split_frame.groupby(["video", "size_bin"], observed=True):
                        smoke = group.loc[group["class_name"] == "smoke", metric].to_numpy(dtype=float)
                        other = group.loc[group["class_name"] == other_class, metric].to_numpy(dtype=float)
                        if np.isfinite(smoke).sum() < 10 or np.isfinite(other).sum() < 10:
                            continue
                        _, cliff, _ = auc_and_cliff(smoke, other)
                        stratum_deltas.append((video, size, cliff))
                    deltas = np.asarray([item[2] for item in stratum_deltas], dtype=float)
                    macro_cliff = float(deltas.mean()) if deltas.size else float("nan")
                    rows.append({
                        "subset": subset_name,
                        "split": split_name,
                        "comparison": f"smoke_vs_{other_class}",
                        "metric": metric,
                        "valid_video_size_strata": deltas.size,
                        "macro_cliff_delta": macro_cliff,
                        "macro_auc": 0.5 + abs(macro_cliff) / 2 if np.isfinite(macro_cliff) else float("nan"),
                        "median_cliff_delta": float(np.median(deltas)) if deltas.size else float("nan"),
                        "direction_consistency": (float(np.mean(np.sign(deltas) == np.sign(macro_cliff)))
                                                  if deltas.size and macro_cliff != 0 else float("nan")),
                        "strata": ";".join(f"{video}:{size}" for video, size, _ in stratum_deltas),
                    })
    return pd.DataFrame(rows)


def build_video_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each metric by split, video, and class."""
    rows = []
    for (split, video, class_name), group in frame.groupby(["split", "video", "class_name"], observed=True):
        for metric in PRIMARY_METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size:
                rows.append({
                    "split": split,
                    "video": video,
                    "class_name": class_name,
                    "metric": metric,
                    "n": values.size,
                    "mean": values.mean(),
                    "median": np.median(values),
                    "q1": np.quantile(values, 0.25),
                    "q3": np.quantile(values, 0.75),
                })
    return pd.DataFrame(rows)


def build_size_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each metric by split, size bin, and class."""
    rows = []
    for (split, size, class_name), group in frame.groupby(["split", "size_bin", "class_name"], observed=True):
        for metric in PRIMARY_METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size:
                rows.append({
                    "split": split,
                    "size_bin": size,
                    "class_name": class_name,
                    "metric": metric,
                    "n": values.size,
                    "mean": values.mean(),
                    "median": np.median(values),
                    "q1": np.quantile(values, 0.25),
                    "q3": np.quantile(values, 0.75),
                })
    return pd.DataFrame(rows)


def plot_distributions(frame: pd.DataFrame, output: Path, seed: int) -> None:
    """Plot representative class-wise physical distributions."""
    metrics = (
        "rgb_saturation_mean",
        "rgb_dark_mean",
        "rgb_gradient_mean",
        "rgb_spectral_high_low_log10",
        "rgb_contrast_abs",
        "ir_contrast_abs",
        "cross_gradient_corr",
        "cross_zmad",
        "cross_contrast_gap",
    )
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    classes = ("smoke", "fire", "person")
    colors = ("#777777", "#d95f02", "#1b9e77")
    for axis, metric in zip(axes.flat, metrics):
        data = []
        for class_name in classes:
            values = frame.loc[frame["class_name"] == class_name, metric].dropna().to_numpy(dtype=float)
            if values.size > 3000:
                values = rng.choice(values, size=3000, replace=False)
            data.append(values)
        box = axis.boxplot(data, tick_labels=classes, showfliers=False, patch_artist=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ADR-004 P01: Class-wise Physical Properties", fontsize=16)
    fig.tight_layout()
    fig.savefig(output / "class_physical_distributions.png", dpi=180)
    plt.close(fig)


def plot_size_trends(size_summary: pd.DataFrame, output: Path) -> None:
    """Plot class medians across the ADR-004 size bins."""
    metrics = ("rgb_gradient_mean", "rgb_laplacian_energy", "rgb_dark_mean", "ir_gradient_mean",
               "ir_contrast_abs", "cross_edge_iou")
    bins = tuple(name for _, _, name in SIZE_BINS)
    classes = ("smoke", "fire", "person")
    colors = {"smoke": "#777777", "fire": "#d95f02", "person": "#1b9e77"}
    weighted = size_summary.assign(weighted_mean=size_summary["mean"] * size_summary["n"])
    combined = weighted.groupby(["size_bin", "class_name", "metric"], observed=True).agg(
        weighted_sum=("weighted_mean", "sum"), n=("n", "sum")).reset_index()
    combined["mean"] = combined["weighted_sum"] / combined["n"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, metric in zip(axes.flat, metrics):
        metric_frame = combined[combined["metric"] == metric]
        for class_name in classes:
            class_frame = metric_frame[metric_frame["class_name"] == class_name].set_index("size_bin")
            values = [class_frame.at[size, "mean"] if size in class_frame.index and class_frame.at[size, "n"] >= 30
                      else np.nan for size in bins]
            axis.plot(bins, values, marker="o", label=class_name, color=colors[class_name])
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("ADR-004 P01: Physical Properties by Object Size", fontsize=16)
    fig.tight_layout()
    fig.savefig(output / "size_stratified_trends.png", dpi=180)
    plt.close(fig)


def plot_effect_heatmap(effects: pd.DataFrame, output: Path) -> None:
    """Plot signed Cliff's delta for the primary class comparisons."""
    selected = effects[(effects["subset"] == "full") & (effects["split"] == "all")]
    pivot = selected.pivot(index="metric", columns="comparison", values="cliff_delta").reindex(PRIMARY_METRICS)
    fig, axis = plt.subplots(figsize=(7, 11))
    image = axis.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Global Unadjusted Cliff's Delta (positive = smoke higher)")
    fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output / "class_effect_heatmap.png", dpi=180)
    plt.close(fig)


def plot_stratified_effect_heatmap(stratified_effects: pd.DataFrame, output: Path) -> None:
    """Plot video-by-size stratified macro Cliff's delta."""
    selected = stratified_effects[(stratified_effects["subset"] == "full")
                                  & (stratified_effects["split"] == "all")]
    pivot = selected.pivot(index="metric", columns="comparison", values="macro_cliff_delta").reindex(PRIMARY_METRICS)
    fig, axis = plt.subplots(figsize=(7, 11))
    image = axis.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Video x Size Stratified Cliff's Delta (positive = smoke higher)")
    fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output / "stratified_effect_heatmap.png", dpi=180)
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    output: Path,
    settings: dict[str, object],
) -> None:
    """Write a concise evidence report from the generated statistics."""

    def markdown_table(table: pd.DataFrame, float_digits: int = 4) -> str:
        """Render a small DataFrame without pandas' optional tabulate dependency."""
        columns = [str(column) for column in table.columns]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for values in table.itertuples(index=False, name=None):
            cells = []
            for value in values:
                if isinstance(value, (float, np.floating)):
                    cells.append("nan" if not np.isfinite(value) else f"{value:.{float_digits}f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    main = effects[(effects["subset"] == "full") & (effects["split"] == "all")].copy()
    main["passes_auc"] = main["auc"] >= 0.70
    strongest = main.sort_values(["comparison", "auc"], ascending=[True, False]).groupby("comparison").head(10)
    class_counts = frame.groupby(["split", "class_name"], observed=True).size().unstack(fill_value=0)
    clean_count = int((frame["overlap_fraction"] <= float(settings["clean_overlap"])).sum())

    lines = [
        "# ADR-004 P01 RGB/IR 物理属性画像报告",
        "",
        "## 1. 运行设置",
        "",
        f"- 数据集根目录：`{settings['root']}`",
        f"- 处理图像：`{settings['image_count']}`",
        f"- 处理实例：`{len(frame)}`",
        f"- 背景环扩展系数：`{settings['ring_scale']}`",
        f"- 低重叠阈值：`overlap_fraction <= {settings['clean_overlap']}`，保留 `{clean_count}` 个实例",
        f"- bootstrap：按 video 聚类，`{settings['bootstrap']}` 次",
        "",
        "## 2. 实例数量",
        "",
        markdown_table(class_counts.reset_index(), float_digits=0),
        "",
        "## 3. 最强单变量类别差异",
        "",
        "`cliff_delta > 0` 表示 smoke 更高，`< 0` 表示 smoke 更低；AUC 已转换为方向无关值。",
        "",
        markdown_table(strongest[["comparison", "metric", "smoke_mean", "other_mean", "auc", "cliff_delta",
                                  "video_direction_consistency"]]),
        "",
        "## 4. P01 门槛检查",
        "",
    ]
    for comparison in ("smoke_vs_fire", "smoke_vs_person"):
        passed = main[(main["comparison"] == comparison) & main["passes_auc"]]
        lines.append(f"- `{comparison}`：{len(passed)} / {len(PRIMARY_METRICS)} 个指标达到 AUC >= 0.70。")

    lines.extend((
        "",
        "P01 只验证类别物理差异，不直接证明某个先验能够修复 B2。是否进入结构实验还需要 P02/P04 对",
        "`a1_hit_b2_miss` 和冻结特征增量信息进行验证。",
        "",
        "## 5. 产物",
        "",
        "- `instance_physical_metrics.csv`：逐实例原始统计",
        "- `class_summary.csv`：类别、split 和重叠敏感性汇总",
        "- `video_stratified_summary.csv`：按视频统计",
        "- `size_stratified_summary.csv`：按目标尺寸统计",
        "- `pairwise_effects.csv`：全局 AUC 与 Cliff's delta",
        "- `video_pairwise_effects.csv`：逐视频效应量",
        "- `size_pairwise_effects.csv`：逐尺寸类别效应量",
        "- `stratified_pairwise_effects.csv`：video × size 分层宏观效应量",
        "- `class_physical_distributions.png`：代表性指标分布",
        "- `class_effect_heatmap.png`：类别效应方向",
        "- `stratified_effect_heatmap.png`：控制 video × size 后的类别效应方向",
        "- `size_stratified_trends.png`：类别物理量的尺寸趋势",
        "",
        "## 6. 统计边界",
        "",
        "- RGBT-3M 提供检测框而非 smoke mask，框内统计不可避免包含部分背景。",
        "- smoke 与 fire 经常共现，因此同时提供 `low_overlap` 敏感性分析。",
        "- train/val 来自相同视频序列，不能把相邻帧当作完全独立样本；本报告以跨视频方向一致性为主要稳健性依据。",
        "- 原始图像统计只能建立相关性，不能单独证明网络内部响应退化机制。",
        "",
    ))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run ADR-004 Pilot Study P01."""
    args = parse_args()
    if args.ring_scale <= 1:
        raise ValueError(f"--ring-scale must be greater than 1, got {args.ring_scale}")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for split in args.splits:
        ids = [line.strip() for line in (args.root / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
        if args.limit is not None:
            ids = ids[:args.limit]
        tasks.extend((str(args.root), split, image_id, args.ring_scale) for image_id in ids)

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, image_rows in enumerate(executor.map(process_image, tasks, chunksize=8), 1):
            rows.extend(image_rows)
            if index % 500 == 0 or index == len(tasks):
                print(f"Processed {index}/{len(tasks)} images, {len(rows)} instances", flush=True)
    if not rows:
        raise RuntimeError("No labeled instances were found.")

    frame = pd.DataFrame(rows).sort_values(["split", "image_id", "object_index"]).reset_index(drop=True)
    summary = build_class_summary(frame, args.clean_overlap, args.bootstrap, args.seed)
    effects, video_effects = build_effect_tables(frame, args.clean_overlap)
    size_effects = build_size_effects(frame, args.clean_overlap)
    stratified_effects = build_stratified_effects(frame, args.clean_overlap)
    video_summary = build_video_summary(frame)
    size_summary = build_size_summary(frame)
    frame.to_csv(args.output / "instance_physical_metrics.csv", index=False, float_format="%.8g")
    summary.to_csv(args.output / "class_summary.csv", index=False, float_format="%.8g")
    effects.to_csv(args.output / "pairwise_effects.csv", index=False, float_format="%.8g")
    video_effects.to_csv(args.output / "video_pairwise_effects.csv", index=False, float_format="%.8g")
    size_effects.to_csv(args.output / "size_pairwise_effects.csv", index=False, float_format="%.8g")
    stratified_effects.to_csv(args.output / "stratified_pairwise_effects.csv", index=False, float_format="%.8g")
    video_summary.to_csv(args.output / "video_stratified_summary.csv", index=False, float_format="%.8g")
    size_summary.to_csv(args.output / "size_stratified_summary.csv", index=False, float_format="%.8g")
    plot_distributions(frame, args.output, args.seed)
    plot_effect_heatmap(effects, args.output)
    plot_stratified_effect_heatmap(stratified_effects, args.output)
    plot_size_trends(size_summary, args.output)
    settings = {
        "root": str(args.root),
        "splits": list(args.splits),
        "image_count": len(tasks),
        "instance_count": len(frame),
        "ring_scale": args.ring_scale,
        "clean_overlap": args.clean_overlap,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "workers": args.workers,
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    write_report(frame, summary, effects, args.output, settings)
    print(f"P01 outputs saved to {args.output}")


if __name__ == "__main__":
    main()
