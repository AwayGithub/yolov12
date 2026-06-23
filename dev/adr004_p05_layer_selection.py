# Ultralytics YOLO 🚀, AGPL-3.0 license
"""ADR-004 P05: determine which feature layer benefits most from physical features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import rankdata

from ultralytics import YOLO
from ultralytics.nn.modules import DMGFusion, DMGFusionInit8d, DMGFusionPosAlpha, FreDFTFusion, M2DLocalIlluminationFusion


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
PHYSICAL_COLUMNS = list(RGB_METRICS + IR_METRICS + CROSS_METRICS)
IMAGE_SIZE = (480, 640)
STAGES = ("p2", "p3", "p4", "p5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p04-dir", type=Path, default=Path("runs/detect/adr004/pilot/P04_physical_increment"))
    parser.add_argument("--weights", type=Path,
                        default=Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4/weights/last.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/detect/adr004/pilot/P05_layer_selection"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


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


def ensure_pcross_compat(model: torch.nn.Module, device: torch.device) -> None:
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


def extract_per_layer_features(args: argparse.Namespace, manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    cache = args.output / "b2_roi_features_per_layer.npz"
    if cache.exists() and not args.force_features:
        loaded = np.load(cache)
        return {stage: loaded[stage] for stage in STAGES}
    device = select_device(args.device)
    model = YOLO(str(args.weights)).model.to(device).eval()
    ensure_pcross_compat(model, device)
    root = Path(manifest["root"].iloc[0]) if "root" in manifest.columns else Path("/data/xwh/dataset/RGBT-3M/RGBT-3M")
    features = {stage: [] for stage in STAGES}
    grouped = manifest.groupby(["split", "image_id"], sort=False)
    total = len(grouped)
    with torch.no_grad():
        for index, ((split, image_id), group) in enumerate(grouped, 1):
            tensor, image_shape = load_pair_tensor(root, split, image_id, device)
            fused = fused_features(model, tensor)
            for _, row in group.iterrows():
                box = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
                for stage in STAGES:
                    features[stage].append(roi_vector(fused[stage], box, image_shape))
            if index % 50 == 0 or index == total:
                print(f"Extracted per-layer B2 ROI features for {index}/{total} images", flush=True)
    arrays = {stage: np.stack(features[stage]).astype(np.float32) for stage in STAGES}
    np.savez(cache, **arrays)
    return arrays


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


def run_logo(manifest: pd.DataFrame, matrix: np.ndarray, feature_set: str, stage: str) -> pd.DataFrame:
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
        x = matrix[mask]
        for video in sorted(set(videos)):
            test = videos == video
            train = ~test
            if labels[train].sum() == 0 or (labels[train] == 0).sum() == 0:
                continue
            if labels[test].sum() == 0 or (labels[test] == 0).sum() == 0:
                continue
            metrics = evaluate_fold(x[train], labels[train], x[test], labels[test])
            rows.append({"task": task_name, "feature_set": feature_set, "stage": stage,
                         "heldout_video": video, **metrics})
    return pd.DataFrame(rows)


def build_design_matrices(per_layer: dict[str, np.ndarray], physical: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    groups = {}
    for stage in STAGES:
        base = per_layer[stage]
        groups[stage] = {
            f"base_{stage}": base,
            f"base_{stage}_rgb": np.concatenate([base, physical[:, :6]], axis=1),
            f"base_{stage}_ir": np.concatenate([base, physical[:, 6:12]], axis=1),
            f"base_{stage}_combined": np.concatenate([base, physical], axis=1),
        }
    return groups


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    base_rows = results[results["feature_set"].str.startswith("base_") & ~results["feature_set"].str.endswith(("_rgb", "_ir", "_combined"))]
    base_rows = base_rows[["task", "stage", "heldout_video", "auc", "pr_auc", "recall"]].rename(
        columns={"auc": "base_auc", "pr_auc": "base_pr_auc", "recall": "base_recall"})
    joined = results.merge(base_rows, on=["task", "stage", "heldout_video"], how="left")
    joined["delta_auc"] = joined["auc"] - joined["base_auc"]
    joined["delta_pr_auc"] = joined["pr_auc"] - joined["base_pr_auc"]
    joined["delta_recall"] = joined["recall"] - joined["base_recall"]
    summary = joined.groupby(["task", "feature_set"], as_index=False).agg(
        stage=("stage", "first"),
        folds=("heldout_video", "nunique"),
        auc_mean=("auc", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
        recall_mean=("recall", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_pr_auc_mean=("delta_pr_auc", "mean"),
        delta_recall_mean=("delta_recall", "mean"),
        improved_auc_folds=("delta_auc", lambda x: int((x > 0).sum())),
        improved_pr_auc_folds=("delta_pr_auc", lambda x: int((x > 0).sum())),
    )
    return summary


def markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.5f}"
        return str(value)

    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(output: Path, manifest: pd.DataFrame, results: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# ADR-004 P05：物理信息更适合注入哪个特征层？",
        "",
        "## 一、先说结论",
        "",
    ]
    best_by_task = {}
    for task in summary["task"].unique():
        sub = summary[(summary["task"] == task) & (~summary["feature_set"].str.endswith(("_rgb", "_ir", "_combined"), na=False))]
        if not sub.empty:
            continue
        sub = summary[summary["task"] == task]
        if sub.empty:
            continue
        best = sub.loc[sub["delta_auc_mean"].idxmax()]
        best_by_task[task] = best
    if best_by_task:
        lines.append("P05 逐层比较结果：")
        lines.append("")
        for task, best in best_by_task.items():
            lines.append(f"- **{task}**：最强为 `{best['feature_set']}`，AUC {best['auc_mean']:.5f}，ΔAUC {best['delta_auc_mean']:+.5f}，{best['improved_auc_folds']}/{best['folds']} 视频改善。")
        lines.append("")
    lines += [
        "P05 回答的是：在 B2 的 P2/P3/P4/P5 中，哪一层更适合追加物理信息。",
        "",
        "## 二、实验设置",
        "",
        f"- 样本：{len(manifest)} 个候选框",
        f"- 视频：{manifest['video'].nunique()} 个",
        "- 冻结 B2，逐层抽取 ROI avg+max pooling 特征",
        "- 在每个单层特征上分别追加 RGB / IR / Combined 物理信息",
        "- 判断器：ridge logistic regression",
        "- 划分：leave-one-video-out",
        "",
        "## 三、主结果",
        "",
        "### 3.1 按任务汇总",
        "",
        markdown_table(summary),
        "",
        "### 3.2 逐视频结果",
        "",
        markdown_table(results),
        "",
        "## 四、对后续实验的影响",
        "",
        "1. 如果某层 `Base_L + X` 在主要任务上 8/8 视频改善，说明该信息适合注入该层。",
        "2. 若相邻多层都有效但组合无进一步提升，则优先选单层，不增加结构复杂度。",
        "3. P05 只定位层级，仍不直接批准 C 系列；下一步 C 系列需设计具体注入模块并训练完整 YOLO。",
        "",
        "## 五、输出文件",
        "",
        "| 文件 | 内容 |",
        "| --- | --- |",
        "| `p05_manifest.csv` | P04 manifest 副本 |",
        "| `b2_roi_features_per_layer.npz` | 逐层 ROI 特征 |",
        "| `p05_logo_results.csv` | 逐视频结果 |",
        "| `p05_summary.csv` | 汇总结果 |",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.p04_dir / "p04_manifest.csv")
    manifest.to_csv(args.output / "p05_manifest.csv", index=False, float_format="%.8g")
    per_layer = extract_per_layer_features(args, manifest)
    physical = manifest[PHYSICAL_COLUMNS].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    groups = build_design_matrices(per_layer, physical)
    frames = []
    for stage, matrices in groups.items():
        for feature_set, matrix in matrices.items():
            frames.append(run_logo(manifest, matrix, feature_set, stage))
    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results)
    results.to_csv(args.output / "p05_logo_results.csv", index=False, float_format="%.8g")
    summary.to_csv(args.output / "p05_summary.csv", index=False, float_format="%.8g")
    settings = {"p04_dir": str(args.p04_dir), "weights": str(args.weights), "seed": args.seed,
                "sample_count": len(manifest), "feature_dims": {k: v.shape[1] for k, v in per_layer.items()}}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    write_report(args.output, manifest, results, summary)
    print(f"P05 outputs saved to {args.output}")


if __name__ == "__main__":
    main()
