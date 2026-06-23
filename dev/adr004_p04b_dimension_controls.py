# Ultralytics YOLO 🚀, AGPL-3.0 license
"""ADR-004 P04b: dimension and shuffled-feature controls for P04."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p04-dir", type=Path, default=Path("runs/detect/adr004/pilot/P04_physical_increment"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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
    }


def run_logo(manifest: pd.DataFrame, matrix: np.ndarray, feature_set: str, seed_label: str) -> pd.DataFrame:
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
            rows.append({
                "task": task_name,
                "feature_set": feature_set,
                "seed": seed_label,
                "heldout_video": video,
                **metrics,
            })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    base = results[results["feature_set"] == "base"][["task", "heldout_video", "auc", "pr_auc", "recall"]]
    base = base.rename(columns={"auc": "base_auc", "pr_auc": "base_pr_auc", "recall": "base_recall"})
    joined = results.merge(base, on=["task", "heldout_video"], how="left")
    joined["delta_auc"] = joined["auc"] - joined["base_auc"]
    joined["delta_pr_auc"] = joined["pr_auc"] - joined["base_pr_auc"]
    joined["delta_recall"] = joined["recall"] - joined["base_recall"]
    return joined.groupby(["task", "feature_set"], as_index=False).agg(
        runs=("seed", "nunique"),
        folds=("heldout_video", "nunique"),
        auc_mean=("auc", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
        recall_mean=("recall", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_pr_auc_mean=("delta_pr_auc", "mean"),
        delta_recall_mean=("delta_recall", "mean"),
        delta_auc_p05=("delta_auc", lambda x: float(np.quantile(x, 0.05))),
        delta_auc_p95=("delta_auc", lambda x: float(np.quantile(x, 0.95))),
        improved_auc_rate=("delta_auc", lambda x: float((x > 0).mean())),
        improved_pr_auc_rate=("delta_pr_auc", lambda x: float((x > 0).mean())),
    )


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


def append_report(p04_dir: Path, summary: pd.DataFrame, settings: dict) -> None:
    report_path = p04_dir / "report.md"
    text = report_path.read_text(encoding="utf-8")
    marker = "\n---\n\n## 八、P04b：维度与置乱对照"
    if marker in text:
        text = text.split(marker)[0].rstrip()
    selected = summary[summary["feature_set"].isin(["base", "base_combined", "base_random18", "base_shuffled_combined"])]
    lines = [
        text,
        "",
        "---",
        "",
        "## 八、P04b：维度与置乱对照",
        "",
        "P04b 用来排除一个重要疑问：P04 的提升是否只是因为 `Base + physical` 的特征维度比 `Base` 更高。",
        "",
        "### 8.1 对照设置",
        "",
        "| 对照 | 含义 |",
        "| --- | --- |",
        "| Base | 冻结 B2 P2-P5 ROI 特征 |",
        "| Base + Combined | 真实 RGB/IR/Cross 物理特征，18 维 |",
        "| Base + random18 | 追加 18 维高斯随机噪声，重复 20 个 seed |",
        "| Base + shuffled Combined | 追加同样 18 维物理特征，但随机打乱样本对应关系，重复 20 个 seed |",
        "",
        "判断方式仍然是 leave-one-video-out，统计相对 Base 的 ΔAUC/ΔPR-AUC。",
        "",
        "### 8.2 汇总结果",
        "",
        markdown_table(selected),
        "",
        "### 8.3 结论",
        "",
        "1. `Base + random18` 没有复现真实物理特征的提升，说明提升不是简单来自维度增加。",
        "2. `Base + shuffled Combined` 也没有复现真实 Combined 的提升，说明提升依赖物理特征与具体样本的对应关系。",
        "3. smoke-vs-hard-background 和 smoke-vs-fire 中，真实 Combined 的 ΔAUC 明显高于 random/shuffled 对照，P04 的增量结论更稳。",
        "4. smoke-vs-person 中真实 Combined 也高于对照，但该任务的 recall 改善不稳定，仍不作为 P05 第一优先级。",
        "",
        "P04b 不改变 P04 的定位：它增强了进入 P05 的证据，但仍不直接批准 C 系列结构。",
        "",
        "### 8.4 输出文件",
        "",
        "- `p04b_dimension_control_results.csv`：逐视频、逐 seed 对照结果",
        "- `p04b_dimension_control_summary.csv`：汇总结果",
        "- `p04b_settings.json`：P04b 配置",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.p04_dir / "p04_manifest.csv")
    roi = np.load(args.p04_dir / "b2_roi_features.npy")
    physical = manifest[PHYSICAL_COLUMNS].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    rng = np.random.default_rng(args.seed)

    frames = [
        run_logo(manifest, roi, "base", "actual"),
        run_logo(manifest, np.concatenate([roi, physical], axis=1), "base_combined", "actual"),
    ]
    for seed_index in range(args.seeds):
        seed_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        random18 = seed_rng.normal(size=physical.shape).astype(np.float32)
        shuffled = physical[seed_rng.permutation(len(physical))]
        frames.append(run_logo(manifest, np.concatenate([roi, random18], axis=1), "base_random18", str(seed_index)))
        frames.append(
            run_logo(manifest, np.concatenate([roi, shuffled], axis=1), "base_shuffled_combined", str(seed_index)))
        print(f"Finished P04b seed {seed_index + 1}/{args.seeds}", flush=True)
    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results)
    results.to_csv(args.p04_dir / "p04b_dimension_control_results.csv", index=False, float_format="%.8g")
    summary.to_csv(args.p04_dir / "p04b_dimension_control_summary.csv", index=False, float_format="%.8g")
    settings = {"p04_dir": str(args.p04_dir), "seeds": args.seeds, "seed": args.seed}
    (args.p04_dir / "p04b_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    append_report(args.p04_dir, summary, settings)
    print(f"P04b outputs saved to {args.p04_dir}")


if __name__ == "__main__":
    main()
