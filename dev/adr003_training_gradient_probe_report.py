# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KEY_METRICS = (
    "cos/param/p4_total/smoke_fire",
    "cos/param/p4_total/smoke_person",
    "cos/param/shared_neck/smoke_fire",
    "cos/param/shared_neck/smoke_person",
    "cos/param/detection_head/smoke_fire",
    "cos/param/detection_head/smoke_person",
    "assign/smoke/positive_count",
    "assign/smoke/score_mean",
)


def read_csv(path: Path) -> list[dict]:
    """Read a CSV if it exists."""
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows with a union header."""
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_experiment(name: str, run_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load probe summaries and validation metrics for one experiment."""
    summaries = read_csv(run_dir / "gradient_probe" / "epoch_summary.csv")
    for row in summaries:
        row["experiment"] = name
    results = read_csv(run_dir / "results.csv")
    for row in results:
        row["experiment"] = name
    return summaries, results


def metric_series(rows: list[dict], experiment: str, loss_view: str, metric: str, value_key: str) -> tuple[list[int], list[float]]:
    """Return an epoch-sorted probe metric series."""
    points = [
        (int(row["epoch"]), float(row[value_key]))
        for row in rows
        if row["experiment"] == experiment
        and row.get("loss_view") == loss_view
        and row["metric"] == metric
        and row.get(value_key) not in ("", None)
    ]
    points.sort()
    return [x for x, _ in points], [y for _, y in points]


def validation_series(rows: list[dict], experiment: str, key: str) -> tuple[list[int], list[float]]:
    """Return an epoch-sorted validation metric series."""
    points = [
        (int(float(row["epoch"])), float(row[key]))
        for row in rows
        if row["experiment"] == experiment and row.get(key) not in ("", None)
    ]
    points.sort()
    return [x for x, _ in points], [y for _, y in points]


def correlation_with_recall(summaries: list[dict], results: list[dict], experiment: str, metric: str) -> float | None:
    """Return correlation between probe conflict rate and smoke Recall at matching epochs."""
    recall = {
        int(float(row["epoch"])): float(row["metrics/smoke/recall(B)"])
        for row in results
        if row["experiment"] == experiment and row.get("metrics/smoke/recall(B)") not in ("", None)
    }
    pairs = [
        (float(row["conflict_rate"]), recall[int(row["epoch"])])
        for row in summaries
        if row["experiment"] == experiment
        and row.get("loss_view") == "cls"
        and row["metric"] == metric
        and row.get("conflict_rate") not in ("", None)
        and int(row["epoch"]) in recall
    ]
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs).T
    return float(np.corrcoef(x, y)[0, 1])


def plot_comparison(summaries: list[dict], results: list[dict], output_dir: Path) -> None:
    """Plot smoke Recall and key shared-neck conflict trajectories."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for experiment in ("A1", "B2"):
        epochs, values = validation_series(results, experiment, "metrics/smoke/recall(B)")
        axes[0].plot(epochs, values, marker="o", markersize=3, label=experiment)
        for pair, linestyle in (("smoke_fire", "-"), ("smoke_person", "--")):
            epochs, values = metric_series(
                summaries, experiment, "cls", f"cos/param/shared_neck/{pair}", "conflict_rate"
            )
            axes[1].plot(epochs, values, linestyle=linestyle, marker="o", markersize=3, label=f"{experiment} {pair}")
    axes[0].set_ylabel("smoke Recall")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("shared-neck conflict rate")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "smoke_recall_vs_shared_neck_conflict.png", dpi=200)
    plt.close(fig)


def build_report(summaries: list[dict], results: list[dict]) -> str:
    """Build a concise, rerunnable Markdown report."""
    lines = ["# ADR-003 A1/B2 训练全过程梯度竞争诊断", ""]
    for experiment in ("A1", "B2"):
        probe_epochs = sorted({int(row["epoch"]) for row in summaries if row["experiment"] == experiment})
        val_epochs = sorted({int(float(row["epoch"])) for row in results if row["experiment"] == experiment})
        lines.append(f"- {experiment}: probe epochs `{probe_epochs}`; validation latest `{max(val_epochs) if val_epochs else 'N/A'}`")
    lines.extend(["", "## 关键相关性", "", "| Experiment | Metric | corr(conflict rate, smoke Recall) |", "| -- | -- | --: |"])
    for experiment in ("A1", "B2"):
        for pair in ("smoke_fire", "smoke_person"):
            metric = f"cos/param/shared_neck/{pair}"
            corr = correlation_with_recall(summaries, results, experiment, metric)
            lines.append(f"| {experiment} | {pair} | {corr:.4f} |" if corr is not None else f"| {experiment} | {pair} | N/A |")
    lines.extend(
        [
            "",
            "## 判读规则",
            "",
            "- 先判断 P4 冲突是否早于 smoke Recall 下降，再判断冲突是否在 shared neck/head 被放大。",
            "- 对比 `cls` 与 `det` 视角，区分分类置信度竞争和定位竞争。",
            "- 若 assignment 数量与 score 稳定，而 Recall 下降，则排除 TAL 分配收缩作为主要原因。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-dir", default="runs/detect/train_A1_gradient_probe_seed0_v4")
    parser.add_argument("--b2-dir", default="runs/detect/train_B2_gradient_probe_seed0_v4")
    parser.add_argument("--output-dir", default="runs/detect/adr003/training_gradient_probe_report")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, results = [], []
    for name, run_dir in (("A1", Path(args.a1_dir)), ("B2", Path(args.b2_dir))):
        experiment_summaries, experiment_results = load_experiment(name, run_dir)
        summaries.extend(experiment_summaries)
        results.extend(experiment_results)
    key_rows = [row for row in summaries if row["metric"] in KEY_METRICS]
    write_csv(output_dir / "key_probe_metrics.csv", key_rows)
    plot_comparison(summaries, results, output_dir)
    (output_dir / "report.md").write_text(build_report(summaries, results))


if __name__ == "__main__":
    main()
