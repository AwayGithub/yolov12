"""Plot neck-stage smoke feature progression from a saved trace summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path("/tmp/neck_layer_trace_summary.json")
OUT_DIR = Path("/data/xwh/code/yolov12/runs/detect/adr003/smoke_delta")
OUT_PNG = OUT_DIR / "neck_trace_local_contrast.png"
OUT_PDF = OUT_DIR / "neck_trace_local_contrast.pdf"
OUT_JSON = OUT_DIR / "neck_trace_local_contrast_stats.json"
CN_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

STAGES = ["topdown_p4", "topdown_p3", "topdown_p2", "bottomup_p3", "bottomup_p4", "bottomup_p5"]
LABELS = ["TD P4", "TD P3", "TD P2", "BU P3", "BU P4", "BU P5"]


def _median_q1_q3(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }


def main() -> None:
    data = json.loads(SUMMARY_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    font_manager.fontManager.addfont(CN_FONT)
    font_prop = font_manager.FontProperties(fname=CN_FONT)
    font_name = font_prop.get_name()
    plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    stats: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    for group in ["lost", "control"]:
        stats[group] = {}
        for model in ["a", "b"]:
            stats[group][model] = {}
            for metric in ["local_contrast", "roi_peak", "roi_energy"]:
                stats[group][model][metric] = {}
                for stage in STAGES:
                    vals = [rec[model][stage][metric] for rec in data[group]]
                    stats[group][model][metric][stage] = _median_q1_q3(vals)

    OUT_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    colors = {"a": "#1f77b4", "b": "#d62728"}
    linestyles = {"lost": "-", "control": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, group, title in zip(axes, ["lost", "control"], ["A1 检出 / B2 漏检", "A1 与 B2 都检出"]):
        xs = np.arange(len(STAGES))
        for model in ["a", "b"]:
            med = [stats[group][model]["local_contrast"][s]["median"] for s in STAGES]
            q1 = [stats[group][model]["local_contrast"][s]["q1"] for s in STAGES]
            q3 = [stats[group][model]["local_contrast"][s]["q3"] for s in STAGES]
            ax.plot(
                xs,
                med,
                marker="o",
                linewidth=2.2,
                color=colors[model],
                linestyle=linestyles[group],
                label="A1" if model == "a" else "B2",
            )
            ax.fill_between(xs, q1, q3, color=colors[model], alpha=0.12)

        for model in ["a", "b"]:
            for rec in data[group][:40]:
                ys = [rec[model][s]["local_contrast"] for s in STAGES]
                ax.plot(xs, ys, color=colors[model], alpha=0.05, linewidth=0.8)

        ax.set_title(title, fontproperties=font_prop)
        ax.set_xticks(xs)
        ax.set_xticklabels(LABELS)
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Neck stage", fontproperties=font_prop)
        ax.set_ylabel("GT box local contrast", fontproperties=font_prop)

    axes[0].legend(loc="best", frameon=False, prop=font_prop)
    fig.suptitle("烟雾特征在 neck 各阶段的变化", y=1.02, fontsize=14, fontproperties=font_prop)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
