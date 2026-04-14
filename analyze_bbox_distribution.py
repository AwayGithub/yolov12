"""
Bounding box width/height distribution analysis for RGBT-3M dataset.

Analyzes per-class bbox size statistics and recommends appropriate
downsampling stride for small target detection.

Usage:
    python analyze_bbox_distribution.py
"""

import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_ROOT = Path("E:/Yan-Unifiles/lab/exp/yolov12/RGBT-3M")
LABEL_DIRS = [DATASET_ROOT / "labels" / "train", DATASET_ROOT / "labels" / "val"]
CLASS_NAMES = ["smoke", "fire", "person"]
IMG_W, IMG_H = 640, 480          # pixel dimensions
OUT_DIR = Path("./bbox_analysis")  # where to save figures

# Small target thresholds (pixels, per side)
SMALL_THR = 32      # < 32px → small target (COCO definition)
MEDIUM_THR = 96     # 32–96px → medium target

# Typical YOLO feature-map strides to evaluate
STRIDES = [4, 8, 16, 32]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_labels(label_dirs):
    """Return dict {class_id: list of (w_px, h_px)} for all labels."""
    data = defaultdict(list)
    n_files = 0
    for label_dir in label_dirs:
        for txt in Path(label_dir).glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    data[cls].append((w_norm * IMG_W, h_norm * IMG_H))
            n_files += 1
    print(f"Loaded {n_files} label files.")
    return data


def bbox_stats(wh_list):
    """Return dict of statistics for a list of (w, h) tuples."""
    if not wh_list:
        return {}
    arr = np.array(wh_list)
    w, h = arr[:, 0], arr[:, 1]
    area = w * h
    shorter = np.minimum(w, h)     # shorter side drives feature-map resolution
    longer  = np.maximum(w, h)
    diag    = np.sqrt(w**2 + h**2)
    return {
        "count": len(arr),
        "w":  {"mean": w.mean(),  "median": np.median(w),  "p10": np.percentile(w, 10),  "p90": np.percentile(w, 90)},
        "h":  {"mean": h.mean(),  "median": np.median(h),  "p10": np.percentile(h, 10),  "p90": np.percentile(h, 90)},
        "area": {"mean": area.mean(), "median": np.median(area)},
        "shorter": shorter,
        "longer":  longer,
        "diag":    diag,
        "small_frac":  (shorter < SMALL_THR).mean(),
        "medium_frac": ((shorter >= SMALL_THR) & (shorter < MEDIUM_THR)).mean(),
        "large_frac":  (shorter >= MEDIUM_THR).mean(),
    }


def stride_analysis(shorter_sides, stride):
    """
    For a given downsampling stride, check how many targets collapse to
    < 2 feature-map cells on their shorter side (essentially invisible).
    Feature-map cell size = stride px.
    < 2 cells → very hard to detect.
    """
    fm_cells = shorter_sides / stride
    too_small_frac = (fm_cells < 2).mean()
    small_frac     = ((fm_cells >= 2) & (fm_cells < 4)).mean()
    ok_frac        = (fm_cells >= 4).mean()
    return {"<2 cells": too_small_frac, "2-4 cells": small_frac, ">=4 cells": ok_frac}


# ── Plot helpers ──────────────────────────────────────────────────────────────

COLOR_MAP = {"smoke": "#E07B54", "fire": "#C0392B", "person": "#2980B9"}
ALL_COLOR  = "#7F8C8D"

def plot_2d_scatter(data, out_path):
    """Scatter plot of w vs h for each class + all combined."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    all_wh = []

    for ax, (cls_id, cls_name) in zip(axes[:3], enumerate(CLASS_NAMES)):
        wh = np.array(data.get(cls_id, []))
        all_wh.extend(data.get(cls_id, []))
        if len(wh) == 0:
            ax.set_title(f"{cls_name} (no data)")
            continue
        ax.scatter(wh[:, 0], wh[:, 1], s=2, alpha=0.3, color=COLOR_MAP[cls_name])
        ax.axvline(SMALL_THR, color="red", lw=1, ls="--", label=f"{SMALL_THR}px")
        ax.axhline(SMALL_THR, color="red", lw=1, ls="--")
        ax.axvline(MEDIUM_THR, color="orange", lw=1, ls="--", label=f"{MEDIUM_THR}px")
        ax.axhline(MEDIUM_THR, color="orange", lw=1, ls="--")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title(f"{cls_name}  (n={len(wh):,})")
        ax.set_xlim(0, IMG_W); ax.set_ylim(0, IMG_H)
        ax.legend(fontsize=7)

    ax_all = axes[3]
    all_arr = np.array(all_wh) if all_wh else np.empty((0, 2))
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        wh = np.array(data.get(cls_id, []))
        if len(wh):
            ax_all.scatter(wh[:, 0], wh[:, 1], s=2, alpha=0.25,
                           color=COLOR_MAP[cls_name], label=cls_name)
    ax_all.axvline(SMALL_THR,  color="red",    lw=1, ls="--")
    ax_all.axhline(SMALL_THR,  color="red",    lw=1, ls="--")
    ax_all.axvline(MEDIUM_THR, color="orange", lw=1, ls="--")
    ax_all.axhline(MEDIUM_THR, color="orange", lw=1, ls="--")
    ax_all.set_xlabel("Width (px)"); ax_all.set_ylabel("Height (px)")
    ax_all.set_title(f"All classes (n={len(all_arr):,})")
    ax_all.set_xlim(0, IMG_W); ax_all.set_ylim(0, IMG_H)
    ax_all.legend(fontsize=7, markerscale=4)

    fig.suptitle("Bounding Box Width vs Height — RGBT-3M", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_histograms(data, out_path):
    """Per-class histograms of w and h."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        wh = np.array(data.get(cls_id, []))
        ax_w, ax_h = axes[cls_id]
        color = COLOR_MAP[cls_name]

        if len(wh) == 0:
            ax_w.set_title(f"{cls_name} width — no data")
            ax_h.set_title(f"{cls_name} height — no data")
            continue

        w, h = wh[:, 0], wh[:, 1]

        for ax, vals, label in [(ax_w, w, "Width (px)"), (ax_h, h, "Height (px)")]:
            ax.hist(vals, bins=60, color=color, alpha=0.7, edgecolor="white", lw=0.3)
            ax.axvline(SMALL_THR,  color="red",    lw=1.5, ls="--", label=f"small={SMALL_THR}px")
            ax.axvline(MEDIUM_THR, color="orange", lw=1.5, ls="--", label=f"medium={MEDIUM_THR}px")
            ax.axvline(np.median(vals), color="black", lw=1.5, ls="-", label=f"median={np.median(vals):.1f}px")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(f"{cls_name} — {label}  (n={len(vals):,})")
            ax.legend(fontsize=8)

    fig.suptitle("Bounding Box Size Histograms — RGBT-3M", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_shorter_side_cdf(data, out_path):
    """CDF of shorter-side length with stride reference lines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        wh = np.array(data.get(cls_id, []))
        if len(wh) == 0:
            continue
        shorter = np.minimum(wh[:, 0], wh[:, 1])
        xs = np.sort(shorter)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=COLOR_MAP[cls_name], lw=2, label=cls_name)

    # stride × 2 reference: minimum recommended feature-map resolution
    for stride in STRIDES:
        min_px = stride * 2
        ax.axvline(min_px, lw=1, ls=":", alpha=0.6,
                   label=f"stride={stride} (2-cell min={min_px}px)")

    ax.set_xlabel("Shorter side (px)")
    ax.set_ylabel("CDF")
    ax.set_title("Shorter-Side CDF — coverage at each stride's 2-cell threshold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 200)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_stride_heatmap(data, out_path):
    """Heatmap: per-class × stride → fraction of targets with <2 feature-map cells."""
    all_classes = list(enumerate(CLASS_NAMES)) + [(-1, "all")]
    fracs = np.zeros((len(all_classes), len(STRIDES)))

    all_shorter = []
    for cls_id, cls_name in all_classes:
        if cls_id == -1:
            shorter = np.array(all_shorter)
        else:
            wh = np.array(data.get(cls_id, []))
            shorter = np.minimum(wh[:, 0], wh[:, 1]) if len(wh) else np.array([])
            all_shorter.extend(shorter.tolist())

        for j, stride in enumerate(STRIDES):
            if len(shorter) == 0:
                fracs[cls_id if cls_id != -1 else len(CLASS_NAMES), j] = 0
            else:
                row = len(CLASS_NAMES) if cls_id == -1 else cls_id
                fracs[row, j] = (shorter / stride < 2).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(fracs, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(STRIDES)))
    ax.set_xticklabels([f"stride={s}" for s in STRIDES])
    ax.set_yticks(range(len(all_classes)))
    ax.set_yticklabels([n for _, n in all_classes])
    for i in range(fracs.shape[0]):
        for j in range(fracs.shape[1]):
            ax.text(j, i, f"{fracs[i,j]:.1%}", ha="center", va="center",
                    color="white" if fracs[i, j] > 0.5 else "black", fontsize=10)
    plt.colorbar(im, ax=ax, label="Fraction with <2 FM cells (hard to detect)")
    ax.set_title("Fraction of targets smaller than 2 feature-map cells\n(per class × stride)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RGBT-3M Bounding Box Distribution Analysis")
    print("=" * 60)

    data = load_labels(LABEL_DIRS)

    # ── Per-class statistics ──────────────────────────────────────────────────
    all_wh = []
    print()
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        wh_list = data.get(cls_id, [])
        all_wh.extend(wh_list)
        s = bbox_stats(wh_list)
        if not s:
            print(f"[{cls_name}] No data.")
            continue
        print(f"[{cls_name}] n={s['count']:,}")
        print(f"  Width  : mean={s['w']['mean']:.1f}  median={s['w']['median']:.1f}  p10={s['w']['p10']:.1f}  p90={s['w']['p90']:.1f}  (px)")
        print(f"  Height : mean={s['h']['mean']:.1f}  median={s['h']['median']:.1f}  p10={s['h']['p10']:.1f}  p90={s['h']['p90']:.1f}  (px)")
        print(f"  Area   : mean={s['area']['mean']:.0f}  median={s['area']['median']:.0f}  (px²)")
        print(f"  Size dist (by shorter side):  small(<{SMALL_THR}px)={s['small_frac']:.1%}  "
              f"medium({SMALL_THR}-{MEDIUM_THR}px)={s['medium_frac']:.1%}  large(>{MEDIUM_THR}px)={s['large_frac']:.1%}")

    # All classes combined
    s_all = bbox_stats(all_wh)
    print()
    print(f"[ALL] n={s_all['count']:,}")
    print(f"  Width  : mean={s_all['w']['mean']:.1f}  median={s_all['w']['median']:.1f}")
    print(f"  Height : mean={s_all['h']['mean']:.1f}  median={s_all['h']['median']:.1f}")
    print(f"  Size dist:  small={s_all['small_frac']:.1%}  medium={s_all['medium_frac']:.1%}  large={s_all['large_frac']:.1%}")

    # ── Stride suitability analysis ───────────────────────────────────────────
    print()
    print("=" * 60)
    print("Stride Suitability (fraction of targets with < 2 FM cells)")
    print("(< 2 cells: anchor-based detection becomes very unreliable)")
    print("=" * 60)
    header = f"{'Class':<10}" + "".join(f"  stride={s:<4}" for s in STRIDES)
    print(header)
    print("-" * len(header))

    collect_shorter = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        wh = np.array(data.get(cls_id, []))
        if len(wh) == 0:
            collect_shorter[cls_name] = np.array([])
            continue
        shorter = np.minimum(wh[:, 0], wh[:, 1])
        collect_shorter[cls_name] = shorter
        row = f"{cls_name:<10}"
        for stride in STRIDES:
            frac = (shorter / stride < 2).mean()
            row += f"  {frac:>8.1%}    "
        print(row)

    all_shorter_arr = np.array(all_wh)
    if len(all_shorter_arr):
        all_shorter_arr = np.minimum(all_shorter_arr[:, 0], all_shorter_arr[:, 1])
    row = f"{'all':<10}"
    for stride in STRIDES:
        frac = (all_shorter_arr / stride < 2).mean() if len(all_shorter_arr) else 0
        row += f"  {frac:>8.1%}    "
    print(row)

    # ── Recommendation ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Small-Target Analysis & Recommendation")
    print("=" * 60)
    if len(all_shorter_arr):
        p10_all = np.percentile(all_shorter_arr, 10)
        p25_all = np.percentile(all_shorter_arr, 25)
        p50_all = np.percentile(all_shorter_arr, 50)
        print(f"Shorter-side percentiles (all classes):")
        print(f"  p10={p10_all:.1f}px  p25={p25_all:.1f}px  p50={p50_all:.1f}px")
        print()

        # Recommend smallest stride where < 5% of objects vanish (<2 cells)
        for stride in STRIDES:
            frac_lost = (all_shorter_arr / stride < 2).mean()
            if frac_lost < 0.05:
                print(f"Recommended minimum detection stride: {stride}px")
                print(f"  At stride={stride}: {frac_lost:.1%} of objects have < 2 FM cells")
                print(f"  → P3 (stride=8) head is {'sufficient' if stride <= 8 else 'borderline'} "
                      f"for the bulk of targets.")
                print(f"  → A stride-4 P2 head would cover {(all_shorter_arr / 4 >= 2).mean():.1%} of all objects with ≥ 2 cells.")
                break

        print()
        print("Per-class p10 shorter side vs. stride:")
        for cls_name in CLASS_NAMES:
            shorter = collect_shorter.get(cls_name, np.array([]))
            if len(shorter) == 0:
                continue
            p10 = np.percentile(shorter, 10)
            needed_stride = p10 / 2   # stride where 10th-pct object = 2 cells
            print(f"  {cls_name:<8}: p10={p10:.1f}px → needs stride ≤ {needed_stride:.1f}px to keep at ≥ 2 cells")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print()
    print("Generating plots...")
    plot_2d_scatter(data,           OUT_DIR / "01_scatter_w_vs_h.png")
    plot_histograms(data,           OUT_DIR / "02_histograms_wh.png")
    plot_shorter_side_cdf(data,     OUT_DIR / "03_shorter_side_cdf.png")
    plot_stride_heatmap(data,       OUT_DIR / "04_stride_heatmap.png")

    print()
    print(f"All outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
