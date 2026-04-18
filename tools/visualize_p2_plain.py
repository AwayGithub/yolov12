# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Visualize plain Concat+Conv1x1 P2 fusion intermediate feature maps.

Counterpart to visualize_p2_dmg.py — for models using p2_fusion=plain
(e.g. dual_MF_plainP2_ChWP4P5_CMAP5_P2345).

Usage:
    python tools/visualize_p2_plain.py \
        --ckpt runs/detect/RGBT-3M/dual_MF_plainP2_ChWP4P5_CMAP5_P2345/best.pt \
        --rgb  datasets/RGBT-3M/RGB/val/video1_frame_01390.jpg \
        --ir   datasets/RGBT-3M/IR/val/video1_frame_01390.jpg \
        --out  tools/vis_p2_plain

Outputs (in --out/<stem> directory):
    <stem>_mean.png              — 4-panel channel-mean overview (x_rgb, x_ir, D, fused)
    <stem>_ch_x_rgb_f1..f8.png  — all 64 channels of x_rgb
    <stem>_ch_x_ir_f1..f8.png   — all 64 channels of x_ir
    <stem>_ch_D_f1..f8.png      — all 64 channels of D = |x_rgb - x_ir|
    <stem>_ch_fused_f1..f8.png  — all 64 channels of fused output
"""

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def _hook_plain_fusion(model):
    """Register a forward hook on the P2 fusion Conv (plain Concat+Conv1x1).

    For DualStreamDetectionModel with p2_fusion=plain, the P2 fusion module is
    a Conv(2C, C, 1, 1).  We hook into DualStreamDetectionModel._predict_once
    to capture the P2 inputs before concat and the fused output.

    Args:
        model: nn.Module (the inner model, e.g. YOLO().model).

    Returns:
        dict that will be populated with keys
        {x_rgb, x_ir, fused} after the next forward pass.
    """
    from ultralytics.nn.tasks import DualStreamDetectionModel

    if not isinstance(model, DualStreamDetectionModel):
        raise RuntimeError("Model is not a DualStreamDetectionModel. "
                           "Is the checkpoint from a dual-stream experiment?")

    # The P2 fusion conv is model.fusion_convs["p2"]
    if "p2" not in model.fusion_convs:
        raise RuntimeError("No P2 fusion conv found in model.fusion_convs.")
    p2_conv = model.fusion_convs["p2"]

    captured = {}

    def _fwd(_module, inp, out):
        # inp[0] is the concatenated tensor (B, 2C, H, W)
        x_cat = inp[0].detach().cpu()
        C = x_cat.shape[1] // 2
        captured["x_rgb"] = x_cat[:, :C]   # first half = rgb (see tasks.py line 571)
        captured["x_ir"] = x_cat[:, C:]    # second half = ir
        captured["fused"] = out.detach().cpu()

    p2_conv.register_forward_hook(_fwd)
    print(f"[hook] Registered on P2 fusion Conv — in_channels={p2_conv.conv.in_channels}, "
          f"out_channels={p2_conv.conv.out_channels}")
    return captured


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    """Load BGR image with OpenCV and return as RGB uint8 HWC."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(rgb_np: np.ndarray) -> torch.Tensor:
    """HWC uint8 → (1,3,H,W) float32 [0,1]."""
    return torch.from_numpy(rgb_np).permute(2, 0, 1).float()[None] / 255.0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _chan_mean(t: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) → (H,W) float32 numpy, batch index 0."""
    return t[0].mean(0).numpy().astype(np.float32)


def _plot_mean_overview(tensors: dict, title: str, out_path: Path) -> None:
    """4-panel channel-mean overview figure."""
    keys = ["x_rgb", "x_ir", "D", "fused"]
    labels = ["x_rgb (mean)", "x_ir (mean)", "D=|R-I| (mean)", "fused (mean)"]
    cmaps = ["viridis", "viridis", "inferno", "viridis"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.suptitle(title, fontsize=9)

    for ax, k, lbl, cmap in zip(axes, keys, labels, cmaps):
        arr = _chan_mean(tensors[k])
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(lbl, fontsize=7.5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [mean]  → {out_path}")


_ROWS_PER_FIG = 2
_COLS_PER_FIG = 4
_CH_PER_FIG = _ROWS_PER_FIG * _COLS_PER_FIG  # 8 channels per figure


def _plot_per_channel(t: torch.Tensor, key: str,
                      title_prefix: str, out_dir: Path, stem: str,
                      cmap: str = "viridis") -> None:
    """Save all channels as paginated 2×4 figures (8 channels each)."""
    arr = t[0].numpy().astype(np.float32)  # (C, H, W)
    C = arr.shape[0]

    if C == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.2))
        fig.suptitle(f"{title_prefix} — {key}", fontsize=8)
        im = ax.imshow(arr[0], cmap=cmap)
        ax.set_title("ch0", fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        out_path = out_dir / f"{stem}_ch_{key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ch]    → {out_path}")
        return

    n_figs = math.ceil(C / _CH_PER_FIG)
    for fig_idx in range(n_figs):
        ch_start = fig_idx * _CH_PER_FIG
        ch_end = min(ch_start + _CH_PER_FIG, C)
        n_in_fig = ch_end - ch_start

        fig, axes = plt.subplots(_ROWS_PER_FIG, _COLS_PER_FIG,
                                 figsize=(_COLS_PER_FIG * 3.2, _ROWS_PER_FIG * 2.8 + 0.6),
                                 squeeze=False)
        fig.suptitle(
            f"{title_prefix} — {key}  [ch {ch_start}–{ch_end - 1}]  "
            f"(fig {fig_idx + 1}/{n_figs})",
            fontsize=8,
        )

        for slot in range(_CH_PER_FIG):
            r, c = divmod(slot, _COLS_PER_FIG)
            ax = axes[r][c]
            ch = ch_start + slot
            if slot < n_in_fig:
                im = ax.imshow(arr[ch], cmap=cmap)
                ax.set_title(f"ch{ch}", fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.axis("off")

        fig.tight_layout()
        out_path = out_dir / f"{stem}_ch_{key}_f{fig_idx + 1}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ch]    → {out_path}  (ch {ch_start}–{ch_end - 1})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize plain Concat+Conv1x1 P2 feature maps")
    p.add_argument("--ckpt", required=True,
                   help="Path to YOLO checkpoint (.pt), e.g. runs/detect/.../best.pt")
    p.add_argument("--rgb", required=True,
                   help="Path to RGB image file")
    p.add_argument("--ir", required=True,
                   help="Path to IR image file (must be spatially aligned with --rgb)")
    p.add_argument("--out", default="tools/vis_p2_plain",
                   help="Output directory (created if absent)")
    p.add_argument("--device", default="cpu",
                   help="Inference device, e.g. 'cpu', '0', 'cuda:0'. Default: cpu")
    return p.parse_args()


def main():
    args = parse_args()
    stem = Path(args.rgb).stem
    out_dir = Path(args.out) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    from ultralytics import YOLO
    print(f"[load] {args.ckpt}")
    yolo = YOLO(args.ckpt)
    inner_model = yolo.model
    inner_model.eval()

    captured = _hook_plain_fusion(inner_model)

    # ── Build 6-channel input ────────────────────────────────────────────────
    rgb_np = _load_image(args.rgb)
    ir_np = _load_image(args.ir)

    # Resize IR to match RGB if shapes differ
    if rgb_np.shape != ir_np.shape:
        ir_np = cv2.resize(ir_np, (rgb_np.shape[1], rgb_np.shape[0]))

    rgb_t = _to_tensor(rgb_np)
    ir_t = _to_tensor(ir_np)
    # Model convention (from Format bgr=0.0 channel flip + _predict_once split):
    #   0:3 = IR (BGR order),  3:6 = RGB (BGR order)
    # _to_tensor produces RGB order, so flip each to BGR before concat.
    x6 = torch.cat([ir_t.flip(1), rgb_t.flip(1)], dim=1)  # (1, 6, H, W)

    dev = torch.device(args.device if args.device != "cpu" else "cpu")
    if args.device != "cpu":
        inner_model.to(dev)

    print(f"[infer] input shape {tuple(x6.shape)}")
    with torch.no_grad():
        inner_model(x6.to(dev))

    if not captured:
        raise RuntimeError("Hook did not fire — check that the model uses plain P2 fusion.")

    # ── Build tensors dict ──────────────────────────────────────────────────
    # Note: for plain fusion, the concat order in tasks.py line 571 is
    #   fc(torch.cat([r, i], dim=1))  where r=feats_rgb, i=feats_ir
    # So the hook captures x_rgb = first half, x_ir = second half.
    # However, we need to verify this matches the actual concat order.
    # In tasks.py: r, i = feats_rgb[stage_name], feats_ir[stage_name]
    # then fc(torch.cat([r, i], dim=1)) → first C channels = rgb, next C = ir ✓

    tensors = {
        "x_rgb": captured["x_rgb"],
        "x_ir": captured["x_ir"],
        "D": torch.abs(captured["x_rgb"] - captured["x_ir"]),
        "fused": captured["fused"],
    }

    title_base = f"{stem} | Plain Concat+Conv1x1@P2"

    # ── 1. Channel-mean overview ─────────────────────────────────────────────
    _plot_mean_overview(tensors, title_base, out_dir / f"{stem}_mean.png")

    # ── 2. Per-channel grids ─────────────────────────────────────────────────
    ch_cfg = [
        ("x_rgb", "viridis"),
        ("x_ir", "viridis"),
        ("D", "inferno"),
        ("fused", "viridis"),
    ]
    for key, cmap in ch_cfg:
        _plot_per_channel(
            tensors[key],
            key=key,
            title_prefix=title_base,
            out_dir=out_dir,
            stem=stem,
            cmap=cmap,
        )

    print(f"\n[done] All outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
