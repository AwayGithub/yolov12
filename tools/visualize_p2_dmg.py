# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Visualize DMGFusion@P2 intermediate feature maps for modality fusion diagnosis.

Usage:
    python tools/visualize_p2_dmg.py \\
        --ckpt runs/detect/train_7a/weights/best.pt \\
        --rgb  datasets/RGBT-3M/RGB/val/frame_0042.jpg \\
        --ir   datasets/RGBT-3M/IR/val/frame_0042.jpg \\
        --out  tools/vis_p2

Outputs (in --out directory, named by image stem):
    <stem>_mean.png              — 7-panel channel-mean overview
    <stem>_ch_x_rgb_f1..f8.png  — all 64 channels of x_rgb, 8 channels per figure (2×4)
    <stem>_ch_x_ir_f1..f8.png   — all 64 channels of x_ir
    <stem>_ch_D_f1..f8.png      — all 64 channels of D = |x_rgb - x_ir|
    <stem>_ch_W_rgb.png          — W_rgb spatial weight map (single channel)
    <stem>_ch_W_ir.png           — W_ir spatial weight map (single channel)
    <stem>_ch_S_f1..f8.png      — all 64 channels of S (saliency)
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

def _hook_dmgfusion(model):
    """Register a forward hook on the first DMGFusion module found in *model*.

    Args:
        model: nn.Module (the inner model, e.g. YOLO().model).

    Returns:
        dict that will be populated with keys
        {x_rgb, x_ir, D, W, S, fused} after the next forward pass.
    """
    from ultralytics.nn.modules.block import DMGFusion

    captured = {}

    def _fwd(module, inp, out):
        x_rgb_in = inp[0].detach().cpu()
        x_ir_in  = inp[1].detach().cpu()
        D_raw    = torch.abs(x_rgb_in - x_ir_in)

        dev = next(module.parameters()).device
        with torch.no_grad():
            stacked    = torch.cat([x_rgb_in, x_ir_in, D_raw], dim=1).to(dev)
            sel_logits = module.sel(stacked).cpu()
            W          = torch.softmax(sel_logits, dim=1)
            S          = torch.sigmoid(module.diff_enc(D_raw.to(dev))).cpu()

        captured["x_rgb"] = x_rgb_in
        captured["x_ir"]  = x_ir_in
        captured["D"]     = D_raw
        captured["W"]     = W
        captured["S"]     = S
        captured["fused"] = out.detach().cpu()

    for m in model.modules():
        if isinstance(m, DMGFusion):
            m.register_forward_hook(_fwd)
            print(f"[hook] Registered on {type(m).__name__} — channels={m.out_proj.conv.in_channels}")
            return captured

    raise RuntimeError("No DMGFusion module found in model. Is the checkpoint from a DMGFusion experiment?")


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
    """7-panel channel-mean overview figure."""
    keys    = ["x_rgb", "x_ir", "D", "W_rgb", "W_ir", "S", "fused"]
    labels  = ["x_rgb (mean)", "x_ir (mean)", "D=|R-I| (mean)",
               "W_rgb", "W_ir", "S saliency (mean)", "fused (mean)"]
    cmaps   = ["viridis", "viridis", "inferno", "RdBu_r", "RdBu_r", "inferno", "viridis"]

    fig, axes = plt.subplots(1, 7, figsize=(24, 3.8))
    fig.suptitle(title, fontsize=9)

    for ax, k, lbl, cmap in zip(axes, keys, labels, cmaps):
        arr = _chan_mean(tensors[k])
        im  = ax.imshow(arr, cmap=cmap)
        ax.set_title(lbl, fontsize=7.5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [mean]  → {out_path}")


_ROWS_PER_FIG = 2
_COLS_PER_FIG = 4
_CH_PER_FIG   = _ROWS_PER_FIG * _COLS_PER_FIG  # 8 channels per figure


def _plot_per_channel(t: torch.Tensor, key: str,
                      title_prefix: str, out_dir: Path, stem: str,
                      cmap: str = "viridis") -> None:
    """Save all channels as paginated 2×4 figures (8 channels each).

    For a C=64 tensor this produces f1..f8 files.
    Single-channel tensors (W_rgb, W_ir) produce one file with no page suffix.

    Args:
        t:            Tensor (B, C, H, W).
        key:          Variable name, used in title and filename.
        title_prefix: e.g. "frame_0042 | DMGFusion@P2".
        out_dir:      Directory to write files into.
        stem:         Image filename stem, e.g. "frame_0042".
        cmap:         Matplotlib colormap name.
    """
    arr = t[0].numpy().astype(np.float32)  # (C, H, W)
    C   = arr.shape[0]

    if C == 1:
        # Single-channel (W_rgb, W_ir): one plain figure
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

    # Multi-channel: paginate into 2×4 grids
    n_figs = math.ceil(C / _CH_PER_FIG)
    for fig_idx in range(n_figs):
        ch_start = fig_idx * _CH_PER_FIG
        ch_end   = min(ch_start + _CH_PER_FIG, C)
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
            ax   = axes[r][c]
            ch   = ch_start + slot
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
    p = argparse.ArgumentParser(description="Visualize DMGFusion@P2 feature maps")
    p.add_argument("--ckpt", required=True,
                   help="Path to YOLO checkpoint (.pt), e.g. runs/detect/train_7a/weights/best.pt")
    p.add_argument("--rgb",  required=True,
                   help="Path to RGB image file")
    p.add_argument("--ir",   required=True,
                   help="Path to IR image file (must be spatially aligned with --rgb)")
    p.add_argument("--out",  default="tools/vis_p2",
                   help="Output directory (created if absent)")
    p.add_argument("--device", default="cpu",
                   help="Inference device, e.g. 'cpu', '0', 'cuda:0'. Default: cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.rgb).stem

    # ── Load model ──────────────────────────────────────────────────────────
    from ultralytics import YOLO
    print(f"[load] {args.ckpt}")
    yolo = YOLO(args.ckpt)
    inner_model = yolo.model
    inner_model.eval()

    captured = _hook_dmgfusion(inner_model)

    # ── Build 6-channel input ────────────────────────────────────────────────
    rgb_np = _load_image(args.rgb)
    ir_np  = _load_image(args.ir)

    # Resize IR to match RGB if shapes differ
    if rgb_np.shape != ir_np.shape:
        ir_np = cv2.resize(ir_np, (rgb_np.shape[1], rgb_np.shape[0]))

    rgb_t = _to_tensor(rgb_np)
    ir_t  = _to_tensor(ir_np)
    x6    = torch.cat([rgb_t, ir_t], dim=1)  # (1, 6, H, W)

    dev = torch.device(args.device if args.device != "cpu" else "cpu")
    if args.device != "cpu":
        inner_model.to(dev)

    print(f"[infer] input shape {tuple(x6.shape)}")
    with torch.no_grad():
        inner_model(x6.to(dev))

    if not captured:
        raise RuntimeError("Hook did not fire — check that the model uses DMGFusion.")

    # ── Build tensors dict (expand W from 2ch to two 1ch tensors) ────────────
    tensors = {
        "x_rgb": captured["x_rgb"],
        "x_ir":  captured["x_ir"],
        "D":     captured["D"],
        "W_rgb": captured["W"][:, 0:1],   # (B,1,H,W) — spatial weight map
        "W_ir":  captured["W"][:, 1:2],
        "S":     captured["S"],
        "fused": captured["fused"],
    }

    title_base = f"{stem} | DMGFusion@P2"

    # ── 1. Channel-mean overview ─────────────────────────────────────────────
    _plot_mean_overview(tensors, title_base, out_dir / f"{stem}_mean.png")

    # ── 2. Per-channel grids ─────────────────────────────────────────────────
    ch_cfg = [
        ("x_rgb", "viridis"),
        ("x_ir",  "viridis"),
        ("D",     "inferno"),
        ("W_rgb", "RdBu_r"),
        ("W_ir",  "RdBu_r"),
        ("S",     "inferno"),
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
