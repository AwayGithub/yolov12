# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Probe whether P4 head horizontal bands survive on a constant input.

If the bands are from A2C2f area attention, they should persist on
zero/constant input (independent of image content). If they are DFL
semantics, they should vanish (no targets → uniform/near-zero DFL).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--const", type=float, default=0.0,
                    help="constant fill value for input (0.0=zero, 0.5=mid-gray)")
    ap.add_argument("--imgsz", nargs=2, type=int, default=[480, 640],
                    help="H W")
    ap.add_argument("--channels", type=int, default=6,
                    help="input channels (6 for dual, 3 for rgb/ir only)")
    ap.add_argument("--out", default="tools/vis_probe", help="output dir")
    ap.add_argument("--channels-show", nargs="+", type=int,
                    default=[24, 25, 26, 27, 28, 29, 30, 31],
                    help="DFL channels to visualize")
    return ap.parse_args()


def _find_detect(model) -> Detect:
    for m in model.modules():
        if isinstance(m, Detect):
            return m
    raise RuntimeError("Detect module not found")


def _plot_grid(tensor, title, path, cmap="viridis"):
    # tensor: (C, H, W)
    C = tensor.shape[0]
    ncol = 4
    nrow = (C + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_2d(axes).reshape(nrow, ncol)
    for k in range(nrow * ncol):
        ax = axes[k // ncol, k % ncol]
        if k < C:
            im = ax.imshow(tensor[k].cpu().numpy(), cmap=cmap)
            ax.set_title(f"ch{k}")
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    yolo = YOLO(args.ckpt)
    model = yolo.model.to(device).eval()

    # Locate Detect and register a pre-forward hook to capture per-scale inputs
    detect = _find_detect(model)
    captured = []

    def pre_hook(_mod, inputs):
        feats = inputs[0]
        captured.append([f.detach().clone() for f in feats])

    h = detect.register_forward_pre_hook(pre_hook)

    # Build constant input
    H, W = args.imgsz
    x = torch.full((1, args.channels, H, W), args.const,
                   dtype=torch.float32, device=device)

    with torch.no_grad():
        _ = model(x)

    h.remove()
    if not captured:
        raise RuntimeError("Detect forward did not fire")

    feats = captured[0]  # list of per-scale feature tensors (pre-head)
    print(f"Detected {len(feats)} scales:")
    for i, f in enumerate(feats):
        print(f"  scale{i}: shape={tuple(f.shape)}")

    # Compute per-scale head output (B, 67, H, W) by running Detect's cv2/cv3
    with torch.no_grad():
        head_outs = []
        for i, f in enumerate(feats):
            box = detect.cv2[i](f)  # (B, reg_max*4, H, W)
            cls = detect.cv3[i](f)  # (B, nc, H, W)
            out = torch.cat((box, cls), dim=1)
            head_outs.append(out)

    # Scale naming by stride
    strides = detect.stride.tolist()
    names = []
    for s in strides:
        if s == 4: names.append("P2")
        elif s == 8: names.append("P3")
        elif s == 16: names.append("P4")
        elif s == 32: names.append("P5")
        else: names.append(f"S{int(s)}")

    tag = f"const{args.const:g}"
    for name, out in zip(names, head_outs):
        C = out.shape[1]
        nc = detect.nc
        reg_max = (C - nc) // 4
        t = out[0]  # (C, H, W)

        # (a) DFL channels requested (default 24-31 = top-side bins 8-15)
        ch_ids = [c for c in args.channels_show if 0 <= c < C]
        stack = torch.stack([t[c] for c in ch_ids], dim=0)
        _plot_grid(
            stack,
            f"{name} head | input={tag} | DFL ch{ch_ids[0]}..{ch_ids[-1]}",
            out_dir / f"{tag}_{name}_dfl_{ch_ids[0]}-{ch_ids[-1]}.png",
        )

        # (b) Class logits
        cls_ids = list(range(4 * reg_max, C))
        cls_stack = torch.stack([t[c] for c in cls_ids], dim=0)
        _plot_grid(
            cls_stack,
            f"{name} head | input={tag} | cls logits ch{cls_ids[0]}..{cls_ids[-1]}",
            out_dir / f"{tag}_{name}_cls.png",
        )

        # (c) stats per channel — row-wise std; a pure-horizontal-band signal
        # has very low std within each row and high std across rows.
        row_mean = t.mean(dim=2)        # (C, H)
        col_mean = t.mean(dim=1)        # (C, W)
        row_std = row_mean.std(dim=1)   # per-channel variation across rows
        col_std = col_mean.std(dim=1)   # per-channel variation across cols
        print(f"{name}: row-std/col-std ratio (first 8 DFL ch):")
        for c in ch_ids[:8]:
            r, cc = row_std[c].item(), col_std[c].item()
            ratio = r / max(cc, 1e-9)
            print(f"  ch{c:>2}: row_std={r:.4f}  col_std={cc:.4f}  ratio={ratio:.2f}")

    print(f"\nSaved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
