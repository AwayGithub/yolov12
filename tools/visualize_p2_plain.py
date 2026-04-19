# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Visualize dual-stream plain Concat+Conv1x1 fusion model features across all scales.

Captures:
  - Fusion inputs/outputs at every level (P2/P3/P4/P5, all plain Conv1x1)
  - Detection head inputs (post-neck features) for each scale
  - Input RGB/IR images and detection result overlays

Usage:
    python tools/visualize_p2_plain.py \\
        --ckpt runs/detect/RGBT-3M/dual_MF_plainP2_ChWP4P5_CMAP5_P2345/best.pt \\
        --rgb  RGBT-3M/RGB/val/video1_frame_01390.jpg \\
        --ir   RGBT-3M/IR/val/video1_frame_01390.jpg \\
        --out  tools/vis_plain

Output layout (under --out/<image_stem>/):
    inputs/
        rgb.png, ir.png                    raw input images
        rgb_pred.png, ir_pred.png          with detection overlays
    fusion_p2/ fusion_p3/ fusion_p4/ fusion_p5/
        overview_mean.png                  3-panel (x_rgb / x_ir / fused)
        x_rgb_ch_f*.png, x_ir_ch_f*.png, fused_ch_f*.png
    head_p2/ head_p3/ head_p4/ head_p5/
        overview_mean.png                  single-panel channel mean
        feat_ch_f*.png                     per-channel grid
"""

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Image / tensor helpers
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(rgb_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_np).permute(2, 0, 1).float()[None] / 255.0


def _chan_mean(t: torch.Tensor) -> np.ndarray:
    return t[0].mean(0).numpy().astype(np.float32)


def _save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


_BOX_COLORS = [(30, 144, 255), (50, 205, 50), (255, 99, 71),
               (255, 215, 0), (186, 85, 211), (0, 206, 209)]


def _draw_boxes(img_rgb: np.ndarray, dets: np.ndarray, names: dict) -> np.ndarray:
    img = img_rgb.copy()
    for x1, y1, x2, y2, conf, cls in dets:
        c = int(cls)
        col = _BOX_COLORS[c % len(_BOX_COLORS)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        label = f"{names.get(c, str(c))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4),
                      (int(x1) + tw + 2, int(y1)), col, -1)
        cv2.putText(img, label, (int(x1) + 1, int(y1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


_GT_COLORS = [(255, 255, 255), (180, 255, 180), (255, 180, 180),
              (180, 180, 255), (255, 255, 180), (180, 255, 255)]


def _load_gt_labels(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """Load YOLO-format label file → (N, 6) float32 [x1, y1, x2, y2, 1.0, cls]."""
    if not label_path.exists():
        print(f"  [warn] label not found: {label_path}")
        return np.zeros((0, 6), dtype=np.float32)
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
            x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2, 1.0, cls])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 6), dtype=np.float32)


def _draw_gt_boxes(img_rgb: np.ndarray, gts: np.ndarray, names: dict) -> np.ndarray:
    """Draw GT boxes with white/light border and 'GT:' label prefix."""
    img = img_rgb.copy()
    for x1, y1, x2, y2, _, cls in gts:
        c = int(cls)
        col = _GT_COLORS[c % len(_GT_COLORS)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        label = f"GT:{names.get(c, str(c))}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4),
                      (int(x1) + tw + 2, int(y1)), col, -1)
        cv2.putText(img, label, (int(x1) + 1, int(y1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def _stride_to_level(s: float) -> str:
    return {4: "p2", 8: "p3", 16: "p4", 32: "p5"}.get(int(s), f"s{int(s)}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_ROWS, _COLS = 2, 4
_CH_PER_FIG = _ROWS * _COLS


def _plot_overview(panels: list, title: str, out_path: Path) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 3.8))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=9)
    for ax, (lbl, arr, cmap) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(lbl, fontsize=7.5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [overview] → {out_path}")


def _plot_per_channel(t: torch.Tensor, key: str,
                      title_prefix: str, out_dir: Path,
                      cmap: str = "viridis") -> None:
    arr = t[0].numpy().astype(np.float32)
    C = arr.shape[0]

    if C == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.2))
        fig.suptitle(f"{title_prefix} — {key}", fontsize=8)
        im = ax.imshow(arr[0], cmap=cmap)
        ax.set_title("ch0", fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        p = out_dir / f"{key}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ch]  → {p}")
        return

    n_figs = math.ceil(C / _CH_PER_FIG)
    for fi in range(n_figs):
        cs = fi * _CH_PER_FIG
        ce = min(cs + _CH_PER_FIG, C)
        n_in = ce - cs
        fig, axes = plt.subplots(_ROWS, _COLS,
                                 figsize=(_COLS * 3.2, _ROWS * 2.8 + 0.6),
                                 squeeze=False)
        fig.suptitle(f"{title_prefix} — {key}  [ch {cs}–{ce - 1}]  (fig {fi + 1}/{n_figs})",
                     fontsize=8)
        for slot in range(_CH_PER_FIG):
            r, c = divmod(slot, _COLS)
            ax = axes[r][c]
            ch = cs + slot
            if slot < n_in:
                im = ax.imshow(arr[ch], cmap=cmap)
                ax.set_title(f"ch{ch}", fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.axis("off")
        fig.tight_layout()
        p = out_dir / f"{key}_ch_f{fi + 1}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ch]  → {p}  (ch {cs}–{ce - 1})")


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _hook_fusions(model) -> tuple:
    """Hook every plain Conv1x1 fusion in DualStreamDetectionModel.

    Returns (captured_dict, hook_handles_list).
    captured_dict: {stage: {x_rgb, x_ir, fused}}
    """
    from ultralytics.nn.tasks import DualStreamDetectionModel

    if not isinstance(model, DualStreamDetectionModel):
        raise RuntimeError("Not a DualStreamDetectionModel")

    captured: dict = {}
    handles = []

    for stage, mod in model.fusion_convs.items():
        captured[stage] = {}

        def _make_hook(s):
            def _fwd(_mod, inp, _out):
                xc = inp[0].detach().cpu()
                C = xc.shape[1] // 2
                captured[s].update(
                    x_rgb=xc[:, :C], x_ir=xc[:, C:],
                    fused=_out.detach().cpu(),
                )
            return _fwd

        handles.append(mod.register_forward_hook(_make_hook(stage)))
        print(f"[hook] plain Conv@{stage}  "
              f"in_ch={mod.conv.in_channels}  out_ch={mod.conv.out_channels}")

    return captured, handles


def _hook_detect(model) -> tuple:
    """Hook Detect module to capture its input feature list.

    Returns (captured_dict, hook_handle).
    """
    from ultralytics.nn.modules.head import Detect

    captured: dict = {}
    for m in model.modules():
        if isinstance(m, Detect):
            def _fwd(_mod, inp, _out):
                feats = inp[0]
                captured["feats"] = [t.detach().cpu() for t in feats]
                captured["strides"] = _mod.stride.detach().cpu().tolist()
            handle = m.register_forward_hook(_fwd)
            print(f"[hook] Detect  strides={m.stride.tolist()}  nc={m.nc}")
            return captured, handle

    raise RuntimeError("No Detect module found in model")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_fusion_stage(stage: str, cap: dict, stem: str, stage_dir: Path) -> None:
    title = f"{stem} | fusion@{stage.upper()} (plain Concat+Conv1x1)"
    D = torch.abs(cap["x_rgb"] - cap["x_ir"])

    panels = [
        ("x_rgb (mean)", _chan_mean(cap["x_rgb"]), "viridis"),
        ("x_ir (mean)",  _chan_mean(cap["x_ir"]),  "viridis"),
        ("D=|R-I| (mean)", _chan_mean(D),           "inferno"),
        ("fused (mean)", _chan_mean(cap["fused"]),  "viridis"),
    ]
    _plot_overview(panels, title, stage_dir / "overview_mean.png")

    _plot_per_channel(cap["x_rgb"], "x_rgb",  title, stage_dir, "viridis")
    _plot_per_channel(cap["x_ir"],  "x_ir",   title, stage_dir, "viridis")
    _plot_per_channel(D,            "D",       title, stage_dir, "inferno")
    _plot_per_channel(cap["fused"], "fused",   title, stage_dir, "viridis")


def _save_head_scale(lvl: str, feat: torch.Tensor,
                     stride: float, stem: str, head_dir: Path) -> None:
    title = f"{stem} | head input @{lvl.upper()} (stride={int(stride)}, C={feat.shape[1]})"
    _plot_overview([("feat mean", _chan_mean(feat), "viridis")],
                   title, head_dir / "overview_mean.png")
    _plot_per_channel(feat, "feat", title, head_dir, "viridis")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize plain Concat+Conv1x1 dual-stream model features")
    p.add_argument("--ckpt",      required=True,
                   help="Path to checkpoint (.pt)")
    p.add_argument("--frame",     required=True,
                   help="Frame relative path, e.g. val/video10_frame_01154.jpg")
    p.add_argument("--data-root", default="RGBT-3M",
                   help="Dataset root containing RGB/ and IR/ subdirs")
    p.add_argument("--out",       default="tools/vis_plain", help="Output root directory")
    p.add_argument("--device",    default="cpu",  help="Inference device ('cpu', '0', 'cuda:0')")
    p.add_argument("--conf",      type=float, default=0.25, help="NMS confidence threshold")
    p.add_argument("--iou",       type=float, default=0.45, help="NMS IoU threshold")
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    frame     = Path(args.frame)          # e.g. val/video10_frame_01154.jpg
    rgb_path  = data_root / "RGB" / frame
    ir_path   = data_root / "IR"  / frame
    stem      = frame.stem                # video10_frame_01154

    out_root = Path(args.out) / stem
    (out_root / "inputs").mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    from ultralytics import YOLO
    from ultralytics.utils import ops

    print(f"[load] {args.ckpt}")
    yolo = YOLO(args.ckpt)
    model = yolo.model
    model.eval()

    fused_cap, fhooks = _hook_fusions(model)
    det_cap, dhook    = _hook_detect(model)

    # ── Build 6-channel input ────────────────────────────────────────────────
    rgb_np = _load_image(str(rgb_path))
    ir_np  = _load_image(str(ir_path))
    if rgb_np.shape != ir_np.shape:
        ir_np = cv2.resize(ir_np, (rgb_np.shape[1], rgb_np.shape[0]))

    rgb_t = _to_tensor(rgb_np)
    ir_t  = _to_tensor(ir_np)
    # Training pipeline's _format_img does transpose+[::-1] on the concatenated
    # 6ch HWC, reversing the full channel axis → final CHW is [I_R,I_G,I_B,R,G,B].
    # Model expects ch0:3 = IR (RGB order), ch3:6 = RGB (RGB order).
    # _load_image already does BGR→RGB, so no .flip(1) needed.
    x6 = torch.cat([ir_t, rgb_t], dim=1)

    dev = torch.device(args.device if args.device != "cpu" else "cpu")
    if args.device != "cpu":
        model.to(dev)

    print(f"[infer] input shape {tuple(x6.shape)}")
    with torch.no_grad():
        raw_out = model(x6.to(dev))

    # ── NMS ─────────────────────────────────────────────────────────────────
    preds = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
    dets_t = ops.non_max_suppression(preds, conf_thres=args.conf, iou_thres=args.iou)[0]
    dets = dets_t.detach().cpu().numpy() if len(dets_t) > 0 else np.zeros((0, 6))
    print(f"[det]  {len(dets)} detections")

    names = getattr(model, "names", {i: str(i) for i in range(10)})

    # ── Load GT labels ───────────────────────────────────────────────────────
    label_path = data_root / "labels" / frame.parent / (frame.stem + ".txt")
    h, w = rgb_np.shape[:2]
    gts = _load_gt_labels(label_path, h, w)
    print(f"[gt]   {len(gts)} ground-truth boxes from {label_path}")

    # ── Save input images & overlays ─────────────────────────────────────────
    _save_rgb(out_root / "inputs" / "rgb.png",      rgb_np)
    _save_rgb(out_root / "inputs" / "ir.png",       ir_np)
    _save_rgb(out_root / "inputs" / "rgb_pred.png", _draw_boxes(rgb_np, dets, names))
    _save_rgb(out_root / "inputs" / "ir_pred.png",  _draw_boxes(ir_np,  dets, names))
    _save_rgb(out_root / "inputs" / "rgb_gt.png",   _draw_gt_boxes(rgb_np, gts, names))
    _save_rgb(out_root / "inputs" / "ir_gt.png",    _draw_gt_boxes(ir_np,  gts, names))
    print(f"[save] inputs → {out_root / 'inputs'}")

    # ── Save fusion features per stage ───────────────────────────────────────
    for stage, cap in fused_cap.items():
        if not cap:
            print(f"[warn] No data for fusion@{stage} — skipped")
            continue
        stage_dir = out_root / f"fusion_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[fusion@{stage}]  shape={tuple(cap['fused'].shape)}")
        _save_fusion_stage(stage, cap, stem, stage_dir)

    # ── Save detection head inputs per scale ─────────────────────────────────
    if det_cap:
        for feat, s in zip(det_cap["feats"], det_cap["strides"]):
            lvl = _stride_to_level(s)
            head_dir = out_root / f"head_{lvl}"
            head_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[head@{lvl}]  stride={int(s)}  shape={tuple(feat.shape)}")
            _save_head_scale(lvl, feat, s, stem, head_dir)
    else:
        print("[warn] Detect hook did not fire — head inputs not saved")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    for h in fhooks:
        h.remove()
    dhook.remove()

    print(f"\n[done] All outputs in: {out_root.resolve()}")


if __name__ == "__main__":
    main()
