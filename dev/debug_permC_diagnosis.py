#!/usr/bin/env python3
"""Diagnose why Perm-C produces mAP=0 by comparing intermediate features with B2."""

import torch
from ultralytics import YOLO

DEVICE = "cuda:0"


def _load_model(ckpt_path):
    model = YOLO(ckpt_path).model.to(DEVICE).eval()
    return model


def _forward_model(model, x):
    x_ir = x[:, :3, ...]
    x_rgb = x[:, 3:, ...]
    y_rgb, y_ir = [], []
    feats_rgb, feats_ir = {}, {}

    for m_rgb, m_ir in zip(model.backbone_rgb, model.backbone_ir):
        li = m_rgb.i
        if m_rgb.f != -1:
            x_rgb = y_rgb[m_rgb.f] if isinstance(m_rgb.f, int) else [x_rgb if j == -1 else y_rgb[j] for j in m_rgb.f]
        if m_ir.f != -1:
            x_ir = y_ir[m_ir.f] if isinstance(m_ir.f, int) else [x_ir if j == -1 else y_ir[j] for j in m_ir.f]
        pcross_stages = getattr(model, "_parallel_cross_layer_to_stage", {})
        if li in pcross_stages:
            x_rgb, x_ir = m_rgb(x_rgb, x_ir)
        else:
            x_rgb = m_rgb(x_rgb)
            x_ir = m_ir(x_ir)
        y_rgb.append(x_rgb if li in model.save else None)
        y_ir.append(x_ir if li in model.save else None)
        for sn, si in model.FUSION_LAYER_INDICES.items():
            if li == si:
                feats_rgb[sn] = x_rgb
                feats_ir[sn] = x_ir

    fused = {}
    for sn in model.FUSION_LAYER_INDICES:
        r, i = feats_rgb[sn], feats_ir[sn]
        fc = model.fusion_convs[sn]
        fc_name = fc.__class__.__name__
        if fc_name in ("DMGFusion", "DMGFusionPosAlpha", "DMGFusionInit8d", "FreDFTFusion",
                       "M2DLocalIlluminationFusion"):
            fused[sn] = fc(r, i)
        else:
            fused[sn] = fc(torch.cat([r, i], dim=1))

    y = [None] * (max(model.FUSION_LAYER_INDICES.values()) + 1)
    for sn, li in model.FUSION_LAYER_INDICES.items():
        y[li] = fused[sn]

    x = fused["p5"]
    for m in model.head:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in model.save else None)
    return fused, x


def _stats(t):
    t = t.float().detach()
    return round(t.min().item(), 4), round(t.max().item(), 4), round(t.mean().item(), 4), round(t.std().item(), 4)


def _main_head_conf(det_out):
    """Extract main head confidence values (det_out[0] for concat head)."""
    d = det_out[0] if isinstance(det_out, (list, tuple)) else det_out
    return d[..., 4:].reshape(-1)


def diagnose(label, ckpt):
    model = _load_model(ckpt)
    dummy = torch.randn(4, 6, 480, 640, device=DEVICE)
    with torch.no_grad():
        fused, det_out = _forward_model(model, dummy)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print("\n--- Stage-wise fused feature stats (min / max / mean / std) ---")
    for stage in ["p2", "p3", "p4", "p5"]:
        mn, mx, mean, std = _stats(fused[stage])
        print(f"  {stage}: {mn:>8.2f}  {mx:>8.2f}  {mean:>8.2f}  {std:>8.2f}")

    print("\n--- Main head stats ---")
    main = det_out[0] if isinstance(det_out, (list, tuple)) else det_out
    mn, mx, mean, std = _stats(main)
    print(f"  shape={list(main.shape)}  min={mn:>10.4f}  max={mx:>10.4f}  mean={mean:>10.4f}")

    confs = _main_head_conf(det_out)
    print("\n--- Raw confidence (pre-sigmoid) per-stride heads ---")
    # det_out[1] is list of aux heads [p2, p3, p4, p5]
    if isinstance(det_out, (list, tuple)) and len(det_out) > 1:
        for si, stride_name in enumerate(["p2_aux", "p3_aux", "p4_aux", "p5_aux"]):
            if si < len(det_out[1]):
                d = det_out[1][si]
                d_conf = d[..., 4:].reshape(-1)
                mn_c, mx_c = d_conf.min().item(), d_conf.max().item()
                n_gt = (d_conf > 0.001).sum().item()
                print(f"  {stride_name} [{list(d.shape)}]: conf min={mn_c:.4f} max={mx_c:.4f}  "
                      f">0.001={n_gt}/{d_conf.numel()}")

    print("\n--- Main head confidence distribution ---")
    percentiles = [0, 50, 90, 95, 99, 99.9, 100]
    ps = torch.quantile(confs.float(), torch.tensor([p/100 for p in percentiles], device=confs.device))
    for p, v in zip(percentiles, ps):
        print(f"  raw_conf p{p:>5}: {v.item():>10.4f}  sigmoid={torch.sigmoid(v).item():.6f}")

    n_gt = (confs > 0.001).sum().item()
    n_pos = (conf_sigmoid := torch.sigmoid(confs)) > 0.001
    print(f"\n  >0.001 (raw): {n_gt}/{confs.numel()}   "
          f">0.001 (sigmoid): {n_pos.sum().item()}/{confs.numel()}")

    print("\n--- P3/P4 ParallelCross gamma ---")
    if hasattr(model, "adapter_debug_state"):
        dbg = model.adapter_debug_state()
        for k in sorted(dbg):
            if 'gamma' in k:
                print(f"  {k}: {dbg[k]:.6f}")

    return fused, det_out


if __name__ == "__main__":
    b2_ckpt = "runs/detect/train_fp_B2_dual_seed0_cls01/weights/last.pt"
    permc_ckpt = "runs/detect/train_fp_permC_fredftP2_pcrossP3_dmgP4_seed0_cls01/weights/last.pt"

    fuse_b2, det_b2 = diagnose("B2 (DMG@P2 + FreDFT@P3 + ParallelCross@P4)", b2_ckpt)
    fuse_pc, det_pc = diagnose("Perm-C (FreDFT@P2 + ParallelCross@P3 + DMG@P4)", permc_ckpt)

    print("\n\n" + "=" * 70)
    print("  COMPARISON: fused feature mean/std ratios (Perm-C / B2)")
    print("=" * 70)
    for stage in ["p2", "p3", "p4", "p5"]:
        _, _, m_b2, s_b2 = _stats(fuse_b2[stage])
        _, _, m_pc, s_pc = _stats(fuse_pc[stage])
        r_mean = m_pc / (m_b2 + 1e-6) if m_b2 != 0 else float("inf")
        r_std = s_pc / (s_b2 + 1e-6) if s_b2 != 0 else float("inf")
        print(f"  {stage}: mean {m_b2:.2f} -> {m_pc:.2f} (x{r_mean:.1f})  |  "
              f"std {s_b2:.2f} -> {s_pc:.2f} (x{r_std:.1f})")

    print("\n\n" + "=" * 70)
    print("  COMPARISON: main head output")
    print("=" * 70)
    for name, det in ("B2", det_b2), ("Perm-C", det_pc):
        main = det[0] if isinstance(det, (list, tuple)) else det
        mn, mx, mean, std = _stats(main)
        confs = _main_head_conf(det)
        print(f"  {name}: shape={list(main.shape)}  min={mn:.2f}  max={mx:.2f}  mean={mean:.2f}  "
              f"conf>0.001={(confs>0.001).sum().item()}/{confs.numel()}")
