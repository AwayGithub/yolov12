"""
CP-YOLOv11-MF: Cross-modal Progressive YOLOv11 with Mid-term Fusion

Reference: Zhang et al. "A UAV-Based Multi-Scenario RGB-Thermal Dataset and Fusion
Model for Enhanced Forest Fire Detection" (Remote Sens. 2025, 17, 2593)

Backbone: YOLOv11-n (width=0.25, depth=0.50) exactly matching ultralytics/cfg/models/11/yolo11.yaml.
Fusion: CPCA + PPAS at mid-term (after backbone, before FPN neck).
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ultralytics.nn.modules import C3k2, SPPF, C2PSA, Conv


# ---------------------------------------------------------------------------
# CPCA — Channel Prior Convolutional Attention (Eq.4-6)
# ---------------------------------------------------------------------------
class CPCA(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid())
        self.spatial = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False),
            nn.Conv2d(ch, 1, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_mlp(x).unsqueeze(-1).unsqueeze(-1) * self.spatial(x)


# ---------------------------------------------------------------------------
# PPAS — Parallel Patch-Aware Splicing (Fig.12-13)
# ---------------------------------------------------------------------------
class PPAS(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.local_br = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.global_br = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.serial_br = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_ch))
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(out_ch, out_ch // 4), nn.ReLU(inplace=True),
            nn.Linear(out_ch // 4, out_ch), nn.Sigmoid())
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_ch, 1, 7, 1, 3, bias=False), nn.Sigmoid())
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        fl = self.local_br(x)
        fg = self.global_br(x)
        fs = self.serial_br(x)
        fused = fl + fg + fs
        fused = fused * self.channel_attn(fused).unsqueeze(-1).unsqueeze(-1)
        fused = fused * self.spatial_attn(fused)
        return self.act(self.bn(fused))


# ---------------------------------------------------------------------------
# YOLOv11-n backbone — EXACT match to ultralytics/cfg/models/11/yolo11.yaml
# width=0.25, depth=0.50 → n scale
# ---------------------------------------------------------------------------
class YOLO11Backbone(nn.Module):
    """
    Outputs the 3 feature maps consumed by FPN neck:
      - fpn_p3: layer 4 output (C3k2 at P3 level) → 128ch
      - fpn_p4: layer 6 output (C3k2 at P4 level) → 128ch
      - fpn_p5: layer 10 output (C2PSA) → 256ch
    """
    def __init__(self, c_in=3):
        super().__init__()
        # With width=0.25: actual channels = base * 0.25
        # stem
        self.layer0 = Conv(c_in, 16, 3, 2)     # P1/2
        self.layer1 = Conv(16, 32, 3, 2)        # P2/4
        # P2 block
        self.layer2 = C3k2(32, 64, 1, False, e=0.25)   # P2
        # P3 block
        self.layer3 = Conv(64, 64, 3, 2)         # P3/8
        self.layer4 = C3k2(64, 128, 1, False, e=0.25)  # P3 feature
        # P4 block
        self.layer5 = Conv(128, 128, 3, 2)       # P4/16
        self.layer6 = C3k2(128, 128, 1, True)    # P4 feature
        # P5 block
        self.layer7 = Conv(128, 256, 3, 2)       # P5/32
        self.layer8 = C3k2(256, 256, 1, True)    # P5
        self.layer9 = SPPF(256, 256, 5)
        self.layer10 = C2PSA(256, 256, 1)

    def forward(self, x):
        x = self.layer0(x)   # 16ch, /2
        x = self.layer1(x)   # 32ch, /4
        x = self.layer2(x)   # 64ch, /4
        x = self.layer3(x)   # 64ch, /8
        p3 = self.layer4(x)  # 128ch, /8  → FPN P3
        x = self.layer5(p3)  # 128ch, /16
        p4 = self.layer6(x)  # 128ch, /16 → FPN P4
        x = self.layer7(p4)  # 256ch, /32
        x = self.layer8(x)   # 256ch, /32
        x = self.layer9(x)   # 256ch
        p5 = self.layer10(x) # 256ch      → FPN P5
        return p3, p4, p5


# ---------------------------------------------------------------------------
# FPN neck + Detect — matches ultralytics YOLOv11 head exactly
# ---------------------------------------------------------------------------
class FPNNeck(nn.Module):
    def __init__(self, nc=2):
        super().__init__()
        # FPN top-down
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.fpn4 = C3k2(256 + 128, 128, 1, False)   # concat(P5_up, P4) → 128ch
        self.fpn3 = C3k2(128 + 128, 64, 1, False)    # concat(fpn4_up, P3) → 64ch
        # PAN bottom-up
        self.pan3 = Conv(64, 64, 3, 2)
        self.pan4 = C3k2(64 + 128, 128, 1, False)    # concat(pan3, fpn4) → 128ch
        self.pan5_pre = Conv(128, 128, 3, 2)
        self.pan5 = C3k2(128 + 256, 256, 1, True)    # concat(pan5_pre, P5) → 256ch
        # Detect
        from ultralytics.nn.modules.head import Detect
        self.detect = Detect(nc, [64, 128, 256])

    def forward(self, p3, p4, p5):
        # FPN
        fpn4 = self.fpn4(torch.cat([self.up(p5), p4], 1))    # 128ch
        fpn3 = self.fpn3(torch.cat([self.up(fpn4), p3], 1))  # 64ch
        # PAN
        pan4 = self.pan4(torch.cat([self.pan3(fpn3), fpn4], 1))  # 128ch
        pan5 = self.pan5(torch.cat([self.pan5_pre(pan4), p5], 1))  # 256ch
        return self.detect([fpn3, pan4, pan5])


# ---------------------------------------------------------------------------
# CP-YOLOv11-MF — full model
# ---------------------------------------------------------------------------
class CPYOLOv11MF(nn.Module):
    """
    Dual YOLOv11-n backbone + CPCA + PPAS + FPN.
    Scheme 2 (best balanced, Table 11): PPAS at P3/P4, Concat+CPCA at P5.
    """
    def __init__(self, nc=2):
        super().__init__()
        self.rgb_bb = YOLO11Backbone(3)
        self.ir_bb = YOLO11Backbone(3)

        # PPAS for P3/P4 fusion (Scheme 2)
        self.ppas_p3 = PPAS(128 * 2, 128)   # concat 128+128=256 → 128
        self.ppas_p4 = PPAS(128 * 2, 128)   # concat 128+128=256 → 128

        # CPCA for all scales
        self.cpca_p3 = CPCA(128, reduction=16)
        self.cpca_p4 = CPCA(128, reduction=16)
        self.cpca_p5 = CPCA(256 * 2, reduction=16)  # P5 concat=512

        # P5 projection: 512 → 256 (after concat+CPCA)
        self.proj_p5 = Conv(512, 256, 1)

        self.neck = FPNNeck(nc)

    def forward(self, x):
        x_rgb, x_ir = x[:, :3], x[:, 3:]

        rgb_p3, rgb_p4, rgb_p5 = self.rgb_bb(x_rgb)
        ir_p3, ir_p4, ir_p5 = self.ir_bb(x_ir)

        # P3: PPAS + CPCA
        fuse_p3 = self.ppas_p3(torch.cat([rgb_p3, ir_p3], 1))
        fuse_p3 = fuse_p3 + self.cpca_p3(fuse_p3)

        # P4: PPAS + CPCA
        fuse_p4 = self.ppas_p4(torch.cat([rgb_p4, ir_p4], 1))
        fuse_p4 = fuse_p4 + self.cpca_p4(fuse_p4)

        # P5: concat + CPCA + project (Scheme 2: no PPAS for large targets)
        fuse_p5 = torch.cat([rgb_p5, ir_p5], 1)
        fuse_p5 = fuse_p5 + self.cpca_p5(fuse_p5)
        fuse_p5 = self.proj_p5(fuse_p5)

        return self.neck(fuse_p3, fuse_p4, fuse_p5)


if __name__ == "__main__":
    model = CPYOLOv11MF(nc=2).eval()
    x = torch.randn(1, 6, 480, 640)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, list):
        for i, o in enumerate(out):
            shp = o.shape if hasattr(o, 'shape') else type(o)
            print(f"output {i}: {shp}")
    else:
        print(f"output: {out.shape if hasattr(out, 'shape') else type(out)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"total params: {total:,} ({total/1e6:.1f}M)")
