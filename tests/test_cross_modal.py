"""Smoke tests for cross-modal attention modules (Exp-4b: bidirectional residual CMA)."""

import torch

from ultralytics.nn.modules.block import CrossModalA2C2f, CrossModalAAttn


def test_cross_modal_aattn_shape():
    """CrossModalAAttn should preserve spatial dims, Q from x_self, KV from x_other."""
    m = CrossModalAAttn(dim=64, num_heads=2, area=4)
    x_self = torch.randn(2, 64, 8, 8)
    x_other = torch.randn(2, 64, 8, 8)
    out = m(x_self, x_other)
    assert out.shape == (2, 64, 8, 8)


def test_cross_modal_aattn_area1():
    """area=1 means no spatial partitioning (used at P5)."""
    m = CrossModalAAttn(dim=64, num_heads=2, area=1)
    x_self = torch.randn(2, 64, 4, 4)
    x_other = torch.randn(2, 64, 4, 4)
    out = m(x_self, x_other)
    assert out.shape == (2, 64, 4, 4)


def test_cross_modal_a2c2f_shape():
    """CrossModalA2C2f: output shape matches input shape."""
    m = CrossModalA2C2f(c1=128, c2=128, n=2, area=4)
    x_self = torch.randn(2, 128, 8, 8)
    x_other = torch.randn(2, 128, 8, 8)
    out = m(x_self, x_other)
    assert out.shape == (2, 128, 8, 8)


def test_cross_modal_a2c2f_group_split():
    """n=2 -> all 2 groups in m_self (full self-attn capacity) + 1 cross-modal residual."""
    m = CrossModalA2C2f(c1=128, c2=128, n=2, area=4)
    assert len(m.m_self) == 2   # ALL n groups keep self-attention
    assert len(m.m_cross) == 1  # last 1 group also has cross-modal residual
    assert hasattr(m, "cross_scale"), "cross_scale parameter must exist"


def test_cross_modal_a2c2f_p5_config():
    """P5 config: 256ch, area=1."""
    m = CrossModalA2C2f(c1=256, c2=256, n=2, area=1)
    x_self = torch.randn(2, 256, 4, 4)
    x_other = torch.randn(2, 256, 4, 4)
    out = m(x_self, x_other)
    assert out.shape == (2, 256, 4, 4)


def test_dual_stream_cma_forward():
    """DualStreamDetectionModel with cma_stages produces valid output."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual.yaml", nc=3)
    model.eval()
    x = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out is not None


def test_dual_stream_cma_and_cmg_combined():
    """cma_stages and cmg_stages work simultaneously."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual.yaml", nc=3)
    model.eval()
    assert len(model.cmg_modules) > 0, "CMG modules should be present"
    assert len(model._cma_layer_to_stage) > 0, "CMA stages should be configured"
    x = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out is not None


def test_dual_stream_cma_bidirectional():
    """Both RGB and IR backbones have CrossModalA2C2f (bidirectional CMA)."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual.yaml", nc=3, verbose=False)
    for layer_idx in model._cma_layer_to_stage:
        assert isinstance(model.backbone_rgb[layer_idx], CrossModalA2C2f), \
            f"backbone_rgb[{layer_idx}] should be CrossModalA2C2f"
        assert isinstance(model.backbone_ir[layer_idx], CrossModalA2C2f), \
            f"backbone_ir[{layer_idx}] should be CrossModalA2C2f (bidirectional)"
        # Weights must be independent (different objects)
        assert model.backbone_rgb[layer_idx] is not model.backbone_ir[layer_idx]


def test_dmg_fusion_output_shape():
    """DMGFusion preserves spatial dims and channel count."""
    from ultralytics.nn.modules.block import DMGFusion
    m = DMGFusion(channels=64)
    x_rgb = torch.randn(2, 64, 120, 160)
    x_ir  = torch.randn(2, 64, 120, 160)
    out = m(x_rgb, x_ir)
    assert out.shape == (2, 64, 120, 160), f"Expected (2,64,120,160), got {out.shape}"


def test_dmg_fusion_neutral_init():
    """At alpha=0, beta=1, gradients reach both inputs."""
    from ultralytics.nn.modules.block import DMGFusion
    m = DMGFusion(channels=32)
    assert m.alpha.item() == 0.0, "alpha must initialise to 0"
    assert m.beta.item()  == 1.0, "beta must initialise to 1"
    x_rgb = torch.randn(2, 32, 8, 8, requires_grad=True)
    x_ir  = torch.randn(2, 32, 8, 8, requires_grad=True)
    m.eval()
    out = m(x_rgb, x_ir)
    out.sum().backward()
    assert x_rgb.grad is not None, "gradient must flow to x_rgb"
    assert x_ir.grad  is not None, "gradient must flow to x_ir"


def test_dmg_fusion_param_count():
    """DMGFusion for C=64 should have fewer than 20K parameters."""
    from ultralytics.nn.modules.block import DMGFusion
    m = DMGFusion(channels=64)
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params < 20_000, f"Too many params: {n_params} (expected < 20K)"
