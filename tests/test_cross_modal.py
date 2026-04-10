"""Smoke tests for cross-modal attention modules (Exp-4a)."""

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
    """n=2 -> 1 self-attn group + 1 cross-modal group."""
    m = CrossModalA2C2f(c1=128, c2=128, n=2, area=4)
    assert len(m.m_self) == 1
    assert len(m.m_cross) == 1


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
