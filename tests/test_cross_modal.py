"""Smoke tests for cross-modal attention modules (Exp-4b: bidirectional residual CMA)."""

import pytest
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


def test_dual_stream_p2_four_scale_stride():
    """DualStream with p2 YAML produces 4 detection scales with strides [4,8,16,32]."""
    from ultralytics.nn.tasks import DualStreamDetectionModel
    model = DualStreamDetectionModel("yolov12-dual-p2.yaml", nc=3, verbose=False)
    model.eval()
    expected = torch.tensor([4.0, 8.0, 16.0, 32.0])
    assert torch.equal(model.stride.sort().values, expected), \
        f"Expected strides [4,8,16,32], got {model.stride}"


def test_dual_stream_p2_forward_shape():
    """DualStream P2 model forward pass returns valid output for 480x640 input."""
    from ultralytics.nn.tasks import DualStreamDetectionModel
    model = DualStreamDetectionModel("yolov12-dual-p2.yaml", nc=3, verbose=False)
    model.eval()
    x = torch.zeros(1, 6, 480, 640)
    with torch.no_grad():
        out = model(x)
    assert out is not None


def test_dual_stream_p2_uses_dmg_fusion():
    """With p2_fusion=dmg, fusion_convs['p2'] is a DMGFusion instance."""
    from ultralytics.nn.modules.block import DMGFusion
    from ultralytics.nn.tasks import DualStreamDetectionModel
    model = DualStreamDetectionModel("yolov12-dual-p2.yaml", nc=3, verbose=False)
    assert isinstance(model.fusion_convs["p2"], DMGFusion), \
        "fusion_convs['p2'] should be DMGFusion when p2_fusion=dmg"


def test_dual_stream_p2_rejects_cmg_at_p2():
    """Setting cmg_stages=[p2] must raise ValueError (p2 is not a valid CMG stage)."""
    import pytest
    from ultralytics.nn.tasks import DualStreamDetectionModel
    from ultralytics.utils import yaml_load
    from ultralytics.utils.checks import check_yaml
    cfg_path = check_yaml("yolov12-dual-p2.yaml")
    cfg = yaml_load(cfg_path)
    cfg["cmg_stages"] = ["p2"]
    with pytest.raises(ValueError, match="p2"):
        DualStreamDetectionModel(cfg, nc=3, verbose=False)


def test_sgmc_calibrates_multiscale_features():
    """SGMC preserves target feature shapes and exposes bounded gate debug state."""
    from ultralytics.nn.modules.block import SemanticGuidedMultiScaleCalibration

    channels = {"p3": 128, "p4": 256, "p5": 512}
    m = SemanticGuidedMultiScaleCalibration(
        channels=channels,
        source="p5",
        targets=("p3", "p4", "p5"),
        ratio=0.25,
        gate_limit=0.1,
        init_gate=0.001,
    )
    feats = {
        "p3": torch.randn(2, 128, 32, 32),
        "p4": torch.randn(2, 256, 16, 16),
        "p5": torch.randn(2, 512, 8, 8),
    }
    out = m(feats)
    assert set(out) == set(feats)
    for stage_name, feat in feats.items():
        assert out[stage_name].shape == feat.shape
    debug = m.debug_state()
    assert debug["p3_gate"] == pytest.approx(0.001, abs=1e-6)
    assert debug["p4_gate"] == pytest.approx(0.001, abs=1e-6)
    assert debug["p5_gate"] == pytest.approx(0.001, abs=1e-6)


def test_dual_stream_p2_sgmc_forward_and_debug_state():
    """SGMC YAML builds a four-scale model and logs per-stage SGMC gate scalars."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual-p2-sgmc.yaml", nc=3, verbose=False)
    model.eval()
    x = torch.zeros(1, 6, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out is not None
    debug = model.adapter_debug_state()
    assert "sgmc/p3_gate" in debug
    assert "sgmc/p4_gate" in debug
    assert "sgmc/p5_gate" in debug


@pytest.mark.parametrize(
    ("cfg", "expected_p4_module", "uses_aux_head"),
    (
        ("yolov12-dual-p2-plain-a2c2fp4-noaux.yaml", "A2C2f", False),
        ("yolov12-dual-p2-plain-a2c2fp4-p3aux.yaml", "A2C2f", True),
        ("yolov12-dual-p2-plain-c3k2p4-noaux.yaml", "C3k2", False),
    ),
)
def test_plain_p2_fair_ablation_cfgs_instantiate_expected_models(cfg, expected_p4_module, uses_aux_head):
    """Fair plainP2 ablation YAMLs instantiate the expected P2 fusion and P4 block."""
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel(cfg, nc=3, verbose=False)

    assert model.yaml["p2_fusion"] == "plain"
    assert model.yaml["backbone"][6][2] == expected_p4_module
    assert type(model.backbone_rgb[6]).__name__ == expected_p4_module
    assert type(model.backbone_ir[6]).__name__ == expected_p4_module
    assert isinstance(model.fusion_convs["p2"], Conv)
    assert not any(type(m).__name__ == "DMGFusion" for m in model.modules())

    if uses_aux_head:
        assert "noaux" not in cfg
    else:
        assert "noaux" in cfg
