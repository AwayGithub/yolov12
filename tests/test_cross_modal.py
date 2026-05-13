"""Smoke tests for dual-stream DMG and plain P2 configurations."""

import pytest
import torch

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


def test_dmg_posalpha_bounds_alpha():
    """DMGFusionPosAlpha keeps differential gain non-negative and bounded."""
    from ultralytics.nn.modules.block import DMGFusionPosAlpha

    m = DMGFusionPosAlpha(channels=32, alpha_max=3.0, alpha_init=1.0)
    assert 0.0 <= m.alpha.item() <= 3.0
    assert m.alpha.item() == pytest.approx(1.0, abs=1e-6)

    x_rgb = torch.randn(2, 32, 8, 8)
    x_ir = torch.randn(2, 32, 8, 8)
    out = m(x_rgb, x_ir)
    assert out.shape == x_rgb.shape


def test_dmg_init8d_starts_from_differential_amplifier_prior():
    """DMGFusionInit8d starts from positive alpha and weak negative beta."""
    from ultralytics.nn.modules.block import DMGFusionInit8d

    m = DMGFusionInit8d(channels=32, alpha_init=1.0, beta_init=-0.1)
    assert m.alpha.item() == pytest.approx(1.0, abs=1e-6)
    assert m.beta.item() == pytest.approx(-0.1, abs=1e-6)


def test_dual_stream_rejects_unknown_p2_fusion_modes():
    """Unknown P2 fusion modes should fail fast instead of silently falling back to plain fusion."""
    from ultralytics.nn.tasks import DualStreamDetectionModel
    from ultralytics.utils import yaml_load
    from ultralytics.utils.checks import check_yaml

    cfg = yaml_load(check_yaml("yolov12-dual-p2.yaml"))
    cfg["p2_fusion"] = "unsupported_fusion"

    with pytest.raises(ValueError, match="unsupported_fusion"):
        DualStreamDetectionModel(cfg, nc=3, verbose=False)


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
    from ultralytics.utils import yaml_load
    from ultralytics.utils.checks import check_yaml

    cfg = yaml_load(check_yaml("yolov12-dual-p2.yaml"))
    cfg["p2_fusion"] = "dmg"
    model = DualStreamDetectionModel(cfg, nc=3, verbose=False)
    assert isinstance(model.fusion_convs["p2"], DMGFusion), \
        "fusion_convs['p2'] should be DMGFusion when p2_fusion=dmg"


@pytest.mark.parametrize(
    ("cfg", "expected_type"),
    (
        ("yolov12-dual-p2-dmg-posalpha.yaml", "DMGFusionPosAlpha"),
        ("yolov12-dual-p2-dmg-init8d.yaml", "DMGFusionInit8d"),
    ),
)
def test_dmg_p2_variant_cfgs_instantiate_expected_fusion(cfg, expected_type):
    """The active DMG follow-up YAMLs use P4 A2C2f, P3 aux, and intended P2 fusion variants."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel(cfg, nc=3, verbose=False)

    assert model.yaml["p2_fusion"] in {"dmg_posalpha", "dmg_init8d"}
    assert type(model.fusion_convs["p2"]).__name__ == expected_type
    assert model.yaml["backbone"][6][2] == "A2C2f"
    assert type(model.backbone_rgb[6]).__name__ == "A2C2f"
    assert type(model.backbone_ir[6]).__name__ == "A2C2f"
    assert model.use_aux_head is True


def test_dmg_p2_variant_debug_state_logs_alpha_beta():
    """DMG P2 variants expose alpha/beta scalars so results.csv records the gate trajectory."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual-p2-dmg-posalpha.yaml", nc=3, verbose=False)
    debug = model.adapter_debug_state()

    assert "dmg/p2_alpha" in debug
    assert "dmg/p2_beta" in debug
    assert debug["dmg/p2_alpha"] == pytest.approx(1.0, abs=1e-6)
    assert debug["dmg/p2_beta"] == pytest.approx(1.0, abs=1e-6)
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
