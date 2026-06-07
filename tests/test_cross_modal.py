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


def test_fredft_fusion_output_shape_and_gradients():
    """FreDFTFusion preserves shape and trains cross-modal QK plus dilated FFN paths."""
    from ultralytics.nn.modules.block import FreDFTFusion

    m = FreDFTFusion(channels=64, expansion=1.0, qkv_expand=1.0)
    x_rgb = torch.randn(2, 64, 20, 24, requires_grad=True)
    x_ir = torch.randn(2, 64, 20, 24, requires_grad=True)

    out = m(x_rgb, x_ir)
    out.mean().backward()

    assert out.shape == x_rgb.shape
    assert x_rgb.grad is not None
    assert x_ir.grad is not None
    assert m.freq_attn.to_hidden.weight.grad is not None
    assert m.ffn.project_in.weight.grad is not None
    assert m.fuse.weight.grad is not None


def test_fredft_fusion_uses_cross_qk_dilated_ffn_structure():
    """FreDFTFusion should allow C->3C attention, 1C dilated FFN, and no outer scale residual."""
    from ultralytics.nn.modules.block import FreDFTFusion

    channels = 64
    m = FreDFTFusion(channels=channels, expansion=1.0, qkv_expand=1.0)
    expected_ffn_hidden = channels + (-channels) % 3

    assert m.freq_attn.to_hidden.out_channels == channels * 3
    assert m.ffn.project_in.out_channels == expected_ffn_hidden
    assert m.ffn.dwconv_d1.dilation == (1, 1)
    assert m.ffn.dwconv_d2.dilation == (2, 2)
    assert m.ffn.dwconv_d3.dilation == (3, 3)
    assert not hasattr(m, "plain_fuse")
    assert not hasattr(m, "freq_fuse")
    assert not hasattr(m, "freq_scale")


def test_fredft_fusion_checkpoint_ffn_keeps_training_backward():
    """FreDFT FFN checkpointing should be configurable and preserve gradients."""
    from ultralytics.nn.modules.block import FreDFTFusion

    m = FreDFTFusion(channels=8, expansion=1.0, qkv_expand=1.0, checkpoint_ffn=True)
    m.train()
    x_rgb = torch.randn(1, 8, 8, 8, requires_grad=True)
    x_ir = torch.randn(1, 8, 8, 8, requires_grad=True)

    y = m(x_rgb, x_ir)
    y.mean().backward()

    assert m.checkpoint_ffn is True
    assert x_rgb.grad is not None
    assert x_ir.grad is not None
    assert m.ffn.project_out.weight.grad is not None


def test_m2d_lifusion_uses_rgb_illumination_map_and_backprops():
    """M2D-LIF-style fusion uses an RGB-derived illumination map and preserves feature shape."""
    from ultralytics.nn.modules.block import M2DLocalIlluminationFusion, M2DLocalIlluminationGate

    gate = M2DLocalIlluminationGate()
    fusion = M2DLocalIlluminationFusion(stage="p4")
    x_rgb_image = torch.randn(2, 3, 480, 640, requires_grad=True)
    illum = gate(x_rgb_image)

    x_rgb = torch.randn(2, 64, 30, 40, requires_grad=True)
    x_ir = torch.randn(2, 64, 30, 40, requires_grad=True)
    out = fusion(x_rgb, x_ir, illum)
    out.mean().backward()

    assert illum.shape == (2, 1, 60, 80)
    assert out.shape == x_rgb.shape
    assert x_rgb_image.grad is not None
    assert x_rgb.grad is not None
    assert x_ir.grad is not None


def test_cross_aattn_preserves_query_shape_and_uses_guide_kv():
    """CrossAAttn should use target Q with guide K/V while preserving the target BCHW contract."""
    from ultralytics.nn.modules.block import CrossAAttn

    m = CrossAAttn(dim=64, num_heads=2, area=4)
    x_rgb = torch.randn(2, 64, 12, 16, requires_grad=True)
    x_ir = torch.randn(2, 64, 12, 16, requires_grad=True)

    y = m(x_rgb, x_ir)
    y.mean().backward()

    assert y.shape == x_rgb.shape
    assert x_rgb.grad is not None
    assert x_ir.grad is not None
    assert m.q.conv.weight.grad is not None
    assert m.kv.conv.weight.grad is not None
    assert m.proj.conv.weight.grad is not None


def test_dual_parallel_cross_a2c2f_uses_four_half_width_concat_and_gamma_residual():
    """Parallel cross A2C2f should split into self/cross halves, concat four paths, and gate the residual."""
    from ultralytics.nn.modules.block import CrossABlock, DualParallelCrossA2C2f

    m = DualParallelCrossA2C2f(c1=128, c2=128, n=2, area=4, cross_mlp_ratio=1.0, scale_init=0.01)
    x_rgb = torch.randn(2, 128, 12, 16, requires_grad=True)
    x_ir = torch.randn(2, 128, 12, 16, requires_grad=True)

    y_rgb, y_ir = m(x_rgb, x_ir)
    (y_rgb.mean() + y_ir.mean()).backward()

    assert y_rgb.shape == x_rgb.shape
    assert y_ir.shape == x_ir.shape
    assert m.c_branch == 64
    assert m.cv1_rgb.conv.out_channels == 128
    assert m.cv2_rgb.conv.in_channels == 256
    assert len(m.self_rgb) == 2
    assert len(m.cross_rgb) == 2
    assert all(isinstance(block, CrossABlock) for block in m.cross_rgb)
    assert m.self_rgb[0].mlp[0].conv.out_channels == 128
    assert m.cross_rgb[0].mlp[0].conv.out_channels == 64
    assert m.gamma_rgb.item() == pytest.approx(0.01, abs=1e-6)
    assert m.gamma_ir.item() == pytest.approx(0.01, abs=1e-6)
    assert m.gamma_rgb.grad is not None
    assert m.gamma_ir.grad is not None


def test_dual_parallel_cross_a2c2f_keeps_cross_guide_fixed_to_original_split():
    """The second cross block should still attend to the opposite modality's original split feature."""
    import torch.nn as nn
    from ultralytics.nn.modules.block import DualParallelCrossA2C2f

    class IdentityBlock(nn.Module):

        def forward(self, x):
            return x

    class RecordingCrossBlock(nn.Module):

        def __init__(self):
            super().__init__()
            self.guide_ids = []

        def forward(self, x, guide):
            self.guide_ids.append(id(guide))
            return x + 1

    m = DualParallelCrossA2C2f(c1=128, c2=128, n=2, area=4, scale_init=0.01)
    m.self_rgb = nn.ModuleList([IdentityBlock(), IdentityBlock()])
    m.self_ir = nn.ModuleList([IdentityBlock(), IdentityBlock()])
    m.cross_rgb = nn.ModuleList([RecordingCrossBlock(), RecordingCrossBlock()])
    m.cross_ir = nn.ModuleList([RecordingCrossBlock(), RecordingCrossBlock()])

    y_rgb, y_ir = m(torch.randn(1, 128, 12, 16), torch.randn(1, 128, 12, 16))

    assert y_rgb.shape == (1, 128, 12, 16)
    assert y_ir.shape == (1, 128, 12, 16)
    assert m.cross_rgb[0].guide_ids[0] == m.cross_rgb[1].guide_ids[0]
    assert m.cross_ir[0].guide_ids[0] == m.cross_ir[1].guide_ids[0]


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


def test_dual_stream_fredft_p3_cfg_extends_confirmed_dmg_init8d_baseline():
    """FreDFT config should extend P3 aux + P2 DMG Init8d, without CMG/CMA."""
    from ultralytics.nn.modules.block import DMGFusionInit8d, FreDFTFusion
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual-p2-fredft-p3.yaml", nc=3, verbose=False)

    assert model.yaml["p2_fusion"] == "dmg_init8d"
    assert model.yaml["dmg_alpha_init"] == pytest.approx(1.0)
    assert model.yaml["dmg_beta_init"] == pytest.approx(-0.1)
    assert model.yaml["freq_fusion_stages"] == ["p3"]
    assert model.yaml["fredft_expansion"] == pytest.approx(3.0)
    assert model.yaml.get("fredft_qkv_expand", 2.0) == pytest.approx(2.0)
    assert isinstance(model.fusion_convs["p2"], DMGFusionInit8d)
    assert model.fusion_convs["p2"].alpha.item() == pytest.approx(1.0)
    assert model.fusion_convs["p2"].beta.item() == pytest.approx(-0.1)
    assert isinstance(model.fusion_convs["p3"], FreDFTFusion)
    assert isinstance(model.fusion_convs["p4"], Conv)
    assert isinstance(model.fusion_convs["p5"], Conv)
    assert model.use_aux_head is True
    assert "cmg_stages" not in model.yaml
    assert "cma_stages" not in model.yaml
    assert "fredft_scale_init" not in model.yaml


@pytest.mark.parametrize(
    ("cfg", "expected_stages"),
    (
        ("yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3.yaml", ["p3"]),
        ("yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3p4.yaml", ["p3", "p4"]),
        ("yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3p4p5.yaml", ["p3", "p4", "p5"]),
    ),
)
def test_fredft_stage_sweep_cfgs_extend_dmg_init8d_p3aux_baseline(cfg, expected_stages):
    """FreDFT stage-sweep configs should only vary frequency fusion stages."""
    from ultralytics.nn.modules.block import DMGFusionInit8d, FreDFTFusion
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel(cfg, nc=3, verbose=False)

    assert model.yaml["p2_fusion"] == "dmg_init8d"
    assert model.yaml["dmg_alpha_init"] == pytest.approx(1.0)
    assert model.yaml["dmg_beta_init"] == pytest.approx(-0.1)
    assert model.yaml["freq_fusion_stages"] == expected_stages
    expected_expansion = 3.0 if cfg == "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3.yaml" else 1.0
    assert model.yaml["fredft_expansion"] == pytest.approx(expected_expansion)
    assert model.yaml["fredft_qkv_expand"] == pytest.approx(1.0)
    assert model.yaml["backbone"][6][2] == "A2C2f"
    assert isinstance(model.fusion_convs["p2"], DMGFusionInit8d)
    assert model.fusion_convs["p2"].alpha.item() == pytest.approx(1.0)
    assert model.fusion_convs["p2"].beta.item() == pytest.approx(-0.1)
    for stage_name in ("p3", "p4", "p5"):
        expected_type = FreDFTFusion if stage_name in expected_stages else Conv
        assert isinstance(model.fusion_convs[stage_name], expected_type)
        if stage_name in expected_stages:
            fusion = model.fusion_convs[stage_name]
            channels = fusion.freq_attn.to_hidden.in_channels
            expected_ffn_hidden = int(channels * expected_expansion)
            expected_ffn_hidden += (-expected_ffn_hidden) % 3
            assert fusion.freq_attn.to_hidden.out_channels == channels * 3
            assert fusion.ffn.project_in.out_channels == expected_ffn_hidden
            assert fusion.checkpoint_ffn is bool(model.yaml.get("fredft_checkpoint_ffn", False))
    assert model.use_aux_head is True
    assert "cmg_stages" not in model.yaml
    assert "cma_stages" not in model.yaml
    assert "fredft_scale_init" not in model.yaml


def test_dual_stream_m2d_lif_p4_cfg_extends_confirmed_dmg_init8d_baseline():
    """M2D-LIF config should add illumination-aware fusion at P4 only."""
    from ultralytics.nn.modules.block import DMGFusionInit8d, M2DLocalIlluminationFusion
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual-p2-m2dlif-p4.yaml", nc=3, verbose=False)

    assert model.yaml["p2_fusion"] == "dmg_init8d"
    assert model.yaml["dmg_alpha_init"] == pytest.approx(1.0)
    assert model.yaml["dmg_beta_init"] == pytest.approx(-0.1)
    assert model.yaml["lif_fusion_stages"] == ["p4"]
    assert isinstance(model.fusion_convs["p2"], DMGFusionInit8d)
    assert model.fusion_convs["p2"].alpha.item() == pytest.approx(1.0)
    assert model.fusion_convs["p2"].beta.item() == pytest.approx(-0.1)
    assert isinstance(model.fusion_convs["p3"], Conv)
    assert isinstance(model.fusion_convs["p4"], M2DLocalIlluminationFusion)
    assert isinstance(model.fusion_convs["p5"], Conv)
    assert hasattr(model, "lif_gate")
    assert model.use_aux_head is True
    assert "freq_fusion_stages" not in model.yaml
    assert "cmg_stages" not in model.yaml
    assert "cma_stages" not in model.yaml


def test_dual_stream_parallel_cross_p4_cfg_uses_light_cross_mlp_only():
    """P4-only Parallel Cross config should reduce only the cross-modal MLP ratio to 1.0."""
    from ultralytics.nn.modules.block import A2C2f, DualParallelCrossA2C2f
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel(
        "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-pcross-p4.yaml",
        nc=3,
        verbose=False,
    )

    p4 = model.backbone_rgb[6]
    assert model.yaml["parallel_cross_a2c2f_stages"] == ["p4"]
    assert model.yaml["parallel_cross_mlp_ratio"] == pytest.approx(1.0)
    assert isinstance(p4, DualParallelCrossA2C2f)
    assert p4 is model.backbone_ir[6]
    assert p4.self_rgb[0].mlp[0].conv.out_channels == 128
    assert p4.cross_rgb[0].mlp[0].conv.out_channels == 64
    assert isinstance(model.backbone_rgb[8], A2C2f)


def test_dual_stream_parallel_cross_p4p5_cfg_extends_a1_baseline():
    """Parallel Cross A2C2f P4/P5 config should preserve P2 DMGInit8d and P3 FreAtt."""
    from ultralytics.nn.modules.block import DMGFusionInit8d, DualParallelCrossA2C2f, FreDFTFusion
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel(
        "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-pcross-p4p5.yaml",
        nc=3,
        verbose=False,
    )

    assert model.yaml["p2_fusion"] == "dmg_init8d"
    assert model.yaml["dmg_alpha_init"] == pytest.approx(1.0)
    assert model.yaml["dmg_beta_init"] == pytest.approx(-0.1)
    assert model.yaml["freq_fusion_stages"] == ["p3"]
    assert model.yaml["parallel_cross_a2c2f_stages"] == ["p4", "p5"]
    assert model.yaml["parallel_cross_gamma_init"] == pytest.approx(0.01)
    assert isinstance(model.fusion_convs["p2"], DMGFusionInit8d)
    assert isinstance(model.fusion_convs["p3"], FreDFTFusion)
    assert isinstance(model.fusion_convs["p4"], Conv)
    assert isinstance(model.fusion_convs["p5"], Conv)
    assert isinstance(model.backbone_rgb[6], DualParallelCrossA2C2f)
    assert isinstance(model.backbone_rgb[8], DualParallelCrossA2C2f)
    assert model.backbone_rgb[6] is model.backbone_ir[6]
    assert model.backbone_rgb[8] is model.backbone_ir[8]
    assert model.backbone_rgb[6].cross_rgb[0].attn.area == 4
    assert model.backbone_rgb[8].cross_rgb[0].attn.area == 1
    assert model.use_aux_head is True
    assert "cmg_stages" not in model.yaml
    assert "cma_stages" not in model.yaml


def test_dual_stream_parallel_cross_p4p5_forward_keeps_four_scale_stride():
    """Parallel Cross A2C2f P4/P5 config should forward 6-channel inputs with 4 detection scales."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel(
        "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-pcross-p4p5.yaml",
        nc=3,
        verbose=False,
    )
    model.eval()

    with torch.no_grad():
        out = model(torch.zeros(1, 6, 480, 640))

    assert out is not None
    assert torch.equal(model.stride.sort().values, torch.tensor([4.0, 8.0, 16.0, 32.0]))


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


def test_single_stream_p2_p3_aux_cfg_instantiates_aux_head():
    """Single-stream IR P2 config should attach one training-only P3 aux head."""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolov12-ir-p2-p3aux.yaml", nc=3, verbose=False)

    assert model.yaml["p3_aux"] is True
    assert model.p3_aux_layer == 4
    assert model.use_p3_aux is True
    assert model.aux_head.stride.tolist() == [8.0]


def test_single_stream_p2_p3_aux_forward_shape():
    """Single-stream IR P2 + P3 aux model forwards 480x640 inputs and keeps 4 detection scales."""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolov12-ir-p2-p3aux.yaml", nc=3, verbose=False)
    model.eval()

    with torch.no_grad():
        out = model(torch.zeros(1, 3, 480, 640))

    assert out is not None
    assert torch.equal(model.stride.sort().values, torch.tensor([4.0, 8.0, 16.0, 32.0]))
