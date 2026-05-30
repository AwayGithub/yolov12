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


def test_linear_area_attention_preserves_area_shape_and_gradients():
    """LinearAAttn keeps AAttn's BCHW contract and supports area partitioning."""
    from ultralytics.nn.modules.block import LinearAAttn

    m = LinearAAttn(dim=64, num_heads=2, area=4)
    x = torch.randn(2, 64, 12, 16, requires_grad=True)

    y = m(x)
    y.mean().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert m.qk.conv.weight.grad is not None
    assert m.v.conv.weight.grad is not None
    assert m.proj.conv.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fp16 is required to reproduce AMP overflow")
def test_elu_linear_attention_uses_stable_fp32_accumulation_for_fp16_inputs():
    """ELU+1 linear attention should not overflow its KV accumulation under AMP-like fp16 inputs."""
    from ultralytics.nn.modules.block import _elu_linear_attention

    q = torch.full((1, 1, 1200, 32), 20.0, device="cuda", dtype=torch.float16)
    k = torch.full((1, 1, 1200, 32), 20.0, device="cuda", dtype=torch.float16)
    v = torch.full((1, 1, 1200, 32), 20.0, device="cuda", dtype=torch.float16)

    y = _elu_linear_attention(q, k, v)

    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()


def test_cross_linear_area_attention_preserves_query_shape_and_uses_other_modality():
    """CrossLinearAAttn should use Q from target and K/V from guide while preserving target shape."""
    from ultralytics.nn.modules.block import CrossLinearAAttn

    m = CrossLinearAAttn(dim=64, num_heads=2, area=4)
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


def test_dual_linear_cross_a2c2f_runs_self_then_cross_blocks():
    """DualLinearCrossA2C2f replaces each 2xABlock pair with self-attn then cross-attn."""
    from ultralytics.nn.modules.block import CrossLinearABlock, DualLinearCrossA2C2f, LinearABlock

    m = DualLinearCrossA2C2f(c1=64, c2=64, n=2, area=4, scale_init=0.01)
    x_rgb = torch.randn(2, 64, 12, 16, requires_grad=True)
    x_ir = torch.randn(2, 64, 12, 16, requires_grad=True)

    y_rgb, y_ir = m(x_rgb, x_ir)
    (y_rgb.mean() + y_ir.mean()).backward()

    assert y_rgb.shape == x_rgb.shape
    assert y_ir.shape == x_ir.shape
    assert len(m.m) == 2
    assert all(isinstance(pair["self_rgb"], LinearABlock) for pair in m.m)
    assert all(isinstance(pair["self_ir"], LinearABlock) for pair in m.m)
    assert all(isinstance(pair["cross_rgb"], CrossLinearABlock) for pair in m.m)
    assert all(isinstance(pair["cross_ir"], CrossLinearABlock) for pair in m.m)
    assert all(pair["cross_rgb"].gamma.item() == pytest.approx(0.01, abs=1e-6) for pair in m.m)
    assert all(pair["cross_ir"].gamma.item() == pytest.approx(0.01, abs=1e-6) for pair in m.m)
    assert x_rgb.grad is not None
    assert x_ir.grad is not None


def test_dual_linear_cross_a2c2f_keeps_self_and_cross_channel_widths_aligned():
    """The second cross-attention block must keep the first self-attention block's QKV and MLP widths."""
    from ultralytics.nn.modules.block import DualLinearCrossA2C2f

    m = DualLinearCrossA2C2f(c1=64, c2=128, n=1, area=4, mlp_ratio=1.5, e=0.5)
    pair = m.m[0]
    self_block = pair["self_rgb"]
    cross_block = pair["cross_rgb"]

    assert self_block.attn.head_dim == cross_block.attn.head_dim
    assert self_block.attn.num_heads == cross_block.attn.num_heads
    assert self_block.attn.qk.conv.out_channels == cross_block.attn.kv.conv.out_channels
    assert self_block.attn.v.conv.out_channels == cross_block.attn.q.conv.out_channels
    assert self_block.attn.proj.conv.in_channels == cross_block.attn.proj.conv.in_channels
    assert self_block.attn.proj.conv.out_channels == cross_block.attn.proj.conv.out_channels
    assert self_block.mlp[0].conv.in_channels == cross_block.mlp[0].conv.in_channels
    assert self_block.mlp[0].conv.out_channels == cross_block.mlp[0].conv.out_channels
    assert self_block.mlp[1].conv.in_channels == cross_block.mlp[1].conv.in_channels
    assert self_block.mlp[1].conv.out_channels == cross_block.mlp[1].conv.out_channels


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


def test_dual_stream_linear_area_cross_p4p5_cfg_extends_a1_baseline():
    """Linear Area Cross P4/P5 config should preserve P2 DMGInit8d and P3 FreAtt."""
    from ultralytics.nn.modules.block import DMGFusionInit8d, DualLinearCrossA2C2f, FreDFTFusion
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel(
        "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-lineararea-p4p5.yaml",
        nc=3,
        verbose=False,
    )

    assert model.yaml["p2_fusion"] == "dmg_init8d"
    assert model.yaml["dmg_alpha_init"] == pytest.approx(1.0)
    assert model.yaml["dmg_beta_init"] == pytest.approx(-0.1)
    assert model.yaml["freq_fusion_stages"] == ["p3"]
    assert model.yaml["area_linear_cross_stages"] == ["p4", "p5"]
    assert model.yaml["area_linear_cross_scale_init"] == pytest.approx(0.01)
    assert isinstance(model.fusion_convs["p2"], DMGFusionInit8d)
    assert isinstance(model.fusion_convs["p3"], FreDFTFusion)
    assert isinstance(model.fusion_convs["p4"], Conv)
    assert isinstance(model.fusion_convs["p5"], Conv)
    assert isinstance(model.backbone_rgb[6], DualLinearCrossA2C2f)
    assert isinstance(model.backbone_rgb[8], DualLinearCrossA2C2f)
    assert model.backbone_rgb[6] is model.backbone_ir[6]
    assert model.backbone_rgb[8] is model.backbone_ir[8]
    assert model.backbone_rgb[6].m[0]["self_rgb"].attn.area == 4
    assert model.backbone_rgb[8].m[0]["self_rgb"].attn.area == 1
    assert model.use_aux_head is True
    assert "cmg_stages" not in model.yaml
    assert "cma_stages" not in model.yaml


def test_dual_stream_linear_area_cross_p4p5_forward_keeps_four_scale_stride():
    """Linear Area Cross P4/P5 config should forward 6-channel inputs with 4 detection scales."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel(
        "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-lineararea-p4p5.yaml",
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
