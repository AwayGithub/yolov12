# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Smoke tests for the MCDet baseline reproduction."""

import torch
import pytest

from baselines.MCDet.model import MCDetDualStreamDetectionModel, MCDetYOLO
from baselines.MCDet.modules import (
    BidirectionalVisualStateSpaceModule,
    ContentGuidedAttention,
    DCNv2,
    FeatureRefinementModule,
    MultidimensionalRepresentationCollaborativeFusion,
    TwoPropertySpectralFeatureFusion,
)
from baselines.MCDet.losses import WIoUBboxLoss, wise_iou_v1


def test_bvssm_shape_and_backward():
    """BVSSM should preserve feature shape and support gradients."""
    if not torch.cuda.is_available():
        pytest.skip("mamba-ssm selective_scan is CUDA-only.")
    module = BidirectionalVisualStateSpaceModule(16).cuda()
    rgb = torch.randn(2, 16, 10, 12, device="cuda", requires_grad=True)
    ir = torch.randn(2, 16, 10, 12, device="cuda", requires_grad=True)
    out = module(rgb, ir)
    assert out.shape == rgb.shape
    out.mean().backward()
    assert rgb.grad is not None
    assert ir.grad is not None


def test_tsff_uses_channel_pair_group_mixing():
    """TSFF should mix every RGB/IR channel pair as one group."""
    module = TwoPropertySpectralFeatureFusion(16)
    assert isinstance(module.vis_local[0], DCNv2)
    assert isinstance(module.ir_local[0], DCNv2)
    assert isinstance(module.post_mix_dcn[0], DCNv2)
    assert module.group_mix.in_channels == 32
    assert module.group_mix.out_channels == 16
    assert module.group_mix.groups == 16


def test_dcnv2_shape():
    """DCNv2 should keep spatial shape when stride is one."""
    module = DCNv2(8, 12)
    x = torch.randn(2, 8, 11, 13)
    assert module(x).shape == (2, 12, 11, 13)


def test_frm_and_mrcf_preserve_shape():
    """FRM and MRCF should return same-resolution fused features."""
    frm = FeatureRefinementModule(16)
    rgb = torch.randn(2, 16, 10, 12)
    ir = torch.randn(2, 16, 10, 12)
    assert frm(rgb).shape == rgb.shape
    if not torch.cuda.is_available():
        pytest.skip("mamba-ssm selective_scan is CUDA-only.")
    mrcf = MultidimensionalRepresentationCollaborativeFusion(16).cuda()
    rgb = rgb.cuda()
    ir = ir.cuda()
    assert mrcf(rgb, ir).shape == rgb.shape


def test_cgan_preserves_concat_feature_shape():
    """CGAN should refine FPN/PAN concat features without changing channels."""
    module = ContentGuidedAttention(24)
    assert isinstance(module.spatial_dcn, DCNv2)
    x = torch.randn(2, 24, 20, 16)
    assert module(x).shape == x.shape


def test_wiou_loss_is_finite():
    """WIoU bbox loss should be finite for valid positive boxes."""
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    target = torch.tensor([[1.0, 1.0, 9.0, 9.0], [6.0, 6.0, 16.0, 16.0]])
    wiou, liou = wise_iou_v1(pred, target)
    assert torch.isfinite(wiou).all()
    assert torch.isfinite(liou).all()
    loss = WIoUBboxLoss(16)
    assert loss.liou_mean.item() == 1.0


def test_mcdet_yolo_wrapper_builds_dual_stream_model():
    """The baseline wrapper should build MCDet without changing the main YOLO task map."""
    model = MCDetYOLO("baselines/MCDet/cfg/MCDet-yolov5n-fire-person.yaml").model
    assert isinstance(model, MCDetDualStreamDetectionModel)
    assert set(model.mcdet_fusion_stages) == {"p3", "p4", "p5"}
    assert set(model.FUSION_LAYER_INDICES) == {"p3", "p4", "p5"}


def test_mcdet_forward_smoke():
    """A small dual-input forward pass should complete."""
    if not torch.cuda.is_available():
        pytest.skip("MCDet MRCF uses CUDA-only mamba-ssm selective_scan.")
    model = MCDetYOLO("baselines/MCDet/cfg/MCDet-yolov5n-fire-person.yaml").model.cuda()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 6, 256, 256, device="cuda"))
    assert out is not None
