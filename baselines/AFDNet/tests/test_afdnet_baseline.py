# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Smoke tests for the AFDNet baseline reproduction."""

import torch

from baselines.AFDNet.model import AFDNetDualStreamDetectionModel, AFDNetYOLO
from baselines.AFDNet.modules import (
    AsymmetricFrequencyDecoupledFusion,
    HaarWavelet2d,
    SmokeMaskedHighFrequencyRestoration,
    ThermalGuidedLowFrequencyAggregation,
)


def test_haar_wavelet_round_trip_odd_shape():
    """Haar DWT/IDWT should preserve the original tensor shape and values."""
    wavelet = HaarWavelet2d()
    x = torch.randn(2, 8, 15, 17)
    ll, lh, hl, hh, shape = wavelet.dwt(x)
    y = wavelet.idwt(ll, lh, hl, hh, shape)
    assert y.shape == x.shape
    assert torch.allclose(x, y, atol=1e-6)


def test_haar_wavelet_matches_slice_formula():
    """Fixed convolution Haar filters should match the original slice implementation."""
    wavelet = HaarWavelet2d()
    x = torch.randn(2, 4, 16, 18)
    ll, lh, hl, hh, _ = wavelet.dwt(x)

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    assert torch.allclose(ll, (x00 + x01 + x10 + x11) * 0.5, atol=1e-6)
    assert torch.allclose(lh, (x00 - x01 + x10 - x11) * 0.5, atol=1e-6)
    assert torch.allclose(hl, (x00 + x01 - x10 - x11) * 0.5, atol=1e-6)
    assert torch.allclose(hh, (x00 - x01 - x10 + x11) * 0.5, atol=1e-6)


def test_afdnet_fusion_backward():
    """AFDNet fusion should support backpropagation."""
    fusion = AsymmetricFrequencyDecoupledFusion(16)
    rgb = torch.randn(2, 16, 15, 17, requires_grad=True)
    ir = torch.randn(2, 16, 15, 17, requires_grad=True)
    out = fusion(rgb, ir)
    assert out.shape == rgb.shape
    out.mean().backward()
    assert rgb.grad is not None
    assert ir.grad is not None


def test_tla_matches_paper_branches():
    """TLA should use IR spatial masking and variance-based alpha weighting."""
    tla = ThermalGuidedLowFrequencyAggregation(16)
    assert tla.ir_spatial[0].in_channels == 2
    assert tla.ir_spatial[0].kernel_size == (5, 5)
    assert tla.alpha_fc[0].in_features == 16
    assert tla.alpha_fc[0].out_features == 16


def test_shr_uses_three_visible_subbands_and_dilated_branches():
    """SHR should generate the mask from concatenated VIS LH/HL/HH features."""
    shr = SmokeMaskedHighFrequencyRestoration(16)
    assert shr.conv_d1.in_channels == 48
    assert shr.conv_d1.out_channels == 16
    assert shr.conv_d1.groups == 16
    assert shr.conv_d1.dilation == (1, 1)
    assert shr.conv_d3.in_channels == 48
    assert shr.conv_d3.out_channels == 16
    assert shr.conv_d3.groups == 16
    assert shr.conv_d3.dilation == (3, 3)
    assert shr.conv_d5.in_channels == 48
    assert shr.conv_d5.out_channels == 16
    assert shr.conv_d5.groups == 16
    assert shr.conv_d5.dilation == (5, 5)
    assert shr.mask[0].in_channels == 48
    assert shr.mask[0].out_channels == 16


def test_afdnet_yolo_wrapper_builds_dual_stream_model():
    """The baseline wrapper should build AFDNet without changing the main YOLO task map."""
    model = AFDNetYOLO("baselines/AFDNet/cfg/AFDNet-y11n-fire-person.yaml").model
    assert isinstance(model, AFDNetDualStreamDetectionModel)
    assert set(model.afd_fusion_stages) == {"p3", "p4", "p5"}
    assert "p2" not in model.fusion_convs
