# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Tests for B5/B6 Parallel Cross regularization controls."""

from types import SimpleNamespace

import pytest
import torch

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.block import DualParallelCrossA2C2f
from ultralytics.nn.tasks import DualStreamDetectionModel


B5_CFG = "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-pcross-p4-reg.yaml"
B6_CFG = "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-pcross-p4-posgamma.yaml"
B7_CFG = "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-pcross-p4-self4-cross2.yaml"
B8_CFG = "yolov12-dual-p2-dmg-init8d-p3aux-fredft-p3-pcross-p4-self4-cross4.yaml"



@pytest.mark.parametrize("cfg,self_depth,cross_depth", [(B7_CFG, 4, 2), (B8_CFG, 4, 4)])
def test_parallel_cross_configs_use_independent_branch_depths(cfg, self_depth, cross_depth):
    """B7 and B8 vary only the intended P4 self and cross branch depths."""
    model = DualStreamDetectionModel(cfg, nc=3, verbose=False)
    layer = model.backbone_rgb[6]

    assert len(layer.self_rgb) == len(layer.self_ir) == self_depth
    assert len(layer.cross_rgb) == len(layer.cross_ir) == cross_depth
    assert layer.cross_rgb[0].mlp[0].conv.out_channels == 128


def test_shared_cross_drop_path_and_fixed_scale():
    """B5 DropPath shares a per-sample mask across directions and keeps scales fixed."""
    module = DualParallelCrossA2C2f(128, 128, cross_drop_path=0.5, learnable_cross_scale=False)
    module.train()
    torch.manual_seed(0)
    rgb, ir = module._drop_cross_delta(torch.ones(32, 2, 1, 1), torch.full((32, 2, 1, 1), 2.0))

    assert torch.equal(ir, rgb * 2)
    assert (rgb == 0).any() and (rgb != 0).any()
    assert module.cross_scale_rgb.requires_grad is False
    assert module.cross_scale_ir.requires_grad is False


def test_bounded_positive_gamma_initialization_and_range():
    """B6 gamma starts at 0.01 and remains positive and bounded."""
    module = DualParallelCrossA2C2f(128, 128, gamma_mode="bounded_positive", gamma_max=0.35, scale_init=0.01)

    assert module.effective_gamma_rgb().item() == pytest.approx(0.01, abs=1e-6)
    with torch.no_grad():
        module.gamma_rgb_logit.fill_(100.0)
        module.gamma_ir_logit.fill_(-100.0)
    assert 0.0 < module.effective_gamma_rgb().item() <= 0.35
    assert 0.0 <= module.effective_gamma_ir().item() < 0.35


@pytest.mark.parametrize("cfg,gamma_mode,drop_path", [(B5_CFG, "free", 0.05), (B6_CFG, "bounded_positive", 0.0)])
def test_b5_b6_configs_preserve_b2_cross_capacity(cfg, gamma_mode, drop_path):
    """B5/B6 independently preserve B2 cross MLP capacity and apply their intended controls."""
    model = DualStreamDetectionModel(cfg, nc=3, verbose=False)
    layer = model.backbone_rgb[6]

    assert layer.cross_rgb[0].mlp[0].conv.out_channels == 128
    assert layer.gamma_mode == gamma_mode
    assert layer.cross_drop_path == pytest.approx(drop_path)
    assert layer.cross_scale_rgb.item() == pytest.approx(1.0)
    assert layer.cross_scale_rgb.requires_grad is False


def test_b5_optimizer_applies_cross_and_gamma_lr_multiplier():
    """B5 cross branch and gamma parameters use one tenth of the base learning rate."""
    model = DualStreamDetectionModel(B5_CFG, nc=3, verbose=False)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(lr0=0.01, momentum=0.937, warmup_bias_lr=0.0)
    optimizer = trainer.build_optimizer(model, name="SGD", lr=0.01, momentum=0.937, decay=5e-4)
    param_groups = {id(param): group for group in optimizer.param_groups for param in group["params"]}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert id(param) not in param_groups
            continue
        is_reduced_lr = any(key in name for key in (".cross_rgb.", ".cross_ir.", "gamma_rgb", "gamma_ir"))
        assert param_groups[id(param)]["lr_mult"] == pytest.approx(0.1 if is_reduced_lr else 1.0)
