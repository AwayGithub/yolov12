# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Tests for ADR-004 Pilot Study P01 physical statistics."""

import numpy as np

from dev.adr004_p01_physical_profile import (
    auc_and_cliff,
    global_ssim,
    make_ring_mask,
    overlap_fraction,
    size_bin,
    spectral_metrics,
    yolo_box_to_xyxy,
)


def test_yolo_box_and_size_bins():
    """YOLO boxes should be clipped and assigned by their largest side."""
    assert yolo_box_to_xyxy((0.5, 0.5, 0.5, 0.5), 640, 480) == (160, 120, 480, 360)
    assert yolo_box_to_xyxy((0.0, 0.0, 0.2, 0.2), 640, 480) == (0, 0, 64, 48)
    assert size_bin(31, 20) == "<32"
    assert size_bin(32, 20) == "32-64"
    assert size_bin(100, 128) == "128-256"


def test_ring_mask_excludes_all_gt_boxes():
    """The paired background ring must exclude both current and neighboring GT boxes."""
    current = (4, 4, 8, 8)
    neighbor = (8, 5, 10, 7)
    mask = make_ring_mask(current, [current, neighbor], (12, 12), scale=2.0)
    assert mask.sum() > 0
    assert not mask[4:8, 4:8].any()
    assert not mask[5:7, 8:10].any()


def test_overlap_fraction_uses_union_area():
    """Overlapping neighboring boxes should not be double counted."""
    box = (0, 0, 10, 10)
    assert overlap_fraction(box, [(0, 0, 5, 10)]) == 0.5
    assert overlap_fraction(box, [(0, 0, 5, 10), (0, 0, 5, 10)]) == 0.5


def test_auc_cliff_direction_and_ties():
    """AUC should be orientation-free while Cliff's delta retains direction."""
    auc, cliff, direction = auc_and_cliff(np.array([3.0, 4.0]), np.array([1.0, 2.0]))
    assert auc == 1.0
    assert cliff == 1.0
    assert direction == "higher"
    auc, cliff, direction = auc_and_cliff(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert auc == 1.0
    assert cliff == -1.0
    assert direction == "lower"
    auc, cliff, _ = auc_and_cliff(np.ones(3), np.ones(4))
    assert auc == 0.5
    assert cliff == 0.0


def test_cross_modal_and_spectral_metrics_are_finite():
    """Identical structured patches should have SSIM one and valid spectral descriptors."""
    patch = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
    assert np.isclose(global_ssim(patch, patch), 1.0)
    high_low, slope = spectral_metrics(patch)
    assert np.isfinite(high_low)
    assert np.isfinite(slope)
