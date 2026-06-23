# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Tests for ADR-004 P02 smoke-failure physical analysis."""

import numpy as np
import pandas as pd

from dev.adr004_p02_smoke_failure_physics import (
    average_precision,
    benjamini_hochberg,
    box_iou,
    fit_ridge_logistic,
    match_cases_controls,
    roc_auc,
)


def test_box_iou_and_rank_metrics():
    """Box matching and rank metrics should handle exact matches and ties."""
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.0, 0.0, 1.0, 1.0])) == 1.0
    assert average_precision(labels, np.array([0.0, 0.0, 1.0, 1.0])) == 1.0
    assert roc_auc(labels, np.ones(4)) == 0.5


def test_benjamini_hochberg_is_monotone_in_rank():
    """Adjusted p-values should remain ordered for ordered raw p-values."""
    adjusted = benjamini_hochberg([0.001, 0.01, 0.2, np.nan])
    assert np.all(np.diff(adjusted[:3]) >= 0)
    assert np.isnan(adjusted[3])


def test_matching_keeps_exact_strata_and_unique_controls():
    """Primary matching should preserve strata and avoid control reuse."""
    rows = []
    for index, outcome in enumerate(("a1_hit_b2_miss", "both_hit", "both_hit", "both_hit")):
        rows.append({
            "outcome": outcome,
            "image_id": f"image{index}",
            "gt_index": 0,
            "video": "video1",
            "size_bin": "32-64",
            "fire_present": True,
            "person_present": False,
            "log_sqrt_area": 3.5 + index * 0.01,
            "overlap_fraction": 0.1,
            "ring_to_box_ratio": 1.0,
        })
    matched = match_cases_controls(pd.DataFrame(rows), controls_per_case=3, seed=0)
    assert matched["failure"].sum() == 1
    assert len(matched) == 4
    assert matched.loc[matched["failure"] == 0, "image_id"].nunique() == 3
    assert matched["video"].nunique() == matched["size_bin"].nunique() == 1


def test_ridge_logistic_separates_simple_data():
    """The local SciPy logistic implementation should learn a simple direction."""
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1, 1])
    parameters = fit_ridge_logistic(features, labels, penalty=0.01)
    scores = parameters[0] + features @ parameters[1:]
    assert parameters[1] > 0
    assert roc_auc(labels, scores) == 1.0
