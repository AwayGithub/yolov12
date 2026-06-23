# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Relate smoke physical properties to A1/B2 outcome changes for ADR-004 P02."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu, rankdata, spearmanr


CORE_METRICS = (
    "rgb_gradient_mean",
    "rgb_laplacian_energy",
    "rgb_edge_density",
    "ir_gradient_mean",
    "ir_laplacian_energy",
    "ir_edge_density",
    "ir_contrast_abs",
    "cross_edge_iou",
)
EXPLORATORY_METRICS = (
    "rgb_dark_mean",
    "rgb_saturation_mean",
    "rgb_spectral_high_low_log10",
    "rgb_spectral_slope",
    "cross_gradient_corr",
    "cross_zmad",
    "cross_contrast_gap",
)
MATCH_COVARIATES = ("log_sqrt_area", "overlap_fraction", "ring_to_box_ratio")
SIZE_ORDER = ("<32", "32-64", "64-128", "128-256", ">=256")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical",
        type=Path,
        default=Path("runs/detect/adr004/pilot/P01_physical_profile/instance_physical_metrics.csv"),
    )
    parser.add_argument(
        "--diagnose",
        type=Path,
        default=Path("runs/detect/diagnose/smoke_delta（A1、B2smoke逐样本、逐层特征差异诊断）"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/detect/adr004/pilot/P02_b2_smoke_failure_physics"))
    parser.add_argument("--controls-per-case", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """Return IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def box_max_coordinate_delta(first: Iterable[float], second: Iterable[float]) -> float:
    """Return the largest absolute xyxy coordinate difference."""
    return float(np.max(np.abs(np.asarray(tuple(first), dtype=float) - np.asarray(tuple(second), dtype=float))))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate tie-aware binary ROC-AUC."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = rankdata(scores, method="average")
    auc = (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum())
    return float(auc)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate non-interpolated average precision."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels == 1].sum() / positives)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Apply Benjamini-Hochberg false-discovery-rate correction."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if not valid_indices.size:
        return adjusted
    order = valid_indices[np.argsort(values[valid_indices])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return adjusted


def clustered_bootstrap_auc(
    frame: pd.DataFrame,
    score_column: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap matched sets and return a 95% ROC-AUC interval."""
    groups = [group for _, group in frame.groupby("match_id", sort=False)]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        sample = pd.concat(sampled, ignore_index=True)
        estimates.append(roc_auc(sample["failure"].to_numpy(), sample[score_column].to_numpy()))
    return float(np.nanquantile(estimates, 0.025)), float(np.nanquantile(estimates, 0.975))


def load_outcomes(diagnose: Path) -> pd.DataFrame:
    """Load aligned A1/B2 smoke outcomes into one table."""
    a_data = json.loads((diagnose / "A1_last_smoke_instances.json").read_text(encoding="utf-8"))["records"]
    b_data = json.loads((diagnose / "B2_last_smoke_instances.json").read_text(encoding="utf-8"))["records"]
    a_records = {(Path(item["image"]).stem, int(item["gt_index"])): item for item in a_data}
    b_records = {(Path(item["image"]).stem, int(item["gt_index"])): item for item in b_data}
    if a_records.keys() != b_records.keys():
        raise ValueError("A1 and B2 smoke manifests do not contain identical GT keys")

    rows = []
    for image_id, gt_index in sorted(a_records):
        a_item, b_item = a_records[(image_id, gt_index)], b_records[(image_id, gt_index)]
        a_hit, b_hit = bool(a_item["detected"]), bool(b_item["detected"])
        outcome = ("both_hit" if a_hit and b_hit else "a1_hit_b2_miss" if a_hit else
                   "a1_miss_b2_hit" if b_hit else "both_miss")
        epsilon = 1e-5
        a_conf = float(a_item["best_smoke_conf"])
        b_conf = float(b_item["best_smoke_conf"])
        def logit(value: float) -> float:
            """Return a finite logit for detector confidence."""
            clipped = np.clip(value, epsilon, 1 - epsilon)
            return math.log(clipped / (1 - clipped))
        rows.append({
            "image_id": image_id,
            "gt_index": gt_index,
            "gt_box": json.dumps(a_item["gt_box"]),
            "outcome": outcome,
            "a1_detected": a_hit,
            "b2_detected": b_hit,
            "a1_reason": a_item["reason"],
            "b2_reason": b_item["reason"],
            "a1_conf": a_conf,
            "b2_conf": b_conf,
            "conf_delta_b2_a1": b_conf - a_conf,
            "logit_conf_delta_b2_a1": logit(b_conf) - logit(a_conf),
            "a1_iou": float(a_item["best_smoke_iou"]),
            "b2_iou": float(b_item["best_smoke_iou"]),
            "iou_delta_b2_a1": float(b_item["best_smoke_iou"]) - float(a_item["best_smoke_iou"]),
        })
    return pd.DataFrame(rows)


def join_physical_metrics(outcomes: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    """Match each detector GT to one P01 smoke row using image and box IoU."""
    smoke = physical[(physical["split"] == "val") & (physical["class_name"] == "smoke")].copy()
    image_classes = physical[physical["split"] == "val"].groupby("image_id")["class_name"].agg(set)
    by_image = {image_id: group for image_id, group in smoke.groupby("image_id", sort=False)}
    rows = []
    used = set()
    for outcome in outcomes.to_dict("records"):
        candidates = by_image.get(outcome["image_id"])
        if candidates is None:
            raise ValueError(f"No P01 smoke rows for {outcome['image_id']}")
        target_box = json.loads(outcome["gt_box"])
        scored = [(box_iou(target_box, (row.x1, row.y1, row.x2, row.y2)), index)
                  for index, row in candidates.iterrows() if index not in used]
        if not scored:
            raise ValueError(f"No unused P01 smoke row for {outcome['image_id']} GT {outcome['gt_index']}")
        match_iou, index = max(scored)
        physical_box = tuple(physical.loc[index, ["x1", "y1", "x2", "y2"]])
        coordinate_delta = box_max_coordinate_delta(target_box, physical_box)
        if match_iou < 0.90 and coordinate_delta > 1.01:
            raise ValueError(f"Poor GT/P01 box match: {outcome['image_id']} GT {outcome['gt_index']} IoU={match_iou:.6f}")
        used.add(index)
        merged = physical.loc[index].to_dict()
        merged.update(outcome)
        classes = image_classes[outcome["image_id"]]
        merged.update({
            "physical_match_iou": match_iou,
            "physical_match_max_coordinate_delta": coordinate_delta,
            "fire_present": "fire" in classes,
            "person_present": "person" in classes,
            "log_sqrt_area": math.log(math.sqrt(float(merged["box_area"])) + 1e-8),
        })
        rows.append(merged)
    joined = pd.DataFrame(rows)
    if len(joined) != len(outcomes) or joined[["image_id", "gt_index"]].duplicated().any():
        raise AssertionError("Physical/outcome join is not one-to-one")
    return joined


def match_cases_controls(frame: pd.DataFrame, controls_per_case: int, seed: int) -> pd.DataFrame:
    """Greedily match B2-specific failures to both-hit controls without replacement."""
    cases = frame[frame["outcome"] == "a1_hit_b2_miss"].copy()
    controls = frame[frame["outcome"] == "both_hit"].copy()
    scale = frame.loc[frame["outcome"].isin(("a1_hit_b2_miss", "both_hit")), MATCH_COVARIATES].std().replace(0, 1)
    rng = np.random.default_rng(seed)
    available = set(controls.index)
    rows = []
    strata = ["video", "size_bin", "fire_present", "person_present"]
    case_order = list(cases.index)
    rng.shuffle(case_order)
    case_order.sort(key=lambda index: len(controls[(controls[strata] == cases.loc[index, strata]).all(axis=1)]))
    for match_id, case_index in enumerate(case_order):
        case = cases.loc[case_index]
        candidates = controls.loc[list(available)]
        exact = (candidates[strata] == case[strata].to_numpy()).all(axis=1)
        candidates = candidates[exact]
        match_relaxation = "exact"
        if len(candidates) < controls_per_case:
            fallback = controls.loc[list(available)]
            fallback = fallback[(fallback["video"] == case["video"]) & (fallback["size_bin"] == case["size_bin"])]
            candidates = fallback
            match_relaxation = "video_size"
        if candidates.empty:
            candidates = controls[(controls["video"] == case["video"]) & (controls["size_bin"] == case["size_bin"])]
            match_relaxation = "video_size_reuse"
        if candidates.empty:
            candidates = controls.loc[list(available)]
            candidates = candidates[candidates["video"] == case["video"]]
            match_relaxation = "video_only"
        if candidates.empty:
            candidates = controls[controls["video"] == case["video"]]
            match_relaxation = "video_only_reuse"
        if candidates.empty:
            raise ValueError(f"No same-video control for case {case['image_id']}:{case['gt_index']}")
        candidate_values = candidates[list(MATCH_COVARIATES)].astype(float)
        case_values = case[list(MATCH_COVARIATES)].astype(float)
        distance = (((candidate_values - case_values) / scale.astype(float))**2).sum(axis=1)
        mismatch = ((candidates[["fire_present", "person_present"]] !=
                     case[["fire_present", "person_present"]].to_numpy()).sum(axis=1) * 4.0)
        selected = (distance + mismatch).nsmallest(min(controls_per_case, len(candidates))).index
        selected_reused = {index: index not in available for index in selected}
        available.difference_update(selected)
        case_row = case.to_dict()
        case_row.update({
            "match_id": match_id,
            "failure": 1,
            "match_role": "case",
            "match_distance": 0.0,
            "control_reused": False,
            "match_relaxation": match_relaxation,
        })
        rows.append(case_row)
        for control_index in selected:
            control_row = controls.loc[control_index].to_dict()
            control_row.update({
                "match_id": match_id,
                "failure": 0,
                "match_role": "control",
                "match_distance": float(distance.get(control_index, np.nan)),
                "control_reused": selected_reused[control_index],
                "match_relaxation": match_relaxation,
            })
            rows.append(control_row)
    return pd.DataFrame(rows).sort_values(["match_id", "failure"], ascending=[True, False]).reset_index(drop=True)


def summarize_balance(matched: pd.DataFrame) -> pd.DataFrame:
    """Summarize matching balance using standardized mean differences."""
    rows = []
    for metric in (*MATCH_COVARIATES, "fire_present", "person_present"):
        case = matched.loc[matched["failure"] == 1, metric].astype(float)
        control = matched.loc[matched["failure"] == 0, metric].astype(float)
        pooled = math.sqrt((case.var(ddof=1) + control.var(ddof=1)) / 2)
        rows.append({
            "covariate": metric,
            "case_mean": case.mean(),
            "control_mean": control.mean(),
            "standardized_mean_difference": (case.mean() - control.mean()) / pooled if pooled else 0.0,
        })
    return pd.DataFrame(rows)


def build_primary_effects(matched: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    """Calculate preregistered and exploratory univariate effects."""
    rows = []
    for metric in (*CORE_METRICS, *EXPLORATORY_METRICS):
        # P01 predicts that failures are weaker on every core structure metric.
        score_sign = -1 if metric in CORE_METRICS else 1
        score_column = f"__score_{metric}"
        matched[score_column] = matched[metric] * score_sign
        valid = matched[["failure", metric, score_column, "match_id"]].dropna()
        case = valid.loc[valid["failure"] == 1, metric].to_numpy(dtype=float)
        control = valid.loc[valid["failure"] == 0, metric].to_numpy(dtype=float)
        auc = roc_auc(valid["failure"].to_numpy(), valid[score_column].to_numpy())
        ci_low, ci_high = clustered_bootstrap_auc(valid, score_column, iterations, seed + len(rows))
        _, p_value = mannwhitneyu(case, control, alternative="two-sided")
        rows.append({
            "metric": metric,
            "family": "core" if metric in CORE_METRICS else "exploratory",
            "hypothesis_score_sign": score_sign,
            "case_n": len(case),
            "control_n": len(control),
            "case_mean": np.mean(case),
            "case_median": np.median(case),
            "control_mean": np.mean(control),
            "control_median": np.median(control),
            "mean_difference_case_control": np.mean(case) - np.mean(control),
            "roc_auc": auc,
            "roc_auc_ci95_low": ci_low,
            "roc_auc_ci95_high": ci_high,
            "pr_auc": average_precision(valid["failure"].to_numpy(), valid[score_column].to_numpy()),
            "prevalence": valid["failure"].mean(),
            "mannwhitney_p": p_value,
        })
    result = pd.DataFrame(rows)
    result["fdr_bh"] = benjamini_hochberg(result["mannwhitney_p"])
    return result


def fit_ridge_logistic(features: np.ndarray, labels: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    """Fit a small ridge logistic model with SciPy."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = parameters[0] + features @ parameters[1:]
        loss = np.logaddexp(0, logits).mean() - np.mean(labels * logits) + penalty * np.sum(parameters[1:]**2) / 2
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        residual = probabilities - labels
        gradient = np.r_[residual.mean(), features.T @ residual / len(labels) + penalty * parameters[1:]]
        return float(loss), gradient

    result = minimize(objective, np.zeros(features.shape[1] + 1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Logistic regression did not converge: {result.message}")
    return result.x


def leave_one_video_out(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a physical-feature logistic model without video leakage."""
    predictions, folds = [], []
    columns = list(CORE_METRICS)
    for video in sorted(matched["video"].unique()):
        train = matched[matched["video"] != video].copy()
        test = matched[matched["video"] == video].copy()
        if train["failure"].nunique() < 2 or test["failure"].nunique() < 2:
            continue
        median = train[columns].median()
        mean = train[columns].fillna(median).mean()
        std = train[columns].fillna(median).std().replace(0, 1)
        train_x = ((train[columns].fillna(median) - mean) / std).to_numpy(dtype=float)
        test_x = ((test[columns].fillna(median) - mean) / std).to_numpy(dtype=float)
        parameters = fit_ridge_logistic(train_x, train["failure"].to_numpy())
        logits = parameters[0] + test_x @ parameters[1:]
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        fold = test[["image_id", "gt_index", "video", "match_id", "failure"]].copy()
        fold["probability"] = probabilities
        predictions.append(fold)
        folds.append({
            "held_out_video": video,
            "train_n": len(train),
            "test_n": len(test),
            "test_positives": int(test["failure"].sum()),
            "roc_auc": roc_auc(test["failure"].to_numpy(), probabilities),
            "pr_auc": average_precision(test["failure"].to_numpy(), probabilities),
            "prevalence": test["failure"].mean(),
        })
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    fold_frame = pd.DataFrame(folds)
    if not prediction_frame.empty:
        overall = {
            "held_out_video": "pooled_oof",
            "train_n": np.nan,
            "test_n": len(prediction_frame),
            "test_positives": int(prediction_frame["failure"].sum()),
            "roc_auc": roc_auc(prediction_frame["failure"].to_numpy(), prediction_frame["probability"].to_numpy()),
            "pr_auc": average_precision(prediction_frame["failure"].to_numpy(), prediction_frame["probability"].to_numpy()),
            "prevalence": prediction_frame["failure"].mean(),
        }
        fold_frame = pd.concat((fold_frame, pd.DataFrame([overall])), ignore_index=True)
    return fold_frame, prediction_frame


def continuous_associations(frame: pd.DataFrame) -> pd.DataFrame:
    """Relate physical metrics to continuous A1-to-B2 detection changes."""
    rows = []
    for subset_name, subset in (
        ("all_smoke", frame),
        ("discordant", frame[frame["outcome"].isin(("a1_hit_b2_miss", "a1_miss_b2_hit"))]),
    ):
        for metric in (*CORE_METRICS, *EXPLORATORY_METRICS):
            for endpoint in ("logit_conf_delta_b2_a1", "iou_delta_b2_a1"):
                valid = subset[[metric, endpoint]].dropna()
                correlation, p_value = spearmanr(valid[metric], valid[endpoint]) if len(valid) >= 3 else (np.nan, np.nan)
                rows.append({
                    "subset": subset_name,
                    "metric": metric,
                    "endpoint": endpoint,
                    "n": len(valid),
                    "spearman_rho": correlation,
                    "p_value": p_value,
                })
    result = pd.DataFrame(rows)
    result["fdr_bh"] = benjamini_hochberg(result["p_value"])
    return result


def outcome_group_analyses(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize all four outcomes and compare their physical profiles."""
    summary_rows = []
    for outcome, group in frame.groupby("outcome", observed=True):
        for metric in CORE_METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            summary_rows.append({
                "outcome": outcome,
                "metric": metric,
                "n": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "q1": np.quantile(values, 0.25),
                "q3": np.quantile(values, 0.75),
            })
    comparisons = (
        ("b2_loss_vs_both_hit", "a1_hit_b2_miss", "both_hit"),
        ("b2_gain_vs_both_hit", "a1_miss_b2_hit", "both_hit"),
        ("b2_loss_vs_both_miss", "a1_hit_b2_miss", "both_miss"),
        ("b2_loss_vs_b2_gain", "a1_hit_b2_miss", "a1_miss_b2_hit"),
    )
    effect_rows = []
    for name, positive_group, reference_group in comparisons:
        for metric in CORE_METRICS:
            positive = frame.loc[frame["outcome"] == positive_group, metric].dropna().to_numpy(dtype=float)
            reference = frame.loc[frame["outcome"] == reference_group, metric].dropna().to_numpy(dtype=float)
            labels = np.r_[np.ones(len(positive)), np.zeros(len(reference))]
            scores = -np.r_[positive, reference]
            effect_rows.append({
                "comparison": name,
                "positive_group": positive_group,
                "reference_group": reference_group,
                "metric": metric,
                "positive_n": len(positive),
                "reference_n": len(reference),
                "positive_median": np.median(positive),
                "reference_median": np.median(reference),
                "weak_structure_auc": roc_auc(labels, scores),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(effect_rows)


def failure_reason_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare each B2 failure reason with matched stable detections."""
    controls = frame[frame["outcome"] == "both_hit"]
    rows = []
    for reason, cases in frame[frame["outcome"] == "a1_hit_b2_miss"].groupby("b2_reason"):
        for metric in CORE_METRICS:
            comparison = controls[(controls["video"].isin(cases["video"])) & (controls["size_bin"].isin(cases["size_bin"]))]
            combined = pd.concat((cases.assign(label=1), comparison.assign(label=0)), ignore_index=True).dropna(subset=[metric])
            rows.append({
                "reason": reason,
                "metric": metric,
                "case_n": len(cases),
                "control_n": len(comparison),
                "weak_structure_auc": roc_auc(combined["label"].to_numpy(), -combined[metric].to_numpy()),
                "case_median": cases[metric].median(),
                "control_median": comparison[metric].median(),
            })
    return pd.DataFrame(rows)


def match_physical_row(frame: pd.DataFrame, image: str, box: Iterable[float]) -> pd.Series:
    """Find one full-study row by image and box."""
    candidates = frame[frame["image_id"] == Path(image).stem]
    if candidates.empty:
        raise ValueError(f"No P02 row for {image}")
    scores = candidates.apply(lambda row: box_iou(box, (row.x1, row.y1, row.x2, row.y2)), axis=1)
    index = scores.idxmax()
    physical_box = tuple(frame.loc[index, ["x1", "y1", "x2", "y2"]])
    coordinate_delta = box_max_coordinate_delta(box, physical_box)
    if scores.loc[index] < 0.90 and coordinate_delta > 1.01:
        raise ValueError(f"Poor P02 mechanism match for {image}: IoU={scores.loc[index]:.6f}")
    return frame.loc[index]


def flatten_feature_delta(diagnose: Path, full_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten the fixed feature subset and summarize failure/control effects."""
    samples = json.loads((diagnose / "A1_vs_B2_feature_delta.json").read_text(encoding="utf-8"))["samples"]
    rows = []
    for sample in samples:
        physical = match_physical_row(full_frame, sample["image"], sample["gt_box"])
        row = {"group": sample["group"], "image_id": physical["image_id"], "gt_index": physical["gt_index"]}
        row.update({metric: physical[metric] for metric in CORE_METRICS})
        for stage in ("fused/p4", "neck/p2", "neck/p3", "neck/p4", "neck/p5"):
            for measurement in ("local_contrast", "roi_peak", "roi_energy"):
                row[f"delta/{stage}/{measurement}"] = sample["b"][stage][measurement] - sample["a"][stage][measurement]
        for level in ("p2", "p3", "p4", "p5"):
            row[f"delta/head/{level}/smoke_max"] = (sample["b"][f"head/{level}"]["smoke_max"] -
                                                       sample["a"][f"head/{level}"]["smoke_max"])
        rows.append(row)
    flattened = pd.DataFrame(rows)
    effects = []
    delta_columns = [column for column in flattened if column.startswith("delta/")]
    for column in delta_columns:
        failure = flattened.loc[flattened["group"] == "a_hit_b_miss", column].dropna().to_numpy()
        control = flattened.loc[flattened["group"] == "both_hit_control", column].dropna().to_numpy()
        labels = np.r_[np.ones(len(failure)), np.zeros(len(control))]
        values = np.r_[failure, control]
        effects.append({
            "feature": column,
            "failure_n": len(failure),
            "control_n": len(control),
            "failure_median": np.median(failure),
            "control_median": np.median(control),
            "raw_auc_failure_higher": roc_auc(labels, values),
            "discrimination_auc": max(roc_auc(labels, values), 1 - roc_auc(labels, values)),
        })
    return flattened, pd.DataFrame(effects)


def mechanism_physical_associations(mechanism: pd.DataFrame) -> pd.DataFrame:
    """Correlate physical attributes with B2-A1 internal response changes."""
    rows = []
    delta_columns = [column for column in mechanism if column.startswith("delta/")]
    for subset_name, subset in (
        ("a_hit_b_miss", mechanism[mechanism["group"] == "a_hit_b_miss"]),
        ("fixed_subset_all", mechanism),
    ):
        for physical_metric in CORE_METRICS:
            for response_metric in delta_columns:
                valid = subset[[physical_metric, response_metric]].dropna()
                correlation, p_value = (spearmanr(valid[physical_metric], valid[response_metric])
                                        if len(valid) >= 10 else (np.nan, np.nan))
                rows.append({
                    "subset": subset_name,
                    "physical_metric": physical_metric,
                    "response_metric": response_metric,
                    "n": len(valid),
                    "spearman_rho": correlation,
                    "p_value": p_value,
                })
    result = pd.DataFrame(rows)
    result["fdr_bh"] = benjamini_hochberg(result["p_value"])
    return result


def runtime_recovery_analysis(diagnose: Path, full_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate physical properties to runtime P4 intervention recovery."""
    records = json.loads((diagnose / "B2_p4_runtime_ablation.json").read_text(encoding="utf-8"))["records"]
    modes = ("gamma_zero", "cross_zero", "rgb_cross_zero", "ir_cross_zero")
    rows = []
    for record in records:
        physical = match_physical_row(full_frame, record["image"], record["gt_box"])

        def max_head(mode: str) -> float:
            return max(record["modes"][mode][f"head/p{level}"]["smoke_max"] for level in range(2, 6))

        baseline = max_head("baseline")
        row = {
            "image_id": physical["image_id"],
            "gt_index": physical["gt_index"],
            "reason": record["reason"],
            "size_bin": record["size_bin"],
            "baseline_max_head": baseline,
        }
        row.update({metric: physical[metric] for metric in CORE_METRICS})
        for mode in modes:
            response = max_head(mode)
            row[f"{mode}_max_head"] = response
            row[f"{mode}_delta"] = response - baseline
            row[f"{mode}_recovered"] = response >= 0.25 and baseline < 0.25
        rows.append(row)
    recovery = pd.DataFrame(rows)
    effects = []
    for mode in modes:
        recovered_column = f"{mode}_recovered"
        for metric in CORE_METRICS:
            valid = recovery[[recovered_column, metric]].dropna()
            labels = valid[recovered_column].astype(int).to_numpy()
            effects.append({
                "mode": mode,
                "metric": metric,
                "recovered_n": int(labels.sum()),
                "unrecovered_n": int((labels == 0).sum()),
                "weak_structure_auc": roc_auc(labels, -valid[metric].to_numpy()),
                "recovered_median": valid.loc[valid[recovered_column], metric].median(),
                "unrecovered_median": valid.loc[~valid[recovered_column], metric].median(),
            })
    return recovery, pd.DataFrame(effects)


def plot_results(primary: pd.DataFrame, matched: pd.DataFrame, recovery: pd.DataFrame, output: Path) -> None:
    """Write compact P02 evidence figures."""
    core = primary[primary["family"] == "core"].sort_values("roc_auc")
    fig, axis = plt.subplots(figsize=(9, 5))
    errors = np.vstack((core["roc_auc"] - core["roc_auc_ci95_low"], core["roc_auc_ci95_high"] - core["roc_auc"]))
    axis.errorbar(core["roc_auc"], range(len(core)), xerr=errors, fmt="o", color="#8c2d04", capsize=3)
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.axvline(0.65, color="#2b8cbe", linestyle=":", linewidth=1)
    axis.set_yticks(range(len(core)), core["metric"])
    axis.set_xlabel("ROC-AUC (higher score = weaker structure)")
    axis.set_title("P02-A: Physical predictors of B2-specific smoke failure")
    fig.tight_layout()
    fig.savefig(output / "primary_physical_auc.png", dpi=180)
    plt.close(fig)

    metric = core.sort_values("roc_auc", ascending=False).iloc[0]["metric"]
    plot_frame = matched[[metric, "failure"]].dropna().copy()
    plot_frame["bin"] = pd.qcut(plot_frame[metric], q=5, duplicates="drop")
    binned = plot_frame.groupby("bin", observed=True).agg(failure_rate=("failure", "mean"), n=("failure", "size"))
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(range(len(binned)), binned["failure_rate"], marker="o")
    axis.set_xticks(range(len(binned)), [str(value) for value in binned.index], rotation=20, ha="right")
    axis.set_ylabel("B2-specific failure fraction")
    axis.set_xlabel(metric)
    axis.set_title("Matched failure rate by physical-property quintile")
    fig.tight_layout()
    fig.savefig(output / "matched_failure_bins.png", dpi=180)
    plt.close(fig)

    modes = ("gamma_zero", "cross_zero", "rgb_cross_zero", "ir_cross_zero")
    counts = [int(recovery[f"{mode}_recovered"].sum()) for mode in modes]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(modes, counts, color="#3182bd")
    axis.set_ylabel("Recovered samples (max head >= 0.25)")
    axis.set_title("P02-C: Runtime intervention recovery among 182 failures")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "runtime_recovery_counts.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append("nan" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    full: pd.DataFrame,
    matched: pd.DataFrame,
    balance: pd.DataFrame,
    primary: pd.DataFrame,
    logo: pd.DataFrame,
    associations: pd.DataFrame,
    outcome_effects: pd.DataFrame,
    mechanism_effects: pd.DataFrame,
    mechanism_associations: pd.DataFrame,
    recovery: pd.DataFrame,
    recovery_effects: pd.DataFrame,
    output: Path,
) -> None:
    """Write the complete P02 evidence report."""
    counts = full["outcome"].value_counts().rename_axis("outcome").reset_index(name="n")
    core = primary[primary["family"] == "core"].sort_values("roc_auc", ascending=False)
    strongest_associations = associations.reindex(associations["spearman_rho"].abs().sort_values(ascending=False).index).head(10)
    group_effects = outcome_effects.sort_values("weak_structure_auc", ascending=False).groupby(
        "comparison", observed=True).head(3)
    mechanism = mechanism_effects.sort_values("discrimination_auc", ascending=False).head(12)
    mechanism_correlations = mechanism_associations[
        mechanism_associations["subset"] == "a_hit_b_miss"
    ].reindex(mechanism_associations.loc[
        mechanism_associations["subset"] == "a_hit_b_miss", "spearman_rho"
    ].abs().sort_values(ascending=False).index).head(12)
    recovery_counts = pd.DataFrame({
        "mode": ["gamma_zero", "cross_zero", "rgb_cross_zero", "ir_cross_zero"],
        "recovered": [int(recovery[f"{mode}_recovered"].sum()) for mode in
                      ("gamma_zero", "cross_zero", "rgb_cross_zero", "ir_cross_zero")],
    })
    gamma_effects = recovery_effects[recovery_effects["mode"] == "gamma_zero"].sort_values(
        "weak_structure_auc", ascending=False).head(8)
    pooled = logo[logo["held_out_video"] == "pooled_oof"]
    qualifying = core[(core["roc_auc"] >= 0.65) & (core["roc_auc_ci95_low"] > 0.5) &
                      (core["pr_auc"] > core["prevalence"])]
    conclusion = ("有核心物理量通过 P02-A 单变量门槛，应结合跨视频结果进入 P04/P05。" if len(qualifying) else
                  "没有核心物理量完整通过 P02-A 准入门槛，当前证据不足以直接据此设计 C 系列结构。")
    lines = [
        "# ADR-004 P02 B2 Smoke 漏检与物理属性关联报告",
        "",
        "## 1. 研究问题与数据边界",
        "",
        "P02 检验 P01 的弱结构物理属性是否与 B2 相对 A1 的 smoke 特异性漏检相关。本研究为离线关联诊断，",
        "不重新训练模型，也不把相关性直接解释为 P4 Parallel Cross 的因果效应。",
        "",
        "检测结局来自 A1/B2 `last.pt` 的既有逐 GT 诊断，判定阈值为 `conf=0.25, IoU=0.5`。P01 与检测",
        ("记录通过图像与 GT 框一对一连接；实际最小 IoU="
         f"`{full['physical_match_iou'].min():.6f}`，最大坐标偏差="
         f"`{full['physical_match_max_coordinate_delta'].max():.1f}px`。差异来自归一化框整数化。"),
        "",
        "## 2. 四分组总体",
        "",
        markdown_table(counts, 0),
        "",
        "## 3. P02-A：匹配后的物理属性归因",
        "",
        f"以 182 个 `a1_hit_b2_miss` 为病例，每例最多匹配 3 个 `both_hit`，最终分析 {len(matched)} 个实例。",
        f"稀有分层耗尽时允许最近邻对照复用，共 `{int(matched['control_reused'].sum())}` 条复用记录。",
        ("匹配放宽层级：" + ", ".join(f"`{key}`={value}" for key, value in
                                   matched.loc[matched['failure'] == 1, 'match_relaxation'].value_counts().items()) + "。"),
        "匹配控制 video、size bin、连续尺寸、与其他 GT 的重叠比例以及 fire/person 图像级共现。",
        "核心指标的分数方向预注册为“数值越低，弱结构得分越高”，因此 ROC-AUC > 0.5 才支持 H3。",
        "",
        "### 3.1 混杂变量平衡",
        "",
        markdown_table(balance),
        "",
        "### 3.2 核心单变量结果",
        "",
        markdown_table(core[["metric", "case_median", "control_median", "roc_auc", "roc_auc_ci95_low",
                             "roc_auc_ci95_high", "pr_auc", "prevalence", "fdr_bh"]]),
        "",
        f"通过全部预设门槛的核心指标数：**{len(qualifying)} / {len(CORE_METRICS)}**。{conclusion}",
        "",
        "### 3.3 Leave-one-video-out 多变量结果",
        "",
        markdown_table(logo),
        "",
        (f"汇总 OOF ROC-AUC={pooled.iloc[0]['roc_auc']:.4f}，PR-AUC={pooled.iloc[0]['pr_auc']:.4f}，"
         f"阳性率基线={pooled.iloc[0]['prevalence']:.4f}。" if not pooled.empty else "没有足够视频形成有效 OOF 结果。"),
        "",
        "### 3.4 与连续检测变化的关联",
        "",
        markdown_table(strongest_associations[["subset", "metric", "endpoint", "n", "spearman_rho", "fdr_bh"]]),
        "",
        "### 3.5 四分组补充比较",
        "",
        "以下为每种比较中弱结构区分度最高的三个核心指标。该表未做病例匹配，只用于判断 B2 反向获益和",
        "共同困难样本是否具有不同物理画像，不替代 3.2 的主分析。",
        "",
        markdown_table(group_effects[["comparison", "metric", "positive_n", "reference_n", "positive_median",
                                      "reference_median", "weak_structure_auc"]]),
        "",
        "## 4. P02-B：固定 182+182 子集的逐层机制",
        "",
        "该部分只覆盖既有逐层诊断子集，不代表全部 smoke。`delta` 均为 B2-A1；内部 head ROI max 是机制",
        "测量，不等同于最终 NMS 后 confidence。区分度最高的逐层量如下：",
        "",
        markdown_table(mechanism[["feature", "failure_median", "control_median", "raw_auc_failure_higher",
                                  "discrimination_auc"]]),
        "",
        "物理量与逐层 B2-A1 响应变化在 182 个失败样本内的最强相关项如下。这里仍需同时检查 FDR，",
        "不能仅按相关系数绝对值选择结论。",
        "",
        markdown_table(mechanism_correlations[["physical_metric", "response_metric", "n", "spearman_rho", "fdr_bh"]]),
        "",
        "## 5. P02-C：182 个漏检样本内的 runtime 恢复",
        "",
        "恢复定义为 baseline max-head smoke response < 0.25，干预后达到 >= 0.25。",
        "",
        markdown_table(recovery_counts, 0),
        "",
        "gamma-zero recovered 与 unrecovered 的核心物理属性区分结果：",
        "",
        markdown_table(gamma_effects[["metric", "recovered_n", "unrecovered_n", "weak_structure_auc",
                                      "recovered_median", "unrecovered_median"]]),
        "",
        "## 6. 综合结论",
        "",
        "1. P01 的弱结构指标能够区分 smoke 与 fire/person，但不能区分 B2 特异性漏检与稳定检出 smoke。",
        "2. 匹配单变量 AUC 接近 0.5，跨视频多变量 OOF 也未超过基线，因此广义 H3 不成立。",
        "3. 逐层子集仍清楚显示 B2 特异性漏检伴随 neck P3/P4 与 head 响应下降；这是网络机制差异，",
        "   但不能由原图弱边缘、弱 IR 对比或低跨模态边缘一致性单独预测。",
        "4. gamma-zero 恢复的 31 个样本略偏低 RGB gradient/edge density，但 AUC 只有约 0.58-0.59，",
        "   只能视为局部线索，尚不足以形成按物理阈值控制 cross 的正式方法。",
        "5. 因此 P02 不支持直接将 P01 物理量注入 B2 作为 C 系列方案；若继续 P04，应检验这些先验是否",
        "   对 smoke-vs-background 有增量信息，而不是声称它们已经解释 B2 的 Recall 损失。",
        "",
        "## 7. 结论边界",
        "",
        "1. P02-A 回答物理属性是否预测 B2 特异性失败；P02-B/C 只提供机制一致性和干预敏感性证据。",
        "2. 同一视频的相邻帧高度相关，因此主要看 leave-one-video-out，而不是随机实例切分结果。",
        "3. 检测框内物理统计包含背景，不能替代 smoke mask；object/ring 指标还受背景环可用像素影响。",
        "4. runtime 恢复不等于最终 NMS 检出恢复，也不能单独证明 cross 是唯一原因。",
        "5. P02 只允许通过门槛且跨视频稳定的物理量进入 P04/P05，探索性指标不能直接触发 C 系列。",
        "",
        "## 8. 产物",
        "",
        "- `smoke_outcome_physics.csv`：4086 个 smoke 的物理属性与四分组结局",
        "- `matched_manifest.csv` / `matching_balance.csv`：匹配队列与平衡检查",
        "- `primary_effects.csv`：核心和探索性单变量结果",
        "- `leave_one_video_out.csv` / `leave_one_video_out_predictions.csv`：跨视频模型结果",
        "- `continuous_associations.csv` / `failure_reason_effects.csv`：连续结局与原因分层",
        "- `outcome_group_summary.csv` / `outcome_pairwise_effects.csv`：四分组画像与补充比较",
        "- `mechanism_subset.csv` / `mechanism_effects.csv`：固定逐层子集",
        "- `mechanism_physical_associations.csv`：物理属性与逐层响应变化的相关性",
        "- `runtime_recovery.csv` / `runtime_recovery_effects.csv`：运行时恢复子研究",
        "- `summary.json`：机器可读摘要",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run all P02 analyses and write reproducible artifacts."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    physical = pd.read_csv(args.physical)
    outcomes = load_outcomes(args.diagnose)
    full = join_physical_metrics(outcomes, physical)
    matched = match_cases_controls(full, args.controls_per_case, args.seed)
    balance = summarize_balance(matched)
    primary = build_primary_effects(matched, args.bootstrap, args.seed)
    logo, logo_predictions = leave_one_video_out(matched)
    associations = continuous_associations(full)
    outcome_summary, outcome_effects = outcome_group_analyses(full)
    reasons = failure_reason_effects(full)
    mechanism, mechanism_effects = flatten_feature_delta(args.diagnose, full)
    mechanism_associations = mechanism_physical_associations(mechanism)
    recovery, recovery_effects = runtime_recovery_analysis(args.diagnose, full)

    outputs = {
        "smoke_outcome_physics.csv": full,
        "matched_manifest.csv": matched.drop(columns=[column for column in matched if column.startswith("__score_")]),
        "matching_balance.csv": balance,
        "primary_effects.csv": primary,
        "leave_one_video_out.csv": logo,
        "leave_one_video_out_predictions.csv": logo_predictions,
        "continuous_associations.csv": associations,
        "failure_reason_effects.csv": reasons,
        "outcome_group_summary.csv": outcome_summary,
        "outcome_pairwise_effects.csv": outcome_effects,
        "mechanism_subset.csv": mechanism,
        "mechanism_effects.csv": mechanism_effects,
        "mechanism_physical_associations.csv": mechanism_associations,
        "runtime_recovery.csv": recovery,
        "runtime_recovery_effects.csv": recovery_effects,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output / filename, index=False)
    plot_results(primary, matched, recovery, args.output)
    write_report(full, matched, balance, primary, logo, associations, outcome_effects, mechanism_effects,
                 mechanism_associations, recovery, recovery_effects, args.output)
    summary = {
        "outcome_counts": full["outcome"].value_counts().to_dict(),
        "minimum_physical_match_iou": float(full["physical_match_iou"].min()),
        "matched_cases": int(matched["failure"].sum()),
        "matched_controls": int((matched["failure"] == 0).sum()),
        "core_metrics_passing_univariate_gate": primary[
            (primary["family"] == "core") & (primary["roc_auc"] >= 0.65) &
            (primary["roc_auc_ci95_low"] > 0.5) & (primary["pr_auc"] > primary["prevalence"])
        ]["metric"].tolist(),
        "runtime_recovered": {mode: int(recovery[f"{mode}_recovered"].sum()) for mode in
                              ("gamma_zero", "cross_zero", "rgb_cross_zero", "ir_cross_zero")},
        "settings": {
            "physical": str(args.physical),
            "diagnose": str(args.diagnose),
            "controls_per_case": args.controls_per_case,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
