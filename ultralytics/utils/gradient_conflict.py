# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import xywh2xyxy


@dataclass
class PositiveClassLossState:
    """Container for positive classification loss diagnostics."""

    pred_scores: torch.Tensor
    target_scores: torch.Tensor
    target_scores_sum: torch.Tensor
    positive_bce: torch.Tensor
    fg_mask: torch.Tensor
    gt_labels: torch.Tensor
    pred_distri: torch.Tensor
    pred_bboxes: torch.Tensor
    target_bboxes: torch.Tensor
    anchor_points: torch.Tensor
    stride_tensor: torch.Tensor


class PositiveClassLossExtractor:
    """Read-only helper that reproduces the positive classification part of v8DetectionLoss."""

    def __init__(self, model, tal_topk: int = 10):
        self.criterion = v8DetectionLoss(model, tal_topk=tal_topk)

    @property
    def nc(self) -> int:
        return self.criterion.nc

    def build_state(self, preds, batch: dict) -> PositiveClassLossState:
        """Return intermediate tensors required for per-class positive BCE diagnostics."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([x.view(feats[0].shape[0], self.criterion.no, -1) for x in feats], 2).split(
            (self.criterion.reg_max * 4, self.criterion.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.criterion.device, dtype=dtype) * self.criterion.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.criterion.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.criterion.preprocess(
            targets.to(self.criterion.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.criterion.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.criterion.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)
        positive_bce = self.criterion.bce(pred_scores, target_scores.to(dtype))
        return PositiveClassLossState(
            pred_scores=pred_scores,
            target_scores=target_scores,
            target_scores_sum=target_scores_sum,
            positive_bce=positive_bce,
            fg_mask=fg_mask,
            gt_labels=gt_labels,
            pred_distri=pred_distri,
            pred_bboxes=pred_bboxes,
            target_bboxes=target_bboxes,
            anchor_points=anchor_points,
            stride_tensor=stride_tensor,
        )

    def per_class_positive_losses(self, state: PositiveClassLossState) -> dict[int, torch.Tensor]:
        """Return per-class positive BCE losses normalized exactly like v8DetectionLoss."""
        losses = {}
        for cls_idx in range(self.nc):
            cls_scores = state.target_scores[..., cls_idx]
            cls_mask = cls_scores > 0
            losses[cls_idx] = state.positive_bce[..., cls_idx][cls_mask].sum() / state.target_scores_sum
        return losses

    def total_positive_bce(self, state: PositiveClassLossState) -> torch.Tensor:
        """Return the full positive BCE term for equivalence checks."""
        return state.positive_bce[state.target_scores > 0].sum() / state.target_scores_sum

    def per_class_detection_losses(self, state: PositiveClassLossState) -> dict[int, torch.Tensor]:
        """Return class-positive classification plus class-owned box and DFL losses."""
        losses = {}
        target_bboxes = state.target_bboxes / state.stride_tensor
        for cls_idx in range(self.nc):
            cls_scores = state.target_scores[..., cls_idx]
            cls_mask = cls_scores > 0
            cls_loss = state.positive_bce[..., cls_idx][cls_mask].sum() / state.target_scores_sum
            if cls_mask.any():
                box_loss, dfl_loss = self.criterion.bbox_loss(
                    state.pred_distri,
                    state.pred_bboxes,
                    state.anchor_points,
                    target_bboxes,
                    state.target_scores,
                    state.target_scores_sum,
                    cls_mask,
                )
                hyp = self.criterion.hyp
                cls_gain = hyp["cls"] if isinstance(hyp, dict) else hyp.cls
                box_gain = hyp["box"] if isinstance(hyp, dict) else hyp.box
                dfl_gain = hyp["dfl"] if isinstance(hyp, dict) else hyp.dfl
                cls_loss = cls_loss * cls_gain + box_loss * box_gain + dfl_loss * dfl_gain
            losses[cls_idx] = cls_loss
        return losses


def collect_module_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    """Return unique trainable parameters from the given modules while preserving order."""
    params = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                params.append(param)
    return params


def flatten_gradients(grads: Iterable[torch.Tensor | None], device: torch.device | None = None) -> torch.Tensor:
    """Flatten a gradient list into a single vector, skipping None entries."""
    flat = []
    for grad in grads:
        if grad is None:
            continue
        flat.append(grad.detach().reshape(-1))
    if flat:
        return torch.cat(flat)
    target_device = device or torch.device("cpu")
    return torch.zeros(0, device=target_device)


def safe_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1e-12) -> float | None:
    """Return cosine similarity for two flattened gradient vectors, or None if one side is empty."""
    if vec_a.numel() == 0 or vec_b.numel() == 0:
        return None
    denom = vec_a.norm() * vec_b.norm()
    if denom <= eps:
        return None
    return float(torch.dot(vec_a, vec_b) / denom)


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float] | None:
    """Return percentile bootstrap confidence interval for the sample mean."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        x = float(arr[0])
        return x, x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def preprocess_detection_targets(targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Standalone target preprocessing mirroring v8DetectionLoss.preprocess for tests."""
    nl, ne = targets.shape
    if nl == 0:
        return torch.zeros(batch_size, 0, ne - 1, device=device)
    image_idx = targets[:, 0]
    _, counts = image_idx.unique(return_counts=True)
    counts = counts.to(dtype=torch.int32)
    out = torch.zeros(batch_size, counts.max(), ne - 1, device=device)
    for batch_idx in range(batch_size):
        matches = image_idx == batch_idx
        if num := matches.sum():
            out[batch_idx, :num] = targets[matches, 1:]
    out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out
