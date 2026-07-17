# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""MCDet loss functions."""

import types

import torch
import torch.nn as nn

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


def wise_iou_v1(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return WIoUv1 loss and plain LIoU for xyxy boxes."""
    iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, eps=eps).clamp(0.0, 1.0)
    px1, py1, px2, py2 = pred_bboxes.chunk(4, -1)
    tx1, ty1, tx2, ty2 = target_bboxes.chunk(4, -1)
    p_cx, p_cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    t_cx, t_cy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    center_dist = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)
    cw = px2.maximum(tx2) - px1.minimum(tx1)
    ch = py2.maximum(ty2) - py1.minimum(ty1)
    enclosing_diag = cw.pow(2) + ch.pow(2) + eps
    liou = 1.0 - iou
    distance_gain = (center_dist / enclosing_diag).clamp(min=0.0, max=4.0)
    return torch.exp(distance_gain) * liou, liou


class WIoUBboxLoss(BboxLoss):
    """Bbox loss that replaces CIoU with WIoU while keeping DFL unchanged."""

    def __init__(self, reg_max=16, alpha: float = 1.9, delta: float = 3.0, momentum: float = 0.01):
        super().__init__(reg_max)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.momentum = float(momentum)
        self.register_buffer("liou_mean", torch.tensor(1.0))

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Compute WIoU bbox loss and the standard DFL term."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        wiou_v1, liou = wise_iou_v1(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        with torch.no_grad():
            batch_mean = liou.detach().mean().clamp(min=1e-6)
            self.liou_mean.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean)
        beta = (liou.detach() / self.liou_mean.clamp(min=1e-6)).clamp(min=1e-6, max=10.0)
        alpha = beta.new_tensor(self.alpha)
        gain = beta / (self.delta * torch.pow(alpha, beta - self.delta).clamp(min=1e-6))
        gain = gain.clamp(max=10.0)
        loss_iou = (gain * wiou_v1 * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl


class MCDetDetectionLoss(v8DetectionLoss):
    """YOLO detection loss with MCDet WIoU bbox term."""

    def __init__(self, model):
        super().__init__(model)
        self.bbox_loss = WIoUBboxLoss(self.reg_max).to(self.device)


class MCDetDualStreamDetectionLoss:
    """Main MCDet WIoU loss plus optional inherited RGB/IR P3 auxiliary losses."""

    def __init__(self, model, aux_weight=0.25):
        self.aux_weight = aux_weight
        self._model = model
        self.main_criterion = MCDetDetectionLoss(model)
        self._aux_crit_rgb = MCDetDetectionLoss(
            types.SimpleNamespace(
                model=nn.ModuleList([model.aux_head_rgb]),
                args=model.args,
                parameters=model.aux_head_rgb.parameters,
            )
        )
        self._aux_crit_ir = MCDetDetectionLoss(
            types.SimpleNamespace(
                model=nn.ModuleList([model.aux_head_ir]),
                args=model.args,
                parameters=model.aux_head_ir.parameters,
            )
        )

    def __call__(self, preds, batch):
        """Compute MCDet loss and keep trainer loss item layout compatible with dual-stream training."""
        main_loss, main_items = self.main_criterion(preds, batch)
        aux_rgb_val = main_items.new_zeros(1)
        aux_ir_val = main_items.new_zeros(1)
        if self._model._aux_rgb is not None:
            a_rgb, a_rgb_items = self._aux_crit_rgb(self._model._aux_rgb, batch)
            a_ir, a_ir_items = self._aux_crit_ir(self._model._aux_ir, batch)
            aux_rgb_val = a_rgb_items.sum().reshape(1).detach()
            aux_ir_val = a_ir_items.sum().reshape(1).detach()
            main_loss = main_loss + self.aux_weight * (a_rgb + a_ir)
            self._model._aux_rgb = None
            self._model._aux_ir = None
        return main_loss, torch.cat([main_items, aux_rgb_val, aux_ir_val])
