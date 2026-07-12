import numpy as np
import torch

try:
    from . import nms_rotated_ext
except ImportError:
    nms_rotated_ext = None


def _hbb_nms(boxes, scores, iou_thr):
    """Pure PyTorch fallback NMS on horizontal boxes."""
    keep = []
    order = scores.argsort(descending=True)
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        cur = boxes[i]
        rest = boxes[order[1:]]
        xx1 = torch.maximum(cur[0], rest[:, 0])
        yy1 = torch.maximum(cur[1], rest[:, 1])
        xx2 = torch.minimum(cur[2], rest[:, 2])
        yy2 = torch.minimum(cur[3], rest[:, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_cur = (cur[2] - cur[0]).clamp(min=0) * (cur[3] - cur[1]).clamp(min=0)
        area_rest = (rest[:, 2] - rest[:, 0]).clamp(min=0) * (rest[:, 3] - rest[:, 1]).clamp(min=0)
        iou = inter / (area_cur + area_rest - inter + 1e-7)
        order = order[1:][iou <= iou_thr]
    return torch.stack(keep) if keep else boxes.new_zeros(0, dtype=torch.long)


def _rbox_to_hbb(dets):
    cx, cy, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=1)

def obb_nms(dets, scores, iou_thr, device_id=None):
    """
    RIoU NMS - iou_thr.
    Args:
        dets (tensor/array): (num, [cx cy w h θ]) θ∈[-pi/2, pi/2)
        scores (tensor/array): (num)
        iou_thr (float): (1)
    Returns:
        dets (tensor): (n_nms, [cx cy w h θ])
        inds (tensor): (n_nms), nms index of dets
    """
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be eithr a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if dets_th.numel() == 0: # len(dets)
        inds = dets_th.new_zeros(0, dtype=torch.int64)
    else:
        # same bug will happen when bboxes is too small
        too_small = dets_th[:, [2, 3]].min(1)[0] < 0.001 # [n]
        if too_small.all(): # all the bboxes is too small
            inds = dets_th.new_zeros(0, dtype=torch.int64)
        else:
            ori_inds = torch.arange(dets_th.size(0), device=dets_th.device) # 0 ~ n-1
            ori_inds = ori_inds[~too_small]
            dets_th = dets_th[~too_small] # (n_filter, 5)
            scores = scores[~too_small]

            if nms_rotated_ext is None:
                inds = _hbb_nms(_rbox_to_hbb(dets_th), scores, iou_thr)
            else:
                inds = nms_rotated_ext.nms_rotated(dets_th, scores, iou_thr)
            inds = ori_inds[inds]

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def poly_nms(dets, iou_thr, device_id=None):
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be eithr a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if nms_rotated_ext is None:
        x = dets_th[:, 0:8:2]
        y = dets_th[:, 1:8:2]
        boxes = torch.stack((x.min(1)[0], y.min(1)[0], x.max(1)[0], y.max(1)[0]), dim=1)
        inds = _hbb_nms(boxes, dets_th[:, 8], iou_thr)
    else:
        if dets_th.device == torch.device('cpu'):
            raise NotImplementedError
        inds = nms_rotated_ext.nms_poly(dets_th.float(), iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds

if __name__ == '__main__':
    rboxes_opencv = torch.tensor(([136.6, 111.6, 200, 100, -60],
                                  [136.6, 111.6, 100, 200, -30],
                                  [100, 100, 141.4, 141.4, -45],
                                  [100, 100, 141.4, 141.4, -45]))
    rboxes_longedge = torch.tensor(([136.6, 111.6, 200, 100, -60],
                                    [136.6, 111.6, 200, 100, 120],
                                    [100, 100, 141.4, 141.4, 45],
                                    [100, 100, 141.4, 141.4, 135]))
    
