# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Trace layer-wise feature changes for smoke targets detected by A1 but missed by B2."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.block import DMGFusion, DMGFusionInit8d, DMGFusionPosAlpha, FreDFTFusion, M2DLocalIlluminationFusion


def patch_legacy(model):
    if not hasattr(model, "_parallel_cross_layer_to_stage"):
        model._parallel_cross_layer_to_stage = {}
    if not hasattr(model, "parallel_cross_a2c2f_stages"):
        model.parallel_cross_a2c2f_stages = set()
    for module in model.modules():
        if type(module).__name__ == "FreDFTFusion" and not hasattr(module, "checkpoint_ffn"):
            module.checkpoint_ffn = False
        if type(module).__name__ == "DualParallelCrossA2C2f":
            if not hasattr(module, "cross_scale_rgb"):
                module.register_buffer("cross_scale_rgb", torch.tensor(1.0))
                module.register_buffer("cross_scale_ir", torch.tensor(1.0))
            if not hasattr(module, "gamma_mode"):
                module.gamma_mode = "free"


def load_input(rgb_path, device):
    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(rgb_path.replace("/RGB/", "/IR/"))
    if rgb is None or ir is None:
        raise FileNotFoundError(rgb_path)
    image = np.concatenate((rgb, ir), axis=-1)[..., ::-1].transpose(2, 0, 1).copy()
    return torch.from_numpy(image).unsqueeze(0).to(device).float() / 255.0


def roi_slices(box, feat):
    _, _, h, w = feat.shape
    x1, y1, x2, y2 = box
    sx, sy = w / 640.0, h / 480.0
    x1, x2 = int(math.floor(x1 * sx)), int(math.ceil(x2 * sx))
    y1, y2 = int(math.floor(y1 * sy)), int(math.ceil(y2 * sy))
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))
    pad = max(1, int(round(max(x2 - x1, y2 - y1) * 0.25)))
    return (slice(y1, y2), slice(x1, x2)), (slice(max(0, y1 - pad), min(h, y2 + pad)), slice(max(0, x1 - pad), min(w, x2 + pad)))


def feature_stats(feat, box):
    feat = feat.detach().float()
    roi_s, outer_s = roi_slices(box, feat)
    energy = feat.square().mean(1).sqrt()[0]
    roi = energy[roi_s]
    outer = energy[outer_s]
    roi_mean = float(roi.mean())
    outer_mean = float(outer.mean())
    return {"roi_energy": roi_mean, "local_contrast": roi_mean / (outer_mean + 1e-9), "roi_peak": float(roi.max())}


def modality_stats(rgb, ir, box):
    out = {}
    out.update({f"rgb_{k}": v for k, v in feature_stats(rgb, box).items()})
    out.update({f"ir_{k}": v for k, v in feature_stats(ir, box).items()})
    rs, _ = roi_slices(box, rgb)
    a = rgb.detach().float()[0, :, rs[0], rs[1]].flatten()
    b = ir.detach().float()[0, :, rs[0], rs[1]].flatten()
    out["rgb_ir_cosine"] = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    return out


def capture_internal_p4(model):
    captured, hooks = {}, []
    layer = model.backbone_rgb[6]

    def hook(name):
        def save(_, __, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach()
        return save

    if type(layer).__name__ == "DualParallelCrossA2C2f":
        for branch in ("self_rgb", "self_ir", "cross_rgb", "cross_ir"):
            for index, module in enumerate(getattr(layer, branch)):
                hooks.append(module.register_forward_hook(hook(f"p4_internal/{branch}_{index + 1}")))
    else:
        for modality, backbone in (("rgb", model.backbone_rgb), ("ir", model.backbone_ir)):
            for index, module in enumerate(backbone[6].m):
                hooks.append(module.register_forward_hook(hook(f"p4_internal/{modality}_group_{index + 1}")))
    return captured, hooks


def trace(model, image, box):
    internal, hooks = capture_internal_p4(model)
    x_ir, x_rgb = image[:, :3], image[:, 3:]
    feats_rgb, feats_ir = model._forward_both_backbones(x_rgb, x_ir)
    lif = model.lif_gate(x_rgb) if model.lif_gate is not None else None
    fused = {}
    for stage in model.FUSION_LAYER_INDICES:
        r, i, fc = feats_rgb[stage], feats_ir[stage], model.fusion_convs[stage]
        if isinstance(fc, M2DLocalIlluminationFusion):
            fused[stage] = fc(r, i, lif)
        elif isinstance(fc, (DMGFusion, DMGFusionPosAlpha, DMGFusionInit8d, FreDFTFusion)):
            fused[stage] = fc(r, i)
        else:
            fused[stage] = fc(torch.cat((r, i), 1))

    y = [None] * (max(model.FUSION_LAYER_INDICES.values()) + 1)
    for stage, index in model.FUSION_LAYER_INDICES.items():
        y[index] = fused[stage]
    neck = {}
    x = fused["p5"]
    for module in model.head:
        if module.f != -1:
            x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
        if type(module).__name__ == "Detect":
            detect_inputs = [z.detach() for z in x]
        x = module(x)
        y.append(x if module.i in model.save else None)
        if module.i in (17, 20, 23, 26):
            neck[{17: "p2", 20: "p3", 23: "p4", 26: "p5"}[module.i]] = x

    raw = x[1]
    result = {}
    for stage in model.FUSION_LAYER_INDICES:
        result[f"backbone/{stage}"] = modality_stats(feats_rgb[stage], feats_ir[stage], box)
        result[f"fused/{stage}"] = feature_stats(fused[stage], box)
    for stage in ("p2", "p3", "p4", "p5"):
        result[f"neck/{stage}"] = feature_stats(neck[stage], box)
    for name, tensor in internal.items():
        result[name] = feature_stats(tensor, box)
    for index, stage in enumerate(("p2", "p3", "p4", "p5")):
        smoke = raw[index][0, 64].sigmoid()
        rs, outer = roi_slices(box, raw[index])
        roi, around = smoke[rs], smoke[outer]
        result[f"head/{stage}"] = {
            "smoke_max": float(roi.max()), "smoke_mean": float(roi.mean()),
            "smoke_contrast": float(roi.mean() / (around.mean() + 1e-9)),
        }
    for h in hooks:
        h.remove()
    return result


def stratified_controls(report, a_records, b_records, count):
    lost = report["a_detected_b_missed"]
    need = defaultdict(int)
    for x in lost:
        need[x["b"]["size_bin"]] += 1
    key = lambda x: (x["image"], x["gt_index"])
    am, bm = {key(x): x for x in a_records}, {key(x): x for x in b_records}
    candidates = defaultdict(list)
    for k in am.keys() & bm.keys():
        if am[k]["detected"] and bm[k]["detected"]:
            candidates[bm[k]["size_bin"]].append(bm[k])
    rng = random.Random(0)
    controls = []
    for size_bin, n in need.items():
        rng.shuffle(candidates[size_bin])
        controls.extend(candidates[size_bin][:n])
    return controls[:count]


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}/{key}" if prefix else key, child, out)
    else:
        out[prefix] = value


def aggregate(rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {key: {"median": float(np.median([r[key] for r in rows if key in r])), "mean": float(np.mean([r[key] for r in rows if key in r]))} for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="runs/detect/last.pt")
    parser.add_argument("--b", default="runs/detect/train2/weights/last.pt")
    parser.add_argument("--report", default="runs/detect/adr003/smoke_delta/A1_vs_B2_smoke_delta.json")
    parser.add_argument("--a-instances", default="runs/detect/adr003/smoke_delta/A1_last_smoke_instances.json")
    parser.add_argument("--b-instances", default="runs/detect/adr003/smoke_delta/B2_last_smoke_instances.json")
    parser.add_argument("--output", default="runs/detect/adr003/smoke_delta/A1_vs_B2_feature_delta.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    wrappers = [YOLO(args.a), YOLO(args.b)]
    models = [x.model.to(device).eval() for x in wrappers]
    for model in models:
        patch_legacy(model)
    report = json.loads(Path(args.report).read_text())
    a_instances = json.loads(Path(args.a_instances).read_text())["records"]
    b_instances = json.loads(Path(args.b_instances).read_text())["records"]
    lost = [x["b"] for x in report["a_detected_b_missed"]]
    controls = stratified_controls(report, a_instances, b_instances, len(lost))
    if args.limit:
        lost, controls = lost[: args.limit], controls[: args.limit]

    payload = {"weights": {"a": args.a, "b": args.b}, "groups": {}, "samples": []}
    with torch.inference_mode():
        for group, records in (("a_hit_b_miss", lost), ("both_hit_control", controls)):
            delta_rows = []
            for index, record in enumerate(records):
                image = load_input(record["image"], device)
                traces = [trace(model, image, record["gt_box"]) for model in models]
                flat = []
                for trace_data in traces:
                    row = {}; flatten("", trace_data, row); flat.append(row)
                delta = {key: flat[1][key] - flat[0][key] for key in flat[0].keys() & flat[1].keys()}
                log_ratio = {key: math.log((flat[1][key] + 1e-9) / (flat[0][key] + 1e-9)) for key in flat[0].keys() & flat[1].keys() if flat[0][key] >= 0 and flat[1][key] >= 0}
                delta_rows.append({f"delta/{k}": v for k, v in delta.items()} | {f"log_ratio/{k}": v for k, v in log_ratio.items()})
                payload["samples"].append({"group": group, "image": record["image"], "gt_box": record["gt_box"], "size_bin": record["size_bin"], "reason": record["reason"], "a": traces[0], "b": traces[1]})
                if (index + 1) % 20 == 0:
                    print(group, index + 1, "/", len(records), flush=True)
            payload["groups"][group] = aggregate(delta_rows)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
