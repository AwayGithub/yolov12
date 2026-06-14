# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.gradient_conflict import (
    PositiveClassLossExtractor,
    bootstrap_ci,
    collect_module_parameters,
    flatten_gradients,
    safe_cosine_similarity,
)


EXPERIMENTS = {
    "F2": {
        "root": Path("runs/detect/train_MF_plainP2_P2345_P4A2C2f_P3aux"),
        "weights": {
            "best": Path("runs/detect/train_MF_plainP2_P2345_P4A2C2f_P3aux/weights/best.pt"),
            "last": Path("runs/detect/train_MF_plainP2_P2345_P4A2C2f_P3aux/weights/last.pt"),
            "epoch185": Path("runs/detect/train_MF_plainP2_P2345_P4A2C2f_P3aux/weights/epoch185.pt"),
        },
    },
    "D2.1": {
        "root": Path("runs/detect/train_MF_DMGInit8dP2_P2345_P4A2C2f_P3aux"),
        "weights": {
            "last": Path("runs/detect/train_MF_DMGInit8dP2_P2345_P4A2C2f_P3aux/last.pt"),
        },
    },
    "A1": {
        "root": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3"),
        "weights": {
            "best": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3/weights/best.pt"),
            "last": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3/weights/last.pt"),
        },
    },
    "B2": {
        "root": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4"),
        "weights": {
            "best": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4/weights/best.pt"),
            "last": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4/weights/last.pt"),
        },
    },
    "B8": {
        "root": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4_self4_cross4"),
        "weights": {
            "best": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4_self4_cross4/weights/best.pt"),
            "last": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4_self4_cross4/weights/last.pt"),
            "epoch189": Path("runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4_self4_cross4/weights/epoch189.pt"),
        },
    },
}

CLASS_PAIRS = [("smoke", "fire"), ("smoke", "person"), ("fire", "person")]
PROBE_STRATA = ("smoke_only", "smoke_fire", "smoke_person", "smoke_fire_person")


@dataclass
class CheckpointEntry:
    experiment: str
    label: str
    path: Path
    approx_epoch: int | None


def read_results_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    clean = []
    for row in rows:
        out = {}
        for key, value in row.items():
            value = value.strip()
            if key == "epoch":
                out[key] = int(float(value))
            else:
                if value.endswith("="):
                    value = value[:-1]
                out[key] = float(value)
        clean.append(out)
    return clean


def infer_checkpoint_epoch(rows: list[dict], label: str) -> int | None:
    if label.startswith("epoch"):
        return int(label.replace("epoch", ""))
    if label == "last":
        return int(rows[-1]["epoch"])
    if label == "best":
        best = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
        return int(best["epoch"])
    return None


def available_checkpoints() -> list[CheckpointEntry]:
    entries = []
    for exp_name, spec in EXPERIMENTS.items():
        rows = read_results_csv(spec["root"] / "results.csv")
        for label, path in spec["weights"].items():
            if path.exists():
                entries.append(
                    CheckpointEntry(
                        experiment=exp_name,
                        label=label,
                        path=path,
                        approx_epoch=infer_checkpoint_epoch(rows, label),
                    )
                )
    return entries


def build_base_trainer(device: str, batch: int, workers: int, project: Path) -> DetectionTrainer:
    ref_weight = EXPERIMENTS["A1"]["weights"]["best"]
    model, _ = attempt_load_one_weight(ref_weight, device="cpu", fuse=False)
    data_cfg = model.args["data"]
    if isinstance(data_cfg, dict):
        data_cfg = dict(data_cfg)
        data_root = Path(str(data_cfg.get("path", "")))
        if not data_root.exists():
            data_cfg["path"] = "/data/xwh/dataset/RGBT-3M/RGBT-3M"
    trainer = DetectionTrainer(
        overrides={
            "model": str(ref_weight),
            "data": data_cfg,
            "imgsz": model.args["imgsz"],
            "batch": batch,
            "workers": workers,
            "device": device,
            "project": str(project),
            "name": "gradient_conflict_probe",
            "exist_ok": True,
            "save": False,
            "plots": False,
            "rect": True,
            "task": "detect",
        }
    )
    trainer.model = model
    trainer.set_model_attributes()
    return trainer


def _dataset_image_path(dataset, index: int) -> str:
    for attr in ("im_files", "im_files_rgb", "files"):
        if hasattr(dataset, attr):
            files = getattr(dataset, attr)
            if isinstance(files, list) and index < len(files):
                return str(files[index])
    return str(index)


def build_probe_manifest(dataset, max_per_stratum: int, seed: int) -> dict:
    names = dataset.data["names"]
    if isinstance(names, dict):
        inv_names = {name: int(idx) for idx, name in names.items()}
    else:
        inv_names = {name: idx for idx, name in enumerate(names)}
    smoke = inv_names["smoke"]
    fire = inv_names["fire"]
    person = inv_names["person"]

    groups = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        present = {int(x) for x in np.array(label["cls"]).reshape(-1).tolist()}
        has_smoke = smoke in present
        has_fire = fire in present
        has_person = person in present
        if has_smoke and not has_fire and not has_person:
            groups["smoke_only"].append(idx)
        elif has_smoke and has_fire and not has_person:
            groups["smoke_fire"].append(idx)
        elif has_smoke and has_person and not has_fire:
            groups["smoke_person"].append(idx)
        elif has_smoke and has_fire and has_person:
            groups["smoke_fire_person"].append(idx)

    rng = np.random.default_rng(seed)
    selected = []
    strata = {}
    for name in PROBE_STRATA:
        indices = list(groups.get(name, []))
        rng.shuffle(indices)
        picked = indices[:max_per_stratum]
        strata[name] = picked
        selected.extend(picked)
    selected = sorted(set(selected))
    return {
        "seed": seed,
        "max_per_stratum": max_per_stratum,
        "selected_indices": selected,
        "strata": strata,
        "images": [{"index": idx, "path": _dataset_image_path(dataset, idx)} for idx in selected],
    }


def build_probe_loader(dataset, selected_indices: list[int], batch_size: int, workers: int) -> DataLoader:
    subset = Subset(dataset, selected_indices)
    return DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        collate_fn=getattr(dataset, "collate_fn", None),
    )


def ensure_parallel_cross_runtime_state(model) -> None:
    """Rebuild missing runtime metadata for older DualStreamDetectionModel checkpoints."""
    if not hasattr(model, "backbone_rgb"):
        return
    if not hasattr(model, "lif_gate"):
        model.lif_gate = None
    for module in model.modules():
        if type(module).__name__ == "FreDFTFusion" and not hasattr(module, "checkpoint_ffn"):
            module.checkpoint_ffn = False
        if type(module).__name__ == "DualParallelCrossA2C2f":
            if not hasattr(module, "stage_concat"):
                module.stage_concat = False
            if not hasattr(module, "gamma_mode"):
                module.gamma_mode = "free"
            if not hasattr(module, "gamma_max"):
                module.gamma_max = 0.35
            if not hasattr(module, "cross_drop_path"):
                module.cross_drop_path = 0.0
            if not hasattr(module, "cross_scale_rgb"):
                module.register_buffer("cross_scale_rgb", torch.tensor(1.0))
            if not hasattr(module, "cross_scale_ir"):
                module.register_buffer("cross_scale_ir", torch.tensor(1.0))
            if module.stage_concat:
                if not hasattr(module, "cross_mid_scale_rgb"):
                    module.register_buffer("cross_mid_scale_rgb", torch.tensor(1.0))
                if not hasattr(module, "cross_mid_scale_ir"):
                    module.register_buffer("cross_mid_scale_ir", torch.tensor(1.0))
    if hasattr(model, "_parallel_cross_layer_to_stage"):
        return
    layer_to_stage = {}
    idx_to_stage = {idx: stage for stage, idx in getattr(model, "FUSION_LAYER_INDICES", {}).items()}
    for layer in model.backbone_rgb:
        if type(layer).__name__ == "DualParallelCrossA2C2f" and hasattr(layer, "i"):
            layer_to_stage[layer.i] = idx_to_stage.get(layer.i, f"layer{layer.i}")
    model._parallel_cross_layer_to_stage = layer_to_stage


def build_validator(trainer: DetectionTrainer):
    validator = yolo.detect.DetectionValidator(
        save_dir=trainer.save_dir,
        args=copy(trainer.args),
        _callbacks=trainer.callbacks,
    )
    validator.data = trainer.data
    validator.device = trainer.device
    return validator


def resolve_group_modules(model) -> dict[str, list[torch.nn.Module]]:
    return {
        "p4_module": [model.backbone_rgb[6], model.backbone_ir[6]],
        "shared_neck": [model.head],
        "detection_head": [model.model[-1]],
    }


def grad_vector(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    device = params[0].device if params else torch.device("cpu")
    if not params or (not loss.requires_grad):
        return torch.zeros(0, device=device)
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return flatten_gradients(grads, device=device)


def set_train_forward_mode(model: torch.nn.Module) -> None:
    """Use training-path forward while keeping normalization statistics frozen."""
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def positive_assignment_summary(state, class_names: list[str]) -> dict[str, float]:
    summary = {}
    for cls_idx, cls_name in enumerate(class_names):
        cls_scores = state.target_scores[..., cls_idx]
        cls_mask = cls_scores > 0
        count = int(cls_mask.sum().item())
        summary[f"assign/{cls_name}/positive_count"] = count
        summary[f"assign/{cls_name}/score_sum"] = float(cls_scores[cls_mask].sum().item()) if count else 0.0
        summary[f"assign/{cls_name}/score_mean"] = float(cls_scores[cls_mask].mean().item()) if count else 0.0
    return summary


def summarize_pair(values: list[float], severe_threshold: float = -0.2) -> dict[str, float | int | list[float] | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "conflict_rate": None,
            "severe_conflict_rate": None,
            "ci95": None,
            "n": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "conflict_rate": float((arr < 0).mean()),
        "severe_conflict_rate": float((arr < severe_threshold).mean()),
        "ci95": bootstrap_ci(values),
        "n": int(arr.size),
    }


def analyze_checkpoint(
    checkpoint: CheckpointEntry,
    trainer: DetectionTrainer,
    probe_loader: DataLoader,
    class_names: list[str],
    output_dir: Path,
) -> tuple[list[dict], dict]:
    model, _ = attempt_load_one_weight(checkpoint.path, device=trainer.device, fuse=False)
    ensure_parallel_cross_runtime_state(model)
    model = model.to(trainer.device).float().eval()
    model.requires_grad_(True)
    validator = build_validator(trainer)
    extractor = PositiveClassLossExtractor(model)
    group_params = {
        group_name: collect_module_parameters(modules)
        for group_name, modules in resolve_group_modules(model).items()
    }
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    batch_rows = []
    grouped_cosines = defaultdict(list)
    assignment_aggregate = defaultdict(list)

    for batch_idx, raw_batch in enumerate(probe_loader):
        batch = validator.preprocess(raw_batch)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            set_train_forward_mode(model)
            preds = model(batch["img"])
            state = extractor.build_state(preds, batch)
            losses = extractor.per_class_positive_losses(state)

            class_vectors = {}
            for cls_name in class_names:
                cls_loss = losses[class_to_idx[cls_name]]
                class_vectors[cls_name] = {
                    group_name: grad_vector(cls_loss, params)
                    for group_name, params in group_params.items()
                }

        row_base = {
            "experiment": checkpoint.experiment,
            "checkpoint_label": checkpoint.label,
            "checkpoint_path": str(checkpoint.path),
            "approx_epoch": checkpoint.approx_epoch,
            "batch_index": batch_idx,
        }
        row_base.update(positive_assignment_summary(state, class_names))
        for key, value in list(row_base.items()):
            if key.startswith("assign/") and key.endswith("positive_count"):
                assignment_aggregate[key].append(value)

        for group_name in group_params:
            norms = {}
            for cls_name in class_names:
                vector = class_vectors[cls_name][group_name]
                norms[cls_name] = float(vector.norm().item()) if vector.numel() else 0.0
                row_base[f"{group_name}/{cls_name}_norm"] = norms[cls_name]
            if norms["fire"] > 0:
                row_base[f"{group_name}/norm_ratio_smoke_fire"] = norms["smoke"] / norms["fire"]
            else:
                row_base[f"{group_name}/norm_ratio_smoke_fire"] = None
            if norms["person"] > 0:
                row_base[f"{group_name}/norm_ratio_smoke_person"] = norms["smoke"] / norms["person"]
            else:
                row_base[f"{group_name}/norm_ratio_smoke_person"] = None

            for a_name, b_name in CLASS_PAIRS:
                cosine = safe_cosine_similarity(class_vectors[a_name][group_name], class_vectors[b_name][group_name])
                row_base[f"{group_name}/cos/{a_name}_{b_name}"] = cosine
                if cosine is not None:
                    grouped_cosines[(group_name, a_name, b_name)].append(cosine)

        batch_rows.append(dict(row_base))

    summary = {
        "experiment": checkpoint.experiment,
        "checkpoint_label": checkpoint.label,
        "checkpoint_path": str(checkpoint.path),
        "approx_epoch": checkpoint.approx_epoch,
        "pair_metrics": {},
        "assignment": {},
    }
    for key, values in assignment_aggregate.items():
        arr = np.asarray(values, dtype=np.float64)
        summary["assignment"][key] = {
            "mean": float(arr.mean()) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
        }
    for (group_name, a_name, b_name), values in grouped_cosines.items():
        summary["pair_metrics"][f"{group_name}/{a_name}_{b_name}"] = summarize_pair(values)

    batch_csv = output_dir / "batch_metrics.csv"
    write_csv(batch_csv, batch_rows)
    return batch_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compute_recall_delta(exp_name: str, approx_epoch: int | None) -> dict[str, float | None]:
    rows = read_results_csv(EXPERIMENTS[exp_name]["root"] / "results.csv")
    if approx_epoch is None:
        return {cls: None for cls in ("smoke", "fire", "person")}
    target = min(rows, key=lambda row: abs(int(row["epoch"]) - int(approx_epoch)))
    last = rows[-1]
    return {
        cls: float(last[f"metrics/{cls}/recall(B)"] - target[f"metrics/{cls}/recall(B)"])
        for cls in ("smoke", "fire", "person")
    }


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_report(
    summaries: list[dict],
    manifest: dict,
    output_dir: Path,
    class_names: list[str],
) -> str:
    lines = []
    lines.append("# ADR-003 梯度冲突诊断报告")
    lines.append("")
    lines.append("## 1. 诊断范围与约束")
    lines.append("")
    lines.append(f"- Probe set 图像数：`{len(manifest['selected_indices'])}`")
    lines.append(f"- 分层：`{', '.join(PROBE_STRATA)}`")
    lines.append("- 参数组：`p4_module`、`shared_neck`、`detection_head`")
    lines.append("- 注意：A1/B2/D2.1 未保留 40/80/120/160/190 全部阶段 checkpoint，本次报告对这些实验只能基于现存 `best/last` 做离线诊断；F2/B8 额外包含单个周期 checkpoint。")
    lines.append("")
    lines.append("## 2. Checkpoint 可用性")
    lines.append("")
    lines.append("| 实验 | 可用 checkpoint | 近似 epoch |")
    lines.append("| -- | -- | -- |")
    for summary in summaries:
        lines.append(
            f"| {summary['experiment']} | {summary['checkpoint_label']} | {summary['approx_epoch'] if summary['approx_epoch'] is not None else 'N/A'} |"
        )
    lines.append("")
    lines.append("## 3. Shared Neck 主结论")
    lines.append("")
    lines.append("| 实验 | checkpoint | smoke-fire mean cos | smoke-fire conflict rate | smoke-person mean cos | smoke-person conflict rate | fire-person mean cos |")
    lines.append("| -- | -- | --: | --: | --: | --: | --: |")
    for summary in summaries:
        smf = summary["pair_metrics"].get("shared_neck/smoke_fire", {})
        smp = summary["pair_metrics"].get("shared_neck/smoke_person", {})
        fip = summary["pair_metrics"].get("shared_neck/fire_person", {})
        lines.append(
            f"| {summary['experiment']} | {summary['checkpoint_label']} | "
            f"{format_metric(smf.get('mean'))} | {format_metric(smf.get('conflict_rate'))} | "
            f"{format_metric(smp.get('mean'))} | {format_metric(smp.get('conflict_rate'))} | "
            f"{format_metric(fip.get('mean'))} |"
        )
    lines.append("")

    smoke_corr_inputs = []
    smoke_corr_targets = []
    for summary in summaries:
        smf = summary["pair_metrics"].get("shared_neck/smoke_fire", {})
        if smf.get("conflict_rate") is None:
            continue
        recall_delta = compute_recall_delta(summary["experiment"], summary["approx_epoch"])
        if recall_delta["smoke"] is None:
            continue
        smoke_corr_inputs.append(float(smf["conflict_rate"]))
        smoke_corr_targets.append(float(recall_delta["smoke"]))
    corr = pearson_corr(smoke_corr_inputs, smoke_corr_targets)

    lines.append("## 4. 诊断判断")
    lines.append("")
    if corr is not None:
        lines.append(f"- `shared_neck smoke-fire conflict rate` 与同期到最终的 smoke Recall 变化相关系数：`{corr:.4f}`。")
    else:
        lines.append("- 可用 checkpoint 数不足，无法稳定估计 `shared_neck smoke-fire conflict rate` 与 smoke Recall 回落的相关系数。")
    lines.append("- 请以 `shared_neck` 为主结论、`p4_module` 为机制解释、`detection_head` 为辅助对照阅读下面的图表与 JSON 汇总。")
    lines.append("")
    lines.append("## 5. 产物")
    lines.append("")
    lines.append("- `manifest.json`")
    lines.append("- `checkpoint_summary.json`")
    lines.append("- `checkpoint_summary.csv`")
    lines.append("- `batch_metrics.csv`")
    lines.append("- `shared_neck_conflict_rate.png`")
    lines.append("- `shared_neck_mean_cosine.png`")
    lines.append("")
    return "\n".join(lines)


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def plot_summary(summaries: list[dict], output_dir: Path) -> None:
    labels = [f"{item['experiment']}-{item['checkpoint_label']}" for item in summaries]
    smoke_fire_conflict = [
        item["pair_metrics"].get("shared_neck/smoke_fire", {}).get("conflict_rate") or 0.0 for item in summaries
    ]
    smoke_person_conflict = [
        item["pair_metrics"].get("shared_neck/smoke_person", {}).get("conflict_rate") or 0.0 for item in summaries
    ]
    smoke_fire_mean = [item["pair_metrics"].get("shared_neck/smoke_fire", {}).get("mean") or 0.0 for item in summaries]
    smoke_person_mean = [
        item["pair_metrics"].get("shared_neck/smoke_person", {}).get("mean") or 0.0 for item in summaries
    ]

    x = np.arange(len(labels))
    plt.figure(figsize=(14, 5))
    plt.bar(x - 0.15, smoke_fire_conflict, width=0.3, label="smoke-fire")
    plt.bar(x + 0.15, smoke_person_conflict, width=0.3, label="smoke-person")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("conflict rate")
    plt.title("Shared Neck Conflict Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "shared_neck_conflict_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(x, smoke_fire_mean, marker="o", label="smoke-fire")
    plt.plot(x, smoke_person_mean, marker="o", label="smoke-person")
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("mean cosine")
    plt.title("Shared Neck Mean Cosine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "shared_neck_mean_cosine.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-per-stratum", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/detect/adr003/gradient_conflict")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = build_base_trainer(device=args.device, batch=args.batch, workers=args.workers, project=output_dir)
    val_dataset = trainer.build_dataset(trainer.testset, mode="val", batch=args.batch)
    manifest = build_probe_manifest(val_dataset, max_per_stratum=args.max_per_stratum, seed=args.seed)
    probe_loader = build_probe_loader(val_dataset, manifest["selected_indices"], batch_size=args.batch, workers=args.workers)

    class_names = ["smoke", "fire", "person"]
    all_rows = []
    summaries = []
    for checkpoint in available_checkpoints():
        rows, summary = analyze_checkpoint(checkpoint, trainer, probe_loader, class_names, output_dir)
        all_rows.extend(rows)
        summaries.append(summary)

    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    with (output_dir / "checkpoint_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)
    write_csv(output_dir / "checkpoint_summary.csv", summaries)
    write_csv(output_dir / "batch_metrics.csv", all_rows)
    plot_summary(summaries, output_dir)
    report = build_report(summaries, manifest, output_dir, class_names)
    (output_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
