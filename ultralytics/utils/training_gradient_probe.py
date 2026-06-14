# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import csv
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.gradient_conflict import (
    PositiveClassLossExtractor,
    collect_module_parameters,
    flatten_gradients,
    safe_cosine_similarity,
)

CLASS_NAMES = ("smoke", "fire", "person")
CLASS_PAIRS = (("smoke", "fire"), ("smoke", "person"), ("fire", "person"))
PROBE_STRATA = ("smoke_only", "smoke_fire", "smoke_person", "smoke_fire_person")


def _append_csv(path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV while preserving a stable header."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _image_path(dataset, index: int) -> str:
    for attr in ("im_files", "im_files_rgb", "files"):
        files = getattr(dataset, attr, None)
        if isinstance(files, list) and index < len(files):
            return str(files[index])
    return str(index)


def build_probe_manifest(dataset, max_per_stratum: int, seed: int) -> dict:
    """Build a deterministic smoke-centered, class-cooccurrence-stratified probe manifest."""
    names = dataset.data["names"]
    inv_names = {name: int(idx) for idx, name in names.items()} if isinstance(names, dict) else {
        name: idx for idx, name in enumerate(names)
    }
    smoke, fire, person = (inv_names[name] for name in CLASS_NAMES)
    groups = defaultdict(list)
    for index, label in enumerate(dataset.labels):
        present = {int(x) for x in np.asarray(label["cls"]).reshape(-1).tolist()}
        key = None
        if smoke in present and fire not in present and person not in present:
            key = "smoke_only"
        elif smoke in present and fire in present and person not in present:
            key = "smoke_fire"
        elif smoke in present and person in present and fire not in present:
            key = "smoke_person"
        elif smoke in present and fire in present and person in present:
            key = "smoke_fire_person"
        if key:
            groups[key].append(index)

    rng = np.random.default_rng(seed)
    selected, strata = [], {}
    for name in PROBE_STRATA:
        indices = list(groups[name])
        rng.shuffle(indices)
        strata[name] = indices[:max_per_stratum]
        selected.extend(strata[name])
    selected = sorted(set(selected))
    return {
        "seed": seed,
        "max_per_stratum": max_per_stratum,
        "selected_indices": selected,
        "strata": strata,
        "images": [{"index": idx, "path": _image_path(dataset, idx)} for idx in selected],
    }


def build_probe_loader(dataset, indices: list[int], batch_size: int) -> DataLoader:
    """Return a deterministic probe loader using the validation dataset collate function."""
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )


def _p4_module(model):
    return model.backbone_rgb[model.FUSION_LAYER_INDICES["p4"]]


def resolve_parameter_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    """Resolve semantic parameter groups shared by A1 and B2."""
    p4 = _p4_module(model)
    modules = {
        "p4_total": [p4],
        "p4_fusion": [model.fusion_convs["p4"]],
        "shared_neck": [model.head],
        "detection_head": [model.model[-1]],
    }
    for name in ("self_rgb", "self_ir", "cross_rgb", "cross_ir", "cv2_rgb", "cv2_ir"):
        module = getattr(p4, name, None)
        if module is not None:
            modules[f"p4_{name}"] = [module]
    return {name: collect_module_parameters(group) for name, group in modules.items()}


def resolve_scalar_parameters(model) -> dict[str, torch.nn.Parameter]:
    """Resolve trainable cross-modal scalar parameters."""
    p4 = _p4_module(model)
    names = ("gamma_rgb", "gamma_ir", "gamma_rgb_logit", "gamma_ir_logit", "cross_scale_rgb", "cross_scale_ir")
    return {name: value for name in names if isinstance((value := getattr(p4, name, None)), torch.nn.Parameter)}


def resolve_activation_modules(model) -> dict[str, torch.nn.Module]:
    """Resolve compact activation checkpoints along P4 and the shared neck."""
    p4 = _p4_module(model)
    modules = {"p4_output": p4, "fused_p4": model.fusion_convs["p4"]}
    for name in ("self_rgb", "self_ir", "cross_rgb", "cross_ir"):
        blocks = getattr(p4, name, None)
        if blocks is not None:
            for index, block in enumerate(blocks, 1):
                modules[f"p4_{name}_{index}"] = block
    # Head indices are stable for the A1/B2 model family.
    head_names = {
        2: "topdown_p4",
        5: "topdown_p3",
        8: "topdown_p2",
        11: "bottomup_p3",
        14: "bottomup_p4",
        17: "bottomup_p5",
    }
    for index, name in head_names.items():
        if index < len(model.head):
            modules[name] = model.head[index]
    return modules


def _tensor_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        tensors = [x for x in output if isinstance(x, torch.Tensor)]
        return tensors[0] if tensors else None
    return None


def _capture_activation(store: dict[str, torch.Tensor | None], name: str):
    """Return a forward hook that stores the first tensor output."""
    def hook(_module, _inputs, output):
        store[name] = _tensor_output(output)

    return hook


@contextmanager
def probe_forward_state(model):
    """Freeze normalization and use the raw-output Detect path without mutating model state."""
    training = {module: module.training for module in model.modules()}
    bn_state = {
        module: (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }
    model.eval()
    for module in model.modules():
        if isinstance(module, Detect):
            module.train()
    try:
        yield
    finally:
        for module, was_training in training.items():
            module.training = was_training
        for module, (mean, var, tracked) in bn_state.items():
            module.running_mean.copy_(mean)
            module.running_var.copy_(var)
            module.num_batches_tracked.copy_(tracked)


def _grad_vector(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    if not params or not loss.requires_grad:
        return torch.zeros(0, device=loss.device)
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return flatten_gradients(grads, device=loss.device)


def _summarize(rows: list[dict], epoch: int, probe_kind: str) -> list[dict]:
    summaries = []
    for loss_view in sorted({row.get("loss_view", "cls") for row in rows}):
        view_rows = [row for row in rows if row.get("loss_view", "cls") == loss_view]
        metric_keys = sorted(
            {key for row in view_rows for key in row if key.startswith(("cos/", "norm/", "assign/"))}
        )
        for key in metric_keys:
            values = [float(row[key]) for row in view_rows if row.get(key) not in (None, "")]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            summaries.append(
                {
                    "epoch": epoch,
                    "probe_kind": probe_kind,
                    "loss_view": loss_view,
                    "metric": key,
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "conflict_rate": float((arr < 0).mean()) if key.startswith("cos/") else "",
                    "severe_conflict_rate": float((arr < -0.2).mean()) if key.startswith("cos/") else "",
                    "n": int(arr.size),
                }
            )
    return summaries


class TrainingGradientProbe:
    """Run deterministic, read-only class-gradient diagnostics during training."""

    def __init__(
        self,
        small_period: int = 2,
        small_per_stratum: int = 12,
        full_per_stratum: int = 64,
        full_epochs: tuple[int, ...] = (),
        batch_size: int = 2,
        seed: int = 0,
    ):
        self.small_period = small_period
        self.small_per_stratum = small_per_stratum
        self.full_per_stratum = full_per_stratum
        self.full_epochs = set(full_epochs)
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, trainer) -> None:
        """Create fixed manifests and loaders once the validation dataset is available."""
        output_dir = Path(trainer.save_dir) / "gradient_probe"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = trainer.build_dataset(trainer.testset, mode="val", batch=self.batch_size)
        full = build_probe_manifest(dataset, self.full_per_stratum, self.seed)
        small = build_probe_manifest(dataset, self.small_per_stratum, self.seed)
        self.output_dir = output_dir
        self.small_loader = build_probe_loader(dataset, small["selected_indices"], self.batch_size)
        self.full_loader = build_probe_loader(dataset, full["selected_indices"], self.batch_size)
        (output_dir / "manifest_small.json").write_text(json.dumps(small, indent=2))
        (output_dir / "manifest_full.json").write_text(json.dumps(full, indent=2))
        self.completed = self._completed_epochs()
        if not hasattr(trainer, "epoch"):
            trainer.epoch = trainer.start_epoch
        self.run(trainer, 0, "full", self.full_loader)

    def _completed_epochs(self) -> set[tuple[int, str]]:
        path = self.output_dir / "epoch_summary.csv"
        if not path.exists():
            return set()
        with path.open() as f:
            return {(int(row["epoch"]), row["probe_kind"]) for row in csv.DictReader(f)}

    def on_fit_epoch_end(self, trainer) -> None:
        """Run the scheduled probe after validation and metric logging."""
        epoch = trainer.epoch + 1
        if epoch in self.full_epochs:
            self.run(trainer, epoch, "full", self.full_loader)
        elif epoch % self.small_period == 0:
            self.run(trainer, epoch, "small", self.small_loader)

    def run(self, trainer, epoch: int, probe_kind: str, loader: DataLoader) -> None:
        """Run one scheduled probe without changing optimizer or model state."""
        if (epoch, probe_kind) in self.completed:
            return
        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        extractor = PositiveClassLossExtractor(model)
        parameter_groups = resolve_parameter_groups(model)
        scalar_params = resolve_scalar_parameters(model)
        activation_modules = resolve_activation_modules(model) if probe_kind == "full" else {}
        class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        rows, scalar_rows = [], []

        with probe_forward_state(model), torch.enable_grad():
            for batch_index, raw_batch in enumerate(loader):
                batch = trainer.preprocess_batch(raw_batch)
                activations = {}
                hooks = []
                for name, module in activation_modules.items():
                    hooks.append(module.register_forward_hook(_capture_activation(activations, name)))
                preds = model(batch["img"])
                for hook in hooks:
                    hook.remove()
                state = extractor.build_state(preds, batch)
                loss_views = {"cls": extractor.per_class_positive_losses(state)}
                if probe_kind == "full":
                    loss_views["det"] = extractor.per_class_detection_losses(state)

                for loss_view, losses in loss_views.items():
                    class_vectors = {}
                    activation_vectors = {}
                    for class_name, class_idx in class_to_idx.items():
                        loss = losses[class_idx]
                        class_vectors[class_name] = {
                            group: _grad_vector(loss, params) for group, params in parameter_groups.items()
                        }
                        activation_vectors[class_name] = {
                            name: _grad_vector(loss, [value])
                            if value is not None and value.requires_grad
                            else torch.zeros(0, device=loss.device)
                            for name, value in activations.items()
                        }
                        if scalar_params:
                            scalar_grads = torch.autograd.grad(
                                loss, list(scalar_params.values()), retain_graph=True, allow_unused=True
                            )
                            scalar_rows.append(
                                {
                                    "epoch": epoch,
                                    "probe_kind": probe_kind,
                                    "loss_view": loss_view,
                                    "batch_index": batch_index,
                                    "class": class_name,
                                    **{
                                        name: float(grad.detach().item()) if grad is not None else ""
                                        for name, grad in zip(scalar_params, scalar_grads)
                                    },
                                }
                            )

                    row = {
                        "epoch": epoch,
                        "probe_kind": probe_kind,
                        "loss_view": loss_view,
                        "batch_index": batch_index,
                    }
                    for class_name, class_idx in class_to_idx.items():
                        scores = state.target_scores[..., class_idx]
                        mask = scores > 0
                        row[f"assign/{class_name}/positive_count"] = int(mask.sum().item())
                        row[f"assign/{class_name}/score_mean"] = float(scores[mask].mean().item()) if mask.any() else ""
                    for group, _params in parameter_groups.items():
                        for class_name in CLASS_NAMES:
                            row[f"norm/param/{group}/{class_name}"] = float(class_vectors[class_name][group].norm().item())
                        for left, right in CLASS_PAIRS:
                            row[f"cos/param/{group}/{left}_{right}"] = safe_cosine_similarity(
                                class_vectors[left][group], class_vectors[right][group]
                            )
                    for name in activations:
                        for class_name in CLASS_NAMES:
                            row[f"norm/activation/{name}/{class_name}"] = float(
                                activation_vectors[class_name][name].norm().item()
                            )
                        for left, right in CLASS_PAIRS:
                            row[f"cos/activation/{name}/{left}_{right}"] = safe_cosine_similarity(
                                activation_vectors[left][name], activation_vectors[right][name]
                            )
                    rows.append(row)
                del preds, state, loss_views, class_vectors, activation_vectors, activations

        _append_csv(self.output_dir / f"batch_metrics_{probe_kind}.csv", rows)
        _append_csv(self.output_dir / "scalar_gradients.csv", scalar_rows)
        _append_csv(self.output_dir / "epoch_summary.csv", _summarize(rows, epoch, probe_kind))
        self.completed.add((epoch, probe_kind))
        torch.cuda.empty_cache()
