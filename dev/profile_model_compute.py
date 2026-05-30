# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Profile per-module compute for YOLOv12 experiment checkpoints.

The script uses forward hooks and a zero tensor input to estimate MACs/FLOPs by
module. It also adds an explicit FFT estimate for FreDFT frequency attention,
because torch.fft calls are functions and are otherwise invisible to module hooks.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics import YOLO


CONTAINER_TYPES = (nn.Sequential, nn.ModuleList, nn.ModuleDict)


@dataclass
class ProfileRow:
    """One profiled operation or module estimate."""

    name: str
    module_id: int
    module_type: str
    params: int
    macs: float
    flops: float
    input_shape: str
    output_shape: str
    note: str = ""


def _shape(obj: Any) -> str:
    """Return a compact tensor/list shape string."""
    if isinstance(obj, torch.Tensor):
        return "x".join(str(dim) for dim in obj.shape)
    if isinstance(obj, (list, tuple)):
        return "[" + ", ".join(_shape(item) for item in obj) + "]"
    return type(obj).__name__


def _first_tensor(obj: Any) -> torch.Tensor | None:
    """Find the first tensor in nested hook inputs/outputs."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _params(module: nn.Module, recursive: bool = False) -> int:
    """Count module parameters."""
    return sum(parameter.numel() for parameter in module.parameters(recurse=recursive))


def _conv2d_macs(module: nn.Conv2d, output: Any) -> float:
    """Estimate Conv2d multiply-accumulate operations."""
    y = _first_tensor(output)
    if y is None or y.ndim != 4:
        return 0.0
    batch, out_channels, out_h, out_w = y.shape
    kernel_h, kernel_w = module.kernel_size
    kernel_mul = module.in_channels // module.groups * kernel_h * kernel_w
    return float(batch * out_channels * out_h * out_w * kernel_mul)


def _linear_macs(module: nn.Linear, output: Any) -> float:
    """Estimate Linear multiply-accumulate operations."""
    y = _first_tensor(output)
    if y is None:
        return 0.0
    return float(y.numel() * module.in_features)


def _batchnorm_flops(output: Any) -> float:
    """Approximate affine normalization cost."""
    y = _first_tensor(output)
    return float(y.numel() * 2) if y is not None else 0.0


def _layernorm_flops(output: Any) -> float:
    """Approximate LayerNorm cost over normalized values."""
    y = _first_tensor(output)
    return float(y.numel() * 5) if y is not None else 0.0


def _upsample_flops(output: Any) -> float:
    """Nearest upsample is mostly memory movement; keep a small element-count proxy."""
    y = _first_tensor(output)
    return float(y.numel()) if y is not None else 0.0


def _fft_flops(module: nn.Module, inputs: Any) -> float:
    """Estimate rfft2/irfft2 cost used by FreDFT frequency attention.

    FreDFT calls _frequency_gate twice. Each gate performs rfft2(q), rfft2(k),
    complex elementwise multiplication, and irfft2(response). A practical FFT
    estimate is 5 * N * log2(N) real FLOPs per transform.
    """
    if type(module).__name__ != "_FreDFTFrequencyAttention":
        return 0.0
    first = _first_tensor(inputs)
    if first is None or first.ndim != 4:
        return 0.0
    batch, _channels, height, width = first.shape
    qkv_channels = getattr(module.project_out, "in_channels", _channels)
    n = height * width
    if n <= 1:
        return 0.0
    real_fft_flops = 5.0 * n * math.log2(n)
    complex_bins = height * (width // 2 + 1)
    complex_multiply_flops = 6.0 * complex_bins
    one_gate = 3.0 * real_fft_flops + complex_multiply_flops
    return float(2 * batch * qkv_channels * one_gate)


def _leaf_macs_and_flops(module: nn.Module, inputs: Any, output: Any) -> tuple[float, float, str]:
    """Return MACs, FLOPs, and note for directly countable leaf operations."""
    if isinstance(module, nn.Conv2d):
        macs = _conv2d_macs(module, output)
        return macs, macs * 2.0, "conv2d"
    if isinstance(module, nn.Linear):
        macs = _linear_macs(module, output)
        return macs, macs * 2.0, "linear"
    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        flops = _batchnorm_flops(output)
        return 0.0, flops, "batchnorm"
    if isinstance(module, nn.LayerNorm):
        flops = _layernorm_flops(output)
        return 0.0, flops, "layernorm"
    if isinstance(module, nn.Upsample):
        flops = _upsample_flops(output)
        return 0.0, flops, "upsample-memory-proxy"

    fft_flops = _fft_flops(module, inputs)
    if fft_flops:
        return 0.0, fft_flops, "estimated torch.fft rfft2/irfft2"
    return 0.0, 0.0, ""


def _is_leaf_or_custom(module: nn.Module) -> bool:
    """Select modules whose hooks should create profile rows."""
    if isinstance(module, CONTAINER_TYPES):
        return False
    if any(module.children()):
        return type(module).__name__ == "_FreDFTFrequencyAttention"
    return True


def _module_group(name: str) -> str:
    """Group leaf rows into major model regions."""
    parts = name.split(".")
    if not parts:
        return name
    if parts[0] in {"backbone_rgb", "backbone_ir", "head", "fusion_convs", "aux_head_rgb", "aux_head_ir"}:
        return ".".join(parts[:2]) if len(parts) > 1 else parts[0]
    return parts[0]


def profile_model(model: nn.Module, channels: int, imgsz: tuple[int, int], device: str) -> list[ProfileRow]:
    """Run one forward pass and collect per-module compute rows."""
    rows: list[ProfileRow] = []
    handles = []
    module_names = {module: name for name, module in model.named_modules()}

    def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        macs, flops, note = _leaf_macs_and_flops(module, inputs, output)
        if macs == 0.0 and flops == 0.0:
            return
        rows.append(
            ProfileRow(
                name=module_names.get(module, "<unnamed>"),
                module_id=id(module),
                module_type=type(module).__name__,
                params=_params(module),
                macs=macs,
                flops=flops,
                input_shape=_shape(inputs),
                output_shape=_shape(output),
                note=note,
            )
        )

    for module in model.modules():
        if _is_leaf_or_custom(module):
            handles.append(module.register_forward_hook(hook))

    model.eval().to(device)
    dummy = torch.zeros(1, channels, imgsz[0], imgsz[1], device=device)
    with torch.inference_mode():
        model(dummy)

    for handle in handles:
        handle.remove()
    return rows


def print_report(rows: list[ProfileRow], topk: int) -> None:
    """Print top operations and grouped compute summary."""
    total_macs = sum(row.macs for row in rows)
    total_flops = sum(row.flops for row in rows)
    unique_params = {row.module_id: row.params for row in rows}
    total_params = sum(unique_params.values())

    print(f"Profiled leaf/custom rows: {len(rows)}")
    print("Note: repeated rows mean the same module instance was called multiple times; compute is counted per call.")
    print(f"Unique params in profiled rows: {total_params:,}")
    print(f"Total MACs: {total_macs / 1e9:.4f} GMACs")
    print(f"Total FLOPs: {total_flops / 1e9:.4f} GFLOPs")
    print()

    print(f"Top {topk} operations by FLOPs")
    print("| rank | name | type | params | GMACs | GFLOPs | in -> out | note |")
    print("| ---: | --- | --- | ---: | ---: | ---: | --- | --- |")
    for rank, row in enumerate(sorted(rows, key=lambda item: item.flops, reverse=True)[:topk], start=1):
        print(
            f"| {rank} | `{row.name}` | {row.module_type} | {row.params:,} | "
            f"{row.macs / 1e9:.4f} | {row.flops / 1e9:.4f} | "
            f"`{row.input_shape}` -> `{row.output_shape}` | {row.note} |"
        )

    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"macs": 0.0, "flops": 0.0})
    group_param_ids: dict[str, dict[int, int]] = defaultdict(dict)
    for row in rows:
        group = _module_group(row.name)
        grouped[group]["macs"] += row.macs
        grouped[group]["flops"] += row.flops
        group_param_ids[group][row.module_id] = row.params

    print()
    print("Top module groups by FLOPs")
    print("| rank | group | params | GMACs | GFLOPs | share |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for rank, (group, stats) in enumerate(
        sorted(grouped.items(), key=lambda item: item[1]["flops"], reverse=True)[:topk],
        start=1,
    ):
        share = stats["flops"] / total_flops * 100.0 if total_flops else 0.0
        group_params = sum(group_param_ids[group].values())
        print(
            f"| {rank} | `{group}` | {group_params:,} | "
            f"{stats['macs'] / 1e9:.4f} | {stats['flops'] / 1e9:.4f} | {share:.2f}% |"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default=(
            r"E:\Yan-Unifiles\lab\exp\yolov12\runs\detect\RGBT-3M\fair"
            r"\train_MF_DMGInit8dP2_FreAttP3\weights\best.pt"
        ),
        help="Checkpoint or model YAML to profile.",
    )
    parser.add_argument("--imgsz", nargs=2, type=int, default=(480, 640), metavar=("H", "W"))
    parser.add_argument("--channels", type=int, default=None, help="Input channels. Defaults to model.ch or 6.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0.")
    parser.add_argument("--topk", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights/model file not found: {weights}")

    model = YOLO(str(weights)).model
    channels = args.channels or int(getattr(model, "ch", 6))
    rows = profile_model(model, channels=channels, imgsz=tuple(args.imgsz), device=args.device)
    print_report(rows, topk=args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
