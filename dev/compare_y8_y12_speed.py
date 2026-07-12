# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Compare YOLOv8n and YOLOv12n structure, compute, and latency.

This script focuses on three outputs:
1. Stage-by-stage feature map size changes.
2. Total params/FLOPs and per-layer compute summary.
3. Per-module latency ranking to identify which modules are slower in YOLOv12n.

Example:
    conda run -n yolov12 python dev/compare_y8_y12_speed.py --device cuda:1
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import A2C2f, AAttn, C2f, C3k2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_model_compute import ProfileRow, profile_model


@dataclass
class StageRow:
    idx: int
    name: str
    module_type: str
    input_shape: str
    output_shape: str
    params: int


@dataclass
class LatencyRow:
    name: str
    module_type: str
    params: int
    input_shape: str
    output_shape: str
    avg_ms: float


def _shape(obj: Any) -> str:
    if isinstance(obj, torch.Tensor):
        return "x".join(str(dim) for dim in obj.shape)
    if isinstance(obj, (list, tuple)):
        return "[" + ", ".join(_shape(item) for item in obj) + "]"
    return type(obj).__name__


def _first_tensor(obj: Any) -> torch.Tensor | None:
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


def _params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def capture_stage_rows(model: nn.Module, channels: int, imgsz: tuple[int, int], device: str) -> list[StageRow]:
    rows: list[StageRow] = []
    handles = []

    def hook(name: str, idx: int, module: nn.Module):
        def _hook(_module: nn.Module, inputs: Any, output: Any) -> None:
            rows.append(
                StageRow(
                    idx=idx,
                    name=name,
                    module_type=type(module).__name__,
                    input_shape=_shape(inputs),
                    output_shape=_shape(output),
                    params=sum(p.numel() for p in module.parameters()),
                )
            )

        return _hook

    for idx, module in enumerate(model.model):
        handles.append(module.register_forward_hook(hook(f"model.{idx}", idx, module)))

    model.eval().to(device)
    dummy = torch.zeros(1, channels, imgsz[0], imgsz[1], device=device)
    with torch.inference_mode():
        model(dummy)

    for handle in handles:
        handle.remove()
    return rows


def benchmark_modules(model: nn.Module, channels: int, imgsz: tuple[int, int], device: str, warmup: int,
                      steps: int) -> list[LatencyRow]:
    rows: list[LatencyRow] = []
    interesting = (nn.Conv2d, C2f, C3k2, A2C2f, AAttn)
    handles = []
    names = {m: n for n, m in model.named_modules()}
    timings: dict[int, list[float]] = {}
    meta: dict[int, LatencyRow] = {}

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        module.__start_time = time.perf_counter()
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize(device)

    def post_hook(module: nn.Module, inputs: Any, output: Any) -> None:
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - module.__start_time) * 1000.0
        mid = id(module)
        timings.setdefault(mid, []).append(elapsed)
        meta[mid] = LatencyRow(
            name=names.get(module, "<unnamed>"),
            module_type=type(module).__name__,
            params=sum(p.numel() for p in module.parameters()),
            input_shape=_shape(inputs),
            output_shape=_shape(output),
            avg_ms=0.0,
        )

    for module in model.modules():
        if isinstance(module, interesting):
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))

    model.eval().to(device)
    dummy = torch.zeros(1, channels, imgsz[0], imgsz[1], device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy)
        for _ in range(steps):
            model(dummy)

    for handle in handles:
        handle.remove()

    for mid, samples in timings.items():
        row = meta[mid]
        row.avg_ms = statistics.mean(samples)
        rows.append(row)
    rows.sort(key=lambda x: x.avg_ms, reverse=True)
    return rows


def print_stage_table(title: str, rows: list[StageRow], limit: int | None = None) -> None:
    print()
    print(title)
    print("| idx | module | type | params | input -> output |")
    print("| ---: | --- | --- | ---: | --- |")
    for row in rows[:limit] if limit else rows:
        print(
            f"| {row.idx} | `{row.name}` | {row.module_type} | {row.params:,} | "
            f"`{row.input_shape}` -> `{row.output_shape}` |"
        )


def print_latency_table(title: str, rows: list[LatencyRow], topk: int) -> None:
    print()
    print(title)
    print("| rank | module | type | params | avg ms | input -> output |")
    print("| ---: | --- | --- | ---: | ---: | --- |")
    for rank, row in enumerate(rows[:topk], start=1):
        print(
            f"| {rank} | `{row.name}` | {row.module_type} | {row.params:,} | {row.avg_ms:.4f} | "
            f"`{row.input_shape}` -> `{row.output_shape}` |"
        )


def summarize_profile(rows: list[ProfileRow]) -> tuple[int, float]:
    params = sum({row.module_id: row.params for row in rows}.values())
    flops = sum(row.flops for row in rows) / 1e9
    return params, flops


def build_model(cfg: str, channels: int, nc: int) -> tuple[nn.Module, int]:
    model = DetectionModel(cfg=cfg, ch=channels, nc=nc, verbose=False)
    return model, channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", nargs=2, type=int, default=(480, 640), metavar=("H", "W"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--channels", type=int, default=3, help="force input channels for both models")
    parser.add_argument("--nc", type=int, default=80, help="force class count for both models")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg_v8 = repo / "ultralytics/cfg/models/v8/yolov8.yaml"
    cfg_v12 = repo / "ultralytics/cfg/models/v12/yolov12.yaml"

    model_v8, ch_v8 = build_model(str(cfg_v8), args.channels, args.nc)
    model_v12, ch_v12 = build_model(str(cfg_v12), args.channels, args.nc)

    print(
        f"Comparing with imgsz={tuple(args.imgsz)}, device={args.device}, "
        f"forced_channels={args.channels}, forced_nc={args.nc}"
    )

    stage_v8 = capture_stage_rows(model_v8, ch_v8, tuple(args.imgsz), args.device)
    stage_v12 = capture_stage_rows(model_v12, ch_v12, tuple(args.imgsz), args.device)

    prof_v8 = profile_model(model_v8, channels=ch_v8, imgsz=tuple(args.imgsz), device=args.device)
    prof_v12 = profile_model(model_v12, channels=ch_v12, imgsz=tuple(args.imgsz), device=args.device)
    params_v8, flops_v8 = summarize_profile(prof_v8)
    params_v12, flops_v12 = summarize_profile(prof_v12)

    print("\nSummary")
    print("| model | params (profiled leaf sum) | FLOPs | top-level layers |")
    print("| --- | ---: | ---: | ---: |")
    print(f"| yolov8n | {params_v8:,} | {flops_v8:.4f} GFLOPs | {len(stage_v8)} |")
    print(f"| yolov12n | {params_v12:,} | {flops_v12:.4f} GFLOPs | {len(stage_v12)} |")

    print_stage_table("YOLOv8n stage flow", stage_v8)
    print_stage_table("YOLOv12n stage flow", stage_v12)

    lat_v8 = benchmark_modules(model_v8, ch_v8, tuple(args.imgsz), args.device, args.warmup, args.steps)
    lat_v12 = benchmark_modules(model_v12, ch_v12, tuple(args.imgsz), args.device, args.warmup, args.steps)
    print_latency_table("YOLOv8n slowest modules", lat_v8, args.topk)
    print_latency_table("YOLOv12n slowest modules", lat_v12, args.topk)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
