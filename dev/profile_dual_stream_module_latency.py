# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Profile logical per-module latency for dual-stream YOLOv12 models.

This script is intended for C/G-series fire/person models. It times the main
logical modules used by ``DualStreamDetectionModel``:

- RGB backbone layers
- IR backbone layers
- fusion modules at P2/P3/P4/P5
- neck/detect head layers

The timings are raw model forward timings only. They do not include dataloader,
image decode, preprocessing, NMS, plotting, or any post-processing outside the
model forward call.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ultralytics.nn.tasks import DualStreamDetectionModel


@dataclass
class TargetModule:
    """A logical module selected for latency profiling."""

    name: str
    group: str
    module_type: str
    params: int
    input_shape: str = ""
    output_shape: str = ""


@dataclass
class LatencyRow:
    """Aggregated latency statistics for one profiled module."""

    name: str
    group: str
    module_type: str
    params: int
    calls: int
    mean_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    share_of_total_mean_pct: float
    input_shape: str
    output_shape: str


def shape_str(obj: Any) -> str:
    """Return a compact tensor/list/dict shape string."""
    if torch.is_tensor(obj):
        return "x".join(str(dim) for dim in obj.shape)
    if isinstance(obj, (list, tuple)):
        return "[" + ", ".join(shape_str(item) for item in obj) + "]"
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{key}: {shape_str(value)}" for key, value in obj.items()) + "}"
    return type(obj).__name__


def count_params(module: nn.Module) -> int:
    """Count recursive parameters for a logical module."""
    return sum(parameter.numel() for parameter in module.parameters(recurse=True))


def collect_targets(model: DualStreamDetectionModel) -> list[tuple[TargetModule, nn.Module]]:
    """Collect non-overlapping logical modules in forward order."""
    targets: list[tuple[TargetModule, nn.Module]] = []

    for branch_name, branch in (("backbone_rgb", model.backbone_rgb), ("backbone_ir", model.backbone_ir)):
        for idx, module in enumerate(branch):
            layer_idx = getattr(module, "i", idx)
            name = f"{branch_name}.{idx:02d}_layer{layer_idx}_{type(module).__name__}"
            targets.append(
                (
                    TargetModule(
                        name=name,
                        group=branch_name,
                        module_type=type(module).__name__,
                        params=count_params(module),
                    ),
                    module,
                )
            )

    stage_order = {"p2": 0, "p3": 1, "p4": 2, "p5": 3}
    for stage_name, module in sorted(model.fusion_convs.items(), key=lambda item: stage_order.get(item[0], 99)):
        targets.append(
            (
                TargetModule(
                    name=f"fusion_convs.{stage_name}_{type(module).__name__}",
                    group="fusion_convs",
                    module_type=type(module).__name__,
                    params=count_params(module),
                ),
                module,
            )
        )

    for idx, module in enumerate(model.head):
        layer_idx = getattr(module, "i", idx)
        name = f"head.{idx:02d}_layer{layer_idx}_{type(module).__name__}"
        targets.append(
            (
                TargetModule(
                    name=name,
                    group="head",
                    module_type=type(module).__name__,
                    params=count_params(module),
                ),
                module,
            )
        )

    return targets


def profile_latency(
    cfg: str,
    imgsz: tuple[int, int],
    device: str,
    warmup: int,
    iters: int,
    nc: int,
    half: bool,
) -> tuple[list[LatencyRow], dict[str, Any]]:
    """Profile module latency with CUDA events."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for module latency profiling.")

    torch.backends.cudnn.benchmark = False
    model = DualStreamDetectionModel(cfg, nc=nc, verbose=False).to(device).eval()
    if half:
        model = model.half()
    else:
        model = model.float()

    dtype = torch.float16 if half else torch.float32
    dummy = torch.zeros(1, 6, imgsz[0], imgsz[1], device=device, dtype=dtype)

    targets = collect_targets(model)
    module_to_target = {module: target for target, module in targets}
    records: dict[str, list[float]] = defaultdict(list)
    total_times: list[float] = []
    pending: dict[int, list[tuple[str, torch.cuda.Event]]] = defaultdict(list)
    event_pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    enabled = False

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        if not enabled:
            return
        target = module_to_target[module]
        target.input_shape = shape_str(inputs)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        pending[id(module)].append((target.name, start))

    def post_hook(module: nn.Module, inputs: Any, output: Any) -> None:
        if not enabled:
            return
        target = module_to_target[module]
        target.output_shape = shape_str(output)
        name, start = pending[id(module)].pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        event_pairs.append((name, start, end))

    handles = []
    for _, module in targets:
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(dummy)
            torch.cuda.synchronize()

            for _ in range(iters):
                event_pairs.clear()
                start_total = torch.cuda.Event(enable_timing=True)
                end_total = torch.cuda.Event(enable_timing=True)
                enabled = True
                start_total.record()
                model(dummy)
                end_total.record()
                enabled = False
                torch.cuda.synchronize()

                total_times.append(start_total.elapsed_time(end_total))
                for name, start, end in event_pairs:
                    records[name].append(start.elapsed_time(end))
    finally:
        enabled = False
        for handle in handles:
            handle.remove()

    total_mean = float(np.mean(total_times)) if total_times else 0.0
    rows: list[LatencyRow] = []
    target_by_name = {target.name: target for target, _ in targets}
    for name, values in records.items():
        arr = np.asarray(values, dtype=float)
        target = target_by_name[name]
        rows.append(
            LatencyRow(
                name=name,
                group=target.group,
                module_type=target.module_type,
                params=target.params,
                calls=int(arr.size),
                mean_ms=float(arr.mean()),
                median_ms=float(np.median(arr)),
                p10_ms=float(np.percentile(arr, 10)),
                p90_ms=float(np.percentile(arr, 90)),
                min_ms=float(arr.min()),
                max_ms=float(arr.max()),
                share_of_total_mean_pct=float(arr.mean() / total_mean * 100.0) if total_mean else 0.0,
                input_shape=target.input_shape,
                output_shape=target.output_shape,
            )
        )

    metadata = {
        "cfg": cfg,
        "imgsz": list(imgsz),
        "device": device,
        "dtype": "fp16" if half else "fp32",
        "warmup": warmup,
        "iters": iters,
        "total_forward_mean_ms": total_mean,
        "total_forward_median_ms": float(np.median(total_times)) if total_times else 0.0,
        "total_forward_p10_ms": float(np.percentile(total_times, 10)) if total_times else 0.0,
        "total_forward_p90_ms": float(np.percentile(total_times, 90)) if total_times else 0.0,
        "params": count_params(model),
        "detect_f": list(model.model[-1].f),
        "strides": [float(stride) for stride in model.model[-1].stride.tolist()],
        "raw_output_note": "training raw output is P2/P3/P4/P5 feature maps for C65; eval decoded output is post-Detect concat",
    }
    return sorted(rows, key=lambda row: row.mean_ms, reverse=True), metadata


def write_outputs(rows: list[LatencyRow], metadata: dict[str, Any], out_dir: Path, prefix: str) -> None:
    """Write CSV and JSON profiling outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}_module_latency.csv"
    json_path = out_dir / f"{prefix}_module_latency.json"
    summary_path = out_dir / f"{prefix}_module_latency_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    group_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"mean_ms_sum": 0.0, "params": 0.0})
    for row in rows:
        group_stats[row.group]["mean_ms_sum"] += row.mean_ms
        group_stats[row.group]["params"] += row.params

    payload = {
        "metadata": metadata,
        "rows": [asdict(row) for row in rows],
        "groups": group_stats,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total = metadata["total_forward_mean_ms"]
    lines = [
        f"# Module latency summary: {prefix}",
        "",
        f"- cfg: `{metadata['cfg']}`",
        f"- input: `1x6x{metadata['imgsz'][0]}x{metadata['imgsz'][1]}`",
        f"- dtype: `{metadata['dtype']}`",
        f"- total forward mean: `{total:.4f} ms`",
        f"- total forward median: `{metadata['total_forward_median_ms']:.4f} ms`",
        "",
        "## Groups",
        "",
        "| group | summed module mean | share of total | params |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group, stats in sorted(group_stats.items(), key=lambda item: item[1]["mean_ms_sum"], reverse=True):
        share = stats["mean_ms_sum"] / total * 100.0 if total else 0.0
        lines.append(f"| `{group}` | {stats['mean_ms_sum']:.4f} ms | {share:.2f}% | {int(stats['params']):,} |")

    lines.extend(
        [
            "",
            "## Top Modules",
            "",
            "| rank | module | type | mean | median | p90 | share | output |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, row in enumerate(rows[:30], start=1):
        lines.append(
            f"| {rank} | `{row.name}` | `{row.module_type}` | {row.mean_ms:.4f} ms | "
            f"{row.median_ms:.4f} ms | {row.p90_ms:.4f} ms | {row.share_of_total_mean_pct:.2f}% | "
            f"`{row.output_shape}` |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="ultralytics/cfg/models/v12/C65-y8n-backbone-fire-person.yaml")
    parser.add_argument("--imgsz", nargs=2, type=int, default=(480, 640), metavar=("H", "W"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nc", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--half", action="store_true", help="Profile FP16 instead of FP32.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/detect/profile_module_latency"))
    parser.add_argument("--prefix", default="c65_fp32")
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    rows, metadata = profile_latency(
        cfg=args.cfg,
        imgsz=tuple(args.imgsz),
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        nc=args.nc,
        half=args.half,
    )
    write_outputs(rows, metadata, out_dir=args.out_dir, prefix=args.prefix)

    print()
    print(
        f"Total forward: mean={metadata['total_forward_mean_ms']:.4f} ms, "
        f"median={metadata['total_forward_median_ms']:.4f} ms"
    )
    print(f"Top {args.topk} modules by mean latency:")
    print("| rank | module | group | type | mean | median | p90 | share |")
    print("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for rank, row in enumerate(rows[: args.topk], start=1):
        print(
            f"| {rank} | `{row.name}` | `{row.group}` | `{row.module_type}` | "
            f"{row.mean_ms:.4f} | {row.median_ms:.4f} | {row.p90_ms:.4f} | "
            f"{row.share_of_total_mean_pct:.2f}% |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
