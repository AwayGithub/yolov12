# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Profile internal latency phases of RS-SQF fusion modules.

The normal module-level profiler can only time the whole RS-SQF module. This
script replays the RS-SQF forward path step by step so functional operations
such as matmul, topk, gather, softmax, and query-to-spatial projection are also
timed.

It profiles the P4 and P5 RS-SQF modules from a dual-stream model config using
synthetic feature tensors with the same shapes as the real model stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ultralytics.nn.tasks import DualStreamDetectionModel


@dataclass
class PhaseRow:
    """Latency statistics for one RS-SQF internal phase."""

    stage: str
    phase: str
    mean_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    share_of_total_mean_pct: float
    note: str


def time_cuda(fn: Callable[[], object]) -> tuple[object, float]:
    """Run a function and return CUDA elapsed time in milliseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))


def sparse_read_step(read_module, queries: torch.Tensor, x: torch.Tensor, stage: str, record: Callable):
    """Replay _RSQFSparseSemanticRead.forward with internal timing."""
    bsz, dim, height, width = x.shape
    num_pos = height * width

    q, ms = time_cuda(lambda: read_module.q_proj(read_module.norm_q(queries)))
    record(f"{stage}.read.q_norm_q_proj", ms, "LayerNorm + Linear on learned queries")

    k, ms = time_cuda(lambda: read_module.k_proj(x).flatten(2).transpose(1, 2))
    record(f"{stage}.read.k_proj_flatten", ms, "1x1 Conv key projection + flatten")

    v, ms = time_cuda(lambda: read_module.v_proj(x).flatten(2).transpose(1, 2))
    record(f"{stage}.read.v_proj_flatten", ms, "1x1 Conv value projection + flatten")

    score, ms = time_cuda(lambda: torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim))
    record(f"{stage}.read.score_matmul", ms, "query-to-spatial dense scores")

    topk = min(read_module.topk, num_pos)
    (topv, topi), ms = time_cuda(lambda: torch.topk(score, k=topk, dim=-1))
    record(f"{stage}.read.topk", ms, f"topk={topk} over {num_pos} spatial positions")

    def gather_reduce():
        index = topi.unsqueeze(-1).expand(-1, -1, -1, dim)
        v_expand = v.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
        v_sel = torch.gather(v_expand, dim=2, index=index)
        attn = torch.softmax(topv, dim=-1).unsqueeze(-1)
        sem = (attn * v_sel).sum(dim=2)
        return read_module.norm_z(sem)

    sem, ms = time_cuda(gather_reduce)
    record(f"{stage}.read.gather_softmax_reduce_norm", ms, "gather selected values + softmax + weighted sum + LayerNorm")
    return sem, score


def query_to_spatial(sem: torch.Tensor, score: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Project semantic query tokens back to a BCHW spatial context map."""
    assign = torch.softmax(score.transpose(1, 2), dim=-1)
    ctx = torch.matmul(assign, sem)
    return ctx.transpose(1, 2).contiguous().view(sem.shape[0], sem.shape[2], height, width)


def profile_one_stage(module, stage: str, shape: tuple[int, int, int, int], device: str, dtype: torch.dtype, iters: int):
    """Profile one RS-SQF module stage."""
    channels = shape[1]
    height = shape[2]
    width = shape[3]
    rgb_in = torch.zeros(shape, device=device, dtype=dtype)
    ir_in = torch.zeros(shape, device=device, dtype=dtype)
    records: dict[str, list[float]] = {}
    notes: dict[str, str] = {}
    total_times: list[float] = []

    def record(phase: str, ms: float, note: str) -> None:
        records.setdefault(phase, []).append(ms)
        notes[phase] = note

    for _ in range(iters):
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()

        rgb, ms = time_cuda(lambda: module.rgb_proj(rgb_in))
        record(f"{stage}.01_rgb_proj", ms, f"{channels}->{rgb.shape[1]} projection")

        ir, ms = time_cuda(lambda: module.ir_proj(ir_in))
        record(f"{stage}.02_ir_proj", ms, f"{channels}->{ir.shape[1]} projection")

        rgb_clean, ms = time_cuda(lambda: rgb + module.gamma_red * (-module.rgb_red(rgb) * rgb))
        record(f"{stage}.03_rgb_redundancy_gate", ms, "depthwise gate + residual suppression")

        ir_clean, ms = time_cuda(lambda: ir + module.gamma_red * (-module.ir_red(ir) * ir))
        record(f"{stage}.04_ir_redundancy_gate", ms, "depthwise gate + residual suppression")

        queries, ms = time_cuda(lambda: module.semantic_queries.unsqueeze(0).expand(shape[0], -1, -1))
        record(f"{stage}.05_query_expand", ms, "expand learned query tokens to batch")

        rgb_sem, rgb_score = sparse_read_step(module.rgb_read, queries, rgb_clean, f"{stage}.06_rgb", record)
        ir_sem, ir_score = sparse_read_step(module.ir_read, queries, ir_clean, f"{stage}.07_ir", record)

        msg_rgb, ms = time_cuda(lambda: module.cross_query.rgb_from_ir(rgb_sem, ir_sem))
        record(f"{stage}.08_cross_rgb_from_ir_attention", ms, "token attention: RGB queries attend IR semantic tokens")

        msg_ir, ms = time_cuda(lambda: module.cross_query.ir_from_rgb(ir_sem, rgb_sem))
        record(f"{stage}.09_cross_ir_from_rgb_attention", ms, "token attention: IR queries attend RGB semantic tokens")

        gate_rgb, ms = time_cuda(
            lambda: module.cross_query.gate_rgb(torch.cat([rgb_sem, msg_rgb, rgb_sem * msg_rgb], dim=-1))
        )
        record(f"{stage}.10_cross_rgb_gate", ms, "MLP gate for RGB semantic update")

        gate_ir, ms = time_cuda(
            lambda: module.cross_query.gate_ir(torch.cat([ir_sem, msg_ir, ir_sem * msg_ir], dim=-1))
        )
        record(f"{stage}.11_cross_ir_gate", ms, "MLP gate for IR semantic update")

        rgb_sem2, ms = time_cuda(lambda: rgb_sem + gate_rgb * msg_rgb)
        record(f"{stage}.12_cross_rgb_update", ms, "gated residual semantic update")

        ir_sem2, ms = time_cuda(lambda: ir_sem + gate_ir * msg_ir)
        record(f"{stage}.13_cross_ir_update", ms, "gated residual semantic update")

        rgb_ctx, ms = time_cuda(lambda: query_to_spatial(rgb_sem2, rgb_score, height, width))
        record(f"{stage}.14_rgb_query_to_spatial", ms, "softmax score + matmul assignment back to HxW")

        ir_ctx, ms = time_cuda(lambda: query_to_spatial(ir_sem2, ir_score, height, width))
        record(f"{stage}.15_ir_query_to_spatial", ms, "softmax score + matmul assignment back to HxW")

        rgb_enh, ms = time_cuda(lambda: rgb_in + module.gamma_rgb * module.rgb_out(rgb_ctx))
        record(f"{stage}.16_rgb_out_residual", ms, "project context back to original channels + residual")

        ir_enh, ms = time_cuda(lambda: ir_in + module.gamma_ir * module.ir_out(ir_ctx))
        record(f"{stage}.17_ir_out_residual", ms, "project context back to original channels + residual")

        _, ms = time_cuda(lambda: module.relu(module.fuse(torch.cat([rgb_enh, ir_enh], dim=1))))
        record(f"{stage}.18_final_cat_fuse_relu", ms, "concat enhanced streams + 1x1 fuse + ReLU")

        total_end.record()
        torch.cuda.synchronize()
        total_times.append(float(total_start.elapsed_time(total_end)))

    total_mean = float(np.mean(total_times))
    rows: list[PhaseRow] = []
    for phase, values in records.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            PhaseRow(
                stage=stage,
                phase=phase,
                mean_ms=float(arr.mean()),
                median_ms=float(np.median(arr)),
                p10_ms=float(np.percentile(arr, 10)),
                p90_ms=float(np.percentile(arr, 90)),
                min_ms=float(arr.min()),
                max_ms=float(arr.max()),
                share_of_total_mean_pct=float(arr.mean() / total_mean * 100.0) if total_mean else 0.0,
                note=notes[phase],
            )
        )
    rows.sort(key=lambda row: row.mean_ms, reverse=True)
    return rows, {
        "stage": stage,
        "shape": list(shape),
        "total_mean_ms": total_mean,
        "total_median_ms": float(np.median(total_times)),
        "hidden_channels": int(module.semantic_queries.shape[-1]),
        "num_queries": int(module.semantic_queries.shape[0]),
        "topk": int(module.rgb_read.topk),
    }


def write_outputs(rows: list[PhaseRow], metadata: dict, out_dir: Path, prefix: str) -> None:
    """Write CSV, JSON, and Markdown summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}_rsqf_internal_latency.csv"
    json_path = out_dir / f"{prefix}_rsqf_internal_latency.json"
    md_path = out_dir / f"{prefix}_rsqf_internal_latency_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    payload = {"metadata": metadata, "rows": [asdict(row) for row in rows]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# RS-SQF internal latency: {prefix}",
        "",
        f"- cfg: `{metadata['cfg']}`",
        f"- dtype: `{metadata['dtype']}`",
        f"- device: `{metadata['device']}`",
        f"- iters: `{metadata['iters']}`",
        "",
        "## Stage Metadata",
        "",
        "| stage | shape | hidden | queries | topk | total mean | total median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in metadata["stages"]:
        lines.append(
            f"| `{stage['stage']}` | `{stage['shape']}` | {stage['hidden_channels']} | {stage['num_queries']} | "
            f"{stage['topk']} | {stage['total_mean_ms']:.4f} ms | {stage['total_median_ms']:.4f} ms |"
        )

    lines.extend(
        [
            "",
            "## Top Internal Phases",
            "",
            "| rank | stage | phase | mean | median | p90 | share | note |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, row in enumerate(rows[:40], start=1):
        lines.append(
            f"| {rank} | `{row.stage}` | `{row.phase}` | {row.mean_ms:.4f} ms | {row.median_ms:.4f} ms | "
            f"{row.p90_ms:.4f} ms | {row.share_of_total_mean_pct:.2f}% | {row.note} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Summary: {md_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default="ultralytics/cfg/models/v12/C65-y8n-backbone-fire-person.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/detect/profile_module_latency"))
    parser.add_argument("--prefix", default="c65_fp32_gpu1")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RS-SQF internal latency profiling.")

    dtype = torch.float16 if args.half else torch.float32
    model = DualStreamDetectionModel(args.cfg, nc=2, verbose=False).to(args.device).eval()
    model = model.half() if args.half else model.float()

    p4 = model.fusion_convs["p4"]
    p5 = model.fusion_convs["p5"]
    stage_specs = {
        "p4": (p4, (1, 128, 30, 40)),
        "p5": (p5, (1, 256, 15, 20)),
    }

    # Warm up both modules before detailed timing.
    with torch.inference_mode():
        for _ in range(args.warmup):
            for module, shape in stage_specs.values():
                x_rgb = torch.zeros(shape, device=args.device, dtype=dtype)
                x_ir = torch.zeros(shape, device=args.device, dtype=dtype)
                module(x_rgb, x_ir)
        torch.cuda.synchronize()

        all_rows: list[PhaseRow] = []
        stage_meta = []
        for stage, (module, shape) in stage_specs.items():
            rows, meta = profile_one_stage(module, stage, shape, args.device, dtype, args.iters)
            all_rows.extend(rows)
            stage_meta.append(meta)

    all_rows.sort(key=lambda row: row.mean_ms, reverse=True)
    metadata = {
        "cfg": args.cfg,
        "device": args.device,
        "dtype": "fp16" if args.half else "fp32",
        "iters": args.iters,
        "warmup": args.warmup,
        "stages": stage_meta,
        "note": "Synthetic feature tensors with C65 P4/P5 shapes; functional operations inside RS-SQF are timed explicitly.",
    }
    write_outputs(all_rows, metadata, args.out_dir, args.prefix)

    print()
    print("Top RS-SQF internal phases:")
    print("| rank | stage | phase | mean | median | p90 | share |")
    print("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for rank, row in enumerate(all_rows[:30], start=1):
        print(
            f"| {rank} | `{row.stage}` | `{row.phase}` | {row.mean_ms:.4f} | "
            f"{row.median_ms:.4f} | {row.p90_ms:.4f} | {row.share_of_total_mean_pct:.2f}% |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
