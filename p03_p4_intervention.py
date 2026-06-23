#!/usr/bin/env python3
"""ADR-004 P03: inference-time intervention on B2 P4 Parallel Cross A2C2f.

This script runs the full RGBT-3M validation under a set of controlled
interventions on the P4 `DualParallelCrossA2C2f` module.  It does not retrain
anything; only the learned residual/cross scales are temporarily overwritten
during validation and restored afterwards.

Interventions:
  original          : no change (C0 / B2 control)
  gamma_zero        : gamma_rgb = gamma_ir = 0  (turn off the whole P4 transform)
  gamma_rgb_zero    : gamma_rgb = 0             (turn off RGB residual path)
  gamma_ir_zero     : gamma_ir = 0              (turn off IR residual path)
  cross_zero        : cross_scale_rgb = cross_scale_ir = 0
  cross_rgb_zero    : cross_scale_rgb = 0       (RGB receives no cross guidance)
  cross_ir_zero     : cross_scale_ir = 0        (IR receives no cross guidance)
  cross_scale_0.5   : halve cross-modal delta strength
  cross_scale_2.0   : double cross-modal delta strength
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# Make repo modules importable when script is run directly
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.block import DualParallelCrossA2C2f


DATA_YAML = "ultralytics/cfg/datasets/RGBT-3M.yaml"
DATASET_ROOT = "/data/xwh/dataset/RGBT-3M/RGBT-3M"
WEIGHTS_DEFAULT = "runs/detect/train_MF_DMGInit8dP2_FreAttP3_pcrossP4/weights/last.pt"
IMGSZ = [480, 640]
BATCH = 16
DEVICE = "0"
WORKERS = 0


INTERVENTIONS = {
    "original": {},
    "gamma_zero": {"gamma_rgb": 0.0, "gamma_ir": 0.0},
    "gamma_rgb_zero": {"gamma_rgb": 0.0},
    "gamma_ir_zero": {"gamma_ir": 0.0},
    "cross_zero": {"cross_scale_rgb": 0.0, "cross_scale_ir": 0.0},
    "cross_rgb_zero": {"cross_scale_rgb": 0.0},
    "cross_ir_zero": {"cross_scale_ir": 0.0},
    "cross_scale_0.5": {"cross_scale_rgb": 0.5, "cross_scale_ir": 0.5},
    "cross_scale_2.0": {"cross_scale_rgb": 2.0, "cross_scale_ir": 2.0},
}


def load_data_cfg():
    """Load dataset YAML and patch the root path to the local dataset."""
    from ultralytics.utils import yaml_load
    cfg = yaml_load(DATA_YAML)
    cfg["path"] = DATASET_ROOT
    return cfg


def find_p4_modules(model):
    """Return all DualParallelCrossA2C2f modules in the model."""
    modules = []
    for m in model.model.modules():
        if isinstance(m, DualParallelCrossA2C2f):
            modules.append(m)
    return modules


def repair_module(m):
    """Add missing attributes for checkpoints trained before they existed.

    Old checkpoints only saved gamma_rgb/gamma_ir.  The current forward expects
    cross_scale_rgb/ir and gamma_mode/gamma_max, which conceptually default to
    the free-gamma mode with full cross-modal deltas (scale = 1.0).
    """
    dev = next(m.parameters()).device
    if not hasattr(m, "gamma_mode"):
        m.gamma_mode = "free"
    if not hasattr(m, "gamma_max"):
        m.gamma_max = 0.35
    if not hasattr(m, "cross_scale_rgb"):
        m.cross_scale_rgb = nn.Parameter(torch.tensor(1.0, device=dev))
    if not hasattr(m, "cross_scale_ir"):
        m.cross_scale_ir = nn.Parameter(torch.tensor(1.0, device=dev))
    if not hasattr(m, "stage_concat"):
        m.stage_concat = False


def snapshot_params(modules, keys):
    """Capture current parameter values for restoration."""
    snap = []
    for m in modules:
        snap.append({k: getattr(m, k).detach().clone() for k in keys if hasattr(m, k)})
    return snap


def restore_params(modules, snap):
    """Restore parameter values from a snapshot."""
    for m, s in zip(modules, snap):
        for k, v in s.items():
            with torch.no_grad():
                getattr(m, k).copy_(v)


def apply_intervention(modules, intervention):
    """Temporarily overwrite target parameters."""
    for m in modules:
        for k, v in intervention.items():
            if not hasattr(m, k):
                continue
            with torch.no_grad():
                getattr(m, k).fill_(float(v))


def run_validation(model, save_dir, plots=False):
    """Run RGBT-3M validation and return metrics."""
    data_cfg = load_data_cfg()
    metrics = model.val(
        data=data_cfg,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        plots=plots,
        save=plots,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
    )
    return metrics


def extract_metrics(metrics):
    """Extract all/smoke/fire/person P/R/mAP50/mAP50-95 from validation metrics."""
    box = metrics.box if hasattr(metrics, "box") else metrics

    def _cls(arr, idx):
        return float(arr[idx]) if arr is not None and len(arr) > idx else float("nan")

    return {
        "all_p": float(box.mp),
        "all_r": float(box.mr),
        "all_map50": float(box.map50),
        "all_map50_95": float(box.map),
        "smoke_p": _cls(box.p, 0),
        "smoke_r": _cls(box.r, 0),
        "smoke_map50": _cls(box.ap50, 0),
        "smoke_map50_95": _cls(box.ap, 0),
        "fire_p": _cls(box.p, 1),
        "fire_r": _cls(box.r, 1),
        "fire_map50": _cls(box.ap50, 1),
        "fire_map50_95": _cls(box.ap, 1),
        "person_p": _cls(box.p, 2),
        "person_r": _cls(box.r, 2),
        "person_map50": _cls(box.ap50, 2),
        "person_map50_95": _cls(box.ap, 2),
    }


def main(args):
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading B2 checkpoint: {args.weights}")
    model = YOLO(args.weights)
    p4_modules = find_p4_modules(model)
    print(f"Found {len(p4_modules)} DualParallelCrossA2C2f module(s).")

    for m in p4_modules:
        repair_module(m)

    # Print learned values for the record
    print("\nLearned P4 parameters:")
    for i, m in enumerate(p4_modules):
        gamma_rgb = m.gamma_rgb.item() if hasattr(m, "gamma_rgb") else float("nan")
        gamma_ir = m.gamma_ir.item() if hasattr(m, "gamma_ir") else float("nan")
        cross_rgb = m.cross_scale_rgb.item() if hasattr(m, "cross_scale_rgb") else float("nan")
        cross_ir = m.cross_scale_ir.item() if hasattr(m, "cross_scale_ir") else float("nan")
        print(
            f"  [{i}] gamma_rgb={gamma_rgb:.5f}  gamma_ir={gamma_ir:.5f}  "
            f"cross_scale_rgb={cross_rgb:.5f}  cross_scale_ir={cross_ir:.5f}"
        )

    # Snapshot all parameters that could be touched
    all_keys = {"gamma_rgb", "gamma_ir", "cross_scale_rgb", "cross_scale_ir"}
    original_snap = snapshot_params(p4_modules, all_keys)

    results = []
    for name, intervention in INTERVENTIONS.items():
        print(f"\n{'='*60}\nRunning intervention: {name}\n{'='*60}")
        apply_intervention(p4_modules, intervention)

        save_dir = out_root / name
        try:
            metrics = run_validation(
                model,
                save_dir,
                plots=(name == "original" and args.plots_original),
            )
            record = {"intervention": name, **extract_metrics(metrics)}
        except Exception as e:
            print(f"ERROR during {name}: {e}")
            record = {"intervention": name, "error": str(e)}
        finally:
            restore_params(p4_modules, original_snap)

        results.append(record)
        for k, v in record.items():
            if k != "intervention" and not k.startswith("error"):
                print(f"  {k}: {v:.5f}")

    # Save results
    df = pd.DataFrame(results)
    csv_path = out_root / "p03_intervention_metrics.csv"
    json_path = out_root / "p03_intervention_metrics.json"
    df.to_csv(csv_path, index=False, float_format="%.5f")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADR-004 P03 P4 intervention study")
    parser.add_argument(
        "--weights",
        type=str,
        default=WEIGHTS_DEFAULT,
        help="Path to B2 checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/detect/adr004/pilot/P03_p4_intervention",
        help="Output directory for P03 results",
    )
    parser.add_argument(
        "--plots-original",
        action="store_true",
        help="Generate validation plots only for the original B2 run",
    )
    args = parser.parse_args()
    main(args)
