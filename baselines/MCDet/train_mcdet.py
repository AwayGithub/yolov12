# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the MCDet baseline reproduction on RGBT-3M fire/person data."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.MCDet.model import MCDetDualStreamDetectionModel, MCDetYOLO  # noqa: E402
from ultralytics.utils import yaml_load, yaml_save  # noqa: E402


def parse_args():
    """Parse MCDet baseline training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(ROOT / "baselines" / "MCDet" / "cfg" / "MCDet-yolov5n-fire-person.yaml"),
        help="MCDet baseline model YAML.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "ultralytics" / "cfg" / "datasets" / "RGBT-3M-dual-fire-person-local.yaml"),
        help="RGBT-3M fire/person dataset YAML.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="train_fp_MCDet_yolov5n_seed0_cls01_noaug")
    parser.add_argument("--cls", type=float, default=0.1)
    parser.add_argument("--save_period", type=int, default=-1)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--enable_aux_head",
        action="store_true",
        help="Enable the repo's optional P3 RGB/IR aux heads.",
    )
    parser.add_argument("--augment", action="store_true", help="Enable Ultralytics data augmentation. Default is off.")
    return parser.parse_args()


def _copy_training_sources(trainer, train_script_path, cfg_path):
    """Copy baseline sources into the run directory for reproducibility."""
    snapshot_dir = Path(trainer.save_dir) / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        train_script_path,
        cfg_path,
        ROOT / "baselines" / "MCDet" / "model.py",
        ROOT / "baselines" / "MCDet" / "modules.py",
        ROOT / "baselines" / "MCDet" / "losses.py",
    ):
        try:
            shutil.copy2(path, snapshot_dir / Path(path).name)
        except OSError as exc:
            print(f"[source-snapshot] WARNING: failed to copy {path}: {exc}")

    model_yaml = getattr(trainer.model, "yaml", None)
    if model_yaml is not None:
        try:
            yaml_save(snapshot_dir / "model_yaml.yaml", model_yaml)
        except Exception as exc:
            print(f"[source-snapshot] WARNING: failed to save model_yaml.yaml: {exc}")
    print(f"[source-snapshot] Saved MCDet baseline sources to {snapshot_dir}")


def _set_aux_head(trainer, enable_aux_head):
    """Enable or disable the inherited dual-stream auxiliary heads."""
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    if isinstance(model, MCDetDualStreamDetectionModel):
        model.use_aux_head = bool(enable_aux_head)
        model.aux_loss_weight = 0.25
        if not enable_aux_head:
            print("[aux-head] Disabled for MCDet baseline reproduction.")


def _patch_loss_names(trainer):
    """Keep results.csv columns consistent when the dual-stream model has aux-loss slots."""
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    if not isinstance(model, MCDetDualStreamDetectionModel):
        return
    trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", "aux_rgb", "aux_ir")
    metric_keys = (
        trainer.validator.results_csv_keys()
        if hasattr(trainer.validator, "results_csv_keys")
        else trainer.validator.metrics.keys
    )
    metric_keys += trainer.label_loss_items(prefix="val")
    trainer.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))


def _resolve_dataset_paths(data_cfg):
    """Resolve relative train/val paths after loading a dataset YAML as a dict."""
    dataset_root = Path(data_cfg["path"]).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    data_cfg["path"] = str(dataset_root)
    for key in ("train", "val", "test"):
        if key not in data_cfg or data_cfg[key] is None:
            continue
        split_path = Path(data_cfg[key]).expanduser()
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        data_cfg[key] = str(split_path)
    return data_cfg


def main():
    """Run MCDet baseline training."""
    args = parse_args()
    cfg_path = Path(args.cfg).resolve()
    train_script_path = Path(__file__).resolve()

    data_cfg = _resolve_dataset_paths(yaml_load(args.data))
    data_cfg["input_mode"] = "dual_input"

    model = MCDetYOLO(str(cfg_path))
    model.add_callback("on_train_start", lambda trainer: _set_aux_head(trainer, args.enable_aux_head))
    model.add_callback("on_train_start", _patch_loss_names)
    model.add_callback("on_train_start", lambda trainer: _copy_training_sources(trainer, train_script_path, cfg_path))

    train_kwargs = dict(
        data=data_cfg,
        epochs=args.epochs,
        imgsz=[480, 640],
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        name=args.name,
        seed=0,
        deterministic=False,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        cls=args.cls,
        cos_lr=False,
        val_period=2,
        amp=not args.no_amp,
        save_period=args.save_period,
    )
    if not args.augment:
        train_kwargs.update(
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=0,
        )
        print("[augment] Disabled for MCDet baseline reproduction.")

    return model.train(**train_kwargs)


if __name__ == "__main__":
    main()
