import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import yaml_load, yaml_save


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/RGBT-3M.yaml",
        metavar="YAML",
        help="数据集 YAML 路径，默认使用 RGBT-3M 三分类配置",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        metavar="DIR",
        help="RGBT-3M 数据根目录，用于覆盖默认数据集路径",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="dual_input",
        choices=["dual_input", "rgb_input", "ir_input"],
    )
    parser.add_argument(
        "--fusion_stage",
        type=str,
        default="middle",
        choices=["early", "middle"],
        help="early: 6ch 直接拼接输入单分支; middle: 双分支中期融合",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        metavar="YAML",
        help="覆盖默认模型 YAML，例如 yolov12-dual-p2.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="覆盖默认训练轮次（默认 300）",
    )
    parser.add_argument("--device", type=str, default="0", help="训练设备，例如 0、1 或 cpu")
    parser.add_argument("--name", type=str, default=None, help="显式指定 runs/detect 下的实验目录名")
    parser.add_argument("--save_period", type=int, default=-1, help="每隔N个epoch保存checkpoint；小于1时关闭")
    parser.add_argument("--cls", type=float, default=0.5, help="分类损失权重，默认 0.5")
    parser.add_argument("--batch", type=int, default=16, help="训练 batch size，默认 16")
    parser.add_argument("--no_amp", action="store_true", help="关闭 AMP 混合精度训练")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="从指定 checkpoint 继续训练，例如 runs/detect/train/weights/last.pt",
    )
    parser.add_argument(
        "--aux_loss_weight",
        type=float,
        default=0.25,
        metavar="W",
        help="RGB/IR 辅助检测头损失权重（dual_stream 模式下生效，默认 0.25）",
    )
    parser.add_argument(
        "--disable_aux_head",
        action="store_true",
        help="关闭 RGB/IR 辅助检测头（P3 aux head），跳过其前向与损失（默认启用）",
    )
    parser.add_argument(
        "--grad_debug",
        action="store_true",
        help="打印前 N 个 step 的梯度范数，用于检查跨模态路径的梯度流",
    )
    parser.add_argument(
        "--grad_debug_steps",
        type=int,
        default=10,
        metavar="N",
        help="每个 epoch 打印前 N 个 step 的梯度范数（默认 10）",
    )
    parser.add_argument(
        "--gradient_probe",
        action="store_true",
        help="在固定验证 probe set 上运行只读类别梯度竞争诊断",
    )
    parser.add_argument("--gradient_probe_small_period", type=int, default=2)
    parser.add_argument("--gradient_probe_small_per_stratum", type=int, default=12)
    parser.add_argument("--gradient_probe_full_per_stratum", type=int, default=64)
    parser.add_argument("--gradient_probe_batch", type=int, default=2)
    parser.add_argument(
        "--gradient_probe_full_epochs",
        type=str,
        default="4,10,20,28,32,36,40,50,60,80,100,120,140,160,180,190,200",
    )
    return parser.parse_args()


def _resolve_model_cfg_path(cfg):
    """Resolve a model YAML path from an absolute path, relative path, or Ultralytics model config name."""
    if not cfg:
        return None
    cfg_path = Path(cfg)
    if cfg_path.is_file():
        return cfg_path
    local_path = Path.cwd() / cfg
    if local_path.is_file():
        return local_path
    matches = sorted((Path.cwd() / "ultralytics" / "cfg" / "models").rglob(cfg_path.name))
    return matches[0] if matches else None


def _copy_training_sources(trainer, train_script_path, cfg_path):
    """Copy train.py and the active model config into the run directory for reproducibility."""
    snapshot_dir = Path(trainer.save_dir) / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(train_script_path, snapshot_dir / Path(train_script_path).name)
    except OSError as e:
        print(f"[source-snapshot] WARNING: failed to copy train.py: {e}")

    if cfg_path is not None:
        try:
            shutil.copy2(cfg_path, snapshot_dir / Path(cfg_path).name)
        except OSError as e:
            print(f"[source-snapshot] WARNING: failed to copy cfg {cfg_path}: {e}")

    model_yaml = getattr(trainer.model, "yaml", None)
    if model_yaml is not None:
        try:
            yaml_save(snapshot_dir / "model_yaml.yaml", model_yaml)
        except Exception as e:
            print(f"[source-snapshot] WARNING: failed to save model_yaml.yaml: {e}")

    print(f"[source-snapshot] Saved train.py and model cfg snapshot to {snapshot_dir}")


# ---------------------------------------------------------------------------
# 梯度调试 callbacks（仅 --grad_debug 时启用）
# ---------------------------------------------------------------------------

def _register_grad_hooks(trainer):
    """on_train_epoch_start: 每个 epoch 开始时注册梯度钩子。"""
    keywords = ["backbone_rgb", "backbone_ir", "cross_scale", "cmg_modules"]
    n_steps = trainer._grad_debug_steps
    step = [0]
    hooks = []

    for name, param in trainer.model.named_parameters():
        if not param.requires_grad:
            continue
        if not any(k in name for k in keywords):
            continue

        def make_hook(n):
            def hook(grad):
                if step[0] < n_steps:
                    norm = grad.norm().item() if grad is not None else float("nan")
                    print(f"[grad ep={trainer.epoch+1} step={step[0]:3d}] {n:75s} {norm:.3e}")
            return hook

        hooks.append(param.register_hook(make_hook(name)))

    print(f"[grad-hooks] Epoch {trainer.epoch + 1}: {len(hooks)} hooks registered.")
    trainer._grad_debug_state = {"hooks": hooks, "step": step}


def _step_or_remove_grad_hooks(trainer):
    """on_train_batch_end: 计步，达到 N 步后移除钩子（epoch 结束前提前停止打印）。"""
    if not hasattr(trainer, "_grad_debug_state"):
        return
    state = trainer._grad_debug_state
    state["step"][0] += 1
    if state["step"][0] >= trainer._grad_debug_steps:
        for h in state["hooks"]:
            h.remove()
        del trainer._grad_debug_state


def _remove_grad_hooks(trainer):
    """on_train_epoch_end: epoch 结束时清理残余钩子（steps < N 的 epoch）。"""
    if not hasattr(trainer, "_grad_debug_state"):
        return
    for h in trainer._grad_debug_state["hooks"]:
        h.remove()
    del trainer._grad_debug_state


def _print_cross_scale(trainer):
    """on_train_batch_end: 每 100 步打印一次所有 cross_scale 参数的当前值。"""
    if not hasattr(trainer, "_cross_scale_step"):
        trainer._cross_scale_step = 0
    trainer._cross_scale_step += 1
    if trainer._cross_scale_step % 100 != 0:
        return
    values = {
        name: f"{param.item():.5f}"
        for name, param in trainer.model.named_parameters()
        if "cross_scale" in name
    }
    if values:
        vals_str = "  ".join(f"{k}={v}" for k, v in values.items())
        print(f"\n[cross_scale ep={trainer.epoch+1} step={trainer._cross_scale_step}] {vals_str}")


if __name__ == "__main__":
    args = parse_args()
    data_cfg = yaml_load(args.data)
    data_cfg["input_mode"] = args.input_mode
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
        if (data_root / "train.txt").is_file():
            dataset_root = data_root
        elif (data_root / "RGBT-3M" / "train.txt").is_file():
            dataset_root = data_root / "RGBT-3M"
        else:
            raise FileNotFoundError(
                f"Could not find train.txt under {data_root} or {data_root / 'RGBT-3M'}"
            )
        data_cfg["path"] = str(dataset_root)
        data_cfg["train"] = str(dataset_root / "train.txt")
        data_cfg["val"] = str(dataset_root / "val.txt")
        print(f"[data-root] Using dataset root: {dataset_root}")

    train_script_path = Path(__file__).resolve()
    cfg_for_snapshot = None
    if args.resume:
        model = YOLO(args.resume)
    elif args.cfg:
        cfg_for_snapshot = _resolve_model_cfg_path(args.cfg)
        model = YOLO(args.cfg)
    elif args.fusion_stage == "middle" and args.input_mode == "dual_input":
        cfg_for_snapshot = _resolve_model_cfg_path("yolov12-dual.yaml")
        model = YOLO("yolov12-dual.yaml")  # 双分支中期融合，n scale
    else:
        cfg_for_snapshot = _resolve_model_cfg_path("yolov12.yaml")
        model = YOLO("yolov12.yaml")       # 单分支（early fusion 或单模态）

    # 在训练开始前把 aux_loss_weight 写入模型（init_criterion 懒初始化，此时 model.args 已就绪）
    _aux_w = args.aux_loss_weight
    _use_aux = not args.disable_aux_head
    def _set_aux_weight(trainer):
        from ultralytics.nn.tasks import DualStreamDetectionModel
        m = trainer.model
        if hasattr(m, "module"):  # DDP
            m = m.module
        if isinstance(m, DualStreamDetectionModel):
            m.aux_loss_weight = _aux_w
            m.use_aux_head = _use_aux
            if not _use_aux:
                print("[aux-head] Disabled via --disable_aux_head: skipping RGB/IR P3 aux head forward & loss.")

    def _patch_loss_names(trainer):
        from ultralytics.nn.tasks import DualStreamDetectionModel
        m = trainer.model
        if hasattr(m, "module"):
            m = m.module
        if isinstance(m, DualStreamDetectionModel):
            trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", "aux_rgb", "aux_ir")
            # Re-sync trainer.metrics so val-loss columns match new loss_names (5 vs 3).
            # Without this, save_metrics writes 5 train cols but only 3 val cols,
            # creating a CSV header/row column count mismatch on validation epochs.
            metric_keys = (
                trainer.validator.results_csv_keys()
                if hasattr(trainer.validator, "results_csv_keys")
                else trainer.validator.metrics.keys
            )
            metric_keys += trainer.label_loss_items(prefix="val")
            trainer.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))

    model.add_callback("on_train_start", _set_aux_weight)
    model.add_callback("on_train_start", _patch_loss_names)
    model.add_callback(
        "on_train_start",
        lambda trainer: _copy_training_sources(trainer, train_script_path, cfg_for_snapshot),
    )

    if args.grad_debug:
        # 将参数附到 trainer 上，供 callback 读取
        def _inject_config(trainer):
            trainer._grad_debug_steps = args.grad_debug_steps

        model.add_callback("on_train_start", _inject_config)
        model.add_callback("on_train_epoch_start", _register_grad_hooks)
        model.add_callback("on_train_batch_end", _step_or_remove_grad_hooks)
        model.add_callback("on_train_batch_end", _print_cross_scale)
        model.add_callback("on_train_epoch_end", _remove_grad_hooks)
        print(f"[grad-hooks] Debug mode: first {args.grad_debug_steps} steps per epoch.")

    if args.gradient_probe:
        from ultralytics.utils.training_gradient_probe import TrainingGradientProbe

        full_epochs = tuple(int(value) for value in args.gradient_probe_full_epochs.split(",") if value.strip())
        gradient_probe = TrainingGradientProbe(
            small_period=args.gradient_probe_small_period,
            small_per_stratum=args.gradient_probe_small_per_stratum,
            full_per_stratum=args.gradient_probe_full_per_stratum,
            full_epochs=full_epochs,
            batch_size=args.gradient_probe_batch,
            seed=0,
        )
        model.add_callback("on_train_start", gradient_probe.setup)
        model.add_callback("on_fit_epoch_end", gradient_probe.on_fit_epoch_end)
        print(
            "[gradient-probe] Enabled: "
            f"small every {args.gradient_probe_small_period} epochs; full epochs={full_epochs}."
        )

    results = model.train(
        data=data_cfg,
        resume=bool(args.resume),
        epochs=args.epochs if args.epochs is not None else 200,
        imgsz=[480, 640],  # 输入模型的尺寸，也是验证的尺寸
        batch=args.batch,
        workers=0,
        device=args.device,
        name=args.name,
        seed=0,
        deterministic=True,
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
