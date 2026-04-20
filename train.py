from ultralytics import YOLO
from ultralytics.utils import yaml_load
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


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
    data_cfg = yaml_load("ultralytics/cfg/datasets/RGBT-3M.yaml")
    data_cfg["input_mode"] = args.input_mode

    if args.resume:
        model = YOLO(args.resume)
    elif args.cfg:
        model = YOLO(args.cfg)
    elif args.fusion_stage == "middle" and args.input_mode == "dual_input":
        model = YOLO("yolov12-dual.yaml")  # 双分支中期融合，n scale
    else:
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
            metric_keys = trainer.validator.metrics.keys + trainer.label_loss_items(prefix="val")
            trainer.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))

    model.add_callback("on_train_start", _set_aux_weight)
    model.add_callback("on_train_start", _patch_loss_names)

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

    results = model.train(
        data=data_cfg,
        resume=bool(args.resume),
        epochs=args.epochs if args.epochs is not None else 300,
        imgsz=[480, 640],  # 输入模型的尺寸，也是验证的尺寸
        batch=16,
        workers=0,
        device=0,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        cos_lr=False,
        val_period=2,
    )
