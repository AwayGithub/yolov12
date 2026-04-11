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
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="从指定 checkpoint 继续训练，例如 runs/detect/train/weights/last.pt",
    )
    parser.add_argument(
        "--grad_debug",
        action="store_true",
        help="打印前 N 个 step 的梯度范数，用于检查跨模态路径的梯度流",
    )
    parser.add_argument(
        "--grad_debug_steps",
        type=int,
        default=100,
        metavar="N",
        help="每个 epoch 打印前 N 个 step 的梯度范数（默认 100）",
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


if __name__ == "__main__":
    args = parse_args()
    data_cfg = yaml_load("ultralytics/cfg/datasets/RGBT-3M.yaml")
    data_cfg["input_mode"] = args.input_mode

    if args.resume:
        model = YOLO(args.resume)
    elif args.fusion_stage == "middle" and args.input_mode == "dual_input":
        model = YOLO("yolov12-dual.yaml")  # 双分支中期融合，n scale
    else:
        model = YOLO("yolov12.yaml")       # 单分支（early fusion 或单模态）

    if args.grad_debug:
        # 将参数附到 trainer 上，供 callback 读取
        def _inject_config(trainer):
            trainer._grad_debug_steps = args.grad_debug_steps

        model.add_callback("on_train_start", _inject_config)
        model.add_callback("on_train_epoch_start", _register_grad_hooks)
        model.add_callback("on_train_batch_end", _step_or_remove_grad_hooks)
        model.add_callback("on_train_epoch_end", _remove_grad_hooks)
        print(f"[grad-hooks] Debug mode: first {args.grad_debug_steps} steps per epoch.")

    results = model.train(
        data=data_cfg,
        resume=bool(args.resume),
        epochs=300,
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
