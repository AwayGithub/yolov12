# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import thop

from ultralytics import YOLO
from ultralytics.nn.tasks import DualStreamDetectionModel  # noqa: F401 — 确保自定义类可被 torch.load 反序列化
from ultralytics.utils import yaml_load


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DUAL_DATA = "ultralytics/cfg/datasets/RGBT-3M.yaml"
DEFAULT_IR_FIRE_PERSON_DATA = "ultralytics/cfg/datasets/RGBT-3M-ir-fire-person.yaml"
INPUT_MODES = {"dual_input", "rgb_input", "ir_input"}


def _checkpoint_train_args(yolo_model):
    """Return training args embedded in a checkpoint, if present."""
    ckpt = getattr(yolo_model, "ckpt", None)
    if not isinstance(ckpt, dict):
        return {}
    train_args = ckpt.get("train_args", {})
    return train_args if isinstance(train_args, dict) else {}


def _resolve_repo_path(path):
    """Resolve a CLI/checkpoint path relative to cwd or this script directory."""
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), SCRIPT_DIR):
        candidate = base / path
        if candidate.exists():
            return candidate
    return SCRIPT_DIR / path


def _first_conv_in_channels(yolo_model):
    """Infer the number of channels expected by the first convolution."""
    model = getattr(yolo_model, "model", yolo_model)
    yaml_ch = getattr(model, "yaml", {}).get("ch") if hasattr(model, "yaml") else None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module.in_channels
    return yaml_ch


def _expects_dual_input(yolo_model):
    """Return whether the loaded checkpoint expects a 6-channel RGBT tensor."""
    model = getattr(yolo_model, "model", yolo_model)
    if isinstance(model, DualStreamDetectionModel) or type(model).__name__ == "DualStreamDetectionModel":
        return True
    return _first_conv_in_channels(yolo_model) == 6


def _looks_like_ir_checkpoint(args, yolo_model):
    """Heuristic fallback for old checkpoints without complete train_args."""
    train_args = _checkpoint_train_args(yolo_model)
    haystack = " ".join(
        str(x)
        for x in (
            getattr(args, "weights", ""),
            train_args.get("model", ""),
            train_args.get("data", ""),
            train_args.get("name", ""),
        )
    ).lower()
    return "ir" in haystack or "thermal" in haystack or "fire_person" in haystack


def _looks_like_rgb_checkpoint(args, yolo_model):
    """Heuristic fallback for old checkpoints without complete train_args."""
    train_args = _checkpoint_train_args(yolo_model)
    haystack = " ".join(
        str(x)
        for x in (
            getattr(args, "weights", ""),
            train_args.get("model", ""),
            train_args.get("data", ""),
            train_args.get("name", ""),
        )
    ).lower()
    return "rgb" in haystack or "visible" in haystack or "vis" in haystack


def resolve_data_path(args, yolo_model):
    """Resolve the validation dataset YAML, preferring checkpoint provenance over legacy defaults."""
    if args.data:
        return _resolve_repo_path(args.data)

    train_data = _checkpoint_train_args(yolo_model).get("data")
    if train_data:
        return _resolve_repo_path(train_data)

    if not _expects_dual_input(yolo_model) and _looks_like_ir_checkpoint(args, yolo_model):
        return _resolve_repo_path(DEFAULT_IR_FIRE_PERSON_DATA)

    return _resolve_repo_path(DEFAULT_DUAL_DATA)


def load_validation_data_cfg(data_path):
    """Load a validation dataset YAML from a resolved path."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config does not exist: {data_path}")
    return yaml_load(str(data_path))


def resolve_input_mode(args, yolo_model, data_cfg):
    """Resolve input mode, avoiding 6-channel batches for 3-channel single-stream checkpoints."""
    if args.input_mode != "auto":
        return args.input_mode

    cfg_mode = data_cfg.get("input_mode", "dual_input")
    if _expects_dual_input(yolo_model):
        return "dual_input"
    if cfg_mode in {"rgb_input", "ir_input"}:
        return cfg_mode
    if _looks_like_ir_checkpoint(args, yolo_model):
        return "ir_input"
    if _looks_like_rgb_checkpoint(args, yolo_model):
        return "rgb_input"
    if _first_conv_in_channels(yolo_model) == 3 and cfg_mode == "dual_input":
        raise ValueError(
            "Cannot auto-select rgb_input or ir_input for this 3-channel checkpoint. "
            "Pass --input_mode ir_input or --input_mode rgb_input explicitly."
        )
    return cfg_mode if cfg_mode in INPUT_MODES else "dual_input"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv12 RGBT model")
    parser.add_argument(
        "--weights",
        type=str,
        default=r"runs\detect\RGBT-3M\dual_MF\weights\best.pt",
        help="Path to the trained model weights"
    )
    parser.add_argument(
        "--fusion_stage",
        type=str,
        default="middle",
        choices=["early", "middle"],
        help="early: 6ch 单分支; middle: 双分支中期融合",
    )
    parser.add_argument(
        "--input_mode",
        "--input-mode",
        type=str,
        default="auto",
        choices=["auto", "dual_input", "rgb_input", "ir_input"],
        help="Input mode for validation (auto, dual_input, rgb_input, ir_input)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset configuration yaml. Defaults to the data YAML saved in the checkpoint."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[480, 640],
        help="Image size for inference [H, W]"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size for validation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use for inference (e.g., '0' or 'cpu')"
    )
    return parser.parse_args()

def patch_validator_plot_batches(validator):
    """
    YOLO 默认在验证时只保存前3个 batch 的可视化结果。
    通过这个回调函数，我们可以强制它保存所有 batch 的验证结果图。
    """
    if not hasattr(validator, "patched_plot_batches"):
        # 将 plot_batches 设置为所有 batch 的索引
        validator.plot_batches = range(len(validator.dataloader))
        validator.patched_plot_batches = True

if __name__ == "__main__":
    args = parse_args()

    # 1. 加载训练好的模型，以便优先使用 checkpoint 中保存的训练配置
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)

    # 2. 加载并修改数据集配置以支持指定的 input_mode
    data_path = resolve_data_path(args, model)
    data_cfg = load_validation_data_cfg(data_path)
    input_mode = resolve_input_mode(args, model, data_cfg)
    data_cfg["input_mode"] = input_mode

    print(f"Using dataset config: {data_path}")
    print(f"Using input_mode: {input_mode}")
    
    # 3. 注册回调函数以保存所有图片的检测结果图
    model.add_callback("on_val_batch_start", patch_validator_plot_batches)

    # 4. 获取模型信息（参数量、GFLOPs）并立即输出
    n_l, n_p, n_g, flops = model.model.info(verbose=True, imgsz=args.imgsz)
    # model.info() 内部用首个参数的 shape[1] 构造输入，对双分支模型只有 3ch 导致 forward 失败返回 0
    # 这里用正确的 6ch 输入手动计算
    if flops == 0.0:
        try:
            m = deepcopy(model.model).cpu()
            stride = max(int(m.stride.max()), 32) if hasattr(m, "stride") else 32
            input_channels = 6 if _expects_dual_input(model) or input_mode == "dual_input" else 3
            im = torch.empty((1, input_channels, stride, stride))
            flops = thop.profile(m, inputs=[im], verbose=False)[0] / 1e9 * 2
            flops = flops * args.imgsz[0] / stride * args.imgsz[1] / stride
        except Exception:
            print("Warning: 无法计算 GFLOPs（可能是旧版权重与新代码不兼容）")
            flops = 0.0
        finally:
            del m
            torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("Model Info:")
    print(f"  Parameters: {n_p:,}")
    print(f"  GFLOPs:     {flops:.2f}")
    print("="*50 + "\n")

    # 5. 运行验证
    print(f"Running validation with input_mode={input_mode}...")
    
    # 存储每次推理的速度信息
    all_speeds = []
    
    # 第一次运行：完整验证（保存图片、计算指标）
    print("Run 1/10 (Full Validation)...")
    metrics = model.val(
        data=data_cfg,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,   # 确保生成图表
        save=True,    # 确保保存结果
        workers=0,    # 数据加载线程数，设置为0以避免 Windows 上 multiprocessing 的 WinError 1455
    )
    if hasattr(metrics, "speed"):
        all_speeds.append(metrics.speed)

    # 后续9次运行：仅推理（不保存图片，不打印详细日志）
    print("\nRunning 9 additional inference runs for average timing...")
    # 移除之前的回调，避免后续运行中不必要的 plotting 操作（虽然 plots=False 应该会阻止大部分）
    # model.callbacks["on_val_batch_start"].remove(patch_validator_plot_batches) # 这是一个可能的优化，但在 ultralytics 中直接操作 callbacks 可能比较复杂，且 plots=False 应该足够

    for i in range(9):
        print(f"Run {i+2}/10...")
        m = model.val(
            data=data_cfg,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=False,  # 不保存图片
            save=False,   # 不保存结果
            workers=0,    # 数据加载线程数，设置为0以避免 Windows 上 multiprocessing 的 WinError 1455
            verbose=False # 减少日志输出
        )
        if hasattr(m, "speed"):
            all_speeds.append(m.speed)
    
    # 6. 输出各项指标和平均推理时间
    print("\n" + "="*50)
    print("Final Validation Results (Averaged over 10 runs for speed):")
    print("="*50)

    # 获取指标 (使用第一次运行的完整结果)
    if hasattr(metrics, "results_dict"):
        results_dict = metrics.results_dict
        print(f"mAP50:      {results_dict.get('metrics/mAP50(B)', 0.0):.5f}")
        print(f"mAP50-95:   {results_dict.get('metrics/mAP50-95(B)', 0.0):.5f}")
        print(f"Precision:  {results_dict.get('metrics/precision(B)', 0.0):.5f}")
        print(f"Recall:     {results_dict.get('metrics/recall(B)', 0.0):.5f}")
    
    # 计算并输出平均推理时间
    if all_speeds:
        # speed 字典通常包含: preprocess, inference, loss, postprocess
        avg_speed = {}
        keys = all_speeds[0].keys()
        for k in keys:
            avg_speed[k] = sum(s.get(k, 0.0) for s in all_speeds) / len(all_speeds)
        
        print("\nAverage Speed (ms/image):")
        print(f"  Preprocess:  {avg_speed.get('preprocess', 0.0):.4f} ms")
        print(f"  Inference:   {avg_speed.get('inference', 0.0):.4f} ms")
        print(f"  Loss:        {avg_speed.get('loss', 0.0):.4f} ms")
        print(f"  Postprocess: {avg_speed.get('postprocess', 0.0):.4f} ms")
        print(f"  Total:       {sum(avg_speed.values()):.4f} ms")

    print("="*50)
    print(f"All detection result images (from first run) are saved in: {metrics.save_dir}")
