import argparse
import random
import time
from copy import deepcopy
from pathlib import Path

import torch
import thop

from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.nn.tasks import DualStreamDetectionModel  # noqa: F401 — 确保自定义类可被 torch.load 反序列化

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
        type=str,
        default="dual_input",
        choices=["dual_input", "rgb_input", "ir_input"],
        help="Input mode for validation (dual_input, rgb_input, ir_input)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/RGBT-3M.yaml",
        help="Path to dataset configuration yaml"
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
    
    # 1. 加载并修改数据集配置以支持指定的 input_mode
    data_cfg = yaml_load(args.data)
    data_cfg["input_mode"] = args.input_mode
    
    # 2. 加载训练好的模型
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    
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
            im = torch.empty((1, 6, stride, stride))
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
    print(f"Running validation with input_mode={args.input_mode}...")
    
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
        print(f"mAP50:      {results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
        print(f"mAP50-95:   {results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
        print(f"Precision:  {results_dict.get('metrics/precision(B)', 0.0):.4f}")
        print(f"Recall:     {results_dict.get('metrics/recall(B)', 0.0):.4f}")
    
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
