import torch

CKPT_PATH = r"E:\Yan-Unifiles\lab\exp\yolov12\runs\detect\RGBT-3M\rgb_640_all\weights\best.pt"

def main():
    # 这里建议继续用 weights_only=False，因为是你自己的模型
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    model = None

    if isinstance(ckpt, dict):
        print(f"Loaded checkpoint dict with keys: {list(ckpt.keys())}")

        # 1) 先尝试用 ckpt['model']
        model = ckpt.get("model", None)

        # 2) 如果 model 为 None 或不是一个 nn.Module，就退而求其次用 ckpt['ema']
        if model is None or not hasattr(model, "parameters"):
            ema_model = ckpt.get("ema", None)
            if ema_model is None:
                raise RuntimeError("checkpoint 中 model 和 ema 都不是有效的 nn.Module，无法检查 dtype")
            print("Using EMA model from ckpt['ema'] for dtype inspection.")
            model = ema_model
    else:
        # 不是 dict，当作裸模型处理
        model = ckpt
        print("Loaded raw model (非标准字典 checkpoint).")

    # 到这里 model 一定是一个 nn.Module
    dtypes = {p.dtype for p in model.parameters()}
    print(f"Unique dtypes in model parameters: {dtypes}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 打印前几个参数看得更直观
    for name, p in list(model.named_parameters())[:5]:
        print(f"{name}: shape={tuple(p.shape)}, dtype={p.dtype}")

if __name__ == "__main__":
    main()