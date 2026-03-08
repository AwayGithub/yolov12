from ultralytics.data.dataset import FLAME2Dataset
from ultralytics.utils import yaml_load
import numpy as np
import copy

cfg0 = yaml_load("ultralytics/cfg/datasets/FLAME2.yaml")

for mode in ["dual_input", "rgb_input", "ir_input"]:
    cfg = copy.deepcopy(cfg0)
    cfg["input_mode"] = mode
    ds = FLAME2Dataset(
        img_path=cfg["train"],
        imgsz=cfg["img_size"][0],
        data=cfg,
        task="detect",
        augment=False,
    )

    sample = ds[0]
    img = sample["img"].permute(1, 2, 0).cpu().numpy()  # [H, W, 6]

    print(mode, "shape:", img.shape)

    if mode == "rgb_input":
        # IR 在前 3 个通道，rgb_input 时应被置零
        print("IR 通道 max:", img[:, :, :3].max())
        assert np.allclose(img[:, :, :3], 0.0)
    elif mode == "ir_input":
        # RGB 在后 3 个通道，ir_input 时应被置零
        print("RGB 通道 max:", img[:, :, 3:].max())
        assert np.allclose(img[:, :, 3:], 0.0)
