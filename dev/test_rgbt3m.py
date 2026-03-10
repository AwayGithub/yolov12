import os
import torch
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

def test_rgbt3m_dataset():
    # 模拟 cfg 对象
    class Config:
        def __init__(self):
            self.imgsz = [480, 640]
            self.rect = False
            self.cache = None
            self.single_cls = False
            self.task = "detect"
            self.classes = None
            self.fraction = 1.0
            self.mosaic = 1.0
            self.mixup = 0.0
            self.hsv_h = 0.015
            self.hsv_s = 0.7
            self.hsv_v = 0.4
            self.degrees = 0.0
            self.translate = 0.1
            self.scale = 0.5
            self.shear = 0.0
            self.perspective = 0.0
            self.flipud = 0.0
            self.fliplr = 0.5
            self.mask_ratio = 4
            self.overlap_mask = True
            self.bgr = 0.0
            self.input_mode = "dual_input"

    cfg = Config()
    
    # 加载数据集 YAML
    data_path = r"e:\Yan-Unifiles\lab\exp\yolov12\ultralytics\cfg\datasets\RGBT-3M.yaml"
    data = yaml_load(data_path)
    data["path"] = r"e:\Yan-Unifiles\lab\exp\yolov12\RGBT-3M" # 确保路径正确
    
    # 模拟 img_path (train.txt)
    img_path = os.path.join(data["path"], data["train"])
    
    print(f"开始构建 RGBT3MDataset...")
    dataset = build_yolo_dataset(cfg, img_path, batch=4, data=data, mode="train")
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取第一个样本
    sample = dataset[0]
    img = sample["img"]
    
    print(f"样本图像形状 (C, H, W): {img.shape}")
    print(f"样本标注: {sample['cls']}, {sample['bboxes']}")
    
    # 验证通道数
    assert img.shape[0] == 6, f"预期 6 通道，但得到 {img.shape[0]} 通道"
    print("验证通过: RGBT3MDataset 成功加载 6 通道数据。")

if __name__ == "__main__":
    test_rgbt3m_dataset()
