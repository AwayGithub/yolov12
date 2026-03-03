from ultralytics.data.dataset import FLAME2Dataset
from ultralytics.utils import yaml_load
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 加载配置
cfg = yaml_load("ultralytics/cfg/datasets/FLAME2.yaml")
# 初始化数据集
dataset = FLAME2Dataset(
    img_path=cfg["train"],
    imgsz=cfg["img_size"][0],
    data=cfg,
    task="detect",
    augment=False
)

id = 29378
sample = dataset[id]

# 1. 将 Tensor 转换为 HWC 格式的 Numpy 数组
img_hwc = sample['img'].permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0

# 2. 分离 RGB 和 IR 通道
ir_view = img_hwc[:, :, 0:3][:, :, ::-1]
rgb_view = img_hwc[:, :, 3:6][:, :, ::-1]

# 3. 获取标注框 (Shape: [6, 4]，格式为归一化的 xywh)
bboxes = sample['bboxes'].cpu().numpy()
h, w = img_hwc.shape[:2]

# 4. 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 绘制 RGB 图
ax1.imshow(rgb_view)
ax1.set_title(f"RGB Modality (ID: {id})")
ax1.axis('off')

# 绘制 IR 图并叠加 BBoxes
ax2.imshow(ir_view)
for box in bboxes:
    # 将归一化的 xywh (中心点格式) 转换为像素坐标的 (左上角 x, 左上角 y, 宽, 高)
    xc, yc, bw, bh = box
    x_left = (xc - bw / 2) * w
    y_top = (yc - bh / 2) * h
    rect_w = bw * w
    rect_h = bh * h
    
    # 创建红色矩形框
    rect = patches.Rectangle((x_left, y_top), rect_w, rect_h, 
                             linewidth=2, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

ax2.set_title("IR Modality with BBoxes")
ax2.axis('off')

plt.tight_layout()
plt.show()