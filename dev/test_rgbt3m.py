from ultralytics.data.dataset import RGBT3MDataset
from ultralytics.utils import yaml_load
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

cfg = yaml_load("ultralytics/cfg/datasets/RGBT-3M.yaml")
dataset = RGBT3MDataset(
    img_path=cfg["train"],
    imgsz=cfg["img_size"],
    data=cfg,
    task="detect",
    augment=False,
)

id = 0
sample = dataset[id]
img_hwc = sample["img"].permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
ir_view = img_hwc[:, :, 0:3]
rgb_view = img_hwc[:, :, 3:6]
print(ir_view.shape, rgb_view.shape)
bboxes = sample["bboxes"].cpu().numpy()
h, w = img_hwc.shape[:2]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(rgb_view)
ax1.set_title(f"RGB Modality (ID: {id})")
ax1.axis("off")
ax2.imshow(ir_view)
for box in bboxes:
    xc, yc, bw, bh = box
    x_left = (xc - bw / 2) * w
    y_top = (yc - bh / 2) * h
    rect_w = bw * w
    rect_h = bh * h
    rect = patches.Rectangle(
        (x_left, y_top),
        rect_w,
        rect_h,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax2.add_patch(rect)
ax2.set_title("IR Modality with BBoxes")
ax2.axis("off")
plt.tight_layout()
plt.show()
