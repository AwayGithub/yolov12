"""集成测试：DualStreamDetectionModel + bidir_cma_stages: [p5]"""
import sys
sys.path.insert(0, r"E:\Yan-Unifiles\lab\exp\yolov12\.claude\worktrees\magical-burnell")

import torch
from ultralytics.nn.tasks import DualStreamDetectionModel

# 找到 YAML 路径
import os
yaml_path = r"E:\Yan-Unifiles\lab\exp\yolov12\.claude\worktrees\magical-burnell\ultralytics\cfg\models\v12\yolov12-dual-p2.yaml"
assert os.path.exists(yaml_path), f"YAML not found: {yaml_path}"

# 加载 YAML 并注入 bidir_cma_stages
from ultralytics.utils import yaml_load
cfg = yaml_load(yaml_path)
cfg["bidir_cma_stages"] = ["p5"]
cfg["cma_stages"] = []          # 确保不冲突
cfg["cmg_stages"] = []

print("Building model with bidir_cma_stages=[p5] ...")
model = DualStreamDetectionModel(cfg=cfg, nc=3, verbose=False)
model.eval()

# 前向：(B, 6, H, W)，0:3=IR_RGB, 3:6=VIS_RGB
x = torch.randn(1, 6, 480, 640)
with torch.no_grad():
    out = model(x)

if isinstance(out, (list, tuple)):
    print(f"Output: {[o.shape for o in out if hasattr(o, 'shape')]}")
else:
    print(f"Output shape: {out.shape}")

# 参数量对比
n_bidir  = sum(p.numel() for p in model._bidir_cma_modules.parameters())
n_total  = sum(p.numel() for p in model.parameters())
print(f"BidirCrossModalA2C2f params: {n_bidir:,}")
print(f"Total model params:          {n_total:,}")

# 梯度流
model.train()
out_train = model(x)
if isinstance(out_train, (list, tuple)):
    loss = sum(o.sum() for o in out_train if hasattr(o, 'sum'))
else:
    loss = out_train.sum()
loss.backward()
bidir_grads = {n: p.grad for n, p in model._bidir_cma_modules.named_parameters() if p.grad is not None}
print(f"Bidir module grad-bearing params: {len(bidir_grads)}/{sum(1 for _ in model._bidir_cma_modules.parameters())}")
print("INTEGRATION PASS")
