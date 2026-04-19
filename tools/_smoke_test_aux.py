"""Quick smoke test for aux head changes."""
import sys
sys.path.insert(0, ".")
import torch
from ultralytics.nn.tasks import DualStreamDetectionModel, DualStreamDetectionLoss

model = DualStreamDetectionModel("ultralytics/cfg/models/v12/yolov12-dual-p2.yaml", verbose=False)

print(f"P4 backbone type : {type(model.backbone_rgb[6]).__name__}")
print(f"aux_head_rgb     : {type(model.aux_head_rgb).__name__}, stride={model.aux_head_rgb.stride}")
print(f"aux_head_ir      : {type(model.aux_head_ir).__name__},  stride={model.aux_head_ir.stride}")
print(f"aux_loss_weight  : {model.aux_loss_weight}")

x = torch.zeros(1, 6, 480, 640)

# Eval forward — aux preds should NOT be set
# Eval forward — aux preds should remain None
model.eval()
with torch.no_grad():
    out = model(x)
assert model._aux_rgb is None, f"Aux preds should be None in eval mode, got {model._aux_rgb}"
print("Eval forward OK, _aux_rgb is None ✓")

# Train forward — aux preds should be set after forward
model.train()
with torch.no_grad():
    out2 = model(x)
assert model._aux_rgb is not None, "Aux preds should be set in train mode"
print(f"Train forward OK, _aux_rgb[0].shape={model._aux_rgb[0].shape} ✓")
print(f"Train forward OK, _aux_ir[0].shape ={model._aux_ir[0].shape}  ✓")

# DualStreamDetectionLoss init (needs model.args which is set during training — mock it)
import types
model.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
from ultralytics.nn.tasks import DualStreamDetectionLoss
loss_fn = DualStreamDetectionLoss(model, aux_weight=0.25)
print("DualStreamDetectionLoss init OK ✓")

print("\nAll checks passed.")
