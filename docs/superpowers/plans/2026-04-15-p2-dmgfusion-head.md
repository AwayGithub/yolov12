# P2 DMGFusion Detection Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stride-4 P2 detection head to `DualStreamDetectionModel`, fusing RGB and IR streams at P2 via the new `DMGFusion` module (differential modality-guided fusion).

**Architecture:** `DMGFusion` replaces the plain `Concat + Conv1×1` at P2, using `|RGB-IR|` to derive per-pixel modality selection weights and a learnable differential amplification gate. `DualStreamDetectionModel` is extended with `FUSION_LAYER_INDICES["p2"]=2` and a pluggable fusion dispatch. A new YAML `yolov12-dual-p2.yaml` defines the 4-scale head.

**Tech Stack:** PyTorch, YOLO v12 ultralytics fork, pytest.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ultralytics/nn/modules/block.py` | Modify | Add `DMGFusion` class; add to `__all__` |
| `ultralytics/nn/modules/__init__.py` | Modify | Import and re-export `DMGFusion` |
| `ultralytics/nn/tasks.py` | Modify | Import `DMGFusion`; extend `DualStreamDetectionModel` |
| `ultralytics/cfg/models/v12/yolov12-dual-p2.yaml` | Create | 4-scale head YAML with `p2_fusion: dmg` |
| `tests/test_cross_modal.py` | Modify | Add tests for `DMGFusion` and 4-scale model |

---

## Task 1: DMGFusion module

**Files:**
- Modify: `ultralytics/nn/modules/block.py` (append after line 1527, the end of `CrossModalA2C2f.forward`)
- Modify: `ultralytics/nn/modules/__init__.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cross_modal.py`:

```python
def test_dmg_fusion_output_shape():
    """DMGFusion preserves spatial dims and channel count."""
    from ultralytics.nn.modules.block import DMGFusion
    m = DMGFusion(channels=64)
    x_rgb = torch.randn(2, 64, 120, 160)
    x_ir  = torch.randn(2, 64, 120, 160)
    out = m(x_rgb, x_ir)
    assert out.shape == (2, 64, 120, 160), f"Expected (2,64,120,160), got {out.shape}"


def test_dmg_fusion_neutral_init():
    """At alpha=0, beta=1, output is a linear function of the mean; gradients reach both inputs."""
    from ultralytics.nn.modules.block import DMGFusion
    m = DMGFusion(channels=32)
    assert m.alpha.item() == 0.0, "alpha must initialise to 0"
    assert m.beta.item()  == 1.0, "beta must initialise to 1"
    x_rgb = torch.randn(2, 32, 8, 8, requires_grad=True)
    x_ir  = torch.randn(2, 32, 8, 8, requires_grad=True)
    out = m(x_rgb, x_ir)
    out.sum().backward()
    assert x_rgb.grad is not None, "gradient must flow to x_rgb"
    assert x_ir.grad  is not None, "gradient must flow to x_ir"


def test_dmg_fusion_param_count():
    """DMGFusion for C=64 should have ≈12K parameters (lightweight)."""
    from ultralytics.nn.modules.block import DMGFusion
    m = DMGFusion(channels=64)
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params < 20_000, f"Too many params: {n_params} (expected < 20K)"
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_cross_modal.py::test_dmg_fusion_output_shape tests/test_cross_modal.py::test_dmg_fusion_neutral_init tests/test_cross_modal.py::test_dmg_fusion_param_count -v
```

Expected: `ImportError: cannot import name 'DMGFusion'`

- [ ] **Step 3: Implement `DMGFusion` in `block.py`**

Append immediately after the closing line of `CrossModalA2C2f.forward` (line 1527):

```python


class DMGFusion(nn.Module):
    """Differential Modality-Guided Fusion for P2 RGB-IR streams (ADR-001 §6, Exp-7).

    Computes a per-pixel modality selection map W from the concatenation of both streams
    and their absolute difference D = |RGB - IR|. Regions of high disagreement are
    amplified by a learnable differential gate alpha, which starts at 0 (neutral) and
    grows only if the training signal warrants it.

    Args:
        channels (int): Channel count C of each input stream.
        diff_hidden_ratio (float): Bottleneck ratio for the difference encoder. Default 0.25.

    Inputs:
        x_rgb (Tensor): (B, C, H, W) — RGB branch P2 features.
        x_ir  (Tensor): (B, C, H, W) — IR  branch P2 features.

    Returns:
        Tensor: (B, C, H, W) fused features.
    """

    def __init__(self, channels: int, diff_hidden_ratio: float = 0.25):
        """Initialise DMGFusion with modality selection and differential encoder branches."""
        super().__init__()
        c_diff = max(8, int(channels * diff_hidden_ratio))
        # Modality selection: [R; I; D] → (B, 2, H, W) softmax weights
        self.sel = nn.Sequential(
            Conv(channels * 3, c_diff, 1),           # 1×1 cross-channel mixing
            Conv(c_diff, c_diff, 3, g=c_diff),        # 3×3 DWConv for local spatial context
            nn.Conv2d(c_diff, 2, 1, bias=True),        # 2-ch logits: {w_rgb, w_ir}
        )
        # Differential amplitude encoder: D → (B, C, H, W) saliency in [0, 1]
        self.diff_enc = nn.Sequential(
            Conv(channels, c_diff, 1),
            nn.Conv2d(c_diff, channels, 1, bias=True),
        )
        # Gate scales — conservative init so the module starts as a plain mean
        self.alpha = nn.Parameter(torch.zeros(1))   # differential amplification; starts at 0
        self.beta  = nn.Parameter(torch.ones(1))    # mean-residual weight; starts at 1
        self.out_proj = Conv(channels, channels, 1)  # output stabilisation

    def forward(self, x_rgb, x_ir):
        """Compute differential-guided fusion of RGB and IR P2 features."""
        D = torch.abs(x_rgb - x_ir)                                           # (B, C, H, W)
        W = torch.softmax(self.sel(torch.cat([x_rgb, x_ir, D], dim=1)), dim=1)  # (B, 2, H, W)
        w_rgb, w_ir = W[:, 0:1], W[:, 1:2]                                     # (B, 1, H, W) each
        S = torch.sigmoid(self.diff_enc(D))                                    # (B, C, H, W)
        modal_selected = w_rgb * x_rgb + w_ir * x_ir                          # (B, C, H, W)
        fused = (1.0 + self.alpha * S) * modal_selected + self.beta * 0.5 * (x_rgb + x_ir)
        return self.out_proj(fused)
```

- [ ] **Step 4: Add `"DMGFusion"` to `block.py` `__all__`**

In `block.py`, the `__all__` tuple starts at line 13. Add `"DMGFusion"` to the end:

```python
    "CrossModalGating",
    "CrossModalA2C2f",
    "DMGFusion",
)
```

- [ ] **Step 5: Export from `ultralytics/nn/modules/__init__.py`**

In the `from .block import (...)` block (line 20–63), add `DMGFusion` after `CrossModalA2C2f`:

```python
    CrossModalGating,
    CrossModalA2C2f,
    DMGFusion,
)
```

And in `__all__` (line 94–168), add after `"CrossModalGating"`:

```python
    "CrossModalGating",
    "DMGFusion",
)
```

- [ ] **Step 6: Run tests — expect PASS**

```
pytest tests/test_cross_modal.py::test_dmg_fusion_output_shape tests/test_cross_modal.py::test_dmg_fusion_neutral_init tests/test_cross_modal.py::test_dmg_fusion_param_count -v
```

Expected: 3 × PASSED

- [ ] **Step 7: Commit**

```bash
git add ultralytics/nn/modules/block.py ultralytics/nn/modules/__init__.py tests/test_cross_modal.py
git commit -m "feat: add DMGFusion module for P2 RGB-IR differential fusion"
```

---

## Task 2: YAML — 4-scale head

**Files:**
- Create: `ultralytics/cfg/models/v12/yolov12-dual-p2.yaml`

No code changes — pure config. No test needed beyond the model-forward test in Task 3.

- [ ] **Step 1: Create the YAML**

```yaml
# YOLOv12-MF-P2: Dual-Stream RGB-IR with P2 detection head + DMGFusion (ADR-001 Exp-7)
# 在 Exp-4c 基础上增加 P2 检测头（stride=4），P2 融合使用 DMGFusion
# Backbone: 与 yolov12-dual.yaml 完全一致（层 0-8）
# P2 融合点: backbone layer 2（C3k2 256, stride=4）

nc: 3           # smoke, fire, person
ch: 6           # 6-channel dual-stream input
dual_stream: true

cmg_stages: [p4, p5]   # Exp-4c: SE channel gating at P4, P5
cma_stages: [p5]        # Exp-4c: bidirectional CMA at P5 only
p2_fusion: dmg          # P2 fusion mode: dmg = DMGFusion, plain = Concat+Conv1x1

scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]

# Backbone — identical to yolov12-dual.yaml (do not modify)
# P2 fusion point = layer 2 output (C3k2 256, stride=4, 120×160 at 480×640 input)
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,   [64, 3, 2]]           # 0-P1/2
  - [-1, 1, Conv,   [128, 3, 2, 1, 2]]    # 1-P2/4
  - [-1, 2, C3k2,   [256, False, 0.25]]   # 2 ← P2 fusion point
  - [-1, 1, Conv,   [256, 3, 2, 1, 4]]    # 3-P3/8
  - [-1, 2, C3k2,   [512, False, 0.25]]   # 4 ← P3 fusion point
  - [-1, 1, Conv,   [512, 3, 2]]          # 5-P4/16
  - [-1, 4, A2C2f,  [512, True, 4]]       # 6 ← P4 fusion point
  - [-1, 1, Conv,   [1024, 3, 2]]         # 7-P5/32
  - [-1, 4, A2C2f,  [1024, True, 1]]      # 8 ← P5 fusion point

# Head — extended FPN+PAN to 4 scales (P2/P3/P4/P5)
# Global layer indices continue from 9 (backbone ends at 8)
head:
  # ── Top-down path (P5→P4→P3→P2) ──────────────────────────────────────────
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 9  — upsample P5
  - [[-1, 6], 1, Concat, [1]]                    # 10 — cat P4 backbone
  - [-1, 2, A2C2f, [512, False, -1]]              # 11 — P4 top-down result

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 12 — upsample P4 top-down
  - [[-1, 4], 1, Concat, [1]]                    # 13 — cat P3 backbone
  - [-1, 2, A2C2f, [256, False, -1]]              # 14 — P3 top-down result

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 15 — upsample P3 top-down
  - [[-1, 2], 1, Concat, [1]]                    # 16 — cat P2 fused (from DualStream y[2])
  - [-1, 2, C3k2, [128, False, 0.25]]             # 17 — P2/4 output ← new

  # ── Bottom-up path (P2→P3→P4→P5) ─────────────────────────────────────────
  - [-1, 1, Conv, [128, 3, 2]]                   # 18 — downsample P2→P3
  - [[-1, 14], 1, Concat, [1]]                   # 19 — cat P3 top-down (layer 14)
  - [-1, 2, A2C2f, [256, False, -1]]              # 20 — P3/8 final

  - [-1, 1, Conv, [256, 3, 2]]                   # 21 — downsample P3→P4
  - [[-1, 11], 1, Concat, [1]]                   # 22 — cat P4 top-down (layer 11)
  - [-1, 2, A2C2f, [512, False, -1]]              # 23 — P4/16 final

  - [-1, 1, Conv, [512, 3, 2]]                   # 24 — downsample P4→P5
  - [[-1, 8], 1, Concat, [1]]                    # 25 — cat P5 fused (from DualStream y[8])
  - [-1, 2, C3k2, [1024, True]]                  # 26 — P5/32 final

  - [[17, 20, 23, 26], 1, Detect, [nc]]           # 27 — Detect(P2, P3, P4, P5)
```

- [ ] **Step 2: Commit**

```bash
git add ultralytics/cfg/models/v12/yolov12-dual-p2.yaml
git commit -m "feat: add yolov12-dual-p2.yaml with 4-scale head"
```

---

## Task 3: Extend `DualStreamDetectionModel`

**Files:**
- Modify: `ultralytics/nn/tasks.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cross_modal.py`:

```python
def test_dual_stream_p2_four_scale_stride():
    """DualStream with p2 YAML produces 4 detection scales with strides [4,8,16,32]."""
    from ultralytics.nn.tasks import DualStreamDetectionModel
    model = DualStreamDetectionModel("yolov12-dual-p2.yaml", nc=3, verbose=False)
    model.eval()
    expected = torch.tensor([4.0, 8.0, 16.0, 32.0])
    assert torch.equal(model.stride.sort().values, expected), \
        f"Expected strides [4,8,16,32], got {model.stride}"


def test_dual_stream_p2_forward_shape():
    """DualStream P2 model forward pass returns valid output for 480×640 input."""
    from ultralytics.nn.tasks import DualStreamDetectionModel
    model = DualStreamDetectionModel("yolov12-dual-p2.yaml", nc=3, verbose=False)
    model.eval()
    x = torch.zeros(1, 6, 480, 640)
    with torch.no_grad():
        out = model(x)
    assert out is not None


def test_dual_stream_p2_uses_dmg_fusion():
    """With p2_fusion=dmg, fusion_convs['p2'] is a DMGFusion instance."""
    from ultralytics.nn.modules.block import DMGFusion
    from ultralytics.nn.tasks import DualStreamDetectionModel
    model = DualStreamDetectionModel("yolov12-dual-p2.yaml", nc=3, verbose=False)
    assert isinstance(model.fusion_convs["p2"], DMGFusion), \
        "fusion_convs['p2'] should be DMGFusion when p2_fusion=dmg"


def test_dual_stream_p2_rejects_cmg_at_p2():
    """Setting cmg_stages=[p2] must raise ValueError (p2 is not a valid CMG stage)."""
    import pytest
    from ultralytics.nn.tasks import DualStreamDetectionModel
    import copy, yaml
    from ultralytics.utils import yaml_load
    cfg = yaml_load("yolov12-dual-p2.yaml")
    cfg["cmg_stages"] = ["p2"]
    with pytest.raises(ValueError, match="p2"):
        DualStreamDetectionModel(cfg, nc=3, verbose=False)
```

- [ ] **Step 2: Run to confirm FAIL**

```
pytest tests/test_cross_modal.py::test_dual_stream_p2_four_scale_stride tests/test_cross_modal.py::test_dual_stream_p2_forward_shape tests/test_cross_modal.py::test_dual_stream_p2_uses_dmg_fusion tests/test_cross_modal.py::test_dual_stream_p2_rejects_cmg_at_p2 -v
```

Expected: 4 × FAIL (model construction errors or wrong strides).

- [ ] **Step 3: Add `DMGFusion` to the import in `tasks.py`**

In `tasks.py` at lines 67–70, extend the import:

```python
    A2C2f,
    CrossModalGating,
    CrossModalA2C2f,
    DMGFusion,
)
```

- [ ] **Step 4: Extend `FUSION_LAYER_INDICES` and add CMG/CMA whitelist**

In `DualStreamDetectionModel`, replace line 404:

```python
    FUSION_LAYER_INDICES = {"p3": 4, "p4": 6, "p5": 8}
```

with:

```python
    FUSION_LAYER_INDICES = {"p2": 2, "p3": 4, "p4": 6, "p5": 8}
    # CMG and CMA are only valid at stages that use A2C2f (area-attention layers).
    # p2 uses C3k2, so applying CMG/CMA there would break the layer-inspection logic.
    _CMG_CMA_VALID_STAGES = frozenset({"p3", "p4", "p5"})
```

- [ ] **Step 5: Replace CMG validation to use whitelist**

In `__init__`, replace the CMG validation block (lines 456–458):

```python
        for stage_name in self._cmg_stages:
            if stage_name not in self.FUSION_LAYER_INDICES:
                raise ValueError(f"cmg_stages 中的 '{stage_name}' 不在 FUSION_LAYER_INDICES")
```

with:

```python
        for stage_name in self._cmg_stages:
            if stage_name not in self._CMG_CMA_VALID_STAGES:
                raise ValueError(
                    f"cmg_stages 中的 '{stage_name}' 不合法，合法值为 {self._CMG_CMA_VALID_STAGES}"
                )
```

And replace the CMA validation block (lines 434–436):

```python
        for stage_name in self._cma_stages:
            if stage_name not in self.FUSION_LAYER_INDICES:
                raise ValueError(f"cma_stages 中的 '{stage_name}' 不在 FUSION_LAYER_INDICES")
```

with:

```python
        for stage_name in self._cma_stages:
            if stage_name not in self._CMG_CMA_VALID_STAGES:
                raise ValueError(
                    f"cma_stages 中的 '{stage_name}' 不合法，合法值为 {self._CMG_CMA_VALID_STAGES}"
                )
```

- [ ] **Step 6: Replace `fusion_convs` construction with pluggable dispatch**

Replace lines 465–469:

```python
        # P3/P4/P5 融合：concat → 1×1 conv 降维
        self.fusion_convs = nn.ModuleDict()
        for stage_name, layer_idx in self.FUSION_LAYER_INDICES.items():
            c_out = self._get_layer_out_channels(self.backbone_rgb[layer_idx])
            self.fusion_convs[stage_name] = Conv(c_out * 2, c_out, 1, 1)
```

with:

```python
        # P2/P3/P4/P5 融合：P2 可选 DMGFusion，其余 concat → 1×1 conv 降维
        _p2_fusion_mode = self.yaml.get("p2_fusion", "plain")
        self.fusion_convs = nn.ModuleDict()
        for stage_name, layer_idx in self.FUSION_LAYER_INDICES.items():
            c_out = self._get_layer_out_channels(self.backbone_rgb[layer_idx])
            if stage_name == "p2" and _p2_fusion_mode == "dmg":
                self.fusion_convs[stage_name] = DMGFusion(c_out)
            else:
                self.fusion_convs[stage_name] = Conv(c_out * 2, c_out, 1, 1)
```

- [ ] **Step 7: Replace fusion dispatch in `_predict_once`**

Replace lines 555–559:

```python
        # concat + 1×1 conv 融合
        fused = {}
        for stage_name in self.FUSION_LAYER_INDICES:
            cat = torch.cat([feats_rgb[stage_name], feats_ir[stage_name]], dim=1)
            fused[stage_name] = self.fusion_convs[stage_name](cat)
```

with:

```python
        # 融合：DMGFusion 接受 (rgb, ir)；其余接受 concat tensor
        fused = {}
        for stage_name in self.FUSION_LAYER_INDICES:
            r, i = feats_rgb[stage_name], feats_ir[stage_name]
            fc = self.fusion_convs[stage_name]
            fused[stage_name] = fc(r, i) if isinstance(fc, DMGFusion) else fc(torch.cat([r, i], dim=1))
```

- [ ] **Step 8: Run tests — expect PASS**

```
pytest tests/test_cross_modal.py::test_dual_stream_p2_four_scale_stride tests/test_cross_modal.py::test_dual_stream_p2_forward_shape tests/test_cross_modal.py::test_dual_stream_p2_uses_dmg_fusion tests/test_cross_modal.py::test_dual_stream_p2_rejects_cmg_at_p2 -v
```

Expected: 4 × PASSED

- [ ] **Step 9: Run full test suite to check no regressions**

```
pytest tests/test_cross_modal.py -v
```

Expected: all existing tests (including `test_dual_stream_cma_forward`, `test_dual_stream_cma_bidirectional`, etc.) still PASS.

- [ ] **Step 10: Commit**

```bash
git add ultralytics/nn/tasks.py tests/test_cross_modal.py
git commit -m "feat: extend DualStreamDetectionModel with P2 DMGFusion 4-scale head"
```

---

## Task 4: Smoke-test end-to-end training

No new code — verify the full pipeline runs without error.

- [ ] **Step 1: 2-epoch smoke train**

```bash
python train.py --input_mode dual_input --cfg yolov12-dual-p2.yaml --epochs 2
```

Expected: training starts, loss decreases each step, val runs after epoch 2, no exceptions.

Specific things to verify in output:
- Model summary lists 4 detection strides: `stride=[4, 8, 16, 32]`
- No `KeyError` in `_predict_once` for `fused["p2"]`
- Val mAP numbers are non-zero

- [ ] **Step 2: Verify P2 head processes correct resolution**

In the training log, look for the model summary line like:
```
                 from  n    params  module
                   -1  1   ...     ultralytics.nn.modules.conv.Conv
```
The first head layer at P2 should report spatial size `120×160` (for 480×640 input).

- [ ] **Step 3: Commit (if any last-minute fixes were needed)**

```bash
git add -p   # stage only relevant changes
git commit -m "fix: <describe fix if any>"
```

---

## Task 5: Update ADR

**Files:**
- Modify: `docs/ADR-001-dual-stream-yolov12-mf.md`

- [ ] **Step 1: Add Exp-7 entries to §3.1 table and §六**

In §3.1 汇总表, append three rows:

```markdown
| Exp-7a | dual + CMG@P4P5 + CMA@P5 + DMGFusion@P2 + 4-scale head | — | — | — | — | ~5.6M | — | 待实验 |
| Exp-7b | dual + CMG@P4P5 + CMA@P5 + plain concat@P2 + 4-scale head | — | — | — | — | ~5.5M | — | 待实验 |
| Exp-7c | Exp-7a，但冻结 alpha=0, beta=1（消融差异门控） | — | — | — | — | ~5.6M | — | 待实验 |
```

In §六 备选想法, append:

```markdown
### 想法 J：DMGFusion@P2 + P2 检测头（当前实验 Exp-7）

**动机（数据驱动）：** fire/person 的 p10 短边 ≈ 9px，在 stride=8 (P3) 下 35–38% 不足 2 格子。添加 stride=4 的 P2 检测头将覆盖率从 62% 提升至 95%（见 §3.0）。

**DMGFusion 核心公式：**
```
D = |RGB - IR|
W = softmax(sel_net([R; I; D]))    # (B, 2, H, W) — 逐像素模态选择权重
S = sigmoid(diff_enc(D))           # (B, C, H, W) — 差异幅度调制
fused = (1 + alpha·S)·(w_rgb·R + w_ir·I) + beta·0.5·(R+I)
```

**参数量（scale n，C=64）：** ~12K（可忽略），P2 head 约 +0.3M。

**消融实验设计：**
- Exp-7a：完整方案（DMGFusion@P2 + 4-scale head）
- Exp-7b：消融 DMGFusion（plain concat + P2 head），验证 P2 head 单独增益
- Exp-7c：冻结 alpha=0, beta=1，验证差异门控是否被主动学习

**实现：** `ultralytics/cfg/models/v12/yolov12-dual-p2.yaml`（`p2_fusion: dmg`）
```

- [ ] **Step 2: Commit**

```bash
git add docs/ADR-001-dual-stream-yolov12-mf.md
git commit -m "docs: add Exp-7 DMGFusion@P2 entries to ADR-001"
```
