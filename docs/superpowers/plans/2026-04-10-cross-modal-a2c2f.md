# CrossModalA2C2f (Exp-4a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CrossModalA2C2f — embed cross-modal Q-KV attention inside A2C2f blocks in the RGB backbone, so RGB features can query IR features at token-level spatial precision (Exp-4a: single-direction, Q=RGB KV=IR).

**Architecture:** Add three new modules (`CrossModalAAttn`, `CrossModalABlock`, `CrossModalA2C2f`) to `block.py`. In `DualStreamDetectionModel`, replace the RGB backbone's A2C2f layers (at stages specified by YAML `cma_stages`) with `CrossModalA2C2f`. IR backbone runs first unchanged; RGB backbone's `CrossModalA2C2f` layers receive IR features as KV. Existing CMG (SE gating) remains independently configurable via `cmg_stages`.

**Tech Stack:** PyTorch, existing ultralytics nn module infrastructure (Conv, AAttn, ABlock, A2C2f)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ultralytics/nn/modules/block.py` | Modify | Add `CrossModalAAttn`, `CrossModalABlock`, `CrossModalA2C2f` after `CrossModalGating` |
| `ultralytics/nn/modules/__init__.py` | Modify | Export `CrossModalA2C2f` |
| `ultralytics/nn/tasks.py` | Modify | Import `CrossModalA2C2f`, modify `DualStreamDetectionModel` to support `cma_stages` |
| `ultralytics/cfg/models/v12/yolov12-dual.yaml` | Modify | Add `cma_stages` config field |
| `tests/test_cross_modal.py` | Create | Smoke tests for new modules + integrated model forward pass |

---

### Task 1: Add CrossModalAAttn and CrossModalABlock to block.py

**Files:**
- Modify: `ultralytics/nn/modules/block.py` (after `CrossModalGating` class, ~L1397)

- [ ] **Step 1: Write smoke test for CrossModalAAttn**

Create `tests/test_cross_modal.py`:

```python
import torch
from ultralytics.nn.modules.block import CrossModalAAttn


def test_cross_modal_aattn_shape():
    """CrossModalAAttn should preserve spatial dims, Q from x_self, KV from x_other."""
    m = CrossModalAAttn(dim=64, num_heads=2, area=4)
    x_self = torch.randn(2, 64, 8, 8)
    x_other = torch.randn(2, 64, 8, 8)
    out = m(x_self, x_other)
    assert out.shape == (2, 64, 8, 8)


def test_cross_modal_aattn_area1():
    """area=1 means no spatial partitioning (used at P5)."""
    m = CrossModalAAttn(dim=64, num_heads=2, area=1)
    x_self = torch.randn(2, 64, 4, 4)
    x_other = torch.randn(2, 64, 4, 4)
    out = m(x_self, x_other)
    assert out.shape == (2, 64, 4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cross_modal.py::test_cross_modal_aattn_shape -v`
Expected: FAIL — `ImportError: cannot import name 'CrossModalAAttn'`

- [ ] **Step 3: Implement CrossModalAAttn and CrossModalABlock**

Add to `ultralytics/nn/modules/block.py` after the `CrossModalGating` class (~L1397):

```python
class CrossModalAAttn(nn.Module):
    """Area-Attention cross-modal variant: Q from self modality, K/V from other modality.

    Preserves original AAttn area-partition mechanism for computational efficiency.
    Position encoding (pe) operates on V from the other modality.
    """

    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.q = Conv(dim, all_head_dim, 1, act=False)
        self.kv = Conv(dim, all_head_dim * 2, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

    def forward(self, x_self, x_other):
        B, C, H, W = x_self.shape
        N = H * W

        q = self.q(x_self).flatten(2).transpose(1, 2)
        kv = self.kv(x_other)
        v_spatial = kv[:, C:, :, :]
        pp = self.pe(v_spatial)
        kv = kv.flatten(2).transpose(1, 2)

        if self.area > 1:
            q = q.reshape(B * self.area, N // self.area, C)
            kv = kv.reshape(B * self.area, N // self.area, C * 2)
            B, N, _ = q.shape

        k, v = kv.split([C, C], dim=2)

        q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        max_attn = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_attn)
        attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
        x = (v @ attn.transpose(-2, -1)).permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)


class CrossModalABlock(nn.Module):
    """ABlock cross-modal variant: attention queries across modalities."""

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        self.attn = CrossModalAAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_self, x_other):
        x_self = x_self + self.attn(x_self, x_other)
        x_self = x_self + self.mlp(x_self)
        return x_self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cross_modal.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add ultralytics/nn/modules/block.py tests/test_cross_modal.py
git commit -m "feat: add CrossModalAAttn and CrossModalABlock modules"
```

---

### Task 2: Add CrossModalA2C2f to block.py

**Files:**
- Modify: `ultralytics/nn/modules/block.py` (after `CrossModalABlock`)
- Modify: `ultralytics/nn/modules/__init__.py` (export)

- [ ] **Step 1: Write smoke test for CrossModalA2C2f**

Append to `tests/test_cross_modal.py`:

```python
from ultralytics.nn.modules.block import CrossModalA2C2f


def test_cross_modal_a2c2f_shape():
    """CrossModalA2C2f: front groups self-attn, back groups cross-modal."""
    m = CrossModalA2C2f(c1=128, c2=128, n=2, area=4)
    x_self = torch.randn(2, 128, 8, 8)
    x_other = torch.randn(2, 128, 8, 8)
    out = m(x_self, x_other)
    assert out.shape == (2, 128, 8, 8)


def test_cross_modal_a2c2f_group_split():
    """n=2 should produce 1 self-attn group + 1 cross-modal group."""
    m = CrossModalA2C2f(c1=128, c2=128, n=2, area=4)
    assert len(m.m_self) == 1   # n//2 = 1
    assert len(m.m_cross) == 1  # n - n//2 = 1


def test_cross_modal_a2c2f_p5_config():
    """P5 config: c=256, area=1."""
    m = CrossModalA2C2f(c1=256, c2=256, n=2, area=1)
    x_self = torch.randn(2, 256, 4, 4)
    x_other = torch.randn(2, 256, 4, 4)
    out = m(x_self, x_other)
    assert out.shape == (2, 256, 4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cross_modal.py::test_cross_modal_a2c2f_shape -v`
Expected: FAIL — `ImportError: cannot import name 'CrossModalA2C2f'`

- [ ] **Step 3: Implement CrossModalA2C2f**

Add to `ultralytics/nn/modules/block.py` after `CrossModalABlock`:

```python
class CrossModalA2C2f(nn.Module):
    """A2C2f cross-modal variant (R-ELAN + Cross-Modal Attention).

    Front n//2 groups use ABlock for intra-modal self-attention,
    back n - n//2 groups use CrossModalABlock for cross-modal attention (Q=self, KV=other).
    nano: n=2 -> 1 self-attn group (2 ABlocks) + 1 cross-modal group (1 CrossModalABlock).
    """

    def __init__(self, c1, c2, n=2, area=4, mlp_ratio=2.0, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Hidden dim must be a multiple of 32."
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv1_other = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        n_self = n // 2
        n_cross = n - n_self

        self.m_self = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2)))
            for _ in range(n_self)
        )
        self.m_cross = nn.ModuleList(
            CrossModalABlock(c_, num_heads, mlp_ratio, area)
            for _ in range(n_cross)
        )

    def forward(self, x_self, x_other):
        feat = self.cv1(x_self)
        feat_other = self.cv1_other(x_other)

        y = [feat]
        for m in self.m_self:
            y.append(m(y[-1]))
        for m in self.m_cross:
            y.append(m(y[-1], feat_other))

        return self.cv2(torch.cat(y, 1))
```

- [ ] **Step 4: Export CrossModalA2C2f**

In `ultralytics/nn/modules/__init__.py`, add `CrossModalA2C2f` to the import from `.block`:

```python
from .block import (
    ...
    CrossModalGating,
    CrossModalA2C2f,   # <-- add
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cross_modal.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add ultralytics/nn/modules/block.py ultralytics/nn/modules/__init__.py tests/test_cross_modal.py
git commit -m "feat: add CrossModalA2C2f module (R-ELAN + cross-modal attention)"
```

---

### Task 3: Integrate CrossModalA2C2f into DualStreamDetectionModel

**Files:**
- Modify: `ultralytics/nn/tasks.py:14-68` (imports), `ultralytics/nn/tasks.py:394-537` (DualStreamDetectionModel)

- [ ] **Step 1: Write integration test**

Append to `tests/test_cross_modal.py`:

```python
def test_dual_stream_cma_forward():
    """DualStreamDetectionModel with cma_stages should produce valid detections."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual.yaml", nc=3)
    model.eval()
    x = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        out = model(x)
    # Detect head returns list of 3 scale outputs (or a single tensor)
    assert out is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cross_modal.py::test_dual_stream_cma_forward -v`
Expected: FAIL — model doesn't know about `cma_stages` yet

- [ ] **Step 3: Add CrossModalA2C2f import to tasks.py**

In `ultralytics/nn/tasks.py`, add to the import block (~L67-68):

```python
    A2C2f,
    CrossModalGating,
    CrossModalA2C2f,   # <-- add
)
```

- [ ] **Step 4: Modify DualStreamDetectionModel.__init__**

In `ultralytics/nn/tasks.py`, in `DualStreamDetectionModel.__init__`, add the following block **after** `self.backbone_ir = deepcopy(self.backbone_rgb)` (~L428) and **before** `self.head = ...` (~L431):

```python
        # CrossModalA2C2f: replace RGB backbone's A2C2f with cross-modal version (Exp-4a)
        self._cma_stages = set(self.yaml.get("cma_stages", []))
        self._cma_layer_to_stage = {}
        for stage_name in self._cma_stages:
            if stage_name not in self.FUSION_LAYER_INDICES:
                raise ValueError(f"cma_stages 中的 '{stage_name}' 不在 FUSION_LAYER_INDICES")
            layer_idx = self.FUSION_LAYER_INDICES[stage_name]
            old_layer = self.backbone_rgb[layer_idx]
            c1 = old_layer.cv1.conv.in_channels
            c2 = old_layer.cv2.conv.out_channels
            n = len(old_layer.m)
            area = old_layer.m[0][0].attn.area
            new_layer = CrossModalA2C2f(c1, c2, n=n, area=area)
            new_layer.i = old_layer.i
            new_layer.f = old_layer.f
            new_layer.type = f"{CrossModalA2C2f.__module__}.{CrossModalA2C2f.__name__}"
            self.backbone_rgb[layer_idx] = new_layer
            self._cma_layer_to_stage[layer_idx] = stage_name
```

- [ ] **Step 5: Modify _forward_backbone to support cross-modal features**

Replace the `_forward_backbone` method:

```python
    def _forward_backbone(self, backbone, x, cross_feats=None):
        """Run a backbone branch, collecting fusion-point outputs.

        Args:
            cross_feats: if provided, dict of {stage_name: tensor} from the other
                         modality. CrossModalA2C2f layers will use these as KV input.
        """
        feats = {}
        y = []
        for m in backbone:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, CrossModalA2C2f) and cross_feats is not None:
                stage = self._cma_layer_to_stage[m.i]
                x = m(x, cross_feats[stage])
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)
            for stage_name, layer_idx in self.FUSION_LAYER_INDICES.items():
                if m.i == layer_idx:
                    feats[stage_name] = x
        return feats
```

- [ ] **Step 6: Modify _predict_once execution order**

In `_predict_once`, change the backbone execution from:

```python
        feats_rgb = self._forward_backbone(self.backbone_rgb, x_rgb)
        feats_ir  = self._forward_backbone(self.backbone_ir,  x_ir)
```

to:

```python
        # IR backbone runs first (standard self-attention, no cross-modal)
        feats_ir  = self._forward_backbone(self.backbone_ir,  x_ir)
        # RGB backbone with cross-modal attention using IR features
        feats_rgb = self._forward_backbone(self.backbone_rgb, x_rgb, cross_feats=feats_ir)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_cross_modal.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add ultralytics/nn/tasks.py tests/test_cross_modal.py
git commit -m "feat: integrate CrossModalA2C2f into DualStreamDetectionModel (cma_stages)"
```

---

### Task 4: Update YAML config and final verification

**Files:**
- Modify: `ultralytics/cfg/models/v12/yolov12-dual.yaml`

- [ ] **Step 1: Add cma_stages to YAML config**

In `yolov12-dual.yaml`, add the `cma_stages` field after `cmg_stages`:

```yaml
cmg_stages: [p4, p5]  # Exp-2: CMG@P4P5
cma_stages: [p4, p5]  # Exp-4a: CrossModalA2C2f@P4P5 (RGB: Q=RGB KV=IR)
```

- [ ] **Step 2: Write config combination test**

Append to `tests/test_cross_modal.py`:

```python
def test_dual_stream_cma_and_cmg_combined():
    """cma_stages and cmg_stages should work simultaneously."""
    from ultralytics.nn.tasks import DualStreamDetectionModel

    model = DualStreamDetectionModel("yolov12-dual.yaml", nc=3)
    model.eval()
    # Verify both module types exist
    assert len(model.cmg_modules) > 0, "CMG modules should be present"
    assert len(model._cma_layer_to_stage) > 0, "CMA stages should be configured"
    # Forward pass
    x = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out is not None
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_cross_modal.py -v`
Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add ultralytics/cfg/models/v12/yolov12-dual.yaml tests/test_cross_modal.py
git commit -m "feat: enable Exp-4a config (cma_stages + cmg_stages in YAML)"
```
