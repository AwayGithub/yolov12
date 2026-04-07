# ADR-001: 双分支 YOLOv12-MF RGB-红外融合检测架构

**Status:** In Progress
**Date:** 2026-04-04
**Updated:** 2026-04-06
**Deciders:** 研究者本人

---

## Context

当前代码库已实现 **早期融合 (early fusion)** 方案：RGB(3ch) + IR(3ch) 拼接为 6 通道输入，直接送入单个 backbone。这种方式简单但存在根本缺陷——第一层卷积就被迫混合两种物理特性截然不同的模态信号，低层特征提取缺乏模态针对性。

目标是改为 **中期融合 (mid-fusion) 双分支架构**，让每个模态独立提取低层特征，在高语义层级进行跨模态交互和融合。

### 现有代码架构关键约束

| 组件 | 文件 | 核心机制 |
|------|------|---------|
| 模型构建 | `tasks.py:320` | `parse_model()` 将 YAML 解析为 `nn.Sequential`，**线性执行** |
| 前向传播 | `tasks.py:131-158` | `_predict_once()` 逐层遍历 `self.model`，用 `m.f` 索引做跳连 |
| 损失计算 | `tasks.py:291` | `self.forward(batch["img"])` — 只传图像张量 |
| 数据加载 | `dataset.py:666` | `np.concatenate([rgb, ir], axis=-1)` 输出 6ch |
| 模态切片 | `train.py:77` | `preprocess_batch` 中按 `input_mode` 切片 |
| 通道注册 | `train.py:151` | `get_model()` 根据 `input_mode` 传 `ch=6` 或 `ch=3` |

**核心约束：** `parse_model()` 产出的是 `nn.Sequential`，`_predict_once()` 假设单一数据流。双分支需要打破这个假设。

---

## Decision

采用 **Option B: 自定义 DualStreamDetectionModel** 方案，在 Python 层面实现双分支，不修改 YAML 解析器。

---

## Options Considered

### Option A: 扩展 YAML 解析器支持双分支

在 YAML 中定义 `backbone_rgb` 和 `backbone_tir` 两个段，修改 `parse_model()` 支持多分支。

| 维度 | 评估 |
|------|------|
| 复杂度 | **High** — 需改 `parse_model` 核心循环、`_predict_once`、跳连索引系统 |
| 灵活性 | High — 未来可通过 YAML 任意配置融合点 |
| 风险 | High — 索引系统的改动可能破坏现有单模态配置兼容性 |
| 维护性 | Low — `parse_model` 已有 150 行，加双分支逻辑会更难维护 |

**Pros:** 配置驱动，切换融合策略只改 YAML
**Cons:** 改动面太大，容易引入 bug，且对当前研究阶段的需求来说 over-engineering

### Option B: 自定义 DualStreamDetectionModel（推荐）

继承 `DetectionModel`，在 `__init__` 中手动构建两个 backbone 分支 + 融合模块，重写 `_predict_once()`。YAML 仍用标准格式定义单分支结构。

| 维度 | 评估 |
|------|------|
| 复杂度 | **Medium** — 只改一个文件，新增一个类 |
| 灵活性 | Medium — 融合策略在 Python 代码中调整 |
| 风险 | **Low** — 不修改 `parse_model`，完全向后兼容 |
| 维护性 | High — 逻辑集中，易读易改 |

**Pros:** 改动集中、风险小、可快速迭代融合策略
**Cons:** 切换融合策略需要改代码而非 YAML

### Option C: 保持 6ch 早期融合，加跨模态注意力

不改架构，在现有 6ch backbone 的 A2C2f 前加入通道分组交互模块。

| 维度 | 评估 |
|------|------|
| 复杂度 | **Low** — 只加模块 |
| 灵活性 | Low — 本质仍是单分支 |
| 创新性 | Low — 无法论证"双分支"贡献 |

**Pros:** 改动最少
**Cons:** 无法做双分支消融实验，论文创新点不足

---

## 具体实现方案

### 架构总览

```
Input: batch["img"] shape (B, 6, H, W)
         │
    ┌────┴────┐
    │ split   │
    ▼         ▼
 RGB(B,3,H,W)  IR(B,3,H,W)
    │              │
┌───┴───┐    ┌────┴────┐
│ Conv0 │    │ Conv0'  │   Stage 0: P1/2
│ Conv1 │    │ Conv1'  │   Stage 1: P2/4
│ C3k2  │    │ C3k2'   │   Stage 2
│ Conv3 │    │ Conv3'  │   Stage 3: P3/8
│ C3k2  │    │ C3k2'   │   Stage 4           ← P3 特征
│ Conv5 │    │ Conv5'  │   Stage 5: P4/16
│ A2C2f │    │ A2C2f'  │   Stage 6           ← P4 特征
│       │◄──CMG──►│    │   CrossModalGating @ P4
│ Conv7 │    │ Conv7'  │   Stage 7: P5/32
│ A2C2f │    │ A2C2f'  │   Stage 8           ← P5 特征
│       │◄──CMG──►│    │   CrossModalGating @ P5
└───┬───┘    └────┬────┘
    │              │
    ▼              ▼
 Concat + 1×1 Conv (at P3, P4, P5)
    │
    ▼
  Shared Neck (FPN)
    │
    ▼
  Detect Head
```

### 第一步：修改文件清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `ultralytics/nn/modules/block.py` | **新增** | `CrossModalGating` 模块 |
| `ultralytics/nn/modules/__init__.py` | **修改** | 导出新模块 |
| `ultralytics/nn/tasks.py` | **新增** | `DualStreamDetectionModel` 类 |
| `ultralytics/cfg/models/v12/yolov12-dual.yaml` | **新增** | 双分支配置文件 |
| `ultralytics/models/yolo/detect/train.py` | **修改** | `get_model()` 支持双分支模型 |
| `ultralytics/data/dataset.py` | **不改** | 数据管线保持 6ch 输出不变 |
| `train.py` | **修改** | 指定新 YAML |

### 第二步：具体代码改动

#### 2.1 新增 CrossModalGating 模块

**文件:** `ultralytics/nn/modules/block.py` — 追加到文件末尾（约 L1378 之后）

```python
class CrossModalGating(nn.Module):
    """跨模态门控：用 guide 模态生成通道注意力权重，调制 target 模态特征。

    轻量设计：仅 GAP + 1×1 Conv + Sigmoid，参数量 = c * c + c ≈ c²。
    对于 c=64 (nano P4)，参数量仅 4160，几乎不影响计算量。

    与 A2C2f 的分工：
    - A2C2f 负责模态内的全局特征建模（区域注意力）
    - CMG 负责模态间的互补信息选择性传递（通道门控）
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, target, guide):
        """target: 被调制的特征, guide: 提供门控信号的特征"""
        w = self.gate(guide).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return target * w + target  # 残差门控：保底不损失信息
```

**注意:** 使用 `nn.Linear` 而非 `nn.Conv2d(c,c,1)` 是因为 GAP 后空间维度已经是 1×1，语义更清晰。效果完全等价。

#### 2.2 新增 DualStreamDetectionModel

**文件:** `ultralytics/nn/tasks.py` — 在 `DetectionModel` 类之后（约 L345）新增

```python
class DualStreamDetectionModel(DetectionModel):
    """双分支 RGB-IR 中期融合检测模型。

    架构：
    - 两个独立的 backbone（共享结构，独立权重）
    - 在 P4/P5 层级进行 CrossModalGating 跨模态交互
    - P3/P4/P5 特征 concat + 1×1 conv 融合后送入共享 neck+head
    """

    # backbone 中 P3/P4/P5 特征的输出层索引（对应 yolov12.yaml）
    FUSION_LAYER_INDICES = {
        'p3': 4,   # C3k2 输出, 通道数 512 (缩放前)
        'p4': 6,   # A2C2f 输出, 通道数 512 (缩放前)
        'p5': 8,   # A2C2f 输出, 通道数 1024 (缩放前)
    }
    # 在哪些层级做 CrossModalGating（P4 和 P5 有 A2C2f，语义层级高）
    CMG_STAGES = {'p4', 'p5'}

    def __init__(self, cfg="yolov12n.yaml", ch=None, nc=None, verbose=True):
        # 先用 ch=3 构建标准单分支模型（作为结构模板）
        # 不调用 super().__init__() 因为它会用 ch=6 构建
        nn.Module.__init__(self)  # 只初始化 nn.Module

        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc

        # 强制 ch=3，因为每个分支只处理 3 通道
        self.yaml["ch"] = 3

        # 构建完整模型（backbone + neck + head）
        full_model, self.save = parse_model(deepcopy(self.yaml), ch=3, verbose=verbose)

        # 分离 backbone 和 head (neck+detect)
        backbone_end = max(self.FUSION_LAYER_INDICES.values()) + 1  # 9

        # RGB 分支：使用原始 backbone 层
        self.backbone_rgb = nn.Sequential(*list(full_model.children())[:backbone_end])

        # IR 分支：深拷贝一份独立权重的 backbone
        self.backbone_ir = deepcopy(self.backbone_rgb)

        # 保留 backbone 层的元数据 (m.i, m.f, m.type)
        for i, (m_rgb, m_ir) in enumerate(zip(self.backbone_rgb, self.backbone_ir)):
            m_ir.i = m_rgb.i  # 确保索引一致
            m_ir.f = m_rgb.f
            m_ir.type = m_rgb.type

        # 共享的 neck + head
        self.head = nn.Sequential(*list(full_model.children())[backbone_end:])

        # 跨模态门控模块
        self.cmg_modules = nn.ModuleDict()
        for stage_name in self.CMG_STAGES:
            layer_idx = self.FUSION_LAYER_INDICES[stage_name]
            # 获取该层输出通道数
            c_out = self._get_layer_out_channels(self.backbone_rgb[layer_idx])
            self.cmg_modules[stage_name] = nn.ModuleDict({
                'rgb2ir': CrossModalGating(c_out),  # RGB 引导 IR
                'ir2rgb': CrossModalGating(c_out),  # IR 引导 RGB
            })

        # P3/P4/P5 融合层：concat(rgb, ir) → 1×1 conv 降维
        self.fusion_convs = nn.ModuleDict()
        for stage_name, layer_idx in self.FUSION_LAYER_INDICES.items():
            c_out = self._get_layer_out_channels(self.backbone_rgb[layer_idx])
            self.fusion_convs[stage_name] = Conv(c_out * 2, c_out, 1, 1)

        # 构建完整的 self.model 用于兼容（stride 计算等）
        self.model = full_model
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides (与 DetectionModel.__init__ 相同)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            def _forward(x):
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)
            # 用 6 通道假输入计算 stride
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, 6, s, s))])
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)

    @staticmethod
    def _get_layer_out_channels(layer):
        """获取层的输出通道数"""
        if hasattr(layer, 'cv2'):  # C3k2, A2C2f, C2f
            return layer.cv2.conv.out_channels
        elif hasattr(layer, 'conv'):  # Conv
            return layer.conv.out_channels
        else:
            raise ValueError(f"Cannot determine output channels for {type(layer)}")

    def _forward_backbone(self, backbone, x):
        """运行单个 backbone 分支，返回所有层输出"""
        outputs = {}
        y = []
        for m in backbone:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            # 记录融合层的输出
            for stage_name, layer_idx in self.FUSION_LAYER_INDICES.items():
                if m.i == layer_idx:
                    outputs[stage_name] = x
        return outputs, x

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """双分支前向传播"""
        # 1. 分离 RGB 和 IR 输入
        # 经 augment.py _format_img 的 [::-1] 通道反转后:
        #   0:3 = IR_RGB, 3:6 = VIS_RGB (与 train.py 切片约定一致)
        assert x.shape[1] == 6, f"Expected 6-channel input, got {x.shape[1]}"
        x_ir  = x[:, :3, ...]   # 0:3 = IR_RGB
        x_rgb = x[:, 3:, ...]   # 3:6 = VIS_RGB

        # 2. 独立运行两个 backbone
        feats_rgb, _ = self._forward_backbone(self.backbone_rgb, x_rgb)
        feats_ir, _ = self._forward_backbone(self.backbone_ir, x_ir)

        # 3. 跨模态门控交互（在 P4, P5）
        for stage_name in self.CMG_STAGES:
            cmg = self.cmg_modules[stage_name]
            rgb_feat = feats_rgb[stage_name]
            ir_feat = feats_ir[stage_name]
            feats_rgb[stage_name] = cmg['ir2rgb'](rgb_feat, ir_feat)  # IR 引导 RGB
            feats_ir[stage_name] = cmg['rgb2ir'](ir_feat, rgb_feat)   # RGB 引导 IR

        # 4. 特征融合：concat + 1×1 conv
        fused_feats = {}
        for stage_name in self.FUSION_LAYER_INDICES:
            fused = torch.cat([feats_rgb[stage_name], feats_ir[stage_name]], dim=1)
            fused_feats[stage_name] = self.fusion_convs[stage_name](fused)

        # 5. 运行共享 neck + head
        # neck 期望的输入是 backbone 最后一层输出 (P5)
        # 以及通过跳连获取的 P4, P3
        # 需要构造 y 列表来支持 head 中的跳连索引
        y = [None] * (max(self.FUSION_LAYER_INDICES.values()) + 1)
        for stage_name, layer_idx in self.FUSION_LAYER_INDICES.items():
            y[layer_idx] = fused_feats[stage_name]

        x = fused_feats['p5']  # neck 从 P5 开始

        for m in self.head:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)

        return x
```

**关键设计决策解释：**

1. **为什么不修改 `parse_model()`：** `parse_model` 返回的是线性 `nn.Sequential`，其跳连系统通过 `m.f` 索引实现。强行塞入双分支会破坏索引一致性。直接在 Python 层面分离 backbone 更安全。

2. **为什么在 P4/P5 做 CMG 而不是 P3：** P3 特征分辨率大、语义层级低，跨模态交互效果弱且计算量大。P4/P5 的 A2C2f 已经做了模态内的全局注意力建模，此时加 CMG 是"精准的补充"而非"重复的注意力"。


#### 2.3 修改 trainer 的 get_model()

**文件:** `ultralytics/models/yolo/detect/train.py` — 修改 `get_model()` 方法（约 L147-155）

```python
def get_model(self, cfg=None, weights=None, verbose=True):
    """Return a YOLO detection model."""
    input_mode = self.data.get("input_mode", "dual_input")
    ch = 6 if input_mode == "dual_input" else 3

    # 检查 YAML 配置是否指定双分支模式
    yaml_cfg = yaml_model_load(cfg) if isinstance(cfg, str) else cfg
    use_dual_stream = yaml_cfg.get("dual_stream", False) if yaml_cfg else False

    if use_dual_stream and input_mode == "dual_input":
        model = DualStreamDetectionModel(cfg, ch=ch, nc=self.data["nc"],
                                          verbose=verbose and RANK == -1)
    else:
        model = DetectionModel(cfg, ch=ch, nc=self.data["nc"],
                               verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)
    return model
```

#### 2.4 新增双分支 YAML 配置

**文件:** `ultralytics/cfg/models/v12/yolov12-dual.yaml`

```yaml
# YOLOv12-MF: Dual-Stream Multi-modal Fusion
# 双分支 RGB-IR 中期融合检测模型

# Parameters
nc: 3  # smoke, fire, person
ch: 6  # RGB(3) + IR(3) — 数据管线输出 6 通道
dual_stream: true  # 标记使用 DualStreamDetectionModel

scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]

# 单分支结构定义（会被复制为 RGB 和 IR 两个分支）
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]           # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]    # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]    # 2
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]     # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]    # 4 ← P3 特征 (融合点)
  - [-1, 1, Conv,  [512, 3, 2]]           # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]        # 6 ← P4 特征 (融合点 + CMG)
  - [-1, 1, Conv,  [1024, 3, 2]]          # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]       # 8 ← P5 特征 (融合点 + CMG)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]      # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]      # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]      # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]           # 20

  - [[14, 17, 20], 1, Detect, [nc]]
```

#### 2.5 修改 train.py 入口

```python
model = YOLO("yolov12-dual-n.yaml")  # 使用 n 规模先做实验
```

### 第三步：需要注意的实现细节

#### 3.1 `self.save` 索引兼容问题

`parse_model()` 返回的 `self.save` 是需要保存输出的层索引列表（用于跳连）。在 `DualStreamDetectionModel` 中，backbone 层索引和 head 层索引必须保持一致。

**解决方案:** `self.save` 直接从 `parse_model` 获取，不做修改。`_predict_once` 中手动构造 `y` 列表时，确保 `y[i]` 的索引与 head 中 `m.f` 的引用一致。

#### 3.2 权重加载兼容性

预训练权重是单分支的。加载时 RGB 分支可以用原始权重（第一层 Conv 除外，因为 ch 从 6 变 3），IR 分支也可以复用。

```python
# 加载预训练权重的策略
def load_pretrained(self, weights_path):
    """从单分支预训练权重初始化双分支"""
    state = torch.load(weights_path)['model'].state_dict()
    # RGB backbone: 直接加载（第一层除外）
    self.backbone_rgb.load_state_dict(state, strict=False)
    # IR backbone: 也用 RGB 权重初始化（迁移学习）
    self.backbone_ir.load_state_dict(state, strict=False)
    # head: 直接加载
    self.head.load_state_dict(state, strict=False)
```

#### 3.3 显存预估（yolov12n 规模）

| 组件 | 参数量 | 说明 |
|------|--------|------|
| 单分支 backbone | ~1.2M | 原始 yolov12n |
| 双分支 backbone | ~2.4M | ×2 |
| CMG (P4 + P5) | ~0.02M | 几乎可忽略 |
| 融合 1×1 Conv | ~0.4M | P3 + P4 + P5 |
| Neck + Head | ~1.0M | 共享，不变 |
| **总计** | **~3.8M** | 原始 ~2.5M 的 1.5 倍 |

Batch size 2 + 480×640 输入，预计显存 ~4-5 GB（单卡），可在消费级 GPU 上运行。

#### 3.4 通道顺序说明（非 Bug）

数据管线的通道变换链如下：

1. `dataset.py:666` — `np.concatenate([rgb_BGR, thermal_BGR])` → `[VIS_BGR, IR_BGR]` (HWC)
2. `augment.py:2111` — `transpose(2,0,1)` → `[VIS_BGR, IR_BGR]` (CHW)
3. `augment.py:2112` — `img[::-1]`（`bgr=0.0` 时几乎 100% 执行）→ **`[IR_RGB, VIS_RGB]`** (CHW)

第 3 步的 `[::-1]` 沿通道轴反转全部 6 通道，同时完成了 BGR→RGB 转换和模态位置互换。

因此模型实际收到的输入是 **`0:3 = IR_RGB, 3:6 = VIS_RGB`**，与 `train.py` 的切片一致：
```python
x_ir  = x[:, :3, ...]   # 0:3 = IR_RGB
x_rgb = x[:, 3:, ...]   # 3:6 = VIS_RGB
```

在 `DualStreamDetectionModel._predict_once` 中应按此顺序切分。

---

## 实验数据

以下为 YOLOv12n + RGBT-3M 数据集，输入分辨率 480×640 的实验结果：

### 总体结果

| 实验 | 输入配置 | mAP50 | mAP50-95 | Precision | Recall | 参数量 | 推理耗时(ms) | 状态 |
|------|---------|-------|----------|-----------|--------|--------|------------|------|
| **Exp-0** | 单分支 6ch early fusion | 0.919 | 0.605 | 0.914 | 0.877 | 2.51M | 4.51 | 已完成 |
| **Exp-1** | 对称双分支 concat-only (无 CMG) | 0.933 | 0.623 | 0.924 | 0.893 | — | 6.96 | 已完成 |
| **Exp-2** | **对称双分支 + CMG@P4P5** | **0.934** | **0.627** | **0.925** | **0.894** | **4.21M** | 7.13 | **已完成** |
| Exp-5 | RGB-only 3ch | 0.907 | 0.583 | 0.920 | 0.854 | 2.51M | 4.08 | 已完成 |
| Exp-6 | IR-only 3ch | 0.887 | 0.564 | 0.888 | 0.831 | 2.51M | 4.33 | 已完成 |

### Exp-1 分类别结果

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|------|--------|-----------|-----------|--------|-------|----------|
| all | 3366 | 9227 | 0.924 | 0.893 | 0.933 | 0.623 |
| smoke | 2398 | 4086 | 0.937 | 0.914 | 0.946 | 0.735 |
| fire | 2606 | 3401 | 0.933 | 0.890 | 0.935 | 0.575 |
| person | 1243 | 1740 | 0.901 | 0.874 | 0.919 | 0.559 |

### Exp-1 推理速度（10 次平均）

| 阶段 | 耗时 (ms/image) |
|------|----------------|
| Preprocess | 0.3659 |
| Inference | 6.9619 |
| Loss | 0.0010 |
| Postprocess | 0.7176 |
| **Total** | **8.0463** |

### Exp-1 vs Exp-0 对比分析

| 指标 | Exp-0 (early fusion) | Exp-1 (dual-stream) | 差值 | 分析 |
|------|---------------------|--------------------|----|------|
| mAP50 | 0.919 | **0.933** | **+1.4** | 超过预期目标 0.925 |
| mAP50-95 | 0.605 | **0.623** | **+1.8** | 超过预期目标 0.62 |
| Precision | 0.914 | **0.924** | **+1.0** | 双分支减少了误检 |
| Recall | 0.877 | **0.893** | **+1.6** | 双分支找到更多目标 |
| 推理耗时 | 4.51ms | 6.96ms | +2.45ms (+54%) | 双 backbone 的合理代价 |

**结论：对称双分支 concat-only（甚至不加 CMG）已显著超越 early fusion 基准线。** 双分支架构本身的独立特征提取能力是增益的主要来源。

### 分类别增益对比（Exp-1 vs Exp-0）

| 类别 | Exp-0 mAP50 | Exp-1 mAP50 | Δ | Exp-0 mAP50-95 | Exp-1 mAP50-95 | Δ |
|------|------------|------------|---|---------------|---------------|---|
| smoke | 0.937 | **0.946** | +0.9 | 0.717 | **0.735** | +1.8 |
| fire | 0.921 | **0.935** | +1.4 | 0.545 | **0.575** | +3.0 |
| person | 0.899 | **0.919** | +2.0 | 0.542 | **0.559** | +1.7 |

- **Fire 类增益最显著（mAP50-95 +3.0）**：印证了"独立 IR 分支能更好地利用热辐射信息"的假设
- **Person 类 mAP50 提升最多（+2.0）**：双分支分离了纹理（RGB）和热轮廓（IR），互补性充分体现
- **Smoke 类增益也不错（mAP50-95 +1.8）**：即使没有 CMG 门控抑制，独立特征提取也减少了 IR 对 smoke 的干扰

### Exp-2 分类别结果

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|------|--------|-----------|-----------|--------|-------|----------|
| all | 3366 | 9227 | 0.925 | 0.894 | 0.934 | 0.627 |
| smoke | 2398 | 4086 | 0.941 | 0.903 | 0.949 | 0.739 |
| fire | 2606 | 3401 | 0.930 | 0.892 | 0.929 | 0.581 |
| person | 1243 | 1740 | 0.903 | 0.886 | 0.924 | 0.562 |

### Exp-2 推理速度（10 次平均）

| 阶段 | 耗时 (ms/image) |
|------|----------------|
| Preprocess | 0.4898 |
| Inference | 7.1286 |
| Loss | 0.0010 |
| Postprocess | 0.7580 |
| **Total** | **8.3774** |

### Exp-2 vs Exp-1 对比分析（CMG 的边际增益）

| 指标 | Exp-1 (concat) | Exp-2 (+CMG) | Δ | 分析 |
|------|---------------|-------------|---|------|
| mAP50 | 0.933 | 0.934 | **+0.1** | 几乎无提升 |
| mAP50-95 | 0.623 | 0.627 | **+0.4** | 微小提升，定位精度略好 |
| Precision | 0.924 | 0.925 | +0.1 | 基本持平 |
| Recall | 0.893 | 0.894 | +0.1 | 基本持平 |
| 推理耗时 | 6.96ms | 7.13ms | +0.17ms | CMG 计算开销极小 |

**分类别 CMG 增益：**

| 类别 | Exp-1 mAP50 | Exp-2 mAP50 | Δ | Exp-1 mAP50-95 | Exp-2 mAP50-95 | Δ |
|------|------------|------------|---|---------------|---------------|---|
| smoke | 0.946 | **0.949** | +0.3 | 0.735 | **0.739** | +0.4 |
| fire | **0.935** | 0.929 | **-0.6** | 0.575 | **0.581** | +0.6 |
| person | 0.919 | **0.924** | +0.5 | 0.559 | **0.562** | +0.3 |

**Exp-2 关键发现：**

1. **CMG 整体增益极其微小（mAP50 +0.1）**，远低于 0.5 的判断阈值 → 简单 concat 已足够，CMG 不是关键贡献
2. **Smoke 类：** Precision 从 0.937→0.941（+0.4），验证了 CMG 门控能轻微抑制 IR 噪声，但幅度有限
3. **Fire 类出现矛盾信号：** mAP50 下降 0.6（检出略少），但 mAP50-95 上升 0.6（定位更准） → CMG 门控可能过度抑制了部分弱火焰的 IR 信号，同时让保留的信号定位更精准
4. **Person 类：** 一致性提升（mAP50 +0.5, mAP50-95 +0.3），CMG 对多模态互补类别有稳定正向作用
5. **参数量 4.21M**（Exp-0 的 1.68 倍），CMG 仅增加约 0.02M 参数，开销可忽略

**结论：双分支架构本身（Exp-1）是增益的主要来源（mAP50 +1.4），CMG 仅贡献了边际改善（+0.1）。** 论文应重点论证"独立特征提取 + 层级融合"的架构设计，CMG 作为可选增强模块而非核心贡献。

### 早期分析（2026-03-18 周报）

#### 分类别模态差异

| 类别 | 优势模态 | 证据 | 对双分支设计的启示 |
|------|---------|------|------------------|
| **Smoke** | RGB 主导 | RGB mAP50=0.947 与双模态持平；IR 加入后 Precision 从 0.941 降至 0.939，引入轻微误检 | IR 对烟雾检测是**噪声源**，融合时需要门控抑制 IR 对 smoke 的干扰 |
| **Fire** | IR 主导 | IR 的 mAP50-95=0.524 > RGB 0.497；IR Recall=0.831 > RGB 0.825 | IR 的热辐射信息对火焰定位更精准，融合应放大 IR 对 fire 的贡献 |
| **Person** | RGB 略优 | RGB Precision=0.900 > IR 0.876；RGB mAP50-95=0.515 > IR 0.504 | 人体纹理/外观依赖可见光，但差距不如 smoke 明显 |

#### 关键数字

- Early fusion 对比 RGB-only: **mAP50 +1.2, mAP50-95 +2.2, Recall +2.3** — 双模态增益显著且稳定
- Early fusion 对比 IR-only: **mAP50 +3.2, mAP50-95 +4.1** — IR 单独不够
- Early fusion 代价: 推理耗时 +0.43ms (+10.6%)，参数量几乎不变 (+432)
- ~~需要超越的基准线: mAP50 > 0.919, mAP50-95 > 0.605~~ **✓ 已由 Exp-1 超越**

---

## 实验数据对方案设计的启示

### 1. 模态特异性强烈支持非对称双分支

实验数据表明不同类别对模态的依赖差异很大（smoke→RGB, fire→IR）。这直接论证了：
- **非对称 backbone 有物理意义：** RGB 分支保留 A2C2f 注意力（捕捉烟雾纹理的全局关系），IR 分支用纯 C3k2（高效提取热轮廓）
- 这不是为了非对称而非对称，而是 **modality-aware** 的设计

### 2. CMG 的门控机制效果有限（Exp-2 已验证）

~~实验显示 IR 对 smoke 检测引入了误检。CMG 的 Sigmoid 门控天然具备抑制能力。~~

**Exp-2 实测结论：** CMG@P4P5 仅带来 mAP50 +0.1 的边际增益。可能原因：
- 双分支独立提取特征后做 concat，已经天然将模态信息"分离"到不同通道区间，后续 1×1 Conv 融合层本身就具备通道选择能力
- CMG 的简单 GAP→Linear→Sigmoid 门控信号太粗（全局通道级），无法提供空间维度的精细选择
- **启示：** 如果要做跨模态交互，可能需要更强的模块（如空间注意力、交叉注意力），或者接受"简单 concat 已足够"的结论，将论文重心放在架构设计上

### 3. 增益天花板的预判（已完全验证）

- ~~合理的预期目标：mAP50 ≥ 0.925, mAP50-95 ≥ 0.62~~ **✓ Exp-1 已达 0.933/0.623**
- ~~如果 fire 类的 mAP50-95 能从 0.545 提到 0.56+~~ **✓ Exp-1 fire mAP50-95=0.575 (+3.0)**
- ~~新预期：加上 CMG 后 mAP50 能否达到 0.94+？~~ **✗ Exp-2 仅 0.934，CMG 增益极小**
- **核心结论：双分支架构本身贡献了绝大部分增益（Exp-1 +1.4 mAP50），CMG 仅贡献 +0.1**
- 增益主要来自独立特征提取而非跨模态交互，当前架构的天花板大约在 mAP50 ≈ 0.935

### 4. 推理耗时约束（已实测）

- Exp-1（无 CMG）：**6.96ms**，Exp-2（+CMG）：**7.13ms** — CMG 仅增加 0.17ms
- 参数量 4.21M（Exp-0 的 1.68 倍），推理耗时 7.13ms（Exp-0 的 1.58 倍），仍可接受（~140 FPS）
- 后续非对称 backbone（IR 用 C3k2 替代 A2C2f）可能将推理耗时压回 6.5ms 以下

---

## 消融实验设计

| 实验编号 | 配置 | 目的 | 状态 | mAP50 | mAP50-95 |
|---------|------|------|------|-------|----------|
| Exp-0 | 单分支 6ch early fusion | 基准 | **已完成** | 0.919 | 0.605 |
| **Exp-1** | **对称双分支 concat-only (无 CMG)** | **验证双分支本身的增益** | **已完成** | **0.933** | **0.623** |
| **Exp-2** | **对称双分支 + CMG@P4P5** | **验证跨模态门控的增益** | **已完成** | **0.934** | **0.627** |
| Exp-3 | 非对称双分支 concat-only (RGB:A2C2f, IR:C3k2) | 验证模态特异化 backbone + 轻量化 | **下一步** | — | — |
| Exp-4 | 对称双分支 + CMG@P3P4P5 | 验证更多交互点是否有益 | 可选 | — | — |
| Exp-5 | 单分支 RGB-only (ch=3) | 单模态 baseline | **已完成** | 0.907 | 0.583 |
| Exp-6 | 单分支 IR-only (ch=3) | 单模态 baseline | **已完成** | 0.887 | 0.564 |

### 已完成实验的核心发现

1. **双分支架构是核心贡献**：Exp-1 相比 Exp-0 提升 mAP50 +1.4、mAP50-95 +1.8，增益显著
2. **CMG 增益极其有限**：Exp-2 相比 Exp-1 仅提升 mAP50 +0.1、mAP50-95 +0.4，不构成论文核心贡献
3. **Fire 类对双分支最敏感**：mAP50-95 从 Exp-0→Exp-1 +3.0，从 Exp-1→Exp-2 +0.6，验证了独立 IR 分支的价值
4. **参数量代价合理**：4.21M（×1.68），推理 7.13ms（×1.58），仍满足实时要求

### 下一步实验计划

**优先级 1: Exp-3 — 非对称双分支 concat-only (无 CMG)**

- **目的：** 既然 CMG 增益有限，转向探索非对称 backbone 能否在**保持精度的同时压缩参数和推理耗时**
- **配置：** RGB 分支保持 A2C2f（捕捉烟雾纹理的全局关系），IR 分支将 A2C2f 替换为 C3k2（轻量高效提取热轮廓），**不加 CMG**
- **操作：** 新建 `yolov12-dual-asym.yaml`，需修改 `DualStreamDetectionModel` 支持 IR 分支使用不同的 backbone YAML
- **预期效果：**
  - IR 分支去掉 A2C2f 的注意力开销 → 参数量减少、推理提速
  - Fire 类精度至少持平（IR 的热轮廓不需要全局注意力建模）
  - Smoke 类可能略升（IR 分支更简单 = 对 smoke 的干扰更少）
- **关键判断：**
  - 如果 mAP50 ≥ 0.930 且推理 < 6.5ms → 非对称方案成为最优选择（精度持平 + 更快）
  - 如果 mAP50 下降 > 1.0 → 放弃非对称，以 Exp-1/Exp-2 的对称 concat 作为最终方案

**优先级 2: Exp-4 — 对称双分支 + CMG@P3P4P5（可选）**

- **目的：** 验证在 P3 也加 CMG 是否能弥补 Exp-2 中 CMG 增益不足的问题
- **判断依据：** 仅在 Exp-3 结果不理想时执行。如果 Exp-3 已拿到满意结果，跳过此实验
- **风险：** P3 分辨率大（60×80），CMG 的 GAP 会丢失大量空间信息，预期增益不大

### 论文写作方向调整

基于 Exp-1/Exp-2 的实验结论，论文主线应调整为：

1. **核心贡献 = 双分支独立特征提取 + 层级 concat 融合**（Exp-1 已充分验证）
2. CMG 作为消融对照，论证"独立提取比交互融合更重要"的发现（这本身是有价值的 insight）
3. 如果 Exp-3 成功 → 增加"模态感知的非对称设计"作为第二贡献点（轻量化 + 精度保持）
4. 消融故事线：Exp-0 → Exp-1（+双分支） → Exp-2（+CMG，边际）→ Exp-3（非对称，轻量化）

### 实验执行备忘

- 训练超参统一：epochs=300, batch=16, SGD lr0=0.01, lrf=0.01, cos_lr=False, val_period=2
- 注意 CPU RAM 问题：训练到 ~epoch 77 可能触发 `_ArrayMemoryError`，需监控内存并准备 `--resume` 续训
- 每次实验的 val.py 评估统一跑 10 次取平均速度

---

## Consequences

### 变容易的事
- 每种模态有独立的特征提取路径，可以做模态 dropout、模态质量加权等高级操作
- CMG 的门控机制可以自适应抑制特定模态对特定类别的噪声（如 IR 对 smoke 的误检），这是 early fusion 无法做到的
- CMG 模块可以方便替换为更复杂的跨模态注意力（如交叉注意力），支持渐进式研究
- 消融实验清晰：关闭 CMG = Exp-1，非对称 vs 对称 = Exp-3 vs Exp-2

### 变困难的事
- 参数量增加 ~50%（2.5M → ~3.8M），训练时间增加
- 需要维护双分支的权重同步（初始化、学习率等）
- 部署时模型体积更大，不利于无人机轻量化（后期需蒸馏/剪枝）
- 非对称 backbone 增加了 YAML 配置和模型初始化的复杂度

### 后续需要重新审视的事
- ~~如果 CMG 增益不明显，考虑替换为交叉注意力~~ **✓ 已验证 CMG 增益 +0.1 mAP50，确实有限；** 但鉴于增益天花板约 0.935，更复杂的交互模块可能也收益有限，优先转向非对称轻量化（Exp-3）
- 如果非对称双分支效果好，可进一步探索 IR 分支用更窄的通道数（宽度非对称）来压缩参数
- 如果显存紧张，考虑共享低层 backbone 权重（stage 0-2）
- 轻量化部署策略需要单独规划
- 关注 fire 类的 mAP50-95 作为融合质量的敏感指标（IR 主导类别，最能体现融合策略的好坏）

---

## Action Items

1. [x] ~~确认 RGB/IR 通道顺序~~ — 已确认：经 `_format_img` 的 `[::-1]` 反转后 `0:3=IR, 3:6=VIS`，与 `train.py` 一致
2. [x] ~~Exp-0/5/6 baseline 实验~~ — 已完成，early fusion mAP50=0.919 为需要超越的基准
3. [x] ~~实现 `CrossModalGating` 模块并注册到 `__init__.py`~~ — 已完成
4. [x] ~~实现 `DualStreamDetectionModel` 类（支持对称/非对称两种模式）~~ — 已完成，CMG 由 YAML `cmg_stages` 控制
5. [x] ~~创建 `yolov12-dual.yaml`~~ — 已完成
6. [x] ~~修改 `get_model()` 支持双分支~~ — 已完成
7. [x] ~~Exp-1: 对称双分支 concat-only~~ — **mAP50=0.933 (+1.4), mAP50-95=0.623 (+1.8)，显著超越 early fusion**
8. [x] ~~Exp-2: 对称双分支 + CMG@P4P5~~ — **mAP50=0.934 (+0.1), mAP50-95=0.627 (+0.4)，CMG 增益极小**
9. [ ] **Exp-3: 非对称双分支 concat-only** — 需新建 `yolov12-dual-asym.yaml`（IR 分支 A2C2f→C3k2），修改模型支持非对称 backbone
10. [ ] 重点观察非对称方案的推理耗时和 fire 类精度变化
