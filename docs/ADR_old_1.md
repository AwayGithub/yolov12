# ADR-001: 双分支 YOLOv12-MF RGB-红外融合检测架构

**Status:** In Progress
**Date:** 2026-04-04
**Updated:** 2026-04-07
**Deciders:** 研究者本人

***

## Context

当前代码库已实现 **早期融合 (early fusion)** 方案：RGB(3ch) + IR(3ch) 拼接为 6 通道输入，直接送入单个 backbone。这种方式简单但存在根本缺陷——第一层卷积就被迫混合两种物理特性截然不同的模态信号，低层特征提取缺乏模态针对性。

目标是改为 **中期融合 (mid-fusion) 双分支架构**，让每个模态独立提取低层特征，在高语义层级进行跨模态交互和融合。

### 现有代码架构关键约束

| 组件   | 文件                 | 核心机制                                                |
| ---- | ------------------ | --------------------------------------------------- |
| 模型构建 | `tasks.py:320`     | `parse_model()` 将 YAML 解析为 `nn.Sequential`，**线性执行** |
| 前向传播 | `tasks.py:131-158` | `_predict_once()` 逐层遍历 `self.model`，用 `m.f` 索引做跳连   |
| 损失计算 | `tasks.py:291`     | `self.forward(batch["img"])` — 只传图像张量               |
| 数据加载 | `dataset.py:666`   | `np.concatenate([rgb, ir], axis=-1)` 输出 6ch         |
| 模态切片 | `train.py:77`      | `preprocess_batch` 中按 `input_mode` 切片               |
| 通道注册 | `train.py:151`     | `get_model()` 根据 `input_mode` 传 `ch=6` 或 `ch=3`     |

**核心约束：** `parse_model()` 产出的是 `nn.Sequential`，`_predict_once()` 假设单一数据流。双分支需要打破这个假设。

***

## Decision

采用 **Option B: 自定义 DualStreamDetectionModel** 方案，在 Python 层面实现双分支，不修改 YAML 解析器。

***

## Options Considered

### Option A: 扩展 YAML 解析器支持双分支

在 YAML 中定义 `backbone_rgb` 和 `backbone_tir` 两个段，修改 `parse_model()` 支持多分支。

| 维度  | 评估                                                      |
| --- | ------------------------------------------------------- |
| 复杂度 | **High** — 需改 `parse_model` 核心循环、`_predict_once`、跳连索引系统 |
| 灵活性 | High — 未来可通过 YAML 任意配置融合点                               |
| 风险  | High — 索引系统的改动可能破坏现有单模态配置兼容性                            |
| 维护性 | Low — `parse_model` 已有 150 行，加双分支逻辑会更难维护                |

**Pros:** 配置驱动，切换融合策略只改 YAML
**Cons:** 改动面太大，容易引入 bug，且对当前研究阶段的需求来说 over-engineering

### Option B: 自定义 DualStreamDetectionModel（推荐）

继承 `DetectionModel`，在 `__init__` 中手动构建两个 backbone 分支 + 融合模块，重写 `_predict_once()`。YAML 仍用标准格式定义单分支结构。

| 维度  | 评估                                 |
| --- | ---------------------------------- |
| 复杂度 | **Medium** — 只改一个文件，新增一个类          |
| 灵活性 | Medium — 融合策略在 Python 代码中调整        |
| 风险  | **Low** — 不修改 `parse_model`，完全向后兼容 |
| 维护性 | High — 逻辑集中，易读易改                   |

**Pros:** 改动集中、风险小、可快速迭代融合策略
**Cons:** 切换融合策略需要改代码而非 YAML

### Option C: 保持 6ch 早期融合，加跨模态注意力

不改架构，在现有 6ch backbone 的 A2C2f 前加入通道分组交互模块。

| 维度  | 评估                |
| --- | ----------------- |
| 复杂度 | **Low** — 只加模块    |
| 灵活性 | Low — 本质仍是单分支     |
| 创新性 | Low — 无法论证"双分支"贡献 |

**Pros:** 改动最少
**Cons:** 无法做双分支消融实验，论文创新点不足

***

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

| 文件                                             | 改动类型   | 说明                           |
| ---------------------------------------------- | ------ | ---------------------------- |
| `ultralytics/nn/modules/block.py`              | **新增** | `CrossModalGating` 模块        |
| `ultralytics/nn/modules/__init__.py`           | **修改** | 导出新模块                        |
| `ultralytics/nn/tasks.py`                      | **新增** | `DualStreamDetectionModel` 类 |
| `ultralytics/cfg/models/v12/yolov12-dual.yaml` | **新增** | 双分支配置文件                      |
| `ultralytics/models/yolo/detect/train.py`      | **修改** | `get_model()` 支持双分支模型        |
| `ultralytics/data/dataset.py`                  | **不改** | 数据管线保持 6ch 输出不变              |
| `train.py`                                     | **修改** | 指定新 YAML                     |

### 第二步：具体代码改动

#### 2.1 CrossModalGating 模块（SE 式通道门控）

**文件:** `ultralytics/nn/modules/block.py` — 约 L1380

当前实现为 **SE-CMG**（Squeeze-and-Excitation 式跨模态门控），Exp-2 验证有效。CM-CBAM 变体已在 Exp-2b 中否定（mAP50 -0.7、推理 +166%），代码已回退至 SE 版本。

```python
class CrossModalGating(nn.Module):
    """跨模态 SE 门控：用 guide 模态的全局通道统计量生成门控信号，调制 target 模态特征。

    SE (Squeeze-and-Excitation) 式设计：
    - GAP 压缩空间维度 → Linear 生成通道权重 → Sigmoid 门控
    - 残差连接保证信息不丢失：output = target * gate + target
    """

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, target, guide):
        """target: 被调制的特征, guide: 提供注意力信号的特征。"""
        gate = torch.sigmoid(self.fc(self.pool(guide).flatten(1)))  # (B, C)
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return target * gate + target  # 残差门控
```

**设计要点：**

- 所有注意力信号均来自 **guide 模态**（纯跨模态注意力，不是自注意力）
- GAP 沿空间轴规约，内存访问连续，GPU cache 友好
- `tasks.py` 中调用 `CrossModalGating(c_out)` 即可

**参数量（nano 规模，每个实例）：**

| 层级         | 参数量                   |
| ---------- | ---------------------- |
| P4 (128ch) | 128×128+128 = 16,512   |
| P5 (256ch) | 256×256+256 = 65,792   |
| **4 实例合计** | **164,608 (~0.16M)**   |

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

1. **为什么不修改** **`parse_model()`：** `parse_model` 返回的是线性 `nn.Sequential`，其跳连系统通过 `m.f` 索引实现。强行塞入双分支会破坏索引一致性。直接在 Python 层面分离 backbone 更安全。
2. **为什么在 P4/P5 做 CMG 而不是 P3：** P3 特征分辨率大、语义层级低，跨模态交互效果弱且计算量大。P4/P5 的 A2C2f 已经做了模态内的全局注意力建模，此时加 CMG 是"精准的补充"而非"重复的注意力"。

#### 2.3 修改 trainer 的 get\_model()

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

| 组件            | 参数量        | 说明                |
| ------------- | ---------- | ----------------- |
| 单分支 backbone  | \~1.2M     | 原始 yolov12n       |
| 双分支 backbone  | \~2.4M     | ×2                |
| CMG (P4 + P5) | \~0.02M    | 几乎可忽略             |
| 融合 1×1 Conv   | \~0.4M     | P3 + P4 + P5      |
| Neck + Head   | \~1.0M     | 共享，不变             |
| **总计**        | **\~3.8M** | 原始 \~2.5M 的 1.5 倍 |

Batch size 2 + 480×640 输入，预计显存 \~4-5 GB（单卡），可在消费级 GPU 上运行。

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

***

## 实验数据

以下为 YOLOv12n + RGBT-3M 数据集，输入分辨率 480×640 的实验结果：

### 总体结果

| 实验        | 输入配置                      | mAP50     | mAP50-95  | Precision | Recall    | 参数量       | 推理耗时(ms) | 状态      |
| --------- | ------------------------- | --------- | --------- | --------- | --------- | --------- | -------- | ------- |
| **Exp-0** | 单分支 6ch early fusion      | 0.919     | 0.605     | 0.914     | 0.877     | 2.51M     | 4.51     | 已完成     |
| **Exp-1** | 对称双分支 concat-only (无 CMG) | 0.933     | 0.623     | 0.924     | 0.893     | —         | 6.96     | 已完成     |
| **Exp-2** | **对称双分支 + CMG\@P4P5**     | **0.934** | **0.627** | **0.925** | **0.894** | **4.21M** | 7.13     | **已完成** |
| Exp-2b    | 对称双分支 + CM-CBAM\@P4P5     | 0.927     | 0.627     | 0.925     | 0.882     | 4.13M     | 18.91    | 已完成（否定） |
| Exp-5     | RGB-only 3ch              | 0.907     | 0.583     | 0.920     | 0.854     | 2.51M     | 4.08     | 已完成     |
| Exp-6     | IR-only 3ch               | 0.887     | 0.564     | 0.888     | 0.831     | 2.51M     | 4.33     | 已完成     |

### Exp-1 分类别结果

| 类别     | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| ------ | ------ | --------- | --------- | ------ | ----- | -------- |
| all    | 3366   | 9227      | 0.924     | 0.893  | 0.933 | 0.623    |
| smoke  | 2398   | 4086      | 0.937     | 0.914  | 0.946 | 0.735    |
| fire   | 2606   | 3401      | 0.933     | 0.890  | 0.935 | 0.575    |
| person | 1243   | 1740      | 0.901     | 0.874  | 0.919 | 0.559    |

### Exp-1 推理速度（10 次平均）

| 阶段          | 耗时 (ms/image) |
| ----------- | ------------- |
| Preprocess  | 0.3659        |
| Inference   | 6.9619        |
| Loss        | 0.0010        |
| Postprocess | 0.7176        |
| **Total**   | **8.0463**    |

### Exp-1 vs Exp-0 对比分析

| 指标        | Exp-0 (early fusion) | Exp-1 (dual-stream) | 差值             | 分析               |
| --------- | -------------------- | ------------------- | -------------- | ---------------- |
| mAP50     | 0.919                | **0.933**           | **+1.4**       | 超过预期目标 0.925     |
| mAP50-95  | 0.605                | **0.623**           | **+1.8**       | 超过预期目标 0.62      |
| Precision | 0.914                | **0.924**           | **+1.0**       | 双分支减少了误检         |
| Recall    | 0.877                | **0.893**           | **+1.6**       | 双分支找到更多目标        |
| 推理耗时      | 4.51ms               | 6.96ms              | +2.45ms (+54%) | 双 backbone 的合理代价 |

**结论：对称双分支 concat-only（甚至不加 CMG）已显著超越 early fusion 基准线。** 双分支架构本身的独立特征提取能力是增益的主要来源。

### 分类别增益对比（Exp-1 vs Exp-0）

| 类别     | Exp-0 mAP50 | Exp-1 mAP50 | Δ    | Exp-0 mAP50-95 | Exp-1 mAP50-95 | Δ    |
| ------ | ----------- | ----------- | ---- | -------------- | -------------- | ---- |
| smoke  | 0.937       | **0.946**   | +0.9 | 0.717          | **0.735**      | +1.8 |
| fire   | 0.921       | **0.935**   | +1.4 | 0.545          | **0.575**      | +3.0 |
| person | 0.899       | **0.919**   | +2.0 | 0.542          | **0.559**      | +1.7 |

- **Fire 类增益最显著（mAP50-95 +3.0）**：印证了"独立 IR 分支能更好地利用热辐射信息"的假设
- **Person 类 mAP50 提升最多（+2.0）**：双分支分离了纹理（RGB）和热轮廓（IR），互补性充分体现
- **Smoke 类增益也不错（mAP50-95 +1.8）**：即使没有 CMG 门控抑制，独立特征提取也减少了 IR 对 smoke 的干扰

### Exp-2 分类别结果

| 类别     | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| ------ | ------ | --------- | --------- | ------ | ----- | -------- |
| all    | 3366   | 9227      | 0.925     | 0.894  | 0.934 | 0.627    |
| smoke  | 2398   | 4086      | 0.941     | 0.903  | 0.949 | 0.739    |
| fire   | 2606   | 3401      | 0.930     | 0.892  | 0.929 | 0.581    |
| person | 1243   | 1740      | 0.903     | 0.886  | 0.924 | 0.562    |

### Exp-2 推理速度（10 次平均）

| 阶段          | 耗时 (ms/image) |
| ----------- | ------------- |
| Preprocess  | 0.4898        |
| Inference   | 7.1286        |
| Loss        | 0.0010        |
| Postprocess | 0.7580        |
| **Total**   | **8.3774**    |

### Exp-2 vs Exp-1 对比分析（CMG 的边际增益）

| 指标        | Exp-1 (concat) | Exp-2 (+CMG) | Δ        | 分析          |
| --------- | -------------- | ------------ | -------- | ----------- |
| mAP50     | 0.933          | 0.934        | **+0.1** | 几乎无提升       |
| mAP50-95  | 0.623          | 0.627        | **+0.4** | 微小提升，定位精度略好 |
| Precision | 0.924          | 0.925        | +0.1     | 基本持平        |
| Recall    | 0.893          | 0.894        | +0.1     | 基本持平        |
| 推理耗时      | 6.96ms         | 7.13ms       | +0.17ms  | CMG 计算开销极小  |

**分类别 CMG 增益：**

| 类别     | Exp-1 mAP50 | Exp-2 mAP50 | Δ        | Exp-1 mAP50-95 | Exp-2 mAP50-95 | Δ    |
| ------ | ----------- | ----------- | -------- | -------------- | -------------- | ---- |
| smoke  | 0.946       | **0.949**   | +0.3     | 0.735          | **0.739**      | +0.4 |
| fire   | **0.935**   | 0.929       | **-0.6** | 0.575          | **0.581**      | +0.6 |
| person | 0.919       | **0.924**   | +0.5     | 0.559          | **0.562**      | +0.3 |

**Exp-2 关键发现：**

1. **CMG 整体增益极其微小（mAP50 +0.1）**，远低于 0.5 的判断阈值 → 简单 concat 已足够，CMG 不是关键贡献
2. **Smoke 类：** Precision 从 0.937→0.941（+0.4），验证了 CMG 门控能轻微抑制 IR 噪声，但幅度有限
3. **Fire 类出现矛盾信号：** mAP50 下降 0.6（检出略少），但 mAP50-95 上升 0.6（定位更准） → CMG 门控可能过度抑制了部分弱火焰的 IR 信号，同时让保留的信号定位更精准
4. **Person 类：** 一致性提升（mAP50 +0.5, mAP50-95 +0.3），CMG 对多模态互补类别有稳定正向作用
5. **参数量 4.21M**（Exp-0 的 1.68 倍），CMG 仅增加约 0.02M 参数，开销可忽略

**结论：双分支架构本身（Exp-1）是增益的主要来源（mAP50 +1.4），CMG 仅贡献了边际改善（+0.1）。** 论文应重点论证"独立特征提取 + 层级融合"的架构设计，CMG 作为可选增强模块而非核心贡献。

**针对 CMG 增益不足的改进（2026-04-07）及 Exp-2b 结论（2026-04-08）：**

原 SE 式 CMG 升级为 **CM-CBAM** 后实验结果如下：

### Exp-2b 分类别结果

| 类别     | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| ------ | ------ | --------- | --------- | ------ | ----- | -------- |
| all    | 3366   | 9227      | 0.925     | 0.882  | 0.927 | 0.627    |
| smoke  | 2398   | 4086      | 0.949     | 0.885  | 0.944 | 0.739    |
| fire   | 2606   | 3401      | 0.925     | 0.887  | 0.925 | 0.578    |
| person | 1243   | 1740      | 0.903     | 0.875  | 0.912 | 0.563    |

### Exp-2b 推理速度（10 次平均）

| 阶段          | 耗时 (ms/image) |
| ----------- | ------------- |
| Preprocess  | 0.5221        |
| Inference   | 16.7253       |
| Loss        | 0.0038        |
| Postprocess | 1.6602        |
| **Total**   | **18.9114**   |

### Exp-2b vs Exp-2 对比分析（CM-CBAM 与 SE-CMG 的对比）

| 指标        | Exp-2 (SE-CMG) | Exp-2b (CM-CBAM) | Δ                   | 分析                           |
| --------- | -------------- | ---------------- | ------------------- | ---------------------------- |
| mAP50     | 0.934          | **0.927**        | **-0.7**            | 精度反而下降                       |
| mAP50-95  | 0.627          | 0.627            | 0.0                 | 持平                           |
| Precision | 0.925          | 0.925            | 0.0                 | 持平                           |
| Recall    | 0.894          | **0.882**        | **-1.2**            | 漏检增多                         |
| 参数量       | 4.22M          | **4.13M**        | -93K                | CM-CBAM 参数更少（bottleneck MLP） |
| 推理耗时      | 7.13ms         | **18.9ms**       | **+11.8ms (+166%)** | **严重退化**                     |

**Exp-2b 关键发现：参数减少 ≠ 延时减少（内存访问模式才是瓶颈）**

推理延时增加 2.65× 的根本原因是空间注意力中的 channel-wise 规约：

```python
avg_sp = torch.mean(guide, dim=1, keepdim=True)   # 沿 channel 轴规约
max_sp = torch.max(guide, dim=1, keepdim=True)[0]  # 沿 channel 轴规约
```

BCHW 张量沿 `dim=1`（channel 轴）规约需要**跨步（strided）内存访问**，对 GPU cache 极不友好。相比之下，原 SE-CMG 的 `AdaptiveAvgPool2d(1)` 是沿空间轴规约，访问连续内存，有高度优化的 CUDA kernel。CM-CBAM 在 P4+P5 双向共执行 4 次此类规约，造成严重的内存带宽瓶颈。参数量节省（来自 MLP bottleneck）完全被访问开销抵消。

**Exp-2b 结论：CM-CBAM 精度下降 + 推理增加 2.65×，不可取。** SE-CMG 在通道注意力上已足够，增加空间注意力不仅没有提升精度，还引入了难以接受的推理开销。CMG 的改进方向应聚焦于降低通道注意力的计算开销（如用 depthwise conv 替代 Linear），而非叠加空间注意力。

### 早期分析（2026-03-18 周报）

#### 分类别模态差异

| 类别         | 优势模态   | 证据                                                              | 对双分支设计的启示                                 |
| ---------- | ------ | --------------------------------------------------------------- | ----------------------------------------- |
| **Smoke**  | RGB 主导 | RGB mAP50=0.947 与双模态持平；IR 加入后 Precision 从 0.941 降至 0.939，引入轻微误检 | IR 对烟雾检测是**噪声源**，融合时需要门控抑制 IR 对 smoke 的干扰 |
| **Fire**   | IR 主导  | IR 的 mAP50-95=0.524 > RGB 0.497；IR Recall=0.831 > RGB 0.825     | IR 的热辐射信息对火焰定位更精准，融合应放大 IR 对 fire 的贡献     |
| **Person** | RGB 略优 | RGB Precision=0.900 > IR 0.876；RGB mAP50-95=0.515 > IR 0.504    | 人体纹理/外观依赖可见光，但差距不如 smoke 明显               |

#### 关键数字

- Early fusion 对比 RGB-only: **mAP50 +1.2, mAP50-95 +2.2, Recall +2.3** — 双模态增益显著且稳定
- Early fusion 对比 IR-only: **mAP50 +3.2, mAP50-95 +4.1** — IR 单独不够
- Early fusion 代价: 推理耗时 +0.43ms (+10.6%)，参数量几乎不变 (+432)
- ~~需要超越的基准线: mAP50 > 0.919, mAP50-95 > 0.605~~ **✓ 已由 Exp-1 超越**

***

## 实验数据对方案设计的启示

### 1. 模态特异性强烈支持非对称双分支

实验数据表明不同类别对模态的依赖差异很大（smoke→RGB, fire→IR）。这直接论证了：

- **非对称 backbone 有物理意义：** RGB 分支保留 A2C2f 注意力（捕捉烟雾纹理的全局关系），IR 分支用纯 C3k2（高效提取热轮廓）
- 这不是为了非对称而非对称，而是 **modality-aware** 的设计

### 2. CMG 的门控机制效果有限（Exp-2 已验证）

~~实验显示 IR 对 smoke 检测引入了误检。CMG 的 Sigmoid 门控天然具备抑制能力。~~

**Exp-2 实测结论：** CMG\@P4P5 仅带来 mAP50 +0.1 的边际增益。可能原因：

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
- 参数量 4.21M（Exp-0 的 1.68 倍），推理耗时 7.13ms（Exp-0 的 1.58 倍），仍可接受（\~140 FPS）
- 后续非对称 backbone（IR 用 C3k2 替代 A2C2f）可能将推理耗时压回 6.5ms 以下

***

## 消融实验设计

| 实验编号       | 配置                                      | 目的                            | 状态      | mAP50     | mAP50-95  |
| ---------- | --------------------------------------- | ----------------------------- | ------- | --------- | --------- |
| Exp-0      | 单分支 6ch early fusion                    | 基准                            | **已完成** | 0.919     | 0.605     |
| **Exp-1**  | **对称双分支 concat-only (无 CMG)**           | **验证双分支本身的增益**                | **已完成** | **0.933** | **0.623** |
| **Exp-2**  | **对称双分支 + CMG\@P4P5**                   | **验证跨模态门控的增益**                | **已完成** | **0.934** | **0.627** |
| **Exp-2b** | **对称双分支 + CM-CBAM\@P4P5**               | **验证 CBAM 式跨模态注意力（通道+空间）的增益** | **已完成** | **0.927** | **0.627** |
| Exp-3      | 非对称双分支 concat-only (RGB:A2C2f, IR:C3k2) | 验证模态特异化 backbone + 轻量化        | 下一步     | —         | —         |
| Exp-4      | 对称双分支 + CrossModalA2C2f\@P4P5           | 验证深度跨模态 Transformer 融合（Q/KV 跨模态）| 计划中    | —         | —         |
| Exp-5      | 单分支 RGB-only (ch=3)                     | 单模态 baseline                  | **已完成** | 0.907     | 0.583     |
| Exp-6      | 单分支 IR-only (ch=3)                      | 单模态 baseline                  | **已完成** | 0.887     | 0.564     |

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

**优先级 2: Exp-4 — CrossModalA2C2f（跨模态 Transformer 融合）**

详见下方「创新点：CrossModalA2C2f」章节。

### 论文写作方向调整

基于 Exp-1/Exp-2/Exp-2b 的实验结论，论文主线应调整为：

1. **核心贡献 = 双分支独立特征提取 + 层级 concat 融合**（Exp-1 已充分验证）
2. CMG/CM-CBAM 作为消融对照，论证"独立提取比后融合门控更重要"的发现
3. 如果 Exp-3 成功 → 增加"模态感知的非对称设计"作为第二贡献点（轻量化 + 精度保持）
4. 如果 Exp-4 成功 → 增加"将跨模态交互内嵌到 backbone 特征提取阶段"作为第三贡献点
5. 消融故事线：Exp-0 → Exp-1（+双分支）→ Exp-2（+CMG 后融合，边际）→ Exp-3（非对称轻量化）→ Exp-4（深度跨模态交互）

### 实验执行备忘

- 训练超参统一：epochs=300, batch=16, SGD lr0=0.01, lrf=0.01, cos\_lr=False, val\_period=2
- 注意 CPU RAM 问题：训练到 \~epoch 77 可能触发 `_ArrayMemoryError`，需监控内存并准备 `--resume` 续训
- 每次实验的 val.py 评估统一跑 10 次取平均速度

***

## Consequences

### 变容易的事

- 每种模态有独立的特征提取路径，可以做模态 dropout、模态质量加权等高级操作
- CMG 的门控机制可以自适应抑制特定模态对特定类别的噪声（如 IR 对 smoke 的误检），这是 early fusion 无法做到的
- CMG 模块可以方便替换为更复杂的跨模态注意力（如交叉注意力），支持渐进式研究
- 消融实验清晰：关闭 CMG = Exp-1，非对称 vs 对称 = Exp-3 vs Exp-2

### 变困难的事

- 参数量增加 \~50%（2.5M → \~3.8M），训练时间增加
- 需要维护双分支的权重同步（初始化、学习率等）
- 部署时模型体积更大，不利于无人机轻量化（后期需蒸馏/剪枝）
- 非对称 backbone 增加了 YAML 配置和模型初始化的复杂度

### 后续需要重新审视的事

- ~~如果 CMG 增益不明显，考虑替换为交叉注意力~~ **✓ Exp-2b 已验证 CM-CBAM 全面劣于 SE-CMG（mAP50 -0.7，推理 +166%）。** 核心教训：channel-wise 空间规约（`torch.max(dim=1)`）对 GPU cache 极不友好，参数量减少不等于推理加速。接受"简单 concat/SE-CMG 已足够"的结论，CMG 不再是研究重点，转向非对称轻量化（Exp-3）
- 如果非对称双分支效果好，可进一步探索 IR 分支用更窄的通道数（宽度非对称）来压缩参数
- 如果显存紧张，考虑共享低层 backbone 权重（stage 0-2）
- 轻量化部署策略需要单独规划
- 关注 fire 类的 mAP50-95 作为融合质量的敏感指标（IR 主导类别，最能体现融合策略的好坏）

***

## Action Items

1. [x] ~~确认 RGB/IR 通道顺序~~ — 已确认：经 `_format_img` 的 `[::-1]` 反转后 `0:3=IR, 3:6=VIS`，与 `train.py` 一致
2. [x] ~~Exp-0/5/6 baseline 实验~~ — 已完成，early fusion mAP50=0.919 为需要超越的基准
3. [x] ~~实现~~ ~~`CrossModalGating`~~ ~~模块并注册到~~ ~~`__init__.py`~~ — 已完成
4. [x] ~~实现~~ ~~`DualStreamDetectionModel`~~ ~~类（支持对称/非对称两种模式）~~ — 已完成，CMG 由 YAML `cmg_stages` 控制
5. [x] ~~创建~~ ~~`yolov12-dual.yaml`~~ — 已完成
6. [x] ~~修改~~ ~~`get_model()`~~ ~~支持双分支~~ — 已完成
7. [x] ~~Exp-1: 对称双分支 concat-only~~ — **mAP50=0.933 (+1.4), mAP50-95=0.623 (+1.8)，显著超越 early fusion**
8. [x] ~~Exp-2: 对称双分支 + CMG\@P4P5~~ — **mAP50=0.934 (+0.1), mAP50-95=0.627 (+0.4)，CMG 增益极小**
9. [x] ~~Exp-2b: 对称双分支 + CM-CBAM\@P4P5~~ — **mAP50=0.927 (-0.7 vs Exp-2), 推理 18.9ms (+166%)，CM-CBAM 全面劣于 SE-CMG，方案否定**
10. [ ] **Exp-3: 非对称双分支 concat-only** — 需新建 `yolov12-dual-asym.yaml`（IR 分支 A2C2f→C3k2），修改模型支持非对称 backbone
11. [ ] 重点观察非对称方案的推理耗时和 fire 类精度变化
12. [ ] **Exp-4: CrossModalA2C2f** — 详见「创新点：CrossModalA2C2f」章节，优先级在 Exp-3 之后

---

## 创新点：CrossModalA2C2f — 将跨模态交互内嵌到 Backbone

### 动机

Exp-2/Exp-2b 的共同局限：CMG 是一个**后置模块**——先各自提取特征，再用另一模态的全局统计量（GAP/GMP）生成粗粒度门控信号，作用于已经完成自注意力建模的特征。这类"打补丁式"的跨模态交互有两个根本缺陷：

1. **信息太晚**：A2C2f 已经在模态内完成了 4 个 ABlock（nano：repeat=4×depth 0.50=2 组，每组 2 个，共 4 个）的自注意力，内部特征分布已经"固化"，后置的通道门控只能做粗调
2. **信号太粗**：CMG 用全局平均（GAP）或全局最大（GMP）把整张特征图压缩成一个向量，完全丢失了空间位置信息

**物理直觉**：IR 对火的探测是高精度的空间信号（哪个像素是热点），不是一个通道向量能表达的。如果能在 A2C2f 内部的 Attention 层中直接用 IR 的空间特征作为 KV，RGB 的 Query 就可以在精确的空间位置上"问"IR："这里有热源吗？"

---

### 模块层次结构

```
A2C2f (当前自注意力)               CrossModalA2C2f（新设计）
─────────────────────              ─────────────────────────────────
cv1: Conv(c1 → c_)                 cv1_self, cv1_other: Conv(c1 → c_)
  │                                     │                    │
  ▼                                     ▼                    ▼
[ABlock, ABlock] × n_groups        [ABlock, ABlock] × 1组  (other 特征缓存)
  │  (Q,K,V 来自 x_self)                │     ← 自注意力组
  │  nano: 2组×2=4个 ABlock             ▼
  │                              CrossModalABlock × 1组
  │                                (Q 来自 x_self, K,V 来自 x_other)
  ▼                                     │     ← 跨模态组
cv2: Conv → output                 cv2: Conv → output
```

**nano 具体数值**（来源：YAML `repeat=4, depth=0.50` → n=2 组，[block.py:1364](ultralytics/nn/modules/block.py#L1364) 每组含 2 个 ABlock）：
- 原 A2C2f：2 组 × 2 = **4 个 ABlock**，全部自注意力
- CrossModalA2C2f：前 **1 组（2 个 ABlock）** 自注意力 + 后 **1 组（2 个 CrossModalABlock）** 跨模态注意力

核心思路：**前 1 组 ABlock 建立模态内自理解，后 1 组 CrossModalABlock 用另一模态精准查询**。

---

### CrossModalAAttn 设计

`AAttn` 的关键代码（[block.py:1211-1213](ultralytics/nn/modules/block.py#L1211)）：
```python
self.qk = Conv(dim, all_head_dim * 2, 1, act=False)  # Q 和 K 来自同一输入
self.v  = Conv(dim, all_head_dim,     1, act=False)
self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)  # 位置编码
```

`CrossModalAAttn` 改动：Q 来自 `x_self`，K/V 来自 `x_other`，位置编码来自 `x_other` 的 V（捕捉 guide 模态的局部空间结构）：

```python
class CrossModalAAttn(nn.Module):
    """Area-Attention 的跨模态变体：Q 来自 self 模态，K/V 来自 other 模态。

    保留原始 AAttn 的 Area 分割机制，确保计算效率不变。
    位置编码 pe 作用于 V (other 模态)，引入 guide 的局部空间偏置。
    """
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * num_heads

        self.q   = Conv(dim, all_head_dim,     1, act=False)  # Q ← x_self
        self.kv  = Conv(dim, all_head_dim * 2, 1, act=False)  # K,V ← x_other
        self.proj = Conv(all_head_dim, dim,    1, act=False)
        self.pe   = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)  # pe ← V(x_other)

    def forward(self, x_self, x_other):
        B, C, H, W = x_self.shape
        N = H * W

        q  = self.q(x_self).flatten(2).transpose(1, 2)   # (B, N, C)
        kv = self.kv(x_other)
        v_spatial = kv[:, C:, :, :]                       # V 的空间形式，用于 pe
        pp = self.pe(v_spatial)                            # 位置偏置来自 guide 模态
        kv = kv.flatten(2).transpose(1, 2)                # (B, N, 2C)

        if self.area > 1:
            q  = q.reshape(B * self.area, N // self.area, C)
            kv = kv.reshape(B * self.area, N // self.area, C * 2)
            B, N, _ = q.shape  # 重新赋值 B=B*area, N=N//area（与原 AAttn 一致）

        k, v = kv.split([C, C], dim=2)

        # Scaled dot-product attention（与原 AAttn 完全对齐）
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
```

---

### CrossModalABlock 与 CrossModalA2C2f

```python
class CrossModalABlock(nn.Module):
    """ABlock 的跨模态变体，attention 改为跨模态查询。"""
    def __init__(self, dim, num_heads, mlp_ratio=2.0, area=1):
        super().__init__()
        self.attn = CrossModalAAttn(dim, num_heads, area)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden, 1),
            Conv(mlp_hidden, dim, 1, act=False)
        )

    def forward(self, x_self, x_other):
        x_self = x_self + self.attn(x_self, x_other)
        x_self = x_self + self.mlp(x_self)
        return x_self


class CrossModalA2C2f(nn.Module):
    """A2C2f 的跨模态变体。

    前 n_groups//2 组（每组含 2 个 ABlock）做模态内自注意力，
    后 n_groups//2 组（每组含 1 个 CrossModalABlock）做跨模态查询。
    nano: n_groups=2 → 前 1 组自注意力，后 1 组跨模态注意力。
    forward(x_self, x_other) 输出 x_self 的增强特征。
    """
    def __init__(self, c1, c2, n=2, area=4, mlp_ratio=2.0, e=0.5):
        # n: A2C2f 的 group 数（已经过 depth scaling），nano=2
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv1_other = Conv(c1, c_, 1, 1)  # x_other 独立的通道映射
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        n_self  = n // 2      # 前半段组数：自注意力（nano=1组）
        n_cross = n - n_self  # 后半段组数：跨模态注意力（nano=1组）

        # 每组含 2 个 ABlock，与原 A2C2f 保持一致
        self.self_blocks = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2)))
            for _ in range(n_self)
        )
        # 跨模态组：每组 1 个 CrossModalABlock（已含 attn + mlp）
        self.cross_blocks = nn.ModuleList(
            CrossModalABlock(c_, num_heads, mlp_ratio, area)
            for _ in range(n_cross)
        )

    def forward(self, x_self, x_other):
        # cv1 将输入压到 hidden dim
        feat = self.cv1(x_self)
        # x_other 也需要相同的 hidden dim 映射（共享 cv1 权重不合理，单独映射）
        feat_other = self.cv1_other(x_other)

        y = [feat]
        for m in self.self_blocks:
            y.append(m(y[-1]))
        for m in self.cross_blocks:
            y.append(m(y[-1], feat_other))

        return self.cv2(torch.cat(y, 1))
```

---

### 与当前架构的集成方式

当前 `_predict_once` 两个 backbone 是顺序执行的：

```python
feats_rgb = self._forward_backbone(self.backbone_rgb, x_rgb)
feats_ir  = self._forward_backbone(self.backbone_ir,  x_ir)
```

集成 CrossModalA2C2f 需要将 backbone 执行拆分为分步执行，在 P4/P5 的 A2C2f 层处交换特征：

```
backbone_rgb[0:6)  ──────────────────────────────►  rgb_pre_p4
backbone_ir [0:6)  ──────────────────────────────►  ir_pre_p4
                                                         │
                          CrossModalA2C2f @ P4:          ▼
                rgb_p4 = CrossModalA2C2f_rgb(rgb_pre_p4, ir_pre_p4)
                ir_p4  = CrossModalA2C2f_ir (ir_pre_p4, rgb_pre_p4)
                                                         │
backbone_rgb[6:8)  ──── Conv7 ─────────────────►  rgb_pre_p5
backbone_ir [6:8)  ──── Conv7' ────────────────►  ir_pre_p5
                                                         │
                          CrossModalA2C2f @ P5:          ▼
                rgb_p5 = CrossModalA2C2f_rgb(rgb_pre_p5, ir_pre_p5)
                ir_p5  = CrossModalA2C2f_ir (ir_pre_p5, rgb_pre_p5)
```

`DualStreamDetectionModel` 需要新增 `_forward_backbone_steps()` 支持逐层步进，或将 backbone 拆分为多个 `nn.Sequential` 段。

---

### 非对称方向设计建议

结合物理特性和 Exp-2 的实验数据：

| 分支 | CrossModalA2C2f 方向 | 物理动机 |
|------|---------------------|---------|
| RGB 分支 | Q=RGB, KV=**IR** | RGB 在精确位置向 IR 查询"这里是热源吗？" → 增强 fire/person 定位精度 |
| IR 分支 | Q=IR, KV=**RGB** | IR 向 RGB 查询"这里有烟的视觉纹理吗？" → 弥补 IR 对 smoke 的盲区 |

注意 IR 分支的跨模态方向价值存疑（Exp-2 实验显示 RGB→IR 贡献接近零），可先只激活 RGB 分支的跨模态路径，做单向消融。

---

### 与 CMG 的本质区别

| 维度 | CMG（后置门控） | CrossModalA2C2f（内嵌 Attention） |
|------|--------------|----------------------------------|
| 信息粒度 | 全局向量（GAP/GMP → 1 值/通道） | 每个空间位置的 query-key 匹配 |
| 作用时机 | A2C2f 完成后 | A2C2f 内部的后半段 |
| 空间感知 | 无（或需 channel-wise 规约，很慢） | 有（area attention 保留局部空间结构）|
| 参数开销 | ~0.165M（SE-CMG 4 实例） | 约为一个 A2C2f 的 1.5×（增加 CrossModalABlock 和 cv1_other） |
| 实现复杂度 | 低（不改 backbone 执行逻辑） | 中（需 backbone 分步执行） |

---

### 预期参数量与计算开销（nano 规模估算）

每个 CrossModalA2C2f 实例（P4，c_=64，**nano: n=2组**）：
- `cv1_other`: Conv(128, 64, 1) = 8,192
- `cross_blocks` × 1组（1 个 CrossModalABlock）：
  - CrossModalAAttn: q(64×64) + kv(64×128) + proj(64×64) + pe(depthwise 64×5×5/g=64) ≈ 17K
  - MLP: Conv(64→128) + Conv(128→64) ≈ 16K
  - 合计约 33K
- 新增总量约 **41K/实例**，4 实例（P4×2方向 + P5×2方向，P5 通道数翻倍约 160K）约 **+0.20M**

推理耗时：CrossModalAAttn 的 KV 投影 + attention 计算与原 AAttn 基本等价（同维度矩阵乘法，无 channel-wise 规约），预期推理增量 **< 1ms**。

---

### 实验计划（Exp-4）

- **配置**：对称双分支，P4/P5 的 A2C2f 替换为 CrossModalA2C2f（双向）
- **消融变体**：
  - Exp-4a：仅 RGB 分支使用 CrossModalA2C2f（Q=RGB, KV=IR），IR 分支保持自注意力
  - Exp-4b：双向 CrossModalA2C2f
- **关键判断**：
  - mAP50 ≥ 0.936（超过 Exp-2 的 0.934）且推理 < 8ms → 方案可行，作为论文主要贡献
  - fire 类 mAP50-95 是否突破 0.585（IR 主导类，最能体现空间级跨模态查询的价值）
  - smoke Recall 是否恢复到 0.91+（IR 作为 KV 时，RGB 的 Q 应能"学会忽略"IR 对 smoke 无响应的区域）

---

## 备选想法：跨模态交互的其他实现路径

> 以下方案基于 Exp-1/2/2b 的实验结论——**concat + 1×1 conv 已能隐式做通道级模态选择，后置全局门控（SE/CBAM）几乎无增益**——探索不同于"后置门控"范式的跨模态交互方式。按实现复杂度从低到高排列。

### 想法 A：Depthwise 跨模态卷积门控（最轻量替代 CMG）

**动机**：CM-CBAM 的失败不一定意味着空间注意力无价值——其推理退化源于 `torch.mean/max(dim=1)` 的 channel-wise 规约对 GPU cache 不友好。用 Depthwise Conv 实现空间门控可避免此问题。

```python
class CrossModalDWGating(nn.Module):
    """用 depthwise conv 从 guide 模态生成空间门控信号。"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size,
                      padding=kernel_size // 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, target, guide):
        gate = torch.sigmoid(self.dw(guide))  # (B, C, H, W) 空间+通道门控
        return target * gate + target
```

**特点**：
- DWConv 沿空间维度滑窗，内存访问模式连续，GPU 友好
- 参数量极小：C×K×K（P4: 128×5×5=3,200，P5: 256×5×5=6,400，4 实例共 ~19K）
- 提供空间+通道联合门控，但信号来自局部感受野（5×5），非全局
- **适合场景**：快速消融，验证"空间门控方向是否有价值"（排除 CM-CBAM 因实现低效而非方案本身无效的可能）

### 想法 B：Channel Shuffle 跨模态交换（零参数消融）

**动机**：如果 concat + 1×1 conv 已经足够，那么更早的、零开销的跨模态信息混合是否也有效？Channel shuffle 强制后续层在混合通道上学习，不增加任何参数或计算。

```python
# 在 P4/P5 融合点，交换 25% 通道后各自继续 backbone
c = x_rgb.shape[1]
swap = c // 4
x_rgb_new = torch.cat([x_rgb[:, :c-swap], x_ir[:, :swap]], dim=1)
x_ir_new  = torch.cat([x_ir[:, :c-swap], x_rgb[:, :swap]], dim=1)
```

**特点**：
- 零参数、零额外计算（仅 tensor 索引操作）
- ShuffleNet (Zhang et al., 2018) 已验证通道交换对信息流动的有效性
- 交换比例（25%/50%）可作为超参消融
- **适合场景**：作为消融实验的极端对照——如果 channel shuffle 有增益，说明跨模态信息流动本身有价值，值得用更复杂的方案进一步提升；如果无增益，说明当前 concat 融合已充分利用了跨模态互补性

### 想法 C：双向特征蒸馏（训练时增强，推理零开销）

**动机**：所有跨模态交互模块都增加推理开销。如果部署目标是无人机实时检测，最理想的方案是训练时引导每个分支"理解"另一模态的表征，推理时不需要任何额外模块。

```python
# 训练时在 loss 中添加辅助蒸馏项
# project: 1×1 conv 对齐通道（如果两分支通道数不同）
loss_distill = (
    F.mse_loss(project_rgb(feats_rgb['p4']), feats_ir['p4'].detach()) +
    F.mse_loss(project_ir(feats_ir['p4']), feats_rgb['p4'].detach()) +
    F.mse_loss(project_rgb(feats_rgb['p5']), feats_ir['p5'].detach()) +
    F.mse_loss(project_ir(feats_ir['p5']), feats_rgb['p5'].detach())
)
total_loss = loss_detect + lambda_distill * loss_distill  # lambda_distill ~ 0.1
```

**特点**：
- 推理时移除 project 头和蒸馏损失，**零推理开销**
- 迫使 RGB 分支在特征空间中逼近 IR 的热响应模式（反之亦然），间接增强跨模态互补
- 可与任何其他跨模态模块叠加使用
- **风险**：MSE 损失可能过度约束，导致两个分支特征趋同（失去模态特异性）。可用 cosine similarity 或对比损失替代
- **适合场景**：部署推理速度优先的场景（无人机边缘计算），或作为所有方案的辅助增强

### 想法 D：Selective Kernel 跨模态融合（替代 concat + 1×1 conv）

**动机**：当前 concat + 1×1 conv 对两个模态的特征进行通道级线性混合，但无法根据输入内容动态调整每个模态的贡献比例。Selective Kernel (Li et al., 2019) 的注意力选择机制可以提供数据依赖的模态加权。

```python
class SelectiveModalFusion(nn.Module):
    """用 Selective Kernel 机制动态选择 RGB/IR 特征的贡献比例。"""
    def __init__(self, channels):
        super().__init__()
        self.conv_rgb = Conv(channels, channels, 3, 1)  # 小感受野（纹理）
        self.conv_ir  = Conv(channels, channels, 5, 1)  # 大感受野（热区）
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(channels // 4, 32)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
        )
        self.select_rgb = nn.Linear(mid, channels, bias=False)
        self.select_ir  = nn.Linear(mid, channels, bias=False)

    def forward(self, x_rgb, x_ir):
        feat_rgb = self.conv_rgb(x_rgb)
        feat_ir  = self.conv_ir(x_ir)
        feat_sum = feat_rgb + feat_ir
        z = self.fc(self.pool(feat_sum).flatten(1))          # (B, mid)
        a_rgb = self.select_rgb(z).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        a_ir  = self.select_ir(z).unsqueeze(-1).unsqueeze(-1)
        weights = torch.softmax(torch.stack([a_rgb, a_ir], dim=0), dim=0)
        return weights[0] * feat_rgb + weights[1] * feat_ir
```

**特点**：
- 替代 P3/P4/P5 的 `concat + 1×1 conv` 融合层，而非替代 CMG
- 不同感受野（3×3 vs 5×5）适配 RGB 纹理和 IR 热区的尺度差异
- Softmax 选择保证输出范围稳定
- 参数量略多于 1×1 conv（增加 SK 分支和选择头），但推理开销适中
- **风险**：与 CMG 类似都是全局通道级选择（GAP），可能面临同样的"信号太粗"问题

### 想法 E：分阶段渐进融合（Progressive Fusion）

**动机**：当前在 P3/P4/P5 同时独立融合，各层级的跨模态交互相互独立。但语义层级有层次关系——P5 的高语义跨模态关联可以自顶向下引导 P4/P3 的融合决策。

```
P5: CrossModalA2C2f(RGB_P5, IR_P5)  → fused_P5
        ↓ (上采样 + 条件注入)
P4: CrossModalA2C2f(RGB_P4, IR_P4, cond=fused_P5) → fused_P4
        ↓ (上采样 + 条件注入)
P3: concat(RGB_P3, IR_P3) + 1×1 conv → fused_P3  （P3 分辨率大，不做 attention）
```

**特点**：
- 高语义层先建立跨模态全局理解，低语义层利用高层先验做精细融合
- P3 保持简单 concat（分辨率 60×80，attention 计算量大），P4/P5 用 CrossModalA2C2f
- 需要修改 neck 的执行顺序（当前是先 P5→P4→P3 top-down，可自然集成）
- **复杂度**：高——需要同时实现 CrossModalA2C2f 和条件注入机制
- **适合场景**：如果 Exp-4 的独立层级 CrossModalA2C2f 效果好，可作为进一步提升的方向

### 想法 F：自适应模态选择（Adaptive Modality Selection）

**动机**：Exp-0 数据显示 smoke→RGB 主导、fire→IR 主导、person→RGB 略优。理想的融合应在空间级别选择最优模态，而非简单混合。

```python
class AdaptiveModalitySelect(nn.Module):
    """学习每个空间位置选择 RGB 还是 IR 特征（软选择）。"""
    def __init__(self, channels):
        super().__init__()
        # 用两个模态的拼接特征预测选择权重
        self.gate = nn.Sequential(
            Conv(channels * 2, channels, 1, 1),
            nn.Conv2d(channels, 2, 1, bias=True),  # 输出 2 通道：RGB 权重 + IR 权重
        )

    def forward(self, x_rgb, x_ir):
        cat = torch.cat([x_rgb, x_ir], dim=1)              # (B, 2C, H, W)
        weights = torch.softmax(self.gate(cat), dim=1)      # (B, 2, H, W)
        return weights[:, 0:1] * x_rgb + weights[:, 1:2] * x_ir  # (B, C, H, W)
```

**特点**：
- 逐像素软选择，每个位置独立决定模态贡献比例
- 烟雾区域理论上会学到 RGB 权重高、IR 权重低；火焰区域反之
- 参数量极小（仅 1×1 conv），推理开销低
- 可视化 `weights` 可作为论文可解释性分析的素材
- **风险**：Softmax 强制两个权重和为 1，可能限制"两个模态都重要"的情况。可改为独立 Sigmoid

### 优先级建议

| 优先级 | 想法 | 理由 |
|--------|------|------|
| **P0** | Exp-3（非对称 backbone，已计划） | 不涉及跨模态注意力，独立验证轻量化方向 |
| **P1** | 想法 A（DWConv 门控） | 1 小时实现，快速排除"空间门控方向是否有价值" |
| **P1** | 想法 B（Channel Shuffle） | 零参数消融，验证跨模态信息流动是否还有提升空间 |
| **P2** | Exp-4 / 想法 E（CrossModalA2C2f） | 工程量最大但理论上限最高，依赖 P1 结果决定是否投入 |
| **P3** | 想法 D（SK 融合） | 替代 concat 的融合层，与 CMG 正交，可独立验证 |
| **P3** | 想法 F（模态选择） | 轻量 + 可视化价值，适合论文分析 |
| **P4** | 想法 C（特征蒸馏） | 推理零开销但训练复杂度高，适合作为辅助增强或 future work |