# ADR-001: 双分支 YOLOv12-MF RGB-红外融合检测架构

**Status:** In Progress — Exp-4a（CMG\@P4P5 + CMA\@P4P5）训练中
**Date:** 2026-04-04
**Updated:** 2026-04-11
**Deciders:** 研究者本人

***

## 项目概要

**任务：** YOLOv12n + RGBT-3M 数据集，RGB+IR 双模态目标检测，nc=3（smoke/fire/person），输入分辨率 480×640。

**动机：** 早期融合（6ch 直接拼接）被迫在第一层就混合两种物理特性截然不同的模态信号，低层特征提取缺乏模态针对性。改为中期融合双分支架构，让每个模态独立提取低层特征，在高语义层级进行跨模态交互。

**决策：** 采用 **自定义 DualStreamDetectionModel**（Python 层实现双分支，不修改 `parse_model()`，YAML 继续定义单分支结构）。

```
Input: (B, 6, H, W)
         │
    ┌────┴────┐  split: 0:3=IR_RGB, 3:6=VIS_RGB
    ▼         ▼  （经 augment.py [::-1] 通道反转，非 Bug）
 RGB(B,3,H,W)  IR(B,3,H,W)
    │              │
┌───┴───┐    ┌────┴────┐
│ Conv  │    │  Conv'  │   Stage 0-3: P1/2 → P3/8
│ C3k2  │    │  C3k2'  │   Stage 4  ← P3 特征
│ A2C2f │    │ A2C2f'  │   Stage 6  ← P4（CMG，Exp-4a: RGB 分支换为 CMA）
│ A2C2f │    │ A2C2f'  │   Stage 8  ← P5（CMG，Exp-4a: RGB 分支换为 CMA）
└───┬───┘    └────┬────┘
    │              │
    ▼              ▼
 Concat + 1×1 Conv（P3, P4, P5）
    │
    ▼
  Shared Neck (FPN) → Detect Head
```

YAML 控制标志（`ultralytics/cfg/models/v12/yolov12-dual.yaml`）：

- `cmg_stages: [p4, p5]` — SE-CMG 跨模态门控
- `cma_stages: [p4, p5]` — CrossModalA2C2f（Exp-4a）

***

## 关键模块代码

### CrossModalGating（SE-CMG）

`ultralytics/nn/modules/block.py`

SE 式通道门控：GAP → Linear → Sigmoid，残差连接。Exp-2b 验证 CM-CBAM（空间注意力）因 `torch.mean/max(dim=1)` 的跨通道规约导致 GPU cache 不友好，推理 +166%，全面劣于 SE-CMG，已放弃。

```python
class CrossModalGating(nn.Module):
    """SE 式跨模态门控：guide 模态的全局通道统计量调制 target 模态特征。"""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, target, guide):
        gate = torch.sigmoid(self.fc(self.pool(guide).flatten(1)))  # (B, C)
        gate = gate.unsqueeze(-1).unsqueeze(-1)                     # (B, C, 1, 1)
        return target * gate + target                                # 残差门控
```

### CrossModalAAttn / CrossModalABlock / CrossModalA2C2f（Exp-4a）

`ultralytics/nn/modules/block.py`

**动机：** CMG 是后置模块——A2C2f 已完成模态内自注意力后，才施加全局粗粒度门控，信息太晚且信号太粗（GAP 丢失空间位置）。CrossModalA2C2f 将跨模态交互内嵌进 A2C2f 内部：前半段组做自注意力（建立模态内理解），后半段组用另一模态空间特征作 K/V 进行精准空间查询。

**nano 配置：** n=2 组 → 前 1 组（2×ABlock）自注意力 + 后 1 组（1×CrossModalABlock）跨模态注意力。

**Exp-4a 非对称：** 仅 RGB 分支 backbone 的 P4/P5 层替换为 CrossModalA2C2f（Q=RGB, KV=IR）；IR 分支保持标准 A2C2f。物理动机：RGB 向 IR 查询"这里是热源吗" → 增强 fire/person 定位精度。

```python
class CrossModalAAttn(nn.Module):
    """Area-Attention 跨模态变体：Q ← x_self，K/V ← x_other。"""

    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * num_heads
        self.q    = Conv(dim, all_head_dim,     1, act=False)  # Q ← x_self
        self.kv   = Conv(dim, all_head_dim * 2, 1, act=False)  # K,V ← x_other
        self.proj = Conv(all_head_dim, dim,     1, act=False)
        self.pe   = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)  # pe ← V(x_other)

    def forward(self, x_self, x_other):
        B, C, H, W = x_self.shape
        N = H * W
        q  = self.q(x_self).flatten(2).transpose(1, 2)   # (B, N, C)
        kv = self.kv(x_other)
        v_spatial = kv[:, C:, :, :]                       # V 空间形式，用于 pe
        pp = self.pe(v_spatial)                            # 位置偏置来自 guide 模态
        kv = kv.flatten(2).transpose(1, 2)                # (B, N, 2C)
        if self.area > 1:
            q  = q.reshape(B * self.area, N // self.area, C)
            kv = kv.reshape(B * self.area, N // self.area, C * 2)
            B, N, _ = q.shape  # 重新赋值（与原 AAttn 对齐，避免 reshape bug）
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
    """ABlock 的跨模态变体，attention 改为跨模态查询。"""

    def __init__(self, dim, num_heads, mlp_ratio=2.0, area=1):
        super().__init__()
        self.attn = CrossModalAAttn(dim, num_heads, area)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden, 1), Conv(mlp_hidden, dim, 1, act=False))

    def forward(self, x_self, x_other):
        x_self = x_self + self.attn(x_self, x_other)
        return x_self + self.mlp(x_self)


class CrossModalA2C2f(nn.Module):
    """A2C2f 跨模态变体：前 n//2 组自注意力，后 n-n//2 组跨模态注意力。"""

    def __init__(self, c1, c2, n=2, area=4, mlp_ratio=2.0, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0
        num_heads = c_ // 32
        self.cv1       = Conv(c1, c_, 1, 1)
        self.cv1_other = Conv(c1, c_, 1, 1)       # x_other 独立通道映射
        self.cv2       = Conv((1 + n) * c_, c2, 1)
        n_self  = n // 2
        n_cross = n - n_self
        self.m_self  = nn.ModuleList(              # 前半段：自注意力（每组 2×ABlock）
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2)))
            for _ in range(n_self))
        self.m_cross = nn.ModuleList(              # 后半段：跨模态注意力
            CrossModalABlock(c_, num_heads, mlp_ratio, area)
            for _ in range(n_cross))

    def forward(self, x_self, x_other):
        feat       = self.cv1(x_self)
        feat_other = self.cv1_other(x_other)
        y = [feat]
        for m in self.m_self:  y.append(m(y[-1]))
        for m in self.m_cross: y.append(m(y[-1], feat_other))
        return self.cv2(torch.cat(y, 1))
```

**与 CMG 的本质区别：**

| 维度   | CMG（后置门控）             | CrossModalA2C2f            |
| ---- | --------------------- | -------------------------- |
| 信息粒度 | 全局向量（GAP → 1 值/通道）    | 每个空间位置的 Q-K 匹配             |
| 作用时机 | A2C2f 完成后             | A2C2f 内部后半段                |
| 空间感知 | 无                     | 有（area attention 保留局部空间结构） |
| 参数开销 | \~0.165M（SE-CMG 4 实例） | \~+0.20M                   |

### 模型入口 get\_model()

`ultralytics/models/yolo/detect/train.py`

```python
def get_model(self, cfg=None, weights=None, verbose=True):
    input_mode = self.data.get("input_mode", "dual_input")
    yaml_cfg = yaml_model_load(cfg) if isinstance(cfg, str) else cfg
    use_dual_stream = yaml_cfg.get("dual_stream", False) if yaml_cfg else False
    if use_dual_stream and input_mode == "dual_input":
        model = DualStreamDetectionModel(cfg, ch=6, nc=self.data["nc"],
                                          verbose=verbose and RANK == -1)
    else:
        model = DetectionModel(cfg, ch=6 if input_mode == "dual_input" else 3,
                               nc=self.data["nc"], verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)
    return model
```

***

## 实验结果

YOLOv12n + RGBT-3M，480×640，epochs=300，batch=16，SGD

### 汇总表

| 实验           | 配置                                  | mAP50     | mAP50-95  | Precision | Recall    | 参数量       | 推理(ms)    | 状态     |
| ------------ | ----------------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | ------ |
| Exp-0        | 单分支 6ch early fusion                | 0.919     | 0.605     | 0.914     | 0.877     | 2.51M     | 4.51      | 完成     |
| Exp-1        | 对称双分支 concat-only                   | 0.933     | 0.623     | 0.924     | 0.893     | —         | 6.96      | 完成     |
| **Exp-2**    | **对称双分支 + CMG\@P4P5**               | **0.934** | **0.627** | **0.925** | **0.894** | **4.21M** | **7.13**  | **完成** |
| Exp-2b       | 对称双分支 + CM-CBAM\@P4P5               | 0.927     | 0.627     | 0.925     | 0.882     | 4.13M     | 18.91     | 完成（否定） |
| Exp-4a       | 对称双分支 + CMG\@P4P5 + 单向CMA\@P4P5     | 0.934     | 0.625     | —         | —         | —         | —         | 完成     |
| Exp-4b       | 对称双分支 + CMG\@P4P5 + 双向残差CMA\@P4P5   | 0.934     | 0.627     | 0.923     | 0.890     | 5.50M     | 17.55     | 完成     |
| Exp-4b-noCMG | 对称双分支 + 双向残差CMA\@P4P5（无CMG）         | 0.930     | 0.628     | —         | —         | —         | —         | 初步     |
| **Exp-4c**   | **对称双分支 + CMG\@P4P5 + 双向残差CMA\@P5** | **0.934** | **0.632** | **0.916** | **0.892** | **5.24M** | **16.05** | **完成** |
| Exp-4d       | 对称双分支 + CMG\@P4P5 + 双向残差CMA\@P4     | 0.931     | 0.623     | 0.923     | 0.887     | 4.48M     | 16.68     | 完成     |
| Exp-5        | RGB-only 3ch                        | 0.907     | 0.583     | 0.920     | 0.854     | 2.51M     | 4.08      | 完成     |
| Exp-6        | IR-only 3ch                         | 0.887     | 0.564     | 0.888     | 0.831     | 2.51M     | 4.33      | 完成     |

### 分类别结果

**Exp-1（对称双分支 concat-only）：**

| 类别     | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| all    | 0.924     | 0.893  | 0.933 | 0.623    |
| smoke  | 0.937     | 0.914  | 0.946 | 0.735    |
| fire   | 0.933     | 0.890  | 0.935 | 0.575    |
| person | 0.901     | 0.874  | 0.919 | 0.559    |

**Exp-2（+ CMG\@P4P5）：**

| 类别     | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| all    | 0.925     | 0.894  | 0.934 | 0.627    |
| smoke  | 0.941     | 0.903  | 0.949 | 0.739    |
| fire   | 0.930     | 0.892  | 0.929 | 0.581    |
| person | 0.903     | 0.886  | 0.924 | 0.562    |

**Exp-2b（+ CM-CBAM\@P4P5，已否定）：**

| 类别     | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| all    | 0.925     | 0.882  | 0.927 | 0.627    |
| smoke  | 0.949     | 0.885  | 0.944 | 0.739    |
| fire   | 0.925     | 0.887  | 0.925 | 0.578    |
| person | 0.903     | 0.875  | 0.912 | 0.563    |

**Exp-4d（CMG\@P4P5 + 双向残差CMA\@P4，4.48M params，19.60 GFLOPs）：**

| 类别     | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| all    | 0.923     | 0.887  | 0.931 | 0.623    |
| smoke  | 0.949     | 0.898  | 0.948 | 0.738    |
| fire   | 0.927     | 0.883  | 0.929 | 0.573    |
| person | 0.892     | 0.880  | 0.916 | 0.559    |

**Exp-4c（CMG\@P4P5 + 双向残差CMA\@P5，5.24M params，19.65 GFLOPs）：**

| 类别     | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| all    | 0.916     | 0.892  | 0.934 | 0.632    |
| smoke  | 0.937     | 0.897  | 0.946 | 0.741    |
| fire   | 0.922     | 0.890  | 0.929 | 0.582    |
| person | 0.889     | 0.890  | 0.927 | 0.573    |

*Exp-4b（CMG\@P4P5 + 双向残差CMA\@P4P5，5.50M params，18.57 GFLOPs*）：\*

| 类别     | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| all    | 0.923     | 0.890  | 0.934 | 0.627    |
| smoke  | 0.943     | 0.894  | 0.946 | 0.739    |
| fire   | 0.926     | 0.892  | 0.933 | 0.576    |
| person | 0.899     | 0.885  | 0.924 | 0.566    |

\*GFLOPs 数值存疑（thop 对双流模型追踪可能不准），以实际 inference 17.55ms 为参考。

### 关键发现

**1. 双分支是核心贡献（Exp-0 → Exp-1，整体 mAP50 +1.4%）**

双分支独立特征提取贡献了绝大部分增益，三类均有提升：

| 类别     | mAP50 (Exp-0→Exp-1) | mAP50-95 (Exp-0→Exp-1) | 主要原因                         |
| ------ | ------------------- | ---------------------- | ---------------------------- |
| smoke  | +2.7%               | +1.3%                  | IR 分支独立提取，避免可见光烟雾纹理与 IR 噪声混淆 |
| fire   | +1.6%               | +3.2%                  | IR 独立分支充分利用热辐射信号，定位精度显著提升    |
| person | +2.0%               | +1.6%                  | 双模态互补：RGB 纹理 + IR 热轮廓联合描述人体  |

**结论：双分支本身是最大创新点，后续模块（CMG/CMA）均是在此基础上的精细化优化。**

***

**2. CMG 增益结构性极小（Exp-1 → Exp-2）**

| 类别     | ΔmAP50 | ΔmAP50-95 | ΔRecall | 解读                           |
| ------ | ------ | --------- | ------- | ---------------------------- |
| smoke  | +0.3%  | +0.4%     | -1.1%   | 轻微提升，Recall 略降（门控抑制了部分弱烟雾信号） |
| fire   | -0.6%  | +0.6%     | +0.2%   | 矛盾信号：检出略少但定位更准               |
| person | +0.5%  | +0.3%     | +1.2%   | 小幅正向                         |

全局 mAP50 仅 +0.1%。根本原因：concat 后的 1×1 Conv 本身已具备通道选择能力，全局 GAP 门控（1个标量/通道）信号粒度过粗，难以提供增量价值。

***

**3. CM-CBAM 全面否定（Exp-2b，-0.7% mAP50，推理 +166%）**

`torch.mean/max(dim=1)` 沿 channel 轴规约，需跨步内存访问，GPU cache 极不友好。SE-CMG 的 `AdaptiveAvgPool2d(1)` 沿空间轴规约，访问连续内存，有高度优化的 CUDA kernel。**参数量减少 ≠ 推理加速，根本在于内存访问模式。**

***

**4. 模态特异性（Exp-5/6 单模态基线）**

| 类别     | RGB mAP50 | IR mAP50  | 优势模态   | 启示                      |
| ------ | --------- | --------- | ------ | ----------------------- |
| smoke  | 0.907（全局） | 0.887（全局） | RGB 主导 | IR 对烟雾是噪声，融合需门控抑制 IR    |
| fire   | —         | —         | IR 主导  | 热辐射定位更精准，IR 分支需充分特征提取   |
| person | —         | —         | RGB 略优 | 人体纹理依赖可见光，差距不如 smoke 明显 |

***

**5. CMA 的核心贡献在于 person 类定位（Exp-2 → Exp-4c）**

与 CMG 基线（Exp-2）相比，CMA\@P5 的分类别增益：

| 类别         | ΔmAP50    | ΔmAP50-95 | ΔRecall   | ΔPrecision |
| ---------- | --------- | --------- | --------- | ---------- |
| smoke      | -0.3%     | +0.2%     | -0.6%     | -0.4%      |
| fire       | 0.0%      | +0.1%     | -0.2%     | -0.8%      |
| **person** | **+0.3%** | **+1.1%** | **+0.4%** | **-1.4%**  |

**Person 类是 CMA\@P5 的主要受益者**：mAP50-95 +1.1%，这是本实验系列中单类别最大改善。物理解释：person 检测高度依赖语义级别的身体结构理解，P5（最高语义层）的跨模态注意力让模型在高语义空间中查询"IR 视角下人体轮廓"，有效补充 RGB 的纹理信息。代价是 Precision 略降（-1.4%），模型召回更多目标的同时引入少量误检。

***

**6. CMA\@P4 对 person 类有害（Exp-4d vs Exp-2）**

| 类别         | ΔmAP50 (4d-2) | ΔmAP50-95 (4d-2) | ΔRecall   |
| ---------- | ------------- | ---------------- | --------- |
| smoke      | -0.1%         | -0.1%            | -0.5%     |
| fire       | 0.0%          | -0.8%            | -0.9%     |
| **person** | **-0.8%**     | **-0.3%**        | **-0.6%** |

P4（30×40，中级语义）的跨模态注意力引入了噪声而非有效信息，person 类三项指标全部下降。P4 特征尚未充分建立语义表征，跨模态查询时两个模态的特征空间对齐程度不足，反而干扰了各自的特征提取。

**结论：CMA 应只部署在 P5，不应在 P4。**

***

**7. Smoke 对 CMA 不敏感，对 CMG 微弱正响应**

横跨所有 CMA 实验，smoke 类 mAP50 稳定在 0.946~~0.949，mAP50-95 稳定在 0.738~~0.741，几乎不受 CMA 层级选择影响：

| 实验                        | smoke mAP50 | smoke mAP50-95 |
| ------------------------- | ----------- | -------------- |
| Exp-1（无CMG/CMA）           | 0.946       | 0.735          |
| Exp-2（+ CMG）              | 0.949       | 0.739          |
| Exp-4d（+ CMG + CMA\@P4）   | 0.948       | 0.738          |
| Exp-4c（+ CMG + CMA\@P5）   | 0.946       | 0.741          |
| Exp-4b（+ CMG + CMA\@P4P5） | 0.946       | 0.739          |

Smoke 是大目标（均值面积远大于 fire/person），高语义层（P5）的空间注意力对大目标作用有限，检测主要依赖较低层的纹理/边缘特征。CMG 对 smoke 有微弱正效果（+0.4% mAP50-95），可能因为全局门控有效抑制了 IR 对烟雾区域的噪声响应。

***

**8. CMA\@P4P5 组合：fire 受益但 person 不如 CMA\@P5 alone（Exp-4b vs Exp-4c）**

| 类别              | Exp-4b (CMA\@P4P5) | Exp-4c (CMA\@P5) | Δ(4b-4c)   |
| --------------- | ------------------ | ---------------- | ---------- |
| smoke mAP50-95  | 0.739              | 0.741            | -0.002     |
| fire mAP50      | **0.933**          | 0.929            | **+0.004** |
| fire mAP50-95   | 0.576              | **0.582**        | -0.006     |
| person mAP50    | 0.924              | **0.927**        | -0.003     |
| person mAP50-95 | 0.566              | **0.573**        | -0.007     |

CMA\@P4P5 相比 CMA\@P5：fire mAP50 更高（P4 的中级特征对火焰检测有一定帮助），但 fire 和 person 的 mAP50-95 均更低，说明加入 P4 CMA 反而损害了定位精度。

**对于 fire/person 这两个安全关键类别，CMA\@P5 alone 是更优选择。**

***

**9. 单向 CMA 无效，双向是关键（Exp-4a vs Exp-2）**

单向 CMA（仅 IR→RGB，Exp-4a）与 CMG 基线（Exp-2）相比 mAP50 持平（0.934），mAP50-95 -0.002，无任何提升。双向残差 CMA（Exp-4b/4c）则有明显改善。

**物理解释：** 单向查询只让 RGB 分支"参考 IR"，但 IR 分支未得到 RGB 的高频纹理信息补充，整体特征互补性不足。双向交互使两个分支在各自的决策中都融入了对方的空间信息，形成协同增强而非单边增强。

***

**10. 当前最优配置及推理代价分析**

| 配置                    | mAP50     | mAP50-95  | person mAP50-95 | 推理(ms)    | 相对 Exp-2          |
| --------------------- | --------- | --------- | --------------- | --------- | ----------------- |
| Exp-2（CMG only）       | 0.934     | 0.627     | 0.562           | 7.13      | 基准                |
| Exp-4d（+ CMA\@P4）     | 0.931     | 0.623     | 0.559           | 16.68     | +134% 推理，负收益      |
| Exp-4b（+ CMA\@P4P5）   | 0.934     | 0.627     | 0.566           | 17.55     | +146% 推理，持平       |
| **Exp-4c（+ CMA\@P5）** | **0.934** | **0.632** | **0.573**       | **16.05** | **+125% 推理，最优收益** |

**Exp-4c 是当前最优候选**：以约 2.25× 推理时间代价，换取 person mAP50-95 +1.1%（安全任务中最重要的指标），整体 mAP50-95 +0.5%。CMA\@P4 系列（Exp-4d/4b）的推理代价相当或更高，但收益更低甚至为负，不值得部署。

***

## 消融实验状态

| 实验           | 配置                                  | 目的              | 状态     | mAP50     | mAP50-95  |
| ------------ | ----------------------------------- | --------------- | ------ | --------- | --------- |
| Exp-0        | 单分支 early fusion                    | 基准              | 完成     | 0.919     | 0.605     |
| Exp-1        | 对称双分支 concat-only                   | 验证双分支增益         | 完成     | 0.933     | 0.623     |
| Exp-2        | 对称双分支 + CMG\@P4P5                   | 验证 SE 门控增益      | 完成     | 0.934     | 0.627     |
| Exp-2b       | 对称双分支 + CM-CBAM\@P4P5               | 验证空间注意力         | 完成（否定） | 0.927     | 0.627     |
| Exp-4a       | 对称双分支 + CMG\@P4P5 + 单向CMA\@P4P5     | 验证单向空间跨模态查询     | 完成     | 0.934     | 0.625     |
| Exp-4b       | 对称双分支 + CMG\@P4P5 + 双向残差CMA\@P4P5   | 验证双向CMA + 残差叠加  | 完成     | 0.934     | 0.627     |
| Exp-4b-noCMG | 对称双分支 + 双向残差CMA\@P4P5（无CMG）         | 消融：CMG是否与CMA冲突  | 初步     | 0.930     | 0.628     |
| **Exp-4c**   | **对称双分支 + CMG\@P4P5 + 双向残差CMA\@P5** | **层级消融：CMA仅P5** | **完成** | **0.934** | **0.632** |
| Exp-4d       | 对称双分支 + CMG\@P4P5 + 双向残差CMA\@P4     | 层级消融：CMA仅P4     | 完成     | 0.931     | 0.623     |
| Exp-5        | RGB-only 3ch                        | 单模态基准           | 完成     | 0.907     | 0.583     |
| Exp-6        | IR-only 3ch                         | 单模态基准           | 完成     | 0.887     | 0.564     |

**当前最优候选（已确认）：**

- **综合最优：Exp-4c**（CMG\@P4P5 + 双向残差CMA\@P5）——mAP50-95=0.632（全系列最高），person mAP50-95=0.573（+1.1% vs CMG基线），推理代价最低（16.05ms）
- 待跑完整 val：Exp-4b-noCMG（消融 CMG 的作用）

**实验执行备忘：**

- 训练超参：epochs=300, batch=16, SGD lr0=0.01 lrf=0.01 cos\_lr=False val\_period=2
- 关注 CPU RAM：训练到 \~epoch 77 可能触发 `_ArrayMemoryError`，需监控并准备 `--resume`
- val.py 速度评估统一跑 10 次取平均

***

## 备选想法

> 以下方案基于 Exp-1/2/2b 的结论——concat + 1×1 conv 已能隐式做通道级模态选择，后置全局门控（SE/CBAM）几乎无增益——探索不同于"后置门控"范式的跨模态交互方式。

### 想法 A：Depthwise 跨模态卷积门控（最轻量替代 CMG）

DWConv 沿空间维度滑窗，内存访问连续，GPU 友好。参数量极小（\~19K），提供空间+通道联合门控。适合快速消融"空间门控方向是否有价值"（排除 CM-CBAM 因实现低效而非方案本身无效的可能）。

```python
class CrossModalDWGating(nn.Module):
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

### 想法 B：Channel Shuffle 跨模态交换（零参数消融）

零参数、零额外计算。强制后续层在混合通道上学习，验证"跨模态信息流动本身是否有价值"。

```python
# 在 P4/P5 融合点，交换 25% 通道后各自继续 backbone
c = x_rgb.shape[1]; swap = c // 4
x_rgb_new = torch.cat([x_rgb[:, :c-swap], x_ir[:, :swap]], dim=1)
x_ir_new  = torch.cat([x_ir[:, :c-swap], x_rgb[:, :swap]], dim=1)
```

### 想法 C：双向特征蒸馏（训练时增强，推理零开销）

训练时在 loss 中添加辅助蒸馏项，迫使每个分支在特征空间逼近另一模态的表征；推理时移除 project 头和蒸馏损失，零推理开销。适合部署推理速度优先的场景。

```python
loss_distill = (
    F.mse_loss(project_rgb(feats_rgb['p4']), feats_ir['p4'].detach()) +
    F.mse_loss(project_ir(feats_ir['p4']), feats_rgb['p4'].detach()) +
    F.mse_loss(project_rgb(feats_rgb['p5']), feats_ir['p5'].detach()) +
    F.mse_loss(project_ir(feats_ir['p5']), feats_rgb['p5'].detach())
)
total_loss = loss_detect + lambda_distill * loss_distill  # lambda_distill ~ 0.1
```

风险：MSE 损失可能过度约束，导致两个分支特征趋同（失去模态特异性），可改用 cosine similarity 或对比损失。

### 想法 D：Selective Kernel 跨模态融合（替代 concat + 1×1 conv）

动态调整每个模态的贡献比例，替代 P3/P4/P5 的 `concat + 1×1 conv` 融合层。不同感受野（3×3 vs 5×5）适配 RGB 纹理和 IR 热区的尺度差异。

```python
class SelectiveModalFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_rgb = Conv(channels, channels, 3, 1)
        self.conv_ir  = Conv(channels, channels, 5, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(channels // 4, 32)
        self.fc = nn.Sequential(nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True))
        self.select_rgb = nn.Linear(mid, channels, bias=False)
        self.select_ir  = nn.Linear(mid, channels, bias=False)

    def forward(self, x_rgb, x_ir):
        feat_rgb = self.conv_rgb(x_rgb); feat_ir = self.conv_ir(x_ir)
        z = self.fc(self.pool(feat_rgb + feat_ir).flatten(1))
        a_rgb = self.select_rgb(z).unsqueeze(-1).unsqueeze(-1)
        a_ir  = self.select_ir(z).unsqueeze(-1).unsqueeze(-1)
        weights = torch.softmax(torch.stack([a_rgb, a_ir], dim=0), dim=0)
        return weights[0] * feat_rgb + weights[1] * feat_ir
```

风险：与 CMG 类似都是全局通道级选择（GAP），可能面临同样的"信号太粗"问题。

### 想法 E：分阶段渐进融合（Progressive Fusion）

P5 高语义跨模态理解自顶向下引导 P4/P3 融合决策，与 neck 的 top-down 执行顺序自然集成。P3 保持简单 concat（分辨率 60×80，attention 计算量过大）。

```
P5: CrossModalA2C2f(RGB_P5, IR_P5) → fused_P5
          ↓ (上采样 + 条件注入)
P4: CrossModalA2C2f(RGB_P4, IR_P4, cond=fused_P5) → fused_P4
          ↓ (上采样 + 条件注入)
P3: concat(RGB_P3, IR_P3) + 1×1 conv → fused_P3
```

适合 Exp-4a 效果好后进一步提升的方向。

### 想法 F：自适应模态选择（逐像素软选择）

每个空间位置独立决定 RGB/IR 贡献比例。烟雾区域理论上会学到 RGB 权重高、IR 权重低；火焰区域反之。权重图可作为论文可解释性分析的素材。

```python
class AdaptiveModalitySelect(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            Conv(channels * 2, channels, 1, 1),
            nn.Conv2d(channels, 2, 1, bias=True),
        )

    def forward(self, x_rgb, x_ir):
        weights = torch.softmax(self.gate(torch.cat([x_rgb, x_ir], dim=1)), dim=1)
        return weights[:, 0:1] * x_rgb + weights[:, 1:2] * x_ir
```

风险：Softmax 强制权重和为 1，可能限制"两个模态都重要"的情况，可改为独立 Sigmoid。

### 想法 G：差异驱动融合（CMG 改进方向，最推荐）

**动机：** 现有 CMG/SE/CBAM 的逻辑是"学一个权重，猜哪里重要"。换一个问法：RGB 与 IR 差异大的区域（火焰、高温目标）正是需要跨模态补充最多的地方——用显式差异图驱动融合门控，不需要学，直接告诉模型。与 SE/CBAM 有本质逻辑区别，物理动机强，实现轻量。

```python
class DifferenceGuidedFusion(nn.Module):
    """差异驱动融合：RGB 与 IR 差异大的空间位置获得更多跨模态注入。"""

    def __init__(self, channels):
        super().__init__()
        self.diff_encoder = Conv(channels, channels // 4, 1)
        self.gate = nn.Sequential(
            Conv(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_rgb, x_ir):
        diff = torch.abs(x_rgb - x_ir)           # 显式空间差异图 (B, C, H, W)
        gate = self.gate(self.diff_encoder(diff)) # 差异大 → gate 大 → 更多 IR 注入
        return x_rgb + gate * x_ir               # 残差式注入
```

**审稿优势：** 逻辑与 SE/CBAM 不同（差异驱动 vs 全局统计驱动），好写动机，可视化差异图作为论文分析素材。

***

### 想法 H：频率解耦融合

**动机：** IR 热成像对火焰的表达主要体现在低频热区轮廓与高频边缘梯度，和 RGB 纹理频率分布不同。在频域分离融合可以让两个模态的互补信息在各自擅长的频段叠加，避免直接在空间域混合导致的模糊。

```
x_ir  → FFT → 低频分量 ──→ 调制 x_rgb 低频（热区轮廓对齐）
             → 高频分量 ──→ 调制 x_rgb 高频（火焰边缘锐化）
                              ↓
                           IFFT → 融合特征
```

实现上可用 `torch.fft.rfft2` / `irfft2`，或用 DCT 近似（无需复数运算）。工程量中等，适合作为论文里"面向火焰红外特性的专项设计"展示。

***

### 想法 I：类别感知动态门控

**动机：** 不同类别对两个模态的依赖不同（smoke 主要依赖 RGB 纹理，fire/person 主要依赖 IR 热辐射）。让门控信号受类别原型条件化，动态调整每个空间位置的模态融合比例，可解释性强、适合论文可视化分析。

```python
# 用全局平均特征 + 轻量 MLP 估计"当前 patch 更像哪类"
# smoke 原型 → RGB 权重↑，IR 权重↓
# fire/person 原型 → IR 权重↑，RGB 权重↓
```

**风险：** 需避免与检测头形成循环依赖（soft 类别估计须在 neck 之前完成）；消融设计相对复杂。更适合作为分析工具或论文可视化而非主干模块。

***

### 优先级

| 优先级    | 想法                         | 理由                                            |
| ------ | -------------------------- | --------------------------------------------- |
| **P0** | **Exp-4b / 4b-noCMG（进行中）** | 验证双向残差 CMA 有效性                                |
| **P1** | 想法 G（差异驱动融合）               | 实现简单（\~30 行），物理动机强，与 SE/CBAM 逻辑不同，作为 CMG 改进方向 |
| **P1** | 想法 A（DWConv 门控）            | 快速排除"空间门控是否有价值"（与想法 G 互为对照）                   |
| **P2** | 想法 H（频率解耦融合）               | 对火焰/IR 有针对性，工程量中等，适合论文亮点设计                    |
| **P2** | 想法 B（Channel Shuffle）      | 零参数消融，验证跨模态信息流动天花板                            |
| **P3** | 想法 D（SK 融合）                | 替代 concat 融合层，与 CMG 正交，可独立验证                  |
| **P3** | 想法 F（模态选择）                 | 轻量 + 可视化价值，适合论文分析                             |
| **P3** | 想法 I（类别感知动态门控）             | 可解释性强但设计复杂，更适合作为分析工具                          |
| **P4** | 想法 C（特征蒸馏）                 | 推理零开销但训练复杂度高，适合 future work                   |
| **P4** | 想法 E（渐进融合）                 | 依赖 CMA 成功，工程量较大                               |

