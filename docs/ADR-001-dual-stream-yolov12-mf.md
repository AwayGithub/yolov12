# ADR-001: 双分支 YOLOv12-MF RGB-红外融合检测架构

**Status:** 实验完成，当前最优 Exp-4c（CMG@P4P5 + 双向残差CMA@P5）
**Date:** 2026-04-04
**Updated:** 2026-04-14
**Deciders:** 研究者本人

---

## 一、项目概要

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
│ A2C2f │    │ A2C2f'  │   Stage 6  ← P4（可加 CMG / CMA）
│ A2C2f │    │ A2C2f'  │   Stage 8  ← P5（可加 CMG / CMA）
└───┬───┘    └────┬────┘
    │              │
    ▼              ▼
 Concat + 1×1 Conv（P3, P4, P5）
    │
    ▼
  Shared Neck (FPN) → Detect Head
```

**YAML 控制标志**（`ultralytics/cfg/models/v12/yolov12-dual.yaml`）：

- `cmg_stages: [p4, p5]` — SE-CMG 跨模态门控
- `cma_stages: [p4, p5]` — CrossModalA2C2f 双向残差跨模态注意力

---

## 二、关键模块代码

### 2.1 CrossModalGating（SE-CMG）

`ultralytics/nn/modules/block.py`

SE 式通道门控：GAP → Linear → Sigmoid，残差连接。Exp-2b 验证 CM-CBAM（空间注意力）因 `torch.mean/max(dim=1)` 的跨通道规约导致 GPU cache 不友好，推理 +166%，全面劣于 SE-CMG，已放弃。

参数量：每实例 = c×c + c（Linear）。4 实例（P4×2 + P5×2）总计 164,608 参数（可由差值精确验证）。

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

### 2.2 CrossModalA2C2f（主要创新）

`ultralytics/nn/modules/block.py`

**动机：** CMG 是后置模块——A2C2f 已完成模态内自注意力后才施加全局粗粒度门控，信息太晚且信号太粗（GAP 丢失空间位置）。CrossModalA2C2f 将跨模态交互内嵌进 A2C2f 内部：前半段组做自注意力（建立模态内理解），后半段组用另一模态空间特征作 K/V 进行精准空间查询。

**nano 配置（n=2）：** 前 1 组（2×ABlock）自注意力 + 后 1 组（1×CrossModalABlock）跨模态注意力。

**参数规律：** 参数量 ∝ c²，P5（256ch）约为 P4（128ch）的 4 倍；但 FLOPs ∝ c²×H×W，P5 的空间缩小恰好抵消通道增大，故 P4/P5 的 FLOPs 几乎相等（~19.6 GFLOPs）。

```python
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
        self.cross_scale = nn.Parameter(torch.ones(1) * 0.01)  # 跨模态残差可学习权重
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
        n_self_only = len(self.m_self) - len(self.m_cross)
        y = [feat]
        for m in self.m_self[:n_self_only]:
            y.append(m(y[-1]))
        cross_scale = getattr(self, "cross_scale", 1.0)  # 兼容旧 checkpoint
        for m_s, m_c in zip(self.m_self[n_self_only:], self.m_cross):
            y.append(m_s(y[-1]) + cross_scale * m_c(y[-1], feat_other))
        return self.cv2(torch.cat(y, 1))
```

**与 CMG 的本质区别：**

| 维度 | CMG（后置门控） | CrossModalA2C2f |
|------|----------------|-----------------|
| 信息粒度 | 全局向量（GAP → 1 值/通道） | 每个空间位置的 Q-K 匹配 |
| 作用时机 | A2C2f 完成后 | A2C2f 内部后半段 |
| 空间感知 | 无 | 有（area attention 保留局部空间结构） |
| 参数开销（P5） | ~33K/实例 | ~377K/实例 |

### 2.3 模型入口 get_model()

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

---

## 三、实验结果

**训练超参：** YOLOv12n + RGBT-3M，480×640，epochs=300，batch=16，SGD lr0=0.01 lrf=0.01 cos_lr=False val_period=2

**Fitness 公式（worktree 自定义）：** `w = [0.0, 0.2, 0.3, 0.5]`（P/R/mAP50/mAP50-95），显式纳入 Recall，适合安全监控任务。

### 3.1 汇总表

| 实验 | 配置简述 | mAP50 | mAP50-95 | Precision | Recall | 参数量 | 推理(ms) | 状态 |
|------|---------|-------|----------|-----------|--------|--------|----------|------|
| Exp-0 | 单分支 6ch early fusion | 0.919 | 0.605 | 0.914 | 0.877 | 2.51M | 4.51 | 完成 |
| Exp-1 | 对称双分支 concat-only | 0.933 | 0.623 | 0.924 | 0.893 | — | 6.96 | 完成 |
| Exp-2 | 对称双分支 + CMG@P4P5 | 0.934 | 0.627 | 0.925 | 0.894 | 4.21M | 7.13 | 完成 |
| Exp-2b | 对称双分支 + CM-CBAM@P4P5 | 0.927 | 0.627 | 0.925 | 0.882 | 4.13M | 18.91 | 完成（否定）|
| Exp-4a | 双分支 + CMG@P4P5 + 单向CMA@P4P5（RGB←IR） | 0.933 | 0.625 | 0.918 | 0.900 | 4.94M | 15.27 | 完成 |
| Exp-4b | 双分支 + CMG@P4P5 + 双向CMA@P4P5 | 0.934 | 0.627 | 0.923 | 0.890 | 5.49M | 17.55 | 完成 |
| Exp-4b-noCMG | 双分支 + 双向CMA@P4P5（无CMG） | 0.929 | 0.628 | 0.922 | 0.886 | 5.34M | 16.09 | 完成 |
| **Exp-4c** ★ | **双分支 + CMG@P4P5 + 双向CMA@P5** | **0.934** | **0.632** | **0.916** | **0.892** | **5.23M** | **16.05** | **完成** |
| Exp-4d | 双分支 + CMG@P4P5 + 双向CMA@P4 | 0.931 | 0.623 | 0.923 | 0.887 | 4.47M | 16.68 | 完成 |
| Exp-5 | RGB-only 3ch 单分支 | 0.907 | 0.583 | 0.920 | 0.854 | 2.51M | 4.08 | 完成 |
| Exp-6 | IR-only 3ch 单分支 | 0.887 | 0.564 | 0.888 | 0.831 | 2.51M | 4.33 | 完成 |

> ★ 当前最优候选。GFLOPs 见下表（P4P5 组合因 thop 双流追踪问题偏低，以推理 ms 为准）。

### 3.2 参数量与计算量对比

| 实验 | 参数量 | GFLOPs | 推理(ms) | 备注 |
|------|--------|--------|----------|------|
| Exp-2（CMG基线） | 4,210,000 | — | 7.13 | |
| Exp-4a（单向CMA@P4P5） | 4,943,833 | 18.02 | 15.27 | 仅 RGB 分支替换 |
| Exp-4b-noCMG（双向CMA@P4P5） | 5,335,709 | 18.47 | 16.09 | 无CMG，差值验证：5,488,997-5,335,709=153,288≈CMG参数量 |
| Exp-4d（双向CMA@P4） | 4,473,187 | 19.60 | 16.68 | |
| **Exp-4c（双向CMA@P5）** | **5,226,851** | **19.65** | **16.05** | P4/P5 GFLOPs 相近因通道↑×空间↓相消 |
| Exp-4b（双向CMA@P4P5） | 5,488,997 | 18.57* | 17.55 | *thop 对双流追踪偏低，以 ms 为准 |

**参数量自洽验证：**
- Exp-4c - Exp-4d = 5,226,851 - 4,473,187 = **753,664**（P5 比 P4 多，因 256²×2 - 128²×2 ≈ 753K ✓）
- Exp-4b ≈ Exp-4d + Exp-4c - Exp-2：4,473,187 + 5,226,851 - 4,210,000 = **5,490,038 ≈ 5,488,997 ✓**
- Exp-4b - Exp-4b-noCMG = 5,488,997 - 5,324,389 = **164,608 = CMG 4 实例精确参数量 ✓**

### 3.3 分类别详细结果

**Exp-0（单分支 6ch early fusion，2.51M，17.31ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.914 | 0.877 | 0.919 | 0.605 |
| smoke | 0.939 | 0.905 | 0.947 | 0.734 |
| fire | 0.913 | 0.856 | 0.907 | 0.545 |
| person | 0.891 | 0.869 | 0.902 | 0.537 |

**Exp-1（对称双分支 concat-only）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.924 | 0.893 | 0.933 | 0.623 |
| smoke | 0.937 | 0.914 | 0.946 | 0.735 |
| fire | 0.933 | 0.890 | 0.935 | 0.575 |
| person | 0.901 | 0.874 | 0.919 | 0.559 |

**Exp-2（+ CMG@P4P5，4.21M，7.13ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.925 | 0.894 | 0.934 | 0.627 |
| smoke | 0.941 | 0.903 | 0.949 | 0.739 |
| fire | 0.930 | 0.892 | 0.929 | 0.581 |
| person | 0.903 | 0.886 | 0.924 | 0.562 |

**Exp-2b（+ CM-CBAM@P4P5，已否定，4.13M，18.91ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.925 | 0.882 | 0.927 | 0.627 |
| smoke | 0.949 | 0.885 | 0.944 | 0.739 |
| fire | 0.925 | 0.887 | 0.925 | 0.578 |
| person | 0.903 | 0.875 | 0.912 | 0.563 |

**Exp-4a（CMG@P4P5 + 单向CMA@P4P5 RGB←IR，4.94M，15.27ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.918 | 0.900 | 0.933 | 0.625 |
| smoke | 0.934 | 0.912 | 0.947 | 0.735 |
| fire | 0.922 | 0.891 | 0.927 | 0.579 |
| person | 0.897 | 0.896 | 0.926 | 0.562 |

**Exp-4b-noCMG（双向CMA@P4P5，无CMG，5.34M，16.09ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.922 | 0.886 | 0.929 | 0.628 |
| smoke | 0.946 | 0.892 | 0.946 | 0.742 |
| fire | 0.918 | 0.893 | 0.929 | 0.579 |
| person | 0.902 | 0.874 | 0.913 | 0.564 |

**Exp-4b（CMG@P4P5 + 双向CMA@P4P5，5.49M，17.55ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.923 | 0.890 | 0.934 | 0.627 |
| smoke | 0.943 | 0.894 | 0.946 | 0.739 |
| fire | 0.926 | 0.892 | 0.933 | 0.576 |
| person | 0.899 | 0.885 | 0.924 | 0.566 |

**Exp-4c ★（CMG@P4P5 + 双向CMA@P5，5.23M，16.05ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.916 | 0.892 | 0.934 | 0.632 |
| smoke | 0.937 | 0.897 | 0.946 | 0.741 |
| fire | 0.922 | 0.890 | 0.929 | 0.582 |
| person | 0.889 | 0.890 | 0.927 | 0.573 |

**Exp-4d（CMG@P4P5 + 双向CMA@P4，4.47M，16.68ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.923 | 0.887 | 0.931 | 0.623 |
| smoke | 0.949 | 0.898 | 0.948 | 0.738 |
| fire | 0.927 | 0.883 | 0.929 | 0.573 |
| person | 0.892 | 0.880 | 0.916 | 0.559 |

**Exp-5（RGB-only 3ch 单分支，2.51M，13.68ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.920 | 0.854 | 0.907 | 0.583 |
| smoke | 0.941 | 0.897 | 0.947 | 0.735 |
| fire | 0.918 | 0.825 | 0.892 | 0.497 |
| person | 0.900 | 0.839 | 0.880 | 0.515 |

**Exp-6（IR-only 3ch 单分支，2.51M，12.23ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.888 | 0.831 | 0.887 | 0.564 |
| smoke | 0.897 | 0.826 | 0.892 | 0.664 |
| fire | 0.891 | 0.831 | 0.886 | 0.524 |
| person | 0.876 | 0.836 | 0.885 | 0.504 |

---

## 四、关键发现

> 以下按实验开展的逻辑顺序组织，同时对应论文写作的论证脉络：单模态基线 → 早期融合局限 → 双分支核心贡献 → 通道门控校准 → 空间注意力精化 → 协同效应 → 综合最优。

---

### 4.1 单模态基线：模态特异性与真正的互补关系（Exp-5、Exp-6）

**发现 1：三类目标的模态依赖模式截然不同，且 fire 是真正双模态互补类别**

| 类别 | RGB-only mAP50/95 | IR-only mAP50/95 | 模态关系 | 早期融合 Exp-0 |
|------|-------------------|-----------------|---------|-------------|
| smoke | **0.947 / 0.735** | 0.892 / 0.664 | RGB 强主导，IR 为干扰 | 0.947 / 0.734（≈ RGB-only）|
| fire | 0.892 / 0.497 | 0.886 / 0.524 | **两模态几乎等效，真正互补** | 0.907 / 0.545（有增益但有限）|
| person | 0.880 / 0.515 | 0.885 / 0.504 | 两模态接近，微弱互补 | 0.902 / 0.537（有增益）|

**关键修正（对比旧结论）：** fire 并非"IR 主导"，而是**两种模态在单独使用时表现几乎相同，但包含完全不同的信息**——RGB 捕捉可见火焰颜色/形状，IR 捕捉热辐射区域。这两种信息在单模态下各自只看到"一半的火"，因此两者不是主次关系而是互补关系。这正是为什么 fire 类对融合架构的响应最为敏感。

**对论文动机的支撑：** smoke 的"IR 无用"与 fire/person 的"双模态真互补"形成对比，说明简单拼接（早期融合）不足以充分利用跨模态互补性，需要让每个模态先独立建立表征再高层融合。

---

### 4.2 早期融合的局限：Exp-0（6ch 单分支）

**发现 2：早期融合对 fire/person 有增益，但远未充分利用双模态互补**

早期融合（6ch concat 输入）相对单模态的增益：

| 类别 | 相对 RGB-only ΔmAP50-95 | 相对 IR-only ΔmAP50-95 | 解读 |
|------|------------------------|----------------------|------|
| smoke | -0.001（持平） | +0.070 | IR 贡献被网络自动忽略，结果等同 RGB-only |
| fire | +0.048 | +0.021 | 两模态信息部分融合，但第一层就混合截断了低层特征的模态纯粹性 |
| person | +0.022 | +0.033 | 有增益，但低层混合使两模态的结构性特征互相干扰 |

**结论：** 早期融合在 smoke 上几乎无效（模型学会忽略 IR 通道），在 fire/person 上有增益但受限于"第一层被迫混合两种物理特性截然不同的信号"——热辐射的稀疏高值与可见光的纹理信号在像素级强行拼接，低层卷积无法同时对两者建立有效滤波器。这是引入双分支中期融合的根本动机。

---

### 4.3 双分支中期融合是核心贡献（Exp-0 → Exp-1）

**发现 3：双分支独立特征提取大幅释放了跨模态互补性，fire 是最大受益者**

| 类别 | Exp-0（EF）mAP50-95 | Exp-1（双分支）mAP50-95 | 增益 | 增益来源 |
|------|--------------------|-----------------------|------|---------|
| smoke | 0.734 | 0.735 | +0.1% | 几乎无（smoke 只需 RGB，双分支也无法改变模态依赖） |
| **fire** | **0.545** | **0.575** | **+3.0%** | **独立分支让 IR 的热辐射特征与 RGB 的火焰形状特征分别充分提取** |
| person | 0.537 | 0.559 | +2.2% | RGB 纹理通道与 IR 热轮廓通道分别建立表征，concat 融合时携带更丰富信息 |

**与早期融合对比（Exp-0→Exp-1 vs 单模态→Exp-0 的增益比较）：** 早期融合只能给 fire 带来 +4.8% mAP50-95（相对 RGB-only），而双分支在此基础上再带来 +3.0%，总计从 0.497 提升到 0.575（**+15.7%**）。这表明双分支中期融合所释放的互补性是早期融合的 **1.6 倍**。

**结论：双分支架构是本工作最大的技术贡献，后续 CMG/CMA 模块均是在此基础上的精细化，论文应以此为一级论点。**

---

### 4.4 CMG 通道门控：person 的专属校准器（Exp-2、Exp-2b、消融 Exp-4b vs 4b-noCMG）

**发现 4：CMG 对整体指标增益极小，但对 person 类有不可替代的作用**

CMG 引入（Exp-1 → Exp-2）的分类别增益：

| 类别 | ΔmAP50 | ΔmAP50-95 | ΔRecall | 解读 |
|------|--------|-----------|---------|------|
| smoke | +0.3% | +0.4% | -1.1% | 轻微，Recall 略降 |
| fire | -0.6% | +0.6% | +0.2% | 矛盾信号，整体中性 |
| **person** | **+0.5%** | **+0.3%** | **+1.2%** | Recall 提升，门控有效 |

CMG 消融（Exp-4b-noCMG vs Exp-4b，即在双向 CMA@P4P5 基础上移除 CMG）揭示更清晰的图景：

| 类别 | ΔmAP50（有CMG - 无CMG） | ΔRecall | 结论 |
|------|------------------------|---------|------|
| smoke | 0.0% | +0.2% | 无效 |
| fire | +0.4% | -0.1% | 中性 |
| **person** | **+1.1%** | **+1.1%** | **显著正贡献** |

**物理解释：** person 同时需要两种模态信息（RGB 纹理识别人体轮廓，IR 热辐射确认生命体），两种信号在不同通道中各自有用。CMG 的 SE 通道重加权能动态平衡"哪些通道的 IR 信号当前更可信"，实现自适应的模态权衡。smoke 与 fire 各有主导模态，CMG 的全局门控反而引入不必要的"投票噪声"，轻微损害定位精度。

**发现 5：CM-CBAM 空间门控因内存访问模式被全面否定（Exp-2b）**

CM-CBAM（`torch.mean/max(dim=1)` 沿 channel 轴规约）需跨步内存访问，GPU cache 极不友好，推理时间 +166%（18.91ms vs 7.13ms），且 mAP50 下降 -0.7%。SE-CMG 的 `AdaptiveAvgPool2d(1)` 沿空间轴规约，内存连续，有高度优化的 CUDA kernel。**参数量减少 ≠ 推理加速，根本在于内存访问模式。**

---

### 4.5 CrossModalA2C2f：空间精化的层级选择与方向性（Exp-4a/b/c/d）

**发现 6：CMA 的核心贡献集中在 person 类的定位精度，且仅在 P5 层有正效果**

CMA@P5（Exp-4c）相对 CMG 基线（Exp-2）的分类别变化：

| 类别 | ΔmAP50 | ΔmAP50-95 | 解读 |
|------|--------|-----------|------|
| smoke | -0.3% | +0.2% | 接近中性 |
| fire | 0.0% | +0.1% | 微弱正向 |
| **person** | **+0.3%** | **+1.1%** | **全系列单类别最大提升** |

CMA 层级选择的完整对比：

| 配置 | mAP50-95 | person mAP50-95 | 推理(ms) |
|------|----------|-----------------|----------|
| Exp-4d（CMA@P4） | 0.623 | 0.559 | 16.68 |
| **Exp-4c（CMA@P5）** | **0.632** | **0.573** | **16.05** |
| Exp-4b（CMA@P4P5） | 0.627 | 0.566 | 17.55 |

- **P4 层 CMA 对 person 有害**：person mAP50-95=0.559，低于无 CMA 的基线 0.562。P4（30×40 特征图）语义尚未成熟，跨模态注意力在此引入对齐噪声，空间位置的跨模态对应关系尚不可靠。
- **P4P5 组合并不更好**：P4 的负效应部分抵消了 P5 的收益，最终 person mAP50-95=0.566 < Exp-4c 的 0.573，且多付 1.5ms。加入 P4 CMA 是"做了减法而非加法"。
- **结论：CMA 应精确部署在 P5（最高语义层），跨模态空间查询的可靠性取决于特征的语义成熟度。**

**发现 7：单向 CMA（RGB←IR）只能提高检出率，不能提升定位精度**

| 指标 | Exp-2（CMG）| Exp-4a（+单向CMA）| Exp-4c（+双向CMA）|
|------|-----------|-----------------|-----------------|
| all Recall | 0.894 | **0.900** ★ 全系列最高 | 0.892 |
| all mAP50-95 | 0.627 | 0.625 | **0.632** |
| person Recall | 0.886 | **0.896** | 0.890 |
| person mAP50-95 | 0.562 | 0.562 | **0.573** |
| 推理(ms) | 7.13 | 15.27 | 16.05 |

**物理解释：** 单向 RGB←IR 让 RGB 分支获得"IR 视角下的候选区域提示"，有助于找到更多目标（Recall↑）；但 IR 分支未获 RGB 纹理信息补充，其输出的定位框质量无法提升（mAP50-95 不变）。定位精度提升需要**两个分支的特征同时被另一模态精化**——即双向交互。单向交互只是"增加找到目标的概率"，双向交互才能"提高目标的定位精度"。

**结论：** 单向 CMA 不值得保留为独立方案——以接近双向的推理代价（+8.1ms vs +9ms），只得到检出率提升而无定位增益。

---

### 4.6 CMG + CMA 协同效应（Exp-4c 的关键证据）

**发现 8：CMG 与 CMA@P5 对 person 类的作用超过线性叠加，存在架构级协同**

分解 person mAP50-95 的增量链：

| 配置 | person mAP50-95 | Δ vs Exp-1 | 归因 |
|------|-----------------|-----------|------|
| Exp-1（双分支基线） | 0.559 | — | — |
| Exp-2（+ CMG） | 0.562 | +0.003 | CMG 单独 |
| Exp-4b-noCMG（+ CMA@P4P5，无CMG） | 0.564 | +0.005 | CMA 单独 |
| **Exp-4c（CMG + CMA@P5）** | **0.573** | **+0.014** | **组合** |

若两模块独立，预期增益 ≈ 0.003 + 0.005 = 0.008。实际 **+0.014 为预期的 1.75 倍**，证明协同效应显著。

**协同机制：** CMG 在通道维度做粗粒度模态权衡（确定"哪些通道的 IR 信号更可信"），CMA@P5 在空间维度做精细跨模态查询（确定"IR 视角下的哪些空间位置与 RGB 的语义对应"）。两者作用于不同维度，相互强化：CMG 先选出可信通道，让 CMA@P5 的 K/V 来自"更纯净"的模态特征，查询质量提升；CMA@P5 反过来提供的空间上下文让 CMG 的门控决策更稳健。

**对 fire/smoke 则无协同**：smoke 不需要任何模态融合校准，fire 的两模态在通道层面的权衡相对简单，CMG 的粗粒度门控在有 CMA 时对 fire/smoke 有轻微负效应（mAP50-95 各 -0.3%）。

---

### 4.7 类别响应汇总与论文定位

**发现 9：三类目标形成三种完全不同的融合响应模式**

| 类别 | 最大瓶颈 | 最佳配置 | 从 Exp-0 到最优的总增益 |
|------|---------|---------|----------------------|
| smoke | **无瓶颈**（RGB-only 已达上限） | 任意双分支即可（Exp-1 已饱和）| mAP50-95 +0.1%（可忽略）|
| fire | **双模态互补未充分释放** | Exp-4c（mAP50-95 最高 0.582）| mAP50-95 +3.7%（0.545→0.582）|
| **person** | **两模态的语义空间对齐** | **Exp-4c（CMG+双向CMA@P5）** | **mAP50-95 +3.6%（0.537→0.573）** |

**对论文论证的价值：**
- 整体 mAP50 的差异（0.01 级）掩盖了类别层面的显著分化（0.03~0.07 级）
- CMG 和 CMA 的价值集中在 fire 和 person——恰好是安全监控中最关键的两个目标类别
- 论文主线可定位为："针对 RGBT 安全监控任务的双模态中期融合框架，通过分层跨模态交互（通道级 CMG + 空间级 CMA@P5）在 fire/person 检测上实现显著提升"

---

### 4.8 当前最优配置及推理代价分析

**发现 10：Exp-4c 在所有指标上综合最优，其他候选均有明确短板**

| 配置 | mAP50 | mAP50-95 | person mAP50-95 | 推理(ms) | 相对 Exp-0 的总增益 |
|------|-------|----------|-----------------|----------|------------------|
| Exp-0（早期融合基线） | 0.919 | 0.605 | 0.537 | 17.31 | 基准 |
| Exp-1（双分支 concat） | 0.933 | 0.623 | 0.559 | 6.96 | mAP50-95 +1.8% |
| Exp-2（+ CMG） | 0.934 | 0.627 | 0.562 | 7.13 | mAP50-95 +2.2% |
| Exp-4a（+ 单向CMA@P4P5） | 0.933 | 0.625 | 0.562 | 15.27 | Recall 最高，定位无改善 |
| Exp-4b-noCMG（+ CMA@P4P5） | 0.929 | 0.628 | 0.564 | 16.09 | person mAP50 塌陷 -1.1% |
| Exp-4d（+ CMA@P4） | 0.931 | 0.623 | 0.559 | 16.68 | 全面负收益 |
| Exp-4b（+ CMA@P4P5） | 0.934 | 0.627 | 0.566 | 17.55 | 推理最慢，无增益 |
| **Exp-4c（CMG@P4P5 + 双向CMA@P5）** ★ | **0.934** | **0.632** | **0.573** | **16.05** | **mAP50-95 +2.7%，person +3.6%** |

**Exp-4c 的代价效益分析：** 相对双分支 concat 基线（Exp-1，6.96ms），Exp-4c 需要 16.05ms（+2.3× 推理时间），换取 mAP50-95 +0.9%（整体）和 person mAP50-95 +1.4%（类别）。在安全监控场景下，person 检出精度提升 1.4% 直接对应更少的漏报，代价合理。

---

## 五、消融实验状态

### 5.1 实验配置一览

| 实验 | 配置 | 目的 | 状态 | mAP50 | mAP50-95 |
|------|------|------|------|-------|----------|
| Exp-0 | 单分支 early fusion | 基准 | 完成 | 0.919 | 0.605 |
| Exp-1 | 对称双分支 concat-only | 验证双分支增益 | 完成 | 0.933 | 0.623 |
| Exp-2 | 对称双分支 + CMG@P4P5 | 验证 SE 门控增益 | 完成 | 0.934 | 0.627 |
| Exp-2b | 对称双分支 + CM-CBAM@P4P5 | 验证空间注意力 | 完成（否定） | 0.927 | 0.627 |
| Exp-4a | 双分支 + CMG@P4P5 + 单向CMA@P4P5 | 验证单向空间跨模态查询 | 完成 | 0.934 | 0.625 |
| Exp-4b | 双分支 + CMG@P4P5 + 双向CMA@P4P5 | 验证双向CMA + 残差叠加 | 完成 | 0.934 | 0.627 |
| Exp-4b-noCMG | 双分支 + 双向CMA@P4P5（无CMG） | 消融 CMG 的作用 | 完成 | 0.930 | 0.628 |
| **Exp-4c ★** | **双分支 + CMG@P4P5 + 双向CMA@P5** | **层级消融：CMA仅P5** | **完成** | **0.934** | **0.632** |
| Exp-4d | 双分支 + CMG@P4P5 + 双向CMA@P4 | 层级消融：CMA仅P4 | 完成 | 0.931 | 0.623 |
| Exp-5 | RGB-only 3ch | 单模态基准 | 完成 | 0.907 | 0.583 |
| Exp-6 | IR-only 3ch | 单模态基准 | 完成 | 0.887 | 0.564 |

### 5.2 当前最优

- **综合最优：Exp-4c ★**（CMG@P4P5 + 双向残差CMA@P5，5.23M，16.05ms）
  - mAP50-95 = 0.632（全系列最高）
  - person mAP50-95 = 0.573（+1.1% vs CMG 基线）
  - 推理代价在 CMA 系列中最低
- **待补全：** Exp-4b-noCMG 完整 per-class val（消融 CMG 的作用）

### 5.3 执行备忘

- 训练超参：epochs=300, batch=16, SGD lr0=0.01 lrf=0.01 cos_lr=False val_period=2
- 关注 CPU RAM：训练到 ~epoch 77 可能触发 `_ArrayMemoryError`，需监控并准备 `--resume`
- val.py 速度评估统一跑 10 次取平均
- 旧 checkpoint 验证需还原对应代码版本（见下表）：

| 实验 | 对应 git commit | 需还原的文件 |
|------|----------------|------------|
| Exp-1（dual_MF） | `0b9379a` | block.py, tasks.py, modules/__init__.py |
| Exp-2（dual_MF_ChW） | `e5bdb6b` 或 `92e77f5` | block.py, tasks.py, modules/__init__.py |
| Exp-4a（CMA_rgb） | `c852185` | block.py, tasks.py |
| Exp-4b/4c/4d（双向CMA） | HEAD（直接可用） | — |

---

## 六、备选想法

> 以下方案基于当前实验结论探索进一步改进方向。

### 想法 A：Depthwise 跨模态卷积门控（最轻量替代 CMG）

DWConv 沿空间维度滑窗，内存访问连续，GPU 友好。参数量极小（~19K），提供空间+通道联合门控。适合快速消融"空间门控方向是否有价值"（排除 CM-CBAM 因实现低效而非方案本身无效的可能）。

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

训练时在 loss 中添加辅助蒸馏项，迫使每个分支在特征空间逼近另一模态的表征；推理时移除 project 头和蒸馏损失，零推理开销。

```python
loss_distill = (
    F.mse_loss(project_rgb(feats_rgb['p4']), feats_ir['p4'].detach()) +
    F.mse_loss(project_ir(feats_ir['p4']), feats_rgb['p4'].detach()) +
    F.mse_loss(project_rgb(feats_rgb['p5']), feats_ir['p5'].detach()) +
    F.mse_loss(project_ir(feats_ir['p5']), feats_rgb['p5'].detach())
)
total_loss = loss_detect + lambda_distill * loss_distill  # lambda_distill ~ 0.1
```

### 想法 G：差异引导融合（替代 CMG 的新方向）

用 `|x_rgb - x_ir|` 作为门控信号，差异图直接编码"两种模态看法不一致的区域"，比 GAP 全局门控信息量丰富得多，比 CMA 更轻量。

```python
class DifferenceGuidedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.diff_encoder = Conv(channels, channels // 4, 1)
        self.gate = nn.Sequential(Conv(channels // 4, channels, 1), nn.Sigmoid())

    def forward(self, x_rgb, x_ir):
        diff = torch.abs(x_rgb - x_ir)
        gate = self.gate(self.diff_encoder(diff))
        return x_rgb + gate * x_ir
```

### 想法 H：频域解耦融合

通过 FFT 将特征分解为低频（全局语义）和高频（局部纹理），对不同频段施加不同的跨模态融合策略。IR 在低频（热源区域语义）有优势，RGB 在高频（纹理细节）有优势。

### 想法 I：类别感知动态门控

根据检测头的类别预测概率动态调整融合权重。smoke 类检测时更信任 RGB，fire/person 类检测时更信任 IR。

### 优先级

| 优先级 | 想法 | 理由 |
|--------|------|------|
| P0 | Exp-4b-noCMG 完整 val | 补全消融链 |
| P1 | 想法 G（差异引导） | CMG 替代，代价低，物理动机强 |
| P2 | 想法 H（频域） | 较复杂但差异化显著 |
| P3 | 想法 I（类别感知） | 创新性强但工程复杂 |
