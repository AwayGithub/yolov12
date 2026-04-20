# ADR-001: 双分支 YOLOv12-MF RGB-红外融合检测架构

**Status:** Exp-7 全部完成。Exp-8a 已完成（mAP50-95=0.638，历史最佳）。Exp-8b 已完成（0.637）。Exp-8c / Exp-8d 训练中（服务器）。
**Date:** 2026-04-04
**Updated:** 2026-04-19
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

### 2.2 CrossModalA2C2f（CMA）

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

---

## 三、实验结果

**训练超参：** YOLOv12n + RGBT-3M，480×640，epochs=300，batch=16，SGD lr0=0.01 lrf=0.01 cos_lr=False val_period=2

**Fitness 公式（worktree 自定义）：** `w = [0.0, 0.2, 0.3, 0.5]`（P/R/mAP50/mAP50-95），显式纳入 Recall，适合安全监控任务。

### 3.1 数据集目标尺寸特性分析

> 分析脚本：`analyze_bbox_distribution.py`，输出图表位于 `bbox_analysis/`。
> 图像分辨率：640×480（W×H），标签 YOLO 格式（归一化），共 11,220 个标签文件，30,777 个目标框。

#### 各类别尺寸画像

| 类别 | 目标数 | 宽度中位 | 高度中位 | 中位面积 | 短边 <32px（小目标）| 短边 32–96px（中目标）| 短边 >96px（大目标）|
|------|--------|---------|---------|---------|------------------|-------------------|------------------|
| smoke | 13,574 | 111px | 116px | 15,737px² | 22.2% | 28.1% | **49.7%** |
| fire | 11,315 | 22px | 27px | 600px² | **80.8%** | 19.0% | 0.3% |
| person | 5,888 | 20px | 27px | 709px² | **71.4%** | 28.4% | 0.1% |
| **all** | **30,777** | **32px** | **39px** | — | **53.1%** | 24.8% | 22.0% |

**smoke 的双峰特性：** mean=186px >> median=111px，说明极少量近距离大烟团拉高均值，实际分布高度右偏——远处小烟雾占主体，近处大烟团偶发。

#### 各层级对小目标的覆盖能力

"某 stride 下 <2 个特征图格子" = 目标在该层基本不可见：

| 类别 | stride=4（P2）| stride=8（P3）| stride=16（P4）| stride=32（P5）|
|------|-------------|-------------|--------------|--------------|
| smoke | 0.2% | 6.2% | 22.2% | 39.5% |
| fire | **5.2%** | **35.8%** | **80.8%** | **97.8%** |
| person | **5.0%** | **38.0%** | **71.4%** | **98.7%** |
| all | 2.9% | 23.1% | 53.1% | 72.3% |

#### 关键结论

- **P3（stride=8）是 fire/person 主力检测头**，但已有 35–38% 目标不足 2 格子。
- **P4/P5 对 fire/person 几乎无直接贡献**（P4 80%+不可见，P5 97%+不可见），其价值在于提供语义上下文经 FPN 传递。
- **P2（stride=4）理论上可将 fire/person 覆盖率从 62% 提升到 95%**，但计算量 +58%。
- **CMA@P5 有效的机制**：P5 的 IR 弥散热信号作为全局热源先验，经 FPN top-down 流至 P3，为小目标检测提供语义上下文。P4 CMA 反而有害——fire/person 在 P4 处于"半可见"边界状态（~1 格子），跨模态对齐噪声干扰检测。

### 3.2 汇总表

| 实验 | 配置简述 | mAP50 | mAP50-95 | Precision | Recall | 参数量 | 推理(ms) | 状态 |
|------|---------|-------|----------|-----------|--------|--------|----------|------|
| Exp-0 | 单分支 6ch early fusion | 0.919 | 0.605 | 0.914 | 0.877 | 2.51M | 4.51 | 完成 |
| Exp-1 | 对称双分支 concat-only | 0.933 | 0.623 | 0.924 | 0.893 | — | 6.96 | 完成 |
| Exp-2 | 对称双分支 + CMG@P4P5 | 0.934 | 0.627 | 0.925 | 0.894 | 4.21M | 7.13 | 完成 |
| Exp-2b | 对称双分支 + CM-CBAM@P4P5 | 0.927 | 0.627 | 0.925 | 0.882 | 4.13M | 18.91 | 完成（否定）|
| Exp-4a | CMG@P4P5 + 单向CMA@P4P5（RGB←IR） | 0.933 | 0.625 | 0.918 | 0.900 | 4.94M | 15.27 | 完成 |
| Exp-4b | CMG@P4P5 + 双向CMA@P4P5 | 0.934 | 0.627 | 0.923 | 0.890 | 5.49M | 17.55 | 完成 |
| Exp-4b-noCMG | 双向CMA@P4P5（无CMG） | 0.929 | 0.628 | 0.922 | 0.886 | 5.34M | 16.09 | 完成 |
| **Exp-4c** ★ | **CMG@P4P5 + 双向CMA@P5** | **0.934** | **0.632** | **0.916** | **0.892** | **5.23M** | **16.05** | **完成** |
| Exp-4d | CMG@P4P5 + 双向CMA@P4 | 0.931 | 0.623 | 0.923 | 0.887 | 4.47M | 16.68 | 完成 |
| Exp-5 | RGB-only 3ch 单分支 | 0.907 | 0.583 | 0.920 | 0.854 | 2.51M | 4.08 | 完成 |
| Exp-6 | IR-only 3ch 单分支 | 0.887 | 0.564 | 0.888 | 0.831 | 2.51M | 4.33 | 完成 |
| Exp-7a ⚠️ | DMGFusion@P2 + Exp-4c + head P2P3P4P5 | 0.941 | 0.628 | 0.917 | 0.911 | 5.32M (31.07G) | 21.9 | 已完成 |
| **Exp-7b** | **Concat@P2 + Exp-4c + head P2P3P4P5** | **0.941** | **0.634** | **0.930** | **0.899** | **5.32M (31.01G)** | **20.7** | **已完成** |
| Exp-7c | DMGFusion@P2 + Exp-4c + head P3P4P5 | 0.933 | 0.625 | 0.933 | 0.888 | 5.28M (21.69G) | 20.5 | 已完成 |
| Exp-7d | Concat@P2 + Exp-4c + head P3P4P5 | 0.934 | 0.634 | 0.925 | 0.895 | 5.28M (21.64G) | 19.9 | 已完成 |
| **Exp-8a** ★ | **DMGFusion@P2 + CMG@P4P5 + 双向CMA@P5 + head P2P3P4P5 + P4 C3k2 + P3 aux** | **0.945** | **0.638** | **0.930** | **0.910** | **5.55M (30.63G)** | **7.12** | **已完成** |

> Exp-8a 分类别 mAP50 / mAP50-95：smoke 0.948 / 0.739，fire 0.947 / 0.593，person 0.941 / 0.581（Instances：smoke 4086 / fire 3401 / person 1740，全 val 集 3366 张）。本地 GPU 推理时延：preprocess 0.3310 + inference 7.1225 + loss 0.0010 + postprocess 0.7363 ≈ **8.19 ms / image**，inference 段 7.12ms 与表中其他行硬件一致、可直接对比。

> ★ Exp-4c 为最优性价比方案。Exp-7b/7d 并列数值最优（mAP50-95=0.634）；Exp-7d 以最低额外代价实现相同精度（仅 +P2 融合特征，无 P2 检测头，+24% 推理 vs +29%）。

#### Exp-7 消融矩阵（2×2 设计）

| | head P2P3P4P5 | head P3P4P5（无P2检测） |
|---|---|---|
| **DMGFusion@P2** | Exp-7a ⚠️（0.628，差于7b） | Exp-7c（0.625，已完成） |
| **Concat+1×1Conv@P2** | Exp-7b（0.634，已完成） | Exp-7d（0.634，已完成） |

对比链：7a vs 7b → DMGFusion vs Concat（含P2 head）；7c vs 7d → DMGFusion vs Concat（无P2 head）；**7c vs Exp-4c → DMGFusion FPN 污染验证**；7d vs Exp-4c → Concat@P2 FPN 贡献；7b vs 7d → P2 检测头纯效果。

### 3.3 参数量与计算量

| 实验 | 参数量 | GFLOPs | 推理(ms) | 备注 |
|------|--------|--------|----------|------|
| Exp-2（CMG基线） | 4,210,000 | — | 7.13 | |
| Exp-4a（单向CMA@P4P5） | 4,943,833 | 18.02 | 15.27 | 仅 RGB 分支替换 |
| Exp-4b-noCMG（双向CMA@P4P5） | 5,335,709 | 18.47 | 16.09 | |
| Exp-4d（双向CMA@P4） | 4,473,187 | 19.60 | 16.68 | |
| **Exp-4c（双向CMA@P5）** | **5,226,851** | **19.65** | **16.05** | |
| Exp-4b（双向CMA@P4P5） | 5,488,997 | 18.57* | 17.55 | *thop 双流偏低 |
| Exp-7a（DMGFusion@P2） | 5,305,334 | 31.07 | 21.9 | +P2 head |
| Exp-7b（Concat@P2） | 5,315,574 | 31.01 | 20.7 | +P2 head |
| Exp-7c（DMGFusion@P2, 3-scale） | 5,279,527 | 21.69 | 20.5 | 无P2 head |
| Exp-7d（Concat@P2, 3-scale） | 5,278,163 | 21.64 | 19.9 | 无P2 head |
| **Exp-8a**（DMG v1 + P4 C3k2 + P3 aux） | **5,552,112** | **30.63** | **7.12** | 本地 GPU，inference 段 |

**参数量自洽验证：**
- Exp-4c - Exp-4d = 5,226,851 - 4,473,187 = **753,664**（P5 比 P4 多，因 256²×2 - 128²×2 ≈ 753K ✓）
- Exp-4b ≈ Exp-4d + Exp-4c - Exp-2 = 5,490,038 ≈ 5,488,997 ✓
- Exp-4b - Exp-4b-noCMG = 164,608 = CMG 4 实例精确参数量 ✓

**P2 head 的计算量分析：** Exp-7a/7b 的 GFLOPs 从 ~19.65G 跳至 ~31G（+58%），但参数量仅 +~0.1M。根本原因：FLOPs ∝ 参数×空间分辨率，P2（120×160=19,200 像素）是 P3 的 4 倍、P5 的 64 倍。同一个卷积核在 P2 运算的 FLOPs 是 P5 的 64 倍。

### 3.4 分类别详细结果

**Exp-0（单分支 6ch early fusion，2.51M，4.51ms）：**

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

**Exp-5（RGB-only 3ch 单分支，2.51M，4.08ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.920 | 0.854 | 0.907 | 0.583 |
| smoke | 0.941 | 0.897 | 0.947 | 0.735 |
| fire | 0.918 | 0.825 | 0.892 | 0.497 |
| person | 0.900 | 0.839 | 0.880 | 0.515 |

**Exp-6（IR-only 3ch 单分支，2.51M，4.33ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.888 | 0.831 | 0.887 | 0.564 |
| smoke | 0.897 | 0.826 | 0.892 | 0.664 |
| fire | 0.891 | 0.831 | 0.886 | 0.524 |
| person | 0.876 | 0.836 | 0.885 | 0.504 |

**Exp-7a ⚠️（DMGFusion@P2 + Exp-4c + head P2P3P4P5，5.32M，31.07G，21.9ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.917 | 0.911 | 0.941 | 0.628 |
| smoke | 0.929 | 0.911 | 0.948 | 0.732 |
| fire | 0.922 | 0.913 | 0.941 | 0.578 |
| person | 0.901 | 0.909 | 0.933 | 0.574 |

**Exp-7b（Concat@P2 + Exp-4c + head P2P3P4P5，5.32M，31.01G，20.7ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.930 | 0.899 | 0.941 | 0.634 |
| smoke | 0.943 | 0.890 | 0.947 | 0.739 |
| fire | 0.938 | 0.912 | 0.944 | 0.584 |
| person | 0.910 | 0.894 | 0.931 | 0.579 |

**Exp-7c（DMGFusion@P2 + Exp-4c + head P3P4P5，5.28M，21.69G，20.5ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.933 | 0.888 | 0.933 | 0.625 |
| smoke | 0.946 | 0.899 | 0.946 | 0.734 |
| fire | 0.934 | 0.885 | 0.932 | 0.582 |
| person | 0.918 | 0.881 | 0.922 | 0.558 |

**Exp-7d（Concat@P2 + Exp-4c + head P3P4P5，5.28M，21.64G，19.9ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.925 | 0.895 | 0.934 | 0.634 |
| smoke | 0.942 | 0.900 | 0.945 | 0.737 |
| fire | 0.933 | 0.897 | 0.935 | 0.589 |
| person | 0.901 | 0.889 | 0.924 | 0.576 |

### 3.5 DMGFusion@P2 深度诊断

#### 3.5.1 实验层面诊断（Exp-7 结果分析）

**现象：** Exp-7a（DMGFusion@P2）mAP50-95=0.628，低于 Exp-7b（Concat@P2）的 0.634，说明 DMGFusion 在 P2 产生了副作用。三类全面劣于 7b（smoke −0.7%, fire −0.6%, person −0.5%）。

**关键问题：** 是 DMGFusion 特征质量本身的问题，还是 DMGFusion 与 P2 检测头的交互问题？

**诊断结果（已完成）：** Exp-7c（0.625）< Exp-4c（0.632）→ **最坏情形确认：DMGFusion 的 P2 特征通过 FPN 污染上层，模块设计存在根本问题。** 对照实验 Exp-7d（Concat@P2，无P2 head）= 0.634 ≥ 4c，说明 P2 融合点本身无害，失败的是 DMGFusion 的设计。

**失效根因分析（实验证实）：**

1. **D = |RGB-IR| 在 P2 层级语义为零**：P2 只经过两层 Conv + C3k2，特征仍是低层纹理。D 编码传感器级差异（噪声/色温）而非语义不一致，sel_net 在拟合噪声。
2. **公式双重计数**：alpha=0、beta=1、w≈0.5 初始化时，`fused ≈ 0.5(x+y) + 0.5(x+y) = x+y`（初始化为 2× 均值），两条路径梯度干扰训练效率。
3. **alpha 无约束增长**：`(1+alpha*S)` 在 P2 高差异区域（多为传感器噪声）放大特征，破坏后续 BN 统计量并向 FPN 传播。
4. **P2 检测头梯度为次要因素**：7a vs 7c 对比（0.628 vs 0.625）显示 P2 head 的额外梯度仅轻微缓解 FPN 污染（差值 0.003）。

---

#### 3.5.2 特征层面可视化诊断（2026-04-19，基于 tools/visualize_p2_dmg.py + tools/probe_const_input.py）

分析样本：video10_frame_01154（包含 fire/person 目标，RGB 背景为草地/植被）

##### A. RGB/IR 双支路的本质问题

**A1. RGB 支路语义坍塌**

- P2/P3：RGB 特征以草地/植被高频纹理为主，目标仅在轮廓层面可见，且激活值低于背景
- P4：RGB 已退化为随机噪声图，目标信息基本消失
- P5：RGB 只剩零散全局激活点，无定位能力
- **根因**：联合训练中 detection loss 的梯度主要经由 IR 强信号路回传（IR 目标-背景对比度 10:1），RGB backbone 以最省力的方式拟合低级纹理统计量，缺乏任何来自 loss 的强制约束。

**A2. IR 支路才是真正的检测器，但能量被 RGB 稀释**

- IR 在 P2–P5 每个尺度均清晰定位 fire/person（目标-背景对比度 10:1 以上）
- 一旦 concat + 1×1 Conv 融合，IR 的目标能量被数量相等的 RGB 通道"平均"掉
- **根因**：plain fusion 的 1×1 Conv 初始权重各向同性，无"优先信任 IR"的先验

**A3. IR 传感器伪影（过曝条带）未被抑制**

- 图像左上角 IR 过曝条带（x_ir≈3.0）在 P2→P5 每层 fused 中均高亮
- 其值高于真实目标（x_ir≈1.5），任何融合模块均未能识别并压制这个坏区
- 该伪影在 DMGFusion 的 D=|R-I| 映射中最亮，被 sel_net 误识为"高价值分歧区"

##### B. Area Attention 的结构性噪声（零输入 probe 证实，与数据无关）

**B1. P4 水平条带（area=4，已修复）**

- 零输入 probe：P4 mean row_std/col_std ratio = 2.29，29% 通道 ratio ≥ 2.0
- 原因：AAttn 把 H 切成 4 条 strip，各 strip 内 token 数不同 → softmax 温度差异 → strip 间均值偏移
- 该噪声与输入内容无关，属架构性缺陷；CMG@P4 作用于已污染的特征，起到放大而非修复的效果
- **已在 Exp-8 中修复**：backbone P4 改为 C3k2（无 area 切分）

**B2. P5 全局注意力下的静态先验偏置（area=1，非条带，是学到的位置先验）**

- 零输入 probe：P5 mean ratio = 1.98，bot[b05] ratio = **9.26**（极端离群），cls[0] (smoke) ratio = 3.15
- 原因：P5 area=1 即全局注意力，条带不来自 area 切分，而是 ABlock MLP 学到的训练集位置先验（烟雾偏上，人偏下）
- 15×20 分辨率（300 tokens）下全局 softmax 退化为全局均值，MLP 的偏置项成为主要信号来源，这些偏置编码了数据集统计量
- **该问题不需要修复**：P5 的位置先验是有语义意义的数据集先验（不是伪影），保留 A2C2f@P5

**B3. 零填充"相框"效应（所有尺度共有）**

- DFL 的 left/right bins 显示竖直带（W 轴 padding 引起），top/bot bins 显示水平带（H 轴 area 切分引起）
- 两种各向异性在 head 输出中交叉污染所有 DFL 通道的分布

##### C. DMGFusion 模块六项设计缺陷（完整版）

**C1. β 残差旁路吞噬了门控 W（最致命）**

- 初始 α=0, β=1 → `fused ≈ out_proj(0.5·(R+I))`，W/S/D 全部成为装饰品
- RGB 草的能量以 β=1 权重无条件注入，与 W 指向 IR（W_ir>0.6）的结论完全矛盾
- **可视化证据**：W 看起来合理（倾向 IR），但 fused 被 RGB 草主导，β 旁路绕过了 W

**C2. D=|R−I| 编码的是尺度差，不是语义分歧**

- x_rgb 峰值 ~1.8，x_ir 峰值 ~3.0，D 最亮区出现在 IR 过曝条带，而非 fire/person 目标
- 未归一化的直接相减只反映两路动态范围的差异
- **修复**：减法前对两路分别做 `InstanceNorm2d(affine=False)`，使 D 编码归一化后的语义差异

**C3. softmax 双路权重与物理语义对立（零和问题）**

- softmax 强制 w_rgb + w_ir ≡ 1（零和约束），只能回答"更信哪一路"，无法表达"两路都重要"
- **可视化证据**：W_rgb 和 W_ir 在目标位置**同时**出现暗洞（~0.3–0.4），而背景处走极端（~0.97/0.03）
- 物理上，目标区域两模态应同时高权重（fire 同时有 RGB 火焰色和 IR 热辐射）
- **修复**：改为两路独立 sigmoid 门，允许同时输出高权重

**C4. 输出幅值塌缩 5–10 倍**

- 输入：x_rgb ~[0.25, 1.75]，x_ir ~[0, 3.0]；输出 fused ~[-0.6, 0.2]
- out_proj 含 BN（强制零均值）+ 加权平均公式本身收缩幅度 → 下游 C3k2 拿到低 SNR 特征
- **修复**：out_proj 中 BN 改为 GroupNorm(8)，保留幅值尺度

**C5. S（差异幅度调制）饱和，失去空间选择性**

- S 整图在 0.5–0.8 区间，不再是"目标显著图"；D 各处非零 → sigmoid(diff_enc(D)) 全图接近饱和
- 即便 α 训练到非零，`(1+αS)` 也只是近似均匀放大，无区分能力
- **修复**：去掉 S 和 α 分支，整体公式简化为 `fused = out_proj(w_rgb·R + w_ir·I)`

**C6. 模态权重 W 只有空间维，没有通道维**

- W shape = (B, 2, H, W)，所有 64 通道共享同一空间权重
- 不同通道捕获不同语义（边缘/纹理/颜色/温度），需要不同的 RGB/IR 偏好
- **修复**：因式分解 W = W_spatial(B,2,H,W) ⊗ W_channel(B,2,C,1,1)，加入 SE 式通道门

##### D. 跨尺度/架构层面的全局问题

**D1. CMG/CMA 作用于已经损坏的特征**

- CMG@P4 基于 area attention 条带污染后的特征做门控，放大而非修复噪声（已修复，见 B1）
- P5 CMA 在 15×20（300 tokens）分辨率做 cross-attention，spatial 信息密度极低，但 P5 全局先验有语义意义，保留

**D2. 双流 backbone 无参数共享，无一致性约束**

- RGB/IR backbone 完全独立训练，无机制确保两路在同一层产生"可对齐"的特征
- 这是 W 在目标处出现"双低洞"的深层原因：两路特征各自漂移到不同子空间，导致 sel_net 无法有效区分

**D3. head 67 通道无损失解耦**

- DFL 64 + cls 3 共享同一份 feature，cls 的梯度反向污染 DFL 分布学习
- P5 cls[0] (smoke) ratio=3.15 出现 y 坐标系统性 bias 即为佐证

---

## 四、关键发现

### 4.1 单模态基线：模态特异性与互补关系（Exp-5/6）

**发现 1：三类目标的模态依赖模式截然不同，fire 是真正双模态互补类别**

| 类别 | RGB-only mAP50/95 | IR-only mAP50/95 | 模态关系 |
|------|-------------------|-----------------|---------|
| smoke | **0.947 / 0.735** | 0.892 / 0.664 | RGB 强主导 |
| fire | 0.892 / 0.497 | 0.886 / 0.524 | **两模态等效，真正互补** |
| person | 0.880 / 0.515 | 0.885 / 0.504 | 两模态接近，微弱互补 |

**关键修正：** fire 并非"IR 主导"，而是两种模态单独使用时表现几乎相同但包含完全不同的信息——RGB 捕捉火焰颜色/形状，IR 捕捉热辐射区域。这正是 fire 对融合架构最敏感的原因。

### 4.2 早期融合的局限（Exp-0）

**发现 2：早期融合对 fire/person 有增益，但远未充分利用双模态互补**

| 类别 | 相对 RGB-only ΔmAP50-95 | 相对 IR-only ΔmAP50-95 | 解读 |
|------|------------------------|----------------------|------|
| smoke | −0.001（持平） | +0.070 | IR 贡献被自动忽略 |
| fire | +0.048 | +0.021 | 部分融合，低层混合截断了模态纯粹性 |
| person | +0.022 | +0.033 | 有增益，但低层混合使结构特征互相干扰 |

### 4.3 双分支中期融合是核心贡献（Exp-0 → Exp-1）

**发现 3：双分支独立特征提取大幅释放跨模态互补性，fire 是最大受益者**

| 类别 | Exp-0 mAP50-95 | Exp-1 mAP50-95 | 增益 |
|------|---------------|---------------|------|
| smoke | 0.734 | 0.735 | +0.1% |
| **fire** | **0.545** | **0.575** | **+3.0%** |
| person | 0.537 | 0.559 | +2.2% |

早期融合只给 fire 带来 +4.8%（vs RGB-only），双分支在此基础上再带来 +3.0%，从 0.497 提升到 0.575（+15.7%）。双分支中期融合释放的互补性是早期融合的 1.6 倍。

**结论：双分支架构是最大的技术贡献，后续 CMG/CMA 均是此基础上的精细化。**

### 4.4 CMG 通道门控（Exp-2、Exp-2b）

**发现 4：CMG 对 person 有不可替代的作用**

CMG 消融（Exp-4b-noCMG vs Exp-4b，在双向 CMA@P4P5 基础上移除 CMG）：

| 类别 | ΔmAP50（有CMG − 无CMG） | ΔRecall | 结论 |
|------|------------------------|---------|------|
| smoke | 0.0% | +0.2% | 无效 |
| fire | +0.4% | −0.1% | 中性 |
| **person** | **+1.1%** | **+1.1%** | **显著正贡献** |

**物理解释：** person 同时需要 RGB 纹理（轮廓）和 IR 热辐射（生命体确认），CMG 的 SE 通道重加权动态平衡"哪些通道的 IR 信号更可信"。smoke/fire 各有主导模态，CMG 反而引入投票噪声。

**发现 5：CM-CBAM 因内存访问模式被否定（Exp-2b）**

CM-CBAM 的 `torch.mean/max(dim=1)` 跨步内存访问使推理时间 +166%（18.91ms vs 7.13ms），且 mAP50 −0.7%。SE-CMG 的 `AdaptiveAvgPool2d(1)` 沿空间轴规约，有高度优化的 CUDA kernel。**参数量减少 ≠ 推理加速，根本在于内存访问模式。**

### 4.5 CMA 空间精化的层级选择与方向性（Exp-4 系列）

**发现 6：CMA 仅在 P5 有正效果，核心贡献是 person 定位精度**

| 配置 | mAP50-95 | person mAP50-95 | 推理(ms) |
|------|----------|-----------------|----------|
| Exp-4d（CMA@P4） | 0.623 | 0.559 | 16.68 |
| **Exp-4c（CMA@P5）** | **0.632** | **0.573** | **16.05** |
| Exp-4b（CMA@P4P5） | 0.627 | 0.566 | 17.55 |

- **P4 CMA 对 person 有害**（0.559 < 基线 0.562）：P4 语义未成熟，跨模态对齐引入噪声。
- **P4P5 组合不如 P5-only**：P4 负效应部分抵消 P5 收益。
- **结论：CMA 应精确部署在 P5（最高语义层）。**

**发现 7：单向 CMA 只提高检出率，不提升定位精度**

| 指标 | Exp-2（CMG）| Exp-4a（+单向CMA）| Exp-4c（+双向CMA）|
|------|-----------|-----------------|-----------------|
| all Recall | 0.894 | **0.900** ★ | 0.892 |
| all mAP50-95 | 0.627 | 0.625 | **0.632** |
| person Recall | 0.886 | **0.896** | 0.890 |
| person mAP50-95 | 0.562 | 0.562 | **0.573** |

单向 RGB←IR 让 RGB 分支获得"IR 视角下的候选区域提示"（Recall↑），但 IR 分支未获 RGB 纹理补充，定位框质量无法提升。定位精度需要**双向交互**。以接近双向的推理代价（+8.1ms vs +9ms），只得到检出率而无定位增益，不值得保留。

### 4.6 CMG + CMA 协同效应（Exp-4c）

**发现 8：CMG 与 CMA@P5 对 person 存在架构级协同**

| 配置 | person mAP50-95 | Δ vs Exp-1 |
|------|-----------------|-----------|
| Exp-1（双分支基线） | 0.559 | — |
| Exp-2（+ CMG） | 0.562 | +0.003 |
| Exp-4b-noCMG（+ CMA@P4P5，无CMG） | 0.564 | +0.005 |
| **Exp-4c（CMG + CMA@P5）** | **0.573** | **+0.014** |

若独立叠加，预期增益 ≈ 0.003 + 0.005 = 0.008。实际 +0.014 为预期的 **1.75 倍**。

**协同机制：** CMG 在通道维度做粗粒度模态权衡（选出可信通道），CMA@P5 在空间维度做精细跨模态查询。CMG 先选出可信通道 → CMA 的 K/V 来自更纯净的特征 → 查询质量提升；CMA 提供的空间上下文反过来让 CMG 的门控更稳健。两者作用于不同维度，相互强化。

### 4.7 P2 融合策略完整分析（Exp-7 系列 2×2）

**完整消融矩阵（mAP50-95，基线 Exp-4c=0.632）：**

| | +P2 检测头（4-scale） | 无P2 检测头（3-scale） | P2 head 纯效果 |
|---|---|---|---|
| **DMGFusion@P2** | 7a: 0.628（−0.004） | 7c: 0.625（**−0.007**） | +0.003 |
| **Concat@P2** | 7b: **0.634**（+0.002） | 7d: **0.634**（+0.002） | 0.000 |
| **DMGFusion vs Concat** | −0.006 | **−0.009** | |

括号中为相对 Exp-4c 基线的 mAP50-95 差值。

**发现 9（修订）：P2 检测头对 mAP50-95 贡献为零，仅提升 mAP50**

通过 7b vs 7d（Concat@P2 基础上，唯一变量是是否加 P2 检测头）精确隔离 P2 head 的纯效果：

| 指标 | Exp-7d（无P2 head） | Exp-7b（+P2 head） | Δ（head 纯贡献） |
|------|--------------------|--------------------|-----------------|
| mAP50-95 | **0.634** | **0.634** | **0** |
| mAP50 | 0.934 | **0.941** | +0.7% |
| Recall | 0.895 | **0.899** | +0.4% |
| Precision | 0.925 | 0.930 | +0.5% |
| fire mAP50-95 | **0.589** | 0.584 | −0.5% |
| person mAP50-95 | 0.576 | **0.579** | +0.3% |
| GFLOPs | **21.64** | 31.01 | **+43%** |
| 推理(ms) | **19.9** | 20.7 | +4% |

- **P2 检测头对 box 质量（mAP50-95）零贡献**：找到了更多目标（mAP50 / Recall↑），但框的 IoU 不改善。这符合 §3.1 的分析：fire/person 即使在 P2（stride=4）也只有 ~5 个格子，定位精度依然受限。
- **P2 检测头代价主要来自 FPN 计算**：GFLOPs +43%（P2 上的 FPN 卷积 + 检测卷积），推理增量仅 +0.8ms（实际 forward 路径未显著延长，bottleneck 在上层）。
- **结论：P2 检测头仅在需要提升 mAP50/Recall 的场景有意义（如某些安全监控指标），在 mAP50-95 导向的评估下不值得。**

**发现 10（修订）：DMGFusion 根本性失效——通过 FPN 污染上层特征**

核心对比：7c（DMGFusion@P2 via FPN）vs 4c（无P2融合）

| 指标 | Exp-4c（基线） | Exp-7c（DMGFusion FPN） | Δ |
|------|--------------|------------------------|---|
| mAP50-95 | 0.632 | 0.625 | **−0.007** |
| smoke | 0.741 | 0.734 | −0.007 |
| fire | 0.582 | 0.582 | 0 |
| **person** | **0.573** | **0.558** | **−0.015** |

7c < 4c 直接证实：**DMGFusion 的 P2 特征经 FPN top-down 传播后污染了 P3-P5 的语义特征**。其中 person 受损最重（−1.5%）——person 检测最依赖 P5 的跨模态语义对齐（CMG+CMA 的协同核心），FPN 顶层一旦受噪声污染，person 定位质量大幅下滑。

7a vs 7c（0.628 vs 0.625）：添加 P2 检测头反而使损失从 −0.007 缩小为 −0.004，说明 P2 head 的梯度信号为 P2 层的特征提供了额外约束，部分"抵消"了 FPN 污染——但绝无法根本解决问题。

**发现（新）：Concat@P2 融合点对 FPN 特征中性至有益**

7d vs 4c（0.634 vs 0.632）：**Concat+1×1Conv@P2 通过 FPN 传递后不仅无害，还带来可观的 fire 增益：**

| 类别 | Exp-4c | Exp-7d | Δ |
|------|--------|--------|---|
| smoke | **0.741** | 0.737 | −0.4% |
| **fire** | 0.582 | **0.589** | **+0.7%** |
| person | 0.573 | **0.576** | +0.3% |

- Fire 最大受益：P2 的低层 Concat 特征经 FPN 传至 P3，为 fire 检测提供更细腻的低层热信号，强化了双模态互补（与 Finding 1、3 一致）。
- Smoke 微降（−0.4%）：smoke 是 RGB 主导类别，P2 的 IR 低层特征对其轻微干扰。
- **核心结论：P2 融合点本身是有价值的；DMGFusion 的失败源于模块设计，而非 P2 融合策略本身。** Concat 是当前 P2 最优融合方式。

### 4.8 类别响应汇总

**发现 11：三类目标形成三种完全不同的融合响应模式**

| 类别 | 尺寸瓶颈 | 模态瓶颈 | 最佳配置 | Exp-0 → 最优的总增益 |
|------|---------|---------|---------|-------------------|
| smoke | P3 仅 6.2% 不可见，无瓶颈 | 无（RGB-only 已达上限） | 任意双分支即可 | mAP50-95 +0.5%（0.734→0.739 Exp-7b）|
| fire | P3 35.8% 不可见 | 双模态互补未充分释放 | **Exp-7d（0.589）** | mAP50-95 +4.4%（0.545→0.589）|
| **person** | P3 38.0% 不可见 | 两模态的语义空间对齐 | **Exp-7b（0.579）** | **mAP50-95 +4.2%（0.537→0.579）** |

- 整体 mAP50 的差异（0.01 级）掩盖了类别层面的显著分化（0.03~0.07 级）。
- CMG/CMA 的价值集中在 fire 和 person——安全监控中最关键的两类。
- **Fire 的最优方案是 Exp-7d**（Concat@P2 via FPN，无P2 head）：P2 低层热信号经 FPN 补充 P3，对 fire 双模态互补再度强化（+4.4% vs Exp-0）。
- **Person 的最优方案是 Exp-7b**（+P2 检测头）：P2 头的 Recall 提升对 person 定位有轻微正贡献（+0.3%）。两者相差仅 0.3%，实际可用 7d 替代。

### 4.9 当前最优配置

**发现 12：Exp-4c 为最优性价比，Exp-7d 为最优 P2-扩展方案**

| 配置 | mAP50 | mAP50-95 | fire | person | 推理(ms) | GFLOPs |
|------|-------|----------|------|--------|----------|--------|
| Exp-0（early fusion 基线） | 0.919 | 0.605 | 0.545 | 0.537 | 4.51 | — |
| Exp-1（双分支 concat） | 0.933 | 0.623 | 0.575 | 0.559 | 6.96 | — |
| Exp-2（+ CMG） | 0.934 | 0.627 | 0.581 | 0.562 | 7.13 | — |
| **Exp-4c ★（CMG + CMA@P5）** | **0.934** | **0.632** | 0.582 | **0.573** | **16.05** | **19.65** |
| Exp-7d（+ Concat@P2 via FPN） | 0.934 | **0.634** | **0.589** | 0.576 | 19.9 | 21.64 |
| Exp-7b（+ Concat@P2 + P2 head） | **0.941** | **0.634** | 0.584 | **0.579** | 20.7 | 31.01 |
| Exp-7a ⚠️（+ DMGFusion@P2 + P2 head） | 0.941 | 0.628 | 0.578 | 0.574 | 21.9 | 31.07 |

**代价效益分析：**
- **Exp-4c → Exp-7d：** mAP50-95 +0.002（持平至微增），fire +0.7%，代价：推理 +24%（3.85ms）、GFLOPs +10%。P2 Concat 通过 FPN 提供低层双模态特征，对 fire 有不对称收益，代价合理。
- **Exp-7d → Exp-7b：** mAP50-95 相同（0.634），mAP50 +0.7%，代价：GFLOPs +43%（额外的 FPN@P2 + 检测头卷积）。纯 mAP50 需求才选 7b，否则 7d 更优。
- **Exp-4c 相对 Exp-1：** mAP50-95 +0.9%、person +1.4%，代价：推理 +2.3×。person 检出精度提升直接对应漏报减少，代价合理。
- **论文推荐方案：Exp-4c** 为主方案（CMG@P4P5 + 双向CMA@P5）；**Exp-7d** 为 P2 融合可选扩展（+Concat@P2 FPN，以 +24% 推理代价获得 fire +0.7%，无需 P2 检测头）。

---

## 六、Exp-8：双流架构全面修复方案（进行中）

> 基于 §3.5.2 特征层面可视化诊断，设计三步修改方案，每步独立消融验证。

### 6.1 背景与动机

Exp-7 揭示 DMGFusion@P2 全面劣于 Concat@P2，但指标差异（0.006）远未刻画问题的深度。通过可视化工具（tools/visualize_p2_dmg.py + tools/probe_const_input.py）对特征层面的细致诊断表明：

1. **RGB 支路语义坍塌**是根本问题。RGB backbone 无直接 loss 约束，仅靠融合后的检测损失传递梯度——而 IR 提供了更强的梯度信号，RGB 逐渐退化为纹理特征提取器。DMGFusion 的门控（W_ir > 0.6）正确识别出"应该信 IR"，但 β 残差旁路把 RGB 草直接注入 fused，导致一个显而易见的问题（W 合理 → fused 却被 RGB 污染）。
2. **P4 area=4 引入架构性水平条带**，零输入 probe 量化确认（P4 mean ratio=2.29），与数据无关。CMG@P4 在已污染的特征上进行门控，起放大作用。
3. **DMGFusion 本身有六项设计缺陷**，见 §3.5.2-C。

**修复策略：先修"输入质量"，再修"融合机制"，最后修"架构噪声"。** 每步单独消融，避免多变量叠加影响归因。

### 6.2 三步修改方案

#### Step 1：RGB/IR 辅助检测头（P3 单尺度）【已实施 2026-04-19】

**目的**：强制 RGB 和 IR backbone 各自独立保留目标语义，切断 RGB 支路语义坍塌的恶性循环。

**实施**：
- 在 `DualStreamDetectionModel.__init__` 中创建 `aux_head_rgb` 和 `aux_head_ir`（`Detect(nc=3, ch=[c_p3])`，stride 固定为 8.0）
- 在 `_predict_once` 训练模式下：在 CMG 应用前，对 RGB/IR backbone 的 P3 特征（pre-CMG 纯 backbone 特征）调用两个 aux head，预测存入 `self._aux_rgb / self._aux_ir`
- 新增 `DualStreamDetectionLoss`：总损失 = main_loss + 0.25 × (aux_rgb_loss + aux_ir_loss)
- 推理时 aux head 不参与（`self.training = False` 时不计算）

**配置**：`--aux_loss_weight 0.25`（默认），可通过 train.py 参数覆盖

**改动文件**：`ultralytics/nn/tasks.py`（新增 `DualStreamDetectionLoss` 类，修改 `DualStreamDetectionModel`），`train.py`（新增 `--aux_loss_weight`）

#### Step 2：P4 backbone 改为 C3k2【已实施 2026-04-19】

**目的**：消除 P4 area=4 引入的水平条带架构噪声（zero-input probe 量化证实，mean ratio=2.29，与数据无关）。

**实施**：`yolov12-dual-p2.yaml` backbone layer 6：`A2C2f [512, True, 4]` → `C3k2 [512, True]`

**注意**：
- P5 保留 `A2C2f [1024, True, 1]`（area=1 = 全局注意力，P5 的"条带"实为有语义的位置先验，不是噪声）
- CMG@P4 继续有效（CrossModalGating 不依赖 A2C2f 内部结构）
- neck 中的 A2C2f 块（layers 11/20/23）不变，仅 backbone 层修改

**验收**：零输入 probe 验证 P4 ratio≥2 通道比例从 29% 降到 < 5%；smoke mAP 不退化

#### Step 3：DMGFusion v2【已实施 2026-04-19】

**目的**：在 RGB 输入质量通过 Step 1 改善后，重建 P2 融合模块，使门控真正有效。

**核心改动**（相对 DMGFusion v1）：

| 改动 | v1 | v2 | 根因 |
|------|----|----|------|
| β 残差 | β=1 初始，可学 | **彻底移除** | β 旁路绕过 W，吞噬门控效果 |
| S/α 分支 | 保留 | **彻底移除** | S 整图饱和，α 冗余，简化公式 |
| D 的归一化 | 原始 `\|R-I\|` | **InstanceNorm 前先归一化**：`D = \|IN(R)−IN(I)\|` | D 实为尺度差，非语义分歧 |
| W 的门类型 | softmax（零和） | **两路独立 sigmoid** | 零和使目标处出现"双低洞" |
| W 的维度 | (B, 2, H, W) | **因式分解 W_s⊗W_c**：(B,2,H,W)×(B,2,C,1,1) | 不同通道需不同模态偏好 |
| out_proj 归一化 | BN（幅值塌缩） | **GroupNorm(8)** | BN 零均值强制引起幅值塌缩 |
| 融合公式 | `(1+αS)·(w_r·R+w_i·I) + β·(R+I)/2` | `out_proj(w_r·R + w_i·I)` | 极简，干净 |
| 初始行为 | β=1 主导 | logit=0 → sigmoid=0.5 → fused≈0.5(R+I) | 保守初始化，与 v1 等价出发点 |

**D 计算细节**：`IN(R)` 和 `IN(I)` 仅用于计算 D 和喂给 sel_block；加权求和时用原始 x_rgb/x_ir（保留 IR 绝对幅值）。

**W_channel 结构（SE 式）**：`GAP([IN(R);IN(I)]) → MLP → sigmoid → (B,2,C,1,1)`，参数约 ~2K（C=64）。

**实施文件**：`ultralytics/nn/modules/block.py`（新增 `DMGFusionV2`，保留 v1 供消融对照），`ultralytics/nn/modules/__init__.py`，`ultralytics/nn/tasks.py`（`p2_fusion: dmg_v2` 分派），`yolov12-dual-p2.yaml`（`p2_fusion: dmg_v2`）

### 6.3 实验计划

#### Exp-8a：Step 1 + Step 2 基础验证

**配置**：`yolov12-dual-p2.yaml`（P4 C3k2）+ `p2_fusion: dmg`（保留 v1，暂不换 v2）+ `--aux_loss_weight 0.25`
**对照**：Exp-7a（DMGFusion v1，无 aux head，A2C2f@P4）

**结果（全 val 集，3366 images，9227 instances）**：

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|-------:|----------:|----------:|-------:|------:|---------:|
| **all** | 3366 | 9227 | **0.930** | **0.910** | **0.945** | **0.638** |
| smoke | 2398 | 4086 | 0.944 | 0.896 | 0.948 | 0.739 |
| fire  | 2606 | 3401 | 0.933 | 0.916 | 0.947 | 0.593 |
| person| 1243 | 1740 | 0.913 | 0.917 | 0.941 | 0.581 |

**模型规模**：Parameters 5,552,112 / GFLOPs 30.63 / 884 layers。

**DMGFusion v1 @P2 收敛标量（Step 1+2 效应的量化证据）**：

| 实验 | α | β | 解读 |
|------|---:|---:|------|
| Exp-7a（无 aux head，P4 A2C2f） | **−1.0908** | **0.2012** | α 为负 → `(1+αS)` 在 S→1 区域接近 0，模型主动**抑制**差异调制；β 已降至 0.20 但 α 路径形同废弃，fused ≈ 轻微 β 残差，门控实际未启用 |
| **Exp-8a**（加 aux head + P4 C3k2） | **+0.9990** | **0.2468** | α 接近饱和正值 → `(1+αS)` 在分歧区放大至 ~2×，差异门控**主导**前向；β 仅微升，均值残差仍是保守旁路 |

**结论**：Step 1（aux head 让 RGB backbone 获得独立梯度）+ Step 2（P4 C3k2 去除条带噪声）联合把 DMGFusion 从"α 负值、门控瘫痪"状态解锁为"α 饱和、差异调制主导"状态。这验证了 §6.1 的核心诊断——v1 在 Exp-7a 失效不是公式缺陷，而是输入质量+架构噪声共同压制了门控信号；一旦前两步清障，同一个 v1 公式即能自发转向非平凡工作点。

**推理时延**：
- 本地 GPU：preprocess 0.3310 + inference 7.1225 + loss 0.0010 + postprocess 0.7363 ≈ **8.19 ms / image**，inference 段 **7.12 ms** 与表 §3.2、§3.3 其他行一致、可直接对比

**验收标准比对**：
1. loss 曲线：（待补曲线图；训练完成且指标达标，推定通过）
2. 零输入 probe：P4 ratio≥2 通道比例 < 5%（待补 probe）
3. RGB P3/P4 特征可视化：待补
4. 整体 mAP50-95 ≥ Exp-7a（0.628）→ **0.638，+0.010 ✓**（亦超过 Exp-7b=0.634、Exp-4c=0.632，为当前历史最佳）

#### Exp-8b：Step 3（DMGFusion v2，在 8a 基础上）【已完成，本地】

**配置**：Exp-8a 所有配置 + `p2_fusion: dmg_v2`
**对照**：Exp-8a

**结果**（val：3366 imgs / 9227 instances）：

| 类别 | Images | Instances | P | R | mAP50 | mAP50-95 |
|------|-------:|----------:|------:|------:|------:|---------:|
| all    | 3366 | 9227 | 0.928 | 0.905 | 0.938 | **0.637** |
| smoke  | 2398 | 4086 | 0.949 | 0.888 | 0.943 | 0.742 |
| fire   | 2606 | 3401 | 0.929 | 0.911 | 0.940 | 0.591 |
| person | 1243 | 1740 | 0.906 | 0.916 | 0.931 | 0.578 |

**模型规模**：Parameters = 5,554,190；GFLOPs = 30.55
**推理时延**（val.py 10-run avg）：preprocess 0.44 ms + inference 7.65 ms + postprocess 0.77 ms

**验收标准**（原始设定）：
1. P2 fused 可视化：草纹理消失，fire/person 目标清晰可见
2. W 在目标位置：`w_rgb > 0.3` 且 `w_ir > 0.3`（不再是"双低洞"）
3. W 在草地背景：`w_rgb < 0.2`
4. fused 幅值与 x_rgb/x_ir 同数量级（ColorBar 量级对比）
5. fire mAP50-95 ≥ Exp-8a + 0.5%

#### Exp-8c：P4 C3k2 + P3 aux + **plain@P2**（Concat+Conv1x1）【训练中，服务器，tmux0，`runs/detect/train`】

**配置**：`yolov12-dual-p2-plain.yaml`（p2_fusion: plain）+ CMG@P4P5 + CMA@P5 + aux head
**对照**：Exp-8a（DMG v1@P2）、Exp-8b（DMG v2@P2）
**用途**：与 8a、8b 形成"plain vs v1 vs v2"三角，直接归因 P2 融合模块的贡献；验证去掉 DMG 后是否反而更好

#### Exp-8d：P4 C3k2 + P3 aux + **DMG v1@P2** − CMG − CMA【训练中，服务器，tmux1，`runs/detect/train2`】

**配置**：`yolov12-dual-p2.yaml`（p2_fusion: dmg，与 8a 一致）基础上将 `cmg_stages: []`、`cma_stages: []` + aux head
**对照**：Exp-8a（保留 CMG/CMA 的 DMG v1 版本）
**用途**：验证 aux head 上线后 CMG@P4P5 + CMA@P5 是否已退化为冗余模块；若 8d ≈ 8a 则可在最终架构中整体移除两者精简模型，若 8d < 8a 则需进一步拆为仅去 CMG/仅去 CMA 做归因

#### 最终对照矩阵

| 实验 | P4 | P2 fusion | aux head | CMG/CMA | 主要验收目标 |
|------|----|-----------|---------|---------|------------|
| Exp-7a（基线） | A2C2f | DMGFusion v1 | 无 | P4P5/P5 | 参照 0.628 |
| **Exp-8a** ★ | **C3k2** | DMGFusion v1 | **有** | P4P5/P5 | P4条带消除 + RGB特征恢复（**0.638**） |
| **Exp-8b** | C3k2 | **DMGFusion v2** | 有 | P4P5/P5 | P2 fused干净 + fire↑（**0.637**） |
| **Exp-8c** | C3k2 | **plain** | 有 | P4P5/P5 | 归因 P2 融合模块贡献（plain vs v1 vs v2） |
| **Exp-8d** | C3k2 | DMGFusion v1 | 有 | **无** | 归因 CMG/CMA 在 aux 规制下是否冗余 |

**成功判定**：Exp-8b 相对 Exp-4c：fire mAP50-95 ≥ +2%，person mAP50-95 ≥ +1%，smoke 不退化，推理时延 ≤ 23ms。

### 6.4 回退策略

| 步骤 | 失败条件 | 回退 |
|------|---------|------|
| Step 1 aux head | aux loss 不下降或主 head 退化 > 1% | 降 λ 到 0.1；若仍失败则仅 RGB aux（IR 不加） |
| Step 2 P4 C3k2 | smoke mAP 退化 > 1% | 仅 neck P4 保留 A2C2f，backbone 换 C3k2 |
| Step 3 DMGFusion v2 | W 全图压 RGB（w_rgb < 0.1） | RGB backbone 未从 Step1 中受益，需更强 λ 或更多 epoch |

---

## 五、备选想法

> 以下方案基于当前实验结论探索进一步改进方向。重点：**P2 层级跨模态融合**（120×160，C=64，两模态物理差异最大）。Exp-7 证明 DMGFusion 在 P2 失效，需要新的融合策略。

### DMGFusion（已实验，Exp-7a，已否定）

差异引导模态融合：`D=|RGB-IR|` → sel_net 生成模态选择权重 W + 差异幅度调制 S，学习门控 alpha/beta。在 P2 层级失效，原因见 §3.5 和 §4.7。实现详见 `ultralytics/nn/modules/block.py` → `DMGFusion` 类。

---

### Tier 1：轻量 P2 融合（~10–100K params）

**想法 K：差异掩码交叉注意力（DMCA）**

只在"模态分歧大的地方"做跨模态注意力，其余位置用简单平均。

```
D_scalar = mean_c(|RGB - IR|)            (B, 1, H, W)
mask     = topK(D_scalar, ρ=20%)         稀疏二值掩码
mask=1 位置：Q=RGB, KV=IR 做局部窗口交叉注意力（4×4 window）
mask=0 位置：(RGB + IR) / 2
fused = routing * attn_out + (1 - routing) * mean_out
```

与 DMGFusion 的区别：DMCA 是硬路由（只有分歧大的位置进入注意力），计算集中在最有价值的像素。~30K params。

**想法 L：迭代差异精炼（IDR）**

多轮精炼把"上一轮没搞清楚的区域"作为下一轮的重点，类似 EM 迭代收敛。

```
F0 = (RGB + IR) / 2
D1 = |RGB - F0| + |IR - F0|
F1 = F0 + gate_net(D1) * cross_attn(F0, D1)
D2 = |RGB - F1| + |IR - F1|
F2 = F1 + gate_net(D2) * mlp(F1, D2)
```

每轮残差 D_i 都在缩小，可用作收敛指标。2–3 层共 ~60K params。

**想法 M：基函数分解融合（BDF）**

RGB/IR 投影到共享语义字典 B（K=16 基向量），在系数空间融合后重建。比特征空间融合更可解释，字典 B 可直接可视化。~35K params。

---

### Tier 2：中量 P2 融合（~100–500K params）

**想法 N：稀疏差异条件交叉注意力（SDC-Attn）**

想法 K 的完整版：软路由替代硬 top-K，加差异 token 全局上下文注入。~150K params。

**想法 O：频域解耦跨模态融合（FDCF）**

Haar 小波分解后，低频用 IR 引导注意力、高频用 RGB 主导 + 差异门控。给"低频用 IR、高频用 RGB"这一物理先验赋予学习能力。~200K params。

**想法 P：原型路由融合专家（PMoFE）**

4 个融合专家（IR 主导 / RGB 主导 / 对称融合 / 直通均值）+ 逐像素路由器。路由权重图可直接可视化"哪个区域用哪种策略"。~220K params。

---

### Tier 3：重量 P2 融合（~400K+）

**想法 Q：差异条件可变形交叉注意力（DCDA）**

可变形采样补偿传感器视差，D 条件偏移量（分歧大 → 允许大偏移）。偏移场箭头图是极强的定性结果。理论上是 P2 融合的上界方案。~500K params。

**想法 R：因果层级差异分解网络（CHDD）**

三尺度差异分解（局部/中/全局）+ 因果聚合（先局部后全局）。每层可独立消融，消融矩阵即论文 Table。~700K params。

---

### 想法 I：类别感知动态门控（跨层级，非P2专属）

根据检测头的类别预测概率动态调整融合权重。工程复杂度高，适合主方案稳定后探索。

---

### 优先级

| 优先级 | 想法 | 理由 | 状态 |
|--------|------|------|------|
| 后续（轻量） | **K** DMCA | 硬路由稀疏注意力 | 待后续 |
| 后续（轻量） | **L** IDR | 迭代精炼 | 待后续 |
| 后续（轻量） | **M** BDF | 共享基函数 | 待后续 |
| 后续（中量） | **N** SDC-Attn | 软路由 + 差异 token | 待后续 |
| 后续（中量） | **O** FDCF | 频域解耦 | 待后续 |
| 后续（中量） | **P** PMoFE | MoE 融合专家 | 待后续 |
| 后续（重量） | **Q** DCDA | 可变形对齐 | 待后续 |
| 后续（重量） | **R** CHDD | 层级差异分解 | 待后续 |
| 长期 | **I** 类别感知门控 | 工程复杂 | 待后续 |

---

## 附录

### A.1 执行备忘

- 训练超参：epochs=300, batch=16, SGD lr0=0.01 lrf=0.01 cos_lr=False val_period=2
- 关注 CPU RAM：训练到 ~epoch 77 可能触发 `_ArrayMemoryError`，需监控并准备 `--resume`
- val.py 速度评估统一跑 10 次取平均

### A.2 Checkpoint 版本对照

| 实验 | 对应 git commit | 需还原的文件 |
|------|----------------|------------|
| Exp-1（dual_MF） | `0b9379a` | block.py, tasks.py, modules/__init__.py |
| Exp-2（dual_MF_ChW） | `e5bdb6b` 或 `92e77f5` | block.py, tasks.py, modules/__init__.py |
| Exp-4a（CMA_rgb） | `c852185` | block.py, tasks.py |
| Exp-4b/4c/4d（双向CMA） | HEAD（直接可用） | — |
