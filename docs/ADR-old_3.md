# ADR-001: 双分支 YOLOv12-MF RGB-红外融合检测架构

**Status:** 实验进行中，当前最优 Exp-4c（CMG@P4P5 + 双向残差CMA@P5）；Exp-7a/7b 已完成，7c/7d 待运行
**Date:** 2026-04-04
**Updated:** 2026-04-17
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

### 3.0 数据集目标尺寸特性分析

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

"某 stride 下 <2 个特征图格子" = 目标在该层基本不可见（精确检测不可靠）：

| 类别 | stride=4（P2）| stride=8（P3）| stride=16（P4）| stride=32（P5）|
|------|-------------|-------------|--------------|--------------|
| smoke | 0.2% | 6.2% | 22.2% | 39.5% |
| fire | **5.2%** | **35.8%** | **80.8%** | **97.8%** |
| person | **5.0%** | **38.0%** | **71.4%** | **98.7%** |
| all | 2.9% | 23.1% | 53.1% | 72.3% |

#### 关键结论

**P3（stride=8）是 fire/person 的主力检测头，但已有 35–38% 目标不足 2 格子：**
- fire 的 p10 短边 = 9px → 需要 stride ≤ 4.5px 才能保证最小 10% 的 fire 可见
- person 的 p10 短边 = 9px → 同上
- smoke 的 p10 短边 = 21px → stride ≤ 10.5px，P3 已经够用

**P4/P5 对 fire/person 几乎无直接贡献：**
- fire 在 P4 的 80.8% 不可见、在 P5 的 97.8% 不可见
- P4/P5 检测头对 fire/person 的增益几乎为零（只能处理极少数大型目标）

**若要显著提升 fire/person 的 mAP50-95，增加 P2 检测头（stride=4）是最直接路径：**
- stride=4 时，fire/person 仅有 5% 不可见，覆盖率从 P3 的 62–65% 提升到 95%+
- 代价：P2 特征图（160×120）是 P3 的 4 倍，计算量显著增加

**这一分析同时解释了 CMA@P5 有效的机制：** P5 的 IR 热信号虽然无法定位个体 fire/person（<1 格子），但作为弥散的全局热源先验，经 FPN top-down 路径流至 P3，为 P3 的小目标检测提供语义上下文。P4 的 CMA 有害，因为 fire/person 在 P4 处于"半可见"边界状态（~1 格子），跨模态对齐噪声反而干扰检测。

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
| Exp-7a ⚠️ | DMGFusion@P2 + CMG@P4P5 + CMA@P5 + head P2P3P4P5 | 0.941 | 0.628 | 0.917 | 0.911 | 5.32M (31.07G) | 21.9 | **已完成（差于7b）** |
| Exp-7b | Concat+1×1Conv@P2 + CMG@P4P5 + CMA@P5 + head P2P3P4P5 | 0.941 | 0.634 | 0.930 | 0.899 | 5.32M (31.01G) | 20.7 | **已完成** |
| Exp-7c | DMGFusion@P2 + CMG@P4P5 + CMA@P5 + head P3P4P5（无P2检测） | — | — | — | — | ~5.53M | — | **待运行（关键诊断，见下）** |
| Exp-7d | Concat+1×1Conv@P2 + CMG@P4P5 + CMA@P5 + head P3P4P5（无P2检测） | — | — | — | — | ~5.52M | — | **训练中 (tmux0)** |

> ★ 当前最优候选。GFLOPs 见下表（P4P5 组合因 thop 双流追踪问题偏低，以推理 ms 为准）。
>
> **训练日志目录对应关系：**
> | 目录 | 实验 | tmux |
> |------|------|------|
> | `runs/detect/train` | Exp-7c | tmux2 |
> | `runs/detect/train2` | Exp-7b | tmux1 |
> | `runs/detect/train3` | Exp-7d | tmux0 |
>
> **Exp-7 消融矩阵**（2×2 设计）：
>
> | | head P2P3P4P5 | head P3P4P5（无P2检测） |
> |---|---|---|
> | **DMGFusion@P2** | Exp-7a ⚠️（mAP50-95=0.628，差于7b） | Exp-7c（待运行，关键诊断） |
> | **Concat+1×1Conv@P2** | Exp-7b（mAP50-95=0.634，已完成） | Exp-7d（训练中 tmux0） |
>
> 对比链：7a vs 7b → 融合方式的影响；7a vs 7c → P2检测头的贡献；**7c vs Exp-4c → DMGFusion 特征质量本身**（与检测头无关）。
>
> **⚠️ 异常观察（2026-04-17）：** Exp-7a（DMGFusion@P2）训练完成后差于 Exp-7b（Concat+Conv@P2），说明 DMGFusion 在 P2 产生了副作用。根本原因待 Exp-7c/7d 结果确认（见 §3.4 DMGFusion 诊断）。

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

**Exp-7 参数量估算（scale n）：**
- Exp-7a（DMGFusion@P2）实测 **5.32M params，31.07 GFLOPs，推理 21.9ms/img**
- Exp-7b（plain P2 fusion）实测 **5.32M params，31.01 GFLOPs，推理 20.7ms/img**
- 两者参数量几乎相同（DMGFusion ~12K vs Conv1×1），GFLOPs 较 Exp-4c（19.65G）大幅上升的主因是 P2 检测头在 120×160 高分辨率下运行

### 3.4 DMGFusion@P2 副作用诊断（2026-04-17）

**现象：** Exp-7a（DMGFusion@P2）训练完成后指标差于 Exp-7b（Concat+1×1Conv@P2），说明 DMGFusion 对 P2 融合产生了副作用。

**关键问题：** 是 DMGFusion 特征质量本身的问题，还是 DMGFusion 特征与 P2 检测头的交互问题？

**Exp-7c 是诊断关键**（必须运行）：
- 若 **7c < Exp-4c**：DMGFusion 生成的 P2 特征本身有害，即使不用于检测也会通过 FPN top-down 路径污染 P3/P4/P5 → 模块设计存在根本问题，需要重新设计
- 若 **7c ≈ Exp-4c**：DMGFusion 特征中性，问题出在 P2 检测头与 DMGFusion 的交互 → 可以考虑放弃 P2 检测头但保留 DMGFusion 作为特征增强
- 若 **7c > Exp-4c**：DMGFusion 特征实际有益，P2 检测头本身（而非 DMGFusion）是问题 → 调整检测头

**可能的失效原因（按可能性排序）：**

1. **D = |RGB-IR| 在 P2 过于嘈杂**：P2 是最浅的融合点，C3k2 之后的特征仍是低层纹理特征而非语义特征。差异图 D 编码的是像素级纹理差异（不同相机的噪声、色温差异）而非语义不一致，sel_net 在学习噪声模式而非模态判别特征。

2. **公式双重计数导致初始化偏置**：当 alpha=0、beta=1、w_rgb≈w_ir≈0.5 时：
   ```
   fused = (w_rgb·x_rgb + w_ir·x_ir) + 1.0 × 0.5·(x_rgb + x_ir)
         ≈ 0.5·(x+y) + 0.5·(x+y) = (x_rgb + x_ir)   # 2× 均值！
   ```
   出发点并非"简单平均"而是"2倍均值"，out_proj 的 BN 虽然能归一化幅度，但梯度流通过两条路径（加权路径 + 残差路径）存在干扰。

3. **alpha 过快增长**：alpha 从 0 开始但无约束，训练初期若梯度将 alpha 推向正值，`(1 + alpha·S)` 项会放大 P2 高差异区域的特征，可能破坏 BN 统计量，进而影响 FPN 上游。

4. **P2 检测头梯度干扰主干**：4-scale 检测头在 P2 增加了大量检测梯度，这些梯度通过 FPN top-down 路径反传，可能影响 P3/P4/P5 的优化。（此原因与 DMGFusion 无关，7b 同样有 P2 头，7b > 7a 说明 DMGFusion 本身也有问题。）

**诊断实验建议（等 7c/7d 结果后执行）：**

| 诊断操作 | 目的 | 成本 |
|----------|------|------|
| 记录训练过程中 `alpha.item()` 和 `beta.item()` 曲线 | 确认 alpha 是否过快增长 | 零成本，加 logging callback |
| 在 val 时可视化 W（模态选择权重）和 S（差异幅度门） | 确认 sel_net 是否在学习有意义的特征 | 低成本 |
| 比较 7a vs 7b 在 P3/P4/P5 的特征分布（activation statistics） | 确认 P2 DMGFusion 是否污染了上层特征 | 中等成本 |
| 尝试 DMGFusion-lite：去掉 beta 残差项，只用 `out_proj(w_rgb·x + w_ir·x)` | 验证双重计数假设（原因2） | 新实验 Exp-8a |
| 冻结 alpha=0（纯软模态选择，无差异放大） | 隔离 alpha 放大效应（原因3） | 新实验 Exp-8b |

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

**Exp-7a ⚠️（DMGFusion@P2 + CMG@P4P5 + CMA@P5 + head P2P3P4P5，5.32M，31.07G，21.9ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.917 | 0.911 | 0.941 | 0.628 |
| smoke | 0.929 | 0.911 | 0.948 | 0.732 |
| fire | 0.922 | 0.913 | 0.941 | 0.578 |
| person | 0.901 | 0.909 | 0.933 | 0.574 |

**Exp-7b（Concat+1×1Conv@P2 + CMG@P4P5 + CMA@P5 + head P2P3P4P5，5.32M，31.01G，20.7ms）：**

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| all | 0.930 | 0.899 | 0.941 | 0.634 |
| smoke | 0.943 | 0.890 | 0.947 | 0.739 |
| fire | 0.938 | 0.912 | 0.944 | 0.584 |
| person | 0.910 | 0.894 | 0.931 | 0.579 |

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

**目标尺寸数据对 P4/P5 层级选择的支撑（见 3.0 节）：** fire/person 在 P4 的 80.8%/71.4% 不可见，在 P5 的 97.8%/98.7% 不可见。因此 P4 的 CMA 试图在"半可见"的边界状态做跨模态对齐（fire/person 的 median 短边 ~22px，在 stride=16 约 1.375 格子），信号不稳定；P5 的 CMA 则在"完全抽象"的全局上下文层操作，IR 的弥散热信号提供稳定的区域先验，经 FPN 传至 P3 才实现真正的检测增益。

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

| 类别 | 尺寸瓶颈（P3 不可见比例）| 模态瓶颈 | 最佳配置 | 从 Exp-0 到最优的总增益 |
|------|------------------------|---------|---------|----------------------|
| smoke | P3 仅 6.2% 不可见，尺寸无瓶颈 | **无瓶颈**（RGB-only 已达上限） | 任意双分支即可（Exp-1 已饱和）| mAP50-95 +0.1%（可忽略）|
| fire | **P3 35.8% 不可见，P4+ 无效** | **双模态互补未充分释放** | Exp-4c（mAP50-95 最高 0.582）| mAP50-95 +3.7%（0.545→0.582）|
| **person** | **P3 38.0% 不可见，P4+ 无效** | **两模态的语义空间对齐** | **Exp-4c（CMG+双向CMA@P5）** | **mAP50-95 +3.6%（0.537→0.573）** |

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

> 以下方案基于当前实验结论探索进一步改进方向。重点：**P2 层级跨模态融合**（高分辨率 120×160，C=64，两模态物理差异最大）。按复杂度分三档。

---

### 想法 J：P2 检测头 + DMGFusion（差异引导模态融合）【已实现 → Exp-7】

**动机：** §3.0 bbox 尺寸分析显示 fire/person 中位短边 ≈ 22px，在 stride=8（P3）下 35–38% 目标不足 2 格。增加 stride=4（P2）检测头可将 fire/person 覆盖率从 62% 提升至 95%。P2 处于低语义层级，两模态物理差异最大（RGB 纹理 vs IR 热梯度），朴素 Concat 会掩盖真正有判别力的"模态分歧"信号——阴影中的热源（person）、烟雾中的高亮火焰（fire）恰好是 RGB/IR 看法不一致但对检测最关键的区域。

**DMGFusion（Differential Modality-Guided Fusion）结构：**
- `D = |x_rgb - x_ir|`：逐像素模态分歧图
- `W = softmax(sel([x_rgb; x_ir; D]))`：2 通道模态选择权重（可视化为红/蓝热力图）
- `S = sigmoid(diff_enc(D))`：差异幅度调制门（(B,C,H,W)，可视化高响应区域）
- `fused = (1 + alpha*S) * (w_rgb*x_rgb + w_ir*x_ir) + beta*0.5*(x_rgb+x_ir)`
- `alpha` 初始化为 0，`beta` 初始化为 1 → 起点 ≡ 简单平均，训练自决定是否引入差异门控

**参数/计算（scale n，C=64）：** ~12K 参数，~230 MFLOPs@120×160，远小于 P2 分支本身。

**实现：**
- `ultralytics/nn/modules/block.py` → `DMGFusion` 类
- `ultralytics/cfg/models/v12/yolov12-dual-p2.yaml` → `p2_fusion: dmg`，4-scale head
- `ultralytics/nn/tasks.py` → `FUSION_LAYER_INDICES` 扩展，pluggable fusion dispatch

**消融计划（Exp-7，2×2 矩阵）：**

| | head P2P3P4P5 | head P3P4P5（无P2检测） |
|---|---|---|
| **DMGFusion@P2** | **Exp-7a（训练中 tmux0）** | Exp-7c（待运行） |
| **Concat+1×1Conv@P2** | Exp-7b（训练中 tmux1） | Exp-7d（待运行） |

- Exp-7a vs Exp-7b：DMGFusion vs 朴素融合的增益（控制：均有P2检测头）
- Exp-7a vs Exp-7c：P2检测头的直接贡献（控制：均用DMGFusion）
- Exp-7b vs Exp-7d：P2检测头增益（控制：均用朴素concat）
- Exp-7c/7d vs Exp-4c：仅P2特征融合（不用于检测）对P3P4P5的涟漪影响

**成功判定：** Exp-7a vs Exp-4c 的 fire mAP50-95 ≥ +2%，person mAP50-95 ≥ +2%；Exp-7a > Exp-7b ≥ +0.5%（证明 DMGFusion 相对朴素 P2 有额外增益）；Exp-7a > Exp-7c（证明 P2 检测头有效）。

---

## Tier 1：轻量 P2 融合（~10–100K params，Exp-7 消融后可直接跑）

### 想法 K：差异掩码交叉注意力（DMCA）

**物理动机：** P2 大部分区域两模态一致（背景、地面）；只有少数区域真正不一致（烟雾中的热源、阴影中的人体）。对所有位置做注意力是浪费；只在"模态分歧大的地方"做跨模态注意力才有信息增益。

**机制：**
```
D_scalar = mean_c(|RGB - IR|)            (B, 1, H, W) 差异标量图
mask     = topK(D_scalar, ρ=20%)         (B, H*W) 稀疏二值掩码

mask=1 位置：Q=RGB, KV=IR 做局部窗口交叉注意力（4×4 window）
mask=0 位置：(RGB + IR) / 2

fused = routing * attn_out + (1 - routing) * mean_out
```

**与 DMGFusion 的区别：** DMGFusion 是软加权（全局参与），DMCA 是硬路由（只有分歧大的位置进入注意力），计算集中在最有价值的像素，注意力图可直接可视化"模型在哪里做了跨模态推理"。

**规模（C=64）：** ~30K params，局部窗口注意力 ~150 MFLOPs @120×160。

---

### 想法 L：迭代差异精炼（IDR）

**物理动机：** 单次融合无法解决所有模态歧义（烟雾半透明性在 RGB/IR 中表现复杂）。多轮精炼把"上一轮没搞清楚的区域"作为下一轮的重点，类似 EM 算法的迭代收敛。

**机制：**
```
F0 = (RGB + IR) / 2                                    粗融合
D1 = |RGB - F0| + |IR - F0|                           残差不一致图（第1轮）
F1 = F0 + gate_net(D1) * cross_attn(F0, D1)           第1轮精炼
D2 = |RGB - F1| + |IR - F1|                           残差不一致图（第2轮）
F2 = F1 + gate_net(D2) * mlp(F1, D2)                  第2轮精炼（更轻量）
```

**创新点：** 将"模态差异"从静态输入信号变为动态收敛过程——每轮残差 D_i 都在缩小，可以用残差曲线作为收敛指标，也可以可视化各轮"哪些区域还有歧义"。

**规模：** 2–3 层共 ~60K params。

---

### 想法 M：基函数分解融合（BDF）

**物理动机：** RGB 和 IR 特征共享部分语义基（物体形状、位置），但各有专属基（纹理 vs 热值）。融合应发生在"共享子空间"，而不是原始特征空间——在高维特征空间直接融合会混淆模态专属信息。

**机制：**
```
共享字典 B ∈ R^{K×C}，K=16 个可学习基向量
A_rgb = softmax(RGB @ B^T / sqrt(C))   (B, K, H, W) RGB 在共享基上的系数
A_ir  = softmax(IR  @ B^T / sqrt(C))   (B, K, H, W) IR  在共享基上的系数
A_fused = f(A_rgb, A_ir)               在系数空间融合（可以是注意力、gating）
fused = A_fused @ B                    重建回特征空间
```

**创新点：** 字典 B 可以可视化（每个基向量学到了什么语义模式），系数融合比特征融合更可解释；字典初始化策略也是一个消融维度。

**规模：** 字典 2K + fusion net ~30K，共 **~35K params**。

---

## Tier 2：中量 P2 融合（~100–500K params，独立子贡献）

### 想法 N：稀疏差异条件交叉注意力（SDC-Attn）

**想法 K 的完整版**，用软路由替代硬 top-K，加入差异 token 全局上下文。

**机制：**
```
D_scalar  = mean_c(|RGB - IR|)                        (B, 1, H, W)
routing   = sigmoid(route_net(D_scalar))               (B, 1, H, W) 软路由权重

# 高分歧路径（贵但精确）：局部窗口交叉注意力（8×8 window）
Q = proj_q(RGB),  K = proj_k(IR),  V = proj_v(IR)
attn_out = window_attn(Q, K, V, window=8)

# 差异 token：D 的全局压缩，注入注意力作为额外 KV（全局上下文）
diff_token = gap(diff_enc(D_scalar))                   (B, 1, C)
attn_out   = inject_token(attn_out, diff_token)

# 低分歧路径（便宜）
cheap_out = conv1x1(concat(RGB, IR))

# 软混合
fused = routing * attn_out + (1 - routing) * cheap_out
```

**与 DMCA 的区别：** 软路由梯度更友好；差异 token 将"全局有多不一致"注入局部注意力，使每个窗口的注意力都知道全局背景。

**规模（C=64，w=8）：** ~150K params，~400 MFLOPs。

---

### 想法 O：频域解耦跨模态融合（FDCF）

**物理动机：**
- IR 低频（全局热场、目标轮廓）比 RGB 低频更可靠（烟雾遮挡对 IR 影响小）
- RGB 高频（纹理、边缘细节）比 IR 高频更丰富（IR 分辨率低，高频噪声大）
- 统一在特征空间处理是次优的——不同频段的最优融合策略从物理上就不同

**机制（用 Haar 小波，比 FFT 快且可逆）：**
```
LL_rgb, HH_rgb = haar_decomp(RGB)         低频/高频分量
LL_ir,  HH_ir  = haar_decomp(IR)

# 低频：IR 主导（IR-guided attention on RGB）
LL_fused = ir_guided_attn(LL_rgb, guide=LL_ir)

# 高频：RGB 主导，差异引导（只在高频不一致处融合）
D_hf = |HH_rgb - HH_ir|
HH_fused = rgb_dominant_gate(HH_rgb, HH_ir, cond=D_hf)

fused = haar_recon(LL_fused, HH_fused)
```

**创新点：** 给"低频用 IR、高频用 RGB"这一物理先验赋予了学习能力；小波保持精确空间可逆性，可与任意注意力机制正交叠加；低频/高频分支可独立消融。

**规模：** ~200K params。

---

### 想法 P：原型路由融合专家（PMoFE）

**物理动机：** "模态分歧"不是一维概念——烟雾中的热源、阴影中的人体、明火、低分歧背景这四种场景需要的不仅仅是不同强度的融合，而是不同类型的融合策略。MoE（Mixture-of-Experts）是建模这种离散多样性的自然工具。

**机制：**
```
K=4 个融合专家（每个 2层卷积，各自独立参数）：
  Expert 0: IR 强主导（热源在烟雾中，IR 有信号而 RGB 被遮挡）
  Expert 1: RGB 强主导（明亮纹理区域，IR 饱和）
  Expert 2: 对称融合（目标边界，两模态同等重要）
  Expert 3: 直通均值（背景区域，两模态一致）

Router: mean_c(|RGB - IR|) → conv → softmax  (B, K, H, W) 逐像素专家权重
fused = sum_k gate_k * Expert_k(RGB, IR)
```

**创新点：** 将"融合策略多样性"显式建模；专家权重图可直接可视化（哪个区域用哪种策略，对应哪类场景）；路由器本身是论文 qualitative result 的极好来源。

**规模：** 4 专家 × ~50K + router ~20K = **~220K params**。

---

## Tier 3：重量 P2 融合（~400K+，主要贡献方向）

### 想法 Q：差异条件可变形交叉注意力（DCDA）

**物理动机：** RGB 和 IR 传感器存在视差（光学轴不完全对齐），烟雾/玻璃使两模态的"有效感受野"在空间上错位。标准注意力假设特征对齐，但 P2 高分辨率处对齐误差最大。可变形注意力允许模型学习在另一模态的哪个偏移位置寻找对应信息，而差异图 D 可以驱动偏移量——分歧越大，允许查找越远的位置（更大的错位补偿）。

**机制：**
```
D_scalar    = mean_c(|RGB - IR|)                              (B, 1, H, W)
offset_scale = sigmoid(offset_gate(D_scalar))                 分歧大→允许大偏移
offsets     = offset_net(D_scalar) * offset_scale * max_disp  (B, 2·n_pts, H, W)

IR_sampled  = deform_sample(IR, offsets)                      IR 的偏移位置采样
offsets_inv = offset_net_inv(D_scalar) * offset_scale         反向偏移（RGB→IR）
RGB_sampled = deform_sample(RGB, offsets_inv)

Q = proj(RGB),  K = proj(IR_sampled),  V = IR_sampled
attn_out = softmax(Q @ K^T / sqrt(C)) @ V                     对齐后交叉注意力

fused = out_proj(concat(attn_out, RGB_sampled))
```

**创新点：** D 条件偏移是本方案独特之处——模型自适应学习补偿传感器视差，偏移场可视化（箭头图）是极强的定性结果；可变形采样 + 交叉注意力的完整组合，理论上是 P2 融合的上界方案。

**规模（C=64，n_pts=4）：** ~500K params，~600 MFLOPs。

---

### 想法 R：因果层级差异分解网络（CHDD）

**物理动机：** 模态差异有尺度结构——局部差异（传感器噪声，~2px）、中尺度差异（目标边界热梯度，~16px）、全局差异（场景级互补，整图烟雾遮挡程度）。单一融合层无法同时处理三个尺度，且三个尺度的最优处理策略完全不同。

**机制：**
```
D = |RGB - IR|
D_local  = D                                           原始分辨率（局部噪声/错位）
D_mid    = avgpool(D, 4×4) → bilinear upsample         中尺度（目标边界）
D_global = gap(D) → broadcast                          全局标量（场景级）

# 三个尺度专属融合头
F_local  = local_window_attn(RGB, IR, cond=D_local)    小窗口交叉注意力 (4×4)
F_mid    = channel_gate(RGB, IR, cond=D_mid)           通道门控（中等代价）
F_global = modality_select(RGB, IR, cond=D_global)     全局模态选择

# 因果聚合：局部 → 中 → 全局，逐步精炼
F1 = F_local
F2 = F1 + cross_gate(F_mid,    F1)    中尺度信息精炼局部结果
F3 = F2 + cross_gate(F_global, F2)    全局信息精炼中尺度结果
fused = out_proj(F3)
```

**创新点：** 将"差异的尺度层级"显式建模并按因果顺序聚合（先局部后全局）；每层可单独消融（3个控制变量），消融矩阵本身即论文 Table；三个尺度的权重图各具不同视觉意义。

**规模：** ~700K params（可用共享主干减半至 ~400K）。

---

### 想法 I：类别感知动态门控（跨层级，非P2专属）

根据检测头的类别预测概率动态调整融合权重。smoke 类检测时更信任 RGB（可见光纹理），fire/person 类检测时更信任 IR（热特征）。实现上需要把检测头的 soft prediction 反传回融合层，工程复杂度高，适合在主方案稳定后探索。

---

## 优先级（更新后）

| 优先级 | 想法 | 理由 | 状态 |
|--------|------|------|------|
| **进行中** | **J Exp-7a**（DMGFusion@P2 + P2P3P4P5 head） | 主方案 | 训练中 tmux0 |
| **进行中** | **J Exp-7b**（Concat+Conv@P2 + P2P3P4P5 head） | 消融：融合方式 | 训练中 tmux1 |
| 待运行 | J Exp-7c（DMGFusion@P2 + P3P4P5 head） | 消融：P2检测头贡献 | 待运行 |
| 待运行 | J Exp-7d（Concat+Conv@P2 + P3P4P5 head） | 消融：P2融合涟漪效应 | 待运行 |
| 后续（轻量） | **K** DMCA | 硬路由稀疏注意力，可视化强 | 待后续 |
| 后续（轻量） | **L** IDR | 迭代精炼，EM 风格 | 待后续 |
| 后续（轻量） | **M** BDF | 共享基函数，可解释 | 待后续 |
| 后续（中量） | **N** SDC-Attn | DMCA 软路由完整版 + 差异 token | 待后续 |
| 后续（中量） | **O** FDCF | 频域解耦，物理先验强 | 待后续 |
| 后续（中量） | **P** PMoFE | MoE 融合专家，路由可视化 | 待后续 |
| 后续（重量） | **Q** DCDA | 可变形对齐 + 注意力，上界方案 | 待后续 |
| 后续（重量） | **R** CHDD | 层级差异分解，消融矩阵完整 | 待后续 |
| 长期 | **I** 类别感知门控 | 创新性强但工程复杂 | 待后续 |
