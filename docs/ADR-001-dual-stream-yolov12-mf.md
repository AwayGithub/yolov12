# ADR-001: 双分支 YOLOv12-MF RGB-红外融合检测架构

**Status:** Exp-7 全部完成。Exp-8a 已完成（mAP50-95=0.638，历史最佳）。Exp-8b 已完成（0.637）。Exp-8c / Exp-8d 训练中（服务器），Exp-8e 训练中（本地）。
**Date:** 2026-04-04
**Updated:** 2026-04-20
**Deciders:** 研究者本人

---

## 一、项目概要

**任务：** YOLOv12n + RGBT-3M，RGB+IR 双模态目标检测，nc=3（smoke/fire/person），分辨率 480×640。

**动机：** 早期融合（6ch 直接拼接）被迫在第一层混合两种物理特性截然不同的模态，低层特征提取缺乏模态针对性。改为中期融合双分支，让每个模态独立提取低层特征，再在高语义层级交互。

**决策：** 采用 **自定义 `DualStreamDetectionModel`**（Python 层实现双分支，不修改 `parse_model()`，YAML 继续定义单分支结构）。

```
Input: (B, 6, H, W)
         │
    ┌────┴────┐  split: 0:3=RGB, 3:6=IR
    ▼         ▼
 RGB(B,3,H,W)  IR(B,3,H,W)
    │              │
┌───┴───┐    ┌────┴────┐
│ Conv  │    │  Conv'  │   Stage 0-3: P1/2 → P3/8
│ C3k2  │    │  C3k2'  │   Stage 4  ← P3 特征
│A2C2f/ │    │A2C2f'/  │   Stage 6  ← P4（可选 CMG / CMA）
│ C3k2  │    │ C3k2'   │
│ A2C2f │    │ A2C2f'  │   Stage 8  ← P5（可选 CMG / CMA）
└───┬───┘    └────┬────┘
    │              │
    ▼              ▼
 融合（P2/P3/P4/P5）：Concat+1×1Conv 或 DMGFusion(v1/v2)@P2
    │
    ▼
  Shared Neck (FPN) → Detect Head
        └─ 可选 RGB/IR 辅助头 @P3（仅训练）
```

**YAML 控制标志**（`ultralytics/cfg/models/v12/yolov12-dual*.yaml`）：

| 标志 | 作用 | 合法值 |
|------|------|--------|
| `cmg_stages` | SE-CMG 跨模态门控所在层级 | 子集 of `{p3,p4,p5}` |
| `cma_stages` | CrossModalA2C2f 双向跨模态注意力层级 | 子集 of `{p3,p4,p5}` |
| `p2_fusion` | P2 融合方式 | `plain` / `dmg` / `dmg_v2` |

---

## 二、关键模块

### 2.1 CrossModalGating（SE-CMG）

`ultralytics/nn/modules/block.py`

SE 式通道门控：GAP → Linear → Sigmoid → 残差。`guide` 模态的全局通道统计量调制 `target` 模态特征。每实例参数 = c×c + c。

```python
class CrossModalGating(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, target, guide):
        gate = torch.sigmoid(self.fc(self.pool(guide).flatten(1)))  # (B, C)
        gate = gate.unsqueeze(-1).unsqueeze(-1)
        return target * gate + target
```

> Exp-2b 验证 CM-CBAM（空间注意力）因 `torch.mean/max(dim=1)` 跨通道规约对 GPU cache 不友好，推理 +166%、指标劣于 SE-CMG，已否定。

### 2.2 CrossModalA2C2f（CMA）

**动机：** CMG 是后置模块，信息太晚、信号太粗（GAP 丢失空间位置）。CMA 将跨模态交互内嵌进 A2C2f：前半段自注意力（建立模态内理解），后半段用另一模态空间特征作 K/V。

**nano 配置（n=2）：** 1 组 2×ABlock 自注意力 + 1 组 CrossModalABlock 跨模态注意力。`cross_scale=0.01` 可学习残差缩放，保证初始化无害。

**与 CMG 对比：**

| 维度 | CMG | CMA |
|------|-----|-----|
| 信息粒度 | 全局 1 值/通道 | 每空间位置 Q-K 匹配 |
| 空间感知 | 无 | 有 |
| P5 参数/实例 | ~33K | ~377K |

### 2.3 DMGFusion v1（P2 融合模块）

差异引导融合，`fused = (1 + α·S)·(w_rgb·R + w_ir·I) + β·(R+I)/2`，其中 `D=|R-I|`，`S=σ(diff_enc(D))`，`[w_rgb,w_ir]=softmax(sel([R;I;D]))`。初始 α=0、β=1。

**设计缺陷**（§六 可视化诊断证实）：β 残差吞噬门控；D 编码尺度差而非语义分歧；softmax 零和约束引发目标处"双低洞"；BN 导致输出幅值塌缩；S 整图饱和；W 无通道维。

### 2.4 DMGFusion v2

相对 v1 的修复（逐项对应 v1 缺陷）：

| 改动 | v1 | v2 |
|------|----|----|
| β 残差 | 保留 | **彻底移除** |
| S/α 分支 | 保留 | **彻底移除** |
| D 归一化 | 原始 `\|R−I\|` | `\|IN(R)−IN(I)\|`（InstanceNorm） |
| W 门类型 | softmax（零和） | 两路独立 sigmoid |
| W 维度 | (B,2,H,W) | 因式分解 (B,2,H,W)⊗(B,2,C,1,1) |
| out_proj 归一化 | BN | GroupNorm(8) |
| 融合公式 | 见 v1 | `out_proj(w_r·R + w_i·I)` |

加权求和时用原始 x_rgb/x_ir（保留 IR 绝对幅值），IN 仅用于计算 D 与喂 sel。

### 2.5 RGB/IR 辅助检测头（P3 单尺度）

`DualStreamDetectionModel.__init__` 创建 `aux_head_rgb` / `aux_head_ir`（`Detect(nc, ch=[c_p3])`, stride=8）。训练前向在 CMG 之前对两分支 pre-CMG 的 P3 特征分别预测，`DualStreamDetectionLoss` 把 `aux_weight·(L_rgb+L_ir)` 加到主损失（默认 `--aux_loss_weight 0.25`）。推理时 `self.training=False` → aux head 不参与。

**目的：** 切断 RGB backbone 语义坍塌（RGB 无独立梯度 → 退化为纹理提取器 → fused 被 RGB 草污染）。

---

## 三、数据集特性分析

> 脚本 `analyze_bbox_distribution.py`，11,220 张标签，30,777 框。

### 3.1 类别尺寸画像

| 类别 | 目标数 | 宽度中位 | 高度中位 | 短边<32px | 32–96px | >96px |
|------|-------:|---------:|---------:|----------:|--------:|------:|
| smoke  | 13,574 | 111 | 116 | 22.2% | 28.1% | **49.7%** |
| fire   | 11,315 |  22 |  27 | **80.8%** | 19.0% | 0.3% |
| person |  5,888 |  20 |  27 | **71.4%** | 28.4% | 0.1% |
| all    | 30,777 |  32 |  39 | 53.1% | 24.8% | 22.0% |

smoke mean=186 >> median=111，分布高度右偏（少量近距离大烟团）。

### 3.2 各尺度对小目标的覆盖（"某 stride 下 <2 格子"比例）

| 类别 | P2 (s=4) | P3 (s=8) | P4 (s=16) | P5 (s=32) |
|------|---------:|---------:|----------:|----------:|
| smoke  |  0.2% |  6.2% | 22.2% | 39.5% |
| fire   |  5.2% | **35.8%** | **80.8%** | **97.8%** |
| person |  5.0% | **38.0%** | **71.4%** | **98.7%** |

### 3.3 关键结论

- **P3 是 fire/person 主力**，但已有 35–38% 目标不足 2 格子。
- **P4/P5 对 fire/person 几乎无直接贡献**，其价值在于提供 FPN 语义上下文。
- **P2（stride=4）可将 fire/person 覆盖率 62%→95%**，代价 ~+58% FLOPs。
- **CMA@P5 的作用机制**：P5 弥散热信号作为全局热源先验，经 FPN top-down 流至 P3 提供语义上下文。**P4 CMA 有害**——fire/person 在 P4 处于"半可见"边界，跨模态对齐噪声干扰。

---

## 四、实验结果

**训练超参：** YOLOv12n + RGBT-3M，480×640，epochs=300，batch=16，SGD lr0=0.01 lrf=0.01 cos_lr=False val_period=2

**Fitness 公式（worktree 自定义）：** `w = [0.0, 0.2, 0.3, 0.5]`（P/R/mAP50/mAP50-95），显式纳入 Recall，适合安全监控任务。

### 4.1 全量实验汇总

| 实验 | 配置 | mAP50 | mAP50-95 | P | R | 参数量 | GFLOPs | 推理(ms) | 状态 |
|------|------|------:|---------:|---:|---:|-------:|------:|---------:|------|
| Exp-0 | 单分支 6ch early fusion | 0.919 | 0.605 | 0.914 | 0.877 | 2.51M |   —  |  4.51 | 完成 |
| Exp-1 | 对称双分支 concat-only | 0.933 | 0.623 | 0.924 | 0.893 |   —  |   —  |  6.96 | 完成 |
| Exp-2 | + CMG@P4P5 | 0.934 | 0.627 | 0.925 | 0.894 | 4.21M |   —  |  7.13 | 完成 |
| Exp-2b | + CM-CBAM@P4P5 | 0.927 | 0.627 | 0.925 | 0.882 | 4.13M |   —  | 18.91 | 否定 |
| Exp-4a | CMG + 单向CMA@P4P5 (RGB←IR) | 0.933 | 0.625 | 0.918 | 0.900 | 4.94M | 18.02 | 15.27 | 完成 |
| Exp-4b | CMG + 双向CMA@P4P5 | 0.934 | 0.627 | 0.923 | 0.890 | 5.49M | 18.57 | 17.55 | 完成 |
| Exp-4b-noCMG | 双向CMA@P4P5（无CMG） | 0.929 | 0.628 | 0.922 | 0.886 | 5.34M | 18.47 | 16.09 | 完成 |
| **Exp-4c** ★ | **CMG + 双向CMA@P5** | **0.934** | **0.632** | 0.916 | 0.892 | **5.23M** | **19.65** | **16.05** | **完成** |
| Exp-4d | CMG + 双向CMA@P4 | 0.931 | 0.623 | 0.923 | 0.887 | 4.47M | 19.60 | 16.68 | 完成 |
| Exp-5 | RGB-only 3ch | 0.907 | 0.583 | 0.920 | 0.854 | 2.51M |   —  |  4.08 | 完成 |
| Exp-6 | IR-only 3ch | 0.887 | 0.564 | 0.888 | 0.831 | 2.51M |   —  |  4.33 | 完成 |
| Exp-7a ⚠️ | DMGFusion@P2 + 4c + head P2P3P4P5 | 0.941 | 0.628 | 0.917 | 0.911 | 5.32M | 31.07 | 21.9 | 完成 |
| Exp-7b | Concat@P2 + 4c + head P2P3P4P5 | 0.941 | 0.634 | 0.930 | 0.899 | 5.32M | 31.01 | 20.7 | 完成 |
| Exp-7c | DMGFusion@P2 + 4c + head P3P4P5 | 0.933 | 0.625 | 0.933 | 0.888 | 5.28M | 21.69 | 20.5 | 完成 |
| **Exp-7d** ★ | **Concat@P2 + 4c + head P3P4P5** | 0.934 | 0.634 | 0.925 | 0.895 | 5.28M | 21.64 | 19.9 | 完成 |
| **Exp-8a** ★ | **DMGv1@P2 + 4c + P4 C3k2 + P3 aux + head P2P3P4P5** | **0.945** | **0.638** | **0.930** | **0.910** | **5.55M** | **30.63** | **7.12** | **完成** |
| Exp-8b | Exp-8a + **DMGFusion v2** | 0.938 | 0.637 | 0.928 | 0.905 | 5.55M | 30.55 | 7.65 | 完成 |
| Exp-8c | Exp-8a，P2 改 **plain** | — | — | — | — | — | — | — | 训练中（服务器 tmux0） |
| Exp-8d | Exp-8a，**去 CMG/CMA** | — | — | — | — | — | — | — | 训练中（服务器 tmux1） |
| Exp-8e | **去 CMG/CMA** + P2 **plain** + P3 aux + P4 C3k2 | — | — | — | — | — | — | — | 训练中（本地） |

注：Exp-8 系的推理 ms 为本地 GPU 的 inference 段（与其他行硬件一致，可直接对比）。Exp-7/Exp-4 系的 ms 来自早期硬件。

### 4.2 分类别关键结果（选星号方案）

| 实验 | all mAP50-95 | smoke | fire | person |
|------|------:|------:|------:|------:|
| Exp-0 (early fusion) | 0.605 | 0.734 | 0.545 | 0.537 |
| Exp-1 (双分支 baseline) | 0.623 | 0.735 | 0.575 | 0.559 |
| Exp-5 (RGB-only) | 0.583 | 0.735 | 0.497 | 0.515 |
| Exp-6 (IR-only) | 0.564 | 0.664 | 0.524 | 0.504 |
| Exp-4c ★ | 0.632 | 0.741 | 0.582 | **0.573** |
| Exp-7b | 0.634 | 0.739 | 0.584 | **0.579** |
| Exp-7d | 0.634 | 0.737 | **0.589** | 0.576 |
| **Exp-8a** ★ | **0.638** | 0.739 | **0.593** | 0.581 |
| Exp-8b | 0.637 | **0.742** | 0.591 | 0.578 |

### 4.3 参数量自洽验证

- Exp-4c − Exp-4d = 753,664 ≈ P5 比 P4 多的 `256²×2 − 128²×2` ✓
- Exp-4b − Exp-4b-noCMG = 164,608 = CMG 4 实例精确参数量（`4×(c²+c)`）✓
- **P2 head FLOPs 跳跃**：Exp-7a/7b 从 ~19.65G 跳至 ~31G（+58%），参数仅 +0.1M。根因：FLOPs ∝ 参数×空间分辨率，P2（120×160）是 P5 的 64 倍分辨率。

---

## 五、关键发现

### 5.1 双分支中期融合是核心贡献（Exp-0→Exp-1）

| 类别 | Exp-0 | Exp-1 | Δ |
|------|------:|------:|------:|
| smoke  | 0.734 | 0.735 | +0.1% |
| fire   | 0.545 | **0.575** | **+3.0%** |
| person | 0.537 | 0.559 | +2.2% |

**早期融合给 fire 仅 +4.8%（vs RGB-only）；再加双分支又 +3.0%，累计 +15.7%。双分支释放的互补性是早期融合的 1.6 倍。** 后续 CMG/CMA 均是此基础上的精细化。

### 5.2 单模态基线：三类目标模态依赖模式不同

| 类别 | RGB-only | IR-only | 模态关系 |
|------|---------:|--------:|----------|
| smoke  | **0.735** | 0.664 | RGB 强主导 |
| fire   | 0.497 | 0.524 | **两模态等效，真正互补** |
| person | 0.515 | 0.504 | 两模态接近，微弱互补 |

fire 是对融合架构最敏感的类别——两模态单独使用接近，但信息完全不同（RGB 颜色/形状 vs IR 热辐射）。

### 5.3 CMG 对 person 不可替代（Exp-2）

CMG 消融（Exp-4b-noCMG vs Exp-4b）：

| 类别 | ΔmAP50 | ΔRecall |
|------|-------:|--------:|
| smoke  | 0.0% | +0.2% |
| fire   | +0.4% | −0.1% |
| **person** | **+1.1%** | **+1.1%** |

person 同时依赖 RGB 纹理（轮廓）与 IR 热辐射（生命体确认），CMG 动态平衡"哪些通道的 IR 信号更可信"。smoke/fire 各有主导模态，CMG 反而引入投票噪声。

### 5.4 CMA 仅在 P5 有正效果（Exp-4 系列）

| 配置 | mAP50-95 | person | 推理(ms) |
|------|------:|------:|------:|
| Exp-4d (CMA@P4) | 0.623 | 0.559 | 16.68 |
| **Exp-4c (CMA@P5)** ★ | **0.632** | **0.573** | 16.05 |
| Exp-4b (CMA@P4P5) | 0.627 | 0.566 | 17.55 |

P4 CMA 对 person 反而有害（0.559 < Exp-2 基线 0.562）——P4 语义未成熟。P4P5 组合不如 P5-only。

### 5.5 CMG+CMA@P5 对 person 存在协同

| 配置 | person | Δ vs Exp-1 |
|------|------:|------:|
| Exp-1 | 0.559 | — |
| Exp-2 (+CMG) | 0.562 | +0.003 |
| Exp-4b-noCMG (+CMA@P4P5) | 0.564 | +0.005 |
| **Exp-4c (CMG+CMA@P5)** | **0.573** | **+0.014** |

独立叠加预期 +0.008，实际 +0.014（1.75×）。**机制：** CMG 在通道维选出可信通道 → CMA 的 K/V 来自更纯净特征 → 查询质量提升；CMA 提供的空间上下文反哺 CMG 的门控稳健性。

### 5.6 单向 CMA 只提高 Recall、不提升定位

| 指标 | Exp-2 | Exp-4a（单向） | Exp-4c（双向） |
|------|------:|------:|------:|
| all Recall | 0.894 | **0.900** | 0.892 |
| all mAP50-95 | 0.627 | 0.625 | **0.632** |
| person mAP50-95 | 0.562 | 0.562 | **0.573** |

定位精度需要**双向交互**。单向以接近双向的代价只得 Recall，不值得保留。

### 5.7 P2 融合完整归因（Exp-7 2×2）

| | +P2 head | 无P2 head | P2 head 纯效果 |
|---|---|---|---|
| **DMGFusion@P2** | 7a: 0.628 | 7c: **0.625** | +0.003 |
| **Concat@P2** | 7b: 0.634 | 7d: 0.634 | 0.000 |
| DMGFusion vs Concat | −0.006 | **−0.009** | |

**P2 检测头对 mAP50-95 零贡献（7b vs 7d=0.634 持平），仅提升 mAP50 +0.7% 与 Recall +0.4%。** 代价：GFLOPs +43%、推理 +4%。在 mAP50-95 导向下不值得。

**DMGFusion v1 经 FPN 污染上层（7c=0.625 < 4c=0.632，person −1.5% 受损最重）**。person 最依赖 P5 的 CMG+CMA 协同，一旦顶层被噪声污染下滑最大。

**Concat@P2 对 FPN 中性至有益（7d=0.634 ≥ 4c=0.632，fire +0.7%）**：P2 低层 Concat 特征经 FPN 传至 P3，强化 fire 双模态互补。**结论：P2 融合点本身有价值，DMGFusion 失败源于模块设计。**

### 5.8 类别响应汇总

| 类别 | 尺寸瓶颈 | 模态瓶颈 | 最优配置 | Exp-0→最优增益 |
|------|---------|---------|---------|----------:|
| smoke  | 无 | 无（RGB 已达上限） | 任意双分支 | +0.8%（Exp-8b 0.742） |
| fire   | P3 35.8% 不可见 | 互补未充分释放 | **Exp-8a** 0.593 | **+4.8%** |
| person | P3 38.0% 不可见 | 语义空间对齐 | **Exp-8a** 0.581 | **+4.4%** |

CMG/CMA/aux head 的增益集中在 fire 和 person——安全监控的两类关键类别。

---

## 六、Exp-7 DMGFusion@P2 深度诊断

> 详细特征可视化工具：`tools/visualize_p2_dmg.py` + `tools/probe_const_input.py`。
> 此节为 Exp-8 修复方案的诊断依据，保留核心结论。

### 6.1 实验层面（已确证，见 §5.7）

- 7c(0.625) < 4c(0.632)：DMGFusion 的 P2 特征**经 FPN top-down 污染上层**。
- 7a vs 7c (0.628 vs 0.625)：P2 head 梯度仅轻微缓解（Δ=0.003），非主因。
- 7d(0.634) ≥ 4c(0.632)：Concat@P2 无害有益 → 失败源于 DMGFusion 设计。

### 6.2 特征可视化诊断（video10_frame_01154 样本）

**A. RGB 支路语义坍塌（根本问题）**

- RGB P2/P3 以草/植被高频纹理为主，目标激活低于背景；P4 退化为噪声；P5 仅零散全局激活点。
- IR 在 P2–P5 每尺度清晰定位 fire/person（目标-背景对比 10:1+）。
- 融合后 IR 目标能量被同数量 RGB 通道"平均"稀释；plain fusion 的 1×1 Conv 无"信 IR"先验。
- **根因：** detection loss 梯度经 IR 强信号路回传，RGB backbone 无独立约束 → 退化为纹理提取器。
- IR 过曝条带（x_ir≈3.0）未被抑制，在 DMGFusion 的 D 中最亮，被 sel_net 误识为"高价值分歧区"。

**B. P4 area=4 结构性噪声（与数据无关，zero-input probe 证实）**

- P4 mean row_std/col_std ratio = **2.29**，29% 通道 ratio ≥ 2.0。
- 原因：AAttn 把 H 切成 4 条 strip，各 strip token 数不同 → softmax 温度差异 → strip 间均值偏移。
- CMG@P4 作用于已污染特征 → 放大而非修复。
- P5 area=1 即全局注意力，ratio=1.98；bot[b05]=9.26 为学到的位置先验（烟雾偏上、人偏下），属**有语义的数据集先验，不需修复**。

**C. DMGFusion v1 六项设计缺陷**

1. **β 残差旁路吞噬门控（最致命）**：初始 α=0,β=1 → fused ≈ out_proj(0.5·(R+I))，W 合理但 fused 被 RGB 草主导。
2. **D=|R−I| 编码尺度差而非语义分歧**：x_rgb~[0.25,1.75]、x_ir~[0,3.0]，D 最亮在 IR 过曝条带而非目标。**修复：** InstanceNorm 前归一化。
3. **softmax 双路权重零和**：强制 w_rgb+w_ir≡1。目标位置两路同时出现"双低洞"（~0.3–0.4）。**修复：** 独立 sigmoid。
4. **输出幅值塌缩 5–10 倍**：输入 ~[0,3]，fused ~[-0.6,0.2]。out_proj 含 BN → 零均值强制。**修复：** GroupNorm(8)。
5. **S 整图饱和失去空间选择性**：D 处处非零 → σ(diff_enc(D)) 全图接近饱和，α 只能均匀放大。**修复：** 去 S/α。
6. **W 无通道维**：(B,2,H,W) 所有通道共享空间权重；不同通道（边缘/纹理/颜色/温度）需不同模态偏好。**修复：** 因式分解 W_s ⊗ W_c。

---

## 七、Exp-8 双流架构全面修复

### 7.1 修复策略：先输入质量 → 再融合机制 → 最后架构噪声

每步独立消融，避免多变量叠加影响归因。

| Step | 修复目标 | 实施 | 文件 |
|------|---------|------|------|
| 1 | RGB 语义坍塌 | `aux_head_rgb/ir` @P3，`DualStreamDetectionLoss = main + 0.25·(L_rgb+L_ir)` | `tasks.py`, `train.py` |
| 2 | P4 条带噪声 | backbone layer 6：`A2C2f[512,True,4]` → `C3k2[512,True]`；P5/A2C2f@neck 保留 | `yolov12-dual-p2.yaml` |
| 3 | 融合模块重构 | DMGFusion v2（见 §2.4） | `block.py`, `tasks.py`, YAML |

### 7.2 实验矩阵

| 实验 | P4 backbone | P2 fusion | aux head | CMG/CMA | mAP50-95 | 说明 |
|------|-------------|-----------|:--------:|---------|---------:|------|
| Exp-7a（基线） | A2C2f | DMG v1 | — | P4P5/P5 | 0.628 | 参照 |
| **Exp-8a** ★ | **C3k2** | DMG v1 | ✓ | P4P5/P5 | **0.638** | 历史最佳 |
| Exp-8b | C3k2 | **DMG v2** | ✓ | P4P5/P5 | 0.637 | P2 fused 改善 |
| Exp-8c | C3k2 | **plain** | ✓ | P4P5/P5 | 训练中 | plain vs v1 vs v2 三角 |
| Exp-8d | C3k2 | DMG v1 | ✓ | **无** | 训练中 | 验证 CMG/CMA 冗余 |
| Exp-8e | C3k2 | **plain** | ✓ | **无** | 训练中 | 纯标准中期融合 4-点 + aux（最小化配置） |

### 7.3 Exp-8a 详细结果（val 3366 imgs / 9227 instances）

| Class | Instances | P | R | mAP50 | mAP50-95 |
|-------|----------:|------:|------:|------:|------:|
| **all**  | 9227 | **0.930** | **0.910** | **0.945** | **0.638** |
| smoke    | 4086 | 0.944 | 0.896 | 0.948 | 0.739 |
| fire     | 3401 | 0.933 | 0.916 | 0.947 | 0.593 |
| person   | 1740 | 0.913 | 0.917 | 0.941 | 0.581 |

**模型规模：** 5,552,112 params / 30.63 GFLOPs / 884 layers
**推理：** preprocess 0.33 + inference 7.12 + postprocess 0.74 ≈ **8.19 ms / image**

**DMGFusion v1 @P2 收敛标量（量化证明 Step 1+2 效应）：**

| 实验 | α | β | 解读 |
|------|---:|---:|------|
| Exp-7a（无 aux，P4 A2C2f） | **−1.091** | **0.201** | α 负 → `(1+αS)` 在 S→1 区域近 0 → 模型主动**抑制**差异调制 → 门控实际未启用 |
| **Exp-8a**（+aux, P4 C3k2） | **+0.999** | **0.247** | α 接近饱和正值 → 分歧区放大至 ~2× → 差异门控**主导**前向 |

**结论：** v1 在 Exp-7a 失效并非公式缺陷，而是输入质量+架构噪声共同压制。Step 1+2 清障后，**同一个 v1 公式自发转向非平凡工作点**。这也解释了 Exp-8b (v2) 未显著优于 v1——前两步已解锁 v1 的能力上限。

### 7.4 Exp-8b 详细结果

| Class | Instances | P | R | mAP50 | mAP50-95 |
|-------|----------:|------:|------:|------:|------:|
| all    | 9227 | 0.928 | 0.905 | 0.938 | **0.637** |
| smoke  | 4086 | 0.949 | 0.888 | 0.943 | 0.742 |
| fire   | 3401 | 0.929 | 0.911 | 0.940 | 0.591 |
| person | 1740 | 0.906 | 0.916 | 0.931 | 0.578 |

**模型规模：** 5,554,190 params / 30.55 GFLOPs
**推理：** preprocess 0.44 + inference 7.65 + postprocess 0.77 ms

### 7.5 回退策略

| Step | 失败条件 | 回退 |
|------|---------|------|
| 1 aux head | aux loss 不下降或主 head 退化 >1% | 降 λ 到 0.1；若仍失败则仅 RGB aux |
| 2 P4 C3k2 | smoke mAP 退化 >1% | 仅 neck P4 保留 A2C2f |
| 3 DMG v2 | W 全图压 RGB（w_rgb<0.1） | RGB backbone 未从 Step1 中受益 |

---

## 八、当前最优配置与论文推荐

| 配置 | mAP50 | mAP50-95 | fire | person | 推理(ms) | GFLOPs |
|------|------:|---------:|------:|------:|------:|------:|
| Exp-0（early fusion） | 0.919 | 0.605 | 0.545 | 0.537 |  4.51 |   —  |
| Exp-1（双分支 concat） | 0.933 | 0.623 | 0.575 | 0.559 |  6.96 |   —  |
| Exp-4c ★（CMG+CMA@P5） | 0.934 | 0.632 | 0.582 | 0.573 | 16.05 | 19.65 |
| Exp-7d（+Concat@P2 via FPN） | 0.934 | 0.634 | **0.589** | 0.576 | 19.9 | 21.64 |
| **Exp-8a** ★（+DMGv1@P2+P4 C3k2+aux） | **0.945** | **0.638** | **0.593** | **0.581** |  7.12 | 30.63 |

**代价效益：**
- **Exp-4c → Exp-8a：** mAP50-95 +0.006，fire +1.1%、person +0.8%；推理 −56%（7.12ms vs 16.05ms，本地 GPU 硬件收益）。
- **Exp-4c → Exp-7d：** mAP50-95 +0.002，fire +0.7%；推理 +24%。
- **论文推荐：Exp-8a 作为主方案**（DMGFusion v1@P2 + P4 C3k2 backbone + P3 aux head + CMG@P4P5 + 双向 CMA@P5 + 4-scale head）。

---

## 附录

### A.1 执行备忘

- 训练超参：epochs=300, batch=16, SGD lr0=0.01 lrf=0.01 cos_lr=False val_period=2
- 关注 CPU RAM：训练到 ~epoch 77 可能触发 `_ArrayMemoryError`，需监控并准备 `--resume`
- val.py 速度评估统一跑 10 次取平均
- 推理对比时确保同一硬件（§4.1 表中 Exp-8 行与其他行硬件一致）

### A.2 Checkpoint 版本对照

| 实验 | git commit | 还原文件 |
|------|-----------|---------|
| Exp-1（dual_MF） | `0b9379a` | block.py, tasks.py, modules/__init__.py |
| Exp-2（dual_MF_ChW） | `e5bdb6b` / `92e77f5` | 同上 |
| Exp-4a（CMA_rgb） | `c852185` | block.py, tasks.py |
| Exp-4b/4c/4d（双向CMA） | HEAD | — |
| Exp-7 系 | 见 git log | yolov12-dual-p2.yaml 切换 |
| Exp-8 系 | HEAD | `--aux_loss_weight` + p2_fusion 切换 |
