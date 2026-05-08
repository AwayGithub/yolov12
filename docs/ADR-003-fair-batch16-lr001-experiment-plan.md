# ADR-003: 统一 batch=16 / lr0=0.01 公平实验记录

**Status:** F1-F4 与 M1-M3 已完成  
**Date:** 2026-04-28  
**Related:** [ADR-001](ADR-001-dual-stream-yolov12-mf.md), [ADR-002](ADR-002-minimal-dual-stream-dmg-fusion.md)

## 一、公平训练参数设置

除非另有明确记录，ADR-003 下实验均采用以下公平口径：

| 参数 | 统一值 |
| -- | -- |
| 数据集 | `ultralytics/cfg/datasets/RGBT-3M.yaml` |
| input_mode | `dual_input` |
| imgsz | `[480, 640]` |
| epochs | 同一组对照保持一致 |
| batch | **16** |
| optimizer | SGD |
| lr0 | **0.01** |
| lrf | 0.01 |
| momentum | 0.937 |
| weight_decay | 5e-4 |
| warmup_epochs | 3.0 |
| warmup_momentum | 0.8 |
| warmup_bias_lr | 0.0 |
| cos_lr | False |
| val_period | 2 |
| workers | 0 |
| device | 0 |
| seed | **0** |
| deterministic | **True** |
| aux_loss_weight | 0.25（启用 aux 时） |

结果记录口径：

| 项目 | 记录规则 |
| -- | -- |
| 主验证表与分类别结果 | 优先采用 `results1.png` 的 best 权重重新验证结果 |
| `results.csv` | 仅用于训练 loss、验证 loss、best/last epoch 曲线和过拟合趋势分析 |
| F1-F3 特例 | 当前按复查需求记录 `results.csv` 最后一行分类别结果 |
| F4 特例 | 使用已复核的 `results1.png` 结果 |

## 二、F1-F4：plainP2 2x2 正交消融

### 2.1 训练配置

| 编号 | 实验目录 | P2 fusion | P4 block | P3 aux | 数据来源 |
| -- | -- | -- | -- | -- | -- |
| F1 | `train_MF_plainP2_P2345_P4A2C2f_P3noaux` | plain | A2C2f | off | CSV last epoch 197 |
| F2 | `train_MF_plainP2_P2345_P4A2C2f_P3aux` | plain | A2C2f | on | CSV last epoch 200 |
| F3 | `dual_MF_plainP2_P2345_P4C3k2_P3noaux` | plain | C3k2 | off | CSV last epoch 200 |
| F4 | `dual_MF_plainP2_P2345_P4C3k2_P3aux` | plain | C3k2 | on | `results1.png` |

### 2.2 F1 验证结果

| Class | P | R | mAP50 | mAP50-95 |
| -- | --: | --: | --: | --: |
| all | 0.92850 | 0.89365 | 0.93870 | **0.63138** |
| smoke | 0.95008 | 0.88042 | 0.94273 | 0.73185 |
| fire | 0.92772 | 0.90191 | 0.93479 | 0.58004 |
| person | 0.90770 | 0.89863 | 0.93859 | 0.58226 |

### 2.3 F2 验证结果

| Class | P | R | mAP50 | mAP50-95 |
| -- | --: | --: | --: | --: |
| all | 0.92889 | 0.90679 | 0.94096 | **0.63481** |
| smoke | 0.94353 | 0.89970 | 0.94859 | 0.73933 |
| fire | 0.93152 | 0.91032 | 0.93736 | 0.58632 |
| person | 0.91162 | 0.91034 | 0.93694 | 0.57880 |

### 2.4 F3 验证结果

| Class | P | R | mAP50 | mAP50-95 |
| -- | --: | --: | --: | --: |
| all | 0.92986 | 0.89502 | 0.93661 | **0.63075** |
| smoke | 0.95158 | 0.87543 | 0.93933 | 0.73410 |
| fire | 0.92570 | 0.90503 | 0.93790 | 0.58255 |
| person | 0.91231 | 0.90460 | 0.93260 | 0.57560 |

### 2.5 F4 验证结果

| Class | P | R | mAP50 | mAP50-95 |
| -- | --: | --: | --: | --: |
| all | 0.930 | 0.906 | 0.941 | **0.634** |
| smoke | 0.946 | 0.885 | 0.942 | 0.737 |
| fire | 0.931 | 0.920 | 0.942 | 0.588 |
| person | 0.913 | 0.915 | 0.937 | 0.578 |

### 2.6 简要分析

| 对比 | all mAP50-95 变化 | 观察 |
| -- | --: | -- |
| F2 - F1 | +0.00343 | A2C2f@P4 下加入 P3 aux 后整体提升，主要来自 recall、smoke 与 fire 的改善 |
| F4 - F3 | 约 +0.00325 | C3k2@P4 下加入 P3 aux 同样提升，说明 P3 aux 是该组消融中最稳定的有效因素 |
| F3 - F1 | -0.00063 | 无 P3 aux 时，C3k2@P4 未优于 A2C2f@P4 |
| F4 - F2 | 约 -0.00081 | 有 P3 aux 时，C3k2@P4 与 A2C2f@P4 基本持平，未形成明确优势 |

F1-F4 表明，公平口径下最明确的收益来自 P3 aux；P4 block 从 A2C2f 替换为 C3k2 没有证明出稳定增益。若用于论文消融，P3 aux 可以作为更稳的结构贡献，P4 C3k2 更适合谨慎表述为结构调整或近似持平方案。

## 三、M1-M3：结构主线实验

### 3.1 训练配置

| 编号 | 实验目录 | P2 fusion | P4 block | P3 aux | SGMC | 数据来源 |
| -- | -- | -- | -- | -- | -- | -- |
| M1 | `dual_MF_DMGFusionP2_P2345_P4C3k2_P3aux` | DMG v1 | C3k2 | on | off | `results1.png` |
| M2 | `dual_MF_plainP2_P2345_P4C3k2_P3aux` | plain | C3k2 | on | off | `results1.png` |
| M3 | `dual_MF_DMGFusionP2_SGMCP5_P2345_P4C3k2_P3aux` | DMG v1 | C3k2 | on | on | `results1.png` |

### 3.2 M1 验证结果

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
| -- | --: | --: | --: | --: | --: | --: |
| all | 3366 | 9227 | 0.931 | 0.909 | 0.942 | **0.634** |
| smoke | 2398 | 4086 | 0.946 | 0.892 | 0.945 | 0.735 |
| fire | 2606 | 3401 | 0.938 | 0.920 | 0.947 | 0.587 |
| person | 1243 | 1740 | 0.908 | 0.915 | 0.934 | 0.579 |

### 3.3 M2 验证结果

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
| -- | --: | --: | --: | --: | --: | --: |
| all | 3366 | 9227 | 0.930 | 0.906 | 0.941 | **0.634** |
| smoke | 2398 | 4086 | 0.946 | 0.885 | 0.942 | 0.737 |
| fire | 2606 | 3401 | 0.931 | 0.920 | 0.942 | 0.588 |
| person | 1243 | 1740 | 0.913 | 0.915 | 0.937 | 0.578 |

### 3.4 M3 验证结果

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
| -- | --: | --: | --: | --: | --: | --: |
| all | 3366 | 9227 | 0.927 | 0.906 | 0.940 | **0.635** |
| smoke | 2398 | 4086 | 0.942 | 0.890 | 0.942 | 0.731 |
| fire | 2606 | 3401 | 0.934 | 0.916 | 0.944 | 0.594 |
| person | 1243 | 1740 | 0.905 | 0.913 | 0.934 | 0.580 |

### 3.5 简要分析

| 对比 | all mAP50-95 变化 | 观察 |
| -- | --: | -- |
| M1 - M2 | 0.000 | DMG v1@P2 与 plainP2 总体持平，公平口径下未证明 DMG v1 带来明确全局增益 |
| M3 - M1 | +0.001 | SGMC 在 DMG v1 基础上只有很小的整体提升，低于强结论所需幅度 |
| M3 - M2 | +0.001 | SGMC+DMG 相比 plainP2 也只是边界收益 |
| M3 fire - M2 fire | +0.006 | SGMC 对 fire 类 mAP50-95 的改善最明显，是当前最有解释价值的局部收益 |
| M3 smoke - M2 smoke | -0.006 | SGMC 损伤 smoke 类 mAP50-95，说明高层语义校准存在类别间 trade-off |

M1-M3 表明，DMG v1@P2 在本轮公平复现中不能继续作为“明确提升精度”的结论；SGMC 的全局收益很小，但对 fire 类有局部改善。后续若继续使用 SGMC，需要围绕 fire 类增强与类别间权衡进行表述，而不能把它包装为稳定的大幅提升模块。
