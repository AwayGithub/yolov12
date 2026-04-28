# ADR-003: 统一 batch=16 / lr0=0.01 公平复现实验计划

**Status:** 进行中。用于重建公平实验坐标系，不替代 ADR-001 的历史档案。
**Date:** 2026-04-28
**Related:** [ADR-001](ADR-001-dual-stream-yolov12-mf.md), [ADR-002](ADR-002-minimal-dual-stream-dmg-fusion.md)

---

## 一、背景

ADR-002 复查发现，历史最好 Exp-8d 使用的训练口径为：

| 项目 | 历史 Exp-8d |
|------|------|
| batch | 32 |
| lr0 | 0.02 |
| best mAP50-95 | 0.64361 |

而后续 BidirLiCMA、Adapter、SGMC、plainP2 复现实验主要使用：

| 项目 | 当前统一口径 |
|------|------|
| batch | 16 |
| lr0 | 0.01 |

虽然两者满足近似线性缩放关系，但实际训练中 batch size 会影响 BN 统计、梯度噪声、warmup、scheduler 与最优 basin。因此，旧 Exp-8d 的 0.64361 暂时只能作为历史上限，不能直接作为后续新模块的公平比较基线。

ADR-003 的目标是：在完全一致的训练参数下，重跑一组最小正交消融，重新确认各结构因素的真实贡献。

---

## 二、统一公平训练参数

除非另有明确记录，ADR-003 下所有实验必须使用以下参数：

| 参数 | 统一值 |
|------|------|
| 数据集 | `ultralytics/cfg/datasets/RGBT-3M.yaml` |
| input_mode | `dual_input` |
| imgsz | `[480, 640]` |
| epochs | 200 或 300；同一组对照必须一致 |
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
| 禁止变量 | 不混用 batch=32/lr0=0.02，不中途更换 optimizer/scheduler/epoch 口径 |

每次训练必须保存：

- `results.csv`
- `args.yaml`
- `source_snapshot/train.py`
- `source_snapshot/<model_cfg>.yaml`
- `source_snapshot/model_yaml.yaml`
- best 权重的 `val.py` 分类别表格截图或文本

---

## 三、第一批必跑实验：plainP2 的 2×2 正交消融

先固定 P2 fusion = plain Concat+1×1Conv，只考察两个历史关键因素：

| 因素 | 取值 |
|------|------|
| P4 block | A2C2f / C3k2 |
| P3 aux | off / on |

实验矩阵：

| 编号 | 实验名建议 | P2 fusion | P4 block | P3 aux | 状态 | 目的 |
|------|------|------|------|------|------|------|
| F1 | `fair_plainP2_A2C2fP4_noaux_P2345` | plain | A2C2f | off | 待跑 | 原始 plainP2 底座 |
| F2 | `fair_plainP2_A2C2fP4_P3aux_P2345` | plain | A2C2f | on | 待跑 | 单独评估 aux 在旧 P4 下的收益 |
| F3 | `fair_plainP2_C3k2P4_noaux_P2345` | plain | C3k2 | off | 待跑 | 单独评估 C3k2 在无 aux 下的收益 |
| F4 | `train_MF_plainP2_P2345_P4C3k2_P3aux` | plain | C3k2 | on | 进行中，服务器 `tmux1/train3` | 干净 plainP2 对照；对应旧 Exp-8e 口径 |

这四个实验回答：

1. P4 C3k2 是否在公平口径下仍然有效；
2. P3 aux 是否在公平口径下仍然有效；
3. C3k2 与 aux 是否存在叠加或互补；
4. plainP2 干净基线到底是多少。

---

## 四、第一批结构主线实验

在 F4 的干净 plainP2 基线上，对比 DMG 与 SGMC：

| 编号 | 实验名建议 | P2 fusion | P4 block | P3 aux | SGMC | 状态 | 目的 |
|------|------|------|------|------|------|------|------|
| M1 | `dual_MF_DMGFusionP2_P2345_P4C3k2_P3aux` | DMG v1 | C3k2 | on | off | 已完成，服务器 `tmux0/train` | 公平口径下 DMG v1 基线，best mAP50-95=0.63348 |
| M2 | `train_MF_plainP2_P2345_P4C3k2_P3aux` | plain | C3k2 | on | off | 进行中，服务器 `tmux1/train3` | 判断 DMG v1 相对 plain 是否仍有增益 |
| M3 | `dual_MF_DMGFusionP2_SGMCp5_P345_P4C3k2_P3aux` | DMG v1 | C3k2 | on | on | 进行中，服务器 `tmux2/train2` | 判断 SGMC 是否能作为第二创新点 |

关键判据：

| 判断 | 条件 |
|------|------|
| DMG v1 仍有效 | M1 > M2，且差值 ≥ 0.003 或 fire/person 至少一类明显提升 |
| SGMC 有论文价值 | M3 > M1，且差值 ≥ 0.003，或三类中至少 fire/person 有稳定收益 |
| SGMC 边界有效 | M3 与 M1 持平，但参数/速度代价很低，且分类别有可解释局部收益 |
| SGMC 放弃 | M3 < M1 且三类 mAP50-95 无亮点 |

---

## 五、第二批候选实验

第一批完成后，再决定是否补以下实验。不要在第一批未出结果前扩展变量。

| 编号 | 实验名建议 | 配置 | 目的 | 触发条件 |
|------|------|------|------|------|
| S1 | `fair_DMGv2P2_C3k2P4_P3aux_P2345` | DMG v2 + C3k2 + aux | 重算 v1/v2 排序 | M1 明显优于 M2 后再跑 |
| S2 | `fair_DMGv1P2_A2C2fP4_P3aux_P2345` | DMG v1 + A2C2f + aux | 判断 DMG 是否依赖 C3k2 清障 | F2 与 F4 差距较大时跑 |
| S3 | `fair_SGMC_plainP2_C3k2P4_P3aux_P2345` | plain + SGMC + C3k2 + aux | 判断 SGMC 是否依赖 DMG | M3 有正收益时跑 |
| S4 | `fair_DMGv1mini_C3k2P4_P3aux_P2345` | DMG v1-mini + C3k2 + aux | 论文轻量消融 | M1 > M2 后跑 |

暂不优先继续：

- BidirLiCMA@P5；
- Adapter@P5；
- dropout/drop-path 细调；
- IR backbone 大改；
- SGMC ratio/gate 网格搜索。

原因：这些方向依赖公平坐标系。第一批结果未稳定前继续扩展，会再次引入归因混乱。

---

## 六、当前服务器实验映射

| tmux | 目录 | 实验 | 状态 |
|------|------|------|------|
| tmux0 | `train` | M1: DMG v1 + C3k2 + P3 aux | 已完成，best mAP50-95=0.63348 |
| tmux1 | `train3` | M2/F4: plainP2 + C3k2 + P3 aux | 进行中 |
| tmux2 | `train2` | M3: SGMC + DMG v1 + C3k2 + P3 aux | 进行中 |

---

## 七、结果记录模板

每个实验完成后补充：

| 字段 | 记录 |
|------|------|
| 实验名 |  |
| 保存目录 |  |
| cfg snapshot |  |
| batch / lr0 | 16 / 0.01 |
| best epoch |  |
| P / R / mAP50 / mAP50-95 |  |
| smoke mAP50-95 |  |
| fire mAP50-95 |  |
| person mAP50-95 |  |
| train loss sum @best |  |
| val loss sum @best |  |
| 过拟合判断 |  |
| 结论 |  |

---

## 八、阶段决策规则

1. 第一批先跑完 F1-F4 与 M1-M3，不追加新变量。
2. 所有主表只使用统一 batch=16/lr0=0.01 的结果。
3. 旧 Exp-8d 0.64361 只作为历史最好和上限，不作为公平消融主表基线。
4. 若 M1 不超过 M2，则 DMG v1 作为主创新点需要降级，仅保留机制分析或历史证据。
5. 若 M3 超过 M1，则 SGMC 可以作为第二创新点继续细化；否则 SGMC 作为边界实验记录。
6. 若 F4 显著优于 F1-F3，则“P4 C3k2 + P3 aux”仍可作为稳定结构修复点；否则需要重新审视原主线叙事。
