# ADR-003: 统一 batch=16 / lr0=0.01 公平复现实验计划

**Status:** 第一批结构主线（M1/M2/M3）已完成。公平口径下 DMG v1@P2 未证明相对 plainP2 的正增益；SGMC 仅呈现边界收益。
**Date:** 2026-04-28
**Related:** [ADR-001](ADR-001-dual-stream-yolov12-mf.md), [ADR-002](ADR-002-minimal-dual-stream-dmg-fusion.md)

***

## 一、背景

ADR-002 复查发现，历史最好 Exp-8d 使用的训练口径为：

| 项目            | 历史 Exp-8d |
| ------------- | --------- |
| batch         | 32        |
| lr0           | 0.02      |
| best mAP50-95 | 0.64361   |

而后续 BidirLiCMA、Adapter、SGMC、plainP2 复现实验主要使用：

| 项目    | 当前统一口径 |
| ----- | ------ |
| batch | 16     |
| lr0   | 0.01   |

虽然两者满足近似线性缩放关系，但实际训练中 batch size 会影响 BN 统计、梯度噪声、warmup、scheduler 与最优 basin。因此，旧 Exp-8d 的 0.64361 暂时只能作为历史上限，不能直接作为后续新模块的公平比较基线。

ADR-003 的目标是：在完全一致的训练参数下，重跑一组最小正交消融，重新确认各结构因素的真实贡献。

***

## 二、统一公平训练参数

除非另有明确记录，ADR-003 下所有实验必须使用以下参数：

| 参数                | 统一值                                                      |
| ----------------- | -------------------------------------------------------- |
| 数据集               | `ultralytics/cfg/datasets/RGBT-3M.yaml`                  |
| input\_mode       | `dual_input`                                             |
| imgsz             | `[480, 640]`                                             |
| epochs            | 200 或 300；同一组对照必须一致                                      |
| batch             | **16**                                                   |
| optimizer         | SGD                                                      |
| lr0               | **0.01**                                                 |
| lrf               | 0.01                                                     |
| momentum          | 0.937                                                    |
| weight\_decay     | 5e-4                                                     |
| warmup\_epochs    | 3.0                                                      |
| warmup\_momentum  | 0.8                                                      |
| warmup\_bias\_lr  | 0.0                                                      |
| cos\_lr           | False                                                    |
| val\_period       | 2                                                        |
| workers           | 0                                                        |
| device            | 0                                                        |
| seed              | **0**                                                    |
| deterministic     | **True**                                                 |
| aux\_loss\_weight | 0.25（启用 aux 时）                                           |
| 禁止变量              | 不混用 batch=32/lr0=0.02，不中途更换 optimizer/scheduler/epoch 口径 |

每次训练必须保存：

- `results.csv`
- `args.yaml`
- `source_snapshot/train.py`
- `source_snapshot/<model_cfg>.yaml`
- `source_snapshot/model_yaml.yaml`
- best 权重的 `val.py` 分类别表格截图或文本

结果记录口径：

- **分类别结果与主表验证结果以** **`results1.png`** **为准**，即 best 权重重新验证得到的 per-class 表格；
- `results.csv` 仅用于训练 loss、验证 loss、best/last epoch 曲线和过拟合趋势分析；
- 若 `results1.png` 与 CSV best 行存在轻微差异，论文表格和 ADR 主结论优先采用 `results1.png`；
- CSV best 行可辅助定位 best epoch，但不作为分类别结果来源。

***

## 三、第一批必跑实验：plainP2 的 2×2 正交消融

先固定 P2 fusion = plain Concat+1×1Conv，只考察两个历史关键因素：

| 因素       | 取值           |
| -------- | ------------ |
| P4 block | A2C2f / C3k2 |
| P3 aux   | off / on     |

实验矩阵：

| 编号 | 实验名建议                                 | P2 fusion | P4 block | P3 aux | 状态                     | 目的                          |
| -- | ------------------------------------- | --------- | -------- | ------ | ---------------------- | --------------------------- |
| F1 | `fair_plainP2_A2C2fP4_noaux_P2345`    | plain     | A2C2f    | off    | 待跑                     | 原始 plainP2 底座               |
| F2 | `fair_plainP2_A2C2fP4_P3aux_P2345`    | plain     | A2C2f    | on     | 待跑                     | 单独评估 aux 在旧 P4 下的收益         |
| F3 | `fair_plainP2_C3k2P4_noaux_P2345`     | plain     | C3k2     | off    | 待跑                     | 单独评估 C3k2 在无 aux 下的收益       |
| F4 | `dual_MF_plainP2_P2345_P4C3k2_P3aux` | plain     | C3k2     | on     | **已完成，服务器 `tmux1/train3`，results1 mAP50-95=0.634** | 干净 plainP2 对照；对应旧 Exp-8e 口径 |

这四个实验回答：

1. P4 C3k2 是否在公平口径下仍然有效；
2. P3 aux 是否在公平口径下仍然有效；
3. C3k2 与 aux 是否存在叠加或互补；
4. plainP2 干净基线到底是多少。

***

## 四、第一批结构主线实验

在 F4 的干净 plainP2 基线上，对比 DMG 与 SGMC：

| 编号 | 实验名建议                                          | P2 fusion | P4 block | P3 aux | SGMC | 状态                             | 目的                                    |
| -- | ---------------------------------------------- | --------- | -------- | ------ | ---- | ------------------------------ | ------------------------------------- |
| M1 | `dual_MF_DMGFusionP2_P2345_P4C3k2_P3aux`       | DMG v1    | C3k2     | on     | off  | 已完成，results1 mAP50-95=0.634 | 公平口径下 DMG v1 对照；未超过 plainP2 |
| M2 | `dual_MF_plainP2_P2345_P4C3k2_P3aux`           | plain     | C3k2     | on     | off  | **已完成，服务器 `tmux1/train3`，results1 mAP50-95=0.634** | 判断 DMG v1 相对 plain 是否仍有增益 |
| M3 | `dual_MF_DMGFusionP2_SGMCp5_P345_P4C3k2_P3aux` | DMG v1    | C3k2     | on     | on   | **已完成，results1 mAP50-95=0.635** | 判断 SGMC 是否能作为第二创新点 |

关键判据：

| 判断         | 条件                                           |
| ---------- | -------------------------------------------- |
| DMG v1 仍有效 | M1 > M2，且差值 ≥ 0.003 或 fire/person 至少一类明显提升 |
| DMG v1 未确认正增益 | M1 与 M2 持平或低于 M2，且分类别 mAP50-95 无稳定收益 |
| SGMC 有论文价值 | M3 > M1，且差值 ≥ 0.003，或三类中至少 fire/person 有稳定收益 |
| SGMC 边界有效  | M3 与 M1 持平，但参数/速度代价很低，且分类别有可解释局部收益           |
| SGMC 放弃    | M3 < M1 且三类 mAP50-95 无亮点                     |

***

## 五、第二批候选实验

第一批完成后，再决定是否补以下实验。不要在第一批未出结果前扩展变量。

| 编号 | 实验名建议                                  | 配置                        | 目的                  | 触发条件           |
| -- | -------------------------------------- | ------------------------- | ------------------- | -------------- |
| S1 | `yolov12-dual-p2-plain-a2c2fp4-noaux.yaml` | plain + A2C2f + no aux | 补齐 2×2 正交消融底座 | 需要确认 C3k2 / aux 独立贡献时跑 |
| S2 | `yolov12-dual-p2-plain-a2c2fp4-p3aux.yaml` | plain + A2C2f + aux | 单独评估 P3 aux 在旧 P4 下的收益 | 需要确认 aux 独立贡献时跑 |
| S3 | `yolov12-dual-p2-plain-c3k2p4-noaux.yaml` | plain + C3k2 + no aux | 单独评估 C3k2 在无 aux 下的收益 | 需要确认 C3k2 独立贡献时跑 |
| S4 | `fair_SGMC_plainP2_C3k2P4_P3aux_P2345` | plain + SGMC + C3k2 + aux | 判断 SGMC 是否依赖 DMG | 若仍考虑 SGMC 作为边界创新点时跑 |

暂不优先继续：

- DMG v1 变体扩展；
- BidirLiCMA\@P5；
- Adapter\@P5；
- dropout/drop-path 细调；
- IR backbone 大改；
- SGMC ratio/gate 网格搜索。

原因：M1 vs M2 已显示 DMG v1@P2 在公平口径下未确认正增益，继续扩展 DMG v1 变体会优先级偏低。若继续补实验，应优先补齐 plainP2 的 2×2 正交消融，确认 P4 C3k2 与 P3 aux 是否才是稳定有效结构因素。

***

## 六、当前未完成实验映射

已完成实验不再在本节重复记录，详见 §七。

| tmux  | 目录       | 实验                             | 状态  |
| ----- | -------- | ------------------------------ | --- |
| 暂无 | - | - | - |

***

## 七、fair 目录已完成实验记录

结果目录：`E:\Yan-Unifiles\lab\exp\yolov12\runs\detect\RGBT-3M\fair`

> 注：分类别结果与主表验证结果均以 `results1.png` 为准；CSV 仅用于 loss 与过拟合趋势分析。

### 7.1 M1：DMG v1 fair 基线

| 字段       | 记录                                                                               |
| -------- | -------------------------------------------------------------------------------- |
| 实验编号     | M1                                                                               |
| 实验目录     | `dual_MF_DMGFusionP2_P2345_P4C3k2_P3aux`                                         |
| cfg      | `yolov12-dual-p2.yaml`                                                           |
| results1 | `fair/dual_MF_DMGFusionP2_P2345_P4C3k2_P3aux/results1.png`                       |
| CSV 用途   | best epoch 170；train loss sum 6.34666；val loss sum 3.61185；last mAP50-95 0.63332 |
| 结论       | 公平口径 DMG v1 对照。相对历史 Exp-8d results1 0.644 下降约 0.010；且与 M2 plainP2 持平，因此 DMG v1@P2 的正增益在本轮公平复现中未被确认 |

| Class  | Images | Instances |     P |     R | mAP50 |  mAP50-95 |
| ------ | -----: | --------: | ----: | ----: | ----: | --------: |
| all    |   3366 |      9227 | 0.931 | 0.909 | 0.942 | **0.634** |
| smoke  |   2398 |      4086 | 0.946 | 0.892 | 0.945 |     0.735 |
| fire   |   2606 |      3401 | 0.938 | 0.920 | 0.947 |     0.587 |
| person |   1243 |      1740 | 0.908 | 0.915 | 0.934 |     0.579 |

### 7.2 M2/F4：plainP2 fair 对照

| 字段       | 记录                                                                                                                 |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| 实验编号     | M2 / F4                                                                                                            |
| 运行位置     | 服务器 `tmux1`                                                                                                      |
| 工作目录     | `train3`                                                                                                           |
| 实验目录     | `dual_MF_plainP2_P2345_P4C3k2_P3aux`                                                                                |
| cfg      | plainP2 + P4 C3k2 + P3 aux + P2/P3/P4/P5 head                                                                       |
| results1 | `fair/dual_MF_plainP2_P2345_P4C3k2_P3aux/results1.png`                                                             |
| CSV 用途   | best epoch 199；train loss sum 6.03498；val loss sum 3.57501；last mAP50-95 0.63421                                  |
| 结论       | plainP2 在公平口径下达到 results1 mAP50-95=0.634，与 M1 的 DMG v1 持平；DMG v1 相对 plainP2 的明确数值增益在本轮公平复现中未被确认 |

| Class  | Images | Instances |     P |     R | mAP50 |  mAP50-95 |
| ------ | -----: | --------: | ----: | ----: | ----: | --------: |
| all    |   3366 |      9227 | 0.930 | 0.906 | 0.941 | **0.634** |
| smoke  |   2398 |      4086 | 0.946 | 0.885 | 0.942 |     0.737 |
| fire   |   2606 |      3401 | 0.931 | 0.920 | 0.942 |     0.588 |
| person |   1243 |      1740 | 0.913 | 0.915 | 0.937 |     0.578 |

### 7.3 M3：SGMC + DMG v1

| 字段       | 记录                                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------------------- |
| 实验编号     | M3                                                                                                               |
| 实验目录     | `dual_MF_DMGFusionP2_SGMCP5_P2345_P4C3k2_P3aux`                                                                  |
| cfg      | `yolov12-dual-p2-sgmc.yaml`                                                                                      |
| results1 | `fair/dual_MF_DMGFusionP2_SGMCP5_P2345_P4C3k2_P3aux/results1.png`                                                |
| CSV 用途   | best epoch 169；train loss sum 6.35125；val loss sum 3.49580；last mAP50-95 0.63492                                 |
| 结论       | SGMC 相对 M1 全局 +0.001、相对 M2 全局 +0.001（results1 口径），未达到 +0.003 强判据；fire 是主要收益类别，暂定为边界有效 |

| Class  | Images | Instances |     P |     R | mAP50 |  mAP50-95 |
| ------ | -----: | --------: | ----: | ----: | ----: | --------: |
| all    |   3366 |      9227 | 0.927 | 0.906 | 0.940 | **0.635** |
| smoke  |   2398 |      4086 | 0.942 | 0.890 | 0.942 |     0.731 |
| fire   |   2606 |      3401 | 0.934 | 0.916 | 0.944 |     0.594 |
| person |   1243 |      1740 | 0.905 | 0.913 | 0.934 |     0.580 |

### 7.4 M1 / M2 / M3 当前判断

| 对比 | Δ mAP50-95 | 判断 |
| -- | --: | -- |
| M1 - M2（DMG v1 vs plainP2） | **0.000** | all mAP50-95 持平；DMG v1 的公平数值增益未被确认 |
| M3 - M1（SGMC vs DMG v1） | **+0.001** | 小幅正收益，但低于 ADR-003 预设的 +0.003 强增益阈值 |
| M3 - M2（SGMC+DMG vs plainP2） | **+0.001** | 整体收益仍很小，不足以单独支撑强结论 |
| smoke：M3 - M2 | **-0.006** | SGMC 损伤 smoke mAP50-95 |
| fire：M3 - M2 | **+0.006** | SGMC 对 fire 有最清晰收益 |
| person：M3 - M2 | **+0.002** | person 小幅收益 |

阶段结论：公平口径下，M2 plainP2 与 M1 DMG v1 的 all mAP50-95 同为 0.634，且 smoke/fire 的 mAP50-95 还分别低于 plainP2 0.002/0.001。DMG v1@P2 没有证明出稳定正作用，不能再作为“明确提升精度”的主创新点。后续叙事应转向 plainP2 + P4 C3k2 + P3 aux 这个稳定底座，或继续验证 SGMC 这类高层语义校准是否能提供第二个边界有效点。SGMC 当前仅带来 all +0.001，主要改善 fire（相对 M2 +0.006），不足以直接声明为强第二创新点。

***

## 八、结果记录模板

每个实验完成后补充：

| 字段                          | 记录        |
| --------------------------- | --------- |
| 实验名                         | <br />    |
| 保存目录                        | <br />    |
| cfg snapshot                | <br />    |
| batch / lr0                 | 16 / 0.01 |
| best epoch                  | <br />    |
| P / R / mAP50 / mAP50-95    | <br />    |
| smoke mAP50-95              | <br />    |
| fire mAP50-95               | <br />    |
| person mAP50-95             | <br />    |
| CSV best epoch / last epoch | <br />    |
| train loss sum @CSV best    | <br />    |
| val loss sum @CSV best      | <br />    |
| 过拟合判断                       | <br />    |
| 结论                          | <br />    |

***

## 九、阶段决策规则

1. 第一批先跑完 F1-F4 与 M1-M3，不追加新变量。
2. 所有主表只使用统一 batch=16/lr0=0.01 且来自 `results1.png` 的结果。
3. 旧 Exp-8d 0.64361 只作为历史最好和上限，不作为公平消融主表基线。
4. M1 未超过 M2，DMG v1@P2 作为主创新点应降级；当前仅保留为机制分析或历史证据，不能作为公平精度增益结论。
5. M3 仅小幅超过 M1/M2，SGMC 暂作为边界实验记录；若要继续，需要补跑 plainP2+SGMC 判断其是否依赖 DMG。
6. 若 F4 显著优于 F1-F3，则“P4 C3k2 + P3 aux”仍可作为稳定结构修复点；否则需要重新审视原主线叙事。
