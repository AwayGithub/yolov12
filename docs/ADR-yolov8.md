# ADR：YOLOv8 Backbone 支线实验

**状态：** 进行中  
**创建日期：** 2026-07-05  
**最后更新：** 2026-07-10  
**相关文档：** [ADR-005](ADR-005-fire-person-binary-scope.md)

## 一、当前结论

当前 YOLOv8 backbone 支线主链固定为：

`F0 -> F2 -> D2.1 -> A1-y8n-backbone -> C6.5-y8n-backbone`

已完成实验中，`A1-y8n-backbone` 是当前最优基础点：overall `mAP50-95=0.59292`。`C6.5` 按 epoch180/181 记录达到 `0.59452`，超过 A1。因此 P4/P5 高层融合方式固定为 `C6.5` 的 RS-SQF 配置。

主结论：

1. `YOLOv8n C2f + SPPF` 主体适合作为 fire/person 二分类双流支线基础。
2. `P2 DMGInit8d + P3 aux + P3 CFRA` 是当前有效组合。
3. `CFRA` 只放 `P3` 最稳，扩展到 `P4` 后不增益。
4. 同一 A1 创新组合下，YOLOv8n 骨架明显好于 YOLOv12n 骨架。
5. 高层语义交互支线中，`C6.5` 已首次显示出超过 `A1-y8n-backbone` 的趋势，P4/P5 融合后续固定采用 `C6.5` 配置。
6. 6 个层级消融已完成：三个同构铺满实验均低于 C6.5，两个高层错位实验明显低于 C6.5；`Perm-A` 接近 C6.5 但未超过。

## 二、统一实验口径

- 任务：fire/person 二分类检测
- 数据：`ultralytics/cfg/datasets/RGBT-3M-dual-fire-person-local.yaml`
- 训练入口：`python train.py`
- 输入：`input_mode=dual_input`，`imgsz=[480,640]`
- 训练参数：`epochs=200`，`batch=16`，`lr0=0.01`，`cls=0.1`，`seed=0`，`amp=True`
- 指标口径：默认取 `results.csv` 最后一行；已停止实验记录停止时 latest csv

术语：

- `plain`：普通 `concat + 1x1 conv` 融合。
- `CFRA`：频域融合，在代码中对应 `FreDFTFusion`。
- `CSPA`：高层跨模态空间/语义交互模块。
- `RS-SQF`：`RedundancySuppressedSparseSemanticQueryFusion`，先抑制模态内冗余，再用稀疏语义 query 做跨模态读写。

## 三、YOLOv12 与 YOLOv8 主体差异

本支线只替换检测主体风格，不改变 fire/person 二分类任务和训练口径。

| 维度 | YOLOv12n 主体 | YOLOv8n 主体 | 对本支线的含义 |
| --- | --- | --- | --- |
| 基础 block | `C3k2 + A2C2f` | `C2f + SPPF` | YOLOv8n 更偏卷积/CSP，YOLOv12n 引入 area-attention |
| 高层特征 | P4/P5 使用 `A2C2f` | P4/P5 使用 `C2f`，顶部保留 `SPPF` | YOLOv8n 高层更简单，可能更利于小数据/二分类稳定训练 |
| Neck/head 主体 | 多处 `A2C2f/C3k2` 融合 | 多处 `C2f` 融合 | YOLOv8n 的融合路径更直接，额外创新模块更容易隔离影响 |
| 实现代价 | 注意力算子多，实际 latency 容易偏高 | 卷积算子为主，工程上更稳定 | 本 ADR 重点看精度，但 YOLOv8n 也更适合作为轻量实验底座 |

关键实验现象：同一 `P2 DMGInit8d + P3 aux + P3 CFRA` 组合下，YOLOv12 主体的 `A1` 为 `mAP50-95=0.57409`，而 `A1-y8n-backbone` 达到 `0.59292`，提升 `+0.01883`。这说明当前 fire/person 二分类口径下，效果瓶颈不在 YOLOv8n 主体表达能力，反而是 YOLOv12n 的 attention 主体未带来收益；后续因此把 YOLOv8n backbone 作为此支线主参考骨架。

## 四、实验含义与状态

| 实验 | 配置 | 含义 | 状态 | 结论 |
| --- | --- | --- | --- | --- |
| RGB teacher | `yolov8n.yaml` | 单模态 RGB YOLOv8n 参考基线 | 已完成 | 明显低于双流结果 |
| IR teacher | `yolov8n.yaml` | 单模态 IR YOLOv8n 参考基线 | 已完成 | 强于 RGB，但仍低于双流结果 |
| F0-y8n-backbone | `F0-y8n-backbone-fire-person.yaml` | YOLOv8n 双流基础三尺度起点，无 P2、无 P3 aux | 已完成 | 主链起点 |
| F2-y8n-backbone | `F2-y8n-backbone-fire-person.yaml` | 在 F0 上加入普通 P2 融合与 P3 aux | 已完成 | 相对 F0 明显提升 |
| D2.1-y8n-backbone | `D21-y8n-backbone-fire-person.yaml` | 在 F2 上把 P2 普通融合换成 `DMGInit8d` | 已完成 | 证明 P2 DMG 有收益 |
| A1-y8n-backbone | `A1-y8n-backbone-fire-person.yaml` | 在 D2.1 上加入 P3 CFRA | 已完成 | 当前 YOLOv8 主参考点 |
| A2-y8n-backbone | `A2-y8n-backbone-fire-person.yaml` | 在 A1 上把 CFRA 扩展到 P4 | 已完成 | 不如 A1，P4 CFRA 无收益 |
| C1 | `C1-y8n-backbone-fire-person.yaml` | 在 A1 上于 P4/P5 加 CSPA | 已完成 | 接近 A1，但未超越 |
| C2 | `C2-y8n-backbone-fire-person.yaml` | 在 A1 上仅 P4 加 CSPA | 已停止 | 低于 C1，单层 P4 不足 |
| B2-y8n-backbone | `B2-y8n-backbone-fire-person.yaml` | 在 A1 上把 P4 C2f 直接替换为 pcrossA2C2f | 已停止 | 直接迁移失败 |
| C3 | `C3-y8n-backbone-fire-person.yaml` | 在 C1 上中等增容 CSPA | 已停止 | 增容未带来收益 |
| C4 | `C4-y8n-backbone-fire-person.yaml` | 在 C1 上大幅增容 CSPA | 已停止 | 增容未带来收益 |
| C5 | `C5-y8n-backbone-fire-person.yaml` | 在 C1 上把 P4/P5 CSPA 堆叠到 2 层 | 已停止 | 堆叠未带来收益 |
| C6.1 | `C61-y8n-backbone-fire-person.yaml` | 在 A1 上用最小参数 RS-SQF 替换 P4/P5 融合 | 已完成 | 低于 C1/A1 |
| C6.2 | `C62-y8n-backbone-fire-person.yaml` | 在 C6.1 上增加 query/read 预算 | 已完成 | 略高于 C1，但仍低于 A1 |
| C6.3 | `C63-y8n-backbone-fire-person.yaml` | 在 C6.2 上增大 RS-SQF 隐空间宽度 | 已停止 | 不如 C6.5，不采用 |
| C6.4 | `C64-y8n-backbone-fire-person.yaml` | 在 C6.3 上继续增加 query/top-k 预算 | 已停止 | 不如 C6.5，不采用 |
| C6.5 | `C65-y8n-backbone-fire-person.yaml` | 在 C6.4 上增加 attention heads | 已完成 | epoch180/181 超过 A1 |

## 五、C6 参数递进

| 实验 | hidden_ratio | heads | P4 queries | P4 topk | P5 queries | P5 topk | 设计意图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C6.1 | 0.25 | 4 | 8 | 4 | 4 | 4 | 最小参数闭环 |
| C6.2 | 0.25 | 4 | 8 | 6 | 6 | 4 | 先增加 query/read 预算 |
| C6.3 | 0.50 | 4 | 8 | 6 | 6 | 4 | 增加隐空间宽度 |
| C6.4 | 0.50 | 4 | 12 | 8 | 8 | 6 | 继续增加 query 数与 top-k |
| C6.5 | 0.50 | 8 | 12 | 8 | 8 | 6 | 增加 attention heads |

训练目录：

- `runs/detect/train_fp_C65_y8nbackbone_seed0_cls01/`

`C6.3/C6.4` 已停止，因为当前结果低于 `C6.5`，不采用其配置。`C6.5` 按 epoch180/181 指标已超过 `A1-y8n-backbone`，P4/P5 融合方式固定为 `C6.5` 配置。

## 六、C6.5 层级消融结果

为验证 `DMG@P2 + CFRA@P3 + RS-SQF@P4/P5` 是层级匹配收益，而不是单纯模块堆叠，沿用 YOLOv12n 骨干的 6 个消融思路。统一训练口径仍使用第二节设置，指标取各自 `results.csv` 最后一行。

消融结论：

1. 三个同构铺满实验均低于 `C6.5`，说明不是把某个融合模块铺到 P2-P5 就能带来收益。
2. `Perm-B/Perm-C` 明显低于 `C6.5`，支持 `RS-SQF` 不适合前移到 P3，`CFRA/DMG` 也不适合高层 P4/P5。
3. `Perm-A` 接近 `C6.5`，说明 P2/P3 的 `DMG` 与 `CFRA` 存在一定互换容忍度；但 `C6.5` overall 更高且 person 更好，仍作为固定配置。

### 6.1 P2345 同构消融

| 消融 | 设置 | checkpoint | P | R | mAP50 | mAP50-95 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| y8n-P2345-DMG | `DMGInit8d@P2/P3/P4/P5` | csv epoch200 | 0.91490 | 0.89685 | 0.93376 | 0.58725 | 明显低于 C6.5 |
| y8n-P2345-CFRA | `CFRA@P2/P3/P4/P5` | csv epoch200 | 0.91699 | 0.90680 | 0.94102 | 0.59012 | 低于 C6.5，fire 接近 |
| y8n-P2345-RS-SQF | `RS-SQF@P2/P3/P4/P5` | csv epoch200 | 0.92391 | 0.89255 | 0.93717 | 0.58949 | 低于 C6.5 |

### 6.2 层级错位消融

| 消融 | 设置 | checkpoint | P | R | mAP50 | mAP50-95 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| y8n-Perm-A | `CFRA@P2 + DMG@P3 + RS-SQF@P4/P5` | csv epoch193 | 0.92360 | 0.89173 | 0.93709 | 0.59406 | 接近 C6.5，但未超过 |
| y8n-Perm-B | `DMG@P2 + RS-SQF@P3 + CFRA@P4/P5` | csv epoch200 | 0.91041 | 0.89730 | 0.93629 | 0.59057 | 低于 C6.5 |
| y8n-Perm-C | `CFRA@P2 + RS-SQF@P3 + DMG@P4/P5` | csv epoch200 | 0.92867 | 0.89119 | 0.93917 | 0.58906 | 低于 C6.5 |

### 6.3 消融完整类别指标

| 消融 | checkpoint | 类别 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| y8n-P2345-DMG | csv epoch200 | all | 0.91490 | 0.89685 | 0.93376 | 0.58725 |
| y8n-P2345-DMG | csv epoch200 | fire | 0.92308 | 0.88571 | 0.93543 | 0.58738 |
| y8n-P2345-DMG | csv epoch200 | person | 0.90671 | 0.90799 | 0.93209 | 0.58712 |
| y8n-P2345-CFRA | csv epoch200 | all | 0.91699 | 0.90680 | 0.94102 | 0.59012 |
| y8n-P2345-CFRA | csv epoch200 | fire | 0.93127 | 0.90562 | 0.94769 | 0.59872 |
| y8n-P2345-CFRA | csv epoch200 | person | 0.90271 | 0.90799 | 0.93434 | 0.58152 |
| y8n-P2345-RS-SQF | csv epoch200 | all | 0.92391 | 0.89255 | 0.93717 | 0.58949 |
| y8n-P2345-RS-SQF | csv epoch200 | fire | 0.93152 | 0.88474 | 0.93645 | 0.59286 |
| y8n-P2345-RS-SQF | csv epoch200 | person | 0.91631 | 0.90035 | 0.93789 | 0.58612 |
| y8n-Perm-A | csv epoch193 | all | 0.92360 | 0.89173 | 0.93709 | 0.59406 |
| y8n-Perm-A | csv epoch193 | fire | 0.93710 | 0.89268 | 0.94107 | 0.60157 |
| y8n-Perm-A | csv epoch193 | person | 0.91011 | 0.89078 | 0.93312 | 0.58654 |
| y8n-Perm-B | csv epoch200 | all | 0.91041 | 0.89730 | 0.93629 | 0.59057 |
| y8n-Perm-B | csv epoch200 | fire | 0.91984 | 0.89474 | 0.93838 | 0.59248 |
| y8n-Perm-B | csv epoch200 | person | 0.90097 | 0.89986 | 0.93419 | 0.58866 |
| y8n-Perm-C | csv epoch200 | all | 0.92867 | 0.89119 | 0.93917 | 0.58906 |
| y8n-Perm-C | csv epoch200 | fire | 0.94344 | 0.88503 | 0.94181 | 0.59174 |
| y8n-Perm-C | csv epoch200 | person | 0.91391 | 0.89734 | 0.93653 | 0.58638 |

## 七、完整指标数据

### 7.1 单模态参考基线

| 实验 | checkpoint | 类别 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| RGB teacher | last.pt | all | 0.89009 | 0.80784 | 0.86835 | 0.50820 |
| RGB teacher | last.pt | fire | 0.89343 | 0.79624 | 0.86190 | 0.48027 |
| RGB teacher | last.pt | person | 0.88675 | 0.81944 | 0.87481 | 0.53613 |
| IR teacher | last.pt | all | 0.89260 | 0.80500 | 0.86965 | 0.53840 |
| IR teacher | last.pt | fire | 0.89028 | 0.81357 | 0.87679 | 0.54279 |
| IR teacher | last.pt | person | 0.89492 | 0.79643 | 0.86251 | 0.53400 |

### 7.2 YOLOv8 Backbone 双流实验

| 实验 | checkpoint | 类别 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| F0-y8n-backbone | csv epoch200 | all | 0.91756 | 0.89105 | 0.92927 | 0.57586 |
| F0-y8n-backbone | csv epoch200 | fire | 0.92871 | 0.89768 | 0.93811 | 0.58526 |
| F0-y8n-backbone | csv epoch200 | person | 0.90641 | 0.88442 | 0.92043 | 0.56646 |
| F2-y8n-backbone | csv epoch200 | all | 0.91269 | 0.90674 | 0.93679 | 0.58679 |
| F2-y8n-backbone | csv epoch200 | fire | 0.92710 | 0.90491 | 0.93926 | 0.59120 |
| F2-y8n-backbone | csv epoch200 | person | 0.89828 | 0.90857 | 0.93432 | 0.58238 |
| D2.1-y8n-backbone | csv epoch200 | all | 0.92339 | 0.89703 | 0.93810 | 0.59015 |
| D2.1-y8n-backbone | csv epoch200 | fire | 0.93728 | 0.89297 | 0.94270 | 0.59314 |
| D2.1-y8n-backbone | csv epoch200 | person | 0.90950 | 0.90109 | 0.93350 | 0.58716 |
| A1-y8n-backbone | csv epoch200 | all | 0.92606 | 0.90036 | 0.93936 | 0.59292 |
| A1-y8n-backbone | csv epoch200 | fire | 0.93735 | 0.89503 | 0.94382 | 0.59892 |
| A1-y8n-backbone | csv epoch200 | person | 0.91478 | 0.90569 | 0.93491 | 0.58691 |
| A2-y8n-backbone | csv epoch200 | all | 0.91830 | 0.90084 | 0.93917 | 0.59045 |
| A2-y8n-backbone | csv epoch200 | fire | 0.93128 | 0.89885 | 0.94378 | 0.59353 |
| A2-y8n-backbone | csv epoch200 | person | 0.90533 | 0.90282 | 0.93457 | 0.58737 |
| C1 | csv epoch200 | all | 0.92338 | 0.90066 | 0.93961 | 0.59094 |
| C1 | csv epoch200 | fire | 0.93200 | 0.89415 | 0.94227 | 0.59953 |
| C1 | csv epoch200 | person | 0.91476 | 0.90718 | 0.93694 | 0.58235 |
| C2 | latest csv epoch166 | all | 0.92192 | 0.90342 | 0.93677 | 0.58994 |
| C2 | latest csv epoch166 | fire | 0.93372 | 0.89884 | 0.93860 | 0.59184 |
| C2 | latest csv epoch166 | person | 0.91012 | 0.90799 | 0.93494 | 0.58805 |
| B2-y8n-backbone | latest csv epoch152 | all | 0.91848 | 0.89620 | 0.93703 | 0.58830 |
| B2-y8n-backbone | latest csv epoch152 | fire | 0.92805 | 0.89591 | 0.93976 | 0.58942 |
| B2-y8n-backbone | latest csv epoch152 | person | 0.90890 | 0.89649 | 0.93430 | 0.58719 |
| C3 | latest csv epoch131 | all | 0.92082 | 0.90229 | 0.93753 | 0.58466 |
| C3 | latest csv epoch131 | fire | 0.93515 | 0.90709 | 0.94439 | 0.58919 |
| C3 | latest csv epoch131 | person | 0.90649 | 0.89749 | 0.93067 | 0.58014 |
| C4 | latest csv epoch135 | all | 0.92363 | 0.90015 | 0.93866 | 0.58593 |
| C4 | latest csv epoch135 | fire | 0.93959 | 0.90094 | 0.94491 | 0.59443 |
| C4 | latest csv epoch135 | person | 0.90767 | 0.89937 | 0.93242 | 0.57743 |
| C5 | latest csv epoch103 | all | 0.92187 | 0.89855 | 0.93443 | 0.58359 |
| C5 | latest csv epoch103 | fire | 0.92975 | 0.89887 | 0.93911 | 0.58720 |
| C5 | latest csv epoch103 | person | 0.91399 | 0.89824 | 0.92974 | 0.57997 |
| C6.1-y8n-backbone | csv epoch200 | all | 0.92277 | 0.89753 | 0.93527 | 0.58826 |
| C6.1-y8n-backbone | csv epoch200 | fire | 0.93467 | 0.89591 | 0.93935 | 0.59358 |
| C6.1-y8n-backbone | csv epoch200 | person | 0.91087 | 0.89915 | 0.93118 | 0.58294 |
| C6.2-y8n-backbone | csv epoch200 | all | 0.91889 | 0.89395 | 0.93584 | 0.59125 |
| C6.2-y8n-backbone | csv epoch200 | fire | 0.92556 | 0.89198 | 0.93961 | 0.59723 |
| C6.2-y8n-backbone | csv epoch200 | person | 0.91221 | 0.89592 | 0.93207 | 0.58527 |
| C6.3-y8n-backbone | latest csv epoch153 | all | 0.91723 | 0.90626 | 0.93922 | 0.59023 |
| C6.3-y8n-backbone | latest csv epoch153 | fire | 0.93099 | 0.90830 | 0.94259 | 0.59209 |
| C6.3-y8n-backbone | latest csv epoch153 | person | 0.90347 | 0.90421 | 0.93584 | 0.58838 |
| C6.4-y8n-backbone | latest csv epoch153 | all | 0.91652 | 0.90303 | 0.93877 | 0.58707 |
| C6.4-y8n-backbone | latest csv epoch153 | fire | 0.92656 | 0.90268 | 0.94054 | 0.58850 |
| C6.4-y8n-backbone | latest csv epoch153 | person | 0.90647 | 0.90339 | 0.93700 | 0.58564 |
| C6.5-y8n-backbone | csv epoch180/181 | all | 0.93357 | 0.88779 | 0.94046 | 0.59452 |
| C6.5-y8n-backbone | csv epoch180/181 | fire | 0.94452 | 0.88598 | 0.94322 | 0.59961 |
| C6.5-y8n-backbone | csv epoch180/181 | person | 0.92263 | 0.88959 | 0.93770 | 0.58943 |

## 八、排序摘要

主线与结构探索实验按 overall `mAP50-95` 排序：

| 排名 | 实验 | mAP50-95 | 备注 |
| ---: | --- | ---: | --- |
| 1 | C6.5-y8n-backbone | 0.59452 | epoch180/181，已超过 A1 |
| 2 | A1-y8n-backbone | 0.59292 | 当前最优完成点 |
| 3 | C6.2-y8n-backbone | 0.59125 | RS-SQF 适中参数版 |
| 4 | C1 | 0.59094 | CSPA 接近 A1，但未超越 |
| 5 | A2-y8n-backbone | 0.59045 | P4 CFRA 不增益 |
| 6 | C6.3-y8n-backbone | 0.59023 | 已停止，不如 C6.5 |
| 7 | D2.1-y8n-backbone | 0.59015 | P2 DMG 有效 |
| 8 | C2 | 0.58994 | 仅 P4 CSPA 不如 P4/P5 |
| 9 | B2-y8n-backbone | 0.58830 | pcrossA2C2f 直接迁移失败 |
| 10 | C6.1-y8n-backbone | 0.58826 | 最小 RS-SQF 不足 |
| 11 | C6.4-y8n-backbone | 0.58707 | 已停止，不如 C6.5 |
| 12 | F2-y8n-backbone | 0.58679 | 普通 P2 + P3 aux 有效 |
| 13 | C4 | 0.58593 | CSPA 增容无效 |
| 14 | C3 | 0.58466 | CSPA 增容无效 |
| 15 | C5 | 0.58359 | CSPA 堆叠无效 |
| 16 | F0-y8n-backbone | 0.57586 | 主链起点 |

层级消融按 overall `mAP50-95` 排序：

| 排名 | 实验 | mAP50-95 | 备注 |
| ---: | --- | ---: | --- |
| 1 | C6.5-y8n-backbone | 0.59452 | 固定配置，epoch180/181 |
| 2 | y8n-Perm-A | 0.59406 | 接近 C6.5，P2/P3 互换容忍度较高 |
| 3 | y8n-Perm-B | 0.59057 | RS-SQF 前移到 P3 后低于 C6.5 |
| 4 | y8n-P2345-CFRA | 0.59012 | CFRA 铺满低于 C6.5 |
| 5 | y8n-P2345-RS-SQF | 0.58949 | RS-SQF 铺满低于 C6.5 |
| 6 | y8n-Perm-C | 0.58906 | 三机制错位低于 C6.5 |
| 7 | y8n-P2345-DMG | 0.58725 | DMG 铺满最低 |

## 九、后续动作

1. `C6.5` 按 epoch180/181 指标作为 YOLOv8 支线当前参考点。
2. `C6.3/C6.4` 不再继续训练；P4/P5 融合固定为 `C6.5` 配置。
3. 论文论证中使用同构消融失败与 `Perm-B/Perm-C` 退化作为层级匹配证据；同时说明 `Perm-A` 接近 C6.5，P2/P3 模块边界不是绝对硬约束。
