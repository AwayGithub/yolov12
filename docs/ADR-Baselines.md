# ADR：Baselines 结果记录

**状态：** 进行中  
**创建日期：** 2026-07-06  
**最后更新：** 2026-07-19

## 一、口径

本文件集中记录 RGBT-3M fire/person 二分类任务中的单模态基线、双模态外部方法、模型复杂度与 GPU 推理延时，避免与主创新链 ADR 混写。

除单独说明外，本文复现/对比实验采用：

| 项目 | 设置 |
| --- | --- |
| 数据集 | RGBT-3M fire/person 二分类 |
| 输入尺寸 | `[480, 640]` |
| batch | 16 |
| optimizer | SGD |
| lr0 / cls | `0.01 / 0.1` |
| 主要指标 | P, R, mAP50, mAP50-95 |

指标来源说明：

- `last.pt` / `best.pt` 独立验证：重新加载 checkpoint 验证得到。
- `results.csv`：训练期间验证记录，可能没有分类别细项或不是独立复测。
- 不同来源不可混成同一种口径；表格中均标注来源。
- 历史 run 的 `args.yaml` 可能保留 Ultralytics 默认增强字段，但 RGBT-3M 双流实际走自定义 6 通道 transform，当前未真正接入 Mosaic、RandomPerspective、HSV、CopyPaste 或 MixUp。因此“无数据增强”以实际数据管线为准。

## 二、原论文信息

评级口径：会议按 CCF 推荐国际学术会议目录记录；期刊按期刊官方 JCR 分区记录。MDPI 期刊不属于 CCF 会议口径，故不写 CCF 等级。

| 本文复现方法 | 原论文/方法来源 | 发表会议/刊物 | 评级 | 发表时间 | 来源 |
| --- | --- | --- | --- | --- | --- |
| CALNet | He et al., "Multispectral Object Detection via Cross-Modal Conflict-Aware Learning" | ACM MM 2023, Proceedings of the 31st ACM International Conference on Multimedia | CCF A（计算机图形学与多媒体，ACM MM） | 2023；会议日期 2023-10-29 至 2023-11-03 | SIGMM TOC / CALNet README |
| CP-YOLOv11-MF | Zhang et al., "A UAV-Based Multi-Scenario RGB-Thermal Dataset and Fusion Model for Enhanced Forest Fire Detection" | Remote Sensing 2025, 17(15), 2593 | JCR Q1（Remote Sensing；Geosciences, Multidisciplinary） | 2025-07-25 | MDPI, DOI: `10.3390/rs17152593` |
| MCDet | Xu et al., "MCDet: Target-Aware Fusion for RGB-T Fire Detection" | Forests 2025, 16(7), 1088 | JCR Q2（Forestry）/ CiteScore Q1（Forestry） | 2025-06-30 | MDPI, DOI: `10.3390/f16071088` |
| AFDNet | Chen et al., "Asymmetric Frequency-Decoupled Network for Robust Visible-Infrared Fire Detection" | Remote Sensing 2026, 18(11), 1777 | JCR Q1（Remote Sensing；Geosciences, Multidisciplinary） | 2026-06-01 | MDPI, DOI: `10.3390/rs18111777` |
| M2D-LIF | Zhao et al., "Rethinking Multi-modal Object Detection from the Perspective of Mono-Modality Feature Learning" | ICCV 2025, Proceedings of the IEEE/CVF International Conference on Computer Vision | CCF A（人工智能，ICCV） | 2025-10；CVF 页码 6364-6373 | CVF Open Access / M2D-LIF README |

## 三、核心结果

### 3.1 双模态 N/nano 公平对比

`YOLOv8n-C65` 是本文模型参照项，完整实验链条见 `docs/ADR-yolov8.md`。本表用于论文中与外部双模态 N/nano baselines 对齐。

| 模型 | 尺度/结构 | 指标来源 | P | R | mAP50 | mAP50-95 | Params | FLOPs | mean latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n-C65 | P2-DMG + P3-CFRA + P4/P5-RS-SQF，P2-P5 检测 | `ADR-yolov8` csv epoch180/181 + profiling | 0.93357 | 0.88779 | 0.94046 | 0.59452 | 5.22M | 13.49G | 21.19 ms |
| M2D-LIF nano | YOLOv8n/nano 双流学生 | `last.pt` 独立验证 | 0.90598 | 0.87115 | 0.90781 | 0.57304 | 4.06M | 13.01G | 9.82 ms |
| CALNet-nano | 双流 OBB，报告 HBB 指标 | `best.pt` 独立验证 | 0.865 | 0.824 | 0.863 | 0.464 | 12.99M | 12.93G | 51.09 ms |
| CP-YOLOv11-MF | YOLOv11-n 双流复现 | `last.pt` 独立验证 | 0.91452 | 0.86491 | 0.91606 | 0.56538 | 7.80M | 29.09G | 17.02 ms |
| AFDNet | YOLOv11-n 双流复现 | `results.csv` epoch200 | 0.91190 | 0.89633 | 0.92505 | 0.55631 | 5.36M | 8.04G | 15.24 ms |
| MCDet | YOLOv5n 双流复现，MRCF + CGAN | `results.csv` epoch200 | 0.92165 | 0.89231 | 0.92877 | 0.56336 | 10.28M | 14.83G | 29.30 ms |

备注：

- `YOLOv8n-C65` 的检测指标摘自 `docs/ADR-yolov8.md`，复杂度与延时摘自本 ADR 的 profiling 表。
- CALNet 使用 OBB head，本表报告其验证脚本输出的 HBB 指标。
- AFDNet / MCDet 的分类别指标来自训练期 `results.csv`，还不是独立重验 `last.pt`。
- CP-YOLOv11-MF 的 mAP50-95 可能受简化 DFL 解码影响；mAP50 和 P/R 仍可作为复现对比。

### 3.2 大尺度补充基线

| 模型 | 尺度/说明 | 来源 | epoch | P | R | mAP50 | mAP50-95 | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M2D-LIF M | M 尺度双流学生，容量更大 | `results.csv` last row | 151 | 0.93051 | 0.89502 | 0.92712 | 0.60570 | early stopping；不纳入 N/nano 公平对比 |

### 3.3 单模态基线

| 方法 | 模态 | 尺度/结构 | 来源 | P | R | mAP50 | mAP50-95 | Params | FLOPs |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | RGB | n | `last.pt` 独立验证 | 0.89009 | 0.80784 | 0.86835 | 0.50820 | - | - |
| YOLOv8n | IR | n | `last.pt` 独立验证 | 0.89260 | 0.80500 | 0.86965 | 0.53840 | - | - |
| YOLOv12n | RGB | n | `last.pt` 独立验证 | 0.89031 | 0.85234 | 0.88899 | 0.49201 | - | - |
| YOLOv12n | IR | n | `last.pt` 独立验证 | 0.88929 | 0.83556 | 0.88825 | 0.50485 | - | - |
| YOLOv11n | RGB | n | `results.csv` epoch200 | 0.89885 | 0.84221 | 0.89737 | 0.50971 | - | - |
| YOLOv11n | IR | n | `results.csv` epoch200 | 0.89931 | 0.83528 | 0.90625 | 0.54026 | - | - |
| RT-DETRv2 | RGB | S，PResNet-18 + HybridEncoder + 3-layer decoder | `last.pth` 独立验证，120ep | 0.92080 | 0.91162 | 0.92013 | 0.53593 | 20.08M | 46.44G |
| RT-DETRv2 | IR | S，PResNet-18 + HybridEncoder + 3-layer decoder | `last.pth` 独立验证，120ep | 0.91647 | 0.91628 | 0.92665 | 0.56813 | 20.08M | 46.44G |

## 四、实验证据索引

### 4.1 双模态方法

| 方法 | checkpoint / run | data | 关键设置 | 备注 |
| --- | --- | --- | --- | --- |
| M2D-LIF nano | `baselines/M2D-LIF/runs/m2d_lif/m2dlif_student_fp_dual/weights/last.pt` | `baselines/M2D-LIF/data/RGBT3M-fire-person-dual.yaml` | YOLOv8n/nano；验证目录 `baselines/M2D-LIF/val_fire_person/val_m2dlif_student_last_v1/`；验证时间 2026-07-06 | 记录 `last.pt`；历史 run 实际使用 `model_yaml/yolov8n-LIF-fire-person.yaml` |
| M2D-LIF M | `baselines/M2D-LIF/runs/m2d_lif/m2dlif_student_fp_dual_m_b16_noaug/` | `data/RGBT3M-fire-person-dual.yaml` | `model=./model_yaml/yolov8-LIF.yaml`；`device=0,1`；`patience=50` | `results.csv` 只有 overall；epoch151 early stopping |
| CALNet-nano | `baselines/CALNet-Dronevehicle/runs/train/calnet_n_hbb_b16_2gpu_noaug/weights/best.pt` | `baselines/CALNet-Dronevehicle/data/RGBT-3M-fire-person.yaml` | 从零初始化；DDP 2 GPU；原始 CALNet `hyp.finetune_DroneVehicle.yaml`；验证 `conf=0.01, iou=0.4`；验证时间 2026-07-15 | 为支持双输入验证，对 `val.py`、`models/common.py`、`utils/datasets.py` 做了最小适配 |
| CP-YOLOv11-MF | `baselines/CP-YOLOv11-MF/runs/detect/train_cp_yolo11mf/last.pt` | `/data/xwh/dataset/RGBT-3M/RGBT-3M` | YOLOv11-n；CPCA + PPAS；验证 `conf=0.001, iou=0.6`；验证时间 2026-07-14 | 原论文用 YOLOv11-s，本文为公平比较改 n；代码复现自论文描述，非官方开源代码 |
| AFDNet | `baselines/AFDNet/train_fp_AFDNet_y11n_seed0_cls01_noaug/weights/last.pt` | `ultralytics/cfg/datasets/RGBT-3M-dual-fire-person-local.yaml` | YOLOv11-n；P3/P4/P5 frequency-decoupled fusion；`deterministic=False` | 指标来自 `results.csv`；输出 `[1, 6, 6300]` |
| MCDet | `runs/detect/train_fp_MCDet_yolov5n_seed0_cls01_noaug_noamp/weights/last.pt` | `ultralytics/cfg/datasets/RGBT-3M-dual-fire-person-local.yaml` | YOLOv5n；P3/P4/P5 MRCF；neck concat 后 CGAN；WIoU；`amp=False, workers=4` | 指标来自 `results.csv`；输出 `[1, 6, 6300]` |

### 4.2 双模态分类别指标

| 方法 | 类别 | P | R | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| M2D-LIF nano | all | 0.90598 | 0.87115 | 0.90781 | 0.57304 |
| M2D-LIF nano | fire | 0.90937 | 0.87915 | 0.91518 | 0.57767 |
| M2D-LIF nano | person | 0.90259 | 0.86314 | 0.90044 | 0.56841 |
| M2D-LIF M | all | 0.93051 | 0.89502 | 0.92712 | 0.60570 |
| CALNet-nano | all | 0.865 | 0.824 | 0.863 | 0.464 |
| CALNet-nano | fire | 0.885 | 0.826 | 0.870 | 0.477 |
| CALNet-nano | person | 0.844 | 0.822 | 0.856 | 0.450 |
| CP-YOLOv11-MF | all | 0.91452 | 0.86491 | 0.91606 | 0.56538 |
| CP-YOLOv11-MF | fire | 0.93353 | 0.85975 | 0.91607 | 0.56217 |
| CP-YOLOv11-MF | person | 0.89592 | 0.87073 | 0.91606 | 0.56858 |
| AFDNet | all | 0.91190 | 0.89633 | 0.92505 | 0.55631 |
| AFDNet | fire | 0.92411 | 0.90032 | 0.93485 | 0.56948 |
| AFDNet | person | 0.89970 | 0.89234 | 0.91525 | 0.54313 |
| MCDet | all | 0.92165 | 0.89231 | 0.92877 | 0.56336 |
| MCDet | fire | 0.93618 | 0.89719 | 0.93472 | 0.57388 |
| MCDet | person | 0.90713 | 0.88743 | 0.92283 | 0.55284 |

### 4.3 单模态分类别指标

| 模型 | 模态 | 类别 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n | RGB | all | 0.89009 | 0.80784 | 0.86835 | 0.50820 |
| YOLOv8n | RGB | fire | 0.89343 | 0.79624 | 0.86190 | 0.48027 |
| YOLOv8n | RGB | person | 0.88675 | 0.81944 | 0.87481 | 0.53613 |
| YOLOv8n | IR | all | 0.89260 | 0.80500 | 0.86965 | 0.53840 |
| YOLOv8n | IR | fire | 0.89028 | 0.81357 | 0.87679 | 0.54279 |
| YOLOv8n | IR | person | 0.89492 | 0.79643 | 0.86251 | 0.53400 |
| YOLOv12n | RGB | all | 0.89031 | 0.85234 | 0.88899 | 0.49201 |
| YOLOv12n | RGB | fire | 0.90196 | 0.83859 | 0.88362 | 0.47660 |
| YOLOv12n | RGB | person | 0.87866 | 0.86609 | 0.89436 | 0.50742 |
| YOLOv12n | IR | all | 0.88929 | 0.83556 | 0.88825 | 0.50485 |
| YOLOv12n | IR | fire | 0.90031 | 0.82476 | 0.87976 | 0.51177 |
| YOLOv12n | IR | person | 0.87827 | 0.84637 | 0.89675 | 0.49793 |
| YOLOv11n | RGB | all | 0.89885 | 0.84221 | 0.89737 | 0.50971 |
| YOLOv11n | RGB | fire | 0.90000 | 0.82358 | 0.87533 | 0.47276 |
| YOLOv11n | RGB | person | 0.89769 | 0.86084 | 0.91941 | 0.54667 |
| YOLOv11n | IR | all | 0.89931 | 0.83528 | 0.90625 | 0.54026 |
| YOLOv11n | IR | fire | 0.90878 | 0.80800 | 0.88797 | 0.53679 |
| YOLOv11n | IR | person | 0.88984 | 0.86256 | 0.92454 | 0.54373 |
| RT-DETRv2-S | RGB | all | 0.92080 | 0.91162 | 0.92013 | 0.53593 |
| RT-DETRv2-S | RGB | fire | 0.93618 | 0.90139 | 0.91866 | 0.51201 |
| RT-DETRv2-S | RGB | person | 0.90542 | 0.92184 | 0.92159 | 0.55984 |
| RT-DETRv2-S | IR | all | 0.91647 | 0.91628 | 0.92665 | 0.56813 |
| RT-DETRv2-S | IR | fire | 0.92295 | 0.90865 | 0.91987 | 0.57392 |
| RT-DETRv2-S | IR | person | 0.91000 | 0.92391 | 0.93342 | 0.56234 |

### 4.4 单模态 RT-DETR 证据

| 模型 | 模态 | checkpoint / 指标文件 | 配置 | 备注 |
| --- | --- | --- | --- | --- |
| RT-DETRv2-S | RGB | `baselines/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_s_rgb_fire_person_noaug_480x640/last.pth`；`last_ultralytics_metrics.json` | `configs/rgbt3m/rtdetrv2_s_rgb_fire_person_480x640.yml` | 官方 PyTorch RT-DETRv2-S；`epoches=120`，正常跑满 0-119；P/R/mAP 为独立验证重算 |
| RT-DETRv2-S | IR | `baselines/RT-DETR/rtdetrv2_pytorch/output/rtdetrv2_s_ir_fire_person_noaug_480x640/last.pth`；`last_ultralytics_metrics.json` | `configs/rgbt3m/rtdetrv2_s_ir_fire_person_480x640.yml` | 同上；参数量 `20.0843M`，FLOPs `46.4429G`，输入 `1x3x480x640` |

## 五、复杂度与延时

### 5.1 结果

| 模型 | checkpoint | Params | FLOPs | mean latency | median latency | p10 latency | p90 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2D-LIF nano | `baselines/M2D-LIF/runs/m2d_lif/m2dlif_student_fp_dual/weights/last.pt` | 4.06M | 13.01G | 9.82 ms | 9.81 ms | 9.54 ms | 10.04 ms |
| YOLOv8n-C65 | `runs/detect/train_fp_C65_y8nbackbone_seed0_cls01/weights/last.pt` | 5.22M | 13.49G | 21.19 ms | 21.02 ms | 20.23 ms | 22.01 ms |
| AFDNet | `baselines/AFDNet/train_fp_AFDNet_y11n_seed0_cls01_noaug/weights/last.pt` | 5.36M | 8.04G | 15.24 ms | 14.50 ms | 14.34 ms | 17.00 ms |
| MCDet | `runs/detect/train_fp_MCDet_yolov5n_seed0_cls01_noaug_noamp/weights/last.pt` | 10.28M | 14.83G | 29.30 ms | 28.88 ms | 27.57 ms | 31.65 ms |
| CP-YOLOv11-MF | `baselines/CP-YOLOv11-MF/runs/detect/train_cp_yolo11mf/last.pt` | 7.80M | 29.09G | 17.02 ms | 16.90 ms | 16.18 ms | 17.58 ms |
| CALNet-nano | `baselines/CALNet-Dronevehicle/runs/train/calnet_n_hbb_b16_2gpu_noaug/weights/best.pt` | 12.99M | 12.93G | 51.09 ms | 51.00 ms | 50.70 ms | 51.35 ms |

### 5.2 统计口径

| 项目 | 设置 |
| --- | --- |
| 设备 | NVIDIA GeForce RTX 2080 Ti |
| dtype | FP32 |
| 输入 | 双流模型使用等价 `1x6x480x640`；CALNet 使用 `RGB 1x3x480x640 + IR 1x3x480x640` |
| FLOPs | `torch.profiler.profile(with_flops=True)` |
| latency | CUDA Event raw forward，warmup 50 次，计时 200 次 |
| 不包含 | 图像读取、预处理、NMS、绘图、dataloader |

注意：

- CALNet 采用其验证脚本一致的 fused 推理模型，参数量为融合后模型参数。
- YOLOv8n-C65 有 P2 检测输出，输出网格数量为 25,500；M2D-LIF、AFDNet、MCDet、CP-YOLOv11-MF 为 P3/P4/P5 输出，输出网格数量为 6,300。
- FLOPs 只统计 `torch.profiler` 可识别算子；不同模型含有自定义模块时，FLOPs 与实际 latency 不必严格单调一致。
