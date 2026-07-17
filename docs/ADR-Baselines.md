# ADR：Baselines 结果记录

**状态：** 进行中  
**创建日期：** 2026-07-06  
**最后更新：** 2026-07-17

## 一、目的与口径

本文件集中记录 fire/person 二分类任务中的单模态基线、双模态外部方法、模型复杂度与 GPU 推理延时，避免与主创新链 ADR 混写。

### 1.1 通用设置

除单独说明外，本文件中的本文复现/对比实验采用：

| 项目 | 设置 |
| --- | --- |
| 数据集 | RGBT-3M fire/person 二分类 |
| 输入尺寸 | `[480, 640]` |
| batch | 16 |
| optimizer | SGD |
| lr0 / cls | `0.01 / 0.1` |
| 主要指标 | P, R, mAP50, mAP50-95 |

### 1.2 数据增强说明

文档中历史 run 的 `args.yaml` 可能保留 Ultralytics 默认增强参数，例如 `mosaic=1.0`、`fliplr=0.5`、`copy_paste=0.1`。但对 RGBT-3M 双流数据，实际训练走 `RGBT3MDataset -> FLAME2Dataset` 的自定义 6 通道 transform，目前没有真正接入 Mosaic、RandomPerspective、HSV、CopyPaste 或 MixUp。因此本文件中“无数据增强”以实际数据管线为准，而不是只看 `args.yaml` 字段。

### 1.3 指标来源说明

- `last.pt` / `best.pt` 独立验证：用对应验证脚本重新加载 checkpoint 得到。
- `results.csv`：训练期间验证记录，可能没有分类别细项或不是独立复测。
- 不同来源不可直接混成同一种口径；表格中均标注来源。

## 二、总览

### 2.1 双模态方法 overall 指标

| 方法 | 尺度/说明 | checkpoint/来源 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| M2D-LIF nano | YOLOv8n/nano 双流学生 | `last.pt` 独立验证 | 0.90598 | 0.87115 | 0.90781 | 0.57304 |
| M2D-LIF M | M 尺度双流学生，容量更大 | `results.csv` epoch151 | 0.93051 | 0.89502 | 0.92712 | 0.60570 |
| CALNet-nano | 双流 OBB，报告 HBB 指标 | `best.pt` 独立验证 | 0.865 | 0.824 | 0.863 | 0.464 |
| CP-YOLOv11-MF | YOLOv11-n 双流复现 | `last.pt` 独立验证 | 0.91452 | 0.86491 | 0.91606 | 0.56538 |
| AFDNet | YOLOv11-n 尺度双流复现 | `results.csv` epoch200 | 0.91190 | 0.89633 | 0.92505 | 0.55631 |
| MCDet | YOLOv5n 尺度双流复现，MRCF + CGAN | `results.csv` epoch200 | 0.92165 | 0.89231 | 0.92877 | 0.56336 |

### 2.2 单模态方法 overall 指标

| 方法 | 模态 | 来源 | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n | RGB | `last.pt` 独立验证 | 0.89009 | 0.80784 | 0.86835 | 0.50820 |
| YOLOv8n | IR | `last.pt` 独立验证 | 0.89260 | 0.80500 | 0.86965 | 0.53840 |
| YOLOv12n | RGB | `last.pt` 独立验证 | 0.89031 | 0.85234 | 0.88899 | 0.49201 |
| YOLOv12n | IR | `last.pt` 独立验证 | 0.88929 | 0.83556 | 0.88825 | 0.50485 |
| YOLOv11n | RGB | `results.csv` epoch200 | 0.89885 | 0.84221 | 0.89737 | 0.50971 |
| YOLOv11n | IR | `results.csv` epoch200 | 0.89931 | 0.83528 | 0.90625 | 0.54026 |

### 2.3 模型复杂度与 GPU 延时

统一口径见第六节。

| 模型 | checkpoint | Params | FLOPs | mean latency | median latency | p10 latency | p90 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2D-LIF nano | `baselines/M2D-LIF/runs/m2d_lif/m2dlif_student_fp_dual/weights/last.pt` | 4.06M | 13.01G | 9.82 ms | 9.81 ms | 9.54 ms | 10.04 ms |
| YOLOv8n-C65 | `runs/detect/train_fp_C65_y8nbackbone_seed0_cls01/weights/last.pt` | 5.22M | 13.49G | 21.19 ms | 21.02 ms | 20.23 ms | 22.01 ms |
| AFDNet | `baselines/AFDNet/train_fp_AFDNet_y11n_seed0_cls01_noaug/weights/last.pt` | 5.36M | 8.04G | 15.24 ms | 14.50 ms | 14.34 ms | 17.00 ms |
| MCDet | `runs/detect/train_fp_MCDet_yolov5n_seed0_cls01_noaug_noamp/weights/last.pt` | 10.28M | 14.83G | 29.30 ms | 28.88 ms | 27.57 ms | 31.65 ms |
| CP-YOLOv11-MF | `baselines/CP-YOLOv11-MF/runs/detect/train_cp_yolo11mf/last.pt` | 7.80M | 29.09G | 17.02 ms | 16.90 ms | 16.18 ms | 17.58 ms |
| CALNet-nano | `baselines/CALNet-Dronevehicle/runs/train/calnet_n_hbb_b16_2gpu_noaug/weights/best.pt` | 12.99M | 12.93G | 51.09 ms | 51.00 ms | 50.70 ms | 51.35 ms |

## 三、单模态基线

### 3.1 实验定义

- 任务：RGBT-3M fire/person 二分类单模态检测。
- YOLOv8n RGB/IR：M2D-LIF `teacherTraining` 的 nano 教师；来源 `docs/ADR-yolov8.md` §7.1。
- YOLOv12n RGB/IR：原始单分支 YOLOv12n；来源 `docs/ADR-005-fire-person-binary-scope.md` §6.3。
- YOLOv11n RGB/IR：原始单分支 YOLOv11n；`runs/detect/train_fp_y11n_{rgb,ir}_seed0_cls01`。
- 口径：YOLOv8n / YOLOv12n 为 `last.pt` 独立验证；YOLOv11n 为训练目录 `results.csv` 最后一行。

### 3.2 分类别指标

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

## 四、双模态外部基线

### 4.1 M2D-LIF nano 双流学生

实验信息：

| 项目 | 设置 |
| --- | --- |
| 方法 | `M2D-LIF` 双流学生模型，YOLOv8n/nano 尺度 |
| checkpoint | `baselines/M2D-LIF/runs/m2d_lif/m2dlif_student_fp_dual/weights/last.pt` |
| data | `baselines/M2D-LIF/data/RGBT3M-fire-person-dual.yaml` |
| 验证目录 | `baselines/M2D-LIF/val_fire_person/val_m2dlif_student_last_v1/` |
| 验证时间 | 2026-07-06 |

分类别结果：

| 类别 | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.90598 | 0.87115 | 0.90781 | 0.57304 |
| fire | 0.90937 | 0.87915 | 0.91518 | 0.57767 |
| person | 0.90259 | 0.86314 | 0.90044 | 0.56841 |

备注：

- 这是 `last.pt` 的独立验证结果，不是训练期 `results.csv` 直接摘录。
- 该模型训练采用 early stopping，最佳结果出现在更早 epoch；本节仅记录用户要求的 `last.pt` 口径。
- 该历史 run 实际使用 `model_yaml/yolov8n-LIF-fire-person.yaml`，即 YOLOv8n/nano 尺度；选择该尺度是为了与本文 YOLOv8n 主线进行参数规模公平比较。

### 4.2 M2D-LIF M 尺度学生

本节记录 M2D-LIF student 中 `batch=16`、M 尺度骨干的训练结果。该实验规模大于本文 N 系列模型，因此只作为外部强基线补充，不用于参数规模公平比较。

| 项目 | 设置 |
| --- | --- |
| 实验目录 | `baselines/M2D-LIF/runs/m2d_lif/m2dlif_student_fp_dual_m_b16_noaug/` |
| model | `./model_yaml/yolov8-LIF.yaml` |
| data | `data/RGBT3M-fire-person-dual.yaml` |
| batch | 16 |
| imgsz | `[480, 640]` |
| device | `0,1` |
| epochs | 200 |
| patience | 50 |
| optimizer | SGD |
| lr0 | 0.01 |
| cls | 0.1 |

`results.csv` 最后一行为 epoch 151，说明该 run 在 early stopping 下提前结束。该 CSV 仅包含 overall 指标，没有分类别 fire/person 行。

| checkpoint/来源 | epoch | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `results.csv` last row | 151 | 0.93051 | 0.89502 | 0.92712 | 0.60570 |

备注：

- 该结果高于本文 N 系列模型是合理的，因为它使用 M 尺度骨干，模型容量更大。
- 本表按用户要求直接记录训练目录中 `results.csv` 的最后一行，不等同于独立重新验证的 `last.pt` 结果。
- 若后续论文需要正式外部对比，应再用统一验证脚本对 `weights/last.pt` 或 `weights/best.pt` 做一次独立验证，并补充分类别指标。

### 4.3 CALNet-nano 双流 OBB

实验信息：

| 项目 | 设置 |
| --- | --- |
| 方法 | `CALNet-nano` 双流 OBB 模型 |
| checkpoint | `baselines/CALNet-Dronevehicle/runs/train/calnet_n_hbb_b16_2gpu_noaug/weights/best.pt` |
| data | `baselines/CALNet-Dronevehicle/data/RGBT-3M-fire-person.yaml` |
| 训练 | 从零初始化，`batch_size=16`，`imgsz=[480,640]`，DDP 2 GPU，原始 CALNet `hyp.finetune_DroneVehicle.yaml` |
| 验证命令 | `python val.py --data data/RGBT-3M-fire-person.yaml --weights runs/train/calnet_n_hbb_b16_2gpu_noaug/weights/best.pt --task test --imgsz 640 --batch-size 16 --device 0 --verbose` |
| 验证参数 | `conf_thres=0.01`，`iou_thres=0.4` |
| 验证目录 | `baselines/CALNet-Dronevehicle/runs/val/val_calnet_n_hbb_b16_2gpu_noaug_best/` |
| 验证时间 | 2026-07-15 |

分类别结果：

| 类别 | P | R | HBB mAP50 | HBB mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.865 | 0.824 | 0.863 | 0.464 |
| fire | 0.885 | 0.826 | 0.870 | 0.477 |
| person | 0.844 | 0.822 | 0.856 | 0.450 |

备注：

- 这是 `best.pt` 的独立验证结果，模型融合后为 12,985,413 参数。
- CALNet 使用 OBB head；本表报告其验证脚本输出的 HBB 指标，不应与 OBB 指标混用。
- 为使仓库自带 `val.py` 支持 CALNet 双输入验证，对 `baselines/CALNet-Dronevehicle/val.py`、`models/common.py`、`utils/datasets.py` 做了最小适配。

### 4.4 CP-YOLOv11-MF 双流模型

实验信息：

| 项目 | 设置 |
| --- | --- |
| 方法 | `CP-YOLOv11-MF`，Cross-modal Progressive YOLOv11 with Mid-term Fusion |
| 论文 | Zhang et al., "A UAV-Based Multi-Scenario RGB-Thermal Dataset and Fusion Model for Enhanced Forest Fire Detection", Remote Sens. 2025 |
| 骨干 | YOLOv11-n；原论文用 YOLOv11-s，为尺度公平使用 n |
| 融合 | CPCA + PPAS，中期融合 |
| checkpoint | `baselines/CP-YOLOv11-MF/runs/detect/train_cp_yolo11mf/last.pt` |
| data | `/data/xwh/dataset/RGBT-3M/RGBT-3M` |
| 训练 | `batch=16`，`epochs=200`，`lr0=0.01`，`cls=0.1`，`imgsz=[480,640]`，SGD，无实际数据增强 |
| 验证 | `conf=0.001`，`iou=0.6` |
| 验证时间 | 2026-07-14 |
| 代码 | `baselines/CP-YOLOv11-MF/model.py` |

分类别结果：

| 类别 | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.91452 | 0.86491 | 0.91606 | 0.56538 |
| fire | 0.93353 | 0.85975 | 0.91607 | 0.56217 |
| person | 0.89592 | 0.87073 | 0.91606 | 0.56858 |

备注：

- 这是 `last.pt` 的独立验证结果（epoch 200，最终 avg_loss=13.63）。
- 本模型 bbox 解码使用简化 DFL（未完全解码分布），mAP50-95 可能偏低；mAP50 和 P/R 指标可靠。
- 原论文（YOLOv11-s，batch=4）报告 P=92.5%, R=93.5%, mAP50=96.3%, mAP50-95=62.9%；我们的 YOLOv11-n 版本参数更少，指标差距合理。
- 代码复现自论文描述，非官方开源代码。

### 4.5 AFDNet 双流模型

实验信息：

| 项目 | 设置 |
| --- | --- |
| 方法 | `AFDNet`，Asymmetric Frequency-Decoupled Network |
| 论文 | Chen et al., 2026, "Asymmetric frequency-decoupled network for robust visible-infrared fire detection" |
| 骨干 | YOLOv11-n 尺度双流模型 |
| 融合 | frequency-decoupled fusion at P3/P4/P5 |
| 训练目录 | `baselines/AFDNet/train_fp_AFDNet_y11n_seed0_cls01_noaug/` |
| checkpoint | `baselines/AFDNet/train_fp_AFDNet_y11n_seed0_cls01_noaug/weights/last.pt` |
| data | `ultralytics/cfg/datasets/RGBT-3M-dual-fire-person-local.yaml` |
| 训练 | `batch=16`，`epochs=200`，`lr0=0.01`，`cls=0.1`，`imgsz=[480,640]`，SGD，无实际数据增强，`deterministic=False` |
| 代码 | `baselines/AFDNet/model.py`、`baselines/AFDNet/modules.py` |

`results.csv` 最后一行，即 epoch 200 的训练期验证指标：

| 类别 | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.91190 | 0.89633 | 0.92505 | 0.55631 |
| fire | 0.92411 | 0.90032 | 0.93485 | 0.56948 |
| person | 0.89970 | 0.89234 | 0.91525 | 0.54313 |

训练 loss 与验证 loss：

| epoch | train/box | train/cls | train/dfl | val/box | val/cls | val/dfl |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.12322 | 0.02689 | 0.75568 | 1.29447 | 0.14006 | 0.93522 |

AFDNet 单模型 profiling：

| Params | FLOPs | mean latency | median latency | p10 latency | p90 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5.36M | 8.04G | 15.24 ms | 14.50 ms | 14.34 ms | 17.00 ms |

备注：

- AFDNet 指标来自 `results.csv`，不是独立重新验证 `last.pt` 的分类别输出。
- profiling 只统计模型 raw forward，不包含 dataloader、图像解码、letterbox、NMS 或绘图耗时。
- AFDNet 输出张量形状为 `[1, 6, 6300]`，与 fire/person 二分类三尺度检测头一致。

### 4.6 MCDet 双流模型

实验信息：

| 项目 | 设置 |
| --- | --- |
| 方法 | `MCDet`，Target-aware fusion for RGB-T fire detection |
| 骨干 | YOLOv5n 尺度双流复现；原论文使用 YOLOv5s，为公平比较改为 n 尺度 |
| 融合 | P3/P4/P5 使用 MRCF，neck concat 后使用 CGAN；主 loss 接入 WIoU |
| 训练目录 | `runs/detect/train_fp_MCDet_yolov5n_seed0_cls01_noaug_noamp/` |
| checkpoint | `runs/detect/train_fp_MCDet_yolov5n_seed0_cls01_noaug_noamp/weights/last.pt` |
| data | `ultralytics/cfg/datasets/RGBT-3M-dual-fire-person-local.yaml` |
| 训练 | `batch=16`，`epochs=200`，`lr0=0.01`，`cls=0.1`，`imgsz=[480,640]`，SGD，无数据增强，`deterministic=False`，`amp=False`，`workers=4` |
| 代码 | `baselines/MCDet/model.py`、`baselines/MCDet/modules.py`、`baselines/MCDet/losses.py` |

`results.csv` 最后一行，即 epoch 200 的训练期验证指标：

| 类别 | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.92165 | 0.89231 | 0.92877 | 0.56336 |
| fire | 0.93618 | 0.89719 | 0.93472 | 0.57388 |
| person | 0.90713 | 0.88743 | 0.92283 | 0.55284 |

训练 loss 与验证 loss：

| epoch | train/box | train/cls | train/dfl | val/box | val/cls | val/dfl |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.05187 | 0.02245 | 0.75180 | 1.30022 | 0.15333 | 0.95026 |

MCDet 单模型 profiling：

| Params | FLOPs | mean latency | median latency | p10 latency | p90 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10.28M | 14.83G | 29.30 ms | 28.88 ms | 27.57 ms | 31.65 ms |

备注：

- MCDet 指标来自 `results.csv`，不是独立重新验证 `last.pt` 的分类别输出。
- profiling 使用 `last.pt`，在 GPU2 上统计 raw forward；不包含 dataloader、图像解码、letterbox、NMS 或绘图耗时。
- MCDet 输出张量形状为 `[1, 6, 6300]`，与 fire/person 二分类三尺度检测头一致。

## 五、本文模型参照项

本文件只记录 baseline 与复杂度对比，本文主模型的完整实验链条见 `docs/ADR-yolov8.md`。当前复杂度表中保留 `YOLOv8n-C65`，用于和外部双模态方法对齐参数、FLOPs 和延时口径。

| 模型 | 结构摘要 | checkpoint | Params | FLOPs | mean latency |
| --- | --- | --- | ---: | ---: | ---: |
| YOLOv8n-C65 | P2-DMG + P3-CFRA + P4/P5-RS-SQF，P2-P5 四尺度检测头 | `runs/detect/train_fp_C65_y8nbackbone_seed0_cls01/weights/last.pt` | 5.22M | 13.49G | 21.19 ms |

## 六、复杂度与延时统计口径

统一 profiling 口径：

| 项目 | 设置 |
| --- | --- |
| 设备 | NVIDIA GeForce RTX 2080 Ti |
| dtype | FP32 |
| 输入 | 双流模型使用等价 `1×6×480×640`；CALNet 使用 `RGB 1×3×480×640 + IR 1×3×480×640` |
| FLOPs | `torch.profiler.profile(with_flops=True)` |
| latency | CUDA Event raw forward，warmup 50 次，计时 200 次 |
| 不包含 | 图像读取、预处理、NMS、绘图、dataloader |

备注：

- CALNet 采用其验证脚本一致的 fused 推理模型，参数量为融合后模型参数。
- YOLOv8n-C65 有 P2 检测输出，因此输出网格数量为 25,500；M2D-LIF、AFDNet、MCDet、CP-YOLOv11-MF 为 P3/P4/P5 输出，输出网格数量为 6,300。
- FLOPs 只统计 `torch.profiler` 可识别算子；不同模型含有自定义模块时，FLOPs 与实际 latency 不必严格单调一致。
