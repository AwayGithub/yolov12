# ADR：Baselines 结果记录

**状态：** 进行中  
**创建日期：** 2026-07-06  
**最后更新：** 2026-07-12

## 一、目的

本文件用于集中记录各类基线与外部对比方法的独立验证结果，避免与主创新链 ADR 混写。

## 二、YOLOv8n 与 YOLOv12n 单模态基线

### 2.1 实验定义

- 任务：RGBT-3M fire/person 二分类单模态检测。
- 口径：均为 `last.pt` 的独立验证结果。
- YOLOv8n RGB/IR：M2D-LIF `teacherTraining` 的 nano 教师；来源 `docs/ADR-yolov8.md` §7.1。
- YOLOv12n RGB/IR：原始单分支 YOLOv12n；来源 `docs/ADR-005-fire-person-binary-scope.md` §6.3。

### 2.2 独立验证结果

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

## 三、M2D-LIF 双流学生模型

### 3.1 实验定义

- 方法：`M2D-LIF` 双流学生模型（YOLOv8n/nano 尺度）
- checkpoint：`baselines/M2D-LIF/runs/m2d_lif/m2dlif_student_fp_dual/weights/last.pt`
- 数据：`baselines/M2D-LIF/data/RGBT3M-fire-person-dual.yaml`
- 验证输出目录：`baselines/M2D-LIF/val_fire_person/val_m2dlif_student_last_v1/`
- 验证时间：2026-07-06

### 3.2 `last.pt` 独立验证结果

| 类别 | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.90598 | 0.87115 | 0.90781 | 0.57304 |
| fire | 0.90937 | 0.87915 | 0.91518 | 0.57767 |
| person | 0.90259 | 0.86314 | 0.90044 | 0.56841 |

### 3.3 备注

1. 这是 `last.pt` 的独立验证结果，不是训练期 `results.csv` 直接摘录。
2. 该模型训练采用 early stopping，最佳结果出现在更早 epoch；本节仅记录用户要求的 `last.pt` 口径。
3. 该历史 run 实际使用 `model_yaml/yolov8n-LIF-fire-person.yaml`，即 YOLOv8n/nano 尺度；选择该尺度是为了与
   本项目的 YOLOv8n 主线进行参数规模公平比较，不代表 M2D-LIF 上游默认的 m 尺度配置。

## 四、CALNet-nano 双流 OBB 模型

### 4.1 实验定义

- 方法：`CALNet-nano` 双流 OBB 模型
- checkpoint：`baselines/CALNet-Dronevehicle/runs/train/calnet_yolov5n_r3m_fire_person_b8/weights/best.pt`
- 数据：`baselines/CALNet-Dronevehicle/data/RGBT-3M-fire-person.yaml`
- 训练：从零初始化，`batch_size=8`，`imgsz=[480, 640]`，原始 CALNet `hyp.finetune_DroneVehicle.yaml`
- 验证输出目录：`baselines/CALNet-Dronevehicle/runs/train/calnet_yolov5n_r3m_fire_person_b8/`
- 验证时间：2026-07-12

### 4.2 `best.pt` 独立验证结果

| 类别 | P | R | HBB mAP50 | HBB mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all | 0.819 | 0.793 | 0.837 | 0.389 |
| fire | 0.870 | 0.807 | 0.859 | 0.408 |
| person | 0.769 | 0.779 | 0.815 | 0.370 |

### 4.3 备注

1. 这是 `best.pt` 的独立验证结果，模型融合后为 13,228,953 参数。
2. CALNet 使用 OBB head；本表报告其验证脚本输出的 HBB 指标，不应与 OBB 指标混用。
