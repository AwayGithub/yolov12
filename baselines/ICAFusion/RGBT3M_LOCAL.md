# ICAFusion on RGBT-3M Fire/Person

## Source

- Official repository: https://github.com/chanchanchan97/ICAFusion
- Paper: ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection, Pattern Recognition.

## Method Summary

ICAFusion uses a dual-stream detector for aligned RGB and thermal image pairs. RGB and thermal images are first processed by two independent YOLOv5-style backbones. Intermediate C3, C4 and C5 features are fused by the Dual-modal Feature Fusion module.

The Dual-modal Feature Fusion module contains:

- Spatial Feature Shrinking: average and max pooling compress the spatial size before attention.
- Iterative Cross-modal Feature Enhancement: two cross-attention transformers enhance RGB from thermal and thermal from RGB. The iterative blocks share parameters.
- NIN fusion: the enhanced RGB and thermal features are concatenated and reduced by a 1x1 convolution before the YOLO neck/head.

For this local baseline, the official YOLOv5n Transfusion configuration is used because it is the closest official scale to the YOLOv8n/nano mainline.

## Local Training

Dataset view:

- `data/RGBT-3M-fire-person/visible` links to `/data/xwh/dataset/RGBT-3M/RGBT-3M/RGB`
- `data/RGBT-3M-fire-person/infrared` links to `/data/xwh/dataset/RGBT-3M/RGBT-3M/IR`
- `data/RGBT-3M-fire-person/labels` links to `/data/xwh/dataset/RGBT-3M/RGBT-3M/labels_fire_person`

Training command:

```bash
conda run -n yolov12 python train.py \
  --weights "" \
  --cfg models/transformer/yolov5n_Transfusion_RGBT3M_fire_person.yaml \
  --data data/multispectral/RGBT-3M-fire-person.yaml \
  --hyp data/hyp.rgbt3m-noaug.yaml \
  --epochs 200 \
  --batch-size 16 \
  --img-size 640 640 \
  --rect-size 480 640 \
  --rect \
  --device 0 \
  --workers 4 \
  --project runs/train \
  --name train_fp_ICAFusion_yolov5n_seed0_cls01_noaug_480x640 \
  --exist-ok
```

Local compatibility changes:

- Added `--rect-size H W` so train and validation tensors are explicitly `480x640`.
- Replaced deprecated `np.int` uses with `int` for the current NumPy version.
- Scaled object loss by rectangular image area, `H*W/640^2`.
