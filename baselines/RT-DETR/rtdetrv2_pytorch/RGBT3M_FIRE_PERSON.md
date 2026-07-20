# RT-DETRv2-S RGBT-3M Fire/Person Baselines

This directory uses the official `lyuwenyu/RT-DETR` RT-DETRv2 PyTorch implementation.

Local baseline scope:

- Model scale: official RT-DETRv2-S, implemented by `rtdetrv2_r18vd_120e_coco.yml`.
- Task: RGBT-3M fire/person two-class single-modal detection.
- Modalities: RGB and IR are trained as two separate single-modal models.
- Data format: YOLO labels converted to COCO JSON for the official RT-DETRv2 dataloader.
- Training recipe: official RT-DETRv2-S optimizer/model defaults, no real data augmentation beyond resize and tensor/box conversion to match the local baseline policy.
- Tuning checkpoint: official COCO RT-DETRv2-S checkpoint, with class-head shape mismatches skipped by the official tuning loader.
- `PResNet.pretrained` is disabled in these local configs because the full official RT-DETRv2-S tuning checkpoint initializes the backbone.

Configs:

- `configs/rgbt3m/rtdetrv2_s_rgb_fire_person.yml`
- `configs/rgbt3m/rtdetrv2_s_ir_fire_person.yml`

COCO annotation conversion:

```bash
python tools/convert_rgbt3m_fire_person_to_coco.py \
  --image-dir /data/xwh/dataset/RGBT-3M/RGBT-3M/RGB/train \
  --label-dir /data/xwh/dataset/RGBT-3M/RGBT-3M/labels_fire_person/train \
  --out dataset/rgbt3m_fire_person/annotations/instances_rgb_train.json
```
