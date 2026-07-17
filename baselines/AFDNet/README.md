# AFDNet Baseline

This directory contains the local reproduction of:

Chen et al., 2026, *Asymmetric frequency-decoupled network for robust visible-infrared fire detection*.

This is a baseline implementation. It is intentionally kept under `baselines/AFDNet` and is not registered as a
mainline YOLOv12 project module.

## Files

- `modules.py`: Haar frequency split/reconstruction and AFDNet fusion blocks.
- `model.py`: `AFDNetDualStreamDetectionModel` and `AFDNetYOLO`, used only by this baseline.
- `cfg/AFDNet-y11n-fire-person.yaml`: YOLO11n-size dual-stream fire/person model with AFDNet fusion at P3/P4/P5.
- `train_afdnet.py`: training entrypoint with the same fair settings used for local baselines.
- `tests/test_afdnet_baseline.py`: minimal unit/smoke tests.

## Training

Default settings:

- image size: `480x640`
- batch size: `16`
- optimizer: `SGD`
- seed: `0`
- data augmentation: disabled by default
- auxiliary RGB/IR P3 heads: disabled by default

Example:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n yolov12 \
  python baselines/AFDNet/train_afdnet.py \
  --epochs 200 \
  --batch 16 \
  --device 0 \
  --name train_fp_AFDNet_y11n_seed0_cls01_noaug
```

With `CUDA_VISIBLE_DEVICES=3`, the script prints logical `CUDA:0`, but it maps to physical GPU3.

## Module Mapping

The implementation follows the paper figure as:

- MFD: one-level Haar DWT splits each RGB/IR feature into `LL`, `LH`, `HL`, and `HH`.
- TLA: IR `LL` produces a spatial mask from channel max/avg pooling, `Conv5x5`, and sigmoid. IR local standard
  deviation produces `alpha` through global average pooling, a fully connected layer, and sigmoid. The low-frequency
  output is `alpha * X_LL_IR + (1 - alpha) * F_LL_VIS`, followed by `Conv3x3 + BN + SiLU`.
- SHR: visible `LH/HL/HH` are concatenated, passed through lightweight grouped dilation `D=1/3/5` branches,
  concatenated again, negated, and mapped by `Conv1x1 + BN + Sigmoid` to `M_occ`. Each high-frequency subband is
  restored as `F_VIS * (1 - M_occ) + F_IR * M_occ`.
- IDWT: reconstructed low/high-frequency outputs are mapped back to a spatial feature and combined with the paper's
  `0.2` residual path.
