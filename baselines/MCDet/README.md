# MCDet Baseline

This directory contains an isolated reproduction of **MCDet: Target-Aware Fusion for RGB-T Fire Detection**.

The code is kept under `baselines/MCDet` so it does not change the main YOLOv12 model registry or the current paper
innovation chain.

## Paper Mapping

The implementation follows the paper figures as follows:

- **Figure 1, overall architecture**: dual RGB/IR YOLOv5-style backbones, MRCF fusion at P3/P4/P5, FPN/PAN neck, and
  a YOLOv5 detection head.
- **Figure 1, MRCF**: `MultidimensionalRepresentationCollaborativeFusion` combines BVSSM, TSFF and FRM.
- **Figure 2, VSSM**: `BidirectionalVisualStateSpaceModule` flattens visible and infrared features into forward and
  reverse sequences, sends them through real Mamba state-space blocks, splits them back into two modality features,
  and performs weighted fusion.
- **Figure 3, CGAN**: `ContentGuidedAttention` computes spatial attention, channel attention, and a gate that balances
  the two before applying a residual refinement.

## Fair Local Setting

The paper uses YOLOv5s and training augmentation. For comparison with the local N-series models, this reproduction uses:

- YOLOv5n compound scaling: `n: [0.33, 0.25, 1024]`
- dual input with 6 channels: IR first, RGB second, matching this repository's `dual_input` convention
- fire/person two-class dataset YAML
- no Mosaic, HSV, flipping, scaling, mixup or copy-paste augmentation by default
- `imgsz=[480, 640]`, `batch=16`, `SGD`, `lr0=0.01`, `seed=0`

## Engineering Notes

This baseline now uses the paper-required external operators:

- BVSSM uses `mamba-ssm`.
- TSFF and CGAN use `torchvision.ops.DeformConv2d` through the local `DCNv2` wrapper.
- WIoU is wired into the MCDet training criterion as the main bbox regression loss. Classification and DFL remain on the
  existing Ultralytics path.

The WIoU connection is local to `baselines/MCDet`; it does not alter the shared loss used by the main YOLOv12 models.

For the current `yolov12` environment, the tested dependency set is:

- `torch==2.2.2`
- `torchvision==0.17.2`
- `mamba-ssm==2.2.4`
- `transformers==4.41.2`
- `huggingface-hub==0.23.2`
- `safetensors==0.4.3`

Do not install the latest `mamba-ssm` from source here unless necessary; it tries to compile many CUDA kernels. The
working installation used the official prebuilt wheel for `cu12 + torch2.2 + cp311 + cxx11abiFALSE`.

## Files

- `cfg/MCDet-yolov5n-fire-person.yaml`: YOLOv5n dual-stream model config.
- `modules.py`: MRCF, BVSSM, TSFF, FRM and CGAN modules.
- `losses.py`: MCDet WIoU detection loss.
- `model.py`: YOLO wrapper that builds the MCDet dual-stream detector.
- `train_mcdet.py`: training entrypoint, not launched automatically.
- `tests/test_mcdet_baseline.py`: shape and build smoke tests.

## Training Command

Training has not been started. When needed:

```bash
conda run -n yolov12 python baselines/MCDet/train_mcdet.py --device 3 --batch 16 --workers 4
```

Use a persistent launcher when running the full 200-epoch experiment.
