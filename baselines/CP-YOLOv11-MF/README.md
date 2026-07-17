# CP-YOLOv11-MF: Cross-modal Progressive YOLOv11 with Mid-term Fusion

Reimplementation of:
> Zhang, Y.; Rui, X.; Song, W. "A UAV-Based Multi-Scenario RGB-Thermal Dataset and Fusion Model for Enhanced Forest Fire Detection."
> *Remote Sens.* 2025, 17, 2593. https://doi.org/10.3390/rs17152593

## Architecture

Dual YOLOv11-n backbone (RGB + IR) with mid-term fusion at P3/P4/P5 scales.

```
  RGB Image ──→ YOLOv11 Backbone ──→ P3, P4, P5
                                           │
  IR  Image ──→ YOLOv11 Backbone ──→ P3, P4, P5
                                           │
              ┌────────────────────────────┘
              │
   P3: PPAS(Concat) + CPCA    ← best small/medium targets
   P4: PPAS(Concat) + CPCA    ← best small/medium targets
   P5: Concat + CPCA           ← no PPAS for large targets (Scheme 2)
              │
              ▼
         FPN Neck → Detect
```

### Key Modules

| Module | Full Name | Description |
|--------|-----------|-------------|
| **CPCA** | Channel Prior Convolutional Attention | Channel attention (CBAM-style) + spatial attention (depthwise conv) for cross-modal feature interaction. Eq.4-6 in paper. |
| **PPAS** | Parallel Patch-Aware Splicing | 3-branch (local/global/serial) feature fusion + channel-spatial attention. Replaces simple concat at P3/P4. Fig.12-13 in paper. |

### Fusion Scheme

- **Scheme 2** (best balanced, Table 11): PPAS at P3 + P4, plain Concat at P5.
- CPCA applied after each fusion as residual refinement.

## Files

```
CP-YOLOv11-MF/
├── model.py              # Full model (CPCA + PPAS + dual backbone + FPN + Detect)
├── scripts/
│   └── train.py          # Training entrypoint (skeleton)
├── ultralytics/
│   └── nn/modules/
│       ├── __init__.py
│       ├── cpca.py       # Standalone CPCA module
│       └── ppas.py       # Standalone PPAS module
└── README.md
```

## Training

```bash
conda run -n yolov12 python scripts/train.py \
  --batch 16 --epochs 200 --lr0 0.01 --device 0
```

**Note:** The training script currently has a skeleton dataloader. Replace the placeholder loop with real RGBT-3M data loading before actual training.

## Paper Results (YOLOv11-s backbone)

| Model | P | R | mAP50 | mAP50-95 | Params | Size |
|-------|---|---|-------|----------|--------|------|
| YOLOv11 (visible) | 90.7% | 90.3% | 93.6% | 55.0% | 9.41M | 18.3 MB |
| YOLOv11 (infrared) | 91.2% | 88.6% | 93.6% | 57.6% | 9.41M | 18.3 MB |
| YOLOv11-MF (plain concat) | 90.4% | 91.3% | 95.3% | 58.7% | - | - |
| YOLOv11-MF + CPCA | 91.9% | 92.3% | 96.0% | 61.5% | - | - |
| **CP-YOLOv11-MF** | **92.5%** | **93.5%** | **96.3%** | **62.9%** | 11.83M | 23 MB |

Our reimplementation uses YOLOv11-**n** (not s) for scale fairness with our YOLOv12-n baseline.

## License

For research comparison only. Original paper code not open-sourced.
