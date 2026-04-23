# AGENTS.md - Developer Guidelines for YOLOv12

## Overview

This is the YOLOv12 repository, an attention-centric real-time object detector built on Ultralytics. The project includes detection, classification, and instance segmentation models.

## Branch Workflow

- All future YOLOv12 project changes must be made on the `claude/magical-burnell` branch.
- The user will review changes on this branch and manually push or merge them to `main`.
- Do not make project edits directly on `main` unless the user explicitly overrides this instruction.

## Project Structure

```
yolov12/
├── ultralytics/           # Main package
│   ├── models/            # Model definitions (YOLO, SAM, RTDETR, etc.)
│   ├── nn/                # Neural network modules
│   ├── data/              # Data loading and augmentation
│   ├── utils/             # Utilities (metrics, plotting, downloads, etc.)
│   ├── engine/            # Training/inference engine
│   ├── trackers/          # Multi-object tracking (BoT-SORT, ByteTrack)
│   ├── solutions/         # Solution integrations
│   └── cfg/               # Configuration files
├── tests/                 # Test suite
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Build/Lint/Test Commands

### Installation (Development Mode)

```bash
pip install -e .
pip install -e ".[dev]"  # With dev dependencies
```

### Running Tests

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_python.py

# Run a single test function
pytest tests/test_python.py::test_model_forward

# Run tests with coverage
pytest --cov=ultralytics --cov-report=html

# Run tests excluding slow tests
pytest -m "not slow"

# Run specific test markers
pytest -m slow  # Only slow tests
```

### Linting and Formatting

```bash
# Format code (YAPF)
yapf -r ultralytics/ tests/ --in-place

# Sort imports (isort)
isort ultralytics/ tests/

# Lint code (ruff)
ruff check ultralytics/ tests/

# Check spelling
codespell ultralytics/ tests/

# Run all checks
ruff check . && yapf --diff . && codespell ultralytics/ tests/
```

### Type Checking

```bash
# Run mypy (if installed)
mypy ultralytics/ --ignore-missing-imports
```

## Code Style Guidelines

### General Principles

- **License**: All source files must start with the Ultralytics AGPL-3.0 license header
- **Line length**: Maximum 120 characters
- **Python version**: >=3.8

### Imports

```python
# Standard library first
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Third-party imports
import cv2
import numpy as np
import torch
import yaml

# Local imports
from ultralytics.utils import ASSETS, ROOT
from ultralytics.models import YOLO

# Use isort for automatic sorting
# Group: stdlib, third-party, local
# Within each group: alphabetically
```

### Formatting (YAPF)

- Based on PEP 8
- Column limit: 120
- Space before comment: 2
- Coalesce brackets: true
- Spaces around power operator: true

### Naming Conventions

```python
# Variables and functions: snake_case
def process_image(image_data):
    learning_rate = 0.001

# Classes: PascalCase
class DetectionModel(nn.Module):
    pass

# Constants: UPPER_SNAKE_CASE
MAX_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 640

# Private methods: prefix with underscore
def _private_method(self):
    pass
```

### Type Hints

```python
# Use type hints for function signatures
def train_model(
    model: torch.nn.Module,
    data: str,
    epochs: int = 100,
    imgsz: int = 640,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train the model with specified parameters."""
    pass
```

### Docstrings

```python
def predict(self, source=None, imgsz=640, **kwargs):
    """
    Perform object detection on images.

    Args:
        source: Image source (path, URL, PIL Image, numpy array)
        imgsz: Target image size for inference
        **kwargs: Additional arguments for prediction

    Returns:
        A list of Results objects containing predictions

    Example:
        >>> model = YOLO('yolov12n.pt')
        >>> results = model.predict('image.jpg')
    """
```

### Error Handling

```python
# Use descriptive error messages
try:
    model = YOLO(weights)
except FileNotFoundError:
    raise ValueError(f"Model weights not found: {weights}") from None

# Use warnings for non-critical issues
import warnings
warnings.warn("This feature is deprecated", DeprecationWarning)
```

### Git Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Keep subject line under 72 characters
- Reference issues: "Fix #123: resolve detection issue"

## Testing Guidelines

### Test File Structure

```python
# tests/test_module.py
import pytest
from ultralytics import YOLO

MODEL = "yolo12n.pt"

def test_function_name():
    """Test description."""
    # Arrange
    model = YOLO(MODEL)
    
    # Act
    result = model.predict()
    
    # Assert
    assert result is not None
```

### Test Fixtures

- Use `tests/conftest.py` for shared fixtures
- Use `tests/__init__.py` for test constants (MODEL, SOURCE, TMP, etc.)

### Skip Conditions

```python
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA not available")
def test_gpu_functionality():
    pass
```

## Model Configuration

### YAML Config Files

Model configurations are in `ultralytics/cfg/models/`:

```yaml
# Example: yolo12n.yaml
# YOLOv12n model
nc: 80  # number of classes
scales:
  n: [0.33, 0.25, 1024]
```

### Training Examples

```python
from ultralytics import YOLO

model = YOLO('yolo12n.yaml')
results = model.train(
    data='coco.yaml',
    epochs=600,
    batch=256,
    imgsz=640,
)
```

## Common Tasks

### Export Model

```python
model.export(format="onnx")  # or "engine", "tflite", etc.
```

### Run Validation

```python
model.val(data='coco.yaml')
```

### Multi-GPU Training

```python
results = model.train(data='coco.yaml', device='0,1,2,3')
```

## Resources

- [Ultralytics Docs](https://docs.ultralytics.com)
- [YOLOv12 Paper](https://arxiv.org/abs/2502.12524)
