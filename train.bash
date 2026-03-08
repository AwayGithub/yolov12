#!/usr/bin/env bash
set -e

# 1. 让 git bash 里可以使用 conda
source "/d/ProgramData/miniconda3/etc/profile.d/conda.sh"

# 2. 激活环境
conda activate yolov12

# 3. 运行两次训练
cd "/e/Yan-Unifiles/lab/exp/yolov12"

python train.py --input_mode rgb_input
python train.py --input_mode ir_input