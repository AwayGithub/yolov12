# 20260525_122822 Innovation: FreDFT Cross-QK Dilated FFN

## 1. 本轮选择的论文

- 论文标题：FreDFT: Frequency Domain Fusion Transformer for Visible-Infrared Object Detection
- 作者/年份：Wencong Wu, Xiuwei Zhang, Hanlin Yin, Shun Dai, Hongxi Zhang, Yanning Zhang, 2025
- 论文库文件：`docs/literature_rgbt_fusion/pdfs/2025_arXiv_FreDFT.pdf`
- GitHub 仓库：https://github.com/WenCongWu/FreDFT
- 官方代码参考：`models/common.py` 中 `FDCA`、`FDFFN`、`FDFTM`

选择原因：当前确认实验基线是 `P3 aux + P2 DMGFusionInit8d(alpha_init=1.0, beta_init=-0.1)`，不包含 CMG/CMA。已有 FreDFT P3 迁移版能提供频域融合路径，但它保留了同模态 Q/K 频域响应和 3/5/7 FFT FFN。本轮改造继续基于 FreDFT 的 RGB/T 频域互补思想，重点验证跨模态 Q/K 响应与空间多膨胀率 FFN 是否比原迁移版更适合烟雾、火焰、人员检测。

## 2. 本轮迁移的唯一创新点

创新点名称：FreDFT-CQK-Dilated (`FreDFTFusion`)

原论文中的作用：官方 FreDFT 的 `FDFTM` 在 P3/P4/P5 处对 RGB/IR 特征做频域注意力和频域 FFN 融合。原始 `FDCA` 使用同模态 `FFT(Q) * FFT(K)` 得到频域响应，再用对侧 `V` 调制；原始 `FDFFN` 使用 3x3/5x5/7x7 depthwise 分支、FFT split/exchange 和 IFFT 重组。

本轮在 YOLOv12 中保留 `FreDFTFusion.forward(x_rgb, x_ir) -> fused` 接口，但替换内部逻辑：

- `_FreDFTFrequencyAttention` 改为双向跨模态 Q/K：`FFT(Q_RGB) * FFT(K_IR)` 与 `FFT(Q_IR) * FFT(K_RGB)`。
- delta 路由采用同流 V 回同流：`RGB_delta = Conv(RGB_V * Norm(Q_IR*K_RGB))`，`IR_delta = Conv(IR_V * Norm(Q_RGB*K_IR))`。
- `_FreDFTFeedForward` 删除 FFT/IFFT，改为 dilation=1/2/3 的空间 depthwise 分支，并沿用通道 split/exchange。
- `FreDFTFusion` 不再使用 `plain + freq_scale * freq` 总残差，改为注意力残差、FFN 残差后 `ReLU(Conv(cat(rgb, ir)))` 输出。

## 3. 与已有创新点的差异

本轮是对 `docs/20260517_165515_innovation_fredft_p3.md` 中旧 FreDFT P3 迁移版的替换，不新增并行旧版配置。旧版 FreDFT 的关键机制是同模态 Q/K 频域响应、3/5/7 mixed-scale FFT FFN，以及 `freq_scale` 控制的总残差输出；本轮改成跨模态 Q/K、空间膨胀率 split/exchange FFN，并移除 `freq_scale`。

与其他已有创新点也不重复：

- 不改 P2 `DMGFusionInit8d`，仍保留 `alpha_init=1.0`、`beta_init=-0.1`。
- 不引入 CMG/CMA，也不用 `|RGB-IR|` 差异图门控。
- 不改 P3 auxiliary head、loss、data pipeline 或检测 head。
- 不与 M2D-LIF 的 RGB 局部照明可靠性路由重叠；本轮仍只在 P3 FreDFT 路径内建模频域/空间多尺度互补。

## 4. 代码修改说明

修改文件：

- `ultralytics/nn/modules/block.py`
  - `_FreDFTFrequencyAttention.forward()`：从同模态 Q/K 改为跨模态 Q/K。
  - `_FreDFTFeedForward`：从 3/5/7 FFT FFN 改为 dilation=1/2/3 spatial split/exchange FFN。
  - `FreDFTFusion`：默认 `expansion=3.0`，移除 `scale_init`、`plain_fuse`、`freq_fuse`、`freq_scale`，新增 `fuse` 和 `relu`。
- `ultralytics/nn/tasks.py`
  - `FreDFTFusion` 实例化不再传 `scale_init`。
  - `fredft_expansion` 默认值从 `1.0` 改为 `3.0`。
- `ultralytics/cfg/models/v12/yolov12-dual-p2-fredft-p3.yaml`
  - 保持 `p2_fusion: dmg_init8d`。
  - 保持 `freq_fusion_stages: [p3]`。
  - `fredft_expansion` 改为 `3.0`。
  - 删除 `fredft_scale_init`。
- `tests/test_cross_modal.py`
  - 更新 FreDFT 梯度测试。
  - 新增 Cross-QK + dilation FFN 结构测试。
  - 更新 YAML 配置断言。
- `docs/20260517_165515_innovation_fredft_p3.md`
  - 顶部追加旧 FreDFT 迁移语义已被本轮实现替换的说明。

输入输出维度：

- `FreDFTFusion.forward(x_rgb, x_ir)` 输入均为 `(B, C, H, W)`，输出为 `(B, C, H, W)`。
- P3 n scale 当前通道约为 `C=128`，`fredft_expansion=3.0` 时 FFN hidden 为 `384`。
- FDCA 保持 `C -> 6C` 投影，chunk 后 Q/K/V 各为 `2C`。

参数量：

- P3 `FreDFTFusion` 模块参数量：389,376。
- `yolov12-dual-p2-fredft-p3.yaml` 当前总参数量：4,728,046。

## 5. 验证结果

已执行 TDD 红灯：

```powershell
$env:YOLO_CONFIG_DIR='.yolo_test_config'
conda run --no-capture-output -n yolov12 python -m pytest tests/test_cross_modal.py::test_fredft_fusion_output_shape_and_gradients tests/test_cross_modal.py::test_fredft_fusion_uses_cross_qk_dilated_ffn_structure tests/test_cross_modal.py::test_dual_stream_fredft_p3_cfg_extends_confirmed_dmg_init8d_baseline -q
```

结果：`3 failed`，失败点分别是旧实现缺少 `fuse`、缺少 `dwconv_d1`，以及 YAML 仍为 `fredft_expansion: 1.0`。

实现后验证：

```powershell
$env:YOLO_CONFIG_DIR='.yolo_test_config'
conda run --no-capture-output -n yolov12 python -m pytest tests/test_cross_modal.py::test_fredft_fusion_output_shape_and_gradients tests/test_cross_modal.py::test_fredft_fusion_uses_cross_qk_dilated_ffn_structure tests/test_cross_modal.py::test_dual_stream_fredft_p3_cfg_extends_confirmed_dmg_init8d_baseline -q
```

结果：`3 passed in 0.94s`。

```powershell
$env:YOLO_CONFIG_DIR='.yolo_test_config'
conda run --no-capture-output -n yolov12 python -m pytest tests/test_cross_modal.py -q
```

结果：`22 passed in 8.03s`。

```powershell
$env:YOLO_CONFIG_DIR='.yolo_test_config'
conda run --no-capture-output -n yolov12 python -m compileall -q ultralytics\nn\modules\block.py ultralytics\nn\tasks.py tests\test_cross_modal.py
```

结果：通过，无输出。

```powershell
$env:YOLO_CONFIG_DIR='.yolo_test_config'
conda run --no-capture-output -n yolov12 python -m ruff check ultralytics\nn\modules\block.py ultralytics\nn\tasks.py tests\test_cross_modal.py
```

结果：`All checks passed!`

最小 forward：

- 构建 `DualStreamDetectionModel('yolov12-dual-p2-fredft-p3.yaml', nc=3, verbose=False)`。
- 输入 `torch.zeros(1, 6, 480, 640)`。
- 结果：`stride [4.0, 8.0, 16.0, 32.0]`，`out_type tuple`，`num_output_tensors 5`，`finite True`。
- P2：`DMGFusionInit8d alpha=1.0 beta=-0.10000000149011612`。
- P3：`FreDFTFusion`。
- `has_freq_scale False`，`aux True`。

未执行完整训练，原因是本轮目标是代码改造和最小可运行性验证；完整训练需要单独 GPU 实验。

## 6. 预期影响

- smoke：跨模态 Q/K 可能让 RGB 弥散纹理和 IR 背景结构形成更直接的频域响应；空间 dilation FFN 可能帮助烟雾边缘和大范围低对比区域，但也可能放大云雾/背景热噪声。
- fire：RGB 火焰颜色边缘与 IR 热源轮廓更适合跨模态 Q/K 查询，预期对 fire recall 和定位稳定性有帮助。
- person：IR 人体轮廓与 RGB 局部边缘通过跨模态频域响应结合，可能改善中小尺度人员，但 dilation=3 分支也可能引入周边热源干扰。
- RGB/T fusion：本轮从“同模态频域响应 + 对侧 V 调制”转为“跨模态 Q/K 响应 + 同流 V 回同流”，更明确地建模两个模态之间的结构对应关系。
- 参数/速度：相对旧 FreDFT P3 迁移版参数增加，且 `expansion=3.0` 会增加 P3 FFN 计算量；但删除 FFN 内 FFT/IFFT 后，空间 dilation 分支可能比原 FFN 更易训练和部署。

## 7. 后续实验建议

- 主实验：使用 `yolov12-dual-p2-fredft-p3.yaml`，保持 `batch=16`、`lr0=0.01`，与 F0、F2、旧 FreDFT P3 记录做对比。
- 消融 1：`fredft_expansion=1.0`，只验证 FFN 宽度是否过大导致过拟合。
- 消融 2：`freq_fusion_stages: [p4]`，观察更低分辨率层是否更适合 cross-QK。
- 消融 3：把 `FFT(Q) * FFT(K)` 改为 `FFT(Q) * conj(FFT(K))`，单独检验复共轭相关性。
- 风险观察：重点看 smoke mAP50-95 是否被 dilation 背景响应拖低，以及 fire/person recall 是否能抵消新增计算量。
