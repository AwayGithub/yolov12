# RGB-T Feature Interaction Literature Notes

更新日期：2026-05-07

本目录聚焦 **红外/热红外与可见光的特征级交互与融合**。这里的“融合”不是把 RGB 图像和红外图像合成一张图，而是 RGB/IR 双分支 backbone、cross-modal attention、feature calibration、feature alignment、reliability/gating、prompt/adapter 等网络结构，最终服务检测、分割或显著目标检测任务。

## 检索口径

- 主任务优先级：目标检测 > 语义分割 > 显著目标检测/跟踪。
- 模态要求：必须涉及 visible/RGB 与 infrared/thermal/RGB-T 的特征交互或任务模型融合。
- 工程要求：优先有公开代码仓库；未确认代码的论文只放补充表。
- 等级标注：CVPR/ICCV/ECCV/ACM MM 记为 CCF-A；AAAI/IJCAI 记为 CCF-B；TGRS/TNNLS/TITS/Pattern Recognition 属于期刊或 journal article，单独注明“非会议，CCF会议等级不适用”。
- PDF：可公开下载的 PDF 放入 `pdfs/`；ACM/IEEE 等不可直接下载的，记录原因或保留网页/代码链接。

## 本地文件

- `pdfs/`：已下载 PDF。
- `notes/`：后续逐篇精读笔记。
- `download_status.tsv`：PDF 下载状态与来源。
- `README.md`：当前筛选和研究总结。

## 一、最直接相关：RGB/IR 检测网络

| 优先级 | 论文 | 来源 | 等级 | 年份 | 任务 | 公开代码 | PDF | 结构关键词 | 对当前 YOLOv12-RGBT 的直接价值 |
| -- | -- | -- | -- | --: | -- | -- | -- | -- | -- |
| 1 | Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection (CMM) | CVPR | CCF-A | 2024 | RGB-T pedestrian detection | https://github.com/ssbin0914/Causal-Mode-Multiplexer | `pdfs/2024_CVPR_CMM.pdf` | modality bias, causal mode, counterfactual intervention, RGB/T visibility modes | 很适合支撑“热红外并非总是正收益”的论文动机；可解释火情数据中 smoke/fire/person 对 RGB/IR 依赖不同，提示后续模块应做可靠性选择而非强融合 |
| 2 | DAMSDet: Dynamic Adaptive Multispectral Detection Transformer with Competitive Query Selection and Adaptive Feature Fusion | ECCV | CCF-A | 2024 | infrared-visible object detection | https://github.com/gjj45/DAMSDet | `pdfs/2024_ECCV_DAMSDet.pdf` | competitive query selection, deformable cross-attention, adaptive feature fusion, misalignment | 检测方向非常直接；其“对象级选择基础模态 + deformable cross-attention 聚合多层语义”的思路，可作为 SGMC/P5 引导 P345 的高等级参考 |
| 3 | Multispectral Object Detection via Cross-Modal Conflict-Aware Learning (CALNet) | ACM MM Oral | CCF-A | 2023 | RGB-IR object detection / DroneVehicle | https://github.com/hexiao0275/CALNet-Dronevehicle | ACM 403，未下载 | Cross-Modal Conflict Rectification, Selected Cross-modal Fusion, conflict-aware feature fusion | 最贴合当前负迁移问题：RGB/IR 特征存在冲突，不应简单 concat 或强替换；可支撑 BidirLiCMA@P5 过拟合/收益不稳的解释 |
| 4 | Multimodal Object Detection via Probabilistic Ensembling (ProbEn) | ECCV | CCF-A | 2022 | RGB-T object detection | https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling | `pdfs/2022_ECCV_ProbEn.pdf` | probabilistic late fusion, modality uncertainty, detection-level ensembling | 更偏检测后融合，但“显式建模模态不确定性”很有价值；可作为 head-level 或 ensemble baseline 的相关工作 |
| 5 | C²Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection | IEEE TGRS | 非会议；CCF会议等级不适用 | 2024 | RGB-IR object detection | https://github.com/yuanmaoxun/C2Former | `pdfs/2024_TGRS_C2Former.pdf` | inter-modality cross-attention, adaptive feature sampling, calibrated complementary feature | 和你尝试的 BidirLiCMA@P5 最像；重点看它如何降采样 attention、如何避免全局交叉注意力成本和误校准 |
| 6 | ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection | Pattern Recognition | 非会议；CCF会议等级不适用 | 2024 | multispectral object detection | https://github.com/chanchanchan97/ICAFusion | `pdfs/2024_PatternRecognition_ICAFusion.pdf` | dual cross-attention, iterative feature fusion, parameter sharing | 可以作为“轻量迭代式交互”的参考；比一次性重注意力替换更符合你现在降低过拟合风险的需求 |
| 7 | TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection | IEEE TNNLS | 非会议；CCF会议等级不适用 | 2025 | RGB-T pedestrian detection | https://github.com/XueZ-phd/TFDet | `pdfs/2025_TNNLS_TFDet.pdf` | target-aware fusion, false-positive suppression, feature contrast | 对火情检测很实用：你的 smoke/fire/person 中背景误检和类别干扰明显，target-aware/FP suppression 的叙事比“更复杂注意力”更稳 |

## 二、分割 / SOD：可迁移的特征交互模块

这些论文不是 bbox 检测，但结构上更接近“多尺度特征校准、双分支交互、可靠性选择”，适合借鉴模块设计与论文叙事。

| 优先级 | 论文 | 来源 | 等级 | 年份 | 任务 | 公开代码 | PDF | 结构关键词 | 可借鉴点 |
| -- | -- | -- | -- | --: | -- | -- | -- | -- | -- |
| 1 | AMDANet: Attention-Driven Multi-Perspective Discrepancy Alignment for RGB-Infrared Image Fusion and Segmentation | ICCV | CCF-A | 2025 | RGB-Infrared semantic segmentation + feature alignment | https://github.com/Zhonghaifeng6/AMDANet | `pdfs/2025_ICCV_AMDANet.pdf` | feature discrepancy alignment, local/global alignment, semantic consistency inference | 支持“差异压制/语义一致性”思路；可用来包装 SGMC 或 P5-guided calibration，而不是强行深层替换 |
| 2 | OpenRSS: Open-Vocabulary RGB-Thermal Semantic Segmentation | ECCV | CCF-A | 2024 | RGB-T semantic segmentation | https://github.com/SXDR/OpenRSS | `pdfs/2024_ECCV_OpenRSS.pdf` | thermal prompt, dynamic LoRA, SAM/CLIP adaptation | 适合借鉴“轻量 adapter/提示注入”；说明不一定要改大 backbone，也可以用参数高效的方式注入热红外信息 |
| 3 | CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers | IEEE TITS / arXiv | 非会议；CCF会议等级不适用 | 2023 | RGB-X semantic segmentation, includes RGB-T | https://github.com/huaaaliu/RGBX_Semantic_Segmentation | `pdfs/2023_TITS_CMX.pdf` | Cross-Modal Feature Rectification, Feature Fusion Module, cross-attention | 非常适合当前 SGMC：先 rectification，再 fusion；可支撑“P5 生成校准权重，残差修正 P3/P4/P5” |
| 4 | XMSNet: Towards Cross-Modal Semantic Understanding for RGB-D and RGB-T Semantic Segmentation | ACM MM | CCF-A | 2023 | RGB-D/RGB-T semantic segmentation | 代码需继续复核 | `pdfs/2023_ACMMM_XMSNet.pdf` | cross-modal semantic understanding, modality interaction | 可借鉴分割任务中的语义一致性监督，适合支持 P3 aux 的语义约束叙事 |
| 5 | Samba: A Unified Mamba-based Framework for General Salient Object Detection | CVPR Highlight | CCF-A | 2025 | RGB/RGB-D/RGB-T salient object detection | https://github.com/Jia-hao999/Samba | `pdfs/2025_CVPR_Samba.pdf` | Mamba, spatial neighboring scanning, context-aware upsampling | 不是双分支检测主线，但可为 neck/head 的轻量全局上下文模块提供参考 |
| 6 | Saliency Prototype for RGB-D and RGB-T Salient Object Detection (SPNet) | ACM MM | CCF-A | 2023 | RGB-D/RGB-T SOD | https://github.com/ZZ2490/SPNet | ACM PDF 未下载 | saliency prototype, auxiliary modality quality weighting, semantic enhancement | “先估计辅助模态质量，再融合”很适合烟火场景；可迁移为类别/区域级可靠性门控 |

## 三、方法参考：相关但不作为主线引用

| 论文 | 来源 | 等级 | 年份 | 代码 | PDF | 为什么保留 |
| -- | -- | -- | --: | -- | -- | -- |
| Multimodal Token Fusion for Vision Transformers (TokenFusion) | CVPR | CCF-A | 2022 | https://github.com/yikaiw/TokenFusion | `pdfs/2022_CVPR_TokenFusion.pdf` | 不专门做 RGB-T，但“只替换不可靠 token、保留主干结构”的思想适合解释为什么残差门控比强替换更稳 |
| Delivering Arbitrary-Modal Semantic Segmentation (CMNeXt / DeLiVER) | CVPR | CCF-A | 2023 | 公开代码需继续复核 | `pdfs/2023_CVPR_DeLiVER_CMNeXt.pdf` | 任意模态分割，不是专门 RGB-T；可借鉴模态缺失鲁棒性与 adapter 设计 |
| Visible-Thermal UAV Tracking (VTUAV / HMFT) | CVPR | CCF-A | 2022 | 公开代码需继续复核 | `pdfs/2022_CVPR_VTUAV_HMFT.pdf` | 跟踪不是检测，但可借鉴 RGB-T 表征、可见/热红外互补性与数据集叙事 |
| Bridging Search Region Interaction with Template for RGB-T Tracking (TBSI) | CVPR | CCF-A | 2023 | 公开代码需继续复核 | `pdfs/2023_CVPR_TBSI_RGBT_Tracking.pdf` | 跟踪任务；cross-modal interaction 结构可作为方法参考，不建议作为主实验依据 |
| Visual Prompt Multi-Modal Tracking (ViPT) | CVPR | CCF-A | 2023 | 公开代码需继续复核 | `pdfs/2023_CVPR_ViPT_MultiModal_Tracking.pdf` | prompt tuning 思想可支撑轻量适配，但任务不同 |
| USTrack: Unified RGB-T Tracking | IJCAI | CCF-B | 2024 | 公开代码需继续复核 | `pdfs/2024_IJCAI_USTrack.pdf` | 可作为 RGB-T 任务的近期补充，不放检测主线 |

## 四、候选但暂不作为主推

| 论文 | 来源 | 等级 | 年份 | 代码状态 | 原因 |
| -- | -- | -- | --: | -- | -- |
| Translation, Scale and Rotation: Cross-Modal Alignment Meets RGB-Infrared Vehicle Detection (TSRA / TSFADet) | ECCV | CCF-A | 2022 | 未确认官方可用代码 | 非常适合“跨模态错位”动机，但工程复现不如 CMM/DAMSDet/CALNet/C²Former 直接 |
| Attentive Alignment Network for Multispectral Pedestrian Detection (AANet) | ACM MM | CCF-A | 2023 | 未确认公开代码 | alignment + reliability 思想很有价值；若写相关工作可以引用，暂不作为实现参考 |
| M2FNet: Mask-guided Multi-level Fusion for RGB-T Pedestrian Detection | IEEE TMM | 非会议；CCF会议等级不适用 | 2024 | 未确认官方可用代码 | mask-guided multi-level fusion 与检测任务相关；如果后续要写多尺度可靠性/目标区域引导，可作为补充 |

## 五、对当前 YOLOv12-RGBT 实验的直接启发

### 5.1 BidirLiCMA@P5 为什么容易不稳

CMM、CALNet、AANet、C²Former、AMDANet 都指向同一个事实：RGB/IR 深层特征不是天然互补，存在统计偏置、空间错位、语义冲突和模态可靠性差异。直接把 P5 换成双向交叉注意力，可能把另一模态的噪声、错位和类别偏差一起注入主干，导致：

- 验证指标波动，尤其是 smoke/fire/person 收益不一致；
- train loss 继续下降但 val mAP 不涨；
- dropout 只能缓解容量，不能解决语义冲突；
- 深层强融合更容易破坏已经稳定的 P2/P3/P4 表征。

因此，BidirLiCMA 后续更适合做成 **残差、门控、低秩/降采样、延迟开启** 的形式，而不是替换式大模块。

### 5.2 SGMC / P5 引导 P345 的论文支撑

DAMSDet、CMX、AMDANet 都支持“高层语义指导多层特征校准”的方向。更合理的表述是：

- P5 不直接替换 P3/P4/P5，而是生成轻量校准权重；
- 对 P3/P4/P5 做 residual channel calibration；
- gate 初始接近 0，训练早期尽量保持 baseline；
- 分类别观察 fire/person/smoke 是否有一致收益；
- 若只提升 recall 但 precision 掉得多，应考虑 target-aware 或 conflict-aware 约束。

### 5.3 P3 aux 与 P4 C3k2 的叙事

P3 aux 不应只写成“加一个辅助头”，更合适的论文表述是：

- 对浅中层融合特征施加语义监督；
- 减少 RGB/IR 融合后低层纹理特征的语义漂移；
- 让热源、烟雾纹理更早接触检测目标；
- 与 CALNet/AMDANet/CMX 的“冲突抑制、语义一致性、特征校准”思想一致。

P4 C3k2 或 A2C2f 的比较，则更适合写成“neck/backbone 中层去噪与语义增强”的正交消融，而不是单独包装为大创新。

## 六、建议精读顺序

1. `pdfs/2024_CVPR_CMM.pdf`：先读 modality bias 与 causal mode，适合写动机。
2. `pdfs/2024_ECCV_DAMSDet.pdf`：重点看 query selection 与 multispectral deformable cross-attention。
3. CALNet 代码和论文页：重点看 conflict-aware learning 与 selected fusion。
4. `pdfs/2024_TGRS_C2Former.pdf`：重点看 ICA + AFS，和 BidirLiCMA@P5 直接相关。
5. `pdfs/2024_PatternRecognition_ICAFusion.pdf`：重点看 iterative/shared-parameter cross-attention。
6. `pdfs/2023_TITS_CMX.pdf`：重点看 feature rectification，可服务 SGMC。
7. `pdfs/2025_TNNLS_TFDet.pdf`：重点看 target-aware fusion 和 false-positive suppression。

## 七、当前不建议继续纳入的论文类型

- 传统 IVIF：只把 RGB/IR 两张图生成一张融合图，而不服务检测/分割网络的特征交互。
- 只有图像质量评价、无下游检测/分割结构的融合论文。
- 只有 RGB-D、没有 thermal/infrared 的论文，除非结构非常适合借鉴。
- 只有 README、无代码主体、无法复现的仓库。

## 八、可用于论文相关工作的分类

| 类别 | 可引用论文 | 可支撑的观点 |
| -- | -- | -- |
| RGB/IR 检测特征融合 | CMM, DAMSDet, CALNet, ProbEn, C²Former, ICAFusion, TFDet | 检测任务需要特征级跨模态交互，但不能简单强融合 |
| 模态冲突 / 偏置 / 错位 | CMM, CALNet, DAMSDet, AANet, TSRA, AMDANet | RGB/IR 存在统计偏置、语义冲突、空间错位和可靠性差异 |
| 多尺度语义校准 | DAMSDet, CMX, AMDANet, C²Former | 高层语义可以指导多尺度特征，但应采用残差校准和轻量门控 |
| 轻量适配 / prompt / token选择 | OpenRSS, TokenFusion, ViPT | 保留主干结构，用 adapter/prompt/token replacement 注入跨模态信息更稳 |
| 可靠性 / 显著性引导 | CMM, SPNet, Samba, ProbEn, TFDet | 先判断可靠区域/目标区域，再融合，可降低背景噪声和负迁移 |
