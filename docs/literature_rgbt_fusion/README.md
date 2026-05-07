# RGB-T Feature Interaction Literature Notes

更新日期：2026-05-07

本目录聚焦 **红外/热红外与可见光的特征级交互与融合**。这里的“融合”不是把 RGB 图像和红外图像合成一张图，而是 RGB/IR 双分支 backbone、cross-modal attention、feature calibration、feature alignment、reliability/gating、prompt/adapter 等网络结构，最终服务检测、分割或显著目标检测任务。

## 检索口径

- 主任务优先级：目标检测 > 语义分割 > 显著目标检测/跟踪。
- 模态要求：必须涉及 visible/RGB 与 infrared/thermal/RGB-T 的特征交互或任务模型融合。
- 工程要求：优先有公开代码仓库；未确认代码的论文只放补充表。
- 等级标注：依据《中国计算机学会推荐国际学术会议和期刊目录第七版（2026年3月更新）》核定。CVPR/ICCV/ACM MM/AAAI 为 CCF-A 会议；ECCV/IJCAI 为 CCF-B 会议；TMM 为 CCF-A 期刊；TGRS/TITS/TNNLS/Pattern Recognition 为 CCF-B 期刊。
- PDF：可公开下载的 PDF 放入 `pdfs/`；ACM/IEEE 等不可直接下载的，记录原因或保留网页/代码链接。

## 本地文件

- `pdfs/`：已下载 PDF。
- `notes/`：后续逐篇精读笔记。
- `download_status.tsv`：PDF 下载状态与来源。
- `README.md`：当前筛选和研究总结。

<br />

## 一、最直接相关：RGB/IR 检测网络

| 优先级 | 论文                                                                                                                                                       | 来源                  | 等级       |   年份 | 任务                                                    | 公开代码                                                                                   | PDF                                          | 结构关键词                                                                                          | 对当前 YOLOv12-RGBT 的直接价值                                                                  |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------- | ---: | ----------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1   | Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection (CMM)                                                         | CVPR                | CCF-A 会议 | 2024 | RGB-T pedestrian detection                            | <https://github.com/ssbin0914/Causal-Mode-Multiplexer>                                 | `pdfs/2024_CVPR_CMM.pdf`                     | modality bias, causal mode, counterfactual intervention, RGB/T visibility modes                | 很适合支撑“热红外并非总是正收益”的论文动机；可解释火情数据中 smoke/fire/person 对 RGB/IR 依赖不同，提示后续模块应做可靠性选择而非强融合      |
| 2   | WaveMamba: Wavelet-Driven Mamba Fusion for RGB-Infrared Object Detection                                                                                 | ICCV                | CCF-A 会议 | 2025 | RGB-IR object detection                               | 代码待确认                                                                                  | `pdfs/2025_ICCV_WaveMamba.pdf`               | wavelet decomposition, low-frequency Mamba fusion, gated attention, high-frequency enhancement | 很适合支撑“RGB 与 IR 的优势频段不同”：IR 更适合低频轮廓，RGB 更适合高频纹理；可为你后续做轻量频域/分支特化提供高等级参考                   |
| 3   | Rethinking Multi-modal Object Detection from the Perspective of Mono-Modality Feature Learning (M2D-LIF)                                                 | ICCV                | CCF-A 会议 | 2025 | RGB-IR object detection                               | <https://github.com/Zhao-Tian-yi/M2D-LIF>                                              | `pdfs/2025_ICCV_M2D_LIF.pdf`                 | fusion degradation, mono-modality distillation, local illumination-aware fusion                | 非常贴合当前公平实验结论：强融合可能削弱单模态学习，导致 fusion degradation；可解释 DMG/BidirLiCMA 没有稳定正收益              |
| 4   | Fusion Meets Diverse Conditions: A High-diversity Benchmark and Baseline for UAV-based Multimodal Object Detection with Condition Cues (PCDF / ATR-UMOD) | ICCV                | CCF-A 会议 | 2025 | UAV RGB-IR object detection                           | 代码待确认                                                                                  | `pdfs/2025_ICCV_PCDF_ATR_UMOD.pdf`           | condition cues, prompt-guided dynamic fusion, soft-gating, condition decoupling                | 支持“按场景/成像条件动态分配模态贡献”的叙事；对烟雾、火焰、弱光、过曝等条件差异很有启发                                           |
| 5   | DAMSDet: Dynamic Adaptive Multispectral Detection Transformer with Competitive Query Selection and Adaptive Feature Fusion                               | ECCV                | CCF-B 会议 | 2024 | infrared-visible object detection                     | <https://github.com/gjj45/DAMSDet>                                                     | `pdfs/2024_ECCV_DAMSDet.pdf`                 | competitive query selection, deformable cross-attention, adaptive feature fusion, misalignment | 检测方向非常直接；其“对象级选择基础模态 + deformable cross-attention 聚合多层语义”的思路，可作为 SGMC/P5 引导 P345 的高等级参考 |
| 6   | Multispectral Object Detection via Cross-Modal Conflict-Aware Learning (CALNet)                                                                          | ACM MM Oral         | CCF-A 会议 | 2023 | RGB-IR object detection / DroneVehicle                | <https://github.com/hexiao0275/CALNet-Dronevehicle>                                    | ACM 403，未下载                                  | Cross-Modal Conflict Rectification, Selected Cross-modal Fusion, conflict-aware feature fusion | 最贴合当前负迁移问题：RGB/IR 特征存在冲突，不应简单 concat 或强替换；可支撑 BidirLiCMA\@P5 过拟合/收益不稳的解释                |
| 7   | Translation, Scale and Rotation: Cross-Modal Alignment Meets RGB-Infrared Vehicle Detection (TSRA / TSFADet)                                             | ECCV                | CCF-B 会议 | 2022 | RGB-IR aerial vehicle detection                       | 未确认官方可用代码                                                                              | `pdfs/2022_ECCV_TSRA_TSFADet.pdf`            | feature-level alignment, translation-scale-rotation calibration, modality selection            | 对无人机/俯视 RGB-IR 检测的错位问题很直接；可支撑“跨模态融合前需要校准/选择”的动机                                         |
| 8   | Multimodal Object Detection via Probabilistic Ensembling (ProbEn)                                                                                        | ECCV                | CCF-B 会议 | 2022 | RGB-T object detection                                | <https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling> | `pdfs/2022_ECCV_ProbEn.pdf`                  | probabilistic late fusion, modality uncertainty, detection-level ensembling                    | 更偏检测后融合，但“显式建模模态不确定性”很有价值；可作为 head-level 或 ensemble baseline 的相关工作                      |
| 9   | C²Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection                                                                     | IEEE TGRS           | CCF-B 期刊 | 2024 | RGB-IR object detection                               | <https://github.com/yuanmaoxun/C2Former>                                               | `pdfs/2024_TGRS_C2Former.pdf`                | inter-modality cross-attention, adaptive feature sampling, calibrated complementary feature    | 和你尝试的 BidirLiCMA\@P5 最像；重点看它如何降采样 attention、如何避免全局交叉注意力成本和误校准                           |
| 10  | ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection                                                            | Pattern Recognition | CCF-B 期刊 | 2024 | multispectral object detection                        | <https://github.com/chanchanchan97/ICAFusion>                                          | `pdfs/2024_PatternRecognition_ICAFusion.pdf` | dual cross-attention, iterative feature fusion, parameter sharing                              | 可以作为“轻量迭代式交互”的参考；比一次性重注意力替换更符合你现在降低过拟合风险的需求                                             |
| 11  | TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection                                                                                                | IEEE TNNLS          | CCF-B 期刊 | 2025 | RGB-T pedestrian detection                            | <https://github.com/XueZ-phd/TFDet>                                                    | `pdfs/2025_TNNLS_TFDet.pdf`                  | target-aware fusion, false-positive suppression, feature contrast                              | 对火情检测很实用：你的 smoke/fire/person 中背景误检和类别干扰明显，target-aware/FP suppression 的叙事比“更复杂注意力”更稳   |
| 12  | Pseudo Visible Feature Fine-Grained Fusion for Thermal Object Detection (PFGF)                                                                           | CVPR                | CCF-A 会议 | 2025 | thermal object detection with pseudo-visible features | <https://github.com/liting1018/PFGF>                                                   | `pdfs/2025_CVPR_PFGF.pdf`                    | pseudo-visible features, graph feature fusion, Inter-Mamba, cascade knowledge integration      | 不是严格的 RGB-IR 双输入，但提供“可见光语义/纹理作为红外检测补充”的高等级参考；适合放在补充相关工作，不建议作为主线对照                       |

## 二、分割 / SOD：可迁移的特征交互模块

这些论文不是 bbox 检测，但结构上更接近“多尺度特征校准、双分支交互、可靠性选择”，适合借鉴模块设计与论文叙事。

| 优先级 | 论文                                                                                                               | 来源                | 等级       |   年份 | 任务                                                     | 公开代码                                                     | PDF                          | 结构关键词                                                                                 | 可借鉴点                                                              |
| --- | ---------------------------------------------------------------------------------------------------------------- | ----------------- | -------- | ---: | ------------------------------------------------------ | -------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1   | AMDANet: Attention-Driven Multi-Perspective Discrepancy Alignment for RGB-Infrared Image Fusion and Segmentation | ICCV              | CCF-A 会议 | 2025 | RGB-Infrared semantic segmentation + feature alignment | <https://github.com/Zhonghaifeng6/AMDANet>               | `pdfs/2025_ICCV_AMDANet.pdf` | feature discrepancy alignment, local/global alignment, semantic consistency inference | 支持“差异压制/语义一致性”思路；可用来包装 SGMC 或 P5-guided calibration，而不是强行深层替换     |
| 2   | OpenRSS: Open-Vocabulary RGB-Thermal Semantic Segmentation                                                       | ECCV              | CCF-B 会议 | 2024 | RGB-T semantic segmentation                            | <https://github.com/SXDR/OpenRSS>                        | `pdfs/2024_ECCV_OpenRSS.pdf` | thermal prompt, dynamic LoRA, SAM/CLIP adaptation                                     | 适合借鉴“轻量 adapter/提示注入”；说明不一定要改大 backbone，也可以用参数高效的方式注入热红外信息        |
| 3   | CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers                                        | IEEE TITS / arXiv | CCF-B 期刊 | 2023 | RGB-X semantic segmentation, includes RGB-T            | <https://github.com/huaaaliu/RGBX_Semantic_Segmentation> | `pdfs/2023_TITS_CMX.pdf`     | Cross-Modal Feature Rectification, Feature Fusion Module, cross-attention             | 非常适合当前 SGMC：先 rectification，再 fusion；可支撑“P5 生成校准权重，残差修正 P3/P4/P5” |
| 4   | XMSNet: Towards Cross-Modal Semantic Understanding for RGB-D and RGB-T Semantic Segmentation                     | ACM MM            | CCF-A 会议 | 2023 | RGB-D/RGB-T semantic segmentation                      | 代码需继续复核                                                  | `pdfs/2023_ACMMM_XMSNet.pdf` | cross-modal semantic understanding, modality interaction                              | 可借鉴分割任务中的语义一致性监督，适合支持 P3 aux 的语义约束叙事                              |
| 5   | Samba: A Unified Mamba-based Framework for General Salient Object Detection                                      | CVPR Highlight    | CCF-A 会议 | 2025 | RGB/RGB-D/RGB-T salient object detection               | <https://github.com/Jia-hao999/Samba>                    | `pdfs/2025_CVPR_Samba.pdf`   | Mamba, spatial neighboring scanning, context-aware upsampling                         | 不是双分支检测主线，但可为 neck/head 的轻量全局上下文模块提供参考                            |
| 6   | Saliency Prototype for RGB-D and RGB-T Salient Object Detection (SPNet)                                          | ACM MM            | CCF-A 会议 | 2023 | RGB-D/RGB-T SOD                                        | <https://github.com/ZZ2490/SPNet>                        | ACM PDF 未下载                  | saliency prototype, auxiliary modality quality weighting, semantic enhancement        | “先估计辅助模态质量，再融合”很适合烟火场景；可迁移为类别/区域级可靠性门控                            |

## 三、方法参考：相关但不作为主线引用

| 论文                                                                                           | 来源    | 等级       |   年份 | 代码                                                                             | PDF                                               | 为什么保留                                                                                             | <br />                                           |
| -------------------------------------------------------------------------------------------- | ----- | -------- | ---: | ------------------------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------- | :----------------------------------------------- |
| Multimodal Token Fusion for Vision Transformers (TokenFusion)                                | CVPR  | CCF-A 会议 | 2022 | <https://github.com/yikaiw/TokenFusion>                                        | `pdfs/2022_CVPR_TokenFusion.pdf`                  | 不专门做 RGB-T，但“只替换不可靠 token、保留主干结构”的思想适合解释为什么残差门控比强替换更稳                                             | <br />                                           |
| Delivering Arbitrary-Modal Semantic Segmentation (CMNeXt / DeLiVER)                          | CVPR  | CCF-A 会议 | 2023 | 公开代码需继续复核                                                                      | `pdfs/2023_CVPR_DeLiVER_CMNeXt.pdf`               | 任意模态分割，不是专门 RGB-T；可借鉴模态缺失鲁棒性与 adapter 设计                                                          | <br />                                           |
| Visible-Thermal UAV Tracking (VTUAV / HMFT)                                                  | CVPR  | CCF-A 会议 | 2022 | 公开代码需继续复核                                                                      | `pdfs/2022_CVPR_VTUAV_HMFT.pdf`                   | 跟踪不是检测，但可借鉴 RGB-T 表征、可见/热红外互补性与数据集叙事                                                              | <br />                                           |
| Bridging Search Region Interaction with Template for RGB-T Tracking (TBSI)                   | CVPR  | CCF-A 会议 | 2023 | 公开代码需继续复核                                                                      | `pdfs/2023_CVPR_TBSI_RGBT_Tracking.pdf`           | 跟踪任务；cross-modal interaction 结构可作为方法参考，不建议作为主实验依据                                                 | <br />                                           |
| Visual Prompt Multi-Modal Tracking (ViPT)                                                    | CVPR  | CCF-A 会议 | 2023 | 公开代码需继续复核                                                                      | `pdfs/2023_CVPR_ViPT_MultiModal_Tracking.pdf`     | prompt tuning 思想可支撑轻量适配，但任务不同                                                                     | <br />                                           |
| USTrack: Unified RGB-T Tracking                                                              | IJCAI | CCF-B 会议 | 2024 | 公开代码需继续复核                                                                      | `pdfs/2024_IJCAI_USTrack.pdf`                     | 可作为 RGB-T 任务的近期补充，不放检测主线                                                                          | <br />                                           |
| Bi-directional Adapter for Multimodal Tracking (BAT)                                         | AAAI  | CCF-A 会议 | 2024 | <https://github.com/SparkTempest/BAT>                                          | AAAI 503，未下载                                      | 双向 adapter、参数高效 prompt fusion；可支撑“轻量双向适配比大规模替换更稳”的设计原则                                            | <br />                                           |
| Temporal Adaptive RGBT Tracking with Modality Prompt (TATrack)                               | AAAI  | CCF-A 会议 | 2024 | 代码待确认                                                                          | AAAI 503，未下载                                      | 时序模板 + modality prompt + spatio-temporal interaction；任务不同，但动态模态提示对可靠性门控有参考价值                      | <br />                                           |
| Attribute-based Progressive Fusion Network for RGBT Tracking (APFNet)                        | AAAI  | CCF-A 会议 | 2022 | <https://github.com/yangmengmeng1997/APFNet>                                   | AAAI 503，未下载                                      | 按 challenge attribute 解耦融合分支，再用 SKNet 聚合；可借鉴“按场景属性选择融合路径”                                         | <br />                                           |
| Cross-Modal Object Tracking: Modality-Aware Representations and A Unified Benchmark (MArMOT) | AAAI  | CCF-A 会议 | 2022 | <https://github.com/mmic-lcl/source-code>                                      | `pdfs/2022_AAAI_MArMOT.pdf`                       | RGB/NIR 跨模态跟踪与 modality-aware representation；任务不同，但可作为模态差异建模参考                                    | <br />                                           |
| RGBT Tracking via All-layer Multimodal Interactions with Progressive Fusion Mamba (AINet)    | AAAI  | CCF-A 会议 | 2025 | 代码待确认                                                                          | `pdfs/2025_AAAI_AINet_ProgressiveFusionMamba.pdf` | all-layer interaction + Difference-based Fusion Mamba；可支撑多层交互但需控制复杂度的论述                           | <br />                                           |
| Cross-modulated Attention Transformer for RGBT Tracking (CAFormer)                           | AAAI  | CCF-A 会议 | 2025 | 代码待确认                                                                          | `pdfs/2025_AAAI_CAFormer_RGBT_Tracking.pdf`       | cross-modulated attention、相关性校正、token elimination；和 BidirLiCMA 的“双向注意力”概念接近，但更强调 attention map 质量 | <br />                                           |
| MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking   | CVPR  | CCF-A 会议 | 2025 | <https://github.com/q2479036243/MUST-Multispectral-UAV-Single-Object-Tracking> | `pdfs/2025_CVPR_MUST_UNTrack.pdf`                 | UAV multispectral tracking, spectrum prompt, asymmetric transformer                               | 不是检测，但提供 UAV 多光谱数据和 prompt/光谱背景抑制思路；适合写应用背景或方法参考 |

## 四、候选但暂不作为主推

| 论文                                                                         | 来源       | 等级       |   年份 | 代码状态      | 原因                                                               |
| -------------------------------------------------------------------------- | -------- | -------- | ---: | --------- | ---------------------------------------------------------------- |
| Attentive Alignment Network for Multispectral Pedestrian Detection (AANet) | ACM MM   | CCF-A 会议 | 2023 | 未确认公开代码   | alignment + reliability 思想很有价值；若写相关工作可以引用，暂不作为实现参考               |
| M2FNet: Mask-guided Multi-level Fusion for RGB-T Pedestrian Detection      | IEEE TMM | CCF-A 期刊 | 2024 | 未确认官方可用代码 | mask-guided multi-level fusion 与检测任务相关；如果后续要写多尺度可靠性/目标区域引导，可作为补充 |

## 五、对当前 YOLOv12-RGBT 实验的直接启发

### 5.1 BidirLiCMA\@P5 为什么容易不稳

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

1. `pdfs/2025_ICCV_M2D_LIF.pdf`：先读 fusion degradation 与 mono-modality insufficient learning，最贴合当前 DMG/SGMC 公平复现实验的负结果解释。
2. `pdfs/2024_CVPR_CMM.pdf`：读 modality bias 与 causal mode，适合写“热红外不总是正收益”的动机。
3. `pdfs/2025_ICCV_WaveMamba.pdf`：重点看 RGB/IR 的频域互补、低频 Mamba 和 gated attention，适合支撑“不要简单强融合”。
4. `pdfs/2025_ICCV_PCDF_ATR_UMOD.pdf`：重点看 condition cues 与 prompt-guided dynamic fusion，可服务烟雾/弱光/过曝等条件差异叙事。
5. `pdfs/2024_ECCV_DAMSDet.pdf`：重点看 query selection 与 multispectral deformable cross-attention。
6. CALNet 代码和论文页：重点看 conflict-aware learning 与 selected fusion。
7. `pdfs/2024_TGRS_C2Former.pdf`：重点看 ICA + AFS，和 BidirLiCMA\@P5 直接相关。
8. `pdfs/2022_ECCV_TSRA_TSFADet.pdf`：重点看 feature-level alignment 与 modality selection，适合补充跨模态错位动机。
9. `pdfs/2023_TITS_CMX.pdf`：重点看 feature rectification，可服务 SGMC。
10. `pdfs/2025_TNNLS_TFDet.pdf`：重点看 target-aware fusion 和 false-positive suppression。

## 七、当前不建议继续纳入的论文类型

- 传统 IVIF：只把 RGB/IR 两张图生成一张融合图，而不服务检测/分割网络的特征交互。
- 只有图像质量评价、无下游检测/分割结构的融合论文。
- 只有 RGB-D、没有 thermal/infrared 的论文，除非结构非常适合借鉴。
- 只有 README、无代码主体、无法复现的仓库。

## 八、可用于论文相关工作的分类

| 类别                      | 可引用论文                                                                                    | 可支撑的观点                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| RGB/IR 检测特征融合           | CMM, WaveMamba, M2D-LIF, PCDF, DAMSDet, CALNet, ProbEn, C²Former, ICAFusion, TFDet, PFGF | 检测任务需要特征级跨模态交互，但不能简单强融合                                |
| 模态冲突 / 偏置 / 错位          | CMM, M2D-LIF, CALNet, DAMSDet, AANet, TSRA, AMDANet                                      | RGB/IR 存在统计偏置、语义冲突、空间错位、单模态学习不足和可靠性差异                  |
| 多尺度语义校准                 | DAMSDet, CMX, AMDANet, C²Former, PCDF, WaveMamba                                         | 高层语义、条件提示或频域结构可以指导多尺度特征，但应采用残差校准和轻量门控                  |
| 轻量适配 / prompt / token选择 | OpenRSS, TokenFusion, ViPT, BAT, TATrack, MUST                                           | 保留主干结构，用 adapter/prompt/token replacement 注入跨模态信息更稳    |
| 可靠性 / 显著性引导             | CMM, PCDF, APFNet, SPNet, Samba, ProbEn, TFDet                                           | 先判断可靠区域、目标区域、场景条件或 challenge attribute，再融合，可降低背景噪声和负迁移 |
| 高效序列建模 / Mamba          | WaveMamba, AINet, PFGF, CAFormer                                                         | Mamba/线性序列建模可作为替代高成本全局 attention 的方向，但需要防止过拟合和不稳定融合    |

