# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""MCDet baseline model wrapper.

This file keeps the reproduction outside the main model registry. It maps an MCDet YAML to a YOLOv5n-sized
dual-stream detector with MRCF fusion at P3/P4/P5 and CGAN after neck concatenation layers.
"""

import torch
from ultralytics.models import yolo
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules import (
    CrossModalSemanticPrototypeAttention,
    Detect,
    DMGFusion,
    DMGFusionInit8d,
    DMGFusionPosAlpha,
    FreDFTFusion,
    M2DLocalIlluminationFusion,
    RedundancySuppressedSparseSemanticQueryFusion,
    StackedCrossModalSemanticPrototypeAttention,
)
from ultralytics.nn.modules.conv import Concat
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    DualStreamDetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    yaml_model_load,
)
from ultralytics.utils import LOGGER, RANK

from .modules import LazyContentGuidedAttention, MultidimensionalRepresentationCollaborativeFusion
from .losses import MCDetDualStreamDetectionLoss


class MCDetDualStreamDetectionModel(DualStreamDetectionModel):
    """YOLOv5n dual-stream MCDet reproduction."""

    FUSION_LAYER_INDICES = {"p3": 4, "p4": 6, "p5": 9}

    def __init__(self, cfg="baselines/MCDet/cfg/MCDet-yolov5n-fire-person.yaml", ch=6, nc=None, verbose=True):
        self.mcdet_fusion_stages = set()
        self._mcdet_ready_for_aux = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)

        stages = self.yaml.get("mcdet_fusion_stages", ["p3", "p4", "p5"])
        if isinstance(stages, str):
            stages = [stage.strip() for stage in stages.split(",") if stage.strip()]
        self.mcdet_fusion_stages = set(stages)
        unknown_stages = self.mcdet_fusion_stages - set(self.FUSION_LAYER_INDICES)
        if unknown_stages:
            raise ValueError(f"mcdet_fusion_stages contains unsupported stages: {sorted(unknown_stages)}.")

        for stage_name in sorted(self.mcdet_fusion_stages):
            c_out = self._get_layer_out_channels(self.backbone_rgb[self.FUSION_LAYER_INDICES[stage_name]])
            self.fusion_convs[stage_name] = MultidimensionalRepresentationCollaborativeFusion(c_out)

        self.cgan_after_concat = torch.nn.ModuleDict()
        self.use_cgan = bool(self.yaml.get("mcdet_cgan", True))
        c_p3 = self._get_layer_out_channels(self.backbone_rgb[self.FUSION_LAYER_INDICES["p3"]])
        self.aux_head_rgb = Detect(nc=self.yaml["nc"], ch=[c_p3])
        self.aux_head_ir = Detect(nc=self.yaml["nc"], ch=[c_p3])
        for aux_head in (self.aux_head_rgb, self.aux_head_ir):
            aux_head.stride = torch.tensor([8.0])
            aux_head.inplace = self.inplace
            aux_head.bias_init()
        self._mcdet_ready_for_aux = True

        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        """Use MCDet WIoU for the main detection loss."""
        return MCDetDualStreamDetectionLoss(self, aux_weight=self.aux_loss_weight)

    @staticmethod
    def _get_layer_out_channels(layer):
        """Return output channels for YOLOv5 C3, YOLOv12 C2f/A2C2f/C3k2 and Conv layers."""
        if hasattr(layer, "cv2_rgb"):
            return layer.cv2_rgb.conv.out_channels
        if hasattr(layer, "cv3") and hasattr(layer.cv3, "conv"):
            return layer.cv3.conv.out_channels
        if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"):
            return layer.cv2.conv.out_channels
        if hasattr(layer, "conv"):
            return layer.conv.out_channels
        raise ValueError(f"Cannot determine output channels for {type(layer)}.")

    def _fuse_stage(self, stage_name, rgb_feat, ir_feat, lif_illumination):
        fusion = self.fusion_convs[stage_name]
        if isinstance(fusion, MultidimensionalRepresentationCollaborativeFusion):
            return fusion(rgb_feat, ir_feat)
        if isinstance(fusion, M2DLocalIlluminationFusion):
            return fusion(rgb_feat, ir_feat, lif_illumination)
        if isinstance(
            fusion,
            (
                DMGFusion,
                DMGFusionPosAlpha,
                DMGFusionInit8d,
                FreDFTFusion,
                CrossModalSemanticPrototypeAttention,
                RedundancySuppressedSparseSemanticQueryFusion,
                StackedCrossModalSemanticPrototypeAttention,
            ),
        ):
            return fusion(rgb_feat, ir_feat)
        return fusion(torch.cat([rgb_feat, ir_feat], dim=1))

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Run dual YOLOv5n backbones, MRCF fusion and CGAN-enhanced FPN/PAN."""
        del profile, visualize, embed
        assert x.shape[1] == 6, f"Expected 6-channel input, got {x.shape[1]}"
        x_ir = x[:, :3, ...]
        x_rgb = x[:, 3:, ...]

        feats_rgb, feats_ir = self._forward_both_backbones(x_rgb, x_ir)
        lif_illumination = self.lif_gate(x_rgb) if self.lif_gate is not None else None

        if self.training and getattr(self, "use_aux_head", True) and getattr(self, "_mcdet_ready_for_aux", False):
            self._aux_rgb = self.aux_head_rgb([feats_rgb["p3"]])
            self._aux_ir = self.aux_head_ir([feats_ir["p3"]])

        fused = {}
        for stage_name in self.FUSION_LAYER_INDICES:
            fused[stage_name] = self._fuse_stage(
                stage_name,
                feats_rgb[stage_name],
                feats_ir[stage_name],
                lif_illumination,
            )
            if stage_name in self.physical_guidance_stages:
                fused[stage_name] = self.physical_guidance[stage_name](fused[stage_name], x_rgb, x_ir)

        y = [None] * (max(self.FUSION_LAYER_INDICES.values()) + 1)
        for stage_name, layer_idx in self.FUSION_LAYER_INDICES.items():
            y[layer_idx] = fused[stage_name]

        x = fused["p5"]
        for module in self.head:
            if module.f != -1:
                x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
            x = module(x)
            if getattr(self, "use_cgan", False) and isinstance(module, Concat) and hasattr(self, "cgan_after_concat"):
                key = str(module.i)
                if key not in self.cgan_after_concat:
                    self.cgan_after_concat[key] = LazyContentGuidedAttention()
                x = self.cgan_after_concat[key](x)
            y.append(x if module.i in self.save else None)

        return x


class MCDetYOLO(YOLO):
    """YOLO wrapper that maps MCDet dual-stream YAMLs to the MCDet baseline model."""

    @property
    def task_map(self):
        """Map detection to MCDet while leaving other Ultralytics tasks unchanged."""

        def detect_model(cfg, *args, **kwargs):
            if cfg.get("dual_stream", False):
                return MCDetDualStreamDetectionModel(cfg, *args, **kwargs)
            return DetectionModel(cfg, *args, **kwargs)

        class MCDetDetectionTrainer(yolo.detect.DetectionTrainer):
            """Detection trainer that rebuilds MCDet during train()."""

            def get_model(self, cfg=None, weights=None, verbose=True):
                input_mode = self.data.get("input_mode", "dual_input")
                ch = 6 if input_mode == "dual_input" else 3
                yaml_cfg = yaml_model_load(cfg) if isinstance(cfg, str) else cfg
                use_dual_stream = yaml_cfg.get("dual_stream", False) if yaml_cfg else False

                if use_dual_stream and input_mode == "dual_input":
                    model = MCDetDualStreamDetectionModel(
                        cfg,
                        ch=ch,
                        nc=self.data["nc"],
                        verbose=verbose and RANK == -1,
                    )
                else:
                    model = DetectionModel(cfg, ch=ch, nc=self.data["nc"], verbose=verbose and RANK == -1)
                if weights:
                    model.load(weights)
                return model

        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": detect_model,
                "trainer": MCDetDetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }
