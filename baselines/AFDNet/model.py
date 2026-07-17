# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""AFDNet baseline model wrapper.

This file intentionally keeps the reproduction code outside the project backbone. It defines a YOLO wrapper that
uses AFDNet only when the baseline training script imports this package.
"""

import torch
from ultralytics.models import yolo
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules import (
    CrossModalSemanticPrototypeAttention,
    DMGFusion,
    DMGFusionInit8d,
    DMGFusionPosAlpha,
    FreDFTFusion,
    M2DLocalIlluminationFusion,
    RedundancySuppressedSparseSemanticQueryFusion,
    StackedCrossModalSemanticPrototypeAttention,
)
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

from .modules import AsymmetricFrequencyDecoupledFusion


def _as_stage_set(value):
    """Normalize a YAML stage field to a set of strings."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return set(value)


class AFDNetDualStreamDetectionModel(DualStreamDetectionModel):
    """Dual-stream YOLO11n model whose selected fusion stages are replaced by AFDNet blocks."""

    VALID_AFD_STAGES = {"p3", "p4", "p5"}

    def __init__(self, cfg="baselines/AFDNet/cfg/AFDNet-y11n-fire-person.yaml", ch=6, nc=None, verbose=True):
        self.afd_fusion_stages = set()
        self.disabled_fusion_stages = set()
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)

        afd_stages = _as_stage_set(self.yaml.get("afd_fusion_stages", ["p3", "p4", "p5"]))
        unknown_afd = afd_stages - self.VALID_AFD_STAGES
        if unknown_afd:
            raise ValueError(f"afd_fusion_stages only supports p3/p4/p5, got {sorted(unknown_afd)}.")

        disabled_stages = _as_stage_set(self.yaml.get("disabled_fusion_stages", []))
        unknown_disabled = disabled_stages - set(self.FUSION_LAYER_INDICES)
        if unknown_disabled:
            raise ValueError(f"disabled_fusion_stages contains unsupported stages: {sorted(unknown_disabled)}.")

        self.afd_fusion_stages = afd_stages
        self.disabled_fusion_stages = disabled_stages

        for stage_name in sorted(self.disabled_fusion_stages):
            if stage_name in self.fusion_convs:
                del self.fusion_convs[stage_name]

        residual_scale = float(self.yaml.get("afd_residual_scale", 0.2))
        for stage_name in sorted(self.afd_fusion_stages):
            c_out = self._get_layer_out_channels(self.backbone_rgb[self.FUSION_LAYER_INDICES[stage_name]])
            self.fusion_convs[stage_name] = AsymmetricFrequencyDecoupledFusion(c_out, residual_scale=residual_scale)

        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Run RGB/IR backbones and AFDNet fusion without modifying the main repository model class."""
        del profile, visualize, embed
        assert x.shape[1] == 6, f"Expected 6-channel input, got {x.shape[1]}"
        x_ir = x[:, :3, ...]
        x_rgb = x[:, 3:, ...]

        feats_rgb, feats_ir = self._forward_both_backbones(x_rgb, x_ir)
        lif_illumination = self.lif_gate(x_rgb) if self.lif_gate is not None else None

        if self.training and getattr(self, "use_aux_head", True):
            self._aux_rgb = self.aux_head_rgb([feats_rgb["p3"]])
            self._aux_ir = self.aux_head_ir([feats_ir["p3"]])

        fused = {}
        for stage_name in self.FUSION_LAYER_INDICES:
            if stage_name in self.disabled_fusion_stages:
                continue

            rgb_feat, ir_feat = feats_rgb[stage_name], feats_ir[stage_name]
            fusion = self.fusion_convs[stage_name]
            if isinstance(fusion, AsymmetricFrequencyDecoupledFusion):
                fused[stage_name] = fusion(rgb_feat, ir_feat)
            elif isinstance(fusion, M2DLocalIlluminationFusion):
                fused[stage_name] = fusion(rgb_feat, ir_feat, lif_illumination)
            elif isinstance(
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
                fused[stage_name] = fusion(rgb_feat, ir_feat)
            else:
                fused[stage_name] = fusion(torch.cat([rgb_feat, ir_feat], dim=1))

            if stage_name in self.physical_guidance_stages:
                fused[stage_name] = self.physical_guidance[stage_name](fused[stage_name], x_rgb, x_ir)

        y = [None] * (max(self.FUSION_LAYER_INDICES.values()) + 1)
        for stage_name, layer_idx in self.FUSION_LAYER_INDICES.items():
            if stage_name in self.disabled_fusion_stages:
                continue
            y[layer_idx] = fused[stage_name]

        x = fused["p5"]
        for module in self.head:
            if module.f != -1:
                x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
            x = module(x)
            y.append(x if module.i in self.save else None)

        return x


class AFDNetYOLO(YOLO):
    """YOLO wrapper that maps detect YAMLs to the AFDNet baseline dual-stream model."""

    @property
    def task_map(self):
        """Map detection to the baseline model while keeping other tasks unchanged."""

        def detect_model(cfg, *args, **kwargs):
            if cfg.get("dual_stream", False):
                return AFDNetDualStreamDetectionModel(cfg, *args, **kwargs)
            return DetectionModel(cfg, *args, **kwargs)

        class AFDNetDetectionTrainer(yolo.detect.DetectionTrainer):
            """Detection trainer that rebuilds AFDNet during train(), not the main dual-stream model."""

            def get_model(self, cfg=None, weights=None, verbose=True):
                input_mode = self.data.get("input_mode", "dual_input")
                ch = 6 if input_mode == "dual_input" else 3
                yaml_cfg = yaml_model_load(cfg) if isinstance(cfg, str) else cfg
                use_dual_stream = yaml_cfg.get("dual_stream", False) if yaml_cfg else False

                if use_dual_stream and input_mode == "dual_input":
                    model = AFDNetDualStreamDetectionModel(
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
                "trainer": AFDNetDetectionTrainer,
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
