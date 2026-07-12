import pytest

from ultralytics.nn.modules.block import C2f, DMGFusionInit8d, DualParallelCrossA2C2f, FreDFTFusion
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DualStreamDetectionModel


def test_y8n_b2_cfg_replaces_p4_c2f_with_dual_parallel_cross_a2c2f():
    """B2-y8n-backbone should preserve P2/P3 innovations and replace only P4 C2f."""
    model = DualStreamDetectionModel("B2-y8n-backbone-fire-person.yaml", nc=2, verbose=False)

    assert model.yaml["p2_fusion"] == "dmg_init8d"
    assert model.yaml["freq_fusion_stages"] == ["p3"]
    assert model.yaml["parallel_cross_a2c2f_stages"] == ["p4"]
    assert isinstance(model.fusion_convs["p2"], DMGFusionInit8d)
    assert isinstance(model.fusion_convs["p3"], FreDFTFusion)
    assert isinstance(model.fusion_convs["p4"], Conv)
    assert isinstance(model.fusion_convs["p5"], Conv)
    assert isinstance(model.backbone_rgb[6], DualParallelCrossA2C2f)
    assert model.backbone_rgb[6] is model.backbone_ir[6]
    assert isinstance(model.backbone_rgb[8], C2f)
    assert model.backbone_rgb[6].cross_rgb[0].attn.area == 4
    assert model.backbone_rgb[6].effective_gamma_rgb().item() == pytest.approx(0.01)
