# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""MCDet modules implemented as an isolated baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mamba_ssm.modules.mamba_simple import Mamba
from torchvision.ops import DeformConv2d

from ultralytics.nn.modules import Conv


class DCNv2(nn.Module):
    """Modulated deformable convolution used by MCDet TSFF and CGAN."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        num_points = kernel_size * kernel_size
        self.offset_mask = nn.Conv2d(c_in, 3 * num_points, kernel_size, stride=stride, padding=padding)
        self.dcn = DeformConv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        with autocast(enabled=False):
            x_fp32 = x.float()
            offset_mask = self.offset_mask(x_fp32)
            offset_channels = 2 * self.dcn.kernel_size[0] * self.dcn.kernel_size[1]
            offset = offset_mask[:, :offset_channels]
            mask = offset_mask[:, offset_channels:].sigmoid()
            out = self.dcn(x_fp32, offset, mask)
        return out.to(out_dtype)


class BidirectionalVisualStateSpaceModule(nn.Module):
    """MCDet BVSSM using Mamba state-space blocks on forward and reverse modality sequences."""

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.vis_proj = Conv(channels, channels, 1)
        self.ir_proj = Conv(channels, channels, 1)
        self.vis_dw = Conv(channels, channels, 3, g=channels)
        self.ir_dw = Conv(channels, channels, 3, g=channels)
        self.forward_vssm = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.reverse_vssm = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm_vis = nn.LayerNorm(channels)
        self.norm_ir = nn.LayerNorm(channels)
        self.weight_vis = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.weight_ir = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.out = Conv(channels * 2, channels, 1, act=False)

    @staticmethod
    def _to_sequence(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_feature(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], height, width)

    def forward(self, x_vis: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x_vis.shape
        vis_seq = self._to_sequence(self.vis_dw(self.vis_proj(x_vis)))
        ir_seq = self._to_sequence(self.ir_dw(self.ir_proj(x_ir)))
        concat_seq = torch.cat([vis_seq, ir_seq], dim=1)

        # Mamba's selective scan is numerically fragile in fp16 on Turing GPUs.
        # Keep the state-space mixing in fp32 while allowing the surrounding YOLO graph to use AMP.
        with autocast(enabled=False):
            concat_seq_fp32 = concat_seq.float()
            forward_seq = self.forward_vssm(concat_seq_fp32)
            reverse_seq = self.reverse_vssm(torch.flip(concat_seq_fp32, dims=[1]))
            mixed = 0.5 * (forward_seq + torch.flip(reverse_seq, dims=[1]))

        vis_mixed, ir_mixed = mixed.split(h * w, dim=1)
        vis_feat = self._to_feature(self.norm_vis(vis_mixed), h, w).to(x_vis.dtype) * self.weight_vis.to(x_vis.dtype)
        ir_feat = self._to_feature(self.norm_ir(ir_mixed), h, w).to(x_ir.dtype) * self.weight_ir.to(x_ir.dtype)
        return self.out(torch.cat([vis_feat, ir_feat], dim=1)) + x_vis + x_ir


class TwoPropertySpectralFeatureFusion(nn.Module):
    """MCDet TSFF with DCNv2 local extraction and channel-wise RGB/IR mixing."""

    def __init__(self, channels: int):
        super().__init__()
        self.vis_local = nn.Sequential(
            DCNv2(channels, channels, 3),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.ir_local = nn.Sequential(
            DCNv2(channels, channels, 3),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.post_mix_dcn = nn.Sequential(
            DCNv2(channels, channels, 3),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.group_mix = nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels, bias=False)
        self.group_bn = nn.BatchNorm2d(channels)
        self.fcom = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    @staticmethod
    def _interleave_channels(x_vis: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([x_vis, x_ir], dim=2)
        return stacked.flatten(1, 2)

    def forward(self, x_vis: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        vis = self.vis_local(x_vis)
        ir = self.ir_local(x_ir)
        mixed = self.group_bn(self.group_mix(self._interleave_channels(vis, ir)))
        mixed = F.silu(mixed, inplace=True)
        mixed = self.post_mix_dcn(mixed)
        return mixed * self.fcom(mixed)


class FeatureRefinementModule(nn.Module):
    """MCDet FRM: foreground-mask guided channel refinement."""

    def __init__(self, channels: int):
        super().__init__()
        self.seg = nn.Sequential(Conv(channels, channels, 3), Conv(channels, channels, 3), nn.Conv2d(channels, 1, 1))
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.seg(x))
        mask_feat = mask.expand_as(x)
        x_vec = F.normalize(F.adaptive_avg_pool2d(x, 1), dim=1)
        mask_vec = F.normalize(F.adaptive_avg_pool2d(mask_feat, 1), dim=1)
        cosine_vector = x_vec * mask_vec
        return x * self.proj(cosine_vector)


class MultidimensionalRepresentationCollaborativeFusion(nn.Module):
    """MCDet MRCF block combining BVSSM, TSFF and FRM."""

    def __init__(self, channels: int):
        super().__init__()
        self.bvssm = BidirectionalVisualStateSpaceModule(channels)
        self.tsff = TwoPropertySpectralFeatureFusion(channels)
        self.frm = FeatureRefinementModule(channels)

    def forward(self, x_vis: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        return self.frm(self.bvssm(x_vis, x_ir) + self.tsff(x_vis, x_ir))


class ContentGuidedAttention(nn.Module):
    """MCDet CGAN block for FPN/PAN concatenation features."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.spatial_offset = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.spatial_dcn = DCNv2(2, 1, 3)
        self.spatial_out = nn.Sequential(nn.BatchNorm2d(1), nn.Sigmoid())
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_pool = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        spatial_weight = self.spatial_out(
            self.spatial_dcn(spatial_pool + torch.tanh(self.spatial_offset(spatial_pool)))
        )
        spatial_weight = spatial_weight.expand_as(x)
        channel_weight = self.channel_attn(x).expand_as(x)
        gate = self.gate(torch.cat([channel_weight, spatial_weight], dim=1))
        fused_weight = gate * channel_weight + (1.0 - gate) * spatial_weight
        return x + x * fused_weight


class LazyContentGuidedAttention(nn.Module):
    """Create CGAN after the first concat because parsed YOLO heads only know channels at runtime."""

    def __init__(self):
        super().__init__()
        self.block = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block is None:
            self.block = ContentGuidedAttention(int(x.shape[1])).to(device=x.device)
        return self.block(x)
