# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""AFDNet fusion modules kept inside the baseline directory."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv


class HaarWavelet2d(nn.Module):
    """Fixed one-level Haar transform used by the AFDNet baseline."""

    def __init__(self):
        super().__init__()
        analysis = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5], [0.5, -0.5]],
                [[0.5, 0.5], [-0.5, -0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        synthesis = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5], [0.5, -0.5]],
                [[0.5, 0.5], [-0.5, -0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("analysis_kernel", analysis, persistent=False)
        self.register_buffer("synthesis_kernel", synthesis, persistent=False)

    @staticmethod
    def _expand_kernel(kernel: torch.Tensor, channels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return kernel.to(device=device, dtype=dtype).repeat(channels, 1, 1, 1)

    def dwt(self, x: torch.Tensor):
        """Split a feature map into LL, LH, HL and HH subbands."""
        height, width = x.shape[-2:]
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        channels = x.shape[1]
        weight = self._expand_kernel(self.analysis_kernel, channels, x.dtype, x.device)
        coeffs = F.conv2d(x, weight, stride=2, groups=channels)
        coeffs = coeffs.view(x.shape[0], channels, 4, coeffs.shape[-2], coeffs.shape[-1])
        ll, lh, hl, hh = coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]
        return ll, lh, hl, hh, (height, width)

    def idwt(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, shape):
        """Reconstruct a feature map from LL, LH, HL and HH subbands."""
        channels = ll.shape[1]
        coeffs = torch.stack([ll, lh, hl, hh], dim=2).reshape(ll.shape[0], channels * 4, ll.shape[-2], ll.shape[-1])
        weight = self._expand_kernel(self.synthesis_kernel, channels, ll.dtype, ll.device)
        out = F.conv_transpose2d(coeffs, weight, stride=2, groups=channels)
        return out[..., : shape[0], : shape[1]]


class ThermalGuidedLowFrequencyAggregation(nn.Module):
    """Use thermal contrast to guide low-frequency RGB/IR fusion."""

    def __init__(self, channels: int, reduction: int = 4, contrast_kernel: int = 5):
        super().__init__()
        del reduction
        padding = contrast_kernel // 2
        self.ir_spatial = nn.Sequential(
            nn.Conv2d(2, 1, contrast_kernel, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        self.alpha_fc = nn.Sequential(nn.Linear(channels, channels, bias=True), nn.Sigmoid())
        self.smooth = Conv(channels, channels, 3)

    @staticmethod
    def _local_std(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        padding = kernel_size // 2
        avg_square = F.avg_pool2d(x * x, kernel_size, stride=1, padding=padding)
        square_avg = F.avg_pool2d(x, kernel_size, stride=1, padding=padding) ** 2
        variance = F.relu(avg_square - square_avg)
        return torch.sqrt(variance + 1e-6)

    def forward(self, ll_rgb: torch.Tensor, ll_ir: torch.Tensor) -> torch.Tensor:
        ir_avg = ll_ir.mean(dim=1, keepdim=True)
        ir_max = ll_ir.amax(dim=1, keepdim=True)
        ir_mask = self.ir_spatial(torch.cat([ir_avg, ir_max], dim=1))
        ir_enhanced = ll_ir * ir_mask

        sigma_ir = self._local_std(ll_ir)
        alpha = self.alpha_fc(F.adaptive_avg_pool2d(sigma_ir, 1).flatten(1)).view(
            ll_ir.shape[0], ll_ir.shape[1], 1, 1
        )
        fused_low = alpha * ir_enhanced + (1.0 - alpha) * ll_rgb
        return self.smooth(fused_low)


class SmokeMaskedHighFrequencyRestoration(nn.Module):
    """Blend RGB and IR high-frequency details with a smoke-response gate."""

    def __init__(self, channels: int):
        super().__init__()
        in_channels = channels * 3
        self.conv_d1 = nn.Conv2d(in_channels, channels, 3, padding=1, dilation=1, groups=channels, bias=False)
        self.conv_d3 = nn.Conv2d(in_channels, channels, 3, padding=3, dilation=3, groups=channels, bias=False)
        self.conv_d5 = nn.Conv2d(in_channels, channels, 3, padding=5, dilation=5, groups=channels, bias=False)
        self.mask = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        lh_rgb: torch.Tensor,
        hl_rgb: torch.Tensor,
        hh_rgb: torch.Tensor,
        lh_ir: torch.Tensor,
        hl_ir: torch.Tensor,
        hh_ir: torch.Tensor,
    ):
        vis_high = torch.cat([lh_rgb, hl_rgb, hh_rgb], dim=1)
        multi_scale_vis = torch.cat(
            [self.conv_d1(vis_high), self.conv_d3(vis_high), self.conv_d5(vis_high)],
            dim=1,
        )
        occ_mask = self.mask(-multi_scale_vis)
        keep_rgb = 1.0 - occ_mask
        return (
            keep_rgb * lh_rgb + occ_mask * lh_ir,
            keep_rgb * hl_rgb + occ_mask * hl_ir,
            keep_rgb * hh_rgb + occ_mask * hh_ir,
        )


class AsymmetricFrequencyDecoupledFusion(nn.Module):
    """One-stage AFDNet RGB/IR fusion block."""

    def __init__(self, channels: int, residual_scale: float = 0.2):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        self.wavelet = HaarWavelet2d()
        self.tla = ThermalGuidedLowFrequencyAggregation(channels)
        self.shr = SmokeMaskedHighFrequencyRestoration(channels)
        self.residual_proj = Conv(channels * 2, channels, 1, act=False)
        self.residual_scale = float(residual_scale)
        self.out = nn.ReLU(inplace=True)

    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        ll_rgb, lh_rgb, hl_rgb, hh_rgb, shape = self.wavelet.dwt(x_rgb)
        ll_ir, lh_ir, hl_ir, hh_ir, _ = self.wavelet.dwt(x_ir)

        ll = self.tla(ll_rgb, ll_ir)
        lh, hl, hh = self.shr(lh_rgb, hl_rgb, hh_rgb, lh_ir, hl_ir, hh_ir)
        reconstructed = self.wavelet.idwt(ll, lh, hl, hh, shape)
        residual = self.residual_proj(torch.cat([x_rgb, x_ir], dim=1))
        return self.out(reconstructed + self.residual_scale * residual)
