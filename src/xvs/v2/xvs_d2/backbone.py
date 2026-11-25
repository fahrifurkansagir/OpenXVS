"""Stride-2 Modified StarNet-P2 Backbone & Nearest-Upsample FPN-P2 Neck (`XVS-D2`, `DR-01`).

Implements `SP2_M2_MODELS_DESIGN.md` §3.2 (`V2-M2L-2`):
- 4 stages at strides `[2, 4, 8, 16]` (never collapsing to stride 32).
- `StarBlock`: `DWConv7x7 -> BN -> dual PW(C -> rC) [SiLU * linear] -> PW(rC -> C) -> BN -> DWConv7x7 + residual`.
- Top-down `FPN-P2` uses `F.interpolate(..., scale_factor=2.0, mode="nearest")` (zero `align_corners`!)
  merging `S4 -> S3 -> UnrolledConvGRU(S2) -> S1` into stride-2 `F_P2 [B, C_fpn, H/2, W/2]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from xvs.v2.xvs_d2.stem import ResidualHaarStem
from xvs.v2.xvs_d2.temporal import DepthwiseSeparableConv2d, UnrolledConvGRUCell


class StarBlock(nn.Module):
    """StarNet element-wise multiplicative polynomial block (`Ma et al., CVPR 2024`; `DR-01:56`)."""

    def __init__(self, dim: int, expand_ratio: int = 2) -> None:
        super().__init__()
        mid = int(dim) * int(expand_ratio)
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.f1 = nn.Conv2d(dim, mid, kernel_size=1, stride=1, padding=0, bias=True)
        self.f2 = nn.Conv2d(dim, mid, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.ReLU6(inplace=False)
        self.g = nn.Conv2d(mid, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.dw2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.bn1(self.dw1(x))
        y = self.act(self.f1(y)) * self.f2(y)
        y = self.dw2(self.bn2(self.g(y)))
        return residual + y


class DownsampleConv(nn.Module):
    """Stride-2 3x3 stage transition (`DR-01:56`)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class StarNetP2Backbone(nn.Module):
    """Four-stage stride `[2, 4, 8, 16]` StarNet-P2 encoder with lossless Haar stem."""

    def __init__(
        self,
        widths: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        expand_ratio: int = 2,
        in_channels: int = 6,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (int(w) for w in widths)
        d1, d2, d3, d4 = (int(d) for d in depths)
        self.widths = (c1, c2, c3, c4)
        self.depths = (d1, d2, d3, d4)
        self.expand_ratio = int(expand_ratio)

        self.stem = ResidualHaarStem(in_channels=in_channels, out_channels=c1)
        self.stage1 = nn.Sequential(*[StarBlock(c1, expand_ratio=self.expand_ratio) for _ in range(d1)])
        self.down2 = DownsampleConv(c1, c2)
        self.stage2 = nn.Sequential(*[StarBlock(c2, expand_ratio=self.expand_ratio) for _ in range(d2)])
        self.down3 = DownsampleConv(c2, c3)
        self.stage3 = nn.Sequential(*[StarBlock(c3, expand_ratio=self.expand_ratio) for _ in range(d3)])
        self.down4 = DownsampleConv(c3, c4)
        self.stage4 = nn.Sequential(*[StarBlock(c4, expand_ratio=self.expand_ratio) for _ in range(d4)])

    def forward(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.stage1(self.stem(x_t))
        s2 = self.stage2(self.down2(s1))
        s3 = self.stage3(self.down3(s2))
        s4 = self.stage4(self.down4(s3))
        return s1, s2, s3, s4


class FPNP2Neck(nn.Module):
    """Top-down FPN-P2 neck using `Resize(mode="nearest")` + stride-4 unrolled ConvGRU (`DR-01:61`)."""

    def __init__(
        self,
        widths: tuple[int, int, int, int],
        c_fpn: int,
        c_gru: int,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (int(w) for w in widths)
        self.c_fpn = int(c_fpn)
        self.c_gru = int(c_gru)

        self.lat1 = nn.Conv2d(c1, self.c_fpn, kernel_size=1, bias=False)
        self.lat2 = nn.Conv2d(c2, self.c_fpn, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(c3, self.c_fpn, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(c4, self.c_fpn, kernel_size=1, bias=False)

        self.conv_gru = UnrolledConvGRUCell(self.c_fpn, self.c_gru)
        self.gru_to_fpn = (
            nn.Identity()
            if self.c_gru == self.c_fpn
            else nn.Conv2d(self.c_gru, self.c_fpn, kernel_size=1, bias=False)
        )

        self.smooth3 = nn.Sequential(
            nn.Conv2d(self.c_fpn, self.c_fpn, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.c_fpn),
            nn.ReLU6(inplace=False),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(self.c_fpn, self.c_fpn, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.c_fpn),
            nn.ReLU6(inplace=False),
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(self.c_fpn, self.c_fpn, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.c_fpn),
            nn.ReLU6(inplace=False),
        )

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
        s4: torch.Tensor,
        h_prev_warped: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        p4 = self.lat4(s4)
        p3 = self.smooth3(self.lat3(s3) + F.interpolate(p4, scale_factor=2.0, mode="nearest"))
        s2_fused = self.lat2(s2) + F.interpolate(p3, scale_factor=2.0, mode="nearest")
        h_next = self.conv_gru(s2_fused, h_prev_warped)
        p2_s4 = self.smooth2(self.gru_to_fpn(h_next))
        f_p2 = self.smooth1(self.lat1(s1) + F.interpolate(p2_s4, scale_factor=2.0, mode="nearest"))
        pyramid = {"p2": f_p2, "p3": p2_s4, "p4": p3}
        return f_p2, h_next, pyramid
