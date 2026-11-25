"""Lossless Fixed 2D Haar Wavelet Stem & Dual-Interval Input Contract (`XVS-D2`, `DR-01`).

Implements `SP2_M2_MODELS_DESIGN.md` §3.2 (`V2-M2L-2`):
- Scene normalization `(I - mu) / max(sigma, 8.0)` clamped to `[-5.0, +5.0]`.
- 6-channel primary input tensor `x_t [B, 6, H, W]`:
  `[I_t, I_tm1_warped, I_tm5_warped, (I_t - I_tm1_warped)*R_tm1, (I_t - I_tm5_warped)*R_tm5, R_tm1*R_tm5]`.
- Frozen orthonormal depthwise 2D Haar `Conv2d(6, 24, kernel_size=2, stride=2, groups=6, bias=False)`
  guaranteeing exact Parseval L2 energy preservation `||Haar(x)||_2 == ||x||_2` (`< 1e-5`),
  combined with a 3x3 stride-1 projection and parallel learnable 3x3 stride-2 residual spatial
  branch into `C1` channels (zero `SpaceToDepth` for 100% WebGPU compatibility).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def scene_normalize_frame(
    frame: torch.Tensor,
    min_std: float = 8.0,
    clamp_val: float = 5.0,
) -> torch.Tensor:
    """Normalize grayscale frame `[B, 1, H, W]` or `[B, H, W]` per-frame via `(I - mu)/max(sigma, 8.0)`."""
    if frame.ndim == 3:
        dims = (-2, -1)
    elif frame.ndim == 4:
        dims = (-2, -1)
    else:
        raise ValueError(f"Expected 3D or 4D frame tensor, got shape {tuple(frame.shape)}")

    f32 = frame.to(dtype=torch.float32)
    mean = f32.mean(dim=dims, keepdim=True)
    std = f32.std(dim=dims, unbiased=False, keepdim=True).clamp(min=float(min_std))
    normed = ((f32 - mean) / std).clamp(min=-float(clamp_val), max=float(clamp_val))
    return normed.to(dtype=frame.dtype)


def build_dual_interval_input(
    i_t: torch.Tensor,
    i_tm1_warped: torch.Tensor,
    i_tm5_warped: torch.Tensor,
    r_tm1: torch.Tensor,
    r_tm5: torch.Tensor,
    *,
    already_normalized: bool = False,
) -> torch.Tensor:
    """Construct `x_t [B, 6, H, W]` per `DR-01:48` and `SP2 §3.2`.

    Args:
        i_t: Current grayscale frame `[B, 1, H, W]`.
        i_tm1_warped: Short-lag (`0.1 s`) warped frame `[B, 1, H, W]`.
        i_tm5_warped: Medium-lag (`0.5 s`) warped frame `[B, 1, H, W]`.
        r_tm1: Valid-warp binary mask for `t-1` `[B, 1, H, W]` in `{0, 1}`.
        r_tm5: Valid-warp binary mask for `t-5` `[B, 1, H, W]` in `{0, 1}`.
        already_normalized: Skip scene normalization if inputs are already normalized.
    """
    for name, tensor in (
        ("i_t", i_t),
        ("i_tm1_warped", i_tm1_warped),
        ("i_tm5_warped", i_tm5_warped),
        ("r_tm1", r_tm1),
        ("r_tm5", r_tm5),
    ):
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B, 1, H, W], got {tuple(tensor.shape)}")

    if already_normalized:
        c0 = i_t
        c1 = i_tm1_warped
        c2 = i_tm5_warped
    else:
        c0 = scene_normalize_frame(i_t)
        c1 = scene_normalize_frame(i_tm1_warped)
        c2 = scene_normalize_frame(i_tm5_warped)

    m1 = r_tm1.to(dtype=c0.dtype)
    m5 = r_tm5.to(dtype=c0.dtype)
    c3 = (c0 - c1) * m1
    c4 = (c0 - c2) * m5
    c5 = m1 * m5
    return torch.cat([c0, c1, c2, c3, c4, c5], dim=1)


class FixedHaarWavelet2D(nn.Module):
    """Frozen orthonormal 2D Haar wavelet downsampler (`Conv2d(C, 4C, k=2, s=2, groups=C)`).

    Exact orthogonal basis (`LL, LH, HL, HH`) with Frobenius norm `1.0` per filter guarantees
    `||FixedHaarWavelet2D(x)||_2 == ||x||_2` on any even-sized `(H, W)` tensor.
    """

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels * 4,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=self.in_channels,
            bias=False,
        )
        haar_4 = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],    # LL (approximation)
                [[0.5, -0.5], [0.5, -0.5]],  # LH (horizontal detail)
                [[0.5, 0.5], [-0.5, -0.5]],  # HL (vertical detail)
                [[0.5, -0.5], [-0.5, 0.5]],  # HH (diagonal detail)
            ],
            dtype=torch.float32,
        )  # [4, 2, 2]
        weight = haar_4.unsqueeze(1).repeat(self.in_channels, 1, 1, 1)  # [4*C, 1, 2, 2]
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualHaarStem(nn.Module):
    """Lossless Fixed Haar Wavelet Stem + Parallel Learnable Residual Spatial Branch (`DR-01:55`).

    Maps `[B, 6, H, W]` -> `[B, C1, H/2, W/2]` without `SpaceToDepth`.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.haar = FixedHaarWavelet2D(self.in_channels)
        self.haar_proj = nn.Sequential(
            nn.Conv2d(self.in_channels * 4, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_feat = self.haar_proj(self.haar(x))
        s_feat = self.spatial_branch(x)
        return self.act(w_feat + s_feat)
