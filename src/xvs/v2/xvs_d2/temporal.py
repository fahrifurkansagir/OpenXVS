"""Dual-Interval 6-DoF WGS-84 Registration Helper & Explicitly Unrolled Stride-4 ConvGRU (`DR-01`).

Implements `SP2_M2_MODELS_DESIGN.md` §3.2 (`V2-M2L-2`):
- Strict `docs/xvs/CONVENTIONS.md` frames: `ECEF`, local tangent `NED` `{+x N, +y E, +z D}`,
  and camera frame `CAM` `{+x right, +y down, +z forward}` (ZERO internal ENU!).
- Dual-interval (`0.1 s` short lag & `0.5 s` medium lag) sky-rotation and WGS-84 tangent-ground
  planar homographies, plus externalized hidden-state warping (`H_prev_warped`).
- Explicitly unrolled 1-step depthwise-separable `3x3` ConvGRU using pure
  `Conv2d / Concat / Sigmoid / Tanh / Mul / Sub / Add` — ZERO `Loop` / `Scan` / `GRU` ONNX ops.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lla_to_ecef(lat_rad: float, lon_rad: float, h_m: float) -> np.ndarray:
    """Convert WGS-84 geodetic `(lat_rad, lon_rad, h_m)` to `ECEF` metres."""
    sin_lat = float(np.sin(lat_rad))
    cos_lat = float(np.cos(lat_rad))
    sin_lon = float(np.sin(lon_rad))
    cos_lon = float(np.cos(lon_rad))
    n_val = WGS84_A_M / float(np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat))
    x = (n_val + h_m) * cos_lat * cos_lon
    y = (n_val + h_m) * cos_lat * sin_lon
    z = (n_val * (1.0 - WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def r_ecef_to_ned(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Return 3x3 DCM `R_ned_ecef` mapping `ECEF` vectors into local `NED`."""
    slat = float(np.sin(lat_rad))
    clat = float(np.cos(lat_rad))
    slon = float(np.sin(lon_rad))
    clon = float(np.cos(lon_rad))
    return np.array(
        [
            [-slat * clon, -slat * slon, clat],
            [-slon, clon, 0.0],
            [-clat * clon, -clat * slon, -slat],
        ],
        dtype=np.float64,
    )


def compute_sky_rotation_homography(
    k_mat: np.ndarray,
    r_cam_ned_prev: np.ndarray,
    r_cam_ned_curr: np.ndarray,
) -> np.ndarray:
    """Compute exact infinite-range sky homography `H_{prev -> curr}` in `CAM` pixels.

    `p_curr ~ K * R_{cam_curr <- cam_prev} * K^{-1} * p_prev`,
    where `R_{cam_curr <- cam_prev} = R_{cam_curr <- ned} @ R_{cam_prev <- ned}^T`.
    """
    k = np.asarray(k_mat, dtype=np.float64)
    r_rel = np.asarray(r_cam_ned_curr, dtype=np.float64) @ np.asarray(r_cam_ned_prev, dtype=np.float64).T
    h_mat = k @ r_rel @ np.linalg.inv(k)
    if abs(h_mat[2, 2]) > 1e-12:
        h_mat = h_mat / h_mat[2, 2]
    return h_mat


def compute_wgs84_ground_homography(
    k_mat: np.ndarray,
    r_cam_ned_prev: np.ndarray,
    r_cam_ned_curr: np.ndarray,
    pos_ecef_prev: np.ndarray,
    pos_ecef_curr: np.ndarray,
    ref_lat_rad: float,
    ref_lon_rad: float,
    alt_agl_prev_m: float,
) -> np.ndarray:
    """Compute 6-DoF WGS-84 tangent-plane ground homography `H_{prev -> curr}` in `NED`/`CAM`.

    Uses `CONVENTIONS.md` `NED` tangent plane (`n_ned = [0, 0, -1]` pointing Up from ground
    toward camera, with perpendicular distance `d = alt_agl_prev_m > 0`).
    """
    if alt_agl_prev_m <= 0.5:
        raise ValueError(f"alt_agl_prev_m must be > 0.5 m, got {alt_agl_prev_m}")

    k = np.asarray(k_mat, dtype=np.float64)
    r_prev = np.asarray(r_cam_ned_prev, dtype=np.float64)
    r_curr = np.asarray(r_cam_ned_curr, dtype=np.float64)
    r_ned_ecef = r_ecef_to_ned(ref_lat_rad, ref_lon_rad)

    dp_ned = r_ned_ecef @ (
        np.asarray(pos_ecef_curr, dtype=np.float64) - np.asarray(pos_ecef_prev, dtype=np.float64)
    )
    # Camera-frame relative pose: X_curr = R_rel * X_prev + t_rel
    # Since X_cam = R_cam_ned * (P_ned - C_ned), t_rel = -R_curr * (C_curr - C_prev) = -R_curr * dp_ned
    r_rel = r_curr @ r_prev.T
    t_rel = -r_curr @ dp_ned  # [3]

    # Ground plane in previous camera frame: n_prev^T X_prev + d = 0
    # In NED, points on ground satisfy (P_ned - C_prev)_z = +alt_agl_prev_m, so n_ned = [0, 0, +1]
    # gives n_ned^T (P - C_prev) = +alt_agl_prev_m -> (1/d) * n_prev^T X_prev = 1
    n_prev = r_prev @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    h_cam = r_rel + np.outer(t_rel, n_prev) / float(alt_agl_prev_m)
    h_mat = k @ h_cam @ np.linalg.inv(k)
    if abs(h_mat[2, 2]) > 1e-12:
        h_mat = h_mat / h_mat[2, 2]
    return h_mat


def scale_homography(h_mat: np.ndarray, scale: float) -> np.ndarray:
    """Scale a pixel-space homography `H` to a feature map at stride `1 / scale` (e.g. `scale=0.25`)."""
    s = np.diag([float(scale), float(scale), 1.0])
    s_inv = np.diag([1.0 / float(scale), 1.0 / float(scale), 1.0])
    h_scaled = s @ np.asarray(h_mat, dtype=np.float64) @ s_inv
    if abs(h_scaled[2, 2]) > 1e-12:
        h_scaled = h_scaled / h_scaled[2, 2]
    return h_scaled


def warp_tensor_with_homography(
    src: torch.Tensor,
    h_prev_to_curr: np.ndarray,
    *,
    feature_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-side bilinear warp of `[B, C, H, W]` using `H_{prev -> curr}` outside the ONNX graph.

    Returns `(warped [B, C, H, W], valid_mask [B, 1, H, W])`.
    """
    if src.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B, C, H, W], got {tuple(src.shape)}")

    b_sz, _, h_sz, w_sz = src.shape
    h_eff = scale_homography(h_prev_to_curr, feature_scale) if feature_scale != 1.0 else np.asarray(h_prev_to_curr, dtype=np.float64)
    h_inv = np.linalg.inv(h_eff)

    yy, xx = np.meshgrid(
        np.arange(h_sz, dtype=np.float64),
        np.arange(w_sz, dtype=np.float64),
        indexing="ij",
    )
    ones = np.ones_like(xx)
    coords = np.stack([xx, yy, ones], axis=-1)  # [H, W, 3]
    mapped = coords @ h_inv.T
    w_denom = np.where(np.abs(mapped[..., 2:3]) < 1e-12, 1e-12, mapped[..., 2:3])
    src_x = mapped[..., 0] / w_denom[..., 0]
    src_y = mapped[..., 1] / w_denom[..., 0]

    valid = (
        (src_x >= 0.0)
        & (src_x <= float(w_sz - 1))
        & (src_y >= 0.0)
        & (src_y <= float(h_sz - 1))
    )
    grid_x = (2.0 * src_x / max(float(w_sz - 1), 1.0)) - 1.0
    grid_y = (2.0 * src_y / max(float(h_sz - 1), 1.0)) - 1.0
    grid = torch.from_numpy(np.stack([grid_x, grid_y], axis=-1).astype(np.float32)).to(
        device=src.device, dtype=src.dtype
    )
    grid_b = grid.unsqueeze(0).expand(b_sz, -1, -1, -1)

    warped = F.grid_sample(src, grid_b, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid_mask = (
        torch.from_numpy(valid.astype(np.float32))
        .to(device=src.device, dtype=src.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(b_sz, 1, -1, -1)
    )
    return warped * valid_mask, valid_mask


class DepthwiseSeparableConv2d(nn.Module):
    """ONNX-WebGPU-safe depthwise 3x3 + pointwise 1x1 convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class UnrolledConvGRUCell(nn.Module):
    """Explicitly unrolled 1-step depthwise-separable `3x3` ConvGRU at stride 4 (`DR-01:57-60`).

    Uses ONLY primitive ops (`Conv2d`, `Concat`, `Sigmoid`, `Tanh`, `Mul`, `Sub`, `Add`) so
    exported ONNX graphs contain ZERO `Loop`, `Scan`, or `GRU` nodes.
    """

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        cat_ch = self.in_channels + self.hidden_channels
        self.conv_z = DepthwiseSeparableConv2d(cat_ch, self.hidden_channels, kernel_size=3, padding=1, bias=True)
        self.conv_r = DepthwiseSeparableConv2d(cat_ch, self.hidden_channels, kernel_size=3, padding=1, bias=True)
        self.conv_h = DepthwiseSeparableConv2d(cat_ch, self.hidden_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x_s4: torch.Tensor, h_prev_warped: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_s4, h_prev_warped], dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        candidate = torch.tanh(self.conv_h(torch.cat([x_s4, r * h_prev_warped], dim=1)))
        return (1.0 - z) * h_prev_warped + z * candidate
