"""Calibrated 8-Channel CenterNet Stride-2 Heads & Local 3x3 Soft-Argmax Decoder (`XVS-D2`, `DR-01`).

Implements `SP2_M2_MODELS_DESIGN.md` §3.2 (`V2-M2L-2`):
- 4 decoupled heads emitting 8 dense channels at stride 2:
  1. `hm_logits [B, 1, H/2, W/2]`: `SpectralNorm` on penultimate conv, prior bias `b_0 = -11.09`
     (crop training) or `-14.04` (full-frame).
  2. `offset [B, 2, H/2, W/2]`: `1.5 * tanh(...)` (`±3.0` full-res pixels).
  3. `log_scale [B, 2, H/2, W/2]`: `[ln s_mom, ln r_wh]`.
  4. `log_var [B, 3, H/2, W/2]`: `[l_u, l_v, l_lns]` strictly on `F_P2.detach()`, clamped `[-6.0, +4.0]`.
- `peak_mask [B, 1, H/2, W/2]`: `3x3 MaxPool(stride=1, padding=1)` local-maxima mask (ZERO
  `TopK` / `NonMaxSuppression` inside ONNX!).
- Host-side `decode_peaks_local_softargmax` refines centroids via local `3x3` windowed soft-argmax.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from xvs.v2.xvs_d2.temporal import DepthwiseSeparableConv2d

CROP_PRIOR_BIAS = -11.09
FULL_FRAME_PRIOR_BIAS = -14.04


class _HeadTower(nn.Module):
    """Two 3x3 depthwise-separable blocks with optional SpectralNorm on the penultimate conv."""

    def __init__(self, channels: int, *, apply_spectral_norm_last: bool = False) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU6(inplace=False)
        self.conv2 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        if apply_spectral_norm_last:
            self.conv2.pw = spectral_norm(self.conv2.pw)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.ReLU6(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(self.conv1(x)))
        return self.act2(self.bn2(self.conv2(y)))


class DecoupledCenterNetHeads(nn.Module):
    """Exact 8-channel calibrated stride-2 CenterNet heads + 3x3 MaxPool peak mask (`DR-01:73-79`)."""

    def __init__(
        self,
        in_channels: int,
        prior_bias: float = CROP_PRIOR_BIAS,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.prior_bias = float(prior_bias)

        self.hm_tower = _HeadTower(self.in_channels, apply_spectral_norm_last=bool(use_spectral_norm))
        self.hm_out = nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.hm_out.bias.fill_(self.prior_bias)

        self.offset_tower = _HeadTower(self.in_channels, apply_spectral_norm_last=False)
        self.offset_out = nn.Conv2d(self.in_channels, 2, kernel_size=1, bias=True)

        self.scale_tower = _HeadTower(self.in_channels, apply_spectral_norm_last=False)
        self.scale_out = nn.Conv2d(self.in_channels, 2, kernel_size=1, bias=True)

        self.var_tower = _HeadTower(self.in_channels, apply_spectral_norm_last=False)
        self.var_out = nn.Conv2d(self.in_channels, 3, kernel_size=1, bias=True)

    def forward(
        self, f_p2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hm_logits = self.hm_out(self.hm_tower(f_p2))
        offset = 1.5 * torch.tanh(self.offset_out(self.offset_tower(f_p2)))
        log_scale = self.scale_out(self.scale_tower(f_p2))

        # Detached-feature heteroscedastic uncertainty head (`DR-01:79`): stop-gradient into backbone
        f_p2_detached = f_p2.detach()
        log_var = self.var_out(self.var_tower(f_p2_detached)).clamp(min=-6.0, max=4.0)

        # 3x3 MaxPool local-maxima mask (zero TopK / NMS inside ONNX graph)
        hm_max = F.max_pool2d(hm_logits, kernel_size=3, stride=1, padding=1)
        peak_mask = (hm_logits == hm_max).to(dtype=hm_logits.dtype)
        return hm_logits, offset, log_scale, log_var, peak_mask


@torch.no_grad()
def decode_peaks_local_softargmax(
    hm_logits: torch.Tensor,
    offset: torch.Tensor,
    log_scale: torch.Tensor,
    log_var: torch.Tensor,
    peak_mask: torch.Tensor,
    *,
    score_thr: float = 0.25,
    max_k: int = 64,
    stride: float = 2.0,
    offset_blend: float = 0.5,
) -> list[dict[str, Any]]:
    """Host-side peak scan & local 3x3 windowed soft-argmax sub-pixel centroid decoder (`DR-01:76`).

    Operates on batch index 0 (`[1, C, H/2, W/2]` or `[C, H/2, W/2]`).
    """
    if hm_logits.ndim == 4:
        hm_l = hm_logits[0, 0]
        off = offset[0]
        lsc = log_scale[0]
        lvr = log_var[0]
        pmk = peak_mask[0, 0]
    elif hm_logits.ndim == 3:
        hm_l = hm_logits[0]
        off = offset
        lsc = log_scale
        lvr = log_var
        pmk = peak_mask[0]
    else:
        raise ValueError(f"Expected 3D or 4D hm_logits, got {tuple(hm_logits.shape)}")

    probs = torch.sigmoid(hm_l.to(dtype=torch.float32))
    h_sz, w_sz = probs.shape
    keep = (probs >= float(score_thr)) & (pmk > 0.5)
    ys, xs = torch.where(keep)
    if ys.numel() == 0:
        return []

    scores = probs[ys, xs]
    order = torch.argsort(scores, descending=True)[: int(max_k)]
    ys = ys[order]
    xs = xs[order]

    detections: list[dict[str, Any]] = []
    for idx in range(int(ys.numel())):
        cy = int(ys[idx].item())
        cx = int(xs[idx].item())
        score = float(probs[cy, cx].item())

        # Local 3x3 windowed soft-argmax around (cy, cx)
        y0 = max(0, cy - 1)
        y1 = min(h_sz, cy + 2)
        x0 = max(0, cx - 1)
        x1 = min(w_sz, cx + 2)
        patch = hm_l[y0:y1, x0:x1].to(dtype=torch.float32)
        weights = torch.softmax(patch.reshape(-1), dim=0).reshape(patch.shape)

        gy = torch.arange(y0, y1, dtype=torch.float32, device=patch.device).unsqueeze(1)
        gx = torch.arange(x0, x1, dtype=torch.float32, device=patch.device).unsqueeze(0)
        soft_u = float((weights * gx).sum().item())
        soft_v = float((weights * gy).sum().item())

        du_cell = float(off[0, cy, cx].item())
        dv_cell = float(off[1, cy, cx].item())
        pred_u = float(cx) + du_cell
        pred_v = float(cy) + dv_cell

        b = float(offset_blend)
        cell_u = b * pred_u + (1.0 - b) * soft_u
        cell_v = b * pred_v + (1.0 - b) * soft_v

        # Convert stride-2 cell index to full-resolution pixel coordinate
        u_px = (cell_u + 0.5) * float(stride)
        v_px = (cell_v + 0.5) * float(stride)

        ln_s = float(lsc[0, cy, cx].item())
        ln_r = float(lsc[1, cy, cx].item())
        lu = float(lvr[0, cy, cx].item())
        lv = float(lvr[1, cy, cx].item())
        ls = float(lvr[2, cy, cx].item())

        detections.append(
            {
                "u_px": u_px,
                "v_px": v_px,
                "score": score,
                "ln_s_mom": ln_s,
                "s_mom_px": math.exp(max(-4.0, min(8.0, ln_s))),
                "ln_r_wh": ln_r,
                "var_u": math.exp(lu),
                "var_v": math.exp(lv),
                "var_ln_s": math.exp(ls),
                "cell_x": cx,
                "cell_y": cy,
            }
        )
    return detections
