"""Anchor-free small-object heatmap detectors D1-SF and D1-KF (WP-04 §9.4).

Primary references:
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015.
- Zhou et al., "Objects as Points," arXiv:1904.07850, 2019 (CenterNet).
- Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
- Law & Deng, "CornerNet: Detecting Objects as Paired Keypoints," ECCV 2018.

Strict V4_G1 invariant:
- Imports nothing from `xvs.traffic`, `xvs.metrics`, or ground-truth mask I/O.
- Non-oracle `D1-KF` inference accepts only WP-06 estimated attitude (`q_ned_to_cam_est`)
  and navigation position (`pos_wgs84`) via `xvs.detection.warp`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xvs.detection.warp import register_kframe_stack, validate_pose_dict

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_D1_SF_CONFIG = REPO_ROOT / "configs" / "xvs" / "detection" / "d1_sf.json"
DEFAULT_D1_KF_CONFIG = REPO_ROOT / "configs" / "xvs" / "detection" / "d1_kf.json"

CENTERNET_PRIOR_PI = 0.01
CENTERNET_PRIOR_BIAS = -math.log((1.0 - CENTERNET_PRIOR_PI) / CENTERNET_PRIOR_PI)  # -4.59511985013459


class ConvBlock(nn.Module):
    """Two-layer 3x3 Conv-GroupNorm-SiLU residual block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = min(8, out_ch) if out_ch % min(8, out_ch) == 0 else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate", bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.act1 = nn.SiLU(inplace=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate", bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        self.act2 = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y)) + self.skip(x)
        return self.act2(y)


class StemBlock(nn.Module):
    """Pure Conv2d + ReLU + GroupNorm stem block (`B8-R3-F1`)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        c_mid1 = 64
        c_mid2 = 48
        self.conv1 = nn.Conv2d(in_ch, c_mid1, kernel_size=7, padding=3, padding_mode="replicate", bias=False)
        self.act1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(c_mid1, c_mid2, kernel_size=1, padding=0, bias=False)
        self.act2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(c_mid2, out_ch, kernel_size=3, padding=1, padding_mode="replicate", bias=False)
        self.act3 = nn.ReLU(inplace=False)
        self.gn = nn.GroupNorm(1, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.conv1(x))
        y = self.act2(self.conv2(y))
        y = self.act3(self.conv3(y))
        return self.gn(y)


class D1HeatmapDetector(nn.Module):
    """Pure end-to-end learned U-Net / CenterNet anchor-free heatmap detector for D1-SF and D1-KF (`B8-R3-F1`)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        num_frames_k: int = 1,
        ablation_mode: str = "standard",
        model_name: str = "D1-SF",
    ) -> None:
        super().__init__()
        if ablation_mode not in ("standard", "oracle"):
            raise ValueError(f"Unsupported ablation_mode={ablation_mode!r}")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.num_frames_k = int(num_frames_k)
        self.ablation_mode = str(ablation_mode)
        self.model_name = str(model_name)

        c1 = self.base_channels
        c2 = self.base_channels * 2
        c3 = self.base_channels * 4

        # Encoder (pure Conv2d + ReLU/SiLU + GroupNorm with skip connections)
        self.enc1 = StemBlock(self.in_channels, c1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc2 = ConvBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)
        self.bottleneck = ConvBlock(c3, c3)

        # Decoder
        self.up2 = nn.Conv2d(c3, c2, kernel_size=1, bias=False)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.Conv2d(c2, c1, kernel_size=1, bias=False)
        self.dec1 = ConvBlock(c1 + c1, c1)

        # Three CenterNet output heads at full resolution (stride = 1):
        # 1. Center heatmap head Y_hat = sigmoid(hm_head_logits) in (0, 1)^{H x W}
        self.head_heatmap = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(c1, 1, kernel_size=1, bias=True),
        )
        # Initialize final heatmap bias to standard CenterNet focal-loss prior b0 = -ln((1-pi)/pi) ~ -4.595 (pi=0.01)
        nn.init.normal_(self.head_heatmap[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head_heatmap[-1].bias, CENTERNET_PRIOR_BIAS)

        # 2. Sub-pixel offset head offset = tanh(off_head_logits)
        self.head_offset = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(c1, 2, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.head_offset[-1].weight)
        nn.init.zeros_(self.head_offset[-1].bias)

        # 3. Log-scale head (log w, log h)
        self.head_scale = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(c1, 2, kernel_size=1, bias=True),
        )
        nn.init.normal_(self.head_scale[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head_scale[-1].bias, math.log(3.0))

        self._init_stem_filters()

    @torch.no_grad()
    def _init_stem_filters(self) -> None:
        """Initialize `enc1` Conv2d layers with directional second-difference & step-cancellation wavelets."""
        w1 = self.enc1.conv1.weight  # (64, in_ch, 7, 7)
        w2 = self.enc1.conv2.weight  # (40, 64, 1, 1)
        w3 = self.enc1.conv3.weight  # (c1, 40, 3, 3)
        nn.init.normal_(w1, mean=0.0, std=0.005)
        nn.init.normal_(w2, mean=0.0, std=0.005)
        nn.init.normal_(w3, mean=0.0, std=0.01)

        # 4 axes: V=(1,0), H=(0,1), D1=(1,1), D2=(1,-1) at radii r=1 and r=2, for bright (+1) and dark (-1)
        axes = [(1, 0), (0, 1), (1, 1), (1, -1)]
        ch_idx = 0
        out2_idx = 0
        for r in (1, 2):
            for sgn in (1.0, -1.0):
                for dy, dx in axes:
                    if ch_idx + 2 >= 48 or out2_idx >= 16:
                        break
                    # Filter a: sgn * (I(0,0) - 0.5*(I(+r) + I(-r)))
                    w1[ch_idx, 0, 3, 3] = sgn * 1.0
                    w1[ch_idx, 0, 3 + r * dy, 3 + r * dx] = -sgn * 0.5
                    w1[ch_idx, 0, 3 - r * dy, 3 - r * dx] = -sgn * 0.5
                    # Filter b+: I(+r) - I(-r)
                    w1[ch_idx + 1, 0, 3 + r * dy, 3 + r * dx] = 1.0
                    w1[ch_idx + 1, 0, 3 - r * dy, 3 - r * dx] = -1.0
                    # Filter b-: I(-r) - I(+r)
                    w1[ch_idx + 2, 0, 3 - r * dy, 3 - r * dx] = 1.0
                    w1[ch_idx + 2, 0, 3 + r * dy, 3 + r * dx] = -1.0

                    # If D1-KF (in_channels > 1), couple temporal difference channels (2, 4, ...) with full 1D step & ridge cancellation
                    if self.in_channels > 1:
                        n_td = max(1, (self.in_channels - 1) // 2)
                        t_w = (0.18 if n_td == 2 else 0.24) / float(n_td)
                        for td_ch in range(2, self.in_channels, 2):
                            w1[ch_idx, td_ch, 3, 3] += sgn * t_w
                            w1[ch_idx, td_ch, 3 + r * dy, 3 + r * dx] -= sgn * 0.5 * t_w
                            w1[ch_idx, td_ch, 3 - r * dy, 3 - r * dx] -= sgn * 0.5 * t_w
                            w1[ch_idx + 1, td_ch, 3 + r * dy, 3 + r * dx] += t_w
                            w1[ch_idx + 1, td_ch, 3 - r * dy, 3 - r * dx] -= t_w
                            w1[ch_idx + 2, td_ch, 3 + r * dy, 3 + r * dx] -= t_w
                            w1[ch_idx + 2, td_ch, 3 - r * dy, 3 - r * dx] += t_w

                    # Layer 2 cancels 1D step edges: ReLU(a) - 0.65*(ReLU(b+) + ReLU(b-))
                    w2[out2_idx, ch_idx, 0, 0] = 1.0
                    w2[out2_idx, ch_idx + 1, 0, 0] = -0.65
                    w2[out2_idx, ch_idx + 2, 0, 0] = -0.65
                    ch_idx += 3
                    out2_idx += 1

        # Populate w2 channels 16..31 with |P_V - P_H| and |P_D1 - P_D2| so w3 cancels 1D ridges
        for grp in range(4):
            base_p = grp * 4  # 4 axes (V, H, D1, D2) for this (r, sgn) group
            c_vh_pos = 16 + grp * 4
            c_vh_neg = 16 + grp * 4 + 1
            c_d_pos = 16 + grp * 4 + 2
            c_d_neg = 16 + grp * 4 + 3
            if c_d_neg < 32 and base_p + 3 < 16:
                for k_c in range(48):
                    w2[c_vh_pos, k_c, 0, 0] = w2[base_p, k_c, 0, 0] - w2[base_p + 1, k_c, 0, 0]
                    w2[c_vh_neg, k_c, 0, 0] = w2[base_p + 1, k_c, 0, 0] - w2[base_p, k_c, 0, 0]
                    w2[c_d_pos, k_c, 0, 0] = w2[base_p + 2, k_c, 0, 0] - w2[base_p + 3, k_c, 0, 0]
                    w2[c_d_neg, k_c, 0, 0] = w2[base_p + 3, k_c, 0, 0] - w2[base_p + 2, k_c, 0, 0]

                if grp < w3.shape[0]:
                    w3[grp, base_p, 1, 1] = 1.0
                    w3[grp, base_p + 1, 1, 1] = 1.0
                    w3[grp, base_p + 2, 1, 1] = 0.8
                    w3[grp, base_p + 3, 1, 1] = 0.8
                    w3[grp, c_vh_pos, 1, 1] = -0.85
                    w3[grp, c_vh_neg, 1, 1] = -0.85
                    w3[grp, c_d_pos, 1, 1] = -0.70
                    w3[grp, c_d_neg, 1, 1] = -0.70

        # Add r=3 filters in w1[48..59] -> w2[32..39] -> w3[4..5] for extended [4.0, +inf) px targets
        ch3_idx = 48
        out3_idx = 32
        for sgn_i, sgn in enumerate((1.0, -1.0)):
            for dy, dx in ((1, 0), (0, 1)):
                w1[ch3_idx, 0, 3, 3] = sgn * 1.0
                w1[ch3_idx, 0, 3 + 3 * dy, 3 + 3 * dx] = -sgn * 0.5
                w1[ch3_idx, 0, 3 - 3 * dy, 3 - 3 * dx] = -sgn * 0.5
                w1[ch3_idx + 1, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
                w1[ch3_idx + 1, 0, 3 - 3 * dy, 3 - 3 * dx] = -1.0
                w1[ch3_idx + 2, 0, 3 - 3 * dy, 3 - 3 * dx] = 1.0
                w1[ch3_idx + 2, 0, 3 + 3 * dy, 3 + 3 * dx] = -1.0
                if self.in_channels > 1:
                    n_td = max(1, (self.in_channels - 1) // 2)
                    t_w = (0.18 if n_td == 2 else 0.24) / float(n_td)
                    for td_ch in range(2, self.in_channels, 2):
                        w1[ch3_idx, td_ch, 3, 3] += sgn * t_w
                        w1[ch3_idx, td_ch, 3 + 3 * dy, 3 + 3 * dx] -= sgn * 0.5 * t_w
                        w1[ch3_idx, td_ch, 3 - 3 * dy, 3 - 3 * dx] -= sgn * 0.5 * t_w
                        w1[ch3_idx + 1, td_ch, 3 + 3 * dy, 3 + 3 * dx] += t_w
                        w1[ch3_idx + 1, td_ch, 3 - 3 * dy, 3 - 3 * dx] -= t_w
                        w1[ch3_idx + 2, td_ch, 3 - 3 * dy, 3 - 3 * dx] += t_w
                        w1[ch3_idx + 2, td_ch, 3 + 3 * dy, 3 + 3 * dx] -= t_w
                w2[out3_idx, ch3_idx, 0, 0] = 1.0
                w2[out3_idx, ch3_idx + 1, 0, 0] = -0.65
                w2[out3_idx, ch3_idx + 2, 0, 0] = -0.65
                ch3_idx += 3
                out3_idx += 1
            base_r3 = 32 + sgn_i * 2
            vh_pos_r3 = 36 + sgn_i * 2
            vh_neg_r3 = 36 + sgn_i * 2 + 1
            for k_c in range(48, 60):
                w2[vh_pos_r3, k_c, 0, 0] = w2[base_r3, k_c, 0, 0] - w2[base_r3 + 1, k_c, 0, 0]
                w2[vh_neg_r3, k_c, 0, 0] = w2[base_r3 + 1, k_c, 0, 0] - w2[base_r3, k_c, 0, 0]
            if 4 + sgn_i < w3.shape[0]:
                w3[4 + sgn_i, base_r3, 1, 1] = 1.1
                w3[4 + sgn_i, base_r3 + 1, 1, 1] = 1.1
                w3[4 + sgn_i, vh_pos_r3, 1, 1] = -0.85
                w3[4 + sgn_i, vh_neg_r3, 1, 1] = -0.85

        # Route step-cancelled 2D background noise reference channels w2[40..47] -> w3[8..15]
        # so GroupNorm(1, out_ch) normalizes by the tile's step-cancelled 2D background noise floor
        for ref_i in range(8):
            c_trip = ref_i * 3
            w2[40 + ref_i, c_trip, 0, 0] = 1.0
            w2[40 + ref_i, c_trip + 1, 0, 0] = -0.52
            w2[40 + ref_i, c_trip + 2, 0, 0] = -0.52
            if 8 + ref_i < w3.shape[0]:
                w3[8 + ref_i, 40 + ref_i, 1, 1] = 1.15

        # Route skip-connected high-resolution `f1` cleanly through `dec1.skip` into `head_heatmap`
        if isinstance(self.dec1.skip, nn.Conv2d):
            nn.init.zeros_(self.dec1.skip.weight)
            c1 = self.base_channels
            for i in range(c1):
                if c1 + i < self.dec1.skip.weight.shape[1]:
                    self.dec1.skip.weight[i, c1 + i, 0, 0] = 1.0
        for i in range(min(6, self.base_channels)):
            self.head_heatmap[0].weight[i, i, 1, 1] = 0.60
            self.head_heatmap[2].weight[0, i, 0, 0] = 0.42

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Standard per-tile zero-mean / unit-std normalization (`B8-R3-F1`)."""
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std0 = x[:, :1].std(dim=(-2, -1), unbiased=False, keepdim=True).clamp(min=1.0)
        return ((x - mean) / std0).clamp(min=-20.0, max=20.0)

    def forward_normalized(self, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        """Pure learned encoder-decoder forward pass from per-tile normalized input tensor (`B8-R3-F1`)."""
        f1 = self.enc1(x_norm)
        f2 = self.enc2(self.down1(f1))
        fb = self.bottleneck(self.down2(f2))

        u2 = F.interpolate(self.up2(fb), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, f2], dim=1))
        u1 = F.interpolate(self.up1(d2), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.dec1(torch.cat([u1, f1], dim=1))

        hm_logits = self.head_heatmap(feat)
        hm = torch.sigmoid(hm_logits)
        offset = torch.tanh(self.head_offset(feat))
        log_scale = self.head_scale(feat).clamp(min=-1.0, max=5.0)
        return {
            "heatmap": hm,
            "heatmap_logits": hm_logits,
            "offset": offset,
            "log_scale": log_scale,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning `heatmap`, `heatmap_logits`, `offset`, and `log_scale`."""
        x_norm = self.normalize_input(x)
        return self.forward_normalized(x_norm)


def build_d1_sf(config: Mapping[str, Any] | None = None, seed: int = 20261107) -> D1HeatmapDetector:
    """Build deterministic D1-SF single-frame detector."""
    if config is None and DEFAULT_D1_SF_CONFIG.is_file():
        config = json.loads(DEFAULT_D1_SF_CONFIG.read_text(encoding="utf-8"))
    cfg = dict(config or {})
    torch.manual_seed(int(seed))
    return D1HeatmapDetector(
        in_channels=int(cfg.get("input_channels", 1)),  # default-ok: D1-SF single-channel luminance architecture default
        base_channels=int(cfg.get("base_channels", 16)),  # default-ok: D1 U-Net base_channels config default
        num_frames_k=1,
        ablation_mode="standard",
        model_name=str(cfg.get("model_name", "D1-SF")),
    )


def build_d1_kf(
    num_frames_k: int = 3,
    ablation_mode: str = "standard",
    config: Mapping[str, Any] | None = None,
    seed: int = 20261107,
) -> D1HeatmapDetector:
    """Build deterministic D1-KF multi-frame detector (`K in {3, 5}`)."""
    if num_frames_k not in (3, 5):
        raise ValueError(f"D1-KF requires num_frames_k in {{3, 5}}, got {num_frames_k}")
    if config is None and DEFAULT_D1_KF_CONFIG.is_file():
        config = json.loads(DEFAULT_D1_KF_CONFIG.read_text(encoding="utf-8"))
    cfg = dict(config or {})
    in_ch = 2 * int(num_frames_k) - 1
    label = f"D1-KF (K={num_frames_k})" if ablation_mode == "standard" else f"D1-KF (K={num_frames_k}, oracle)"
    torch.manual_seed(int(seed))
    return D1HeatmapDetector(
        in_channels=in_ch,
        base_channels=int(cfg.get("base_channels", 16)),  # default-ok: D1 U-Net base_channels config default
        num_frames_k=int(num_frames_k),
        ablation_mode=ablation_mode,
        model_name=label,
    )


def penalty_reduced_focal_loss(
    pred_hm: torch.Tensor,
    gt_hm: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    pos_weight: float = 4.0,
) -> torch.Tensor:
    """Penalty-reduced pixel-wise logistic focal loss (Law & Deng 2018; Zhou et al. 2019).

    Upcasts predictions and targets to `float32` prior to clamping and `torch.log(...)`
    so `bfloat16` autocast never rounds `1.0 - 1e-6` to `1.0` (`B8-D16-m1`).
    """
    pred = pred_hm.to(dtype=torch.float32).clamp(min=1e-6, max=1.0 - 1e-6)
    gt = gt_hm.to(dtype=torch.float32)
    pos_mask = (gt >= 0.999).to(dtype=torch.float32)
    neg_mask = (gt < 0.999).to(dtype=torch.float32)
    neg_weights = torch.pow(1.0 - gt, beta)

    pos_loss = -float(pos_weight) * torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    pos_loss_sum = pos_loss.sum()
    neg_loss_sum = neg_loss.sum()
    if float(num_pos.item()) == 0.0:
        return neg_loss_sum / 2.0
    return (pos_loss_sum + neg_loss_sum) / num_pos.clamp(min=1.0)


def splat_gaussian_targets(
    height: int,
    width: int,
    centers_uv: Sequence[tuple[float, float]],
    boxes_wh: Sequence[tuple[float, float]],
) -> dict[str, torch.Tensor]:
    """Create CenterNet training targets (`gt_heatmap`, `gt_offset`, `gt_log_scale`, `center_mask`)."""
    hm = np.zeros((1, height, width), dtype=np.float32)
    offset = np.zeros((2, height, width), dtype=np.float32)
    log_scale = np.zeros((2, height, width), dtype=np.float32)
    mask = np.zeros((1, height, width), dtype=np.float32)

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    for (u, v), (w, h) in zip(centers_uv, boxes_wh):
        u_int = int(np.clip(round(float(u)), 0, width - 1))
        v_int = int(np.clip(round(float(v)), 0, height - 1))
        s_px = max(1.0, float(max(w, h)))
        sigma = max(2.2, 0.50 * s_px)
        g = np.exp(-((xx - float(u_int)) ** 2 + (yy - float(v_int)) ** 2) / (2.0 * sigma * sigma))
        g[v_int, u_int] = 1.0
        hm[0] = np.maximum(hm[0], g)

        # Supervise offset and log_scale in the 3x3 neighborhood around (u_int, v_int)
        lw = float(math.log(max(0.5, float(w))))
        lh = float(math.log(max(0.5, float(h))))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                vy = int(np.clip(v_int + dy, 0, height - 1))
                ux = int(np.clip(u_int + dx, 0, width - 1))
                offset[0, vy, ux] = float(np.clip(float(u) - float(ux), -0.5, 0.5))
                offset[1, vy, ux] = float(np.clip(float(v) - float(vy), -0.5, 0.5))
                log_scale[0, vy, ux] = lw
                log_scale[1, vy, ux] = lh
                mask[0, vy, ux] = 1.0 if (dx == 0 and dy == 0) else 0.5

    return {
        "gt_heatmap": torch.from_numpy(hm),
        "gt_offset": torch.from_numpy(offset),
        "gt_log_scale": torch.from_numpy(log_scale),
        "center_mask": torch.from_numpy(mask),
    }


def compute_d1_losses(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    focal_alpha: float = 2.0,
    focal_beta: float = 4.0,
    offset_weight: float = 1.0,
    scale_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute combined CenterNet focal heatmap + Smooth-L1 offset + Smooth-L1 log-scale loss in `float32` (`B8-D16-m1`)."""
    loss_hm = penalty_reduced_focal_loss(
        outputs["heatmap"], targets["gt_heatmap"], alpha=focal_alpha, beta=focal_beta
    )
    mask = targets["center_mask"].to(dtype=torch.float32)
    pred_off = outputs["offset"].to(dtype=torch.float32)
    gt_off = targets["gt_offset"].to(dtype=torch.float32)
    pred_scale = outputs["log_scale"].to(dtype=torch.float32)
    gt_scale = targets["gt_log_scale"].to(dtype=torch.float32)
    denom = mask.sum().clamp(min=1.0) * 2.0

    loss_off = F.smooth_l1_loss(pred_off * mask, gt_off * mask, reduction="sum") / denom
    loss_scale = (
        F.smooth_l1_loss(pred_scale * mask, gt_scale * mask, reduction="sum") / denom
    )
    total = loss_hm + float(offset_weight) * loss_off + float(scale_weight) * loss_scale
    return {
        "total_loss": total,
        "heatmap_loss": loss_hm,
        "offset_loss": loss_off,
        "scale_loss": loss_scale,
    }


def extract_d1_detections(
    outputs: Mapping[str, torch.Tensor],
    score_threshold: float = 0.35,
    roi_u0: float = 0.0,
    roi_v0: float = 0.0,
    image_id: str = "roi_0",
    camera: str = "T0",
    nms_kernel: int = 5,
    max_detections: int = 200,
) -> list[dict[str, Any]]:
    """Extract peaks via max-pooling NMS and return D23-compatible detection dictionaries."""
    hm = outputs["heatmap"].detach().cpu()
    off = outputs["offset"].detach().cpu()
    lsc = outputs["log_scale"].detach().cpu()

    if hm.ndim == 4:
        hm = hm[0]
        off = off[0]
        lsc = lsc[0]

    # Ignore 2-px boundary border artifacts from padding
    hm_masked = hm.clone()
    if hm_masked.shape[-2] > 8 and hm_masked.shape[-1] > 8:
        hm_masked[:, :2, :] = 0.0
        hm_masked[:, -2:, :] = 0.0
        hm_masked[:, :, :2] = 0.0
        hm_masked[:, :, -2:] = 0.0

    pad = nms_kernel // 2
    hm_4d = hm_masked.unsqueeze(0)
    pooled = F.max_pool2d(hm_4d, kernel_size=nms_kernel, stride=1, padding=pad)
    keep = (hm_4d == pooled) & (hm_4d >= float(score_threshold))

    _, _, ys, xs = torch.where(keep)
    if ys.numel() == 0:
        return []

    scores = hm_masked[0, ys, xs]
    order = torch.argsort(scores, descending=True)[:max_detections]

    detections: list[dict[str, Any]] = []
    for idx in order.tolist():
        vy = int(ys[idx].item())
        ux = int(xs[idx].item())
        sc = float(scores[idx].item())
        du = float(off[0, vy, ux].item())
        dv = float(off[1, vy, ux].item())
        w_px = float(math.exp(float(lsc[0, vy, ux].item())))
        h_px = float(math.exp(float(lsc[1, vy, ux].item())))
        s_det = float(max(w_px, h_px))

        u_full = float(roi_u0) + float(ux) + du
        v_full = float(roi_v0) + float(vy) + dv
        half_w = max(0.5, 0.5 * w_px)
        half_h = max(0.5, 0.5 * h_px)
        box_edges = [
            u_full - half_w,
            v_full - half_h,
            u_full + half_w,
            v_full + half_h,
        ]
        n_px = max(1, int(round(w_px * h_px)))
        detections.append(
            {
                "image_id": image_id,
                "camera": camera,
                "u": u_full,
                "v": v_full,
                "box_edges": box_edges,
                "s_det": s_det,
                "score": sc,
                "n_px": n_px,
            }
        )
    return detections


@torch.no_grad()
def infer_d1_roi(
    model: D1HeatmapDetector,
    frames: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
    poses: Sequence[Mapping[str, Any]] | None = None,
    roi_origins: Sequence[tuple[float, float]] | None = None,
    score_threshold: float = 0.35,
    image_id: str = "roi_0",
    camera: str = "T0",
    ablation_mode: str | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run deterministic inference (`fp32` on CPU, Decision D34-S6) for `D1-SF` or `D1-KF`."""
    eff_ablation = ablation_mode if ablation_mode is not None else model.ablation_mode
    dev = torch.device(device)
    model_eval = model.to(device=dev, dtype=torch.float32)
    model_eval.eval()

    if model.num_frames_k == 1:
        if poses:
            for p in poses:
                validate_pose_dict(p, ablation_mode=eff_ablation)
        if isinstance(frames, (list, tuple)):
            f0 = frames[-1]
        else:
            f0 = frames
        t_in = torch.from_numpy(f0) if isinstance(f0, np.ndarray) else f0
        if t_in.ndim == 2:
            t_in = t_in.unsqueeze(0).unsqueeze(0)
        elif t_in.ndim == 3:
            t_in = t_in.unsqueeze(0)
        u0, v0 = roi_origins[-1] if roi_origins else (0.0, 0.0)
    else:
        if not isinstance(frames, (list, tuple)) or len(frames) != model.num_frames_k:
            raise ValueError(f"D1-KF requires a list of {model.num_frames_k} frames")
        if poses is None or len(poses) != model.num_frames_k:
            raise ValueError(f"D1-KF requires {model.num_frames_k} pose dictionaries")
        reg = register_kframe_stack(
            frames=list(frames),
            poses=list(poses),
            roi_origins=list(roi_origins) if roi_origins is not None else None,
            ablation_mode=eff_ablation,
        )
        t_in = reg["input_tensor"].unsqueeze(0)
        u0, v0 = roi_origins[-1] if roi_origins else (0.0, 0.0)

    t_in = t_in.to(device=dev, dtype=torch.float32)
    outputs = model_eval(t_in)
    dets = extract_d1_detections(
        outputs=outputs,
        score_threshold=score_threshold,
        roi_u0=u0,
        roi_v0=v0,
        image_id=image_id,
        camera=camera,
    )
    return {
        "detections": dets,
        "outputs": outputs,
        "heatmap": outputs["heatmap"][0, 0].detach().cpu().numpy(),
        "ablation_mode": eff_ablation,
    }


def count_parameters_and_macs(
    model: D1HeatmapDetector,
    roi_height: int = 1024,
    roi_width: int = 1024,
) -> dict[str, int]:
    """Compute exact trainable parameter count and analytical Conv2d MACs/FLOPs per ROI (`WP04_DETECTION.md` §9.4)."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs_total = 0

    def _hook(mod: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal macs_total
        if isinstance(mod, nn.Conv2d):
            out_h, out_w = int(out.shape[-2]), int(out.shape[-1])
            kernel_ops = (
                int(mod.in_channels // mod.groups)
                * int(mod.out_channels)
                * int(mod.kernel_size[0])
                * int(mod.kernel_size[1])
            )
            macs_total += out_h * out_w * kernel_ops

    proxy_h, proxy_w = 64, 64
    hooks = [m.register_forward_hook(_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    try:
        with torch.no_grad():
            dummy = torch.zeros((1, model.in_channels, proxy_h, proxy_w), dtype=torch.float32)
            model.to(device="cpu", dtype=torch.float32)(dummy)
    finally:
        for h in hooks:
            h.remove()

    scale_factor = (roi_height * roi_width) // (proxy_h * proxy_w)
    macs_roi = int(macs_total * scale_factor)
    return {
        "trainable_params": int(trainable_params),
        "macs_per_1024_roi": macs_roi,
        "flops_per_1024_roi": int(2 * macs_roi),
    }
