"""Exact Four-Tier Nested `XVS-D2` Architecture & Stride-2 FP16 Memory Estimator (`DR-01`).

Implements `SP2_M2_MODELS_DESIGN.md` §3.2 (`V2-M2L-2`):
- `teacher`:   `[48, 96, 192, 384]`, depths `[2, 3, 6, 3]`, `c_fpn=96`, `c_gru=96`, `r=3`
- `edge`:      `[24, 48, 96, 192]`,  depths `[1, 2, 4, 2]`, `c_fpn=48`, `c_gru=48`, `r=2`
- `edge_slim`: `[16, 32, 64, 128]`,  depths `[1, 2, 3, 2]`, `c_fpn=32`, `c_gru=32`, `r=2`
- `web`:       `[16, 32, 64, 96]`,   depths `[1, 1, 2, 1]`, `c_fpn=24`, `c_gru=24`, `r=2`
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xvs.v2.xvs_d2.backbone import FPNP2Neck, StarNetP2Backbone
from xvs.v2.xvs_d2.heads import CROP_PRIOR_BIAS, DecoupledCenterNetHeads


@dataclass(frozen=True)
class XvsD2TierSpec:
    """Frozen architectural configuration for one `XVS-D2` deployment tier."""

    name: str
    widths: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    c_fpn: int
    c_gru: int
    expand_ratio: int
    default_input_hw: tuple[int, int]
    target_param_min: int
    target_param_max: int


XVS_D2_TIER_SPECS: dict[str, XvsD2TierSpec] = {
    "teacher": XvsD2TierSpec(
        name="teacher",
        widths=(48, 96, 192, 384),
        depths=(2, 3, 6, 3),
        c_fpn=96,
        c_gru=96,
        expand_ratio=3,
        default_input_hw=(2048, 2448),
        target_param_min=6_500_000,
        target_param_max=9_000_000,
    ),
    "edge": XvsD2TierSpec(
        name="edge",
        widths=(24, 48, 96, 192),
        depths=(1, 2, 4, 2),
        c_fpn=48,
        c_gru=48,
        expand_ratio=2,
        default_input_hw=(2048, 2448),
        target_param_min=1_100_000,
        target_param_max=1_800_000,
    ),
    "edge_slim": XvsD2TierSpec(
        name="edge_slim",
        widths=(16, 32, 64, 128),
        depths=(1, 2, 3, 2),
        c_fpn=32,
        c_gru=32,
        expand_ratio=2,
        default_input_hw=(2048, 2448),
        target_param_min=480_000,
        target_param_max=800_000,
    ),
    "web": XvsD2TierSpec(
        name="web",
        widths=(16, 32, 64, 96),
        depths=(1, 1, 2, 1),
        c_fpn=24,
        c_gru=24,
        expand_ratio=2,
        default_input_hw=(1024, 1224),
        target_param_min=220_000,
        target_param_max=420_000,
    ),
}


def get_tier_spec(tier: str) -> XvsD2TierSpec:
    """Return the locked `XvsD2TierSpec` for `tier` (`teacher`, `edge`, `edge_slim`, `web`)."""
    key = tier.strip().lower()
    if key not in XVS_D2_TIER_SPECS:
        raise ValueError(f"Unknown XVS-D2 tier {tier!r}; expected one of {sorted(XVS_D2_TIER_SPECS)}")
    return XVS_D2_TIER_SPECS[key]


def estimate_peak_s2_fp16_mib(
    tier: str,
    height: int | None = None,
    width: int | None = None,
) -> float:
    """Estimate peak stride-2 FP16 activation buffer footprint in MiB (`DR-01:64-69`).

    Computes `(H/2) * (W/2) * C1 * 2 bytes / (1024^2)` (matching `DR-01`: `114.8 MiB` Teacher,
    `57.4 MiB` Edge, `38.2 MiB` Edge-Slim at `2448x2048`, and `9.56 MiB <= 16 MiB` Web at `1224x1024`).
    """
    spec = get_tier_spec(tier)
    h = int(spec.default_input_hw[0] if height is None else height)
    w = int(spec.default_input_hw[1] if width is None else width)
    s2_pixels = (h // 2) * (w // 2)
    c_peak = spec.widths[0]
    return float(s2_pixels * c_peak * 2) / float(1024 * 1024)


class XvsD2Model(nn.Module):
    """Unified `XVS-D2` spatio-temporal civil airborne traffic detector (`DR-01` / `SP2 §3.2`).

    Inputs:
        `x_t`: `[B, 6, H, W]` dual-interval normalized stack (`H, W` divisible by `16`).
        `h_prev_warped`: `[B, C_gru, H/4, W/4]` externalized stride-4 warped recurrent state
            (if `None`, zero-initialized on `x_t`'s device/dtype).
    Outputs (dict):
        `hm_logits [B, 1, H/2, W/2]`, `offset [B, 2, H/2, W/2]`, `log_scale [B, 2, H/2, W/2]`,
        `log_var [B, 3, H/2, W/2]`, `peak_mask [B, 1, H/2, W/2]`, `h_next [B, C_gru, H/4, W/4]`,
        and optionally `pyramid` (`p2`, `p3`, `p4`) for `CSCWD` / `CrossKD` distillation.
    """

    def __init__(
        self,
        tier: str = "web",
        prior_bias: float = CROP_PRIOR_BIAS,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.spec = get_tier_spec(tier)
        self.tier = self.spec.name
        self.backbone = StarNetP2Backbone(
            widths=self.spec.widths,
            depths=self.spec.depths,
            expand_ratio=self.spec.expand_ratio,
            in_channels=6,
        )
        self.neck = FPNP2Neck(
            widths=self.spec.widths,
            c_fpn=self.spec.c_fpn,
            c_gru=self.spec.c_gru,
        )
        self.heads = DecoupledCenterNetHeads(
            in_channels=self.spec.c_fpn,
            prior_bias=prior_bias,
            use_spectral_norm=use_spectral_norm,
        )

    def init_hidden(self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            (int(batch_size), self.spec.c_gru, int(height) // 4, int(width) // 4),
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev_warped: torch.Tensor | None = None,
        *,
        return_pyramid: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if x_t.ndim != 4 or x_t.shape[1] != 6:
            raise ValueError(f"Expected x_t shape [B, 6, H, W], got {tuple(x_t.shape)}")
        b_sz, _, h_sz, w_sz = x_t.shape
        if h_sz % 16 != 0 or w_sz % 16 != 0:
            raise ValueError(f"Input spatial dims (H={h_sz}, W={w_sz}) must be divisible by 16")

        if h_prev_warped is None:
            h_prev = self.init_hidden(b_sz, h_sz, w_sz, x_t.device, x_t.dtype)
        else:
            expected_h = (b_sz, self.spec.c_gru, h_sz // 4, w_sz // 4)
            if tuple(h_prev_warped.shape) != expected_h:
                raise ValueError(
                    f"Expected h_prev_warped shape {expected_h}, got {tuple(h_prev_warped.shape)}"
                )
            h_prev = h_prev_warped

        s1, s2, s3, s4 = self.backbone(x_t)
        f_p2, h_next, pyramid = self.neck(s1, s2, s3, s4, h_prev)
        hm_logits, offset, log_scale, log_var, peak_mask = self.heads(f_p2)

        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "hm_logits": hm_logits,
            "offset": offset,
            "log_scale": log_scale,
            "log_var": log_var,
            "peak_mask": peak_mask,
            "h_next": h_next,
        }
        if return_pyramid:
            out["pyramid"] = pyramid
        return out


def build_xvs_d2(
    tier: str = "web",
    *,
    prior_bias: float = CROP_PRIOR_BIAS,
    use_spectral_norm: bool = True,
) -> XvsD2Model:
    """Instantiate one of the 4 nested `XVS-D2` tiers (`teacher`, `edge`, `edge_slim`, `web`)."""
    return XvsD2Model(tier=tier, prior_bias=prior_bias, use_spectral_norm=use_spectral_norm)


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count total (or trainable-only) scalar parameters in `model`."""
    if trainable_only:
        return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return sum(int(p.numel()) for p in model.parameters())
