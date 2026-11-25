"""Target signal and contrast estimation module (WP-03 V8 / WP-04 §4).

Computes background statistics, integrated contrast A_C, mean contrast C, and degraded SNR
conforming to docs/xvs/WP04_DETECTION.md v0.1 §4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy import ndimage


@dataclass
class TargetSignalResult:
    """Target signal and contrast estimation outcome."""
    l_bar_bg: float
    sigma_bg: float
    a_c: float
    mean_contrast: float
    degraded_snr: float | None
    n_amodal: int
    n_d: int


def estimate_target_signal(
    image: np.ndarray,
    amodal_mask: np.ndarray | None = None,
    center_uv: tuple[float, float] | None = None,
    other_masks: Sequence[np.ndarray] | None = None,
    mad_scale: float = 1.4826,
    is_degraded: bool = False,
) -> TargetSignalResult:
    """Compute target signal properties for a single GT intruder instance.

    Args:
        image: 2D linear image (float64)
        amodal_mask: Boolean 2D mask of target intruder (optional)
        center_uv: Projected centre (u, v) in image coordinates
        other_masks: Sequence of boolean 2D masks of other intruders to exclude from background
        mad_scale: MAD normal consistency factor (default 1.4826)
        is_degraded: If True, computes degraded SNR in electrons

    Returns:
        TargetSignalResult with l_bar_bg, sigma_bg, a_c, mean_contrast, degraded_snr, n_amodal, n_d
    """
    h, w = image.shape
    if amodal_mask is not None and np.any(amodal_mask):
        mask_bool = np.asarray(amodal_mask, dtype=bool)
        n_amodal = int(np.sum(mask_bool))
        # Chessboard distance from target mask
        dt = ndimage.distance_transform_cdt(~mask_bool, metric="chessboard")
        # Annulus at chessboard distance 3..8 px
        annulus = (dt >= 3) & (dt <= 8)
        # Region D is amodal mask dilated by 2 px (dt <= 2)
        region_d = (dt <= 2)
    else:
        # Sub-pixel target: use 5x5 around center pixel
        n_amodal = 0
        if center_uv is not None:
            u, v = center_uv
            cx, cy = int(round(u)), int(round(v))
        else:
            cx, cy = w // 2, h // 2

        # 5x5 region D
        region_d = np.zeros((h, w), dtype=bool)
        y0 = max(0, cy - 2)
        y1 = min(h, cy + 3)
        x0 = max(0, cx - 2)
        x1 = min(w, cx + 3)
        region_d[y0:y1, x0:x1] = True

        # Annulus: chessboard distance 3..8 from center pixel
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_inf = np.maximum(np.abs(x_grid - cx), np.abs(y_grid - cy))
        annulus = (dist_inf >= 3) & (dist_inf <= 8)

    # Exclude other intruders from background annulus
    if other_masks is not None:
        for om in other_masks:
            if om is not None and np.any(om):
                om_bool = np.asarray(om, dtype=bool)
                dt_other = ndimage.distance_transform_cdt(~om_bool, metric="chessboard")
                annulus = annulus & (dt_other > 2)

    bg_vals = image[annulus]
    if len(bg_vals) == 0:
        # Fallback to whole-image robust stats if annulus was fully masked
        bg_vals = image.ravel()

    l_bar_bg = float(np.median(bg_vals))
    mad = float(np.median(np.abs(bg_vals - l_bar_bg)))
    sigma_bg = float(mad_scale * mad)

    denom_bg = max(1e-12, abs(l_bar_bg))
    d_vals = image[region_d]
    n_d = len(d_vals)

    # Integrated contrast: A_C = sum_{p in D} (L_p - L_bar_bg) / L_bar_bg
    a_c = float(np.sum((d_vals - l_bar_bg) / denom_bg))

    # Mean contrast: C = A_C / n_amodal
    mean_contrast = float(a_c / n_amodal) if n_amodal >= 1 else 0.0

    # Degraded SNR: sum_{p in D} (E_p - E_bar_bg) / (sigma_bg * sqrt(|D|))
    if is_degraded:
        denom_snr = max(1e-12, sigma_bg * math.sqrt(max(1, n_d)))
        degraded_snr = float(np.sum(d_vals - l_bar_bg) / denom_snr)
    else:
        degraded_snr = None

    return TargetSignalResult(
        l_bar_bg=l_bar_bg,
        sigma_bg=sigma_bg,
        a_c=a_c,
        mean_contrast=mean_contrast,
        degraded_snr=degraded_snr,
        n_amodal=n_amodal,
        n_d=n_d,
    )
