"""Sub-pixel Earth-limb edge detector for masks, RGB, and degraded sensor imagery.

Conforms strictly to docs/xvs/WP06_HORIZON.md v0.1 §1.1 and CONVENTIONS.md §5, §7.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
from scipy.ndimage import gaussian_filter1d

from xvs.rig.rig import Rig
from xvs.rig.camera import PinholeCamera
from xvs.sensor.srgb import srgb_eotf
from xvs.sensor.distortion import undistort_points
from xvs.sim import horizon
from xvs.horizon_att.limb_model import LimbRay, LimbRayBatch


# Precomputed lookup table for uint8 sRGB EOTF for bit-exact and 12x faster linearisation
_SRGB_LUT = srgb_eotf(np.arange(256, dtype=np.uint8))


def extract_feature_map(img: np.ndarray) -> np.ndarray:
    """Extract 2D sky/ground feature map from input image (mask or RGB).

    - For masks (or monochrome): normalized intensity in [0, 1].
    - For RGB: linear luminance + blue ratio from linear sRGB.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        if np.issubdtype(arr.dtype, np.integer):
            max_val = 65535.0 if arr.dtype == np.uint16 else 255.0
            return (arr.astype(np.float64) / max_val).clip(0.0, 1.0)
        return arr.astype(np.float64).clip(0.0, 1.0)

    if arr.ndim == 3:
        # Check if all channels are identical (e.g. 3-channel grayscale / mask)
        c0 = arr[:, :, 0]
        if np.array_equal(c0, arr[:, :, 1]) and np.array_equal(c0, arr[:, :, 2]):
            max_val = 65535.0 if arr.dtype == np.uint16 else 255.0
            return (c0.astype(np.float64) / max_val).clip(0.0, 1.0)

        # RGB image: linearize via sRGB EOTF (use LUT if uint8)
        if arr.dtype == np.uint8:
            lin = _SRGB_LUT[arr]
        else:
            lin = srgb_eotf(arr)
        r = lin[:, :, 0]
        g = lin[:, :, 1]
        b = lin[:, :, 2]

        # Rec. 709 linear luminance
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        # Blue ratio
        sum_rgb = np.maximum(1e-5, r + g + b)
        blue_ratio = b / sum_rgb

        # Composite feature emphasizing the sky/terrain boundary
        feature = lum + 0.5 * blue_ratio
        return feature.astype(np.float64)

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def find_subpixel_edge_1d(
    profile: np.ndarray,
    search_range: tuple[int, int] | None = None,
    min_gradient: float = 0.02,
    sigma: float = 1.0,
) -> tuple[float, float] | None:
    """Find strongest sky-to-ground transition along a 1D profile with sub-pixel parabolic peak fit.

    Args:
        profile: 1D array of feature values.
        search_range: Optional (start_idx, end_idx) bounding the search window.
        min_gradient: Minimum gradient magnitude required to accept a peak.
        sigma: Gaussian smoothing scale (0.0 disables smoothing).

    Returns:
        (subpixel_pos, peak_gradient) or None if no transition exceeds threshold.
    """
    p = np.asarray(profile, dtype=np.float64)
    n = len(p)
    if n < 4:
        return None

    if sigma > 0.0:
        smoothed = gaussian_filter1d(p, sigma=sigma, mode="nearest")
    else:
        smoothed = p

    # Finite difference between adjacent pixels: diff[k] = smoothed[k+1] - smoothed[k]
    # Edge is located near index k + 0.5
    diff = np.abs(np.diff(smoothed))

    if search_range is not None:
        start_idx = max(1, min(search_range[0], n - 2))
        end_idx = max(start_idx + 1, min(search_range[1], n - 1))
        diff_window = diff[start_idx:end_idx]
        if len(diff_window) == 0:
            return None
        peak_offset = int(np.argmax(diff_window))
        k = start_idx + peak_offset
    else:
        if len(diff) < 3:
            return None
        k = int(np.argmax(diff[1:-1])) + 1

    g_0 = diff[k]
    if g_0 < min_gradient:
        return None

    # Parabolic refinement using 3 points: k-1, k, k+1
    g_m1 = diff[k - 1]
    g_p1 = diff[k + 1] if k + 1 < len(diff) else diff[k]

    denom = 2.0 * (g_m1 - 2.0 * g_0 + g_p1)
    if abs(denom) > 1e-12:
        delta = (g_m1 - g_p1) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0

    sub_pos = (k + 0.5) + delta
    return sub_pos, float(g_0)


def detect_camera_edges(
    img: np.ndarray,
    cam_id: str,
    cam: PinholeCamera,
    prior_pose: dict[str, Any] | None = None,
    step: int = 8,
    search_band_px: int = 50,
    min_gradient: float = 0.015,
    sigma: float | None = None,
) -> list[dict[str, float]]:
    """Detect Earth-limb edge points in a single camera image.

    Selects column scanning for |dv/du| <= 1 and row scanning for |dv/du| > 1.
    If prior_pose is provided, search bands are centered around predicted horizon.
    """
    feature = extract_feature_map(img)
    h_img, w_img = feature.shape[:2]

    # For binary/antialiased masks with sharp transitions, default sigma is 0
    # For RGB images with texture/noise, default sigma is 1.0
    if sigma is None:
        # Check if mask-like: few unique values
        unique_count = len(np.unique(feature[:100, :100]))
        sigma = 0.0 if unique_count < 10 else 1.0

    edges: list[dict[str, float]] = []

    # Check predicted horizon visibility and slope if prior pose is available
    has_prior = False
    p_cam: np.ndarray | None = None
    r_cam: np.ndarray | None = None
    if prior_pose is not None:
        pos = prior_pose.get("p_ecef_cam", prior_pose.get("position"))
        if pos is not None:
            p_cam = np.asarray(pos, dtype=np.float64)

        if "r_ecef_cam" in prior_pose and prior_pose["r_ecef_cam"] is not None:
            r_cam = np.asarray(prior_pose["r_ecef_cam"], dtype=np.float64)
        elif "direction" in prior_pose and "up" in prior_pose and "right" in prior_pose:
            # Reconstruct R_ecef_cam from Cesium camera vectors (CONVENTIONS.md §6)
            right = np.asarray(prior_pose["right"], dtype=np.float64)
            up = np.asarray(prior_pose["up"], dtype=np.float64)
            direction = np.asarray(prior_pose["direction"], dtype=np.float64)
            r_cam = np.column_stack([right, -up, direction])

        if p_cam is not None and r_cam is not None and r_cam.ndim == 2:
            has_prior = True

    # 1. Column-wise scan
    for col in range(20, w_img - 20, step):
        u_col = float(col)
        v_search_range = None

        if has_prior:
            v_roots = horizon.analytic_horizon_v_at_u(p_cam, r_cam, cam, u_col)
            v_in = [vr for vr in v_roots if -50.0 <= vr <= h_img + 50.0]
            if len(v_in) != 1:
                continue
            v_pred = v_in[0]
            slope = horizon.analytic_horizon_slope(p_cam, r_cam, cam, u_col, v_pred)
            if abs(slope) > 1.0:
                # Steep slope: skip column scan, will be captured in row scan
                continue
            v_search_range = (
                max(0, int(math.floor(v_pred - search_band_px))),
                min(h_img, int(math.ceil(v_pred + search_band_px))),
            )

        col_profile = feature[:, col]
        res = find_subpixel_edge_1d(
            col_profile, search_range=v_search_range, min_gradient=min_gradient, sigma=sigma
        )
        if res is not None:
            v_sub, grad_val = res
            if 0.0 <= v_sub <= h_img - 1.0:
                edges.append({"u": u_col, "v": v_sub, "gradient": grad_val})

    # 2. Row-wise scan for steep slopes (|dv/du| > 1)
    if has_prior:
        for row in range(20, h_img - 20, step):
            v_row = float(row)
            u_roots = horizon.analytic_horizon_u_at_v(p_cam, r_cam, cam, v_row)
            u_in = [ur for ur in u_roots if -50.0 <= ur <= w_img + 50.0]
            if len(u_in) != 1:
                continue
            u_pred = u_in[0]
            slope = horizon.analytic_horizon_slope(p_cam, r_cam, cam, u_pred, v_row)
            if abs(slope) <= 1.0:
                # Shallow slope: already captured in column scan
                continue
            u_search_range = (
                max(0, int(math.floor(u_pred - search_band_px))),
                min(w_img, int(math.ceil(u_pred + search_band_px))),
            )
            row_profile = feature[row, :]
            res = find_subpixel_edge_1d(
                row_profile, search_range=u_search_range, min_gradient=min_gradient, sigma=sigma
            )
            if res is not None:
                u_sub, grad_val = res
                if 0.0 <= u_sub <= w_img - 1.0:
                    edges.append({"u": u_sub, "v": v_row, "gradient": grad_val})

    return edges


def edges_to_ray_batch(
    edges_by_cam: dict[str, list[dict[str, float]]],
    rig: Rig,
    sensor_config: Any | None = None,
) -> LimbRayBatch:
    """Convert detected image edge points into body-frame unit rays.

    Applies lens undistortion if sensor_config specifies distortion parameters.
    """
    cam = rig.camera_model
    rays: list[LimbRay] = []

    for cam_id, edge_list in edges_by_cam.items():
        if len(edge_list) == 0:
            continue
        mount = rig.cam_by_id[cam_id]
        r_bc = mount.r_body_cam
        lev = mount.lever_arm_m

        uvs = np.array([[e["u"], e["v"]] for e in edge_list], dtype=np.float64)
        grads = np.array([e.get("gradient", 1.0) for e in edge_list], dtype=np.float64)  # default-ok: optional fallback default

        # Apply lens undistortion if WP-03 distortion was applied
        if sensor_config is not None and (
            abs(sensor_config.k1) > 1e-9
            or abs(sensor_config.k2) > 1e-9
            or abs(sensor_config.k3) > 1e-9
            or abs(sensor_config.p1) > 1e-9
            or abs(sensor_config.p2) > 1e-9
        ):
            uvs_undist = undistort_points(
                uvs,
                fx=sensor_config.fx,
                fy=sensor_config.fy,
                cx=sensor_config.cx,
                cy=sensor_config.cy,
                k1=sensor_config.k1,
                k2=sensor_config.k2,
                k3=sensor_config.k3,
                p1=sensor_config.p1,
                p2=sensor_config.p2,
            )
        else:
            uvs_undist = uvs

        for i in range(len(edge_list)):
            u_clean = float(uvs_undist[i, 0])
            v_clean = float(uvs_undist[i, 1])
            d_cam = cam.unproject(u_clean, v_clean)
            d_body = r_bc @ d_cam
            rays.append(
                LimbRay(
                    cam_id=cam_id,
                    d_body=d_body,
                    lever_arm_body=lev,
                    u=u_clean,
                    v=v_clean,
                    weight=grads[i],
                )
            )

    return LimbRayBatch.from_rays(rays)


def detect_camera_edges_multiscale(
    img: np.ndarray,
    cam_id: str,
    cam: PinholeCamera,
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Classical multi-scale coarse-to-fine horizon strip extractor (WP-07 G1 Item 1)."""
    from xvs.integrity.g1_extensions import detect_camera_edges_multiscale as _impl
    return _impl(img, cam_id, cam, *args, **kwargs)


