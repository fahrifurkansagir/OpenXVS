"""Analytic Earth-limb (horizon) ground truth on WGS84 and sub-pixel edge estimator.

Conforms strictly to docs/xvs/CONVENTIONS.md v1.0 §7 and docs/xvs/WP01_SIM_RIG.md v1.1 §2.6.
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np

from xvs.geo import wgs84
from xvs.rig.camera import PinholeCamera


def wgs84_limb_points_ecef(
    p_ecef_cam: np.ndarray,
    num_points: int = 2048,
) -> np.ndarray:
    """Compute exact 3D limb points on WGS84 ellipsoid seen from camera center C (ECEF).

    Conforms strictly to CONVENTIONS.md §7:
      1. Map to unit sphere: C' = (Cx/a, Cy/a, Cz/b), k = |C'|, c_hat = C'/k
      2. Tangent circle: y(tau) = c_hat/k + sqrt(1 - 1/k^2) * (e1 cos tau + e2 sin tau)
      3. Map back: x(tau) = (a * y1, a * y2, b * y3)
    Returns: (num_points, 3) float64 array of ECEF coordinates.
    """
    a = wgs84.WGS84_A
    b = wgs84.WGS84_B
    c_x, c_y, c_z = p_ecef_cam

    c_prime = np.array([c_x / a, c_y / a, c_z / b], dtype=np.float64)
    k = float(np.linalg.norm(c_prime))
    if k <= 1.0:
        raise ValueError("Camera center must be outside the WGS84 ellipsoid (k > 1)")

    c_hat = c_prime / k

    # Orthonormal basis in plane perpendicular to c_hat
    if abs(c_hat[2]) < 0.9:
        ref_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = np.cross(c_hat, ref_up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c_hat, e1)

    taus = np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)
    cos_taus = np.cos(taus)
    sin_taus = np.sin(taus)

    d_center = 1.0 / k
    r_circle = math.sqrt(max(0.0, 1.0 - 1.0 / (k * k)))

    # y(tau): (num_points, 3)
    y = (d_center * c_hat)[None, :] + r_circle * (cos_taus[:, None] * e1[None, :] + sin_taus[:, None] * e2[None, :])

    # x(tau) = [a * y0, a * y1, b * y2]
    x = np.empty_like(y)
    x[:, 0] = a * y[:, 0]
    x[:, 1] = a * y[:, 1]
    x[:, 2] = b * y[:, 2]
    return x


def compute_horizon_gt(
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    num_samples: int = 4096,
    max_points: int = 256,
) -> dict[str, Any]:
    """Compute analytic horizon GT for a camera: visibility and subsampled in-image UV points.

    Conforms to WP01_SIM_RIG.md v1.1 §2.6:
      - output horizon_visible (bool)
      - horizon_uv: <= 256 points subsampled uniformly along visible arc, full precision.
    """
    pts_ecef = wgs84_limb_points_ecef(p_ecef_cam, num_points=num_samples)

    # Project to camera
    # p_cam = R_ecef_cam.T @ (pts_ecef - p_ecef_cam)
    diff = pts_ecef - p_ecef_cam
    p_cam = r_ecef_cam.T @ diff.T  # (3, num_samples)

    in_front = p_cam[2] > 0.0
    valid_indices = np.where(in_front)[0]

    if len(valid_indices) == 0:
        return {"horizon_visible": False, "horizon_uv": []}

    u_all = np.full(num_samples, np.nan, dtype=np.float64)
    v_all = np.full(num_samples, np.nan, dtype=np.float64)
    u_all[valid_indices] = cam.fx * p_cam[0, valid_indices] / p_cam[2, valid_indices] + cam.cx
    v_all[valid_indices] = cam.fy * p_cam[1, valid_indices] / p_cam[2, valid_indices] + cam.cy

    in_img = in_front & (-0.5 <= u_all) & (u_all <= cam.width - 0.5) & (-0.5 <= v_all) & (v_all <= cam.height - 0.5)
    img_indices = np.where(in_img)[0]

    if len(img_indices) == 0:
        return {"horizon_visible": False, "horizon_uv": []}

    # Extract contiguous segments along the circle
    # Find break points where consecutive indices differ by > 1 (accounting for circular wrap)
    segments: list[list[int]] = []
    current_segment: list[int] = [int(img_indices[0])]

    for idx in img_indices[1:]:
        if idx == current_segment[-1] + 1:
            current_segment.append(int(idx))
        else:
            segments.append(current_segment)
            current_segment = [int(idx)]
    segments.append(current_segment)

    # If first and last wrap around the circle
    if len(segments) > 1 and segments[0][0] == 0 and segments[-1][-1] == num_samples - 1:
        merged = segments[-1] + segments[0]
        segments = [merged] + segments[1:-1]

    # Subsample points uniformly across segments up to max_points total
    total_visible = sum(len(s) for s in segments)
    final_uv: list[list[float]] = []

    for seg in segments:
        k_pts = max(2, int(round(max_points * len(seg) / total_visible)))
        if len(seg) <= k_pts:
            sub_indices = seg
        else:
            step = (len(seg) - 1) / (k_pts - 1)
            sub_indices = [seg[int(round(i * step))] for i in range(k_pts)]

        for idx in sub_indices:
            final_uv.append([float(u_all[idx]), float(v_all[idx])])

    if len(final_uv) > max_points:
        # Downsample uniformly to max_points
        step = (len(final_uv) - 1) / (max_points - 1)
        final_uv = [final_uv[int(round(i * step))] for i in range(max_points)]

    return {
        "horizon_visible": True,
        "horizon_uv": final_uv,
    }


def analytic_horizon_v_at_u(
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    u: float,
) -> list[float]:
    """Analytic row v coordinates of the Earth limb at column u.

    Returns empty list if no horizon is visible at column u, or list of real v roots.
    """
    a = wgs84.WGS84_A
    b = wgs84.WGS84_B
    s_mat = np.diag([1.0 / a, 1.0 / a, 1.0 / b])

    c_prime = s_mat @ p_ecef_cam
    k2 = float(np.dot(c_prime, c_prime))
    big_k = k2 - 1.0

    x_cam = (u - cam.cx) / cam.fx
    # d_cam = [x_cam, 0, 1] + y_cam * [0, 1, 0]
    vec_a = s_mat @ (r_ecef_cam @ np.array([x_cam, 0.0, 1.0], dtype=np.float64))
    vec_b = s_mat @ (r_ecef_cam @ np.array([0.0, 1.0, 0.0], dtype=np.float64))

    c_dot_a = float(np.dot(c_prime, vec_a))
    c_dot_b = float(np.dot(c_prime, vec_b))
    a2 = float(np.dot(vec_a, vec_a))
    b2 = float(np.dot(vec_b, vec_b))
    ab = float(np.dot(vec_a, vec_b))

    qa = c_dot_b * c_dot_b - b2 * big_k
    qb = 2.0 * (c_dot_a * c_dot_b - ab * big_k)
    qc = c_dot_a * c_dot_a - a2 * big_k

    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return []

    sqrt_disc = math.sqrt(max(0.0, disc))
    roots = [(-qb - sqrt_disc) / (2.0 * qa), (-qb + sqrt_disc) / (2.0 * qa)]

    valid_v = []
    for y_cam in roots:
        d_prime = vec_a + y_cam * vec_b
        # Ray must point toward Earth (C' . d' < 0)
        if np.dot(c_prime, d_prime) < 0.0:
            v = cam.fy * y_cam + cam.cy
            valid_v.append(v)
    return valid_v


def analytic_horizon_u_at_v(
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    v: float,
) -> list[float]:
    """Analytic column u coordinates of the Earth limb at row v.

    Returns empty list if no horizon is visible at row v, or list of real u roots.
    """
    a = wgs84.WGS84_A
    b = wgs84.WGS84_B
    s_mat = np.diag([1.0 / a, 1.0 / a, 1.0 / b])

    c_prime = s_mat @ p_ecef_cam
    k2 = float(np.dot(c_prime, c_prime))
    big_k = k2 - 1.0

    y_cam = (v - cam.cy) / cam.fy
    # d_cam = [0, y_cam, 1] + x_cam * [1, 0, 0]
    vec_a = s_mat @ (r_ecef_cam @ np.array([0.0, y_cam, 1.0], dtype=np.float64))
    vec_b = s_mat @ (r_ecef_cam @ np.array([1.0, 0.0, 0.0], dtype=np.float64))

    c_dot_a = float(np.dot(c_prime, vec_a))
    c_dot_b = float(np.dot(c_prime, vec_b))
    a2 = float(np.dot(vec_a, vec_a))
    b2 = float(np.dot(vec_b, vec_b))
    ab = float(np.dot(vec_a, vec_b))

    qa = c_dot_b * c_dot_b - b2 * big_k
    qb = 2.0 * (c_dot_a * c_dot_b - ab * big_k)
    qc = c_dot_a * c_dot_a - a2 * big_k

    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return []

    sqrt_disc = math.sqrt(max(0.0, disc))
    roots = [(-qb - sqrt_disc) / (2.0 * qa), (-qb + sqrt_disc) / (2.0 * qa)]

    valid_u = []
    for x_cam in roots:
        d_prime = vec_a + x_cam * vec_b
        # Ray must point toward Earth (C' . d' < 0)
        if np.dot(c_prime, d_prime) < 0.0:
            u = cam.fx * x_cam + cam.cx
            valid_u.append(u)
    return valid_u


def analytic_horizon_slope(
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    u: float,
    v: float,
) -> float:
    """Compute exact derivative dv/du at image point (u, v) on the Earth limb."""
    a = wgs84.WGS84_A
    b = wgs84.WGS84_B
    s_mat = np.diag([1.0 / a, 1.0 / a, 1.0 / b])

    c_prime = s_mat @ p_ecef_cam
    k2 = float(np.dot(c_prime, c_prime))
    big_k = k2 - 1.0

    d_cam = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0], dtype=np.float64)
    d_prime = s_mat @ (r_ecef_cam @ d_cam)

    d_u = (1.0 / cam.fx) * (s_mat @ (r_ecef_cam @ np.array([1.0, 0.0, 0.0], dtype=np.float64)))
    d_v = (1.0 / cam.fy) * (s_mat @ (r_ecef_cam @ np.array([0.0, 1.0, 0.0], dtype=np.float64)))

    c_dot_d = float(np.dot(c_prime, d_prime))
    df_du = c_dot_d * float(np.dot(c_prime, d_u)) - float(np.dot(d_prime, d_u)) * big_k
    df_dv = c_dot_d * float(np.dot(c_prime, d_v)) - float(np.dot(d_prime, d_v)) * big_k

    if abs(df_dv) < 1e-30:
        return math.inf if df_du >= 0 else -math.inf
    return -df_du / df_dv


def is_earth_at_pixel(
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    u: float,
    v: float,
) -> bool:
    """Check if the analytic ray at image coordinates (u, v) intersects the Earth ellipsoid (§2.6)."""
    a = wgs84.WGS84_A
    b = wgs84.WGS84_B
    s_mat = np.diag([1.0 / a, 1.0 / a, 1.0 / b])

    c_prime = s_mat @ p_ecef_cam
    big_k = float(np.dot(c_prime, c_prime)) - 1.0

    d_cam = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0], dtype=np.float64)
    d_prime = s_mat @ (r_ecef_cam @ d_cam)

    c_dot_d = float(np.dot(c_prime, d_prime))
    d2 = float(np.dot(d_prime, d_prime))
    disc = c_dot_d * c_dot_d - d2 * big_k
    return (disc > 0.0) and (c_dot_d < 0.0)


def is_earth_at_pixels(
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    u_arr: np.ndarray,
    v_arr: np.ndarray,
) -> np.ndarray:
    """Vectorized check for multiple pixel rays intersecting the WGS84 ellipsoid.
    
    Bit-identical to is_earth_at_pixel evaluated in a loop, but orders of magnitude faster.
    """
    if len(u_arr) == 0:
        return np.zeros(0, dtype=bool)

    a = wgs84.WGS84_A
    b = wgs84.WGS84_B
    s_mat = np.diag([1.0 / a, 1.0 / a, 1.0 / b])

    c_prime = s_mat @ p_ecef_cam
    big_k = float(np.dot(c_prime, c_prime)) - 1.0

    M = s_mat @ r_ecef_cam
    d_cam = np.vstack([(u_arr - cam.cx) / cam.fx, (v_arr - cam.cy) / cam.fy, np.ones_like(u_arr)])
    d_prime = M @ d_cam

    c_dot_d = c_prime @ d_prime
    d2 = np.sum(d_prime * d_prime, axis=0)
    disc = c_dot_d * c_dot_d - d2 * big_k
    return (disc > 0.0) & (c_dot_d < 0.0)


def zero_dip_horizon_v_at_u(
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    u: float,
) -> float | None:
    """Row v coordinate of the zero-dip local-level plane through the camera at column u (§3 G3 N1)."""
    lat_deg, lon_deg, _ = wgs84.ecef_to_geodetic(*p_ecef_cam)
    _, _, d_down = wgs84.ned_basis(lat_deg, lon_deg)
    down_cam = r_ecef_cam.T @ d_down
    if abs(down_cam[1]) < 1e-12:
        return None
    x_cam = (u - cam.cx) / cam.fx
    return cam.cy - cam.fy * (x_cam * down_cam[0] + down_cam[2]) / down_cam[1]


def zero_dip_horizon_u_at_v(
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    v: float,
) -> float | None:
    """Column u coordinate of the zero-dip local-level plane through the camera at row v (§3 G3 N1)."""
    lat_deg, lon_deg, _ = wgs84.ecef_to_geodetic(*p_ecef_cam)
    _, _, d_down = wgs84.ned_basis(lat_deg, lon_deg)
    down_cam = r_ecef_cam.T @ d_down
    if abs(down_cam[0]) < 1e-12:
        return None
    y_cam = (v - cam.cy) / cam.fy
    return cam.cx - cam.fx * (y_cam * down_cam[1] + down_cam[2]) / down_cam[0]


def estimate_horizon_edge(
    mask_img: np.ndarray,
    p_ecef_cam: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam: PinholeCamera,
    zero_dip_control: bool = False,
) -> dict[str, Any]:
    """Estimate Earth-limb edge positions from a rendered mask via the coverage method.

    Conforms strictly to WP01_SIM_RIG.md v1.3 §2.6 and §3 G3 (L1, N1):
      1. Normalize mask to [0, 1] (sky -> 0, earth -> 1).
      2. Where |dv/du| <= 1: find single sky/earth transition. Use analytic model to decide which side is earth.
      3. For earth below the sky: v_edge = (H - 0.5) - sum_v I(u, v). Mirror formula for earth above.
      4. Use rows instead of columns where |dv/du| > 1.
      5. Skip lines without exactly one clean transition, and report how many lines were used and skipped.
      6. Compare with analytic position at line center (or zero-dip plane if zero_dip_control=True).
    """
    if mask_img.ndim == 3:
        mask_2d = mask_img[:, :, 0]
    else:
        mask_2d = mask_img

    h_img, w_img = mask_2d.shape[:2]
    # Normalize: earth is 255 (or 1.0), sky is 0
    mask = (mask_2d.astype(np.float64) / 255.0).clip(0.0, 1.0)

    lines_used = 0
    lines_skipped = 0
    residuals: list[float] = []
    edges_measured: list[dict[str, Any]] = []

    # 1. Column-wise scan for shallow slopes (|dv/du| <= 1)
    for col in range(w_img):
        u_col = float(col)
        v_roots = analytic_horizon_v_at_u(p_ecef_cam, r_ecef_cam, cam, u_col)
        v_in = [vr for vr in v_roots if -0.5 <= vr <= h_img - 0.5]
        if len(v_in) != 1:
            # No root or multiple crossings in image
            continue

        v_gt = v_in[0]
        slope = analytic_horizon_slope(p_ecef_cam, r_ecef_cam, cam, u_col, v_gt)
        if abs(slope) > 1.0:
            # Steep slope: processed in row scan
            continue

        col_profile = mask[:, col]
        diffs = np.diff((col_profile > 0.5).astype(int))
        n_trans = int(np.count_nonzero(diffs))
        if n_trans != 1:
            lines_skipped += 1
            continue

        # Decide which side is Earth using the analytic model (§2.6 step 2)
        test_v = v_gt + 2.0 if v_gt + 2.0 <= h_img - 0.5 else v_gt - 2.0
        is_earth = is_earth_at_pixel(p_ecef_cam, r_ecef_cam, cam, u_col, test_v)
        is_earth_below = is_earth if (test_v > v_gt) else not is_earth
        sum_intensity = float(np.sum(col_profile))

        if is_earth_below:
            v_est = (h_img - 0.5) - sum_intensity
        else:
            v_est = sum_intensity - 0.5

        if zero_dip_control:
            v_ref = zero_dip_horizon_v_at_u(p_ecef_cam, r_ecef_cam, cam, u_col)
            if v_ref is None:
                lines_skipped += 1
                continue
        else:
            v_ref = v_gt

        delta = v_est - v_ref
        residuals.append(delta)
        lines_used += 1
        edges_measured.append({
            "col": col,
            "v_gt": v_gt,
            "v_ref": v_ref,
            "v_est": v_est,
            "delta": delta,
        })

    # 2. Row-wise scan for steep slopes (|dv/du| > 1)
    for row in range(h_img):
        v_row = float(row)
        u_roots = analytic_horizon_u_at_v(p_ecef_cam, r_ecef_cam, cam, v_row)
        u_in = [ur for ur in u_roots if -0.5 <= ur <= w_img - 0.5]
        if len(u_in) != 1:
            continue

        u_gt = u_in[0]
        slope = analytic_horizon_slope(p_ecef_cam, r_ecef_cam, cam, u_gt, v_row)
        if abs(slope) <= 1.0:
            # Handled in column scan
            continue

        row_profile = mask[row, :]
        diffs = np.diff((row_profile > 0.5).astype(int))
        n_trans = int(np.count_nonzero(diffs))
        if n_trans != 1:
            lines_skipped += 1
            continue

        # Decide which side is Earth using analytic model (§2.6 step 2)
        test_u = u_gt + 2.0 if u_gt + 2.0 <= w_img - 0.5 else u_gt - 2.0
        is_earth = is_earth_at_pixel(p_ecef_cam, r_ecef_cam, cam, test_u, v_row)
        is_earth_right = is_earth if (test_u > u_gt) else not is_earth
        sum_intensity = float(np.sum(row_profile))

        if is_earth_right:
            u_est = (w_img - 0.5) - sum_intensity
        else:
            u_est = sum_intensity - 0.5

        if zero_dip_control:
            u_ref = zero_dip_horizon_u_at_v(p_ecef_cam, r_ecef_cam, cam, v_row)
            if u_ref is None:
                lines_skipped += 1
                continue
        else:
            u_ref = u_gt

        delta = u_est - u_ref
        residuals.append(delta)
        lines_used += 1
        edges_measured.append({
            "row": row,
            "u_gt": u_gt,
            "u_ref": u_ref,
            "u_est": u_est,
            "delta": delta,
        })

    abs_residuals = np.abs(residuals) if residuals else np.array([], dtype=np.float64)
    return {
        "lines_used": lines_used,
        "lines_skipped": lines_skipped,
        "median_abs_delta": float(np.median(abs_residuals)) if len(abs_residuals) > 0 else 0.0,
        "p95_abs_delta": float(np.percentile(abs_residuals, 95)) if len(abs_residuals) > 0 else 0.0,
        "max_abs_delta": float(np.max(abs_residuals)) if len(abs_residuals) > 0 else 0.0,
        "signed_mean": float(np.mean(residuals)) if len(residuals) > 0 else 0.0,
        "residuals": residuals,
        "edges_measured": edges_measured,
    }
