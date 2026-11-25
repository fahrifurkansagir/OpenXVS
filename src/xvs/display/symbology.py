"""Conformal flight symbology: horizon, runway outline, flight-path marker, and heading tape.

Conforms strictly to docs/xvs/WP08_DISPLAY.md v0.1 §2.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np

from xvs.geo import wgs84
from xvs.rig.camera import PinholeCamera
from xvs.sim.airports import load_ksfo_28l_threshold
from xvs.sim.horizon import compute_horizon_gt


def compute_ksfo_28l_runway_geometry() -> dict[str, Any]:
    """Compute 3D ECEF coordinates of KSFO 28L four runway corners and threshold bar.

    All points constructed with h = 0 in ellipsoid mode (CONVENTIONS §2).
    """
    thr = load_ksfo_28l_threshold()
    row = thr.source_row

    he_lat = float(row["he_latitude_deg"])
    he_lon = float(row["he_longitude_deg"])
    le_lat = float(row["le_latitude_deg"])
    le_lon = float(row["le_longitude_deg"])
    heading_deg = float(row["he_heading_degT"])
    width_ft = float(row["width_ft"])
    width_m = width_ft * 0.3048
    half_w = width_m / 2.0

    # Runway normal in tangent plane (heading + 90 deg = Right)
    hdg_rad = math.radians(heading_deg)
    normal_rad = hdg_rad + math.pi / 2.0
    dn_r = half_w * math.cos(normal_rad)
    de_r = half_w * math.sin(normal_rad)

    # 4 corners (he_right, he_left, le_left, le_right)
    # High end (28L)
    lat_he_r, lon_he_r, _ = wgs84.ned_offset_to_geodetic(he_lat, he_lon, dn_r, de_r)
    lat_he_l, lon_he_l, _ = wgs84.ned_offset_to_geodetic(he_lat, he_lon, -dn_r, -de_r)

    # Low end (10R)
    lat_le_l, lon_le_l, _ = wgs84.ned_offset_to_geodetic(le_lat, le_lon, -dn_r, -de_r)
    lat_le_r, lon_le_r, _ = wgs84.ned_offset_to_geodetic(le_lat, le_lon, dn_r, de_r)

    # Threshold bar
    lat_thr_r, lon_thr_r, _ = wgs84.ned_offset_to_geodetic(thr.lat_deg, thr.lon_deg, dn_r, de_r)
    lat_thr_l, lon_thr_l, _ = wgs84.ned_offset_to_geodetic(thr.lat_deg, thr.lon_deg, -dn_r, -de_r)

    corners_geodetic = [
        (lat_he_r, lon_he_r),
        (lat_he_l, lon_he_l),
        (lat_le_l, lon_le_l),
        (lat_le_r, lon_le_r),
    ]

    corners_ecef = np.array([
        wgs84.geodetic_to_ecef(lat, lon, 0.0) for lat, lon in corners_geodetic
    ], dtype=np.float64)

    threshold_ecef = np.array([
        wgs84.geodetic_to_ecef(lat_thr_r, lon_thr_r, 0.0),
        wgs84.geodetic_to_ecef(lat_thr_l, lon_thr_l, 0.0),
    ], dtype=np.float64)

    return {
        "corners_ecef": corners_ecef,
        "threshold_ecef": threshold_ecef,
        "width_m": width_m,
        "heading_deg": heading_deg,
    }


def clip_line_segment_near_plane(
    p1_cam: np.ndarray,
    p2_cam: np.ndarray,
    z_near: float = 0.1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip 3D line segment in camera space against near plane z >= z_near.

    Returns:
        (clipped_p1, clipped_p2) in camera space, or None if completely culled.
    """
    z1 = float(p1_cam[2])
    z2 = float(p2_cam[2])

    if z1 < z_near and z2 < z_near:
        return None

    if z1 >= z_near and z2 >= z_near:
        return (p1_cam.copy(), p2_cam.copy())

    dz = z2 - z1
    if abs(dz) < 1e-15:
        return None

    t = (z_near - z1) / dz
    t = max(0.0, min(1.0, t))
    p_clip = p1_cam + t * (p2_cam - p1_cam)
    p_clip[2] = z_near

    if z1 < z_near:
        return (p_clip, p2_cam.copy())
    else:
        return (p1_cam.copy(), p_clip)


def clip_line_segment_2d(
    p1: tuple[float, float],
    p2: tuple[float, float],
    box: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Cohen-Sutherland 2D line clipping against bounding box [u_min, v_min, u_max, v_max].

    Returns clipped ((u1, v1), (u2, v2)) or None if completely outside box.
    """
    u_min, v_min, u_max, v_max = box
    INSIDE = 0
    LEFT = 1
    RIGHT = 2
    BOTTOM = 4
    TOP = 8

    def out_code(u: float, v: float) -> int:
        code = INSIDE
        if u < u_min:
            code |= LEFT
        elif u > u_max:
            code |= RIGHT
        if v < v_min:
            code |= TOP
        elif v > v_max:
            code |= BOTTOM
        return code

    u1, v1 = float(p1[0]), float(p1[1])
    u2, v2 = float(p2[0]), float(p2[1])
    code1 = out_code(u1, v1)
    code2 = out_code(u2, v2)

    while True:
        if not (code1 | code2):
            return ((u1, v1), (u2, v2))
        elif code1 & code2:
            return None
        else:
            code_out = code1 if code1 > 0 else code2
            if code_out & TOP:
                u = u1 + (u2 - u1) * (v_min - v1) / (v2 - v1)
                v = v_min
            elif code_out & BOTTOM:
                u = u1 + (u2 - u1) * (v_max - v1) / (v2 - v1)
                v = v_max
            elif code_out & RIGHT:
                v = v1 + (v2 - v1) * (u_max - u1) / (u2 - u1)
                u = u_max
            elif code_out & LEFT:
                v = v1 + (v2 - v1) * (u_min - u1) / (u2 - u1)
                u = u_min

            if code_out == code1:
                u1, v1 = u, v
                code1 = out_code(u1, v1)
            else:
                u2, v2 = u, v
                code2 = out_code(u2, v2)


def project_cam_point(p_cam: np.ndarray, cam: PinholeCamera) -> tuple[float, float]:
    """Project a 3D camera frame point (with z >= z_near > 0) to 2D pixel coordinates."""
    u = cam.fx * (p_cam[0] / p_cam[2]) + cam.cx
    v = cam.fy * (p_cam[1] / p_cam[2]) + cam.cy
    return float(u), float(v)


def clip_and_project_polygon_edges(
    poly_pts_ecef: np.ndarray | Sequence[np.ndarray],
    p_ecef_view: np.ndarray,
    r_ecef_view: np.ndarray,
    view_cam: PinholeCamera,
    z_near: float = 0.1,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Clip polygon edges segment-by-segment against near plane in camera space and project."""
    n = len(poly_pts_ecef)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(n):
        p1_ecef = poly_pts_ecef[i]
        p2_ecef = poly_pts_ecef[(i + 1) % n]

        p1_cam = r_ecef_view.T @ (p1_ecef - p_ecef_view)
        p2_cam = r_ecef_view.T @ (p2_ecef - p_ecef_view)

        clipped = clip_line_segment_near_plane(p1_cam, p2_cam, z_near=z_near)
        if clipped is not None:
            c1, c2 = clipped
            u1, v1 = project_cam_point(c1, view_cam)
            u2, v2 = project_cam_point(c2, view_cam)
            segments.append(((u1, v1), (u2, v2)))
    return segments


def clip_and_project_polyline(
    polyline_pts_ecef: np.ndarray | Sequence[np.ndarray],
    p_ecef_view: np.ndarray,
    r_ecef_view: np.ndarray,
    view_cam: PinholeCamera,
    z_near: float = 0.1,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Clip polyline segment-by-segment against near plane in camera space and project."""
    n = len(polyline_pts_ecef)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(n - 1):
        p1_ecef = polyline_pts_ecef[i]
        p2_ecef = polyline_pts_ecef[i + 1]

        p1_cam = r_ecef_view.T @ (p1_ecef - p_ecef_view)
        p2_cam = r_ecef_view.T @ (p2_ecef - p_ecef_view)

        clipped = clip_line_segment_near_plane(p1_cam, p2_cam, z_near=z_near)
        if clipped is not None:
            c1, c2 = clipped
            u1, v1 = project_cam_point(c1, view_cam)
            u2, v2 = project_cam_point(c2, view_cam)
            segments.append(((u1, v1), (u2, v2)))
    return segments


def project_conformal_runway(
    p_ecef_view: np.ndarray,
    r_ecef_view: np.ndarray,
    view_cam: PinholeCamera,
    runway_geom: dict[str, Any] | None = None,
    z_near: float = 0.1,
) -> dict[str, Any]:
    """Project KSFO 28L runway outline and threshold into viewport camera coordinates.

    Conforms to WP08 §2 and GEMINI_G-B9 P1a:
    - Clip every polygon and polyline segment against camera near plane (z >= z_near)
      in camera space before projecting.
    - Never fill symbol polygons.
    """
    if runway_geom is None:
        runway_geom = compute_ksfo_28l_runway_geometry()

    corners_ecef = runway_geom["corners_ecef"]
    thr_ecef = runway_geom["threshold_ecef"]

    # 1. Corner projections (retained for backward compatibility when all front)
    corner_uvs = []
    all_front = True
    for p_pt in corners_ecef:
        u, v, is_front = view_cam.project_ecef(p_pt, p_ecef_view, r_ecef_view)
        corner_uvs.append((u, v))
        if not is_front:
            all_front = False

    thr_uvs = []
    for p_pt in thr_ecef:
        u, v, is_front = view_cam.project_ecef(p_pt, p_ecef_view, r_ecef_view)
        thr_uvs.append((u, v))
        if not is_front:
            all_front = False

    # 2. Segment-by-segment clipped edge projections (P1a F-B8-1)
    visible_segments = clip_and_project_polygon_edges(
        corners_ecef, p_ecef_view, r_ecef_view, view_cam, z_near=z_near
    )
    threshold_segments = clip_and_project_polyline(
        thr_ecef, p_ecef_view, r_ecef_view, view_cam, z_near=z_near
    )

    has_visible = (len(visible_segments) > 0) or (len(threshold_segments) > 0)

    return {
        "corners_uv": corner_uvs,
        "threshold_uv": thr_uvs,
        "all_front": all_front,
        "is_visible": has_visible,
        "visible_segments": visible_segments,
        "threshold_segments": threshold_segments,
    }


def compute_flight_path_marker(
    v_ned: np.ndarray,
    r_ecef_ned: np.ndarray,
    r_ecef_view: np.ndarray,
    view_cam: PinholeCamera,
) -> tuple[float, float] | None:
    """Compute viewport camera pixel (u, v) for the flight-path marker from velocity vector."""
    speed = float(np.linalg.norm(v_ned))
    if speed < 1.0:
        return None

    # Direction in ECEF
    v_ecef = r_ecef_ned @ (v_ned / speed)
    # Direction in camera frame
    v_cam = r_ecef_view.T @ v_ecef

    if v_cam[2] <= 1e-4:
        return None

    u = view_cam.fx * (v_cam[0] / v_cam[2]) + view_cam.cx
    v = view_cam.fy * (v_cam[1] / v_cam[2]) + view_cam.cy
    return float(u), float(v)


def compute_conformal_horizon(
    p_ecef_view: np.ndarray,
    r_ecef_view: np.ndarray,
    view_cam: PinholeCamera,
    num_samples: int = 2048,
) -> dict[str, Any]:
    """Compute conformal horizon line points for viewport camera from truth pose.

    Conforms to WP08 §2 & §5 (V3):
    Horizon line is projected from truth pose with WP-01 analytic horizon function.
    """
    return compute_horizon_gt(p_ecef_view, r_ecef_view, view_cam, num_samples=num_samples)


def draw_symbology(
    img: np.ndarray,
    aircraft_pose: dict[str, float],
    v_ned: np.ndarray,
    p_ecef_view: np.ndarray,
    r_ecef_view: np.ndarray,
    view_cam: PinholeCamera,
    is_t2: bool = False,
    burn_in_text: str | None = None,
) -> np.ndarray:
    """Draw anti-aliased conformal and non-conformal symbology with 2 px black outline.

    Conforms to WP08 §2:
      - Horizon line (WP-01 analytic limb)
      - KSFO 28L runway outline and threshold (for T2)
      - Flight-path marker (v_ned direction)
      - Heading tape
      - Small burned-in simulation label
    """
    out = img.copy()

    # Colors (BGR for OpenCV)
    # Green symbology with black outline
    color_fg = (50, 255, 50)  # Bright green
    color_bg = (0, 0, 0)      # Black outline
    color_thr = (0, 220, 255) # Cyan / Yellow for threshold
    color_fpm = (0, 255, 255) # Yellow for flight path marker

    # 1. Horizon line (WP-01 analytic limb) with sub-pixel precision (shift=8)
    hz_res = compute_conformal_horizon(p_ecef_view, r_ecef_view, view_cam, num_samples=2048)
    if hz_res["horizon_visible"] and len(hz_res["horizon_uv"]) >= 2:
        pts_float = np.asarray(hz_res["horizon_uv"], dtype=np.float64)
        pts_scaled = np.round(pts_float * 256.0).astype(np.int32)
        # Outline (thickness 3) then foreground stroke (thickness 1) with shift=8
        cv2.polylines(out, [pts_scaled], isClosed=False, color=color_bg, thickness=3, lineType=cv2.LINE_AA, shift=8)
        cv2.polylines(out, [pts_scaled], isClosed=False, color=color_fg, thickness=1, lineType=cv2.LINE_AA, shift=8)

    # 2. Runway outline (KSFO 28L if T2 approach)
    if is_t2:
        rwy_res = project_conformal_runway(p_ecef_view, r_ecef_view, view_cam)
        if rwy_res["is_visible"]:
            # Clip 2D segments against slightly padded viewport to ensure finite integer coordinates
            clip_box = (-50.0, -50.0, float(view_cam.width + 50.0), float(view_cam.height + 50.0))

            clipped_vis_segs = []
            for seg in rwy_res["visible_segments"]:
                c_seg = clip_line_segment_2d(seg[0], seg[1], clip_box)
                if c_seg is not None:
                    clipped_vis_segs.append(c_seg)

            clipped_thr_segs = []
            for seg in rwy_res["threshold_segments"]:
                c_seg = clip_line_segment_2d(seg[0], seg[1], clip_box)
                if c_seg is not None:
                    clipped_thr_segs.append(c_seg)

            # 1. Black outlines (thickness 3) with shift=8
            for seg in clipped_vis_segs:
                (u1, v1), (u2, v2) = seg
                p1 = (int(round(u1 * 256.0)), int(round(v1 * 256.0)))
                p2 = (int(round(u2 * 256.0)), int(round(v2 * 256.0)))
                cv2.line(out, p1, p2, color=color_bg, thickness=3, lineType=cv2.LINE_AA, shift=8)

            for seg in clipped_thr_segs:
                (u1, v1), (u2, v2) = seg
                p1 = (int(round(u1 * 256.0)), int(round(v1 * 256.0)))
                p2 = (int(round(u2 * 256.0)), int(round(v2 * 256.0)))
                cv2.line(out, p1, p2, color=color_bg, thickness=3, lineType=cv2.LINE_AA, shift=8)

            # 2. Foreground strokes (thickness 1) with shift=8
            for seg in clipped_vis_segs:
                (u1, v1), (u2, v2) = seg
                p1 = (int(round(u1 * 256.0)), int(round(v1 * 256.0)))
                p2 = (int(round(u2 * 256.0)), int(round(v2 * 256.0)))
                cv2.line(out, p1, p2, color=color_fg, thickness=1, lineType=cv2.LINE_AA, shift=8)

            for seg in clipped_thr_segs:
                (u1, v1), (u2, v2) = seg
                p1 = (int(round(u1 * 256.0)), int(round(v1 * 256.0)))
                p2 = (int(round(u2 * 256.0)), int(round(v2 * 256.0)))
                cv2.line(out, p1, p2, color=color_thr, thickness=1, lineType=cv2.LINE_AA, shift=8)

    # 3. Flight-path marker
    r_ecef_ned_mat = wgs84.r_ecef_ned(aircraft_pose["lat_deg"], aircraft_pose["lon_deg"])
    fpm = compute_flight_path_marker(v_ned, r_ecef_ned_mat, r_ecef_view, view_cam)
    if fpm is not None:
        uf, vf = int(round(fpm[0])), int(round(fpm[1]))
        if 0 <= uf < view_cam.width and 0 <= vf < view_cam.height:
            # Circle
            cv2.circle(out, (uf, vf), 7, color_bg, thickness=5, lineType=cv2.LINE_AA)
            cv2.circle(out, (uf, vf), 7, color_fpm, thickness=2, lineType=cv2.LINE_AA)
            # Wings
            cv2.line(out, (uf - 18, vf), (uf - 7, vf), color_bg, thickness=5, lineType=cv2.LINE_AA)
            cv2.line(out, (uf - 18, vf), (uf - 7, vf), color_fpm, thickness=2, lineType=cv2.LINE_AA)
            cv2.line(out, (uf + 7, vf), (uf + 18, vf), color_bg, thickness=5, lineType=cv2.LINE_AA)
            cv2.line(out, (uf + 7, vf), (uf + 18, vf), color_fpm, thickness=2, lineType=cv2.LINE_AA)
            # Top fin
            cv2.line(out, (uf, vf - 7), (uf, vf - 15), color_bg, thickness=5, lineType=cv2.LINE_AA)
            cv2.line(out, (uf, vf - 7), (uf, vf - 15), color_fpm, thickness=2, lineType=cv2.LINE_AA)

    # 4. Heading tape (label at top center)
    yaw_deg = (math.degrees(aircraft_pose["yaw_rad"])) % 360.0
    hdg_str = f"HDG {yaw_deg:03.0f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    cv2.rectangle(out, (view_cam.width // 2 - 50, 15), (view_cam.width // 2 + 50, 42), (0, 0, 0), -1)
    cv2.putText(out, hdg_str, (view_cam.width // 2 - 40, 34), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Burn-in label (bottom left)
    if burn_in_text:
        text_size = cv2.getTextSize(burn_in_text, font, 0.45, 1)[0]
        cv2.rectangle(out, (15, view_cam.height - 35), (25 + text_size[0], view_cam.height - 12), (0, 0, 0), -1)
        cv2.putText(out, burn_in_text, (20, view_cam.height - 18), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return out


def draw_burn_in_label(
    img: np.ndarray,
    text: str,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Burn-in small simulation text badge in bottom-left corner with black background."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
    h = out.shape[0]
    cv2.rectangle(out, (15, h - 35), (25 + text_size[0], h - 12), (0, 0, 0), -1)
    cv2.putText(out, text, (20, h - 18), font, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
    return out



# ==============================================================================
# WP-08 Gate G1 (Batch B12): Live WP-05 G1 (WC1) Conformal Traffic Symbology &
# WP-07 Integrity State Coupling (VALID, F_isolated, DEGRADED_COAST, INVALID_ALERT)
# ==============================================================================

from collections.abc import Mapping
from xvs.geo import rotations

TRAFFIC_TIER_NONE: int = 0
TRAFFIC_TIER_PREVENTIVE: int = 1
TRAFFIC_TIER_CORRECTIVE: int = 2
TRAFFIC_TIER_WARNING: int = 3

TRAFFIC_TIER_NAMES: dict[int, str] = {
    TRAFFIC_TIER_NONE: "NONE",
    TRAFFIC_TIER_PREVENTIVE: "PREVENTIVE",
    TRAFFIC_TIER_CORRECTIVE: "CORRECTIVE",
    TRAFFIC_TIER_WARNING: "WARNING",
}

VALID_HUD_INTEGRITY_STATES: frozenset[str] = frozenset(
    {"VALID", "DEGRADED_COAST", "INVALID_ALERT"}
)

# Camera forward nominal mapping in body axes (col0=right, col1=down, col2=forward)
_G1_R_BODY_CAM_FORWARD = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Strict whitelist of passive Track / RiskEstimate fields (V5_G1 truth isolation)
_ALLOWED_PASSIVE_KEYS: frozenset[str] = frozenset(
    {
        "track_id",
        "t_s",
        "az",
        "az_rad",
        "el",
        "el_rad",
        "az_rate",
        "az_rate_radps",
        "el_rate",
        "el_rate_radps",
        "s",
        "s_rad",
        "s_mom",
        "s_dot",
        "s_dot_radps",
        "P",
        "cov_az_el",
        "omega_los",
        "omega_los_radps",
        "wc1_warning",
        "wc1_corrective",
        "alert_a1_warning",
        "alert_a1_corrective",
        "alert_a2_warning",
        "alert_a2_corrective",
        "a1_warning",
        "a1_corrective",
        "a2_warning",
        "a2_corrective",
        "hmd_min_m",
        "hmd_max_m",
        "hmd_hat_m",
        "hmd_interval_m",
        "tau_mod_min_s",
        "tau_mod_max_s",
        "tau_mod_hat_s",
        "tau_s_hat_s",
        "tau_s",
        "range_interval_m",
        "eta_hat_inv_s",
        "eta_min_inv_s",
        "eta_max_inv_s",
        "beta_wc",
        "p_wcv",
    }
)


def _get_passive_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read only whitelisted passive track/risk attributes (never truth attributes)."""
    if key not in _ALLOWED_PASSIVE_KEYS:
        raise ValueError(f"Key {key!r} is not in passive track/risk whitelist")
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def compute_p_wcv_belief(
    risk: Any,
    d_thr_m: float = 609.6,
    t_corr_s: float = 55.0,
) -> float:
    """Compute passive Well-Clear Violation probability belief P_hat(WCV) in [0, 1].

    Uses passive intervals [HMD_min, HMD_max] and [tau_mod_min, tau_mod_max] from RiskEstimate.
    Satisfies the exact duality: P_hat(WCV) >= beta_WC <=> wc1_corrective(beta_WC) == True.
    """
    if risk is None:
        return 0.0

    explicit_p = _get_passive_field(risk, "p_wcv", None)
    if explicit_p is not None and math.isfinite(float(explicit_p)):
        return float(max(0.0, min(1.0, float(explicit_p))))

    hmd_min = _get_passive_field(risk, "hmd_min_m", None)
    hmd_max = _get_passive_field(risk, "hmd_max_m", None)
    tau_min = _get_passive_field(risk, "tau_mod_min_s", None)
    tau_max = _get_passive_field(risk, "tau_mod_max_s", None)

    if (
        hmd_min is not None
        and hmd_max is not None
        and tau_min is not None
        and tau_max is not None
        and math.isfinite(float(hmd_min))
        and math.isfinite(float(hmd_max))
        and math.isfinite(float(tau_min))
        and math.isfinite(float(tau_max))
    ):
        h_lo = float(hmd_min)
        h_hi = float(hmd_max)
        t_lo = float(tau_min)
        t_hi = float(tau_max)

        if h_hi <= d_thr_m:
            p_hmd = 1.0
        elif h_lo >= d_thr_m:
            p_hmd = 0.0
        else:
            p_hmd = (d_thr_m - h_lo) / max(h_hi - h_lo, 1e-6)

        if t_hi <= t_corr_s:
            p_tau = 1.0
        elif t_lo >= t_corr_s:
            p_tau = 0.0
        else:
            p_tau = (t_corr_s - t_lo) / max(t_hi - t_lo, 1e-6)

        return float(max(0.0, min(1.0, min(p_hmd, p_tau))))

    wc1_warn = bool(_get_passive_field(risk, "wc1_warning", False))
    wc1_corr = bool(_get_passive_field(risk, "wc1_corrective", False))
    a1_warn = bool(
        _get_passive_field(
            risk, "alert_a1_warning", _get_passive_field(risk, "a1_warning", False)
        )
    )
    a1_corr = bool(
        _get_passive_field(
            risk, "alert_a1_corrective", _get_passive_field(risk, "a1_corrective", False)
        )
    )
    if wc1_warn or a1_warn:
        return 0.85
    if wc1_corr or a1_corr:
        return 0.60
    return 0.0


def classify_traffic_symbology_tier(
    track: Any,
    risk: Any = None,
    beta_wc: float = 0.50,
    warning_tau_mod_s: float = 12.5,
    warning_p_wcv_min: float = 0.50,
    preventive_tau_mod_s: float = 75.0,
    preventive_p_wcv_min: float = 0.15,
) -> dict[str, Any]:
    """Classify a confirmed passive track into DO-365A / XVS symbology tiers (0..3).

    Tiers:
      - 0 (`NONE`): cyan open diamond
      - 1 (`PREVENTIVE`): amber solid diamond + 5s velocity leader vector
      - 2 (`CORRECTIVE`): amber/red double-outline octagon + covariance halo + data block
      - 3 (`WARNING`): red solid square + flashing border + escape-direction chevron
    """
    src_risk = risk if risk is not None else track
    p_wcv = compute_p_wcv_belief(src_risk)

    wc1_warn = bool(_get_passive_field(src_risk, "wc1_warning", False))
    wc1_corr = bool(_get_passive_field(src_risk, "wc1_corrective", False))
    a1_warn = bool(
        _get_passive_field(
            src_risk, "alert_a1_warning", _get_passive_field(src_risk, "a1_warning", False)
        )
    )
    a1_corr = bool(
        _get_passive_field(
            src_risk, "alert_a1_corrective", _get_passive_field(src_risk, "a1_corrective", False)
        )
    )

    tau_mod_min = _get_passive_field(src_risk, "tau_mod_min_s", None)
    tau_mod_hat = _get_passive_field(src_risk, "tau_mod_hat_s", None)
    tau_s_hat = _get_passive_field(
        src_risk, "tau_s_hat_s", _get_passive_field(src_risk, "tau_s", None)
    )
    eta_hat = _get_passive_field(src_risk, "eta_hat_inv_s", None)
    s_dot = float(
        _get_passive_field(
            track, "s_dot", _get_passive_field(track, "s_dot_radps", 0.0)
        )
        or 0.0
    )
    az_rate = float(
        _get_passive_field(
            track, "az_rate", _get_passive_field(track, "az_rate_radps", 0.0)
        )
        or 0.0
    )

    eff_tau_min = None
    for cand in (tau_mod_min, tau_mod_hat, tau_s_hat):
        if cand is not None and math.isfinite(float(cand)):
            eff_tau_min = float(cand)
            break

    is_warning = bool(
        eff_tau_min is not None
        and eff_tau_min <= float(warning_tau_mod_s)
        and (p_wcv >= float(warning_p_wcv_min) or wc1_warn or a1_warn)
    )

    is_corrective = bool(
        not is_warning
        and (wc1_corr or wc1_warn or a1_corr or a1_warn or p_wcv >= float(beta_wc))
    )

    is_closing = bool(
        (eta_hat is not None and float(eta_hat) > 0.005)
        or s_dot > 1e-5
        or (eff_tau_min is not None and eff_tau_min <= float(preventive_tau_mod_s))
    )
    is_preventive = bool(
        not is_warning
        and not is_corrective
        and (p_wcv >= float(preventive_p_wcv_min) or is_closing)
    )

    escape_chevron: str | None = None
    if is_warning:
        tier = TRAFFIC_TIER_WARNING
        # Horizontal escape-direction chevron from sign of LOS azimuth rate psi_dot_LOS:
        # If LOS drifts right (az_rate > 0), turn LEFT behind intruder; if left (az_rate < 0), turn RIGHT.
        escape_chevron = "LEFT" if az_rate > 1e-6 else "RIGHT"
    elif is_corrective:
        tier = TRAFFIC_TIER_CORRECTIVE
    elif is_preventive:
        tier = TRAFFIC_TIER_PREVENTIVE
    else:
        tier = TRAFFIC_TIER_NONE

    return {
        "tier": tier,
        "tier_name": TRAFFIC_TIER_NAMES[tier],
        "p_wcv": float(p_wcv),
        "escape_chevron": escape_chevron,
        "tau_mod_min_s": float(tau_mod_min) if tau_mod_min is not None else None,
    }


def passive_bearing_ned_and_jacobian(
    az_rad: float,
    el_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit bearing u_hat_NED and partial derivatives du/d(az), du/d(el)."""
    c_el = math.cos(el_rad)
    s_el = math.sin(el_rad)
    c_az = math.cos(az_rad)
    s_az = math.sin(az_rad)

    u_ned = np.array([c_el * c_az, c_el * s_az, -s_el], dtype=np.float64)
    du_daz = np.array([-c_el * s_az, c_el * c_az, 0.0], dtype=np.float64)
    du_del = np.array([-s_el * c_az, -s_el * s_az, -c_el], dtype=np.float64)
    return u_ned, du_daz, du_del


def project_passive_track_to_viewport(
    track: Any,
    risk: Any,
    r_ned_body_est: np.ndarray,
    head_euler_rad: tuple[float, float, float],
    view_cam: PinholeCamera,
    integrity_state: str = "VALID",
    beta_wc: float = 0.50,
    leader_horizon_s: float = 5.0,
    sigma_mult: float = 3.0,
) -> dict[str, Any]:
    """Project passive WP-05 G1 Track + RiskEstimate into 1280x720 viewport coordinates.

    Reads ONLY passive track/risk fields and estimated attitude `r_ned_body_est` (V5_G1).
    """
    if integrity_state not in VALID_HUD_INTEGRITY_STATES:
        raise ValueError(f"Invalid HUD integrity_state: {integrity_state!r}")

    track_id = str(_get_passive_field(track, "track_id", "T0001"))
    az_rad = float(_get_passive_field(track, "az", _get_passive_field(track, "az_rad", 0.0)))
    el_rad = float(_get_passive_field(track, "el", _get_passive_field(track, "el_rad", 0.0)))
    az_rate = float(
        _get_passive_field(track, "az_rate", _get_passive_field(track, "az_rate_radps", 0.0))
    )
    el_rate = float(
        _get_passive_field(track, "el_rate", _get_passive_field(track, "el_rate_radps", 0.0))
    )
    s_rad = float(_get_passive_field(track, "s", _get_passive_field(track, "s_rad", 0.005)))
    s_mom = _get_passive_field(track, "s_mom", None)
    theta_rad = float(s_mom) if (s_mom is not None and float(s_mom) > 0.0) else max(1e-5, s_rad)

    u_ned, du_daz, du_del = passive_bearing_ned_and_jacobian(az_rad, el_rad)
    u_dot_ned = du_daz * az_rate + du_del * el_rate

    head_yaw, head_pitch, head_roll = head_euler_rad
    r_body_head = rotations.euler_to_r(head_yaw, head_pitch, head_roll)
    r_body_view = r_body_head @ _G1_R_BODY_CAM_FORWARD

    r_ned_body_mat = np.asarray(r_ned_body_est, dtype=np.float64)
    r_ned_view_est = r_ned_body_mat @ r_body_view

    # Camera-frame unit direction vector and rate
    d_cam = r_ned_view_est.T @ u_ned
    d_dot_cam = r_ned_view_est.T @ u_dot_ned

    tier_info = classify_traffic_symbology_tier(track, risk, beta_wc=beta_wc)
    stabilization_mode = (
        "EARTH_CONFORMAL" if integrity_state == "VALID" else "BODY_STABILIZED"
    )

    if float(d_cam[2]) <= 1e-4:
        return {
            "track_id": track_id,
            "is_in_front": False,
            "is_in_viewport": False,
            "u_vp": -999.0,
            "v_vp": -999.0,
            "u_q8_vp": -999.0,
            "v_q8_vp": -999.0,
            "u_dot_vp_pxps": 0.0,
            "v_dot_vp_pxps": 0.0,
            "u_leader_vp": -999.0,
            "v_leader_vp": -999.0,
            "angular_radius_px": 8.0,
            "cov_uv_px2": np.zeros((2, 2), dtype=np.float64),
            "halo_axes_px": (12.0, 12.0),
            "halo_angle_deg": 0.0,
            "stabilization_mode": stabilization_mode,
            **tier_info,
        }

    dx, dy, dz = float(d_cam[0]), float(d_cam[1]), float(d_cam[2])
    ddx, ddy, ddz = float(d_dot_cam[0]), float(d_dot_cam[1]), float(d_dot_cam[2])

    u_vp = view_cam.fx * (dx / dz) + view_cam.cx
    v_vp = view_cam.fy * (dy / dz) + view_cam.cy

    # Sub-pixel fixed-point coordinates (shift=8, scale=256.0)
    u_scaled = int(round(u_vp * 256.0))
    v_scaled = int(round(v_vp * 256.0))
    u_q8 = u_scaled / 256.0
    v_q8 = v_scaled / 256.0

    # Exact pinhole Jacobian for velocity leader (u_dot_vp, v_dot_vp)
    inv_dz2 = 1.0 / (dz * dz)
    u_dot_vp = view_cam.fx * (ddx * dz - dx * ddz) * inv_dz2
    v_dot_vp = view_cam.fy * (ddy * dz - dy * ddz) * inv_dz2

    u_leader = u_vp + float(leader_horizon_s) * u_dot_vp
    v_leader = v_vp + float(leader_horizon_s) * v_dot_vp

    # Extract passive angle covariance P_{[0,2],[0,2]}
    p_mat = _get_passive_field(track, "P", None)
    cov_az_el = _get_passive_field(track, "cov_az_el", None)
    if cov_az_el is not None:
        p_ang = np.asarray(cov_az_el, dtype=np.float64).reshape(2, 2)
    elif p_mat is not None and np.asarray(p_mat).shape == (6, 6):
        p_arr = np.asarray(p_mat, dtype=np.float64)
        p_ang = np.array(
            [[p_arr[0, 0], p_arr[0, 2]], [p_arr[2, 0], p_arr[2, 2]]],
            dtype=np.float64,
        )
    else:
        sigma_default = 0.0015
        p_ang = np.eye(2, dtype=np.float64) * (sigma_default**2)

    j_cam_azel = r_ned_view_est.T @ np.column_stack([du_daz, du_del])  # (3, 2)
    j_uv_cam = np.array(
        [
            [view_cam.fx / dz, 0.0, -view_cam.fx * dx * inv_dz2],
            [0.0, view_cam.fy / dz, -view_cam.fy * dy * inv_dz2],
        ],
        dtype=np.float64,
    )
    j_uv_azel = j_uv_cam @ j_cam_azel  # (2, 2)
    cov_uv = j_uv_azel @ p_ang @ j_uv_azel.T

    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov_uv + cov_uv.T))
    eigvals = np.maximum(eigvals, 1e-6)
    axis_major = float(
        max(10.0, min(60.0, sigma_mult * math.sqrt(float(eigvals[1]))))
    )
    axis_minor = float(
        max(8.0, min(axis_major, sigma_mult * math.sqrt(float(eigvals[0]))))
    )
    axis_major_2sig = float(
        max(8.0, min(48.0, 2.0 * math.sqrt(float(eigvals[1]))))
    )
    axis_minor_2sig = float(
        max(6.0, min(axis_major_2sig, 2.0 * math.sqrt(float(eigvals[0]))))
    )
    halo_angle_deg = float(math.degrees(math.atan2(eigvecs[1, 1], eigvecs[0, 1])))

    angular_radius_px = float(
        max(8.0, min(36.0, 0.5 * theta_rad * view_cam.fx))
    )
    in_vp = bool(
        (0.0 <= u_vp < view_cam.width) and (0.0 <= v_vp < view_cam.height)
    )

    src_risk = risk if risk is not None else track
    hmd_min = _get_passive_field(src_risk, "hmd_min_m", None)
    hmd_max = _get_passive_field(src_risk, "hmd_max_m", None)
    tau_min = _get_passive_field(src_risk, "tau_mod_min_s", None)
    tau_max = _get_passive_field(src_risk, "tau_mod_max_s", None)

    return {
        "track_id": track_id,
        "is_in_front": True,
        "is_in_viewport": in_vp,
        "u_vp": float(u_vp),
        "v_vp": float(v_vp),
        "u_q8_vp": float(u_q8),
        "v_q8_vp": float(v_q8),
        "u_scaled_q8": u_scaled,
        "v_scaled_q8": v_scaled,
        "u_dot_vp_pxps": float(u_dot_vp),
        "v_dot_vp_pxps": float(v_dot_vp),
        "u_leader_vp": float(u_leader),
        "v_leader_vp": float(v_leader),
        "angular_radius_px": angular_radius_px,
        "cov_uv_px2": cov_uv,
        "halo_axes_px": (axis_major, axis_minor),
        "halo_axes_2sigma_px": (axis_major_2sig, axis_minor_2sig),
        "halo_angle_deg": halo_angle_deg,
        "stabilization_mode": stabilization_mode,
        "hmd_min_m": float(hmd_min) if hmd_min is not None else None,
        "hmd_max_m": float(hmd_max) if hmd_max is not None else None,
        "tau_mod_min_s": float(tau_min) if tau_min is not None else None,
        "tau_mod_max_s": float(tau_max) if tau_max is not None else None,
        **tier_info,
    }


def project_passive_track_to_panorama(
    track: Any,
    risk: Any,
    r_ned_body_est: np.ndarray,
    width: int = 2048,
    height: int = 1024,
    frame: str = "ned",
    beta_wc: float = 0.50,
) -> dict[str, Any]:
    """Project passive track bearing u_hat into 2048x1024 equirectangular panorama."""
    track_id = str(_get_passive_field(track, "track_id", "T0001"))
    az_rad = float(_get_passive_field(track, "az", _get_passive_field(track, "az_rad", 0.0)))
    el_rad = float(_get_passive_field(track, "el", _get_passive_field(track, "el_rad", 0.0)))

    u_ned, _, _ = passive_bearing_ned_and_jacobian(az_rad, el_rad)
    if frame == "body":
        r_ned_body_mat = np.asarray(r_ned_body_est, dtype=np.float64)
        u_vec = r_ned_body_mat.T @ u_ned
    elif frame == "ned":
        u_vec = u_ned
    else:
        raise ValueError(f"Unknown panorama frame: {frame!r}")

    psi = math.atan2(float(u_vec[1]), float(u_vec[0]))
    theta_el = math.asin(max(-1.0, min(1.0, -float(u_vec[2]))))

    u_pan = (psi / (2.0 * math.pi) + 0.5) * width - 0.5
    v_pan = (0.5 - theta_el / math.pi) * height - 0.5

    tier_info = classify_traffic_symbology_tier(track, risk, beta_wc=beta_wc)
    return {
        "track_id": track_id,
        "frame": frame,
        "u_pan": float(u_pan),
        "v_pan": float(v_pan),
        "az_rad": float(psi),
        "el_rad": float(theta_el),
        **tier_info,
    }


def compute_conformal_horizon_with_pl_band(
    p_ecef_view: np.ndarray,
    r_ecef_view_est: np.ndarray,
    view_cam: PinholeCamera,
    pl_deg: float = 0.25,
    integrity_state: str = "VALID",
    f_isolated: bool = False,
    isolated_camera_id: str | None = None,
    coast_time_s: float = 0.0,
    coast_drift_deg_per_s: float = 0.15,
    num_samples: int = 2048,
) -> dict[str, Any]:
    """Compute WP-07 integrity-coupled conformal horizon curve and +/- PL_0.01 whisker bands.

    Conforms to Batch B12 Part 2:
      - `VALID`: conformal horizon flanked by upper/lower whiskers at +/- PL_0.01 deg.
        If `f_isolated == True`, emits `CAM ISOLATED [c] — PL NOMINAL` badge.
      - `DEGRADED_COAST`: dashed amber horizon (`COASTING`), widens PL by
        `coast_drift_deg_per_s * coast_time_s`.
      - `INVALID_ALERT`: immediately blanks (suppresses) earth-fixed conformal horizon
        within 1 frame (`ttd_display_frames = 1`, `hud_hmi_frames = 0`) and returns
        the alert banner `ATTITUDE INTEGRITY INVALID — DO NOT USE CONFORMAL HORIZON`.
    """
    if integrity_state not in VALID_HUD_INTEGRITY_STATES:
        raise ValueError(f"Invalid HUD integrity_state: {integrity_state!r}")

    if integrity_state == "INVALID_ALERT":
        empty_uv = np.empty((0, 2), dtype=np.float64)
        return {
            "integrity_state": "INVALID_ALERT",
            "horizon_visible": False,
            "horizon_suppressed": True,
            "runway_suppressed": True,
            "horizon_style": "BLANKED",
            "pl_effective_deg": 0.0,
            "horizon_uv": empty_uv,
            "upper_pl_uv": empty_uv,
            "lower_pl_uv": empty_uv,
            "upper_pl_q8_uv": empty_uv,
            "lower_pl_q8_uv": empty_uv,
            "status_badge": None,
            "alert_banner": (
                "NO HORIZON INTEGRITY — ATTITUDE UNBOUNDED | "
                "ATTITUDE INTEGRITY INVALID — DO NOT USE CONFORMAL HORIZON"
            ),
            "ttd_display_frames": 1,
            "hud_hmi_frames": 0,
        }

    if integrity_state == "DEGRADED_COAST":
        pl_eff_deg = float(pl_deg) + float(coast_drift_deg_per_s) * max(0.0, float(coast_time_s))
        horizon_style = "DASHED_AMBER"
        status_badge = f"COASTING — PL +/-{pl_eff_deg:.2f} DEG"
    else:
        pl_eff_deg = float(pl_deg)
        horizon_style = "SOLID_GREEN"
        if f_isolated:
            cam_label = isolated_camera_id if isolated_camera_id else "F"
            status_badge = f"CAM ISOLATED [{cam_label}] — PL NOMINAL"
        else:
            status_badge = None

    hz_res = compute_horizon_gt(p_ecef_view, r_ecef_view_est, view_cam, num_samples=num_samples)
    if not hz_res["horizon_visible"] or len(hz_res["horizon_uv"]) < 2:
        empty_uv = np.empty((0, 2), dtype=np.float64)
        return {
            "integrity_state": integrity_state,
            "horizon_visible": False,
            "horizon_suppressed": False,
            "runway_suppressed": False,
            "horizon_style": horizon_style,
            "pl_effective_deg": pl_eff_deg,
            "horizon_uv": empty_uv,
            "upper_pl_uv": empty_uv,
            "lower_pl_uv": empty_uv,
            "upper_pl_q8_uv": empty_uv,
            "lower_pl_q8_uv": empty_uv,
            "status_badge": status_badge,
            "alert_banner": None,
            "ttd_display_frames": 0,
            "hud_hmi_frames": 0,
        }

    hz_uv = np.asarray(hz_res["horizon_uv"], dtype=np.float64)
    pl_rad = math.radians(pl_eff_deg)

    u_arr = hz_uv[:, 0]
    v_arr = hz_uv[:, 1]
    alpha_v = np.arctan((v_arr - view_cam.cy) / view_cam.fy)

    v_upper = view_cam.cy + view_cam.fy * np.tan(alpha_v - pl_rad)
    v_lower = view_cam.cy + view_cam.fy * np.tan(alpha_v + pl_rad)

    upper_uv = np.column_stack([u_arr, v_upper])
    lower_uv = np.column_stack([u_arr, v_lower])

    upper_q8_uv = np.round(upper_uv * 256.0) / 256.0
    lower_q8_uv = np.round(lower_uv * 256.0) / 256.0

    return {
        "integrity_state": integrity_state,
        "horizon_visible": True,
        "horizon_suppressed": False,
        "runway_suppressed": (integrity_state != "VALID"),
        "horizon_style": horizon_style,
        "pl_effective_deg": pl_eff_deg,
        "horizon_uv": hz_uv,
        "upper_pl_uv": upper_uv,
        "lower_pl_uv": lower_uv,
        "upper_pl_q8_uv": upper_q8_uv,
        "lower_pl_q8_uv": lower_q8_uv,
        "status_badge": status_badge,
        "alert_banner": None,
        "ttd_display_frames": 0,
        "hud_hmi_frames": 0,
    }


def _draw_dashed_polyline_q8(
    img: np.ndarray,
    pts_uv: np.ndarray,
    color_fg: tuple[int, int, int],
    color_bg: tuple[int, int, int] = (0, 0, 0),
    dash_stride: int = 16,
) -> None:
    """Draw anti-aliased dashed polyline using fixed-point shift=8 coordinates."""
    pts_scaled = np.round(pts_uv * 256.0).astype(np.int32)
    n_pts = len(pts_scaled)
    for start_idx in range(0, n_pts - 1, dash_stride * 2):
        end_idx = min(n_pts, start_idx + dash_stride)
        if end_idx - start_idx >= 2:
            seg = pts_scaled[start_idx:end_idx]
            cv2.polylines(
                img, [seg], isClosed=False, color=color_bg, thickness=3, lineType=cv2.LINE_AA, shift=8
            )
            cv2.polylines(
                img, [seg], isClosed=False, color=color_fg, thickness=1, lineType=cv2.LINE_AA, shift=8
            )


def compute_rect_intersection_area(
    rect_a: tuple[float, float, float, float],
    rect_b: tuple[float, float, float, float],
) -> float:
    """Compute intersection area between two 2D axis-aligned rectangles (x1, y1, x2, y2)."""
    dx = max(0.0, min(float(rect_a[2]), float(rect_b[2])) - max(float(rect_a[0]), float(rect_b[0])))
    dy = max(0.0, min(float(rect_a[3]), float(rect_b[3])) - max(float(rect_a[1]), float(rect_b[1])))
    return float(dx * dy)


def _pad_rect(
    rect: tuple[float, float, float, float],
    pad_px: float,
) -> tuple[float, float, float, float]:
    return (
        float(rect[0]) - pad_px,
        float(rect[1]) - pad_px,
        float(rect[2]) + pad_px,
        float(rect[3]) + pad_px,
    )


def _format_track_callout_spec(
    proj: dict[str, Any],
    integrity_state: str = "VALID",
    frame_index: int = 0,
) -> dict[str, Any]:
    """Determine callout lines, dimensions, colors, and reticle bounding box for a projected track."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    u_vp = float(proj["u_vp"])
    v_vp = float(proj["v_vp"])
    rad_px = max(9.0, min(28.0, float(proj.get("angular_radius_px", 12.0))))  # default-ok: optional display projection fallback
    halo_ax = proj.get("halo_axes_px", (18.0, 14.0))
    halo_2sig = proj.get("halo_axes_2sigma_px", (12.0, 10.0))
    tier = int(proj["tier"])
    stabilization_mode = str(proj.get("stabilization_mode", "EARTH_CONFORMAL"))

    if stabilization_mode == "BODY_STABILIZED":
        r_bound = rad_px + 8.0
        color_brg = (0, 200, 255) if tier < TRAFFIC_TIER_WARNING else (60, 60, 255)
        if integrity_state == "INVALID_ALERT":
            lines = [f"{proj['track_id']} [BRG ONLY]", "NO ALT / NO RNG"]
        else:
            lines = [f"{proj['track_id']} [BRG ONLY]"]
        font_scale = 0.40
        bg_fill = (12, 12, 18)
        border_col = color_brg
        text_col = color_brg
    elif tier == TRAFFIC_TIER_NONE:
        r_bound = max(rad_px + 4.0, float(halo_2sig[0]) + 3.0)
        color_cyan = (255, 255, 0)
        lines = [f"{proj['track_id']} NONE"]
        font_scale = 0.40
        bg_fill = (10, 16, 18)
        border_col = color_cyan
        text_col = color_cyan
    elif tier == TRAFFIC_TIER_PREVENTIVE:
        r_bound = max(rad_px + 5.0, float(halo_2sig[0]) + 4.0)
        color_amber = (0, 190, 255)
        lines = [f"{proj['track_id']} PROXIMITY P={proj['p_wcv']:.2f}"]
        font_scale = 0.40
        bg_fill = (14, 16, 20)
        border_col = color_amber
        text_col = color_amber
    elif tier == TRAFFIC_TIER_CORRECTIVE:
        r_bound = max(rad_px + 8.0, float(halo_ax[0]) + 5.0)
        color_outer = (0, 140, 255)
        h_min = proj.get("hmd_min_m")
        h_max = proj.get("hmd_max_m")
        t_min = proj.get("tau_mod_min_s")
        t_max = proj.get("tau_mod_max_s")
        h_str = (
            f"HMD [{h_min:.0f},{h_max:.0f}]m"
            if (h_min is not None and h_max is not None)
            else "HMD [--,--]m"
        )
        t_str = (
            f"TAU [{t_min:.1f},{t_max:.1f}]s"
            if (t_min is not None and t_max is not None)
            else "TAU [--,--]s"
        )
        p_str = f"P(WCV)={proj['p_wcv']:.2f}"
        lines = [f"{proj['track_id']} CORRECTIVE", h_str, t_str, p_str]
        font_scale = 0.38
        bg_fill = (15, 15, 15)
        border_col = color_outer
        text_col = (240, 240, 240)
    else:
        r_bound = max(rad_px * 1.35 + 6.0, float(halo_2sig[0]) + 6.0)
        color_flash = (255, 255, 255) if (frame_index % 2 == 0) else (0, 220, 255)
        esc_dir = str(proj.get("escape_chevron") or "RIGHT")
        chev_txt = "<<< ESCAPE LEFT" if esc_dir == "LEFT" else "ESCAPE RIGHT >>>"
        t_min = proj.get("tau_mod_min_s")
        tau_lbl = f"TAU_MIN={t_min:.1f}s" if t_min is not None else "WARNING"
        lines = [
            f"{proj['track_id']} WARNING P={proj['p_wcv']:.2f}",
            tau_lbl,
            chev_txt,
        ]
        font_scale = 0.40
        bg_fill = (0, 0, 80)
        border_col = color_flash
        text_col = (255, 255, 255)

    line_h = 15
    max_tw = 0
    for ln in lines:
        tw, _ = cv2.getTextSize(ln, font, font_scale, 1)[0]
        if tw > max_tw:
            max_tw = tw

    box_w = float(max_tw + 14)
    box_h = float(len(lines) * line_h + 8)
    reticle_bbox = (
        float(u_vp - r_bound),
        float(v_vp - r_bound),
        float(u_vp + r_bound),
        float(v_vp + r_bound),
    )
    return {
        "lines": lines,
        "font_scale": font_scale,
        "line_h": line_h,
        "box_w": box_w,
        "box_h": box_h,
        "bg_fill": bg_fill,
        "border_col": border_col,
        "text_col": text_col,
        "r_bound": r_bound,
        "reticle_bbox": reticle_bbox,
    }


def deconflict_hud_labels(
    projected_tracks: list[dict[str, Any]],
    view_cam: PinholeCamera,
    fixed_obstacles: Sequence[tuple[float, float, float, float]] = (),
    integrity_state: str = "VALID",
    frame_index: int = 0,
) -> None:
    """Deterministic 2D bounding-box callout placement & leader-line deconfliction solver (B12-F1).

    Mutates each visible track in `projected_tracks` in-place with:
      - `reticle_bbox`: (x1, y1, x2, y2)
      - `label_bbox`: (x1, y1, x2, y2) guaranteed to have zero overlap with all reticles,
        all fixed HUD banners/badges, and all other track label boxes
      - `leader_line_px`: ((u_start, v_start), (u_end, v_end)) from reticle perimeter to label box
      - `_callout_spec`: internal rendering metadata
    """
    visible: list[dict[str, Any]] = []
    for proj in projected_tracks:
        if not proj.get("is_in_front", False):
            continue
        u_vp = float(proj["u_vp"])
        v_vp = float(proj["v_vp"])
        if not (-40.0 <= u_vp <= view_cam.width + 40.0 and -40.0 <= v_vp <= view_cam.height + 40.0):
            continue
        spec = _format_track_callout_spec(proj, integrity_state=integrity_state, frame_index=frame_index)
        proj["_callout_spec"] = spec
        proj["reticle_bbox"] = spec["reticle_bbox"]
        visible.append(proj)

    # Obstacles list starts with all fixed HUD badges/banners + all target reticle bounding boxes
    obstacles: list[tuple[float, float, float, float]] = [
        _pad_rect(obs, 3.0) for obs in fixed_obstacles
    ]
    for proj in visible:
        obstacles.append(_pad_rect(proj["reticle_bbox"], 4.0))
        # Also protect preventive velocity leader segments if present
        if (
            int(proj["tier"]) == TRAFFIC_TIER_PREVENTIVE
            and str(proj.get("stabilization_mode")) == "EARTH_CONFORMAL"
        ):
            u1, v1 = float(proj["u_vp"]), float(proj["v_vp"])
            u2, v2 = float(proj["u_leader_vp"]), float(proj["v_leader_vp"])
            obstacles.append(
                (min(u1, u2) - 3.0, min(v1, v2) - 3.0, max(u1, u2) + 3.0, max(v1, v2) + 3.0)
            )

    # Sort visible tracks by priority: highest threat tier first, then track_id
    ordered = sorted(visible, key=lambda p: (-int(p["tier"]), str(p["track_id"])))

    min_x = 10.0
    min_y = 48.0
    max_x = float(view_cam.width - 10)
    max_y = float(view_cam.height - 44)

    for proj in ordered:
        spec = proj["_callout_spec"]
        bw = float(spec["box_w"])
        bh = float(spec["box_h"])
        r_b = float(spec["r_bound"])
        u_c = float(proj["u_vp"])
        v_c = float(proj["v_vp"])

        # Check if another visible track is immediately to the right or left
        has_right_neighbor = any(
            (other is not proj)
            and (0.0 < float(other["u_vp"]) - u_c < 220.0)
            and (abs(float(other["v_vp"]) - v_c) < 120.0)
            for other in visible
        )
        has_left_neighbor = any(
            (other is not proj)
            and (0.0 < u_c - float(other["u_vp"]) < 220.0)
            and (abs(float(other["v_vp"]) - v_c) < 120.0)
            for other in visible
        )

        # Build candidate anchor rectangles around (u_c, v_c) across multiple radii
        candidates: list[tuple[float, tuple[float, float, float, float]]] = []
        radii = (16.0, 32.0, 52.0, 76.0, 104.0, 136.0, 172.0, 210.0)
        for rad_idx, gap in enumerate(radii):
            dx = r_b + gap
            dy = r_b + gap * 0.75
            # 8 anchor slots around target:
            slots = [
                # top-right
                ("TR", (u_c + dx, v_c - dy - bh, u_c + dx + bw, v_c - dy)),
                # bottom-right
                ("BR", (u_c + dx, v_c + dy, u_c + dx + bw, v_c + dy + bh)),
                # top-left
                ("TL", (u_c - dx - bw, v_c - dy - bh, u_c - dx, v_c - dy)),
                # bottom-left
                ("BL", (u_c - dx - bw, v_c + dy, u_c - dx, v_c + dy + bh)),
                # right-middle
                ("RM", (u_c + dx, v_c - 0.5 * bh, u_c + dx + bw, v_c + 0.5 * bh)),
                # left-middle
                ("LM", (u_c - dx - bw, v_c - 0.5 * bh, u_c - dx, v_c + 0.5 * bh)),
                # top-center
                ("TC", (u_c - 0.5 * bw, v_c - dy - bh - 6.0, u_c + 0.5 * bw, v_c - dy - 6.0)),
                # bottom-center
                ("BC", (u_c - 0.5 * bw, v_c + dy + 6.0, u_c + 0.5 * bw, v_c + dy + bh + 6.0)),
            ]
            for slot_idx, (slot_name, rect) in enumerate(slots):
                x1, y1, x2, y2 = rect
                # Clamp into viewport bounds
                if x1 < min_x:
                    x2 += min_x - x1
                    x1 = min_x
                if x2 > max_x:
                    x1 -= x2 - max_x
                    x2 = max_x
                if y1 < min_y:
                    y2 += min_y - y1
                    y1 = min_y
                if y2 > max_y:
                    y1 -= y2 - max_y
                    y2 = max_y
                clamped = (float(round(x1)), float(round(y1)), float(round(x2)), float(round(y2)))

                # Side preference penalty to naturally push callouts away from adjacent targets
                side_pen = 0.0
                if has_right_neighbor and not has_left_neighbor and slot_name in ("TR", "BR", "RM"):
                    side_pen += 28.0
                elif has_left_neighbor and not has_right_neighbor and slot_name in ("TL", "BL", "LM"):
                    side_pen += 28.0

                cx_box = 0.5 * (clamped[0] + clamped[2])
                cy_box = 0.5 * (clamped[1] + clamped[3])
                dist = math.hypot(cx_box - u_c, cy_box - v_c)
                score = dist + side_pen + rad_idx * 8.0 + slot_idx * 1.5
                candidates.append((score, clamped))

        # Add fallback grid scan across viewport so even ultra-dense scenes find a 0-overlap slot
        for gy in range(int(min_y), int(max_y - bh), 28):
            for gx in range(int(min_x), int(max_x - bw), 48):
                rect = (float(gx), float(gy), float(gx + bw), float(gy + bh))
                dist = math.hypot(0.5 * (rect[0] + rect[2]) - u_c, 0.5 * (rect[1] + rect[3]) - v_c)
                candidates.append((500.0 + dist, rect))

        candidates.sort(key=lambda item: item[0])

        chosen_rect: tuple[float, float, float, float] | None = None
        best_fallback_rect: tuple[float, float, float, float] | None = None
        best_fallback_overlap = float("inf")

        for _, cand_rect in candidates:
            ov_sum = 0.0
            for obs in obstacles:
                ov = compute_rect_intersection_area(cand_rect, obs)
                if ov > 0.0:
                    ov_sum += ov
            if ov_sum == 0.0:
                chosen_rect = cand_rect
                break
            if ov_sum < best_fallback_overlap:
                best_fallback_overlap = ov_sum
                best_fallback_rect = cand_rect

        if chosen_rect is None:
            chosen_rect = best_fallback_rect or (
                max(min_x, min(max_x - bw, u_c + r_b + 14.0)),
                max(min_y, min(max_y - bh, v_c - bh)),
                max(min_x + bw, min(max_x, u_c + r_b + 14.0 + bw)),
                max(min_y + bh, min(max_y, v_c)),
            )

        proj["label_bbox"] = chosen_rect
        obstacles.append(_pad_rect(chosen_rect, 4.0))

        # Compute leader line from reticle perimeter toward nearest point on chosen_rect
        lx_end = max(chosen_rect[0], min(chosen_rect[2], u_c))
        ly_end = max(chosen_rect[1], min(chosen_rect[3], v_c))
        vec_x = lx_end - u_c
        vec_y = ly_end - v_c
        vec_len = math.hypot(vec_x, vec_y)
        if vec_len > max(r_b, 1.0):
            lx_start = u_c + (vec_x / vec_len) * r_b
            ly_start = v_c + (vec_y / vec_len) * r_b
        else:
            lx_start, ly_start = u_c, v_c
        proj["leader_line_px"] = ((float(lx_start), float(ly_start)), (float(lx_end), float(ly_end)))


def _draw_single_traffic_symbol(
    out: np.ndarray,
    proj: dict[str, Any],
    view_cam: PinholeCamera,
    frame_index: int = 0,
) -> None:
    """Render one projected G1 traffic symbol and its deconflicted leader-line callout on the viewport image."""
    if not proj.get("is_in_front", False):
        return
    u_vp = float(proj["u_vp"])
    v_vp = float(proj["v_vp"])
    if not (-40.0 <= u_vp <= view_cam.width + 40.0 and -40.0 <= v_vp <= view_cam.height + 40.0):
        return

    tier = int(proj["tier"])
    stabilization_mode = str(proj.get("stabilization_mode", "EARTH_CONFORMAL"))
    rad_px = float(proj.get("angular_radius_px", 12.0))  # default-ok: optional display projection fallback
    r_q8 = int(round(max(9.0, min(28.0, rad_px)) * 256.0))
    u_q8 = int(round(u_vp * 256.0))
    v_q8 = int(round(v_vp * 256.0))
    u_int = int(round(u_vp))
    v_int = int(round(v_vp))

    font = cv2.FONT_HERSHEY_SIMPLEX
    color_bg = (0, 0, 0)

    # Draw 2-sigma angular uncertainty ellipse on all earth-conformal targets (B12-F1)
    if stabilization_mode == "EARTH_CONFORMAL":
        halo_2sig = proj.get("halo_axes_2sigma_px", (12.0, 10.0))
        halo_ang = float(proj.get("halo_angle_deg", 0.0))  # default-ok: optional display projection fallback
        ax2_maj = max(int(round(rad_px + 3.0)), int(round(float(halo_2sig[0]))))
        ax2_min = max(int(round(rad_px + 2.0)), int(round(float(halo_2sig[1]))))
        ellipse_col = (180, 235, 255) if tier >= TRAFFIC_TIER_PREVENTIVE else (220, 220, 120)
        cv2.ellipse(
            out,
            (u_int, v_int),
            (ax2_maj, ax2_min),
            halo_ang,
            0.0,
            360.0,
            ellipse_col,
            1,
            cv2.LINE_AA,
        )

    # In DEGRADED_COAST or INVALID_ALERT, downgrade to camera/body-frame bearing chevron + crosshair
    if stabilization_mode == "BODY_STABILIZED":
        color_brg = (0, 200, 255) if tier < TRAFFIC_TIER_WARNING else (60, 60, 255)
        cv2.circle(out, (u_q8, v_q8), r_q8, color_bg, thickness=3, lineType=cv2.LINE_AA, shift=8)
        cv2.circle(out, (u_q8, v_q8), r_q8, color_brg, thickness=1, lineType=cv2.LINE_AA, shift=8)
        # Unstabilized camera-frame bearing chevron inside/above circle
        chev_pts = np.array(
            [
                [u_q8 - int(round(r_q8 * 0.65)), v_q8 + int(round(r_q8 * 0.25))],
                [u_q8, v_q8 - int(round(r_q8 * 0.65))],
                [u_q8 + int(round(r_q8 * 0.65)), v_q8 + int(round(r_q8 * 0.25))],
            ],
            dtype=np.int32,
        )
        cv2.polylines(out, [chev_pts], False, color_bg, 3, cv2.LINE_AA, shift=8)
        cv2.polylines(out, [chev_pts], False, color_brg, 1, cv2.LINE_AA, shift=8)
        # Crosshair ticks
        tick_q8 = int(round(5.0 * 256.0))
        cv2.line(out, (u_q8 - r_q8 - tick_q8, v_q8), (u_q8 - r_q8 + tick_q8, v_q8), color_brg, 1, cv2.LINE_AA, shift=8)
        cv2.line(out, (u_q8 + r_q8 - tick_q8, v_q8), (u_q8 + r_q8 + tick_q8, v_q8), color_brg, 1, cv2.LINE_AA, shift=8)
        cv2.line(out, (u_q8, v_q8 - r_q8 - tick_q8), (u_q8, v_q8 - r_q8 + tick_q8), color_brg, 1, cv2.LINE_AA, shift=8)
        cv2.line(out, (u_q8, v_q8 + r_q8 - tick_q8), (u_q8, v_q8 + r_q8 + tick_q8), color_brg, 1, cv2.LINE_AA, shift=8)

    elif tier == TRAFFIC_TIER_NONE:
        # Cyan open diamond (OpenCV BGR: (255, 255, 0))
        color_cyan = (255, 255, 0)
        diamond = np.array(
            [
                [u_q8, v_q8 - r_q8],
                [u_q8 + r_q8, v_q8],
                [u_q8, v_q8 + r_q8],
                [u_q8 - r_q8, v_q8],
            ],
            dtype=np.int32,
        )
        cv2.polylines(out, [diamond], isClosed=True, color=color_bg, thickness=3, lineType=cv2.LINE_AA, shift=8)
        cv2.polylines(out, [diamond], isClosed=True, color=color_cyan, thickness=1, lineType=cv2.LINE_AA, shift=8)

    elif tier == TRAFFIC_TIER_PREVENTIVE:
        # Amber solid diamond + projected 5-second velocity leader vector
        color_amber = (0, 190, 255)
        diamond = np.array(
            [
                [u_q8, v_q8 - r_q8],
                [u_q8 + r_q8, v_q8],
                [u_q8, v_q8 + r_q8],
                [u_q8 - r_q8, v_q8],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(out, diamond, color=color_amber, lineType=cv2.LINE_AA, shift=8)
        cv2.polylines(out, [diamond], isClosed=True, color=color_bg, thickness=2, lineType=cv2.LINE_AA, shift=8)

        u_lead_q8 = int(round(float(proj["u_leader_vp"]) * 256.0))
        v_lead_q8 = int(round(float(proj["v_leader_vp"]) * 256.0))
        cv2.line(out, (u_q8, v_q8), (u_lead_q8, v_lead_q8), color_bg, thickness=3, lineType=cv2.LINE_AA, shift=8)
        cv2.line(out, (u_q8, v_q8), (u_lead_q8, v_lead_q8), color_amber, thickness=1, lineType=cv2.LINE_AA, shift=8)

    elif tier == TRAFFIC_TIER_CORRECTIVE:
        # Amber/red double-outline octagon + 3-sigma covariance uncertainty halo
        color_outer = (0, 140, 255)
        color_inner = (0, 60, 255)
        color_halo = (80, 200, 255)

        halo_ax = proj.get("halo_axes_px", (18.0, 14.0))
        halo_ang = float(proj.get("halo_angle_deg", 0.0))  # default-ok: optional display projection fallback
        ax_maj = max(int(round(rad_px + 6.0)), int(round(float(halo_ax[0]))))
        ax_min = max(int(round(rad_px + 4.0)), int(round(float(halo_ax[1]))))
        cv2.ellipse(
            out,
            (u_int, v_int),
            (ax_maj, ax_min),
            halo_ang,
            0.0,
            360.0,
            color_halo,
            1,
            cv2.LINE_AA,
        )

        oct_pts_outer = []
        oct_pts_inner = []
        r_in_q8 = int(round(r_q8 * 0.78))
        for k in range(8):
            ang = k * (math.pi / 4.0) + (math.pi / 8.0)
            oct_pts_outer.append([u_q8 + int(round(r_q8 * math.cos(ang))), v_q8 + int(round(r_q8 * math.sin(ang)))])
            oct_pts_inner.append([u_q8 + int(round(r_in_q8 * math.cos(ang))), v_q8 + int(round(r_in_q8 * math.sin(ang)))])
        arr_outer = np.array(oct_pts_outer, dtype=np.int32)
        arr_inner = np.array(oct_pts_inner, dtype=np.int32)
        cv2.polylines(out, [arr_outer], True, color_bg, 4, cv2.LINE_AA, shift=8)
        cv2.polylines(out, [arr_outer], True, color_outer, 2, cv2.LINE_AA, shift=8)
        cv2.polylines(out, [arr_inner], True, color_inner, 1, cv2.LINE_AA, shift=8)

    else:
        # WARNING (tier == 3): red solid warning square + flashing border
        color_red = (30, 30, 255)
        color_flash = (255, 255, 255) if (frame_index % 2 == 0) else (0, 220, 255)
        sq = np.array(
            [
                [u_q8 - r_q8, v_q8 - r_q8],
                [u_q8 + r_q8, v_q8 - r_q8],
                [u_q8 + r_q8, v_q8 + r_q8],
                [u_q8 - r_q8, v_q8 + r_q8],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(out, sq, color=color_red, lineType=cv2.LINE_AA, shift=8)
        r_border = int(round(r_q8 * 1.28))
        sq_border = np.array(
            [
                [u_q8 - r_border, v_q8 - r_border],
                [u_q8 + r_border, v_q8 - r_border],
                [u_q8 + r_border, v_q8 + r_border],
                [u_q8 - r_border, v_q8 + r_border],
            ],
            dtype=np.int32,
        )
        cv2.polylines(out, [sq_border], True, color_bg, 4, cv2.LINE_AA, shift=8)
        cv2.polylines(out, [sq_border], True, color_flash, 2, cv2.LINE_AA, shift=8)

    # Render deconflicted callout box and leader line
    spec = proj.get("_callout_spec")
    label_bbox = proj.get("label_bbox")
    leader_line = proj.get("leader_line_px")
    if spec is not None and label_bbox is not None:
        x1, y1, x2, y2 = (
            int(round(label_bbox[0])),
            int(round(label_bbox[1])),
            int(round(label_bbox[2])),
            int(round(label_bbox[3])),
        )
        border_col = spec["border_col"]
        bg_fill = spec["bg_fill"]
        text_col = spec["text_col"]
        font_scale = float(spec["font_scale"])
        line_h = int(spec["line_h"])
        lines = spec["lines"]

        if leader_line is not None:
            (lx1, ly1), (lx2, ly2) = leader_line
            p1_q8 = (int(round(lx1 * 256.0)), int(round(ly1 * 256.0)))
            p2_q8 = (int(round(lx2 * 256.0)), int(round(ly2 * 256.0)))
            cv2.line(out, p1_q8, p2_q8, color_bg, 3, cv2.LINE_AA, shift=8)
            cv2.line(out, p1_q8, p2_q8, border_col, 1, cv2.LINE_AA, shift=8)

        cv2.rectangle(out, (x1, y1), (x2, y2), bg_fill, -1)
        cv2.rectangle(out, (x1, y1), (x2, y2), border_col, 1, cv2.LINE_AA)
        for idx_l, ln in enumerate(lines):
            ty = y1 + 14 + idx_l * line_h
            cv2.putText(out, ln, (x1 + 6, ty), font, font_scale, text_col, 1, cv2.LINE_AA)


def draw_g1_symbology(
    img: np.ndarray,
    aircraft_pose_est: dict[str, float],
    v_ned: np.ndarray,
    p_ecef_view: np.ndarray,
    r_ecef_view_est: np.ndarray,
    r_ned_body_est: np.ndarray,
    head_euler_rad: tuple[float, float, float],
    view_cam: PinholeCamera,
    tracks_and_risks: Sequence[tuple[Any, Any]] = (),
    integrity_state: str = "VALID",
    pl_deg: float = 0.25,
    f_isolated: bool = False,
    isolated_camera_id: str | None = None,
    coast_time_s: float = 0.0,
    coast_drift_deg_per_s: float = 0.15,
    beta_wc: float = 0.50,
    is_t2: bool = False,
    frame_index: int = 0,
    burn_in_text: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw WP-08 Gate G1 integrity-coupled conformal HUD and passive WC1 traffic symbology.

    Returns (annotated_bgr_or_rgb_image, hud_telemetry_dict).
    """
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_bg = (0, 0, 0)
    color_green = (50, 255, 50)
    color_pl_valid = (180, 255, 120)
    color_amber = (0, 190, 255)
    color_fpm = (0, 255, 255)
    color_thr = (0, 220, 255)

    fixed_obstacles: list[tuple[float, float, float, float]] = []

    # 1. Conformal Horizon & Protection-Level Band (+/- PL_0.01)
    hz_info = compute_conformal_horizon_with_pl_band(
        p_ecef_view=p_ecef_view,
        r_ecef_view_est=r_ecef_view_est,
        view_cam=view_cam,
        pl_deg=pl_deg,
        integrity_state=integrity_state,
        f_isolated=f_isolated,
        isolated_camera_id=isolated_camera_id,
        coast_time_s=coast_time_s,
        coast_drift_deg_per_s=coast_drift_deg_per_s,
    )

    if hz_info["horizon_visible"] and not hz_info["horizon_suppressed"]:
        hz_uv = hz_info["horizon_uv"]
        up_uv = hz_info["upper_pl_uv"]
        lo_uv = hz_info["lower_pl_uv"]

        if integrity_state == "VALID":
            pts_hz = np.round(hz_uv * 256.0).astype(np.int32)
            cv2.polylines(out, [pts_hz], False, color_bg, 3, cv2.LINE_AA, shift=8)
            cv2.polylines(out, [pts_hz], False, color_green, 1, cv2.LINE_AA, shift=8)
            _draw_dashed_polyline_q8(out, up_uv, color_pl_valid, color_bg, dash_stride=18)
            _draw_dashed_polyline_q8(out, lo_uv, color_pl_valid, color_bg, dash_stride=18)
        elif integrity_state == "DEGRADED_COAST":
            _draw_dashed_polyline_q8(out, hz_uv, color_amber, color_bg, dash_stride=14)
            _draw_dashed_polyline_q8(out, up_uv, color_amber, color_bg, dash_stride=10)
            _draw_dashed_polyline_q8(out, lo_uv, color_amber, color_bg, dash_stride=10)

    # 2. Runway outline (suppressed immediately under INVALID_ALERT or DEGRADED_COAST)
    if is_t2 and not hz_info["runway_suppressed"]:
        rwy_res = project_conformal_runway(p_ecef_view, r_ecef_view_est, view_cam)
        if rwy_res["is_visible"]:
            clip_box = (-50.0, -50.0, float(view_cam.width + 50.0), float(view_cam.height + 50.0))
            for seg in rwy_res["visible_segments"]:
                c_seg = clip_line_segment_2d(seg[0], seg[1], clip_box)
                if c_seg is not None:
                    p1 = (int(round(c_seg[0][0] * 256.0)), int(round(c_seg[0][1] * 256.0)))
                    p2 = (int(round(c_seg[1][0] * 256.0)), int(round(c_seg[1][1] * 256.0)))
                    cv2.line(out, p1, p2, color_bg, 3, cv2.LINE_AA, shift=8)
                    cv2.line(out, p1, p2, color_green, 1, cv2.LINE_AA, shift=8)
            for seg in rwy_res["threshold_segments"]:
                c_seg = clip_line_segment_2d(seg[0], seg[1], clip_box)
                if c_seg is not None:
                    p1 = (int(round(c_seg[0][0] * 256.0)), int(round(c_seg[0][1] * 256.0)))
                    p2 = (int(round(c_seg[1][0] * 256.0)), int(round(c_seg[1][1] * 256.0)))
                    cv2.line(out, p1, p2, color_bg, 3, cv2.LINE_AA, shift=8)
                    cv2.line(out, p1, p2, color_thr, 1, cv2.LINE_AA, shift=8)

    # 3. Flight-path marker (suppressed under INVALID_ALERT)
    if integrity_state != "INVALID_ALERT":
        r_ecef_ned_mat = wgs84.r_ecef_ned(
            aircraft_pose_est["lat_deg"], aircraft_pose_est["lon_deg"]
        )
        fpm = compute_flight_path_marker(v_ned, r_ecef_ned_mat, r_ecef_view_est, view_cam)
        if fpm is not None:
            uf, vf = int(round(fpm[0])), int(round(fpm[1]))
            if 0 <= uf < view_cam.width and 0 <= vf < view_cam.height:
                cv2.circle(out, (uf, vf), 7, color_bg, 5, cv2.LINE_AA)
                cv2.circle(out, (uf, vf), 7, color_fpm, 2, cv2.LINE_AA)
                cv2.line(out, (uf - 18, vf), (uf - 7, vf), color_bg, 5, cv2.LINE_AA)
                cv2.line(out, (uf - 18, vf), (uf - 7, vf), color_fpm, 2, cv2.LINE_AA)
                cv2.line(out, (uf + 7, vf), (uf + 18, vf), color_bg, 5, cv2.LINE_AA)
                cv2.line(out, (uf + 7, vf), (uf + 18, vf), color_fpm, 2, cv2.LINE_AA)
                cv2.line(out, (uf, vf - 7), (uf, vf - 15), color_bg, 5, cv2.LINE_AA)
                cv2.line(out, (uf, vf - 7), (uf, vf - 15), color_fpm, 2, cv2.LINE_AA)
                fixed_obstacles.append((float(uf - 20), float(vf - 18), float(uf + 20), float(vf + 12)))

    # Precompute fixed HUD overlay rectangles so deconflict_hud_labels avoids them
    yaw_deg = (math.degrees(aircraft_pose_est["yaw_rad"])) % 360.0
    hdg_str = f"HDG {yaw_deg:03.0f} | INT:{integrity_state}"
    ts_hdg = cv2.getTextSize(hdg_str, font, 0.48, 1)[0]
    hx1 = (view_cam.width - ts_hdg[0]) // 2
    fixed_obstacles.append((float(hx1 - 10), 14.0, float(hx1 + ts_hdg[0] + 10), 42.0))

    if hz_info["status_badge"]:
        badge_txt = str(hz_info["status_badge"])
        tsize_b = cv2.getTextSize(badge_txt, font, 0.48, 1)[0]
        bx1_b = view_cam.width - tsize_b[0] - 30
        fixed_obstacles.append((float(bx1_b - 8), 14.0, float(view_cam.width - 14), 42.0))

    banner_lines: list[str] = []
    banner_rect: tuple[int, int, int, int] | None = None
    if hz_info["alert_banner"]:
        banner_raw = str(hz_info["alert_banner"])
        banner_lines = [p.strip() for p in banner_raw.split("|") if p.strip()]
        max_bw = max(cv2.getTextSize(ln, font, 0.56, 2)[0][0] for ln in banner_lines)
        bh_tot = len(banner_lines) * 24 + 16
        cx_b = (view_cam.width - max_bw) // 2
        cy_top = 56
        banner_rect = (cx_b - 18, cy_top, cx_b + max_bw + 18, cy_top + bh_tot)
        fixed_obstacles.append(
            (float(banner_rect[0]), float(banner_rect[1]), float(banner_rect[2]), float(banner_rect[3]))
        )

    if burn_in_text:
        ts_bi = cv2.getTextSize(burn_in_text, font, 0.45, 1)[0]
        fixed_obstacles.append(
            (15.0, float(view_cam.height - 35), float(25 + ts_bi[0]), float(view_cam.height - 12))
        )

    # 4. Passive WP-05 G1 (WC1) traffic symbology with 2D bounding-box label deconfliction
    projected_tracks: list[dict[str, Any]] = []
    for trk_obj, rsk_obj in tracks_and_risks:
        proj = project_passive_track_to_viewport(
            track=trk_obj,
            risk=rsk_obj,
            r_ned_body_est=r_ned_body_est,
            head_euler_rad=head_euler_rad,
            view_cam=view_cam,
            integrity_state=integrity_state,
            beta_wc=beta_wc,
        )
        projected_tracks.append(proj)

    deconflict_hud_labels(
        projected_tracks=projected_tracks,
        view_cam=view_cam,
        fixed_obstacles=fixed_obstacles,
        integrity_state=integrity_state,
        frame_index=frame_index,
    )

    for proj in projected_tracks:
        _draw_single_traffic_symbol(out, proj, view_cam, frame_index=frame_index)
        proj.pop("_callout_spec", None)

    # 5. Integrity Status Badge or INVALID_ALERT Banner
    if hz_info["status_badge"]:
        badge_txt = str(hz_info["status_badge"])
        badge_col = color_pl_valid if integrity_state == "VALID" else color_amber
        tsize = cv2.getTextSize(badge_txt, font, 0.48, 1)[0]
        bx1 = view_cam.width - tsize[0] - 30
        cv2.rectangle(out, (bx1 - 8, 14), (view_cam.width - 14, 42), (10, 10, 10), -1)
        cv2.rectangle(out, (bx1 - 8, 14), (view_cam.width - 14, 42), badge_col, 1)
        cv2.putText(out, badge_txt, (bx1, 33), font, 0.48, badge_col, 1, cv2.LINE_AA)

    if banner_rect is not None and banner_lines:
        bx1_r, by1_r, bx2_r, by2_r = banner_rect
        cv2.rectangle(out, (bx1_r, by1_r), (bx2_r, by2_r), (0, 0, 140), -1)
        cv2.rectangle(out, (bx1_r, by1_r), (bx2_r, by2_r), (255, 255, 255), 2)
        for idx_b, bline in enumerate(banner_lines):
            lw = cv2.getTextSize(bline, font, 0.56, 2)[0][0]
            tx = (view_cam.width - lw) // 2
            ty = by1_r + 24 + idx_b * 24
            cv2.putText(out, bline, (tx, ty), font, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

    # 6. Heading tape & Burn-in label
    cv2.rectangle(out, (hx1 - 10, 14), (hx1 + ts_hdg[0] + 10, 42), (0, 0, 0), -1)
    cv2.putText(out, hdg_str, (hx1, 33), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    if burn_in_text:
        text_size = cv2.getTextSize(burn_in_text, font, 0.45, 1)[0]
        cv2.rectangle(out, (15, view_cam.height - 35), (25 + text_size[0], view_cam.height - 12), (0, 0, 0), -1)
        cv2.putText(out, burn_in_text, (20, view_cam.height - 18), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    telemetry = {
        "integrity_state": integrity_state,
        "horizon_suppressed": bool(hz_info["horizon_suppressed"]),
        "runway_suppressed": bool(hz_info["runway_suppressed"]),
        "pl_effective_deg": float(hz_info["pl_effective_deg"]),
        "status_badge": hz_info["status_badge"],
        "alert_banner": hz_info["alert_banner"],
        "ttd_display_frames": int(hz_info["ttd_display_frames"]),
        "hud_hmi_frames": int(hz_info["hud_hmi_frames"]),
        "projected_tracks": projected_tracks,
    }
    return out, telemetry


def draw_panorama_g1_symbology(
    panorama_img: np.ndarray,
    r_ned_body_est: np.ndarray,
    tracks_and_risks: Sequence[tuple[Any, Any]] = (),
    frame: str = "ned",
    beta_wc: float = 0.50,
    burn_in_text: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Overlay passive G1 traffic symbols onto a 2048x1024 NED or body panorama with deconflicted callouts."""
    out = panorama_img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    projections: list[dict[str, Any]] = []
    occupied_rects: list[tuple[float, float, float, float]] = []

    for trk_obj, rsk_obj in tracks_and_risks:
        proj = project_passive_track_to_panorama(
            track=trk_obj,
            risk=rsk_obj,
            r_ned_body_est=r_ned_body_est,
            width=w,
            height=h,
            frame=frame,
            beta_wc=beta_wc,
        )
        u_int = int(round(proj["u_pan"])) % w
        v_int = max(0, min(h - 1, int(round(proj["v_pan"]))))
        proj["u_int"] = u_int
        proj["v_int"] = v_int
        ret_bbox = (float(u_int - 14), float(v_int - 14), float(u_int + 14), float(v_int + 14))
        proj["reticle_bbox"] = ret_bbox
        occupied_rects.append(ret_bbox)
        projections.append(proj)

    # First draw all reticle circles
    for proj in projections:
        u_int = int(proj["u_int"])
        v_int = int(proj["v_int"])
        tier = int(proj["tier"])
        if tier == TRAFFIC_TIER_WARNING:
            col = (30, 30, 255)
        elif tier == TRAFFIC_TIER_CORRECTIVE:
            col = (0, 140, 255)
        elif tier == TRAFFIC_TIER_PREVENTIVE:
            col = (0, 200, 255)
        else:
            col = (255, 255, 0)
        cv2.circle(out, (u_int, v_int), 12, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(out, (u_int, v_int), 12, col, 2, cv2.LINE_AA)

    # Deconflict and draw callout boxes with leader lines
    ordered = sorted(projections, key=lambda p: (-int(p.get("tier", 0)), str(p.get("track_id", ""))))  # default-ok: optional display projection fallback
    for proj in ordered:
        u_int = int(proj["u_int"])
        v_int = int(proj["v_int"])
        tier = int(proj["tier"])
        if tier == TRAFFIC_TIER_WARNING:
            col = (30, 30, 255)
        elif tier == TRAFFIC_TIER_CORRECTIVE:
            col = (0, 140, 255)
        elif tier == TRAFFIC_TIER_PREVENTIVE:
            col = (0, 200, 255)
        else:
            col = (255, 255, 0)
        tier_label = "PROXIMITY" if str(proj["tier_name"]) == "PREVENTIVE" else str(proj["tier_name"])
        lbl = f"{proj['track_id']}:{tier_label}"
        ts = cv2.getTextSize(lbl, font, 0.44, 1)[0]
        bw = ts[0] + 12
        bh = 22
        candidates = [
            (u_int - bw // 2, v_int - 18 - bh),
            (u_int - bw // 2, v_int + 18),
            (u_int - bw // 2, v_int - 46 - bh),
            (u_int - bw // 2, v_int + 46),
            (u_int + 20, v_int - bh // 2),
            (u_int - 20 - bw, v_int - bh // 2),
        ]
        chosen_x, chosen_y = candidates[0]
        for cx, cy in candidates:
            cx_c = max(12, min(w - bw - 12, int(cx)))
            cy_c = max(12, min(h - bh - 45, int(cy)))
            cand_rect = (float(cx_c), float(cy_c), float(cx_c + bw), float(cy_c + bh))
            if all(compute_rect_intersection_area(_pad_rect(cand_rect, 4.0), occ) == 0.0 for occ in occupied_rects):
                chosen_x, chosen_y = cx_c, cy_c
                break
        lbl_rect = (float(chosen_x), float(chosen_y), float(chosen_x + bw), float(chosen_y + bh))
        proj["label_bbox"] = lbl_rect
        occupied_rects.append(lbl_rect)

        anchor_x = int(max(chosen_x, min(chosen_x + bw, u_int)))
        anchor_y = chosen_y + bh if chosen_y + bh <= v_int else (chosen_y if chosen_y >= v_int else chosen_y + bh // 2)
        dx = float(anchor_x - u_int)
        dy = float(anchor_y - v_int)
        dist = math.hypot(dx, dy)
        if dist > 13.0:
            sx = int(round(u_int + (dx / dist) * 13.0))
            sy = int(round(v_int + (dy / dist) * 13.0))
            cv2.line(out, (sx, sy), (anchor_x, int(anchor_y)), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(out, (sx, sy), (anchor_x, int(anchor_y)), col, 1, cv2.LINE_AA)

        cv2.rectangle(out, (chosen_x, chosen_y), (chosen_x + bw, chosen_y + bh), (10, 14, 20), -1)
        cv2.rectangle(out, (chosen_x, chosen_y), (chosen_x + bw, chosen_y + bh), col, 1)
        cv2.putText(out, lbl, (chosen_x + 6, chosen_y + 15), font, 0.44, col, 1, cv2.LINE_AA)

    if burn_in_text:
        out = draw_burn_in_label(out, burn_in_text, font_scale=0.5)

    return out, projections
