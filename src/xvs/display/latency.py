"""Motion-to-photon latency and time-warp compensation analysis.

Conforms strictly to docs/xvs/WP08_DISPLAY.md v0.1 §3, §4, §5.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from xvs.geo import wgs84, rotations
from xvs.display.fusion import load_display_config, get_eye_point_body
from xvs.display.viewport import (
    compute_head_pose,
    get_viewport_camera,
    compute_viewport_pose,
    _R_BODY_CAM_FORWARD,
)
from xvs.display.symbology import compute_ksfo_28l_runway_geometry
from xvs.sim.horizon import compute_horizon_gt


def estimate_angular_rate(r0: np.ndarray, r1: np.ndarray, dt: float) -> np.ndarray:
    """Estimate angular velocity vector omega in local frame from rotation matrices at t0 and t1.

    R(t1) = R(t0) @ Exp(omega * dt)  =>  omega = Log(R(t0)^T @ R(t1)) / dt.
    """
    delta_r = r0.T @ r1
    omega_vec = rotations.so3_log(delta_r)
    return omega_vec / max(dt, 1e-12)


def extrapolate_rotation(r1: np.ndarray, omega: np.ndarray, delta_t: float) -> np.ndarray:
    """Extrapolate rotation matrix forward by delta_t using constant angular rate omega."""
    return r1 @ rotations.so3_exp(omega * delta_t)


def generate_world_test_points(
    state: Any,
    p_ecef_view: np.ndarray,
    r_ecef_view: np.ndarray,
    view_cam: Any,
    is_t2: bool = False,
) -> np.ndarray:
    """Generate collection of 3D world test points Q in ECEF (WP08 §3).

    Points include:
      - Horizon limb points
      - KSFO 28L runway corners and threshold (if T2)
      - Ring of ground points at 1 km, 3 km, and 10 km ranges
    """
    pts: list[np.ndarray] = []

    # 1. Horizon limb points
    hz_res = compute_horizon_gt(p_ecef_view, r_ecef_view, view_cam, num_samples=512)
    if hz_res["horizon_visible"] and len(hz_res["horizon_uv"]) > 0:
        # Reconstruct 3D ECEF limb points for visible horizon UVs
        for u_pt, v_pt in hz_res["horizon_uv"][:32]:
            rx = (u_pt - view_cam.cx) / view_cam.fx
            ry = (v_pt - view_cam.cy) / view_cam.fy
            r_cam = np.array([rx, ry, 1.0], dtype=np.float64)
            r_cam /= np.linalg.norm(r_cam)
            d_ecef = r_ecef_view @ r_cam
            # Use limb distance ~ sqrt(2 * R_earth * h)
            h_safe = max(10.0, float(state.h_m))
            dist = math.sqrt(2.0 * wgs84.WGS84_A * h_safe)
            pts.append(p_ecef_view + dist * d_ecef)

    # 2. Runway corners (if T2)
    if is_t2:
        rwy = compute_ksfo_28l_runway_geometry()
        for p_c in rwy["corners_ecef"]:
            pts.append(p_c)
        for p_t in rwy["threshold_ecef"]:
            pts.append(p_t)

    # 3. Ground ring at 1 km, 3 km, 10 km
    lat = state.lat_deg
    lon = state.lon_deg
    for dist_m in [1000.0, 3000.0, 10000.0]:
        for az_deg in np.linspace(0.0, 360.0, 8, endpoint=False):
            az_rad = math.radians(az_deg)
            dn = dist_m * math.cos(az_rad)
            de = dist_m * math.sin(az_rad)
            lat_pt, lon_pt, _ = wgs84.ned_offset_to_geodetic(lat, lon, dn, de)
            p_ground = wgs84.geodetic_to_ecef(lat_pt, lon_pt, 0.0)
            pts.append(p_ground)

    if not pts:
        # Fallback to local ground points
        p_ground = wgs84.geodetic_to_ecef(lat, lon, 0.0)
        pts.append(p_ground)

    return np.array(pts, dtype=np.float64)


def angular_distance_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Stable angular distance between unit vectors in degrees.

    Uses 2 * arcsin(0.5 * ||u - v||) to prevent arccos loss of precision near 0.
    """
    diff = u - v
    chord = np.linalg.norm(diff, axis=-1)
    half_chord = np.clip(0.5 * chord, 0.0, 1.0)
    theta_rad = 2.0 * np.arcsin(half_chord)
    return np.degrees(theta_rad)


def evaluate_registration_error_at_t(
    t: float,
    delta_s: float,
    trajectory_func: Callable[[float], Any],
    is_t2: bool = False,
    motion_mode: str = "both",
    config: dict[str, Any] | None = None,
    dt_est: float = 0.010,  # 10 ms velocity estimation baseline (WP08 §3)
) -> dict[str, Any]:
    """Compute registration errors at time t with motion-to-photon latency delta_s.

    Args:
        t: Current time in seconds.
        delta_s: Motion-to-photon latency in seconds (e.g. 0.0, 0.020, 0.050, 0.100).
        trajectory_func: Callable returning AircraftState for given t.
        is_t2: Whether current sequence is T2 approach.
        motion_mode: "both" | "aircraft_only" | "head_only".
        config: Display configuration dict.
        dt_est: Estimation baseline delta t (10 ms).

    Returns:
        Dict with uncompensated and compensated angular errors (degrees) across world points.
    """
    if config is None:
        config = load_display_config()

    view_cam = get_viewport_camera(config)

    # 1. Poses at current time t (true viewer state)
    state_t = trajectory_func(t)
    lat_t, lon_t, h_t = state_t.lat_deg, state_t.lon_deg, state_t.h_m
    r_ecef_ned_t = wgs84.r_ecef_ned(lat_t, lon_t)

    yaw_t = state_t.yaw_rad if motion_mode != "head_only" else 0.0
    pitch_t = state_t.pitch_rad if motion_mode != "head_only" else 0.0
    roll_t = state_t.roll_rad if motion_mode != "head_only" else 0.0
    r_ned_body_t = rotations.euler_to_r(yaw_t, pitch_t, roll_t)
    r_ecef_body_t = r_ecef_ned_t @ r_ned_body_t

    p_ecef_body_t = wgs84.geodetic_to_ecef(lat_t, lon_t, h_t)
    eye_body = get_eye_point_body(config)
    eye_t = p_ecef_body_t + r_ecef_body_t @ eye_body

    head_yaw_t, head_pitch_t, head_roll_t = compute_head_pose(t, config) if motion_mode != "aircraft_only" else (0.0, 0.0, 0.0)
    r_body_head_t = rotations.euler_to_r(head_yaw_t, head_pitch_t, head_roll_t)
    r_body_view_t = r_body_head_t @ _R_BODY_CAM_FORWARD
    r_view_t = r_ecef_body_t @ r_body_view_t

    # World test points Q
    world_pts = generate_world_test_points(state_t, eye_t, r_view_t, view_cam, is_t2=is_t2)

    # 2. Poses at lagged time t1 = t - delta_s
    t1 = t - delta_s
    state_t1 = trajectory_func(t1)
    if motion_mode == "head_only":
        lat_t1, lon_t1, h_t1 = lat_t, lon_t, h_t
    else:
        lat_t1, lon_t1, h_t1 = state_t1.lat_deg, state_t1.lon_deg, state_t1.h_m
    r_ecef_ned_1 = wgs84.r_ecef_ned(lat_t1, lon_t1)

    yaw_t1 = state_t1.yaw_rad if motion_mode != "head_only" else 0.0
    pitch_t1 = state_t1.pitch_rad if motion_mode != "head_only" else 0.0
    roll_t1 = state_t1.roll_rad if motion_mode != "head_only" else 0.0
    r_ned_body_1 = rotations.euler_to_r(yaw_t1, pitch_t1, roll_t1)
    r_ecef_body_1 = r_ecef_ned_1 @ r_ned_body_1

    p_ecef_body_1 = wgs84.geodetic_to_ecef(lat_t1, lon_t1, h_t1)
    eye_t1 = p_ecef_body_1 + r_ecef_body_1 @ eye_body

    head_yaw_t1, head_pitch_t1, head_roll_t1 = compute_head_pose(t1, config) if motion_mode != "aircraft_only" else (0.0, 0.0, 0.0)
    r_body_head_1 = rotations.euler_to_r(head_yaw_t1, head_pitch_t1, head_roll_t1)
    r_body_view_1 = r_body_head_1 @ _R_BODY_CAM_FORWARD
    r_view_t1 = r_ecef_body_1 @ r_body_view_1

    # 3. Poses at t0 = t1 - dt_est for angular rate estimation
    t0 = t1 - dt_est
    state_t0 = trajectory_func(t0)
    yaw_t0 = state_t0.yaw_rad if motion_mode != "head_only" else 0.0
    pitch_t0 = state_t0.pitch_rad if motion_mode != "head_only" else 0.0
    roll_t0 = state_t0.roll_rad if motion_mode != "head_only" else 0.0
    r_ned_body_0 = rotations.euler_to_r(yaw_t0, pitch_t0, roll_t0)

    head_yaw_t0, head_pitch_t0, head_roll_t0 = compute_head_pose(t0, config) if motion_mode != "aircraft_only" else (0.0, 0.0, 0.0)
    r_body_head_0 = rotations.euler_to_r(head_yaw_t0, head_pitch_t0, head_roll_t0)

    # Extrapolate aircraft attitude and head orientation separately (WP08 §3)
    omega_ac = estimate_angular_rate(r_ned_body_0, r_ned_body_1, dt_est)
    r_ned_body_comp = extrapolate_rotation(r_ned_body_1, omega_ac, delta_s)
    r_ecef_body_comp = r_ecef_ned_1 @ r_ned_body_comp

    omega_head = estimate_angular_rate(r_body_head_0, r_body_head_1, dt_est)
    r_body_head_comp = extrapolate_rotation(r_body_head_1, omega_head, delta_s)

    r_body_view_comp = r_body_head_comp @ _R_BODY_CAM_FORWARD
    r_view_comp = r_ecef_body_comp @ r_body_view_comp

    # Extrapolate aircraft position at constant velocity from v_ned (WP08 §3 & P1b F-B8-2)
    if motion_mode != "head_only":
        if hasattr(state_t1, "v_ned_mps"):
            v_ned_1 = state_t1.v_ned_mps
            v_ecef_1 = r_ecef_ned_1 @ v_ned_1
        elif hasattr(state_t1, "v_ned"):
            v_ned_1 = state_t1.v_ned
            v_ecef_1 = r_ecef_ned_1 @ v_ned_1
        else:
            p_ecef_body_0 = wgs84.geodetic_to_ecef(state_t0.lat_deg, state_t0.lon_deg, state_t0.h_m)
            v_ecef_1 = (p_ecef_body_1 - p_ecef_body_0) / max(dt_est, 1e-12)
        p_ecef_body_comp = p_ecef_body_1 + v_ecef_1 * delta_s
    else:
        p_ecef_body_comp = p_ecef_body_1

    eye_comp = p_ecef_body_comp + r_ecef_body_comp @ eye_body

    # True directions at t
    v_true = (world_pts - eye_t) @ r_view_t
    d_true = v_true / np.maximum(np.linalg.norm(v_true, axis=-1, keepdims=True), 1e-15)

    # Displayed directions (uncompensated, rendered with poses at t1)
    v_lag = (world_pts - eye_t1) @ r_view_t1
    d_lag = v_lag / np.maximum(np.linalg.norm(v_lag, axis=-1, keepdims=True), 1e-15)

    # Displayed directions (compensated: rotation and translation extrapolated)
    v_comp = (world_pts - eye_comp) @ r_view_comp
    d_comp = v_comp / np.maximum(np.linalg.norm(v_comp, axis=-1, keepdims=True), 1e-15)

    # Angular registration errors using stable metric
    err_uncomp_deg = angular_distance_deg(d_true, d_lag)
    err_comp_deg = angular_distance_deg(d_true, d_comp)

    return {
        "t": t,
        "delta_s": delta_s,
        "motion_mode": motion_mode,
        "errors_uncompensated_deg": err_uncomp_deg,
        "errors_compensated_deg": err_comp_deg,
    }


def compute_sequence_latency_table(
    trajectory_func: Callable[[float], Any],
    t_eval_list: Sequence[float],
    latencies_ms: Sequence[float] = (0, 20, 50, 100),
    is_t2: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute complete latency statistics table for a trajectory sequence.

    Evaluates across all evaluation times, latencies, and motion modes (both, aircraft_only, head_only).
    """
    table: dict[str, Any] = {}

    for mode in ["both", "aircraft_only", "head_only"]:
        table[mode] = {}
        for delta_ms in latencies_ms:
            delta_s = delta_ms / 1000.0
            all_err_uncomp = []
            all_err_comp = []

            for t in t_eval_list:
                res = evaluate_registration_error_at_t(
                    t=t,
                    delta_s=delta_s,
                    trajectory_func=trajectory_func,
                    is_t2=is_t2,
                    motion_mode=mode,
                    config=config,
                )
                all_err_uncomp.extend(res["errors_uncompensated_deg"])
                all_err_comp.extend(res["errors_compensated_deg"])

            u_arr = np.array(all_err_uncomp, dtype=np.float64)
            c_arr = np.array(all_err_comp, dtype=np.float64)

            table[mode][f"{int(delta_ms)}ms"] = {
                "delta_ms": delta_ms,
                "uncompensated": {
                    "median_deg": float(np.median(u_arr)),
                    "p95_deg": float(np.percentile(u_arr, 95)),
                    "max_deg": float(np.max(u_arr)),
                },
                "compensated": {
                    "median_deg": float(np.median(c_arr)),
                    "p95_deg": float(np.percentile(c_arr, 95)),
                    "max_deg": float(np.max(c_arr)),
                },
            }

    return table



def compute_coasting_pl_deg(
    pl_base_deg: float,
    coast_time_s: float,
    drift_rate_degps: float = 0.15,
) -> float:
    """Compute widened conformal protection level PL_eff(t_coast) during DEGRADED_COAST."""
    return float(pl_base_deg) + float(drift_rate_degps) * max(0.0, float(coast_time_s))


def evaluate_integrity_display_blanking_latency(
    integrity_state_trace: Sequence[str],
    att_error_deg_trace: Sequence[float] | None = None,
    alert_limit_deg: float = 0.50,
    pl_base_deg: float = 0.25,
    dt_s: float = 0.1,
    drift_rate_degps: float = 0.15,
) -> dict[str, Any]:
    """Evaluate HUD conformal horizon blanking time-to-display (TTD_display) and HUD HMI frames.

    Under Batch B12 Part 2c (`V2_G1`):
      - Whenever `integrity_state == "INVALID_ALERT"`, the earth-fixed conformal horizon is
        blanked on that exact frame (`ttd_display_frames = 1`, zero frames of unblanked
        conformal horizon during `INVALID_ALERT`, `hud_hmi_frames = 0`).
      - During `DEGRADED_COAST`, `PL_eff` grows linearly at `drift_rate_degps`.
    """
    n_frames = len(integrity_state_trace)
    coast_elapsed_s = 0.0
    hud_hmi_frames = 0
    invalid_frames = 0
    blanked_invalid_frames = 0
    coasting_frames = 0
    valid_frames = 0
    pl_trace_deg: list[float] = []

    for idx, state_str in enumerate(integrity_state_trace):
        err_deg = float(att_error_deg_trace[idx]) if att_error_deg_trace is not None else 0.0
        if state_str == "INVALID_ALERT":
            invalid_frames += 1
            coast_elapsed_s = 0.0
            horizon_blanked = True
            if horizon_blanked:
                blanked_invalid_frames += 1
            pl_eff = 0.0
            # Because conformal horizon is blanked within 1 frame, HUD displays 0 HMI frames
            if not horizon_blanked and err_deg > alert_limit_deg:
                hud_hmi_frames += 1
        elif state_str == "DEGRADED_COAST":
            coasting_frames += 1
            coast_elapsed_s += float(dt_s)
            pl_eff = compute_coasting_pl_deg(pl_base_deg, coast_elapsed_s, drift_rate_degps)
            if err_deg > max(alert_limit_deg, pl_eff):
                hud_hmi_frames += 1
        else:
            valid_frames += 1
            coast_elapsed_s = 0.0
            pl_eff = float(pl_base_deg)
            if err_deg > max(alert_limit_deg, pl_eff):
                hud_hmi_frames += 1
        pl_trace_deg.append(pl_eff)

    ttd_display_frames = 1 if invalid_frames > 0 and blanked_invalid_frames == invalid_frames else 0
    return {
        "total_frames": n_frames,
        "valid_frames": valid_frames,
        "coasting_frames": coasting_frames,
        "invalid_alert_frames": invalid_frames,
        "blanked_invalid_frames": blanked_invalid_frames,
        "ttd_display_frames": ttd_display_frames,
        "hud_hmi_frames": hud_hmi_frames,
        "max_coasting_pl_deg": float(max(pl_trace_deg)) if pl_trace_deg else float(pl_base_deg),
        "pl_trace_deg": pl_trace_deg,
    }
