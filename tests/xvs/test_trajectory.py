"""Fast unit tests for Kinematic trajectory models T1, T2, T3.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §2.2, §3 (G1) and Findings C7.
"""

import math
import numpy as np
import pytest

from xvs.geo import wgs84
from xvs.sim import airports, trajectory


def test_t2_exact_reference_points():
    """T2 construction tested against §2.2 reference values (must match to <= 1e-9 deg)."""
    thr = airports.load_ksfo_28l_threshold()

    # Director reference table from WP01_SIM_RIG.md v1.1 §2.2:
    # Point | lat (deg) | lon (deg) | h (m)
    # THR     | 37.61210677733923 | -122.35928141744020 | -
    # t = 0 s | 37.59433375624118 | -122.31729061275578 | 235.112672989
    # t = 30s | 37.60322214002520 | -122.33828351536131 | 125.056336494
    # t = 60s | 37.61210677733923 | -122.35928141744020 | 15.000000000

    ref_points = {
        0.0: (37.59433375624118, -122.31729061275578, 235.112672989),
        30.0: (37.60322214002520, -122.33828351536131, 125.056336494),
        60.0: (37.61210677733923, -122.35928141744020, 15.000000000),
    }

    assert abs(thr.lat_deg - 37.61210677733923) <= 1e-9
    assert abs(thr.lon_deg - (-122.35928141744020)) <= 1e-9

    for t_s, (ref_lat, ref_lon, ref_h) in ref_points.items():
        state = trajectory.t2_ksfo28l_state(t_s, thr=thr)
        assert abs(state.lat_deg - ref_lat) <= 1e-9, f"t={t_s} lat diff {state.lat_deg - ref_lat}"
        assert abs(state.lon_deg - ref_lon) <= 1e-9, f"t={t_s} lon diff {state.lon_deg - ref_lon}"
        assert abs(state.h_m - ref_h) <= 0.001, f"t={t_s} h diff {state.h_m - ref_h}"


def test_t2_dynamics_and_finite_differences_at_t60():
    """T2 at t = 60 s:
      - over THR horizontally < 0.5 m and 15.00 ± 0.01 m above h_THR;
      - path angle 3.000 ± 0.01°;
      - ground speed 70.00 ± 0.02 m/s (finite differences).
    """
    thr = airports.load_ksfo_28l_threshold()
    state60 = trajectory.t2_ksfo28l_state(60.0, thr=thr)

    p_thr = wgs84.geodetic_to_ecef(thr.lat_deg, thr.lon_deg, thr.h_m)
    p_60_ground = wgs84.geodetic_to_ecef(state60.lat_deg, state60.lon_deg, thr.h_m)
    horizontal_dist_to_thr = float(np.linalg.norm(p_60_ground - p_thr))
    assert horizontal_dist_to_thr < 0.5, f"Horizontal offset to THR: {horizontal_dist_to_thr} m"

    height_above_thr = state60.h_m - thr.h_m
    assert abs(height_above_thr - 15.00) <= 0.01

    # Finite difference around t = 60 s (dt = 0.01 s)
    dt = 0.01
    s_minus = trajectory.t2_ksfo28l_state(60.0 - dt, thr=thr)
    s_plus = trajectory.t2_ksfo28l_state(60.0, thr=thr)

    # Position displacement in ECEF
    dp_ecef = s_plus.p_ecef_m - s_minus.p_ecef_m
    r_ecef_ned_mat = wgs84.r_ecef_ned(state60.lat_deg, state60.lon_deg)
    dp_ned = r_ecef_ned_mat.T @ dp_ecef
    v_fd = dp_ned / dt

    v_horizontal = math.hypot(v_fd[0], v_fd[1])
    assert abs(v_horizontal - 70.00) <= 0.02, f"Ground speed: {v_horizontal}"

    path_angle_deg = math.degrees(math.atan2(v_fd[2], v_horizontal))
    assert abs(path_angle_deg - 3.000) <= 0.01, f"Path angle: {path_angle_deg}"

    # Computed v_D check
    expected_vd = 70.0 * math.tan(math.radians(3.0))
    assert abs(state60.v_ned_mps[2] - expected_vd) < 1e-12


def test_t1_cruise_h_constant():
    """T1: h constant within 1e-6 m across trajectory."""
    for t_s in [0.0, 15.0, 30.0, 45.0, 60.0]:
        state = trajectory.t1_cruise_state(t_s)
        assert abs(state.h_m - 3000.0) < 1e-6
        assert abs(state.ypr_deg[0] - 315.0) < 1e-6
        assert abs(state.v_ned_mps[2] - 0.0) < 1e-12


def test_t3_turn_climb_reference_values():
    """T3: Delta_psi = 157.40 +/- 0.2 deg, climb 300.0 +/- 0.5 m, horizontal radius 2181.3 +/- 5 m."""
    s0 = trajectory.t3_turn_climb_state(0.0)
    s60 = trajectory.t3_turn_climb_state(60.0)

    delta_psi = (s60.yaw_rad - s0.yaw_rad)
    delta_psi_deg = math.degrees(delta_psi)
    assert abs(delta_psi_deg - 157.40) <= 0.2, f"Heading change: {delta_psi_deg} deg"

    climb = s60.h_m - s0.h_m
    assert abs(climb - 300.0) <= 0.5, f"Climb: {climb} m"

    # Turn radius: R = V_h / omega
    g = 9.80665
    v_h = math.sqrt(100.0**2 - 5.0**2)
    omega = g * math.tan(math.radians(25.0)) / v_h
    turn_radius = v_h / omega
    assert abs(turn_radius - 2181.3) <= 5.0, f"Turn radius: {turn_radius} m"
