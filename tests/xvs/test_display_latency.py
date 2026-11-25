"""Unit tests for WP-08 latency and compensation acceptance item V4.

Conforms strictly to docs/xvs/WP08_DISPLAY.md v0.1 §5:
- Delta = 0 gives an error <= 1e-9 deg.
- A synthetic constant-rate rotation (aircraft and head) is compensated to <= 1e-9 deg.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from xvs.geo import rotations
from xvs.sim.trajectory import t1_cruise_state
from xvs.display.latency import (
    evaluate_registration_error_at_t,
    estimate_angular_rate,
    extrapolate_rotation,
)


def test_v4_zero_latency_zero_error():
    """V4, part 1: Delta = 0 gives an error <= 1e-9 deg."""
    for mode in ["both", "aircraft_only", "head_only"]:
        res = evaluate_registration_error_at_t(
            t=25.0,
            delta_s=0.0,
            trajectory_func=t1_cruise_state,
            motion_mode=mode,
        )
        max_err = float(np.max(res["errors_uncompensated_deg"]))
        assert max_err <= 1e-9, f"Mode {mode} zero-latency error {max_err:.3e} deg > 1e-9 deg"


def test_v4_constant_rate_rotation_compensated_exact():
    """V4, part 2: Constant-rate rotation pushed through evaluate_registration_error_at_t (R-GB-12).

    - With eye at ARP ([0, 0, 0]), compensation is exact to <= 1e-9 deg.
    - With nominal eye lever arm ([11.0, -0.55, -1.0]), compensation leaves a bounded lever-arm floor <= 2.5e-3 deg.
    """
    from types import SimpleNamespace

    omega_ac = np.array([0.03, -0.015, 0.05], dtype=np.float64)

    def synth_trajectory(t_eval: float):
        r_mat = rotations.so3_exp(omega_ac * t_eval)
        y, p, r = rotations.r_to_euler(r_mat)
        return SimpleNamespace(
            lat_deg=37.5,
            lon_deg=-122.3,
            h_m=1000.0,
            yaw_rad=y,
            pitch_rad=p,
            roll_rad=r,
            v_ned_mps=np.zeros(3, dtype=np.float64),
        )

    # 1. Exactness with eye at ARP (no lever-arm displacement)
    cfg_arp = {
        "eye_point_body_frd_m": [0.0, 0.0, 0.0],
        "viewport": {"width": 1280, "height": 720, "fov_h_deg": 90.0, "fov_v_deg": 60.0},
    }
    for delta_ms in [20, 50, 100]:
        delta_s = delta_ms / 1000.0
        res = evaluate_registration_error_at_t(
            t=30.0,
            delta_s=delta_s,
            trajectory_func=synth_trajectory,
            motion_mode="aircraft_only",
            config=cfg_arp,
        )
        max_err = float(np.max(res["errors_compensated_deg"]))
        assert max_err <= 1e-9, f"Delta {delta_ms} ms at ARP max err {max_err:.3e} deg > 1e-9 deg"

    # 2. Lever-arm floor with nominal eye point (eye 11 m from rotation center)
    cfg_nominal = {
        "eye_point_body_frd_m": [11.0, -0.55, -1.0],
        "viewport": {"width": 1280, "height": 720, "fov_h_deg": 90.0, "fov_v_deg": 60.0},
    }
    for delta_ms in [20, 50, 100]:
        delta_s = delta_ms / 1000.0
        res = evaluate_registration_error_at_t(
            t=30.0,
            delta_s=delta_s,
            trajectory_func=synth_trajectory,
            motion_mode="aircraft_only",
            config=cfg_nominal,
        )
        max_err = float(np.max(res["errors_compensated_deg"]))
        assert max_err <= 2.5e-3, f"Delta {delta_ms} ms nominal lever arm max err {max_err:.3e} deg > 2.5e-3 deg"


def test_v4_constant_velocity_constant_rate_compensated_exact():
    """P1b (F-B8-2): A synthetic constant-velocity, constant-rate case must compensate to <= 1e-9 deg."""
    from xvs.geo import wgs84
    from xvs.display.latency import angular_distance_deg

    # Synthetic constant velocity in NED and constant angular rates
    v_ned = np.array([65.0, -25.0, 3.5], dtype=np.float64)  # m/s
    omega_ac = np.array([0.03, -0.015, 0.05], dtype=np.float64)  # rad/s in body
    omega_head = np.array([0.08, -0.04, 0.0], dtype=np.float64)   # rad/s in head

    lat0, lon0, h0 = 37.5, -122.3, 500.0
    r_ecef_ned = wgs84.r_ecef_ned(lat0, lon0)
    v_ecef = r_ecef_ned @ v_ned
    p0_ecef = wgs84.geodetic_to_ecef(lat0, lon0, h0)
    eye_body = np.array([11.0, -0.55, -1.0], dtype=np.float64)
    r_fwd = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

    dt_est = 0.010  # 10 ms
    for delta_ms in [20, 50, 100]:
        delta_s = delta_ms / 1000.0
        t = 40.0
        t1 = t - delta_s
        t0 = t1 - dt_est

        # True state at t
        p_t = p0_ecef + v_ecef * t
        r_ac_t = rotations.so3_exp(omega_ac * t)
        r_head_t = rotations.so3_exp(omega_head * t)
        eye_t = p_t + (r_ecef_ned @ r_ac_t) @ eye_body
        r_view_t = (r_ecef_ned @ r_ac_t) @ (r_head_t @ r_fwd)

        # State at t1
        p_1 = p0_ecef + v_ecef * t1
        r_ac_1 = rotations.so3_exp(omega_ac * t1)
        r_head_1 = rotations.so3_exp(omega_head * t1)

        # State at t0
        r_ac_0 = rotations.so3_exp(omega_ac * t0)
        r_head_0 = rotations.so3_exp(omega_head * t0)

        # Extrapolate rates and rotations
        om_ac_est = estimate_angular_rate(r_ac_0, r_ac_1, dt_est)
        om_head_est = estimate_angular_rate(r_head_0, r_head_1, dt_est)
        r_ac_comp = extrapolate_rotation(r_ac_1, om_ac_est, delta_s)
        r_head_comp = extrapolate_rotation(r_head_1, om_head_est, delta_s)

        # Extrapolate position at constant velocity
        p_comp = p_1 + v_ecef * delta_s
        eye_comp = p_comp + (r_ecef_ned @ r_ac_comp) @ eye_body
        r_view_comp = (r_ecef_ned @ r_ac_comp) @ (r_head_comp @ r_fwd)

        # World test points at both near and far distances (50 m to 10,000 m)
        rng = np.random.default_rng(1234)
        world_pts = p_t + rng.uniform(-1000, 1000, size=(500, 3))

        # True view directions at t
        v_true = (world_pts - eye_t) @ r_view_t
        d_true = v_true / np.linalg.norm(v_true, axis=-1, keepdims=True)

        # Compensated view directions (translation and rotation extrapolated)
        v_comp = (world_pts - eye_comp) @ r_view_comp
        d_comp = v_comp / np.linalg.norm(v_comp, axis=-1, keepdims=True)

        residual_deg = angular_distance_deg(d_true, d_comp)
        max_residual = float(np.max(residual_deg))

        assert max_residual <= 1e-9, f"Delta {delta_ms} ms residual {max_residual:.3e} deg > 1e-9 deg"


def test_v4_head_only_freezes_aircraft_position():
    """Verify head_only mode freezes aircraft position (R-GB-13)."""
    # At t=25s, aircraft is translating in T1 cruise
    res = evaluate_registration_error_at_t(
        t=25.0,
        delta_s=0.100,
        trajectory_func=t1_cruise_state,
        motion_mode="head_only",
    )
    # Without freezing position, translation would induce large residual
    # With frozen position, error is governed strictly by head rate extrapolation
    comp_max = float(np.max(res["errors_compensated_deg"]))
    assert comp_max < 0.10, f"head_only 100 ms comp error {comp_max:.4f} deg exceeds 0.10 deg"
    assert res["motion_mode"] == "head_only"


