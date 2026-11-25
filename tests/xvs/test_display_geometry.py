"""Unit tests for WP-08 display geometry acceptance item V1.

Conforms strictly to docs/xvs/WP08_DISPLAY.md v0.1 §5:
- 10,000 seeded rays that hit the ellipsoid: P -> camera pixel -> back-projection returns P within 1e-6 m.
- Parallax angle equals independent numpy formula within 1e-9 rad.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from xvs.geo import wgs84, rotations
from xvs.rig.rig import Rig
from xvs.sim.trajectory import t1_cruise_state
from xvs.display.fusion import (
    intersect_rays_ellipsoid,
    intersect_ray_ellipsoid,
    compute_parallax_angle,
    get_eye_point_body,
    load_display_config,
)


def _setup_v1_test_geometry():
    """Helper to set up 10,000 seeded downward rays and ground points on WGS84 ellipsoid."""
    config = load_display_config()
    rig = Rig.load_nominal().make_true(seed=20260922)

    state = t1_cruise_state(0.0)
    lat = state.lat_deg
    lon = state.lon_deg
    h = state.h_m
    yaw = state.yaw_rad
    pitch = state.pitch_rad
    roll = state.roll_rad

    r_ecef_ned_mat = wgs84.r_ecef_ned(lat, lon)
    r_ned_body_mat = rotations.euler_to_r(yaw, pitch, roll)
    r_ecef_body_mat = r_ecef_ned_mat @ r_ned_body_mat
    p_ecef_body = wgs84.geodetic_to_ecef(lat, lon, h)

    eye_body = get_eye_point_body(config)
    eye_ecef = p_ecef_body + r_ecef_body_mat @ eye_body

    cam_poses = rig.compute_poses(lat, lon, h, yaw, pitch, roll)

    rng = np.random.default_rng(20260925)
    n_target = 10000

    el_rad = rng.uniform(-math.radians(80.0), -math.radians(5.0), size=n_target)
    az_rad = rng.uniform(-math.pi, math.pi, size=n_target)

    cos_el = np.cos(el_rad)
    sin_el = np.sin(el_rad)
    cos_az = np.cos(az_rad)
    sin_az = np.sin(az_rad)

    d_ned = np.stack([cos_el * cos_az, cos_el * sin_az, -sin_el], axis=-1)
    d_ecef = d_ned @ r_ecef_ned_mat.T

    pts_ecef, _, hit_mask = intersect_rays_ellipsoid(eye_ecef, d_ecef)
    assert np.all(hit_mask), "All chosen downward rays must hit the ellipsoid"
    return pts_ecef, eye_ecef, cam_poses, rig


def test_v1_geometry_roundtrip_exactness():
    """V1, part 1: 10,000 seeded rays: P -> camera pixel -> back-projection returns P within 1e-6 m."""
    pts_ecef, eye_ecef, cam_poses, rig = _setup_v1_test_geometry()

    max_roundtrip_err = 0.0
    tested_projections = 0

    w_cam = rig.width
    h_cam = rig.height
    fx = rig.camera_model.fx
    fy = rig.camera_model.fy
    cx = rig.camera_model.cx
    cy = rig.camera_model.cy

    for cam in rig.cameras:
        c_pose = cam_poses[cam.id]
        c_ecef = c_pose["p_ecef_cam"]
        r_ecef_cam = c_pose["r_ecef_cam"]

        p_cam = (pts_ecef - c_ecef) @ r_ecef_cam
        z_cam = p_cam[:, 2]

        front = z_cam > 0.1
        if not np.any(front):
            continue

        p_front = p_cam[front]
        pts_front = pts_ecef[front]

        u = fx * (p_front[:, 0] / p_front[:, 2]) + cx
        v = fy * (p_front[:, 1] / p_front[:, 2]) + cy

        in_fov = (u >= 0.0) & (u <= w_cam - 1.0) & (v >= 0.0) & (v <= h_cam - 1.0)
        if not np.any(in_fov):
            continue

        u_vis = u[in_fov]
        v_vis = v[in_fov]
        pts_vis = pts_front[in_fov]

        rx = (u_vis - cx) / fx
        ry = (v_vis - cy) / fy
        rz = np.ones_like(rx)
        r_cam = np.stack([rx, ry, rz], axis=-1)
        r_cam = r_cam / np.linalg.norm(r_cam, axis=-1, keepdims=True)

        d_cam_ecef = r_cam @ r_ecef_cam.T
        p_back, _, hits_back = intersect_rays_ellipsoid(c_ecef, d_cam_ecef)
        assert np.all(hits_back), "Back-projected camera ray must intersect ellipsoid"

        err = np.linalg.norm(p_back - pts_vis, axis=-1)
        max_roundtrip_err = max(max_roundtrip_err, float(np.max(err)))
        tested_projections += len(err)

    assert tested_projections >= 10000, f"Expected at least 10,000 projections, got {tested_projections}"
    assert max_roundtrip_err < 1e-6, f"Max roundtrip error {max_roundtrip_err:.3e} m exceeds 1e-6 m threshold"


def test_v1_infinity_parallax_independent_formula():
    """V1, part 2: Infinity parallax angle equals independent numpy formula within 1e-9 rad."""
    pts_ecef, eye_ecef, cam_poses, _ = _setup_v1_test_geometry()

    c_ecef_d = cam_poses["D"]["p_ecef_cam"]
    computed_parallax = compute_parallax_angle(pts_ecef, eye_ecef, c_ecef_d)

    v1 = (pts_ecef - c_ecef_d) / np.linalg.norm(pts_ecef - c_ecef_d, axis=-1, keepdims=True)
    v2 = (pts_ecef - eye_ecef) / np.linalg.norm(pts_ecef - eye_ecef, axis=-1, keepdims=True)
    expected_parallax = np.arccos(np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0))

    max_parallax_diff = float(np.max(np.abs(computed_parallax - expected_parallax)))
    assert max_parallax_diff < 1e-9, f"Parallax angle diff {max_parallax_diff:.3e} rad exceeds 1e-9 rad"


def test_v1_display_geometry_exactness():
    """Combined V1 test for backwards compatibility."""
    test_v1_geometry_roundtrip_exactness()
    test_v1_infinity_parallax_independent_formula()
