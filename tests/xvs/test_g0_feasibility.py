"""Unit and regression tests for WP-01 Gate G0 feasibility spike components.

Conforms to docs/xvs/WP01_SIM_RIG.md §3 (G0) and docs/xvs/CONVENTIONS.md.
"""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest

from xvs.geo import wgs84, rotations
from xvs.rig.camera import PinholeCamera
from xvs.rig.rig import Rig
from xvs.sim import airports, trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_wgs84_basic_and_ned():
    """Verify equator and north pole ECEF coordinates and local NED basis at (0, 0)."""
    # (0, 0, 0) -> (a, 0, 0)
    p_eq = wgs84.geodetic_to_ecef(0.0, 0.0, 0.0)
    assert np.allclose(p_eq, [wgs84.WGS84_A, 0.0, 0.0], atol=1e-6)

    # (90, 0, 0) -> (0, 0, b)
    p_pole = wgs84.geodetic_to_ecef(90.0, 0.0, 0.0)
    assert np.allclose(p_pole, [0.0, 0.0, wgs84.WGS84_B], atol=1e-6)

    # Round trip
    lat, lon, h = wgs84.ecef_to_geodetic(p_eq[0], p_eq[1], p_eq[2])
    assert abs(lat - 0.0) < 1e-9
    assert abs(lon - 0.0) < 1e-9
    assert abs(h - 0.0) < 1e-6

    # NED basis at (0, 0): n=(0,0,1), e=(0,1,0), d=(-1,0,0) (CONVENTIONS §3 line 248)
    n, e, d = wgs84.ned_basis(0.0, 0.0)
    assert np.allclose(n, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(e, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(d, [-1.0, 0.0, 0.0], atol=1e-12)

    r_ned = wgs84.r_ecef_ned(0.0, 0.0)
    assert abs(np.linalg.det(r_ned) - 1.0) < 1e-12


def test_rotations_mandatory_signs():
    """Verify CONVENTIONS §4 mandatory sign tests."""
    # psi = 90 deg -> body x points East
    r_yaw90 = rotations.euler_to_r(math.radians(90.0), 0.0, 0.0)
    body_x_in_ned = r_yaw90 @ np.array([1.0, 0.0, 0.0])
    # East is NED y = [0, 1, 0]
    assert np.allclose(body_x_in_ned, [0.0, 1.0, 0.0], atol=1e-12)

    # theta > 0 -> nose up (body x has negative Down component)
    r_pitch_up = rotations.euler_to_r(0.0, math.radians(10.0), 0.0)
    body_x_pitch = r_pitch_up @ np.array([1.0, 0.0, 0.0])
    assert body_x_pitch[2] < 0.0  # Down is positive z, so nose up is negative z

    # phi > 0 -> right wing down (body y has positive Down component)
    r_roll_right = rotations.euler_to_r(0.0, 0.0, math.radians(10.0))
    body_y_roll = r_roll_right @ np.array([0.0, 1.0, 0.0])
    assert body_y_roll[2] > 0.0  # Right wing down is positive z


def test_pinhole_camera_intrinsics():
    """Verify CONVENTIONS §5 pinhole reference values."""
    cam = PinholeCamera(width=1024, height=1024, fov_h_deg=100.0)
    assert abs(cam.fx - 429.6190111627674) < 1e-9
    assert abs(cam.fy - 429.6190111627674) < 1e-9
    assert cam.cx == 511.5
    assert cam.cy == 511.5

    # Principal ray maps to (cx, cy)
    u, v, front = cam.project_cam(np.array([0.0, 0.0, 10.0]))
    assert front is True
    assert abs(u - 511.5) < 1e-9
    assert abs(v - 511.5) < 1e-9

    # Project/unproject roundtrip
    ray = cam.unproject(511.5, 511.5)
    assert np.allclose(ray, [0.0, 0.0, 1.0], atol=1e-9)


def test_rig_nominal_loading():
    """Verify nominal rig loads with 6 cameras and matching boresights."""
    rig = Rig.load_nominal()
    assert len(rig.cameras) == 6
    assert tuple(c.id for c in rig.cameras) == ("F", "R", "B", "L", "U", "D")

    # Check determinant of R_body_cam
    for c in rig.cameras:
        assert abs(np.linalg.det(c.r_body_cam) - 1.0) < 1e-12
        # Check boresight
        calculated_boresight = c.r_body_cam @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(calculated_boresight, c.boresight_body, atol=1e-12)


def test_ksfo_28l_threshold_and_t2():
    """Verify KSFO 28L threshold computation and T2 trajectory at t=30s."""
    thr = airports.load_ksfo_28l_threshold()
    assert thr.airport_ident == "KSFO"
    assert thr.runway_ident == "28L"
    assert thr.heading_deg == 298.0
    assert thr.h_m == 0.0
    assert abs(thr.displaced_threshold_m - 91.44) < 1e-6

    # Evaluate T2 at t=30s
    state = trajectory.t2_ksfo28l_state(30.0, thr=thr)
    # Expected h(30) = 125.056 m (WP01_SIM_RIG line 93 reference)
    assert abs(state.h_m - 125.0563) < 0.01
    assert state.ypr_deg[0] == 298.0
    assert state.ypr_deg[1] == 2.5
    assert abs(state.ypr_deg[2]) < 1e-12  # roll at t=30 is sin(5*pi) = 0


@pytest.mark.render
def test_cesium_binary_exact_hash():
    """Verify Cesium.js size and SHA-256 against Director reference."""
    cesium_js = REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium" / "Cesium.js"
    assert cesium_js.is_file()
    assert cesium_js.stat().st_size == 6018837
    h = hashlib.sha256(cesium_js.read_bytes()).hexdigest()
    assert h == "dbb7a1606ef2150c7266eee6eb10bfeba1bd1351cdce1df85787a45491f482e3"


from xvs.paths import get_data_root


@pytest.mark.data
def test_g0_artifacts_verified(check_data):
    """Verify all G0 output JSON artifacts exist, report PASS, and meet G0 thresholds."""
    data_root = get_data_root()
    g0_dir = check_data(data_root / "xvs" / "wp01" / "g0", "G0 output artifacts directory")
    assert (g0_dir / "inventory.json").is_file()
    assert (g0_dir / "install_record.json").is_file()
    assert (g0_dir / "cesium_record.json").is_file()
    assert (g0_dir / "webgl_matrix.json").is_file()
    assert (g0_dir / "usgs_level_probe.json").is_file()
    assert (g0_dir / "render_readback_comparison.json").is_file()
    assert (g0_dir / "network_proof.json").is_file()

    # Network proof must have 0 violations and valid page host counts
    net_proof = json.loads((g0_dir / "network_proof.json").read_text(encoding="utf-8"))
    assert net_proof["violations_count"] == 0
    assert net_proof["status"] == "PASS"
    assert "127.0.0.1" in net_proof["page_requests_by_host"]
    for host in net_proof["page_requests_by_host"]:
        assert host in net_proof["page_allowed_hosts"]

    # All 7 PNGs exist
    renders_dir = g0_dir / "renders"
    for fname in [
        "t2_t30_camF_rgb.png",
        "t2_t30_camR_rgb.png",
        "t2_t30_camB_rgb.png",
        "t2_t30_camL_rgb.png",
        "t2_t30_camU_rgb.png",
        "t2_t30_camD_rgb.png",
        "t2_t30_camF_mask_earth.png",
    ]:
        fpath = renders_dir / fname
        assert fpath.is_file()
        assert fpath.stat().st_size > 0

    # Renders in render_readback_comparison.json must have 0 incomplete frames and 0 readback error
    comp_data = json.loads((g0_dir / "render_readback_comparison.json").read_text(encoding="utf-8"))
    assert len(comp_data["renders"]) == 7
    for r in comp_data["renders"]:
        assert r["incomplete"] is False, f"Camera {r['camera_id']} ({r['profile']}) was incomplete!"

    for key, rb in comp_data["readbacks"].items():
        errs = rb["errors"]
        assert errs["position_error_m"] < 1e-6
        assert errs["direction_error_deg"] < 1e-6
        assert errs["up_error_deg"] < 1e-6
        assert errs["right_error_deg"] < 1e-6


def test_rig_true_seeded_perturbations():
    """Verify true rig creation with seeded perturbations, reproducibility, and clip bounds."""
    rig = Rig.load_nominal()
    true_rig1 = rig.make_true(20260922)
    true_rig2 = rig.make_true(20260922)

    assert len(true_rig1.cameras) == 6
    clip_rot = math.radians(0.60)
    clip_trans = 0.015

    for c1, c2 in zip(true_rig1.cameras, true_rig2.cameras):
        # Reproducibility
        assert np.allclose(c1.omega_cam_rad, c2.omega_cam_rad)
        assert np.allclose(c1.delta_lever_arm_m, c2.delta_lever_arm_m)
        assert np.allclose(c1.r_body_cam, c2.r_body_cam)
        # Clip limits
        assert np.all(np.abs(c1.omega_cam_rad) <= clip_rot + 1e-12)
        assert np.all(np.abs(c1.delta_lever_arm_m) <= clip_trans + 1e-12)
        # Rotation matrix orthonormal with det +1
        assert abs(np.linalg.det(c1.r_body_cam) - 1.0) < 1e-12
        assert np.allclose(c1.r_body_cam.T @ c1.r_body_cam, np.eye(3), atol=1e-12)


def test_so3_and_quaternion_math():
    """Verify so(3) Exp/Log round trips and Hamilton quaternion conversion."""
    # Zero rotation
    w_zero = np.zeros(3, dtype=np.float64)
    r_zero = rotations.so3_exp(w_zero)
    assert np.allclose(r_zero, np.eye(3))
    assert np.allclose(rotations.so3_log(r_zero), w_zero)

    # Small perturbations
    w_test = np.array([0.003, -0.005, 0.002], dtype=np.float64)
    r_test = rotations.so3_exp(w_test)
    w_rec = rotations.so3_log(r_test)
    assert np.allclose(w_test, w_rec, atol=1e-12)

    # Quaternion roundtrip
    q = rotations.r_to_quat(r_test)
    assert q[0] >= 0.0  # Canonical w >= 0
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12
    r_from_q = rotations.quat_to_r(q)
    assert np.allclose(r_test, r_from_q, atol=1e-12)


def test_wgs84_edge_cases():
    """Verify WGS84 roundtrips at South Pole, extreme elevations, and radii of curvature."""
    # South Pole (-90, 0, 0) -> (0, 0, -b)
    p_south = wgs84.geodetic_to_ecef(-90.0, 0.0, 0.0)
    assert np.allclose(p_south, [0.0, 0.0, -wgs84.WGS84_B], atol=1e-6)
    lat, lon, h = wgs84.ecef_to_geodetic(p_south[0], p_south[1], p_south[2])
    assert abs(lat - (-90.0)) < 1e-9
    assert abs(h - 0.0) < 1e-6

    # Extreme altitude: h = 20,000 m
    p_high = wgs84.geodetic_to_ecef(37.5, -122.0, 20000.0)
    lat, lon, h = wgs84.ecef_to_geodetic(p_high[0], p_high[1], p_high[2])
    assert abs(lat - 37.5) < 1e-9
    assert abs(lon - (-122.0)) < 1e-9
    assert abs(h - 20000.0) < 1e-6

    # Sub-sea level: h = -100 m
    p_low = wgs84.geodetic_to_ecef(37.5, -122.0, -100.0)
    lat, lon, h = wgs84.ecef_to_geodetic(p_low[0], p_low[1], p_low[2])
    assert abs(lat - 37.5) < 1e-9
    assert abs(lon - (-122.0)) < 1e-9
    assert abs(h - (-100.0)) < 1e-6

    # Radii of curvature
    m = wgs84.radius_m(math.radians(37.5))
    n = wgs84.radius_n(math.radians(37.5))
    assert 6.3e6 < m < 6.4e6
    assert 6.3e6 < n < 6.4e6
    assert m < n  # M < N everywhere except poles where M = N


def test_camera_bounds_and_clipping():
    """Verify pinhole camera boundary classification and ray geometry."""
    cam = PinholeCamera(1024, 1024, 100.0)
    # Corners
    assert cam.in_image(-0.5, -0.5) is True
    assert cam.in_image(1023.5, 1023.5) is True
    assert cam.in_image(-0.51, 0.0) is False
    assert cam.in_image(1023.51, 0.0) is False
    assert cam.in_image(0.0, -0.51) is False
    assert cam.in_image(0.0, 1023.51) is False

    # Point behind camera (z <= 0)
    _, _, front = cam.project_cam(np.array([0.0, 0.0, -10.0]))
    assert front is False
    _, _, front_zero = cam.project_cam(np.array([0.0, 0.0, 0.0]))
    assert front_zero is False

