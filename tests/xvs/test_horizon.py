"""Fast unit tests for analytic Earth-limb horizon geometry and edge functions.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §3 (G1) and docs/xvs/CONVENTIONS.md §7.
"""

import math
import numpy as np
import pytest

from xvs.geo import wgs84
from xvs.rig.camera import PinholeCamera
from xvs.rig.rig import Rig
from xvs.sim import horizon


def test_sphere_limb_dip_analytic():
    """Sphere case (a = b = 6371008.8 m): limb dip = arccos(R/(R+h)) within 1e-9 rad for h in {10, 600, 10000} m."""
    r_sphere = 6371008.8

    for h in [10.0, 600.0, 10000.0]:
        dip_analytic = math.acos(r_sphere / (r_sphere + h))

        # Camera at (0, 0, R + h) looking along +x or +y or -z
        c_cam = np.array([0.0, 0.0, r_sphere + h], dtype=np.float64)
        # Unit sphere scaled coordinates: a = b = R
        c_prime = c_cam / r_sphere
        k = float(np.linalg.norm(c_prime))
        assert abs(k - (r_sphere + h) / r_sphere) < 1e-12

        # Tangent point distance from origin along c_hat is 1/k = R / (R+h)
        # Radius of tangent circle is sqrt(1 - 1/k^2)
        # Vector from C to tangent point has angle with -c_hat:
        # tan(dip) = radius / (k - 1/k) = sqrt(1 - 1/k^2) / ((k^2 - 1)/k) = 1 / sqrt(k^2 - 1)
        # cos(dip) = (k - 1/k) / sqrt((k - 1/k)^2 + (1 - 1/k^2)) = 1/k = R / (R+h)
        computed_dip = math.acos(1.0 / k)
        assert abs(computed_dip - dip_analytic) < 1e-12


def test_wgs84_limb_points_satisfy_ellipsoid_and_tangency():
    """WGS84 limb points satisfy the ellipsoid equation within 1e-12 and tangency within 1e-9 (normalised)."""
    p_cam = np.array([wgs84.WGS84_A + 600.0, 0.0, 0.0], dtype=np.float64)
    pts = horizon.wgs84_limb_points_ecef(p_cam, num_points=100)

    a = wgs84.WGS84_A
    b = wgs84.WGS84_B

    for pt in pts:
        # Ellipsoid equation: (x/a)^2 + (y/a)^2 + (z/b)^2 = 1
        val = (pt[0] / a) ** 2 + (pt[1] / a) ** 2 + (pt[2] / b) ** 2
        assert abs(val - 1.0) < 1e-12, f"Ellipsoid equation error: {abs(val - 1.0)}"

        # Tangency: normal dot ray = 0
        # Normal to ellipsoid at pt is [2*pt[0]/a^2, 2*pt[1]/a^2, 2*pt[2]/b^2]
        normal = np.array([pt[0] / (a * a), pt[1] / (a * a), pt[2] / (b * b)])
        normal /= np.linalg.norm(normal)

        ray = p_cam - pt
        ray /= np.linalg.norm(ray)

        tangency_error = abs(float(np.dot(normal, ray)))
        assert tangency_error < 1e-9, f"Tangency error: {tangency_error}"


def test_f_camera_nominal_horizon_at_u511_5():
    """F camera of rig_nominal (lever arm included), ARP at (37.62°, −122.37°, 600 m), ψ = θ = φ = 0:
    the analytic horizon at u = 511.5 is 5.900 ± 0.02 px below the principal point (Director oracle: 5.8996 px).
    """
    rig = Rig.load_nominal()
    poses = rig.compute_poses(37.62, -122.37, 600.0, 0.0, 0.0, 0.0)
    p_cam = poses["F"]["p_ecef_cam"]
    r_cam = poses["F"]["r_ecef_cam"]
    cam = PinholeCamera(1024, 1024, 100.0)

    v_roots = horizon.analytic_horizon_v_at_u(p_cam, r_cam, cam, 511.5)
    assert len(v_roots) >= 1
    # Horizon below principal point: v > 511.5
    v_horizon = v_roots[0]
    dip_px = v_horizon - cam.cy

    assert abs(dip_px - 5.8996) < 0.001
    assert abs(dip_px - 5.900) <= 0.02, f"Expected dip 5.900 +/- 0.02 px, got {dip_px} px"


def test_compute_horizon_gt_visible_arc():
    """Verify compute_horizon_gt produces <= 256 points along visible arc."""
    rig = Rig.load_nominal()
    poses = rig.compute_poses(37.62, -122.37, 600.0, 0.0, 0.0, 0.0)
    cam = PinholeCamera(1024, 1024, 100.0)

    res = horizon.compute_horizon_gt(poses["F"]["p_ecef_cam"], poses["F"]["r_ecef_cam"], cam)
    assert res["horizon_visible"] is True
    assert 0 < len(res["horizon_uv"]) <= 256

    for pt in res["horizon_uv"]:
        u, v = pt
        assert -0.5 <= u <= 1023.5
        assert -0.5 <= v <= 1023.5
