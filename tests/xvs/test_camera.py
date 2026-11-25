"""Fast unit tests for PinholeCamera model, projection, back-projection, and intrinsics.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §3 (G1) and docs/xvs/CONVENTIONS.md §5.
"""

import math
import numpy as np
import pytest

from xvs.rig.camera import PinholeCamera


def test_pinhole_camera_intrinsics_exact():
    """fx = 429.619011162767 ± 1e-9; the principal ray maps to (511.5, 511.5)."""
    cam = PinholeCamera(width=1024, height=1024, fov_h_deg=100.0)
    assert abs(cam.fx - 429.6190111627674) < 1e-9
    assert abs(cam.fy - 429.6190111627674) < 1e-9
    assert abs(cam.cx - 511.5) < 1e-9
    assert abs(cam.cy - 511.5) < 1e-9

    u, v, in_front = cam.project_cam(np.array([0.0, 0.0, 10.0]))
    assert in_front is True
    assert abs(u - 511.5) < 1e-9
    assert abs(v - 511.5) < 1e-9


def test_horizontal_rays_at_50_degrees():
    """Horizontal rays at +50° / −50° map to u = 1023.5 / −0.5 (± 1e-9)."""
    cam = PinholeCamera(width=1024, height=1024, fov_h_deg=100.0)

    # Ray at +50 deg: [tan(50 deg), 0, 1]
    ray_pos = np.array([math.tan(math.radians(50.0)), 0.0, 1.0])
    u_pos, v_pos, in_front_pos = cam.project_cam(ray_pos)
    assert in_front_pos is True
    assert abs(u_pos - 1023.5) < 1e-9
    assert abs(v_pos - 511.5) < 1e-9

    # Ray at -50 deg: [tan(-50 deg), 0, 1]
    ray_neg = np.array([math.tan(math.radians(-50.0)), 0.0, 1.0])
    u_neg, v_neg, in_front_neg = cam.project_cam(ray_neg)
    assert in_front_neg is True
    assert abs(u_neg - (-0.5)) < 1e-9
    assert abs(v_neg - 511.5) < 1e-9


def test_project_unproject_roundtrip():
    """project ∘ unproject round trip < 1e-9 px."""
    cam = PinholeCamera(width=1024, height=1024, fov_h_deg=100.0)
    rng = np.random.default_rng(20260922)

    n_samples = 1000
    u_samples = rng.uniform(-0.5, 1023.5, size=n_samples)
    v_samples = rng.uniform(-0.5, 1023.5, size=n_samples)

    for u0, v0 in zip(u_samples, v_samples):
        ray = cam.unproject(u0, v0)
        # Scale ray to random depth
        depth = rng.uniform(1.0, 1000.0)
        p_cam = ray * depth
        u1, v1, in_front = cam.project_cam(p_cam)
        assert in_front is True
        assert abs(u1 - u0) < 1e-9
        assert abs(v1 - v0) < 1e-9


def test_boundary_clipping_and_rear_points():
    """Verify in_image boundaries [-0.5, W-0.5] x [-0.5, H-0.5] and rear point rejection."""
    cam = PinholeCamera(width=1024, height=1024, fov_h_deg=100.0)
    assert cam.in_image(-0.5, -0.5) is True
    assert cam.in_image(1023.5, 1023.5) is True
    assert cam.in_image(-0.50001, 511.5) is False
    assert cam.in_image(1023.50001, 511.5) is False
    assert cam.in_image(511.5, -0.50001) is False
    assert cam.in_image(511.5, 1023.50001) is False

    # Points with z <= 0
    _, _, f1 = cam.project_cam(np.array([0.0, 0.0, 0.0]))
    assert f1 is False
    _, _, f2 = cam.project_cam(np.array([0.0, 0.0, -10.0]))
    assert f2 is False
