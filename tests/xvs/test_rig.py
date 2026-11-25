"""Fast unit tests for Rig geometry, spherical camera coverage, and true rig reproducibility.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §3 (G1) and docs/xvs/CONVENTIONS.md §8.
"""

import math
import numpy as np
import pytest

from xvs.rig.camera import PinholeCamera
from xvs.rig.rig import Rig, CAMERA_ORDER


def test_nominal_rig_properties():
    """Nominal rig loads; matrices orthonormal with det +1; boresights match."""
    rig = Rig.load_nominal()
    assert len(rig.cameras) == 6
    assert tuple(c.id for c in rig.cameras) == CAMERA_ORDER

    for c in rig.cameras:
        # Orthonormal with det +1
        assert abs(np.linalg.det(c.r_body_cam) - 1.0) < 1e-12
        assert np.allclose(c.r_body_cam.T @ c.r_body_cam, np.eye(3), atol=1e-12)

        # Boresight matches z-axis of camera in body frame
        z_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        boresight_calc = c.r_body_cam @ z_cam
        assert np.allclose(boresight_calc, c.boresight_body, atol=1e-12)


def test_100k_directions_coverage():
    """100k random directions: every one is covered by >= 1 camera; fraction covered by >= 2 is 0.1867 +/- 0.005."""
    rig = Rig.load_nominal()
    cam = PinholeCamera(1024, 1024, 100.0)

    # Sample 100k random unit vectors uniformly on unit sphere
    rng = np.random.default_rng(20260922)
    n_rays = 100000
    z = rng.uniform(-1.0, 1.0, size=n_rays)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n_rays)
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    rays_body = np.stack([x, y, z], axis=0)  # (3, 100000)

    coverage_counts = np.zeros(n_rays, dtype=np.int32)

    for c in rig.cameras:
        # ray in cam frame = R_body_cam.T @ ray_body
        ray_cam = c.r_body_cam.T @ rays_body
        in_front = ray_cam[2] > 0.0

        u = np.full(n_rays, np.nan, dtype=np.float64)
        v = np.full(n_rays, np.nan, dtype=np.float64)
        u[in_front] = cam.fx * ray_cam[0, in_front] / ray_cam[2, in_front] + cam.cx
        v[in_front] = cam.fy * ray_cam[1, in_front] / ray_cam[2, in_front] + cam.cy

        in_image = in_front & (-0.5 <= u) & (u <= 1023.5) & (-0.5 <= v) & (v <= 1023.5)
        coverage_counts += in_image.astype(np.int32)

    min_coverage = int(np.min(coverage_counts))
    fraction_ge_2 = float(np.mean(coverage_counts >= 2))

    assert min_coverage >= 1, f"Expected every direction covered by >= 1 camera, but min was {min_coverage}"
    expected_frac = 0.1867
    assert abs(fraction_ge_2 - expected_frac) <= 0.005, f"Expected fraction >= 2 to be {expected_frac} +/- 0.005, got {fraction_ge_2}"


def test_rig_true_reproducibility_and_clips():
    """rig_true is reproducible from the seed; perturbations stay within the clips."""
    rig = Rig.load_nominal()
    true1 = rig.make_true(20260922)
    true2 = rig.make_true(20260922)

    clip_rot = math.radians(0.60)
    clip_trans = 0.015

    for c1, c2 in zip(true1.cameras, true2.cameras):
        # Reproducibility
        assert np.allclose(c1.omega_cam_rad, c2.omega_cam_rad, atol=1e-15)
        assert np.allclose(c1.delta_lever_arm_m, c2.delta_lever_arm_m, atol=1e-15)
        assert np.allclose(c1.r_body_cam, c2.r_body_cam, atol=1e-15)
        assert np.allclose(c1.lever_arm_m, c2.lever_arm_m, atol=1e-15)

        # Clip bounds
        assert np.all(np.abs(c1.omega_cam_rad) <= clip_rot + 1e-12)
        assert np.all(np.abs(c1.delta_lever_arm_m) <= clip_trans + 1e-12)

        # Orthonormality and det +1
        assert abs(np.linalg.det(c1.r_body_cam) - 1.0) < 1e-12
        assert np.allclose(c1.r_body_cam.T @ c1.r_body_cam, np.eye(3), atol=1e-12)


def test_rig_record_nominal_and_true():
    """Verify rig.record() returns exact metadata without silent fallbacks (C16)."""
    nom = Rig.load_nominal()
    assert nom.record() == {"kind": "nominal", "seed": None}

    true_rig = nom.make_true(20260922)
    assert true_rig.record() == {"kind": "true", "seed": 20260922}
