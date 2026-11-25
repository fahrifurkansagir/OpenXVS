"""Unit tests and Gate G0 Acceptance V1 for WGS84 tangent cone limb model.

Conforms strictly to docs/xvs/WP06_HORIZON.md v0.1 §1.2, §3 (V1).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from xvs.geo import wgs84, rotations
from xvs.rig.rig import Rig
from xvs.horizon_att.limb_model import (
    LimbRay,
    LimbRayBatch,
    compute_cone_residuals,
    compute_residuals_and_jacobian,
    solve_attitude_gn,
    generate_synthetic_limb_rays,
)


def test_zero_residual_at_exact_limb_truth():
    """Synthetic noise-free limb rays must produce zero residual (<= 1e-13 rad) at truth."""
    rig = Rig.load_nominal().make_true(20260922)
    lat_deg, lon_deg, h_m = 37.62, -122.37, 600.0
    yaw_rad = math.radians(45.0)
    pitch_rad = math.radians(4.0)
    roll_rad = math.radians(-7.0)

    batch = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad, pts_per_cam=20
    )
    assert batch.n_rays > 20

    res = compute_cone_residuals(lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad, batch)
    assert np.max(np.abs(res)) < 1e-13, f"Max residual at truth: {np.max(np.abs(res))}"


def test_analytical_jacobian_matches_finite_differences():
    """Verify analytical Jacobian matches central finite differences to high precision."""
    rig = Rig.load_nominal().make_true(20260922)
    lat_deg, lon_deg, h_m = 47.40, 8.50, 3000.0
    yaw_rad = math.radians(130.0)
    pitch_rad = math.radians(8.0)
    roll_rad = math.radians(15.0)

    batch = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad, pts_per_cam=10
    )
    assert batch.n_rays > 10

    # Slightly perturb attitude away from truth
    p_eval = pitch_rad + math.radians(0.5)
    r_eval = roll_rad - math.radians(0.7)

    res, jac = compute_residuals_and_jacobian(
        lat_deg, lon_deg, h_m, yaw_rad, p_eval, r_eval, batch
    )

    eps = 1e-8
    # Numerical FD for roll
    r_plus = compute_cone_residuals(lat_deg, lon_deg, h_m, yaw_rad, p_eval, r_eval + eps, batch)
    r_minus = compute_cone_residuals(lat_deg, lon_deg, h_m, yaw_rad, p_eval, r_eval - eps, batch)
    j_roll_fd = (r_plus - r_minus) / (2.0 * eps)

    # Numerical FD for pitch
    p_plus = compute_cone_residuals(lat_deg, lon_deg, h_m, yaw_rad, p_eval + eps, r_eval, batch)
    p_minus = compute_cone_residuals(lat_deg, lon_deg, h_m, yaw_rad, p_eval - eps, r_eval, batch)
    j_pitch_fd = (p_plus - p_minus) / (2.0 * eps)

    diff_roll = np.abs(jac[:, 0] - j_roll_fd)
    diff_pitch = np.abs(jac[:, 1] - j_pitch_fd)

    assert np.max(diff_roll) < 1e-6, f"Roll Jacobian mismatch: {np.max(diff_roll)}"
    assert np.max(diff_pitch) < 1e-6, f"Pitch Jacobian mismatch: {np.max(diff_pitch)}"


def test_v1_exactness_1000_seeded_poses():
    """G0 Acceptance V1: Synthetic noise-free edge rays recover attitude within 1e-6 deg.

    1,000 seeded poses with |phi| <= 60 deg, |theta| <= 30 deg, h from 50 m to 12 km,
    using all cameras that see the limb.
    """
    rig = Rig.load_nominal().make_true(20260922)
    seed = 20260925
    rng = np.random.default_rng(seed)

    n_poses = 1000
    errors_deg = []
    rms_residuals_urad = []

    for _ in range(n_poses):
        lat_deg = float(rng.uniform(-65.0, 65.0))
        lon_deg = float(rng.uniform(-180.0, 180.0))
        h_m = float(rng.uniform(50.0, 12000.0))
        yaw_rad = float(rng.uniform(0.0, 2.0 * math.pi))

        pitch_true_rad = float(rng.uniform(math.radians(-30.0), math.radians(30.0)))
        roll_true_rad = float(rng.uniform(math.radians(-60.0), math.radians(60.0)))

        batch = generate_synthetic_limb_rays(
            rig, lat_deg, lon_deg, h_m, yaw_rad, pitch_true_rad, roll_true_rad,
            pts_per_cam=16, num_samples=512
        )
        if batch.n_rays < 3:
            continue

        # Initial guess within +/- 1.0 deg of truth
        pitch_init_rad = pitch_true_rad + float(rng.uniform(-math.radians(1.0), math.radians(1.0)))
        roll_init_rad = roll_true_rad + float(rng.uniform(-math.radians(1.0), math.radians(1.0)))

        fit = solve_attitude_gn(
            lat_deg, lon_deg, h_m, yaw_rad, pitch_init_rad, roll_init_rad, batch,
            max_iter=15, tol_rad=1e-12
        )

        assert fit["status"] == "CONVERGED"
        err_pitch_deg = abs(fit["pitch_deg"] - math.degrees(pitch_true_rad))
        err_roll_deg = abs(fit["roll_deg"] - math.degrees(roll_true_rad))
        err_total_deg = math.hypot(err_pitch_deg, err_roll_deg)

        errors_deg.append(err_total_deg)
        rms_residuals_urad.append(fit["rms_residual_urad"])

        # Hard V1 threshold per pose: 1e-6 deg
        assert err_total_deg < 1e-6, (
            f"V1 exactness failed at h={h_m:.1f}m, "
            f"pitch_true={math.degrees(pitch_true_rad):.3f}, roll_true={math.degrees(roll_true_rad):.3f}: "
            f"error={err_total_deg:.2e} deg"
        )

    assert len(errors_deg) >= 990
    max_err_deg = max(errors_deg)
    median_err_deg = float(np.median(errors_deg))
    p95_err_deg = float(np.percentile(errors_deg, 95))

    print(
        f"\nV1 Results over {len(errors_deg)} poses:\n"
        f"  Max error:    {max_err_deg:.2e} deg (threshold: 1.0e-06 deg)\n"
        f"  p95 error:    {p95_err_deg:.2e} deg\n"
        f"  Median error: {median_err_deg:.2e} deg\n"
        f"  Max RMS res:  {max(rms_residuals_urad):.2e} urad"
    )

    assert max_err_deg < 1e-6
