"""Fast unit tests for Rotations, Euler angles, Quaternions, and so(3) Lie algebra.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §3 (G1) and docs/xvs/CONVENTIONS.md §4.
"""

import math
import numpy as np
import pytest

from xvs.geo import rotations


def test_conventions_section4_mandatory_signs():
    """Mandatory sign tests:
      - psi = 90 deg => body x points East.
      - theta > 0 => nose up (body x has a negative Down component).
      - phi > 0 => right wing down (body y has a positive Down component).
    """
    # psi = 90 deg -> body x points East (NED y = [0, 1, 0])
    r_yaw90 = rotations.euler_to_r(math.radians(90.0), 0.0, 0.0)
    body_x_ned = r_yaw90 @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(body_x_ned, [0.0, 1.0, 0.0], atol=1e-12)

    # theta > 0 -> nose up (body x has negative Down component, NED z < 0)
    for theta_deg in [1.0, 10.0, 45.0]:
        r_pitch = rotations.euler_to_r(0.0, math.radians(theta_deg), 0.0)
        body_x_p = r_pitch @ np.array([1.0, 0.0, 0.0])
        assert body_x_p[2] < 0.0, f"Expected negative Down component for theta={theta_deg}"

    # phi > 0 -> right wing down (body y has positive Down component, NED z > 0)
    for phi_deg in [1.0, 10.0, 45.0]:
        r_roll = rotations.euler_to_r(0.0, 0.0, math.radians(phi_deg))
        body_y_r = r_roll @ np.array([0.0, 1.0, 0.0])
        assert body_y_r[2] > 0.0, f"Expected positive Down component for phi={phi_deg}"


def test_euler_r_quat_roundtrips():
    """Euler <-> R <-> quat round trips (|theta| < 89.9 deg) < 1e-12."""
    rng = np.random.default_rng(20260922)
    n_samples = 500

    yaws = rng.uniform(-math.pi, math.pi, size=n_samples)
    pitches = rng.uniform(-math.radians(89.0), math.radians(89.0), size=n_samples)
    rolls = rng.uniform(-math.pi, math.pi, size=n_samples)

    for i in range(n_samples):
        yaw = yaws[i]
        pitch = pitches[i]
        roll = rolls[i]

        r_mat = rotations.euler_to_r(yaw, pitch, roll)
        # Check orthogonality & det
        assert abs(np.linalg.det(r_mat) - 1.0) < 1e-12
        assert np.allclose(r_mat.T @ r_mat, np.eye(3), atol=1e-12)

        # R -> Euler -> R
        y_rec, p_rec, r_rec = rotations.r_to_euler(r_mat)
        r_rec_mat = rotations.euler_to_r(y_rec, p_rec, r_rec)
        assert np.allclose(r_mat, r_rec_mat, atol=1e-12)

        # R -> quat -> R
        q = rotations.r_to_quat(r_mat)
        assert q[0] >= 0.0  # Canonical w >= 0
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12
        r_from_q = rotations.quat_to_r(q)
        assert np.allclose(r_mat, r_from_q, atol=1e-12)


def test_so3_exp_log_roundtrip():
    """so(3) Exp/Log round trips < 1e-12."""
    rng = np.random.default_rng(20260922)
    n_samples = 500

    # Test zero
    w0 = np.zeros(3, dtype=np.float64)
    assert np.allclose(rotations.so3_exp(w0), np.eye(3), atol=1e-12)
    assert np.allclose(rotations.so3_log(np.eye(3)), w0, atol=1e-12)

    # Random small to moderate rotation vectors
    omegas = rng.uniform(-math.radians(45.0), math.radians(45.0), size=(n_samples, 3))
    for omega in omegas:
        r = rotations.so3_exp(omega)
        assert abs(np.linalg.det(r) - 1.0) < 1e-12
        w_rec = rotations.so3_log(r)
        assert np.allclose(omega, w_rec, atol=1e-12)
