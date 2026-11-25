"""Fast unit tests for WGS84 ellipsoid geometry and local NED frames.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §3 (G1) and docs/xvs/CONVENTIONS.md §2, §3.
"""

import math
import numpy as np
import pytest

from xvs.geo import wgs84


def test_wgs84_poles_and_equator():
    """(0, 0, 0) -> (a, 0, 0); north pole -> (0, 0, b); south pole -> (0, 0, -b)."""
    p_eq = wgs84.geodetic_to_ecef(0.0, 0.0, 0.0)
    assert np.allclose(p_eq, [wgs84.WGS84_A, 0.0, 0.0], atol=1e-6)

    p_np = wgs84.geodetic_to_ecef(90.0, 0.0, 0.0)
    assert np.allclose(p_np, [0.0, 0.0, wgs84.WGS84_B], atol=1e-6)

    p_sp = wgs84.geodetic_to_ecef(-90.0, 0.0, 0.0)
    assert np.allclose(p_sp, [0.0, 0.0, -wgs84.WGS84_B], atol=1e-6)


def test_wgs84_10k_random_round_trips():
    """10,000 random round trips (phi in [-89.9, 89.9], lon in (-180, 180], h in [-500, 20000] m): max error < 1e-6 m."""
    rng = np.random.default_rng(20260922)
    n_samples = 10000

    lats = rng.uniform(-89.9, 89.9, size=n_samples)
    lons = rng.uniform(-180.0, 180.0, size=n_samples)
    heights = rng.uniform(-500.0, 20000.0, size=n_samples)

    max_pos_err = 0.0
    for i in range(n_samples):
        lat0 = lats[i]
        lon0 = lons[i]
        h0 = heights[i]

        p = wgs84.geodetic_to_ecef(lat0, lon0, h0)
        lat1, lon1, h1 = wgs84.ecef_to_geodetic(p[0], p[1], p[2])
        p_re = wgs84.geodetic_to_ecef(lat1, lon1, h1)

        err = float(np.linalg.norm(p - p_re))
        if err > max_pos_err:
            max_pos_err = err

    assert max_pos_err < 1e-6, f"Max roundtrip error was {max_pos_err} m (threshold < 1e-6 m)"


def test_ned_basis_properties():
    """NED basis orthonormal with det +1; at (0,0): n = (0,0,1), e = (0,1,0), d = (-1,0,0)."""
    n, e, d = wgs84.ned_basis(0.0, 0.0)
    assert np.allclose(n, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(e, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(d, [-1.0, 0.0, 0.0], atol=1e-12)

    r_mat = wgs84.r_ecef_ned(0.0, 0.0)
    assert abs(np.linalg.det(r_mat) - 1.0) < 1e-12
    assert np.allclose(r_mat.T @ r_mat, np.eye(3), atol=1e-12)

    # Test random locations for orthonormality and det +1
    rng = np.random.default_rng(42)
    for _ in range(50):
        lat = rng.uniform(-89.0, 89.0)
        lon = rng.uniform(-180.0, 180.0)
        r = wgs84.r_ecef_ned(lat, lon)
        assert abs(np.linalg.det(r) - 1.0) < 1e-12
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)


def test_ned_offset_to_geodetic_exact():
    """Exact tangent-plane construction matches reference values and roundtrips."""
    lat0, lon0 = 37.61210677733923, -122.35928141744020
    # Zero offset returns origin
    lat_z, lon_z, h_z = wgs84.ned_offset_to_geodetic(lat0, lon0, 0.0, 0.0)
    assert abs(lat_z - lat0) < 1e-12
    assert abs(lon_z - lon0) < 1e-12
    assert abs(h_z - 0.0) < 1e-9

    # 1 km North offset
    lat_n, lon_n, h_n = wgs84.ned_offset_to_geodetic(lat0, lon0, 1000.0, 0.0)
    assert lat_n > lat0
    assert abs(lon_n - lon0) < 1e-9
