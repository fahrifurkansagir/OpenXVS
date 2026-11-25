"""Unit test for WP-07 G0 acceptance item V3: Fault injection reproducibility.

Spec requirement:
- V3, fault injection reproducible: Same seed, identical images (SHA-256),
  and the fault log matches.
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest

from xvs.integrity.faults import (
    compute_sha256,
    inject_occlusion,
    inject_spurious_edge,
    inject_altitude_bias,
    inject_glare,
    SeededFaultInjector,
)


def _create_synthetic_test_image(height: int = 256, width: int = 256) -> np.ndarray:
    """Generate reproducible base synthetic image with sky/ground gradient."""
    rng = np.random.Generator(np.random.PCG64(42))
    base = rng.integers(50, 200, (height, width), dtype=np.uint8)
    # Add a horizontal transition mimicking horizon
    base[: height // 2, :] = np.clip(base[: height // 2, :] + 30, 0, 255)
    return base


def test_v3_occlusion_reproducibility():
    """V3: Occlusion injection produces identical SHA-256 and matching fault logs for same seed."""
    img = _create_synthetic_test_image()

    injector1 = SeededFaultInjector(master_seed=20260927)
    rng1 = injector1.get_rng("T1_cruise", "occlusion", seed_idx=1)
    out1, log1 = inject_occlusion(img, rng=rng1, cam_id="F")

    injector2 = SeededFaultInjector(master_seed=20260927)
    rng2 = injector2.get_rng("T1_cruise", "occlusion", seed_idx=1)
    out2, log2 = inject_occlusion(img, rng=rng2, cam_id="F")

    # Identical images
    np.testing.assert_array_equal(out1, out2)
    assert compute_sha256(out1) == compute_sha256(out2)
    assert log1 == log2

    # Different seed produces distinct draw
    rng3 = injector1.get_rng("T1_cruise", "occlusion", seed_idx=2)
    out3, log3 = inject_occlusion(img, rng=rng3, cam_id="F")
    assert compute_sha256(out1) != compute_sha256(out3)
    assert log1["col_start"] != log3["col_start"] or log1["sha256"] != log3["sha256"]


def test_v3_spurious_edge_reproducibility():
    """V3: Spurious edge injection produces identical SHA-256 and matching fault logs."""
    img = _create_synthetic_test_image()
    horizon_uv = [[float(u), 128.0 + 0.05 * u] for u in range(0, 256, 10)]

    injector1 = SeededFaultInjector(master_seed=20260927)
    rng1 = injector1.get_rng("T1_cruise", "spurious_edge", seed_idx=3)
    out1, log1 = inject_spurious_edge(img, horizon_uv=horizon_uv, rng=rng1, cam_id="R")

    injector2 = SeededFaultInjector(master_seed=20260927)
    rng2 = injector2.get_rng("T1_cruise", "spurious_edge", seed_idx=3)
    out2, log2 = inject_spurious_edge(img, horizon_uv=horizon_uv, rng=rng2, cam_id="R")

    np.testing.assert_array_equal(out1, out2)
    assert compute_sha256(out1) == compute_sha256(out2)
    assert log1 == log2

    # Verifications on fault specifications
    assert 5.0 <= abs(log1["offset_px"]) <= 40.0
    assert 0.40 <= log1["width_fraction"] <= 1.00


def test_v3_altitude_bias_reproducibility():
    """V3: Altitude bias injection matches exactly h_true + 200 m and logs match."""
    h_true = 3000.0
    h_meas1, log1 = inject_altitude_bias(h_true)
    h_meas2, log2 = inject_altitude_bias(h_true)

    assert h_meas1 == h_true + 200.0
    assert h_meas1 == h_meas2
    assert log1 == log2
    assert log1["bias_m"] == 200.0


def test_v3_glare_reproducibility():
    """V3: Glare injection produces identical SHA-256 and matching fault logs."""
    img = _create_synthetic_test_image()
    horizon_uv = [[float(u), 128.0] for u in range(0, 256, 10)]

    injector1 = SeededFaultInjector(master_seed=20260927)
    rng1 = injector1.get_rng("T2_ksfo28l", "glare", seed_idx=7)
    out1, log1 = inject_glare(img, horizon_uv=horizon_uv, radius_px=50.0, rng=rng1, cam_id="B")

    injector2 = SeededFaultInjector(master_seed=20260927)
    rng2 = injector2.get_rng("T2_ksfo28l", "glare", seed_idx=7)
    out2, log2 = inject_glare(img, horizon_uv=horizon_uv, radius_px=50.0, rng=rng2, cam_id="B")

    np.testing.assert_array_equal(out1, out2)
    assert compute_sha256(out1) == compute_sha256(out2)
    assert log1 == log2
    assert log1["radius_px"] == 50.0
    assert log1["pixel_count"] > 0


def test_v3_onset_and_camera_selection_reproducible():
    """Verify onset frame and target camera selection are deterministic and valid."""
    injector1 = SeededFaultInjector(master_seed=20260927)
    injector2 = SeededFaultInjector(master_seed=20260927)

    onset1 = injector1.choose_onset_frame(120, "T1_cruise", "occlusion", 5)
    onset2 = injector2.choose_onset_frame(120, "T1_cruise", "occlusion", 5)
    assert onset1 == onset2
    assert 0 < onset1 < 120

    cam1 = injector1.choose_target_camera("T1_cruise", "spurious_edge", 5)
    cam2 = injector2.choose_target_camera("T1_cruise", "spurious_edge", 5)
    assert cam1 == cam2
    assert cam1 in ("F", "R", "B", "L")
