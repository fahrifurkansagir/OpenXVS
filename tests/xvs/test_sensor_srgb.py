"""Tests for sRGB EOTF/OETF and monochrome conversion (V1).

Spec reference: docs/xvs/WP03_SENSOR.md §3 item V1.
"""

from __future__ import annotations

import numpy as np
import pytest

from xvs.sensor.srgb import (
    srgb_eotf,
    srgb_oetf,
    srgb_to_8bit,
    linear_rgb_to_monochrome,
    REC709_R_WEIGHT,
    REC709_G_WEIGHT,
    REC709_B_WEIGHT,
)


def test_v1_srgb_float64_inverse_exactness():
    """V1 requirement: EOTF and OETF are exact inverses in float64: error <= 1e-12."""
    c = np.linspace(0.0, 1.0, 100_000, dtype=np.float64)
    linear = srgb_eotf(c)
    c_rec = srgb_oetf(linear)
    max_err_c = np.max(np.abs(c - c_rec))
    assert max_err_c <= 1e-12, f"Max sRGB round-trip error {max_err_c} exceeds 1e-12"

    l_grid = np.linspace(0.0, 1.0, 100_000, dtype=np.float64)
    c_grid = srgb_oetf(l_grid)
    l_rec = srgb_eotf(c_grid)
    max_err_l = np.max(np.abs(l_grid - l_rec))
    assert max_err_l <= 1e-12, f"Max linear round-trip error {max_err_l} exceeds 1e-12"


def test_v1_srgb_8bit_round_trip():
    """V1 requirement: An 8-bit round trip is exact for all 256 codes."""
    codes = np.arange(256, dtype=np.uint8)
    linear = srgb_eotf(codes)
    srgb = srgb_oetf(linear)
    recovered_codes = srgb_to_8bit(srgb)
    mismatches = np.sum(codes != recovered_codes)
    assert mismatches == 0, f"Found {mismatches} mismatches in 8-bit round trip"


def test_linear_rgb_to_monochrome():
    """Verify Rec. 709 weights and shape preservation."""
    rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float64)
    mono = linear_rgb_to_monochrome(rgb)
    assert mono.shape == (1, 3)
    np.testing.assert_allclose(mono[0, 0], REC709_R_WEIGHT)
    np.testing.assert_allclose(mono[0, 1], REC709_G_WEIGHT)
    np.testing.assert_allclose(mono[0, 2], REC709_B_WEIGHT)
