"""Tests for optical vignetting model (V4).

Spec reference: docs/xvs/WP03_SENSOR.md §3 item V4:
"With a flat field in noise-free mode, the output profile matches the model within 0.1 %."
"""

from __future__ import annotations

import numpy as np
import pytest

from xvs.sensor.vignetting import compute_vignetting_map, apply_vignetting


PRESETS = [
    {
        "name": "ring_mono_v0",
        "W": 2448,
        "H": 2048,
        "fx": 2319.0,
        "fy": 2319.0,
        "cx": 1223.5,
        "cy": 1023.5,
        "v1": 0.0,
        "v2": 0.0,
    },
    {
        "name": "rig_mono_v0",
        "W": 1024,
        "H": 1024,
        "fx": 429.6190111627674,
        "fy": 429.6190111627674,
        "cx": 511.5,
        "cy": 511.5,
        "v1": 0.0,
        "v2": 0.0,
    },
]


@pytest.mark.parametrize("p", PRESETS)
def test_v4_vignetting_flat_field_exact_match(p):
    """V4: With a flat field in noise-free mode, output profile matches model within 0.1 %."""
    v_map = compute_vignetting_map(
        width=p["W"],
        height=p["H"],
        fx=p["fx"],
        fy=p["fy"],
        cx=p["cx"],
        cy=p["cy"],
        v1=p["v1"],
        v2=p["v2"],
    )

    flat_field = np.ones((p["H"], p["W"]), dtype=np.float64)
    output = apply_vignetting(flat_field, v_map)

    # Relative difference between output and theoretical model
    rel_diff = np.abs(output - v_map) / v_map
    max_rel_diff = float(np.max(rel_diff))

    assert max_rel_diff <= 0.001, f"[{p['name']}] Max relative error {max_rel_diff:.2e} exceeds 0.1 %"


def test_vignetting_on_axis_unity():
    """At optical center (u=cx, v=cy), vignetting should be exactly 1.0."""
    v_map = compute_vignetting_map(
        width=100, height=100, fx=50.0, fy=50.0, cx=50.0, cy=50.0, v1=-0.1, v2=0.05
    )
    assert np.isclose(v_map[50, 50], 1.0, atol=1e-12)


def test_vignetting_radial_decay():
    """Vignetting should decay towards the image corners."""
    v_map = compute_vignetting_map(
        width=1024, height=1024, fx=429.6, fy=429.6, cx=511.5, cy=511.5, v1=0.0, v2=0.0
    )
    center_val = v_map[512, 512]
    corner_val = v_map[0, 0]
    assert center_val > corner_val
    assert corner_val > 0.0
