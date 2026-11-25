"""Tests for Earth-limb validation, coverage method, and zero-dip negative control (Gate G3).

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.3 §2.6, §3 (G3), §4.
"""

from __future__ import annotations
import math
from pathlib import Path
import tempfile
import numpy as np
import pytest

from xvs.rig.rig import Rig
from xvs.rig.camera import PinholeCamera
from xvs.sim import horizon
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer
from scripts.xvs.validate_limb import (
    build_g3_scenes,
    evaluate_limb_scene,
    THRESHOLDS,
)
from tests.xvs.test_renderer import xvs_server


def test_build_g3_scenes_counts():
    """Verify scene generation produces 128 horizontal, 16 U/D, and 8 binary scenes (152 total)."""
    scenes = build_g3_scenes()
    assert len(scenes) == 152

    horizontal = [s for s in scenes if s["group"] == "horizontal"]
    ud = [s for s in scenes if s["group"] == "ud"]
    binary = [s for s in scenes if s["group"] == "informational_binary"]

    assert len(horizontal) == 128
    assert len(ud) == 16
    assert len(binary) == 8

    n1_scenes = [s for s in scenes if s.get("is_n1_control")]
    assert len(n1_scenes) == 8
    for s in n1_scenes:
        assert s["h_m"] == 600.0
        assert s["pitch_deg"] == 0.0
        assert s["roll_deg"] == 0.0


def test_is_earth_at_pixel_classification():
    """Verify is_earth_at_pixel accurately distinguishes space vs Earth near the horizon."""
    rig = Rig.load_nominal()
    poses = rig.compute_poses(37.62, -122.37, 600.0, 0.0, 0.0, 0.0)
    cam = PinholeCamera(1024, 1024, 100.0)
    p = poses["F"]["p_ecef_cam"]
    r = poses["F"]["r_ecef_cam"]

    v_roots = horizon.analytic_horizon_v_at_u(p, r, cam, 511.5)
    assert len(v_roots) == 1
    v_gt = v_roots[0]  # ~517.4

    # 10 px above horizon (sky / space)
    assert horizon.is_earth_at_pixel(p, r, cam, 511.5, v_gt - 10.0) is False
    # 10 px below horizon (Earth)
    assert horizon.is_earth_at_pixel(p, r, cam, 511.5, v_gt + 10.0) is True


def test_zero_dip_horizon_difference_at_600m():
    """Verify zero-dip local-level plane differs from true WGS84 limb by ~5.9 px at 600m altitude."""
    rig = Rig.load_nominal()
    poses = rig.compute_poses(37.62, -122.37, 600.0, 0.0, 0.0, 0.0)
    cam = PinholeCamera(1024, 1024, 100.0)
    p = poses["F"]["p_ecef_cam"]
    r = poses["F"]["r_ecef_cam"]

    v_roots = horizon.analytic_horizon_v_at_u(p, r, cam, 511.5)
    v_gt = v_roots[0]
    v_zero_dip = horizon.zero_dip_horizon_v_at_u(p, r, cam, 511.5)

    assert v_zero_dip is not None
    delta = v_gt - v_zero_dip
    # Dip at 600m should be ~5.8985 px
    assert abs(delta - 5.8985) < 0.01
    assert delta >= THRESHOLDS["n1_median_abs_delta_min"]


@pytest.mark.render
def test_live_limb_render_and_n1_control(xvs_server):
    """Render a live mask_earth frame on ref and assert L1 and N1 control thresholds."""
    server, port = xvs_server
    rig = Rig.load_nominal().make_true(20260922)

    scene = {
        "scene_id": "test_H_F_h600_p0_r0_y0",
        "group": "horizontal",
        "camera_id": "F",
        "lat_deg": 37.62,
        "lon_deg": -122.37,
        "h_m": 600.0,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "is_n1_control": True,
        "profile": "mask_earth",
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with XVSRenderer(port=port, backend="ref", server=server) as renderer:
            res = evaluate_limb_scene(
                renderer=renderer,
                rig=rig,
                scene_spec=scene,
                out_backend_dir=tmp_path,
            )

            assert res["m3_pass"] is True
            assert res["l1_pass"] is True
            assert res["is_n1_control"] is True
            assert res["n1_pass"] is True
            assert res["all_pass"] is True

            # L1 accuracy checks
            assert res["l1_median_abs_delta"] <= THRESHOLDS["l1_median_abs_delta"]
            assert res["l1_p95_abs_delta"] <= THRESHOLDS["l1_p95_abs_delta"]
            assert res["l1_max_abs_delta"] <= THRESHOLDS["l1_max_abs_delta"]
            assert abs(res["l1_signed_mean"]) <= THRESHOLDS["l1_abs_signed_mean"]

            # N1 negative control check
            assert res["n1_median_abs_delta"] >= THRESHOLDS["n1_median_abs_delta_min"]
            assert res["drawing_buffer"] == [1024, 1024]
            assert len(res["rgb_sha256"]) == 64
