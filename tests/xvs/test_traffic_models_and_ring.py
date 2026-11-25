"""Fast unit tests for WP-02 traffic models, ring camera, and server routes."""

import json
from pathlib import Path
import pytest
import numpy as np

from xvs.traffic import models
from xvs.traffic.ring import TrafficRing, RingCamera
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer


def test_traffic_model_builders_and_limits():
    builders = [
        ("C1_UAS", models.build_c1_uas),
        ("C2_SEP", models.build_c2_sep),
        ("C3_TWIN", models.build_c3_twin),
        ("C4_BIZJET", models.build_c4_bizjet),
        ("C5_TRANSPORT", models.build_c5_transport),
        ("C6_HELI", models.build_c6_heli),
    ]

    for cls_name, builder in builders:
        res = builder(seed=1)
        assert res.cls_id == cls_name
        assert res.triangle_count <= 20000
        assert res.dimensions_m["span_m"] > 0
        assert res.dimensions_m["length_m"] > 0


def test_calibration_assets_creation():
    cal_spheres = models.build_cal_spheres()
    assert cal_spheres.model_id == "CAL_SPHERES"
    assert len(cal_spheres.parts) == 6

    cal = models.build_cal()
    assert cal.model_id == "CAL"

    sphere_r2 = models.build_sphere_r2()
    assert sphere_r2.model_id == "SPHERE_R2"

    panel = models.build_panel()
    assert panel.model_id == "PANEL"


def test_model_export_determinism():
    repo_root = Path(__file__).resolve().parents[2]
    tmp_dir = repo_root / "output" / "test_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    m1 = models.build_c2_sep(seed=42)
    m2 = models.build_c2_sep(seed=42)

    p1 = tmp_dir / "m1.glb"
    p2 = tmp_dir / "m2.glb"

    sha1 = models.export_glb(m1, p1)
    sha2 = models.export_glb(m2, p2)

    assert sha1 == sha2
    assert p1.read_bytes() == p2.read_bytes()


def test_traffic_ring_geometry():
    spec_path = Path(__file__).resolve().parents[2] / "docs" / "xvs" / "spec" / "traffic_ring_v1.json"
    ring = TrafficRing(spec_path=spec_path)

    assert len(ring.cameras) == 8
    assert ring.width == 2448
    assert ring.height == 2048
    assert ring.fx == 2319.0
    assert ring.fy == 2319.0
    assert ring.cx == 1223.5
    assert ring.cy == 1023.5

    # Check azimuth spacing: 45 deg intervals
    for k, cam in enumerate(ring.cameras):
        expected_az = k * 45.0
        assert pytest.approx(cam.azimuth_body_deg, abs=1e-5) == expected_az


def test_traffic_ring_roi_computation():
    spec_path = Path(__file__).resolve().parents[2] / "docs" / "xvs" / "spec" / "traffic_ring_v1.json"
    ring = TrafficRing(spec_path=spec_path)

    # Center of image
    u0, v0 = ring.compute_roi(1223.5, 1023.5)
    assert u0 == 712
    assert v0 == 512

    # Top-left corner clamp
    u0, v0 = ring.compute_roi(10.0, 10.0)
    assert u0 == 0
    assert v0 == 0

    # Bottom-right corner clamp
    u0, v0 = ring.compute_roi(2400.0, 2000.0)
    assert u0 == 2448 - 1024  # 1424
    assert v0 == 2048 - 1024  # 1024


def test_render_profiles_traffic_masks():
    profiles_path = Path(__file__).resolve().parents[2] / "configs" / "xvs" / "render_profiles.json"
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles_doc = json.load(f)

    profiles = profiles_doc["profiles"]
    for p in ["mask_visible", "mask_amodal", "mask_calib"]:
        assert p in profiles
        assert p in XVSRenderer.SUPPORTED_PROFILES
        assert profiles[p]["msaa_samples"] == 1
