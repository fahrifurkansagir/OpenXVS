"""V5 determinism test for WP-08 display pipeline.

Verifies:
- Byte-identical rendering and statistics across two independent runs.
- Identical SHA-256 PNG hashes for rendered panorama, viewport, and symbology stills.
- Deterministic JSON serialisation.
Conforms strictly to WP08_DISPLAY.md §5 (V5).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import numpy as np
from PIL import Image

from xvs.rig.rig import Rig, CAMERA_ORDER
from xvs.display.fusion import (
    render_display_panorama,
    compute_world_points,
    generate_panorama_rays,
)
from xvs.display.viewport import (
    get_viewport_camera,
    compute_viewport_pose,
    render_viewport_frame,
)
from xvs.display.symbology import draw_symbology


def _generate_synthetic_test_cams(rig: Rig):
    rng = np.random.default_rng(20260925)
    cams = {}
    for name in CAMERA_ORDER:
        img = rng.integers(30, 220, size=(rig.height, rig.width, 3), dtype=np.uint8)
        cams[name] = img
    return cams


def test_v5_display_determinism():
    """V5: verify byte-identical rendering, PNG hashing, and metrics determinism."""
    rig = Rig.load_nominal().make_true(seed=20260922)
    cams = _generate_synthetic_test_cams(rig)

    aircraft_pose = {
        "lat_deg": 37.6188,
        "lon_deg": -122.3750,
        "h_m": 250.0,
        "yaw_rad": math.radians(280.0),
        "pitch_rad": math.radians(3.0),
        "roll_rad": math.radians(0.0),
    }
    head_euler = (math.radians(10.0), math.radians(-5.0), 0.0)
    v_ned = np.array([30.0, -60.0, 3.5], dtype=np.float64)

    # --- Run 1 ---
    pan1, stats1 = render_display_panorama(
        cams, rig, aircraft_pose, depth_model="ellipsoid", width=256, height=128
    )
    vp1 = render_viewport_frame(
        cams, rig, aircraft_pose, head_euler, depth_model="ellipsoid"
    )
    vp_cam = get_viewport_camera()
    vp_pose = compute_viewport_pose(aircraft_pose, head_euler)
    sym1 = draw_symbology(
        vp1,
        aircraft_pose,
        v_ned,
        vp_pose["p_ecef_view"],
        vp_pose["r_ecef_view"],
        vp_cam,
        is_t2=True,
    )

    # --- Run 2 ---
    pan2, stats2 = render_display_panorama(
        cams, rig, aircraft_pose, depth_model="ellipsoid", width=256, height=128
    )
    vp2 = render_viewport_frame(
        cams, rig, aircraft_pose, head_euler, depth_model="ellipsoid"
    )
    sym2 = draw_symbology(
        vp2,
        aircraft_pose,
        v_ned,
        vp_pose["p_ecef_view"],
        vp_pose["r_ecef_view"],
        vp_cam,
        is_t2=True,
    )

    # 1. Exact array equality
    assert np.array_equal(pan1, pan2), "Panorama rendering is non-deterministic"
    assert np.array_equal(vp1, vp2), "Viewport rendering is non-deterministic"
    assert np.array_equal(sym1, sym2), "Symbology overlay is non-deterministic"

    # 2. Exact PNG hash identity
    b1_pan = io.BytesIO()
    Image.fromarray(pan1).save(b1_pan, format="PNG")
    h1_pan = hashlib.sha256(b1_pan.getvalue()).hexdigest()

    b2_pan = io.BytesIO()
    Image.fromarray(pan2).save(b2_pan, format="PNG")
    h2_pan = hashlib.sha256(b2_pan.getvalue()).hexdigest()

    assert h1_pan == h2_pan, f"Panorama PNG hash mismatch: {h1_pan} != {h2_pan}"

    b1_sym = io.BytesIO()
    Image.fromarray(sym1).save(b1_sym, format="PNG")
    h1_sym = hashlib.sha256(b1_sym.getvalue()).hexdigest()

    b2_sym = io.BytesIO()
    Image.fromarray(sym2).save(b2_sym, format="PNG")
    h2_sym = hashlib.sha256(b2_sym.getvalue()).hexdigest()

    assert h1_sym == h2_sym, f"Symbology PNG hash mismatch: {h1_sym} != {h2_sym}"

    # 3. Exact JSON serialization identity
    json1 = json.dumps(stats1, sort_keys=True, indent=2)
    json2 = json.dumps(stats2, sort_keys=True, indent=2)
    assert json1 == json2, "Statistics JSON is non-deterministic"
