"""Unit tests for WP-08 seam quality acceptance item V2.

Conforms strictly to docs/xvs/WP08_DISPLAY.md v0.1 §5:
- On every sequence, the ground-overlap mean absolute difference with ellipsoid is <= the one with infinity.
- Report both and their ratio.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from xvs.paths import get_data_root
from xvs.rig.rig import Rig, CAMERA_ORDER
from xvs.display.fusion import render_display_panorama


@pytest.mark.data
@pytest.mark.parametrize(
    "seq_name, dir_name",
    [
        ("T2", "T2_ksfo28l"),
        ("T1", "T1_cruise"),
        ("T3", "T3_turn_climb"),
    ],
)
def test_v2_seam_quality_ellipsoid_vs_infinity(check_data, seq_name: str, dir_name: str):
    """V2, seams:
    On every sequence, the ground-overlap MAD with ellipsoid is <= the one with infinity.
    Report both and their ratio.
    """
    data_root = get_data_root()
    seq_dir = data_root / "xvs" / "wp01" / "g4" / dir_name
    frames_file = check_data(seq_dir / "frames.jsonl", f"G4 {seq_name} frames.jsonl")

    with open(frames_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
    frame_record = json.loads(first_line)

    rig = Rig.load_nominal().make_true(seed=20260922)
    ac = frame_record["aircraft"]
    aircraft_pose = {
        "lat_deg": ac["lat_deg"],
        "lon_deg": ac["lon_deg"],
        "h_m": ac["h_m"],
        "yaw_rad": math.radians(ac["ypr_deg"][0]),
        "pitch_rad": math.radians(ac["ypr_deg"][1]),
        "roll_rad": math.radians(ac["ypr_deg"][2]),
    }

    # Load 6 camera images
    cam_images = {}
    for cam_id in CAMERA_ORDER:
        rel_png = frame_record["cams"][cam_id]["png"]
        img_path = check_data(seq_dir / rel_png, f"Camera {cam_id} PNG")
        im = Image.open(img_path).convert("RGB")
        cam_images[cam_id] = np.array(im, dtype=np.uint8)

    # Render with ellipsoid (downscaled resolution for fast test)
    _, stats_ell = render_display_panorama(
        cam_images, rig, aircraft_pose, depth_model="ellipsoid", width=1024, height=512
    )

    # Render with infinity
    _, stats_inf = render_display_panorama(
        cam_images, rig, aircraft_pose, depth_model="infinity", width=1024, height=512
    )

    mad_ell = stats_ell["seam_photometry"]["ground_overlap_mad"]
    mad_inf = stats_inf["seam_photometry"]["ground_overlap_mad"]
    ratio = mad_ell / max(mad_inf, 1e-9)

    print(f"\n[V2 Seam Test - {seq_name}] Ground MAD Ellipsoid: {mad_ell:.2f}, Infinity: {mad_inf:.2f}, Ratio: {ratio:.3f}")
    assert mad_ell <= mad_inf, f"Sequence {seq_name}: Ellipsoid ground MAD {mad_ell:.2f} > Infinity MAD {mad_inf:.2f}"
