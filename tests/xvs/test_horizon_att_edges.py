"""Unit tests for Earth-limb edge detector (edges.py) on masks and RGB frames.

Conforms strictly to docs/xvs/WP06_HORIZON.md v0.1 §1.1 and §3.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from xvs.paths import get_data_root
from xvs.rig.rig import Rig
from xvs.sensor.chain import SensorConfig
from xvs.horizon_att.edges import (
    extract_feature_map,
    find_subpixel_edge_1d,
    detect_camera_edges,
    edges_to_ray_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_1d_subpixel_parabolic_peak():
    """Verify 1D parabolic peak fit accurately recovers sub-pixel transition."""
    # Create smooth step edge at true position x = 50.35
    true_x = 50.35
    x = np.arange(100, dtype=np.float64)
    # Sigmoid transition with width ~1.5 px
    profile = 1.0 / (1.0 + np.exp(-(x - true_x) / 0.8))

    res = find_subpixel_edge_1d(profile, search_range=(40, 60), min_gradient=0.05, sigma=0.0)
    assert res is not None
    sub_x, peak_g = res
    err = abs(sub_x - true_x)
    assert err < 0.05, f"Subpixel error {err:.4f} px exceeds 0.05 px"


@pytest.mark.data
def test_edge_detection_on_g3_mask(check_data):
    """Verify edge detection on G3 reference mask matches analytic ground truth."""
    data_root = get_data_root()
    mask_path = check_data(data_root / "xvs" / "wp01" / "g3" / "ref" / "pngs" / "H_F_h600_p0_r0_y0.png", "G3 ref mask")
    pose_path = check_data(data_root / "xvs" / "wp01" / "g3" / "ref" / "poses" / "H_F_h600_p0_r0_y0_pose.json", "G3 ref pose")

    with open(pose_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    img = np.array(Image.open(mask_path))
    rig = Rig.load_nominal().make_true(20260922)
    cam = rig.camera_model

    prior_pose = pose_data["camera_pose_requested"]
    edges = detect_camera_edges(img, "F", cam, prior_pose=prior_pose, step=16)

    assert len(edges) >= 40

    # Compare with GT horizon_uv from pose file
    gt_uv = np.array(pose_data["gt"]["horizon_uv"])
    gt_u = gt_uv[:, 0]
    gt_v = gt_uv[:, 1]

    diffs = []
    for e in edges:
        u_val = e["u"]
        v_val = e["v"]
        # Interpolate GT v at u_val
        if gt_u.min() <= u_val <= gt_u.max():
            v_gt_interp = np.interp(u_val, gt_u[::-1], gt_v[::-1])
            diffs.append(abs(v_val - v_gt_interp))

    assert len(diffs) > 30
    median_diff = float(np.median(diffs))
    max_diff = float(np.max(diffs))
    assert median_diff < 0.25, f"Median diff on mask: {median_diff:.3f} px"
    assert max_diff < 0.50, f"Max diff on mask: {max_diff:.3f} px"


@pytest.mark.data
def test_edge_detection_on_g4_rgb_frames(check_data):
    """Verify edge detection on G4 RGB frames (T1, T2, T3) matches ground truth."""
    data_root = get_data_root()
    rig = Rig.load_nominal().make_true(20260922)
    cam = rig.camera_model

    for seq in ("T1_cruise", "T2_ksfo28l", "T3_turn_climb"):
        img_path = check_data(data_root / "xvs" / "wp01" / "g4" / seq / "F" / "000000.png", f"G4 {seq} F frame")
        frames_path = check_data(data_root / "xvs" / "wp01" / "g4" / seq / "frames.jsonl", f"G4 {seq} frames.jsonl")

        with open(frames_path, "r", encoding="utf-8") as f:
            frame0 = json.loads(f.readline())

        cam_data = frame0["cams"]["F"]
        prior_pose = {
            "p_ecef_cam": cam_data["p_ecef_m"],
            "r_ecef_cam": cam_data["R_ecef_cam"],
        }

        img = np.array(Image.open(img_path))
        edges = detect_camera_edges(img, "F", cam, prior_pose=prior_pose, step=16)
        assert len(edges) >= 30, f"Too few edges ({len(edges)}) on {seq}"

        gt_uv = np.array(cam_data["gt"]["horizon_uv"])
        gt_u = gt_uv[:, 0]
        gt_v = gt_uv[:, 1]

        diffs = []
        for e in edges:
            u_val = e["u"]
            v_val = e["v"]
            if gt_u.min() <= u_val <= gt_u.max():
                v_gt_interp = np.interp(u_val, gt_u[::-1], gt_v[::-1])
                diffs.append(abs(v_val - v_gt_interp))

        median_diff = float(np.median(diffs))
        assert median_diff < 1.0, f"Median diff on {seq}: {median_diff:.3f} px"


def test_edges_to_ray_batch_with_undistortion():
    """Verify edges_to_ray_batch transforms points and applies undistortion."""
    rig = Rig.load_nominal().make_true(20260922)
    sensor_config = SensorConfig.from_file(
        REPO_ROOT / "configs" / "xvs" / "sensors" / "rig_mono_v0.json"
    )

    dummy_edges = {"F": [{"u": 512.0, "v": 530.0, "gradient": 0.5}]}
    batch = edges_to_ray_batch(dummy_edges, rig, sensor_config=sensor_config)
    assert batch.n_rays == 1
    # Boresight for F is forward (+X in body)
    assert batch.d_body[0, 0] > 0.5
    assert abs(np.linalg.norm(batch.d_body[0]) - 1.0) < 1e-12
