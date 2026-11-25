"""Unit tests and Gate G0 Acceptance V2 (G3 masks evaluation).

Conforms strictly to docs/xvs/WP06_HORIZON.md v0.1 §1.3, §3 (V2).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from xvs.rig.rig import Rig
from xvs.horizon_att.limb_model import (
    LimbRay,
    LimbRayBatch,
    generate_synthetic_limb_rays,
)
from xvs.horizon_att.edges import (
    detect_camera_edges,
    edges_to_ray_batch,
)
from xvs.horizon_att.fit import (
    ransac_fit_attitude,
    estimate_attitude_from_images,
)

from xvs.paths import get_data_root

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ransac_rejects_outliers():
    """Verify RANSAC successfully filters out large angular outliers."""
    rig = Rig.load_nominal().make_true(20260922)
    lat_deg, lon_deg, h_m = 37.62, -122.37, 1000.0
    yaw_rad = math.radians(30.0)
    pitch_true_rad = math.radians(3.5)
    roll_true_rad = math.radians(-5.0)

    clean_batch = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad, pitch_true_rad, roll_true_rad, pts_per_cam=15
    )
    assert clean_batch.n_rays >= 40

    rng = np.random.default_rng(20260925)
    # Inject 25% outlier rays with large random directions (1-5 deg error)
    n_outliers = int(0.25 * clean_batch.n_rays)
    outlier_indices = rng.choice(clean_batch.n_rays, size=n_outliers, replace=False)

    d_corrupted = np.copy(clean_batch.d_body)
    for idx in outlier_indices:
        pert = rng.normal(size=3)
        pert -= np.dot(pert, d_corrupted[idx]) * d_corrupted[idx]
        pert = pert / np.linalg.norm(pert) * math.radians(rng.uniform(1.0, 5.0))
        d_corrupted[idx] = (d_corrupted[idx] + pert) / np.linalg.norm(d_corrupted[idx] + pert)

    corrupted_batch = LimbRayBatch(
        d_corrupted,
        clean_batch.lever_arm_body,
        clean_batch.cam_ids,
    )

    fit = ransac_fit_attitude(
        lat_deg, lon_deg, h_m, yaw_rad, corrupted_batch,
        initial_guess_rad=(pitch_true_rad + math.radians(0.5), roll_true_rad - math.radians(0.5)),
        max_iterations=60,
        inlier_threshold_rad=math.radians(0.3),
        rng_seed=20260925,
    )

    assert fit["status"] == "CONVERGED"
    err_p = abs(fit["pitch_deg"] - math.degrees(pitch_true_rad))
    err_r = abs(fit["roll_deg"] - math.degrees(roll_true_rad))
    assert err_p < 0.05, f"Pitch error with outliers: {err_p:.4f} deg"
    assert err_r < 0.05, f"Roll error with outliers: {err_r:.4f} deg"


@pytest.mark.data
def test_v2_acceptance_g3_masks(check_data):
    """Gate G0 Acceptance V2: Attitude recovery from G3 reference masks.

    On G3 ref masks (horizontal cameras, h >= 600m, single camera per scene):
      - |pitch error| median <= 0.02 deg, max <= 0.10 deg
      - |roll error|  median <= 0.05 deg, max <= 0.25 deg
    Report h = 150m separately.
    """
    data_root = get_data_root()
    g3_dir = data_root / "xvs" / "wp01" / "g3" / "ref"
    pose_dir = check_data(g3_dir / "poses", "G3 reference poses")
    png_dir = check_data(g3_dir / "pngs", "G3 reference pngs")

    pose_files = sorted(list(pose_dir.glob("H_*.json")))
    assert len(pose_files) == 128, f"Expected 128 horizontal scenes, found {len(pose_files)}"

    rig = Rig.load_nominal().make_true(20260922)
    cam = rig.camera_model

    h_ge_600_pitch_errs = []
    h_ge_600_roll_errs = []
    h_150_pitch_errs = []
    h_150_roll_errs = []

    for pf in pose_files:
        with open(pf, "r", encoding="utf-8") as f:
            pose_data = json.load(f)

        cam_id = pose_data["camera_id"]
        act = pose_data["aircraft_state"]
        h_m = float(act["h_m"])
        lat_deg = float(act["lat_deg"])
        lon_deg = float(act["lon_deg"])
        yaw_rad = math.radians(float(act["yaw_deg"]))
        p_true_deg = float(act["pitch_deg"])
        r_true_deg = float(act["roll_deg"])

        img_file = png_dir / (pf.stem.replace("_pose", "") + ".png")
        if not img_file.is_file():
            continue

        img = np.array(Image.open(img_file))
        edges = detect_camera_edges(img, cam_id, cam, prior_pose=None, step=8)
        assert len(edges) >= 10, f"Too few edges ({len(edges)}) on {img_file.name}"

        batch = edges_to_ray_batch({cam_id: edges}, rig)

        fit = ransac_fit_attitude(
            lat_deg, lon_deg, h_m, yaw_rad, batch,
            initial_guess_rad=(0.0, 0.0),
            max_iterations=40,
            inlier_threshold_rad=math.radians(0.3),
            rng_seed=20260925,
        )

        assert fit["status"] == "CONVERGED"
        p_err = abs(fit["pitch_deg"] - p_true_deg)
        r_err = abs(fit["roll_deg"] - r_true_deg)

        if h_m >= 600.0:
            h_ge_600_pitch_errs.append(p_err)
            h_ge_600_roll_errs.append(r_err)
        else:
            h_150_pitch_errs.append(p_err)
            h_150_roll_errs.append(r_err)

    # Verify counts: 96 scenes at h >= 600m, 32 scenes at h = 150m
    assert len(h_ge_600_pitch_errs) == 96
    assert len(h_150_pitch_errs) == 32

    # V2 Acceptance criteria for h >= 600m
    med_pitch_600 = float(np.median(h_ge_600_pitch_errs))
    max_pitch_600 = float(np.max(h_ge_600_pitch_errs))
    med_roll_600 = float(np.median(h_ge_600_roll_errs))
    max_roll_600 = float(np.max(h_ge_600_roll_errs))

    med_pitch_150 = float(np.median(h_150_pitch_errs))
    max_pitch_150 = float(np.max(h_150_pitch_errs))
    med_roll_150 = float(np.median(h_150_roll_errs))
    max_roll_150 = float(np.max(h_150_roll_errs))

    print(
        f"\nV2 Results on G3 Masks:\n"
        f"h >= 600m (96 scenes):\n"
        f"  |pitch error|: median = {med_pitch_600:.5f} deg (threshold: <= 0.02 deg), max = {max_pitch_600:.5f} deg (threshold: <= 0.10 deg)\n"
        f"  |roll error|:  median = {med_roll_600:.5f} deg (threshold: <= 0.05 deg), max = {max_roll_600:.5f} deg (threshold: <= 0.25 deg)\n"
        f"h = 150m (32 scenes, reported separately):\n"
        f"  |pitch error|: median = {med_pitch_150:.5f} deg, max = {max_pitch_150:.5f} deg\n"
        f"  |roll error|:  median = {med_roll_150:.5f} deg, max = {max_roll_150:.5f} deg\n"
    )

    # Assert V2 thresholds strictly
    assert med_pitch_600 <= 0.02, f"Pitch median {med_pitch_600:.4f} > 0.02"
    assert max_pitch_600 <= 0.10, f"Pitch max {max_pitch_600:.4f} > 0.10"
    assert med_roll_600 <= 0.05, f"Roll median {med_roll_600:.4f} > 0.05"
    assert max_roll_600 <= 0.25, f"Roll max {max_roll_600:.4f} > 0.25"
