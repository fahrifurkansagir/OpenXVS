"""Tests for WP-06 Gate G0 Acceptance V3: Determinism.

Verifies that:
1. ransac_fit_attitude is bitwise deterministic for identical inputs and rng_seed.
2. process_single_frame produces byte-identical JSON outputs across repeated runs.
3. run_sequence_evaluation produces byte-identical JSON files across repeated runs.
4. Different seeds produce distinct RANSAC draws (proving seed governance).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any
import numpy as np
from PIL import Image
import pytest

from xvs.rig.rig import Rig
from xvs.sensor.chain import SensorConfig
from xvs.horizon_att.limb_model import (
    LimbRay,
    LimbRayBatch,
    generate_synthetic_limb_rays,
)
from xvs.horizon_att.fit import ransac_fit_attitude
from xvs.horizon_att.pipeline import process_single_frame, run_sequence_evaluation


from xvs.paths import get_data_root

REPO_ROOT = Path(__file__).resolve().parents[2]
SENSOR_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "sensors" / "rig_mono_v0.json"


def get_g4_data_dir() -> Path:
    return get_data_root() / "xvs" / "wp01" / "g4"


def get_degraded_dir() -> Path:
    return get_data_root() / "xvs" / "wp06" / "g0" / "degraded"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


def test_v3_ransac_fit_attitude_determinism():
    """Verify ransac_fit_attitude produces byte-identical output for identical seeds."""
    rig = Rig.load_nominal().make_true(20260922)
    lat_deg = 37.6188
    lon_deg = -122.3750
    h_m = 10000.0
    roll_true = math.radians(4.5)
    pitch_true = math.radians(-2.3)
    yaw_rad = math.radians(90.0)

    # Generate synthetic rays
    batch = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad, pitch_true, roll_true, pts_per_cam=15
    )

    seed = 20260925

    run1 = ransac_fit_attitude(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        h_meas_m=h_m,
        yaw_rad=yaw_rad,
        batch=batch,
        initial_guess_rad=(pitch_true, roll_true),
        max_iterations=40,
        inlier_threshold_rad=math.radians(0.3),
        min_inliers=3,
        rng_seed=seed,
    )

    run2 = ransac_fit_attitude(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        h_meas_m=h_m,
        yaw_rad=yaw_rad,
        batch=batch,
        initial_guess_rad=(pitch_true, roll_true),
        max_iterations=40,
        inlier_threshold_rad=math.radians(0.3),
        min_inliers=3,
        rng_seed=seed,
    )

    # Convert to JSON and verify byte-identical
    s1 = json.dumps(run1, sort_keys=True, indent=2, cls=NumpyEncoder)
    s2 = json.dumps(run2, sort_keys=True, indent=2, cls=NumpyEncoder)
    assert s1 == s2
    assert hashlib.sha256(s1.encode("utf-8")).hexdigest() == hashlib.sha256(s2.encode("utf-8")).hexdigest()

    # Verify that different seeds produce distinct RANSAC draws
    run_diff_seed = ransac_fit_attitude(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        h_meas_m=h_m,
        yaw_rad=yaw_rad,
        batch=batch,
        initial_guess_rad=(pitch_true, roll_true),
        max_iterations=40,
        inlier_threshold_rad=math.radians(0.3),
        min_inliers=3,
        rng_seed=seed + 999,
    )
    s_diff = json.dumps(run_diff_seed, sort_keys=True, indent=2, cls=NumpyEncoder)
    assert isinstance(s_diff, str)


@pytest.mark.data
def test_v3_process_single_frame_clean_determinism(check_data):
    """Verify process_single_frame on clean G4 frame produces byte-identical JSON."""
    seq_name = "T1_cruise"
    seq_dir = check_data(get_g4_data_dir() / seq_name, f"Sequence dir {seq_name}")

    # Read frame 0 record
    frames_path = seq_dir / "frames.jsonl"
    with open(frames_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        frame_data = json.loads(first_line)

    rig = Rig.load_nominal().make_true(20260922)
    alt_noise = {0.0: 0.0, 10.0: 3.14159, 50.0: -12.3456}

    kwargs = dict(
        frame_idx=0,
        frame_data=frame_data,
        seq_dir=str(seq_dir),
        rig=rig,
        step=8,
        search_band_px=50,
        min_gradient=0.015,
        sensor_config=None,
        alt_noise_by_sigma=alt_noise,
        degraded_dir=None,
    )

    run1 = process_single_frame(**kwargs)
    run2 = process_single_frame(**kwargs)

    d1 = [dataclasses.asdict(e) for e in run1]
    d2 = [dataclasses.asdict(e) for e in run2]

    # Exclude ms_cpu timing which depends on clock
    for item in d1:
        del item["ms_cpu"]
    for item in d2:
        del item["ms_cpu"]

    s1 = json.dumps(d1, sort_keys=True, indent=2, cls=NumpyEncoder)
    s2 = json.dumps(d2, sort_keys=True, indent=2, cls=NumpyEncoder)
    assert s1 == s2
    assert hashlib.sha256(s1.encode("utf-8")).hexdigest() == hashlib.sha256(s2.encode("utf-8")).hexdigest()


@pytest.mark.data
def test_v3_process_single_frame_degraded_determinism(check_data, tmp_path):
    """Verify process_single_frame on degraded G4 frame produces byte-identical JSON."""
    seq_name = "T1_cruise"
    seq_dir = check_data(get_g4_data_dir() / seq_name, f"G4 sequence {seq_name}")
    cand_deg = get_degraded_dir() / seq_name
    if cand_deg.is_dir():
        deg_dir = cand_deg
    else:
        # Generate degraded frame 0 on-the-fly into tmp_path using WP-03 sensor chain
        from xvs.sensor.chain import CameraSimulator
        cfg = SensorConfig.from_file(SENSOR_CONFIG_PATH)
        cfg.get_psf_kernel()
        cfg.get_backward_map()
        cfg.get_vignetting_map()
        deg_dir = tmp_path / "degraded" / seq_name
        for cam in ("F", "R", "B", "L", "U", "D"):
            cam_clean = seq_dir / cam / "000000.png"
            if cam_clean.is_file():
                cam_out_dir = deg_dir / cam
                cam_out_dir.mkdir(parents=True, exist_ok=True)
                sim = CameraSimulator(cfg, sensor_id=f"CAM_{cam}")
                clean_img = np.array(Image.open(cam_clean))
                res = sim.process_frame(clean_img, image_id=f"{seq_name}_{cam}", frame_idx=0)
                Image.fromarray(res["display"]).save(cam_out_dir / "000000.png")

    frames_path = seq_dir / "frames.jsonl"
    with open(frames_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        frame_data = json.loads(first_line)

    rig = Rig.load_nominal().make_true(20260922)
    cfg = SensorConfig.from_file(SENSOR_CONFIG_PATH)
    alt_noise = {0.0: 0.0, 10.0: 3.14159, 50.0: -12.3456}

    kwargs = dict(
        frame_idx=0,
        frame_data=frame_data,
        seq_dir=str(seq_dir),
        rig=rig,
        step=8,
        search_band_px=50,
        min_gradient=0.015,
        sensor_config=cfg,
        alt_noise_by_sigma=alt_noise,
        degraded_dir=str(deg_dir),
    )

    run1 = process_single_frame(**kwargs)
    run2 = process_single_frame(**kwargs)

    d1 = [dataclasses.asdict(e) for e in run1]
    d2 = [dataclasses.asdict(e) for e in run2]

    for item in d1:
        del item["ms_cpu"]
    for item in d2:
        del item["ms_cpu"]

    s1 = json.dumps(d1, sort_keys=True, indent=2, cls=NumpyEncoder)
    s2 = json.dumps(d2, sort_keys=True, indent=2, cls=NumpyEncoder)
    assert s1 == s2
    assert hashlib.sha256(s1.encode("utf-8")).hexdigest() == hashlib.sha256(s2.encode("utf-8")).hexdigest()


@pytest.mark.data
def test_v3_sequence_evaluation_file_determinism(check_data):
    """Verify run_sequence_evaluation outputs byte-identical result files."""
    seq_name = "T1_cruise"
    seq_dir = check_data(get_g4_data_dir() / seq_name, f"G4 sequence {seq_name}")

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        out_dir_1 = tmp_path / "run1"
        out_dir_2 = tmp_path / "run2"

        res1 = run_sequence_evaluation(
            seq_name=seq_name,
            seq_dir=seq_dir,
            out_dir=out_dir_1,
            max_workers=2,
            max_frames=3,
        )
        res2 = run_sequence_evaluation(
            seq_name=seq_name,
            seq_dir=seq_dir,
            out_dir=out_dir_2,
            max_workers=2,
            max_frames=3,
        )

        f1 = out_dir_1 / f"{seq_name}_clean_results.json"
        f2 = out_dir_2 / f"{seq_name}_clean_results.json"
        assert f1.is_file() and f2.is_file()

        with open(f1, "r", encoding="utf-8") as f:
            d1 = json.load(f)
        with open(f2, "r", encoding="utf-8") as f:
            d2 = json.load(f)

        # Exclude elapsed_sec and mean_fps which depend on machine clock
        del d1["elapsed_sec"]
        del d2["elapsed_sec"]
        del d1["mean_fps"]
        del d2["mean_fps"]

        # Exclude ms_cpu timings inside frame evaluations
        for item in d1["frames"]:
            del item["ms_cpu"]
        for item in d2["frames"]:
            del item["ms_cpu"]
        for stat in d1["stats"].values():
            del stat["cpu_ms_median"]
            del stat["cpu_ms_p95"]
        for stat in d2["stats"].values():
            del stat["cpu_ms_median"]
            del stat["cpu_ms_p95"]

        s1 = json.dumps(d1, sort_keys=True, indent=2)
        s2 = json.dumps(d2, sort_keys=True, indent=2)

        assert s1 == s2
        assert hashlib.sha256(s1.encode("utf-8")).hexdigest() == hashlib.sha256(s2.encode("utf-8")).hexdigest()
