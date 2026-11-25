"""Gate G0 Acceptance V2: Camera-consistency exactness (D27).

Conforms strictly to docs/xvs/WP07_INTEGRITY.md §2 and §5 (V2):
- On synthetic noise-free rays, every camera-alone fit equals the rest-fit within 1e-9 deg.
- With one camera given a known pitch bias b, the D27 statistic equals b within 1e-6 deg
  and names that camera.
- The dataset generator and the monitor call the same function; a test proves it.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from xvs.rig.rig import Rig
from xvs.horizon_att.limb_model import (
    generate_synthetic_limb_rays,
    solve_attitude_gn,
    LimbRayBatch,
)
from xvs.integrity.monitor import (
    camera_consistency,
    CameraConsistencyResult,
    compute_leave_one_out_consistency,
    LeaveOneOutResult,
)


def test_v2_noise_free_camera_consistency_exactness():
    """Verify that on synthetic noise-free rays, every camera-alone fit equals rest-fit within 1e-9 deg."""
    rig = Rig.load_nominal().make_true(20260922)
    lat_deg, lon_deg, h_m = 37.62, -122.37, 2500.0
    yaw_rad = math.radians(45.0)
    pitch_true_rad = math.radians(2.5)
    roll_true_rad = math.radians(-1.5)

    batch = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad, pitch_true_rad, roll_true_rad, pts_per_cam=25
    )
    assert batch.n_rays >= 60

    full_fit = solve_attitude_gn(
        lat_deg, lon_deg, h_m, yaw_rad,
        pitch_true_rad, roll_true_rad,
        batch, max_iter=25, tol_rad=1e-12,
    )
    assert full_fit["status"] == "CONVERGED"

    res = camera_consistency(
        full_fit, batch, rig, lat_deg, lon_deg, h_m, yaw_rad, max_iter=25, tol_rad=1e-12
    )

    assert isinstance(res, CameraConsistencyResult)
    assert len(res.diffs_deg) >= 4  # F, R, B, L

    # On synthetic noise-free rays, camera-alone fit equals rest-fit within <= 1e-9 deg
    for cam_id, diff_deg in res.diffs_deg.items():
        assert diff_deg <= 1e-9, f"Camera {cam_id} consistency difference {diff_deg:.2e} deg > 1e-9 deg"

    assert res.max_angle_deg <= 1e-9, f"Max angle {res.max_angle_deg:.2e} deg > 1e-9 deg"


@pytest.mark.parametrize("pitch_bias_deg", [0.01, 0.05, 0.10])
def test_v2_known_pitch_bias_exactness(pitch_bias_deg: float):
    """Verify that with one camera given a known pitch bias, D27 equals bias within 1e-6 deg and names that camera."""
    rig = Rig.load_nominal().make_true(20260922)
    lat_deg, lon_deg, h_m = 37.62, -122.37, 2500.0
    yaw_rad = math.radians(0.0)
    pitch_true_rad = math.radians(2.0)
    roll_true_rad = math.radians(-1.0)
    bias_rad = math.radians(pitch_bias_deg)

    # 1. Generate biased rays for Camera F
    batch_F = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad,
        pitch_true_rad + bias_rad, roll_true_rad,
        pts_per_cam=25,
    )
    mask_F = [cid == "F" for cid in batch_F.cam_ids]

    # 2. Generate unbiased rays for other cameras (R, B, L)
    batch_other = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad,
        pitch_true_rad, roll_true_rad,
        pts_per_cam=25,
    )
    mask_other = [cid != "F" for cid in batch_other.cam_ids]

    # 3. Assemble combined batch where only camera F has pitch bias
    d_combined = np.vstack([batch_F.d_body[mask_F], batch_other.d_body[mask_other]])
    lev_combined = np.vstack([batch_F.lever_arm_body[mask_F], batch_other.lever_arm_body[mask_other]])
    cams_combined = ["F"] * sum(mask_F) + [cid for cid in batch_other.cam_ids if cid != "F"]
    batch_combined = LimbRayBatch(d_combined, lev_combined, cams_combined)

    # 4. Full multi-camera fit
    full_fit = solve_attitude_gn(
        lat_deg, lon_deg, h_m, yaw_rad,
        pitch_true_rad, roll_true_rad,
        batch_combined, max_iter=25, tol_rad=1e-12,
    )
    assert full_fit["status"] == "CONVERGED"

    # 5. Monitor camera consistency evaluation
    res = camera_consistency(
        full_fit, batch_combined, rig, lat_deg, lon_deg, h_m, yaw_rad, max_iter=25, tol_rad=1e-12
    )

    # D27 consistency statistic must equal the known pitch bias within 1e-6 deg
    stat = res.consistency_statistic
    error_from_bias = abs(stat - pitch_bias_deg)

    assert error_from_bias <= 1e-6, (
        f"Monitor consistency statistic {stat:.8f} deg differs from known bias "
        f"{pitch_bias_deg:.8f} deg by {error_from_bias:.2e} deg (> 1e-6 deg)"
    )

    # The arg-max camera must be Camera F
    assert res.arg_max_camera == "F", f"Expected arg_max_camera 'F', got {res.arg_max_camera}"

    # The specific camera F discrepancy must also equal the bias within 1e-6 deg
    assert abs(res.cam_diffs_deg["F"] - pitch_bias_deg) <= 1e-6


def test_v2_generator_and_monitor_call_same_function():
    """Verify that dataset generator and monitor call the exact same camera_consistency implementation (D27)."""
    import scripts.xvs.integrity.make_wp07_data as gen_mod
    from xvs.integrity.monitor import camera_consistency as monitor_func
    assert gen_mod.camera_consistency is monitor_func, (
        f"Generator function {gen_mod.camera_consistency} is not monitor function {monitor_func}"
    )


def _synthetic_distorted_edges(rig: Rig, sensor_config, lat_deg: float, lon_deg: float, h_m: float, yaw_rad: float, pitch_rad: float, roll_rad: float) -> dict[str, list[dict[str, float]]]:
    """Project synthetic limb rays through forward radial distortion into pixel edge dicts."""
    batch_true = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad, pts_per_cam=25
    )
    k1 = float(sensor_config.k1)
    k2 = float(sensor_config.k2)
    fx = float(sensor_config.fx)
    fy = float(sensor_config.fy)
    cx = float(sensor_config.cx)
    cy = float(sensor_config.cy)
    edges_by_cam: dict[str, list[dict[str, float]]] = {c: [] for c in ("F", "R", "B", "L", "U", "D")}
    for d_b, cid in zip(batch_true.d_body, batch_true.cam_ids):
        r_bc = rig.cam_by_id[cid].r_body_cam
        d_c = r_bc.T @ d_b
        if d_c[2] <= 0:
            continue
        x_n = float(d_c[0] / d_c[2])
        y_n = float(d_c[1] / d_c[2])
        r2 = x_n * x_n + y_n * y_n
        scale = 1.0 + k1 * r2 + k2 * r2 * r2
        u_d = float(cx + fx * x_n * scale)
        v_d = float(cy + fy * y_n * scale)
        edges_by_cam[cid].append({"u": u_d, "v": v_d, "gradient": 0.25})
    return edges_by_cam


def test_v2_sensor_config_undistortion_regression():
    """Regression test: missing sensor_config in make_wp07_data / eval_w1_subset biases fits and D27 on distorted edges."""
    import scripts.xvs.integrity.make_wp07_data as gen_mod
    import scripts.xvs.integrity.eval_w1_subset as w1_mod
    from xvs.horizon_att.pipeline import load_pipeline_rig
    from xvs.sensor.chain import SensorConfig

    sensor_cfg_path = gen_mod.DEFAULT_SENSOR_CONFIG
    sensor_cfg = SensorConfig.from_file(sensor_cfg_path)
    rig = load_pipeline_rig()

    # Worker initialization must load rig_true (seed 20260922) and cache sensor_config
    gen_mod._init_worker(str(sensor_cfg_path))
    assert gen_mod._WORKER_SENSOR_CONFIG is not None
    assert not gen_mod._WORKER_RIG.is_nominal
    for cid in ("F", "R", "B", "L", "U", "D"):
        np.testing.assert_allclose(
            gen_mod._WORKER_RIG.cam_by_id[cid].r_body_cam,
            rig.cam_by_id[cid].r_body_cam,
            atol=1e-12,
        )

    lat_deg, lon_deg, h_m = 37.62, -122.37, 3000.0
    yaw_rad = math.radians(30.0)
    pitch_deg, roll_deg = 6.0, -12.0
    edges_distorted = _synthetic_distorted_edges(
        rig, sensor_cfg, lat_deg, lon_deg, h_m, yaw_rad,
        math.radians(pitch_deg), math.radians(roll_deg),
    )
    base_meta = {"seq_name": "T3_turn_climb", "split": "validation", "seed_idx": 11, "mode": "degraded", "condition": "degraded"}

    # Before fix (sensor_config=None): distortion causes large attitude error and D27 discrepancy
    recs_unfixed = gen_mod.evaluate_frame_fits(
        edges_distorted, rig, lat_deg, lon_deg, h_m, h_m, yaw_rad,
        pitch_deg, roll_deg, {0.0: 0.0, 10.0: 0.0, 50.0: 0.0},
        frame_k=5, base_record=base_meta, rng_seed=20260925, sensor_config=None,
    )
    frbl_unfixed = next(r for r in recs_unfixed if r["cam_set"] == "FRBL" and r["sigma_h"] == 0.0)
    assert frbl_unfixed["total_error_deg"] > 0.02
    assert frbl_unfixed["camera_consistency_deg"] > 0.05

    # After fix (sensor_config=sensor_cfg): undistortion recovers exact attitude and D27 near zero
    recs_fixed = gen_mod.evaluate_frame_fits(
        edges_distorted, rig, lat_deg, lon_deg, h_m, h_m, yaw_rad,
        pitch_deg, roll_deg, {0.0: 0.0, 10.0: 0.0, 50.0: 0.0},
        frame_k=5, base_record=base_meta, rng_seed=20260925, sensor_config=sensor_cfg,
    )
    frbl_fixed = next(r for r in recs_fixed if r["cam_set"] == "FRBL" and r["sigma_h"] == 0.0)
    assert frbl_fixed["total_error_deg"] < 1e-3, f"Expected < 1e-3 deg error, got {frbl_fixed['total_error_deg']}"
    assert frbl_fixed["camera_consistency_deg"] < 1e-3, f"Expected < 1e-3 deg D27, got {frbl_fixed['camera_consistency_deg']}"

    w1_recs = w1_mod.evaluate_frame_subset(
        edges_distorted, rig, lat_deg, lon_deg, h_m, h_m, yaw_rad,
        {0.0: 0.0, 10.0: 0.0, 50.0: 0.0}, frame_k=5, base_record=base_meta,
        rng_seed=20260925, sensor_config=sensor_cfg,
    )
    w1_frbl = next(r for r in w1_recs if r["cam_set"] == "FRBL" and r["sigma_h"] == 0.0)
    assert w1_frbl["stat3_d27"] < 1e-3


def test_v2_wp07_wp06_pipeline_parity(tmp_path):
    """Parity test: WP-07's full fit equals the WP-06 pipeline's fit across camera sets and sigma_h."""
    from PIL import Image
    import scripts.xvs.integrity.make_wp07_data as gen_mod
    import xvs.horizon_att.pipeline as pipe_mod
    from xvs.horizon_att.edges import detect_camera_edges
    from xvs.sensor.chain import SensorConfig

    # 1. Verify WP-07 imports the WP-06 pipeline's own functions directly
    assert gen_mod.load_pipeline_rig is pipe_mod.load_pipeline_rig
    assert gen_mod.build_pipeline_ray_batch is pipe_mod.build_pipeline_ray_batch
    assert gen_mod.fit_pipeline_attitude is pipe_mod.fit_pipeline_attitude

    rig = pipe_mod.load_pipeline_rig()
    sensor_cfg = SensorConfig.from_file(gen_mod.DEFAULT_SENSOR_CONFIG)

    # 2. Render synthetic step-horizon images for all 6 cameras at a non-zero frame index k=7
    k = 7
    h_px, w_px = rig.camera_model.height, rig.camera_model.width
    yy = np.arange(h_px, dtype=np.float64)[:, None]
    xx = np.arange(w_px, dtype=np.float64)[None, :]
    for idx, c in enumerate(("F", "R", "B", "L", "U", "D")):
        cam_dir = tmp_path / c
        cam_dir.mkdir(parents=True, exist_ok=True)
        v_h = 0.48 * h_px + 0.03 * (xx - 0.5 * w_px) + (idx - 2) * 3.0
        img = np.where(yy < v_h, 210, 45).astype(np.uint8)
        Image.fromarray(img).save(cam_dir / f"{k:06d}.png")

    frame_data = {
        "k": k,
        "t_s": 3.5,
        "aircraft": {
            "lat_deg": 37.62,
            "lon_deg": -122.37,
            "h_m": 2500.0,
            "ypr_deg": [45.0, 2.0, -1.5],
        },
    }
    alt_noise = {0.0: 0.0, 10.0: 4.25, 50.0: -18.5}

    # Run WP-06 process_single_frame (with sensor_config)
    wp06_evals = pipe_mod.process_single_frame(
        frame_idx=0,
        frame_data=frame_data,
        seq_dir=tmp_path,
        rig=rig,
        sensor_config=sensor_cfg,
        alt_noise_by_sigma=alt_noise,
        degraded_dir=tmp_path,
    )
    wp06_by_key = {(ev.cam_set, ev.sigma_h): ev for ev in wp06_evals}

    # Run WP-07 evaluate_frame_fits on the same detected edges
    edges_by_cam = {
        c: detect_camera_edges(
            np.array(Image.open(tmp_path / c / f"{k:06d}.png")),
            c, rig.camera_model, step=8, search_band_px=50, min_gradient=0.015,
        )
        for c in ("F", "R", "B", "L", "U", "D")
    }
    wp07_recs = gen_mod.evaluate_frame_fits(
        edges_by_cam=edges_by_cam,
        rig=rig,
        lat_deg=37.62,
        lon_deg=-122.37,
        h_true_m=2500.0,
        h_meas_base_m=2500.0,
        yaw_rad=math.radians(45.0),
        pitch_true_deg=2.0,
        roll_true_deg=-1.5,
        alt_noise_dict=alt_noise,
        frame_k=k,
        base_record={"seq_name": "T1_cruise", "split": "validation", "seed_idx": 11, "mode": "degraded", "condition": "degraded"},
        rng_seed=20260925,
        sensor_config=sensor_cfg,
    )
    cam_map = {"F": "F", "FRBL": "horizontal", "all": "all"}
    for rec in wp07_recs:
        ev = wp06_by_key[(cam_map[rec["cam_set"]], rec["sigma_h"])]
        assert rec["status"] == ev.status
        assert rec["inlier_count"] == ev.inlier_count
        assert rec["inlier_counts_by_cam"] == ev.inlier_counts_by_cam
        assert rec["pitch_est_deg"] == pytest.approx(ev.pitch_est_deg, abs=1e-12)
        assert rec["roll_est_deg"] == pytest.approx(ev.roll_est_deg, abs=1e-12)
        assert rec["pitch_error_deg"] == pytest.approx(ev.pitch_error_deg, abs=1e-12)
        assert rec["roll_error_deg"] == pytest.approx(ev.roll_error_deg, abs=1e-12)
        assert rec["total_error_deg"] == pytest.approx(ev.total_error_deg, abs=1e-12)
        assert rec["rms_residual_urad"] == pytest.approx(ev.rms_residual_urad, abs=1e-12)


def test_v2_camera_consistency_rejects_raw_edge_dict():
    """D27: camera_consistency accepts LimbRayBatch only and rejects raw edge dicts."""
    rig = Rig.load_nominal().make_true(20260922)
    full_fit = {"pitch_deg": 2.0, "roll_deg": -1.5, "status": "CONVERGED"}
    with pytest.raises(TypeError, match="LimbRayBatch"):
        camera_consistency(full_fit, {"F": [{"u": 100.0, "v": 200.0}]}, rig, 37.62, -122.37, 2500.0, 0.0)

