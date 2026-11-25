"""Gate G0 Acceptance V4: Determinism.

Conforms strictly to docs/xvs/WP07_INTEGRITY.md §5 (V4):
- Evaluator functions, conformal fitting, and metrics reporting produce byte-identical JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import pytest

from xvs.rig.rig import Rig
from xvs.horizon_att.limb_model import (
    generate_synthetic_limb_rays,
    solve_attitude_gn,
)
from xvs.integrity.monitor import (
    MonitorThresholds,
    compute_leave_one_out_consistency,
    evaluate_monitor_alarm,
    tune_monitor_thresholds,
)
from xvs.integrity.conformal import (
    fit_conformal_model,
    ConformalModel,
)
from scripts.xvs.integrity.run_wp07_g0 import evaluate_test_metrics


def test_v4_leave_one_out_determinism():
    """Verify compute_leave_one_out_consistency produces identical numbers across repeated runs."""
    rig = Rig.load_nominal().make_true(20260922)
    lat_deg, lon_deg, h_m = 37.62, -122.37, 2000.0
    yaw_rad = math.radians(10.0)
    pitch_rad = math.radians(2.0)
    roll_rad = math.radians(-1.0)

    batch = generate_synthetic_limb_rays(
        rig, lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad, pts_per_cam=15
    )
    full_fit = solve_attitude_gn(lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad, batch)

    res1 = compute_leave_one_out_consistency(full_fit, batch, rig, lat_deg, lon_deg, h_m, yaw_rad)
    res2 = compute_leave_one_out_consistency(full_fit, batch, rig, lat_deg, lon_deg, h_m, yaw_rad)

    json1 = json.dumps({
        "max_angle_deg": res1.max_angle_deg,
        "diffs_deg": res1.diffs_deg,
        "max_cam_diff_deg": res1.max_cam_diff_deg,
        "cam_diffs_deg": res1.cam_diffs_deg,
    }, sort_keys=True)

    json2 = json.dumps({
        "max_angle_deg": res2.max_angle_deg,
        "diffs_deg": res2.diffs_deg,
        "max_cam_diff_deg": res2.max_cam_diff_deg,
        "cam_diffs_deg": res2.cam_diffs_deg,
    }, sort_keys=True)

    h1 = hashlib.sha256(json1.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(json2.encode("utf-8")).hexdigest()

    assert h1 == h2, f"Leave-one-out JSON sha256 mismatch: {h1} != {h2}"


def test_v4_monitor_alarm_determinism():
    """Verify monitor alarm evaluation produces byte-identical results."""
    th = MonitorThresholds(far_target=0.01)
    record = {
        "status": "CONVERGED",
        "rms_residual_urad": 350.0,
        "condition_number": 420.0,
        "inlier_fraction": 0.85,
        "inlier_count": 18,
        "cam_set": "FRBL",
        "leave_one_out_consistency_deg": 0.045,
    }

    res1 = evaluate_monitor_alarm(record, th)
    res2 = evaluate_monitor_alarm(record, th)

    j1 = json.dumps(res1, sort_keys=True)
    j2 = json.dumps(res2, sort_keys=True)

    assert hashlib.sha256(j1.encode()).hexdigest() == hashlib.sha256(j2.encode()).hexdigest()


def test_v4_conformal_model_determinism():
    """Verify conformal calibration produces byte-identical model JSON."""
    rng = np.random.default_rng(20260927)
    records = []
    for i in range(100):
        records.append({
            "split": "calibration",
            "alarm": False,
            "status": "CONVERGED",
            "nonconformity_score": float(rng.uniform(0.5, 4.0)),
            "cam_set": "FRBL" if i % 2 == 0 else "all",
            "mode": "degraded" if i % 3 == 0 else "clean",
        })

    m1 = fit_conformal_model(records)
    m2 = fit_conformal_model(records)

    j1 = json.dumps(m1.to_dict(), sort_keys=True)
    j2 = json.dumps(m2.to_dict(), sort_keys=True)

    assert hashlib.sha256(j1.encode()).hexdigest() == hashlib.sha256(j2.encode()).hexdigest()


def test_v4_test_metrics_byte_identical_json():
    """Verify test metrics evaluation runner produces byte-identical JSON outputs."""
    th = MonitorThresholds(far_target=0.01)
    conf = ConformalModel(
        standard_quantiles={"0.1": 2.5, "0.01": 3.8, "0.001": 4.5},
        mondrian_quantiles={},
        n_cal_standard=100,
        n_cal_mondrian={},
    )

    rng = np.random.default_rng(12345)
    test_recs = []
    for i in range(50):
        test_recs.append({
            "seq_name": "T1_cruise",
            "split": "test",
            "seed_idx": 16,
            "mode": "degraded",
            "condition": "degraded",
            "k": i,
            "t_s": float(i * 0.1),
            "h_true_m": 2000.0,
            "pitch_true_deg": 2.0,
            "roll_true_deg": -1.0,
            "fault_type": "none" if i < 25 else "occlusion",
            "fault_active": (i >= 30),
            "k_onset": 30 if i >= 25 else None,
            "target_cam": "F" if i >= 25 else None,
            "cam_set": "FRBL",
            "sigma_h": 10.0,
            "status": "CONVERGED",
            "rms_residual_urad": float(rng.uniform(100.0, 500.0)),
            "condition_number": float(rng.uniform(10.0, 500.0)),
            "inlier_fraction": float(rng.uniform(0.6, 0.95)),
            "inlier_count": int(rng.integers(10, 30)),
            "leave_one_out_consistency_deg": float(rng.uniform(0.01, 0.08)),
            "total_error_deg": float(rng.uniform(0.01, 0.25)),
            "sigma_hat_urad": 150.0,
        })

    m1 = evaluate_test_metrics(test_recs, th, conf)
    m2 = evaluate_test_metrics(test_recs, th, conf)

    j1 = json.dumps(m1, sort_keys=True)
    j2 = json.dumps(m2, sort_keys=True)

    h1 = hashlib.sha256(j1.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(j2.encode("utf-8")).hexdigest()

    assert h1 == h2, f"Metrics evaluation SHA-256 mismatch:\n{h1}\n!=\n{h2}"
