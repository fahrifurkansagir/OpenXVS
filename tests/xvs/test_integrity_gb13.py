"""Acceptance and unit tests for G-B13 findings (F-B12-1 through F-B12-5).

Uses synthetic fixtures with fast execution (< 2 seconds total).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from xvs.integrity.conformal import (
    fit_conformal_model,
    ConformalModel,
    compute_conformal_quantile,
)
from xvs.integrity.monitor import (
    MonitorThresholds,
    compute_leave_one_out_consistency,
    evaluate_monitor_alarm,
    tune_monitor_thresholds,
    LeaveOneOutResult,
)
from xvs.horizon_att.limb_model import LimbRayBatch
from xvs.rig.rig import Rig


def test_gb13_f_b12_1_calibration_ignores_faulty_frames():
    """F-B12-1: Protection levels must calibrate strictly on fault_type == 'none' frames.

    Faulty frames (e.g. spurious_edge, occlusion) that pass unalarmed must NOT inflate
    the conformal quantile q(alpha).
    """
    rng = np.random.default_rng(20260927)

    # 100 degraded nominal frames with score ~ U[1.0, 2.0]
    records = []
    for _ in range(100):
        records.append({
            "split": "calibration",
            "status": "CONVERGED",
            "fault_type": "none",
            "condition": "degraded",
            "cam_set": "FRBL",
            "nonconformity_score": float(rng.uniform(1.0, 2.0)),
        })

    # Contaminate with 10 faulty unalarmed frames with massive score = 100.0
    for _ in range(10):
        records.append({
            "split": "calibration",
            "status": "CONVERGED",
            "fault_type": "spurious_edge",
            "condition": "degraded",
            "cam_set": "FRBL",
            "nonconformity_score": 100.0,
        })

    model = fit_conformal_model(records, alphas=[0.1, 0.01, 0.001])

    # Standard quantile at alpha=0.01 must reflect ONLY the nominal scores (<= 2.0)
    q_01 = model.standard_quantiles["0.01"]
    assert q_01 <= 2.0, f"q(0.01) was inflated by faulty frames: {q_01} > 2.0"
    assert model.n_cal_standard == 100, f"n_cal_standard should be 100, got {model.n_cal_standard}"


def test_gb13_f_b12_2_leave_one_out_requirements():
    """F-B12-2: Compute leave-one-out only when remaining cameras hold >= 2 cameras with >= 10 inliers.

    Otherwise the statistic is 'not available' (NaN), and monitor evaluation does not trigger LOO alarm.
    """
    rig = Rig.load_nominal()
    lat, lon, h = 37.62, -122.37, 2000.0
    yaw = 0.0

    # Case 1: Insufficient valid cameras (< 3 cameras with >= 10 inliers)
    fit_insufficient = {
        "pitch_deg": 2.0,
        "roll_deg": -1.0,
        "inlier_count_by_cam": {"F": 15, "R": 12, "B": 4, "L": 0, "U": 0, "D": 0},
    }
    # Only 2 cameras (F and R) have >= 10 inliers. Removing F leaves only R (1 camera < 2).
    res_nan = compute_leave_one_out_consistency(
        fit_insufficient,
        LimbRayBatch(np.zeros((0, 3)), np.zeros((0, 3)), []),
        rig, lat, lon, h, yaw,
    )
    assert math.isnan(res_nan.max_angle_deg), f"Expected NaN max_angle_deg, got {res_nan.max_angle_deg}"
    assert math.isnan(res_nan.consistency_statistic)

    # Monitor alarm should NOT trigger LOO_HIGH when LOO is NaN
    th = MonitorThresholds(far_target=0.01, max_leave_one_out_deg=0.15)
    rec_nan = {
        "status": "CONVERGED",
        "rms_residual_urad": 300.0,
        "condition_number": 100.0,
        "inlier_fraction": 0.5,
        "inlier_count": 27,
        "cam_set": "FRBL",
        "leave_one_out_consistency_deg": float("nan"),
    }
    eval_res = evaluate_monitor_alarm(rec_nan, th)
    assert not eval_res["alarm"], f"NaN LOO triggered unexpected alarm: {eval_res['reasons']}"
    assert not any("LOO_HIGH" in r for r in eval_res["reasons"])

    # High LOO should still trigger alarm
    rec_high = dict(rec_nan, leave_one_out_consistency_deg=0.50)
    eval_high = evaluate_monitor_alarm(rec_high, th)
    assert eval_high["alarm"]
    assert any("LOO_HIGH" in r for r in eval_high["reasons"])


def test_gb13_f_b12_3_tuning_raises_on_missing_features():
    """F-B12-3: Threshold tuning must raise ValueError on missing features; silent defaults eliminated."""
    # 1. Empty input records list
    with pytest.raises(ValueError, match="nominal_val_records is empty"):
        tune_monitor_thresholds([])

    # 2. No converged records
    with pytest.raises(ValueError, match="No CONVERGED records"):
        tune_monitor_thresholds([{"status": "DIVERGED"}])

    base_rec = {
        "status": "CONVERGED",
        "rms_residual_urad": 300.0,
        "condition_number": 100.0,
        "inlier_fraction": 0.5,
        "inlier_count": 25,
        "cam_set": "FRBL",
        "leave_one_out_consistency_deg": 0.05,
    }

    # 3. Missing each required feature must raise
    for missing_field, match_str in [
        ("rms_residual_urad", "rms_residual_urad"),
        ("condition_number", "condition_number"),
        ("inlier_fraction", "inlier_fraction"),
        ("inlier_count", "inlier_count"),
        ("leave_one_out_consistency_deg", "leave_one_out_consistency_deg"),
    ]:
        bad_rec = dict(base_rec)
        del bad_rec[missing_field]
        with pytest.raises(ValueError, match=match_str):
            tune_monitor_thresholds([bad_rec])

    # 4. Valid record succeeds
    th = tune_monitor_thresholds([base_rec])
    assert isinstance(th, MonitorThresholds)


def test_gb13_f_b12_4_condition_field_required():
    """F-B12-4: Conformal calibration and PL retrieval must use explicit condition field.
    
    Per D28: If k = ceil((n+1)(1-alpha)) > n, q(alpha) = +inf.
    With n=1 and alpha=0.1, k=2 > 1 => q = +inf.
    With n=1 and alpha=0.5, k=1 <= 1 => q is finite.
    """
    cal_records = [
        {"split": "calibration", "status": "CONVERGED", "fault_type": "none", "cam_set": "FRBL", "condition": "clean", "nonconformity_score": 1.25},
        {"split": "calibration", "status": "CONVERGED", "fault_type": "none", "cam_set": "FRBL", "condition": "degraded", "nonconformity_score": 3.75},
    ]

    model = fit_conformal_model(cal_records, alphas=[0.1, 0.5])
    assert "FRBL:clean" in model.mondrian_quantiles
    assert "FRBL:degraded" in model.mondrian_quantiles
    assert model.n_cal_mondrian["FRBL:clean"] == 1
    assert model.n_cal_mondrian["FRBL:degraded"] == 1

    # D28: alpha=0.1 with n=1 yields +inf (never sample max)
    assert math.isinf(model.get_quantile(0.1, cam_set="FRBL", condition="clean", mondrian=True))
    assert math.isinf(model.get_quantile(0.1, cam_set="FRBL", condition="degraded", mondrian=True))

    # Retrieval via condition parameter at alpha=0.5 where k=1 <= 1
    q_clean = model.get_quantile(0.5, cam_set="FRBL", condition="clean", mondrian=True)
    q_deg = model.get_quantile(0.5, cam_set="FRBL", condition="degraded", mondrian=True)

    assert q_clean == 1.25
    assert q_deg == 3.75


def test_gb13_f_b12_5_scale_comparison_and_sys_score():
    """F-B12-5: Report sigma_hat vs error and fit second score r = error / (sigma_hat + sigma_sys)."""
    from scripts.xvs.integrity.run_wp07_g0 import compute_scale_comparison
    from xvs.integrity.conformal import fit_sigma_sys

    records = [
        {
            "status": "CONVERGED",
            "fault_type": "none",
            "cam_set": "FRBL",
            "condition": "clean",
            "total_error_deg": 0.45,
            "total_error_rad": math.radians(0.45),
            "sigma_hat_urad": float(math.radians(0.05) * 1e6),  # 0.05 deg ~ 872 urad
            "nonconformity_score": 0.45 / 0.05,  # 9.0
        },
        {
            "status": "CONVERGED",
            "fault_type": "none",
            "cam_set": "FRBL",
            "condition": "degraded",
            "total_error_deg": 0.55,
            "total_error_rad": math.radians(0.55),
            "sigma_hat_urad": float(math.radians(0.05) * 1e6),
            "nonconformity_score": 0.55 / 0.05,  # 11.0
        },
    ]

    # 1. Scale comparison check
    stats = compute_scale_comparison(records)
    assert "FRBL" in stats
    s_frbl = stats["FRBL"]
    assert s_frbl["n"] == 2
    assert abs(s_frbl["mean_error_deg"] - 0.50) < 1e-4
    assert abs(s_frbl["mean_sigma_hat_deg"] - 0.05) < 1e-4
    assert abs(s_frbl["ratio_mean"] - 10.0) < 1e-4
    assert abs(s_frbl["fitted_sigma_sys_deg"] - 0.45) < 1e-4

    # 2. fit_sigma_sys check (calibrated across all nominal frames)
    s_sys = fit_sigma_sys(records)
    assert "FRBL" in s_sys
    assert abs(math.degrees(s_sys["FRBL"]) - 0.45) < 1e-4

    # 3. Model calibration with dual score (tested at alpha=0.5 for n=1 degraded sample)
    model = fit_conformal_model(records, alphas=[0.5])
    assert "0.5" in model.standard_quantiles
    assert "0.5" in model.standard_quantiles_sys

    # Standard score: 11.0 (from degraded frame)
    # Systematic score: err / (0.05 + 0.45) = err / 0.50 -> 0.55 / 0.50 = 1.10
    q_std = model.standard_quantiles["0.5"]
    q_sys = model.standard_quantiles_sys["0.5"]
    assert q_std > 8.0, f"Expected inflated standard quantile, got {q_std}"
    assert q_sys <= 1.5, f"Expected calibrated sys quantile near 1.1, got {q_sys}"

    # 4. PL calculation
    pl_std = model.compute_pl_deg(sigma_hat_rad=math.radians(0.05), alpha=0.5)
    pl_sys = model.compute_pl_deg_sys(sigma_hat_rad=math.radians(0.05), alpha=0.5, cam_set="FRBL")
    assert pl_std > 0.0
    assert pl_sys > 0.0
    # PLs should be roughly comparable because (0.05 * 11) ~ 0.55 and ((0.05 + 0.45) * 1.1) ~ 0.55
    assert abs(pl_std - pl_sys) < 0.2

    # 5. Serialization roundtrip
    d = model.to_dict()
    m2 = ConformalModel.from_dict(d)
    assert m2.standard_quantiles_sys == model.standard_quantiles_sys
    assert m2.sigma_sys_rad == model.sigma_sys_rad


def test_d28_quantile_small_sample_and_mondrian_isolation():
    """D28: Quantiles must be +inf when k > n, and clean frames must not enter pooled set."""
    # 1. compute_conformal_quantile small sample
    scores = [1.0, 2.0, 3.0]
    # alpha=0.01: k = ceil(4 * 0.99) = ceil(3.96) = 4 > 3 -> +inf
    q_small = compute_conformal_quantile(scores, 0.01)
    assert math.isinf(q_small)

    # 2. Clean frames excluded from standard pooled set
    cal_records = [
        {"split": "calibration", "status": "CONVERGED", "fault_type": "none", "cam_set": "FRBL", "condition": "clean", "nonconformity_score": 1.0}
        for _ in range(50)
    ]
    model = fit_conformal_model(cal_records, alphas=[0.1])
    assert model.n_cal_standard == 0
    # Standard quantile is inf or nan
    assert math.isinf(model.standard_quantiles.get("0.1", float("inf")))

    # 3. Missing Mondrian slice has no fallback to pooled
    # Requesting non-existent camera set / condition
    q_missing = model.get_quantile(0.1, cam_set="F", condition="clean", mondrian=True)
    assert math.isnan(q_missing)
