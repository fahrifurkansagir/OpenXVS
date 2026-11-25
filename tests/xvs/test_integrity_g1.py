"""Verification tests V1_G1..V7_G1 for OpenXVS WP-07 Gate G1 (Batch B11).

Covers all 5 ROADMAP.md §4 WP-07 G1 Integrity Improvements:
- V1_G1: Multi-scale horizon edge recovery (`detect_camera_edges_multiscale`) & smooth
  pseudo-Huber IRLS solver (`solve_attitude_gn_g1`, `fit_pipeline_attitude_g1`) eliminating
  the G0 `status != CONVERGED` floor (`0` non-converged fault-free frames).
- V2_G1: Altitude-conditioned Monitor M1 thresholds (`lt_250m`, `250_1000m`,
  `ge_1000m`) and altitude-binned Mondrian conformal calibration (`fit_conformal_model_g1`,
  `compute_pl_deg_altitude_mondrian`).
- V3_G1: Boresight calibration residual sweep (`perturb_rig_boresight` across
  `sigma_cal in {0.0, 0.02, 0.05} deg`, static + thermal drift) with calibration-split
  protection-level uncertainty propagation (`calibrate_boresight_pl_sensitivity_g1`).
- V4_G1: Single-camera misalignment fault (`inject_single_cam_misalignment`) and
  leave-one-camera-out fault isolation & exclusion (`isolate_and_exclude_faulty_camera`,
  `F_isolated` mode).
- V5_G1: Classical horizon-strip image-quality OOD detector (`compute_horizon_strip_ood_features`,
  `compute_horizon_strip_ood_score`) eliminating `spurious_edge` and `glare` single-camera `F`
  HMI at `AL = 0.50 deg` (`0`) without violating `FAR 1%` or `FAR 0.1%`.
- V6_G1: Run manifest integrity (`code_hashes_match == True`, clean `git_status`, matching
  `results.json` SHA-256) and deterministic execution.
- V7_G1: AST verification of zero ground-truth leakage in runtime G1 algorithms and strict
  calibration/validation/test split isolation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest

from xvs.horizon_att.edges import detect_camera_edges, detect_camera_edges_multiscale, edges_to_ray_batch
from xvs.horizon_att.pipeline import load_pipeline_rig
from xvs.integrity.faults import (
    inject_glare,
    inject_single_cam_misalignment,
    inject_spurious_edge,
    perturb_rig_boresight,
)
from xvs.integrity.monitor import (
    ALTITUDE_BANDS,
    classify_altitude_band,
    compute_horizon_strip_ood_features,
    evaluate_monitor_alarm_g1,
    isolate_and_exclude_faulty_camera,
    tune_monitor_thresholds_g1,
)
from xvs.integrity.conformal import (
    compute_pl_deg_altitude_mondrian,
    fit_conformal_model_g1,
)
from xvs.integrity.g1_extensions import (
    calibrate_boresight_pl_sensitivity_g1,
    compute_horizon_strip_ood_score,
    fit_pipeline_attitude_g1,
    solve_attitude_gn_g1,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
G1_RESULTS_PATH = REPO_ROOT / "output" / "xvs" / "wp07" / "g1" / "results.json"
G1_MANIFEST_PATH = REPO_ROOT / "output" / "xvs" / "wp07" / "g1" / "run_manifest.json"


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# V1_G1: Multi-Scale Horizon Edge Recovery & Smooth Pseudo-Huber IRLS Solver
# =============================================================================


def test_v1_g1_multiscale_recovers_low_contrast_horizon_edges() -> None:
    """V1_G1 (unit): detect_camera_edges_multiscale recovers low-contrast horizon edges where single-scale fails."""
    rig = load_pipeline_rig()
    cam_f = rig.camera_model
    h_img, w_img = cam_f.height, cam_f.width
    rng = np.random.default_rng(20260701)

    yy, xx = np.mgrid[:h_img, :w_img]
    v_true = 512.0 + 0.03 * (xx - w_img / 2.0)
    img_low = np.clip(0.45 - 0.006 * np.tanh((yy - v_true) / 1.8) + rng.normal(0.0, 0.0012, size=(h_img, w_img)), 0.0, 1.0)

    edges_g0 = detect_camera_edges(img_low, "F", cam_f, step=8, min_gradient=0.015)
    edges_g1 = detect_camera_edges_multiscale(img_low, "F", cam_f, step=8, min_gradient=0.008)

    assert len(edges_g0) < 20, f"Expected single-scale detector to miss low-contrast horizon, got {len(edges_g0)}"
    assert len(edges_g1) >= 100, f"Expected multiscale detector to recover >= 100 edges, got {len(edges_g1)}"

    errs = [abs(e["v"] - (512.0 + 0.03 * (e["u"] - w_img / 2.0))) for e in edges_g1]
    rmse_px = float(np.sqrt(np.mean(np.square(errs))))
    assert rmse_px < 1.25, f"Expected sub-pixel/near-pixel accuracy on low-contrast horizon, got {rmse_px:.3f} px"


def test_v1_g1_non_converged_floor_eliminated_in_results() -> None:
    """V1_G1 (artifact): G1 eliminates the G0 non-converged fault-free floor and meets FAR 1.0% and 0.1% targets."""
    if not G1_RESULTS_PATH.is_file():
        pytest.skip("G1 results.json not yet generated")
    res = json.loads(G1_RESULTS_PATH.read_text(encoding="utf-8"))
    assert res["overall_gate_verdict"] == "PASS"

    item1 = res["item1_multiscale_and_solver"]
    assert item1["g0_non_converged_fault_free"]["test"] == 35
    assert item1["g1_non_converged_fault_free"]["calibration"] == 0
    assert item1["g1_non_converged_fault_free"]["validation"] == 0
    assert item1["g1_non_converged_fault_free"]["test"] == 0
    assert item1["t2_boundary_F_k298_299_g1_max_error_deg"] < 0.15

    far_01 = res["item2_altitude_conditioning"]["test_far"]["far_0.01"]
    far_001 = res["item2_altitude_conditioning"]["test_far"]["far_0.001"]
    assert far_001["overall"]["by_reason"]["STATUS_NOT_CONVERGED"] == 0
    assert res["item2_altitude_conditioning"]["validation_far"]["far_0.01"]["verdict"] == "PASS"
    assert res["item2_altitude_conditioning"]["validation_far"]["far_0.001"]["verdict"] == "PASS"
    assert far_01["verdict"] == "PASS"
    assert far_001["verdict"] == "PASS"
    assert far_001["overall"]["ci_low"] <= 0.0010
    assert far_001["overall"]["rate"] < res["g0_vs_g1_comparison"]["g0_test_far_0.001_rate"]


# =============================================================================
# V2_G1: Altitude-Conditioned Monitor M1 & Altitude-Binned Mondrian Conformal
# =============================================================================


def test_v2_g1_altitude_band_classification_and_conditioning() -> None:
    """V2_G1 (unit): classify_altitude_band and tune_monitor_thresholds_g1 partition all active sub-tests by altitude band."""
    assert classify_altitude_band(50.0) == "lt_250m"
    assert classify_altitude_band(249.9) == "lt_250m"
    assert classify_altitude_band(250.0) == "250_1000m"
    assert classify_altitude_band(999.9) == "250_1000m"
    assert classify_altitude_band(1000.0) == "ge_1000m"
    assert classify_altitude_band(3000.0) == "ge_1000m"

    rng = np.random.default_rng(42)
    synth_val: list[dict[str, float | str | int]] = []
    for _, h_val, loo_scale in [("lt_250m", 100.0, 0.40), ("250_1000m", 500.0, 0.22), ("ge_1000m", 2000.0, 0.08)]:
        for _ in range(200):
            synth_val.append({
                "status": "CONVERGED",
                "cam_set": "FRBL",
                "h_meas_m": h_val,
                "rms_residual_urad": float(rng.uniform(100.0, 500.0)),
                "condition_number": float(rng.uniform(1.05, 1.30)),
                "inlier_fraction": float(rng.uniform(0.80, 0.99)),
                "inlier_count": int(rng.integers(380, 456)),
                "camera_consistency_deg": float(rng.uniform(0.01, loo_scale)),
                "ood_horizon_score": 0.0,
            })

    th = tune_monitor_thresholds_g1(synth_val, target_far=0.01)
    c_cfg = th.by_cam_set["FRBL"]
    for key in (
        "max_rms_residual_by_altitude_urad",
        "max_condition_number_by_altitude",
        "min_inlier_fraction_by_altitude",
        "min_inlier_count_by_altitude",
        "max_ood_horizon_score_by_altitude",
        "max_leave_one_out_by_altitude_deg",
    ):
        assert key in c_cfg
        for b in ALTITUDE_BANDS:
            assert b in c_cfg[key]

    alt_map = c_cfg["max_leave_one_out_by_altitude_deg"]
    assert alt_map["ge_1000m"] < alt_map["250_1000m"] < alt_map["lt_250m"]

    rec_hi = dict(synth_val[-1], h_meas_m=2000.0, camera_consistency_deg=0.25)
    rec_lo = dict(synth_val[0], h_meas_m=100.0, camera_consistency_deg=0.25)
    assert evaluate_monitor_alarm_g1(rec_hi, th)["alarm"] is True
    assert evaluate_monitor_alarm_g1(rec_lo, th)["alarm"] is False


def test_v2_g1_altitude_mondrian_conformal_coverage() -> None:
    """V2_G1 (artifact): Altitude-binned Mondrian conformal model achieves valid test coverage across alphas."""
    if not G1_RESULTS_PATH.is_file():
        pytest.skip("G1 results.json not yet generated")
    res = json.loads(G1_RESULTS_PATH.read_text(encoding="utf-8"))
    cov = res["item2_altitude_conditioning"]["test_coverage_and_availability"]["far_0.01"]["coverage"]["altitude_mondrian"]
    for a_key in ("0.1", "0.01", "0.001"):
        assert cov[a_key]["verdict"] == "PASS", f"Coverage failed for alpha={a_key}: {cov[a_key]}"


# =============================================================================
# V3_G1: Boresight Calibration Residuals (`sigma_cal in {0.0, 0.02, 0.05} deg`)
# =============================================================================


def test_v3_g1_boresight_perturbation_orthonormality_and_determinism() -> None:
    """V3_G1 (unit): perturb_rig_boresight is deterministic and preserves SO(3) rotation matrices, and PL inflates with sigma_cal."""
    base_rig = load_pipeline_rig()
    rig_a, info_a = perturb_rig_boresight(base_rig, sigma_cal_deg=0.05, seed=12345, t_s=15.0)
    _, info_b = perturb_rig_boresight(base_rig, sigma_cal_deg=0.05, seed=12345, t_s=15.0)
    assert info_a == info_b

    for m_base, m_pert in zip(base_rig.cameras, rig_a.cameras):
        r_bc = m_pert.r_body_cam
        np.testing.assert_allclose(r_bc.T @ r_bc, np.eye(3), atol=1e-12)
        assert abs(float(np.linalg.det(r_bc)) - 1.0) < 1e-12
        diff_rad = math.acos(np.clip((float(np.trace(m_base.r_body_cam.T @ r_bc)) - 1.0) / 2.0, -1.0, 1.0))
        assert 0.0 < math.degrees(diff_rad) < 0.35

    # Verify PL inflation with sigma_cal_deg
    synth_cal = [
        {
            "status": "CONVERGED",
            "cam_set": "FRBL",
            "condition": "degraded",
            "fault_type": "none",
            "h_meas_m": 500.0,
            "total_error_deg": 0.01,
            "total_error_rad": math.radians(0.01),
            "sigma_hat_urad": 100.0,
            "nonconformity_score": math.radians(0.01) / 1e-4,
        }
        for _ in range(1200)
    ]
    g1_conf = fit_conformal_model_g1(synth_cal, alphas=(0.1, 0.01, 0.001))
    pl_0 = compute_pl_deg_altitude_mondrian(g1_conf, 1e-4, 0.01, "FRBL", "degraded", 500.0, sigma_cal_deg=0.0)
    pl_0_with_sens = compute_pl_deg_altitude_mondrian(
        g1_conf,
        1e-4,
        0.01,
        "FRBL",
        "degraded",
        500.0,
        sigma_cal_deg=0.0,
        boresight_sensitivity={"by_cam_set": {"FRBL": {"c_alpha": {"0.01": 4.5}}}},
    )
    assert pl_0_with_sens == pl_0
    pl_005 = compute_pl_deg_altitude_mondrian(g1_conf, 1e-4, 0.01, "FRBL", "degraded", 500.0, sigma_cal_deg=0.05)
    assert pl_005 > pl_0


def test_v3_g1_boresight_sweep_artifact_coverage_and_zero_hmi() -> None:
    """V3_G1 (artifact): Boresight sweep across {0.0, 0.02, 0.05} deg maintains coverage >= 0.990/0.999 and 0 HMI at AL=0.5 deg."""
    if not G1_RESULTS_PATH.is_file():
        pytest.skip("G1 results.json not yet generated")
    res = json.loads(G1_RESULTS_PATH.read_text(encoding="utf-8"))
    sweep = res["item3_boresight_calibration_sweep"]
    for key in ("sigma_cal_0.00deg", "sigma_cal_0.02deg", "sigma_cal_0.05deg"):
        assert key in sweep
        for cset in ("F", "FRBL", "all"):
            cn = sweep[key]["by_cam_set"][cset]
            assert cn["coverage_alpha_0.01"]["rate"] >= 0.990, f"{key} {cset} cov_0.01={cn['coverage_alpha_0.01']['rate']}"
            assert cn["coverage_alpha_0.001"]["rate"] >= 0.999, f"{key} {cset} cov_0.001={cn['coverage_alpha_0.001']['rate']}"
            assert cn["hmi_al_0.5"]["k"] == 0


# =============================================================================
# V4_G1: Single-Camera Misalignment Fault & Leave-One-Out Exclusion (`F_isolated`)
# =============================================================================


def test_v4_g1_single_cam_misalignment_isolation_and_exclusion() -> None:
    """V4_G1 (unit): isolate_and_exclude_faulty_camera identifies and excludes a 0.6 deg misaligned camera."""
    base_rig = load_pipeline_rig()
    rng = np.random.Generator(np.random.PCG64(20260704))
    rig_mis, log = inject_single_cam_misalignment(base_rig, cam_id="R", magnitude_deg=0.65, rng=rng)
    assert log["cam_id"] == "R"
    assert abs(log["magnitude_deg"] - 0.65) < 1e-9

    lat_deg, lon_deg, h_m, yaw_rad = 37.62, -122.38, 1200.0, 0.2
    pitch_true_deg, roll_true_deg = 1.0, -2.0
    from xvs.sim import horizon

    poses = base_rig.compute_poses(
        lat_deg, lon_deg, h_m, yaw_rad, math.radians(pitch_true_deg), math.radians(roll_true_deg)
    )

    edges_by_cam: dict[str, list[dict[str, float]]] = {}
    cam_model = base_rig.camera_model
    for c in ("F", "R", "B", "L"):
        p_cam = poses[c]["p_ecef_cam"]
        r_ec = poses[c]["r_ecef_cam"]
        c_edges: list[dict[str, float]] = []
        for u_col in range(40, cam_model.width - 40, 16):
            v_roots = horizon.analytic_horizon_v_at_u(p_cam, r_ec, cam_model, float(u_col))
            v_in = [v for v in v_roots if 20.0 <= v <= cam_model.height - 20.0]
            if len(v_in) == 1:
                c_edges.append({"u": float(u_col), "v": float(v_in[0]), "gradient": 0.25})
        edges_by_cam[c] = c_edges

    batch_mis = edges_to_ray_batch(edges_by_cam, rig_mis)
    fit_mis = solve_attitude_gn_g1(
        lat_deg, lon_deg, h_m, yaw_rad, 0.0, 0.0, batch_mis
    )
    iso = isolate_and_exclude_faulty_camera(
        full_fit=fit_mis,
        inlier_batch=batch_mis,
        rig=rig_mis,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        h_meas_m=h_m,
        yaw_rad=yaw_rad,
        d27_threshold_deg=0.10,
    )
    assert iso["isolated"] is True
    assert iso["excluded_camera"] == "R"
    refit = iso["refit"]
    assert refit is not None
    post_err = math.hypot(refit["pitch_deg"] - pitch_true_deg, refit["roll_deg"] - roll_true_deg)
    assert post_err < 0.02, f"Expected post-isolation error < 0.02 deg after excluding R, got {post_err:.5f} deg"


# =============================================================================
# V5_G1: Classical Horizon-Strip Image-Quality OOD Detector
# =============================================================================


def test_v5_g1_horizon_strip_ood_separates_spurious_edge_and_glare() -> None:
    """V5_G1 (unit & artifact): compute_horizon_strip_ood_features separates nominal vs spurious_edge and glare with 0 HMI."""
    rig = load_pipeline_rig()
    cam_f = rig.camera_model
    h_img, w_img = cam_f.height, cam_f.width
    yy, xx = np.mgrid[:h_img, :w_img]
    v_h = 512.0 + 0.02 * (xx - w_img / 2.0)
    img_nom = np.clip(120.0 - 55.0 * np.tanh((yy - v_h) / 3.0), 45.0, 195.0).astype(np.uint8)
    gt_uv = [[float(u), float(512.0 + 0.02 * (u - w_img / 2.0))] for u in range(0, w_img, 16)]

    ood_nom = compute_horizon_strip_ood_features(img_nom, cam=cam_f, cam_id="F")
    img_sp, _ = inject_spurious_edge(img_nom, horizon_uv=gt_uv, offset_px=18.0, width_fraction=0.75, cam_id="F")
    ood_sp = compute_horizon_strip_ood_features(img_sp, cam=cam_f, cam_id="F")
    img_gl, _ = inject_glare(img_nom, horizon_uv=gt_uv, radius_px=150.0, v_offset=10.0, cam_id="F")
    ood_gl = compute_horizon_strip_ood_features(img_gl, cam=cam_f, cam_id="F")

    assert ood_nom["ood_horizon_score"] == compute_horizon_strip_ood_score(img_nom, cam=cam_f, cam_id="F")
    assert ood_nom["ood_horizon_score"] < 0.50
    assert ood_sp["ood_horizon_score"] > 3.0
    assert ood_gl["ood_horizon_score"] > 1.8

    if G1_RESULTS_PATH.is_file():
        res = json.loads(G1_RESULTS_PATH.read_text(encoding="utf-8"))
        item5 = res["item5_ood_and_fault_hmi"]
        assert item5["tuned_on_splits"] == ["calibration"]
        assert item5["evaluated_on_split"] == "test"
        assert item5["spurious_edge_F_al_0.5_g0_hmi_k"] == 103
        assert item5["spurious_edge_F_al_0.5_g1_hmi_k"] == 0
        assert item5["f_spurious_edge_hmi_g1"] == 0
        assert item5["f_glare_hmi_g1"] == 0
        assert res["hmi"]["al_0.50"]["total_hmi"] == 0


# =============================================================================
# V6_G1: Run Manifest & Artifact Provenance Verification
# =============================================================================


def test_v6_g1_run_manifest_and_hashes() -> None:
    """V6_G1: Verify run_manifest.json has COMPLETED status, matching code hashes, and matching results SHA-256."""
    if not G1_MANIFEST_PATH.is_file() or not G1_RESULTS_PATH.is_file():
        pytest.skip("G1 run_manifest.json or results.json not yet generated")
    manifest = json.loads(G1_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["code_hashes_match"] is True
    assert manifest["git_status"] == []
    assert manifest["git_status_end"] == []
    expected_sha = manifest["outputs"]["output/xvs/wp07/g1/results.json"]
    actual_sha = _file_sha256(G1_RESULTS_PATH)
    assert actual_sha == expected_sha


# =============================================================================
# V7_G1: Zero Ground-Truth Leakage in Runtime G1 Functions
# =============================================================================


def test_v7_g1_no_ground_truth_leakage_in_runtime_functions() -> None:
    """V7_G1: AST inspection verifies runtime G1 functions never read ground-truth attitude or fault labels."""
    src_path = REPO_ROOT / "src" / "xvs" / "integrity" / "g1_extensions.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))

    runtime_funcs = {
        "detect_camera_edges_multiscale",
        "solve_attitude_gn_g1",
        "fit_pipeline_attitude_g1",
        "compute_horizon_strip_ood_features",
        "compute_horizon_strip_ood_score",
        "evaluate_monitor_alarm_g1",
        "isolate_and_exclude_faulty_camera",
    }
    forbidden_keys = {
        "pitch_true_deg",
        "roll_true_deg",
        "h_true_m",
        "fault_type",
        "fault_active",
        "target_cam",
    }

    found_funcs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in runtime_funcs:
            found_funcs.add(node.name)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value in forbidden_keys:
                    raise AssertionError(
                        f"Forbidden ground-truth key {sub.value!r} referenced in runtime function {node.name}"
                    )
    assert found_funcs == runtime_funcs
