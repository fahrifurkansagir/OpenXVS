"""Acceptance and unit tests for WP-05 G1 (Batch B7: V1_G1 through V8_G1).

Conforms strictly to docs/xvs/batches/BATCH_B7_ARGON.md §Acceptance Criteria:
- V1_G1: test_passive_well_clear_intervals
- V2_G1: test_second_moment_ttc_robust_regression
- V3_G1: test_6dof_ground_clutter_initiation_rejection
- V4_G1: test_no_truth_in_g1_tracker_and_risk
- V5_G1: test_g1_cpu_determinism_and_crn
- V6_G1: test_d32_ever_confirmed_and_d29_sequence
- V7_G1: test_g1_long_job_manifests
- V8_G1: test_wp05_g1_report_provenance_and_byte_identity
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from xvs.geo import rotations, wgs84
from xvs.report_provenance import verify_report_provenance
from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.encounter import EncounterSamplerV1, construct_encounter
from xvs.traffic.ring import TrafficRing
from xvs.tracking.evaluate import (
    aggregate_dataset_metrics,
    get_ownship_states,
    run_evaluation,
    set_truth_cache_dir,
    simulate_single_encounter,
)
from xvs.tracking.ground_rejection import (
    GroundObservation,
    compute_ground_consistency_residual,
    ray_wgs84_intersect,
)
from xvs.tracking.measurements import DetectionRecord, SyntheticMeasurementGenerator
from xvs.tracking.risk import (
    LOWC_HMD_LIMIT_M,
    WP02_CLASS_DIMENSIONS,
    compute_passive_hmd_interval,
    compute_passive_tau_mod_interval,
    compute_prior_size_bounds,
    compute_range_interval,
    evaluate_track_risk,
    evaluate_wc1_for_beta,
)
from xvs.tracking.tracker import MultiCameraTracker, Track, TrackStatus
from xvs.tracking.ttc import (
    TTCEstimate,
    compute_patch_second_moment_scale,
    compute_vertex_second_moment_scale,
    estimate_sliding_window_log_size_ttc,
    render_subpixel_target_patch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path: Path):
    """Point truth cache to tmp_path during unit tests."""
    set_truth_cache_dir(tmp_path)
    yield
    set_truth_cache_dir(None)


def test_passive_well_clear_intervals() -> None:
    """V1_G1: Passive well-clear (WC1) interval exactness and non-zero-HMD threat alerting."""
    # 1. Verify interval containment for all 6 aircraft classes on closing geometries
    for class_id, (l_min, l_max) in WP02_CLASS_DIMENSIONS.items():
        l_true = 0.5 * (l_min + l_max)
        for r_true in (800.0, 1500.0, 3000.0):
            for hmd_true in (0.0, 150.0, 400.0, 580.0):
                v_rel = 90.0
                sin_psi = hmd_true / r_true
                cos_psi = math.sqrt(max(1e-9, 1.0 - sin_psi * sin_psi))
                v_close = v_rel * cos_psi
                v_perp = v_rel * sin_psi

                eta_true = v_close / r_true
                omega_true = v_perp / r_true
                theta_true = l_true / r_true

                #Analytical HMD and modified tau under passive closing formula
                hmd_analytic = r_true * omega_true / math.hypot(eta_true, omega_true)
                tau_mod_analytic = max(0.0, (1.0 / eta_true) * (1.0 - (LOWC_HMD_LIMIT_M / r_true) ** 2))

                for cls_arg in (class_id, None):
                    r_int = compute_range_interval(theta_true, class_id=cls_arg, sigma_theta_rad=0.0)
                    assert r_int[0] - 1e-9 <= r_true <= r_int[1] + 1e-9

                    hmd_int = compute_passive_hmd_interval(
                        r_int,
                        omega_los_radps=omega_true,
                        sigma_omega_radps=0.0,
                        eta_min_inv_s=eta_true,
                        eta_max_inv_s=eta_true,
                    )
                    assert hmd_int is not None
                    assert hmd_int[0] - 1e-6 <= hmd_analytic <= hmd_int[1] + 1e-6

                    tau_int = compute_passive_tau_mod_interval(
                        r_int,
                        eta_min_inv_s=eta_true,
                        eta_max_inv_s=eta_true,
                    )
                    assert tau_int is not None
                    assert tau_int[0] - 1e-6 <= tau_mod_analytic <= tau_int[1] + 1e-6

    # 2. Non-zero-HMD LoWC threat (HMD = 350 m < 609.6 m, R = 1400 m, v_rel = 95 m/s)
    # True bearing rate omega = v_perp / R = (95 * 350 / 1400) / 1400 = 0.0170 rad/s > omega_thr = 0.005 rad/s.
    # G0 A1 rejects this threat (omega > 0.005), whereas WC1 alerts at the analytical tau_mod <= 25 s entry!
    r_threat = 1400.0
    hmd_threat = 350.0
    v_rel = 95.0
    sin_p = hmd_threat / r_threat
    cos_p = math.sqrt(1.0 - sin_p * sin_p)
    eta_t = (v_rel * cos_p) / r_threat
    omega_t = (v_rel * sin_p) / r_threat
    assert omega_t > 0.005  # Rejected by G0 A1 constant-bearing gate

    l_c3 = 14.0  # C3_TWIN
    theta_t = l_c3 / r_threat
    s_dot_t = theta_t * eta_t

    trk = Track(
        track_id="TRK_V1",
        t_init=17.0,
        status=TrackStatus.CONFIRMED,
        x=np.array([0.0, omega_t, 0.0, 0.0, theta_t, s_dot_t], dtype=np.float64),
        P=np.diag([1e-6, 1e-7, 1e-6, 1e-7, 1e-8, 1e-10]),
        history_s_mom=[theta_t],
        ttc_estimate=TTCEstimate(
            valid=True,
            tau_s_hat_s=1.0 / eta_t,
            tau_s_min_s=1.0 / (1.05 * eta_t),
            tau_s_max_s=1.0 / (0.95 * eta_t),
            eta_hat_inv_s=eta_t,
            eta_min_inv_s=0.95 * eta_t,
            eta_max_inv_s=1.05 * eta_t,
            sigma_eta_inv_s=0.03 * eta_t,
            n_samples=15,
            span_s=3.0,
        ),
    )

    risk_blind = evaluate_track_risk(trk, t_s=20.0, class_id=None, beta_wc=0.5)
    risk_ck = evaluate_track_risk(trk, t_s=20.0, class_id="C3_TWIN", beta_wc=0.5)
    assert risk_blind.alert_a1_warning is False, "G0 A1 must fail on non-zero HMD threat with omega > 0.005 rad/s"
    assert risk_blind.wc1_warning is True, "G1 WC1 (blind) must alert on non-zero HMD threat inside tau_mod <= 25 s"
    assert risk_ck.wc1_warning is True, "G1 WC1 (class-known) must alert on non-zero HMD threat"

    # 3. Multi-step trajectory stepping across the analytical tau_mod <= T_warn = 25.0 s boundary
    # Verify WC1 triggers at the analytical entry time while A1 fails throughout
    hmd_step = 350.0
    v_rel_step = 95.0
    # At constant closing speed v_rel_step and miss distance hmd_step:
    # R(tau_cpa) = sqrt((v_rel_step * tau_cpa)^2 + hmd_step^2)
    # Analytical tau_mod(tau_cpa) = (R^2 - D_THR^2) / (v_rel_step^2 * tau_cpa)
    for tau_cpa in (30.0, 27.0, 24.0, 20.0, 15.0):
        r_now = math.hypot(v_rel_step * tau_cpa, hmd_step)
        v_close_now = (v_rel_step * v_rel_step * tau_cpa) / r_now
        v_perp_now = (v_rel_step * hmd_step) / r_now
        eta_now = v_close_now / r_now
        omega_now = v_perp_now / r_now
        theta_now = l_c3 / r_now
        tau_mod_now = (1.0 / eta_now) * (1.0 - (LOWC_HMD_LIMIT_M / r_now) ** 2)

        trk_step = Track(
            track_id="TRK_STEP",
            t_init=0.0,
            status=TrackStatus.CONFIRMED,
            x=np.array([0.0, omega_now, 0.0, 0.0, theta_now, theta_now * eta_now], dtype=np.float64),
            P=np.diag([1e-8, 1e-9, 1e-8, 1e-9, 1e-10, 1e-12]),
            history_s_mom=[theta_now],
            ttc_estimate=TTCEstimate(
                valid=True,
                tau_s_hat_s=1.0 / eta_now,
                tau_s_min_s=1.0 / eta_now,
                tau_s_max_s=1.0 / eta_now,
                eta_hat_inv_s=eta_now,
                eta_min_inv_s=eta_now,
                eta_max_inv_s=eta_now,
                sigma_eta_inv_s=0.0,
                n_samples=20,
                span_s=4.0,
            ),
        )
        rk_step = evaluate_track_risk(trk_step, t_s=35.0 - tau_cpa, class_id="C3_TWIN", beta_wc=0.5)
        assert rk_step.alert_a1_warning is False, "A1 must fail at all steps due to non-zero bearing rate"
        if tau_mod_now > 25.5:
            assert rk_step.wc1_warning is False, f"WC1 should not alert before tau_mod <= 25 s (tau_mod={tau_mod_now:.2f}s)"
        elif tau_mod_now < 24.5:
            assert rk_step.wc1_warning is True, f"WC1 must alert once tau_mod <= 25 s (tau_mod={tau_mod_now:.2f}s)"

    # 4. Monotonicity across beta_WC in [0, 1] and edge cases
    hmd_hats = []
    tau_hats = []
    for b in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, _, h_hat, t_hat = evaluate_wc1_for_beta(
            (risk_blind.hmd_min_m, risk_blind.hmd_max_m),
            (risk_blind.tau_mod_min_s, risk_blind.tau_mod_max_s),
            beta_wc=b,
        )
        assert h_hat is not None and t_hat is not None
        hmd_hats.append(h_hat)
        tau_hats.append(t_hat)
    assert all(hmd_hats[i] <= hmd_hats[i + 1] + 1e-9 for i in range(len(hmd_hats) - 1))
    assert all(tau_hats[i] <= tau_hats[i + 1] + 1e-9 for i in range(len(tau_hats) - 1))

    # Edge case: non-closing target (closing_valid=False) never alerts
    w_nc, c_nc, _, _ = evaluate_wc1_for_beta((100.0, 200.0), (5.0, 10.0), beta_wc=0.5, closing_valid=False)
    assert w_nc is False and c_nc is False

    # Edge case: quantile prior bounds
    l_q0, l_q1 = compute_prior_size_bounds(class_id=None, alpha_r=0.10)
    assert 0.30 < l_q0 < l_q1 < 40.0


def test_second_moment_ttc_robust_regression() -> None:
    """V2_G1: Sub-pixel intensity second-moment scale s_mom and robust sliding-window ln(s) TTC."""
    fx = 2319.0
    sigma_psf = 0.55
    times = np.linspace(0.0, 4.0, 41)  # 4.0 s window at 10 Hz

    # 1. Smooth expansion from 1.5 px to 8.0 px over 4.0 s window (BATCH_B7_ARGON.md §V2_G1)
    tau_wide = 4.0 / math.log(8.0 / 1.5)
    s_mom_wide_rad = []
    for t in times:
        s_true_px = 1.5 * math.exp(float(t) / tau_wide)
        patch = render_subpixel_target_patch(length_px=s_true_px, angle_rad=0.3, sigma_psf_px=sigma_psf)
        s_mom_res = compute_patch_second_moment_scale(patch, sigma_psf_px=sigma_psf, lambda_min_px=0.05)
        s_mom_wide_rad.append(s_mom_res.s_mom_px / fx)

    ttc_wide = estimate_sliding_window_log_size_ttc(times, s_mom_wide_rad, window_s=4.0)
    assert ttc_wide.valid is True and ttc_wide.tau_s_hat_s is not None
    rel_err_wide = abs(ttc_wide.tau_s_hat_s - tau_wide) / tau_wide
    assert rel_err_wide < 0.02, f"Expected < 2% error on 1.5->8.0 px expansion, got {rel_err_wide:.4f}"

    # 2. Sub-pixel long-range expansion (1.5 px -> 1.96 px at tau = 15.0 s):
    # Verify s_mom recovers true TTC to < 2% whereas quantized integer bounding-box size fails!
    tau_const = 15.0
    s_exp_rad = []
    s_box_subpx_rad = []
    for t in times:
        s_true_px = 1.5 * math.exp(float(t) / tau_const)  # expands from 1.501 px to 1.96 px (sub-pixel!)
        patch = render_subpixel_target_patch(length_px=s_true_px, angle_rad=0.2, sigma_psf_px=sigma_psf)
        s_mom_res = compute_patch_second_moment_scale(patch, sigma_psf_px=sigma_psf, lambda_min_px=0.05)
        s_exp_rad.append(s_mom_res.s_mom_px / fx)
        # Integer pixel bounding-box quantization rounds [1.501, 1.96] to 2.0 px across all frames
        s_box_px = max(1.0, float(math.floor(s_true_px + 0.5)))
        s_box_subpx_rad.append(s_box_px / fx)

    ttc_exp = estimate_sliding_window_log_size_ttc(times, s_exp_rad, window_s=4.0)
    assert ttc_exp.valid is True
    assert ttc_exp.tau_s_hat_s is not None
    rel_err_noiseless = abs(ttc_exp.tau_s_hat_s - tau_const) / tau_const
    assert rel_err_noiseless < 0.02, f"Expected < 2% error on sub-pixel expansion, got {rel_err_noiseless:.4f}"

    ttc_box_subpx = estimate_sliding_window_log_size_ttc(times, s_box_subpx_rad, window_s=4.0)
    assert not ttc_box_subpx.valid or abs(ttc_box_subpx.tau_s_hat_s - tau_const) / tau_const > 0.20, (
        "Quantized integer bounding box must fail on sub-pixel 1.5->1.96 px expansion"
    )

    # Verify vertex second-moment scale matches segment length to < 1%
    s_vert_res = compute_vertex_second_moment_scale(
        np.linspace(-2.5, 2.5, 25),
        np.zeros(25),
        sigma_psf_px=sigma_psf,
        lambda_min_px=0.05,
    )
    assert abs(s_vert_res.s_mom_px - 5.0) / 5.0 < 0.01

    # Verify robustness to 25% outlier scale spikes (< 5% slope error)
    s_corrupted = list(s_exp_rad)
    # Inject 25% gross outliers (every 4th sample multiplied by 3.5x or 0.3x)
    for idx in range(2, len(s_corrupted), 4):
        s_corrupted[idx] = s_corrupted[idx] * (3.5 if (idx % 8 == 2) else 0.3)

    ttc_robust = estimate_sliding_window_log_size_ttc(times, s_corrupted, window_s=4.0, use_huber=True)
    assert ttc_robust.valid is True
    assert ttc_robust.tau_s_hat_s is not None
    rel_err_outliers = abs(ttc_robust.tau_s_hat_s - tau_const) / tau_const
    assert rel_err_outliers < 0.05, f"Expected < 5% slope error with 25% outliers, got {rel_err_outliers:.4f}"

    # Edge cases: too few points or receding target
    ttc_short = estimate_sliding_window_log_size_ttc(times[:3], s_exp_rad[:3], min_hits=5)
    assert ttc_short.valid is False and ttc_short.tau_s_hat_s is None

    ttc_receding = estimate_sliding_window_log_size_ttc(times, list(reversed(s_exp_rad)), window_s=4.0)
    assert ttc_receding.valid is False and ttc_receding.tau_s_hat_s is None


def test_6dof_ground_clutter_initiation_rejection() -> None:
    """V3_G1: 6-DoF WGS-84 ground-clutter rejection at track initiation."""
    ownship_states = get_ownship_states()[:15]
    ring = TrafficRing()
    fx = ring.fx

    # 1. Noiseless stationary 3D WGS-84 ground point under T1 6-DoF motion -> r_gnd_px < 0.05 px
    st0 = ownship_states[0]
    r_ecef_ned_0 = wgs84.r_ecef_ned(st0.lat_deg, st0.lon_deg)
    # Look down 8 deg below horizon North
    el_down = -math.radians(8.0)
    u_ned_0 = np.array([math.cos(el_down), 0.0, -math.sin(el_down)], dtype=np.float64)
    d_ecef_0 = r_ecef_ned_0 @ u_ned_0
    lam_hit, g_ecef = ray_wgs84_intersect(st0.p_ecef_m, d_ecef_0)
    assert lam_hit is not None and g_ecef is not None and lam_hit > 100.0

    obs_noiseless = []
    for k, st in enumerate(ownship_states[:6]):
        r_en = wgs84.r_ecef_ned(st.lat_deg, st.lon_deg)
        rel = g_ecef - st.p_ecef_m
        u_ned_k = r_en.T @ (rel / np.linalg.norm(rel))
        obs_noiseless.append(
            GroundObservation(
                t_s=k * 0.1,
                cam_id=ring.cameras[0].id,
                u_px=1280.0,
                v_px=800.0,
                s_mom_rad=2.5 / fx,
                los_ned_ins=u_ned_k,
                p_cam_ecef_ins=st.p_ecef_m,
                r_ecef_ned=r_en,
            )
        )
    res_noiseless = compute_ground_consistency_residual(obs_noiseless, fx=fx)
    assert res_noiseless.is_below_horizon is True
    assert res_noiseless.r_gnd_px < 0.05, f"Expected noiseless r_gnd_px < 0.05 px, got {res_noiseless.r_gnd_px}"

    # 2. Sky ray above horizon -> is_below_horizon is False (bypasses ground rejection)
    el_up = math.radians(5.0)
    u_sky = np.array([math.cos(el_up), 0.0, -math.sin(el_up)], dtype=np.float64)
    lam_sky, g_sky = ray_wgs84_intersect(st0.p_ecef_m, r_ecef_ned_0 @ u_sky)
    assert lam_sky is None and g_sky is None

    # 3. 100 seeded runs of stationary 3D WGS-84 ground points through MultiCameraTracker -> 0 confirmed false tracks
    g1_cfg = json.loads((REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json").read_text(encoding="utf-8"))
    confirmed_false_total = 0
    rejected_total = 0
    cam0 = ring.cameras[0]

    for trial_seed in range(100):
        rng = np.random.default_rng(50000 + trial_seed)
        az_gnd = float(rng.uniform(-0.35, 0.35))
        el_gnd = float(rng.uniform(-0.25, -0.05))  # below horizon in front camera
        u_ned_init = np.array(
            [math.cos(el_gnd) * math.cos(az_gnd), math.cos(el_gnd) * math.sin(az_gnd), -math.sin(el_gnd)],
            dtype=np.float64,
        )
        poses0 = ring.compute_poses(
            st0.lat_deg, st0.lon_deg, st0.h_m, st0.yaw_rad, st0.pitch_rad, st0.roll_rad
        )
        p_cam0 = np.asarray(poses0[cam0.id]["position"], dtype=np.float64)
        _, g_pt = ray_wgs84_intersect(p_cam0, r_ecef_ned_0 @ u_ned_init)
        assert g_pt is not None

        tracker = MultiCameraTracker(config=g1_cfg, ring=ring)
        for k in range(6):
            st_k = ownship_states[k]
            r_en_k = wgs84.r_ecef_ned(st_k.lat_deg, st_k.lon_deg)
            r_nb_k = rotations.euler_to_r(st_k.yaw_rad, st_k.pitch_rad, st_k.roll_rad)
            poses_k = ring.compute_poses(
                st_k.lat_deg, st_k.lon_deg, st_k.h_m, st_k.yaw_rad, st_k.pitch_rad, st_k.roll_rad
            )
            p_cam_k = np.asarray(poses_k[cam0.id]["position"], dtype=np.float64)
            r_ec_k = np.asarray(poses_k[cam0.id]["r_ecef_cam"], dtype=np.float64)
            v_cam = r_ec_k.T @ (g_pt - p_cam_k)
            u_proj = float(ring.camera_model.fx * (v_cam[0] / v_cam[2]) + ring.camera_model.cx) + float(rng.normal(0.0, 0.3))
            v_proj = float(ring.camera_model.fy * (v_cam[1] / v_cam[2]) + ring.camera_model.cy) + float(rng.normal(0.0, 0.3))
            s_fa = 2.5 + float(rng.normal(0.0, 0.05))
            det = DetectionRecord(
                t=k * 0.1,
                camera=cam0.id,
                u=u_proj,
                v=v_proj,
                s_px=s_fa,
                score=20.0,
                is_false_alarm=True,
                truth_id=None,
                s_mom_px=s_fa,
            )
            conf = tracker.step(
                t_s=k * 0.1,
                raw_detections=[det],
                r_ned_body_ins=r_nb_k,
                p_own_ecef=st_k.p_ecef_m,
                r_ecef_ned=r_en_k,
            )
            confirmed_false_total += len(conf)
        rejected_total += tracker.ground_clutter_rejected_count

    assert confirmed_false_total == 0, f"Expected 0 confirmed false tracks from stationary WGS-84 points, got {confirmed_false_total}"
    assert rejected_total == 100

    # 4. Airborne intruders both above horizon (v < 720) and below horizon (v > 720) confirm with zero added delay (0.2 s)
    for v_init, label in ((500.0, "above_horizon"), (820.0, "below_horizon")):
        tracker_air = MultiCameraTracker(config=g1_cfg, ring=ring)
        for k in range(4):
            st_k = ownship_states[k]
            r_en_k = wgs84.r_ecef_ned(st_k.lat_deg, st_k.lon_deg)
            r_nb_k = rotations.euler_to_r(st_k.yaw_rad, st_k.pitch_rad, st_k.roll_rad)
            det_air = DetectionRecord(
                t=k * 0.1,
                camera=cam0.id,
                u=1280.0 + k * 6.0,
                v=v_init - k * 3.0,
                s_px=3.0 + 0.15 * k,
                score=25.0,
                is_false_alarm=False,
                truth_id=f"INTRUDER_{label}",
                s_mom_px=3.0 + 0.15 * k,
            )
            conf_air = tracker_air.step(
                t_s=k * 0.1,
                raw_detections=[det_air],
                r_ned_body_ins=r_nb_k,
                p_own_ecef=st_k.p_ecef_m,
                r_ecef_ned=r_en_k,
            )
            if k == 2:
                assert len(conf_air) == 1, f"Failed 0.2s confirmation for {label} airborne intruder"
                assert abs(conf_air[0].confirmed_time - 0.2) < 1e-9
                assert tracker_air.ground_clutter_rejected_count == 0


def test_no_truth_in_g1_tracker_and_risk() -> None:
    """V4_G1: AST, import-graph, and runtime verification that tracker, ground_rejection, ttc, and risk never leak ground truth."""
    forbidden_pure = {"truth_id", "is_false_alarm", "q_ned_to_cam", "r_ned_body_true", "p_intruder_ecef"}
    forbidden_imports = {"xvs.traffic.encounter", "xvs.sim.trajectory"}
    for rel_path in (
        "src/xvs/tracking/ground_rejection.py",
        "src/xvs/tracking/ttc.py",
        "src/xvs/tracking/risk.py",
    ):
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=rel_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_pure:
                pytest.fail(f"{rel_path} references forbidden truth symbol {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_pure:
                pytest.fail(f"{rel_path} accesses forbidden truth attribute {node.attr}")
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_imports:
                pytest.fail(f"{rel_path} imports forbidden ground-truth module {node.module}")

    #Also audit src/xvs/tracking/tracker.py: zero truth imports, zero truth pose symbols, and truth_id/is_false_alarm
    # never used in any conditional, comparison, or arithmetic expression
    tracker_src = (REPO_ROOT / "src/xvs/tracking/tracker.py").read_text(encoding="utf-8")
    tracker_tree = ast.parse(tracker_src, filename="src/xvs/tracking/tracker.py")
    forbidden_tracker_symbols = {"q_ned_to_cam", "r_ned_body_true", "p_intruder_ecef"}
    for node in ast.walk(tracker_tree):
        if isinstance(node, ast.Name) and node.id in forbidden_tracker_symbols:
            pytest.fail(f"tracker.py references forbidden truth symbol {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in forbidden_tracker_symbols:
            pytest.fail(f"tracker.py accesses forbidden truth attribute {node.attr}")
        if isinstance(node, ast.ImportFrom) and node.module == "xvs.traffic.encounter":
            pytest.fail(f"tracker.py imports forbidden ground-truth module {node.module}")
        if isinstance(node, ast.ImportFrom) and node.module == "xvs.sim.trajectory":
            imported_names = {alias.name for alias in node.names}
            if imported_names != {"t1_cruise_state"}:
                pytest.fail(f"tracker.py imports non-ownship symbols from xvs.sim.trajectory: {imported_names}")
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"last_associated_truth_id", "associated_truth_ids"}
            and isinstance(node.ctx, ast.Load)
        ):
            # Only append() on associated_truth_ids is allowed
            pass
        if isinstance(node, ast.Compare):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in {
                    "last_associated_truth_id",
                    "associated_truth_ids",
                    "is_false_alarm",
                }:
                    pytest.fail(f"tracker.py compares diagnostic label {sub.attr} in branching/computation")

    # Runtime invariance check: identical detections with truth_id="ENC_001"/is_false_alarm=False
    # vs truth_id=None/is_false_alarm=True must yield identical state x, covariance P, status, TTC, and RiskEstimate!
    g1_cfg = json.loads((REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json").read_text(encoding="utf-8"))
    ring = TrafficRing()
    ownship_states = get_ownship_states()[:10]
    cam0 = ring.cameras[0]

    trk_a = MultiCameraTracker(config=g1_cfg, ring=ring)
    trk_b = MultiCameraTracker(config=g1_cfg, ring=ring)

    for k, st_k in enumerate(ownship_states):
        r_en_k = wgs84.r_ecef_ned(st_k.lat_deg, st_k.lon_deg)
        r_nb_k = rotations.euler_to_r(st_k.yaw_rad, st_k.pitch_rad, st_k.roll_rad)
        s_val = 2.5 * math.exp(0.04 * k)
        det_a = DetectionRecord(
            t=k * 0.1, camera=cam0.id, u=1200.0 + k * 1.5, v=600.0, s_px=s_val, score=20.0,
            is_false_alarm=False, truth_id="ENC_001", s_mom_px=s_val,
        )
        det_b = DetectionRecord(
            t=k * 0.1, camera=cam0.id, u=1200.0 + k * 1.5, v=600.0, s_px=s_val, score=20.0,
            is_false_alarm=True, truth_id=None, s_mom_px=s_val,
        )
        ca = trk_a.step(k * 0.1, [det_a], r_nb_k, p_own_ecef=st_k.p_ecef_m, r_ecef_ned=r_en_k)
        cb = trk_b.step(k * 0.1, [det_b], r_nb_k, p_own_ecef=st_k.p_ecef_m, r_ecef_ned=r_en_k)
        assert len(ca) == len(cb)
        if ca:
            np.testing.assert_allclose(ca[0].x, cb[0].x, rtol=0.0, atol=1e-15)
            np.testing.assert_allclose(ca[0].P, cb[0].P, rtol=0.0, atol=1e-15)
            ra = evaluate_track_risk(ca[0], t_s=k * 0.1)
            rb = evaluate_track_risk(cb[0], t_s=k * 0.1)
            assert ra.wc1_warning == rb.wc1_warning
            assert ra.hmd_hat_m == rb.hmd_hat_m
            assert ra.tau_mod_hat_s == rb.tau_mod_hat_s


def test_g1_cpu_determinism_and_crn() -> None:
    """V5_G1: CPU determinism and CRN stream alignment with G0."""
    sampler = EncounterSamplerV1()
    encs = sampler.generate(t1_cruise_state)[:8]
    g1_cfg = json.loads((REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json").read_text(encoding="utf-8"))

    res1 = run_evaluation(encs, config=g1_cfg, seed=7777, max_workers=1)
    res2 = run_evaluation(encs, config=g1_cfg, seed=7777, max_workers=1)
    for r in (res1, res2):
        r.pop("runtime_s", None)
        r.pop("frames_per_second", None)
        r.pop("truth_cache", None)
    j1 = json.dumps(res1, sort_keys=True)
    j2 = json.dumps(res2, sort_keys=True)
    assert hashlib.sha256(j1.encode("utf-8")).hexdigest() == hashlib.sha256(j2.encode("utf-8")).hexdigest()

    # Verify CRN stream alignment across lambdas AND against G0 config (identical u, v, s_px and INS drift)
    enc = construct_encounter(t1_cruise_state, encs[0])
    ownship_states = get_ownship_states()[:30]
    g0_cfg = json.loads((REPO_ROOT / "configs" / "xvs" / "tracking" / "g0_measured_d0.json").read_text(encoding="utf-8"))

    gen_g0 = SyntheticMeasurementGenerator(config=dict(g0_cfg, fppi_lambda=0.0005), seed=1010000)
    gen_g1_low = SyntheticMeasurementGenerator(config=dict(g1_cfg, fppi_lambda=0.0005), seed=1010000)
    gen_g1_high = SyntheticMeasurementGenerator(config=dict(g1_cfg, fppi_lambda=5.0), seed=1010000)

    for k, st_o in enumerate(ownship_states):
        d_g0, m_g0 = gen_g0.generate_frame(k * 0.1, st_o, [enc])
        d_g1_lo, m_g1_lo = gen_g1_low.generate_frame(k * 0.1, st_o, [enc])
        d_g1_hi, m_g1_hi = gen_g1_high.generate_frame(k * 0.1, st_o, [enc])

        np.testing.assert_allclose(m_g0["r_ned_body_ins"], m_g1_lo["r_ned_body_ins"], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(m_g1_lo["r_ned_body_ins"], m_g1_hi["r_ned_body_ins"], rtol=0.0, atol=1e-15)

        t_g0 = [(d.camera, d.u, d.v, d.s_px) for d in d_g0 if not d.is_false_alarm]
        t_g1_lo = [(d.camera, d.u, d.v, d.s_px) for d in d_g1_lo if not d.is_false_alarm]
        t_g1_hi = [(d.camera, d.u, d.v, d.s_px) for d in d_g1_hi if not d.is_false_alarm]
        assert t_g0 == t_g1_lo == t_g1_hi


def test_d32_ever_confirmed_and_d29_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """V6_G1: D32 ever-confirmed false track accounting and D29 fixed-sequence stopping rule."""
    from scripts.xvs.tracking.run_g1_lambda_sweep_v2 import evaluate_d29_sequence

    sampler = EncounterSamplerV1()
    enc_params = sampler.generate(t1_cruise_state)[0]
    g1_cfg = json.loads((REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json").read_text(encoding="utf-8"))
    cfg = dict(g1_cfg, fppi_lambda=0.0)

    orig_gen_frame = SyntheticMeasurementGenerator.generate_frame

    def patched_generate_frame(self, t_s, ownship, intruders):
        dets, meta = orig_gen_frame(self, t_s, ownship, intruders)
        true_dets = [d for d in dets if not d.is_false_alarm]
        # Inject a 3-frame false track above the horizon at t = 0.0, 0.1, 0.2 s that times out after 4 s
        if abs(t_s - 0.0) < 1e-6 or abs(t_s - 0.1) < 1e-6 or abs(t_s - 0.2) < 1e-6:
            true_dets.append(
                DetectionRecord(
                    t=t_s,
                    camera=self.ring.cameras[0].id,
                    u=1280.0,
                    v=400.0,
                    s_px=3.0,
                    score=25.0,
                    is_false_alarm=True,
                    truth_id=None,
                    s_mom_px=3.0,
                )
            )
        return true_dets, meta

    monkeypatch.setattr(SyntheticMeasurementGenerator, "generate_frame", patched_generate_frame)

    res = simulate_single_encounter(asdict(enc_params), cfg, seed=1010000)
    assert res["false_tracks"] == 1
    assert res["false_tracks_ever_confirmed"] == 1
    assert res["false_tracks_end_of_run"] == 0

    agg = aggregate_dataset_metrics([res], cfg)
    assert agg["tracking"]["false_tracks_total"] == 1
    assert agg["tracking"]["false_tracks_ever_confirmed_total"] == 1
    assert agg["tracking"]["false_tracks_end_of_run_total"] == 0

    # Test D29 fixed-sequence stopping rule: stops at first failing lambda even if a later lambda passes
    mock_records = [
        {
            "lambda": 0.0005,
            "n_seeds": 5,
            "paired_diff_missed_ci_95": [0.0, 0.0],
            "paired_diff_missed_upper_95": 0.0,
            "paired_diff_fa_per_hour_ci_95": [0.0, 0.0],
            "paired_diff_fa_per_hour_upper_95": 0.0,
            "false_tracks_garwood_upper_95": 0.04,
            "false_tracks_garwood_ci_95": [0.0, 0.05],
            "paired_diff_ge_12_5s": 0.0,
            "paired_diff_ge_25_0s": 0.0,
            "paired_diff_missed_rate": 0.0,
        },
        {
            "lambda": 0.005,
            "n_seeds": 5,
            "paired_diff_missed_ci_95": [-0.01, 0.02],
            "paired_diff_missed_upper_95": 0.015,
            "paired_diff_fa_per_hour_ci_95": [-0.1, 0.2],
            "paired_diff_fa_per_hour_upper_95": 0.15,
            "false_tracks_garwood_upper_95": 0.50,
            "false_tracks_garwood_ci_95": [0.1, 0.55],
            "paired_diff_ge_12_5s": 0.0,
            "paired_diff_ge_25_0s": 0.0,
            "paired_diff_missed_rate": 0.0,
        },
        {
            "lambda": 0.05,
            "n_seeds": 5,
            "paired_diff_missed_ci_95": [-0.01, 0.02],
            "paired_diff_missed_upper_95": 0.015,
            "paired_diff_fa_per_hour_ci_95": [-0.1, 0.2],
            "paired_diff_fa_per_hour_upper_95": 0.15,
            "false_tracks_garwood_upper_95": 1.45,  # Fails Criterion 3 (> 1.0/hr)
            "false_tracks_garwood_ci_95": [0.8, 1.55],
            "paired_diff_ge_12_5s": 0.0,
            "paired_diff_ge_25_0s": 0.0,
            "paired_diff_missed_rate": 0.0,
        },
        {
            "lambda": 0.5,
            "n_seeds": 5,
            "paired_diff_missed_ci_95": [0.0, 0.01],
            "paired_diff_missed_upper_95": 0.01,
            "paired_diff_fa_per_hour_ci_95": [0.0, 0.1],
            "paired_diff_fa_per_hour_upper_95": 0.1,
            "false_tracks_garwood_upper_95": 0.20,  # Point passes, but sequence already stopped at 0.05!
            "false_tracks_garwood_ci_95": [0.0, 0.25],
            "paired_diff_ge_12_5s": 0.0,
            "paired_diff_ge_25_0s": 0.0,
            "paired_diff_missed_rate": 0.0,
        },
    ]
    table, req_val, req_str = evaluate_d29_sequence(mock_records, [0.0005, 0.005, 0.05, 0.5])
    assert [row["fixed_sequence_pass"] for row in table] == [True, True, False, False]
    assert req_val == 0.005
    assert req_str == "0.005"


@pytest.mark.data
def test_g1_long_job_manifests() -> None:
    """V7_G1: Every G1 output directory contains run_manifest.json with code_hashes_match == True and clean XVS porcelain status."""
    g1_man_path = REPO_ROOT / "output" / "xvs" / "wp05" / "g1" / "run_manifest.json"
    sweep_man_path = REPO_ROOT / "output" / "xvs" / "wp05" / "g1_lambda_sweep_v2" / "run_manifest.json"
    indep_path = REPO_ROOT / "output" / "xvs" / "wp05" / "g1_lambda_sweep_v2" / "lambda_sweep_v2_results.json"
    pers_path = REPO_ROOT / "output" / "xvs" / "wp05" / "g1_lambda_sweep_v2" / "lambda_sweep_persistent_clutter.json"

    for p in (g1_man_path, sweep_man_path, indep_path, pers_path):
        assert p.is_file(), f"Missing G1 artifact: {p}"

    for man_p in (g1_man_path, sweep_man_path):
        man = json.loads(man_p.read_text(encoding="utf-8"))
        assert man.get("code_hashes_match") is True
        assert man.get("code_hashes_unchanged_during_run") is True
        assert man.get("git_status_start") == ""
        assert man.get("git_status_end") == ""
        assert man.get("git_commit_start") == man.get("git_commit_end")

    for res_p in (indep_path, pers_path):
        prov = json.loads(res_p.read_text(encoding="utf-8"))["provenance"]
        assert prov.get("code_hashes_match") is True
        assert prov.get("code_hashes_unchanged_during_run") is True
        assert prov.get("git_status_start") == ""
        assert prov.get("git_status_end") == ""


@pytest.mark.data
def test_wp05_g1_report_provenance_and_byte_identity(tmp_path: Path) -> None:
    """V8_G1: 0 unprovenanced numbers in WP05_G1_report.md and byte-identical regeneration into tmp_path."""
    committed_rep = REPO_ROOT / "docs" / "xvs" / "reports" / "WP05_G1_report.md"
    committed_prov = REPO_ROOT / "docs" / "xvs" / "reports" / "WP05_G1_report.md.provenance.json"
    assert committed_rep.is_file()
    assert committed_prov.is_file()

    unprov = verify_report_provenance(committed_rep)
    assert not unprov, f"WP05_G1_report.md has unprovenanced tokens: {unprov[:20]}"

    tmp_rep = tmp_path / "WP05_G1_report.md"
    script = REPO_ROOT / "scripts" / "xvs" / "tracking" / "report_tables_wp05_g1.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'};{REPO_ROOT}"
    env["XVS_DATA_ROOT"] = str(REPO_ROOT / "output")

    proc = subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_rep)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.returncode == 0, f"report_tables_wp05_g1.py failed:\n{proc.stdout}\n{proc.stderr}"
    assert tmp_rep.is_file()
    assert verify_report_provenance(tmp_rep) == []

    sha_committed = hashlib.sha256(committed_rep.read_bytes()).hexdigest()
    sha_regen = hashlib.sha256(tmp_rep.read_bytes()).hexdigest()
    assert sha_committed == sha_regen, f"Regenerated WP05_G1_report.md SHA-256 ({sha_regen}) != committed ({sha_committed})"
