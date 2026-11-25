"""WP-07 Gate G1 Integrity Extensions (Batch B11 — ROADMAP.md §4 Items 1–5).

Implements all five WP-07 G1 architectural improvements while preserving 100%
backward compatibility with WP-07 G0 functions and tests:

1. Item 1 — Low-Contrast Multi-Scale Horizon Edge Recovery & Smooth Pseudo-Huber IRLS:
   - `detect_camera_edges_multiscale`: classical multi-scale directional gradient
     accumulation (sigma in {1.0, 2.0, 3.5} px) with adaptive low-contrast local
     normalization and coarse-to-fine Theil-Sen horizon strip consensus.
   - `solve_attitude_gn_g1` & `fit_pipeline_attitude_g1`: smooth pseudo-Huber IRLS
     attitude refinement eliminating 2-cycle IRLS step oscillation (`MAX_ITER`) on
     low-altitude multi-camera frames with large terrain/altitude dip residuals.

2. Item 2 — Altitude-Conditioned D27 Multi-Camera Consistency Thresholds & Mondrian Slices:
   - `classify_altitude_band`: partitions measured altitude `h_meas_m` into
     `lt_250m` (< 250 m), `250_1000m` (250..1000 m), and `ge_1000m` (>= 1000 m).
   - `tune_monitor_thresholds_g1` & `evaluate_monitor_alarm_g1`: conditions all active
     sub-tests on `(cam_set, altitude_band)` with Bonferroni union-bound budgeting
     and finite-sample pooled standardized tail tolerance (`B11-F2`).
   - `fit_conformal_model_g1`: calibrates altitude-binned Mondrian conformal quantiles
     (`cam_set:condition:altitude_band`).

3. Item 3 — Boresight Calibration Residuals (`sigma_cal > 0`):
   - `perturb_rig_boresight`: applies deterministic seeded static mounting residuals
     plus slow sinusoidal thermal drift (`sigma_cal in {0.0, 0.02, 0.05} deg`) across
     all cameras in a Rig.
   - `calibrate_boresight_pl_sensitivity_g1`: calibrates RSS boresight sensitivity
     factors strictly on the `calibration` split (`B11-F3`) and propagates `sigma_cal`
     into Mondrian conformal protection levels `compute_pl_deg_altitude_mondrian`.

4. Item 4 — Single-Camera Mechanical Misalignment Fault Mode & Fault Isolation (`F_isolated`):
   - `inject_single_cam_misalignment`: deterministic step boresight pitch/roll shift on
     one active camera.
   - `isolate_and_exclude_faulty_camera`: D27 leave-one-camera-out fault identification
     and exclusion when >= 3 cameras are active (`FRBL`, `all`), restoring valid attitude
     estimation (`isolation_status = 'F_isolated'`).

5. Item 5 — Classical Horizon-Strip Image-Quality OOD Detector (`spurious_edge` & `glare`):
   - `compute_horizon_strip_ood_features` & `compute_horizon_strip_ood_score`: classical
     gradient-coherence, bimodal vertical edge peak ratio, and horizon-strip saturation
     detector targeting `spurious_edge` and `glare` without violating fault-free FAR 1.0%
     or FAR 0.1% budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from xvs.geo import rotations
from xvs.horizon_att.edges import extract_feature_map
from xvs.horizon_att.limb_model import (
    LimbRayBatch,
    compute_cone_residuals,
    compute_residuals_and_jacobian,
)
from xvs.integrity.conformal import (
    ALPHA_LEVELS,
    ConformalModel,
    compute_conformal_quantile,
    compute_protection_level_deg,
)
from xvs.integrity.faults import compute_sha256
from xvs.integrity.monitor import (
    MonitorThresholds,
    camera_consistency,
)
from xvs.rig.camera import PinholeCamera
from xvs.rig.rig import CameraMount, Rig
from xvs.sim import horizon


ALTITUDE_BANDS = ("lt_250m", "250_1000m", "ge_1000m")


def classify_altitude_band(
    h_meas_m: float,
    low_max_m: float = 250.0,
    mid_max_m: float = 1000.0,
) -> str:
    """Classify measured altitude `h_meas_m` into a physical D27 dip-sensitivity band.

    Strictly uses measured altitude `h_meas_m` (never ground-truth `h_true_m`).
    """
    h_val = float(h_meas_m)
    if math.isnan(h_val) or h_val < low_max_m:
        return "lt_250m"
    if h_val < mid_max_m:
        return "250_1000m"
    return "ge_1000m"


# =============================================================================
# ITEM 1: Classical Multi-Scale Directional Gradient Horizon Strip Extractor
#         and Smooth Pseudo-Huber IRLS Attitude Refinement
# =============================================================================


def detect_camera_edges_multiscale(
    img: np.ndarray,
    cam_id: str,
    cam: PinholeCamera,
    prior_pose: dict[str, Any] | None = None,
    step: int = 8,
    search_band_px: int = 50,
    min_gradient: float = 0.008,
    scales: tuple[float, ...] = (1.0, 2.0, 3.5),
    scale_weights: tuple[float, ...] = (0.50, 0.35, 0.15),
    strip_half_band_px: int = 24,
    border_margin_px: int = 64,
) -> list[dict[str, float]]:
    """Detect Earth-limb horizon edges via classical multi-scale directional gradient
    accumulation and coarse-to-fine Theil-Sen strip consensus (Item 1).
    """
    arr = np.asarray(img)
    h_img, w_img = arr.shape[:2]

    has_prior = False
    p_cam: np.ndarray | None = None
    r_cam: np.ndarray | None = None
    if prior_pose is not None:
        pos = prior_pose.get("p_ecef_cam", prior_pose.get("position"))
        if pos is not None:
            p_cam = np.asarray(pos, dtype=np.float64)
        if "r_ecef_cam" in prior_pose and prior_pose["r_ecef_cam"] is not None:
            r_cam = np.asarray(prior_pose["r_ecef_cam"], dtype=np.float64)
        elif "direction" in prior_pose and "up" in prior_pose and "right" in prior_pose:
            right = np.asarray(prior_pose["right"], dtype=np.float64)
            up = np.asarray(prior_pose["up"], dtype=np.float64)
            direction = np.asarray(prior_pose["direction"], dtype=np.float64)
            r_cam = np.column_stack([right, -up, direction])
        if p_cam is not None and r_cam is not None and r_cam.ndim == 2:
            has_prior = True

    v_lo_default = max(4, min(border_margin_px, h_img // 6))
    v_hi_default = min(h_img - 4, max(h_img - border_margin_px, h_img - h_img // 6))

    cols = list(range(20, max(21, w_img - 20), step))
    coarse_u: list[float] = []
    coarse_v: list[float] = []
    col_data: dict[int, tuple[np.ndarray, tuple[int, int] | None, float]] = {}

    if cols:
        feat_cols = extract_feature_map(arr[:, cols] if arr.ndim == 2 else arr[:, cols, :]).astype(np.float64)
        unique_count = len(np.unique(feat_cols[: min(100, h_img), : min(16, len(cols))]))
        is_mask = unique_count < 10
        eff_scales = (0.0,) if is_mask else scales
        eff_weights = (1.0,) if is_mask else scale_weights
        g_base_cols = np.zeros((h_img - 1, len(cols)), dtype=np.float64)
        for sig, w_s in zip(eff_scales, eff_weights):
            sm_cols = gaussian_filter1d(feat_cols, sigma=sig, axis=0, mode="nearest") if sig > 0.0 else feat_cols
            d_raw = sm_cols[1:, :] - sm_cols[:-1, :]
            polarity_w = np.where(d_raw < 0.0, 1.12, 0.88)
            g_base_cols += w_s * np.abs(d_raw) * polarity_w
    else:
        feat_cols = np.empty((h_img, 0), dtype=np.float64)
        g_base_cols = np.empty((h_img - 1, 0), dtype=np.float64)

    for j_col, col in enumerate(cols):
        u_col = float(col)
        v_search_range: tuple[int, int] | None = None
        if has_prior:
            v_roots = horizon.analytic_horizon_v_at_u(p_cam, r_cam, cam, u_col)
            v_in = [vr for vr in v_roots if -50.0 <= vr <= h_img + 50.0]
            if len(v_in) != 1:
                continue
            v_pred = v_in[0]
            slope = horizon.analytic_horizon_slope(p_cam, r_cam, cam, u_col, v_pred)
            if abs(slope) > 1.0:
                continue
            v_search_range = (
                max(2, int(math.floor(v_pred - search_band_px))),
                min(h_img - 2, int(math.ceil(v_pred + search_band_px))),
            )

        p = feat_cols[:, j_col]
        s_lo = v_search_range[0] if v_search_range is not None else v_lo_default
        s_hi = v_search_range[1] if v_search_range is not None else v_hi_default
        if s_hi <= s_lo + 2:
            continue

        p_win = p[s_lo:s_hi]
        p_span = float(np.percentile(p_win, 95) - np.percentile(p_win, 5)) if len(p_win) > 8 else 0.0
        noise_est = float(np.median(np.abs(np.diff(p_win)))) * 1.4826 if len(p_win) > 8 else 0.0
        snr_span = p_span / max(noise_est, 1e-6)
        if (0.004 <= p_span < 0.18) and (snr_span >= 3.5):
            gain = min(4.0, 0.22 / max(p_span, 1e-3))
        else:
            gain = 1.0

        g_acc = g_base_cols[:, j_col] * gain if gain != 1.0 else g_base_cols[:, j_col]

        col_data[col] = (g_acc, v_search_range, gain)
        win = g_acc[s_lo:s_hi]
        if len(win) > 0:
            k_best = s_lo + int(np.argmax(win))
            if g_acc[k_best] >= min_gradient:
                coarse_u.append(u_col)
                coarse_v.append(float(k_best) + 0.5)

    has_strip = (not has_prior) and (len(coarse_u) >= 5)
    m_strip = 0.0
    c_strip = float(h_img) / 2.0
    if has_strip:
        u_arr = np.asarray(coarse_u, dtype=np.float64)
        v_arr = np.asarray(coarse_v, dtype=np.float64)
        slopes: list[float] = []
        step_idx = max(1, len(u_arr) // 24)
        for i in range(0, len(u_arr), step_idx):
            for j in range(i + 1, len(u_arr), step_idx):
                du = u_arr[j] - u_arr[i]
                if abs(du) >= 48.0:
                    slopes.append(float((v_arr[j] - v_arr[i]) / du))
        m_strip = float(np.median(slopes)) if slopes else 0.0
        c_strip = float(np.median(v_arr - m_strip * u_arr))

    edges: list[dict[str, float]] = []
    for col in cols:
        if col not in col_data:
            continue
        g_acc, v_search_range, gain = col_data[col]
        u_col = float(col)
        if v_search_range is not None:
            s_lo, s_hi = v_search_range
        elif has_strip:
            v_pred = m_strip * u_col + c_strip
            s_lo = max(2, int(math.floor(v_pred - strip_half_band_px)))
            s_hi = min(h_img - 2, int(math.ceil(v_pred + strip_half_band_px)))
        else:
            s_lo, s_hi = v_lo_default, v_hi_default

        if s_hi <= s_lo + 1:
            continue
        win = g_acc[s_lo:s_hi]
        if len(win) == 0:
            continue
        k = s_lo + int(np.argmax(win))
        g_0 = float(g_acc[k])
        if g_0 < min_gradient:
            continue

        g_m1 = float(g_acc[k - 1]) if k >= 1 else g_0
        g_p1 = float(g_acc[k + 1]) if k + 1 < len(g_acc) else g_0
        denom = 2.0 * (g_m1 - 2.0 * g_0 + g_p1)
        if abs(denom) > 1e-12:
            delta = float(np.clip((g_m1 - g_p1) / denom, -0.5, 0.5))
        else:
            delta = 0.0
        v_sub = (k + 0.5) + delta
        if 0.0 <= v_sub <= h_img - 1.0:
            edges.append({
                "u": u_col,
                "v": float(v_sub),
                "gradient": float(g_0 / max(gain, 1.0)),
                "ms_gradient": float(g_0),
            })

    return edges


def solve_attitude_gn_g1(
    lat_deg: float,
    lon_deg: float,
    h_meas_m: float,
    yaw_rad: float,
    pitch_init_rad: float,
    roll_init_rad: float,
    batch: LimbRayBatch,
    max_iter: int = 60,
    tol_rad: float = 1e-6,
    damping: float = 1e-5,
    huber_delta_rad: float | None = math.radians(0.15),
) -> dict[str, Any]:
    """Smooth pseudo-Huber Levenberg-Marquardt attitude solver (Item 1)."""
    n = batch.n_rays
    if n < 3:
        return {
            "status": "FAILED_TOO_FEW_RAYS",
            "roll_deg": math.degrees(roll_init_rad),
            "pitch_deg": math.degrees(pitch_init_rad),
            "roll_rad": roll_init_rad,
            "pitch_rad": pitch_init_rad,
            "covariance_deg2": np.full((2, 2), np.nan),
            "covariance_rad2": np.full((2, 2), np.nan),
            "rms_residual_urad": np.nan,
            "inlier_count": n,
            "iterations": 0,
        }

    r_est = float(roll_init_rad)
    p_est = float(pitch_init_rad)
    converged = False
    iterations = 0
    weights = np.copy(batch.weights)
    lam = float(damping)
    prev_step: np.ndarray | None = None

    for it in range(max_iter):
        iterations = it + 1
        residuals, jacobian = compute_residuals_and_jacobian(
            lat_deg, lon_deg, h_meas_m, yaw_rad, p_est, r_est, batch
        )

        if huber_delta_rad is not None and huber_delta_rad > 0:
            scaled_r = residuals / float(huber_delta_rad)
            huber_w = 1.0 / np.sqrt(1.0 + scaled_r * scaled_r)
            effective_w = weights * huber_w
        else:
            effective_w = weights

        sqrt_w = np.sqrt(effective_w)[:, None]
        j_weighted = jacobian * sqrt_w
        r_weighted = residuals * sqrt_w.ravel()

        jt_j = j_weighted.T @ j_weighted
        jt_r = j_weighted.T @ r_weighted

        diag_jt_j = np.diag(np.diag(jt_j))
        lhs = jt_j + lam * (diag_jt_j + 1e-7 * np.eye(2))

        try:
            step = np.linalg.solve(lhs, -jt_r)
        except np.linalg.LinAlgError:
            step, _, _, _ = np.linalg.lstsq(lhs, -jt_r, rcond=None)

        if prev_step is not None and float(np.dot(step, prev_step)) < 0.0:
            step = 0.5 * step
            lam = min(1e-1, lam * 2.0)
        prev_step = step

        r_est += float(step[0])
        p_est += float(step[1])

        step_norm = float(np.linalg.norm(step))
        grad_norm = float(np.linalg.norm(jt_r)) / max(1, n)
        if step_norm < tol_rad or grad_norm < 1e-9:
            converged = True
            break

    residuals, jacobian = compute_residuals_and_jacobian(
        lat_deg, lon_deg, h_meas_m, yaw_rad, p_est, r_est, batch
    )
    rms_res_rad = float(np.sqrt(np.mean(residuals * residuals)))
    rms_res_urad = rms_res_rad * 1e6

    jt_j = jacobian.T @ (jacobian * weights[:, None])
    try:
        cov_rad2 = np.linalg.inv(jt_j) * max(1e-18, rms_res_rad * rms_res_rad)
    except np.linalg.LinAlgError:
        cov_rad2 = np.linalg.pinv(jt_j) * max(1e-18, rms_res_rad * rms_res_rad)

    rad2deg = 180.0 / math.pi
    cov_deg2 = cov_rad2 * (rad2deg * rad2deg)

    return {
        "status": "CONVERGED" if converged else "MAX_ITER",
        "roll_deg": math.degrees(r_est),
        "pitch_deg": math.degrees(p_est),
        "roll_rad": r_est,
        "pitch_rad": p_est,
        "covariance_deg2": cov_deg2,
        "covariance_rad2": cov_rad2,
        "rms_residual_urad": rms_res_urad,
        "inlier_count": n,
        "iterations": iterations,
    }


def fit_pipeline_attitude_g1(
    lat_deg: float,
    lon_deg: float,
    h_meas_m: float,
    yaw_rad: float,
    batch: LimbRayBatch,
    frame_k: int,
    rng_seed_base: int = 20260925,
    max_iterations: int = 40,
    inlier_threshold_rad: float = math.radians(0.30),
    min_inliers: int = 3,
    huber_delta_rad: float = math.radians(0.15),
) -> dict[str, Any]:
    """RANSAC + smooth pseudo-Huber LM attitude fit for WP-07 G1 (Item 1)."""
    n = batch.n_rays
    all_cams = ("F", "R", "B", "L", "U", "D")
    if n < 3:
        return {
            "status": "FAILED_TOO_FEW_RAYS",
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_rad": 0.0,
            "pitch_rad": 0.0,
            "covariance_deg2": np.full((2, 2), np.nan),
            "covariance_rad2": np.full((2, 2), np.nan),
            "rms_residual_urad": np.nan,
            "inlier_count": n,
            "inlier_count_by_cam": {c: 0 for c in all_cams},
            "total_rays": n,
            "iterations": 0,
        }

    rng = np.random.default_rng(rng_seed_base + frame_k)
    best_inlier_mask: np.ndarray | None = None
    best_inlier_count = 0
    best_inlier_cost = float("inf")
    best_candidate_p = 0.0
    best_candidate_r = 0.0

    for _ in range(max_iterations):
        sample_indices = rng.choice(n, size=3, replace=False)
        sample_batch = LimbRayBatch(
            batch.d_body[sample_indices],
            batch.lever_arm_body[sample_indices],
            [batch.cam_ids[i] for i in sample_indices],
        )
        fit_min = solve_attitude_gn_g1(
            lat_deg, lon_deg, h_meas_m, yaw_rad, 0.0, 0.0,
            sample_batch, max_iter=8, tol_rad=1e-5, damping=1e-4, huber_delta_rad=None,
        )
        if fit_min["status"] != "CONVERGED":
            continue
        cand_p = fit_min["pitch_rad"]
        cand_r = fit_min["roll_rad"]
        res = compute_cone_residuals(lat_deg, lon_deg, h_meas_m, yaw_rad, cand_p, cand_r, batch)
        abs_res = np.abs(res)
        inliers = abs_res <= inlier_threshold_rad
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count >= min_inliers:
            cost = float(np.sum(abs_res[inliers])) / inlier_count
            if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and cost < best_inlier_cost):
                best_inlier_count = inlier_count
                best_inlier_cost = cost
                best_inlier_mask = inliers
                best_candidate_p = cand_p
                best_candidate_r = cand_r

    if best_inlier_mask is not None and best_inlier_count >= min_inliers:
        inlier_indices = np.where(best_inlier_mask)[0]
        inlier_batch = LimbRayBatch(
            batch.d_body[inlier_indices],
            batch.lever_arm_body[inlier_indices],
            [batch.cam_ids[i] for i in inlier_indices],
            weights=batch.weights[inlier_indices],
            uvs=batch.uvs[inlier_indices],
        )
        fit_final = solve_attitude_gn_g1(
            lat_deg, lon_deg, h_meas_m, yaw_rad,
            best_candidate_p, best_candidate_r,
            inlier_batch,
            max_iter=60, tol_rad=1e-6, damping=1e-5,
            huber_delta_rad=huber_delta_rad,
        )
        final_inliers = inlier_indices
    else:
        fit_final = solve_attitude_gn_g1(
            lat_deg, lon_deg, h_meas_m, yaw_rad,
            0.0, 0.0,
            batch,
            max_iter=60, tol_rad=1e-6, damping=1e-5,
            huber_delta_rad=huber_delta_rad,
        )
        res = compute_cone_residuals(
            lat_deg, lon_deg, h_meas_m, yaw_rad,
            fit_final["pitch_rad"], fit_final["roll_rad"], batch
        )
        final_inliers = np.where(np.abs(res) <= inlier_threshold_rad)[0]

    inlier_counts_by_cam: dict[str, int] = {c: 0 for c in all_cams}
    for idx in final_inliers:
        cid = batch.cam_ids[idx]
        if cid in inlier_counts_by_cam:
            inlier_counts_by_cam[cid] += 1

    return {
        "status": fit_final["status"],
        "roll_deg": fit_final["roll_deg"],
        "pitch_deg": fit_final["pitch_deg"],
        "roll_rad": fit_final["roll_rad"],
        "pitch_rad": fit_final["pitch_rad"],
        "covariance_deg2": fit_final["covariance_deg2"],
        "covariance_rad2": fit_final["covariance_rad2"],
        "rms_residual_urad": fit_final["rms_residual_urad"],
        "inlier_count": len(final_inliers),
        "inlier_count_by_cam": inlier_counts_by_cam,
        "inliers": final_inliers,
        "total_rays": n,
        "iterations": fit_final["iterations"],
    }


# =============================================================================
# ITEM 5: Classical Horizon-Strip Image-Quality OOD Detector (`spurious_edge`, `glare`)
# =============================================================================


def compute_horizon_strip_ood_features(
    img: np.ndarray,
    edges: Sequence[dict[str, float]] | None = None,
    cam: PinholeCamera | None = None,
    cam_id: str = "F",
    strip_half_window_px: int = 45,
) -> dict[str, float]:
    """Compute classical horizon-strip image-quality OOD features (Item 5)."""
    arr = np.asarray(img)
    h_img, w_img = arr.shape[:2]

    if edges is None:
        if cam is None:
            raise ValueError("Either edges or cam must be provided to compute_horizon_strip_ood_features")
        edges = detect_camera_edges_multiscale(arr, cam_id, cam)

    if not edges:
        return {
            "bimodal_edge_fraction": 1.0,
            "strip_grad_p90": 0.0,
            "strip_saturation_fraction": 1.0,
            "strip_contour_jump_px": float(h_img),
            "ood_horizon_score": 10.0,
        }

    u_all = list(range(20, max(21, w_img - 20), 8))
    if not u_all:
        u_all = [w_img // 2]

    e_u = np.array([float(e["u"]) for e in edges], dtype=np.float64)
    e_v = np.array([float(e["v"]) for e in edges], dtype=np.float64)
    if len(e_u) >= 2 and (np.max(e_u) - np.min(e_u)) > 16.0:
        m_line, c_line = np.polyfit(e_u, e_v, 1)
    else:
        m_line, c_line = 0.0, float(np.median(e_v))

    sub_arr = arr[:, u_all] if arr.ndim == 2 else arr[:, u_all, :]
    if np.issubdtype(arr.dtype, np.integer):
        max_int = 65535.0 if arr.dtype == np.uint16 else 255.0
        if sub_arr.ndim == 2:
            norm_sub = sub_arr.astype(np.float64) / max_int
        else:
            norm_sub = np.mean(sub_arr.astype(np.float64), axis=2) / max_int
        raw_max = sub_arr if sub_arr.ndim == 2 else np.max(sub_arr, axis=2)
        sat_mask_sub = raw_max >= (int(max_int) - 1)
    else:
        norm_sub = sub_arr.astype(np.float64) if sub_arr.ndim == 2 else np.mean(sub_arr.astype(np.float64), axis=2)
        norm_sub = np.clip(norm_sub, 0.0, 1.0)
        sat_mask_sub = norm_sub >= 0.992

    bimodal_cols = 0
    sat_pixels = 0
    total_pixels = 0
    grad_list: list[float] = []
    u_sorted_v: list[tuple[float, float]] = [(float(e["u"]), float(e["v"])) for e in edges]

    for j_idx, u_col in enumerate(u_all):
        v_val = float(m_line * u_col + c_line)
        v_idx = int(np.clip(round(v_val), 1, h_img - 2))

        v0 = max(1, v_idx - strip_half_window_px)
        v1 = min(h_img - 1, v_idx + strip_half_window_px)
        col_slice = norm_sub[v0:v1, j_idx]
        if len(col_slice) < 6:
            continue

        diff_p = np.abs(np.diff(col_slice))
        k_main = int(np.argmax(diff_p))
        g_main = float(diff_p[k_main])
        grad_list.append(g_main)

        diff_masked = diff_p.copy()
        m_lo = max(0, k_main - 4)
        m_hi = min(len(diff_p), k_main + 5)
        diff_masked[m_lo:m_hi] = 0.0
        g_sec = float(np.max(diff_masked)) if len(diff_masked) > 0 else 0.0

        if g_sec >= 0.12 and (g_sec / max(g_main, 1e-6)) >= 0.22:
            bimodal_cols += 1

        sat_pixels += int(np.count_nonzero(sat_mask_sub[v0:v1, j_idx]))
        total_pixels += len(col_slice)

    n_e = max(1, len(grad_list))
    bimodal_frac = float(bimodal_cols / n_e)
    grad_p90 = float(np.percentile(grad_list, 90)) if grad_list else 0.0
    sat_frac = float(sat_pixels / max(1, total_pixels))
    missing_frac = max(0.0, 1.0 - float(len(edges)) / float(max(1, len(u_all))))

    u_sorted_v.sort(key=lambda x: x[0])
    if len(u_sorted_v) >= 3:
        vs = np.array([x[1] for x in u_sorted_v], dtype=np.float64)
        dvs = np.abs(np.diff(vs))
        jump_p95 = float(np.percentile(dvs, 95))
    else:
        jump_p95 = 0.0

    ood_score = float(
        2.5 * bimodal_frac
        + 3.0 * max(0.0, grad_p90 - 0.55)
        + 8.0 * sat_frac
        + 2.0 * max(0.0, missing_frac - 0.12)
        + 0.05 * max(0.0, jump_p95 - 3.5)
    )

    return {
        "bimodal_edge_fraction": bimodal_frac,
        "strip_grad_p90": grad_p90,
        "strip_saturation_fraction": sat_frac,
        "strip_contour_jump_px": jump_p95,
        "ood_horizon_score": ood_score,
    }


def compute_horizon_strip_ood_score(
    img: np.ndarray,
    edges: Sequence[dict[str, float]] | None = None,
    cam: PinholeCamera | None = None,
    cam_id: str = "F",
    strip_half_window_px: int = 45,
) -> float:
    """Return scalar `ood_horizon_score` from classical horizon-strip OOD features (Item 5)."""
    return float(
        compute_horizon_strip_ood_features(
            img, edges=edges, cam=cam, cam_id=cam_id, strip_half_window_px=strip_half_window_px
        )["ood_horizon_score"]
    )


# =============================================================================
# ITEM 2: Altitude-Conditioned Monitor M1 Thresholds & Mondrian Conformal Model
# =============================================================================


@dataclass
class MonitorThresholdsG1:
    """Monitor M1 threshold configuration with altitude-conditioned D27 and OOD gates."""
    far_target: float
    max_rms_residual_urad: float | None = None
    max_condition_number: float | None = None
    min_inlier_fraction: float | None = None
    min_inlier_count: int | None = None
    max_leave_one_out_deg: float | None = None
    max_ood_horizon_score: float | None = None
    by_cam_set: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_altitude_band: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "far_target": self.far_target,
            "max_rms_residual_urad": self.max_rms_residual_urad,
            "max_condition_number": self.max_condition_number,
            "min_inlier_fraction": self.min_inlier_fraction,
            "min_inlier_count": self.min_inlier_count,
            "max_leave_one_out_deg": self.max_leave_one_out_deg,
            "max_ood_horizon_score": self.max_ood_horizon_score,
            "by_cam_set": self.by_cam_set,
            "by_altitude_band": self.by_altitude_band,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonitorThresholdsG1:
        return cls(
            far_target=float(data.get("far_target", 0.01)),  # default-ok: optional fallback default
            max_rms_residual_urad=float(data["max_rms_residual_urad"]) if data.get("max_rms_residual_urad") is not None else None,
            max_condition_number=float(data["max_condition_number"]) if data.get("max_condition_number") is not None else None,
            min_inlier_fraction=float(data["min_inlier_fraction"]) if data.get("min_inlier_fraction") is not None else None,
            min_inlier_count=int(data["min_inlier_count"]) if data.get("min_inlier_count") is not None else None,
            max_leave_one_out_deg=float(data["max_leave_one_out_deg"]) if data.get("max_leave_one_out_deg") is not None else None,
            max_ood_horizon_score=float(data["max_ood_horizon_score"]) if data.get("max_ood_horizon_score") is not None else None,
            by_cam_set=data.get("by_cam_set", {}),
            by_altitude_band=data.get("by_altitude_band", {}),
        )


def _finite_sample_upper_quantiles_by_band(
    band_values: dict[str, list[float]],
    tail_prob: float,
) -> tuple[float, dict[str, float]]:
    """Compute overall and altitude-binned upper thresholds with pooled standardized
    tail ratio inflation when n_bin < 1 / tail_prob (B11-F2).

    Strictly uses only the provided calibration/validation samples (zero test peeking).
    """
    q_pct = 100.0 * (1.0 - tail_prob)
    all_vals = [x for vals in band_values.values() for x in vals]
    if not all_vals:
        return float("nan"), {}

    overall_emp = float(np.percentile(all_vals, q_pct))
    mu_by_b: dict[str, float] = {}
    sig_by_b: dict[str, float] = {}
    z_pooled: list[float] = []
    for aband, vals in band_values.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mu_b = float(np.median(arr))
        p95_b = float(np.percentile(arr, 95.0))
        std_b = float(np.std(arr))
        sig_b = max(1e-9, p95_b - mu_b, std_b)
        mu_by_b[aband] = mu_b
        sig_by_b[aband] = sig_b
        z_pooled.extend(((arr - mu_b) / sig_b).tolist())

    z_arr = np.asarray(z_pooled, dtype=np.float64)
    z_emp = float(np.percentile(z_arr, q_pct))
    z_99 = float(np.percentile(z_arr, 99.0))
    z_999 = float(np.percentile(z_arr, 99.9))
    z_95 = float(np.percentile(z_arr, 95.0))
    dz_decade = max(z_999 - z_99, 0.65 * (z_99 - z_95), 0.35)

    n_total = len(all_vals)
    if n_total < (1.5 / max(tail_prob, 1e-9)):
        decades_beyond_99 = max(0.0, math.log10(0.01 / max(tail_prob, 1e-9)))
        fin_factor_all = 1.0 + 0.32 / math.sqrt(max(0.20, n_total * tail_prob))
        z_tol_all = max(z_emp, z_99 + dz_decade * decades_beyond_99) * fin_factor_all
        overall_mu = float(np.median(all_vals))
        overall_sig = max(1e-9, float(np.percentile(all_vals, 95.0)) - overall_mu, float(np.std(all_vals)))
        overall_th = max(
            overall_emp * (1.0 + 0.08 / math.sqrt(max(0.20, n_total * tail_prob))),
            overall_mu + overall_sig * z_tol_all,
        )
    else:
        overall_th = overall_emp

    by_band_th: dict[str, float] = {}
    for aband in ALTITUDE_BANDS:
        vals = band_values.get(aband, [])
        if not vals:
            by_band_th[aband] = overall_th
            continue
        n_b = len(vals)
        emp_b = float(np.percentile(vals, q_pct))
        if n_b < (1.5 / max(tail_prob, 1e-9)):
            decades = max(0.0, math.log10(0.01 / max(tail_prob, 1e-9)))
            eff_count = max(0.15, n_b * tail_prob)
            fin_mult = 1.0 + 0.38 / math.sqrt(eff_count)
            z_tol_b = max(z_emp, z_99 + dz_decade * decades) * fin_mult
            pooled_proj = mu_by_b[aband] + sig_by_b[aband] * z_tol_b
            emp_padded = emp_b * (1.0 + 0.12 / math.sqrt(eff_count))
            by_band_th[aband] = float(max(emp_padded, pooled_proj))
        else:
            eff_count = n_b * tail_prob
            if eff_count < 6.0:
                pooled_proj = mu_by_b[aband] + sig_by_b[aband] * z_emp * (1.0 + 0.10 / math.sqrt(eff_count))
                by_band_th[aband] = float(max(emp_b, pooled_proj))
            else:
                by_band_th[aband] = emp_b

    return float(overall_th), by_band_th


def _finite_sample_lower_quantiles_by_band(
    band_values: dict[str, list[float]],
    tail_prob: float,
    is_integer: bool = False,
    min_floor: float = 0.0,
) -> tuple[float | int, dict[str, float | int]]:
    """Compute overall and altitude-binned lower thresholds with pooled standardized
    lower-tail inflation when n_bin < 1 / tail_prob (B11-F2).
    """
    q_pct = 100.0 * tail_prob
    all_vals = [x for vals in band_values.values() for x in vals]
    if not all_vals:
        return (0 if is_integer else float("nan")), {}

    overall_emp = float(np.percentile(all_vals, q_pct))
    mu_by_b: dict[str, float] = {}
    sig_by_b: dict[str, float] = {}
    z_low_pooled: list[float] = []
    for aband, vals in band_values.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mu_b = float(np.median(arr))
        p05_b = float(np.percentile(arr, 5.0))
        std_b = float(np.std(arr))
        sig_b = max(1e-9, mu_b - p05_b, std_b)
        mu_by_b[aband] = mu_b
        sig_by_b[aband] = sig_b
        z_low_pooled.extend(((mu_b - arr) / sig_b).tolist())

    z_arr = np.asarray(z_low_pooled, dtype=np.float64)
    z_emp = float(np.percentile(z_arr, 100.0 * (1.0 - tail_prob)))
    z_99 = float(np.percentile(z_arr, 99.0))
    z_999 = float(np.percentile(z_arr, 99.9))
    z_95 = float(np.percentile(z_arr, 95.0))
    dz_decade = max(z_999 - z_99, 0.65 * (z_99 - z_95), 0.35)

    def _format_low(val: float, vals_ref: list[float]) -> float | int:
        v_clamped = max(min_floor, val)
        if is_integer:
            cnt = int(math.floor(v_clamped))
            tie_count = sum(1 for v in vals_ref if v < cnt)
            if tie_count > int(math.ceil(len(vals_ref) * tail_prob)) and cnt > int(min_floor):
                cnt -= 1
            return int(max(min_floor, cnt))
        return float(v_clamped)

    n_total = len(all_vals)
    if n_total < (1.5 / max(tail_prob, 1e-9)):
        decades = max(0.0, math.log10(0.01 / max(tail_prob, 1e-9)))
        eff_count = max(0.20, n_total * tail_prob)
        z_tol_all = max(z_emp, z_99 + dz_decade * decades) * (1.0 + 0.30 / math.sqrt(eff_count))
        overall_mu = float(np.median(all_vals))
        overall_sig = max(1e-9, overall_mu - float(np.percentile(all_vals, 5.0)), float(np.std(all_vals)))
        raw_overall = min(overall_emp * (1.0 - 0.06 / math.sqrt(eff_count)), overall_mu - overall_sig * z_tol_all)
    else:
        raw_overall = overall_emp

    overall_out = _format_low(raw_overall, all_vals)
    by_band_out: dict[str, float | int] = {}
    for aband in ALTITUDE_BANDS:
        vals = band_values.get(aband, [])
        if not vals:
            by_band_out[aband] = overall_out
            continue
        n_b = len(vals)
        emp_b = float(np.percentile(vals, q_pct))
        if n_b < (1.5 / max(tail_prob, 1e-9)):
            decades = max(0.0, math.log10(0.01 / max(tail_prob, 1e-9)))
            eff_count = max(0.15, n_b * tail_prob)
            z_tol_b = max(z_emp, z_99 + dz_decade * decades) * (1.0 + 0.35 / math.sqrt(eff_count))
            pooled_proj = mu_by_b[aband] - sig_by_b[aband] * z_tol_b
            emp_padded = emp_b * (1.0 - 0.08 / math.sqrt(eff_count))
            raw_b = min(emp_padded, pooled_proj)
        else:
            eff_count = n_b * tail_prob
            if eff_count < 6.0:
                pooled_proj = mu_by_b[aband] - sig_by_b[aband] * z_emp * (1.0 + 0.08 / math.sqrt(eff_count))
                raw_b = min(emp_b, pooled_proj)
            else:
                raw_b = emp_b
        by_band_out[aband] = _format_low(raw_b, vals)

    return overall_out, by_band_out


def tune_monitor_thresholds_g1(
    nominal_val_records: list[dict[str, Any]],
    target_far: float = 0.01,
) -> MonitorThresholdsG1:
    """Tune Monitor M1 thresholds with altitude-conditioned sub-tests, Bonferroni
    union-bound budgeting, and finite-sample pooled standardized tail tolerance (Item 2 & Item 5, B11-F2).

    Strictly uses only fault-free calibration/validation records (zero test-split peeking).
    """
    if not nominal_val_records:
        raise ValueError("nominal_val_records is empty; cannot tune G1 monitor thresholds.")

    converged_recs = [
        r for r in nominal_val_records
        if r.get("status") == "CONVERGED" and r.get("fault_type", "none") in ("none", None)
    ]
    if not converged_recs:
        raise ValueError("No CONVERGED records in nominal_val_records; cannot tune G1 monitor thresholds.")

    def _get_cc(r: dict[str, Any]) -> float:
        v = r.get("camera_consistency_deg")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = r.get("leave_one_out_consistency_deg", float("nan"))
        return float(v) if v is not None else float("nan")

    by_cam_set: dict[str, dict[str, Any]] = {}
    cam_sets = sorted(set(r.get("cam_set", "all") for r in converged_recs))

    for cset in cam_sets:
        c_recs = [r for r in converged_recs if r.get("cam_set", "all") == cset]
        if not c_recs:
            continue
        has_ood = any("ood_horizon_score" in r and r["ood_horizon_score"] is not None for r in c_recs)
        has_loo = cset in ("FRBL", "horizontal", "all") and any(not math.isnan(_get_cc(r)) for r in c_recs)
        m_active = 4 + (1 if has_loo else 0) + (1 if has_ood else 0)
        tail_prob = max(1e-6, float(target_far) / float(m_active))

        rms_by_b: dict[str, list[float]] = {b: [] for b in ALTITUDE_BANDS}
        cond_by_b: dict[str, list[float]] = {b: [] for b in ALTITUDE_BANDS}
        frac_by_b: dict[str, list[float]] = {b: [] for b in ALTITUDE_BANDS}
        cnt_by_b: dict[str, list[float]] = {b: [] for b in ALTITUDE_BANDS}
        loo_by_b: dict[str, list[float]] = {b: [] for b in ALTITUDE_BANDS}
        ood_by_b: dict[str, list[float]] = {b: [] for b in ALTITUDE_BANDS}

        for r in c_recs:
            aband = classify_altitude_band(r.get("h_meas_m", 0.0))  # default-ok: optional fallback default
            rms_v = float(r.get("rms_residual_urad", float("nan")))
            if not math.isnan(rms_v):
                rms_by_b[aband].append(rms_v)
            cond_v = float(r.get("condition_number", float("nan")))
            if not math.isnan(cond_v):
                cond_by_b[aband].append(cond_v)
            frac_v = float(r.get("inlier_fraction", float("nan")))
            if not math.isnan(frac_v):
                frac_by_b[aband].append(frac_v)
            if "inlier_count" in r and r["inlier_count"] is not None:
                cnt_by_b[aband].append(float(r["inlier_count"]))
            if has_loo:
                cc_v = _get_cc(r)
                if not math.isnan(cc_v):
                    loo_by_b[aband].append(cc_v)
            if has_ood and r.get("ood_horizon_score") is not None:
                ood_v = float(r["ood_horizon_score"])
                if not math.isnan(ood_v):
                    ood_by_b[aband].append(ood_v)

        th_c_rms, rms_alt_map = _finite_sample_upper_quantiles_by_band(rms_by_b, tail_prob)
        if "250_1000m" in rms_alt_map and "lt_250m" in rms_alt_map:
            rms_alt_map["250_1000m"] = float(min(rms_alt_map["250_1000m"], rms_alt_map["lt_250m"]))
        th_c_cond, cond_alt_map = _finite_sample_upper_quantiles_by_band(cond_by_b, tail_prob)
        th_c_frac_raw, frac_alt_raw = _finite_sample_lower_quantiles_by_band(frac_by_b, tail_prob, is_integer=False, min_floor=0.05)
        th_c_cnt_raw, cnt_alt_raw = _finite_sample_lower_quantiles_by_band(cnt_by_b, tail_prob, is_integer=True, min_floor=3.0)

        th_c_frac = float(th_c_frac_raw)
        frac_alt_map = {k: float(v) for k, v in frac_alt_raw.items()}
        if "250_1000m" in frac_alt_map and "lt_250m" in frac_alt_map:
            frac_alt_map["250_1000m"] = float(max(frac_alt_map["250_1000m"], frac_alt_map["lt_250m"]))
        th_c_cnt = int(th_c_cnt_raw)
        cnt_alt_map = {k: int(v) for k, v in cnt_alt_raw.items()}

        th_c_ood: float | None = None
        ood_alt_map: dict[str, float] = {}
        if has_ood:
            raw_ood, raw_ood_alt = _finite_sample_upper_quantiles_by_band(ood_by_b, tail_prob)
            if not math.isnan(raw_ood):
                th_c_ood = max(1.25, float(raw_ood))
                ood_alt_map = {k: max(1.25, float(v)) for k, v in raw_ood_alt.items()}

        th_c_loo: float | None = None
        loo_by_alt: dict[str, float] = {}
        loo_reason: str | None = None
        if cset in ("FRBL", "horizontal", "all"):
            if has_loo:
                th_c_loo, loo_by_alt = _finite_sample_upper_quantiles_by_band(loo_by_b, tail_prob)
                if "250_1000m" in loo_by_alt and "lt_250m" in loo_by_alt:
                    loo_by_alt["250_1000m"] = float(min(loo_by_alt["250_1000m"], loo_by_alt["lt_250m"]))
        else:
            loo_reason = "Single-camera set F (c=1): leave-one-camera-out consistency requires c>=2 (D27)"

        cam_entry: dict[str, Any] = {
            "m_active_subtests": int(m_active),
            "alpha_sub": float(tail_prob),
            "max_rms_residual_urad": th_c_rms,
            "max_rms_residual_by_altitude_urad": rms_alt_map,
            "max_condition_number": th_c_cond,
            "max_condition_number_by_altitude": cond_alt_map,
            "min_inlier_fraction": th_c_frac,
            "min_inlier_fraction_by_altitude": frac_alt_map,
            "min_inlier_count": th_c_cnt,
            "min_inlier_count_by_altitude": cnt_alt_map,
            "max_leave_one_out_deg": th_c_loo,
            "max_ood_horizon_score": th_c_ood,
        }
        if ood_alt_map:
            cam_entry["max_ood_horizon_score_by_altitude"] = ood_alt_map
        if loo_by_alt:
            cam_entry["max_leave_one_out_by_altitude_deg"] = loo_by_alt
        if loo_reason is not None:
            cam_entry["max_leave_one_out_deg_reason"] = loo_reason
        by_cam_set[cset] = cam_entry

    by_altitude_band: dict[str, dict[str, float]] = {}
    tail_prob_all = max(1e-6, float(target_far) / 6.0)
    mc_loo_by_b: dict[str, list[float]] = {b: [] for b in ALTITUDE_BANDS}
    for r in converged_recs:
        if r.get("cam_set") in ("FRBL", "horizontal", "all"):
            cc_v = _get_cc(r)
            if not math.isnan(cc_v):
                mc_loo_by_b[classify_altitude_band(r.get("h_meas_m", 0.0))].append(cc_v)  # default-ok: optional fallback default

    _, mc_loo_th_by_b = _finite_sample_upper_quantiles_by_band(mc_loo_by_b, tail_prob_all)
    if "250_1000m" in mc_loo_th_by_b and "lt_250m" in mc_loo_th_by_b:
        mc_loo_th_by_b["250_1000m"] = float(min(mc_loo_th_by_b["250_1000m"], mc_loo_th_by_b["lt_250m"]))
    for aband in ALTITUDE_BANDS:
        b_mc = mc_loo_by_b.get(aband, [])
        if b_mc:
            by_altitude_band[aband] = {
                "max_leave_one_out_deg": float(mc_loo_th_by_b[aband]),
                "median_leave_one_out_deg": float(np.median(b_mc)),
                "p99_leave_one_out_deg": float(np.percentile(b_mc, 99.0)),
                "n_validation_frames": len(b_mc),
            }

    q_up_all = 100.0 * (1.0 - tail_prob_all)
    q_lo_all = 100.0 * tail_prob_all
    all_rms = [float(r["rms_residual_urad"]) for r in converged_recs if not math.isnan(r.get("rms_residual_urad", float("nan")))]
    all_cond = [float(r["condition_number"]) for r in converged_recs if not math.isnan(r.get("condition_number", float("nan")))]
    all_frac = [float(r["inlier_fraction"]) for r in converged_recs if not math.isnan(r.get("inlier_fraction", float("nan")))]
    all_cnt = [int(r["inlier_count"]) for r in converged_recs if "inlier_count" in r]
    all_loo = [_get_cc(r) for r in converged_recs if r.get("cam_set") in ("FRBL", "horizontal", "all") and not math.isnan(_get_cc(r))]
    all_ood = [
        float(r["ood_horizon_score"])
        for r in converged_recs
        if r.get("ood_horizon_score") is not None and not math.isnan(float(r["ood_horizon_score"]))
    ]

    return MonitorThresholdsG1(
        far_target=float(target_far),
        max_rms_residual_urad=float(np.percentile(all_rms, q_up_all)) if all_rms else None,
        max_condition_number=float(np.percentile(all_cond, q_up_all)) if all_cond else None,
        min_inlier_fraction=float(np.percentile(all_frac, q_lo_all)) if all_frac else None,
        min_inlier_count=int(np.percentile(all_cnt, q_lo_all)) if all_cnt else None,
        max_leave_one_out_deg=float(np.percentile(all_loo, q_up_all)) if all_loo else None,
        max_ood_horizon_score=max(1.25, float(np.percentile(all_ood, q_up_all))) if all_ood else None,
        by_cam_set=by_cam_set,
        by_altitude_band=by_altitude_band,
    )


def evaluate_monitor_alarm_g1(
    record: dict[str, Any],
    thresholds: MonitorThresholdsG1 | MonitorThresholds | dict[str, Any],
    use_altitude_conditioned_d27: bool = True,
    use_ood_layer: bool = True,
    sigma_cal_deg: float = 0.0,
    boresight_sensitivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate Monitor M1 alarm rule with altitude-conditioned sub-tests, OOD gate,
    and optional boresight calibration uncertainty propagation (Items 2, 3, 5).
    """
    if isinstance(thresholds, dict):
        th = MonitorThresholdsG1.from_dict(thresholds)
    elif isinstance(thresholds, MonitorThresholds):
        th = MonitorThresholdsG1.from_dict(thresholds.to_dict())
    else:
        th = thresholds

    reasons: list[str] = []
    status = record.get("status", "")
    if status != "CONVERGED":
        reasons.append(f"STATUS_{status or 'NOT_CONVERGED'}")

    cam_set = record.get("cam_set", "")
    th_cam = th.by_cam_set.get(cam_set, {}) if th.by_cam_set else {}
    aband = classify_altitude_band(record.get("h_meas_m", 0.0))  # default-ok: optional fallback default

    if use_altitude_conditioned_d27 and aband in th_cam.get("max_rms_residual_by_altitude_urad", {}):
        max_rms = th_cam["max_rms_residual_by_altitude_urad"][aband]
    else:
        max_rms = th_cam.get("max_rms_residual_urad", th.max_rms_residual_urad)

    if use_altitude_conditioned_d27 and aband in th_cam.get("max_condition_number_by_altitude", {}):
        max_cond = th_cam["max_condition_number_by_altitude"][aband]
    else:
        max_cond = th_cam.get("max_condition_number", th.max_condition_number)

    if use_altitude_conditioned_d27 and aband in th_cam.get("min_inlier_fraction_by_altitude", {}):
        min_frac = th_cam["min_inlier_fraction_by_altitude"][aband]
    else:
        min_frac = th_cam.get("min_inlier_fraction", th.min_inlier_fraction)

    if use_altitude_conditioned_d27 and aband in th_cam.get("min_inlier_count_by_altitude", {}):
        min_cnt = th_cam["min_inlier_count_by_altitude"][aband]
    else:
        min_cnt = th_cam.get("min_inlier_count", th.min_inlier_count)
    if min_cnt is not None:
        min_cnt = int(min_cnt)

    alt_loo_map = th_cam.get("max_leave_one_out_by_altitude_deg", {})
    if use_altitude_conditioned_d27 and aband in alt_loo_map:
        max_loo = alt_loo_map[aband]
    else:
        max_loo = th_cam.get("max_leave_one_out_deg", th.max_leave_one_out_deg)

    if use_altitude_conditioned_d27 and aband in th_cam.get("max_ood_horizon_score_by_altitude", {}):
        max_ood = th_cam["max_ood_horizon_score_by_altitude"][aband]
    else:
        max_ood = th_cam.get("max_ood_horizon_score", th.max_ood_horizon_score)

    eff_sig_cal = float(sigma_cal_deg or record.get("sigma_cal_deg", 0.0) or 0.0)  # default-ok: optional integrity record/sensitivity fallback
    if eff_sig_cal > 0.0:
        sens_c = (boresight_sensitivity or {}).get("by_cam_set", {}).get(cam_set, {})
        c_d27 = float(sens_c.get("d27_loo_factor", 6.0))  # default-ok: optional integrity record/sensitivity fallback
        c_rms_urad_per_deg = float(sens_c.get("rms_urad_per_deg", 45000.0))  # default-ok: optional integrity record/sensitivity fallback
        if max_loo is not None and not math.isnan(max_loo):
            max_loo = math.hypot(float(max_loo), c_d27 * eff_sig_cal)
        if max_rms is not None and not math.isnan(max_rms):
            max_rms = math.hypot(float(max_rms), c_rms_urad_per_deg * eff_sig_cal)

    if max_rms is not None:
        rms = record.get("rms_residual_urad", float("nan"))
        if math.isnan(rms) or rms > max_rms:
            reasons.append(f"RMS_HIGH_{rms:.1f}urad")

    if max_cond is not None:
        cond = record.get("condition_number", float("nan"))
        if math.isnan(cond) or cond > max_cond:
            reasons.append(f"COND_HIGH_{cond:.1e}")

    if min_frac is not None:
        inlier_frac = record.get("inlier_fraction", float("nan"))
        if math.isnan(inlier_frac) or inlier_frac < min_frac:
            reasons.append(f"INLIER_FRAC_LOW_{inlier_frac:.2f}")

    if min_cnt is not None:
        inlier_cnt = record.get("inlier_count", 0)  # default-ok: optional fallback default
        if inlier_cnt < min_cnt:
            reasons.append(f"INLIER_COUNT_LOW_{inlier_cnt}")

    if cam_set in ("FRBL", "horizontal", "all") and max_loo is not None:
        loo = record.get("camera_consistency_deg")
        if loo is None or (isinstance(loo, float) and math.isnan(loo)):
            loo = record.get("leave_one_out_consistency_deg", float("nan"))
        if loo is not None and not math.isnan(loo) and not math.isnan(max_loo) and loo > max_loo:
            reasons.append(f"LOO_HIGH_{aband}_{loo:.4f}deg")

    if use_ood_layer and max_ood is not None:
        ood_val = record.get("ood_horizon_score")
        if ood_val is not None and not math.isnan(float(ood_val)) and float(ood_val) > float(max_ood):
            reasons.append(f"OOD_STRIP_HIGH_{float(ood_val):.3f}")

    return {
        "alarm": len(reasons) > 0,
        "reasons": reasons,
        "altitude_band": aband,
    }


ConformalModelG1 = ConformalModel


def fit_sigma_sys_g1(alarm_free_cal_records: list[dict[str, Any]]) -> dict[str, float]:
    """Fit systematic error scale sigma_sys (in radians) per camera set from calibration frames."""
    from xvs.integrity.conformal import fit_sigma_sys
    return fit_sigma_sys(alarm_free_cal_records)


def fit_conformal_model_g1(
    alarm_free_cal_records: list[dict[str, Any]],
    alphas: Sequence[float] = ALPHA_LEVELS,
) -> dict[str, Any]:
    """Fit both standard/Mondrian ConformalModel and altitude-binned Mondrian quantiles (Item 2)."""
    from xvs.integrity.conformal import fit_conformal_model

    base_model = fit_conformal_model(alarm_free_cal_records, alphas=alphas)
    alt_mondrian_scores: dict[str, list[float]] = {}

    for r in alarm_free_cal_records:
        if r.get("fault_type", "none") not in ("none", None):
            continue
        score = r.get("nonconformity_score", float("nan"))
        if math.isnan(score) or math.isinf(score):
            continue
        cset = r.get("cam_set", "all")
        if cset == "horizontal":
            cset = "FRBL"
        cond = str(r.get("condition", "degraded"))
        grp = "clean" if cond in ("clean", "nominal") else "degraded"
        aband = classify_altitude_band(r.get("h_meas_m", 0.0))  # default-ok: optional fallback default
        key = f"{cset}:{grp}:{aband}"
        alt_mondrian_scores.setdefault(key, []).append(float(score))

    alt_mondrian_quantiles: dict[str, dict[str, float | None]] = {}
    n_cal_alt_mondrian: dict[str, int] = {}
    for key, s_list in sorted(alt_mondrian_scores.items()):
        n_cal_alt_mondrian[key] = len(s_list)
        alt_mondrian_quantiles[key] = {}
        for a in alphas:
            a_key = f"{a:.4f}".rstrip("0")
            q_val = compute_conformal_quantile(s_list, a)
            base_slice = ":".join(key.split(":")[:2])
            q_fallback = base_model.mondrian_quantiles.get(base_slice, {}).get(a_key, float("inf"))
            if math.isinf(q_val):
                q_val = float("inf") if q_fallback is None else float(q_fallback)
            elif a <= 0.001 and q_fallback is not None and not math.isinf(float(q_fallback)):
                # At alpha=0.001, a single altitude band has <200 unique trajectory frames;
                # regularize against the trajectory-pooled Mondrian quantile to prevent finite-trajectory undercoverage.
                q_val = max(float(q_val), 0.70 * float(q_fallback))
            alt_mondrian_quantiles[key][a_key] = None if math.isinf(q_val) else float(q_val)

    return {
        "base_model": base_model,
        "altitude_mondrian_quantiles": alt_mondrian_quantiles,
        "n_cal_altitude_mondrian": n_cal_alt_mondrian,
    }


def calibrate_boresight_pl_sensitivity_g1(
    cal_boresight_records: list[dict[str, Any]],
    g1_conformal_bundle: dict[str, Any],
    alphas: Sequence[float] = ALPHA_LEVELS,
    thermal_drift_fraction: float = 0.35,
) -> dict[str, Any]:
    """Calibrate RSS boresight calibration uncertainty sensitivity factors c_{alpha, cam_set}
    strictly on the calibration split (Item 3, B11-F3 & B11-R2-m2).

    At sigma_cal == 0.0, protection levels equal the unmodified altitude-Mondrian PL_0 (s0 = 1.0).
    For sigma_cal > 0.0, propagates rig boresight uncertainty (static mounting + thermal drift)
    into the Mondrian conformal protection level:
        PL_alpha(sigma_cal) = sqrt(PL_{alpha, 0}^2 + (c_{alpha, cam_set} * sigma_{cal, eff})^2)
    where sigma_{cal, eff} = sigma_cal * sqrt(1 + 0.5 * thermal_drift_fraction^2).
    """
    therm_scale = math.sqrt(1.0 + 0.5 * float(thermal_drift_fraction) ** 2)
    by_cam_set: dict[str, dict[str, Any]] = {}

    for cset in ("F", "FRBL", "all"):
        sub_c = [
            r for r in cal_boresight_records
            if r.get("cam_set") == cset and float(r.get("sigma_cal_deg", 0.0)) > 1e-6 and r.get("status") == "CONVERGED"  # default-ok: optional integrity record/sensitivity fallback
        ]
        c_factors_by_alpha: dict[str, float] = {}
        for alpha in alphas:
            a_key = f"{alpha:.4f}".rstrip("0")
            norm_excess: list[float] = []
            for r in sub_c:
                sig_cal = float(r["sigma_cal_deg"])
                sig_eff = sig_cal * therm_scale
                sig_hat_rad = float(r["sigma_hat_urad"]) * 1e-6
                pl_0 = compute_pl_deg_altitude_mondrian(
                    g1_conformal_bundle,
                    sig_hat_rad,
                    alpha,
                    cset,
                    str(r.get("condition", "degraded")),
                    float(r.get("h_meas_m", 0.0)),  # default-ok: optional integrity record/sensitivity fallback
                )
                err_deg = float(r["total_error_deg"])
                excess_sq = max(0.0, err_deg * err_deg - pl_0 * pl_0)
                norm_excess.append(math.sqrt(excess_sq) / max(1e-9, sig_eff))

            chi3_floor = float(math.sqrt(stats.chi2.ppf(1.0 - alpha, df=3)) * 1.35)
            if norm_excess:
                q_c = compute_conformal_quantile(norm_excess, alpha)
                if math.isinf(q_c):
                    n_c = len(norm_excess)
                    max_c = float(np.max(norm_excess))
                    p99_c = float(np.percentile(norm_excess, 99.0))
                    p95_c = float(np.percentile(norm_excess, 95.0))
                    slope_c = max(max_c - p99_c, p99_c - p95_c, 0.25)
                    q_c = (max_c + slope_c * math.log10(max(1.0, (1.0 / alpha) / max(1, n_c)))) * (
                        1.0 + 1.25 / math.sqrt(max(1, n_c))
                    )
                else:
                    n_c = len(norm_excess)
                    q_c = float(q_c) * (1.0 + 0.5 / math.sqrt(max(1.0, n_c * alpha)))
                q_c = max(float(q_c), chi3_floor)
            else:
                q_c = max(3.5, chi3_floor)
            c_factors_by_alpha[a_key] = float(q_c)

        sorted_alphas = sorted(alphas, reverse=True)
        running_max_c = 0.0
        for alpha in sorted_alphas:
            a_key = f"{alpha:.4f}".rstrip("0")
            running_max_c = max(running_max_c, c_factors_by_alpha[a_key])
            c_factors_by_alpha[a_key] = float(running_max_c)

        d27_ratios = [
            float(r["camera_consistency_deg"]) / max(1e-9, float(r["sigma_cal_deg"]) * therm_scale)
            for r in sub_c
            if r.get("camera_consistency_deg") is not None and not math.isnan(float(r["camera_consistency_deg"]))
        ]
        rms_ratios = [
            float(r["rms_residual_urad"]) / max(1e-9, float(r["sigma_cal_deg"]) * therm_scale)
            for r in sub_c
            if not math.isnan(float(r.get("rms_residual_urad", float("nan"))))
        ]
        by_cam_set[cset] = {
            "c_alpha": c_factors_by_alpha,
            "d27_loo_factor": float(np.percentile(d27_ratios, 99.5) * 1.15) if d27_ratios else 6.0,
            "rms_urad_per_deg": float(np.percentile(rms_ratios, 99.5) * 1.15) if rms_ratios else 45000.0,
            "n_calibration_records": len(sub_c),
        }

    return {
        "thermal_drift_fraction": float(thermal_drift_fraction),
        "thermal_scale": float(therm_scale),
        "by_cam_set": by_cam_set,
    }


def compute_pl_deg_altitude_mondrian(
    g1_conformal_bundle: dict[str, Any],
    sigma_hat_rad: float,
    alpha: float,
    cam_set: str,
    condition: str,
    h_meas_m: float,
    sigma_cal_deg: float = 0.0,
    boresight_sensitivity: dict[str, Any] | None = None,
) -> float:
    """Compute protection level in degrees using the altitude-conditioned Mondrian quantile,
    with RSS boresight calibration uncertainty propagation when sigma_cal_deg > 0 (Items 2 & 3, B11-R2-m2).
    """
    base_model: ConformalModel = g1_conformal_bundle["base_model"]
    alt_q_map = g1_conformal_bundle["altitude_mondrian_quantiles"]
    if cam_set == "horizontal":
        cam_set = "FRBL"
    grp = "clean" if condition in ("clean", "nominal") else "degraded"
    aband = classify_altitude_band(h_meas_m)
    key = f"{cam_set}:{grp}:{aband}"
    a_key = f"{alpha:.4f}".rstrip("0")
    q_val = alt_q_map.get(key, {}).get(a_key)
    if q_val is None:
        pl_0 = base_model.compute_pl_deg(sigma_hat_rad, alpha, cam_set=cam_set, condition=condition, mondrian=True)
    else:
        pl_0 = compute_protection_level_deg(math.degrees(sigma_hat_rad), float(q_val))

    if sigma_cal_deg > 0.0:
        sens = boresight_sensitivity or g1_conformal_bundle.get("boresight_sensitivity", {})
        therm_scale = float(sens.get("thermal_scale", math.sqrt(1.0 + 0.5 * 0.35 * 0.35)))
        c_alpha = float(
            sens.get("by_cam_set", {}).get(cam_set, {}).get("c_alpha", {}).get(a_key, 3.5)  # default-ok: optional integrity record/sensitivity fallback
        )
        sig_eff = float(sigma_cal_deg) * therm_scale
        return float(math.hypot(pl_0, c_alpha * sig_eff))

    return float(pl_0)


# =============================================================================
# ITEM 3: Boresight Calibration Residuals (`sigma_cal in {0.0, 0.02, 0.05} deg`)
# =============================================================================


def perturb_rig_boresight(
    base_rig: Rig,
    sigma_cal_deg: float,
    seed: int = 20261107,
    t_s: float = 0.0,
    thermal_drift_fraction: float = 0.35,
    thermal_period_s: float = 120.0,
) -> tuple[Rig, dict[str, dict[str, float]]]:
    """Perturb camera extrinsics (`r_body_cam`) with static mounting + slow thermal drift residuals (Item 3)."""
    if sigma_cal_deg <= 0.0:
        zero_info = {
            m.id: {"static_norm_deg": 0.0, "thermal_norm_deg": 0.0, "total_norm_deg": 0.0}
            for m in base_rig.cameras
        }
        return base_rig, zero_info

    rng = np.random.default_rng(seed)
    sigma_rad = math.radians(float(sigma_cal_deg))
    new_mounts: list[CameraMount] = []
    perturbation_log: dict[str, dict[str, float]] = {}

    for m in base_rig.cameras:
        omega_static = rng.normal(0.0, sigma_rad, size=3)
        drift_dir = rng.normal(0.0, 1.0, size=3)
        drift_dir = drift_dir / max(1e-12, float(np.linalg.norm(drift_dir)))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        omega_therm = (
            thermal_drift_fraction
            * sigma_rad
            * math.sin(2.0 * math.pi * float(t_s) / max(1.0, thermal_period_s) + phase)
            * drift_dir
        )
        omega_tot = omega_static + omega_therm
        r_err = rotations.so3_exp(omega_tot)
        r_bc_pert = m.r_body_cam @ r_err

        new_mounts.append(
            CameraMount(
                id=m.id,
                name=m.name,
                mount=m.mount,
                lever_arm_m=m.lever_arm_m.copy(),
                r_body_cam=r_bc_pert,
                boresight_body=r_bc_pert @ np.array([0.0, 0.0, 1.0], dtype=np.float64),
                omega_cam_rad=omega_tot,
            )
        )
        perturbation_log[m.id] = {
            "static_norm_deg": float(math.degrees(np.linalg.norm(omega_static))),
            "thermal_norm_deg": float(math.degrees(np.linalg.norm(omega_therm))),
            "total_norm_deg": float(math.degrees(np.linalg.norm(omega_tot))),
        }

    pert_rig = Rig(
        rig_id=f"{base_rig.rig_id}_sigma{sigma_cal_deg:.3f}",
        cameras=new_mounts,
        is_nominal=False,
        seed=seed,
        width=base_rig.width,
        height=base_rig.height,
        fov_h_deg=base_rig.fov_h_deg,
    )
    return pert_rig, perturbation_log


# =============================================================================
# ITEM 4: Single-Camera Mechanical Misalignment Fault & D27 Fault Isolation
# =============================================================================


def inject_single_cam_misalignment(
    batch_or_rig: LimbRayBatch | Rig,
    target_cam: str | None = None,
    step_pitch_deg: float | None = None,
    step_roll_deg: float | None = None,
    rng: np.random.Generator | None = None,
    min_step_deg: float = 0.35,
    max_step_deg: float = 1.25,
    cam_id: str | None = None,
    magnitude_deg: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Inject a single-camera mechanical boresight step misalignment fault into a LimbRayBatch or Rig (Item 4)."""
    eff_cam = cam_id or target_cam or "F"
    if step_pitch_deg is None or step_roll_deg is None:
        if magnitude_deg is not None:
            ang = float(rng.uniform(0.0, 2.0 * math.pi)) if rng is not None else 0.5
            dp_deg = float(magnitude_deg) * math.cos(ang)
            dr_deg = float(magnitude_deg) * math.sin(ang)
        elif rng is not None:
            mag = float(rng.uniform(min_step_deg, max_step_deg))
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            dp_deg = mag * math.cos(ang)
            dr_deg = mag * math.sin(ang)
        else:
            dp_deg = 0.65
            dr_deg = -0.35
    else:
        dp_deg = float(step_pitch_deg)
        dr_deg = float(step_roll_deg)

    step_norm_deg = float(math.hypot(dp_deg, dr_deg))
    omega_body = np.array([math.radians(dr_deg), math.radians(dp_deg), 0.0], dtype=np.float64)
    r_shift = rotations.so3_exp(omega_body)

    if isinstance(batch_or_rig, Rig):
        new_mounts: list[CameraMount] = []
        for m in batch_or_rig.cameras:
            if m.id == eff_cam:
                r_bc_new = r_shift @ m.r_body_cam
                new_mounts.append(
                    CameraMount(
                        id=m.id,
                        name=m.name,
                        mount=m.mount,
                        lever_arm_m=m.lever_arm_m.copy(),
                        r_body_cam=r_bc_new,
                        boresight_body=r_bc_new @ np.array([0.0, 0.0, 1.0], dtype=np.float64),
                        omega_cam_rad=omega_body,
                    )
                )
            else:
                new_mounts.append(m)
        mis_rig = Rig(
            rig_id=f"{batch_or_rig.rig_id}_misalign_{eff_cam}",
            cameras=new_mounts,
            is_nominal=False,
            width=batch_or_rig.width,
            height=batch_or_rig.height,
            fov_h_deg=batch_or_rig.fov_h_deg,
        )
        log = {
            "fault_type": "single_cam_misalignment",
            "target_cam": eff_cam,
            "cam_id": eff_cam,
            "step_pitch_deg": dp_deg,
            "step_roll_deg": dr_deg,
            "step_total_deg": step_norm_deg,
            "magnitude_deg": step_norm_deg,
        }
        return mis_rig, log

    batch = batch_or_rig
    d_new = np.copy(batch.d_body)
    n_shifted = 0
    for i, cid in enumerate(batch.cam_ids):
        if cid == eff_cam:
            d_new[i] = r_shift @ d_new[i]
            d_new[i] /= max(1e-12, float(np.linalg.norm(d_new[i])))
            n_shifted += 1

    new_batch = LimbRayBatch(
        d_new,
        np.copy(batch.lever_arm_body),
        list(batch.cam_ids),
        weights=np.copy(batch.weights),
        uvs=np.copy(batch.uvs),
    )
    log = {
        "fault_type": "single_cam_misalignment",
        "target_cam": eff_cam,
        "cam_id": eff_cam,
        "step_pitch_deg": dp_deg,
        "step_roll_deg": dr_deg,
        "step_total_deg": step_norm_deg,
        "magnitude_deg": step_norm_deg,
        "n_rays_shifted": int(n_shifted),
        "d_body_sha256": compute_sha256(d_new),
    }
    return new_batch, log


def isolate_and_exclude_faulty_camera(
    full_fit: dict[str, Any],
    batch: LimbRayBatch | None = None,
    rig: Rig | None = None,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    h_meas_m: float = 0.0,
    yaw_rad: float = 0.0,
    thresholds: MonitorThresholdsG1 | MonitorThresholds | dict[str, Any] | None = None,
    cam_set_name: str = "FRBL",
    frame_k: int = 0,
    min_active_cams: int = 3,
    min_inliers_per_cam: int = 10,
    inlier_batch: LimbRayBatch | None = None,
    d27_threshold_deg: float = 0.15,
) -> dict[str, Any]:
    """D27 Leave-One-Camera-Out Fault Isolation and Exclusion (`F_isolated`, Item 4)."""
    if batch is None:
        batch = inlier_batch
    if batch is None or rig is None:
        raise ValueError("batch/inlier_batch and rig are required for isolate_and_exclude_faulty_camera")

    if thresholds is None:
        th = MonitorThresholdsG1(
            far_target=0.01,
            max_rms_residual_urad=4500.0,
            max_condition_number=25.0,
            min_inlier_fraction=0.30,
            min_inlier_count=60,
            max_leave_one_out_deg=float(d27_threshold_deg),
        )
    elif isinstance(thresholds, dict):
        th = MonitorThresholdsG1.from_dict(thresholds)
    elif isinstance(thresholds, MonitorThresholds):
        th = MonitorThresholdsG1.from_dict(thresholds.to_dict())
    else:
        th = thresholds

    inlier_idx = full_fit.get("inliers", np.arange(batch.n_rays))
    inlier_batch = LimbRayBatch(
        batch.d_body[inlier_idx],
        batch.lever_arm_body[inlier_idx],
        [batch.cam_ids[i] for i in inlier_idx],
        weights=batch.weights[inlier_idx],
        uvs=batch.uvs[inlier_idx],
    )
    cc_full = camera_consistency(full_fit, inlier_batch, rig, lat_deg, lon_deg, h_meas_m, yaw_rad)

    cam_counts: dict[str, int] = {}
    for cid in inlier_batch.cam_ids:
        cam_counts[cid] = cam_counts.get(cid, 0) + 1  # default-ok: frequency counter accumulator
    valid_cams = [c for c, cnt in cam_counts.items() if cnt >= min_inliers_per_cam]

    if cam_set_name == "F" or len(valid_cams) < min_active_cams or cc_full.arg_max_camera is None:
        return {
            "isolated": False,
            "excluded_camera": None,
            "remaining_cameras": valid_cams,
            "refit": None,
            "isolation_attempted": False,
            "isolation_succeeded": False,
            "isolation_status": "NOT_ELIGIBLE",
            "isolated_camera": None,
            "pre_isolation_d27_deg": float(cc_full.max_angle_deg),
            "post_isolation_d27_deg": float("nan"),
            "post_isolation_fit": None,
            "alarm_after_isolation": True,
        }

    suspect_cam = cc_full.arg_max_camera
    keep_mask = np.array([cid != suspect_cam for cid in batch.cam_ids], dtype=bool)
    reduced_batch = LimbRayBatch(
        batch.d_body[keep_mask],
        batch.lever_arm_body[keep_mask],
        [cid for cid, m in zip(batch.cam_ids, keep_mask) if m],
        weights=batch.weights[keep_mask],
        uvs=batch.uvs[keep_mask],
    )
    rem_cams = sorted(set(reduced_batch.cam_ids))

    fit_red = fit_pipeline_attitude_g1(
        lat_deg, lon_deg, h_meas_m, yaw_rad, reduced_batch, frame_k=frame_k
    )
    if fit_red["status"] != "CONVERGED" or len(fit_red.get("inliers", [])) < 20:
        return {
            "isolated": False,
            "excluded_camera": suspect_cam,
            "remaining_cameras": rem_cams,
            "refit": fit_red,
            "isolation_attempted": True,
            "isolation_succeeded": False,
            "isolation_status": "REDUCED_FIT_FAILED",
            "isolated_camera": suspect_cam,
            "pre_isolation_d27_deg": float(cc_full.max_angle_deg),
            "post_isolation_d27_deg": float("nan"),
            "post_isolation_fit": fit_red,
            "alarm_after_isolation": True,
        }

    red_inlier_idx = fit_red["inliers"]
    red_inlier_batch = LimbRayBatch(
        reduced_batch.d_body[red_inlier_idx],
        reduced_batch.lever_arm_body[red_inlier_idx],
        [reduced_batch.cam_ids[i] for i in red_inlier_idx],
        weights=reduced_batch.weights[red_inlier_idx],
        uvs=reduced_batch.uvs[red_inlier_idx],
    )
    cc_red = camera_consistency(fit_red, red_inlier_batch, rig, lat_deg, lon_deg, h_meas_m, yaw_rad)

    cov_rad2 = fit_red["covariance_rad2"]
    tr_cov = float(np.trace(cov_rad2)) if cov_rad2 is not None else float("nan")
    sigma_hat_urad = max(1e-6, math.sqrt(max(0.0, tr_cov))) * 1e6 if not math.isnan(tr_cov) else float("nan")
    cond_num = float(np.linalg.cond(cov_rad2)) if cov_rad2 is not None else float("nan")

    n_total_cams = max(3, len(valid_cams))
    scale_cams = float(n_total_cams - 1) / float(n_total_cams)
    scaled_inlier_cnt = int(round(fit_red["inlier_count"] / scale_cams))

    red_rec = {
        "status": fit_red["status"],
        "cam_set": cam_set_name,
        "h_meas_m": h_meas_m,
        "rms_residual_urad": float(fit_red["rms_residual_urad"]),
        "condition_number": cond_num,
        "inlier_fraction": float(fit_red["inlier_count"]) / max(1, reduced_batch.n_rays),
        "inlier_count": scaled_inlier_cnt,
        "camera_consistency_deg": None if math.isnan(cc_red.max_angle_deg) else float(cc_red.max_angle_deg),
        "leave_one_out_consistency_deg": None if math.isnan(cc_red.max_angle_deg) else float(cc_red.max_angle_deg),
        "sigma_hat_urad": sigma_hat_urad,
    }
    ev_red = evaluate_monitor_alarm_g1(red_rec, th, use_altitude_conditioned_d27=True, use_ood_layer=False)
    succeeded = not ev_red["alarm"]

    return {
        "isolated": bool(succeeded),
        "excluded_camera": suspect_cam,
        "remaining_cameras": rem_cams,
        "refit": fit_red,
        "isolation_attempted": True,
        "isolation_succeeded": bool(succeeded),
        "isolation_status": "F_isolated" if succeeded else "ISOLATION_REJECTED_BY_M1",
        "isolated_camera": suspect_cam,
        "pre_isolation_d27_deg": float(cc_full.max_angle_deg),
        "post_isolation_d27_deg": float(cc_red.max_angle_deg),
        "post_isolation_fit": fit_red,
        "post_isolation_record": red_rec,
        "alarm_after_isolation": not succeeded,
    }
