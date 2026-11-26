"""4-DoF Horizon-Constrained MAP-PnPL & 5 Hz Approach EKF for XVS-Land (DR-03, SP2 §3.4).

Strictly obeys CONVENTIONS.md v1.0:
  - Frames: local runway-threshold NED (x=North, y=East, z=Down), BODY (FRD), CAM (OpenCV).
  - Zero internal ENU.
  - Known roll `phi_rad` and pitch `theta_rad` from horizon/INS reduce PnPL to 4-DoF
    `[t_n, t_e, t_d, psi_rad]`, combining anisotropic covariance-weighted keypoints and
    continuous Hesse keylines via damped Gauss-Newton / Levenberg-Marquardt.
  - 5 Hz Approach EKF tracks 11 states `[t_w(3), v_w(3), psi(1), b_a(3), b_gz(1)]` with
    fail-closed Mahalanobis innovation gating and Joseph-form covariance update.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from xvs.geo.rotations import euler_to_r
from xvs.v2.xvs_land.raim_landing import compute_hybrid_conformal_raim_pl

# Forward camera R_body_cam from docs/xvs/spec/rig_nominal.json (F camera)
R_BODY_CAM_FORWARD: np.ndarray = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PnplSolution:
    """Result of 4-DoF Horizon-Constrained MAP-PnPL solver."""

    pos_ned_m: np.ndarray       # (3,) camera position in runway-threshold NED
    yaw_rad: float              # true heading psi (rad)
    cov_4x4: np.ndarray         # (4, 4) posterior covariance (J^T W J + P_prior^-1)^-1
    jacobian_obs: np.ndarray    # (M, 4) observation Jacobian at solution (for V-RAIM)
    weight_diag: np.ndarray     # (M,) inverse-variance weights for V-RAIM slopes
    rms_reproj_px: float
    chi2: float
    dof: int
    converged: bool
    iterations: int


@dataclass(frozen=True)
class EkfUpdateResult:
    """Result of 5 Hz Approach EKF PnPL measurement update."""

    accepted: bool
    mahalanobis_sq: float
    gate_chi2: float
    pos_ned_m: np.ndarray
    vel_ned_mps: np.ndarray
    yaw_rad: float


@dataclass(frozen=True)
class ApproachEkfTrajectoryResult:
    """5 Hz multi-frame ApproachEkf5Hz trajectory evaluation metrics (strictly NED)."""

    num_frames: int
    accepted_updates: int
    rejected_updates: int
    ekf_pos_rmse_m: float
    ekf_horiz_rmse_m: float
    ekf_vert_rmse_m: float
    single_frame_pnpl_rmse_m: float
    v_hpl_median_m: float
    v_vpl_median_m: float
    v_hpl_p95_m: float
    v_vpl_p95_m: float
    cat1_availability: float
    hmi_count: int
    start_range_m: float
    end_range_m: float


def _wrap_angle_rad(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _safe_spd_inv_cholesky_2x2(
    cov_2x2: np.ndarray,
    min_eig: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Return `(inv_L, var_u, var_v)` for a 2x2 covariance matrix, guarding non-finite/non-PSD inputs."""
    c_raw = np.asarray(cov_2x2, dtype=np.float64)
    if c_raw.shape != (2, 2) or not np.all(np.isfinite(c_raw)):
        inv_l = np.eye(2, dtype=np.float64) * 1e-3
        return inv_l, 1e6, 1e6
    c_mat = 0.5 * (c_raw + c_raw.T)
    c00 = float(c_mat[0, 0]) + min_eig
    c01 = float(c_mat[0, 1])
    c11 = float(c_mat[1, 1]) + min_eig
    det = c00 * c11 - c01 * c01
    if c00 <= min_eig or c11 <= min_eig or det <= (min_eig * min_eig):
        eigvals, eigvecs = np.linalg.eigh(c_mat)
        clipped = np.maximum(eigvals, float(min_eig))
        c_psd = (eigvecs * clipped[None, :]) @ eigvecs.T
        c00 = float(c_psd[0, 0])
        c01 = float(c_psd[0, 1])
        c11 = float(c_psd[1, 1])
    l00 = math.sqrt(max(c00, min_eig))
    l10 = c01 / l00
    l11_sq = c11 - l10 * l10
    l11 = math.sqrt(max(l11_sq, min_eig))
    inv_l00 = 1.0 / l00
    inv_l11 = 1.0 / l11
    inv_l = np.array([[inv_l00, 0.0], [-l10 * inv_l00 * inv_l11, inv_l11]], dtype=np.float64)
    return inv_l, max(c00, min_eig), max(c11, min_eig)


def build_ksfo_28l_reference_features(
    runway_width_m: float = 61.0,
    runway_length_m: float = 3471.0,
    true_heading_deg: float = 297.7,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build 12 canonical 3D runway keypoints in runway-threshold NED + 5 keyline endpoint pairs.

    Origin (0, 0, 0) = KSFO 28L threshold centerline on WGS-84 tangent plane (`Down = 0`).
    12 keypoints span threshold corners/center, aiming-point bars (300 m), touchdown zone (600 m),
    midfield (1500 m), and far-end corners/center.
    5 keylines: left edge, right edge, centerline, threshold bar, aiming-point bar.
    """
    hdg = math.radians(true_heading_deg)
    fwd = np.array([math.cos(hdg), math.sin(hdg), 0.0], dtype=np.float64)
    right = np.array([-math.sin(hdg), math.cos(hdg), 0.0], dtype=np.float64)
    hw = 0.5 * runway_width_m

    stations_m = (0.0, 300.0, 900.0, runway_length_m)
    pts: list[np.ndarray] = []
    for s in stations_m:
        center = fwd * s
        pts.append(center - right * hw)  # left edge
        pts.append(center)               # centerline
        pts.append(center + right * hw)  # right edge

    pts_arr = np.vstack(pts)  # (12, 3)
    # 5 keylines indexed into pts_arr:
    # 0: left edge (0 -> 9), 1: right edge (2 -> 11), 2: centerline (1 -> 10),
    # 3: threshold bar (0 -> 2), 4: aiming bar (3 -> 5)
    keyline_pairs = [(0, 9), (2, 11), (1, 10), (0, 2), (3, 5)]
    return pts_arr, keyline_pairs


def project_runway_points_ned(
    pts_ned: np.ndarray,
    cam_pos_ned: np.ndarray,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    r_body_cam: np.ndarray = R_BODY_CAM_FORWARD,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D runway-NED points into OpenCV CAM pinhole pixels `(u, v)` (`CONVENTIONS.md` §5)."""
    r_ned_body = euler_to_r(yaw_rad, pitch_rad, roll_rad)
    r_ned_cam = r_ned_body @ r_body_cam
    rel_ned = pts_ned - cam_pos_ned[None, :]  # (K, 3)
    p_cam = rel_ned @ r_ned_cam  # equivalent to (r_ned_cam.T @ rel_ned.T).T
    z = p_cam[:, 2]
    z_safe = np.where(z > 1e-6, z, 1e-6)
    u = fx * (p_cam[:, 0] / z_safe) + cx
    v = fy * (p_cam[:, 1] / z_safe) + cy
    return np.stack([u, v], axis=-1), z


def compute_hesse_line_from_points(p0_uv: np.ndarray, p1_uv: np.ndarray) -> np.ndarray:
    """Compute canonical Hesse line `(rho, cos_theta, sin_theta)` passing through two 2D image points."""
    du = float(p1_uv[0] - p0_uv[0])
    dv = float(p1_uv[1] - p0_uv[1])
    norm = math.hypot(du, dv)
    if norm < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    nx = -dv / norm
    ny = du / norm
    rho = nx * float(p0_uv[0]) + ny * float(p0_uv[1])
    if rho < 0.0:
        rho, nx, ny = -rho, -nx, -ny
    return np.array([rho, nx, ny], dtype=np.float64)


def solve_4dof_map_pnpl(
    pts_ned: np.ndarray,
    obs_uv: np.ndarray,
    obs_cov_2x2: np.ndarray,
    *,
    pitch_rad: float,
    roll_rad: float,
    init_pos_ned: np.ndarray,
    init_yaw_rad: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    keyline_pairs: list[tuple[int, int]] | None = None,
    obs_keylines_hesse: np.ndarray | None = None,
    keyline_sigma_px: float = 1.5,
    keyline_sigmas_px: np.ndarray | None = None,
    huber_delta_sigma: float = 3.0,
    prior_pos_ned: np.ndarray | None = None,
    prior_yaw_rad: float | None = None,
    prior_std_4: tuple[float, float, float, float] = (500.0, 500.0, 200.0, 0.5),
    max_iters: int = 25,
    r_body_cam: np.ndarray = R_BODY_CAM_FORWARD,
) -> PnplSolution:
    """Solve 4-DoF Horizon-Constrained MAP-PnPL `[t_n, t_e, t_d, psi]` via Huber-IRLS LM.

    Supports optional per-keyline pixel standard deviations `keyline_sigmas_px` and robust
    Huber IRLS reweighting (`huber_delta_sigma`) with redescending Cauchy polish for gross
    single-feature faults so corrupted keylines/keypoints cannot hijack the 4-DoF pose.
    """
    pts = np.asarray(pts_ned, dtype=np.float64)
    uv_obs = np.asarray(obs_uv, dtype=np.float64)
    covs = np.asarray(obs_cov_2x2, dtype=np.float64)
    n_kp = int(pts.shape[0])

    # Precompute Cholesky inverse factors for anisotropic 2x2 keypoint covariances (PSD/NaN guarded)
    inv_l_arr = np.zeros((n_kp, 2, 2), dtype=np.float64)
    kp_w_diag = np.zeros((2 * n_kp,), dtype=np.float64)
    for i in range(n_kp):
        inv_l, var_u, var_v = _safe_spd_inv_cholesky_2x2(covs[i], min_eig=1e-6)
        inv_l_arr[i] = inv_l
        kp_w_diag[2 * i] = 1.0 / var_u
        kp_w_diag[2 * i + 1] = 1.0 / var_v

    has_lines = keyline_pairs is not None and obs_keylines_hesse is not None and len(keyline_pairs) > 0
    n_kl = len(keyline_pairs) if (has_lines and keyline_pairs is not None) else 0
    lines_hesse = np.asarray(obs_keylines_hesse, dtype=np.float64) if has_lines else np.zeros((0, 3))
    default_line_sig = max(float(keyline_sigma_px), 1e-3)
    if has_lines and keyline_pairs is not None:
        if keyline_sigmas_px is not None:
            kl_sig_arr = np.asarray(keyline_sigmas_px, dtype=np.float64).reshape(-1)
            if kl_sig_arr.size < n_kl:
                raise ValueError(
                    f"keyline_sigmas_px length ({kl_sig_arr.size}) < number of keylines ({n_kl})"
                )
            kl_sigs = np.where(
                np.isfinite(kl_sig_arr[:n_kl]),
                np.maximum(kl_sig_arr[:n_kl], 1e-3),
                default_line_sig,
            ).astype(np.float64)
        else:
            kl_sigs = np.full(n_kl, default_line_sig, dtype=np.float64)
        inv_line_sigs = 1.0 / kl_sigs
        kl_i0 = np.array([int(p[0]) for p in keyline_pairs], dtype=np.intp)
        kl_i1 = np.array([int(p[1]) for p in keyline_pairs], dtype=np.intp)
        kl_rho = lines_hesse[:n_kl, 0].astype(np.float64, copy=False)
        kl_cos = lines_hesse[:n_kl, 1].astype(np.float64, copy=False)
        kl_sin = lines_hesse[:n_kl, 2].astype(np.float64, copy=False)
    else:
        inv_line_sigs = np.zeros(0, dtype=np.float64)
        kl_i0 = np.zeros(0, dtype=np.intp)
        kl_i1 = np.zeros(0, dtype=np.intp)
        kl_rho = np.zeros(0, dtype=np.float64)
        kl_cos = np.zeros(0, dtype=np.float64)
        kl_sin = np.zeros(0, dtype=np.float64)

    use_robust = math.isfinite(float(huber_delta_sigma)) and float(huber_delta_sigma) > 0.0
    delta = float(huber_delta_sigma) if use_robust else math.inf

    use_prior = prior_pos_ned is not None or prior_yaw_rad is not None
    x_prior = np.array(
        [
            float(init_pos_ned[0] if prior_pos_ned is None else prior_pos_ned[0]),
            float(init_pos_ned[1] if prior_pos_ned is None else prior_pos_ned[1]),
            float(init_pos_ned[2] if prior_pos_ned is None else prior_pos_ned[2]),
            float(init_yaw_rad if prior_yaw_rad is None else prior_yaw_rad),
        ],
        dtype=np.float64,
    )
    inv_p_prior = np.diag([1.0 / (s * s) for s in prior_std_4]) if use_prior else np.zeros((4, 4), dtype=np.float64)

    x = np.array([float(init_pos_ned[0]), float(init_pos_ned[1]), float(init_pos_ned[2]), float(init_yaw_rad)])

    # Precompute constant pitch/roll -> camera rotation (yaw varies per LM step)
    r_pr_cam = euler_to_r(0.0, float(pitch_rad), float(roll_rad)) @ np.asarray(r_body_cam, dtype=np.float64)
    r_pr_row0 = r_pr_cam[0]
    r_pr_row1 = r_pr_cam[1]
    r_pr_row2 = r_pr_cam[2]
    fx_f, fy_f, cx_f, cy_f = float(fx), float(fy), float(cx), float(cy)
    n_groups = n_kp + n_kl

    def _project_fast(pos_ned: np.ndarray, yaw: float) -> np.ndarray:
        cy_y = math.cos(yaw)
        sy_y = math.sin(yaw)
        r_ned_cam = np.empty((3, 3), dtype=np.float64)
        r_ned_cam[0] = cy_y * r_pr_row0 - sy_y * r_pr_row1
        r_ned_cam[1] = sy_y * r_pr_row0 + cy_y * r_pr_row1
        r_ned_cam[2] = r_pr_row2
        p_cam = (pts - pos_ned) @ r_ned_cam
        z_safe = np.maximum(p_cam[:, 2], 1e-6)
        inv_z = 1.0 / z_safe
        uv = np.empty((n_kp, 2), dtype=np.float64)
        uv[:, 0] = fx_f * (p_cam[:, 0] * inv_z) + cx_f
        uv[:, 1] = fy_f * (p_cam[:, 1] * inv_z) + cy_f
        return uv

    def _eval_whitened_residuals(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uv_pred = _project_fast(state[:3], float(state[3]))
        diff = uv_pred - uv_obs  # (n_kp, 2)
        res_raw_px = diff.reshape(-1)
        if n_kp > 0:
            # Vectorized 2x2 Cholesky whitening: w_diff[i] = inv_l_arr[i] @ diff[i]
            w_kp = np.einsum("nij,nj->ni", inv_l_arr, diff).reshape(-1)
        else:
            w_kp = np.zeros(0, dtype=np.float64)

        if has_lines:
            d0 = (kl_cos * uv_pred[kl_i0, 0] + kl_sin * uv_pred[kl_i0, 1] - kl_rho) * inv_line_sigs
            d1 = (kl_cos * uv_pred[kl_i1, 0] + kl_sin * uv_pred[kl_i1, 1] - kl_rho) * inv_line_sigs
            res_white = np.empty((2 * n_groups,), dtype=np.float64)
            res_white[: 2 * n_kp] = w_kp
            res_white[2 * n_kp :: 2] = d0
            res_white[2 * n_kp + 1 :: 2] = d1
            return res_white, res_raw_px
        return w_kp, res_raw_px

    def _group_norms_and_irls(
        r_vec: np.ndarray,
        mode: str = "huber",
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Vectorized grouped 2D residual norms, M-estimator objective, and per-element IRLS weights."""
        if not use_robust or n_groups == 0:
            w_elem = np.ones_like(r_vec)
            return w_elem, 0.5 * float(r_vec @ r_vec), np.zeros(n_groups, dtype=np.float64)

        pairs = r_vec.reshape(n_groups, 2)
        norms = np.hypot(pairs[:, 0], pairs[:, 1])
        inlier = norms <= delta
        m_safe = np.where(inlier, 1.0, np.maximum(norms, 1e-12))

        if mode == "tukey":
            # Exact C1-smooth Huber-to-Tukey redescending M-estimator:
            # quadratic for m <= delta, smoothly decays to w=0 at c = 2.5*delta (zero pull from gross faults)
            c_cut = 2.5 * delta
            span = c_cut - delta
            mid = (~inlier) & (norms < c_cut)
            t = np.where(mid, (norms - delta) / span, 0.0)
            one_m_t2 = 1.0 - t * t
            w_grp = np.where(
                inlier,
                1.0,
                np.where(mid, (delta / m_safe) * (one_m_t2 * one_m_t2), 0.0),
            )
            t3 = t * t * t
            obj_arr = np.where(
                inlier,
                0.5 * norms * norms,
                np.where(
                    mid,
                    0.5 * delta * delta + delta * span * (t - (2.0 / 3.0) * t3 + 0.2 * t3 * t * t),
                    0.5 * delta * delta + (8.0 / 15.0) * delta * span,
                ),
            )
        else:
            w_grp = np.where(inlier, 1.0, delta / m_safe)
            obj_arr = np.where(inlier, 0.5 * norms * norms, delta * (norms - 0.5 * delta))

        w_elem = np.repeat(w_grp, 2)
        return w_elem, float(np.sum(obj_arr)), norms

    def _eval_jacobian(state: np.ndarray) -> np.ndarray:
        eps_vec = (1e-4, 1e-4, 1e-4, 1e-7)
        j_mat = np.empty((2 * n_groups, 4), dtype=np.float64)
        sp = state.copy()
        sm = state.copy()
        for col in range(4):
            h = eps_vec[col]
            s_orig = state[col]
            sp[col] = s_orig + h
            sm[col] = s_orig - h
            rp, _ = _eval_whitened_residuals(sp)
            rm, _ = _eval_whitened_residuals(sm)
            sp[col] = s_orig
            sm[col] = s_orig
            j_mat[:, col] = (rp - rm) / (2.0 * h)
        return j_mat

    def _run_lm_phase(x_init: np.ndarray, n_iters: int, mode: str) -> tuple[np.ndarray, bool, int]:
        x_cur = x_init.copy()
        lam = 1e-3
        conv = False
        used_iters = 0
        for it in range(1, n_iters + 1):
            used_iters = it
            r_vec, _ = _eval_whitened_residuals(x_cur)
            w_elem, obj_obs, _ = _group_norms_and_irls(r_vec, mode=mode)
            j_mat = _eval_jacobian(x_cur)
            dx_prior = np.array(
                [
                    x_cur[0] - x_prior[0],
                    x_cur[1] - x_prior[1],
                    x_cur[2] - x_prior[2],
                    _wrap_angle_rad(x_cur[3] - x_prior[3]),
                ]
            )

            jw = j_mat * w_elem[:, None]
            h_mat = jw.T @ j_mat + inv_p_prior
            g_vec = jw.T @ r_vec + inv_p_prior @ dx_prior
            cost_cur = obj_obs + 0.5 * float(dx_prior @ (inv_p_prior @ dx_prior))

            h_damped = h_mat + lam * np.diag(np.maximum(np.diag(h_mat), 1e-6))
            try:
                step = -np.linalg.solve(h_damped, g_vec)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            x_cand = x_cur + step
            x_cand[3] = _wrap_angle_rad(float(x_cand[3]))
            r_cand, _ = _eval_whitened_residuals(x_cand)
            _, obj_cand, _ = _group_norms_and_irls(r_cand, mode=mode)
            dx_p_cand = np.array(
                [
                    x_cand[0] - x_prior[0],
                    x_cand[1] - x_prior[1],
                    x_cand[2] - x_prior[2],
                    _wrap_angle_rad(x_cand[3] - x_prior[3]),
                ]
            )
            cost_cand = obj_cand + 0.5 * float(dx_p_cand @ (inv_p_prior @ dx_p_cand))

            if cost_cand <= cost_cur:
                x_cur = x_cand
                lam = max(1e-7, lam * 0.3)
                if float(np.linalg.norm(step[:3])) < 1e-5 and abs(float(step[3])) < 1e-7:
                    conv = True
                    break
            else:
                lam = min(1e6, lam * 5.0)
        return x_cur, conv, used_iters

    # Phase 1: Globally convergent convex Huber IRLS Levenberg-Marquardt
    x, converged, iters = _run_lm_phase(x, max_iters, mode="huber")

    # Phase 2: If a gross outlier (> 2.5*delta) remains while majority of features are tight inliers,
    # polish with exact C1 Huber-Tukey redescending tail so single corrupted keylines/keypoints exert 0 pull.
    final_mode = "huber"
    if use_robust:
        r_mid, _ = _eval_whitened_residuals(x)
        _, _, grp_norms = _group_norms_and_irls(r_mid, mode="huber")
        if grp_norms.size >= 4 and float(np.max(grp_norms)) > 2.5 * delta and float(np.median(grp_norms)) <= 2.0 * delta:
            x, conv_p2, iters_p2 = _run_lm_phase(x, max(10, max_iters // 2), mode="tukey")
            converged = bool(converged or conv_p2)
            iters += iters_p2
            final_mode = "tukey"

    r_final, raw_px = _eval_whitened_residuals(x)
    w_irls, _, _ = _group_norms_and_irls(r_final, mode=final_mode)
    sqrt_w = np.sqrt(np.maximum(w_irls, 0.0))
    j_raw = _eval_jacobian(x)
    j_eff = j_raw * sqrt_w[:, None]
    h_final = j_eff.T @ j_eff + inv_p_prior
    h_reg = 0.5 * (h_final + h_final.T) + 1e-9 * np.eye(4, dtype=np.float64)
    if not np.all(np.isfinite(h_reg)):
        cov_4x4 = np.diag([1e6, 1e6, 1e6, 1.0]).astype(np.float64)
    else:
        try:
            cov_4x4 = np.linalg.inv(h_reg)
        except np.linalg.LinAlgError:
            cov_4x4 = np.linalg.pinv(h_reg)
        cov_4x4 = 0.5 * (cov_4x4 + cov_4x4.T)

    if has_lines and keyline_pairs is not None:
        kl_w = np.repeat(inv_line_sigs * inv_line_sigs, 2)
        w_full_arr = np.concatenate([kp_w_diag, kl_w]) * w_irls
    else:
        w_full_arr = kp_w_diag * w_irls

    rms_px = float(np.sqrt(np.mean(raw_px * raw_px))) if raw_px.size > 0 else 0.0
    r_eff = r_final * sqrt_w
    chi2 = float(r_eff @ r_eff)
    dof = max(1, int(r_final.size) - 4)

    return PnplSolution(
        pos_ned_m=x[:3].copy(),
        yaw_rad=float(x[3]),
        cov_4x4=cov_4x4,
        jacobian_obs=j_eff,
        weight_diag=w_full_arr,
        rms_reproj_px=rms_px,
        chi2=chi2,
        dof=dof,
        converged=converged,
        iterations=iters,
    )


class ApproachEkf5Hz:
    """5 Hz 11-state Approach EKF `[t_n, t_e, t_d, v_n, v_e, v_d, psi, b_an, b_ae, b_ad, b_gz]`."""

    STATE_DIM: int = 11

    def __init__(
        self,
        init_pos_ned: np.ndarray,
        init_vel_ned: np.ndarray,
        init_yaw_rad: float,
        init_pos_std_m: float = 15.0,
        init_vel_std_mps: float = 3.0,
        init_yaw_std_rad: float = 0.05,
    ) -> None:
        self.x = np.zeros(self.STATE_DIM, dtype=np.float64)
        self.x[0:3] = np.asarray(init_pos_ned, dtype=np.float64)
        self.x[3:6] = np.asarray(init_vel_ned, dtype=np.float64)
        self.x[6] = _wrap_angle_rad(float(init_yaw_rad))
        # Biases x[7:10] = accel bias NED (m/s^2), x[10] = yaw-rate gyro bias (rad/s)

        self.p_cov = np.diag(
            [
                init_pos_std_m**2,
                init_pos_std_m**2,
                init_pos_std_m**2,
                init_vel_std_mps**2,
                init_vel_std_mps**2,
                init_vel_std_mps**2,
                init_yaw_std_rad**2,
                0.05**2,
                0.05**2,
                0.05**2,
                0.005**2,
            ]
        )

    def predict(
        self,
        accel_ned_mps2: np.ndarray,
        yaw_rate_rad_s: float,
        dt: float = 0.2,
        sigma_accel: float = 0.25,
        sigma_gyro_z: float = 0.005,
    ) -> None:
        """Propagate 11-state kinematics and covariance over `dt` (default 0.2 s = 5 Hz)."""
        dt_f = float(dt)
        if not (math.isfinite(dt_f) and dt_f > 0.0):
            raise ValueError(f"ApproachEkf5Hz.predict dt must be finite > 0, got {dt!r}")
        acc_in = np.asarray(accel_ned_mps2, dtype=np.float64).reshape(3)
        wz_in = float(yaw_rate_rad_s)
        sig_a = float(sigma_accel)
        sig_g = float(sigma_gyro_z)
        if not (
            np.all(np.isfinite(acc_in))
            and math.isfinite(wz_in)
            and math.isfinite(sig_a)
            and sig_a >= 0.0
            and math.isfinite(sig_g)
            and sig_g >= 0.0
        ):
            raise ValueError("ApproachEkf5Hz.predict received non-finite or negative-noise inputs")

        a_corr = acc_in - self.x[7:10]
        wz_corr = wz_in - float(self.x[10])

        self.x[0:3] += self.x[3:6] * dt_f + 0.5 * a_corr * (dt_f * dt_f)
        self.x[3:6] += a_corr * dt_f
        self.x[6] = _wrap_angle_rad(float(self.x[6]) + wz_corr * dt_f)

        f_mat = np.eye(self.STATE_DIM, dtype=np.float64)
        f_mat[0:3, 3:6] = np.eye(3) * dt_f
        f_mat[0:3, 7:10] = -0.5 * (dt_f * dt_f) * np.eye(3)
        f_mat[3:6, 7:10] = -dt_f * np.eye(3)
        f_mat[6, 10] = -dt_f

        q_mat = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)
        q_pos = (0.5 * sig_a * dt_f * dt_f) ** 2
        q_vel = (sig_a * dt_f) ** 2
        q_mat[0:3, 0:3] = np.eye(3) * (q_pos + 1e-4)
        q_mat[3:6, 3:6] = np.eye(3) * q_vel
        q_mat[6, 6] = (sig_g * dt_f) ** 2
        q_mat[7:10, 7:10] = np.eye(3) * ((1e-3) ** 2 * dt_f)
        q_mat[10, 10] = (1e-4) ** 2 * dt_f

        self.p_cov = f_mat @ self.p_cov @ f_mat.T + q_mat
        self.p_cov = 0.5 * (self.p_cov + self.p_cov.T)

    def update_pnpl(
        self,
        pos_ned_meas: np.ndarray,
        yaw_meas_rad: float,
        r_4x4: np.ndarray,
        *,
        gate_chi2: float = 13.277,
    ) -> EkfUpdateResult:
        """Mahalanobis-gated 4-DoF PnPL update with Joseph-form covariance preservation."""
        z = np.array(
            [float(pos_ned_meas[0]), float(pos_ned_meas[1]), float(pos_ned_meas[2]), float(yaw_meas_rad)],
            dtype=np.float64,
        )
        r_mat = np.asarray(r_4x4, dtype=np.float64)
        gate_f = float(gate_chi2)
        if (
            r_mat.shape != (4, 4)
            or not (np.all(np.isfinite(z)) and np.all(np.isfinite(r_mat)) and math.isfinite(gate_f) and gate_f >= 0.0)
        ):
            return EkfUpdateResult(False, math.inf, gate_chi2, self.x[0:3].copy(), self.x[3:6].copy(), float(self.x[6]))

        r_sym = 0.5 * (r_mat + r_mat.T)
        if float(np.linalg.eigvalsh(r_sym)[0]) < -1e-9:
            return EkfUpdateResult(False, math.inf, gate_chi2, self.x[0:3].copy(), self.x[3:6].copy(), float(self.x[6]))

        h_mat = np.zeros((4, self.STATE_DIM), dtype=np.float64)
        h_mat[0:3, 0:3] = np.eye(3)
        h_mat[3, 6] = 1.0

        y_inn = np.array(
            [
                z[0] - self.x[0],
                z[1] - self.x[1],
                z[2] - self.x[2],
                _wrap_angle_rad(z[3] - float(self.x[6])),
            ],
            dtype=np.float64,
        )
        s_mat = h_mat @ self.p_cov @ h_mat.T + r_sym
        s_mat = 0.5 * (s_mat + s_mat.T)
        if not np.all(np.isfinite(s_mat)):
            return EkfUpdateResult(False, math.inf, gate_chi2, self.x[0:3].copy(), self.x[3:6].copy(), float(self.x[6]))
        try:
            chol_s = np.linalg.cholesky(s_mat)
            w_inn = np.linalg.solve(chol_s, y_inn)
            s_inv = np.linalg.solve(chol_s.T, np.linalg.solve(chol_s, np.eye(4, dtype=np.float64)))
        except np.linalg.LinAlgError:
            return EkfUpdateResult(False, math.inf, gate_chi2, self.x[0:3].copy(), self.x[3:6].copy(), float(self.x[6]))

        d2 = float(w_inn @ w_inn)
        # Fail-closed Mahalanobis gate (explicitly requires finite non-negative d2)
        if not (math.isfinite(d2) and 0.0 <= d2 <= gate_f):
            return EkfUpdateResult(False, d2, gate_chi2, self.x[0:3].copy(), self.x[3:6].copy(), float(self.x[6]))

        k_gain = self.p_cov @ h_mat.T @ s_inv
        self.x = self.x + k_gain @ y_inn
        self.x[6] = _wrap_angle_rad(float(self.x[6]))

        # Joseph-form update preserves symmetry & PSD strictly
        i_kh = np.eye(self.STATE_DIM) - k_gain @ h_mat
        self.p_cov = i_kh @ self.p_cov @ i_kh.T + k_gain @ r_sym @ k_gain.T
        self.p_cov = 0.5 * (self.p_cov + self.p_cov.T)

        return EkfUpdateResult(True, d2, gate_chi2, self.x[0:3].copy(), self.x[3:6].copy(), float(self.x[6]))


def build_approach_trajectory_poses(
    *,
    runway_heading_deg: float = 297.7,
    start_range_m: float = 1500.0,
    end_range_m: float = 250.0,
    glideslope_deg: float = 3.0,
    num_frames: int = 50,
    dt: float = 0.2,
    lateral_drift_amp_m: float = 3.0,
    crab_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate a smooth 5 Hz 3-deg glideslope approach sequence strictly in runway-threshold NED."""
    n = max(2, int(num_frames))
    hdg = math.radians(float(runway_heading_deg))
    fwd = np.array([math.cos(hdg), math.sin(hdg), 0.0], dtype=np.float64)
    right = np.array([-math.sin(hdg), math.cos(hdg), 0.0], dtype=np.float64)
    tan_gs = math.tan(math.radians(float(glideslope_deg)))

    ranges = np.linspace(float(start_range_m), float(end_range_m), n)
    positions: list[np.ndarray] = []
    for idx, r_m in enumerate(ranges):
        phase = 2.0 * math.pi * float(idx) / float(max(1, n - 1))
        lat_m = float(lateral_drift_amp_m) * math.sin(phase) * (float(r_m) / float(max(start_range_m, 1.0)))
        alt_m = max(14.0, float(r_m) * tan_gs + 1.5)
        pos_ned = -fwd * float(r_m) + right * lat_m + np.array([0.0, 0.0, -alt_m], dtype=np.float64)
        positions.append(pos_ned)

    poses: list[dict[str, Any]] = []
    for idx in range(n):
        pos = positions[idx]
        if idx < n - 1:
            vel = (positions[idx + 1] - pos) / float(dt)
        else:
            vel = (pos - positions[idx - 1]) / float(dt)
        if 0 < idx < n - 1:
            acc = (positions[idx + 1] - 2.0 * pos + positions[idx - 1]) / float(dt * dt)
        else:
            acc = np.zeros(3, dtype=np.float64)
        aim_ned = fwd * 300.0
        to_aim = aim_ned - pos
        yaw_rad = math.atan2(float(to_aim[1]), float(to_aim[0])) + math.radians(float(crab_deg))
        pitch_rad = math.radians(-float(glideslope_deg) - 0.8)
        roll_rad = math.radians(float(roll_deg))
        poses.append(
            {
                "frame_idx": idx,
                "cam_pos_ned": pos,
                "vel_ned_mps": vel,
                "accel_ned_mps2": acc,
                "yaw_rad": float(yaw_rad),
                "pitch_rad": float(pitch_rad),
                "roll_rad": float(roll_rad),
                "yaw_rate_rad_s": 0.0,
                "range_m": float(ranges[idx]),
            }
        )
    return poses


def evaluate_approach_ekf_trajectory(
    frames: Sequence[dict[str, Any]],
    *,
    dt: float = 0.2,
    gate_chi2: float = 13.277,
    huber_delta_sigma: float = 3.0,
    conformal_quantile_rad: float = 4.0e-4,
    sigma_accel: float = 0.10,
    sigma_gyro_z: float = 0.005,
    init_pos_offset_ned: np.ndarray | None = None,
    init_vel_offset_ned: np.ndarray | None = None,
    init_yaw_offset_rad: float = 0.0,
    r_body_cam: np.ndarray = R_BODY_CAM_FORWARD,
) -> ApproachEkfTrajectoryResult:
    """Run 5 Hz `ApproachEkf5Hz` + Huber-IRLS 4-DoF MAP-PnPL + Conformal-RAIM over a multi-frame approach.

    Each element of `frames` provides camera pose ground-truth (`cam_pos_ned`, `yaw_rad`, `pitch_rad`,
    `roll_rad`, `range_m`), intrinsics (`fx`, `fy`, `cx`, `cy`), runway geometry (`pts_ned`, optional
    `keyline_pairs`), and visual observations (`obs_uv`, `obs_cov_2x2`, optional `obs_keylines_hesse`,
    `keyline_sigmas_px` / `keyline_sigma_px`).
    """
    if len(frames) == 0:
        raise ValueError("evaluate_approach_ekf_trajectory requires at least 1 frame")

    f0 = frames[0]
    pos0_gt = np.asarray(f0["cam_pos_ned"], dtype=np.float64)
    if "vel_ned_mps" in f0 and f0["vel_ned_mps"] is not None:
        vel0_gt = np.asarray(f0["vel_ned_mps"], dtype=np.float64)
    elif len(frames) >= 2:
        pos1_gt = np.asarray(frames[1]["cam_pos_ned"], dtype=np.float64)
        vel0_gt = (pos1_gt - pos0_gt) / float(dt)
    else:
        vel0_gt = np.zeros(3, dtype=np.float64)

    pos_off = (
        np.zeros(3, dtype=np.float64)
        if init_pos_offset_ned is None
        else np.asarray(init_pos_offset_ned, dtype=np.float64)
    )
    vel_off = (
        np.zeros(3, dtype=np.float64)
        if init_vel_offset_ned is None
        else np.asarray(init_vel_offset_ned, dtype=np.float64)
    )

    ekf = ApproachEkf5Hz(
        init_pos_ned=pos0_gt + pos_off,
        init_vel_ned=vel0_gt + vel_off,
        init_yaw_rad=float(f0["yaw_rad"]) + float(init_yaw_offset_rad),
        init_pos_std_m=max(10.0, float(np.linalg.norm(pos_off)) * 2.0 + 5.0),
        init_vel_std_mps=max(1.5, float(np.linalg.norm(vel_off)) * 2.0 + 1.0),
        init_yaw_std_rad=max(0.03, abs(float(init_yaw_offset_rad)) * 2.0 + 0.02),
    )

    idx_4 = np.array([0, 1, 2, 6], dtype=np.int64)
    ekf_pos_errs: list[float] = []
    ekf_horiz_errs: list[float] = []
    ekf_vert_errs: list[float] = []
    single_pos_errs: list[float] = []
    v_hpls: list[float] = []
    v_vpls: list[float] = []
    accepted = 0
    rejected = 0
    cat1_ok = 0
    hmi_count = 0

    for t_idx, frame in enumerate(frames):
        if t_idx > 0:
            acc = (
                np.asarray(frame["accel_ned_mps2"], dtype=np.float64)
                if "accel_ned_mps2" in frame and frame["accel_ned_mps2"] is not None
                else np.zeros(3, dtype=np.float64)
            )
            wz = float(frame["yaw_rate_rad_s"]) if "yaw_rate_rad_s" in frame and frame["yaw_rate_rad_s"] is not None else 0.0
            ekf.predict(acc, wz, dt=float(dt), sigma_accel=float(sigma_accel), sigma_gyro_z=float(sigma_gyro_z))

        keyline_sigma_val = float(frame["keyline_sigma_px"]) if ("keyline_sigma_px" in frame and frame["keyline_sigma_px"] is not None) else 1.5
        sol = solve_4dof_map_pnpl(
            frame["pts_ned"],
            frame["obs_uv"],
            frame["obs_cov_2x2"],
            pitch_rad=float(frame["pitch_rad"]),
            roll_rad=float(frame["roll_rad"]),
            init_pos_ned=ekf.x[0:3].copy(),
            init_yaw_rad=float(ekf.x[6]),
            fx=float(frame["fx"]),
            fy=float(frame["fy"]),
            cx=float(frame["cx"]),
            cy=float(frame["cy"]),
            keyline_pairs=frame["keyline_pairs"] if "keyline_pairs" in frame else None,
            obs_keylines_hesse=frame["obs_keylines_hesse"] if "obs_keylines_hesse" in frame else None,
            keyline_sigma_px=keyline_sigma_val,
            keyline_sigmas_px=frame["keyline_sigmas_px"] if "keyline_sigmas_px" in frame else None,
            huber_delta_sigma=float(huber_delta_sigma),
            r_body_cam=r_body_cam,
        )

        true_pos = np.asarray(frame["cam_pos_ned"], dtype=np.float64)
        single_pos_errs.append(float(np.linalg.norm(sol.pos_ned_m - true_pos)))

        upd = ekf.update_pnpl(
            sol.pos_ned_m,
            sol.yaw_rad,
            sol.cov_4x4,
            gate_chi2=float(gate_chi2),
        )
        if upd.accepted:
            accepted += 1
        else:
            rejected += 1

        err_vec = ekf.x[0:3] - true_pos
        horiz_err = float(math.hypot(float(err_vec[0]), float(err_vec[1])))
        vert_err = abs(float(err_vec[2]))
        ekf_pos_errs.append(float(np.linalg.norm(err_vec)))
        ekf_horiz_errs.append(horiz_err)
        ekf_vert_errs.append(vert_err)

        # Hybrid Conformal-RAIM on EKF-fused 4-DoF posterior covariance
        ekf_cov_4x4 = ekf.p_cov[np.ix_(idx_4, idx_4)]
        raim = compute_hybrid_conformal_raim_pl(
            sol.jacobian_obs,
            ekf_cov_4x4,
            range_m=float(frame["range_m"]),
            conformal_quantile_rad=float(conformal_quantile_rad),
        )
        v_hpls.append(float(raim.v_hpl_m))
        v_vpls.append(float(raim.v_vpl_m))
        if raim.hpl_within_limit and raim.vpl_within_limit:
            cat1_ok += 1
            if horiz_err > raim.v_hpl_m or vert_err > raim.v_vpl_m:
                hmi_count += 1

    n_total = len(frames)
    return ApproachEkfTrajectoryResult(
        num_frames=n_total,
        accepted_updates=accepted,
        rejected_updates=rejected,
        ekf_pos_rmse_m=float(math.sqrt(float(np.mean(np.square(ekf_pos_errs))))),
        ekf_horiz_rmse_m=float(math.sqrt(float(np.mean(np.square(ekf_horiz_errs))))),
        ekf_vert_rmse_m=float(math.sqrt(float(np.mean(np.square(ekf_vert_errs))))),
        single_frame_pnpl_rmse_m=float(math.sqrt(float(np.mean(np.square(single_pos_errs))))),
        v_hpl_median_m=float(np.median(v_hpls)),
        v_vpl_median_m=float(np.median(v_vpls)),
        v_hpl_p95_m=float(np.percentile(v_hpls, 95)),
        v_vpl_p95_m=float(np.percentile(v_vpls, 95)),
        cat1_availability=float(cat1_ok) / float(max(1, n_total)),
        hmi_count=int(hmi_count),
        start_range_m=float(frames[0]["range_m"]),
        end_range_m=float(frames[-1]["range_m"]),
    )
