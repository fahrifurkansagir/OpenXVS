"""6-DoF WGS-84 Ground-Clutter Rejection for Track Initiation (WP-05 G1 Step 3).

Conforms strictly to docs/xvs/batches/BATCH_B7_ARGON.md §3:
- Uses ONLY ownship navigation ECEF position, local NED-to-ECEF rotation, and INS-estimated body attitude
  (zero ground-truth access, verified by V4_G1);
- Intersects tentative track LOS rays with the WGS-84 ellipsoid (a = 6378137.0 m, b = 6356752.314245 m);
- Rays pointing above the WGS-84 horizon (lambda_hit is None) bypass the ground-clutter gate;
- Computes multi-frame stationary-ground reprojection residual r_gnd_px across the tentative observation window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

from xvs.geo import wgs84


@dataclass
class GroundObservation:
    """Single-frame observation record for 6-DoF WGS-84 ground-consistency evaluation."""
    t_s: float
    cam_id: str
    u_px: float
    v_px: float
    s_mom_rad: float
    los_ned_ins: np.ndarray
    p_cam_ecef_ins: np.ndarray
    r_ecef_ned: np.ndarray


@dataclass
class GroundConsistencyResult:
    """Result of multi-frame 6-DoF WGS-84 stationary-ground consistency check."""
    is_below_horizon: bool
    lambda_hit_m: float | None
    r_gnd_px: float
    d_gnd_ang_rad: float
    n_frames: int


def ray_wgs84_intersect(
    p_origin_ecef: np.ndarray,
    d_ray_ecef: np.ndarray,
) -> tuple[float | None, np.ndarray | None]:
    """Intersect a 3D ray p(lambda) = p_origin_ecef + lambda * d_ray_ecef with the WGS-84 ellipsoid.

    Following BATCH_B7_ARGON.md §3.3:
      S = diag(1/a, 1/a, 1/b)
      A = ||S d||^2, B = 2 (S p)^T (S d), C = ||S p||^2 - 1
      Delta = B^2 - 4 A C
      lambda_hit = (-B - sqrt(Delta)) / (2 A)

    Returns (lambda_hit_m, g_ecef) if the ray intersects the ellipsoid in front of the origin
    (Delta >= 0 and lambda_hit > 0), or (None, None) if the ray points above the horizon.
    """
    p_ecef = np.asarray(p_origin_ecef, dtype=np.float64)
    d_ecef = np.asarray(d_ray_ecef, dtype=np.float64)
    d_norm = float(np.linalg.norm(d_ecef))
    if d_norm <= 1e-12:
        return None, None
    d_unit = d_ecef / d_norm

    inv_a = 1.0 / wgs84.WGS84_A
    inv_b = 1.0 / wgs84.WGS84_B
    p_tilde = np.array([p_ecef[0] * inv_a, p_ecef[1] * inv_a, p_ecef[2] * inv_b], dtype=np.float64)
    d_tilde = np.array([d_unit[0] * inv_a, d_unit[1] * inv_a, d_unit[2] * inv_b], dtype=np.float64)

    a_coef = float(np.dot(d_tilde, d_tilde))
    b_coef = 2.0 * float(np.dot(p_tilde, d_tilde))
    c_coef = float(np.dot(p_tilde, p_tilde)) - 1.0

    disc = b_coef * b_coef - 4.0 * a_coef * c_coef
    if disc < 0.0 or a_coef <= 1e-24:
        return None, None

    sqrt_disc = math.sqrt(max(0.0, disc))
    lam_hit = (-b_coef - sqrt_disc) / (2.0 * a_coef)
    if lam_hit <= 0.0:
        return None, None

    g_ecef = p_ecef + lam_hit * d_unit
    return float(lam_hit), g_ecef


def compute_ground_consistency_residual(
    observations: Sequence[GroundObservation],
    fx: float = 2319.0,
    max_window_obs: int = 10,
) -> GroundConsistencyResult:
    """Compute multi-frame 6-DoF WGS-84 stationary-ground reprojection residual r_gnd_px.

    Conforms to BATCH_B7_ARGON.md §3.3:
    1. Take the anchor observation t_1 at the start of the evaluation window (up to max_window_obs recent hits).
    2. Intersect its INS-estimated ray d_ECEF(t_1) = R_ECEF<-NED(t_1) u_NED(t_1) from p_cam_ECEF(t_1)
       with the WGS-84 ellipsoid.
    3. If lambda_hit is None (above horizon), return is_below_horizon=False and r_gnd_px=inf.
    4. For each subsequent observation k in {2, ..., m}, project the stationary ground point g_ECEF(t_1)
       into NED at t_k and compute the angular discrepancy d_gnd_ang(t_k) and pixel residual
       r_gnd_px(t_k) = fx * d_gnd_ang(t_k).
    5. Aggregate via RMS: r_gnd_px = sqrt((1 / (m - 1)) * sum_{k=2}^m r_gnd_px(t_k)^2).
    """
    n_total = len(observations)
    if n_total == 0:
        return GroundConsistencyResult(
            is_below_horizon=False,
            lambda_hit_m=None,
            r_gnd_px=float("inf"),
            d_gnd_ang_rad=float("inf"),
            n_frames=0,
        )

    win_obs = observations[-max_window_obs:] if max_window_obs > 0 and n_total > max_window_obs else observations
    m = len(win_obs)
    obs_1 = win_obs[0]

    u_ned_1 = np.asarray(obs_1.los_ned_ins, dtype=np.float64)
    u_norm_1 = float(np.linalg.norm(u_ned_1))
    if u_norm_1 <= 1e-12:
        return GroundConsistencyResult(
            is_below_horizon=False,
            lambda_hit_m=None,
            r_gnd_px=float("inf"),
            d_gnd_ang_rad=float("inf"),
            n_frames=m,
        )
    u_ned_1 = u_ned_1 / u_norm_1
    r_ecef_ned_1 = np.asarray(obs_1.r_ecef_ned, dtype=np.float64)
    p_cam_1 = np.asarray(obs_1.p_cam_ecef_ins, dtype=np.float64)
    d_ecef_1 = r_ecef_ned_1 @ u_ned_1

    lam_hit, g_ecef_1 = ray_wgs84_intersect(p_cam_1, d_ecef_1)
    if lam_hit is None or g_ecef_1 is None:
        return GroundConsistencyResult(
            is_below_horizon=False,
            lambda_hit_m=None,
            r_gnd_px=float("inf"),
            d_gnd_ang_rad=float("inf"),
            n_frames=m,
        )

    if m < 2:
        return GroundConsistencyResult(
            is_below_horizon=True,
            lambda_hit_m=lam_hit,
            r_gnd_px=0.0,
            d_gnd_ang_rad=0.0,
            n_frames=1,
        )

    g_candidates: list[np.ndarray] = [g_ecef_1]
    init_slice = win_obs[: min(3, m)]
    init_g_points: list[np.ndarray] = []
    for obs_j in init_slice:
        p_j = np.asarray(obs_j.p_cam_ecef_ins, dtype=np.float64)
        r_j = np.asarray(obs_j.r_ecef_ned, dtype=np.float64)
        u_j = np.asarray(obs_j.los_ned_ins, dtype=np.float64)
        un_j = float(np.linalg.norm(u_j))
        if un_j <= 1e-12:
            continue
        lam_j, g_j = ray_wgs84_intersect(p_j, r_j @ (u_j / un_j))
        if lam_j is None or g_j is None:
            return GroundConsistencyResult(
                is_below_horizon=False,
                lambda_hit_m=None,
                r_gnd_px=float("inf"),
                d_gnd_ang_rad=float("inf"),
                n_frames=m,
            )
        init_g_points.append(g_j)
    if len(init_g_points) >= 2:
        g_candidates.append(np.mean(init_g_points, axis=0))

    eval_obs = win_obs[1:]

    best_rms_ang = float("inf")
    valid_pairs = 0
    for g_ref in g_candidates:
        sq_ang_sum = 0.0
        vp = 0
        for obs_k in eval_obs:
            p_cam_k = np.asarray(obs_k.p_cam_ecef_ins, dtype=np.float64)
            r_ecef_ned_k = np.asarray(obs_k.r_ecef_ned, dtype=np.float64)
            u_ned_k = np.asarray(obs_k.los_ned_ins, dtype=np.float64)
            u_norm_k = float(np.linalg.norm(u_ned_k))
            if u_norm_k <= 1e-12:
                continue
            u_ned_k = u_ned_k / u_norm_k

            d_ecef_k = r_ecef_ned_k @ u_ned_k
            lam_k, _ = ray_wgs84_intersect(p_cam_k, d_ecef_k)
            if lam_k is None:
                return GroundConsistencyResult(
                    is_below_horizon=False,
                    lambda_hit_m=None,
                    r_gnd_px=float("inf"),
                    d_gnd_ang_rad=float("inf"),
                    n_frames=m,
                )

            rel_ecef_k = g_ref - p_cam_k
            rel_norm_k = float(np.linalg.norm(rel_ecef_k))
            if rel_norm_k <= 1e-9:
                continue
            u_gnd_ned_k = r_ecef_ned_k.T @ (rel_ecef_k / rel_norm_k)
            u_gnd_norm = float(np.linalg.norm(u_gnd_ned_k))
            if u_gnd_norm > 1e-12:
                u_gnd_ned_k = u_gnd_ned_k / u_gnd_norm

            dot_val = max(-1.0, min(1.0, float(np.dot(u_ned_k, u_gnd_ned_k))))
            d_ang = math.acos(dot_val)
            sq_ang_sum += d_ang * d_ang
            vp += 1
        if vp > 0:
            valid_pairs = vp
            cand_rms = math.sqrt(sq_ang_sum / float(vp))
            if cand_rms < best_rms_ang:
                best_rms_ang = cand_rms

    if valid_pairs == 0 or math.isinf(best_rms_ang):
        return GroundConsistencyResult(
            is_below_horizon=True,
            lambda_hit_m=lam_hit,
            r_gnd_px=0.0,
            d_gnd_ang_rad=0.0,
            n_frames=m,
        )

    rms_ang = best_rms_ang
    rms_px = float(fx) * rms_ang
    return GroundConsistencyResult(
        is_below_horizon=True,
        lambda_hit_m=lam_hit,
        r_gnd_px=rms_px,
        d_gnd_ang_rad=rms_ang,
        n_frames=m,
    )
