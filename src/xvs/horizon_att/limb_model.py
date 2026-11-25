"""WGS84 Earth-limb tangent cone model, ray residuals, and analytic Jacobian.

Conforms strictly to docs/xvs/WP06_HORIZON.md v0.1 §1.2 and CONVENTIONS.md §4, §7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from xvs.geo import wgs84, rotations
from xvs.rig.rig import Rig
from xvs.rig.camera import PinholeCamera
from xvs.sim import horizon

# WGS84 scaling matrix S = diag(1/a, 1/a, 1/b) mapping ellipsoid to unit sphere
_S_DIAG = np.array([1.0 / wgs84.WGS84_A, 1.0 / wgs84.WGS84_A, 1.0 / wgs84.WGS84_B], dtype=np.float64)


@dataclass
class LimbRay:
    """Individual line-of-sight ray observation of the Earth limb."""
    cam_id: str
    d_body: np.ndarray          # (3,) unit vector in aircraft body frame
    lever_arm_body: np.ndarray  # (3,) camera origin in body frame (metres)
    u: float = 0.0              # image column (optional)
    v: float = 0.0              # image row (optional)
    weight: float = 1.0         # weight / confidence


class LimbRayBatch:
    """Vectorized collection of limb ray observations for efficient batched evaluation."""

    def __init__(
        self,
        d_body: np.ndarray,
        lever_arm_body: np.ndarray,
        cam_ids: list[str] | np.ndarray | None = None,
        weights: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
    ):
        self.d_body = np.asarray(d_body, dtype=np.float64)
        if self.d_body.ndim == 1:
            self.d_body = self.d_body.reshape(1, 3)
        self.lever_arm_body = np.asarray(lever_arm_body, dtype=np.float64)
        if self.lever_arm_body.ndim == 1:
            self.lever_arm_body = self.lever_arm_body.reshape(1, 3)
        if len(self.lever_arm_body) == 1 and len(self.d_body) > 1:
            self.lever_arm_body = np.repeat(self.lever_arm_body, len(self.d_body), axis=0)

        assert len(self.d_body) == len(self.lever_arm_body), "Mismatched ray and lever arm counts"
        self.n_rays = len(self.d_body)

        if cam_ids is None:
            self.cam_ids = ["F"] * self.n_rays
        else:
            self.cam_ids = list(cam_ids)

        if weights is None:
            self.weights = np.ones(self.n_rays, dtype=np.float64)
        else:
            self.weights = np.asarray(weights, dtype=np.float64)

        if uvs is None:
            self.uvs = np.zeros((self.n_rays, 2), dtype=np.float64)
        else:
            self.uvs = np.asarray(uvs, dtype=np.float64)

    @classmethod
    def from_rays(cls, rays: Sequence[LimbRay]) -> LimbRayBatch:
        if len(rays) == 0:
            return cls(np.zeros((0, 3)), np.zeros((0, 3)), [])
        d_body = np.stack([r.d_body for r in rays], axis=0)
        lever_arm_body = np.stack([r.lever_arm_body for r in rays], axis=0)
        cam_ids = [r.cam_id for r in rays]
        weights = np.array([r.weight for r in rays], dtype=np.float64)
        uvs = np.array([[r.u, r.v] for r in rays], dtype=np.float64)
        return cls(d_body, lever_arm_body, cam_ids, weights, uvs)


def compute_cone_residuals(
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
    batch: LimbRayBatch,
) -> np.ndarray:
    """Compute signed angular residuals (radians) of rays to the WGS84 tangent cone.

    Residual is positive if the ray points above the horizon (into space/sky)
    and negative if it intersects the Earth ellipsoid.
    """
    if batch.n_rays == 0:
        return np.zeros(0, dtype=np.float64)

    r_ecef_ned_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
    r_ned_body = rotations.euler_to_r(yaw_rad, pitch_rad, roll_rad)
    r_ecef_body = r_ecef_ned_mat @ r_ned_body
    p_ecef_body = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)

    # Camera centers and ray directions in ECEF
    p_cams = p_ecef_body + batch.lever_arm_body @ r_ecef_body.T
    d_ecefs = batch.d_body @ r_ecef_body.T

    # Map to unit sphere coordinates: C' = S @ C, d' = S @ d
    c_prime = p_cams * _S_DIAG
    k = np.linalg.norm(c_prime, axis=1, keepdims=True)
    k = np.maximum(1.0 + 1e-12, k)
    c_hat = c_prime / k
    theta_cone = np.arcsin(np.clip(1.0 / k, -1.0, 1.0))

    d_prime = d_ecefs * _S_DIAG
    norm_dp = np.linalg.norm(d_prime, axis=1, keepdims=True)
    hat_dp = d_prime / norm_dp

    # cos(theta') = -c_hat . hat_dp (angle of ray with vector to Earth center)
    cos_th = -np.sum(c_hat * hat_dp, axis=1, keepdims=True)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th_prime = np.arccos(cos_th)

    res = th_prime - theta_cone
    return res.ravel()


def compute_residuals_and_jacobian(
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
    batch: LimbRayBatch,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed residuals and the exact analytical Jacobian w.r.t [roll, pitch].

    Returns:
        residuals: (N,) array of angular residuals in radians.
        jacobian: (N, 2) array where col 0 is d(res)/d(roll) and col 1 is d(res)/d(pitch).
    """
    n = batch.n_rays
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    r_ecef_ned_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
    r0 = r_ecef_ned_mat @ rotations.rz(yaw_rad)

    c_r, s_r = math.cos(roll_rad), math.sin(roll_rad)
    c_p, s_p = math.cos(pitch_rad), math.sin(pitch_rad)

    # Elementary rotation derivatives
    # dRx/droll
    drx = np.array([[0.0, 0.0, 0.0], [0.0, -s_r, -c_r], [0.0, c_r, -s_r]], dtype=np.float64)
    # dRy/dpitch
    dry = np.array([[-s_p, 0.0, c_p], [0.0, 0.0, 0.0], [-c_p, 0.0, -s_p]], dtype=np.float64)

    ry = rotations.ry(pitch_rad)
    rx = rotations.rx(roll_rad)

    dr_droll = r0 @ ry @ drx
    dr_dpitch = r0 @ dry @ rx
    r_ecef_body = r0 @ ry @ rx
    p_ecef_body = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)

    # Base values
    p_cams = p_ecef_body + batch.lever_arm_body @ r_ecef_body.T
    d_ecefs = batch.d_body @ r_ecef_body.T

    c_prime = p_cams * _S_DIAG
    k = np.linalg.norm(c_prime, axis=1, keepdims=True)
    k = np.maximum(1.0 + 1e-12, k)
    c_hat = c_prime / k
    k_sq = k * k
    denom_cone = k * np.sqrt(np.maximum(1e-15, k_sq - 1.0))
    theta_cone = np.arcsin(np.clip(1.0 / k, -1.0, 1.0))

    d_prime = d_ecefs * _S_DIAG
    norm_dp = np.linalg.norm(d_prime, axis=1, keepdims=True)
    hat_dp = d_prime / norm_dp

    cos_th = -np.sum(c_hat * hat_dp, axis=1, keepdims=True)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    sin_th = np.sqrt(np.maximum(1e-15, 1.0 - cos_th * cos_th))
    th_prime = np.arccos(cos_th)

    residuals = (th_prime - theta_cone).ravel()

    # Analytical Jacobian computation: [d/d_roll, d/d_pitch]
    jacobian = np.zeros((n, 2), dtype=np.float64)

    for col_idx, dr in enumerate([dr_droll, dr_dpitch]):
        dc = batch.lever_arm_body @ dr.T
        dd = batch.d_body @ dr.T

        dc_prime = dc * _S_DIAG
        dd_prime = dd * _S_DIAG

        dk = np.sum(c_hat * dc_prime, axis=1, keepdims=True)
        dc_hat = (dc_prime - dk * c_hat) / k
        dtheta_cone = -dk / denom_cone

        dnorm_dp = np.sum(hat_dp * dd_prime, axis=1, keepdims=True)
        dhat_dp = (dd_prime - dnorm_dp * hat_dp) / norm_dp

        dcos_th = -(np.sum(dc_hat * hat_dp, axis=1, keepdims=True) + np.sum(c_hat * dhat_dp, axis=1, keepdims=True))
        dth_prime = -dcos_th / sin_th

        jacobian[:, col_idx : col_idx + 1] = dth_prime - dtheta_cone

    return residuals, jacobian


def solve_attitude_gn(
    lat_deg: float,
    lon_deg: float,
    h_meas_m: float,
    yaw_rad: float,
    pitch_init_rad: float,
    roll_init_rad: float,
    batch: LimbRayBatch,
    max_iter: int = 25,
    tol_rad: float = 1e-12,
    damping: float = 1e-6,
    huber_delta_rad: float | None = None,
) -> dict[str, Any]:
    """Solve for aircraft attitude (roll, pitch) using robust Gauss-Newton / LM.

    Args:
        lat_deg, lon_deg: Horizontal position.
        h_meas_m: Measured altitude above WGS84 ellipsoid.
        yaw_rad: Aircraft heading (fixed, from nav record).
        pitch_init_rad, roll_init_rad: Initial attitude guess.
        batch: Limb ray observations.
        max_iter: Max iterations.
        tol_rad: Step size stopping threshold in radians.
        damping: Levenberg-Marquardt diagonal damping factor.
        huber_delta_rad: Optional Huber threshold for robust loss.

    Returns:
        Dictionary with:
          roll_deg, pitch_deg, roll_rad, pitch_rad,
          covariance_deg2 (2x2), covariance_rad2 (2x2),
          rms_residual_urad, inlier_count, status ('CONVERGED', 'MAX_ITER', 'FAILED').
    """
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

    for it in range(max_iter):
        iterations = it + 1
        residuals, jacobian = compute_residuals_and_jacobian(
            lat_deg, lon_deg, h_meas_m, yaw_rad, p_est, r_est, batch
        )

        # Apply Huber loss weighting if enabled
        if huber_delta_rad is not None and huber_delta_rad > 0:
            abs_r = np.abs(residuals)
            huber_w = np.where(abs_r <= huber_delta_rad, 1.0, huber_delta_rad / np.maximum(abs_r, 1e-15))
            effective_w = weights * huber_w
        else:
            effective_w = weights

        sqrt_w = np.sqrt(effective_w)[:, None]
        j_weighted = jacobian * sqrt_w
        r_weighted = residuals * sqrt_w.ravel()

        jt_j = j_weighted.T @ j_weighted
        jt_r = j_weighted.T @ r_weighted

        # Levenberg-Marquardt regularisation
        diag_jt_j = np.diag(np.diag(jt_j))
        lhs = jt_j + damping * (diag_jt_j + 1e-9 * np.eye(2))

        try:
            step = np.linalg.solve(lhs, -jt_r)
        except np.linalg.LinAlgError:
            step, _, _, _ = np.linalg.lstsq(lhs, -jt_r, rcond=None)

        r_est += float(step[0])
        p_est += float(step[1])

        step_norm = float(np.linalg.norm(step))
        if step_norm < tol_rad:
            converged = True
            break

    # Final residuals
    residuals, jacobian = compute_residuals_and_jacobian(
        lat_deg, lon_deg, h_meas_m, yaw_rad, p_est, r_est, batch
    )
    rms_res_rad = float(np.sqrt(np.mean(residuals * residuals)))
    rms_res_urad = rms_res_rad * 1e6

    # Covariance computation: (J^T W J)^-1 * sigma^2
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


def generate_synthetic_limb_rays(
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
    pts_per_cam: int = 20,
    num_samples: int = 1024,
) -> LimbRayBatch:
    """Generate noise-free synthetic limb rays from the analytic WGS84 model.

    Used by V1 exactness tests and simulator ground truth validation.
    """
    poses = rig.compute_poses(lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad)
    cam = rig.camera_model

    d_bodies = []
    levers = []
    cam_ids = []
    uvs = []

    for cam_id in ("F", "R", "B", "L", "U", "D"):
        p_c = poses[cam_id]["p_ecef_cam"]
        r_c = poses[cam_id]["r_ecef_cam"]

        gt = horizon.compute_horizon_gt(p_c, r_c, cam, num_samples=num_samples, max_points=pts_per_cam)
        if not gt["horizon_visible"] or len(gt["horizon_uv"]) == 0:
            continue

        r_bc = rig.cam_by_id[cam_id].r_body_cam
        lev = rig.cam_by_id[cam_id].lever_arm_m

        for u, v in gt["horizon_uv"]:
            d_cam = cam.unproject(u, v)
            d_body = r_bc @ d_cam
            d_bodies.append(d_body)
            levers.append(lev)
            cam_ids.append(cam_id)
            uvs.append([u, v])

    if len(d_bodies) == 0:
        return LimbRayBatch(np.zeros((0, 3)), np.zeros((0, 3)), [])

    return LimbRayBatch(
        np.array(d_bodies, dtype=np.float64),
        np.array(levers, dtype=np.float64),
        cam_ids,
        uvs=np.array(uvs, dtype=np.float64),
    )
