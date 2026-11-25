"""Robust attitude estimation (RANSAC + Huber Levenberg-Marquardt) from limb rays.

Conforms strictly to docs/xvs/WP06_HORIZON.md v0.1 §1.3, §1.4.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np

from xvs.rig.rig import Rig
from xvs.sensor.chain import SensorConfig
from xvs.horizon_att.limb_model import (
    LimbRayBatch,
    compute_cone_residuals,
    compute_residuals_and_jacobian,
    solve_attitude_gn,
)
from xvs.horizon_att.edges import (
    detect_camera_edges,
    edges_to_ray_batch,
)


def ransac_fit_attitude(
    lat_deg: float,
    lon_deg: float,
    h_meas_m: float,
    yaw_rad: float,
    batch: LimbRayBatch,
    initial_guess_rad: tuple[float, float] | None = None,
    max_iterations: int = 60,
    inlier_threshold_rad: float = math.radians(0.25),  # ~2.5 px at 100 deg FOV
    min_inliers: int = 5,
    huber_delta_rad: float = math.radians(0.15),       # ~1.5 px (wide linear threshold for altitude errors)
    damping: float = 1e-6,
    rng_seed: int | None = 20260925,
) -> dict[str, Any]:
    """Robust attitude estimator combining RANSAC (3-point minimal sets) with Huber LM.

    Args:
        lat_deg, lon_deg: Aircraft horizontal position.
        h_meas_m: Aircraft altitude above WGS84 ellipsoid (with measurement noise).
        yaw_rad: Aircraft heading (from nav record).
        batch: Detected limb ray observations from all cameras.
        initial_guess_rad: (pitch_init, roll_init) in radians, or None for (0, 0).
        max_iterations: Number of RANSAC trials.
        inlier_threshold_rad: Angular distance threshold to classify inliers.
        min_inliers: Minimum inliers required to accept fit.
        huber_delta_rad: Huber loss linear threshold for final refinement.
        damping: Levenberg-Marquardt damping.
        rng_seed: Random seed for deterministic RANSAC sampling.

    Returns:
        dict containing roll_deg, pitch_deg, covariance_deg2, inlier_count_by_cam,
        rms_residual_urad, status, etc.
    """
    n = batch.n_rays
    all_cams = ("F", "R", "B", "L", "U", "D")

    if n < 3:
        p_init = initial_guess_rad[0] if initial_guess_rad else 0.0
        r_init = initial_guess_rad[1] if initial_guess_rad else 0.0
        return {
            "status": "FAILED_TOO_FEW_RAYS",
            "roll_deg": math.degrees(r_init),
            "pitch_deg": math.degrees(p_init),
            "roll_rad": r_init,
            "pitch_rad": p_init,
            "covariance_deg2": np.full((2, 2), np.nan),
            "covariance_rad2": np.full((2, 2), np.nan),
            "rms_residual_urad": np.nan,
            "inlier_count": n,
            "inlier_count_by_cam": {c: 0 for c in all_cams},
            "total_rays": n,
            "iterations": 0,
        }

    p_init = float(initial_guess_rad[0]) if initial_guess_rad else 0.0
    r_init = float(initial_guess_rad[1]) if initial_guess_rad else 0.0

    rng = np.random.default_rng(rng_seed)

    best_inlier_mask: np.ndarray | None = None
    best_inlier_count = 0
    best_inlier_cost = float("inf")
    best_candidate_p = p_init
    best_candidate_r = r_init

    # RANSAC minimal sampling: minimal sets of 3 points
    for _ in range(max_iterations):
        sample_indices = rng.choice(n, size=3, replace=False)
        sample_batch = LimbRayBatch(
            batch.d_body[sample_indices],
            batch.lever_arm_body[sample_indices],
            [batch.cam_ids[i] for i in sample_indices],
        )

        # Quick minimal solve (3-5 iterations)
        fit_min = solve_attitude_gn(
            lat_deg, lon_deg, h_meas_m, yaw_rad, p_init, r_init,
            sample_batch, max_iter=5, tol_rad=1e-5, damping=1e-4
        )
        if fit_min["status"] != "CONVERGED":
            continue

        cand_p = fit_min["pitch_rad"]
        cand_r = fit_min["roll_rad"]

        # Evaluate candidate against all rays
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

    # If RANSAC found consensus, refine on all inliers with Huber loss
    if best_inlier_mask is not None and best_inlier_count >= min_inliers:
        inlier_indices = np.where(best_inlier_mask)[0]
        inlier_batch = LimbRayBatch(
            batch.d_body[inlier_indices],
            batch.lever_arm_body[inlier_indices],
            [batch.cam_ids[i] for i in inlier_indices],
            weights=batch.weights[inlier_indices],
            uvs=batch.uvs[inlier_indices],
        )
        fit_final = solve_attitude_gn(
            lat_deg, lon_deg, h_meas_m, yaw_rad,
            best_candidate_p, best_candidate_r,
            inlier_batch,
            max_iter=40, tol_rad=1e-6, damping=damping,
            huber_delta_rad=huber_delta_rad,
        )
        status = fit_final["status"]
        p_final = fit_final["pitch_rad"]
        r_final = fit_final["roll_rad"]
        cov_deg2 = fit_final["covariance_deg2"]
        cov_rad2 = fit_final["covariance_rad2"]
        rms_urad = fit_final["rms_residual_urad"]
        iters = fit_final["iterations"]
        final_inliers = inlier_indices
    else:
        # Fallback to direct Huber LM on all rays
        fit_final = solve_attitude_gn(
            lat_deg, lon_deg, h_meas_m, yaw_rad,
            p_init, r_init,
            batch,
            max_iter=40, tol_rad=1e-6, damping=damping,
            huber_delta_rad=huber_delta_rad,
        )
        status = fit_final["status"]
        p_final = fit_final["pitch_rad"]
        r_final = fit_final["roll_rad"]
        cov_deg2 = fit_final["covariance_deg2"]
        cov_rad2 = fit_final["covariance_rad2"]
        rms_urad = fit_final["rms_residual_urad"]
        iters = fit_final["iterations"]
        # Inliers according to threshold
        res = compute_cone_residuals(lat_deg, lon_deg, h_meas_m, yaw_rad, p_final, r_final, batch)
        final_inliers = np.where(np.abs(res) <= inlier_threshold_rad)[0]

    # Inlier counts by camera
    inlier_counts_by_cam: dict[str, int] = {c: 0 for c in all_cams}
    for idx in final_inliers:
        cam_id = batch.cam_ids[idx]
        if cam_id in inlier_counts_by_cam:
            inlier_counts_by_cam[cam_id] += 1

    return {
        "status": status,
        "roll_deg": math.degrees(r_final),
        "pitch_deg": math.degrees(p_final),
        "roll_rad": r_final,
        "pitch_rad": p_final,
        "covariance_deg2": cov_deg2,
        "covariance_rad2": cov_rad2,
        "rms_residual_urad": rms_urad,
        "inlier_count": len(final_inliers),
        "inlier_count_by_cam": inlier_counts_by_cam,
        "inliers": final_inliers,
        "total_rays": n,
        "iterations": iters,
    }


def estimate_attitude_from_images(
    images_by_cam: dict[str, np.ndarray],
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_meas_m: float,
    yaw_rad: float,
    prior_aircraft_state: dict[str, Any] | None = None,
    step: int = 8,
    search_band_px: int = 50,
    min_gradient: float = 0.015,
    sensor_config: SensorConfig | None = None,
    use_ransac: bool = True,
    rng_seed: int | None = 20260925,
) -> dict[str, Any]:
    """End-to-end attitude estimator from multi-camera imagery.

    Steps:
      1. Detect Earth-limb edges in each camera.
      2. Undistort points (if sensor_config specifies distortion) and form ray batch.
      3. Fit (roll, pitch) via RANSAC + Huber Levenberg-Marquardt.
    """
    prior_cam_poses = None
    initial_guess_rad = None
    if prior_aircraft_state is not None:
        p_guess_rad = math.radians(prior_aircraft_state.get("pitch_deg", 0.0))  # default-ok: optional fallback default
        r_guess_rad = math.radians(prior_aircraft_state.get("roll_deg", 0.0))  # default-ok: optional fallback default
        initial_guess_rad = (p_guess_rad, r_guess_rad)
        prior_cam_poses = rig.compute_poses(
            lat_deg, lon_deg, h_meas_m, yaw_rad, p_guess_rad, r_guess_rad
        )

    edges_by_cam: dict[str, list[dict[str, float]]] = {}
    for cam_id, img in images_by_cam.items():
        prior_p = prior_cam_poses[cam_id] if prior_cam_poses else None
        edges = detect_camera_edges(
            img, cam_id, rig.camera_model,
            prior_pose=prior_p,
            step=step,
            search_band_px=search_band_px,
            min_gradient=min_gradient,
        )
        edges_by_cam[cam_id] = edges

    batch = edges_to_ray_batch(edges_by_cam, rig, sensor_config=sensor_config)

    if use_ransac:
        result = ransac_fit_attitude(
            lat_deg, lon_deg, h_meas_m, yaw_rad, batch,
            initial_guess_rad=initial_guess_rad,
            rng_seed=rng_seed,
        )
    else:
        p_init = initial_guess_rad[0] if initial_guess_rad else 0.0
        r_init = initial_guess_rad[1] if initial_guess_rad else 0.0
        fit = solve_attitude_gn(
            lat_deg, lon_deg, h_meas_m, yaw_rad, p_init, r_init, batch,
            max_iter=25, tol_rad=1e-12, huber_delta_rad=math.radians(0.05)
        )
        all_cams = ("F", "R", "B", "L", "U", "D")
        inlier_counts = {c: 0 for c in all_cams}
        for c in batch.cam_ids:
            if c in inlier_counts:
                inlier_counts[c] += 1
        result = {
            "status": fit["status"],
            "roll_deg": fit["roll_deg"],
            "pitch_deg": fit["pitch_deg"],
            "roll_rad": fit["roll_rad"],
            "pitch_rad": fit["pitch_rad"],
            "covariance_deg2": fit["covariance_deg2"],
            "covariance_rad2": fit["covariance_rad2"],
            "rms_residual_urad": fit["rms_residual_urad"],
            "inlier_count": batch.n_rays,
            "inlier_count_by_cam": inlier_counts,
            "total_rays": batch.n_rays,
            "iterations": fit["iterations"],
        }

    result["edges_by_cam_count"] = {c: len(edges_by_cam.get(c, [])) for c in images_by_cam}
    return result
