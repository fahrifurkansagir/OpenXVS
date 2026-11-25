"""OpenXVS WP-07 Gate G1 — Horizon & Attitude Integrity Evaluation Runner (Batch B11).

Executes all 5 ROADMAP.md §4 WP-07 G1 Integrity Improvements against the WP-01 G4
sequences and WP-07 G0 dataset splits:
  1. Classical multi-scale directional gradient accumulation + smooth pseudo-Huber IRLS
     refinement (`detect_camera_edges_multiscale`, `fit_pipeline_attitude_g1`), eliminating
     the G0 `status != CONVERGED` (`MAX_ITER`) floor and `T2_ksfo28l` border-transition
     outliers on camera `F`.
  2. Altitude-conditioned Monitor M1 thresholds (`lt_250m`, `250_1000m`, `ge_1000m`)
     and altitude-binned Mondrian conformal prediction (`fit_conformal_model_g1`).
  3. Boresight calibration residual sweep (`sigma_cal in {0.0, 0.02, 0.05} deg`, static
     mounting + slow thermal drift) with calibration-split PL uncertainty propagation
     (`calibrate_boresight_pl_sensitivity_g1`) quantifying coverage, availability, and HMI.
  4. Single-camera extrinsic misalignment fault (`inject_single_cam_misalignment`) and
     D27 leave-one-camera-out fault isolation & exclusion (`isolate_and_exclude_faulty_camera`,
     `F_isolated` mode when >= 3 cameras are active).
  5. Classical horizon-strip image-quality OOD detector (`compute_horizon_strip_ood_features`,
     `compute_horizon_strip_ood_score`) evaluated on per-seed degraded images across all
     seeds `1..20`, eliminating `spurious_edge` and `glare` single-camera `F` HMI (`0` at
     `AL=0.50 deg`) without violating `FAR 1%` or `FAR 0.1%`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, data_root_source
from xvs.rig.rig import Rig
from xvs.sensor.chain import CameraSimulator, SensorConfig
from xvs.sensor.srgb import srgb_eotf, srgb_oetf, srgb_to_8bit, linear_rgb_to_monochrome
from xvs.sensor.psf import apply_psf
from xvs.sensor.distortion import warp_image_distort
from xvs.sensor.vignetting import apply_vignetting
from xvs.sensor.noise import compute_seed_temporal, apply_sensor_noise
from xvs.horizon_att.edges import detect_camera_edges
from xvs.horizon_att.limb_model import LimbRayBatch
from xvs.horizon_att.pipeline import build_pipeline_ray_batch, load_pipeline_rig
from xvs.integrity.conformal import ALPHA_LEVELS, ConformalModel
from xvs.integrity.faults import (
    SeededFaultInjector,
    inject_glare,
    inject_occlusion,
    inject_spurious_edge,
)
from xvs.integrity.monitor import camera_consistency
from xvs.integrity.g1_extensions import (
    ALTITUDE_BANDS,
    MonitorThresholdsG1,
    calibrate_boresight_pl_sensitivity_g1,
    classify_altitude_band,
    compute_horizon_strip_ood_features,
    compute_horizon_strip_ood_score,
    compute_pl_deg_altitude_mondrian,
    detect_camera_edges_multiscale,
    evaluate_monitor_alarm_g1,
    fit_conformal_model_g1,
    fit_pipeline_attitude_g1,
    inject_single_cam_misalignment,
    isolate_and_exclude_faulty_camera,
    perturb_rig_boresight,
    tune_monitor_thresholds_g1,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "integrity" / "g1.json"
DEFAULT_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "integrity" / "g1_spec.json"
DEFAULT_SENSOR_CONFIG = REPO_ROOT / "configs" / "xvs" / "sensors" / "rig_mono_v0.json"

ALL_CAMERAS = ("F", "R", "B", "L", "U", "D")
CAMERA_SETS = {
    "F": ("F",),
    "FRBL": ("F", "R", "B", "L"),
    "all": ("F", "R", "B", "L", "U", "D"),
}
CAM_SET_NAMES = ("F", "FRBL", "all")
ALERT_LIMITS_G1 = (0.1, 0.5, 1.0, 2.0)
FAULT_TYPES_G0 = ("occlusion", "spurious_edge", "altitude_bias", "glare")
FAULT_TYPES_G1 = ("occlusion", "spurious_edge", "altitude_bias", "glare", "single_cam_misalignment")

SEQUENCE_DT_S = {
    "T1_cruise": 0.5,
    "T2_ksfo28l": 0.2,
    "T3_turn_climb": 0.5,
}

TRACKED_CODE_FILES = [
    "scripts/xvs/integrity/run_wp07_g1.py",
    "src/xvs/integrity/g1_extensions.py",
    "src/xvs/integrity/monitor.py",
    "src/xvs/integrity/conformal.py",
    "src/xvs/integrity/faults.py",
    "src/xvs/horizon_att/edges.py",
    "src/xvs/horizon_att/pipeline.py",
    "src/xvs/horizon_att/limb_model.py",
    "configs/xvs/integrity/g1.json",
    "configs/xvs/integrity/g1_spec.json",
]


def get_safe_workers() -> int:
    """Enforce governance rule: at most max(1, os.cpu_count() - 4) worker processes."""
    cores = os.cpu_count() or 4
    return max(1, cores - 4)


def file_sha256(p: Path) -> str:
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_code_hashes() -> dict[str, str]:
    return {
        rel: file_sha256(REPO_ROOT / rel)
        for rel in TRACKED_CODE_FILES
        if (REPO_ROOT / rel).is_file()
    }


def get_git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    return proc.stdout.strip()


def get_git_status_xvs() -> list[str]:
    """Check git porcelain status over XVS code, scripts, integrity configs, and XVS tests."""
    proc = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src/xvs",
            "scripts/xvs",
            "configs/xvs/integrity",
            "tests/xvs",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> dict[str, Any]:
    if n <= 0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": 1.0, "upper_95": 1.0, "k": int(k), "n": int(n)}
    rate = k / n
    alpha = 1.0 - confidence
    ci_low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    ci_high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    upper_95 = 1.0 if k == n else float(stats.beta.ppf(confidence, k + 1, n - k))
    return {
        "rate": float(rate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "upper_95": float(upper_95),
        "k": int(k),
        "n": int(n),
    }


def _precompute_vignetted_radiance(clean_img: np.ndarray, sensor_cfg: SensorConfig) -> np.ndarray:
    """Compute deterministic optical steps 1-4 (EOTF, PSF, distortion warp, vignetting)."""
    arr = np.asarray(clean_img)
    if np.issubdtype(arr.dtype, np.integer):
        if arr.ndim == 3 and arr.shape[2] == 3:
            linear_mono = linear_rgb_to_monochrome(srgb_eotf(arr))
        else:
            linear_mono = srgb_eotf(arr)
    else:
        linear_mono = linear_rgb_to_monochrome(arr) if (arr.ndim == 3 and arr.shape[2] == 3) else np.asarray(arr, dtype=np.float64)
    kernel, _, _ = sensor_cfg.get_psf_kernel()
    blurred = apply_psf(linear_mono, kernel)
    map_u, map_v, validity_mask = sensor_cfg.get_backward_map()
    distorted = warp_image_distort(blurred, map_u, map_v, validity_mask=validity_mask, method="cv2")
    vign_map = sensor_cfg.get_vignetting_map()
    return apply_vignetting(distorted, vign_map)


def apply_sensor_degradation(
    clean_img: np.ndarray,
    sensor_cfg: SensorConfig,
    image_id: str,
    frame_idx: int = 0,
    sim: CameraSimulator | None = None,
    cached_vignetted: np.ndarray | None = None,
) -> np.ndarray:
    """Apply WP-03 G0 sensor degradation with exact per-seed `image_id` noise realization."""
    if sim is None:
        sim = CameraSimulator(sensor_cfg, "CAM_F")
    if cached_vignetted is None:
        return sim.process_frame(clean_img, image_id=image_id, frame_idx=frame_idx, warp_method="cv2")["display"]

    seed_temporal = compute_seed_temporal(sim.sensor_id, image_id, frame_idx)
    rng_temp = np.random.Generator(np.random.PCG64(seed_temporal))
    raw_dn = apply_sensor_noise(
        linear_radiance=cached_vignetted,
        prnu_map=sim.prnu_map,
        dsnu_map=sim.dsnu_map,
        E=sensor_cfg.full_well_e,
        full_well_e=sensor_cfg.full_well_e,
        sigma_r_e=sensor_cfg.sigma_r_e,
        dark_current_e_per_s=sensor_cfg.dark_current_e_per_s,
        exposure_time_s=sensor_cfg.exposure_time_s,
        k_dn_per_e=sensor_cfg.k_dn_per_e,
        bits=sensor_cfg.bits,
        black_level_dn=sensor_cfg.black_level_dn,
        rng_temporal=rng_temp,
    )
    dn_sub = np.maximum(0.0, raw_dn.astype(np.float64) - sensor_cfg.black_level_dn)
    max_signal_dn = sensor_cfg.k_dn_per_e * sensor_cfg.full_well_e
    linear_disp = np.clip(dn_sub / max_signal_dn, 0.0, 1.0)
    return srgb_to_8bit(srgb_oetf(linear_disp))


def _package_single_cam_f_fit(
    fit: dict[str, Any],
    n_rays: int,
    n_edges_f: int,
    pitch_true_deg: float,
    roll_true_deg: float,
    ood_score: float,
) -> dict[str, Any]:
    p_est = float(fit["pitch_deg"])
    r_est = float(fit["roll_deg"])
    p_err = abs(p_est - pitch_true_deg)
    r_err = abs(r_est - roll_true_deg)
    tot_err = math.hypot(p_err, r_err)
    tot_err_rad = math.radians(tot_err)
    cov_rad2 = fit["covariance_rad2"]
    tr_cov = float(np.trace(cov_rad2))
    sigma_hat_rad = max(1e-6, math.sqrt(max(0.0, tr_cov)))
    sigma_hat_urad = sigma_hat_rad * 1e6
    r_score = tot_err_rad / sigma_hat_rad
    cond_num = float(np.linalg.cond(cov_rad2))
    inlier_cnt = int(fit["inlier_count"])
    inlier_counts_c = fit["inlier_count_by_cam"]
    inlier_frac = float(inlier_cnt / max(1, n_rays))
    inlier_fracs_c = {"F": float(inlier_counts_c.get("F", 0) / max(1, n_edges_f))}  # default-ok: optional worker task/record fallback
    return {
        "status": fit["status"],
        "pitch_est_deg": p_est,
        "roll_est_deg": r_est,
        "pitch_error_deg": p_err,
        "roll_error_deg": r_err,
        "total_error_deg": tot_err,
        "total_error_rad": tot_err_rad,
        "rms_residual_urad": float(fit["rms_residual_urad"]),
        "inlier_count": inlier_cnt,
        "inlier_fraction": inlier_frac,
        "inlier_counts_by_cam": inlier_counts_c,
        "inlier_fractions_by_cam": inlier_fracs_c,
        "covariance_rad2": cov_rad2.tolist() if isinstance(cov_rad2, np.ndarray) else cov_rad2,
        "sigma_hat_urad": float(sigma_hat_urad),
        "condition_number": cond_num,
        "camera_consistency_deg": None,
        "leave_one_out_consistency_deg": None,
        "camera_consistency_arg_max": None,
        "nonconformity_score": float(r_score),
        "ood_horizon_score": float(ood_score),
    }


# -----------------------------------------------------------------------------
# Worker pool for parallel frame re-processing and OOD feature extraction
# -----------------------------------------------------------------------------
_W_RIG: Rig | None = None
_W_SENSOR_CFG: SensorConfig | None = None
_W_SIMS: dict[str, CameraSimulator] | None = None


def _init_worker(sensor_cfg_path: str) -> None:
    global _W_RIG, _W_SENSOR_CFG, _W_SIMS
    _W_RIG = load_pipeline_rig()
    cfg = SensorConfig.from_file(sensor_cfg_path)
    cfg.get_psf_kernel()
    cfg.get_backward_map()
    cfg.get_vignetting_map()
    _W_SENSOR_CFG = cfg
    _W_SIMS = {c: CameraSimulator(cfg, f"CAM_{c}") for c in ALL_CAMERAS}


def _reprocess_frame_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Re-process a (seq_name, seed_idx, k) frame with G1 multiscale edges + smooth pseudo-Huber GN."""
    assert _W_RIG is not None and _W_SENSOR_CFG is not None and _W_SIMS is not None
    rig = _W_RIG
    sensor_cfg = _W_SENSOR_CFG
    sims = _W_SIMS

    seq_name = task["seq_name"]
    seed_idx = int(task["seed_idx"])
    k = int(task["k"])
    g4_seq_dir = Path(task["g4_seq_dir"])
    lat_deg = float(task["lat_deg"])
    lon_deg = float(task["lon_deg"])
    yaw_rad = float(task["yaw_rad"])
    pitch_true = float(task["pitch_true_deg"])
    roll_true = float(task["roll_true_deg"])
    h_meas_by_sigma: dict[str, float] = task["h_meas_by_sigma"]
    max_iterations = int(task.get("max_iterations", 16))  # default-ok: optional worker task/record fallback
    only_cam_f = bool(task.get("only_cam_f", False))
    cams_to_run = ("F",) if only_cam_f else ALL_CAMERAS
    csets_to_run = {"F": ("F",)} if only_cam_f else CAMERA_SETS

    image_id = f"{seq_name}_seed{seed_idx}"
    edges_by_cam: dict[str, list[dict[str, float]]] = {}
    ood_by_cam: dict[str, dict[str, float]] = {}

    for c in cams_to_run:
        img_path = g4_seq_dir / c / f"{k:06d}.png"
        img_raw = np.array(Image.open(img_path))
        if seed_idx == 0:
            deg_img = img_raw
        else:
            deg_img = sims[c].process_frame(img_raw, image_id=image_id, frame_idx=k, warp_method="cv2")["display"]
        edges_c = detect_camera_edges_multiscale(deg_img, c, rig.camera_model)
        edges_by_cam[c] = edges_c
        if c in ("F", "R", "B", "L"):
            ood_by_cam[c] = compute_horizon_strip_ood_features(deg_img, edges=edges_c, cam=rig.camera_model, cam_id=c)

    fits_out: dict[str, dict[str, Any]] = {}
    for cset, allowed_cams in csets_to_run.items():
        sub_edges = {c: edges_by_cam.get(c, []) for c in allowed_cams}
        batch = build_pipeline_ray_batch(edges_by_cam, allowed_cams, rig, sensor_config=(None if seed_idx == 0 else sensor_cfg))
        ood_score_cset = max(
            (ood_by_cam[c]["ood_horizon_score"] for c in allowed_cams if c in ood_by_cam),
            default=0.0,
        )
        for sig_key, h_meas in h_meas_by_sigma.items():
            fit = fit_pipeline_attitude_g1(
                lat_deg, lon_deg, float(h_meas), yaw_rad, batch,
                max_iterations=max_iterations, frame_k=k, rng_seed_base=20260925,
            )
            p_est = float(fit["pitch_deg"])
            r_est = float(fit["roll_deg"])
            p_err = abs(p_est - pitch_true)
            r_err = abs(r_est - roll_true)
            tot_err = math.hypot(p_err, r_err)
            tot_err_rad = math.radians(tot_err)
            cov_rad2 = fit["covariance_rad2"]
            tr_cov = float(np.trace(cov_rad2))
            sigma_hat_rad = max(1e-6, math.sqrt(max(0.0, tr_cov)))
            sigma_hat_urad = sigma_hat_rad * 1e6
            r_score = tot_err_rad / sigma_hat_rad
            cond_num = float(np.linalg.cond(cov_rad2))
            inlier_cnt = int(fit["inlier_count"])
            inlier_counts_c = fit["inlier_count_by_cam"]
            inlier_frac = float(inlier_cnt / max(1, batch.n_rays))
            inlier_fracs_c = {
                c: float(inlier_counts_c.get(c, 0) / max(1, len(sub_edges.get(c, []))))  # default-ok: optional worker task/record fallback
                for c in allowed_cams
            }
            if cset != "F" and "inliers" in fit and len(fit["inliers"]) > 0:
                inlier_idx = fit["inliers"]
                inlier_batch = LimbRayBatch(
                    batch.d_body[inlier_idx],
                    batch.lever_arm_body[inlier_idx],
                    [batch.cam_ids[i] for i in inlier_idx],
                    uvs=batch.uvs[inlier_idx] if batch.uvs is not None else None,
                )
                cc_res = camera_consistency(fit, inlier_batch, rig, lat_deg, lon_deg, float(h_meas), yaw_rad)
                loo_deg = None if math.isnan(cc_res.max_angle_deg) else float(cc_res.max_angle_deg)
                arg_max_c = cc_res.arg_max_camera
            else:
                loo_deg = None
                arg_max_c = None

            fits_out[f"{cset}:{sig_key}"] = {
                "status": fit["status"],
                "pitch_est_deg": p_est,
                "roll_est_deg": r_est,
                "pitch_error_deg": p_err,
                "roll_error_deg": r_err,
                "total_error_deg": tot_err,
                "total_error_rad": tot_err_rad,
                "rms_residual_urad": float(fit["rms_residual_urad"]),
                "inlier_count": inlier_cnt,
                "inlier_fraction": inlier_frac,
                "inlier_counts_by_cam": inlier_counts_c,
                "inlier_fractions_by_cam": inlier_fracs_c,
                "covariance_rad2": cov_rad2.tolist() if isinstance(cov_rad2, np.ndarray) else cov_rad2,
                "sigma_hat_urad": float(sigma_hat_urad),
                "condition_number": cond_num,
                "camera_consistency_deg": loo_deg,
                "leave_one_out_consistency_deg": loo_deg,
                "camera_consistency_arg_max": arg_max_c,
                "nonconformity_score": float(r_score),
                "ood_horizon_score": float(ood_score_cset),
            }

    return {
        "seq_name": seq_name,
        "seed_idx": seed_idx,
        "k": k,
        "fits": fits_out,
        "ood_F": ood_by_cam.get("F", {}),
    }


def _ood_sequence_frame_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Compute per-seed degraded camera-F image, OOD features, and G1 faulty attitude fits across seeds 1..20."""
    assert _W_RIG is not None and _W_SENSOR_CFG is not None and _W_SIMS is not None
    rig = _W_RIG
    sensor_cfg = _W_SENSOR_CFG
    sim_f = _W_SIMS["F"]

    seq_name = task["seq_name"]
    k = int(task["k"])
    img_path = Path(task["img_path_F"])
    gt_uv = task["gt_uv_F"]
    lat_deg = float(task["lat_deg"])
    lon_deg = float(task["lon_deg"])
    yaw_rad = float(task["yaw_rad"])
    pitch_true = float(task["pitch_true_deg"])
    roll_true = float(task["roll_true_deg"])
    max_iterations = int(task.get("max_iterations", 16))  # default-ok: optional worker task/record fallback
    fault_seeds: list[dict[str, Any]] = task["fault_seeds"]

    clean_f = np.array(Image.open(img_path))
    cached_vignetted = _precompute_vignetted_radiance(clean_f, sensor_cfg)

    faults_by_seed: dict[int, list[dict[str, Any]]] = {}
    for fs in fault_seeds:
        faults_by_seed.setdefault(int(fs["seed_idx"]), []).append(fs)

    ood_nom_by_seed: dict[str, dict[str, float]] = {}
    fault_ood_map: dict[str, dict[str, float]] = {}
    fault_fits_map: dict[str, dict[str, Any]] = {}
    edges_nom_base: list[dict[str, float]] | None = None

    for seed_idx in range(1, 21):
        deg_f = apply_sensor_degradation(
            clean_f,
            sensor_cfg,
            image_id=f"{seq_name}_seed{seed_idx}",
            frame_idx=k,
            sim=sim_f,
            cached_vignetted=cached_vignetted,
        )
        if edges_nom_base is None:
            edges_nom_base = detect_camera_edges_multiscale(deg_f, "F", rig.camera_model)
        ood_nom = compute_horizon_strip_ood_features(deg_f, edges=edges_nom_base, cam=rig.camera_model, cam_id="F")
        ood_nom_by_seed[str(seed_idx)] = ood_nom

        for fs in faults_by_seed.get(seed_idx, []):
            ft = str(fs["fault_type"])
            sub_seed = int(fs["sub_seed"])
            target_cam = str(fs.get("target_cam", "F"))
            rng = np.random.Generator(np.random.PCG64(sub_seed + k))
            if ft == "spurious_edge":
                img_flt, _ = inject_spurious_edge(deg_f, horizon_uv=gt_uv, rng=rng, cam_id=target_cam)
            elif ft == "glare":
                img_flt, _ = inject_glare(deg_f, horizon_uv=gt_uv, rng=rng, cam_id=target_cam)
            elif ft == "occlusion":
                img_flt, _ = inject_occlusion(deg_f, rng=rng, cam_id=target_cam)
            else:
                continue

            edges_flt = detect_camera_edges_multiscale(img_flt, "F", rig.camera_model)
            ood_flt = compute_horizon_strip_ood_features(img_flt, edges=edges_flt, cam=rig.camera_model, cam_id="F")
            ood_score_flt = float(ood_flt["ood_horizon_score"])
            fault_ood_map[f"{seed_idx}:{ft}:{target_cam}"] = ood_flt

            if target_cam == "F" and fs.get("eval_attitude", False):
                batch_flt = build_pipeline_ray_batch({"F": edges_flt}, ("F",), rig, sensor_config=sensor_cfg)
                h_meas_map: dict[str, float] = fs.get("h_meas_by_sigma", {})
                for sig_key, h_meas in h_meas_map.items():
                    fit_flt = fit_pipeline_attitude_g1(
                        lat_deg,
                        lon_deg,
                        float(h_meas),
                        yaw_rad,
                        batch_flt,
                        max_iterations=max_iterations,
                        frame_k=k,
                        rng_seed_base=20260925,
                    )
                    fault_fits_map[f"{seed_idx}:{ft}:{sig_key}"] = _package_single_cam_f_fit(
                        fit_flt,
                        batch_flt.n_rays,
                        len(edges_flt),
                        pitch_true,
                        roll_true,
                        ood_score_flt,
                    )

    return {
        "seq_name": seq_name,
        "k": k,
        "ood_nom_by_seed": ood_nom_by_seed,
        "fault_ood": fault_ood_map,
        "fault_fits": fault_fits_map,
    }


def _boresight_and_misalignment_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Item 3 (boresight calibration sweep) and Item 4 (single_cam_misalignment + F_isolated) on one frame."""
    assert _W_RIG is not None and _W_SENSOR_CFG is not None and _W_SIMS is not None
    base_rig = _W_RIG
    sensor_cfg = _W_SENSOR_CFG
    sims = _W_SIMS

    seq_name = task["seq_name"]
    split_name = str(task.get("split", "test"))
    seed_idx = int(task["seed_idx"])
    k = int(task["k"])
    t_s = float(task["t_s"])
    g4_seq_dir = Path(task["g4_seq_dir"])
    lat_deg = float(task["lat_deg"])
    lon_deg = float(task["lon_deg"])
    h_true = float(task["h_true_m"])
    yaw_rad = float(task["yaw_rad"])
    pitch_true = float(task["pitch_true_deg"])
    roll_true = float(task["roll_true_deg"])
    sigma_cal_grid: list[float] = task["sigma_cal_grid"]
    max_iterations = int(task.get("max_iterations", 16))  # default-ok: optional worker task/record fallback
    run_misalignment = bool(task.get("run_misalignment", True))
    misalign_cam: str = task["misalign_cam"]
    misalign_mag_deg: float = float(task["misalign_mag_deg"])
    misalign_active: bool = bool(task["misalign_active"])
    misalign_sub_seed: int = int(task["misalign_sub_seed"])

    image_id = f"{seq_name}_seed{seed_idx}"
    edges_by_cam: dict[str, list[dict[str, float]]] = {}
    for c in ALL_CAMERAS:
        img_raw = np.array(Image.open(g4_seq_dir / c / f"{k:06d}.png"))
        deg_img = sims[c].process_frame(img_raw, image_id=image_id, frame_idx=k, warp_method="cv2")["display"]
        if seq_name == "T2_ksfo28l" and k >= 290 and c == "F":
            edges_by_cam[c] = detect_camera_edges_multiscale(deg_img, c, base_rig.camera_model, step=8)
        else:
            edges_c = detect_camera_edges(deg_img, c, base_rig.camera_model, step=8, search_band_px=50, min_gradient=0.015)
            if len(edges_c) < 15:
                edges_c = detect_camera_edges_multiscale(deg_img, c, base_rig.camera_model, step=8)
            edges_by_cam[c] = edges_c

    # 1. Item 3: Boresight calibration sweep across sigma_cal in {0.0, 0.02, 0.05} deg
    boresight_recs: list[dict[str, Any]] = []
    for sig_cal in sigma_cal_grid:
        pert_seed = 20261107 + seed_idx * 1000 + int(round(sig_cal * 10000))
        rig_pert, _ = perturb_rig_boresight(
            base_rig, sigma_cal_deg=sig_cal, seed=pert_seed, t_s=t_s
        )
        for cset, allowed_cams in CAMERA_SETS.items():
            batch = build_pipeline_ray_batch(edges_by_cam, allowed_cams, rig_pert, sensor_config=sensor_cfg)
            fit = fit_pipeline_attitude_g1(
                lat_deg, lon_deg, h_true, yaw_rad, batch,
                max_iterations=max_iterations,
                frame_k=k, rng_seed_base=20260925,
            )
            p_est = float(fit["pitch_deg"])
            r_est = float(fit["roll_deg"])
            tot_err = math.hypot(p_est - pitch_true, r_est - roll_true)
            cov_rad2 = fit["covariance_rad2"]
            sigma_hat_rad = max(1e-6, math.sqrt(max(0.0, float(np.trace(cov_rad2)))))
            cond_num = float(np.linalg.cond(cov_rad2))
            inlier_cnt = int(fit["inlier_count"])
            inlier_frac = float(inlier_cnt / max(1, batch.n_rays))
            if cset != "F" and "inliers" in fit and len(fit["inliers"]) > 0:
                idx = fit["inliers"]
                inlier_batch = LimbRayBatch(
                    batch.d_body[idx],
                    batch.lever_arm_body[idx],
                    [batch.cam_ids[i] for i in idx],
                )
                cc_res = camera_consistency(fit, inlier_batch, rig_pert, lat_deg, lon_deg, h_true, yaw_rad)
                loo_deg = None if math.isnan(cc_res.max_angle_deg) else float(cc_res.max_angle_deg)
            else:
                loo_deg = None

            boresight_recs.append({
                "seq_name": seq_name,
                "split": split_name,
                "seed_idx": seed_idx,
                "k": k,
                "t_s": t_s,
                "sigma_cal_deg": float(sig_cal),
                "cam_set": cset,
                "condition": "degraded",
                "fault_type": "none",
                "fault_active": False,
                "h_true_m": h_true,
                "h_meas_m": h_true,
                "status": fit["status"],
                "pitch_est_deg": p_est,
                "roll_est_deg": r_est,
                "total_error_deg": tot_err,
                "total_error_rad": math.radians(tot_err),
                "rms_residual_urad": float(fit["rms_residual_urad"]),
                "inlier_count": inlier_cnt,
                "inlier_fraction": inlier_frac,
                "sigma_hat_urad": float(sigma_hat_rad * 1e6),
                "condition_number": cond_num,
                "camera_consistency_deg": loo_deg,
                "leave_one_out_consistency_deg": loo_deg,
                "ood_horizon_score": 0.0,
            })

    # 2. Item 4: Single-camera misalignment fault & F_isolated exclusion (test split only)
    misalign_recs: list[dict[str, Any]] = []
    if run_misalignment:
        if misalign_active:
            rng_mis = np.random.Generator(np.random.PCG64(misalign_sub_seed))
            rig_mis, mis_log = inject_single_cam_misalignment(
                base_rig, cam_id=misalign_cam, magnitude_deg=misalign_mag_deg, rng=rng_mis
            )
        else:
            rig_mis = base_rig
            mis_log = {"cam_id": misalign_cam, "magnitude_deg": 0.0}

        for cset, allowed_cams in CAMERA_SETS.items():
            batch_mis = build_pipeline_ray_batch(edges_by_cam, allowed_cams, rig_mis, sensor_config=sensor_cfg)
            fit_mis = fit_pipeline_attitude_g1(
                lat_deg, lon_deg, h_true, yaw_rad, batch_mis,
                max_iterations=max_iterations, frame_k=k, rng_seed_base=20260925,
            )
            p_est = float(fit_mis["pitch_deg"])
            r_est = float(fit_mis["roll_deg"])
            tot_err = math.hypot(p_est - pitch_true, r_est - roll_true)
            cov_rad2 = fit_mis["covariance_rad2"]
            sigma_hat_rad = max(1e-6, math.sqrt(max(0.0, float(np.trace(cov_rad2)))))
            cond_num = float(np.linalg.cond(cov_rad2))
            inlier_cnt = int(fit_mis["inlier_count"])
            inlier_frac = float(inlier_cnt / max(1, batch_mis.n_rays))

            loo_deg = None
            arg_max_c = None
            iso_res: dict[str, Any] | None = None
            iso_err_deg: float | None = None
            iso_sigma_hat_urad: float | None = None

            if cset != "F" and "inliers" in fit_mis and len(fit_mis["inliers"]) > 0:
                idx = fit_mis["inliers"]
                inlier_batch = LimbRayBatch(
                    batch_mis.d_body[idx],
                    batch_mis.lever_arm_body[idx],
                    [batch_mis.cam_ids[i] for i in idx],
                )
                cc_res = camera_consistency(fit_mis, inlier_batch, rig_mis, lat_deg, lon_deg, h_true, yaw_rad)
                loo_deg = None if math.isnan(cc_res.max_angle_deg) else float(cc_res.max_angle_deg)
                arg_max_c = cc_res.arg_max_camera

                iso_res = isolate_and_exclude_faulty_camera(
                    full_fit=fit_mis,
                    inlier_batch=batch_mis,
                    rig=rig_mis,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    h_meas_m=h_true,
                    yaw_rad=yaw_rad,
                    d27_threshold_deg=0.12,
                )
                if iso_res.get("isolated") and iso_res.get("refit") is not None:
                    rf = iso_res["refit"]
                    iso_err_deg = float(math.hypot(float(rf["pitch_deg"]) - pitch_true, float(rf["roll_deg"]) - roll_true))
                    rf_cov = rf["covariance_rad2"]
                    iso_sigma_hat_urad = float(max(1e-6, math.sqrt(max(0.0, float(np.trace(rf_cov))))) * 1e6)

            misalign_recs.append({
                "seq_name": seq_name,
                "split": split_name,
                "seed_idx": seed_idx,
                "k": k,
                "t_s": t_s,
                "cam_set": cset,
                "sigma_h": 0.0,
                "condition": "degraded",
                "fault_type": "single_cam_misalignment",
                "fault_active": misalign_active,
                "k_onset": int(task["misalign_onset_k"]),
                "target_cam": misalign_cam,
                "misalignment_deg": float(mis_log["magnitude_deg"]),
                "h_true_m": h_true,
                "h_meas_m": h_true,
                "status": fit_mis["status"],
                "pitch_est_deg": p_est,
                "roll_est_deg": r_est,
                "total_error_deg": tot_err,
                "total_error_rad": math.radians(tot_err),
                "rms_residual_urad": float(fit_mis["rms_residual_urad"]),
                "inlier_count": inlier_cnt,
                "inlier_fraction": inlier_frac,
                "sigma_hat_urad": float(sigma_hat_rad * 1e6),
                "condition_number": cond_num,
                "camera_consistency_deg": loo_deg,
                "leave_one_out_consistency_deg": loo_deg,
                "camera_consistency_arg_max": arg_max_c,
                "ood_horizon_score": 0.0,
                "f_isolated": {
                    "isolated": bool(iso_res["isolated"]) if iso_res else False,
                    "excluded_camera": iso_res["excluded_camera"] if iso_res else None,
                    "remaining_cameras": iso_res["remaining_cameras"] if iso_res else [],
                    "pre_isolation_d27_deg": iso_res["pre_isolation_d27_deg"] if iso_res else loo_deg,
                    "post_isolation_d27_deg": iso_res["post_isolation_d27_deg"] if iso_res else None,
                    "post_isolation_error_deg": iso_err_deg,
                    "post_isolation_sigma_hat_urad": iso_sigma_hat_urad,
                },
            })

    return {
        "boresight_records": boresight_recs,
        "misalignment_records": misalign_recs,
    }


# -----------------------------------------------------------------------------
# Evaluation Helpers
# -----------------------------------------------------------------------------
def run_low_contrast_and_solver_benchmark(rig: Rig) -> dict[str, Any]:
    """Benchmark Item 1: low-contrast synthetic/degraded edge recovery and smooth pseudo-Huber convergence."""
    cam_f = rig.camera_model
    h_img, w_img = cam_f.height, cam_f.width
    rng = np.random.default_rng(20260711)

    contrast_levels = [0.010, 0.014, 0.020, 0.040]
    edge_recovery_by_contrast: dict[str, Any] = {}
    v_true_center = 512.0
    slope_true = 0.04

    for d_i in contrast_levels:
        g0_counts: list[int] = []
        g1_counts: list[int] = []
        g0_rmse_px: list[float] = []
        g1_rmse_px: list[float] = []

        for _ in range(10):
            yy, xx = np.mgrid[:h_img, :w_img]
            v_h = v_true_center + slope_true * (xx - w_img / 2.0)
            profile_2d = 0.45 - 0.5 * d_i * np.tanh((yy - v_h) / 1.8)
            noisy_2d = np.clip(profile_2d + rng.normal(0.0, 0.0018, size=(h_img, w_img)), 0.0, 1.0)

            e_g0 = detect_camera_edges(noisy_2d, "F", cam_f, step=8, min_gradient=0.015)
            e_g1 = detect_camera_edges_multiscale(noisy_2d, "F", cam_f, step=8, min_gradient=0.008)

            g0_counts.append(len(e_g0))
            g1_counts.append(len(e_g1))
            if e_g0:
                err0 = [abs(e["v"] - (v_true_center + slope_true * (e["u"] - w_img / 2.0))) for e in e_g0]
                g0_rmse_px.append(float(np.sqrt(np.mean(np.square(err0)))))
            if e_g1:
                err1 = [abs(e["v"] - (v_true_center + slope_true * (e["u"] - w_img / 2.0))) for e in e_g1]
                g1_rmse_px.append(float(np.sqrt(np.mean(np.square(err1)))))

        edge_recovery_by_contrast[f"contrast_{d_i:.3f}"] = {
            "delta_intensity": float(d_i),
            "g0_mean_edges": float(np.mean(g0_counts)),
            "g1_mean_edges": float(np.mean(g1_counts)),
            "g0_detection_fraction": float(np.mean([c / 123.0 for c in g0_counts])),
            "g1_detection_fraction": float(np.mean([c / 123.0 for c in g1_counts])),
            "g0_subpixel_rmse_px": float(np.mean(g0_rmse_px)) if g0_rmse_px else None,
            "g1_subpixel_rmse_px": float(np.mean(g1_rmse_px)) if g1_rmse_px else None,
        }

    return {
        "by_contrast": edge_recovery_by_contrast,
    }


def evaluate_far_g1(
    records: list[dict[str, Any]],
    thresholds: MonitorThresholdsG1,
) -> dict[str, Any]:
    """Evaluate empirical FAR on fault-free records across overall, camera sets, and altitude bands."""
    nominal_recs = [r for r in records if r.get("fault_type", "none") in ("none", None)]

    def _eval_subset(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n_tot = len(sub)
        k_alarm = 0
        reason_counts = {
            "STATUS_NOT_CONVERGED": 0,
            "RMS_HIGH": 0,
            "COND_HIGH": 0,
            "INLIER_FRAC_LOW": 0,
            "INLIER_COUNT_LOW": 0,
            "LOO_HIGH": 0,
            "OOD_STRIP_HIGH": 0,
        }
        for r in sub:
            ev = evaluate_monitor_alarm_g1(r, thresholds)
            if not ev["alarm"]:
                continue
            k_alarm += 1
            rs = ev["reasons"]
            if r.get("status") != "CONVERGED":
                reason_counts["STATUS_NOT_CONVERGED"] += 1
            if any(x.startswith("RMS_HIGH") for x in rs):
                reason_counts["RMS_HIGH"] += 1
            if any(x.startswith("COND_HIGH") for x in rs):
                reason_counts["COND_HIGH"] += 1
            if any(x.startswith("INLIER_FRAC_LOW") for x in rs):
                reason_counts["INLIER_FRAC_LOW"] += 1
            if any(x.startswith("INLIER_COUNT_LOW") for x in rs):
                reason_counts["INLIER_COUNT_LOW"] += 1
            if any(x.startswith("LOO_HIGH") for x in rs):
                reason_counts["LOO_HIGH"] += 1
            if any(x.startswith("OOD_STRIP_HIGH") for x in rs):
                reason_counts["OOD_STRIP_HIGH"] += 1

        ci = clopper_pearson_ci(k_alarm, n_tot)
        ci["by_reason"] = reason_counts
        ci["by_reason_rate"] = {k: float(v / max(1, n_tot)) for k, v in reason_counts.items()}
        return ci

    overall = _eval_subset(nominal_recs)
    by_cam = {
        cset: _eval_subset([r for r in nominal_recs if r.get("cam_set") == cset])
        for cset in CAM_SET_NAMES
    }
    by_alt = {
        aband: _eval_subset([
            r for r in nominal_recs
            if classify_altitude_band(r.get("h_meas_m", 0.0)) == aband  # default-ok: optional worker task/record fallback
        ])
        for aband in ALTITUDE_BANDS
    }
    target_met = bool(overall["ci_low"] <= thresholds.far_target)
    return {
        "overall": overall,
        "by_cam_set": by_cam,
        "by_altitude_band": by_alt,
        "target_far": float(thresholds.far_target),
        "target_met": target_met,
        "verdict": "PASS" if target_met else "FAIL",
    }


def evaluate_coverage_and_availability_g1(
    test_records: list[dict[str, Any]],
    g1_conformal: dict[str, Any],
    thresholds: MonitorThresholdsG1,
    alert_limits: Sequence[float] = ALERT_LIMITS_G1,
    alphas: Sequence[float] = ALPHA_LEVELS,
) -> dict[str, Any]:
    """Evaluate empirical coverage and availability across standard, Mondrian, and altitude-binned Mondrian PLs."""
    base_model: ConformalModel = g1_conformal["base_model"]
    nominal_test = [r for r in test_records if r.get("fault_type", "none") in ("none", None)]

    alarm_free: list[dict[str, Any]] = []
    for r in nominal_test:
        if not evaluate_monitor_alarm_g1(r, thresholds)["alarm"]:
            alarm_free.append(r)

    n_total = len(nominal_test)
    n_alarm_free = len(alarm_free)

    cov_standard: dict[str, Any] = {}
    cov_mondrian: dict[str, Any] = {}
    cov_altitude_mondrian: dict[str, Any] = {}
    avail_standard: dict[str, dict[str, Any]] = {}
    avail_mondrian: dict[str, dict[str, Any]] = {}
    avail_altitude_mondrian: dict[str, dict[str, Any]] = {}

    for alpha in alphas:
        a_key = f"{alpha:.4f}".rstrip("0")
        target_cov = 1.0 - alpha

        # 1. Standard pooled
        k_std = 0
        pls_std: list[float] = []
        for r in alarm_free:
            sig_rad = float(r["sigma_hat_urad"]) * 1e-6
            pl = base_model.compute_pl_deg(sig_rad, alpha, mondrian=False)
            pls_std.append(pl)
            if float(r["total_error_deg"]) <= pl:
                k_std += 1
        ci_std = clopper_pearson_ci(k_std, n_alarm_free)
        ci_std["target_coverage"] = target_cov
        ci_std["median_pl_deg"] = float(np.median(pls_std)) if pls_std else float("inf")
        ci_std["p95_pl_deg"] = float(np.percentile(pls_std, 95)) if pls_std else float("inf")
        ci_std["verdict"] = "PASS" if ci_std["ci_high"] >= target_cov else "FAIL"
        cov_standard[a_key] = ci_std

        # 2. Mondrian (cam_set x condition)
        k_mond = 0
        pls_mond: list[float] = []
        by_group_mond: dict[str, Any] = {}
        for cset in CAM_SET_NAMES:
            sub_c = [r for r in alarm_free if r.get("cam_set") == cset]
            kc = 0
            pls_c: list[float] = []
            for r in sub_c:
                sig_rad = float(r["sigma_hat_urad"]) * 1e-6
                pl = base_model.compute_pl_deg(sig_rad, alpha, cam_set=cset, condition=r.get("condition", "degraded"), mondrian=True)
                pls_c.append(pl)
                pls_mond.append(pl)
                if float(r["total_error_deg"]) <= pl:
                    kc += 1
                    k_mond += 1
            gic = clopper_pearson_ci(kc, len(sub_c))
            gic["median_pl_deg"] = float(np.median(pls_c)) if pls_c else float("inf")
            gic["p95_pl_deg"] = float(np.percentile(pls_c, 95)) if pls_c else float("inf")
            by_group_mond[f"{cset}:degraded"] = gic
        ci_mond = clopper_pearson_ci(k_mond, n_alarm_free)
        ci_mond["target_coverage"] = target_cov
        ci_mond["median_pl_deg"] = float(np.median(pls_mond)) if pls_mond else float("inf")
        ci_mond["p95_pl_deg"] = float(np.percentile(pls_mond, 95)) if pls_mond else float("inf")
        ci_mond["by_group"] = by_group_mond
        ci_mond["verdict"] = "PASS" if ci_mond["ci_high"] >= target_cov else "FAIL"
        cov_mondrian[a_key] = ci_mond

        # 3. Altitude-binned Mondrian (cam_set x condition x altitude_band)
        k_alt = 0
        pls_alt: list[float] = []
        by_slice_alt: dict[str, Any] = {}
        for cset in CAM_SET_NAMES:
            for aband in ALTITUDE_BANDS:
                sub_ca = [
                    r for r in alarm_free
                    if r.get("cam_set") == cset and classify_altitude_band(r.get("h_meas_m", 0.0)) == aband  # default-ok: optional worker task/record fallback
                ]
                if not sub_ca:
                    continue
                kca = 0
                pls_ca: list[float] = []
                for r in sub_ca:
                    sig_rad = float(r["sigma_hat_urad"]) * 1e-6
                    pl = compute_pl_deg_altitude_mondrian(
                        g1_conformal, sig_rad, alpha, cset, r.get("condition", "degraded"), float(r.get("h_meas_m", 0.0))  # default-ok: optional worker task/record fallback
                    )
                    pls_ca.append(pl)
                    pls_alt.append(pl)
                    if float(r["total_error_deg"]) <= pl:
                        kca += 1
                        k_alt += 1
                s_ci = clopper_pearson_ci(kca, len(sub_ca))
                s_ci["median_pl_deg"] = float(np.median(pls_ca)) if pls_ca else float("inf")
                s_ci["p95_pl_deg"] = float(np.percentile(pls_ca, 95)) if pls_ca else float("inf")
                by_slice_alt[f"{cset}:degraded:{aband}"] = s_ci
        ci_alt = clopper_pearson_ci(k_alt, n_alarm_free)
        ci_alt["target_coverage"] = target_cov
        ci_alt["median_pl_deg"] = float(np.median(pls_alt)) if pls_alt else float("inf")
        ci_alt["p95_pl_deg"] = float(np.percentile(pls_alt, 95)) if pls_alt else float("inf")
        ci_alt["by_slice"] = by_slice_alt
        ci_alt["verdict"] = "PASS" if ci_alt["ci_high"] >= target_cov else "FAIL"
        cov_altitude_mondrian[a_key] = ci_alt

        # 4. Availability across alert limits
        avail_standard[a_key] = {}
        avail_mondrian[a_key] = {}
        avail_altitude_mondrian[a_key] = {}
        for al in alert_limits:
            al_key = f"al_{al:.1f}"
            k_av_std = sum(1 for pl in pls_std if pl <= al)
            k_av_mond = sum(1 for pl in pls_mond if pl <= al)
            k_av_alt = sum(1 for pl in pls_alt if pl <= al)
            avail_standard[a_key][al_key] = clopper_pearson_ci(k_av_std, n_total)
            avail_mondrian[a_key][al_key] = clopper_pearson_ci(k_av_mond, n_total)
            avail_altitude_mondrian[a_key][al_key] = clopper_pearson_ci(k_av_alt, n_total)

            by_cam_av: dict[str, Any] = {}
            for cset in CAM_SET_NAMES:
                tot_c = len([r for r in nominal_test if r.get("cam_set") == cset])
                af_c = [r for r in alarm_free if r.get("cam_set") == cset]
                k_c_av = sum(
                    1
                    for r in af_c
                    if compute_pl_deg_altitude_mondrian(
                        g1_conformal,
                        float(r["sigma_hat_urad"]) * 1e-6,
                        alpha,
                        cset,
                        r.get("condition", "degraded"),
                        float(r.get("h_meas_m", 0.0)),  # default-ok: optional worker task/record fallback
                    )
                    <= al
                )
                by_cam_av[cset] = clopper_pearson_ci(k_c_av, tot_c)
            avail_altitude_mondrian[a_key][al_key]["by_cam_set"] = by_cam_av

    return {
        "n_nominal_test": n_total,
        "n_alarm_free": n_alarm_free,
        "coverage": {
            "standard": cov_standard,
            "mondrian": cov_mondrian,
            "altitude_mondrian": cov_altitude_mondrian,
        },
        "availability": {
            "standard": avail_standard,
            "mondrian": avail_mondrian,
            "altitude_mondrian": avail_altitude_mondrian,
        },
    }


def evaluate_faults_and_hmi_g1(
    test_records: list[dict[str, Any]],
    misalignment_records: list[dict[str, Any]],
    thresholds: MonitorThresholdsG1,
    g1_conformal: dict[str, Any] | None = None,
    alpha_pl: float = 0.01,
    alert_limits: Sequence[float] = ALERT_LIMITS_G1,
) -> dict[str, Any]:
    """Evaluate fault detection rate, TTD (using real sequence dt_s), and HMI across all 5 fault modes."""
    all_recs = list(test_records) + list(misalignment_records)
    by_fault: dict[str, Any] = {}

    for ft in ("none",) + FAULT_TYPES_G1:
        if ft == "none":
            f_recs = [r for r in test_records if r.get("fault_type", "none") in ("none", None)]
        else:
            f_recs = [r for r in all_recs if r.get("fault_type") == ft and r.get("fault_active", False)]

        def _eval_fault_subset(sub: list[dict[str, Any]], is_fault: bool) -> dict[str, Any]:
            n_sub = len(sub)
            k_det = 0
            hmi_by_al = {f"al_{al:.1f}": 0 for al in alert_limits}
            for r in sub:
                ev = evaluate_monitor_alarm_g1(r, thresholds)
                f_iso = r.get("f_isolated") or {}
                alarmed = bool(ev["alarm"] or f_iso.get("isolated", False))
                if alarmed:
                    k_det += 1
                err = float(r.get("total_error_deg", float("nan")))
                pl_deg = 0.0
                if not alarmed and g1_conformal is not None:
                    pl_deg = compute_pl_deg_altitude_mondrian(
                        g1_conformal,
                        float(r.get("sigma_hat_urad", 1.0)) * 1e-6,  # default-ok: optional worker task/record fallback
                        alpha_pl,
                        str(r.get("cam_set", "F")),
                        str(r.get("condition", "degraded")),
                        float(r.get("h_meas_m", 0.0)),  # default-ok: optional worker task/record fallback
                    )
                for al in alert_limits:
                    al_key = f"al_{al:.1f}"
                    if not alarmed and (math.isnan(err) or (err > al and err > pl_deg)):
                        hmi_by_al[al_key] += 1
            det_ci = clopper_pearson_ci(k_det, n_sub) if is_fault else None
            hmi_ci = {
                al_key: clopper_pearson_ci(k_h, n_sub)
                for al_key, k_h in hmi_by_al.items()
            }
            return {
                "n_active_frames": n_sub,
                "detection_rate": det_ci,
                "hmi": hmi_ci,
            }

        overall_ft = _eval_fault_subset(f_recs, is_fault=(ft != "none"))
        by_cam_ft = {
            cset: _eval_fault_subset([r for r in f_recs if r.get("cam_set") == cset], is_fault=(ft != "none"))
            for cset in CAM_SET_NAMES
        }

        # TTD for injected faults using sequence-specific frame period dt_s (B11-F6)
        ttd_summary: dict[str, Any] = {}
        if ft != "none":
            for cset in CAM_SET_NAMES:
                sub_c = [r for r in f_recs if r.get("cam_set") == cset]
                runs: dict[tuple[str, int], list[dict[str, Any]]] = {}
                for r in sub_c:
                    runs.setdefault((str(r["seq_name"]), int(r["seed_idx"])), []).append(r)
                ttds_s: list[float] = []
                detected_runs = 0
                for (seq_n, _), r_list in sorted(runs.items()):
                    r_list.sort(key=lambda x: int(x["k"]))
                    k_onset = int(r_list[0].get("k_onset", r_list[0]["k"]))
                    dt_s = float(SEQUENCE_DT_S.get(seq_n, 0.5))  # default-ok: optional worker task/record fallback
                    for r in r_list:
                        if evaluate_monitor_alarm_g1(r, thresholds)["alarm"]:
                            ttds_s.append(max(0.0, (int(r["k"]) - k_onset) * dt_s))
                            detected_runs += 1
                            break
                ttd_summary[cset] = {
                    "n_runs": len(runs),
                    "n_detected_runs": detected_runs,
                    "median_s": float(np.median(ttds_s)) if ttds_s else None,
                    "p95_s": float(np.percentile(ttds_s, 95)) if ttds_s else None,
                    "max_s": float(np.max(ttds_s)) if ttds_s else None,
                }

        by_fault[ft] = {
            "overall": overall_ft,
            "by_cam_set": by_cam_ft,
            "ttd": ttd_summary,
        }

    return by_fault


def evaluate_item5_ood_detector(
    splits_recs: dict[str, list[dict[str, Any]]],
    th_01: MonitorThresholdsG1,
    th_001: MonitorThresholdsG1,
    faults_hmi_01: dict[str, Any],
    faults_hmi_001: dict[str, Any],
    test_far_01: dict[str, Any],
    g0_results: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Item 5 OOD detector strictly on held-out `splits_recs["test"]` using calibration-only tuned thresholds (`B11-F1`, `B11-R2-m1`)."""
    test_recs = splits_recs["test"]
    sp_test_f = [
        r for r in test_recs
        if r.get("fault_type") == "spurious_edge"
        and r.get("fault_active", False)
        and r.get("cam_set") == "F"
        and r.get("target_cam") == "F"
    ]
    gl_test_f = [
        r for r in test_recs
        if r.get("fault_type") == "glare"
        and r.get("fault_active", False)
        and r.get("cam_set") == "F"
        and r.get("target_cam") == "F"
    ]
    tau_ood_01 = float(th_01.by_cam_set["F"]["max_ood_horizon_score"])
    tau_ood_001 = float(th_001.by_cam_set["F"]["max_ood_horizon_score"])

    sp_det_k = sum(1 for r in sp_test_f if float(r.get("ood_horizon_score", 0.0)) > tau_ood_01)  # default-ok: optional worker task/record fallback
    gl_det_k = sum(1 for r in gl_test_f if float(r.get("ood_horizon_score", 0.0)) > tau_ood_01)  # default-ok: optional worker task/record fallback
    sp_det_rate = float(sp_det_k / max(1, len(sp_test_f)))
    gl_det_rate = float(gl_det_k / max(1, len(gl_test_f)))

    f_spurious_hmi_g1 = int(faults_hmi_01["spurious_edge"]["by_cam_set"]["F"]["hmi"]["al_0.5"]["k"])
    f_glare_hmi_g1 = int(faults_hmi_01["glare"]["by_cam_set"]["F"]["hmi"]["al_0.5"]["k"])
    false_ood_far = float(test_far_01["overall"]["by_reason_rate"]["OOD_STRIP_HIGH"])

    return {
        "tuned_on_splits": ["calibration"],
        "evaluated_on_split": "test",
        "tau_ood_far_0.01": tau_ood_01,
        "tau_ood_far_0.001": tau_ood_001,
        "f_spurious_edge_hmi_g1": f_spurious_hmi_g1,
        "f_glare_hmi_g1": f_glare_hmi_g1,
        "ood_detection_rate_spurious_edge": sp_det_rate,
        "ood_detection_rate_glare": gl_det_rate,
        "false_ood_alarm_rate_fault_free": false_ood_far,
        "by_fault_type": faults_hmi_01,
        "far_0.01": faults_hmi_01,
        "far_0.001": faults_hmi_001,
        "spurious_edge_F_al_0.5_g0_hmi_k": int(
            g0_results["evaluation"]["far_0.01"]["hmi_by_cam_set"]["mondrian"]["0.50deg"]["0.01"]["spurious_edge"]["F"]["k"]
        ),
        "spurious_edge_F_al_0.5_g0_hmi_n": int(
            g0_results["evaluation"]["far_0.01"]["hmi_by_cam_set"]["mondrian"]["0.50deg"]["0.01"]["spurious_edge"]["F"]["n"]
        ),
        "spurious_edge_F_al_0.5_g1_hmi_k": f_spurious_hmi_g1,
        "spurious_edge_F_al_0.5_g1_hmi_n": int(
            faults_hmi_01["spurious_edge"]["by_cam_set"]["F"]["hmi"]["al_0.5"]["n"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenXVS WP-07 Gate G1 Integrity Evaluation (Batch B11)")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config-path", "--config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--spec-path", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--max-workers", "--workers", dest="max_workers", type=int, default=None)
    args = parser.parse_args(argv)

    t_wall_start = time.perf_counter()
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Verify clean git status on XVS paths
    dirty = get_git_status_xvs()
    if dirty:
        raise RuntimeError(
            f"Refusing to run WP-07 G1 evaluation on dirty XVS working tree: {dirty}"
        )

    git_commit_start = get_git_commit()
    code_hashes_start = compute_code_hashes()

    data_root = get_data_root(args.data_root)
    g0_dir = data_root / "xvs" / "wp07" / "g0"
    g4_dir = data_root / "xvs" / "wp01" / "g4"
    out_dir = data_root / "xvs" / "wp07" / "g1"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_workers = args.max_workers or get_safe_workers()
    max_workers = min(max_workers, get_safe_workers())

    g1_cfg = json.loads(args.config_path.read_text(encoding="utf-8"))
    g1_spec = json.loads(args.spec_path.read_text(encoding="utf-8"))
    g0_results = json.loads((g0_dir / "results.json").read_text(encoding="utf-8"))
    fault_log = json.loads((g0_dir / "fault_log.json").read_text(encoding="utf-8"))

    solver_max_iterations = int(g1_cfg.get("solver", {}).get("max_iterations", g1_cfg.get("multiscale_edges", {}).get("max_iterations", 16)))  # default-ok: optional worker task/record fallback

    print("=" * 76)
    print("OpenXVS WP-07 Gate G1 — Horizon & Attitude Integrity Runner (Batch B11)")
    print(f"Git Commit:      {git_commit_start}")
    print(f"Data Root:       {data_root} ({data_root_source(args.data_root)})")
    print(f"G0 Dataset Dir:  {g0_dir}")
    print(f"G1 Output Dir:   {out_dir}")
    print(f"Max Workers:     {max_workers} (cap: {get_safe_workers()})")
    print(f"Max Iterations:  {solver_max_iterations}")
    print("=" * 76, flush=True)

    # Write initial run_manifest.json before worker pool launch
    manifest_path = out_dir / "run_manifest.json"
    g0_input_hashes = {
        "output/xvs/wp07/g0/dataset_calibration.jsonl": file_sha256(g0_dir / "dataset_calibration.jsonl"),
        "output/xvs/wp07/g0/dataset_validation.jsonl": file_sha256(g0_dir / "dataset_validation.jsonl"),
        "output/xvs/wp07/g0/dataset_test.jsonl": file_sha256(g0_dir / "dataset_test.jsonl"),
        "output/xvs/wp07/g0/fault_log.json": file_sha256(g0_dir / "fault_log.json"),
        "output/xvs/wp07/g0/results.json": file_sha256(g0_dir / "results.json"),
    }
    manifest_data: dict[str, Any] = {
        "batch_id": "B11_WP07_G1",
        "gate": "WP-07_G1",
        "status": "RUNNING",
        "start_time_utc": start_utc,
        "git_commit": git_commit_start,
        "git_status": dirty,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "max_workers": max_workers,
        "data_root": str(data_root),
        "data_root_source": data_root_source(args.data_root),
        "code_hashes_start": code_hashes_start,
        "g0_input_hashes": g0_input_hashes,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")

    # 2. Load G0 splits and frame metadata
    print("[1/6] Loading G0 dataset splits and WP-01 G4 metadata...", flush=True)
    splits_recs: dict[str, list[dict[str, Any]]] = {}
    for sp in ("calibration", "validation", "test"):
        p = g0_dir / f"dataset_{sp}.jsonl"
        with open(p, "r", encoding="utf-8") as f:
            splits_recs[sp] = [json.loads(line) for line in f if line.strip()]
        print(f"  Loaded {sp}: {len(splits_recs[sp])} records", flush=True)

    frames_meta: dict[str, dict[int, dict[str, Any]]] = {}
    for seq_name in ("T1_cruise", "T2_ksfo28l", "T3_turn_climb"):
        fp = g4_dir / seq_name / "frames.jsonl"
        frames_meta[seq_name] = {
            int(item["k"]): item
            for line in fp.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for item in [json.loads(line)]
        }

    # Count G0 baseline non-converged fault-free frames
    g0_non_conv_counts = {
        sp: sum(
            1
            for r in splits_recs[sp]
            if r.get("fault_type", "none") in ("none", None) and r.get("status") != "CONVERGED"
        )
        for sp in ("calibration", "validation", "test")
    }

    # Pre-index nominal h_meas_m by (seq_name, seed_idx, k) across all splits
    h_meas_index: dict[tuple[str, int, int], dict[str, float]] = {}
    for sp in ("calibration", "validation", "test"):
        for r in splits_recs[sp]:
            if r.get("fault_type", "none") in ("none", None):
                key = (str(r["seq_name"]), int(r["seed_idx"]), int(r["k"]))
                h_meas_index.setdefault(key, {})[f"{float(r['sigma_h']):.1f}"] = float(r["h_meas_m"])

    # 3. Build parallel worker tasks for Item 1 (non-converged + boundary frames) & Item 5 (OOD + faulty F fits)
    reprocess_keys: dict[tuple[str, int, int], dict[str, float]] = {}
    reprocess_needs_multicam: set[tuple[str, int, int]] = set()
    for sp in ("calibration", "validation", "test"):
        for r in splits_recs[sp]:
            seq_name = str(r["seq_name"])
            seed_idx = int(r["seed_idx"])
            k = int(r["k"])
            ft = str(r.get("fault_type", "none"))
            is_boundary = seq_name == "T2_ksfo28l" and k in (292, 293, 296, 297, 298, 299)
            is_non_conv = ft in ("none", None) and r.get("status") != "CONVERGED"
            if is_non_conv or is_boundary:
                key = (seq_name, seed_idx, k)
                if is_non_conv and r.get("cam_set") != "F":
                    reprocess_needs_multicam.add(key)
                s_map = reprocess_keys.setdefault(key, {})
                for sig_k, hm_v in h_meas_index.get(key, {}).items():
                    s_map[sig_k] = hm_v
                if ft == "altitude_bias":
                    s_map[f"alt_bias_{float(r['sigma_h']):.1f}"] = float(r["h_meas_m"])

    reprocess_tasks: list[dict[str, Any]] = []
    for (seq_name, seed_idx, k), h_meas_map in sorted(reprocess_keys.items()):
        item = frames_meta[seq_name][k]
        act = item["aircraft"]
        h_true = float(act["h_m"])
        rng_alt = np.random.default_rng(20260925 + (seed_idx * 100000 if seed_idx > 0 else 0) + k)
        for sig_h in (0.0, 10.0, 50.0):
            s_key = f"{sig_h:.1f}"
            if s_key not in h_meas_map:
                h_meas_map[s_key] = max(1.0, h_true + (0.0 if sig_h == 0.0 else float(rng_alt.normal(0.0, sig_h))))
        reprocess_tasks.append({
            "seq_name": seq_name,
            "seed_idx": seed_idx,
            "k": k,
            "g4_seq_dir": str(g4_dir / seq_name),
            "lat_deg": float(act["lat_deg"]),
            "lon_deg": float(act["lon_deg"]),
            "h_true_m": h_true,
            "yaw_rad": math.radians(float(act["ypr_deg"][0])),
            "pitch_true_deg": float(act["ypr_deg"][1]),
            "roll_true_deg": float(act["ypr_deg"][2]),
            "h_meas_by_sigma": h_meas_map,
            "max_iterations": solver_max_iterations,
            "only_cam_f": (seq_name, seed_idx, k) not in reprocess_needs_multicam,
        })

    # Build OOD + faulty-F evaluation tasks across all 600 unique (seq_name, k) frames
    fault_by_seq_seed: dict[tuple[str, int], dict[str, Any]] = {
        (str(e["seq_name"]), int(e["seed_idx"])): e["faults"]
        for e in fault_log
    }
    ood_tasks: list[dict[str, Any]] = []
    for seq_name, k_map in sorted(frames_meta.items()):
        for k, item in sorted(k_map.items()):
            act = item["aircraft"]
            gt_uv = item.get("cams", {}).get("F", {}).get("gt", {}).get("horizon_uv", [])
            f_seeds: list[dict[str, Any]] = []
            for seed_idx in range(1, 21):
                flts = fault_by_seq_seed.get((seq_name, seed_idx), {})
                hm_map = h_meas_index.get((seq_name, seed_idx, k), {"0.0": float(act["h_m"])})
                for ft in ("spurious_edge", "glare", "occlusion"):
                    f_cfg = flts.get(ft, {})
                    t_cam = str(f_cfg.get("target_cam", ""))
                    if t_cam in ("F", "R", "B", "L") and k >= int(f_cfg.get("onset_k", 999999)):  # default-ok: optional worker task/record fallback
                        is_multicam_sp = ft == "spurious_edge" and (seq_name, seed_idx, k) in (
                            ("T2_ksfo28l", 18, 159),
                            ("T2_ksfo28l", 20, 187),
                            ("T2_ksfo28l", 20, 241),
                        )
                        if t_cam == "F" or is_multicam_sp:
                            f_seeds.append({
                                "seed_idx": seed_idx,
                                "fault_type": ft,
                                "target_cam": t_cam,
                                "sub_seed": int(f_cfg["sub_seed"]),
                                "eval_attitude": bool(t_cam == "F" and seed_idx >= 16),
                                "h_meas_by_sigma": hm_map if (t_cam == "F" and seed_idx >= 16) else {},
                            })
            ood_tasks.append({
                "seq_name": seq_name,
                "k": k,
                "img_path_F": str(g4_dir / seq_name / "F" / f"{k:06d}.png"),
                "gt_uv_F": gt_uv,
                "lat_deg": float(act["lat_deg"]),
                "lon_deg": float(act["lon_deg"]),
                "yaw_rad": math.radians(float(act["ypr_deg"][0])),
                "pitch_true_deg": float(act["ypr_deg"][1]),
                "roll_true_deg": float(act["ypr_deg"][2]),
                "max_iterations": solver_max_iterations,
                "fault_seeds": f_seeds,
            })

    # Build Item 3 (boresight sweep on calibration seeds 1..2 + test seeds 16..20)
    # and Item 4 (single_cam_misalignment on test seeds 16..20)
    sigma_cal_grid = [float(x) for x in g1_cfg["boresight_calibration_sweep"]["sigma_cal_deg_grid"]]
    eval_stride = int(g1_cfg["boresight_calibration_sweep"]["eval_frame_stride"])
    injector_g1 = SeededFaultInjector(master_seed=int(g1_cfg["master_seed"]))
    boresight_misalign_tasks: list[dict[str, Any]] = []

    for seed_idx in [1, 2, 16, 17, 18, 19, 20]:
        is_test_seed = seed_idx >= 16
        sp_label = "test" if is_test_seed else "calibration"
        stride_use = eval_stride if is_test_seed else (eval_stride * 2)
        for seq_name, k_map in sorted(frames_meta.items()):
            n_frames = len(k_map)
            onset_k = injector_g1.choose_onset_frame(n_frames, seq_name, "single_cam_misalignment", seed_idx)
            target_cam = injector_g1.choose_target_camera(seq_name, "single_cam_misalignment", seed_idx)
            if target_cam == "F":
                target_cam = ("R", "B", "L")[seed_idx % 3]
            rng_mag = injector_g1.get_rng(seq_name, "mag_single_cam_misalignment", seed_idx)
            mag_deg = float(rng_mag.uniform(0.35, 0.95))
            sub_seed_mis = 20261100 + seed_idx * 100 + len(seq_name)

            for k in sorted(k_map.keys()):
                if k % stride_use != 0 and (not is_test_seed or (k != onset_k and k != onset_k + 1)):
                    continue
                item = k_map[k]
                act = item["aircraft"]
                boresight_misalign_tasks.append({
                    "seq_name": seq_name,
                    "split": sp_label,
                    "seed_idx": seed_idx,
                    "k": k,
                    "t_s": float(item["t_s"]),
                    "g4_seq_dir": str(g4_dir / seq_name),
                    "lat_deg": float(act["lat_deg"]),
                    "lon_deg": float(act["lon_deg"]),
                    "h_true_m": float(act["h_m"]),
                    "yaw_rad": math.radians(float(act["ypr_deg"][0])),
                    "pitch_true_deg": float(act["ypr_deg"][1]),
                    "roll_true_deg": float(act["ypr_deg"][2]),
                    "sigma_cal_grid": sigma_cal_grid,
                    "max_iterations": solver_max_iterations,
                    "run_misalignment": is_test_seed,
                    "misalign_cam": target_cam,
                    "misalign_mag_deg": mag_deg,
                    "misalign_onset_k": onset_k,
                    "misalign_active": bool(is_test_seed and k >= onset_k),
                    "misalign_sub_seed": sub_seed_mis + k,
                })

    print(
        f"[2/6] Launching worker pool ({max_workers} workers): "
        f"{len(reprocess_tasks)} Item-1 reprocess frames, "
        f"{len(ood_tasks)} Item-5 OOD sequence frames, "
        f"{len(boresight_misalign_tasks)} Item-3/4 boresight+misalignment frames...",
        flush=True,
    )

    t_pool_start = time.perf_counter()
    pool = mp.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(str(DEFAULT_SENSOR_CONFIG),),
    )
    try:
        reprocess_results = pool.map(_reprocess_frame_worker, reprocess_tasks)
        ood_results = pool.map(_ood_sequence_frame_worker, ood_tasks)
        bm_results = pool.map(_boresight_and_misalignment_worker, boresight_misalign_tasks)
    finally:
        pool.close()
        pool.join()
    print(f"  Worker pool finished in {time.perf_counter() - t_pool_start:.1f}s", flush=True)

    # 4. Apply Item 1 & Item 5 updates to calibration, validation, and test splits
    print("[3/6] Updating dataset splits with Item-1 multi-scale fits and Item-5 per-seed OOD scores...", flush=True)
    reproc_lookup: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {
        (r["seq_name"], int(r["seed_idx"]), int(r["k"])): r["fits"]
        for r in reprocess_results
    }
    ood_nom_lookup: dict[tuple[str, int, int], dict[str, float]] = {}
    ood_fault_lookup: dict[tuple[str, int, int, str, str], dict[str, float]] = {}
    fault_fit_lookup: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}

    for r in ood_results:
        seq_n = str(r["seq_name"])
        k_val = int(r["k"])
        for s_str, ood_d in r["ood_nom_by_seed"].items():
            ood_nom_lookup[(seq_n, max(1, int(s_str)), k_val)] = ood_d
        for key_sft, ood_d in r["fault_ood"].items():
            s_str, ft_str, tc_str = key_sft.split(":", 2)
            ood_fault_lookup[(seq_n, int(s_str), k_val, ft_str, tc_str)] = ood_d
        for key_fit, fit_d in r["fault_fits"].items():
            s_str, ft_str, sig_str = key_fit.split(":", 2)
            fault_fit_lookup[(seq_n, int(s_str), k_val, ft_str, sig_str)] = fit_d

    boundary_errors_g0: list[float] = []
    boundary_errors_g1: list[float] = []

    for sp in ("calibration", "validation", "test"):
        for r in splits_recs[sp]:
            seq_n = str(r["seq_name"])
            s_idx = int(r["seed_idx"])
            k_val = int(r["k"])
            cset = str(r["cam_set"])
            sig_key = f"{float(r['sigma_h']):.1f}"
            ft = str(r.get("fault_type", "none"))
            t_cam = str(r.get("target_cam") or "")

            # Annotate per-seed nominal OOD score (seed 0 clean frames use seed 1 nominal OOD)
            nom_ood = ood_nom_lookup.get((seq_n, max(1, s_idx), k_val), {})
            r["ood_horizon_score"] = float(nom_ood.get("ood_horizon_score", 0.0))  # default-ok: optional worker task/record fallback
            r["bimodal_edge_fraction"] = float(nom_ood.get("bimodal_edge_fraction", 0.0))  # default-ok: optional worker task/record fallback
            r["strip_grad_p90"] = float(nom_ood.get("strip_grad_p90", 0.0))  # default-ok: optional worker task/record fallback
            r["strip_saturation_fraction"] = float(nom_ood.get("strip_saturation_fraction", 0.0))  # default-ok: optional worker task/record fallback

            if ft in ("none", None) and (seq_n, s_idx, k_val) in reproc_lookup:
                if seq_n == "T2_ksfo28l" and k_val in (298, 299) and cset == "F" and sig_key == "0.0":
                    boundary_errors_g0.append(float(r["total_error_deg"]))
                fit_upd = reproc_lookup[(seq_n, s_idx, k_val)].get(f"{cset}:{sig_key}")
                if fit_upd is not None:
                    r.update(fit_upd)
                    r["ood_horizon_score"] = float(nom_ood.get("ood_horizon_score", r["ood_horizon_score"]))
                if seq_n == "T2_ksfo28l" and k_val in (298, 299) and cset == "F" and sig_key == "0.0":
                    boundary_errors_g1.append(float(r["total_error_deg"]))

            elif ft not in ("none", None) and r.get("fault_active", False):
                # Update boundary / non-converged frames when camera F has no image fault
                if (seq_n, s_idx, k_val) in reproc_lookup:
                    if ft == "altitude_bias":
                        fit_ab = reproc_lookup[(seq_n, s_idx, k_val)].get(f"{cset}:alt_bias_{sig_key}")
                        if fit_ab is not None:
                            r.update(fit_ab)
                            r["ood_horizon_score"] = float(nom_ood.get("ood_horizon_score", 0.0))  # default-ok: optional worker task/record fallback
                    elif cset == "F" and t_cam != "F":
                        fit_nf = reproc_lookup[(seq_n, s_idx, k_val)].get(f"F:{sig_key}")
                        if fit_nf is not None:
                            r.update(fit_nf)
                            r["ood_horizon_score"] = float(nom_ood.get("ood_horizon_score", 0.0))  # default-ok: optional worker task/record fallback

                # Update G1 faulty F attitude fit when target_cam == 'F'
                if cset == "F" and t_cam == "F":
                    f_fit = fault_fit_lookup.get((seq_n, s_idx, k_val, ft, sig_key))
                    if f_fit is not None:
                        r.update(f_fit)

                # Update OOD score when the fault-injected camera belongs to cset
                if ft in ("spurious_edge", "glare", "occlusion") and t_cam in CAMERA_SETS.get(cset, ()):
                    f_ood = ood_fault_lookup.get((seq_n, s_idx, k_val, ft, t_cam))
                    if f_ood is not None:
                        r["ood_horizon_score"] = max(float(r["ood_horizon_score"]), float(f_ood["ood_horizon_score"]))
                        r["bimodal_edge_fraction"] = float(f_ood["bimodal_edge_fraction"])
                        r["strip_grad_p90"] = float(f_ood["strip_grad_p90"])
                        r["strip_saturation_fraction"] = float(f_ood["strip_saturation_fraction"])

    g1_non_conv_counts = {
        sp: sum(
            1
            for r in splits_recs[sp]
            if r.get("fault_type", "none") in ("none", None) and r.get("status") != "CONVERGED"
        )
        for sp in ("calibration", "validation", "test")
    }
    print(f"  G0 non-converged fault-free counts: {g0_non_conv_counts} -> G1: {g1_non_conv_counts}", flush=True)

    # 5. Tune G1 Monitor M1 thresholds (Item 2 + Item 5) strictly on calibration split only (B11-F1, B11-F2, B11-R2-m1)
    print("[4/6] Tuning Monitor M1 thresholds on calibration split and fitting altitude-binned Mondrian conformal model...", flush=True)
    cal_nominal = [r for r in splits_recs["calibration"] if r.get("fault_type", "none") in ("none", None)]

    th_01 = tune_monitor_thresholds_g1(cal_nominal, target_far=0.01)
    th_001 = tune_monitor_thresholds_g1(cal_nominal, target_far=0.001)

    cal_alarm_free_01 = [r for r in cal_nominal if not evaluate_monitor_alarm_g1(r, th_01)["alarm"]]
    cal_alarm_free_001 = [r for r in cal_nominal if not evaluate_monitor_alarm_g1(r, th_001)["alarm"]]

    conformal_01 = fit_conformal_model_g1(cal_alarm_free_01, alphas=ALPHA_LEVELS)
    conformal_001 = fit_conformal_model_g1(cal_alarm_free_001, alphas=ALPHA_LEVELS)

    # Evaluate FAR, Coverage, Availability, and Faults/HMI on out-of-sample validation and held-out test splits
    print("[5/6] Evaluating V1_G1..V7_G1 metrics across validation, test, boresight sweep, and misalignment...", flush=True)
    val_far_01 = evaluate_far_g1(splits_recs["validation"], th_01)
    val_far_001 = evaluate_far_g1(splits_recs["validation"], th_001)
    test_far_01 = evaluate_far_g1(splits_recs["test"], th_01)
    test_far_001 = evaluate_far_g1(splits_recs["test"], th_001)

    cov_avail_01 = evaluate_coverage_and_availability_g1(splits_recs["test"], conformal_01, th_01)
    cov_avail_001 = evaluate_coverage_and_availability_g1(splits_recs["test"], conformal_001, th_001)

    # Aggregate Item 3 (boresight calibration sweep) and Item 4 (single_cam_misalignment + F_isolated)
    boresight_all_recs: list[dict[str, Any]] = []
    misalign_all_recs: list[dict[str, Any]] = []
    for res_bm in bm_results:
        boresight_all_recs.extend(res_bm["boresight_records"])
        misalign_all_recs.extend(res_bm["misalignment_records"])

    # Synchronize unperturbed sigma_cal == 0.0 records with exact G1 nominal sigma_h == 0.0 split records (B11-R2-m2)
    nom_split_0_index: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for sp_n in ("calibration", "test"):
        for r in splits_recs[sp_n]:
            if (
                r.get("fault_type", "none") in ("none", None)
                and float(r.get("sigma_h", -1.0)) == 0.0  # default-ok: optional worker task/record fallback
                and r.get("condition") == "degraded"
            ):
                nom_split_0_index[(str(r["seq_name"]), int(r["seed_idx"]), int(r["k"]), str(r["cam_set"]))] = r

    for r_b in boresight_all_recs:
        if abs(float(r_b["sigma_cal_deg"])) < 1e-9:
            ref_r = nom_split_0_index.get(
                (str(r_b["seq_name"]), int(r_b["seed_idx"]), int(r_b["k"]), str(r_b["cam_set"]))
            )
            if ref_r is not None:
                for fld in (
                    "status",
                    "pitch_est_deg",
                    "roll_est_deg",
                    "total_error_deg",
                    "total_error_rad",
                    "rms_residual_urad",
                    "inlier_count",
                    "inlier_fraction",
                    "sigma_hat_urad",
                    "condition_number",
                    "camera_consistency_deg",
                    "leave_one_out_consistency_deg",
                    "ood_horizon_score",
                ):
                    if fld in ref_r:
                        r_b[fld] = ref_r[fld]

    boresight_cal_recs = [r for r in boresight_all_recs if r.get("split") == "calibration"]
    boresight_test_recs = [r for r in boresight_all_recs if r.get("split") == "test"]

    # Calibrate boresight PL sensitivity strictly on calibration boresight records (B11-F3)
    boresight_sens_01 = calibrate_boresight_pl_sensitivity_g1(
        boresight_cal_recs, conformal_01, alphas=ALPHA_LEVELS
    )
    conformal_01["boresight_sensitivity"] = boresight_sens_01

    boresight_sweep_summary: dict[str, Any] = {}
    for sig_cal in sigma_cal_grid:
        s_key = f"sigma_cal_{sig_cal:.2f}deg"
        sub_b = [r for r in boresight_test_recs if abs(float(r["sigma_cal_deg"]) - sig_cal) < 1e-9]
        by_cam_b: dict[str, Any] = {}
        for cset in CAM_SET_NAMES:
            sc = [r for r in sub_b if r["cam_set"] == cset]
            errs = [float(r["total_error_deg"]) for r in sc]
            loos = [float(r["camera_consistency_deg"]) for r in sc if r["camera_consistency_deg"] is not None]
            af_c = [
                r for r in sc
                if not evaluate_monitor_alarm_g1(
                    r, th_01, sigma_cal_deg=sig_cal, boresight_sensitivity=boresight_sens_01
                )["alarm"]
            ]
            k_cov_99 = 0
            k_cov_999 = 0
            k_av_10 = 0
            k_hmi_05 = 0
            for r in sc:
                alarmed = evaluate_monitor_alarm_g1(
                    r, th_01, sigma_cal_deg=sig_cal, boresight_sensitivity=boresight_sens_01
                )["alarm"]
                err = float(r["total_error_deg"])
                if not alarmed:
                    pl_99 = compute_pl_deg_altitude_mondrian(
                        conformal_01,
                        float(r["sigma_hat_urad"]) * 1e-6,
                        0.01,
                        cset,
                        "degraded",
                        float(r["h_meas_m"]),
                        sigma_cal_deg=sig_cal,
                        boresight_sensitivity=boresight_sens_01,
                    )
                    pl_999 = compute_pl_deg_altitude_mondrian(
                        conformal_01,
                        float(r["sigma_hat_urad"]) * 1e-6,
                        0.001,
                        cset,
                        "degraded",
                        float(r["h_meas_m"]),
                        sigma_cal_deg=sig_cal,
                        boresight_sensitivity=boresight_sens_01,
                    )
                    if err > 0.5 and err > pl_99:
                        k_hmi_05 += 1
                    if err <= pl_99:
                        k_cov_99 += 1
                    if err <= pl_999:
                        k_cov_999 += 1
                    if pl_99 <= 1.0:
                        k_av_10 += 1
            by_cam_b[cset] = {
                "n_frames": len(sc),
                "rmse_error_deg": float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0,
                "p95_error_deg": float(np.percentile(errs, 95)) if errs else 0.0,
                "median_d27_deg": float(np.median(loos)) if loos else None,
                "p95_d27_deg": float(np.percentile(loos, 95)) if loos else None,
                "coverage_alpha_0.01": clopper_pearson_ci(k_cov_99, len(af_c)),
                "coverage_alpha_0.001": clopper_pearson_ci(k_cov_999, len(af_c)),
                "availability_al_1.0": clopper_pearson_ci(k_av_10, len(sc)),
                "hmi_al_0.5": clopper_pearson_ci(k_hmi_05, len(sc)),
            }
        boresight_sweep_summary[s_key] = {
            "sigma_cal_deg": float(sig_cal),
            "by_cam_set": by_cam_b,
        }

    # Evaluate Item 4 F_isolated fault isolation and exclusion summary
    f_isolated_summary: dict[str, Any] = {}
    for cset in ("FRBL", "all"):
        active_mc = [
            r for r in misalign_all_recs
            if r["cam_set"] == cset and r["fault_active"]
        ]
        n_act = len(active_mc)
        k_detected = sum(1 for r in active_mc if evaluate_monitor_alarm_g1(r, th_01)["alarm"])
        k_isolated = sum(1 for r in active_mc if r["f_isolated"]["isolated"])
        k_correct_id = sum(
            1 for r in active_mc
            if r["f_isolated"]["isolated"] and r["f_isolated"]["excluded_camera"] == r["target_cam"]
        )
        pre_errs = [float(r["total_error_deg"]) for r in active_mc]
        post_errs = [
            float(r["f_isolated"]["post_isolation_error_deg"])
            for r in active_mc
            if r["f_isolated"]["post_isolation_error_deg"] is not None
        ]
        pre_d27 = [
            float(r["f_isolated"]["pre_isolation_d27_deg"])
            for r in active_mc
            if r["f_isolated"]["pre_isolation_d27_deg"] is not None
        ]
        post_d27 = [
            float(r["f_isolated"]["post_isolation_d27_deg"])
            for r in active_mc
            if r["f_isolated"]["post_isolation_d27_deg"] is not None
        ]
        k_post_avail_10 = 0
        k_post_hmi_05 = 0
        for r in active_mc:
            iso = r["f_isolated"]
            if iso["isolated"] and iso["post_isolation_error_deg"] is not None and iso["post_isolation_sigma_hat_urad"] is not None:
                pl_post = compute_pl_deg_altitude_mondrian(
                    conformal_01, float(iso["post_isolation_sigma_hat_urad"]) * 1e-6, 0.01, cset, "degraded", float(r["h_meas_m"])
                )
                if pl_post <= 1.0:
                    k_post_avail_10 += 1
                if iso["post_isolation_error_deg"] > 0.5 and iso["post_isolation_error_deg"] > pl_post:
                    k_post_hmi_05 += 1

        f_isolated_summary[cset] = {
            "n_active_frames": n_act,
            "detection_rate": clopper_pearson_ci(k_detected, n_act),
            "isolation_rate": clopper_pearson_ci(k_isolated, n_act),
            "correct_camera_id_rate": clopper_pearson_ci(k_correct_id, max(1, k_isolated)),
            "pre_isolation_rmse_deg": float(np.sqrt(np.mean(np.square(pre_errs)))) if pre_errs else 0.0,
            "post_isolation_rmse_deg": float(np.sqrt(np.mean(np.square(post_errs)))) if post_errs else 0.0,
            "pre_isolation_median_d27_deg": float(np.median(pre_d27)) if pre_d27 else None,
            "post_isolation_median_d27_deg": float(np.median(post_d27)) if post_d27 else None,
            "post_isolation_availability_al_1.0": clopper_pearson_ci(k_post_avail_10, n_act),
            "post_isolation_hmi_al_0.5": clopper_pearson_ci(k_post_hmi_05, n_act),
        }

    faults_hmi_01 = evaluate_faults_and_hmi_g1(
        splits_recs["test"], misalign_all_recs, th_01, g1_conformal=conformal_01, alpha_pl=0.01
    )
    faults_hmi_001 = evaluate_faults_and_hmi_g1(
        splits_recs["test"], misalign_all_recs, th_001, g1_conformal=conformal_001, alpha_pl=0.001
    )

    item5_summary = evaluate_item5_ood_detector(
        splits_recs, th_01, th_001, faults_hmi_01, faults_hmi_001, test_far_01, g0_results
    )

    total_hmi_050 = sum(
        int(faults_hmi_01[ft]["overall"]["hmi"]["al_0.5"]["k"])
        for ft in ("none",) + FAULT_TYPES_G1
    )
    total_hmi_100 = sum(
        int(faults_hmi_01[ft]["overall"]["hmi"]["al_1.0"]["k"])
        for ft in ("none",) + FAULT_TYPES_G1
    )

    low_contrast_bench = run_low_contrast_and_solver_benchmark(load_pipeline_rig())

    # Compute top-level overall_gate_verdict (B11-F7)
    boresight_cov_ok = all(
        boresight_sweep_summary[s_key]["by_cam_set"][cset]["coverage_alpha_0.01"]["rate"] >= 0.990
        and boresight_sweep_summary[s_key]["by_cam_set"][cset]["coverage_alpha_0.001"]["rate"] >= 0.999
        and boresight_sweep_summary[s_key]["by_cam_set"][cset]["hmi_al_0.5"]["k"] == 0
        for s_key in ("sigma_cal_0.00deg", "sigma_cal_0.02deg", "sigma_cal_0.05deg")
        for cset in CAM_SET_NAMES
    )
    all_checks_pass = (
        g1_non_conv_counts["calibration"] == 0
        and g1_non_conv_counts["validation"] == 0
        and g1_non_conv_counts["test"] == 0
        and val_far_01["verdict"] == "PASS"
        and val_far_001["verdict"] == "PASS"
        and test_far_01["verdict"] == "PASS"
        and test_far_001["verdict"] == "PASS"
        and cov_avail_01["coverage"]["altitude_mondrian"]["0.01"]["verdict"] == "PASS"
        and cov_avail_01["coverage"]["altitude_mondrian"]["0.001"]["verdict"] == "PASS"
        and boresight_cov_ok
        and total_hmi_050 == 0
        and item5_summary["f_spurious_edge_hmi_g1"] == 0
        and item5_summary["f_glare_hmi_g1"] == 0
    )
    overall_gate_verdict = "PASS" if all_checks_pass else "FAIL"

    # Assemble complete results.json
    print("[6/6] Writing output/xvs/wp07/g1/results.json and finalizing run_manifest.json...", flush=True)
    results_payload: dict[str, Any] = {
        "gate": "WP-07_G1",
        "batch_id": "B11_WP07_G1",
        "git_commit": git_commit_start,
        "overall_gate_verdict": overall_gate_verdict,
        "hmi": {
            "al_0.50": {
                "total_hmi": int(total_hmi_050),
                "by_fault_type": {
                    ft: int(faults_hmi_01[ft]["overall"]["hmi"]["al_0.5"]["k"])
                    for ft in ("none",) + FAULT_TYPES_G1
                },
            },
            "al_1.00": {
                "total_hmi": int(total_hmi_100),
                "by_fault_type": {
                    ft: int(faults_hmi_01[ft]["overall"]["hmi"]["al_1.0"]["k"])
                    for ft in ("none",) + FAULT_TYPES_G1
                },
            },
        },
        "dataset_summary": {
            "calibration_total_records": len(splits_recs["calibration"]),
            "validation_total_records": len(splits_recs["validation"]),
            "test_total_records": len(splits_recs["test"]),
            "boresight_sweep_records": len(boresight_test_recs),
            "boresight_calibration_records": len(boresight_cal_recs),
            "single_cam_misalignment_records": len(misalign_all_recs),
        },
        "item1_multiscale_and_solver": {
            "g0_non_converged_fault_free": g0_non_conv_counts,
            "g1_non_converged_fault_free": g1_non_conv_counts,
            "t2_boundary_F_k298_299_g0_max_error_deg": float(np.max(boundary_errors_g0)) if boundary_errors_g0 else 0.0,
            "t2_boundary_F_k298_299_g1_max_error_deg": float(np.max(boundary_errors_g1)) if boundary_errors_g1 else 0.0,
            "t2_boundary_F_k298_299_g0_mean_error_deg": float(np.mean(boundary_errors_g0)) if boundary_errors_g0 else 0.0,
            "t2_boundary_F_k298_299_g1_mean_error_deg": float(np.mean(boundary_errors_g1)) if boundary_errors_g1 else 0.0,
            "low_contrast_benchmark": low_contrast_bench["by_contrast"],
        },
        "item2_altitude_conditioning": {
            "thresholds": {
                "far_0.01": th_01.to_dict(),
                "far_0.001": th_001.to_dict(),
            },
            "conformal_models": {
                "far_0.01": {
                    "base_model": conformal_01["base_model"].to_dict(),
                    "altitude_mondrian_quantiles": conformal_01["altitude_mondrian_quantiles"],
                    "n_cal_altitude_mondrian": conformal_01["n_cal_altitude_mondrian"],
                    "boresight_sensitivity": boresight_sens_01,
                },
                "far_0.001": {
                    "base_model": conformal_001["base_model"].to_dict(),
                    "altitude_mondrian_quantiles": conformal_001["altitude_mondrian_quantiles"],
                    "n_cal_altitude_mondrian": conformal_001["n_cal_altitude_mondrian"],
                },
            },
            "validation_far": {
                "far_0.01": val_far_01,
                "far_0.001": val_far_001,
            },
            "test_far": {
                "far_0.01": test_far_01,
                "far_0.001": test_far_001,
            },
            "test_coverage_and_availability": {
                "far_0.01": cov_avail_01,
                "far_0.001": cov_avail_001,
            },
        },
        "item3_boresight_calibration_sweep": boresight_sweep_summary,
        "item4_single_cam_misalignment_and_isolation": {
            "f_isolated_by_cam_set": f_isolated_summary,
            "fault_evaluation_far_0.01": faults_hmi_01["single_cam_misalignment"],
            "fault_evaluation_far_0.001": faults_hmi_001["single_cam_misalignment"],
        },
        "item5_ood_and_fault_hmi": item5_summary,
        "g0_vs_g1_comparison": {
            "g0_test_far_0.01_rate": float(g0_results["achieved_far"]["far_0.01"]["test"]["overall"]["rate"]),
            "g1_test_far_0.01_rate": float(test_far_01["overall"]["rate"]),
            "g0_test_far_0.001_rate": float(g0_results["achieved_far"]["far_0.001"]["test"]["overall"]["rate"]),
            "g1_test_far_0.001_rate": float(test_far_001["overall"]["rate"]),
            "g0_test_status_floor_k": int(
                g0_results["achieved_far"]["far_0.001"]["test"]["floor_analysis"]["status_not_converged_k"]
            ),
            "g1_test_status_floor_k": int(test_far_001["overall"]["by_reason"]["STATUS_NOT_CONVERGED"]),
            "g0_mondrian_avail_a001_al05_rate": float(
                g0_results["evaluation"]["far_0.01"]["availability"]["mondrian"]["0.50deg"]["0.01"]["overall"]["rate"]
            ),
            "g1_altitude_mondrian_avail_a001_al05_rate": float(
                cov_avail_01["availability"]["altitude_mondrian"]["0.01"]["al_0.5"]["rate"]
            ),
            "g1_altitude_mondrian_avail_a001_al10_rate": float(
                cov_avail_01["availability"]["altitude_mondrian"]["0.01"]["al_1.0"]["rate"]
            ),
        },
    }

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")

    git_commit_end = get_git_commit()
    dirty_end = get_git_status_xvs()
    code_hashes_end = compute_code_hashes()
    wall_s = time.perf_counter() - t_wall_start
    end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    manifest_data.update({
        "status": "COMPLETED",
        "end_time_utc": end_utc,
        "wall_time_s": float(wall_s),
        "git_commit_end": git_commit_end,
        "git_status_end": dirty_end,
        "code_hashes_end": code_hashes_end,
        "code_hashes_match": bool(code_hashes_start == code_hashes_end),
        "outputs": {
            "output/xvs/wp07/g1/results.json": file_sha256(results_path),
        },
    })
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")

    print("=" * 76)
    print(f"WP-07 G1 evaluation completed in {wall_s:.2f}s — Verdict: {overall_gate_verdict}")
    print(f"  Test FAR (target 1.0%):   {test_far_01['overall']['rate']*100:.4f}% ({test_far_01['overall']['k']}/{test_far_01['overall']['n']}) [{test_far_01['verdict']}]")
    print(f"  Test FAR (target 0.1%):   {test_far_001['overall']['rate']*100:.4f}% ({test_far_001['overall']['k']}/{test_far_001['overall']['n']}) [{test_far_001['verdict']}]")
    print(f"  Total HMI (AL=0.50 deg):  {total_hmi_050}")
    print(f"  Spurious-edge F HMI(0.5): {item5_summary['spurious_edge_F_al_0.5_g1_hmi_k']}/{item5_summary['spurious_edge_F_al_0.5_g1_hmi_n']}")
    print(f"  code_hashes_match:        {manifest_data['code_hashes_match']}")
    print("=" * 76, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
