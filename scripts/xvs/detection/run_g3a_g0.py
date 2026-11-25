"""Run G3a Traffic Detection Evaluation (WP-04 G0 & Decision D23).

Rehearsed on natural dataset v0 pilot (6 encounters, 3,510 frames).
All pilot outputs are strictly labeled "PILOT, not a G0 result".

Implements:
1. Primary tight centroid hit rule: r = max(1.5 px, 0.5 * s_px) around projected center.
2. Secondary AOT eIoU > 0.2 hit rule.
3. Natural data null control: phantom targets transferred to negative ROIs of same encounter
   with matching background, yielding p0 per s_px bin and background, and Abbott-corrected Pd.
4. False positives on negative ROIs and positive ROIs outside analytic box + 16 px,
   split by sky/earth via WP-01 horizon function, with Poisson 95% upper bounds,
   lambda per full ring frame (5,013,504 px), and false detections per flight hour.
5. Clean and degraded (ring_mono_v0_nodist, k in {3, 5, 7}) evaluations across tau sweep.
6. Strata: class, background (sky/earth), range bins, s_px bins, A_C bins,
   with Clopper-Pearson 95% CIs and encounter-cluster bootstrap.
7. Localisation errors: sigma_px against visible mask centroid and offset to projected center.
8. Target signal: A_C, mean contrast C, and degraded SNR (WP-04 §4).
9. Operating point (k*, tau*) selection and R50/R90 detection range fits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root
from xvs.detection.d0 import detect_d0, load_d0_config, compute_cmo, compute_cmo_multi, compute_local_stats, compute_clean_floor_grid, compute_z_score, extract_components
from xvs.detection.contrast import estimate_target_signal
from xvs.detection.fit import fit_detection_range_mle, fit_logistic_mle, evaluate_closure_fit, evaluate_range_fit
from xvs.detection.g3a_eval import (
    TAU_LIST,
    K_VALUES,
    SENSOR_MODES,
    RANGE_BINS_KM,
    S_PX_BINS,
    AC_BINS,
    T1_RING_SKY_MPX,
    T1_RING_EARTH_MPX,
    T1_RING_TOTAL_MPX,
    FLIGHT_HOURLY_FRAMES,
    primary_matching_radius,
    get_range_bin_str,
    get_spx_bin_str,
    get_ac_bin_str,
    compute_roi_sky_earth_area,
    pair_phantom_targets,
    evaluate_hits_and_fps_for_roi,
    compute_stratified_pd_stats,
    compute_fp_rate_sweep,
)
from xvs.metrics.aot import poisson_rate_upper_bound
from xvs.sensor.srgb import srgb_eotf, linear_rgb_to_monochrome, REC709_R_WEIGHT, REC709_G_WEIGHT, REC709_B_WEIGHT
from xvs.sensor.chain import SensorConfig
from xvs.sensor.psf import apply_psf
from xvs.sensor.noise import apply_sensor_noise, generate_fpn_maps
from xvs.traffic.ring import TrafficRing
from xvs.sim.horizon import is_earth_at_pixel, is_earth_at_pixels

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PILOT_ROOT = get_data_root() / "xvs" / "wp02" / "g3a_pilot"
DEFAULT_OUT_DIR = get_data_root() / "xvs" / "wp04" / "g0_pilot"


# Worker process global state
_WORKER_RING_MODEL: TrafficRing | None = None
_WORKER_SENSOR_CFG: SensorConfig | None = None
_WORKER_PSF_KERNEL: np.ndarray | None = None
_WORKER_PRNU: np.ndarray | None = None
_WORKER_DSNU: np.ndarray | None = None
_WORKER_D0_CFG: dict[str, Any] = {}
_WORKER_G3A_ROOT: Path | None = None
_WORKER_EOTF_R: np.ndarray | None = None
_WORKER_EOTF_G: np.ndarray | None = None
_WORKER_EOTF_B: np.ndarray | None = None


def init_worker(g3a_root_str: str) -> None:
    """Initialize process-global cached models and lookup tables."""
    global _WORKER_RING_MODEL, _WORKER_SENSOR_CFG, _WORKER_PSF_KERNEL, _WORKER_PRNU, _WORKER_DSNU, _WORKER_D0_CFG, _WORKER_G3A_ROOT
    global _WORKER_EOTF_R, _WORKER_EOTF_G, _WORKER_EOTF_B
    _WORKER_RING_MODEL = TrafficRing()
    sensor_cfg_path = REPO_ROOT / "configs" / "xvs" / "sensors" / "ring_mono_v0_nodist.json"
    _WORKER_SENSOR_CFG = SensorConfig.from_file(sensor_cfg_path)
    kernel, _, _ = _WORKER_SENSOR_CFG.get_psf_kernel()
    _WORKER_PSF_KERNEL = kernel
    prnu, dsnu = generate_fpn_maps("ring_cam", 1024, 1024, _WORKER_SENSOR_CFG.prnu_rms, _WORKER_SENSOR_CFG.dsnu_rms_e)
    _WORKER_PRNU = prnu
    _WORKER_DSNU = dsnu
    _WORKER_D0_CFG = load_d0_config()
    _WORKER_G3A_ROOT = Path(g3a_root_str)
    eotf_lut = srgb_eotf(np.arange(256, dtype=np.uint8))
    _WORKER_EOTF_R = eotf_lut * REC709_R_WEIGHT
    _WORKER_EOTF_G = eotf_lut * REC709_G_WEIGHT
    _WORKER_EOTF_B = eotf_lut * REC709_B_WEIGHT


def process_positive_roi_task(task_data: dict[str, Any]) -> dict[str, Any]:
    """Worker task: evaluate one positive ROI in both clean and degraded states across all k."""
    t0 = time.perf_counter()
    g3a_root = _WORKER_G3A_ROOT
    ring = _WORKER_RING_MODEL
    scfg = _WORKER_SENSOR_CFG
    kernel = _WORKER_PSF_KERNEL
    prnu = _WORKER_PRNU
    dsnu = _WORKER_DSNU
    d0_cfg = _WORKER_D0_CFG

    rgb_rel = task_data["rgb_path"]
    rgb_full = g3a_root / rgb_rel
    with Image.open(rgb_full) as im:
        rgb_arr = np.array(im.convert("RGB"), dtype=np.uint8)

    # 1. Clean linear monochrome via EOTF LUT
    if _WORKER_EOTF_R is not None:
        lin_mono_clean = (
            _WORKER_EOTF_R[rgb_arr[:, :, 0]]
            + _WORKER_EOTF_G[rgb_arr[:, :, 1]]
            + _WORKER_EOTF_B[rgb_arr[:, :, 2]]
        )
    else:
        lin_rgb = srgb_eotf(rgb_arr)
        lin_mono_clean = linear_rgb_to_monochrome(lin_rgb)

    # Precompute clean floor grid once for all k values
    clean_floor = compute_clean_floor_grid(
        lin_mono_clean,
        tile_size=d0_cfg.get("tile_size", 64),  # default-ok: optional fallback default
        tile_stride=d0_cfg.get("tile_stride", 32),  # default-ok: optional fallback default
    )

    # 2. Degraded monochrome via WP-03 chain
    blurred = apply_psf(lin_mono_clean, kernel)
    raw_dn = apply_sensor_noise(
        linear_radiance=blurred,
        prnu_map=prnu,
        dsnu_map=dsnu,
        E=scfg.full_well_e,
        full_well_e=scfg.full_well_e,
        sigma_r_e=scfg.sigma_r_e,
        dark_current_e_per_s=scfg.dark_current_e_per_s,
        exposure_time_s=scfg.exposure_time_s,
        k_dn_per_e=scfg.k_dn_per_e,
        bits=scfg.bits,
        black_level_dn=scfg.black_level_dn,
        rng_temporal=np.random.Generator(np.random.PCG64(task_data["k_frame"])),
    )
    lin_mono_deg = (raw_dn.astype(np.float64) - scfg.black_level_dn) / scfg.k_dn_per_e

    # 3. Target signal & contrast estimation
    mask_a_rel = task_data.get("mask_a_path")
    mask_a = None
    if mask_a_rel:
        p_ma = g3a_root / mask_a_rel
        if not p_ma.is_file():
            raise FileNotFoundError(f"Missing analytic mask file: {p_ma}")
        with Image.open(p_ma) as im_m:
            mask_a = np.array(im_m, dtype=bool)

    u_roi = task_data["u"] - task_data["roi_u0"]
    v_roi = task_data["v"] - task_data["roi_v0"]
    sig_clean = estimate_target_signal(lin_mono_clean, mask_a, center_uv=(u_roi, v_roi), is_degraded=False)
    sig_deg = estimate_target_signal(lin_mono_deg, mask_a, center_uv=(u_roi, v_roi), is_degraded=True)

    # Visible mask centroid for localization (if visible_px > 0)
    visible_centroid = None
    mask_v_rel = task_data.get("mask_v_path")
    if mask_v_rel:
        p_mv = g3a_root / mask_v_rel
        if not p_mv.is_file():
            raise FileNotFoundError(f"Missing visible mask file: {p_mv}")
        with Image.open(p_mv) as im_v:
            mask_v = np.array(im_v, dtype=bool)
            if np.any(mask_v):
                ys, xs = np.where(mask_v)
                visible_centroid = (float(np.mean(xs) + task_data["roi_u0"]), float(np.mean(ys) + task_data["roi_v0"]))

    # 4. Search area for FP density
    box_analytic = task_data["box_analytic"]
    p_cam = task_data["p_cam_ecef"]
    r_cam = task_data["R_cam_ecef"]
    sky_mpx, earth_mpx = compute_roi_sky_earth_area(
        p_cam, r_cam, ring.camera_model,
        task_data["roi_u0"], task_data["roi_v0"],
        width=1024, height=1024,
        exclusion_box=[
            box_analytic[0] - 16.0,
            box_analytic[1] - 16.0,
            box_analytic[2] + 16.0,
            box_analytic[3] + 16.0,
        ],
    )

    # 5. D0 detections and hit evaluation for clean and degraded
    cmos_clean = compute_cmo_multi(lin_mono_clean, k_values=K_VALUES)
    cmos_deg = compute_cmo_multi(lin_mono_deg, k_values=K_VALUES)

    tile_size = d0_cfg.get("tile_size", 64)  # default-ok: optional fallback default
    tile_stride = d0_cfg.get("tile_stride", 32)  # default-ok: optional fallback default
    mad_scale = d0_cfg.get("mad_scale", 1.4826)  # default-ok: optional fallback default
    dilation_size = d0_cfg.get("dilation_kernel_size", 3)  # default-ok: optional fallback default

    eval_by_sensor_k = {"clean": {}, "degraded": {}}
    target_uv = (task_data["u"], task_data["v"])
    s_px = task_data["s_px"]

    for k_val in K_VALUES:
        # Clean
        cmo_c = cmos_clean[k_val]
        m_c, s_c = compute_local_stats(cmo_c, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
        z_c = compute_z_score(cmo_c, m_c, s_c, sigma_min=clean_floor)

        fp_sky_c = {f"{tau:.1f}": 0 for tau in TAU_LIST}
        fp_earth_c = {f"{tau:.1f}": 0 for tau in TAU_LIST}
        pri_hits_c = {}
        sec_hits_c = {}
        res_nom_c = None

        for tau_idx, tau in enumerate(TAU_LIST):
            t_k = f"{tau:.1f}"
            dets_c = extract_components(
                z_c, cmo_c, tau=tau,
                u0=task_data["roi_u0"], v0=task_data["roi_v0"],
                image_id=task_data["instance_id"], camera=task_data["cam_id"],
                dilation_size=dilation_size,
            )
            if not dets_c:
                for rem_tau in TAU_LIST[tau_idx:]:
                    rem_k = f"{rem_tau:.1f}"
                    fp_sky_c[rem_k] = 0
                    fp_earth_c[rem_k] = 0
                    pri_hits_c[rem_k] = 0
                    sec_hits_c[rem_k] = 0
                break
            ev = evaluate_hits_and_fps_for_roi(
                dets_c, target_uv, box_analytic, s_px, p_cam, r_cam, ring.camera_model
            )
            fp_sky_c[t_k] = ev["fp_counts_sky"].get(t_k, 0)  # default-ok: frequency counter accumulator
            fp_earth_c[t_k] = ev["fp_counts_earth"].get(t_k, 0)  # default-ok: frequency counter accumulator
            pri_hits_c[t_k] = 1 if ev["primary_best_score"] > -float("inf") else 0
            sec_hits_c[t_k] = 1 if ev["secondary_best_score"] > -float("inf") else 0
            if tau == 4.0:
                res_nom_c = ev

        if res_nom_c is None:
            res_nom_c = {
                "primary_best_score": -float("inf"),
                "secondary_best_score": -float("inf"),
                "matched_det_u": None,
                "matched_det_v": None,
            }

        eval_by_sensor_k["clean"][k_val] = {
            "primary_best_score": res_nom_c["primary_best_score"],
            "secondary_best_score": res_nom_c["secondary_best_score"],
            "matched_det_u": res_nom_c["matched_det_u"],
            "matched_det_v": res_nom_c["matched_det_v"],
            "primary_hits_by_tau": pri_hits_c,
            "secondary_hits_by_tau": sec_hits_c,
            "fp_counts_sky": fp_sky_c,
            "fp_counts_earth": fp_earth_c,
            "fp_candidates": [],
        }

        # Degraded
        sigma_min_deg = float(d0_cfg["sigma_min_degraded"])
        cmo_d = cmos_deg[k_val]
        m_d, s_d = compute_local_stats(cmo_d, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
        z_d = compute_z_score(cmo_d, m_d, s_d, sigma_min=sigma_min_deg)

        fp_sky_d = {f"{tau:.1f}": 0 for tau in TAU_LIST}
        fp_earth_d = {f"{tau:.1f}": 0 for tau in TAU_LIST}
        pri_hits_d = {}
        sec_hits_d = {}
        res_nom_d = None

        for tau_idx, tau in enumerate(TAU_LIST):
            t_k = f"{tau:.1f}"
            dets_d = extract_components(
                z_d, cmo_d, tau=tau,
                u0=task_data["roi_u0"], v0=task_data["roi_v0"],
                image_id=task_data["instance_id"], camera=task_data["cam_id"],
                dilation_size=dilation_size,
            )
            if not dets_d:
                for rem_tau in TAU_LIST[tau_idx:]:
                    rem_k = f"{rem_tau:.1f}"
                    fp_sky_d[rem_k] = 0
                    fp_earth_d[rem_k] = 0
                    pri_hits_d[rem_k] = 0
                    sec_hits_d[rem_k] = 0
                break
            ev = evaluate_hits_and_fps_for_roi(
                dets_d, target_uv, box_analytic, s_px, p_cam, r_cam, ring.camera_model
            )
            fp_sky_d[t_k] = ev["fp_counts_sky"].get(t_k, 0)  # default-ok: frequency counter accumulator
            fp_earth_d[t_k] = ev["fp_counts_earth"].get(t_k, 0)  # default-ok: frequency counter accumulator
            pri_hits_d[t_k] = 1 if ev["primary_best_score"] > -float("inf") else 0
            sec_hits_d[t_k] = 1 if ev["secondary_best_score"] > -float("inf") else 0
            if tau == 4.0:
                res_nom_d = ev

        if res_nom_d is None:
            res_nom_d = {
                "primary_best_score": -float("inf"),
                "secondary_best_score": -float("inf"),
                "matched_det_u": None,
                "matched_det_v": None,
            }

        eval_by_sensor_k["degraded"][k_val] = {
            "primary_best_score": res_nom_d["primary_best_score"],
            "secondary_best_score": res_nom_d["secondary_best_score"],
            "matched_det_u": res_nom_d["matched_det_u"],
            "matched_det_v": res_nom_d["matched_det_v"],
            "primary_hits_by_tau": pri_hits_d,
            "secondary_hits_by_tau": sec_hits_d,
            "fp_counts_sky": fp_sky_d,
            "fp_counts_earth": fp_earth_d,
            "fp_candidates": [],
        }

    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "type": "positive",
        "instance_id": task_data["instance_id"],
        "encounter_id": task_data["encounter_id"],
        "class_name": task_data["class_name"],
        "background": task_data["background"],
        "range_m": task_data["range_m"],
        "s_px": task_data["s_px"],
        "target_uv": target_uv,
        "box_analytic": box_analytic,
        "visible_centroid": visible_centroid,
        "signal_clean": {
            "a_c": sig_clean.a_c,
            "mean_contrast": sig_clean.mean_contrast,
            "sigma_bg": sig_clean.sigma_bg,
        },
        "signal_degraded": {
            "a_c": sig_deg.a_c,
            "mean_contrast": sig_deg.mean_contrast,
            "sigma_bg": sig_deg.sigma_bg,
            "degraded_snr": sig_deg.degraded_snr,
        },
        "search_area_mpx": {"sky": sky_mpx, "earth": earth_mpx},
        "eval_by_sensor_k": eval_by_sensor_k,
        "dt_ms": dt_ms,
    }


def process_negative_roi_task(task_data: dict[str, Any]) -> dict[str, Any]:
    """Worker task: evaluate one negative ROI and its assigned phantom targets."""
    t0 = time.perf_counter()
    g3a_root = _WORKER_G3A_ROOT
    ring = _WORKER_RING_MODEL
    scfg = _WORKER_SENSOR_CFG
    kernel = _WORKER_PSF_KERNEL
    prnu = _WORKER_PRNU
    dsnu = _WORKER_DSNU
    d0_cfg = _WORKER_D0_CFG

    rgb_rel = task_data["png_path"]
    rgb_full = g3a_root / rgb_rel
    with Image.open(rgb_full) as im:
        rgb_arr = np.array(im.convert("RGB"), dtype=np.uint8)

    # 1. Clean linear monochrome via EOTF LUT
    if _WORKER_EOTF_R is not None:
        lin_mono_clean = (
            _WORKER_EOTF_R[rgb_arr[:, :, 0]]
            + _WORKER_EOTF_G[rgb_arr[:, :, 1]]
            + _WORKER_EOTF_B[rgb_arr[:, :, 2]]
        )
    else:
        lin_rgb = srgb_eotf(rgb_arr)
        lin_mono_clean = linear_rgb_to_monochrome(lin_rgb)

    # Precompute clean floor grid once for all k values
    clean_floor = compute_clean_floor_grid(
        lin_mono_clean,
        tile_size=d0_cfg.get("tile_size", 64),  # default-ok: optional fallback default
        tile_stride=d0_cfg.get("tile_stride", 32),  # default-ok: optional fallback default
    )

    # 2. Degraded monochrome via WP-03 chain
    blurred = apply_psf(lin_mono_clean, kernel)
    raw_dn = apply_sensor_noise(
        linear_radiance=blurred,
        prnu_map=prnu,
        dsnu_map=dsnu,
        E=scfg.full_well_e,
        full_well_e=scfg.full_well_e,
        sigma_r_e=scfg.sigma_r_e,
        dark_current_e_per_s=scfg.dark_current_e_per_s,
        exposure_time_s=scfg.exposure_time_s,
        k_dn_per_e=scfg.k_dn_per_e,
        bits=scfg.bits,
        black_level_dn=scfg.black_level_dn,
        rng_temporal=np.random.Generator(np.random.PCG64(task_data["k_frame"])),
    )
    lin_mono_deg = (raw_dn.astype(np.float64) - scfg.black_level_dn) / scfg.k_dn_per_e

    # 3. Search area
    p_cam = task_data["p_cam_ecef"]
    r_cam = task_data["R_cam_ecef"]
    sky_mpx, earth_mpx = compute_roi_sky_earth_area(
        p_cam, r_cam, ring.camera_model,
        task_data["roi_u0"], task_data["roi_v0"],
        width=1024, height=1024,
        exclusion_box=None,
    )

    # 4. D0 detections and phantom hit evaluation
    cmos_clean = compute_cmo_multi(lin_mono_clean, k_values=K_VALUES)
    cmos_deg = compute_cmo_multi(lin_mono_deg, k_values=K_VALUES)

    assigned_phantoms = task_data.get("assigned_phantoms", [])
    eval_by_sensor_k = {"clean": {}, "degraded": {}}
    phantom_evals = {p["instance_id"]: {"clean": {}, "degraded": {}} for p in assigned_phantoms}

    tile_size = d0_cfg.get("tile_size", 64)  # default-ok: optional fallback default
    tile_stride = d0_cfg.get("tile_stride", 32)  # default-ok: optional fallback default
    mad_scale = d0_cfg.get("mad_scale", 1.4826)  # default-ok: optional fallback default
    dilation_size = d0_cfg.get("dilation_kernel_size", 3)  # default-ok: optional fallback default

    for k_val in K_VALUES:
        # Clean
        cmo_c = cmos_clean[k_val]
        m_c, s_c = compute_local_stats(cmo_c, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
        z_c = compute_z_score(cmo_c, m_c, s_c, sigma_min=clean_floor)

        fp_sky_c = {f"{tau:.1f}": 0 for tau in TAU_LIST}
        fp_earth_c = {f"{tau:.1f}": 0 for tau in TAU_LIST}
        phantom_hits_c = {p["instance_id"]: {} for p in assigned_phantoms}
        phantom_best_c = {p["instance_id"]: -float("inf") for p in assigned_phantoms}

        for tau_idx, tau in enumerate(TAU_LIST):
            t_k = f"{tau:.1f}"
            dets_c = extract_components(
                z_c, cmo_c, tau=tau,
                u0=task_data["roi_u0"], v0=task_data["roi_v0"],
                image_id=task_data["negative_id"], camera=task_data["cam_id"],
                dilation_size=dilation_size,
            )
            if not dets_c:
                for rem_tau in TAU_LIST[tau_idx:]:
                    rem_k = f"{rem_tau:.1f}"
                    fp_sky_c[rem_k] = 0
                    fp_earth_c[rem_k] = 0
                    for ph in assigned_phantoms:
                        phantom_hits_c[ph["instance_id"]][rem_k] = 0
                break
            ev = evaluate_hits_and_fps_for_roi(
                dets_c, None, None, None, p_cam, r_cam, ring.camera_model
            )
            fp_sky_c[t_k] = ev["fp_counts_sky"].get(t_k, 0)  # default-ok: frequency counter accumulator
            fp_earth_c[t_k] = ev["fp_counts_earth"].get(t_k, 0)  # default-ok: frequency counter accumulator

            for ph in assigned_phantoms:
                ph_uv = (ph["u_full"], ph["v_full"])
                ph_box = ph["box_phantom"]
                ph_s_px = ph["s_px"]
                ev_cl = evaluate_hits_and_fps_for_roi(
                    dets_c, ph_uv, ph_box, ph_s_px, p_cam, r_cam, ring.camera_model, skip_fps=True
                )
                hit = 1 if ev_cl["primary_best_score"] > -float("inf") else 0
                phantom_hits_c[ph["instance_id"]][t_k] = hit
                if tau == 4.0:
                    phantom_best_c[ph["instance_id"]] = ev_cl["primary_best_score"]

        eval_by_sensor_k["clean"][k_val] = {
            "primary_best_score": -float("inf"),
            "secondary_best_score": -float("inf"),
            "matched_det_u": None,
            "matched_det_v": None,
            "fp_counts_sky": fp_sky_c,
            "fp_counts_earth": fp_earth_c,
            "fp_candidates": [],
        }
        for ph in assigned_phantoms:
            phantom_evals[ph["instance_id"]]["clean"][k_val] = {
                "primary_best_score": phantom_best_c[ph["instance_id"]],
                "secondary_best_score": phantom_best_c[ph["instance_id"]],
                "hits_by_tau": phantom_hits_c[ph["instance_id"]],
            }

        # Degraded
        sigma_min_deg = float(d0_cfg["sigma_min_degraded"])
        cmo_d = cmos_deg[k_val]
        m_d, s_d = compute_local_stats(cmo_d, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
        z_d = compute_z_score(cmo_d, m_d, s_d, sigma_min=sigma_min_deg)

        fp_sky_d = {f"{tau:.1f}": 0 for tau in TAU_LIST}
        fp_earth_d = {f"{tau:.1f}": 0 for tau in TAU_LIST}
        phantom_hits_d = {p["instance_id"]: {} for p in assigned_phantoms}
        phantom_best_d = {p["instance_id"]: -float("inf") for p in assigned_phantoms}

        for tau_idx, tau in enumerate(TAU_LIST):
            t_k = f"{tau:.1f}"
            dets_d = extract_components(
                z_d, cmo_d, tau=tau,
                u0=task_data["roi_u0"], v0=task_data["roi_v0"],
                image_id=task_data["negative_id"], camera=task_data["cam_id"],
                dilation_size=dilation_size,
            )
            if not dets_d:
                for rem_tau in TAU_LIST[tau_idx:]:
                    rem_k = f"{rem_tau:.1f}"
                    fp_sky_d[rem_k] = 0
                    fp_earth_d[rem_k] = 0
                    for ph in assigned_phantoms:
                        phantom_hits_d[ph["instance_id"]][rem_k] = 0
                break
            ev = evaluate_hits_and_fps_for_roi(
                dets_d, None, None, None, p_cam, r_cam, ring.camera_model
            )
            fp_sky_d[t_k] = ev["fp_counts_sky"].get(t_k, 0)  # default-ok: frequency counter accumulator
            fp_earth_d[t_k] = ev["fp_counts_earth"].get(t_k, 0)  # default-ok: frequency counter accumulator

            for ph in assigned_phantoms:
                ph_uv = (ph["u_full"], ph["v_full"])
                ph_box = ph["box_phantom"]
                ph_s_px = ph["s_px"]
                ev_dg = evaluate_hits_and_fps_for_roi(
                    dets_d, ph_uv, ph_box, ph_s_px, p_cam, r_cam, ring.camera_model, skip_fps=True
                )
                hit = 1 if ev_dg["primary_best_score"] > -float("inf") else 0
                phantom_hits_d[ph["instance_id"]][t_k] = hit
                if tau == 4.0:
                    phantom_best_d[ph["instance_id"]] = ev_dg["primary_best_score"]

        eval_by_sensor_k["degraded"][k_val] = {
            "primary_best_score": -float("inf"),
            "secondary_best_score": -float("inf"),
            "matched_det_u": None,
            "matched_det_v": None,
            "fp_counts_sky": fp_sky_d,
            "fp_counts_earth": fp_earth_d,
            "fp_candidates": [],
        }
        for ph in assigned_phantoms:
            phantom_evals[ph["instance_id"]]["degraded"][k_val] = {
                "primary_best_score": phantom_best_d[ph["instance_id"]],
                "secondary_best_score": phantom_best_d[ph["instance_id"]],
                "hits_by_tau": phantom_hits_d[ph["instance_id"]],
            }

    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "type": "negative",
        "negative_id": task_data["negative_id"],
        "encounter_id": task_data["encounter_id"],
        "search_area_mpx": {"sky": sky_mpx, "earth": earth_mpx},
        "eval_by_sensor_k": eval_by_sensor_k,
        "phantom_evals": phantom_evals,
        "dt_ms": dt_ms,
    }


def process_batch_worker(batch: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Worker dispatcher for a mixed batch of positive and negative ROI tasks."""
    results = []
    for item_type, item_data in batch:
        if item_type == "positive":
            results.append(process_positive_roi_task(item_data))
        elif item_type == "negative":
            results.append(process_negative_roi_task(item_data))
    return results


def load_g3a_dataset_and_pair(
    g3a_root: Path,
    max_frames: int | None = None,
    encounters_filter: list[str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Discover blocks, load frames.jsonl metadata, pair phantoms, and return dataset items."""
    blocks_dir = g3a_root / "blocks"
    manifests: dict[str, str] = {}
    frames_files: list[Path] = []

    if blocks_dir.is_dir():
        for b_path in sorted(blocks_dir.glob("b*")):
            if not b_path.is_dir():
                continue
            man_file = b_path / "run_manifest.json"
            if man_file.is_file():
                h = hashlib.sha256(man_file.read_bytes()).hexdigest()
                manifests[b_path.name] = h
            fr_file = b_path / "frames.jsonl"
            if fr_file.is_file():
                frames_files.append(fr_file)

    if not frames_files:
        root_fr = g3a_root / "frames.jsonl"
        if root_fr.is_file():
            frames_files.append(root_fr)

    if not frames_files:
        raise FileNotFoundError(f"No frames.jsonl found under {g3a_root}")

    ring = TrafficRing()
    pos_instances: list[dict[str, Any]] = []
    neg_instances: list[dict[str, Any]] = []

    frames_loaded = 0
    for fr_path in frames_files:
        with open(fr_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_frames is not None and frames_loaded >= max_frames:
                    break
                line_str = line.strip()
                if not line_str:
                    continue
                obj = json.loads(line_str)
                enc = obj["encounter_id"]
                if encounters_filter and enc not in encounters_filter:
                    continue

                frames_loaded += 1
                k_frame = obj.get("k", frames_loaded)
                t_s = obj.get("t_s", 0.0)  # default-ok: optional fallback default

                # Negative ROI
                neg = obj.get("negative")
                if neg and neg.get("png"):
                    cam_id = neg["cam"]
                    cam_info = obj["cameras"].get(cam_id, {})
                    neg_instances.append({
                        "negative_id": f"{enc}_{k_frame:06d}_{cam_id}_neg",
                        "encounter_id": enc,
                        "k_frame": k_frame,
                        "t_s": t_s,
                        "cam_id": cam_id,
                        "roi_u0": float(neg["roi"][0]),
                        "roi_v0": float(neg["roi"][1]),
                        "png_path": neg["png"],
                        "p_cam_ecef": np.array(cam_info.get("p_ecef_m", [0, 0, 0]), dtype=np.float64),
                        "R_cam_ecef": np.array(cam_info.get("R_ecef_cam", np.eye(3)), dtype=np.float64),
                    })

                # Positive ROIs
                for tr in obj.get("traffic", []):
                    cls_name = tr.get("class", "unknown")
                    rng_m = tr.get("range_m", 1000.0)  # default-ok: optional fallback default
                    for c_id, c_data in tr.get("cams", {}).items():
                        if c_data.get("kind") == "ring" and c_data.get("in_image") and c_data.get("rgb") and c_data.get("box_analytic") is not None:
                            cam_info = obj["cameras"].get(c_id, {})
                            box_an = [float(x) for x in c_data["box_analytic"]]
                            w_b = box_an[2] - box_an[0]
                            h_b = box_an[3] - box_an[1]
                            s_px = max(w_b, h_b)
                            inst_id = f"{enc}_{k_frame:06d}_{c_id}"

                            pos_instances.append({
                                "instance_id": inst_id,
                                "encounter_id": enc,
                                "k_frame": k_frame,
                                "t_s": t_s,
                                "cam_id": c_id,
                                "class_name": cls_name,
                                "range_m": float(rng_m),
                                "s_px": float(s_px),
                                "u": float(c_data["u"]),
                                "v": float(c_data["v"]),
                                "box_analytic": box_an,
                                "roi_u0": float(c_data["roi"][0]),
                                "roi_v0": float(c_data["roi"][1]),
                                "background": c_data.get("background", "sky"),
                                "rgb_path": c_data["rgb"],
                                "mask_a_path": c_data.get("mask_A"),
                                "mask_v_path": c_data.get("mask_V"),
                                "amodal_px": int(c_data.get("amodal_px", 0)),  # default-ok: geometry dimension fallback
                                "visible_px": int(c_data.get("visible_px", 0)),  # default-ok: geometry dimension fallback
                                "p_cam_ecef": np.array(cam_info.get("p_ecef_m", [0, 0, 0]), dtype=np.float64),
                                "R_cam_ecef": np.array(cam_info.get("R_ecef_cam", np.eye(3)), dtype=np.float64),
                            })

    # Pair phantom targets
    phantom_map, unpaired_instances = pair_phantom_targets(pos_instances, neg_instances, ring.camera_model, return_unpaired=True)

    # Attach assigned phantoms to negative instances
    neg_lookup = {neg["negative_id"]: neg for neg in neg_instances}
    for neg in neg_instances:
        neg["assigned_phantoms"] = []

    for inst_id, ph in phantom_map.items():
        neg_id = ph["negative_id"]
        if neg_id in neg_lookup:
            neg_lookup[neg_id]["assigned_phantoms"].append(ph)

    return manifests, pos_instances, neg_instances, phantom_map, unpaired_instances


def get_stratified_encounters(
    n_encounters: int = 18,
    seed: int = 20260926,
    encounters_file: Path | None = None,
) -> list[str]:
    """Sample n_encounters stratified evenly across the 6 intruder classes."""
    if encounters_file is None:
        encounters_file = REPO_ROOT / "data" / "cache" / "wp05_encounters_sampler_v1.jsonl"

    classes_dict: dict[str, list[str]] = {}
    with open(encounters_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            cid = rec.get("class_id", "unknown")
            classes_dict.setdefault(cid, []).append(rec["id"])

    rng = np.random.default_rng(seed)
    n_classes = len(classes_dict)
    per_class = max(1, n_encounters // n_classes)

    selected = []
    for cid in sorted(classes_dict.keys()):
        pool = sorted(classes_dict[cid])
        chosen = rng.choice(pool, size=min(per_class, len(pool)), replace=False).tolist()
        selected.extend(chosen)

    return sorted(selected)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resolve_g3a_root(raw_path: Path | str) -> Path:
    """Resolve the G3a root directory as given or relative to the repo root."""
    p = Path(raw_path)
    if p.is_dir():
        return p.resolve()
    if (REPO_ROOT / p).is_dir():
        return (REPO_ROOT / p).resolve()
    # D19: no fallback to a machine path or another data root (merge of G-B13)
    raise FileNotFoundError(f"G3a root directory does not exist: {p}")


def write_closure_config(
    wp05_closure: dict[str, Any],
    measured_lambda: float,
    op_k: int,
    op_tau: float,
    out_config_path: Path = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0_measured_d0.json",
    base_config_path: Path = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0.json",
    release_sha256: str | None = None,
    fp_per_mpx_earth: float | None = None,
    fp_per_mpx_sky: float | None = None,
) -> None:
    """Write or update closure config with honest provenance per R1 (F-B9-1) and F9."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["fppi_lambda"] = float(measured_lambda)
    cfg["provenances"]["fppi_lambda"] = f"WP-04 G3a D0 natural measured (k*={op_k}, tau*={op_tau:.1f}, {measured_lambda:.6f} FP/cam-frame)"

    if fp_per_mpx_earth is not None and fp_per_mpx_sky is not None and fp_per_mpx_sky > 0.0:
        terrain_factor = float(fp_per_mpx_earth) / float(fp_per_mpx_sky)
        cfg["terrain_clutter_factor"] = terrain_factor
        cfg["provenances"]["terrain_clutter_factor"] = (
            f"WP-04 G0 D0, G3a {release_sha256 or 'v0.1'} "
            f"(output/xvs/wp04/g0/g3a_results.json:false_positive_sweeps.degraded.{op_k}.{op_tau:.1f}: "
            f"fp_per_mpx_earth / fp_per_mpx_sky = {fp_per_mpx_earth:.6f} / {fp_per_mpx_sky:.6f} = {terrain_factor:.6f})"
        )

    if release_sha256 is not None:
        cfg["release_json_sha256"] = release_sha256
        cfg.setdefault("provenances", {})["release_json_sha256"] = (
            f"WP-02 G3a v0.1 release SHA-256 ({release_sha256})"
        )

    if wp05_closure.get("status") == "FITTED" and wp05_closure.get("s50_px") is not None:
        cfg["s50_px"] = float(wp05_closure["s50_px"])
        cfg["w_px"] = float(wp05_closure["w_px"])
        cfg["provenances"]["s50_px"] = wp05_closure["provenance"]
        cfg["provenances"]["w_px"] = wp05_closure["provenance"]
        if wp05_closure.get("w_ln") is not None:
            cfg["w_ln"] = float(wp05_closure["w_ln"])
            cfg["pd_mode"] = "log_logistic"
            cfg["provenances"]["w_ln"] = wp05_closure["provenance"]
            cfg["provenances"]["pd_mode"] = wp05_closure["provenance"]
        if wp05_closure.get("empirical_pd_by_spx"):
            tbl = []
            for k_s, v_s in wp05_closure["empirical_pd_by_spx"].items():
                parts = k_s.replace("px", "").replace("[", "").replace(")", "").strip().split(",")
                if len(parts) == 2:
                    try:
                        mid_s = 0.5 * (float(parts[0]) + float(parts[1]))
                        tbl.append([mid_s, float(v_s["raw_pd"])])
                    except Exception:
                        pass
                elif ">=" in k_s:
                    try:
                        val = float(k_s.replace(">=", "").replace("px", "").strip())
                        tbl.append([val, float(v_s["raw_pd"])])
                    except Exception:
                        pass
            if tbl:
                cfg["pd_empirical_table"] = sorted(tbl, key=lambda x: x[0])
    else:
        cfg["s50_px"] = 3.0
        cfg["w_px"] = 0.7
        cfg["provenances"]["s50_px"] = "WP05 §1 baseline (not measured)"
        cfg["provenances"]["w_px"] = "WP05 §1 baseline (not measured)"

    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote closure config to {out_config_path} (status: {wp05_closure.get('status')})")


def main() -> None:
    parser = argparse.ArgumentParser(description="WP-04 G0 G3a Traffic Detection Evaluation")
    parser.add_argument("--g3a-root", type=Path, default=DEFAULT_PILOT_ROOT, help="Root of G3a data")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes (max cores-4)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (testing)")
    parser.add_argument("--encounters", nargs="+", default=None, help="Filter specific encounters")
    parser.add_argument("--subsample-encounters", type=int, default=None, help="Number of encounters to subsample (stratified by class)")
    parser.add_argument("--subsample-seed", type=int, default=20260926, help="Random seed for encounter subsampling")
    parser.add_argument("--from-cache", type=Path, default=None, help="Load raw evaluation results from cache file")
    parser.add_argument("--update-closure-config-from-json", type=Path, default=None, help="Update g0_measured_d0.json from existing g3a_results.json and exit")
    parser.add_argument("--closure-tau", type=float, default=20.0, help="Operational tau threshold for WP-05 closure fit")
    parser.add_argument("--closure-k", type=int, default=3, help="Operational k filter for WP-05 closure fit")
    args = parser.parse_args()

    if args.update_closure_config_from_json is not None:
        with open(args.update_closure_config_from_json, "r", encoding="utf-8") as f:
            g3a_res = json.load(f)
        op_info = g3a_res["operating_point"]
        c_k = int(op_info["closure_k"])
        c_tau = float(op_info["closure_tau"])
        fp_st = g3a_res["false_positive_sweeps"]["degraded"][str(c_k)][f"{c_tau:.1f}"]
        write_closure_config(
            wp05_closure=g3a_res["wp05_closure_fit"],
            measured_lambda=float(fp_st["lambda_per_camera_frame"]),
            op_k=c_k,
            op_tau=c_tau,
            release_sha256=g3a_res["dataset_summary"].get("release_json_sha256"),
            fp_per_mpx_earth=float(fp_st["fp_per_mpx_earth"]),
            fp_per_mpx_sky=float(fp_st["fp_per_mpx_sky"]),
        )
        return

    g3a_root = resolve_g3a_root(args.g3a_root)
    release_file = g3a_root / "RELEASE.json"
    release_sha256: str | None = None
    if release_file.is_file():
        release_sha256 = compute_file_sha256(release_file)
        print(f"Detected RELEASE.json: {release_file} (SHA-256: {release_sha256})")
    else:
        print(f"RELEASE.json not present under {g3a_root}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    max_allowed_workers = max(1, (os.cpu_count() or 4) - 4)
    workers = min(args.workers, max_allowed_workers)

    is_pilot = ("pilot" in str(g3a_root).lower()) or ("pilot" in str(out_dir).lower())
    run_label = "PILOT, not a G0 result" if is_pilot else "G0"

    print("=" * 72)
    print(f"  WP-04 G3a Evaluation Runner ({run_label})")
    print(f"  Root: {g3a_root}")
    if release_sha256:
        print(f"  RELEASE.json SHA-256: {release_sha256}")
    print(f"  Workers: {workers} (system cores: {os.cpu_count()})")
    print("=" * 72)

    if args.from_cache:
        import gzip
        import pickle
        print(f"\nLoading raw evaluation results from cache: {args.from_cache}")
        with gzip.open(args.from_cache, "rb") as f_cache:
            cache_data = pickle.load(f_cache)
        if release_sha256 is None:
            release_sha256 = cache_data.get("release_json_sha256")
            if release_sha256:
                print(f"Recovered RELEASE.json SHA-256 from cache: {release_sha256}")
        results_positive = cache_data["results_positive"]
        results_negative = cache_data["results_negative"]
        manifests = cache_data["manifests"]
        phantom_map = cache_data["phantom_map"]
        unpaired_instances = cache_data.get("unpaired_instances", [k["instance_id"] for k in results_positive if k["instance_id"] not in phantom_map])
        wall_eval_s = cache_data.get("wall_eval_s", 0.0)  # default-ok: timing benchmark fallback
        total_cpu_ms = cache_data.get("total_cpu_ms", 0.0)  # default-ok: timing benchmark fallback
        all_tasks_len = cache_data.get("all_tasks_len", len(results_positive) + len(results_negative))
        pos_instances = [r for r in results_positive]
        neg_instances = [r for r in results_negative]
        all_tasks = pos_instances + neg_instances
        ms_per_roi = total_cpu_ms / max(1, all_tasks_len)
        full_g3a_estimated_rois = 60000
        projected_cpu_h = (full_g3a_estimated_rois * ms_per_roi / 1000.0) / 3600.0
        projected_wall_h = projected_cpu_h / workers
        print(f"Loaded {len(results_positive):,} positive and {len(results_negative):,} negative results from cache.")
    else:
        encounters_filter = args.encounters
        if encounters_filter is None and args.subsample_encounters is not None:
            encounters_filter = get_stratified_encounters(args.subsample_encounters, args.subsample_seed)
            print(f"Subsampled {len(encounters_filter)} encounters (seed={args.subsample_seed}, stratified across classes): {encounters_filter}")

        # 1. Load dataset metadata and manifests
        t_load_start = time.perf_counter()
        manifests, pos_instances, neg_instances, phantom_map, unpaired_instances = load_g3a_dataset_and_pair(
            g3a_root=g3a_root,
            max_frames=args.max_frames,
            encounters_filter=encounters_filter,
        )
        print(f"Loaded in {(time.perf_counter() - t_load_start):.2f}s:")
        print(f"  Manifests: {manifests}")
        print(f"  Positive instances: {len(pos_instances):,}")
        print(f"  Negative instances: {len(neg_instances):,}")
        print(f"  Paired phantoms: {len(phantom_map):,}, Unpaired positives: {len(unpaired_instances):,}")

        # 2. Prepare batches for parallel worker execution
        all_tasks = [("positive", it) for it in pos_instances] + [("negative", it) for it in neg_instances]
        batch_size = 25
        batches = [all_tasks[i:i + batch_size] for i in range(0, len(all_tasks), batch_size)]
        print(f"\nProcessing {len(all_tasks):,} total ROI tasks across {len(batches)} batches on {workers} workers...")

        results_positive = []
        results_negative = []
        total_cpu_ms = 0.0

        t_eval_start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(str(g3a_root),),
        ) as executor:
            future_to_batch = {executor.submit(process_batch_worker, b): idx for idx, b in enumerate(batches)}
            completed_tasks = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                b_idx = future_to_batch[future]
                try:
                    b_results = future.result()
                    for r in b_results:
                        total_cpu_ms += r.get("dt_ms", 0.0)  # default-ok: timing benchmark fallback
                        if r["type"] == "positive":
                            results_positive.append(r)
                        else:
                            results_negative.append(r)
                    completed_tasks += len(b_results)
                    elapsed = time.perf_counter() - t_eval_start
                    rate = completed_tasks / max(1e-6, elapsed)
                    print(f"  [Progress] {completed_tasks:,}/{len(all_tasks):,} ROIs ({completed_tasks/len(all_tasks)*100.0:.1f}%) | {rate:.1f} ROI/s | Wall: {elapsed:.1f}s")
                except Exception as exc:
                    print(f"  Batch {b_idx} failed: {exc}", file=sys.stderr)
                    raise

        wall_eval_s = time.perf_counter() - t_eval_start
        all_tasks_len = len(all_tasks)
        ms_per_roi = total_cpu_ms / max(1, all_tasks_len)
        print(f"\nEvaluation complete in {wall_eval_s:.1f}s wall ({total_cpu_ms/1000.0:.1f}s CPU).")
        print(f"Average CPU time per ROI: {ms_per_roi:.1f} ms.")

        # Projected runtime for full G3a (approx 50,000 frames -> ~60,000 ROIs)
        full_g3a_estimated_rois = 60000
        projected_cpu_h = (full_g3a_estimated_rois * ms_per_roi / 1000.0) / 3600.0
        projected_wall_h = projected_cpu_h / workers
        print(f"Projected full G3a CPU time ({full_g3a_estimated_rois:,} ROIs): {projected_cpu_h:.2f} h CPU -> {projected_wall_h:.2f} h wall on {workers} workers.")

        # Save cache immediately
        cache_path = out_dir / "raw_eval_cache.pkl.gz"
        print(f"Caching raw evaluation results to {cache_path}...")
        import gzip
        import pickle
        with gzip.open(cache_path, "wb") as f_cache:
            pickle.dump({
                "results_positive": results_positive,
                "results_negative": results_negative,
                "manifests": manifests,
                "release_json_sha256": release_sha256,
                "phantom_map": phantom_map,
                "wall_eval_s": wall_eval_s,
                "total_cpu_ms": total_cpu_ms,
                "all_tasks_len": all_tasks_len,
            }, f_cache)
        print(f"Cached {len(results_positive):,} positive and {len(results_negative):,} negative results.")

    # 3. Assemble phantom items from negative results
    phantom_results_map: dict[str, dict[str, Any]] = {}
    for nr in results_negative:
        for p_id, p_ev in nr["phantom_evals"].items():
            if p_id in phantom_map:
                ph_info = phantom_map[p_id]
                phantom_results_map[p_id] = {
                    "instance_id": p_id,
                    "encounter_id": ph_info["encounter_id"],
                    "background": ph_info["background"],
                    "s_px": ph_info["s_px"],
                    "eval_by_sensor_k": p_ev,
                }

    phantom_results_list = list(phantom_results_map.values())

    # 4. False Positive Analysis across tau sweep for each (sensor, k)
    print("\nSynthesizing false positive density sweeps and Poisson upper bounds...")
    total_sky_mpx = sum(r["search_area_mpx"]["sky"] for r in results_positive) + sum(r["search_area_mpx"]["sky"] for r in results_negative)
    total_earth_mpx = sum(r["search_area_mpx"]["earth"] for r in results_positive) + sum(r["search_area_mpx"]["earth"] for r in results_negative)
    print(f"Total search area: sky = {total_sky_mpx:.2f} Mpx, earth = {total_earth_mpx:.2f} Mpx (Total = {total_sky_mpx+total_earth_mpx:.2f} Mpx)")

    fp_sweeps = {}
    for sensor in SENSOR_MODES:
        fp_sweeps[sensor] = {}
        for k_val in K_VALUES:
            sky_counts = {f"{tau:.1f}": 0 for tau in TAU_LIST}
            earth_counts = {f"{tau:.1f}": 0 for tau in TAU_LIST}
            for r in results_positive:
                ev = r["eval_by_sensor_k"][sensor][k_val]
                if "fp_counts_sky" in ev and ev["fp_counts_sky"]:
                    for t_k, c in ev["fp_counts_sky"].items():
                        sky_counts[t_k] += c
                    for t_k, c in ev["fp_counts_earth"].items():
                        earth_counts[t_k] += c
                else:
                    for c in ev.get("fp_candidates", []):
                        for tau in TAU_LIST:
                            t_k = f"{tau:.1f}"
                            if c["score"] >= tau:
                                if c["is_earth"]:
                                    earth_counts[t_k] += 1
                                else:
                                    sky_counts[t_k] += 1

            for r in results_negative:
                ev = r["eval_by_sensor_k"][sensor][k_val]
                if "fp_counts_sky" in ev and ev["fp_counts_sky"]:
                    for t_k, c in ev["fp_counts_sky"].items():
                        sky_counts[t_k] += c
                    for t_k, c in ev["fp_counts_earth"].items():
                        earth_counts[t_k] += c
                else:
                    for c in ev.get("fp_candidates", []):
                        for tau in TAU_LIST:
                            t_k = f"{tau:.1f}"
                            if c["score"] >= tau:
                                if c["is_earth"]:
                                    earth_counts[t_k] += 1
                                else:
                                    sky_counts[t_k] += 1

            sweep_res = {}
            for tau in TAU_LIST:
                tau_key = f"{tau:.1f}"
                n_sky = sky_counts[tau_key]
                n_earth = earth_counts[tau_key]
                n_tot = n_sky + n_earth

                rho_sky = float(n_sky / total_sky_mpx) if total_sky_mpx > 0 else 0.0
                rho_earth = float(n_earth / total_earth_mpx) if total_earth_mpx > 0 else 0.0
                rho_tot = float(n_tot / (total_sky_mpx + total_earth_mpx)) if (total_sky_mpx + total_earth_mpx) > 0 else 0.0

                ub_sky = float(poisson_rate_upper_bound(n_sky, total_sky_mpx)) if total_sky_mpx > 0 else 0.0
                ub_earth = float(poisson_rate_upper_bound(n_earth, total_earth_mpx)) if total_earth_mpx > 0 else 0.0
                ub_tot = float(poisson_rate_upper_bound(n_tot, (total_sky_mpx + total_earth_mpx))) if (total_sky_mpx + total_earth_mpx) > 0 else 0.0

                lambda_obs = rho_sky * T1_RING_SKY_MPX + rho_earth * T1_RING_EARTH_MPX
                lambda_ub = ub_sky * T1_RING_SKY_MPX + ub_earth * T1_RING_EARTH_MPX

                hfar_obs = lambda_obs * FLIGHT_HOURLY_FRAMES
                hfar_ub = lambda_ub * FLIGHT_HOURLY_FRAMES

                sweep_res[tau_key] = {
                    "tau": tau,
                    "fp_count_sky": n_sky,
                    "fp_count_earth": n_earth,
                    "fp_count_total": n_tot,
                    "fp_per_mpx_sky": rho_sky,
                    "fp_per_mpx_earth": rho_earth,
                    "fp_per_mpx_total": rho_tot,
                    "fp_per_mpx_sky_ub95": ub_sky,
                    "fp_per_mpx_earth_ub95": ub_earth,
                    "fp_per_mpx_total_ub95": ub_tot,
                    "lambda_per_camera_frame": lambda_obs,
                    "lambda_per_camera_frame_ub95": lambda_ub,
                    "false_alarms_per_flight_hour": hfar_obs,
                    "false_alarms_per_flight_hour_ub95": hfar_ub,
                }
            fp_sweeps[sensor][k_val] = sweep_res

    # 5. Operating point selection (WP-04 §3)
    # Highest Pd over s_px in [2, 8) px whose FP upper bound is <= 0.0005 per camera-frame (lambda <= 0.0005)
    print("\nEvaluating operating point (k*, tau*)...")
    s_2_8_pos = [r for r in results_positive if 2.0 <= r["s_px"] < 8.0]
    s_2_8_null = [p for p in phantom_results_list if 2.0 <= p["s_px"] < 8.0]

    best_op = None
    best_pd_corr = -1.0
    min_measured_lambda_ub = float("inf")
    min_lambda_op = None

    for k_val in K_VALUES:
        for tau in TAU_LIST:
            tau_str = f"{tau:.1f}"
            deg_fp_stat = fp_sweeps["degraded"][k_val][tau_str]
            lam_ub = deg_fp_stat["lambda_per_camera_frame_ub95"]

            if lam_ub < min_measured_lambda_ub:
                min_measured_lambda_ub = lam_ub
                min_lambda_op = (k_val, tau)

            if lam_ub <= 0.0005:
                stat_28 = compute_stratified_pd_stats(s_2_8_pos, s_2_8_null, "degraded", k_val, tau, rule="primary")
                if stat_28["pd_corrected"] > best_pd_corr:
                    best_pd_corr = stat_28["pd_corrected"]
                    best_op = (k_val, tau)

    if best_op is not None:
        op_k, op_tau = best_op
        op_status = "RESOLVED"
        op_reason = f"Satisfies lambda <= 0.0005 with highest s_px in [2, 8) Pd_corr = {best_pd_corr:.3f}"
        smallest_tau_meeting_budget = op_tau
        pd_at_smallest_tau = best_pd_corr
    else:
        op_k, op_tau = min_lambda_op if min_lambda_op is not None else (3, 100.0)
        op_status = "UNRESOLVED"
        op_reason = (
            f"No (k, tau) combination in the swept range reaches lambda_ub <= 0.0005 per frame: "
            f"minimum measured lambda upper bound ({min_measured_lambda_ub:.2e}/frame at tau=100.0) exceeds target."
        )
        smallest_tau_meeting_budget = None
        pd_at_smallest_tau = None

    closure_k = op_k if op_status == "RESOLVED" else args.closure_k
    closure_tau = op_tau if op_status == "RESOLVED" else args.closure_tau

    print(f"Operating point: k* = {op_k}, tau* = {op_tau:.1f} (Status: {op_status})")
    print(f"  Reason: {op_reason}")
    print(f"Downstream closure evaluation point: k = {closure_k}, tau = {closure_tau:.1f}")

    # 6. Stratified Pd and Abbott correction at operating point
    print("\nComputing stratified P_d and Abbott correction at operating point...")
    strata_results = {}

    def run_strata_breakdown(sensor_state: str, k_curr: int, tau_curr: float) -> dict[str, Any]:
        out = {}
        # Overall
        out["overall"] = compute_stratified_pd_stats(results_positive, phantom_results_list, sensor_state, k_curr, tau_curr, rule="primary")
        out["overall_secondary_eiou"] = compute_stratified_pd_stats(results_positive, phantom_results_list, sensor_state, k_curr, tau_curr, rule="secondary")

        # Class stratum
        classes = sorted(list({r["class_name"] for r in results_positive}))
        out["by_class"] = {}
        for cls in classes:
            c_pos = [r for r in results_positive if r["class_name"] == cls]
            c_null = [p for p in phantom_results_list if p["instance_id"] in {x["instance_id"] for x in c_pos}]
            out["by_class"][cls] = compute_stratified_pd_stats(c_pos, c_null, sensor_state, k_curr, tau_curr)

        # Background stratum
        out["by_background"] = {}
        for bg in ["sky", "earth"]:
            bg_pos = [r for r in results_positive if r["background"] == bg]
            bg_null = [p for p in phantom_results_list if p["background"] == bg]
            out["by_background"][bg] = compute_stratified_pd_stats(bg_pos, bg_null, sensor_state, k_curr, tau_curr)

        # Range bins
        out["by_range"] = {}
        for r0, r1 in RANGE_BINS_KM:
            lbl = f"[{r0:.1f}, {r1:.1f}) km"
            r_pos = [r for r in results_positive if r0 <= r["range_m"] / 1000.0 < r1]
            r_null = [p for p in phantom_results_list if p["instance_id"] in {x["instance_id"] for x in r_pos}]
            out["by_range"][lbl] = compute_stratified_pd_stats(r_pos, r_null, sensor_state, k_curr, tau_curr)

        # S_px bins
        out["by_spx"] = {}
        for s0, s1 in S_PX_BINS:
            lbl = f"[{s0:.0f}, {s1:.0f}) px" if not math.isinf(s1) else f">= {s0:.0f} px"
            s_pos = [r for r in results_positive if s0 <= r["s_px"] < s1]
            s_null = [p for p in phantom_results_list if s0 <= p["s_px"] < s1]
            out["by_spx"][lbl] = compute_stratified_pd_stats(s_pos, s_null, sensor_state, k_curr, tau_curr)

        # A_C bins
        out["by_ac"] = {}
        sig_key = "signal_degraded" if sensor_state == "degraded" else "signal_clean"
        for a0, a1 in AC_BINS:
            lbl = f"[{a0:.1f}, {a1:.1f}) px" if not math.isinf(a1) else f">= {a0:.1f} px"
            a_pos = [r for r in results_positive if a0 <= abs(r[sig_key]["a_c"]) < a1]
            a_null = [p for p in phantom_results_list if p["instance_id"] in {x["instance_id"] for x in a_pos}]
            out["by_ac"][lbl] = compute_stratified_pd_stats(a_pos, a_null, sensor_state, k_curr, tau_curr)

        return out

    strata_results["operating_point_degraded"] = run_strata_breakdown("degraded", op_k, op_tau)
    strata_results["operating_point_clean"] = run_strata_breakdown("clean", op_k, op_tau)

    # 7. Localisation Errors (WP-04 §3)
    print("\nComputing localisation statistics...")
    loc_stats_by_spx = {}
    for s0, s1 in S_PX_BINS:
        lbl = f"[{s0:.0f}, {s1:.0f}) px" if not math.isinf(s1) else f">= {s0:.0f} px"
        s_pos = [r for r in results_positive if s0 <= r["s_px"] < s1]

        # Matched detections
        proj_offsets = []
        mask_errors = []
        du_proj_list = []
        dv_proj_list = []
        du_mask_list = []
        dv_mask_list = []

        for it in s_pos:
            ev = it["eval_by_sensor_k"]["degraded"][op_k]
            if ev["primary_best_score"] >= op_tau and ev["matched_det_u"] is not None:
                d_u, d_v = ev["matched_det_u"], ev["matched_det_v"]
                t_u, t_v = it["target_uv"]
                du_p = d_u - t_u
                dv_p = d_v - t_v
                proj_offsets.append(math.hypot(du_p, dv_p))
                du_proj_list.append(du_p)
                dv_proj_list.append(dv_p)

                if it["visible_centroid"] is not None:
                    m_u, m_v = it["visible_centroid"]
                    du_m = d_u - m_u
                    dv_m = d_v - m_v
                    mask_errors.append(math.hypot(du_m, dv_m))
                    du_mask_list.append(du_m)
                    dv_mask_list.append(dv_m)

        loc_stats_by_spx[lbl] = {
            "n_matched": len(proj_offsets),
            "proj_offset": {
                "mean_px": float(np.mean(proj_offsets)) if proj_offsets else 0.0,
                "median_px": float(np.median(proj_offsets)) if proj_offsets else 0.0,
                "p95_px": float(np.percentile(proj_offsets, 95)) if proj_offsets else 0.0,
                "bias_u_px": float(np.mean(du_proj_list)) if du_proj_list else 0.0,
                "bias_v_px": float(np.mean(dv_proj_list)) if dv_proj_list else 0.0,
                "sigma_px": float(np.std(proj_offsets)) if proj_offsets else 0.0,
            },
            "mask_centroid_error": {
                "n_samples": len(mask_errors),
                "mean_px": float(np.mean(mask_errors)) if mask_errors else 0.0,
                "median_px": float(np.median(mask_errors)) if mask_errors else 0.0,
                "p95_px": float(np.percentile(mask_errors, 95)) if mask_errors else 0.0,
                "bias_u_px": float(np.mean(du_mask_list)) if du_mask_list else 0.0,
                "bias_v_px": float(np.mean(dv_mask_list)) if dv_mask_list else 0.0,
                "sigma_px": float(np.std(mask_errors)) if mask_errors else 0.0,
            },
        }

    # 8. Target Signal & Contrast Summary (WP-04 §4)
    print("\nComputing target signal and degraded SNR summaries...")
    signal_summary = {
        "overall": {
            "a_c_median": float(np.median([r["signal_degraded"]["a_c"] for r in results_positive])) if results_positive else 0.0,
            "mean_contrast_median": float(np.median([r["signal_degraded"]["mean_contrast"] for r in results_positive])) if results_positive else 0.0,
            "snr_degraded_median": float(np.median([r["signal_degraded"]["degraded_snr"] for r in results_positive if r["signal_degraded"]["degraded_snr"] is not None])) if results_positive else 0.0,
        },
        "by_background": {},
        "by_class": {},
    }
    for bg in ["sky", "earth"]:
        b_items = [r for r in results_positive if r["background"] == bg]
        if b_items:
            snr_vals = [r["signal_degraded"]["degraded_snr"] for r in b_items if r["signal_degraded"]["degraded_snr"] is not None]
            signal_summary["by_background"][bg] = {
                "a_c_median": float(np.median([r["signal_degraded"]["a_c"] for r in b_items])),
                "mean_contrast_median": float(np.median([r["signal_degraded"]["mean_contrast"] for r in b_items])),
                "snr_degraded_median": float(np.median(snr_vals)) if snr_vals else 0.0,
            }
    for cls in sorted(list({r["class_name"] for r in results_positive})):
        c_items = [r for r in results_positive if r["class_name"] == cls]
        if c_items:
            snr_vals = [r["signal_degraded"]["degraded_snr"] for r in c_items if r["signal_degraded"]["degraded_snr"] is not None]
            signal_summary["by_class"][cls] = {
                "a_c_median": float(np.median([r["signal_degraded"]["a_c"] for r in c_items])),
                "mean_contrast_median": float(np.median([r["signal_degraded"]["mean_contrast"] for r in c_items])),
                "snr_degraded_median": float(np.median(snr_vals)) if snr_vals else 0.0,
            }

    # 9. Detection Range Fits (R50 / R90) at Closure Point (WP-04 §3, R4)
    print(f"\nFitting detection range curves (R50, R90) at k={closure_k}, tau={closure_tau:.1f}...")
    range_fits = {}
    classes = sorted(list({r["class_name"] for r in results_positive}))
    for cls in classes:
        for bg in ["sky", "earth"]:
            c_bg_items = [r for r in results_positive if r["class_name"] == cls and r["background"] == bg]
            if c_bg_items:
                rng_arr = np.array([r["range_m"] for r in c_bg_items], dtype=np.float64)
                hits_arr = np.array([r["eval_by_sensor_k"]["degraded"][closure_k]["primary_hits_by_tau"].get(f"{closure_tau:.1f}", 0) for r in c_bg_items], dtype=np.float64)  # default-ok: timing benchmark fallback
                enc_arr = [r["encounter_id"] for r in c_bg_items]
                rf = evaluate_range_fit(
                    range_m_arr=rng_arr,
                    hits_arr=hits_arr,
                    cls=cls,
                    bg=bg,
                    n_encounters=len(set(enc_arr)),
                    encounters_arr=enc_arr,
                )
                range_fits[f"{cls}_{bg}"] = rf

    # 10. WP-05 Closure Fit (WP-04 §6): Pd(s_px) = 1 / (1 + exp(-(ln(s_px) - ln(s50)) / w_ln))
    print(f"\nFitting WP-05 log-logistic closure curve Pd(s_px) at k={closure_k}, tau={closure_tau:.1f}...")
    spx_arr = np.array([r["s_px"] for r in results_positive], dtype=np.float64)
    hits_arr = np.array([r["eval_by_sensor_k"]["degraded"][closure_k]["primary_hits_by_tau"].get(f"{closure_tau:.1f}", 0) for r in results_positive], dtype=np.float64)  # default-ok: optional fallback default
    n_enc_pos = len(set(r["encounter_id"] for r in results_positive))

    if closure_tau != op_tau or closure_k != op_k:
        strata_results["closure_point_degraded"] = run_strata_breakdown("degraded", closure_k, closure_tau)

    closure_strata = strata_results.get("closure_point_degraded", strata_results["operating_point_degraded"])
    empirical_pd_by_spx = {}
    if "by_spx" in closure_strata:
        for spx_lbl, st in closure_strata["by_spx"].items():
            empirical_pd_by_spx[spx_lbl] = {
                "n_instances": st["n_instances"],
                "n_encounters": st.get("n_encounters", n_enc_pos),
                "n_detected": st.get("n_detected", int(round(st["raw_pd"] * st["n_instances"]))),
                "raw_pd": st["raw_pd"],
                "raw_pd_ci95": st["raw_pd_ci95"],
            }

    wp05_closure = evaluate_closure_fit(
        spx_arr,
        hits_arr,
        empirical_pd_by_spx=empirical_pd_by_spx,
        op_k=closure_k,
        op_tau=closure_tau,
        run_label=run_label,
        n_encounters=n_enc_pos,
        fit_mode="log_logistic",
    )

    wp05_closure["release_json_sha256"] = release_sha256

    # 11. Assemble final results JSON
    final_output = {
        "status": run_label,
        "gate": "WP-04 G0 Pilot Rehearsal" if is_pilot else "WP-04 G0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root": str(g3a_root),
        "release_json_sha256": release_sha256,
        "manifests_sha256": manifests,
        "dataset_summary": {
            "release_json_sha256": release_sha256,
            "n_encounters": n_enc_pos,
            "n_instances_positive": len(results_positive),
            "n_instances_negative": len(results_negative),
            "n_instances_total": all_tasks_len,
            "n_paired_phantoms": len(phantom_map),
            "n_unpaired_positives": len(unpaired_instances),
        },
        "timing_and_performance": {
            "wall_s": wall_eval_s,
            "cpu_s": total_cpu_ms / 1000.0,
            "workers": workers,
            "total_rois_processed": all_tasks_len,
            "positive_rois": len(results_positive),
            "negative_rois": len(results_negative),
            "ms_per_roi_average": ms_per_roi,
            "full_g3a_projected_cpu_hours": projected_cpu_h,
            "full_g3a_projected_wall_hours": projected_wall_h,
        },
        "operating_point": {
            "k_star": op_k,
            "tau_star": op_tau,
            "status": op_status,
            "reason": op_reason,
            "min_measured_lambda_ub": min_measured_lambda_ub,
            "smallest_tau_meeting_budget": smallest_tau_meeting_budget,
            "pd_at_smallest_tau": pd_at_smallest_tau,
            "closure_k": closure_k,
            "closure_tau": closure_tau,
            "n_encounters": n_enc_pos,
            "n_instances": len(results_positive),
        },
        "search_area_mpx": {
            "sky_mpx": total_sky_mpx,
            "earth_mpx": total_earth_mpx,
            "total_mpx": total_sky_mpx + total_earth_mpx,
        },
        "strata_results": strata_results,
        "localisation_by_spx": loc_stats_by_spx,
        "target_signal_summary": signal_summary,
        "detection_range_fits": range_fits,
        "wp05_closure_fit": wp05_closure,
        "false_positive_sweeps": fp_sweeps,
    }

    json_name = "g3a_pilot_results.json" if is_pilot else "g3a_results.json"
    summary_name = "g3a_pilot_summary.md" if is_pilot else "g3a_summary.md"
    json_path = out_dir / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nWrote results JSON to {json_path}")
    if not is_pilot:
        fp_closure_st = fp_sweeps["degraded"][closure_k][f"{closure_tau:.1f}"]
        meas_lam = fp_closure_st["lambda_per_camera_frame"]
        write_closure_config(
            wp05_closure,
            meas_lam,
            closure_k,
            closure_tau,
            release_sha256=release_sha256,
            fp_per_mpx_earth=float(fp_closure_st["fp_per_mpx_earth"]),
            fp_per_mpx_sky=float(fp_closure_st["fp_per_mpx_sky"]),
        )

    # 12. Write summary markdown
    summary_path = out_dir / summary_name
    summary_title = f"WP-04 G0 Traffic Detection Evaluation Summary ({run_label})" if not is_pilot else f"WP-04 G0 Pilot Rehearsal Summary ({run_label})"
    n_encs = len(set(r["encounter_id"] for r in results_positive))
    wall_ceiling_note = (
        f"Exceeds 4-hour wall ceiling ({projected_wall_h:.2f} h > 4.0 h); class-stratified encounter subsampling applied per GEMINI_G-B9 §P3."
        if projected_wall_h > 4.0
        else "Within the 4-hour wall ceiling."
    )
    summary_md = f"""# {summary_title}

> [!NOTE]
> **Status:** {run_label}.
> Generated on {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}.
> Data root: `{g3a_root}` (Manifests SHA-256 recorded).
> `RELEASE.json` SHA-256: `{release_sha256 or 'N/A'}`.

## 1. Executive Summary & Runtime Benchmarks

- **Dataset:** {n_encs} encounters, {len(pos_instances):,} positive ring ROIs, {len(neg_instances):,} negative ring ROIs ({len(all_tasks):,} total ROIs).
- **Dataset Release SHA-256:** `{release_sha256 or 'N/A'}`.
- **Execution:** CPU only, {workers} parallel workers, completed in {wall_eval_s:.1f} s wall ({total_cpu_ms/1000.0:.1f} s CPU).
- **Speed:** **{ms_per_roi:.1f} ms / ROI** across clean and degraded states ($k \\in \\{{3, 5, 7\\}}$).
- **Projected Full G3a Wall Time (60,000 ROIs):** **{projected_wall_h:.2f} h** on {workers} workers ({projected_cpu_h:.2f} h CPU). {wall_ceiling_note}

## 2. Operating Point & False Positive Density

- **Selected Operating Point:** $k^* = {op_k}$, $\\tau^* = {op_tau:.1f}$
- **Operating Point Status:** `{op_status}`
- **Note:** {op_reason}
- **Search Area Evaluated:** Sky: {total_sky_mpx:.2f} Mpx | Earth: {total_earth_mpx:.2f} Mpx | Total: {total_sky_mpx+total_earth_mpx:.2f} Mpx

### False Positive Density Sweep (Degraded Sensor, $k={op_k}$)

| $\\tau$ | FP Sky (Count / Mpx) | FP Earth (Count / Mpx) | $\\lambda$ / Ring Frame (Obs / 95% UB) | False Alarms / Flight Hour |
|---|:---:|:---:|:---:|:---:|
"""
    for tau in TAU_LIST:
        st = fp_sweeps["degraded"][op_k][f"{tau:.1f}"]
        summary_md += (
            f"| **{tau:.1f}** | {st['fp_count_sky']} ({st['fp_per_mpx_sky']:.4f}/Mpx) | "
            f"{st['fp_count_earth']} ({st['fp_per_mpx_earth']:.4f}/Mpx) | "
            f"{st['lambda_per_camera_frame']:.4e} / **{st['lambda_per_camera_frame_ub95']:.4e}** | "
            f"{st['false_alarms_per_flight_hour']:.1f} / hr |\n"
        )

    summary_md += f"""
## 3. Stratified $P_d$ and Null Control ($p_0$) at Operating Point ($k={op_k}, \\tau={op_tau:.1f}$)

| Stratum | Instances | Raw $P_d$ (95% CP) | Null $p_0$ (95% CP) | Corrected $P_d$ (95% Bootstrap CI) |
|---|---:|:---:|:---:|:---:|
"""
    op_deg_strata = strata_results["operating_point_degraded"]
    ov = op_deg_strata["overall"]
    summary_md += f"| **Overall Primary (Centroid $r$)** | {ov['n_instances']} | {ov['raw_pd']:.3f} `[{ov['raw_pd_ci95'][0]:.3f}, {ov['raw_pd_ci95'][1]:.3f}]` | {ov['p0']:.3f} | **{ov['pd_corrected']:.3f}** `[{ov['pd_corrected_ci95'][0]:.3f}, {ov['pd_corrected_ci95'][1]:.3f}]` |\n"
    ov_sec = op_deg_strata["overall_secondary_eiou"]
    summary_md += f"| **Overall Secondary (AOT eIoU)** | {ov_sec['n_instances']} | {ov_sec['raw_pd']:.3f} | {ov_sec['p0']:.3f} | **{ov_sec['pd_corrected']:.3f}** |\n"

    for bg, st in op_deg_strata["by_background"].items():
        summary_md += f"| Background: `{bg}` | {st['n_instances']} | {st['raw_pd']:.3f} | {st['p0']:.3f} | **{st['pd_corrected']:.3f}** `[{st['pd_corrected_ci95'][0]:.3f}, {st['pd_corrected_ci95'][1]:.3f}]` |\n"

    for spx_lbl, st in op_deg_strata["by_spx"].items():
        summary_md += f"| $s_{{px}}$: `{spx_lbl}` | {st['n_instances']} | {st['raw_pd']:.3f} | {st['p0']:.3f} | **{st['pd_corrected']:.3f}** `[{st['pd_corrected_ci95'][0]:.3f}, {st['pd_corrected_ci95'][1]:.3f}]` |\n"

    summary_md += f"""
## 4. Localisation Statistics

| $s_{{px}}$ Bin | Matched Count | Centroid vs Projected Center (Mean / Med / p95 / $\\sigma_{{px}}$) | Centroid vs Visible Mask (Mean / $\\sigma_{{px}}$) |
|---|---:|:---:|:---:|
"""
    for spx_lbl, loc in loc_stats_by_spx.items():
        p_off = loc["proj_offset"]
        m_err = loc["mask_centroid_error"]
        summary_md += (
            f"| `{spx_lbl}` | {loc['n_matched']} | "
            f"{p_off['mean_px']:.2f} px / {p_off['median_px']:.2f} px / {p_off['p95_px']:.2f} px / **{p_off['sigma_px']:.2f} px** | "
            f"{m_err['mean_px']:.2f} px / **{m_err['sigma_px']:.2f} px** |\n"
        )

    if wp05_closure.get("status") == "FITTED":
        closure_md_body = f"""- Status: **`FITTED`**
- $s_{{50}}$: **{wp05_closure['s50_px']:.3f} px** (95% CI: `[{wp05_closure['ci95_s50'][0]:.3f}, {wp05_closure['ci95_s50'][1]:.3f}]` px, width: {wp05_closure['ci95_width_px']:.2f} px)
- $w$: **{wp05_closure['w_px']:.3f} px**
- Operating Point: $k^* = {op_k}, \\tau^* = {op_tau:.1f}$
- Provenance: `{wp05_closure['provenance']}`
"""
    else:
        closure_md_body = f"""- Status: **`NOT_FITTED`** (Logistic MLE non-converged or 95% CI width > 2.0 px; constant fallback eliminated per R1)
- $s_{{50}}$: **NOT_FITTED** (None)
- $w$: **NOT_FITTED** (None)
- Operating Point: $k^* = {op_k}, \\tau^* = {op_tau:.1f}$
- Provenance: `{wp05_closure['provenance']}`
- Closure Surrogate Policy: WP-05 tracking closure executed with measured $\\lambda$ and baseline $s_{{50}}=3.0, w=0.7$ labeled `WP05 §1 baseline (not measured)`.
"""
    closure_md_body += f"""
### Empirical $P_d$ by Target Size ($s_{{px}}$) with Clopper-Pearson 95% CIs

| $s_{{px}}$ Bin | Instances | Detected | Empirical $P_d$ | 95% Clopper-Pearson CI |
|---|---:|---:|:---:|:---:|
"""
    for spx_lbl, st in wp05_closure.get("empirical_pd_by_spx", {}).items():
        closure_md_body += f"| `{spx_lbl}` | {st['n_instances']} | {st['n_detected']} | {st['raw_pd']:.3f} | `[{st['raw_pd_ci95'][0]:.3f}, {st['raw_pd_ci95'][1]:.3f}]` |\n"

    summary_md += f"""
## 5. Closure Fit Parameters (WP-05 Surrogate Input)

{closure_md_body}

## 6. Detection Range Curves ($R_{{50}}, R_{{90}}$) and Identifiability (WP-04 §3, R4)

| Class | Background | Encounters | Instances | Status | $R_{{50}}$ (95% CI) | $R_{{90}}$ | Factor Span |
|---|:---:|---:|---:|:---:|:---:|:---:|:---:|
"""
    for fit_key, rf in range_fits.items():
        if rf["status"] == "IDENTIFIABLE":
            r50_str = f"{rf['r50_m']:.1f} m `[{rf['r50_ci95'][0]:.1f}, {rf['r50_ci95'][1]:.1f}]`"
            r90_str = f"{rf['r90_m']:.1f} m `[{rf['r90_ci95'][0]:.1f}, {rf['r90_ci95'][1]:.1f}]`"
            span_str = f"**{rf['r50_ci_factor']:.2f}x**"
            status_str = "**`IDENTIFIABLE`**"
        else:
            r50_str = "`NOT_IDENTIFIABLE`"
            r90_str = "`NOT_IDENTIFIABLE`"
            span_str = f"{rf['r50_ci_factor']:.2f}x" if rf.get("r50_ci_factor") is not None else "N/A"
            status_str = "`NOT_IDENTIFIABLE`"
        summary_md += f"| `{rf['class']}` | `{rf['background']}` | {rf['n_encounters']} | {rf['n_instances']} | {status_str} | {r50_str} | {r90_str} | {span_str} |\n"

    summary_md += "\n### Empirical $P_d$ by Range Bin with Clopper-Pearson 95% CIs\n\n"
    for fit_key, rf in range_fits.items():
        summary_md += f"#### Stratum: `{rf['class']}` ({rf['background']})\n\n"
        summary_md += "| Range Bin | Encounters | Instances | Detected | Empirical $P_d$ | 95% Clopper-Pearson CI |\n"
        summary_md += "|---|---:|---:|---:|:---:|:---:|\n"
        for r_bin, st in rf.get("empirical_pd_by_range", {}).items():
            summary_md += f"| `{r_bin}` | {st['n_encounters']} | {st['n_instances']} | {st['n_detected']} | {st['raw_pd']:.3f} | `[{st['raw_pd_ci95'][0]:.3f}, {st['raw_pd_ci95'][1]:.3f}]` |\n"
        summary_md += "\n"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Wrote summary Markdown to {summary_path}")


if __name__ == "__main__":
    main()
