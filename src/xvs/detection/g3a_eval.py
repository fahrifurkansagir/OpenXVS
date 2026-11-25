"""Core evaluation engine for G3a Traffic Detection (WP-04 §3–§6, Decision D23).

Implements:
1. Primary hit rule: tight centroid distance r = max(1.5 px, 0.5 * s_px) around projected center.
2. Secondary hit rule: AOT eIoU > 0.2 against analytic box.
3. Natural data null control: phantom targets transferred to negative ROIs of same encounter
   with matching background, computing chance hit rate p0 and Abbott-corrected Pd.
4. False positives on negative ROIs and positive ROIs outside analytic box + 16 px,
   split by sky/earth using WP-01 WGS84 horizon function, with Poisson 95% upper bounds,
   lambda per full ring frame (5,013,504 px), and false detections per flight hour.
5. Strata evaluation across class, background, range bins, s_px bins, A_C bins,
   with Clopper-Pearson and encounter-cluster bootstrap confidence intervals.
6. Localization error: sigma_px against visible-mask centroid and offset to projected center.
7. Target signal A_C, mean contrast C, and degraded SNR summaries (WP-04 §4).
8. Operating point (k*, tau*) selection and detection range fits (R50, R90).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image

from xvs.metrics.aot import (
    compute_eiou,
    clopper_pearson_interval,
    poisson_rate_upper_bound,
)
from xvs.detection.d0 import detect_d0, load_d0_config
from xvs.detection.contrast import estimate_target_signal
from xvs.detection.fit import fit_detection_range_mle, fit_logistic_mle
from xvs.sensor.srgb import srgb_eotf, linear_rgb_to_monochrome
from xvs.sensor.chain import SensorConfig
from xvs.sensor.psf import apply_psf
from xvs.sensor.noise import apply_sensor_noise, generate_fpn_maps
from xvs.traffic.ring import TrafficRing
from xvs.sim.horizon import is_earth_at_pixel, is_earth_at_pixels

TAU_LIST = [4.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0]
K_VALUES = [3, 5, 7]
SENSOR_MODES = ["clean", "degraded"]

RANGE_BINS_KM = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 6.0),
    (6.0, 10.0),
]

S_PX_BINS = [
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (8.0, 16.0),
    (16.0, float("inf")),
]

AC_BINS = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 50.0),
    (50.0, float("inf")),
]

# T1 ownship cruise ring camera mean areas (WP-04 §3)
T1_RING_SKY_MPX = 2.6876
T1_RING_EARTH_MPX = 2.3259
T1_RING_TOTAL_MPX = 5.013504
FLIGHT_HOURLY_FRAMES = 288000.0  # 8 cameras @ 10 Hz * 3600 s


def primary_matching_radius(s_px: float) -> float:
    """Primary hit rule radius: r = max(1.5 px, 0.5 * s_px) around projected center."""
    return max(1.5, 0.5 * float(s_px))


def get_range_bin_str(range_m: float) -> str:
    r_km = range_m / 1000.0
    for r0, r1 in RANGE_BINS_KM:
        if r0 <= r_km < r1:
            return f"[{r0:.1f}, {r1:.1f}) km"
    return ">= 10.0 km"


def get_spx_bin_str(s_px: float) -> str:
    for s0, s1 in S_PX_BINS:
        if s0 <= s_px < s1:
            if math.isinf(s1):
                return f">= {s0:.0f} px"
            return f"[{s0:.0f}, {s1:.0f}) px"
    return "unknown"


def get_ac_bin_str(ac: float) -> str:
    abs_ac = abs(ac)
    for a0, a1 in AC_BINS:
        if a0 <= abs_ac < a1:
            if math.isinf(a1):
                return f">= {a0:.1f} px"
            return f"[{a0:.1f}, {a1:.1f}) px"
    return "unknown"


def compute_roi_sky_earth_area(
    p_cam_ecef: np.ndarray | Sequence[float],
    r_ecef_cam: np.ndarray | Sequence[Sequence[float]],
    cam_model: Any,
    u0: float,
    v0: float,
    width: int = 1024,
    height: int = 1024,
    exclusion_box: list[float] | None = None,
) -> tuple[float, float]:
    """Compute sky and earth search area (in Mpx) inside a 1024x1024 ROI.

    Uses WP-01 WGS84 horizon bisection across columns.
    If exclusion_box is provided, subtracts the excluded area from the appropriate background.
    """
    p_cam = np.asarray(p_cam_ecef, dtype=np.float64)
    r_cam = np.asarray(r_ecef_cam, dtype=np.float64)

    u_samples = np.linspace(u0, u0 + width, 17)
    v_horiz = []
    for u in u_samples:
        low, high = float(v0), float(v0 + height)
        is_low_e = is_earth_at_pixel(p_cam, r_cam, cam_model, u, low)
        is_high_e = is_earth_at_pixel(p_cam, r_cam, cam_model, u, high)
        if is_low_e and is_high_e:
            v_horiz.append(low)
        elif not is_low_e and not is_high_e:
            v_horiz.append(high)
        elif not is_low_e and is_high_e:
            for _ in range(16):
                mid = (low + high) / 2.0
                if is_earth_at_pixel(p_cam, r_cam, cam_model, u, mid):
                    high = mid
                else:
                    low = mid
            v_horiz.append((low + high) / 2.0)
        else:
            for _ in range(16):
                mid = (low + high) / 2.0
                if is_earth_at_pixel(p_cam, r_cam, cam_model, u, mid):
                    low = mid
                else:
                    high = mid
            v_horiz.append((low + high) / 2.0)

    # Check top sample earth classification to determine whether sky is above or below horizon
    is_top_earth = is_earth_at_pixel(p_cam, r_cam, cam_model, u_samples[len(u_samples) // 2], float(v0))
    if is_top_earth:
        sky_h = np.clip(float(v0 + height) - np.array(v_horiz), 0.0, float(height))
    else:
        sky_h = np.clip(np.array(v_horiz) - v0, 0.0, float(height))
    sky_px = float(np.trapezoid(sky_h, u_samples))
    total_px = float(width * height)
    earth_px = total_px - sky_px

    if exclusion_box is not None:
        bx0 = max(u0, min(u0 + width, exclusion_box[0]))
        by0 = max(v0, min(v0 + height, exclusion_box[1]))
        bx1 = max(u0, min(u0 + width, exclusion_box[2]))
        by1 = max(v0, min(v0 + height, exclusion_box[3]))
        ex_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        cx = (bx0 + bx1) / 2.0
        cy = (by0 + by1) / 2.0
        is_ex_earth = is_earth_at_pixel(p_cam, r_cam, cam_model, cx, cy)
        if is_ex_earth:
            earth_px = max(0.0, earth_px - ex_area)
        else:
            sky_px = max(0.0, sky_px - ex_area)

    return sky_px / 1.0e6, earth_px / 1.0e6


def pair_phantom_targets(
    pos_instances: list[dict[str, Any]],
    neg_instances: list[dict[str, Any]],
    ring_model: Any,
    return_unpaired: bool = False,
) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], list[str]]:
    """Pair each positive instance with a negative frame of the same encounter with matching background.

    Returns dict mapping pos instance_id -> phantom specification dict.
    If return_unpaired is True, returns (phantom_pairs, unpaired_instance_ids).
    Never pairs across backgrounds (G-B14 W6 / R-GB2-10).
    """
    neg_by_enc: dict[str, list[dict[str, Any]]] = {}
    for neg in neg_instances:
        neg_by_enc.setdefault(neg["encounter_id"], []).append(neg)

    pos_by_enc: dict[str, list[dict[str, Any]]] = {}
    for pos in pos_instances:
        pos_by_enc.setdefault(pos["encounter_id"], []).append(pos)

    phantom_pairs: dict[str, dict[str, Any]] = {}
    unpaired_instances: list[str] = []

    for enc, p_list in pos_by_enc.items():
        n_list = neg_by_enc.get(enc, [])
        if not n_list:
            for pos in p_list:
                unpaired_instances.append(pos["instance_id"])
            continue
        n_neg = len(n_list)

        for p_idx, pos in enumerate(p_list):
            target_bg = pos.get("background", "sky")
            u_roi = pos["u"] - pos["roi_u0"]
            v_roi = pos["v"] - pos["roi_v0"]
            s_px = pos["s_px"]
            w_b = pos["box_analytic"][2] - pos["box_analytic"][0]
            h_b = pos["box_analytic"][3] - pos["box_analytic"][1]

            # Find matching negative in same encounter
            chosen_neg = None
            for offset in range(n_neg):
                cand = n_list[(p_idx + offset) % n_neg]
                p_cam = cand["p_cam_ecef"]
                r_cam = cand["R_cam_ecef"]
                u_full = cand["roi_u0"] + u_roi
                v_full = cand["roi_v0"] + v_roi
                is_e = is_earth_at_pixel(p_cam, r_cam, ring_model, u_full, v_full)
                cand_bg = "earth" if is_e else "sky"
                if cand_bg == target_bg:
                    chosen_neg = cand
                    break

            if chosen_neg is None:
                # Never pair across backgrounds (R-GB2-10 / G-B14 W6)
                unpaired_instances.append(pos["instance_id"])
                continue

            u_phantom_full = chosen_neg["roi_u0"] + u_roi
            v_phantom_full = chosen_neg["roi_v0"] + v_roi
            box_phantom = [
                u_phantom_full - w_b / 2.0,
                v_phantom_full - h_b / 2.0,
                u_phantom_full + w_b / 2.0,
                v_phantom_full + h_b / 2.0,
            ]

            phantom_pairs[pos["instance_id"]] = {
                "instance_id": pos["instance_id"],
                "encounter_id": enc,
                "negative_id": chosen_neg["negative_id"],
                "background": target_bg,
                "s_px": s_px,
                "u_full": u_phantom_full,
                "v_full": v_phantom_full,
                "box_phantom": box_phantom,
                "matching_radius": primary_matching_radius(s_px),
            }

    if return_unpaired:
        return phantom_pairs, unpaired_instances
    return phantom_pairs


def evaluate_hits_and_fps_for_roi(
    dets: list[dict[str, Any]],
    target_uv: tuple[float, float] | None,
    box_analytic: list[float] | None,
    s_px: float | None,
    p_cam_ecef: np.ndarray,
    r_ecef_cam: np.ndarray,
    cam_model: Any,
    skip_fps: bool = False,
) -> dict[str, Any]:
    """Evaluate detections in an ROI for target hit rules and candidate false positives."""
    primary_best = -float("inf")
    secondary_best = -float("inf")
    matched_det_u = None
    matched_det_v = None

    if target_uv is not None and s_px is not None:
        r = primary_matching_radius(s_px)
        t_u, t_v = target_uv
        for d in dets:
            d_u, d_v, sc = d["u"], d["v"], d["score"]
            dist = math.hypot(d_u - t_u, d_v - t_v)
            if dist <= r and sc > primary_best:
                primary_best = sc
                matched_det_u = d_u
                matched_det_v = d_v

            if box_analytic is not None and dist <= max(32.0, 3.0 * s_px) and compute_eiou(box_analytic, d["box_edges"]) > 0.2:
                if sc > secondary_best:
                    secondary_best = sc

    # Candidate false positives
    fp_candidates = []
    fp_counts_sky = {f"{tau:.1f}": 0 for tau in TAU_LIST}
    fp_counts_earth = {f"{tau:.1f}": 0 for tau in TAU_LIST}

    if not skip_fps:
        outside_dets = []
        for d in dets:
            d_u, d_v, sc = d["u"], d["v"], d["score"]
            is_outside = True
            if box_analytic is not None:
                # Exclude box_analytic dilated by 16 px
                if (box_analytic[0] - 16.0 <= d_u <= box_analytic[2] + 16.0) and (
                    box_analytic[1] - 16.0 <= d_v <= box_analytic[3] + 16.0
                ):
                    is_outside = False
            if is_outside:
                outside_dets.append((d_u, d_v, float(sc)))

        if outside_dets:
            u_arr = np.array([x[0] for x in outside_dets], dtype=np.float64)
            v_arr = np.array([x[1] for x in outside_dets], dtype=np.float64)
            is_e_arr = is_earth_at_pixels(p_cam_ecef, r_ecef_cam, cam_model, u_arr, v_arr)
            scores_arr = np.array([x[2] for x in outside_dets], dtype=np.float64)

            for tau in TAU_LIST:
                tau_key = f"{tau:.1f}"
                mask_ge = (scores_arr >= tau)
                fp_counts_earth[tau_key] = int(np.sum(mask_ge & is_e_arr))
                fp_counts_sky[tau_key] = int(np.sum(mask_ge & ~is_e_arr))

            for i in range(len(outside_dets)):
                fp_candidates.append({"score": float(scores_arr[i]), "is_earth": bool(is_e_arr[i])})

    return {
        "primary_best_score": primary_best,
        "secondary_best_score": secondary_best,
        "matched_det_u": matched_det_u,
        "matched_det_v": matched_det_v,
        "fp_candidates": fp_candidates,
        "fp_counts_sky": fp_counts_sky,
        "fp_counts_earth": fp_counts_earth,
    }


def compute_stratified_pd_stats(
    pos_items: list[dict[str, Any]],
    phantom_items: list[dict[str, Any]],
    sensor: str,
    k_val: int,
    tau: float,
    rule: str = "primary",
) -> dict[str, Any]:
    """Compute stratified P_d, null rate p0, and Abbott-corrected P_d with bootstrap CIs."""
    score_key = f"{rule}_best_score"

    def get_hits(items: list[dict[str, Any]]) -> list[int]:
        res = []
        tau_str = f"{tau:.1f}"
        for it in items:
            ev = it["eval_by_sensor_k"][sensor][k_val]
            hits_by_tau = ev.get(f"{rule}_hits_by_tau")
            if hits_by_tau is not None and tau_str in hits_by_tau:
                res.append(hits_by_tau[tau_str])
            elif "hits_by_tau" in ev and tau_str in ev["hits_by_tau"]:
                res.append(ev["hits_by_tau"][tau_str])
            else:
                sc = ev.get(score_key, -float("inf"))
                res.append(1 if sc >= tau else 0)
        return res

    pos_hits = get_hits(pos_items)
    n_pos = len(pos_hits)
    k_pos = sum(pos_hits)
    raw_pd = float(k_pos / n_pos) if n_pos > 0 else 0.0
    cp_ci = clopper_pearson_interval(k_pos, n_pos) if n_pos > 0 else (0.0, 0.0)

    # Phantom hits
    null_hits = get_hits(phantom_items) if phantom_items else []
    n_null = len(null_hits)
    k_null = sum(null_hits)
    p0 = float(k_null / n_null) if n_null > 0 else 0.0
    p0_ci = clopper_pearson_interval(k_null, n_null) if n_null > 0 else (0.0, 0.0)

    # Abbott correction: (P_d - p0) / (1 - p0)
    if p0 < 1.0:
        pd_corr = max(0.0, min(1.0, (raw_pd - p0) / (1.0 - p0)))
    else:
        pd_corr = 0.0

    # Encounter-cluster bootstrap for corrected Pd
    enc_keys = sorted(list({it["encounter_id"] for it in pos_items}))
    n_enc = len(enc_keys)
    if n_enc > 1 and n_pos > 0:
        pos_by_enc = {enc: [h for it, h in zip(pos_items, pos_hits) if it["encounter_id"] == enc] for enc in enc_keys}
        null_by_enc = {enc: [h for it, h in zip(phantom_items, null_hits) if it["encounter_id"] == enc] for enc in enc_keys}

        rng = np.random.RandomState(20260926)
        boot_estimates = []
        for _ in range(1000):
            resamp = rng.choice(enc_keys, size=n_enc, replace=True)
            b_pos = [h for enc in resamp for h in pos_by_enc.get(enc, [])]
            b_null = [h for enc in resamp for h in null_by_enc.get(enc, [])]
            if not b_pos:
                continue
            b_pd = float(np.mean(b_pos))
            b_p0 = float(np.mean(b_null)) if b_null else 0.0
            if b_p0 < 1.0:
                b_corr = max(0.0, min(1.0, (b_pd - b_p0) / (1.0 - b_p0)))
            else:
                b_corr = 0.0
            boot_estimates.append(b_corr)

        ci_low = float(np.percentile(boot_estimates, 2.5)) if boot_estimates else pd_corr
        ci_high = float(np.percentile(boot_estimates, 97.5)) if boot_estimates else pd_corr
    else:
        ci_low, ci_high = pd_corr, pd_corr

    return {
        "n_instances": n_pos,
        "n_encounters": n_enc,
        "n_detected": k_pos,
        "raw_pd": raw_pd,
        "raw_pd_ci95": [float(cp_ci[0]), float(cp_ci[1])],
        "p0": p0,
        "p0_ci95": [float(p0_ci[0]), float(p0_ci[1])],
        "pd_corrected": pd_corr,
        "pd_corrected_ci95": [ci_low, ci_high],
    }


def compute_fp_rate_sweep(
    all_fp_candidates: list[dict[str, Any]],
    sky_area_mpx: float,
    earth_area_mpx: float,
    tau_list: Sequence[float] = TAU_LIST,
) -> dict[str, dict[str, Any]]:
    """Compute false positive density per Mpx and Poisson 95% upper bounds across tau."""
    total_area_mpx = sky_area_mpx + earth_area_mpx

    sweep = {}
    for tau in tau_list:
        tau_key = f"{tau:.1f}"
        sky_fps = [c for c in all_fp_candidates if not c["is_earth"] and c["score"] >= tau]
        earth_fps = [c for c in all_fp_candidates if c["is_earth"] and c["score"] >= tau]

        n_sky = len(sky_fps)
        n_earth = len(earth_fps)
        n_tot = n_sky + n_earth

        rho_sky = float(n_sky / sky_area_mpx) if sky_area_mpx > 0 else 0.0
        rho_earth = float(n_earth / earth_area_mpx) if earth_area_mpx > 0 else 0.0
        rho_tot = float(n_tot / total_area_mpx) if total_area_mpx > 0 else 0.0

        ub_sky = float(poisson_rate_upper_bound(n_sky, sky_area_mpx)) if sky_area_mpx > 0 else 0.0
        ub_earth = float(poisson_rate_upper_bound(n_earth, earth_area_mpx)) if earth_area_mpx > 0 else 0.0
        ub_tot = float(poisson_rate_upper_bound(n_tot, total_area_mpx)) if total_area_mpx > 0 else 0.0

        # Converted metrics for full ring camera frame at T1 cruise
        lambda_frame = T1_RING_SKY_MPX * rho_sky + T1_RING_EARTH_MPX * rho_earth
        lambda_ub = T1_RING_SKY_MPX * ub_sky + T1_RING_EARTH_MPX * ub_earth
        hourly_rate = lambda_frame * FLIGHT_HOURLY_FRAMES
        hourly_rate_ub = lambda_ub * FLIGHT_HOURLY_FRAMES

        sweep[tau_key] = {
            "tau": float(tau),
            "fp_count_sky": n_sky,
            "fp_count_earth": n_earth,
            "fp_count_total": n_tot,
            "fp_per_mpx_sky": rho_sky,
            "fp_per_mpx_earth": rho_earth,
            "fp_per_mpx_total": rho_tot,
            "fp_per_mpx_sky_ub95": ub_sky,
            "fp_per_mpx_earth_ub95": ub_earth,
            "fp_per_mpx_total_ub95": ub_tot,
            "lambda_per_camera_frame": lambda_frame,
            "lambda_per_camera_frame_ub95": lambda_ub,
            "false_alarms_per_flight_hour": hourly_rate,
            "false_alarms_per_flight_hour_ub95": hourly_rate_ub,
        }

    return sweep
