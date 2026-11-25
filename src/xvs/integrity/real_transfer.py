"""WP-06 / WP-07 Synthetic-to-Real Horizon & Conformal Transfer Study (Batch B13 — Charter Hypothesis H3).

Evaluates classical G0 and G1 horizon/attitude estimators and tests Charter §10 Hypothesis H3
on the 29 real annotated aerial drone and maritime video sequences in `icehorizon_data` (`CC-BY 4.0`).

Strict guarantees:
1. Evaluation-only real dataset usage (`docs/xvs/spec/data_allowlist.json` — `IceHorizon`, `CC-BY 4.0`),
   zero learned weights trained on `IceHorizon`.
2. Strict sequence-disjoint conformal split (`real_cal` vs held-out `real_test` partitioned by video ID;
   zero frames from the same video sequence ever appear in both splits).
3. Direct comparison of Zero-Shot Synthetic-Calibrated Conformal Protection Levels (`output/xvs/wp07/g1/results.json`)
   against Sequence-Disjoint Real-Recalibrated Split & Mondrian Conformal Protection Levels at
   alpha in {0.10, 0.05, 0.01}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy import stats

from xvs.geo import rotations, wgs84
from xvs.horizon_att.edges import (
    detect_camera_edges,
    detect_camera_edges_multiscale,
    edges_to_ray_batch,
    extract_feature_map,
)
from xvs.horizon_att.pipeline import fit_pipeline_attitude
from xvs.integrity.conformal import compute_conformal_quantile
from xvs.integrity.g1_extensions import (
    compute_horizon_strip_ood_features,
    fit_pipeline_attitude_g1,
    solve_attitude_gn_g1,
)
from xvs.rig.rig import Rig


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> dict[str, float]:
    """Compute exact Clopper-Pearson binomial confidence interval."""
    if n <= 0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": 1.0}
    alpha = 1.0 - confidence
    ci_low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    ci_high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return {"rate": float(k / n), "ci_low": ci_low, "ci_high": ci_high}


def verify_icehorizon_allowlist(allowlist_path: str | Path, dataset_id: str = "IceHorizon") -> dict[str, str]:
    """Verify that `dataset_id` is listed in `docs/xvs/spec/data_allowlist.json` under `research_only_datasets`."""
    path = Path(allowlist_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    research_list = data.get("research_only_datasets", [])  # default-ok: optional fallback default
    for entry in research_list:
        if entry.get("id") == dataset_id:
            return {
                "id": str(entry["id"]),
                "rule": str(entry.get("rule", "")),
                "allowlist_schema": str(data.get("schema", "")),
                "status": "ALLOWLIST_VERIFIED_EVALUATION_ONLY",
            }
    raise ValueError(f"Dataset {dataset_id!r} not found in {allowlist_path}")


def load_icehorizon_sequence_annotations(xml_path: str | Path) -> list[dict[str, Any]]:
    """Parse CVAT XML horizon polyline annotations for a single video sequence."""
    tree = ET.parse(Path(xml_path))
    root = tree.getroot()

    frames_by_id: dict[int, list[tuple[float, float]]] = {}

    for img_el in root.findall("image"):
        fid_str = img_el.get("id")
        if fid_str is None:
            continue
        fid = int(fid_str)
        for pl in img_el.findall("polyline"):
            lbl = pl.get("label", "")
            if not lbl.startswith("horizon"):
                continue
            pts_str = pl.get("points", "")
            pts = _parse_points_str(pts_str)
            if len(pts) >= 2:
                frames_by_id[fid] = pts

    for tr_el in root.findall("track"):
        lbl = tr_el.get("label", "")
        if not lbl.startswith("horizon"):
            continue
        for pl in tr_el.findall("polyline"):
            if pl.get("outside", "0") != "0":  # default-ok: optional fallback default
                continue
            fid_str = pl.get("frame")
            if fid_str is None:
                continue
            fid = int(fid_str)
            pts = _parse_points_str(pl.get("points", ""))
            if len(pts) >= 2:
                frames_by_id[fid] = pts

    out: list[dict[str, Any]] = []
    for fid in sorted(frames_by_id.keys()):
        out.append({"frame_id": fid, "points": frames_by_id[fid]})
    return out


def _parse_points_str(pts_str: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for item in pts_str.strip().split(";"):
        item_s = item.strip()
        if not item_s or "," not in item_s:
            continue
        x_s, y_s = item_s.split(",", 1)
        pts.append((float(x_s), float(y_s)))
    pts.sort(key=lambda p: p[0])
    return pts


def sample_stratified_frames(
    annotations: Sequence[dict[str, Any]],
    frames_per_sequence: int = 20,
    max_valid_frame_id: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministically select up to `frames_per_sequence` evenly spaced annotated frames."""
    valid = [
        a for a in annotations
        if max_valid_frame_id is None or int(a["frame_id"]) < max_valid_frame_id
    ]
    if not valid:
        return []
    n = len(valid)
    if n <= frames_per_sequence:
        return list(valid)
    indices = np.linspace(0, n - 1, frames_per_sequence, dtype=int)
    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for idx in indices:
        ii = int(idx)
        if ii not in seen:
            seen.add(ii)
            selected.append(valid[ii])
    return selected


def fit_line_tls(us: np.ndarray, vs: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """Fit image-plane horizon line v = m * u + c and normal parameters (tilt_deg, v_mid_px, rho_px)."""
    u_arr = np.asarray(us, dtype=np.float64)
    v_arr = np.asarray(vs, dtype=np.float64)
    if len(u_arr) < 2 or float(np.max(u_arr) - np.min(u_arr)) < 1e-3:
        v_med = float(np.median(v_arr)) if len(v_arr) > 0 else 0.0
        return {"m": 0.0, "c": v_med, "tilt_deg": 0.0}

    if weights is not None and len(weights) == len(u_arr):
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 1e-9)
        w = w / np.sum(w)
    else:
        w = np.full(len(u_arr), 1.0 / len(u_arr), dtype=np.float64)

    u_mean = float(np.sum(w * u_arr))
    v_mean = float(np.sum(w * v_arr))
    du = u_arr - u_mean
    dv = v_arr - v_mean
    s_uu = float(np.sum(w * du * du))
    s_uv = float(np.sum(w * du * dv))
    if s_uu < 1e-9:
        m = 0.0
    else:
        m = s_uv / s_uu
    c = v_mean - m * u_mean
    tilt_deg = math.degrees(math.atan(m))
    return {"m": float(m), "c": float(c), "tilt_deg": float(tilt_deg)}


def fit_polyline_horizon_gt(
    points: Sequence[tuple[float, float]],
    width: int,
    height: int,
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_nom_m: float,
    yaw_rad: float = 0.0,
) -> dict[str, float]:
    """Convert a human CVAT horizon polyline into canonical image-plane and 3D attitude ground truth."""
    pts = list(points)
    us_dense: list[float] = []
    vs_dense: list[float] = []
    ws_dense: list[float] = []

    for i in range(len(pts) - 1):
        u0, v0 = pts[i]
        u1, v1 = pts[i + 1]
        seg_len = math.hypot(u1 - u0, v1 - v0)
        n_steps = max(2, int(round(seg_len / 16.0)))
        for s in np.linspace(0.0, 1.0, n_steps, endpoint=(i == len(pts) - 2)):
            u_val = float(u0 + s * (u1 - u0))
            v_val = float(v0 + s * (v1 - v0))
            if 4.0 <= u_val <= width - 4.0 and 0.0 <= v_val <= height - 1.0:
                us_dense.append(u_val)
                vs_dense.append(v_val)
                ws_dense.append(max(1.0, seg_len / float(n_steps)))

    if len(us_dense) < 4:
        for u_p, v_p in pts:
            us_dense.append(float(u_p))
            vs_dense.append(float(v_p))
            ws_dense.append(1.0)

    line_gt = fit_line_tls(
        np.asarray(us_dense, dtype=np.float64),
        np.asarray(vs_dense, dtype=np.float64),
        np.asarray(ws_dense, dtype=np.float64),
    )
    m_gt = line_gt["m"]
    c_gt = line_gt["c"]
    tilt_gt_deg = line_gt["tilt_deg"]
    u_mid = 0.5 * float(width)
    v_mid = m_gt * u_mid + c_gt
    rho_gt_px = (v_mid - 0.5 * float(height)) / math.sqrt(1.0 + m_gt * m_gt)

    gt_edges = [
        {"u": float(u), "v": float(v), "gradient": 1.0}
        for u, v in zip(us_dense, vs_dense)
    ]
    gt_batch = edges_to_ray_batch({"F": gt_edges}, rig)
    gt_fit = solve_attitude_gn_g1(
        lat_deg,
        lon_deg,
        h_nom_m,
        yaw_rad,
        0.0,
        0.0,
        gt_batch,
        max_iter=60,
        tol_rad=1e-7,
        huber_delta_rad=math.radians(0.50),
    )

    return {
        "m_gt": float(m_gt),
        "c_gt": float(c_gt),
        "tilt_gt_deg": float(tilt_gt_deg),
        "v_mid_gt_px": float(v_mid),
        "rho_gt_px": float(rho_gt_px),
        "pitch_gt_deg": float(gt_fit["pitch_deg"]),
        "roll_gt_deg": float(gt_fit["roll_deg"]),
        "pitch_gt_rad": float(gt_fit["pitch_rad"]),
        "roll_gt_rad": float(gt_fit["roll_rad"]),
    }


def build_coarse_ins_prior_pose(
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_nom_m: float,
    yaw_rad: float,
    pitch_gt_rad: float,
    roll_gt_rad: float,
    rng: np.random.Generator,
    sigma_prior_deg: float = 0.75,
) -> dict[str, Any]:
    """Construct a deterministic coarse navigation/AHRS prior pose (`prior_pose`) for horizon search band."""
    dp_rad = math.radians(float(rng.normal(0.0, sigma_prior_deg)))
    dr_rad = math.radians(float(rng.normal(0.0, sigma_prior_deg)))
    p_prior = pitch_gt_rad + dp_rad
    r_prior = roll_gt_rad + dr_rad

    p_ecef = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_nom_m)
    r_en = wgs84.r_ecef_ned(lat_deg, lon_deg)
    r_nb = rotations.euler_to_r(yaw_rad, p_prior, r_prior)
    r_ecef_cam = r_en @ r_nb @ rig.cam_by_id["F"].r_body_cam
    return {
        "p_ecef_cam": p_ecef,
        "r_ecef_cam": r_ecef_cam,
        "pitch_prior_deg": math.degrees(p_prior),
        "roll_prior_deg": math.degrees(r_prior),
    }


def refine_real_multiscale_edges_g1(
    rgb: np.ndarray,
    raw_edges: Sequence[dict[str, float]],
    strip_half_band_px: float = 20.0,
) -> list[dict[str, float]]:
    """Apply Theil-Sen strip consensus and regional sky-vs-surface step weighting to G1 multi-scale edges."""
    if not raw_edges:
        return []
    feat = extract_feature_map(rgb)
    h_img, w_img = feat.shape[:2]

    us = np.array([float(e["u"]) for e in raw_edges], dtype=np.float64)
    vs = np.array([float(e["v"]) for e in raw_edges], dtype=np.float64)

    slopes: list[float] = []
    step_i = max(1, len(us) // 24)
    for i in range(0, len(us), step_i):
        for j in range(i + 1, len(us), step_i):
            du = us[j] - us[i]
            if abs(du) >= 64.0:
                slopes.append(float((vs[j] - vs[i]) / du))
    m_ts = float(np.median(slopes)) if slopes else 0.0
    c_ts = float(np.median(vs - m_ts * us))

    refined: list[dict[str, float]] = []
    for e in raw_edges:
        u_val = float(e["u"])
        v_val = float(e["v"])
        v_pred = m_ts * u_val + c_ts
        if abs(v_val - v_pred) > strip_half_band_px:
            continue
        u_i = int(np.clip(round(u_val), 0, w_img - 1))
        v_i = int(np.clip(round(v_val), 2, h_img - 3))
        above = float(np.mean(feat[max(0, v_i - 22) : max(1, v_i - 2), u_i]))
        below = float(np.mean(feat[min(h_img - 2, v_i + 2) : min(h_img, v_i + 22), u_i]))
        reg_step = abs(above - below)
        base_g = float(e.get("ms_gradient", e.get("gradient", 0.01)))  # default-ok: optional fallback default
        e_out = dict(e)
        e_out["gradient"] = float(base_g * (0.25 + 2.5 * reg_step))
        e_out["regional_step"] = float(reg_step)
        refined.append(e_out)

    if len(refined) < 6:
        return list(raw_edges)
    return refined


def evaluate_single_real_frame(
    rgb: np.ndarray,
    seq_name: str,
    frame_id: int,
    points: Sequence[tuple[float, float]],
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_nom_m: float,
    yaw_rad: float,
    rng_seed: int,
    detector_params: dict[str, Any],
) -> dict[str, Any]:
    """Run G0 and G1 horizon/attitude and OOD detectors on a single real video frame."""
    h_img, w_img = rgb.shape[:2]
    gt = fit_polyline_horizon_gt(points, w_img, h_img, rig, lat_deg, lon_deg, h_nom_m, yaw_rad)

    rng = np.random.default_rng(rng_seed)
    prior_pose = build_coarse_ins_prior_pose(
        rig, lat_deg, lon_deg, h_nom_m, yaw_rad, gt["pitch_gt_rad"], gt["roll_gt_rad"], rng
    )

    step_px = int(detector_params["step_px"])
    g0_min_grad = float(detector_params["g0_min_gradient"])
    g1_min_grad = float(detector_params["g1_min_gradient"])
    g1_scales = tuple(float(s) for s in detector_params["g1_scales"])
    g1_weights = tuple(float(w) for w in detector_params["g1_scale_weights"])
    g1_strip_band = int(detector_params["g1_strip_half_band_px"])
    g1_border = int(detector_params["g1_border_margin_px"])

    # 1. G0 single-scale edge detector + G0 RANSAC/GN fit
    edges_g0 = detect_camera_edges(
        rgb,
        "F",
        rig.camera_model,
        prior_pose=prior_pose,
        step=step_px,
        search_band_px=75,
        min_gradient=g0_min_grad,
    )
    batch_g0 = edges_to_ray_batch({"F": edges_g0}, rig)
    fit_g0 = fit_pipeline_attitude(lat_deg, lon_deg, h_nom_m, yaw_rad, batch_g0, frame_k=frame_id, rng_seed_base=rng_seed)

    # 2. G1 multi-scale edge detector + regional step strip consensus + smooth pseudo-Huber IRLS fit
    edges_g1_raw = detect_camera_edges_multiscale(
        rgb,
        "F",
        rig.camera_model,
        prior_pose=prior_pose,
        step=step_px,
        search_band_px=55,
        min_gradient=g1_min_grad,
        scales=g1_scales,
        scale_weights=g1_weights,
        strip_half_band_px=g1_strip_band,
        border_margin_px=g1_border,
    )
    edges_g1 = refine_real_multiscale_edges_g1(rgb, edges_g1_raw, strip_half_band_px=22.0)
    batch_g1 = edges_to_ray_batch({"F": edges_g1}, rig)
    fit_g1 = fit_pipeline_attitude_g1(
        lat_deg,
        lon_deg,
        h_nom_m,
        yaw_rad,
        batch_g1,
        frame_k=frame_id,
        rng_seed_base=rng_seed,
        max_iterations=40,
        inlier_threshold_rad=math.radians(0.16),
        min_inliers=3,
        huber_delta_rad=math.radians(0.06),
    )

    # 3. Low-contrast polar haze stress test (0.22x contrast around mean luminance)
    rgb_f = rgb.astype(np.float64)
    mean_lum = float(np.mean(rgb_f))
    hazy_rgb = np.clip(mean_lum + 0.22 * (rgb_f - mean_lum), 0.0, 255.0).astype(np.uint8)
    edges_g0_hazy = detect_camera_edges(
        hazy_rgb, "F", rig.camera_model, prior_pose=prior_pose, step=step_px, search_band_px=75, min_gradient=g0_min_grad
    )
    fit_g0_hazy = fit_pipeline_attitude(
        lat_deg, lon_deg, h_nom_m, yaw_rad, edges_to_ray_batch({"F": edges_g0_hazy}, rig), frame_k=frame_id, rng_seed_base=rng_seed
    )
    edges_g1_hazy_raw = detect_camera_edges_multiscale(
        hazy_rgb,
        "F",
        rig.camera_model,
        prior_pose=prior_pose,
        step=step_px,
        search_band_px=55,
        min_gradient=g1_min_grad,
        scales=g1_scales,
        scale_weights=g1_weights,
        strip_half_band_px=g1_strip_band,
        border_margin_px=g1_border,
    )
    edges_g1_hazy = refine_real_multiscale_edges_g1(hazy_rgb, edges_g1_hazy_raw, strip_half_band_px=22.0)
    fit_g1_hazy = fit_pipeline_attitude_g1(
        lat_deg,
        lon_deg,
        h_nom_m,
        yaw_rad,
        edges_to_ray_batch({"F": edges_g1_hazy}, rig),
        frame_k=frame_id,
        rng_seed_base=rng_seed,
        max_iterations=40,
        inlier_threshold_rad=math.radians(0.16),
        min_inliers=3,
        huber_delta_rad=math.radians(0.06),
    )

    # 4. Classical horizon-strip OOD features on the real frame
    ood_feats = compute_horizon_strip_ood_features(rgb, edges=edges_g1, cam=rig.camera_model, cam_id="F")

    # 5. Image-plane line metrics for G0 and G1
    g0_line = _eval_edge_line_metrics(edges_g0, fit_g0, w_img, h_img, gt)
    g1_line = _eval_edge_line_metrics(edges_g1, fit_g1, w_img, h_img, gt)

    # 6. 3D attitude errors and covariance scale
    g0_roll_err = abs(float(fit_g0["roll_deg"]) - gt["roll_gt_deg"])
    g0_pitch_err = abs(float(fit_g0["pitch_deg"]) - gt["pitch_gt_deg"])
    g0_tot_err = math.hypot(g0_roll_err, g0_pitch_err)

    g1_roll_err = abs(float(fit_g1["roll_deg"]) - gt["roll_gt_deg"])
    g1_pitch_err = abs(float(fit_g1["pitch_deg"]) - gt["pitch_gt_deg"])
    g1_tot_err = math.hypot(g1_roll_err, g1_pitch_err)

    cov_g1 = fit_g1["covariance_rad2"]
    tr_cov_g1 = float(np.trace(cov_g1)) if cov_g1 is not None and not np.any(np.isnan(cov_g1)) else 1e-6
    sigma_solver_rad = max(1e-6, math.sqrt(max(0.0, tr_cov_g1)))
    sigma_solver_deg = math.degrees(sigma_solver_rad)
    rms_res_urad = float(fit_g1["rms_residual_urad"]) if not math.isnan(float(fit_g1["rms_residual_urad"])) else 5000.0

    platform = "drone" if seq_name.startswith("drone") else "ship"

    return {
        "seq_name": seq_name,
        "platform": platform,
        "frame_id": int(frame_id),
        "width": int(w_img),
        "height": int(h_img),
        "gt": gt,
        "g0": {
            "status": str(fit_g0["status"]),
            "n_edges": len(edges_g0),
            "inlier_count": int(fit_g0["inlier_count"]),
            "roll_est_deg": float(fit_g0["roll_deg"]),
            "pitch_est_deg": float(fit_g0["pitch_deg"]),
            "roll_error_deg": float(g0_roll_err),
            "pitch_error_deg": float(g0_pitch_err),
            "total_error_deg": float(g0_tot_err),
            "rms_residual_urad": float(fit_g0["rms_residual_urad"]) if not math.isnan(float(fit_g0["rms_residual_urad"])) else -1.0,
            "tilt_error_deg": float(g0_line["tilt_error_deg"]),
            "v_mid_error_px": float(g0_line["v_mid_error_px"]),
            "rho_error_px": float(g0_line["rho_error_px"]),
            "hazy_status": str(fit_g0_hazy["status"]),
            "hazy_n_edges": len(edges_g0_hazy),
            "hazy_total_error_deg": float(
                math.hypot(
                    float(fit_g0_hazy["roll_deg"]) - gt["roll_gt_deg"],
                    float(fit_g0_hazy["pitch_deg"]) - gt["pitch_gt_deg"],
                )
            ),
        },
        "g1": {
            "status": str(fit_g1["status"]),
            "n_edges": len(edges_g1),
            "inlier_count": int(fit_g1["inlier_count"]),
            "roll_est_deg": float(fit_g1["roll_deg"]),
            "pitch_est_deg": float(fit_g1["pitch_deg"]),
            "roll_error_deg": float(g1_roll_err),
            "pitch_error_deg": float(g1_pitch_err),
            "total_error_deg": float(g1_tot_err),
            "rms_residual_urad": float(rms_res_urad),
            "sigma_solver_rad": float(sigma_solver_rad),
            "sigma_solver_deg": float(sigma_solver_deg),
            "tilt_error_deg": float(g1_line["tilt_error_deg"]),
            "v_mid_error_px": float(g1_line["v_mid_error_px"]),
            "rho_error_px": float(g1_line["rho_error_px"]),
            "m_est": float(g1_line["m_est"]),
            "c_est": float(g1_line["c_est"]),
            "hazy_status": str(fit_g1_hazy["status"]),
            "hazy_n_edges": len(edges_g1_hazy),
            "hazy_total_error_deg": float(
                math.hypot(
                    float(fit_g1_hazy["roll_deg"]) - gt["roll_gt_deg"],
                    float(fit_g1_hazy["pitch_deg"]) - gt["pitch_gt_deg"],
                )
            ),
        },
        "ood": ood_feats,
    }


def _eval_edge_line_metrics(
    edges: Sequence[dict[str, float]],
    fit: dict[str, Any],
    width: int,
    height: int,
    gt: dict[str, float],
) -> dict[str, float]:
    if not edges:
        return {
            "m_est": 0.0,
            "c_est": 0.5 * float(height),
            "tilt_error_deg": 90.0,
            "v_mid_error_px": 0.5 * float(height),
            "rho_error_px": 0.5 * float(height),
        }
    inliers = fit.get("inliers")
    if inliers is not None and len(inliers) >= 2:
        sub = [edges[int(i)] for i in inliers if int(i) < len(edges)]
    else:
        sub = list(edges)
    us = np.array([float(e["u"]) for e in sub], dtype=np.float64)
    vs = np.array([float(e["v"]) for e in sub], dtype=np.float64)
    ws = np.array([float(e.get("gradient", 1.0)) for e in sub], dtype=np.float64)  # default-ok: optional fallback default
    line = fit_line_tls(us, vs, ws)
    m_est = line["m"]
    c_est = line["c"]
    u_mid = 0.5 * float(width)
    v_mid_est = m_est * u_mid + c_est
    rho_est = (v_mid_est - 0.5 * float(height)) / math.sqrt(1.0 + m_est * m_est)
    return {
        "m_est": float(m_est),
        "c_est": float(c_est),
        "tilt_error_deg": abs(float(line["tilt_deg"]) - gt["tilt_gt_deg"]),
        "v_mid_error_px": abs(float(v_mid_est) - gt["v_mid_gt_px"]),
        "rho_error_px": abs(float(rho_est) - gt["rho_gt_px"]),
    }


def summarize_g0_vs_g1_benchmark(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics comparing G0 and G1 on real IceHorizon frames."""
    def _stats_for(recs: Sequence[dict[str, Any]], key_model: str) -> dict[str, float]:
        n = len(recs)
        conv = [r for r in recs if r[key_model]["status"] == "CONVERGED"]
        hazy_conv = [r for r in recs if r[key_model]["hazy_status"] == "CONVERGED"]
        tot_errs = np.array([float(r[key_model]["total_error_deg"]) for r in recs], dtype=np.float64)
        roll_errs = np.array([float(r[key_model]["roll_error_deg"]) for r in recs], dtype=np.float64)
        pitch_errs = np.array([float(r[key_model]["pitch_error_deg"]) for r in recs], dtype=np.float64)
        tilt_errs = np.array([float(r[key_model]["tilt_error_deg"]) for r in recs], dtype=np.float64)
        vmid_errs = np.array([float(r[key_model]["v_mid_error_px"]) for r in recs], dtype=np.float64)
        rho_errs = np.array([float(r[key_model]["rho_error_px"]) for r in recs], dtype=np.float64)
        hazy_errs = np.array([float(r[key_model]["hazy_total_error_deg"]) for r in recs], dtype=np.float64)
        return {
            "n_frames": int(n),
            "n_converged": int(len(conv)),
            "convergence_rate": float(len(conv) / max(1, n)),
            "hazy_n_converged": int(len(hazy_conv)),
            "hazy_convergence_rate": float(len(hazy_conv) / max(1, n)),
            "hazy_median_total_error_deg": float(np.median(hazy_errs)),
            "hazy_p90_total_error_deg": float(np.percentile(hazy_errs, 90)),
            "median_total_error_deg": float(np.median(tot_errs)),
            "mean_total_error_deg": float(np.mean(tot_errs)),
            "p90_total_error_deg": float(np.percentile(tot_errs, 90)),
            "p95_total_error_deg": float(np.percentile(tot_errs, 95)),
            "max_total_error_deg": float(np.max(tot_errs)),
            "median_roll_error_deg": float(np.median(roll_errs)),
            "p90_roll_error_deg": float(np.percentile(roll_errs, 90)),
            "median_pitch_error_deg": float(np.median(pitch_errs)),
            "p90_pitch_error_deg": float(np.percentile(pitch_errs, 90)),
            "median_tilt_error_deg": float(np.median(tilt_errs)),
            "median_v_mid_error_px": float(np.median(vmid_errs)),
            "p90_v_mid_error_px": float(np.percentile(vmid_errs, 90)),
            "median_rho_error_px": float(np.median(rho_errs)),
        }

    overall_g0 = _stats_for(records, "g0")
    overall_g1 = _stats_for(records, "g1")

    drone_recs = [r for r in records if r["platform"] == "drone"]
    ship_recs = [r for r in records if r["platform"] == "ship"]

    ood_scores = np.array([float(r["ood"]["ood_horizon_score"]) for r in records], dtype=np.float64)
    g1_errs = np.array([float(r["g1"]["total_error_deg"]) for r in records], dtype=np.float64)
    ood_med = float(np.median(ood_scores))
    low_ood_errs = g1_errs[ood_scores <= ood_med]
    high_ood_errs = g1_errs[ood_scores > ood_med]

    return {
        "overall": {"g0": overall_g0, "g1": overall_g1},
        "by_platform": {
            "drone": {"g0": _stats_for(drone_recs, "g0"), "g1": _stats_for(drone_recs, "g1")},
            "ship": {"g0": _stats_for(ship_recs, "g0"), "g1": _stats_for(ship_recs, "g1")},
        },
        "ood_quality_stratification": {
            "median_ood_score": ood_med,
            "low_ood_n": int(len(low_ood_errs)),
            "low_ood_median_error_deg": float(np.median(low_ood_errs)) if len(low_ood_errs) else 0.0,
            "low_ood_p90_error_deg": float(np.percentile(low_ood_errs, 90)) if len(low_ood_errs) else 0.0,
            "high_ood_n": int(len(high_ood_errs)),
            "high_ood_median_error_deg": float(np.median(high_ood_errs)) if len(high_ood_errs) else 0.0,
            "high_ood_p90_error_deg": float(np.percentile(high_ood_errs, 90)) if len(high_ood_errs) else 0.0,
        },
    }


def compute_real_nonconformity_scale_deg(
    rec: dict[str, Any],
    sigma_floor_deg: float,
    rms_weight_deg_per_mrad: float,
    ood_weight_deg: float,
) -> float:
    """Compute real-imagery uncertainty scale `sigma_real_deg` combining solver covariance,
    IRLS residual RMS, and horizon-strip OOD score."""
    sigma_sol = float(rec["g1"]["sigma_solver_deg"])
    rms_mrad = float(rec["g1"]["rms_residual_urad"]) * 1e-3
    ood_s = float(rec["ood"]["ood_horizon_score"])
    return float(
        math.sqrt(
            sigma_sol * sigma_sol
            + (rms_weight_deg_per_mrad * rms_mrad) ** 2
            + (ood_weight_deg * ood_s) ** 2
            + sigma_floor_deg * sigma_floor_deg
        )
    )


def evaluate_h3_conformal_transfer(
    records: Sequence[dict[str, Any]],
    synthetic_g1_results: dict[str, Any],
    cal_sequences: Sequence[str],
    test_sequences: Sequence[str],
    conformal_params: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Charter §10 Hypothesis H3: Zero-Shot Synthetic-Calibrated Conformal Protection Levels
    vs Sequence-Disjoint Real-Recalibrated Split & Mondrian Conformal Protection Levels."""
    cal_set = set(cal_sequences)
    test_set = set(test_sequences)
    overlap = cal_set & test_set
    if overlap:
        raise ValueError(f"Sequence leakage detected between real_cal and real_test: {sorted(overlap)}")

    cal_recs = [r for r in records if r["seq_name"] in cal_set]
    test_recs = [r for r in records if r["seq_name"] in test_set]
    if not cal_recs or not test_recs:
        raise ValueError("Both real_cal and real_test must be non-empty")

    alpha_levels = [float(a) for a in conformal_params["alpha_levels"]]
    alert_limits_deg = [float(al) for al in conformal_params["alert_limits_deg"]]
    sigma_floor_deg = float(conformal_params["real_sigma_floor_deg"])
    rms_w = float(conformal_params["real_rms_weight_deg_per_mrad"])
    ood_w = float(conformal_params["real_ood_weight_deg"])
    cov_tol = float(conformal_params["coverage_tolerance"])

    # Extract synthetic calibration quantiles from WP-07 G1 results.json
    cm_syn = synthetic_g1_results["item2_altitude_conditioning"]["conformal_models"]["far_0.01"]
    base_syn = cm_syn["base_model"]
    alt_q_syn = cm_syn["altitude_mondrian_quantiles"]
    sigma_sys_f_rad = float(base_syn["sigma_sys_rad"]["F"])

    # Synthetic quantiles for single-camera F without artificial barometer dip inflation (F:clean:ge_1000m)
    q_syn_slice = alt_q_syn.get("F:clean:ge_1000m", base_syn["mondrian_quantiles"]["F:clean"])
    q_syn_010 = float(q_syn_slice["0.1"])
    q_syn_001 = float(q_syn_slice["0.01"])
    q_syn_0001 = float(q_syn_slice["0.001"])
    # Exact log-interpolation at alpha=0.05 between alpha=0.10 and alpha=0.01
    w_005 = (math.log10(0.10) - math.log10(0.05)) / (math.log10(0.10) - math.log10(0.01))
    q_syn_005 = float(q_syn_010 + w_005 * (q_syn_001 - q_syn_010))
    syn_quantiles_by_alpha: dict[str, float] = {
        "0.1": q_syn_010,
        "0.05": q_syn_005,
        "0.01": q_syn_001,
        "0.001": q_syn_0001,
    }

    # Calibrate Real Conformal Quantiles strictly on `real_cal` (Standard + Platform Mondrian)
    cal_scores_all = np.array(
        [
            float(r["g1"]["total_error_deg"])
            / compute_real_nonconformity_scale_deg(r, sigma_floor_deg, rms_w, ood_w)
            for r in cal_recs
        ],
        dtype=np.float64,
    )
    med_all = max(1e-9, float(np.median(cal_scores_all)))
    real_standard_quantiles: dict[str, float] = {}
    real_mondrian_quantiles: dict[str, dict[str, float]] = {"drone": {}, "ship": {}}
    n_cal_mondrian: dict[str, int] = {}

    for plat in ("drone", "ship"):
        sub_c = [
            float(r["g1"]["total_error_deg"])
            / compute_real_nonconformity_scale_deg(r, sigma_floor_deg, rms_w, ood_w)
            for r in cal_recs
            if r["platform"] == plat
        ]
        n_cal_mondrian[plat] = len(sub_c)
        sub_arr = np.asarray(sub_c, dtype=np.float64)
        med_plat = float(np.median(sub_arr)) if len(sub_arr) > 0 else med_all
        scale_plat = max(1.0, med_plat / med_all)
        for alpha in alpha_levels:
            a_key = f"{alpha:.4f}".rstrip("0")
            q_std = compute_conformal_quantile(cal_scores_all, alpha)
            real_standard_quantiles[a_key] = float(q_std)
            q_plat = compute_conformal_quantile(sub_arr, alpha) if len(sub_arr) >= 20 else float("inf")
            if math.isinf(q_plat) or math.isnan(q_plat):
                # Finite-sample pooled standardized tail quantile when k > n_slice (e.g. n_drone=60 at alpha=0.01)
                q_plat = float(q_std * scale_plat)
            real_mondrian_quantiles[plat][a_key] = float(max(q_std, q_plat))

    # Evaluate on held-out `real_test` (300 frames across 15 disjoint sequences)
    by_alpha: dict[str, Any] = {}
    n_test = len(test_recs)

    for alpha in alpha_levels:
        a_key = f"{alpha:.4f}".rstrip("0")
        target_cov = 1.0 - alpha
        q_syn = float(syn_quantiles_by_alpha[a_key])
        q_real_std = float(real_standard_quantiles[a_key])

        syn_covered = 0
        real_std_covered = 0
        real_mond_covered = 0

        syn_pls: list[float] = []
        real_std_pls: list[float] = []
        real_mond_pls: list[float] = []

        for r in test_recs:
            err_deg = float(r["g1"]["total_error_deg"])
            # 1. Zero-shot synthetic PL
            sig_syn_deg = math.degrees(math.hypot(float(r["g1"]["sigma_solver_rad"]), sigma_sys_f_rad))
            pl_syn = q_syn * sig_syn_deg
            syn_pls.append(pl_syn)
            if err_deg <= pl_syn:
                syn_covered += 1

            # 2. Real-recalibrated standard split-conformal PL
            sig_real_deg = compute_real_nonconformity_scale_deg(r, sigma_floor_deg, rms_w, ood_w)
            pl_real_std = q_real_std * sig_real_deg
            real_std_pls.append(pl_real_std)
            if err_deg <= pl_real_std:
                real_std_covered += 1

            # 3. Real-recalibrated Platform-Mondrian PL
            q_real_m = float(real_mondrian_quantiles[r["platform"]][a_key])
            pl_real_m = q_real_m * sig_real_deg
            real_mond_pls.append(pl_real_m)
            if err_deg <= pl_real_m:
                real_mond_covered += 1

        cov_syn = float(syn_covered / n_test)
        ci_syn = clopper_pearson_ci(syn_covered, n_test)
        gap_syn = float(cov_syn - target_cov)

        cov_real_std = float(real_std_covered / n_test)
        ci_real_std = clopper_pearson_ci(real_std_covered, n_test)
        gap_real_std = float(cov_real_std - target_cov)

        cov_real_mond = float(real_mond_covered / n_test)
        ci_real_mond = clopper_pearson_ci(real_mond_covered, n_test)
        gap_real_mond = float(cov_real_mond - target_cov)

        mean_pl_syn = float(np.mean(syn_pls))
        mean_pl_real_std = float(np.mean(real_std_pls))
        mean_pl_real_mond = float(np.mean(real_mond_pls))

        by_alpha[a_key] = {
            "alpha": float(alpha),
            "target_coverage": float(target_cov),
            "n_test": int(n_test),
            "zero_shot_synthetic": {
                "quantile": float(q_syn),
                "k_covered": int(syn_covered),
                "empirical_coverage": cov_syn,
                "ci_low": float(ci_syn["ci_low"]),
                "ci_high": float(ci_syn["ci_high"]),
                "coverage_gap": gap_syn,
                "mean_pl_deg": mean_pl_syn,
                "median_pl_deg": float(np.median(syn_pls)),
                "p90_pl_deg": float(np.percentile(syn_pls, 90)),
                "transfers_without_recalibration": bool(abs(gap_syn) <= cov_tol),
            },
            "real_recalibrated_standard": {
                "quantile": float(q_real_std),
                "k_covered": int(real_std_covered),
                "empirical_coverage": cov_real_std,
                "ci_low": float(ci_real_std["ci_low"]),
                "ci_high": float(ci_real_std["ci_high"]),
                "coverage_gap": gap_real_std,
                "mean_pl_deg": mean_pl_real_std,
                "median_pl_deg": float(np.median(real_std_pls)),
                "p90_pl_deg": float(np.percentile(real_std_pls, 90)),
                "pl_ratio_vs_synthetic": float(mean_pl_real_std / max(1e-9, mean_pl_syn)),
                "pass": bool(cov_real_std >= target_cov - cov_tol),
            },
            "real_recalibrated_mondrian": {
                "quantiles_by_platform": {
                    "drone": float(real_mondrian_quantiles["drone"][a_key]),
                    "ship": float(real_mondrian_quantiles["ship"][a_key]),
                },
                "k_covered": int(real_mond_covered),
                "empirical_coverage": cov_real_mond,
                "ci_low": float(ci_real_mond["ci_low"]),
                "ci_high": float(ci_real_mond["ci_high"]),
                "coverage_gap": gap_real_mond,
                "mean_pl_deg": mean_pl_real_mond,
                "median_pl_deg": float(np.median(real_mond_pls)),
                "p90_pl_deg": float(np.percentile(real_mond_pls, 90)),
                "pl_ratio_vs_synthetic": float(mean_pl_real_mond / max(1e-9, mean_pl_syn)),
                "pass": bool(cov_real_mond >= target_cov - cov_tol),
            },
        }

    # Evaluate HMI rate and availability on `real_test` at alpha=0.05 and alpha=0.01 across alert limits
    hmi_and_availability: dict[str, Any] = {}
    for al_deg in alert_limits_deg:
        al_key = f"AL_{al_deg:.1f}deg"
        syn_hmi = 0
        real_hmi = 0
        real_avail = 0
        for r in test_recs:
            err_deg = float(r["g1"]["total_error_deg"])
            sig_syn_deg = math.degrees(math.hypot(float(r["g1"]["sigma_solver_rad"]), sigma_sys_f_rad))
            pl_syn_001 = float(syn_quantiles_by_alpha["0.01"]) * sig_syn_deg
            if pl_syn_001 <= al_deg and err_deg > al_deg:
                syn_hmi += 1

            sig_real_deg = compute_real_nonconformity_scale_deg(r, sigma_floor_deg, rms_w, ood_w)
            pl_real_001 = float(real_mondrian_quantiles[r["platform"]]["0.01"]) * sig_real_deg
            if pl_real_001 <= al_deg:
                real_avail += 1
                if err_deg > al_deg:
                    real_hmi += 1

        ci_syn_hmi = clopper_pearson_ci(syn_hmi, n_test)
        ci_real_hmi = clopper_pearson_ci(real_hmi, n_test)
        hmi_and_availability[al_key] = {
            "alert_limit_deg": float(al_deg),
            "n_test": int(n_test),
            "zero_shot_synthetic_hmi_count": int(syn_hmi),
            "zero_shot_synthetic_hmi_rate": float(syn_hmi / n_test),
            "zero_shot_synthetic_hmi_ci_high": float(ci_syn_hmi["ci_high"]),
            "real_recalibrated_hmi_count": int(real_hmi),
            "real_recalibrated_hmi_rate": float(real_hmi / n_test),
            "real_recalibrated_hmi_ci_high": float(ci_real_hmi["ci_high"]),
            "real_recalibrated_available_count": int(real_avail),
            "real_recalibrated_availability": float(real_avail / n_test),
        }

    return {
        "split_summary": {
            "n_cal_sequences": len(cal_set),
            "n_test_sequences": len(test_set),
            "sequence_overlap_count": len(overlap),
            "n_cal_frames": len(cal_recs),
            "n_test_frames": len(test_recs),
            "n_cal_by_platform": n_cal_mondrian,
            "cal_sequences": list(cal_sequences),
            "test_sequences": list(test_sequences),
            "tuned_on_splits": ["real_cal"],
            "evaluated_on_split": "real_test",
        },
        "synthetic_reference_quantiles": syn_quantiles_by_alpha,
        "real_standard_quantiles": real_standard_quantiles,
        "real_mondrian_quantiles": real_mondrian_quantiles,
        "by_alpha": by_alpha,
        "hmi_and_availability": hmi_and_availability,
        "hypothesis_h3_confirmed": bool(
            abs(by_alpha["0.05"]["zero_shot_synthetic"]["coverage_gap"]) > cov_tol
            and abs(by_alpha["0.01"]["zero_shot_synthetic"]["coverage_gap"]) > cov_tol
            and by_alpha["0.05"]["real_recalibrated_mondrian"]["pass"]
            and by_alpha["0.01"]["real_recalibrated_mondrian"]["pass"]
        ),
    }
