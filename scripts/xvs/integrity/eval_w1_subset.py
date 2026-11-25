"""W1: Compute and compare the three camera-consistency statistics on the seeded subset.

Subset: every 60th frame of T1/T2/T3, all 20 seeds and the clean pass.
Statistics:
1. pre-fix function (git show 78628f5)
2. HEAD generator function (at 54b0eae)
3. D27 (spec v0.1.1 §2: camera c alone against the rest, using RANSAC inliers)

Evaluates fault-free validation frames per camera set (F, FRBL, all).
Reports median, p95, p99, and arg-max camera histogram for D27.
If D27 p99 > 0.5 deg, lists the 10 worst frames with:
  camera, inlier count, inlier column span, sub-fit condition number.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, REPO_ROOT
from xvs.rig.rig import Rig
from xvs.sensor.chain import SensorConfig, CameraSimulator
from xvs.horizon_att.edges import detect_camera_edges, edges_to_ray_batch
from xvs.horizon_att.pipeline import (
    load_pipeline_rig,
    build_pipeline_ray_batch,
    fit_pipeline_attitude,
)
from xvs.horizon_att.limb_model import LimbRayBatch, solve_attitude_gn
from xvs.integrity.monitor import camera_consistency, CameraConsistencyResult
from xvs.integrity.faults import (
    inject_occlusion,
    inject_spurious_edge,
    inject_altitude_bias,
    inject_glare,
    SeededFaultInjector,
)

ALL_CAMERAS = ("F", "R", "B", "L", "U", "D")
CAMERA_SETS = {
    "F": ("F",),
    "FRBL": ("F", "R", "B", "L"),
    "all": ("F", "R", "B", "L", "U", "D"),
}
SIGMA_H_SET = (0.0, 10.0, 50.0)
SEQS = ("T1_cruise", "T2_ksfo28l", "T3_turn_climb")
STRIDE = 60

# Worker globals
_WORKER_SIMS: dict[str, CameraSimulator] | None = None
_WORKER_RIG: Rig | None = None
_WORKER_SENSOR_CONFIG: SensorConfig | None = None


def _init_worker(sensor_config_path: str) -> None:
    global _WORKER_SIMS, _WORKER_RIG, _WORKER_SENSOR_CONFIG
    _WORKER_RIG = load_pipeline_rig()
    cfg = SensorConfig.from_file(sensor_config_path)
    cfg.get_psf_kernel()
    cfg.get_backward_map()
    cfg.get_vignetting_map()
    _WORKER_SENSOR_CONFIG = cfg
    _WORKER_SIMS = {c: CameraSimulator(cfg, f"CAM_{c}") for c in ALL_CAMERAS}


# --- Statistic 1: Pre-fix function (78628f5) ---
def stat1_prefix_78628f5(
    full_fit: dict[str, Any],
    sub_edges: dict[str, list[dict[str, float]]],
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_meas_m: float,
    yaw_rad: float,
    sensor_config: SensorConfig | None = None,
) -> float:
    cams = [c for c, ed in sub_edges.items() if len(ed) > 0]
    if len(cams) <= 1:
        return 0.0
    p_full = full_fit["pitch_deg"]
    r_full = full_fit["roll_deg"]
    diffs = {}
    for c_leave in cams:
        remaining_edges = {c: ed for c, ed in sub_edges.items() if c != c_leave}
        batch_sub = edges_to_ray_batch(remaining_edges, rig, sensor_config=sensor_config)
        if batch_sub.n_rays >= 3:
            fit_sub = solve_attitude_gn(
                lat_deg, lon_deg, h_meas_m, yaw_rad,
                math.radians(p_full), math.radians(r_full),
                batch_sub, max_iter=20, tol_rad=1e-10,
            )
            d_p = fit_sub["pitch_deg"] - p_full
            d_r = fit_sub["roll_deg"] - r_full
            diffs[c_leave] = float(math.hypot(d_p, d_r))
        else:
            diffs[c_leave] = float("nan")
    valid = [v for v in diffs.values() if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# --- Statistic 2: HEAD generator function (54b0eae) ---
def stat2_head_generator(
    full_fit: dict[str, Any],
    sub_edges: dict[str, list[dict[str, float]]],
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_meas_m: float,
    yaw_rad: float,
    sensor_config: SensorConfig | None = None,
) -> float:
    inlier_counts = full_fit.get("inlier_count_by_cam") or full_fit.get("inlier_counts_by_cam") or {}
    valid_cams = sorted([c for c in sub_edges.keys() if inlier_counts.get(c, 0) >= 10])  # default-ok: frequency counter accumulator
    if len(valid_cams) < 3:
        return float("nan")
    p_full = full_fit["pitch_deg"]
    r_full = full_fit["roll_deg"]
    diffs = {}
    for c_leave in valid_cams:
        remaining_edges = {c: ed for c, ed in sub_edges.items() if c in valid_cams and c != c_leave}
        batch_sub = edges_to_ray_batch(remaining_edges, rig, sensor_config=sensor_config)
        if batch_sub.n_rays >= 3:
            fit_sub = solve_attitude_gn(
                lat_deg, lon_deg, h_meas_m, yaw_rad,
                math.radians(p_full), math.radians(r_full),
                batch_sub, max_iter=20, tol_rad=1e-10,
            )
            if fit_sub.get("status") == "CONVERGED":
                d_p = fit_sub["pitch_deg"] - p_full
                d_r = fit_sub["roll_deg"] - r_full
                diffs[c_leave] = float(math.hypot(d_p, d_r))
            else:
                diffs[c_leave] = float("nan")
        else:
            diffs[c_leave] = float("nan")
    valid = [v for v in diffs.values() if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


def evaluate_frame_subset(
    edges_by_cam: dict[str, list[dict[str, float]]],
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_true_m: float,
    h_meas_base_m: float,
    yaw_rad: float,
    alt_noise_dict: dict[float, float],
    frame_k: int,
    base_record: dict[str, Any],
    rng_seed: int,
    sensor_config: SensorConfig | None = None,
) -> list[dict[str, Any]]:
    records = []
    for cam_set_name, allowed_cams in CAMERA_SETS.items():
        sub_edges = {c: edges_by_cam.get(c, []) for c in allowed_cams}
        batch = build_pipeline_ray_batch(edges_by_cam, allowed_cams, rig, sensor_config=sensor_config)

        for sigma_h in SIGMA_H_SET:
            noise_h = alt_noise_dict.get(sigma_h, 0.0)  # default-ok: optional fallback default
            h_meas = max(1.0, h_meas_base_m + noise_h)

            if batch.n_rays >= 3:
                fit = fit_pipeline_attitude(
                    lat_deg, lon_deg, h_meas, yaw_rad, batch,
                    frame_k=frame_k, rng_seed_base=rng_seed,
                )

                if cam_set_name == "F":
                    s1 = 0.0
                    s2 = 0.0
                    s3 = float("nan")
                    arg_max_cam = None
                    worst_meta = None
                else:
                    s1 = stat1_prefix_78628f5(fit, sub_edges, rig, lat_deg, lon_deg, h_meas, yaw_rad, sensor_config=sensor_config)
                    s2 = stat2_head_generator(fit, sub_edges, rig, lat_deg, lon_deg, h_meas, yaw_rad, sensor_config=sensor_config)

                    # Statistic 3: D27
                    if "inliers" in fit and len(fit["inliers"]) > 0:
                        inlier_idx = fit["inliers"]
                        inlier_batch = LimbRayBatch(
                            batch.d_body[inlier_idx],
                            batch.lever_arm_body[inlier_idx],
                            [batch.cam_ids[i] for i in inlier_idx],
                            uvs=batch.uvs[inlier_idx] if batch.uvs is not None else None,
                        )
                        cc_res = camera_consistency(fit, inlier_batch, rig, lat_deg, lon_deg, h_meas, yaw_rad)
                        s3 = cc_res.max_angle_deg
                        arg_max_cam = cc_res.arg_max_camera

                        # Detail metadata for worst-frame analysis
                        if arg_max_cam is not None and not math.isnan(s3):
                            mask_best = [cid == arg_max_cam for cid in inlier_batch.cam_ids]
                            cnt_cam = int(sum(mask_best))
                            uvs_cam = inlier_batch.uvs[mask_best]
                            col_span = float(np.ptp(uvs_cam[:, 0])) if len(uvs_cam) > 0 else 0.0
                            batch_best = LimbRayBatch(
                                inlier_batch.d_body[mask_best],
                                inlier_batch.lever_arm_body[mask_best],
                                [cid for cid, m in zip(inlier_batch.cam_ids, mask_best) if m],
                            )
                            p_full_rad = full_fit = fit.get("pitch_rad", math.radians(fit["pitch_deg"]))
                            r_full_rad = full_fit = fit.get("roll_rad", math.radians(fit["roll_deg"]))
                            fit_best = solve_attitude_gn(
                                lat_deg, lon_deg, h_meas, yaw_rad,
                                p_full_rad, r_full_rad, batch_best,
                            )
                            cov_sub = fit_best.get("covariance_rad2")
                            try:
                                cond_sub = float(np.linalg.cond(cov_sub)) if cov_sub is not None else float("nan")
                            except Exception:
                                cond_sub = float("nan")
                            worst_meta = {
                                "camera": arg_max_cam,
                                "inlier_count": cnt_cam,
                                "inlier_col_span": col_span,
                                "sub_fit_cond": cond_sub,
                            }
                        else:
                            worst_meta = None
                    else:
                        s3 = float("nan")
                        arg_max_cam = None
                        worst_meta = None
            else:
                s1 = float("nan")
                s2 = float("nan")
                s3 = float("nan")
                arg_max_cam = None
                worst_meta = None

            rec = dict(base_record)
            rec.update({
                "cam_set": cam_set_name,
                "sigma_h": float(sigma_h),
                "h_meas_m": float(h_meas),
                "stat1_prefix_78628f5": s1,
                "stat2_head_generator": s2,
                "stat3_d27": s3,
                "arg_max_camera": arg_max_cam,
                "worst_meta": worst_meta,
            })
            records.append(rec)
    return records


def _process_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    global _WORKER_SIMS, _WORKER_RIG, _WORKER_SENSOR_CONFIG
    rig = _WORKER_RIG
    sims = _WORKER_SIMS
    sensor_cfg = _WORKER_SENSOR_CONFIG
    seq = task["seq_name"]
    k = task["k"]
    seed_idx = task["seed_idx"]
    split = task["split"]
    alt_noise = task["alt_noise_dict"]
    aircraft = task["aircraft"]
    fault_cfg = task["fault_config"]

    # We evaluate the nominal condition (fault_type == "none") and all fault conditions
    # For reporting W1 validation metrics, fault_type == "none" is authoritative.
    # Read clean images
    clean_imgs = {}
    for c in ALL_CAMERAS:
        img_path = task["clean_cam_paths"][c]
        clean_imgs[c] = np.array(Image.open(img_path))

    # Apply sensor simulation (nominal)
    image_id = f"{seq}_seed{seed_idx}"
    nominal_edges = {}
    for c in ALL_CAMERAS:
        res = sims[c].process_frame(clean_imgs[c], image_id=image_id, frame_idx=k, warp_method="cv2")
        nominal_edges[c] = detect_camera_edges(res["display"], c, rig.camera_model, step=8, search_band_px=50, min_gradient=0.015)

    base_rec = {
        "seq_name": seq,
        "k": k,
        "t_s": task["t_s"],
        "seed_idx": seed_idx,
        "split": split,
        "condition": "nominal",
        "fault_type": "none",
        "h_true_m": float(aircraft["h_m"]),
        "pitch_true_deg": float(aircraft["ypr_deg"][1]),
        "roll_true_deg": float(aircraft["ypr_deg"][2]),
    }

    recs = evaluate_frame_subset(
        nominal_edges, rig,
        float(aircraft["lat_deg"]), float(aircraft["lon_deg"]),
        float(aircraft["h_m"]), float(aircraft["h_m"]),
        math.radians(float(aircraft["ypr_deg"][0])),
        alt_noise, k, base_rec, rng_seed=20260925,
        sensor_config=sensor_cfg,
    )
    return recs


def main():
    parser = argparse.ArgumentParser(description="W1: Compute three statistics on seeded subset")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker processes (max 2 for W1)")
    args = parser.parse_args()

    data_root = get_data_root()
    g4 = data_root / "output" / "xvs" / "wp01" / "g4"
    if not g4.exists():
        g4 = REPO_ROOT / "output" / "xvs" / "wp01" / "g4"

    print(f"Data root for G4: {g4}")

    frames = {}
    for seq in SEQS:
        with open(g4 / seq / "frames.jsonl", encoding="utf-8") as f:
            frames[seq] = [json.loads(l) for l in f if l.strip() and json.loads(l)["k"] % STRIDE == 0]

    injector = SeededFaultInjector(master_seed=20260927)
    tasks = []
    for seed_idx in range(1, 21):
        split = "calibration" if seed_idx <= 10 else ("validation" if seed_idx <= 15 else "test")
        for seq in SEQS:
            for item in frames[seq]:
                k = item["k"]
                rng_alt = np.random.default_rng(20260925 + 1000 * seed_idx + k)
                alt_noise = {0.0: 0.0, 10.0: float(rng_alt.normal(0.0, 10.0)), 50.0: float(rng_alt.normal(0.0, 50.0))}
                tasks.append({
                    "seq_name": seq,
                    "k": k,
                    "t_s": item["t_s"],
                    "seed_idx": seed_idx,
                    "split": split,
                    "clean_cam_paths": {c: str(g4 / seq / c / f"{k:06d}.png") for c in ALL_CAMERAS},
                    "aircraft": item["aircraft"],
                    "fault_config": {},
                    "alt_noise_dict": alt_noise,
                })

    print(f"Total subset tasks: {len(tasks)} (frames across 20 seeds)")
    sensor_cfg_path = REPO_ROOT / "configs" / "xvs" / "sensors" / "rig_mono_v0.json"

    t0 = time.perf_counter()
    all_records = []
    # Process tasks using multiprocessing pool (at most 2 workers)
    workers = max(1, min(args.workers, 2))
    print(f"Starting worker pool with {workers} processes...")
    with mp.Pool(processes=workers, initializer=_init_worker, initargs=(str(sensor_cfg_path),)) as pool:
        for i, recs in enumerate(pool.imap(_process_task, tasks, chunksize=2)):
            all_records.extend(recs)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                print(f"Progress: {i + 1}/{len(tasks)} tasks ({time.perf_counter() - t0:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Processed {len(all_records)} records in {elapsed:.1f}s.")

    # Filter to fault-free validation records
    val_recs = [r for r in all_records if r["split"] == "validation" and r["fault_type"] == "none"]
    print(f"Fault-free validation records: {len(val_recs)}")

    summary = {}
    worst_records = []

    for cset in ("FRBL", "all", "F"):
        cset_recs = [r for r in val_recs if r["cam_set"] == cset]
        s1_vals = [r["stat1_prefix_78628f5"] for r in cset_recs if not math.isnan(r["stat1_prefix_78628f5"])]
        s2_vals = [r["stat2_head_generator"] for r in cset_recs if not math.isnan(r["stat2_head_generator"])]
        s3_vals = [r["stat3_d27"] for r in cset_recs if not math.isnan(r["stat3_d27"])]

        def get_stats(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"n": 0, "median": float("nan"), "p95": float("nan"), "p99": float("nan")}
            return {
                "n": len(vals),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
            }

        summary[cset] = {
            "n_total": len(cset_recs),
            "stat1_prefix_78628f5": get_stats(s1_vals),
            "stat2_head_generator": get_stats(s2_vals),
            "stat3_d27": get_stats(s3_vals),
        }

        # Collect arg-max camera histogram for D27
        hist = {}
        for r in cset_recs:
            cam = r.get("arg_max_camera")
            if cam:
                hist[cam] = hist.get(cam, 0) + 1  # default-ok: frequency counter accumulator
        summary[cset]["arg_max_hist"] = hist

        for r in cset_recs:
            if not math.isnan(r["stat3_d27"]):
                worst_records.append(r)

    # Sort worst records descending by stat3_d27
    worst_records.sort(key=lambda r: r["stat3_d27"], reverse=True)
    ten_worst = []
    for r in worst_records[:10]:
        meta = r.get("worst_meta") or {}
        ten_worst.append({
            "seq_name": r["seq_name"],
            "k": r["k"],
            "seed_idx": r["seed_idx"],
            "cam_set": r["cam_set"],
            "sigma_h": r["sigma_h"],
            "d27_deg": r["stat3_d27"],
            "arg_max_cam": r["arg_max_camera"],
            "camera": meta.get("camera"),
            "inlier_count": meta.get("inlier_count"),
            "inlier_col_span": meta.get("inlier_col_span"),
            "sub_fit_cond": meta.get("sub_fit_cond"),
        })

    summary["ten_worst_frames"] = ten_worst

    out_dir = REPO_ROOT / "output" / "xvs" / "wp07" / "g0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "w1_subset_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== W1 SUBSET STATISTICS SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {out_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
