"""Generate nominal and fault dataset for WP-07 G0 integrity layer.

Conforms strictly to docs/xvs/WP07_INTEGRITY.md §1 and GEMINI_G-B12.md §1 item 2:
- Measure a pilot and project CPU time.
- If projection exceeds 8 h wall, cut seeds to 12 (splits 1-6, 7-9, 10-12) and say so.
- Generate nominal set: 20 seeds (or 12) degraded images (rig_mono_v0), sigma_h in {0, 10, 50} m,
  camera sets F, horizontal (FRBL), all, clean images once.
- Generate fault set: occlusion, spurious edge, altitude bias, glare.
- Split by seed: calibration, validation, test. Clean frames go to calibration.
  No frame appears in two splits.
- Multiprocessing cap: at most (logical cores - 4) = 8 worker processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, REPO_ROOT, data_root_source
from xvs.rig.rig import Rig
from xvs.sensor.chain import SensorConfig, CameraSimulator
from xvs.horizon_att.edges import detect_camera_edges
from xvs.horizon_att.pipeline import (
    load_pipeline_rig,
    build_pipeline_ray_batch,
    fit_pipeline_attitude,
)
from xvs.horizon_att.limb_model import LimbRayBatch
from xvs.integrity.monitor import camera_consistency
from xvs.integrity.faults import (
    inject_occlusion,
    inject_spurious_edge,
    inject_altitude_bias,
    inject_glare,
    SeededFaultInjector,
    compute_sha256,
)
from xvs.paths import get_data_root, REPO_ROOT, data_root_source

DEFAULT_OUT_DIR = get_data_root() / "xvs" / "wp07" / "g0"
DEFAULT_SENSOR_CONFIG = REPO_ROOT / "configs" / "xvs" / "sensors" / "rig_mono_v0.json"
CAMERA_SETS = {
    "F": ("F",),
    "FRBL": ("F", "R", "B", "L"),
    "all": ("F", "R", "B", "L", "U", "D"),
}
SIGMA_H_SET = (0.0, 10.0, 50.0)
ALL_CAMERAS = ("F", "R", "B", "L", "U", "D")

# Worker-local global caches for multiprocessing
_WORKER_SIMULATORS: dict[str, CameraSimulator] | None = None
_WORKER_RIG: Rig | None = None
_WORKER_SENSOR_CONFIG: SensorConfig | None = None


def _init_worker(sensor_config_path: str) -> None:
    """Initialize worker process with cached CameraSimulator, calibrated Rig, and SensorConfig."""
    global _WORKER_SIMULATORS, _WORKER_RIG, _WORKER_SENSOR_CONFIG
    _WORKER_RIG = load_pipeline_rig()
    cfg = SensorConfig.from_file(sensor_config_path)
    cfg.get_psf_kernel()
    cfg.get_backward_map()
    cfg.get_vignetting_map()
    _WORKER_SENSOR_CONFIG = cfg
    _WORKER_SIMULATORS = {c: CameraSimulator(cfg, f"CAM_{c}") for c in ALL_CAMERAS}


def evaluate_frame_fits(
    edges_by_cam: dict[str, list[dict[str, float]]],
    rig: Rig,
    lat_deg: float,
    lon_deg: float,
    h_true_m: float,
    h_meas_base_m: float,
    yaw_rad: float,
    pitch_true_deg: float,
    roll_true_deg: float,
    alt_noise_dict: dict[float, float],
    frame_k: int,
    base_record: dict[str, Any],
    rng_seed: int,
    sensor_config: SensorConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all camera sets and sigma_h for a single frame edge set."""
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
                p_est = fit["pitch_deg"]
                r_est = fit["roll_deg"]
                status = fit["status"]
                try:
                    inlier_cnt = fit["inlier_count"]
                    inlier_counts_c = fit["inlier_count_by_cam"]
                    rms_urad = fit["rms_residual_urad"]
                    cov_rad2 = fit["covariance_rad2"]

                    p_err = abs(p_est - pitch_true_deg)
                    r_err = abs(r_est - roll_true_deg)
                    tot_err = math.hypot(p_err, r_err)
                    tot_err_rad = math.radians(tot_err)

                    # Trace and sigma_hat
                    if cov_rad2 is None:
                        raise ValueError("cov_rad2 is None")
                    tr_cov = float(np.trace(cov_rad2))
                    sigma_hat_rad = max(1e-6, math.sqrt(max(0.0, tr_cov)))  # floored at 1 urad
                    sigma_hat_urad = sigma_hat_rad * 1e6
                    r_score = tot_err_rad / sigma_hat_rad

                    # Condition number of covariance (D28/W2: no 1e9 fallback on exception)
                    cond_num = float(np.linalg.cond(cov_rad2))

                    # Inlier fractions
                    inlier_frac = inlier_cnt / max(1, batch.n_rays)
                    inlier_fracs_c = {
                        c: inlier_counts_c.get(c, 0) / max(1, len(sub_edges.get(c, [])))  # default-ok: frequency counter accumulator
                        for c in allowed_cams
                    }

                    # Camera consistency (D27)
                    if cam_set_name != "F" and "inliers" in fit and len(fit["inliers"]) > 0:
                        inlier_idx = fit["inliers"]
                        inlier_batch = LimbRayBatch(
                            batch.d_body[inlier_idx],
                            batch.lever_arm_body[inlier_idx],
                            [batch.cam_ids[i] for i in inlier_idx],
                            uvs=batch.uvs[inlier_idx] if batch.uvs is not None else None,
                        )
                        cc_res = camera_consistency(
                            fit, inlier_batch, rig, lat_deg, lon_deg, h_meas, yaw_rad
                        )
                        loo_angle_deg = cc_res.max_angle_deg
                        arg_max_cam = cc_res.arg_max_camera
                    else:
                        loo_angle_deg = float("nan")
                        arg_max_cam = None
                except Exception:
                    # D28 / W2: An exception while computing a feature sets the status to feature_error
                    # and raises an alarm (fail-safe).
                    status = "feature_error"
                    cond_num = float("nan")
                    inlier_cnt = 0
                    inlier_frac = 0.0
                    inlier_fracs_c = {c: 0.0 for c in allowed_cams}
                    rms_urad = float("nan")
                    sigma_hat_rad = 1e-6
                    sigma_hat_urad = 1.0
                    r_score = float("nan")
                    loo_angle_deg = float("nan")
                    arg_max_cam = None
            else:
                p_est = float("nan")
                r_est = float("nan")
                status = "FAILED_NO_EDGES"
                inlier_cnt = 0
                inlier_counts_c = {c: 0 for c in allowed_cams}
                inlier_frac = 0.0
                inlier_fracs_c = {c: 0.0 for c in allowed_cams}
                rms_urad = float("nan")
                cov_rad2 = np.full((2, 2), np.nan)
                p_err = float("nan")
                r_err = float("nan")
                tot_err = float("nan")
                tot_err_rad = float("nan")
                sigma_hat_urad = float("nan")
                r_score = float("nan")
                cond_num = float("nan")
                loo_angle_deg = float("nan")
                arg_max_cam = None

            rec = dict(base_record)
            rec.update({
                "cam_set": cam_set_name,
                "sigma_h": float(sigma_h),
                "h_meas_m": float(h_meas),
                "pitch_est_deg": float(p_est),
                "roll_est_deg": float(r_est),
                "pitch_error_deg": float(p_err),
                "roll_error_deg": float(r_err),
                "total_error_deg": float(tot_err),
                "total_error_rad": float(tot_err_rad),
                "rms_residual_urad": float(rms_urad),
                "inlier_count": int(inlier_cnt),
                "inlier_fraction": float(inlier_frac),
                "inlier_counts_by_cam": inlier_counts_c,
                "inlier_fractions_by_cam": inlier_fracs_c,
                "covariance_rad2": cov_rad2.tolist() if isinstance(cov_rad2, np.ndarray) else cov_rad2,
                "sigma_hat_urad": float(sigma_hat_urad),
                "condition_number": float(cond_num),
                "camera_consistency_deg": None if math.isnan(loo_angle_deg) else float(loo_angle_deg),
                "leave_one_out_consistency_deg": None if math.isnan(loo_angle_deg) else float(loo_angle_deg),
                "camera_consistency_arg_max": arg_max_cam,
                "nonconformity_score": float(r_score),
                "status": status,
            })
            records.append(rec)

    return records


def _process_degraded_frame_worker(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker task processing all conditions (nominal + 4 faults) for a single degraded frame."""
    global _WORKER_SIMULATORS, _WORKER_RIG, _WORKER_SENSOR_CONFIG
    rig = _WORKER_RIG
    sims = _WORKER_SIMULATORS
    sensor_cfg = _WORKER_SENSOR_CONFIG

    seq_name = task["seq_name"]
    k = task["k"]
    t_s = task["t_s"]
    seed_idx = task["seed_idx"]
    split = task["split"]
    clean_cam_paths = task["clean_cam_paths"]
    act = task["aircraft"]
    cams_info = task["cams_info"]
    fault_config = task["fault_config"]
    alt_noise_dict = task["alt_noise_dict"]

    lat_deg = float(act["lat_deg"])
    lon_deg = float(act["lon_deg"])
    h_true = float(act["h_m"])
    yaw_deg = float(act["ypr_deg"][0])
    pitch_true = float(act["ypr_deg"][1])
    roll_true = float(act["ypr_deg"][2])
    yaw_rad = math.radians(yaw_deg)

    # 1. Generate degraded images through WP-03 sensor chain
    image_id = f"{seq_name}_seed{seed_idx}"
    deg_imgs: dict[str, np.ndarray] = {}
    for c in ALL_CAMERAS:
        p = clean_cam_paths.get(c)
        if not p or not Path(p).is_file():
            raise FileNotFoundError(f"Missing clean camera image for camera {c}: {p}")
        img_raw = np.array(Image.open(p))
        res = sims[c].process_frame(img_raw, image_id=image_id, frame_idx=k, warp_method="cv2")
        deg_imgs[c] = res["display"]

    # 2. Extract nominal edges
    nominal_edges: dict[str, list[dict[str, float]]] = {}
    for c in ALL_CAMERAS:
        nominal_edges[c] = detect_camera_edges(
            deg_imgs[c], c, rig.camera_model, step=8, search_band_px=50, min_gradient=0.015
        )

    all_frame_records: list[dict[str, Any]] = []

    # 3. Base metadata
    base_meta = {
        "seq_name": seq_name,
        "split": split,
        "seed_idx": seed_idx,
        "mode": "degraded",
        "condition": "degraded",
        "k": k,
        "t_s": t_s,
        "h_true_m": h_true,
        "pitch_true_deg": pitch_true,
        "roll_true_deg": roll_true,
    }

    # Condition 1: Nominal degraded (no fault)
    nom_meta = dict(base_meta)
    nom_meta.update({
        "fault_type": "none",
        "fault_active": False,
        "k_onset": None,
        "target_cam": None,
    })
    nom_recs = evaluate_frame_fits(
        nominal_edges, rig, lat_deg, lon_deg, h_true, h_true, yaw_rad,
        pitch_true, roll_true, alt_noise_dict, k, nom_meta, rng_seed=20260925,
        sensor_config=sensor_cfg,
    )
    all_frame_records.extend(nom_recs)

    # Condition 2: Occlusion fault
    occ_cfg = fault_config["occlusion"]
    occ_active = (k >= occ_cfg["onset_k"])
    occ_meta = dict(base_meta)
    occ_meta.update({
        "fault_type": "occlusion",
        "fault_active": occ_active,
        "k_onset": occ_cfg["onset_k"],
        "target_cam": occ_cfg["target_cam"],
    })
    if occ_active:
        t_cam = occ_cfg["target_cam"]
        rng_occ = np.random.Generator(np.random.PCG64(occ_cfg["sub_seed"] + k))
        img_occ, _ = inject_occlusion(deg_imgs[t_cam], rng=rng_occ, cam_id=t_cam)
        occ_edges = dict(nominal_edges)
        occ_edges[t_cam] = detect_camera_edges(
            img_occ, t_cam, rig.camera_model, step=8, search_band_px=50, min_gradient=0.015
        )
    else:
        occ_edges = nominal_edges

    occ_recs = evaluate_frame_fits(
        occ_edges, rig, lat_deg, lon_deg, h_true, h_true, yaw_rad,
        pitch_true, roll_true, alt_noise_dict, k, occ_meta, rng_seed=20260925,
        sensor_config=sensor_cfg,
    )
    all_frame_records.extend(occ_recs)

    # Condition 3: Spurious edge fault
    sp_cfg = fault_config["spurious_edge"]
    sp_active = (k >= sp_cfg["onset_k"])
    sp_meta = dict(base_meta)
    sp_meta.update({
        "fault_type": "spurious_edge",
        "fault_active": sp_active,
        "k_onset": sp_cfg["onset_k"],
        "target_cam": sp_cfg["target_cam"],
    })
    if sp_active:
        t_cam = sp_cfg["target_cam"]
        gt_uv = cams_info.get(t_cam, {}).get("gt", {}).get("horizon_uv", [])
        rng_sp = np.random.Generator(np.random.PCG64(sp_cfg["sub_seed"] + k))
        img_sp, _ = inject_spurious_edge(deg_imgs[t_cam], horizon_uv=gt_uv, rng=rng_sp, cam_id=t_cam)
        sp_edges = dict(nominal_edges)
        sp_edges[t_cam] = detect_camera_edges(
            img_sp, t_cam, rig.camera_model, step=8, search_band_px=50, min_gradient=0.015
        )
    else:
        sp_edges = nominal_edges

    sp_recs = evaluate_frame_fits(
        sp_edges, rig, lat_deg, lon_deg, h_true, h_true, yaw_rad,
        pitch_true, roll_true, alt_noise_dict, k, sp_meta, rng_seed=20260925,
        sensor_config=sensor_cfg,
    )
    all_frame_records.extend(sp_recs)

    # Condition 4: Altitude bias fault (h_meas = h_true + 200 m)
    alt_cfg = fault_config["altitude_bias"]
    alt_active = (k >= alt_cfg["onset_k"])
    alt_meta = dict(base_meta)
    alt_meta.update({
        "fault_type": "altitude_bias",
        "fault_active": alt_active,
        "k_onset": alt_cfg["onset_k"],
        "target_cam": None,
    })
    h_meas_alt = (h_true + 200.0) if alt_active else h_true
    alt_recs = evaluate_frame_fits(
        nominal_edges, rig, lat_deg, lon_deg, h_true, h_meas_alt, yaw_rad,
        pitch_true, roll_true, alt_noise_dict, k, alt_meta, rng_seed=20260925,
        sensor_config=sensor_cfg,
    )
    all_frame_records.extend(alt_recs)

    # Condition 5: Glare fault
    gl_cfg = fault_config["glare"]
    gl_active = (k >= gl_cfg["onset_k"])
    gl_meta = dict(base_meta)
    gl_meta.update({
        "fault_type": "glare",
        "fault_active": gl_active,
        "k_onset": gl_cfg["onset_k"],
        "target_cam": gl_cfg["target_cam"],
    })
    if gl_active:
        t_cam = gl_cfg["target_cam"]
        gt_uv = cams_info.get(t_cam, {}).get("gt", {}).get("horizon_uv", [])
        rng_gl = np.random.Generator(np.random.PCG64(gl_cfg["sub_seed"] + k))
        img_gl, _ = inject_glare(deg_imgs[t_cam], horizon_uv=gt_uv, radius_px=150.0, rng=rng_gl, cam_id=t_cam)
        gl_edges = dict(nominal_edges)
        gl_edges[t_cam] = detect_camera_edges(
            img_gl, t_cam, rig.camera_model, step=8, search_band_px=50, min_gradient=0.015
        )
    else:
        gl_edges = nominal_edges

    gl_recs = evaluate_frame_fits(
        gl_edges, rig, lat_deg, lon_deg, h_true, h_true, yaw_rad,
        pitch_true, roll_true, alt_noise_dict, k, gl_meta, rng_seed=20260925,
        sensor_config=sensor_cfg,
    )
    all_frame_records.extend(gl_recs)

    return all_frame_records


def run_clean_evaluation(
    seq_name: str,
    frames_path: Path,
    clean_seq_dir: Path,
    rig: Rig,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate clean G4 imagery once for the calibration split."""
    records = []
    with open(frames_path, "r", encoding="utf-8") as f:
        frame_items = [json.loads(line) for line in f if line.strip()]
    if max_frames is not None:
        frame_items = frame_items[:max_frames]

    for item in frame_items:
        k = item["k"]
        t_s = item["t_s"]
        act = item["aircraft"]
        lat_deg = float(act["lat_deg"])
        lon_deg = float(act["lon_deg"])
        h_true = float(act["h_m"])
        yaw_rad = math.radians(float(act["ypr_deg"][0]))
        pitch_true = float(act["ypr_deg"][1])
        roll_true = float(act["ypr_deg"][2])

        edges: dict[str, list[dict[str, float]]] = {}
        for c in ALL_CAMERAS:
            p = clean_seq_dir / c / f"{k:06d}.png"
            if not p.is_file():
                raise FileNotFoundError(f"Missing clean camera image: {p}")
            img = np.array(Image.open(p))
            edges[c] = detect_camera_edges(img, c, rig.camera_model, step=8, search_band_px=50, min_gradient=0.015)

        base_meta = {
            "seq_name": seq_name,
            "split": "calibration",
            "seed_idx": 0,
            "mode": "clean",
            "condition": "clean",
            "fault_type": "none",
            "fault_active": False,
            "k_onset": None,
            "target_cam": None,
            "k": k,
            "t_s": t_s,
            "h_true_m": h_true,
            "pitch_true_deg": pitch_true,
            "roll_true_deg": roll_true,
        }

        # Deterministic altitude noise
        rng_alt = np.random.default_rng(20260925 + k)
        alt_noise = {
            0.0: 0.0,
            10.0: float(rng_alt.normal(0.0, 10.0)),
            50.0: float(rng_alt.normal(0.0, 50.0)),
        }

        recs = evaluate_frame_fits(
            edges, rig, lat_deg, lon_deg, h_true, h_true, yaw_rad,
            pitch_true, roll_true, alt_noise, k, base_meta, rng_seed=20260925,
            sensor_config=None,
        )
        records.extend(recs)

    return records


def measure_pilot(
    data_dir: Path,
    sensor_config_path: Path,
    max_workers: int = 8,
    pilot_frames: int = 5,
) -> dict[str, Any]:
    """Measure a pilot run on T1_cruise and project wall-clock time."""
    print("\n[PILOT] Measuring execution benchmark on 5 frames ...")
    t1_frames_file = data_dir / "T1_cruise" / "frames.jsonl"
    with open(t1_frames_file, "r", encoding="utf-8") as f:
        pilot_items = [json.loads(next(f)) for _ in range(pilot_frames)]

    injector = SeededFaultInjector(master_seed=20260927)
    fault_cfg = {
        "occlusion": {"onset_k": 2, "target_cam": "F", "sub_seed": 101},
        "spurious_edge": {"onset_k": 2, "target_cam": "R", "sub_seed": 102},
        "altitude_bias": {"onset_k": 2, "sub_seed": 103},
        "glare": {"onset_k": 2, "target_cam": "B", "sub_seed": 104},
    }

    tasks = []
    for item in pilot_items:
        k = item["k"]
        cam_paths = {c: str(data_dir / "T1_cruise" / c / f"{k:06d}.png") for c in ALL_CAMERAS}
        tasks.append({
            "seq_name": "T1_cruise",
            "k": k,
            "t_s": item["t_s"],
            "seed_idx": 1,
            "split": "calibration",
            "clean_cam_paths": cam_paths,
            "aircraft": item["aircraft"],
            "cams_info": item.get("cams", {}),
            "fault_config": fault_cfg,
            "alt_noise_dict": {0.0: 0.0, 10.0: 1.2, 50.0: -5.4},
        })

    # Benchmark single worker core processing
    _init_worker(str(sensor_config_path))
    t0 = time.perf_counter()
    records = []
    for t in tasks:
        records.extend(_process_degraded_frame_worker(t))
    elapsed_single = time.perf_counter() - t0

    sec_per_frame_core = elapsed_single / pilot_frames
    total_g4_frames = 120 + 300 + 120  # 540 frames
    total_frame_instances_20_seeds = total_g4_frames * 20
    projected_wall_20_seeds_sec = (total_frame_instances_20_seeds * sec_per_frame_core) / max_workers
    projected_wall_20_seeds_hours = projected_wall_20_seeds_sec / 3600.0

    total_frame_instances_12_seeds = total_g4_frames * 12
    projected_wall_12_seeds_sec = (total_frame_instances_12_seeds * sec_per_frame_core) / max_workers
    projected_wall_12_seeds_hours = projected_wall_12_seeds_sec / 3600.0

    limit_hours = 8.0
    exceeds = (projected_wall_20_seeds_hours > limit_hours)
    chosen_seeds = 12 if exceeds else 20

    print(f"  Pilot completed {pilot_frames} frames in {elapsed_single:.2f} s")
    print(f"  Single-core rate: {sec_per_frame_core:.3f} s per frame (all 5 conditions)")
    print(f"  Projected wall time (20 seeds, {max_workers} workers): {projected_wall_20_seeds_hours:.2f} h ({projected_wall_20_seeds_sec:.1f} s)")
    print(f"  Projected wall time (12 seeds, {max_workers} workers): {projected_wall_12_seeds_hours:.2f} h ({projected_wall_12_seeds_sec:.1f} s)")

    if exceeds:
        print(f"  >>> PROJECTION EXCEEDS 8 H WALL! Cutting seeds to 12 (splits 1-6, 7-9, 10-12).")
    else:
        print(f"  >>> Projection is <= 8 h wall. Keeping full 20 seeds (splits 1-10, 11-15, 16-20).")

    return {
        "pilot_frames": pilot_frames,
        "elapsed_single_sec": elapsed_single,
        "sec_per_frame_core": sec_per_frame_core,
        "workers": max_workers,
        "projected_wall_20_seeds_hours": projected_wall_20_seeds_hours,
        "projected_wall_12_seeds_hours": projected_wall_12_seeds_hours,
        "limit_hours": limit_hours,
        "exceeds_limit": bool(exceeds),
        "chosen_seeds": chosen_seeds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WP-07 G0 Integrity Dataset")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sensor-config", type=Path, default=DEFAULT_SENSOR_CONFIG)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 12) - 4)),
        help="Multiprocessing worker cap (logical cores - 4)",
    )
    parser.add_argument("--force-seeds", type=int, default=None, help="Force specific number of seeds (e.g. 12 or 20)")
    parser.add_argument("--pilot-only", action="store_true", help="Run only the pilot benchmark and exit")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per sequence (testing only)")
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    data_root = get_data_root(args.data_root)
    g4_dir = data_root / "xvs" / "wp01" / "g4"
    if not g4_dir.is_dir():  # D19: no fallback to a machine path or another data root (merge of G-B13)
        raise FileNotFoundError(f"WP-01 G4 sequences not found under the data root: {g4_dir}")
    out_dir = args.out_dir or (data_root / "xvs" / "wp07" / "g0")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _file_sha256(p: Path) -> str:
        if not p.is_file():
            return ""
        h = hashlib.sha256()
        with open(p, "rb") as f_in:
            while chunk := f_in.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    tracked_code_files = [
        "scripts/xvs/integrity/make_wp07_data.py",
        "scripts/xvs/integrity/run_wp07_g0.py",
        "src/xvs/integrity/monitor.py",
        "src/xvs/integrity/conformal.py",
        "src/xvs/integrity/faults.py",
        "src/xvs/horizon_att/edges.py",
        "src/xvs/horizon_att/fit.py",
        "src/xvs/horizon_att/pipeline.py",
        "src/xvs/horizon_att/limb_model.py",
        "src/xvs/rig/rig.py",
        "src/xvs/sensor/chain.py",
    ]
    tracked_config_files = [
        str(Path(args.sensor_config).resolve().relative_to(REPO_ROOT)),
        "configs/xvs/rig/nominal_6cam.json",
    ]

    def _capture_snapshot() -> tuple[str, list[str], dict[str, str], dict[str, str]]:
        c_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
        g_stat = [
            ln for ln in subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.splitlines()
            if ln.strip()
        ]
        c_hashes = {rel: _file_sha256(REPO_ROOT / rel) for rel in tracked_code_files if (REPO_ROOT / rel).is_file()}
        cfg_hashes = {rel: _file_sha256(REPO_ROOT / rel) for rel in tracked_config_files if (REPO_ROOT / rel).is_file()}
        return c_head, g_stat, c_hashes, cfg_hashes

    start_time_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_commit, git_status_start, code_hashes_start, configs_sha256_start = _capture_snapshot()

    print("=" * 70)
    print("OpenXVS WP-07 Gate G0 — Dataset Generator")
    print(f"Data Root:       {data_root} (source: {data_root_source(args.data_root)})")
    print(f"G4 Directory:    {g4_dir}")
    print(f"Output Directory:{out_dir}")
    print(f"Sensor Config:   {args.sensor_config}")
    print(f"Max Workers:     {args.max_workers} (cap: {(os.cpu_count() or 12) - 4})")
    print("=" * 70)

    # 1. Pilot benchmark
    pilot_res = measure_pilot(
        g4_dir,
        args.sensor_config,
        max_workers=args.max_workers,
        pilot_frames=5,
    )
    pilot_file = out_dir / "pilot_projection.json"
    with open(pilot_file, "w", encoding="utf-8") as f:
        json.dump(pilot_res, f, indent=2)
    print(f"Saved pilot projection to {pilot_file}")

    if args.pilot_only:
        return 0

    if args.force_seeds is not None:
        n_seeds = args.force_seeds
        print(f"Forcing seed count to {n_seeds}")
    else:
        n_seeds = pilot_res["chosen_seeds"]

    if n_seeds == 12:
        cal_seeds = set(range(1, 7))
        val_seeds = set(range(7, 10))
        test_seeds = set(range(10, 13))
        seed_list = list(range(1, 13))
    elif n_seeds == 20:
        cal_seeds = set(range(1, 11))
        val_seeds = set(range(11, 16))
        test_seeds = set(range(16, 21))
        seed_list = list(range(1, 21))
    else:
        seed_list = list(range(1, n_seeds + 1))
        c_end = max(1, n_seeds // 2)
        v_end = max(c_end + 1, n_seeds - 1)
        cal_seeds = set(range(1, c_end + 1))
        val_seeds = set(range(c_end + 1, v_end + 1))
        test_seeds = set(range(v_end + 1, n_seeds + 1))

    print(f"\nActive Seeds ({len(seed_list)}): {seed_list}")
    print(f"  Calibration seeds: {sorted(list(cal_seeds))}")
    print(f"  Validation seeds:  {sorted(list(val_seeds))}")
    print(f"  Test seeds:        {sorted(list(test_seeds))}")

    # Prepare output JSONL files
    cal_file = out_dir / "dataset_calibration.jsonl"
    val_file = out_dir / "dataset_validation.jsonl"
    test_file = out_dir / "dataset_test.jsonl"
    fault_log_file = out_dir / "fault_log.json"

    for f_path in (cal_file, val_file, test_file):
        if f_path.is_file():
            f_path.unlink()

    f_cal = open(cal_file, "a", encoding="utf-8")
    f_val = open(val_file, "a", encoding="utf-8")
    f_test = open(test_file, "a", encoding="utf-8")

    def write_records(recs: list[dict[str, Any]]) -> None:
        for r in recs:
            line = json.dumps(r) + "\n"
            sp = r["split"]
            if sp == "calibration":
                f_cal.write(line)
            elif sp == "validation":
                f_val.write(line)
            elif sp == "test":
                f_test.write(line)
        f_cal.flush()
        f_val.flush()
        f_test.flush()

    # 2. Clean evaluation once (all clean frames go to calibration)
    print("\n---> Running clean evaluation once (calibration split)...", flush=True)
    rig = load_pipeline_rig()
    clean_count = 0
    t_clean_start = time.perf_counter()
    for seq in ("T1_cruise", "T2_ksfo28l", "T3_turn_climb"):
        clean_seq_dir = g4_dir / seq
        frames_path = clean_seq_dir / "frames.jsonl"
        recs = run_clean_evaluation(seq, frames_path, clean_seq_dir, rig, max_frames=args.max_frames)
        write_records(recs)
        clean_count += len(recs)
        print(f"  Clean {seq}: generated {len(recs)} records", flush=True)
    print(f"Clean evaluation completed: {clean_count} records in {time.perf_counter() - t_clean_start:.1f} s", flush=True)

    # 3. Degraded evaluation across active seeds
    injector = SeededFaultInjector(master_seed=20260927)
    fault_logs: list[dict[str, Any]] = []

    print(f"\n---> Running degraded evaluations with {args.max_workers} worker processes ...")
    t_deg_start = time.perf_counter()
    total_degraded_records = 0

    pool = mp.Pool(
        processes=args.max_workers,
        initializer=_init_worker,
        initargs=(str(args.sensor_config),),
    )

    try:
        for seed_idx in seed_list:
            t_seed_start = time.perf_counter()
            if seed_idx in cal_seeds:
                split = "calibration"
            elif seed_idx in val_seeds:
                split = "validation"
            else:
                split = "test"

            print(f"\n[Seed {seed_idx:02d} / {n_seeds}] Split: {split.upper()} ...")
            tasks = []

            for seq in ("T1_cruise", "T2_ksfo28l", "T3_turn_climb"):
                seq_dir = g4_dir / seq
                frames_file = seq_dir / "frames.jsonl"
                with open(frames_file, "r", encoding="utf-8") as f:
                    frame_items = [json.loads(line) for line in f if line.strip()]

                if args.max_frames is not None:
                    frame_items = frame_items[: args.max_frames]

                n_f = len(frame_items)
                onset_occ = injector.choose_onset_frame(n_f, seq, "occlusion", seed_idx)
                cam_occ = injector.choose_target_camera(seq, "occlusion", seed_idx)
                onset_sp = injector.choose_onset_frame(n_f, seq, "spurious_edge", seed_idx)
                cam_sp = injector.choose_target_camera(seq, "spurious_edge", seed_idx)
                onset_alt = injector.choose_onset_frame(n_f, seq, "altitude_bias", seed_idx)
                onset_gl = injector.choose_onset_frame(n_f, seq, "glare", seed_idx)
                cam_gl = injector.choose_target_camera(seq, "glare", seed_idx)

                fault_cfg = {
                    "occlusion": {"onset_k": onset_occ, "target_cam": cam_occ, "sub_seed": 1000 * seed_idx + 1},
                    "spurious_edge": {"onset_k": onset_sp, "target_cam": cam_sp, "sub_seed": 1000 * seed_idx + 2},
                    "altitude_bias": {"onset_k": onset_alt, "sub_seed": 1000 * seed_idx + 3},
                    "glare": {"onset_k": onset_gl, "target_cam": cam_gl, "sub_seed": 1000 * seed_idx + 4},
                }

                fault_logs.append({
                    "seed_idx": seed_idx,
                    "split": split,
                    "seq_name": seq,
                    "n_frames": n_f,
                    "faults": fault_cfg,
                })

                for item in frame_items:
                    k = item["k"]
                    cam_paths = {c: str(seq_dir / c / f"{k:06d}.png") for c in ALL_CAMERAS}

                    rng_alt = np.random.default_rng(20260925 + 1000 * seed_idx + k)
                    alt_noise = {
                        0.0: 0.0,
                        10.0: float(rng_alt.normal(0.0, 10.0)),
                        50.0: float(rng_alt.normal(0.0, 50.0)),
                    }

                    tasks.append({
                        "seq_name": seq,
                        "k": k,
                        "t_s": item["t_s"],
                        "seed_idx": seed_idx,
                        "split": split,
                        "clean_cam_paths": cam_paths,
                        "aircraft": item["aircraft"],
                        "cams_info": item.get("cams", {}),
                        "fault_config": fault_cfg,
                        "alt_noise_dict": alt_noise,
                    })

            # Process all frames for this seed in parallel chunked
            seed_recs_count = 0
            for frame_recs in pool.imap_unordered(_process_degraded_frame_worker, tasks, chunksize=4):
                write_records(frame_recs)
                seed_recs_count += len(frame_recs)

            total_degraded_records += seed_recs_count
            elapsed_seed = time.perf_counter() - t_seed_start
            print(f"  Seed {seed_idx} done: {seed_recs_count} records in {elapsed_seed:.1f} s ({seed_recs_count / max(0.1, elapsed_seed):.1f} rec/s)")

    finally:
        pool.close()
        pool.join()
        f_cal.close()
        f_val.close()
        f_test.close()

    # Save fault log
    with open(fault_log_file, "w", encoding="utf-8") as f:
        json.dump(fault_logs, f, indent=2)

    total_time = time.perf_counter() - t_deg_start

    def get_file_sha256(p: Path) -> str:
        if not p.is_file():
            return ""
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    file_hashes = {
        "calibration": get_file_sha256(cal_file),
        "validation": get_file_sha256(val_file),
        "test": get_file_sha256(test_file),
        "fault_log": get_file_sha256(fault_log_file),
        "pilot": get_file_sha256(pilot_file),
    }

    code_hashes = {}
    for rel_c in [
        "scripts/xvs/integrity/make_wp07_data.py",
        "scripts/xvs/integrity/run_wp07_g0.py",
        "src/xvs/integrity/monitor.py",
        "src/xvs/integrity/conformal.py",
        "src/xvs/integrity/faults.py",
        "src/xvs/horizon_att/edges.py",
        "src/xvs/horizon_att/fit.py",
        "src/xvs/horizon_att/pipeline.py",
        "src/xvs/rig/rig.py",
        "src/xvs/sensor/chain.py",
    ]:
        c_path = REPO_ROOT / rel_c
        if c_path.is_file():
            code_hashes[rel_c] = get_file_sha256(c_path)

    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        git_commit = proc.stdout.strip()
    except Exception:
        git_commit = ""

    end_time_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    end_commit, git_status_end, code_hashes_end, configs_sha256_end = _capture_snapshot()
    hashes_unchanged = (
        code_hashes_start == code_hashes_end
        and configs_sha256_start == configs_sha256_end
        and start_commit == end_commit
    )
    if not hashes_unchanged:
        raise RuntimeError(
            "G-B14 §1 long-job check failed: code or config hashes changed during make_wp07_data.py execution."
        )

    manifest = {
        "schema": "xvs.wp07_data_manifest/1",
        "created_at": end_time_utc,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "commit": end_commit,
        "start_commit": start_commit,
        "end_commit": end_commit,
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "argv": list(sys.argv),
        "total_seeds": n_seeds,
        "calibration_seeds": sorted(list(cal_seeds)),
        "validation_seeds": sorted(list(val_seeds)),
        "test_seeds": sorted(list(test_seeds)),
        "clean_records": clean_count,
        "degraded_records": total_degraded_records,
        "total_records": clean_count + total_degraded_records,
        "total_wall_sec": total_time,
        "files": {
            "calibration": str(cal_file.relative_to(REPO_ROOT)),
            "validation": str(val_file.relative_to(REPO_ROOT)),
            "test": str(test_file.relative_to(REPO_ROOT)),
            "fault_log": str(fault_log_file.relative_to(REPO_ROOT)),
            "pilot": str(pilot_file.relative_to(REPO_ROOT)),
        },
        "file_sha256": file_hashes,
        "code_hashes": code_hashes_end,
        "code_hashes_start": code_hashes_start,
        "code_hashes_end": code_hashes_end,
        "configs_read_sha256_start": configs_sha256_start,
        "configs_read_sha256_end": configs_sha256_end,
        "code_hashes_unchanged_during_run": hashes_unchanged,
    }

    manifest_file = out_dir / "data_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Generate sample contact sheets for Director evidence
    contact_dir = out_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    try:
        sample_img_path = g4_dir / "T1_cruise" / "F" / "000000.png"
        if sample_img_path.is_file():
            img_raw = np.array(Image.open(sample_img_path))
            cfg = SensorConfig.from_file(args.sensor_config)
            sim = CameraSimulator(cfg, "CAM_F")
            nom_deg = sim.process_frame(img_raw, "contact_test", 0, warp_method="cv2")["display"]

            # Occlusion
            occ_img, _ = inject_occlusion(nom_deg, cam_id="F", col_start=200)
            sheet_occ = np.hstack([nom_deg, occ_img])
            Image.fromarray(sheet_occ).save(contact_dir / "contact_occlusion.png")

            # Spurious edge
            t1_frames = data_root / "xvs" / "wp01" / "g4" / "T1_cruise" / "frames.jsonl"
            with open(t1_frames, "r", encoding="utf-8") as f:
                first_f = json.loads(next(f))
            gt_uv = first_f.get("cams", {}).get("F", {}).get("gt", {}).get("horizon_uv", [])
            sp_img, _ = inject_spurious_edge(nom_deg, horizon_uv=gt_uv, offset_px=20.0, cam_id="F")
            sheet_sp = np.hstack([nom_deg, sp_img])
            Image.fromarray(sheet_sp).save(contact_dir / "contact_spurious_edge.png")

            # Glare
            gl_img, _ = inject_glare(nom_deg, horizon_uv=gt_uv, radius_px=150.0, cam_id="F")
            sheet_gl = np.hstack([nom_deg, gl_img])
            Image.fromarray(sheet_gl).save(contact_dir / "contact_glare.png")
            print(f"Saved Director contact sheets to {contact_dir}")
    except Exception as e:
        print(f"Warning: could not write contact sheets: {e}")

    print("\n" + "=" * 70)
    print(f"WP-07 G0 Dataset Generation COMPLETE in {total_time / 60:.1f} minutes")
    print(f"Total records generated: {clean_count + total_degraded_records}")
    print(f"  Calibration records:   {clean_count + (len(cal_seeds) * (total_degraded_records // n_seeds))}")
    print(f"  Validation records:    {len(val_seeds) * (total_degraded_records // n_seeds)}")
    print(f"  Test records:          {len(test_seeds) * (total_degraded_records // n_seeds)}")
    print(f"Manifest written to:     {manifest_file}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    # Windows multiprocessing freeze support
    mp.freeze_support()
    sys.exit(main())
