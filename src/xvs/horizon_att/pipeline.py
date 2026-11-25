"""Sequence processing pipeline for G4 evaluation (clean and degraded).

Conforms strictly to docs/xvs/WP06_HORIZON.md v0.1 §2, §3 and GEMINI_G-B4.md.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from PIL import Image
import numpy as np

from xvs.rig.rig import Rig
from xvs.sensor.chain import SensorConfig
from xvs.horizon_att.edges import detect_camera_edges, edges_to_ray_batch
from xvs.horizon_att.fit import ransac_fit_attitude
from xvs.horizon_att.limb_model import LimbRayBatch

CAMERA_SETS: dict[str, tuple[str, ...]] = {
    "F": ("F",),
    "horizontal": ("F", "R", "B", "L"),
    "all": ("F", "R", "B", "L", "U", "D"),
}

SIGMA_H_SET: tuple[float, ...] = (0.0, 10.0, 50.0)
DEFAULT_RIG_SEED: int = 20260922


def load_pipeline_rig(rig_kind: str = "true", rig_seed: int = DEFAULT_RIG_SEED) -> Rig:
    """Load the calibrated rig used by the WP-06 horizon attitude pipeline."""
    rig = Rig.load_nominal()
    if rig_kind == "true":
        rig = rig.make_true(rig_seed)
    return rig


def build_pipeline_ray_batch(
    edges_by_cam: dict[str, list[dict[str, float]]],
    allowed_cams: Sequence[str],
    rig: Rig,
    sensor_config: SensorConfig | None = None,
) -> LimbRayBatch:
    """Build a LimbRayBatch for allowed_cams using the WP-06 pipeline convention."""
    sub_edges = {c: edges_by_cam.get(c, []) for c in allowed_cams}
    return edges_to_ray_batch(sub_edges, rig, sensor_config=sensor_config)


def fit_pipeline_attitude(
    lat_deg: float,
    lon_deg: float,
    h_meas: float,
    yaw_rad: float,
    batch: LimbRayBatch,
    frame_k: int,
    rng_seed_base: int = 20260925,
) -> dict[str, Any]:
    """Run the WP-06 RANSAC + GN attitude fit for a single frame and ray batch."""
    return ransac_fit_attitude(
        lat_deg,
        lon_deg,
        h_meas,
        yaw_rad,
        batch,
        initial_guess_rad=(0.0, 0.0),
        max_iterations=40,
        inlier_threshold_rad=math.radians(0.3),
        min_inliers=3,
        rng_seed=rng_seed_base + frame_k,
    )



@dataclass
class FrameEvaluation:
    k: int
    t_s: float
    cam_set: str
    sigma_h: float
    h_true_m: float
    h_meas_m: float
    pitch_true_deg: float
    roll_true_deg: float
    pitch_est_deg: float
    roll_est_deg: float
    pitch_error_deg: float
    roll_error_deg: float
    total_error_deg: float
    rms_residual_urad: float
    inlier_count: int
    inlier_counts_by_cam: dict[str, int]
    status: str
    ms_cpu: float


def generate_altitude_noise(seed: int, n_frames: int, sigma_h: float) -> np.ndarray:
    """Generate deterministic normal altitude measurement errors."""
    if sigma_h == 0.0:
        return np.zeros(n_frames, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma_h, size=n_frames)


def process_single_frame(
    frame_idx: int,
    frame_data: dict[str, Any],
    seq_dir: str | Path,
    rig: Rig,
    step: int = 8,
    search_band_px: int = 50,
    min_gradient: float = 0.015,
    sensor_config: SensorConfig | None = None,
    alt_noise_by_sigma: dict[float, float] | None = None,
    degraded_dir: str | Path | None = None,
) -> list[FrameEvaluation]:
    """Process a single frame across all camera sets and sigma_h configurations."""
    t0 = time.perf_counter()
    k = frame_data["k"]
    t_s = frame_data["t_s"]
    act = frame_data["aircraft"]

    lat_deg = float(act["lat_deg"])
    lon_deg = float(act["lon_deg"])
    h_true = float(act["h_m"])
    yaw_deg = float(act["ypr_deg"][0])
    pitch_true = float(act["ypr_deg"][1])
    roll_true = float(act["ypr_deg"][2])
    yaw_rad = math.radians(yaw_deg)

    # Load images and detect edges for all 6 cameras
    cam_model = rig.camera_model
    edges_by_cam: dict[str, list[dict[str, float]]] = {}

    for c in ("F", "R", "B", "L", "U", "D"):
        if degraded_dir is not None:
            img_path = Path(degraded_dir) / c / f"{k:06d}.png"
        else:
            img_path = Path(seq_dir) / c / f"{k:06d}.png"

        if not img_path.is_file():
            edges_by_cam[c] = []
            continue

        img = np.array(Image.open(img_path))
        # Operational edge detection: use unconstrained global search (no GT peeking)
        edges = detect_camera_edges(
            img, c, cam_model,
            prior_pose=None,
            step=step,
            search_band_px=search_band_px,
            min_gradient=min_gradient,
        )
        edges_by_cam[c] = edges

    t_edge = (time.perf_counter() - t0) * 1000.0

    evals: list[FrameEvaluation] = []

    # Evaluate each camera set and each sigma_h
    for cam_set_name, allowed_cams in CAMERA_SETS.items():
        batch = build_pipeline_ray_batch(edges_by_cam, allowed_cams, rig, sensor_config=sensor_config)

        for sigma_h in SIGMA_H_SET:
            t_fit_start = time.perf_counter()
            err_h = alt_noise_by_sigma.get(sigma_h, 0.0) if alt_noise_by_sigma else 0.0  # default-ok: optional fallback default
            h_meas = max(1.0, h_true + err_h)

            if batch.n_rays >= 3:
                fit = fit_pipeline_attitude(
                    lat_deg, lon_deg, h_meas, yaw_rad, batch,
                    frame_k=k, rng_seed_base=20260925,
                )
                p_est = fit["pitch_deg"]
                r_est = fit["roll_deg"]
                status = fit["status"]
                inlier_cnt = fit["inlier_count"]
                inlier_by_c = fit["inlier_count_by_cam"]
                rms_urad = fit["rms_residual_urad"]
                p_err = abs(p_est - pitch_true)
                r_err = abs(r_est - roll_true)
                tot_err = math.hypot(p_err, r_err)
            else:
                p_est = float("nan")
                r_est = float("nan")
                status = "FAILED_NO_EDGES"
                inlier_cnt = 0
                inlier_by_c = {c: 0 for c in allowed_cams}
                rms_urad = float("nan")
                p_err = float("nan")
                r_err = float("nan")
                tot_err = float("nan")

            t_fit = (time.perf_counter() - t_fit_start) * 1000.0
            total_cpu_ms = t_edge + t_fit

            evals.append(
                FrameEvaluation(
                    k=k,
                    t_s=t_s,
                    cam_set=cam_set_name,
                    sigma_h=sigma_h,
                    h_true_m=h_true,
                    h_meas_m=h_meas,
                    pitch_true_deg=pitch_true,
                    roll_true_deg=roll_true,
                    pitch_est_deg=p_est,
                    roll_est_deg=r_est,
                    pitch_error_deg=p_err,
                    roll_error_deg=r_err,
                    total_error_deg=tot_err,
                    rms_residual_urad=rms_urad,
                    inlier_count=inlier_cnt,
                    inlier_counts_by_cam=inlier_by_c,
                    status=status,
                    ms_cpu=total_cpu_ms,
                )
            )

    return evals


def _worker_process_frame(args: tuple) -> list[dict[str, Any]]:
    """Helper entry point for multiprocessing pool."""
    (
        frame_idx,
        frame_data,
        seq_dir_str,
        rig_kind,
        rig_seed,
        step,
        search_band_px,
        min_gradient,
        sensor_config_path,
        noise_dict,
        degraded_dir_str,
    ) = args

    rig = load_pipeline_rig(rig_kind=rig_kind, rig_seed=rig_seed)

    sensor_config = None
    if sensor_config_path is not None:
        sensor_config = SensorConfig.from_file(sensor_config_path)

    results = process_single_frame(
        frame_idx=frame_idx,
        frame_data=frame_data,
        seq_dir=Path(seq_dir_str),
        rig=rig,
        step=step,
        search_band_px=search_band_px,
        min_gradient=min_gradient,
        sensor_config=sensor_config,
        alt_noise_by_sigma=noise_dict,
        degraded_dir=Path(degraded_dir_str) if degraded_dir_str else None,
    )
    return [asdict(r) for r in results]


def run_sequence_evaluation(
    seq_name: str,
    seq_dir: str | Path,
    out_dir: str | Path,
    rig_seed: int = 20260922,
    noise_seed_base: int = 20260925,
    max_workers: int = 8,
    sensor_config_path: str | Path | None = None,
    degraded_dir: str | Path | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Execute evaluation over all frames of a sequence.

    Args:
        seq_name: Sequence identifier (e.g. 'T1_cruise', 'T2_ksfo28l', 'T3_turn_climb').
        seq_dir: Path to directory containing frames.jsonl and camera folders.
        out_dir: Output directory for saving sequence results JSON.
        rig_seed: Seed for rig_true.
        noise_seed_base: Seed for altitude measurement noise generator.
        max_workers: Multiprocessing pool workers (capped at logical cores - 4).
        sensor_config_path: Path to sensor config JSON for lens undistortion.
        degraded_dir: Path to degraded image directory, or None for clean images.
        max_frames: Optional frame limit for testing.

    Returns:
        Structured summary dictionary.
    """
    seq_dir = Path(seq_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_file = seq_dir / "frames.jsonl"
    with open(frames_file, "r", encoding="utf-8") as f:
        all_frames = [json.loads(line) for line in f]

    if max_frames is not None:
        all_frames = all_frames[:max_frames]

    n_frames = len(all_frames)

    # Pre-generate altitude measurement noise for each sigma_h
    noise_by_sigma: dict[float, np.ndarray] = {}
    for s_idx, s in enumerate(SIGMA_H_SET):
        noise_by_sigma[s] = generate_altitude_noise(
            seed=noise_seed_base + int(s * 100) + s_idx,
            n_frames=n_frames,
            sigma_h=s,
        )

    # Build tasks for worker pool
    tasks = []
    for idx, frame_data in enumerate(all_frames):
        k = frame_data["k"]
        frame_noise = {s: float(noise_by_sigma[s][idx]) for s in SIGMA_H_SET}
        tasks.append((
            idx,
            frame_data,
            str(seq_dir),
            "true",
            rig_seed,
            8,   # step
            50,  # search_band_px
            0.015,
            str(sensor_config_path) if sensor_config_path else None,
            frame_noise,
            str(degraded_dir) if degraded_dir else None,
        ))

    # Cap workers at min(max_workers, os.cpu_count() - 4)
    cpu_limit = max(1, (os.cpu_count() or 12) - 4)
    actual_workers = min(max_workers, cpu_limit)

    t_start = time.perf_counter()
    if actual_workers > 1:
        with mp.Pool(actual_workers) as pool:
            nested_results = pool.map(_worker_process_frame, tasks)
    else:
        nested_results = [_worker_process_frame(t) for t in tasks]

    t_elapsed = time.perf_counter() - t_start

    # Flatten results
    all_evals = [item for sublist in nested_results for item in sublist]

    # Aggregate statistics by (cam_set, sigma_h)
    stats: dict[str, dict[str, Any]] = {}
    for cam_set in CAMERA_SETS:
        for s in SIGMA_H_SET:
            key = f"{cam_set}_sigma{int(s)}"
            matched = [
                e for e in all_evals if e["cam_set"] == cam_set and e["sigma_h"] == s
            ]
            if len(matched) == 0:
                continue

            valid_p = [e["pitch_error_deg"] for e in matched if not math.isnan(e["pitch_error_deg"])]
            valid_r = [e["roll_error_deg"] for e in matched if not math.isnan(e["roll_error_deg"])]
            valid_tot = [e["total_error_deg"] for e in matched if not math.isnan(e["total_error_deg"])]
            cpus = [e["ms_cpu"] for e in matched]
            failures = sum(1 for e in matched if e["status"] not in ("CONVERGED", "SUCCESS"))

            stats[key] = {
                "cam_set": cam_set,
                "sigma_h": s,
                "n_frames": len(matched),
                "pitch_error_median_deg": float(np.median(valid_p)) if valid_p else float("nan"),
                "pitch_error_p95_deg": float(np.percentile(valid_p, 95)) if valid_p else float("nan"),
                "pitch_error_max_deg": float(np.max(valid_p)) if valid_p else float("nan"),
                "roll_error_median_deg": float(np.median(valid_r)) if valid_r else float("nan"),
                "roll_error_p95_deg": float(np.percentile(valid_r, 95)) if valid_r else float("nan"),
                "roll_error_max_deg": float(np.max(valid_r)) if valid_r else float("nan"),
                "total_error_median_deg": float(np.median(valid_tot)) if valid_tot else float("nan"),
                "total_error_p95_deg": float(np.percentile(valid_tot, 95)) if valid_tot else float("nan"),
                "total_error_max_deg": float(np.max(valid_tot)) if valid_tot else float("nan"),
                "failures": failures,
                "failure_rate": float(failures) / len(matched),
                "cpu_ms_median": float(np.median(cpus)),
                "cpu_ms_p95": float(np.percentile(cpus, 95)),
            }

    mode_str = "degraded" if degraded_dir else "clean"
    out_file = out_dir / f"{seq_name}_{mode_str}_results.json"
    result_doc = {
        "schema": "xvs.horizon_evaluation/1",
        "seq_name": seq_name,
        "mode": mode_str,
        "n_frames": n_frames,
        "workers": actual_workers,
        "elapsed_sec": t_elapsed,
        "mean_fps": n_frames / max(1e-3, t_elapsed),
        "stats": stats,
        "frames": all_evals,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2)

    return result_doc
