"""Measure WP-06 Gauss-Newton attitude solver dip regression slopes on analytic tangent rays.

Evaluates the WP-06 attitude solver on exact WGS84 analytic tangent rays (no images)
for the level cone benchmark (h in {125, 1648, 3000} m, yaw=30 deg) and the three G4
trajectories (T1_cruise, T2_ksfo28l, T3_turn_climb) across camera sets F and FRBL
(horizontal), using the pipeline's own functions (load_pipeline_rig,
generate_synthetic_limb_rays, fit_pipeline_attitude, generate_altitude_noise).

Writes output/xvs/wp06/g0/solver_dip_slopes.json with commit, git_status, argv,
code_hashes, and inputs_sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.geo import wgs84
from xvs.horizon_att.limb_model import LimbRayBatch, _S_DIAG, generate_synthetic_limb_rays
from xvs.horizon_att.pipeline import (
    CAMERA_SETS,
    SIGMA_H_SET,
    fit_pipeline_attitude,
    generate_altitude_noise,
    load_pipeline_rig,
)
from xvs.paths import get_data_root

SPEC_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "horizon_att" / "g0_spec.json"
DEFAULT_RIG_SEED = 20260922
DEFAULT_NOISE_SEED_BASE = 20260925
TRACKED_CODE_FILES = (
    "scripts/xvs/horizon_att/measure_solver_dip_slopes.py",
    "src/xvs/horizon_att/pipeline.py",
    "src/xvs/horizon_att/limb_model.py",
    "src/xvs/horizon_att/fit.py",
)


def _get_git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    return proc.stdout.strip()


def _get_git_status_porcelain() -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    return proc.stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in TRACKED_CODE_FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            out[rel] = _sha256_file(p)
    return out


def compute_cone_dip_deg(lat_deg: float, lon_deg: float, h_m: float) -> float:
    """Compute WGS84 Earth-limb tangent cone dip angle in degrees."""
    p_ecef = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)
    c_prime = p_ecef * _S_DIAG
    k = float(np.linalg.norm(c_prime))
    k = max(1.0 + 1e-12, k)
    return math.degrees(math.acos(1.0 / k))


def _filter_batch_to_cam_set(batch: LimbRayBatch, allowed_cams: tuple[str, ...]) -> LimbRayBatch:
    allowed = set(allowed_cams)
    idx = [i for i, cname in enumerate(batch.cam_ids) if cname in allowed]
    return LimbRayBatch(
        d_body=batch.d_body[idx],
        lever_arm_body=batch.lever_arm_body[idx],
        cam_ids=[batch.cam_ids[i] for i in idx],
        weights=batch.weights[idx] if batch.weights is not None else None,
        uvs=batch.uvs[idx] if batch.uvs is not None else None,
    )


def _regress_through_origin(xs: list[float], ys: list[float]) -> dict[str, Any]:
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    n = int(len(xs_arr))
    denom = float(np.sum(xs_arr**2))
    if n == 0 or denom <= 0.0:
        return {
            "slope_k": float("nan"),
            "ci95_half_width": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "r2": float("nan"),
            "n": n,
        }
    k_val = float(np.sum(xs_arr * ys_arr) / denom)
    residuals = ys_arr - k_val * xs_arr
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum(ys_arr**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    if n > 1:
        s_err = float(math.sqrt(ss_res / (n - 1)) / math.sqrt(denom))
    else:
        s_err = 0.0
    ci95 = float(1.96 * s_err)
    return {
        "slope_k": k_val,
        "ci95_half_width": ci95,
        "ci95": [float(k_val - ci95), float(k_val + ci95)],
        "r2": r2,
        "n": n,
    }


def _evaluate_level_cone(rig: Any, spec_cfg: dict[str, Any]) -> dict[str, Any]:
    tgrid = spec_cfg["trajectory_grid"]
    altitudes_m = [
        float(tgrid["h_cone_1"]),
        float(tgrid["h_cone_2"]),
        float(tgrid["h_cone_3"]),
    ]
    yaw_deg = float(tgrid["yaw_cone"])
    yaw_rad = math.radians(yaw_deg)
    lat_deg, lon_deg = 37.05, -120.85
    dh_perturbations = (-50.0, -25.0, -10.0, 10.0, 25.0, 50.0)

    cam_set_keys = {"F": CAMERA_SETS["F"], "FRBL": CAMERA_SETS["horizontal"]}
    by_cam_set: dict[str, Any] = {}

    for cs_label, allowed_cams in cam_set_keys.items():
        slopes_per_alt: dict[str, dict[str, Any]] = {}
        all_xs: list[float] = []
        all_ys: list[float] = []

        for h_idx, h_true in enumerate(altitudes_m):
            full_batch = generate_synthetic_limb_rays(
                rig=rig,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                h_m=h_true,
                yaw_rad=yaw_rad,
                pitch_rad=0.0,
                roll_rad=0.0,
            )
            sub_batch = _filter_batch_to_cam_set(full_batch, allowed_cams)
            fit_0 = fit_pipeline_attitude(
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                h_meas=h_true,
                yaw_rad=yaw_rad,
                batch=sub_batch,
                frame_k=h_idx * 10,
                rng_seed_base=DEFAULT_NOISE_SEED_BASE,
            )
            pitch_0_deg = float(fit_0["pitch_deg"])
            dip_true = compute_cone_dip_deg(lat_deg, lon_deg, h_true)

            xs_h: list[float] = []
            ys_h: list[float] = []
            for p_idx, dh in enumerate(dh_perturbations):
                h_meas = max(1.0, h_true + dh)
                fit_m = fit_pipeline_attitude(
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    h_meas=h_meas,
                    yaw_rad=yaw_rad,
                    batch=sub_batch,
                    frame_k=h_idx * 10 + p_idx + 1,
                    rng_seed_base=DEFAULT_NOISE_SEED_BASE,
                )
                pitch_m_deg = float(fit_m["pitch_deg"])
                x = compute_cone_dip_deg(lat_deg, lon_deg, h_meas) - dip_true
                y = pitch_m_deg - pitch_0_deg
                xs_h.append(x)
                ys_h.append(y)
                all_xs.append(x)
                all_ys.append(y)

            slopes_per_alt[f"{int(round(h_true))}"] = _regress_through_origin(xs_h, ys_h)

        k_vals = [v["slope_k"] for v in slopes_per_alt.values()]
        by_cam_set[cs_label] = {
            "combined": _regress_through_origin(all_xs, all_ys),
            "by_altitude_m": slopes_per_alt,
            "slope_k_min": float(min(k_vals)),
            "slope_k_max": float(max(k_vals)),
        }

    return {
        "yaw_deg": yaw_deg,
        "altitudes_m": altitudes_m,
        "camera_sets": by_cam_set,
    }


def _evaluate_sequence(
    rig: Any,
    seq_name: str,
    frames_path: Path,
    max_frames: int | None = None,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    with open(frames_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    if max_frames is not None:
        frames = frames[: int(max_frames)]

    n_frames = len(frames)
    noise_by_sigma: dict[float, np.ndarray] = {}
    for s_idx, s in enumerate(SIGMA_H_SET):
        noise_by_sigma[s] = generate_altitude_noise(
            seed=DEFAULT_NOISE_SEED_BASE + int(s * 100) + s_idx,
            n_frames=n_frames,
            sigma_h=s,
        )

    cam_set_keys = {"F": ("F", CAMERA_SETS["F"]), "FRBL": ("horizontal", CAMERA_SETS["horizontal"])}
    sigma_values = (10.0, 50.0)

    xs_by_cs_sig: dict[tuple[str, int], list[float]] = {
        (cs_lbl, int(sig)): [] for cs_lbl in cam_set_keys for sig in sigma_values
    }
    ys_by_cs_sig: dict[tuple[str, int], list[float]] = {
        (cs_lbl, int(sig)): [] for cs_lbl in cam_set_keys for sig in sigma_values
    }

    for idx, frame in enumerate(frames):
        k = int(frame["k"])
        act = frame.get("aircraft", frame)
        lat_deg = float(act["lat_deg"])
        lon_deg = float(act["lon_deg"])
        h_true = float(act["h_m"])
        yaw_deg, pitch_deg, roll_deg = act["ypr_deg"]
        yaw_rad = math.radians(float(yaw_deg))
        pitch_rad = math.radians(float(pitch_deg))
        roll_rad = math.radians(float(roll_deg))

        full_batch = generate_synthetic_limb_rays(
            rig=rig,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            h_m=h_true,
            yaw_rad=yaw_rad,
            pitch_rad=pitch_rad,
            roll_rad=roll_rad,
        )
        dip_true = compute_cone_dip_deg(lat_deg, lon_deg, h_true)

        for cs_lbl, (pipeline_cs_name, allowed_cams) in cam_set_keys.items():
            sub_batch = _filter_batch_to_cam_set(full_batch, allowed_cams)
            fit_0 = fit_pipeline_attitude(
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                h_meas=h_true,
                yaw_rad=yaw_rad,
                batch=sub_batch,
                frame_k=k,
                rng_seed_base=DEFAULT_NOISE_SEED_BASE,
            )
            pitch_0_deg = float(fit_0["pitch_deg"])

            for sig in sigma_values:
                dh = float(noise_by_sigma[sig][idx])
                h_meas = max(1.0, h_true + dh)
                fit_sig = fit_pipeline_attitude(
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    h_meas=h_meas,
                    yaw_rad=yaw_rad,
                    batch=sub_batch,
                    frame_k=k,
                    rng_seed_base=DEFAULT_NOISE_SEED_BASE,
                )
                pitch_sig_deg = float(fit_sig["pitch_deg"])
                x = compute_cone_dip_deg(lat_deg, lon_deg, h_meas) - dip_true
                y = pitch_sig_deg - pitch_0_deg
                xs_by_cs_sig[(cs_lbl, int(sig))].append(x)
                ys_by_cs_sig[(cs_lbl, int(sig))].append(y)

    out_cs: dict[str, Any] = {}
    for cs_lbl in cam_set_keys:
        reg_10 = _regress_through_origin(xs_by_cs_sig[(cs_lbl, 10)], ys_by_cs_sig[(cs_lbl, 10)])
        reg_50 = _regress_through_origin(xs_by_cs_sig[(cs_lbl, 50)], ys_by_cs_sig[(cs_lbl, 50)])
        xs_comb = xs_by_cs_sig[(cs_lbl, 10)] + xs_by_cs_sig[(cs_lbl, 50)]
        ys_comb = ys_by_cs_sig[(cs_lbl, 10)] + ys_by_cs_sig[(cs_lbl, 50)]
        reg_comb = _regress_through_origin(xs_comb, ys_comb)
        out_cs[cs_lbl] = {
            "sigma_10": reg_10,
            "sigma_50": reg_50,
            "combined": reg_comb,
            "slope_k": reg_10["slope_k"],
            "ci95_half_width": reg_10["ci95_half_width"],
            "ci95": reg_10["ci95"],
            "r2": reg_10["r2"],
            "n": reg_10["n"],
        }
    out_cs["horizontal"] = out_cs["FRBL"]
    return {
        "n_frames": n_frames,
        "camera_sets": out_cs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure WP-06 solver dip regression slopes on analytic tangent rays."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory for XVS data/outputs (default: XVS_DATA_ROOT or output/)",
    )
    parser.add_argument(
        "--g4-dir",
        type=Path,
        default=None,
        help="Directory containing T1_cruise, T2_ksfo28l, T3_turn_climb frames.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <data-root>/xvs/wp06/g0/solver_dip_slopes.json)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on frames per sequence for smoke testing",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_root = get_data_root(args.data_root)
    g4_dir = args.g4_dir if args.g4_dir is not None else (data_root / "xvs" / "wp01" / "g4")
    out_path = args.out if args.out is not None else (data_root / "xvs" / "wp06" / "g0" / "solver_dip_slopes.json")

    with open(SPEC_CONFIG_PATH, "r", encoding="utf-8") as f:
        spec_cfg: dict[str, Any] = json.load(f)

    rig = load_pipeline_rig(rig_kind="true", rig_seed=DEFAULT_RIG_SEED)
    level_cone = _evaluate_level_cone(rig, spec_cfg)

    sequences = ("T1_cruise", "T2_ksfo28l", "T3_turn_climb")
    inputs_sha256: dict[str, str] = {
        str(SPEC_CONFIG_PATH): _sha256_file(SPEC_CONFIG_PATH),
    }
    trajectories: dict[str, Any] = {}
    for seq in sequences:
        frames_path = g4_dir / seq / "frames.jsonl"
        if not frames_path.is_file():
            raise FileNotFoundError(f"Missing sequence frames.jsonl at {frames_path}")
        inputs_sha256[str(frames_path)] = _sha256_file(frames_path)
        trajectories[seq] = _evaluate_sequence(rig, seq, frames_path, max_frames=args.max_frames)

    commit = _get_git_commit()
    git_status = _get_git_status_porcelain()
    code_hashes = _compute_code_hashes()
    timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload: dict[str, Any] = {
        "gate": "WP-06 G0 Analytic Tangent-Ray Solver Dip Regression Slopes",
        "timestamp_utc": timestamp_utc,
        "commit": commit,
        "git_status": git_status,
        "argv": list(sys.argv),
        "code_hashes": code_hashes,
        "inputs_sha256": inputs_sha256,
        "rig_seed": DEFAULT_RIG_SEED,
        "subsampled_max_frames": args.max_frames,
        "level_cone": level_cone,
        "trajectories": trajectories,
        "provenance": {
            "commit": commit,
            "git_status": git_status,
            "argv": list(sys.argv),
            "code_hashes": code_hashes,
            "inputs_sha256": inputs_sha256,
            "timestamp_utc": timestamp_utc,
            "wall_time_s": time.perf_counter() - t0,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote solver dip slopes to {out_path} in {payload['provenance']['wall_time_s']:.2f}s:")
    for seq, sdata in trajectories.items():
        kf = sdata["camera_sets"]["F"]["slope_k"]
        kfrbl = sdata["camera_sets"]["FRBL"]["slope_k"]
        n_pts = sdata["camera_sets"]["F"]["n"]
        print(f"  {seq} (n={n_pts}): k_F = {kf:+.4f}, k_FRBL = {kfrbl:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
