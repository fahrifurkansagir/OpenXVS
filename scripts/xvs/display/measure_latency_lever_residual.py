"""Measure constant-rate rotation latency lever-arm residual using src/xvs/display/latency.py.

Evaluates the constant-rate rotation case of test_v4_constant_rate_rotation_compensated_exact
across the configured latency values in configs/xvs/display/g0.json, both at the aircraft
reference point (ARP = [0, 0, 0]) and at the nominal pilot eye lever arm from
configs/xvs/display/g0.json, reading tolerance limits from configs/xvs/display/g0_spec.json.

Writes output/xvs/wp08/g0/latency_lever_residual.json with commit, git_status, argv,
code_hashes, and inputs_sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.display.latency import evaluate_registration_error_at_t, load_display_config
from xvs.geo import rotations
from xvs.paths import get_data_root

DISPLAY_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g0.json"
SPEC_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g0_spec.json"
TRACKED_CODE_FILES = (
    "scripts/xvs/display/measure_latency_lever_residual.py",
    "src/xvs/display/latency.py",
    "src/xvs/geo/rotations.py",
    "tests/xvs/test_display_latency.py",
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


def _summarize_errors(arr: np.ndarray) -> dict[str, float]:
    return {
        "min_deg": float(np.min(arr)),
        "median_deg": float(np.median(arr)),
        "p95_deg": float(np.percentile(arr, 95)),
        "max_deg": float(np.max(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure constant-rate rotation latency lever-arm residual."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory for XVS data/outputs (default: XVS_DATA_ROOT or output/)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DISPLAY_CONFIG_PATH,
        help="Display config JSON path (default: configs/xvs/display/g0.json)",
    )
    parser.add_argument(
        "--spec-config",
        type=Path,
        default=SPEC_CONFIG_PATH,
        help="Display spec config JSON path (default: configs/xvs/display/g0_spec.json)",
    )
    parser.add_argument(
        "--latencies-ms",
        type=int,
        nargs="+",
        default=None,
        help="Optional latency values in ms (default: latencies_ms from g0.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <data-root>/xvs/wp08/g0/latency_lever_residual.json)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_root = get_data_root(args.data_root)
    out_path = args.out if args.out is not None else (data_root / "xvs" / "wp08" / "g0" / "latency_lever_residual.json")

    cfg = load_display_config(args.config)
    with open(args.spec_config, "r", encoding="utf-8") as f:
        spec_cfg: dict[str, Any] = json.load(f)

    latencies_ms = [int(x) for x in (args.latencies_ms if args.latencies_ms is not None else cfg["latencies_ms"])]
    arp_limit_deg = float(spec_cfg["lat_tol_deg_v4"])
    lever_limit_deg = float(spec_cfg["lever_tol_deg_v4"])

    omega_ac = np.array([0.03, -0.015, 0.05], dtype=np.float64)
    t_eval_s = 30.0

    def synth_trajectory(t_curr: float) -> SimpleNamespace:
        r_mat = rotations.so3_exp(omega_ac * t_curr)
        y, p, r = rotations.r_to_euler(r_mat)
        return SimpleNamespace(
            lat_deg=37.5,
            lon_deg=-122.3,
            h_m=1000.0,
            yaw_rad=y,
            pitch_rad=p,
            roll_rad=r,
            v_ned_mps=np.zeros(3, dtype=np.float64),
        )

    cfg_arp = {
        "eye_point_body_frd_m": [0.0, 0.0, 0.0],
        "viewport": dict(cfg["viewport"]),
    }
    cfg_nominal = {
        "eye_point_body_frd_m": list(cfg["eye_point_body_frd_m"]),
        "viewport": dict(cfg["viewport"]),
    }

    arp_by_latency: dict[str, Any] = {}
    lever_by_latency: dict[str, Any] = {}
    all_arp_comp: list[np.ndarray] = []
    all_lever_comp: list[np.ndarray] = []

    for delta_ms in latencies_ms:
        delta_s = float(delta_ms) / 1000.0
        key = f"{delta_ms}ms"

        res_arp = evaluate_registration_error_at_t(
            t=t_eval_s,
            delta_s=delta_s,
            trajectory_func=synth_trajectory,
            motion_mode="aircraft_only",
            config=cfg_arp,
        )
        arp_comp = np.asarray(res_arp["errors_compensated_deg"], dtype=np.float64)
        arp_uncomp = np.asarray(res_arp["errors_uncompensated_deg"], dtype=np.float64)
        all_arp_comp.append(arp_comp)
        arp_by_latency[key] = {
            "delta_ms": delta_ms,
            "compensated": _summarize_errors(arp_comp),
            "uncompensated": _summarize_errors(arp_uncomp),
        }

        res_lev = evaluate_registration_error_at_t(
            t=t_eval_s,
            delta_s=delta_s,
            trajectory_func=synth_trajectory,
            motion_mode="aircraft_only",
            config=cfg_nominal,
        )
        lev_comp = np.asarray(res_lev["errors_compensated_deg"], dtype=np.float64)
        lev_uncomp = np.asarray(res_lev["errors_uncompensated_deg"], dtype=np.float64)
        all_lever_comp.append(lev_comp)
        lever_by_latency[key] = {
            "delta_ms": delta_ms,
            "compensated": _summarize_errors(lev_comp),
            "uncompensated": _summarize_errors(lev_uncomp),
        }

    arp_concat = np.concatenate(all_arp_comp)
    lever_concat = np.concatenate(all_lever_comp)
    lever_max_per_delta = [v["compensated"]["max_deg"] for v in lever_by_latency.values()]
    lever_med_per_delta = [v["compensated"]["median_deg"] for v in lever_by_latency.values()]

    commit = _get_git_commit()
    git_status = _get_git_status_porcelain()
    code_hashes = _compute_code_hashes()
    inputs_sha256 = {
        str(args.config): _sha256_file(args.config),
        str(args.spec_config): _sha256_file(args.spec_config),
    }
    timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload: dict[str, Any] = {
        "gate": "WP-08 G0 V4 Constant-Rate Rotation Latency & Lever-Arm Residual",
        "timestamp_utc": timestamp_utc,
        "commit": commit,
        "git_status": git_status,
        "argv": list(sys.argv),
        "code_hashes": code_hashes,
        "inputs_sha256": inputs_sha256,
        "parameters": {
            "t_eval_s": t_eval_s,
            "omega_ac_rad_s": [float(x) for x in omega_ac],
            "latencies_ms": latencies_ms,
            "eye_point_body_frd_m": [float(x) for x in cfg["eye_point_body_frd_m"]],
            "viewport": dict(cfg["viewport"]),
        },
        "limits": {
            "arp_limit_deg": arp_limit_deg,
            "lever_limit_deg": lever_limit_deg,
        },
        "arp_residual": {
            "overall": _summarize_errors(arp_concat),
            "by_latency": arp_by_latency,
            "limit_deg": arp_limit_deg,
            "within_limit": bool(float(np.max(arp_concat)) <= arp_limit_deg),
        },
        "lever_arm_residual": {
            "min_deg": float(np.min(lever_concat)),
            "median_deg": float(np.median(lever_concat)),
            "p95_deg": float(np.percentile(lever_concat, 95)),
            "max_deg": float(np.max(lever_concat)),
            "min_max_across_latencies_deg": float(min(lever_max_per_delta)),
            "median_max_across_latencies_deg": float(np.median(lever_max_per_delta)),
            "max_max_across_latencies_deg": float(max(lever_max_per_delta)),
            "min_median_across_latencies_deg": float(min(lever_med_per_delta)),
            "max_median_across_latencies_deg": float(max(lever_med_per_delta)),
            "by_latency": lever_by_latency,
            "limit_deg": lever_limit_deg,
            "within_limit": bool(float(np.max(lever_concat)) <= lever_limit_deg),
        },
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

    print(
        f"Wrote {out_path}: lever-arm residual min={payload['lever_arm_residual']['min_deg']:.4e} deg, "
        f"median={payload['lever_arm_residual']['median_deg']:.4e} deg, "
        f"max={payload['lever_arm_residual']['max_deg']:.4e} deg "
        f"(limit <= {lever_limit_deg:.4e} deg)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
