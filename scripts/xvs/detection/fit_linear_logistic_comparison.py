"""Fit linear-in-s logistic MLE on G3a v0.1 per-ROI cache alongside log-logistic closure fit.

Reads the G3a v0.1 per-ROI evaluation cache (output/xvs/wp04/g0/raw_eval_cache.pkl.gz)
and the log-logistic closure fit from output/xvs/wp04/g0/g3a_results.json at the
same closure operating point (k=3, tau=20.0, degraded sensor), fits the linear-in-s
logistic model P_d(s) = 1 / (1 + exp(-(s - s50) / w)) via MLE, computes P_d(0),
and writes output/xvs/wp04/g0/linear_fit_comparison.json with full provenance.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.detection.d0 import detect_d0, load_d0_config
from xvs.detection.fit import evaluate_closure_fit, fit_logistic_mle
from xvs.paths import get_data_root
from xvs.sensor.chain import SensorConfig
from xvs.sensor.noise import apply_sensor_noise, generate_fpn_maps
from xvs.sensor.psf import apply_psf

TRACKED_CODE_FILES = (
    "scripts/xvs/detection/fit_linear_logistic_comparison.py",
    "src/xvs/detection/d0.py",
    "src/xvs/detection/fit.py",
    "src/xvs/sensor/chain.py",
    "src/xvs/sensor/noise.py",
    "src/xvs/sensor/psf.py",
)


def _measure_v6_runtime_benchmark() -> dict[str, Any]:
    """Run the 1024x1024 ROI D0 runtime benchmark across Clean and Degraded for k in [3, 5, 7]."""
    rng = np.random.default_rng(20260925)
    img_1024 = rng.uniform(0.1, 0.4, (1024, 1024)).astype(np.float64)

    d0_cfg = load_d0_config()
    sensor_cfg = SensorConfig.from_file(REPO_ROOT / "configs" / "xvs" / "sensors" / "ring_mono_v0_nodist.json")
    psf_kernel, _, _ = sensor_cfg.get_psf_kernel()
    prnu, dsnu = generate_fpn_maps("bench_cam", 1024, 1024, sensor_cfg.prnu_rms, sensor_cfg.dsnu_rms_e)

    blurred = apply_psf(img_1024, psf_kernel)
    raw_dn = apply_sensor_noise(
        linear_radiance=blurred,
        prnu_map=prnu,
        dsnu_map=dsnu,
        E=sensor_cfg.full_well_e,
        full_well_e=sensor_cfg.full_well_e,
        sigma_r_e=sensor_cfg.sigma_r_e,
        dark_current_e_per_s=sensor_cfg.dark_current_e_per_s,
        exposure_time_s=sensor_cfg.exposure_time_s,
        k_dn_per_e=sensor_cfg.k_dn_per_e,
        bits=sensor_cfg.bits,
        black_level_dn=sensor_cfg.black_level_dn,
        rng_temporal=np.random.Generator(np.random.PCG64(999)),
    )
    img_deg_1024 = (raw_dn.astype(np.float64) - sensor_cfg.black_level_dn) / sensor_cfg.k_dn_per_e

    _ = detect_d0(img_1024, k=3, is_clean=True, config=d0_cfg)

    k_values = [3, 5, 7]
    clean_ms: list[float] = []
    clean_fps: list[float] = []
    deg_ms: list[float] = []
    deg_fps: list[float] = []
    clean_by_k: dict[str, float] = {}
    deg_by_k: dict[str, float] = {}

    for k in k_values:
        t0 = time.perf_counter()
        _ = detect_d0(img_1024, k=k, tau=4.0, is_clean=True, config=d0_cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        clean_ms.append(elapsed_ms)
        clean_fps.append(1000.0 / elapsed_ms)
        clean_by_k[str(k)] = elapsed_ms

    for k in k_values:
        t0 = time.perf_counter()
        _ = detect_d0(img_deg_1024, k=k, tau=4.0, is_clean=False, config=d0_cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        deg_ms.append(elapsed_ms)
        deg_fps.append(1000.0 / elapsed_ms)
        deg_by_k[str(k)] = elapsed_ms

    return {
        "v6_c1": int(round(min(clean_ms))),
        "v6_c2": int(round(max(clean_ms))),
        "v6_cf1": float(min(clean_fps)),
        "v6_cf2": float(max(clean_fps)),
        "v6_d1": int(round(min(deg_ms))),
        "v6_d2": int(round(max(deg_ms))),
        "v6_df1": float(min(deg_fps)),
        "v6_df2": float(max(deg_fps)),
        "clean_ms_by_k": clean_by_k,
        "degraded_ms_by_k": deg_by_k,
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit linear-in-s logistic MLE on G3a v0.1 per-ROI cache and compare to log-logistic fit."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory for XVS data/outputs (default: XVS_DATA_ROOT or output/)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Explicit path to raw_eval_cache.pkl.gz (overrides <data-root>/xvs/wp04/g0/raw_eval_cache.pkl.gz)",
    )
    parser.add_argument(
        "--g3a-json",
        type=Path,
        default=None,
        help="Explicit path to g3a_results.json (overrides <data-root>/xvs/wp04/g0/g3a_results.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Explicit output JSON path (overrides <data-root>/xvs/wp04/g0/linear_fit_comparison.json)",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="degraded",
        help="Sensor state for closure fit (default: degraded)",
    )
    parser.add_argument(
        "--closure-k",
        type=int,
        default=None,
        help="Override closure k (default: read from g3a_results.json operating_point.closure_k)",
    )
    parser.add_argument(
        "--closure-tau",
        type=float,
        default=None,
        help="Override closure tau (default: read from g3a_results.json operating_point.closure_tau)",
    )
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Optional limit on positive ROIs for smoke testing",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    data_root = get_data_root(args.data_root)
    cache_path = args.cache if args.cache is not None else (data_root / "xvs" / "wp04" / "g0" / "raw_eval_cache.pkl.gz")
    g3a_json_path = args.g3a_json if args.g3a_json is not None else (data_root / "xvs" / "wp04" / "g0" / "g3a_results.json")
    out_path = args.out if args.out is not None else (data_root / "xvs" / "wp04" / "g0" / "linear_fit_comparison.json")

    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing per-ROI evaluation cache at {cache_path}")
    if not g3a_json_path.is_file():
        raise FileNotFoundError(f"Missing g3a_results.json at {g3a_json_path}")

    with open(g3a_json_path, "r", encoding="utf-8") as f:
        g3a_data: dict[str, Any] = json.load(f)

    op_info = g3a_data["operating_point"]
    closure_k = int(args.closure_k) if args.closure_k is not None else int(op_info["closure_k"])
    closure_tau = float(args.closure_tau) if args.closure_tau is not None else float(op_info["closure_tau"])
    sensor = str(args.sensor)
    tau_key = f"{closure_tau:.1f}"

    with gzip.open(cache_path, "rb") as f_cache:
        cache_data: dict[str, Any] = pickle.load(f_cache)

    results_positive: list[dict[str, Any]] = list(cache_data["results_positive"])
    if args.max_rois is not None:
        results_positive = results_positive[: int(args.max_rois)]

    spx_list: list[float] = []
    hits_list: list[float] = []
    for r in results_positive:
        spx_list.append(float(r["s_px"]))
        by_k = r["eval_by_sensor_k"][sensor]
        ev = by_k[closure_k] if closure_k in by_k else by_k[str(closure_k)]
        hits_list.append(float(ev["primary_hits_by_tau"][tau_key]))

    spx_arr = np.asarray(spx_list, dtype=np.float64)
    hits_arr = np.asarray(hits_list, dtype=np.float64)
    n_enc_pos = len({r["encounter_id"] for r in results_positive})

    fit_res = fit_logistic_mle(spx_arr, hits_arr)
    closure_eval_lin = evaluate_closure_fit(
        spx_arr,
        hits_arr,
        op_k=closure_k,
        op_tau=closure_tau,
        run_label="SMOKE" if args.max_rois is not None else "G0",
        n_encounters=n_enc_pos,
        fit_mode="linear_logistic",
    )

    ci_w_lin = float(fit_res.ci95_high - fit_res.ci95_low)
    if math.isfinite(fit_res.s50) and math.isfinite(fit_res.w) and fit_res.w != 0.0:
        pd_0_lin = float(1.0 / (1.0 + math.exp(fit_res.s50 / fit_res.w)))
    else:
        pd_0_lin = float("nan")

    wp05_log = g3a_data["wp05_closure_fit"]
    pd_0_log = float(1.0 / (1.0 + math.exp(math.inf)))
    v6_bench = _measure_v6_runtime_benchmark()

    commit = _get_git_commit()
    git_status = _get_git_status_porcelain()
    code_hashes = _compute_code_hashes()
    inputs_sha256 = {
        str(cache_path): _sha256_file(cache_path),
        str(g3a_json_path): _sha256_file(g3a_json_path),
    }
    if cache_data.get("release_json_sha256"):
        inputs_sha256["RELEASE.json"] = str(cache_data["release_json_sha256"])

    timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload: dict[str, Any] = {
        "gate": "WP-04 G0 Linear vs Log-Logistic Closure Fit Comparison",
        "timestamp_utc": timestamp_utc,
        "commit": commit,
        "git_status": git_status,
        "argv": list(sys.argv),
        "code_hashes": code_hashes,
        "inputs_sha256": inputs_sha256,
        "closure_point": {
            "sensor": sensor,
            "k": closure_k,
            "tau": closure_tau,
            "n_instances": int(len(results_positive)),
            "n_encounters": int(n_enc_pos),
            "subsampled_max_rois": args.max_rois,
        },
        "linear_logistic": {
            "fit_mode": "linear_logistic",
            "formula": "1 / (1 + exp(-(s_px - s50_px) / w_px))",
            "converged": bool(fit_res.converged),
            "iterations": int(fit_res.iterations),
            "closure_gate_status": closure_eval_lin["status"],
            "s50_px": float(fit_res.s50),
            "w_px": float(fit_res.w),
            "se_s50_px": float(fit_res.se_s50),
            "pd_0": pd_0_lin,
            "ci95_s50": [float(fit_res.ci95_low), float(fit_res.ci95_high)],
            "ci95_width_px": ci_w_lin,
            "n_instances": int(len(spx_arr)),
            "n_encounters": int(n_enc_pos),
        },
        "log_logistic": {
            "fit_mode": str(wp05_log["fit_mode"]) if "fit_mode" in wp05_log else "log_logistic",
            "formula": "1 / (1 + exp(-(ln(s_px) - ln(s50_px)) / w_ln))",
            "status": str(wp05_log["status"]),
            "s50_px": float(wp05_log["s50_px"]),
            "w_ln": float(wp05_log["w_ln"]),
            "pd_0": pd_0_log,
            "ci95_s50": [float(wp05_log["ci95_s50"][0]), float(wp05_log["ci95_s50"][1])],
            "ci95_width_px": float(wp05_log["ci95_width_px"]),
            "n_instances": int(wp05_log["n_instances"]),
            "n_encounters": int(wp05_log["n_encounters"]),
        },
        "v6_runtime_benchmark": v6_bench,
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
        f"Wrote {out_path}: linear_logistic s50={fit_res.s50:.3f} px, "
        f"w={fit_res.w:.3f} px, P_d(0)={pd_0_lin:.4f} "
        f"| log_logistic s50={wp05_log['s50_px']:.3f} px, w_ln={wp05_log['w_ln']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
