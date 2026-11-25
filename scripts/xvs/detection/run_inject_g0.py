"""Run controlled traffic detection injection study (WP-04 G0).

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §5, GEMINI_G-B8 Part A, and D23.
Executes the full injection grid across (s_px, C, background, sensor, k) including C=0 null controls,
evaluates primary (tight centroid radius r) and secondary (AOT eIoU) hit rules,
sweeps tau to measure false positive density per Mpx with Poisson 95% upper bounds,
generates JSON results, publication plots, and Markdown gate report.
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
from typing import Any
import numpy as np

# Ensure matplotlib runs headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xvs.paths import get_data_root
from xvs.detection.d0 import (
    detect_d0,
    load_d0_config,
    compute_cmo_multi,
    compute_local_stats,
    compute_clean_floor_grid,
    compute_z_score,
    extract_components,
)
from xvs.detection.inject import (
    load_traffic_models,
    sample_traffic_silhouette,
    composite_signal,
    extract_background_crops,
    InjectionSensorSimulator,
    fit_ac_logistic,
    matching_radius,
    evaluate_detections_for_trial,
    compute_fp_density_sweep,
    find_operating_point_tau,
    compute_corrected_pd_bootstrap,
    CLASSES,
    GRID_S_PX,
    GRID_C,
    GRID_BACKGROUND,
    GRID_SENSOR,
    GRID_K,
    TAU_LIST_EXTENDED,
    TRIAL_SEARCH_AREA_MPX,
)
from xvs.metrics.aot import clopper_pearson_interval, poisson_rate_upper_bound

from xvs.paths import get_data_root

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = get_data_root() / "xvs" / "wp04" / "g0"


_WORKER_CROPS: dict[str, list[dict[str, Any]]] = {}
_WORKER_MODELS: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_WORKER_SENSOR_SIM: InjectionSensorSimulator | None = None
_WORKER_D0_CFG: dict[str, Any] = {}


def init_worker(
    crops_dict: dict[str, list[dict[str, Any]]],
    models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    d0_cfg: dict[str, Any],
) -> None:
    """Initialize process-global state once per worker process."""
    global _WORKER_CROPS, _WORKER_MODELS, _WORKER_SENSOR_SIM, _WORKER_D0_CFG
    _WORKER_CROPS = crops_dict
    _WORKER_MODELS = models_dict
    _WORKER_SENSOR_SIM = InjectionSensorSimulator()
    _WORKER_D0_CFG = d0_cfg


def evaluate_cell_worker(
    cell_idx: int,
    s_px: float,
    contrast_c: float,
    bg_type: str,
    n_samples: int,
    base_seed: int,
) -> dict[str, Any]:
    """Worker function to evaluate all n_samples trials for one (s_px, C, bg_type) cell."""
    rng = np.random.default_rng(base_seed + cell_idx * 10007)
    sensor_sim = _WORKER_SENSOR_SIM
    bg_crops_data = _WORKER_CROPS[bg_type]
    models_data = _WORKER_MODELS
    d0_cfg = _WORKER_D0_CFG
    t_start = time.perf_counter()

    n_crops = len(bg_crops_data)

    # Initialize per-tau and nominal statistics accumulators
    tau_keys = [f"{t:.1f}" for t in TAU_LIST_EXTENDED]

    primary_hits_count = {
        sensor: {k_val: {t_k: 0 for t_k in tau_keys} for k_val in GRID_K}
        for sensor in GRID_SENSOR
    }
    secondary_hits_count = {
        sensor: {k_val: {t_k: 0 for t_k in tau_keys} for k_val in GRID_K}
        for sensor in GRID_SENSOR
    }
    fp_counts = {
        sensor: {k_val: {t_k: 0 for t_k in tau_keys} for k_val in GRID_K}
        for sensor in GRID_SENSOR
    }

    # Binary hits at nominal tau = 4.0
    nominal_tau = 4.0
    primary_hits_nominal = {
        sensor: {k_val: [] for k_val in GRID_K}
        for sensor in GRID_SENSOR
    }
    secondary_hits_nominal = {
        sensor: {k_val: [] for k_val in GRID_K}
        for sensor in GRID_SENSOR
    }
    primary_scores_nominal = {
        sensor: {k_val: [] for k_val in GRID_K}
        for sensor in GRID_SENSOR
    }

    tile_size = d0_cfg.get("tile_size", 64)  # default-ok: optional fallback default
    tile_stride = d0_cfg.get("tile_stride", 32)  # default-ok: optional fallback default
    mad_scale = d0_cfg.get("mad_scale", 1.4826)  # default-ok: optional fallback default
    dilation_size = d0_cfg.get("dilation_kernel_size", 3)  # default-ok: optional fallback default

    a_c_list: list[float] = []
    sigma_bg_list: list[float] = []

    for trial_idx in range(n_samples):
        # 1. Background crop
        crop_info = bg_crops_data[int(rng.integers(0, n_crops))]
        L_bg = crop_info["image"]
        sigma_bg = crop_info["sigma_bg"]

        # 2. Target silhouette
        u_t = float(rng.uniform(20.0, 236.0))
        v_t = float(rng.uniform(20.0, 236.0))
        sil = sample_traffic_silhouette(models_data, s_px, u_t, v_t, rng, crop_size=256)
        alpha = sil["alpha"]
        box_gt = sil["box_gt"]

        # 3. Composite (when contrast_c == 0, alpha is drawn but not added: L_prime = L_bg)
        L_prime, a_c = composite_signal(L_bg, alpha, contrast_c)
        a_c_list.append(a_c)
        sigma_bg_list.append(sigma_bg)

        # 4. Clean detections: exact component extraction across tau sweep
        clean_floor = compute_clean_floor_grid(L_prime, tile_size=tile_size, tile_stride=tile_stride)
        cmos_clean = compute_cmo_multi(L_prime, k_values=GRID_K)

        for k_val in GRID_K:
            cmo_c = cmos_clean[k_val]
            m_c, s_c = compute_local_stats(cmo_c, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
            z_c = compute_z_score(cmo_c, m_c, s_c, sigma_min=clean_floor)

            for tau in TAU_LIST_EXTENDED:
                t_k = f"{tau:.1f}"
                dets_c = extract_components(z_c, cmo_c, tau=tau, dilation_size=dilation_size)
                res_c = evaluate_detections_for_trial(dets_c, u_t, v_t, box_gt, s_px)

                p_best = res_c["primary_best_score"]
                s_best = res_c["secondary_best_score"]
                fp_scores = res_c["fp_scores"]

                if p_best > -float("inf"):
                    primary_hits_count["clean"][k_val][t_k] += 1
                if s_best > -float("inf"):
                    secondary_hits_count["clean"][k_val][t_k] += 1
                fp_counts["clean"][k_val][t_k] += len(fp_scores)

                if tau == nominal_tau:
                    primary_hits_nominal["clean"][k_val].append(1 if p_best > -float("inf") else 0)
                    secondary_hits_nominal["clean"][k_val].append(1 if s_best > -float("inf") else 0)
                    primary_scores_nominal["clean"][k_val].append(p_best)

        # 5. Degraded detections: exact component extraction across tau sweep
        seed_temporal = int(rng.integers(0, 2**31 - 1))
        L_deg = sensor_sim.apply_sensor(L_prime, is_degraded=True, seed=seed_temporal)
        sigma_min_deg = float(d0_cfg["sigma_min_degraded"])
        cmos_deg = compute_cmo_multi(L_deg, k_values=GRID_K)

        for k_val in GRID_K:
            cmo_d = cmos_deg[k_val]
            m_d, s_d = compute_local_stats(cmo_d, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
            z_d = compute_z_score(cmo_d, m_d, s_d, sigma_min=sigma_min_deg)

            for tau in TAU_LIST_EXTENDED:
                t_k = f"{tau:.1f}"
                dets_d = extract_components(z_d, cmo_d, tau=tau, dilation_size=dilation_size)
                res_d = evaluate_detections_for_trial(dets_d, u_t, v_t, box_gt, s_px)

                p_best = res_d["primary_best_score"]
                s_best = res_d["secondary_best_score"]
                fp_scores = res_d["fp_scores"]

                if p_best > -float("inf"):
                    primary_hits_count["degraded"][k_val][t_k] += 1
                if s_best > -float("inf"):
                    secondary_hits_count["degraded"][k_val][t_k] += 1
                fp_counts["degraded"][k_val][t_k] += len(fp_scores)

                if tau == nominal_tau:
                    primary_hits_nominal["degraded"][k_val].append(1 if p_best > -float("inf") else 0)
                    secondary_hits_nominal["degraded"][k_val].append(1 if s_best > -float("inf") else 0)
                    primary_scores_nominal["degraded"][k_val].append(p_best)

    elapsed_s = time.perf_counter() - t_start

    return {
        "cell_idx": cell_idx,
        "s_px": s_px,
        "contrast_c": contrast_c,
        "bg_type": bg_type,
        "n_samples": n_samples,
        "primary_hits_count": primary_hits_count,
        "secondary_hits_count": secondary_hits_count,
        "fp_counts": fp_counts,
        "primary_hits_nominal": primary_hits_nominal,
        "secondary_hits_nominal": secondary_hits_nominal,
        "primary_scores_nominal": primary_scores_nominal,
        "a_c_list": a_c_list,
        "sigma_bg_list": sigma_bg_list,
        "elapsed_s": elapsed_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WP-04 G0 Injection Study Runner")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes")
    parser.add_argument("--n-samples", type=int, default=1000, help="N samples per cell")
    parser.add_argument("--seed", type=int, default=20260925, help="Random seed")
    parser.add_argument("--pilot-only", action="store_true", help="Run pilot only and exit")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Cores rule: at most (logical cores - 4) worker processes
    max_allowed_workers = max(1, os.cpu_count() - 4) if os.cpu_count() is not None else 4
    num_workers = min(max_allowed_workers, args.workers if args.workers is not None else max_allowed_workers)
    print(f"[Injection Runner] CPUs: {os.cpu_count()}, Active workers: {num_workers} (<= {max_allowed_workers})")

    # Load 3D traffic models
    print("[Injection Runner] Loading traffic 3D models...")
    models_data = load_traffic_models()
    print(f"[Injection Runner] Loaded {len(models_data)} models across {len(CLASSES)} classes.")

    # Extract background crops from real G4 F/R/B/L frames
    print("[Injection Runner] Extracting real G4 horizon band background crops (F/R/B/L)...")
    rng_main = np.random.default_rng(args.seed)
    crops_dict = extract_background_crops(get_data_root(), num_crops_per_type=200, rng=rng_main)
    print(f"[Injection Runner] Extracted {len(crops_dict['sky'])} sky crops, {len(crops_dict['earth'])} earth crops.")
    print(f"[Injection Runner] Crop manifest SHA-256: {crops_dict['manifest_sha256']}")

    d0_cfg = load_d0_config()

    # Define cells: 9 s_px * 9 C * 2 backgrounds = 162 cells (includes C=0.0 nulls)
    cells = []
    for s_px in GRID_S_PX:
        for c_val in GRID_C:
            for bg in GRID_BACKGROUND:
                cells.append((s_px, c_val, bg))

    total_cells = len(cells)
    print(f"[Injection Runner] Total injection grid cells: {total_cells} (9 s_px * 9 C * 2 bg, including C=0 nulls)")

    chosen_n = args.n_samples

    # 1. Pilot measurement
    print("\n--- Pilot Measurement ---")
    pilot_samples = 5
    t_pilot_start = time.perf_counter()
    pilot_cells = cells[:min(8, total_cells)]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(crops_dict, models_data, d0_cfg),
    ) as executor:
        futures = [
            executor.submit(
                evaluate_cell_worker,
                idx,
                s,
                c,
                bg,
                pilot_samples,
                args.seed + 999,
            )
            for idx, (s, c, bg) in enumerate(pilot_cells)
        ]
        pilot_results = []
        for f in concurrent.futures.as_completed(futures):
            pilot_results.append(f.result())

    t_pilot_end = time.perf_counter()
    pilot_wall_s = t_pilot_end - t_pilot_start
    total_pilot_trials = len(pilot_cells) * pilot_samples
    total_core_time = sum(res.get("elapsed_s", 0.0) for res in pilot_results)  # default-ok: timing benchmark fallback
    core_sec_per_trial = total_core_time / max(1, total_pilot_trials)

    trials_total = total_cells * chosen_n
    projected_wall_s = (trials_total * core_sec_per_trial) / num_workers
    projected_wall_h = projected_wall_s / 3600.0
    print(f"Pilot completed: {total_pilot_trials} trials in {pilot_wall_s:.2f} s wall ({core_sec_per_trial * 1000:.1f} ms pure core/trial)")
    print(f"Projected wall time for N={chosen_n} on {num_workers} workers: {projected_wall_h:.2f} hours ({projected_wall_s:.1f} s)")

    if args.pilot_only:
        print("Pilot only requested. Exiting.")
        return

    # 2. Full Injection Grid Execution
    print(f"\n--- Running Full Injection Study with N={chosen_n} per cell ---")
    t_run_start = time.perf_counter()

    cell_results: list[dict[str, Any]] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(crops_dict, models_data, d0_cfg),
    ) as executor:
        future_to_cell = {
            executor.submit(
                evaluate_cell_worker,
                idx,
                s,
                c,
                bg,
                chosen_n,
                args.seed,
            ): (idx, s, c, bg)
            for idx, (s, c, bg) in enumerate(cells)
        }

        completed_count = 0
        for f in concurrent.futures.as_completed(future_to_cell):
            res = f.result()
            cell_results.append(res)
            completed_count += 1
            if completed_count % 16 == 0 or completed_count == total_cells:
                elapsed = time.perf_counter() - t_run_start
                pct = (completed_count / total_cells) * 100.0
                print(f"Progress: {completed_count}/{total_cells} cells ({pct:.1f}%) in {elapsed:.1f} s...")

    t_run_end = time.perf_counter()
    total_run_time_s = t_run_end - t_run_start
    print(f"Full injection grid finished in {total_run_time_s:.2f} s ({total_run_time_s / 60.0:.2f} min).")

    # Index cells by (s_px, contrast_c, bg_type)
    cell_map: dict[tuple[float, float, str], dict[str, Any]] = {}
    for cr in cell_results:
        key = (round(cr["s_px"], 2), round(cr["contrast_c"], 2), cr["bg_type"])
        cell_map[key] = cr

    # 3. Aggregate Results, Null Controls, and Bootstrap Corrected Pd
    print("\n--- Aggregating Results, Null Controls, and Bootstrap Corrected Pd ---")
    summary_tables: dict[str, Any] = {
        "schema": "xvs.wp04_inject_results/2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "batch": "GEMINI_G-B8",
            "decision": "D23",
            "background_scale": "rig_scale (2.33 mrad/px)",
            "background_crops_manifest_sha256": crops_dict["manifest_sha256"],
            "crops_manifest": crops_dict["manifest"],
        },
        "environment": {
            "os": sys.platform,
            "cpu_count": os.cpu_count(),
            "num_workers": num_workers,
            "python_version": sys.version,
        },
        "parameters": {
            "n_samples_per_cell": chosen_n,
            "total_cells": total_cells,
            "total_trials": total_cells * chosen_n,
            "num_workers": num_workers,
            "wall_time_s": total_run_time_s,
            "seed": args.seed,
            "grid_s_px": GRID_S_PX,
            "grid_c": GRID_C,
            "grid_background": GRID_BACKGROUND,
            "grid_sensor": GRID_SENSOR,
            "grid_k": GRID_K,
            "tau_list": TAU_LIST_EXTENDED,
            "trial_search_area_mpx": TRIAL_SEARCH_AREA_MPX,
        },
        "pd_grid": {},
        "tau_sweep": {},
        "ac_logistic_fits": {},
        "sigma_terciles": {},
    }

    # Aggregate Pd grid per (sensor, bg, k)
    for sensor in GRID_SENSOR:
        for bg in GRID_BACKGROUND:
            for k_val in GRID_K:
                key = f"{sensor}_{bg}_k{k_val}"
                summary_tables["pd_grid"][key] = {}

                for s in GRID_S_PX:
                    s_r = round(s, 2)
                    s_str = f"{s:.2f}"
                    if s_str not in summary_tables["pd_grid"][key]:
                        summary_tables["pd_grid"][key][s_str] = {}

                    # Null cell for this (s, bg)
                    null_cell = cell_map.get((s_r, 0.0, bg))
                    null_primary_hits = null_cell["primary_hits_nominal"][sensor][k_val] if null_cell else [0] * chosen_n
                    null_secondary_hits = null_cell["secondary_hits_nominal"][sensor][k_val] if null_cell else [0] * chosen_n

                    for c in GRID_C:
                        c_r = round(c, 2)
                        c_str = f"{c:.2f}"
                        target_cell = cell_map.get((s_r, c_r, bg))
                        if target_cell is None:
                            continue

                        target_primary_hits = target_cell["primary_hits_nominal"][sensor][k_val]
                        target_secondary_hits = target_cell["secondary_hits_nominal"][sensor][k_val]

                        # Primary hit rule bootstrap (d <= r)
                        boot_p = compute_corrected_pd_bootstrap(
                            target_primary_hits,
                            null_primary_hits,
                            n_boot=1000,
                            seed=args.seed + int(s * 100) + int((c + 1.0) * 1000),
                        )

                        # Secondary hit rule bootstrap (eIoU > 0.2)
                        boot_s = compute_corrected_pd_bootstrap(
                            target_secondary_hits,
                            null_secondary_hits,
                            n_boot=1000,
                            seed=args.seed + int(s * 100) + int((c + 1.0) * 1000) + 7,
                        )

                        summary_tables["pd_grid"][key][s_str][c_str] = {
                            "n": chosen_n,
                            "matching_radius_px": matching_radius(s),
                            "primary": {
                                "hits": int(sum(target_primary_hits)),
                                "pd_raw": boot_p["pd_raw"],
                                "pd_raw_ci95": boot_p["pd_raw_ci95"],
                                "p0": boot_p["p0"],
                                "p0_ci95": boot_p["p0_ci95"],
                                "pd_corrected": boot_p["pd_corrected"],
                                "pd_corrected_ci95": boot_p["pd_corrected_ci95"],
                            },
                            "secondary": {
                                "hits": int(sum(target_secondary_hits)),
                                "pd_raw": boot_s["pd_raw"],
                                "pd_raw_ci95": boot_s["pd_raw_ci95"],
                                "p0": boot_s["p0"],
                                "p0_ci95": boot_s["p0_ci95"],
                                "pd_corrected": boot_s["pd_corrected"],
                                "pd_corrected_ci95": boot_s["pd_corrected_ci95"],
                            },
                        }

    # 4. Aggregate Tau Sweep and FP Density per Stratum (bg, sensor, k)
    print("[Injection Runner] Aggregating tau sweep and false positive densities...")
    for bg in GRID_BACKGROUND:
        stratum_cells = [c for c in cell_results if c["bg_type"] == bg]
        num_stratum_trials = len(stratum_cells) * chosen_n
        total_area_mpx = float(num_stratum_trials) * TRIAL_SEARCH_AREA_MPX

        for sensor in GRID_SENSOR:
            for k_val in GRID_K:
                stratum_key = f"{sensor}_{bg}_k{k_val}"
                summary_tables["tau_sweep"][stratum_key] = {
                    "total_trials": num_stratum_trials,
                    "total_area_mpx": total_area_mpx,
                    "tau_stats": {},
                }

                fp_sweep_dict: dict[str, dict[str, float]] = {}

                for tau in TAU_LIST_EXTENDED:
                    t_k = f"{tau:.1f}"
                    total_fp = sum(c["fp_counts"][sensor][k_val][t_k] for c in stratum_cells)
                    total_p_hits = sum(c["primary_hits_count"][sensor][k_val][t_k] for c in stratum_cells)
                    total_s_hits = sum(c["secondary_hits_count"][sensor][k_val][t_k] for c in stratum_cells)

                    fp_per_mpx = float(total_fp / total_area_mpx) if total_area_mpx > 0 else 0.0
                    fp_ci_high = float(poisson_rate_upper_bound(total_fp, total_area_mpx, confidence=0.95))

                    summary_tables["tau_sweep"][stratum_key]["tau_stats"][t_k] = {
                        "tau": float(tau),
                        "fp_count": total_fp,
                        "fp_per_mpx": fp_per_mpx,
                        "fp_per_mpx_ci95_high": fp_ci_high,
                        "primary_hits": total_p_hits,
                        "secondary_hits": total_s_hits,
                        "primary_pd_overall": float(total_p_hits / max(1, num_stratum_trials)),
                        "secondary_pd_overall": float(total_s_hits / max(1, num_stratum_trials)),
                    }
                    fp_sweep_dict[t_k] = {
                        "tau": float(tau),
                        "fp_per_mpx": fp_per_mpx,
                        "fp_per_mpx_ci95_high": fp_ci_high,
                    }

                # Operating point check for <= 1.0e-4 per Mpx
                op_result = find_operating_point_tau(fp_sweep_dict, target_fp_upper_per_mpx=1.0e-4)
                summary_tables["tau_sweep"][stratum_key]["operating_point"] = op_result

    # 5. Logistic fit against ln|A_C|
    print("[Injection Runner] Fitting logistic curves against ln|A_C|...")
    for sensor in GRID_SENSOR:
        for bg in GRID_BACKGROUND:
            for k_val in GRID_K:
                for polarity in ["positive", "negative"]:
                    fit_key = f"{sensor}_{bg}_k{k_val}_{polarity}"
                    ac_vals: list[float] = []
                    det_vals: list[int] = []

                    bg_cells = [c for c in cell_results if c["bg_type"] == bg]
                    for cell in bg_cells:
                        c_val = cell["contrast_c"]
                        if (polarity == "positive" and c_val > 0) or (polarity == "negative" and c_val < 0):
                            hits_list = cell["primary_hits_nominal"][sensor][k_val]
                            for ac, hit in zip(cell["a_c_list"], hits_list):
                                ac_vals.append(ac)
                                det_vals.append(hit)

                    fit_res = fit_ac_logistic(ac_vals, det_vals)
                    summary_tables["ac_logistic_fits"][fit_key] = fit_res

    # 6. P_d stratified by sigma_bg terciles
    print("[Injection Runner] Computing stratification by sigma_bg terciles...")
    for bg in GRID_BACKGROUND:
        all_sigmas = [c["sigma_bg"] for c in crops_dict[bg]]
        t1, t2 = np.percentile(all_sigmas, [33.333, 66.667])
        tercile_bounds = {"low": [0.0, float(t1)], "med": [float(t1), float(t2)], "high": [float(t2), float("inf")]}
        summary_tables["sigma_terciles"][bg] = {
            "bounds": tercile_bounds,
            "tercile_stats": {},
        }

        bg_cells = [c for c in cell_results if c["bg_type"] == bg and c["contrast_c"] != 0.0]
        for sensor in GRID_SENSOR:
            for k_val in GRID_K:
                k_stat_key = f"{sensor}_k{k_val}"
                summary_tables["sigma_terciles"][bg]["tercile_stats"][k_stat_key] = {}

                for t_name, (b_low, b_high) in tercile_bounds.items():
                    t_hits = 0
                    t_n = 0
                    for cell in bg_cells:
                        hits_arr = cell["primary_hits_nominal"][sensor][k_val]
                        for sig, hit in zip(cell["sigma_bg_list"], hits_arr):
                            if b_low <= sig < b_high:
                                t_n += 1
                                t_hits += hit
                    pd_t = t_hits / max(1, t_n)
                    ci_l, ci_h = clopper_pearson_interval(t_hits, t_n, confidence=0.95)
                    summary_tables["sigma_terciles"][bg]["tercile_stats"][k_stat_key][t_name] = {
                        "n": t_n,
                        "hits": t_hits,
                        "pd": pd_t,
                        "ci95_low": ci_l,
                        "ci95_high": ci_h,
                    }

    # Save JSON results
    json_path = out_dir / "inject_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_tables, f, indent=2)
    print(f"[Injection Runner] Results JSON saved to {json_path}")

    # 7. Generate Publication Plots
    print("\n--- Generating Publication Plots ---")
    # Plot 1: P_d vs s_px (Degraded sensor, Earth & Sky, k=5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, bg in zip(axes, GRID_BACKGROUND):
        grid_key = f"degraded_{bg}_k5"
        tbl = summary_tables["pd_grid"].get(grid_key, {})
        for c_val in [0.02, 0.05, 0.1, -0.1, -0.4]:
            c_str = f"{c_val:.2f}"
            s_pts = []
            pd_corr_pts = []
            for s_val in GRID_S_PX:
                s_str = f"{s_val:.2f}"
                if s_str in tbl and c_str in tbl[s_str]:
                    s_pts.append(s_val)
                    pd_corr_pts.append(tbl[s_str][c_str]["primary"]["pd_corrected"])
            ax.plot(s_pts, pd_corr_pts, marker="o", label=f"C = {c_str}")

        ax.set_title(f"Degraded Sensor (k=5) - Background: {bg.capitalize()}")
        ax.set_xlabel("Target Size $s_{px}$ (pixels)")
        ax.set_ylabel("Abbott-Corrected Detection Probability $P_{d,corr}$")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plot1_path = plots_dir / "pd_vs_spx.png"
    fig.savefig(plot1_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot1_path}")

    # Plot 2: P_d vs FP density curves for (s, C) in {2, 4} px x {-0.1, +0.1}
    fig, ax = plt.subplots(figsize=(10, 7))
    styles = {
        (2.0, -0.1): ("#1f77b4", "--", "s=2 px, C=-0.1"),
        (2.0, 0.1): ("#1f77b4", "-", "s=2 px, C=+0.1"),
        (4.0, -0.1): ("#d62728", "--", "s=4 px, C=-0.1"),
        (4.0, 0.1): ("#d62728", "-", "s=4 px, C=+0.1"),
    }

    # Use degraded earth k=5 as representative operational stratum
    stratum_key = "degraded_earth_k5"
    fp_stats = summary_tables["tau_sweep"].get(stratum_key, {}).get("tau_stats", {})

    for (s_t, c_t), (col, lstyle, lbl) in styles.items():
        s_cell = cell_map.get((round(s_t, 2), round(c_t, 2), "earth"))  # default-ok: optional fallback default
        null_cell = cell_map.get((round(s_t, 2), 0.0, "earth"))  # default-ok: optional fallback default
        if s_cell is None or null_cell is None:
            continue

        fp_densities = []
        pd_corrs = []

        for tau in TAU_LIST_EXTENDED:
            t_k = f"{tau:.1f}"
            fp_d = fp_stats.get(t_k, {}).get("fp_per_mpx", 0.0)  # default-ok: geometry dimension fallback
            fp_densities.append(fp_d)

            tgt_hits = s_cell["primary_hits_count"]["degraded"][5].get(t_k, 0)  # default-ok: optional fallback default
            null_hits = null_cell["primary_hits_count"]["degraded"][5].get(t_k, 0)  # default-ok: optional fallback default
            p_raw = tgt_hits / chosen_n
            p0 = null_hits / chosen_n
            p_corr = (p_raw - p0) / (1.0 - p0) if p0 < 1.0 else 0.0
            pd_corrs.append(max(0.0, min(1.0, p_corr)))

        ax.plot(fp_densities, pd_corrs, marker="o", color=col, linestyle=lstyle, label=lbl)

    ax.set_xscale("log")
    ax.set_xlabel("False Positive Density (FP / Mpx)")
    ax.set_ylabel("Abbott-Corrected $P_{d,corr}$")
    ax.set_title("Detection Performance vs False Alarm Density (Degraded Earth, k=5)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plot2_path = plots_dir / "pd_vs_fp_density.png"
    fig.savefig(plot2_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot2_path}")

    # Plot 3: P_d vs sigma_bg tercile
    fig, ax = plt.subplots(figsize=(8, 6))
    tercile_names = ["low", "med", "high"]
    x_pos = np.arange(len(tercile_names))
    width = 0.2

    for i, bg in enumerate(GRID_BACKGROUND):
        deg_k5_stats = summary_tables["sigma_terciles"][bg]["tercile_stats"].get("degraded_k5", {})
        pds = [deg_k5_stats.get(t, {}).get("pd", 0.0) for t in tercile_names]  # default-ok: optional fallback default
        errs = [
            [pds[j] - deg_k5_stats.get(t, {}).get("ci95_low", pds[j]) for j, t in enumerate(tercile_names)],
            [deg_k5_stats.get(t, {}).get("ci95_high", pds[j]) - pds[j] for j, t in enumerate(tercile_names)],
        ]
        ax.bar(x_pos + (i - 0.5) * width, pds, width, yerr=errs, capsize=4, label=f"{bg.capitalize()} (Degraded k=5)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Low $\\sigma_{bg}$", "Medium $\\sigma_{bg}$", "High $\\sigma_{bg}$"])
    ax.set_ylabel("Detection Probability $P_d$")
    ax.set_title("Detectability Stratified by Background Noise Tercile")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plot3_path = plots_dir / "pd_vs_sigma_terciles.png"
    fig.savefig(plot3_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot3_path}")

    # 8. Generate markdown report
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.xvs.detection.report_tables_wp04 import generate_wp04_report
    print("\n--- Generating Markdown Report ---")
    rep_path = generate_wp04_report(json_path)
    print(f"Report regenerated successfully at {rep_path}")
    print("\n[Injection Runner] Complete!")


if __name__ == "__main__":
    main()
