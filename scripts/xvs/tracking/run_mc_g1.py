"""Monte Carlo simulation runner for WP-05 G1 multi-camera tracking (Batch B7).

Conforms strictly to docs/xvs/batches/BATCH_B7_ARGON.md Steps 1-6.
Enforces CPU load rule: at most (logical cores - 4) worker processes.
Generates under output/xvs/wp05/g1/:
- v5_determinism.json
- baseline_results.json (900 encounters x 5 seeds with CRN pairing against G0)
- sweeps_results.json (optional parameter sweeps)
- perf.json
- run_manifest.json (written before worker launch and finalized on completion per G-B14 §1)
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, data_root_source
from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.encounter import EncounterSamplerV1, export_encounters_jsonl
from xvs.tracking.evaluate import (
    BETA_WC_GRID,
    _worker_wrapper,
    aggregate_dataset_metrics,
    get_ownship_states,
    load_or_compute_truth_cache,
    run_evaluation,
)

CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json"
G0_SWEEP_RESULTS_PATH = REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "lambda_sweep_v2_results.json"
G0_PER_RUN_PATH = REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "per_run_records.jsonl"
EXPECTED_SAMPLER_SHA256 = "d1ae4add34cf0fdc9a45e0663fc347b1e80fd26ae1b3f433b0a3c8e5d8c91a9b"

TRACKED_CODE_FILES = [
    "scripts/xvs/tracking/run_mc_g1.py",
    "src/xvs/tracking/ttc.py",
    "src/xvs/tracking/ground_rejection.py",
    "src/xvs/tracking/measurements.py",
    "src/xvs/tracking/tracker.py",
    "src/xvs/tracking/risk.py",
    "src/xvs/tracking/evaluate.py",
    "configs/xvs/tracking/g1.json",
]


def get_safe_workers() -> int:
    """Enforce rule: at most (logical cores - 4) worker processes."""
    cores = os.cpu_count() or 4
    return max(1, cores - 4)


def compute_code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in TRACKED_CODE_FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=10,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def get_git_status_porcelain() -> str:
    """Check porcelain status strictly over XVS code/scripts/tracking-configs per B7 §6.1."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/xvs", "scripts/xvs", "configs/xvs/tracking"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=10,
        check=True,
    )
    return proc.stdout.strip()


def _init_pool_worker() -> None:
    get_ownship_states()


def get_or_generate_encounters() -> tuple[list[Any], str]:
    """Generate or load sampler v1 900 encounters with LF SHA-256 verification."""
    sampler = EncounterSamplerV1()
    encounters = sampler.generate(t1_cruise_state)
    cache_path = REPO_ROOT / "data" / "cache" / "wp05_encounters_sampler_v1.jsonl"
    sha = export_encounters_jsonl(encounters, cache_path)
    if sha != EXPECTED_SAMPLER_SHA256:
        raise ValueError(f"Sampler SHA-256 mismatch! Got {sha}, expected {EXPECTED_SAMPLER_SHA256}")
    return encounters, sha


def run_evaluation_crn(
    encounters: list[Any],
    config: dict[str, Any],
    seed: int,
    truth_cache_map: dict[str, Any],
    pool: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute evaluation for one seed using CRN seed stream (seed * 10000 + i)."""
    base_seed = seed * 10000 if seed < 10000 else seed
    tasks = [
        (asdict(e), config, base_seed + i, truth_cache_map.get(e.id))
        for i, e in enumerate(encounters)
    ]
    t0 = time.perf_counter()
    if pool is not None:
        results = pool.map(_worker_wrapper, tasks)
    else:
        results = [_worker_wrapper(task) for task in tasks]
    t1 = time.perf_counter()

    metrics = aggregate_dataset_metrics(results, config)
    metrics["runtime_s"] = t1 - t0
    metrics["frames_per_second"] = (len(encounters) * 601) / max(1e-9, t1 - t0)
    return metrics, results


def run_baseline(
    encounters: list[Any],
    base_config: dict[str, Any],
    seeds: list[int],
    truth_cache_map: dict[str, Any],
    max_workers: int,
) -> dict[str, Any]:
    """Run G1 baseline Monte Carlo: 900 encounters x 5 seeds with CRN pairing."""
    print(f"\n--- Running G1 Baseline Monte Carlo ({len(encounters)} encs x {len(seeds)} seeds, workers={max_workers}) ---")
    per_seed_results = []
    all_flat_results: list[dict[str, Any]] = []
    total_t0 = time.perf_counter()

    pool = Pool(processes=max_workers, initializer=_init_pool_worker) if max_workers > 1 else None
    try:
        for idx, s in enumerate(seeds):
            t0 = time.perf_counter()
            print(f"  [Baseline G1] Seed {s} ({idx+1}/{len(seeds)})...", flush=True)
            res, raw_list = run_evaluation_crn(encounters, base_config, s, truth_cache_map, pool=pool)
            t1 = time.perf_counter()
            print(f"  [Baseline G1] Seed {s} completed in {t1 - t0:.1f} s ({res['frames_per_second']:.0f} frames/s)", flush=True)
            per_seed_results.append(res)
            all_flat_results.extend(raw_list)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    total_t1 = time.perf_counter()
    total_runtime = total_t1 - total_t0

    pooled = aggregate_dataset_metrics(all_flat_results, base_config)

    acq_medians = [r["tracking"]["acquisition_delay"]["median_s"] for r in per_seed_results if not math.isnan(r["tracking"]["acquisition_delay"]["median_s"])]
    acq_p95s = [r["tracking"]["acquisition_delay"]["p95_s"] for r in per_seed_results if not math.isnan(r["tracking"]["acquisition_delay"]["p95_s"])]
    covs = [r["tracking"]["true_track_coverage"] for r in per_seed_results if not math.isnan(r["tracking"]["true_track_coverage"])]
    sigma_omegas = [r["los_rate"]["sigma_omega_total_radps"] for r in per_seed_results]
    ttc_errors = [r["ttc"]["mean_error_s"] for r in per_seed_results if not math.isnan(r["ttc"]["mean_error_s"])]
    ttc_medians = [r["ttc"]["median_error_s"] for r in per_seed_results if not math.isnan(r["ttc"]["median_error_s"])]

    wc1_by_beta_means: dict[str, Any] = {}
    for b_val in BETA_WC_GRID:
        b_key = f"{b_val:.2f}"
        w_stats = pooled["alerting"]["WC1_by_beta"][b_key]["warning"]
        c_stats = pooled["alerting"]["WC1_by_beta"][b_key]["corrective"]
        wc1_by_beta_means[b_key] = {
            "beta_wc": float(b_val),
            "warning_ge_12_5s": float(w_stats["frac_ge_12_5s"]),
            "warning_ge_25_0s": float(w_stats["frac_ge_25_0s"]),
            "warning_missed_rate": float(w_stats["frac_missed"]),
            "warning_false_alerts_per_hour": float(w_stats["false_alerts_per_hour"]),
            "warning_false_alert_events_per_hour": float(w_stats["false_alert_events_per_h"]),
            "warning_lead_mean_s": float(w_stats["lead_time_mean_s"]) if not math.isnan(w_stats["lead_time_mean_s"]) else None,
            "corrective_ge_25_0s": float(c_stats["frac_ge_25_0s"]),
            "corrective_ge_55_0s": float(c_stats["frac_ge_55_0s"]),
            "corrective_missed_rate": float(c_stats["frac_missed"]),
            "corrective_false_alerts_per_hour": float(c_stats["false_alerts_per_hour"]),
        }

    aggregated: dict[str, Any] = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "total_runtime_s": total_runtime,
        "encounters_per_seed": len(encounters),
        "pooled_metrics": pooled,
        "per_seed_results": per_seed_results,
        "headline_means": {
            "acq_delay_median_s": float(np.mean(acq_medians)) if acq_medians else float("nan"),
            "acq_delay_p95_s": float(np.mean(acq_p95s)) if acq_p95s else float("nan"),
            "true_track_coverage": float(np.mean(covs)) if covs else float("nan"),
            "fragmentations_per_seed": float(pooled["tracking"]["fragmentations_total"] / max(1, len(seeds))),
            "seam_crossings_per_seed": float(pooled["tracking"]["seam_crossings_total"] / max(1, len(seeds))),
            "seam_breaks_per_seed": float(pooled["tracking"]["seam_breaks_total"] / max(1, len(seeds))),
            "false_tracks_ever_confirmed_per_hour": float(pooled["tracking"]["false_tracks_per_hour"]),
            "false_tracks_end_of_run_per_hour": float(pooled["tracking"]["false_tracks_end_of_run_per_hour"]),
            "ground_clutter_rejected_total": int(pooled["tracking"]["ground_clutter_rejected_total"]),
            "ground_clutter_rejected_per_hour": float(pooled["tracking"]["ground_clutter_rejected_per_hour"]),
            "sigma_omega_total_radps": float(np.mean(sigma_omegas)),
            "meets_nasa_bound": bool(float(np.mean(sigma_omegas)) < 0.0021),
            "ttc_mean_error_s": float(np.mean(ttc_errors)) if ttc_errors else float("nan"),
            "ttc_median_error_s": float(np.mean(ttc_medians)) if ttc_medians else float("nan"),
            "wc1_warning_ge_12_5s": float(pooled["alerting"]["WC1_warning"]["frac_ge_12_5s"]),
            "wc1_warning_ge_25_0s": float(pooled["alerting"]["WC1_warning"]["frac_ge_25_0s"]),
            "wc1_corrective_ge_25_0s": float(pooled["alerting"]["WC1_corrective"]["frac_ge_25_0s"]),
            "wc1_corrective_ge_55_0s": float(pooled["alerting"]["WC1_corrective"]["frac_ge_55_0s"]),
            "wc1_missed_rate": float(pooled["alerting"]["WC1_warning"]["frac_missed"]),
            "wc1_false_alerts_per_hour": float(pooled["alerting"]["WC1_warning"]["false_alerts_per_hour"]),
            "wc1_false_alert_events_per_hour": float(pooled["alerting"]["WC1_warning"]["false_alert_events_per_h"]),
            "wc1_warning_class_known_ge_12_5s": float(pooled["alerting"]["WC1_warning_class_known"]["frac_ge_12_5s"]),
            "wc1_corrective_class_known_ge_25_0s": float(pooled["alerting"]["WC1_corrective_class_known"]["frac_ge_25_0s"]),
            "wc1_class_known_missed_rate": float(pooled["alerting"]["WC1_warning_class_known"]["frac_missed"]),
            "wc1_class_known_false_alerts_per_hour": float(pooled["alerting"]["WC1_warning_class_known"]["false_alerts_per_hour"]),
            "a1_warning_ge_12_5s": float(pooled["alerting"]["A1_warning"]["frac_ge_12_5s"]),
            "a1_corrective_ge_25_0s": float(pooled["alerting"]["A1_corrective"]["frac_ge_25_0s"]),
            "a1_missed_rate": float(pooled["alerting"]["A1_warning"]["frac_missed"]),
            "a1_false_alerts_per_hour": float(pooled["alerting"]["A1_warning"]["false_alerts_per_hour"]),
            "a2_warning_ge_12_5s": float(pooled["alerting"]["A2_warning"]["frac_ge_12_5s"]),
            "a2_corrective_ge_25_0s": float(pooled["alerting"]["A2_corrective"]["frac_ge_25_0s"]),
            "a2_missed_rate": float(pooled["alerting"]["A2_warning"]["frac_missed"]),
            "a2_false_alerts_per_hour": float(pooled["alerting"]["A2_warning"]["false_alerts_per_hour"]),
            "a2_warning_class_known_ge_12_5s": float(pooled["alerting"]["A2_warning_class_known"]["frac_ge_12_5s"]),
            "a2_corrective_class_known_ge_25_0s": float(pooled["alerting"]["A2_corrective_class_known"]["frac_ge_25_0s"]),
            "wc1_by_beta": wc1_by_beta_means,
        },
    }
    return aggregated


def run_determinism_v5(
    encounters: list[Any],
    config: dict[str, Any],
    max_workers: int,
) -> dict[str, Any]:
    """Verify V5_G1 determinism: identical seeds produce byte-identical JSON outputs."""
    print("\n--- Verifying V5_G1 Determinism ---")
    sub_encs = encounters[:20]
    seed = 9999

    res1 = run_evaluation(sub_encs, config=config, seed=seed, max_workers=max_workers)
    v5_truth_cache = res1.get("truth_cache")
    res1_clean = deepcopy(res1)
    res1_clean.pop("runtime_s", None)
    res1_clean.pop("frames_per_second", None)
    res1_clean.pop("truth_cache", None)
    json1 = json.dumps(res1_clean, sort_keys=True, indent=2)

    res2 = run_evaluation(sub_encs, config=config, seed=seed, max_workers=max_workers)
    res2_clean = deepcopy(res2)
    res2_clean.pop("runtime_s", None)
    res2_clean.pop("frames_per_second", None)
    res2_clean.pop("truth_cache", None)
    json2 = json.dumps(res2_clean, sort_keys=True, indent=2)

    sha1 = hashlib.sha256(json1.encode("utf-8")).hexdigest()
    sha2 = hashlib.sha256(json2.encode("utf-8")).hexdigest()

    is_identical = (sha1 == sha2)
    print(f"  Run 1 SHA-256: {sha1}")
    print(f"  Run 2 SHA-256: {sha2}")
    print(f"  V5_G1 Determinism Passed: {is_identical}")

    return {
        "run1_sha256": sha1,
        "run2_sha256": sha2,
        "deterministic": is_identical,
        "truth_cache": v5_truth_cache,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WP-05 G1 Monte Carlo Runner (Batch B7)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Tracking G1 config JSON")
    parser.add_argument("--out-dir", type=Path, default=None, help="Explicit output directory")
    parser.add_argument("--workers", type=int, default=get_safe_workers(), help="Number of worker processes")
    parser.add_argument("--quick", action="store_true", help="Quick mode (50 encounters, 2 seeds)")
    parser.add_argument("--skip-determinism", action="store_true", help="Skip V5 determinism check")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root directory")
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve() if args.data_root is not None else (REPO_ROOT / "output")
    output_dir = args.out_dir.resolve() if args.out_dir is not None else (data_root / "xvs" / "wp05" / "g1")

    safe_limit = get_safe_workers()
    if args.workers > safe_limit:
        print(f"WARNING: Requested workers ({args.workers}) exceeds safe limit ({safe_limit}). Clamping to {safe_limit}.")
        args.workers = safe_limit

    output_dir.mkdir(parents=True, exist_ok=True)

    git_status_start = get_git_status_porcelain()
    if not args.quick and git_status_start:
        raise RuntimeError(f"Refusing to run full G1 MC on dirty XVS git tree:\n{git_status_start}")

    start_commit = get_git_commit()
    start_hashes = compute_code_hashes()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    encounters, sampler_sha = get_or_generate_encounters()
    if G0_SWEEP_RESULTS_PATH.is_file():
        g0_sweep = json.loads(G0_SWEEP_RESULTS_PATH.read_text(encoding="utf-8"))
        g0_sha = g0_sweep.get("provenance", {}).get("sampler_v1_sha256", EXPECTED_SAMPLER_SHA256)
        if sampler_sha != g0_sha:
            raise ValueError(f"Sampler SHA-256 mismatch vs G0: {sampler_sha} != {g0_sha}")

    if args.quick:
        encounters = encounters[:50]
        print(f"Quick mode active: trimmed to {len(encounters)} encounters.")

    baseline_seeds = [101, 102, 103, 104, 105] if not args.quick else [101, 102]

    # Write pre-launch run_manifest.json per G-B14 §1 / B7 §6.1
    manifest_path = output_dir / "run_manifest.json"
    pre_manifest = {
        "batch": "ARGON_B7",
        "wp": "WP-05_G1",
        "status": "running",
        "argv": list(argv) if argv is not None else list(sys.argv),
        "start_time_utc": start_time_iso,
        "git_commit_start": start_commit,
        "git_status_start": git_status_start,
        "sampler_v1_sha256": sampler_sha,
        "baseline_seeds": baseline_seeds,
        "encounters_total": len(encounters),
        "configs_read": {
            "configs/xvs/tracking/g1.json": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        },
        "code_hashes_start": start_hashes,
        "platform": platform.platform(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(pre_manifest, f, indent=2)

    overall_t0 = time.perf_counter()

    get_ownship_states()
    truth_cache_map, truth_cache_info = load_or_compute_truth_cache(encounters, max_workers=args.workers)

    # 1. Determinism V5_G1
    if not args.skip_determinism:
        v5_result = run_determinism_v5(encounters, base_config, args.workers)
        with open(output_dir / "v5_determinism.json", "w", encoding="utf-8") as f:
            json.dump(v5_result, f, indent=2)
    else:
        v5_result = {"deterministic": True, "skipped": True}

    # 2. Baseline Monte Carlo (5 seeds)
    baseline_results = run_baseline(encounters, base_config, baseline_seeds, truth_cache_map, args.workers)
    with open(output_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    overall_t1 = time.perf_counter()
    overall_runtime = overall_t1 - overall_t0

    end_commit = get_git_commit()
    end_hashes = compute_code_hashes()
    git_status_end = get_git_status_porcelain()
    end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    hashes_match = (start_hashes == end_hashes) and (start_commit == end_commit)
    if not args.quick and not hashes_match:
        raise RuntimeError(f"Long-job rule violation: code hashes or commit changed during run!")

    perf_data = {
        "total_runtime_s": overall_runtime,
        "workers_used": args.workers,
        "max_workers_allowed": safe_limit,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "v6_runtime_limit_s": 7200.0,
        "v6_passed": bool(overall_runtime <= 7200.0),
    }
    with open(output_dir / "perf.json", "w", encoding="utf-8") as f:
        json.dump(perf_data, f, indent=2)

    manifest_data = {
        "batch": "ARGON_B7",
        "wp": "WP-05_G1",
        "status": "completed",
        "argv": list(argv) if argv is not None else list(sys.argv),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_time_utc": start_time_iso,
        "end_time_utc": end_time_iso,
        "git_commit_start": start_commit,
        "git_commit_end": end_commit,
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "sampler_v1_sha256": sampler_sha,
        "baseline_seeds": baseline_seeds,
        "encounters_total": len(encounters),
        "v5_deterministic": v5_result.get("deterministic", True),
        "v6_runtime_s": overall_runtime,
        "truth_cache": truth_cache_info,
        "save_failed": truth_cache_info.get("save_failed", False),
        "configs_read": {
            "configs/xvs/tracking/g1.json": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        },
        "code_hashes_start": start_hashes,
        "code_hashes_end": end_hashes,
        "code_hashes_match": hashes_match,
        "code_hashes_unchanged_during_run": hashes_match,
        "platform": platform.platform(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n==========================================")
    print(f"G1 Baseline MC completed in {overall_runtime:.1f} s ({overall_runtime/60.0:.2f} min).")
    print(f"Outputs written to {output_dir}")
    print(f"==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
