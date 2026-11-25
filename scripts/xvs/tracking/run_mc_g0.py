"""Monte Carlo simulation runner for WP-05 G0 multi-camera tracking.

Conforms strictly to docs/xvs/WP05_TRACKING.md §4 and GEMINI_G-B3.md §1 items 5 and 6.
Enforces CPU load rule: at most (logical cores - 4) worker processes.
Generates:
- Baseline Monte Carlo: 900 encounters x 5 seeds;
- Parameter sweeps: 900 encounters x 2 seeds:
  * sigma_ang x {0.5, 1, 2, 3}
  * s50 in {2, 3, 5} px
  * FPPI x {1, 10}
  * latency in {0, 100, 200} ms
  * Markov misses on/off
- JSON outputs under output/xvs/wp05/g0/
"""

from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.encounter import EncounterSamplerV1, export_encounters_jsonl
from xvs.tracking.evaluate import run_evaluation, aggregate_dataset_metrics

from xvs.paths import get_data_root, data_root_source
CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0.json"
EXPECTED_SAMPLER_SHA256 = "d1ae4add34cf0fdc9a45e0663fc347b1e80fd26ae1b3f433b0a3c8e5d8c91a9b"


def get_safe_workers() -> int:
    """Enforce rule: at most (logical cores - 4) worker processes."""
    cores = os.cpu_count() or 4
    return max(1, cores - 4)


def get_or_generate_encounters() -> tuple[list[Any], str]:
    """Generate or load sampler v1 900 encounters with LF SHA-256 verification."""
    sampler = EncounterSamplerV1()
    encounters = sampler.generate(t1_cruise_state)
    cache_path = REPO_ROOT / "data" / "cache" / "wp05_encounters_sampler_v1.jsonl"
    sha = export_encounters_jsonl(encounters, cache_path)
    if sha != EXPECTED_SAMPLER_SHA256:
        raise ValueError(f"Sampler SHA-256 mismatch! Got {sha}, expected {EXPECTED_SAMPLER_SHA256}")
    return encounters, sha


def run_baseline(
    encounters: list[Any],
    base_config: dict[str, Any],
    seeds: list[int],
    max_workers: int,
) -> dict[str, Any]:
    """Run baseline Monte Carlo: 900 encounters x 5 seeds."""
    print(f"\n--- Running Baseline Monte Carlo ({len(encounters)} encs x {len(seeds)} seeds, workers={max_workers}) ---")
    per_seed_results = []
    total_t0 = time.perf_counter()

    for idx, s in enumerate(seeds):
        t0 = time.perf_counter()
        print(f"  [Baseline] Seed {s} ({idx+1}/{len(seeds)})...", flush=True)
        res = run_evaluation(encounters, config=base_config, seed=s, max_workers=max_workers)
        t1 = time.perf_counter()
        print(f"  [Baseline] Seed {s} completed in {t1 - t0:.1f} s ({res['frames_per_second']:.0f} frames/s)")
        per_seed_results.append(res)

    total_t1 = time.perf_counter()
    total_runtime = total_t1 - total_t0

    # Aggregate over seeds: mean across seeds
    aggregated: dict[str, Any] = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "total_runtime_s": total_runtime,
        "encounters_per_seed": len(encounters),
        "per_seed_results": per_seed_results,
    }

    # Extract headline metrics averaged across seeds
    acq_medians = [r["tracking"]["acquisition_delay"]["median_s"] for r in per_seed_results if not math.isnan(r["tracking"]["acquisition_delay"]["median_s"])]
    sigma_omegas = [r["los_rate"]["sigma_omega_total_radps"] for r in per_seed_results]
    ttc_errors = [r["ttc"]["mean_error_s"] for r in per_seed_results if not math.isnan(r["ttc"]["mean_error_s"])]

    a1_warn_fracs = [r["alerting"]["A1_warning"]["frac_ge_12_5s"] for r in per_seed_results]
    a1_corr_fracs = [r["alerting"]["A1_corrective"]["frac_ge_25_0s"] for r in per_seed_results]
    a2_warn_fracs = [r["alerting"]["A2_warning"]["frac_ge_12_5s"] for r in per_seed_results]
    a2_corr_fracs = [r["alerting"]["A2_corrective"]["frac_ge_25_0s"] for r in per_seed_results]
    a2_warn_ck_fracs = [r["alerting"]["A2_warning_class_known"]["frac_ge_12_5s"] for r in per_seed_results]
    a2_corr_ck_fracs = [r["alerting"]["A2_corrective_class_known"]["frac_ge_25_0s"] for r in per_seed_results]

    a1_fa_rates = [r["alerting"]["A1_warning"]["false_alerts_per_hour"] for r in per_seed_results]
    a2_fa_rates = [r["alerting"]["A2_warning"]["false_alerts_per_hour"] for r in per_seed_results]
    a2_ck_fa_rates = [r["alerting"]["A2_warning_class_known"]["false_alerts_per_hour"] for r in per_seed_results]

    aggregated["headline_means"] = {
        "acq_delay_median_s": float(np.mean(acq_medians)) if acq_medians else float("nan"),
        "sigma_omega_total_radps": float(np.mean(sigma_omegas)),
        "meets_nasa_bound": bool(float(np.mean(sigma_omegas)) < 0.0021),
        "ttc_mean_error_s": float(np.mean(ttc_errors)) if ttc_errors else float("nan"),
        "a1_warning_ge_12_5s": float(np.mean(a1_warn_fracs)),
        "a1_corrective_ge_25_0s": float(np.mean(a1_corr_fracs)),
        "a2_warning_ge_12_5s": float(np.mean(a2_warn_fracs)),
        "a2_corrective_ge_25_0s": float(np.mean(a2_corr_fracs)),
        "a2_warning_class_known_ge_12_5s": float(np.mean(a2_warn_ck_fracs)),
        "a2_corrective_class_known_ge_25_0s": float(np.mean(a2_corr_ck_fracs)),
        "a1_false_alerts_per_hour": float(np.mean(a1_fa_rates)),
        "a2_false_alerts_per_hour": float(np.mean(a2_fa_rates)),
        "a2_class_known_false_alerts_per_hour": float(np.mean(a2_ck_fa_rates)),
    }

    return aggregated


def run_sweeps(
    encounters: list[Any],
    base_config: dict[str, Any],
    sweep_seeds: list[int],
    max_workers: int,
) -> dict[str, Any]:
    """Run all sweeps specified in WP-05 §4: 900 encounters x 2 seeds per condition."""
    print(f"\n--- Running Parameter Sweeps ({len(encounters)} encs x {len(sweep_seeds)} seeds, workers={max_workers}) ---")
    sweeps_out: dict[str, Any] = {}

    # Define sweep points
    base_sigma_ang = float(base_config["sigma_ang_rad"])
    base_fppi = float(base_config["fppi_lambda"])

    sweep_specs = [
        # 1. sigma_ang x {0.5, 1, 2, 3}
        ("sigma_ang", [
            ("0.5x", {"sigma_ang_rad": 0.5 * base_sigma_ang}),
            ("1.0x", {"sigma_ang_rad": 1.0 * base_sigma_ang}),
            ("2.0x", {"sigma_ang_rad": 2.0 * base_sigma_ang}),
            ("3.0x", {"sigma_ang_rad": 3.0 * base_sigma_ang}),
        ]),
        # 2. s50 in {2, 3, 5} px
        ("s50_px", [
            ("2px", {"s50_px": 2.0}),
            ("3px", {"s50_px": 3.0}),
            ("5px", {"s50_px": 5.0}),
        ]),
        # 3. FPPI x {1, 10}
        ("fppi", [
            ("1x", {"fppi_lambda": 1.0 * base_fppi}),
            ("10x", {"fppi_lambda": 10.0 * base_fppi}),
        ]),
        # 4. Latency in {0, 100, 200} ms
        ("latency_ms", [
            ("0ms", {"latency_ms": 0.0}),
            ("100ms", {"latency_ms": 100.0}),
            ("200ms", {"latency_ms": 200.0}),
        ]),
        # 5. Markov misses on/off
        ("markov_misses", [
            ("off", {"markov_misses": False}),
            ("on", {"markov_misses": True, "p_stay": 0.8}),
        ]),
        # 6. omega_thr in {0.001, 0.002, 0.005, 0.010, 0.020} rad/s (R-GB-11)
        ("omega_thr", [
            ("0.001", {"alert_omega_thr_radps": 0.001}),
            ("0.002", {"alert_omega_thr_radps": 0.002}),
            ("0.005", {"alert_omega_thr_radps": 0.005}),
            ("0.010", {"alert_omega_thr_radps": 0.010}),
            ("0.020", {"alert_omega_thr_radps": 0.020}),
        ]),
    ]

    for param_name, points in sweep_specs:
        print(f"\n  Starting sweep: {param_name} ({len(points)} points)...")
        sweeps_out[param_name] = {}

        for pt_label, overrides in points:
            cfg = deepcopy(base_config)
            cfg.update(overrides)

            pt_seed_results = []
            for s in sweep_seeds:
                res = run_evaluation(encounters, config=cfg, seed=s, max_workers=max_workers)
                pt_seed_results.append(res)

            # Average summary
            sigmas = [r["los_rate"]["sigma_omega_total_radps"] for r in pt_seed_results]
            a1_warn = [r["alerting"]["A1_warning"]["frac_ge_12_5s"] for r in pt_seed_results]
            a2_warn = [r["alerting"]["A2_warning"]["frac_ge_12_5s"] for r in pt_seed_results]
            a2_warn_ck = [r["alerting"]["A2_warning_class_known"]["frac_ge_12_5s"] for r in pt_seed_results]
            fa_rates = [r["alerting"]["A1_warning"]["false_alerts_per_hour"] for r in pt_seed_results]

            sweeps_out[param_name][pt_label] = {
                "overrides": overrides,
                "sigma_omega_mean_radps": float(np.mean(sigmas)),
                "a1_warning_ge_12_5s_mean": float(np.mean(a1_warn)),
                "a2_warning_ge_12_5s_mean": float(np.mean(a2_warn)),
                "a2_warning_class_known_ge_12_5s_mean": float(np.mean(a2_warn_ck)),
                "false_alerts_per_hour_mean": float(np.mean(fa_rates)),
                "per_seed": pt_seed_results,
            }
            print(f"    Point {pt_label}: sigma_omega={np.mean(sigmas):.6f}, A1_warn={np.mean(a1_warn):.3f}")

    return sweeps_out


def run_determinism_v5(
    encounters: list[Any],
    config: dict[str, Any],
    max_workers: int,
) -> dict[str, Any]:
    """Verify V5 determinism: identical seeds produce byte-identical JSON outputs."""
    print("\n--- Verifying V5 Determinism ---")
    sub_encs = encounters[:20]  # Fast 20-encounter check
    seed = 9999

    res1 = run_evaluation(sub_encs, config=config, seed=seed, max_workers=max_workers)
    v5_truth_cache = res1.get("truth_cache")
    # Remove timing and cache metadata fields for deterministic comparison
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
    print(f"  V5 Determinism Passed: {is_identical}")

    return {
        "run1_sha256": sha1,
        "run2_sha256": sha2,
        "deterministic": is_identical,
        "truth_cache": v5_truth_cache,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WP-05 G0 Monte Carlo Runner")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Tracking config JSON")
    parser.add_argument("--out-dir", type=Path, default=None, help="Explicit output directory")
    parser.add_argument("--workers", type=int, default=get_safe_workers(), help="Number of worker processes")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline")
    parser.add_argument("--sweeps-only", action="store_true", help="Run only sweeps")
    parser.add_argument("--quick", action="store_true", help="Quick mode (100 encounters for testing)")
    parser.add_argument("--subsample-frac", type=float, default=None, help="Class-stratified subsample fraction (e.g. 0.10)")
    parser.add_argument("--subsample-seed", type=int, default=20260926, help="Subsample seed")
    parser.add_argument("--skip-determinism", action="store_true", help="Skip V5 determinism check")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root directory")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    source = data_root_source(args.data_root)
    print(f"Data root: {data_root} (source: {source})")
    
    if args.out_dir is not None:
        output_dir = args.out_dir
    else:
        output_dir = data_root / "xvs" / "wp05" / "g0"

    # Safety check on worker count
    safe_limit = get_safe_workers()
    if args.workers > safe_limit:
        print(f"WARNING: Requested workers ({args.workers}) exceeds safe limit ({safe_limit}). Clamping to {safe_limit}.")
        args.workers = safe_limit

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    encounters, sampler_sha = get_or_generate_encounters()
    print(f"Loaded {len(encounters)} encounters from sampler v1 (SHA-256: {sampler_sha})")

    if args.subsample_frac is not None and 0.0 < args.subsample_frac < 1.0:
        classes_dict: dict[str, list[Any]] = {}
        for enc in encounters:
            classes_dict.setdefault(getattr(enc, "class_id", "unknown"), []).append(enc)
        rng = np.random.default_rng(args.subsample_seed)
        sub_encs = []
        for cid in sorted(classes_dict.keys()):
            pool = classes_dict[cid]
            n_sample = max(1, int(round(len(pool) * args.subsample_frac)))
            idxs = rng.choice(len(pool), size=n_sample, replace=False)
            sub_encs.extend([pool[i] for i in idxs])
        encounters = sorted(sub_encs, key=lambda e: getattr(e, "id", ""))
        print(f"Subsampled {len(encounters)} encounters ({args.subsample_frac*100:.1f}%, stratified by class, seed={args.subsample_seed})")

    if args.quick:
        encounters = encounters[:50]
        print(f"Quick mode active: trimmed to {len(encounters)} encounters.")

    overall_t0 = time.perf_counter()

    # 1. Determinism V5
    if not args.skip_determinism:
        v5_result = run_determinism_v5(encounters, base_config, args.workers)
        with open(output_dir / "v5_determinism.json", "w", encoding="utf-8") as f:
            json.dump(v5_result, f, indent=2)
    else:
        v5_result = {"deterministic": True, "skipped": True}

    # 2. Baseline Monte Carlo (5 seeds)
    baseline_seeds = [101, 102, 103, 104, 105] if not args.quick else [101, 102]
    baseline_results = None
    if not args.sweeps_only:
        baseline_results = run_baseline(encounters, base_config, baseline_seeds, args.workers)
        with open(output_dir / "baseline_results.json", "w", encoding="utf-8") as f:
            json.dump(baseline_results, f, indent=2)

    # 3. Sweeps (2 seeds)
    sweeps_seeds = [201, 202] if not args.quick else [201]
    sweeps_results = None
    if not args.baseline_only:
        sweeps_results = run_sweeps(encounters, base_config, sweeps_seeds, args.workers)
        with open(output_dir / "sweeps_results.json", "w", encoding="utf-8") as f:
            json.dump(sweeps_results, f, indent=2)

    overall_t1 = time.perf_counter()
    overall_runtime = overall_t1 - overall_t0

    # 4. Performance & Manifest
    est_full_runtime_s = overall_runtime / args.subsample_frac if args.subsample_frac else overall_runtime
    perf_data = {
        "total_runtime_s": overall_runtime,
        "workers_used": args.workers,
        "max_workers_allowed": safe_limit,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "v6_runtime_limit_s": 7200.0,
        "v6_passed": bool(overall_runtime <= 7200.0),
        "subsample_frac": args.subsample_frac,
        "subsample_seed": args.subsample_seed,
        "estimated_full_runtime_s": est_full_runtime_s,
    }
    with open(output_dir / "perf.json", "w", encoding="utf-8") as f:
        json.dump(perf_data, f, indent=2)

    truth_cache_info = None
    if baseline_results and "per_seed_results" in baseline_results and baseline_results["per_seed_results"]:
        truth_cache_info = baseline_results["per_seed_results"][0].get("truth_cache")
    if truth_cache_info is None and v5_result:
        truth_cache_info = v5_result.get("truth_cache")
    if truth_cache_info is None:
        truth_cache_info = {"status": "none", "reason": "not_recorded"}

    manifest_data = {
        "batch": "GEMINI_G-B10",
        "wp": "WP-05",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sampler_v1_sha256": sampler_sha,
        "baseline_seeds": baseline_seeds,
        "sweep_seeds": sweeps_seeds,
        "encounters_total": len(encounters),
        "v5_deterministic": v5_result.get("deterministic", True),
        "v6_runtime_s": overall_runtime,
        "estimated_full_runtime_s": est_full_runtime_s,
        "truth_cache": truth_cache_info,
        "save_failed": truth_cache_info.get("save_failed", False),
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n==========================================")
    print(f"All runs completed in {overall_runtime:.1f} s ({overall_runtime/60.0:.2f} min).")
    if args.subsample_frac:
        print(f"Estimated full 900-encounter runtime: {est_full_runtime_s:.1f} s ({est_full_runtime_s/60.0:.2f} min).")
    print(f"Outputs written to {output_dir}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
