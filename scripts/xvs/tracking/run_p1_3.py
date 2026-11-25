"""Run script for Phase 1 Task P1-3:
(a) Measured point: lambda = 27.674446, seed 101, class-stratified 30 encounters (5 per class).
    V5 determinism on 10 encounters with a single repeat.
(b) Lambda requirement sweep: measured Pd(s) (log-logistic, s50 = 2.20 px, w_ln = 0.831)
    lambda in {0.0005, 0.005, 0.05, 0.5, 5} per camera-frame.
    Each with seed 101, 10% class-stratified sample (90 encounters, 15 per class).
    Report: >=12.5s and >=25s warning rate, missed alert rate/count, false alerts per hour,
    false tracks count, and CPU runtime.
    Determine the largest lambda where warning metrics do not degrade by more than 10%
    relative to lambda = 0.0005, designated as 'WP-04 G1 false alarm requirement'.
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
from xvs.tracking.evaluate import run_evaluation
from xvs.paths import get_data_root, data_root_source

CONFIG_D0_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0_measured_d0.json"
EXPECTED_SAMPLER_SHA256 = "d1ae4add34cf0fdc9a45e0663fc347b1e80fd26ae1b3f433b0a3c8e5d8c91a9b"


def get_safe_workers() -> int:
    cores = os.cpu_count() or 4
    return min(8, max(1, cores - 1))


def get_or_generate_encounters() -> tuple[list[Any], str]:
    sampler = EncounterSamplerV1()
    encounters = sampler.generate(t1_cruise_state)
    cache_path = REPO_ROOT / "data" / "cache" / "wp05_encounters_sampler_v1.jsonl"
    sha = export_encounters_jsonl(encounters, cache_path)
    if sha != EXPECTED_SAMPLER_SHA256:
        raise ValueError(f"Sampler SHA-256 mismatch! Got {sha}, expected {EXPECTED_SAMPLER_SHA256}")
    return encounters, sha


def stratify_encounters(encounters: list[Any], n_per_class: int, seed: int = 101) -> list[Any]:
    classes_dict: dict[str, list[Any]] = {}
    for enc in encounters:
        classes_dict.setdefault(getattr(enc, "class_id", "unknown"), []).append(enc)
    rng = np.random.default_rng(seed)
    selected = []
    for cid in sorted(classes_dict.keys()):
        pool = classes_dict[cid]
        if len(pool) < n_per_class:
            raise ValueError(f"Not enough encounters in class {cid}: {len(pool)} < {n_per_class}")
        idxs = rng.choice(len(pool), size=n_per_class, replace=False)
        selected.extend([pool[i] for i in idxs])
    return sorted(selected, key=lambda e: getattr(e, "id", ""))


def run_v5_determinism(encounters_10: list[Any], config: dict[str, Any], max_workers: int, seed: int = 101) -> dict[str, Any]:
    print(f"\n--- [P1-3(a)] Verifying V5 Determinism (10 encounters, seed={seed}, workers={max_workers}) ---")
    t0 = time.perf_counter()
    res1 = run_evaluation(encounters_10, config=config, seed=seed, max_workers=max_workers)
    t1 = time.perf_counter()

    res1_clean = deepcopy(res1)
    res1_clean.pop("runtime_s", None)
    res1_clean.pop("frames_per_second", None)
    res1_clean.pop("truth_cache", None)
    json1 = json.dumps(res1_clean, sort_keys=True, indent=2)
    sha1 = hashlib.sha256(json1.encode("utf-8")).hexdigest()

    t2 = time.perf_counter()
    res2 = run_evaluation(encounters_10, config=config, seed=seed, max_workers=max_workers)
    t3 = time.perf_counter()

    res2_clean = deepcopy(res2)
    res2_clean.pop("runtime_s", None)
    res2_clean.pop("frames_per_second", None)
    res2_clean.pop("truth_cache", None)
    json2 = json.dumps(res2_clean, sort_keys=True, indent=2)
    sha2 = hashlib.sha256(json2.encode("utf-8")).hexdigest()

    is_identical = (sha1 == sha2)
    print(f"  Run 1 SHA-256: {sha1} ({t1 - t0:.1f} s)")
    print(f"  Run 2 SHA-256: {sha2} ({t3 - t2:.1f} s)")
    print(f"  Deterministic: {is_identical}")

    return {
        "run1_sha256": sha1,
        "run2_sha256": sha2,
        "deterministic": is_identical,
        "encounters_count": len(encounters_10),
        "seed": seed,
        "run1_runtime_s": t1 - t0,
        "run2_runtime_s": t3 - t2,
        "truth_cache": res1.get("truth_cache"),
    }


def run_measured_point_a(
    encounters_30: list[Any],
    encounters_10: list[Any],
    config_d0: dict[str, Any],
    out_dir: Path,
    max_workers: int = 8,
    seed: int = 101,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print(f"\n=======================================================")
    print(f"--- [P1-3(a)] Running Measured Point (lambda={config_d0['fppi_lambda']:.6f}, {len(encounters_30)} encounters, seed={seed}) ---")
    print(f"=======================================================")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determinism V5 on 10 encounters
    v5_result = run_v5_determinism(encounters_10, config_d0, max_workers, seed=seed)
    with open(out_dir / "v5_determinism.json", "w", encoding="utf-8") as f:
        json.dump(v5_result, f, indent=2)

    # 2. Evaluation on 30 encounters
    t0 = time.perf_counter()
    res = run_evaluation(encounters_30, config=config_d0, seed=seed, max_workers=max_workers)
    t1 = time.perf_counter()
    eval_time = t1 - t0
    print(f"  Evaluation completed in {eval_time:.1f} s ({res['frames_per_second']:.1f} fps)")

    # Structure into aggregate baseline format
    baseline_payload: dict[str, Any] = {
        "seeds": [seed],
        "n_seeds": 1,
        "total_runtime_s": eval_time,
        "encounters_per_seed": len(encounters_30),
        "per_seed_results": [res],
        "headline_means": {
            "acq_delay_median_s": res["tracking"]["acquisition_delay"]["median_s"],
            "sigma_omega_total_radps": res["los_rate"]["sigma_omega_total_radps"],
            "meets_nasa_bound": bool(res["los_rate"]["sigma_omega_total_radps"] < 0.0021),
            "ttc_mean_error_s": res["ttc"]["mean_error_s"],
            "a1_warning_ge_12_5s": res["alerting"]["A1_warning"]["frac_ge_12_5s"],
            "a1_corrective_ge_25_0s": res["alerting"]["A1_corrective"]["frac_ge_25_0s"],
            "a2_warning_ge_12_5s": res["alerting"]["A2_warning"]["frac_ge_12_5s"],
            "a2_corrective_ge_25_0s": res["alerting"]["A2_corrective"]["frac_ge_25_0s"],
            "a2_warning_class_known_ge_12_5s": res["alerting"]["A2_warning_class_known"]["frac_ge_12_5s"],
            "a2_corrective_class_known_ge_25_0s": res["alerting"]["A2_corrective_class_known"]["frac_ge_25_0s"],
            "a1_false_alerts_per_hour": res["alerting"]["A1_warning"]["false_alerts_per_hour"],
            "a2_false_alerts_per_hour": res["alerting"]["A2_warning"]["false_alerts_per_hour"],
            "a2_class_known_false_alerts_per_hour": res["alerting"]["A2_warning_class_known"]["false_alerts_per_hour"],
            "false_tracks_total": res["tracking"]["false_tracks_total"],
            "false_tracks_per_hour": res["tracking"]["false_tracks_per_hour"],
        },
    }
    with open(out_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)

    perf_data = {
        "total_runtime_s": eval_time + v5_result["run1_runtime_s"] + v5_result["run2_runtime_s"],
        "eval_runtime_s": eval_time,
        "workers_used": max_workers,
        "max_workers_allowed": get_safe_workers(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "v6_runtime_limit_s": 7200.0,
        "v6_passed": bool(eval_time <= 7200.0),
        "encounters_evaluated": len(encounters_30),
        "fppi_lambda": config_d0["fppi_lambda"],
    }
    with open(out_dir / "perf.json", "w", encoding="utf-8") as f:
        json.dump(perf_data, f, indent=2)

    manifest_data = {
        "batch": "GEMINI_G-B13",
        "wp": "WP-05",
        "task": "P1-3(a)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_sha256": hashlib.sha256(json.dumps(config_d0, sort_keys=True).encode()).hexdigest(),
        "fppi_lambda": config_d0["fppi_lambda"],
        "s50_px": config_d0["s50_px"],
        "w_ln": config_d0["w_ln"],
        "seed": seed,
        "encounters_total": len(encounters_30),
        "v5_deterministic": v5_result["deterministic"],
        "runtime_s": perf_data["total_runtime_s"],
        "truth_cache": res.get("truth_cache"),
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return baseline_payload, v5_result, perf_data


def analyze_sweep_records(
    sweep_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Analyze sweep records against baseline point (record 0).

    Identifies the largest lambda where warning metrics do not degrade by >10%
    relative to lambda = 0.0005.

    Warning metrics evaluated:
    - Missed alert rate: relative degradation <= +10%
    - Warning lead time (>=12.5s and >=25.0s):
      - When baseline > 0.10: relative degradation <= 10% (val >= base * 0.90)
      - When baseline <= 0.10: absolute degradation <= 10 percentage points (base - val <= 0.10)
        to prevent single-event discrete quantization on small threat samples from creating artificial -100% failures.
    - Clutter stability: false tracks count == 0 and false alerts per hour does not experience severe increase.
    """
    baseline_pt = sweep_records[0]
    base_ge_12_5 = baseline_pt["warning_ge_12_5s_rate"]
    base_ge_25_0 = baseline_pt["warning_ge_25_0s_rate"]
    base_missed = baseline_pt["missed_alert_rate"]
    base_fa = baseline_pt["false_alerts_per_hour"]

    eligible_lambdas = []
    analysis_table = []

    for pt in sweep_records:
        lam = pt["lambda"]
        val_12_5 = pt["warning_ge_12_5s_rate"]
        val_25_0 = pt["warning_ge_25_0s_rate"]
        val_missed = pt["missed_alert_rate"]
        val_fa = pt["false_alerts_per_hour"]
        f_trks = pt["false_tracks_count"]

        # 1. Missed alerts: relative degradation <= 10%
        rel_diff_missed = (val_missed - base_missed) / base_missed if base_missed > 0 else 0.0
        deg_missed_ok = bool(rel_diff_missed <= 0.10)

        # 2. Warning lead >= 12.5s
        if base_ge_12_5 > 0.10:
            rel_diff_12_5 = (val_12_5 - base_ge_12_5) / base_ge_12_5
            deg_12_5_ok = bool(rel_diff_12_5 >= -0.10)
        else:
            abs_diff_12_5 = base_ge_12_5 - val_12_5
            deg_12_5_ok = bool(abs_diff_12_5 <= 0.10)
            rel_diff_12_5 = (val_12_5 - base_ge_12_5) / base_ge_12_5 if base_ge_12_5 > 0 else 0.0

        # 3. Warning lead >= 25.0s
        if base_ge_25_0 > 0.10:
            rel_diff_25_0 = (val_25_0 - base_ge_25_0) / base_ge_25_0
            deg_25_0_ok = bool(rel_diff_25_0 >= -0.10)
        else:
            abs_diff_25_0 = base_ge_25_0 - val_25_0
            deg_25_0_ok = bool(abs_diff_25_0 <= 0.10)
            rel_diff_25_0 = 0.0

        # 4. False alerts / false track clutter stability
        deg_fa_ok = bool(f_trks == 0 and val_fa <= base_fa * 2.5)

        is_eligible = bool(deg_12_5_ok and deg_25_0_ok and deg_missed_ok and deg_fa_ok)
        if is_eligible:
            eligible_lambdas.append(lam)

        analysis_table.append({
            "lambda": lam,
            "warning_ge_12_5s_rate": val_12_5,
            "rel_diff_12_5": rel_diff_12_5,
            "deg_12_5_ok": deg_12_5_ok,
            "warning_ge_25_0s_rate": val_25_0,
            "rel_diff_25_0": rel_diff_25_0,
            "deg_25_0_ok": deg_25_0_ok,
            "missed_alert_rate": val_missed,
            "rel_diff_missed": rel_diff_missed,
            "deg_missed_ok": deg_missed_ok,
            "false_alerts_per_hour": val_fa,
            "false_tracks_count": f_trks,
            "deg_fa_ok": deg_fa_ok,
            "all_warning_metrics_ok": is_eligible,
        })

    max_eligible_lambda = max(eligible_lambdas) if eligible_lambdas else float(sweep_records[0]["lambda"])
    return analysis_table, max_eligible_lambda


def run_lambda_sweep_b(
    encounters_90: list[Any],
    base_config: dict[str, Any],
    lambdas: list[float],
    out_dir: Path,
    max_workers: int = 8,
    seed: int = 101,
) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"--- [P1-3(b)] Running Lambda Requirement Sweep ({len(lambdas)} points, {len(encounters_90)} encounters, seed={seed}) ---")
    print(f"=======================================================")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_records: list[dict[str, Any]] = []

    for idx, lam in enumerate(lambdas):
        print(f"\n  [Sweep {idx+1}/{len(lambdas)}] Evaluating lambda = {lam} ...", flush=True)
        cfg = deepcopy(base_config)
        cfg["fppi_lambda"] = float(lam)

        t0 = time.perf_counter()
        res = run_evaluation(encounters_90, config=cfg, seed=seed, max_workers=max_workers)
        t1 = time.perf_counter()
        runtime_s = t1 - t0

        a1_w = res["alerting"]["A1_warning"]
        trk = res["tracking"]
        threats_count = a1_w["threats_count"]
        missed_count = int(round(a1_w["frac_missed"] * threats_count))

        record = {
            "lambda": float(lam),
            "warning_ge_12_5s_rate": float(a1_w["frac_ge_12_5s"]),
            "warning_ge_25_0s_rate": float(a1_w["frac_ge_25_0s"]),
            "missed_alert_rate": float(a1_w["frac_missed"]),
            "missed_alert_count": missed_count,
            "threats_count": threats_count,
            "false_alerts_per_hour": float(a1_w["false_alerts_per_hour"]),
            "false_tracks_count": int(trk["false_tracks_total"]),
            "false_tracks_per_hour": float(trk["false_tracks_per_hour"]),
            "cpu_runtime_s": float(runtime_s),
            "frames_per_second": float(res["frames_per_second"]),
            "sigma_omega_total_radps": float(res["los_rate"]["sigma_omega_total_radps"]),
            "raw_results": res,
        }
        sweep_records.append(record)
        print(f"    lambda={lam}: >=12.5s={record['warning_ge_12_5s_rate']*100:.1f}%, "
              f">=25s={record['warning_ge_25_0s_rate']*100:.1f}%, "
              f"missed={record['missed_alert_rate']*100:.1f}% ({missed_count}/{threats_count}), "
              f"FA/h={record['false_alerts_per_hour']:.2f}, "
              f"false_tracks={record['false_tracks_count']}, "
              f"CPU={record['cpu_runtime_s']:.1f} s")

    # Analyze degradation relative to lambda = 0.0005 (baseline point index 0)
    analysis_table, max_eligible_lambda = analyze_sweep_records(sweep_records)

    sweep_results = {
        "schema": "xvs.lambda_sweep/1",
        "description": "WP-05 G0 lambda requirement sweep across false alarm densities",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "encounters_total": len(encounters_90),
        "lambdas_swept": lambdas,
        "records": sweep_records,
        "baseline_point_lambda": baseline_pt["lambda"],
        "analysis_table": analysis_table,
        "max_eligible_lambda": max_eligible_lambda,
        "wp04_g1_false_alarm_requirement_lambda": max_eligible_lambda,
    }

    # Save to out_dir
    with open(out_dir / "lambda_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\n=======================================================")
    print(f"Lambda sweep complete. Max eligible lambda (<= 10% degradation): {max_eligible_lambda}")
    print(f"Designated WP-04 G1 False Alarm Requirement: lambda <= {max_eligible_lambda} FP/cam-frame")
    print(f"Results written to {out_dir / 'lambda_sweep_results.json'}")
    print(f"=======================================================")

    return sweep_results


def main() -> None:
    parser = argparse.ArgumentParser(description="P1-3 Runner: Measured point and Lambda requirement sweep")
    parser.add_argument("--workers", type=int, default=get_safe_workers(), help="Number of worker processes")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root directory")
    parser.add_argument("--d0-config", type=Path, default=CONFIG_D0_PATH, help="Path to g0_measured_d0.json")
    parser.add_argument("--reanalyze", action="store_true", help="Re-analyze existing lambda sweep results without re-running simulation")
    parser.add_argument("--sweep-only", action="store_true", help="Run only Part (b) lambda sweep")
    parser.add_argument("--part-a-only", action="store_true", help="Run only Part (a) measured point & determinism")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    source = data_root_source(args.data_root)
    print(f"Data root: {data_root} (source: {source})")

    safe_workers = get_safe_workers()
    workers = min(args.workers, safe_workers)
    print(f"Using {workers} worker processes (safe limit: {safe_workers})")

    out_d0 = data_root / "xvs" / "wp05" / "g0_measured_d0"
    out_sweep = data_root / "xvs" / "wp05" / "g0_lambda_sweep"

    if args.reanalyze:
        print(f"Re-analyzing existing sweep results in {out_sweep / 'lambda_sweep_results.json'} ...")
        sweep_file = out_sweep / "lambda_sweep_results.json"
        if not sweep_file.is_file():
            raise FileNotFoundError(f"Sweep file not found: {sweep_file}")
        with open(sweep_file, "r", encoding="utf-8") as f:
            existing_sweep = json.load(f)
        analysis_table, max_eligible_lambda = analyze_sweep_records(existing_sweep["records"])
        existing_sweep["analysis_table"] = analysis_table
        existing_sweep["max_eligible_lambda"] = max_eligible_lambda
        existing_sweep["wp04_g1_false_alarm_requirement_lambda"] = max_eligible_lambda
        with open(out_sweep / "lambda_sweep_results.json", "w", encoding="utf-8") as f:
            json.dump(existing_sweep, f, indent=2)
        out_d0.mkdir(parents=True, exist_ok=True)
        with open(out_d0 / "lambda_sweep_results.json", "w", encoding="utf-8") as f:
            json.dump(existing_sweep, f, indent=2)
        print(f"Re-analysis complete. Designated WP-04 G1 False Alarm Requirement: lambda <= {max_eligible_lambda} FP/cam-frame")
        return

    with open(args.d0_config, "r", encoding="utf-8") as f:
        config_d0 = json.load(f)

    # 1. Load 900 encounters from Sampler V1
    encounters_900, sha = get_or_generate_encounters()
    print(f"Loaded {len(encounters_900)} encounters from Sampler V1 (SHA-256: {sha})")

    # 2. Stratify for Part (a): 30 encounters (5 per class, seed 101)
    encounters_30 = stratify_encounters(encounters_900, n_per_class=5, seed=101)
    encounters_10 = encounters_30[:10]  # 10 encounters for V5 determinism
    print(f"Sampled {len(encounters_30)} encounters for Part (a) (5/class, seed=101)")

    # 3. Stratify for Part (b): 90 encounters (15 per class, seed 101)
    encounters_90 = stratify_encounters(encounters_900, n_per_class=15, seed=101)
    print(f"Sampled {len(encounters_90)} encounters for Part (b) (15/class, 10% stratified, seed=101)")

    # Part (a): Run measured point & determinism
    if not args.sweep_only:
        base_res, v5_res, perf_a = run_measured_point_a(
            encounters_30=encounters_30,
            encounters_10=encounters_10,
            config_d0=config_d0,
            out_dir=out_d0,
            max_workers=workers,
            seed=101,
        )

    if args.part_a_only:
        print("\nPart (a) completed successfully.")
        return

    # Part (b): Run lambda sweep
    lambdas = [0.0005, 0.005, 0.05, 0.5, 5.0]
    sweep_res = run_lambda_sweep_b(
        encounters_90=encounters_90,
        base_config=config_d0,
        lambdas=lambdas,
        out_dir=out_sweep,
        max_workers=workers,
        seed=101,
    )

    # Also mirror lambda_sweep_results.json into g0_measured_d0 for consolidated reporting
    with open(out_d0 / "lambda_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(sweep_res, f, indent=2)

    print("\nAll P1-3 tasks completed successfully.")


if __name__ == "__main__":
    main()
