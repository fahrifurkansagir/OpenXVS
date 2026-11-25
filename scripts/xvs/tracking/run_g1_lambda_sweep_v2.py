"""WP-05 G1 / Batch B7 — Lambda Sweep v2 Runner under Decisions D29 & D32.

Implements:
- Common random numbers (CRN) across lambda in {0.0005, 0.005, 0.05, 0.5, 1.0, 2.0, 5.0} FP/cam-frame
  over all 900 stratified encounters x 5 seeds (with automatic 2-seed fallback if a lambda's pilot
  projects > 1.5 h wall time).
- Primary alerting rule: G1 Passive Well-Clear (WC1) from src/xvs/tracking/risk.py, with legacy A1/A2
  retained for paired diagnostics.
- Paired differences (G1 - G0) at each lambda against output/xvs/wp05/g0_lambda_sweep_v2/per_run_records.jsonl
  (verified by SHA-256) via 2,000-resample seeded encounter-level bootstrap.
- W7 miss decomposition at t_CPA - 12.5 s for WC1 (and legacy A1), separating rule-limited vs
  estimation-limited misses against ground truth.
- D29 fixed-sequence pass rule (stop at first failure) with D32 ever-confirmed false tracks per flight hour.
- Long-job manifest integrity (G-B14 §1 / B7 §6.1).
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
import shutil
import subprocess
import sys
import time
from typing import Any
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.encounter import EncounterSamplerV1, export_encounters_jsonl
from xvs.tracking.evaluate import (
    _worker_wrapper,
    aggregate_dataset_metrics,
    get_ownship_states,
    load_or_compute_truth_cache,
)

G1_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json"
G0_SWEEP_JSON_PATH = REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "lambda_sweep_v2_results.json"
G0_PER_RUN_JSONL_PATH = REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "per_run_records.jsonl"
EXPECTED_SAMPLER_SHA256 = "d1ae4add34cf0fdc9a45e0663fc347b1e80fd26ae1b3f433b0a3c8e5d8c91a9b"

LAMBDAS_V2 = [0.0005, 0.005, 0.05, 0.5, 1.0, 2.0, 5.0]
FULL_SEEDS = [101, 102, 103, 104, 105]
FALLBACK_SEEDS = [101, 102]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261020
PILOT_WALL_LIMIT_S = 1.5 * 3600.0

TRACKED_CODE_FILES = [
    "scripts/xvs/tracking/run_g1_lambda_sweep_v2.py",
    "src/xvs/tracking/ttc.py",
    "src/xvs/tracking/ground_rejection.py",
    "src/xvs/tracking/measurements.py",
    "src/xvs/tracking/tracker.py",
    "src/xvs/tracking/risk.py",
    "src/xvs/tracking/evaluate.py",
    "configs/xvs/tracking/g1.json",
]


def get_safe_workers() -> int:
    """At most (logical cores - 4) worker processes."""
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


def garwood_poisson_rate_ci(k: int, exposure_hours: float, confidence: float = 0.95) -> dict[str, float]:
    """Compute exact Garwood Poisson confidence interval and one-sided 95% upper bound for rate k / T."""
    if exposure_hours <= 0.0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": float("inf"), "upper_95": float("inf"), "k": int(k), "exposure_hours": 0.0}
    alpha = 1.0 - confidence
    rate = k / exposure_hours
    low = 0.0 if k <= 0 else float(stats.chi2.ppf(alpha / 2.0, 2 * k) / (2.0 * exposure_hours))
    high = float(stats.chi2.ppf(1.0 - alpha / 2.0, 2 * (k + 1)) / (2.0 * exposure_hours))
    upper_95 = float(stats.chi2.ppf(confidence, 2 * (k + 1)) / (2.0 * exposure_hours))
    return {
        "rate": float(rate),
        "ci_low": float(low),
        "ci_high": float(high),
        "upper_95": float(upper_95),
        "k": int(k),
        "exposure_hours": float(exposure_hours),
    }


def encounter_bootstrap_ci(
    values_per_encounter: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute 2,000-resample seeded encounter-level bootstrap 95% CI and 95% upper bound."""
    n = len(values_per_encounter)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "upper_95": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(values_per_encounter[idx], axis=1)
    return {
        "mean": float(np.mean(values_per_encounter)),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "upper_95": float(np.percentile(boot_means, 95.0)),
    }


def summarize_wc1_miss_decomposition(threat_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize G1 WC1 miss decomposition at t_CPA - 12.5 s across threat runs with lead < 12.5 s or no warning."""
    n_threat_runs = len(threat_runs)
    late_or_missed = [
        r for r in threat_runs
        if r.get("lead_wc1_warn_s") is None or float(r["lead_wc1_warn_s"]) < 12.5
    ]
    n_late = len(late_or_missed)
    n_no_warning = sum(
        1 for r in threat_runs
        if r.get("lead_wc1_warn_s") is None or float(r["lead_wc1_warn_s"]) <= 0.0
    )

    marginal_counts = {
        "no_confirmed_track": 0,
        "ttc_unavailable": 0,
        "tau_mod_hat_gt_T": 0,
        "hmd_hat_gt_d_thr": 0,
    }
    joint_counts: dict[str, int] = {}
    overall_limit_counts = {
        "rule_limited": 0,
        "estimation_limited": 0,
        "no_confirmed_track": 0,
        "no_confirmed_track_rule_limited": 0,
        "passed_at_t_check": 0,
    }

    for r in late_or_missed:
        mc = r.get("miss_check_at_tcpa_minus_12_5", {})
        if mc.get("no_confirmed_track"):
            marginal_counts["no_confirmed_track"] += 1
        if mc.get("wc1_ttc_unavailable"):
            marginal_counts["ttc_unavailable"] += 1
        if mc.get("wc1_tau_mod_hat_gt_T"):
            marginal_counts["tau_mod_hat_gt_T"] += 1
        if mc.get("wc1_hmd_hat_gt_d_thr"):
            marginal_counts["hmd_hat_gt_d_thr"] += 1

        jkey = str(mc.get("wc1_joint_condition", "unknown"))
        joint_counts[jkey] = joint_counts.get(jkey, 0) + 1  # default-ok: counter

        ovl = str(mc.get("wc1_overall_limit_type", "passed_at_t_check"))
        if ovl in overall_limit_counts:
            overall_limit_counts[ovl] += 1

    denom = max(1, n_late)
    joint_table = {
        k: {"count": int(v), "fraction_of_late_or_missed": float(v / denom)}
        for k, v in sorted(joint_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    return {
        "rule": "WC1",
        "threat_runs_total": int(n_threat_runs),
        "late_or_missed_runs_lt_12_5s": int(n_late),
        "no_warning_runs": int(n_no_warning),
        "marginal_counts": marginal_counts,
        "marginal_fractions": {k: float(v / denom) for k, v in marginal_counts.items()},
        "joint_table": joint_table,
        "overall_limit_counts": overall_limit_counts,
        "overall_limit_fractions": {k: float(v / denom) for k, v in overall_limit_counts.items()},
    }


def summarize_a1_miss_decomposition(threat_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize legacy A1 miss decomposition at t_CPA - 12.5 s for paired diagnostic comparison."""
    n_threat_runs = len(threat_runs)
    late_or_missed = [
        r for r in threat_runs
        if r.get("lead_a1_warn_s") is None or float(r["lead_a1_warn_s"]) < 12.5
    ]
    n_late = len(late_or_missed)
    n_no_warning = sum(
        1 for r in threat_runs
        if r.get("lead_a1_warn_s") is None or float(r["lead_a1_warn_s"]) <= 0.0
    )

    marginal_counts = {
        "no_confirmed_track": 0,
        "tau_s_unavailable": 0,
        "tau_s_ge_T": 0,
        "omega_ge_thr": 0,
    }
    joint_counts: dict[str, int] = {}
    overall_limit_counts = {
        "rule_limited": 0,
        "estimation_limited": 0,
        "no_confirmed_track": 0,
        "no_confirmed_track_rule_limited": 0,
        "passed_at_t_check": 0,
    }

    for r in late_or_missed:
        mc = r.get("miss_check_at_tcpa_minus_12_5", {})
        if mc.get("no_confirmed_track"):
            marginal_counts["no_confirmed_track"] += 1
        if mc.get("tau_s_unavailable"):
            marginal_counts["tau_s_unavailable"] += 1
        if mc.get("tau_s_ge_T"):
            marginal_counts["tau_s_ge_T"] += 1
        if mc.get("omega_ge_thr"):
            marginal_counts["omega_ge_thr"] += 1

        jkey = str(mc.get("joint_condition", "unknown"))
        joint_counts[jkey] = joint_counts.get(jkey, 0) + 1  # default-ok: counter

        ovl = str(mc.get("overall_limit_type", "passed_at_t_check"))
        if ovl in overall_limit_counts:
            overall_limit_counts[ovl] += 1

    denom = max(1, n_late)
    joint_table = {
        k: {"count": int(v), "fraction_of_late_or_missed": float(v / denom)}
        for k, v in sorted(joint_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    return {
        "rule": "A1",
        "threat_runs_total": int(n_threat_runs),
        "late_or_missed_runs_lt_12_5s": int(n_late),
        "no_warning_runs": int(n_no_warning),
        "marginal_counts": marginal_counts,
        "marginal_fractions": {k: float(v / denom) for k, v in marginal_counts.items()},
        "joint_table": joint_table,
        "overall_limit_counts": overall_limit_counts,
        "overall_limit_fractions": {k: float(v / denom) for k, v in overall_limit_counts.items()},
    }


def load_g0_per_run_index(
    g0_jsonl_path: Path = G0_PER_RUN_JSONL_PATH,
    g0_sweep_json_path: Path = G0_SWEEP_JSON_PATH,
) -> tuple[dict[tuple[float, int, str], dict[str, Any]], str | None]:
    """Load G0 per-run records indexed by (lambda, seed, encounter_id) and verify SHA-256."""
    if not g0_jsonl_path.is_file():
        return {}, None
    raw_bytes = g0_jsonl_path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if g0_sweep_json_path.is_file():
        g0_sweep = json.loads(g0_sweep_json_path.read_text(encoding="utf-8"))
        expected_sha = g0_sweep.get("per_run_records_sha256")
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"G0 per_run_records.jsonl SHA-256 mismatch: {actual_sha} != {expected_sha}"
            )
    idx: dict[tuple[float, int, str], dict[str, Any]] = {}
    for line in raw_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (float(row["lambda"]), int(row["seed"]), str(row["encounter"]))
        idx[key] = row
    return idx, actual_sha


def _init_pool_worker() -> None:
    get_ownship_states()


def evaluate_d29_sequence(
    records: list[dict[str, Any]],
    lams: list[float],
    max_missed_inc: float = 0.05,
    max_fa_inc: float = 0.5,
    max_ft_ub: float = 1.0,
) -> tuple[list[dict[str, Any]], float | str, str]:
    """Evaluate D29 fixed-sequence pass rule (stop at first failure) across records."""
    analysis_table: list[dict[str, Any]] = []
    sequence_stopped = False
    last_passing_lambda: float | None = None

    for r in records:
        lam = float(r["lambda"])
        ub_miss_inc = float(r["paired_diff_missed_ci_95"][1])
        ub_fa_inc = float(r["paired_diff_fa_per_hour_ci_95"][1])
        ub_ft_garwood = float(r["false_tracks_garwood_upper_95"])
        ub_ft_garwood_2sided = float(r["false_tracks_garwood_ci_95"][1])

        pass_miss = bool(ub_miss_inc <= max_missed_inc + 1e-12)
        pass_fa = bool(ub_fa_inc <= max_fa_inc + 1e-12)
        pass_ft = bool(ub_ft_garwood <= max_ft_ub + 1e-12)
        point_passes = bool(pass_miss and pass_fa and pass_ft)

        if lam == lams[0]:
            point_passes = bool(pass_ft)

        if not sequence_stopped and point_passes:
            seq_pass = True
            last_passing_lambda = lam
        else:
            seq_pass = False
            sequence_stopped = True

        analysis_table.append({
            "lambda": lam,
            "n_seeds": int(r["n_seeds"]),
            "ub_missed_increase": ub_miss_inc,
            "ub_missed_increase_one_sided_95": float(r["paired_diff_missed_upper_95"]),
            "pass_missed_increase_le_0_05": pass_miss,
            "ub_fa_per_hour_increase": ub_fa_inc,
            "ub_fa_per_hour_increase_one_sided_95": float(r["paired_diff_fa_per_hour_upper_95"]),
            "pass_fa_increase_le_0_5": pass_fa,
            "garwood_ub_false_tracks_per_hour": ub_ft_garwood,
            "garwood_ci_high_false_tracks_per_hour": ub_ft_garwood_2sided,
            "pass_false_tracks_le_1_0": pass_ft,
            "point_passes_all_criteria": point_passes,
            "fixed_sequence_pass": seq_pass,
            "all_warning_metrics_ok": seq_pass,
            "deg_missed_ok": pass_miss,
            "deg_fa_ok": bool(pass_fa and pass_ft),
            "rel_diff_12_5": float(r["paired_diff_ge_12_5s"]),
            "rel_diff_25_0": float(r["paired_diff_ge_25_0s"]),
            "rel_diff_missed": float(r["paired_diff_missed_rate"]),
        })

    if last_passing_lambda is None or (len(lams) > 1 and last_passing_lambda == lams[0] and not analysis_table[1]["fixed_sequence_pass"]):
        if last_passing_lambda == lams[0]:
            lambda_req_str = f"{lams[0]} (< {lams[1]})"
            lambda_req_val: float | str = float(lams[0])
        else:
            lambda_req_str = f"< {lams[0]}"
            lambda_req_val = f"< {lams[0]}"
    elif last_passing_lambda == lams[-1]:
        lambda_req_str = f">= {lams[-1]}"
        lambda_req_val = float(lams[-1])
    else:
        lambda_req_str = str(last_passing_lambda)
        lambda_req_val = float(last_passing_lambda)

    return analysis_table, lambda_req_val, lambda_req_str


def run_g1_lambda_sweep_v2(
    encounters: list[Any],
    base_config: dict[str, Any],
    truth_cache_map: dict[str, Any],
    max_workers: int,
    lambdas: list[float] | None = None,
    seeds_full: list[int] | None = None,
    seeds_fallback: list[int] | None = None,
    g0_index: dict[tuple[float, int, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute full D29 / B7 G1 lambda sweep with CRN, pilot projection, paired bootstrap vs lambda=0.0005 and vs G0, and W7 miss decomposition."""
    lams = lambdas if lambdas is not None else list(LAMBDAS_V2)
    s_full = seeds_full if seeds_full is not None else list(FULL_SEEDS)
    s_fb = seeds_fallback if seeds_fallback is not None else list(FALLBACK_SEEDS)

    n_enc = len(encounters)
    enc_dicts = [asdict(e) for e in encounters]
    pilot_n = min(50, n_enc)

    baseline_by_seed: dict[int, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    per_run_records: list[dict[str, Any]] = []
    pilot_projections: dict[str, Any] = {}

    pool = Pool(processes=max_workers, initializer=_init_pool_worker) if max_workers > 1 else None
    try:
        for lam in lams:
            cfg = deepcopy(base_config)
            cfg["fppi_lambda"] = float(lam)

            # 1. Pilot benchmark on pilot_n encounters to project 5-seed wall time
            pilot_tasks = [
                (enc_dicts[i], cfg, s_full[0] * 10000 + i, truth_cache_map.get(encounters[i].id))
                for i in range(pilot_n)
            ]
            t_p0 = time.perf_counter()
            if pool is not None:
                pilot_res = pool.map(_worker_wrapper, pilot_tasks)
            else:
                pilot_res = [_worker_wrapper(t) for t in pilot_tasks]
            t_p1 = time.perf_counter()
            pilot_sec = t_p1 - t_p0
            proj_per_seed_sec = pilot_sec * (n_enc / max(1, pilot_n))
            proj_5seed_sec = proj_per_seed_sec * len(s_full)
            exceeds_1_5h = bool(proj_5seed_sec > PILOT_WALL_LIMIT_S and lam != lams[0])
            active_seeds = s_fb if exceeds_1_5h else s_full

            pilot_projections[str(lam)] = {
                "lambda": float(lam),
                "pilot_encounters": int(pilot_n),
                "pilot_wall_s": float(pilot_sec),
                "projected_5seed_wall_s": float(proj_5seed_sec),
                "projected_5seed_wall_h": float(proj_5seed_sec / 3600.0),
                "exceeds_1_5h_limit": exceeds_1_5h,
                "chosen_seeds": list(active_seeds),
                "n_seeds": len(active_seeds),
            }
            print(
                f"\n[Pilot G1 lambda={lam}] {pilot_n} encs in {pilot_sec:.2f}s -> "
                f"projected {len(s_full)} seeds = {proj_5seed_sec/60.0:.1f} min "
                f"(chosen_seeds={active_seeds})",
                flush=True,
            )

            # 2. Run all active seeds for this lambda with Common Random Numbers (seed = s * 10000 + i)
            per_seed_enc_results: dict[int, list[dict[str, Any]]] = {}
            all_enc_results_flat: list[dict[str, Any]] = []
            t_lam0 = time.perf_counter()

            for s_idx, s in enumerate(active_seeds):
                t_s0 = time.perf_counter()
                if s_idx == 0 and s == s_full[0] and pilot_n > 0:
                    rem_tasks = [
                        (enc_dicts[i], cfg, s * 10000 + i, truth_cache_map.get(encounters[i].id))
                        for i in range(pilot_n, n_enc)
                    ]
                    if rem_tasks:
                        if pool is not None:
                            rem_res = pool.map(_worker_wrapper, rem_tasks)
                        else:
                            rem_res = [_worker_wrapper(t) for t in rem_tasks]
                        res_list = list(pilot_res) + list(rem_res)
                    else:
                        res_list = list(pilot_res)
                    t_s1 = time.perf_counter()
                    seed_wall_s = pilot_sec + (t_s1 - t_s0)
                else:
                    tasks = [
                        (enc_dicts[i], cfg, s * 10000 + i, truth_cache_map.get(encounters[i].id))
                        for i in range(n_enc)
                    ]
                    if pool is not None:
                        res_list = pool.map(_worker_wrapper, tasks)
                    else:
                        res_list = [_worker_wrapper(t) for t in tasks]
                    t_s1 = time.perf_counter()
                    seed_wall_s = t_s1 - t_s0
                per_seed_enc_results[s] = res_list
                all_enc_results_flat.extend(res_list)
                for r_item in res_list:
                    per_run_records.append({
                        "lambda": float(lam),
                        "seed": int(s),
                        "encounter": str(r_item["id"]),
                        "class_id": str(r_item["class_id"]),
                        "is_threat": bool(r_item["is_threat"]),
                        "lead_wc1_warn_s": r_item["lead_wc1_warn_s"],
                        "lead_wc1_corr_s": r_item["lead_wc1_corr_s"],
                        "lead_wc1_warn_ck_s": r_item["lead_wc1_warn_ck_s"],
                        "lead_wc1_corr_ck_s": r_item["lead_wc1_corr_ck_s"],
                        "has_alert_wc1_warn": bool(r_item["has_alert_wc1_warn"]),
                        "has_alert_wc1_corr": bool(r_item["has_alert_wc1_corr"]),
                        "has_fa_wc1_warn": bool(r_item["has_fa_wc1_warn"]),
                        "fa_wc1_warn_events": int(r_item["fa_wc1_warn_events"]),
                        "lead_a1_warn_s": r_item["lead_a1_warn_s"],
                        "lead_a1_corr_s": r_item["lead_a1_corr_s"],
                        "lead_a2_warn_s": r_item["lead_a2_warn_s"],
                        "lead_a2_corr_s": r_item["lead_a2_corr_s"],
                        "has_alert_a1_warn": bool(r_item["has_alert_a1_warn"]),
                        "has_alert_a1_corr": bool(r_item["has_alert_a1_corr"]),
                        "has_alert_a2_warn": bool(r_item["has_alert_a2_warn"]),
                        "has_alert_a2_corr": bool(r_item["has_alert_a2_corr"]),
                        "has_fa_a1_warn": bool(r_item["has_fa_a1_warn"]),
                        "fa_a1_warn_events": int(r_item["fa_a1_warn_events"]),
                        "false_tracks": int(r_item["false_tracks"]),
                        "false_tracks_ever_confirmed": int(r_item["false_tracks_ever_confirmed"]),
                        "false_tracks_end_of_run": int(r_item["false_tracks_end_of_run"]),
                        "ground_clutter_rejected_count": int(r_item["ground_clutter_rejected_count"]),
                        "fragmentations": int(r_item["fragmentations"]),
                        "acq_delay_s": r_item["acq_delay_s"],
                        "detectable_frames": int(r_item["detectable_frames"]),
                        "confirmed_on_detectable_frames": int(r_item["confirmed_on_detectable_frames"]),
                        "miss_check": r_item["miss_check_at_tcpa_minus_12_5"],
                    })
                print(
                    f"  [G1 lambda={lam}] Seed {s} ({s_idx+1}/{len(active_seeds)}) "
                    f"done in {seed_wall_s:.1f}s",
                    flush=True,
                )

            t_lam1 = time.perf_counter()
            wall_s = pilot_sec + (t_lam1 - t_lam0)
            fps = (n_enc * len(active_seeds) * 601) / max(1e-9, wall_s)

            if lam == lams[0]:
                baseline_by_seed = per_seed_enc_results

            agg = aggregate_dataset_metrics(all_enc_results_flat, cfg)
            n_seeds = len(active_seeds)
            total_flight_hours = (n_enc * n_seeds * 60.0) / 3600.0

            threat_indices = [i for i in range(n_enc) if per_seed_enc_results[active_seeds[0]][i]["is_threat"]]
            non_threat_indices = [i for i in range(n_enc) if not per_seed_enc_results[active_seeds[0]][i]["is_threat"]]
            n_threats_per_seed = len(threat_indices)
            n_non_threats_per_seed = len(non_threat_indices)
            non_threat_flight_hours = (n_non_threats_per_seed * n_seeds * 60.0) / 3600.0

            # Primary G1 rule is WC1 warning
            miss_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            miss_base_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead12_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead12_base_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead25_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead25_base_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            miss_g0_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead12_g0_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead25_g0_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            has_g0_pairing = False

            for ti_pos, enc_i in enumerate(threat_indices):
                m_cur = []
                m_bas = []
                l12_cur = []
                l12_bas = []
                l25_cur = []
                l25_bas = []
                m_g0 = []
                l12_g0 = []
                l25_g0 = []
                for s in active_seeds:
                    rc = per_seed_enc_results[s][enc_i]
                    rb = baseline_by_seed[s][enc_i]
                    lc = rc["lead_wc1_warn_s"]
                    lb = rb["lead_wc1_warn_s"]
                    m_cur.append(1.0 if (lc is None or lc <= 0.0) else 0.0)
                    m_bas.append(1.0 if (lb is None or lb <= 0.0) else 0.0)
                    l12_cur.append(1.0 if (lc is not None and lc >= 12.5) else 0.0)
                    l12_bas.append(1.0 if (lb is not None and lb >= 12.5) else 0.0)
                    l25_cur.append(1.0 if (lc is not None and lc >= 25.0) else 0.0)
                    l25_bas.append(1.0 if (lb is not None and lb >= 25.0) else 0.0)
                    if g0_index is not None:
                        g0_row = g0_index.get((float(lam), int(s), str(rc["id"])))
                        if g0_row is not None:
                            has_g0_pairing = True
                            lg0 = g0_row.get("lead_a1_warn_s")
                            m_g0.append(1.0 if (lg0 is None or lg0 <= 0.0) else 0.0)
                            l12_g0.append(1.0 if (lg0 is not None and lg0 >= 12.5) else 0.0)
                            l25_g0.append(1.0 if (lg0 is not None and lg0 >= 25.0) else 0.0)
                miss_vec[ti_pos] = float(np.mean(m_cur))
                miss_base_vec[ti_pos] = float(np.mean(m_bas))
                lead12_vec[ti_pos] = float(np.mean(l12_cur))
                lead12_base_vec[ti_pos] = float(np.mean(l12_bas))
                lead25_vec[ti_pos] = float(np.mean(l25_cur))
                lead25_base_vec[ti_pos] = float(np.mean(l25_bas))
                if m_g0:
                    miss_g0_vec[ti_pos] = float(np.mean(m_g0))
                    lead12_g0_vec[ti_pos] = float(np.mean(l12_g0))
                    lead25_g0_vec[ti_pos] = float(np.mean(l25_g0))

            fa_hr_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            fa_hr_base_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            fa_ev_hr_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            fa_ev_hr_base_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            fa_hr_g0_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            k_fa_total = 0
            k_fa_events_total = 0
            for nti_pos, enc_i in enumerate(non_threat_indices):
                f_cur = []
                f_bas = []
                fe_cur = []
                fe_bas = []
                f_g0 = []
                for s in active_seeds:
                    rc = per_seed_enc_results[s][enc_i]
                    rb = baseline_by_seed[s][enc_i]
                    has_fa_c = 1.0 if rc["has_alert_wc1_warn"] else 0.0
                    has_fa_b = 1.0 if rb["has_alert_wc1_warn"] else 0.0
                    ev_fa_c = float(rc["fa_wc1_warn_events"])
                    ev_fa_b = float(rb["fa_wc1_warn_events"])
                    f_cur.append(has_fa_c * 60.0)
                    f_bas.append(has_fa_b * 60.0)
                    fe_cur.append(ev_fa_c * 60.0)
                    fe_bas.append(ev_fa_b * 60.0)
                    if rc["has_alert_wc1_warn"]:
                        k_fa_total += 1
                    k_fa_events_total += int(rc["fa_wc1_warn_events"])
                    if g0_index is not None:
                        g0_row = g0_index.get((float(lam), int(s), str(rc["id"])))
                        if g0_row is not None:
                            f_g0.append((1.0 if g0_row.get("has_alert_a1_warn") else 0.0) * 60.0)
                fa_hr_vec[nti_pos] = float(np.mean(f_cur))
                fa_hr_base_vec[nti_pos] = float(np.mean(f_bas))
                fa_ev_hr_vec[nti_pos] = float(np.mean(fe_cur))
                fa_ev_hr_base_vec[nti_pos] = float(np.mean(fe_bas))
                if f_g0:
                    fa_hr_g0_vec[nti_pos] = float(np.mean(f_g0))

            ft_hr_vec = np.zeros(n_enc, dtype=np.float64)
            ft_hr_base_vec = np.zeros(n_enc, dtype=np.float64)
            ft_hr_g0_vec = np.zeros(n_enc, dtype=np.float64)
            k_ft_total = 0
            k_ft_end_total = 0
            k_gnd_rej_total = 0
            for enc_i in range(n_enc):
                ft_cur = []
                ft_bas = []
                ft_g0 = []
                for s in active_seeds:
                    rc = per_seed_enc_results[s][enc_i]
                    rb = baseline_by_seed[s][enc_i]
                    ft_cur.append(float(rc["false_tracks"]) * 60.0)
                    ft_bas.append(float(rb["false_tracks"]) * 60.0)
                    k_ft_total += int(rc["false_tracks"])
                    k_ft_end_total += int(rc["false_tracks_end_of_run"])
                    k_gnd_rej_total += int(rc["ground_clutter_rejected_count"])
                    if g0_index is not None:
                        g0_row = g0_index.get((float(lam), int(s), str(rc["id"])))
                        if g0_row is not None:
                            ft_g0.append(float(g0_row["false_tracks"]) * 60.0)
                ft_hr_vec[enc_i] = float(np.mean(ft_cur))
                ft_hr_base_vec[enc_i] = float(np.mean(ft_bas))
                if ft_g0:
                    ft_hr_g0_vec[enc_i] = float(np.mean(ft_g0))

            miss_boot = encounter_bootstrap_ci(miss_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 1)
            miss_diff_boot = encounter_bootstrap_ci(miss_vec - miss_base_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 2)
            lead12_diff_boot = encounter_bootstrap_ci(lead12_vec - lead12_base_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 3)
            lead25_diff_boot = encounter_bootstrap_ci(lead25_vec - lead25_base_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 4)

            fa_hr_boot = encounter_bootstrap_ci(fa_hr_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 5)
            fa_hr_diff_boot = encounter_bootstrap_ci(fa_hr_vec - fa_hr_base_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 6)
            fa_garwood = garwood_poisson_rate_ci(k_fa_total, non_threat_flight_hours)

            fa_ev_hr_boot = encounter_bootstrap_ci(fa_ev_hr_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 8)
            fa_ev_hr_diff_boot = encounter_bootstrap_ci(fa_ev_hr_vec - fa_ev_hr_base_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 9)
            fa_ev_garwood = garwood_poisson_rate_ci(k_fa_events_total, non_threat_flight_hours)

            ft_garwood = garwood_poisson_rate_ci(k_ft_total, total_flight_hours)
            ft_end_garwood = garwood_poisson_rate_ci(k_ft_end_total, total_flight_hours)
            ft_hr_diff_boot = encounter_bootstrap_ci(ft_hr_vec - ft_hr_base_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 7)

            paired_vs_g0: dict[str, Any] | None = None
            if has_g0_pairing:
                g1_vs_g0_miss = encounter_bootstrap_ci(miss_vec - miss_g0_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 11)
                g1_vs_g0_lead12 = encounter_bootstrap_ci(lead12_vec - lead12_g0_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 12)
                g1_vs_g0_lead25 = encounter_bootstrap_ci(lead25_vec - lead25_g0_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 13)
                g1_vs_g0_fa = encounter_bootstrap_ci(fa_hr_vec - fa_hr_g0_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 14)
                g1_vs_g0_ft = encounter_bootstrap_ci(ft_hr_vec - ft_hr_g0_vec, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED + 15)
                paired_vs_g0 = {
                    "diff_missed_rate_mean": g1_vs_g0_miss["mean"],
                    "diff_missed_rate_ci_95": [g1_vs_g0_miss["ci_low"], g1_vs_g0_miss["ci_high"]],
                    "diff_lead_ge_12_5s_mean": g1_vs_g0_lead12["mean"],
                    "diff_lead_ge_12_5s_ci_95": [g1_vs_g0_lead12["ci_low"], g1_vs_g0_lead12["ci_high"]],
                    "diff_lead_ge_25_0s_mean": g1_vs_g0_lead25["mean"],
                    "diff_lead_ge_25_0s_ci_95": [g1_vs_g0_lead25["ci_low"], g1_vs_g0_lead25["ci_high"]],
                    "diff_fa_per_hour_mean": g1_vs_g0_fa["mean"],
                    "diff_fa_per_hour_ci_95": [g1_vs_g0_fa["ci_low"], g1_vs_g0_fa["ci_high"]],
                    "diff_false_tracks_per_hour_mean": g1_vs_g0_ft["mean"],
                    "diff_false_tracks_per_hour_ci_95": [g1_vs_g0_ft["ci_low"], g1_vs_g0_ft["ci_high"]],
                }

            threat_runs_all = [r for r in all_enc_results_flat if r["is_threat"]]
            wc1_miss_decomp = summarize_wc1_miss_decomposition(threat_runs_all)
            a1_miss_decomp = summarize_a1_miss_decomposition(threat_runs_all)

            wc1_w = agg["alerting"]["WC1_warning"]
            a1_w = agg["alerting"]["A1_warning"]
            k_missed_total = int(round(wc1_w["frac_missed"] * wc1_w["threats_count"]))
            k_ge12_total = int(round(wc1_w["frac_ge_12_5s"] * wc1_w["threats_count"]))
            k_ge25_total = int(round(wc1_w["frac_ge_25_0s"] * wc1_w["threats_count"]))

            rec = {
                "lambda": float(lam),
                "primary_alert_rule": "WC1",
                "seeds": list(active_seeds),
                "n_seeds": int(n_seeds),
                "encounters_per_seed": int(n_enc),
                "total_runs": int(n_enc * n_seeds),
                "total_flight_hours": float(total_flight_hours),
                "non_threat_flight_hours": float(non_threat_flight_hours),
                "threats_count": int(wc1_w["threats_count"]),
                "non_threats_count": int(wc1_w["non_threats_count"]),
                # Primary metric 1: Missed-alert fraction on threat encounters (WC1)
                "missed_alert_count": int(k_missed_total),
                "missed_alert_rate": float(wc1_w["frac_missed"]),
                "missed_alert_cp_ci_95": list(wc1_w["ci_missed"]),
                "missed_alert_boot_ci_95": [miss_boot["ci_low"], miss_boot["ci_high"]],
                "paired_diff_missed_rate": float(miss_diff_boot["mean"]),
                "paired_diff_missed_ci_95": [miss_diff_boot["ci_low"], miss_diff_boot["ci_high"]],
                "paired_diff_missed_upper_95": float(miss_diff_boot["upper_95"]),
                # Primary metric 2: False alerts per flight hour on non-threats (WC1)
                "false_alerts_count": int(k_fa_total),
                "false_alerts_per_hour": float(wc1_w["false_alerts_per_hour"]),
                "false_alerts_garwood_ci_95": [fa_garwood["ci_low"], fa_garwood["ci_high"]],
                "false_alerts_boot_ci_95": [fa_hr_boot["ci_low"], fa_hr_boot["ci_high"]],
                "runs_with_false_alert_count": int(k_fa_total),
                "runs_with_false_alert_per_h": float(wc1_w["runs_with_false_alert_per_h"]),
                "runs_with_false_alert_garwood_ci_95": [fa_garwood["ci_low"], fa_garwood["ci_high"]],
                "runs_with_false_alert_boot_ci_95": [fa_hr_boot["ci_low"], fa_hr_boot["ci_high"]],
                "false_alert_events_count": int(k_fa_events_total),
                "false_alert_events_per_h": float(wc1_w["false_alert_events_per_h"]),
                "false_alert_events_garwood_ci_95": [fa_ev_garwood["ci_low"], fa_ev_garwood["ci_high"]],
                "false_alert_events_boot_ci_95": [fa_ev_hr_boot["ci_low"], fa_ev_hr_boot["ci_high"]],
                "paired_diff_fa_per_hour": float(fa_hr_diff_boot["mean"]),
                "paired_diff_fa_per_hour_ci_95": [fa_hr_diff_boot["ci_low"], fa_hr_diff_boot["ci_high"]],
                "paired_diff_fa_per_hour_upper_95": float(fa_hr_diff_boot["upper_95"]),
                "paired_diff_fa_events_per_h": float(fa_ev_hr_diff_boot["mean"]),
                "paired_diff_fa_events_per_h_ci_95": [fa_ev_hr_diff_boot["ci_low"], fa_ev_hr_diff_boot["ci_high"]],
                "paired_diff_fa_events_per_h_upper_95": float(fa_ev_hr_diff_boot["upper_95"]),
                # Primary metric 3: False confirmed tracks per flight hour (D32: ever-confirmed primary)
                "false_tracks_count": int(k_ft_total),
                "false_tracks_per_hour": float(ft_garwood["rate"]),
                "false_tracks_garwood_ci_95": [ft_garwood["ci_low"], ft_garwood["ci_high"]],
                "false_tracks_garwood_upper_95": float(ft_garwood["upper_95"]),
                "false_tracks_end_of_run": int(k_ft_end_total),
                "false_tracks_end_of_run_per_hour": float(ft_end_garwood["rate"]),
                "false_tracks_end_of_run_garwood_ci_95": [ft_end_garwood["ci_low"], ft_end_garwood["ci_high"]],
                "false_tracks_end_of_run_garwood_upper_95": float(ft_end_garwood["upper_95"]),
                "ground_clutter_rejected_count": int(k_gnd_rej_total),
                "ground_clutter_rejected_per_hour": float(k_gnd_rej_total / max(1e-9, total_flight_hours)),
                "paired_diff_ft_per_hour": float(ft_hr_diff_boot["mean"]),
                "paired_diff_ft_per_hour_ci_95": [ft_hr_diff_boot["ci_low"], ft_hr_diff_boot["ci_high"]],
                # Secondary metrics
                "warning_ge_12_5s_count": int(k_ge12_total),
                "warning_ge_12_5s_rate": float(wc1_w["frac_ge_12_5s"]),
                "warning_ge_12_5s_ci_95": list(wc1_w["ci_ge_12_5s"]),
                "paired_diff_ge_12_5s": float(lead12_diff_boot["mean"]),
                "paired_diff_ge_12_5s_ci_95": [lead12_diff_boot["ci_low"], lead12_diff_boot["ci_high"]],
                "warning_ge_25_0s_count": int(k_ge25_total),
                "warning_ge_25_0s_rate": float(wc1_w["frac_ge_25_0s"]),
                "warning_ge_25_0s_ci_95": list(wc1_w["ci_ge_25_0s"]),
                "paired_diff_ge_25_0s": float(lead25_diff_boot["mean"]),
                "paired_diff_ge_25_0s_ci_95": [lead25_diff_boot["ci_low"], lead25_diff_boot["ci_high"]],
                # Legacy A1 diagnostic summary
                "legacy_a1_missed_alert_rate": float(a1_w["frac_missed"]),
                "legacy_a1_warning_ge_12_5s_rate": float(a1_w["frac_ge_12_5s"]),
                "legacy_a1_false_alerts_per_hour": float(a1_w["false_alerts_per_hour"]),
                "true_track_coverage": float(agg["tracking"]["true_track_coverage"]),
                "detectable_frames_total": int(agg["tracking"]["detectable_frames_total"]),
                "confirmed_on_detectable_frames_total": int(agg["tracking"]["confirmed_on_detectable_frames_total"]),
                "fragmentations_total": int(agg["tracking"]["fragmentations_total"]),
                "fragmentations_per_seed": float(agg["tracking"]["fragmentations_total"] / n_seeds),
                "acq_delay_median_s": float(agg["tracking"]["acquisition_delay"]["median_s"]),
                "acq_delay_p95_s": float(agg["tracking"]["acquisition_delay"]["p95_s"]),
                "sigma_omega_total_radps": float(agg["los_rate"]["sigma_omega_total_radps"]),
                "ttc_mean_error_s": float(agg["ttc"]["mean_error_s"]),
                "ttc_median_error_s": float(agg["ttc"]["median_error_s"]),
                "cpu_runtime_s": float(wall_s),
                "frames_per_second": float(fps),
                "miss_decomposition": wc1_miss_decomp,
                "legacy_a1_miss_decomposition": a1_miss_decomp,
                "paired_vs_g0": paired_vs_g0,
                "raw_results": agg,
            }
            records.append(rec)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    analysis_table, lambda_req_val, lambda_req_str = evaluate_d29_sequence(records, lams)

    return {
        "schema": "xvs.g1_lambda_sweep_v2/1",
        "description": "WP-05 G1 / Batch B7 lambda sweep v2 under D29 & D32 (Passive Well-Clear WC1, sliding-window second-moment TTC, 6-DoF WGS-84 ground-clutter rejection, CRN paired against G0)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encounters_total": int(n_enc),
        "lambdas_swept": [float(x) for x in lams],
        "bootstrap_resamples": int(BOOTSTRAP_RESAMPLES),
        "bootstrap_seed": int(BOOTSTRAP_SEED),
        "pilot_projections": pilot_projections,
        "records": records,
        "analysis_table": analysis_table,
        "lambda_req": lambda_req_val,
        "lambda_req_statement": lambda_req_str,
        "wp04_g1_false_alarm_requirement_lambda": lambda_req_val,
        "_per_run_records": per_run_records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WP-05 G1 / Batch B7 Lambda Sweep v2 Runner (D29/D32)")
    parser.add_argument("--workers", type=int, default=get_safe_workers(), help="Worker process count")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--quick-encounters", type=int, default=None, help="Optional encounter cap for fast testing")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None, help="Optional custom lambda list")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional custom seed list")
    args = parser.parse_args(argv)

    safe_limit = get_safe_workers()
    if args.workers > safe_limit:
        print(f"WARNING: Clamping workers from {args.workers} to safe limit {safe_limit}")
        args.workers = safe_limit

    data_root = args.data_root.resolve() if args.data_root is not None else (REPO_ROOT / "output")
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (data_root / "xvs" / "wp05" / "g1_lambda_sweep_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = (data_root / "xvs" / "wp05" / "g1") if args.out_dir is None else None
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)

    git_status_start = get_git_status_porcelain()
    if args.quick_encounters is None and git_status_start:
        raise RuntimeError(f"Refusing to run full G1 lambda sweep on dirty XVS git tree:\n{git_status_start}")

    start_commit = get_git_commit()
    start_hashes = compute_code_hashes()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print("=" * 72)
    print("WP-05 G1 / Batch B7 — Lambda Sweep v2 Runner (D29 / D32)")
    print(f"Data root:    {data_root}")
    print(f"Output dir:   {out_dir}")
    print(f"Workers:      {args.workers} (safe limit: {safe_limit})")
    print(f"Git commit:   {start_commit}")
    print("=" * 72)

    with open(G1_CONFIG_PATH, "r", encoding="utf-8") as f:
        g1_cfg = json.load(f)

    sampler = EncounterSamplerV1()
    encounters = sampler.generate(t1_cruise_state)
    cache_path = REPO_ROOT / "data" / "cache" / "wp05_encounters_sampler_v1.jsonl"
    sha = export_encounters_jsonl(encounters, cache_path)
    if sha != EXPECTED_SAMPLER_SHA256:
        raise ValueError(f"Sampler SHA-256 mismatch: {sha} != {EXPECTED_SAMPLER_SHA256}")

    if args.quick_encounters is not None:
        encounters = encounters[: args.quick_encounters]
        print(f"Quick mode: using {len(encounters)} encounters.")

    g0_index, g0_per_run_sha = load_g0_per_run_index()

    # Write pre-launch run_manifest.json per G-B14 §1 / B7 §6.1
    manifest_file = out_dir / "run_manifest.json"
    pre_manifest = {
        "batch": "ARGON_B7",
        "wp": "WP-05_G1_LAMBDA_SWEEP_V2",
        "status": "running",
        "argv": list(argv) if argv is not None else list(sys.argv),
        "start_time_utc": start_time_iso,
        "git_commit_start": start_commit,
        "git_status_start": git_status_start,
        "sampler_v1_sha256": sha,
        "g0_per_run_records_sha256": g0_per_run_sha,
        "configs_read": {
            "configs/xvs/tracking/g1.json": hashlib.sha256(G1_CONFIG_PATH.read_bytes()).hexdigest(),
        },
        "code_hashes_start": start_hashes,
        "platform": platform.platform(),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(pre_manifest, f, indent=2)

    get_ownship_states()
    truth_cache_map, truth_cache_info = load_or_compute_truth_cache(encounters, max_workers=args.workers)

    t0 = time.perf_counter()
    sweep_res = run_g1_lambda_sweep_v2(
        encounters=encounters,
        base_config=g1_cfg,
        truth_cache_map=truth_cache_map,
        max_workers=args.workers,
        lambdas=args.lambdas,
        seeds_full=args.seeds,
        g0_index=g0_index,
    )
    t1 = time.perf_counter()
    total_wall_s = t1 - t0

    per_run_rows = sweep_res.pop("_per_run_records", [])
    per_run_file = out_dir / "per_run_records.jsonl"
    with open(per_run_file, "w", encoding="utf-8") as f:
        for row in per_run_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    per_run_sha256 = hashlib.sha256(per_run_file.read_bytes()).hexdigest()

    end_commit = get_git_commit()
    end_hashes = compute_code_hashes()
    git_status_end = get_git_status_porcelain()
    end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    hashes_match = (start_hashes == end_hashes) and (start_commit == end_commit)
    if args.quick_encounters is None and not hashes_match:
        raise RuntimeError(
            f"Long-job rule violation: code hashes or git commit changed during run! "
            f"start={start_hashes}, end={end_hashes}"
        )

    sweep_res["total_wall_s"] = float(total_wall_s)
    sweep_res["workers_used"] = int(args.workers)
    sweep_res["truth_cache"] = truth_cache_info
    sweep_res["per_run_records_file"] = "output/xvs/wp05/g1_lambda_sweep_v2/per_run_records.jsonl"
    sweep_res["per_run_records_sha256"] = per_run_sha256
    sweep_res["g0_per_run_records_sha256"] = g0_per_run_sha
    sweep_res["provenance"] = {
        "batch": "ARGON_B7",
        "wp": "WP-05_G1_LAMBDA_SWEEP_V2",
        "status": "completed",
        "argv": list(argv) if argv is not None else list(sys.argv),
        "start_time_utc": start_time_iso,
        "end_time_utc": end_time_iso,
        "git_commit_start": start_commit,
        "git_commit_end": end_commit,
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "sampler_v1_sha256": sha,
        "per_run_records_file": "output/xvs/wp05/g1_lambda_sweep_v2/per_run_records.jsonl",
        "per_run_records_sha256": per_run_sha256,
        "g0_per_run_records_sha256": g0_per_run_sha,
        "configs_read": {
            "configs/xvs/tracking/g1.json": hashlib.sha256(G1_CONFIG_PATH.read_bytes()).hexdigest(),
        },
        "code_hashes_start": start_hashes,
        "code_hashes_end": end_hashes,
        "code_hashes_match": hashes_match,
        "code_hashes_unchanged_during_run": hashes_match,
        "platform": platform.platform(),
    }

    out_file = out_dir / "lambda_sweep_v2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sweep_res, f, indent=2)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(sweep_res["provenance"], f, indent=2)

    if mirror_dir is not None and mirror_dir != out_dir:
        shutil.copy2(out_file, mirror_dir / "lambda_sweep_v2_results.json")
        shutil.copy2(per_run_file, mirror_dir / "per_run_records.jsonl")

    print("\n" + "=" * 72)
    print(f"G1 Lambda Sweep v2 completed in {total_wall_s:.1f} s ({total_wall_s/60.0:.2f} min)")
    print(f"Designated G1 frame-independent lambda_req (D29): {sweep_res['lambda_req_statement']}")
    print(f"Saved results to: {out_file}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
