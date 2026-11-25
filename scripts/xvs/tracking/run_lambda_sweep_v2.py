"""WP-05 G0 / G-B14 W7 — Lambda Sweep v2 Runner under Decision D29.

Implements:
- Common random numbers (CRN) across lambda in {0.0005, 0.005, 0.05, 0.5, 1.0, 2.0, 5.0} FP/cam-frame
  over all 900 stratified encounters x 5 seeds (with automatic 2-seed fallback if a lambda's pilot
  projects > 1.5 h wall time).
- Primary metrics per lambda with 95% CI and paired difference vs lambda=0.0005 via 2,000-resample
  seeded encounter-level bootstrap:
  1. Missed-alert fraction on threat encounters (A1 warning).
  2. False alerts per flight hour on non-threat encounters.
  3. False confirmed tracks per flight hour (with exact Garwood Poisson 95% CI and 95% upper bound).
- Secondary metrics: lead >= 12.5 s and >= 25.0 s fractions, true-track coverage (P_d > 0.5),
  fragmentations, acquisition delay median and p95.
- Miss decomposition at t_CPA - 12.5 s across all threat runs with lead < 12.5 s or no warning:
  joint condition table of (no_confirmed_track, tau_s_unavailable, tau_s_ge_T, omega_ge_thr)
  and rule-limited vs estimation-limited classification against ground truth.
- Fixed-sequence pass rule (stop at first failure) determining lambda_req.
- Long-job rule: records commit and SHA-256 hashes of script, tracking modules, and config at start
  and end, asserting zero mid-run code changes.
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
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, data_root_source
from xvs.metrics.aot import clopper_pearson_interval
from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.encounter import EncounterSamplerV1, export_encounters_jsonl
from xvs.tracking.evaluate import (
    _worker_wrapper,
    aggregate_dataset_metrics,
    get_ownship_states,
    load_or_compute_truth_cache,
)

MEASURED_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0_measured_d0.json"
EXPECTED_SAMPLER_SHA256 = "d1ae4add34cf0fdc9a45e0663fc347b1e80fd26ae1b3f433b0a3c8e5d8c91a9b"

LAMBDAS_V2 = [0.0005, 0.005, 0.05, 0.5, 1.0, 2.0, 5.0]
FULL_SEEDS = [101, 102, 103, 104, 105]
FALLBACK_SEEDS = [101, 102]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261020
PILOT_WALL_LIMIT_S = 1.5 * 3600.0  # 1.5 hours per W7

TRACKED_CODE_FILES = [
    "scripts/xvs/tracking/run_lambda_sweep_v2.py",
    "src/xvs/tracking/measurements.py",
    "src/xvs/tracking/tracker.py",
    "src/xvs/tracking/risk.py",
    "src/xvs/tracking/evaluate.py",
    "configs/xvs/tracking/g0_measured_d0.json",
]


def get_safe_workers() -> int:
    """At most (logical cores - 4) worker processes."""
    cores = os.cpu_count() or 4
    return max(1, cores - 4)


def compute_code_hashes() -> dict[str, str]:
    hashes = {}
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
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
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


def summarize_miss_decomposition(threat_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize W7 miss decomposition at t_CPA - 12.5 s across threat runs with lead < 12.5 s or no warning."""
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
    tau_limit_counts = {"rule_limited": 0, "estimation_limited": 0, "passed_or_no_track": 0}
    omega_limit_counts = {"rule_limited": 0, "estimation_limited": 0, "passed_or_no_track": 0}
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

        tl = str(mc.get("tau_limit_type", "passed_or_no_track"))
        if tl in tau_limit_counts:
            tau_limit_counts[tl] += 1
        ol = str(mc.get("omega_limit_type", "passed_or_no_track"))
        if ol in omega_limit_counts:
            omega_limit_counts[ol] += 1
        ovl = str(mc.get("overall_limit_type", "passed_at_t_check"))
        if ovl in overall_limit_counts:
            overall_limit_counts[ovl] += 1

    denom = max(1, n_late)
    joint_table = {
        k: {"count": int(v), "fraction_of_late_or_missed": float(v / denom)}
        for k, v in sorted(joint_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    return {
        "threat_runs_total": int(n_threat_runs),
        "late_or_missed_runs_lt_12_5s": int(n_late),
        "no_warning_runs": int(n_no_warning),
        "marginal_counts": marginal_counts,
        "marginal_fractions": {k: float(v / denom) for k, v in marginal_counts.items()},
        "joint_table": joint_table,
        "tau_limit_counts": tau_limit_counts,
        "omega_limit_counts": omega_limit_counts,
        "overall_limit_counts": overall_limit_counts,
        "overall_limit_fractions": {k: float(v / denom) for k, v in overall_limit_counts.items()},
    }


def _init_pool_worker() -> None:
    """Warmup ownship states inside each worker process once."""
    get_ownship_states()


def run_lambda_sweep_v2(
    encounters: list[Any],
    base_config: dict[str, Any],
    truth_cache_map: dict[str, Any],
    max_workers: int,
    lambdas: list[float] | None = None,
    seeds_full: list[int] | None = None,
    seeds_fallback: list[int] | None = None,
) -> dict[str, Any]:
    """Execute full D29 / W7 lambda sweep with CRN, pilot projection, paired bootstrap, and miss decomposition."""
    lams = lambdas if lambdas is not None else list(LAMBDAS_V2)
    s_full = seeds_full if seeds_full is not None else list(FULL_SEEDS)
    s_fb = seeds_fallback if seeds_fallback is not None else list(FALLBACK_SEEDS)

    n_enc = len(encounters)
    enc_dicts = [asdict(e) for e in encounters]
    pilot_n = min(50, n_enc)

    # Store per-(seed, encounter_idx) results for baseline lambda=0.0005 for paired bootstrap
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
                f"\n[Pilot lambda={lam}] {pilot_n} encs in {pilot_sec:.2f}s -> "
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
                        "fragmentations": int(r_item["fragmentations"]),
                        "acq_delay_s": r_item["acq_delay_s"],
                        "detectable_frames": int(r_item["detectable_frames"]),
                        "confirmed_on_detectable_frames": int(r_item["confirmed_on_detectable_frames"]),
                        "miss_check": r_item["miss_check_at_tcpa_minus_12_5"],
                    })
                print(
                    f"  [lambda={lam}] Seed {s} ({s_idx+1}/{len(active_seeds)}) "
                    f"done in {seed_wall_s:.1f}s",
                    flush=True,
                )

            t_lam1 = time.perf_counter()
            wall_s = pilot_sec + (t_lam1 - t_lam0)
            fps = (n_enc * len(active_seeds) * 601) / max(1e-9, wall_s)

            if lam == lams[0]:
                baseline_by_seed = per_seed_enc_results

            # Aggregate standard dataset metrics across all (encounter, seed) runs
            agg = aggregate_dataset_metrics(all_enc_results_flat, cfg)
            n_seeds = len(active_seeds)
            total_flight_hours = (n_enc * n_seeds * 60.0) / 3600.0

            # Organize per-encounter arrays across active_seeds for encounter-level bootstrap & paired diff
            threat_indices = [i for i in range(n_enc) if per_seed_enc_results[active_seeds[0]][i]["is_threat"]]
            non_threat_indices = [i for i in range(n_enc) if not per_seed_enc_results[active_seeds[0]][i]["is_threat"]]
            n_threats_per_seed = len(threat_indices)
            n_non_threats_per_seed = len(non_threat_indices)
            non_threat_flight_hours = (n_non_threats_per_seed * n_seeds * 60.0) / 3600.0

            # Per-encounter average missed alert (over active_seeds)
            miss_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            miss_base_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead12_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead12_base_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead25_vec = np.zeros(n_threats_per_seed, dtype=np.float64)
            lead25_base_vec = np.zeros(n_threats_per_seed, dtype=np.float64)

            for ti_pos, enc_i in enumerate(threat_indices):
                m_cur = []
                m_bas = []
                l12_cur = []
                l12_bas = []
                l25_cur = []
                l25_bas = []
                for s in active_seeds:
                    rc = per_seed_enc_results[s][enc_i]
                    rb = baseline_by_seed[s][enc_i]
                    lc = rc["lead_a1_warn_s"]
                    lb = rb["lead_a1_warn_s"]
                    m_cur.append(1.0 if (lc is None or lc <= 0.0) else 0.0)
                    m_bas.append(1.0 if (lb is None or lb <= 0.0) else 0.0)
                    l12_cur.append(1.0 if (lc is not None and lc >= 12.5) else 0.0)
                    l12_bas.append(1.0 if (lb is not None and lb >= 12.5) else 0.0)
                    l25_cur.append(1.0 if (lc is not None and lc >= 25.0) else 0.0)
                    l25_bas.append(1.0 if (lb is not None and lb >= 25.0) else 0.0)
                miss_vec[ti_pos] = float(np.mean(m_cur))
                miss_base_vec[ti_pos] = float(np.mean(m_bas))
                lead12_vec[ti_pos] = float(np.mean(l12_cur))
                lead12_base_vec[ti_pos] = float(np.mean(l12_bas))
                lead25_vec[ti_pos] = float(np.mean(l25_cur))
                lead25_base_vec[ti_pos] = float(np.mean(l25_bas))

            # Per-encounter false alert rate per flight hour on non-threats (1 encounter = 1/60 hr -> * 60.0)
            # Split into runs_with_false_alert_per_h and false_alert_events_per_h (F16)
            fa_hr_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            fa_hr_base_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            fa_ev_hr_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            fa_ev_hr_base_vec = np.zeros(n_non_threats_per_seed, dtype=np.float64)
            k_fa_total = 0
            k_fa_events_total = 0
            for nti_pos, enc_i in enumerate(non_threat_indices):
                f_cur = []
                f_bas = []
                fe_cur = []
                fe_bas = []
                for s in active_seeds:
                    rc = per_seed_enc_results[s][enc_i]
                    rb = baseline_by_seed[s][enc_i]
                    has_fa_c = 1.0 if rc["has_alert_a1_warn"] else 0.0
                    has_fa_b = 1.0 if rb["has_alert_a1_warn"] else 0.0
                    ev_fa_c = float(rc["fa_a1_warn_events"])
                    ev_fa_b = float(rb["fa_a1_warn_events"])
                    f_cur.append(has_fa_c * 60.0)
                    f_bas.append(has_fa_b * 60.0)
                    fe_cur.append(ev_fa_c * 60.0)
                    fe_bas.append(ev_fa_b * 60.0)
                    if rc["has_alert_a1_warn"]:
                        k_fa_total += 1
                    k_fa_events_total += int(rc["fa_a1_warn_events"])
                fa_hr_vec[nti_pos] = float(np.mean(f_cur))
                fa_hr_base_vec[nti_pos] = float(np.mean(f_bas))
                fa_ev_hr_vec[nti_pos] = float(np.mean(fe_cur))
                fa_ev_hr_base_vec[nti_pos] = float(np.mean(fe_bas))

            # Per-encounter false confirmed tracks per flight hour across all 900 encounters
            # Primary (D32): ever-confirmed tracks not on intruder; Secondary: end-of-run confirmed tracks
            ft_hr_vec = np.zeros(n_enc, dtype=np.float64)
            ft_hr_base_vec = np.zeros(n_enc, dtype=np.float64)
            k_ft_total = 0
            k_ft_end_total = 0
            for enc_i in range(n_enc):
                ft_cur = []
                ft_bas = []
                for s in active_seeds:
                    rc = per_seed_enc_results[s][enc_i]
                    rb = baseline_by_seed[s][enc_i]
                    ft_cur.append(float(rc["false_tracks"]) * 60.0)
                    ft_bas.append(float(rb["false_tracks"]) * 60.0)
                    k_ft_total += int(rc["false_tracks"])
                    k_ft_end_total += int(rc["false_tracks_end_of_run"])
                ft_hr_vec[enc_i] = float(np.mean(ft_cur))
                ft_hr_base_vec[enc_i] = float(np.mean(ft_bas))

            # Bootstrap CIs and paired differences vs lambda=0.0005
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

            # W7 Miss decomposition at t_CPA - 12.5 s
            threat_runs_all = [r for r in all_enc_results_flat if r["is_threat"]]
            miss_decomp = summarize_miss_decomposition(threat_runs_all)

            a1_w = agg["alerting"]["A1_warning"]
            k_missed_total = int(round(a1_w["frac_missed"] * a1_w["threats_count"]))
            k_ge12_total = int(round(a1_w["frac_ge_12_5s"] * a1_w["threats_count"]))
            k_ge25_total = int(round(a1_w["frac_ge_25_0s"] * a1_w["threats_count"]))

            rec = {
                "lambda": float(lam),
                "seeds": list(active_seeds),
                "n_seeds": int(n_seeds),
                "encounters_per_seed": int(n_enc),
                "total_runs": int(n_enc * n_seeds),
                "total_flight_hours": float(total_flight_hours),
                "non_threat_flight_hours": float(non_threat_flight_hours),
                "threats_count": int(a1_w["threats_count"]),
                "non_threats_count": int(a1_w["non_threats_count"]),
                # Primary metric 1: Missed-alert fraction on threat encounters
                "missed_alert_count": int(k_missed_total),
                "missed_alert_rate": float(a1_w["frac_missed"]),
                "missed_alert_cp_ci_95": list(a1_w["ci_missed"]),
                "missed_alert_boot_ci_95": [miss_boot["ci_low"], miss_boot["ci_high"]],
                "paired_diff_missed_rate": float(miss_diff_boot["mean"]),
                "paired_diff_missed_ci_95": [miss_diff_boot["ci_low"], miss_diff_boot["ci_high"]],
                "paired_diff_missed_upper_95": float(miss_diff_boot["upper_95"]),
                # Primary metric 2: False alerts per flight hour on non-threats (split into runs_with_false_alert and false_alert_events per F16)
                "false_alerts_count": int(k_fa_total),
                "false_alerts_per_hour": float(a1_w["false_alerts_per_hour"]),
                "false_alerts_garwood_ci_95": [fa_garwood["ci_low"], fa_garwood["ci_high"]],
                "false_alerts_boot_ci_95": [fa_hr_boot["ci_low"], fa_hr_boot["ci_high"]],
                "runs_with_false_alert_count": int(k_fa_total),
                "runs_with_false_alert_per_h": float(a1_w["runs_with_false_alert_per_h"]),
                "runs_with_false_alert_garwood_ci_95": [fa_garwood["ci_low"], fa_garwood["ci_high"]],
                "runs_with_false_alert_boot_ci_95": [fa_hr_boot["ci_low"], fa_hr_boot["ci_high"]],
                "false_alert_events_count": int(k_fa_events_total),
                "false_alert_events_per_h": float(a1_w["false_alert_events_per_h"]),
                "false_alert_events_garwood_ci_95": [fa_ev_garwood["ci_low"], fa_ev_garwood["ci_high"]],
                "false_alert_events_boot_ci_95": [fa_ev_hr_boot["ci_low"], fa_ev_hr_boot["ci_high"]],
                "paired_diff_fa_per_hour": float(fa_hr_diff_boot["mean"]),
                "paired_diff_fa_per_hour_ci_95": [fa_hr_diff_boot["ci_low"], fa_hr_diff_boot["ci_high"]],
                "paired_diff_fa_per_hour_upper_95": float(fa_hr_diff_boot["upper_95"]),
                "paired_diff_fa_events_per_h": float(fa_ev_hr_diff_boot["mean"]),
                "paired_diff_fa_events_per_h_ci_95": [fa_ev_hr_diff_boot["ci_low"], fa_ev_hr_diff_boot["ci_high"]],
                "paired_diff_fa_events_per_h_upper_95": float(fa_ev_hr_diff_boot["upper_95"]),
                # Primary metric 3: False confirmed tracks per flight hour (D32: ever-confirmed primary, end-of-run secondary)
                "false_tracks_count": int(k_ft_total),
                "false_tracks_per_hour": float(ft_garwood["rate"]),
                "false_tracks_garwood_ci_95": [ft_garwood["ci_low"], ft_garwood["ci_high"]],
                "false_tracks_garwood_upper_95": float(ft_garwood["upper_95"]),
                "false_tracks_end_of_run": int(k_ft_end_total),
                "false_tracks_end_of_run_per_hour": float(ft_end_garwood["rate"]),
                "false_tracks_end_of_run_garwood_ci_95": [ft_end_garwood["ci_low"], ft_end_garwood["ci_high"]],
                "false_tracks_end_of_run_garwood_upper_95": float(ft_end_garwood["upper_95"]),
                "paired_diff_ft_per_hour": float(ft_hr_diff_boot["mean"]),
                "paired_diff_ft_per_hour_ci_95": [ft_hr_diff_boot["ci_low"], ft_hr_diff_boot["ci_high"]],
                # Secondary metrics (reported, not gating)
                "warning_ge_12_5s_count": int(k_ge12_total),
                "warning_ge_12_5s_rate": float(a1_w["frac_ge_12_5s"]),
                "warning_ge_12_5s_ci_95": list(a1_w["ci_ge_12_5s"]),
                "paired_diff_ge_12_5s": float(lead12_diff_boot["mean"]),
                "paired_diff_ge_12_5s_ci_95": [lead12_diff_boot["ci_low"], lead12_diff_boot["ci_high"]],
                "warning_ge_25_0s_count": int(k_ge25_total),
                "warning_ge_25_0s_rate": float(a1_w["frac_ge_25_0s"]),
                "warning_ge_25_0s_ci_95": list(a1_w["ci_ge_25_0s"]),
                "paired_diff_ge_25_0s": float(lead25_diff_boot["mean"]),
                "paired_diff_ge_25_0s_ci_95": [lead25_diff_boot["ci_low"], lead25_diff_boot["ci_high"]],
                "true_track_coverage": float(agg["tracking"]["true_track_coverage"]),
                "detectable_frames_total": int(agg["tracking"]["detectable_frames_total"]),
                "confirmed_on_detectable_frames_total": int(agg["tracking"]["confirmed_on_detectable_frames_total"]),
                "fragmentations_total": int(agg["tracking"]["fragmentations_total"]),
                "fragmentations_per_seed": float(agg["tracking"]["fragmentations_total"] / n_seeds),
                "acq_delay_median_s": float(agg["tracking"]["acquisition_delay"]["median_s"]),
                "acq_delay_p95_s": float(agg["tracking"]["acquisition_delay"]["p95_s"]),
                "sigma_omega_total_radps": float(agg["los_rate"]["sigma_omega_total_radps"]),
                "cpu_runtime_s": float(wall_s),
                "frames_per_second": float(fps),
                "miss_decomposition": miss_decomp,
                "raw_results": agg,
            }
            records.append(rec)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # 3. Fixed-sequence pass rule (D29): test in increasing lambda, stop at first failure
    # - upper bound of missed-alert increase <= 0.05 (5 percentage points)
    # - upper bound of false-alert increase <= 0.5 per flight hour
    # - Garwood 95% upper bound of false confirmed tracks <= 1.0 per flight hour
    analysis_table = []
    sequence_stopped = False
    last_passing_lambda: float | None = None

    for r in records:
        lam = r["lambda"]
        # Upper bound from 95% paired bootstrap CI (and also check one-sided 95% upper bound)
        ub_miss_inc = float(r["paired_diff_missed_ci_95"][1])
        ub_fa_inc = float(r["paired_diff_fa_per_hour_ci_95"][1])
        ub_ft_garwood = float(r["false_tracks_garwood_upper_95"])
        ub_ft_garwood_2sided = float(r["false_tracks_garwood_ci_95"][1])

        pass_miss = bool(ub_miss_inc <= 0.05 + 1e-12)
        pass_fa = bool(ub_fa_inc <= 0.5 + 1e-12)
        pass_ft = bool(ub_ft_garwood <= 1.0 + 1e-12)
        point_passes = bool(pass_miss and pass_fa and pass_ft)

        if lam == lams[0]:
            # Baseline lambda=0.0005 is the reference for paired increases; check false tracks bound
            point_passes = bool(pass_ft)

        if not sequence_stopped and point_passes:
            seq_pass = True
            last_passing_lambda = lam
        else:
            seq_pass = False
            sequence_stopped = True

        analysis_table.append({
            "lambda": float(lam),
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
            # Legacy compatibility keys for report_tables_wp05.py
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

    return {
        "schema": "xvs.lambda_sweep_v2/1",
        "description": "WP-05 G0 / G-B14 W7 lambda sweep v2 under D29 (Common Random Numbers, 900 encounters, paired bootstrap, Garwood UB, miss decomposition)",
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
    parser = argparse.ArgumentParser(description="WP-05 G0 / G-B14 W7 Lambda Sweep v2 Runner (D29)")
    parser.add_argument("--workers", type=int, default=get_safe_workers(), help="Worker process count")
    parser.add_argument("--data-root", type=Path, default=None, help="Data root directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--quick-encounters", type=int, default=None, help="Optional encounter cap for fast testing")
    args = parser.parse_args(argv)

    safe_limit = get_safe_workers()
    if args.workers > safe_limit:
        print(f"WARNING: Clamping workers from {args.workers} to safe limit {safe_limit}")
        args.workers = safe_limit

    # Write only to worktree output by default (F6)
    data_root = args.data_root.resolve() if args.data_root is not None else (REPO_ROOT / "output")
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (data_root / "xvs" / "wp05" / "g0_lambda_sweep_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    git_status_start = get_git_status_porcelain()
    if args.quick_encounters is None and git_status_start:
        raise RuntimeError(f"Refusing to run full lambda sweep on dirty git working tree:\n{git_status_start}")

    start_commit = get_git_commit()
    start_hashes = compute_code_hashes()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print("=" * 72)
    print("WP-05 G0 / G-B14 W7 — Lambda Sweep v2 Runner (D29)")
    print(f"Data root:    {data_root}")
    print(f"Output dir:   {out_dir}")
    print(f"Workers:      {args.workers} (safe limit: {safe_limit})")
    print(f"Git commit:   {start_commit}")
    print("=" * 72)

    with open(MEASURED_CONFIG_PATH, "r", encoding="utf-8") as f:
        measured_cfg = json.load(f)

    sampler = EncounterSamplerV1()
    encounters = sampler.generate(t1_cruise_state)
    cache_path = REPO_ROOT / "data" / "cache" / "wp05_encounters_sampler_v1.jsonl"
    sha = export_encounters_jsonl(encounters, cache_path)
    if sha != EXPECTED_SAMPLER_SHA256:
        raise ValueError(f"Sampler SHA-256 mismatch: {sha} != {EXPECTED_SAMPLER_SHA256}")

    if args.quick_encounters is not None:
        encounters = encounters[: args.quick_encounters]
        print(f"Quick mode: using {len(encounters)} encounters.")

    get_ownship_states()
    truth_cache_map, truth_cache_info = load_or_compute_truth_cache(encounters, max_workers=args.workers)

    t0 = time.perf_counter()
    sweep_res = run_lambda_sweep_v2(
        encounters=encounters,
        base_config=measured_cfg,
        truth_cache_map=truth_cache_map,
        max_workers=args.workers,
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
    if not hashes_match:
        raise RuntimeError(
            f"Long-job rule violation: code hashes or git commit changed during run! "
            f"start={start_hashes}, end={end_hashes}"
        )

    sweep_res["total_wall_s"] = float(total_wall_s)
    sweep_res["workers_used"] = int(args.workers)
    sweep_res["truth_cache"] = truth_cache_info
    sweep_res["per_run_records_file"] = "output/xvs/wp05/g0_lambda_sweep_v2/per_run_records.jsonl"
    sweep_res["per_run_records_sha256"] = per_run_sha256
    sweep_res["provenance"] = {
        "batch": "GEMINI_G-B14_W7",
        "argv": list(argv) if argv is not None else list(sys.argv),
        "start_time_utc": start_time_iso,
        "end_time_utc": end_time_iso,
        "git_commit_start": start_commit,
        "git_commit_end": end_commit,
        "git_status_start": git_status_start,
        "git_status_end": git_status_end,
        "sampler_v1_sha256": sha,
        "per_run_records_file": "output/xvs/wp05/g0_lambda_sweep_v2/per_run_records.jsonl",
        "per_run_records_sha256": per_run_sha256,
        "configs_read": {
            "configs/xvs/tracking/g0_measured_d0.json": hashlib.sha256(MEASURED_CONFIG_PATH.read_bytes()).hexdigest(),
        },
        "code_hashes_start": start_hashes,
        "code_hashes_end": end_hashes,
        "code_hashes_unchanged_during_run": hashes_match,
        "platform": platform.platform(),
    }

    out_file = out_dir / "lambda_sweep_v2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sweep_res, f, indent=2)

    # Also write run_manifest.json in out_dir
    manifest_file = out_dir / "run_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(sweep_res["provenance"], f, indent=2)

    print("\n" + "=" * 72)
    print(f"Lambda Sweep v2 completed in {total_wall_s:.1f} s ({total_wall_s/60.0:.2f} min)")
    print(f"Designated lambda_req (D29): {sweep_res['lambda_req_statement']}")
    print(f"Saved results to: {out_file}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
