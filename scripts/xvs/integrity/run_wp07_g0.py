"""OpenXVS WP-07 Gate G0 — Integrity Evaluation Runner.

Executes the full G0 integrity assessment pipeline:
1. Tunes Monitor M0 thresholds on validation split for target FAR in {0.01, 0.001}.
2. Calibrates standard and Mondrian conformal protection levels on calibration split.
3. Evaluates test split metrics (coverage, HMI rates per fault type & camera set,
   fault-free and faulty availability, time-to-alarm per fault type in seconds,
   missed-detection table, and fault-free alarm reason breakdown).
4. Emits configs/xvs/integrity/g0.json and output/xvs/wp07/g0/results.json.

Conforms strictly to docs/xvs/WP07_INTEGRITY.md v0.1.3 §2, §3, §4, §5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, REPO_ROOT
from xvs.integrity.monitor import (
    MonitorThresholds,
    evaluate_monitor_alarm,
    tune_monitor_thresholds,
)
from xvs.integrity.conformal import (
    ALPHA_LEVELS,
    ConformalModel,
    fit_conformal_model,
)

DEFAULT_OUT_DIR = get_data_root() / "xvs" / "wp07" / "g0"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "integrity" / "g0.json"
ALERT_LIMITS_DEG = (0.1, 0.5)
FAULT_TYPES_ALL = ("none", "occlusion", "spurious_edge", "altitude_bias", "glare")
FAULT_TYPES_INJECTED = ("occlusion", "spurious_edge", "altitude_bias", "glare")
CAM_SETS = ("F", "FRBL", "all")


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> dict[str, float]:
    """Compute Clopper-Pearson exact binomial confidence interval and 95% upper bound."""
    if n <= 0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": 1.0, "upper_95": 1.0, "k": k, "n": n}

    rate = k / n
    alpha = 1.0 - confidence
    ci_low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    ci_high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    upper_95 = 1.0 - (alpha ** (1.0 / n)) if k == 0 else float(stats.beta.ppf(confidence, k + 1, n - k))

    return {
        "rate": float(rate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "upper_95": float(upper_95),
        "k": int(k),
        "n": int(n),
    }


def load_jsonl(fpath: Path, expected_split: str | None = None) -> list[dict[str, Any]]:
    if not fpath.is_file():
        raise FileNotFoundError(f"Missing dataset file: {fpath}")
    records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("cam_set") == "horizontal":
                    r["cam_set"] = "FRBL"
                # D28 / W2: Condition must be explicit. No fallback to mode.
                if "condition" not in r:
                    raise KeyError(f"Record in {fpath.name} missing required 'condition' field (no fallback to mode).")
                if expected_split is not None and r.get("split") and r.get("split") != expected_split:
                    raise ValueError(f"Record split '{r.get('split')}' does not match expected '{expected_split}' in {fpath.name}")
                records.append(r)
    return records


def _compute_tta_summary(seq_groups: dict[tuple, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compute time-to-alarm statistics (in seconds and frames) for a set of fault sequences."""
    tta_frames_list: list[int] = []
    tta_seconds_list: list[float] = []
    never_alarmed_count = 0
    total_fault_sequences = len(seq_groups)

    for grp_key, seq_recs in seq_groups.items():
        seq_recs_sorted = sorted(seq_recs, key=lambda x: x["k"])
        k_onset = seq_recs_sorted[0]["k_onset"]
        t_onset = None
        for r in seq_recs_sorted:
            if r["k"] == k_onset:
                t_onset = r.get("t_s", 0.0)  # default-ok: optional fallback default
                break
        if t_onset is None and seq_recs_sorted:
            t_onset = seq_recs_sorted[0].get("t_s", 0.0)  # default-ok: optional fallback default

        alarmed_rec = None
        for r in seq_recs_sorted:
            if r["k"] >= k_onset and r["alarm"]:
                alarmed_rec = r
                break

        if alarmed_rec is not None:
            delta_k = int(alarmed_rec["k"] - k_onset)
            delta_t = float(alarmed_rec.get("t_s", 0.0) - t_onset)  # default-ok: optional fallback default
            tta_frames_list.append(delta_k)
            tta_seconds_list.append(delta_t)
        else:
            never_alarmed_count += 1

    n_seq = max(1, total_fault_sequences)
    frac_fr_1 = sum(1 for dk in tta_frames_list if dk <= 1) / n_seq
    frac_fr_3 = sum(1 for dk in tta_frames_list if dk <= 3) / n_seq
    frac_fr_10 = sum(1 for dk in tta_frames_list if dk <= 10) / n_seq

    frac_0_5s = sum(1 for dt in tta_seconds_list if dt <= 0.5 + 1e-9) / n_seq
    frac_1_5s = sum(1 for dt in tta_seconds_list if dt <= 1.5 + 1e-9) / n_seq
    frac_5_0s = sum(1 for dt in tta_seconds_list if dt <= 5.0 + 1e-9) / n_seq
    frac_never = never_alarmed_count / n_seq

    return {
        "total_fault_sequences": int(total_fault_sequences),
        "alarmed_sequences": int(len(tta_frames_list)),
        "never_alarmed_count": int(never_alarmed_count),
        "fraction_alarmed_within_0_5s": float(frac_0_5s),
        "fraction_alarmed_within_1_5s": float(frac_1_5s),
        "fraction_alarmed_within_5_0s": float(frac_5_0s),
        "fraction_alarmed_within_1_frame": float(frac_fr_1),
        "fraction_alarmed_within_3_frames": float(frac_fr_3),
        "fraction_alarmed_within_10_frames": float(frac_fr_10),
        "fraction_never_alarmed": float(frac_never),
        "median_tta_frames": float(np.median(tta_frames_list)) if tta_frames_list else float("nan"),
        "mean_tta_frames": float(np.mean(tta_frames_list)) if tta_frames_list else float("nan"),
        "median_tta_seconds": float(np.median(tta_seconds_list)) if tta_seconds_list else float("nan"),
        "mean_tta_seconds": float(np.mean(tta_seconds_list)) if tta_seconds_list else float("nan"),
    }


def evaluate_test_metrics(
    test_records: list[dict[str, Any]],
    thresholds: MonitorThresholds,
    conformal_model: ConformalModel,
    alphas: tuple[float, ...] = ALPHA_LEVELS,
    alert_limits: tuple[float, ...] = ALERT_LIMITS_DEG,
) -> dict[str, Any]:
    """Compute comprehensive test metrics under a specific threshold configuration."""
    # 1. Annotate each test record with alarm status
    annotated = []
    for r in test_records:
        rec = dict(r)
        alarm_res = evaluate_monitor_alarm(rec, thresholds)
        rec["alarm"] = alarm_res["alarm"]
        rec["alarm_reasons"] = alarm_res["reasons"]
        annotated.append(rec)

    # 2. Coverage on fault-free test frames without alarm
    fault_free_unalarmed = [
        r for r in annotated
        if r.get("fault_type") == "none" and not r["alarm"] and not math.isnan(r.get("total_error_deg", float("nan")))
    ]

    has_sys = bool(conformal_model.standard_quantiles_sys)
    coverage_results: dict[str, Any] = {}
    coverage_by_cam_results: dict[str, Any] = {}
    coverage_sys_results: dict[str, Any] = {}
    coverage_sys_by_cam_results: dict[str, Any] = {}

    for mondrian in (False, True):
        variant = "mondrian" if mondrian else "standard"
        coverage_results[variant] = {}
        coverage_by_cam_results[variant] = {}
        if has_sys:
            coverage_sys_results[variant] = {}
            coverage_sys_by_cam_results[variant] = {}
        for alpha in alphas:
            a_key = f"{alpha:.4f}".rstrip("0")
            n_tot = len(fault_free_unalarmed)
            covered = 0
            covered_sys = 0
            by_cam_cov: dict[str, Any] = {}
            by_cam_cov_sys: dict[str, Any] = {}

            for cset in CAM_SETS:
                recs_c = [r for r in fault_free_unalarmed if r.get("cam_set") == cset]
                cov_c = 0
                cov_sys_c = 0
                for r in recs_c:
                    if "sigma_hat_urad" not in r or r["sigma_hat_urad"] is None:
                        raise KeyError("Record missing required 'sigma_hat_urad' field")
                    if "condition" not in r or not r["condition"]:
                        raise KeyError("Record missing required 'condition' field")
                    sig_rad = float(r["sigma_hat_urad"]) * 1e-6
                    condition = str(r["condition"])
                    pl_deg = conformal_model.compute_pl_deg(sig_rad, alpha, cam_set=cset, condition=condition, mondrian=mondrian)
                    if r["total_error_deg"] <= pl_deg:
                        cov_c += 1
                    if has_sys:
                        pl_sys_deg = conformal_model.compute_pl_deg_sys(sig_rad, alpha, cam_set=cset, condition=condition, mondrian=mondrian)
                        if r["total_error_deg"] <= pl_sys_deg:
                            cov_sys_c += 1
                covered += cov_c
                covered_sys += cov_sys_c

                target_cov = 1.0 - alpha
                cd = clopper_pearson_ci(cov_c, len(recs_c))
                cd["target"] = target_cov
                cd["within_band_2pct"] = abs(cd["rate"] - target_cov) <= 0.02
                by_cam_cov[cset] = cd

                if has_sys:
                    cds = clopper_pearson_ci(cov_sys_c, len(recs_c))
                    cds["target"] = target_cov
                    cds["within_band_2pct"] = abs(cds["rate"] - target_cov) <= 0.02
                    by_cam_cov_sys[cset] = cds

            target_cov = 1.0 - alpha
            cov_dict = clopper_pearson_ci(covered, n_tot)
            cov_dict["target"] = target_cov
            cov_dict["within_band_2pct"] = abs(cov_dict["rate"] - target_cov) <= 0.02
            coverage_results[variant][a_key] = cov_dict
            by_cam_cov["overall"] = cov_dict
            coverage_by_cam_results[variant][a_key] = by_cam_cov

            if has_sys:
                cov_sys_dict = clopper_pearson_ci(covered_sys, n_tot)
                cov_sys_dict["target"] = target_cov
                cov_sys_dict["within_band_2pct"] = abs(cov_sys_dict["rate"] - target_cov) <= 0.02
                coverage_sys_results[variant][a_key] = cov_sys_dict
                by_cam_cov_sys["overall"] = cov_sys_dict
                coverage_sys_by_cam_results[variant][a_key] = by_cam_cov_sys

    # 3. Hazardously Misleading Information (HMI)
    # HMI: error > AL, error > PL(alpha), and alarm == False
    # Spec v0.1.3 §4: report fault-free and each fault type, per camera set (F, FRBL, all) and overall.
    hmi_results: dict[str, Any] = {}
    hmi_by_cam_results: dict[str, Any] = {}
    hmi_sys_results: dict[str, Any] = {}
    hmi_sys_by_cam_results: dict[str, Any] = {}

    for mondrian in (False, True):
        variant = "mondrian" if mondrian else "standard"
        hmi_results[variant] = {}
        hmi_by_cam_results[variant] = {}
        if has_sys:
            hmi_sys_results[variant] = {}
            hmi_sys_by_cam_results[variant] = {}
        for al in alert_limits:
            al_key = f"{al:.2f}deg"
            hmi_results[variant][al_key] = {}
            hmi_by_cam_results[variant][al_key] = {}
            if has_sys:
                hmi_sys_results[variant][al_key] = {}
                hmi_sys_by_cam_results[variant][al_key] = {}
            for alpha in alphas:
                a_key = f"{alpha:.4f}".rstrip("0")
                by_fault: dict[str, Any] = {}
                by_fault_cam: dict[str, dict[str, Any]] = {}
                by_fault_sys: dict[str, Any] = {}
                by_fault_cam_sys: dict[str, dict[str, Any]] = {}
                tot_hmi = 0
                tot_hmi_sys = 0
                tot_frames = 0
                tot_by_cam_k = {c: 0 for c in CAM_SETS}
                tot_by_cam_k_sys = {c: 0 for c in CAM_SETS}
                tot_by_cam_n = {c: 0 for c in CAM_SETS}

                for ft in FAULT_TYPES_ALL:
                    # Active frames of this fault type (note: includes FAILED_NO_EDGES frames, which are alarmed so not HMI,
                    # or converged frames; wait: keep exact denominator compatibility by checking fault_active or ft == 'none')
                    recs_ft_all = [
                        r for r in annotated
                        if (r.get("fault_type") == ft) and (r.get("fault_active", True) or ft == "none")
                        and not math.isnan(r.get("total_error_deg", float("nan")))
                    ]
                    by_fault_cam[ft] = {}
                    if has_sys:
                        by_fault_cam_sys[ft] = {}

                    k_ft_tot = 0
                    k_ft_sys_tot = 0
                    n_ft_tot = 0

                    for cset in CAM_SETS:
                        recs_ft_c = [r for r in recs_ft_all if r.get("cam_set") == cset]
                        n_c = len(recs_ft_c)
                        k_c = 0
                        k_c_sys = 0
                        for r in recs_ft_c:
                            if "sigma_hat_urad" not in r or r["sigma_hat_urad"] is None:
                                raise KeyError("Record missing required 'sigma_hat_urad' field")
                            if "condition" not in r or not r["condition"]:
                                raise KeyError("Record missing required 'condition' field")
                            sig_rad = float(r["sigma_hat_urad"]) * 1e-6
                            condition = str(r["condition"])
                            pl_deg = conformal_model.compute_pl_deg(sig_rad, alpha, cam_set=cset, condition=condition, mondrian=mondrian)
                            err = r["total_error_deg"]
                            if (not r["alarm"]) and (err > al) and (err > pl_deg):
                                k_c += 1
                            if has_sys:
                                pl_sys_deg = conformal_model.compute_pl_deg_sys(sig_rad, alpha, cam_set=cset, condition=condition, mondrian=mondrian)
                                if (not r["alarm"]) and (err > al) and (err > pl_sys_deg):
                                    k_c_sys += 1
                        by_fault_cam[ft][cset] = clopper_pearson_ci(k_c, n_c)
                        if has_sys:
                            by_fault_cam_sys[ft][cset] = clopper_pearson_ci(k_c_sys, n_c)
                        k_ft_tot += k_c
                        k_ft_sys_tot += k_c_sys
                        n_ft_tot += n_c
                        tot_by_cam_k[cset] += k_c
                        tot_by_cam_k_sys[cset] += k_c_sys
                        tot_by_cam_n[cset] += n_c

                    by_fault[ft] = clopper_pearson_ci(k_ft_tot, n_ft_tot)
                    by_fault_cam[ft]["overall"] = by_fault[ft]
                    tot_hmi += k_ft_tot
                    if has_sys:
                        by_fault_sys[ft] = clopper_pearson_ci(k_ft_sys_tot, n_ft_tot)
                        by_fault_cam_sys[ft]["overall"] = by_fault_sys[ft]
                        tot_hmi_sys += k_ft_sys_tot
                    tot_frames += n_ft_tot

                by_fault["overall"] = clopper_pearson_ci(tot_hmi, tot_frames)
                by_fault_cam["overall"] = {
                    cset: clopper_pearson_ci(tot_by_cam_k[cset], tot_by_cam_n[cset])
                    for cset in CAM_SETS
                }
                by_fault_cam["overall"]["overall"] = by_fault["overall"]

                hmi_results[variant][al_key][a_key] = by_fault
                hmi_by_cam_results[variant][al_key][a_key] = by_fault_cam
                if has_sys:
                    by_fault_sys["overall"] = clopper_pearson_ci(tot_hmi_sys, tot_frames)
                    by_fault_cam_sys["overall"] = {
                        cset: clopper_pearson_ci(tot_by_cam_k_sys[cset], tot_by_cam_n[cset])
                        for cset in CAM_SETS
                    }
                    by_fault_cam_sys["overall"]["overall"] = by_fault_sys["overall"]
                    hmi_sys_results[variant][al_key][a_key] = by_fault_sys
                    hmi_sys_by_cam_results[variant][al_key][a_key] = by_fault_cam_sys

    # 4. Availability (Spec v0.1.3 §4)
    # Primary availability: computed on FAULT-FREE test frames, per camera set and overall.
    # Faulty-frame availability: computed on active FAULTY test frames, reported separately.
    def _compute_avail_subset(subset_records: list[dict[str, Any]], use_sys: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for mondrian in (False, True):
            variant = "mondrian" if mondrian else "standard"
            out[variant] = {}
            for al in alert_limits:
                al_key = f"{al:.2f}deg"
                out[variant][al_key] = {}
                for alpha in alphas:
                    a_key = f"{alpha:.4f}".rstrip("0")
                    avail_by_cam: dict[str, Any] = {}
                    for cset in CAM_SETS:
                        recs_c = [r for r in subset_records if r.get("cam_set") == cset]
                        k_avail = 0
                        n_c = len(recs_c)
                        for r in recs_c:
                            if "sigma_hat_urad" not in r or r["sigma_hat_urad"] is None:
                                raise KeyError("Record missing required 'sigma_hat_urad' field")
                            if "condition" not in r or not r["condition"]:
                                raise KeyError("Record missing required 'condition' field")
                            sig_rad = float(r["sigma_hat_urad"]) * 1e-6
                            condition = str(r["condition"])
                            if use_sys:
                                pl_deg = conformal_model.compute_pl_deg_sys(sig_rad, alpha, cam_set=cset, condition=condition, mondrian=mondrian)
                            else:
                                pl_deg = conformal_model.compute_pl_deg(sig_rad, alpha, cam_set=cset, condition=condition, mondrian=mondrian)
                            if (not r["alarm"]) and (pl_deg <= al):
                                k_avail += 1
                        avail_by_cam[cset] = clopper_pearson_ci(k_avail, n_c)
                    k_all = sum(avail_by_cam[c]["k"] for c in CAM_SETS)
                    n_all = sum(avail_by_cam[c]["n"] for c in CAM_SETS)
                    avail_by_cam["overall"] = clopper_pearson_ci(k_all, n_all)
                    out[variant][al_key][a_key] = avail_by_cam
        return out

    fault_free_test_recs = [r for r in annotated if r.get("fault_type") == "none"]
    faulty_active_test_recs = [
        r for r in annotated
        if r.get("fault_type") in FAULT_TYPES_INJECTED and r.get("fault_active", True)
    ]

    availability_results = _compute_avail_subset(fault_free_test_recs, use_sys=False)
    availability_faulty_results = _compute_avail_subset(faulty_active_test_recs, use_sys=False)
    availability_all_frames_results = _compute_avail_subset(annotated, use_sys=False)
    if has_sys:
        availability_sys_results = _compute_avail_subset(fault_free_test_recs, use_sys=True)
        availability_faulty_sys_results = _compute_avail_subset(faulty_active_test_recs, use_sys=True)

    # 5. Time to Alarm (Fault Set) — overall and per fault type
    fault_sequences: dict[tuple, list[dict[str, Any]]] = {}
    fault_sequences_by_ft: dict[str, dict[tuple, list[dict[str, Any]]]] = {ft: {} for ft in FAULT_TYPES_INJECTED}
    for r in annotated:
        ft = r.get("fault_type", "none")
        if ft in FAULT_TYPES_INJECTED and r.get("k_onset") is not None:
            grp_key = (r["seq_name"], r.get("seed_idx", 0), ft, r.get("cam_set"), r.get("sigma_h"))  # default-ok: optional fallback default
            fault_sequences.setdefault(grp_key, []).append(r)
            fault_sequences_by_ft[ft].setdefault(grp_key, []).append(r)

    tta_results = _compute_tta_summary(fault_sequences)
    tta_results["by_fault_type"] = {
        ft: _compute_tta_summary(fault_sequences_by_ft[ft])
        for ft in FAULT_TYPES_INJECTED
    }

    # 6. Missed-Detection Table: fault_type x cam_set
    missed_detection_table: dict[str, Any] = {}
    for al in alert_limits:
        al_key = f"{al:.2f}deg"
        matrix: dict[str, dict[str, int]] = {}
        for ft in FAULT_TYPES_INJECTED:
            matrix[ft] = {}
            for cset in CAM_SETS:
                faulty_recs = [
                    r for r in annotated
                    if r.get("fault_type") == ft and r.get("cam_set") == cset
                    and r.get("fault_active", True)
                    and not math.isnan(r.get("total_error_deg", float("nan")))
                ]
                missed = sum(1 for r in faulty_recs if (not r["alarm"]) and (r["total_error_deg"] > al))
                matrix[ft][cset] = int(missed)
        missed_detection_table[al_key] = matrix

    res_dict = {
        "far_target": thresholds.far_target,
        "n_feature_errors": sum(1 for r in test_records if r.get("status") == "feature_error"),
        "coverage": coverage_results,
        "coverage_by_cam_set": coverage_by_cam_results,
        "hmi": hmi_results,
        "hmi_by_cam_set": hmi_by_cam_results,
        "availability": availability_results,
        "availability_faulty": availability_faulty_results,
        "availability_all_test_frames": availability_all_frames_results,
        "time_to_alarm": tta_results,
        "missed_detections": missed_detection_table,
    }
    if has_sys:
        res_dict["coverage_sys"] = coverage_sys_results
        res_dict["coverage_sys_by_cam_set"] = coverage_sys_by_cam_results
        res_dict["hmi_sys"] = hmi_sys_results
        res_dict["hmi_sys_by_cam_set"] = hmi_sys_by_cam_results
        res_dict["availability_sys"] = availability_sys_results
        res_dict["availability_faulty_sys"] = availability_faulty_sys_results
    return res_dict


def compute_achieved_far_with_reasons(
    records: list[dict[str, Any]],
    thresholds: MonitorThresholds,
) -> dict[str, Any]:
    """Compute achieved FAR on fault-free records with full breakdown by alarm reason."""
    def _analyze_subset(recs: list[dict[str, Any]], cset_hint: str | None = None) -> dict[str, Any]:
        n_tot = len(recs)
        k_alarm = 0
        reason_counts = {
            "STATUS_NOT_CONVERGED": 0,
            "FEATURE_ERROR": 0,
            "RMS_HIGH": 0,
            "COND_HIGH": 0,
            "INLIER_FRAC_LOW": 0,
            "INLIER_COUNT_LOW": 0,
            "LOO_HIGH": 0,
            "NOT_AVAILABLE": 0,
        }
        sole_reason_counts = {k: 0 for k in reason_counts}
        n_loo_not_available = 0
        n_min_cnt_boundary_ties = 0

        for r in recs:
            cset = r.get("cam_set", cset_hint or "")
            th_cam = thresholds.by_cam_set.get(cset, {}) if thresholds.by_cam_set else {}
            min_cnt = th_cam.get("min_inlier_count", thresholds.min_inlier_count)
            if min_cnt is not None and r.get("status") == "CONVERGED":
                if int(r["inlier_count"]) == int(min_cnt):
                    n_min_cnt_boundary_ties += 1

            if cset in ("FRBL", "horizontal", "all"):
                loo = r.get("camera_consistency_deg")
                if loo is None or (isinstance(loo, float) and math.isnan(loo)):
                    loo2 = r.get("leave_one_out_consistency_deg", float("nan"))
                    if loo2 is None or (isinstance(loo2, float) and math.isnan(loo2)):
                        n_loo_not_available += 1

            ev = evaluate_monitor_alarm(r, thresholds)
            if not ev["alarm"]:
                continue
            k_alarm += 1
            reasons = ev["reasons"]
            active_cats = []
            if r.get("status") == "feature_error":
                reason_counts["FEATURE_ERROR"] += 1
                active_cats.append("FEATURE_ERROR")
            elif r.get("status") != "CONVERGED":
                reason_counts["STATUS_NOT_CONVERGED"] += 1
                active_cats.append("STATUS_NOT_CONVERGED")

            if any(x.startswith("RMS_HIGH") for x in reasons):
                reason_counts["RMS_HIGH"] += 1
                active_cats.append("RMS_HIGH")
            if any(x.startswith("COND_HIGH") for x in reasons):
                reason_counts["COND_HIGH"] += 1
                active_cats.append("COND_HIGH")
            if any(x.startswith("INLIER_FRAC_LOW") for x in reasons):
                reason_counts["INLIER_FRAC_LOW"] += 1
                active_cats.append("INLIER_FRAC_LOW")
            if any(x.startswith("INLIER_COUNT_LOW") for x in reasons):
                reason_counts["INLIER_COUNT_LOW"] += 1
                active_cats.append("INLIER_COUNT_LOW")
            if any(x.startswith("LOO_HIGH") for x in reasons):
                reason_counts["LOO_HIGH"] += 1
                active_cats.append("LOO_HIGH")

            if len(active_cats) == 1:
                sole_reason_counts[active_cats[0]] += 1

        ci = clopper_pearson_ci(k_alarm, n_tot)
        ci["by_reason"] = reason_counts
        ci["by_reason_rate"] = {k: (v / n_tot if n_tot > 0 else 0.0) for k, v in reason_counts.items()}
        ci["sole_reason_counts"] = sole_reason_counts
        ci["n_loo_not_available"] = int(n_loo_not_available)
        ci["n_min_cnt_boundary_ties"] = int(n_min_cnt_boundary_ties)
        return ci

    by_cam = {}
    for cset in CAM_SETS:
        recs_c = [r for r in records if r.get("cam_set") == cset]
        by_cam[cset] = _analyze_subset(recs_c, cset_hint=cset)
    overall = _analyze_subset(records)

    # Check if an uncontrollable floor (STATUS_NOT_CONVERGED or integer quantization on min_cnt) exceeds target_far
    n_tot = max(1, overall["n"])
    status_floor_k = overall["by_reason"]["STATUS_NOT_CONVERGED"]
    status_floor_rate = status_floor_k / n_tot
    cnt_low_k = overall["by_reason"]["INLIER_COUNT_LOW"]
    cnt_low_rate = cnt_low_k / n_tot
    uncontrollable_k = overall["sole_reason_counts"]["STATUS_NOT_CONVERGED"]
    # Also check union of STATUS_NOT_CONVERGED and INLIER_COUNT_LOW vs continuous thresholds
    continuous_only_k = 0
    for r in records:
        if r.get("status") != "CONVERGED":
            continue
        ev = evaluate_monitor_alarm(r, thresholds)
        if any(
            x.startswith(("RMS_HIGH", "COND_HIGH", "INLIER_FRAC_LOW", "LOO_HIGH"))
            for x in ev["reasons"]
        ):
            continuous_only_k += 1
    continuous_only_rate = continuous_only_k / n_tot
    non_continuous_floor_k = overall["k"] - continuous_only_k
    non_continuous_floor_rate = non_continuous_floor_k / n_tot

    target_met = bool(overall["ci_low"] <= thresholds.far_target)
    verdict = "PASS" if target_met else "FAIL"

    return {
        "overall": overall,
        "by_cam_set": by_cam,
        "target_far": thresholds.far_target,
        "target_met": target_met,
        "verdict": verdict,
        "floor_analysis": {
            "status_not_converged_k": int(status_floor_k),
            "status_not_converged_rate": float(status_floor_rate),
            "inlier_count_low_k": int(cnt_low_k),
            "inlier_count_low_rate": float(cnt_low_rate),
            "continuous_features_only_k": int(continuous_only_k),
            "continuous_features_only_rate": float(continuous_only_rate),
            "non_continuous_floor_k": int(non_continuous_floor_k),
            "non_continuous_floor_rate": float(non_continuous_floor_rate),
        },
    }


def compute_scale_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute and report sigma_hat against actual error per camera set (F-B12-5)."""
    nominal = [r for r in records if r.get("fault_type", "none") in ("none", None) and r.get("status") == "CONVERGED"]
    stats_out = {}
    for cset in CAM_SETS:
        recs = [r for r in nominal if r.get("cam_set") == cset]
        if not recs:
            continue
        errs = [r["total_error_deg"] for r in recs if not math.isnan(r.get("total_error_deg", float("nan")))]
        sigs_deg = []
        for r in recs:
            if "sigma_hat_urad" not in r or r["sigma_hat_urad"] is None:
                raise KeyError("Record missing required 'sigma_hat_urad' field")
            val_sig = float(r["sigma_hat_urad"])
            if not math.isnan(val_sig):
                sigs_deg.append(math.degrees(val_sig * 1e-6))
        if not errs or not sigs_deg:
            continue
        e_arr = np.array(errs)
        s_arr = np.array(sigs_deg)
        ratio_arr = e_arr / s_arr
        s_sys_deg = max(0.0, float(np.mean(e_arr) - np.mean(s_arr)))
        stats_out[cset] = {
            "n": len(e_arr),
            "mean_error_deg": float(np.mean(e_arr)),
            "median_error_deg": float(np.median(e_arr)),
            "rmse_error_deg": float(np.sqrt(np.mean(e_arr ** 2))),
            "mean_sigma_hat_deg": float(np.mean(s_arr)),
            "median_sigma_hat_deg": float(np.median(s_arr)),
            "rmse_sigma_hat_deg": float(np.sqrt(np.mean(s_arr ** 2))),
            "ratio_mean": float(np.mean(ratio_arr)),
            "ratio_median": float(np.median(ratio_arr)),
            "fitted_sigma_sys_deg": s_sys_deg,
            "fitted_sigma_sys_urad": float(math.radians(s_sys_deg) * 1e6),
        }
    return stats_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run WP-07 G0 Integrity Evaluation")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    data_root = get_data_root(args.data_root)
    data_dir = args.data_dir or (data_root / "xvs" / "wp07" / "g0")
    config_path = args.config_path
    out_json = args.out_json or (data_dir / "results.json")

    print("=" * 70)
    print("OpenXVS WP-07 Gate G0 — Integrity Evaluation Runner")
    print(f"Data Root:       {data_root}")
    print(f"Data Directory:  {data_dir}")
    print(f"Config File:     {config_path}")
    print(f"Output Results:  {out_json}")
    print("=" * 70)

    # Refuse a dirty git working tree
    git_status_proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    dirty_lines = [ln for ln in git_status_proc.stdout.splitlines() if ln.strip()]
    if dirty_lines:
        raise RuntimeError(f"Refusing to run on a dirty git working tree: {dirty_lines}")

    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()

    # W4 provenance check: frame periods from G4 timestamps (0.5 s for T1/T3, 0.2 s for T2)
    g4_dir = data_root / "xvs" / "wp01" / "g4"
    if not g4_dir.is_dir():
        raise FileNotFoundError(f"Missing WP-01 G4 directory required for frame-period verification: {g4_dir}")
    for seq, exp_dt in (("T1_cruise", 0.5), ("T3_turn_climb", 0.5), ("T2_ksfo28l", 0.2)):
        f_path = g4_dir / seq / "frames.jsonl"
        if not f_path.is_file():
            raise FileNotFoundError(f"Missing G4 frames.jsonl file for {seq}: {f_path}")
        with open(f_path, "r", encoding="utf-8") as f:
            l1 = f.readline()
            l2 = f.readline()
            if not (l1 and l2):
                raise ValueError(f"Insufficient frame records in {f_path}")
            dt = float(json.loads(l2)["t_s"]) - float(json.loads(l1)["t_s"])
            assert abs(dt - exp_dt) < 1e-6, f"Frame period mismatch for {seq}: got {dt}, expected {exp_dt}"

    # W4 provenance check: verify manifest.code_hashes against git show <manifest.commit>:<path>
    manifest_file = data_dir / "data_manifest.json"
    manifest_commit = None
    if manifest_file.is_file():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if "data_generation_code_hashes" in manifest:
            raise RuntimeError("data_manifest.json contains hand-edited key 'data_generation_code_hashes'")
        manifest_commit = manifest.get("commit", "")
        if not manifest_commit:
            raise RuntimeError("data_manifest.json missing 'commit' field")
        if "code_hashes" not in manifest or not manifest["code_hashes"]:
            raise RuntimeError("data_manifest.json missing 'code_hashes' field")
        for rel_c, exp_h in manifest["code_hashes"].items():
            blob_proc = subprocess.run(
                ["git", "show", f"{manifest_commit}:{rel_c}"],
                capture_output=True,
                cwd=REPO_ROOT,
            )
            if blob_proc.returncode != 0:
                raise RuntimeError(
                    f"git show {manifest_commit}:{rel_c} failed: {blob_proc.stderr.decode('utf-8', errors='replace')}"
                )
            blob = blob_proc.stdout
            lf = blob.replace(b"\r\n", b"\n")
            crlf = lf.replace(b"\n", b"\r\n")
            valid_hashes = {
                hashlib.sha256(blob).hexdigest(),
                hashlib.sha256(lf).hexdigest(),
                hashlib.sha256(crlf).hexdigest(),
            }
            if exp_h not in valid_hashes:
                raise RuntimeError(
                    f"Manifest code_hash mismatch for {rel_c} at commit {manifest_commit}: "
                    f"recorded {exp_h}, git blob hashes {sorted(valid_hashes)}"
                )

    val_file = data_dir / "dataset_validation.jsonl"
    cal_file = data_dir / "dataset_calibration.jsonl"
    test_file = data_dir / "dataset_test.jsonl"

    print("\n---> 1. Loading datasets ...")
    print(f"Loading validation split: {val_file}")
    val_records = load_jsonl(val_file, expected_split="validation")
    print(f"  Loaded {len(val_records)} validation records.")

    print(f"Loading calibration split: {cal_file}")
    cal_records = load_jsonl(cal_file, expected_split="calibration")
    print(f"  Loaded {len(cal_records)} calibration records.")

    print(f"Loading test split: {test_file}")
    test_records = load_jsonl(test_file, expected_split="test")
    print(f"  Loaded {len(test_records)} test records.")

    # 1. Tune thresholds on validation split
    print("\n---> 2. Tuning Monitor M0 thresholds on validation split ...")
    nominal_val = [r for r in val_records if r.get("fault_type") == "none"]
    print(f"  Nominal fault-free validation records: {len(nominal_val)}")

    th_01 = tune_monitor_thresholds(nominal_val, target_far=0.01)
    th_001 = tune_monitor_thresholds(nominal_val, target_far=0.001)

    print(f"  Tuned FAR 1.0%: max_rms={th_01.max_rms_residual_urad:.1f} urad, "
          f"max_cond={th_01.max_condition_number:.1e}, min_frac={th_01.min_inlier_fraction:.2f}, "
          f"min_cnt={th_01.min_inlier_count}, max_loo={th_01.max_leave_one_out_deg:.4f} deg")

    print(f"  Tuned FAR 0.1%: max_rms={th_001.max_rms_residual_urad:.1f} urad, "
          f"max_cond={th_001.max_condition_number:.1e}, min_frac={th_001.min_inlier_fraction:.2f}, "
          f"min_cnt={th_001.min_inlier_count}, max_loo={th_001.max_leave_one_out_deg:.4f} deg")

    # Save to configs/xvs/integrity/g0.json
    config_path.parent.mkdir(parents=True, exist_ok=True)
    configs_data = {
        "far_0.01": th_01.to_dict(),
        "far_0.001": th_001.to_dict(),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(configs_data, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Saved threshold configs to {config_path}")

    # 2. Calibrate Conformal Models on alarm-free calibration records
    print("\n---> 3. Calibrating Conformal Models on calibration split ...")
    cal_alarm_free_01 = [
        r for r in cal_records
        if (r.get("fault_type", "none") in ("none", None))
        and (not evaluate_monitor_alarm(r, th_01)["alarm"])
        and r.get("status") == "CONVERGED"
    ]
    conformal_01 = fit_conformal_model(cal_alarm_free_01)

    cal_alarm_free_001 = [
        r for r in cal_records
        if (r.get("fault_type", "none") in ("none", None))
        and (not evaluate_monitor_alarm(r, th_001)["alarm"])
        and r.get("status") == "CONVERGED"
    ]
    conformal_001 = fit_conformal_model(cal_alarm_free_001)

    print(f"  FAR 0.01: {len(cal_alarm_free_01)} alarm-free calibration records.")
    print(f"    Standard quantiles: {conformal_01.standard_quantiles}")
    print(f"  FAR 0.001: {len(cal_alarm_free_001)} alarm-free calibration records.")
    print(f"    Standard quantiles: {conformal_001.standard_quantiles}")

    # Compute scale comparison (F-B12-5)
    print("\n---> 3b. Computing sigma_hat vs actual error scale comparison ...")
    scale_comp = compute_scale_comparison(cal_records)
    for cset, s in scale_comp.items():
        print(f"  [{cset}] n={s['n']}, mean_error={s['mean_error_deg']:.4f} deg, "
              f"mean_sigma_hat={s['mean_sigma_hat_deg']:.4f} deg, "
              f"ratio_mean={s['ratio_mean']:.2f}x, ratio_median={s['ratio_median']:.2f}x, "
              f"fitted_sigma_sys={s['fitted_sigma_sys_deg']:.4f} deg ({s['fitted_sigma_sys_urad']:.0f} urad)")

    # 3. Evaluate Metrics on Test Split
    print("\n---> 4. Evaluating test split metrics ...")
    metrics_01 = evaluate_test_metrics(test_records, th_01, conformal_01)
    metrics_001 = evaluate_test_metrics(test_records, th_001, conformal_001)

    # 4. Achieved FAR on validation and test splits with full reason breakdown (D28 / W2)
    nominal_val = [r for r in val_records if r.get("fault_type", "none") in ("none", None)]
    nominal_test = [r for r in test_records if r.get("fault_type", "none") in ("none", None)]

    achieved_far = {
        "far_0.01": {
            "validation": compute_achieved_far_with_reasons(nominal_val, th_01),
            "test": compute_achieved_far_with_reasons(nominal_test, th_01),
        },
        "far_0.001": {
            "validation": compute_achieved_far_with_reasons(nominal_val, th_001),
            "test": compute_achieved_far_with_reasons(nominal_test, th_001),
        },
    }

    feature_errors = {
        "validation": sum(1 for r in val_records if r.get("status") == "feature_error"),
        "calibration": sum(1 for r in cal_records if r.get("status") == "feature_error"),
        "test": sum(1 for r in test_records if r.get("status") == "feature_error"),
    }

    runner_code_files = [
        "scripts/xvs/integrity/run_wp07_g0.py",
        "scripts/xvs/integrity/make_wp07_data.py",
        "src/xvs/integrity/monitor.py",
        "src/xvs/integrity/conformal.py",
        "src/xvs/integrity/faults.py",
        "configs/xvs/integrity/g0.json",
    ]
    runner_code_hashes = {
        rel_p: hashlib.sha256((REPO_ROOT / rel_p).read_bytes()).hexdigest()
        for rel_p in runner_code_files
        if (REPO_ROOT / rel_p).is_file()
    }
    git_status_end = [
        ln for ln in subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.splitlines()
        if ln.strip()
    ]

    val_floor_pct = achieved_far["far_0.001"]["validation"]["floor_analysis"]["status_not_converged_rate"] * 100.0
    test_floor_pct = achieved_far["far_0.001"]["test"]["floor_analysis"]["status_not_converged_rate"] * 100.0
    deviations_list = [
        {
            "id": "DEV-WP07-FAR-FLOOR",
            "title": "FAR 0.1% Target Floor (`FAIL`)",
            "val_floor_pct": float(val_floor_pct),
            "test_floor_pct": float(test_floor_pct),
            "val_floor_k": int(achieved_far["far_0.001"]["validation"]["floor_analysis"]["status_not_converged_k"]),
            "test_floor_k": int(achieved_far["far_0.001"]["test"]["floor_analysis"]["status_not_converged_k"]),
        },
        {
            "id": "DEV-WP07-PILOT-BOUNDS",
            "title": "Benchmark and execution bounds",
        },
        {
            "id": "DEV-WP07-MANIFEST-RESTORATION",
            "title": "Manifest provenance restoration (F4 / F10)",
            "hand_edit_date": "2026-10-20",
            "hand_edit_time": "17:37",
            "restored_date": "2026-11-04",
            "fields_not_recorded_at_generation": ["git status", "configs read", "start/end hash sets"],
            "summary": (
                "the manifest was hand-edited on 2026-10-20 at 17:37, restored on 2026-11-04; "
                "fields not recorded at generation: git status, configs read, start/end hash sets."
            ),
        },
    ]

    all_results = {
        "thresholds": configs_data,
        "scale_comparison": scale_comp,
        "calibration": {
            "far_0.01": conformal_01.to_dict(),
            "far_0.001": conformal_001.to_dict(),
        },
        "evaluation": {
            "far_0.01": metrics_01,
            "far_0.001": metrics_001,
        },
        "achieved_far": achieved_far,
        "feature_errors": feature_errors,
        "deviations": deviations_list,
        "provenance": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commit": head_commit,
            "argv": list(argv) if argv is not None else list(sys.argv),
            "git_status": git_status_end,
            "code_hashes": runner_code_hashes,
            "manifest_commit": manifest_commit,
            "manifest_code_hashes_verified": bool(manifest_commit is not None),
            "validation_file": str(val_file),
            "calibration_file": str(cal_file),
            "test_file": str(test_file),
            "validation_records_read": len(val_records),
            "calibration_records_read": len(cal_records),
            "test_records_read": len(test_records),
        },
        "n_samples": {
            "validation_total": len(val_records),
            "calibration_total": len(cal_records),
            "test_total": len(test_records),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        f.write("\n")
    print(f"\nSaved full evaluation results to {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
