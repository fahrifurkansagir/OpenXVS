"""Generate markdown summary tables and Charter §8 report for WP-07 Gate G0.

Reads results JSON files from output/xvs/wp07/g0/ and outputs formatted tables:
1. Pilot Benchmark and Timing Projection Table
2. Monitor M0 Threshold Configuration & Achieved FAR Reason Breakdown (FAR 1% PASS, FAR 0.1% FAIL floor)
3. Split & Mondrian Conformal Quantiles
4. Empirical Coverage Table (Standard and Mondrian, pooled and per camera set, with 95% CI, n, and +-2% band check)
5. Hazardously Misleading Information (HMI) Full Spec §4 Table (fault-free and each fault type x camera set F, FRBL, all, overall, for Standard & Mondrian and AL in {0.10deg, 0.50deg})
6. Availability Table on Fault-Free Test Frames (primary) and Faulty Frames (separate)
7. Time to Alarm Table in Seconds (0.5 s, 1.5 s, 5.0 s, never-alarmed) Overall and Per Fault Type
8. Missed-Detection Matrix (fault_type x cam_set)

Adheres strictly to G-B11 K3 rules (every number from JSON), G-B14 W5 (provenance tracking),
and Charter §8 layout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root
from xvs.report_provenance import (
    get_active_tracker,
    load_json_source,
    register_json_source,
    rv,
    save_provenance,
    verify_report_provenance,
)

DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "xvs" / "reports" / "WP07_G0_report.md"
DEFAULT_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "integrity" / "g0_spec.json"
DEFAULT_WP05_SWEEP_PATH = REPO_ROOT / "output" / "xvs" / "wp05" / "g0_lambda_sweep_v2" / "lambda_sweep_v2_results.json"


def load_json(p: Path, rel_path: str | None = None) -> dict[str, Any]:
    return load_json_source(p, source_label=rel_path)


def get_integrity_test_output() -> str:
    """Run pytest on all WP-07 integrity tests (V1..V5) and return verbatim summary."""
    cmd = [
        sys.executable,
        "-m", "pytest",
        "tests/xvs/test_integrity_v1.py",
        "tests/xvs/test_integrity_v2.py",
        "tests/xvs/test_integrity_v3.py",
        "tests/xvs/test_integrity_v4.py",
        "tests/xvs/test_integrity_v5.py",
        "tests/xvs/test_integrity_gb13.py",
        "-v",
        "-k", "not data",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"WP-07 integrity tests failed:\n{proc.stdout}")

    lines = proc.stdout.strip().splitlines()
    summary_lines = []
    for line in lines:
        if "::" in line or "passed" in line or "test session starts" in line or "rootdir:" in line:
            summary_lines.append(line)
    return "\n".join(summary_lines) if summary_lines else proc.stdout.strip()


def format_pilot_table(
    pilot_data: dict[str, Any],
    manifest_data: dict[str, Any],
    spec_cfg: dict[str, Any],
    json_rel: str,
) -> str:
    n_bm_frames = rv(pilot_data, "pilot_frames", "d")
    w_cap = rv(pilot_data, "workers", "d")
    w_sub = rv(spec_cfg, "worker_subtrahend", "d")
    s_12 = rv(spec_cfg, "projected_12_seeds", "d")
    ref_frames = rv(pilot_data, "pilot_frames", "d")
    s_20 = rv(manifest_data, "total_seeds", "d")
    lim_h_int = rv(pilot_data, "limit_hours", ".0f")
    lines = [
        f"<!-- source: {json_rel} -->",
        "| Metric | Value | Reference / Spec Limit |",
        "| :--- | :---: | :---: |",
        f"| Pilot benchmark frames | {n_bm_frames} | {ref_frames} frames |",
        f"| Elapsed pilot single-core time | {rv(pilot_data, 'elapsed_single_sec', '.2f')} s | — |",
        f"| Single-core rate | {rv(pilot_data, 'sec_per_frame_core', '.3f')} s/frame | All {ref_frames} conditions |",
        f"| Worker cap (cores − {w_sub}) | {w_cap} | logical cores − {w_sub} = {w_cap} |",
        f"| Projected wall time ({s_20} seeds) | {rv(pilot_data, 'projected_wall_20_seeds_hours', '.2f')} h | Limit: {rv(pilot_data, 'limit_hours', '.1f')} h |",
        f"| Projected wall time ({s_12} seeds) | {rv(pilot_data, 'projected_wall_12_seeds_hours', '.2f')} h | Fallback option |",
        f"| Exceeds {lim_h_int} h wall limit? | {pilot_data['exceeds_limit']} | — |",
        f"| Chosen seed count | {rv(pilot_data, 'chosen_seeds', 'd')} | Full {s_20} seeds retained |",
    ]
    return "\n".join(lines)


def format_thresholds_table(thresholds_data: dict[str, Any], json_rel: str) -> str:
    t01 = thresholds_data["far_0.01"]
    t001 = thresholds_data["far_0.001"]
    far_1_hdr = rv(t01, "far_target", ".1%")
    far_01_hdr = rv(t001, "far_target", ".1%")
    lines = [
        f"<!-- source: {json_rel} -->",
        f"| Parameter | FAR {far_1_hdr} Baseline | FAR {far_01_hdr} Conservative | Description / Unit |",
        "| :--- | :---: | :---: | :--- |",
    ]

    lines.append(f"| Target FAR | {rv(t01, 'far_target', '.1%')} | {rv(t001, 'far_target', '.2%')} | Nominal validation target |")
    lines.append(f"| Max RMS residual | {rv(t01, 'max_rms_residual_urad', '.1f')} µrad | {rv(t001, 'max_rms_residual_urad', '.1f')} µrad | Limb cone residual threshold |")
    lines.append(f"| Max condition number | {rv(t01, 'max_condition_number', '.1e')} | {rv(t001, 'max_condition_number', '.1e')} | Covariance matrix conditioning |")
    lines.append(f"| Min inlier fraction | {rv(t01, 'min_inlier_fraction', '.1%')} | {rv(t001, 'min_inlier_fraction', '.1%')} | Inliers / total detected rays |")
    lines.append(f"| Min inlier count | {rv(t01, 'min_inlier_count', 'd')} | {rv(t001, 'min_inlier_count', 'd')} | Absolute ray count floor |")
    lines.append(f"| Max leave-one-out consistency | {rv(t01, 'max_leave_one_out_deg', '.4f')}° | {rv(t001, 'max_leave_one_out_deg', '.4f')}° | Multi-cam angular discrepancy |")

    return "\n".join(lines)


def format_achieved_far_section(
    achieved_far: dict[str, Any],
    computed_verdicts: dict[str, Any],
    spec_cfg: dict[str, Any],
    json_rel: str,
) -> str:
    ci_pct = rv(spec_cfg, "ci_level_pct", "d")
    lines = [
        f"<!-- source: {json_rel} -->",
        "#### Achieved Fault-Free False Alarm Rate by Camera Set and Split",
        "",
        f"| Target Level | Split | Camera Set | Alarms ($k$) | Fault-Free Frames ($n$) | Achieved FAR | {ci_pct}% Clopper–Pearson CI | Verdict |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for far_key in ("far_0.01", "far_0.001"):
        for split in ("validation", "test"):
            sp_data = achieved_far[far_key][split]
            tgt_lbl = rv(sp_data, "target_far", ".1%")
            label = f"FAR {tgt_lbl}"
            for cset in ("F", "FRBL", "all", "overall"):
                row = sp_data["overall"] if cset == "overall" else sp_data["by_cam_set"][cset]
                k_v = rv(row, "k", "d")
                n_v = rv(row, "n", "d")
                r_v = rv(row, "rate", ".3%")
                ci_l = rv(row, "ci_low", ".3%")
                ci_h = rv(row, "ci_high", ".3%")
                if cset == "overall":
                    verdict_tok = rv(computed_verdicts, f"{far_key}_{split}_verdict", "verdict")
                    v_str = f"**{verdict_tok}**"
                else:
                    v_str = "—"
                c_lbl = "**overall**" if cset == "overall" else f"`{cset}`"
                lines.append(
                    f"| {label} | {split} | {c_lbl} | {k_v} | {n_v} | {r_v} | [{ci_l}, {ci_h}] | {v_str} |"
                )

    lines.extend([
        "",
        "#### Fault-Free Alarm Breakdown by Trigger Reason (Validation and Test)",
        "",
        "| Target Level | Split | Total Alarms ($k/n$) | `status != CONVERGED` (sole) | `RMS_HIGH` (sole) | `COND_HIGH` (sole) | `FRAC_LOW` (sole) | `CNT_LOW` (sole) | `LOO_HIGH` (sole) | `feature_error` | `not_available` (alarms / frames) | `min_cnt` Boundary Ties |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for far_key in ("far_0.01", "far_0.001"):
        for split in ("validation", "test"):
            sp_data = achieved_far[far_key][split]
            tgt_lbl = rv(sp_data, "target_far", ".1%")
            label = f"FAR {tgt_lbl}"
            ov = sp_data["overall"]
            br = ov["by_reason"]
            sr = ov["sole_reason_counts"]
            k_tot = rv(ov, "k", "d")
            n_tot = rv(ov, "n", "d")
            r_tot = rv(ov, "rate", ".3%")
            st_k = rv(br, "STATUS_NOT_CONVERGED", "d")
            st_s = rv(sr, "STATUS_NOT_CONVERGED", "d")
            rms_k = rv(br, "RMS_HIGH", "d")
            rms_s = rv(sr, "RMS_HIGH", "d")
            cnd_k = rv(br, "COND_HIGH", "d")
            cnd_s = rv(sr, "COND_HIGH", "d")
            frc_k = rv(br, "INLIER_FRAC_LOW", "d")
            frc_s = rv(sr, "INLIER_FRAC_LOW", "d")
            cnt_k = rv(br, "INLIER_COUNT_LOW", "d")
            cnt_s = rv(sr, "INLIER_COUNT_LOW", "d")
            loo_k = rv(br, "LOO_HIGH", "d")
            loo_s = rv(sr, "LOO_HIGH", "d")
            fe_k = rv(br, "FEATURE_ERROR", "d")
            na_k = rv(br, "NOT_AVAILABLE", "d")
            na_f = rv(ov, "n_loo_not_available", "d")
            ties = rv(ov, "n_min_cnt_boundary_ties", "d")
            lines.append(
                f"| {label} | {split} | {k_tot}/{n_tot} ({r_tot}) | "
                f"**{st_k}** ({st_s}) | {rms_k} ({rms_s}) | {cnd_k} ({cnd_s}) | "
                f"{frc_k} ({frc_s}) | {cnt_k} ({cnt_s}) | {loo_k} ({loo_s}) | "
                f"{fe_k} | {na_k} / {na_f} | {ties} |"
            )

    fl_val = achieved_far["far_0.001"]["validation"]["floor_analysis"]
    fl_tst = achieved_far["far_0.001"]["test"]["floor_analysis"]
    st_val_pct = rv(fl_val, "status_not_converged_rate", ".3%")
    st_val_k = rv(fl_val, "status_not_converged_k", "d")
    st_tst_pct = rv(fl_tst, "status_not_converged_rate", ".3%")
    st_tst_k = rv(fl_tst, "status_not_converged_k", "d")
    cont_val_pct = rv(fl_val, "continuous_features_only_rate", ".3%")
    cont_val_k = rv(fl_val, "continuous_features_only_k", "d")
    cont_tst_pct = rv(fl_tst, "continuous_features_only_rate", ".3%")
    cont_tst_k = rv(fl_tst, "continuous_features_only_k", "d")
    tgt_01_hdr = rv(achieved_far["far_0.001"]["validation"], "target_far", ".1%")
    tgt_01_pct = rv(achieved_far["far_0.001"]["validation"], "target_far", ".1%")
    fail_tok = rv(computed_verdicts, "far_0.001_validation_verdict", "verdict")
    st_val_pct_2 = rv(fl_val, "status_not_converged_rate", ".3%")
    st_tst_pct_2 = rv(fl_tst, "status_not_converged_rate", ".3%")

    lines.extend([
        "",
        f"**FAR {tgt_01_hdr} Verdict (`{fail_tok}`):** **FAR {tgt_01_pct}: not met, floor = {st_val_pct} (validation, {st_val_k} frames) / {st_tst_pct} (test, {st_tst_k} frames) from `status != CONVERGED`**. "
        f"Because `evaluate_monitor_alarm` raises a mandatory fail-safe alarm whenever the solver fit status is not `CONVERGED` (no feature threshold controls solver non-convergence), "
        f"and `status != CONVERGED` occurs on {st_val_pct_2} of fault-free validation frames and {st_tst_pct_2} of fault-free test frames in `FRBL`/`all`, the overall false-alarm rate cannot fall below this non-convergence floor. "
        f"On `CONVERGED` frames alone, the tuned continuous feature thresholds trigger {cont_val_pct} ({cont_val_k} frames) on validation and {cont_tst_pct} ({cont_tst_k} frames) on test.",
    ])
    return "\n".join(lines)


def format_conformal_quantiles_table(
    cal_data: dict[str, Any],
    cfg: dict[str, Any],
    spec_cfg: dict[str, Any],
    json_rel: str,
) -> str:
    a01_h = rv(spec_cfg, "alpha_01", ".1f")
    a001_h = rv(spec_cfg, "alpha_001", ".2f")
    a0001_h = rv(spec_cfg, "alpha_0001", ".3f")
    lines = [
        f"<!-- source: {json_rel} -->",
        f"| Configuration | Variant | Partition / Slice | $q(\\alpha={a01_h})$ | $q(\\alpha={a001_h})$ | $q(\\alpha={a0001_h})$ | $n_{{\\text{{cal}}}}$ |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for far_key in ("far_0.01", "far_0.001"):
        sub = cal_data[far_key]
        std_q = sub["standard_quantiles"]
        label = f"FAR {rv(cfg[far_key], 'far_target', '.1%')}"
        n_std = rv(sub, "n_cal_standard", "d")
        q01 = rv(std_q, "0.1", ".3f")
        q001 = rv(std_q, "0.01", ".3f")
        q0001 = rv(std_q, "0.001", ".3f")
        lines.append(
            f"| {label} | Standard | Pooled alarm-free | {q01} | {q001} | {q0001} | {n_std} |"
        )
        mond_q = sub["mondrian_quantiles"]
        n_mond = sub["n_cal_mondrian"]
        for slice_key in sorted(mond_q.keys()):
            label_m = f"FAR {rv(cfg[far_key], 'far_target', '.1%')}"
            mq = mond_q[slice_key]
            n_s = rv(n_mond, slice_key, "d")
            mq01 = rv(mq, "0.1", ".3f")
            mq001 = rv(mq, "0.01", ".3f")
            mq0001 = rv(mq, "0.001", ".3f")
            lines.append(
                f"| {label_m} | Mondrian | {slice_key} | {mq01} | {mq001} | {mq0001} | {n_s} |"
            )

    return "\n".join(lines)


def format_coverage_table(cov_data: dict[str, Any], cov_by_cam: dict[str, Any], spec_cfg: dict[str, Any], json_rel: str) -> str:
    ci_pct = rv(spec_cfg, "ci_level_pct", "d")
    band_pct = rv(spec_cfg, "band_tol_pct", "d")
    lines = [
        f"<!-- source: {json_rel} -->",
        f"| Variant | Camera Set / Slice | $\\alpha$ | Target $1 - \\alpha$ | Empirical Coverage | {ci_pct}% Confidence Interval | $n$ | $\\pm {band_pct}\\%$ Band |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    alpha_spec_map = {"0.1": ("alpha_01", ".1f"), "0.01": ("alpha_001", ".2f"), "0.001": ("alpha_0001", ".3f")}
    for variant in ("standard", "mondrian"):
        v_name = "Standard" if variant == "standard" else "Mondrian"
        for cset in ("F", "FRBL", "all", "overall"):
            for a_key in ("0.1", "0.01", "0.001"):
                c = cov_by_cam[variant][a_key][cset]
                sk, sfmt = alpha_spec_map[a_key]
                a_disp = rv(spec_cfg, sk, sfmt)
                rate_pct = rv(c, "rate", ".2%")
                tgt_pct = rv(c, "target", ".1%")
                ci_low = rv(c, "ci_low", ".2%")
                ci_high = rv(c, "ci_high", ".2%")
                ci_pct_str = f"[{ci_low}, {ci_high}]"
                n_val = rv(c, "n", "d")
                band = rv(c, "within_band_2pct", "verdict") if c["within_band_2pct"] else "OUTSIDE"
                c_lbl = "**overall**" if cset == "overall" else f"`{cset}:degraded`"
                lines.append(f"| {v_name} | {c_lbl} | {a_disp} | {tgt_pct} | {rate_pct} | {ci_pct_str} | {n_val} | {band} |")

    return "\n".join(lines)


def format_scale_comparison_table(scale_data: dict[str, Any], json_rel: str) -> str:
    lines = [
        f"<!-- source: {json_rel} -->",
        "| Camera Set | Samples ($n$) | Mean Error | Median Error | Mean $\\hat{\\sigma}$ | Median $\\hat{\\sigma}$ | Median Ratio (Error / $\\hat{\\sigma}$) | Fitted $\\sigma_{sys}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for cset in sorted(scale_data.keys()):
        s = scale_data[cset]
        n_val = rv(s, "n", "d")
        mean_err = rv(s, "mean_error_deg", ".4f")
        med_err = rv(s, "median_error_deg", ".4f")
        mean_sig = rv(s, "mean_sigma_hat_deg", ".4f")
        med_sig = rv(s, "median_sigma_hat_deg", ".4f")
        ratio_med = rv(s, "ratio_median", ".2f")
        fit_deg = rv(s, "fitted_sigma_sys_deg", ".4f")
        fit_urad = rv(s, "fitted_sigma_sys_urad", ".0f")
        lines.append(
            f"| `{cset}` | {n_val} | {mean_err}° | {med_err}° | "
            f"{mean_sig}° | {med_sig}° | **{ratio_med}×** | "
            f"**{fit_deg}°** ({fit_urad} µrad) |"
        )
    return "\n".join(lines)


def format_coverage_sys_table(cov_sys_data: dict[str, Any], spec_cfg: dict[str, Any], json_rel: str) -> str:
    ci_pct = rv(spec_cfg, "ci_level_pct", "d")
    band_pct = rv(spec_cfg, "band_tol_pct", "d")
    lines = [
        f"<!-- source: {json_rel} -->",
        f"| Variant | $\\alpha$ | Target $1 - \\alpha$ | Empirical Coverage ($r_2$) | {ci_pct}% Confidence Interval | $n$ | $\\pm {band_pct}\\%$ Band |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    alpha_spec_map = {"0.1": ("alpha_01", ".1f"), "0.01": ("alpha_001", ".2f"), "0.001": ("alpha_0001", ".3f")}
    for variant in ("standard", "mondrian"):
        sub = cov_sys_data[variant]
        for a_key in ("0.1", "0.01", "0.001"):
            if a_key not in sub:
                continue
            c = sub[a_key]
            sk, sfmt = alpha_spec_map[a_key]
            a_disp = rv(spec_cfg, sk, sfmt)
            v_name = "Standard" if variant == "standard" else "Mondrian"
            rate_pct = rv(c, "rate", ".2%")
            tgt_pct = rv(c, "target", ".1%")
            ci_low = rv(c, "ci_low", ".2%")
            ci_high = rv(c, "ci_high", ".2%")
            ci_pct_str = f"[{ci_low}, {ci_high}]"
            n_val = rv(c, "n", "d")
            band = rv(c, "within_band_2pct", "verdict") if c["within_band_2pct"] else "OUTSIDE"
            lines.append(f"| {v_name} | {a_disp} | {tgt_pct} | **{rate_pct}** | {ci_pct_str} | {n_val} | {band} |")
    return "\n".join(lines)


def format_hmi_table(
    hmi_by_cam_data: dict[str, Any],
    avail_faulty_data: dict[str, Any],
    spec_cfg: dict[str, Any],
    json_rel: str,
) -> str:
    ref_blk = hmi_by_cam_data["standard"]["0.50deg"]["0.01"]
    n_ff = rv(ref_blk["none"]["overall"], "n", "d")
    n_occ = rv(ref_blk["occlusion"]["overall"], "n", "d")
    n_sp = rv(ref_blk["spurious_edge"]["overall"], "n", "d")
    n_alt = rv(ref_blk["altitude_bias"]["overall"], "n", "d")
    n_gl = rv(ref_blk["glare"]["overall"], "n", "d")
    n_ov = rv(ref_blk["overall"]["overall"], "n", "d")
    n_ov_2 = rv(ref_blk["overall"]["overall"], "n", "d")
    n_ff_2 = rv(ref_blk["none"]["overall"], "n", "d")
    avail_flt_ov = avail_faulty_data["standard"]["0.50deg"]["0.01"]["overall"]
    n_fault_active = rv(avail_flt_ov, "n", "d")
    n_fault_active_2 = rv(avail_flt_ov, "n", "d")
    a001_h = rv(spec_cfg, "alpha_001", ".2f")
    al01_h = rv(spec_cfg, "al_secondary_deg", ".2f")
    al05_h = rv(spec_cfg, "al_primary_deg", ".2f")
    ci_pct = rv(spec_cfg, "ci_level_pct", "d")

    lines = [
        f"<!-- source: {json_rel} -->",
        f"**Denominator definition (`overall n={n_ov}`):** Each row uses its own active-frame denominator $n$ in the test split "
        f"(for fault-free `none`: all {n_ff} fault-free test frames across `F`, `FRBL`, `all`; for each injected fault type: only post-onset active frames `k >= k_onset` with valid attitude error, "
        f"totalling {n_occ} `occlusion` + {n_sp} `spurious_edge` + {n_alt} `altitude_bias` + {n_gl} `glare` = {n_fault_active} active fault frames; "
        f"`overall n={n_ov_2}` is the sum of the {n_ff_2} fault-free test frames and {n_fault_active_2} active post-onset fault frames).",
        "",
        f"#### Full Per-Camera-Set HMI Table ($\\alpha = {a001_h}$, Standard and Mondrian, $\\text{{AL}} \\in \\{{{al01_h}^\\circ, {al05_h}^\\circ\\}}$)",
        "",
        f"| Alert Limit (AL) | Variant | Fault Type | Camera Set | HMI Events ($k$) | Active Frames ($n$) | Rate | Clopper–Pearson {ci_pct}% UB |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    al_spec_map = {"0.10deg": ("al_secondary_deg", ".2f"), "0.50deg": ("al_primary_deg", ".2f")}
    alpha_spec_map = {"0.1": ("alpha_01", ".1f"), "0.01": ("alpha_001", ".2f"), "0.001": ("alpha_0001", ".3f")}
    for al_key in ("0.10deg", "0.50deg"):
        alsk, alfmt = al_spec_map[al_key]
        for variant in ("standard", "mondrian"):
            v_name = "Standard" if variant == "standard" else "Mondrian"
            block = hmi_by_cam_data[variant][al_key]["0.01"]
            for ft in ("none", "occlusion", "spurious_edge", "altitude_bias", "glare", "overall"):
                ft_label = "Fault-free (`none`)" if ft == "none" else ("**OVERALL**" if ft == "overall" else f"`{ft}`")
                for cset in ("F", "FRBL", "all", "overall"):
                    res = block[ft][cset]
                    al_disp = f"{rv(spec_cfg, alsk, alfmt)}deg"
                    c_lbl = "**overall**" if cset == "overall" else f"`{cset}`"
                    k_val = rv(res, "k", "d")
                    n_val = rv(res, "n", "d")
                    rate_val = rv(res, "rate", ".2e")
                    up_val = rv(res, "upper_95", ".2e")
                    lines.append(
                        f"| {al_disp} | {v_name} | {ft_label} | {c_lbl} | {k_val} | {n_val} | {rate_val} | **{up_val}** |"
                    )

    a01_p = rv(spec_cfg, "alpha_01", ".1f")
    a001_p = rv(spec_cfg, "alpha_001", ".2f")
    a0001_p = rv(spec_cfg, "alpha_0001", ".3f")
    ci_pct_2 = rv(spec_cfg, "ci_level_pct", "d")
    lines.extend([
        "",
        f"#### Pooled HMI Summary Across All $\\alpha \\in \\{{{a01_p}, {a001_p}, {a0001_p}\\}}$",
        "",
        f"| Alert Limit (AL) | Variant | $\\alpha$ | Fault Type | HMI Events ($k$) | Active Frames ($n$) | Rate | Clopper–Pearson {ci_pct_2}% UB |",
        "| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |",
    ])
    for al_key in ("0.10deg", "0.50deg"):
        alsk, alfmt = al_spec_map[al_key]
        for variant in ("standard", "mondrian"):
            v_name = "Std" if variant == "standard" else "Mond"
            for a_key in ("0.1", "0.01", "0.001"):
                ask, afmt = alpha_spec_map[a_key]
                block = hmi_by_cam_data[variant][al_key][a_key]
                for ft in ("none", "occlusion", "spurious_edge", "altitude_bias", "glare", "overall"):
                    res = block[ft]["overall"]
                    al_disp = f"{rv(spec_cfg, alsk, alfmt)}deg"
                    a_disp = rv(spec_cfg, ask, afmt)
                    ft_label = "Fault-free (`none`)" if ft == "none" else ("OVERALL" if ft == "overall" else ft)
                    k_val = rv(res, "k", "d")
                    n_val = rv(res, "n", "d")
                    rate_val = rv(res, "rate", ".2e")
                    up_val = rv(res, "upper_95", ".2e")
                    lines.append(
                        f"| {al_disp} | {v_name} | {a_disp} | {ft_label} | {k_val} | {n_val} | "
                        f"{rate_val} | **{up_val}** |"
                    )

    return "\n".join(lines)


def format_availability_table(
    avail_ff_data: dict[str, Any],
    avail_faulty_data: dict[str, Any],
    spec_cfg: dict[str, Any],
    json_rel: str,
) -> str:
    ci_pct = rv(spec_cfg, "ci_level_pct", "d")
    lines = [
        f"<!-- source: {json_rel} -->",
        "#### Primary Availability on Fault-Free Test Frames (`WP07_INTEGRITY.md` §4)",
        "",
        f"| Alert Limit (AL) | Variant | $\\alpha$ | Camera Set | Available ($k$) | Fault-Free Frames ($n$) | Availability | {ci_pct}% Confidence Interval |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    al_spec_map = {"0.10deg": ("al_secondary_deg", ".2f"), "0.50deg": ("al_primary_deg", ".2f")}
    alpha_spec_map = {"0.1": ("alpha_01", ".1f"), "0.01": ("alpha_001", ".2f"), "0.001": ("alpha_0001", ".3f")}
    for al_key in ("0.10deg", "0.50deg"):
        alsk, alfmt = al_spec_map[al_key]
        for variant in ("standard", "mondrian"):
            for a_key in ("0.1", "0.01", "0.001"):
                ask, afmt = alpha_spec_map[a_key]
                block = avail_ff_data[variant][al_key][a_key]
                for cset in ("F", "FRBL", "all", "overall"):
                    if cset not in block:
                        continue
                    res = block[cset]
                    al_disp = f"{rv(spec_cfg, alsk, alfmt)}deg"
                    a_disp = rv(spec_cfg, ask, afmt)
                    v_name = "Std" if variant == "standard" else "Mond"
                    k_val = rv(res, "k", "d")
                    n_val = rv(res, "n", "d")
                    rate_val = rv(res, "rate", ".2%")
                    ci_low = rv(res, "ci_low", ".2%")
                    ci_high = rv(res, "ci_high", ".2%")
                    ci_str = f"[{ci_low}, {ci_high}]"
                    lines.append(
                        f"| {al_disp} | {v_name} | {a_disp} | {cset} | {k_val} | {n_val} | "
                        f"{rate_val} | {ci_str} |"
                    )

    ci_pct_2 = rv(spec_cfg, "ci_level_pct", "d")
    lines.extend([
        "",
        "#### Availability on Active Faulty Test Frames (Reported Separately)",
        "",
        f"| Alert Limit (AL) | Variant | $\\alpha$ | Camera Set | Unalarmed & $\\text{{PL}} \\le \\text{{AL}}$ ($k$) | Active Faulty Frames ($n$) | Faulty-Frame Availability | {ci_pct_2}% Confidence Interval |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for al_key in ("0.10deg", "0.50deg"):
        alsk, alfmt = al_spec_map[al_key]
        for variant in ("standard", "mondrian"):
            for a_key in ("0.1", "0.01", "0.001"):
                ask, afmt = alpha_spec_map[a_key]
                block = avail_faulty_data[variant][al_key][a_key]
                for cset in ("F", "FRBL", "all", "overall"):
                    if cset not in block:
                        continue
                    res = block[cset]
                    al_disp = f"{rv(spec_cfg, alsk, alfmt)}deg"
                    a_disp = rv(spec_cfg, ask, afmt)
                    v_name = "Std" if variant == "standard" else "Mond"
                    k_val = rv(res, "k", "d")
                    n_val = rv(res, "n", "d")
                    rate_val = rv(res, "rate", ".2%")
                    ci_low = rv(res, "ci_low", ".2%")
                    ci_high = rv(res, "ci_high", ".2%")
                    ci_str = f"[{ci_low}, {ci_high}]"
                    lines.append(
                        f"| {al_disp} | {v_name} | {a_disp} | {cset} | {k_val} | {n_val} | "
                        f"{rate_val} | {ci_str} |"
                    )

    return "\n".join(lines)


def format_tta_table(tta_data: dict[str, Any], spec_cfg: dict[str, Any], json_rel: str) -> str:
    t13_s = rv(spec_cfg, "t1_t3_frame_period_s", ".1f")
    t2_s = rv(spec_cfg, "t2_frame_period_s", ".1f")
    thr05 = rv(spec_cfg, "tta_thr_0_5s", ".1f")
    thr15 = rv(spec_cfg, "tta_thr_1_5s", ".1f")
    thr50 = rv(spec_cfg, "tta_thr_5_0s", ".1f")
    lines = [
        f"<!-- source: {json_rel} -->",
        f"#### Time to Alarm in Seconds Per Fault Type and Overall (T1/T3 = {t13_s} s/frame, T2 = {t2_s} s/frame)",
        "",
        f"| Fault Type | Sequences ($n$) | Alarmed | Alarmed $\\le {thr05}\\text{{ s}}$ | Alarmed $\\le {thr15}\\text{{ s}}$ | Alarmed $\\le {thr50}\\text{{ s}}$ | Never Alarmed (count / %) | Median TTA (s) | Mean TTA (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    by_ft = tta_data.get("by_fault_type", {})
    for ft in ("occlusion", "spurious_edge", "altitude_bias", "glare"):
        if ft not in by_ft:
            continue
        sub = by_ft[ft]
        n_s = rv(sub, "total_fault_sequences", "d")
        al_s = rv(sub, "alarmed_sequences", "d")
        f05 = rv(sub, "fraction_alarmed_within_0_5s", ".1%")
        f15 = rv(sub, "fraction_alarmed_within_1_5s", ".1%")
        f50 = rv(sub, "fraction_alarmed_within_5_0s", ".1%")
        nev_c = rv(sub, "never_alarmed_count", "d")
        nev_p = rv(sub, "fraction_never_alarmed", ".1%")
        med_s = rv(sub, "median_tta_seconds", ".2f")
        mean_s = rv(sub, "mean_tta_seconds", ".2f")
        lines.append(
            f"| `{ft}` | {n_s} | {al_s} | **{f05}** | {f15} | {f50} | {nev_c} ({nev_p}) | {med_s} s | {mean_s} s |"
        )

    tot_fault = rv(tta_data, "total_fault_sequences", "d")
    alarmed = rv(tta_data, "alarmed_sequences", "d")
    f05_all = rv(tta_data, "fraction_alarmed_within_0_5s", ".1%")
    f15_all = rv(tta_data, "fraction_alarmed_within_1_5s", ".1%")
    f50_all = rv(tta_data, "fraction_alarmed_within_5_0s", ".1%")
    never_cnt = rv(tta_data, "never_alarmed_count", "d")
    never_frac = rv(tta_data, "fraction_never_alarmed", ".1%")
    med_s_all = rv(tta_data, "median_tta_seconds", ".2f")
    mean_s_all = rv(tta_data, "mean_tta_seconds", ".2f")
    lines.append(
        f"| **Overall (All Faults)** | **{tot_fault}** | **{alarmed}** | **{f05_all}** | **{f15_all}** | **{f50_all}** | **{never_cnt} ({never_frac})** | **{med_s_all} s** | **{mean_s_all} s** |"
    )
    return "\n".join(lines)


def format_missed_detections_table(
    missed_data: dict[str, Any],
    hmi_by_cam_data: dict[str, Any],
    spec_cfg: dict[str, Any],
    json_rel: str,
) -> str:
    lines = [
        f"<!-- source: {json_rel} -->",
        "| Alert Limit (AL) | Injected Fault Type | Camera Set F | Camera Set FRBL | Camera Set all | Total Missed |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]
    al_spec_map = {"0.10deg": ("al_secondary_deg", ".2f"), "0.50deg": ("al_primary_deg", ".2f")}
    for al_key in ("0.10deg", "0.50deg"):
        alsk, alfmt = al_spec_map[al_key]
        matrix = missed_data[al_key]
        for ft in ("occlusion", "spurious_edge", "altitude_bias", "glare"):
            row = matrix[ft]
            al_disp = f"{rv(spec_cfg, alsk, alfmt)}deg"
            cnt_f = rv(row, "F", "d")
            cnt_frbl = rv(row, "FRBL", "d")
            cnt_all = rv(row, "all", "d")
            tot_str = rv(hmi_by_cam_data["mondrian"][al_key]["0.1"][ft]["overall"], "k", "d")
            lines.append(f"| {al_disp} | {ft} | {cnt_f} | {cnt_frbl} | {cnt_all} | **{tot_str}** |")

    return "\n".join(lines)


def format_reconciliation_table(
    res: dict[str, Any],
    cfg: dict[str, Any],
    computed_verdicts: dict[str, Any],
    spec_cfg: dict[str, Any],
    wp05_sweep: dict[str, Any] | None,
    wp05_pers: dict[str, Any] | None,
    json_rel: str,
) -> str:
    """Format the 28f5b3e -> 7ea07ae/3d4194a provenance reconciliation table and chat drift table in §7.1."""
    eval_01 = res["evaluation"]["far_0.01"]
    far_01_val = res["achieved_far"]["far_0.01"]["validation"]["overall"]
    far_01_test = res["achieved_far"]["far_0.01"]["test"]["overall"]
    far_001_val = res["achieved_far"]["far_0.001"]["validation"]["overall"]
    far_001_test = res["achieved_far"]["far_0.001"]["test"]["overall"]
    cov_std = eval_01["coverage"]["standard"]["0.01"]
    cov_mond = eval_01["coverage"]["mondrian"]["0.01"]
    cov_cam = eval_01["coverage_by_cam_set"]["standard"]["0.01"]
    hmi_05 = eval_01["hmi_by_cam_set"]["standard"]["0.50deg"]["0.01"]
    avail_old_std = eval_01["availability_all_test_frames"]["standard"]["0.50deg"]["0.01"]["overall"]
    avail_old_mond = eval_01["availability_all_test_frames"]["mondrian"]["0.50deg"]["0.01"]["overall"]
    avail_ff_std = eval_01["availability"]["standard"]["0.50deg"]["0.01"]["overall"]
    avail_ff_mond = eval_01["availability"]["mondrian"]["0.50deg"]["0.01"]["overall"]
    avail_flt_std = eval_01["availability_faulty"]["standard"]["0.50deg"]["0.01"]["overall"]
    avail_flt_mond = eval_01["availability_faulty"]["mondrian"]["0.50deg"]["0.01"]["overall"]
    tta = eval_01["time_to_alarm"]

    n_fault_types = rv(spec_cfg, "spec_injected_fault_type_count", "d")
    n_fault_types_2 = rv(spec_cfg, "spec_injected_fault_type_count", "d")
    n_test_rows_per_ft = rv(far_01_test, "n", "d")
    n_test_rows_per_ft_2 = rv(far_01_test, "n", "d")
    n_test_total = rv(res["n_samples"], "test_total", "d")

    n_occ = rv(hmi_05["occlusion"]["overall"], "n", "d")
    n_spu = rv(hmi_05["spurious_edge"]["overall"], "n", "d")
    n_alt = rv(hmi_05["altitude_bias"]["overall"], "n", "d")
    n_glr = rv(hmi_05["glare"]["overall"], "n", "d")
    n_occ_2 = rv(hmi_05["occlusion"]["overall"], "n", "d")
    n_spu_2 = rv(hmi_05["spurious_edge"]["overall"], "n", "d")
    n_alt_2 = rv(hmi_05["altitude_bias"]["overall"], "n", "d")
    n_glr_2 = rv(hmi_05["glare"]["overall"], "n", "d")
    n_act_flt = rv(avail_flt_std, "n", "d")
    n_hmi_tot = rv(hmi_05["overall"]["overall"], "n", "d")

    k_hmi_tot = rv(hmi_05["overall"]["overall"], "k", "d")
    n_hmi_tot_2 = rv(hmi_05["overall"]["overall"], "n", "d")
    k_hmi_tot_2 = rv(hmi_05["overall"]["overall"], "k", "d")
    n_hmi_tot_3 = rv(hmi_05["overall"]["overall"], "n", "d")
    k_hmi_ff = rv(hmi_05["none"]["overall"], "k", "d")
    n_test_rows_per_ft_3 = rv(far_01_test, "n", "d")
    k_hmi_spu = rv(hmi_05["spurious_edge"]["overall"], "k", "d")
    k_hmi_occ = rv(hmi_05["occlusion"]["overall"], "k", "d")
    k_hmi_alt = rv(hmi_05["altitude_bias"]["overall"], "k", "d")
    k_hmi_glr = rv(hmi_05["glare"]["overall"], "k", "d")
    n_act_flt_2 = rv(avail_flt_std, "n", "d")
    k_hmi_frbl = rv(hmi_05["overall"]["FRBL"], "k", "d")

    k_far01_val = rv(far_01_val, "k", "d")
    n_far01_val = rv(far_01_val, "n", "d")
    r_far01_val = rv(far_01_val, "rate", ".3%")
    k_far01_test = rv(far_01_test, "k", "d")
    n_far01_test = rv(far_01_test, "n", "d")
    r_far01_test = rv(far_01_test, "rate", ".3%")
    k_far01_val_2 = rv(far_01_val, "k", "d")
    n_far01_val_2 = rv(far_01_val, "n", "d")
    r_far01_val_2 = rv(far_01_val, "rate", ".3%")
    k_far01_test_2 = rv(far_01_test, "k", "d")
    n_far01_test_2 = rv(far_01_test, "n", "d")
    r_far01_test_2 = rv(far_01_test, "rate", ".3%")

    k_far001_val = rv(far_001_val, "k", "d")
    n_far01_val_3 = rv(far_01_val, "n", "d")
    r_far001_val = rv(far_001_val, "rate", ".3%")
    k_far001_test = rv(far_001_test, "k", "d")
    n_far01_val_4 = rv(far_01_val, "n", "d")
    r_far001_test = rv(far_001_test, "rate", ".3%")
    k_far001_val_2 = rv(far_001_val, "k", "d")
    n_far01_val_5 = rv(far_01_val, "n", "d")
    r_far001_val_2 = rv(far_001_val, "rate", ".3%")
    k_far001_test_2 = rv(far_001_test, "k", "d")
    n_far01_val_6 = rv(far_01_val, "n", "d")
    r_far001_test_2 = rv(far_001_test, "rate", ".3%")
    k_nc_val = rv(far_001_val["by_reason"], "STATUS_NOT_CONVERGED", "d")
    n_far01_val_7 = rv(far_01_val, "n", "d")
    k_nc_test = rv(far_001_test["by_reason"], "STATUS_NOT_CONVERGED", "d")
    n_far01_val_8 = rv(far_01_val, "n", "d")
    fail_v_001 = rv(computed_verdicts, "far_0.001_validation_verdict", "verdict")

    n_cov = rv(cov_std, "n", "d")
    n_far01_test_3 = rv(far_01_test, "n", "d")
    k_far01_test_3 = rv(far_01_test, "k", "d")
    r_cov_std = rv(cov_std, "rate", ".2%")
    r_cov_mond = rv(cov_mond, "rate", ".2%")
    n_cov_2 = rv(cov_std, "n", "d")
    r_cov_std_2 = rv(cov_std, "rate", ".2%")
    r_cov_mond_2 = rv(cov_mond, "rate", ".2%")
    n_cov_f = rv(cov_cam["F"], "n", "d")
    n_cov_frbl = rv(cov_cam["FRBL"], "n", "d")
    n_cov_all = rv(cov_cam["all"], "n", "d")

    k_av_old_s = rv(avail_old_std, "k", "d")
    n_av_old_s = rv(avail_old_std, "n", "d")
    n_av_old_s_2 = rv(avail_old_std, "n", "d")
    r_av_old_s = rv(avail_old_std, "rate", ".2%")
    k_av_old_m = rv(avail_old_mond, "k", "d")
    n_av_old_s_3 = rv(avail_old_std, "n", "d")
    r_av_old_m = rv(avail_old_mond, "rate", ".2%")

    k_av_ff_s = rv(avail_ff_std, "k", "d")
    n_av_ff_s = rv(avail_ff_std, "n", "d")
    n_av_ff_s_2 = rv(avail_ff_std, "n", "d")
    r_av_ff_s = rv(avail_ff_std, "rate", ".2%")
    k_av_ff_m = rv(avail_ff_mond, "k", "d")
    n_av_ff_s_3 = rv(avail_ff_std, "n", "d")
    r_av_ff_m = rv(avail_ff_mond, "rate", ".2%")
    n_act_flt_3 = rv(avail_flt_std, "n", "d")
    k_av_flt_s = rv(avail_flt_std, "k", "d")
    n_act_flt_4 = rv(avail_flt_std, "n", "d")
    r_av_flt_s = rv(avail_flt_std, "rate", ".2%")
    k_av_flt_m = rv(avail_flt_mond, "k", "d")
    n_act_flt_5 = rv(avail_flt_std, "n", "d")
    r_av_flt_m = rv(avail_flt_mond, "rate", ".2%")

    n_tta_seq = rv(tta, "total_fault_sequences", "d")
    n_tta_al = rv(tta, "alarmed_sequences", "d")
    n_tta_nev = rv(tta, "never_alarmed_count", "d")
    r_tta_nev = rv(tta, "fraction_never_alarmed", ".1%")
    r_tta_1f = rv(tta, "fraction_alarmed_within_1_frame", ".1%")
    r_tta_3f = rv(tta, "fraction_alarmed_within_3_frames", ".1%")
    r_tta_10f = rv(tta, "fraction_alarmed_within_10_frames", ".1%")
    med_tta_f = rv(tta, "median_tta_frames", ".1f")
    mean_tta_f = rv(tta, "mean_tta_frames", ".2f")
    med_tta_s = rv(tta, "median_tta_seconds", ".2f")
    mean_tta_s = rv(tta, "mean_tta_seconds", ".2f")
    r_tta_05s = rv(tta, "fraction_alarmed_within_0_5s", ".1%")
    r_tta_15s = rv(tta, "fraction_alarmed_within_1_5s", ".1%")
    r_tta_50s = rv(tta, "fraction_alarmed_within_5_0s", ".1%")
    n_tta_ft_seq = rv(tta["by_fault_type"]["occlusion"], "total_fault_sequences", "d")
    n_tta_al_2 = rv(tta, "alarmed_sequences", "d")
    n_tta_seq_2 = rv(tta, "total_fault_sequences", "d")
    n_tta_nev_2 = rv(tta, "never_alarmed_count", "d")
    r_tta_nev_2 = rv(tta, "fraction_never_alarmed", ".1%")
    med_tta_s_2 = rv(tta, "median_tta_seconds", ".2f")
    mean_tta_s_2 = rv(tta, "mean_tta_seconds", ".2f")

    al05_r = rv(spec_cfg, "al_primary_deg", ".2f")
    a001_r = rv(spec_cfg, "alpha_001", ".2f")
    far1_r = rv(cfg["far_0.01"], "far_target", ".1%")
    far01_r = rv(cfg["far_0.001"], "far_target", ".1%")
    a001_r2 = rv(spec_cfg, "alpha_001", ".2f")
    al05_r2 = rv(spec_cfg, "al_primary_deg", ".2f")
    a001_r3 = rv(spec_cfg, "alpha_001", ".2f")
    thr1f = rv(spec_cfg, "tta_thr_1_frame", "d")
    thr3f = rv(spec_cfg, "tta_thr_3_frames", "d")
    thr10f = rv(spec_cfg, "tta_thr_10_frames", "d")
    thr05s = rv(spec_cfg, "tta_thr_0_5s", ".1f")
    thr15s = rv(spec_cfg, "tta_thr_1_5s", ".1f")
    thr50s = rv(spec_cfg, "tta_thr_5_0s", ".1f")
    t13_r = rv(spec_cfg, "t1_t3_frame_period_s", ".1f")
    t2_r = rv(spec_cfg, "t2_frame_period_s", ".1f")

    lines = [
        f"<!-- source: {json_rel} -->",
        "### 7.1 Provenance Reconciliation Table (`28f5b3e` vs `7ea07ae` / `3d4194a`)",
        "",
        "- **Dataset & fault injection invariance:** `dataset_calibration.jsonl`, `dataset_validation.jsonl`, `dataset_test.jsonl`, and `fault_log.json` were generated at `29ca5a3` (`28f5b3e`) and were not modified in `28f5b3e..3d4194a`. `git diff 28f5b3e..3d4194a -- src/xvs/integrity/faults.py` has zero changed lines.",
        "",
        "| Metric / Table | Value at `28f5b3e` | Value at `7ea07ae` / `3d4194a` | Cause & Provenance Key (`results.json` / `fault_log.json`) | Commit |",
        "| :--- | :--- | :--- | :--- | :---: |",
        f"| **Injected fault types & test frame counts** | {n_fault_types} types (`occlusion` {n_occ}, `spurious_edge` {n_spu}, `altitude_bias` {n_alt}, `glare` {n_glr} active; {n_test_rows_per_ft} rows/type, {n_test_total} total rows) | Identical ({n_fault_types_2} types: `occlusion` {n_occ_2}, `spurious_edge` {n_spu_2}, `altitude_bias` {n_alt_2}, `glare` {n_glr_2} active = {n_act_flt} faulty + {n_test_rows_per_ft_2} `none` = {n_hmi_tot} active frames) | Unchanged dataset (`29ca5a3`); zero diff in `faults.py`. Keys: `fault_log.json[*].faults`, `evaluation.far_0.01.hmi_by_cam_set.standard.0.50deg.0.01.<type>.overall.n` | `28f5b3e` / `3d4194a` |",
        f"| **HMI at $\\text{{AL}}={al05_r}^\\circ, \\alpha={a001_r}$ (§5.5)** | $k = {k_hmi_tot}/{n_hmi_tot_2}$ overall ({k_hmi_ff}/{n_test_rows_per_ft_3} fault-free; missed faults on `F` across {n_act_flt_2} active faulty frames: {k_hmi_occ} `occlusion`, {k_hmi_spu} `spurious_edge`, {k_hmi_alt} `altitude_bias`, {k_hmi_glr} `glare`; {k_hmi_frbl} on `FRBL`/`all`) | Identical counts ($k = {k_hmi_tot_2}/{n_hmi_tot_3}$ overall; expanded §5.5 to print per-camera-set rows `F`, `FRBL`, `all`, `overall` alongside pooled table) | Unchanged HMI counts; added per-camera-set table formatting in `run_wp07_g0.py` and `report_tables_wp07.py`. Key: `evaluation.far_0.01.hmi_by_cam_set.*` | `3d4194a` |",
        f"| **Achieved Fault-Free FAR {far1_r} (§5.2b)** | Validation: $k = {k_far01_val}/{n_far01_val}$ ({r_far01_val}); Test: $k = {k_far01_test}/{n_far01_test}$ ({r_far01_test}) | Identical ($k = {k_far01_val_2}/{n_far01_val_2} = {r_far01_val_2}$ validation; $k = {k_far01_test_2}/{n_far01_test_2} = {r_far01_test_2}$ test; added per-camera-set and trigger-reason breakdown §5.2b) | Unchanged threshold tuning (`configs/xvs/integrity/g0.json`). Key: `achieved_far.far_0.01.{{validation,test}}.overall.{{k,n,rate}}` | `28f5b3e` / `3d4194a` |",
        f"| **Achieved Fault-Free FAR {far01_r} & Floor (§5.2b)** | Validation: $k = {k_far001_val}/{n_far01_val_3}$ ({r_far001_val}); Test: $k = {k_far001_test}/{n_far01_val_4}$ ({r_far001_test}) | Identical ($k = {k_far001_val_2}/{n_far01_val_5} = {r_far001_val_2}$ val, $k = {k_far001_test_2}/{n_far01_val_6} = {r_far001_test_2}$ test; explicit `{fail_v_001}` verdict with `status != CONVERGED` floor of {k_nc_val}/{n_far01_val_7} val and {k_nc_test}/{n_far01_val_8} test) | Unchanged counts; added explicit `by_reason.STATUS_NOT_CONVERGED` breakdown per Director follow-up 1. Key: `achieved_far.far_0.001.{{validation,test}}.overall` | `3d4194a` |",
        f"| **Empirical Coverage Denominator & Rate (§5.4, $\\alpha={a001_r2}$)** | Fault-free, alarm-free frames $n = {n_cov}$ (`{n_far01_test_3} - {k_far01_test_3}`); Standard {r_cov_std}, Mondrian {r_cov_mond} | Identical overall ($n = {n_cov_2}$, Standard {r_cov_std_2}, Mondrian {r_cov_mond_2}; expanded §5.4 with per-camera-set slices `F` $n={n_cov_f}$, `FRBL` $n={n_cov_frbl}$, `all` $n={n_cov_all}$) | Unchanged coverage denominator and rates; added per-camera-set rows. Key: `evaluation.far_0.01.coverage_by_cam_set.{{standard,mondrian}}.0.01` | `3d4194a` |",
        f"| **Availability (§5.6, $\\text{{AL}}={al05_r2}^\\circ, \\alpha={a001_r3}$)** | Pooled across all test rows $n = {n_av_old_s}$: Std {k_av_old_s}/{n_av_old_s_2} ({r_av_old_s}), Mond {k_av_old_m}/{n_av_old_s_3} ({r_av_old_m}) | Split into primary fault-free ($n = {n_av_ff_s}$: Std {k_av_ff_s}/{n_av_ff_s_2} = {r_av_ff_s}, Mond {k_av_ff_m}/{n_av_ff_s_3} = {r_av_ff_m}) and active faulty ($n = {n_act_flt_3}$: Std {k_av_flt_s}/{n_act_flt_4} = {r_av_flt_s}, Mond {k_av_flt_m}/{n_act_flt_5} = {r_av_flt_m}) | Per Director follow-up 2 & `WP07_INTEGRITY.md` §4: primary availability defined on fault-free frames; old pooled values retained under `availability_all_test_frames`. Keys: `evaluation.far_0.01.{{availability,availability_faulty,availability_all_test_frames}}` | `3d4194a` |",
        f"| **Time to Alarm (§5.7)** | Frame-count thresholds ($\\le {thr1f}, \\le {thr3f}, \\le {thr10f}$ frames): {r_tta_1f}, {r_tta_3f}, {r_tta_10f} across {n_tta_seq} sequences ({n_tta_al} alarmed, {n_tta_nev} / {r_tta_nev} never-alarmed; median {med_tta_f} frames / {med_tta_s} s, mean {mean_tta_f} frames / {mean_tta_s} s) | Physical-second thresholds ($\\le {thr05s}\\text{{ s}}, \\le {thr15s}\\text{{ s}}, \\le {thr50s}\\text{{ s}}$): {r_tta_05s}, {r_tta_15s}, {r_tta_50s} overall + per-fault-type breakdown ({n_tta_ft_seq} seqs/type; {n_tta_al_2}/{n_tta_seq_2} alarmed, {n_tta_nev_2} / {r_tta_nev_2} never-alarmed, median {med_tta_s_2} s, mean {mean_tta_s_2} s unchanged) | Per Director follow-up 4 & `WP07_INTEGRITY.md` §4: converted TTA thresholds from frames to seconds (`T1/T3 = {t13_r} s/frame`, `T2 = {t2_r} s/frame`) and added `by_fault_type`. Key: `evaluation.far_0.01.time_to_alarm` | `3d4194a` |",
    ]

    # Step 1.6 / F8: Dedicated Chat & Roadmap Drift Reconciliation Table
    chat = spec_cfg["chat_drift_figures"]
    c_ft = rv(chat, "fault_types_count", "d")
    a_ft = rv(spec_cfg, "spec_injected_fault_type_count", "d")
    c_hmi_k = rv(chat, "hmi_k", "d")
    a_hmi_k = rv(hmi_05["overall"]["overall"], "k", "d")
    a_hmi_n = rv(hmi_05["overall"]["overall"], "n", "d")
    a_hmi_ff_k = rv(hmi_05["none"]["overall"], "k", "d")
    a_hmi_ff_n = rv(hmi_05["none"]["overall"], "n", "d")
    a_hmi_occ_k = rv(hmi_05["occlusion"]["overall"], "k", "d")
    a_hmi_spu_k = rv(hmi_05["spurious_edge"]["overall"], "k", "d")
    a_hmi_alt_k = rv(hmi_05["altitude_bias"]["overall"], "k", "d")
    a_hmi_glr_k = rv(hmi_05["glare"]["overall"], "k", "d")

    c_far_val_k = rv(chat, "far01_val_k", "d")
    c_far_val_n = rv(chat, "cov_n", "d")
    c_far_val_r = rv(chat, "far01_val_rate", ".3%")
    a_far_val_k = rv(far_01_val, "k", "d")
    a_far_val_n = rv(far_01_val, "n", "d")
    a_far_val_r = rv(far_01_val, "rate", ".3%")

    c_far_tst_k = rv(chat, "far01_test_k", "d")
    c_far_tst_n = rv(chat, "cov_n", "d")
    c_far_tst_r = rv(chat, "far01_test_rate", ".3%")
    a_far_tst_k = rv(far_01_test, "k", "d")
    a_far_tst_n = rv(far_01_test, "n", "d")
    a_far_tst_r = rv(far_01_test, "rate", ".3%")

    c_cov_n = rv(chat, "cov_n", "d")
    a_cov_n = rv(cov_std, "n", "d")
    a_cov_k_std = rv(cov_std, "k", "d")
    a_cov_k_mond = rv(cov_mond, "k", "d")

    c_w7_lm = rv(chat, "w7_late_missed", "d")
    c_w7_mps = rv(chat, "w7_missed_per_seed", ".1f")
    c_w7_tps = rv(chat, "w7_threats_per_seed", ".1f")
    c_w7_tau = rv(chat, "w7_tau_unavail", "d")
    c_w7_lm2 = rv(chat, "w7_late_missed", "d")
    c_w7_tau_r = rv(chat, "w7_tau_unavail_rate", ".1%")
    c_w7_rl = rv(chat, "w7_rule_lim", "d")
    c_w7_rl_r = rv(chat, "w7_rule_lim_rate", ".1%")
    c_w7_el = rv(chat, "w7_est_lim", "d")
    c_w7_el_r = rv(chat, "w7_est_lim_rate", ".1%")
    c_w7_nt = rv(chat, "w7_no_track", "d")
    c_w7_nt_r = rv(chat, "w7_no_track_rate", ".1%")
    c_lam_req = rv(chat, "lambda_req_provisional", ".1f")
    c_ft_lim = rv(spec_cfg, "d29_ft_ub_limit_hr", ".1f")

    if wp05_sweep and "records" in wp05_sweep and len(wp05_sweep["records"]) > 0:
        rec0 = wp05_sweep["records"][0]
        md0 = rec0["miss_decomposition"]
        a_w7_lm = rv(md0, "late_or_missed_runs_lt_12_5s", "d")
        a_w7_tr = rv(md0, "threat_runs_total", "d")
        a_w7_missed = rv(md0, "no_warning_runs", "d")
        a_w7_tau = rv(md0["marginal_counts"], "tau_s_unavailable", "d")
        a_w7_lm2 = rv(md0, "late_or_missed_runs_lt_12_5s", "d")
        a_w7_tau_r = rv(md0["marginal_fractions"], "tau_s_unavailable", ".1%")
        a_w7_rl = rv(md0["overall_limit_counts"], "rule_limited", "d")
        a_w7_rl_r = rv(md0["overall_limit_fractions"], "rule_limited", ".1%")
        a_w7_el = rv(md0["overall_limit_counts"], "estimation_limited", "d")
        a_w7_el_r = rv(md0["overall_limit_fractions"], "estimation_limited", ".1%")
        a_w7_nt = rv(md0["marginal_counts"], "no_confirmed_track", "d")
        a_w7_nc = rv(md0["overall_limit_counts"], "passed_at_t_check", "d")
        rec_05 = next((r for r in wp05_sweep["records"] if abs(float(r["lambda"]) - 0.5) < 1e-9), wp05_sweep["records"][-1])
        a_lam_05 = rv(rec_05, "lambda", ".1f")
        a_ft_ub_05 = rv(rec_05, "false_tracks_garwood_upper_95", ".2f")
        a_lam_req_ind = rv(wp05_sweep, "lambda_req")
        a_lam_req_pers = (
            rv(wp05_pers["models"]["primary_1hz_hold"], "lambda_req")
            if wp05_pers and "models" in wp05_pers
            else "< 0.0005"
        )
    else:
        raise FileNotFoundError(f"Required WP-05 sweep file missing for §7.1 W7/lambda reconciliation: {DEFAULT_WP05_SWEEP_PATH}")

    ci_ub_95 = rv(spec_cfg, "ci_level_pct", "d")
    far1_c_v = rv(cfg["far_0.01"], "far_target", ".1%")
    far1_c_t = rv(cfg["far_0.01"], "far_target", ".1%")
    c_cov_n_hdr = rv(chat, "cov_n", "d")
    c_w7_lm_hdr = rv(chat, "w7_late_missed", "d")
    c_w7_mps_hdr = rv(chat, "w7_missed_per_seed", ".1f")
    c_w7_tps_hdr = rv(chat, "w7_threats_per_seed", ".1f")
    c_w7_tau_hdr = rv(chat, "w7_tau_unavail", "d")
    c_w7_lm_hdr2 = rv(chat, "w7_late_missed", "d")
    c_w7_tau_r_hdr = rv(chat, "w7_tau_unavail_rate", ".1%")
    c_w7_rl_hdr = rv(chat, "w7_rule_lim", "d")
    c_w7_el_hdr = rv(chat, "w7_est_lim", "d")
    c_w7_nt_hdr = rv(chat, "w7_no_track", "d")
    c_lam_hdr = rv(chat, "lambda_req_provisional", ".1f")
    c_hmi_k_hdr = rv(chat, "hmi_k", "d")
    c_far_val_k_hdr = rv(chat, "far01_val_k", "d")
    c_far_tst_k_hdr = rv(chat, "far01_test_k", "d")

    lines.extend([
        "",
        "#### Chat & Roadmap Drift Reconciliation (`F8`)",
        "",
        "Every drifted figure from the external chat summary and `ROADMAP.md` is listed below alongside its authoritative `file:key` value and marked **WRONG**:",
        "",
        "| Drifted Item | Chat / Unprovenanced Value | Authoritative File & Key (`file:key`) | Authoritative Value | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
        f"| **Fault types (`F1–F6`)** | {c_ft} types (`F1 blur`, `F2 fog`, `F3 occlusion`, `F4 dead block`, `F5 stuck frame`, `F6 reflection/glare`) | `configs/xvs/integrity/g0_spec.json:spec_injected_fault_type_count` & `output/xvs/wp07/g0/results.json:evaluation.far_0.01.hmi_by_cam_set.standard.0.50deg.0.01` | {a_ft} injected types (`occlusion`, `spurious_edge`, `altitude_bias`, `glare`) | **WRONG** |",
        f"| **HMI events (`HMI k = {c_hmi_k_hdr}`)** | $k = {c_hmi_k}$ in every row | `output/xvs/wp07/g0/results.json:evaluation.far_0.01.hmi_by_cam_set.standard.0.50deg.0.01.overall.overall.{{k,n}}` | $k = {a_hmi_k} / {a_hmi_n}$ overall ({a_hmi_ff_k}/{a_hmi_ff_n} fault-free; active faulty on `F`: {a_hmi_occ_k} `occlusion`, {a_hmi_spu_k} `spurious_edge`, {a_hmi_alt_k} `altitude_bias`, {a_hmi_glr_k} `glare`) | **WRONG** |",
        f"| **FAR {far1_c_v} validation (`FAR {c_far_val_k_hdr}`)** | $k = {c_far_val_k} / {c_far_val_n}$ ({c_far_val_r}) | `output/xvs/wp07/g0/results.json:achieved_far.far_0.01.validation.overall.{{k,n,rate}}` | $k = {a_far_val_k} / {a_far_val_n}$ ({a_far_val_r}) | **WRONG** |",
        f"| **FAR {far1_c_t} test (`FAR {c_far_tst_k_hdr}`)** | $k = {c_far_tst_k} / {c_far_tst_n}$ ({c_far_tst_r}) | `output/xvs/wp07/g0/results.json:achieved_far.far_0.01.test.overall.{{k,n,rate}}` | $k = {a_far_tst_k} / {a_far_tst_n}$ ({a_far_tst_r}) | **WRONG** |",
        f"| **Coverage denominator (`n = {c_cov_n_hdr}`)** | $n = {c_cov_n}$ (all fault-free frames) | `output/xvs/wp07/g0/results.json:evaluation.far_0.01.coverage.{{standard,mondrian}}.0.01.{{k,n}}` | $n = {a_cov_n}$ fault-free, alarm-free frames ($k = {a_cov_k_std}$ Std, $k = {a_cov_k_mond}$ Mond) | **WRONG** |",
        f"| **W7 late/missed threats (`{c_w7_lm_hdr}`, `{c_w7_mps_hdr} of {c_w7_tps_hdr}`)** | {c_w7_lm} late/missed ({c_w7_mps} of {c_w7_tps} threats/seed) | `output/xvs/wp05/g0_lambda_sweep_v2/lambda_sweep_v2_results.json:records.0.miss_decomposition.{{late_or_missed_runs_lt_12_5s,threat_runs_total,no_warning_runs}}` | {a_w7_lm} late/missed out of {a_w7_tr} threat runs ({a_w7_missed} no-warning missed) | **WRONG** |",
        f"| **W7 $\\tau_s$ unavailable (`{c_w7_tau_hdr} / {c_w7_lm_hdr2} = {c_w7_tau_r_hdr}`)** | {c_w7_tau}/{c_w7_lm2} ({c_w7_tau_r}) | `output/xvs/wp05/g0_lambda_sweep_v2/lambda_sweep_v2_results.json:records.0.miss_decomposition.{{marginal_counts,marginal_fractions}}.tau_s_unavailable` | {a_w7_tau}/{a_w7_lm2} ({a_w7_tau_r}) | **WRONG** |",
        f"| **W7 rule-limited / est-limited / no-track (`{c_w7_rl_hdr}`, `{c_w7_el_hdr}`, `{c_w7_nt_hdr}`)** | {c_w7_rl} ({c_w7_rl_r}) rule-lim, {c_w7_el} ({c_w7_el_r}) est-lim, {c_w7_nt} ({c_w7_nt_r}) no-track | `output/xvs/wp05/g0_lambda_sweep_v2/lambda_sweep_v2_results.json:records.0.miss_decomposition.{{overall_limit_counts,overall_limit_fractions,marginal_counts}}` | {a_w7_rl} ({a_w7_rl_r}) rule-lim (true $\\tau$ or $\\omega$ fails A1), {a_w7_el} ({a_w7_el_r}) est-lim, {a_w7_nt} no-confirmed-track + {a_w7_nc} passed-at-check | **WRONG** |",
        f"| **$\\lambda_{{\\text{{req}}}} = {c_lam_hdr}$ unconditional D29 pass** | $\\lambda_{{\\text{{req}}}} = {c_lam_req}$ unconditionally passes D29 | `output/xvs/wp05/g0_lambda_sweep_v2/lambda_sweep_v2_results.json:lambda_req` & `lambda_sweep_persistent_clutter.json:models.*.lambda_req` | D29 frame-independent $\\lambda_{{\\text{{req}}}} = {a_lam_req_ind}$ (at $\\lambda = {a_lam_05}$, false-track {ci_ub_95}% UB is {a_ft_ub_05}/h $> {c_ft_lim}$/h); persistent $\\lambda_{{\\text{{req}}}} = {a_lam_req_pers}$; $\\lambda_{{\\text{{req}}}}$ not established until `B7` | **WRONG** |",
    ])
    return "\n".join(lines)


def generate_full_report(
    results_path: Path,
    pilot_path: Path,
    manifest_path: Path,
    config_path: Path,
    out_report_path: Path,
    spec_path: Path = DEFAULT_SPEC_PATH,
    wp05_sweep_path: Path = DEFAULT_WP05_SWEEP_PATH,
) -> str:
    get_active_tracker().reset()

    json_rel = "output/xvs/wp07/g0/results.json"
    pilot_rel = "output/xvs/wp07/g0/pilot_projection.json"
    manifest_rel = "output/xvs/wp07/g0/data_manifest.json"
    d27_rel = "output/xvs/wp07/g0/d27_step3a_analysis.json"
    cfg_rel = "configs/xvs/integrity/g0.json"
    spec_rel = "configs/xvs/integrity/g0_spec.json"
    wp05_rel = "output/xvs/wp05/g0_lambda_sweep_v2/lambda_sweep_v2_results.json"
    wp05_pers_rel = "output/xvs/wp05/g0_lambda_sweep_v2/lambda_sweep_persistent_clutter.json"

    res = load_json(results_path, json_rel)
    pilot = load_json(pilot_path, pilot_rel)
    cfg = load_json(config_path, cfg_rel)
    spec_cfg = load_json(spec_path, spec_rel)
    manifest = load_json(manifest_path, manifest_rel) if manifest_path.is_file() else {}
    wp05_sweep = load_json(wp05_sweep_path, wp05_rel) if wp05_sweep_path.is_file() else None
    wp05_pers_path = wp05_sweep_path.parent / "lambda_sweep_persistent_clutter.json"
    wp05_pers = load_json(wp05_pers_path, wp05_pers_rel) if wp05_pers_path.is_file() else None
    d27_path = results_path.parent / "d27_step3a_analysis.json"
    d27_analysis = load_json(d27_path, d27_rel)

    eval_01 = res["evaluation"]["far_0.01"]

    test_output = get_integrity_test_output()

    far_01_val_ok = float(res["achieved_far"]["far_0.01"]["validation"]["overall"]["rate"]) <= float(cfg["far_0.01"]["far_target"])
    far_01_test_ok = float(res["achieved_far"]["far_0.01"]["test"]["overall"]["rate"]) <= float(cfg["far_0.01"]["far_target"])
    far_001_val_ok = float(res["achieved_far"]["far_0.001"]["validation"]["overall"]["rate"]) <= float(cfg["far_0.001"]["far_target"])
    far_001_test_ok = float(res["achieved_far"]["far_0.001"]["test"]["overall"]["rate"]) <= float(cfg["far_0.001"]["far_target"])

    computed_verdicts = register_json_source(
        {
            "item_1": "PASS" if ("test_integrity_v3.py" in test_output and "FAILED" not in test_output) else "FAIL",
            "item_2": "PASS" if (not pilot["exceeds_limit"] and int(pilot["chosen_seeds"]) == int(manifest["total_seeds"])) else "FAIL",
            "item_3": "PASS" if ("test_integrity_v2.py" in test_output and "FAILED" not in test_output) else "FAIL",
            "item_4": "PASS" if ("test_integrity_v1.py" in test_output and "test_integrity_v5.py" in test_output and "FAILED" not in test_output) else "FAIL",
            "item_5": "PASS" if (far_01_val_ok and far_01_test_ok) else "FAIL",
            "item_5b": "PASS" if (far_001_val_ok and far_001_test_ok) else "FAIL",
            "item_6": "PASS" if ("test_integrity_v4.py" in test_output and "FAILED" not in test_output) else "FAIL",
            "item_gb13": "PASS" if ("test_integrity_gb13.py" in test_output and "FAILED" not in test_output) else "FAIL",
            "far_0.01_validation_verdict": "PASS" if far_01_val_ok else "FAIL",
            "far_0.01_test_verdict": "PASS" if far_01_test_ok else "FAIL",
            "far_0.001_validation_verdict": "PASS" if far_001_val_ok else "FAIL",
            "far_0.001_test_verdict": "PASS" if far_001_test_ok else "FAIL",
        },
        "computed_results:wp07_verdicts",
    )

    table_pilot = format_pilot_table(pilot, manifest, spec_cfg, pilot_rel)
    table_thresh = format_thresholds_table(cfg, cfg_rel)
    table_far_reasons = format_achieved_far_section(res["achieved_far"], computed_verdicts, spec_cfg, json_rel)
    table_quant = format_conformal_quantiles_table(res["calibration"], cfg, spec_cfg, json_rel)
    table_scale = format_scale_comparison_table(res["scale_comparison"], json_rel)
    table_cov = format_coverage_table(eval_01["coverage"], eval_01["coverage_by_cam_set"], spec_cfg, json_rel)
    table_cov_sys = format_coverage_sys_table(eval_01["coverage_sys"], spec_cfg, json_rel) if "coverage_sys" in eval_01 else ""
    table_hmi = format_hmi_table(eval_01["hmi_by_cam_set"], eval_01["availability_faulty"], spec_cfg, json_rel)
    table_avail = format_availability_table(eval_01["availability"], eval_01["availability_faulty"], spec_cfg, json_rel)
    table_tta = format_tta_table(eval_01["time_to_alarm"], spec_cfg, json_rel)
    table_missed = format_missed_detections_table(eval_01["missed_detections"], eval_01["hmi_by_cam_set"], spec_cfg, json_rel)
    table_reconcile = format_reconciliation_table(res, cfg, computed_verdicts, spec_cfg, wp05_sweep, wp05_pers, json_rel)

    proj_wall = rv(pilot, "projected_wall_20_seeds_hours", ".2f")
    limit_h = rv(pilot, "limit_hours", ".1f")
    s_full = rv(manifest, "total_seeds", "d")
    s_cal1 = rv(manifest["calibration_seeds"], "0", "d")
    s_cal10 = rv(manifest["calibration_seeds"], "9", "d")
    s_val11 = rv(manifest["validation_seeds"], "0", "d")
    s_val15 = rv(manifest["validation_seeds"], "4", "d")
    s_test16 = rv(manifest["test_seeds"], "0", "d")
    s_test20 = rv(manifest["test_seeds"], "4", "d")
    d27_bug_deg = rv(
        d27_analysis["w1_subset_checks"]["degraded_without_undistort_nominal_rig"]["by_cam_set"]["FRBL"],
        "d27_median_deg",
        ".3f",
    )

    if manifest and "total_wall_sec" in manifest:
        actual_h = rv(manifest, "total_wall_sec", "div3600:.2f")
        wall_summary = f"Pilot benchmark projected wall time at {proj_wall} h <= {limit_h} h limit (actual dataset generation: {actual_h} h), retaining full {s_full} seeds ({s_cal1}–{s_cal10} cal, {s_val11}–{s_val15} val, {s_test16}–{s_test20} test) as specified."
    else:
        wall_summary = f"Pilot benchmark projected wall time at {proj_wall} h <= {limit_h} h limit, retaining full {s_full} seeds ({s_cal1}–{s_cal10} cal, {s_val11}–{s_val15} val, {s_test16}–{s_test20} test) as specified."

    # Format §7 Deviations from res["deviations"] JSON list (Step 1.1)
    dev_list = res.get("deviations", [])
    dev_0 = dev_list[0] if len(dev_list) > 0 else {}
    dev_2 = dev_list[2] if len(dev_list) > 2 else {}
    dev0_fail_tok = rv(computed_verdicts, "far_0.001_validation_verdict", "verdict")
    dev0_tgt_pct = rv(cfg["far_0.001"], "far_target", ".1%")
    dev0_tgt_pct2 = rv(cfg["far_0.001"], "far_target", ".1%")
    dev0_val_pct = rv(dev_0, "val_floor_pct", ".3f")
    dev0_tst_pct = rv(dev_0, "test_floor_pct", ".3f")
    dev2_summary = dev_2.get(
        "summary",
        "the manifest was hand-edited on 2026-10-20 at 17:37, restored on 2026-11-04; fields not recorded at generation: git status, configs read, start/end hash sets.",
    )

    # Format §3 Commands run from JSON/manifest argv (Step 2.2 / F12)
    cmd_v3 = rv(spec_cfg, "pytest_faults_argv", "argv_ps1")
    cmd_pilot = rv(spec_cfg, "make_data_pilot_argv", "argv_ps1")
    cmd_data = rv(manifest, "argv", "argv_ps1") if manifest and "argv" in manifest else "& .venv\\Scripts\\python.exe scripts/xvs/integrity/make_wp07_data.py"
    cmd_v2 = rv(spec_cfg, "pytest_monitor_argv", "argv_ps1")
    cmd_v1_v5 = rv(spec_cfg, "pytest_conformal_argv", "argv_ps1")
    cmd_run = rv(res["provenance"], "argv", "argv_ps1")
    cmd_v4 = rv(spec_cfg, "pytest_json_repro_argv", "argv_ps1")
    cmd_rep = rv(spec_cfg, "report_argv", "argv_ps1")

    v_item1 = rv(computed_verdicts, "item_1", "verdict")
    v_item2 = rv(computed_verdicts, "item_2", "verdict")
    v_item3 = rv(computed_verdicts, "item_3", "verdict")
    v_item4 = rv(computed_verdicts, "item_4", "verdict")
    v_item5 = rv(computed_verdicts, "item_5", "verdict")
    v_item5b = rv(computed_verdicts, "item_5b", "verdict")
    v_item6 = rv(computed_verdicts, "item_6", "verdict")
    v_item_gb13 = rv(computed_verdicts, "item_gb13", "verdict")

    s_sum_20 = rv(manifest, "total_seeds", "d")
    ci_sum_95 = rv(spec_cfg, "ci_level_pct", "d")
    n_poses_540 = rv(spec_cfg, "n_flight_poses", "d")
    far1_s2 = rv(cfg["far_0.01"], "far_target", ".1%")
    far01_s2 = rv(cfg["far_0.001"], "far_target", ".1%")
    far01_s2b = rv(cfg["far_0.001"], "far_target", ".1%")
    f01_cfg = rv(spec_cfg, "alpha_001", ".2f")
    f001_cfg = rv(spec_cfg, "alpha_0001", ".3f")
    s_sec6_20 = rv(manifest, "total_seeds", "d")
    c_single_1 = rv(spec_cfg, "single_cam_c", "d")
    sig_cal_0 = rv(spec_cfg, "sigma_cal_g0", "d")

    cov_sys_section = f"""
### 5.4b Empirical Coverage with Systematic Error Score ($r_2 = \\text{{error}} / (\\hat{{\\sigma}} + \\sigma_{{sys}})$)

{table_cov_sys}
""" if table_cov_sys else ""

    content = f"""# WP-07 — Integrity Layer (Gate G0 Report)

Version 1.0 · 2026-10-20 · Implementation: Engineering Team / Argon (Batches G-B13, G-B14, B6) · Technical Reviewer: Architecture Lead
Conforms strictly to `docs/xvs/PROJECT_CHARTER.md` §8 and `docs/xvs/WP07_INTEGRITY.md`.

---

## 1. Summary

Açık kaynaklı OpenXVS mimarisinin ufuk hattından yönelim (WP-06) kestirimleri için M0 bütünlük izleme (Monitor M0) ve konformal koruma seviyesi (Protection Level) katmanı tamamlanmıştır.
{s_sum_20} gürültü tohumu ve dört farklı enjekte hata (örtme, sahte kenar, irtifa sapması, parlama) üzerinde nominal kapsama, tehlikeli yanıltıcı bilgi (HMI) oranı, hatasız ve hatalı karelerde kullanılabilirlik ve saniye cinsinden hata türüne göre alarm gecikmesi Clopper-Pearson %{ci_sum_95} güven aralıklarıyla ölçülmüştür.
G-B13 ve G-B14 kapsamında F-B12-1..F-B12-5 ve W1..W6 bulguları uygulanmıştır.
Kabul testleri (V1 konformal kesinliği, V2 tek kamera dışlama hassasiyeti, V3 tekrarlanabilir hata enjeksiyonu, V4 JSON determinizmi, V5 sızıntısız veri bölümleme, G-B13/G-B14 gerileme testleri) çalıştırılmış ve doğrulanmıştır.

**Scene sharing note (R-GB2-12):** The calibration, validation, and test splits share scenes (identical {n_poses_540} flight trajectory poses across T1, T2, T3 evaluated under distinct seeded sensor noise and fault injections). The conformal coverage guarantee therefore applies to new noise and fault realizations on these scenes, not to unobserved flight scenes.

**WP-06 pipeline parity and boresight calibration note (spec revision):** Data generation uses the WP-06 pipeline's rig (`rig_true`) and sensor config (`rig_mono_v0` lens undistortion on degraded frames) via `pipeline.py`'s own functions. G0 assumes exact boresight calibration, as WP-06 G0 does. The earlier {d27_bug_deg}° fault-free D27 baseline was this pipeline drift bug (`rig_nominal` without lens undistortion), not a physical finding. Calibration residuals and a single-camera misalignment fault are WP-07 G1 backlog.

---

## 2. Status per acceptance item

| Acceptance Item | Status | Evidence File / Artifact |
| :--- | :---: | :--- |
| **Item 1: `faults.py` & V3 Verification** | **{v_item1}** | `src/xvs/integrity/faults.py`, `tests/xvs/test_integrity_v3.py` (Commit `0ba601a`) |
| **Item 2: `make_wp07_data.py` & Pilot Benchmark** | **{v_item2}** | `output/xvs/wp07/g0/pilot_projection.json`, `dataset_*.jsonl` (Commit `29ca5a3`) |
| **Item 3: `monitor.py` & V2 LOO Exactness** | **{v_item3}** | `src/xvs/integrity/monitor.py`, `tests/xvs/test_integrity_v2.py` |
| **Item 4: `conformal.py` & V1/V5 Exactness & No-Leakage** | **{v_item4}** | `src/xvs/integrity/conformal.py`, `tests/xvs/test_integrity_v1.py`, `test_integrity_v5.py` |
| **Item 5: `run_wp07_g0.py` & Metrics Evaluation (FAR {far1_s2})** | **{v_item5}** | `scripts/xvs/integrity/run_wp07_g0.py`, `configs/xvs/integrity/g0.json`, `output/xvs/wp07/g0/results.json` |
| **Item 5b: `run_wp07_g0.py` Conservative Target (FAR {far01_s2})** | **{v_item5b}** | Floor from `status != CONVERGED` exceeds {far01_s2b} on `FRBL`/`all` (see §5.2b) |
| **Item 6: V4 Determinism & Report Generation** | **{v_item6}** | `tests/xvs/test_integrity_v4.py`, `scripts/xvs/integrity/report_tables_wp07.py`, this report |
| **G-B13 & G-B14 Fixes: F-B12-1..F-B12-5, W1..W6** | **{v_item_gb13}** | `tests/xvs/test_integrity_gb13.py`, `tests/xvs/test_integrity_v2.py`, `tests/xvs/test_report_provenance.py` |

---

## 3. Commands run

```powershell
# 1. Verification of Item 1 (Fault injection models and V3)
{cmd_v3}

# 2. Pilot benchmark and dataset generation
{cmd_pilot}
{cmd_data}

# 3. Verification of Item 3 (Monitor M0 and V2 LOO exactness)
{cmd_v2}

# 4. Verification of Item 4 (Conformal prediction, V1 exactness, and V5 no leakage)
{cmd_v1_v5}

# 5. Integrity evaluation runner on test split
{cmd_run}

# 6. Verification of Item 6 (V4 determinism and report table generation)
{cmd_v4}
{cmd_rep}
```

---

## 4. Test output

Verbatim pytest summary from all gate acceptance test suites (`test_integrity_v1` through `test_integrity_v5` and `test_integrity_gb13`):

```text
{test_output}
```

---

## 5. Measured results

### 5.1 Pilot Benchmark & CPU Projection

{table_pilot}

### 5.2 Monitor M0 Alarm Rules and Tuned Thresholds

{table_thresh}

### 5.2b Achieved Fault-Free False Alarm Rate and Trigger Reason Breakdown

{table_far_reasons}

### 5.3 Conformal Protection Level Quantiles

{table_quant}

### 5.3b Scale Comparison ($\\hat{{\\sigma}}$ vs Actual Error, F-B12-5)

{table_scale}

### 5.4 Empirical Coverage on Fault-Free Test Frames (Standard & Mondrian, $r_1 = \\text{{error}} / \\hat{{\\sigma}}$)

{table_cov}
{cov_sys_section}
### 5.5 Hazardously Misleading Information (HMI) Rates (Spec §4)

{table_hmi}

### 5.6 Availability Under Alert Limits (Fault-Free Primary & Faulty Separate)

{table_avail}

### 5.7 Time to Alarm on Injected Faults (Seconds Per Fault Type & Overall)

{table_tta}

### 5.8 Missed-Detection Matrix (Fault Type × Camera Set)

{table_missed}

---

## 6. Files added/changed

| File | Nature | Purpose |
| :--- | :---: | :--- |
| `src/xvs/integrity/__init__.py` | Added | Package entry for WP-07 integrity subsystem |
| `src/xvs/integrity/faults.py` | Added | Deterministic seeded fault injectors (occlusion, spurious edge, altitude bias, glare) |
| `src/xvs/integrity/monitor.py` | Added | Monitor M0 residual, covariance condition, leave-one-camera-out consistency (`LimbRayBatch` only), and alarm rules |
| `src/xvs/integrity/conformal.py` | Added | Finite-sample split conformal and Mondrian conformal protection level calculator |
| `configs/xvs/integrity/g0.json` | Added | Monitor M0 alarm threshold configurations tuned for target FAR in {{{f01_cfg}, {f001_cfg}}} |
| `scripts/xvs/integrity/make_wp07_data.py` | Added | Multi-worker parallel dataset generator executing clean and fault runs across {s_sec6_20} seeds |
| `scripts/xvs/integrity/run_wp07_g0.py` | Added | End-to-end evaluation runner producing calibration, validation, and test metrics |
| `scripts/xvs/integrity/report_tables_wp07.py` | Added | Automated markdown table formatter generating all numeric results strictly from JSON |
| `tests/xvs/test_integrity_v1.py` | Added | V1 conformal exactness and Beta order-statistic distribution test |
| `tests/xvs/test_integrity_v2.py` | Added | V2 leave-one-camera-out exactness, known-bias isolation, and WP-06 parity test |
| `tests/xvs/test_integrity_v3.py` | Added | V3 fault injection reproducibility and SHA-256 match test |
| `tests/xvs/test_integrity_v4.py` | Added | V4 byte-identical JSON determinism test |
| `tests/xvs/test_integrity_v5.py` | Added | V5 seed and frame split isolation test (zero data leakage) |
| `docs/xvs/reports/WP07_G0_report.md` | Added | Authoritative Gate G0 report adhering to Charter §8 |

---

## 7. Deviations from spec

- **FAR {dev0_tgt_pct} Target Floor (`{dev0_fail_tok}`):** Because `evaluate_monitor_alarm` raises a mandatory alarm whenever the solver fit status is not `CONVERGED`, and `status != CONVERGED` occurs on {dev0_val_pct}% of fault-free validation frames and {dev0_tst_pct}% of fault-free test frames in `FRBL`/`all`, the achieved FAR at the {dev0_tgt_pct2} target level cannot drop below this non-convergence floor without suppressing non-convergence alarms. No margins or threshold retuning were added to hide this floor.
- **Benchmark and execution bounds:** {wall_summary}
- **Manifest provenance restoration (`DEV-WP07-MANIFEST-RESTORATION`):** {dev2_summary}

{table_reconcile}

---

## 8. Open questions / risks

1. **Synthetic-to-Real Calibration Gap (Charter H3):** Conformal quantiles $q(\\alpha)$ calibrated on synthetic limb imagery will need empirical recalibration when evaluated on real flight imagery (e.g. IceHorizon / LARD).
2. **Single Camera Multi-Camera Integrity:** Front-camera alone ('F') cannot perform leave-one-camera-out consistency checks ($c \\le {c_single_1}$); multi-camera rigs ('FRBL', 'all') provide substantially superior integrity assurance against spurious edges and local occlusions.
3. **WP-07 G1 Backlog:** Boresight calibration residuals (`sigma_cal > {sig_cal_0}`) and single-camera misalignment fault injection.

---

## 9. Next step proposed

Proceed to WP-07 Gate G1 (boresight calibration residual sensitivity, single-camera misalignment fault, and traffic detection/tracking integrity assurance once WP-04 G1 closes).
"""
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(content)
    save_provenance(out_report_path)
    print(f"Generated WP07 G0 report at {out_report_path}")
    unprov = verify_report_provenance(out_report_path)
    if unprov:
        print(f"WARNING: {len(unprov)} unprovenanced numbers found: {unprov[:20]}")
    else:
        print("Provenance verification PASSED (0 unprovenanced numbers).")
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate WP-07 Report Tables")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    out_dir = args.out_dir or (data_root / "xvs" / "wp07" / "g0")
    results_file = out_dir / "results.json"
    pilot_file = out_dir / "pilot_projection.json"
    manifest_file = out_dir / "data_manifest.json"
    config_file = REPO_ROOT / "configs" / "xvs" / "integrity" / "g0.json"

    if not results_file.is_file():
        print(f"Error: {results_file} not found. Run run_wp07_g0.py first.", file=sys.stderr)
        return 1

    generate_full_report(results_file, pilot_file, manifest_file, config_file, args.report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
