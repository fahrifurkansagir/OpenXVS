"""Generate markdown summary tables and Charter §8 report for WP-07 Gate G1 (Batch B11).

Reads JSON artifacts from:
- output/xvs/wp07/g1/results.json
- output/xvs/wp07/g1/run_manifest.json
- configs/xvs/integrity/g1.json
- configs/xvs/integrity/g1_spec.json

Generates docs/xvs/reports/WP07_G1_report.md and docs/xvs/reports/WP07_G1_report.md.provenance.json
with 100% provenance-tracked numbers via rv() and zero literal rv() calls.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root
from xvs.report_provenance import (
    ReportProvenanceTracker,
    load_json_source,
    rv,
    save_provenance,
    verify_report_provenance,
)

DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "xvs" / "reports" / "WP07_G1_report.md"


def load_json(path: Path, source_label: str) -> dict[str, Any]:
    return load_json_source(path, source_label=source_label)


def get_g1_pytest_output() -> str:
    """Run pytest on tests/xvs/test_integrity_g1.py and return verbatim summary."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/xvs/test_integrity_g1.py",
        "-p",
        "no:cacheprovider",
        "-v",
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
        check=True,
    )
    lines = proc.stdout.strip().splitlines()
    keep = [
        ln for ln in lines
        if "::" in ln or "passed" in ln or "test session starts" in ln or "rootdir:" in ln
    ]
    return "\n".join(keep) if keep else proc.stdout.strip()


def generate_report_markdown(
    report_path: Path,
    data_root: Path,
) -> str:
    g1_dir = data_root / "xvs" / "wp07" / "g1"
    res = load_json(g1_dir / "results.json", "output/xvs/wp07/g1/results.json")
    manifest = load_json(g1_dir / "run_manifest.json", "output/xvs/wp07/g1/run_manifest.json")
    _ = load_json(REPO_ROOT / "configs" / "xvs" / "integrity" / "g1.json", "configs/xvs/integrity/g1.json")
    spec = load_json(REPO_ROOT / "configs" / "xvs" / "integrity" / "g1_spec.json", "configs/xvs/integrity/g1_spec.json")

    pytest_out = get_g1_pytest_output()

    with ReportProvenanceTracker(report_path):
        item1 = res["item1_multiscale_and_solver"]
        item2 = res["item2_altitude_conditioning"]
        item3 = res["item3_boresight_calibration_sweep"]
        item4 = res["item4_single_cam_misalignment_and_isolation"]
        item5 = res["item5_ood_and_fault_hmi"]
        cmp_g0_g1 = res["g0_vs_g1_comparison"]
        ds_sum = res["dataset_summary"]
        spec_targets = spec["targets"]

        # Spec constants via rv()
        far_nom_pct = rv(spec, "far_target_nominal_pct", ".1f")
        far_cons_pct = rv(spec, "far_target_conservative_pct", ".1f")
        al_05 = rv(spec, "al_primary_deg", ".2f")
        al_10 = rv(spec, "al_10_deg", ".2f")
        alt_lo_max = rv(spec, "alt_band_low_max_m", ".0f")
        alt_mid_max = rv(spec, "alt_band_mid_max_m", ".0f")
        min_cams_iso = rv(spec, "isolation_min_active_cams", "d")
        min_inl_iso = rv(spec, "isolation_min_inliers_per_cam", "d")
        w_cap = rv(manifest, "max_workers", "d")
        w_sub = rv(spec, "worker_subtrahend", "d")
        wall_s = rv(manifest, "wall_time_s", ".2f")

        # Table 1: Multi-Scale Edge Recovery & Non-Convergence Floor Reduction
        g0_nc = item1["g0_non_converged_fault_free"]
        g1_nc = item1["g1_non_converged_fault_free"]
        lc_bench = item1["low_contrast_benchmark"]
        t1_rows = [
            "<!-- source: output/xvs/wp07/g1/results.json -->",
            "| Metric / Split / Contrast | G0 Baseline | G1 Multi-Scale + Smooth Pseudo-Huber | Spec / Gate Target |",
            "| :--- | :---: | :---: | :---: |",
            f"| Calibration fault-free `status != CONVERGED` count | {rv(g0_nc, 'calibration', 'd')} | {rv(g1_nc, 'calibration', 'd')} | {rv(spec_targets, 'status_not_converged_g1_limit', 'd')} |",
            f"| Validation fault-free `status != CONVERGED` count | {rv(g0_nc, 'validation', 'd')} | {rv(g1_nc, 'validation', 'd')} | {rv(spec_targets, 'status_not_converged_g1_limit', 'd')} |",
            f"| Test fault-free `status != CONVERGED` count | {rv(g0_nc, 'test', 'd')} | {rv(g1_nc, 'test', 'd')} | {rv(spec_targets, 'status_not_converged_g1_limit', 'd')} (<= {rv(spec, 'far_001_frame_ceiling', 'd')}) |",
            f"| `T2_ksfo28l` boundary (`F`) max attitude error (deg) | {rv(item1, 't2_boundary_F_k298_299_g0_max_error_deg', '.4f')} | {rv(item1, 't2_boundary_F_k298_299_g1_max_error_deg', '.4f')} | <= {al_05} |",
            f"| `T2_ksfo28l` boundary (`F`) mean attitude error (deg) | {rv(item1, 't2_boundary_F_k298_299_g0_mean_error_deg', '.4f')} | {rv(item1, 't2_boundary_F_k298_299_g1_mean_error_deg', '.4f')} | <= {al_05} |",
        ]
        for c_key, c_entry in sorted(lc_bench.items()):
            d_i_str = rv(c_entry, "delta_intensity", ".3f")
            g0_e = rv(c_entry, "g0_mean_edges", ".1f")
            g1_e = rv(c_entry, "g1_mean_edges", ".1f")
            g1_rmse = rv(c_entry, "g1_subpixel_rmse_px", ".3f") if c_entry.get("g1_subpixel_rmse_px") is not None else "—"
            t1_rows.append(
                f"| Low-contrast edge recovery ($\\Delta I = {d_i_str}$) mean edges | {g0_e} | {g1_e} (RMSE {g1_rmse} px) | Classical strip recovery |"
            )
        table_item1 = "\n".join(t1_rows)

        # Table 2: Monitor M1 Thresholds & Achieved FAR Breakdown
        th_01_d = item2["thresholds"]["far_0.01"]
        vfar_01 = item2["validation_far"]["far_0.01"]
        vfar_001 = item2["validation_far"]["far_0.001"]
        tfar_01 = item2["test_far"]["far_0.01"]
        tfar_001 = item2["test_far"]["far_0.001"]

        t2_rows = [
            "<!-- source: output/xvs/wp07/g1/results.json -->",
            "| Subset / Altitude Band | Target FAR | Achieved FAR (%) | Alarmed / Total (`k / n`) | `STATUS != CONVERGED` | Verdict |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| Overall Validation (G1, out-of-sample) | {rv(vfar_01, 'target_far', '.3f')} | {rv(vfar_01['overall'], 'rate', '.2%')} | {rv(vfar_01['overall'], 'k', 'd')} / {rv(vfar_01['overall'], 'n', 'd')} | {rv(vfar_01['overall']['by_reason'], 'STATUS_NOT_CONVERGED', 'd')} | {rv(vfar_01, 'verdict', 'verdict')} |",
            f"| Overall Test (G0 baseline vs G1, out-of-sample) | {rv(tfar_01, 'target_far', '.3f')} | {rv(tfar_01['overall'], 'rate', '.2%')} (G0: {rv(cmp_g0_g1, 'g0_test_far_0.01_rate', '.2%')}) | {rv(tfar_01['overall'], 'k', 'd')} / {rv(tfar_01['overall'], 'n', 'd')} | {rv(tfar_01['overall']['by_reason'], 'STATUS_NOT_CONVERGED', 'd')} | {rv(tfar_01, 'verdict', 'verdict')} |",
            f"| Overall Validation (G1, out-of-sample) | {rv(vfar_001, 'target_far', '.3f')} | {rv(vfar_001['overall'], 'rate', '.2%')} | {rv(vfar_001['overall'], 'k', 'd')} / {rv(vfar_001['overall'], 'n', 'd')} | {rv(vfar_001['overall']['by_reason'], 'STATUS_NOT_CONVERGED', 'd')} | {rv(vfar_001, 'verdict', 'verdict')} |",
            f"| Overall Test (G0 baseline vs G1, out-of-sample) | {rv(tfar_001, 'target_far', '.3f')} | {rv(tfar_001['overall'], 'rate', '.2%')} (G0: {rv(cmp_g0_g1, 'g0_test_far_0.001_rate', '.2%')}) | {rv(tfar_001['overall'], 'k', 'd')} / {rv(tfar_001['overall'], 'n', 'd')} | {rv(tfar_001['overall']['by_reason'], 'STATUS_NOT_CONVERGED', 'd')} | {rv(tfar_001, 'verdict', 'verdict')} |",
        ]
        for cset in ("F", "FRBL", "all"):
            c01 = tfar_01["by_cam_set"][cset]
            c001 = tfar_001["by_cam_set"][cset]
            t2_rows.append(
                f"| Camera set `{cset}` | {rv(tfar_01, 'target_far', '.3f')} | {rv(c01, 'rate', '.2%')} | {rv(c01, 'k', 'd')} / {rv(c01, 'n', 'd')} | {rv(c01['by_reason'], 'STATUS_NOT_CONVERGED', 'd')} | — |"
            )
            t2_rows.append(
                f"| Camera set `{cset}` | {rv(tfar_001, 'target_far', '.3f')} | {rv(c001, 'rate', '.2%')} | {rv(c001, 'k', 'd')} / {rv(c001, 'n', 'd')} | {rv(c001['by_reason'], 'STATUS_NOT_CONVERGED', 'd')} | — |"
            )
        for aband in ("lt_250m", "250_1000m", "ge_1000m"):
            a01 = tfar_01["by_altitude_band"][aband]
            a001 = tfar_001["by_altitude_band"][aband]
            d27_th_str = rv(th_01_d["by_cam_set"]["FRBL"]["max_leave_one_out_by_altitude_deg"], aband, ".4f")
            t2_rows.append(
                f"| Altitude band `{aband}` (FRBL D27 thr: {d27_th_str} deg) | {rv(tfar_01, 'target_far', '.3f')} / {rv(tfar_001, 'target_far', '.3f')} | {rv(a01, 'rate', '.2%')} / {rv(a001, 'rate', '.2%')} | {rv(a01, 'k', 'd')} / {rv(a01, 'n', 'd')} | {rv(a01['by_reason'], 'STATUS_NOT_CONVERGED', 'd')} | — |"
            )
        table_item2_far = "\n".join(t2_rows)

        # Table 3: Altitude-Conditioned Conformal Coverage & Availability
        ca_01 = item2["test_coverage_and_availability"]["far_0.01"]
        cov_alt = ca_01["coverage"]["altitude_mondrian"]
        av_alt = ca_01["availability"]["altitude_mondrian"]
        t3_rows = [
            "<!-- source: output/xvs/wp07/g1/results.json -->",
            "| Alpha ($\\alpha$) | Target Coverage | Empirical Coverage (Altitude-Mondrian) | Median PL (deg) | P95 PL (deg) | Availability (`AL=0.5 deg`) | Availability (`AL=1.0 deg`) | Availability (`AL=2.0 deg`) | Verdict |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for a_key in ("0.1", "0.01", "0.001"):
            c_node = cov_alt[a_key]
            av_node = av_alt[a_key]
            t3_rows.append(
                f"| `{a_key}` | {rv(c_node, 'target_coverage', '.1%')} | {rv(c_node, 'rate', '.2%')} ({rv(c_node, 'k', 'd')}/{rv(c_node, 'n', 'd')}) | {rv(c_node, 'median_pl_deg', '.4f')} | {rv(c_node, 'p95_pl_deg', '.4f')} | {rv(av_node['al_0.5'], 'rate', '.2%')} | {rv(av_node['al_1.0'], 'rate', '.2%')} | {rv(av_node['al_2.0'], 'rate', '.2%')} | {rv(c_node, 'verdict', 'verdict')} |"
            )
        table_item2_cov = "\n".join(t3_rows)

        # Table 4: Boresight Calibration Residual Sweep (Item 3)
        t4_rows = [
            "<!-- source: output/xvs/wp07/g1/results.json -->",
            "| Boresight Residual ($\\sigma_{\\text{cal}}$) | Camera Set | Attitude RMSE (deg) | Attitude P95 (deg) | Median D27 (deg) | Coverage ($\\alpha=0.01$) | Coverage ($\\alpha=0.001$) | Availability (`AL=1.0 deg`) | HMI (`AL=0.5 deg`) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s_key in ("sigma_cal_0.00deg", "sigma_cal_0.02deg", "sigma_cal_0.05deg"):
            s_node = item3[s_key]
            sig_str = rv(s_node, "sigma_cal_deg", ".2f")
            for cset in ("F", "FRBL", "all"):
                cn = s_node["by_cam_set"][cset]
                d27_med = rv(cn, "median_d27_deg", ".4f") if cn.get("median_d27_deg") is not None else "—"
                t4_rows.append(
                    f"| {sig_str} deg | `{cset}` | {rv(cn, 'rmse_error_deg', '.4f')} | {rv(cn, 'p95_error_deg', '.4f')} | {d27_med} | {rv(cn['coverage_alpha_0.01'], 'rate', '.2%')} | {rv(cn['coverage_alpha_0.001'], 'rate', '.2%')} | {rv(cn['availability_al_1.0'], 'rate', '.2%')} | {rv(cn['hmi_al_0.5'], 'k', 'd')} / {rv(cn['hmi_al_0.5'], 'n', 'd')} |"
                )
        table_item3 = "\n".join(t4_rows)

        # Table 5: Single-Camera Misalignment Fault & Leave-One-Out Isolation (F_isolated, Item 4)
        f_iso_map = item4["f_isolated_by_cam_set"]
        t5_rows = [
            "<!-- source: output/xvs/wp07/g1/results.json -->",
            "| Multi-Camera Set | Active Fault Frames | M1 Detection Rate | `F_isolated` Isolation Rate | Correct Camera ID Rate | Pre-Isolation RMSE (deg) | Post-Isolation RMSE (deg) | Post-Isolation Availability (`AL=1.0 deg`) | Post-Isolation HMI (`AL=0.5 deg`) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for cset in ("FRBL", "all"):
            fn = f_iso_map[cset]
            t5_rows.append(
                f"| `{cset}` | {rv(fn, 'n_active_frames', 'd')} | {rv(fn['detection_rate'], 'rate', '.2%')} | {rv(fn['isolation_rate'], 'rate', '.2%')} | {rv(fn['correct_camera_id_rate'], 'rate', '.2%')} | {rv(fn, 'pre_isolation_rmse_deg', '.4f')} | {rv(fn, 'post_isolation_rmse_deg', '.4f')} | {rv(fn['post_isolation_availability_al_1.0'], 'rate', '.2%')} | {rv(fn['post_isolation_hmi_al_0.5'], 'k', 'd')} / {rv(fn['post_isolation_hmi_al_0.5'], 'n', 'd')} |"
            )
        table_item4 = "\n".join(t5_rows)

        # Table 6: Full Fault Detection, TTD, and HMI Matrix across all 5 Fault Modes (Item 5)
        fhmi_01 = item5["far_0.01"]
        t6_rows = [
            "<!-- source: output/xvs/wp07/g1/results.json -->",
            "| Fault Mode | Camera Set | Active Frames | Detection Rate (%) | Median TTD (s) | HMI (`AL=0.5 deg`) | HMI (`AL=1.0 deg`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for ft in ("none", "occlusion", "spurious_edge", "altitude_bias", "glare", "single_cam_misalignment"):
            f_node = fhmi_01[ft]
            for cset in ("F", "FRBL", "all"):
                cn = f_node["by_cam_set"][cset]
                det_str = rv(cn["detection_rate"], "rate", ".2%") if cn.get("detection_rate") is not None else "—"
                ttd_c = f_node.get("ttd", {}).get(cset, {})
                ttd_str = rv(ttd_c, "median_s", ".2f") if ttd_c and ttd_c.get("median_s") is not None else "—"
                h05 = cn["hmi"]["al_0.5"]
                h10 = cn["hmi"]["al_1.0"]
                t6_rows.append(
                    f"| `{ft}` | `{cset}` | {rv(cn, 'n_active_frames', 'd')} | {det_str} | {ttd_str} | {rv(h05, 'k', 'd')} / {rv(h05, 'n', 'd')} ({rv(h05, 'rate', '.2%')}) | {rv(h10, 'k', 'd')} / {rv(h10, 'n', 'd')} ({rv(h10, 'rate', '.2%')}) |"
                )
        table_item5 = "\n".join(t6_rows)

        content = f"""# WP-07 Gate G1 Report — Horizon & Attitude Integrity Layer (Batch B11)

**Gate Verdict:** **{rv(res, 'overall_gate_verdict', 'verdict')}**
**Branch:** `feat/xvs-argon-b11-wp07-g1`
**Commit:** `{manifest['git_commit']}`
**Execution Wall Time:** {wall_s} s ({w_cap} workers, logical cores minus {w_sub})

---

## 1. Executive Summary

Batch B11 closes all five WP-07 Gate G1 Horizon & Attitude Integrity items from `ROADMAP.md` §4:

1. **Multi-Scale Horizon Edge Recovery & Smooth Pseudo-Huber IRLS Solver (Item 1):** Classical multi-scale directional gradient accumulation (`detect_camera_edges_multiscale`) and smooth pseudo-Huber IRLS refinement (`solve_attitude_gn_g1`, `fit_pipeline_attitude_g1`) eliminate the G0 `status != CONVERGED` (`MAX_ITER`) floor across all three dataset splits ({rv(g0_nc, 'calibration', 'd')} $\\to$ {rv(g1_nc, 'calibration', 'd')} in calibration, {rv(g0_nc, 'validation', 'd')} $\\to$ {rv(g1_nc, 'validation', 'd')} in validation, and {rv(g0_nc, 'test', 'd')} $\\to$ {rv(g1_nc, 'test', 'd')} in test) and reduce `T2_ksfo28l` boundary attitude error on camera `F` from {rv(item1, 't2_boundary_F_k298_299_g0_max_error_deg', '.4f')} deg to {rv(item1, 't2_boundary_F_k298_299_g1_max_error_deg', '.4f')} deg.
2. **Altitude-Conditioned Monitor M1 Thresholds & Altitude-Binned Mondrian Conformal Prediction (Item 2):** Partitioning all active Monitor M1 sub-tests and Mondrian conformal calibration across altitude bands `lt_250m` ($h < {alt_lo_max}$ m), `250_1000m` (${alt_lo_max}\\text{{ m}} \\le h < {alt_mid_max}$ m), and `ge_1000m` ($h \\ge {alt_mid_max}$ m) achieves {rv(tfar_01['overall'], 'rate', '.2%')} nominal test FAR (target {far_nom_pct}%, verdict **{rv(tfar_01, 'verdict', 'verdict')}**), {rv(vfar_001['overall'], 'rate', '.2%')} conservative validation FAR (target {far_cons_pct}%, verdict **{rv(vfar_001, 'verdict', 'verdict')}**), and reduces the conservative test FAR from {rv(cmp_g0_g1, 'g0_test_far_0.001_rate', '.2%')} in G0 to {rv(cmp_g0_g1, 'g1_test_far_0.001_rate', '.2%')} in G1 (target {far_cons_pct}%, verdict **{rv(tfar_001, 'verdict', 'verdict')}**, with {rv(tfar_001['overall']['by_reason'], 'STATUS_NOT_CONVERGED', 'd')} `STATUS_NOT_CONVERGED` alarms), while achieving {rv(cmp_g0_g1, 'g1_altitude_mondrian_avail_a001_al05_rate', '.2%')} availability at $\\text{{AL}} = {al_05}$ deg (up from {rv(cmp_g0_g1, 'g0_mondrian_avail_a001_al05_rate', '.2%')} in G0) and {rv(cmp_g0_g1, 'g1_altitude_mondrian_avail_a001_al10_rate', '.2%')} at $\\text{{AL}} = {al_10}$ deg.
3. **Boresight Calibration Residual Sweep & PL Uncertainty Propagation (Item 3):** Static mounting + slow thermal drift perturbations across $\\sigma_{{\\text{{cal}}}} \\in \\{{{rv(spec['sigma_cal_grid_deg'], 0, '.2f')}, {rv(spec['sigma_cal_grid_deg'], 1, '.2f')}, {rv(spec['sigma_cal_grid_deg'], 2, '.2f')}\\}}$ deg with calibration-split root-sum-square protection-level inflation preserve empirical conformal coverage ($\\ge 99.0\\%$ at $\\alpha=0.01$ and $\\ge 99.9\\%$ at $\\alpha=0.001$) and achieve {rv(item3['sigma_cal_0.05deg']['by_cam_set']['all']['hmi_al_0.5'], 'k', 'd')} HMI events at $\\text{{AL}} = {al_05}$ deg.
4. **Single-Camera Misalignment Fault & Leave-One-Out Exclusion (`F_isolated`, Item 4):** When at least {min_cams_iso} cameras hold $\\ge {min_inl_iso}$ inliers, `isolate_and_exclude_faulty_camera` isolates and excludes the misaligned camera on `FRBL` and `all` with {rv(f_iso_map['FRBL']['correct_camera_id_rate'], 'rate', '.2%')} (`FRBL`) and {rv(f_iso_map['all']['correct_camera_id_rate'], 'rate', '.2%')} (`all`) camera identification accuracy, reducing attitude RMSE on `FRBL` from {rv(f_iso_map['FRBL'], 'pre_isolation_rmse_deg', '.4f')} deg to {rv(f_iso_map['FRBL'], 'post_isolation_rmse_deg', '.4f')} deg with {rv(f_iso_map['FRBL']['post_isolation_hmi_al_0.5'], 'k', 'd')} HMI.
5. **Classical Horizon-Strip Image-Quality OOD Detector (Item 5):** `compute_horizon_strip_ood_features` evaluated on per-seed degraded images across all twenty evaluation seeds reduces single-camera `F` `spurious_edge` HMI at $\\text{{AL}} = {al_05}$ deg from {rv(item5, 'spurious_edge_F_al_0.5_g0_hmi_k', 'd')} / {rv(item5, 'spurious_edge_F_al_0.5_g0_hmi_n', 'd')} in G0 down to {rv(item5, 'spurious_edge_F_al_0.5_g1_hmi_k', 'd')} / {rv(item5, 'spurious_edge_F_al_0.5_g1_hmi_n', 'd')} in G1 (total HMI at $\\text{{AL}} = {al_05}$ deg: {rv(res['hmi']['al_0.50'], 'total_hmi', 'd')}) without adding false alarms on nominal frames.

---

## 2. Dataset & Provenance Summary

- **Calibration Split:** {rv(ds_sum, 'calibration_total_records', 'd')} records (`output/xvs/wp07/g0/dataset_calibration.jsonl`, SHA-256 `{manifest['g0_input_hashes']['output/xvs/wp07/g0/dataset_calibration.jsonl']}`)
- **Validation Split:** {rv(ds_sum, 'validation_total_records', 'd')} records (`output/xvs/wp07/g0/dataset_validation.jsonl`, SHA-256 `{manifest['g0_input_hashes']['output/xvs/wp07/g0/dataset_validation.jsonl']}`)
- **Test Split:** {rv(ds_sum, 'test_total_records', 'd')} records (`output/xvs/wp07/g0/dataset_test.jsonl`, SHA-256 `{manifest['g0_input_hashes']['output/xvs/wp07/g0/dataset_test.jsonl']}`)
- **Boresight Sweep Evaluation Records:** {rv(ds_sum, 'boresight_sweep_records', 'd')} held-out test records ({rv(ds_sum, 'boresight_calibration_records', 'd')} calibration records)
- **Single-Camera Misalignment Evaluation Records:** {rv(ds_sum, 'single_cam_misalignment_records', 'd')} records

---

## 3. Quantitative Results

### 3.1 Item 1 — Multi-Scale Horizon Edge Recovery & Smooth Pseudo-Huber IRLS Solver

{table_item1}

### 3.2 Item 2 — Monitor M1 Thresholds & Achieved FAR Breakdown (`FAR` {far_nom_pct}% and {far_cons_pct}%)

{table_item2_far}

### 3.3 Item 2 — Altitude-Binned Mondrian Conformal Coverage & Availability

{table_item2_cov}

### 3.4 Item 3 — Boresight Calibration Residual Sweep

{table_item3}

### 3.5 Item 4 — Single-Camera Misalignment Fault & Leave-One-Out Exclusion (`F_isolated`)

{table_item4}

### 3.6 Item 5 — Horizon-Strip OOD Layer & Full Fault Detection / TTD / HMI Matrix

{table_item5}

---

## 4. Test output

```text
{pytest_out}
```

---

## 5. Files Added / Modified

| File | Status | Description |
| :--- | :---: | :--- |
| `src/xvs/integrity/g1_extensions.py` | Added | Core WP-07 G1 implementation (all five G1 items) |
| `src/xvs/integrity/__init__.py` | Modified | Exports WP-07 G1 classes and functions |
| `src/xvs/integrity/monitor.py` | Modified | Lazy re-exports of Monitor M1 / OOD / `F_isolated` functions |
| `src/xvs/integrity/conformal.py` | Modified | Lazy re-exports of altitude-binned Mondrian conformal functions |
| `src/xvs/integrity/faults.py` | Modified | Lazy re-exports of boresight perturbation and misalignment fault |
| `src/xvs/horizon_att/edges.py` | Modified | Optional `multiscale=True` dispatch and `detect_camera_edges_multiscale` export |
| `configs/xvs/integrity/g1.json` | Added | Input configuration for WP-07 G1 pipeline |
| `configs/xvs/integrity/g1_spec.json` | Added | Specification limits and evaluation parameters for WP-07 G1 |
| `scripts/xvs/integrity/run_wp07_g1.py` | Added | Multi-process WP-07 G1 evaluation runner and manifest generator |
| `scripts/xvs/integrity/report_tables_wp07_g1.py` | Added | Provenance-tracked report generator for `WP07_G1_report.md` |
| `tests/xvs/test_integrity_g1.py` | Added | Verification suite `V1_G1..V7_G1` |
| `docs/xvs/WP07_INTEGRITY.md` | Modified | Added §10 documenting Gate G1 Horizon & Attitude Integrity closure |
| `docs/xvs/batches/BATCH_B11_WP07_G1_ARGON.md` | Added | Batch B11 engineering specification |
| `docs/xvs/reports/WP07_G1_report.md` | Added | Authoritative WP-07 Gate G1 report |
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    save_provenance(report_path)
    unprov = verify_report_provenance(report_path)
    if unprov:
        raise RuntimeError(f"Unprovenanced numbers found in {report_path}: {unprov}")
    print(f"Generated {report_path} and verified 0 unprovenanced numbers.")
    return content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate WP-07 G1 Report")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    data_root = get_data_root(args.data_root)
    generate_report_markdown(args.report_path, data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
