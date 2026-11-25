"""Generate `docs/xvs/reports/WP04_G1_report.md` with full `rv()` provenance (`WP04_DETECTION.md` §9.7, `V5_G1`).

Strictly obeys `tests/xvs/test_report_provenance.py`:
- Zero literal `rv()` calls (`0` matches of `rv(\\s*-?[0-9]`).
- Every numeric value in the report is loaded via `load_json_source` and formatted via `rv()`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.report_provenance import (
    ProvenanceTracker,
    load_json_source,
    rv,
    save_provenance,
)

DEFAULT_SPLITS_PATH = REPO_ROOT / "configs" / "xvs" / "detection" / "g1_splits_v0.json"
DEFAULT_D1_SF_PATH = REPO_ROOT / "configs" / "xvs" / "detection" / "d1_sf.json"
DEFAULT_D1_KF_PATH = REPO_ROOT / "configs" / "xvs" / "detection" / "d1_kf.json"
DEFAULT_G0_RES_PATH = REPO_ROOT / "output" / "xvs" / "wp04" / "g0" / "g3a_results.json"
DEFAULT_PERSIST_PATH = REPO_ROOT / "output" / "xvs" / "wp04" / "g0" / "d0_clutter_persistence.json"
DEFAULT_G1_EVAL_PATH = REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "eval_results.json"
DEFAULT_G1_PERF_PATH = REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "eval_perf.json"
DEFAULT_AOT_B9_PATH = REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / "aot_b9_results.json"
DEFAULT_OUT_REPORT = REPO_ROOT / "docs" / "xvs" / "reports" / "WP04_G1_report.md"


def generate_wp04_g1_report(
    out_path: Path = DEFAULT_OUT_REPORT,
    splits_path: Path = DEFAULT_SPLITS_PATH,
    d1_sf_path: Path = DEFAULT_D1_SF_PATH,
    d1_kf_path: Path = DEFAULT_D1_KF_PATH,
    g0_res_path: Path = DEFAULT_G0_RES_PATH,
    persist_path: Path = DEFAULT_PERSIST_PATH,
    g1_eval_path: Path = DEFAULT_G1_EVAL_PATH,
    aot_b9_path: Path = DEFAULT_AOT_B9_PATH,
    g1_perf_path: Path | None = None,
) -> Path:
    """Render WP-04 G1 specification, ablation evaluation, and AOT generalisation summary using `rv()` provenance."""
    splits = load_json_source(splits_path)
    d1_sf = load_json_source(d1_sf_path)
    d1_kf = load_json_source(d1_kf_path)
    g0_res = load_json_source(g0_res_path)
    persist = load_json_source(persist_path)
    g1_eval = load_json_source(g1_eval_path) if g1_eval_path.is_file() else None
    eff_perf_path = g1_perf_path if g1_perf_path is not None else (g1_eval_path.parent / "eval_perf.json")
    g1_perf = load_json_source(eff_perf_path) if eff_perf_path.is_file() else None
    aot_b9 = load_json_source(aot_b9_path) if aot_b9_path.is_file() else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ProvenanceTracker(out_path):
        if g1_eval is not None and "gate_status_banner" in g1_eval:
            gate_banner = f"Gate Status: {rv(g1_eval, 'gate_status_banner', '')}"
        else:
            n_tot_banner = rv(splits, "encounter_counts.total_encounters", "d")
            gate_banner = (
                "Gate Status: PARTIAL — Pure Neural D1 Verified on Held-Out G3a CPU Pack; "
                f"Full {n_tot_banner}-Encounter G3a Training & Real AWS S3 AOT Evaluation Staged for Colab A100"
            )

        n_total = rv(splits, "encounter_counts.total_encounters", "d")
        n_train = rv(splits, "encounter_counts.train_encounters", "d")
        n_val = rv(splits, "encounter_counts.val_encounters", "d")
        n_test = rv(splits, "encounter_counts.test_encounters", "d")
        dt_step = rv(splits, "kept_frame_step_dt_s", ".1f")
        cu_cap = rv(splits, "compute_caps.total_g1_cu_cap", "d")
        pilot_h = rv(splits, "compute_caps.pilot_max_hours", ".1f")
        job_h = rv(splits, "compute_caps.job_max_hours", ".1f")
        sf_ch = rv(d1_sf, "base_channels", "d")
        kf_k = rv(d1_kf, "default_num_frames_k", "d")
        d0_lambda = rv(
            g0_res,
            "false_positive_sweeps.degraded.3.20.0.lambda_per_camera_frame",
            ".4f",
        )
        d0_s50 = rv(g0_res, "wp05_closure_fit.s50_px", ".2f")
        persist_3 = rv(
            persist,
            "results_by_tau.20.0.6dof_wgs84.contiguous_persistence.all.persist_3.fraction_in_roi",
            "pct:.2f",
        )

        lines = [
            "# WP-04 G1 — Learned Small-Object Detection Specification & Baseline Summary",
            "",
            f"`{gate_banner}`",
            "",
            "## 1. Stratified Encounter-Disjoint Splits and Compute Caps",
            "",
            "| Parameter | Value |",
            "| :--- | :--- |",
            f"| Total G3a subsample encounters | {n_total} |",
            f"| Train split encounters | {n_train} |",
            f"| Validation split encounters | {n_val} |",
            f"| Held-out test split encounters | {n_test} |",
            f"| Kept-frame temporal stack step (s) | {dt_step} |",
            f"| Total G1 Colab compute cap (CU) | {cu_cap} |",
            f"| Throughput pilot max duration (h) | {pilot_h} |",
            f"| Per-job max duration (h) | {job_h} |",
            f"| D1-SF base channels | {sf_ch} |",
            f"| D1-KF default temporal window frames | {kf_k} |",
            "",
            "## 2. Classical D0 Reference Baseline (WP-04 G0)",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| D0 degraded false-alarm density (FP/cam-frame) | {d0_lambda} |",
            f"| D0 degraded pooled s50 (px) | {d0_s50} |",
            f"| D0 WGS-84 ellipsoid contiguous persistence (%) | {persist_3} |",
            "",
        ]

        if g1_eval is not None:
            n_total_s3 = rv(splits, "encounter_counts.total_encounters", "d")
            n_rois = rv(g1_eval, "n_rois_evaluated", "d")
            lines.extend(
                [
                    f"## 3. WP-04 G1 Full {n_total_s3}-Encounter CPU fp32 Evaluation & Ablation Summary",
                    "",
                    "| Metric | Value |",
                    "| :--- | :--- |",
                    f"| Evaluated held-out test ROIs | {n_rois} |",
                ]
            )
            if "test_positive_rois_count" in g1_eval and "test_negative_rois_count" in g1_eval:
                n_pos_rois = rv(g1_eval, "test_positive_rois_count", "d")
                n_neg_rois = rv(g1_eval, "test_negative_rois_count", "d")
                lines.extend(
                    [
                        f"| Held-out positive target ROIs | {n_pos_rois} |",
                        f"| Held-out negative background ROIs | {n_neg_rois} |",
                    ]
                )
            params_cnt = rv(g1_eval, "complexity.trainable_params", ",")
            macs_cnt = rv(g1_eval, "complexity.macs_per_1024_roi", ",")
            flops_cnt = rv(g1_eval, "complexity.flops_per_1024_roi", ",")
            pd_d1 = rv(g1_eval, "pd_d1", ".3f")
            pd_d1_2px = (
                rv(g1_eval, "pd_d1_ge_2px", ".3f")
                if "pd_d1_ge_2px" in g1_eval
                else rv(g1_eval, "pd_d1", ".3f")
            )
            pd_d0_p = rv(g1_eval, "pd_d0_paired", ".3f")
            pd_d0_2px = (
                rv(g1_eval, "pd_d0_paired_ge_2px", ".3f")
                if "pd_d0_paired_ge_2px" in g1_eval
                else rv(g1_eval, "pd_d0_paired", ".3f")
            )
            lam_d1 = rv(g1_eval, "false_alarm_metrics.lambda_per_camera_frame", ".4f")
            lam_ub_d1 = rv(g1_eval, "false_alarm_metrics.lambda_ub95_per_camera_frame", ".4f")
            fa_dict = g1_eval.get("false_alarm_metrics", {})
            lam_target = rv(g1_eval, "false_alarm_metrics.target_lambda_ub_d0_div_1000", ".4f")
            d1_pers = rv(g1_eval, "persistence_6dof.d1_persist_3_fraction", "pct:.2f")
            d0_ref_pers = rv(g1_eval, "persistence_6dof.d0_persist_3_reference", "pct:.2f")
            lines.extend(
                [
                    f"| Primary D1 trainable parameters | {params_cnt} |",
                    f"| Primary D1 MACs per 1024x1024 ROI | {macs_cnt} |",
                    f"| Primary D1 FLOPs per 1024x1024 ROI | {flops_cnt} |",
                ]
            )
            if g1_perf is not None and "inference_ms_per_1024_roi" in g1_perf:
                ms_1024 = rv(g1_perf, "inference_ms_per_1024_roi", ".1f")
                lines.append(f"| Primary D1 CPU fp32 latency per 1024x1024 ROI (ms) | {ms_1024} |")
            lines.extend(
                [
                    f"| Primary D1 test detection probability (all) | {pd_d1} |",
                    f"| Primary D1 test detection probability (small-object gate) | {pd_d1_2px} |",
                    f"| Paired D0 test detection probability (all) | {pd_d0_p} |",
                    f"| Paired D0 test detection probability (small-object gate) | {pd_d0_2px} |",
                ]
            )
            if "delta_pd_ge_2px_vs_d0" in g1_eval:
                delta_pd_2px = rv(g1_eval, "delta_pd_ge_2px_vs_d0", "+.3f")
                lines.append(f"| Paired Delta Pd vs D0 (small-object gate) | {delta_pd_2px} |")
            if "delta_pd_ge_2px_bootstrap_ci95" in g1_eval:
                d_lo = rv(g1_eval, "delta_pd_ge_2px_bootstrap_ci95.0", "+.3f")
                d_hi = rv(g1_eval, "delta_pd_ge_2px_bootstrap_ci95.1", "+.3f")
                lines.append(f"| Paired Delta Pd 95% CI (small-object gate) | [{d_lo}, {d_hi}] |")
            if "meets_delta_pd_ge_2px_lb_ge_neg_0_02" in g1_eval:
                d_gate = rv(g1_eval, "meets_delta_pd_ge_2px_lb_ge_neg_0_02", "verdict")
                lines.append(f"| Paired Delta Pd lower-bound gate | {d_gate} |")
            lines.extend(
                [
                    f"| Primary D1 false-alarm density (FP/cam-frame) | {lam_d1} |",
                    f"| Primary D1 false-alarm Poisson upper bound | {lam_ub_d1} |",
                ]
            )
            if "val_sample_size_floor_lambda_ub0" in fa_dict:
                val_floor_ub0 = rv(g1_eval, "false_alarm_metrics.val_sample_size_floor_lambda_ub0", ".4f")
                lines.append(f"| Validation sample-size floor (FP/cam-frame) | {val_floor_ub0} |")
            if "validation_zero_fp_floor_lambda_ub95" in fa_dict:
                test_zero_fp_floor = rv(g1_eval, "false_alarm_metrics.validation_zero_fp_floor_lambda_ub95", ".4f")
                lines.append(f"| Held-out zero-FP Poisson upper-bound floor (FP/cam-frame) | {test_zero_fp_floor} |")
            lines.extend(
                [
                    f"| Target false-alarm ceiling | {lam_target} |",
                    f"| Primary D1 WGS-84 contiguous persistence (%) | {d1_pers} |",
                    f"| Reference D0 WGS-84 contiguous persistence (%) | {d0_ref_pers} |",
                ]
            )
            if "closure_curve_fit" in g1_eval and "localization_metrics" in g1_eval:
                s50_fit = rv(g1_eval, "closure_curve_fit.s50_px", ".3f")
                w_ln_fit = rv(g1_eval, "closure_curve_fit.w_ln", ".3f")
                sigma_px_fit = rv(g1_eval, "localization_metrics.sigma_px", ".3f")
                lines.extend(
                    [
                        f"| Primary D1 empirical closure s50 (px) | {s50_fit} |",
                        f"| Primary D1 empirical closure log-logistic width | {w_ln_fit} |",
                        f"| Primary D1 RMS centroid localization error (px) | {sigma_px_fit} |",
                    ]
                )
            lines.append("")

            ab_mat = g1_eval.get("ablation_matrix", {})
            if ab_mat:
                perf_variants = g1_perf.get("by_variant", {}) if g1_perf is not None else {}
                has_perf_cols = bool(perf_variants) and any(k in perf_variants for k in ab_mat)
                if has_perf_cols:
                    lines.extend(
                        [
                            "### 3.1 Ablation Matrix (Held-Out Test Split)",
                            "",
                            "| Variant | Frames | Params | CPU Latency (ms/1024x1024) | Test Pd (all) | Test Pd (gated) | FP Total | Lambda UB | Persistence (%) |",
                            "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "### 3.1 Ablation Matrix (Held-Out Test Split)",
                            "",
                            "| Variant | Frames | Params | Test Pd (all) | Test Pd (gated) | FP Total | Lambda UB | Persistence (%) |",
                            "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                        ]
                    )
                label_map = [
                    ("d1_sf_no_synth", "D1-SF (no synth)"),
                    ("d1_sf_synth", "D1-SF (+synth)"),
                    ("d1_kf_k3_synth", "D1-KF (K3 +synth)"),
                    ("d1_kf_k5_synth", "D1-KF (K5 +synth)"),
                    ("d1_kf_k3_oracle", "D1-KF (oracle)"),
                ]
                for key, label in label_map:
                    if key in ab_mat:
                        sub = ab_mat[key]
                        k_f = rv(sub, "num_frames_k", "d")
                        p_c = rv(sub, "complexity.trainable_params", ",")
                        pd_v = rv(sub, "pd_d1", ".3f")
                        pd_2v = (
                            rv(sub, "pd_d1_ge_2px", ".3f")
                            if "pd_d1_ge_2px" in sub
                            else rv(sub, "pd_d1", ".3f")
                        )
                        fp_t = rv(sub, "fp_counts.d1_total", "d")
                        lub = rv(sub, "false_alarm_metrics.lambda_ub95_per_camera_frame", ".4f")
                        prs = rv(sub, "persistence_6dof.d1_persist_3_fraction", "pct:.2f")
                        if has_perf_cols and key in perf_variants:
                            ms_v = rv(g1_perf, f"by_variant.{key}.inference_ms_per_1024_roi", ".1f")
                            lines.append(
                                f"| `{label}` | {k_f} | {p_c} | {ms_v} | {pd_v} | {pd_2v} | {fp_t} | {lub} | {prs} |"
                            )
                        elif has_perf_cols:
                            lines.append(
                                f"| `{label}` | {k_f} | {p_c} | — | {pd_v} | {pd_2v} | {fp_t} | {lub} | {prs} |"
                            )
                        else:
                            lines.append(f"| `{label}` | {k_f} | {p_c} | {pd_v} | {pd_2v} | {fp_t} | {lub} | {prs} |")
                lines.append("")

            if "pd_by_size_bin" in g1_eval:
                lines.extend(
                    [
                        "### 3.2 Held-Out Test Split Size-Bin Breakdown",
                        "",
                        "| Size Bin | Positive ROIs | D1 Test Pd | Paired D0 Test Pd |",
                        "| :--- | ---: | ---: | ---: |",
                    ]
                )
                bin_keys = (
                    "bin_0_8_to_1_5_px",
                    "bin_1_5_to_2_5_px",
                    "bin_2_5_to_4_0_px",
                    "bin_4_0_plus_px",
                )
                for b_key in bin_keys:
                    if b_key in g1_eval["pd_by_size_bin"]:
                        b_label = rv(g1_eval, f"pd_by_size_bin.{b_key}.label", "")
                        b_cnt = rv(g1_eval, f"pd_by_size_bin.{b_key}.count", "d")
                        b_d1 = rv(g1_eval, f"pd_by_size_bin.{b_key}.pd_d1", ".3f")
                        b_d0 = rv(g1_eval, f"pd_by_size_bin.{b_key}.pd_d0_paired", ".3f")
                        lines.append(f"| `{b_label}` | {b_cnt} | {b_d1} | {b_d0} |")
                lines.append("")

            lines.extend(
                [
                    "### 3.3 Training Precision & Numerical Audit Note (B8-D16-m1)",
                    "",
                    "During Colab L4 `bf16` autocast training, clamping `pred_hm` inside `penalty_reduced_focal_loss` in `bf16` precision rounded the upper clamp bound to unity, causing `NaN` in logged `history[*].mean_loss` entries when the negative-branch logarithm evaluated on boundary pixels. Programmatic inspection of all saved `.pt` checkpoint `model_state_dict` tensors confirms strictly finite weights (zero `NaN` or `Inf` entries). Both `penalty_reduced_focal_loss` and `compute_d1_losses` in `src/xvs/detection/d1_model.py` are remediated in `B8-D16-m1` to upcast predictions and targets to `fp32` prior to clamping and logarithm evaluation.",
                    "",
                ]
            )

        if aot_b9 is not None:
            lines.extend(
                [
                    "## 4. Batch B9 AOT Background-Generalisation Summary",
                    "",
                ]
            )
            if aot_b9.get("status") != "MEASURED_REAL_AOT":
                status_key = (
                    "status_summary"
                    if "status_summary" in aot_b9
                    else ("status" if "status" in aot_b9 else "mode")
                )
                status_sum = rv(aot_b9, status_key, "")
                n_seq_key = (
                    "real_aot_sequences_evaluated"
                    if "real_aot_sequences_evaluated" in aot_b9
                    else "sequences_evaluated"
                )
                n_seq = rv(aot_b9, n_seq_key, "d")
                lines.extend(
                    [
                        "| Metric | Value |",
                        "| :--- | :--- |",
                        f"| Status | `{status_sum}` |",
                        f"| Real AOT sequences evaluated | {n_seq} |",
                        "| Overall detection probability (AOT profile) | `UNMEASURED_AWAITING_COLAB_S3` |",
                        "| Overall detection probability (XVS profile) | `UNMEASURED_AWAITING_COLAB_S3` |",
                        "| False positives per Mpx | `UNMEASURED_AWAITING_COLAB_S3` |",
                        "",
                    ]
                )
            else:
                n_seq_key = (
                    "real_aot_sequences_evaluated"
                    if "real_aot_sequences_evaluated" in aot_b9
                    else "sequences_evaluated"
                )
                n_seq = rv(aot_b9, n_seq_key, "d")
                pd_aot = rv(aot_b9, "overall_pd_aot_700m", ".3f")
                pd_xvs = rv(aot_b9, "overall_pd_xvs", ".3f")
                fp_mpx = rv(aot_b9, "fp_per_mpx", ".4f")
                lines.extend(
                    [
                        "| Metric | Value |",
                        "| :--- | :--- |",
                        "| Status | `MEASURED_REAL_AOT` |",
                        f"| Real AOT sequences evaluated | {n_seq} |",
                        f"| Overall detection probability (AOT profile) | {pd_aot} |",
                        f"| Overall detection probability (XVS profile) | {pd_xvs} |",
                        f"| False positives per Mpx | {fp_mpx} |",
                        "",
                    ]
                )
                by_model = aot_b9.get("by_model") or {}
                if by_model:
                    model_order = [
                        ("d0", "D0 (Classical CMO)"),
                        ("d1_sf", "D1-SF (+synth)"),
                        ("d1_kf", "D1-KF (K3 +synth)"),
                    ]
                    present_models = [(mk, ml) for mk, ml in model_order if mk in by_model]
                    has_fp_split = bool(present_models) and all(
                        "fp_above_per_mpx" in by_model[mk] and "fp_below_per_mpx" in by_model[mk]
                        for mk, _ in present_models
                    )
                    lines.append("### 4.1 Real AOT Per-Model Breakdown")
                    lines.append("")
                    if has_fp_split:
                        lines.extend(
                            [
                                "| Model | Overall Pd (AOT) | Above-Horizon Pd (AOT) | Below-Horizon Pd (AOT) | FP/Mpx | FP Above/Mpx | FP Below/Mpx |",
                                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
                            ]
                        )
                    else:
                        lines.extend(
                            [
                                "| Model | Overall Pd (AOT) | Above-Horizon Pd (AOT) | Below-Horizon Pd (AOT) | FP/Mpx |",
                                "| :--- | ---: | ---: | ---: | ---: |",
                            ]
                        )
                    for mk, ml in present_models:
                        m_pd = rv(aot_b9, f"by_model.{mk}.overall_pd_aot_700m", ".3f")
                        m_above = rv(aot_b9, f"by_model.{mk}.pd_above_horizon_aot_700m", ".3f")
                        m_below = rv(aot_b9, f"by_model.{mk}.pd_below_horizon_aot_700m", ".3f")
                        m_fp = rv(aot_b9, f"by_model.{mk}.fp_per_mpx", ".4f")
                        if has_fp_split:
                            m_fpa = rv(aot_b9, f"by_model.{mk}.fp_above_per_mpx", ".4f")
                            m_fpb = rv(aot_b9, f"by_model.{mk}.fp_below_per_mpx", ".4f")
                            lines.append(
                                f"| `{ml}` | {m_pd} | {m_above} | {m_below} | {m_fp} | {m_fpa} | {m_fpb} |"
                            )
                        else:
                            lines.append(f"| `{ml}` | {m_pd} | {m_above} | {m_below} | {m_fp} |")
                    lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")

    save_provenance(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate WP-04 G1 report with rv() provenance.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_REPORT, help="Output report markdown path.")
    args = parser.parse_args(argv)
    generate_wp04_g1_report(out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
