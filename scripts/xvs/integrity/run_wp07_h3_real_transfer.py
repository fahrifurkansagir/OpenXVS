r"""Batch B13 Producer — WP-06 / WP-07 Synthetic-to-Real Horizon & Conformal Transfer Study (`icehorizon_data`).

Executes the empirical evaluation of G0 vs G1 classical horizon/attitude estimation and
Charter §10 Hypothesis H3 (Synthetic-to-Real Conformal Uncertainty Transfer) across all
29 real annotated aerial drone and maritime video sequences (`580` stratified frames).

Usage:
    .\.venv\Scripts\python.exe scripts/xvs/integrity/run_wp07_h3_real_transfer.py --config configs/xvs/integrity/h3_real_transfer.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.integrity.real_transfer import (
    evaluate_h3_conformal_transfer,
    evaluate_single_real_frame,
    load_icehorizon_sequence_annotations,
    sample_stratified_frames,
    summarize_g0_vs_g1_benchmark,
    verify_icehorizon_allowlist,
)
from xvs.rig.rig import Rig


IGNORED_DIRTY_PATHS = {
    "configs/xvs/encounters/sampler_v1.json",
    "docs/phase2_evaluation_protocol.md",
    "src/icehorizon/stats.py",
    "src/icehorizon/visualizer.py",
    "tests/test_phase2_protocol_and_manifest.py",
}

CODE_FILES_TO_HASH = (
    "scripts/xvs/integrity/run_wp07_h3_real_transfer.py",
    "src/xvs/integrity/real_transfer.py",
    "src/xvs/integrity/g1_extensions.py",
    "src/xvs/integrity/conformal.py",
    "src/xvs/horizon_att/edges.py",
    "src/xvs/horizon_att/pipeline.py",
    "configs/xvs/integrity/h3_real_transfer.json",
    "docs/xvs/spec/data_allowlist.json",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in CODE_FILES_TO_HASH:
        p = REPO_ROOT / rel
        out[rel] = _sha256_file(p) if p.exists() else "MISSING"
    return out


def _get_git_commit() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def _get_git_status_tracked() -> list[str]:
    res = subprocess.run(
        ["git", "status", "--porcelain", "-uno"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    lines: list[str] = []
    for raw in res.stdout.splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = s.split(maxsplit=1)
        rel_path = parts[1].replace("\\", "/") if len(parts) > 1 else s
        if rel_path in IGNORED_DIRTY_PATHS:
            continue
        lines.append(s)
    return lines


def _render_figures(
    fig_dir: Path,
    records: list[dict[str, Any]],
    benchmark: dict[str, Any],
    h3_res: dict[str, Any],
    overlay_samples: list[dict[str, Any]],
) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    # Figure 1: G0 vs G1 Real-Imagery Horizon Error CDF & Polar Haze Stress Comparison
    fig1_path = fig_dir / "fig1_g0_vs_g1_real_horizon_errors.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=140)

    g0_errs = np.sort([float(r["g0"]["total_error_deg"]) for r in records])
    g1_errs = np.sort([float(r["g1"]["total_error_deg"]) for r in records])
    y_cdf = np.arange(1, len(records) + 1, dtype=np.float64) / float(len(records))

    ax0 = axes[0]
    ax0.plot(g0_errs, y_cdf, label=f"WP-06/07 G0 Single-Scale (med={benchmark['overall']['g0']['median_total_error_deg']:.2f}°, P90={benchmark['overall']['g0']['p90_total_error_deg']:.2f}°)", color="#d95f02", lw=2.0, ls="--")
    ax0.plot(g1_errs, y_cdf, label=f"WP-07 G1 Multi-Scale + Pseudo-Huber (med={benchmark['overall']['g1']['median_total_error_deg']:.2f}°, P90={benchmark['overall']['g1']['p90_total_error_deg']:.2f}°)", color="#1b9e77", lw=2.4)
    ax0.set_xlim(0.0, 4.0)
    ax0.set_ylim(0.0, 1.02)
    ax0.set_xlabel("Total Horizon Attitude Error (deg)")
    ax0.set_ylabel("Empirical CDF across 580 Real Frames")
    ax0.set_title("A. Real IceHorizon Attitude Error CDF (29 Videos, N=580)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower right", fontsize=8.5)

    ax1 = axes[1]
    categories = ["Raw Real\nConvergence", "Polar Haze (0.22x)\nConvergence", "Raw Real\nP90 Error (deg)", "Polar Haze (0.22x)\nMedian Error (deg)"]
    g0_vals = [
        benchmark["overall"]["g0"]["convergence_rate"] * 100.0,
        benchmark["overall"]["g0"]["hazy_convergence_rate"] * 100.0,
        benchmark["overall"]["g0"]["p90_total_error_deg"],
        benchmark["overall"]["g0"]["hazy_median_total_error_deg"],
    ]
    g1_vals = [
        benchmark["overall"]["g1"]["convergence_rate"] * 100.0,
        benchmark["overall"]["g1"]["hazy_convergence_rate"] * 100.0,
        benchmark["overall"]["g1"]["p90_total_error_deg"],
        benchmark["overall"]["g1"]["hazy_median_total_error_deg"],
    ]
    x_pos = np.arange(2)
    width = 0.34
    ax1.bar(x_pos - width / 2, g0_vals[:2], width, label="G0 Single-Scale", color="#d95f02", alpha=0.85)
    ax1.bar(x_pos + width / 2, g1_vals[:2], width, label="G1 Multi-Scale + Strip", color="#1b9e77", alpha=0.90)
    for idx_b in range(2):
        ax1.text(x_pos[idx_b] - width / 2, g0_vals[idx_b] + 1.5, f"{g0_vals[idx_b]:.1f}%", ha="center", fontsize=8.5)
        ax1.text(x_pos[idx_b] + width / 2, g1_vals[idx_b] + 1.5, f"{g1_vals[idx_b]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories[:2])
    ax1.set_ylim(0.0, 112.0)
    ax1.set_ylabel("Solver Convergence Rate (%)")
    ax1.set_title("B. Raw vs Low-Contrast Polar Haze Convergence (N=580)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(fig1_path)
    plt.close(fig)
    saved["fig1_g0_vs_g1_real_horizon_errors"] = str(fig1_path.relative_to(REPO_ROOT)).replace("\\", "/")

    # Figure 2: Charter Hypothesis H3 — Zero-Shot Synthetic vs Real-Recalibrated Conformal Coverage & PL
    fig2_path = fig_dir / "fig2_h3_conformal_calibration_curve.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=140)

    alphas_order = ["0.1", "0.05", "0.01"]
    nom_covs = [float(h3_res["by_alpha"][a]["target_coverage"]) * 100.0 for a in alphas_order]
    syn_covs = [float(h3_res["by_alpha"][a]["zero_shot_synthetic"]["empirical_coverage"]) * 100.0 for a in alphas_order]
    std_covs = [float(h3_res["by_alpha"][a]["real_recalibrated_standard"]["empirical_coverage"]) * 100.0 for a in alphas_order]
    mond_covs = [float(h3_res["by_alpha"][a]["real_recalibrated_mondrian"]["empirical_coverage"]) * 100.0 for a in alphas_order]

    ax0 = axes[0]
    x_idx = np.arange(len(alphas_order))
    w_bar = 0.25
    ax0.bar(x_idx - w_bar, syn_covs, w_bar, label="Zero-Shot Synthetic (`syn_cal`)", color="#e41a1c", alpha=0.85)
    ax0.bar(x_idx, std_covs, w_bar, label="Real Split-Conformal (`real_cal`)", color="#377eb8", alpha=0.88)
    ax0.bar(x_idx + w_bar, mond_covs, w_bar, label="Real Platform-Mondrian (`real_cal`)", color="#4daf4a", alpha=0.90)
    for i_a, nom in enumerate(nom_covs):
        ax0.hlines(nom, i_a - 0.4, i_a + 0.4, colors="black", linestyles=":", lw=1.6, label="Target $(1-\\alpha)$" if i_a == 0 else None)
        ax0.text(i_a - w_bar, syn_covs[i_a] + 1.5, f"{syn_covs[i_a]:.1f}%", ha="center", fontsize=8, color="#b00000", fontweight="bold")
        ax0.text(i_a, std_covs[i_a] + 1.5, f"{std_covs[i_a]:.1f}%", ha="center", fontsize=8)
        ax0.text(i_a + w_bar, mond_covs[i_a] + 1.5, f"{mond_covs[i_a]:.1f}%", ha="center", fontsize=8, fontweight="bold")

    ax0.set_xticks(x_idx)
    ax0.set_xticklabels([f"1-α = {nom:.0f}%\n(α={a})" for a, nom in zip(alphas_order, nom_covs)])
    ax0.set_ylim(0.0, 108.0)
    ax0.set_ylabel("Held-Out Real Test Coverage (%) [N=300, 15 Videos]")
    ax0.set_title("A. Hypothesis H3: Zero-Shot Synthetic vs Real-Recalibrated Coverage")
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend(loc="lower right", fontsize=8)

    ax1 = axes[1]
    syn_pls = [float(h3_res["by_alpha"][a]["zero_shot_synthetic"]["mean_pl_deg"]) for a in alphas_order]
    std_pls = [float(h3_res["by_alpha"][a]["real_recalibrated_standard"]["mean_pl_deg"]) for a in alphas_order]
    mond_pls = [float(h3_res["by_alpha"][a]["real_recalibrated_mondrian"]["mean_pl_deg"]) for a in alphas_order]
    ax1.bar(x_idx - w_bar, syn_pls, w_bar, label="Zero-Shot Synthetic Mean PL", color="#e41a1c", alpha=0.85)
    ax1.bar(x_idx, std_pls, w_bar, label="Real Split-Conformal Mean PL", color="#377eb8", alpha=0.88)
    ax1.bar(x_idx + w_bar, mond_pls, w_bar, label="Real Platform-Mondrian Mean PL", color="#4daf4a", alpha=0.90)
    for i_a in range(len(alphas_order)):
        ax1.text(i_a - w_bar, syn_pls[i_a] + 0.08, f"{syn_pls[i_a]:.2f}°", ha="center", fontsize=8)
        ax1.text(i_a, std_pls[i_a] + 0.08, f"{std_pls[i_a]:.2f}°", ha="center", fontsize=8)
        ax1.text(i_a + w_bar, mond_pls[i_a] + 0.08, f"{mond_pls[i_a]:.2f}°", ha="center", fontsize=8, fontweight="bold")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"1-α = {nom:.0f}%\n(α={a})" for a, nom in zip(alphas_order, nom_covs)])
    ax1.set_ylabel("Mean Protection Level (deg)")
    ax1.set_title("B. Mean Protection Level Scaling (Synthetic vs Real)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig2_path)
    plt.close(fig)
    saved["fig2_h3_conformal_calibration_curve"] = str(fig2_path.relative_to(REPO_ROOT)).replace("\\", "/")

    # Figure 3: 2x2 Real Frame Overlays (CVAT GT vs G1 Horizon Fit)
    fig3_path = fig_dir / "fig3_real_frame_horizon_overlays.png"
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6), dpi=130)
    for idx_s, sample in enumerate(overlay_samples[:4]):
        ax = axes[idx_s // 2, idx_s % 2]
        rgb_small = sample["rgb_small"]
        scale_x = float(sample["scale_x"])
        scale_y = float(sample["scale_y"])
        w_orig = float(sample["width"])
        rec = sample["record"]

        ax.imshow(rgb_small)
        u_vals = np.array([0.0, w_orig], dtype=np.float64)
        v_gt = rec["gt"]["m_gt"] * u_vals + rec["gt"]["c_gt"]
        v_g1 = rec["g1"]["m_est"] * u_vals + rec["g1"]["c_est"]

        ax.plot(u_vals * scale_x, v_gt * scale_y, color="#00ff44", lw=2.2, label="CVAT Human GT")
        ax.plot(u_vals * scale_x, v_g1 * scale_y, color="#00e5ff", lw=1.8, ls="--", label=f"G1 Fit (err={rec['g1']['total_error_deg']:.2f}°)")
        ax.set_title(
            f"{rec['seq_name']} (frame {rec['frame_id']}) | G0={rec['g0']['total_error_deg']:.2f}°, G1={rec['g1']['total_error_deg']:.2f}°, OOD={rec['ood']['ood_horizon_score']:.2f}",
            fontsize=9.5,
        )
        ax.set_axis_off()
        ax.legend(loc="lower right", fontsize=8, framealpha=0.75)

    fig.suptitle("IceHorizon (CC-BY 4.0) Real Video Frames: CVAT Ground-Truth Horizon vs WP-07 G1 Multi-Scale Fit", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig3_path)
    plt.close(fig)
    saved["fig3_real_frame_horizon_overlays"] = str(fig3_path.relative_to(REPO_ROOT)).replace("\\", "/")

    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Batch B13 WP-06/WP-07 Synthetic-to-Real Transfer Study (H3).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/xvs/integrity/h3_real_transfer.json",
        help="Path to H3 real transfer config JSON.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow running on a dirty git worktree (for pilot/debug runs only).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    git_status_start = _get_git_status_tracked()
    if git_status_start and not args.allow_dirty:
        raise RuntimeError(
            f"Refusing to run producer on dirty tracked worktree: {git_status_start}. "
            "Commit all changes first or pass --allow-dirty for pilot debugging."
        )

    code_hashes_start = _compute_code_hashes()
    commit_sha = _get_git_commit()

    cfg_path = REPO_ROOT / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    allowlist_info = verify_icehorizon_allowlist(REPO_ROOT / cfg["allowlist_path"], cfg["dataset_id"])

    syn_results_path = REPO_ROOT / cfg["synthetic_g1_results_path"]
    with syn_results_path.open("r", encoding="utf-8") as f:
        syn_g1_results = json.load(f)

    dataset_root = REPO_ROOT / cfg["dataset_root"]
    videos_dir = dataset_root / "videos"
    ann_dir = dataset_root / "Annotations"
    out_dir = REPO_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_seqs: list[str] = list(cfg["sequence_split"]["calibration_sequences"])
    test_seqs: list[str] = list(cfg["sequence_split"]["test_sequences"])
    all_seqs = sorted(cal_seqs + test_seqs)

    frames_per_seq = int(cfg["sampling"]["frames_per_sequence"])
    cam_proj = cfg["camera_projection"]
    fov_h_deg = float(cam_proj["fov_h_deg"])
    drone_h_m = float(cam_proj["drone_altitude_nom_m"])
    ship_h_m = float(cam_proj["ship_altitude_nom_m"])
    lat_nom = float(cam_proj["lat_nom_deg"])
    lon_nom = float(cam_proj["lon_nom_deg"])
    yaw_nom = float(cam_proj["yaw_nom_rad"])
    base_seed = int(cfg["rng_seed"])

    rig_base = Rig.load_nominal()
    rig_cache: dict[tuple[int, int], Rig] = {}

    video_files_by_stem: dict[str, Path] = {}
    for vp in sorted(videos_dir.iterdir()):
        if vp.is_file() and vp.suffix.lower() in (".mov", ".mp4"):
            video_files_by_stem[vp.stem] = vp

    records: list[dict[str, Any]] = []
    sequence_summaries: list[dict[str, Any]] = []
    overlay_targets = {"drone1", "ship1", "ship4", "ship16"}
    overlay_samples: list[dict[str, Any]] = []

    for seq_idx, seq_name in enumerate(all_seqs):
        if seq_name not in video_files_by_stem:
            raise FileNotFoundError(f"Video file for sequence {seq_name!r} not found in {videos_dir}")
        v_path = video_files_by_stem[seq_name]
        x_path = ann_dir / f"{seq_name}.xml"
        if not x_path.exists():
            raise FileNotFoundError(f"Annotation XML for sequence {seq_name!r} not found at {x_path}")

        anns = load_icehorizon_sequence_annotations(x_path)
        cap = cv2.VideoCapture(str(v_path))
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sampled = sample_stratified_frames(
            anns,
            frames_per_sequence=frames_per_seq,
            max_valid_frame_id=n_vid_frames if n_vid_frames > 0 else None,
        )

        rig_key = (w_vid, h_vid)
        if rig_key not in rig_cache:
            rig_cache[rig_key] = Rig(
                rig_id=f"real_{w_vid}x{h_vid}",
                cameras=rig_base.cameras,
                is_nominal=True,
                seed=0,
                width=w_vid,
                height=h_vid,
                fov_h_deg=fov_h_deg,
            )
        rig_seq = rig_cache[rig_key]
        h_nom_m = drone_h_m if seq_name.startswith("drone") else ship_h_m
        split_role = "real_cal" if seq_name in cal_seqs else "real_test"

        seq_recs: list[dict[str, Any]] = []
        sampled_by_fid = {int(item["frame_id"]): item for item in sampled}
        max_fid = max(sampled_by_fid.keys()) if sampled_by_fid else -1
        item_idx = 0
        for cur_fid in range(max_fid + 1):
            if not cap.grab():
                break
            if cur_fid not in sampled_by_fid:
                continue
            ok, bgr = cap.retrieve()
            if not ok or bgr is None:
                continue
            item = sampled_by_fid[cur_fid]
            fid = cur_fid
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_seed = base_seed + seq_idx * 1000 + fid
            rec = evaluate_single_real_frame(
                rgb=rgb,
                seq_name=seq_name,
                frame_id=fid,
                points=item["points"],
                rig=rig_seq,
                lat_deg=lat_nom,
                lon_deg=lon_nom,
                h_nom_m=h_nom_m,
                yaw_rad=yaw_nom,
                rng_seed=frame_seed,
                detector_params=cfg["detector_params"],
            )
            rec["split"] = split_role
            records.append(rec)
            seq_recs.append(rec)

            if seq_name in overlay_targets and item_idx == 0:
                small_w, small_h = 640, 360
                rgb_small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
                overlay_samples.append({
                    "seq_name": seq_name,
                    "rgb_small": rgb_small,
                    "width": w_vid,
                    "height": h_vid,
                    "scale_x": float(small_w) / float(w_vid),
                    "scale_y": float(small_h) / float(h_vid),
                    "record": rec,
                })
            item_idx += 1

        cap.release()

        g0_seq_errs = [float(r["g0"]["total_error_deg"]) for r in seq_recs]
        g1_seq_errs = [float(r["g1"]["total_error_deg"]) for r in seq_recs]
        sequence_summaries.append({
            "seq_name": seq_name,
            "platform": "drone" if seq_name.startswith("drone") else "ship",
            "split": split_role,
            "width": w_vid,
            "height": h_vid,
            "total_annotated_frames": len(anns),
            "sampled_frames": len(seq_recs),
            "g0_median_error_deg": float(np.median(g0_seq_errs)),
            "g1_median_error_deg": float(np.median(g1_seq_errs)),
            "g0_p90_error_deg": float(np.percentile(g0_seq_errs, 90)),
            "g1_p90_error_deg": float(np.percentile(g1_seq_errs, 90)),
        })

    benchmark = summarize_g0_vs_g1_benchmark(records)
    h3_transfer = evaluate_h3_conformal_transfer(
        records=records,
        synthetic_g1_results=syn_g1_results,
        cal_sequences=cal_seqs,
        test_sequences=test_seqs,
        conformal_params=cfg["conformal_params"],
    )

    fig_paths = _render_figures(
        fig_dir=out_dir / "figures",
        records=records,
        benchmark=benchmark,
        h3_res=h3_transfer,
        overlay_samples=overlay_samples,
    )

    # Verification checks V1_B13 .. V6_B13
    v1_pass = (
        allowlist_info["status"] == "ALLOWLIST_VERIFIED_EVALUATION_ONLY"
        and len(sequence_summaries) == 29
        and len(records) == 580
    )
    v2_pass = (
        h3_transfer["split_summary"]["sequence_overlap_count"] == 0
        and h3_transfer["split_summary"]["n_cal_sequences"] == 14
        and h3_transfer["split_summary"]["n_test_sequences"] == 15
        and h3_transfer["split_summary"]["n_cal_frames"] == 280
        and h3_transfer["split_summary"]["n_test_frames"] == 300
    )
    v3_pass = (
        benchmark["overall"]["g1"]["convergence_rate"] >= benchmark["overall"]["g0"]["convergence_rate"]
        and benchmark["overall"]["g1"]["hazy_convergence_rate"] > benchmark["overall"]["g0"]["hazy_convergence_rate"]
        and benchmark["overall"]["g1"]["median_total_error_deg"] <= benchmark["overall"]["g0"]["median_total_error_deg"]
        and benchmark["overall"]["g1"]["p90_total_error_deg"] < benchmark["overall"]["g0"]["p90_total_error_deg"]
    )
    v4_pass = (
        not h3_transfer["by_alpha"]["0.05"]["zero_shot_synthetic"]["transfers_without_recalibration"]
        and not h3_transfer["by_alpha"]["0.01"]["zero_shot_synthetic"]["transfers_without_recalibration"]
        and h3_transfer["by_alpha"]["0.05"]["zero_shot_synthetic"]["coverage_gap"] < -0.05
        and h3_transfer["by_alpha"]["0.01"]["zero_shot_synthetic"]["coverage_gap"] < -0.05
    )
    v5_pass = (
        bool(h3_transfer["by_alpha"]["0.1"]["real_recalibrated_mondrian"]["pass"])
        and bool(h3_transfer["by_alpha"]["0.05"]["real_recalibrated_mondrian"]["pass"])
        and bool(h3_transfer["by_alpha"]["0.01"]["real_recalibrated_mondrian"]["pass"])
        and bool(h3_transfer["hypothesis_h3_confirmed"])
    )

    code_hashes_end = _compute_code_hashes()
    git_status_end = _get_git_status_tracked()
    code_hashes_match = code_hashes_start == code_hashes_end
    v6_pass = bool(code_hashes_match and (len(git_status_end) == 0 or args.allow_dirty))

    overall_verdict = "PASS" if (v1_pass and v2_pass and v3_pass and v4_pass and v5_pass and v6_pass) else "FAIL"

    verification_matrix = {
        "V1_B13_allowlist_and_29_sequences": "PASS" if v1_pass else "FAIL",
        "V2_B13_sequence_disjoint_split_hygiene": "PASS" if v2_pass else "FAIL",
        "V3_B13_g1_vs_g0_real_horizon_improvement": "PASS" if v3_pass else "FAIL",
        "V4_B13_hypothesis_h3_synthetic_gap_measured": "PASS" if v4_pass else "FAIL",
        "V5_B13_real_conformal_recalibration_coverage": "PASS" if v5_pass else "FAIL",
        "V6_B13_clean_provenance_and_hashes": "PASS" if v6_pass else "FAIL",
        "overall_gate_verdict": overall_verdict,
    }

    elapsed_s = float(time.perf_counter() - t0)
    finished_utc = datetime.now(timezone.utc).isoformat()

    results_payload = {
        "schema": "xvs.wp07.h3_real_transfer_results/1",
        "commit": commit_sha,
        "config_path": args.config,
        "allowlist_verification": allowlist_info,
        "dataset_summary": {
            "total_sequences": len(sequence_summaries),
            "drone_sequences": sum(1 for s in sequence_summaries if s["platform"] == "drone"),
            "ship_sequences": sum(1 for s in sequence_summaries if s["platform"] == "ship"),
            "total_annotated_frames_in_dataset": sum(int(s["total_annotated_frames"]) for s in sequence_summaries),
            "sampled_frames_total": len(records),
            "frames_per_sequence": frames_per_seq,
            "sequences": sequence_summaries,
        },
        "g0_vs_g1_benchmark": benchmark,
        "h3_conformal_transfer": h3_transfer,
        "figures": fig_paths,
        "verification_matrix": verification_matrix,
        "overall_gate_verdict": overall_verdict,
        "elapsed_s": elapsed_s,
    }

    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    records_path = out_dir / "per_frame_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    manifest_payload = {
        "schema": "xvs.run_manifest/1",
        "batch": "B13",
        "wp": "WP-06/WP-07",
        "gate": "H3_real_transfer",
        "commit": commit_sha,
        "config_path": args.config,
        "config_sha256": _sha256_file(cfg_path),
        "synthetic_g1_results_path": cfg["synthetic_g1_results_path"],
        "synthetic_g1_results_sha256": _sha256_file(syn_results_path),
        "results_sha256": _sha256_file(results_path),
        "per_frame_records_sha256": _sha256_file(records_path),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_s": elapsed_s,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_status": git_status_end,
        "code_hashes_start": code_hashes_start,
        "code_hashes_end": code_hashes_end,
        "code_hashes_match": code_hashes_match,
    }
    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    print(
        f"[WP-07 H3 Real Transfer] DONE in {elapsed_s:.2f}s | "
        f"overall_gate_verdict={overall_verdict} | "
        f"G0 med={benchmark['overall']['g0']['median_total_error_deg']:.3f} deg (P90={benchmark['overall']['g0']['p90_total_error_deg']:.3f} deg, hazy_conv={benchmark['overall']['g0']['hazy_convergence_rate']*100:.1f}%) | "
        f"G1 med={benchmark['overall']['g1']['median_total_error_deg']:.3f} deg (P90={benchmark['overall']['g1']['p90_total_error_deg']:.3f} deg, hazy_conv={benchmark['overall']['g1']['hazy_convergence_rate']*100:.1f}%) | "
        f"H3 syn_cov(0.05)={h3_transfer['by_alpha']['0.05']['zero_shot_synthetic']['empirical_coverage']*100:.2f}% -> real_mond_cov(0.05)={h3_transfer['by_alpha']['0.05']['real_recalibrated_mondrian']['empirical_coverage']*100:.2f}%"
    )
    return 0 if overall_verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
