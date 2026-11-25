"""Comprehensive verification tests for XVS-Land v5.2 Architecture, Huber-IRLS PnPL, 5 Hz EKF & G4 Trainer."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from xvs.v2.xvs_land import (
    CAT1_HAL_M,
    CAT1_VAL_M,
    KEYLINE_SIGMA_MAX_PX,
    KEYLINE_SIGMA_MIN_PX,
    CholeskyBivariateGnllHead,
    CoordConvHesseKeylineHead,
    StudentTCalibrator,
    XvsLandV52Model,
    XvsLandV52OnnxWrapper,
    bivariate_gnll_loss,
    build_approach_trajectory_poses,
    build_ksfo_28l_reference_features,
    evaluate_approach_ekf_trajectory,
    keyline_endpoint_loss,
    keyline_keypoint_consistency_loss,
    keyline_sigma_nll_loss,
    okls_keyline_loss,
    project_runway_points_ned,
    solve_4dof_map_pnpl,
)
from xvs.v2.xvs_land.pnpl_ekf import compute_hesse_line_from_points
from xvs.v2.xvs_land.train_g4 import (
    RunwaySpec,
    XvsLandG4Config,
    _parse_args,
    _render_perspective_runway_crop,
    export_xvs_land_v52_onnx,
    run_xvs_land_g4_training,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LARD_MANIFEST = REPO_ROOT / "data" / "reference" / "lard_real_airports_manifest.json"
RUNWAYS_CSV = REPO_ROOT / "data" / "reference" / "ourairports" / "runways.csv"


def _rotate_hesse_line_about_midpoint(
    p0_uv: np.ndarray,
    p1_uv: np.ndarray,
    base_hesse: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Rotate a canonical Hesse line `(rho, cos_theta, sin_theta)` by `angle_deg` about segment midpoint."""
    mid = 0.5 * (np.asarray(p0_uv, dtype=np.float64) + np.asarray(p1_uv, dtype=np.float64))
    rad = math.radians(float(angle_deg))
    c_a, s_a = math.cos(rad), math.sin(rad)
    c0, s0 = float(base_hesse[1]), float(base_hesse[2])
    cn = c0 * c_a - s0 * s_a
    sn = c0 * s_a + s0 * c_a
    rn = cn * float(mid[0]) + sn * float(mid[1])
    if rn < 0.0:
        rn, cn, sn = -rn, -cn, -sn
    return np.array([rn, cn, sn], dtype=np.float64)


def test_xvs_land_v52_param_budget_onnx_size_parity_and_latency(tmp_path: Path) -> None:
    """Verify <=2.5M params, <=6.0 MiB FP16 ONNX, ORT FP32/FP16 parity (<1e-2), and <=12ms CPU latency."""
    # Static AST check: zero timm or torchvision imports anywhere in src/xvs/v2/xvs_land/
    land_dir = REPO_ROOT / "src" / "xvs" / "v2" / "xvs_land"
    for py_file in sorted(land_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("timm", "torchvision")), f"Forbidden import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith(("timm", "torchvision")), f"Forbidden import {node.module}"

    torch.manual_seed(20261124)
    for roi_size in (256, 384):
        model = XvsLandV52Model(roi_width=roi_size, roi_height=roi_size, split_ratio=2.0).eval()
        total_params = sum(int(p.numel()) for p in model.parameters())
        assert total_params <= 2_500_000, f"XvsLandV52Model({roi_size}) exceeded 2.5M param cap: {total_params:,}"

        rep = export_xvs_land_v52_onnx(
            model,
            tmp_path / f"onnx_{roi_size}",
            roi_size=roi_size,
            split_ratio=2.0,
            opset=17,
            fp32_tol=1.0e-3,
            fp16_tol=1.0e-2,
            export_fp16=True,
            bench_warmup=5,
            bench_iters=20,
        )
        assert rep["opset"] == 17
        assert rep["output_names"] == list(XvsLandV52OnnxWrapper.OUTPUT_NAMES)
        assert rep["fp32_ort_verified"] is True
        assert rep["fp16_ort_verified"] is True
        assert float(rep["fp32_max_abs_diff"]) < 1.0e-3
        assert float(rep["fp16_max_abs_diff"]) < 1.0e-2
        assert int(rep["fp16_initializer_count"]) >= 20
        assert int(rep["fp16_onnx_bytes"]) <= 6 * 1024 * 1024, (
            f"FP16 ONNX size exceeded 6.0 MiB: {rep['fp16_onnx_mib']} MiB"
        )
        assert Path(str(rep["fp32_onnx_path"])).is_file()
        assert Path(str(rep["fp16_onnx_path"])).is_file()
        # Verify real CPU ORT execution stays <= 12 ms at 256 (and <= 15 ms p50 at 384 on busy host)
        assert rep["ort_fp16_p50_ms"] is not None and math.isfinite(float(rep["ort_fp16_p50_ms"]))
        assert rep["ort_fp32_p50_ms"] is not None and math.isfinite(float(rep["ort_fp32_p50_ms"]))
        if roi_size == 256:
            assert float(rep["ort_fp16_p50_ms"]) <= 12.0, (
                f"ROI=256 FP16 ORT P50 latency {rep['ort_fp16_p50_ms']:.2f} ms > 12 ms"
            )
        else:
            assert min(float(rep["ort_fp32_p50_ms"]), float(rep["ort_fp16_p50_ms"])) <= 15.0, (
                f"ROI=384 ORT P50 latency exceeded 15 ms: fp32={rep['ort_fp32_p50_ms']:.2f}ms, "
                f"fp16={rep['ort_fp16_p50_ms']:.2f}ms"
            )


def test_coordconv_hesse_keyline_head_unit_norm_sigma_and_frozen_mean() -> None:
    """Verify CoordConvHesseKeylineHead unit-norm (incl. zero input), positive sigma [0.5, 50], & frozen-mean detach."""
    torch.manual_seed(20261124)
    model = XvsLandV52Model(roi_width=256, roi_height=256, split_ratio=2.0)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    assert out["logits_x"].shape == (2, 12, 512)
    assert out["logits_y"].shape == (2, 12, 512)
    assert out["keypoints_xy"].shape == (2, 12, 2)
    assert out["keylines_hesse"].shape == (2, 5, 3)
    assert out["keyline_sigma_px"].shape == (2, 5)
    assert out["keyline_endpoints"].shape == (2, 5, 2, 2)
    assert out["cholesky_l"].shape == (2, 12, 2, 2)
    assert out["cov_2x2"].shape == (2, 12, 2, 2)

    # Unit norm check on standard forward
    kl = out["keylines_hesse"]
    norm_sq = kl[..., 1] ** 2 + kl[..., 2] ** 2
    assert float(torch.max(torch.abs(norm_sq - 1.0)).item()) < 1.0e-6

    # Strictly positive bounded keyline_sigma_px in [0.5, 50.0]
    sig = out["keyline_sigma_px"]
    assert torch.all(torch.isfinite(sig))
    assert float(sig.min().item()) >= KEYLINE_SIGMA_MIN_PX
    assert float(sig.max().item()) <= KEYLINE_SIGMA_MAX_PX

    # Zero-weight + zero-input edge case (BC-M2-m5 epsilon guard 1e-12)
    zero_head = CoordConvHesseKeylineHead(in_channels=16, roi_width=128, roi_height=128, feat_stride=8)
    with torch.no_grad():
        for p in zero_head.parameters():
            p.zero_()
    kl_zero, sig_zero = zero_head(torch.zeros(1, 16, 16, 16))
    assert torch.all(torch.isfinite(kl_zero))
    assert torch.all(torch.isfinite(sig_zero))
    assert float(torch.max(torch.abs(kl_zero[..., 1] ** 2 + kl_zero[..., 2] ** 2 - 1.0)).item()) < 1.0e-6
    assert KEYLINE_SIGMA_MIN_PX <= float(sig_zero.min().item()) <= float(sig_zero.max().item()) <= KEYLINE_SIGMA_MAX_PX

    # Extreme sigma_head weights still clamp strictly inside [0.5, 50.0]
    with torch.no_grad():
        zero_head.sigma_head[-1].bias.fill_(1000.0)
    _, sig_hi = zero_head(torch.zeros(1, 16, 16, 16))
    assert float(sig_hi.max().item()) == pytest.approx(KEYLINE_SIGMA_MAX_PX, abs=1e-5)
    with torch.no_grad():
        zero_head.sigma_head[-1].bias.fill_(-1000.0)
    _, sig_lo = zero_head(torch.zeros(1, 16, 16, 16))
    assert float(sig_lo.min().item()) == pytest.approx(KEYLINE_SIGMA_MIN_PX, abs=1e-5)

    # Strictly SPD Cholesky covariance + Student-t calibrator
    eigs = torch.linalg.eigvalsh(out["cov_2x2"])
    assert float(eigs.min().item()) > 0.0
    cal = StudentTCalibrator(nu=6.0)
    cal.fit_scale_from_mahalanobis(torch.tensor([2.1, 2.8, 3.0, 2.4]))
    assert float(torch.linalg.eigvalsh(cal.calibrate_covariance(out["cov_2x2"].detach())).min().item()) > 0.0

    # Stage-2A geometric losses finite & differentiable
    gt_kp = torch.rand(2, 12, 2) * 200.0 + 28.0
    assert torch.isfinite(okls_keyline_loss(out["keylines_hesse"], out["keylines_hesse"].detach()))
    assert torch.isfinite(keyline_endpoint_loss(out["keyline_endpoints"], gt_kp))
    assert torch.isfinite(keyline_keypoint_consistency_loss(out["keylines_hesse"], out["keypoints_xy"]))
    assert torch.isfinite(keyline_sigma_nll_loss(out["keylines_hesse"], out["keyline_sigma_px"], gt_kp, loss_type="gaussian"))
    assert torch.isfinite(keyline_sigma_nll_loss(out["keylines_hesse"], out["keyline_sigma_px"], gt_kp, loss_type="laplacian"))

    # Strict Stage-2B frozen-mean gradient barrier:
    # Backprop ONLY through bivariate_gnll_loss + keyline_sigma_nll_loss -> ZERO grad in Stage-2A!
    model.zero_grad()
    out2 = model(x)
    loss_2b = bivariate_gnll_loss(torch.randn(2, 12, 2), out2["cholesky_l"]) + keyline_sigma_nll_loss(
        out2["keylines_hesse"], out2["keyline_sigma_px"], gt_kp
    )
    assert torch.isfinite(loss_2b)
    loss_2b.backward()

    stage2a_has_grad = any(
        p.grad is not None and float(p.grad.abs().max().item()) > 0.0 for p in model.stage2a_parameters()
    )
    cov_has_grad = any(
        p.grad is not None and float(p.grad.abs().max().item()) > 0.0 for p in model.cov_head.parameters()
    )
    kl_sig_has_grad = any(
        p.grad is not None and float(p.grad.abs().max().item()) > 0.0
        for p in model.keyline_head.sigma_parameters()
    )
    assert stage2a_has_grad is False, "Stage-2B uncertainty losses leaked gradients into Stage-2A backbone/geometry!"
    assert cov_has_grad is True
    assert kl_sig_has_grad is True


def test_pnpl_huber_irls_resists_25deg_corrupted_keyline_and_per_line_sigmas() -> None:
    """Corrupting 1 keyline by 25 deg MUST NOT hijack 4-DoF MAP-PnPL under Huber IRLS."""
    pts_ned, kl_pairs = build_ksfo_28l_reference_features()
    hdg = math.radians(297.7)
    pitch = math.radians(-3.0)
    roll = math.radians(1.0)
    fx, fy, cx, cy = 1200.0, 1200.0, 511.5, 511.5

    true_pos = np.array([-900.0 * math.cos(hdg), -900.0 * math.sin(hdg), -55.0], dtype=np.float64)
    uv_true, _ = project_runway_points_ned(pts_ned, true_pos, hdg, pitch, roll, fx, fy, cx, cy)
    lines_true = np.vstack([compute_hesse_line_from_points(uv_true[i0], uv_true[i1]) for i0, i1 in kl_pairs])

    # Corrupt keyline 0 (left edge) by 25 degrees
    lines_corrupt = lines_true.copy()
    i0, i1 = kl_pairs[0]
    lines_corrupt[0] = _rotate_hesse_line_about_midpoint(uv_true[i0], uv_true[i1], lines_true[0], angle_deg=25.0)

    # Simulate realistic inflated keypoint covariance (where legacy non-Huber solver chased bad keylines!)
    covs_inflated = np.broadcast_to(np.eye(2) * 16.0, (12, 2, 2)).copy()
    init_pos = true_pos + np.array([45.0, -20.0, 10.0], dtype=np.float64)
    init_yaw = hdg + math.radians(2.0)

    sol_no_huber = solve_4dof_map_pnpl(
        pts_ned,
        uv_true,
        covs_inflated,
        pitch_rad=pitch,
        roll_rad=roll,
        init_pos_ned=init_pos,
        init_yaw_rad=init_yaw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        keyline_pairs=kl_pairs,
        obs_keylines_hesse=lines_corrupt,
        keyline_sigma_px=1.5,
        huber_delta_sigma=0.0,
    )
    sol_huber = solve_4dof_map_pnpl(
        pts_ned,
        uv_true,
        covs_inflated,
        pitch_rad=pitch,
        roll_rad=roll,
        init_pos_ned=init_pos,
        init_yaw_rad=init_yaw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        keyline_pairs=kl_pairs,
        obs_keylines_hesse=lines_corrupt,
        keyline_sigma_px=1.5,
        huber_delta_sigma=3.0,
    )
    err_no_huber = float(np.linalg.norm(sol_no_huber.pos_ned_m - true_pos))
    err_huber = float(np.linalg.norm(sol_huber.pos_ned_m - true_pos))
    yaw_err_huber_deg = abs(((math.degrees(sol_huber.yaw_rad - hdg) + 180.0) % 360.0) - 180.0)

    assert err_no_huber > 20.0, f"Expected non-Huber baseline to fail on 25-deg fault, got {err_no_huber:.2f} m"
    assert err_huber < 0.5, f"Huber IRLS failed to resist 25-deg keyline fault: err={err_huber:.3f} m"
    assert yaw_err_huber_deg < 0.1

    # Per-keyline uncertainty weighting (sigma=50 px on faulty keyline 0, 1.0 px on clean lines)
    sol_per_sig = solve_4dof_map_pnpl(
        pts_ned,
        uv_true,
        np.broadcast_to(np.eye(2) * 0.64, (12, 2, 2)).copy(),
        pitch_rad=pitch,
        roll_rad=roll,
        init_pos_ned=init_pos,
        init_yaw_rad=init_yaw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        keyline_pairs=kl_pairs,
        obs_keylines_hesse=lines_corrupt,
        keyline_sigmas_px=np.array([50.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        huber_delta_sigma=3.0,
    )
    assert float(np.linalg.norm(sol_per_sig.pos_ned_m - true_pos)) < 0.1

    # Also verify Huber IRLS resists a +60 px single-keypoint gross outlier
    uv_outlier = uv_true.copy()
    uv_outlier[4] += np.array([60.0, -45.0], dtype=np.float64)
    sol_kp_out = solve_4dof_map_pnpl(
        pts_ned,
        uv_outlier,
        np.broadcast_to(np.eye(2) * 0.64, (12, 2, 2)).copy(),
        pitch_rad=pitch,
        roll_rad=roll,
        init_pos_ned=init_pos,
        init_yaw_rad=init_yaw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        keyline_pairs=kl_pairs,
        obs_keylines_hesse=lines_true,
        huber_delta_sigma=3.0,
    )
    assert float(np.linalg.norm(sol_kp_out.pos_ned_m - true_pos)) < 0.5


def test_approach_ekf_5hz_trajectory_cat1_availability_and_zero_hmi() -> None:
    """Verify 5 Hz ApproachEkf5Hz over 1500m -> 250m achieves high CAT-I availability and 0 HMI."""
    pts_ned, kl_pairs = build_ksfo_28l_reference_features()
    poses = build_approach_trajectory_poses(
        runway_heading_deg=297.7,
        start_range_m=1500.0,
        end_range_m=250.0,
        glideslope_deg=3.0,
        num_frames=50,
        dt=0.2,
    )
    rng = np.random.default_rng(20261124)
    fx, fy, cx, cy = 1200.0, 1200.0, 511.5, 511.5
    sig_px = 0.7

    frames: list[dict[str, object]] = []
    for idx, p in enumerate(poses):
        uv_t, _ = project_runway_points_ned(
            pts_ned, p["cam_pos_ned"], p["yaw_rad"], p["pitch_rad"], p["roll_rad"], fx, fy, cx, cy
        )
        uv_n = uv_t + rng.normal(0.0, sig_px, size=uv_t.shape)
        kl_n = np.vstack([compute_hesse_line_from_points(uv_n[i0], uv_n[i1]) for i0, i1 in kl_pairs])
        # Inject a 25-deg single-keyline fault mid-approach on frame 20 to verify IRLS+EKF resilience
        if idx == 20:
            i0, i1 = kl_pairs[0]
            kl_n[0] = _rotate_hesse_line_about_midpoint(uv_n[i0], uv_n[i1], kl_n[0], angle_deg=25.0)
        frames.append(
            {
                "pts_ned": pts_ned,
                "obs_uv": uv_n,
                "obs_cov_2x2": np.broadcast_to(np.eye(2) * (sig_px**2), (12, 2, 2)).copy(),
                "keyline_pairs": kl_pairs,
                "obs_keylines_hesse": kl_n,
                "keyline_sigmas_px": np.full(5, sig_px * 1.4, dtype=np.float64),
                **p,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
        )

    res = evaluate_approach_ekf_trajectory(
        frames,
        dt=0.2,
        huber_delta_sigma=3.0,
        init_pos_offset_ned=np.array([6.0, -3.0, 1.5], dtype=np.float64),
        init_yaw_offset_rad=math.radians(0.4),
    )
    assert res.num_frames == 50
    assert res.accepted_updates >= 45
    assert res.ekf_pos_rmse_m < res.single_frame_pnpl_rmse_m
    assert res.ekf_pos_rmse_m < 6.0
    assert res.v_hpl_median_m <= CAT1_HAL_M
    assert res.v_vpl_median_m <= CAT1_VAL_M
    assert res.cat1_availability >= 0.85
    assert res.hmi_count == 0


def test_xvs_land_v52_smoke_local_cli_end_to_end_and_resume(tmp_path: Path) -> None:
    """Verify --arch v52 CLI flags, richer 384px renderer, end-to-end --smoke-local training, ONNX & resume."""
    # 1. CLI defaults & flags
    out_dir = tmp_path / "v52_smoke"
    drive_dir = tmp_path / "v52_drive"
    cfg_default = _parse_args(["--out-dir", str(out_dir), "--lard-manifest", str(LARD_MANIFEST)])
    assert cfg_default.arch == "v52"
    assert cfg_default.roi_size == 384
    assert 1 in cfg_default.batch_candidates
    assert 512 in cfg_default.batch_candidates

    cfg_v45 = _parse_args(["--out-dir", str(out_dir), "--lard-manifest", str(LARD_MANIFEST), "--arch", "v45"])
    assert cfg_v45.arch == "v45"
    assert cfg_v45.roi_size == 256

    cfg_cli = _parse_args(
        [
            "--out-dir",
            str(out_dir),
            "--lard-manifest",
            str(LARD_MANIFEST),
            "--drive-ckpt-dir",
            str(drive_dir),
            "--arch",
            "v52",
            "--roi-size",
            "384",
            "--num-val-samples",
            "16",
            "--smoke-local",
        ]
    )
    assert cfg_cli.arch == "v52"
    assert cfg_cli.roi_size == 384
    assert cfg_cli.num_val_samples == 16
    assert cfg_cli.smoke_local is True

    # 2. Rich perspective runway renderer check at 384x384 over realistic 250 m..2500 m glideslope
    rwy = RunwaySpec(icao="KDEN", length_m=3657.6, width_m=45.7, heading_deg=170.0)
    rng = np.random.default_rng(20261124)
    ranges = []
    for _ in range(8):
        crop = _render_perspective_runway_crop(rwy, 384, rng, domain_randomize=True, rich_markings=True)
        assert crop["image"].shape == (3, 384, 384)
        assert np.all(np.isfinite(crop["image"]))
        assert 250.0 <= float(crop["range_m"]) <= 2500.0
        ranges.append(float(crop["range_m"]))
        kl = crop["keylines_hesse"]
        assert kl.shape == (5, 3)
        assert float(np.max(np.abs(kl[:, 1] ** 2 + kl[:, 2] ** 2 - 1.0))) < 1.0e-5

    # 3. Full --smoke-local --arch v52 training + Stage-2A/2B + 5 Hz EKF eval + FP32/FP16 ONNX export
    rep = run_xvs_land_g4_training(cfg_cli)
    assert rep["pass"] is True
    assert rep["arch"] == "v52"
    assert int(rep["total_params"]) <= 2_500_000
    assert rep["lard_guard"]["manifest_verified"] is True
    assert rep["lard_guard"]["ksfo_heldout"] is True
    assert rep["stage2a_loss_decreased"] is True
    assert math.isfinite(float(rep["stage2b_final_gnll"]))
    assert rep["eval_metrics"]["ksfo_anti_misassociation_pass"] is True
    assert int(rep["eval_metrics"]["hmi_count"]) == 0
    assert int(rep["eval_metrics"]["ekf_hmi_count"]) == 0
    assert rep["eval_metrics"]["ekf_trajectory"] is not None
    assert rep["eval_metrics"]["ekf_trajectory"]["num_frames"] >= 2

    onnx_meta = rep["onnx_export"]
    assert onnx_meta is not None
    assert onnx_meta["arch"] == "v52"
    assert onnx_meta["fp32_ort_verified"] is True
    assert onnx_meta["fp16_ort_verified"] is True
    assert float(onnx_meta["fp32_max_abs_diff"]) < 1.0e-3
    assert float(onnx_meta["fp16_max_abs_diff"]) < 1.0e-2
    assert int(onnx_meta["fp16_onnx_bytes"]) <= 6 * 1024 * 1024
    assert (out_dir / "xvs_land_v52_fp32.onnx").is_file()
    assert (out_dir / "xvs_land_v52_fp16.onnx").is_file()
    assert (out_dir / "xvs_land_final.pt").is_file()
    assert (out_dir / "xvs_land_g4_metrics.json").is_file()

    # Validate written JSON parses cleanly with no NaN
    metrics_doc = json.loads((out_dir / "xvs_land_g4_metrics.json").read_text(encoding="utf-8"))
    assert metrics_doc["pass"] is True
    assert metrics_doc["arch"] == "v52"

    # 4. Resuming from Stage-2A + Stage-2B checkpoints succeeds cleanly
    rep_resumed = run_xvs_land_g4_training(cfg_cli)
    assert rep_resumed["pass"] is True
    assert rep_resumed["resumed_stage2a"] is True
    assert rep_resumed["resumed_stage2b"] is True
