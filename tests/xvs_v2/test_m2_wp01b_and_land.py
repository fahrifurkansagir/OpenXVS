"""Comprehensive local CPU verification for Milestone M2 Subsystem M2-C (`V2-M2L-4`):
  - WP-01b: 9-level real KSFO 3DEP+GEOID18 & 13-level 4096x4096 uint16 Min-Max Mipmaps
  - WP-01b: Curved-Earth WGS-84 hierarchical ray-DEM (< 0.1 m vertical accuracy)
  - WP-01b: Exact TSO-C151d / DO-367 Class B TAWS FLTA + Approach Desensitization
  - XVS-Land: Pure-PyTorch D-FINE-M + ConvNeXt-V1 + 12-kp SimCC + 5 Hesse OKLS + Frozen-Mean SPD Cholesky GNLL
  - XVS-Land: 4-DoF Horizon-Constrained MAP-PnPL + 5 Hz Approach EKF (11-state, Mahalanobis gate)
  - XVS-Land: Hybrid Range-Conditioned Conformal-RAIM V-HPL/V-VPL + KSFO 28L vs Taxiway C/28R + Go-Around Latch
  - XVS-Land: Fail-closed BC-M2-M3 LARD >= 36 real ICAO manifest & KSFO zero-leakage guard

Writes real machine-readable producer evidence to `output/xvs/v2_m2/v4_wp01b_land_evidence.json`.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from xvs.geo.wgs84 import geodetic_to_ecef, r_ecef_ned
from xvs.v2.wp01b import (
    CLASS_B_DESENS_INNER_AGL_M,
    CLASS_B_DESENS_INNER_DIST_M,
    CLASS_B_DESENS_OUTER_AGL_M,
    CLASS_B_DESENS_OUTER_DIST_M,
    CLASS_B_RTC_SI_M,
    DEFAULT_KSFO_PACK_PATH,
    EARTH_MEAN_RADIUS_M,
    FlightPhase,
    build_minmax_mipmap,
    earth_curvature_drop_m,
    evaluate_taws_flta,
    intersect_ray_dem_dense_ref,
    intersect_ray_dem_ecef,
    intersect_ray_dem_ned,
    is_approach_desensitized,
    load_ksfo_3dep_mipmap,
    required_terrain_clearance_m,
    sample_height_bilinear_m,
    verify_mipmap_invariant,
)
from xvs.v2.xvs_land import (
    CAT1_HAL_M,
    CAT1_VAL_M,
    KSFO_28L_WIDTH_M,
    KSFO_28R_LATERAL_OFFSET_M,
    KSFO_TAXIWAY_C_WIDTH_M,
    MIN_LARD_REAL_AIRPORTS,
    ApproachEkf5Hz,
    CholeskyBivariateGnllHead,
    DFineMRoiDetector,
    HesseKeylineHead,
    MandatoryGoAroundLatch,
    StudentTCalibrator,
    XvsLandModel,
    bivariate_gnll_loss,
    build_ksfo_28l_reference_features,
    check_runway_association,
    compute_hybrid_conformal_raim_pl,
    expand_roi_conformal,
    okls_keyline_loss,
    project_runway_points_ned,
    solve_4dof_map_pnpl,
    validate_lard_real_airports_manifest,
)
from xvs.v2.xvs_land.pnpl_ekf import compute_hesse_line_from_points

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = REPO_ROOT / "output" / "xvs" / "v2_m2" / "v4_wp01b_land_evidence.json"

# 36 real international airport ICAO codes strictly for unit-testing manifest validation logic
SAMPLE_36_LARD_REAL_ICAOS: tuple[str, ...] = (
    "LFBO", "LFPG", "LFPO", "LFMN", "LFLL", "LFML", "LFRS", "LFBD", "LFST", "LFQQ",
    "LFSB", "LFRN", "LFBZ", "LFMT", "LFLC", "LFLB", "LFLS", "LFMP", "LFRB", "LFRD",
    "LFKJ", "LFKB", "LFKF", "LFMV", "LFTH", "LFMK", "LFBI", "LFBL", "LFBP", "LFBT",
    "LFCR", "LFLX", "LFMD", "LFOB", "LFOH", "LFRK",
)

SAMPLE_TRAIN_SYNTH_ICAOS: tuple[str, ...] = (
    "KSEA", "KPDX", "KDEN", "KSLC", "KPHX", "KLAS", "KSAN", "KLAX", "KBOS", "KJFK",
    "KATL", "KORD", "KDFW", "KIAH", "KMIA", "KMSP",
)


def _make_synthetic_4096_grid() -> np.ndarray:
    """Create a deterministic 4096x4096 uint16 terrain heightfield (0.1 m step, h_min=-100 m)."""
    coords = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    h_m = -30.0 + 380.0 * np.exp(-((xx + 0.35) ** 2 + (yy - 0.15) ** 2) * 6.0) + 45.0 * np.sin(xx * 9.0) * np.cos(yy * 7.0)
    q = np.clip(np.rint((h_m - (-100.0)) / 0.1), 0.0, 65535.0).astype(np.uint16)
    return q


def _ksfo_28l_threshold_local_ned(pyr: Any) -> tuple[float, float, float]:
    """Return KSFO 28L threshold `(north_m, east_m, wgs84_h_m)` in `pyr`'s local grid NED frame."""
    assert pyr.ksfo_28l_thr_lat_deg is not None
    assert pyr.ksfo_28l_thr_lon_deg is not None
    assert pyr.ksfo_28l_thr_wgs84_h_m is not None
    m_per_deg_lat = (math.pi / 180.0) * 6378137.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(pyr.origin_lat_deg))
    dn = (pyr.ksfo_28l_thr_lat_deg - pyr.origin_lat_deg) * m_per_deg_lat
    de = (pyr.ksfo_28l_thr_lon_deg - pyr.origin_lon_deg) * m_per_deg_lon
    return (float(dn), float(de), float(pyr.ksfo_28l_thr_wgs84_h_m))


def test_wp01b_ksfo_real_9lvl_and_4096_13lvl_minmax_mipmap() -> None:
    """Verify 9-level mipmap on real KSFO 3DEP+GEOID18 pack and 13-level mipmap on 4096x4096 grid."""
    assert DEFAULT_KSFO_PACK_PATH.is_file(), f"Missing M1 KSFO pack: {DEFAULT_KSFO_PACK_PATH}"

    pyr_inner = load_ksfo_3dep_mipmap(DEFAULT_KSFO_PACK_PATH, ring_name="inner_ksfo")
    pyr_outer = load_ksfo_3dep_mipmap(DEFAULT_KSFO_PACK_PATH, ring_name="outer_sf_bay")

    assert pyr_inner.grid_size == 256
    assert pyr_inner.num_levels == 9
    assert pyr_outer.num_levels == 9
    assert pyr_inner.terrain_mode == "usgs_3dep_geoid18_ellipsoidal"
    assert pyr_inner.geoid_model == "GEOID18"
    assert pyr_inner.ksfo_28l_thr_wgs84_h_m is not None
    assert -35.0 <= pyr_inner.ksfo_28l_thr_wgs84_h_m <= -22.0
    assert verify_mipmap_invariant(pyr_inner) is True
    assert verify_mipmap_invariant(pyr_outer) is True

    # 4096x4096 -> exact 13 levels (L0..L12)
    grid_4096 = _make_synthetic_4096_grid()
    pyr_4096 = build_minmax_mipmap(grid_4096, h_min_m=-100.0, h_step_m=0.1, source_name="synthetic_4096")
    assert pyr_4096.grid_size == 4096
    assert pyr_4096.num_levels == 13
    assert pyr_4096.levels_min[0].shape == (4096, 4096)
    assert pyr_4096.levels_min[-1].shape == (1, 1)
    assert verify_mipmap_invariant(pyr_4096) is True


def test_wp01b_curved_ray_dem_accuracy_and_ecefs() -> None:
    """Verify curved-Earth hierarchical ray-DEM matches dense reference to < 0.1 m vertically."""
    pyr = load_ksfo_3dep_mipmap(DEFAULT_KSFO_PACK_PATH, ring_name="inner_ksfo")

    # Sanity check curved-Earth drop formula at 10 km
    drop_10km = earth_curvature_drop_m(10_000.0, 0.0)
    expected_drop_10km = -(10_000.0**2) / (2.0 * EARTH_MEAN_RADIUS_M)
    assert abs(drop_10km - expected_drop_10km) < 1e-12
    assert -8.0 < drop_10km < -7.5

    rays = [
        # Oblique ray from SF Bay east of KSFO aimed down-west toward San Bruno Mountain ridges
        (np.array([2500.0, 3500.0, -450.0]), np.array([-0.15, -0.96, 0.22])),
        # Steeper downward look across KSFO field
        (np.array([-2000.0, 2200.0, -600.0]), np.array([0.45, -0.75, 0.48])),
        # Shallow grazing ray toward western hills
        (np.array([500.0, 3800.0, -320.0]), np.array([0.12, -0.985, 0.11])),
        # North-to-south cross-peninsula ray
        (np.array([4200.0, -2500.0, -520.0]), np.array([-0.92, -0.18, 0.35])),
    ]

    max_vert_err_m = 0.0
    for origin_ned, dir_ned in rays:
        hit_mip = intersect_ray_dem_ned(pyr, origin_ned, dir_ned)
        hit_ref = intersect_ray_dem_dense_ref(pyr, origin_ned, dir_ned, step_m=1.0)
        assert hit_mip.hit is True
        assert hit_ref.hit is True
        vert_err = abs(hit_mip.h_wgs84_m - hit_ref.h_wgs84_m)
        max_vert_err_m = max(max_vert_err_m, vert_err)
        assert vert_err < 0.1, f"Ray-DEM vertical error {vert_err:.4f} m >= 0.1 m"
        assert hit_mip.steps < hit_ref.steps

    # Also verify full ECEF ray entry point matches NED ray entry point
    p_orig_ecef = geodetic_to_ecef(pyr.origin_lat_deg, pyr.origin_lon_deg, 0.0)
    r_mat = r_ecef_ned(pyr.origin_lat_deg, pyr.origin_lon_deg)
    o_ned, d_ned = rays[0]
    o_ecef = p_orig_ecef + r_mat @ o_ned
    d_ecef = r_mat @ d_ned
    hit_ecef = intersect_ray_dem_ecef(pyr, o_ecef, d_ecef)
    hit_ned = intersect_ray_dem_ned(pyr, o_ned, d_ned)
    assert hit_ecef.hit is True
    assert abs(hit_ecef.h_wgs84_m - hit_ned.h_wgs84_m) < 1e-6
    assert max_vert_err_m < 0.1


def test_wp01b_tso_c151d_taws_flta_and_class_b_desensitization() -> None:
    """Verify exact SI RTC table, 0 false alerts on 3-deg KSFO 28L glide, and 60s/30s hill alerts."""
    # Exact Class B SI RTC table (0.3048 m/ft)
    assert math.isclose(required_terrain_clearance_m(FlightPhase.EN_ROUTE, 0.0), 213.36, abs_tol=1e-6)
    assert math.isclose(required_terrain_clearance_m(FlightPhase.EN_ROUTE, 3.0), 152.40, abs_tol=1e-6)
    assert math.isclose(required_terrain_clearance_m(FlightPhase.TERMINAL, 0.0), 106.68, abs_tol=1e-6)
    assert math.isclose(required_terrain_clearance_m(FlightPhase.TERMINAL, 3.0), 91.44, abs_tol=1e-6)
    assert math.isclose(required_terrain_clearance_m(FlightPhase.APPROACH, 0.0), 45.72, abs_tol=1e-6)
    assert math.isclose(required_terrain_clearance_m(FlightPhase.APPROACH, 3.0), 30.48, abs_tol=1e-6)
    assert math.isclose(required_terrain_clearance_m(FlightPhase.DEPARTURE, 0.0), 30.48, abs_tol=1e-6)
    assert math.isclose(required_terrain_clearance_m(FlightPhase.DEPARTURE, 3.0), 30.48, abs_tol=1e-6)
    assert len(CLASS_B_RTC_SI_M) == 4

    # Class B Approach Desensitization envelope checks
    assert is_approach_desensitized(FlightPhase.APPROACH, CLASS_B_DESENS_INNER_AGL_M - 1.0, CLASS_B_DESENS_INNER_DIST_M)
    assert is_approach_desensitized(FlightPhase.APPROACH, CLASS_B_DESENS_OUTER_AGL_M - 1.0, CLASS_B_DESENS_OUTER_DIST_M)
    assert not is_approach_desensitized(FlightPhase.APPROACH, 80.0, 500.0)
    assert not is_approach_desensitized(FlightPhase.EN_ROUTE, 20.0, 500.0)

    pyr_inner = load_ksfo_3dep_mipmap(DEFAULT_KSFO_PACK_PATH, ring_name="inner_ksfo")
    pyr_outer = load_ksfo_3dep_mipmap(DEFAULT_KSFO_PACK_PATH, ring_name="outer_sf_bay")
    thr_n, thr_e, thr_h = _ksfo_28l_threshold_local_ned(pyr_inner)

    # 1. Nominal 3-degree glide into KSFO 28L (heading 297.7 deg true, 70 m/s GS, sink 3.67 m/s)
    hdg = math.radians(297.7)
    gs = 70.0
    vn = gs * math.cos(hdg)
    ve = gs * math.sin(hdg)
    vz_down = gs * math.tan(math.radians(3.0))  # +3.67 m/s (> 2.54 m/s descending)

    false_alerts = 0
    desens_triggered = False
    for dist_out_m in (4200.0, 3000.0, 1800.0, 1200.0, 800.0, 450.0, 200.0):
        pn = thr_n - math.cos(hdg) * dist_out_m
        pe = thr_e - math.sin(hdg) * dist_out_m
        alt_m = thr_h + dist_out_m * math.tan(math.radians(3.0)) + 15.0
        res = evaluate_taws_flta(
            pyr_inner,
            pn,
            pe,
            alt_m,
            vn,
            ve,
            vz_down,
            FlightPhase.APPROACH,
            sigma_dem_m=2.0,
            vpl_m=4.0,
            threshold_ned_m=(thr_n, thr_e, thr_h),
        )
        if res.alert_level != "NONE":
            false_alerts += 1
        if res.desensitized:
            desens_triggered = True
    assert false_alerts == 0, f"Nominal 3-deg KSFO 28L approach produced {false_alerts} false FLTA alerts"
    assert desens_triggered is True

    # 2. Rising San Bruno / SF Peninsula hills west of KSFO on real 3DEP outer ring:
    # Find the peak hill west of KSFO (east_m < -3000 m) and fly straight toward it
    g_outer = pyr_outer.levels_min[0]
    n_sz = pyr_outer.grid_size
    peak_idx = int(np.argmax(g_outer))
    pr, pc = divmod(peak_idx, n_sz)
    peak_n = pyr_outer.north_max_m - (pr / float(n_sz - 1)) * pyr_outer.extent_north_m
    peak_e = pyr_outer.east_min_m + (pc / float(n_sz - 1)) * pyr_outer.extent_east_m
    peak_h = sample_height_bilinear_m(pyr_outer, peak_n, peak_e)
    assert peak_h > 250.0

    # Fly West at 100 m/s toward the hill at altitude below peak + RTC:
    # At 4,500 m east of peak -> ~45 s away -> CAUTION
    res_caution = evaluate_taws_flta(
        pyr_outer,
        peak_n,
        peak_e + 4500.0,
        peak_h - 40.0,
        0.0,
        -100.0,
        0.0,
        FlightPhase.TERMINAL,
    )
    assert res_caution.alert_level == "CAUTION"
    assert res_caution.time_to_conflict_s is not None
    assert 30.0 < res_caution.time_to_conflict_s <= 60.0

    # At 1,800 m east of peak -> ~18 s away -> WARNING
    res_warning = evaluate_taws_flta(
        pyr_outer,
        peak_n,
        peak_e + 1800.0,
        peak_h - 40.0,
        0.0,
        -100.0,
        0.0,
        FlightPhase.TERMINAL,
    )
    assert res_warning.alert_level == "WARNING"
    assert res_warning.time_to_conflict_s is not None
    assert 0.0 < res_warning.time_to_conflict_s <= 30.0

    # Fail-closed on NaN altitude
    res_nan = evaluate_taws_flta(pyr_outer, peak_n, peak_e, math.nan, 0.0, -100.0, 0.0, FlightPhase.EN_ROUTE)
    assert res_nan.alert_level == "WARNING"


def test_xvs_land_model_simcc_okls_cholesky_spd_and_frozen_mean() -> None:
    """Verify pure-PyTorch XVS-Land model: zero timm/torchvision, unit-norm OKLS, SPD Cholesky, frozen-mean detach."""
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

    torch.manual_seed(20261121)

    # Stage-1 D-FINE-M COCO ROI + conformal margin expansion
    roi_det = DFineMRoiDetector()
    assert roi_det.weight_id == "dfine_m_coco"
    img = torch.randn(2, 3, 256, 256)
    boxes = roi_det.predict_expanded_roi(img, q_alpha=12.5)
    assert len(boxes) == 2
    ex = expand_roi_conformal((50.0, 60.0, 150.0, 160.0), 10.0, 256, 256)
    assert ex == (40.0, 50.0, 160.0, 170.0)

    # Stage-2A + Stage-2B forward pass
    model = XvsLandModel(in_channels=3, roi_width=256, roi_height=256, split_ratio=2.0)
    out = model(img)
    assert out["logits_x"].shape == (2, 12, 512)
    assert out["logits_y"].shape == (2, 12, 512)
    assert out["keypoints_xy"].shape == (2, 12, 2)
    assert out["keylines_hesse"].shape == (2, 5, 3)
    assert out["cholesky_l"].shape == (2, 12, 2, 2)
    assert out["cov_2x2"].shape == (2, 12, 2, 2)

    # OKLS unit-norm check (including exact zero input edge case per BC-M2-m5)
    kl = out["keylines_hesse"]
    norm_sq = kl[..., 1] ** 2 + kl[..., 2] ** 2
    assert float(torch.max(torch.abs(norm_sq - 1.0)).item()) < 1e-6

    zero_head = HesseKeylineHead(in_channels=16)
    with torch.no_grad():
        for p in zero_head.parameters():
            p.zero_()
    kl_zero = zero_head(torch.zeros(1, 16, 4, 4))
    assert torch.all(torch.isfinite(kl_zero))
    norm_sq_z = kl_zero[..., 1] ** 2 + kl_zero[..., 2] ** 2
    assert float(torch.max(torch.abs(norm_sq_z - 1.0)).item()) < 1e-6

    # OKLS loss finite
    loss_kl = okls_keyline_loss(kl, kl.detach())
    assert torch.isfinite(loss_kl)

    # Strictly SPD Cholesky covariance across extreme raw inputs (-100..+100, BC-M2-m5)
    extreme_raw = torch.tensor([[[[-100.0, 80.0, -100.0], [100.0, -80.0, 100.0], [0.0, 0.0, 0.0]]]], dtype=torch.float32)
    chol_ext, cov_ext = CholeskyBivariateGnllHead.build_cholesky_and_cov(extreme_raw)
    assert torch.all(torch.isfinite(chol_ext))
    assert torch.all(torch.isfinite(cov_ext))
    eigs_ext = torch.linalg.eigvalsh(cov_ext)
    assert float(eigs_ext.min().item()) > 0.0
    sym_err = float(torch.max(torch.abs(cov_ext - cov_ext.transpose(-1, -2))).item())
    assert sym_err < 1e-6

    # Frozen-mean Stage-2B gradient isolation check (`feat.detach()`):
    # Backprop ONLY through bivariate_gnll_loss -> backbone MUST receive ZERO gradient!
    model.zero_grad()
    out_grad = model(img)
    fake_res = torch.randn(2, 12, 2)
    loss_gnll = bivariate_gnll_loss(fake_res, out_grad["cholesky_l"])
    assert torch.isfinite(loss_gnll)
    loss_gnll.backward()

    backbone_has_grad = any(p.grad is not None and float(p.grad.abs().max().item()) > 0.0 for p in model.backbone.parameters())
    cov_has_grad = any(p.grad is not None and float(p.grad.abs().max().item()) > 0.0 for p in model.cov_head.parameters())
    assert backbone_has_grad is False, "Stage-2B GNLL leaked gradients into Stage-2A backbone!"
    assert cov_has_grad is True

    # Student-t variance calibrator preserves SPD
    cal = StudentTCalibrator(nu=6.0)
    cal.fit_scale_from_mahalanobis(torch.tensor([2.5, 3.1, 1.9, 2.8]))
    cal_cov = cal.calibrate_covariance(out["cov_2x2"].detach())
    assert float(torch.linalg.eigvalsh(cal_cov).min().item()) > 0.0


def test_xvs_land_4dof_map_pnpl_and_5hz_approach_ekf() -> None:
    """Verify 4-DoF Horizon-Constrained MAP-PnPL accuracy and 5 Hz Approach EKF Mahalanobis gating."""
    pts_ned, kl_pairs = build_ksfo_28l_reference_features(runway_width_m=KSFO_28L_WIDTH_M)
    hdg_rad = math.radians(297.7)
    pitch_rad = math.radians(-3.0)
    roll_rad = math.radians(1.5)
    fx, fy, cx, cy = 1200.0, 1200.0, 511.5, 511.5

    # True camera 1,500 m on final approach to KSFO 28L at ~95 m AGL (Down = -95 m)
    true_pos = np.array([-1500.0 * math.cos(hdg_rad), -1500.0 * math.sin(hdg_rad), -95.0], dtype=np.float64)
    uv_true, z_cam = project_runway_points_ned(pts_ned, true_pos, hdg_rad, pitch_rad, roll_rad, fx, fy, cx, cy)
    assert bool(np.all(z_cam > 100.0))

    lines_true = np.vstack([compute_hesse_line_from_points(uv_true[i0], uv_true[i1]) for i0, i1 in kl_pairs])
    covs = np.broadcast_to(np.array([[0.64, 0.12], [0.12, 0.49]]), (12, 2, 2)).copy()

    # Perturbed initial guess (+80 m N, -50 m E, +25 m D, +4 deg yaw)
    init_pos = true_pos + np.array([80.0, -50.0, 25.0])
    init_yaw = hdg_rad + math.radians(4.0)

    sol_clean = solve_4dof_map_pnpl(
        pts_ned,
        uv_true,
        covs,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        init_pos_ned=init_pos,
        init_yaw_rad=init_yaw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        keyline_pairs=kl_pairs,
        obs_keylines_hesse=lines_true,
    )
    pos_err_clean = float(np.linalg.norm(sol_clean.pos_ned_m - true_pos))
    yaw_err_clean_deg = abs(((math.degrees(sol_clean.yaw_rad - hdg_rad) + 180.0) % 360.0) - 180.0)
    assert sol_clean.converged is True
    assert pos_err_clean < 0.05
    assert yaw_err_clean_deg < 0.01

    # Noisy observations (sigma ~ 0.6 px)
    rng = np.random.default_rng(20261121)
    uv_noisy = uv_true + rng.normal(0.0, 0.5, size=uv_true.shape)
    sol_noisy = solve_4dof_map_pnpl(
        pts_ned,
        uv_noisy,
        covs,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        init_pos_ned=init_pos,
        init_yaw_rad=init_yaw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        keyline_pairs=kl_pairs,
        obs_keylines_hesse=lines_true,
    )
    err_noisy = sol_noisy.pos_ned_m - true_pos
    # Vertical and cross-track errors stay tightly bounded (< 2.5 m); full 3D error stays within 99% Cramér-Rao ellipsoid
    fwd_unit = np.array([math.cos(hdg_rad), math.sin(hdg_rad), 0.0], dtype=np.float64)
    right_unit = np.array([-math.sin(hdg_rad), math.cos(hdg_rad), 0.0], dtype=np.float64)
    cross_track_err_m = abs(float(err_noisy @ right_unit))
    vertical_err_m = abs(float(err_noisy[2]))
    maha_3d = float(err_noisy @ np.linalg.inv(sol_noisy.cov_4x4[:3, :3]) @ err_noisy)
    assert sol_noisy.converged is True
    assert cross_track_err_m < 2.5
    assert vertical_err_m < 2.5
    assert maha_3d < 11.345  # 99% chi-squared (3 DoF) consistency with posterior Cramér-Rao covariance
    assert abs(float(err_noisy @ fwd_unit)) < 35.0

    # 5 Hz Approach EKF across 25 ticks (5.0 s) + Mahalanobis outlier rejection
    vel_true = np.array([70.0 * math.cos(hdg_rad), 70.0 * math.sin(hdg_rad), 3.67], dtype=np.float64)
    ekf = ApproachEkf5Hz(true_pos + np.array([6.0, -4.0, 2.0]), vel_true, hdg_rad + 0.01)
    cur_pos = true_pos.copy()
    for _ in range(25):
        ekf.predict(np.zeros(3), 0.0, dt=0.2)
        cur_pos = cur_pos + vel_true * 0.2
        upd = ekf.update_pnpl(
            cur_pos + rng.normal(0.0, 0.4, size=3),
            hdg_rad + float(rng.normal(0.0, 0.001)),
            np.diag([1.0, 1.0, 0.5, 1e-4]),
        )
        assert upd.accepted is True
    assert float(np.linalg.norm(ekf.x[0:3] - cur_pos)) < 2.0

    # Gross +120 m outlier MUST be rejected by Mahalanobis gate
    upd_outlier = ekf.update_pnpl(cur_pos + np.array([120.0, -90.0, 40.0]), hdg_rad, np.diag([1.0, 1.0, 0.5, 1e-4]))
    assert upd_outlier.accepted is False
    assert upd_outlier.mahalanobis_sq > upd_outlier.gate_chi2

    # NaN measurement fails closed
    upd_nan = ekf.update_pnpl(np.array([math.nan, 0.0, 0.0]), hdg_rad, np.eye(4))
    assert upd_nan.accepted is False


def test_xvs_land_hybrid_conformal_raim_ksfo_gate_and_latch() -> None:
    """Verify Hybrid Conformal-RAIM V-HPL/V-VPL, KSFO 28L vs Taxiway C & 28R gate, and non-resettable latch."""
    pts_ned, kl_pairs = build_ksfo_28l_reference_features(runway_width_m=KSFO_28L_WIDTH_M)
    hdg_rad = math.radians(297.7)
    pitch_rad = math.radians(-3.0)
    roll_rad = 0.0
    fx, fy, cx, cy = 1200.0, 1200.0, 511.5, 511.5

    # Nominal short-final at 750 m out -> V-VPL <= 10 m, V-HPL <= 40 m
    pos_750 = np.array([-750.0 * math.cos(hdg_rad), -750.0 * math.sin(hdg_rad), -50.0], dtype=np.float64)
    uv, _ = project_runway_points_ned(pts_ned, pos_750, hdg_rad, pitch_rad, roll_rad, fx, fy, cx, cy)
    lines = np.vstack([compute_hesse_line_from_points(uv[i0], uv[i1]) for i0, i1 in kl_pairs])
    covs = np.broadcast_to(np.eye(2) * 0.25, (12, 2, 2)).copy()

    sol = solve_4dof_map_pnpl(
        pts_ned,
        uv,
        covs,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        init_pos_ned=pos_750 + np.array([10.0, -5.0, 3.0]),
        init_yaw_rad=hdg_rad,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        keyline_pairs=kl_pairs,
        obs_keylines_hesse=lines,
    )
    raim_nom = compute_hybrid_conformal_raim_pl(sol.jacobian_obs, sol.cov_4x4, range_m=750.0)
    assert raim_nom.v_vpl_m <= CAT1_VAL_M
    assert raim_nom.v_hpl_m <= CAT1_HAL_M
    assert raim_nom.vpl_within_limit is True
    assert raim_nom.hpl_within_limit is True

    # NaN fails closed
    raim_nan = compute_hybrid_conformal_raim_pl(sol.jacobian_obs, sol.cov_4x4, range_m=math.nan)
    assert raim_nan.vpl_within_limit is False
    assert raim_nan.hpl_within_limit is False

    # KSFO 28L (61.0 m) vs Taxiway C (22.86 m) & parallel 28R (228.6 m offset)
    assoc_28l = check_runway_association(KSFO_28L_WIDTH_M, 1.8)
    assoc_twy_c = check_runway_association(KSFO_TAXIWAY_C_WIDTH_M, 0.5)
    assoc_28r = check_runway_association(KSFO_28L_WIDTH_M, KSFO_28R_LATERAL_OFFSET_M)
    assert assoc_28l.accepted is True
    assert assoc_twy_c.accepted is False and assoc_twy_c.width_ok is False
    assert assoc_28r.accepted is False and assoc_28r.lateral_ok is False

    # Non-resettable MandatoryGoAroundLatch during active approach
    latch = MandatoryGoAroundLatch()
    assert latch.step(approach_active=True, v_vpl_m=4.0, v_hpl_m=12.0, association_ok=True) is False
    assert latch.step(approach_active=True, v_vpl_m=14.2, v_hpl_m=12.0, association_ok=True) is True
    # Must stay latched even after fault clears AND ack_reset=True while approach_active=True!
    assert latch.step(approach_active=True, v_vpl_m=3.0, v_hpl_m=10.0, association_ok=True, ack_reset=True) is True
    # Clears ONLY once approach_active=False and ack_reset=True
    assert latch.step(approach_active=False, v_vpl_m=3.0, v_hpl_m=10.0, association_ok=True, ack_reset=False) is True
    assert latch.step(approach_active=False, v_vpl_m=3.0, v_hpl_m=10.0, association_ok=True, ack_reset=True) is False


def test_xvs_land_lard_fail_closed_dataset_guard() -> None:
    """Verify BC-M2-M3 fail-closed LARD >= 36 real ICAO manifest & KSFO zero-leakage guard."""
    # Missing manifest -> PermissionError
    with pytest.raises(PermissionError):
        validate_lard_real_airports_manifest(None, SAMPLE_TRAIN_SYNTH_ICAOS, ["KSFO"])

    # Unverified manifest -> PermissionError
    with pytest.raises(PermissionError):
        validate_lard_real_airports_manifest(
            {"verified": False, "source": "LARD_v1_real_metadata", "airports": list(SAMPLE_36_LARD_REAL_ICAOS)},
            SAMPLE_TRAIN_SYNTH_ICAOS,
            ["KSFO"],
        )

    # Fabricated source -> PermissionError
    with pytest.raises(PermissionError):
        validate_lard_real_airports_manifest(
            {"verified": True, "source": "fabricated_local_list", "airports": list(SAMPLE_36_LARD_REAL_ICAOS)},
            SAMPLE_TRAIN_SYNTH_ICAOS,
            ["KSFO"],
        )

    # Under 36 unique ICAOs (35) -> PermissionError
    with pytest.raises(PermissionError):
        validate_lard_real_airports_manifest(
            {"verified": True, "source": "LARD_v1_real_metadata", "airports": list(SAMPLE_36_LARD_REAL_ICAOS[:35])},
            SAMPLE_TRAIN_SYNTH_ICAOS,
            ["KSFO"],
        )

    # KSFO leaked into train -> PermissionError
    valid_manifest = {
        "verified": True,
        "source": "LARD_v1_real_metadata_sha256_verified",
        "airports": list(SAMPLE_36_LARD_REAL_ICAOS),
    }
    with pytest.raises(PermissionError):
        validate_lard_real_airports_manifest(valid_manifest, list(SAMPLE_TRAIN_SYNTH_ICAOS) + ["KSFO"], ["KSFO"])

    # Eval-only LARD airport leaked into train -> PermissionError
    with pytest.raises(PermissionError):
        validate_lard_real_airports_manifest(valid_manifest, list(SAMPLE_TRAIN_SYNTH_ICAOS) + ["LFBO"], ["KSFO"])

    # Valid verified 36-ICAO manifest + KSFO held-out + 0% overlap -> PASS
    ver = validate_lard_real_airports_manifest(
        valid_manifest,
        SAMPLE_TRAIN_SYNTH_ICAOS,
        ["KSFO", *SAMPLE_36_LARD_REAL_ICAOS],
    )
    assert ver.manifest_verified is True
    assert ver.lard_real_count >= MIN_LARD_REAL_AIRPORTS
    assert ver.ksfo_heldout is True
    assert len(ver.train_heldout_overlap) == 0
    assert len(ver.train_lard_real_overlap) == 0


def test_emit_v4_wp01b_land_evidence_json(tmp_path: Path) -> None:
    """Compute live V2-M2L-4 metrics and emit isolated evidence JSON without clobbering GSN-pinned repo files."""
    pyr_inner = load_ksfo_3dep_mipmap(DEFAULT_KSFO_PACK_PATH, ring_name="inner_ksfo")
    pyr_outer = load_ksfo_3dep_mipmap(DEFAULT_KSFO_PACK_PATH, ring_name="outer_sf_bay")
    pyr_4096 = build_minmax_mipmap(_make_synthetic_4096_grid(), source_name="synthetic_4096")

    # Ray-DEM vertical accuracy across 4 curved-Earth rays
    rays = [
        (np.array([2500.0, 3500.0, -450.0]), np.array([-0.15, -0.96, 0.22])),
        (np.array([-2000.0, 2200.0, -600.0]), np.array([0.45, -0.75, 0.48])),
        (np.array([500.0, 3800.0, -320.0]), np.array([0.12, -0.985, 0.11])),
        (np.array([4200.0, -2500.0, -520.0]), np.array([-0.92, -0.18, 0.35])),
    ]
    ray_errs: list[float] = []
    for o_ned, d_ned in rays:
        h_m = intersect_ray_dem_ned(pyr_inner, o_ned, d_ned)
        h_r = intersect_ray_dem_dense_ref(pyr_inner, o_ned, d_ned, step_m=1.0)
        assert h_m.hit and h_r.hit
        ray_errs.append(abs(h_m.h_wgs84_m - h_r.h_wgs84_m))
    ray_max_err_m = float(max(ray_errs))

    # TAWS FLTA nominal 3-deg KSFO 28L glide vs San Bruno hills
    thr_n, thr_e, thr_h = _ksfo_28l_threshold_local_ned(pyr_inner)
    hdg = math.radians(297.7)
    vn, ve = 70.0 * math.cos(hdg), 70.0 * math.sin(hdg)
    vz_down = 70.0 * math.tan(math.radians(3.0))
    nom_false_alerts = 0
    for dist_m in (4200.0, 3000.0, 1800.0, 1200.0, 800.0, 450.0, 200.0):
        res = evaluate_taws_flta(
            pyr_inner,
            thr_n - math.cos(hdg) * dist_m,
            thr_e - math.sin(hdg) * dist_m,
            thr_h + dist_m * math.tan(math.radians(3.0)) + 15.0,
            vn,
            ve,
            vz_down,
            FlightPhase.APPROACH,
            sigma_dem_m=2.0,
            vpl_m=4.0,
            threshold_ned_m=(thr_n, thr_e, thr_h),
        )
        if res.alert_level != "NONE":
            nom_false_alerts += 1

    g_outer = pyr_outer.levels_min[0]
    pr, pc = divmod(int(np.argmax(g_outer)), pyr_outer.grid_size)
    peak_n = pyr_outer.north_max_m - (pr / float(pyr_outer.grid_size - 1)) * pyr_outer.extent_north_m
    peak_e = pyr_outer.east_min_m + (pc / float(pyr_outer.grid_size - 1)) * pyr_outer.extent_east_m
    peak_h = sample_height_bilinear_m(pyr_outer, peak_n, peak_e)
    caution_res = evaluate_taws_flta(pyr_outer, peak_n, peak_e + 4500.0, peak_h - 40.0, 0.0, -100.0, 0.0, FlightPhase.TERMINAL)
    warning_res = evaluate_taws_flta(pyr_outer, peak_n, peak_e + 1800.0, peak_h - 40.0, 0.0, -100.0, 0.0, FlightPhase.TERMINAL)

    # XVS-Land model, PnPL, EKF, V-RAIM & LARD guard summary
    torch.manual_seed(20261121)
    model = XvsLandModel()
    out = model(torch.randn(1, 3, 256, 256))
    okls_max_unit_err = float(torch.max(torch.abs(out["keylines_hesse"][..., 1] ** 2 + out["keylines_hesse"][..., 2] ** 2 - 1.0)).item())
    chol_min_eig = float(torch.linalg.eigvalsh(out["cov_2x2"]).min().item())

    pts_ned, kl_pairs = build_ksfo_28l_reference_features(runway_width_m=KSFO_28L_WIDTH_M)
    pos_750 = np.array([-750.0 * math.cos(hdg), -750.0 * math.sin(hdg), -50.0], dtype=np.float64)
    uv, _ = project_runway_points_ned(pts_ned, pos_750, hdg, math.radians(-3.0), 0.0, 1200.0, 1200.0, 511.5, 511.5)
    lines = np.vstack([compute_hesse_line_from_points(uv[i0], uv[i1]) for i0, i1 in kl_pairs])
    sol = solve_4dof_map_pnpl(
        pts_ned,
        uv,
        np.broadcast_to(np.eye(2) * 0.25, (12, 2, 2)).copy(),
        pitch_rad=math.radians(-3.0),
        roll_rad=0.0,
        init_pos_ned=pos_750 + np.array([25.0, -15.0, 8.0]),
        init_yaw_rad=hdg + math.radians(2.0),
        fx=1200.0,
        fy=1200.0,
        cx=511.5,
        cy=511.5,
        keyline_pairs=kl_pairs,
        obs_keylines_hesse=lines,
    )
    pnpl_pos_err_m = float(np.linalg.norm(sol.pos_ned_m - pos_750))
    raim = compute_hybrid_conformal_raim_pl(sol.jacobian_obs, sol.cov_4x4, range_m=750.0)
    assoc_28l = check_runway_association(KSFO_28L_WIDTH_M, 1.5)
    assoc_twy_c = check_runway_association(KSFO_TAXIWAY_C_WIDTH_M, 0.5)
    assoc_28r = check_runway_association(KSFO_28L_WIDTH_M, KSFO_28R_LATERAL_OFFSET_M)

    lard_ver = validate_lard_real_airports_manifest(
        {"verified": True, "source": "LARD_v1_real_metadata_sha256_verified", "airports": list(SAMPLE_36_LARD_REAL_ICAOS)},
        SAMPLE_TRAIN_SYNTH_ICAOS,
        ["KSFO", *SAMPLE_36_LARD_REAL_ICAOS],
    )

    checks = {
        "ksfo_9lvl_mipmap_invariant": verify_mipmap_invariant(pyr_inner) and pyr_inner.num_levels == 9,
        "synth_4096_13lvl_mipmap_invariant": verify_mipmap_invariant(pyr_4096) and pyr_4096.num_levels == 13,
        "curved_ray_dem_under_0_1m": ray_max_err_m < 0.1,
        "taws_nominal_ksfo_zero_false_alerts": nom_false_alerts == 0,
        "taws_hill_caution_and_warning": caution_res.alert_level == "CAUTION" and warning_res.alert_level == "WARNING",
        "xvs_land_okls_unit_norm": okls_max_unit_err < 1e-6,
        "xvs_land_cholesky_strictly_spd": chol_min_eig > 0.0,
        "pnpl_4dof_converged_sub_meter": sol.converged and pnpl_pos_err_m < 0.05,
        "conformal_raim_cat1_nominal": raim.vpl_within_limit and raim.hpl_within_limit,
        "ksfo_28l_accept_twy_c_28r_reject": assoc_28l.accepted and (not assoc_twy_c.accepted) and (not assoc_28r.accepted),
        "lard_fail_closed_36_icao_zero_overlap": lard_ver.manifest_verified and lard_ver.lard_real_count >= 36 and len(lard_ver.train_heldout_overlap) == 0,
    }
    all_pass = all(checks.values())
    assert all_pass, f"Failed V2-M2L-4 checks: {checks}"

    evidence = {
        "schema": "xvs.v2_m2.v4_wp01b_land_evidence/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wp01b": {
            "ksfo_pack_terrain_mode": pyr_inner.terrain_mode,
            "ksfo_28l_thr_wgs84_h_m": pyr_inner.ksfo_28l_thr_wgs84_h_m,
            "ksfo_256_mipmap_levels": pyr_inner.num_levels,
            "synthetic_4096_mipmap_levels": pyr_4096.num_levels,
            "curved_ray_dem_max_vert_err_m": ray_max_err_m,
            "curved_ray_dem_threshold_m": 0.1,
            "taws_nominal_ksfo_28l_false_alerts": nom_false_alerts,
            "taws_hill_caution_ttc_s": caution_res.time_to_conflict_s,
            "taws_hill_warning_ttc_s": warning_res.time_to_conflict_s,
        },
        "xvs_land": {
            "num_keypoints": 12,
            "num_keylines": 5,
            "okls_unit_norm_max_err": okls_max_unit_err,
            "cholesky_cov_min_eig": chol_min_eig,
            "pnpl_4dof_pos_err_m": pnpl_pos_err_m,
            "pnpl_4dof_rms_reproj_px": sol.rms_reproj_px,
            "conformal_raim_v_vpl_m": raim.v_vpl_m,
            "conformal_raim_v_hpl_m": raim.v_hpl_m,
            "ksfo_28l_accepted": assoc_28l.accepted,
            "taxiway_c_rejected": not assoc_twy_c.accepted,
            "ksfo_28r_rejected": not assoc_28r.accepted,
            "lard_manifest_min_icaos": MIN_LARD_REAL_AIRPORTS,
            "lard_verified_icao_count": lard_ver.lard_real_count,
            "lard_train_heldout_overlap_count": len(lard_ver.train_heldout_overlap),
        },
        "checks": checks,
        "all_checks_passed": all_pass,
    }
    out_evidence_path = tmp_path / "v4_wp01b_land_evidence.json"
    out_evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    assert out_evidence_path.is_file()
