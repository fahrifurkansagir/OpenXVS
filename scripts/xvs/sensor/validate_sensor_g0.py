"""Master validation script for WP-03 G0 sensor and image-formation model.

Executes all verification items (V1..V7) in full scale, saves outputs and contact sheets,
benchmarks CPU execution time, and writes results to output/xvs/wp03/g0/results.json.

Spec reference: docs/xvs/WP03_SENSOR.md and docs/xvs/batches/GEMINI_G-B2.md.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from xvs.paths import get_data_root, data_root_source

import cv2
import numpy as np
from PIL import Image

from xvs.sensor.srgb import (
    srgb_eotf,
    srgb_oetf,
    srgb_to_8bit,
    linear_rgb_to_monochrome,
)
from xvs.sensor.psf import (
    compute_psf_kernel,
    compute_dtft,
    aliased_analytic_transfer_function,
)
from xvs.sensor.distortion import (
    distort_normalized,
    undistort_normalized,
    distort_points,
    undistort_points,
    compute_distortion_backward_map,
    warp_image_distort,
    warp_mask_distort,
    compute_max_displacement,
    compute_roi_grid_displacements,
)
from xvs.sensor.gt_transform import (
    transform_gt_polyline,
    transform_mesh_box,
    transform_gt_mask,
)
from xvs.sensor.vignetting import compute_vignetting_map, apply_vignetting
from xvs.sensor.noise import generate_fpn_maps, apply_sensor_noise
from xvs.sensor.emva1288 import analyze_emva1288
from xvs.sensor.chain import SensorConfig, CameraSimulator, save_raw_png, save_display_png


def validate_v1() -> dict[str, Any]:
    print("[V1] Validating sRGB EOTF/OETF exact inversion and 8-bit round trip...")
    # Exact float64 inversion test over 100k points
    c_grid = np.linspace(0.0, 1.0, 100_000, dtype=np.float64)
    l_grid = srgb_eotf(c_grid)
    c_rec = srgb_oetf(l_grid)
    max_err_c = float(np.max(np.abs(c_grid - c_rec)))

    l_test = np.linspace(0.0, 1.0, 100_000, dtype=np.float64)
    c_test = srgb_oetf(l_test)
    l_rec = srgb_eotf(c_test)
    max_err_l = float(np.max(np.abs(l_test - l_rec)))

    # 8-bit round trip
    codes = np.arange(256, dtype=np.uint8)
    l_8 = srgb_eotf(codes)
    c_8 = srgb_oetf(l_8)
    rec_codes = srgb_to_8bit(c_8)
    mismatches = int(np.sum(codes != rec_codes))

    status = "PASS" if (max_err_c <= 1e-12 and max_err_l <= 1e-12 and mismatches == 0) else "FAIL"
    print(f"  max_err_c: {max_err_c:.2e}, max_err_l: {max_err_l:.2e}, mismatches: {mismatches} -> {status}")
    return {
        "status": status,
        "max_err_c_to_l_to_c": max_err_c,
        "max_err_l_to_c_to_l": max_err_l,
        "tolerance_eotf_oetf": 1e-12,
        "mismatches_8bit": mismatches,
        "total_codes_tested": 256,
    }


def validate_v2(ring_cfg: SensorConfig, rig_cfg: SensorConfig) -> dict[str, Any]:
    print("[V2] Validating PSF kernel normalization and DTFT frequency response...")
    results: dict[str, Any] = {}

    for name, cfg in [("ring_mono_v0", ring_cfg), ("rig_mono_v0", rig_cfg)]:
        kernel, radius, trunc_energy = compute_psf_kernel(
            f_number=cfg.f_number,
            wavelength_m=cfg.wavelength_m,
            pixel_pitch_m=cfg.pixel_pitch_m,
            sigma_px=cfg.sigma_px,
            max_radius=25,
        )
        sum_err = abs(float(np.sum(kernel)) - 1.0)

        # 64x64 frequency grid
        nu_grid = np.linspace(-0.5, 0.5, 64, endpoint=False)
        nu_x, nu_y = np.meshgrid(nu_grid, nu_grid, indexing="xy")
        h_aliased = aliased_analytic_transfer_function(
            nu_x=nu_x,
            nu_y=nu_y,
            f_number=cfg.f_number,
            wavelength_m=cfg.wavelength_m,
            pixel_pitch_m=cfg.pixel_pitch_m,
            sigma_px=cfg.sigma_px,
            m_range=(-3, 3),
            n_range=(-3, 3),
        )
        dtft_h = compute_dtft(kernel, nu_x, nu_y)
        max_dtft_err = float(np.max(np.abs(dtft_h - h_aliased)))

        pass_v2a = sum_err <= 1e-12
        pass_v2b = max_dtft_err <= 0.005
        status = "PASS" if (pass_v2a and pass_v2b) else "FAIL"

        print(f"  [{name}] radius: {radius}, sum_err: {sum_err:.2e}, max_dtft_err: {max_dtft_err:.5f} -> {status}")
        results[name] = {
            "status": status,
            "recorded_radius_px": radius,
            "truncated_energy": float(trunc_energy),
            "sum_error": sum_err,
            "tolerance_sum": 1e-12,
            "max_dtft_error": max_dtft_err,
            "tolerance_dtft": 0.005,
        }

    overall_status = "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL"
    results["status"] = overall_status
    return results


def validate_v3(ring_cfg: SensorConfig, rig_cfg: SensorConfig) -> dict[str, Any]:
    print("[V3] Validating Brown-Conrady distortion, inversion, blobs, masks and displacement...")
    results: dict[str, Any] = {}

    # V3(a): Round-trip on 64x64 grid
    for name, cfg in [("ring_mono_v0", ring_cfg), ("rig_mono_v0", rig_cfg)]:
        u_grid = np.linspace(0.0, cfg.image_width - 1.0, 64)
        v_grid = np.linspace(0.0, cfg.image_height - 1.0, 64)
        u, v = np.meshgrid(u_grid, v_grid, indexing="xy")
        pts_undist = np.stack([u.ravel(), v.ravel()], axis=-1)

        pts_dist = distort_points(
            pts_undist, cfg.fx, cfg.fy, cfg.cx, cfg.cy,
            cfg.k1, cfg.k2, cfg.k3, cfg.p1, cfg.p2
        )
        pts_rec = undistort_points(
            pts_dist, cfg.fx, cfg.fy, cfg.cx, cfg.cy,
            cfg.k1, cfg.k2, cfg.k3, cfg.p1, cfg.p2
        )
        roundtrip_max_err = float(np.max(np.hypot(pts_rec[:, 0] - pts_undist[:, 0], pts_rec[:, 1] - pts_undist[:, 1])))
        pass_v3a = roundtrip_max_err <= 1e-6

        # V3(d): Maximum displacement
        max_disp = compute_max_displacement(
            cfg.image_width, cfg.image_height, cfg.fx, cfg.fy, cfg.cx, cfg.cy,
            cfg.k1, cfg.k2, cfg.k3, cfg.p1, cfg.p2
        )

        results[f"{name}_roundtrip"] = {
            "max_error_px": roundtrip_max_err,
            "tolerance_px": 1e-6,
            "status": "PASS" if pass_v3a else "FAIL",
            "max_displacement_px": max_disp,
        }
        print(f"  [{name}] V3a roundtrip max err: {roundtrip_max_err:.2e} px, max disp: {max_disp:.2f} px")

    # V3(b): Gaussian blobs centroid accuracy on rig_mono_v0
    p = rig_cfg
    domain_w, domain_h = 512, 512
    offset_u = (p.image_width - domain_w) // 2
    offset_v = (p.image_height - domain_h) // 2

    u_out, v_out = np.meshgrid(np.arange(domain_w) + offset_u, np.arange(domain_h) + offset_v, indexing="xy")
    xd = (u_out - p.cx) / p.fx
    yd = (v_out - p.cy) / p.fy
    x_in, y_in = undistort_normalized(xd, yd, p.k1, p.k2, p.k3, p.p1, p.p2)
    map_u = p.fx * x_in + p.cx
    map_v = p.fy * y_in + p.cy
    valid_mask = (
        (map_u >= -0.5) & (map_u <= p.image_width - 0.5) &
        (map_v >= -0.5) & (map_v <= p.image_height - 0.5)
    )

    sigma = 1.5
    grid_u = np.arange(offset_u + 60, offset_u + domain_w - 60, 50)
    grid_v = np.arange(offset_v + 60, offset_v + domain_h - 60, 50)
    img_undist = np.zeros((p.image_height, p.image_width), dtype=np.float64)
    blob_centers = []
    rad = int(np.ceil(6 * sigma))
    for v0 in grid_v:
        for u0 in grid_u:
            uu, vv = np.meshgrid(np.arange(u0 - rad, u0 + rad + 1), np.arange(v0 - rad, v0 + rad + 1), indexing="xy")
            blob = np.exp(-((uu - u0) ** 2 + (vv - v0) ** 2) / (2.0 * sigma ** 2))
            img_undist[vv, uu] += blob
            blob_centers.append((u0, v0))

    warped = warp_image_distort(img_undist, map_u, map_v, validity_mask=valid_mask, method="scipy")

    errors = []
    win_r = 7
    for u0, v0 in blob_centers:
        pt_dist = distort_points(
            np.array([[u0, v0]]), p.fx, p.fy, p.cx, p.cy,
            p.k1, p.k2, p.k3, p.p1, p.p2
        )[0]
        ud_true, vd_true = pt_dist[0], pt_dist[1]
        ud_rel = ud_true - offset_u
        vd_rel = vd_true - offset_v
        r_check = int(np.ceil(4 * sigma))
        u_min = int(np.round(ud_rel)) - r_check
        u_max = int(np.round(ud_rel)) + r_check
        v_min = int(np.round(ud_rel)) - r_check
        v_max = int(np.round(ud_rel)) + r_check
        if u_min < 0 or u_max >= domain_w or v_min < 0 or v_max >= domain_h:
            continue
        if not np.all(valid_mask[v_min : v_max + 1, u_min : u_max + 1]):
            continue

        cu_idx = int(np.round(ud_rel))
        cv_idx = int(np.round(vd_rel))
        sub_u, sub_v = np.meshgrid(
            np.arange(cu_idx - win_r, cu_idx + win_r + 1) + offset_u,
            np.arange(cv_idx - win_r, cv_idx + win_r + 1) + offset_v,
            indexing="xy",
        )
        weights = np.maximum(0.0, warped[cv_idx - win_r : cv_idx + win_r + 1, cu_idx - win_r : cu_idx + win_r + 1])
        total_w = np.sum(weights)
        if total_w > 0:
            c_u = np.sum(sub_u * weights) / total_w
            c_v = np.sum(sub_v * weights) / total_w
            errors.append(np.hypot(c_u - ud_true, c_v - vd_true))

    med_blob_err = float(np.median(errors))
    max_blob_err = float(np.max(errors))
    pass_v3b = (med_blob_err <= 0.02) and (max_blob_err <= 0.05)
    print(f"  V3b blob centroid med: {med_blob_err:.4f} px, max: {max_blob_err:.4f} px -> {'PASS' if pass_v3b else 'FAIL'}")
    results["v3b_blobs"] = {
        "status": "PASS" if pass_v3b else "FAIL",
        "blobs_measured": len(errors),
        "median_error_px": med_blob_err,
        "tolerance_median_px": 0.02,
        "max_error_px": max_blob_err,
        "tolerance_max_px": 0.05,
    }

    # V3(c): Synthetic polygon mask warp
    map_u_full, map_v_full, valid_mask_full = p.get_backward_map()
    poly = np.array([[400.0, 350.0], [550.0, 320.0], [600.0, 480.0], [520.0, 560.0], [380.0, 500.0]])
    mask_undist = np.zeros((p.image_height, p.image_width), dtype=np.uint8)
    cv2.fillPoly(mask_undist, [poly.astype(np.int32)], 1)
    warped_mask, mask_box = transform_gt_mask(mask_undist, map_u_full, map_v_full, validity_mask=valid_mask_full)
    _, analytic_box = transform_mesh_box(
        poly, p.fx, p.fy, p.cx, p.cy, p.k1, p.k2, p.k3, p.p1, p.p2
    )
    edge_diffs = [abs(float(mask_box[i] - analytic_box[i])) for i in range(4)]
    max_edge_diff = float(np.max(edge_diffs))
    pass_v3c = max_edge_diff <= 1.0
    print(f"  V3c polygon mask max edge difference: {max_edge_diff:.3f} px -> {'PASS' if pass_v3c else 'FAIL'}")
    results["v3c_polygon_mask"] = {
        "status": "PASS" if pass_v3c else "FAIL",
        "mask_box": mask_box,
        "analytic_box": analytic_box,
        "edge_differences_px": edge_diffs,
        "max_edge_difference_px": max_edge_diff,
        "tolerance_px": 1.0,
    }

    # V3(d): ROI displacement grid for ring_mono_v0
    roi_info = compute_roi_grid_displacements(
        ring_cfg.image_width, ring_cfg.image_height, 1024, 1024,
        ring_cfg.fx, ring_cfg.fy, ring_cfg.cx, ring_cfg.cy,
        ring_cfg.k1, ring_cfg.k2, ring_cfg.k3, ring_cfg.p1, ring_cfg.p2,
        roi_step=64,
    )
    results["v3d_roi_displacements"] = roi_info
    print(f"  V3d max ROI displacement over grid: {roi_info['max_roi_displacement_px']:.2f} px")

    overall_pass = (
        results["ring_mono_v0_roundtrip"]["status"] == "PASS"
        and results["rig_mono_v0_roundtrip"]["status"] == "PASS"
        and results["v3b_blobs"]["status"] == "PASS"
        and results["v3c_polygon_mask"]["status"] == "PASS"
    )
    results["status"] = "PASS" if overall_pass else "FAIL"
    return results


def validate_v4(ring_cfg: SensorConfig, rig_cfg: SensorConfig) -> dict[str, Any]:
    print("[V4] Validating optical vignetting flat field accuracy...")
    results: dict[str, Any] = {}

    for name, cfg in [("ring_mono_v0", ring_cfg), ("rig_mono_v0", rig_cfg)]:
        v_map = cfg.get_vignetting_map()
        flat_field = np.ones((cfg.image_height, cfg.image_width), dtype=np.float64)
        output = apply_vignetting(flat_field, v_map)
        rel_diff = np.abs(output - v_map) / v_map
        max_rel_diff = float(np.max(rel_diff))
        status = "PASS" if max_rel_diff <= 0.001 else "FAIL"
        print(f"  [{name}] max relative difference: {max_rel_diff:.2e} -> {status}")
        results[name] = {
            "status": status,
            "max_relative_difference": max_rel_diff,
            "tolerance": 0.001,
        }

    overall_status = "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL"
    results["status"] = overall_status
    return results


def validate_v5() -> dict[str, Any]:
    print("[V5] Validating full EMVA 1288 simulation and self-consistency...")
    h, w = 512, 512
    fw = 10500.0
    sigma_r = 2.4
    d_curr = 5.0
    t_exp = 0.001
    prnu_rms = 0.005
    dsnu_rms = 1.0
    k_gain = 0.35
    bits = 12
    black_lvl = 64.0

    prnu_map, dsnu_map = generate_fpn_maps("master_val_cam", h, w, prnu_rms, dsnu_rms)

    # 45 levels from dark to 110%
    levels_low = np.linspace(0.0, 0.78, 30)
    levels_high = np.linspace(0.80, 1.10, 15)
    levels = np.unique(np.concatenate([levels_low, levels_high]))

    rng_temp = np.random.Generator(np.random.PCG64(42))
    pairs = []
    print("  Generating photon transfer series (45 levels x 2 frames)...")
    for idx, L_val in enumerate(levels):
        flat = np.full((h, w), L_val, dtype=np.float64)
        f_a = apply_sensor_noise(flat, prnu_map, dsnu_map, fw, fw, sigma_r, d_curr, t_exp, k_gain, bits, black_lvl, rng_temp)
        f_b = apply_sensor_noise(flat, prnu_map, dsnu_map, fw, fw, sigma_r, d_curr, t_exp, k_gain, bits, black_lvl, rng_temp)
        pairs.append((f_a, f_b))

    print("  Generating 16 dark and 16 bright frames for spatial FPN...")
    dark_frames = [
        apply_sensor_noise(np.zeros((h, w)), prnu_map, dsnu_map, fw, fw, sigma_r, d_curr, t_exp, k_gain, bits, black_lvl, rng_temp)
        for _ in range(16)
    ]
    bright_frames = [
        apply_sensor_noise(np.full((h, w), 0.5), prnu_map, dsnu_map, fw, fw, sigma_r, d_curr, t_exp, k_gain, bits, black_lvl, rng_temp)
        for _ in range(16)
    ]

    print("  Analyzing EMVA 1288 parameters...")
    analysis = analyze_emva1288(levels, pairs, dark_frames, bright_frames, bits=bits)

    k_est = float(analysis["k_est"])
    sigma_r_est = float(analysis["sigma_r_est"])
    sat_cap_est = float(analysis["sat_capacity_est"])
    dsnu_est = float(analysis["dsnu_est"])
    prnu_est = float(analysis["prnu_est"])
    lin_err = float(analysis["linearity_error"])

    k_err_pct = abs(k_est - k_gain) / k_gain * 100.0
    sigma_r_err_pct = abs(sigma_r_est - sigma_r) / sigma_r * 100.0
    sat_cap_err_pct = abs(sat_cap_est - fw) / fw * 100.0
    dsnu_err_pct = abs(dsnu_est - dsnu_rms) / dsnu_rms * 100.0
    prnu_err_pct = abs(prnu_est - prnu_rms) / prnu_rms * 100.0
    lin_err_pct = lin_err * 100.0

    pass_k = k_err_pct <= 1.0
    pass_sigma_r = sigma_r_err_pct <= 3.0
    pass_sat = sat_cap_err_pct <= 3.0
    pass_dsnu = dsnu_err_pct <= 10.0
    pass_prnu = prnu_err_pct <= 10.0
    pass_lin = lin_err_pct < 0.5

    status = "PASS" if (pass_k and pass_sigma_r and pass_sat and pass_dsnu and pass_prnu and pass_lin) else "FAIL"

    print(f"  K: true={k_gain}, est={k_est:.4f} (err={k_err_pct:.2f}%, tol=1%) -> {'PASS' if pass_k else 'FAIL'}")
    print(f"  sigma_r: true={sigma_r}, est={sigma_r_est:.4f} (err={sigma_r_err_pct:.2f}%, tol=3%) -> {'PASS' if pass_sigma_r else 'FAIL'}")
    print(f"  sat_cap: true={fw}, est={sat_cap_est:.1f} (err={sat_cap_err_pct:.2f}%, tol=3%) -> {'PASS' if pass_sat else 'FAIL'}")
    print(f"  DSNU: true={dsnu_rms}, est={dsnu_est:.4f} (err={dsnu_err_pct:.2f}%, tol=10%) -> {'PASS' if pass_dsnu else 'FAIL'}")
    print(f"  PRNU: true={prnu_rms}, est={prnu_est:.5f} (err={prnu_err_pct:.2f}%, tol=10%) -> {'PASS' if pass_prnu else 'FAIL'}")
    print(f"  linearity: err={lin_err_pct:.4f}% (tol<0.5%) -> {'PASS' if pass_lin else 'FAIL'}")

    return {
        "status": status,
        "k_true": k_gain,
        "k_est": k_est,
        "k_error_pct": k_err_pct,
        "k_tolerance_pct": 1.0,
        "sigma_r_true": sigma_r,
        "sigma_r_est": sigma_r_est,
        "sigma_r_error_pct": sigma_r_err_pct,
        "sigma_r_tolerance_pct": 3.0,
        "sat_capacity_true": fw,
        "sat_capacity_est": sat_cap_est,
        "sat_capacity_error_pct": sat_cap_err_pct,
        "sat_capacity_tolerance_pct": 3.0,
        "dsnu_true": dsnu_rms,
        "dsnu_est": dsnu_est,
        "dsnu_error_pct": dsnu_err_pct,
        "dsnu_tolerance_pct": 10.0,
        "prnu_true": prnu_rms,
        "prnu_est": prnu_est,
        "prnu_error_pct": prnu_err_pct,
        "prnu_tolerance_pct": 10.0,
        "linearity_error_pct": lin_err_pct,
        "linearity_tolerance_pct": 0.5,
    }


def validate_v6(rig_cfg: SensorConfig) -> dict[str, Any]:
    print("[V6] Validating pipeline determinism and frame seed separation...")
    sim1 = CameraSimulator(rig_cfg, sensor_id="CAM_DETERMINISM_01")
    sim2 = CameraSimulator(rig_cfg, sensor_id="CAM_DETERMINISM_01")

    # Synthetic render
    rng = np.random.Generator(np.random.PCG64(777))
    test_img = rng.integers(0, 256, (rig_cfg.image_height, rig_cfg.image_width, 3), dtype=np.uint8)

    res1 = sim1.process_frame(test_img, image_id="scene_det", frame_idx=10)
    res2 = sim2.process_frame(test_img, image_id="scene_det", frame_idx=10)

    # Identical check
    raw_match = bool(np.array_equal(res1["raw"], res2["raw"]))
    disp_match = bool(np.array_equal(res1["display"], res2["display"]))

    buf1 = io.BytesIO()
    buf2 = io.BytesIO()
    Image.fromarray(res1["raw"]).save(buf1, format="PNG")
    Image.fromarray(res2["raw"]).save(buf2, format="PNG")
    raw_bytes_match = (buf1.getvalue() == buf2.getvalue())

    # Different frame check
    res_f_diff = sim1.process_frame(test_img, image_id="scene_det", frame_idx=11)
    diff_noise = bool(np.any(res1["raw"] != res_f_diff["raw"]))
    same_fpn = bool(np.array_equal(sim1.prnu_map, sim2.prnu_map) and np.array_equal(sim1.dsnu_map, sim2.dsnu_map))

    pass_v6 = raw_match and disp_match and raw_bytes_match and diff_noise and same_fpn
    status = "PASS" if pass_v6 else "FAIL"
    print(f"  raw match: {raw_match}, disp match: {disp_match}, PNG byte-match: {raw_bytes_match}, diff noise: {diff_noise} -> {status}")
    return {
        "status": status,
        "raw_array_identical": raw_match,
        "display_array_identical": disp_match,
        "png_bytes_identical": raw_bytes_match,
        "different_frame_temporal_noise_varies": diff_noise,
        "fixed_pattern_maps_preserved": same_fpn,
    }


def draw_polyline_on_image(image_bgr: np.ndarray, polyline_pts: list[list[float]], color: tuple[int, int, int]) -> np.ndarray:
    """Draw polyline onto image."""
    img = np.copy(image_bgr)
    if not polyline_pts or len(polyline_pts) < 2:
        return img
    pts = np.asarray(polyline_pts, dtype=np.float32)
    # Clip to image bounds for drawing
    int_pts = np.round(pts).astype(np.int32)
    for i in range(len(int_pts) - 1):
        p1 = tuple(int_pts[i])
        p2 = tuple(int_pts[i + 1])
        cv2.line(img, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
    return img


def validate_v7(
    g4_root: Path,
    out_dir: Path,
    rig_cfg: SensorConfig,
    num_frames: int = 20,
) -> dict[str, Any]:
    print(f"[V7] Processing {num_frames} G4 T2 frames for cameras F and D...")
    frames_jsonl_path = g4_root / "frames.jsonl"
    if not frames_jsonl_path.exists():
        raise FileNotFoundError(f"Missing frames.jsonl at {frames_jsonl_path}")

    # Read records from frames.jsonl
    records = []
    with open(frames_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= num_frames:
                break

    cameras = ["F", "D"]
    simulators = {
        cam: CameraSimulator(rig_cfg, sensor_id=f"G4_RIG_{cam}")
        for cam in cameras
    }

    # Output directories
    out_wp03 = out_dir / "output" / "xvs" / "wp03" / "g0"
    contact_dir = out_wp03 / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    timing_per_frame = []
    processed_count = 0

    for cam in cameras:
        cam_raw_dir = out_wp03 / cam / "raw"
        cam_disp_dir = out_wp03 / cam / "display"
        cam_raw_dir.mkdir(parents=True, exist_ok=True)
        cam_disp_dir.mkdir(parents=True, exist_ok=True)

        sim = simulators[cam]

        for k, record in enumerate(records):
            cam_info = record["cams"].get(cam)
            if not cam_info:
                continue

            rel_png = cam_info["png"]  # e.g. "F/000000.png"
            in_png_path = g4_root / rel_png
            if not in_png_path.exists():
                print(f"  Warning: missing input frame {in_png_path}")
                continue

            ideal_bgr = cv2.imread(str(in_png_path))
            ideal_rgb = cv2.cvtColor(ideal_bgr, cv2.COLOR_BGR2RGB)

            # Horizon GT
            gt = cam_info.get("gt", {})
            horizon_uv = gt.get("horizon_uv", [])

            # Run full sensor pipeline and measure time
            t0 = time.perf_counter()
            res = sim.process_frame(ideal_rgb, image_id="T2_ksfo28l", frame_idx=k, auto_exposure=False)
            t1 = time.perf_counter()
            timing_per_frame.append(t1 - t0)

            # Save raw and display
            frame_stem = f"{k:06d}.png"
            save_raw_png(res["raw"], cam_raw_dir / frame_stem)
            save_display_png(res["display"], cam_disp_dir / frame_stem)

            # Carry horizon GT through distortion
            if horizon_uv:
                dist_horizon = transform_gt_polyline(
                    np.array(horizon_uv),
                    rig_cfg.fx, rig_cfg.fy, rig_cfg.cx, rig_cfg.cy,
                    rig_cfg.k1, rig_cfg.k2, rig_cfg.k3, rig_cfg.p1, rig_cfg.p2,
                ).tolist()
            else:
                dist_horizon = []

            # Create side-by-side contact sheet
            # Left: ideal image with original horizon GT (cyan: 255, 255, 0 in BGR is yellow, cyan is 255, 255, 0 in RGB -> 0, 255, 255 in BGR)
            left_img = draw_polyline_on_image(ideal_bgr, horizon_uv, (255, 255, 0))  # Cyan/teal
            # Right: sensor display image with distorted horizon GT (magenta: 255, 0, 255 in BGR)
            disp_bgr = cv2.cvtColor(res["display"], cv2.COLOR_GRAY2BGR)
            right_img = draw_polyline_on_image(disp_bgr, dist_horizon, (255, 0, 255))

            # Header text
            cv2.putText(left_img, f"Ideal Render (Cam {cam} #{k})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(right_img, f"Sensor + Distorted GT (Cam {cam} #{k})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            contact_sheet = np.hstack([left_img, right_img])
            sheet_path = contact_dir / f"contact_{cam}_{k:06d}.png"
            cv2.imwrite(str(sheet_path), contact_sheet)

            processed_count += 1

    # CPU timing metrics
    mean_time_1024 = float(np.mean(timing_per_frame)) if timing_per_frame else 0.0
    # 2448x2048 has ~5.01M pixels, a 1024x1024 has ~1.05M pixels -> ratio is ~4.78
    roi_set_equivalent_ratio = (2448.0 * 2048.0) / (1024.0 * 1024.0)
    time_2448x2048_equiv = mean_time_1024 * roi_set_equivalent_ratio

    print(f"  Processed {processed_count} frames across F and D.")
    print(f"  Mean CPU time per 1024^2 mono image: {mean_time_1024*1000:.2f} ms (target <= 1000 ms)")
    print(f"  Equivalent time per 2448x2048 ROI set: {time_2448x2048_equiv*1000:.2f} ms")

    # Check G3a availability
    data_root = get_data_root()
    g3a_root = data_root / "xvs" / "wp02" / "g3a"
    g3a_present = g3a_root.exists()

    status = "PASS" if processed_count >= 40 else "PARTIAL"
    return {
        "status": status,
        "processed_frames_count": processed_count,
        "cameras_processed": cameras,
        "mean_time_per_1024_image_s": mean_time_1024,
        "mean_time_per_1024_image_ms": mean_time_1024 * 1000.0,
        "target_time_s": 1.0,
        "time_per_2448x2048_roi_set_s": time_2448x2048_equiv,
        "contact_sheets_directory": str(contact_dir),
        "g3a_available": g3a_present,
        "v8_stretch_goal_status": "SKIPPED_G3A_ABSENT" if not g3a_present else "READY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenXVS WP-03 G0: Master Sensor Model Validation")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root data directory containing xvs/ output hierarchy (defaults to XVS_DATA_ROOT or repo output/)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    data_root = get_data_root(args.data_root)
    config_dir = repo_root / "configs" / "xvs" / "sensors"
    output_dir = repo_root / "output" / "xvs" / "wp03" / "g0"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.json"

    print("======================================================================")
    print("OpenXVS WP-03 G0: Master Sensor Model Validation")
    print(f"Workspace: {repo_root}")
    print(f"Data Root: {data_root} (source: {data_root_source(args.data_root)})")
    print(f"Output: {results_file}")
    print("======================================================================")

    # Load sensor configs
    ring_cfg = SensorConfig.from_file(config_dir / "ring_mono_v0.json")
    rig_cfg = SensorConfig.from_file(config_dir / "rig_mono_v0.json")

    summary_results: dict[str, Any] = {
        "schema": "xvs.sensor_results/1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sensor_configs": {
            "ring_mono_v0": ring_cfg.name,
            "rig_mono_v0": rig_cfg.name,
        },
    }

    # V1
    summary_results["V1_srgb"] = validate_v1()

    # V2
    summary_results["V2_psf"] = validate_v2(ring_cfg, rig_cfg)

    # V3
    summary_results["V3_distortion"] = validate_v3(ring_cfg, rig_cfg)

    # V4
    summary_results["V4_vignetting"] = validate_v4(ring_cfg, rig_cfg)

    # V5
    summary_results["V5_emva1288"] = validate_v5()

    # V6
    summary_results["V6_determinism"] = validate_v6(rig_cfg)

    # V7
    g4_data_root = data_root / "xvs" / "wp01" / "g4" / "T2_ksfo28l"
    if not g4_data_root.exists():
        raise FileNotFoundError(f"G4 data directory not found: {g4_data_root}")
    summary_results["V7_real_renders"] = validate_v7(
        g4_root=g4_data_root,
        out_dir=repo_root,
        rig_cfg=rig_cfg,
        num_frames=20,
    )

    # Overall verdict
    all_hard_pass = (
        summary_results["V1_srgb"]["status"] == "PASS"
        and summary_results["V2_psf"]["status"] == "PASS"
        and summary_results["V3_distortion"]["status"] == "PASS"
        and summary_results["V4_vignetting"]["status"] == "PASS"
        and summary_results["V5_emva1288"]["status"] == "PASS"
        and summary_results["V6_determinism"]["status"] == "PASS"
        and summary_results["V7_real_renders"]["status"] == "PASS"
    )
    summary_results["verdict"] = "PASS" if all_hard_pass else "FAIL"

    # Write results.json
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print("======================================================================")
    print(f"Validation finished. Verdict: {summary_results['verdict']}")
    print(f"Results saved to: {results_file}")
    print("======================================================================")
    return 0 if all_hard_pass else 1


if __name__ == "__main__":
    sys.exit(main())
