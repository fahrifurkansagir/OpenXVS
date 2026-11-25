"""WP-02 Gate G0 validation script (Feasibility spike, T-CAL, T-ROI, T-SIZE, T-OCC, G3 regression, T-LIMB-ROI).

Conforms strictly to docs/xvs/WP02_TRAFFIC.md §3 G0 and §4 thresholds.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, set_data_root, data_root_source
from xvs.rig.rig import Rig
from xvs.rig.camera import PinholeCamera
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer
from xvs.geo import wgs84, rotations
from xvs.sim import horizon, trajectory, airports
from xvs.traffic.ring import TrafficRing

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# WP-01 §4 L1 limits; T-LIMB-ROI threshold from WP02 spec v1.1 (§3 G0 item 6b, §4; Director decision 2026-09-24).
L1_LIMITS = {"median_abs_delta_px": 0.15, "p95_abs_delta_px": 0.35, "max_abs_delta_px": 0.50, "abs_signed_mean_px": 0.05}


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_limb_v(p_ecef: np.ndarray, r_ecef: np.ndarray, cam: PinholeCamera, col: float) -> float | None:
    roots = horizon.analytic_horizon_v_at_u(p_ecef, r_ecef, cam, col)
    return float(roots[0]) if roots else None


def run_t_cal(renderer: XVSRenderer, backend: str, out_dir: Path) -> dict[str, Any]:
    """T-CAL: Calibration spheres placement and orientation validation (WP-02 §3 G0)."""
    print("\n--- Running T-CAL (Calibration Spheres Placement & Orientation) ---")
    page = renderer.new_camera_page("F", profile="mask_calib")
    true_rig = Rig.load_nominal().make_true(20260922)
    f_cam = true_rig.cam_by_id["F"]

    # Ownship at (37.62, -122.37, 1000m), level, psi=0
    lat, lon, h = 37.62, -122.37, 1000.0
    cam_poses = true_rig.compute_poses(lat, lon, h, 0.0, 0.0, 0.0)
    f_pose = cam_poses["F"]

    renderer.set_camera(
        position=f_pose["position"],
        direction=f_pose["direction"],
        up=f_pose["up"],
        fov_rad=f_pose["fov_rad"],
        aspect_ratio=f_pose["aspect_ratio"],
        page=page,
    )

    p_cam_ecef = np.array(f_pose["position"], dtype=np.float64)
    r_ecef_cam = np.array(f_pose["r_ecef_cam"], dtype=np.float64)
    b_ecef = r_ecef_cam[:, 2]

    # Center of CAL_SPHERES 30 m ahead of F camera
    p_model_ecef = p_cam_ecef + 30.0 * b_ecef
    r_ecef_ned = wgs84.r_ecef_ned(lat, lon)

    # 6 sphere definitions in BODY frame and pure colors
    spheres_def = [
        {"name": "nose", "pos_body": np.array([+5.5, 0.0, 0.0]), "rgb": [255, 0, 0]},
        {"name": "tail", "pos_body": np.array([-5.5, 0.0, 0.0]), "rgb": [0, 255, 0]},
        {"name": "right_wingtip", "pos_body": np.array([0.0, +6.5, 0.0]), "rgb": [0, 0, 255]},
        {"name": "left_wingtip", "pos_body": np.array([0.0, -6.5, 0.0]), "rgb": [255, 255, 0]},
        {"name": "fin_top", "pos_body": np.array([-4.5, 0.0, -3.0]), "rgb": [255, 0, 255]},
        {"name": "belly", "pos_body": np.array([0.0, 0.0, +1.5]), "rgb": [0, 255, 255]},
    ]

    model_url = f"http://127.0.0.1:{renderer.port}/assets/traffic/CALIBRATION/CAL_SPHERES.glb"
    renderer.clear_models(page=page)
    renderer.add_model(url=model_url, custom_shader="unlit_passthrough", page=page)

    attitudes_deg = [
        (45.0, 20.0, 30.0),
        (-35.0, -15.0, 60.0),
        (135.0, 10.0, -40.0),
    ]

    pose_results = []
    all_errors = []

    for att_idx, (yaw_d, pitch_d, roll_d) in enumerate(attitudes_deg):
        r_ned_model = rotations.euler_to_r(math.radians(yaw_d), math.radians(pitch_d), math.radians(roll_d))
        r_ecef_model = r_ecef_ned @ r_ned_model

        # 4x4 column-major matrix
        m4 = np.eye(4, dtype=np.float64)
        m4[:3, :3] = r_ecef_model
        m4[:3, 3] = p_model_ecef
        cesium_m = m4.T.flatten().tolist()

        renderer.set_model_pose(0, cesium_m, page=page)
        res = renderer.render_frame(page=page, transfer_method="png")

        png_p = out_dir / f"t_cal_att{att_idx}_{backend}.png"
        renderer.save_png(res.image, png_p)

        img_rgb = res.image  # (1024, 1024, 3)

        sphere_errors = []
        for s in spheres_def:
            p_s_ecef = p_model_ecef + r_ecef_model @ s["pos_body"]
            p_s_cam = r_ecef_cam.T @ (p_s_ecef - p_cam_ecef)
            u_proj, v_proj, _ = true_rig.camera_model.project_cam(p_s_cam)

            # Segmentation by pure color
            target_rgb = s["rgb"]
            if target_rgb == [255, 0, 0]:      # Red
                mask = (img_rgb[:, :, 0] > 180) & (img_rgb[:, :, 1] < 60) & (img_rgb[:, :, 2] < 60)
            elif target_rgb == [0, 255, 0]:    # Green
                mask = (img_rgb[:, :, 0] < 60) & (img_rgb[:, :, 1] > 180) & (img_rgb[:, :, 2] < 60)
            elif target_rgb == [0, 0, 255]:    # Blue
                mask = (img_rgb[:, :, 0] < 60) & (img_rgb[:, :, 1] < 60) & (img_rgb[:, :, 2] > 180)
            elif target_rgb == [255, 255, 0]:  # Yellow
                mask = (img_rgb[:, :, 0] > 180) & (img_rgb[:, :, 1] > 180) & (img_rgb[:, :, 2] < 60)
            elif target_rgb == [255, 0, 255]:  # Magenta
                mask = (img_rgb[:, :, 0] > 180) & (img_rgb[:, :, 1] < 60) & (img_rgb[:, :, 2] > 180)
            elif target_rgb == [0, 255, 255]:  # Cyan
                mask = (img_rgb[:, :, 0] < 60) & (img_rgb[:, :, 1] > 180) & (img_rgb[:, :, 2] > 180)
            else:
                mask = np.zeros((1024, 1024), dtype=bool)

            pixel_count = int(np.count_nonzero(mask))
            if pixel_count == 0:
                print(f"  [ERROR] Att {att_idx} Sphere {s['name']} not detected in mask!")
                continue

            v_indices, u_indices = np.where(mask)
            u_c = float(np.mean(u_indices))
            v_c = float(np.mean(v_indices))

            err = math.sqrt((u_c - u_proj) ** 2 + (v_c - v_proj) ** 2)
            sphere_errors.append({
                "sphere": s["name"],
                "u_proj": u_proj,
                "v_proj": v_proj,
                "u_centroid": u_c,
                "v_centroid": v_c,
                "error_px": err,
                "pixel_count": pixel_count,
            })
            all_errors.append(err)

        pose_results.append({
            "attitude_ypr_deg": [yaw_d, pitch_d, roll_d],
            "spheres": sphere_errors,
        })

    # Render CAL in rgb for visual check
    renderer.clear_models(page=page)
    cal_url = f"http://127.0.0.1:{renderer.port}/assets/traffic/CALIBRATION/CAL.glb"
    renderer.set_profile("rgb", page=page)
    renderer.add_model(url=cal_url, page=page)
    renderer.set_model_pose(0, cesium_m, page=page)
    res_cal = renderer.render_frame(page=page)
    cal_png = out_dir / f"t_cal_airframe_rgb_{backend}.png"
    renderer.save_png(res_cal.image, cal_png)

    errs_arr = np.array(all_errors)
    med_err = float(np.median(errs_arr))
    max_err = float(np.max(errs_arr))
    # Threshold on ref: median <= 0.15 px, max <= 0.50 px
    passes = (med_err <= 0.15 and max_err <= 0.50) if backend == "ref" else True

    print(f"  T-CAL ({backend}): Median error = {med_err:.4f} px (limit <= 0.15), Max = {max_err:.4f} px (limit <= 0.50) -> {'PASS' if passes else 'FAIL'}")
    return {
        "median_error_px": med_err,
        "max_error_px": max_err,
        "all_pass": passes,
        "poses": pose_results,
    }


def run_t_roi(renderer: XVSRenderer, backend: str, out_dir: Path) -> dict[str, Any]:
    """T-ROI: Ring camera T0 ROI frustum validation with 40 markers (WP-02 §3 G0)."""
    print("\n--- Running T-ROI (Traffic Ring Camera T0 ROI Frustum) ---")
    ring = TrafficRing()
    page = renderer.new_camera_page("T0", profile="markers")

    # Nom pose for ownship
    lat, lon, h = 37.62, -122.37, 1000.0
    ring_poses = ring.compute_poses(lat, lon, h, 0.0, 0.0, 0.0)
    t0_pose = ring_poses["T0"]
    p_t0_ecef = np.array(t0_pose["position"], dtype=np.float64)
    r_ecef_t0 = np.array(t0_pose["r_ecef_cam"], dtype=np.float64)

    rois = [
        (712, 512),
        (0, 0),
        (1424, 1024),
    ]

    roi_summaries = []
    all_rois_pass = True

    for r_idx, (u0, v0) in enumerate(rois):
        # Set ROI frustum and camera pose
        renderer.set_camera(
            position=t0_pose["position"],
            direction=t0_pose["direction"],
            up=t0_pose["up"],
            fov_rad=math.radians(ring.fov_h_deg),
            aspect_ratio=1.0,
            page=page,
        )
        renderer.set_roi_frustum(u0=u0, v0=v0, fx=ring.fx, fy=ring.fy, cx=ring.cx, cy=ring.cy, page=page)

        # Generate 40 markers in ROI pixels [20, 1003]^2
        rng = np.random.default_rng(20260923 + r_idx)
        markers = []
        u_roi_pts = rng.uniform(20.0, 1003.0, size=40)
        v_roi_pts = rng.uniform(20.0, 1003.0, size=40)

        # Distances: >=10 at 15-50m, >=10 at 1-5km, rest at 100-500m
        dists = np.concatenate([
            rng.uniform(15.0, 50.0, size=12),
            rng.uniform(1000.0, 5000.0, size=12),
            rng.uniform(100.0, 500.0, size=16),
        ])
        rng.shuffle(dists)

        for i in range(40):
            ur, vr = float(u_roi_pts[i]), float(v_roi_pts[i])
            u_full = ur + float(u0)
            v_full = vr + float(v0)

            # Unproject ray in T0 camera frame
            ray_cam = ring.camera_model.unproject(u_full, v_full)
            p_m_cam = ray_cam * dists[i]
            p_m_ecef = p_t0_ecef + r_ecef_t0 @ p_m_cam

            # Assign distinct color
            hue = (i * 360.0 / 40.0)
            color_rgba = [1.0, 1.0, 1.0, 1.0]

            markers.append({
                "id": i,
                "position": p_m_ecef.tolist(),
                "p_ecef": p_m_ecef.tolist(),
                "color": color_rgba,
                "pixel_size": 6,
                "u_roi_expected": ur,
                "v_roi_expected": vr,
            })

        renderer.set_markers(markers, page=page)

        # M2: worldToWindowCoordinates
        m2_projs = renderer.get_marker_projections(markers, page=page)
        m2_errors = []
        for i, proj in enumerate(m2_projs):
            if proj is not None:
                du = proj["u"] - markers[i]["u_roi_expected"]
                dv = proj["v"] - markers[i]["v_roi_expected"]
                m2_errors.append(math.sqrt(du ** 2 + dv ** 2))

        m2_max = float(np.max(m2_errors)) if m2_errors else 999.0
        m2_pass = (m2_max <= 1e-3)

        # M1: Render and detect centroids
        res = renderer.render_frame(page=page, transfer_method="png")
        png_p = out_dir / f"t_roi_{r_idx}_{u0}_{v0}_{backend}.png"
        renderer.save_png(res.image, png_p)

        # Measure detected markers in rendered image
        img_gray = res.image[:, :, 0]
        detected = 0
        e_list = []
        du_list = []
        dv_list = []

        for m in markers:
            ur = m["u_roi_expected"]
            vr = m["v_roi_expected"]
            # Crop 15x15 window around expected
            col0 = max(0, int(round(ur)) - 7)
            col1 = min(1024, int(round(ur)) + 8)
            row0 = max(0, int(round(vr)) - 7)
            row1 = min(1024, int(round(vr)) + 8)

            patch = img_gray[row0:row1, col0:col1].astype(float)
            if np.max(patch) > 50:
                detected += 1
                total_w = np.sum(patch)
                cols_grid, rows_grid = np.meshgrid(np.arange(col0, col1), np.arange(row0, row1))
                uc = np.sum(cols_grid * patch) / total_w
                vc = np.sum(rows_grid * patch) / total_w

                du = uc - ur
                dv = vc - vr
                du_list.append(du)
                dv_list.append(dv)
                e_list.append(math.sqrt(du ** 2 + dv ** 2))

        m1_detected = detected
        m1_mean_du = float(np.mean(du_list)) if du_list else 999.0
        m1_mean_dv = float(np.mean(dv_list)) if dv_list else 999.0
        m1_med_e = float(np.median(e_list)) if e_list else 999.0
        m1_max_e = float(np.max(e_list)) if e_list else 999.0

        # WP-01 §4 thresholds: 40/40, |mean| <= 0.05, median <= 0.10, max <= 0.40, M2 <= 1e-3
        m1_pass = (
            m1_detected == 40
            and abs(m1_mean_du) <= 0.05
            and abs(m1_mean_dv) <= 0.05
            and m1_med_e <= 0.10
            and m1_max_e <= 0.40
        )
        roi_pass = m1_pass and m2_pass
        if not roi_pass:
            all_rois_pass = False

        print(f"  ROI ({u0}, {v0}) [{backend}]: M1={m1_detected}/40, med|e|={m1_med_e:.4f} px, max|e|={m1_max_e:.4f} px; M2 max={m2_max:.2e} px -> {'PASS' if roi_pass else 'FAIL'}")
        roi_summaries.append({
            "roi": [u0, v0],
            "detected": m1_detected,
            "mean_du": m1_mean_du,
            "mean_dv": m1_mean_dv,
            "median_e": m1_med_e,
            "max_e": m1_max_e,
            "m2_max": m2_max,
            "pass": roi_pass,
        })

    renderer.clear_markers(page=page)
    return {
        "all_pass": all_rois_pass,
        "rois": roi_summaries,
    }


def run_t_size(renderer: XVSRenderer, backend: str, out_dir: Path) -> dict[str, Any]:
    """T-SIZE: Scale validation of C5_TRANSPORT at 1 km and 5 km (WP-02 §3 G0)."""
    print("\n--- Running T-SIZE (C5 Transport Scale at 1 km and 5 km) ---")
    ring = TrafficRing()
    page = renderer.new_camera_page("T0", profile="mask_amodal")

    lat, lon, h = 37.62, -122.37, 1000.0
    ring_poses = ring.compute_poses(lat, lon, h, 0.0, 0.0, 0.0)
    t0_pose = ring_poses["T0"]
    p_t0_ecef = np.array(t0_pose["position"], dtype=np.float64)
    r_ecef_t0 = np.array(t0_pose["r_ecef_cam"], dtype=np.float64)
    b_t0 = r_ecef_t0[:, 2]

    renderer.set_camera(
        position=t0_pose["position"],
        direction=t0_pose["direction"],
        up=t0_pose["up"],
        fov_rad=math.radians(ring.fov_h_deg),
        aspect_ratio=1.0,
        page=page,
    )

    c5_meta_path = REPO_ROOT / "assets" / "traffic" / "C5_TRANSPORT" / "C5_TRANSPORT_0001.json"
    with open(c5_meta_path, "r", encoding="utf-8") as f:
        c5_meta = json.load(f)

    # Load C5 GLB model
    c5_url = f"http://127.0.0.1:{renderer.port}/assets/traffic/C5_TRANSPORT/C5_TRANSPORT_0001.glb"
    renderer.clear_models(page=page)
    renderer.add_model(url=c5_url, custom_shader="unlit_white", page=page)

    # We extract all vertices from C5 model builder
    from xvs.traffic.models import build_c5_transport
    c5_model_res = build_c5_transport(seed=1)
    all_verts_body = np.concatenate([p.vertices for p in c5_model_res.parts], axis=0)

    # Broadside orientation: heading +90 deg relative to boresight
    r_ecef_ned = wgs84.r_ecef_ned(lat, lon)
    r_ned_c5 = rotations.euler_to_r(math.radians(90.0), 0.0, 0.0)
    r_ecef_c5 = r_ecef_ned @ r_ned_c5

    distances = [1000.0, 5000.0]
    size_results = []
    all_size_pass = True

    for dist in distances:
        p_c5_ecef = p_t0_ecef + dist * b_t0

        m4 = np.eye(4, dtype=np.float64)
        m4[:3, :3] = r_ecef_c5
        m4[:3, 3] = p_c5_ecef
        cesium_m = m4.T.flatten().tolist()

        # Compute full-image analytic projected extent of all vertices
        verts_ecef = (r_ecef_c5 @ all_verts_body.T).T + p_c5_ecef
        verts_cam = (r_ecef_t0.T @ (verts_ecef - p_t0_ecef).T).T

        u_projs = ring.fx * verts_cam[:, 0] / verts_cam[:, 2] + ring.cx
        v_projs = ring.fy * verts_cam[:, 1] / verts_cam[:, 2] + ring.cy

        u_min_an = float(np.min(u_projs))
        u_max_an = float(np.max(u_projs))
        v_min_an = float(np.min(v_projs))
        v_max_an = float(np.max(v_projs))
        u_center_an = (u_min_an + u_max_an) * 0.5
        v_center_an = (v_min_an + v_max_an) * 0.5

        # Compute ROI top-left around intruder center
        u0, v0 = ring.compute_roi(u_center_an, v_center_an)
        renderer.set_roi_frustum(u0=u0, v0=v0, fx=ring.fx, fy=ring.fy, cx=ring.cx, cy=ring.cy, page=page)
        renderer.set_model_pose(0, cesium_m, page=page)

        res = renderer.render_frame(page=page, transfer_method="png")
        png_p = out_dir / f"t_size_{int(dist)}m_{backend}.png"
        renderer.save_png(res.image, png_p)

        # Measure rendered mask extent in ROI, then shift to full image
        mask = res.image[:, :, 0] > 128
        v_idx, u_idx = np.where(mask)
        if len(u_idx) == 0:
            print(f"  [ERROR] T-SIZE: rendered mask is completely empty at {dist}m!")
            extent_mask = 0.0
        else:
            u_min_mask = float(np.min(u_idx)) + float(u0)
            u_max_mask = float(np.max(u_idx)) + float(u0)
            extent_mask = (u_max_mask - u_min_mask + 1.0)

        extent_an = float(math.floor(u_max_an) - math.ceil(u_min_an) + 1)
        diff_px = abs(extent_mask - extent_an)

        passes = (diff_px <= 1.0)
        if not passes:
            all_size_pass = False

        print(f"  T-SIZE @ {dist:.0f}m [{backend}]: mask extent = {extent_mask:.0f} px, analytic = {extent_an:.0f} px (diff = {diff_px:.1f} px, limit <= 1 px) -> {'PASS' if passes else 'FAIL'}")
        size_results.append({
            "distance_m": dist,
            "extent_mask_px": extent_mask,
            "extent_analytic_px": extent_an,
            "diff_px": diff_px,
            "pass": passes,
        })

    return {
        "all_pass": all_size_pass,
        "results": size_results,
    }


def run_t_occ(renderer: XVSRenderer, backend: str, out_dir: Path) -> dict[str, Any]:
    """T-OCC: Earth occlusion test of PANEL at 500 km behind horizon (WP-02 §3 G0 item 5)."""
    print("\n--- Running T-OCC (Earth Ellipsoid Horizon Occlusion) ---")
    page = renderer.new_camera_page("F", profile="mask_visible")
    true_rig = Rig.load_nominal().make_true(20260922)
    f_cam = true_rig.cam_by_id["F"]

    # Ownship level at 10 km
    lat, lon, h = 37.62, -122.37, 10000.0
    cam_poses = true_rig.compute_poses(lat, lon, h, 0.0, 0.0, 0.0)
    f_pose = cam_poses["F"]

    renderer.set_camera(
        position=f_pose["position"],
        direction=f_pose["direction"],
        up=f_pose["up"],
        fov_rad=f_pose["fov_rad"],
        aspect_ratio=f_pose["aspect_ratio"],
        page=page,
    )

    p_cam_ecef = np.array(f_pose["position"], dtype=np.float64)
    r_ecef_cam = np.array(f_pose["r_ecef_cam"], dtype=np.float64)

    # Analytic limb at column 511.5
    r_ecef_cam_row = np.array(f_cam.r_body_cam, dtype=np.float64) # in body
    v_limb_center = get_limb_v(p_cam_ecef, r_ecef_cam, true_rig.camera_model, 511.5)

    # Tangent ray through column 511.5 and row v_limb_center
    ray_cam = true_rig.camera_model.unproject(511.5, v_limb_center)
    ray_ecef = r_ecef_cam @ ray_cam

    # Center of PANEL 500 km along that ray
    p_panel_ecef = p_cam_ecef + 500000.0 * ray_ecef

    # Panel orientation: facing camera (-X along ray_ecef, +Z camera down, +Y right)
    x_panel = -ray_ecef
    z_panel = r_ecef_cam[:, 1] # camera y (down in camera frame)
    y_panel = np.cross(z_panel, x_panel)
    y_panel /= np.linalg.norm(y_panel)
    z_panel = np.cross(x_panel, y_panel)
    z_panel /= np.linalg.norm(z_panel)
    r_ecef_panel = np.stack([x_panel, y_panel, z_panel], axis=1)

    m4 = np.eye(4, dtype=np.float64)
    m4[:3, :3] = r_ecef_panel
    m4[:3, 3] = p_panel_ecef
    cesium_m = m4.T.flatten().tolist()

    panel_url = f"http://127.0.0.1:{renderer.port}/assets/traffic/CALIBRATION/PANEL.glb"
    renderer.clear_models(page=page)
    renderer.add_model(url=panel_url, custom_shader="unlit_white", page=page)
    renderer.set_model_pose(0, cesium_m, page=page)

    # Test with depthTestAgainstTerrain = false and true
    occlusion_evals = {}
    chosen_setting = None

    for depth_test in (False, True):
        renderer.set_depth_test_against_terrain(depth_test, page=page)
        res = renderer.render_frame(page=page, transfer_method="png")
        png_p = out_dir / f"t_occ_depthTest_{str(depth_test).lower()}_{backend}.png"
        renderer.save_png(res.image, png_p)

        # Evaluate lowest visible panel pixel per column
        img_gray = res.image[:, :, 0]
        # Columns where panel is visible
        col_errors = []
        for col in range(256, 768, 8):
            an_v = get_limb_v(p_cam_ecef, r_ecef_cam, true_rig.camera_model, float(col))
            if an_v is None:
                continue

            col_pixels = np.where(img_gray[:, col] > 128)[0]
            if len(col_pixels) > 0:
                # Lowest visible panel row + 0.5
                lowest_v = float(np.max(col_pixels)) + 0.5
                diff = abs(lowest_v - an_v)
                col_errors.append(diff)

        if col_errors:
            med_diff = float(np.median(col_errors))
            max_diff = float(np.max(col_errors))
        else:
            med_diff, max_diff = 999.0, 999.0

        passes = (med_diff <= 0.35 and max_diff <= 0.75)
        print(f"  depthTestAgainstTerrain={depth_test} [{backend}]: median |Δ| = {med_diff:.4f} px (limit <= 0.35), max = {max_diff:.4f} px (limit <= 0.75) -> {'PASS' if passes else 'FAIL'}")

        occlusion_evals[str(depth_test)] = {
            "depth_test": depth_test,
            "median_diff_px": med_diff,
            "max_diff_px": max_diff,
            "pass": passes,
        }
        if passes and chosen_setting is None:
            chosen_setting = depth_test

    return {
        "chosen_setting": chosen_setting,
        "evaluations": occlusion_evals,
        "pass": chosen_setting is not None,
    }


def run_g3_regression(renderer: XVSRenderer, out_dir: Path) -> dict[str, Any]:
    """G3 Regression check: re-render 8 level 600m G3 mask_earth scenes on ref (WP-02 §3 G0 item 6)."""
    print("\n--- Running G3 Regression Check on ref ---")
    data_root = get_data_root()
    g3_dir = data_root / "xvs" / "wp01" / "g3"
    if not g3_dir.exists():
        raise FileNotFoundError(f"G3 directory not found: {g3_dir}")
    target_ids = [
        "H_F_h600_p0_r0_y0",
        "H_R_h600_p0_r0_y0",
        "H_B_h600_p0_r0_y0",
        "H_L_h600_p0_r0_y0",
        "H_F_h600_p0_r0_y137",
        "H_R_h600_p0_r0_y137",
        "H_B_h600_p0_r0_y137",
        "H_L_h600_p0_r0_y137",
    ]

    matches = 0
    total = len(target_ids)

    for sid in target_ids:
        pose_path = g3_dir / "ref" / "poses" / f"{sid}_pose.json"
        with open(pose_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)

        cam_id = pdata["camera_id"]
        c_page = renderer.new_camera_page(cam_id, profile="mask_earth")
        cp = pdata["camera_pose_requested"]
        renderer.set_camera(
            position=cp["position"],
            direction=cp["direction"],
            up=cp["up"],
            fov_rad=cp["fov_rad"],
            aspect_ratio=cp["aspect_ratio"],
            page=c_page,
        )
        res = renderer.render_frame(page=c_page, transfer_method="png")
        if res.rgb_sha256 == pdata["rgb_sha256"]:
            matches += 1
        else:
            print(f"  [MISMATCH] {sid}: expected {pdata['rgb_sha256']}, got {res.rgb_sha256}")

    print(f"  G3 Regression: {matches}/{total} scenes identical rgb_sha256 -> {'PASS' if matches == total else 'FAIL'}")
    return {
        "matches": matches,
        "total": total,
        "pass": (matches == total),
    }


def run_t_limb_roi(renderer: XVSRenderer, backend: str, out_dir: Path) -> dict[str, Any]:
    """T-LIMB-ROI: Ring camera T0 ROIs containing Earth limb (WP-02 §3 G0 item 6b)."""
    print("\n--- Running T-LIMB-ROI (Limb in Ring ROIs) ---")
    ring = TrafficRing()
    page = renderer.new_camera_page("T0", profile="mask_earth")

    # Ownship at (37.62, -122.37) at 600m and 10km
    heights = [600.0, 10000.0]
    roi_results = []

    for h in heights:
        ring_poses = ring.compute_poses(37.62, -122.37, h, 0.0, 0.0, 0.0)
        t0_p = ring_poses["T0"]
        p_ecef = np.array(t0_p["position"], dtype=np.float64)
        r_ecef_t0 = np.array(t0_p["r_ecef_cam"], dtype=np.float64)

        v_limb = get_limb_v(p_ecef, r_ecef_t0, ring.camera_model, ring.cx)

        renderer.set_camera(
            position=t0_p["position"],
            direction=t0_p["direction"],
            up=t0_p["up"],
            fov_rad=math.radians(ring.fov_h_deg),
            aspect_ratio=1.0,
            page=page,
        )

        v0_target = int(np.clip(np.round(v_limb - 512.0), 0, ring.height - 1024))
        test_rois = [
            (712, v0_target),
            (0, v0_target),
        ]

        for u0, v0 in test_rois:
            renderer.set_roi_frustum(u0=u0, v0=v0, fx=ring.fx, fy=ring.fy, cx=ring.cx, cy=ring.cy, page=page)
            t_r = time.perf_counter()
            res = renderer.render_frame(page=page, transfer_method="png")
            ms_render = (time.perf_counter() - t_r) * 1000.0
            png_p = out_dir / f"t_limb_roi_{int(h)}m_{u0}_{v0}_{backend}.png"
            renderer.save_png(res.image, png_p)

            # Spec v1.1 (§3 G0 item 6b, §4): the WP-01 §2.6 estimator on every line of the ROI, with the
            # ring intrinsics and the ROI offset (principal point cx - u0, cy - v0), against the L1 limits.
            roi_cam = PinholeCamera(width=1024, height=1024, fx=ring.fx, fy=ring.fy, cx=ring.cx - u0, cy=ring.cy - v0)
            est = horizon.estimate_horizon_edge(res.image, p_ecef, r_ecef_t0, roi_cam)
            l1 = {
                "lines_used": est["lines_used"],
                "lines_skipped": est["lines_skipped"],
                "median_abs_delta_px": est["median_abs_delta"],
                "p95_abs_delta_px": est["p95_abs_delta"],
                "max_abs_delta_px": est["max_abs_delta"],
                "signed_mean_px": est["signed_mean"],
            }
            l1["pass"] = bool(
                est["lines_used"] > 0
                and est["median_abs_delta"] <= L1_LIMITS["median_abs_delta_px"]
                and est["p95_abs_delta"] <= L1_LIMITS["p95_abs_delta_px"]
                and est["max_abs_delta"] <= L1_LIMITS["max_abs_delta_px"]
                and abs(est["signed_mean"]) <= L1_LIMITS["abs_signed_mean_px"]
            )

            # Estimate edge in ROI
            img_mask = res.image[:, :, 0]
            col_diffs = []
            for col_roi in range(100, 924, 10):
                col_full = float(col_roi + u0)
                an_v_full = get_limb_v(p_ecef, r_ecef_t0, ring.camera_model, col_full)
                if an_v_full is None:
                    continue
                an_v_roi = an_v_full - float(v0)

                # Coverage method in column
                col_data = img_mask[:, col_roi] / 255.0
                v_edge_roi = (1024.0 - 0.5) - np.sum(col_data)
                col_diffs.append(abs(v_edge_roi - an_v_roi))

            med_diff = float(np.median(col_diffs)) if col_diffs else 999.0
            p95_diff = float(np.percentile(col_diffs, 95)) if col_diffs else 999.0
            print(f"  T-LIMB-ROI @ {h:.0f}m ROI ({u0}, {v0}) [{backend}]: sampled median = {med_diff:.4f} px, p95 = {p95_diff:.4f} px; "
                  f"L1 ({l1['lines_used']} lines): median {l1['median_abs_delta_px']:.4f}, p95 {l1['p95_abs_delta_px']:.4f}, "
                  f"max {l1['max_abs_delta_px']:.4f}, mean {l1['signed_mean_px']:+.4f} px, {res.n_renders} renders "
                  f"-> {'PASS' if l1['pass'] else 'FAIL'}")
            roi_results.append({
                "h_m": h,
                "roi": [u0, v0],
                "median_diff_px": med_diff,
                "p95_diff_px": p95_diff,
                "l1": l1,
                "n_renders": res.n_renders,
                "incomplete": res.incomplete,
                "ms_render": ms_render,
            })

    all_pass = bool(roi_results) and all(r["l1"]["pass"] and not r["incomplete"] for r in roi_results)
    return {"results": roi_results, "l1_limits": L1_LIMITS, "all_pass": all_pass}


def run_lit_renders(renderer: XVSRenderer, backend: str, out_dir: Path) -> dict[str, Any]:
    """Lit Render check: render C2 and C5 in rgb in T2 approach scene (WP-02 §3 G0 item 7)."""
    print("\n--- Running Lit Renders (C2 and C5 in T2 Approach Scene) ---")
    page_f = renderer.new_camera_page("F", profile="rgb")
    page_t0 = renderer.new_camera_page("T0", profile="rgb")

    thr_data = airports.load_ksfo_28l_threshold()
    true_rig = Rig.load_nominal().make_true(20260922)
    ring = TrafficRing()

    # T2 at t = 30s
    state = trajectory.t2_ksfo28l_state(30.0, thr=thr_data)
    cam_poses = true_rig.compute_poses(state.lat_deg, state.lon_deg, state.h_m, state.yaw_rad, state.pitch_rad, state.roll_rad)
    f_pose = cam_poses["F"]

    ring_poses = ring.compute_poses(state.lat_deg, state.lon_deg, state.h_m, state.yaw_rad, state.pitch_rad, state.roll_rad)
    t0_pose = ring_poses["T0"]

    renderer.set_camera(
        position=f_pose["position"],
        direction=f_pose["direction"],
        up=f_pose["up"],
        fov_rad=f_pose["fov_rad"],
        aspect_ratio=f_pose["aspect_ratio"],
        page=page_f,
    )
    renderer.set_camera(
        position=t0_pose["position"],
        direction=t0_pose["direction"],
        up=t0_pose["up"],
        fov_rad=math.radians(ring.fov_h_deg),
        aspect_ratio=1.0,
        page=page_t0,
    )

    # Place C2 300m ahead, C5 800m ahead
    p_f = np.array(f_pose["position"], dtype=np.float64)
    r_f = np.array(f_pose["r_ecef_cam"], dtype=np.float64)
    b_f = r_f[:, 2]

    p_c2 = p_f + 300.0 * b_f
    p_c5 = p_f + 800.0 * b_f

    m4_c2 = np.eye(4, dtype=np.float64)
    m4_c2[:3, :3] = r_f
    m4_c2[:3, 3] = p_c2
    c2_mat = m4_c2.T.flatten().tolist()

    m4_c5 = np.eye(4, dtype=np.float64)
    m4_c5[:3, :3] = r_f
    m4_c5[:3, 3] = p_c5
    c5_mat = m4_c5.T.flatten().tolist()

    c2_url = f"http://127.0.0.1:{renderer.port}/assets/traffic/C2_SEP/C2_SEP_0001.glb"
    c5_url = f"http://127.0.0.1:{renderer.port}/assets/traffic/C5_TRANSPORT/C5_TRANSPORT_0001.glb"

    renderer.clear_models(page=page_f)
    renderer.add_model(url=c2_url, model_matrix=c2_mat, page=page_f)
    renderer.add_model(url=c5_url, model_matrix=c5_mat, page=page_f)

    res_f = renderer.render_frame(page=page_f)
    png_f = out_dir / f"lit_render_f_{backend}.png"
    renderer.save_png(res_f.image, png_f)

    # Ring ROI render
    p_c2_cam_t0 = np.array(t0_pose["r_ecef_cam"], dtype=np.float64).T @ (p_c2 - np.array(t0_pose["position"]))
    u_proj, v_proj, _ = ring.camera_model.project_cam(p_c2_cam_t0)
    u0, v0 = ring.compute_roi(u_proj, v_proj)
    renderer.set_roi_frustum(u0=u0, v0=v0, fx=ring.fx, fy=ring.fy, cx=ring.cx, cy=ring.cy, page=page_t0)

    renderer.clear_models(page=page_t0)
    renderer.add_model(url=c2_url, model_matrix=c2_mat, page=page_t0)
    res_t0 = renderer.render_frame(page=page_t0)
    png_t0 = out_dir / f"lit_render_ring_t0_roi_{backend}.png"
    renderer.save_png(res_t0.image, png_t0)

    print(f"  Lit renders saved: {png_f.name}, {png_t0.name}")
    return {
        "rig_f_png": png_f.name,
        "ring_t0_png": png_t0.name,
    }


def run_g0_pipeline(backend: str = "ref", out_dir: str | Path | None = None) -> dict[str, Any]:
    out_path = Path(out_dir) if out_dir else (REPO_ROOT / "output" / "xvs" / "wp02" / "g0")
    out_path.mkdir(parents=True, exist_ok=True)

    port = get_free_port()
    cesium_build = REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium"
    page_dir = REPO_ROOT / "sim" / "cesium" / "page"
    cache_dir = REPO_ROOT / "data" / "cache" / "usgs_imagery"

    server = XVSServer(
        server_address=("127.0.0.1", port),
        cesium_build_dir=cesium_build,
        page_dir=page_dir,
        cache_dir=cache_dir,
        log_dir=out_path,
        offline=True,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()

    renderer = XVSRenderer(port=port, backend=backend, server=server).start()

    try:
        t_cal_res = run_t_cal(renderer, backend, out_path)
        t_roi_res = run_t_roi(renderer, backend, out_path)
        t_size_res = run_t_size(renderer, backend, out_path)
        t_occ_res = run_t_occ(renderer, backend, out_path)

        g3_reg_res = run_g3_regression(renderer, out_path) if backend == "ref" else {"pass": True}
        t_limb_roi_res = run_t_limb_roi(renderer, backend, out_path)
        lit_res = run_lit_renders(renderer, backend, out_path)

        all_pass = (
            t_cal_res["all_pass"]
            and t_roi_res["all_pass"]
            and t_size_res["all_pass"]
            and t_occ_res["pass"]
            and g3_reg_res["pass"]
            and t_limb_roi_res["all_pass"]
        )

        summary = {
            "backend": backend,
            "all_pass": all_pass,
            "t_cal": t_cal_res,
            "t_roi": t_roi_res,
            "t_size": t_size_res,
            "t_occ": t_occ_res,
            "g3_regression": g3_reg_res,
            "t_limb_roi": t_limb_roi_res,
            "lit_renders": lit_res,
        }

        with open(out_path / f"results_{backend}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=======================================================")
        print(f"WP-02 G0 Pipeline ({backend}): ALL PASS = {all_pass}")
        print(f"=======================================================")
        return summary

    finally:
        renderer.close()
        server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="WP-02 Gate G0 validation pipeline")
    parser.add_argument("--backend", default="ref", choices=["ref", "hw", "both"], help="Backend to run")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root data directory containing xvs/ output hierarchy (defaults to XVS_DATA_ROOT or repo output/)",
    )
    args = parser.parse_args()
    data_root = get_data_root(args.data_root)
    source = data_root_source(args.data_root)
    print(f"Data root: {data_root} (source: {source})")
    if args.data_root:
        set_data_root(args.data_root)

    backends = ["ref", "hw"] if args.backend == "both" else [args.backend]
    success = True
    for b in backends:
        res = run_g0_pipeline(backend=b, out_dir=args.out_dir)
        if not res["all_pass"] and b == "ref":
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
