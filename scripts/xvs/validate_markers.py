"""Reprojection and Camera Geometry Validator (Gate G2).

Executes M1, M2, and M3 validation across 30 scenes:
- P1-P3 across all 6 cameras (F, R, B, L, U, D) = 18 images
- Near-vertical set for U and D (6 near-vertical pitches x 2 cameras) = 12 images
Total: 30 images, each with 40 markers (1,200 markers total).

Evaluates both 'ref' (SwiftShader, hard pass required) and 'hw' (D3D11, reported as-is).
Saves every validation PNG alongside its exact pose JSON under output/xvs/wp01/g2/.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.2 §3 (G2), §4 and CONVENTIONS.md.
"""

from __future__ import annotations
import argparse
import colorsys
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.rig.rig import Rig, CAMERA_ORDER
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer

OUTPUT_G2_DIR = REPO_ROOT / "output" / "xvs" / "wp01" / "g2"
CESIUM_BUILD = REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium"
PAGE_DIR = REPO_ROOT / "sim" / "cesium" / "page"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "usgs_imagery"

# Normative Thresholds from WP01_SIM_RIG.md §4
THRESHOLDS = {
    "m1_detected": 40,
    "m1_mean_du": 0.05,
    "m1_mean_dv": 0.05,
    "m1_median_e": 0.10,
    "m1_max_e": 0.40,
    "m2_max_diff": 1e-3,
    "m3_pos_err_m": 1e-3,     # 1 mm
    "m3_dir_err_deg": 1e-6,   # 1e-6 deg
    "m3_up_err_deg": 1e-6,    # 1e-6 deg
}


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two 3D vectors in degrees via atan2(|a x b|, a . b) (C13)."""
    cross_norm = float(np.linalg.norm(np.cross(v1, v2)))
    dot_val = float(np.dot(v1, v2))
    return math.degrees(math.atan2(cross_norm, dot_val))


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_g2_scene_specs() -> list[dict[str, Any]]:
    """Construct the 30 scene specifications defined in §3 G2."""
    scenes = []

    # 1. Standard poses P1-P3 for all 6 cameras
    standard_poses = [
        {"name": "P1", "lat": 47.40, "lon": 8.50, "h": 600.0, "yaw": 30.0, "pitch": 0.0, "roll": 0.0},
        {"name": "P2", "lat": -33.90, "lon": 151.20, "h": 3000.0, "yaw": 200.0, "pitch": 5.0, "roll": -15.0},
        {"name": "P3", "lat": 64.10, "lon": -21.90, "h": 150.0, "yaw": 95.0, "pitch": -8.0, "roll": 25.0},
    ]

    for sp in standard_poses:
        for cam_id in CAMERA_ORDER:
            scene_id = f"{sp['name']}_{cam_id}"
            scenes.append({
                "scene_id": scene_id,
                "group": "standard",
                "pose_name": sp["name"],
                "camera_id": cam_id,
                "lat_deg": sp["lat"],
                "lon_deg": sp["lon"],
                "h_m": sp["h"],
                "yaw_deg": sp["yaw"],
                "pitch_deg": sp["pitch"],
                "roll_deg": sp["roll"],
            })

    # 2. Near-vertical set for U and D
    nv_pitches = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    for pitch in nv_pitches:
        for cam_id in ("U", "D"):
            pitch_str = f"{pitch:.1f}"
            scene_id = f"NV_pitch{pitch_str}_{cam_id}"
            scenes.append({
                "scene_id": scene_id,
                "group": "near_vertical",
                "pose_name": f"NV_pitch{pitch_str}",
                "camera_id": cam_id,
                "lat_deg": 37.62,
                "lon_deg": -122.37,
                "h_m": 1000.0,
                "yaw_deg": 40.0,
                "pitch_deg": pitch,
                "roll_deg": 0.0,
            })

    assert len(scenes) == 30, f"Expected 30 scenes, got {len(scenes)}"
    return scenes


def generate_scene_markers(
    rig: Rig,
    cam_pose: dict[str, Any],
    seed: int,
    n_markers: int = 40,
) -> list[dict[str, Any]]:
    """Generate 40 seeded markers placed in [20, 1003]^2 and back-projected to ECEF.

    Enforces §3 G2 requirements:
    - >= 10 markers at 15-50 m (we use exactly 10)
    - >= 10 markers at 1-5 km (we use exactly 10)
    - rest (20) at 100-500 m
    - unique colors per marker
    - minimum distance 25 px between marker centres
    """
    rng = np.random.default_rng(seed)
    cam_model = rig.camera_model

    # Target distances
    d_close = rng.uniform(15.0, 50.0, size=10)
    d_far = rng.uniform(1000.0, 5000.0, size=10)
    d_mid = rng.uniform(100.0, 500.0, size=20)
    distances = np.concatenate([d_close, d_far, d_mid])

    # Rejection sampling for (u, v) in [20, 1003]^2 with min_dist 25 px
    coords: list[tuple[float, float]] = []
    min_dist_sq = 25.0 * 25.0
    max_attempts = 10000

    for _ in range(n_markers):
        accepted = False
        for _ in range(max_attempts):
            u_cand = float(rng.uniform(20.0, 1003.0))
            v_cand = float(rng.uniform(20.0, 1003.0))

            too_close = False
            for prev_u, prev_v in coords:
                du = u_cand - prev_u
                dv = v_cand - prev_v
                if du * du + dv * dv < min_dist_sq:
                    too_close = True
                    break

            if not too_close:
                coords.append((u_cand, v_cand))
                accepted = True
                break

        if not accepted:
            # Fallback if area packed (should never happen for 40 points in 984^2)
            coords.append((float(rng.uniform(20.0, 1003.0)), float(rng.uniform(20.0, 1003.0))))

    markers = []
    pos_ecef_cam = cam_pose["position"]
    r_ecef_cam = cam_pose["r_ecef_cam"]

    for i in range(n_markers):
        u_gt, v_gt = coords[i]
        d = float(distances[i])

        # Unique bright color via golden-ratio HSV spacing
        hue = (i * 0.618033988749895) % 1.0
        r_col, g_col, b_col = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        color = [round(r_col, 4), round(g_col, 4), round(b_col, 4), 1.0]

        # Back-project to CAM frame, then to ECEF (CONVENTIONS §5, §6)
        ray_cam = cam_model.unproject(u_gt, v_gt)
        p_cam = ray_cam * d
        p_ecef = pos_ecef_cam + r_ecef_cam @ p_cam

        # Sanity check pinhole projection round-trip
        u_proj, v_proj, is_front = cam_model.project_cam(p_cam)
        assert is_front, f"Marker {i} must be in front of camera"
        assert abs(u_proj - u_gt) < 1e-9, f"Roundtrip u mismatch: {u_proj} vs {u_gt}"
        assert abs(v_proj - v_gt) < 1e-9, f"Roundtrip v mismatch: {v_proj} vs {v_gt}"

        markers.append({
            "id": i,
            "u_gt": float(u_gt),
            "v_gt": float(v_gt),
            "distance_m": float(d),
            "p_ecef": p_ecef.tolist(),
            "color": color,
        })

    return markers


def detect_marker_centroids(
    image_rgb: np.ndarray,
    markers: list[dict[str, Any]],
    window_radius: int = 7,
) -> list[dict[str, Any]]:
    """Compute intensity-weighted subpixel centroids in rendered image (M1)."""
    detection_results = []
    h, w, _ = image_rgb.shape

    for m in markers:
        u_gt = m["u_gt"]
        v_gt = m["v_gt"]
        ru = int(round(u_gt))
        rv = int(round(v_gt))

        r_min = max(0, rv - window_radius)
        r_max = min(h, rv + window_radius + 1)
        c_min = max(0, ru - window_radius)
        c_max = min(w, ru + window_radius + 1)

        patch = image_rgb[r_min:r_max, c_min:c_max]
        weights = patch.astype(np.float64).mean(axis=-1)
        tot_w = np.sum(weights)

        if tot_w > 10.0:
            gy, gx = np.indices(weights.shape)
            c_u = float(np.sum((gx + c_min) * weights) / tot_w)
            c_v = float(np.sum((gy + r_min) * weights) / tot_w)
            du = c_u - u_gt
            dv = c_v - v_gt
            e = math.hypot(du, dv)
            detected = True
        else:
            c_u = None
            c_v = None
            du = None
            dv = None
            e = None
            detected = False

        detection_results.append({
            "id": m["id"],
            "detected": detected,
            "u_gt": u_gt,
            "v_gt": v_gt,
            "c_u": c_u,
            "c_v": c_v,
            "du": du,
            "dv": dv,
            "e": e,
            "total_weight": float(tot_w),
        })

    return detection_results


def evaluate_scene(
    renderer: XVSRenderer,
    rig: Rig,
    scene_spec: dict[str, Any],
    out_backend_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Execute M1, M2, M3 validation for a single scene and save PNG + pose JSON."""
    cam_id = scene_spec["camera_id"]
    scene_id = scene_spec["scene_id"]

    # 1. Compute ground-truth camera pose
    poses = rig.compute_poses(
        lat_deg=scene_spec["lat_deg"],
        lon_deg=scene_spec["lon_deg"],
        h_m=scene_spec["h_m"],
        yaw_rad=math.radians(scene_spec["yaw_deg"]),
        pitch_rad=math.radians(scene_spec["pitch_deg"]),
        roll_rad=math.radians(scene_spec["roll_deg"]),
    )
    cam_pose = poses[cam_id]

    # 2. Generate 40 seeded markers
    markers = generate_scene_markers(rig, cam_pose, seed=seed, n_markers=40)

    # 3. Direct camera assignment in Cesium
    fov_rad = cam_pose["fov_rad"]
    aspect_ratio = cam_pose["aspect_ratio"]
    readback_set, ms_set_cam = renderer.set_camera(
        position=cam_pose["position"],
        direction=cam_pose["direction"],
        up=cam_pose["up"],
        fov_rad=fov_rad,
        aspect_ratio=aspect_ratio,
    )

    # 4. Upload markers to PointPrimitiveCollection
    renderer.set_markers(markers)

    # 5. M2 check: worldToWindowCoordinates - 0.5 vs pinhole projection
    projections = renderer.get_marker_projections(markers)
    m2_records = []
    m2_errors = []
    for m, proj in zip(markers, projections):
        if proj is not None:
            u_cesium = proj["u"]
            v_cesium = proj["v"]
            du_m2 = abs(u_cesium - m["u_gt"])
            dv_m2 = abs(v_cesium - m["v_gt"])
            max_m2 = max(du_m2, dv_m2)
            m2_errors.append(max_m2)
            m2_records.append({
                "id": m["id"],
                "u_cesium": u_cesium,
                "v_cesium": v_cesium,
                "x_cesium": proj["x_cesium"],
                "y_cesium": proj["y_cesium"],
                "du_m2": du_m2,
                "dv_m2": dv_m2,
                "max_err_px": max_m2,
            })
        else:
            m2_records.append({"id": m["id"], "error": "projection_failed"})

    m2_max_err = float(np.max(m2_errors)) if m2_errors else 999.0

    # 6. Render frame with in-page stability check
    render_res = renderer.render_frame(ms_set_camera=ms_set_cam)

    # 7. M1 check: intensity-weighted centroids in rendered image
    m1_detections = detect_marker_centroids(render_res.image, markers)
    detected_markers = [d for d in m1_detections if d["detected"]]
    n_detected = len(detected_markers)

    if n_detected == 40:
        dus = [d["du"] for d in detected_markers]
        dvs = [d["dv"] for d in detected_markers]
        es = [d["e"] for d in detected_markers]
        mean_du = float(np.mean(dus))
        mean_dv = float(np.mean(dvs))
        abs_mean_du = abs(mean_du)
        abs_mean_dv = abs(mean_dv)
        median_e = float(np.median(es))
        max_e = float(np.max(es))
    else:
        abs_mean_du = 999.0
        abs_mean_dv = 999.0
        median_e = 999.0
        max_e = 999.0
        mean_du = 999.0
        mean_dv = 999.0

    # 8. M3 check: Camera readback vs requested pose (both after set and after render, C13)
    pos_req = cam_pose["position"]
    dir_req = cam_pose["direction"]
    up_req = cam_pose["up"]

    # Readback 1: after setting camera
    pos_rb_set = np.array(readback_set["position"], dtype=np.float64)
    dir_rb_set = np.array(readback_set["direction"], dtype=np.float64)
    up_rb_set = np.array(readback_set["up"], dtype=np.float64)

    pos_err_set_m = float(np.linalg.norm(pos_rb_set - pos_req))
    dir_err_set_deg = angle_between_vectors(dir_rb_set, dir_req)
    up_err_set_deg = angle_between_vectors(up_rb_set, up_req)

    # Readback 2: after final render
    readback_render = render_res.readback
    pos_rb_render = np.array(readback_render["position"], dtype=np.float64)
    dir_rb_render = np.array(readback_render["direction"], dtype=np.float64)
    up_rb_render = np.array(readback_render["up"], dtype=np.float64)

    pos_err_render_m = float(np.linalg.norm(pos_rb_render - pos_req))
    dir_err_render_deg = angle_between_vectors(dir_rb_render, dir_req)
    up_err_render_deg = angle_between_vectors(up_rb_render, up_req)

    m3_set_pass = (
        (pos_err_set_m <= THRESHOLDS["m3_pos_err_m"]) and
        (dir_err_set_deg <= THRESHOLDS["m3_dir_err_deg"]) and
        (up_err_set_deg <= THRESHOLDS["m3_up_err_deg"])
    )
    m3_render_pass = (
        (pos_err_render_m <= THRESHOLDS["m3_pos_err_m"]) and
        (dir_err_render_deg <= THRESHOLDS["m3_dir_err_deg"]) and
        (up_err_render_deg <= THRESHOLDS["m3_up_err_deg"])
    )
    m3_pass = (m3_set_pass and m3_render_pass)

    pos_err_m = max(pos_err_set_m, pos_err_render_m)
    dir_err_deg = max(dir_err_set_deg, dir_err_render_deg)
    up_err_deg = max(up_err_set_deg, up_err_render_deg)

    # 9. Verify thresholds
    m1_pass = (
        (n_detected == THRESHOLDS["m1_detected"]) and
        (abs_mean_du <= THRESHOLDS["m1_mean_du"]) and
        (abs_mean_dv <= THRESHOLDS["m1_mean_dv"]) and
        (median_e <= THRESHOLDS["m1_median_e"]) and
        (max_e <= THRESHOLDS["m1_max_e"])
    )
    m2_pass = (m2_max_err <= THRESHOLDS["m2_max_diff"])
    all_pass = (m1_pass and m2_pass and m3_pass)

    # 10. Save PNG and Pose JSON
    png_dir = out_backend_dir / "pngs"
    pose_dir = out_backend_dir / "poses"
    png_path = png_dir / f"{scene_id}.png"
    pose_path = pose_dir / f"{scene_id}_pose.json"

    file_size_bytes, ms_encode = renderer.save_png(render_res.image, png_path)

    # Consolidate per-marker results
    marker_details = []
    for m, det, m2_r in zip(markers, m1_detections, m2_records):
        marker_details.append({
            "id": m["id"],
            "u_gt": m["u_gt"],
            "v_gt": m["v_gt"],
            "distance_m": m["distance_m"],
            "p_ecef": m["p_ecef"],
            "color": m["color"],
            "detected": det["detected"],
            "c_u": det["c_u"],
            "c_v": det["c_v"],
            "du": det["du"],
            "dv": det["dv"],
            "e": det["e"],
            "m2_u_cesium": m2_r.get("u_cesium"),
            "m2_v_cesium": m2_r.get("v_cesium"),
            "m2_err_px": m2_r.get("max_err_px"),
        })

    rig_meta = rig.record()

    pose_artifact = {
        "schema": "xvs.marker_pose/1",
        "backend": renderer.backend,
        "scene_id": scene_id,
        "group": scene_spec["group"],
        "pose_name": scene_spec["pose_name"],
        "camera_id": cam_id,
        "rig": rig_meta,
        "drawing_buffer": list(render_res.drawing_buffer),
        "rgb_sha256": render_res.rgb_sha256,
        "aircraft_state": {
            "lat_deg": scene_spec["lat_deg"],
            "lon_deg": scene_spec["lon_deg"],
            "h_m": scene_spec["h_m"],
            "yaw_deg": scene_spec["yaw_deg"],
            "pitch_deg": scene_spec["pitch_deg"],
            "roll_deg": scene_spec["roll_deg"],
        },
        "camera_pose_requested": {
            "position": cam_pose["position"].tolist(),
            "direction": cam_pose["direction"].tolist(),
            "up": cam_pose["up"].tolist(),
            "right": cam_pose["right"].tolist(),
            "fov_rad": fov_rad,
            "aspect_ratio": aspect_ratio,
        },
        "camera_readback": readback_render,
        "camera_readback_set": readback_set,
        "camera_readback_render": readback_render,
        "metrics": {
            "m1": {
                "detected": n_detected,
                "total": 40,
                "abs_mean_du": abs_mean_du,
                "abs_mean_dv": abs_mean_dv,
                "median_e": median_e,
                "max_e": max_e,
                "pass": m1_pass,
            },
            "m2": {
                "max_err_px": m2_max_err,
                "pass": m2_pass,
            },
            "m3": {
                "pos_err_m": pos_err_m,
                "dir_err_deg": dir_err_deg,
                "up_err_deg": up_err_deg,
                "set_pos_err_m": pos_err_set_m,
                "set_dir_err_deg": dir_err_set_deg,
                "set_up_err_deg": up_err_set_deg,
                "render_pos_err_m": pos_err_render_m,
                "render_dir_err_deg": dir_err_render_deg,
                "render_up_err_deg": up_err_render_deg,
                "m3_set_pass": m3_set_pass,
                "m3_render_pass": m3_render_pass,
                "pass": m3_pass,
            },
            "overall_pass": all_pass,
        },
        "render_stats": {
            "n_renders": render_res.n_renders,
            "stable": render_res.stable,
            "incomplete": render_res.incomplete,
            "last_max_diff": render_res.last_max_diff,
            "last_n_diff": render_res.last_n_diff,
            "ms_set_camera": round(render_res.ms_set_camera, 2),
            "ms_render_loop": round(render_res.ms_render_loop, 2),
            "ms_readback": round(render_res.ms_readback, 2),
            "ms_encode": round(ms_encode, 2),
            "file_size_bytes": file_size_bytes,
            "drawing_buffer": list(render_res.drawing_buffer),
            "rgb_sha256": render_res.rgb_sha256,
        },
        "png_path": str(png_path.relative_to(REPO_ROOT)).replace("\\", "/") if png_path.is_relative_to(REPO_ROOT) else str(png_path).replace("\\", "/"),
        "markers": marker_details,
    }

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.write_text(json.dumps(pose_artifact, indent=2), encoding="utf-8")

    rel_png = str(png_path.relative_to(REPO_ROOT)).replace("\\", "/") if png_path.is_relative_to(REPO_ROOT) else str(png_path).replace("\\", "/")
    rel_pose = str(pose_path.relative_to(REPO_ROOT)).replace("\\", "/") if pose_path.is_relative_to(REPO_ROOT) else str(pose_path).replace("\\", "/")

    return {
        "scene_id": scene_id,
        "group": scene_spec["group"],
        "camera_id": cam_id,
        "rig": rig_meta,
        "drawing_buffer": list(render_res.drawing_buffer),
        "rgb_sha256": render_res.rgb_sha256,
        "m1_detected": n_detected,
        "m1_abs_mean_du": abs_mean_du,
        "m1_abs_mean_dv": abs_mean_dv,
        "m1_median_e": median_e,
        "m1_max_e": max_e,
        "m2_max_err": m2_max_err,
        "m3_pos_err_m": pos_err_m,
        "m3_dir_err_deg": dir_err_deg,
        "m3_up_err_deg": up_err_deg,
        "m1_pass": m1_pass,
        "m2_pass": m2_pass,
        "m3_pass": m3_pass,
        "all_pass": all_pass,
        "n_renders": render_res.n_renders,
        "ms_render_loop": render_res.ms_render_loop,
        "ms_readback": render_res.ms_readback,
        "png_path": rel_png,
        "pose_json_path": rel_pose,
    }


def run_marker_validation_for_backend(
    backend: str,
    server: XVSServer,
    port: int,
    out_dir: Path = OUTPUT_G2_DIR,
) -> dict[str, Any]:
    """Execute complete M1-M3 validation across all 30 scenes for one backend."""
    print(f"\n=======================================================")
    print(f"Executing G2 Marker Validation on Backend: '{backend}'")
    print(f"=======================================================")

    backend_dir = out_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    (backend_dir / "pngs").mkdir(parents=True, exist_ok=True)
    (backend_dir / "poses").mkdir(parents=True, exist_ok=True)

    rig = Rig.load_nominal().make_true(20260922)
    scene_specs = build_g2_scene_specs()

    scene_results = []
    t0_backend = time.perf_counter()

    with XVSRenderer(port=port, backend=backend, server=server) as renderer:
        diagnostics = renderer.get_webgl_diagnostics()
        print(f"[{backend}] Vendor: {diagnostics.get('unmaskedVendor')}, Renderer: {diagnostics.get('unmaskedRenderer')}")

        renderer.set_profile("markers")

        for idx, sc in enumerate(scene_specs):
            t_s0 = time.perf_counter()
            scene_seed = 20260923 + idx * 100
            res = evaluate_scene(
                renderer=renderer,
                rig=rig,
                scene_spec=sc,
                out_backend_dir=backend_dir,
                seed=scene_seed,
            )
            scene_results.append(res)
            dt_s = (time.perf_counter() - t_s0) * 1000.0

            status_str = "PASS" if res["all_pass"] else "FAIL"
            print(
                f"[{backend}] ({idx+1:02d}/30) {res['scene_id']:<18} | "
                f"M1: {res['m1_detected']}/40, med|e|={res['m1_median_e']:.4f}px, max|e|={res['m1_max_e']:.4f}px | "
                f"M2: {res['m2_max_err']:.2e}px | "
                f"M3: pos={res['m3_pos_err_m']:.2e}m, dir={res['m3_dir_err_deg']:.2e}° | "
                f"{status_str} ({dt_s:.0f} ms)"
            )

    total_wall_s = time.perf_counter() - t0_backend

    # Compute aggregate metrics across all 30 scenes
    all_scenes_pass = all(r["all_pass"] for r in scene_results)
    m1_scenes_pass = all(r["m1_pass"] for r in scene_results)
    m2_scenes_pass = all(r["m2_pass"] for r in scene_results)
    m3_scenes_pass = all(r["m3_pass"] for r in scene_results)

    total_detected = sum(r["m1_detected"] for r in scene_results)
    total_expected = len(scene_results) * 40
    worst_m1_mean_du = max(r["m1_abs_mean_du"] for r in scene_results)
    worst_m1_mean_dv = max(r["m1_abs_mean_dv"] for r in scene_results)
    worst_m1_median_e = max(r["m1_median_e"] for r in scene_results)
    worst_m1_max_e = max(r["m1_max_e"] for r in scene_results)
    worst_m2_err = max(r["m2_max_err"] for r in scene_results)
    worst_m3_pos_err = max(r["m3_pos_err_m"] for r in scene_results)
    worst_m3_dir_err = max(r["m3_dir_err_deg"] for r in scene_results)
    worst_m3_up_err = max(r["m3_up_err_deg"] for r in scene_results)

    backend_summary = {
        "backend": backend,
        "rig": rig.record(),
        "drawing_buffer": [1024, 1024],
        "webgl_diagnostics": diagnostics,
        "n_scenes": len(scene_results),
        "total_markers": total_expected,
        "total_detected": total_detected,
        "all_scenes_pass": all_scenes_pass,
        "m1_all_pass": m1_scenes_pass,
        "m2_all_pass": m2_scenes_pass,
        "m3_all_pass": m3_scenes_pass,
        "worst_metrics": {
            "m1_detected_min": min(r["m1_detected"] for r in scene_results),
            "m1_abs_mean_du_max": round(worst_m1_mean_du, 5),
            "m1_abs_mean_dv_max": round(worst_m1_mean_dv, 5),
            "m1_median_e_max": round(worst_m1_median_e, 5),
            "m1_max_e_max": round(worst_m1_max_e, 5),
            "m2_max_err_max": worst_m2_err,
            "m3_pos_err_m_max": worst_m3_pos_err,
            "m3_dir_err_deg_max": worst_m3_dir_err,
            "m3_up_err_deg_max": worst_m3_up_err,
        },
        "thresholds": THRESHOLDS,
        "total_wall_time_s": round(total_wall_s, 2),
        "mean_scene_ms": round(total_wall_s / len(scene_results) * 1000.0, 1),
        "scenes": scene_results,
    }

    out_file = out_dir / f"results_{backend}.json"
    out_file.write_text(json.dumps(backend_summary, indent=2), encoding="utf-8")
    print(f"Saved {backend} results to: {out_file}")

    return backend_summary


def run_g2_validation(
    backends: list[str] | None = None,
    out_dir: Path = OUTPUT_G2_DIR,
) -> dict[str, Any]:
    """Execute complete Gate G2 validation on ref and hw backends."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_backends = backends or ["ref", "hw"]

    port = get_free_port()
    server = XVSServer(
        server_address=("127.0.0.1", port),
        cesium_build_dir=CESIUM_BUILD,
        page_dir=PAGE_DIR,
        cache_dir=CACHE_DIR,
        log_dir=out_dir / "logs",
        offline=True,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Started threaded XVSServer on 127.0.0.1:{port}")

    combined_results: dict[str, Any] = {
        "schema": "xvs.g2_validation/1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rig": {"kind": "true", "seed": 20260922},
        "drawing_buffer": [1024, 1024],
        "backends": {},
    }

    try:
        for b in target_backends:
            res = run_marker_validation_for_backend(b, server, port, out_dir=out_dir)
            combined_results["backends"][b] = res

        summary_file = out_dir / "results.json"
        summary_file.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
        print(f"\nSuccessfully wrote consolidated G2 results to: {summary_file}")

    finally:
        server.shutdown()
        server.server_close()
        print("Stopped XVSServer.")

    return combined_results


def main() -> None:
    parser = argparse.ArgumentParser(description="WP-01 Gate G2 Reprojection and Marker Validation")
    parser.add_argument("--backend", choices=["ref", "hw", "both"], default="both", help="Renderer backend to evaluate")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_G2_DIR, help="Output directory for G2 artifacts")
    args = parser.parse_args()

    backends = ["ref", "hw"] if args.backend == "both" else [args.backend]
    res = run_g2_validation(backends=backends, out_dir=args.out_dir)

    # Enforce spec rule: ref must pass; hw is reported as-is
    if "ref" in res["backends"]:
        ref_pass = res["backends"]["ref"]["all_scenes_pass"]
        if not ref_pass:
            print("\n[ERROR] Gate G2 FAIL: Backend 'ref' failed one or more hard thresholds!")
            sys.exit(1)
        else:
            print("\n[SUCCESS] Gate G2 PASS: Backend 'ref' passed all hard thresholds!")

    if "hw" in res["backends"]:
        hw_pass = res["backends"]["hw"]["all_scenes_pass"]
        print(f"[REPORT] Backend 'hw' pass status: {hw_pass}")


if __name__ == "__main__":
    main()
