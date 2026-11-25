"""Analytic Earth-Limb Geometry and Zero-Dip Control Validator (Gate G3).

Executes L1 and N1 validation across Earth-limb scenes:
- L1 horizontal set: 128 scenes (F, R, B, L x 4 altitudes x 4 (pitch, roll) x 2 yaws) in mask_earth (MSAA 4)
- L1 near-vertical / U and D set: 16 scenes (U, D x 2 altitudes x 4 attitudes) in mask_earth (MSAA 4)
  Only scored if analytic limb spans >= 200 px.
- N1 negative control: All 8 level flight 600m images compared against zero-dip local-level plane.
  Median |Δ| MUST be >= 3.0 px (failing L1 limits by a large margin).
- Informational: 8 images in mask_earth_binary (MSAA 1) documenting the +/-0.5 px quantization floor.

Evaluates both 'ref' (SwiftShader, hard pass required) and 'hw' (D3D11, reported as-is).
Saves every mask PNG alongside its exact pose JSON under output/xvs/wp01/g3/{ref,hw}/.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.3 §2.6, §3 (G3), §4, §5 and CONVENTIONS.md.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.rig.rig import Rig
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer
from xvs.sim.horizon import (
    compute_horizon_gt,
    estimate_horizon_edge,
)

OUTPUT_G3_DIR = REPO_ROOT / "output" / "xvs" / "wp01" / "g3"
CESIUM_BUILD = REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium"
PAGE_DIR = REPO_ROOT / "sim" / "cesium" / "page"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "usgs_imagery"

# Normative Thresholds from WP01_SIM_RIG.md v1.3 §4
THRESHOLDS = {
    "m3_pos_err_m": 1e-3,            # 1 mm
    "m3_dir_err_deg": 1e-6,          # 1e-6 deg
    "m3_up_err_deg": 1e-6,           # 1e-6 deg
    "l1_median_abs_delta": 0.15,     # <= 0.15 px
    "l1_p95_abs_delta": 0.35,        # <= 0.35 px
    "l1_max_abs_delta": 0.50,        # <= 0.50 px
    "l1_abs_signed_mean": 0.05,      # <= 0.05 px
    "n1_median_abs_delta_min": 3.0,  # >= 3.0 px (must fail L1)
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


def build_g3_scenes() -> list[dict[str, Any]]:
    """Construct all scene specifications defined in §3 G3."""
    scenes = []

    # 1. Horizontal cameras F, R, B, L (128 images)
    altitudes_h = [150.0, 600.0, 3000.0, 10000.0]
    attitudes_h = [(0.0, 0.0), (-5.0, 0.0), (0.0, 20.0), (5.0, -30.0)]
    yaws_h = [0.0, 137.0]
    cams_h = ["F", "R", "B", "L"]

    for alt in altitudes_h:
        for pitch, roll in attitudes_h:
            for yaw in yaws_h:
                for cam_id in cams_h:
                    p_str = f"p{int(pitch)}" if pitch.is_integer() else f"p{pitch:.1f}"
                    r_str = f"r{int(roll)}" if roll.is_integer() else f"r{roll:.1f}"
                    y_str = f"y{int(yaw)}"
                    h_str = f"h{int(alt)}"
                    scene_id = f"H_{cam_id}_{h_str}_{p_str}_{r_str}_{y_str}"
                    is_n1 = (alt == 600.0 and pitch == 0.0 and roll == 0.0)
                    scenes.append({
                        "scene_id": scene_id,
                        "group": "horizontal",
                        "camera_id": cam_id,
                        "lat_deg": 37.62,
                        "lon_deg": -122.37,
                        "h_m": alt,
                        "yaw_deg": yaw,
                        "pitch_deg": pitch,
                        "roll_deg": roll,
                        "is_n1_control": is_n1,
                        "profile": "mask_earth",
                    })

    assert len(scenes) == 128, f"Expected 128 horizontal scenes, got {len(scenes)}"

    # 2. Near-vertical U and D cameras (16 images)
    altitudes_ud = [600.0, 10000.0]
    attitudes_ud = [(0.0, 70.0), (0.0, -70.0), (60.0, 0.0), (-60.0, 0.0)]
    cams_ud = ["U", "D"]
    yaw_ud = 0.0

    for alt in altitudes_ud:
        for pitch, roll in attitudes_ud:
            for cam_id in cams_ud:
                p_str = f"p{int(pitch)}" if pitch.is_integer() else f"p{pitch:.1f}"
                r_str = f"r{int(roll)}" if roll.is_integer() else f"r{roll:.1f}"
                h_str = f"h{int(alt)}"
                scene_id = f"UD_{cam_id}_{h_str}_{p_str}_{r_str}_y0"
                scenes.append({
                    "scene_id": scene_id,
                    "group": "ud",
                    "camera_id": cam_id,
                    "lat_deg": 37.62,
                    "lon_deg": -122.37,
                    "h_m": alt,
                    "yaw_deg": yaw_ud,
                    "pitch_deg": pitch,
                    "roll_deg": roll,
                    "is_n1_control": False,
                    "profile": "mask_earth",
                })

    assert len(scenes) == 128 + 16, f"Expected 144 L1 scenes, got {len(scenes)}"

    # 3. Informational binary mask scenes (8 images)
    for yaw in [0.0, 137.0]:
        for cam_id in cams_h:
            scene_id = f"BIN_{cam_id}_h600_p0_r0_y{int(yaw)}"
            scenes.append({
                "scene_id": scene_id,
                "group": "informational_binary",
                "camera_id": cam_id,
                "lat_deg": 37.62,
                "lon_deg": -122.37,
                "h_m": 600.0,
                "yaw_deg": yaw,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "is_n1_control": False,
                "profile": "mask_earth_binary",
            })

    assert len(scenes) == 144 + 8, f"Expected 152 total scenes, got {len(scenes)}"
    return scenes


def evaluate_limb_scene(
    renderer: XVSRenderer,
    rig: Rig,
    scene_spec: dict[str, Any],
    out_backend_dir: Path,
) -> dict[str, Any]:
    """Execute Earth-limb evaluation for one scene."""
    scene_id = scene_spec["scene_id"]
    cam_id = scene_spec["camera_id"]
    profile = scene_spec["profile"]

    # 1. Switch profile if different from active profile
    if renderer._active_profile != profile:
        renderer.set_profile(profile)

    # 2. Compute true camera pose from aircraft state
    poses = rig.compute_poses(
        lat_deg=scene_spec["lat_deg"],
        lon_deg=scene_spec["lon_deg"],
        h_m=scene_spec["h_m"],
        yaw_rad=math.radians(scene_spec["yaw_deg"]),
        pitch_rad=math.radians(scene_spec["pitch_deg"]),
        roll_rad=math.radians(scene_spec["roll_deg"]),
    )
    cam_pose = poses[cam_id]
    fov_rad = cam_pose["fov_rad"]
    aspect_ratio = cam_pose["aspect_ratio"]
    cam_model = rig.camera_model

    # 3. Direct camera assignment in Cesium
    readback_set, ms_set_cam = renderer.set_camera(
        position=cam_pose["position"],
        direction=cam_pose["direction"],
        up=cam_pose["up"],
        fov_rad=fov_rad,
        aspect_ratio=aspect_ratio,
    )

    # 4. Render frame with in-page stability check
    render_res = renderer.render_frame(ms_set_camera=ms_set_cam)
    readback_render = render_res.readback

    # 5. Compute analytic horizon GT
    p_cam = cam_pose["p_ecef_cam"]
    r_cam = cam_pose["r_ecef_cam"]
    horizon_gt = compute_horizon_gt(p_cam, r_cam, cam_model)

    # Span check for U and D cameras (§3 G3): score only images where analytic limb spans >= 200 px
    limb_span_px = 0.0
    if horizon_gt["horizon_visible"] and len(horizon_gt["horizon_uv"]) > 1:
        uv_pts = np.array(horizon_gt["horizon_uv"])
        span_u = float(uv_pts[:, 0].max() - uv_pts[:, 0].min())
        span_v = float(uv_pts[:, 1].max() - uv_pts[:, 1].min())
        limb_span_px = max(span_u, span_v)

    score_image = True
    if scene_spec["group"] == "ud":
        score_image = (horizon_gt["horizon_visible"] and limb_span_px >= 200.0)

    # 6. Estimate horizon edge using coverage method (§2.6)
    edge_res = estimate_horizon_edge(
        mask_img=render_res.image,
        p_ecef_cam=p_cam,
        r_ecef_cam=r_cam,
        cam=cam_model,
        zero_dip_control=False,
    )

    # 7. N1 negative control evaluation if applicable
    n1_res: dict[str, Any] | None = None
    if scene_spec.get("is_n1_control", False):
        n1_res = estimate_horizon_edge(
            mask_img=render_res.image,
            p_ecef_cam=p_cam,
            r_ecef_cam=r_cam,
            cam=cam_model,
            zero_dip_control=True,
        )

    # 8. M3 check: Camera readback vs requested pose (both after set and after render, C13)
    pos_req = cam_pose["position"]
    dir_req = cam_pose["direction"]
    up_req = cam_pose["up"]

    pos_rb_set = np.array(readback_set["position"], dtype=np.float64)
    dir_rb_set = np.array(readback_set["direction"], dtype=np.float64)
    up_rb_set = np.array(readback_set["up"], dtype=np.float64)

    pos_err_set_m = float(np.linalg.norm(pos_rb_set - pos_req))
    dir_err_set_deg = angle_between_vectors(dir_rb_set, dir_req)
    up_err_set_deg = angle_between_vectors(up_rb_set, up_req)

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

    # 9. Verify L1 thresholds
    l1_median = edge_res["median_abs_delta"]
    l1_p95 = edge_res["p95_abs_delta"]
    l1_max = edge_res["max_abs_delta"]
    l1_signed_mean = edge_res["signed_mean"]
    l1_abs_signed_mean = abs(l1_signed_mean)

    l1_pass = (
        (l1_median <= THRESHOLDS["l1_median_abs_delta"]) and
        (l1_p95 <= THRESHOLDS["l1_p95_abs_delta"]) and
        (l1_max <= THRESHOLDS["l1_max_abs_delta"]) and
        (l1_abs_signed_mean <= THRESHOLDS["l1_abs_signed_mean"])
    )

    # N1 pass: must fail L1 by a large margin (median >= 3.0 px)
    n1_median = n1_res["median_abs_delta"] if n1_res else None
    n1_pass = (n1_median >= THRESHOLDS["n1_median_abs_delta_min"]) if n1_median is not None else True

    all_pass = m3_pass and (l1_pass if score_image else True) and n1_pass

    # 10. Save PNG and Pose JSON
    png_dir = out_backend_dir / "pngs"
    pose_dir = out_backend_dir / "poses"
    png_path = png_dir / f"{scene_id}.png"
    pose_path = pose_dir / f"{scene_id}_pose.json"

    file_size_bytes, ms_encode = renderer.save_png(render_res.image, png_path)

    rig_meta = rig.record()

    pose_artifact = {
        "schema": "xvs.limb_pose/1",
        "backend": renderer.backend,
        "scene_id": scene_id,
        "group": scene_spec["group"],
        "camera_id": cam_id,
        "profile": profile,
        "resolved_profile": renderer.resolved_settings,
        "score_image": score_image,
        "limb_span_px": round(limb_span_px, 2),
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
        "gt": horizon_gt,
        "edge_estimation": {
            "lines_used": edge_res["lines_used"],
            "lines_skipped": edge_res["lines_skipped"],
            "median_abs_delta": round(l1_median, 5),
            "p95_abs_delta": round(l1_p95, 5),
            "max_abs_delta": round(l1_max, 5),
            "signed_mean": round(l1_signed_mean, 5),
            "edges_measured": edge_res["edges_measured"],
        },
        "n1": {
            "is_n1_control": scene_spec.get("is_n1_control", False),
            "median_abs_delta": round(n1_median, 5) if n1_median is not None else None,
            "pass": n1_pass,
        } if scene_spec.get("is_n1_control", False) else None,
        "metrics": {
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
            "l1": {
                "median_abs_delta": round(l1_median, 5),
                "p95_abs_delta": round(l1_p95, 5),
                "max_abs_delta": round(l1_max, 5),
                "signed_mean": round(l1_signed_mean, 5),
                "pass": l1_pass,
                "scored": score_image,
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
    }

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.write_text(json.dumps(pose_artifact, indent=2), encoding="utf-8")

    rel_png = str(png_path.relative_to(REPO_ROOT)).replace("\\", "/") if png_path.is_relative_to(REPO_ROOT) else str(png_path).replace("\\", "/")
    rel_pose = str(pose_path.relative_to(REPO_ROOT)).replace("\\", "/") if pose_path.is_relative_to(REPO_ROOT) else str(pose_path).replace("\\", "/")

    return {
        "scene_id": scene_id,
        "group": scene_spec["group"],
        "camera_id": cam_id,
        "profile": profile,
        "score_image": score_image,
        "limb_span_px": round(limb_span_px, 2),
        "rig": rig_meta,
        "drawing_buffer": list(render_res.drawing_buffer),
        "rgb_sha256": render_res.rgb_sha256,
        "lines_used": edge_res["lines_used"],
        "lines_skipped": edge_res["lines_skipped"],
        "l1_median_abs_delta": round(l1_median, 5),
        "l1_p95_abs_delta": round(l1_p95, 5),
        "l1_max_abs_delta": round(l1_max, 5),
        "l1_signed_mean": round(l1_signed_mean, 5),
        "l1_pass": l1_pass,
        "is_n1_control": scene_spec.get("is_n1_control", False),
        "n1_median_abs_delta": round(n1_median, 5) if n1_median is not None else None,
        "n1_pass": n1_pass,
        "m3_pos_err_m": pos_err_m,
        "m3_dir_err_deg": dir_err_deg,
        "m3_up_err_deg": up_err_deg,
        "m3_pass": m3_pass,
        "all_pass": all_pass,
        "n_renders": render_res.n_renders,
        "ms_render_loop": render_res.ms_render_loop,
        "ms_readback": render_res.ms_readback,
        "png_path": rel_png,
        "pose_json_path": rel_pose,
    }


def run_limb_validation_for_backend(
    backend: str,
    server: XVSServer,
    port: int,
    out_dir: Path = OUTPUT_G3_DIR,
) -> dict[str, Any]:
    """Execute complete Gate G3 validation for one backend."""
    print(f"\n=======================================================")
    print(f"Executing G3 Limb Validation on Backend: '{backend}'")
    print(f"=======================================================")

    backend_dir = out_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    (backend_dir / "pngs").mkdir(parents=True, exist_ok=True)
    (backend_dir / "poses").mkdir(parents=True, exist_ok=True)

    rig = Rig.load_nominal().make_true(20260922)
    scene_specs = build_g3_scenes()

    scene_results = []
    t0_backend = time.perf_counter()

    with XVSRenderer(port=port, backend=backend, server=server) as renderer:
        diagnostics = renderer.get_webgl_diagnostics()
        print(f"[{backend}] Vendor: {diagnostics.get('unmaskedVendor')}, Renderer: {diagnostics.get('unmaskedRenderer')}")

        # C15: Resolve and store all 4 profile configurations from renderer
        resolved_profiles = {}
        for p_name in ("rgb", "mask_earth", "mask_earth_binary", "markers"):
            resolved_profiles[p_name] = renderer.set_profile(p_name)

        renderer.set_profile("mask_earth")

        for idx, sc in enumerate(scene_specs):
            t_s0 = time.perf_counter()
            res = evaluate_limb_scene(
                renderer=renderer,
                rig=rig,
                scene_spec=sc,
                out_backend_dir=backend_dir,
            )
            scene_results.append(res)
            dt_s = (time.perf_counter() - t_s0) * 1000.0

            status_str = "PASS" if res["all_pass"] else "FAIL"
            n1_str = f" | N1 med|delta|={res['n1_median_abs_delta']:.2f}px" if res["is_n1_control"] else ""
            print(
                f"[{backend}] ({idx+1:03d}/{len(scene_specs)}) {res['scene_id']:<24} | "
                f"L1: med={res['l1_median_abs_delta']:.4f}px, p95={res['l1_p95_abs_delta']:.4f}px, max={res['l1_max_abs_delta']:.4f}px, mean={res['l1_signed_mean']:+.4f}px | "
                f"lines={res['lines_used']}/{res['lines_used']+res['lines_skipped']}{n1_str} | "
                f"{status_str} ({dt_s:.0f} ms)"
            )

    total_wall_s = time.perf_counter() - t0_backend

    # Separate scored L1 scenes (horizontal + scored U/D) from informational binary
    l1_scenes = [r for r in scene_results if r["group"] in ("horizontal", "ud") and r["score_image"]]
    n1_scenes = [r for r in scene_results if r["is_n1_control"]]
    binary_scenes = [r for r in scene_results if r["group"] == "informational_binary"]

    all_l1_pass = all(r["l1_pass"] for r in l1_scenes)
    all_n1_pass = all(r["n1_pass"] for r in n1_scenes)
    all_m3_pass = all(r["m3_pass"] for r in scene_results)
    overall_backend_pass = all_l1_pass and all_n1_pass and all_m3_pass

    worst_l1_median = max(r["l1_median_abs_delta"] for r in l1_scenes)
    worst_l1_p95 = max(r["l1_p95_abs_delta"] for r in l1_scenes)
    worst_l1_max = max(r["l1_max_abs_delta"] for r in l1_scenes)
    worst_l1_abs_signed_mean = max(abs(r["l1_signed_mean"]) for r in l1_scenes)

    min_n1_median = min(r["n1_median_abs_delta"] for r in n1_scenes)
    mean_n1_median = float(np.mean([r["n1_median_abs_delta"] for r in n1_scenes]))

    total_lines_used = sum(r["lines_used"] for r in l1_scenes)
    total_lines_skipped = sum(r["lines_skipped"] for r in l1_scenes)

    binary_medians = [r["l1_median_abs_delta"] for r in binary_scenes]

    backend_summary = {
        "backend": backend,
        "rig": rig.record(),
        "drawing_buffer": [1024, 1024],
        "render_profiles": resolved_profiles,
        "resolved_settings": resolved_profiles,
        "webgl_diagnostics": diagnostics,
        "n_total_scenes": len(scene_results),
        "n_l1_scored_scenes": len(l1_scenes),
        "n_n1_scenes": len(n1_scenes),
        "n_binary_scenes": len(binary_scenes),
        "all_scenes_pass": overall_backend_pass,
        "l1_all_pass": all_l1_pass,
        "n1_all_pass": all_n1_pass,
        "m3_all_pass": all_m3_pass,
        "l1_worst_metrics": {
            "median_abs_delta_max": round(worst_l1_median, 5),
            "p95_abs_delta_max": round(worst_l1_p95, 5),
            "max_abs_delta_max": round(worst_l1_max, 5),
            "abs_signed_mean_max": round(worst_l1_abs_signed_mean, 5),
        },
        "n1_metrics": {
            "min_median_abs_delta": round(min_n1_median, 5),
            "mean_median_abs_delta": round(mean_n1_median, 5),
            "threshold_min": THRESHOLDS["n1_median_abs_delta_min"],
            "pass": all_n1_pass,
        },
        "coverage_lines": {
            "total_lines_used": total_lines_used,
            "total_lines_skipped": total_lines_skipped,
            "mean_lines_used_per_image": round(total_lines_used / len(l1_scenes), 1),
            "mean_lines_skipped_per_image": round(total_lines_skipped / len(l1_scenes), 1),
        },
        "informational_binary_quantization": {
            "mean_median_abs_delta": round(float(np.mean(binary_medians)), 5),
            "max_median_abs_delta": round(float(np.max(binary_medians)), 5),
            "min_median_abs_delta": round(float(np.min(binary_medians)), 5),
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


def run_g3_validation(
    backends: list[str] | None = None,
    out_dir: Path = OUTPUT_G3_DIR,
) -> dict[str, Any]:
    """Execute complete Gate G3 Earth-limb validation on ref and hw backends."""
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
        "schema": "xvs.g3_validation/1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rig": {"kind": "true", "seed": 20260922},
        "drawing_buffer": [1024, 1024],
        "render_profiles": {},
        "resolved_settings": {},
        "backends": {},
    }

    try:
        for b in target_backends:
            res = run_limb_validation_for_backend(b, server, port, out_dir=out_dir)
            combined_results["backends"][b] = res
            if not combined_results["render_profiles"] and "render_profiles" in res:
                combined_results["render_profiles"] = res["render_profiles"]
                combined_results["resolved_settings"] = res["resolved_settings"]

        summary_file = out_dir / "results.json"
        summary_file.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
        print(f"\nSuccessfully wrote consolidated G3 results to: {summary_file}")

    finally:
        server.shutdown()
        server.server_close()
        print("Stopped XVSServer.")

    return combined_results


def main() -> None:
    parser = argparse.ArgumentParser(description="WP-01 Gate G3 Earth-Limb Validation")
    parser.add_argument("--backend", choices=["ref", "hw", "both"], default="both", help="Renderer backend to evaluate")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_G3_DIR, help="Output directory for G3 artifacts")
    args = parser.parse_args()

    backends = ["ref", "hw"] if args.backend == "both" else [args.backend]
    res = run_g3_validation(backends=backends, out_dir=args.out_dir)

    # Enforce spec rule: ref must pass; hw is reported as-is
    if "ref" in res["backends"]:
        ref_pass = res["backends"]["ref"]["all_scenes_pass"]
        if not ref_pass:
            print("\n[ERROR] Gate G3 FAIL: Backend 'ref' failed one or more hard thresholds!")
            sys.exit(1)
        else:
            print("\n[SUCCESS] Gate G3 PASS: Backend 'ref' passed all hard thresholds!")

    if "hw" in res["backends"]:
        hw_pass = res["backends"]["hw"]["all_scenes_pass"]
        print(f"[REPORT] Backend 'hw' pass status: {hw_pass}")


if __name__ == "__main__":
    main()
