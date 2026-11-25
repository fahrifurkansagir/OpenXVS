"""Offline runner for WP-08 Gate G0 Virtual Cockpit study and demo rendering.

Executes according to WP08_DISPLAY.md and GEMINI_G-B8.md Part B:
- Loads G4 sequences T1, T2, T3 (T2 first).
- Evaluates seam photometry MAD for ellipsoid, infinity, sphere_R.
- Evaluates parallax angular errors for infinity baseline across range bins.
- Evaluates motion-to-photon latency and compensation (0, 20, 50, 100 ms).
- Evaluates coverage fractions (0, 1, 2+ cameras).
- Measures CPU rendering performance (ms/panorama, ms/viewport).
- Renders demo videos:
  (a) ellipsoid panorama (2048x1024);
  (b) head-tracked viewport with conformal symbology (1280x720);
  (c) split-screen seam comparison (infinity vs ellipsoid, 1280x720).
- Saves first, mid, and last frame PNG stills with SHA-256 hashes.
- Outputs display_results.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, get_repo_root
from xvs.geo import wgs84
from xvs.rig.rig import Rig, CAMERA_ORDER
from xvs.rig.camera import PinholeCamera
from xvs.display.fusion import (
    load_display_config,
    get_eye_point_body,
    render_display_panorama,
    compute_world_points,
    compute_parallax_angle,
    generate_panorama_rays,
)
from xvs.display.viewport import (
    get_viewport_camera,
    compute_viewport_pose,
    compute_head_pose,
    render_viewport_frame,
)
from xvs.display.symbology import draw_symbology, draw_burn_in_label
from xvs.display.latency import (
    compute_sequence_latency_table,
)
from xvs.sim.trajectory import (
    t1_cruise_state,
    t2_ksfo28l_state,
    t3_turn_climb_state,
)

CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g0.json"
DEFAULT_OUT_DIR = get_data_root() / "xvs" / "wp08" / "g0"

SEQUENCE_DIRS = {
    "T1": "T1_cruise",
    "T2": "T2_ksfo28l",
    "T3": "T3_turn_climb",
}

SEQUENCE_TRAJECTORIES = {
    "T1": t1_cruise_state,
    "T2": t2_ksfo28l_state,
    "T3": t3_turn_climb_state,
}


def make_video_writer(video_path: Path, width: int, height: int, fps: float = 10.0) -> tuple[str, Any, list[Any]]:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-movflags", "+faststart",
            str(video_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ("ffmpeg", proc, [])
    else:
        return ("pillow", None, [])


def write_video_frame(writer_tuple: tuple[str, Any, list[Any]], frame_rgb: np.ndarray) -> None:
    w_type, proc, frame_list = writer_tuple
    if w_type == "ffmpeg":
        proc.stdin.write(frame_rgb.tobytes())
    else:
        frame_list.append(Image.fromarray(frame_rgb))


def close_video_writer(writer_tuple: tuple[str, Any, list[Any]], video_path: Path, fps: float = 10.0) -> str:
    w_type, proc, frame_list = writer_tuple
    if w_type == "ffmpeg":
        proc.stdin.close()
        proc.wait()
        return "ffmpeg_h264"
    else:
        if frame_list:
            webp_path = video_path.with_suffix(".webp")
            frame_list[0].save(
                webp_path,
                save_all=True,
                append_images=frame_list[1:],
                duration=int(1000.0 / fps),
                loop=0,
            )
        return "pillow_webp"


def evaluate_parallax_ground_ranges(
    frames: list[dict[str, Any]],
    rig: Rig,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute parallax angular error of infinity baseline against ellipsoid ground points across range bins."""
    range_bins = [
        ("0_0.5km", 0.0, 500.0),
        ("0.5_2km", 500.0, 2000.0),
        ("2_10km", 2000.0, 10000.0),
        ("10km_plus", 10000.0, float("inf")),
    ]
    angles_by_bin: dict[str, list[float]] = {b[0]: [] for b in range_bins}
    all_angles: list[float] = []

    mid_idx = len(frames) // 2
    f = frames[mid_idx]
    ac = f["aircraft_pose"]
    lat, lon, h = ac["lat_deg"], ac["lon_deg"], ac["h_m"]
    yaw, pitch, roll = ac["yaw_rad"], ac["pitch_rad"], ac["roll_rad"]

    cam_poses = rig.compute_poses(lat, lon, h, yaw, pitch, roll)
    p_ecef_body = wgs84.geodetic_to_ecef(lat, lon, h)
    r_ecef_ned_mat = wgs84.r_ecef_ned(lat, lon)
    from xvs.geo import rotations
    r_ecef_body_mat = r_ecef_ned_mat @ rotations.euler_to_r(yaw, pitch, roll)

    eye_body = get_eye_point_body(config)
    eye_ecef = p_ecef_body + r_ecef_body_mat @ eye_body

    # Sample rays on mid-frame (equirectangular grid in NED)
    rays_ned = generate_panorama_rays(512, 256, frame="ned")
    rays_ecef = rays_ned.reshape(-1, 3) @ r_ecef_ned_mat.T

    # Intersect with ellipsoid
    pts_ell, _, hit_mask = compute_world_points(eye_ecef, rays_ecef, depth_model="ellipsoid")
    ground_pts = pts_ell[hit_mask]

    cam_model = rig.camera_model
    if len(ground_pts) > 0:
        dists = np.linalg.norm(ground_pts - eye_ecef, axis=-1)
        for cid, c_info in cam_poses.items():
            c_pos = c_info["p_ecef_cam"]
            r_cam = c_info["r_ecef_cam"]
            # Project ground_pts into camera frame: p_cam = R^T @ (P - C)
            p_cam = (ground_pts - c_pos) @ r_cam
            front = p_cam[:, 2] > 0.1
            if not np.any(front):
                continue
            z_front = p_cam[front, 2]
            u = cam_model.fx * (p_cam[front, 0] / z_front) + cam_model.cx
            v = cam_model.fy * (p_cam[front, 1] / z_front) + cam_model.cy
            in_view = (u >= 0.0) & (u < cam_model.width) & (v >= 0.0) & (v < cam_model.height)
            if not np.any(in_view):
                continue

            vis_indices = np.where(front)[0][in_view]
            vis_pts = ground_pts[vis_indices]
            vis_dists = dists[vis_indices]

            err_rad = compute_parallax_angle(vis_pts, eye_ecef, c_pos)
            err_deg = np.degrees(err_rad)

            for b_name, d_min, d_max in range_bins:
                mask = (vis_dists >= d_min) & (vis_dists < d_max)
                if np.any(mask):
                    angles_by_bin[b_name].extend(err_deg[mask].tolist())
            all_angles.extend(err_deg.tolist())

    pixel_thresh_deg = 90.0 / 1280.0  # 0.0703 deg

    bin_results = {}
    for b_name, _, _ in range_bins:
        arr = np.array(angles_by_bin[b_name])
        if len(arr) > 0:
            bin_results[b_name] = {
                "count": len(arr),
                "median_deg": float(np.median(arr)),
                "p95_deg": float(np.percentile(arr, 95)),
                "max_deg": float(np.max(arr)),
                "frac_above_1px": float(np.mean(arr > pixel_thresh_deg)),
            }
        else:
            bin_results[b_name] = {"count": 0, "median_deg": 0.0, "p95_deg": 0.0, "max_deg": 0.0, "frac_above_1px": 0.0}

    all_arr = np.array(all_angles)
    total_summary = {
        "count": len(all_arr),
        "median_deg": float(np.median(all_arr)) if len(all_arr) > 0 else 0.0,
        "p95_deg": float(np.percentile(all_arr, 95)) if len(all_arr) > 0 else 0.0,
        "max_deg": float(np.max(all_arr)) if len(all_arr) > 0 else 0.0,
        "frac_above_1px": float(np.mean(all_arr > pixel_thresh_deg)) if len(all_arr) > 0 else 0.0,
    }

    return {"bins": bin_results, "total": total_summary}


def load_sequence_frames(seq_name: str, data_root: Path, max_frames: int | None = None) -> list[dict[str, Any]]:
    seq_dir_name = SEQUENCE_DIRS[seq_name]
    seq_path = data_root / "xvs" / "wp01" / "g4" / seq_dir_name
    frames_file = seq_path / "frames.jsonl"
    if not frames_file.is_file():
        raise FileNotFoundError(f"Missing frames.jsonl at {frames_file}")

    frames = []
    with open(frames_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_frames is not None and idx >= max_frames:
                break
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            ac = rec["aircraft"]
            rec["aircraft_pose"] = {
                "lat_deg": ac["lat_deg"],
                "lon_deg": ac["lon_deg"],
                "h_m": ac["h_m"],
                "yaw_rad": math.radians(ac["ypr_deg"][0]),
                "pitch_rad": math.radians(ac["ypr_deg"][1]),
                "roll_rad": math.radians(ac["ypr_deg"][2]),
            }
            rec["v_ned_mps"] = np.array(ac["v_ned_mps"], dtype=np.float64)
            # Store image paths
            rec["cam_png_paths"] = {
                cam_id: seq_path / cam_info["png"]
                for cam_id, cam_info in rec["cams"].items()
            }
            frames.append(rec)
    return frames


def load_frame_cam_images(fr: dict[str, Any]) -> dict[str, np.ndarray]:
    cams = {}
    for cam_id, p in fr["cam_png_paths"].items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing camera image: {p}")
        with Image.open(p) as im:
            cams[cam_id] = np.array(im.convert("RGB"), dtype=np.uint8)
    return cams


def run_sequence_processing(
    seq_name: str,
    data_root: Path,
    out_dir: Path,
    rig: Rig,
    config: dict[str, Any],
    skip_videos: bool = False,
    viewport_only: bool = False,
) -> dict[str, Any]:
    print(f"\n==================================================")
    print(f"   Processing Sequence {seq_name} ({SEQUENCE_DIRS[seq_name]})   ")
    print(f"==================================================")
    frames = load_sequence_frames(seq_name, data_root)
    n_frames = len(frames)
    print(f"[{seq_name}] Loaded metadata for {n_frames} frames.")

    mid_idx = n_frames // 2
    mid_frame = frames[mid_idx]
    mid_cams = load_frame_cam_images(mid_frame)
    mid_pose = mid_frame["aircraft_pose"]

    existing_seq = None
    results_file = out_dir / "display_results.json"
    if results_file.is_file():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                d_res = json.load(f)
                existing_seq = d_res.get("sequences", {}).get(seq_name)
        except Exception:
            pass

    stills_hashes: dict[str, str] = {}
    if viewport_only and existing_seq is not None:
        print(f"[{seq_name}] viewport_only: reusing existing seam, coverage, parallax, latency metrics...")
        seam_metrics = existing_seq.get("seam_photometry", {})
        seam_ratio = existing_seq.get("seam_ratio_ellipsoid_vs_infinity", 0.0)  # default-ok: optional fallback default
        coverage = existing_seq.get("coverage", {})
        parallax_res = existing_seq.get("parallax", {})
        lat_table = existing_seq.get("latency", {})
        t_pan_ms = existing_seq.get("timing_ms", {}).get("panorama_2048x1024", 0.0)  # default-ok: timing benchmark fallback
        t_vp_ms = existing_seq.get("timing_ms", {}).get("viewport_1280x720", 0.0)  # default-ok: timing benchmark fallback
        stills_hashes = existing_seq.get("stills_sha256", {}).copy()
    else:
        # 1. Seam Photometry Evaluation across depth models over 12 evenly spaced frames (R-GB-15)
        print(f"[{seq_name}] Evaluating seam photometry MAD across depth models over 12 frames...")
        depth_models = config.get("depth_models", ["ellipsoid", "infinity", "sphere_R"])
        frame_indices_12 = np.linspace(0, n_frames - 1, 12).astype(int)
        seam_12_frames = []
        for f_idx in frame_indices_12:
            fr = frames[f_idx]
            t_s = float(fr["t_s"])
            k_val = int(fr["k"])
            pose_fr = fr["aircraft_pose"]
            cams_fr = load_frame_cam_images(fr)

            f_metrics = {}
            for dm in depth_models:
                _, stats = render_display_panorama(
                    cams_fr, rig, pose_fr, depth_model=dm, width=1024, height=512, config=config
                )
                f_metrics[dm] = stats["seam_photometry"]

            g_ell = f_metrics["ellipsoid"]["ground_overlap_mad"]
            g_inf = f_metrics["infinity"]["ground_overlap_mad"]
            ratio = float(g_ell / max(g_inf, 1e-9))
            viol = bool(g_ell > g_inf)
            seam_12_frames.append({
                "frame_idx": int(f_idx),
                "k": k_val,
                "t_s": t_s,
                "ground_mad_ellipsoid": float(g_ell),
                "ground_mad_infinity": float(g_inf),
                "ratio": ratio,
                "violation": viol,
                "metrics": f_metrics,
            })

        ratios_12 = [r["ratio"] for r in seam_12_frames]
        violations_12 = [r for r in seam_12_frames if r["violation"]]
        print(f"  [{seq_name}] 12-frame Seam Ratios: median={np.median(ratios_12):.4f}, min={min(ratios_12):.4f}, max={max(ratios_12):.4f}, violations={len(violations_12)}/12")
        for v in violations_12:
            print(f"    Exception reported: t={v['t_s']:.1f}s ratio={v['ratio']:.4f} (ellipsoid {v['ground_mad_ellipsoid']:.4f} > inf {v['ground_mad_infinity']:.4f})")

        # Pick mid-frame for baseline summary compatibility
        mid_row = [r for r in seam_12_frames if r["frame_idx"] == mid_idx]
        if not mid_row:
            mid_row = [seam_12_frames[len(seam_12_frames) // 2]]
        seam_metrics = mid_row[0]["metrics"]
        seam_ratio = mid_row[0]["ratio"]

        # 2. Coverage statistics
        _, mid_stats = render_display_panorama(
            mid_cams, rig, mid_pose, depth_model="ellipsoid", width=1024, height=512, config=config
        )
        coverage = mid_stats["coverage"]
        print(f"  Coverage: multi={coverage['overlap_fraction']:.3f}, single={coverage['single_camera_fraction']:.3f}, unseen={coverage['unseen_fraction']:.3f}")

        # 3. Parallax error table for infinity baseline
        print(f"[{seq_name}] Evaluating infinity baseline parallax error table...")
        parallax_res = evaluate_parallax_ground_ranges(frames, rig, config)
        for b_name, b_val in parallax_res["bins"].items():
            print(f"  [{b_name:10s}] med: {b_val['median_deg']:.3f} deg, p95: {b_val['p95_deg']:.3f} deg, frac > 1px: {b_val['frac_above_1px']*100.0:.1f}%")

        # 4. Latency error table
        print(f"[{seq_name}] Evaluating motion-to-photon latency (0, 20, 50, 100 ms)...")
        traj_fn = SEQUENCE_TRAJECTORIES[seq_name]
        t_eval = mid_frame["t_s"]
        t_eval_samples = [fr["t_s"] for fr in frames[::10]]
        lat_table = compute_sequence_latency_table(
            trajectory_func=traj_fn,
            t_eval_list=t_eval_samples,
            latencies_ms=config["latencies_ms"],
            is_t2=(seq_name == "T2"),
            config=config,
        )

        # 5. Timing benchmarks
        print(f"[{seq_name}] Benchmarking rendering speeds...")
        t0 = time.perf_counter()
        _ = render_display_panorama(mid_cams, rig, mid_pose, depth_model="ellipsoid", width=2048, height=1024, config=config)
        t_pan_ms = (time.perf_counter() - t0) * 1000.0

        head_euler = compute_head_pose(mid_frame["t_s"], config)
        t0 = time.perf_counter()
        _ = render_viewport_frame(mid_cams, rig, mid_pose, head_euler, depth_model="ellipsoid", config=config)
        t_vp_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  Rendering time: Panorama 2048x1024: {t_pan_ms:.1f} ms | Viewport 1280x720: {t_vp_ms:.1f} ms")

    # 6. Video Rendering and Stills
    encoder_used = "none"

    seq_dt = frames[1]["t_s"] - frames[0]["t_s"]
    seq_fps = float(round(1.0 / seq_dt, 2))

    if not skip_videos:
        videos_dir = out_dir / "videos"
        stills_dir = out_dir / "stills"
        videos_dir.mkdir(parents=True, exist_ok=True)
        stills_dir.mkdir(parents=True, exist_ok=True)

        vid_b_path = videos_dir / f"{seq_name}_viewport_symbology.mp4"
        w_b = make_video_writer(vid_b_path, 1280, 720, fps=seq_fps)

        if not viewport_only:
            vid_a_path = videos_dir / f"{seq_name}_panorama_ellipsoid.mp4"
            vid_c_path = videos_dir / f"{seq_name}_seam_split.mp4"
            w_a = make_video_writer(vid_a_path, 2048, 1024, fps=seq_fps)
            w_c = make_video_writer(vid_c_path, 1280, 720, fps=seq_fps)
        else:
            w_a = None
            w_c = None

        # Seam viewing angle: looking across F-R camera seam (yaw=45 deg, pitch=-10 deg down at ground)
        seam_euler = (math.radians(45.0), math.radians(-10.0), 0.0)
        vp_cam = get_viewport_camera(config)

        still_indices = {
            "first": 0,
            "mid": n_frames // 2,
            "last": n_frames - 1,
        }

        mode_desc = "(b) viewport" if viewport_only else "(a), (b), (c)"
        print(f"[{seq_name}] Rendering demo videos {mode_desc} across {n_frames} frames...")
        t_vid_start = time.perf_counter()

        for idx, fr in enumerate(frames):
            t_s = fr["t_s"]
            ac_pose = fr["aircraft_pose"]
            cams = load_frame_cam_images(fr)
            head_euler = compute_head_pose(t_s, config)

            # (a) Panorama ellipsoid (if not viewport_only)
            pan_labeled = None
            if w_a is not None:
                pan_rgb, _ = render_display_panorama(
                    cams, rig, ac_pose, depth_model="ellipsoid", width=2048, height=1024, config=config
                )
                label_a = f"{seq_name} t={t_s:05.2f}s ellipsoid SIMULATION"
                pan_labeled = draw_burn_in_label(pan_rgb, label_a, font_scale=0.6)
                write_video_frame(w_a, pan_labeled)

            # (b) Viewport with conformal symbology
            vp_rgb = render_viewport_frame(
                cams, rig, ac_pose, head_euler, depth_model="ellipsoid", config=config
            )
            vp_pose = compute_viewport_pose(ac_pose, head_euler, config)
            label_b = f"{seq_name} t={t_s:05.2f}s ellipsoid SIMULATION"
            vp_sym = draw_symbology(
                vp_rgb, ac_pose, fr["v_ned_mps"],
                vp_pose["p_ecef_view"], vp_pose["r_ecef_view"], vp_cam,
                is_t2=(seq_name == "T2"),
                burn_in_text=label_b,
            )
            write_video_frame(w_b, vp_sym)

            # (c) Seam Split: left infinity, right ellipsoid (if not viewport_only)
            split_labeled = None
            if w_c is not None:
                seam_inf_rgb = render_viewport_frame(
                    cams, rig, ac_pose, seam_euler, depth_model="infinity", config=config
                )
                seam_ell_rgb = render_viewport_frame(
                    cams, rig, ac_pose, seam_euler, depth_model="ellipsoid", config=config
                )
                split_rgb = np.zeros_like(seam_inf_rgb)
                split_rgb[:, :640] = seam_inf_rgb[:, :640]
                split_rgb[:, 640:] = seam_ell_rgb[:, 640:]
                split_rgb[:, 639:641] = [255, 255, 255]  # White vertical seam line

                label_c = f"{seq_name} t={t_s:05.2f}s split-seam SIMULATION"
                split_labeled = draw_burn_in_label(split_rgb, label_c, font_scale=0.5)
                write_video_frame(w_c, split_labeled)

            # Save stills at first, mid, last
            for s_tag, s_idx in still_indices.items():
                if idx == s_idx:
                    still_items = [(f"{seq_name}_viewport_{s_tag}", vp_sym)]
                    if pan_labeled is not None:
                        still_items.append((f"{seq_name}_panorama_{s_tag}", pan_labeled))
                    if split_labeled is not None:
                        still_items.append((f"{seq_name}_seam_split_{s_tag}", split_labeled))

                    for name_prefix, frame_data in still_items:
                        still_path = stills_dir / f"{name_prefix}.png"
                        Image.fromarray(frame_data).save(still_path)
                        h_val = hashlib.sha256(still_path.read_bytes()).hexdigest()
                        stills_hashes[name_prefix] = h_val

            if (idx + 1) % 50 == 0 or (idx + 1) == n_frames:
                el = time.perf_counter() - t_vid_start
                print(f"  Frame {idx+1}/{n_frames} ({((idx+1)/n_frames)*100.0:.1f}%) in {el:.1f} s...")

        enc_b = close_video_writer(w_b, vid_b_path)
        encoder_used = enc_b
        if w_a is not None:
            close_video_writer(w_a, vid_a_path)
        if w_c is not None:
            close_video_writer(w_c, vid_c_path)
        print(f"[{seq_name}] Finished demo videos. Encoder: {encoder_used}.")

    return {
        "sequence": seq_name,
        "n_frames": n_frames,
        "seam_photometry": seam_metrics,
        "seam_ratio_ellipsoid_vs_infinity": seam_ratio,
        "coverage": coverage,
        "parallax": parallax_res,
        "latency": lat_table,
        "video_fps": seq_fps,
        "seam_12_frames": {
            "frames": seam_12_frames,
            "median_ratio": float(np.median(ratios_12)),
            "min_ratio": float(min(ratios_12)),
            "max_ratio": float(max(ratios_12)),
            "n_violations": len(violations_12),
            "violations": [{"k": v["k"], "t_s": v["t_s"], "ratio": v["ratio"]} for v in violations_12],
        },
        "timing_ms": {
            "panorama_2048x1024": t_pan_ms,
            "viewport_1280x720": t_vp_ms,
        },
        "stills_sha256": stills_hashes,
        "video_encoder": encoder_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WP-08 G0 Virtual Cockpit Runner")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--data-root", type=Path, default=None, help="Explicit data root")
    parser.add_argument("--sequences", nargs="+", default=["T2", "T1", "T3"], help="Sequences to run (T2 first)")
    parser.add_argument("--skip-videos", action="store_true", help="Skip heavy video rendering (metrics only)")
    parser.add_argument("--viewport-only", action="store_true", help="Render only viewport video and stills (reusing existing metrics)")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_display_config(CONFIG_PATH)
    rig = Rig.load_nominal().make_true(seed=20260922)

    print("==================================================")
    print("   OpenXVS WP-08 Gate G0 Virtual Cockpit Runner   ")
    print("==================================================")
    print(f"Data root: {data_root}")
    print(f"Output directory: {out_dir}")
    print(f"Sequences: {args.sequences}")

    results_file = out_dir / "display_results.json"
    existing_sequences: dict[str, Any] = {}
    if results_file.is_file():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                d_old = json.load(f)
                existing_sequences = d_old.get("sequences", {})
        except Exception:
            pass

    results: dict[str, Any] = {
        "gate": "G0",
        "work_package": "WP-08",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sequences": existing_sequences,
        "open_questions": [
            "Non-square viewport pixels (fx=640.0, fy=623.54 to fit 90x60 deg into 1280x720, aspect ratio 1.026)"
        ],
        "deviations": [
            "Videos rendered at sequence framerate (2 fps for T1/T3, 5 fps for T2) instead of fixed 10 fps"
        ],
    }

    t_total_start = time.perf_counter()

    for seq in args.sequences:
        seq_res = run_sequence_processing(
            seq,
            data_root,
            out_dir,
            rig,
            config,
            skip_videos=args.skip_videos,
            viewport_only=args.viewport_only,
        )
        results["sequences"][seq] = seq_res

    t_total_end = time.perf_counter()
    results["total_elapsed_s"] = t_total_end - t_total_start

    results_file = out_dir / "display_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n==================================================")
    print(f"WP-08 G0 processing finished in {results['total_elapsed_s']:.2f} s.")
    print(f"Results written to: {results_file}")
    print("==================================================")


if __name__ == "__main__":
    main()
