"""Sequence runner for OpenXVS camera rig (WP01 §2.7, §5).

Renders T1, T2, T3 scenarios with 6-camera rig, generates exact horizon GT,
and writes standard sequence artifacts.
"""

from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.rig.rig import Rig
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer
from xvs.sim.allowlist import URLClassifier
from xvs.sim import trajectory, airports
from xvs.sim.horizon import compute_horizon_gt

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_git_state() -> dict[str, Any]:
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip()
        dirty = len(status) > 0
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": False}


def _tile_counters(server: XVSServer) -> dict[str, int]:
    """Snapshot of the proxy's event counters (server.stats is updated per served tile)."""
    with server._stats_lock:
        st = server.stats["usgs_imagery"]
        return {"served": st["tiles_served_total"], "upstream": st["tiles_upstream_fetches"], "errors": st["tile_errors"]}


def run_sequence(
    scenario_path: str | Path,
    out_dir: str | Path,
    rate_hz: float | None = None,
    profile: str = "rgb",
    backend: str = "hw",
    offline: bool = False,
    png_compress: int = 6,
    max_frames: int | None = None,
) -> dict[str, Any]:
    scenario_path = Path(scenario_path)
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_config = json.load(f)

    with open(scenario_path, "rb") as f:
        scenario_sha256 = hashlib.sha256(f.read()).hexdigest()

    scenario_id = scenario_config["scenario_id"]
    duration_s = float(scenario_config.get("duration_s", 60.0))  # default-ok: optional fallback default
    rate = float(rate_hz or scenario_config.get("default_rate_hz", 2.0))  # default-ok: framerate default
    n_frames = int(round(duration_s * rate))
    if max_frames is not None and max_frames < n_frames:
        n_frames = max_frames

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Pre-flight: free disk check (spec §2.7)
    # Estimated bytes = frames * 6 * 1.6 MB * 1.3
    estimated_bytes = int(n_frames * 6 * 1.6 * 1024 * 1024 * 1.3)
    free_bytes = shutil.disk_usage(out_path).free
    if free_bytes < estimated_bytes:
        raise RuntimeError(
            f"Pre-flight disk check failed: estimated {estimated_bytes / 1e6:.1f} MB required, but only {free_bytes / 1e6:.1f} MB free"
        )

    # Initialize rig: rig_true per spec §2.1
    nom_rig = Rig.load_nominal()
    true_rig = nom_rig.make_true(20260922)

    # T2 runway reference if applicable
    reference_meta = None
    thr_data = None
    if scenario_config.get("trajectory_type") == "t2_ksfo28l":
        thr_data = airports.load_ksfo_28l_threshold()
        # The file actually read by load_ksfo_28l_threshold() (G4 finding: a wrong path here
        # silently produced an empty SHA-256). No fallback: a missing file or sidecar is an error.
        runways_csv_path = airports.DEFAULT_RUNWAYS_CSV
        with open(runways_csv_path, "rb") as f:
            csv_bytes = f.read()
        csv_sha = hashlib.sha256(csv_bytes).hexdigest()
        sidecar = json.loads(Path(str(runways_csv_path) + ".meta.json").read_text(encoding="utf-8"))
        if sidecar.get("sha256") != csv_sha:
            raise RuntimeError(f"runways.csv SHA-256 {csv_sha} does not match its download sidecar {sidecar.get('sha256')}")
        reference_meta = {
            "airport": "KSFO",
            "runway": "28L",
            "he_latitude_deg": thr_data.lat_deg,
            "he_longitude_deg": thr_data.lon_deg,
            "he_heading_degT": thr_data.heading_deg,
            "he_displaced_threshold_m": thr_data.displaced_threshold_m,
            "h_thr_m": thr_data.h_m,
            "source_csv_sha256": csv_sha,
            "source_csv_path": str(runways_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_csv_bytes": len(csv_bytes),
            "source_sidecar": sidecar,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # Start local server
    port = get_free_port()
    cesium_build = REPO_ROOT / "sim" / "cesium" / "node_modules" / "cesium" / "Build" / "Cesium"
    page_dir = REPO_ROOT / "sim" / "cesium" / "page"
    cache_dir = REPO_ROOT / "data" / "cache" / "usgs_imagery"
    log_dir = out_path

    server = XVSServer(
        server_address=("127.0.0.1", port),
        cesium_build_dir=cesium_build,
        page_dir=page_dir,
        cache_dir=cache_dir,
        log_dir=log_dir,
        offline=offline,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    run_start_wall = time.perf_counter()
    renderer = XVSRenderer(port=port, backend=backend, server=server).start()

    # Pre-create dedicated resident pages for each camera (spec §2.5)
    print(f"Initializing resident camera viewports on {backend} backend...")
    cam_pages = {}
    for cam_id in ("F", "R", "B", "L", "U", "D"):
        p = renderer.new_camera_page(cam_id, profile=profile)
        cam_pages[cam_id] = p
    resolved_profile = renderer.resolved_settings or {}

    # Warm up camera pages at frame 0 pose so initial tile streaming settles cleanly
    traj_type = scenario_config.get("trajectory_type")
    if traj_type == "t1_cruise":
        s0 = trajectory.t1_cruise_state(0.0)
    elif traj_type == "t2_ksfo28l":
        s0 = trajectory.t2_ksfo28l_state(0.0, thr=thr_data)
    elif traj_type == "t3_turn_climb":
        s0 = trajectory.t3_turn_climb_state(0.0)
    else:
        s0 = None

    if s0 is not None:
        c0 = true_rig.compute_poses(s0.lat_deg, s0.lon_deg, s0.h_m, s0.yaw_rad, s0.pitch_rad, s0.roll_rad)
        for cam_id, cp in cam_pages.items():
            renderer.set_camera(
                c0[cam_id]["position"],
                c0[cam_id]["direction"],
                c0[cam_id]["up"],
                c0[cam_id]["fov_rad"],
                c0[cam_id]["aspect_ratio"],
                page=cp,
            )
            for _ in range(5):
                cp.evaluate("window.xvsWidget.render()")
            time.sleep(0.05)

    # Write sequence.json header
    seq_header = {
        "schema": "xvs.sequence/1",
        "run_id": out_path.name,
        "scenario_id": scenario_id,
        "scenario_config": scenario_config,
        "scenario_config_sha256": scenario_sha256,
        "rate_hz": rate,
        "n_frames": n_frames,
        "start_utc": scenario_config.get("start_utc", "2026-06-21T19:00:00Z"),
        "terrain_mode": "ellipsoid",
        "render_profile": resolved_profile,
        "intrinsics": {
            "model": "pinhole_ideal",
            "width": true_rig.width,
            "height": true_rig.height,
            "fov_h_deg": true_rig.fov_h_deg,
            "aspect_ratio": true_rig.aspect_ratio,
        },
        "rig_nominal": [
            {
                "id": c.id,
                "name": c.name,
                "lever_arm_m": c.lever_arm_m.tolist(),
                "r_body_cam": c.r_body_cam.tolist(),
                "boresight_body": c.boresight_body.tolist(),
            }
            for c in nom_rig.cameras
        ],
        "rig_true": {
            "seed": true_rig.seed,
            "cameras": [
                {
                    "id": c.id,
                    "name": c.name,
                    "lever_arm_m": c.lever_arm_m.tolist(),
                    "r_body_cam": c.r_body_cam.tolist(),
                    "boresight_body": c.boresight_body.tolist(),
                    "omega_cam_rad": c.omega_cam_rad.tolist() if c.omega_cam_rad is not None else None,
                    "delta_lever_arm_m": c.delta_lever_arm_m.tolist() if c.delta_lever_arm_m is not None else None,
                }
                for c in true_rig.cameras
            ],
        },
        "rig": true_rig.record(),
        "reference": reference_meta,
    }
    with open(out_path / "sequence.json", "w", encoding="utf-8") as f:
        json.dump(seq_header, f, indent=2)

    # Resume check
    frames_jsonl_path = out_path / "frames.jsonl"
    existing_frames: dict[int, dict[str, Any]] = {}
    if frames_jsonl_path.exists():
        with open(frames_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_frames[item["k"]] = item

    frames_file = open(frames_jsonl_path, "a", encoding="utf-8")

    all_timings: dict[str, list[float]] = {
        "set_camera": [],
        "render_loop": [],
        "readback": [],
        "encode": [],
        "frame_total": [],
    }
    per_cam_timings: dict[str, dict[str, list[float]]] = {
        c: {"render_loop": [], "readback": []} for c in ("F", "R", "B", "L", "U", "D")
    }

    n_incomplete = 0
    total_renders = 0

    print(f"Rendering sequence '{scenario_id}' ({n_frames} frames @ {rate} Hz, profile '{profile}')...")
    t0_render_all = time.perf_counter()

    traj_type = scenario_config.get("trajectory_type")
    start_dt = datetime.strptime(scenario_config["start_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    try:
        for k in range(n_frames):
            t_s = float(k) / rate

            # Check resume: all 6 PNG files exist and SHA-256 matches
            if k in existing_frames:
                frame_rec = existing_frames[k]
                resume_valid = True
                for cam_id, cinfo in frame_rec["cams"].items():
                    png_p = out_path / cinfo["png"]
                    if not png_p.exists():
                        resume_valid = False
                        break
                    with open(png_p, "rb") as f:
                        if hashlib.sha256(f.read()).hexdigest() != cinfo["sha256"]:
                            resume_valid = False
                            break
                if resume_valid:
                    continue

            t_frame_start = time.perf_counter()

            # 1. Compute aircraft state
            if traj_type == "t1_cruise":
                state = trajectory.t1_cruise_state(t_s)
            elif traj_type == "t2_ksfo28l":
                state = trajectory.t2_ksfo28l_state(t_s, thr=thr_data)
            elif traj_type == "t3_turn_climb":
                state = trajectory.t3_turn_climb_state(t_s)
            else:
                raise ValueError(f"Unknown trajectory type '{traj_type}'")

            # 2. Compute camera poses
            cam_poses = true_rig.compute_poses(
                lat_deg=state.lat_deg,
                lon_deg=state.lon_deg,
                h_m=state.h_m,
                yaw_rad=state.yaw_rad,
                pitch_rad=state.pitch_rad,
                roll_rad=state.roll_rad,
            )

            # Compute UTC timestamp string
            # Base: start_utc + t_s
            # start_utc is 2026-06-21T19:00:00Z
            frame_dt = start_dt + timedelta(microseconds=int(round(t_s * 1e6)))
            frame_utc = frame_dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{frame_dt.microsecond // 1000:03d}Z"

            frame_cams_rec: dict[str, Any] = {}

            # Render each camera
            for cam_id in ("F", "R", "B", "L", "U", "D"):
                cam_pose = cam_poses[cam_id]
                cam_p = cam_pages[cam_id]

                readback, ms_set = renderer.set_camera(
                    position=cam_pose["position"],
                    direction=cam_pose["direction"],
                    up=cam_pose["up"],
                    fov_rad=cam_pose["fov_rad"],
                    aspect_ratio=cam_pose["aspect_ratio"],
                    page=cam_p,
                )
                renderer.set_clock(frame_utc, page=cam_p)

                # Stable-frame render; tile counters are deltas of the proxy's event counters
                # around this render (renders are sequential and end with 0 requests in flight).
                tiles_before = _tile_counters(server)
                res = renderer.render_frame(page=cam_p, ms_set_camera=ms_set, transfer_method="png")
                tiles_after = _tile_counters(server)
                total_renders += res.n_renders
                if res.incomplete:
                    n_incomplete += 1

                # Save PNG
                cam_dir = out_path / cam_id
                cam_dir.mkdir(parents=True, exist_ok=True)
                png_rel = f"{cam_id}/{k:06d}.png"
                png_full = out_path / png_rel

                file_size, ms_encode = renderer.save_png(res.image, png_full, compress_level=png_compress)
                with open(png_full, "rb") as f:
                    file_sha = hashlib.sha256(f.read()).hexdigest()

                # Horizon GT
                gt = compute_horizon_gt(
                    p_ecef_cam=cam_pose["position"],
                    r_ecef_cam=cam_pose["r_ecef_cam"],
                    cam=true_rig.camera_model,
                )

                # Record camera frame
                frame_cams_rec[cam_id] = {
                    "png": png_rel,
                    "sha256": file_sha,
                    "p_ecef_m": cam_pose["position"].tolist(),
                    "R_ecef_cam": cam_pose["r_ecef_cam"].tolist(),
                    "readback": readback,
                    "render": {
                        "n_renders": res.n_renders,
                        "incomplete": res.incomplete,
                        "ms_set": ms_set,
                        "ms_render": res.ms_render_loop,
                        "ms_readback": res.ms_readback,
                        "ms_encode": ms_encode,
                        "drawing_buffer": list(res.drawing_buffer),
                        "rgb_sha256": res.rgb_sha256,
                        "tiles_requested": tiles_after["served"] - tiles_before["served"],
                        "tiles_cache_miss": tiles_after["upstream"] - tiles_before["upstream"],
                        "tile_errors": tiles_after["errors"] - tiles_before["errors"],
                    },
                    "gt": gt,
                }

                all_timings["set_camera"].append(ms_set)
                all_timings["render_loop"].append(res.ms_render_loop)
                all_timings["readback"].append(res.ms_readback)
                all_timings["encode"].append(ms_encode)
                per_cam_timings[cam_id]["render_loop"].append(res.ms_render_loop)
                per_cam_timings[cam_id]["readback"].append(res.ms_readback)

            ms_frame_total = (time.perf_counter() - t_frame_start) * 1000.0
            all_timings["frame_total"].append(ms_frame_total)

            frame_entry = {
                "schema": "xvs.frame/1",
                "k": k,
                "t_s": t_s,
                "utc": frame_utc,
                "aircraft": {
                    "lat_deg": state.lat_deg,
                    "lon_deg": state.lon_deg,
                    "h_m": state.h_m,
                    "p_ecef_m": state.p_ecef_m.tolist(),
                    "v_ned_mps": state.v_ned_mps.tolist(),
                    "ypr_deg": list(state.ypr_deg),
                    "q_ned_body_wxyz": state.q_ned_body_wxyz.tolist(),
                    "R_ecef_body": state.r_ecef_body.tolist(),
                },
                "cams": frame_cams_rec,
            }
            frames_file.write(json.dumps(frame_entry) + "\n")
            frames_file.flush()

            if (k + 1) % max(1, n_frames // 10) == 0 or (k + 1) == n_frames:
                elapsed = time.perf_counter() - t0_render_all
                print(f"  Frame {k+1}/{n_frames} ({((k+1)/n_frames)*100:.0f}%) - {elapsed:.1f}s elapsed, {ms_frame_total:.1f}ms/frame")

    finally:
        frames_file.close()
        renderer.close()
        server.shutdown()

    wall_time_s = time.perf_counter() - run_start_wall

    # Violations are appended by the route guard / proxy during the run. Never truncate the log:
    # make sure it exists and count its lines (G4 finding: it used to be emptied at the end).
    violations_path = out_path / "data_violations.jsonl"
    violations_path.touch(exist_ok=True)
    with open(violations_path, "r", encoding="utf-8") as f:
        n_violations = sum(1 for line in f if line.strip())
    with server._stats_lock:
        usgs_stats = json.loads(json.dumps(server.stats["usgs_imagery"]))

    # Compute performance metrics
    perf = {
        "schema": "xvs.perf/1",
        "sequence_id": scenario_id,
        "backend": backend,
        "rate_hz": rate,
        "n_frames": n_frames,
        "wall_time_s": wall_time_s,
        "total_renders": total_renders,
        "incomplete_frames": n_incomplete,
        "violations": n_violations,
        "timings_ms": {
            k: {
                "median": float(np.median(v)) if v else 0.0,
                "p95": float(np.percentile(v, 95)) if v else 0.0,
                "mean": float(np.mean(v)) if v else 0.0,
            }
            for k, v in all_timings.items()
        },
        "per_camera": {
            cam_id: {
                metric: {
                    "median": float(np.median(vals)) if vals else 0.0,
                    "p95": float(np.percentile(vals, 95)) if vals else 0.0,
                }
                for metric, vals in metrics.items()
            }
            for cam_id, metrics in per_cam_timings.items()
        },
        "tiles": {
            "requested": usgs_stats["tiles_served_total"],
            "cache_hit": usgs_stats["tiles_cache_hits"],
            "cache_miss": usgs_stats["tiles_upstream_fetches"],
            "upstream_bytes": usgs_stats["bytes_upstream_fetched"],
            "tile_errors": usgs_stats["tile_errors"],
        },
    }
    with open(out_path / "perf.json", "w", encoding="utf-8") as f:
        json.dump(perf, f, indent=2)

    # Data sources log (CONVENTIONS §9, WP01 §2.3): built only from events of this run.
    sources = []
    if usgs_stats["tiles_served_total"] > 0 or usgs_stats["tile_errors"] > 0:
        sources.append(dict(usgs_stats))
    if reference_meta is not None:
        sc = reference_meta["source_sidecar"]
        sources.append({
            "id": sc.get("id"),
            "attribution": sc.get("attribution"),
            "license": sc.get("license"),
            "host": sc.get("host"),
            "path_prefix": sc.get("path_prefix"),
            "url": sc.get("url"),
            "object_count": 1,
            "bytes": reference_meta["source_csv_bytes"],
            "file_path": reference_meta["source_csv_path"],
            "file_sha256": reference_meta["source_csv_sha256"],
            "downloaded_or_verified_utc": sc.get("verified_utc"),
            "read_utc": reference_meta["read_utc"],
        })
    data_sources = {
        "schema": "xvs.data_sources/1",
        "run_id": out_path.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": sources,
    }
    with open(out_path / "data_sources.json", "w", encoding="utf-8") as f:
        json.dump(data_sources, f, indent=2)

    # Run manifest
    manifest = {
        "schema": "xvs.run_manifest/1",
        "run_id": out_path.name,
        "scenario_id": scenario_id,
        "backend": backend,
        "rate_hz": rate,
        "n_frames": n_frames,
        "git": get_git_state(),
        "wall_time_s": wall_time_s,
        "incomplete_frames": n_incomplete,
        "violations": n_violations,
        "tiles": perf["tiles"],
        "profile": profile,
    }
    with open(out_path / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSequence {scenario_id} completed successfully in {wall_time_s:.1f}s.")
    return perf


def main() -> int:
    parser = argparse.ArgumentParser(description="Render OpenXVS trajectory sequence")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON config")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--rate-hz", type=float, default=None, help="Frame rate in Hz")
    parser.add_argument("--profile", default="rgb", help="Render profile")
    parser.add_argument("--backend", default="hw", choices=["hw", "ref"], help="Renderer backend")
    parser.add_argument("--offline", action="store_true", help="Force offline mode")
    parser.add_argument("--png-compress", type=int, default=6, help="PNG compression level")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to render")
    args = parser.parse_args()

    run_sequence(
        scenario_path=args.scenario,
        out_dir=args.out,
        rate_hz=args.rate_hz,
        profile=args.profile,
        backend=args.backend,
        offline=args.offline,
        png_compress=args.png_compress,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
