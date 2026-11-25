"""Determinism validation for OpenXVS Gate G4 (WP01 §3 G4, §4).

Re-renders 10 seeded-random T2 frames in a fresh process with --offline and compares:
- ref: identical rgb_sha256, and identical PNG file SHA-256 (same machine), between TWO FRESH
  PROCESSES: run once without --ref-baseline (writes det_ref_*.png and determinism_ref_run.json),
  then run again with --ref-baseline <dir of the first run>. (G4 finding: the previous version
  re-rendered the already-stable page in the same process, which cannot detect process-level
  non-determinism.)
- hw: per image max_diff <= 1 and n_diff <= 1% (10486 px) against the original T2 sequence PNGs.
"""

from __future__ import annotations
import argparse
import base64
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.rig.rig import Rig
from xvs.sim.server import XVSServer
from xvs.sim.renderer import XVSRenderer
from xvs.sim import trajectory, airports

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_determinism_check(
    t2_seq_dir: str | Path,
    out_dir: str | Path,
    backend: str = "ref",
    seed: int = 20260923,
    n_sample_frames: int = 10,
    ref_baseline: str | Path | None = None,
) -> dict[str, Any]:
    t2_path = Path(t2_seq_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(t2_path / "sequence.json", "r", encoding="utf-8") as f:
        seq_meta = json.load(f)

    with open(t2_path / "frames.jsonl", "r", encoding="utf-8") as f:
        all_frames = [json.loads(line) for line in f if line.strip()]

    rate_hz = float(seq_meta.get("rate_hz", 5.0))  # default-ok: framerate default
    total_frames = len(all_frames)

    rng = np.random.default_rng(seed)
    sampled_indices = sorted(rng.choice(total_frames, size=n_sample_frames, replace=False).tolist())
    print(f"Sampled frames for {backend} determinism: {sampled_indices}")

    thr_data = airports.load_ksfo_28l_threshold()
    true_rig = Rig.load_nominal().make_true(20260922)

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

    # Pre-create pages
    cam_pages = {}
    for cam_id in ("F", "R", "B", "L", "U", "D"):
        cam_pages[cam_id] = renderer.new_camera_page(cam_id, profile="rgb")

    baseline: dict[str, dict[str, str]] = {}
    if backend == "ref" and ref_baseline is not None:
        bl = json.loads((Path(ref_baseline) / "determinism_ref_run.json").read_text(encoding="utf-8"))
        baseline = {r["image"]: r for r in bl["images"]}
    run_images: list[dict[str, Any]] = []

    results = []
    rgb_sha_matches = 0
    png_sha_matches = 0
    hw_passes = 0
    worst_max_diff = 0
    worst_diff_frac = 0.0

    try:
        for k in sampled_indices:
            t_s = float(k) / rate_hz
            state = trajectory.t2_ksfo28l_state(t_s, thr=thr_data)
            cam_poses = true_rig.compute_poses(
                lat_deg=state.lat_deg,
                lon_deg=state.lon_deg,
                h_m=state.h_m,
                yaw_rad=state.yaw_rad,
                pitch_rad=state.pitch_rad,
                roll_rad=state.roll_rad,
            )
            sec_int = int(t_s)
            ms_frac = int(round((t_s - sec_int) * 1000))
            frame_utc = f"2026-06-21T19:00:{sec_int:02d}.{ms_frac:03d}Z" if sec_int < 60 else f"2026-06-21T19:01:{sec_int-60:02d}.{ms_frac:03d}Z"

            frame_orig = all_frames[k]

            frame_pass = True
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
                render_res = renderer.render_frame(page=cam_p, ms_set_camera=ms_set, transfer_method="png")

                orig_cdata = frame_orig["cams"][cam_id]
                orig_png_path = t2_path / orig_cdata["png"]
                orig_png_sha = orig_cdata["sha256"]
                orig_rgb_sha = orig_cdata["render"]["rgb_sha256"]

                # ref: compared with a previous fresh-process run (--ref-baseline); hw: with the sequence PNG
                new_png_path = out_path / f"det_{backend}_{cam_id}_{k:06d}.png"
                renderer.save_png(render_res.image, new_png_path, compress_level=6)
                with open(new_png_path, "rb") as f:
                    new_png_sha = hashlib.sha256(f.read()).hexdigest()

                new_rgb_sha = render_res.rgb_sha256

                if backend == "ref":
                    key = f"det_ref_{cam_id}_{k:06d}.png"
                    run_images.append({"image": key, "rgb_sha256": new_rgb_sha, "png_sha256": new_png_sha,
                                       "n_renders": render_res.n_renders, "incomplete": render_res.incomplete})
                    if baseline:
                        b = baseline.get(key)
                        is_rgb_match = b is not None and b["rgb_sha256"] == new_rgb_sha
                        is_png_match = b is not None and b["png_sha256"] == new_png_sha
                        if is_rgb_match:
                            rgb_sha_matches += 1
                        if is_png_match:
                            png_sha_matches += 1
                        if not (is_rgb_match and is_png_match):
                            frame_pass = False
                            print(f"  [MISMATCH ref] k={k} cam={cam_id}: rgb_match={is_rgb_match}, png_match={is_png_match}")
                else:  # hw
                    orig_img = np.array(Image.open(orig_png_path).convert("RGB"))
                    diff = np.abs(render_res.image.astype(int) - orig_img.astype(int))
                    max_d = int(np.max(diff))
                    n_d = int(np.count_nonzero(diff.any(axis=-1)))
                    frac_d = (n_d / (1024 * 1024)) * 100.0

                    if max_d > worst_max_diff:
                        worst_max_diff = max_d
                    if frac_d > worst_diff_frac:
                        worst_diff_frac = frac_d

                    # hw threshold: max_diff <= 1 and n_diff <= 1% (10486 px)
                    cam_pass = (max_d <= 1 and n_d <= 10486)
                    if not cam_pass:
                        frame_pass = False
                        print(f"  [MISMATCH hw] k={k} cam={cam_id}: max_diff={max_d}, n_diff={n_d} ({frac_d:.3f}%)")

            if frame_pass:
                hw_passes += 1

    finally:
        renderer.close()
        server.shutdown()

    if backend == "ref":
        with open(out_path / "determinism_ref_run.json", "w", encoding="utf-8") as f:
            json.dump({"backend": "ref", "sampled_k": sampled_indices, "images": run_images}, f, indent=2)

    summary = {
        "backend": backend,
        "mode": ("ref_two_fresh_processes" if (backend == "ref" and baseline) else
                 ("ref_first_run_only" if backend == "ref" else "hw_vs_original_sequence")),
        "ref_baseline": str(ref_baseline) if ref_baseline is not None else None,
        "n_frames": n_sample_frames,
        "total_images": n_sample_frames * 6,
        "sampled_k": sampled_indices,
        "rgb_sha256_matches": rgb_sha_matches,
        "png_sha256_matches": png_sha_matches,
        "passing_frames": hw_passes,
        "worst_max_diff": worst_max_diff,
        "worst_diff_fraction_pct": worst_diff_frac,
        "all_pass": (hw_passes == n_sample_frames) if backend == "hw" else (bool(baseline) and rgb_sha_matches == n_sample_frames * 6 and png_sha_matches == n_sample_frames * 6),
    }

    out_file = out_path / f"determinism_{backend}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved determinism summary to {out_file}: all_pass={summary['all_pass']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Check determinism for G4 sequences")
    parser.add_argument("--t2-dir", required=True, help="Directory of rendered T2 sequence")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--backend", default="ref", choices=["ref", "hw"], help="Backend to test")
    parser.add_argument("--seed", type=int, default=20260923, help="Random seed")
    parser.add_argument("-n", type=int, default=10, help="Number of frames to sample")
    parser.add_argument("--ref-baseline", default=None, help="ref only: out-dir of a previous (fresh-process) ref run to compare with")
    args = parser.parse_args()

    res = run_determinism_check(
        t2_seq_dir=args.t2_dir,
        out_dir=args.out_dir,
        backend=args.backend,
        seed=args.seed,
        n_sample_frames=args.n,
        ref_baseline=args.ref_baseline,
    )
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
