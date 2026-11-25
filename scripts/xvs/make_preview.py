"""Preview generator for OpenXVS sequences (WP01 §2.8).

Generates:
1. Equirectangular (2048x1024) in BODY frame (equirect_body)
2. Equirectangular (2048x1024) in NED world-stabilised frame (equirect_ned)
3. Cube cross layout (cube_cross)
4. GT horizon overlays on selected frames
5. MP4 videos + first/middle/last PNG captures
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.rig.rig import Rig
from xvs.geo import rotations

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def precompute_equirect_rays(width: int = 2048, height: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit direction rays in reference frame for each pixel (col, row).

    az = (u+0.5)/W * 2pi - pi, el = pi/2 - (v+0.5)/H * pi.
    Returns: (height, width, 3) float64 rays array.
    """
    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(cols, rows)

    az = (u_grid + 0.5) / width * (2.0 * math.pi) - math.pi
    el = (math.pi / 2.0) - (v_grid + 0.5) / height * math.pi

    cos_el = np.cos(el)
    sin_el = np.sin(el)
    cos_az = np.cos(az)
    sin_az = np.sin(az)

    # d = [cos(el)*cos(az), cos(el)*sin(az), -sin(el)] (z down)
    dx = cos_el * cos_az
    dy = cos_el * sin_az
    dz = -sin_el

    rays = np.stack([dx, dy, dz], axis=-1)  # (H, W, 3)
    return rays, el


def build_equirect_frame(
    cam_images: dict[str, np.ndarray],
    rig: Rig,
    rays_ref: np.ndarray,
    r_ref_body: np.ndarray | None = None,
    draw_el0: bool = True,
    el_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Project 6 camera images onto an equirectangular frame.

    rays_ref: (H, W, 3) rays in reference frame (BODY or NED).
    r_ref_body: (3, 3) rotation matrix from BODY to reference (e.g. R_ned_body).
      If None, reference frame is BODY.
    """
    h_out, w_out, _ = rays_ref.shape

    if r_ref_body is not None:
        # d_body = R_ref_body.T @ d_ref
        rays_flat = rays_ref.reshape(-1, 3)  # (N, 3)
        rays_body = (r_ref_body.T @ rays_flat.T).T  # (N, 3)
    else:
        rays_body = rays_ref.reshape(-1, 3)

    out_pixels = np.zeros((h_out * w_out, 3), dtype=np.uint8)

    # Map each camera
    # For each ray, find camera with maximum boresight_body dot ray_body
    cam_dots = []
    cam_list = list(rig.cameras)
    for c in cam_list:
        b = c.boresight_body
        dots = rays_body @ b  # (N,)
        cam_dots.append(dots)
    cam_dots_arr = np.stack(cam_dots, axis=0)  # (6, N)
    best_cam_idx = np.argmax(cam_dots_arr, axis=0)  # (N,)

    best_cam_grid = best_cam_idx.reshape(h_out, w_out)
    rays_body_grid = rays_body.reshape(h_out, w_out, 3)
    out_img = np.zeros((h_out, w_out, 3), dtype=np.uint8)

    fx = rig.camera_model.fx
    fy = rig.camera_model.fy
    cx = rig.camera_model.cx
    cy = rig.camera_model.cy

    for idx, c in enumerate(cam_list):
        mask = (best_cam_grid == idx)
        if not np.any(mask):
            continue

        # ray in cam frame: R_body_cam.T @ ray_body
        rays_cam = rays_body_grid @ c.r_body_cam  # (H, W, 3)
        z_cam = rays_cam[:, :, 2]
        valid = (z_cam > 1e-4) & mask

        u_px = np.zeros((h_out, w_out), dtype=np.float32)
        v_px = np.zeros((h_out, w_out), dtype=np.float32)

        u_px[valid] = fx * rays_cam[valid, 0] / z_cam[valid] + cx
        v_px[valid] = fy * rays_cam[valid, 1] / z_cam[valid] + cy

        img = cam_images[c.id]
        h_in, w_in, _ = img.shape
        u_clamped = np.clip(u_px, 0.0, w_in - 1.0).astype(np.float32)
        v_clamped = np.clip(v_px, 0.0, h_in - 1.0).astype(np.float32)

        sampled = cv2.remap(
            img,
            u_clamped,
            v_clamped,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out_img[valid] = sampled[valid]

    if draw_el0 and el_grid is not None:
        # Draw thin horizon line where el = 0 (row where v = H/2 - 0.5)
        zero_row = int(round(h_out / 2.0 - 0.5))
        if 0 <= zero_row < h_out:
            out_img[zero_row, :, :] = [0, 255, 255]  # Cyan line

    return out_img


def build_cube_cross_frame(
    cam_images: dict[str, np.ndarray],
    scale: float = 0.5,
) -> np.ndarray:
    """Create a cube cross layout:
           [ U ]
    [ L ][ F ][ R ][ B ]
           [ D ]
    """
    face_w, face_h = 1024, 1024
    if scale != 1.0:
        scaled_w = int(face_w * scale)
        scaled_h = int(face_h * scale)
        imgs = {k: cv2.resize(v, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA) for k, v in cam_images.items()}
    else:
        scaled_w, scaled_h = face_w, face_h
        imgs = cam_images

    canvas = np.zeros((scaled_h * 3, scaled_w * 4, 3), dtype=np.uint8)

    slots = {
        "U": (0, 1),
        "L": (1, 0),
        "F": (1, 1),
        "R": (1, 2),
        "B": (1, 3),
        "D": (2, 1),
    }

    for cam_id, (row, col) in slots.items():
        if cam_id in imgs:
            r0 = row * scaled_h
            c0 = col * scaled_w
            canvas[r0 : r0 + scaled_h, c0 : c0 + scaled_w] = imgs[cam_id]
            # Add label
            cv2.putText(
                canvas,
                cam_id,
                (c0 + 20, r0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return canvas


def generate_sequence_previews(
    seq_dir: str | Path,
    out_dir: str | Path | None = None,
    modes: list[str] | None = None,
) -> dict[str, str]:
    seq_path = Path(seq_dir)
    out_path = Path(out_dir or (seq_path / "previews"))
    out_path.mkdir(parents=True, exist_ok=True)

    with open(seq_path / "sequence.json", "r", encoding="utf-8") as f:
        seq_meta = json.load(f)

    frames_jsonl_path = seq_path / "frames.jsonl"
    with open(frames_jsonl_path, "r", encoding="utf-8") as f:
        frames = [json.loads(line) for line in f if line.strip()]

    rate_hz = float(seq_meta.get("rate_hz", 2.0))  # default-ok: framerate default
    n_frames = len(frames)
    rig = Rig.load_nominal().make_true(20260922)

    if modes is None:
        modes = ["equirect_ned"]

    rays_ref, el_grid = precompute_equirect_rays(2048, 1024)

    created_files: dict[str, str] = {}

    for mode in modes:
        print(f"Generating preview '{mode}' for {seq_path.name} ({n_frames} frames)...")
        mp4_name = f"{seq_path.name}_{mode}.mp4"
        mp4_path = out_path / mp4_name

        if mode.startswith("equirect"):
            frame_size = (2048, 1024)
        else:  # cube_cross
            frame_size = (2048, 1536)  # 512x512 faces

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, rate_hz, frame_size)

        first_png = out_path / f"{seq_path.name}_{mode}_first.png"
        mid_png = out_path / f"{seq_path.name}_{mode}_mid.png"
        last_png = out_path / f"{seq_path.name}_{mode}_last.png"

        mid_k = n_frames // 2

        for k, frame_data in enumerate(frames):
            cam_imgs = {}
            for cam_id in ("F", "R", "B", "L", "U", "D"):
                png_rel = frame_data["cams"][cam_id]["png"]
                img = cv2.imread(str(seq_path / png_rel))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cam_imgs[cam_id] = img_rgb

            if mode == "equirect_body":
                frame_rgb = build_equirect_frame(cam_imgs, rig, rays_ref, r_ref_body=None, el_grid=el_grid)
            elif mode == "equirect_ned":
                r_ecef_body = np.array(frame_data["aircraft"]["R_ecef_body"], dtype=np.float64)
                # Compute R_ned_body
                lat = frame_data["aircraft"]["lat_deg"]
                lon = frame_data["aircraft"]["lon_deg"]
                from xvs.geo import wgs84
                r_ecef_ned = wgs84.r_ecef_ned(lat, lon)
                r_ned_body = r_ecef_ned.T @ r_ecef_body

                frame_rgb = build_equirect_frame(cam_imgs, rig, rays_ref, r_ref_body=r_ned_body, el_grid=el_grid)
            elif mode == "cube_cross":
                frame_rgb = build_cube_cross_frame(cam_imgs, scale=0.5)
            else:
                raise ValueError(f"Unknown mode '{mode}'")

            # Add timestamp overlay
            cv2.putText(
                frame_rgb,
                f"t={frame_data['t_s']:.1f}s | {seq_meta['scenario_id']} | {mode}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

            if k == 0:
                cv2.imwrite(str(first_png), frame_bgr)
            elif k == mid_k:
                cv2.imwrite(str(mid_png), frame_bgr)
            elif k == n_frames - 1:
                cv2.imwrite(str(last_png), frame_bgr)

        writer.release()
        created_files[mode] = str(mp4_path)
        print(f"  Saved {mp4_path} and capture PNGs.")

    return created_files


def draw_gt_overlays(
    seq_dir: str | Path,
    out_dir: str | Path,
    frame_indices: list[int],
    cams: tuple[str, ...] = ("F", "L", "R", "B"),
) -> list[str]:
    """Draw horizon GT on specified sequence frames for visual verification (G4)."""
    seq_path = Path(seq_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(seq_path / "frames.jsonl", "r", encoding="utf-8") as f:
        frames = [json.loads(line) for line in f if line.strip()]

    saved_images = []

    for k in frame_indices:
        if k >= len(frames):
            continue
        fdata = frames[k]
        for cam_id in cams:
            cdata = fdata["cams"][cam_id]
            png_p = seq_path / cdata["png"]
            img = cv2.imread(str(png_p))

            gt = cdata.get("gt", {})
            if gt.get("horizon_visible", False) and gt.get("horizon_uv"):
                pts = np.array(gt["horizon_uv"], dtype=np.int32)
                # Draw polyline in bright green
                cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # Annotation
            cv2.putText(
                img,
                f"{cam_id} k={k} t={fdata['t_s']:.1f}s GT-HORIZON",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out_file = out_path / f"gt_overlay_{cam_id}_k{k:04d}.png"
            cv2.imwrite(str(out_file), img)
            saved_images.append(str(out_file))

    print(f"Saved {len(saved_images)} GT overlay images to {out_path}.")
    return saved_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenXVS sequence preview MP4s and overlays")
    parser.add_argument("--seq-dir", required=True, help="Path to sequence directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for previews")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["equirect_ned"],
        choices=["equirect_ned", "equirect_body", "cube_cross"],
        help="Preview modes",
    )
    parser.add_argument("--gt-overlay", action="store_true", help="Draw horizon GT overlays")
    args = parser.parse_args()

    generate_sequence_previews(args.seq_dir, args.out_dir, args.modes)
    if args.gt_overlay:
        out_overlay = Path(args.out_dir or Path(args.seq_dir) / "gt_overlays")
        draw_gt_overlays(args.seq_dir, out_overlay, frame_indices=[0, 30, 60, 90, 119])
    return 0


if __name__ == "__main__":
    sys.exit(main())
