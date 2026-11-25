"""Generate degraded G4 image sequences through the WP-03 imaging chain.

Applies:
- sRGB linearisation (EOTF) + Rec. 709 monochrome
- Discrete PSF convolution (optics blur)
- Brown-Conrady geometric lens distortion (k1=-0.10, k2=0.02)
- cos^4 theta + radial polynomial vignetting
- EMVA 1288 linear sensor model (shot noise, dark current, read noise, PRNU, DSNU)
- 12-bit ADC -> 8-bit display PNG output

Outputs saved under output/xvs/wp06/g0/degraded/{seq}/{cam}/{k:06d}.png.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, REPO_ROOT, data_root_source
from xvs.sensor.chain import SensorConfig, CameraSimulator, save_display_png

DEFAULT_OUT_DIR = REPO_ROOT / "output" / "xvs" / "wp06" / "g0" / "degraded"
DEFAULT_SENSOR_CONFIG = REPO_ROOT / "configs" / "xvs" / "sensors" / "rig_mono_v0.json"
CAMERAS = ("F", "R", "B", "L", "U", "D")


def _process_camera_frames(args: tuple) -> tuple[str, str, int]:
    """Worker task processing all frames for a single (seq, cam) pair."""
    seq_name, cam_id, frame_indices, clean_dir_str, out_dir_str, config_path_str = args

    cfg = SensorConfig.from_file(config_path_str)
    # Pre-cache PSF, distortion map, vignetting map
    cfg.get_psf_kernel()
    cfg.get_backward_map()
    cfg.get_vignetting_map()

    sim = CameraSimulator(cfg, sensor_id=f"CAM_{cam_id}")

    clean_cam_dir = Path(clean_dir_str) / seq_name / cam_id
    out_cam_dir = Path(out_dir_str) / seq_name / cam_id
    out_cam_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for k in frame_indices:
        in_path = clean_cam_dir / f"{k:06d}.png"
        out_path = out_cam_dir / f"{k:06d}.png"

        if out_path.is_file():
            count += 1
            continue

        if not in_path.is_file():
            continue

        img_raw = np.array(Image.open(in_path))
        res = sim.process_frame(
            img_raw,
            image_id=f"{seq_name}_{cam_id}",
            frame_idx=k,
            warp_method="cv2",
        )
        save_display_png(res["display"], out_path)
        count += 1

    return seq_name, cam_id, count


def generate_degraded_sequences(
    sequences: list[str],
    clean_base_dir: Path,
    out_base_dir: Path,
    config_path: Path,
    max_workers: int = 8,
) -> None:
    cpu_limit = max(1, (os.cpu_count() or 12) - 4)
    workers = min(max_workers, cpu_limit)

    if not clean_base_dir.is_dir():
        raise FileNotFoundError(f"Clean G4 directory not found: {clean_base_dir}")

    tasks = []
    for seq in sequences:
        clean_seq_dir = clean_base_dir / seq
        out_seq_dir = out_base_dir / seq
        out_seq_dir.mkdir(parents=True, exist_ok=True)

        # Copy frames.jsonl and sequence.json if available
        for fname in ("frames.jsonl", "sequence.json"):
            src_f = clean_seq_dir / fname
            dst_f = out_seq_dir / fname
            if src_f.is_file() and not dst_f.is_file():
                shutil.copy2(src_f, dst_f)

        # Find frames
        frames_f = clean_seq_dir / "frames.jsonl"
        with open(frames_f, "r", encoding="utf-8") as f:
            n_frames = sum(1 for _ in f)
        frame_indices = list(range(n_frames))

        for cam in CAMERAS:
            tasks.append((
                seq, cam, frame_indices, str(clean_base_dir), str(out_base_dir), str(config_path)
            ))

    print(f"Generating degraded sets: {len(tasks)} camera streams across {len(sequences)} sequences on {workers} workers...")
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_process_camera_frames, tasks)
    elapsed = time.time() - t0

    total_images = sum(r[2] for r in results)
    print(f"Degraded generation finished: {total_images} images in {elapsed:.1f}s ({total_images / max(1e-3, elapsed):.1f} img/s).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate degraded G4 sequences via WP-03 chain")
    parser.add_argument("--sequences", type=str, default="T1_cruise,T2_ksfo28l,T3_turn_climb")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root data directory (defaults to XVS_DATA_ROOT or repo output/)",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=None,
        help="Path to clean G4 sequences directory (defaults to data-root / xvs / wp01 / g4)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for degraded images")
    parser.add_argument("--sensor-config", type=Path, default=DEFAULT_SENSOR_CONFIG)
    parser.add_argument("--max-workers", type=int, default=min(8, max(1, (os.cpu_count() or 12) - 4)))

    args = parser.parse_args()
    data_root = get_data_root(args.data_root)
    source = data_root_source(args.data_root)
    print(f"Data root: {data_root} (source: {source})")
    clean_dir = args.clean_dir if args.clean_dir is not None else data_root / "xvs" / "wp01" / "g4"
    out_dir = args.out_dir if args.out_dir is not None else data_root / "xvs" / "wp06" / "g0" / "degraded"
    seq_list = [s.strip() for s in args.sequences.split(",") if s.strip()]

    generate_degraded_sequences(
        sequences=seq_list,
        clean_base_dir=clean_dir,
        out_base_dir=out_dir,
        config_path=args.sensor_config,
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
