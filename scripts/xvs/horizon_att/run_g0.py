"""CLI runner for OpenXVS WP-06 Gate G0 horizon attitude evaluation.

Executes clean and degraded evaluation over G4 sequences across:
- Camera sets: F, horizontal (F+R+B+L), all (F+R+B+L+U+D)
- Altitude errors: sigma_h in {0, 10, 50} m
- Sequences: T1_cruise, T2_ksfo28l, T3_turn_climb

Outputs structured JSON results to output/xvs/wp06/g0/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, REPO_ROOT, data_root_source
from xvs.horizon_att.pipeline import run_sequence_evaluation

DEFAULT_OUT_DIR = REPO_ROOT / "output" / "xvs" / "wp06" / "g0"
DEFAULT_SENSOR_CONFIG = REPO_ROOT / "configs" / "xvs" / "sensors" / "rig_mono_v0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenXVS WP-06 G0 evaluation")
    parser.add_argument(
        "--sequences",
        type=str,
        default="T1_cruise,T2_ksfo28l,T3_turn_climb",
        help="Comma-separated sequence names to evaluate",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="clean",
        help="Comma-separated evaluation modes ('clean', 'degraded', or 'clean,degraded')",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root data directory containing xvs/ output hierarchy (defaults to XVS_DATA_ROOT or repo output/)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to G4 sequences directory (defaults to data-root / xvs / wp01 / g4)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--degraded-base-dir",
        type=Path,
        default=None,
        help="Path to degraded images root (defaults to data-root or repo output/xvs/wp06/g0/degraded)",
    )
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=DEFAULT_SENSOR_CONFIG,
        help="Sensor configuration JSON for lens distortion/undistortion",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 12) - 4)),
        help="Maximum multiprocessing worker processes (capped at logical cores - 4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames per sequence (for fast testing)",
    )
    args = parser.parse_args()
    data_root = get_data_root(args.data_root)
    if args.data_dir is None:
        args.data_dir = data_root / "xvs" / "wp01" / "g4"
    if args.degraded_base_dir is None:
        args.degraded_base_dir = data_root / "xvs" / "wp06" / "g0" / "degraded"
    return args


def main() -> int:
    args = parse_args()
    seq_names = [s.strip() for s in args.sequences.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OpenXVS WP-06 Gate G0 — Horizon Attitude Estimator Runner")
    print(f"Data Root:    {data_root} (source: {data_root_source(args.data_root)})")
    print(f"Sequences:    {seq_names}")
    print(f"Modes:        {modes}")
    print(f"Workers:      {args.max_workers} (system cap: {(os.cpu_count() or 12) - 4})")
    print(f"Output Dir:   {args.out_dir}")
    print("=" * 70)

    all_summaries: dict[str, Any] = {}

    for mode in modes:
        for seq in seq_names:
            seq_dir = args.data_dir / seq
            if not seq_dir.is_dir():
                raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

            degraded_dir = None
            sensor_cfg = None
            if mode == "degraded":
                degraded_dir = args.degraded_base_dir / seq
                sensor_cfg = args.sensor_config
                if not degraded_dir.is_dir():
                    raise FileNotFoundError(f"Degraded directory not found: {degraded_dir}")

            print(f"\n---> Running {seq} [{mode.upper()}] ...")
            t0 = time.time()
            res = run_sequence_evaluation(
                seq_name=seq,
                seq_dir=seq_dir,
                out_dir=args.out_dir,
                max_workers=args.max_workers,
                sensor_config_path=sensor_cfg,
                degraded_dir=degraded_dir,
                max_frames=args.max_frames,
            )
            elapsed = time.time() - t0
            summary_key = f"{seq}_{mode}"
            all_summaries[summary_key] = res["stats"]

            # Print brief summary table
            print(f"Completed {res['n_frames']} frames in {elapsed:.1f}s ({res['mean_fps']:.1f} fps)")
            print(f"{'Camera Set':<12} {'Sigma_h':<8} {'Pitch Med':<10} {'Roll Med':<10} {'Total Med':<10} {'Failures':<8}")
            print("-" * 62)
            for k_stat, stat in res["stats"].items():
                print(
                    f"{stat['cam_set']:<12} "
                    f"{stat['sigma_h']:<8.0f} "
                    f"{stat['pitch_error_median_deg']:<10.4f} "
                    f"{stat['roll_error_median_deg']:<10.4f} "
                    f"{stat['total_error_median_deg']:<10.4f} "
                    f"{stat['failures']:<8}"
                )

    summary_file = args.out_dir / "summary_all_runs.json"
    existing_summaries = {}
    if summary_file.is_file():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                existing_summaries = json.load(f)
        except Exception:
            existing_summaries = {}
    existing_summaries.update(all_summaries)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(existing_summaries, f, indent=2)

    print(f"\nAll results saved under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
