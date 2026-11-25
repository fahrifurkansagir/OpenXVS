"""Build deterministic WP-04 G1 training/validation and held-out test packs (`WP04_DETECTION.md` §9.3).

Writes:
- `<out_dir>/g1_train_val_pack_v0.tar.zst` and `<out_dir>/pack_manifest.json`
- `<out_dir>/g1_test_eval_pack_v0.tar.zst` and `<out_dir>/test_pack_manifest.json`
and executes the mandatory post-build leakage audit (`verify_pack_no_leakage`, Decision D34-S9).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.detection.pack import (
    DEFAULT_G3A_ROOT,
    build_both_g1_packs,
    extract_real_g3a_pack_items_from_splits,
    generate_synthetic_pack_items_from_splits,
)
from xvs.detection.splits import DEFAULT_SPLITS_CONFIG, load_splits_config


def _get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=10,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "HEAD"


def main(argv: list[str] | None = None) -> int:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"

    parser = argparse.ArgumentParser(description="Build WP-04 G1 deterministic train/val and test packs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for G1 tar.zst packs and manifests.",
    )
    parser.add_argument(
        "--splits-config",
        type=Path,
        default=DEFAULT_SPLITS_CONFIG,
        help="Path to g1_splits_v0.json.",
    )
    parser.add_argument(
        "--g3a-root",
        type=Path,
        default=DEFAULT_G3A_ROOT,
        help="Path to G3a v0.1 dataset root (blocks/*/frames.jsonl and rendered PNGs).",
    )
    parser.add_argument(
        "--max-encounters-per-split",
        type=int,
        default=None,
        help="Optional cap on stratified train encounters (None = all 126 train, 27 val, 27 test encounters).",
    )
    parser.add_argument(
        "--max-stacks-per-encounter",
        type=int,
        default=1,
        help="Number of primary K-frame temporal stacks extracted per encounter.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
        help="Spatial crop size (pixels) for train/val ROI stacks and D0 hard-negative crops.",
    )
    parser.add_argument(
        "--test-roi-size",
        type=int,
        default=128,
        help="Spatial ROI size (pixels) for held-out test evaluation ROIs.",
    )
    parser.add_argument(
        "--k-frames",
        type=int,
        default=5,
        help="Number of contiguous Delta_t=1.0 s frames per stack.",
    )
    parser.add_argument(
        "--full-180",
        action="store_true",
        help="Build full 180-encounter compact packs (g1_train_val_pack_full.tar.zst & g1_test_eval_pack_full.tar.zst) with >= 165 Mpx train+val negative exposure.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use fast synthetic smoke items (for unit tests only; default extracts real G3a v0.1 frames).",
    )
    args = parser.parse_args(argv)

    eff_out_dir = (
        args.out_dir
        if args.out_dir is not None
        else (
            REPO_ROOT / "output" / "xvs" / "wp04" / "g1" / ("packs_full" if args.full_180 else "packs")
        )
    )
    tv_archive_name = "g1_train_val_pack_full.tar.zst" if args.full_180 else "g1_train_val_pack_v0.tar.zst"
    te_archive_name = "g1_test_eval_pack_full.tar.zst" if args.full_180 else "g1_test_eval_pack_v0.tar.zst"

    splits_cfg, _ = load_splits_config(args.splits_config)
    if args.smoke:
        train_val_items, test_items = generate_synthetic_pack_items_from_splits(
            splits_cfg,
            crop_size=min(64, int(args.crop_size)),
        )
    else:
        train_val_items, test_items = extract_real_g3a_pack_items_from_splits(
            splits_cfg=splits_cfg,
            g3a_root=args.g3a_root,
            max_encounters_per_split=None if args.full_180 else args.max_encounters_per_split,
            max_stacks_per_encounter=max(3, int(args.max_stacks_per_encounter)) if args.full_180 else args.max_stacks_per_encounter,
            crop_size=args.crop_size,
            test_roi_size=args.test_roi_size,
            k_frames=args.k_frames,
            full_180_mode=bool(args.full_180),
        )

    summary = build_both_g1_packs(
        out_dir=eff_out_dir,
        splits_path=args.splits_config,
        train_val_items=train_val_items,
        test_items=test_items,
        git_commit=_get_git_commit(),
        train_val_archive_name=tv_archive_name,
        test_archive_name=te_archive_name,
    )
    test_eids = sorted({str(it["encounter_id"]) for it in test_items if it["kind"] == "kframe_stack"})
    print(
        json.dumps(
            {
                "status": "OK",
                "mode": "smoke_synthetic" if args.smoke else ("full_180_g3a_v01" if args.full_180 else "real_g3a_v01"),
                "train_val_archive": summary["train_val_archive"],
                "test_archive": summary["test_archive"],
                "train_val_items": summary["train_val_manifest"]["item_count"],
                "test_items": summary["test_manifest"]["item_count"],
                "train_val_encounters_by_split": summary["train_val_manifest"]["encounters_by_split_count"],
                "test_encounters_by_split": summary["test_manifest"]["encounters_by_split_count"],
                "train_val_negative_time_block_exposure_mpx": summary["train_val_manifest"]["negative_time_block_exposure_mpx"],
                "train_val_negative_exposure_mpx": summary["train_val_manifest"]["negative_exposure_mpx"],
                "test_negative_time_block_exposure_mpx": summary["test_manifest"]["negative_time_block_exposure_mpx"],
                "test_positive_rois": summary["test_manifest"]["positive_rois_count"],
                "test_negative_rois": summary["test_manifest"]["negative_rois_count"],
                "test_size_bin_counts": summary["test_manifest"]["size_bin_counts"],
                "test_encounters_covered": len(test_eids),
                "leakage_errors": summary["leakage_errors"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
