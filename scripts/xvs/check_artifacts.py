"""Artifact and results integrity checker for OpenXVS gates (N12).

Decoupled from pytest suites so fresh clones pass unit tests without
requiring pre-generated gate outputs. Run at gate checkpoints.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.paths import get_data_root, data_root_source


def check_g2_artifacts(data_root: Path | str | None = None) -> bool:
    """Verify that Gate G2 artifacts are complete, valid, and green."""
    root = get_data_root(data_root)
    g2_dir = root / "xvs" / "wp01" / "g2"
    if not g2_dir.exists():
        raise FileNotFoundError(f"G2 directory not found: {g2_dir}")
    print(f"Checking G2 artifacts under {g2_dir}...")

    results_path = g2_dir / "results.json"
    results_ref_path = g2_dir / "results_ref.json"
    results_hw_path = g2_dir / "results_hw.json"

    errors = []
    if not results_path.exists():
        errors.append("Consolidated results.json missing")
    if not results_ref_path.exists():
        errors.append("results_ref.json missing")
    if not results_hw_path.exists():
        errors.append("results_hw.json missing")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False

    with open(results_ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    if not ref_data.get("all_scenes_pass", False):
        errors.append("ref backend did not pass all scenes")
    if ref_data.get("n_scenes") != 30:
        errors.append(f"ref expected 30 scenes, got {ref_data.get('n_scenes')}")
    if ref_data.get("total_detected") != 1200:
        errors.append(f"ref expected 1200 detected, got {ref_data.get('total_detected')}")

    ref_png_dir = g2_dir / "ref" / "pngs"
    ref_pose_dir = g2_dir / "ref" / "poses"
    n_ref_pngs = len(list(ref_png_dir.glob("*.png"))) if ref_png_dir.exists() else 0
    n_ref_poses = len(list(ref_pose_dir.glob("*.json"))) if ref_pose_dir.exists() else 0
    if n_ref_pngs != 30:
        errors.append(f"ref expected 30 PNGs, found {n_ref_pngs}")
    if n_ref_poses != 30:
        errors.append(f"ref expected 30 pose JSONs, found {n_ref_poses}")

    with open(results_hw_path, "r", encoding="utf-8") as f:
        hw_data = json.load(f)

    if hw_data.get("n_scenes") != 30:
        errors.append(f"hw expected 30 scenes, got {hw_data.get('n_scenes')}")
    if hw_data.get("total_detected") != 1200:
        errors.append(f"hw expected 1200 detected, got {hw_data.get('total_detected')}")

    hw_png_dir = g2_dir / "hw" / "pngs"
    hw_pose_dir = g2_dir / "hw" / "poses"
    n_hw_pngs = len(list(hw_png_dir.glob("*.png"))) if hw_png_dir.exists() else 0
    n_hw_poses = len(list(hw_pose_dir.glob("*.json"))) if hw_pose_dir.exists() else 0
    if n_hw_pngs != 30:
        errors.append(f"hw expected 30 PNGs, found {n_hw_pngs}")
    if n_hw_poses != 30:
        errors.append(f"hw expected 30 pose JSONs, found {n_hw_poses}")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False

    print("  [PASS] G2 artifacts: 30 ref + 30 hw scenes verified complete and green.")
    return True


def check_g3_artifacts(data_root: Path | str | None = None) -> bool:
    """Verify that Gate G3 artifacts are complete, valid, and green."""
    root = get_data_root(data_root)
    g3_dir = root / "xvs" / "wp01" / "g3"
    if not g3_dir.exists():
        raise FileNotFoundError(f"G3 directory not found: {g3_dir}")
    print(f"Checking G3 artifacts under {g3_dir}...")

    results_ref_path = g3_dir / "results_ref.json"
    results_hw_path = g3_dir / "results_hw.json"
    errors = []

    if not results_ref_path.exists():
        errors.append("results_ref.json missing")
    if not results_hw_path.exists():
        errors.append("results_hw.json missing")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False

    with open(results_ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    if not ref_data.get("all_scenes_pass", False):
        errors.append("ref backend did not pass all scenes")
    if ref_data.get("n_total_scenes") != 152:
        errors.append(f"ref expected 152 scenes, got {ref_data.get('n_total_scenes')}")

    with open(results_hw_path, "r", encoding="utf-8") as f:
        hw_data = json.load(f)
    if hw_data.get("n_total_scenes") != 152:
        errors.append(f"hw expected 152 scenes, got {hw_data.get('n_total_scenes')}")

    ref_png_dir = g3_dir / "ref" / "pngs"
    n_ref_pngs = len(list(ref_png_dir.glob("*.png"))) if ref_png_dir.exists() else 0
    if n_ref_pngs != 152:
        errors.append(f"ref expected 152 PNGs, found {n_ref_pngs}")

    hw_png_dir = g3_dir / "hw" / "pngs"
    n_hw_pngs = len(list(hw_png_dir.glob("*.png"))) if hw_png_dir.exists() else 0
    if n_hw_pngs != 152:
        errors.append(f"hw expected 152 PNGs, found {n_hw_pngs}")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False

    print("  [PASS] G3 artifacts: 152 ref + 152 hw scenes verified complete and green.")
    return True


def check_g4_artifacts(data_root: Path | str | None = None) -> bool:
    """Verify that Gate G4 sequence outputs are complete and valid."""
    root = get_data_root(data_root)
    g4_dir = root / "xvs" / "wp01" / "g4"
    if not g4_dir.exists():
        raise FileNotFoundError(f"G4 directory not found: {g4_dir}")
    print(f"Checking G4 artifacts under {g4_dir}...")
    errors = []

    seqs = ["T1_cruise", "T2_ksfo28l", "T3_turn_climb"]
    for s in seqs:
        s_dir = g4_dir / s
        if not s_dir.exists():
            errors.append(f"Sequence directory {s} missing")
            continue
        if not (s_dir / "sequence.json").exists():
            errors.append(f"{s}/sequence.json missing")
        if not (s_dir / "frames.jsonl").exists():
            errors.append(f"{s}/frames.jsonl missing")
        if not (s_dir / "perf.json").exists():
            errors.append(f"{s}/perf.json missing")
        if not (s_dir / "data_sources.json").exists():
            errors.append(f"{s}/data_sources.json missing")

    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        return False

    print("  [PASS] G4 sequence outputs verified.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenXVS gate artifacts")
    parser.add_argument("--gate", choices=["g2", "g3", "g4", "all"], default="all", help="Gate to check")
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

    success = True
    if args.gate in ("g2", "all"):
        if not check_g2_artifacts(args.data_root):
            success = False
    if args.gate in ("g3", "all"):
        if not check_g3_artifacts(args.data_root):
            success = False
    if args.gate in ("g4", "all"):
        if not check_g4_artifacts(args.data_root):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
