"""Tests for Tier A Amazon AOT `XVS-ActiveRange` passive proof CLI (`scripts/xvs/run_aot_activerange_proof.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "xvs" / "run_aot_activerange_proof.py"


def test_aot_proof_cli_smoke_synthetic_produces_video_metrics_and_pack(tmp_path: Path) -> None:
    """Offline CI gate: `--smoke-synthetic` emits valid 1080p MP4, metrics JSON, report MD, and 1:1 256x256 pack."""
    out_dir = tmp_path / "aot_smoke"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--smoke-synthetic",
            "--out-dir",
            str(out_dir),
            "--num-encounters",
            "6",
            "--max-frames-per-enc",
            "16",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    mp4_path = out_dir / "xvs_activerange_aot_proof_1080p.mp4"
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "V3_M1_aot_realsky_proof.md"
    pack_dir = out_dir / "realsky_aot_pack"
    manifest_path = pack_dir / "manifest.json"

    assert mp4_path.is_file()
    assert mp4_path.stat().st_size > 10_000

    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in (
        "conformal_coverage",
        "cp_msc_overestimation_rate",
        "naive_overestimation_rate",
        "det_only_tau_mae_inv_s",
        "gt_box_physics_tau_mae_inv_s",
        "det_frames_count",
        "coast_frames_count",
        "pre_acq_frames_count",
        "tau_mae_inv_s",
        "xvs_d2_detection_rate",
        "q_hat_conformal",
    ):
        assert key in metrics, f"Missing metric key {key!r}"
    assert (
        int(metrics["det_frames_count"])
        + int(metrics["coast_frames_count"])
        + int(metrics["pre_acq_frames_count"])
        == int(metrics["n_test_frames"])
    )
    assert 0.0 <= float(metrics["conformal_coverage"]) <= 1.0
    assert float(metrics["conformal_coverage"]) >= 0.85
    assert float(metrics["cp_msc_overestimation_rate"]) <= float(metrics["naive_overestimation_rate"])
    assert float(metrics["det_only_tau_mae_inv_s"]) >= 0.0
    assert float(metrics["gt_box_physics_tau_mae_inv_s"]) < 0.25

    assert report_path.is_file()
    report_txt = report_path.read_text(encoding="utf-8")
    assert "CP-MSC" in report_txt
    assert "det_only" in report_txt
    assert "gt_box_physics" in report_txt
    assert "total crisp" in report_txt

    # Verify native 1:1 256x256 Tab 3 browser pack schema & size budget
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "realsky_aot_pack_v1"
    assert float(manifest["focal_px"]) == 2690.0
    assert len(manifest["encounters"]) >= 1

    total_pack_bytes = manifest_path.stat().st_size
    checked_imgs = 0
    for enc in manifest["encounters"]:
        assert len(enc["frames"]) >= 8
        for frm in enc["frames"]:
            img_p = pack_dir / frm["file"]
            assert img_p.is_file(), f"Missing pack crop {img_p}"
            total_pack_bytes += img_p.stat().st_size
            if checked_imgs < 4:
                with Image.open(img_p) as im:
                    assert im.size == (256, 256), f"Expected 1:1 256x256 crop, got {im.size}"
                checked_imgs += 1
            assert len(frm["gt_box_crop"]) == 4
            assert float(frm["range_distance_m"]) > 0.0
    assert total_pack_bytes <= 1_500_000
