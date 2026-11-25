"""Tests for 4-channel aviation multi-sensor media pack builder (`scripts/xvs/prepare_studio_media.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "xvs" / "prepare_studio_media.py"


def test_prepare_studio_media_smoke_fast_creates_all_four_channels(tmp_path: Path) -> None:
    """Offline `--smoke-fast` produces valid manifest.json + crisp color JPEGs across all 4 sensor channels."""
    out_dir = tmp_path / "studio_media_smoke"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--out-dir",
            str(out_dir),
            "--smoke-fast",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    man_path = out_dir / "manifest.json"
    assert man_path.is_file()
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert man["schema"] == "xvs.studio_media_pack/1"
    assert man["coordinate_frame"] == "NED"
    assert int(man["total_media_bytes"]) > 20_000

    expected_specs = {
        "eo_sky": (512, 384),
        "runway_land": (640, 360),
        "thermal_dve": (640, 360),
        "crossview_nav": (512, 512),
    }
    for ch_name, exp_wh in expected_specs.items():
        assert ch_name in man["channels"], f"Missing channel {ch_name!r}"
        ch = man["channels"][ch_name]
        assert int(ch["num_frames"]) >= 4
        assert tuple(ch["resolution"]) == exp_wh
        assert len(ch["frames"]) == int(ch["num_frames"])

        first = ch["frames"][0]
        img_p = out_dir / str(first["file"])
        assert img_p.is_file(), f"Missing frame file {img_p}"
        with Image.open(img_p) as im:
            assert im.size == exp_wh
            assert im.mode == "RGB"

    # Verify Channel 2 exact 12-keypoint projection metadata
    land_f0 = man["channels"]["runway_land"]["frames"][0]
    assert len(land_f0["gt_keypoints_12_px"]) == 12
    assert len(land_f0["runway_bbox_px"]) == 4

    # Verify Channel 3 paired EO-fog + physics LWIR files & clear->fog->stratus_cloud sweep
    th_frames = man["channels"]["thermal_dve"]["frames"]
    regimes = {str(f["regime"]) for f in th_frames}
    assert "clear" in regimes
    assert "radiation_fog" in regimes
    assert "stratus_cloud" in regimes
    for f in th_frames:
        assert (out_dir / str(f["eo_file"])).is_file()
        assert (out_dir / str(f["lwir_file"])).is_file()

    # Verify Channel 4 paired oblique aerial + NAIP satellite tiles & stealth GPS spoof ramp
    nav_frames = man["channels"]["crossview_nav"]["frames"]
    assert any(not bool(f["spoof_active"]) for f in nav_frames)
    assert any(bool(f["spoof_active"]) and float(f["spoof_offset_m"]) > 50.0 for f in nav_frames)
    for f in nav_frames:
        assert (out_dir / str(f["aerial_file"])).is_file()
        assert (out_dir / str(f["sat_ref_file"])).is_file()
        assert len(f["correspondence_pairs_px"]) >= 4
