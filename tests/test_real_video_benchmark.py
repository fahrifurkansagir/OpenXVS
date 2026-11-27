"""Unit tests for `scripts/xvs/run_real_video_benchmark.py`.

Covers:
  1. Benchmark manifest schema, open-license enforcement, SHA-256 & missing-clip detection.
  2. Plain-English HUD label formatting across all 4 test scenarios & every fault phase,
     plus strict forbidden-acronym/jargon denylist enforcement.
  3. 1280x720 plain-English video frame & summary scorecard rendering (all 4 overlay kinds).
  4. 16-frame sample-frame exporter (`4` per scenario + manifest SHA-256 integrity).
  5. Ring-0 Simplex RTA fail-closed approach go-around latch parity.
  6. Fast CLI `--check-sources-only` dry verification without running heavy ONNX inference.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "xvs" / "run_real_video_benchmark.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_real_video_benchmark", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_real_video_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_textured_test_video(path: Path, n_frames: int = 12, w: int = 320, h: int = 240) -> None:
    """Create a tiny in-memory-derived MP4 strictly inside pytest `tmp_path`."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(n_frames):
        b = np.clip(((xx + 17 * k) % 240) + 10, 10, 245).astype(np.uint8)
        g = np.clip(((yy + 13 * k) % 240) + 10, 10, 245).astype(np.uint8)
        r = np.clip((((xx ^ yy) + 9 * k) % 240) + 10, 10, 245).astype(np.uint8)
        writer.write(np.stack([b, g, r], axis=-1))
    writer.release()


def test_plain_english_jargon_guard_blocks_internal_acronyms() -> None:
    mod = _load_benchmark_module()
    # Clean plain English passes
    mod.assert_plain_english_text("TEST 01 · REAL AIR-TO-AIR TRACKING")
    mod.assert_plain_english_text("SAFE REJECT: UNCALIBRATED VIEW (0 UNSAFE OUTPUTS)")
    mod.assert_plain_english_text("GPS SPOOF CAUGHT -> SWITCHED TO VISUAL")
    mod.assert_plain_english_text("CAMERA FROZEN -> SAFE FALLBACK IN 1 FRAME")

    # Every internal acronym / raw code token MUST raise ValueError
    for bad in (
        "P0_INVALID_INHIBIT active",
        "Mean kpSig = 12.4px",
        "HMI=0 verified",
        "q_lwir dropped to 0.00",
        "3-of-5 TBC locked",
        "Calibrated PSR 1.42",
        "8-arm CUSUM triggered",
        "LWIR FFC freeze",
        "DUAL DVE SVS fallback",
        "Ring-0 Simplex RTA",
        "Scene is OOD",
        "MAP PnPL converged",
        "HPL 18m VPL 9m",
        "SimCC peak clarity",
        "Dead pixel FPN lines",
    ):
        with pytest.raises(ValueError, match="Forbidden internal"):
            mod.assert_plain_english_text(bad)


def test_all_four_scenario_label_formatters_emit_clear_plain_english() -> None:
    mod = _load_benchmark_module()

    # Test 01 states
    for det, conf, hits in ((False, 0, 0), (True, 0, 2), (True, 1, 4)):
        lbl = mod.format_test01_labels(
            detected=det,
            confirmed_tracks=conf,
            hits_in_window=hits,
            window_len=5,
            speed_ms=19.4,
            clip_label="Budapest Glider",
            static_clutter_blocked=7,
        )
        assert lbl.top_left_title == "TEST 01 · REAL AIR-TO-AIR TRACKING"
        lbl.validate_plain_english()

    # Test 02 states
    for adm in (False, True):
        lbl = mod.format_test02_labels(
            admitted=adm,
            speed_ms=22.1,
            peak_sharpness=0.14,
            rejected_count=50,
            total_seen=50,
            clip_label="Teterboro Night",
        )
        assert lbl.top_left_title == "TEST 02 · UNSEEN COCKPIT LANDING"
        if not adm:
            assert "SAFE REJECT" in lbl.status_pill
            assert "0 UNSAFE OUTPUTS" in lbl.status_pill
        lbl.validate_plain_english()

    # Test 03 states
    for phase, rec, spoof in (
        ("offmap_nominal", False, False),
        ("heading_spin", True, False),
        ("gps_outage", True, False),
        ("spoof", True, False),
        ("spoof", True, True),
    ):
        lbl = mod.format_test03_labels(
            phase=phase,
            tracked_features=42,
            heading_spin_recovered=rec,
            gps_spoof_caught=spoof,
            drift_m=18.5,
            speed_ms=34.0,
        )
        assert lbl.top_left_title == "TEST 03 · AERIAL DRONE VISUAL NAV"
        if phase == "spoof" and spoof:
            assert "GPS SPOOF CAUGHT" in lbl.status_pill
        lbl.validate_plain_english()

    # Test 04 states
    for phase, caught, unc in (
        ("nominal", False, 14.0),
        ("shutter_freeze", False, 20.0),
        ("shutter_freeze", True, 55.0),
        ("dead_lines", True, 85.0),
        ("white_noise", True, 92.0),
        ("dual_blind", True, 100.0),
    ):
        lbl = mod.format_test04_labels(
            phase=phase,
            fault_caught=caught,
            uncertainty_pct=unc,
            speed_ms=11.5,
        )
        assert lbl.top_left_title == "TEST 04 · AIRBORNE THERMAL FAIL-SAFE"
        lbl.validate_plain_english()


def test_plain_english_frame_and_slate_rendering_1280x720() -> None:
    mod = _load_benchmark_module()
    base = np.full((360, 640, 3), (85, 110, 140), dtype=np.uint8)
    cv2.circle(base, (320, 180), 40, (220, 225, 230), -1)

    lbl1 = mod.format_test01_labels(
        detected=True,
        confirmed_tracks=1,
        hits_in_window=3,
        window_len=5,
        speed_ms=18.2,
        clip_label="Glider",
        static_clutter_blocked=4,
    )
    f1 = mod.render_plain_english_frame(
        base,
        labels=lbl1,
        overlay_kind="d2",
        overlay_data={"top_box": (290.0, 150.0, 350.0, 210.0, 0.84), "tile_boxes": [(0, 50, 256, 306)]},
    )
    assert f1.shape == (mod.REEL_HEIGHT, mod.REEL_WIDTH, 3)
    assert f1.dtype == np.uint8
    assert float(np.std(f1)) > 20.0

    lbl2 = mod.format_test02_labels(
        admitted=False, speed_ms=20.0, peak_sharpness=0.11, rejected_count=10, total_seen=10, clip_label="KORS"
    )
    f2 = mod.render_plain_english_frame(
        base,
        labels=lbl2,
        overlay_kind="land",
        overlay_data={"kpts_norm": [[0.4 + 0.02 * i, 0.5 + 0.02 * (i % 3)] for i in range(12)], "admitted": False},
    )
    assert f2.shape == (mod.REEL_HEIGHT, mod.REEL_WIDTH, 3)

    lbl3 = mod.format_test03_labels(
        phase="spoof", tracked_features=35, heading_spin_recovered=True, gps_spoof_caught=True, drift_m=24.0, speed_ms=31.0
    )
    f3 = mod.render_plain_english_frame(
        base,
        labels=lbl3,
        overlay_kind="nav",
        overlay_data={"phase": "spoof", "tracked_pts": [(0.3, 0.4), (0.6, 0.5)], "spoof_caught": True, "gps_drift_m": 24.0},
    )
    assert f3.shape == (mod.REEL_HEIGHT, mod.REEL_WIDTH, 3)

    lbl4 = mod.format_test04_labels(phase="dual_blind", fault_caught=True, uncertainty_pct=100.0, speed_ms=10.5)
    f4 = mod.render_plain_english_frame(
        base,
        labels=lbl4,
        overlay_kind="thermal",
        overlay_data={"phase": "dual_blind", "peak_norm_xy": (0.5, 0.5)},
    )
    assert f4.shape == (mod.REEL_HEIGHT, mod.REEL_WIDTH, 3)

    slate = mod.render_title_or_summary_slate(
        title="OPENXVS REAL-VIDEO BENCHMARK",
        subtitle="Plain-English test summary",
        cards=(
            ("TEST 01", "PASS", "Wide-view aircraft tracking."),
            ("TEST 02", "SAFE REJECT", "0 unsafe guidance outputs."),
            ("TEST 03", "PASS", "Heading spin recovered & GPS spoof caught."),
            ("TEST 04", "PASS", "Sensor faults caught in 1 frame."),
        ),
        footer="NON-FLIGHT BENCHMARK",
    )
    assert slate.shape == (mod.REEL_HEIGHT, mod.REEL_WIDTH, 3)
    assert float(np.std(slate)) > 15.0


def test_manifest_validation_and_16_sample_frame_export(tmp_path: Path) -> None:
    mod = _load_benchmark_module()
    catalog = []
    for snum, ch in ((1, "d2"), (2, "land"), (3, "nav"), (4, "thermal")):
        for rep in (1, 2):
            vpath = tmp_path / f"s{snum}_{rep}.mp4"
            _make_textured_test_video(vpath, n_frames=16)
            catalog.append(
                mod.ClipSpec(
                    clip_id=f"test_s{snum}_{rep}",
                    scenario_num=snum,
                    channel=ch,
                    is_new_unseen=(rep == 1),
                    local_rel=str(vpath),
                    display_name=f"Clip S{snum}-{rep}",
                    source_url=f"https://commons.wikimedia.org/wiki/File:Test_{snum}_{rep}.webm",
                    license="Public domain",
                    attribution="Unit Test",
                    description="In-memory unit test video.",
                    start_sec=0.0,
                    eval_duration_sec=1.5,
                )
            )

    manifest = mod.build_and_validate_manifest(catalog, repo_root=tmp_path)
    assert manifest["schema"] == "xvs.v112_real_video_benchmark_manifest/1"
    assert manifest["policy"]["synthetic_videos_allowed"] is False
    assert manifest["policy"]["open_licenses_only"] is True
    assert manifest["summary"]["all_valid"] is True
    assert manifest["summary"]["ready_clips"] == 8
    assert manifest["summary"]["new_unseen_ready_clips"] == 4
    for entry in manifest["clips"]:
        assert len(entry["sha256"]) == 64
        assert entry["probe"]["decodable"] is True
        assert entry["probe"]["non_black"] is True

    # Missing clip must mark all_valid=False fail-closed
    bad_catalog = catalog + [
        mod.ClipSpec(
            clip_id="missing_clip",
            scenario_num=1,
            channel="d2",
            is_new_unseen=True,
            local_rel=str(tmp_path / "does_not_exist.mp4"),
            display_name="Missing",
            source_url="https://example.org/missing.mp4",
            license="CC0",
            attribution="Test",
            description="Missing clip.",
        )
    ]
    bad_man = mod.build_and_validate_manifest(bad_catalog, repo_root=tmp_path)
    assert bad_man["summary"]["all_valid"] is False

    # Sample frame export produces exactly 16 clean JPEG frames (4 per scenario)
    out_dir = tmp_path / "out"
    sf_man = mod.export_real_sample_frames(catalog, out_dir, repo_root=tmp_path)
    assert sf_man["total_frames"] == 16
    assert len(sf_man["frames"]) == 16
    by_s = {1: 0, 2: 0, 3: 0, 4: 0}
    for f_entry in sf_man["frames"]:
        by_s[int(f_entry["scenario_num"])] += 1
        jp = out_dir / "sample_frames" / f_entry["file"]
        assert jp.is_file()
        assert mod.compute_sha256(jp) == f_entry["sha256"]
    assert by_s == {1: 4, 2: 4, 3: 4, 4: 4}


def test_ring0_simplex_rta_fail_closed_approach_latch() -> None:
    mod = _load_benchmark_module()
    rta = mod.Ring0SimplexRta(prefer_ffi=False)
    try:
        ok_dec = rta.step(hpl_m=10.0, hal_m=40.0, vpl_m=5.0, val_m=15.0, approach_active=False)
        assert ok_dec["mode_id"] == 4
        assert ok_dec["mandatory_go_around"] is False

        rej_dec = rta.step(hpl_m=999.0, hal_m=40.0, vpl_m=999.0, val_m=15.0, ml_fault=True, approach_active=True)
        assert rej_dec["mode_id"] == 0
        assert rej_dec["mandatory_go_around"] is True
        assert rej_dec["inhibit_avoidance"] is True
    finally:
        rta.close()


def test_cli_check_sources_only_passes_on_curated_suite(tmp_path: Path) -> None:
    mod = _load_benchmark_module()
    # Point at canonical output dir if raw_clips exist; skip only if external clips aren't present in CI
    default_manifest = mod.DEFAULT_OUT_DIR / "raw_clips" / "new_a2a_glider_budapest.webm"
    if not default_manifest.is_file():
        pytest.skip("Gitignored local field clips not present in clean CI checkout")
    rc = mod.main(["--out-dir", str(mod.DEFAULT_OUT_DIR), "--check-sources-only"])
    assert rc == 0
