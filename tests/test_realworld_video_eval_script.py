"""Focused unit tests for `scripts/xvs/run_realworld_video_eval.py`.

Covers:
  1. CLI `--check-sources-only` with both `PENDING_USER_VIDEO` and valid/tampered entries.
  2. `PENDING_USER_VIDEO` honesty handling (never faked, exits 0 cleanly).
  3. 12-sector optical quality computation + side-by-side `1200x480` HUD panel rendering
     on a 2-frame in-memory test array (strictly in-memory — never written to `data/field_videos/`).
  4. Ring-0 Simplex RTA wrapper deterministic transitions & fail-closed approach latch.
  5. Brutally honest `FIELD_VIDEO_TRUTH_REPORT.md` + `field_video_truth_report.json` schema.
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
SCRIPT_PATH = REPO_ROOT / "scripts" / "xvs" / "run_realworld_video_eval.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("run_realworld_video_eval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_realworld_video_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_in_memory_test_frames() -> list[np.ndarray]:
    """Build 2 deterministic textured in-memory BGR frames for HUD/rendering unit tests only."""
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    frames: list[np.ndarray] = []
    for k in range(2):
        b = np.clip(((xx + 25 * k) % 255), 20, 235).astype(np.uint8)
        g = np.clip(((yy + 15 * k) % 255), 30, 225).astype(np.uint8)
        r = np.clip((((xx ^ yy) + 10 * k) % 255), 25, 230).astype(np.uint8)
        frames.append(np.stack([b, g, r], axis=-1))
    return frames


def test_check_sources_only_handles_pending_user_video_honestly(tmp_path: Path) -> None:
    mod = _load_eval_module()
    manifest = {
        "schema": "xvs.field_videos_manifest/1",
        "scenarios": {
            "scenario_a_air_to_air": {
                "status": "PENDING_USER_VIDEO",
                "note": "Awaiting real air-to-air clip.",
            },
            "scenario_b_runway_approach": {
                "status": "PENDING_USER_VIDEO",
                "note": "Awaiting real daytime cockpit approach clip.",
            },
            "scenario_c_nadir_aerial": {
                "status": "PENDING_USER_VIDEO",
                "note": "Awaiting real nadir aerial clip.",
            },
            "scenario_d_thermal_flight": {
                "status": "PENDING_USER_VIDEO",
                "note": "Awaiting real airborne thermal clip.",
            },
        },
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    res = mod.validate_manifest_sources(mpath, repo_root=tmp_path)
    assert res["all_ready_valid"] is True
    assert res["ready_count"] == 0
    assert res["pending_count"] == 4
    for scen_id, _, _ in mod.SCENARIO_SPECS:
        assert res["scenarios"][scen_id]["status"] == "PENDING_USER_VIDEO"
        assert res["scenarios"][scen_id]["valid"] is True

    rc = mod.main(["--manifest", str(mpath), "--check-sources-only"])
    assert rc == 0


def test_check_sources_only_detects_sha256_tampering(tmp_path: Path) -> None:
    mod = _load_eval_module()
    # Write a tiny 2-frame mp4 inside pytest tmp_path (never inside data/field_videos/)
    vid = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for fr in _make_in_memory_test_frames():
        writer.write(fr)
    writer.release()

    good_sha = mod.compute_sha256(vid)
    good_bytes = vid.stat().st_size

    def _make_manifest(sha_val: str) -> Path:
        doc = {
            "schema": "xvs.field_videos_manifest/1",
            "scenarios": {
                "scenario_a_air_to_air": {
                    "status": "READY",
                    "local_path": "sample.mp4",
                    "source_url": "https://example.org/sample.mp4",
                    "license": "Public domain",
                    "attribution": "Unit test",
                    "sha256": sha_val,
                    "bytes": good_bytes,
                    "width": 320,
                    "height": 240,
                    "fps": 10.0,
                    "duration_sec": 0.2,
                    "camera_intrinsics_status": "UNKNOWN_UNCALIBRATED",
                },
                "scenario_b_runway_approach": {"status": "PENDING_USER_VIDEO", "note": "pending"},
                "scenario_c_nadir_aerial": {"status": "PENDING_USER_VIDEO", "note": "pending"},
                "scenario_d_thermal_flight": {"status": "PENDING_USER_VIDEO", "note": "pending"},
            },
        }
        p = tmp_path / f"manifest_{sha_val[:6]}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    ok_manifest = _make_manifest(good_sha)
    ok_res = mod.validate_manifest_sources(ok_manifest, repo_root=tmp_path)
    assert ok_res["all_ready_valid"] is True
    assert ok_res["ready_count"] == 1
    assert ok_res["pending_count"] == 3
    assert ok_res["scenarios"]["scenario_a_air_to_air"]["probe"]["decodable"] is True

    bad_manifest = _make_manifest("0" * 64)
    bad_res = mod.validate_manifest_sources(bad_manifest, repo_root=tmp_path)
    assert bad_res["all_ready_valid"] is False
    assert any("sha256_mismatch" in e for e in bad_res["scenarios"]["scenario_a_air_to_air"]["errors"])
    assert mod.main(["--manifest", str(bad_manifest), "--check-sources-only"]) == 1


def test_side_by_side_hud_render_on_two_in_memory_frames() -> None:
    mod = _load_eval_module()
    frames = _make_in_memory_test_frames()
    rta = mod.Ring0SimplexWrapper(prefer_ffi=False)

    try:
        for idx, fr in enumerate(frames):
            sec_q = mod.compute_12_sector_optical_quality(fr, modality="eo")
            assert len(sec_q) == 12
            assert all(0.0 <= q <= 1.0 for q in sec_q)

            dec = rta.step(
                hpl_m=12.0 if idx == 0 else 55.0,
                hal_m=30.0,
                vpl_m=6.0 if idx == 0 else 22.0,
                val_m=15.0,
                spoof_alarm=(idx == 1),
                ml_detector_fault=False,
                ood_global_exceeded=False,
                transient_dropout=False,
                approach_active=(idx == 1),
                healthy_cams=6,
            )
            if idx == 0:
                assert dec.mode_label == "P4_NOMINAL_ML"
            else:
                assert dec.mode_label == "P0_INVALID_INHIBIT"
                assert dec.mandatory_go_around is True

            rendered = mod.render_side_by_side_hud(
                fr,
                scenario_id="scenario_a_air_to_air",
                channel="d2",
                frame_idx=idx,
                total_frames=2,
                timestamp_s=float(idx) * 0.1,
                latency_ms=14.2,
                simplex_decision=dec,
                sector_qualities_12=sec_q,
                hpl_m=12.0,
                vpl_m=6.0,
                gate_admitted=(idx == 0),
                gate_status_text="NOMINAL" if idx == 0 else "GO_AROUND_LATCH",
                tbc_hits_in_window=3 if idx == 0 else 1,
                tbc_confirmed_tracks=1 if idx == 0 else 0,
                hmi_count=0,
                cusum_stat=2.4 if idx == 0 else 18.9,
                cusum_quarantined=(idx == 1),
                camera_intrinsics_status="UNKNOWN_UNCALIBRATED",
                channel_overlay={
                    "detection": {"x1": 40.0, "y1": 50.0, "x2": 100.0, "y2": 110.0, "score": 0.87}
                },
                phase_banner="INJECTED TEST PHASE" if idx == 1 else "",
            )
            assert rendered.shape == (mod.PANEL_H, mod.COMBINED_W, 3)
            assert rendered.dtype == np.uint8
            assert float(np.std(rendered[:, : mod.LEFT_PANE_W])) > 10.0
            assert float(np.std(rendered[:, mod.LEFT_PANE_W :])) > 10.0
    finally:
        rta.close()


def test_truth_report_honesty_schema(tmp_path: Path) -> None:
    mod = _load_eval_module()
    scenario_reports = {
        "scenario_a_air_to_air": {
            "scenario_id": "scenario_a_air_to_air",
            "title": "Air-to-Air",
            "channel": "d2",
            "status": "EVALUATED",
            "source_url": "https://commons.wikimedia.org/wiki/File:Example.webm",
            "license": "Public domain",
            "attribution": "NASA",
            "sha256": "a" * 64,
            "camera_intrinsics_status": "UNKNOWN_UNCALIBRATED",
            "output_video": "outputs/field_video_eval/scenario_a_air_to_air_eval.mp4",
            "video_codec": "libx264",
            "frames_processed": 50,
            "mean_latency_ms": 18.5,
            "p95_latency_ms": 24.0,
            "effective_fps": 54.05,
            "admit_rate": 0.4,
            "admitted_frames": 20,
            "fail_closed_rejections": 30,
            "hmi_count": 0,
            "mean_12sector_optical_quality": 0.72,
            "d2_detected_frames": 20,
            "tbc_confirmed_frames": 18,
            "tbc_confirmed_tracks_max": 1,
            "cusum_quarantine_triggered": False,
            "cusum_max_stat": 0.0,
            "ffc_freeze_detected": False,
            "dual_blind_vacuity_max": 0.0,
            "dual_blind_fallback_triggered": False,
            "simplex_mode_histogram": {"P4_NOMINAL_ML": 50},
            "what_held": ["Zero HMI held."],
            "degradations_and_limits": ["Uncalibrated intrinsics."],
            "fail_closed_held": True,
        },
        "scenario_b_runway_approach": {
            "scenario_id": "scenario_b_runway_approach",
            "title": "Runway Approach",
            "channel": "land",
            "status": "PENDING_USER_VIDEO",
            "note": "Awaiting daytime cockpit clip.",
        },
        "scenario_c_nadir_aerial": {
            "scenario_id": "scenario_c_nadir_aerial",
            "title": "Nadir Aerial",
            "channel": "nav",
            "status": "PENDING_USER_VIDEO",
            "note": "Awaiting nadir clip.",
        },
        "scenario_d_thermal_flight": {
            "scenario_id": "scenario_d_thermal_flight",
            "title": "Thermal Flight",
            "channel": "thermal",
            "status": "PENDING_USER_VIDEO",
            "note": "Awaiting airborne thermal clip.",
        },
    }

    md_path, json_path = mod.write_truth_reports(
        tmp_path,
        manifest_path=tmp_path / "manifest.json",
        scenario_reports=scenario_reports,
        model_meta={"xvs_d2": {"sha256": "deadbeef"}},
    )
    assert md_path.is_file()
    assert json_path.is_file()

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    assert doc["schema"] == "xvs.field_video_truth_report/1"
    assert doc["summary"]["scenarios_evaluated"] == 1
    assert doc["summary"]["scenarios_pending_user_video"] == 3
    assert doc["summary"]["total_hmi_count"] == 0
    assert doc["summary"]["all_fail_closed_held"] is True

    md_text = md_path.read_text(encoding="utf-8")
    assert "NON-FLIGHT BENCH EVAL" in md_text
    assert "PENDING_USER_VIDEO" in md_text
    assert "UNKNOWN_UNCALIBRATED" in md_text
