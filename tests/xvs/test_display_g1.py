"""Unit and Integration Tests for WP-08 Gate G1: Live Track/Risk Symbology, Integrity HUD & Closed-Loop SITL Avoidance (Batch B12).

Verifies V1_G1..V7_G1 (`docs/xvs/WP08_DISPLAY.md` §8 & `docs/xvs/batches/BATCH_B12_WP08_G1_ARGON.md`):
  - V1_G1: Conformal passive track reticle projection accuracy (`<= 0.05 px`), 4-tier DO-365B /
           DO-317C symbology, 5.0 s velocity leader, 3-sigma covariance halo, and WCV belief duality.
  - V2_G1: WP-07 Horizon Integrity HUD coupling (`+/- PL_0.01` band `<= 0.05 px`, `F_isolated` badge,
           `DEGRADED_COAST` PL widening, and 1-frame `INVALID_ALERT` blanking with 0 HUD HMI frames).
  - V3_G1: Closed-loop MAVLink `#246` -> ArduPilot SITL horizontal avoidance NMAC resolution
           (`G1 WC1 >= 85%` and strictly outperforming `G0 A1`).
  - V4_G1: `INVALID_ALERT` (`VALID_COORDS == 0`, `NIC == 0`) inhibits autonomous SITL turns (`0` turns).
  - V5_G1: Strict No-Truth enforcement via static AST inspection and runtime `PoisonedTruthTrack` sentinel.
  - V6_G1: Single-process (`workers = 1`) byte-identical JSON and PNG determinism.
  - V7_G1: Zero-typing report provenance and tmp-path regeneration for `WP08_G1_report.md`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.avionics.bridge import PassiveTrackAvionicsBridge
from xvs.avionics.sitl_closed_loop import (
    SitlAvoidanceConfig,
    build_g3a_collision_threat_scenarios,
    simulate_single_encounter_closed_loop,
)
from xvs.display.symbology import (
    compute_conformal_horizon_with_pl_band,
    compute_rect_intersection_area,
    draw_g1_symbology,
)
from xvs.display.viewport import (
    compute_viewport_pose_from_estimated_attitude,
    get_viewport_camera,
)
from xvs.geo import rotations
from xvs.report_provenance import verify_report_provenance

sys.path.insert(0, str(REPO_ROOT / "scripts" / "xvs" / "display"))
from run_display_g1 import (  # noqa: E402
    PoisonedTruthTrack,
    evaluate_v1_conformal_traffic_reticle,
    evaluate_v2_integrity_hud_coupling,
    evaluate_v5_truth_isolation,
    evaluate_v6_determinism,
)

G1_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g1.json"
G1_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g1_spec.json"
G1_RESULTS_PATH = REPO_ROOT / "output" / "xvs" / "wp08" / "g1" / "results.json"
G1_MANIFEST_PATH = REPO_ROOT / "output" / "xvs" / "wp08" / "g1" / "run_manifest.json"
G1_REPORT_PATH = REPO_ROOT / "docs" / "xvs" / "reports" / "WP08_G1_report.md"


def test_v1_g1_conformal_traffic_reticle_projection_accuracy() -> None:
    """V1_G1: Conformal passive track reticle projection error <= 0.05 px and exact WCV belief duality."""
    cfg = json.loads(G1_CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads(G1_SPEC_PATH.read_text(encoding="utf-8"))

    v1 = evaluate_v1_conformal_traffic_reticle(cfg)
    assert v1["samples_count"] == spec["n_reticle_test_samples"]
    assert v1["max_error_px"] <= spec["tol_reticle_px"]
    assert v1["max_continuous_error_px"] <= spec["tol_reticle_px"]
    assert v1["max_panorama_error_px"] <= spec["tol_reticle_px"]
    assert v1["max_leader_endpoint_error_px"] <= spec["tol_reticle_px"]
    assert v1["wcv_duality_mismatches_count"] == 0
    assert v1["tier_classification_exact"] is True

    # Verify rendering on a live viewport canvas produces distinct pixel modifications for all 4 tiers
    view_cam = get_viewport_camera(cfg)
    r_ned_b = rotations.euler_to_r(0.0, 0.0, 0.0)
    vp_pose = compute_viewport_pose_from_estimated_attitude(
        ownship_lat_deg=37.30,
        ownship_lon_deg=-121.00,
        ownship_h_m=1500.0,
        r_ned_body_est=r_ned_b,
        head_euler_rad=(0.0, 0.0, 0.0),
        config=cfg,
    )
    blank = np.zeros((view_cam.height, view_cam.width, 3), dtype=np.uint8)
    tracks = [
        {"track_id": "T1", "az": -0.25, "el": 0.02, "s_dot": 0.0, "wc1_warning": False, "wc1_corrective": False, "hmd_min_m": 800.0, "hmd_max_m": 1200.0, "tau_mod_min_s": 85.0, "tau_mod_max_s": 110.0},
        {"track_id": "T2", "az": -0.08, "el": 0.02, "s_dot": 0.0002, "wc1_warning": False, "wc1_corrective": False, "hmd_min_m": 300.0, "hmd_max_m": 1200.0, "tau_mod_min_s": 25.0, "tau_mod_max_s": 60.0},
        {"track_id": "T3", "az": 0.08, "el": 0.02, "s_dot": 0.0003, "wc1_warning": False, "wc1_corrective": True, "hmd_min_m": 150.0, "hmd_max_m": 450.0, "tau_mod_min_s": 18.0, "tau_mod_max_s": 35.0},
        {"track_id": "T4", "az": 0.25, "el": 0.02, "s_dot": 0.0005, "wc1_warning": True, "wc1_corrective": True, "hmd_min_m": 50.0, "hmd_max_m": 200.0, "tau_mod_min_s": 8.0, "tau_mod_max_s": 18.0},
    ]
    out_img, telemetry = draw_g1_symbology(
        img=blank,
        aircraft_pose_est={"lat_deg": 37.30, "lon_deg": -121.00, "h_m": 1500.0, "yaw_rad": 0.0, "pitch_rad": 0.0, "roll_rad": 0.0},
        v_ned=np.array([100.0, 0.0, 0.0], dtype=np.float64),
        p_ecef_view=vp_pose["p_ecef_view"],
        r_ecef_view_est=vp_pose["r_ecef_view_est"],
        r_ned_body_est=r_ned_b,
        head_euler_rad=(0.0, 0.0, 0.0),
        view_cam=view_cam,
        tracks_and_risks=[(t, t) for t in tracks],
        integrity_state="VALID",
        pl_deg=0.25,
    )
    assert len(telemetry["projected_tracks"]) == 4
    assert [r["tier_name"] for r in telemetry["projected_tracks"]] == [
        "NONE",
        "PREVENTIVE",
        "CORRECTIVE",
        "WARNING",
    ]
    assert int(np.sum(np.abs(out_img.astype(np.int32) - blank.astype(np.int32)))) > 1000

    # B12-F1: Verify zero pairwise bounding-box overlap between HUD callouts and target reticles,
    # including a crowded cluster of 4 closely spaced targets
    crowded_tracks = [
        {"track_id": "C1", "az": 0.10, "el": 0.00, "s_dot": 0.0, "wc1_warning": False, "wc1_corrective": False, "hmd_min_m": 800.0, "hmd_max_m": 1200.0, "tau_mod_min_s": 85.0, "tau_mod_max_s": 110.0},
        {"track_id": "C2", "az": 0.14, "el": 0.01, "s_dot": 0.0002, "wc1_warning": False, "wc1_corrective": False, "hmd_min_m": 300.0, "hmd_max_m": 1200.0, "tau_mod_min_s": 25.0, "tau_mod_max_s": 60.0},
        {"track_id": "C3", "az": 0.18, "el": -0.01, "s_dot": 0.0003, "wc1_warning": False, "wc1_corrective": True, "hmd_min_m": 150.0, "hmd_max_m": 450.0, "tau_mod_min_s": 18.0, "tau_mod_max_s": 35.0},
        {"track_id": "C4", "az": 0.22, "el": 0.01, "s_dot": 0.0005, "wc1_warning": True, "wc1_corrective": True, "hmd_min_m": 50.0, "hmd_max_m": 200.0, "tau_mod_min_s": 8.0, "tau_mod_max_s": 18.0},
    ]
    _, tel_crowded = draw_g1_symbology(
        img=blank,
        aircraft_pose_est={"lat_deg": 37.30, "lon_deg": -121.00, "h_m": 1500.0, "yaw_rad": 0.0, "pitch_rad": 0.0, "roll_rad": 0.0},
        v_ned=np.array([100.0, 0.0, 0.0], dtype=np.float64),
        p_ecef_view=vp_pose["p_ecef_view"],
        r_ecef_view_est=vp_pose["r_ecef_view_est"],
        r_ned_body_est=r_ned_b,
        head_euler_rad=(0.0, 0.0, 0.0),
        view_cam=view_cam,
        tracks_and_risks=[(t, t) for t in crowded_tracks],
        integrity_state="VALID",
        pl_deg=0.25,
    )
    for tel_chk in (telemetry, tel_crowded):
        pts_chk = tel_chk["projected_tracks"]
        for i in range(len(pts_chk)):
            lbl_i = tuple(pts_chk[i]["label_bbox"])
            ret_i = tuple(pts_chk[i]["reticle_bbox"])
            assert compute_rect_intersection_area(lbl_i, ret_i) == 0.0
            for j in range(i + 1, len(pts_chk)):
                lbl_j = tuple(pts_chk[j]["label_bbox"])
                ret_j = tuple(pts_chk[j]["reticle_bbox"])
                assert compute_rect_intersection_area(lbl_i, lbl_j) == 0.0
                assert compute_rect_intersection_area(lbl_i, ret_j) == 0.0
                assert compute_rect_intersection_area(ret_i, lbl_j) == 0.0


def test_v2_g1_conformal_horizon_pl_band_and_invalid_blanking() -> None:
    """V2_G1: Conformal horizon +/- PL_0.01 band error <= 0.05 px and 1-frame INVALID_ALERT blanking."""
    cfg = json.loads(G1_CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads(G1_SPEC_PATH.read_text(encoding="utf-8"))

    v2 = evaluate_v2_integrity_hud_coupling(cfg)
    assert v2["max_pl_band_error_px"] <= spec["tol_pl_band_px"]
    assert v2["coasting_pl_monotonic"] is True
    assert v2["ttd_display_frames"] <= spec["max_ttd_display_frames"]
    assert v2["hud_hmi_frames"] == spec["max_hud_hmi_frames"]
    assert v2["blanked_within_one_frame"] is True

    # Verify INVALID_ALERT blanks the conformal horizon while preserving relative traffic cues
    view_cam = get_viewport_camera(cfg)
    r_ned_b = rotations.euler_to_r(0.0, 0.0, 0.0)
    vp_pose = compute_viewport_pose_from_estimated_attitude(
        ownship_lat_deg=37.30,
        ownship_lon_deg=-121.00,
        ownship_h_m=1500.0,
        r_ned_body_est=r_ned_b,
        head_euler_rad=(0.0, 0.0, 0.0),
        config=cfg,
    )
    hb_inv = compute_conformal_horizon_with_pl_band(
        p_ecef_view=vp_pose["p_ecef_view"],
        r_ecef_view_est=vp_pose["r_ecef_view_est"],
        view_cam=view_cam,
        pl_deg=0.50,
        integrity_state="INVALID_ALERT",
    )
    assert hb_inv["horizon_visible"] is False
    assert hb_inv["horizon_suppressed"] is True
    assert hb_inv["horizon_uv"].shape[0] == 0
    assert "ATTITUDE INTEGRITY INVALID" in hb_inv["alert_banner"]

    blank = np.zeros((view_cam.height, view_cam.width, 3), dtype=np.uint8)
    trk_inv = {"track_id": "T0001", "az": 0.05, "el": 0.01, "wc1_warning": True}
    _, tel_inv = draw_g1_symbology(
        img=blank,
        aircraft_pose_est={"lat_deg": 37.30, "lon_deg": -121.00, "h_m": 1500.0, "yaw_rad": 0.0, "pitch_rad": 0.0, "roll_rad": 0.0},
        v_ned=np.array([100.0, 0.0, 0.0], dtype=np.float64),
        p_ecef_view=vp_pose["p_ecef_view"],
        r_ecef_view_est=vp_pose["r_ecef_view_est"],
        r_ned_body_est=r_ned_b,
        head_euler_rad=(0.0, 0.0, 0.0),
        view_cam=view_cam,
        tracks_and_risks=[(trk_inv, trk_inv)],
        integrity_state="INVALID_ALERT",
    )
    assert tel_inv["horizon_suppressed"] is True
    assert len(tel_inv["projected_tracks"]) == 1
    assert tel_inv["projected_tracks"][0]["stabilization_mode"] == "BODY_STABILIZED"


def test_v3_g1_closed_loop_sitl_avoidance_nmac_resolution() -> None:
    """V3_G1: Closed-loop MAVLink #246 -> ArduPilot SITL horizontal avoidance resolves >= 85% NMACs and beats G0 A1."""
    spec = json.loads(G1_SPEC_PATH.read_text(encoding="utf-8"))

    # Live 2-encounter verification
    cfg = SitlAvoidanceConfig.from_display_config(G1_CONFIG_PATH)
    scenarios = build_g3a_collision_threat_scenarios(count=2, max_open_loop_hmd_m=cfg.open_loop_cpa_max_m)
    bridge = PassiveTrackAvionicsBridge()
    for scen in scenarios:
        res_open = simulate_single_encounter_closed_loop(
            scenario=scen, mode="open_loop", bridge=bridge, config=cfg
        )
        res_wc1 = simulate_single_encounter_closed_loop(
            scenario=scen, mode="closed_loop_g1_wc1_b050", bridge=bridge, config=cfg
        )
        assert res_open["achieved_min_hmd_m"] < cfg.nmac_threshold_m
        assert res_wc1["achieved_min_hmd_m"] > res_open["achieved_min_hmd_m"]
        assert res_wc1["nmac_resolved"] is True
        assert res_wc1["final_cross_track_m"] < res_wc1["max_cross_track_m"]

    # Full 50-encounter benchmark check from results.json if present
    if G1_RESULTS_PATH.is_file():
        res = json.loads(G1_RESULTS_PATH.read_text(encoding="utf-8"))
        v3 = res["v3_closed_loop_sitl_avoidance"]
        m_open = v3["modes"]["open_loop"]
        m_g0 = v3["modes"]["closed_loop_g0_a1"]
        m_b050 = v3["modes"]["closed_loop_g1_wc1_b050"]
        m_b025 = v3["modes"]["closed_loop_g1_wc1_b025"]

        assert m_open["nmac_resolved_count"] == 0
        assert m_b050["nmac_resolution_rate_alerted_pct"] >= spec["min_alerted_nmac_resolution_pct"]
        assert m_b025["nmac_resolution_rate_alerted_pct"] >= spec["min_alerted_nmac_resolution_pct"]
        assert m_b050["nmac_resolved_count"] > m_g0["nmac_resolved_count"]
        assert m_b050["well_clear_recovered_count"] > m_g0["well_clear_recovered_count"]
        assert m_b050["final_cross_track_median_m"] < m_b050["cross_track_median_m"]
        assert res["stills"]["max_callout_overlap_px2"] == 0.0


def test_v4_g1_invalid_alert_inhibits_sitl_maneuvers() -> None:
    """V4_G1: INVALID_ALERT MAVLink #246 packets (VALID_COORDS == 0, NIC == 0) inhibit SITL turns."""
    cfg = SitlAvoidanceConfig.from_display_config(G1_CONFIG_PATH)
    scenarios = build_g3a_collision_threat_scenarios(count=2, max_open_loop_hmd_m=cfg.open_loop_cpa_max_m)
    bridge = PassiveTrackAvionicsBridge()

    for scen in scenarios:
        res_inv = simulate_single_encounter_closed_loop(
            scenario=scen,
            mode="closed_loop_g1_wc1_b050",
            bridge=bridge,
            config=cfg,
            integrity_state="INVALID_ALERT",
        )
        assert res_inv["induced_turns_count"] == 0
        assert res_inv["invalid_coord_inhibits_count"] > 0

    if G1_RESULTS_PATH.is_file():
        res = json.loads(G1_RESULTS_PATH.read_text(encoding="utf-8"))
        v4 = res["v4_invalid_alert_inhibition"]
        assert v4["induced_turns_count"] == 0
        assert v4["invalid_coord_inhibits_count"] > 0


def test_v5_g1_strict_no_truth_enforcement() -> None:
    """V5_G1: Static AST and runtime sentinel-trap audit on symbology.py and sitl_closed_loop.py."""
    cfg = json.loads(G1_CONFIG_PATH.read_text(encoding="utf-8"))
    v5 = evaluate_v5_truth_isolation(cfg)
    assert v5["forbidden_imports_count"] == 0, f"Forbidden imports: {v5['violations']}"
    assert v5["forbidden_attribute_refs_count"] == 0, f"Forbidden attrs: {v5['violations']}"
    assert v5["runtime_sentinel_violations"] == 0
    assert v5["isolated"] is True

    poison = PoisonedTruthTrack()
    with pytest.raises(RuntimeError, match="V5_G1 VIOLATION"):
        _ = poison.true_range_m


def test_v6_g1_determinism_json_and_png() -> None:
    """V6_G1: Repeated single-process runs produce byte-identical JSON and PNG hashes."""
    cfg = json.loads(G1_CONFIG_PATH.read_text(encoding="utf-8"))
    sitl_cfg = SitlAvoidanceConfig.from_display_config(G1_CONFIG_PATH)
    v6 = evaluate_v6_determinism(cfg, sitl_cfg)
    assert v6["workers"] == 1
    assert v6["json_byte_identical"] is True
    assert v6["png_hashes_identical"] is True

    if G1_MANIFEST_PATH.is_file():
        manifest = json.loads(G1_MANIFEST_PATH.read_text(encoding="utf-8"))
        assert manifest["workers"] == 1
        assert manifest["code_hashes_match"] is True


def test_v7_g1_report_provenance_and_regeneration(tmp_path: Path) -> None:
    """V7_G1: WP08_G1_report.md passes provenance verification and regenerates cleanly into tmp_path."""
    if not G1_RESULTS_PATH.is_file() or not G1_REPORT_PATH.is_file():
        pytest.skip("G1 results or report not yet generated")

    unprov = verify_report_provenance(G1_REPORT_PATH)
    assert unprov == [], f"Committed WP08_G1_report.md has unprovenanced tokens: {unprov}"

    tmp_report = tmp_path / "WP08_G1_report.md"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "xvs" / "display" / "report_tables_wp08_g1.py"),
        "--json",
        str(G1_RESULTS_PATH),
        "--out",
        str(tmp_report),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Report generator failed:\n{proc.stdout}\n{proc.stderr}"
    assert tmp_report.is_file()
    assert tmp_report.with_suffix(".md.provenance.json").is_file()

    unprov_tmp = verify_report_provenance(tmp_report)
    assert unprov_tmp == [], f"Regenerated WP08_G1_report.md has unprovenanced tokens: {unprov_tmp}"
