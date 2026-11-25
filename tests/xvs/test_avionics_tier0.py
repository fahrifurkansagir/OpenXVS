"""Verification test suite for Batch B10 Tier-0 Standard Avionics Outputs (`V1_B10`..`V8_B10`).

Tests:
- V1_B10: Exact FAA GDL-90 ICD §2.2.3 Heartbeat CRC-16 (0x8BB3 / wire 0xB3 0x8B)
          and control-octet byte-stuffing (0x7D / 0x7E) round-trip;
- V2_B10: 24-bit signed semicircle lat/lon, 12-bit 25-ft altitude encoding,
          5-ft geometric altitude (0x0B), NIC/NACp mapping, and Traffic Alert Status
          bit (s=1/0) across WC1 and A1/A2 states;
- V3_B10: MAVLink v2 ADSB_VEHICLE (#246, CRC_EXTRA=184) binary pack/unpack exactness;
- V4_B10: Live localhost UDP loopback transmission and reception of GDL-90 + MAVLink
          frames with 0 dropped or corrupted frames;
- V5_B10: AST and runtime sentinel check proving `src/xvs/avionics/*` never imports
          `xvs.traffic` or reads ground-truth intruder position/range (`truth_id`,
          true range, or truth state);
- V6_B10: `tmp_path` byte-identity regeneration of `docs/xvs/reports/WP08_B10_tier0_report.md`
          and `verify_report_provenance` verification (0 unprovenanced numbers);
- V7_B10: Live WP-05 G1 `Track` + `RiskEstimate` (`WC1`) dual-object coupling (`B10-m2`)
          and 6-DoF WGS-84 ground-clutter suppression (`CLUTTER_HOLD` emits 0 traffic packets);
- V8_B10: WP-07 Horizon Integrity state gating (`VALID`, `DEGRADED_COAST`, `INVALID_ALERT`)
          across GDL-90 Heartbeat/Ownship/Traffic and MAVLink v2 `ADSB_VEHICLE` (#246).
"""

from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pytest

from xvs.avionics.bridge import (
    OwnshipNavState,
    PassiveTrackAvionicsBridge,
    PassiveTrackInput,
)
from xvs.avionics.gdl90 import (
    GDL90_ICD_HEARTBEAT_CRC_U16,
    GDL90_ICD_HEARTBEAT_CRC_WIRE,
    GDL90_ICD_HEARTBEAT_FRAMED,
    GDL90_ICD_HEARTBEAT_UNSTUFFED,
    GDL90_MSG_ID_HEARTBEAT,
    GDL90_MSG_ID_OWNSHIP_GEO_ALT,
    GDL90_MSG_ID_OWNSHIP_REPORT,
    GDL90_MSG_ID_TRAFFIC_REPORT,
    Gdl90TargetReport,
    Gdl90UdpEndpoint,
    compute_gdl90_crc,
    decode_altitude_25ft,
    decode_gdl90_frame,
    decode_gdl90_message,
    decode_lat_lon_semicircles,
    encode_altitude_25ft,
    encode_lat_lon_semicircles,
    encode_ownship_geo_altitude,
    encode_target_report,
    frame_gdl90_message,
    map_range_span_to_nic_nacp,
    stuff_gdl90_bytes,
    unstuff_gdl90_bytes,
)
from xvs.avionics.mavlink_adsb import (
    ADSB_FLAGS_VALID_COORDS,
    ADSB_VEHICLE_CRC_EXTRA,
    ADSB_VEHICLE_FRAME_LEN,
    ADSB_VEHICLE_MSG_ID,
    ADSB_VEHICLE_PAYLOAD_LEN,
    DEFAULT_PASSIVE_ADSB_FLAGS,
    MAVLINK_V2_STX,
    MavlinkAdsbUdpStreamer,
    MavlinkAdsbVehicle,
    pack_adsb_vehicle,
    unpack_adsb_vehicle,
)
from xvs.geo import rotations, wgs84
from xvs.report_provenance import verify_report_provenance
from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.ring import TrafficRing
from xvs.tracking.ground_rejection import ray_wgs84_intersect
from xvs.tracking.measurements import DetectionRecord
from xvs.tracking.risk import evaluate_track_risk
from xvs.tracking.tracker import MultiCameraTracker, Track, TrackStatus
from xvs.tracking.ttc import TTCEstimate

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "avionics" / "tier0_spec.json"
REPORT_PATH = REPO_ROOT / "docs" / "xvs" / "reports" / "WP08_B10_tier0_report.md"


def _normalize_report_for_byte_compare(text: str) -> str:
    """Mask dynamic pytest execution timing and rootdir lines in §4 Test output."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("rootdir:"):
            lines.append("rootdir: <NORMALIZED>")
            continue
        line = re.sub(
            r"^(=+\s*)(.*?)in \d+\.\d+s(?:\s*\([^)]*\))?(\s*=+)?$",
            lambda m: f"=== {m.group(2)}in <TIME>s ==="
            if m.group(1)
            else f"{m.group(2)}in <TIME>s",
            line,
        )
        lines.append(line)
    return "\n".join(lines)


def test_v1_b10_gdl90_icd_heartbeat_and_stuffing() -> None:
    """V1_B10: Exact FAA GDL-90 ICD §2.2.3 Heartbeat CRC-16 (0xB38B wire) and byte stuffing."""
    crc = compute_gdl90_crc(GDL90_ICD_HEARTBEAT_UNSTUFFED)
    assert crc == GDL90_ICD_HEARTBEAT_CRC_U16 == 0x8BB3
    wire_crc = bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    assert wire_crc == GDL90_ICD_HEARTBEAT_CRC_WIRE == bytes([0xB3, 0x8B])

    framed = frame_gdl90_message(GDL90_ICD_HEARTBEAT_UNSTUFFED)
    assert framed == GDL90_ICD_HEARTBEAT_FRAMED
    assert decode_gdl90_frame(framed) == GDL90_ICD_HEARTBEAT_UNSTUFFED

    hb_decoded = decode_gdl90_message(framed)
    assert isinstance(hb_decoded, dict)
    assert hb_decoded["msg_id"] == GDL90_MSG_ID_HEARTBEAT
    assert hb_decoded["gps_pos_valid"] is True
    assert hb_decoded["uat_initialized"] is True

    # Byte-stuffing round-trip with 0x7E and 0x7D control bytes
    raw_with_controls = bytes([0x14, 0x7E, 0x7D, 0x00, 0x7D, 0x7E] + [0x42] * 22)
    stuffed = stuff_gdl90_bytes(raw_with_controls)
    assert 0x7E not in stuffed
    assert b"\x7d\x5e" in stuffed
    assert b"\x7d\x5d" in stuffed
    assert unstuff_gdl90_bytes(stuffed) == raw_with_controls

    framed_ctrl = frame_gdl90_message(raw_with_controls)
    assert decode_gdl90_frame(framed_ctrl) == raw_with_controls

    # Corrupted CRC must raise ValueError
    tampered = bytearray(framed)
    tampered[2] ^= 0x01
    with pytest.raises(ValueError, match="CRC mismatch"):
        decode_gdl90_frame(bytes(tampered))


def test_v2_b10_gdl90_geodesy_altitude_and_alert_mapping() -> None:
    """V2_B10: 24-bit semicircle lat/lon, 25-ft altitude, and Traffic Alert Status bit mapping."""
    for angle in (0.0, 37.6188056, -122.3754167, 89.999, -89.999, 179.999, -179.999):
        b3 = encode_lat_lon_semicircles(angle)
        assert len(b3) == 3
        dec = decode_lat_lon_semicircles(b3)
        assert abs(dec - angle) <= 4.3e-5

    for alt_ft in (-1000.0, 0.0, 5000.0, 18500.0, 35000.0):
        code = encode_altitude_25ft(alt_ft)
        dec_alt = decode_altitude_25ft(code)
        assert dec_alt is not None
        assert abs(dec_alt - alt_ft) <= 12.5
    assert encode_altitude_25ft(None) == 0xFFF
    assert decode_altitude_25ft(0xFFF) is None

    # Ownship Geometric Altitude (0x0B) 5-ft encoding
    geo_frame = encode_ownship_geo_altitude(alt_ft=5020.0, vfom_m=3.0, vertical_warning=False)
    geo_dec = decode_gdl90_message(geo_frame)
    assert isinstance(geo_dec, dict)
    assert geo_dec["msg_id"] == GDL90_MSG_ID_OWNSHIP_GEO_ALT
    assert abs(geo_dec["alt_ft"] - 5020.0) <= 2.5
    assert geo_dec["vfom_m"] == 3

    # NIC/NACp passive range interval mapping
    assert map_range_span_to_nic_nacp(50.0) == (9, 9)
    assert map_range_span_to_nic_nacp(300.0) == (7, 7)
    assert map_range_span_to_nic_nacp(1500.0) == (5, 5)
    assert map_range_span_to_nic_nacp(50000.0) == (0, 0)

    # Alert status mapping across WC1 and A1/A2 states
    bridge = PassiveTrackAvionicsBridge(SPEC_PATH)
    own = OwnshipNavState.from_spec(bridge.spec)

    cases = [
        ({"track_id": "T1", "wc1_warning": True, "wc1_corrective": False}, 1, 7700),
        ({"track_id": "T2", "wc1_warning": False, "wc1_corrective": True}, 1, 7700),
        ({"track_id": "T3", "a1_warning": True, "a2_warning": False}, 1, 7700),
        ({"track_id": "T4", "a1_warning": False, "a2_warning": True}, 1, 7700),
        ({"track_id": "T5", "wc1_warning": False, "a1_warning": False, "a2_warning": False}, 0, 1200),
    ]
    for state_dict, expected_s_bit, expected_squawk in cases:
        base = {
            "t_s": 5.0,
            "az_rad": 0.1,
            "el_rad": -0.02,
            "R_min_m": 800.0,
            "R_max_m": 1200.0,
            "tau_mod_hat_s": 18.0,
            "class_id": "C2_SEP",
        }
        base.update(state_dict)
        gdl_rep, mav_msg = bridge.build_traffic_pair(own, base)
        wire = encode_target_report(gdl_rep)
        dec_rep = decode_gdl90_message(wire)
        assert isinstance(dec_rep, Gdl90TargetReport)
        assert dec_rep.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT
        assert dec_rep.alert_status == expected_s_bit
        assert mav_msg.squawk == expected_squawk


def test_v3_b10_mavlink_adsb_vehicle_pack_unpack() -> None:
    """V3_B10: MAVLink v2 ADSB_VEHICLE (#246, CRC_EXTRA=184) binary pack/unpack exactness."""
    msg = MavlinkAdsbVehicle(
        icao_address=0xF00001,
        lat_deg_e7=376188056,
        lon_deg_e7=-1223754167,
        altitude_mm=1524000,
        heading_cdeg=27450,
        hor_velocity_cms=6250,
        ver_velocity_cms=-320,
        flags=DEFAULT_PASSIVE_ADSB_FLAGS,
        squawk=7700,
        altitude_type=1,
        callsign="XVS0001",
        emitter_type=14,
        tslc=1,
        seq=42,
        sysid=1,
        compid=156,
    )
    frame = pack_adsb_vehicle(msg)
    assert len(frame) == ADSB_VEHICLE_FRAME_LEN == 50
    assert frame[0] == MAVLINK_V2_STX == 0xFD
    assert frame[1] == ADSB_VEHICLE_PAYLOAD_LEN == 38
    assert (frame[7] | (frame[8] << 8) | (frame[9] << 16)) == ADSB_VEHICLE_MSG_ID == 246

    unpacked = unpack_adsb_vehicle(frame)
    assert unpacked == msg

    # Verify CRC tampering is rejected
    bad_frame = bytearray(frame)
    bad_frame[15] ^= 0xFF
    with pytest.raises(ValueError, match="CRC mismatch"):
        unpack_adsb_vehicle(bytes(bad_frame))


def test_v4_b10_udp_loopback_streaming() -> None:
    """V4_B10: Live localhost UDP loopback transmission and reception with 0 dropped frames."""
    with (
        Gdl90UdpEndpoint(bind_host="127.0.0.1", bind_port=0) as gdl_rx,
        Gdl90UdpEndpoint(bind_host="127.0.0.1", bind_port=0) as gdl_tx,
        MavlinkAdsbUdpStreamer(bind_host="127.0.0.1", bind_port=0) as mav_rx,
        MavlinkAdsbUdpStreamer(bind_host="127.0.0.1", bind_port=0) as mav_tx,
    ):
        gdl_tx.set_target("127.0.0.1", gdl_rx.port)
        mav_tx.set_target("127.0.0.1", mav_rx.port)

        bridge = PassiveTrackAvionicsBridge(
            spec_path=SPEC_PATH,
            gdl90_endpoint=gdl_tx,
            mavlink_streamer=mav_tx,
        )
        own = OwnshipNavState.from_spec(bridge.spec)

        sent_gdl = 0
        sent_mav = 0
        rx_gdl = 0
        rx_mav = 0

        for step_idx in range(20):
            t_s = step_idx * 0.1
            trk = PassiveTrackInput(
                track_id="T0007",
                t_s=t_s,
                az_rad=0.05,
                el_rad=-0.01,
                az_rate_radps=0.0004,
                el_rate_radps=0.0,
                s_rad=0.008,
                s_dot_radps=0.0004,
                r_min_m=950.0,
                r_max_m=1450.0,
                tau_mod_hat_s=19.5,
                wc1_warning=(step_idx >= 10),
                wc1_corrective=(step_idx >= 5),
                class_id="C2_SEP",
            )
            step_out = bridge.step(t_s=t_s, ownship=own, tracks=[trk])
            sent_gdl += len(step_out.gdl90_frames)
            sent_mav += len(step_out.mavlink_frames)

            for gf in gdl_rx.recv_frames():
                dec = decode_gdl90_message(gf)
                assert dec is not None
                rx_gdl += 1

            for mf in mav_rx.recv_packets():
                m_dec = unpack_adsb_vehicle(mf)
                assert m_dec.callsign == "XVS0007"
                rx_mav += 1

        assert sent_gdl > 0 and rx_gdl == sent_gdl
        assert sent_mav == 20 and rx_mav == sent_mav


def test_v5_b10_strict_truth_isolation() -> None:
    """V5_B10: AST and runtime check proving `src/xvs/avionics/*` never touches ground truth."""
    avionics_dir = REPO_ROOT / "src" / "xvs" / "avionics"
    forbidden_names = {
        "truth_id",
        "true_range",
        "true_range_m",
        "range_true",
        "range_true_m",
        "p_int_ned_true",
        "tau_true_s",
        "omega_true_radps",
    }

    for py_path in sorted(avionics_dir.glob("*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "xvs.traffic" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                assert "xvs.traffic" not in (node.module or "")
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in forbidden_names

    class PoisonedTruthTrack:
        track_id = "T0042"
        t_s = 2.0
        az_rad = -0.15
        el_rad = 0.02
        az_rate_radps = 0.001
        el_rate_radps = -0.0002
        s_rad = 0.007
        s_dot_radps = 0.00035
        R_min_m = 1000.0
        R_max_m = 1600.0
        tau_mod_hat_s = 20.0
        wc1_warning = True
        wc1_corrective = True
        class_id = "C3_TWIN"
        tslc_s = 0

        @property
        def truth_id(self) -> str:
            raise AssertionError("Ground-truth truth_id accessed!")

        @property
        def true_range_m(self) -> float:
            raise AssertionError("Ground-truth true_range_m accessed!")

        @property
        def range_true_m(self) -> float:
            raise AssertionError("Ground-truth range_true_m accessed!")

    bridge = PassiveTrackAvionicsBridge(SPEC_PATH)
    own = OwnshipNavState.from_spec(bridge.spec)
    out = bridge.step(2.0, own, [PoisonedTruthTrack()], force_heartbeat=True)
    assert len(out.gdl90_traffic_reports) == 1
    assert out.gdl90_traffic_reports[0].callsign == "XVS0042"
    assert out.gdl90_traffic_reports[0].alert_status == 1


def test_v6_b10_report_regeneration_and_provenance(tmp_path: Path) -> None:
    """V6_B10: `tmp_path` byte-identity regeneration of WP08_B10_tier0_report.md and provenance check."""
    assert REPORT_PATH.is_file(), f"Committed report missing: {REPORT_PATH}"
    prov_path = REPORT_PATH.parent / f"{REPORT_PATH.name}.provenance.json"
    assert prov_path.is_file(), f"Committed provenance JSON missing: {prov_path}"

    assert verify_report_provenance(REPORT_PATH) == []

    tmp_report = tmp_path / "WP08_B10_tier0_report.md"
    script_path = REPO_ROOT / "scripts" / "xvs" / "avionics" / "report_tables_tier0_b10.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    proc = subprocess.run(
        [sys.executable, str(script_path), "--out", str(tmp_report)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, f"Report generator failed:\n{proc.stdout}\n{proc.stderr}"
    assert tmp_report.is_file()
    assert verify_report_provenance(tmp_report) == []

    gen_norm = _normalize_report_for_byte_compare(tmp_report.read_text(encoding="utf-8"))
    com_norm = _normalize_report_for_byte_compare(REPORT_PATH.read_text(encoding="utf-8"))
    assert gen_norm == com_norm, "Regenerated WP08_B10_tier0_report.md differs from committed report"


def test_v7_b10_live_wp05_g1_track_and_risk_coupling() -> None:
    """V7_B10: Live WP-05 G1 `Track` + `RiskEstimate` (`WC1`) coupling (`B10-m2`) and ground-clutter suppression."""
    bridge = PassiveTrackAvionicsBridge(SPEC_PATH)
    own = OwnshipNavState.from_spec(bridge.spec)

    # 1. Direct Track + RiskEstimate dual-object adapter (B10-m2) on non-zero-HMD WC1 threat
    r_threat = 1400.0
    hmd_threat = 350.0
    v_rel = 95.0
    sin_p = hmd_threat / r_threat
    cos_p = math.sqrt(1.0 - sin_p * sin_p)
    eta_t = (v_rel * cos_p) / r_threat
    omega_t = (v_rel * sin_p) / r_threat
    theta_t = 14.0 / r_threat  # C3_TWIN

    trk_threat = Track(
        track_id="TRK_0011",
        t_init=0.0,
        status=TrackStatus.CONFIRMED,
        x=np.array([0.08, omega_t, -0.01, 0.0, theta_t, theta_t * eta_t], dtype=np.float64),
        P=np.diag([1e-6, 1e-7, 1e-6, 1e-7, 1e-8, 1e-10]),
        history_t=[10.0],
        history_s_mom=[theta_t],
        ttc_estimate=TTCEstimate(
            valid=True,
            tau_s_hat_s=1.0 / eta_t,
            tau_s_min_s=1.0 / (1.05 * eta_t),
            tau_s_max_s=1.0 / (0.95 * eta_t),
            eta_hat_inv_s=eta_t,
            eta_min_inv_s=0.95 * eta_t,
            eta_max_inv_s=1.05 * eta_t,
            sigma_eta_inv_s=0.02 * eta_t,
            n_samples=15,
            span_s=3.0,
        ),
    )
    risk_threat = evaluate_track_risk(trk_threat, t_s=10.0, class_id="C3_TWIN", beta_wc=0.5)
    assert risk_threat.alert_a1_warning is False
    assert risk_threat.wc1_warning is True

    pt_in = PassiveTrackInput.from_track_and_risk(trk_threat, risk_threat, class_id="C3_TWIN")
    assert pt_in.track_id == "TRK_0011"
    assert pt_in.callsign == "XVS0011"
    assert pt_in.alert_active is True
    assert pt_in.alert_rule_source == "WC1"
    assert pt_in.r_min_m is not None and pt_in.r_max_m is not None
    assert pt_in.r_min_m < r_threat < pt_in.r_max_m

    # 2. Closed-loop MultiCameraTracker (g1.json) + stationary WGS-84 ground point suppression
    g1_cfg = json.loads((REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json").read_text(encoding="utf-8"))
    ring = TrafficRing()
    cam0 = ring.cameras[0]
    tracker = MultiCameraTracker(config=g1_cfg, ring=ring)

    st0 = t1_cruise_state(0.0)
    r_en0 = wgs84.r_ecef_ned(st0.lat_deg, st0.lon_deg)
    poses0 = ring.compute_poses(
        st0.lat_deg, st0.lon_deg, st0.h_m, st0.yaw_rad, st0.pitch_rad, st0.roll_rad
    )
    p_cam0 = np.asarray(poses0[cam0.id]["position"], dtype=np.float64)
    u_ned_gnd = np.array(
        [math.cos(-0.14) * math.cos(0.08), math.cos(-0.14) * math.sin(0.08), -math.sin(-0.14)],
        dtype=np.float64,
    )
    _, g_pt_ecef = ray_wgs84_intersect(p_cam0, r_en0 @ u_ned_gnd)
    assert g_pt_ecef is not None

    for k in range(12):
        t_s = round(k * 0.1, 3)
        st_k = t1_cruise_state(t_s)
        r_en_k = wgs84.r_ecef_ned(st_k.lat_deg, st_k.lon_deg)
        r_nb_k = rotations.euler_to_r(st_k.yaw_rad, st_k.pitch_rad, st_k.roll_rad)
        poses_k = ring.compute_poses(
            st_k.lat_deg, st_k.lon_deg, st_k.h_m, st_k.yaw_rad, st_k.pitch_rad, st_k.roll_rad
        )
        p_cam_k = np.asarray(poses_k[cam0.id]["position"], dtype=np.float64)
        r_ec_k = np.asarray(poses_k[cam0.id]["r_ecef_cam"], dtype=np.float64)

        s_air = 18.0 * math.exp(t_s / 14.0)
        det_air = DetectionRecord(
            t=t_s,
            camera=cam0.id,
            u=1280.0 + 0.1 * k,
            v=640.0,
            s_px=s_air,
            score=35.0,
            is_false_alarm=False,
            truth_id=None,
            s_mom_px=s_air,
        )

        v_cam_gnd = r_ec_k.T @ (g_pt_ecef - p_cam_k)
        u_gnd = float(ring.camera_model.fx * (v_cam_gnd[0] / v_cam_gnd[2]) + ring.camera_model.cx)
        v_gnd = float(ring.camera_model.fy * (v_cam_gnd[1] / v_cam_gnd[2]) + ring.camera_model.cy)
        det_gnd = DetectionRecord(
            t=t_s,
            camera=cam0.id,
            u=u_gnd,
            v=v_gnd,
            s_px=2.5,
            score=20.0,
            is_false_alarm=True,
            truth_id=None,
            s_mom_px=2.5,
        )

        conf = tracker.step(
            t_s=t_s,
            raw_detections=[det_air, det_gnd],
            r_ned_body_ins=r_nb_k,
            p_own_ecef=st_k.p_ecef_m,
            r_ecef_ned=r_en_k,
        )
        risk_map = {
            trk.track_id: evaluate_track_risk(trk, t_s=t_s, class_id="C2_SEP", beta_wc=0.5)
            for trk in conf
        }
        step_out = bridge.step_from_tracker(
            ownship=own,
            confirmed_tracks=tracker.tracks,
            risk_by_track_id=risk_map,
            integrity_state="VALID",
            t_s=t_s,
            class_by_track_id={trk.track_id: "C2_SEP" for trk in conf},
        )
        # Ensure stationary ground clutter track (TRK_0002 -> XVS0002) never emits traffic
        for trf_rep in step_out.gdl90_traffic_reports:
            assert trf_rep.callsign == "XVS0001"

    assert tracker.ground_clutter_rejected_count == 1
    assert any(t.status == TrackStatus.CLUTTER_HOLD for t in tracker.tracks)
    assert len(step_out.gdl90_traffic_reports) == 1
    assert step_out.gdl90_traffic_reports[0].alert_status == 1
    assert step_out.mavlink_messages[0].squawk == 7700


def test_v8_b10_wp07_integrity_degraded_and_invalid_gating() -> None:
    """V8_B10: WP-07 Horizon Integrity state gating (`VALID`, `DEGRADED_COAST`, `INVALID_ALERT`) over UDP."""
    with (
        Gdl90UdpEndpoint(bind_host="127.0.0.1", bind_port=0) as gdl_rx,
        Gdl90UdpEndpoint(bind_host="127.0.0.1", bind_port=0) as gdl_tx,
        MavlinkAdsbUdpStreamer(bind_host="127.0.0.1", bind_port=0) as mav_rx,
        MavlinkAdsbUdpStreamer(bind_host="127.0.0.1", bind_port=0) as mav_tx,
    ):
        gdl_tx.set_target("127.0.0.1", gdl_rx.port)
        mav_tx.set_target("127.0.0.1", mav_rx.port)

        bridge = PassiveTrackAvionicsBridge(
            spec_path=SPEC_PATH,
            gdl90_endpoint=gdl_tx,
            mavlink_streamer=mav_tx,
        )
        own = OwnshipNavState.from_spec(bridge.spec)

        trk = Track(
            track_id="TRK_0001",
            t_init=0.0,
            status=TrackStatus.CONFIRMED,
            x=np.array([0.02, 0.0005, -0.005, 0.0, 0.008, 0.0005], dtype=np.float64),
            P=np.diag([1e-6, 1e-7, 1e-6, 1e-7, 1e-8, 1e-10]),
            history_t=[5.0],
            history_s_mom=[0.008],
            ttc_estimate=TTCEstimate(
                valid=True,
                tau_s_hat_s=15.0,
                tau_s_min_s=14.0,
                tau_s_max_s=16.0,
                eta_hat_inv_s=1.0 / 15.0,
                eta_min_inv_s=1.0 / 16.0,
                eta_max_inv_s=1.0 / 14.0,
                sigma_eta_inv_s=0.002,
                n_samples=15,
                span_s=3.0,
            ),
        )
        risk = evaluate_track_risk(trk, t_s=5.0, class_id="C2_SEP", beta_wc=0.5)
        assert risk.wc1_warning is True

        # 1. VALID state
        bridge.step_from_tracker(
            ownship=own,
            confirmed_tracks=[trk],
            risk_by_track_id={"TRK_0001": risk},
            integrity_state="VALID",
            t_s=5.0,
            class_by_track_id={"TRK_0001": "C2_SEP"},
            force_heartbeat=True,
        )
        gdl_valid = [decode_gdl90_message(f) for f in gdl_rx.recv_frames()]
        mav_valid = [unpack_adsb_vehicle(p) for p in mav_rx.recv_packets()]

        hb_v = next(m for m in gdl_valid if isinstance(m, dict) and m["msg_id"] == GDL90_MSG_ID_HEARTBEAT)
        geo_v = next(m for m in gdl_valid if isinstance(m, dict) and m["msg_id"] == GDL90_MSG_ID_OWNSHIP_GEO_ALT)
        own_v = next(m for m in gdl_valid if isinstance(m, Gdl90TargetReport) and m.msg_id == GDL90_MSG_ID_OWNSHIP_REPORT)
        trf_v = next(m for m in gdl_valid if isinstance(m, Gdl90TargetReport) and m.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT)
        assert hb_v["gps_pos_valid"] is True
        assert hb_v["maint_reqd"] is False
        assert geo_v["vertical_warning"] is False
        assert own_v.nic == 10 and own_v.nacp == 10
        assert trf_v.nic == 6 and trf_v.nacp == 6
        assert trf_v.alert_status == 1
        assert (mav_valid[0].flags & ADSB_FLAGS_VALID_COORDS) != 0
        assert mav_valid[0].squawk == 7700

        # 2. DEGRADED_COAST state: NIC/NACp capped at <= 4, maint_reqd=True, VALID_COORDS kept
        bridge.step_from_tracker(
            ownship=own,
            confirmed_tracks=[trk],
            risk_by_track_id={"TRK_0001": risk},
            integrity_state="DEGRADED_COAST",
            t_s=6.0,
            class_by_track_id={"TRK_0001": "C2_SEP"},
            force_heartbeat=True,
        )
        gdl_deg = [decode_gdl90_message(f) for f in gdl_rx.recv_frames()]
        mav_deg = [unpack_adsb_vehicle(p) for p in mav_rx.recv_packets()]

        hb_d = next(m for m in gdl_deg if isinstance(m, dict) and m["msg_id"] == GDL90_MSG_ID_HEARTBEAT)
        geo_d = next(m for m in gdl_deg if isinstance(m, dict) and m["msg_id"] == GDL90_MSG_ID_OWNSHIP_GEO_ALT)
        own_d = next(m for m in gdl_deg if isinstance(m, Gdl90TargetReport) and m.msg_id == GDL90_MSG_ID_OWNSHIP_REPORT)
        trf_d = next(m for m in gdl_deg if isinstance(m, Gdl90TargetReport) and m.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT)
        assert hb_d["gps_pos_valid"] is True
        assert hb_d["maint_reqd"] is True
        assert geo_d["vertical_warning"] is False
        assert own_d.nic == 4 and own_d.nacp == 4
        assert trf_d.nic == 4 and trf_d.nacp == 4
        assert trf_d.alert_status == 1
        assert (mav_deg[0].flags & ADSB_FLAGS_VALID_COORDS) != 0
        assert mav_deg[0].squawk == 7700

        # 3. INVALID_ALERT state: NIC=0, NACp=0, maint_reqd=True, gps_pos_valid=False, VALID_COORDS cleared
        bridge.step_from_tracker(
            ownship=own,
            confirmed_tracks=[trk],
            risk_by_track_id={"TRK_0001": risk},
            integrity_state="INVALID_ALERT",
            t_s=7.0,
            class_by_track_id={"TRK_0001": "C2_SEP"},
            force_heartbeat=True,
        )
        gdl_inv = [decode_gdl90_message(f) for f in gdl_rx.recv_frames()]
        mav_inv = [unpack_adsb_vehicle(p) for p in mav_rx.recv_packets()]

        hb_i = next(m for m in gdl_inv if isinstance(m, dict) and m["msg_id"] == GDL90_MSG_ID_HEARTBEAT)
        geo_i = next(m for m in gdl_inv if isinstance(m, dict) and m["msg_id"] == GDL90_MSG_ID_OWNSHIP_GEO_ALT)
        own_i = next(m for m in gdl_inv if isinstance(m, Gdl90TargetReport) and m.msg_id == GDL90_MSG_ID_OWNSHIP_REPORT)
        trf_i = next(m for m in gdl_inv if isinstance(m, Gdl90TargetReport) and m.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT)
        assert hb_i["gps_pos_valid"] is False
        assert hb_i["maint_reqd"] is True
        assert geo_i["vertical_warning"] is True
        assert own_i.nic == 0 and own_i.nacp == 0
        assert trf_i.nic == 0 and trf_i.nacp == 0
        assert trf_i.alert_status == 1
        assert (mav_inv[0].flags & ADSB_FLAGS_VALID_COORDS) == 0
        assert mav_inv[0].squawk == 7700

        with pytest.raises(ValueError, match="Unsupported integrity_state"):
            bridge.step_from_tracker(
                ownship=own,
                confirmed_tracks=[trk],
                risk_by_track_id={"TRK_0001": risk},
                integrity_state="BOGUS_STATE",
            )

