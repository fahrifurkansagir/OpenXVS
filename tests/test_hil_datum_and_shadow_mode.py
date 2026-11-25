"""Wave 2 Track 2 Tests: WGS-84 HAE <-> MSL Datum Alignment Guard & Passive Shadow-Mode Logger.

Closes `REALWORLD_DEPLOYABILITY_AUDIT_V5.md` Failure Mode 8:
  1. `DatumAlignmentGuard`: exact WGS-84 HAE <-> EGM96/GEOID18 MSL conversion (`KSFO 28L`),
     fail-closed rejection on `NaN`/`Inf`/`|N_geoid_m| > 110 m`, pre-switch vertical innovation
     gate (`<= 3.0 m` default), and explicit `DATUM_MISMATCH_INHIBITED` /
     `PRE_SWITCH_VERT_INNOV_EXCEEDED` inhibition before `EKF3 42007` visual handoff.
  2. `ShadowModeTelemetryLogger`: hardware-mirror listen-only (`tx_inhibited=True`,
     `actuator_authority=False`, `outbound_mavlink_frames_emitted=0`) SHA-256 hash-chained
     NDJSON flight-recorder for Phase-2 rooftop/car-roof & Phase-3 TX-pin-removed university
     shadow-mode trials, plus live UDP shadow-tap integration under `NON_FLIGHT_SIM_ONLY=1`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import socket
import time

import pytest

from xvs.v2.studio_bridge.hil_bridge import (
    DATUM_REASON_MISMATCH_INHIBITED,
    DATUM_REASON_OK,
    DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED,
    DEFAULT_MAX_PRE_SWITCH_VERT_INNOV_M,
    KSFO_28L_GEOID18_UNDULATION_M,
    KSFO_28L_THR_NAVD88_MSL_M,
    KSFO_28L_THR_WGS84_HAE_M,
    MAX_ABS_GEOID_UNDULATION_M,
    NON_FLIGHT_WATERMARK,
    SHADOW_GENESIS_HASH,
    SHADOW_MODE_LOG_SCHEMA,
    DatumAlignmentGuard,
    NonFlightSimOnlyError,
    ShadowModeTelemetryLogger,
    ShadowModeTxInhibitedError,
    StudioHilBridgeManager,
    msl_to_wgs84_hae,
    validate_geoid_undulation_m,
    verify_shadow_log_chain,
    wgs84_hae_to_msl,
)
from xvs.v2.xvs_mavlink.pod_daa_codec import (
    MSGID_ATTITUDE,
    MSGID_COMMAND_LONG,
    MSGID_GLOBAL_POSITION_INT,
    MSGID_HEARTBEAT,
    MSGID_SET_POSITION_TARGET_LOCAL_NED,
    MavAttitude,
    MavCommandLong,
    MavGlobalPositionInt,
    MavHeartbeat,
    decode_frame,
    encode_frame,
)
from xvs.v2.xvs_rt_bridge.arduplane_bridge import (
    ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
    EKF3_SOURCE_GNSS,
    EKF3_SOURCE_VISION_OPTICAL,
    Ekf3SourceSwitchLatch,
)


def _alloc_free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


# ---- 1. WGS-84 HAE <-> EGM96/GEOID18 MSL Geoid Math & Fail-Closed Bounds ----


def test_ksfo_28l_geoid_conversion_matches_3dep_manifest_and_roundtrips() -> None:
    """Verify exact geodetic `h = H + N` / `H = h - N` against KSFO 28L 3DEP GEOID18 manifest."""
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "livedeck"
        / "public"
        / "assets"
        / "ksfo_3dep_geoid18_v1.manifest.json"
    )
    if manifest_path.is_file():
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert float(doc["ksfo_28l_thr_navd88_m"]) == pytest.approx(KSFO_28L_THR_NAVD88_MSL_M, abs=1e-9)
        assert float(doc["ksfo_28l_geoid_undulation_m"]) == pytest.approx(KSFO_28L_GEOID18_UNDULATION_M, abs=1e-9)
        assert float(doc["ksfo_28l_thr_wgs84_h_m"]) == pytest.approx(KSFO_28L_THR_WGS84_HAE_M, abs=1e-9)

    h_hae = msl_to_wgs84_hae(KSFO_28L_THR_NAVD88_MSL_M, KSFO_28L_GEOID18_UNDULATION_M)
    assert h_hae == pytest.approx(KSFO_28L_THR_WGS84_HAE_M, abs=1e-12)
    assert h_hae == pytest.approx(-28.6714, abs=1e-4)

    h_msl = wgs84_hae_to_msl(h_hae, KSFO_28L_GEOID18_UNDULATION_M)
    assert h_msl == pytest.approx(KSFO_28L_THR_NAVD88_MSL_M, abs=1e-12)

    guard = DatumAlignmentGuard(geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M)
    assert guard.hae_to_msl(KSFO_28L_THR_WGS84_HAE_M) == pytest.approx(KSFO_28L_THR_NAVD88_MSL_M, abs=1e-12)
    assert guard.msl_to_hae(KSFO_28L_THR_NAVD88_MSL_M) == pytest.approx(KSFO_28L_THR_WGS84_HAE_M, abs=1e-12)

    # Physical boundary |N| == 110.0 m passes; |N| > 110.0 m fails closed
    assert validate_geoid_undulation_m(MAX_ABS_GEOID_UNDULATION_M) == pytest.approx(110.0)
    assert validate_geoid_undulation_m(-MAX_ABS_GEOID_UNDULATION_M) == pytest.approx(-110.0)


@pytest.mark.parametrize(
    "bad_n",
    [float("nan"), float("inf"), float("-inf"), 110.001, -110.001, 250.0, -500.0, True],
)
def test_datum_guard_rejects_non_finite_and_out_of_range_geoid_undulation(bad_n: float) -> None:
    """Fail-closed on NaN, ±Inf, boolean, or |N_geoid_m| > 110 m."""
    with pytest.raises(ValueError):
        validate_geoid_undulation_m(bad_n)
    with pytest.raises(ValueError):
        wgs84_hae_to_msl(100.0, bad_n)
    with pytest.raises(ValueError):
        msl_to_wgs84_hae(100.0, bad_n)
    with pytest.raises(ValueError):
        DatumAlignmentGuard(geoid_undulation_m=bad_n)

    # Per-call invalid geoid undulation in evaluate_ekf3_switch fails closed with DATUM_MISMATCH_INHIBITED
    guard = DatumAlignmentGuard(geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M)
    dec = guard.evaluate_ekf3_switch(
        z_visual_m=KSFO_28L_THR_WGS84_HAE_M,
        z_baro_msl_m=KSFO_28L_THR_NAVD88_MSL_M,
        visual_datum="WGS84_HAE",
        baro_datum="MSL_EGM96",
        geoid_undulation_m=bad_n,
    )
    assert dec.allowed is False
    assert dec.reason_code == DATUM_REASON_MISMATCH_INHIBITED


# ---- 2. Failure Mode 8 Pre-Switch Innovation & Datum Mismatch Gating ----


def test_failure_mode_8_ksfo_datum_mismatch_inhibited_and_aligned_switch_allowed() -> None:
    """Mixing KSFO WGS-84 HAE (-28.67 m) with MSL baro (+3.96 m) is blocked until geoid-aligned."""
    # Case A: Missing geoid undulation when visual is WGS84_HAE and baro is MSL -> DATUM_MISMATCH_INHIBITED
    guard_no_n = DatumAlignmentGuard(geoid_undulation_m=None)
    dec_missing = guard_no_n.evaluate_ekf3_switch(
        z_visual_m=KSFO_28L_THR_WGS84_HAE_M,
        z_baro_msl_m=KSFO_28L_THR_NAVD88_MSL_M,
        visual_datum="WGS84_HAE",
        baro_datum="MSL_EGM96",
    )
    assert dec_missing.allowed is False
    assert dec_missing.reason_code == DATUM_REASON_MISMATCH_INHIBITED
    assert dec_missing.datum_conversion_applied is False

    # Case B: Raw unaligned mix (`apply_geoid_conversion=False`) -> DATUM_MISMATCH_INHIBITED (~32.63 m step caught)
    guard_ksfo = DatumAlignmentGuard(geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M)
    dec_raw_mix = guard_ksfo.evaluate_ekf3_switch(
        z_visual_m=KSFO_28L_THR_WGS84_HAE_M,
        z_baro_msl_m=KSFO_28L_THR_NAVD88_MSL_M,
        visual_datum="WGS84_HAE",
        baro_datum="MSL_EGM96",
        apply_geoid_conversion=False,
    )
    assert dec_raw_mix.allowed is False
    assert dec_raw_mix.reason_code == DATUM_REASON_MISMATCH_INHIBITED
    assert dec_raw_mix.abs_vertical_innovation_m is not None
    assert dec_raw_mix.abs_vertical_innovation_m == pytest.approx(abs(KSFO_28L_GEOID18_UNDULATION_M), abs=1e-3)

    # Case C: Misdeclared datum (raw KSFO HAE -28.67 m mistakenly labeled MSL vs +3.96 m baro)
    # -> blocked by PRE_SWITCH_VERT_INNOV_EXCEEDED (> 3.0 m)
    dec_misdeclared = guard_ksfo.evaluate_ekf3_switch(
        z_visual_m=KSFO_28L_THR_WGS84_HAE_M,
        z_baro_msl_m=KSFO_28L_THR_NAVD88_MSL_M,
        visual_datum="MSL_EGM96",
        baro_datum="MSL_EGM96",
    )
    assert dec_misdeclared.allowed is False
    assert dec_misdeclared.reason_code == DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED
    assert dec_misdeclared.abs_vertical_innovation_m is not None
    assert dec_misdeclared.abs_vertical_innovation_m > 28.0

    # Case D: Properly converted WGS84_HAE -> MSL with N = -32.6338 m (180 m cruise) -> ALLOWED
    cruise_msl_m = 180.0
    cruise_hae_m = msl_to_wgs84_hae(cruise_msl_m, KSFO_28L_GEOID18_UNDULATION_M)  # ~147.366 m
    dec_ok = guard_ksfo.evaluate_ekf3_switch(
        z_visual_m=cruise_hae_m + 1.25,
        z_baro_msl_m=cruise_msl_m,
        visual_datum="WGS84_HAE",
        baro_datum="MSL_EGM96",
    )
    assert dec_ok.allowed is True
    assert dec_ok.reason_code == DATUM_REASON_OK
    assert dec_ok.datum_conversion_applied is True
    assert dec_ok.z_visual_msl_m == pytest.approx(cruise_msl_m + 1.25, abs=1e-5)
    assert dec_ok.abs_vertical_innovation_m == pytest.approx(1.25, abs=1e-5)
    assert dec_ok.ned_down_visual_msl_m == pytest.approx(-(cruise_msl_m + 1.25), abs=1e-5)
    assert dec_ok.ned_down_baro_msl_m == pytest.approx(-cruise_msl_m, abs=1e-5)

    # Case E: Exact threshold boundary (3.00 m allowed; 3.01 m inhibited)
    dec_edge_ok = guard_ksfo.evaluate_ekf3_switch(
        z_visual_m=cruise_hae_m + DEFAULT_MAX_PRE_SWITCH_VERT_INNOV_M,
        z_baro_msl_m=cruise_msl_m,
    )
    assert dec_edge_ok.allowed is True
    assert dec_edge_ok.reason_code == DATUM_REASON_OK

    dec_edge_fail = guard_ksfo.evaluate_ekf3_switch(
        z_visual_m=cruise_hae_m + DEFAULT_MAX_PRE_SWITCH_VERT_INNOV_M + 0.01,
        z_baro_msl_m=cruise_msl_m,
    )
    assert dec_edge_fail.allowed is False
    assert dec_edge_fail.reason_code == DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED

    # Case F: Non-finite visual/baro altitude or unsupported ENU datum -> DATUM_MISMATCH_INHIBITED
    assert (
        guard_ksfo.evaluate_ekf3_switch(z_visual_m=float("nan"), z_baro_msl_m=180.0).reason_code
        == DATUM_REASON_MISMATCH_INHIBITED
    )
    assert (
        guard_ksfo.evaluate_ekf3_switch(z_visual_m=147.0, z_baro_msl_m=float("inf")).reason_code
        == DATUM_REASON_MISMATCH_INHIBITED
    )
    assert (
        guard_ksfo.evaluate_ekf3_switch(
            z_visual_m=180.0,
            z_baro_msl_m=180.0,
            visual_datum="ENU_UP",
        ).reason_code
        == DATUM_REASON_MISMATCH_INHIBITED
    )


def test_datum_guard_gates_ekf3_latch_and_live_hil_bridge_42007_emission() -> None:
    """Datum mismatch inhibits EKF3 42007 on both latch and live UDP `StudioHilBridgeManager`."""
    # 1. Direct latch gating proof
    latch = Ekf3SourceSwitchLatch()
    guard = DatumAlignmentGuard(geoid_undulation_m=None, max_pre_switch_vert_innov_m=3.0)

    # Spoof arrives while N_geoid_m is missing -> EKF3 switch inhibited (stays GNSS=1, no 42007)
    st_blk, dec_blk = guard.gate_ekf3_switch_latch(
        latch,
        timestamp_s=1.0,
        gnss_healthy=True,
        spoof_alarm=True,
        z_visual_m=KSFO_28L_THR_WGS84_HAE_M,
        z_baro_msl_m=KSFO_28L_THR_NAVD88_MSL_M,
    )
    assert dec_blk.allowed is False
    assert dec_blk.reason_code == DATUM_REASON_MISMATCH_INHIBITED
    assert st_blk.active_source == EKF3_SOURCE_GNSS
    assert st_blk.mav_cmd_id is None

    # Provide valid KSFO geoid undulation, but 5.5 m vertical residual -> PRE_SWITCH_VERT_INNOV_EXCEEDED
    guard.set_geoid_undulation(KSFO_28L_GEOID18_UNDULATION_M)
    st_hi, dec_hi = guard.gate_ekf3_switch_latch(
        latch,
        timestamp_s=2.0,
        gnss_healthy=True,
        spoof_alarm=True,
        z_visual_m=KSFO_28L_THR_WGS84_HAE_M + 5.5,
        z_baro_msl_m=KSFO_28L_THR_NAVD88_MSL_M,
    )
    assert dec_hi.allowed is False
    assert dec_hi.reason_code == DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED
    assert st_hi.active_source == EKF3_SOURCE_GNSS
    assert st_hi.mav_cmd_id is None

    # Aligned visual altitude (within 0.8 m) -> authorizes EKF3 42007 switch to VISION_OPTICAL (2)
    st_go, dec_go = guard.gate_ekf3_switch_latch(
        latch,
        timestamp_s=3.0,
        gnss_healthy=True,
        spoof_alarm=True,
        z_visual_m=KSFO_28L_THR_WGS84_HAE_M + 0.8,
        z_baro_msl_m=KSFO_28L_THR_NAVD88_MSL_M,
    )
    assert dec_go.allowed is True
    assert dec_go.reason_code == DATUM_REASON_OK
    assert st_go.active_source == EKF3_SOURCE_VISION_OPTICAL
    assert st_go.mav_cmd_id == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
    assert st_go.commanded_source == EKF3_SOURCE_VISION_OPTICAL

    # 2. Live UDP StudioHilBridgeManager integration proof (dev SITL loopback mode)
    mgr = StudioHilBridgeManager(
        non_flight_sim_only=False,
        datum_guard=DatumAlignmentGuard(geoid_undulation_m=None, max_pre_switch_vert_innov_m=3.0),
    )
    tx_listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    injector = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_listener.bind(("127.0.0.1", 0))
        tx_listener.settimeout(1.0)
        tx_port = int(tx_listener.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        mgr.connect(mode="mavlink2_udp", rx_port=rx_port, tx_port=tx_port, tx_command_mode="set_attitude_target")
        cruise_msl = 180.0
        cruise_hae = msl_to_wgs84_hae(cruise_msl, KSFO_28L_GEOID18_UNDULATION_M)
        mgr.update_visual_altitude(visual_alt_m=cruise_hae, visual_datum="WGS84_HAE", geoid_undulation_m=None)
        mgr.set_autonomy_triggers(gnss_spoof_alarm=True, gnss_healthy=True)

        gpi_1 = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=1000,
                lat_e7=376_117_200,
                lon_e7=-1_223_583_670,
                alt_mm=int(cruise_msl * 1000),
                relative_alt_mm=175_000,
                vx_cms=1500,
                vy_cms=-2400,
                vz_cms=0,
                hdg_cdeg=29800,
            ),
            seq=1,
            sysid=1,
            compid=1,
        )
        injector.sendto(gpi_1, ("127.0.0.1", rx_port))
        pkt_first, _ = tx_listener.recvfrom(512)
        dec_first = decode_frame(pkt_first)
        # Only guidance frame emitted; NO #76 COMMAND_LONG(42007) emitted while datum is unaligned!
        assert dec_first.msgid != MSGID_COMMAND_LONG
        st_inhibited = mgr.status()
        assert st_inhibited["ekf3_active_source"] == EKF3_SOURCE_GNSS
        assert st_inhibited["datum_inhibited_switches"] == 1
        assert st_inhibited["last_datum_decision"]["reason_code"] == DATUM_REASON_MISMATCH_INHIBITED

        # Supply explicit KSFO GEOID18 undulation -> next packet allows EKF3 42007 switch cleanly
        mgr.update_visual_altitude(
            visual_alt_m=cruise_hae + 0.5,
            visual_datum="WGS84_HAE",
            geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M,
        )
        gpi_2 = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=1100,
                lat_e7=376_117_500,
                lon_e7=-1_223_584_000,
                alt_mm=int(cruise_msl * 1000),
                relative_alt_mm=175_000,
                vx_cms=1500,
                vy_cms=-2400,
                vz_cms=0,
                hdg_cdeg=29800,
            ),
            seq=2,
            sysid=1,
            compid=1,
        )
        injector.sendto(gpi_2, ("127.0.0.1", rx_port))
        emitted_42007 = False
        for _ in range(2):
            pkt, _ = tx_listener.recvfrom(512)
            dec = decode_frame(pkt)
            if isinstance(dec.message, MavCommandLong) and dec.message.command == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET:
                assert dec.message.param1 == pytest.approx(float(EKF3_SOURCE_VISION_OPTICAL))
                emitted_42007 = True
        assert emitted_42007 is True
        st_aligned = mgr.status()
        assert st_aligned["ekf3_active_source"] == EKF3_SOURCE_VISION_OPTICAL
        assert st_aligned["last_datum_decision"]["allowed"] is True
        assert st_aligned["last_datum_decision"]["reason_code"] == DATUM_REASON_OK
    finally:
        mgr.shutdown()
        tx_listener.close()
        injector.close()


# ---- 3. ShadowModeTelemetryLogger: Listen-Only Invariants & SHA-256 Chain Verification ----


def test_shadow_logger_enforces_immutable_listen_only_invariants_and_monotonic_timestamps(tmp_path: Path) -> None:
    """Hard invariants (`tx_inhibited=True`, `actuator_authority=False`, `0` TX, monotonic ts)."""
    log_file = tmp_path / "phase2_rooftop.ndjson"

    with pytest.raises(ShadowModeTxInhibitedError):
        ShadowModeTelemetryLogger(log_file, tx_inhibited=False)
    with pytest.raises(ShadowModeTxInhibitedError):
        ShadowModeTelemetryLogger(log_file, actuator_authority=True)
    with pytest.raises(ValueError, match="trial_phase"):
        ShadowModeTelemetryLogger(log_file, trial_phase="active_actuator_flight")

    with ShadowModeTelemetryLogger(log_file, trial_phase="phase2_rooftop_car_roof") as logger:
        assert logger.tx_inhibited is True
        assert logger.actuator_authority is False
        assert logger.outbound_mavlink_frames_emitted == 0

        # Outbound TX direction or non-zero outbound frame count MUST raise ShadowModeTxInhibitedError
        with pytest.raises(ShadowModeTxInhibitedError):
            logger.reject_outbound_tx(msgid=75, reason="hardware TX pin removed")
        with pytest.raises(ShadowModeTxInhibitedError):
            logger.record_event(timestamp_mono_us=1_000, event_type="tx_cmd", payload={}, direction="tx")
        with pytest.raises(ShadowModeTxInhibitedError):
            logger.record_event(
                timestamp_mono_us=1_000,
                event_type="rx_mavlink2",
                payload={},
                direction="rx",
                outbound_mavlink_frames=1,
            )

        # NaN payload fails closed
        with pytest.raises(ValueError):
            logger.record_event(
                timestamp_mono_us=1_000,
                event_type="rx_mavlink2",
                payload={"down_m": float("nan")},
            )

        # Write 4 strictly monotonic records
        r0 = logger.record_event(
            timestamp_mono_us=100_000,
            event_type="rx_mavlink2",
            payload={"msgid": MSGID_GLOBAL_POSITION_INT, "down_m": -180.0},
            direction="rx",
        )
        assert r0["seq"] == 0
        assert r0["prev_hash"] == SHADOW_GENESIS_HASH
        assert r0["schema"] == SHADOW_MODE_LOG_SCHEMA
        assert r0["tx_inhibited"] is True
        assert r0["actuator_authority"] is False
        assert r0["outbound_mavlink_frames_emitted"] == 0
        assert r0["watermark"] == NON_FLIGHT_WATERMARK

        # Duplicate and backward timestamps MUST be rejected fail-closed
        with pytest.raises(ValueError, match="Non-monotonic"):
            logger.record_event(timestamp_mono_us=100_000, event_type="rx_mavlink2", payload={"a": 1})
        with pytest.raises(ValueError, match="Non-monotonic"):
            logger.record_event(timestamp_mono_us=99_999, event_type="rx_mavlink2", payload={"a": 1})

        r1 = logger.record_event(
            timestamp_mono_us=200_000,
            event_type="datum_guard_decision",
            payload={"reason_code": DATUM_REASON_OK, "abs_vertical_innovation_m": 0.42},
            direction="internal",
        )
        assert r1["prev_hash"] == r0["entry_hash"]
        r2 = logger.record_event(
            timestamp_mono_us=300_000,
            event_type="rx_mavlink2",
            payload={"msgid": MSGID_ATTITUDE, "roll_deg": 2.5},
            direction="rx",
        )
        assert r2["prev_hash"] == r1["entry_hash"]
        assert logger.record_count == 3
        assert logger.head_hash == r2["entry_hash"]

    ver = verify_shadow_log_chain(log_file)
    assert ver.valid is True
    assert ver.record_count == 3
    assert ver.first_seq == 0
    assert ver.last_seq == 2
    assert ver.tx_inhibited_all is True
    assert ver.actuator_authority_all_false is True
    assert ver.total_outbound_mavlink_frames == 0
    assert ver.monotonic_timestamps_ok is True
    assert ver.errors == ()


def test_shadow_log_chain_verification_catches_payload_tamper_deletion_and_forged_tx(tmp_path: Path) -> None:
    """Verify tamper-evident SHA-256 chain detects 1-byte edits, deletions, rollbacks, & forged TX."""
    base_log = tmp_path / "phase3_shadow.ndjson"
    with ShadowModeTelemetryLogger(base_log, trial_phase="phase3_university_shadow_tx_pin_removed") as logger:
        for i in range(4):
            logger.record_event(
                timestamp_mono_us=(i + 1) * 50_000,
                event_type="rx_mavlink2",
                payload={"frame": i, "down_m": -150.0 - i},
                direction="rx",
            )

    assert ShadowModeTelemetryLogger.verify_chain(base_log).valid is True
    orig_lines = base_log.read_text(encoding="utf-8").splitlines()

    # 1. Single-field payload tamper on line 2 (index 1) -> caught by SHA-256 hash mismatch
    tampered_payload = tmp_path / "tampered_payload.ndjson"
    mod_lines = list(orig_lines)
    rec1 = json.loads(mod_lines[1])
    rec1["payload"]["down_m"] = -999.0
    mod_lines[1] = json.dumps(rec1, sort_keys=True, separators=(",", ":"))
    tampered_payload.write_text("\n".join(mod_lines) + "\n", encoding="utf-8")
    v_tamper = verify_shadow_log_chain(tampered_payload)
    assert v_tamper.valid is False
    assert v_tamper.first_failed_line == 2
    assert any("SHA-256 tamper detected" in err for err in v_tamper.errors)

    # 2. Deleting a middle record -> caught by sequence gap + prev_hash break
    deleted_line = tmp_path / "deleted_line.ndjson"
    deleted_line.write_text("\n".join([orig_lines[0], orig_lines[2], orig_lines[3]]) + "\n", encoding="utf-8")
    v_del = verify_shadow_log_chain(deleted_line)
    assert v_del.valid is False
    assert v_del.first_failed_line == 2

    # 3. Forged outbound frame or tx_inhibited=False (even if attacker re-hashes entire chain!) -> REJECTED
    forged_tx = tmp_path / "forged_tx.ndjson"
    rehashed_lines: list[str] = []
    prev_h = SHADOW_GENESIS_HASH
    for idx, line in enumerate(orig_lines):
        rec = json.loads(line)
        rec["prev_hash"] = prev_h
        if idx == 2:
            rec["tx_inhibited"] = False
            rec["outbound_mavlink_frames_emitted"] = 1
        body = {k: v for k, v in rec.items() if k != "entry_hash"}
        new_h = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        rec["entry_hash"] = new_h
        prev_h = new_h
        rehashed_lines.append(json.dumps(rec, sort_keys=True, separators=(",", ":")))
    forged_tx.write_text("\n".join(rehashed_lines) + "\n", encoding="utf-8")
    v_forged = verify_shadow_log_chain(forged_tx)
    assert v_forged.valid is False
    assert v_forged.tx_inhibited_all is False
    assert v_forged.total_outbound_mavlink_frames == 1

    # 4. Empty or missing log -> fails closed
    empty_log = tmp_path / "empty.ndjson"
    empty_log.write_text("", encoding="utf-8")
    assert verify_shadow_log_chain(empty_log).valid is False
    assert verify_shadow_log_chain(tmp_path / "missing.ndjson").valid is False


def test_live_udp_shadow_tap_under_non_flight_sim_only_emits_zero_tx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live UDP `connect_shadow_tap()` works under `NON_FLIGHT_SIM_ONLY=1` with 0 outbound TX."""
    monkeypatch.setenv("OPENXVS_NON_FLIGHT_SIM_ONLY", "1")
    log_path = tmp_path / "phase3_university_live_tap.ndjson"
    guard = DatumAlignmentGuard(geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M)
    mgr = StudioHilBridgeManager(datum_guard=guard)
    rx_port = _alloc_free_udp_port()
    injector = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        with ShadowModeTelemetryLogger(
            log_path,
            trial_phase="phase3_university_shadow_tx_pin_removed",
            hardware_tx_pin_removed=True,
        ) as logger:
            st_conn = mgr.connect_shadow_tap(
                shadow_logger=logger,
                mode="mavlink2_udp",
                rx_port=rx_port,
            )
            assert st_conn["shadow_mode"] is True
            assert st_conn["tx_inhibited"] is True
            assert st_conn["tx_enabled"] is False
            assert st_conn["actuator_authority"] is False
            assert st_conn["non_flight_sim_only"] is True
            assert mgr._tx_sock is None

            # Attempting TX or enabling TX while tapped MUST raise ShadowModeTxInhibitedError (and NonFlightSimOnlyError)
            hb_wire = encode_frame(MavHeartbeat(custom_mode=15, mav_type=1, autopilot=3), seq=1, sysid=1, compid=1)
            with pytest.raises(ShadowModeTxInhibitedError):
                mgr.send_raw_tx_frame(hb_wire)
            with pytest.raises(NonFlightSimOnlyError):
                mgr.set_autonomy_triggers(tx_enabled=True)

            # Configure visual HAE altitude matching 180 m MSL cruise at KSFO
            cruise_hae = msl_to_wgs84_hae(180.0, KSFO_28L_GEOID18_UNDULATION_M)
            mgr.update_visual_altitude(visual_alt_m=cruise_hae + 0.4, visual_datum="WGS84_HAE")

            att_wire = encode_frame(
                MavAttitude(time_boot_ms=500, roll=0.05, pitch=-0.01, yaw=math.radians(298.0)),
                seq=2,
                sysid=1,
                compid=1,
            )
            gpi_wire = encode_frame(
                MavGlobalPositionInt(
                    time_boot_ms=600,
                    lat_e7=376_117_200,
                    lon_e7=-1_223_583_670,
                    alt_mm=180_000,
                    relative_alt_mm=175_000,
                    vx_cms=1500,
                    vy_cms=-2500,
                    vz_cms=0,
                    hdg_cdeg=29800,
                ),
                seq=3,
                sysid=1,
                compid=1,
            )
            for frame in (hb_wire, att_wire, gpi_wire):
                injector.sendto(frame, ("127.0.0.1", rx_port))
                time.sleep(0.02)

            deadline = time.perf_counter() + 2.0
            while logger.record_count < 3 and time.perf_counter() < deadline:
                time.sleep(0.02)

            st_after = mgr.status()
            assert st_after["rx_packets"] == 3
            assert st_after["tx_packets"] == 0
            assert st_after["tx_msgids_emitted"] == {}
            assert st_after["actuator_authority"] is False
            assert st_after["tx_inhibited"] is True
            assert st_after["msg84_ever_emitted"] is False
            assert MSGID_SET_POSITION_TARGET_LOCAL_NED not in st_after["tx_msgids_emitted"]
            assert st_after["last_datum_decision"] is not None
            assert st_after["last_datum_decision"]["allowed"] is True
            assert st_after["last_datum_decision"]["reason_code"] == DATUM_REASON_OK
            mgr.shutdown()

        ver = verify_shadow_log_chain(log_path)
        assert ver.valid is True
        assert ver.record_count == 3
        assert ver.tx_inhibited_all is True
        assert ver.actuator_authority_all_false is True
        assert ver.total_outbound_mavlink_frames == 0
        assert ver.monotonic_timestamps_ok is True
    finally:
        mgr.shutdown()
        injector.close()
