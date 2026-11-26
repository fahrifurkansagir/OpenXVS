"""Phase 2B Tests: Zero-GPL MAVLink 2.0 Framing, Closed-Loop HIL RX->TX Bridge & Live SITL Proof."""

from __future__ import annotations

import json
import math
from pathlib import Path
import socket
import struct
import sys
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "xvs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_live_arduplane_sitl_proof import (  # noqa: E402
    DEFAULT_PROOF_OUT_PATH,
    MIN_ARDUPLANE_SITL_VERSION,
    parse_arduplane_version,
    run_python_6dof_udp_twin_proof,
)
from xvs.v2.studio_bridge.hil_bridge import (  # noqa: E402
    StudioHilBridgeManager,
    run_hil_loopback_self_test,
)
from xvs.v2.xvs_mavlink.pod_daa_codec import (  # noqa: E402
    CRC_EXTRA_BY_MSGID,
    MAVLINK_IFLAG_SIGNED,
    MAVLINK_V2_SIGNATURE_BYTES,
    MAVLINK_V2_STX,
    MSGID_ATTITUDE,
    MSGID_COMMAND_ACK,
    MSGID_COMMAND_INT,
    MSGID_COMMAND_LONG,
    MSGID_GLOBAL_POSITION_INT,
    MSGID_HEARTBEAT,
    MSGID_SET_ATTITUDE_TARGET,
    MSGID_SET_POSITION_TARGET_LOCAL_NED,
    ArduPlaneSafetyError,
    MavAttitude,
    MavCommandAck,
    MavCommandInt,
    MavCommandLong,
    MavGlobalPositionInt,
    MavHeartbeat,
    MavSetAttitudeTarget,
    MavlinkCrcError,
    MavlinkFrameError,
    PodCivilMissionClass,
    PodPluginHealth,
    PodPluginLifecycleState,
    decode_frame,
    encode_frame,
    extract_mavlink_v2_frames,
    validate_outbound_msgid,
)
from xvs.v2.xvs_rt_bridge.arduplane_bridge import (  # noqa: E402
    ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
    DEFAULT_MAX_BANK_RAD,
    DEFAULT_MAX_PITCH_RAD,
    DEFAULT_V_MAX_MPS,
    DEFAULT_V_MIN_MPS,
    EKF3_SOURCE_GNSS,
    EKF3_SOURCE_VISION_OPTICAL,
    MAV_CMD_DO_REPOSITION,
    MAV_CMD_GUIDED_CHANGE_HEADING,
    Ekf3SourceSwitchLatch,
    build_command_int_avoidance,
    build_command_int_guided_heading,
    build_set_attitude_target,
    decode_arduplane_frame,
    encode_command_int_frame,
    encode_command_long_frame,
    encode_ekf3_source_switch_frame,
    encode_heartbeat_frame,
    encode_set_attitude_target_frame,
    validate_fixed_wing_avoidance_msgid,
)


def _alloc_free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def test_zero_gpl_no_pymavlink_imported_or_referenced() -> None:
    """Strict Zero-GPL Lock: `pymavlink` must never be imported or used by OpenXVS code."""
    assert "pymavlink" not in sys.modules
    owned_sources = (
        REPO_ROOT / "src" / "xvs" / "v2" / "xvs_mavlink" / "pod_daa_codec.py",
        REPO_ROOT / "src" / "xvs" / "v2" / "xvs_rt_bridge" / "arduplane_bridge.py",
        REPO_ROOT / "src" / "xvs" / "v2" / "studio_bridge" / "hil_bridge.py",
        REPO_ROOT / "scripts" / "xvs" / "run_live_arduplane_sitl_proof.py",
    )
    for path in owned_sources:
        text = path.read_text(encoding="utf-8")
        assert "import pymavlink" not in text
        assert "from pymavlink" not in text


def test_standard_mavlink2_crc_extra_and_unsigned_roundtrip() -> None:
    """Verify exact MAVLink common.xml CRC_EXTRA bytes and lossless round-trip for #0/#30/#33/#75/#76/#77/#82."""
    expected_crc_extras = {
        MSGID_HEARTBEAT: 50,
        MSGID_ATTITUDE: 39,
        MSGID_GLOBAL_POSITION_INT: 104,
        MSGID_COMMAND_INT: 158,
        MSGID_COMMAND_LONG: 152,
        MSGID_COMMAND_ACK: 143,
        MSGID_SET_ATTITUDE_TARGET: 49,
    }
    for msgid, expected_ce in expected_crc_extras.items():
        assert CRC_EXTRA_BY_MSGID[msgid] == expected_ce

    messages = [
        MavHeartbeat(custom_mode=15, mav_type=1, autopilot=3, base_mode=0x81, system_status=4),
        MavAttitude(
            time_boot_ms=12345,
            roll=0.25,
            pitch=-0.05,
            yaw=1.25,
            rollspeed=0.01,
            pitchspeed=-0.02,
            yawspeed=0.03,
        ),
        MavGlobalPositionInt(
            time_boot_ms=12350,
            lat_e7=376_117_200,
            lon_e7=-1_223_583_670,
            alt_mm=185_000,
            relative_alt_mm=180_000,
            vx_cms=1650,
            vy_cms=-2200,
            vz_cms=-40,
            hdg_cdeg=29800,
        ),
        MavCommandInt(
            command=MAV_CMD_DO_REPOSITION,
            frame=6,
            param1=24.0,
            param2=0.0,
            param3=0.0,
            param4=0.0,
            x=376_120_000,
            y=-1_223_500_000,
            z=180.0,
        ),
        MavCommandInt(
            command=MAV_CMD_GUIDED_CHANGE_HEADING,
            frame=0,
            param1=0.0,
            param2=350.0,
            param3=4.5,
        ),
        MavCommandLong(
            command=ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
            param1=float(EKF3_SOURCE_VISION_OPTICAL),
        ),
        MavCommandAck(command=ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET, result=0, progress=100, target_system=255),
        MavSetAttitudeTarget(
            time_boot_ms=12400,
            q=(0.98, 0.15, 0.02, -0.12),
            thrust=0.68,
            type_mask=0x07,
        ),
    ]

    stream = bytearray()
    for idx, msg in enumerate(messages):
        wire = encode_frame(msg, seq=idx + 1, sysid=1, compid=191)
        assert wire[0] == MAVLINK_V2_STX
        assert wire[2] == 0x00
        dec = decode_frame(wire)
        assert dec.msgid == msg.MSGID
        assert dec.seq == idx + 1
        assert dec.signed is False
        assert dec.signature_verified is False
        # Float32 fields compare within IEEE-754 f32 precision
        if isinstance(msg, MavAttitude):
            assert isinstance(dec.message, MavAttitude)
            assert dec.message.roll == pytest.approx(msg.roll, abs=1e-6)
            assert dec.message.pitch == pytest.approx(msg.pitch, abs=1e-6)
            assert dec.message.yaw == pytest.approx(msg.yaw, abs=1e-6)
        elif isinstance(msg, MavSetAttitudeTarget):
            assert isinstance(dec.message, MavSetAttitudeTarget)
            assert dec.message.thrust == pytest.approx(msg.thrust, abs=1e-6)
            for a, b in zip(dec.message.q, msg.q):
                assert a == pytest.approx(b, abs=1e-6)
        elif isinstance(msg, MavCommandInt):
            assert isinstance(dec.message, MavCommandInt)
            assert dec.message.command == msg.command
            assert dec.message.x == msg.x and dec.message.y == msg.y
            assert dec.message.param1 == pytest.approx(msg.param1, abs=1e-5)
        elif isinstance(msg, MavCommandLong):
            assert isinstance(dec.message, MavCommandLong)
            assert dec.message.command == msg.command
            assert dec.message.param1 == pytest.approx(msg.param1, abs=1e-6)
        else:
            assert dec.message == msg
        stream.extend(wire)

    # Multi-frame stream extractor recovers all frames verbatim
    extracted, remainder = extract_mavlink_v2_frames(bytes(stream))
    assert remainder == b""
    assert len(extracted) == len(messages)


def test_mavlink2_sha256_signing_roundtrip_and_tamper_rejection() -> None:
    """Verify MAVLink 2.0 13-byte SHA-256 signing (`incompat_flags=0x01`) + 1-bit tamper rejection."""
    key = bytes(range(10, 42))
    wrong_key = bytes(range(20, 52))

    samples = [
        MavHeartbeat(custom_mode=15, mav_type=18, autopilot=8),
        MavAttitude(time_boot_ms=500, roll=0.1, pitch=-0.03, yaw=0.5),
        MavGlobalPositionInt(
            time_boot_ms=510,
            lat_e7=376_117_200,
            lon_e7=-1_223_583_670,
            alt_mm=150_000,
            relative_alt_mm=145_000,
            vx_cms=1200,
            vy_cms=-1800,
            vz_cms=0,
            hdg_cdeg=30300,
        ),
        MavCommandInt(command=MAV_CMD_DO_REPOSITION, param1=22.0, x=376_120_000, y=-1_223_580_000, z=150.0),
        MavCommandLong(command=ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET, param1=2.0),
        MavSetAttitudeTarget(time_boot_ms=520, q=(0.99, 0.14, 0.0, 0.0), thrust=0.65),
        PodPluginHealth.build(
            plugin_id_hash=0xCAFE_BABE_1234_5678,
            timestamp_mono_us=9_000_000,
            last_seq_no=12,
            lifecycle_state=PodPluginLifecycleState.HEALTHY,
            civil_mission_class=PodCivilMissionClass.SEARCH_AND_RESCUE,
        ),
    ]

    for idx, msg in enumerate(samples):
        unsigned = encode_frame(msg, seq=idx, sysid=255, compid=191)
        signed = encode_frame(
            msg,
            seq=idx,
            sysid=255,
            compid=191,
            secret_key=key,
            link_id=3,
            timestamp_48=123_456_789 + idx,
        )
        assert len(signed) == len(unsigned) + MAVLINK_V2_SIGNATURE_BYTES
        assert signed[2] == MAVLINK_IFLAG_SIGNED

        dec = decode_frame(signed, verify_inner_crc32=True, secret_key=key, require_signature=True)
        assert dec.signed is True
        assert dec.signature_verified is True
        assert dec.link_id == 3
        assert dec.timestamp_48 == 123_456_789 + idx

        # Unsigned frame rejected when require_signature=True
        with pytest.raises(MavlinkFrameError, match="require_signature"):
            decode_frame(unsigned, secret_key=key, require_signature=True)

        # Wrong secret key rejected fail-closed
        with pytest.raises(MavlinkCrcError, match="signature"):
            decode_frame(signed, secret_key=wrong_key, require_signature=True)

        # 1-bit tamper in header, payload, CRC, and signature trailer must all be rejected
        for flip_pos in (4, 10, len(unsigned) - 1, len(signed) - 10, len(signed) - 1):
            bad = bytearray(signed)
            bad[flip_pos] ^= 0x01
            with pytest.raises(MavlinkCrcError):
                decode_frame(bytes(bad), secret_key=key, require_signature=True)

    # Unsupported non-signing incompat_flags (0x02) remain hard-rejected
    with pytest.raises(MavlinkFrameError, match="incompat_flags"):
        encode_frame(samples[0], incompat_flags=0x02)


def test_msgid_84_hard_banned_everywhere_raising_value_error() -> None:
    """MAVLink #84 (SET_POSITION_TARGET_LOCAL_NED) is hard-banned in encode, decode, and bridge TX."""
    class _ForbiddenMsg84:
        MSGID = MSGID_SET_POSITION_TARGET_LOCAL_NED

        def pack_payload(self) -> bytes:
            return b"\x00" * 53

    with pytest.raises(ValueError, match="#84"):
        validate_outbound_msgid(MSGID_SET_POSITION_TARGET_LOCAL_NED)
    with pytest.raises(ArduPlaneSafetyError, match="#84"):
        validate_outbound_msgid(MSGID_SET_POSITION_TARGET_LOCAL_NED)
    with pytest.raises(ValueError, match="#84"):
        validate_fixed_wing_avoidance_msgid(MSGID_SET_POSITION_TARGET_LOCAL_NED)
    with pytest.raises(ValueError, match="#84"):
        encode_frame(_ForbiddenMsg84())  # type: ignore[arg-type]

    raw_84 = (
        bytes([MAVLINK_V2_STX])
        + struct.pack("<BBBBBBBBB", 1, 0, 0, 1, 1, 191, MSGID_SET_POSITION_TARGET_LOCAL_NED, 0, 0)
        + b"\x00\x00\x00"
    )
    with pytest.raises(ValueError, match="#84"):
        decode_frame(raw_84)
    with pytest.raises(ValueError, match="#84"):
        decode_arduplane_frame(raw_84)

    mgr = StudioHilBridgeManager()
    with pytest.raises(ValueError, match="#84"):
        mgr.send_raw_tx_frame(raw_84)


def test_ekf3_42007_strictly_gated_by_latch_and_command_int_attitude_clamps() -> None:
    """Verify #76(42007) cannot bypass Ekf3SourceSwitchLatch (2.0s dwell) and #75/#82 clamp safely."""
    key = bytes(range(32))

    # Direct bypass of 42007 via generic COMMAND_LONG encoder must be blocked
    with pytest.raises(ValueError, match="Ekf3SourceSwitchLatch"):
        encode_command_long_frame(command=ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET, param1=2.0)

    latch = Ekf3SourceSwitchLatch(min_gnss_revert_dwell_s=2.0)
    st_nom = latch.step(timestamp_s=0.0, gnss_healthy=True, spoof_alarm=False)
    with pytest.raises(ValueError, match="not authorized"):
        encode_ekf3_source_switch_frame(st_nom)

    # Spoof triggers immediate Vision switch (#76 COMMAND_LONG 42007 param1=2.0)
    st_spoof = latch.step(timestamp_s=1.0, gnss_healthy=True, spoof_alarm=True)
    wire_vis = encode_ekf3_source_switch_frame(st_spoof, seq=7, secret_key=key, timestamp_48=5000)
    dec_vis = decode_arduplane_frame(wire_vis, secret_key=key, require_signature=True)
    assert isinstance(dec_vis.message, MavCommandLong)
    assert dec_vis.message.command == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
    assert dec_vis.message.param1 == pytest.approx(float(EKF3_SOURCE_VISION_OPTICAL))

    # Dwell < 2.0 s stays latched on Vision (no switch authorized)
    st_early = latch.step(timestamp_s=2.0, gnss_healthy=True, spoof_alarm=False)
    st_19s = latch.step(timestamp_s=3.9, gnss_healthy=True, spoof_alarm=False)
    assert st_early.active_source == EKF3_SOURCE_VISION_OPTICAL
    assert st_19s.active_source == EKF3_SOURCE_VISION_OPTICAL
    with pytest.raises(ValueError, match="not authorized"):
        encode_ekf3_source_switch_frame(st_19s)

    # Dwell >= 2.0 s authorizes revert to GNSS (param1=1.0)
    st_revert = latch.step(timestamp_s=4.0, gnss_healthy=True, spoof_alarm=False)
    wire_gnss = encode_ekf3_source_switch_frame(st_revert, seq=8, secret_key=key, timestamp_48=6000)
    dec_gnss = decode_arduplane_frame(wire_gnss, secret_key=key, require_signature=True)
    assert isinstance(dec_gnss.message, MavCommandLong)
    assert dec_gnss.message.param1 == pytest.approx(float(EKF3_SOURCE_GNSS))

    # #75 COMMAND_INT GUIDED_CHANGE_HEADING & DO_REPOSITION clamps + wire encode
    cmd_hdg = build_command_int_guided_heading(
        target_heading_deg=415.0,  # wraps to 55.0
        airspeed_mps=10.0,  # clamps to 18.0
        bank_cmd_rad=math.radians(52.0),  # clamps to 30.0
    )
    assert cmd_hdg.bank_clamped is True and cmd_hdg.airspeed_clamped is True
    assert abs(cmd_hdg.bank_cmd_rad - DEFAULT_MAX_BANK_RAD) < 1e-12
    assert abs(cmd_hdg.airspeed_mps - DEFAULT_V_MIN_MPS) < 1e-12
    wire_hdg = encode_command_int_frame(cmd_hdg, seq=9, secret_key=key, timestamp_48=7000)
    dec_hdg = decode_arduplane_frame(wire_hdg, secret_key=key, require_signature=True)
    assert isinstance(dec_hdg.message, MavCommandInt)
    assert dec_hdg.message.command == MAV_CMD_GUIDED_CHANGE_HEADING
    assert dec_hdg.message.param2 == pytest.approx(55.0, abs=1e-4)

    cmd_repo = build_command_int_avoidance(
        lat_deg=37.62,
        lon_deg=-122.35,
        alt_m=160.0,
        airspeed_mps=99.0,  # clamps to 65.0
        bank_cmd_rad=math.radians(-40.0),
        loiter_radius_m=0.0,
    )
    assert cmd_repo.airspeed_mps == pytest.approx(DEFAULT_V_MAX_MPS)
    wire_repo = encode_command_int_frame(cmd_repo, seq=10)
    dec_repo = decode_arduplane_frame(wire_repo)
    assert isinstance(dec_repo.message, MavCommandInt)
    assert dec_repo.message.command == MAV_CMD_DO_REPOSITION

    # #82 SET_ATTITUDE_TARGET clamps (|roll|<=30, |pitch|<=20, w>=0) + wire encode
    cmd_att = build_set_attitude_target(
        roll_rad=math.radians(-45.0),
        pitch_rad=math.radians(35.0),
        yaw_rad=math.radians(298.0),
        thrust_norm=0.72,
        airspeed_mps=12.0,
        time_boot_ms=4200,
    )
    assert abs(cmd_att.roll_rad + DEFAULT_MAX_BANK_RAD) < 1e-12
    assert abs(cmd_att.pitch_rad - DEFAULT_MAX_PITCH_RAD) < 1e-12
    assert cmd_att.q_ned_body[0] >= 0.0
    wire_att = encode_set_attitude_target_frame(cmd_att, seq=11, secret_key=key, timestamp_48=8000)
    dec_att = decode_arduplane_frame(wire_att, secret_key=key, require_signature=True)
    assert isinstance(dec_att.message, MavSetAttitudeTarget)
    assert dec_att.message.q[0] >= 0.0


def test_studio_hil_bridge_closed_loop_mavlink2_and_flightgear_udp() -> None:
    """Verify real UDP closed-loop RX->TX on `StudioHilBridgeManager` for MAVLink2 (signed) & FlightGear."""
    key = bytes(range(32))
    mgr = StudioHilBridgeManager()
    tx_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_rx.bind(("127.0.0.1", 0))
        tx_rx.settimeout(1.5)
        tx_port = int(tx_rx.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        # 1. Signed MAVLink 2.0 closed-loop RX -> TX (#82, #75, #76(42007))
        mgr.connect(
            mode="mavlink2_udp",
            rx_port=rx_port,
            tx_port=tx_port,
            tx_command_mode="set_attitude_target",
            secret_key=key,
            stale_telemetry_timeout_s=0.3,
        )
        # Before telemetry arrives, verify staleness check after timeout
        assert mgr.is_telemetry_stale(now_mono=time.monotonic() + 1.0) is True

        att_signed = encode_frame(
            MavAttitude(time_boot_ms=100, roll=0.05, pitch=0.01, yaw=math.radians(298.0)),
            seq=1,
            sysid=1,
            compid=1,
            secret_key=key,
            timestamp_48=1000,
        )
        inj.sendto(att_signed, ("127.0.0.1", rx_port))
        pkt1, _ = tx_rx.recvfrom(512)
        d1 = decode_frame(pkt1, secret_key=key, require_signature=True)
        assert d1.msgid == MSGID_SET_ATTITUDE_TARGET
        assert d1.signature_verified is True

        # Trigger CBDR + spoof + COMMAND_INT heading mode
        mgr.set_autonomy_triggers(
            cbdr_threat_active=True,
            cbdr_evade_direction=-1.0,
            gnss_spoof_alarm=True,
            tx_command_mode="command_int_heading",
        )
        gpi_signed = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=200,
                lat_e7=376_117_200,
                lon_e7=-1_223_583_670,
                alt_mm=180_000,
                relative_alt_mm=175_000,
                vx_cms=1500,
                vy_cms=-2400,
                vz_cms=0,
                hdg_cdeg=29800,
            ),
            seq=2,
            sysid=1,
            compid=1,
            secret_key=key,
            timestamp_48=2000,
        )
        inj.sendto(gpi_signed, ("127.0.0.1", rx_port))
        got_ids = []
        for _ in range(2):
            pkt, _ = tx_rx.recvfrom(512)
            dec = decode_frame(pkt, secret_key=key, require_signature=True)
            got_ids.append(dec.msgid)
        assert MSGID_COMMAND_LONG in got_ids
        assert MSGID_COMMAND_INT in got_ids
        assert MSGID_SET_POSITION_TARGET_LOCAL_NED not in got_ids

        st_mav = mgr.status()
        assert st_mav["tx_packets"] >= 3
        assert st_mav["msg84_ever_emitted"] is False
        assert st_mav["last_tx_latency_ms"] is not None and st_mav["last_tx_latency_ms"] >= 0.0
        assert st_mav["ekf3_active_source"] == EKF3_SOURCE_VISION_OPTICAL

        # 2. FlightGear Generic UDP closed-loop RX CSV -> TX clamped CSV
        fg_rx_port = _alloc_free_udp_port()
        mgr.connect(mode="flightgear_udp", rx_port=fg_rx_port, tx_port=tx_port)
        mgr.set_autonomy_triggers(cbdr_threat_active=True, cbdr_evade_direction=1.0)
        inj.sendto(
            b"37.6117200,-122.3583670,180.000,298.000,-1.500,2.000,62.000,15.000,-28.000,0.000\n",
            ("127.0.0.1", fg_rx_port),
        )
        fg_tx_raw, _ = tx_rx.recvfrom(256)
        parts = fg_tx_raw.decode("ascii").strip().split(",")
        assert len(parts) == 4
        assert abs(float(parts[0])) <= 30.0 + 1e-6
        assert abs(float(parts[1])) <= 20.0 + 1e-6
        assert DEFAULT_V_MIN_MPS <= float(parts[2]) <= DEFAULT_V_MAX_MPS
        assert int(parts[3]) == 1

        st_fg = mgr.status()
        assert st_fg["tx_packets"] == 1
        assert st_fg["last_tx_command"]["protocol"] == "flightgear_csv"
    finally:
        mgr.shutdown()
        tx_rx.close()
        inj.close()


def test_run_hil_loopback_self_test_includes_closed_loop_tx_proof() -> None:
    """`run_hil_loopback_self_test()` must verify closed-loop TX (`tx_packets > 0`, zero `#84`)."""
    try:
        report = run_hil_loopback_self_test()
    except FileNotFoundError as exc:
        if "xvs-ffi" in str(exc) or "xvs_core" in str(exc):
            pytest.skip(f"Compiled xvs-ffi library not built yet: {exc}")
        raise
    assert report["pass"] is True
    assert report["gpl_free"] is True
    assert len(report["checks"]) == 4  # Backwards compatibility with v4.5 contract
    cl = report["closed_loop_tx"]
    assert cl["pass"] is True
    assert cl["tx_packets"] > 0
    assert cl["mavlink2_tx_packets"] >= 4
    assert cl["flightgear_tx_packets"] >= 1
    assert cl["ekf3_42007_emitted"] is True
    assert cl["msg84_emitted"] is False
    assert cl["bad_telemetry_and_msg84_rejected"] is True


def test_sitl_proof_script_twin_contract_and_live_proof_artifact(tmp_path: Path) -> None:
    """Verify `run_live_arduplane_sitl_proof.py` version parser, honest twin contract, and live proof JSON."""
    assert parse_arduplane_version("APMVERSION: ArduPlane V4.7.3") == (4, 7, 3)
    assert parse_arduplane_version("4.7.3-FIRMWARE_VERSION_TYPE_OFFICIAL") == (4, 7, 3)
    assert parse_arduplane_version("ArduPlane V4.6.3") < MIN_ARDUPLANE_SITL_VERSION
    assert parse_arduplane_version("invalid") is None

    twin_out = tmp_path / "twin_proof.json"
    twin_rep = run_python_6dof_udp_twin_proof(out_path=twin_out)
    assert twin_out.is_file()
    loaded_twin = json.loads(twin_out.read_text(encoding="utf-8"))
    assert loaded_twin["pass"] is True
    assert loaded_twin["simulator_backend"] == "PYTHON_6DOF_FIXEDWING_UDP_TWIN"
    assert loaded_twin["arduplane_verified"] is False
    assert loaded_twin["gpl_free_openxvs_codec"] is True
    assert loaded_twin["pymavlink_imported"] is False
    assert loaded_twin["msg84_ever_emitted"] is False
    assert loaded_twin["telemetry_summary"]["tx_packets"] > 0
    assert all(bool(v) for v in loaded_twin["checks"].values())
    assert twin_rep["pass"] is True

    # If the live WSL2 ArduPlane >= 4.7.3 proof artifact exists on disk, validate its invariants too
    if DEFAULT_PROOF_OUT_PATH.is_file():
        live_doc = json.loads(DEFAULT_PROOF_OUT_PATH.read_text(encoding="utf-8"))
        assert live_doc["pass"] is True
        assert live_doc["gpl_free_openxvs_codec"] is True
        assert live_doc["pymavlink_imported"] is False
        assert live_doc["msg84_ever_emitted"] is False
        assert live_doc["telemetry_summary"]["tx_packets"] > 0
        if live_doc["arduplane_verified"] is True:
            assert live_doc["simulator_backend"] == "WSL2_ARDUPLANE_OFFICIAL_SITL"
            assert tuple(live_doc["arduplane_version_tuple"]) >= MIN_ARDUPLANE_SITL_VERSION
            assert live_doc["wsl_pid_cleanly_terminated"] is True
            assert live_doc["flight_response"]["ack_42007_accepted_count"] >= 2
