"""Wave 2 Security & Flight-Safety Tests covering Campaign 2 Bugs B1..B8.

Verifies all 8 pure-software/security fixes discovered in
`outputs/sitl_proof/adversarial_c2/CAMPAIGN_2_SITL_TRUTH.md`:
  - B1 [CRITICAL]: Oversized UDP datagrams (4,097..65,507 B) & EMSGSIZE/WinError 10040 never kill `_rx_loop`.
  - B2 [HIGH]: MAVLink 2.0 signature-stripping downgrade blocked (`require_signature=True` when `secret_key` set).
  - B3 [HIGH]: `DatumAlignmentGuard` fails closed (`DATUM_MISMATCH_INHIBITED`) when `visual_alt_m is None`.
  - B4 [HIGH]: `#82 SET_ATTITUDE_TARGET` fixed-wing stall protection (climb/underspeed thrust floor + `0x40` TECS mask).
  - B5 [MEDIUM]: Per-msgid `time_boot_ms` monotonicity / anti-replay with uint32 wraparound & 3-tick reboot confirm.
  - B6 [MEDIUM]: Non-shadowed stale-telemetry TX guard + `send_raw_tx_frame(..., allow_when_stale=False)` enforcement.
  - B7 [MEDIUM]: Zero-GPL `#74 VFR_HUD` (`CRC_EXTRA=20`, `<ffffhH`) decoding & truthful stall-unmasked IAS vs GS.
  - B8 [MEDIUM]: `MavGlobalPositionInt` altitude (`[-1 km, +100 km]`) & heading (`[0, 36000]` or `65535`) bounds.
"""

from __future__ import annotations

import errno
import math
from pathlib import Path
import socket
import struct
import time
from typing import Any

import pytest

from xvs.v2.studio_bridge.hil_bridge import (
    DATUM_REASON_MISMATCH_INHIBITED,
    DATUM_REASON_OK,
    KSFO_28L_GEOID18_UNDULATION_M,
    DatumAlignmentGuard,
    NonFlightSimOnlyError,
    ShadowModeTelemetryLogger,
    ShadowModeTxInhibitedError,
    StaleTelemetryError,
    StudioHilBridgeManager,
    msl_to_wgs84_hae,
)
from xvs.v2.xvs_mavlink.pod_daa_codec import (
    CRC_EXTRA_BY_MSGID,
    MAVLINK_V2_STX,
    MSGID_ATTITUDE,
    MSGID_COMMAND_LONG,
    MSGID_GLOBAL_POSITION_INT,
    MSGID_SET_ATTITUDE_TARGET,
    MSGID_SET_POSITION_TARGET_LOCAL_NED,
    MSGID_VFR_HUD,
    MavAttitude,
    MavCommandLong,
    MavGlobalPositionInt,
    MavHeartbeat,
    MavSetAttitudeTarget,
    MavVfrHud,
    decode_frame,
    encode_frame,
    x25_crc16,
)
from xvs.v2.xvs_rt_bridge.arduplane_bridge import (
    ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
    ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES,
    ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE,
    DEFAULT_MAX_BANK_RAD,
    DEFAULT_MAX_PITCH_RAD,
    DEFAULT_V_MIN_MPS,
    EKF3_SOURCE_GNSS,
    EKF3_SOURCE_VISION_OPTICAL,
    MIN_CLIMB_THRUST_NORM,
    MIN_SAFE_THRUST_NORM,
    Ekf3SourceSwitchLatch,
    build_set_attitude_target,
    decode_arduplane_frame,
    encode_heartbeat_frame,
    encode_set_attitude_target_frame,
)


def _alloc_free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _build_raw_valid_crc_gpi_frame(
    *,
    time_boot_ms: int = 1000,
    lat_e7: int = 376_117_200,
    lon_e7: int = -1_223_583_670,
    alt_mm: int = 180_000,
    relative_alt_mm: int = 175_000,
    vx_cms: int = 1500,
    vy_cms: int = -2000,
    vz_cms: int = 0,
    hdg_cdeg: int = 29800,
    seq: int = 1,
) -> bytes:
    """Pack a raw valid-CRC MAVLink 2.0 `#33` wire frame bypassing dataclass validation for adversarial tests."""
    payload = struct.pack(
        "<IiiiihhhH",
        int(time_boot_ms) & 0xFFFFFFFF,
        int(lat_e7),
        int(lon_e7),
        int(alt_mm),
        int(relative_alt_mm),
        int(vx_cms),
        int(vy_cms),
        int(vz_cms),
        int(hdg_cdeg) & 0xFFFF,
    )
    header_body = struct.pack(
        "<BBBBBBBBB",
        len(payload),
        0,
        0,
        seq & 0xFF,
        1,
        1,
        MSGID_GLOBAL_POSITION_INT & 0xFF,
        (MSGID_GLOBAL_POSITION_INT >> 8) & 0xFF,
        (MSGID_GLOBAL_POSITION_INT >> 16) & 0xFF,
    )
    crc = x25_crc16(header_body + payload + bytes([CRC_EXTRA_BY_MSGID[MSGID_GLOBAL_POSITION_INT]]))
    return bytes([MAVLINK_V2_STX]) + header_body + payload + struct.pack("<H", crc)


# ---- B1: Oversized UDP Datagram RX Thread DoS Protection ----


def test_b1_oversized_udp_datagrams_do_not_kill_rx_thread_mavlink_and_flightgear() -> None:
    """Oversized (4,097..65,507 B) & empty UDP datagrams are rejected without terminating `_rx_loop`."""
    mgr = StudioHilBridgeManager(non_flight_sim_only=False)
    tx_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_rx.bind(("127.0.0.1", 0))
        tx_rx.settimeout(1.5)
        tx_port = int(tx_rx.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        mgr.connect(mode="mavlink2_udp", rx_port=rx_port, tx_port=tx_port, tx_command_mode="set_attitude_target")
        assert mgr._thread is not None and mgr._thread.is_alive()

        # Send hostile oversized datagrams (4,097 B, 16,384 B, 65,507 B max IPv4 UDP), 0 B, and padded frame
        valid_att_0 = encode_frame(MavAttitude(time_boot_ms=100, roll=0.10, pitch=0.0, yaw=0.0), seq=1, sysid=1, compid=1)
        hostile_payloads = [
            b"",
            b"\xAA" * 4097,
            b"\xFD" + (b"\x00" * 16383),
            b"\x55" * 65507,
            valid_att_0 + (b"\x00" * 4096),
        ]
        for payload in hostile_payloads:
            inj.sendto(payload, ("127.0.0.1", rx_port))
            time.sleep(0.015)

        time.sleep(0.1)
        assert mgr._thread is not None and mgr._thread.is_alive(), "RX thread died after oversized UDP flood!"
        st_after_flood = mgr.status()
        assert st_after_flood["rejected_packets"] >= len(hostile_payloads)
        assert st_after_flood["attitude_count"] == 0
        assert st_after_flood["tx_packets"] == 0

        # Subsequent legitimate frame MUST still be processed cleanly and emit autonomous TX
        valid_att_1 = encode_frame(MavAttitude(time_boot_ms=200, roll=0.15, pitch=0.02, yaw=1.0), seq=2, sysid=1, compid=1)
        inj.sendto(valid_att_1, ("127.0.0.1", rx_port))
        pkt, _ = tx_rx.recvfrom(512)
        dec = decode_frame(pkt)
        assert dec.msgid == MSGID_SET_ATTITUDE_TARGET
        assert mgr._thread is not None and mgr._thread.is_alive()
        assert mgr.status()["attitude_count"] == 1

        # Same resilience check in FlightGear UDP mode
        fg_rx_port = _alloc_free_udp_port()
        mgr.connect(mode="flightgear_udp", rx_port=fg_rx_port, tx_port=tx_port)
        inj.sendto(b"9" * 4097, ("127.0.0.1", fg_rx_port))
        inj.sendto(b"9" * 16384, ("127.0.0.1", fg_rx_port))
        time.sleep(0.08)
        assert mgr._thread is not None and mgr._thread.is_alive()
        assert mgr.status()["rejected_packets"] >= 2
        inj.sendto(
            b"37.6117200,-122.3583670,180.000,298.000,-1.000,2.000,60.000,15.000,-22.000,0.000\n",
            ("127.0.0.1", fg_rx_port),
        )
        fg_tx, _ = tx_rx.recvfrom(256)
        assert len(fg_tx.decode("ascii").strip().split(",")) == 4
    finally:
        mgr.shutdown()
        tx_rx.close()
        inj.close()


def test_b1_rx_loop_survives_emsgsize_winerror_10040_and_transient_oserrors() -> None:
    """Direct `_rx_loop` resilience when `recvfrom` raises `EMSGSIZE`, WinError `10040`, or transient errors."""
    mgr = StudioHilBridgeManager(non_flight_sim_only=False)
    mgr._mode = "mavlink2_udp"
    mgr._tx_enabled = False
    mgr._stop_event.clear()

    valid_att = encode_frame(MavAttitude(time_boot_ms=500, roll=0.2, pitch=0.0, yaw=0.0), seq=1, sysid=1, compid=1)

    win_10040 = OSError("WSAEMSGSIZE")
    win_10040.winerror = 10040  # type: ignore[attr-defined]
    win_10035 = OSError("WSAEWOULDBLOCK")
    win_10035.winerror = 10035  # type: ignore[attr-defined]

    events: list[Any] = [
        OSError(errno.EMSGSIZE, "Message too long"),
        win_10040,
        win_10035,
        ConnectionResetError("WSAECONNRESET"),
        (valid_att, ("127.0.0.1", 12345)),
    ]

    class _MockSock:
        def __init__(self) -> None:
            self.idx = 0

        def fileno(self) -> int:
            return 42

        def recvfrom(self, bufsize: int) -> tuple[bytes, tuple[str, int]]:
            assert bufsize == 65535
            if self.idx >= len(events):
                mgr._stop_event.set()
                raise socket.timeout()
            ev = events[self.idx]
            self.idx += 1
            if isinstance(ev, BaseException):
                raise ev
            return ev

    mgr._rx_loop(_MockSock(), None, "mavlink2_udp")  # type: ignore[arg-type]
    st = mgr.status()
    assert st["rejected_packets"] == 2  # EMSGSIZE + WinError 10040
    assert st["attitude_count"] == 1
    assert st["state_ned"]["roll_deg"] == pytest.approx(math.degrees(0.2), abs=1e-4)


# ---- B2: MAVLink 2.0 Signature-Stripping Downgrade Prevention ----


def test_b2_signed_bridge_strictly_rejects_unsigned_frames_10_of_10(tmp_path: Path) -> None:
    """When `secret_key` is configured, 10/10 unsigned frames are rejected fail-closed while signed pass."""
    key = bytes(range(32))
    wrong_key = bytes(range(32, 64))
    mgr = StudioHilBridgeManager(non_flight_sim_only=False)
    tx_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_rx.bind(("127.0.0.1", 0))
        tx_rx.settimeout(1.5)
        tx_port = int(tx_rx.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        mgr.connect(
            mode="mavlink2_udp",
            rx_port=rx_port,
            tx_port=tx_port,
            secret_key=key,
            stale_telemetry_timeout_s=0.3,
        )

        # Inject 10 unsigned valid-CRC #30 ATTITUDE frames (signature-stripping downgrade attack)
        for i in range(10):
            unsigned_att = encode_frame(
                MavAttitude(time_boot_ms=1000 + i * 50, roll=0.45, pitch=-0.25, yaw=1.0),
                seq=i,
                sysid=1,
                compid=1,
            )
            assert unsigned_att[2] == 0x00  # incompat_flags unsigned
            inj.sendto(unsigned_att, ("127.0.0.1", rx_port))
            time.sleep(0.01)

        # Also inject unsigned #33, unsigned #74, unsigned non-subscribed frame, and wrong-key signed #30
        inj.sendto(
            encode_frame(MavVfrHud(airspeed=9.0, groundspeed=9.0, heading=10, throttle=10), seq=20, sysid=1, compid=1),
            ("127.0.0.1", rx_port),
        )
        inj.sendto(
            encode_frame(
                MavAttitude(time_boot_ms=2000, roll=0.45, pitch=0.0, yaw=0.0),
                seq=21,
                sysid=1,
                compid=1,
                secret_key=wrong_key,
                timestamp_48=100,
            ),
            ("127.0.0.1", rx_port),
        )
        # Non-subscribed unsigned msgid=1 (SYS_STATUS)
        hdr_unsub = struct.pack("<BBBBBBBBB", 1, 0x00, 0, 22, 1, 1, 1, 0, 0)
        inj.sendto(bytes([MAVLINK_V2_STX]) + hdr_unsub + b"\x00\x00\x00", ("127.0.0.1", rx_port))

        time.sleep(0.15)
        st = mgr.status()
        assert st["rejected_packets"] >= 13
        assert st["attitude_count"] == 0
        assert st["vfr_hud_count"] == 0
        assert st["tx_packets"] == 0
        assert st["state_ned"]["roll_deg"] == pytest.approx(0.0)
        assert mgr._last_rx_mono_s is None

        # Legitimate signed frame passes and produces signed TX
        signed_att = encode_frame(
            MavAttitude(time_boot_ms=2100, roll=0.12, pitch=0.01, yaw=0.5),
            seq=30,
            sysid=1,
            compid=1,
            secret_key=key,
            timestamp_48=500,
        )
        inj.sendto(signed_att, ("127.0.0.1", rx_port))
        tx_pkt, _ = tx_rx.recvfrom(512)
        dec_tx = decode_frame(tx_pkt, secret_key=key, require_signature=True)
        assert dec_tx.msgid == MSGID_SET_ATTITUDE_TARGET
        assert dec_tx.signature_verified is True
        assert mgr.status()["attitude_count"] == 1
        assert mgr.status()["state_ned"]["roll_deg"] == pytest.approx(math.degrees(0.12), abs=1e-3)

        # Signed shadow tap must ALSO reject unsigned frames
        shadow_port = _alloc_free_udp_port()
        with ShadowModeTelemetryLogger(tmp_path / "signed_shadow.ndjson") as slog:
            mgr.connect_shadow_tap(shadow_logger=slog, mode="mavlink2_udp", rx_port=shadow_port, secret_key=key)
            inj.sendto(
                encode_frame(MavAttitude(time_boot_ms=100, roll=0.3, pitch=0.0, yaw=0.0), seq=1, sysid=1, compid=1),
                ("127.0.0.1", shadow_port),
            )
            time.sleep(0.08)
            assert slog.record_count == 0
            assert mgr.status()["rejected_packets"] >= 1
            mgr.shutdown()
    finally:
        mgr.shutdown()
        tx_rx.close()
        inj.close()


# ---- B3: DatumAlignmentGuard Fail-Closed When visual_alt_m is None ----


def test_b3_datum_guard_fails_closed_when_visual_alt_is_none() -> None:
    """Attaching `DatumAlignmentGuard` with `visual_alt_m=None` MUST inhibit EKF3 `42007` fail-closed."""
    # 1. Direct guard + latch check
    guard = DatumAlignmentGuard(geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M)
    latch = Ekf3SourceSwitchLatch()
    dec_direct = guard.evaluate_ekf3_switch(z_visual_m=None, z_baro_msl_m=180.0)
    assert dec_direct.allowed is False
    assert dec_direct.reason_code == DATUM_REASON_MISMATCH_INHIBITED
    assert "Visual altitude unavailable" in dec_direct.detail
    assert dec_direct.z_visual_input_m is None
    assert dec_direct.z_baro_msl_m == pytest.approx(180.0)

    ekf_st, dec_latch = guard.gate_ekf3_switch_latch(
        latch,
        timestamp_s=1.0,
        gnss_healthy=True,
        spoof_alarm=True,
        z_visual_m=None,
        z_baro_msl_m=180.0,
    )
    assert dec_latch.allowed is False
    assert dec_latch.reason_code == DATUM_REASON_MISMATCH_INHIBITED
    assert ekf_st.active_source == EKF3_SOURCE_GNSS
    assert ekf_st.mav_cmd_id is None

    # 2. Live UDP StudioHilBridgeManager probe (exact C2-04(f) scenario)
    mgr = StudioHilBridgeManager(
        non_flight_sim_only=False,
        datum_guard=DatumAlignmentGuard(geoid_undulation_m=None),
    )
    tx_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_rx.bind(("127.0.0.1", 0))
        tx_rx.settimeout(1.0)
        tx_port = int(tx_rx.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        mgr.connect(mode="mavlink2_udp", rx_port=rx_port, tx_port=tx_port, tx_command_mode="set_attitude_target")
        # Intentionally DO NOT call update_visual_altitude -> _visual_alt_m stays None!
        mgr.set_autonomy_triggers(gnss_spoof_alarm=True, gnss_healthy=True)

        gpi_1 = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=1000,
                lat_e7=376_117_200,
                lon_e7=-1_223_583_670,
                alt_mm=180_000,
                relative_alt_mm=175_000,
                vx_cms=1500,
                vy_cms=-2200,
                vz_cms=0,
                hdg_cdeg=29800,
            ),
            seq=1,
            sysid=1,
            compid=1,
        )
        inj.sendto(gpi_1, ("127.0.0.1", rx_port))
        pkt1, _ = tx_rx.recvfrom(512)
        dec1 = decode_frame(pkt1)
        assert dec1.msgid == MSGID_SET_ATTITUDE_TARGET  # NO #76(42007) emitted!

        st = mgr.status()
        assert st["datum_guard_active"] is True
        assert st["ekf3_active_source"] == EKF3_SOURCE_GNSS
        assert st["datum_inhibited_switches"] == 1
        assert MSGID_COMMAND_LONG not in st["tx_msgids_emitted"]
        assert st["last_datum_decision"] is not None
        assert st["last_datum_decision"]["allowed"] is False
        assert st["last_datum_decision"]["reason_code"] == DATUM_REASON_MISMATCH_INHIBITED
        assert "Visual altitude unavailable" in st["last_datum_decision"]["detail"]

        # Supply aligned visual altitude -> switch now authorized cleanly
        cruise_hae = msl_to_wgs84_hae(180.0, KSFO_28L_GEOID18_UNDULATION_M)
        mgr.update_visual_altitude(
            visual_alt_m=cruise_hae + 0.2,
            visual_datum="WGS84_HAE",
            geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M,
        )
        gpi_2 = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=1100,
                lat_e7=376_117_300,
                lon_e7=-1_223_583_700,
                alt_mm=180_000,
                relative_alt_mm=175_000,
                vx_cms=1500,
                vy_cms=-2200,
                vz_cms=0,
                hdg_cdeg=29800,
            ),
            seq=2,
            sysid=1,
            compid=1,
        )
        inj.sendto(gpi_2, ("127.0.0.1", rx_port))
        emitted_42007 = False
        for _ in range(2):
            pkt, _ = tx_rx.recvfrom(512)
            dec = decode_frame(pkt)
            if isinstance(dec.message, MavCommandLong) and dec.message.command == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET:
                assert dec.message.param1 == pytest.approx(float(EKF3_SOURCE_VISION_OPTICAL))
                emitted_42007 = True
        assert emitted_42007 is True
        st_ok = mgr.status()
        assert st_ok["ekf3_active_source"] == EKF3_SOURCE_VISION_OPTICAL
        assert st_ok["last_datum_decision"]["allowed"] is True
        assert st_ok["last_datum_decision"]["reason_code"] == DATUM_REASON_OK
    finally:
        mgr.shutdown()
        tx_rx.close()
        inj.close()


# ---- B4: Fixed-Wing #82 SET_ATTITUDE_TARGET Stall Protection ----


def test_b4_set_attitude_target_prevents_climb_and_underspeed_stall() -> None:
    """`build_set_attitude_target` enforces active wire thrust floor (`type_mask=0x07`) on climb/underspeed.

    Live WSL2 `ArduPlane 4.7.3` SITL verification proved that setting bit `0x40` (`THROTTLE_IGNORE`)
    causes Guided attitude mode to drop throttle to `0%` (`6.7 m/s` stall), whereas `type_mask=0x07`
    with the pitch-scaled `thrust >= 0.65` floor holds `>= 18.2 m/s` stall-free. Explicit
    `preserve_tecs_throttle=True` still sets bit `0x40` when requested.
    """
    # 1. Exact C2-07 7.1b stall trigger: +60 deg roll, +45 deg pitch, thrust=0.0, airspeed=5.0 m/s
    cmd_stall = build_set_attitude_target(
        roll_rad=math.radians(60.0),
        pitch_rad=math.radians(45.0),
        yaw_rad=math.radians(298.0),
        thrust_norm=0.0,
        airspeed_mps=5.0,
    )
    assert cmd_stall.bank_clamped is True
    assert cmd_stall.airspeed_clamped is True
    assert cmd_stall.thrust_clamped is True
    assert cmd_stall.tecs_protected is False
    assert abs(cmd_stall.roll_rad - DEFAULT_MAX_BANK_RAD) < 1e-12
    assert abs(cmd_stall.pitch_rad - DEFAULT_MAX_PITCH_RAD) < 1e-12
    assert abs(cmd_stall.airspeed_mps - DEFAULT_V_MIN_MPS) < 1e-12
    assert cmd_stall.thrust_norm >= MIN_CLIMB_THRUST_NORM - 1e-12
    assert cmd_stall.type_mask == ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES

    wire = encode_set_attitude_target_frame(cmd_stall, seq=7)
    dec = decode_arduplane_frame(wire)
    assert isinstance(dec.message, MavSetAttitudeTarget)
    assert dec.message.thrust == pytest.approx(cmd_stall.thrust_norm, abs=1e-6)
    assert (dec.message.type_mask & ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE) == 0

    # 2. Nose-up climb (+10 deg) with zero thrust at cruise speed -> active thrust floored, mask=0x07
    cmd_climb = build_set_attitude_target(
        roll_rad=0.0,
        pitch_rad=math.radians(10.0),
        yaw_rad=0.0,
        thrust_norm=0.0,
        airspeed_mps=25.0,
    )
    assert cmd_climb.thrust_clamped is True
    assert cmd_climb.tecs_protected is False
    assert cmd_climb.thrust_norm >= 0.50 - 1e-12
    assert cmd_climb.type_mask == ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES

    # 3. Level flight underspeed (< V_min) with zero thrust -> floored to MIN_SAFE_THRUST_NORM, mask=0x07
    cmd_slow = build_set_attitude_target(
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
        thrust_norm=0.0,
        airspeed_mps=12.0,
    )
    assert cmd_slow.airspeed_clamped is True
    assert cmd_slow.thrust_clamped is True
    assert cmd_slow.tecs_protected is False
    assert cmd_slow.thrust_norm == pytest.approx(MIN_SAFE_THRUST_NORM)
    assert cmd_slow.type_mask == ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES

    # 4. Explicit preserve_tecs_throttle=True at nominal cruise -> bit 0x40 set without mutating thrust
    cmd_tecs = build_set_attitude_target(
        roll_rad=0.1,
        pitch_rad=0.0,
        yaw_rad=0.0,
        thrust_norm=0.55,
        airspeed_mps=24.0,
        preserve_tecs_throttle=True,
    )
    assert cmd_tecs.thrust_clamped is False
    assert cmd_tecs.tecs_protected is True
    assert cmd_tecs.thrust_norm == pytest.approx(0.55)
    assert (cmd_tecs.type_mask & ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE) != 0

    # 5. Controlled descent (pitch < 0, airspeed >= V_min, thrust=0.0) remains allowed without forced thrust
    cmd_dive = build_set_attitude_target(
        roll_rad=0.0,
        pitch_rad=math.radians(-10.0),
        yaw_rad=0.0,
        thrust_norm=0.0,
        airspeed_mps=25.0,
    )
    assert cmd_dive.thrust_clamped is False
    assert cmd_dive.tecs_protected is False
    assert cmd_dive.thrust_norm == pytest.approx(0.0)
    assert cmd_dive.type_mask == ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES


# ---- B5: time_boot_ms Monotonicity / Anti-Replay Check ----


def test_b5_time_boot_ms_monotonicity_rejects_reordered_replays_and_handles_wrap_and_reboot() -> None:
    """Out-of-order/replayed `time_boot_ms` frames are rejected without state rollback or freshness refresh."""
    mgr = StudioHilBridgeManager(non_flight_sim_only=False)
    tx_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_rx.bind(("127.0.0.1", 0))
        tx_rx.settimeout(1.0)
        tx_port = int(tx_rx.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        mgr.connect(
            mode="mavlink2_udp",
            rx_port=rx_port,
            tx_port=tx_port,
            stale_telemetry_timeout_s=0.25,
        )

        # Fresh attitude at t=50_000 ms (+12 deg roll)
        inj.sendto(
            encode_frame(MavAttitude(time_boot_ms=50_000, roll=math.radians(12.0), pitch=0.0, yaw=0.0), seq=1, sysid=1, compid=1),
            ("127.0.0.1", rx_port),
        )
        tx_rx.recvfrom(512)
        assert mgr.status()["state_ned"]["roll_deg"] == pytest.approx(12.0, abs=1e-3)
        tx_before = int(mgr.status()["tx_packets"])

        # Let link go stale (> 0.25 s)
        time.sleep(0.32)
        assert mgr.is_telemetry_stale() is True

        # Inject delayed reordered packet (49_500 ms) AND ancient C2-05 replay (10 ms, repeated 5x)
        stale_frames = [
            encode_frame(MavAttitude(time_boot_ms=49_500, roll=math.radians(-25.0), pitch=0.0, yaw=0.0), seq=2, sysid=1, compid=1),
            *[
                encode_frame(MavAttitude(time_boot_ms=10, roll=math.radians(-29.5), pitch=0.0, yaw=0.0), seq=3 + i, sysid=1, compid=1)
                for i in range(5)
            ],
        ]
        for f in stale_frames:
            inj.sendto(f, ("127.0.0.1", rx_port))
            time.sleep(0.015)

        time.sleep(0.08)
        st_stale = mgr.status()
        assert st_stale["rejected_packets"] >= len(stale_frames)
        assert st_stale["attitude_count"] == 1
        assert st_stale["state_ned"]["roll_deg"] == pytest.approx(12.0, abs=1e-3)
        assert st_stale["tx_packets"] == tx_before
        assert mgr.is_telemetry_stale() is True  # Replays MUST NOT refresh _last_rx_mono_s!

        # Genuine autopilot reboot: 3 consecutive strictly-increasing fresh ticks (100 -> 150 -> 200 ms)
        for idx, boot_ts in enumerate((100, 150, 200)):
            inj.sendto(
                encode_frame(
                    MavAttitude(time_boot_ms=boot_ts, roll=math.radians(5.0 + idx), pitch=0.0, yaw=0.0),
                    seq=20 + idx,
                    sysid=1,
                    compid=1,
                ),
                ("127.0.0.1", rx_port),
            )
            time.sleep(0.02)

        pkt_reboot, _ = tx_rx.recvfrom(512)
        assert decode_frame(pkt_reboot).msgid == MSGID_SET_ATTITUDE_TARGET
        assert mgr.status()["attitude_count"] == 2  # Only the 3rd confirming tick accepted
        assert mgr.status()["state_ned"]["roll_deg"] == pytest.approx(7.0, abs=1e-3)
        assert mgr.is_telemetry_stale() is False

        # Direct uint32 wraparound & pre-wrap rejection test + reconnect reset
        with mgr._lock:
            mgr._last_rx_time_boot_ms.clear()
            mgr._reboot_candidate_boot_ms.clear()
            assert mgr._accept_rx_time_boot_ms_locked(MSGID_GLOBAL_POSITION_INT, 0xFFFF_FFF0) is True
            assert mgr._accept_rx_time_boot_ms_locked(MSGID_GLOBAL_POSITION_INT, 0x0000_0020) is True
            # Stale pre-wrap frame arriving after wraparound must be rejected
            assert mgr._accept_rx_time_boot_ms_locked(MSGID_GLOBAL_POSITION_INT, 0xFFFF_FF80) is False

        # Reconnect resets time_boot_ms map cleanly
        rx_port_2 = _alloc_free_udp_port()
        mgr.connect(mode="mavlink2_udp", rx_port=rx_port_2, tx_port=tx_port)
        assert mgr._last_rx_time_boot_ms == {}
        inj.sendto(
            encode_frame(MavAttitude(time_boot_ms=5, roll=math.radians(3.0), pitch=0.0, yaw=0.0), seq=1, sysid=1, compid=1),
            ("127.0.0.1", rx_port_2),
        )
        tx_rx.recvfrom(512)
        assert mgr.status()["attitude_count"] == 1
    finally:
        mgr.shutdown()
        tx_rx.close()
        inj.close()


# ---- B6: Stale-Telemetry Guard on send_raw_tx_frame & Non-Shadowed RX->TX Gate ----


def test_b6_send_raw_tx_frame_enforces_stale_guard_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`send_raw_tx_frame` blocks stale TX fail-closed unless `allow_when_stale=True`."""
    mgr = StudioHilBridgeManager(non_flight_sim_only=False)
    tx_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_rx.bind(("127.0.0.1", 0))
        tx_rx.settimeout(0.5)
        tx_port = int(tx_rx.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        mgr.connect(
            mode="mavlink2_udp",
            rx_port=rx_port,
            tx_port=tx_port,
            tx_enabled=False,
            stale_telemetry_timeout_s=0.20,
        )
        hb_wire = encode_heartbeat_frame(seq=1)

        # Immediately after connect (within stale timeout), raw TX is allowed
        assert mgr.send_raw_tx_frame(hb_wire) == len(hb_wire)
        tx_rx.recvfrom(256)

        # Wait past stale_telemetry_timeout_s with zero valid RX -> stale fail-closed!
        time.sleep(0.28)
        assert mgr.is_telemetry_stale() is True
        tx_count_before = int(mgr.status()["tx_packets"])
        with pytest.raises(StaleTelemetryError, match="stale"):
            mgr.send_raw_tx_frame(hb_wire)
        assert mgr.status()["tx_packets"] == tx_count_before

        # Explicit opt-in `allow_when_stale=True` succeeds (e.g. bootstrap heartbeat)
        assert mgr.send_raw_tx_frame(hb_wire, allow_when_stale=True) == len(hb_wire)
        tx_rx.recvfrom(256)
        assert mgr.status()["tx_packets"] == tx_count_before + 1

        # Hard safety locks (#84 and NON_FLIGHT_SIM_ONLY=1) STILL override even with allow_when_stale=True
        raw_84 = (
            bytes([MAVLINK_V2_STX])
            + struct.pack("<BBBBBBBBB", 1, 0, 0, 1, 1, 191, MSGID_SET_POSITION_TARGET_LOCAL_NED, 0, 0)
            + b"\x00\x00\x00"
        )
        with pytest.raises(ValueError, match="#84"):
            mgr.send_raw_tx_frame(raw_84, allow_when_stale=True)

        # Fresh telemetry restores link -> default send_raw_tx_frame succeeds again
        inj.sendto(
            encode_frame(MavAttitude(time_boot_ms=100, roll=0.0, pitch=0.0, yaw=0.0), seq=1, sysid=1, compid=1),
            ("127.0.0.1", rx_port),
        )
        deadline = time.perf_counter() + 1.0
        while mgr.is_telemetry_stale() and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert mgr.is_telemetry_stale() is False
        assert mgr.send_raw_tx_frame(hb_wire) == len(hb_wire)
        tx_rx.recvfrom(256)

        # Direct _can_emit_autonomous_tx_locked verification when stale vs fresh
        with mgr._lock:
            mgr._tx_enabled = True
            assert mgr._can_emit_autonomous_tx_locked(mgr._tx_sock, time.monotonic()) is True
            assert mgr._can_emit_autonomous_tx_locked(mgr._tx_sock, time.monotonic() + 10.0) is False

        # NON_FLIGHT_SIM_ONLY=1 still blocks even with allow_when_stale=True
        monkeypatch.setenv("OPENXVS_NON_FLIGHT_SIM_ONLY", "1")
        mgr_locked = StudioHilBridgeManager()
        with pytest.raises(NonFlightSimOnlyError):
            mgr_locked.send_raw_tx_frame(hb_wire, allow_when_stale=True)
    finally:
        mgr.shutdown()
        tx_rx.close()
        inj.close()


# ---- B7: Zero-GPL #74 VFR_HUD Decoding & Truthful Stall-Unmasked Airspeed ----


def test_b7_vfr_hud_codec_and_hil_bridge_true_airspeed_without_stall_masking() -> None:
    """`#74 VFR_HUD` decodes true IAS separately from GS and never masks stalls with `max(18, gs)`."""
    assert CRC_EXTRA_BY_MSGID[MSGID_VFR_HUD] == 20

    # Unsigned & signed round-trip
    key = bytes(range(1, 33))
    vfr = MavVfrHud(airspeed=24.35, groundspeed=9.59, heading=298, throttle=68, alt=125.5, climb=1.25)
    assert len(vfr.pack_payload()) == 20
    wire_u = encode_frame(vfr, seq=4, sysid=1, compid=1)
    dec_u = decode_frame(wire_u)
    assert dec_u.msgid == MSGID_VFR_HUD
    assert isinstance(dec_u.message, MavVfrHud)
    assert dec_u.message.airspeed == pytest.approx(24.35, abs=1e-5)
    assert dec_u.message.groundspeed == pytest.approx(9.59, abs=1e-5)
    assert dec_u.message.heading == 298
    assert dec_u.message.throttle == 68
    assert dec_u.message.alt == pytest.approx(125.5, abs=1e-5)
    assert dec_u.message.climb == pytest.approx(1.25, abs=1e-5)

    wire_s = encode_frame(vfr, seq=5, sysid=1, compid=1, secret_key=key, timestamp_48=999)
    dec_s = decode_frame(wire_s, secret_key=key, require_signature=True)
    assert dec_s.signature_verified is True
    assert isinstance(dec_s.message, MavVfrHud)

    # Trailing-zero payload truncation (heading=0, throttle=0, alt=0, climb=0) round-trips cleanly
    vfr_zero_tail = MavVfrHud(airspeed=6.60, groundspeed=7.10, heading=0, throttle=0, alt=0.0, climb=0.0)
    dec_zt = decode_frame(encode_frame(vfr_zero_tail, seq=6, sysid=1, compid=1))
    assert isinstance(dec_zt.message, MavVfrHud)
    assert dec_zt.message.airspeed == pytest.approx(6.60, abs=1e-5)

    # Fail-closed validation on invalid floats & bounds
    with pytest.raises(ValueError):
        MavVfrHud(airspeed=float("nan"), groundspeed=10.0)
    with pytest.raises(ValueError):
        MavVfrHud(airspeed=10.0, groundspeed=float("inf"))
    with pytest.raises(ValueError):
        MavVfrHud(airspeed=-0.1, groundspeed=10.0)
    with pytest.raises(ValueError):
        MavVfrHud(airspeed=10.0, groundspeed=-1.0)
    with pytest.raises(ValueError):
        MavVfrHud(airspeed=20.0, groundspeed=20.0, heading=-1)
    with pytest.raises(ValueError):
        MavVfrHud(airspeed=20.0, groundspeed=20.0, heading=361)
    with pytest.raises(ValueError):
        MavVfrHud(airspeed=20.0, groundspeed=20.0, throttle=-1)
    with pytest.raises(ValueError):
        MavVfrHud(airspeed=20.0, groundspeed=20.0, throttle=101)

    # Live bridge telemetry check: truthful fallback GS -> VFR_HUD IAS separation -> live stall surfacing
    mgr = StudioHilBridgeManager(non_flight_sim_only=False)
    tx_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_rx.bind(("127.0.0.1", 0))
        tx_rx.settimeout(1.0)
        tx_port = int(tx_rx.getsockname()[1])
        rx_port = _alloc_free_udp_port()

        mgr.connect(mode="mavlink2_udp", rx_port=rx_port, tx_port=tx_port, tx_command_mode="set_attitude_target")

        # 1. Before any #74 VFR_HUD: #33 with 9.5 m/s headwind GS must NOT fake 18.0 m/s!
        gpi_low = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=100,
                lat_e7=376_117_200,
                lon_e7=-1_223_583_670,
                alt_mm=120_000,
                relative_alt_mm=115_000,
                vx_cms=950,
                vy_cms=0,
                vz_cms=0,
                hdg_cdeg=0,
            ),
            seq=1,
            sysid=1,
            compid=1,
        )
        inj.sendto(gpi_low, ("127.0.0.1", rx_port))
        pkt_low, _ = tx_rx.recvfrom(512)
        dec_low = decode_frame(pkt_low)
        assert isinstance(dec_low.message, MavSetAttitudeTarget)
        # Underspeed (<18 m/s) commands active climb-recovery thrust (0.80) with type_mask=0x07
        # (never set 0x40 THROTTLE_IGNORE, which zeroes throttle on live ArduPlane 4.7.3 Guided mode!)
        assert dec_low.message.type_mask == ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES
        assert dec_low.message.thrust == pytest.approx(0.80, abs=1e-5)

        st1 = mgr.status()
        assert st1["airspeed_source"] == "groundspeed_fallback"
        assert st1["state_ned"]["groundspeed_mps"] == pytest.approx(9.5, abs=1e-3)
        assert st1["state_ned"]["airspeed_mps"] == pytest.approx(9.5, abs=1e-3)

        # 2. #74 VFR_HUD arrives under 15 m/s headwind (IAS=24.5 m/s, GS=9.5 m/s)
        inj.sendto(
            encode_frame(
                MavVfrHud(airspeed=24.5, groundspeed=9.5, heading=298, throttle=65, alt=120.0, climb=0.5),
                seq=2,
                sysid=1,
                compid=1,
            ),
            ("127.0.0.1", rx_port),
        )
        deadline = time.perf_counter() + 1.0
        while mgr.status()["vfr_hud_count"] < 1 and time.perf_counter() < deadline:
            time.sleep(0.01)
        st2 = mgr.status()
        assert st2["vfr_hud_count"] == 1
        assert st2["airspeed_source"] == "vfr_hud"
        assert st2["state_ned"]["airspeed_mps"] == pytest.approx(24.5, abs=1e-3)
        assert st2["state_ned"]["groundspeed_mps"] == pytest.approx(9.5, abs=1e-3)

        # 3. Subsequent #33 GPI tailwind (GS=37.6 m/s) updates groundspeed_mps ONLY, preserving true VFR_HUD IAS
        inj.sendto(
            encode_frame(
                MavGlobalPositionInt(
                    time_boot_ms=200,
                    lat_e7=376_117_300,
                    lon_e7=-1_223_583_700,
                    alt_mm=120_000,
                    relative_alt_mm=115_000,
                    vx_cms=3760,
                    vy_cms=0,
                    vz_cms=0,
                    hdg_cdeg=0,
                ),
                seq=3,
                sysid=1,
                compid=1,
            ),
            ("127.0.0.1", rx_port),
        )
        tx_rx.recvfrom(512)
        st3 = mgr.status()
        assert st3["state_ned"]["airspeed_mps"] == pytest.approx(24.5, abs=1e-3)
        assert st3["state_ned"]["groundspeed_mps"] == pytest.approx(37.6, abs=1e-3)

        # 4. Real stall (VFR_HUD IAS = 6.60 m/s) is surfaced accurately without clamping to 18 m/s!
        inj.sendto(
            encode_frame(
                MavVfrHud(airspeed=6.60, groundspeed=7.10, heading=298, throttle=0, alt=110.0, climb=-5.0),
                seq=4,
                sysid=1,
                compid=1,
            ),
            ("127.0.0.1", rx_port),
        )
        deadline = time.perf_counter() + 1.0
        while mgr.status()["vfr_hud_count"] < 2 and time.perf_counter() < deadline:
            time.sleep(0.01)
        st4 = mgr.status()
        assert st4["state_ned"]["airspeed_mps"] == pytest.approx(6.60, abs=1e-3)
        assert st4["state_ned"]["groundspeed_mps"] == pytest.approx(7.10, abs=1e-3)
    finally:
        mgr.shutdown()
        tx_rx.close()
        inj.close()


# ---- B8: MavGlobalPositionInt Altitude & Heading Bounds Validation ----


def test_b8_global_position_int_altitude_and_heading_bounds_fail_closed() -> None:
    """`MavGlobalPositionInt` rejects OOB `alt_mm`, `relative_alt_mm`, and `hdg_cdeg` fail-closed."""
    # Valid boundary values (including 65535 unknown heading sentinel)
    for alt_mm, rel_mm, hdg in (
        (-1_000_000, -1_000_000, 0),
        (100_000_000, 100_000_000, 36000),
        (180_000, 175_000, 65535),
    ):
        msg = MavGlobalPositionInt(
            time_boot_ms=100,
            lat_e7=376_117_200,
            lon_e7=-1_223_583_670,
            alt_mm=alt_mm,
            relative_alt_mm=rel_mm,
            vx_cms=1000,
            vy_cms=1000,
            vz_cms=0,
            hdg_cdeg=hdg,
        )
        dec = decode_frame(encode_frame(msg, seq=1, sysid=1, compid=1))
        assert dec.message == msg

    # Adversarial OOB values must raise ValueError in constructor AND on valid-CRC wire decode
    bad_cases = [
        {"alt_mm": -2_147_483_648},
        {"alt_mm": -1_000_001},
        {"alt_mm": 100_000_001},
        {"relative_alt_mm": -2_147_483_648},
        {"relative_alt_mm": -1_000_001},
        {"relative_alt_mm": 100_000_001},
        {"hdg_cdeg": 36001},
        {"hdg_cdeg": 50000},
        {"hdg_cdeg": 65534},
        {"hdg_cdeg": -1},
    ]
    for overrides in bad_cases:
        kwargs = {
            "time_boot_ms": 1000,
            "lat_e7": 376_117_200,
            "lon_e7": -1_223_583_670,
            "alt_mm": 180_000,
            "relative_alt_mm": 175_000,
            "vx_cms": 1500,
            "vy_cms": -2000,
            "vz_cms": 0,
            "hdg_cdeg": 29800,
            **overrides,
        }
        with pytest.raises(ValueError):
            MavGlobalPositionInt(**kwargs)

    # Wire-level decode_frame & live HIL bridge rejection for valid-CRC hostile frames
    wire_bad_alt = _build_raw_valid_crc_gpi_frame(alt_mm=-2_147_483_648)
    wire_bad_hdg = _build_raw_valid_crc_gpi_frame(hdg_cdeg=50000)
    with pytest.raises(ValueError, match="alt_mm"):
        decode_frame(wire_bad_alt)
    with pytest.raises(ValueError, match="hdg_cdeg"):
        decode_frame(wire_bad_hdg)

    mgr = StudioHilBridgeManager(non_flight_sim_only=False)
    inj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx_port = _alloc_free_udp_port()
        mgr.connect(mode="mavlink2_udp", rx_port=rx_port, tx_enabled=False)
        inj.sendto(wire_bad_alt, ("127.0.0.1", rx_port))
        inj.sendto(wire_bad_hdg, ("127.0.0.1", rx_port))
        time.sleep(0.1)
        st = mgr.status()
        assert st["rejected_packets"] >= 2
        assert st["global_pos_count"] == 0
        assert st["state_ned"]["alt_msl_m"] == pytest.approx(180.0)
        assert st["state_ned"]["heading_deg"] == pytest.approx(0.0)
    finally:
        mgr.shutdown()
        inj.close()
