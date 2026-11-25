"""DR-16 V3.0 MAVLink 2.0 `POD_PLUGIN_HEALTH (#50004)` & Signed UART Budget Tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib

import pytest

from xvs.v2.xvs_mavlink import (
    CRC_EXTRA_BY_MSGID,
    CUSTOM_DIALECT_ID_MAX,
    CUSTOM_DIALECT_ID_MIN,
    DEFAULT_POD_UART_STREAMS,
    DEFAULT_V3_SIGNED_UART_STREAMS,
    MAVLINK_V2_HEADER_LEN,
    MAVLINK_V2_OVERHEAD_BYTES,
    MAVLINK_V2_SIGNATURE_BYTES,
    MAVLINK_V2_STX,
    MAX_BIDIR_UART_BYTES_PER_SEC,
    MAX_SIGNED_UART_BYTES_PER_SEC,
    MAX_UART_UTILIZATION_FRAC,
    MSGID_POD_PLUGIN_HEALTH,
    WIRE_PAYLOAD_LEN_BY_MSGID,
    MavlinkCrcError,
    MavlinkFrameError,
    MavlinkStreamSpec,
    PodCivilMissionClass,
    PodPluginHealth,
    PodPluginLifecycleState,
    assert_uart_bandwidth_budget,
    compute_uart_bandwidth_budget,
    decode_frame,
    encode_frame,
    parse_and_validate_dialect_xml,
    x25_crc16,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POD_DAA_XML = REPO_ROOT / "configs" / "mavlink" / "pod_daa.xml"


def test_pod_plugin_health_50004_present_in_xml_with_exact_crc_extra_and_enums() -> None:
    """Verify `pod_daa.xml` defines `#50004 POD_PLUGIN_HEALTH` with exact CRC_EXTRA and enums."""
    parsed = parse_and_validate_dialect_xml(POD_DAA_XML)
    assert MSGID_POD_PLUGIN_HEALTH == 50004
    assert CUSTOM_DIALECT_ID_MIN <= MSGID_POD_PLUGIN_HEALTH <= CUSTOM_DIALECT_ID_MAX
    assert 50004 in parsed["message_ids"]
    assert parsed["messages"][50004]["name"] == "POD_PLUGIN_HEALTH"
    assert parsed["messages"][50004]["field_count"] == 10
    assert parsed["messages"][50004]["crc_extra"] == CRC_EXTRA_BY_MSGID[MSGID_POD_PLUGIN_HEALTH]
    assert WIRE_PAYLOAD_LEN_BY_MSGID[MSGID_POD_PLUGIN_HEALTH] == 48

    root = ET.parse(POD_DAA_XML).getroot()
    enums_el = root.find("enums")
    assert enums_el is not None
    enums_by_name: dict[str, dict[int, str]] = {}
    for enum_el in enums_el.findall("enum"):
        entries = {int(e.attrib["value"]): str(e.attrib["name"]) for e in enum_el.findall("entry")}
        enums_by_name[str(enum_el.attrib["name"])] = entries

    assert enums_by_name["POD_PLUGIN_LIFECYCLE_STATE"] == {
        0: "POD_PLUGIN_LIFECYCLE_INIT",
        1: "POD_PLUGIN_LIFECYCLE_HEALTHY",
        2: "POD_PLUGIN_LIFECYCLE_DEGRADED",
        3: "POD_PLUGIN_LIFECYCLE_OFFLINE",
    }
    assert enums_by_name["POD_CIVIL_MISSION_CLASS"] == {
        1: "POD_CIVIL_MISSION_WILDFIRE_THERMAL",
        2: "POD_CIVIL_MISSION_SEARCH_AND_RESCUE",
        3: "POD_CIVIL_MISSION_MARITIME_ENVIRONMENTAL",
        4: "POD_CIVIL_MISSION_INFRASTRUCTURE_INSPECTION",
        5: "POD_CIVIL_MISSION_MEDICAL_CARGO_LOGISTICS",
    }


def test_pod_plugin_health_48byte_roundtrip_and_zero_actuator_authority() -> None:
    """Verify exact 48-byte wire layout, MAVLink2 roundtrip, and ACTUATOR_AUTHORITY=False."""
    msg = PodPluginHealth.build(
        plugin_id_hash=0x5A17_9C2E_1104_AF88,
        timestamp_mono_us=987_654_321_012,
        last_seq_no=42_019,
        lifecycle_state=PodPluginLifecycleState.HEALTHY,
        civil_mission_class=PodCivilMissionClass.SEARCH_AND_RESCUE,
        missed_heartbeats=0,
        rss_mb=512,
        cpu_permille=845,
    )
    assert msg.ACTUATOR_AUTHORITY is False
    assert msg.PAYLOAD_BYTES == 48
    assert msg.is_offline_latched is False
    assert msg.verify_ieee_crc32() is True

    raw_payload = msg.pack_payload()
    assert len(raw_payload) == 48
    assert raw_payload[-12:] == bytes(12)  # reserved_zero tail strictly zero

    # Full untrimmed 48-byte MAVLink2 wire frame roundtrip
    wire_full = encode_frame(msg, seq=17, sysid=1, compid=191, truncate_zeros=False)
    assert wire_full[0] == MAVLINK_V2_STX
    assert wire_full[1] == 48
    assert len(wire_full) == MAVLINK_V2_OVERHEAD_BYTES + 48
    decoded_full = decode_frame(wire_full, verify_inner_crc32=True)
    assert decoded_full.msgid == MSGID_POD_PLUGIN_HEALTH
    assert decoded_full.seq == 17
    assert decoded_full.message == msg

    # MAVLink2 trailing-zero truncation shrinks the 12-byte zero tail cleanly while remaining lossless
    wire_trunc = encode_frame(msg, seq=18, sysid=1, compid=191, truncate_zeros=True)
    assert wire_trunc[1] == 36  # 48 - 12 zero tail bytes
    decoded_trunc = decode_frame(wire_trunc, verify_inner_crc32=True)
    assert decoded_trunc.message == msg

    # Non-zero reserved_zero bytes must be rejected fail-closed at construction (ACTUATOR_AUTHORITY=False)
    with pytest.raises(ValueError, match="ACTUATOR_AUTHORITY"):
        replace(msg, reserved_zero=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1))


def test_pod_plugin_health_crc32_and_wire_tamper_detection() -> None:
    """Verify IEEE-802.3 CRC32 catches payload tampering even inside a validly CRC16-framed packet."""
    msg = PodPluginHealth.build(
        plugin_id_hash=0x1122_3344_5566_7788,
        timestamp_mono_us=55_000_123,
        last_seq_no=99,
        lifecycle_state=PodPluginLifecycleState.HEALTHY,
        civil_mission_class=PodCivilMissionClass.WILDFIRE_THERMAL,
        missed_heartbeats=1,
        rss_mb=380,
        cpu_permille=620,
    )
    assert msg.verify_ieee_crc32() is True

    # 1. Tamper a payload field without updating ieee_crc32
    tampered_field = replace(msg, rss_mb=9999)
    assert tampered_field.verify_ieee_crc32() is False
    with pytest.raises(MavlinkCrcError, match="IEEE-CRC32"):
        tampered_field.assert_ieee_crc32()

    # 2. Re-encode tampered packet with valid MAVLink2 X.25 CRC16 -> verify_inner_crc32 catches it!
    wire_tampered_inner = encode_frame(tampered_field, seq=9)
    with pytest.raises(MavlinkCrcError, match="IEEE-CRC32"):
        decode_frame(wire_tampered_inner, verify_inner_crc32=True)

    # 3. Direct bit flip on the MAVLink2 wire frame -> X.25 CRC16 rejects immediately
    wire_good = encode_frame(msg, seq=10, truncate_zeros=False)
    for pos in (4, MAVLINK_V2_HEADER_LEN + 2, len(wire_good) - 1):
        corrupted = bytearray(wire_good)
        corrupted[pos] ^= 0x01
        with pytest.raises(MavlinkCrcError):
            decode_frame(bytes(corrupted), verify_inner_crc32=True)

    # 4. Crafted valid-CRC16 wire frame with non-zero reserved_zero byte -> unpack rejects fail-closed
    raw = bytearray(msg.pack_payload())
    raw[-1] = 0xFF
    header_body = struct.pack(
        "<BBBBBBBBB",
        48,
        0,
        0,
        11,
        1,
        191,
        MSGID_POD_PLUGIN_HEALTH & 0xFF,
        (MSGID_POD_PLUGIN_HEALTH >> 8) & 0xFF,
        (MSGID_POD_PLUGIN_HEALTH >> 16) & 0xFF,
    )
    crc16 = x25_crc16(header_body + bytes(raw) + bytes([CRC_EXTRA_BY_MSGID[MSGID_POD_PLUGIN_HEALTH]]))
    rogue_frame = bytes([MAVLINK_V2_STX]) + header_body + bytes(raw) + struct.pack("<H", crc16)
    with pytest.raises(MavlinkFrameError, match="ACTUATOR_AUTHORITY"):
        decode_frame(rogue_frame)


def test_pod_plugin_health_offline_latch_and_invalid_lifecycle_rejection() -> None:
    """Verify deterministic `PLUGIN_OFFLINE` (3) latch state roundtrips and out-of-range states fail."""
    offline_msg = PodPluginHealth.build(
        plugin_id_hash=0xDEAD_BEEF_CAFE_0001,
        timestamp_mono_us=120_400_000,
        last_seq_no=1_024,
        lifecycle_state=PodPluginLifecycleState.PLUGIN_OFFLINE,
        civil_mission_class=PodCivilMissionClass.INFRASTRUCTURE_INSPECTION,
        missed_heartbeats=5,
        rss_mb=0,
        cpu_permille=0,
    )
    assert offline_msg.lifecycle_state == 3
    assert offline_msg.is_offline_latched is True
    assert offline_msg.verify_ieee_crc32() is True

    wire = encode_frame(offline_msg, seq=44)
    decoded = decode_frame(wire, verify_inner_crc32=True)
    assert isinstance(decoded.message, PodPluginHealth)
    assert decoded.message.is_offline_latched is True
    assert decoded.message.lifecycle_state == int(PodPluginLifecycleState.PLUGIN_OFFLINE)
    assert decoded.message.missed_heartbeats == 5

    # Out-of-range lifecycle state (4) must be rejected fail-closed
    with pytest.raises(ValueError, match="lifecycle_state"):
        PodPluginHealth.build(
            plugin_id_hash=1,
            timestamp_mono_us=2,
            last_seq_no=3,
            lifecycle_state=4,
            civil_mission_class=1,
        )

    # Out-of-range civil_mission_class (0 uninitialized, 6 out-of-bounds) rejected in constructor & unpack_payload
    for bad_class in (0, 6):
        with pytest.raises(ValueError, match="civil_mission_class"):
            PodPluginHealth(
                plugin_id_hash=1,
                timestamp_mono_us=2,
                last_seq_no=3,
                lifecycle_state=1,
                civil_mission_class=bad_class,
                missed_heartbeats=0,
                rss_mb=128,
                cpu_permille=100,
            )
        with pytest.raises(ValueError, match="civil_mission_class"):
            PodPluginHealth.build(
                plugin_id_hash=1,
                timestamp_mono_us=2,
                last_seq_no=3,
                lifecycle_state=1,
                civil_mission_class=bad_class,
            )
        bad_raw = bytearray(offline_msg.pack_payload())
        bad_raw[35] = bad_class  # civil_mission_class byte offset in <QQQIHHHBB12B
        with pytest.raises(ValueError, match="civil_mission_class"):
            PodPluginHealth.unpack_payload(bytes(bad_raw))


def test_rust_python_golden_vector_parity_50004() -> None:
    """Verify bit-for-bit Rust `xvs_os_hal::PluginHealth48B` <-> Python `#50004` golden vector parity."""
    expected_crc32 = 0x8225C4FB
    expected_wire_hex = (
        "8877665544332211"
        "00f1536500000000"
        "2a00000000000000"
        "fbc42582"
        "0000"
        "0002"
        "fa00"
        "01"
        "02"
        "000000000000000000000000"
    )
    msg = PodPluginHealth.build(
        plugin_id_hash=0x1122334455667788,
        timestamp_mono_us=1_700_000_000,
        last_seq_no=42,
        missed_heartbeats=0,
        rss_mb=512,
        cpu_permille=250,
        lifecycle_state=int(PodPluginLifecycleState.HEALTHY),
        civil_mission_class=int(PodCivilMissionClass.SEARCH_AND_RESCUE),
    )
    assert msg.lifecycle_state == 1
    assert msg.civil_mission_class == 2
    assert msg.ieee_crc32 == expected_crc32

    raw_48b = msg.pack_payload()
    assert len(raw_48b) == 48
    assert raw_48b.hex() == expected_wire_hex
    assert (zlib.crc32(raw_48b[0:24] + raw_48b[28:48]) & 0xFFFFFFFF) == expected_crc32
    assert PodPluginHealth.unpack_payload(raw_48b) == msg

    wire = encode_frame(msg, seq=1, truncate_zeros=False)
    decoded = decode_frame(wire, verify_inner_crc32=True)
    assert decoded.message == msg


def test_dr16_signed_mavlink2_uart_budget_under_20pct_of_921k6() -> None:
    """Verify DR-16 full signed MAVLink2 V3.0 stream (+13 B/frame, #50004 @ 5 Hz) stays < 20%."""
    assert MAVLINK_V2_SIGNATURE_BYTES == 13
    assert MAX_SIGNED_UART_BYTES_PER_SEC == pytest.approx(18_432.0)

    # 1. DR-16 V3.0 signed schedule (includes #50004 POD_PLUGIN_HEALTH @ 5 Hz)
    v3_signed = assert_uart_bandwidth_budget(
        DEFAULT_V3_SIGNED_UART_STREAMS,
        ardupilot_version=(4, 7, 3),
        plugin_health_hz=5.0,
        signed_mavlink2=True,
    )
    assert v3_signed["within_budget"] is True
    assert v3_signed["signed_mavlink2"] is True
    assert v3_signed["signature_bytes_per_frame"] == 13
    assert v3_signed["plugin_health_hz"] == pytest.approx(5.0)
    assert float(v3_signed["total_bidir_bytes_per_sec"]) <= MAX_SIGNED_UART_BYTES_PER_SEC
    assert float(v3_signed["utilization_frac"]) < 0.15  # DR-16 OS5 target (< 15%, hard cap < 20%)

    # 2. Legacy DR-11 full IMU/attitude schedule + optional POD_PLUGIN_HEALTH @ 5 Hz + SHA-256 signing
    full_signed = assert_uart_bandwidth_budget(
        DEFAULT_POD_UART_STREAMS,
        ardupilot_version=(4, 7, 3),
        plugin_health_hz=5.0,
        signed_mavlink2=True,
    )
    assert full_signed["within_budget"] is True
    assert full_signed["plugin_health_hz"] == pytest.approx(5.0)
    assert float(full_signed["total_bidir_bytes_per_sec"]) <= MAX_SIGNED_UART_BYTES_PER_SEC
    assert float(full_signed["utilization_frac"]) < MAX_UART_UTILIZATION_FRAC

    # 3. Backward-compatible unsigned call stays <= 16,000 B/s
    legacy_unsigned = assert_uart_bandwidth_budget(
        DEFAULT_POD_UART_STREAMS,
        ardupilot_version=(4, 7, 2),
    )
    assert legacy_unsigned["signed_mavlink2"] is False
    assert legacy_unsigned["signature_bytes_per_frame"] == 0
    assert float(legacy_unsigned["total_bidir_bytes_per_sec"]) <= MAX_BIDIR_UART_BYTES_PER_SEC

    # 4. Signed overhead must strictly increase bandwidth vs unsigned
    v3_unsigned = compute_uart_bandwidth_budget(
        DEFAULT_V3_SIGNED_UART_STREAMS,
        ardupilot_version=(4, 7, 3),
        signed_mavlink2=False,
    )
    assert float(v3_signed["total_bidir_bytes_per_sec"]) > float(v3_unsigned["total_bidir_bytes_per_sec"])

    # 5. Oversubscribed signed stream must trip budget assertion fail-closed
    flood = list(DEFAULT_V3_SIGNED_UART_STREAMS) + [
        MavlinkStreamSpec("POD_PLUGIN_HEALTH_FLOOD", MSGID_POD_PLUGIN_HEALTH, "outbound", 48, 120.0)
    ]
    with pytest.raises(ValueError, match="budget exceeded"):
        assert_uart_bandwidth_budget(flood, ardupilot_version=(4, 7, 3), signed_mavlink2=True)

    # 6. Negative plugin_health_hz rejected
    with pytest.raises(ValueError, match="plugin_health_hz"):
        compute_uart_bandwidth_budget(DEFAULT_POD_UART_STREAMS, plugin_health_hz=-1.0)
