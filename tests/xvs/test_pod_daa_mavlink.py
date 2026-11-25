"""Tests for DR-11 `configs/mavlink/pod_daa.xml` dialect & MAVLink 2.0 codec safety locks."""

from __future__ import annotations

from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import pytest

from xvs.v2.xvs_mavlink import (
    CRC_EXTRA_BY_MSGID,
    CUSTOM_DIALECT_ID_MAX,
    CUSTOM_DIALECT_ID_MIN,
    DEFAULT_POD_UART_STREAMS,
    MAVLINK_V2_STX,
    MAX_BIDIR_UART_BYTES_PER_SEC,
    MAX_UART_UTILIZATION_FRAC,
    MSGID_ODOMETRY,
    MSGID_POD_DAA_TRACK_ARRAY,
    MSGID_POD_SYNC_HEALTH,
    MSGID_POD_VLM_SEMANTIC_VETO,
    MSGID_SET_POSITION_TARGET_LOCAL_NED,
    ArduPlaneSafetyError,
    MavlinkCrcError,
    MavlinkFrameError,
    MavlinkStreamSpec,
    PodDaaTrackArray,
    PodSyncHealth,
    PodVlmSemanticVeto,
    assert_uart_bandwidth_budget,
    compute_uart_bandwidth_budget,
    decode_frame,
    encode_frame,
    parse_and_validate_dialect_xml,
    validate_ardupilot_odometry_version,
    validate_dialect_msgid,
    validate_outbound_msgid,
    x25_crc16,
)
from xvs.v2.xvs_vlm_gate import GateStatus, VlmHazard

REPO_ROOT = Path(__file__).resolve().parents[2]
POD_DAA_XML = REPO_ROOT / "configs" / "mavlink" / "pod_daa.xml"


def test_pod_daa_xml_parses_and_enforces_conflict_free_ids() -> None:
    """Parse `configs/mavlink/pod_daa.xml` via ElementTree and verify IDs, includes, and CRC_EXTRA."""
    assert POD_DAA_XML.is_file(), f"Missing MAVLink dialect XML: {POD_DAA_XML}"

    tree = ET.parse(POD_DAA_XML)
    root = tree.getroot()
    assert root.tag == "mavlink"

    parsed = parse_and_validate_dialect_xml(POD_DAA_XML)
    assert "ardupilotmega.xml" in parsed["includes"]
    assert parsed["message_ids"] == [50001, 50002, 50003, 50004]

    messages = parsed["messages"]
    assert messages[50001]["name"] == "POD_DAA_TRACK_ARRAY"
    assert messages[50002]["name"] == "POD_SYNC_HEALTH"
    assert messages[50003]["name"] == "POD_VLM_SEMANTIC_VETO"
    assert messages[50004]["name"] == "POD_PLUGIN_HEALTH"
    for mid in (50001, 50002, 50003, 50004):
        assert CUSTOM_DIALECT_ID_MIN <= mid <= CUSTOM_DIALECT_ID_MAX
        assert messages[mid]["crc_extra"] == CRC_EXTRA_BY_MSGID[mid]

    # Verify forbidden overlapping ID ranges are rejected fail-closed
    for forbidden_id in (0, 84, 255, 300, 331, 10000, 10001, 10049, 11000, 11999, 42000, 42999, 49999, 50100):
        with pytest.raises(ValueError):
            validate_dialect_msgid(forbidden_id)


def test_mavlink_v2_binary_roundtrip_and_zero_truncation_all_three_messages() -> None:
    """Verify exact encode/decode round-trip (including trailing-zero truncation) for #50001..#50003."""
    samples = [
        PodDaaTrackArray(
            timestamp_us=1_732_186_400_123_456,
            track_count=3,
            azimuth_mrad=-412,
            elevation_mrad=78,
            ang_rate_mrad_s=2,
            t_rad_dk=3125,
            shadow_avoid_hdg_cdeg=28450,
            shadow_avoid_alt_cm=1500,
            flags=0b1111,
        ),
        PodDaaTrackArray(
            timestamp_us=42,
            track_count=0,
            azimuth_mrad=0,
            elevation_mrad=0,
            ang_rate_mrad_s=0,
            t_rad_dk=0,
            shadow_avoid_hdg_cdeg=0,
            shadow_avoid_alt_cm=0,
            flags=0,
        ),
        PodSyncHealth(
            offset_us=-14,
            bos_ffc_state=3,
            jetson_ttp_dc=685,
            mig_dropped_frames=0,
            cam_health_mask=0x3F,
            rta_simplex_mode=2,
        ),
        PodVlmSemanticVeto(
            seq=1042,
            capture_ts_ms=987_654,
            hazard_enum=2,
            azimuth_sector=11,
            conformal_valid=1,
            gate_verdict=1,
            ranked_flz_ids=(4, 1, 7, 255),
        ),
        PodVlmSemanticVeto(
            seq=7,
            capture_ts_ms=500,
            hazard_enum=0,
            azimuth_sector=0,
            conformal_valid=0,
            gate_verdict=0,
            ranked_flz_ids=(0, 0, 0, 0),
        ),
    ]

    for idx, msg in enumerate(samples):
        wire = encode_frame(msg, seq=idx + 10, sysid=1, compid=191)
        assert wire[0] == MAVLINK_V2_STX
        decoded = decode_frame(wire)
        assert decoded.seq == idx + 10
        assert decoded.sysid == 1
        assert decoded.compid == 191
        assert decoded.msgid == msg.MSGID
        assert decoded.message == msg

    # Verify trailing-zero truncation actually shrinks all-zero-tail frame while staying lossless
    zero_tail = encode_frame(samples[1], seq=1)
    assert zero_tail[1] == 1  # only timestamp_us low byte (42) remains non-zero
    assert decode_frame(zero_tail).message == samples[1]


def test_corrupted_crc_and_malformed_frames_are_rejected() -> None:
    """Bit flips in header, payload, or CRC bytes must raise MavlinkCrcError on all 3 messages."""
    msgs = (
        PodDaaTrackArray(1000, 1, 10, -5, 1, 3000, 9000, 500, 1),
        PodSyncHealth(-3, 0, 620, 0, 0x3F, 1),
        PodVlmSemanticVeto(99, 12345, 1, 3, 1, 1, (2, 5, 255, 255)),
    )
    for msg in msgs:
        wire = encode_frame(msg, seq=5)
        for flip_pos in (4, 10, len(wire) - 1):
            corrupted = bytearray(wire)
            corrupted[flip_pos] ^= 0x01
            with pytest.raises(MavlinkCrcError):
                decode_frame(bytes(corrupted))

        # Bad magic byte
        bad_stx = bytearray(wire)
        bad_stx[0] = 0xFE
        with pytest.raises(MavlinkFrameError):
            decode_frame(bytes(bad_stx))

        # Truncated frame
        with pytest.raises(MavlinkFrameError):
            decode_frame(wire[:-1])


def test_dr11_hard_bans_msgid_84_and_enforces_arduplane_472_for_odometry() -> None:
    """Enforce DR-11 ArduPlane locks: #84 hard ban and >= 4.7.2 for #331 ODOMETRY (#32506)."""
    # Hard-ban MAVLink #84 (SET_POSITION_TARGET_LOCAL_NED)
    with pytest.raises(ArduPlaneSafetyError, match="#84"):
        validate_outbound_msgid(MSGID_SET_POSITION_TARGET_LOCAL_NED, ardupilot_version=(4, 7, 2))

    # Also verify decode_frame rejects an incoming/injected #84 frame immediately
    fake_84 = bytearray(
        bytes([MAVLINK_V2_STX])
        + struct.pack("<BBBBBBBBB", 1, 0, 0, 1, 1, 191, 84, 0, 0)
        + b"\x00\x00\x00"
    )
    with pytest.raises(ArduPlaneSafetyError, match="#84"):
        decode_frame(bytes(fake_84))

    # Reject ArduPlane 4.6.x, 4.7.0, 4.7.1 for #331 ODOMETRY (Issue #32506 EKF3 Z snap)
    for bad_ver in ((4, 5, 7), (4, 6, 0), (4, 6, 3), (4, 7, 0), (4, 7, 1)):
        with pytest.raises(ArduPlaneSafetyError, match="32506"):
            validate_ardupilot_odometry_version(bad_ver)
        with pytest.raises(ArduPlaneSafetyError, match="32506"):
            validate_outbound_msgid(MSGID_ODOMETRY, ardupilot_version=bad_ver)

    # Missing version for #331 must also fail closed
    with pytest.raises(ArduPlaneSafetyError):
        validate_outbound_msgid(MSGID_ODOMETRY, ardupilot_version=None)

    # ArduPlane 4.7.2 and newer pass cleanly
    for ok_ver in ((4, 7, 2), (4, 7, 5), (4, 8, 0)):
        validate_ardupilot_odometry_version(ok_ver)
        validate_outbound_msgid(MSGID_ODOMETRY, ardupilot_version=ok_ver)


def test_bidirectional_uart_bandwidth_budget_under_16kbps_and_20pct_of_921k6() -> None:
    """Default pod+FC MAVLink2 schedule stays <= 16,000 B/s (< 20% of 921.6 kbps 8N1)."""
    report = assert_uart_bandwidth_budget(DEFAULT_POD_UART_STREAMS, ardupilot_version=(4, 7, 2))
    assert report["within_budget"] is True
    assert float(report["total_bidir_bytes_per_sec"]) <= MAX_BIDIR_UART_BYTES_PER_SEC
    assert float(report["utilization_frac"]) < MAX_UART_UTILIZATION_FRAC

    # Trying to run the default schedule against ArduPlane 4.6.3 must fail on #331 ODOMETRY (#32506)
    with pytest.raises(ArduPlaneSafetyError, match="32506"):
        compute_uart_bandwidth_budget(DEFAULT_POD_UART_STREAMS, ardupilot_version=(4, 6, 3))

    # Trying to include banned #84 in the outbound schedule must fail
    banned_schedule = list(DEFAULT_POD_UART_STREAMS) + [
        MavlinkStreamSpec("SET_POSITION_TARGET_LOCAL_NED", 84, "outbound", 53, 10.0)
    ]
    with pytest.raises(ArduPlaneSafetyError, match="#84"):
        compute_uart_bandwidth_budget(banned_schedule, ardupilot_version=(4, 7, 2))

    # Oversubscribed UART stream must trip budget assertion
    heavy_schedule = list(DEFAULT_POD_UART_STREAMS) + [
        MavlinkStreamSpec("POD_DAA_TRACK_ARRAY_FLOOD", MSGID_POD_DAA_TRACK_ARRAY, "outbound", 22, 200.0)
    ]
    with pytest.raises(ValueError, match="budget exceeded"):
        assert_uart_bandwidth_budget(heavy_schedule, ardupilot_version=(4, 7, 2))


def test_pod_daa_xml_enums_match_vlm_gate_and_rta_simplex_1_to_1() -> None:
    """Verify `D16-W2-M1`: 1:1 integer/name parity with VlmHazard, GateStatus, and SimplexMode (0..4)."""
    root = ET.parse(POD_DAA_XML).getroot()
    enums_el = root.find("enums")
    assert enums_el is not None

    enums_by_name: dict[str, dict[int, str]] = {}
    for enum_el in enums_el.findall("enum"):
        enum_name = str(enum_el.attrib["name"])
        entries: dict[int, str] = {}
        for entry_el in enum_el.findall("entry"):
            entries[int(entry_el.attrib["value"])] = str(entry_el.attrib["name"])
        enums_by_name[enum_name] = entries

    # 1. POD_VLM_HAZARD_TYPE == xvs.v2.xvs_vlm_gate.VlmHazard
    expected_hazards = {int(h): f"POD_VLM_HAZARD_{h.name}" for h in VlmHazard}
    assert enums_by_name["POD_VLM_HAZARD_TYPE"] == expected_hazards

    # 2. POD_VLM_GATE_VERDICT == xvs.v2.xvs_vlm_gate.GateStatus (0..5)
    expected_verdicts = {idx: f"POD_VLM_GATE_{status.name}" for idx, status in enumerate(GateStatus)}
    assert enums_by_name["POD_VLM_GATE_VERDICT"] == expected_verdicts

    # 3. POD_RTA_SIMPLEX_MODE == canonical 5-state xvs-rta-core SimplexMode #[repr(u8)] (0..4)
    expected_simplex = {
        0: "POD_RTA_INVALID_INHIBIT",
        1: "POD_RTA_FALLBACK_D0_SAFE",
        2: "POD_RTA_DEGRADED_COAST",
        3: "POD_RTA_ISOLATED_5CAM",
        4: "POD_RTA_NOMINAL_ML",
    }
    assert enums_by_name["POD_RTA_SIMPLEX_MODE"] == expected_simplex


def test_unsupported_nonzero_incompat_flags_are_rejected() -> None:
    """Verify `D16-W2-m1`: MAVLink 2.0 receivers reject unsupported non-zero `incompat_flags`."""
    msg = PodSyncHealth(offset_us=-5, bos_ffc_state=0, jetson_ttp_dc=640, mig_dropped_frames=0, cam_health_mask=0x3F, rta_simplex_mode=4)
    with pytest.raises((MavlinkFrameError, ValueError), match="incompat_flags"):
        encode_frame(msg, incompat_flags=0x02)

    # Craft a validly-CRC'd frame with incompat_flags=0x02 and prove decode_frame rejects it
    good = bytearray(encode_frame(msg, seq=3, sysid=1, compid=191))
    good[2] = 0x02  # unsupported incompatible flag bit
    payload_len = good[1]
    crc_extra = CRC_EXTRA_BY_MSGID[MSGID_POD_SYNC_HEALTH]
    recomputed_crc = x25_crc16(bytes(good[1 : 10 + payload_len]) + bytes([crc_extra]))
    struct.pack_into("<H", good, 10 + payload_len, recomputed_crc)

    with pytest.raises((MavlinkFrameError, ValueError), match="incompat_flags"):
        decode_frame(bytes(good))

