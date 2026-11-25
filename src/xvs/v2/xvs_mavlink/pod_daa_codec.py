"""Pure-Python MAVLink 2.0 wire codec and ArduPlane 4.7.2+ safety guards for `pod_daa.xml`.

Implements `DR-11` §4, `DR-16` §3, & `PROJECT_CHARTER.md` §2:
- Conflict-free custom dialect IDs strictly in `50000..50099`:
  - `#50001 POD_DAA_TRACK_ARRAY` (22 B payload)
  - `#50002 POD_SYNC_HEALTH` (11 B payload)
  - `#50003 POD_VLM_SEMANTIC_VETO` (16 B payload)
  - `#50004 POD_PLUGIN_HEALTH` (48 B payload, 5 Hz, ACTUATOR_AUTHORITY=false)
- Standard MAVLink v2 wire framing (`0xFD` STX, 10-byte header, trailing-zero payload
  truncation, X.25 CRC-16/MCRF4XX with official per-message `CRC_EXTRA`).
- Hard-ban on MAVLink `#84` (`SET_POSITION_TARGET_LOCAL_NED`, ArduPilot Issue `#19858`).
- Strict `ArduPlane >= (4, 7, 2)` gate for `#331 ODOMETRY` (ArduPilot Issue `#32506`
  EKF3 vertical Z-snap protection).
- Bidirectional UART bandwidth budget enforcing `<= 16,000 B/s` unsigned and
  `<= 18,432 B/s` (`< 20%` of 921.6 kbps 8N1) under full SHA-256 signed MAVLink 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import hashlib
import hmac
import math
from pathlib import Path
import struct
from typing import Any, Sequence
import xml.etree.ElementTree as ET
import zlib

MAVLINK_V2_STX = 0xFD
MAVLINK_IFLAG_SIGNED = 0x01
MAVLINK_V2_HEADER_LEN = 10
MAVLINK_V2_CHECKSUM_LEN = 2
MAVLINK_V2_SIGNATURE_BYTES = 13
MAVLINK_V2_OVERHEAD_BYTES = MAVLINK_V2_HEADER_LEN + MAVLINK_V2_CHECKSUM_LEN  # 12 bytes unsigned
MAVLINK_V2_SIGNED_OVERHEAD_BYTES = MAVLINK_V2_OVERHEAD_BYTES + MAVLINK_V2_SIGNATURE_BYTES  # 25 bytes signed

CUSTOM_DIALECT_ID_MIN = 50000
CUSTOM_DIALECT_ID_MAX = 50099
FORBIDDEN_DIALECT_ID_RANGES: tuple[tuple[int, int], ...] = (
    (0, 255),
    (300, 10000),
    (10001, 10049),
    (11000, 11999),
    (42000, 42999),
)

MSGID_HEARTBEAT = 0
MSGID_ATTITUDE = 30
MSGID_GLOBAL_POSITION_INT = 33
MSGID_VFR_HUD = 74
MSGID_COMMAND_INT = 75
MSGID_COMMAND_LONG = 76
MSGID_COMMAND_ACK = 77
MSGID_SET_ATTITUDE_TARGET = 82
MSGID_SET_POSITION_TARGET_LOCAL_NED = 84
MSGID_ODOMETRY = 331
MSGID_POD_DAA_TRACK_ARRAY = 50001
MSGID_POD_SYNC_HEALTH = 50002
MSGID_POD_VLM_SEMANTIC_VETO = 50003
MSGID_POD_PLUGIN_HEALTH = 50004

MIN_GLOBAL_POSITION_ALT_MM = -1_000_000
MAX_GLOBAL_POSITION_ALT_MM = 100_000_000
MAV_HEADING_UNKNOWN_CDEG = 65535

STANDARD_MAVLINK_MSGIDS: frozenset[int] = frozenset(
    {
        MSGID_HEARTBEAT,
        MSGID_ATTITUDE,
        MSGID_GLOBAL_POSITION_INT,
        MSGID_VFR_HUD,
        MSGID_COMMAND_INT,
        MSGID_COMMAND_LONG,
        MSGID_COMMAND_ACK,
        MSGID_SET_ATTITUDE_TARGET,
    }
)

BANNED_MAVLINK_MSGIDS: frozenset[int] = frozenset({MSGID_SET_POSITION_TARGET_LOCAL_NED})
MIN_ARDUPILOT_ODOMETRY_VERSION: tuple[int, int, int] = (4, 7, 2)

UART_BAUD_BPS = 921_600
UART_BITS_PER_BYTE_8N1 = 10
UART_CAPACITY_BYTES_PER_SEC = float(UART_BAUD_BPS) / float(UART_BITS_PER_BYTE_8N1)  # 92,160 B/s
MAX_BIDIR_UART_BYTES_PER_SEC = 16_000.0
MAX_UART_UTILIZATION_FRAC = 0.20
MAX_SIGNED_UART_BYTES_PER_SEC = UART_CAPACITY_BYTES_PER_SEC * MAX_UART_UTILIZATION_FRAC  # 18,432.0 B/s


class PodPluginLifecycleState(IntEnum):
    """DR-16 Ring-1 civil plugin supervisor lifecycle state (matches POD_PLUGIN_LIFECYCLE_STATE)."""

    INIT = 0
    HEALTHY = 1
    DEGRADED = 2
    PLUGIN_OFFLINE = 3


class PodCivilMissionClass(IntEnum):
    """DR-16 strictly civil mission plugin taxonomy (1..5, matches Rust CivilMissionClass & XML)."""

    WILDFIRE_THERMAL = 1
    SEARCH_AND_RESCUE = 2
    MARITIME_ENVIRONMENTAL = 3
    INFRASTRUCTURE_INSPECTION = 4
    MEDICAL_CARGO_LOGISTICS = 5

_BASE_TYPE_INFO: dict[str, tuple[str, int]] = {
    "uint64_t": ("Q", 8),
    "int64_t": ("q", 8),
    "double": ("d", 8),
    "uint32_t": ("I", 4),
    "int32_t": ("i", 4),
    "float": ("f", 4),
    "uint16_t": ("H", 2),
    "int16_t": ("h", 2),
    "uint8_t": ("B", 1),
    "int8_t": ("b", 1),
    "char": ("c", 1),
}


class MavlinkFrameError(ValueError):
    """Raised when a MAVLink v2 frame is structurally malformed or unknown."""


class MavlinkCrcError(MavlinkFrameError):
    """Raised when MAVLink v2 X.25 CRC-16 verification fails."""


class ArduPlaneSafetyError(PermissionError, ValueError):
    """Raised when a DR-11 ArduPlane safety lock (#84 ban or #32506 version guard) triggers."""


def _require_finite_float(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite MAVLink field rejected for {name!r}: {value!r}")
    return val


@dataclass(frozen=True)
class _FieldDef:
    name: str
    base_type: str
    array_len: int = 0


def x25_crc16(data: bytes, crc_init: int = 0xFFFF) -> int:
    """Compute MAVLink X.25 CRC-16/MCRF4XX checksum over `data`."""
    crc = int(crc_init) & 0xFFFF
    for byte in data:
        tmp = byte ^ (crc & 0xFF)
        tmp = (tmp ^ ((tmp << 4) & 0xFF)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def _wire_sort_fields(fields: Sequence[_FieldDef]) -> list[_FieldDef]:
    """Sort message fields in MAVLink canonical wire order (descending primitive byte size)."""
    indexed = list(enumerate(fields))
    indexed.sort(key=lambda pair: (-_BASE_TYPE_INFO[pair[1].base_type][1], pair[0]))
    return [f for _, f in indexed]


def compute_crc_extra(message_name: str, fields: Sequence[tuple[str, str, int]]) -> int:
    """Compute official MAVLink `CRC_EXTRA` byte from `(base_type, field_name, array_len)` specs."""
    defs = [_FieldDef(name=fn, base_type=bt, array_len=int(al)) for (bt, fn, al) in fields]
    ordered = _wire_sort_fields(defs)
    crc = x25_crc16(f"{message_name} ".encode("ascii"))
    for field in ordered:
        crc = x25_crc16(f"{field.base_type} {field.name} ".encode("ascii"), crc)
        if field.array_len > 0:
            crc = x25_crc16(bytes([field.array_len]), crc)
    return (crc & 0xFF) ^ (crc >> 8)


# ---- Standard MAVLink 2.0 `minimal.xml` / `common.xml` wire definitions (Zero-GPL clean-room) ----

_HEARTBEAT_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint8_t", "type", 0),
    ("uint8_t", "autopilot", 0),
    ("uint8_t", "base_mode", 0),
    ("uint32_t", "custom_mode", 0),
    ("uint8_t", "system_status", 0),
    ("uint8_t", "mavlink_version", 0),
)
_HEARTBEAT_STRUCT = struct.Struct("<IBBBBB")  # 9 B

_ATTITUDE_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint32_t", "time_boot_ms", 0),
    ("float", "roll", 0),
    ("float", "pitch", 0),
    ("float", "yaw", 0),
    ("float", "rollspeed", 0),
    ("float", "pitchspeed", 0),
    ("float", "yawspeed", 0),
)
_ATTITUDE_STRUCT = struct.Struct("<Iffffff")  # 28 B

_GLOBAL_POSITION_INT_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint32_t", "time_boot_ms", 0),
    ("int32_t", "lat", 0),
    ("int32_t", "lon", 0),
    ("int32_t", "alt", 0),
    ("int32_t", "relative_alt", 0),
    ("int16_t", "vx", 0),
    ("int16_t", "vy", 0),
    ("int16_t", "vz", 0),
    ("uint16_t", "hdg", 0),
)
_GLOBAL_POSITION_INT_STRUCT = struct.Struct("<IiiiihhhH")  # 28 B

_VFR_HUD_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("float", "airspeed", 0),
    ("float", "groundspeed", 0),
    ("int16_t", "heading", 0),
    ("uint16_t", "throttle", 0),
    ("float", "alt", 0),
    ("float", "climb", 0),
)
_VFR_HUD_STRUCT = struct.Struct("<ffffhH")  # 20 B

_COMMAND_INT_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint8_t", "target_system", 0),
    ("uint8_t", "target_component", 0),
    ("uint8_t", "frame", 0),
    ("uint16_t", "command", 0),
    ("uint8_t", "current", 0),
    ("uint8_t", "autocontinue", 0),
    ("float", "param1", 0),
    ("float", "param2", 0),
    ("float", "param3", 0),
    ("float", "param4", 0),
    ("int32_t", "x", 0),
    ("int32_t", "y", 0),
    ("float", "z", 0),
)
_COMMAND_INT_STRUCT = struct.Struct("<ffffiifHBBBBB")  # 35 B

_COMMAND_LONG_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint8_t", "target_system", 0),
    ("uint8_t", "target_component", 0),
    ("uint16_t", "command", 0),
    ("uint8_t", "confirmation", 0),
    ("float", "param1", 0),
    ("float", "param2", 0),
    ("float", "param3", 0),
    ("float", "param4", 0),
    ("float", "param5", 0),
    ("float", "param6", 0),
    ("float", "param7", 0),
)
_COMMAND_LONG_STRUCT = struct.Struct("<7fHBBB")  # 33 B

_COMMAND_ACK_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint16_t", "command", 0),
    ("uint8_t", "result", 0),
)
_COMMAND_ACK_BASE_STRUCT = struct.Struct("<HB")  # 3 B base
_COMMAND_ACK_V2_STRUCT = struct.Struct("<HBBiBB")  # 10 B with MAVLink v2 extensions

_SET_ATTITUDE_TARGET_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint32_t", "time_boot_ms", 0),
    ("uint8_t", "target_system", 0),
    ("uint8_t", "target_component", 0),
    ("uint8_t", "type_mask", 0),
    ("float", "q", 4),
    ("float", "body_roll_rate", 0),
    ("float", "body_pitch_rate", 0),
    ("float", "body_yaw_rate", 0),
    ("float", "thrust", 0),
)
_SET_ATTITUDE_TARGET_STRUCT = struct.Struct("<I4fffffBBB")  # 39 B base (51 B with thrust_body ext)
_SET_ATTITUDE_TARGET_MAX_LEN = _SET_ATTITUDE_TARGET_STRUCT.size + 12

_POD_DAA_TRACK_ARRAY_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint64_t", "timestamp_us", 0),
    ("int16_t", "azimuth_mrad", 0),
    ("int16_t", "elevation_mrad", 0),
    ("int16_t", "ang_rate_mrad_s", 0),
    ("uint16_t", "t_rad_dk", 0),
    ("uint16_t", "shadow_avoid_hdg_cdeg", 0),
    ("int16_t", "shadow_avoid_alt_cm", 0),
    ("uint8_t", "track_count", 0),
    ("uint8_t", "flags", 0),
)
_POD_DAA_TRACK_ARRAY_STRUCT = struct.Struct("<QhhhHHhBB")

_POD_SYNC_HEALTH_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("int32_t", "offset_us", 0),
    ("int16_t", "jetson_ttp_dc", 0),
    ("uint16_t", "mig_dropped_frames", 0),
    ("uint8_t", "bos_ffc_state", 0),
    ("uint8_t", "cam_health_mask", 0),
    ("uint8_t", "rta_simplex_mode", 0),
)
_POD_SYNC_HEALTH_STRUCT = struct.Struct("<ihHBBB")

_POD_VLM_SEMANTIC_VETO_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint32_t", "seq", 0),
    ("uint32_t", "capture_ts_ms", 0),
    ("uint8_t", "hazard_enum", 0),
    ("uint8_t", "azimuth_sector", 0),
    ("uint8_t", "conformal_valid", 0),
    ("uint8_t", "gate_verdict", 0),
    ("uint8_t", "ranked_flz_ids", 4),
)
_POD_VLM_SEMANTIC_VETO_STRUCT = struct.Struct("<IIBBBB4B")

_POD_PLUGIN_HEALTH_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint64_t", "plugin_id_hash", 0),
    ("uint64_t", "timestamp_mono_us", 0),
    ("uint64_t", "last_seq_no", 0),
    ("uint32_t", "ieee_crc32", 0),
    ("uint16_t", "missed_heartbeats", 0),
    ("uint16_t", "rss_mb", 0),
    ("uint16_t", "cpu_permille", 0),
    ("uint8_t", "lifecycle_state", 0),
    ("uint8_t", "civil_mission_class", 0),
    ("uint8_t", "reserved_zero", 12),
)
_POD_PLUGIN_HEALTH_STRUCT = struct.Struct("<QQQIHHHBB12B")
_POD_PLUGIN_HEALTH_CRC_INPUT_STRUCT = struct.Struct("<QQQHHHBB12B")

CRC_EXTRA_BY_MSGID: dict[int, int] = {
    MSGID_HEARTBEAT: compute_crc_extra("HEARTBEAT", _HEARTBEAT_FIELDS),
    MSGID_ATTITUDE: compute_crc_extra("ATTITUDE", _ATTITUDE_FIELDS),
    MSGID_GLOBAL_POSITION_INT: compute_crc_extra("GLOBAL_POSITION_INT", _GLOBAL_POSITION_INT_FIELDS),
    MSGID_VFR_HUD: compute_crc_extra("VFR_HUD", _VFR_HUD_FIELDS),
    MSGID_COMMAND_INT: compute_crc_extra("COMMAND_INT", _COMMAND_INT_FIELDS),
    MSGID_COMMAND_LONG: compute_crc_extra("COMMAND_LONG", _COMMAND_LONG_FIELDS),
    MSGID_COMMAND_ACK: compute_crc_extra("COMMAND_ACK", _COMMAND_ACK_FIELDS),
    MSGID_SET_ATTITUDE_TARGET: compute_crc_extra("SET_ATTITUDE_TARGET", _SET_ATTITUDE_TARGET_FIELDS),
    MSGID_POD_DAA_TRACK_ARRAY: compute_crc_extra("POD_DAA_TRACK_ARRAY", _POD_DAA_TRACK_ARRAY_FIELDS),
    MSGID_POD_SYNC_HEALTH: compute_crc_extra("POD_SYNC_HEALTH", _POD_SYNC_HEALTH_FIELDS),
    MSGID_POD_VLM_SEMANTIC_VETO: compute_crc_extra("POD_VLM_SEMANTIC_VETO", _POD_VLM_SEMANTIC_VETO_FIELDS),
    MSGID_POD_PLUGIN_HEALTH: compute_crc_extra("POD_PLUGIN_HEALTH", _POD_PLUGIN_HEALTH_FIELDS),
}

# Exact MAVLink common.xml / minimal.xml CRC_EXTRA invariants verified at import time.
assert CRC_EXTRA_BY_MSGID[MSGID_HEARTBEAT] == 50
assert CRC_EXTRA_BY_MSGID[MSGID_ATTITUDE] == 39
assert CRC_EXTRA_BY_MSGID[MSGID_GLOBAL_POSITION_INT] == 104
assert CRC_EXTRA_BY_MSGID[MSGID_VFR_HUD] == 20
assert CRC_EXTRA_BY_MSGID[MSGID_COMMAND_INT] == 158
assert CRC_EXTRA_BY_MSGID[MSGID_COMMAND_LONG] == 152
assert CRC_EXTRA_BY_MSGID[MSGID_COMMAND_ACK] == 143
assert CRC_EXTRA_BY_MSGID[MSGID_SET_ATTITUDE_TARGET] == 49

WIRE_PAYLOAD_LEN_BY_MSGID: dict[int, int] = {
    MSGID_HEARTBEAT: _HEARTBEAT_STRUCT.size,
    MSGID_ATTITUDE: _ATTITUDE_STRUCT.size,
    MSGID_GLOBAL_POSITION_INT: _GLOBAL_POSITION_INT_STRUCT.size,
    MSGID_VFR_HUD: _VFR_HUD_STRUCT.size,
    MSGID_COMMAND_INT: _COMMAND_INT_STRUCT.size,
    MSGID_COMMAND_LONG: _COMMAND_LONG_STRUCT.size,
    MSGID_COMMAND_ACK: _COMMAND_ACK_V2_STRUCT.size,
    MSGID_SET_ATTITUDE_TARGET: _SET_ATTITUDE_TARGET_STRUCT.size,
    MSGID_POD_DAA_TRACK_ARRAY: _POD_DAA_TRACK_ARRAY_STRUCT.size,
    MSGID_POD_SYNC_HEALTH: _POD_SYNC_HEALTH_STRUCT.size,
    MSGID_POD_VLM_SEMANTIC_VETO: _POD_VLM_SEMANTIC_VETO_STRUCT.size,
    MSGID_POD_PLUGIN_HEALTH: _POD_PLUGIN_HEALTH_STRUCT.size,
}

MAX_WIRE_PAYLOAD_LEN_BY_MSGID: dict[int, int] = {
    **WIRE_PAYLOAD_LEN_BY_MSGID,
    MSGID_COMMAND_ACK: _COMMAND_ACK_V2_STRUCT.size,
    MSGID_SET_ATTITUDE_TARGET: _SET_ATTITUDE_TARGET_MAX_LEN,
}


@dataclass(frozen=True)
class PodDaaTrackArray:
    """MAVLink `#50001 POD_DAA_TRACK_ARRAY` — EO+LWIR civil DAA summary (zero actuator authority)."""

    timestamp_us: int
    track_count: int
    azimuth_mrad: int
    elevation_mrad: int
    ang_rate_mrad_s: int
    t_rad_dk: int
    shadow_avoid_hdg_cdeg: int
    shadow_avoid_alt_cm: int
    flags: int

    MSGID = MSGID_POD_DAA_TRACK_ARRAY
    NAME = "POD_DAA_TRACK_ARRAY"

    def pack_payload(self) -> bytes:
        return _POD_DAA_TRACK_ARRAY_STRUCT.pack(
            int(self.timestamp_us),
            int(self.azimuth_mrad),
            int(self.elevation_mrad),
            int(self.ang_rate_mrad_s),
            int(self.t_rad_dk),
            int(self.shadow_avoid_hdg_cdeg),
            int(self.shadow_avoid_alt_cm),
            int(self.track_count),
            int(self.flags),
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> PodDaaTrackArray:
        (
            timestamp_us,
            azimuth_mrad,
            elevation_mrad,
            ang_rate_mrad_s,
            t_rad_dk,
            shadow_avoid_hdg_cdeg,
            shadow_avoid_alt_cm,
            track_count,
            flags,
        ) = _POD_DAA_TRACK_ARRAY_STRUCT.unpack(raw_padded)
        return cls(
            timestamp_us=timestamp_us,
            track_count=track_count,
            azimuth_mrad=azimuth_mrad,
            elevation_mrad=elevation_mrad,
            ang_rate_mrad_s=ang_rate_mrad_s,
            t_rad_dk=t_rad_dk,
            shadow_avoid_hdg_cdeg=shadow_avoid_hdg_cdeg,
            shadow_avoid_alt_cm=shadow_avoid_alt_cm,
            flags=flags,
        )


@dataclass(frozen=True)
class PodSyncHealth:
    """MAVLink `#50002 POD_SYNC_HEALTH` — 1PPS/PTP lock, Boson+ FFC, Jetson TTP & RTA health."""

    offset_us: int
    bos_ffc_state: int
    jetson_ttp_dc: int
    mig_dropped_frames: int
    cam_health_mask: int
    rta_simplex_mode: int

    MSGID = MSGID_POD_SYNC_HEALTH
    NAME = "POD_SYNC_HEALTH"

    def pack_payload(self) -> bytes:
        return _POD_SYNC_HEALTH_STRUCT.pack(
            int(self.offset_us),
            int(self.jetson_ttp_dc),
            int(self.mig_dropped_frames),
            int(self.bos_ffc_state),
            int(self.cam_health_mask),
            int(self.rta_simplex_mode),
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> PodSyncHealth:
        (
            offset_us,
            jetson_ttp_dc,
            mig_dropped_frames,
            bos_ffc_state,
            cam_health_mask,
            rta_simplex_mode,
        ) = _POD_SYNC_HEALTH_STRUCT.unpack(raw_padded)
        return cls(
            offset_us=offset_us,
            bos_ffc_state=bos_ffc_state,
            jetson_ttp_dc=jetson_ttp_dc,
            mig_dropped_frames=mig_dropped_frames,
            cam_health_mask=cam_health_mask,
            rta_simplex_mode=rta_simplex_mode,
        )


@dataclass(frozen=True)
class PodVlmSemanticVeto:
    """MAVLink `#50003 POD_VLM_SEMANTIC_VETO` — DR-09 IDAL-D/E advisory telemetry."""

    seq: int
    capture_ts_ms: int
    hazard_enum: int
    azimuth_sector: int
    conformal_valid: int
    gate_verdict: int
    ranked_flz_ids: tuple[int, int, int, int] = (255, 255, 255, 255)

    MSGID = MSGID_POD_VLM_SEMANTIC_VETO
    NAME = "POD_VLM_SEMANTIC_VETO"

    def pack_payload(self) -> bytes:
        if len(self.ranked_flz_ids) != 4:
            raise ValueError(f"ranked_flz_ids must have length 4, got {len(self.ranked_flz_ids)}")
        f0, f1, f2, f3 = [int(v) for v in self.ranked_flz_ids]
        return _POD_VLM_SEMANTIC_VETO_STRUCT.pack(
            int(self.seq),
            int(self.capture_ts_ms),
            int(self.hazard_enum),
            int(self.azimuth_sector),
            int(self.conformal_valid),
            int(self.gate_verdict),
            f0,
            f1,
            f2,
            f3,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> PodVlmSemanticVeto:
        seq, capture_ts_ms, hazard_enum, azimuth_sector, conformal_valid, gate_verdict, f0, f1, f2, f3 = (
            _POD_VLM_SEMANTIC_VETO_STRUCT.unpack(raw_padded)
        )
        return cls(
            seq=seq,
            capture_ts_ms=capture_ts_ms,
            hazard_enum=hazard_enum,
            azimuth_sector=azimuth_sector,
            conformal_valid=conformal_valid,
            gate_verdict=gate_verdict,
            ranked_flz_ids=(f0, f1, f2, f3),
        )


@dataclass(frozen=True)
class PodPluginHealth:
    """MAVLink `#50004 POD_PLUGIN_HEALTH` — DR-16 Ring-1 civil plugin health (48 B, 5 Hz)."""

    plugin_id_hash: int
    timestamp_mono_us: int
    last_seq_no: int
    lifecycle_state: int
    civil_mission_class: int
    missed_heartbeats: int
    rss_mb: int
    cpu_permille: int
    reserved_zero: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    ieee_crc32: int = 0

    MSGID = MSGID_POD_PLUGIN_HEALTH
    NAME = "POD_PLUGIN_HEALTH"
    ACTUATOR_AUTHORITY = False
    PAYLOAD_BYTES = 48

    def __post_init__(self) -> None:
        self._validate_invariants()

    def _validate_invariants(self) -> tuple[int, ...]:
        state = int(self.lifecycle_state)
        if state < int(PodPluginLifecycleState.INIT) or state > int(PodPluginLifecycleState.PLUGIN_OFFLINE):
            raise ValueError(f"Invalid lifecycle_state {state}; expected 0..3")
        mission = int(self.civil_mission_class)
        if (
            mission < int(PodCivilMissionClass.WILDFIRE_THERMAL)
            or mission > int(PodCivilMissionClass.MEDICAL_CARGO_LOGISTICS)
        ):
            raise ValueError(
                f"Invalid civil_mission_class {mission}; expected 1..5 (0 rejected as uninitialized/invalid)"
            )
        if len(self.reserved_zero) != 12:
            raise ValueError(f"reserved_zero must have length 12, got {len(self.reserved_zero)}")
        rz = tuple(int(b) for b in self.reserved_zero)
        if any(b != 0 for b in rz):
            raise ValueError(
                "POD_PLUGIN_HEALTH reserved_zero must be all-zero (ACTUATOR_AUTHORITY = false)"
            )
        return rz

    def compute_ieee_crc32(self) -> int:
        """Compute IEEE-802.3 CRC32 over the 44 canonical wire bytes excluding `ieee_crc32`."""
        rz = self._validate_invariants()
        crc_input = _POD_PLUGIN_HEALTH_CRC_INPUT_STRUCT.pack(
            int(self.plugin_id_hash),
            int(self.timestamp_mono_us),
            int(self.last_seq_no),
            int(self.missed_heartbeats),
            int(self.rss_mb),
            int(self.cpu_permille),
            int(self.lifecycle_state),
            int(self.civil_mission_class),
            *rz,
        )
        return zlib.crc32(crc_input) & 0xFFFFFFFF

    def verify_ieee_crc32(self) -> bool:
        """Return True iff `ieee_crc32` matches the canonical 44-byte payload CRC32."""
        return (int(self.ieee_crc32) & 0xFFFFFFFF) == self.compute_ieee_crc32()

    def assert_ieee_crc32(self) -> None:
        """Raise `MavlinkCrcError` if the payload `ieee_crc32` has been tampered or corrupted."""
        expected = self.compute_ieee_crc32()
        actual = int(self.ieee_crc32) & 0xFFFFFFFF
        if actual != expected:
            raise MavlinkCrcError(
                f"POD_PLUGIN_HEALTH IEEE-CRC32 mismatch: rx=0x{actual:08x}, expected=0x{expected:08x}"
            )

    def with_valid_crc(self) -> PodPluginHealth:
        """Return a copy with `ieee_crc32` populated from the canonical wire bytes."""
        return replace(self, ieee_crc32=self.compute_ieee_crc32())

    @classmethod
    def build(
        cls,
        *,
        plugin_id_hash: int,
        timestamp_mono_us: int,
        last_seq_no: int,
        lifecycle_state: int,
        civil_mission_class: int,
        missed_heartbeats: int = 0,
        rss_mb: int = 0,
        cpu_permille: int = 0,
    ) -> PodPluginHealth:
        """Construct a 48-byte `PodPluginHealth` message with a valid IEEE-802.3 CRC32."""
        msg = cls(
            plugin_id_hash=plugin_id_hash,
            timestamp_mono_us=timestamp_mono_us,
            last_seq_no=last_seq_no,
            lifecycle_state=int(lifecycle_state),
            civil_mission_class=int(civil_mission_class),
            missed_heartbeats=missed_heartbeats,
            rss_mb=rss_mb,
            cpu_permille=cpu_permille,
        )
        return msg.with_valid_crc()

    @property
    def is_offline_latched(self) -> bool:
        """Return True when Ring-0 supervisor has permanently latched `PLUGIN_OFFLINE`."""
        return int(self.lifecycle_state) == int(PodPluginLifecycleState.PLUGIN_OFFLINE)

    def pack_payload(self) -> bytes:
        rz = self._validate_invariants()
        return _POD_PLUGIN_HEALTH_STRUCT.pack(
            int(self.plugin_id_hash),
            int(self.timestamp_mono_us),
            int(self.last_seq_no),
            int(self.ieee_crc32) & 0xFFFFFFFF,
            int(self.missed_heartbeats),
            int(self.rss_mb),
            int(self.cpu_permille),
            int(self.lifecycle_state),
            int(self.civil_mission_class),
            *rz,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> PodPluginHealth:
        unpacked = _POD_PLUGIN_HEALTH_STRUCT.unpack(raw_padded)
        (
            plugin_id_hash,
            timestamp_mono_us,
            last_seq_no,
            ieee_crc32,
            missed_heartbeats,
            rss_mb,
            cpu_permille,
            lifecycle_state,
            civil_mission_class,
        ) = unpacked[:9]
        rz = tuple(int(b) for b in unpacked[9:21])
        if any(b != 0 for b in rz):
            raise MavlinkFrameError(
                "POD_PLUGIN_HEALTH non-zero reserved_zero bytes rejected (ACTUATOR_AUTHORITY = false)"
            )
        if lifecycle_state > int(PodPluginLifecycleState.PLUGIN_OFFLINE):
            raise MavlinkFrameError(f"Invalid POD_PLUGIN_HEALTH lifecycle_state: {lifecycle_state}")
        if (
            civil_mission_class < int(PodCivilMissionClass.WILDFIRE_THERMAL)
            or civil_mission_class > int(PodCivilMissionClass.MEDICAL_CARGO_LOGISTICS)
        ):
            raise MavlinkFrameError(
                f"Invalid POD_PLUGIN_HEALTH civil_mission_class: {civil_mission_class} (expected 1..5)"
            )
        return cls(
            plugin_id_hash=plugin_id_hash,
            timestamp_mono_us=timestamp_mono_us,
            last_seq_no=last_seq_no,
            lifecycle_state=lifecycle_state,
            civil_mission_class=civil_mission_class,
            missed_heartbeats=missed_heartbeats,
            rss_mb=rss_mb,
            cpu_permille=cpu_permille,
            reserved_zero=rz,
            ieee_crc32=ieee_crc32,
        )


@dataclass(frozen=True)
class MavHeartbeat:
    """MAVLink `#0 HEARTBEAT` (`CRC_EXTRA=50`, 9 B)."""

    custom_mode: int = 0
    mav_type: int = 18  # MAV_TYPE_ONBOARD_CONTROLLER
    autopilot: int = 8  # MAV_AUTOPILOT_INVALID (companion) / 3 = MAV_AUTOPILOT_ARDUPILOTMEGA
    base_mode: int = 0
    system_status: int = 4  # MAV_STATE_ACTIVE
    mavlink_version: int = 3

    MSGID = MSGID_HEARTBEAT
    NAME = "HEARTBEAT"

    def pack_payload(self) -> bytes:
        return _HEARTBEAT_STRUCT.pack(
            int(self.custom_mode) & 0xFFFFFFFF,
            int(self.mav_type) & 0xFF,
            int(self.autopilot) & 0xFF,
            int(self.base_mode) & 0xFF,
            int(self.system_status) & 0xFF,
            int(self.mavlink_version) & 0xFF,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> MavHeartbeat:
        custom_mode, mav_type, autopilot, base_mode, system_status, mavlink_version = (
            _HEARTBEAT_STRUCT.unpack(raw_padded[: _HEARTBEAT_STRUCT.size])
        )
        return cls(
            custom_mode=int(custom_mode),
            mav_type=int(mav_type),
            autopilot=int(autopilot),
            base_mode=int(base_mode),
            system_status=int(system_status),
            mavlink_version=int(mavlink_version),
        )


@dataclass(frozen=True)
class MavAttitude:
    """MAVLink `#30 ATTITUDE` (`CRC_EXTRA=39`, 28 B, NED Euler angles in rad)."""

    time_boot_ms: int
    roll: float
    pitch: float
    yaw: float
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    MSGID = MSGID_ATTITUDE
    NAME = "ATTITUDE"

    def __post_init__(self) -> None:
        _require_finite_float("roll", self.roll)
        _require_finite_float("pitch", self.pitch)
        _require_finite_float("yaw", self.yaw)
        _require_finite_float("rollspeed", self.rollspeed)
        _require_finite_float("pitchspeed", self.pitchspeed)
        _require_finite_float("yawspeed", self.yawspeed)

    def pack_payload(self) -> bytes:
        return _ATTITUDE_STRUCT.pack(
            int(self.time_boot_ms) & 0xFFFFFFFF,
            float(self.roll),
            float(self.pitch),
            float(self.yaw),
            float(self.rollspeed),
            float(self.pitchspeed),
            float(self.yawspeed),
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> MavAttitude:
        time_boot_ms, roll, pitch, yaw, rollspeed, pitchspeed, yawspeed = (
            _ATTITUDE_STRUCT.unpack(raw_padded[: _ATTITUDE_STRUCT.size])
        )
        return cls(
            time_boot_ms=int(time_boot_ms),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            rollspeed=float(rollspeed),
            pitchspeed=float(pitchspeed),
            yawspeed=float(yawspeed),
        )


@dataclass(frozen=True)
class MavGlobalPositionInt:
    """MAVLink `#33 GLOBAL_POSITION_INT` (`CRC_EXTRA=104`, 28 B, strictly NED velocities)."""

    time_boot_ms: int
    lat_e7: int
    lon_e7: int
    alt_mm: int
    relative_alt_mm: int
    vx_cms: int
    vy_cms: int
    vz_cms: int
    hdg_cdeg: int = 65535

    MSGID = MSGID_GLOBAL_POSITION_INT
    NAME = "GLOBAL_POSITION_INT"

    def __post_init__(self) -> None:
        if not (-900_000_000 <= int(self.lat_e7) <= 900_000_000):
            raise ValueError(f"GLOBAL_POSITION_INT lat_e7 out of range [-90e7, 90e7]: {self.lat_e7}")
        if not (-1_800_000_000 <= int(self.lon_e7) <= 1_800_000_000):
            raise ValueError(f"GLOBAL_POSITION_INT lon_e7 out of range [-180e7, 180e7]: {self.lon_e7}")
        alt_val = int(self.alt_mm)
        if not (MIN_GLOBAL_POSITION_ALT_MM <= alt_val <= MAX_GLOBAL_POSITION_ALT_MM):
            raise ValueError(
                f"GLOBAL_POSITION_INT alt_mm out of range "
                f"[{MIN_GLOBAL_POSITION_ALT_MM}, {MAX_GLOBAL_POSITION_ALT_MM}]: {self.alt_mm}"
            )
        rel_alt_val = int(self.relative_alt_mm)
        if not (MIN_GLOBAL_POSITION_ALT_MM <= rel_alt_val <= MAX_GLOBAL_POSITION_ALT_MM):
            raise ValueError(
                f"GLOBAL_POSITION_INT relative_alt_mm out of range "
                f"[{MIN_GLOBAL_POSITION_ALT_MM}, {MAX_GLOBAL_POSITION_ALT_MM}]: {self.relative_alt_mm}"
            )
        hdg_val = int(self.hdg_cdeg)
        if not ((0 <= hdg_val <= 36000) or (hdg_val == MAV_HEADING_UNKNOWN_CDEG)):
            raise ValueError(
                f"GLOBAL_POSITION_INT hdg_cdeg out of range [0, 36000] or {MAV_HEADING_UNKNOWN_CDEG}: {self.hdg_cdeg}"
            )

    @property
    def lat_deg(self) -> float:
        return float(self.lat_e7) * 1e-7

    @property
    def lon_deg(self) -> float:
        return float(self.lon_e7) * 1e-7

    @property
    def alt_msl_m(self) -> float:
        return float(self.alt_mm) * 1e-3

    @property
    def down_m(self) -> float:
        return -self.alt_msl_m

    @property
    def relative_alt_m(self) -> float:
        return float(self.relative_alt_mm) * 1e-3

    @property
    def vn_mps(self) -> float:
        return float(self.vx_cms) * 1e-2

    @property
    def ve_mps(self) -> float:
        return float(self.vy_cms) * 1e-2

    @property
    def vd_mps(self) -> float:
        return float(self.vz_cms) * 1e-2

    @property
    def heading_deg(self) -> float:
        if int(self.hdg_cdeg) == 65535:
            return math.degrees(math.atan2(self.ve_mps, self.vn_mps)) % 360.0
        return (float(self.hdg_cdeg) * 1e-2) % 360.0

    def pack_payload(self) -> bytes:
        return _GLOBAL_POSITION_INT_STRUCT.pack(
            int(self.time_boot_ms) & 0xFFFFFFFF,
            int(self.lat_e7),
            int(self.lon_e7),
            int(self.alt_mm),
            int(self.relative_alt_mm),
            int(self.vx_cms),
            int(self.vy_cms),
            int(self.vz_cms),
            int(self.hdg_cdeg) & 0xFFFF,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> MavGlobalPositionInt:
        time_boot_ms, lat, lon, alt, rel_alt, vx, vy, vz, hdg = (
            _GLOBAL_POSITION_INT_STRUCT.unpack(raw_padded[: _GLOBAL_POSITION_INT_STRUCT.size])
        )
        return cls(
            time_boot_ms=int(time_boot_ms),
            lat_e7=int(lat),
            lon_e7=int(lon),
            alt_mm=int(alt),
            relative_alt_mm=int(rel_alt),
            vx_cms=int(vx),
            vy_cms=int(vy),
            vz_cms=int(vz),
            hdg_cdeg=int(hdg),
        )


@dataclass(frozen=True)
class MavVfrHud:
    """MAVLink `#74 VFR_HUD` (`CRC_EXTRA=20`, 20 B, true/indicated airspeed & groundspeed)."""

    airspeed: float
    groundspeed: float
    heading: int = 0
    throttle: int = 0
    alt: float = 0.0
    climb: float = 0.0

    MSGID = MSGID_VFR_HUD
    NAME = "VFR_HUD"

    def __post_init__(self) -> None:
        ias = _require_finite_float("airspeed", self.airspeed)
        gs = _require_finite_float("groundspeed", self.groundspeed)
        _require_finite_float("alt", self.alt)
        _require_finite_float("climb", self.climb)
        if ias < 0.0:
            raise ValueError(f"VFR_HUD airspeed must be >= 0.0 m/s, got {self.airspeed}")
        if gs < 0.0:
            raise ValueError(f"VFR_HUD groundspeed must be >= 0.0 m/s, got {self.groundspeed}")
        hdg = int(self.heading)
        if not (0 <= hdg <= 360):
            raise ValueError(f"VFR_HUD heading out of range [0, 360]: {self.heading}")
        thr = int(self.throttle)
        if not (0 <= thr <= 100):
            raise ValueError(f"VFR_HUD throttle out of range [0, 100]: {self.throttle}")

    def pack_payload(self) -> bytes:
        return _VFR_HUD_STRUCT.pack(
            float(self.airspeed),
            float(self.groundspeed),
            float(self.alt),
            float(self.climb),
            int(self.heading),
            int(self.throttle),
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> MavVfrHud:
        airspeed, groundspeed, alt, climb, heading, throttle = _VFR_HUD_STRUCT.unpack(
            raw_padded[: _VFR_HUD_STRUCT.size]
        )
        return cls(
            airspeed=float(airspeed),
            groundspeed=float(groundspeed),
            heading=int(heading),
            throttle=int(throttle),
            alt=float(alt),
            climb=float(climb),
        )


@dataclass(frozen=True)
class MavCommandInt:
    """MAVLink `#75 COMMAND_INT` (`CRC_EXTRA=158`, 35 B)."""

    command: int
    frame: int = 6  # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    x: int = 0
    y: int = 0
    z: float = 0.0
    target_system: int = 1
    target_component: int = 1
    current: int = 0
    autocontinue: int = 0

    MSGID = MSGID_COMMAND_INT
    NAME = "COMMAND_INT"

    def __post_init__(self) -> None:
        _require_finite_float("param1", self.param1)
        _require_finite_float("param2", self.param2)
        _require_finite_float("param3", self.param3)
        _require_finite_float("param4", self.param4)
        _require_finite_float("z", self.z)

    def pack_payload(self) -> bytes:
        return _COMMAND_INT_STRUCT.pack(
            float(self.param1),
            float(self.param2),
            float(self.param3),
            float(self.param4),
            int(self.x),
            int(self.y),
            float(self.z),
            int(self.command) & 0xFFFF,
            int(self.target_system) & 0xFF,
            int(self.target_component) & 0xFF,
            int(self.frame) & 0xFF,
            int(self.current) & 0xFF,
            int(self.autocontinue) & 0xFF,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> MavCommandInt:
        p1, p2, p3, p4, x, y, z, cmd, tgt_sys, tgt_comp, frame, current, autocont = (
            _COMMAND_INT_STRUCT.unpack(raw_padded[: _COMMAND_INT_STRUCT.size])
        )
        return cls(
            command=int(cmd),
            frame=int(frame),
            param1=float(p1),
            param2=float(p2),
            param3=float(p3),
            param4=float(p4),
            x=int(x),
            y=int(y),
            z=float(z),
            target_system=int(tgt_sys),
            target_component=int(tgt_comp),
            current=int(current),
            autocontinue=int(autocont),
        )


@dataclass(frozen=True)
class MavCommandLong:
    """MAVLink `#76 COMMAND_LONG` (`CRC_EXTRA=152`, 33 B)."""

    command: int
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    param5: float = 0.0
    param6: float = 0.0
    param7: float = 0.0
    target_system: int = 1
    target_component: int = 1
    confirmation: int = 0

    MSGID = MSGID_COMMAND_LONG
    NAME = "COMMAND_LONG"

    def __post_init__(self) -> None:
        for idx, val in enumerate(
            (self.param1, self.param2, self.param3, self.param4, self.param5, self.param6, self.param7),
            start=1,
        ):
            _require_finite_float(f"param{idx}", val)

    def pack_payload(self) -> bytes:
        return _COMMAND_LONG_STRUCT.pack(
            float(self.param1),
            float(self.param2),
            float(self.param3),
            float(self.param4),
            float(self.param5),
            float(self.param6),
            float(self.param7),
            int(self.command) & 0xFFFF,
            int(self.target_system) & 0xFF,
            int(self.target_component) & 0xFF,
            int(self.confirmation) & 0xFF,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> MavCommandLong:
        p1, p2, p3, p4, p5, p6, p7, cmd, tgt_sys, tgt_comp, conf = (
            _COMMAND_LONG_STRUCT.unpack(raw_padded[: _COMMAND_LONG_STRUCT.size])
        )
        return cls(
            command=int(cmd),
            param1=float(p1),
            param2=float(p2),
            param3=float(p3),
            param4=float(p4),
            param5=float(p5),
            param6=float(p6),
            param7=float(p7),
            target_system=int(tgt_sys),
            target_component=int(tgt_comp),
            confirmation=int(conf),
        )


@dataclass(frozen=True)
class MavCommandAck:
    """MAVLink `#77 COMMAND_ACK` (`CRC_EXTRA=143`, 3 B base / 10 B v2 extension)."""

    command: int
    result: int
    progress: int = 0
    result_param2: int = 0
    target_system: int = 0
    target_component: int = 0

    MSGID = MSGID_COMMAND_ACK
    NAME = "COMMAND_ACK"

    def pack_payload(self) -> bytes:
        return _COMMAND_ACK_V2_STRUCT.pack(
            int(self.command) & 0xFFFF,
            int(self.result) & 0xFF,
            int(self.progress) & 0xFF,
            int(self.result_param2),
            int(self.target_system) & 0xFF,
            int(self.target_component) & 0xFF,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> MavCommandAck:
        cmd, result, progress, res_p2, tgt_sys, tgt_comp = (
            _COMMAND_ACK_V2_STRUCT.unpack(raw_padded[: _COMMAND_ACK_V2_STRUCT.size])
        )
        return cls(
            command=int(cmd),
            result=int(result),
            progress=int(progress),
            result_param2=int(res_p2),
            target_system=int(tgt_sys),
            target_component=int(tgt_comp),
        )


@dataclass(frozen=True)
class MavSetAttitudeTarget:
    """MAVLink `#82 SET_ATTITUDE_TARGET` (`CRC_EXTRA=49`, 39 B base, Hamilton `q=(w,x,y,z)`)."""

    time_boot_ms: int
    q: tuple[float, float, float, float]
    body_roll_rate: float = 0.0
    body_pitch_rate: float = 0.0
    body_yaw_rate: float = 0.0
    thrust: float = 0.5
    target_system: int = 1
    target_component: int = 1
    type_mask: int = 0x07

    MSGID = MSGID_SET_ATTITUDE_TARGET
    NAME = "SET_ATTITUDE_TARGET"

    def __post_init__(self) -> None:
        if len(self.q) != 4:
            raise ValueError(f"SET_ATTITUDE_TARGET quaternion must have 4 elements, got {len(self.q)}")
        for idx, c in enumerate(self.q):
            _require_finite_float(f"q[{idx}]", c)
        _require_finite_float("body_roll_rate", self.body_roll_rate)
        _require_finite_float("body_pitch_rate", self.body_pitch_rate)
        _require_finite_float("body_yaw_rate", self.body_yaw_rate)
        _require_finite_float("thrust", self.thrust)

    def pack_payload(self) -> bytes:
        return _SET_ATTITUDE_TARGET_STRUCT.pack(
            int(self.time_boot_ms) & 0xFFFFFFFF,
            float(self.q[0]),
            float(self.q[1]),
            float(self.q[2]),
            float(self.q[3]),
            float(self.body_roll_rate),
            float(self.body_pitch_rate),
            float(self.body_yaw_rate),
            float(self.thrust),
            int(self.target_system) & 0xFF,
            int(self.target_component) & 0xFF,
            int(self.type_mask) & 0xFF,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> MavSetAttitudeTarget:
        t_ms, qw, qx, qy, qz, rr, pr, yr, thrust, tgt_sys, tgt_comp, mask = (
            _SET_ATTITUDE_TARGET_STRUCT.unpack(raw_padded[: _SET_ATTITUDE_TARGET_STRUCT.size])
        )
        return cls(
            time_boot_ms=int(t_ms),
            q=(float(qw), float(qx), float(qy), float(qz)),
            body_roll_rate=float(rr),
            body_pitch_rate=float(pr),
            body_yaw_rate=float(yr),
            thrust=float(thrust),
            target_system=int(tgt_sys),
            target_component=int(tgt_comp),
            type_mask=int(mask),
        )


StandardMavlinkMessage = (
    MavHeartbeat
    | MavAttitude
    | MavGlobalPositionInt
    | MavVfrHud
    | MavCommandInt
    | MavCommandLong
    | MavCommandAck
    | MavSetAttitudeTarget
)
PodMessage = PodDaaTrackArray | PodSyncHealth | PodVlmSemanticVeto | PodPluginHealth
MavlinkMessage = PodMessage | StandardMavlinkMessage

_MSG_CLASS_BY_ID: dict[int, type[Any]] = {
    MSGID_HEARTBEAT: MavHeartbeat,
    MSGID_ATTITUDE: MavAttitude,
    MSGID_GLOBAL_POSITION_INT: MavGlobalPositionInt,
    MSGID_VFR_HUD: MavVfrHud,
    MSGID_COMMAND_INT: MavCommandInt,
    MSGID_COMMAND_LONG: MavCommandLong,
    MSGID_COMMAND_ACK: MavCommandAck,
    MSGID_SET_ATTITUDE_TARGET: MavSetAttitudeTarget,
    MSGID_POD_DAA_TRACK_ARRAY: PodDaaTrackArray,
    MSGID_POD_SYNC_HEALTH: PodSyncHealth,
    MSGID_POD_VLM_SEMANTIC_VETO: PodVlmSemanticVeto,
    MSGID_POD_PLUGIN_HEALTH: PodPluginHealth,
}


@dataclass(frozen=True)
class MavlinkV2Frame:
    """Decoded MAVLink 2.0 frame with verified header metadata, CRC, and optional SHA-256 signature."""

    seq: int
    sysid: int
    compid: int
    msgid: int
    incompat_flags: int
    compat_flags: int
    payload_len: int
    crc16: int
    message: MavlinkMessage
    signed: bool = False
    link_id: int | None = None
    timestamp_48: int | None = None
    signature_verified: bool = False


def _truncate_mavlink_v2_payload(payload: bytes) -> bytes:
    """MAVLink v2 trailing-zero truncation (keeps at least 1 byte if input non-empty)."""
    if not payload:
        return b""
    end = len(payload)
    while end > 1 and payload[end - 1] == 0:
        end -= 1
    return payload[:end]


def _pack_timestamp_48(timestamp_48: int) -> bytes:
    ts = int(timestamp_48)
    if ts < 0 or ts >= (1 << 48):
        raise ValueError(f"MAVLink v2 signing timestamp_48 out of 48-bit range: {ts}")
    return ts.to_bytes(6, byteorder="little", signed=False)


def _compute_mavlink_v2_signature(
    *,
    secret_key: bytes,
    unsigned_frame: bytes,
    link_id: int,
    timestamp_48: int,
) -> bytes:
    """Build exact 13-byte MAVLink 2.0 SHA-256 signature trailer (`link_id + ts48 + sha256[:6]`)."""
    if len(secret_key) != 32:
        raise ValueError(f"MAVLink v2 signing secret_key must be exactly 32 bytes, got {len(secret_key)}")
    sig_hdr = bytes([int(link_id) & 0xFF]) + _pack_timestamp_48(timestamp_48)
    sig_6 = hashlib.sha256(secret_key + unsigned_frame + sig_hdr).digest()[:6]
    return sig_hdr + sig_6


def validate_dialect_msgid(msgid: int) -> None:
    """Enforce conflict-free custom dialect ID range `50000..50099` (`DR-11` §4)."""
    mid = int(msgid)
    for lo, hi in FORBIDDEN_DIALECT_ID_RANGES:
        if lo <= mid <= hi:
            raise ValueError(
                f"Dialect message ID {mid} overlaps reserved MAVLink/ArduPilot range [{lo}, {hi}]"
            )
    if not (CUSTOM_DIALECT_ID_MIN <= mid <= CUSTOM_DIALECT_ID_MAX):
        raise ValueError(
            f"Custom dialect message ID {mid} must lie in [{CUSTOM_DIALECT_ID_MIN}, {CUSTOM_DIALECT_ID_MAX}]"
        )


def validate_ardupilot_odometry_version(version: tuple[int, int, int]) -> None:
    """Require `ArduPlane >= (4, 7, 2)` for `#331 ODOMETRY` (blocks Issue `#32506` EKF3 Z-snap)."""
    if len(version) != 3:
        raise ArduPlaneSafetyError(f"Expected 3-tuple (major, minor, patch), got {version!r}")
    norm = (int(version[0]), int(version[1]), int(version[2]))
    if norm < MIN_ARDUPILOT_ODOMETRY_VERSION:
        raise ArduPlaneSafetyError(
            f"ArduPlane {norm[0]}.{norm[1]}.{norm[2]} is blocked for MAVLink #331 ODOMETRY due to "
            f"Issue #32506 (EKF3 vertical Z snap); ArduPlane >= 4.7.2 is mandatory."
        )


def validate_outbound_msgid(
    msgid: int,
    *,
    ardupilot_version: tuple[int, int, int] | None = None,
) -> None:
    """Enforce DR-11 outbound MAVLink safety locks (`#84` hard ban, `#331` >= 4.7.2 gate)."""
    mid = int(msgid)
    if mid in BANNED_MAVLINK_MSGIDS:
        raise ArduPlaneSafetyError(
            "MAVLink #84 (SET_POSITION_TARGET_LOCAL_NED) is hard-banned on OpenXVS ArduPlane "
            "(unsupported X/Y local pos/vel in GUIDED mode per ArduPilot #19858 & DR-11 §4)."
        )
    if mid == MSGID_ODOMETRY:
        if ardupilot_version is None:
            raise ArduPlaneSafetyError(
                "MAVLink #331 ODOMETRY requires explicit verified ArduPlane version >= 4.7.2 (Issue #32506)."
            )
        validate_ardupilot_odometry_version(ardupilot_version)


def encode_frame(
    message: MavlinkMessage,
    *,
    seq: int = 0,
    sysid: int = 1,
    compid: int = 191,
    incompat_flags: int = 0,
    compat_flags: int = 0,
    truncate_zeros: bool = True,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
) -> bytes:
    """Encode a `pod_daa.xml` or standard MAVLink 2.0 message into a verified binary frame.

    Supports optional 13-byte MAVLink 2.0 SHA-256 signing (`incompat_flags=0x01`) when
    `secret_key` (32 bytes) is provided. Hard-bans `#84 SET_POSITION_TARGET_LOCAL_NED`.
    """
    msgid = int(message.MSGID)
    validate_outbound_msgid(msgid)
    if msgid not in STANDARD_MAVLINK_MSGIDS:
        validate_dialect_msgid(msgid)

    if msgid not in CRC_EXTRA_BY_MSGID:
        raise MavlinkFrameError(f"Unsupported MAVLink message ID: {msgid}")

    eff_incompat = int(incompat_flags) & 0xFF
    if secret_key is not None:
        eff_incompat |= MAVLINK_IFLAG_SIGNED

    if (eff_incompat & ~MAVLINK_IFLAG_SIGNED) != 0:
        raise MavlinkFrameError(
            f"Unsupported non-zero MAVLink v2 incompat_flags: 0x{eff_incompat:02x} (only 0x00 or 0x01 supported)"
        )
    signed = (eff_incompat & MAVLINK_IFLAG_SIGNED) != 0
    if signed and secret_key is None:
        raise MavlinkFrameError(
            "MAVLink v2 incompat_flags=0x01 (MAVLINK_IFLAG_SIGNED) requires a 32-byte secret_key"
        )

    full_payload = message.pack_payload()
    wire_payload = _truncate_mavlink_v2_payload(full_payload) if truncate_zeros else full_payload
    if len(wire_payload) > 255:
        raise MavlinkFrameError(f"Payload length {len(wire_payload)} exceeds MAVLink v2 limit 255")

    header_body = struct.pack(
        "<BBBBBBBBB",
        len(wire_payload),
        eff_incompat,
        int(compat_flags) & 0xFF,
        int(seq) & 0xFF,
        int(sysid) & 0xFF,
        int(compid) & 0xFF,
        msgid & 0xFF,
        (msgid >> 8) & 0xFF,
        (msgid >> 16) & 0xFF,
    )
    crc_extra = CRC_EXTRA_BY_MSGID[msgid]
    crc = x25_crc16(header_body + wire_payload + bytes([crc_extra]))
    unsigned_frame = bytes([MAVLINK_V2_STX]) + header_body + wire_payload + struct.pack("<H", crc)
    if not signed:
        return unsigned_frame

    assert secret_key is not None
    sig_trailer = _compute_mavlink_v2_signature(
        secret_key=secret_key,
        unsigned_frame=unsigned_frame,
        link_id=link_id,
        timestamp_48=timestamp_48,
    )
    return unsigned_frame + sig_trailer


def peek_mavlink_v2_header(frame_bytes: bytes) -> dict[str, int]:
    """Inspect a MAVLink 2.0 frame header without decoding its payload (still validates STX/size)."""
    if len(frame_bytes) < MAVLINK_V2_OVERHEAD_BYTES:
        raise MavlinkFrameError(
            f"Frame too short ({len(frame_bytes)} B < {MAVLINK_V2_OVERHEAD_BYTES} B minimum)"
        )
    if frame_bytes[0] != MAVLINK_V2_STX:
        raise MavlinkFrameError(f"Invalid MAVLink v2 STX magic: 0x{frame_bytes[0]:02x} (expected 0xFD)")
    payload_len = int(frame_bytes[1])
    incompat_flags = int(frame_bytes[2])
    signed = (incompat_flags & MAVLINK_IFLAG_SIGNED) != 0
    total_len = (
        MAVLINK_V2_HEADER_LEN
        + payload_len
        + MAVLINK_V2_CHECKSUM_LEN
        + (MAVLINK_V2_SIGNATURE_BYTES if signed else 0)
    )
    msgid = int(frame_bytes[7]) | (int(frame_bytes[8]) << 8) | (int(frame_bytes[9]) << 16)
    return {
        "payload_len": payload_len,
        "incompat_flags": incompat_flags,
        "compat_flags": int(frame_bytes[3]),
        "seq": int(frame_bytes[4]),
        "sysid": int(frame_bytes[5]),
        "compid": int(frame_bytes[6]),
        "msgid": msgid,
        "signed": int(signed),
        "total_len": total_len,
    }


def extract_mavlink_v2_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Slice zero or more complete MAVLink 2.0 (`0xFD`) frames from a byte stream/datagram."""
    frames: list[bytes] = []
    idx = 0
    n = len(buffer)
    while idx < n:
        stx = buffer.find(bytes([MAVLINK_V2_STX]), idx)
        if stx < 0:
            return frames, b""
        if n - stx < MAVLINK_V2_HEADER_LEN:
            return frames, buffer[stx:]
        payload_len = int(buffer[stx + 1])
        incompat_flags = int(buffer[stx + 2])
        if (incompat_flags & ~MAVLINK_IFLAG_SIGNED) != 0:
            # Resync past corrupted STX byte inside arbitrary stream
            idx = stx + 1
            continue
        sig_len = MAVLINK_V2_SIGNATURE_BYTES if (incompat_flags & MAVLINK_IFLAG_SIGNED) != 0 else 0
        total_len = MAVLINK_V2_HEADER_LEN + payload_len + MAVLINK_V2_CHECKSUM_LEN + sig_len
        if n - stx < total_len:
            return frames, buffer[stx:]
        frames.append(buffer[stx : stx + total_len])
        idx = stx + total_len
    return frames, b""


def decode_frame(
    frame_bytes: bytes,
    *,
    verify_inner_crc32: bool = False,
    secret_key: bytes | None = None,
    require_signature: bool = False,
) -> MavlinkV2Frame:
    """Decode and CRC/signature-verify a MAVLink 2.0 binary frame (`#0,#30,#33,#75,#76,#77,#82,#50001..#50004`)."""
    if len(frame_bytes) < MAVLINK_V2_OVERHEAD_BYTES:
        raise MavlinkFrameError(
            f"Frame too short ({len(frame_bytes)} B < {MAVLINK_V2_OVERHEAD_BYTES} B minimum)"
        )
    if frame_bytes[0] != MAVLINK_V2_STX:
        raise MavlinkFrameError(f"Invalid MAVLink v2 STX magic: 0x{frame_bytes[0]:02x} (expected 0xFD)")

    payload_len = int(frame_bytes[1])
    incompat_flags = int(frame_bytes[2])
    compat_flags = int(frame_bytes[3])
    seq = int(frame_bytes[4])
    sysid = int(frame_bytes[5])
    compid = int(frame_bytes[6])
    msgid = int(frame_bytes[7]) | (int(frame_bytes[8]) << 8) | (int(frame_bytes[9]) << 16)

    if (incompat_flags & ~MAVLINK_IFLAG_SIGNED) != 0:
        raise MavlinkFrameError(
            f"Unsupported non-zero MAVLink v2 incompat_flags: 0x{incompat_flags:02x} (only 0x00 or 0x01 supported)"
        )

    validate_outbound_msgid(msgid)
    signed = (incompat_flags & MAVLINK_IFLAG_SIGNED) != 0
    if require_signature and not signed:
        raise MavlinkFrameError("Unsigned MAVLink v2 frame rejected when require_signature=True")

    sig_bytes = MAVLINK_V2_SIGNATURE_BYTES if signed else 0
    expected_total = MAVLINK_V2_HEADER_LEN + payload_len + MAVLINK_V2_CHECKSUM_LEN + sig_bytes
    if len(frame_bytes) != expected_total:
        raise MavlinkFrameError(
            f"Frame length mismatch: got {len(frame_bytes)} B, header declares {expected_total} B"
        )
    if msgid not in _MSG_CLASS_BY_ID:
        raise MavlinkFrameError(f"Unknown MAVLink message ID: {msgid}")

    wire_payload = frame_bytes[MAVLINK_V2_HEADER_LEN : MAVLINK_V2_HEADER_LEN + payload_len]
    crc_off = MAVLINK_V2_HEADER_LEN + payload_len
    rx_crc = struct.unpack_from("<H", frame_bytes, crc_off)[0]
    expected_crc = x25_crc16(
        frame_bytes[1:MAVLINK_V2_HEADER_LEN] + wire_payload + bytes([CRC_EXTRA_BY_MSGID[msgid]])
    )
    if rx_crc != expected_crc:
        raise MavlinkCrcError(
            f"MAVLink v2 CRC mismatch for msgid={msgid}: rx=0x{rx_crc:04x}, expected=0x{expected_crc:04x}"
        )

    link_id: int | None = None
    timestamp_48: int | None = None
    signature_verified = False
    if signed:
        sig_start = crc_off + MAVLINK_V2_CHECKSUM_LEN
        sig_hdr = frame_bytes[sig_start : sig_start + 7]
        sig_rx = frame_bytes[sig_start + 7 : sig_start + MAVLINK_V2_SIGNATURE_BYTES]
        link_id = int(sig_hdr[0])
        timestamp_48 = int.from_bytes(sig_hdr[1:7], byteorder="little", signed=False)
        if secret_key is not None:
            if len(secret_key) != 32:
                raise ValueError(f"MAVLink v2 signing secret_key must be exactly 32 bytes, got {len(secret_key)}")
            unsigned_part = frame_bytes[:sig_start]
            sig_calc = hashlib.sha256(secret_key + unsigned_part + sig_hdr).digest()[:6]
            if not hmac.compare_digest(sig_rx, sig_calc):
                raise MavlinkCrcError(f"MAVLink v2 SHA-256 signature mismatch for msgid={msgid}")
            signature_verified = True
        elif require_signature:
            raise ValueError("secret_key is required when require_signature=True")

    base_wire_size = WIRE_PAYLOAD_LEN_BY_MSGID[msgid]
    max_wire_size = MAX_WIRE_PAYLOAD_LEN_BY_MSGID[msgid]
    if payload_len > max_wire_size:
        raise MavlinkFrameError(
            f"Payload length {payload_len} exceeds schema wire size {max_wire_size} for msgid={msgid}"
        )
    padded = (
        wire_payload + bytes(base_wire_size - payload_len)
        if payload_len < base_wire_size
        else wire_payload[:base_wire_size]
    )
    msg_cls = _MSG_CLASS_BY_ID[msgid]
    message = msg_cls.unpack_payload(padded)
    if verify_inner_crc32 and isinstance(message, PodPluginHealth):
        message.assert_ieee_crc32()
    return MavlinkV2Frame(
        seq=seq,
        sysid=sysid,
        compid=compid,
        msgid=msgid,
        incompat_flags=incompat_flags,
        compat_flags=compat_flags,
        payload_len=payload_len,
        crc16=rx_crc,
        message=message,
        signed=signed,
        link_id=link_id,
        timestamp_48=timestamp_48,
        signature_verified=signature_verified,
    )


@dataclass(frozen=True)
class MavlinkStreamSpec:
    """Single periodic MAVLink v2 stream used in UART bidirectional bandwidth budgeting."""

    name: str
    msgid: int
    direction: str  # "outbound" (pod->FC) or "inbound" (FC->pod)
    payload_bytes: int
    rate_hz: float

    @property
    def frame_bytes(self) -> int:
        return MAVLINK_V2_OVERHEAD_BYTES + int(self.payload_bytes)

    @property
    def bytes_per_sec(self) -> float:
        return float(self.frame_bytes) * float(self.rate_hz)


DEFAULT_POD_UART_STREAMS: tuple[MavlinkStreamSpec, ...] = (
    # Outbound: Companion Pod -> Flight Controller (ArduPlane 4.7.2+)
    MavlinkStreamSpec("POD_DAA_TRACK_ARRAY", MSGID_POD_DAA_TRACK_ARRAY, "outbound", 22, 20.0),
    MavlinkStreamSpec("POD_SYNC_HEALTH", MSGID_POD_SYNC_HEALTH, "outbound", 11, 5.0),
    MavlinkStreamSpec("POD_VLM_SEMANTIC_VETO", MSGID_POD_VLM_SEMANTIC_VETO, "outbound", 16, 5.0),
    MavlinkStreamSpec("ODOMETRY", MSGID_ODOMETRY, "outbound", 232, 30.0),
    MavlinkStreamSpec("ADSB_VEHICLE", 246, "outbound", 38, 20.0),
    MavlinkStreamSpec("HEARTBEAT_POD", 0, "outbound", 9, 1.0),
    # Inbound: Flight Controller -> Companion Pod
    MavlinkStreamSpec("ATTITUDE_QUATERNION", 31, "inbound", 32, 50.0),
    MavlinkStreamSpec("GLOBAL_POSITION_INT", 33, "inbound", 28, 20.0),
    MavlinkStreamSpec("HIGHRES_IMU", 105, "inbound", 62, 30.0),
    MavlinkStreamSpec("EKF_STATUS_REPORT", 193, "inbound", 22, 5.0),
    MavlinkStreamSpec("HEARTBEAT_FC", 0, "inbound", 9, 1.0),
)


DEFAULT_V3_SIGNED_UART_STREAMS: tuple[MavlinkStreamSpec, ...] = (
    # Outbound: OpenXVS-OS V3.0 Ring-0 -> Flight Controller (ArduPlane 4.7.3+, DR-16 §3)
    MavlinkStreamSpec("ODOMETRY", MSGID_ODOMETRY, "outbound", 232, 30.0),
    MavlinkStreamSpec("POD_DAA_TRACK_ARRAY", MSGID_POD_DAA_TRACK_ARRAY, "outbound", 22, 20.0),
    MavlinkStreamSpec("POD_SYNC_HEALTH", MSGID_POD_SYNC_HEALTH, "outbound", 11, 20.0),
    MavlinkStreamSpec("POD_VLM_SEMANTIC_VETO", MSGID_POD_VLM_SEMANTIC_VETO, "outbound", 16, 10.0),
    MavlinkStreamSpec("POD_PLUGIN_HEALTH", MSGID_POD_PLUGIN_HEALTH, "outbound", 48, 5.0),
    MavlinkStreamSpec("COMMAND_INT", 75, "outbound", 35, 5.0),
    MavlinkStreamSpec("ADSB_VEHICLE", 246, "outbound", 38, 20.0),
    MavlinkStreamSpec("HEARTBEAT_POD", 0, "outbound", 9, 1.0),
    # Inbound: Flight Controller -> Companion Pod
    MavlinkStreamSpec("GLOBAL_POSITION_INT", 33, "inbound", 28, 10.0),
    MavlinkStreamSpec("EKF_STATUS_REPORT", 193, "inbound", 22, 5.0),
    MavlinkStreamSpec("HEARTBEAT_FC", 0, "inbound", 9, 1.0),
)


def compute_uart_bandwidth_budget(
    streams: Sequence[MavlinkStreamSpec] = DEFAULT_POD_UART_STREAMS,
    *,
    ardupilot_version: tuple[int, int, int] = MIN_ARDUPILOT_ODOMETRY_VERSION,
    baud_bps: int = UART_BAUD_BPS,
    max_bidir_bytes_per_sec: float | None = None,
    max_utilization_frac: float = MAX_UART_UTILIZATION_FRAC,
    plugin_health_hz: float = 0.0,
    signed_mavlink2: bool = False,
) -> dict[str, object]:
    """Compute worst-case bidirectional MAVLink2 UART bandwidth and enforce DR-11/DR-16 locks.

    Backward-compatible defaults (`plugin_health_hz=0.0`, `signed_mavlink2=False`) preserve
    the DR-11 unsigned `<= 16,000 B/s` budget. Enabling `signed_mavlink2=True` adds the
    13-byte MAVLink 2.0 SHA-256 signature overhead per frame and defaults the ceiling to
    `<= 18,432 B/s` (`< 20%` of `921.6 kbps` 8N1 per DR-16).
    """
    if float(plugin_health_hz) < 0.0:
        raise ValueError(f"plugin_health_hz must be >= 0.0, got {plugin_health_hz}")

    effective_streams = list(streams)
    has_plugin_health = any(s.msgid == MSGID_POD_PLUGIN_HEALTH for s in effective_streams)
    if float(plugin_health_hz) > 0.0 and not has_plugin_health:
        effective_streams.append(
            MavlinkStreamSpec(
                "POD_PLUGIN_HEALTH",
                MSGID_POD_PLUGIN_HEALTH,
                "outbound",
                WIRE_PAYLOAD_LEN_BY_MSGID[MSGID_POD_PLUGIN_HEALTH],
                float(plugin_health_hz),
            )
        )

    sig_bytes = MAVLINK_V2_SIGNATURE_BYTES if signed_mavlink2 else 0
    outbound_bps = 0.0
    inbound_bps = 0.0
    effective_plugin_hz = 0.0
    per_stream: list[dict[str, object]] = []

    for spec in effective_streams:
        frame_bytes = spec.frame_bytes + sig_bytes
        stream_bps = float(frame_bytes) * float(spec.rate_hz)
        if spec.msgid == MSGID_POD_PLUGIN_HEALTH:
            effective_plugin_hz += float(spec.rate_hz)
        if spec.direction == "outbound":
            validate_outbound_msgid(spec.msgid, ardupilot_version=ardupilot_version)
            outbound_bps += stream_bps
        elif spec.direction == "inbound":
            inbound_bps += stream_bps
        else:
            raise ValueError(f"Invalid stream direction: {spec.direction!r}")
        per_stream.append(
            {
                "name": spec.name,
                "msgid": spec.msgid,
                "direction": spec.direction,
                "frame_bytes": frame_bytes,
                "rate_hz": spec.rate_hz,
                "bytes_per_sec": round(stream_bps, 2),
            }
        )

    total_bps = outbound_bps + inbound_bps
    capacity_bps = float(baud_bps) / float(UART_BITS_PER_BYTE_8N1)
    effective_max_bps = (
        float(max_bidir_bytes_per_sec)
        if max_bidir_bytes_per_sec is not None
        else (capacity_bps * float(max_utilization_frac) if signed_mavlink2 else MAX_BIDIR_UART_BYTES_PER_SEC)
    )
    util_frac = total_bps / capacity_bps
    within_budget = (total_bps <= effective_max_bps) and (util_frac < float(max_utilization_frac))
    return {
        "baud_bps": int(baud_bps),
        "capacity_bytes_per_sec": capacity_bps,
        "signed_mavlink2": bool(signed_mavlink2),
        "signature_bytes_per_frame": int(sig_bytes),
        "plugin_health_hz": float(effective_plugin_hz),
        "outbound_bytes_per_sec": round(outbound_bps, 2),
        "inbound_bytes_per_sec": round(inbound_bps, 2),
        "total_bidir_bytes_per_sec": round(total_bps, 2),
        "headroom_bytes_per_sec": round(effective_max_bps - total_bps, 2),
        "utilization_frac": round(util_frac, 6),
        "max_bidir_bytes_per_sec": effective_max_bps,
        "max_utilization_frac": float(max_utilization_frac),
        "within_budget": bool(within_budget),
        "streams": per_stream,
    }


def assert_uart_bandwidth_budget(
    streams: Sequence[MavlinkStreamSpec] = DEFAULT_POD_UART_STREAMS,
    *,
    ardupilot_version: tuple[int, int, int] = MIN_ARDUPILOT_ODOMETRY_VERSION,
    plugin_health_hz: float = 0.0,
    signed_mavlink2: bool = False,
) -> dict[str, object]:
    """Assert bidirectional MAVLink2 UART load stays strictly within `< 20%` of `921.6 kbps`."""
    report = compute_uart_bandwidth_budget(
        streams,
        ardupilot_version=ardupilot_version,
        plugin_health_hz=plugin_health_hz,
        signed_mavlink2=signed_mavlink2,
    )
    if not report["within_budget"]:
        raise ValueError(
            f"Bidirectional UART budget exceeded: {report['total_bidir_bytes_per_sec']} B/s "
            f"({float(report['utilization_frac']) * 100.0:.2f}% of {report['capacity_bytes_per_sec']} B/s)"
        )
    return report


def _parse_xml_field_type(type_attr: str) -> tuple[str, int]:
    """Parse XML field type like `uint8_t[4]` -> `('uint8_t', 4)` or `int16_t` -> `('int16_t', 0)`."""
    if "[" in type_attr and type_attr.endswith("]"):
        base, rest = type_attr.split("[", 1)
        return base.strip(), int(rest[:-1].strip())
    return type_attr.strip(), 0


def parse_and_validate_dialect_xml(xml_path: Path) -> dict[str, object]:
    """Parse `configs/mavlink/pod_daa.xml`, validate conflict-free IDs, and cross-check CRC_EXTRA."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "mavlink":
        raise ValueError(f"Expected <mavlink> root tag, got <{root.tag}>")

    includes = [elem.text.strip() for elem in root.findall("include") if elem.text is not None]
    if "ardupilotmega.xml" not in includes:
        raise ValueError("pod_daa.xml must <include>ardupilotmega.xml</include>")

    messages_el = root.find("messages")
    if messages_el is None:
        raise ValueError("Missing <messages> element in dialect XML")

    parsed_msgs: dict[int, dict[str, object]] = {}
    for msg_el in messages_el.findall("message"):
        mid = int(msg_el.attrib["id"])
        name = str(msg_el.attrib["name"])
        validate_dialect_msgid(mid)
        if mid in parsed_msgs:
            raise ValueError(f"Duplicate message ID {mid} in {xml_path}")

        field_specs: list[tuple[str, str, int]] = []
        for f_el in msg_el.findall("field"):
            bt, al = _parse_xml_field_type(str(f_el.attrib["type"]))
            if bt not in _BASE_TYPE_INFO:
                raise ValueError(f"Unsupported field base type {bt!r} in message {name}")
            field_specs.append((bt, str(f_el.attrib["name"]), al))

        xml_crc_extra = compute_crc_extra(name, field_specs)
        if mid in CRC_EXTRA_BY_MSGID and xml_crc_extra != CRC_EXTRA_BY_MSGID[mid]:
            raise ValueError(
                f"CRC_EXTRA mismatch between XML and codec for {name} (#{mid}): "
                f"xml={xml_crc_extra}, codec={CRC_EXTRA_BY_MSGID[mid]}"
            )

        parsed_msgs[mid] = {
            "id": mid,
            "name": name,
            "field_count": len(field_specs),
            "crc_extra": xml_crc_extra,
        }

    return {
        "includes": includes,
        "message_ids": sorted(parsed_msgs.keys()),
        "messages": parsed_msgs,
    }
