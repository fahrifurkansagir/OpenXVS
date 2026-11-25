"""FAA GDL-90 Public ICD Rev A Binary Encoder, Decoder, and UDP Transport.

Implements exact FAA GDL-90 Public ICD (Revision A, 2007) framing and messages:
- Table-driven CRC-16-CCITT (poly = 0x1021, init = 0x0000, LSB-first wire order)
  verified against FAA GDL-90 ICD §2.2.3 Heartbeat test vector;
- Control-octet byte-stuffing (0x7E -> 0x7D 0x5E, 0x7D -> 0x7D 0x5D) and unstuffing;
- Message ID 0x00: Heartbeat (7 bytes payload);
- Message ID 0x0A: Ownship Report (28 bytes payload);
- Message ID 0x0B: Ownship Geometric Altitude (5 bytes payload, 5-ft resolution);
- Message ID 0x14: Traffic Report (28 bytes payload: Traffic Alert Status s=1/0,
  24-bit signed semicircle lat/lon, 12-bit 25-ft offset altitude, horizontal/vertical
  velocity, 8-bit track/heading, NIC/NACp passive range uncertainty, emitter category,
  and 8-char ASCII callsign XVS{:04d}).

Strictly isolated from any ground-truth intruder state (V5_B10).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
from typing import Any, Sequence

GDL90_FLAG_BYTE: int = 0x7E
GDL90_ESCAPE_BYTE: int = 0x7D
GDL90_STUFF_XOR: int = 0x20

GDL90_CRC_POLY: int = 0x1021
GDL90_CRC_INIT: int = 0x0000

GDL90_MSG_ID_HEARTBEAT: int = 0x00
GDL90_MSG_ID_OWNSHIP_REPORT: int = 0x0A
GDL90_MSG_ID_OWNSHIP_GEO_ALT: int = 0x0B
GDL90_MSG_ID_TRAFFIC_REPORT: int = 0x14

# Official FAA GDL-90 Public ICD §2.2.3 Heartbeat test vector
GDL90_ICD_HEARTBEAT_UNSTUFFED: bytes = bytes(
    [0x00, 0x81, 0x41, 0xDB, 0xD0, 0x08, 0x02]
)
GDL90_ICD_HEARTBEAT_CRC_U16: int = 0x8BB3
GDL90_ICD_HEARTBEAT_CRC_WIRE: bytes = bytes([0xB3, 0x8B])
GDL90_ICD_HEARTBEAT_FRAMED: bytes = bytes(
    [0x7E, 0x00, 0x81, 0x41, 0xDB, 0xD0, 0x08, 0x02, 0xB3, 0x8B, 0x7E]
)

DEFAULT_NIC_NACP_TIERS: tuple[tuple[float, int, int], ...] = (
    (75.0, 9, 9),
    (185.2, 8, 8),
    (370.4, 7, 7),
    (1111.2, 6, 6),
    (1852.0, 5, 5),
    (3704.0, 4, 4),
    (7408.0, 3, 3),
    (14816.0, 2, 2),
    (37040.0, 1, 1),
)


def _build_crc16_table() -> tuple[int, ...]:
    """Build 256-word CRC-16-CCITT lookup table per FAA GDL-90 Public ICD §2.2.2."""
    table: list[int] = []
    for i in range(256):
        crc = (i << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ GDL90_CRC_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC16_TABLE: tuple[int, ...] = _build_crc16_table()


def compute_gdl90_crc(data: bytes | bytearray | memoryview) -> int:
    """Compute 16-bit GDL-90 Frame Check Sequence per FAA GDL-90 ICD §2.2.2.

    The 16-bit CRC value is returned as an unsigned integer where the wire
    representation appends LSB first (`crc & 0xFF`, `(crc >> 8) & 0xFF`).
    """
    crc = GDL90_CRC_INIT
    for b in data:
        crc = _CRC16_TABLE[(crc >> 8) & 0xFF] ^ ((crc << 8) & 0xFFFF) ^ int(b)
    return crc & 0xFFFF


def stuff_gdl90_bytes(raw: bytes | bytearray) -> bytes:
    """Apply GDL-90 control-octet byte stuffing (0x7E -> 0x7D 0x5E, 0x7D -> 0x7D 0x5D)."""
    out = bytearray()
    for b in raw:
        if b in (GDL90_FLAG_BYTE, GDL90_ESCAPE_BYTE):
            out.append(GDL90_ESCAPE_BYTE)
            out.append(b ^ GDL90_STUFF_XOR)
        else:
            out.append(b)
    return bytes(out)


def unstuff_gdl90_bytes(stuffed: bytes | bytearray) -> bytes:
    """Reverse GDL-90 control-octet byte stuffing."""
    out = bytearray()
    i = 0
    n = len(stuffed)
    while i < n:
        b = stuffed[i]
        if b == GDL90_ESCAPE_BYTE:
            if i + 1 >= n:
                raise ValueError("Truncated GDL-90 escape sequence at end of frame")
            out.append(stuffed[i + 1] ^ GDL90_STUFF_XOR)
            i += 2
        elif b == GDL90_FLAG_BYTE:
            raise ValueError("Unexpected unescaped 0x7E flag byte inside GDL-90 frame body")
        else:
            out.append(b)
            i += 1
    return bytes(out)


def frame_gdl90_message(payload: bytes | bytearray) -> bytes:
    """Append LSB-first CRC-16, byte-stuff, and wrap with 0x7E flag delimiters."""
    raw_payload = bytes(payload)
    crc = compute_gdl90_crc(raw_payload)
    fcs = bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    stuffed = stuff_gdl90_bytes(raw_payload + fcs)
    return bytes([GDL90_FLAG_BYTE]) + stuffed + bytes([GDL90_FLAG_BYTE])


def decode_gdl90_frame(frame: bytes | bytearray) -> bytes:
    """Strip 0x7E delimiters, unstuff bytes, verify CRC-16, and return message payload."""
    if len(frame) < 5:
        raise ValueError(f"GDL-90 frame too short ({len(frame)} bytes)")
    if frame[0] != GDL90_FLAG_BYTE or frame[-1] != GDL90_FLAG_BYTE:
        raise ValueError("GDL-90 frame missing leading or trailing 0x7E flag byte")

    unstuffed = unstuff_gdl90_bytes(frame[1:-1])
    if len(unstuffed) < 3:
        raise ValueError("GDL-90 unstuffed content shorter than 1-byte ID + 2-byte CRC")

    payload = unstuffed[:-2]
    rx_crc = int(unstuffed[-2]) | (int(unstuffed[-1]) << 8)
    calc_crc = compute_gdl90_crc(payload)
    if rx_crc != calc_crc:
        raise ValueError(
            f"GDL-90 CRC mismatch: received 0x{rx_crc:04X}, expected 0x{calc_crc:04X}"
        )
    return payload


def encode_lat_lon_semicircles(angle_deg: float) -> bytes:
    """Encode latitude or longitude in degrees to 24-bit signed semicircles (big-endian)."""
    clamped = max(-180.0, min(180.0, float(angle_deg)))
    scale = float(0x800000) / 180.0
    raw = int(round(clamped * scale))
    if raw > 0x7FFFFF:
        raw = 0x7FFFFF
    elif raw < -0x800000:
        raw = -0x800000
    u24 = raw & 0xFFFFFF
    return bytes([(u24 >> 16) & 0xFF, (u24 >> 8) & 0xFF, u24 & 0xFF])


def decode_lat_lon_semicircles(b3: bytes | bytearray) -> float:
    """Decode 3-byte big-endian 24-bit signed semicircles to degrees."""
    if len(b3) != 3:
        raise ValueError(f"Expected 3 semicircle bytes, got {len(b3)}")
    u24 = (int(b3[0]) << 16) | (int(b3[1]) << 8) | int(b3[2])
    if u24 & 0x800000:
        s24 = u24 - 0x1000000
    else:
        s24 = u24
    return float(s24) * (180.0 / float(0x800000))


def encode_altitude_25ft(alt_ft: float | None) -> int:
    """Encode pressure/geometric altitude in feet to 12-bit 25-ft offset code.

    Per FAA GDL-90 ICD §3.5.1.5:
      code = round((alt_ft + 1000) / 25), valid range [-1000, 101350] ft -> [0x000, 0xFFE].
      0xFFF indicates invalid or unavailable altitude.
    """
    if alt_ft is None or not math.isfinite(alt_ft):
        return 0xFFF
    code = int(round((float(alt_ft) + 1000.0) / 25.0))
    return max(0x000, min(0xFFE, code))


def decode_altitude_25ft(code: int) -> float | None:
    """Decode 12-bit GDL-90 altitude code to feet (or None if 0xFFF)."""
    c = int(code) & 0xFFF
    if c == 0xFFF:
        return None
    return float(c) * 25.0 - 1000.0


def map_range_span_to_nic_nacp(
    range_span_m: float,
    tiers: Sequence[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Map passive range interval uncertainty (R_max - R_min) in metres to (NIC, NACp)."""
    span = abs(float(range_span_m))
    if not math.isfinite(span):
        return (0, 0)
    if tiers is not None:
        for t in tiers:
            if span <= float(t["max_range_span_m"]):
                return (int(t["nic"]) & 0x0F, int(t["nacp"]) & 0x0F)
        return (0, 0)
    for max_span, nic, nacp in DEFAULT_NIC_NACP_TIERS:
        if span <= max_span:
            return (nic, nacp)
    return (0, 0)


def encode_heartbeat_payload(
    utc_time_s: int = 0,
    gps_pos_valid: bool = True,
    maint_reqd: bool = False,
    ident: bool = False,
    addr_talkback: bool = False,
    gps_batt_low: bool = False,
    ratcs: bool = False,
    uat_initialized: bool = True,
    csa_requested: bool = False,
    csa_not_available: bool = False,
    utc_ok: bool = True,
    uplink_count: int = 0,
    basic_and_long_count: int = 0,
) -> bytes:
    """Encode 7-byte unstuffed GDL-90 Heartbeat payload (Message ID 0x00, ICD §3.1)."""
    ts = int(utc_time_s) % 86400
    status1 = (
        ((1 if gps_pos_valid else 0) << 7)
        | ((1 if maint_reqd else 0) << 6)
        | ((1 if ident else 0) << 5)
        | ((1 if addr_talkback else 0) << 4)
        | ((1 if gps_batt_low else 0) << 3)
        | ((1 if ratcs else 0) << 2)
        | (1 if uat_initialized else 0)
    )
    ts_bit16 = (ts >> 16) & 0x01
    status2 = (
        (ts_bit16 << 7)
        | ((1 if csa_requested else 0) << 6)
        | ((1 if csa_not_available else 0) << 5)
        | (1 if utc_ok else 0)
    )
    ts_lsb = ts & 0xFF
    ts_msb = (ts >> 8) & 0xFF
    ul_cnt = max(0, min(31, int(uplink_count)))
    bl_cnt = max(0, min(1023, int(basic_and_long_count)))
    msg_cnt_hi = ((ul_cnt & 0x1F) << 3) | ((bl_cnt >> 8) & 0x03)
    msg_cnt_lo = bl_cnt & 0xFF
    return bytes(
        [
            GDL90_MSG_ID_HEARTBEAT,
            status1 & 0xFF,
            status2 & 0xFF,
            ts_lsb,
            ts_msb,
            msg_cnt_hi,
            msg_cnt_lo,
        ]
    )


def encode_heartbeat(
    utc_time_s: int = 0,
    gps_pos_valid: bool = True,
    maint_reqd: bool = False,
    ident: bool = False,
    utc_ok: bool = True,
    uplink_count: int = 0,
    basic_and_long_count: int = 0,
    uat_initialized: bool = True,
) -> bytes:
    """Encode and frame a complete GDL-90 Heartbeat message (0x7E ... 0x7E)."""
    payload = encode_heartbeat_payload(
        utc_time_s=utc_time_s,
        gps_pos_valid=gps_pos_valid,
        maint_reqd=maint_reqd,
        ident=ident,
        utc_ok=utc_ok,
        uplink_count=uplink_count,
        basic_and_long_count=basic_and_long_count,
        uat_initialized=uat_initialized,
    )
    return frame_gdl90_message(payload)


def decode_heartbeat_payload(payload: bytes | bytearray) -> dict[str, Any]:
    """Decode a 7-byte unstuffed GDL-90 Heartbeat payload."""
    if len(payload) != 7 or payload[0] != GDL90_MSG_ID_HEARTBEAT:
        raise ValueError("Invalid GDL-90 Heartbeat payload")
    s1 = int(payload[1])
    s2 = int(payload[2])
    ts = int(payload[3]) | (int(payload[4]) << 8) | (((s2 >> 7) & 0x01) << 16)
    uplink_count = (int(payload[5]) >> 3) & 0x1F
    basic_and_long_count = ((int(payload[5]) & 0x03) << 8) | int(payload[6])
    return {
        "msg_id": GDL90_MSG_ID_HEARTBEAT,
        "gps_pos_valid": bool(s1 & 0x80),
        "maint_reqd": bool(s1 & 0x40),
        "ident": bool(s1 & 0x20),
        "uat_initialized": bool(s1 & 0x01),
        "utc_ok": bool(s2 & 0x01),
        "utc_time_s": ts,
        "uplink_count": uplink_count,
        "basic_and_long_count": basic_and_long_count,
    }


def encode_ownship_geo_altitude_payload(
    alt_ft: float,
    vfom_m: float = 3.0,
    vertical_warning: bool = False,
) -> bytes:
    """Encode 5-byte Ownship Geometric Altitude payload (Message ID 0x0B, ICD §3.8)."""
    alt_5ft = int(round(float(alt_ft) / 5.0))
    alt_5ft = max(-32768, min(32767, alt_5ft))
    u16_alt = alt_5ft & 0xFFFF

    if not math.isfinite(vfom_m) or vfom_m < 0.0:
        vfom_int = 0x7FFF
    elif vfom_m > 32766.0:
        vfom_int = 0x7FFE
    else:
        vfom_int = int(round(vfom_m)) & 0x7FFF

    metrics_u16 = ((1 if vertical_warning else 0) << 15) | vfom_int
    return bytes(
        [
            GDL90_MSG_ID_OWNSHIP_GEO_ALT,
            (u16_alt >> 8) & 0xFF,
            u16_alt & 0xFF,
            (metrics_u16 >> 8) & 0xFF,
            metrics_u16 & 0xFF,
        ]
    )


def encode_ownship_geo_altitude(
    alt_ft: float,
    vfom_m: float = 3.0,
    vertical_warning: bool = False,
) -> bytes:
    """Encode and frame Message ID 0x0B (Ownship Geometric Altitude)."""
    return frame_gdl90_message(
        encode_ownship_geo_altitude_payload(
            alt_ft=alt_ft, vfom_m=vfom_m, vertical_warning=vertical_warning
        )
    )


def decode_ownship_geo_altitude_payload(payload: bytes | bytearray) -> dict[str, Any]:
    """Decode 5-byte Ownship Geometric Altitude payload (Message ID 0x0B)."""
    if len(payload) != 5 or payload[0] != GDL90_MSG_ID_OWNSHIP_GEO_ALT:
        raise ValueError("Invalid GDL-90 Ownship Geometric Altitude payload")
    u16_alt = (int(payload[1]) << 8) | int(payload[2])
    s16_alt = u16_alt - 0x10000 if (u16_alt & 0x8000) else u16_alt
    metrics_u16 = (int(payload[3]) << 8) | int(payload[4])
    vertical_warning = bool(metrics_u16 & 0x8000)
    vfom_m = metrics_u16 & 0x7FFF
    return {
        "msg_id": GDL90_MSG_ID_OWNSHIP_GEO_ALT,
        "alt_ft": float(s16_alt * 5),
        "vertical_warning": vertical_warning,
        "vfom_m": vfom_m,
    }


@dataclass(frozen=True)
class Gdl90TargetReport:
    """28-byte GDL-90 Target Report (Message ID 0x0A Ownship or 0x14 Traffic, ICD §3.4/§3.5)."""

    msg_id: int
    alert_status: int
    address_type: int
    participant_address: int
    lat_deg: float
    lon_deg: float
    altitude_ft: float | None
    misc_indicators: int
    nic: int
    nacp: int
    horiz_vel_kt: float | None
    vert_vel_fpm: float | None
    track_heading_deg: float
    emitter_category: int
    callsign: str
    emergency_code: int = 0


def encode_target_report_payload(report: Gdl90TargetReport) -> bytes:
    """Encode 28-byte unstuffed GDL-90 Ownship (0x0A) or Traffic (0x14) Report payload."""
    if report.msg_id not in (GDL90_MSG_ID_OWNSHIP_REPORT, GDL90_MSG_ID_TRAFFIC_REPORT):
        raise ValueError(f"Unsupported GDL-90 target report msg_id: 0x{report.msg_id:02X}")

    b_status_type = ((int(report.alert_status) & 0x0F) << 4) | (
        int(report.address_type) & 0x0F
    )
    addr = int(report.participant_address) & 0xFFFFFF
    b_addr = bytes([(addr >> 16) & 0xFF, (addr >> 8) & 0xFF, addr & 0xFF])

    b_lat = encode_lat_lon_semicircles(report.lat_deg)
    b_lon = encode_lat_lon_semicircles(report.lon_deg)

    alt_code = encode_altitude_25ft(report.altitude_ft)
    misc = int(report.misc_indicators) & 0x0F
    b11 = (alt_code >> 4) & 0xFF
    b12 = ((alt_code & 0x0F) << 4) | misc

    b13 = ((int(report.nic) & 0x0F) << 4) | (int(report.nacp) & 0x0F)

    if report.horiz_vel_kt is None or not math.isfinite(report.horiz_vel_kt) or report.horiz_vel_kt < 0.0:
        h_code = 0xFFF
    else:
        h_code = max(0, min(0xFFE, int(round(report.horiz_vel_kt))))

    if report.vert_vel_fpm is None or not math.isfinite(report.vert_vel_fpm):
        v_code = 0x800
    else:
        v_steps = int(round(float(report.vert_vel_fpm) / 64.0))
        v_steps = max(-510, min(510, v_steps))
        v_code = v_steps & 0xFFF

    b14 = (h_code >> 4) & 0xFF
    b15 = ((h_code & 0x0F) << 4) | ((v_code >> 8) & 0x0F)
    b16 = v_code & 0xFF

    trk_deg = float(report.track_heading_deg) % 360.0
    b17_trk = int(round(trk_deg * (256.0 / 360.0))) & 0xFF
    b18_emit = int(report.emitter_category) & 0xFF

    clean_cs = "".join(
        ch for ch in report.callsign.upper() if ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch == " "
    )
    cs_padded = clean_cs[:8].ljust(8, " ").encode("ascii")
    b27_px = (int(report.emergency_code) & 0x0F) << 4

    out = bytearray()
    out.append(report.msg_id & 0xFF)
    out.append(b_status_type)
    out.extend(b_addr)
    out.extend(b_lat)
    out.extend(b_lon)
    out.append(b11)
    out.append(b12)
    out.append(b13)
    out.append(b14)
    out.append(b15)
    out.append(b16)
    out.append(b17_trk)
    out.append(b18_emit)
    out.extend(cs_padded)
    out.append(b27_px)
    assert len(out) == 28
    return bytes(out)


def encode_target_report(report: Gdl90TargetReport) -> bytes:
    """Encode and frame a 28-byte GDL-90 Ownship (0x0A) or Traffic (0x14) Report."""
    return frame_gdl90_message(encode_target_report_payload(report))


def decode_target_report_payload(payload: bytes | bytearray) -> Gdl90TargetReport:
    """Decode a 28-byte unstuffed GDL-90 Ownship (0x0A) or Traffic (0x14) Report payload."""
    if len(payload) != 28 or payload[0] not in (
        GDL90_MSG_ID_OWNSHIP_REPORT,
        GDL90_MSG_ID_TRAFFIC_REPORT,
    ):
        raise ValueError(f"Invalid GDL-90 Target Report payload length={len(payload)}")

    msg_id = int(payload[0])
    alert_status = (int(payload[1]) >> 4) & 0x0F
    address_type = int(payload[1]) & 0x0F
    participant_address = (
        (int(payload[2]) << 16) | (int(payload[3]) << 8) | int(payload[4])
    )
    lat_deg = decode_lat_lon_semicircles(payload[5:8])
    lon_deg = decode_lat_lon_semicircles(payload[8:11])

    alt_code = (int(payload[11]) << 4) | ((int(payload[12]) >> 4) & 0x0F)
    altitude_ft = decode_altitude_25ft(alt_code)
    misc_indicators = int(payload[12]) & 0x0F

    nic = (int(payload[13]) >> 4) & 0x0F
    nacp = int(payload[13]) & 0x0F

    h_code = (int(payload[14]) << 4) | ((int(payload[15]) >> 4) & 0x0F)
    horiz_vel_kt = None if h_code == 0xFFF else float(h_code)

    v_code = ((int(payload[15]) & 0x0F) << 8) | int(payload[16])
    if v_code == 0x800:
        vert_vel_fpm = None
    else:
        v_signed = v_code - 0x1000 if (v_code & 0x800) else v_code
        vert_vel_fpm = float(v_signed * 64)

    track_heading_deg = float(payload[17]) * (360.0 / 256.0)
    emitter_category = int(payload[18])
    callsign = bytes(payload[19:27]).decode("ascii", errors="replace").rstrip(" ")
    emergency_code = (int(payload[27]) >> 4) & 0x0F

    return Gdl90TargetReport(
        msg_id=msg_id,
        alert_status=alert_status,
        address_type=address_type,
        participant_address=participant_address,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        altitude_ft=altitude_ft,
        misc_indicators=misc_indicators,
        nic=nic,
        nacp=nacp,
        horiz_vel_kt=horiz_vel_kt,
        vert_vel_fpm=vert_vel_fpm,
        track_heading_deg=track_heading_deg,
        emitter_category=emitter_category,
        callsign=callsign,
        emergency_code=emergency_code,
    )


def decode_gdl90_message(frame: bytes | bytearray) -> dict[str, Any] | Gdl90TargetReport:
    """Decode any framed GDL-90 message (0x00, 0x0A, 0x0B, 0x14)."""
    payload = decode_gdl90_frame(frame)
    msg_id = payload[0]
    if msg_id == GDL90_MSG_ID_HEARTBEAT:
        return decode_heartbeat_payload(payload)
    if msg_id == GDL90_MSG_ID_OWNSHIP_GEO_ALT:
        return decode_ownship_geo_altitude_payload(payload)
    if msg_id in (GDL90_MSG_ID_OWNSHIP_REPORT, GDL90_MSG_ID_TRAFFIC_REPORT):
        return decode_target_report_payload(payload)
    raise ValueError(f"Unsupported GDL-90 message ID: 0x{msg_id:02X}")


class Gdl90UdpEndpoint:
    """Non-blocking localhost UDP sender/receiver for GDL-90 datagrams."""

    def __init__(
        self,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        target_host: str = "127.0.0.1",
        target_port: int = 4000,
    ) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self._sock.setblocking(False)
        self._sock.bind((bind_host, bind_port))
        self.local_addr: tuple[str, int] = self._sock.getsockname()
        self.target_addr: tuple[str, int] = (target_host, target_port)

    @property
    def port(self) -> int:
        return int(self.local_addr[1])

    def set_target(self, host: str, port: int) -> None:
        self.target_addr = (host, int(port))

    def send_frame(self, frame: bytes) -> int:
        return self._sock.sendto(frame, self.target_addr)

    def recv_frames(self, max_packets: int = 256) -> list[bytes]:
        frames: list[bytes] = []
        for _ in range(max_packets):
            try:
                data, _ = self._sock.recvfrom(4096)
                frames.append(data)
            except (BlockingIOError, socket.error):
                break
        return frames

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> Gdl90UdpEndpoint:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
