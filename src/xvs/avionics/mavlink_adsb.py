"""MAVLink v2 ADSB_VEHICLE (#246) Packer, Parser, and ArduPilot SITL UDP Streamer.

Implements self-contained MAVLink v2 binary framing:
- STX = 0xFD, payload length = 38 bytes, msgid = 246 (0xF6 0x00 0x00);
- X.25 CRC-16/MCRF4XX (init = 0xFFFF) with CRC_EXTRA = 184;
- Wire-ordered little-endian struct packing (`<IiiiHHhHHB9sBB`) for `ADSB_VEHICLE`
  (common.xml #246);
- Passive WP-05 track state -> `ADSB_VEHICLE` coordinate and velocity projection
  using only ownship INS pose and passive range bounds [R_min, R_max] along estimated
  unit LOS vector u_hat_NED (`xvs.geo.wgs84`), preserving strict truth isolation (V5_B10).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
from typing import Any
import numpy as np

from xvs.geo import wgs84

MAVLINK_V2_STX: int = 0xFD
MAVLINK_V2_HEADER_LEN: int = 10
ADSB_VEHICLE_MSG_ID: int = 246
ADSB_VEHICLE_CRC_EXTRA: int = 184
ADSB_VEHICLE_PAYLOAD_LEN: int = 38
ADSB_VEHICLE_CRC_LEN: int = 2
ADSB_VEHICLE_FRAME_LEN: int = (
    MAVLINK_V2_HEADER_LEN + ADSB_VEHICLE_PAYLOAD_LEN + ADSB_VEHICLE_CRC_LEN
)

# MAVLink common.xml ADSB_FLAGS bitmask definitions
ADSB_FLAGS_VALID_COORDS: int = 1
ADSB_FLAGS_VALID_ALTITUDE: int = 2
ADSB_FLAGS_VALID_HEADING: int = 4
ADSB_FLAGS_VALID_VELOCITY: int = 8
ADSB_FLAGS_VALID_CALLSIGN: int = 16
ADSB_FLAGS_VALID_SQUAWK: int = 32
ADSB_FLAGS_SIMULATED: int = 64
ADSB_FLAGS_VERTICAL_VELOCITY_VALID: int = 128
ADSB_FLAGS_BARO_VALID: int = 256
ADSB_FLAGS_SOURCE_UAT: int = 32768

DEFAULT_PASSIVE_ADSB_FLAGS: int = (
    ADSB_FLAGS_VALID_COORDS
    | ADSB_FLAGS_VALID_ALTITUDE
    | ADSB_FLAGS_VALID_HEADING
    | ADSB_FLAGS_VALID_VELOCITY
    | ADSB_FLAGS_VALID_CALLSIGN
    | ADSB_FLAGS_VALID_SQUAWK
    | ADSB_FLAGS_SIMULATED
)

_ADSB_PAYLOAD_STRUCT = struct.Struct("<IiiiHHhHHB9sBB")


def _x25_accumulate(byte_val: int, crc: int) -> int:
    """Accumulate one byte into MAVLink X.25 CRC-16/MCRF4XX."""
    tmp = (int(byte_val) ^ (crc & 0xFF)) & 0xFF
    tmp = (tmp ^ ((tmp << 4) & 0xFF)) & 0xFF
    return (((crc >> 8) & 0xFF) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF


def compute_mavlink_x25_crc(
    header_without_stx: bytes | bytearray,
    payload: bytes | bytearray,
    crc_extra: int = ADSB_VEHICLE_CRC_EXTRA,
) -> int:
    """Compute MAVLink X.25 CRC-16/MCRF4XX over header[1:10] + payload + crc_extra."""
    crc = 0xFFFF
    for b in header_without_stx:
        crc = _x25_accumulate(b, crc)
    for b in payload:
        crc = _x25_accumulate(b, crc)
    crc = _x25_accumulate(crc_extra & 0xFF, crc)
    return crc & 0xFFFF


@dataclass(frozen=True)
class MavlinkAdsbVehicle:
    """MAVLink v2 `ADSB_VEHICLE` (#246) message representation."""

    icao_address: int
    lat_deg_e7: int
    lon_deg_e7: int
    altitude_mm: int
    heading_cdeg: int
    hor_velocity_cms: int
    ver_velocity_cms: int
    flags: int
    squawk: int
    altitude_type: int
    callsign: str
    emitter_type: int
    tslc: int
    seq: int = 0
    sysid: int = 1
    compid: int = 156
    incompat_flags: int = 0
    compat_flags: int = 0

    @property
    def lat_deg(self) -> float:
        return float(self.lat_deg_e7) * 1e-7

    @property
    def lon_deg(self) -> float:
        return float(self.lon_deg_e7) * 1e-7

    @property
    def altitude_m(self) -> float:
        return float(self.altitude_mm) * 1e-3

    @property
    def heading_deg(self) -> float:
        return float(self.heading_cdeg) * 1e-2

    @property
    def hor_velocity_mps(self) -> float:
        return float(self.hor_velocity_cms) * 1e-2

    @property
    def ver_velocity_mps(self) -> float:
        return float(self.ver_velocity_cms) * 1e-2


def pack_adsb_vehicle(msg: MavlinkAdsbVehicle) -> bytes:
    """Pack a `MavlinkAdsbVehicle` into a 50-byte MAVLink v2 wire frame."""
    cs_clean = msg.callsign.encode("ascii", errors="ignore")[:8]
    cs_bytes = cs_clean.ljust(9, b"\x00")

    payload = _ADSB_PAYLOAD_STRUCT.pack(
        int(msg.icao_address) & 0xFFFFFFFF,
        max(-2147483648, min(2147483647, int(msg.lat_deg_e7))),
        max(-2147483648, min(2147483647, int(msg.lon_deg_e7))),
        max(-2147483648, min(2147483647, int(msg.altitude_mm))),
        max(0, min(65535, int(msg.heading_cdeg))),
        max(0, min(65535, int(msg.hor_velocity_cms))),
        max(-32768, min(32767, int(msg.ver_velocity_cms))),
        int(msg.flags) & 0xFFFF,
        int(msg.squawk) & 0xFFFF,
        int(msg.altitude_type) & 0xFF,
        cs_bytes,
        int(msg.emitter_type) & 0xFF,
        int(msg.tslc) & 0xFF,
    )

    header = bytes(
        [
            MAVLINK_V2_STX,
            ADSB_VEHICLE_PAYLOAD_LEN,
            int(msg.incompat_flags) & 0xFF,
            int(msg.compat_flags) & 0xFF,
            int(msg.seq) & 0xFF,
            int(msg.sysid) & 0xFF,
            int(msg.compid) & 0xFF,
            ADSB_VEHICLE_MSG_ID & 0xFF,
            (ADSB_VEHICLE_MSG_ID >> 8) & 0xFF,
            (ADSB_VEHICLE_MSG_ID >> 16) & 0xFF,
        ]
    )
    crc = compute_mavlink_x25_crc(header[1:], payload, ADSB_VEHICLE_CRC_EXTRA)
    crc_bytes = bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    return header + payload + crc_bytes


def unpack_adsb_vehicle(frame: bytes | bytearray) -> MavlinkAdsbVehicle:
    """Unpack and verify a MAVLink v2 `ADSB_VEHICLE` (#246) wire frame."""
    if len(frame) < MAVLINK_V2_HEADER_LEN + ADSB_VEHICLE_CRC_LEN:
        raise ValueError(f"MAVLink v2 frame too short ({len(frame)} bytes)")
    if frame[0] != MAVLINK_V2_STX:
        raise ValueError(f"Invalid MAVLink v2 STX: 0x{frame[0]:02X}")

    payload_len = int(frame[1])
    incompat_flags = int(frame[2])
    compat_flags = int(frame[3])
    seq = int(frame[4])
    sysid = int(frame[5])
    compid = int(frame[6])
    msgid = int(frame[7]) | (int(frame[8]) << 8) | (int(frame[9]) << 16)
    if msgid != ADSB_VEHICLE_MSG_ID:
        raise ValueError(f"Unexpected MAVLink msgid={msgid}, expected {ADSB_VEHICLE_MSG_ID}")

    expected_frame_len = MAVLINK_V2_HEADER_LEN + payload_len + ADSB_VEHICLE_CRC_LEN
    if len(frame) != expected_frame_len or payload_len > ADSB_VEHICLE_PAYLOAD_LEN:
        raise ValueError(
            f"MAVLink v2 length mismatch: len(frame)={len(frame)}, payload_len={payload_len}"
        )

    wire_payload = bytes(frame[MAVLINK_V2_HEADER_LEN : MAVLINK_V2_HEADER_LEN + payload_len])
    rx_crc = int(frame[-2]) | (int(frame[-1]) << 8)
    calc_crc = compute_mavlink_x25_crc(
        frame[1:MAVLINK_V2_HEADER_LEN], wire_payload, ADSB_VEHICLE_CRC_EXTRA
    )
    if rx_crc != calc_crc:
        raise ValueError(
            f"MAVLink X.25 CRC mismatch: received 0x{rx_crc:04X}, expected 0x{calc_crc:04X}"
        )

    padded_payload = wire_payload.ljust(ADSB_VEHICLE_PAYLOAD_LEN, b"\x00")
    (
        icao_address,
        lat_deg_e7,
        lon_deg_e7,
        altitude_mm,
        heading_cdeg,
        hor_velocity_cms,
        ver_velocity_cms,
        flags,
        squawk,
        altitude_type,
        callsign_raw,
        emitter_type,
        tslc,
    ) = _ADSB_PAYLOAD_STRUCT.unpack(padded_payload)

    callsign = callsign_raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").rstrip(" ")
    return MavlinkAdsbVehicle(
        icao_address=int(icao_address),
        lat_deg_e7=int(lat_deg_e7),
        lon_deg_e7=int(lon_deg_e7),
        altitude_mm=int(altitude_mm),
        heading_cdeg=int(heading_cdeg),
        hor_velocity_cms=int(hor_velocity_cms),
        ver_velocity_cms=int(ver_velocity_cms),
        flags=int(flags),
        squawk=int(squawk),
        altitude_type=int(altitude_type),
        callsign=callsign,
        emitter_type=int(emitter_type),
        tslc=int(tslc),
        seq=seq,
        sysid=sysid,
        compid=compid,
        incompat_flags=incompat_flags,
        compat_flags=compat_flags,
    )


def compute_passive_track_continuous_geodetic(
    *,
    ownship_lat_deg: float,
    ownship_lon_deg: float,
    ownship_alt_msl_m: float,
    az_rad: float,
    el_rad: float,
    r_min_m: float,
    r_max_m: float,
) -> tuple[float, float, float]:
    """Compute unquantized continuous WGS84 geodetic coordinates (lat_deg, lon_deg, alt_m)
    from ownship INS pose and passive range midpoint R_hat = 0.5 * (R_min + R_max).
    """
    r_lo = max(1.0, float(r_min_m))
    r_hi = max(r_lo, min(1e6, float(r_max_m)))
    r_hat_m = 0.5 * (r_lo + r_hi)

    cos_el = math.cos(float(el_rad))
    sin_el = math.sin(float(el_rad))
    cos_az = math.cos(float(az_rad))
    sin_az = math.sin(float(az_rad))

    u_ned = np.array([cos_el * cos_az, cos_el * sin_az, -sin_el], dtype=np.float64)
    p_rel_ned = r_hat_m * u_ned

    p_own_ecef = wgs84.geodetic_to_ecef(
        float(ownship_lat_deg), float(ownship_lon_deg), float(ownship_alt_msl_m)
    )
    r_mat = wgs84.r_ecef_ned(float(ownship_lat_deg), float(ownship_lon_deg))
    p_int_ecef = p_own_ecef + (r_mat @ p_rel_ned)
    return wgs84.ecef_to_geodetic(
        float(p_int_ecef[0]), float(p_int_ecef[1]), float(p_int_ecef[2])
    )


def passive_track_to_adsb_vehicle(
    *,
    track_number: int,
    ownship_lat_deg: float,
    ownship_lon_deg: float,
    ownship_alt_msl_m: float,
    ownship_vn_mps: float,
    ownship_ve_mps: float,
    ownship_vd_mps: float,
    az_rad: float,
    el_rad: float,
    az_rate_radps: float,
    el_rate_radps: float,
    r_min_m: float,
    r_max_m: float,
    tau_hat_s: float | None,
    alert_active: bool,
    emitter_type: int = 14,
    tslc_s: int = 0,
    seq: int = 0,
    sysid: int = 1,
    compid: int = 156,
    squawk_nominal: int = 1200,
    squawk_alert: int = 7700,
    valid_coords: bool = True,
) -> MavlinkAdsbVehicle:
    """Project a passive WP-05 bearing/expansion track into `MavlinkAdsbVehicle` (#246).

    Strictly uses ownship INS navigation state and passive range interval midpoint
    R_hat = 0.5 * (R_min + R_max) along the estimated unit LOS vector u_hat_NED.
    When `valid_coords` is False (e.g. WP-07 `INVALID_ALERT`), clears `ADSB_FLAGS_VALID_COORDS`.
    Never accesses any ground-truth intruder state (V5_B10).
    """
    r_lo = max(1.0, float(r_min_m))
    r_hi = max(r_lo, min(1e6, float(r_max_m)))
    r_hat_m = 0.5 * (r_lo + r_hi)

    int_lat_deg, int_lon_deg, int_alt_m = compute_passive_track_continuous_geodetic(
        ownship_lat_deg=ownship_lat_deg,
        ownship_lon_deg=ownship_lon_deg,
        ownship_alt_msl_m=ownship_alt_msl_m,
        az_rad=az_rad,
        el_rad=el_rad,
        r_min_m=r_lo,
        r_max_m=r_hi,
    )

    cos_el = math.cos(float(el_rad))
    sin_el = math.sin(float(el_rad))
    cos_az = math.cos(float(az_rad))
    sin_az = math.sin(float(az_rad))
    u_ned = np.array([cos_el * cos_az, cos_el * sin_az, -sin_el], dtype=np.float64)

    if tau_hat_s is not None and math.isfinite(tau_hat_s) and tau_hat_s > 0.25:
        r_dot_mps = -r_hat_m / max(1.0, float(tau_hat_s))
    else:
        r_dot_mps = 0.0

    du_daz = np.array([-cos_el * sin_az, cos_el * cos_az, 0.0], dtype=np.float64)
    du_del = np.array([-sin_el * cos_az, -sin_el * sin_az, -cos_el], dtype=np.float64)
    u_dot_ned = float(az_rate_radps) * du_daz + float(el_rate_radps) * du_del

    v_rel_ned = r_dot_mps * u_ned + r_hat_m * u_dot_ned
    v_own_ned = np.array(
        [float(ownship_vn_mps), float(ownship_ve_mps), float(ownship_vd_mps)],
        dtype=np.float64,
    )
    v_int_ned = v_own_ned + v_rel_ned

    vn_int = float(v_int_ned[0])
    ve_int = float(v_int_ned[1])
    vd_int = float(v_int_ned[2])

    horiz_speed_mps = math.hypot(vn_int, ve_int)
    if horiz_speed_mps > 0.1:
        heading_deg = math.degrees(math.atan2(ve_int, vn_int)) % 360.0
    else:
        heading_deg = (math.degrees(float(az_rad)) + 180.0) % 360.0

    icao_addr = 0xF00000 | (int(track_number) & 0x0FFFFF)
    callsign = f"XVS{int(track_number) % 10000:04d}"

    lat_e7 = int(round(int_lat_deg * 1e7))
    lon_e7 = int(round(int_lon_deg * 1e7))
    alt_mm = int(round(int_alt_m * 1000.0))
    heading_cdeg = int(round(heading_deg * 100.0)) % 36000
    hor_vel_cms = max(0, min(65535, int(round(horiz_speed_mps * 100.0))))
    ver_vel_cms = max(-32768, min(32767, int(round(-vd_int * 100.0))))

    squawk = int(squawk_alert if alert_active else squawk_nominal)
    flags = DEFAULT_PASSIVE_ADSB_FLAGS | ADSB_FLAGS_VERTICAL_VELOCITY_VALID
    if not valid_coords:
        flags &= ~ADSB_FLAGS_VALID_COORDS

    return MavlinkAdsbVehicle(
        icao_address=icao_addr,
        lat_deg_e7=lat_e7,
        lon_deg_e7=lon_e7,
        altitude_mm=alt_mm,
        heading_cdeg=heading_cdeg,
        hor_velocity_cms=hor_vel_cms,
        ver_velocity_cms=ver_vel_cms,
        flags=flags,
        squawk=squawk,
        altitude_type=1,
        callsign=callsign,
        emitter_type=int(emitter_type) & 0xFF,
        tslc=max(0, min(255, int(tslc_s))),
        seq=int(seq) & 0xFF,
        sysid=int(sysid) & 0xFF,
        compid=int(compid) & 0xFF,
    )


class MavlinkAdsbUdpStreamer:
    """Non-blocking localhost UDP streamer/receiver for MAVLink v2 ADSB_VEHICLE packets."""

    def __init__(
        self,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        target_host: str = "127.0.0.1",
        target_port: int = 14550,
    ) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self._sock.setblocking(False)
        self._sock.bind((bind_host, bind_port))
        self.local_addr: tuple[str, int] = self._sock.getsockname()
        self.target_addr: tuple[str, int] = (target_host, target_port)
        self._seq: int = 0

    @property
    def port(self) -> int:
        return int(self.local_addr[1])

    def set_target(self, host: str, port: int) -> None:
        self.target_addr = (host, int(port))

    def next_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) & 0xFF
        return s

    def send_packet(self, packet: bytes) -> int:
        return self._sock.sendto(packet, self.target_addr)

    def recv_packets(self, max_packets: int = 256) -> list[bytes]:
        packets: list[bytes] = []
        for _ in range(max_packets):
            try:
                data, _ = self._sock.recvfrom(4096)
                packets.append(data)
            except (BlockingIOError, socket.error):
                break
        return packets

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> MavlinkAdsbUdpStreamer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
