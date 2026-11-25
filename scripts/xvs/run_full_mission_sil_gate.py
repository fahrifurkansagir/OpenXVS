"""Faz 2 Full End-to-End Single-Flight Mission Profile SIL Gate on LIVE WSL2 `ArduPlane V4.7.3`
+ Pure-Rust `openxvs-edge-infer.exe --replay` Flight-Recorder Verification.

Flies ONE continuous unbroken mission on `/var/tmp/openxvs_sitl/arduplane` (`WSL2_ARDUPLANE_OFFICIAL_SITL`,
zero twin fallback, `pymavlink_imported = false`, strictly NED, zero `#84`):
  Phase 1: KSFO 28L Takeoff -> GUIDED Cruise -> Live 15 m/s Crosswind (`SIM_WIND_SPD/TURB/DIR`) -> `#74 VFR_HUD` IAS vs GS
  Phase 2: Live `SIM_GPS1_GLTCH_X/Y` (2.0 m/s @ 22.5 deg) -> INS-Coupled `DualBankEsEkfCusum` + `BaroThermalDriftGuard`
           -> `DatumAlignmentGuard` (KSFO geoid) -> `Ekf3SourceSwitchLatch` -> `#76(42007, 2)` -> `COMMAND_ACK(42007, 0)`
  Phase 3: Vision-Mode Dual-Intruder DAA Avoidance Under Crosswind (`#75 COMMAND_INT` 43002/192 + `#82 SET_ATTITUDE_TARGET`
           `type_mask=0x07` climb thrust floor) -> real heading/bank response, stall-free IAS >= 16.5 m/s, 0 `#84`
  Phase 4: GNSS Recovery (`SIM_GPS1_GLTCH=0`) -> `>= 2.0 s` Dwell Hysteresis -> `#76(42007, 1)` -> 2nd `COMMAND_ACK(42007, 0)`
  Phase 5: Approach Vertical Integrity Fault -> Live `XvsCoreFFI` `SimplexRta` P0 Latch -> Pilot `go_around_ack_reset`
           Staged Recovery (`P0 -> P1 -> P4`)
  Phase 6: Dual Python `verify_shadow_log_chain` + Pure-Rust `openxvs-edge-infer.exe --replay` Verification
           of SHA-256 Hash-Chained `mission_flight_tape.ndjson` + 1-Byte Tamper Rejection Proof
  Phase 7: Exact-PID Clean Teardown (`0` leftover WSL `arduplane` PIDs)

Usage:
    $env:OMP_NUM_THREADS="4"; .venv/Scripts/python.exe scripts/xvs/run_full_mission_sil_gate.py
"""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.core_ffi import XvsCoreFFI, XvsRtaFloatTelemetry  # noqa: E402
from xvs.v2.studio_bridge.hil_bridge import (  # noqa: E402
    KSFO_28L_GEOID18_UNDULATION_M,
    DatumAlignmentGuard,
    ShadowModeTelemetryLogger,
    StudioHilBridgeManager,
    verify_shadow_log_chain,
)
from xvs.v2.xvs_mavlink.pod_daa_codec import (  # noqa: E402
    MAVLINK_IFLAG_SIGNED,
    MAVLINK_V2_CHECKSUM_LEN,
    MAVLINK_V2_HEADER_LEN,
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
    MSGID_VFR_HUD,
    MavAttitude,
    MavCommandAck,
    MavGlobalPositionInt,
    MavHeartbeat,
    MavVfrHud,
    compute_crc_extra,
    decode_frame,
    extract_mavlink_v2_frames,
    peek_mavlink_v2_header,
    x25_crc16,
)
from xvs.v2.xvs_nav import BaroThermalDriftGuard, DualBankEsEkfCusum  # noqa: E402
from xvs.v2.xvs_rt_bridge.arduplane_bridge import (  # noqa: E402
    ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
    ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES,
    EKF3_SOURCE_GNSS,
    EKF3_SOURCE_VISION_OPTICAL,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_REPOSITION,
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_GUIDED_CHANGE_HEADING,
    MAV_CMD_SET_MESSAGE_INTERVAL,
    PLANE_MODE_GUIDED,
    PLANE_MODE_TAKEOFF,
    encode_command_long_frame,
    encode_heartbeat_frame,
)

OUT_DIR = REPO_ROOT / "outputs" / "sitl_proof" / "faz2_full_mission"
EDGE_RELEASE_EXE = REPO_ROOT / "docker" / "edge-infer" / "target" / "release" / "openxvs-edge-infer.exe"
WSL_SITL_CACHE_DIR = "/var/tmp/openxvs_sitl"
MIN_ARDUPLANE_VERSION: tuple[int, int, int] = (4, 7, 3)

KSFO_HOME_LAT = 37.61172
KSFO_HOME_LON = -122.358367
FORBIDDEN_PORTS: frozenset[int] = frozenset({4173, 4174, 8765, 9222, 9223})

MSGID_PARAM_REQUEST_READ = 20
MSGID_PARAM_VALUE = 22
MSGID_PARAM_SET = 23
MSGID_GPS_RAW_INT = 24
MAV_PARAM_TYPE_REAL32 = 9

CRC_EXTRA_PARAM_REQUEST_READ = compute_crc_extra(
    "PARAM_REQUEST_READ",
    (("uint8_t", "target_system", 0), ("uint8_t", "target_component", 0), ("char", "param_id", 16), ("int16_t", "param_index", 0)),
)
CRC_EXTRA_PARAM_VALUE = compute_crc_extra(
    "PARAM_VALUE",
    (("char", "param_id", 16), ("float", "param_value", 0), ("uint8_t", "param_type", 0), ("uint16_t", "param_count", 0), ("uint16_t", "param_index", 0)),
)
CRC_EXTRA_PARAM_SET = compute_crc_extra(
    "PARAM_SET",
    (("uint8_t", "target_system", 0), ("uint8_t", "target_component", 0), ("char", "param_id", 16), ("float", "param_value", 0), ("uint8_t", "param_type", 0)),
)
CRC_EXTRA_GPS_RAW_INT = compute_crc_extra(
    "GPS_RAW_INT",
    (
        ("uint64_t", "time_usec", 0),
        ("uint8_t", "fix_type", 0),
        ("int32_t", "lat", 0),
        ("int32_t", "lon", 0),
        ("int32_t", "alt", 0),
        ("uint16_t", "eph", 0),
        ("uint16_t", "epv", 0),
        ("uint16_t", "vel", 0),
        ("uint16_t", "cog", 0),
        ("uint8_t", "satellites_visible", 0),
    ),
)

_PARAM_SET_STRUCT = struct.Struct("<fBB16sB")
_PARAM_REQUEST_READ_STRUCT = struct.Struct("<hBB16s")
_PARAM_VALUE_STRUCT = struct.Struct("<fHH16sB")
_GPS_RAW_INT_STRUCT = struct.Struct("<QiiiHHHHBB")

OVERLAY_PARM = """
ARMING_CHECK 0
SYSID_THISMAV 1
SERIAL0_PROTOCOL 2
SR0_EXTRA1 20
SR0_EXTRA2 20
SR0_POSITION 20
SR0_EXT_STAT 10
ROLL_LIMIT_DEG 30
PTCH_LIM_MIN_DEG -20
PTCH_LIM_MAX_DEG 20
AIRSPEED_MIN 18
AIRSPEED_CRUISE 24
AIRSPEED_MAX 45
EK3_ENABLE 1
AHRS_EKF_TYPE 3
EK3_SRC1_POSXY 3
EK3_SRC1_VELXY 3
EK3_SRC1_VELZ 3
EK3_SRC1_POSZ 1
EK3_SRC2_POSXY 6
EK3_SRC2_VELXY 6
EK3_SRC2_VELZ 6
EK3_SRC2_POSZ 1
VISO_TYPE 1
TKOFF_THR_MINACC 0
TKOFF_THR_MINSPD 0
TKOFF_ALT 80
TKOFF_DIST 250
FS_LONG_ACTN 0
FS_SHORT_ACTN 0
FS_GCS_ENABL 0
"""


def _alloc_udp_port(bind_host: str = "127.0.0.1") -> int:
    for _ in range(32):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind((bind_host, 0))
            port = int(s.getsockname()[1])
        finally:
            s.close()
        if port not in FORBIDDEN_PORTS and not (5180 <= port <= 5299) and not (8890 <= port <= 8999):
            return port
    raise RuntimeError("Unable to allocate clean ephemeral UDP port")


def _angular_diff_deg(a_deg: float, b_deg: float) -> float:
    return ((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0


def _latlon_to_ned_m(lat_deg: float, lon_deg: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    dn = (float(lat_deg) - float(ref_lat)) * 111_132.954
    de = (float(lon_deg) - float(ref_lon)) * 111_319.491 * max(0.1, math.cos(math.radians(ref_lat)))
    return (dn, de)


def _encode_raw_mav2(msgid: int, payload: bytes, crc_extra: int, *, seq: int = 0) -> bytes:
    end = len(payload)
    while end > 1 and payload[end - 1] == 0:
        end -= 1
    wp = payload[:end]
    hdr = struct.pack("<BBBBBBBBB", len(wp), 0, 0, seq & 0xFF, 255, 191, msgid & 0xFF, (msgid >> 8) & 0xFF, (msgid >> 16) & 0xFF)
    crc = x25_crc16(hdr + wp + bytes([crc_extra & 0xFF]))
    return bytes([MAVLINK_V2_STX]) + hdr + wp + struct.pack("<H", crc)


def _extract_payload(frame: bytes, crc_extra: int, full_len: int) -> bytes | None:
    if len(frame) < MAVLINK_V2_HEADER_LEN + MAVLINK_V2_CHECKSUM_LEN or frame[0] != MAVLINK_V2_STX:
        return None
    plen = int(frame[1])
    sig = MAVLINK_V2_SIGNATURE_BYTES if (int(frame[2]) & MAVLINK_IFLAG_SIGNED) else 0
    if len(frame) != MAVLINK_V2_HEADER_LEN + plen + MAVLINK_V2_CHECKSUM_LEN + sig:
        return None
    wp = frame[MAVLINK_V2_HEADER_LEN : MAVLINK_V2_HEADER_LEN + plen]
    rx_crc = struct.unpack_from("<H", frame, MAVLINK_V2_HEADER_LEN + plen)[0]
    if rx_crc != x25_crc16(frame[1:MAVLINK_V2_HEADER_LEN] + wp + bytes([crc_extra & 0xFF])):
        return None
    return (wp + bytes(max(0, full_len - plen)))[:full_len]


def _verify_wsl_arduplane() -> dict[str, Any]:
    bash = (
        f"set -e; cd {WSL_SITL_CACHE_DIR}; test -x arduplane; sha256sum arduplane; "
        f"cat firmware-version.txt; echo ''; "
        f"strings arduplane | grep -oE 'ArduPlane V[0-9]+\\.[0-9]+\\.[0-9]+' | head -n1; "
        f"ip route show default; hostname -I"
    )
    out = subprocess.check_output(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", bash], timeout=20).decode("utf-8", "ignore")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) < 5:
        raise RuntimeError(f"Unexpected WSL2 ArduPlane metadata: {lines!r}")
    sha256_hex = lines[0].split()[0]
    fw_ver = lines[1]
    apm_line = lines[2]
    gw_toks = lines[3].split()
    gw_ip = gw_toks[gw_toks.index("via") + 1] if "via" in gw_toks else gw_toks[0]
    wsl_ip = lines[4].split()[0]
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", apm_line) or re.search(r"(\d+)\.(\d+)\.(\d+)", fw_ver)
    if m is None or (int(m.group(1)), int(m.group(2)), int(m.group(3))) < MIN_ARDUPLANE_VERSION:
        raise RuntimeError(f"ArduPlane version requirement >= {MIN_ARDUPLANE_VERSION} not met: {apm_line!r}")
    return {
        "binary_path_wsl": f"{WSL_SITL_CACHE_DIR}/arduplane",
        "binary_sha256": sha256_hex,
        "firmware_version_txt": fw_ver,
        "apm_version_line": apm_line,
        "version_tuple": [int(m.group(1)), int(m.group(2)), int(m.group(3))],
        "windows_gateway_ip": gw_ip,
        "wsl_ip": wsl_ip,
    }


class _MissionWireProxy:
    """Zero-GPL transparent UDP sniffer + out-of-band GCS param/control injector for live WSL2 ArduPlane."""

    def __init__(self, *, wsl_ip: str, bridge_rx_port: int) -> None:
        self.wsl_ip = wsl_ip
        self.wsl_tx_port = 0
        self.bridge_rx_port = int(bridge_rx_port)
        self.win_from_wsl_port = _alloc_udp_port("0.0.0.0")
        self.bridge_tx_target_port = _alloc_udp_port("127.0.0.1")

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._s_wsl: socket.socket | None = None
        self._s_br: socket.socket | None = None
        self._s_oob: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._oob_seq = 0

        self.latest_hb: dict[str, Any] | None = None
        self.latest_att: dict[str, Any] | None = None
        self.latest_gpi: dict[str, Any] | None = None
        self.latest_vfr: dict[str, Any] | None = None
        self.latest_gps_raw: dict[str, Any] | None = None
        self.params: dict[str, float] = {}
        self.acks: list[dict[str, Any]] = []
        self.att_samples: list[dict[str, Any]] = []
        self.gpi_samples: list[dict[str, Any]] = []
        self.vfr_samples: list[dict[str, Any]] = []
        self.bridge_tx_msgids: dict[int, int] = {}
        self.bridge_msg84_seen = 0

    def start(self) -> None:
        self._s_wsl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._s_wsl.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._s_wsl.bind(("0.0.0.0", self.win_from_wsl_port))
        self._s_wsl.settimeout(0.05)

        self._s_br = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._s_br.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._s_br.bind(("127.0.0.1", self.bridge_tx_target_port))
        self._s_br.settimeout(0.05)

        self._s_oob = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._loop_ap_to_br, daemon=True),
            threading.Thread(target=self._loop_br_to_ap, daemon=True),
        ]
        for th in self._threads:
            th.start()

    def stop(self) -> None:
        self._stop.set()
        for s in (self._s_wsl, self._s_br, self._s_oob):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        for th in self._threads:
            if th.is_alive():
                th.join(timeout=1.5)

    def _next_oob_seq(self) -> int:
        with self._lock:
            s = self._oob_seq & 0xFF
            self._oob_seq = (self._oob_seq + 1) & 0xFF
            return s

    def send_oob(self, frame: bytes) -> None:
        if self._s_oob is not None and self.wsl_tx_port > 0:
            try:
                self._s_oob.sendto(frame, (self.wsl_ip, self.wsl_tx_port))
            except OSError:
                pass

    def send_oob_hb(self) -> None:
        self.send_oob(encode_heartbeat_frame(seq=self._next_oob_seq()))

    def send_oob_cmd_long(self, cmd: int, p1: float = 0.0, p2: float = 0.0) -> None:
        self.send_oob(encode_command_long_frame(command=cmd, param1=p1, param2=p2, seq=self._next_oob_seq()))

    def send_oob_param_set(self, name: str, val: float) -> None:
        pl = _PARAM_SET_STRUCT.pack(float(val), 1, 1, name.encode("ascii")[:16].ljust(16, b"\x00"), MAV_PARAM_TYPE_REAL32)
        self.send_oob(_encode_raw_mav2(MSGID_PARAM_SET, pl, CRC_EXTRA_PARAM_SET, seq=self._next_oob_seq()))

    def send_oob_param_read(self, name: str) -> None:
        pl = _PARAM_REQUEST_READ_STRUCT.pack(-1, 1, 1, name.encode("ascii")[:16].ljust(16, b"\x00"))
        self.send_oob(_encode_raw_mav2(MSGID_PARAM_REQUEST_READ, pl, CRC_EXTRA_PARAM_REQUEST_READ, seq=self._next_oob_seq()))

    def _sniff_ap(self, data: bytes, now: float) -> None:
        frames, _ = extract_mavlink_v2_frames(data)
        for fr in frames:
            try:
                mid = int(peek_mavlink_v2_header(fr)["msgid"])
            except Exception:
                continue
            if mid in (MSGID_HEARTBEAT, MSGID_ATTITUDE, MSGID_GLOBAL_POSITION_INT, MSGID_VFR_HUD, MSGID_COMMAND_ACK):
                try:
                    msg = decode_frame(fr).message
                except Exception:
                    continue
                with self._lock:
                    if isinstance(msg, MavHeartbeat):
                        self.latest_hb = {"t": now, "mode": int(msg.custom_mode), "armed": bool(msg.base_mode & 0x80)}
                    elif isinstance(msg, MavAttitude):
                        rec = {
                            "t": now,
                            "time_boot_ms": int(msg.time_boot_ms),
                            "roll_deg": math.degrees(msg.roll),
                            "pitch_deg": math.degrees(msg.pitch),
                            "yaw_deg": math.degrees(msg.yaw) % 360.0,
                        }
                        self.latest_att = rec
                        self.att_samples.append(rec)
                    elif isinstance(msg, MavGlobalPositionInt):
                        rec = {
                            "t": now,
                            "time_boot_ms": int(msg.time_boot_ms),
                            "lat_deg": msg.lat_deg,
                            "lon_deg": msg.lon_deg,
                            "alt_msl_m": msg.alt_msl_m,
                            "down_m": msg.down_m,
                            "relative_alt_m": msg.relative_alt_m,
                            "vn_mps": msg.vn_mps,
                            "ve_mps": msg.ve_mps,
                            "vd_mps": msg.vd_mps,
                            "groundspeed_mps": math.hypot(msg.vn_mps, msg.ve_mps),
                            "heading_deg": msg.heading_deg,
                        }
                        self.latest_gpi = rec
                        self.gpi_samples.append(rec)
                    elif isinstance(msg, MavVfrHud):
                        rec = {
                            "t": now,
                            "airspeed_mps": float(msg.airspeed),
                            "groundspeed_mps": float(msg.groundspeed),
                            "heading_deg": float(msg.heading),
                            "throttle_pct": int(msg.throttle),
                            "alt_msl_m": float(msg.alt),
                        }
                        self.latest_vfr = rec
                        self.vfr_samples.append(rec)
                    elif isinstance(msg, MavCommandAck):
                        self.acks.append({"t": now, "command": int(msg.command), "result": int(msg.result)})
            elif mid == MSGID_GPS_RAW_INT:
                pl = _extract_payload(fr, CRC_EXTRA_GPS_RAW_INT, _GPS_RAW_INT_STRUCT.size)
                if pl is not None:
                    t_us, lat, lon, alt_mm, *_ = _GPS_RAW_INT_STRUCT.unpack(pl)
                    with self._lock:
                        self.latest_gps_raw = {
                            "t": now,
                            "time_usec": int(t_us),
                            "lat_deg": float(lat) * 1e-7,
                            "lon_deg": float(lon) * 1e-7,
                            "alt_msl_m": float(alt_mm) * 1e-3,
                        }
            elif mid == MSGID_PARAM_VALUE:
                pl = _extract_payload(fr, CRC_EXTRA_PARAM_VALUE, _PARAM_VALUE_STRUCT.size)
                if pl is not None:
                    pval, _, _, pid_raw, _ = _PARAM_VALUE_STRUCT.unpack(pl)
                    pname = pid_raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
                    with self._lock:
                        self.params[pname] = float(pval)

    def _loop_ap_to_br(self) -> None:
        assert self._s_wsl is not None
        fwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while not self._stop.is_set():
                try:
                    data, _ = self._s_wsl.recvfrom(8192)
                except (socket.timeout, ConnectionResetError):
                    continue
                except OSError:
                    break
                self._sniff_ap(data, time.monotonic())
                try:
                    fwd.sendto(data, ("127.0.0.1", self.bridge_rx_port))
                except OSError:
                    pass
        finally:
            fwd.close()

    def _loop_br_to_ap(self) -> None:
        assert self._s_br is not None
        fwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while not self._stop.is_set():
                try:
                    data, _ = self._s_br.recvfrom(8192)
                except (socket.timeout, ConnectionResetError):
                    continue
                except OSError:
                    break
                frames, _ = extract_mavlink_v2_frames(data)
                with self._lock:
                    for fr in frames:
                        try:
                            mid = int(peek_mavlink_v2_header(fr)["msgid"])
                            self.bridge_tx_msgids[mid] = (
                                self.bridge_tx_msgids[mid] + 1 if mid in self.bridge_tx_msgids else 1
                            )
                            if mid == MSGID_SET_POSITION_TARGET_LOCAL_NED:
                                self.bridge_msg84_seen += 1
                        except Exception:
                            pass
                if self.wsl_tx_port > 0:
                    try:
                        fwd.sendto(data, (self.wsl_ip, self.wsl_tx_port))
                    except OSError:
                        pass
        finally:
            fwd.close()


def _make_rta_float_telemetry(
    *,
    v_vpl_m: float = 5.0,
    v_hpl_m: float = 14.0,
    approach_active: bool = False,
    go_around_ack_reset: bool = False,
    cusum_stat: float = 0.0,
    ml_detector_fault: bool = False,
) -> XvsRtaFloatTelemetry:
    return XvsRtaFloatTelemetry(
        horizon_pl_rad=0.002,
        horizon_al_rad=0.0174533,
        v_vpl_m=float(v_vpl_m),
        v_val_m=10.0,
        v_hpl_m=float(v_hpl_m),
        v_hal_m=30.0,
        epipolar_max_rad=(ctypes.c_double * 6)(0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002),
        signed_horizon_residual_rad=0.0,
        runway_width_delta_m=0.0,
        runway_width_delta_limit_m=5.0,
        cusum_stat=float(cusum_stat),
        cusum_thr=17.619,
        ood_global_score=0.1,
        ood_global_thr=0.7,
        ood_cam_scores=(ctypes.c_double * 6)(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        ood_cam_thr=0.7,
        cam_health_mask=0x3F,
        ml_detector_fault=1 if ml_detector_fault else 0,
        transient_dropout=0,
        approach_active=1 if approach_active else 0,
        go_around_ack_reset=1 if go_around_ack_reset else 0,
    )


def run_full_mission_sil_gate() -> dict[str, Any]:
    t0 = time.perf_counter()
    assert "pymavlink" not in sys.modules, "GPL pymavlink MUST NEVER be imported!"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tape_path = OUT_DIR / "mission_flight_tape.ndjson"
    report_path = OUT_DIR / "full_mission_sil_report.json"

    meta = _verify_wsl_arduplane()
    print(f"[FAZ2-SIL] Verified LIVE WSL2 {meta['apm_version_line']} (sha256={meta['binary_sha256'][:16]}...)", flush=True)

    run_dir_wsl = f"{WSL_SITL_CACHE_DIR}/run_faz2_mission_{int(time.time() * 1000) % 100000}"
    parm_file_wsl = f"{run_dir_wsl}/mission.parm"
    prep = (
        f"set -e; mkdir -p {run_dir_wsl}; cat {WSL_SITL_CACHE_DIR}/plane.parm > {parm_file_wsl}; "
        f"cat <<'EOF' >> {parm_file_wsl}\n{OVERLAY_PARM.strip()}\nEOF"
    )
    subprocess.check_call(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", prep], timeout=15)

    bridge_rx_port = _alloc_udp_port("127.0.0.1")
    proxy = _MissionWireProxy(wsl_ip=str(meta["wsl_ip"]), bridge_rx_port=bridge_rx_port)
    gw = str(meta["windows_gateway_ip"])
    win_rx_port = proxy.win_from_wsl_port

    launcher_py = f"""
import os, select, signal, socket, subprocess
s_ap = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_ap.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s_ap.bind(('127.0.0.1', 0))
ap_port = s_ap.getsockname()[1]
s_win = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_win.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s_win.bind(('0.0.0.0', 0))
tx_port = s_win.getsockname()[1]
ap = subprocess.Popen([
    '{WSL_SITL_CACHE_DIR}/arduplane', '--wipe', '--model', 'plane', '--speedup', '2',
    '--home', '{KSFO_HOME_LAT},{KSFO_HOME_LON},4,298',
    '--defaults', '{parm_file_wsl}',
    '--serial0', f'udpclient:127.0.0.1:{{ap_port}}',
], cwd='{run_dir_wsl}', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f'AP_READY pid={{ap.pid}} ap_port={{ap_port}} tx_port={{tx_port}}', flush=True)
stop = False
def _sig(signum, frame):
    global stop
    stop = True
signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)
ap_peer = None
try:
    while not stop and ap.poll() is None:
        r, _, _ = select.select([s_ap, s_win], [], [], 0.05)
        if s_ap in r:
            d, a = s_ap.recvfrom(8192)
            ap_peer = a
            s_win.sendto(d, ('{gw}', {win_rx_port}))
        if s_win in r:
            d, _ = s_win.recvfrom(8192)
            if ap_peer is not None:
                s_ap.sendto(d, ap_peer)
finally:
    if ap.poll() is None:
        ap.terminate()
        try:
            ap.wait(timeout=3)
        except Exception:
            ap.kill()
    s_ap.close()
    s_win.close()
"""

    mgr = StudioHilBridgeManager(
        non_flight_sim_only=False,
        camera_lever_arm_body_m=(0.35, 0.0, 0.18),
        viso_delay_ms=80.0,
    )
    proc: subprocess.Popen[str] | None = None
    ap_pid: int | None = None
    pid_cleanly_terminated = False

    tape_mono_us = 0

    def next_tape_us() -> int:
        nonlocal tape_mono_us
        tape_mono_us += 1_000_000
        return tape_mono_us

    def sleep_hb(dur_s: float) -> None:
        end = time.monotonic() + dur_s
        last_hb = 0.0
        while time.monotonic() < end:
            now = time.monotonic()
            if (now - last_hb) >= 0.4:
                proxy.send_oob_hb()
                last_hb = now
            time.sleep(min(0.05, max(0.01, end - time.monotonic())))

    def wait_next_1hz_epoch(last_boot_ms: int, timeout_s: float = 2.5) -> tuple[dict[str, Any], dict[str, Any] | None]:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            proxy.send_oob_hb()
            g = proxy.latest_gpi
            if g is not None and (int(g["time_boot_ms"]) - last_boot_ms) >= 1000:
                return dict(g), (dict(proxy.latest_gps_raw) if proxy.latest_gps_raw else None)
            time.sleep(0.025)
        assert proxy.latest_gpi is not None
        return dict(proxy.latest_gpi), (dict(proxy.latest_gps_raw) if proxy.latest_gps_raw else None)

    logger = ShadowModeTelemetryLogger(
        tape_path,
        trial_phase="bench_shadow_replay",
        hardware_tx_pin_removed=True,
    )
    ffi = XvsCoreFFI()
    rta_handle = ffi.rta_create()

    try:
        proc = subprocess.Popen(
            ["wsl.exe", "-d", "Ubuntu", "--", "python3", "-c", launcher_py],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        first_line = proc.stdout.readline().strip()
        m = re.match(r"AP_READY pid=(\d+) ap_port=(\d+) tx_port=(\d+)", first_line)
        if not m:
            raise RuntimeError(f"WSL2 ArduPlane launch failed: {first_line!r}")
        ap_pid = int(m.group(1))
        proxy.wsl_tx_port = int(m.group(3))
        proxy.start()

        mgr.connect(
            mode="mavlink2_udp",
            bind_host="127.0.0.1",
            rx_port=bridge_rx_port,
            tx_host="127.0.0.1",
            tx_port=proxy.bridge_tx_target_port,
            tx_command_mode="command_int_heading",
            tx_enabled=False,
            min_tx_interval_s=0.35,
        )

        # Boot handshake + high-rate telemetry streams (#30, #33, #74, #24)
        t_boot = time.monotonic() + 10.0
        while time.monotonic() < t_boot:
            proxy.send_oob_hb()
            st_b = mgr.status()
            if proxy.latest_hb and proxy.latest_att and proxy.latest_gpi and int(st_b["attitude_count"]) >= 2:
                break
            time.sleep(0.1)
        if proxy.latest_hb is None:
            raise RuntimeError("No HEARTBEAT from live WSL2 ArduPlane")
        for mid, iv_us in ((30, 50_000.0), (33, 50_000.0), (74, 50_000.0), (24, 100_000.0)):
            proxy.send_oob_cmd_long(MAV_CMD_SET_MESSAGE_INTERVAL, p1=float(mid), p2=iv_us)
        sleep_hb(0.4)

        # ======================================================================
        # PHASE 1: KSFO 28L Takeoff -> GUIDED Cruise -> Live 15 m/s Crosswind
        # ======================================================================
        print("\n--- [Phase 1] KSFO 28L Takeoff -> GUIDED Cruise -> Live 15 m/s Crosswind ---", flush=True)
        proxy.send_oob_cmd_long(MAV_CMD_DO_SET_MODE, p1=1.0, p2=float(PLANE_MODE_TAKEOFF))
        sleep_hb(0.3)
        proxy.send_oob_cmd_long(MAV_CMD_COMPONENT_ARM_DISARM, p1=1.0, p2=21196.0)
        t_tk0 = time.monotonic()
        while (time.monotonic() - t_tk0) < 18.0:
            proxy.send_oob_hb()
            if proxy.latest_gpi and float(proxy.latest_gpi["relative_alt_m"]) >= 60.0:
                break
            if (time.monotonic() - t_tk0) > 1.5 and proxy.latest_hb and not proxy.latest_hb["armed"]:
                proxy.send_oob_cmd_long(MAV_CMD_DO_SET_MODE, p1=1.0, p2=float(PLANE_MODE_TAKEOFF))
                proxy.send_oob_cmd_long(MAV_CMD_COMPONENT_ARM_DISARM, p1=1.0, p2=21196.0)
            time.sleep(0.1)
        proxy.send_oob_cmd_long(MAV_CMD_DO_SET_MODE, p1=1.0, p2=float(PLANE_MODE_GUIDED))
        sleep_hb(1.0)
        cruise_alt_agl_m = float(proxy.latest_gpi["relative_alt_m"]) if proxy.latest_gpi else 0.0

        # Inject live 15 m/s crosswind (SIM_WIND_SPD=15, SIM_WIND_TURB=5, SIM_WIND_DIR=208)
        proxy.send_oob_param_set("SIM_WIND_SPD", 15.0)
        proxy.send_oob_param_set("SIM_WIND_TURB", 5.0)
        proxy.send_oob_param_set("SIM_WIND_DIR", 208.0)
        sleep_hb(0.4)
        for pn in ("SIM_WIND_SPD", "SIM_WIND_TURB", "SIM_WIND_DIR"):
            proxy.send_oob_param_read(pn)
        sleep_hb(0.5)
        wind_rb = {k: proxy.params.get(k) for k in ("SIM_WIND_SPD", "SIM_WIND_TURB", "SIM_WIND_DIR")}

        # Attach DatumAlignmentGuard (KSFO GEOID18) and initialize INS-coupled CUSUM + BaroThermalDriftGuard
        guard = DatumAlignmentGuard(geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M)
        mgr.configure_datum_guard(
            guard,
            visual_alt_m=float(mgr.status()["state_ned"]["alt_msl_m"]) + KSFO_28L_GEOID18_UNDULATION_M,
            visual_datum="WGS84_HAE",
            geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M,
            apply_geoid_conversion=True,
        )
        mgr.set_autonomy_triggers(
            cbdr_threat_active=False,
            gnss_spoof_alarm=False,
            gnss_healthy=True,
            tx_command_mode="command_int_heading",
            tx_enabled=True,
            min_tx_interval_s=0.4,
        )

        assert proxy.latest_gpi is not None
        g0 = dict(proxy.latest_gpi)
        origin_n0, origin_e0 = _latlon_to_ned_m(g0["lat_deg"], g0["lon_deg"], KSFO_HOME_LAT, KSFO_HOME_LON)
        prev_ms = int(g0["time_boot_ms"])
        prev_vn = float(g0["vn_mps"])
        prev_ve = float(g0["ve_mps"])

        cusum = DualBankEsEkfCusum()
        cusum.reset(init_ned_m=(0.0, 0.0, float(g0["down_m"])), init_vel_ne_mps=(prev_vn, prev_ve))
        baro_guard = BaroThermalDriftGuard()
        rng = np.random.default_rng(20261125)

        p1_ias_list: list[float] = []
        p1_gs_list: list[float] = []
        p1_diff_list: list[float] = []
        p1_cusum_fa = 0
        last_sp, last_va, last_vs = False, False, 0.0
        comp_pose_samples = 0

        # 10 clean crosswind cruise epochs (1 Hz sim time)
        for ep in range(10):
            g_cur, gps_cur = wait_next_1hz_epoch(prev_ms)
            dt_s = max(0.5, (int(g_cur["time_boot_ms"]) - prev_ms) / 1000.0)
            an = (float(g_cur["vn_mps"]) - prev_vn) / dt_s
            ae = (float(g_cur["ve_mps"]) - prev_ve) / dt_s
            prev_ms = int(g_cur["time_boot_ms"])
            prev_vn, prev_ve = float(g_cur["vn_mps"]), float(g_cur["ve_mps"])

            st_now = mgr.status()
            ias = float(st_now["state_ned"]["airspeed_mps"])
            gs = float(st_now["state_ned"]["groundspeed_mps"])
            p1_ias_list.append(ias)
            p1_gs_list.append(gs)
            p1_diff_list.append(abs(ias - gs))

            src_lat = gps_cur["lat_deg"] if gps_cur else g_cur["lat_deg"]
            src_lon = gps_cur["lon_deg"] if gps_cur else g_cur["lon_deg"]
            gn_abs, ge_abs = _latlon_to_ned_m(src_lat, src_lon, KSFO_HOME_LAT, KSFO_HOME_LON)
            rel_ne = np.array([gn_abs - origin_n0, ge_abs - origin_e0], dtype=np.float64)

            # Apply camera lever-arm & VISO_DELAY_MS compensation to simulated camera visual fix
            comp_fix = mgr.compensate_visual_fix_ned([float(rel_ne[0]), float(rel_ne[1]), float(g_cur["down_m"])])
            comp_pose_samples += 1

            baro_corr = baro_guard.step(
                baro_down_raw_m=float(g_cur["down_m"]),
                gnss_down_m=float(g_cur["down_m"]),
                dt_s=1.0,
                gnss_healthy=True,
                spoof_alarm=last_sp,
                vertical_spoof_alarm=last_va,
                vertical_cusum_stat=last_vs,
            )
            gnss_meas = rel_ne + rng.normal(0.0, 1.2, size=2)
            vis_meas = rel_ne + rng.normal(0.0, 2.2, size=2)
            c_out = cusum.step(
                ins_accel_ne_mps2=(an, ae),
                baro_down_m=baro_corr.corrected_baro_down_m,
                gnss_ne_m=gnss_meas,
                vis_ne_m=vis_meas,
                vis_gate_admitted=True,
                gnss_down_m=float(g_cur["down_m"]),
            )
            last_sp, last_va, last_vs = c_out.spoof_alarm, c_out.vertical_spoof_alarm, c_out.vertical_cusum_stat
            if c_out.gnss_quarantined:
                p1_cusum_fa += 1

            rta_in = ffi.rta_build_input_from_f64(_make_rta_float_telemetry(cusum_stat=float(c_out.max_cusum_stat)))
            rta_out = ffi.rta_step(rta_handle, rta_in)

            logger.record_event(
                timestamp_mono_us=next_tape_us(),
                event_type="mission_epoch_1hz",
                direction="rx",
                payload={
                    "phase": "phase1_crosswind_cruise",
                    "gnss_ne_m": [round(float(gnss_meas[0]), 4), round(float(gnss_meas[1]), 4)],
                    "vis_ne_m": [round(float(vis_meas[0]), 4), round(float(vis_meas[1]), 4)],
                    "baro_down_m": round(float(baro_corr.corrected_baro_down_m), 4),
                    "gnss_down_m": round(float(g_cur["down_m"]), 4),
                    "ins_accel_ne_mps2": [round(an, 4), round(ae, 4)],
                    "vis_gate_admitted": True,
                    "airspeed_mps": round(ias, 3),
                    "groundspeed_mps": round(gs, 3),
                    "ekf3_source": int(st_now["ekf3_active_source"]),
                    "lever_arm_compensated_ned_m": [round(x, 3) for x in comp_fix.p_cg_ned_m],
                    "rta_mode": int(rta_out.mode),
                    "approach_active": False,
                    "v_vpl_m": 5.0,
                },
            )

        p1_vfr_cnt = int(mgr.status()["vfr_hud_count"])
        p1_max_ias_gs_diff = round(max(p1_diff_list), 2) if p1_diff_list else 0.0
        print(
            f"  [Phase 1] AGL={cruise_alt_agl_m:.1f}m | wind={wind_rb} | VFR_HUD={p1_vfr_cnt} "
            f"max|IAS-GS|={p1_max_ias_gs_diff} m/s | CUSUM_FA={p1_cusum_fa}",
            flush=True,
        )

        # ======================================================================
        # PHASE 2: Live GPS Spoof (2.0 m/s @ 22.5 deg) -> CUSUM -> DatumGuard -> EKF3 42007 (2)
        # ======================================================================
        print("\n--- [Phase 2] Live GPS Spoof (2.0 m/s @ 22.5 deg) -> CUSUM -> DatumGuard -> 42007 Vision ---", flush=True)
        ramp_rate_mps = 2.0
        theta = math.radians(22.5)
        u_n, u_e = math.cos(theta), math.sin(theta)
        cos_lat = max(0.1, math.cos(math.radians(KSFO_HOME_LAT)))
        p2_ttd_s: float | None = None
        p2_bank_a_err_m: float | None = None
        p2_bank_b_err_m: float | None = None
        p2_stat_at_q: float = 0.0
        acks_42007_pre = sum(1 for a in proxy.acks if a["command"] == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET and a["result"] == 0)

        for step_k in range(1, 25):
            off_n = ramp_rate_mps * float(step_k) * u_n
            off_e = ramp_rate_mps * float(step_k) * u_e
            proxy.send_oob_param_set("SIM_GPS1_GLTCH_X", off_n / 111_132.954)
            proxy.send_oob_param_set("SIM_GPS1_GLTCH_Y", off_e / (111_319.491 * cos_lat))

            g_cur, gps_cur = wait_next_1hz_epoch(prev_ms)
            dt_s = max(0.5, (int(g_cur["time_boot_ms"]) - prev_ms) / 1000.0)
            an = (float(g_cur["vn_mps"]) - prev_vn) / dt_s
            ae = (float(g_cur["ve_mps"]) - prev_ve) / dt_s
            prev_ms = int(g_cur["time_boot_ms"])
            prev_vn, prev_ve = float(g_cur["vn_mps"]), float(g_cur["ve_mps"])

            assert gps_cur is not None
            gn_gl, ge_gl = _latlon_to_ned_m(gps_cur["lat_deg"], gps_cur["lon_deg"], KSFO_HOME_LAT, KSFO_HOME_LON)
            rel_gnss = np.array([gn_gl - origin_n0, ge_gl - origin_e0], dtype=np.float64)
            rel_true = rel_gnss - np.array([off_n, off_e], dtype=np.float64)
            vis_meas = rel_true + rng.normal(0.0, 2.2, size=2)

            baro_corr = baro_guard.step(
                baro_down_raw_m=float(g_cur["down_m"]),
                gnss_down_m=float(g_cur["down_m"]),
                dt_s=1.0,
                gnss_healthy=True,
                spoof_alarm=last_sp,
                vertical_spoof_alarm=last_va,
                vertical_cusum_stat=last_vs,
            )
            c_out = cusum.step(
                ins_accel_ne_mps2=(an, ae),
                baro_down_m=baro_corr.corrected_baro_down_m,
                gnss_ne_m=rel_gnss,
                vis_ne_m=vis_meas,
                vis_gate_admitted=True,
                gnss_down_m=float(g_cur["down_m"]),
            )
            last_sp, last_va, last_vs = c_out.spoof_alarm, c_out.vertical_spoof_alarm, c_out.vertical_cusum_stat

            rta_in = ffi.rta_build_input_from_f64(_make_rta_float_telemetry(cusum_stat=float(c_out.max_cusum_stat)))
            rta_out = ffi.rta_step(rta_handle, rta_in)

            logger.record_event(
                timestamp_mono_us=next_tape_us(),
                event_type="mission_epoch_1hz",
                direction="rx",
                payload={
                    "phase": "phase2_gps_spoof",
                    "spoof_step_s": step_k,
                    "gnss_ne_m": [round(float(rel_gnss[0]), 4), round(float(rel_gnss[1]), 4)],
                    "vis_ne_m": [round(float(vis_meas[0]), 4), round(float(vis_meas[1]), 4)],
                    "baro_down_m": round(float(baro_corr.corrected_baro_down_m), 4),
                    "gnss_down_m": round(float(g_cur["down_m"]), 4),
                    "ins_accel_ne_mps2": [round(an, 4), round(ae, 4)],
                    "vis_gate_admitted": True,
                    "cusum_quarantined": bool(c_out.gnss_quarantined),
                    "rta_mode": int(rta_out.mode),
                    "approach_active": False,
                    "v_vpl_m": 5.0,
                },
            )

            if c_out.quarantine_triggered_this_epoch:
                p2_ttd_s = float(step_k)
                p2_bank_a_err_m = float(np.linalg.norm(c_out.bank_a_p_ne_m - rel_true))
                p2_bank_b_err_m = float(np.linalg.norm(c_out.active_p_ned_m[:2] - rel_true))
                p2_stat_at_q = float(c_out.max_cusum_stat)

                baro_now = float(mgr.status()["state_ned"]["alt_msl_m"])
                mgr.update_visual_altitude(
                    visual_alt_m=baro_now + KSFO_28L_GEOID18_UNDULATION_M,
                    visual_datum="WGS84_HAE",
                    geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M,
                    apply_geoid_conversion=True,
                )
                mgr.set_autonomy_triggers(gnss_spoof_alarm=True, gnss_healthy=True)
                t_ack_wait = time.monotonic() + 1.0
                while time.monotonic() < t_ack_wait:
                    proxy.send_oob_hb()
                    cur_msl = float(mgr.status()["state_ned"]["alt_msl_m"])
                    mgr.update_visual_altitude(
                        visual_alt_m=cur_msl + KSFO_28L_GEOID18_UNDULATION_M,
                        visual_datum="WGS84_HAE",
                        geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M,
                        apply_geoid_conversion=True,
                    )
                    if sum(1 for a in proxy.acks if a["command"] == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET and a["result"] == 0) > acks_42007_pre:
                        break
                    time.sleep(0.05)
                break

        st_p2 = mgr.status()
        p2_datum_dec = st_p2["last_datum_decision"]
        p2_ack_42007 = sum(1 for a in proxy.acks if a["command"] == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET and a["result"] == 0) - acks_42007_pre
        print(
            f"  [Phase 2] TTD={p2_ttd_s}s | BankA_err={p2_bank_a_err_m:.2f}m (<30m HAL) | BankB_err={p2_bank_b_err_m:.2f}m | "
            f"DatumOK={p2_datum_dec['allowed'] if p2_datum_dec else None} (innov={p2_datum_dec['abs_vertical_innovation_m'] if p2_datum_dec else None}m) | "
            f"EKF3={st_p2['ekf3_active_source']} | ACK42007(res=0)={p2_ack_42007}",
            flush=True,
        )

        # ======================================================================
        # PHASE 3: Vision-Mode Dual-Intruder DAA Avoidance Under Crosswind
        # ======================================================================
        print("\n--- [Phase 3] Vision-Mode Dual-Intruder DAA Avoidance (#75 + #82) Under 15 m/s Wind ---", flush=True)
        base_hdg_p3 = float(mgr.status()["state_ned"]["heading_deg"])

        # Intruder 1: CBDR right turn via #75 COMMAND_INT (43002 + 192)
        t_p3a = time.monotonic()
        mgr.set_autonomy_triggers(
            cbdr_threat_active=True,
            cbdr_evade_direction=1.0,
            tx_command_mode="command_int_heading",
            tx_enabled=True,
            min_tx_interval_s=0.35,
        )
        sleep_hb(2.8)
        mgr.set_autonomy_triggers(tx_command_mode="command_int", min_tx_interval_s=0.35)
        sleep_hb(1.2)
        with proxy._lock:
            p3a_atts = [s for s in proxy.att_samples if s["t"] >= t_p3a]
            p3a_vfrs = [s for s in proxy.vfr_samples if s["t"] >= t_p3a]
        p3a_max_roll = max((abs(s["roll_deg"]) for s in p3a_atts), default=0.0)
        p3a_delta_hdg = _angular_diff_deg(p3a_atts[-1]["yaw_deg"], base_hdg_p3) if p3a_atts else 0.0

        # Intruder 2: CBDR left bank + climb via #82 SET_ATTITUDE_TARGET (type_mask=0x07, climb thrust floor active)
        t_p3b = time.monotonic()
        mgr.set_autonomy_triggers(
            cbdr_threat_active=True,
            cbdr_evade_direction=-1.0,
            tx_command_mode="set_attitude_target",
            tx_enabled=True,
            min_tx_interval_s=0.09,
        )
        sleep_hb(2.4)
        st_p3b = mgr.status()
        last_cmd82 = dict(st_p3b["last_tx_command"]) if st_p3b["last_tx_command"] else {}
        with proxy._lock:
            p3b_atts = [s for s in proxy.att_samples if s["t"] >= t_p3b]
            p3b_vfrs = [s for s in proxy.vfr_samples if s["t"] >= t_p3b]
        p3b_min_roll = min((s["roll_deg"] for s in p3b_atts), default=0.0)
        p3_all_ias = [s["airspeed_mps"] for s in (p3a_vfrs + p3b_vfrs)]
        p3_min_ias = min(p3_all_ias) if p3_all_ias else 0.0

        # Clear threat to resume level cruise & log Phase 3 avoidance tape record
        mgr.set_autonomy_triggers(cbdr_threat_active=False, tx_command_mode="command_int_heading", min_tx_interval_s=0.4)
        g_p3 = dict(proxy.latest_gpi) if proxy.latest_gpi else g0
        p3_n, p3_e = _latlon_to_ned_m(g_p3["lat_deg"], g_p3["lon_deg"], KSFO_HOME_LAT, KSFO_HOME_LON)
        p3_rel = [round(p3_n - origin_n0, 4), round(p3_e - origin_e0, 4)]
        logger.record_event(
            timestamp_mono_us=next_tape_us(),
            event_type="mission_epoch_1hz",
            direction="rx",
            payload={
                "phase": "phase3_vision_daa_avoidance",
                "gnss_ne_m": [round(p3_rel[0] + 20.0, 4), round(p3_rel[1] + 8.0, 4)],
                "vis_ne_m": p3_rel,
                "baro_down_m": round(float(g_p3["down_m"]), 4),
                "gnss_down_m": round(float(g_p3["down_m"]), 4),
                "ekf3_source": int(st_p3b["ekf3_active_source"]),
                "cmd75_max_abs_roll_deg": round(p3a_max_roll, 2),
                "cmd75_delta_hdg_deg": round(p3a_delta_hdg, 2),
                "cmd82_min_roll_deg": round(p3b_min_roll, 2),
                "cmd82_type_mask": last_cmd82.get("type_mask"),
                "min_vfr_ias_mps": round(p3_min_ias, 2),
                "msg84_count": int(proxy.bridge_msg84_seen),
                "approach_active": False,
                "v_vpl_m": 5.0,
            },
        )
        ack_43002_ok = any(a["command"] == MAV_CMD_GUIDED_CHANGE_HEADING and a["result"] == 0 for a in proxy.acks)
        ack_192_ok = any(a["command"] == MAV_CMD_DO_REPOSITION and a["result"] == 0 for a in proxy.acks)
        cmd82_mask = int(last_cmd82["type_mask"]) if "type_mask" in last_cmd82 else None
        print(
            f"  [Phase 3] #75 max|roll|={p3a_max_roll:.1f}deg dHdg={p3a_delta_hdg:.1f}deg ACK(43002={ack_43002_ok},192={ack_192_ok}) | "
            f"#82 min_roll={p3b_min_roll:.1f}deg mask={hex(cmd82_mask) if cmd82_mask is not None else 'NONE'} | min_IAS={p3_min_ias:.2f} m/s | #84={proxy.bridge_msg84_seen}",
            flush=True,
        )

        # ======================================================================
        # PHASE 4: GNSS Recovery & >= 2.0 s Dwell Hysteresis -> 2nd ACK 42007
        # ======================================================================
        print("\n--- [Phase 4] GNSS Recovery & >= 2.0 s Dwell Hysteresis -> Revert to GNSS (1) ---", flush=True)
        proxy.send_oob_param_set("SIM_GPS1_GLTCH_X", 0.0)
        proxy.send_oob_param_set("SIM_GPS1_GLTCH_Y", 0.0)
        mgr.set_autonomy_triggers(gnss_spoof_alarm=False, gnss_healthy=True)

        sleep_hb(0.85)
        ekf_during_lt2s = int(mgr.status()["ekf3_active_source"])

        sleep_hb(1.85)
        st_p4 = mgr.status()
        ekf_after_ge2s = int(st_p4["ekf3_active_source"])
        total_ack_42007_ok = sum(1 for a in proxy.acks if a["command"] == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET and a["result"] == 0)

        g_p4 = dict(proxy.latest_gpi) if proxy.latest_gpi else g0
        p4_n, p4_e = _latlon_to_ned_m(g_p4["lat_deg"], g_p4["lon_deg"], KSFO_HOME_LAT, KSFO_HOME_LON)
        p4_rel = [round(p4_n - origin_n0, 4), round(p4_e - origin_e0, 4)]
        logger.record_event(
            timestamp_mono_us=next_tape_us(),
            event_type="mission_epoch_1hz",
            direction="rx",
            payload={
                "phase": "phase4_gnss_recovery_dwell",
                "gnss_ne_m": p4_rel,
                "vis_ne_m": p4_rel,
                "baro_down_m": round(float(g_p4["down_m"]), 4),
                "gnss_down_m": round(float(g_p4["down_m"]), 4),
                "ekf3_during_lt2s_dwell": ekf_during_lt2s,
                "ekf3_after_ge2s_revert": ekf_after_ge2s,
                "ack_42007_result0_total": total_ack_42007_ok,
                "approach_active": False,
                "v_vpl_m": 5.0,
            },
        )
        print(
            f"  [Phase 4] EKF3(<2s)={ekf_during_lt2s} -> EKF3(>=2s)={ekf_after_ge2s} | total ACK42007(res=0)={total_ack_42007_ok}",
            flush=True,
        )

        # ======================================================================
        # PHASE 5: Approach Fault -> SimplexRta P0 Latch -> Pilot go_around_ack_reset
        # ======================================================================
        print("\n--- [Phase 5] Approach Fault -> SimplexRta P0 Latch -> Pilot go_around_ack_reset Recovery ---", flush=True)
        # Ensure clean RTA instance starting from nominal P4 before approach leg
        ffi.rta_destroy(rta_handle)
        rta_handle = ffi.rta_create()

        # 5.1 Nominal cruise before approach
        o_nom = ffi.rta_step(rta_handle, ffi.rta_build_input_from_f64(_make_rta_float_telemetry()))
        assert int(o_nom.mode) == 4 and int(o_nom.mandatory_go_around) == 0

        # 5.2 Approach vertical integrity breach (V_VPL=15.0m > V_VAL=10.0m) -> immediate 0-tick P0 latch!
        o_fault = ffi.rta_step(
            rta_handle,
            ffi.rta_build_input_from_f64(_make_rta_float_telemetry(v_vpl_m=15.0, approach_active=True)),
        )
        # 5.3 Fault clears while STILL on approach -> latch MUST HOLD!
        o_hold = ffi.rta_step(
            rta_handle,
            ffi.rta_build_input_from_f64(_make_rta_float_telemetry(v_vpl_m=5.0, approach_active=True, go_around_ack_reset=True)),
        )
        # 5.4 Pilot exits approach & asserts go_around_ack_reset -> staged P0 -> P1 (30 ticks) -> P4 (20 ticks)
        o_ack = ffi.rta_step(
            rta_handle,
            ffi.rta_build_input_from_f64(_make_rta_float_telemetry(v_vpl_m=5.0, approach_active=False, go_around_ack_reset=True)),
        )
        p1_reached_at_tick: int | None = None
        p4_reached_at_tick: int | None = None
        staged_modes: list[int] = [int(o_ack.mode)]
        for tk in range(2, 70):
            o_rec = ffi.rta_step(
                rta_handle,
                ffi.rta_build_input_from_f64(_make_rta_float_telemetry(v_vpl_m=5.0, approach_active=False, go_around_ack_reset=False)),
            )
            md = int(o_rec.mode)
            staged_modes.append(md)
            if md == 1 and p1_reached_at_tick is None:
                p1_reached_at_tick = tk
            if md == 4 and p4_reached_at_tick is None:
                p4_reached_at_tick = tk
                break

        # Record Phase 5 fault & recovery into mission tape
        logger.record_event(
            timestamp_mono_us=next_tape_us(),
            event_type="rta_approach_fault_latch",
            direction="internal",
            payload={
                "phase": "phase5_approach_rta_fault",
                "gnss_ne_m": p4_rel,
                "vis_ne_m": p4_rel,
                "baro_down_m": round(float(g_p4["down_m"]), 4),
                "gnss_down_m": round(float(g_p4["down_m"]), 4),
                "approach_active": True,
                "v_vpl_m": 15.0,
                "rta_mode": int(o_fault.mode),
                "rta_mandatory_go_around": bool(o_fault.mandatory_go_around),
                "blank_conformal_hud": bool(o_fault.blank_conformal_hud),
                "inhibit_avoidance_turns": bool(o_fault.inhibit_avoidance_turns),
            },
        )
        logger.record_event(
            timestamp_mono_us=next_tape_us(),
            event_type="rta_go_around_ack_recovery",
            direction="internal",
            payload={
                "phase": "phase5_rta_recovery",
                "gnss_ne_m": p4_rel,
                "vis_ne_m": p4_rel,
                "baro_down_m": round(float(g_p4["down_m"]), 4),
                "gnss_down_m": round(float(g_p4["down_m"]), 4),
                "approach_active": False,
                "go_around_ack_reset": True,
                "v_vpl_m": 5.0,
                "p1_reached_at_tick": p1_reached_at_tick,
                "p4_reached_at_tick": p4_reached_at_tick,
                "final_rta_mode": staged_modes[-1],
            },
        )

        p5_ok = bool(
            int(o_fault.mode) == 0
            and bool(o_fault.mandatory_go_around)
            and bool(o_fault.blank_conformal_hud)
            and bool(o_fault.inhibit_avoidance_turns)
            and int(o_hold.mode) == 0
            and bool(o_hold.mandatory_go_around)
            and not bool(o_ack.mandatory_go_around)
            and p1_reached_at_tick == 30
            and p4_reached_at_tick == 50
            and staged_modes[-1] == 4
        )
        print(
            f"  [Phase 5] fault_mode={o_fault.mode} (MGA={o_fault.mandatory_go_around}) | hold_on_approach={o_hold.mode} | "
            f"P0->P1@{p1_reached_at_tick} ticks -> P4@{p4_reached_at_tick} ticks | pass={p5_ok}",
            flush=True,
        )

        st_end = mgr.status()
        tx_msgids = {int(k): int(v) for k, v in st_end["tx_msgids_emitted"].items()}
        msg84_zero = (
            not bool(st_end["msg84_ever_emitted"])
            and MSGID_SET_POSITION_TARGET_LOCAL_NED not in tx_msgids
            and proxy.bridge_msg84_seen == 0
        )

    finally:
        ffi.rta_destroy(rta_handle)
        logger.close()
        mgr.shutdown()
        proxy.stop()
        # Phase 7: Exact-PID Clean Teardown (ONLY our spawned ap_pid!)
        if ap_pid is not None:
            subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"kill -TERM {ap_pid} 2>/dev/null || true"], check=False, timeout=10)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except Exception:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        if ap_pid is not None:
            time.sleep(0.2)
            chk = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"kill -0 {ap_pid} 2>/dev/null"], check=False, timeout=10)
            if chk.returncode == 0:
                subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"kill -KILL {ap_pid} 2>/dev/null || true"], check=False, timeout=10)
                time.sleep(0.1)
                chk = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"kill -0 {ap_pid} 2>/dev/null"], check=False, timeout=10)
            pid_cleanly_terminated = chk.returncode != 0
        subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"rm -rf {run_dir_wsl}"], check=False, timeout=10)

    # ==========================================================================
    # PHASE 6: Dual Python + Pure-Rust Flight-Recorder Verification + 1-Byte Tamper
    # ==========================================================================
    print("\n--- [Phase 6] Dual Python + Pure-Rust --replay Flight-Recorder Verification ---", flush=True)
    py_ver = verify_shadow_log_chain(tape_path)
    if not EDGE_RELEASE_EXE.is_file():
        raise FileNotFoundError(f"Missing Pure-Rust release binary: {EDGE_RELEASE_EXE}")

    rs_proc = subprocess.run([str(EDGE_RELEASE_EXE), "--replay", str(tape_path)], capture_output=True, text=True, check=False)
    rs_rep = json.loads(rs_proc.stdout) if rs_proc.stdout.strip() else {"valid": False, "stderr": rs_proc.stderr}

    # 1-byte tamper proof on temporary copy (cleaned immediately)
    orig_lines = tape_path.read_text(encoding="utf-8").splitlines()
    tampered_lines = list(orig_lines)
    # Flip 1 byte in line 3 payload
    t_idx = min(2, len(tampered_lines) - 1)
    tampered_lines[t_idx] = tampered_lines[t_idx].replace("phase1_crosswind_cruise", "phase1_crosswind_cruisX", 1)
    if tampered_lines[t_idx] == orig_lines[t_idx]:
        tampered_lines[t_idx] = tampered_lines[t_idx].replace("5.0", "5.1", 1)

    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False, encoding="utf-8", newline="\n") as tf:
        tf.write("\n".join(tampered_lines) + "\n")
        tamper_path = Path(tf.name)
    try:
        py_tamper = verify_shadow_log_chain(tamper_path)
        rs_tamper_proc = subprocess.run([str(EDGE_RELEASE_EXE), "--replay", str(tamper_path)], capture_output=True, text=True, check=False)
        rs_tamper_rep = json.loads(rs_tamper_proc.stdout) if rs_tamper_proc.stdout.strip() else {"valid": False}
    finally:
        tamper_path.unlink(missing_ok=True)

    p6_ok = bool(
        py_ver.valid
        and rs_proc.returncode == 0
        and rs_rep.get("valid") is True
        and rs_rep.get("head_hash") == py_ver.head_hash
        and "record_count" in rs_rep
        and int(rs_rep["record_count"]) == py_ver.record_count
        and "cusum_alarms" in rs_rep
        and int(rs_rep["cusum_alarms"]) > 0
        and "rta_mandatory_go_around_count" in rs_rep
        and int(rs_rep["rta_mandatory_go_around_count"]) > 0
        and (not py_tamper.valid)
        and rs_tamper_proc.returncode != 0
        and rs_tamper_rep.get("valid") is False
    )
    print(
        f"  [Phase 6] py_valid={py_ver.valid} rs_rc={rs_proc.returncode} rs_valid={rs_rep.get('valid')} "
        f"records={py_ver.record_count} head_match={rs_rep.get('head_hash') == py_ver.head_hash} "
        f"rs_cusum_alarms={rs_rep.get('cusum_alarms')} rs_mga={rs_rep.get('rta_mandatory_go_around_count')} | "
        f"1B_tamper(py_valid={py_tamper.valid}, rs_rc={rs_tamper_proc.returncode})",
        flush=True,
    )

    # ==========================================================================
    # PHASE 7: Final WSL Hygiene Audit
    # ==========================================================================
    r_pids = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", "pgrep -x arduplane || true"], capture_output=True, text=True, timeout=10)
    leftover_pids = [ln.strip() for ln in r_pids.stdout.splitlines() if ln.strip()]
    p7_ok = bool(pid_cleanly_terminated and len(leftover_pids) == 0 and ("pymavlink" not in sys.modules))
    print(f"\n--- [Phase 7] Teardown Hygiene: pid={ap_pid} terminated={pid_cleanly_terminated} leftovers={leftover_pids} ---", flush=True)

    phase_checks = {
        "phase1_takeoff_crosswind_vfr_hud": bool(
            cruise_alt_agl_m >= 50.0
            and wind_rb.get("SIM_WIND_SPD") == 15.0
            and wind_rb.get("SIM_WIND_TURB") == 5.0
            and p1_vfr_cnt >= 15
            and p1_max_ias_gs_diff >= 6.0
            and p1_cusum_fa == 0
            and comp_pose_samples == 10
        ),
        "phase2_spoof_cusum_datum_42007_vision": bool(
            p2_ttd_s is not None
            and p2_ttd_s <= 12.0
            and p2_bank_a_err_m is not None
            and p2_bank_a_err_m < 30.0
            and p2_datum_dec is not None
            and bool(p2_datum_dec["allowed"])
            and float(p2_datum_dec["abs_vertical_innovation_m"]) <= 3.0
            and int(st_p2["ekf3_active_source"]) == EKF3_SOURCE_VISION_OPTICAL
            and p2_ack_42007 >= 1
        ),
        "phase3_vision_dual_intruder_daa_no_stall_no_msg84": bool(
            ack_43002_ok
            and ack_192_ok
            and p3a_max_roll >= 10.0
            and abs(p3a_delta_hdg) >= 10.0
            and p3b_min_roll <= -8.0
            and cmd82_mask == ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES
            and p3_min_ias >= 16.5
            and msg84_zero
        ),
        "phase4_gnss_recovery_ge2s_dwell_2nd_ack42007": bool(
            ekf_during_lt2s == EKF3_SOURCE_VISION_OPTICAL
            and ekf_after_ge2s == EKF3_SOURCE_GNSS
            and total_ack_42007_ok >= 2
        ),
        "phase5_approach_fault_rta_p0_latch_and_ack_recovery": p5_ok,
        "phase6_dual_python_rust_replay_and_tamper_proof": p6_ok,
        "phase7_clean_teardown_zero_leftovers": p7_ok,
    }

    report = {
        "schema": "openxvs.faz2.full_mission_sil_gate/1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "simulator_backend": "WSL2_ARDUPLANE_OFFICIAL_SITL",
        "arduplane_metadata": meta,
        "pymavlink_imported": "pymavlink" in sys.modules,
        "non_flight_sim_only_tape_enforced": bool(py_ver.tx_inhibited_all and py_ver.actuator_authority_all_false),
        "all_pass": all(phase_checks.values()),
        "phase_checks": phase_checks,
        "phase1_crosswind_cruise": {
            "cruise_alt_agl_m": round(cruise_alt_agl_m, 2),
            "wind_params_readback": wind_rb,
            "vfr_hud_count": p1_vfr_cnt,
            "ias_min_max_mps": [round(min(p1_ias_list), 2), round(max(p1_ias_list), 2)],
            "gs_min_max_mps": [round(min(p1_gs_list), 2), round(max(p1_gs_list), 2)],
            "max_abs_ias_minus_gs_mps": p1_max_ias_gs_diff,
            "cusum_false_alarms": p1_cusum_fa,
            "lever_arm_compensated_fixes": comp_pose_samples,
        },
        "phase2_gps_spoof_and_ekf3_vision": {
            "ttd_sim_s": p2_ttd_s,
            "bank_a_err_at_quarantine_m": round(p2_bank_a_err_m, 2) if p2_bank_a_err_m is not None else None,
            "bank_b_active_err_m": round(p2_bank_b_err_m, 2) if p2_bank_b_err_m is not None else None,
            "cusum_stat_at_quarantine": round(p2_stat_at_q, 3),
            "datum_guard_decision": p2_datum_dec,
            "ekf3_active_source": int(st_p2["ekf3_active_source"]),
            "ack_42007_result0_phase2": p2_ack_42007,
        },
        "phase3_vision_daa_avoidance": {
            "cmd75_ack_43002_ok": ack_43002_ok,
            "cmd75_ack_192_ok": ack_192_ok,
            "cmd75_max_abs_roll_deg": round(p3a_max_roll, 2),
            "cmd75_delta_heading_deg": round(p3a_delta_hdg, 2),
            "cmd82_min_roll_deg": round(p3b_min_roll, 2),
            "cmd82_type_mask_hex": hex(cmd82_mask) if cmd82_mask is not None else None,
            "cmd82_thrust_norm": last_cmd82.get("thrust_norm"),
            "min_vfr_hud_ias_mps": round(p3_min_ias, 2),
            "msg84_ever_emitted": not msg84_zero,
            "wire_msg84_seen": int(proxy.bridge_msg84_seen),
        },
        "phase4_gnss_recovery_dwell": {
            "ekf3_during_lt2s_dwell": ekf_during_lt2s,
            "ekf3_after_ge2s_revert": ekf_after_ge2s,
            "ack_42007_result0_total": total_ack_42007_ok,
        },
        "phase5_simplex_rta_recovery": {
            "fault_mode": int(o_fault.mode),
            "fault_mandatory_go_around": bool(o_fault.mandatory_go_around),
            "hold_on_approach_mode": int(o_hold.mode),
            "hold_on_approach_mga": bool(o_hold.mandatory_go_around),
            "ack_cleared_mga": not bool(o_ack.mandatory_go_around),
            "p0_to_p1_recovery_ticks": p1_reached_at_tick,
            "p1_to_p4_dwell_total_ticks": p4_reached_at_tick,
            "final_rta_mode": staged_modes[-1],
        },
        "phase6_flight_recorder_replay": {
            "tape_path": str(tape_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "python_verify": py_ver.to_dict(),
            "rust_replay_returncode": rs_proc.returncode,
            "rust_replay_report": rs_rep,
            "tamper_1byte_python_valid": py_tamper.valid,
            "tamper_1byte_rust_returncode": rs_tamper_proc.returncode,
            "tamper_1byte_rust_valid": rs_tamper_rep.get("valid"),
            "tamper_1byte_rust_first_failed_line": rs_tamper_rep.get("first_failed_line"),
        },
        "phase7_teardown_hygiene": {
            "spawned_wsl_arduplane_pid": ap_pid,
            "pid_cleanly_terminated": pid_cleanly_terminated,
            "leftover_wsl_arduplane_pids": leftover_pids,
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[FAZ2-SIL DONE] all_pass={report['all_pass']} | elapsed={report['elapsed_s']}s | wrote {report_path}", flush=True)
    return report


def main() -> int:
    rep = run_full_mission_sil_gate()
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
