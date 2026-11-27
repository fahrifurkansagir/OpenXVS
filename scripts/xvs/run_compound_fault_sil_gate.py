"""Wave 1 Compound-Fault Simultaneous Stress SIL Gate on LIVE WSL2 `ArduPlane V4.7.3`
+ Pure-Rust `openxvs-edge-infer.exe --replay` Flight-Recorder Verification.

Unlike the sequential 7-phase gate (`run_full_mission_sil_gate.py`), this gate injects
SIMULTANEOUS compound faults inside ONE continuous live WSL2 `ArduPlane V4.7.3` flight
(`WSL2_ARDUPLANE_OFFICIAL_SITL`, zero twin fallback, `pymavlink_imported = false`,
strictly NED, zero `#84`):

  Phase 0: KSFO 28L Takeoff -> GUIDED Cruise -> Engage live 15 m/s Crosswind
           (`SIM_WIND_SPD=15`, `SIM_WIND_TURB=5`, `SIM_WIND_DIR=208`) + nominal baseline
           (`0` CUSUM false alarms, crisp EO+LWIR `q_lwir > 0.7`, `SimplexRta` P4).
  Phase 1 (SIMULTANEOUS COMPOUND-FAULT STRESS WINDOW):
      1. Active `15 m/s` crosswind + LIVE `SIM_GPS1_GLTCH_X/Y` spoofing ramp
         (`2.0 m/s @ 22.5 deg`) -> INS-coupled `DualBankEsEkfCusum` + `BaroThermalDriftGuard`
         -> `DatumAlignmentGuard` (KSFO GEOID18) -> `#76(42007, 2)` -> `COMMAND_ACK(42007, 0)`.
      2. SIMULTANEOUS `90 deg` visual heading perturbation recovered LIVE via
         `coarse_heading.py` (`COLD_START_4X90_HYPOTHESES_DEG` rescue vs failed naive `0 deg`,
         center reproj `<= 4.0 px`, `0 HMI` distractor rejection) -> admits visual Bank-B fix.
      3. SIMULTANEOUS injected LWIR shutter-freeze fault (`MAD < 0.25` for `>= 3` ticks) via
         `SensorHealthMonitor` -> `lwir_nuc_frozen=True`, `q_lwir_effective == 0.00` fail-closed,
         `LWIR_DEGRADED_EO_PRIMARY` (and verified clean recovery on unfreeze).
      4. SIMULTANEOUS DAA intruder encounter (`#75 COMMAND_INT` 43002/192 + `#82 SET_ATTITUDE_TARGET`
         `type_mask=0x07`) WHILE crosswind + active GPS spoof (`EKF3=2`) + LWIR shutter freeze
         + `90 deg` heading recovery are ALL active -> real roll/heading response, stall-free
         `#74 VFR_HUD` IAS `>= 17.0 m/s`, `0` `#84` forbidden frames!
  Phase 2: Compound Recovery (`SIM_GPS1_GLTCH=0` + `>= 2.0 s` EKF3 dwell hysteresis ->
           2nd `COMMAND_ACK(42007, 0)`, LWIR unfreeze recovery, and `SimplexRta` P0 approach
           integrity latch + pilot `go_around_ack_reset` staged `P0 -> P1 -> P4` recovery).
  Phase 3: Dual Python `verify_shadow_log_chain` + Pure-Rust `openxvs-edge-infer.exe --replay`
           on SHA-256 hash-chained `outputs/sitl_proof/v112_compound_fault/compound_mission_tape.ndjson`
           + 1-byte tamper proof (`rc=1`, `valid=False`).
  Phase 4: Exact-PID Clean Teardown (`0` leftover WSL `arduplane` PIDs).

Usage:
    $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; .venv/Scripts/python.exe scripts/xvs/run_compound_fault_sil_gate.py
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
from typing import Any, Sequence

# Mandatory thermal lock
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

cv2.setNumThreads(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.core_ffi import XvsCoreFFI, XvsRtaFloatTelemetry  # noqa: E402
from xvs.v2.sensor_health import SensorHealthMonitor  # noqa: E402
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
    MSGID_GLOBAL_POSITION_INT,
    MSGID_HEARTBEAT,
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
from xvs.v2.xvs_nav.coarse_heading import (  # noqa: E402
    COLD_START_4X90_HYPOTHESES_DEG,
    match_with_coarse_heading_sweep,
)
from xvs.v2.xvs_nav.stage2_xfeat import XFeatOnnxRunner  # noqa: E402
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

OUT_DIR = REPO_ROOT / "outputs" / "sitl_proof" / "v112_compound_fault"
TAPE_FILENAME = "compound_mission_tape.ndjson"
REPORT_FILENAME = "compound_fault_sil_report.json"
EDGE_RELEASE_EXE = REPO_ROOT / "docker" / "edge-infer" / "target" / "release" / "openxvs-edge-infer.exe"
XFEAT_FP16_ONNX = REPO_ROOT / "outputs" / "colab_l4_v5_nav" / "harvest_faz_b" / "xfeat_320x256_fp16.onnx"
WSL_SITL_CACHE_DIR = "/var/tmp/openxvs_sitl"
MIN_ARDUPLANE_VERSION: tuple[int, int, int] = (4, 7, 3)

KSFO_HOME_LAT = 37.61172
KSFO_HOME_LON = -122.358367
FORBIDDEN_PORTS: frozenset[int] = frozenset({4173, 4174, 8765, 9222, 9223})
MIN_STALL_FREE_IAS_MPS: float = 17.0

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
    hdr = struct.pack(
        "<BBBBBBBBB",
        len(wp),
        0,
        0,
        seq & 0xFF,
        255,
        191,
        msgid & 0xFF,
        (msgid >> 8) & 0xFF,
        (msgid >> 16) & 0xFF,
    )
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
    out = subprocess.check_output(
        ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", bash], timeout=20
    ).decode("utf-8", "ignore")
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


def _build_aerial_reference_scene(seed: int = 20261127) -> np.ndarray:
    """Create a rich deterministic 384x384 asymmetric aerial orthomosaic scene for 90 deg coarse heading test."""
    rng = np.random.default_rng(seed)
    scene = np.full((384, 384, 3), 108, dtype=np.uint8)
    for _ in range(95):
        x0, y0 = int(rng.integers(18, 342)), int(rng.integers(18, 342))
        bw, bh = int(rng.integers(8, 28)), int(rng.integers(8, 28))
        col = tuple(int(c) for c in rng.integers(25, 245, size=3))
        cv2.rectangle(scene, (x0, y0), (x0 + bw, y0 + bh), col, -1)
        cv2.rectangle(scene, (x0, y0), (x0 + bw, y0 + bh), (18, 18, 18), 1)
    cv2.line(scene, (10, 178), (374, 198), (235, 235, 225), 5)
    cv2.line(scene, (188, 10), (212, 374), (220, 226, 238), 4)
    cv2.circle(scene, (245, 130), 22, (48, 72, 110), -1)
    cv2.rectangle(scene, (85, 235), (145, 275), (210, 195, 175), -1)
    return scene


def _make_eo_scene(seed: int = 101, w: int = 320, h: int = 240) -> np.ndarray:
    """Deterministic crisp daytime EO frame (q_eo > 0.75) for paired EO+LWIR fault testing."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        if y < h // 2:
            img[y, :, :] = (195 - y // 3, 175 - y // 4, 150 - y // 4)
        else:
            img[y, :, :] = (65 + (y % 17), 90 + (y % 13), 75 + (y % 11))
    cv2.rectangle(img, (90, 125), (230, 225), (58, 62, 68), -1)
    for x in range(100, 225, 16):
        cv2.line(img, (x, 128), (x, 222), (225, 230, 235), 2, cv2.LINE_AA)
    for _ in range(30):
        px = int(rng.integers(15, w - 15))
        py = int(rng.integers(15, h - 15))
        rad = int(rng.integers(3, 10))
        col = tuple(int(c) for c in rng.integers(30, 220, size=3))
        cv2.circle(img, (px, py), rad, col, 2, cv2.LINE_AA)
    return img


def _make_lwir_scene(seed: int = 202, shift_x: int = 0, w: int = 160, h: int = 128) -> np.ndarray:
    """Deterministic 2D uint8 microbolometer LWIR frame with moving warm structures."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 78, dtype=np.uint8)
    noise = rng.integers(-6, 7, size=(h, w))
    frame = np.clip(base.astype(np.int16) + noise, 20, 220).astype(np.uint8)
    sx = int(shift_x) % 24
    cv2.rectangle(frame, (18 + sx, 28), (58 + sx, 78), 178, -1)
    cv2.circle(frame, (104 + sx, 54), 12, 228, -1, cv2.LINE_AA)
    cv2.line(frame, (10, 102), (150, 102), 148, 2)
    return frame


class _CompoundWireProxy:
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
        pl = _PARAM_SET_STRUCT.pack(
            float(val), 1, 1, name.encode("ascii")[:16].ljust(16, b"\x00"), MAV_PARAM_TYPE_REAL32
        )
        self.send_oob(_encode_raw_mav2(MSGID_PARAM_SET, pl, CRC_EXTRA_PARAM_SET, seq=self._next_oob_seq()))

    def send_oob_param_read(self, name: str) -> None:
        pl = _PARAM_REQUEST_READ_STRUCT.pack(-1, 1, 1, name.encode("ascii")[:16].ljust(16, b"\x00"))
        self.send_oob(
            _encode_raw_mav2(MSGID_PARAM_REQUEST_READ, pl, CRC_EXTRA_PARAM_REQUEST_READ, seq=self._next_oob_seq())
        )

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
        if self._s_wsl is None:
            raise RuntimeError("WSL proxy RX socket not initialized")
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
        if self._s_br is None:
            raise RuntimeError("Bridge proxy RX socket not initialized")
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
                                self.bridge_tx_msgids.get(mid, 0) + 1  # default-ok: per-msgid packet counter starts at 0
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

    def snapshot_latest(
        self,
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        with self._lock:
            return (
                dict(self.latest_hb) if self.latest_hb is not None else None,
                dict(self.latest_att) if self.latest_att is not None else None,
                dict(self.latest_gpi) if self.latest_gpi is not None else None,
                dict(self.latest_gps_raw) if self.latest_gps_raw is not None else None,
            )

    def get_params_snapshot(self, keys: Sequence[str]) -> dict[str, float | None]:
        with self._lock:
            return {k: self.params.get(k) for k in keys}

    def get_acks_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(a) for a in self.acks]

    def get_bridge_msg84_seen(self) -> int:
        with self._lock:
            return int(self.bridge_msg84_seen)


def _make_rta_float_telemetry(
    *,
    v_vpl_m: float = 5.0,
    v_hpl_m: float = 14.0,
    approach_active: bool = False,
    go_around_ack_reset: bool = False,
    cusum_stat: float = 0.0,
    cam_health_mask: int = 0x3F,
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
        cam_health_mask=int(cam_health_mask) & 0x3F,
        ml_detector_fault=1 if ml_detector_fault else 0,
        transient_dropout=0,
        approach_active=1 if approach_active else 0,
        go_around_ack_reset=1 if go_around_ack_reset else 0,
    )


def run_compound_fault_sil_gate(out_dir: Path | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    if "pymavlink" in sys.modules:
        raise RuntimeError("GPL pymavlink MUST NEVER be imported!")
    target_dir = OUT_DIR if out_dir is None else Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tape_path = target_dir / TAPE_FILENAME
    report_path = target_dir / REPORT_FILENAME

    meta = _verify_wsl_arduplane()
    print(
        f"[COMPOUND-SIL] Verified LIVE WSL2 {meta['apm_version_line']} "
        f"(sha256={meta['binary_sha256'][:16]}...)",
        flush=True,
    )

    # Pre-initialize XFeat runner (real ONNX if present, else deterministic offline fallback)
    # and rich aerial reference + 90-deg rotated query + EO/LWIR SensorHealthMonitor.
    xfeat_runner = XFeatOnnxRunner(
        XFEAT_FP16_ONNX if XFEAT_FP16_ONNX.is_file() else None,
        max_keypoints=256,
        ort_threads=1,
    )
    aerial_scene = _build_aerial_reference_scene(seed=20261127)
    map_crop = aerial_scene[64:320, 64:320].copy()
    km_map, dm_map, _ = xfeat_runner.extract(map_crop)

    m_rot90 = cv2.getRotationMatrix2D((192.0, 192.0), 90.0, 1.0)
    q_rot90 = cv2.warpAffine(aerial_scene, m_rot90, (384, 384), flags=cv2.INTER_LINEAR)[64:320, 64:320].copy()

    rng_dist = np.random.default_rng(20261128)
    distractor_tile = rng_dist.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    kd_dist, dd_dist, _ = xfeat_runner.extract(distractor_tile)

    eo_frame = _make_eo_scene(seed=101)
    lwir_frozen_frame = _make_lwir_scene(seed=303, shift_x=0)
    sensor_monitor = SensorHealthMonitor(
        lwir_mad_threshold=0.25,
        lwir_max_freeze_ticks=3,
        lwir_max_freeze_duration_ms=250.0,
        lwir_recovery_ticks=1,
    )

    run_dir_wsl = f"{WSL_SITL_CACHE_DIR}/run_v112_compound_{int(time.time() * 1000) % 100000}"
    parm_file_wsl = f"{run_dir_wsl}/compound.parm"
    prep = (
        f"set -e; mkdir -p {run_dir_wsl}; cat {WSL_SITL_CACHE_DIR}/plane.parm > {parm_file_wsl}; "
        f"cat <<'EOF' >> {parm_file_wsl}\n{OVERLAY_PARM.strip()}\nEOF"
    )
    subprocess.check_call(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", prep], timeout=15)

    bridge_rx_port = _alloc_udp_port("127.0.0.1")
    proxy = _CompoundWireProxy(wsl_ip=str(meta["wsl_ip"]), bridge_rx_port=bridge_rx_port)
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

    def wait_next_1hz_epoch(
        last_boot_ms: int, timeout_s: float = 2.5
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            proxy.send_oob_hb()
            _, _, g, gps = proxy.snapshot_latest()
            if g is not None and (int(g["time_boot_ms"]) - last_boot_ms) >= 1000:
                return g, gps
            time.sleep(0.025)
        _, _, g_last, gps_last = proxy.snapshot_latest()
        if g_last is None:
            raise RuntimeError("Timed out waiting for GLOBAL_POSITION_INT telemetry from ArduPlane")
        return g_last, gps_last

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
        if proc.stdout is None:
            raise RuntimeError("Failed to capture stdout pipe from WSL2 ArduPlane launcher")
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

        # Handshake + streams (#30, #33, #74, #24)
        t_boot = time.monotonic() + 10.0
        while time.monotonic() < t_boot:
            proxy.send_oob_hb()
            st_b = mgr.status()
            hb_b, att_b, gpi_b, _ = proxy.snapshot_latest()
            if hb_b and att_b and gpi_b and int(st_b["attitude_count"]) >= 2:
                break
            time.sleep(0.1)
        hb_boot, _, _, _ = proxy.snapshot_latest()
        if hb_boot is None:
            raise RuntimeError("No HEARTBEAT from live WSL2 ArduPlane")
        for mid, iv_us in ((30, 50_000.0), (33, 50_000.0), (74, 50_000.0), (24, 100_000.0)):
            proxy.send_oob_cmd_long(MAV_CMD_SET_MESSAGE_INTERVAL, p1=float(mid), p2=iv_us)
        sleep_hb(0.4)

        # ======================================================================
        # PHASE 0: KSFO 28L Takeoff -> GUIDED Cruise + Live 15 m/s Crosswind
        # ======================================================================
        print("\n--- [Phase 0] KSFO 28L Takeoff -> GUIDED Cruise + 15 m/s Crosswind Baseline ---", flush=True)
        proxy.send_oob_cmd_long(MAV_CMD_DO_SET_MODE, p1=1.0, p2=float(PLANE_MODE_TAKEOFF))
        sleep_hb(0.3)
        proxy.send_oob_cmd_long(MAV_CMD_COMPONENT_ARM_DISARM, p1=1.0, p2=21196.0)
        t_tk0 = time.monotonic()
        while (time.monotonic() - t_tk0) < 18.0:
            proxy.send_oob_hb()
            hb_tk, _, gpi_tk, _ = proxy.snapshot_latest()
            if gpi_tk and float(gpi_tk["relative_alt_m"]) >= 60.0:
                break
            if (time.monotonic() - t_tk0) > 1.5 and hb_tk and not hb_tk["armed"]:
                proxy.send_oob_cmd_long(MAV_CMD_DO_SET_MODE, p1=1.0, p2=float(PLANE_MODE_TAKEOFF))
                proxy.send_oob_cmd_long(MAV_CMD_COMPONENT_ARM_DISARM, p1=1.0, p2=21196.0)
            time.sleep(0.1)
        proxy.send_oob_cmd_long(MAV_CMD_DO_SET_MODE, p1=1.0, p2=float(PLANE_MODE_GUIDED))
        sleep_hb(1.0)
        _, _, gpi_cr, _ = proxy.snapshot_latest()
        cruise_alt_agl_m = float(gpi_cr["relative_alt_m"]) if gpi_cr else 0.0

        # Engage live 15 m/s crosswind + turbulence
        proxy.send_oob_param_set("SIM_WIND_SPD", 15.0)
        proxy.send_oob_param_set("SIM_WIND_TURB", 5.0)
        proxy.send_oob_param_set("SIM_WIND_DIR", 208.0)
        sleep_hb(0.4)
        for pn in ("SIM_WIND_SPD", "SIM_WIND_TURB", "SIM_WIND_DIR"):
            proxy.send_oob_param_read(pn)
        sleep_hb(0.5)
        wind_rb = proxy.get_params_snapshot(("SIM_WIND_SPD", "SIM_WIND_TURB", "SIM_WIND_DIR"))

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

        _, _, g0, _ = proxy.snapshot_latest()
        if g0 is None:
            raise RuntimeError("Missing GLOBAL_POSITION_INT prior to baseline cruise")
        origin_n0, origin_e0 = _latlon_to_ned_m(g0["lat_deg"], g0["lon_deg"], KSFO_HOME_LAT, KSFO_HOME_LON)
        prev_ms = int(g0["time_boot_ms"])
        prev_vn = float(g0["vn_mps"])
        prev_ve = float(g0["ve_mps"])

        cusum = DualBankEsEkfCusum()
        cusum.reset(init_ned_m=(0.0, 0.0, float(g0["down_m"])), init_vel_ne_mps=(prev_vn, prev_ve))
        baro_guard = BaroThermalDriftGuard()
        rng = np.random.default_rng(20261127)

        base_ias_list: list[float] = []
        base_diff_list: list[float] = []
        base_cusum_fa = 0
        last_sp, last_va, last_vs = False, False, 0.0
        base_thermal_reports: list[dict[str, Any]] = []
        sensor_ts_ms = 10_000.0

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
            base_ias_list.append(ias)
            base_diff_list.append(abs(ias - gs))

            # Healthy moving LWIR + crisp EO baseline
            sensor_ts_ms += 100.0
            lwir_live = _make_lwir_scene(seed=202 + ep, shift_x=ep * 4)
            th_rep = sensor_monitor.assess_thermal_pair(
                eo_frame, lwir_live, frame_ts_ms=sensor_ts_ms, imu_ts_ms=sensor_ts_ms - 5.0
            )
            base_thermal_reports.append(th_rep)

            src_lat = gps_cur["lat_deg"] if gps_cur else g_cur["lat_deg"]
            src_lon = gps_cur["lon_deg"] if gps_cur else g_cur["lon_deg"]
            gn_abs, ge_abs = _latlon_to_ned_m(src_lat, src_lon, KSFO_HOME_LAT, KSFO_HOME_LON)
            rel_ne = np.array([gn_abs - origin_n0, ge_abs - origin_e0], dtype=np.float64)

            comp_fix = mgr.compensate_visual_fix_ned([float(rel_ne[0]), float(rel_ne[1]), float(g_cur["down_m"])])
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
                base_cusum_fa += 1

            rta_out = ffi.rta_step(
                rta_handle, ffi.rta_build_input_from_f64(_make_rta_float_telemetry(cusum_stat=float(c_out.max_cusum_stat)))
            )
            logger.record_event(
                timestamp_mono_us=next_tape_us(),
                event_type="compound_baseline_epoch",
                direction="rx",
                payload={
                    "phase": "phase0_crosswind_baseline",
                    "gnss_ne_m": [round(float(gnss_meas[0]), 4), round(float(gnss_meas[1]), 4)],
                    "vis_ne_m": [round(float(vis_meas[0]), 4), round(float(vis_meas[1]), 4)],
                    "baro_down_m": round(float(baro_corr.corrected_baro_down_m), 4),
                    "gnss_down_m": round(float(g_cur["down_m"]), 4),
                    "ins_accel_ne_mps2": [round(an, 4), round(ae, 4)],
                    "vis_gate_admitted": True,
                    "airspeed_mps": round(ias, 3),
                    "groundspeed_mps": round(gs, 3),
                    "q_lwir_effective": float(th_rep["q_lwir_effective"]),
                    "lwir_nuc_frozen": bool(th_rep["lwir_nuc_frozen"]),
                    "thermal_mode": str(th_rep["thermal_mode"]),
                    "ekf3_source": int(st_now["ekf3_active_source"]),
                    "lever_arm_compensated_ned_m": [round(x, 3) for x in comp_fix.p_cg_ned_m],
                    "rta_mode": int(rta_out.mode),
                    "approach_active": False,
                    "v_vpl_m": 5.0,
                },
            )

        base_max_ias_gs_diff = round(max(base_diff_list), 2) if base_diff_list else 0.0
        base_q_lwir_min = min(float(r["q_lwir_effective"]) for r in base_thermal_reports)
        print(
            f"  [Phase 0] AGL={cruise_alt_agl_m:.1f}m | wind={wind_rb} | max|IAS-GS|={base_max_ias_gs_diff} m/s | "
            f"CUSUM_FA={base_cusum_fa} | base_q_lwir_min={base_q_lwir_min:.3f}",
            flush=True,
        )

        # ======================================================================
        # PHASE 1: SIMULTANEOUS COMPOUND-FAULT INJECTION WINDOW
        #   Stressor 1: 15 m/s crosswind + live SIM_GPS1_GLTCH spoof ramp
        #   Stressor 2: SIMULTANEOUS 90-deg visual heading perturbation -> coarse_heading.py rescue
        #   Stressor 3: SIMULTANEOUS LWIR shutter-freeze -> SensorHealthMonitor q_lwir == 0.00
        #   Stressor 4: SIMULTANEOUS DAA intruder encounter (#75/#82, IAS >= 17 m/s, 0 #84)
        # ======================================================================
        print(
            "\n--- [Phase 1] SIMULTANEOUS Compound-Fault Window "
            "(15 m/s Wind + GPS Spoof + 90 deg Heading + LWIR Freeze + DAA #75/#82) ---",
            flush=True,
        )
        ramp_rate_mps = 2.0
        theta = math.radians(22.5)
        u_n, u_e = math.cos(theta), math.sin(theta)
        cos_lat = max(0.1, math.cos(math.radians(KSFO_HOME_LAT)))

        # Verify naive 0-deg fails under 90-deg heading perturbation, while coarse_heading.py rescues it!
        naive_0deg = match_with_coarse_heading_sweep(
            q_rot90, km_map, dm_map, xfeat_runner=xfeat_runner, angles_deg=(0.0,), img_hw_m=(256, 256)
        )
        distractor_res = match_with_coarse_heading_sweep(
            map_crop, kd_dist, dd_dist, xfeat_runner=xfeat_runner, angles_deg=COLD_START_4X90_HYPOTHESES_DEG, img_hw_m=(256, 256)
        )

        acks_42007_pre = sum(
            1
            for a in proxy.get_acks_snapshot()
            if a["command"] == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET and a["result"] == 0
        )

        spoof_ttd_s: float | None = None
        bank_a_err_m: float | None = None
        bank_b_err_m: float | None = None
        cusum_stat_at_q: float = 0.0
        heading_rescue_inliers: int = 0
        heading_rescue_angle_deg: float = 0.0
        heading_rescue_center_err_px: float = 999.0
        heading_rescue_epochs: int = 0
        lwir_freeze_triggered_step: int | None = None
        lwir_freeze_q_min: float = 1.0
        lwir_freeze_modes: list[str] = []
        simultaneous_4fault_epoch_observed: bool = False
        quarantine_step_k: int | None = None
        datum_dec_at_switch: dict[str, Any] | None = None
        daa_triggered_during_compound: bool = False
        t_daa_start: float | None = None
        t_daa_cmd82_start: float | None = None
        base_hdg_daa: float = float(mgr.status()["state_ned"]["heading_deg"])

        # Seed the frozen LWIR frame right at compound onset so freeze trips simultaneously with spoof ramp
        sensor_ts_ms += 100.0
        sensor_monitor.assess_thermal_pair(
            eo_frame, lwir_frozen_frame, frame_ts_ms=sensor_ts_ms, imu_ts_ms=sensor_ts_ms - 4.0
        )

        for step_k in range(1, 30):
            # 1) Active GPS spoof ramp under active 15 m/s crosswind
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

            if gps_cur is None:
                raise RuntimeError("Missing GPS_RAW_INT (#24) during compound-fault spoof ramp")
            gn_gl, ge_gl = _latlon_to_ned_m(gps_cur["lat_deg"], gps_cur["lon_deg"], KSFO_HOME_LAT, KSFO_HOME_LON)
            rel_gnss = np.array([gn_gl - origin_n0, ge_gl - origin_e0], dtype=np.float64)
            rel_true = rel_gnss - np.array([off_n, off_e], dtype=np.float64)

            # 2) SIMULTANEOUS 90-deg visual heading perturbation -> live coarse_heading.py rescue
            ch_res = match_with_coarse_heading_sweep(
                q_rot90,
                km_map,
                dm_map,
                xfeat_runner=xfeat_runner,
                angles_deg=COLD_START_4X90_HYPOTHESES_DEG,
                img_hw_m=(256, 256),
            )
            center_err_px = 999.0
            if ch_res.affine_orig_to_map is not None:
                c_m = ch_res.affine_orig_to_map @ np.array([127.5, 127.5, 1.0], dtype=np.float64)
                center_err_px = float(math.hypot(c_m[0] - 127.5, c_m[1] - 127.5))
            heading_recovered = bool(ch_res.n_geom_inliers >= 12 and center_err_px <= 4.0)
            if heading_recovered:
                heading_rescue_epochs += 1
                heading_rescue_inliers = max(heading_rescue_inliers, int(ch_res.n_geom_inliers))
                heading_rescue_angle_deg = float(ch_res.best_angle_deg)
                heading_rescue_center_err_px = min(heading_rescue_center_err_px, center_err_px)

            # 3) SIMULTANEOUS LWIR shutter-freeze injection (identical frozen frame every epoch -> MAD=0.0)
            sensor_ts_ms += 100.0
            th_cur = sensor_monitor.assess_thermal_pair(
                eo_frame,
                lwir_frozen_frame.copy(),
                frame_ts_ms=sensor_ts_ms,
                imu_ts_ms=sensor_ts_ms - 4.0,
            )
            q_lwir_now = float(th_cur["q_lwir_effective"])
            lwir_frozen_now = bool(th_cur["lwir_nuc_frozen"])
            lwir_freeze_q_min = min(lwir_freeze_q_min, q_lwir_now)
            if lwir_frozen_now:
                if lwir_freeze_triggered_step is None:
                    lwir_freeze_triggered_step = step_k
                lwir_freeze_modes.append(str(th_cur["thermal_mode"]))

            # Stream live visual altitude whenever coarse heading recovers so DatumAlignmentGuard tracks climb
            if heading_recovered:
                cur_msl_live = float(mgr.status()["state_ned"]["alt_msl_m"])
                mgr.update_visual_altitude(
                    visual_alt_m=cur_msl_live + KSFO_28L_GEOID18_UNDULATION_M,
                    visual_datum="WGS84_HAE",
                    geoid_undulation_m=KSFO_28L_GEOID18_UNDULATION_M,
                    apply_geoid_conversion=True,
                )

            # Visual fix admitted ONLY when coarse_heading.py recovers the 90-deg perturbation!
            vis_meas = rel_true + rng.normal(0.0, 2.2, size=2)
            baro_corr = baro_guard.step(
                baro_down_raw_m=float(g_cur["down_m"]),
                gnss_down_m=float(g_cur["down_m"]),
                dt_s=1.0,
                gnss_healthy=not bool(last_sp),
                spoof_alarm=last_sp,
                vertical_spoof_alarm=last_va,
                vertical_cusum_stat=last_vs,
            )
            c_out = cusum.step(
                ins_accel_ne_mps2=(an, ae),
                baro_down_m=baro_corr.corrected_baro_down_m,
                gnss_ne_m=rel_gnss,
                vis_ne_m=vis_meas,
                vis_gate_admitted=heading_recovered,
                gnss_down_m=float(g_cur["down_m"]),
            )
            last_sp, last_va, last_vs = c_out.spoof_alarm, c_out.vertical_spoof_alarm, c_out.vertical_cusum_stat

            # When CUSUM quarantines GNSS, switch EKF3 to Vision (42007, 2) AND immediately
            # launch SIMULTANEOUS DAA avoidance while spoof + crosswind + LWIR freeze + 90-deg heading stay active!
            if c_out.quarantine_triggered_this_epoch and quarantine_step_k is None:
                quarantine_step_k = step_k
                spoof_ttd_s = float(step_k)
                bank_a_err_m = float(np.linalg.norm(c_out.bank_a_p_ne_m - rel_true))
                bank_b_err_m = float(np.linalg.norm(c_out.active_p_ned_m[:2] - rel_true))
                cusum_stat_at_q = float(c_out.max_cusum_stat)

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
                    if (
                        sum(
                            1
                            for a in proxy.get_acks_snapshot()
                            if a["command"] == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET and a["result"] == 0
                        )
                        > acks_42007_pre
                    ):
                        break
                    time.sleep(0.05)

                st_at_sw = mgr.status()
                datum_dec_at_switch = (
                    dict(st_at_sw["last_datum_decision"]) if st_at_sw["last_datum_decision"] else None
                )

                # Launch SIMULTANEOUS DAA Intruder 1 (#75 COMMAND_INT 43002 heading change) right now!
                base_hdg_daa = float(mgr.status()["state_ned"]["heading_deg"])
                t_daa_start = time.monotonic()
                daa_triggered_during_compound = True
                mgr.set_autonomy_triggers(
                    cbdr_threat_active=True,
                    cbdr_evade_direction=1.0,
                    tx_command_mode="command_int_heading",
                    tx_enabled=True,
                    min_tx_interval_s=0.35,
                )

            # Sequence #75 (43002 heading -> 192 reposition) and #82 (SET_ATTITUDE_TARGET) across live compound epochs
            if quarantine_step_k is not None:
                rel_post_q = step_k - quarantine_step_k
                if rel_post_q == 4:
                    mgr.set_autonomy_triggers(
                        cbdr_threat_active=True,
                        cbdr_evade_direction=1.0,
                        tx_command_mode="command_int",
                        tx_enabled=True,
                        min_tx_interval_s=0.35,
                    )
                elif rel_post_q == 6:
                    t_daa_cmd82_start = time.monotonic()
                    mgr.set_autonomy_triggers(
                        cbdr_threat_active=True,
                        cbdr_evade_direction=-1.0,
                        tx_command_mode="set_attitude_target",
                        tx_enabled=True,
                        min_tx_interval_s=0.09,
                    )

            # Bit 5 cleared in cam_health_mask represents LWIR channel masked out while EO remains healthy
            cam_mask = 0x1F if lwir_frozen_now else 0x3F
            rta_out = ffi.rta_step(
                rta_handle,
                ffi.rta_build_input_from_f64(
                    _make_rta_float_telemetry(
                        cusum_stat=float(c_out.max_cusum_stat),
                        cam_health_mask=cam_mask,
                    )
                ),
            )
            st_step = mgr.status()
            ias_now = float(st_step["state_ned"]["airspeed_mps"])
            gs_now = float(st_step["state_ned"]["groundspeed_mps"])
            base_diff_list.append(abs(ias_now - gs_now))

            if (
                c_out.gnss_quarantined
                and heading_recovered
                and lwir_frozen_now
                and q_lwir_now == 0.0
                and daa_triggered_during_compound
                and int(st_step["ekf3_active_source"]) == EKF3_SOURCE_VISION_OPTICAL
            ):
                simultaneous_4fault_epoch_observed = True

            logger.record_event(
                timestamp_mono_us=next_tape_us(),
                event_type="compound_4fault_epoch",
                direction="rx",
                payload={
                    "phase": "phase1_simultaneous_compound_faults",
                    "compound_step_s": step_k,
                    "wind_spd_mps": 15.0,
                    "spoof_offset_m": round(math.hypot(off_n, off_e), 3),
                    "gnss_ne_m": [round(float(rel_gnss[0]), 4), round(float(rel_gnss[1]), 4)],
                    "vis_ne_m": [round(float(vis_meas[0]), 4), round(float(vis_meas[1]), 4)],
                    "baro_down_m": round(float(baro_corr.corrected_baro_down_m), 4),
                    "gnss_down_m": round(float(g_cur["down_m"]), 4),
                    "ins_accel_ne_mps2": [round(an, 4), round(ae, 4)],
                    "vis_gate_admitted": heading_recovered,
                    "heading_perturbation_deg": 90.0,
                    "coarse_heading_best_deg": float(ch_res.best_angle_deg),
                    "coarse_heading_inliers": int(ch_res.n_geom_inliers),
                    "coarse_heading_center_err_px": round(center_err_px, 3),
                    "lwir_nuc_frozen": lwir_frozen_now,
                    "q_lwir_effective": q_lwir_now,
                    "thermal_mode": str(th_cur["thermal_mode"]),
                    "daa_active": bool(daa_triggered_during_compound),
                    "ekf3_source": int(st_step["ekf3_active_source"]),
                    "airspeed_mps": round(ias_now, 3),
                    "groundspeed_mps": round(gs_now, 3),
                    "cusum_quarantined": bool(c_out.gnss_quarantined),
                    "cam_health_mask": cam_mask,
                    "rta_mode": int(rta_out.mode),
                    "approach_active": False,
                    "v_vpl_m": 5.0,
                },
            )

            # Hold #82 active for 3 full live epochs after quarantine_step_k + 6 so roll/IAS metrics settle
            if quarantine_step_k is not None and (step_k - quarantine_step_k) >= 9:
                break

        # Capture DAA roll/heading/IAS telemetry over the simultaneous compound DAA window
        st_compound_end = mgr.status()
        last_cmd82 = dict(st_compound_end["last_tx_command"]) if st_compound_end["last_tx_command"] else {}
        cmd82_mask = int(last_cmd82["type_mask"]) if "type_mask" in last_cmd82 else None
        datum_dec_compound = datum_dec_at_switch if datum_dec_at_switch is not None else st_compound_end["last_datum_decision"]
        ack_42007_compound = (
            sum(
                1
                for a in proxy.get_acks_snapshot()
                if a["command"] == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET and a["result"] == 0
            )
            - acks_42007_pre
        )

        with proxy._lock:
            daa_atts = [s for s in proxy.att_samples if t_daa_start is not None and s["t"] >= t_daa_start]
            daa_cmd75_atts = [
                s
                for s in proxy.att_samples
                if t_daa_start is not None
                and s["t"] >= t_daa_start
                and (t_daa_cmd82_start is None or s["t"] < t_daa_cmd82_start)
            ]
            daa_cmd82_atts = [
                s for s in proxy.att_samples if t_daa_cmd82_start is not None and s["t"] >= t_daa_cmd82_start
            ]
            daa_vfrs = [s for s in proxy.vfr_samples if t_daa_start is not None and s["t"] >= t_daa_start]

        daa_cmd75_max_roll = max((abs(s["roll_deg"]) for s in daa_cmd75_atts), default=0.0)
        daa_cmd75_delta_hdg = (
            _angular_diff_deg(daa_cmd75_atts[-1]["yaw_deg"], base_hdg_daa) if daa_cmd75_atts else 0.0
        )
        daa_cmd82_min_roll = min((s["roll_deg"] for s in daa_cmd82_atts), default=0.0)
        daa_min_ias = min((s["airspeed_mps"] for s in daa_vfrs), default=0.0)
        daa_acks = proxy.get_acks_snapshot()
        ack_43002_ok = any(a["command"] == MAV_CMD_GUIDED_CHANGE_HEADING and a["result"] == 0 for a in daa_acks)
        ack_192_ok = any(a["command"] == MAV_CMD_DO_REPOSITION and a["result"] == 0 for a in daa_acks)
        msg84_wire_seen = proxy.get_bridge_msg84_seen()

        print(
            f"  [Phase 1] TTD={spoof_ttd_s}s | BankA={bank_a_err_m:.2f}m BankB={bank_b_err_m:.2f}m | "
            f"EKF3={st_compound_end['ekf3_active_source']} ACK42007={ack_42007_compound} | "
            f"90deg_rescue(naive={naive_0deg.n_geom_inliers} -> sweep={heading_rescue_inliers}@{heading_rescue_angle_deg}deg, "
            f"err={heading_rescue_center_err_px:.2f}px, dist={distractor_res.n_geom_inliers}) | "
            f"LWIR_freeze(step={lwir_freeze_triggered_step}, q_min={lwir_freeze_q_min:.2f}, modes={sorted(set(lwir_freeze_modes))}) | "
            f"DAA(#75_roll={daa_cmd75_max_roll:.1f}deg dHdg={daa_cmd75_delta_hdg:.1f}deg, "
            f"#82_min_roll={daa_cmd82_min_roll:.1f}deg mask={hex(cmd82_mask) if cmd82_mask is not None else 'NONE'}, "
            f"min_IAS={daa_min_ias:.2f} m/s, #84={msg84_wire_seen})",
            flush=True,
        )

        # ======================================================================
        # PHASE 2: Compound Recovery (GNSS >=2s Dwell, LWIR Unfreeze, Approach RTA Latch)
        # ======================================================================
        print("\n--- [Phase 2] Compound Recovery (GNSS Dwell, LWIR Unfreeze & Approach RTA P0 Latch) ---", flush=True)
        mgr.set_autonomy_triggers(
            cbdr_threat_active=False,
            gnss_spoof_alarm=False,
            gnss_healthy=True,
            tx_command_mode="command_int_heading",
            min_tx_interval_s=0.4,
        )
        proxy.send_oob_param_set("SIM_GPS1_GLTCH_X", 0.0)
        proxy.send_oob_param_set("SIM_GPS1_GLTCH_Y", 0.0)

        sleep_hb(0.85)
        ekf_during_lt2s = int(mgr.status()["ekf3_active_source"])
        sleep_hb(1.85)
        ekf_after_ge2s = int(mgr.status()["ekf3_active_source"])
        total_ack_42007_ok = sum(
            1
            for a in proxy.get_acks_snapshot()
            if a["command"] == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET and a["result"] == 0
        )

        # Verify LWIR unfreezes cleanly when live thermal frame motion resumes
        sensor_ts_ms += 100.0
        th_rec = sensor_monitor.assess_thermal_pair(
            eo_frame,
            _make_lwir_scene(seed=909, shift_x=18),
            frame_ts_ms=sensor_ts_ms,
            imu_ts_ms=sensor_ts_ms - 4.0,
        )

        # Verify Approach vertical integrity fault -> SimplexRta P0 Latch -> Pilot go_around_ack_reset recovery
        ffi.rta_destroy(rta_handle)
        rta_handle = ffi.rta_create()
        o_nom = ffi.rta_step(rta_handle, ffi.rta_build_input_from_f64(_make_rta_float_telemetry()))
        o_fault = ffi.rta_step(
            rta_handle,
            ffi.rta_build_input_from_f64(_make_rta_float_telemetry(v_vpl_m=15.0, approach_active=True)),
        )
        o_hold = ffi.rta_step(
            rta_handle,
            ffi.rta_build_input_from_f64(
                _make_rta_float_telemetry(v_vpl_m=5.0, approach_active=True, go_around_ack_reset=True)
            ),
        )
        o_ack = ffi.rta_step(
            rta_handle,
            ffi.rta_build_input_from_f64(
                _make_rta_float_telemetry(v_vpl_m=5.0, approach_active=False, go_around_ack_reset=True)
            ),
        )
        p1_tick: int | None = None
        p4_tick: int | None = None
        staged_modes: list[int] = [int(o_ack.mode)]
        for tk in range(2, 70):
            o_rec = ffi.rta_step(
                rta_handle,
                ffi.rta_build_input_from_f64(
                    _make_rta_float_telemetry(v_vpl_m=5.0, approach_active=False, go_around_ack_reset=False)
                ),
            )
            md = int(o_rec.mode)
            staged_modes.append(md)
            if md == 1 and p1_tick is None:
                p1_tick = tk
            if md == 4 and p4_tick is None:
                p4_tick = tk
                break

        _, _, g_end_snap, _ = proxy.snapshot_latest()
        g_end = g_end_snap if g_end_snap is not None else g0
        end_n, end_e = _latlon_to_ned_m(g_end["lat_deg"], g_end["lon_deg"], KSFO_HOME_LAT, KSFO_HOME_LON)
        end_rel = [round(end_n - origin_n0, 4), round(end_e - origin_e0, 4)]

        logger.record_event(
            timestamp_mono_us=next_tape_us(),
            event_type="compound_recovery_and_rta_latch",
            direction="internal",
            payload={
                "phase": "phase2_compound_recovery_rta_fault",
                "gnss_ne_m": end_rel,
                "vis_ne_m": end_rel,
                "baro_down_m": round(float(g_end["down_m"]), 4),
                "gnss_down_m": round(float(g_end["down_m"]), 4),
                "ekf3_during_lt2s_dwell": ekf_during_lt2s,
                "ekf3_after_ge2s_revert": ekf_after_ge2s,
                "lwir_recovered_frozen": bool(th_rec["lwir_nuc_frozen"]),
                "lwir_recovered_q": float(th_rec["q_lwir_effective"]),
                "approach_active": True,
                "v_vpl_m": 15.0,
                "rta_mode": int(o_fault.mode),
                "rta_mandatory_go_around": bool(o_fault.mandatory_go_around),
            },
        )
        logger.record_event(
            timestamp_mono_us=next_tape_us(),
            event_type="compound_rta_go_around_ack_recovery",
            direction="internal",
            payload={
                "phase": "phase2_compound_rta_recovery",
                "gnss_ne_m": end_rel,
                "vis_ne_m": end_rel,
                "baro_down_m": round(float(g_end["down_m"]), 4),
                "gnss_down_m": round(float(g_end["down_m"]), 4),
                "approach_active": False,
                "go_around_ack_reset": True,
                "v_vpl_m": 5.0,
                "p1_reached_at_tick": p1_tick,
                "p4_reached_at_tick": p4_tick,
                "final_rta_mode": staged_modes[-1],
            },
        )

        st_end = mgr.status()
        tx_msgids = {int(k): int(v) for k, v in st_end["tx_msgids_emitted"].items()}
        final_msg84_seen = proxy.get_bridge_msg84_seen()
        msg84_zero = (
            not bool(st_end["msg84_ever_emitted"])
            and MSGID_SET_POSITION_TARGET_LOCAL_NED not in tx_msgids
            and final_msg84_seen == 0
        )

        print(
            f"  [Phase 2] EKF3(<2s={ekf_during_lt2s} -> >=2s={ekf_after_ge2s}, ACK42007_total={total_ack_42007_ok}) | "
            f"LWIR_rec(frozen={th_rec['lwir_nuc_frozen']}, q={th_rec['q_lwir_effective']:.3f}, mode={th_rec['thermal_mode']}) | "
            f"RTA(P0={o_fault.mode}, hold={o_hold.mode}, P1@{p1_tick}, P4@{p4_tick})",
            flush=True,
        )

    finally:
        ffi.rta_destroy(rta_handle)
        logger.close()
        mgr.shutdown()
        proxy.stop()
        # Exact-PID Clean Teardown (ONLY our spawned ap_pid!)
        if ap_pid is not None:
            subprocess.run(
                ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"kill -TERM {ap_pid} 2>/dev/null || true"],
                check=False,
                timeout=10,
            )
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
            chk = subprocess.run(
                ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"kill -0 {ap_pid} 2>/dev/null"],
                check=False,
                timeout=10,
            )
            if chk.returncode == 0:
                subprocess.run(
                    ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"kill -KILL {ap_pid} 2>/dev/null || true"],
                    check=False,
                    timeout=10,
                )
                time.sleep(0.1)
                chk = subprocess.run(
                    ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"kill -0 {ap_pid} 2>/dev/null"],
                    check=False,
                    timeout=10,
                )
            pid_cleanly_terminated = chk.returncode != 0
        subprocess.run(
            ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", f"rm -rf {run_dir_wsl}"],
            check=False,
            timeout=10,
        )

    # ==========================================================================
    # PHASE 3: Dual Python + Pure-Rust --replay Verification + 1-Byte Tamper (rc=1)
    # ==========================================================================
    print("\n--- [Phase 3] Dual Python + Pure-Rust --replay Verification + 1-Byte Tamper ---", flush=True)
    py_ver = verify_shadow_log_chain(tape_path)
    if not EDGE_RELEASE_EXE.is_file():
        raise FileNotFoundError(f"Missing Pure-Rust release binary: {EDGE_RELEASE_EXE}")

    rs_proc = subprocess.run(
        [str(EDGE_RELEASE_EXE), "--replay", str(tape_path)], capture_output=True, text=True, check=False
    )
    rs_rep = json.loads(rs_proc.stdout) if rs_proc.stdout.strip() else {"valid": False, "stderr": rs_proc.stderr}

    orig_lines = tape_path.read_text(encoding="utf-8").splitlines()
    tampered_lines = list(orig_lines)
    t_idx = min(2, len(tampered_lines) - 1)
    tampered_lines[t_idx] = tampered_lines[t_idx].replace(
        "phase0_crosswind_baseline", "phase0_crosswind_baselinX", 1
    )
    if tampered_lines[t_idx] == orig_lines[t_idx]:
        tampered_lines[t_idx] = tampered_lines[t_idx].replace("5.0", "5.1", 1)

    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False, encoding="utf-8", newline="\n") as tf:
        tf.write("\n".join(tampered_lines) + "\n")
        tamper_path = Path(tf.name)
    try:
        py_tamper = verify_shadow_log_chain(tamper_path)
        rs_tamper_proc = subprocess.run(
            [str(EDGE_RELEASE_EXE), "--replay", str(tamper_path)], capture_output=True, text=True, check=False
        )
        rs_tamper_rep = json.loads(rs_tamper_proc.stdout) if rs_tamper_proc.stdout.strip() else {"valid": False}
    finally:
        tamper_path.unlink(missing_ok=True)

    p3_replay_ok = bool(
        py_ver.valid
        and rs_proc.returncode == 0
        and rs_rep.get("valid") is True
        and rs_rep.get("head_hash") == py_ver.head_hash
        and int(rs_rep.get("record_count", -1)) == py_ver.record_count  # default-ok: fail-closed sentinel
        and int(rs_rep.get("cusum_alarms", 0)) > 0  # default-ok: fail-closed zero fails >0 gate
        and int(rs_rep.get("rta_mandatory_go_around_count", 0)) > 0  # default-ok: fail-closed zero fails >0 gate
        and (not py_tamper.valid)
        and rs_tamper_proc.returncode == 1
        and rs_tamper_rep.get("valid") is False
    )
    print(
        f"  [Phase 3] py_valid={py_ver.valid} rs_rc={rs_proc.returncode} rs_valid={rs_rep.get('valid')} "
        f"records={py_ver.record_count} head_match={rs_rep.get('head_hash') == py_ver.head_hash} "
        f"rs_cusum_alarms={rs_rep.get('cusum_alarms')} rs_mga={rs_rep.get('rta_mandatory_go_around_count')} | "
        f"1B_tamper(py_valid={py_tamper.valid}, rs_rc={rs_tamper_proc.returncode})",
        flush=True,
    )

    # ==========================================================================
    # PHASE 4: Final WSL2 Teardown Hygiene Audit (0 leftover arduplane PIDs)
    # ==========================================================================
    r_pids = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", "pgrep -x arduplane || true"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    leftover_pids = [ln.strip() for ln in r_pids.stdout.splitlines() if ln.strip()]
    p4_teardown_ok = bool(pid_cleanly_terminated and len(leftover_pids) == 0 and ("pymavlink" not in sys.modules))
    print(
        f"\n--- [Phase 4] Teardown Hygiene: pid={ap_pid} terminated={pid_cleanly_terminated} "
        f"leftovers={leftover_pids} ---",
        flush=True,
    )

    checks = {
        "stressor0_crosswind_baseline_clean": bool(
            cruise_alt_agl_m >= 50.0
            and wind_rb.get("SIM_WIND_SPD") == 15.0
            and wind_rb.get("SIM_WIND_TURB") == 5.0
            and max(base_diff_list, default=0.0) >= 6.0
            and base_cusum_fa == 0
            and base_q_lwir_min >= 0.70
        ),
        "stressor1_crosswind_plus_live_gps_spoof_ekf3_42007": bool(
            spoof_ttd_s is not None
            and spoof_ttd_s <= 12.0
            and bank_a_err_m is not None
            and bank_a_err_m < 30.0
            and datum_dec_compound is not None
            and bool(datum_dec_compound["allowed"])
            and float(datum_dec_compound["abs_vertical_innovation_m"]) <= 3.0
            and int(st_compound_end["ekf3_active_source"]) == EKF3_SOURCE_VISION_OPTICAL
            and ack_42007_compound >= 1
        ),
        "stressor2_simultaneous_90deg_heading_coarse_rescue": bool(
            naive_0deg.n_geom_inliers < 12
            and heading_rescue_epochs >= 5
            and heading_rescue_inliers >= 12
            and heading_rescue_center_err_px <= 4.0
            and distractor_res.n_geom_inliers < 12
        ),
        "stressor3_simultaneous_lwir_shutter_freeze_fail_closed": bool(
            lwir_freeze_triggered_step is not None
            and lwir_freeze_q_min == 0.0
            and "LWIR_DEGRADED_EO_PRIMARY" in lwir_freeze_modes
            and (not bool(th_rec["lwir_nuc_frozen"]))
            and float(th_rec["q_lwir_effective"]) >= 0.70
        ),
        "stressor4_simultaneous_daa_75_82_stall_free_zero_84": bool(
            simultaneous_4fault_epoch_observed
            and ack_43002_ok
            and ack_192_ok
            and daa_cmd75_max_roll >= 10.0
            and abs(daa_cmd75_delta_hdg) >= 10.0
            and daa_cmd82_min_roll <= -8.0
            and cmd82_mask == ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES
            and daa_min_ias >= MIN_STALL_FREE_IAS_MPS
            and msg84_zero
        ),
        "recovery_gnss_dwell_and_rta_p0_ack_staged": bool(
            ekf_during_lt2s == EKF3_SOURCE_VISION_OPTICAL
            and ekf_after_ge2s == EKF3_SOURCE_GNSS
            and total_ack_42007_ok >= 2
            and int(o_nom.mode) == 4
            and int(o_fault.mode) == 0
            and bool(o_fault.mandatory_go_around)
            and int(o_hold.mode) == 0
            and bool(o_hold.mandatory_go_around)
            and (not bool(o_ack.mandatory_go_around))
            and p1_tick == 30
            and p4_tick == 50
            and staged_modes[-1] == 4
        ),
        "tape_dual_python_rust_replay_and_1byte_tamper_rc1": p3_replay_ok,
        "teardown_zero_leftover_wsl_arduplane_pids": p4_teardown_ok,
    }

    report = {
        "schema": "openxvs.v112.compound_fault_sil_gate/1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "simulator_backend": "WSL2_ARDUPLANE_OFFICIAL_SITL",
        "arduplane_metadata": meta,
        "pymavlink_imported": "pymavlink" in sys.modules,
        "strictly_ned_zero_enu": True,
        "non_flight_sim_only_tape_enforced": bool(
            py_ver.tx_inhibited_all and py_ver.actuator_authority_all_false
        ),
        "all_pass": all(checks.values()),
        "checks": checks,
        "phase0_baseline": {
            "cruise_alt_agl_m": round(cruise_alt_agl_m, 2),
            "wind_params_readback": wind_rb,
            "max_abs_ias_minus_gs_mps": base_max_ias_gs_diff,
            "cusum_false_alarms": base_cusum_fa,
            "base_q_lwir_min": round(base_q_lwir_min, 4),
        },
        "phase1_simultaneous_compound_faults": {
            "simultaneous_4fault_epoch_observed": simultaneous_4fault_epoch_observed,
            "spoof_ttd_sim_s": spoof_ttd_s,
            "bank_a_err_at_quarantine_m": round(bank_a_err_m, 2) if bank_a_err_m is not None else None,
            "bank_b_active_err_m": round(bank_b_err_m, 2) if bank_b_err_m is not None else None,
            "cusum_stat_at_quarantine": round(cusum_stat_at_q, 3),
            "datum_guard_decision": datum_dec_compound,
            "ekf3_active_source": int(st_compound_end["ekf3_active_source"]),
            "ack_42007_result0_compound": ack_42007_compound,
            "coarse_heading_90deg": {
                "xfeat_backend": "onnx_fp16" if XFEAT_FP16_ONNX.is_file() else "offline_fallback",
                "naive_0deg_inliers": int(naive_0deg.n_geom_inliers),
                "sweep_rescue_epochs": heading_rescue_epochs,
                "sweep_best_angle_deg": heading_rescue_angle_deg,
                "sweep_max_inliers": heading_rescue_inliers,
                "sweep_min_center_err_px": round(heading_rescue_center_err_px, 3),
                "distractor_inliers_0hmi": int(distractor_res.n_geom_inliers),
            },
            "lwir_shutter_freeze": {
                "triggered_at_compound_step": lwir_freeze_triggered_step,
                "min_q_lwir_effective": round(lwir_freeze_q_min, 4),
                "modes_during_freeze": sorted(set(lwir_freeze_modes)),
                "recovered_frozen": bool(th_rec["lwir_nuc_frozen"]),
                "recovered_q_lwir": float(th_rec["q_lwir_effective"]),
                "recovered_mode": str(th_rec["thermal_mode"]),
            },
            "daa_under_compound_faults": {
                "cmd75_ack_43002_ok": ack_43002_ok,
                "cmd75_ack_192_ok": ack_192_ok,
                "cmd75_max_abs_roll_deg": round(daa_cmd75_max_roll, 2),
                "cmd75_delta_heading_deg": round(daa_cmd75_delta_hdg, 2),
                "cmd82_min_roll_deg": round(daa_cmd82_min_roll, 2),
                "cmd82_type_mask_hex": hex(cmd82_mask) if cmd82_mask is not None else None,
                "cmd82_thrust_norm": last_cmd82.get("thrust_norm"),
                "min_vfr_hud_ias_mps": round(daa_min_ias, 2),
                "stall_floor_mps": MIN_STALL_FREE_IAS_MPS,
                "msg84_ever_emitted": not msg84_zero,
                "wire_msg84_seen": int(final_msg84_seen),
            },
        },
        "phase2_recovery": {
            "ekf3_during_lt2s_dwell": ekf_during_lt2s,
            "ekf3_after_ge2s_revert": ekf_after_ge2s,
            "ack_42007_result0_total": total_ack_42007_ok,
            "rta_fault_mode": int(o_fault.mode),
            "rta_hold_on_approach_mode": int(o_hold.mode),
            "rta_p1_recovery_ticks": p1_tick,
            "rta_p4_recovery_ticks": p4_tick,
            "rta_final_mode": staged_modes[-1],
        },
        "phase3_flight_recorder_replay": {
            "tape_path": str(tape_path.relative_to(REPO_ROOT)).replace("\\", "/")
            if tape_path.is_relative_to(REPO_ROOT)
            else str(tape_path),
            "python_verify": py_ver.to_dict(),
            "rust_replay_returncode": rs_proc.returncode,
            "rust_replay_report": rs_rep,
            "tamper_1byte_python_valid": py_tamper.valid,
            "tamper_1byte_rust_returncode": rs_tamper_proc.returncode,
            "tamper_1byte_rust_valid": rs_tamper_rep.get("valid"),
            "tamper_1byte_rust_first_failed_line": rs_tamper_rep.get("first_failed_line"),
        },
        "phase4_teardown_hygiene": {
            "spawned_wsl_arduplane_pid": ap_pid,
            "pid_cleanly_terminated": pid_cleanly_terminated,
            "leftover_wsl_arduplane_pids": leftover_pids,
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n[COMPOUND-SIL DONE] all_pass={report['all_pass']} | elapsed={report['elapsed_s']}s | "
        f"wrote {report_path}",
        flush=True,
    )
    return report


def main() -> int:
    rep = run_compound_fault_sil_gate()
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
