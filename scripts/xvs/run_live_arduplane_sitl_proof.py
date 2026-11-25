"""Live Closed-Loop WSL2 `ArduPlane >= 4.7.3` SITL (and Honest 6-DoF UDP Twin) Proof Runner.

Strictly ZERO GPL dependencies in OpenXVS (`pymavlink` strictly forbidden; 100% clean-room
Apache-2.0/MIT MAVLink 2.0 wire framing via `xvs.v2.xvs_mavlink.pod_daa_codec` and
`xvs.v2.xvs_rt_bridge.arduplane_bridge`). Running `ArduPlane` as a separate external OS process
in WSL2 communicating exclusively over UDP sockets preserves full wire-boundary GPL isolation
per DR-06 / DR-16.

Usage:
    PYTHONPATH=src python scripts/xvs/run_live_arduplane_sitl_proof.py
    PYTHONPATH=src python scripts/xvs/run_live_arduplane_sitl_proof.py --mode twin
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xvs.v2.studio_bridge.hil_bridge import StudioHilBridgeManager  # noqa: E402
from xvs.v2.xvs_mavlink.pod_daa_codec import (  # noqa: E402
    MSGID_COMMAND_INT,
    MSGID_COMMAND_LONG,
    MSGID_SET_ATTITUDE_TARGET,
    MSGID_SET_POSITION_TARGET_LOCAL_NED,
    MavAttitude,
    MavCommandAck,
    MavCommandInt,
    MavCommandLong,
    MavGlobalPositionInt,
    MavHeartbeat,
    MavSetAttitudeTarget,
    decode_frame,
    encode_frame,
    extract_mavlink_v2_frames,
    peek_mavlink_v2_header,
)
from xvs.v2.xvs_rt_bridge.arduplane_bridge import (  # noqa: E402
    ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
    DEFAULT_MAX_BANK_RAD,
    DEFAULT_MAX_PITCH_RAD,
    DEFAULT_V_MAX_MPS,
    DEFAULT_V_MIN_MPS,
    EKF3_SOURCE_GNSS,
    EKF3_SOURCE_VISION_OPTICAL,
    G_ACCEL_MPS2,
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

DEFAULT_PROOF_OUT_PATH = REPO_ROOT / "outputs" / "sitl_proof" / "live_arduplane_sitl_proof.json"
MIN_ARDUPLANE_SITL_VERSION: tuple[int, int, int] = (4, 7, 3)

ARDUPLANE_473_BASE_URL = "https://firmware.ardupilot.org/Plane/stable-4.7.3/SITL_x86_64_linux_gnu"
ARDUPLANE_473_PARM_URL = (
    "https://raw.githubusercontent.com/ArduPilot/ardupilot/"
    "d9c5097c3d8333435712ffed86bca8ce00c4bb46/Tools/autotest/models/plane.parm"
)
WSL_SITL_CACHE_DIR = "/var/tmp/openxvs_sitl"

OPENXVS_SITL_OVERLAY_PARM = """
ARMING_CHECK 0
SYSID_THISMAV 1
SERIAL0_PROTOCOL 2
SR0_EXTRA1 20
SR0_POSITION 20
SR0_EXT_STAT 5
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


def parse_arduplane_version(text: str) -> tuple[int, int, int] | None:
    """Extract semantic `(major, minor, patch)` version from ArduPlane version/help output."""
    m = re.search(r"(?:ArduPlane\s+V|^\s*)(\d+)\.(\d+)\.(\d+)", text, flags=re.MULTILINE)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _angular_diff_deg(a_deg: float, b_deg: float) -> float:
    """Signed shortest angle `a - b` wrapped to `[-180, +180)` degrees."""
    return ((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0


def _alloc_udp_port(bind_host: str = "127.0.0.1") -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _quat_to_euler_ned(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Convert Hamilton quaternion `(w, x, y, z)` to NED ZYX `(roll, pitch, yaw)` in radians."""
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


class FixedWing6DofUdpTwin:
    """Honest local 6-DoF bank-to-turn fixed-wing NED UDP twin speaking real zero-GPL MAVLink 2.0.

    Used ONLY when `--mode twin` is requested or when live WSL2 `ArduPlane >= 4.7.3` cannot run.
    Never impersonates ArduPlane; caller always labels backend `PYTHON_6DOF_FIXEDWING_UDP_TWIN`.
    """

    def __init__(self, *, bridge_rx_port: int, twin_rx_port: int) -> None:
        self._bridge_rx_port = int(bridge_rx_port)
        self._twin_rx_port = int(twin_rx_port)
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._seq: int = 0

        # Strictly NED fixed-wing state at KSFO 28L cruise
        self.lat_deg: float = 37.61172
        self.lon_deg: float = -122.358367
        self.alt_m: float = 120.0
        self.roll_rad: float = 0.0
        self.pitch_rad: float = 0.0
        self.yaw_rad: float = math.radians(298.0)
        self.airspeed_mps: float = 24.0
        self.custom_mode: int = PLANE_MODE_GUIDED
        self.armed: bool = True
        self.ekf3_source: int = EKF3_SOURCE_GNSS
        self._target_roll_rad: float = 0.0
        self._target_pitch_rad: float = 0.0
        self._target_heading_rad: float | None = None
        self._cmd_roll_override_until: float = 0.0
        self.rx_commands_total: int = 0
        self.msg84_received: int = 0

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", self._twin_rx_port))
        sock.settimeout(0.02)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _next_seq(self) -> int:
        s = self._seq & 0xFF
        self._seq = (self._seq + 1) & 0xFF
        return s

    def _send(self, msg: Any) -> None:
        if self._sock is None:
            return
        wire = encode_frame(msg, seq=self._next_seq(), sysid=1, compid=1)
        try:
            self._sock.sendto(wire, ("127.0.0.1", self._bridge_rx_port))
        except OSError:
            pass

    def _handle_rx(self, raw_data: bytes, now: float) -> None:
        frames, _ = extract_mavlink_v2_frames(raw_data)
        for frame in frames:
            try:
                hdr = peek_mavlink_v2_header(frame)
            except Exception:
                continue
            mid = int(hdr["msgid"])
            if mid == MSGID_SET_POSITION_TARGET_LOCAL_NED:
                self.msg84_received += 1
                continue
            try:
                dec = decode_frame(frame)
            except Exception:
                continue
            msg = dec.message
            if isinstance(msg, MavCommandInt):
                self.rx_commands_total += 1
                if msg.command == MAV_CMD_GUIDED_CHANGE_HEADING:
                    self._target_heading_rad = math.radians(float(msg.param2) % 360.0)
                    self._cmd_roll_override_until = 0.0
                    self._send(MavCommandAck(command=msg.command, result=0))
                elif msg.command == MAV_CMD_DO_REPOSITION:
                    d_n = (float(msg.x) * 1e-7 - self.lat_deg) * 111_132.954
                    d_e = (float(msg.y) * 1e-7 - self.lon_deg) * 111_319.491 * max(
                        0.1, math.cos(math.radians(self.lat_deg))
                    )
                    self._target_heading_rad = math.atan2(d_e, d_n) % (2.0 * math.pi)
                    self._cmd_roll_override_until = 0.0
                    self._send(MavCommandAck(command=msg.command, result=0))
            elif isinstance(msg, MavSetAttitudeTarget):
                self.rx_commands_total += 1
                r_cmd, p_cmd, _ = _quat_to_euler_ned(msg.q)
                self._target_roll_rad = max(-DEFAULT_MAX_BANK_RAD, min(DEFAULT_MAX_BANK_RAD, r_cmd))
                self._target_pitch_rad = max(-DEFAULT_MAX_PITCH_RAD, min(DEFAULT_MAX_PITCH_RAD, p_cmd))
                self._cmd_roll_override_until = now + 0.6
            elif isinstance(msg, MavCommandLong):
                self.rx_commands_total += 1
                if msg.command == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET:
                    src = int(round(msg.param1))
                    if src in (EKF3_SOURCE_GNSS, EKF3_SOURCE_VISION_OPTICAL):
                        self.ekf3_source = src
                        self._send(MavCommandAck(command=msg.command, result=0))
                    else:
                        self._send(MavCommandAck(command=msg.command, result=4))
                elif msg.command == MAV_CMD_DO_SET_MODE:
                    self.custom_mode = int(round(msg.param2))
                    self._send(MavCommandAck(command=msg.command, result=0))
                elif msg.command == MAV_CMD_COMPONENT_ARM_DISARM:
                    self.armed = bool(round(msg.param1) >= 1)
                    self._send(MavCommandAck(command=msg.command, result=0))
                elif msg.command == MAV_CMD_SET_MESSAGE_INTERVAL:
                    self._send(MavCommandAck(command=msg.command, result=0))

    def _run(self) -> None:
        t_boot = time.monotonic()
        last_step = t_boot
        last_hb = 0.0
        dt = 0.04  # 25 Hz fixed-wing kinematics
        while not self._stop.is_set():
            now = time.monotonic()
            # Drain incoming commands
            if self._sock is not None:
                try:
                    data, _ = self._sock.recvfrom(4096)
                    self._handle_rx(data, now)
                except socket.timeout:
                    pass
                except OSError:
                    break

            if (now - last_step) < dt:
                continue
            step_dt = min(0.1, now - last_step)
            last_step = now

            # Bank-to-turn lateral dynamics (tau_roll = 0.25 s, coordinated turn psi_dot = g*tan(phi)/V)
            if now < self._cmd_roll_override_until:
                des_roll = self._target_roll_rad
            elif self._target_heading_rad is not None:
                err_deg = _angular_diff_deg(math.degrees(self._target_heading_rad), math.degrees(self.yaw_rad))
                des_roll = max(-math.radians(25.0), min(math.radians(25.0), math.radians(err_deg) * 1.1))
            else:
                des_roll = 0.0

            tau_roll = 0.22
            roll_rate = (des_roll - self.roll_rad) / tau_roll
            self.roll_rad = max(
                -DEFAULT_MAX_BANK_RAD,
                min(DEFAULT_MAX_BANK_RAD, self.roll_rad + roll_rate * step_dt),
            )
            pitch_rate = (self._target_pitch_rad - self.pitch_rad) / 0.30
            self.pitch_rad = max(
                -DEFAULT_MAX_PITCH_RAD,
                min(DEFAULT_MAX_PITCH_RAD, self.pitch_rad + pitch_rate * step_dt),
            )

            tas = max(DEFAULT_V_MIN_MPS, min(DEFAULT_V_MAX_MPS, self.airspeed_mps))
            yaw_rate = (G_ACCEL_MPS2 * math.tan(self.roll_rad)) / tas
            self.yaw_rad = (self.yaw_rad + yaw_rate * step_dt) % (2.0 * math.pi)

            vn = tas * math.cos(self.pitch_rad) * math.cos(self.yaw_rad)
            ve = tas * math.cos(self.pitch_rad) * math.sin(self.yaw_rad)
            vd = -tas * math.sin(self.pitch_rad)
            self.lat_deg += (vn * step_dt) / 111_132.954
            self.lon_deg += (ve * step_dt) / (111_319.491 * max(0.1, math.cos(math.radians(self.lat_deg))))
            self.alt_m = max(20.0, self.alt_m - vd * step_dt)

            t_ms = int((now - t_boot) * 1000.0) & 0xFFFFFFFF
            if (now - last_hb) >= 0.4:
                self._send(
                    MavHeartbeat(
                        custom_mode=self.custom_mode,
                        mav_type=1,
                        autopilot=3,
                        base_mode=0x81 if self.armed else 0x01,
                        system_status=4,
                    )
                )
                last_hb = now

            self._send(
                MavAttitude(
                    time_boot_ms=t_ms,
                    roll=self.roll_rad,
                    pitch=self.pitch_rad,
                    yaw=self.yaw_rad if self.yaw_rad <= math.pi else self.yaw_rad - 2.0 * math.pi,
                    rollspeed=roll_rate,
                    pitchspeed=pitch_rate,
                    yawspeed=yaw_rate,
                )
            )
            self._send(
                MavGlobalPositionInt(
                    time_boot_ms=t_ms,
                    lat_e7=int(round(self.lat_deg * 1e7)),
                    lon_e7=int(round(self.lon_deg * 1e7)),
                    alt_mm=int(round(self.alt_m * 1000.0)),
                    relative_alt_mm=int(round(self.alt_m * 1000.0)),
                    vx_cms=int(round(vn * 100.0)),
                    vy_cms=int(round(ve * 100.0)),
                    vz_cms=int(round(vd * 100.0)),
                    hdg_cdeg=int(round((math.degrees(self.yaw_rad) % 360.0) * 100.0)) % 36000,
                )
            )


def _prepare_wsl2_arduplane_binary() -> dict[str, Any]:
    """Ensure official WSL2 `ArduPlane >= 4.7.3` SITL binary and overlay params exist."""
    prep_bash = (
        f"set -e; mkdir -p {WSL_SITL_CACHE_DIR}/run; cd {WSL_SITL_CACHE_DIR}; "
        f"if [ ! -x arduplane ]; then curl -fsSL -o arduplane {ARDUPLANE_473_BASE_URL}/arduplane && chmod +x arduplane; fi; "
        f"if [ ! -s firmware-version.txt ]; then curl -fsSL -o firmware-version.txt {ARDUPLANE_473_BASE_URL}/firmware-version.txt; fi; "
        f"if [ ! -s plane.parm ]; then curl -fsSL -o plane.parm {ARDUPLANE_473_PARM_URL}; fi; "
        f"sha256sum arduplane | awk '{{print $1}}'; "
        f"cat firmware-version.txt; echo ''; "
        f"strings arduplane | grep -oE 'ArduPlane V[0-9]+\\.[0-9]+\\.[0-9]+' | head -n1"
    )
    out = subprocess.check_output(["wsl", "-e", "bash", "-lc", prep_bash], timeout=60).decode("utf-8", "ignore")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"Unexpected WSL2 ArduPlane metadata output: {lines!r}")
    sha256_hex = lines[0]
    fw_ver_str = lines[1]
    apm_ver_line = lines[2]
    ver = parse_arduplane_version(apm_ver_line) or parse_arduplane_version(fw_ver_str)
    if ver is None or ver < MIN_ARDUPLANE_SITL_VERSION:
        raise RuntimeError(
            f"ArduPlane version {ver!r} from {apm_ver_line!r} does not satisfy >= {MIN_ARDUPLANE_SITL_VERSION}"
        )

    # Write combined OpenXVS parameter file in WSL2
    overlay_cmd = (
        f"cat {WSL_SITL_CACHE_DIR}/plane.parm > {WSL_SITL_CACHE_DIR}/openxvs_plane.parm && "
        f"cat <<'EOF' >> {WSL_SITL_CACHE_DIR}/openxvs_plane.parm\n{OPENXVS_SITL_OVERLAY_PARM}\nEOF"
    )
    subprocess.check_call(["wsl", "-e", "bash", "-lc", overlay_cmd], timeout=15)

    net_out = subprocess.check_output(
        ["wsl", "-e", "bash", "-lc", "ip route show default | awk '{print $3}'; hostname -I | awk '{print $1}'"],
        timeout=15,
    ).decode("utf-8", "ignore")
    net_lines = [ln.strip() for ln in net_out.splitlines() if ln.strip()]
    if len(net_lines) < 2:
        raise RuntimeError(f"Failed to resolve WSL2 gateway/IP: {net_lines!r}")

    return {
        "binary_path_wsl": f"{WSL_SITL_CACHE_DIR}/arduplane",
        "binary_sha256": sha256_hex,
        "firmware_version_txt": fw_ver_str,
        "apm_version_line": apm_ver_line,
        "version_tuple": list(ver),
        "windows_gateway_ip": net_lines[0],
        "wsl_ip": net_lines[1],
    }


def _execute_closed_loop_mission_on_bridge(
    mgr: StudioHilBridgeManager,
    *,
    is_live_arduplane: bool,
    takeoff_wait_s: float = 7.0,
) -> dict[str, Any]:
    """Drive full closed-loop mission (Boot -> Takeoff -> #75 -> #82 -> EKF3 42007) via `StudioHilBridgeManager`."""
    seq = 0

    def next_seq() -> int:
        nonlocal seq
        s = seq & 0xFF
        seq += 1
        return s

    def send_hb() -> None:
        mgr.send_raw_tx_frame(encode_heartbeat_frame(seq=next_seq()))

    def wait_with_hb(duration_s: float) -> list[dict[str, float]]:
        samples: list[dict[str, float]] = []
        end = time.monotonic() + duration_s
        last_hb = 0.0
        while time.monotonic() < end:
            now = time.monotonic()
            if (now - last_hb) >= 0.4:
                send_hb()
                last_hb = now
            st = mgr.status()
            ned = st["state_ned"]
            samples.append(
                {
                    "roll_deg": float(ned["roll_deg"]),
                    "pitch_deg": float(ned["pitch_deg"]),
                    "heading_deg": float(ned["heading_deg"]),
                    "relative_alt_m": float(ned["relative_alt_m"]),
                    "airspeed_mps": float(ned["airspeed_mps"]),
                }
            )
            time.sleep(0.1)
        return samples

    # Phase 1: Boot & Telemetry Lock (shadow mode: tx_enabled=False)
    t_boot_deadline = time.monotonic() + 5.0
    while time.monotonic() < t_boot_deadline:
        send_hb()
        st_b = mgr.status()
        if int(st_b["heartbeat_count"]) >= 1 and int(st_b["attitude_count"]) >= 1 and int(st_b["global_pos_count"]) >= 1:
            break
        time.sleep(0.15)
    st_boot = mgr.status()
    if int(st_boot["heartbeat_count"]) < 1:
        raise RuntimeError(f"No #0 HEARTBEAT received from SITL within boot timeout (rx={st_boot['rx_packets']})")

    # Request 20 Hz ATTITUDE (#30) & GLOBAL_POSITION_INT (#33)
    for mid in (30, 33):
        mgr.send_raw_tx_frame(
            encode_command_long_frame(
                command=MAV_CMD_SET_MESSAGE_INTERVAL,
                param1=float(mid),
                param2=50_000.0,
                seq=next_seq(),
            )
        )

    # Phase 2: Takeoff to Cruise (ArduPlane starts on KSFO runway; twin starts airborne)
    if is_live_arduplane:
        mgr.send_raw_tx_frame(
            encode_command_long_frame(
                command=MAV_CMD_DO_SET_MODE,
                param1=1.0,
                param2=float(PLANE_MODE_TAKEOFF),
                seq=next_seq(),
            )
        )
        wait_with_hb(0.4)
        mgr.send_raw_tx_frame(
            encode_command_long_frame(
                command=MAV_CMD_COMPONENT_ARM_DISARM,
                param1=1.0,
                param2=21196.0,
                seq=next_seq(),
            )
        )
        wait_with_hb(takeoff_wait_s)

    # Enter GUIDED (15)
    mgr.send_raw_tx_frame(
        encode_command_long_frame(
            command=MAV_CMD_DO_SET_MODE,
            param1=1.0,
            param2=float(PLANE_MODE_GUIDED),
            seq=next_seq(),
        )
    )
    wait_with_hb(0.8 if is_live_arduplane else 0.3)
    st_guided = mgr.status()
    baseline_ned = dict(st_guided["state_ned"])
    base_heading_deg = float(baseline_ned["heading_deg"])
    cruise_alt_m = float(baseline_ned["relative_alt_m"])

    # Phase 3: Autonomous Closed-Loop CBDR Avoidance via #75 COMMAND_INT (43002 + 192)
    mgr.set_autonomy_triggers(
        cbdr_threat_active=True,
        cbdr_evade_direction=1.0,
        gnss_spoof_alarm=False,
        gnss_healthy=True,
        tx_command_mode="command_int_heading",
        tx_enabled=True,
        min_tx_interval_s=0.4 if is_live_arduplane else 0.12,
    )
    samples_75 = wait_with_hb(3.0 if is_live_arduplane else 1.4)
    # Also exercise #75 DO_REPOSITION (192) cleanly
    mgr.set_autonomy_triggers(tx_command_mode="command_int", min_tx_interval_s=0.4 if is_live_arduplane else 0.12)
    samples_75_repo = wait_with_hb(1.0 if is_live_arduplane else 0.5)
    all_75_samples = samples_75 + samples_75_repo
    max_abs_roll_75_deg = max(abs(s["roll_deg"]) for s in all_75_samples)
    hdg_after_75_deg = all_75_samples[-1]["heading_deg"]
    delta_hdg_75_deg = _angular_diff_deg(hdg_after_75_deg, base_heading_deg)

    # Phase 4: Autonomous Closed-Loop CBDR Avoidance via #82 SET_ATTITUDE_TARGET (left bank -25 deg)
    mgr.set_autonomy_triggers(
        cbdr_threat_active=True,
        cbdr_evade_direction=-1.0,
        tx_command_mode="set_attitude_target",
        tx_enabled=True,
        min_tx_interval_s=0.08,
    )
    samples_82 = wait_with_hb(2.4 if is_live_arduplane else 1.2)
    min_roll_82_deg = min(s["roll_deg"] for s in samples_82)
    final_roll_82_deg = samples_82[-1]["roll_deg"]

    # Phase 5: GPS Spoof -> Latch-Gated #76 COMMAND_LONG(42007) Vision (2) + >=2.0s Dwell Revert to GNSS (1)
    mgr.set_autonomy_triggers(gnss_spoof_alarm=True, gnss_healthy=True)
    wait_with_hb(0.7 if is_live_arduplane else 0.4)
    st_spoof = mgr.status()
    ekf_after_spoof = int(st_spoof["ekf3_active_source"])

    # Clear spoof -> verify < 2.0 s dwell stays latched on Vision (2)
    mgr.set_autonomy_triggers(gnss_spoof_alarm=False, gnss_healthy=True)
    wait_with_hb(0.8)
    st_mid_dwell = mgr.status()
    ekf_during_dwell = int(st_mid_dwell["ekf3_active_source"])

    # Wait remaining >= 2.0 s dwell -> verify automatic revert to GNSS (1) + ACK 42007
    wait_with_hb(1.8)
    st_final = mgr.status()
    ekf_after_revert = int(st_final["ekf3_active_source"])

    ack_list: list[dict[str, int]] = list(st_final["ack_history"])
    accepted_cmds = [int(a["command"]) for a in ack_list if int(a["result"]) == 0]
    ack_42007_count = sum(1 for c in accepted_cmds if c == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET)
    ack_43002_ok = MAV_CMD_GUIDED_CHANGE_HEADING in accepted_cmds
    ack_192_ok = MAV_CMD_DO_REPOSITION in accepted_cmds

    tx_msgids = {int(k): int(v) for k, v in st_final["tx_msgids_emitted"].items()}
    msg84_blocked = (
        not bool(st_final["msg84_ever_emitted"])
        and MSGID_SET_POSITION_TARGET_LOCAL_NED not in tx_msgids
    )

    checks = {
        "rx_heartbeat_att_gpi_verified": bool(
            int(st_final["heartbeat_count"]) >= 2
            and int(st_final["attitude_count"]) >= 20
            and int(st_final["global_pos_count"]) >= 20
        ),
        "airborne_cruise_achieved": bool(cruise_alt_m >= 20.0),
        "cmd75_tx_and_ack_verified": bool(
            MSGID_COMMAND_INT in tx_msgids
            and tx_msgids[MSGID_COMMAND_INT] >= 2
            and ack_43002_ok
            and ack_192_ok
        ),
        "cmd75_roll_and_heading_response": bool(
            max_abs_roll_75_deg >= 10.0 and abs(delta_hdg_75_deg) >= 12.0
        ),
        "cmd82_tx_and_left_bank_response": bool(
            MSGID_SET_ATTITUDE_TARGET in tx_msgids
            and tx_msgids[MSGID_SET_ATTITUDE_TARGET] >= 5
            and min_roll_82_deg <= -10.0
        ),
        "ekf3_42007_spoof_switch_and_2s_dwell_revert": bool(
            MSGID_COMMAND_LONG in tx_msgids
            and ekf_after_spoof == EKF3_SOURCE_VISION_OPTICAL
            and ekf_during_dwell == EKF3_SOURCE_VISION_OPTICAL
            and ekf_after_revert == EKF3_SOURCE_GNSS
            and ack_42007_count >= 2
        ),
        "msg84_never_emitted": msg84_blocked,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "telemetry_summary": {
            "rx_packets": int(st_final["rx_packets"]),
            "tx_packets": int(st_final["tx_packets"]),
            "rejected_packets": int(st_final["rejected_packets"]),
            "heartbeat_count": int(st_final["heartbeat_count"]),
            "attitude_count": int(st_final["attitude_count"]),
            "global_pos_count": int(st_final["global_pos_count"]),
            "tx_msgids_emitted": {str(k): v for k, v in sorted(tx_msgids.items())},
            "last_tx_latency_ms": st_final["last_tx_latency_ms"],
        },
        "flight_response": {
            "cruise_alt_agl_m": round(cruise_alt_m, 2),
            "baseline_heading_deg": round(base_heading_deg, 2),
            "heading_after_cmd75_deg": round(hdg_after_75_deg, 2),
            "delta_heading_cmd75_deg": round(delta_hdg_75_deg, 2),
            "max_abs_roll_cmd75_deg": round(max_abs_roll_75_deg, 2),
            "min_roll_cmd82_deg": round(min_roll_82_deg, 2),
            "final_roll_cmd82_deg": round(final_roll_82_deg, 2),
            "ekf3_after_spoof": ekf_after_spoof,
            "ekf3_during_lt2s_dwell": ekf_during_dwell,
            "ekf3_after_ge2s_revert": ekf_after_revert,
            "ack_42007_accepted_count": ack_42007_count,
            "ack_43002_accepted": ack_43002_ok,
            "ack_192_accepted": ack_192_ok,
            "accepted_commands": sorted(set(accepted_cmds)),
        },
        "msg84_ever_emitted": not msg84_blocked,
    }


def run_python_6dof_udp_twin_proof(
    *,
    fallback_reason: str | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run honest local `PYTHON_6DOF_FIXEDWING_UDP_TWIN` closed-loop UDP proof."""
    t0 = time.perf_counter()
    bridge_rx_port = _alloc_udp_port("127.0.0.1")
    twin_rx_port = _alloc_udp_port("127.0.0.1")
    mgr = StudioHilBridgeManager()
    twin = FixedWing6DofUdpTwin(bridge_rx_port=bridge_rx_port, twin_rx_port=twin_rx_port)
    try:
        mgr.connect(
            mode="mavlink2_udp",
            bind_host="127.0.0.1",
            rx_port=bridge_rx_port,
            tx_host="127.0.0.1",
            tx_port=twin_rx_port,
            tx_enabled=False,
        )
        twin.start()
        mission = _execute_closed_loop_mission_on_bridge(mgr, is_live_arduplane=False, takeoff_wait_s=0.0)
        twin_msg84 = int(twin.msg84_received)
    finally:
        twin.stop()
        mgr.shutdown()

    report: dict[str, Any] = {
        "schema": "openxvs.phase2b.live_sitl_proof/1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "simulator_backend": "PYTHON_6DOF_FIXEDWING_UDP_TWIN",
        "arduplane_verified": False,
        "fallback_reason": fallback_reason if fallback_reason is not None else "explicit_twin_mode",
        "gpl_free_openxvs_codec": True,
        "pymavlink_imported": "pymavlink" in sys.modules,
        "transport": {
            "protocol": "MAVLink_2.0_UDP",
            "bridge_rx_port": bridge_rx_port,
            "twin_rx_port": twin_rx_port,
            "wire_boundary_gpl_isolated": True,
        },
        "twin_msg84_received": twin_msg84,
        **mission,
    }
    report["pass"] = bool(mission["pass"] and twin_msg84 == 0 and not report["pymavlink_imported"])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_live_arduplane_sitl_proof(
    *,
    out_path: Path | None = DEFAULT_PROOF_OUT_PATH,
    allow_twin_fallback: bool = True,
    speedup: int = 2,
) -> dict[str, Any]:
    """Run real live WSL2 `ArduPlane >= 4.7.3` closed-loop UDP SITL proof (with honest fallback)."""
    t0 = time.perf_counter()
    try:
        meta = _prepare_wsl2_arduplane_binary()
    except Exception as exc:
        if allow_twin_fallback:
            return run_python_6dof_udp_twin_proof(
                fallback_reason=f"WSL2 ArduPlane >= 4.7.3 unavailable: {type(exc).__name__}: {exc}",
                out_path=out_path,
            )
        raise

    gw = str(meta["windows_gateway_ip"])
    wsl_ip = str(meta["wsl_ip"])
    win_rx_port = _alloc_udp_port("0.0.0.0")
    wsl_ap_udp_port = 14570
    wsl_tx_udp_port = 14571

    launcher_py = f"""
import os, select, signal, socket, subprocess
s_ap = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_ap.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s_ap.bind(('127.0.0.1', {wsl_ap_udp_port}))
s_win = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_win.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s_win.bind(('0.0.0.0', {wsl_tx_udp_port}))
os.makedirs('{WSL_SITL_CACHE_DIR}/run', exist_ok=True)
ap = subprocess.Popen([
    '{WSL_SITL_CACHE_DIR}/arduplane', '--wipe', '--model', 'plane', '--speedup', '{int(speedup)}',
    '--home', '37.61172,-122.358367,4,298',
    '--defaults', '{WSL_SITL_CACHE_DIR}/openxvs_plane.parm',
    '--serial0', 'udpclient:127.0.0.1:{wsl_ap_udp_port}',
], cwd='{WSL_SITL_CACHE_DIR}/run', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('AP_PID=' + str(ap.pid), flush=True)
stop = False
def _sig(signum, frame):
    global stop
    stop = True
signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)
ap_peer = None
try:
    while not stop and ap.poll() is None:
        r, _, _ = select.select([s_ap, s_win], [], [], 0.1)
        if s_ap in r:
            d, a = s_ap.recvfrom(4096)
            ap_peer = a
            s_win.sendto(d, ('{gw}', {win_rx_port}))
        if s_win in r:
            d, _ = s_win.recvfrom(4096)
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
    mgr = StudioHilBridgeManager()
    proc: subprocess.Popen[str] | None = None
    ap_pid: int | None = None
    pid_cleanly_terminated = False
    try:
        mgr.connect(
            mode="mavlink2_udp",
            bind_host="0.0.0.0",
            rx_port=win_rx_port,
            tx_host=wsl_ip,
            tx_port=wsl_tx_udp_port,
            tx_enabled=False,
        )
        proc = subprocess.Popen(
            ["wsl", "-e", "python3", "-c", launcher_py],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        first_line = proc.stdout.readline().strip()
        if not first_line.startswith("AP_PID="):
            raise RuntimeError(f"Failed to start WSL2 ArduPlane SITL launcher: {first_line!r}")
        ap_pid = int(first_line.split("=", 1)[1])

        mission = _execute_closed_loop_mission_on_bridge(
            mgr,
            is_live_arduplane=True,
            takeoff_wait_s=6.5,
        )
    except Exception as exc:
        if allow_twin_fallback:
            return run_python_6dof_udp_twin_proof(
                fallback_reason=f"Live WSL2 ArduPlane SITL execution error: {type(exc).__name__}: {exc}",
                out_path=out_path,
            )
        raise
    finally:
        mgr.shutdown()
        if ap_pid is not None:
            subprocess.run(
                ["wsl", "-e", "bash", "-lc", f"kill -TERM {ap_pid} 2>/dev/null || true"],
                check=False,
                timeout=10,
            )
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=4.0)
            except Exception:
                proc.kill()
                proc.wait(timeout=2.0)
        if ap_pid is not None:
            chk = subprocess.run(
                ["wsl", "-e", "bash", "-lc", f"kill -0 {ap_pid} 2>/dev/null"],
                check=False,
                timeout=10,
            )
            pid_cleanly_terminated = chk.returncode != 0

    report: dict[str, Any] = {
        "schema": "openxvs.phase2b.live_sitl_proof/1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "simulator_backend": "WSL2_ARDUPLANE_OFFICIAL_SITL",
        "arduplane_verified": True,
        "arduplane_version": ".".join(str(v) for v in meta["version_tuple"]),
        "arduplane_version_tuple": meta["version_tuple"],
        "arduplane_firmware_version_txt": meta["firmware_version_txt"],
        "arduplane_apm_version_line": meta["apm_version_line"],
        "arduplane_binary_sha256": meta["binary_sha256"],
        "wsl_arduplane_pid": ap_pid,
        "wsl_pid_cleanly_terminated": pid_cleanly_terminated,
        "gpl_free_openxvs_codec": True,
        "pymavlink_imported": "pymavlink" in sys.modules,
        "transport": {
            "protocol": "MAVLink_2.0_UDP",
            "windows_rx_port": win_rx_port,
            "wsl_tx_port": wsl_tx_udp_port,
            "wsl_arduplane_udp_port": wsl_ap_udp_port,
            "wire_boundary_gpl_isolated": True,
        },
        **mission,
    }
    report["pass"] = bool(
        mission["pass"]
        and pid_cleanly_terminated
        and not report["pymavlink_imported"]
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenXVS Phase 2B Live WSL2 ArduPlane >=4.7.3 SITL Proof")
    parser.add_argument(
        "--mode",
        choices=("auto", "arduplane", "twin"),
        default="auto",
        help="auto: live WSL2 ArduPlane >=4.7.3 with honest twin fallback; arduplane: require live ArduPlane; twin: Python 6-DoF UDP twin",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_PROOF_OUT_PATH,
        help="Destination JSON path for deterministic SITL proof report",
    )
    parser.add_argument("--speedup", type=int, default=2, help="ArduPlane SITL speedup factor (default 2)")
    args = parser.parse_args(argv)

    if args.mode == "twin":
        report = run_python_6dof_udp_twin_proof(out_path=args.out)
    else:
        report = run_live_arduplane_sitl_proof(
            out_path=args.out,
            allow_twin_fallback=(args.mode == "auto"),
            speedup=args.speedup,
        )

    print(json.dumps(report, indent=2))
    return 0 if bool(report["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
