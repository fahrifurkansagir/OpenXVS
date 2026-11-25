"""ArduPlane Fixed-Wing MAVLink v2 Bridge, EKF3 42007 Latch & UDP Loopback Harness (`V2-M3L-5`).

Strictly enforces `DR-06:171, 193` and `SP3_M3_XVS_RT_BENCH_DESIGN.md` §3.4 (`BC-M3-m6`):
- Hard-rejects MAVLink `#84 SET_POSITION_TARGET_LOCAL_NED` for fixed-wing lateral avoidance.
- Builds finite-checked, envelope-clamped `COMMAND_INT (#75)` and `SET_ATTITUDE_TARGET (#82)`
  commands (`|bank| <= 30 deg`, `V >= V_min`, Hamilton `(w, x, y, z)` with `w >= 0`).
- Implements `Ekf3SourceSwitchLatch` (`42007` / `EK3_SRC1_POSXY`, `>= 2.0 s` dwell hysteresis
  before reverting to GNSS), rejecting `NaN`/`±Inf` and non-monotonic timestamps (`ValueError`).
- Runs local `127.0.0.1` UDP loopback sending/receiving `100` signed `77 B` `#52001
  OPENXVS_BEARING_TRACK` frames via frozen `src/xvs/core_ffi.py:encode_mavlink_bearing_track`,
  verifying byte-exact framing + X.25 CRC + SHA-256 signature AND 1-bit tamper rejection.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import math
import socket
import struct
from typing import Any

from xvs.core_ffi import XvsBearingTrack, XvsCoreFFI
from xvs.v2.xvs_mavlink.pod_daa_codec import (
    MavAttitude,
    MavCommandAck,
    MavCommandInt,
    MavCommandLong,
    MavGlobalPositionInt,
    MavHeartbeat,
    MavSetAttitudeTarget,
    MavlinkV2Frame,
    decode_frame,
    encode_frame,
)

MAVLINK_V2_STX = 0xFD
MAVLINK_IFLAG_SIGNED = 0x01

MAVLINK_MSG_HEARTBEAT = 0
MAVLINK_MSG_ATTITUDE = 30
MAVLINK_MSG_GLOBAL_POSITION_INT = 33
MAVLINK_MSG_COMMAND_INT = 75
MAVLINK_MSG_COMMAND_LONG = 76
MAVLINK_MSG_COMMAND_ACK = 77
MAVLINK_MSG_SET_ATTITUDE_TARGET = 82
MAVLINK_MSG_SET_POSITION_TARGET_LOCAL_NED = 84
MAVLINK_MSG_OPENXVS_BEARING_TRACK = 52001

MAV_CMD_DO_SET_MODE = 176
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_CMD_DO_REPOSITION = 192
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MAV_CMD_GUIDED_CHANGE_HEADING = 43002

PLANE_MODE_FBWA = 5
PLANE_MODE_FBWB = 6
PLANE_MODE_CRUISE = 7
PLANE_MODE_AUTO = 10
PLANE_MODE_TAKEOFF = 13
PLANE_MODE_GUIDED = 15

G_ACCEL_MPS2 = 9.80665

OPENXVS_BEARING_TRACK_CRC_EXTRA = 73
OPENXVS_BEARING_TRACK_PAYLOAD_LEN = 52
OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN = 77
_BEARING_TRACK_STRUCT = struct.Struct("<Q3i3h3hHHHBB6H")
assert _BEARING_TRACK_STRUCT.size == OPENXVS_BEARING_TRACK_PAYLOAD_LEN

ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET = 42007
EKF3_SOURCE_GNSS = 1
EKF3_SOURCE_VISION_OPTICAL = 2

DEFAULT_MAX_BANK_RAD = math.radians(30.0)
DEFAULT_MAX_PITCH_RAD = math.radians(20.0)
DEFAULT_V_MIN_MPS = 18.0
DEFAULT_V_MAX_MPS = 65.0
MIN_EKF3_REVERT_DWELL_S = 2.0
MAX_LEVER_ARM_NORM_M = 10.0
MAX_TRANSPORT_DELAY_MS = 500.0

ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES = 0x07
ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE = 0x40
MIN_SAFE_THRUST_NORM = 0.35
MIN_CLIMB_THRUST_NORM = 0.65

SITL_DEFER_TAG = "[LOCAL_LOOPBACK_SITL_EMULATOR; FULL_ARDUPILOT_BINARY_DEFERRED_TO_PHASE_4]"


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Boolean value rejected for numeric field {name!r}: {value!r}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite input rejected for {name!r}: {value!r}")
    return val


def _require_finite_vec3(name: str, vec: Any) -> tuple[float, float, float]:
    try:
        seq = tuple(vec)
    except TypeError as exc:
        raise ValueError(f"Expected 3-element vector for {name!r}, got {vec!r}") from exc
    if len(seq) != 3:
        raise ValueError(f"Expected 3-element vector for {name!r}, got length {len(seq)}")
    return (
        _require_finite(f"{name}[0]", seq[0]),
        _require_finite(f"{name}[1]", seq[1]),
        _require_finite(f"{name}[2]", seq[2]),
    )


@dataclass(frozen=True)
class CompensatedVisualPoseNed:
    """Strictly NED visual CG position after camera-to-CG lever-arm & transport-delay compensation."""

    p_cg_ned_m: tuple[float, float, float]
    p_cg_at_capture_ned_m: tuple[float, float, float]
    lever_arm_ned_m: tuple[float, float, float]
    delay_shift_ned_m: tuple[float, float, float]
    lever_arm_norm_m: float
    transport_delay_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_cg_ned_m": [round(x, 6) for x in self.p_cg_ned_m],
            "p_cg_at_capture_ned_m": [round(x, 6) for x in self.p_cg_at_capture_ned_m],
            "lever_arm_ned_m": [round(x, 6) for x in self.lever_arm_ned_m],
            "delay_shift_ned_m": [round(x, 6) for x in self.delay_shift_ned_m],
            "lever_arm_norm_m": round(self.lever_arm_norm_m, 6),
            "transport_delay_ms": round(self.transport_delay_ms, 3),
        }


def compensate_visual_pose_ned(
    p_cam_ned_m: tuple[float, float, float] | list[float],
    v_cg_ned_mps: tuple[float, float, float] | list[float],
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
    lever_arm_body_m: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
    transport_delay_ms: float = 0.0,
) -> CompensatedVisualPoseNed:
    """Compensate a camera visual NED position for Body-FRD lever arm and pipeline transport delay.

    Strictly NED (`[North, East, Down]`) and Body FRD (`[Forward, Right, Down]`):
      - `p_cg_at_capture = p_cam_ned - R_ned_body(roll, pitch, yaw) @ lever_arm_body_m`
      - `p_cg_now = p_cg_at_capture + v_cg_ned_mps * (transport_delay_ms / 1000.0)`
    Fail-closed on non-finite inputs, `||lever_arm|| > 10.0 m`, or `transport_delay_ms not in [0, 500] ms`.
    """
    pc = _require_finite_vec3("p_cam_ned_m", p_cam_ned_m)
    vc = _require_finite_vec3("v_cg_ned_mps", v_cg_ned_mps)
    la = _require_finite_vec3("lever_arm_body_m", lever_arm_body_m)
    roll = _require_finite("roll_rad", roll_rad)
    pitch = _require_finite("pitch_rad", pitch_rad)
    yaw = _require_finite("yaw_rad", yaw_rad)
    delay_ms = _require_finite("transport_delay_ms", transport_delay_ms)

    la_norm = math.sqrt(la[0] * la[0] + la[1] * la[1] + la[2] * la[2])
    if la_norm > MAX_LEVER_ARM_NORM_M + 1e-12:
        raise ValueError(
            f"Camera lever-arm norm {la_norm:.4f} m exceeds physical limit {MAX_LEVER_ARM_NORM_M:.1f} m"
        )
    if delay_ms < 0.0 or delay_ms > MAX_TRANSPORT_DELAY_MS:
        raise ValueError(
            f"transport_delay_ms={delay_ms:.3f} ms out of bounds [0.0, {MAX_TRANSPORT_DELAY_MS:.1f}] ms"
        )

    # Exact intrinsic Z-Y-X DCM: R_ned_body = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    la_n = r00 * la[0] + r01 * la[1] + r02 * la[2]
    la_e = r10 * la[0] + r11 * la[1] + r12 * la[2]
    la_d = r20 * la[0] + r21 * la[1] + r22 * la[2]

    p_cap_n = pc[0] - la_n
    p_cap_e = pc[1] - la_e
    p_cap_d = pc[2] - la_d

    dt_s = delay_ms * 1e-3
    sh_n = vc[0] * dt_s
    sh_e = vc[1] * dt_s
    sh_d = vc[2] * dt_s

    p_now = (p_cap_n + sh_n, p_cap_e + sh_e, p_cap_d + sh_d)
    return CompensatedVisualPoseNed(
        p_cg_ned_m=p_now,
        p_cg_at_capture_ned_m=(p_cap_n, p_cap_e, p_cap_d),
        lever_arm_ned_m=(la_n, la_e, la_d),
        delay_shift_ned_m=(sh_n, sh_e, sh_d),
        lever_arm_norm_m=la_norm,
        transport_delay_ms=delay_ms,
    )


def validate_fixed_wing_avoidance_msgid(msg_id: int) -> int:
    """Enforce `DR-06:171, 193`: hard-reject MAVLink `#84` for fixed-wing lateral avoidance."""
    mid = int(msg_id)
    if mid == MAVLINK_MSG_SET_POSITION_TARGET_LOCAL_NED:
        raise ValueError(
            "MAVLink #84 (SET_POSITION_TARGET_LOCAL_NED) is strictly prohibited for fixed-wing "
            "ArduPlane lateral avoidance (ArduPlane ignores lateral fields; use COMMAND_INT #75 "
            "or SET_ATTITUDE_TARGET #82)."
        )
    if mid not in (MAVLINK_MSG_COMMAND_INT, MAVLINK_MSG_SET_ATTITUDE_TARGET):
        raise ValueError(
            f"Unsupported fixed-wing avoidance MAVLink msg_id={mid}; allowed: "
            f"COMMAND_INT ({MAVLINK_MSG_COMMAND_INT}) or SET_ATTITUDE_TARGET ({MAVLINK_MSG_SET_ATTITUDE_TARGET})."
        )
    return mid


@dataclass(frozen=True)
class CommandIntAvoidance:
    """Envelope-clamped MAVLink `COMMAND_INT (#75)` fixed-wing lateral/vertical avoidance command."""

    msg_id: int
    frame: int
    command: int
    lat_e7: int
    lon_e7: int
    alt_m: float
    airspeed_mps: float
    bank_cmd_rad: float
    bank_clamped: bool
    airspeed_clamped: bool
    payload_bytes: bytes


def build_command_int_avoidance(
    *,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    airspeed_mps: float,
    bank_cmd_rad: float = 0.0,
    v_min_mps: float = DEFAULT_V_MIN_MPS,
    v_max_mps: float = DEFAULT_V_MAX_MPS,
    max_bank_rad: float = DEFAULT_MAX_BANK_RAD,
    mav_frame: int = 6,  # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    mav_cmd: int = MAV_CMD_DO_REPOSITION,
    param2_flags: float = 0.0,
    loiter_radius_m: float | None = None,
) -> CommandIntAvoidance:
    """Build finite-checked, bank/airspeed-clamped `COMMAND_INT (#75)` avoidance payload."""
    lat = _require_finite("lat_deg", lat_deg)
    lon = _require_finite("lon_deg", lon_deg)
    alt = _require_finite("alt_m", alt_m)
    spd = _require_finite("airspeed_mps", airspeed_mps)
    bank = _require_finite("bank_cmd_rad", bank_cmd_rad)
    vmin = _require_finite("v_min_mps", v_min_mps)
    vmax = _require_finite("v_max_mps", v_max_mps)
    bmax = _require_finite("max_bank_rad", max_bank_rad)
    p2 = _require_finite("param2_flags", param2_flags)

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"lat_deg out of bounds [-90, 90]: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lon_deg out of bounds [-180, 180]: {lon}")
    if vmin <= 0.0 or vmax < vmin:
        raise ValueError(f"Invalid airspeed envelope: v_min={vmin}, v_max={vmax}")
    if bmax <= 0.0 or bmax > DEFAULT_MAX_BANK_RAD + 1e-12:
        raise ValueError(f"max_bank_rad must be in (0, 30 deg], got {math.degrees(bmax):.3f} deg")

    clamped_bank = max(-bmax, min(bmax, bank))
    clamped_spd = max(vmin, min(vmax, spd))
    lat_e7 = int(round(lat * 1e7))
    lon_e7 = int(round(lon * 1e7))
    p3 = float(clamped_bank) if loiter_radius_m is None else _require_finite("loiter_radius_m", loiter_radius_m)

    # MAVLink COMMAND_INT payload layout (35 bytes: param1..4 float, x/y int32, z float, command uint16, target_sys/comp, frame, current, autocontinue)
    payload = struct.pack(
        "<ffffiifHBBBBB",
        float(clamped_spd),
        float(p2),
        float(p3),
        0.0,
        lat_e7,
        lon_e7,
        float(alt),
        int(mav_cmd),
        1,
        1,
        int(mav_frame),
        0,
        0,
    )
    return CommandIntAvoidance(
        msg_id=MAVLINK_MSG_COMMAND_INT,
        frame=int(mav_frame),
        command=int(mav_cmd),
        lat_e7=lat_e7,
        lon_e7=lon_e7,
        alt_m=alt,
        airspeed_mps=clamped_spd,
        bank_cmd_rad=clamped_bank,
        bank_clamped=(abs(clamped_bank - bank) > 1e-12),
        airspeed_clamped=(abs(clamped_spd - spd) > 1e-12),
        payload_bytes=payload,
    )


def build_command_int_guided_heading(
    *,
    target_heading_deg: float,
    airspeed_mps: float,
    bank_cmd_rad: float,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    alt_m: float = 0.0,
    heading_type: int = 0,  # 0=HEADING_TYPE_COURSE_OVER_GROUND, 1=HEADING_TYPE_HEADING
    v_min_mps: float = DEFAULT_V_MIN_MPS,
    v_max_mps: float = DEFAULT_V_MAX_MPS,
    max_bank_rad: float = DEFAULT_MAX_BANK_RAD,
    mav_frame: int = 0,  # MAV_FRAME_GLOBAL
) -> CommandIntAvoidance:
    """Build clamped `#75 COMMAND_INT` (`MAV_CMD_GUIDED_CHANGE_HEADING=43002`) fixed-wing turn command."""
    hdg = _require_finite("target_heading_deg", target_heading_deg) % 360.0
    lat = _require_finite("lat_deg", lat_deg)
    lon = _require_finite("lon_deg", lon_deg)
    alt = _require_finite("alt_m", alt_m)
    spd = _require_finite("airspeed_mps", airspeed_mps)
    bank = _require_finite("bank_cmd_rad", bank_cmd_rad)
    vmin = _require_finite("v_min_mps", v_min_mps)
    vmax = _require_finite("v_max_mps", v_max_mps)
    bmax = _require_finite("max_bank_rad", max_bank_rad)

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lat/lon out of bounds: ({lat}, {lon})")
    if int(heading_type) not in (0, 1):
        raise ValueError(f"heading_type must be 0 (COG) or 1 (HEADING), got {heading_type}")
    if vmin <= 0.0 or vmax < vmin:
        raise ValueError(f"Invalid airspeed envelope: v_min={vmin}, v_max={vmax}")
    if bmax <= 0.0 or bmax > DEFAULT_MAX_BANK_RAD + 1e-12:
        raise ValueError(f"max_bank_rad must be in (0, 30 deg], got {math.degrees(bmax):.3f} deg")

    clamped_bank = max(-bmax, min(bmax, bank))
    clamped_spd = max(vmin, min(vmax, spd))
    centripetal_accel_mps2 = max(0.5, G_ACCEL_MPS2 * math.tan(abs(clamped_bank)))
    lat_e7 = int(round(lat * 1e7))
    lon_e7 = int(round(lon * 1e7))

    payload = struct.pack(
        "<ffffiifHBBBBB",
        float(int(heading_type)),
        float(hdg),
        float(centripetal_accel_mps2),
        0.0,
        lat_e7,
        lon_e7,
        float(alt),
        int(MAV_CMD_GUIDED_CHANGE_HEADING),
        1,
        1,
        int(mav_frame),
        0,
        0,
    )
    return CommandIntAvoidance(
        msg_id=MAVLINK_MSG_COMMAND_INT,
        frame=int(mav_frame),
        command=MAV_CMD_GUIDED_CHANGE_HEADING,
        lat_e7=lat_e7,
        lon_e7=lon_e7,
        alt_m=alt,
        airspeed_mps=clamped_spd,
        bank_cmd_rad=clamped_bank,
        bank_clamped=(abs(clamped_bank - bank) > 1e-12),
        airspeed_clamped=(abs(clamped_spd - spd) > 1e-12),
        payload_bytes=payload,
    )


def _euler_ned_to_hamilton_quat(roll_rad: float, pitch_rad: float, yaw_rad: float) -> tuple[float, float, float, float]:
    """Convert ZYX `(yaw, pitch, roll)` in NED to canonical Hamilton `(w, x, y, z)` with `w >= 0`."""
    cr = math.cos(0.5 * roll_rad)
    sr = math.sin(0.5 * roll_rad)
    cp = math.cos(0.5 * pitch_rad)
    sp = math.sin(0.5 * pitch_rad)
    cy = math.cos(0.5 * yaw_rad)
    sy = math.sin(0.5 * yaw_rad)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return (w, x, y, z)


@dataclass(frozen=True)
class SetAttitudeTargetCmd:
    """Envelope-clamped MAVLink `SET_ATTITUDE_TARGET (#82)` fixed-wing avoidance command."""

    msg_id: int
    q_ned_body: tuple[float, float, float, float]
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    thrust_norm: float
    airspeed_mps: float
    bank_clamped: bool
    airspeed_clamped: bool
    payload_bytes: bytes
    thrust_clamped: bool = False
    tecs_protected: bool = False
    type_mask: int = ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES


def build_set_attitude_target(
    *,
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
    thrust_norm: float,
    airspeed_mps: float,
    time_boot_ms: int = 0,
    v_min_mps: float = DEFAULT_V_MIN_MPS,
    v_max_mps: float = DEFAULT_V_MAX_MPS,
    max_bank_rad: float = DEFAULT_MAX_BANK_RAD,
    max_pitch_rad: float = DEFAULT_MAX_PITCH_RAD,
    preserve_tecs_throttle: bool = False,
) -> SetAttitudeTargetCmd:
    """Build finite-checked, bank/pitch/airspeed/stall-protected `SET_ATTITUDE_TARGET (#82)` payload.

    Because MAVLink `#82` carries no wire airspeed field, low/zero thrust during a climb (`pitch > 0`)
    or below `V_min` stalls fixed-wing autopilots (`ArduPlane 4.7.3`). Whenever `pitch > 0` or
    `airspeed < V_min`, this builder enforces a safe minimum wire thrust floor (`>= 0.35..0.65`)
    with `type_mask = 0x07` (`ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES`) so `ArduPlane 4.7.3`
    ACTUALLY applies the safe climb/underspeed thrust (live SITL verified: setting bit `0x40`
    `THROTTLE_IGNORE` causes `ArduPlane 4.7.3` Guided Attitude mode to cut throttle to `0%` and
    stall at `6.7 m/s`, whereas `0x07` + `thrust >= 0.65` holds `>= 18.2 m/s` stall-free).
    Bit `0x40` is only set when a caller explicitly passes `preserve_tecs_throttle=True`.
    """
    roll = _require_finite("roll_rad", roll_rad)
    pitch = _require_finite("pitch_rad", pitch_rad)
    yaw = _require_finite("yaw_rad", yaw_rad)
    thrust = _require_finite("thrust_norm", thrust_norm)
    spd = _require_finite("airspeed_mps", airspeed_mps)
    vmin = _require_finite("v_min_mps", v_min_mps)
    vmax = _require_finite("v_max_mps", v_max_mps)
    bmax = _require_finite("max_bank_rad", max_bank_rad)
    pmax = _require_finite("max_pitch_rad", max_pitch_rad)

    if vmin <= 0.0 or vmax < vmin:
        raise ValueError(f"Invalid airspeed envelope: v_min={vmin}, v_max={vmax}")
    if bmax <= 0.0 or bmax > DEFAULT_MAX_BANK_RAD + 1e-12:
        raise ValueError(f"max_bank_rad must be in (0, 30 deg], got {math.degrees(bmax):.3f} deg")
    if pmax <= 0.0 or pmax > math.radians(45.0):
        raise ValueError(f"max_pitch_rad out of bounds: {pmax}")
    if int(time_boot_ms) < 0:
        raise ValueError(f"time_boot_ms must be non-negative, got {time_boot_ms}")

    c_roll = max(-bmax, min(bmax, roll))
    c_pitch = max(-pmax, min(pmax, pitch))
    raw_clamped_thrust = max(0.0, min(1.0, thrust))
    c_spd = max(vmin, min(vmax, spd))
    underspeed = spd < vmin

    # Fixed-wing stall protection: never allow zero/idle thrust while pitching nose-up or below V_min.
    min_required_thrust = 0.0
    if underspeed:
        min_required_thrust = max(min_required_thrust, MIN_SAFE_THRUST_NORM)
    if c_pitch > 0.0:
        climb_frac = min(1.0, c_pitch / DEFAULT_MAX_PITCH_RAD)
        climb_floor = MIN_SAFE_THRUST_NORM + (MIN_CLIMB_THRUST_NORM - MIN_SAFE_THRUST_NORM) * climb_frac
        min_required_thrust = max(min_required_thrust, climb_floor)

    c_thrust = max(min_required_thrust, raw_clamped_thrust)
    thrust_clamped = abs(c_thrust - thrust) > 1e-12
    # Live ArduPlane 4.7.3 verification proved bit 0x40 (THROTTLE_IGNORE) zeroes throttle in Guided
    # attitude mode; therefore ONLY set bit 0x40 when explicitly requested via `preserve_tecs_throttle=True`,
    # and enforce stall protection on ArduPlane via the active wire `c_thrust` floor (`type_mask = 0x07`).
    tecs_protected = bool(preserve_tecs_throttle)
    type_mask = ATTITUDE_TARGET_TYPEMASK_IGNORE_RATES | (
        ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE if tecs_protected else 0
    )

    q = _euler_ned_to_hamilton_quat(c_roll, c_pitch, yaw)

    # MAVLink SET_ATTITUDE_TARGET payload (39 bytes: time_boot_ms u32, q[4] f32, body_roll/pitch/yaw_rate f32, thrust f32, target_sys/comp, type_mask)
    payload = struct.pack(
        "<I4f3ffBBB",
        int(time_boot_ms) & 0xFFFFFFFF,
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
        0.0,
        0.0,
        0.0,
        float(c_thrust),
        1,
        1,
        int(type_mask),
    )
    return SetAttitudeTargetCmd(
        msg_id=MAVLINK_MSG_SET_ATTITUDE_TARGET,
        q_ned_body=q,
        roll_rad=c_roll,
        pitch_rad=c_pitch,
        yaw_rad=yaw,
        thrust_norm=c_thrust,
        airspeed_mps=c_spd,
        bank_clamped=(abs(c_roll - roll) > 1e-12),
        airspeed_clamped=(abs(c_spd - spd) > 1e-12),
        payload_bytes=payload,
        thrust_clamped=thrust_clamped,
        tecs_protected=tecs_protected,
        type_mask=int(type_mask),
    )


@dataclass(frozen=True)
class Ekf3SwitchState:
    active_source: int
    mav_cmd_id: int | None
    commanded_source: int | None
    healthy_dwell_s: float
    timestamp_s: float


class Ekf3SourceSwitchLatch:
    """ArduPilot EKF3 `42007` (`EK3_SRC1_POSXY`) source-switch hysteresis latch (`>= 2.0 s` dwell)."""

    def __init__(self, *, min_gnss_revert_dwell_s: float = MIN_EKF3_REVERT_DWELL_S) -> None:
        dwell = _require_finite("min_gnss_revert_dwell_s", min_gnss_revert_dwell_s)
        if dwell < MIN_EKF3_REVERT_DWELL_S:
            raise ValueError(
                f"min_gnss_revert_dwell_s must be >= {MIN_EKF3_REVERT_DWELL_S} s, got {dwell}"
            )
        self.min_gnss_revert_dwell_s: float = dwell
        self.active_source: int = EKF3_SOURCE_GNSS
        self._last_ts: float | None = None
        self._healthy_since_ts: float | None = None

    def step(
        self,
        *,
        timestamp_s: float,
        gnss_healthy: bool,
        spoof_alarm: bool,
    ) -> Ekf3SwitchState:
        ts = _require_finite("timestamp_s", timestamp_s)
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError(
                f"Non-monotonic timestamp rejected by Ekf3SourceSwitchLatch: {ts} < {self._last_ts}"
            )
        self._last_ts = ts

        cmd_id: int | None = None
        cmd_src: int | None = None

        if bool(spoof_alarm) or not bool(gnss_healthy):
            self._healthy_since_ts = None
            if self.active_source != EKF3_SOURCE_VISION_OPTICAL:
                self.active_source = EKF3_SOURCE_VISION_OPTICAL
                cmd_id = ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
                cmd_src = EKF3_SOURCE_VISION_OPTICAL
            return Ekf3SwitchState(
                active_source=self.active_source,
                mav_cmd_id=cmd_id,
                commanded_source=cmd_src,
                healthy_dwell_s=0.0,
                timestamp_s=ts,
            )

        # GNSS is healthy and no spoof alarm
        if self.active_source == EKF3_SOURCE_VISION_OPTICAL:
            if self._healthy_since_ts is None:
                self._healthy_since_ts = ts
            dwell = ts - self._healthy_since_ts
            if dwell >= self.min_gnss_revert_dwell_s:
                self.active_source = EKF3_SOURCE_GNSS
                self._healthy_since_ts = None
                cmd_id = ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET
                cmd_src = EKF3_SOURCE_GNSS
            return Ekf3SwitchState(
                active_source=self.active_source,
                mav_cmd_id=cmd_id,
                commanded_source=cmd_src,
                healthy_dwell_s=dwell,
                timestamp_s=ts,
            )

        return Ekf3SwitchState(
            active_source=self.active_source,
            mav_cmd_id=None,
            commanded_source=None,
            healthy_dwell_s=0.0,
            timestamp_s=ts,
        )


@dataclass(frozen=True)
class CommandLongEkf3Switch:
    """Validated `#76 COMMAND_LONG` (`MAV_CMD_SET_EKF_SOURCE_SET=42007`) gated by `Ekf3SourceSwitchLatch`."""

    msg_id: int
    command: int
    source_set: int
    target_system: int
    target_component: int
    confirmation: int
    payload_bytes: bytes


def build_ekf3_source_switch_command_long(
    switch_state: Ekf3SwitchState,
    *,
    target_system: int = 1,
    target_component: int = 1,
    confirmation: int = 0,
) -> CommandLongEkf3Switch:
    """Build `#76 COMMAND_LONG(42007)` strictly gated by `Ekf3SourceSwitchLatch` (>= 2.0 s dwell)."""
    if switch_state.mav_cmd_id != ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET:
        raise ValueError(
            "Ekf3SourceSwitchLatch has not authorized a source switch on this tick "
            f"(mav_cmd_id={switch_state.mav_cmd_id!r}, expected {ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET})"
        )
    if switch_state.commanded_source not in (EKF3_SOURCE_GNSS, EKF3_SOURCE_VISION_OPTICAL):
        raise ValueError(
            f"Unsupported EKF3 commanded_source={switch_state.commanded_source!r}; expected 1 (GNSS) or 2 (Vision)"
        )
    msg = MavCommandLong(
        command=ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
        param1=float(int(switch_state.commanded_source)),
        param2=0.0,
        param3=0.0,
        param4=0.0,
        param5=0.0,
        param6=0.0,
        param7=0.0,
        target_system=int(target_system),
        target_component=int(target_component),
        confirmation=int(confirmation),
    )
    return CommandLongEkf3Switch(
        msg_id=MAVLINK_MSG_COMMAND_LONG,
        command=ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
        source_set=int(switch_state.commanded_source),
        target_system=int(target_system),
        target_component=int(target_component),
        confirmation=int(confirmation),
        payload_bytes=msg.pack_payload(),
    )


def encode_heartbeat_frame(
    *,
    seq: int = 0,
    sysid: int = 255,
    compid: int = 191,
    mav_type: int = 18,
    autopilot: int = 8,
    base_mode: int = 0,
    custom_mode: int = 0,
    system_status: int = 4,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
) -> bytes:
    """Encode a clean-room Zero-GPL MAVLink 2.0 `#0 HEARTBEAT` (`CRC_EXTRA=50`) wire frame."""
    msg = MavHeartbeat(
        custom_mode=int(custom_mode),
        mav_type=int(mav_type),
        autopilot=int(autopilot),
        base_mode=int(base_mode),
        system_status=int(system_status),
        mavlink_version=3,
    )
    return encode_frame(
        msg,
        seq=seq,
        sysid=sysid,
        compid=compid,
        secret_key=secret_key,
        link_id=link_id,
        timestamp_48=timestamp_48,
    )


def encode_command_int_frame(
    cmd: CommandIntAvoidance,
    *,
    seq: int = 0,
    sysid: int = 255,
    compid: int = 191,
    target_system: int = 1,
    target_component: int = 1,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
) -> bytes:
    """Encode an envelope-clamped `CommandIntAvoidance` as a verified MAVLink 2.0 `#75` wire frame."""
    validate_fixed_wing_avoidance_msgid(cmd.msg_id)
    unpacked = MavCommandInt.unpack_payload(cmd.payload_bytes)
    msg = MavCommandInt(
        command=unpacked.command,
        frame=unpacked.frame,
        param1=unpacked.param1,
        param2=unpacked.param2,
        param3=unpacked.param3,
        param4=unpacked.param4,
        x=unpacked.x,
        y=unpacked.y,
        z=unpacked.z,
        target_system=int(target_system),
        target_component=int(target_component),
        current=unpacked.current,
        autocontinue=unpacked.autocontinue,
    )
    return encode_frame(
        msg,
        seq=seq,
        sysid=sysid,
        compid=compid,
        secret_key=secret_key,
        link_id=link_id,
        timestamp_48=timestamp_48,
    )


def encode_set_attitude_target_frame(
    cmd: SetAttitudeTargetCmd,
    *,
    seq: int = 0,
    sysid: int = 255,
    compid: int = 191,
    target_system: int = 1,
    target_component: int = 1,
    time_boot_ms: int | None = None,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
) -> bytes:
    """Encode a clamped `SetAttitudeTargetCmd` as a verified MAVLink 2.0 `#82` wire frame."""
    validate_fixed_wing_avoidance_msgid(cmd.msg_id)
    unpacked = MavSetAttitudeTarget.unpack_payload(cmd.payload_bytes)
    msg = MavSetAttitudeTarget(
        time_boot_ms=unpacked.time_boot_ms if time_boot_ms is None else int(time_boot_ms),
        q=cmd.q_ned_body,
        body_roll_rate=unpacked.body_roll_rate,
        body_pitch_rate=unpacked.body_pitch_rate,
        body_yaw_rate=unpacked.body_yaw_rate,
        thrust=cmd.thrust_norm,
        target_system=int(target_system),
        target_component=int(target_component),
        type_mask=unpacked.type_mask,
    )
    return encode_frame(
        msg,
        seq=seq,
        sysid=sysid,
        compid=compid,
        secret_key=secret_key,
        link_id=link_id,
        timestamp_48=timestamp_48,
    )


def encode_ekf3_source_switch_frame(
    switch_state: Ekf3SwitchState,
    *,
    seq: int = 0,
    sysid: int = 255,
    compid: int = 191,
    target_system: int = 1,
    target_component: int = 1,
    confirmation: int = 0,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
) -> bytes:
    """Encode a latch-gated `#76 COMMAND_LONG(42007)` MAVLink 2.0 wire frame (`CRC_EXTRA=152`)."""
    cmd_obj = build_ekf3_source_switch_command_long(
        switch_state,
        target_system=target_system,
        target_component=target_component,
        confirmation=confirmation,
    )
    msg = MavCommandLong.unpack_payload(cmd_obj.payload_bytes)
    return encode_frame(
        msg,
        seq=seq,
        sysid=sysid,
        compid=compid,
        secret_key=secret_key,
        link_id=link_id,
        timestamp_48=timestamp_48,
    )


def encode_command_long_frame(
    *,
    command: int,
    param1: float = 0.0,
    param2: float = 0.0,
    param3: float = 0.0,
    param4: float = 0.0,
    param5: float = 0.0,
    param6: float = 0.0,
    param7: float = 0.0,
    seq: int = 0,
    sysid: int = 255,
    compid: int = 191,
    target_system: int = 1,
    target_component: int = 1,
    confirmation: int = 0,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
) -> bytes:
    """Encode a generic MAVLink 2.0 `#76 COMMAND_LONG` frame (`42007` gated to `encode_ekf3_source_switch_frame`)."""
    if int(command) == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET:
        raise ValueError(
            "MAV_CMD_SET_EKF_SOURCE_SET (42007) cannot bypass Ekf3SourceSwitchLatch; "
            "call encode_ekf3_source_switch_frame(switch_state, ...) instead."
        )
    msg = MavCommandLong(
        command=int(command),
        param1=float(param1),
        param2=float(param2),
        param3=float(param3),
        param4=float(param4),
        param5=float(param5),
        param6=float(param6),
        param7=float(param7),
        target_system=int(target_system),
        target_component=int(target_component),
        confirmation=int(confirmation),
    )
    return encode_frame(
        msg,
        seq=seq,
        sysid=sysid,
        compid=compid,
        secret_key=secret_key,
        link_id=link_id,
        timestamp_48=timestamp_48,
    )


def decode_arduplane_frame(
    frame_bytes: bytes,
    *,
    secret_key: bytes | None = None,
    require_signature: bool = False,
) -> MavlinkV2Frame:
    """Decode & verify a standard/custom MAVLink 2.0 frame (`#84` hard-banned with ValueError)."""
    return decode_frame(
        frame_bytes,
        verify_inner_crc32=True,
        secret_key=secret_key,
        require_signature=require_signature,
    )


def _mavlink_x25_crc(header_no_stx: bytes, payload: bytes, crc_extra: int) -> int:
    crc = 0xFFFF
    for b in header_no_stx + payload + bytes([crc_extra & 0xFF]):
        tmp = (b ^ (crc & 0xFF)) & 0xFF
        tmp = (tmp ^ ((tmp << 4) & 0xFF)) & 0xFF
        crc = (((crc >> 8) & 0xFF) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc & 0xFFFF


def verify_mavlink_bearing_track_frame(frame: bytes, *, secret_key: bytes) -> dict[str, Any]:
    """Verify exact 77-byte signed MAVLink v2 `#52001` framing, X.25 CRC, and SHA-256 signature."""
    if len(secret_key) != 32:
        raise ValueError("MAVLink v2 secret_key must be exactly 32 bytes")
    if len(frame) != OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN:
        raise ValueError(
            f"Expected {OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN}-byte signed frame, got {len(frame)}"
        )
    if frame[0] != MAVLINK_V2_STX:
        raise ValueError(f"Invalid MAVLink v2 STX byte: 0x{frame[0]:02X}")
    if frame[1] != OPENXVS_BEARING_TRACK_PAYLOAD_LEN:
        raise ValueError(f"Invalid payload len {frame[1]}, expected {OPENXVS_BEARING_TRACK_PAYLOAD_LEN}")
    if (frame[2] & MAVLINK_IFLAG_SIGNED) == 0:
        raise ValueError("MAVLink v2 MAVLINK_IFLAG_SIGNED bit not set")

    msgid = frame[7] | (frame[8] << 8) | (frame[9] << 16)
    if msgid != MAVLINK_MSG_OPENXVS_BEARING_TRACK:
        raise ValueError(f"Unexpected MAVLink msgid={msgid}, expected {MAVLINK_MSG_OPENXVS_BEARING_TRACK}")

    header = frame[:10]
    payload = frame[10:62]
    crc_bytes = frame[62:64]
    sig_hdr = frame[64:71]
    sig_rx = frame[71:77]

    rx_crc = crc_bytes[0] | (crc_bytes[1] << 8)
    calc_crc = _mavlink_x25_crc(header[1:], payload, OPENXVS_BEARING_TRACK_CRC_EXTRA)
    if rx_crc != calc_crc:
        raise ValueError(f"MAVLink X.25 CRC mismatch: rx=0x{rx_crc:04X}, calc=0x{calc_crc:04X}")

    sig_calc = hashlib.sha256(secret_key + header + payload + crc_bytes + sig_hdr).digest()[:6]
    if sig_rx != sig_calc:
        raise ValueError("MAVLink v2 SHA-256 signature verification failed")

    vals = _BEARING_TRACK_STRUCT.unpack(payload)
    return {
        "seq": int(frame[4]),
        "sysid": int(frame[5]),
        "compid": int(frame[6]),
        "msgid": int(msgid),
        "time_utc_us": int(vals[0]),
        "track_id": int(vals[12]),
        "sender_id": int(vals[13]),
        "flags": int(vals[14]),
    }


def run_bearing_track_loopback(
    *,
    n_frames: int = 100,
    secret_key: bytes | None = None,
) -> dict[str, Any]:
    """Send/receive `n_frames` signed `77 B` `#52001` frames over `127.0.0.1` UDP + 1-bit tamper test."""
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    key = bytes(range(32)) if secret_key is None else bytes(secret_key)
    if len(key) != 32:
        raise ValueError("secret_key must be exactly 32 bytes")

    ffi = XvsCoreFFI()
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx_sock.bind(("127.0.0.1", 0))
        rx_sock.settimeout(2.0)
        port = int(rx_sock.getsockname()[1])

        frames_sent = 0
        frames_received = 0
        byte_exact_matches = 0
        crc_sig_verified = 0
        sample_frame: bytes = b""

        for idx in range(n_frames):
            angle = 0.01 * float(idx)
            ux = int(round(math.cos(angle) * 32767.0))
            uy = int(round(math.sin(angle) * 32767.0))
            track = XvsBearingTrack(
                time_utc_us=1_732_150_000_000_000 + idx * 100_000,
                own_ecef_cm=(ctypes.c_int32 * 3)(-270_500_000 + idx * 100, -426_100_000, 387_200_000),
                bearing_ecef_q15=(ctypes.c_int16 * 3)(ux, uy, 0),
                bearing_rate_urad_s=(ctypes.c_int16 * 3)(12, -8, 3),
                cone_half_angle_urad=850,
                own_pos_std_cm=120,
                track_id=1000 + idx,
                sender_id=1 + (idx % 3),
                flags=0x01,
                cov_diag_packed=(ctypes.c_uint16 * 6)(10, 10, 15, 2, 2, 3),
            )
            tx_frame = ffi.encode_mavlink_bearing_track(
                track,
                seq=idx & 0xFF,
                sysid=1 + (idx % 3),
                compid=156,
                link_id=0,
                timestamp_48=1_000_000 + idx,
                secret_key=key,
            )
            if idx == 0:
                sample_frame = tx_frame
            tx_sock.sendto(tx_frame, ("127.0.0.1", port))
            frames_sent += 1

            rx_frame, _ = rx_sock.recvfrom(256)
            frames_received += 1
            if rx_frame == tx_frame and len(rx_frame) == OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN:
                byte_exact_matches += 1
            decoded = verify_mavlink_bearing_track_frame(rx_frame, secret_key=key)
            if decoded["track_id"] == 1000 + idx and decoded["seq"] == (idx & 0xFF):
                crc_sig_verified += 1

        # 1-bit tamper negative tests across payload (byte 20), CRC (byte 62), and SHA-256 sig (byte 74)
        tamper_offsets = (20, 62, 74)
        tamper_rejected = 0
        for off in tamper_offsets:
            corrupted = bytearray(sample_frame)
            corrupted[off] ^= 0x01
            tx_sock.sendto(bytes(corrupted), ("127.0.0.1", port))
            rx_bad, _ = rx_sock.recvfrom(256)
            try:
                verify_mavlink_bearing_track_frame(rx_bad, secret_key=key)
            except ValueError:
                tamper_rejected += 1
    finally:
        rx_sock.close()
        tx_sock.close()

    all_ok = bool(
        frames_sent == n_frames
        and frames_received == n_frames
        and byte_exact_matches == n_frames
        and crc_sig_verified == n_frames
        and tamper_rejected == len(tamper_offsets)
    )
    return {
        "sitl_scope": SITL_DEFER_TAG,
        "frame_size_bytes": OPENXVS_BEARING_TRACK_SIGNED_FRAME_LEN,
        "msgid": MAVLINK_MSG_OPENXVS_BEARING_TRACK,
        "frames_sent": frames_sent,
        "frames_received": frames_received,
        "byte_exact_matches": byte_exact_matches,
        "crc_sig_verified": crc_sig_verified,
        "tamper_cases_tested": len(tamper_offsets),
        "tamper_cases_rejected": tamper_rejected,
        "pass": all_ok,
    }
