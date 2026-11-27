"""Clean-Room Zero-GPL PX4 MAVLink 2.0 External Vision Dialect (`#102` & `#331`).

Strictly ZERO GPL dependencies (`pymavlink` strictly forbidden; reuses clean-room
Apache-2.0/MIT X.25 CRC-16/MCRF4XX, header/frame extraction, and non-flight safety
lockouts from `xvs.v2.studio_bridge.hil_bridge` and `xvs.v2.xvs_mavlink.pod_daa_codec`).

Supported PX4 MAVLink 2.0 messages (`common.xml` wire layout):
  - `#102 VISION_POSITION_ESTIMATE` (`CRC_EXTRA = 158`):
      Strictly Local NED position `(x_north_m, y_east_m, z_down_m)` in meters,
      Body FRD orientation `(roll_rad, pitch_rad, yaw_rad)` in radians,
      21-float upper-triangular 6x6 pose covariance, and `reset_counter`.
  - `#331 ODOMETRY` (`CRC_EXTRA = 91`):
      Strictly `MAV_FRAME_LOCAL_NED = 1` (`frame_id`) and
      `MAV_FRAME_BODY_FRD = 12` (`child_frame_id`). ENU (`MAV_FRAME_LOCAL_ENU = 4`),
      FLU, or ANY unknown frame is rejected fail-closed (`Px4FrameViolationError`).

Safety & Regulatory Invariants:
  - Enforces `NON_FLIGHT_SIM_ONLY=1` / `OPENXVS_NON_FLIGHT_SIM_ONLY=1` lockout on
    outbound PX4 frame encoding and UDP transmission (`NonFlightSimOnlyError`).
  - Rejects `NaN`, `+Inf`, `-Inf`, `bool`, non-unit quaternions, and negative
    covariance variances fail-closed on BOTH encode and decode.
  - Strictly NED / FRD (`0` internal ENU). Strictly civil/dual-use scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import socket
import struct
import sys
from typing import Any, Sequence

from xvs.v2.studio_bridge.hil_bridge import (
    EU_PLD_ART4_7_LOCKOUT_MSG,
    MAVLINK_V2_STX,
    NON_FLIGHT_ENV_ALIAS,
    NON_FLIGHT_ENV_VAR,
    NON_FLIGHT_WATERMARK,
    NonFlightSimOnlyError,
    _require_finite_scalar,
    _validate_local_or_private_host,
    extract_mavlink_v2_frames,
    is_non_flight_sim_only,
    peek_mavlink_v2_header,
)
from xvs.v2.xvs_mavlink.pod_daa_codec import (
    MAVLINK_IFLAG_SIGNED,
    MAVLINK_V2_CHECKSUM_LEN,
    MAVLINK_V2_HEADER_LEN,
    MAVLINK_V2_OVERHEAD_BYTES,
    MAVLINK_V2_SIGNATURE_BYTES,
    MavlinkCrcError,
    MavlinkFrameError,
    _compute_mavlink_v2_signature,
    _truncate_mavlink_v2_payload,
    compute_crc_extra,
    x25_crc16,
)

# ---- Canonical MAVLink 2.0 `common.xml` IDs & Frame Constants ----

MSGID_VISION_POSITION_ESTIMATE: int = 102
MSGID_ODOMETRY: int = 331

# MAV_FRAME enum values (strictly NED + FRD permitted; ENU / FLU / others hard-rejected)
MAV_FRAME_LOCAL_NED: int = 1
MAV_FRAME_LOCAL_ENU: int = 4  # Explicitly forbidden!
MAV_FRAME_BODY_FRD: int = 12

# MAV_ESTIMATOR_TYPE enum values (common.xml)
MAV_ESTIMATOR_TYPE_UNKNOWN: int = 0
MAV_ESTIMATOR_TYPE_NAIVE: int = 1
MAV_ESTIMATOR_TYPE_VISION: int = 3
MAV_ESTIMATOR_TYPE_VIO: int = 4
MAV_ESTIMATOR_TYPE_GPS: int = 5
MAV_ESTIMATOR_TYPE_GPS_INS: int = 6

# Upper-triangular 6x6 row-major diagonal indices:
# Row 0: [0..5] -> diag 0; Row 1: [6..10] -> diag 6; Row 2: [11..14] -> diag 11;
# Row 3: [15..17] -> diag 15; Row 4: [18..19] -> diag 18; Row 5: [20] -> diag 20.
COV21_LEN: int = 21
COV21_DIAG_INDICES: tuple[int, ...] = (0, 6, 11, 15, 18, 20)
QUATERNION_NORM_TOL: float = 1e-2

# Canonical MAVLink `common.xml` base fields (before `<extensions/>`) used for CRC_EXTRA
_VISION_POSITION_ESTIMATE_BASE_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint64_t", "usec", 0),
    ("float", "x", 0),
    ("float", "y", 0),
    ("float", "z", 0),
    ("float", "roll", 0),
    ("float", "pitch", 0),
    ("float", "yaw", 0),
)

_ODOMETRY_BASE_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("uint64_t", "time_usec", 0),
    ("uint8_t", "frame_id", 0),
    ("uint8_t", "child_frame_id", 0),
    ("float", "x", 0),
    ("float", "y", 0),
    ("float", "z", 0),
    ("float", "q", 4),
    ("float", "vx", 0),
    ("float", "vy", 0),
    ("float", "vz", 0),
    ("float", "rollspeed", 0),
    ("float", "pitchspeed", 0),
    ("float", "yawspeed", 0),
    ("float", "pose_covariance", 21),
    ("float", "velocity_covariance", 21),
)

CRC_EXTRA_VISION_POSITION_ESTIMATE: int = compute_crc_extra(
    "VISION_POSITION_ESTIMATE", _VISION_POSITION_ESTIMATE_BASE_FIELDS
)
CRC_EXTRA_ODOMETRY: int = compute_crc_extra("ODOMETRY", _ODOMETRY_BASE_FIELDS)

# Verify exact canonical MAVLink `common.xml` CRC_EXTRA values mathematically at import time.
assert CRC_EXTRA_VISION_POSITION_ESTIMATE == 158, (
    f"Expected VISION_POSITION_ESTIMATE CRC_EXTRA=158, got {CRC_EXTRA_VISION_POSITION_ESTIMATE}"
)
assert CRC_EXTRA_ODOMETRY == 91, (
    f"Expected ODOMETRY CRC_EXTRA=91, got {CRC_EXTRA_ODOMETRY}"
)

PX4_CRC_EXTRA_BY_MSGID: dict[int, int] = {
    MSGID_VISION_POSITION_ESTIMATE: CRC_EXTRA_VISION_POSITION_ESTIMATE,
    MSGID_ODOMETRY: CRC_EXTRA_ODOMETRY,
}

# Canonical wire structs (base fields sorted by descending primitive size; MAVLink v2
# extension fields appended in XML declaration order):
# #102: base = <Q6f (32 B); ext = 21f (covariance, +84 B = 116 B) + B (reset_counter, +1 B = 117 B)
_VPE_BASE_STRUCT = struct.Struct("<Q6f")
_VPE_FULL_STRUCT = struct.Struct("<Q6f21fB")
assert _VPE_BASE_STRUCT.size == 32
assert _VPE_FULL_STRUCT.size == 117

# #331: base = <Q (8 B) + 55f (220 B) + BB (frame_id, child_frame_id, 2 B) = 230 B;
#       ext = B (reset_counter) + B (estimator_type) + b (quality) = +3 B -> 233 B
_ODOM_BASE_STRUCT = struct.Struct("<Q55fBB")
_ODOM_FULL_STRUCT = struct.Struct("<Q55fBBBBb")
assert _ODOM_BASE_STRUCT.size == 230
assert _ODOM_FULL_STRUCT.size == 233

PX4_WIRE_PAYLOAD_LEN_BY_MSGID: dict[int, int] = {
    MSGID_VISION_POSITION_ESTIMATE: _VPE_FULL_STRUCT.size,
    MSGID_ODOMETRY: _ODOM_FULL_STRUCT.size,
}


class Px4FrameViolationError(ValueError):
    """Raised fail-closed when a non-NED / non-FRD (e.g., ENU/FLU/unknown) frame is supplied."""


def assert_zero_pymavlink() -> None:
    """Fail-closed audit guard asserting GPL `pymavlink` is NOT imported in the active process."""
    if "pymavlink" in sys.modules:
        raise RuntimeError("GPL pymavlink MUST NEVER be imported in OpenXVS PX4 MAVLink2 module")


def require_px4_tx_allowed(*, non_flight_sim_only: bool | None = None) -> None:
    """Enforce `NON_FLIGHT_SIM_ONLY=1` / `OPENXVS_NON_FLIGHT_SIM_ONLY=1` fail-closed lockout."""
    locked = is_non_flight_sim_only() if non_flight_sim_only is None else bool(non_flight_sim_only)
    if locked:
        raise NonFlightSimOnlyError(
            f"PX4 MAVLink2 pose/odometry TX blocked ({NON_FLIGHT_ENV_VAR} / {NON_FLIGHT_ENV_ALIAS}=1): "
            f"{EU_PLD_ART4_7_LOCKOUT_MSG}"
        )


def _require_uint(name: str, value: int, *, max_val: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer for {name!r}, got {type(value).__name__}: {value!r}")
    if value < 0 or value > max_val:
        raise ValueError(f"Integer field {name!r}={value} out of bounds [0, {max_val}]")
    return value


def _require_int8(name: str, value: int, *, min_val: int = -128, max_val: int = 127) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer for {name!r}, got {type(value).__name__}: {value!r}")
    if value < min_val or value > max_val:
        raise ValueError(f"Signed int8 field {name!r}={value} out of bounds [{min_val}, {max_val}]")
    return value


def _validate_cov21(name: str, cov: Sequence[float]) -> tuple[float, ...]:
    try:
        seq = tuple(cov)
    except TypeError as exc:
        raise ValueError(f"Expected 21-element covariance sequence for {name!r}, got {cov!r}") from exc
    if len(seq) != COV21_LEN:
        raise ValueError(f"Covariance {name!r} must have exactly {COV21_LEN} floats, got {len(seq)}")
    out: list[float] = []
    for idx, val in enumerate(seq):
        fval = _require_finite_scalar(f"{name}[{idx}]", val)
        if idx in COV21_DIAG_INDICES and fval < 0.0:
            raise ValueError(
                f"Diagonal variance {name}[{idx}] must be >= 0.0, got {fval}"
            )
        out.append(fval)
    return tuple(out)


def _validate_unit_quaternion(q: Sequence[float]) -> tuple[float, float, float, float]:
    try:
        seq = tuple(q)
    except TypeError as exc:
        raise ValueError(f"Expected 4-element Hamilton quaternion (w, x, y, z), got {q!r}") from exc
    if len(seq) != 4:
        raise ValueError(f"Hamilton quaternion must have exactly 4 elements, got {len(seq)}")
    qw = _require_finite_scalar("q[0]", seq[0])
    qx = _require_finite_scalar("q[1]", seq[1])
    qy = _require_finite_scalar("q[2]", seq[2])
    qz = _require_finite_scalar("q[3]", seq[3])
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or abs(norm - 1.0) > QUATERNION_NORM_TOL:
        raise ValueError(
            f"Hamilton quaternion must have unit norm within {QUATERNION_NORM_TOL}, got ||q||={norm:.6f}"
        )
    return (qw, qx, qy, qz)


def make_upper_tri_6x6_covariance(
    var_0: float,
    var_1: float,
    var_2: float,
    var_3: float,
    var_4: float,
    var_5: float,
) -> tuple[float, ...]:
    """Build a finite diagonal 21-float upper-triangular 6x6 covariance array (strictly >= 0)."""
    diags = (
        _require_finite_scalar("var_0", var_0),
        _require_finite_scalar("var_1", var_1),
        _require_finite_scalar("var_2", var_2),
        _require_finite_scalar("var_3", var_3),
        _require_finite_scalar("var_4", var_4),
        _require_finite_scalar("var_5", var_5),
    )
    for idx, v in enumerate(diags):
        if v < 0.0:
            raise ValueError(f"Diagonal variance var_{idx} must be >= 0.0, got {v}")
    arr = [0.0] * COV21_LEN
    for d_idx, val in zip(COV21_DIAG_INDICES, diags):
        arr[d_idx] = val
    return tuple(arr)


def euler_frd_to_quaternion_ned(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> tuple[float, float, float, float]:
    """Convert FRD intrinsic Z-Y-X Euler angles (rad) to canonical Hamilton quaternion `(w,x,y,z)` (`w >= 0`)."""
    r = _require_finite_scalar("roll_rad", roll_rad)
    p = _require_finite_scalar("pitch_rad", pitch_rad)
    y = _require_finite_scalar("yaw_rad", yaw_rad)
    cr, sr = math.cos(0.5 * r), math.sin(0.5 * r)
    cp, sp = math.cos(0.5 * p), math.sin(0.5 * p)
    cy, sy = math.cos(0.5 * y), math.sin(0.5 * y)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("Degenerate Euler->Quaternion conversion")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    if qw < 0.0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return (qw, qx, qy, qz)


def validate_px4_odometry_frames(frame_id: int, child_frame_id: int) -> None:
    """Enforce strictly `MAV_FRAME_LOCAL_NED (1)` + `MAV_FRAME_BODY_FRD (12)`; reject ENU/unknown fail-closed."""
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        raise Px4FrameViolationError(f"Invalid frame_id type: {frame_id!r}")
    if isinstance(child_frame_id, bool) or not isinstance(child_frame_id, int):
        raise Px4FrameViolationError(f"Invalid child_frame_id type: {child_frame_id!r}")
    if frame_id != MAV_FRAME_LOCAL_NED:
        hint = " (ENU strictly forbidden)" if frame_id == MAV_FRAME_LOCAL_ENU else ""
        raise Px4FrameViolationError(
            f"PX4 #331 ODOMETRY frame_id={frame_id} rejected{hint}; "
            f"strictly MAV_FRAME_LOCAL_NED ({MAV_FRAME_LOCAL_NED}) required."
        )
    if child_frame_id != MAV_FRAME_BODY_FRD:
        raise Px4FrameViolationError(
            f"PX4 #331 ODOMETRY child_frame_id={child_frame_id} rejected; "
            f"strictly MAV_FRAME_BODY_FRD ({MAV_FRAME_BODY_FRD}) required."
        )


@dataclass(frozen=True)
class Px4VisionPositionEstimate:
    """MAVLink 2.0 `#102 VISION_POSITION_ESTIMATE` — strictly Local NED (m) + FRD Euler (rad)."""

    usec: int
    x_north_m: float
    y_east_m: float
    z_down_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    covariance: tuple[float, ...] = (0.0,) * COV21_LEN
    reset_counter: int = 0

    MSGID: int = MSGID_VISION_POSITION_ESTIMATE
    NAME: str = "VISION_POSITION_ESTIMATE"
    CRC_EXTRA: int = CRC_EXTRA_VISION_POSITION_ESTIMATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "usec", _require_uint("usec", self.usec, max_val=(1 << 64) - 1))
        object.__setattr__(self, "x_north_m", _require_finite_scalar("x_north_m", self.x_north_m))
        object.__setattr__(self, "y_east_m", _require_finite_scalar("y_east_m", self.y_east_m))
        object.__setattr__(self, "z_down_m", _require_finite_scalar("z_down_m", self.z_down_m))
        object.__setattr__(self, "roll_rad", _require_finite_scalar("roll_rad", self.roll_rad))
        object.__setattr__(self, "pitch_rad", _require_finite_scalar("pitch_rad", self.pitch_rad))
        object.__setattr__(self, "yaw_rad", _require_finite_scalar("yaw_rad", self.yaw_rad))
        object.__setattr__(self, "covariance", _validate_cov21("covariance", self.covariance))
        object.__setattr__(
            self, "reset_counter", _require_uint("reset_counter", self.reset_counter, max_val=255)
        )

    @property
    def x(self) -> float:
        return self.x_north_m

    @property
    def y(self) -> float:
        return self.y_east_m

    @property
    def z(self) -> float:
        return self.z_down_m

    @property
    def roll(self) -> float:
        return self.roll_rad

    @property
    def pitch(self) -> float:
        return self.pitch_rad

    @property
    def yaw(self) -> float:
        return self.yaw_rad

    def pack_payload(self) -> bytes:
        return _VPE_FULL_STRUCT.pack(
            self.usec,
            self.x_north_m,
            self.y_east_m,
            self.z_down_m,
            self.roll_rad,
            self.pitch_rad,
            self.yaw_rad,
            *self.covariance,
            self.reset_counter,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> Px4VisionPositionEstimate:
        if len(raw_padded) != _VPE_FULL_STRUCT.size:
            raise MavlinkFrameError(
                f"VISION_POSITION_ESTIMATE padded payload must be {_VPE_FULL_STRUCT.size} B, got {len(raw_padded)} B"
            )
        vals = _VPE_FULL_STRUCT.unpack(raw_padded)
        return cls(
            usec=int(vals[0]),
            x_north_m=float(vals[1]),
            y_east_m=float(vals[2]),
            z_down_m=float(vals[3]),
            roll_rad=float(vals[4]),
            pitch_rad=float(vals[5]),
            yaw_rad=float(vals[6]),
            covariance=tuple(float(v) for v in vals[7:28]),
            reset_counter=int(vals[28]),
        )


@dataclass(frozen=True)
class Px4Odometry:
    """MAVLink 2.0 `#331 ODOMETRY` — strictly `MAV_FRAME_LOCAL_NED (1)` + `MAV_FRAME_BODY_FRD (12)`."""

    time_usec: int
    x_north_m: float
    y_east_m: float
    z_down_m: float
    q: tuple[float, float, float, float]
    vx_mps: float
    vy_mps: float
    vz_mps: float
    rollspeed_rad_s: float = 0.0
    pitchspeed_rad_s: float = 0.0
    yawspeed_rad_s: float = 0.0
    pose_covariance: tuple[float, ...] = (0.0,) * COV21_LEN
    velocity_covariance: tuple[float, ...] = (0.0,) * COV21_LEN
    frame_id: int = MAV_FRAME_LOCAL_NED
    child_frame_id: int = MAV_FRAME_BODY_FRD
    reset_counter: int = 0
    estimator_type: int = MAV_ESTIMATOR_TYPE_VIO
    quality: int = 100

    MSGID: int = MSGID_ODOMETRY
    NAME: str = "ODOMETRY"
    CRC_EXTRA: int = CRC_EXTRA_ODOMETRY

    def __post_init__(self) -> None:
        validate_px4_odometry_frames(self.frame_id, self.child_frame_id)
        object.__setattr__(
            self, "time_usec", _require_uint("time_usec", self.time_usec, max_val=(1 << 64) - 1)
        )
        object.__setattr__(self, "x_north_m", _require_finite_scalar("x_north_m", self.x_north_m))
        object.__setattr__(self, "y_east_m", _require_finite_scalar("y_east_m", self.y_east_m))
        object.__setattr__(self, "z_down_m", _require_finite_scalar("z_down_m", self.z_down_m))
        object.__setattr__(self, "q", _validate_unit_quaternion(self.q))
        object.__setattr__(self, "vx_mps", _require_finite_scalar("vx_mps", self.vx_mps))
        object.__setattr__(self, "vy_mps", _require_finite_scalar("vy_mps", self.vy_mps))
        object.__setattr__(self, "vz_mps", _require_finite_scalar("vz_mps", self.vz_mps))
        object.__setattr__(
            self, "rollspeed_rad_s", _require_finite_scalar("rollspeed_rad_s", self.rollspeed_rad_s)
        )
        object.__setattr__(
            self,
            "pitchspeed_rad_s",
            _require_finite_scalar("pitchspeed_rad_s", self.pitchspeed_rad_s),
        )
        object.__setattr__(
            self, "yawspeed_rad_s", _require_finite_scalar("yawspeed_rad_s", self.yawspeed_rad_s)
        )
        object.__setattr__(
            self, "pose_covariance", _validate_cov21("pose_covariance", self.pose_covariance)
        )
        object.__setattr__(
            self,
            "velocity_covariance",
            _validate_cov21("velocity_covariance", self.velocity_covariance),
        )
        object.__setattr__(
            self, "reset_counter", _require_uint("reset_counter", self.reset_counter, max_val=255)
        )
        object.__setattr__(
            self,
            "estimator_type",
            _require_uint("estimator_type", self.estimator_type, max_val=255),
        )
        object.__setattr__(
            self, "quality", _require_int8("quality", self.quality, min_val=-1, max_val=100)
        )

    @property
    def x(self) -> float:
        return self.x_north_m

    @property
    def y(self) -> float:
        return self.y_east_m

    @property
    def z(self) -> float:
        return self.z_down_m

    @property
    def vx(self) -> float:
        return self.vx_mps

    @property
    def vy(self) -> float:
        return self.vy_mps

    @property
    def vz(self) -> float:
        return self.vz_mps

    def pack_payload(self) -> bytes:
        return _ODOM_FULL_STRUCT.pack(
            self.time_usec,
            self.x_north_m,
            self.y_east_m,
            self.z_down_m,
            self.q[0],
            self.q[1],
            self.q[2],
            self.q[3],
            self.vx_mps,
            self.vy_mps,
            self.vz_mps,
            self.rollspeed_rad_s,
            self.pitchspeed_rad_s,
            self.yawspeed_rad_s,
            *self.pose_covariance,
            *self.velocity_covariance,
            self.frame_id,
            self.child_frame_id,
            self.reset_counter,
            self.estimator_type,
            self.quality,
        )

    @classmethod
    def unpack_payload(cls, raw_padded: bytes) -> Px4Odometry:
        if len(raw_padded) != _ODOM_FULL_STRUCT.size:
            raise MavlinkFrameError(
                f"ODOMETRY padded payload must be {_ODOM_FULL_STRUCT.size} B, got {len(raw_padded)} B"
            )
        vals = _ODOM_FULL_STRUCT.unpack(raw_padded)
        return cls(
            time_usec=int(vals[0]),
            x_north_m=float(vals[1]),
            y_east_m=float(vals[2]),
            z_down_m=float(vals[3]),
            q=(float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])),
            vx_mps=float(vals[8]),
            vy_mps=float(vals[9]),
            vz_mps=float(vals[10]),
            rollspeed_rad_s=float(vals[11]),
            pitchspeed_rad_s=float(vals[12]),
            yawspeed_rad_s=float(vals[13]),
            pose_covariance=tuple(float(v) for v in vals[14:35]),
            velocity_covariance=tuple(float(v) for v in vals[35:56]),
            frame_id=int(vals[56]),
            child_frame_id=int(vals[57]),
            reset_counter=int(vals[58]),
            estimator_type=int(vals[59]),
            quality=int(vals[60]),
        )


Px4MavlinkMessage = Px4VisionPositionEstimate | Px4Odometry

_PX4_MSG_CLASS_BY_ID: dict[int, type[Any]] = {
    MSGID_VISION_POSITION_ESTIMATE: Px4VisionPositionEstimate,
    MSGID_ODOMETRY: Px4Odometry,
}


@dataclass(frozen=True)
class Px4MavlinkV2Frame:
    """Decoded PX4 MAVLink 2.0 frame (`#102` or `#331`) with verified CRC-16/X.25 & optional signature."""

    seq: int
    sysid: int
    compid: int
    msgid: int
    incompat_flags: int
    compat_flags: int
    payload_len: int
    crc16: int
    crc_extra: int
    message: Px4MavlinkMessage
    signed: bool = False
    link_id: int | None = None
    timestamp_48: int | None = None
    signature_verified: bool = False


def encode_px4_frame(
    message: Px4MavlinkMessage,
    *,
    seq: int = 0,
    sysid: int = 1,
    compid: int = 197,  # MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
    incompat_flags: int = 0,
    compat_flags: int = 0,
    truncate_zeros: bool = True,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
    non_flight_sim_only: bool | None = None,
) -> bytes:
    """Encode a PX4 `#102` or `#331` message into a verified MAVLink 2.0 binary frame.

    Enforces `NON_FLIGHT_SIM_ONLY=1` fail-closed lockout and zero `pymavlink` guarantee.
    """
    assert_zero_pymavlink()
    require_px4_tx_allowed(non_flight_sim_only=non_flight_sim_only)

    if not isinstance(message, (Px4VisionPositionEstimate, Px4Odometry)):
        raise MavlinkFrameError(
            f"Unsupported PX4 MAVLink2 message type: {type(message).__name__}"
        )
    msgid = int(message.MSGID)
    if msgid not in PX4_CRC_EXTRA_BY_MSGID:
        raise MavlinkFrameError(f"Unsupported PX4 message ID: {msgid}")

    eff_incompat = int(incompat_flags) & 0xFF
    if secret_key is not None:
        eff_incompat |= MAVLINK_IFLAG_SIGNED
    if (eff_incompat & ~MAVLINK_IFLAG_SIGNED) != 0:
        raise MavlinkFrameError(
            f"Unsupported MAVLink v2 incompat_flags: 0x{eff_incompat:02x} (only 0x00 or 0x01)"
        )
    signed = (eff_incompat & MAVLINK_IFLAG_SIGNED) != 0
    if signed and secret_key is None:
        raise MavlinkFrameError(
            "MAVLink v2 incompat_flags=0x01 (MAVLINK_IFLAG_SIGNED) requires a 32-byte secret_key"
        )

    full_payload = message.pack_payload()
    wire_payload = _truncate_mavlink_v2_payload(full_payload) if truncate_zeros else full_payload
    if len(wire_payload) > 255:
        raise MavlinkFrameError(
            f"PX4 payload length {len(wire_payload)} exceeds MAVLink v2 max 255 B"
        )

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
    crc_extra = PX4_CRC_EXTRA_BY_MSGID[msgid]
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


def decode_px4_frame(
    frame_bytes: bytes,
    *,
    secret_key: bytes | None = None,
    require_signature: bool = False,
) -> Px4MavlinkV2Frame:
    """Decode and CRC/frame/finiteness-verify a MAVLink 2.0 PX4 `#102` or `#331` wire frame.

    Listen-only RX decoding is permitted under `NON_FLIGHT_SIM_ONLY=1` while strictly
    enforcing X.25 CRC-16 + `CRC_EXTRA`, NED/FRD frames, unit quaternion, and finite floats.
    """
    assert_zero_pymavlink()
    if len(frame_bytes) < MAVLINK_V2_OVERHEAD_BYTES:
        raise MavlinkFrameError(
            f"Frame too short ({len(frame_bytes)} B < {MAVLINK_V2_OVERHEAD_BYTES} B minimum)"
        )
    hdr = peek_mavlink_v2_header(frame_bytes)
    payload_len = hdr["payload_len"]
    incompat_flags = hdr["incompat_flags"]
    compat_flags = hdr["compat_flags"]
    seq = hdr["seq"]
    sysid = hdr["sysid"]
    compid = hdr["compid"]
    msgid = hdr["msgid"]
    signed = bool(hdr["signed"])

    if (incompat_flags & ~MAVLINK_IFLAG_SIGNED) != 0:
        raise MavlinkFrameError(
            f"Unsupported MAVLink v2 incompat_flags: 0x{incompat_flags:02x}"
        )
    if require_signature and not signed:
        raise MavlinkFrameError("Unsigned PX4 MAVLink v2 frame rejected when require_signature=True")
    if len(frame_bytes) != hdr["total_len"]:
        raise MavlinkFrameError(
            f"Frame length mismatch: got {len(frame_bytes)} B, header declares {hdr['total_len']} B"
        )
    if msgid not in _PX4_MSG_CLASS_BY_ID:
        raise MavlinkFrameError(
            f"Unsupported PX4 MAVLink2 message ID {msgid} (expected #102 or #331)"
        )

    wire_payload = frame_bytes[MAVLINK_V2_HEADER_LEN : MAVLINK_V2_HEADER_LEN + payload_len]
    crc_off = MAVLINK_V2_HEADER_LEN + payload_len
    rx_crc = struct.unpack_from("<H", frame_bytes, crc_off)[0]
    crc_extra = PX4_CRC_EXTRA_BY_MSGID[msgid]
    expected_crc = x25_crc16(
        frame_bytes[1:MAVLINK_V2_HEADER_LEN] + wire_payload + bytes([crc_extra])
    )
    if rx_crc != expected_crc:
        raise MavlinkCrcError(
            f"PX4 MAVLink v2 CRC mismatch for msgid={msgid} (crc_extra={crc_extra}): "
            f"rx=0x{rx_crc:04x}, expected=0x{expected_crc:04x}"
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
                raise ValueError(
                    f"MAVLink v2 signing secret_key must be exactly 32 bytes, got {len(secret_key)}"
                )
            sig_calc = hashlib.sha256(secret_key + frame_bytes[:sig_start] + sig_hdr).digest()[:6]
            if not hmac.compare_digest(sig_rx, sig_calc):
                raise MavlinkCrcError(f"PX4 MAVLink v2 SHA-256 signature mismatch for msgid={msgid}")
            signature_verified = True
        elif require_signature:
            raise ValueError("secret_key is required when require_signature=True")

    full_wire_size = PX4_WIRE_PAYLOAD_LEN_BY_MSGID[msgid]
    if payload_len > full_wire_size:
        raise MavlinkFrameError(
            f"Payload length {payload_len} exceeds PX4 schema wire size {full_wire_size} for msgid={msgid}"
        )
    padded = wire_payload + bytes(full_wire_size - payload_len)
    message = _PX4_MSG_CLASS_BY_ID[msgid].unpack_payload(padded)
    return Px4MavlinkV2Frame(
        seq=seq,
        sysid=sysid,
        compid=compid,
        msgid=msgid,
        incompat_flags=incompat_flags,
        compat_flags=compat_flags,
        payload_len=payload_len,
        crc16=rx_crc,
        crc_extra=crc_extra,
        message=message,
        signed=signed,
        link_id=link_id,
        timestamp_48=timestamp_48,
        signature_verified=signature_verified,
    )


def encode_vision_position_estimate_frame(
    *,
    usec: int,
    x_north_m: float,
    y_east_m: float,
    z_down_m: float,
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
    covariance: Sequence[float] = (0.0,) * COV21_LEN,
    reset_counter: int = 0,
    seq: int = 0,
    sysid: int = 1,
    compid: int = 197,
    truncate_zeros: bool = True,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
    non_flight_sim_only: bool | None = None,
) -> bytes:
    """Convenience constructor + MAVLink 2.0 encoder for `#102 VISION_POSITION_ESTIMATE`."""
    msg = Px4VisionPositionEstimate(
        usec=usec,
        x_north_m=x_north_m,
        y_east_m=y_east_m,
        z_down_m=z_down_m,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        covariance=tuple(covariance),
        reset_counter=reset_counter,
    )
    return encode_px4_frame(
        msg,
        seq=seq,
        sysid=sysid,
        compid=compid,
        truncate_zeros=truncate_zeros,
        secret_key=secret_key,
        link_id=link_id,
        timestamp_48=timestamp_48,
        non_flight_sim_only=non_flight_sim_only,
    )


def encode_odometry_frame(
    *,
    time_usec: int,
    x_north_m: float,
    y_east_m: float,
    z_down_m: float,
    q: Sequence[float],
    vx_mps: float,
    vy_mps: float,
    vz_mps: float,
    rollspeed_rad_s: float = 0.0,
    pitchspeed_rad_s: float = 0.0,
    yawspeed_rad_s: float = 0.0,
    pose_covariance: Sequence[float] = (0.0,) * COV21_LEN,
    velocity_covariance: Sequence[float] = (0.0,) * COV21_LEN,
    frame_id: int = MAV_FRAME_LOCAL_NED,
    child_frame_id: int = MAV_FRAME_BODY_FRD,
    reset_counter: int = 0,
    estimator_type: int = MAV_ESTIMATOR_TYPE_VIO,
    quality: int = 100,
    seq: int = 0,
    sysid: int = 1,
    compid: int = 197,
    truncate_zeros: bool = True,
    secret_key: bytes | None = None,
    link_id: int = 0,
    timestamp_48: int = 0,
    non_flight_sim_only: bool | None = None,
) -> bytes:
    """Convenience constructor + MAVLink 2.0 encoder for `#331 ODOMETRY` (strictly NED + FRD)."""
    msg = Px4Odometry(
        time_usec=time_usec,
        x_north_m=x_north_m,
        y_east_m=y_east_m,
        z_down_m=z_down_m,
        q=(q[0], q[1], q[2], q[3]) if len(tuple(q)) == 4 else tuple(q),  # type: ignore[arg-type]
        vx_mps=vx_mps,
        vy_mps=vy_mps,
        vz_mps=vz_mps,
        rollspeed_rad_s=rollspeed_rad_s,
        pitchspeed_rad_s=pitchspeed_rad_s,
        yawspeed_rad_s=yawspeed_rad_s,
        pose_covariance=tuple(pose_covariance),
        velocity_covariance=tuple(velocity_covariance),
        frame_id=frame_id,
        child_frame_id=child_frame_id,
        reset_counter=reset_counter,
        estimator_type=estimator_type,
        quality=quality,
    )
    return encode_px4_frame(
        msg,
        seq=seq,
        sysid=sysid,
        compid=compid,
        truncate_zeros=truncate_zeros,
        secret_key=secret_key,
        link_id=link_id,
        timestamp_48=timestamp_48,
        non_flight_sim_only=non_flight_sim_only,
    )


class Px4MavlinkUdpSender:
    """Loopback/LAN-only PX4 MAVLink 2.0 UDP transmitter enforcing `NON_FLIGHT_SIM_ONLY=1`."""

    def __init__(
        self,
        *,
        target_host: str = "127.0.0.1",
        target_port: int = 14540,
        sysid: int = 1,
        compid: int = 197,
        non_flight_sim_only: bool | None = None,
    ) -> None:
        assert_zero_pymavlink()
        self._target_host = _validate_local_or_private_host(target_host, allow_any_bind=False)
        self._target_port = _require_uint("target_port", int(target_port), max_val=65535)
        if self._target_port == 0:
            raise ValueError("target_port must be in [1, 65535]")
        self._sysid = _require_uint("sysid", int(sysid), max_val=255)
        self._compid = _require_uint("compid", int(compid), max_val=255)
        self._non_flight_sim_only = non_flight_sim_only
        self._seq: int = 0
        self._sock: socket.socket | None = None
        self.watermark: str = NON_FLIGHT_WATERMARK

    def send(self, message: Px4MavlinkMessage, *, secret_key: bytes | None = None) -> bytes:
        require_px4_tx_allowed(non_flight_sim_only=self._non_flight_sim_only)
        frame = encode_px4_frame(
            message,
            seq=self._seq,
            sysid=self._sysid,
            compid=self._compid,
            secret_key=secret_key,
            non_flight_sim_only=self._non_flight_sim_only,
        )
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.sendto(frame, (self._target_host, self._target_port))
        self._seq = (self._seq + 1) & 0xFF
        return frame

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> Px4MavlinkUdpSender:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = [
    "COV21_DIAG_INDICES",
    "COV21_LEN",
    "CRC_EXTRA_ODOMETRY",
    "CRC_EXTRA_VISION_POSITION_ESTIMATE",
    "MAV_ESTIMATOR_TYPE_GPS",
    "MAV_ESTIMATOR_TYPE_GPS_INS",
    "MAV_ESTIMATOR_TYPE_NAIVE",
    "MAV_ESTIMATOR_TYPE_UNKNOWN",
    "MAV_ESTIMATOR_TYPE_VIO",
    "MAV_ESTIMATOR_TYPE_VISION",
    "MAV_FRAME_BODY_FRD",
    "MAV_FRAME_LOCAL_ENU",
    "MAV_FRAME_LOCAL_NED",
    "MSGID_ODOMETRY",
    "MSGID_VISION_POSITION_ESTIMATE",
    "NON_FLIGHT_WATERMARK",
    "NonFlightSimOnlyError",
    "PX4_CRC_EXTRA_BY_MSGID",
    "PX4_WIRE_PAYLOAD_LEN_BY_MSGID",
    "Px4FrameViolationError",
    "Px4MavlinkMessage",
    "Px4MavlinkUdpSender",
    "Px4MavlinkV2Frame",
    "Px4Odometry",
    "Px4VisionPositionEstimate",
    "assert_zero_pymavlink",
    "decode_px4_frame",
    "encode_odometry_frame",
    "encode_px4_frame",
    "encode_vision_position_estimate_frame",
    "euler_frd_to_quaternion_ned",
    "extract_mavlink_v2_frames",
    "make_upper_tri_6x6_covariance",
    "require_px4_tx_allowed",
    "validate_px4_odometry_frames",
]
