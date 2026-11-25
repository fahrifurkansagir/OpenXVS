"""Clean-room Apache-2.0 480 Hz Tier-3 Lockstep Master Clock & ArduPlane 4.7.3+ SIM_JSON Bridge.

Implements `DR-12` §1-§3 (Option C-OpenXVS: Decoupled Gazebo Harmonic >= 8.15.0 +
ArduPlane 4.7.3+ SITL isolated over UDP JSON + headless CUDA/Vulkan 3DGUT/LWIR/EVIS):
- Exact `480 Hz` integer-divisor master clock (`DT_PHYS_S = 1 / 480 s`):
  - `k % 1 == 0`:  480.0 Hz Gazebo Harmonic 6-DoF plant + ArduPlane SIM_JSON exchange
  - `k % 2 == 0`:  240.0 Hz optional IMX636 EVIS event-slice hook
  - `k % 8 == 0`:   60.0 Hz radiometric FLIR Boson+ 640x512 LWIR + 8.0 ms VOx IIR lag
  - `k % 16 == 0`:  30.0 Hz 5x global-shutter IMX296LLR EO + OpenXVS Autonomy barrier
    (mathematically co-stamped with every 2nd LWIR frame; zero fractional phase drift).
- Clean-room ArduPilot `SIM_JSON.h` UDP binary servo packet decoder (`servo_packet_16`
  magic `18458` / 40 B, `servo_packet_32` magic `29569` / 72 B) and newline-terminated
  NED / BODY FRD JSON state encoder (`CONVENTIONS.md` v1.0).
- Fail-closed `DR-12` safety locks:
  - Mandatory finite `airspeed` (Pitot EAS) and `velocity_wind` (3D NED) preventing
    ArduPlane Issue `#29113` 70 m/s phantom wind runaway.
  - Hard rejection of `no_lockstep=True` or `no_time_sync=True` (prevents `AHRS_EKF_TYPE=10`
    EKF3 bypass).
  - 10-second resend-guard duplicate `frame_count` drainer (`frame_count <= last_stepped`).
  - Version locks: `ArduPlane >= (4, 7, 3)` (`#32506` + `#29113`) and `gz-sim >= (8, 15, 0)`
    (`#3695` AdvancedLiftDrag pitching-moment fix).

100% Apache-2.0 clean-room Python stdlib: zero `ardupilot_gazebo` LGPL, zero `pymavlink` GPL,
zero `Isaac Sim` Omniverse, zero `v4l2loopback` GPL.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
import struct
import threading
from typing import Any, Callable, Sequence

SIM_RATE_HZ: int = 480
DT_PHYS_S: float = 1.0 / float(SIM_RATE_HZ)

PLANT_DIVISOR: int = 1
EVIS_DIVISOR: int = 2
LWIR_DIVISOR: int = 8
EO_AUTONOMY_DIVISOR: int = 16

EVIS_RATE_HZ: float = float(SIM_RATE_HZ) / float(EVIS_DIVISOR)              # 240.0 Hz
LWIR_RATE_HZ: float = float(SIM_RATE_HZ) / float(LWIR_DIVISOR)              # 60.0 Hz
EO_AUTONOMY_RATE_HZ: float = float(SIM_RATE_HZ) / float(EO_AUTONOMY_DIVISOR)  # 30.0 Hz

VOX_TAU_S: float = 0.008  # 8.0 ms nominal FLIR Boson+ VOx microbolometer time constant

SERVO_PACKET_16_MAGIC: int = 18458  # 0x481A
SERVO_PACKET_32_MAGIC: int = 29569  # 0x7381
_SERVO_16_STRUCT = struct.Struct("<HHI16H")
_SERVO_32_STRUCT = struct.Struct("<HHI32H")
SERVO_PACKET_16_SIZE: int = _SERVO_16_STRUCT.size  # 40 bytes
SERVO_PACKET_32_SIZE: int = _SERVO_32_STRUCT.size  # 72 bytes

MIN_ARDUPLANE_VERSION: tuple[int, int, int] = (4, 7, 3)
MIN_GZ_SIM_VERSION: tuple[int, int, int] = (8, 15, 0)


class SimJsonProtocolError(ValueError):
    """Raised on malformed SIM_JSON binary servo packets or invalid JSON physics states."""


class Tier3SafetyLockError(PermissionError):
    """Raised when a DR-12 lockstep, wind/airspeed (#29113), or version safety lock trips."""


def validate_arduplane_version(version: tuple[int, int, int]) -> None:
    """Enforce `ArduPlane >= (4, 7, 3)` (`#32506` PosDown fix + `#29113` airspeed/wind fix)."""
    if len(version) != 3:
        raise Tier3SafetyLockError(f"Expected 3-tuple ArduPlane version, got {version!r}")
    norm = (int(version[0]), int(version[1]), int(version[2]))
    if norm < MIN_ARDUPLANE_VERSION:
        raise Tier3SafetyLockError(
            f"ArduPlane {norm[0]}.{norm[1]}.{norm[2]} is blocked for Tier-3 SIM_JSON lockstep "
            f"(requires >= 4.7.3 for Issue #32506 PosDown and Issue #29113 phantom wind fixes)."
        )


def validate_gz_sim_version(version: tuple[int, int, int]) -> None:
    """Enforce `gz-sim >= (8, 15, 0)` (Gazebo Harmonic `#3695` AdvancedLiftDrag fix)."""
    if len(version) != 3:
        raise Tier3SafetyLockError(f"Expected 3-tuple gz-sim version, got {version!r}")
    norm = (int(version[0]), int(version[1]), int(version[2]))
    if norm < MIN_GZ_SIM_VERSION:
        raise Tier3SafetyLockError(
            f"gz-sim {norm[0]}.{norm[1]}.{norm[2]} is blocked for Tier-3 aerodynamics "
            f"(requires >= 8.15.0 for AdvancedLiftDrag pitching-moment fix #3695)."
        )


def _check_finite_vec3(name: str, vec: Sequence[float]) -> tuple[float, float, float]:
    if len(vec) != 3:
        raise SimJsonProtocolError(f"Field '{name}' must have length 3, got {len(vec)}")
    out = (float(vec[0]), float(vec[1]), float(vec[2]))
    for val in out:
        if not math.isfinite(val):
            raise SimJsonProtocolError(f"Field '{name}' contains non-finite value: {out!r}")
    return out


@dataclass(frozen=True)
class LockstepTick:
    """Deterministic `480 Hz` master-clock tick firing flags and exact rational timestamp."""

    tick_index: int
    timestamp_s: float
    plant_due: bool
    evis_due: bool
    lwir_due: bool
    eo_autonomy_due: bool


class MasterLockstepClock:
    """Exact `480 Hz` integer-tick lockstep clock (`timestamp_s = tick_index / 480.0`)."""

    def __init__(self, *, rate_hz: int = SIM_RATE_HZ) -> None:
        if int(rate_hz) != SIM_RATE_HZ:
            raise Tier3SafetyLockError(
                f"DR-12 locks MasterLockstepClock strictly to {SIM_RATE_HZ} Hz (got {rate_hz} Hz)"
            )
        self._tick_index: int = 0
        self._last_timestamp_s: float = 0.0

    @property
    def tick_index(self) -> int:
        return self._tick_index

    @property
    def timestamp_s(self) -> float:
        return self._last_timestamp_s

    def advance(self) -> LockstepTick:
        """Advance by exactly 1 integer physics tick (`+1 / 480.0 s`) with strict monotonicity."""
        next_k = self._tick_index + 1
        next_ts = float(next_k) / float(SIM_RATE_HZ)
        if not (next_ts > self._last_timestamp_s):
            raise Tier3SafetyLockError(
                f"Non-monotonic lockstep timestamp: {next_ts} <= {self._last_timestamp_s}"
            )
        self._tick_index = next_k
        self._last_timestamp_s = next_ts
        return LockstepTick(
            tick_index=next_k,
            timestamp_s=next_ts,
            plant_due=(next_k % PLANT_DIVISOR == 0),
            evis_due=(next_k % EVIS_DIVISOR == 0),
            lwir_due=(next_k % LWIR_DIVISOR == 0),
            eo_autonomy_due=(next_k % EO_AUTONOMY_DIVISOR == 0),
        )


class VoxBolometerIIRFilter:
    """Deterministic 1st-order IIR VOx microbolometer thermal lag filter (`DR-12` §3.3).

    Models FLIR Boson+ `tau_th = 8.0 ms` at `60.000 Hz` (`dt = 1/60 s`):
        `alpha = 1 - exp(-dt / tau_th) ~= 0.8755`
        `y[n] = (1 - alpha) * y[n-1] + alpha * x[n]`
    """

    def __init__(self, *, tau_s: float = VOX_TAU_S, dt_s: float = 1.0 / LWIR_RATE_HZ) -> None:
        if tau_s <= 0.0 or dt_s <= 0.0:
            raise ValueError(f"tau_s ({tau_s}) and dt_s ({dt_s}) must be strictly positive")
        self.tau_s = float(tau_s)
        self.dt_s = float(dt_s)
        self.alpha = 1.0 - math.exp(-self.dt_s / self.tau_s)
        self._state: float | None = None

    @property
    def state(self) -> float | None:
        return self._state

    def reset(self, initial_value: float | None = None) -> None:
        self._state = float(initial_value) if initial_value is not None else None

    def step(self, radiance_in: float) -> float:
        x = float(radiance_in)
        if not math.isfinite(x):
            raise ValueError(f"Non-finite radiance input to VoxBolometerIIRFilter: {x!r}")
        if self._state is None:
            self._state = x
        else:
            self._state = (1.0 - self.alpha) * self._state + self.alpha * x
        return self._state


@dataclass(frozen=True)
class SimJsonServoPacket:
    """Clean-room ArduPilot `SIM_JSON.h` little-endian `servo_packet_16` / `servo_packet_32`."""

    magic: int
    frame_rate: int
    frame_count: int
    pwm: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.magic == SERVO_PACKET_16_MAGIC:
            expected_ch = 16
        elif self.magic == SERVO_PACKET_32_MAGIC:
            expected_ch = 32
        else:
            raise SimJsonProtocolError(
                f"Invalid SIM_JSON servo packet magic: {self.magic} "
                f"(expected {SERVO_PACKET_16_MAGIC} or {SERVO_PACKET_32_MAGIC})"
            )
        if len(self.pwm) != expected_ch:
            raise SimJsonProtocolError(
                f"Magic {self.magic} requires {expected_ch} PWM channels, got {len(self.pwm)}"
            )
        if not (0 <= int(self.frame_rate) <= 0xFFFF):
            raise SimJsonProtocolError(f"frame_rate out of uint16 range: {self.frame_rate}")
        if not (0 <= int(self.frame_count) <= 0xFFFFFFFF):
            raise SimJsonProtocolError(f"frame_count out of uint32 range: {self.frame_count}")
        for idx, ch in enumerate(self.pwm):
            if not (0 <= int(ch) <= 0xFFFF):
                raise SimJsonProtocolError(f"pwm[{idx}] out of uint16 range: {ch}")

    def pack(self) -> bytes:
        if self.magic == SERVO_PACKET_16_MAGIC:
            return _SERVO_16_STRUCT.pack(self.magic, self.frame_rate, self.frame_count, *self.pwm)
        return _SERVO_32_STRUCT.pack(self.magic, self.frame_rate, self.frame_count, *self.pwm)

    @classmethod
    def unpack(cls, raw: bytes) -> SimJsonServoPacket:
        if len(raw) == SERVO_PACKET_16_SIZE:
            fields = _SERVO_16_STRUCT.unpack(raw)
            magic, frame_rate, frame_count = int(fields[0]), int(fields[1]), int(fields[2])
            if magic != SERVO_PACKET_16_MAGIC:
                raise SimJsonProtocolError(
                    f"40-byte SIM_JSON packet has bad magic {magic} (expected {SERVO_PACKET_16_MAGIC})"
                )
            return cls(
                magic=magic,
                frame_rate=frame_rate,
                frame_count=frame_count,
                pwm=tuple(int(v) for v in fields[3:]),
            )
        if len(raw) == SERVO_PACKET_32_SIZE:
            fields = _SERVO_32_STRUCT.unpack(raw)
            magic, frame_rate, frame_count = int(fields[0]), int(fields[1]), int(fields[2])
            if magic != SERVO_PACKET_32_MAGIC:
                raise SimJsonProtocolError(
                    f"72-byte SIM_JSON packet has bad magic {magic} (expected {SERVO_PACKET_32_MAGIC})"
                )
            return cls(
                magic=magic,
                frame_rate=frame_rate,
                frame_count=frame_count,
                pwm=tuple(int(v) for v in fields[3:]),
            )
        raise SimJsonProtocolError(
            f"Invalid SIM_JSON servo packet byte length: {len(raw)} B "
            f"(expected {SERVO_PACKET_16_SIZE} B or {SERVO_PACKET_32_SIZE} B)"
        )


@dataclass(frozen=True)
class SimJsonPhysicsState:
    """Outbound ArduPlane `SIM_JSON` state strictly in NED / BODY FRD (`CONVENTIONS.md` v1.0).

    Enforces `DR-12` Issue `#29113` guard (mandatory `airspeed` and `velocity_wind`) and
    strict lockstep invariants (`no_lockstep=False`, `no_time_sync=False`).
    """

    timestamp: float
    imu_gyro: tuple[float, float, float]
    imu_accel_body: tuple[float, float, float]
    position: tuple[float, float, float]
    attitude: tuple[float, float, float]
    velocity: tuple[float, float, float]
    airspeed: float
    velocity_wind: tuple[float, float, float]
    no_lockstep: bool = False
    no_time_sync: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.timestamp)) or float(self.timestamp) <= 0.0:
            raise SimJsonProtocolError(
                f"SIM_JSON timestamp must be strictly positive finite seconds, got {self.timestamp!r}"
            )
        if self.no_lockstep:
            raise Tier3SafetyLockError(
                "SIM_JSON no_lockstep=True is forbidden by DR-12 (breaks deterministic lockstep)."
            )
        if self.no_time_sync:
            raise Tier3SafetyLockError(
                "SIM_JSON no_time_sync=True is forbidden by DR-12 (forces AHRS_EKF_TYPE=10 EKF3 bypass)."
            )
        if not math.isfinite(float(self.airspeed)) or float(self.airspeed) < 0.0:
            raise Tier3SafetyLockError(
                f"Mandatory Pitot EAS 'airspeed' must be finite >= 0.0 (Issue #29113 guard), got {self.airspeed!r}"
            )
        _check_finite_vec3("imu.gyro", self.imu_gyro)
        _check_finite_vec3("imu.accel_body", self.imu_accel_body)
        _check_finite_vec3("position", self.position)
        _check_finite_vec3("attitude", self.attitude)
        _check_finite_vec3("velocity", self.velocity)
        _check_finite_vec3("velocity_wind", self.velocity_wind)

    def to_payload_dict(self) -> dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "imu": {
                "gyro": [float(self.imu_gyro[0]), float(self.imu_gyro[1]), float(self.imu_gyro[2])],
                "accel_body": [
                    float(self.imu_accel_body[0]),
                    float(self.imu_accel_body[1]),
                    float(self.imu_accel_body[2]),
                ],
            },
            "position": [float(self.position[0]), float(self.position[1]), float(self.position[2])],
            "attitude": [float(self.attitude[0]), float(self.attitude[1]), float(self.attitude[2])],
            "velocity": [float(self.velocity[0]), float(self.velocity[1]), float(self.velocity[2])],
            "airspeed": float(self.airspeed),
            "velocity_wind": [
                float(self.velocity_wind[0]),
                float(self.velocity_wind[1]),
                float(self.velocity_wind[2]),
            ],
        }

    def encode_json_bytes(self) -> bytes:
        """Encode newline-terminated UTF-8 JSON physics state packet for ArduPlane `SIM_JSON`."""
        return (json.dumps(self.to_payload_dict(), separators=(",", ":")) + "\n").encode("utf-8")

    @classmethod
    def decode_json_bytes(cls, raw: bytes) -> SimJsonPhysicsState:
        """Decode and validate a newline-terminated SIM_JSON physics state payload."""
        text = raw.decode("utf-8").strip()
        if not text:
            raise SimJsonProtocolError("Empty SIM_JSON physics state packet")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise SimJsonProtocolError("SIM_JSON physics state must decode to a JSON object")

        if "no_lockstep" in payload and bool(payload["no_lockstep"]):
            raise Tier3SafetyLockError("Received forbidden no_lockstep=True in SIM_JSON state")
        if "no_time_sync" in payload and bool(payload["no_time_sync"]):
            raise Tier3SafetyLockError("Received forbidden no_time_sync=True in SIM_JSON state")

        if "airspeed" not in payload:
            raise Tier3SafetyLockError(
                "Missing mandatory 'airspeed' (Pitot EAS) field — blocked by ArduPlane Issue #29113 guard"
            )
        if "velocity_wind" not in payload:
            raise Tier3SafetyLockError(
                "Missing mandatory 'velocity_wind' (3D NED wind) field — blocked by ArduPlane Issue #29113 guard"
            )

        for req in ("timestamp", "imu", "position", "attitude", "velocity"):
            if req not in payload:
                raise SimJsonProtocolError(f"Missing required SIM_JSON field: '{req}'")

        imu = payload["imu"]
        if not isinstance(imu, dict) or "gyro" not in imu or "accel_body" not in imu:
            raise SimJsonProtocolError("SIM_JSON 'imu' must contain 'gyro' and 'accel_body'")

        return cls(
            timestamp=float(payload["timestamp"]),
            imu_gyro=_check_finite_vec3("imu.gyro", imu["gyro"]),
            imu_accel_body=_check_finite_vec3("imu.accel_body", imu["accel_body"]),
            position=_check_finite_vec3("position", payload["position"]),
            attitude=_check_finite_vec3("attitude", payload["attitude"]),
            velocity=_check_finite_vec3("velocity", payload["velocity"]),
            airspeed=float(payload["airspeed"]),
            velocity_wind=_check_finite_vec3("velocity_wind", payload["velocity_wind"]),
            no_lockstep=False,
            no_time_sync=False,
        )


class SimJsonUdpBridge:
    """UDP SIM_JSON lockstep transport with 10s resend-guard duplicate `frame_count` drainer."""

    def __init__(
        self,
        *,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        arduplane_version: tuple[int, int, int] = MIN_ARDUPLANE_VERSION,
        gz_sim_version: tuple[int, int, int] = MIN_GZ_SIM_VERSION,
        socket_timeout_s: float = 2.0,
    ) -> None:
        validate_arduplane_version(arduplane_version)
        validate_gz_sim_version(gz_sim_version)

        self.arduplane_version = arduplane_version
        self.gz_sim_version = gz_sim_version
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind_host, int(bind_port)))
        self._sock.settimeout(float(socket_timeout_s))
        self._last_stepped_frame_count: int = -1
        self._last_sent_timestamp_s: float = 0.0
        self.duplicate_packets_drained: int = 0

    @property
    def local_address(self) -> tuple[str, int]:
        addr = self._sock.getsockname()
        return str(addr[0]), int(addr[1])

    @property
    def last_stepped_frame_count(self) -> int:
        return self._last_stepped_frame_count

    def close(self) -> None:
        self._sock.close()

    def recv_next_servo_packet(self) -> tuple[SimJsonServoPacket, tuple[str, int]]:
        """Receive next fresh servo packet, draining stale resent duplicates (`frame_count <= last`)."""
        while True:
            raw, peer = self._sock.recvfrom(256)
            pkt = SimJsonServoPacket.unpack(raw)
            if pkt.frame_count <= self._last_stepped_frame_count:
                self.duplicate_packets_drained += 1
                continue
            self._last_stepped_frame_count = pkt.frame_count
            return pkt, (str(peer[0]), int(peer[1]))

    def send_physics_state(self, state: SimJsonPhysicsState, peer: tuple[str, int]) -> int:
        """Send strictly monotonic NED / BODY FRD JSON state back to ArduPlane SITL."""
        if not (state.timestamp > self._last_sent_timestamp_s):
            raise Tier3SafetyLockError(
                f"Non-monotonic SIM_JSON state timestamp {state.timestamp} <= {self._last_sent_timestamp_s}"
            )
        wire = state.encode_json_bytes()
        sent = self._sock.sendto(wire, peer)
        self._last_sent_timestamp_s = state.timestamp
        return sent


class Tier3LockstepOrchestrator:
    """Orchestrates deterministic 480 Hz plant + 240 Hz EVIS + 60 Hz LWIR + 30 Hz EO/Autonomy."""

    def __init__(
        self,
        bridge: SimJsonUdpBridge,
        *,
        vox_tau_s: float = VOX_TAU_S,
        wind_ned_mps: tuple[float, float, float] = (3.2, -1.8, 0.0),
    ) -> None:
        self.bridge = bridge
        self.clock = MasterLockstepClock(rate_hz=SIM_RATE_HZ)
        self.vox_filter = VoxBolometerIIRFilter(tau_s=vox_tau_s, dt_s=1.0 / LWIR_RATE_HZ)
        self.wind_ned_mps = _check_finite_vec3("wind_ned_mps", wind_ned_mps)

        self.plant_steps: int = 0
        self.evis_ticks: int = 0
        self.lwir_ticks: int = 0
        self.eo_autonomy_ticks: int = 0
        self.costamped_eo_lwir_matches: int = 0
        self.max_phase_drift_s: float = 0.0
        self.last_lwir_timestamp_s: float = 0.0
        self.last_eo_timestamp_s: float = 0.0
        self.vox_history: list[float] = []

        # Deterministic KSFO 28L 3-degree glideslope approach state in local NED (m)
        self._pos_ned: list[float] = [-1500.0, 2600.0, -165.0]
        self._vel_ned: list[float] = [17.0, -32.0, 1.9]  # ~36.3 m/s groundspeed, 3 deg descent

    def step_once(
        self,
        *,
        autonomy_barrier_fn: Callable[[LockstepTick, SimJsonPhysicsState], None] | None = None,
    ) -> tuple[LockstepTick, SimJsonPhysicsState]:
        """Execute one deterministic 480 Hz lockstep exchange with ArduPlane SITL."""
        pkt, peer = self.bridge.recv_next_servo_packet()
        tick = self.clock.advance()
        self.plant_steps += 1

        # Mild deterministic aerodynamic response to elevator/aileron PWM around 1500 us
        elev_norm = (float(pkt.pwm[1]) - 1500.0) / 500.0
        ail_norm = (float(pkt.pwm[0]) - 1500.0) / 500.0

        # Integrate 6-DoF NED kinematics over exact DT_PHYS_S = 1 / 480 s
        self._pos_ned[0] += self._vel_ned[0] * DT_PHYS_S
        self._pos_ned[1] += self._vel_ned[1] * DT_PHYS_S
        self._pos_ned[2] += (self._vel_ned[2] + 0.05 * elev_norm) * DT_PHYS_S

        # Compute true airspeed vector v_air = v_ground - v_wind in NED -> Pitot EAS
        v_air_n = self._vel_ned[0] - self.wind_ned_mps[0]
        v_air_e = self._vel_ned[1] - self.wind_ned_mps[1]
        v_air_d = self._vel_ned[2] - self.wind_ned_mps[2]
        airspeed_eas = math.sqrt(v_air_n * v_air_n + v_air_e * v_air_e + v_air_d * v_air_d)

        state = SimJsonPhysicsState(
            timestamp=tick.timestamp_s,
            imu_gyro=(0.02 * ail_norm, 0.015 * elev_norm, 0.0),
            imu_accel_body=(0.05, 0.0, -9.80665),
            position=(self._pos_ned[0], self._pos_ned[1], self._pos_ned[2]),
            attitude=(0.03 * ail_norm, -0.05236 + 0.01 * elev_norm, 5.1958),  # ~297.7 deg true (KSFO 28L)
            velocity=(self._vel_ned[0], self._vel_ned[1], self._vel_ned[2]),
            airspeed=airspeed_eas,
            velocity_wind=self.wind_ned_mps,
        )

        if tick.evis_due:
            self.evis_ticks += 1

        if tick.lwir_due:
            self.lwir_ticks += 1
            self.last_lwir_timestamp_s = tick.timestamp_s
            # Synthetic Boson+ thermal radiance step (cold sky 12.0 -> runway/exhaust target 42.0 W/(m^2*sr))
            target_rad = 12.0 if self.lwir_ticks <= 10 else 42.0
            filtered_rad = self.vox_filter.step(target_rad)
            self.vox_history.append(filtered_rad)

        if tick.eo_autonomy_due:
            self.eo_autonomy_ticks += 1
            self.last_eo_timestamp_s = tick.timestamp_s
            drift = abs(self.last_eo_timestamp_s - self.last_lwir_timestamp_s)
            if drift > self.max_phase_drift_s:
                self.max_phase_drift_s = drift
            if drift == 0.0:
                self.costamped_eo_lwir_matches += 1
            if autonomy_barrier_fn is not None:
                autonomy_barrier_fn(tick, state)

        self.bridge.send_physics_state(state, peer)
        return tick, state


def run_tier3_udp_lockstep_loopback(
    *,
    num_ticks: int = SIM_RATE_HZ,
    use_servo_32: bool = False,
    inject_duplicate_frame_at: int | None = 120,
    arduplane_version: tuple[int, int, int] = MIN_ARDUPLANE_VERSION,
    gz_sim_version: tuple[int, int, int] = MIN_GZ_SIM_VERSION,
) -> dict[str, Any]:
    """Run a deterministic real local `127.0.0.1` UDP closed-loop lockstep flight harness."""
    bridge = SimJsonUdpBridge(
        bind_host="127.0.0.1",
        bind_port=0,
        arduplane_version=arduplane_version,
        gz_sim_version=gz_sim_version,
        socket_timeout_s=3.0,
    )
    orchestrator = Tier3LockstepOrchestrator(bridge)
    server_addr = bridge.local_address

    sitl_errors: list[Exception] = []
    sitl_received_timestamps: list[float] = []
    sitl_all_have_airspeed_and_wind = True

    def _mock_arduplane_sitl_peer() -> None:
        nonlocal sitl_all_have_airspeed_and_wind
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(3.0)
        try:
            magic = SERVO_PACKET_32_MAGIC if use_servo_32 else SERVO_PACKET_16_MAGIC
            n_ch = 32 if use_servo_32 else 16
            for frame_idx in range(1, int(num_ticks) + 1):
                pwm = tuple(1500 + ((frame_idx + ch) % 7) for ch in range(n_ch))
                pkt = SimJsonServoPacket(
                    magic=magic,
                    frame_rate=SIM_RATE_HZ,
                    frame_count=frame_idx,
                    pwm=pwm,
                )
                wire = pkt.pack()
                client.sendto(wire, server_addr)

                # Simulate ArduPilot 10-second resend guard injecting a duplicate frame_count
                if inject_duplicate_frame_at is not None and frame_idx == int(inject_duplicate_frame_at):
                    client.sendto(wire, server_addr)

                reply_bytes, _ = client.recvfrom(4096)
                decoded = SimJsonPhysicsState.decode_json_bytes(reply_bytes)
                sitl_received_timestamps.append(decoded.timestamp)
                if not (decoded.airspeed >= 0.0 and len(decoded.velocity_wind) == 3):
                    sitl_all_have_airspeed_and_wind = False
        except Exception as exc:
            sitl_errors.append(exc)
        finally:
            client.close()

    peer_thread = threading.Thread(target=_mock_arduplane_sitl_peer, daemon=True)
    peer_thread.start()
    try:
        for _ in range(int(num_ticks)):
            orchestrator.step_once()
        peer_thread.join(timeout=5.0)
    finally:
        bridge.close()

    if sitl_errors:
        raise sitl_errors[0]
    if peer_thread.is_alive():
        raise TimeoutError("Mock ArduPlane SITL UDP peer thread did not exit cleanly")

    strictly_monotonic = all(
        sitl_received_timestamps[i] > sitl_received_timestamps[i - 1]
        for i in range(1, len(sitl_received_timestamps))
    )
    return {
        "sim_rate_hz": SIM_RATE_HZ,
        "plant_steps": orchestrator.plant_steps,
        "final_timestamp_s": round(orchestrator.clock.timestamp_s, 9),
        "evis_ticks": orchestrator.evis_ticks,
        "lwir_ticks": orchestrator.lwir_ticks,
        "eo_autonomy_ticks": orchestrator.eo_autonomy_ticks,
        "costamped_eo_lwir_matches": orchestrator.costamped_eo_lwir_matches,
        "max_phase_drift_s": orchestrator.max_phase_drift_s,
        "duplicate_packets_drained": bridge.duplicate_packets_drained,
        "vox_alpha": round(orchestrator.vox_filter.alpha, 6),
        "vox_final_radiance": round(orchestrator.vox_history[-1], 6) if orchestrator.vox_history else 0.0,
        "sitl_received_count": len(sitl_received_timestamps),
        "timestamps_strictly_monotonic": bool(strictly_monotonic),
        "all_states_have_airspeed_and_wind": bool(sitl_all_have_airspeed_and_wind),
    }
