"""Zero-GPL MAVLink 2.0 + FlightGear Generic UDP HIL Bridge & 1-Click Loopback Proof (`OpenXVS v4.5`).

Strictly ZERO GPL dependencies (`pymavlink` strictly forbidden; reuses clean-room Apache-2.0/MIT
codecs from `xvs.v2.xvs_rt_bridge`, `xvs.v2.xvs_mavlink`, and `xvs.v2.xvs_active_range`).
Strictly NED coordinates (`north`, `east`, `down`) and localhost `127.0.0.1` UDP binding only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Any

from xvs.v2.xvs_active_range.filter import (
    EVADE_BANK_RAD,
    WIRE_SIZE_BYTES,
    ActiveRangeController,
    ActiveRangeMeasurement,
    ActiveRangeMode,
    ActiveRangeTelemetry,
    OwnshipStateNed,
    decode_wire,
    encode_wire,
)
from xvs.v2.xvs_mavlink.pod_daa_codec import (
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
    MavCommandInt,
    MavCommandLong,
    MavGlobalPositionInt,
    MavHeartbeat,
    MavSetAttitudeTarget,
    MavVfrHud,
    PodCivilMissionClass,
    PodPluginHealth,
    PodPluginLifecycleState,
    decode_frame,
    encode_frame,
    extract_mavlink_v2_frames,
    peek_mavlink_v2_header,
)
from xvs.v2.xvs_rt_bridge.arduplane_bridge import (
    ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
    DEFAULT_MAX_BANK_RAD,
    DEFAULT_MAX_PITCH_RAD,
    DEFAULT_V_MAX_MPS,
    DEFAULT_V_MIN_MPS,
    EKF3_SOURCE_GNSS,
    EKF3_SOURCE_VISION_OPTICAL,
    MAX_LEVER_ARM_NORM_M,
    MAX_TRANSPORT_DELAY_MS,
    MAV_CMD_DO_REPOSITION,
    MAV_CMD_GUIDED_CHANGE_HEADING,
    MAVLINK_MSG_COMMAND_INT,
    MAVLINK_MSG_SET_ATTITUDE_TARGET,
    MAVLINK_MSG_SET_POSITION_TARGET_LOCAL_NED,
    CompensatedVisualPoseNed,
    Ekf3SourceSwitchLatch,
    Ekf3SwitchState,
    build_command_int_avoidance,
    build_command_int_guided_heading,
    build_set_attitude_target,
    compensate_visual_pose_ned,
    encode_command_int_frame,
    encode_ekf3_source_switch_frame,
    encode_set_attitude_target_frame,
    run_bearing_track_loopback,
    validate_fixed_wing_avoidance_msgid,
)

DEFAULT_FG_RX_PORT: int = 5500
DEFAULT_FG_TX_PORT: int = 5501
DEFAULT_MAV_RX_PORT: int = 14550
DEFAULT_MAV_TX_PORT: int = 14540
VALID_HIL_MODES: tuple[str, ...] = ("idle", "flightgear_udp", "mavlink2_udp")
VALID_TX_COMMAND_MODES: tuple[str, ...] = ("set_attitude_target", "command_int", "command_int_heading")
DEFAULT_STALE_TELEMETRY_TIMEOUT_S: float = 2.0
UDP_RX_BUFFER_BYTES: int = 65535
MAX_ACCEPTED_UDP_DATAGRAM_BYTES: int = 4096

_BOOT_MS_UINT32_MOD: int = 1 << 32
_BOOT_MS_UINT32_HALF: int = 1 << 31
_BOOT_MS_WRAP_WINDOW_MS: int = 60_000
_BOOT_MS_REBOOT_DROP_MS: int = 30_000
_BOOT_MS_REBOOT_CONFIRM_TICKS: int = 3
_BOOT_MS_REBOOT_MAX_STEP_MS: int = 5_000

NON_FLIGHT_ENV_VAR: str = "OPENXVS_NON_FLIGHT_SIM_ONLY"
NON_FLIGHT_ENV_ALIAS: str = "NON_FLIGHT_SIM_ONLY"
NON_FLIGHT_WATERMARK: str = (
    "SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT"
)
NON_FLIGHT_WATERMARK_ASCII: str = (
    "SIMULATION & BENCHMARK EVALUATION ONLY -- ZERO ACTUATION -- NOT FOR FLIGHT"
)
EU_PLD_ART4_7_LOCKOUT_MSG: str = (
    "Outbound MAVLink/FlightGear actuation is blocked under OPENXVS_NON_FLIGHT_SIM_ONLY=1 "
    "(EU PLD Directive (EU) 2024/2853 Art. 4(7) non-flight evaluation lockout — zero actuation authority)."
)
_EXPLICIT_DEV_UNLOCK_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})


class NonFlightSimOnlyError(PermissionError):
    """Raised fail-closed when flight/HIL actuation is attempted under OPENXVS_NON_FLIGHT_SIM_ONLY=1."""


class ShadowModeTxInhibitedError(NonFlightSimOnlyError):
    """Raised fail-closed when outbound TX or actuator authority is attempted in shadow mode."""


class StaleTelemetryError(RuntimeError):
    """Raised fail-closed when outbound TX is attempted while RX telemetry is stale."""


# ---- WGS-84 HAE <-> EGM96/GEOID18 MSL Datum Alignment Guard (Failure Mode 8) ----

MAX_ABS_GEOID_UNDULATION_M: float = 110.0
DEFAULT_MAX_PRE_SWITCH_VERT_INNOV_M: float = 3.0
KSFO_28L_GEOID18_UNDULATION_M: float = -32.63380295376555
KSFO_28L_THR_NAVD88_MSL_M: float = 3.9624
KSFO_28L_THR_WGS84_HAE_M: float = -28.671402953765554

MSL_VERTICAL_DATUMS: frozenset[str] = frozenset({"MSL", "MSL_EGM96", "MSL_GEOID18"})
VALID_VERTICAL_DATUMS: frozenset[str] = frozenset({"WGS84_HAE", "MSL", "MSL_EGM96", "MSL_GEOID18"})

DATUM_REASON_OK: str = "OK"
DATUM_REASON_MISMATCH_INHIBITED: str = "DATUM_MISMATCH_INHIBITED"
DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED: str = "PRE_SWITCH_VERT_INNOV_EXCEEDED"

SHADOW_MODE_LOG_SCHEMA: str = "openxvs.shadow_mode_log/1.0"
SHADOW_GENESIS_HASH: str = "0" * 64
VALID_SHADOW_TRIAL_PHASES: tuple[str, ...] = (
    "phase2_rooftop_car_roof",
    "phase3_university_shadow_tx_pin_removed",
    "bench_shadow_replay",
)
ALLOWED_SHADOW_DIRECTIONS: frozenset[str] = frozenset({"rx", "internal"})


def _require_finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Boolean value rejected for numeric field {name!r}: {value!r}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite value rejected for {name!r}: {value!r}")
    return val


def _normalize_vertical_datum(datum: str) -> str:
    norm = str(datum).strip().upper()
    if norm in ("HAE", "WGS84", "WGS-84", "WGS84_ELLIPSOIDAL", "WGS84_HAE"):
        return "WGS84_HAE"
    if norm in ("MSL", "AMSL"):
        return "MSL"
    if norm in ("EGM96", "MSL_EGM96", "MSL/EGM96"):
        return "MSL_EGM96"
    if norm in ("GEOID18", "NAVD88", "MSL_GEOID18", "MSL/GEOID18"):
        return "MSL_GEOID18"
    return norm


def validate_geoid_undulation_m(n_geoid_m: float) -> float:
    """Validate explicit geoid undulation `N_geoid_m` fail-closed (`finite` and `|N| <= 110 m`)."""
    val = _require_finite_scalar("geoid_undulation_m", n_geoid_m)
    if abs(val) > MAX_ABS_GEOID_UNDULATION_M:
        raise ValueError(
            f"Geoid undulation |N_geoid_m|={abs(val):.3f} m exceeds physical Earth limit "
            f"{MAX_ABS_GEOID_UNDULATION_M:.1f} m"
        )
    return val


def wgs84_hae_to_msl(alt_hae_m: float, n_geoid_m: float) -> float:
    """Convert WGS-84 ellipsoidal height `h` to orthometric MSL `H = h - N_geoid` (fail-closed)."""
    h_val = _require_finite_scalar("alt_hae_m", alt_hae_m)
    n_val = validate_geoid_undulation_m(n_geoid_m)
    return h_val - n_val


def msl_to_wgs84_hae(alt_msl_m: float, n_geoid_m: float) -> float:
    """Convert orthometric MSL height `H` to WGS-84 ellipsoidal height `h = H + N_geoid` (fail-closed)."""
    msl_val = _require_finite_scalar("alt_msl_m", alt_msl_m)
    n_val = validate_geoid_undulation_m(n_geoid_m)
    return msl_val + n_val


@dataclass(frozen=True)
class DatumSwitchDecision:
    """Fail-closed pre-switch vertical datum alignment decision for `EKF3 42007` visual handoff."""

    allowed: bool
    reason_code: str
    detail: str
    visual_datum: str
    baro_datum: str
    geoid_undulation_m: float | None
    z_visual_input_m: float | None
    z_visual_msl_m: float | None
    z_baro_msl_m: float | None
    vertical_innovation_m: float | None
    abs_vertical_innovation_m: float | None
    max_pre_switch_vert_innov_m: float
    ned_down_visual_msl_m: float | None
    ned_down_baro_msl_m: float | None
    datum_conversion_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatumAlignmentGuard:
    """Fail-closed `WGS-84 HAE <-> EGM96/GEOID18 MSL` guard gating `EKF3 42007` visual handoffs.

    Prevents Failure Mode 8 (`REALWORLD_DEPLOYABILITY_AUDIT_V5.md`): mixing WGS-84 ellipsoidal
    height (e.g. KSFO 28L `h = -28.67 m`) with MSL/EGM96 barometric altitude (`H = +3.96 m`)
    during an `EKF3 42007` source switch causes an immediate `~29-33 m` vertical innovation step.
    """

    def __init__(
        self,
        *,
        geoid_undulation_m: float | None = None,
        max_pre_switch_vert_innov_m: float = DEFAULT_MAX_PRE_SWITCH_VERT_INNOV_M,
        baro_datum: str = "MSL_EGM96",
        visual_datum: str = "WGS84_HAE",
    ) -> None:
        max_innov = _require_finite_scalar("max_pre_switch_vert_innov_m", max_pre_switch_vert_innov_m)
        if max_innov <= 0.0:
            raise ValueError(
                f"max_pre_switch_vert_innov_m must be finite > 0.0 m, got {max_pre_switch_vert_innov_m!r}"
            )
        norm_baro = _normalize_vertical_datum(baro_datum)
        norm_vis = _normalize_vertical_datum(visual_datum)
        if norm_baro not in VALID_VERTICAL_DATUMS:
            raise ValueError(f"Unsupported baro_datum {baro_datum!r}; expected one of {sorted(VALID_VERTICAL_DATUMS)}")
        if norm_vis not in VALID_VERTICAL_DATUMS:
            raise ValueError(
                f"Unsupported visual_datum {visual_datum!r}; expected one of {sorted(VALID_VERTICAL_DATUMS)}"
            )
        self._geoid_undulation_m: float | None = (
            validate_geoid_undulation_m(geoid_undulation_m) if geoid_undulation_m is not None else None
        )
        self._max_pre_switch_vert_innov_m: float = max_innov
        self._baro_datum: str = norm_baro
        self._visual_datum: str = norm_vis

    @property
    def geoid_undulation_m(self) -> float | None:
        return self._geoid_undulation_m

    @property
    def max_pre_switch_vert_innov_m(self) -> float:
        return self._max_pre_switch_vert_innov_m

    @property
    def baro_datum(self) -> str:
        return self._baro_datum

    @property
    def visual_datum(self) -> str:
        return self._visual_datum

    def set_geoid_undulation(self, n_geoid_m: float) -> float:
        """Update configured geoid undulation `N_geoid_m` fail-closed."""
        validated = validate_geoid_undulation_m(n_geoid_m)
        self._geoid_undulation_m = validated
        return validated

    def hae_to_msl(self, alt_hae_m: float, *, geoid_undulation_m: float | None = None) -> float:
        """Convert WGS-84 HAE altitude to orthometric MSL using explicit or configured `N_geoid_m`."""
        n_eff = self._geoid_undulation_m if geoid_undulation_m is None else geoid_undulation_m
        if n_eff is None:
            raise ValueError("Explicit geoid_undulation_m is required to convert WGS84_HAE to MSL")
        return wgs84_hae_to_msl(alt_hae_m, n_eff)

    def msl_to_hae(self, alt_msl_m: float, *, geoid_undulation_m: float | None = None) -> float:
        """Convert orthometric MSL altitude to WGS-84 HAE using explicit or configured `N_geoid_m`."""
        n_eff = self._geoid_undulation_m if geoid_undulation_m is None else geoid_undulation_m
        if n_eff is None:
            raise ValueError("Explicit geoid_undulation_m is required to convert MSL to WGS84_HAE")
        return msl_to_wgs84_hae(alt_msl_m, n_eff)

    def evaluate_ekf3_switch(
        self,
        *,
        z_visual_m: float | None,
        z_baro_msl_m: float,
        visual_datum: str | None = None,
        baro_datum: str | None = None,
        geoid_undulation_m: float | None = None,
        apply_geoid_conversion: bool = True,
    ) -> DatumSwitchDecision:
        """Validate vertical datum alignment and pre-switch innovation before `EKF3 42007` switch."""
        v_datum = self._visual_datum if visual_datum is None else _normalize_vertical_datum(visual_datum)
        b_datum = self._baro_datum if baro_datum is None else _normalize_vertical_datum(baro_datum)

        if v_datum not in VALID_VERTICAL_DATUMS or b_datum not in VALID_VERTICAL_DATUMS:
            return DatumSwitchDecision(
                allowed=False,
                reason_code=DATUM_REASON_MISMATCH_INHIBITED,
                detail=f"Unsupported vertical datum pair (visual={v_datum!r}, baro={b_datum!r})",
                visual_datum=v_datum,
                baro_datum=b_datum,
                geoid_undulation_m=self._geoid_undulation_m,
                z_visual_input_m=None,
                z_visual_msl_m=None,
                z_baro_msl_m=None,
                vertical_innovation_m=None,
                abs_vertical_innovation_m=None,
                max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
                ned_down_visual_msl_m=None,
                ned_down_baro_msl_m=None,
                datum_conversion_applied=False,
            )

        if z_visual_m is None:
            baro_ok: float | None = None
            try:
                baro_ok = _require_finite_scalar("z_baro_msl_m", z_baro_msl_m)
            except (TypeError, ValueError):
                baro_ok = None
            baro_is_msl_local = baro_ok is not None and b_datum in MSL_VERTICAL_DATUMS
            return DatumSwitchDecision(
                allowed=False,
                reason_code=DATUM_REASON_MISMATCH_INHIBITED,
                detail=(
                    "Visual altitude unavailable while DatumAlignmentGuard is attached; "
                    "EKF3 42007 inhibited fail-closed"
                ),
                visual_datum=v_datum,
                baro_datum=b_datum,
                geoid_undulation_m=self._geoid_undulation_m,
                z_visual_input_m=None,
                z_visual_msl_m=None,
                z_baro_msl_m=round(baro_ok, 6) if baro_is_msl_local and baro_ok is not None else None,
                vertical_innovation_m=None,
                abs_vertical_innovation_m=None,
                max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
                ned_down_visual_msl_m=None,
                ned_down_baro_msl_m=round(-baro_ok, 6) if baro_is_msl_local and baro_ok is not None else None,
                datum_conversion_applied=False,
            )

        try:
            z_vis_in = _require_finite_scalar("z_visual_m", z_visual_m)
            z_baro_in = _require_finite_scalar("z_baro_msl_m", z_baro_msl_m)
        except (TypeError, ValueError) as exc:
            return DatumSwitchDecision(
                allowed=False,
                reason_code=DATUM_REASON_MISMATCH_INHIBITED,
                detail=f"Non-finite pre-switch altitude rejected fail-closed: {exc}",
                visual_datum=v_datum,
                baro_datum=b_datum,
                geoid_undulation_m=self._geoid_undulation_m,
                z_visual_input_m=None,
                z_visual_msl_m=None,
                z_baro_msl_m=None,
                vertical_innovation_m=None,
                abs_vertical_innovation_m=None,
                max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
                ned_down_visual_msl_m=None,
                ned_down_baro_msl_m=None,
                datum_conversion_applied=False,
            )

        # Validate geoid undulation if explicitly supplied on this call
        n_eff: float | None = self._geoid_undulation_m
        if geoid_undulation_m is not None:
            try:
                n_eff = validate_geoid_undulation_m(geoid_undulation_m)
            except ValueError as exc:
                return DatumSwitchDecision(
                    allowed=False,
                    reason_code=DATUM_REASON_MISMATCH_INHIBITED,
                    detail=f"Invalid geoid undulation rejected fail-closed: {exc}",
                    visual_datum=v_datum,
                    baro_datum=b_datum,
                    geoid_undulation_m=None,
                    z_visual_input_m=z_vis_in,
                    z_visual_msl_m=None,
                    z_baro_msl_m=z_baro_in if b_datum in MSL_VERTICAL_DATUMS else None,
                    vertical_innovation_m=None,
                    abs_vertical_innovation_m=None,
                    max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
                    ned_down_visual_msl_m=None,
                    ned_down_baro_msl_m=-z_baro_in if b_datum in MSL_VERTICAL_DATUMS else None,
                    datum_conversion_applied=False,
                )

        vis_is_msl = v_datum in MSL_VERTICAL_DATUMS
        baro_is_msl = b_datum in MSL_VERTICAL_DATUMS
        conversion_applied = False

        if vis_is_msl and baro_is_msl:
            z_vis_msl = z_vis_in
            z_baro_msl = z_baro_in
        elif (not vis_is_msl) and baro_is_msl:
            if not apply_geoid_conversion:
                raw_diff = z_vis_in - z_baro_in
                return DatumSwitchDecision(
                    allowed=False,
                    reason_code=DATUM_REASON_MISMATCH_INHIBITED,
                    detail=(
                        f"Raw unaligned datum mix inhibited ({v_datum} vs {b_datum}, "
                        f"raw vertical step {raw_diff:+.3f} m); explicit geoid conversion required"
                    ),
                    visual_datum=v_datum,
                    baro_datum=b_datum,
                    geoid_undulation_m=n_eff,
                    z_visual_input_m=z_vis_in,
                    z_visual_msl_m=None,
                    z_baro_msl_m=z_baro_in,
                    vertical_innovation_m=round(raw_diff, 6),
                    abs_vertical_innovation_m=round(abs(raw_diff), 6),
                    max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
                    ned_down_visual_msl_m=None,
                    ned_down_baro_msl_m=round(-z_baro_in, 6),
                    datum_conversion_applied=False,
                )
            if n_eff is None:
                return DatumSwitchDecision(
                    allowed=False,
                    reason_code=DATUM_REASON_MISMATCH_INHIBITED,
                    detail=(
                        f"Datum mismatch ({v_datum} vs {b_datum}) with missing geoid undulation "
                        "N_geoid_m; EKF3 42007 switch inhibited fail-closed"
                    ),
                    visual_datum=v_datum,
                    baro_datum=b_datum,
                    geoid_undulation_m=None,
                    z_visual_input_m=z_vis_in,
                    z_visual_msl_m=None,
                    z_baro_msl_m=z_baro_in,
                    vertical_innovation_m=None,
                    abs_vertical_innovation_m=None,
                    max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
                    ned_down_visual_msl_m=None,
                    ned_down_baro_msl_m=round(-z_baro_in, 6),
                    datum_conversion_applied=False,
                )
            z_vis_msl = wgs84_hae_to_msl(z_vis_in, n_eff)
            z_baro_msl = z_baro_in
            conversion_applied = True
        elif vis_is_msl and (not baro_is_msl):
            if (not apply_geoid_conversion) or (n_eff is None):
                return DatumSwitchDecision(
                    allowed=False,
                    reason_code=DATUM_REASON_MISMATCH_INHIBITED,
                    detail=(
                        f"Datum mismatch ({v_datum} vs {b_datum}) without valid geoid conversion; "
                        "EKF3 42007 switch inhibited fail-closed"
                    ),
                    visual_datum=v_datum,
                    baro_datum=b_datum,
                    geoid_undulation_m=n_eff,
                    z_visual_input_m=z_vis_in,
                    z_visual_msl_m=z_vis_in,
                    z_baro_msl_m=None,
                    vertical_innovation_m=None,
                    abs_vertical_innovation_m=None,
                    max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
                    ned_down_visual_msl_m=round(-z_vis_in, 6),
                    ned_down_baro_msl_m=None,
                    datum_conversion_applied=False,
                )
            z_vis_msl = z_vis_in
            z_baro_msl = wgs84_hae_to_msl(z_baro_in, n_eff)
            conversion_applied = True
        else:
            # Both WGS84_HAE: require valid geoid undulation to express consistent MSL NED if provided
            if n_eff is not None and apply_geoid_conversion:
                z_vis_msl = wgs84_hae_to_msl(z_vis_in, n_eff)
                z_baro_msl = wgs84_hae_to_msl(z_baro_in, n_eff)
                conversion_applied = True
            else:
                z_vis_msl = z_vis_in
                z_baro_msl = z_baro_in

        vert_innov = z_vis_msl - z_baro_msl
        abs_innov = abs(vert_innov)
        if abs_innov > self._max_pre_switch_vert_innov_m:
            return DatumSwitchDecision(
                allowed=False,
                reason_code=DATUM_REASON_PRE_SWITCH_VERT_INNOV_EXCEEDED,
                detail=(
                    f"Pre-switch vertical innovation |{vert_innov:+.3f} m| = {abs_innov:.3f} m "
                    f"exceeds {self._max_pre_switch_vert_innov_m:.3f} m limit; EKF3 42007 inhibited"
                ),
                visual_datum=v_datum,
                baro_datum=b_datum,
                geoid_undulation_m=n_eff,
                z_visual_input_m=round(z_vis_in, 6),
                z_visual_msl_m=round(z_vis_msl, 6),
                z_baro_msl_m=round(z_baro_msl, 6),
                vertical_innovation_m=round(vert_innov, 6),
                abs_vertical_innovation_m=round(abs_innov, 6),
                max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
                ned_down_visual_msl_m=round(-z_vis_msl, 6),
                ned_down_baro_msl_m=round(-z_baro_msl, 6),
                datum_conversion_applied=conversion_applied,
            )

        return DatumSwitchDecision(
            allowed=True,
            reason_code=DATUM_REASON_OK,
            detail=(
                f"Vertical datum aligned (|innov|={abs_innov:.3f} m <= "
                f"{self._max_pre_switch_vert_innov_m:.3f} m)"
            ),
            visual_datum=v_datum,
            baro_datum=b_datum,
            geoid_undulation_m=n_eff,
            z_visual_input_m=round(z_vis_in, 6),
            z_visual_msl_m=round(z_vis_msl, 6),
            z_baro_msl_m=round(z_baro_msl, 6),
            vertical_innovation_m=round(vert_innov, 6),
            abs_vertical_innovation_m=round(abs_innov, 6),
            max_pre_switch_vert_innov_m=self._max_pre_switch_vert_innov_m,
            ned_down_visual_msl_m=round(-z_vis_msl, 6),
            ned_down_baro_msl_m=round(-z_baro_msl, 6),
            datum_conversion_applied=conversion_applied,
        )

    def gate_ekf3_switch_latch(
        self,
        latch: Ekf3SourceSwitchLatch,
        *,
        timestamp_s: float,
        gnss_healthy: bool,
        spoof_alarm: bool,
        z_visual_m: float | None,
        z_baro_msl_m: float,
        visual_datum: str | None = None,
        baro_datum: str | None = None,
        geoid_undulation_m: float | None = None,
        apply_geoid_conversion: bool = True,
    ) -> tuple[Ekf3SwitchState, DatumSwitchDecision]:
        """Gate `Ekf3SourceSwitchLatch` so `EKF3 42007` visual handoff is blocked on datum fault."""
        decision = self.evaluate_ekf3_switch(
            z_visual_m=z_visual_m,
            z_baro_msl_m=z_baro_msl_m,
            visual_datum=visual_datum,
            baro_datum=baro_datum,
            geoid_undulation_m=geoid_undulation_m,
            apply_geoid_conversion=apply_geoid_conversion,
        )
        wants_vision_handoff = (
            latch.active_source != EKF3_SOURCE_VISION_OPTICAL
            and (bool(spoof_alarm) or not bool(gnss_healthy))
        )
        if wants_vision_handoff and not decision.allowed:
            # Preserve monotonic timestamp check on latch without switching to Vision or emitting 42007
            inhibited_state = latch.step(
                timestamp_s=timestamp_s,
                gnss_healthy=True,
                spoof_alarm=False,
            )
            return inhibited_state, decision

        state = latch.step(
            timestamp_s=timestamp_s,
            gnss_healthy=gnss_healthy,
            spoof_alarm=spoof_alarm,
        )
        return state, decision


# ---- Hardware-Mirror Listen-Only SHA-256 Hash-Chained Shadow-Mode Telemetry Logger ----


def _reject_json_non_finite_constant(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant rejected fail-closed: {token!r}")


@dataclass(frozen=True)
class ShadowChainVerificationResult:
    """Verification verdict for a SHA-256 hash-chained shadow-mode NDJSON flight-recorder file."""

    valid: bool
    log_path: str
    record_count: int
    first_seq: int | None
    last_seq: int | None
    head_hash: str
    tx_inhibited_all: bool
    actuator_authority_all_false: bool
    total_outbound_mavlink_frames: int
    monotonic_timestamps_ok: bool
    first_failed_line: int | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_shadow_log_chain(log_path: Path | str) -> ShadowChainVerificationResult:
    """Independently verify tamper-evident SHA-256 chain, monotonicity, and zero-TX invariants."""
    path = Path(log_path)
    errors: list[str] = []
    first_failed_line: int | None = None
    record_count = 0
    first_seq: int | None = None
    last_seq: int | None = None
    expected_prev_hash = SHADOW_GENESIS_HASH
    prev_ts: int | None = None
    tx_inhibited_all = True
    actuator_authority_all_false = True
    total_outbound = 0
    monotonic_ok = True

    def _record_err(line_no: int, msg: str) -> None:
        nonlocal first_failed_line
        if first_failed_line is None:
            first_failed_line = line_no
        errors.append(f"line {line_no}: {msg}")

    if not path.is_file():
        return ShadowChainVerificationResult(
            valid=False,
            log_path=str(path.name),
            record_count=0,
            first_seq=None,
            last_seq=None,
            head_hash=SHADOW_GENESIS_HASH,
            tx_inhibited_all=False,
            actuator_authority_all_false=False,
            total_outbound_mavlink_frames=0,
            monotonic_timestamps_ok=False,
            first_failed_line=1,
            errors=(f"Shadow-mode NDJSON log file not found: {path.name}",),
        )

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ShadowChainVerificationResult(
            valid=False,
            log_path=str(path.name),
            record_count=0,
            first_seq=None,
            last_seq=None,
            head_hash=SHADOW_GENESIS_HASH,
            tx_inhibited_all=False,
            actuator_authority_all_false=False,
            total_outbound_mavlink_frames=0,
            monotonic_timestamps_ok=False,
            first_failed_line=1,
            errors=("Empty shadow-mode NDJSON log (0 records)",),
        )

    for idx, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            _record_err(idx, "Blank line forbidden inside hash-chained NDJSON log")
            continue
        try:
            rec = json.loads(raw_line, parse_constant=_reject_json_non_finite_constant)
        except Exception as exc:
            _record_err(idx, f"Malformed or non-finite JSON: {exc}")
            continue
        if not isinstance(rec, dict):
            _record_err(idx, "Record is not a JSON object")
            continue

        expected_seq = record_count
        record_count += 1

        schema = rec["schema"] if "schema" in rec else None
        if schema != SHADOW_MODE_LOG_SCHEMA:
            _record_err(idx, f"Invalid schema {schema!r}, expected {SHADOW_MODE_LOG_SCHEMA!r}")

        seq = rec["seq"] if "seq" in rec else None
        if isinstance(seq, bool) or not isinstance(seq, int) or seq != expected_seq:
            _record_err(idx, f"Sequence mismatch: got {seq!r}, expected {expected_seq}")
        else:
            if first_seq is None:
                first_seq = seq
            last_seq = seq

        ts = rec["timestamp_mono_us"] if "timestamp_mono_us" in rec else None
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
            monotonic_ok = False
            _record_err(idx, f"Invalid timestamp_mono_us: {ts!r}")
        else:
            if prev_ts is not None and ts <= prev_ts:
                monotonic_ok = False
                _record_err(idx, f"Non-monotonic timestamp_mono_us: {ts} <= {prev_ts}")
            prev_ts = ts

        if ("tx_inhibited" not in rec) or (rec["tx_inhibited"] is not True):
            tx_inhibited_all = False
            _record_err(idx, "Hard invariant violated: tx_inhibited must be True")

        if ("actuator_authority" not in rec) or (rec["actuator_authority"] is not False):
            actuator_authority_all_false = False
            _record_err(idx, "Hard invariant violated: actuator_authority must be False")

        out_frames = rec["outbound_mavlink_frames_emitted"] if "outbound_mavlink_frames_emitted" in rec else None
        if isinstance(out_frames, bool) or not isinstance(out_frames, int) or out_frames != 0:
            if isinstance(out_frames, int) and not isinstance(out_frames, bool):
                total_outbound += max(0, out_frames)
            _record_err(idx, f"Hard invariant violated: outbound_mavlink_frames_emitted={out_frames!r} != 0")

        direction = rec["direction"] if "direction" in rec else None
        if direction not in ALLOWED_SHADOW_DIRECTIONS:
            _record_err(idx, f"Forbidden shadow-mode direction {direction!r}")

        prev_hash = rec["prev_hash"] if "prev_hash" in rec else None
        if prev_hash != expected_prev_hash:
            _record_err(idx, f"Hash chain break: prev_hash={prev_hash!r} != expected {expected_prev_hash!r}")

        entry_hash = rec["entry_hash"] if "entry_hash" in rec else None
        if not isinstance(entry_hash, str) or len(entry_hash) != 64:
            _record_err(idx, f"Invalid entry_hash field: {entry_hash!r}")
            expected_prev_hash = str(entry_hash) if isinstance(entry_hash, str) else ""
            continue

        body = {k: v for k, v in rec.items() if k != "entry_hash"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        calc_hash = hashlib.sha256(canonical).hexdigest()
        if calc_hash != entry_hash:
            _record_err(idx, f"SHA-256 tamper detected: entry_hash={entry_hash[:16]}... != calc={calc_hash[:16]}...")
        expected_prev_hash = entry_hash

    valid = (
        len(errors) == 0
        and record_count > 0
        and tx_inhibited_all
        and actuator_authority_all_false
        and total_outbound == 0
        and monotonic_ok
    )
    return ShadowChainVerificationResult(
        valid=valid,
        log_path=str(path.name),
        record_count=record_count,
        first_seq=first_seq,
        last_seq=last_seq,
        head_hash=expected_prev_hash,
        tx_inhibited_all=tx_inhibited_all,
        actuator_authority_all_false=actuator_authority_all_false,
        total_outbound_mavlink_frames=total_outbound,
        monotonic_timestamps_ok=monotonic_ok,
        first_failed_line=first_failed_line,
        errors=tuple(errors),
    )


class ShadowModeTelemetryLogger:
    """Hardware-mirror listen-only SHA-256 hash-chained NDJSON flight-recorder.

    Hard invariants (physically mirroring Phase-2 rooftop/car-roof & Phase-3 TX-pin-removed
    university shadow-mode trials):
      - `tx_inhibited == True` (immutable, cannot be disabled)
      - `actuator_authority == False` (immutable, cannot be enabled)
      - `outbound_mavlink_frames_emitted == 0` (hard-enforced on every record)
      - Strictly monotonic `timestamp_mono_us` and contiguous `seq`
      - Tamper-evident SHA-256 hash chain over canonical UTF-8 JSON (`allow_nan=False`)
    """

    TX_INHIBITED: bool = True
    ACTUATOR_AUTHORITY: bool = False

    def __init__(
        self,
        log_path: Path | str,
        *,
        trial_phase: str = "phase3_university_shadow_tx_pin_removed",
        tx_inhibited: bool = True,
        actuator_authority: bool = False,
        hardware_tx_pin_removed: bool = True,
        fsync_each_record: bool = False,
    ) -> None:
        if tx_inhibited is not True:
            raise ShadowModeTxInhibitedError(
                "ShadowModeTelemetryLogger enforces immutable tx_inhibited=True (hardware listen-only mirror)"
            )
        if actuator_authority is not False:
            raise ShadowModeTxInhibitedError(
                "ShadowModeTelemetryLogger enforces immutable actuator_authority=False (zero actuation)"
            )
        phase = str(trial_phase).strip().lower()
        if phase not in VALID_SHADOW_TRIAL_PHASES:
            raise ValueError(
                f"Unsupported trial_phase {trial_phase!r}; expected one of {VALID_SHADOW_TRIAL_PHASES}"
            )
        self._lock = threading.Lock()
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._trial_phase = phase
        self._hardware_tx_pin_removed = bool(hardware_tx_pin_removed)
        self._fsync_each_record = bool(fsync_each_record)
        self._seq: int = 0
        self._last_ts_mono_us: int | None = None
        self._head_hash: str = SHADOW_GENESIS_HASH
        self._closed: bool = False
        self._fh = self._log_path.open("w", encoding="utf-8", newline="\n")

    @property
    def tx_inhibited(self) -> bool:
        return True

    @property
    def actuator_authority(self) -> bool:
        return False

    @property
    def outbound_mavlink_frames_emitted(self) -> int:
        return 0

    @property
    def trial_phase(self) -> str:
        return self._trial_phase

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def record_count(self) -> int:
        with self._lock:
            return self._seq

    @property
    def head_hash(self) -> str:
        with self._lock:
            return self._head_hash

    @property
    def last_timestamp_mono_us(self) -> int | None:
        with self._lock:
            return self._last_ts_mono_us

    def reject_outbound_tx(self, *, msgid: int | None = None, reason: str = "") -> None:
        """Fail-closed barrier rejecting any outbound MAVLink / actuation attempt in shadow mode."""
        extra = f" (msgid={msgid})" if msgid is not None else ""
        detail = f": {reason}" if reason else ""
        raise ShadowModeTxInhibitedError(
            f"Outbound MAVLink/actuation TX inhibited in listen-only shadow mode{extra}{detail}"
        )

    def record_event(
        self,
        *,
        timestamp_mono_us: int,
        event_type: str,
        payload: dict[str, Any],
        direction: str = "rx",
        outbound_mavlink_frames: int = 0,
    ) -> dict[str, Any]:
        """Append one SHA-256 hash-chained listen-only telemetry record to the NDJSON flight log."""
        dir_clean = str(direction).strip().lower()
        if dir_clean not in ALLOWED_SHADOW_DIRECTIONS:
            raise ShadowModeTxInhibitedError(
                f"ShadowModeTelemetryLogger forbids direction={direction!r}; only {sorted(ALLOWED_SHADOW_DIRECTIONS)} allowed"
            )
        if isinstance(outbound_mavlink_frames, bool) or int(outbound_mavlink_frames) != 0:
            raise ShadowModeTxInhibitedError(
                f"ShadowModeTelemetryLogger requires outbound_mavlink_frames == 0, got {outbound_mavlink_frames!r}"
            )
        if isinstance(timestamp_mono_us, bool) or not isinstance(timestamp_mono_us, int) or timestamp_mono_us < 0:
            raise ValueError(f"timestamp_mono_us must be non-negative int, got {timestamp_mono_us!r}")
        ev_clean = str(event_type).strip()
        if not ev_clean:
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON-serializable dict")

        with self._lock:
            if self._closed:
                raise RuntimeError("ShadowModeTelemetryLogger is already closed")
            if self._last_ts_mono_us is not None and timestamp_mono_us <= self._last_ts_mono_us:
                raise ValueError(
                    f"Non-monotonic timestamp_mono_us rejected: {timestamp_mono_us} <= {self._last_ts_mono_us}"
                )

            body: dict[str, Any] = {
                "schema": SHADOW_MODE_LOG_SCHEMA,
                "seq": self._seq,
                "timestamp_mono_us": int(timestamp_mono_us),
                "trial_phase": self._trial_phase,
                "hardware_tx_pin_removed": self._hardware_tx_pin_removed,
                "event_type": ev_clean,
                "direction": dir_clean,
                "tx_inhibited": True,
                "actuator_authority": False,
                "outbound_mavlink_frames_emitted": 0,
                "non_flight_sim_only": True,
                "watermark": NON_FLIGHT_WATERMARK,
                "payload": payload,
                "prev_hash": self._head_hash,
            }
            canonical_body = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
            entry_hash = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
            record = dict(body)
            record["entry_hash"] = entry_hash
            line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"

            self._fh.write(line)
            self._fh.flush()
            if self._fsync_each_record:
                os.fsync(self._fh.fileno())

            self._seq += 1
            self._last_ts_mono_us = int(timestamp_mono_us)
            self._head_hash = entry_hash
            return record

    def record_rx_mavlink_frame(
        self,
        *,
        timestamp_mono_us: int,
        msgid: int,
        seq: int,
        raw_frame: bytes,
        summary_ned: dict[str, Any],
        datum_decision: DatumSwitchDecision | None = None,
    ) -> dict[str, Any]:
        """Convenience logger for one received MAVLink 2.0 RX wire frame (0 outbound bytes)."""
        payload: dict[str, Any] = {
            "msgid": int(msgid),
            "mav_seq": int(seq),
            "wire_bytes": len(raw_frame),
            "frame_sha256": hashlib.sha256(raw_frame).hexdigest(),
            "telemetry_ned": summary_ned,
        }
        if datum_decision is not None:
            payload["datum_guard"] = datum_decision.to_dict()
        return self.record_event(
            timestamp_mono_us=timestamp_mono_us,
            event_type="rx_mavlink2",
            payload=payload,
            direction="rx",
            outbound_mavlink_frames=0,
        )

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._fh.flush()
                self._fh.close()
                self._closed = True

    def __enter__(self) -> ShadowModeTelemetryLogger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @staticmethod
    def verify_chain(log_path: Path | str) -> ShadowChainVerificationResult:
        return verify_shadow_log_chain(log_path)


def is_non_flight_sim_only() -> bool:
    """Return True when non-flight simulation-only lockout is enabled (fail-closed if env var is set).

    Backwards-compatible with local dev SITL tests when neither env var is set or explicitly set to `0`.
    When `OPENXVS_NON_FLIGHT_SIM_ONLY` (or alias `NON_FLIGHT_SIM_ONLY`) is present in the environment,
    ANY value outside explicit dev opt-out `{"0", "false", "no", "off"}` fails CLOSED (`True`).
    """
    raw = os.environ.get(NON_FLIGHT_ENV_VAR)
    if raw is None:
        raw = os.environ.get(NON_FLIGHT_ENV_ALIAS)
    if raw is None:
        return False
    return raw.strip().lower() not in _EXPLICIT_DEV_UNLOCK_VALUES


@dataclass(frozen=True)
class FlightGearTelemetryNed:
    """Validated FlightGear Generic UDP state packet converted strictly into NED."""

    lat_deg: float
    lon_deg: float
    alt_msl_m: float
    down_m: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    airspeed_kt: float
    airspeed_mps: float
    vn_mps: float
    ve_mps: float
    vd_mps: float


def _finite_float(raw: str, name: str) -> float:
    val = float(raw.strip())
    if not math.isfinite(val):
        raise ValueError(f"Non-finite FlightGear field {name!r}: {raw!r}")
    return val


def parse_flightgear_telemetry_csv(packet: bytes | str) -> FlightGearTelemetryNed:
    """Parse 10-field FlightGear Generic UDP CSV line into validated NED telemetry."""
    text = packet.decode("utf-8", errors="strict").strip() if isinstance(packet, bytes) else str(packet).strip()
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 10:
        raise ValueError(f"Expected 10 comma-separated FlightGear fields, got {len(parts)}")

    lat = _finite_float(parts[0], "lat_deg")
    lon = _finite_float(parts[1], "lon_deg")
    alt_msl = _finite_float(parts[2], "alt_msl_m")
    hdg = _finite_float(parts[3], "heading_deg")
    pit = _finite_float(parts[4], "pitch_deg")
    rol = _finite_float(parts[5], "roll_deg")
    ias_kt = _finite_float(parts[6], "airspeed_kt")
    vn = _finite_float(parts[7], "vn_mps")
    ve = _finite_float(parts[8], "ve_mps")
    vd = _finite_float(parts[9], "vd_mps")

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"FlightGear lat/lon out of bounds: ({lat}, {lon})")
    if ias_kt < 0.0:
        raise ValueError(f"FlightGear airspeed_kt must be >= 0, got {ias_kt}")

    return FlightGearTelemetryNed(
        lat_deg=lat,
        lon_deg=lon,
        alt_msl_m=alt_msl,
        down_m=-alt_msl,
        heading_deg=hdg % 360.0,
        pitch_deg=pit,
        roll_deg=rol,
        airspeed_kt=ias_kt,
        airspeed_mps=ias_kt * 0.514444,
        vn_mps=vn,
        ve_mps=ve,
        vd_mps=vd,
    )


def encode_flightgear_control_csv(
    *,
    bank_cmd_deg: float,
    pitch_cmd_deg: float,
    airspeed_cmd_mps: float,
    shield_active: bool = True,
) -> bytes:
    """Encode envelope-clamped FlightGear avoidance CSV (`|bank| <= 30 deg`, `|pitch| <= 20 deg`, `V >= 18 m/s`)."""
    for nm, val in (("bank_cmd_deg", bank_cmd_deg), ("pitch_cmd_deg", pitch_cmd_deg), ("airspeed_cmd_mps", airspeed_cmd_mps)):
        if not math.isfinite(float(val)):
            raise ValueError(f"Non-finite FlightGear control command {nm!r}: {val!r}")

    max_bank_deg = math.degrees(DEFAULT_MAX_BANK_RAD)
    max_pitch_deg = math.degrees(DEFAULT_MAX_PITCH_RAD)
    c_bank = max(-max_bank_deg, min(max_bank_deg, float(bank_cmd_deg)))
    c_pitch = max(-max_pitch_deg, min(max_pitch_deg, float(pitch_cmd_deg)))
    c_spd = max(DEFAULT_V_MIN_MPS, min(65.0, float(airspeed_cmd_mps)))
    flag = 1 if bool(shield_active) else 0
    return f"{c_bank:.3f},{c_pitch:.3f},{c_spd:.3f},{flag}\n".encode("ascii")


def _verify_active_range_64b_wire_and_tamper() -> dict[str, Any]:
    """Verify exact 64-byte `XVS-ActiveRange` wire + IEEE-802.3 CRC32 round-trip + 1-bit tamper rejection."""
    pkt = ActiveRangeTelemetry(
        seq=77,
        mode=int(ActiveRangeMode.LOCKED),
        active_bank_idx=2,
        flags=0x000D,
        r_hat_m=642.5,
        sigma_r_m=21.06,
        r_safe_lower_m=607.75,
        rho_dot_inv_s=-0.082,
        cmd_bank_rad=0.157,
        min_sc3bf_margin=14.2,
        fim_gain=6.4,
        weights=(0.0001, 0.012, 0.965, 0.0226, 0.0002, 0.0001),
        crc32=0,
    )
    wire = encode_wire(pkt)
    if len(wire) != WIRE_SIZE_BYTES:
        raise AssertionError(f"Expected {WIRE_SIZE_BYTES} B ActiveRange wire, got {len(wire)}")
    dec = decode_wire(wire)
    if dec is None or dec.seq != pkt.seq or dec.mode != pkt.mode:
        raise AssertionError("ActiveRange 64 B wire round-trip mismatch")

    tamper_rejected = 0
    for byte_off in (0, 12, 36, 63):
        corrupted = bytearray(wire)
        corrupted[byte_off] ^= 0x01
        if decode_wire(bytes(corrupted)) is None:
            tamper_rejected += 1

    passed = tamper_rejected == 4
    return {
        "name": "active_range_64b_wire_crc32",
        "wire_bytes": len(wire),
        "roundtrip_ok": True,
        "tamper_cases_tested": 4,
        "tamper_cases_rejected": tamper_rejected,
        "pass": passed,
    }


def _verify_dr06_fixed_wing_safety_and_plugin_health() -> dict[str, Any]:
    """Verify DR-06 `#84` ban, `#75`/`#82` envelope clamp, EKF3 42007 latch, and `#50004` codec."""
    # 1. Hard-reject MAVLink #84 SET_POSITION_TARGET_LOCAL_NED
    msg84_rejected = False
    try:
        validate_fixed_wing_avoidance_msgid(MAVLINK_MSG_SET_POSITION_TARGET_LOCAL_NED)
    except ValueError:
        msg84_rejected = True

    validate_fixed_wing_avoidance_msgid(MAVLINK_MSG_COMMAND_INT)
    validate_fixed_wing_avoidance_msgid(MAVLINK_MSG_SET_ATTITUDE_TARGET)

    # 2. Envelope clamp checks on #75 and #82 (|bank| <= 30 deg, V >= 18 m/s)
    cmd75 = build_command_int_avoidance(
        lat_deg=37.61172,
        lon_deg=-122.358367,
        alt_m=180.0,
        airspeed_mps=10.0,  # below 18 m/s -> must clamp!
        bank_cmd_rad=math.radians(48.0),  # above 30 deg -> must clamp!
    )
    cmd82 = build_set_attitude_target(
        roll_rad=math.radians(-45.0),
        pitch_rad=math.radians(32.0),
        yaw_rad=math.radians(298.0),
        thrust_norm=0.65,
        airspeed_mps=12.0,
    )
    clamp_ok = (
        cmd75.bank_clamped
        and cmd75.airspeed_clamped
        and abs(cmd75.bank_cmd_rad) <= DEFAULT_MAX_BANK_RAD + 1e-9
        and cmd75.airspeed_mps >= DEFAULT_V_MIN_MPS - 1e-9
        and cmd82.bank_clamped
        and cmd82.airspeed_clamped
        and abs(cmd82.roll_rad) <= DEFAULT_MAX_BANK_RAD + 1e-9
        and cmd82.airspeed_mps >= DEFAULT_V_MIN_MPS - 1e-9
    )

    # 3. EKF3 42007 hysteresis latch (>= 2.0 s dwell before GNSS revert)
    latch = Ekf3SourceSwitchLatch()
    st_spoof = latch.step(timestamp_s=1.0, gnss_healthy=True, spoof_alarm=True)
    st_early = latch.step(timestamp_s=2.0, gnss_healthy=True, spoof_alarm=False)
    st_dwell = latch.step(timestamp_s=4.5, gnss_healthy=True, spoof_alarm=False)
    latch_ok = (
        st_spoof.active_source == EKF3_SOURCE_VISION_OPTICAL
        and st_early.active_source == EKF3_SOURCE_VISION_OPTICAL
        and st_dwell.active_source == EKF3_SOURCE_GNSS
    )

    # 4. #50004 POD_PLUGIN_HEALTH (ACTUATOR_AUTHORITY = False) round-trip
    ph = PodPluginHealth(
        plugin_id_hash=0xA1B2C3D4,
        timestamp_mono_us=1_234_567,
        last_seq_no=42,
        lifecycle_state=int(PodPluginLifecycleState.HEALTHY),
        civil_mission_class=int(PodCivilMissionClass.SEARCH_AND_RESCUE),
        missed_heartbeats=0,
        rss_mb=128,
        cpu_permille=220,
    ).with_valid_crc()
    ph_wire = encode_frame(ph, seq=7)
    ph_dec = decode_frame(ph_wire, verify_inner_crc32=True)
    ph_ok = (
        ph.ACTUATOR_AUTHORITY is False
        and isinstance(ph_dec.message, PodPluginHealth)
        and ph_dec.message.verify_ieee_crc32()
    )

    return {
        "name": "dr06_fixed_wing_safety_and_pod_plugin_health",
        "msg84_hard_rejected": msg84_rejected,
        "cmd75_cmd82_envelope_clamped": clamp_ok,
        "ekf3_42007_latch_ok": latch_ok,
        "pod_plugin_health_50004_ok": ph_ok,
        "pass": bool(msg84_rejected and clamp_ok and latch_ok and ph_ok),
    }


def _verify_flightgear_udp_loopback(n_packets: int = 10) -> dict[str, Any]:
    """Send/receive `n_packets` FlightGear Generic UDP telemetry & control CSV frames on 127.0.0.1."""
    rx_tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx_tel.bind(("127.0.0.1", 0))
        rx_ctl.bind(("127.0.0.1", 0))
        rx_tel.settimeout(2.0)
        rx_ctl.settimeout(2.0)
        tel_port = int(rx_tel.getsockname()[1])
        ctl_port = int(rx_ctl.getsockname()[1])

        tel_ok = 0
        ctl_ok = 0
        for idx in range(n_packets):
            csv_line = (
                f"{37.61172 + 0.0001 * idx:.7f},{-122.358367 - 0.0001 * idx:.7f},"
                f"{180.0 - 2.0 * idx:.3f},298.000,-3.000,{4.0 * math.sin(idx):.3f},"
                f"68.000,16.500,-30.800,1.800\n"
            ).encode("ascii")
            tx_sock.sendto(csv_line, ("127.0.0.1", tel_port))
            raw_rx, _ = rx_tel.recvfrom(512)
            parsed = parse_flightgear_telemetry_csv(raw_rx)
            if abs(parsed.down_m + parsed.alt_msl_m) < 1e-6 and parsed.airspeed_mps > 30.0:
                tel_ok += 1

            ctl_bytes = encode_flightgear_control_csv(
                bank_cmd_deg=42.0,  # clamps to 30.0 deg
                pitch_cmd_deg=3.5,
                airspeed_cmd_mps=12.0,  # clamps to 18.0 m/s
                shield_active=True,
            )
            tx_sock.sendto(ctl_bytes, ("127.0.0.1", ctl_port))
            raw_ctl, _ = rx_ctl.recvfrom(256)
            c_parts = raw_ctl.decode("ascii").strip().split(",")
            if len(c_parts) == 4 and abs(float(c_parts[0]) - 30.0) < 1e-3 and abs(float(c_parts[2]) - 18.0) < 1e-3:
                ctl_ok += 1

        # NaN fail-closed rejection check
        nan_rejected = False
        try:
            parse_flightgear_telemetry_csv("nan,-122.35,100.0,298,0,0,65,10,-20,0\n")
        except ValueError:
            nan_rejected = True
    finally:
        rx_tel.close()
        rx_ctl.close()
        tx_sock.close()

    passed = tel_ok == n_packets and ctl_ok == n_packets and nan_rejected
    return {
        "name": "flightgear_generic_udp_loopback",
        "telemetry_packets_verified": tel_ok,
        "control_packets_verified": ctl_ok,
        "nan_rejected": nan_rejected,
        "pass": passed,
    }


def _validate_local_or_private_host(host: str, *, allow_any_bind: bool = False) -> str:
    """Restrict UDP endpoints strictly to loopback or local RFC-1918 WSL2/LAN interfaces."""
    h = host.strip()
    if h == "localhost":
        return "127.0.0.1"
    if allow_any_bind and h == "0.0.0.0":
        return "0.0.0.0"
    addr = ipaddress.ip_address(h)
    if addr.version != 4 or not (addr.is_loopback or addr.is_private):
        raise ValueError(f"UDP host {host!r} rejected; only loopback or local private IPv4 is permitted")
    return str(addr)


def _offset_latlon_ned(lat_deg: float, lon_deg: float, d_north_m: float, d_east_m: float) -> tuple[float, float]:
    """Offset WGS-84 `(lat_deg, lon_deg)` by local NED meters `(d_north_m, d_east_m)`."""
    d_lat = d_north_m / 111_132.954
    cos_lat = max(0.1, math.cos(math.radians(lat_deg)))
    d_lon = d_east_m / (111_319.491 * cos_lat)
    new_lat = max(-89.9, min(89.9, lat_deg + d_lat))
    new_lon = ((lon_deg + d_lon + 180.0) % 360.0) - 180.0
    return (new_lat, new_lon)


def _alloc_ephemeral_udp_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _verify_closed_loop_rx_tx_bridge() -> dict[str, Any]:
    """Verify full real-UDP RX -> Autonomy -> TX closed loop on `StudioHilBridgeManager`."""
    mgr = StudioHilBridgeManager(non_flight_sim_only=False)
    tx_listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    injector = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx_listener.bind(("127.0.0.1", 0))
        tx_listener.settimeout(1.5)
        tx_port = int(tx_listener.getsockname()[1])
        rx_port = _alloc_ephemeral_udp_port()

        # 1. MAVLink 2.0 closed-loop RX (#30/#33) -> Autonomy -> TX (#82, #75, #76(42007))
        mgr.connect(
            mode="mavlink2_udp",
            rx_port=rx_port,
            tx_port=tx_port,
            tx_command_mode="set_attitude_target",
        )

        # Nominal telemetry -> expect #82 SET_ATTITUDE_TARGET
        att_wire = encode_frame(
            MavAttitude(time_boot_ms=1000, roll=0.04, pitch=0.02, yaw=math.radians(298.0)),
            seq=1,
            sysid=1,
            compid=1,
        )
        gpi_wire = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=1100,
                lat_e7=376_117_200,
                lon_e7=-1_223_583_670,
                alt_mm=180_000,
                relative_alt_mm=175_000,
                vx_cms=1500,
                vy_cms=-2800,
                vz_cms=0,
                hdg_cdeg=29800,
            ),
            seq=2,
            sysid=1,
            compid=1,
        )
        injector.sendto(att_wire, ("127.0.0.1", rx_port))
        injector.sendto(gpi_wire, ("127.0.0.1", rx_port))

        mav_rx_msgids: list[int] = []
        ekf3_42007_emitted = False
        for _ in range(2):
            pkt, _ = tx_listener.recvfrom(512)
            dec = decode_frame(pkt)
            mav_rx_msgids.append(int(dec.msgid))

        # Trigger CBDR threat + COMMAND_INT + GNSS spoof alarm -> expect #76(42007) + #75 COMMAND_INT
        mgr.set_autonomy_triggers(
            cbdr_threat_active=True,
            cbdr_evade_direction=1.0,
            gnss_spoof_alarm=True,
            gnss_healthy=True,
            tx_command_mode="command_int",
        )
        gpi_wire2 = encode_frame(
            MavGlobalPositionInt(
                time_boot_ms=1300,
                lat_e7=376_118_500,
                lon_e7=-1_223_586_000,
                alt_mm=180_000,
                relative_alt_mm=175_000,
                vx_cms=1500,
                vy_cms=-2800,
                vz_cms=0,
                hdg_cdeg=29800,
            ),
            seq=3,
            sysid=1,
            compid=1,
        )
        injector.sendto(gpi_wire2, ("127.0.0.1", rx_port))
        for _ in range(2):
            pkt, _ = tx_listener.recvfrom(512)
            dec = decode_frame(pkt)
            mav_rx_msgids.append(int(dec.msgid))
            if isinstance(dec.message, MavCommandLong) and dec.message.command == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET:
                if abs(dec.message.param1 - float(EKF3_SOURCE_VISION_OPTICAL)) < 1e-6:
                    ekf3_42007_emitted = True

        # Inject forbidden #84 frame and corrupted CRC frame -> must be rejected fail-closed with zero TX
        tx_count_before_bad = int(mgr.status()["tx_packets"])
        fake_84 = (
            bytes([MAVLINK_V2_STX])
            + struct.pack("<BBBBBBBBB", 1, 0, 0, 9, 1, 1, MSGID_SET_POSITION_TARGET_LOCAL_NED, 0, 0)
            + b"\x00\x00\x00"
        )
        bad_crc = bytearray(att_wire)
        bad_crc[-1] ^= 0xFF
        injector.sendto(fake_84, ("127.0.0.1", rx_port))
        injector.sendto(bytes(bad_crc), ("127.0.0.1", rx_port))
        time.sleep(0.08)
        st_mav = mgr.status()
        mav_bad_rejected = (
            int(st_mav["rejected_packets"]) >= 2
            and int(st_mav["tx_packets"]) == tx_count_before_bad
        )
        mav_tx_packets = int(st_mav["tx_packets"])
        mav_msg84_never = (
            not bool(st_mav["msg84_ever_emitted"])
            and MSGID_SET_POSITION_TARGET_LOCAL_NED not in mav_rx_msgids
        )

        # 2. FlightGear Generic UDP closed-loop RX CSV -> Autonomy -> TX CSV
        fg_rx_port = _alloc_ephemeral_udp_port()
        mgr.connect(mode="flightgear_udp", rx_port=fg_rx_port, tx_port=tx_port)
        mgr.set_autonomy_triggers(cbdr_threat_active=True, cbdr_evade_direction=1.0)
        fg_csv_valid = b"37.6117200,-122.3583670,180.000,298.000,-2.000,3.000,64.000,15.400,-29.200,0.400\n"
        fg_csv_nan = b"nan,-122.3583670,180.000,298.000,-2.000,3.000,64.000,15.400,-29.200,0.400\n"
        injector.sendto(fg_csv_valid, ("127.0.0.1", fg_rx_port))
        raw_fg_tx, _ = tx_listener.recvfrom(256)
        fg_parts = raw_fg_tx.decode("ascii").strip().split(",")
        fg_bank = float(fg_parts[0])
        fg_pitch = float(fg_parts[1])
        fg_spd = float(fg_parts[2])
        fg_shield = int(fg_parts[3])
        fg_tx_valid = (
            len(fg_parts) == 4
            and abs(fg_bank) <= math.degrees(DEFAULT_MAX_BANK_RAD) + 1e-6
            and abs(fg_pitch) <= math.degrees(DEFAULT_MAX_PITCH_RAD) + 1e-6
            and DEFAULT_V_MIN_MPS - 1e-6 <= fg_spd <= DEFAULT_V_MAX_MPS + 1e-6
            and fg_shield == 1
        )
        fg_tx_before_nan = int(mgr.status()["tx_packets"])
        injector.sendto(fg_csv_nan, ("127.0.0.1", fg_rx_port))
        time.sleep(0.08)
        st_fg = mgr.status()
        fg_nan_rejected = (
            int(st_fg["rejected_packets"]) >= 1
            and int(st_fg["tx_packets"]) == fg_tx_before_nan
        )
        fg_tx_packets = int(st_fg["tx_packets"])
    finally:
        mgr.shutdown()
        tx_listener.close()
        injector.close()

    total_tx = mav_tx_packets + fg_tx_packets
    passed = bool(
        mav_tx_packets >= 4
        and fg_tx_packets >= 1
        and total_tx > 0
        and MSGID_SET_ATTITUDE_TARGET in mav_rx_msgids
        and MSGID_COMMAND_INT in mav_rx_msgids
        and MSGID_COMMAND_LONG in mav_rx_msgids
        and ekf3_42007_emitted
        and mav_msg84_never
        and mav_bad_rejected
        and fg_tx_valid
        and fg_nan_rejected
    )
    return {
        "name": "closed_loop_rx_autonomy_tx",
        "pass": passed,
        "tx_packets": total_tx,
        "mavlink2_tx_packets": mav_tx_packets,
        "flightgear_tx_packets": fg_tx_packets,
        "tx_msgids_verified": sorted(set(mav_rx_msgids)),
        "ekf3_42007_emitted": ekf3_42007_emitted,
        "msg84_emitted": not mav_msg84_never,
        "bad_telemetry_and_msg84_rejected": bool(mav_bad_rejected and fg_nan_rejected),
        "flightgear_tx_clamped_ok": fg_tx_valid,
    }


def run_hil_loopback_self_test() -> dict[str, Any]:
    """Execute full 1-click local `127.0.0.1` UDP + closed-loop RX->TX wire loopback self-test."""
    t0 = time.perf_counter()
    c_mav = run_bearing_track_loopback(n_frames=100)
    c_dr06 = _verify_dr06_fixed_wing_safety_and_plugin_health()
    c_ar = _verify_active_range_64b_wire_and_tamper()
    c_fg = _verify_flightgear_udp_loopback(n_packets=10)
    c_closed = _verify_closed_loop_rx_tx_bridge()

    checks = [
        {
            "name": "mavlink2_52001_signed_udp_loopback",
            "pass": bool(c_mav["pass"]),
            "details": c_mav,
        },
        c_dr06,
        c_ar,
        c_fg,
    ]
    all_pass = all(bool(c["pass"]) for c in checks) and bool(c_closed["pass"])
    total_packets = (
        int(c_mav["crc_sig_verified"])
        + int(c_fg["telemetry_packets_verified"])
        + int(c_fg["control_packets_verified"])
        + int(c_closed["tx_packets"])
        + 1
    )
    tamper_rejected = bool(
        c_mav["tamper_cases_rejected"] == c_mav["tamper_cases_tested"]
        and c_ar["tamper_cases_rejected"] == c_ar["tamper_cases_tested"]
        and c_dr06["msg84_hard_rejected"]
        and c_closed["bad_telemetry_and_msg84_rejected"]
    )
    return {
        "pass": all_pass,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "packets_verified": total_packets,
        "mavlink_52001_packets_verified": int(c_mav["crc_sig_verified"]),
        "tamper_rejected": tamper_rejected,
        "gpl_free": True,
        "non_flight_sim_only": is_non_flight_sim_only(),
        "watermark": NON_FLIGHT_WATERMARK,
        "checks": checks,
        "closed_loop_tx": c_closed,
    }


class StudioHilBridgeManager:
    """Non-blocking closed-loop UDP HIL bridge (`RX -> Autonomy -> TX`) for FlightGear / MAVLink2."""

    def __init__(
        self,
        *,
        non_flight_sim_only: bool | None = None,
        datum_guard: DatumAlignmentGuard | None = None,
        shadow_logger: ShadowModeTelemetryLogger | None = None,
        camera_lever_arm_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        viso_delay_ms: float = 0.0,
    ) -> None:
        # Validate optional visual lever-arm & transport-delay bounds fail-closed
        _init_comp = compensate_visual_pose_ned(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.0,
            0.0,
            0.0,
            lever_arm_body_m=camera_lever_arm_body_m,
            transport_delay_ms=viso_delay_ms,
        )
        self._lock = threading.Lock()
        self._non_flight_sim_only_override: bool | None = non_flight_sim_only
        self._camera_lever_arm_body_m: tuple[float, float, float] = tuple(float(x) for x in camera_lever_arm_body_m)
        self._viso_delay_ms: float = float(_init_comp.transport_delay_ms)
        self._last_compensated_visual_pose: CompensatedVisualPoseNed | None = None
        self._mode: str = "idle"
        self._shadow_mode: bool = False
        self._shadow_logger: ShadowModeTelemetryLogger | None = shadow_logger
        self._shadow_last_mono_us: int = 0
        self._datum_guard: DatumAlignmentGuard | None = datum_guard
        self._visual_alt_m: float | None = None
        self._visual_datum: str = "WGS84_HAE"
        self._visual_geoid_undulation_m: float | None = None
        self._apply_geoid_conversion: bool = True
        self._last_datum_decision: DatumSwitchDecision | None = None
        self._datum_inhibited_switches: int = 0
        self._bound_host: str = "127.0.0.1"
        self._tx_host: str = "127.0.0.1"
        self._rx_port: int = DEFAULT_FG_RX_PORT
        self._tx_port: int = DEFAULT_FG_TX_PORT
        self._rx_packets: int = 0
        self._tx_packets: int = 0
        self._rejected_packets: int = 0
        self._heartbeat_count: int = 0
        self._attitude_count: int = 0
        self._global_pos_count: int = 0
        self._vfr_hud_count: int = 0
        self._vfr_hud_received: bool = False
        self._last_rx_time_boot_ms: dict[int, int] = {}
        self._reboot_candidate_boot_ms: dict[int, tuple[int, int]] = {}
        self._last_heartbeat_mode: int | None = None
        self._last_ack: dict[str, int] | None = None
        self._ack_history: list[dict[str, int]] = []
        self._last_telemetry_ned: dict[str, Any] | None = None
        self._last_tx_command: dict[str, Any] | None = None
        self._last_tx_latency_ms: float | None = None
        self._last_loopback: dict[str, Any] | None = None
        self._tx_msgids_emitted: dict[int, int] = {}
        self._msg84_ever_emitted: bool = False
        self._tx_seq: int = 0
        self._tx_command_mode: str = "set_attitude_target"
        self._tx_enabled: bool = not self._is_non_flight_locked()
        self._min_tx_interval_s: float = 0.0
        self._last_guidance_tx_mono_s: float | None = None
        self._secret_key: bytes | None = None
        self._send_from_rx_sock: bool = False
        self._stale_telemetry_timeout_s: float = DEFAULT_STALE_TELEMETRY_TIMEOUT_S
        self._connect_mono_s: float = time.monotonic()
        self._last_rx_mono_s: float | None = None
        self._cbdr_threat_active: bool = False
        self._cbdr_evade_direction: float = 1.0
        self._gnss_spoof_alarm: bool = False
        self._gnss_healthy: bool = True
        self._ar_step_count: int = 0
        self._active_range: ActiveRangeController = ActiveRangeController()
        self._ekf3_latch: Ekf3SourceSwitchLatch = Ekf3SourceSwitchLatch()
        self._state_ned: dict[str, float] = {
            "lat_deg": 37.61172,
            "lon_deg": -122.358367,
            "alt_msl_m": 180.0,
            "down_m": -180.0,
            "relative_alt_m": 180.0,
            "heading_deg": 0.0,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "airspeed_mps": 22.0,
            "groundspeed_mps": 22.0,
            "vn_mps": 22.0,
            "ve_mps": 0.0,
            "vd_mps": 0.0,
        }
        self._stop_event = threading.Event()
        self._sock: socket.socket | None = None
        self._tx_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def _is_non_flight_locked(self) -> bool:
        if self._non_flight_sim_only_override is not None:
            return bool(self._non_flight_sim_only_override)
        return is_non_flight_sim_only()

    def force_non_flight_lockout(self) -> dict[str, Any]:
        """Fail-closed disconnect any active UDP HIL link and force `_tx_enabled = False`."""
        with self._lock:
            old_sock, old_tx_sock, old_th = self._detach_listener_locked()
            self._mode = "idle"
            self._shadow_mode = False
            self._tx_enabled = False
        self._close_detached_listener(old_sock, old_tx_sock, old_th)
        with self._lock:
            return self._status_locked()

    def configure_datum_guard(
        self,
        guard: DatumAlignmentGuard | None,
        *,
        visual_alt_m: float | None = None,
        visual_datum: str = "WGS84_HAE",
        geoid_undulation_m: float | None = None,
        apply_geoid_conversion: bool = True,
    ) -> dict[str, Any]:
        """Attach or update `DatumAlignmentGuard` and visual vertical observation for `EKF3 42007`."""
        with self._lock:
            self._datum_guard = guard
            self._visual_alt_m = visual_alt_m
            self._visual_datum = str(visual_datum)
            self._visual_geoid_undulation_m = geoid_undulation_m
            self._apply_geoid_conversion = bool(apply_geoid_conversion)
            return self._status_locked()

    def update_visual_altitude(
        self,
        *,
        visual_alt_m: float | None,
        visual_datum: str = "WGS84_HAE",
        geoid_undulation_m: float | None = None,
        apply_geoid_conversion: bool = True,
    ) -> dict[str, Any]:
        """Update latest visual altitude estimate (`WGS84_HAE` or `MSL_*`) for pre-switch datum check."""
        with self._lock:
            self._visual_alt_m = visual_alt_m
            self._visual_datum = str(visual_datum)
            self._visual_geoid_undulation_m = geoid_undulation_m
            self._apply_geoid_conversion = bool(apply_geoid_conversion)
            return self._status_locked()

    def configure_visual_pose_compensation(
        self,
        *,
        lever_arm_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        viso_delay_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Configure camera-to-CG Body-FRD lever arm and `VISO_DELAY_MS` transport delay fail-closed."""
        chk = compensate_visual_pose_ned(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.0,
            0.0,
            0.0,
            lever_arm_body_m=lever_arm_body_m,
            transport_delay_ms=viso_delay_ms,
        )
        with self._lock:
            self._camera_lever_arm_body_m = tuple(float(x) for x in lever_arm_body_m)
            self._viso_delay_ms = float(chk.transport_delay_ms)
            return self._status_locked()

    def compensate_visual_fix_ned(
        self,
        p_cam_ned_m: tuple[float, float, float] | list[float],
        *,
        v_cg_ned_mps: tuple[float, float, float] | list[float] | None = None,
        roll_rad: float | None = None,
        pitch_rad: float | None = None,
        yaw_rad: float | None = None,
        lever_arm_body_m: tuple[float, float, float] | list[float] | None = None,
        transport_delay_ms: float | None = None,
    ) -> CompensatedVisualPoseNed:
        """Transform a camera-frame visual NED fix to aircraft CG at current time using live state."""
        with self._lock:
            st = dict(self._state_ned)
            eff_la = self._camera_lever_arm_body_m if lever_arm_body_m is None else lever_arm_body_m
            eff_delay = self._viso_delay_ms if transport_delay_ms is None else transport_delay_ms
        eff_v = (st["vn_mps"], st["ve_mps"], st["vd_mps"]) if v_cg_ned_mps is None else v_cg_ned_mps
        eff_roll = math.radians(st["roll_deg"]) if roll_rad is None else roll_rad
        eff_pitch = math.radians(st["pitch_deg"]) if pitch_rad is None else pitch_rad
        eff_yaw = math.radians(st["heading_deg"]) if yaw_rad is None else yaw_rad
        res = compensate_visual_pose_ned(
            p_cam_ned_m,
            eff_v,
            eff_roll,
            eff_pitch,
            eff_yaw,
            lever_arm_body_m=eff_la,
            transport_delay_ms=eff_delay,
        )
        with self._lock:
            self._last_compensated_visual_pose = res
        return res

    def _detach_listener_locked(self) -> tuple[socket.socket | None, socket.socket | None, threading.Thread | None]:
        self._stop_event.set()
        sock = self._sock
        tx_sock = self._tx_sock
        th = self._thread
        self._sock = None
        self._tx_sock = None
        self._thread = None
        return sock, tx_sock, th

    def _close_detached_listener(
        self,
        sock: socket.socket | None,
        tx_sock: socket.socket | None,
        th: threading.Thread | None,
    ) -> None:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if tx_sock is not None:
            try:
                tx_sock.close()
            except Exception:
                pass
        if th is not None and th.is_alive():
            th.join(timeout=1.5)

    def set_autonomy_triggers(
        self,
        *,
        cbdr_threat_active: bool | None = None,
        cbdr_evade_direction: float | None = None,
        gnss_spoof_alarm: bool | None = None,
        gnss_healthy: bool | None = None,
        tx_command_mode: str | None = None,
        tx_enabled: bool | None = None,
        min_tx_interval_s: float | None = None,
    ) -> dict[str, Any]:
        """Update closed-loop autonomy trigger inputs (CBDR evade, GNSS spoof EKF3 latch, TX mode)."""
        if tx_command_mode is not None:
            mode_clean = tx_command_mode.strip().lower()
            if mode_clean not in VALID_TX_COMMAND_MODES:
                raise ValueError(
                    f"Unsupported tx_command_mode {tx_command_mode!r}; expected one of {VALID_TX_COMMAND_MODES}"
                )
        else:
            mode_clean = None

        if cbdr_evade_direction is not None:
            ev = float(cbdr_evade_direction)
            if not math.isfinite(ev) or ev == 0.0:
                raise ValueError(f"cbdr_evade_direction must be finite non-zero, got {cbdr_evade_direction!r}")
            ev_sign = 1.0 if ev > 0.0 else -1.0
        else:
            ev_sign = None

        if min_tx_interval_s is not None:
            min_iv = float(min_tx_interval_s)
            if not math.isfinite(min_iv) or min_iv < 0.0:
                raise ValueError(f"min_tx_interval_s must be finite >= 0, got {min_tx_interval_s!r}")
        else:
            min_iv = None

        with self._lock:
            if cbdr_threat_active is not None:
                self._cbdr_threat_active = bool(cbdr_threat_active)
            if ev_sign is not None:
                self._cbdr_evade_direction = ev_sign
            if gnss_spoof_alarm is not None:
                self._gnss_spoof_alarm = bool(gnss_spoof_alarm)
            if gnss_healthy is not None:
                self._gnss_healthy = bool(gnss_healthy)
            if mode_clean is not None:
                self._tx_command_mode = mode_clean
                self._last_guidance_tx_mono_s = None
            if tx_enabled is not None:
                if self._shadow_mode and bool(tx_enabled):
                    self._tx_enabled = False
                    raise ShadowModeTxInhibitedError(
                        "Cannot enable TX while StudioHilBridgeManager is attached to a listen-only shadow tap"
                    )
                if self._is_non_flight_locked() and bool(tx_enabled):
                    self._tx_enabled = False
                    raise NonFlightSimOnlyError(EU_PLD_ART4_7_LOCKOUT_MSG)
                self._tx_enabled = False if (self._shadow_mode or self._is_non_flight_locked()) else bool(tx_enabled)
                if self._tx_enabled:
                    self._last_guidance_tx_mono_s = None
            if min_iv is not None:
                self._min_tx_interval_s = min_iv
            return self._status_locked()

    def is_telemetry_stale(self, *, now_mono: float | None = None) -> bool:
        """Return True when active bridge has received no valid telemetry within stale timeout."""
        cur = time.monotonic() if now_mono is None else float(now_mono)
        with self._lock:
            return self._is_telemetry_stale_locked(cur)

    def _is_telemetry_stale_locked(self, now_mono: float) -> bool:
        if self._mode == "idle":
            return False
        if self._last_rx_mono_s is None:
            return (now_mono - self._connect_mono_s) > self._stale_telemetry_timeout_s
        return (now_mono - self._last_rx_mono_s) > self._stale_telemetry_timeout_s

    def _can_emit_autonomous_tx_locked(self, tx_sock: socket.socket | None, now_mono: float) -> bool:
        return (
            not self._shadow_mode
            and self._tx_enabled
            and tx_sock is not None
            and not self._is_non_flight_locked()
            and self._last_rx_mono_s is not None
            and not self._is_telemetry_stale_locked(now_mono)
        )

    def _accept_rx_time_boot_ms_locked(self, msgid: int, time_boot_ms: int) -> bool:
        """Enforce per-msgid uint32 `time_boot_ms` monotonicity with wraparound & 3-tick reboot confirm."""
        ts = int(time_boot_ms) & 0xFFFFFFFF
        last_ts = self._last_rx_time_boot_ms.get(msgid)
        if last_ts is None:
            self._last_rx_time_boot_ms[msgid] = ts
            self._reboot_candidate_boot_ms.pop(msgid, None)
            return True

        if ts >= last_ts:
            if (ts - last_ts) >= _BOOT_MS_UINT32_HALF:
                # Delayed pre-wraparound frame arriving just after uint32 rollover
                return False
            self._last_rx_time_boot_ms[msgid] = ts
            self._reboot_candidate_boot_ms.pop(msgid, None)
            return True

        # Legitimate uint32 rollover near 2^32 ms (~49.7 days uptime)
        if last_ts >= (_BOOT_MS_UINT32_MOD - _BOOT_MS_WRAP_WINDOW_MS) and ts < _BOOT_MS_WRAP_WINDOW_MS:
            self._last_rx_time_boot_ms[msgid] = ts
            self._reboot_candidate_boot_ms.pop(msgid, None)
            return True

        drop_ms = last_ts - ts
        if drop_ms <= _BOOT_MS_REBOOT_DROP_MS:
            # Delayed/reordered frame within 30 s window: strictly reject
            self._reboot_candidate_boot_ms.pop(msgid, None)
            return False

        # Huge backward drop (> 30 s): reject single/replayed ancient frames; confirm reboot only
        # after 3 consecutive strictly-increasing fresh boot ticks.
        cand = self._reboot_candidate_boot_ms.get(msgid)
        if cand is not None:
            prev_cand_ts, count = cand
            if ts == prev_cand_ts:
                # Duplicate replayed ancient timestamp cannot chain into reboot confirmation
                self._reboot_candidate_boot_ms[msgid] = (ts, 0)
                return False
            if count > 0 and prev_cand_ts < ts <= (prev_cand_ts + _BOOT_MS_REBOOT_MAX_STEP_MS):
                next_count = count + 1
                if next_count >= _BOOT_MS_REBOOT_CONFIRM_TICKS:
                    self._last_rx_time_boot_ms[msgid] = ts
                    self._reboot_candidate_boot_ms.pop(msgid, None)
                    return True
                self._reboot_candidate_boot_ms[msgid] = (ts, next_count)
                return False

        self._reboot_candidate_boot_ms[msgid] = (ts, 1)
        return False

    def connect(
        self,
        *,
        mode: str = "idle",
        rx_port: int | None = None,
        tx_port: int | None = None,
        tx_host: str = "127.0.0.1",
        bind_host: str = "127.0.0.1",
        tx_command_mode: str = "set_attitude_target",
        secret_key: bytes | None = None,
        stale_telemetry_timeout_s: float = DEFAULT_STALE_TELEMETRY_TIMEOUT_S,
        send_from_rx_sock: bool = False,
        tx_enabled: bool = True,
        min_tx_interval_s: float = 0.0,
    ) -> dict[str, Any]:
        """Bind or stop the closed-loop UDP HIL bridge safely."""
        m = mode.strip().lower()
        if m not in VALID_HIL_MODES:
            raise ValueError(f"Unsupported HIL mode {mode!r}; expected one of {VALID_HIL_MODES}")

        if self._is_non_flight_locked():
            self.force_non_flight_lockout()
            if m != "idle":
                raise NonFlightSimOnlyError(EU_PLD_ART4_7_LOCKOUT_MSG)
            tx_enabled = False

        tx_mode = tx_command_mode.strip().lower()
        if tx_mode not in VALID_TX_COMMAND_MODES:
            raise ValueError(
                f"Unsupported tx_command_mode {tx_command_mode!r}; expected one of {VALID_TX_COMMAND_MODES}"
            )
        if secret_key is not None and len(secret_key) != 32:
            raise ValueError(f"MAVLink v2 secret_key must be exactly 32 bytes, got {len(secret_key)}")
        stale_to = float(stale_telemetry_timeout_s)
        if not math.isfinite(stale_to) or stale_to <= 0.0:
            raise ValueError(f"stale_telemetry_timeout_s must be finite > 0, got {stale_telemetry_timeout_s!r}")
        min_iv = float(min_tx_interval_s)
        if not math.isfinite(min_iv) or min_iv < 0.0:
            raise ValueError(f"min_tx_interval_s must be finite >= 0, got {min_tx_interval_s!r}")

        clean_bind = _validate_local_or_private_host(bind_host, allow_any_bind=True)
        clean_tx = _validate_local_or_private_host(tx_host, allow_any_bind=False)

        default_rx = DEFAULT_MAV_RX_PORT if m == "mavlink2_udp" else DEFAULT_FG_RX_PORT
        default_tx = DEFAULT_MAV_TX_PORT if m == "mavlink2_udp" else DEFAULT_FG_TX_PORT
        rp = default_rx if rx_port is None else int(rx_port)
        tp = default_tx if tx_port is None else int(tx_port)
        if not (1024 <= rp <= 65535) or not (1024 <= tp <= 65535):
            raise ValueError(f"UDP ports must be in [1024, 65535], got rx={rp}, tx={tp}")

        with self._lock:
            old_sock, old_tx_sock, old_th = self._detach_listener_locked()
        self._close_detached_listener(old_sock, old_tx_sock, old_th)

        with self._lock:
            self._stop_event.clear()
            self._mode = m
            self._shadow_mode = False
            self._last_datum_decision = None
            self._datum_inhibited_switches = 0
            self._bound_host = clean_bind
            self._tx_host = clean_tx
            self._rx_port = rp
            self._tx_port = tp
            self._tx_command_mode = tx_mode
            self._tx_enabled = bool(tx_enabled)
            self._min_tx_interval_s = min_iv
            self._last_guidance_tx_mono_s = None
            self._secret_key = bytes(secret_key) if secret_key is not None else None
            self._send_from_rx_sock = bool(send_from_rx_sock)
            self._stale_telemetry_timeout_s = stale_to
            self._connect_mono_s = time.monotonic()
            self._last_rx_mono_s = None
            self._rx_packets = 0
            self._tx_packets = 0
            self._rejected_packets = 0
            self._heartbeat_count = 0
            self._attitude_count = 0
            self._global_pos_count = 0
            self._vfr_hud_count = 0
            self._vfr_hud_received = False
            self._last_rx_time_boot_ms = {}
            self._reboot_candidate_boot_ms = {}
            self._last_heartbeat_mode = None
            self._last_ack = None
            self._ack_history = []
            self._last_tx_command = None
            self._last_tx_latency_ms = None
            self._tx_msgids_emitted = {}
            self._msg84_ever_emitted = False
            self._tx_seq = 0
            self._ar_step_count = 0
            self._active_range = ActiveRangeController()
            self._ekf3_latch = Ekf3SourceSwitchLatch()
            self._cbdr_threat_active = False
            self._cbdr_evade_direction = 1.0
            self._gnss_spoof_alarm = False
            self._gnss_healthy = True

            if m == "idle":
                return self._status_locked()

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((clean_bind, rp))
            sock.settimeout(0.25)

            tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock = sock
            self._tx_sock = tx_sock
            self._thread = threading.Thread(
                target=self._rx_loop,
                args=(sock, tx_sock, m),
                daemon=True,
            )
            self._thread.start()
            return self._status_locked()

    def connect_shadow_tap(
        self,
        *,
        shadow_logger: ShadowModeTelemetryLogger,
        mode: str = "mavlink2_udp",
        rx_port: int | None = None,
        bind_host: str = "127.0.0.1",
        secret_key: bytes | None = None,
        stale_telemetry_timeout_s: float = DEFAULT_STALE_TELEMETRY_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Bind strictly RX-only shadow tap mirroring hardware TX-pin-removed trials (`0` outbound TX).

        Safe under `OPENXVS_NON_FLIGHT_SIM_ONLY=1`: no TX socket is allocated, `_tx_enabled` is
        hard-locked `False`, `actuator_authority` is `False`, and every RX packet is appended to
        the SHA-256 hash-chained `ShadowModeTelemetryLogger`.
        """
        if not isinstance(shadow_logger, ShadowModeTelemetryLogger):
            raise TypeError("shadow_logger must be a ShadowModeTelemetryLogger instance")
        if shadow_logger.tx_inhibited is not True or shadow_logger.actuator_authority is not False:
            raise ShadowModeTxInhibitedError("shadow_logger violated listen-only hard invariants")

        m = mode.strip().lower()
        if m not in ("mavlink2_udp", "flightgear_udp"):
            raise ValueError(f"Unsupported shadow tap mode {mode!r}; expected 'mavlink2_udp' or 'flightgear_udp'")
        if secret_key is not None and len(secret_key) != 32:
            raise ValueError(f"MAVLink v2 secret_key must be exactly 32 bytes, got {len(secret_key)}")
        stale_to = float(stale_telemetry_timeout_s)
        if not math.isfinite(stale_to) or stale_to <= 0.0:
            raise ValueError(f"stale_telemetry_timeout_s must be finite > 0, got {stale_telemetry_timeout_s!r}")

        clean_bind = _validate_local_or_private_host(bind_host, allow_any_bind=True)
        default_rx = DEFAULT_MAV_RX_PORT if m == "mavlink2_udp" else DEFAULT_FG_RX_PORT
        rp = default_rx if rx_port is None else int(rx_port)
        if not (1024 <= rp <= 65535):
            raise ValueError(f"Shadow tap rx_port must be in [1024, 65535], got {rp}")

        with self._lock:
            old_sock, old_tx_sock, old_th = self._detach_listener_locked()
        self._close_detached_listener(old_sock, old_tx_sock, old_th)

        with self._lock:
            self._stop_event.clear()
            self._mode = m
            self._shadow_mode = True
            self._shadow_logger = shadow_logger
            self._shadow_last_mono_us = (
                shadow_logger.last_timestamp_mono_us
                if shadow_logger.last_timestamp_mono_us is not None
                else 0
            )
            self._bound_host = clean_bind
            self._tx_host = "127.0.0.1"
            self._rx_port = rp
            self._tx_port = 0
            self._tx_enabled = False
            self._send_from_rx_sock = False
            self._secret_key = bytes(secret_key) if secret_key is not None else None
            self._stale_telemetry_timeout_s = stale_to
            self._connect_mono_s = time.monotonic()
            self._last_rx_mono_s = None
            self._rx_packets = 0
            self._tx_packets = 0
            self._rejected_packets = 0
            self._heartbeat_count = 0
            self._attitude_count = 0
            self._global_pos_count = 0
            self._vfr_hud_count = 0
            self._vfr_hud_received = False
            self._last_rx_time_boot_ms = {}
            self._reboot_candidate_boot_ms = {}
            self._last_heartbeat_mode = None
            self._last_ack = None
            self._ack_history = []
            self._last_tx_command = None
            self._last_tx_latency_ms = None
            self._tx_msgids_emitted = {}
            self._msg84_ever_emitted = False
            self._tx_seq = 0
            self._ar_step_count = 0
            self._active_range = ActiveRangeController()
            self._ekf3_latch = Ekf3SourceSwitchLatch()
            self._last_datum_decision = None
            self._datum_inhibited_switches = 0

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((clean_bind, rp))
            sock.settimeout(0.25)

            self._sock = sock
            self._tx_sock = None
            self._thread = threading.Thread(
                target=self._rx_loop,
                args=(sock, None, m),
                daemon=True,
            )
            self._thread.start()
            return self._status_locked()

    def send_raw_tx_frame(self, frame_bytes: bytes, *, allow_when_stale: bool = False) -> int:
        """Send a pre-encoded MAVLink 2.0 frame to `(tx_host, tx_port)` with `#84` & stale-RX guards."""
        hdr = peek_mavlink_v2_header(frame_bytes)
        msgid = int(hdr["msgid"])
        if msgid == MSGID_SET_POSITION_TARGET_LOCAL_NED:
            raise ValueError("MAVLink #84 (SET_POSITION_TARGET_LOCAL_NED) is hard-banned on TX")
        if self._shadow_mode:
            raise ShadowModeTxInhibitedError(
                f"Outbound MAVLink TX (msgid={msgid}) blocked in listen-only shadow mode"
            )
        if self._is_non_flight_locked():
            raise NonFlightSimOnlyError(EU_PLD_ART4_7_LOCKOUT_MSG)
        with self._lock:
            sock = self._sock if self._send_from_rx_sock else self._tx_sock
            if sock is None or self._mode == "idle":
                raise RuntimeError("StudioHilBridgeManager is not connected")
            if not allow_when_stale and self._is_telemetry_stale_locked(time.monotonic()):
                raise StaleTelemetryError(
                    f"Outbound MAVLink TX (msgid={msgid}) blocked fail-closed: telemetry is stale "
                    f"(>{self._stale_telemetry_timeout_s:.2f} s without valid RX)"
                )
            dest = (self._tx_host, self._tx_port)
            sent = sock.sendto(frame_bytes, dest)
            self._tx_packets += 1
            prev = self._tx_msgids_emitted[msgid] if msgid in self._tx_msgids_emitted else 0
            self._tx_msgids_emitted[msgid] = prev + 1
            return int(sent)

    def _next_seq_locked(self) -> int:
        seq = self._tx_seq & 0xFF
        self._tx_seq = (self._tx_seq + 1) & 0xFF
        return seq

    def _compute_autonomy_step_locked(self, now_mono: float) -> dict[str, Any]:
        """Run deterministic `XVS-ActiveRange` + `Ekf3SourceSwitchLatch` step in strictly NED."""
        st = self._state_ned
        vn = float(st["vn_mps"])
        ve = float(st["ve_mps"])
        vd = float(st["vd_mps"])
        hdg_rad = math.radians(float(st["heading_deg"]))
        roll_rad = math.radians(float(st["roll_deg"]))
        tas = max(DEFAULT_V_MIN_MPS, min(DEFAULT_V_MAX_MPS, float(st["airspeed_mps"])))

        # Lateral centripetal accel estimate in NED from current bank angle
        a_lat = 9.80665 * math.tan(max(-DEFAULT_MAX_BANK_RAD, min(DEFAULT_MAX_BANK_RAD, roll_rad)))
        a_north = -math.sin(hdg_rad) * a_lat
        a_east = math.cos(hdg_rad) * a_lat
        own = OwnshipStateNed(
            v_ned=(vn, ve, vd),
            a_ned=(a_north, a_east, 0.0),
            heading_rad=hdg_rad,
            tas_mps=tas,
        )

        if self._cbdr_threat_active:
            self._ar_step_count += 1
            intruder_az = hdg_rad + 0.06 * self._cbdr_evade_direction
            meas = ActiveRangeMeasurement(
                alpha_rad=intruder_az,
                beta_rad=0.0,
                size_px=12.0 + 0.8 * min(self._ar_step_count, 20),
                size_rate_px_s=4.5,
                track_age=max(6, self._ar_step_count),
                dt_s=0.1,
            )
            ar_tel = self._active_range.step(own, meas)
            ar_bank = float(ar_tel.cmd_bank_rad)
            # If ActiveRange is still converging (< 6 deg), apply bounded CBDR evade bank (25 deg <= 30 deg cap)
            if abs(ar_bank) < math.radians(6.0):
                cmd_bank_rad = self._cbdr_evade_direction * EVADE_BANK_RAD
            else:
                cmd_bank_rad = ar_bank
            cmd_bank_rad = max(-DEFAULT_MAX_BANK_RAD, min(DEFAULT_MAX_BANK_RAD, cmd_bank_rad))
            cmd_pitch_rad = math.radians(2.0)
            shield_active = True
            ar_mode_name = ActiveRangeMode(ar_tel.mode).name
        else:
            cmd_bank_rad = 0.0
            cmd_pitch_rad = 0.0
            shield_active = False
            ar_mode_name = ActiveRangeMode.NOMINAL_HOLD.name

        latch_ts = max(0.0, now_mono - self._connect_mono_s)
        datum_decision: DatumSwitchDecision | None = None
        if self._datum_guard is not None:
            prev_src = self._ekf3_latch.active_source
            wants_vision = (prev_src != EKF3_SOURCE_VISION_OPTICAL) and (
                self._gnss_spoof_alarm or not self._gnss_healthy
            )
            ekf_state, datum_decision = self._datum_guard.gate_ekf3_switch_latch(
                self._ekf3_latch,
                timestamp_s=latch_ts,
                gnss_healthy=self._gnss_healthy,
                spoof_alarm=self._gnss_spoof_alarm,
                z_visual_m=self._visual_alt_m,
                z_baro_msl_m=float(st["alt_msl_m"]),
                visual_datum=self._visual_datum,
                geoid_undulation_m=self._visual_geoid_undulation_m,
                apply_geoid_conversion=self._apply_geoid_conversion,
            )
            self._last_datum_decision = datum_decision
            if wants_vision and not datum_decision.allowed:
                self._datum_inhibited_switches += 1
        else:
            ekf_state = self._ekf3_latch.step(
                timestamp_s=latch_ts,
                gnss_healthy=self._gnss_healthy,
                spoof_alarm=self._gnss_spoof_alarm,
            )
        return {
            "cmd_bank_rad": cmd_bank_rad,
            "cmd_pitch_rad": cmd_pitch_rad,
            "cmd_airspeed_mps": tas,
            "shield_active": shield_active,
            "active_range_mode": ar_mode_name,
            "ekf_state": ekf_state,
            "datum_decision": datum_decision,
        }

    def _build_mavlink_tx_packets_locked(
        self,
        autonomy: dict[str, Any],
        now_mono: float,
    ) -> list[tuple[int, bytes, dict[str, Any]]]:
        """Build clamped zero-GPL MAVLink 2.0 TX packets (`#76(42007)`, `#75`, or `#82`; NEVER `#84`)."""
        packets: list[tuple[int, bytes, dict[str, Any]]] = []
        ekf_state = autonomy["ekf_state"]
        ts48 = int(max(0.0, now_mono) * 100_000.0) & ((1 << 48) - 1)
        time_boot_ms = int(max(0.0, now_mono - self._connect_mono_s) * 1000.0) & 0xFFFFFFFF

        # 1. Latch-gated EKF3 42007 source switch (#76 COMMAND_LONG)
        if ekf_state.mav_cmd_id == ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET:
            seq = self._next_seq_locked()
            ekf_frame = encode_ekf3_source_switch_frame(
                ekf_state,
                seq=seq,
                sysid=255,
                compid=191,
                target_system=1,
                target_component=1,
                secret_key=self._secret_key,
                link_id=0,
                timestamp_48=ts48,
            )
            packets.append(
                (
                    MSGID_COMMAND_LONG,
                    ekf_frame,
                    {
                        "msgid": MSGID_COMMAND_LONG,
                        "mav_cmd": ARDUPILOT_MAV_CMD_SET_EKF_SOURCE_SET,
                        "ekf3_source": int(ekf_state.commanded_source),
                        "wire_bytes": len(ekf_frame),
                        "signed": self._secret_key is not None,
                    },
                )
            )

        # 2. Envelope-clamped fixed-wing avoidance frame (#75 COMMAND_INT or #82 SET_ATTITUDE_TARGET)
        if self._min_tx_interval_s > 0.0 and self._last_guidance_tx_mono_s is not None:
            if (now_mono - self._last_guidance_tx_mono_s) < self._min_tx_interval_s:
                return packets
        self._last_guidance_tx_mono_s = now_mono

        st = self._state_ned
        bank_rad = float(autonomy["cmd_bank_rad"])
        pitch_rad = float(autonomy["cmd_pitch_rad"])
        tas = float(autonomy["cmd_airspeed_mps"])
        hdg_deg = float(st["heading_deg"])

        if self._tx_command_mode == "command_int":
            # Offset waypoint 800 m forward + 450 m lateral in NED evade direction
            lat_shift_deg = hdg_deg + (35.0 * self._cbdr_evade_direction if autonomy["shield_active"] else 0.0)
            lat_shift_rad = math.radians(lat_shift_deg)
            tgt_lat, tgt_lon = _offset_latlon_ned(
                float(st["lat_deg"]),
                float(st["lon_deg"]),
                800.0 * math.cos(lat_shift_rad),
                800.0 * math.sin(lat_shift_rad),
            )
            cmd75 = build_command_int_avoidance(
                lat_deg=tgt_lat,
                lon_deg=tgt_lon,
                alt_m=max(50.0, float(st["relative_alt_m"])),
                airspeed_mps=tas,
                bank_cmd_rad=bank_rad,
                mav_cmd=MAV_CMD_DO_REPOSITION,
                loiter_radius_m=0.0,
            )
            validate_fixed_wing_avoidance_msgid(cmd75.msg_id)
            seq = self._next_seq_locked()
            wire = encode_command_int_frame(
                cmd75,
                seq=seq,
                sysid=255,
                compid=191,
                secret_key=self._secret_key,
                link_id=0,
                timestamp_48=ts48,
            )
            packets.append(
                (
                    MSGID_COMMAND_INT,
                    wire,
                    {
                        "msgid": MSGID_COMMAND_INT,
                        "mav_cmd": MAV_CMD_DO_REPOSITION,
                        "bank_cmd_deg": round(math.degrees(cmd75.bank_cmd_rad), 3),
                        "airspeed_cmd_mps": round(cmd75.airspeed_mps, 3),
                        "target_lat_deg": round(tgt_lat, 7),
                        "target_lon_deg": round(tgt_lon, 7),
                        "shield_active": bool(autonomy["shield_active"]),
                        "active_range_mode": str(autonomy["active_range_mode"]),
                        "ekf3_source": int(ekf_state.active_source),
                        "wire_bytes": len(wire),
                        "signed": self._secret_key is not None,
                    },
                )
            )
        elif self._tx_command_mode == "command_int_heading":
            target_hdg = (hdg_deg + (40.0 * self._cbdr_evade_direction if autonomy["shield_active"] else 0.0)) % 360.0
            cmd75_h = build_command_int_guided_heading(
                target_heading_deg=target_hdg,
                airspeed_mps=tas,
                bank_cmd_rad=bank_rad if autonomy["shield_active"] else math.radians(10.0),
                lat_deg=float(st["lat_deg"]),
                lon_deg=float(st["lon_deg"]),
                alt_m=max(50.0, float(st["relative_alt_m"])),
            )
            validate_fixed_wing_avoidance_msgid(cmd75_h.msg_id)
            seq = self._next_seq_locked()
            wire = encode_command_int_frame(
                cmd75_h,
                seq=seq,
                sysid=255,
                compid=191,
                secret_key=self._secret_key,
                link_id=0,
                timestamp_48=ts48,
            )
            packets.append(
                (
                    MSGID_COMMAND_INT,
                    wire,
                    {
                        "msgid": MSGID_COMMAND_INT,
                        "mav_cmd": MAV_CMD_GUIDED_CHANGE_HEADING,
                        "target_heading_deg": round(target_hdg, 2),
                        "bank_cmd_deg": round(math.degrees(cmd75_h.bank_cmd_rad), 3),
                        "airspeed_cmd_mps": round(cmd75_h.airspeed_mps, 3),
                        "shield_active": bool(autonomy["shield_active"]),
                        "active_range_mode": str(autonomy["active_range_mode"]),
                        "ekf3_source": int(ekf_state.active_source),
                        "wire_bytes": len(wire),
                        "signed": self._secret_key is not None,
                    },
                )
            )
        else:
            observed_ias_mps = float(st["airspeed_mps"])
            # When observed IAS dips below V_min, actively command climb-recovery thrust (0.80)
            # with type_mask=0x07 (never set 0x40 THROTTLE_IGNORE on ArduPlane 4.7.3 Guided mode).
            cmd_thrust = 0.80 if (observed_ias_mps < DEFAULT_V_MIN_MPS) else 0.65
            cmd82 = build_set_attitude_target(
                roll_rad=bank_rad,
                pitch_rad=pitch_rad,
                yaw_rad=math.radians(hdg_deg),
                thrust_norm=cmd_thrust,
                airspeed_mps=observed_ias_mps,
                time_boot_ms=time_boot_ms,
                preserve_tecs_throttle=False,
            )
            validate_fixed_wing_avoidance_msgid(cmd82.msg_id)
            seq = self._next_seq_locked()
            wire = encode_set_attitude_target_frame(
                cmd82,
                seq=seq,
                sysid=255,
                compid=191,
                time_boot_ms=time_boot_ms,
                secret_key=self._secret_key,
                link_id=0,
                timestamp_48=ts48,
            )
            packets.append(
                (
                    MSGID_SET_ATTITUDE_TARGET,
                    wire,
                    {
                        "msgid": MSGID_SET_ATTITUDE_TARGET,
                        "bank_cmd_deg": round(math.degrees(cmd82.roll_rad), 3),
                        "pitch_cmd_deg": round(math.degrees(cmd82.pitch_rad), 3),
                        "airspeed_cmd_mps": round(cmd82.airspeed_mps, 3),
                        "thrust_norm": round(cmd82.thrust_norm, 3),
                        "thrust_clamped": bool(cmd82.thrust_clamped),
                        "tecs_protected": bool(cmd82.tecs_protected),
                        "type_mask": int(cmd82.type_mask),
                        "q_ned_body": [round(c, 6) for c in cmd82.q_ned_body],
                        "shield_active": bool(autonomy["shield_active"]),
                        "active_range_mode": str(autonomy["active_range_mode"]),
                        "ekf3_source": int(ekf_state.active_source),
                        "wire_bytes": len(wire),
                        "signed": self._secret_key is not None,
                    },
                )
            )
        return packets

    def _next_shadow_mono_us_locked(self, now_mono: float) -> int:
        cand = int(max(0.0, now_mono) * 1_000_000.0)
        ts_us = max(self._shadow_last_mono_us + 1, cand)
        self._shadow_last_mono_us = ts_us
        return ts_us

    def _rx_loop(self, sock: socket.socket, tx_sock: socket.socket | None, mode: str) -> None:
        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(UDP_RX_BUFFER_BYTES)
            except socket.timeout:
                continue
            except ConnectionResetError:
                # Windows UDP WSAECONNRESET from ICMP Port Unreachable: ignore and keep listening
                continue
            except OSError as exc:
                err_no = getattr(exc, "errno", None)
                win_err = getattr(exc, "winerror", None)
                if self._stop_event.is_set() or sock.fileno() == -1:
                    break
                if err_no == errno.EMSGSIZE or win_err == 10040:
                    # Oversized UDP datagram must never kill the RX thread (B1 DoS protection)
                    with self._lock:
                        self._rejected_packets += 1
                    continue
                if err_no in (
                    errno.EAGAIN,
                    errno.EWOULDBLOCK,
                    errno.EINTR,
                    errno.ECONNRESET,
                    errno.ENOBUFS,
                ) or win_err in (10035, 10054, 10055):
                    continue
                if err_no in (errno.EBADF, errno.ENOTSOCK) or win_err in (10004, 10038):
                    break
                with self._lock:
                    self._rejected_packets += 1
                continue

            if len(data) == 0 or len(data) > MAX_ACCEPTED_UDP_DATAGRAM_BYTES:
                with self._lock:
                    self._rx_packets += 1
                    self._rejected_packets += 1
                continue

            t_start = time.perf_counter()
            now_mono = time.monotonic()

            if mode == "flightgear_udp":
                try:
                    fg = parse_flightgear_telemetry_csv(data)
                except Exception:
                    with self._lock:
                        self._rx_packets += 1
                        self._rejected_packets += 1
                    continue

                with self._lock:
                    self._rx_packets += 1
                    self._last_rx_mono_s = now_mono
                    self._state_ned.update(
                        {
                            "lat_deg": fg.lat_deg,
                            "lon_deg": fg.lon_deg,
                            "alt_msl_m": fg.alt_msl_m,
                            "down_m": fg.down_m,
                            "relative_alt_m": max(0.0, fg.alt_msl_m),
                            "heading_deg": fg.heading_deg,
                            "roll_deg": fg.roll_deg,
                            "pitch_deg": fg.pitch_deg,
                            "airspeed_mps": fg.airspeed_mps,
                            "groundspeed_mps": math.hypot(fg.vn_mps, fg.ve_mps),
                            "vn_mps": fg.vn_mps,
                            "ve_mps": fg.ve_mps,
                            "vd_mps": fg.vd_mps,
                        }
                    )
                    self._last_telemetry_ned = asdict(fg)
                    if self._shadow_mode and self._shadow_logger is not None:
                        ts_us = self._next_shadow_mono_us_locked(now_mono)
                        self._shadow_logger.record_event(
                            timestamp_mono_us=ts_us,
                            event_type="rx_flightgear_csv",
                            payload={"telemetry_ned": asdict(fg)},
                            direction="rx",
                            outbound_mavlink_frames=0,
                        )
                        continue
                    tx_now_mono = time.monotonic()
                    if not self._can_emit_autonomous_tx_locked(tx_sock, tx_now_mono):
                        continue
                    autonomy = self._compute_autonomy_step_locked(tx_now_mono)
                    tx_bytes = encode_flightgear_control_csv(
                        bank_cmd_deg=math.degrees(float(autonomy["cmd_bank_rad"])),
                        pitch_cmd_deg=math.degrees(float(autonomy["cmd_pitch_rad"])),
                        airspeed_cmd_mps=float(autonomy["cmd_airspeed_mps"]),
                        shield_active=bool(autonomy["shield_active"]),
                    )
                    tx_dest = (self._tx_host, self._tx_port)
                    active_tx_sock = sock if self._send_from_rx_sock else tx_sock
                    assert active_tx_sock is not None
                    try:
                        active_tx_sock.sendto(tx_bytes, tx_dest)
                    except OSError:
                        continue
                    self._tx_packets += 1
                    self._last_tx_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
                    self._last_tx_command = {
                        "protocol": "flightgear_csv",
                        "bank_cmd_deg": round(math.degrees(float(autonomy["cmd_bank_rad"])), 3),
                        "pitch_cmd_deg": round(math.degrees(float(autonomy["cmd_pitch_rad"])), 3),
                        "airspeed_cmd_mps": round(float(autonomy["cmd_airspeed_mps"]), 3),
                        "shield_active": bool(autonomy["shield_active"]),
                        "active_range_mode": str(autonomy["active_range_mode"]),
                        "wire_bytes": len(tx_bytes),
                    }

            elif mode == "mavlink2_udp":
                raw_frames, _ = extract_mavlink_v2_frames(data)
                if not raw_frames:
                    with self._lock:
                        self._rx_packets += 1
                        self._rejected_packets += 1
                    continue

                secret_key = self._secret_key
                for raw_frame in raw_frames:
                    try:
                        hdr = peek_mavlink_v2_header(raw_frame)
                    except Exception:
                        with self._lock:
                            self._rx_packets += 1
                            self._rejected_packets += 1
                        continue

                    mid = int(hdr["msgid"])
                    if mid == MSGID_SET_POSITION_TARGET_LOCAL_NED:
                        with self._lock:
                            self._rx_packets += 1
                            self._rejected_packets += 1
                        continue

                    if mid not in (
                        MSGID_HEARTBEAT,
                        MSGID_ATTITUDE,
                        MSGID_GLOBAL_POSITION_INT,
                        MSGID_VFR_HUD,
                        MSGID_COMMAND_ACK,
                        50001,
                        50002,
                        50003,
                        50004,
                    ):
                        # Non-subscribed ArduPlane telemetry stream frame: enforce signing if configured
                        if secret_key is not None and not int(hdr["signed"]):
                            with self._lock:
                                self._rx_packets += 1
                                self._rejected_packets += 1
                            continue
                        with self._lock:
                            self._rx_packets += 1
                        continue

                    try:
                        dec = decode_frame(
                            raw_frame,
                            verify_inner_crc32=True,
                            secret_key=secret_key,
                            require_signature=(secret_key is not None),
                        )
                    except Exception:
                        with self._lock:
                            self._rx_packets += 1
                            self._rejected_packets += 1
                        continue

                    tx_batch: list[tuple[int, bytes, dict[str, Any]]] = []
                    tx_dest: tuple[str, int] = ("127.0.0.1", DEFAULT_MAV_TX_PORT)
                    active_tx_sock = tx_sock

                    with self._lock:
                        self._rx_packets += 1
                        msg = dec.message

                        # Anti-replay / monotonicity check on time_boot_ms BEFORE mutating state or freshness (B5)
                        # Note: MAVLink #74 VFR_HUD carries no time_boot_ms wire field per common.xml.
                        if isinstance(msg, (MavAttitude, MavGlobalPositionInt)):
                            if not self._accept_rx_time_boot_ms_locked(int(dec.msgid), int(msg.time_boot_ms)):
                                self._rejected_packets += 1
                                continue

                        should_emit_tx = False
                        if isinstance(msg, MavHeartbeat):
                            self._heartbeat_count += 1
                            self._last_heartbeat_mode = int(msg.custom_mode)
                        elif isinstance(msg, MavCommandAck):
                            ack_rec = {
                                "command": int(msg.command),
                                "result": int(msg.result),
                            }
                            self._last_ack = ack_rec
                            self._ack_history.append(ack_rec)
                            if len(self._ack_history) > 64:
                                self._ack_history = self._ack_history[-64:]
                        elif isinstance(msg, MavAttitude):
                            self._attitude_count += 1
                            self._last_rx_mono_s = now_mono
                            self._state_ned["roll_deg"] = math.degrees(msg.roll)
                            self._state_ned["pitch_deg"] = math.degrees(msg.pitch)
                            self._state_ned["heading_deg"] = math.degrees(msg.yaw) % 360.0
                            self._last_telemetry_ned = {
                                "msgid": MSGID_ATTITUDE,
                                "seq": int(dec.seq),
                                "roll_deg": round(self._state_ned["roll_deg"], 4),
                                "pitch_deg": round(self._state_ned["pitch_deg"], 4),
                                "heading_deg": round(self._state_ned["heading_deg"], 4),
                                "down_m": round(self._state_ned["down_m"], 3),
                            }
                            should_emit_tx = True
                        elif isinstance(msg, MavGlobalPositionInt):
                            self._global_pos_count += 1
                            self._last_rx_mono_s = now_mono
                            gs = math.hypot(msg.vn_mps, msg.ve_mps)
                            gpi_update: dict[str, float] = {
                                "lat_deg": msg.lat_deg,
                                "lon_deg": msg.lon_deg,
                                "alt_msl_m": msg.alt_msl_m,
                                "down_m": msg.down_m,
                                "relative_alt_m": msg.relative_alt_m,
                                "heading_deg": msg.heading_deg,
                                "vn_mps": msg.vn_mps,
                                "ve_mps": msg.ve_mps,
                                "vd_mps": msg.vd_mps,
                                "groundspeed_mps": gs,
                            }
                            if not self._vfr_hud_received:
                                # Fallback to truthful groundspeed without fake 18 m/s stall-masking floor (B7)
                                gpi_update["airspeed_mps"] = gs
                            self._state_ned.update(gpi_update)
                            self._last_telemetry_ned = {
                                "msgid": MSGID_GLOBAL_POSITION_INT,
                                "seq": int(dec.seq),
                                "lat_deg": round(msg.lat_deg, 7),
                                "lon_deg": round(msg.lon_deg, 7),
                                "alt_msl_m": round(msg.alt_msl_m, 3),
                                "down_m": round(msg.down_m, 3),
                                "heading_deg": round(msg.heading_deg, 3),
                                "vn_mps": round(msg.vn_mps, 3),
                                "ve_mps": round(msg.ve_mps, 3),
                                "vd_mps": round(msg.vd_mps, 3),
                                "groundspeed_mps": round(gs, 3),
                            }
                            should_emit_tx = True
                        elif isinstance(msg, MavVfrHud):
                            self._vfr_hud_count += 1
                            self._vfr_hud_received = True
                            self._last_rx_mono_s = now_mono
                            self._state_ned["airspeed_mps"] = float(msg.airspeed)
                            self._state_ned["groundspeed_mps"] = float(msg.groundspeed)
                            self._last_telemetry_ned = {
                                "msgid": MSGID_VFR_HUD,
                                "seq": int(dec.seq),
                                "airspeed_mps": round(float(msg.airspeed), 3),
                                "groundspeed_mps": round(float(msg.groundspeed), 3),
                                "heading_deg": int(msg.heading),
                                "throttle_pct": int(msg.throttle),
                                "alt_msl_m": round(float(msg.alt), 3),
                                "climb_mps": round(float(msg.climb), 3),
                            }
                        else:
                            self._last_rx_mono_s = now_mono
                            self._last_telemetry_ned = {
                                "msgid": int(dec.msgid),
                                "seq": int(dec.seq),
                                "bytes": len(raw_frame),
                            }

                        if self._shadow_mode and self._shadow_logger is not None:
                            datum_dec: DatumSwitchDecision | None = None
                            if (
                                self._datum_guard is not None
                                and isinstance(msg, MavGlobalPositionInt)
                            ):
                                datum_dec = self._datum_guard.evaluate_ekf3_switch(
                                    z_visual_m=self._visual_alt_m,
                                    z_baro_msl_m=float(self._state_ned["alt_msl_m"]),
                                    visual_datum=self._visual_datum,
                                    geoid_undulation_m=self._visual_geoid_undulation_m,
                                    apply_geoid_conversion=self._apply_geoid_conversion,
                                )
                                self._last_datum_decision = datum_dec
                            if isinstance(msg, MavHeartbeat):
                                frame_summary: dict[str, Any] = {
                                    "msgid": MSGID_HEARTBEAT,
                                    "custom_mode": int(msg.custom_mode),
                                    "system_status": int(msg.system_status),
                                }
                            elif isinstance(msg, MavCommandAck):
                                frame_summary = {
                                    "msgid": MSGID_COMMAND_ACK,
                                    "command": int(msg.command),
                                    "result": int(msg.result),
                                }
                            elif self._last_telemetry_ned is not None:
                                frame_summary = dict(self._last_telemetry_ned)
                            else:
                                frame_summary = {"msgid": int(dec.msgid)}
                            ts_us = self._next_shadow_mono_us_locked(now_mono)
                            self._shadow_logger.record_rx_mavlink_frame(
                                timestamp_mono_us=ts_us,
                                msgid=int(dec.msgid),
                                seq=int(dec.seq),
                                raw_frame=raw_frame,
                                summary_ned=frame_summary,
                                datum_decision=datum_dec,
                            )
                            continue

                        tx_now_mono = time.monotonic()
                        if should_emit_tx and self._can_emit_autonomous_tx_locked(tx_sock, tx_now_mono):
                            autonomy = self._compute_autonomy_step_locked(tx_now_mono)
                            tx_batch = self._build_mavlink_tx_packets_locked(autonomy, tx_now_mono)
                            tx_dest = (self._tx_host, self._tx_port)
                            active_tx_sock = sock if self._send_from_rx_sock else tx_sock
                            assert active_tx_sock is not None
                            for tx_msgid, tx_wire, tx_summary in tx_batch:
                                if tx_msgid == MSGID_SET_POSITION_TARGET_LOCAL_NED:
                                    # Absolute fail-closed barrier: #84 can never be transmitted
                                    continue
                                try:
                                    active_tx_sock.sendto(tx_wire, tx_dest)
                                except OSError:
                                    continue
                                self._tx_packets += 1
                                prev_c = (
                                    self._tx_msgids_emitted[tx_msgid]
                                    if tx_msgid in self._tx_msgids_emitted
                                    else 0
                                )
                                self._tx_msgids_emitted[tx_msgid] = prev_c + 1
                                self._last_tx_command = tx_summary
                                self._last_tx_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

    def run_loopback_test(self) -> dict[str, Any]:
        """Run 1-click loopback self-test and cache result on manager."""
        report = run_hil_loopback_self_test()
        with self._lock:
            self._last_loopback = report
        return report

    def _status_locked(self) -> dict[str, Any]:
        locked = self._is_non_flight_locked()
        effective_tx = False if (locked or self._shadow_mode) else bool(self._tx_enabled)
        return {
            "mode": self._mode if self._shadow_mode else ("idle" if locked else self._mode),
            "shadow_mode": bool(self._shadow_mode),
            "tx_inhibited": bool(self._shadow_mode or locked or not effective_tx),
            "bound_host": self._bound_host,
            "tx_host": self._tx_host,
            "rx_port": self._rx_port,
            "tx_port": self._tx_port,
            "rx_packets": self._rx_packets,
            "tx_packets": self._tx_packets,
            "rejected_packets": self._rejected_packets,
            "heartbeat_count": self._heartbeat_count,
            "attitude_count": self._attitude_count,
            "global_pos_count": self._global_pos_count,
            "vfr_hud_count": self._vfr_hud_count,
            "airspeed_source": "vfr_hud" if self._vfr_hud_received else "groundspeed_fallback",
            "last_heartbeat_mode": self._last_heartbeat_mode,
            "last_ack": self._last_ack,
            "ack_history": list(self._ack_history),
            "state_ned": dict(self._state_ned),
            "telemetry_stale": self._is_telemetry_stale_locked(time.monotonic()),
            "ekf3_active_source": int(self._ekf3_latch.active_source),
            "datum_guard_active": bool(self._datum_guard is not None),
            "datum_inhibited_switches": int(self._datum_inhibited_switches),
            "last_datum_decision": (
                self._last_datum_decision.to_dict() if self._last_datum_decision is not None else None
            ),
            "camera_lever_arm_body_m": list(self._camera_lever_arm_body_m),
            "viso_delay_ms": float(self._viso_delay_ms),
            "last_compensated_visual_pose": (
                self._last_compensated_visual_pose.to_dict()
                if self._last_compensated_visual_pose is not None
                else None
            ),
            "tx_enabled": effective_tx,
            "actuator_authority": effective_tx,
            "non_flight_sim_only": locked,
            "watermark": NON_FLIGHT_WATERMARK,
            "tx_command_mode": self._tx_command_mode,
            "tx_msgids_emitted": dict(self._tx_msgids_emitted),
            "msg84_ever_emitted": bool(self._msg84_ever_emitted),
            "last_tx_command": self._last_tx_command,
            "last_tx_latency_ms": self._last_tx_latency_ms,
            "last_telemetry_ned": self._last_telemetry_ned,
            "last_loopback": self._last_loopback,
            "gpl_free": True,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def shutdown(self) -> None:
        with self._lock:
            old_sock, old_tx_sock, old_th = self._detach_listener_locked()
            self._mode = "idle"
            self._shadow_mode = False
            logger = self._shadow_logger
        self._close_detached_listener(old_sock, old_tx_sock, old_th)
        if logger is not None:
            logger.flush()
