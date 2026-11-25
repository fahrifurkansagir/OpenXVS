"""Unified WP-05 -> Tier-0 Standard Avionics Output Bridge (Batch B10 Phase 2 — ROADMAP.md §3).

Consumes either:
1. Live WP-05 G1 `Track` + `RiskEstimate` pairs via `PassiveTrackInput.from_track_and_risk`
   and `PassiveTrackAvionicsBridge.step_from_tracker` (`B10-m2`),
2. WP-05 G1 `WC1` well-clear track/risk states (`wc1_warning`, `wc1_corrective`,
   `R_min_m`, `R_max_m`, `hmd_hat_m`, `tau_mod_hat_s`, `tau_s_hat_s`), or
3. WP-05 G0 `A1`/`A2` angular-expansion track/risk states (`a1_warning`, `a2_warning`,
   `a1_corrective`, `a2_corrective`, `tau_s`, `omega_los`),
coupled with WP-07 Horizon Integrity state gating (`VALID`, `DEGRADED_COAST`, `INVALID_ALERT`),
and emits synchronized:
- 1 Hz GDL-90 Heartbeat (`0x00`),
- 10 Hz GDL-90 Ownship Report (`0x0A`) + Ownship Geometric Altitude (`0x0B`),
- 10 Hz GDL-90 Traffic Report (`0x14`) per confirmed track,
- 10 Hz MAVLink v2 `ADSB_VEHICLE` (`#246`, `CRC_EXTRA = 184`) per confirmed track
over non-blocking localhost UDP sockets.

Enforces the V5_B10 truth-isolation invariant: never imports `xvs.traffic` and
never reads any ground-truth intruder state (`truth_id`, true range, or true position).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from xvs.avionics.gdl90 import (
    GDL90_MSG_ID_OWNSHIP_REPORT,
    GDL90_MSG_ID_TRAFFIC_REPORT,
    Gdl90TargetReport,
    Gdl90UdpEndpoint,
    encode_heartbeat,
    encode_ownship_geo_altitude,
    encode_target_report,
    map_range_span_to_nic_nacp,
)
from xvs.avionics.mavlink_adsb import (
    MavlinkAdsbUdpStreamer,
    MavlinkAdsbVehicle,
    compute_passive_track_continuous_geodetic,
    pack_adsb_vehicle,
    passive_track_to_adsb_vehicle,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TIER0_SPEC_PATH = REPO_ROOT / "configs" / "xvs" / "avionics" / "tier0_spec.json"

METERS_TO_FEET: float = 1.0 / 0.3048
MPS_TO_KNOTS: float = 1.0 / 0.5144444444444445
MPS_TO_FPM: float = METERS_TO_FEET * 60.0

VALID_INTEGRITY_STATES = frozenset({"VALID", "DEGRADED_COAST", "INVALID_ALERT"})

DEFAULT_EMITTER_CATEGORY_MAP: dict[str, int] = {
    "C1_UAS": 14,
    "C2_SEP": 1,
    "C3_TWIN": 2,
    "C4_BIZJET": 3,
    "C5_TRANSPORT": 5,
    "C6_HELI": 7,
    "UNKNOWN": 0,
}

PASSIVE_CLASS_SIZE_BOUNDS_M: dict[str, tuple[float, float]] = {
    "C1_UAS": (0.30, 1.20),
    "C2_SEP": (7.00, 12.00),
    "C3_TWIN": (10.00, 18.00),
    "C4_BIZJET": (14.00, 22.00),
    "C5_TRANSPORT": (28.00, 40.00),
    "C6_HELI": (10.00, 14.00),
}
PASSIVE_UNION_SIZE_BOUNDS_M: tuple[float, float] = (0.30, 40.00)


def _extract_track_number(track_id: str | int) -> int:
    if isinstance(track_id, int):
        return max(1, track_id)
    m = re.search(r"(\d+)", str(track_id))
    if m:
        return max(1, int(m.group(1)))
    return 1


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass(frozen=True)
class OwnshipNavState:
    """Ownship INS/GNSS navigation state (cockpit ownship reference)."""

    lat_deg: float = 37.6188056
    lon_deg: float = -122.3754167
    alt_msl_m: float = 1524.0
    alt_geo_m: float = 1530.0
    vn_mps: float = 95.0
    ve_mps: float = 0.0
    vd_mps: float = 0.0
    heading_deg: float = 0.0
    icao_address: int = 0xABCDEF
    callsign: str = "XVSOWN01"
    vfom_m: float = 3.0
    nic: int = 10
    nacp: int = 10
    emitter_category: int = 2

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> OwnshipNavState:
        own = spec.get("ownship_reference", {})
        return cls(
            lat_deg=float(own.get("lat_deg", 37.6188056)),  # default-ok: optional avionics config/state fallback
            lon_deg=float(own.get("lon_deg", -122.3754167)),  # default-ok: optional avionics config/state fallback
            alt_msl_m=float(own.get("alt_msl_m", 1524.0)),  # default-ok: optional avionics config/state fallback
            alt_geo_m=float(own.get("alt_geo_m", 1530.0)),  # default-ok: optional avionics config/state fallback
            vn_mps=float(own.get("vn_mps", 95.0)),  # default-ok: optional avionics config/state fallback
            ve_mps=float(own.get("ve_mps", 0.0)),  # default-ok: optional avionics config/state fallback
            vd_mps=float(own.get("vd_mps", 0.0)),  # default-ok: optional avionics config/state fallback
            heading_deg=float(own.get("heading_deg", 0.0)),  # default-ok: optional avionics config/state fallback
            icao_address=int(own.get("icao_address", 0xABCDEF)),  # default-ok: optional avionics config/state fallback
            callsign=str(own.get("callsign", "XVSOWN01")),
            vfom_m=float(own.get("vfom_m", 3.0)),  # default-ok: optional avionics config/state fallback
        )


@dataclass(frozen=True)
class PassiveTrackInput:
    """Passive WP-05 track & collision-risk input state (WC1 primary, A1/A2 fallback).

    Contains ONLY passive bearing/expansion estimates and passive risk flags.
    """

    track_id: str
    t_s: float
    az_rad: float
    el_rad: float
    az_rate_radps: float = 0.0
    el_rate_radps: float = 0.0
    s_rad: float = 0.005
    s_dot_radps: float = 0.0
    r_min_m: float | None = None
    r_max_m: float | None = None
    hmd_hat_m: float | None = None
    tau_mod_hat_s: float | None = None
    tau_s_hat_s: float | None = None
    tau_s: float | None = None
    omega_los_radps: float | None = None
    wc1_warning: bool = False
    wc1_corrective: bool = False
    a1_warning: bool = False
    a1_corrective: bool = False
    a2_warning: bool = False
    a2_corrective: bool = False
    class_id: str | None = None
    tslc_s: int = 0

    @property
    def track_number(self) -> int:
        return _extract_track_number(self.track_id)

    @property
    def callsign(self) -> str:
        return f"XVS{self.track_number % 10000:04d}"

    @property
    def participant_address(self) -> int:
        return 0xF00000 | (self.track_number & 0x0FFFFF)

    @property
    def alert_active(self) -> bool:
        """Traffic Alert Status bit s=1 when WC1 or A1/A2 warning/corrective is active."""
        return bool(
            self.wc1_warning
            or self.wc1_corrective
            or self.a1_warning
            or self.a2_warning
            or self.a1_corrective
            or self.a2_corrective
        )

    @property
    def alert_rule_source(self) -> str:
        if self.wc1_warning or self.wc1_corrective:
            return "WC1"
        if self.a2_warning or self.a2_corrective:
            return "A2"
        if self.a1_warning or self.a1_corrective:
            return "A1"
        return "NONE"

    @property
    def effective_tau_s(self) -> float | None:
        for candidate in (self.tau_mod_hat_s, self.tau_s_hat_s, self.tau_s):
            if candidate is not None and math.isfinite(candidate) and candidate > 0.0:
                return float(candidate)
        if self.s_rad > 0.0 and self.s_dot_radps > 1e-6:
            return float(self.s_rad / self.s_dot_radps)
        return None

    def resolve_passive_range_interval_m(self) -> tuple[float, float]:
        """Return passive range interval [R_min_m, R_max_m] in metres."""
        if (
            self.r_min_m is not None
            and self.r_max_m is not None
            and math.isfinite(self.r_min_m)
            and math.isfinite(self.r_max_m)
            and self.r_min_m > 0.0
            and self.r_max_m >= self.r_min_m
        ):
            return (float(self.r_min_m), float(self.r_max_m))

        s_eff = max(1e-5, float(self.s_rad))
        if self.class_id and self.class_id in PASSIVE_CLASS_SIZE_BOUNDS_M:
            l_min, l_max = PASSIVE_CLASS_SIZE_BOUNDS_M[self.class_id]
        else:
            l_min, l_max = PASSIVE_UNION_SIZE_BOUNDS_M
        return (l_min / s_eff, l_max / s_eff)

    @classmethod
    def from_track_state(cls, obj: Any) -> PassiveTrackInput:
        """Construct `PassiveTrackInput` from a passive dict or object without reading truth keys."""
        if isinstance(obj, cls):
            return obj

        r_min = _get_field(obj, "R_min_m", _get_field(obj, "r_min_m"))
        r_max = _get_field(obj, "R_max_m", _get_field(obj, "r_max_m"))
        r_interval = _get_field(obj, "range_interval_m")
        if (
            (r_min is None or r_max is None)
            and isinstance(r_interval, (tuple, list))
            and len(r_interval) == 2
        ):
            r_min = float(r_interval[0])
            r_max = float(r_interval[1])

        return cls(
            track_id=str(_get_field(obj, "track_id", "T0001")),
            t_s=float(_get_field(obj, "t_s", 0.0)),
            az_rad=float(_get_field(obj, "az_rad", _get_field(obj, "az", 0.0))),
            el_rad=float(_get_field(obj, "el_rad", _get_field(obj, "el", 0.0))),
            az_rate_radps=float(
                _get_field(obj, "az_rate_radps", _get_field(obj, "az_rate", 0.0))
            ),
            el_rate_radps=float(
                _get_field(obj, "el_rate_radps", _get_field(obj, "el_rate", 0.0))
            ),
            s_rad=float(_get_field(obj, "s_rad", _get_field(obj, "s", 0.005))),
            s_dot_radps=float(
                _get_field(obj, "s_dot_radps", _get_field(obj, "s_dot", 0.0))
            ),
            r_min_m=float(r_min) if r_min is not None else None,
            r_max_m=float(r_max) if r_max is not None else None,
            hmd_hat_m=float(_get_field(obj, "hmd_hat_m"))
            if _get_field(obj, "hmd_hat_m") is not None
            else None,
            tau_mod_hat_s=float(_get_field(obj, "tau_mod_hat_s"))
            if _get_field(obj, "tau_mod_hat_s") is not None
            else None,
            tau_s_hat_s=float(_get_field(obj, "tau_s_hat_s"))
            if _get_field(obj, "tau_s_hat_s") is not None
            else None,
            tau_s=float(_get_field(obj, "tau_s"))
            if _get_field(obj, "tau_s") is not None
            else None,
            omega_los_radps=float(
                _get_field(obj, "omega_los_radps", _get_field(obj, "omega_los", 0.0))
            ),
            wc1_warning=bool(_get_field(obj, "wc1_warning", False)),
            wc1_corrective=bool(_get_field(obj, "wc1_corrective", False)),
            a1_warning=bool(
                _get_field(obj, "a1_warning", _get_field(obj, "alert_a1_warning", False))
            ),
            a1_corrective=bool(
                _get_field(
                    obj, "a1_corrective", _get_field(obj, "alert_a1_corrective", False)
                )
            ),
            a2_warning=bool(
                _get_field(obj, "a2_warning", _get_field(obj, "alert_a2_warning", False))
            ),
            a2_corrective=bool(
                _get_field(
                    obj, "a2_corrective", _get_field(obj, "alert_a2_corrective", False)
                )
            ),
            class_id=_get_field(obj, "class_id"),
            tslc_s=int(_get_field(obj, "tslc_s", 0)),
        )

    @classmethod
    def from_track_and_risk(
        cls,
        track: Any,
        risk: Any | None = None,
        class_id: str | None = "UNKNOWN",
    ) -> PassiveTrackInput:
        """Combine live WP-05 `Track` angular kinematics and `RiskEstimate` (`WC1`/`A1`/`A2`) (`B10-m2`).

        Extracts `(track_id, az, el, az_rate, el_rate, s_mom/s, s_dot, time_since_update)`
        from `track` and `(wc1_warning, wc1_corrective, range_interval_m, hmd_hat_m,
        tau_mod_hat_s, tau_s_hat_s, alert_a1_warning, alert_a2_warning)` from `risk`.
        Never reads any ground-truth attribute (`V5_B10`).
        """
        tid = str(_get_field(track, "track_id", _get_field(risk, "track_id", "T0001")))
        az_val = float(_get_field(track, "az", _get_field(track, "az_rad", 0.0)))
        el_val = float(_get_field(track, "el", _get_field(track, "el_rad", 0.0)))
        az_rate_val = float(
            _get_field(track, "az_rate", _get_field(track, "az_rate_radps", 0.0))
        )
        el_rate_val = float(
            _get_field(track, "el_rate", _get_field(track, "el_rate_radps", 0.0))
        )

        s_mom_val = _get_field(track, "s_mom")
        s_raw = _get_field(track, "s", _get_field(track, "s_rad", 0.005))
        s_val = (
            float(s_mom_val)
            if (s_mom_val is not None and float(s_mom_val) > 0.0)
            else float(s_raw)
        )
        s_dot_val = float(
            _get_field(track, "s_dot", _get_field(track, "s_dot_radps", 0.0))
        )
        tsu = float(_get_field(track, "time_since_update", _get_field(track, "tslc_s", 0.0)))

        hist_t = _get_field(track, "history_t")
        t_fallback = float(hist_t[-1]) if (isinstance(hist_t, Sequence) and len(hist_t) > 0) else 0.0
        t_val = float(_get_field(risk, "t_s", _get_field(track, "t_s", t_fallback))) if risk is not None else float(_get_field(track, "t_s", t_fallback))

        r_min: float | None = None
        r_max: float | None = None
        hmd_hat: float | None = None
        tau_mod_hat: float | None = None
        tau_s_hat: float | None = None
        tau_s_legacy: float | None = None
        omega_los: float = float(_get_field(track, "omega_los", math.hypot(az_rate_val, el_rate_val)))
        wc1_w = False
        wc1_c = False
        a1_w = False
        a1_c = False
        a2_w = False
        a2_c = False

        if risk is not None:
            r_int = _get_field(risk, "range_interval_m")
            if isinstance(r_int, (tuple, list)) and len(r_int) == 2:
                r0, r1 = float(r_int[0]), float(r_int[1])
                if math.isfinite(r0) and math.isfinite(r1) and r0 > 0.0 and r1 >= r0:
                    r_min, r_max = r0, r1
            if r_min is None or r_max is None:
                rm0 = _get_field(risk, "R_min_m", _get_field(risk, "r_min_m"))
                rm1 = _get_field(risk, "R_max_m", _get_field(risk, "r_max_m"))
                if rm0 is not None and rm1 is not None:
                    r_min, r_max = float(rm0), float(rm1)

            # If class_id is provided and known, refine range bounds if risk used blind union [0.30, 40.0]
            if (
                class_id
                and class_id in PASSIVE_CLASS_SIZE_BOUNDS_M
                and s_val > 1e-6
                and r_min is not None
                and r_max is not None
                and (r_max / max(1e-6, r_min)) > 50.0
            ):
                l_min, l_max = PASSIVE_CLASS_SIZE_BOUNDS_M[class_id]
                r_min = l_min / s_val
                r_max = l_max / s_val

            raw_hmd = _get_field(risk, "hmd_hat_m")
            if raw_hmd is not None and math.isfinite(float(raw_hmd)):
                hmd_hat = float(raw_hmd)

            raw_tm = _get_field(risk, "tau_mod_hat_s")
            if raw_tm is not None and math.isfinite(float(raw_tm)):
                tau_mod_hat = float(raw_tm)

            raw_ts_hat = _get_field(risk, "tau_s_hat_s")
            if raw_ts_hat is not None and math.isfinite(float(raw_ts_hat)):
                tau_s_hat = float(raw_ts_hat)

            raw_ts = _get_field(risk, "tau_s")
            if raw_ts is not None and math.isfinite(float(raw_ts)):
                tau_s_legacy = float(raw_ts)

            raw_om = _get_field(risk, "omega_los_radps")
            if raw_om is not None and math.isfinite(float(raw_om)):
                omega_los = float(raw_om)

            wc1_w = bool(_get_field(risk, "wc1_warning", False))
            wc1_c = bool(_get_field(risk, "wc1_corrective", False))
            a1_w = bool(
                _get_field(
                    risk, "alert_a1_warning", _get_field(risk, "a1_warning", False)
                )
            )
            a1_c = bool(
                _get_field(
                    risk, "alert_a1_corrective", _get_field(risk, "a1_corrective", False)
                )
            )
            a2_w = bool(
                _get_field(
                    risk, "alert_a2_warning", _get_field(risk, "a2_warning", False)
                )
            )
            a2_c = bool(
                _get_field(
                    risk, "alert_a2_corrective", _get_field(risk, "a2_corrective", False)
                )
            )

        resolved_cls = (
            str(class_id)
            if class_id is not None
            else str(_get_field(track, "class_id", "UNKNOWN"))
        )

        return cls(
            track_id=tid,
            t_s=t_val,
            az_rad=az_val,
            el_rad=el_val,
            az_rate_radps=az_rate_val,
            el_rate_radps=el_rate_val,
            s_rad=s_val,
            s_dot_radps=s_dot_val,
            r_min_m=r_min,
            r_max_m=r_max,
            hmd_hat_m=hmd_hat,
            tau_mod_hat_s=tau_mod_hat,
            tau_s_hat_s=tau_s_hat,
            tau_s=tau_s_legacy,
            omega_los_radps=omega_los,
            wc1_warning=wc1_w,
            wc1_corrective=wc1_c,
            a1_warning=a1_w,
            a1_corrective=a1_c,
            a2_warning=a2_w,
            a2_corrective=a2_c,
            class_id=resolved_cls,
            tslc_s=max(0, int(round(tsu))),
        )


@dataclass
class Tier0StepOutput:
    """Frames and parsed objects emitted by `PassiveTrackAvionicsBridge` at a single epoch."""

    t_s: float
    heartbeat_emitted: bool
    integrity_state: str = "VALID"
    gdl90_frames: list[bytes] = field(default_factory=list)
    mavlink_frames: list[bytes] = field(default_factory=list)
    gdl90_traffic_reports: list[Gdl90TargetReport] = field(default_factory=list)
    mavlink_messages: list[MavlinkAdsbVehicle] = field(default_factory=list)
    continuous_geodetic: list[tuple[float, float, float]] = field(default_factory=list)


class PassiveTrackAvionicsBridge:
    """Synchronized 1 Hz Heartbeat + 10 Hz Ownship/Traffic GDL-90 & MAVLink v2 bridge."""

    def __init__(
        self,
        spec_path: Path | str = DEFAULT_TIER0_SPEC_PATH,
        gdl90_endpoint: Gdl90UdpEndpoint | None = None,
        mavlink_streamer: MavlinkAdsbUdpStreamer | None = None,
    ) -> None:
        with open(spec_path, "r", encoding="utf-8") as f:
            self.spec: dict[str, Any] = json.load(f)

        self.nic_nacp_tiers: list[dict[str, Any]] = list(
            self.spec.get("nic_nacp_range_uncertainty_tiers", [])
        )
        self.emitter_map: dict[str, int] = dict(DEFAULT_EMITTER_CATEGORY_MAP)
        self.emitter_map.update(self.spec.get("emitter_category_map", {}))

        mav_cfg = self.spec.get("mavlink_adsb_protocol", {})
        self.sysid: int = int(mav_cfg.get("default_sysid", 1))  # default-ok: optional avionics config/state fallback
        self.compid: int = int(mav_cfg.get("default_compid", 156))  # default-ok: optional avionics config/state fallback
        self.squawk_nominal: int = int(mav_cfg.get("squawk_nominal", 1200))  # default-ok: optional avionics config/state fallback
        self.squawk_alert: int = int(mav_cfg.get("squawk_alert", 7700))  # default-ok: optional avionics config/state fallback

        int_cfg = self.spec.get("integrity_gating", {})
        self.degraded_max_nic: int = int(int_cfg.get("degraded_coast_max_nic", 4))  # default-ok: optional avionics config/state fallback
        self.degraded_max_nacp: int = int(int_cfg.get("degraded_coast_max_nacp", 4))  # default-ok: optional avionics config/state fallback
        self.invalid_nic: int = int(int_cfg.get("invalid_alert_nic", 0))  # default-ok: optional avionics config/state fallback
        self.invalid_nacp: int = int(int_cfg.get("invalid_alert_nacp", 0))  # default-ok: optional avionics config/state fallback

        gdl_cfg = self.spec.get("gdl90_protocol", {})
        self.heartbeat_period_s: float = 1.0 / float(gdl_cfg.get("heartbeat_rate_hz", 1.0))  # default-ok: optional avionics config/state fallback

        self.gdl90_endpoint = gdl90_endpoint
        self.mavlink_streamer = mavlink_streamer
        self._last_heartbeat_t_s: float | None = None
        self._mavlink_seq: int = 0

    def _next_mavlink_seq(self) -> int:
        if self.mavlink_streamer is not None:
            return self.mavlink_streamer.next_seq()
        s = self._mavlink_seq
        self._mavlink_seq = (self._mavlink_seq + 1) & 0xFF
        return s

    def _apply_integrity_to_nic_nacp(
        self, nic: int, nacp: int, integrity_state: str
    ) -> tuple[int, int]:
        state = str(integrity_state).upper()
        if state not in VALID_INTEGRITY_STATES:
            raise ValueError(f"Unsupported integrity_state: {integrity_state!r}")
        if state == "INVALID_ALERT":
            return (self.invalid_nic, self.invalid_nacp)
        if state == "DEGRADED_COAST":
            return (min(int(nic), self.degraded_max_nic), min(int(nacp), self.degraded_max_nacp))
        return (int(nic), int(nacp))

    def build_ownship_report(
        self,
        ownship: OwnshipNavState,
        integrity_state: str = "VALID",
    ) -> Gdl90TargetReport:
        """Build 28-byte GDL-90 Ownship Report (0x0A) with WP-07 integrity gating."""
        horiz_mps = math.hypot(ownship.vn_mps, ownship.ve_mps)
        horiz_kt = horiz_mps * MPS_TO_KNOTS
        vert_fpm = -ownship.vd_mps * MPS_TO_FPM
        if horiz_mps > 0.1:
            trk_deg = math.degrees(math.atan2(ownship.ve_mps, ownship.vn_mps)) % 360.0
        else:
            trk_deg = ownship.heading_deg % 360.0

        nic, nacp = self._apply_integrity_to_nic_nacp(
            ownship.nic, ownship.nacp, integrity_state
        )

        return Gdl90TargetReport(
            msg_id=GDL90_MSG_ID_OWNSHIP_REPORT,
            alert_status=0,
            address_type=0,
            participant_address=ownship.icao_address & 0xFFFFFF,
            lat_deg=ownship.lat_deg,
            lon_deg=ownship.lon_deg,
            altitude_ft=ownship.alt_msl_m * METERS_TO_FEET,
            misc_indicators=0x9,
            nic=nic,
            nacp=nacp,
            horiz_vel_kt=horiz_kt,
            vert_vel_fpm=vert_fpm,
            track_heading_deg=trk_deg,
            emitter_category=ownship.emitter_category,
            callsign=ownship.callsign[:8],
            emergency_code=0,
        )

    def build_traffic_pair(
        self,
        ownship: OwnshipNavState,
        track_in: PassiveTrackInput | Mapping[str, Any] | Any,
        integrity_state: str = "VALID",
    ) -> tuple[Gdl90TargetReport, MavlinkAdsbVehicle]:
        """Convert one passive WP-05 track state into GDL-90 Traffic (0x14) and MAVLink (#246)."""
        gdl_traffic, mav_msg, _ = self.build_traffic_pair_with_continuous(
            ownship=ownship,
            track_in=track_in,
            integrity_state=integrity_state,
        )
        return gdl_traffic, mav_msg

    def build_traffic_pair_with_continuous(
        self,
        ownship: OwnshipNavState,
        track_in: PassiveTrackInput | Mapping[str, Any] | Any,
        integrity_state: str = "VALID",
    ) -> tuple[Gdl90TargetReport, MavlinkAdsbVehicle, tuple[float, float, float]]:
        """Convert one passive WP-05 track state into GDL-90 Traffic (0x14), MAVLink (#246),
        and unquantized continuous geodetic estimate `(cont_lat_deg, cont_lon_deg, cont_alt_m)`.
        """
        state = str(integrity_state).upper()
        if state not in VALID_INTEGRITY_STATES:
            raise ValueError(f"Unsupported integrity_state: {integrity_state!r}")

        trk = PassiveTrackInput.from_track_state(track_in)
        r_min_m, r_max_m = trk.resolve_passive_range_interval_m()
        r_span_m = max(0.0, r_max_m - r_min_m)
        raw_nic, raw_nacp = map_range_span_to_nic_nacp(r_span_m, self.nic_nacp_tiers)
        nic, nacp = self._apply_integrity_to_nic_nacp(raw_nic, raw_nacp, state)

        emitter_cat = self.emitter_map.get(
            trk.class_id or "UNKNOWN", self.emitter_map.get("UNKNOWN", 14)  # default-ok: optional avionics config/state fallback
        )
        alert_flag = 1 if trk.alert_active else 0
        valid_coords = state != "INVALID_ALERT"

        cont_geodetic = compute_passive_track_continuous_geodetic(
            ownship_lat_deg=ownship.lat_deg,
            ownship_lon_deg=ownship.lon_deg,
            ownship_alt_msl_m=ownship.alt_msl_m,
            az_rad=trk.az_rad,
            el_rad=trk.el_rad,
            r_min_m=r_min_m,
            r_max_m=r_max_m,
        )

        mav_msg = passive_track_to_adsb_vehicle(
            track_number=trk.track_number,
            ownship_lat_deg=ownship.lat_deg,
            ownship_lon_deg=ownship.lon_deg,
            ownship_alt_msl_m=ownship.alt_msl_m,
            ownship_vn_mps=ownship.vn_mps,
            ownship_ve_mps=ownship.ve_mps,
            ownship_vd_mps=ownship.vd_mps,
            az_rad=trk.az_rad,
            el_rad=trk.el_rad,
            az_rate_radps=trk.az_rate_radps,
            el_rate_radps=trk.el_rate_radps,
            r_min_m=r_min_m,
            r_max_m=r_max_m,
            tau_hat_s=trk.effective_tau_s,
            alert_active=trk.alert_active,
            emitter_type=emitter_cat,
            tslc_s=trk.tslc_s,
            seq=self._next_mavlink_seq(),
            sysid=self.sysid,
            compid=self.compid,
            squawk_nominal=self.squawk_nominal,
            squawk_alert=self.squawk_alert,
            valid_coords=valid_coords,
        )

        cont_lat_deg, cont_lon_deg, cont_alt_m = cont_geodetic
        gdl_traffic = Gdl90TargetReport(
            msg_id=GDL90_MSG_ID_TRAFFIC_REPORT,
            alert_status=alert_flag,
            address_type=1,  # Self-assigned / non-ICAO passive track
            participant_address=trk.participant_address,
            lat_deg=cont_lat_deg,
            lon_deg=cont_lon_deg,
            altitude_ft=cont_alt_m * METERS_TO_FEET,
            misc_indicators=0x9,  # Airborne + True Track
            nic=nic,
            nacp=nacp,
            horiz_vel_kt=mav_msg.hor_velocity_mps * MPS_TO_KNOTS,
            vert_vel_fpm=mav_msg.ver_velocity_mps * MPS_TO_FPM,
            track_heading_deg=mav_msg.heading_deg,
            emitter_category=emitter_cat,
            callsign=trk.callsign,
            emergency_code=1 if trk.alert_active else 0,
        )
        return gdl_traffic, mav_msg, cont_geodetic

    def step(
        self,
        t_s: float,
        ownship: OwnshipNavState,
        tracks: Sequence[PassiveTrackInput | Mapping[str, Any] | Any],
        force_heartbeat: bool = False,
        integrity_state: str = "VALID",
    ) -> Tier0StepOutput:
        """Process one epoch (typically 10 Hz), applying WP-07 integrity gating and optionally sending over UDP."""
        state = str(integrity_state).upper()
        if state not in VALID_INTEGRITY_STATES:
            raise ValueError(f"Unsupported integrity_state: {integrity_state!r}")

        emit_hb = force_heartbeat
        if not emit_hb:
            if self._last_heartbeat_t_s is None or (
                float(t_s) - self._last_heartbeat_t_s
            ) >= (self.heartbeat_period_s - 1e-6):
                emit_hb = True

        out = Tier0StepOutput(
            t_s=float(t_s),
            heartbeat_emitted=emit_hb,
            integrity_state=state,
        )

        maint_reqd = state in ("DEGRADED_COAST", "INVALID_ALERT")
        gps_pos_valid = state != "INVALID_ALERT"
        vertical_warning = state == "INVALID_ALERT"

        if emit_hb:
            self._last_heartbeat_t_s = float(t_s)
            hb_frame = encode_heartbeat(
                utc_time_s=int(round(t_s)) % 86400,
                gps_pos_valid=gps_pos_valid,
                maint_reqd=maint_reqd,
                utc_ok=True,
                basic_and_long_count=len(tracks),
            )
            out.gdl90_frames.append(hb_frame)

        own_rep = self.build_ownship_report(ownship, integrity_state=state)
        out.gdl90_frames.append(encode_target_report(own_rep))
        out.gdl90_frames.append(
            encode_ownship_geo_altitude(
                alt_ft=ownship.alt_geo_m * METERS_TO_FEET,
                vfom_m=ownship.vfom_m,
                vertical_warning=vertical_warning,
            )
        )

        for trk_in in tracks:
            gdl_trf, mav_msg, cont_geo = self.build_traffic_pair_with_continuous(
                ownship, trk_in, integrity_state=state
            )
            gdl_frame = encode_target_report(gdl_trf)
            mav_frame = pack_adsb_vehicle(mav_msg)
            out.gdl90_traffic_reports.append(gdl_trf)
            out.mavlink_messages.append(mav_msg)
            out.continuous_geodetic.append(cont_geo)
            out.gdl90_frames.append(gdl_frame)
            out.mavlink_frames.append(mav_frame)

        if self.gdl90_endpoint is not None:
            for gf in out.gdl90_frames:
                self.gdl90_endpoint.send_frame(gf)

        if self.mavlink_streamer is not None:
            for mf in out.mavlink_frames:
                self.mavlink_streamer.send_packet(mf)

        return out

    def step_from_tracker(
        self,
        ownship: OwnshipNavState,
        confirmed_tracks: Sequence[Any],
        risk_by_track_id: Mapping[str, Any],
        integrity_state: str = "VALID",
        *,
        t_s: float | None = None,
        class_by_track_id: Mapping[str, str] | None = None,
        force_heartbeat: bool = False,
    ) -> Tier0StepOutput:
        """Couple live WP-05 G1 `Track` objects + `RiskEstimate` dict + WP-07 `integrity_state` (`B10-m2`).

        Automatically filters out any track that is in `CLUTTER_HOLD` or flagged
        `ground_clutter_rejected = True` so stationary ground features never emit traffic datagrams.
        """
        passive_inputs: list[PassiveTrackInput] = []
        inferred_t_s: float | None = t_s

        for trk in confirmed_tracks:
            # Skip any non-confirmed or ground-clutter-held track
            if bool(_get_field(trk, "ground_clutter_rejected", False)):
                continue
            status_obj = _get_field(trk, "status")
            status_val = getattr(status_obj, "value", status_obj)
            if status_val is not None and str(status_val).lower() not in ("confirmed", ""):
                continue

            tid = str(_get_field(trk, "track_id", ""))
            risk_obj = risk_by_track_id.get(tid) if isinstance(risk_by_track_id, Mapping) else None
            cls_id = (
                class_by_track_id.get(tid, "UNKNOWN")
                if isinstance(class_by_track_id, Mapping)
                else str(_get_field(trk, "class_id", "UNKNOWN"))
            )
            pt_in = PassiveTrackInput.from_track_and_risk(trk, risk_obj, class_id=cls_id)
            if inferred_t_s is None:
                inferred_t_s = pt_in.t_s
            passive_inputs.append(pt_in)

        if inferred_t_s is None:
            if isinstance(risk_by_track_id, Mapping) and risk_by_track_id:
                first_r = next(iter(risk_by_track_id.values()))
                inferred_t_s = float(_get_field(first_r, "t_s", 0.0))
            else:
                inferred_t_s = 0.0

        return self.step(
            t_s=float(inferred_t_s),
            ownship=ownship,
            tracks=passive_inputs,
            force_heartbeat=force_heartbeat,
            integrity_state=integrity_state,
        )
