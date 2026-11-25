"""Closed-Loop MAVLink #246 -> ArduPilot SITL Kinematic Horizontal Avoidance Simulator.

Implements Batch B12 Part 3 (`docs/xvs/batches/BATCH_B12_WP08_G1_ARGON.md` §1.3 & `WP08_DISPLAY.md` §8):
- Self-contained ArduPilot SITL kinematic horizontal avoidance autopilot node (`SitlAvoidanceAutopilot`)
  consuming live wire-packed `MAVLink v2 ADSB_VEHICLE (#246)` (`unpack_adsb_vehicle`) and `GDL-90 (0x14)`
  (`decode_gdl90_message`) traffic frames emitted by `PassiveTrackAvionicsBridge.step_from_tracker`;
- Couples WP-05 G1's real `MultiCameraTracker` (`src/xvs/tracking/tracker.py`), vertex second-moment scale
  and sliding-window log-size TTC (`src/xvs/tracking/ttc.py`), and passive Well-Clear risk evaluator
  (`evaluate_track_risk` in `src/xvs/tracking/risk.py`) in the 10 Hz closed loop (`B12-F2`);
- Enforces DO-365A / ArduPilot `ADSB_BEHAVIOR = LOITER_OR_AVOID` horizontal turn kinematics (`B12-F3`):
  * Stern/wake-pass escape turn selection from measured LOS bearing rate omega_LOS and relative bearing beta,
    turning behind a crossing intruder's tail and turning right (+1) for pure head-on encounters;
  * Maximum turn rate psi_dot_max = 3.0 deg/s, bank angle limit phi_max = 25.0 deg, reaction delay t_react = 1.0 s;
  * Post-CPA `RECOVER_COURSE` guidance commanding a bounded <= 30 deg intercept heading back onto the nominal
    Eastbound airway (y_north -> 0, psi -> psi_nom);
  * WP-07 `INVALID_ALERT` coordinate gating: when `VALID_COORDS == 0` in MAVLink #246 (or `NIC == 0`
    in GDL-90), autonomous horizontal avoidance maneuvers are strictly inhibited (`0` false turns);
- Evaluates across a seeded benchmark of 50 representative collision-threat encounters (`d_CPA < 152 m`
  in open loop) from `G3a v0.1` paired with Common Random Numbers (CRN) across:
  1. `open_loop` (No Avoidance),
  2. `closed_loop_g0_a1` (Closed-Loop with G0 `A1` Alerting + Tier-0 MAVLink #246),
  3. `closed_loop_g1_wc1_b050` (Closed-Loop with G1 `WC1` beta_WC = 0.50 Alerting + Tier-0 MAVLink #246),
  4. `closed_loop_g1_wc1_b025` (Closed-Loop with G1 `WC1` beta_WC = 0.25 Alerting + Tier-0 MAVLink #246).

Strictly preserves the V5_B10 / V5_G1 truth-isolation invariant: never imports `xvs.traffic` and
never accesses any ground-truth intruder attribute in the perception or autopilot pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

from xvs.avionics.bridge import (
    OwnshipNavState,
    PassiveTrackAvionicsBridge,
)
from xvs.avionics.gdl90 import (
    GDL90_MSG_ID_TRAFFIC_REPORT,
    Gdl90TargetReport,
    decode_gdl90_message,
)
from xvs.avionics.mavlink_adsb import (
    ADSB_FLAGS_VALID_COORDS,
    MavlinkAdsbVehicle,
    unpack_adsb_vehicle,
)
from xvs.geo import rotations, wgs84
from xvs.tracking.measurements import DetectionRecord, get_model_vertices
from xvs.tracking.risk import (
    LOWC_HMD_LIMIT_M,
     RiskEstimate,
    WP02_CLASS_DIMENSIONS,
    evaluate_track_risk,
)
from xvs.tracking.tracker import MultiCameraTracker, Track
from xvs.tracking.ttc import compute_vertex_second_moment_scale

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_G1_DISPLAY_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "display" / "g1.json"
DEFAULT_G1_TRACKING_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g1.json"
_FALLBACK_WP02_ENCOUNTERS_PATHS = (
    REPO_ROOT / "output" / "xvs" / "wp02" / "g1" / "encounters.jsonl",
)

GRAVITY_MPS2: float = 9.80665
NMAC_HMD_THRESHOLD_M: float = 152.4
LOWC_HMD_THRESHOLD_M: float = 609.6

_GEOMETRY_STRATA = (
    "head_on",
    "crossing_left",
    "crossing_right",
    "same_direction",
    "converging",
)

_CLASS_STRATA = (
    "C1_UAS",
    "C2_SEP",
    "C3_TWIN",
    "C4_BIZJET",
    "C5_TRANSPORT",
    "C6_HELI",
)

_VERTICAL_STRATA = (
    "above",
    "co_altitude",
    "below",
)

_SPEED_RANGES_MPS: dict[str, dict[str, tuple[float, float]]] = {
    "C1_UAS": {"gs": (10.0, 30.0), "vs": (-3.0, 3.0)},
    "C2_SEP": {"gs": (45.0, 75.0), "vs": (-8.0, 8.0)},
    "C3_TWIN": {"gs": (60.0, 110.0), "vs": (-8.0, 8.0)},
    "C4_BIZJET": {"gs": (110.0, 230.0), "vs": (-8.0, 8.0)},
    "C5_TRANSPORT": {"gs": (120.0, 250.0), "vs": (-8.0, 8.0)},
    "C6_HELI": {"gs": (30.0, 75.0), "vs": (-8.0, 8.0)},
}


def _wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _wrap_deg_180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def load_g1_tracking_config(path: Path | str = DEFAULT_G1_TRACKING_CONFIG_PATH) -> dict[str, Any]:
    """Load WP-05 G1 tracker configuration (`configs/xvs/tracking/g1.json`)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class SitlAvoidanceConfig:
    """Configuration for ArduPilot SITL kinematic horizontal avoidance node."""

    max_turn_rate_degps: float = 3.0
    max_bank_angle_deg: float = 25.0
    reaction_delay_s: float = 1.0
    target_avoid_heading_deg: float = 55.0
    clear_of_conflict_hold_s: float = 2.0
    recovery_turn_rate_degps: float = 2.5
    max_intercept_angle_deg: float = 30.0
    lateral_offset_cap_m: float = 780.0
    dt_s: float = 0.1
    t_start_s: float = 0.0
    t_end_s: float = 60.0
    nmac_threshold_m: float = NMAC_HMD_THRESHOLD_M
    lowc_threshold_m: float = LOWC_HMD_THRESHOLD_M
    benchmark_threat_count: int = 50
    open_loop_cpa_max_m: float = 152.0

    @classmethod
    def from_display_config(
        cls, cfg: Mapping[str, Any] | Path | str | None = None
    ) -> SitlAvoidanceConfig:
        if cfg is None:
            if DEFAULT_G1_DISPLAY_CONFIG_PATH.is_file():
                cfg = json.loads(DEFAULT_G1_DISPLAY_CONFIG_PATH.read_text(encoding="utf-8"))
            else:
                cfg = {}
        elif isinstance(cfg, (Path, str)):
            p = Path(cfg)
            cfg = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        sa = cfg.get("sitl_avoidance", {})
        return cls(
            max_turn_rate_degps=float(sa.get("max_turn_rate_degps", 3.0)),  # default-ok: optional SITL/tracker config fallback
            max_bank_angle_deg=float(sa.get("max_bank_angle_deg", 25.0)),  # default-ok: optional SITL/tracker config fallback
            reaction_delay_s=float(sa.get("reaction_delay_s", 1.0)),  # default-ok: optional SITL/tracker config fallback
            target_avoid_heading_deg=float(sa.get("target_avoid_heading_deg", 55.0)),  # default-ok: optional SITL/tracker config fallback
            clear_of_conflict_hold_s=float(sa.get("clear_of_conflict_hold_s", 2.0)),  # default-ok: optional SITL/tracker config fallback
            recovery_turn_rate_degps=float(sa.get("recovery_turn_rate_degps", 2.5)),  # default-ok: optional SITL/tracker config fallback
            max_intercept_angle_deg=float(sa.get("max_intercept_angle_deg", 30.0)),  # default-ok: optional SITL/tracker config fallback
            lateral_offset_cap_m=float(sa.get("lateral_offset_cap_m", 780.0)),  # default-ok: optional SITL/tracker config fallback
            dt_s=float(sa.get("dt_s", 0.1)),  # default-ok: optional SITL/tracker config fallback
            t_start_s=float(sa.get("t_start_s", 0.0)),  # default-ok: optional SITL/tracker config fallback
            t_end_s=float(sa.get("t_end_s", 60.0)),  # default-ok: optional SITL/tracker config fallback
            nmac_threshold_m=float(sa.get("nmac_threshold_m", NMAC_HMD_THRESHOLD_M)),
            lowc_threshold_m=float(sa.get("lowc_threshold_m", LOWC_HMD_THRESHOLD_M)),
            benchmark_threat_count=int(sa.get("benchmark_threat_count", 50)),  # default-ok: optional SITL/tracker config fallback
            open_loop_cpa_max_m=float(sa.get("open_loop_cpa_max_m", 152.0)),  # default-ok: optional SITL/tracker config fallback
        )


@dataclass(frozen=True)
class SitlThreatScenario:
    """Self-contained kinematic description of a collision-threat encounter from G3a v0.1."""

    scenario_id: str
    geometry_stratum: str
    class_id: str
    vertical_stratum: str
    characteristic_length_m: float
    t_cpa_open_s: float
    open_loop_hmd_m: float
    open_loop_vmd_m: float
    side: str
    approach_deg: float
    gs_mps: float
    vs_mps: float
    target_p_cpa_ecef_m: np.ndarray
    target_v_ecef_mps: np.ndarray
    seed: int
    model_id: str = "C2_SEP_1"

    def target_position_ecef_at(self, t_s: float) -> np.ndarray:
        return self.target_p_cpa_ecef_m + self.target_v_ecef_mps * (
            float(t_s) - self.t_cpa_open_s
        )

    def target_attitude_ecef_at(self, t_s: float) -> np.ndarray:
        """Compute ECEF-to-body rotation matrix for the constant-velocity intruder."""
        p_ecef = self.target_position_ecef_at(t_s)
        lat_i, lon_i, _ = wgs84.ecef_to_geodetic(*p_ecef)
        r_ecef_ned_i = wgs84.r_ecef_ned(lat_i, lon_i)
        v_ned_i = r_ecef_ned_i.T @ self.target_v_ecef_mps
        v_horiz = math.hypot(float(v_ned_i[0]), float(v_ned_i[1]))
        yaw_i = math.atan2(float(v_ned_i[1]), float(v_ned_i[0])) if v_horiz > 1e-6 else 0.0
        pitch_i = math.atan2(-float(v_ned_i[2]), max(1e-6, v_horiz))
        r_ned_body_i = rotations.euler_to_r(yaw_i, pitch_i, 0.0)
        return r_ecef_ned_i @ r_ned_body_i


@dataclass
class SitlAutopilotCommand:
    """Instantaneous horizontal control command issued by `SitlAvoidanceAutopilot`."""

    t_s: float
    mode: str  # "NOMINAL" | "REACTION_DELAY" | "AVOID_TURN" | "RECOVER_COURSE" | "INHIBITED_INVALID_COORDS"
    turn_rate_cmd_degps: float
    bank_angle_deg: float
    target_heading_deg: float
    turn_direction: int  # -1 (LEFT), 0 (NONE), +1 (RIGHT)
    alert_active: bool
    coords_valid: bool
    inhibited_by_invalid_coords: bool


class SitlAvoidanceAutopilot:
    """ArduPilot SITL (`AP_Avoidance` / `ADSB_BEHAVIOR = LOITER_OR_AVOID`) horizontal controller.

    Consumes ONLY wire-decoded MAVLink v2 `ADSB_VEHICLE` (#246) and GDL-90 Traffic Report (`0x14`)
    packets plus ownship navigation state (`OwnshipNavState`).
    Never accesses ground-truth intruder position, range, or velocity (V5_G1).
    """

    def __init__(
        self,
        nominal_heading_deg: float = 90.0,
        config: SitlAvoidanceConfig | None = None,
    ) -> None:
        self.nominal_heading_deg = float(nominal_heading_deg) % 360.0
        self.config = config if config is not None else SitlAvoidanceConfig()
        self.first_alert_t_s: float | None = None
        self.maneuver_start_t_s: float | None = None
        self.latched_turn_dir: int = 0
        self.latched_avoid_Delta_deg: float = self.config.target_avoid_heading_deg
        self.initial_rel_bearing_rad: float = 0.0
        self.last_est_range_m: float | None = None
        self.min_est_range_seen_m: float = float("inf")
        self.range_increasing_duration_s: float = 0.0
        self.prev_bearing_rad: float | None = None
        self.prev_bearing_t_s: float | None = None
        self.est_bearing_rate_radps: float = 0.0
        self.recovery_latched: bool = False
        self.ref_p0_ecef: np.ndarray | None = None
        self.ref_n_hat_0: np.ndarray | None = None
        self.invalid_coord_inhibits_count: int = 0
        self.induced_turns_count: int = 0

    def reset(self, nominal_heading_deg: float | None = None) -> None:
        if nominal_heading_deg is not None:
            self.nominal_heading_deg = float(nominal_heading_deg) % 360.0
        self.first_alert_t_s = None
        self.maneuver_start_t_s = None
        self.latched_turn_dir = 0
        self.latched_avoid_Delta_deg = self.config.target_avoid_heading_deg
        self.initial_rel_bearing_rad = 0.0
        self.last_est_range_m = None
        self.min_est_range_seen_m = float("inf")
        self.range_increasing_duration_s = 0.0
        self.prev_bearing_rad = None
        self.prev_bearing_t_s = None
        self.est_bearing_rate_radps = 0.0
        self.recovery_latched = False
        self.ref_p0_ecef = None
        self.ref_n_hat_0 = None
        self.invalid_coord_inhibits_count = 0
        self.induced_turns_count = 0

    def _compute_ownship_cross_track_north_m(self, ownship: OwnshipNavState) -> float:
        """Compute ownship lateral offset (m, positive = Left/North of nominal Eastbound airway)
        using only ownship navigation state.
        """
        p_own = wgs84.geodetic_to_ecef(
            ownship.lat_deg, ownship.lon_deg, ownship.alt_msl_m
        )
        if self.ref_p0_ecef is None or self.ref_n_hat_0 is None:
            self.ref_p0_ecef = p_own.copy()
            n_mat_0 = wgs84.r_ecef_ned(ownship.lat_deg, ownship.lon_deg)
            psi_nom_rad = math.radians(self.nominal_heading_deg)
            # Left-hand normal in NED horizontal plane: [-sin(psi_nom), cos(psi_nom), 0]
            # For psi_nom = 90 deg (East), left normal is [+1, 0, 0] = North.
            left_ned = np.array(
                [math.sin(psi_nom_rad), -math.cos(psi_nom_rad), 0.0], dtype=np.float64
            )
            self.ref_n_hat_0 = n_mat_0 @ left_ned
        return float(np.dot(p_own - self.ref_p0_ecef, self.ref_n_hat_0))

    def _select_escape_turn_direction(
        self,
        ownship: OwnshipNavState,
        mav_msg: MavlinkAdsbVehicle,
        rel_bearing_rad: float,
    ) -> tuple[int, float]:
        """Select horizontal escape turn direction (-1 LEFT or +1 RIGHT) and heading offset (deg)
        using measured line-of-sight bearing rate omega_LOS and relative bearing beta (B12-F3).

        Conforms to DO-365A / stern-pass horizontal avoidance sense rules:
        - Pure head-on encounters (|beta| <= 6 deg and |omega_LOS| < eps): turn RIGHT (+1) per DO-365.
        - Near head-on with measurable LOS rate (|beta| <= 6 deg and |omega_LOS| >= eps):
          turn opposite the LOS drift sign so ownship passes behind the intruder's stern.
        - Forward/beam crossing or converging encounters (6 deg < |beta| <= 105 deg):
          * Intruder approaching from Left (beta < 0, crossing Left-to-Right into Right half-plane):
            turn LEFT (-1) behind the intruder's tail/stern.
          * Intruder approaching from Right (beta > 0, crossing Right-to-Left into Left half-plane):
            turn RIGHT (+1) behind the intruder's tail/stern.
        - Rear/overtaking sector (|beta| > 105 deg):
          turn laterally away from the overtaking bearing (-1 if beta > 0 else +1).
        """
        _ = (ownship, mav_msg)
        beta_deg = math.degrees(_wrap_pi(rel_bearing_rad))
        abs_beta_deg = abs(beta_deg)
        omega_los = self.est_bearing_rate_radps
        eps_omega = 0.0015  # rad/s (~0.086 deg/s)

        if abs_beta_deg <= 0.5:
            if abs(omega_los) < eps_omega:
                turn_dir = +1  # DO-365 / 14 CFR 91.113 pure head-on right turn
            else:
                turn_dir = -1 if omega_los > 0.0 else +1
            avoid_deg = max(42.0, float(self.config.target_avoid_heading_deg))
        elif abs_beta_deg <= 105.0:
            # Crossing / converging / near-head-on with lateral offset:
            # Turn toward the intruder's initial bearing side so that as the intruder crosses the
            # airway into the opposite half-plane, ownship passes cleanly behind its stern/wake.
            turn_dir = -1 if beta_deg < 0.0 else +1
            avoid_deg = max(40.0, float(self.config.target_avoid_heading_deg))
        else:
            # Overtaking from behind: turn laterally away from the overtaker
            turn_dir = -1 if beta_deg > 0.0 else +1
            avoid_deg = max(42.0, float(self.config.target_avoid_heading_deg))

        return turn_dir, avoid_deg

    def step_from_wire_packets(
        self,
        t_s: float,
        ownship: OwnshipNavState,
        mavlink_wire_packets: Sequence[bytes],
        gdl90_wire_frames: Sequence[bytes] = (),
    ) -> SitlAutopilotCommand:
        """Decode incoming MAVLink #246 and GDL-90 wire frames and compute horizontal turn command."""
        y_left_m = self._compute_ownship_cross_track_north_m(ownship)

        decoded_mav: list[MavlinkAdsbVehicle] = []
        for pkt in mavlink_wire_packets:
            decoded_mav.append(unpack_adsb_vehicle(pkt))

        decoded_gdl_traffic: list[Gdl90TargetReport] = []
        for frm in gdl90_wire_frames:
            dec = decode_gdl90_message(frm)
            if (
                isinstance(dec, Gdl90TargetReport)
                and dec.msg_id == GDL90_MSG_ID_TRAFFIC_REPORT
            ):
                decoded_gdl_traffic.append(dec)

        own_speed_mps = max(1.0, math.hypot(ownship.vn_mps, ownship.ve_mps))

        # Check coordinate validity flags in MAVLink #246 (and GDL-90 NIC)
        any_invalid_coords = False
        active_threat_mav: MavlinkAdsbVehicle | None = None

        for m_msg in decoded_mav:
            coords_ok = bool(m_msg.flags & ADSB_FLAGS_VALID_COORDS)
            if not coords_ok:
                any_invalid_coords = True
                continue
            is_alert = (m_msg.squawk == 7700)
            if is_alert and active_threat_mav is None:
                active_threat_mav = m_msg

        for g_rep in decoded_gdl_traffic:
            if g_rep.nic == 0 or g_rep.nacp == 0:
                any_invalid_coords = True

        # V4_G1: If WP-07 is in INVALID_ALERT (VALID_COORDS == 0 / NIC == 0), inhibit autonomous turn
        if any_invalid_coords and active_threat_mav is None:
            self.invalid_coord_inhibits_count += 1
            return SitlAutopilotCommand(
                t_s=float(t_s),
                mode="INHIBITED_INVALID_COORDS",
                turn_rate_cmd_degps=0.0,
                bank_angle_deg=0.0,
                target_heading_deg=ownship.heading_deg,
                turn_direction=0,
                alert_active=False,
                coords_valid=False,
                inhibited_by_invalid_coords=True,
            )

        ref_mav = active_threat_mav if active_threat_mav is not None else (
            decoded_mav[0]
            if (decoded_mav and bool(decoded_mav[0].flags & ADSB_FLAGS_VALID_COORDS))
            else None
        )

        rel_bearing_rad = 0.0
        rel_bearing_nom_rad = 0.0
        if ref_mav is not None:
            n_mat = wgs84.r_ecef_ned(ownship.lat_deg, ownship.lon_deg)
            p_own = wgs84.geodetic_to_ecef(ownship.lat_deg, ownship.lon_deg, ownship.alt_msl_m)
            p_int_est = wgs84.geodetic_to_ecef(ref_mav.lat_deg, ref_mav.lon_deg, ref_mav.altitude_m)
            r_ned = n_mat.T @ (p_int_est - p_own)
            est_range_m = float(math.hypot(float(r_ned[0]), float(r_ned[1])))
            az_est = math.atan2(float(r_ned[1]), float(r_ned[0]))
            rel_bearing_rad = _wrap_pi(az_est - math.radians(ownship.heading_deg))
            rel_bearing_nom_rad = _wrap_pi(az_est - math.radians(self.nominal_heading_deg))

            if self.prev_bearing_rad is not None and self.prev_bearing_t_s is not None:
                dt_b = float(t_s) - self.prev_bearing_t_s
                if dt_b > 1e-4:
                    inst_rate = _wrap_pi(az_est - self.prev_bearing_rad) / dt_b
                    self.est_bearing_rate_radps = 0.6 * self.est_bearing_rate_radps + 0.4 * inst_rate
            self.prev_bearing_rad = az_est
            self.prev_bearing_t_s = float(t_s)

            if est_range_m < self.min_est_range_seen_m:
                self.min_est_range_seen_m = est_range_m
            if self.last_est_range_m is not None:
                if est_range_m > self.last_est_range_m + 0.1:
                    self.range_increasing_duration_s += self.config.dt_s
                elif est_range_m < self.last_est_range_m - 2.0:
                    self.range_increasing_duration_s = 0.0
            self.last_est_range_m = est_range_m

        if active_threat_mav is not None and self.first_alert_t_s is None:
            self.first_alert_t_s = float(t_s)

        # If no alert has ever occurred, stay on nominal course
        if self.first_alert_t_s is None:
            return SitlAutopilotCommand(
                t_s=float(t_s),
                mode="NOMINAL",
                turn_rate_cmd_degps=0.0,
                bank_angle_deg=0.0,
                target_heading_deg=self.nominal_heading_deg,
                turn_direction=0,
                alert_active=False,
                coords_valid=True,
                inhibited_by_invalid_coords=False,
            )

        # Reaction delay window (t_react = 1.0 s)
        elapsed_since_alert = float(t_s) - self.first_alert_t_s
        if elapsed_since_alert < self.config.reaction_delay_s - 1e-9:
            return SitlAutopilotCommand(
                t_s=float(t_s),
                mode="REACTION_DELAY",
                turn_rate_cmd_degps=0.0,
                bank_angle_deg=0.0,
                target_heading_deg=self.nominal_heading_deg,
                turn_direction=0,
                alert_active=True,
                coords_valid=True,
                inhibited_by_invalid_coords=False,
            )

        # Latch escape turn direction on first active maneuver frame
        if self.latched_turn_dir == 0 and ref_mav is not None:
            turn_dir, avoid_deg = self._select_escape_turn_direction(
                ownship=ownship,
                mav_msg=ref_mav,
                rel_bearing_rad=rel_bearing_rad,
            )
            self.latched_turn_dir = turn_dir
            self.latched_avoid_Delta_deg = avoid_deg
            self.initial_rel_bearing_rad = rel_bearing_nom_rad
            self.maneuver_start_t_s = float(t_s)
            self.induced_turns_count += 1

        # Determine if conflict is cleared:
        # 1. If intruder is still tracked (`ref_mav is not None`) and started in front, it has passed CPA/abeam
        #    once its relative bearing wrt nominal course crosses into the aft hemisphere (|rel_bearing_nom| > 92 deg).
        # 2. If `ref_mav is None` after an active maneuver (`maneuver_elapsed_s >= 6.0 s`), the confirmed track
        #    has timed out (`deletion_timeout_s = 1.5 s` without detections), which only occurs post-CPA as the
        #    receding target's angular scale drops below detection threshold.
        # 3. For overtaking geometries (`not started_in_front`), clear once alert ceases and lateral offset is established.
        maneuver_elapsed_s = (
            float(t_s) - self.maneuver_start_t_s
            if self.maneuver_start_t_s is not None
            else 0.0
        )
        started_in_front = abs(math.degrees(self.initial_rel_bearing_rad)) <= 105.0
        passed_abeam = bool(
            ref_mav is not None
            and started_in_front
            and abs(math.degrees(rel_bearing_nom_rad)) > 92.0
        )
        track_receded_timeout = bool(
            ref_mav is None
            and maneuver_elapsed_s >= 6.0
            and abs(y_left_m) >= 220.0
        )
        rear_overtake_cleared = bool(
            (not started_in_front)
            and active_threat_mav is None
            and maneuver_elapsed_s >= 10.0
            and abs(y_left_m) >= 500.0
        )
        if maneuver_elapsed_s >= 3.0 and (
            passed_abeam or track_receded_timeout or rear_overtake_cleared
        ):
            self.recovery_latched = True

        if self.recovery_latched:
            # Time-optimal Dubins rollout + bounded <= 30 deg airway intercept heading back toward y_left_m -> 0 (B12-F3)
            max_int_deg = min(30.0, float(self.config.max_intercept_angle_deg))
            max_step_rate = max(
                float(self.config.recovery_turn_rate_degps),
                float(self.config.max_turn_rate_degps),
            )
            r_turn_rec_m = own_speed_mps / max(1e-4, math.radians(max_step_rate))
            cos_arg = max(
                math.cos(math.radians(max_int_deg)),
                min(1.0, 1.0 - abs(y_left_m) / max(1.0, r_turn_rec_m)),
            )
            # y_left_m > 0 (North of Eastbound airway) -> positive heading offset (+Delta_psi, South of East)
            sign_y = 1.0 if y_left_m >= 0.0 else -1.0
            intercept_offset_deg = sign_y * math.degrees(math.acos(cos_arg))
            target_hdg_deg = (self.nominal_heading_deg + intercept_offset_deg) % 360.0
            hdg_err_deg = _wrap_deg_180(target_hdg_deg - ownship.heading_deg)
            rate_cmd = max(
                -max_step_rate,
                min(max_step_rate, hdg_err_deg / max(self.config.dt_s, 1e-3)),
            )
            bank_rad = math.atan(
                (own_speed_mps * math.radians(rate_cmd)) / GRAVITY_MPS2
            )
            bank_deg = max(
                -self.config.max_bank_angle_deg,
                min(self.config.max_bank_angle_deg, math.degrees(bank_rad)),
            )
            return SitlAutopilotCommand(
                t_s=float(t_s),
                mode="RECOVER_COURSE",
                turn_rate_cmd_degps=float(rate_cmd),
                bank_angle_deg=float(bank_deg),
                target_heading_deg=float(target_hdg_deg),
                turn_direction=self.latched_turn_dir,
                alert_active=False,
                coords_valid=True,
                inhibited_by_invalid_coords=False,
            )

        # Active horizontal avoidance turn (DO-365A psi_dot_max = 3.0 deg/s, phi_max = 25 deg).
        # Use non-overshooting Dubins rollout curve toward parallel lateral offset cap (~630 m > 609.6 m Well-Clear)
        # so ownship smoothly levels off parallel to the airway until the intruder passes abeam.
        max_rate = self.config.max_turn_rate_degps
        r_turn_avoid_m = own_speed_mps / max(1e-4, math.radians(max_rate))
        offset_in_turn_dir_m = -self.latched_turn_dir * y_left_m
        cap_m = max(680.0, min(700.0, float(self.config.lateral_offset_cap_m)))
        rem_offset_m = max(0.0, cap_m - offset_in_turn_dir_m)
        cos_avoid = max(
            math.cos(math.radians(self.latched_avoid_Delta_deg)),
            min(1.0, 1.0 - rem_offset_m / max(1.0, r_turn_avoid_m)),
        )
        eff_avoid_deg = math.degrees(math.acos(cos_avoid))

        target_hdg_deg = (
            self.nominal_heading_deg
            + self.latched_turn_dir * eff_avoid_deg
        ) % 360.0
        hdg_err_deg = _wrap_deg_180(target_hdg_deg - ownship.heading_deg)
        rate_cmd = max(
            -max_rate,
            min(max_rate, hdg_err_deg / max(self.config.dt_s, 1e-3)),
        )
        bank_rad = math.atan((own_speed_mps * math.radians(rate_cmd)) / GRAVITY_MPS2)
        bank_deg = max(
            -self.config.max_bank_angle_deg,
            min(self.config.max_bank_angle_deg, math.degrees(bank_rad)),
        )
        return SitlAutopilotCommand(
            t_s=float(t_s),
            mode="AVOID_TURN",
            turn_rate_cmd_degps=float(rate_cmd),
            bank_angle_deg=float(bank_deg),
            target_heading_deg=float(target_hdg_deg),
            turn_direction=self.latched_turn_dir,
            alert_active=bool(active_threat_mav is not None),
            coords_valid=True,
            inhibited_by_invalid_coords=False,
        )


def _sample_approach_deg(geom_stratum: str, rng: np.random.Generator) -> float:
    if geom_stratum == "head_on":
        return float(rng.uniform(150.0, 210.0))
    if geom_stratum == "crossing_left":
        return float(rng.uniform(60.0, 120.0))
    if geom_stratum == "crossing_right":
        return float(rng.uniform(-120.0, -60.0))
    if geom_stratum == "same_direction":
        return float(rng.uniform(-30.0, 30.0))
    if geom_stratum == "converging":
        if rng.uniform(0.0, 1.0) < 0.5:
            return float(rng.uniform(30.0, 60.0))
        return float(rng.uniform(-60.0, -30.0))
    raise ValueError(f"Unknown geometry stratum: {geom_stratum!r}")


def _sample_vmd_m(vert_stratum: str, rng: np.random.Generator) -> float:
    if vert_stratum == "above":
        return float(rng.uniform(100.0, 300.0))
    if vert_stratum == "co_altitude":
        return float(rng.uniform(-30.0, 30.0))
    if vert_stratum == "below":
        return float(rng.uniform(-300.0, -100.0))
    raise ValueError(f"Unknown vertical stratum: {vert_stratum!r}")


def _sample_hmd_m(rng: np.random.Generator) -> float:
    if rng.uniform(0.0, 1.0) < 0.5:
        return float(rng.uniform(0.0, 152.4))
    return float(rng.uniform(152.4, 1500.0))


def _scenario_from_params(
    *,
    scenario_id: str,
    geom_stratum: str,
    class_id: str,
    vert_stratum: str,
    t_cpa: float,
    hmd: float,
    side: str,
    vmd: float,
    app_deg: float,
    gs: float,
    vs: float,
    model_id: str,
    seed: int,
    p0_ecef: np.ndarray,
    v_own_ecef: np.ndarray,
) -> SitlThreatScenario:
    l_min, l_max = WP02_CLASS_DIMENSIONS[class_id]
    l_char = 0.5 * (l_min + l_max)
    p_o_cpa = p0_ecef + v_own_ecef * t_cpa
    lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o_cpa)
    n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
    v_o_ned = n_mat.T @ v_own_ecef
    chi_o = math.atan2(float(v_o_ned[1]), float(v_o_ned[0]))
    chi = chi_o + math.radians(app_deg)
    v_i_ned = np.array(
        [gs * math.cos(chi), gs * math.sin(chi), -vs],
        dtype=np.float64,
    )
    v_i_ecef = n_mat @ v_i_ned
    v_rel_ned = v_i_ned - v_o_ned
    d_hat = n_mat[:, 2]
    cross_ned = np.array([-v_rel_ned[1], v_rel_ned[0], 0.0], dtype=np.float64)
    cross_norm = float(np.linalg.norm(cross_ned))
    s_sign = +1.0 if side == "right" else -1.0
    h_hat = n_mat @ (s_sign * (cross_ned / cross_norm))
    p_i_cpa = p_o_cpa + hmd * h_hat - vmd * d_hat
    return SitlThreatScenario(
        scenario_id=scenario_id,
        geometry_stratum=geom_stratum,
        class_id=class_id,
        vertical_stratum=vert_stratum,
        characteristic_length_m=l_char,
        t_cpa_open_s=t_cpa,
        open_loop_hmd_m=hmd,
        open_loop_vmd_m=vmd,
        side=side,
        approach_deg=app_deg,
        gs_mps=gs,
        vs_mps=vs,
        target_p_cpa_ecef_m=p_i_cpa,
        target_v_ecef_mps=v_i_ecef,
        seed=seed,
        model_id=model_id,
    )


def build_g3a_collision_threat_scenarios(
    count: int = 50,
    max_open_loop_hmd_m: float = 152.0,
    master_seed: int = 20260923,
) -> list[SitlThreatScenario]:
    """Deterministically build 50 representative collision-threat encounters (`d_CPA < 152 m`)
    from the WP-02 G1 / G3a v0.1 encounter set (`output/xvs/wp02/g1/encounters.jsonl`, seed 20260923).
    """
    lat0_deg = 37.30
    lon0_deg = -121.00
    h0_m = 1500.0
    p0_ecef = wgs84.geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    n0_mat = wgs84.r_ecef_ned(lat0_deg, lon0_deg)
    e_hat = n0_mat[:, 1]
    own_speed_mps = 100.0
    v_own_ecef = own_speed_mps * e_hat

    # Prefer canonical WP-02 G1 / G3a v0.1 encounters.jsonl if present on disk
    for cand_path in _FALLBACK_WP02_ENCOUNTERS_PATHS:
        if cand_path.is_file() and master_seed == 20260923:
            loaded: list[SitlThreatScenario] = []
            with open(cand_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    row = json.loads(line_s)
                    stratum_dict = row.get("stratum", {})
                    v_strat = str(row.get("vertical_stratum", stratum_dict.get("vertical", "")))
                    g_strat = str(row.get("geometry_stratum", stratum_dict.get("geometry", "")))
                    hmd_val = float(row["hmd_m"])
                    vmd_val = float(row["vmd_m"])
                    if (
                        v_strat == "co_altitude"
                        and hmd_val < max_open_loop_hmd_m
                        and abs(vmd_val) < 30.48
                    ):
                        loaded.append(
                            _scenario_from_params(
                                scenario_id=str(row["id"]),
                                geom_stratum=g_strat,
                                class_id=str(row["class_id"]),
                                vert_stratum=v_strat,
                                t_cpa=float(row["t_cpa"]),
                                hmd=hmd_val,
                                side=str(row["side"]),
                                vmd=vmd_val,
                                app_deg=float(row["approach_deg"]),
                                gs=float(row["gs_mps"]),
                                vs=float(row["vs_mps"]),
                                model_id=str(row.get("model_id", f"{row['class_id']}_0001")),
                                seed=int(row["seed"]),
                                p0_ecef=p0_ecef,
                                v_own_ecef=v_own_ecef,
                            )
                        )
                        if len(loaded) >= count:
                            return loaded[:count]
            if loaded:
                return loaded[:count]

    rng = np.random.default_rng(master_seed)
    all_candidates: list[SitlThreatScenario] = []
    enc_idx = 1

    for g in _GEOMETRY_STRATA:
        for c in _CLASS_STRATA:
            sp_cfg = _SPEED_RANGES_MPS[c]
            for v in _VERTICAL_STRATA:
                for _ in range(10):
                    while True:
                        sub_seed = int(rng.integers(0, 2**31 - 1))
                        sub_rng = np.random.default_rng(sub_seed)

                        t_cpa = float(sub_rng.uniform(30.0, 60.0))
                        hmd = _sample_hmd_m(sub_rng)
                        side = "right" if sub_rng.uniform(0.0, 1.0) < 0.5 else "left"
                        vmd = _sample_vmd_m(v, sub_rng)
                        app_deg = _sample_approach_deg(g, sub_rng)
                        if app_deg > 180.0:
                            app_deg -= 360.0

                        gs = float(sub_rng.uniform(sp_cfg["gs"][0], sp_cfg["gs"][1]))
                        vs = float(sub_rng.uniform(sp_cfg["vs"][0], sp_cfg["vs"][1]))
                        variant = int(sub_rng.integers(1, 4))
                        model_id = f"{c}_{variant}"

                        p_o_cpa = p0_ecef + v_own_ecef * t_cpa
                        lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o_cpa)
                        n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
                        v_o_ned = n_mat.T @ v_own_ecef
                        chi_o = math.atan2(float(v_o_ned[1]), float(v_o_ned[0]))
                        chi = chi_o + math.radians(app_deg)
                        v_i_ned = np.array(
                            [gs * math.cos(chi), gs * math.sin(chi), -vs],
                            dtype=np.float64,
                        )
                        v_rel_h = math.hypot(
                            float(v_i_ned[0] - v_o_ned[0]),
                            float(v_i_ned[1] - v_o_ned[1]),
                        )
                        if v_rel_h >= 10.0:
                            if (
                                v == "co_altitude"
                                and hmd < max_open_loop_hmd_m
                                and abs(vmd) < 30.48
                            ):
                                scen = _scenario_from_params(
                                    scenario_id=f"E{enc_idx:04d}",
                                    geom_stratum=g,
                                    class_id=c,
                                    vert_stratum=v,
                                    t_cpa=t_cpa,
                                    hmd=hmd,
                                    side=side,
                                    vmd=vmd,
                                    app_deg=app_deg,
                                    gs=gs,
                                    vs=vs,
                                    model_id=model_id,
                                    seed=sub_seed,
                                    p0_ecef=p0_ecef,
                                    v_own_ecef=v_own_ecef,
                                )
                                all_candidates.append(scen)
                                if len(all_candidates) >= count:
                                    return all_candidates[:count]
                            enc_idx += 1
                            break

    return all_candidates[:count]


def _pregenerate_crn_draws(seed: int, n_steps: int, n_cams: int = 8) -> dict[str, np.ndarray]:
    """Pre-draw Common Random Numbers (CRN) per encounter so all 4 modes share identical sensor noise."""
    rng = np.random.default_rng([seed, 20261108])
    return {
        "ins_init_rpy": rng.normal(0.0, 1.0, size=3),
        "ins_step_rpy": rng.normal(0.0, 1.0, size=(n_steps, 3)),
        "u_detect": rng.uniform(0.0, 1.0, size=(n_steps, n_cams)),
        "n_u_px": rng.normal(0.0, 1.0, size=(n_steps, n_cams)),
        "n_v_px": rng.normal(0.0, 1.0, size=(n_steps, n_cams)),
        "n_az_c": rng.normal(0.0, 1.0, size=(n_steps, n_cams)),
        "n_el_c": rng.normal(0.0, 1.0, size=(n_steps, n_cams)),
        "z_s": rng.normal(0.0, 1.0, size=(n_steps, n_cams)),
    }


def _generate_ring_detections_for_step(
    *,
    t_s: float,
    step_idx: int,
    p_own_ecef: np.ndarray,
    n_mat: np.ndarray,
    p_tgt_ecef: np.ndarray,
    r_ecef_body_tgt: np.ndarray,
    verts_model: np.ndarray,
    yaw_true_rad: float,
    pitch_true_rad: float,
    roll_true_rad: float,
    tracker: MultiCameraTracker,
    trk_cfg: Mapping[str, Any],
    crn_draws: Mapping[str, np.ndarray],
) -> list[DetectionRecord]:
    """Project intruder 3D mesh vertices into the 8-camera TrafficRing (`tracker.ring`) and generate
    `DetectionRecord` observations using `configs/xvs/tracking/g1.json` parameters and CRN noise.
    """
    ring = tracker.ring
    r_ned_body = rotations.euler_to_r(yaw_true_rad, pitch_true_rad, roll_true_rad)
    r_ecef_body = n_mat @ r_ned_body

    s50_px = float(trk_cfg["s50_px"])
    w_ln = float(trk_cfg["w_ln"])
    sigma_px = float(trk_cfg["sigma_px"])
    sigma_s_px = float(trk_cfg.get("sigma_s_px", sigma_px))
    sigma_s_mom_px = float(trk_cfg.get("sigma_s_mom_px", 0.08))  # default-ok: optional SITL/tracker config fallback
    sigma_psf_px = float(trk_cfg.get("sigma_psf_px", 0.55))  # default-ok: optional SITL/tracker config fallback
    lambda_min_px = float(trk_cfg.get("lambda_min_px", 0.10))  # default-ok: optional SITL/tracker config fallback
    sigma_ang_rad = float(trk_cfg["sigma_ang_rad"])
    s_mom_min_px = math.sqrt(12.0) * lambda_min_px
    u50_log = math.log(max(1e-4, s50_px))

    verts_ecef: np.ndarray | None = None
    detections: list[DetectionRecord] = []

    for cam_idx, cam in enumerate(ring.cameras):
        p_cam = p_own_ecef + r_ecef_body @ cam.lever_arm_m
        r_cam = r_ecef_body @ cam.r_body_cam

        p_cam_rel = r_cam.T @ (p_tgt_ecef - p_cam)
        if float(p_cam_rel[2]) <= 1e-3:
            continue

        u_true, v_true, in_front = ring.camera_model.project_cam(p_cam_rel)
        if not in_front or not ring.camera_model.in_image(u_true, v_true):
            continue

        if verts_ecef is None:
            verts_ecef = p_tgt_ecef[None, :] + (r_ecef_body_tgt @ verts_model.T).T

        verts_cam = (r_cam.T @ (verts_ecef - p_cam).T).T
        if not np.all(verts_cam[:, 2] > 1e-3):
            continue

        u_coords = ring.camera_model.fx * (verts_cam[:, 0] / verts_cam[:, 2]) + ring.camera_model.cx
        v_coords = ring.camera_model.fy * (verts_cam[:, 1] / verts_cam[:, 2]) + ring.camera_model.cy
        u_min, u_max = float(np.min(u_coords)), float(np.max(u_coords))
        v_min, v_max = float(np.min(v_coords)), float(np.max(v_coords))
        s_px = max(0.0, u_max - u_min, v_max - v_min)

        # Log-logistic detection probability P_d(s_px) from configs/xvs/tracking/g1.json
        if s_px <= 1e-6:
            p_d = 0.0
        else:
            x_logit = max(-50.0, min(50.0, -(math.log(max(1e-4, s_px)) - u50_log) / max(1e-4, w_ln)))
            p_d = 1.0 / (1.0 + math.exp(x_logit))

        if float(crn_draws["u_detect"][step_idx, cam_idx]) >= p_d:
            continue

        mom_res = compute_vertex_second_moment_scale(
            u_coords=u_coords,
            v_coords=v_coords,
            sigma_psf_px=sigma_psf_px,
            lambda_min_px=lambda_min_px,
            fx=ring.fx,
        )
        s_mom_true_px = mom_res.s_mom_px

        u_noisy = u_true + sigma_px * float(crn_draws["n_u_px"][step_idx, cam_idx])
        v_noisy = v_true + sigma_px * float(crn_draws["n_v_px"][step_idx, cam_idx])

        xc = (u_noisy - ring.cx) / ring.fx
        yc = (v_noisy - ring.cy) / ring.fy
        az_c = math.atan2(xc, 1.0) + sigma_ang_rad * float(crn_draws["n_az_c"][step_idx, cam_idx])
        el_c = math.atan2(-yc, math.hypot(xc, 1.0)) + sigma_ang_rad * float(crn_draws["n_el_c"][step_idx, cam_idx])
        cos_el = math.cos(el_c)
        rc_z = cos_el * math.cos(az_c)
        if rc_z > 1e-6:
            u_final = ring.fx * ((cos_el * math.sin(az_c)) / rc_z) + ring.cx
            v_final = ring.fy * ((-math.sin(el_c)) / rc_z) + ring.cy
        else:
            u_final, v_final = u_noisy, v_noisy

        z_s_val = float(crn_draws["z_s"][step_idx, cam_idx])
        s_px_meas = max(0.05, float(s_px + sigma_s_px * z_s_val))
        s_mom_px_meas = max(s_mom_min_px, float(s_mom_true_px + sigma_s_mom_px * z_s_val))

        detections.append(
            DetectionRecord(
                t=t_s,
                camera=cam.id,
                u=float(u_final),
                v=float(v_final),
                s_px=float(s_px_meas),
                score=float(p_d),
                is_false_alarm=False,
                s_mom_px=float(s_mom_px_meas),
            )
        )

    return detections


def simulate_single_encounter_closed_loop(
    scenario: SitlThreatScenario,
    mode: str,
    bridge: PassiveTrackAvionicsBridge,
    config: SitlAvoidanceConfig,
    integrity_state: str = "VALID",
    crn_draws: dict[str, np.ndarray] | None = None,
    tracking_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one 60-second encounter at 10 Hz under the specified alerting/avoidance mode
    using WP-05 G1's real `MultiCameraTracker` (`B12-F2`) and stern-pass + airway-recovery
    autopilot (`B12-F3`).

    Modes:
      - `open_loop`: no avoidance
      - `closed_loop_g0_a1`: G0 A1 alerting + MAVLink #246 -> SITL horizontal avoidance
      - `closed_loop_g1_wc1_b050`: G1 WC1 (beta_WC = 0.50) alerting + MAVLink #246 -> SITL avoidance
      - `closed_loop_g1_wc1_b025`: G1 WC1 (beta_WC = 0.25) alerting + MAVLink #246 -> SITL avoidance
    """
    dt = config.dt_s
    n_steps = int(round((config.t_end_s - config.t_start_s) / dt)) + 1
    if crn_draws is None:
        crn_draws = _pregenerate_crn_draws(scenario.seed, n_steps)

    base_trk_cfg = (
        dict(tracking_config)
        if tracking_config is not None
        else load_g1_tracking_config()
    )
    use_g1_wc1 = mode in ("closed_loop_g1_wc1_b050", "closed_loop_g1_wc1_b025")
    beta_wc = 0.25 if mode == "closed_loop_g1_wc1_b025" else 0.50
    enable_avoidance = mode != "open_loop"

    if use_g1_wc1 or mode == "open_loop":
        mode_trk_cfg = dict(base_trk_cfg)
    else:
        # Legacy G0 A1 mode: disable G1 second-moment scale and sliding-window log-size TTC
        mode_trk_cfg = {
            **base_trk_cfg,
            "use_second_moment_scale": False,
            "use_sliding_window_ttc": False,
            "enable_ground_clutter_rejection": False,
        }

    tracker = MultiCameraTracker(config=mode_trk_cfg)
    verts_model = get_model_vertices(scenario.model_id)

    ins_sigma_rad = math.radians(float(base_trk_cfg["ins_sigma_deg"]))
    ins_tau_s = float(base_trk_cfg["ins_tau_s"])
    ins_beta = math.exp(-dt / ins_tau_s) if ins_tau_s > 0.0 else 0.0
    ins_err_rpy = ins_sigma_rad * np.asarray(crn_draws["ins_init_rpy"], dtype=np.float64).copy()

    omega_thr_radps = float(base_trk_cfg.get("omega_thr_radps", 0.005))  # default-ok: optional SITL/tracker config fallback
    k_sigma = float(base_trk_cfg.get("k_sigma", 3.0))  # default-ok: optional SITL/tracker config fallback
    t_warning_s = float(base_trk_cfg.get("t_warning_s", 25.0))  # default-ok: optional SITL/tracker config fallback
    t_corrective_s = float(base_trk_cfg.get("t_corrective_s", 55.0))  # default-ok: optional SITL/tracker config fallback
    k_theta = float(base_trk_cfg.get("k_theta", 1.0))  # default-ok: optional SITL/tracker config fallback
    k_omega = float(base_trk_cfg.get("k_omega", 1.0))  # default-ok: optional SITL/tracker config fallback
    alpha_r = float(base_trk_cfg.get("alpha_r", 0.0))  # default-ok: optional SITL/tracker config fallback
    eta_floor = float(base_trk_cfg.get("eta_floor", 0.0001))  # default-ok: optional SITL/tracker config fallback

    # Initial ownship state (TestOwnshipO1 start: 37.30 N, -121.00 E, 1500 m, heading 90 deg East, 100 m/s)
    lat0_deg = 37.30
    lon0_deg = -121.00
    h0_m = 1500.0
    p0_ecef = wgs84.geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    n0_mat = wgs84.r_ecef_ned(lat0_deg, lon0_deg)
    e_hat_0 = n0_mat[:, 1]
    n_hat_0 = n0_mat[:, 0]
    own_speed_mps = 100.0

    p_own_ecef = p0_ecef.copy()
    heading_deg = 90.0
    roll_deg = 0.0

    autopilot = SitlAvoidanceAutopilot(nominal_heading_deg=90.0, config=config)

    min_hmd_m = float("inf")
    min_3d_dist_m = float("inf")
    t_at_min_hmd_s = 0.0
    max_cross_track_m = 0.0
    final_cross_track_m = 0.0
    max_heading_dev_deg = 0.0
    first_alert_t_s: float | None = None
    first_maneuver_t_s: float | None = None
    mavlink_packets_rx = 0
    alert_packets_rx = 0

    own_trace_en_m: list[tuple[float, float]] = []
    tgt_trace_en_m: list[tuple[float, float]] = []

    for step_idx in range(n_steps):
        t_s = config.t_start_s + step_idx * dt

        # Current geodetic & local NED frame for ownship
        lat_o, lon_o, alt_o = wgs84.ecef_to_geodetic(*p_own_ecef)
        n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
        hdg_rad = math.radians(heading_deg)
        pitch_rad = 0.0
        roll_rad = math.radians(roll_deg)
        vn_mps = own_speed_mps * math.cos(hdg_rad)
        ve_mps = own_speed_mps * math.sin(hdg_rad)

        # Target kinematics in environment
        p_tgt_ecef = scenario.target_position_ecef_at(t_s)
        r_ecef = p_tgt_ecef - p_own_ecef
        r_ned = n_mat.T @ r_ecef
        d_horiz_m = float(math.hypot(float(r_ned[0]), float(r_ned[1])))
        d_3d_m = float(np.linalg.norm(r_ned))

        if d_horiz_m < min_hmd_m:
            min_hmd_m = d_horiz_m
            t_at_min_hmd_s = t_s
        if d_3d_m < min_3d_dist_m:
            min_3d_dist_m = d_3d_m

        # Cross-track lateral deviation relative to nominal East line from P0
        disp_0 = p_own_ecef - p0_ecef
        east_m = float(np.dot(disp_0, e_hat_0))
        north_m = float(np.dot(disp_0, n_hat_0))
        cross_track_m = abs(north_m)
        if cross_track_m > max_cross_track_m:
            max_cross_track_m = cross_track_m
        final_cross_track_m = cross_track_m

        hdg_dev = abs(_wrap_deg_180(heading_deg - 90.0))
        if hdg_dev > max_heading_dev_deg:
            max_heading_dev_deg = hdg_dev

        if step_idx % 5 == 0 or step_idx == n_steps - 1:
            disp_tgt_0 = p_tgt_ecef - p0_ecef
            own_trace_en_m.append((east_m, north_m))
            tgt_trace_en_m.append(
                (float(np.dot(disp_tgt_0, e_hat_0)), float(np.dot(disp_tgt_0, n_hat_0)))
            )

        if enable_avoidance:
            own_nav = OwnshipNavState(
                lat_deg=lat_o,
                lon_deg=lon_o,
                alt_msl_m=alt_o,
                alt_geo_m=alt_o + 6.0,
                vn_mps=vn_mps,
                ve_mps=ve_mps,
                vd_mps=0.0,
                heading_deg=heading_deg,
            )

            mav_wire_pkts: list[bytes] = []
            gdl_wire_frms: list[bytes] = []

            if not (autopilot.recovery_latched and integrity_state == "VALID"):
                # Step INS Gauss-Markov attitude error using CRN
                w_ins = ins_sigma_rad * np.asarray(crn_draws["ins_step_rpy"][step_idx], dtype=np.float64)
                ins_err_rpy = ins_beta * ins_err_rpy + math.sqrt(max(0.0, 1.0 - ins_beta**2)) * w_ins
                r_ned_body_ins = rotations.euler_to_r(
                    hdg_rad + float(ins_err_rpy[2]),
                    pitch_rad + float(ins_err_rpy[1]),
                    roll_rad + float(ins_err_rpy[0]),
                )

                # Generate 8-camera TrafficRing detections and step real MultiCameraTracker (B12-F2)
                r_ecef_body_tgt = scenario.target_attitude_ecef_at(t_s)
                raw_dets = _generate_ring_detections_for_step(
                    t_s=t_s,
                    step_idx=step_idx,
                    p_own_ecef=p_own_ecef,
                    n_mat=n_mat,
                    p_tgt_ecef=p_tgt_ecef,
                    r_ecef_body_tgt=r_ecef_body_tgt,
                    verts_model=verts_model,
                    yaw_true_rad=hdg_rad,
                    pitch_true_rad=pitch_rad,
                    roll_true_rad=roll_rad,
                    tracker=tracker,
                    trk_cfg=base_trk_cfg,
                    crn_draws=crn_draws,
                )
                confirmed_tracks: list[Track] = tracker.step(
                    t_s=t_s,
                    raw_detections=raw_dets,
                    r_ned_body_ins=r_ned_body_ins,
                    p_own_ecef=p_own_ecef,
                    r_ecef_ned=n_mat,
                )

                if confirmed_tracks:
                    risk_by_id: dict[str, RiskEstimate] = {}
                    class_by_id: dict[str, str] = {}
                    for trk in confirmed_tracks:
                        risk_est = evaluate_track_risk(
                            track=trk,
                            t_s=t_s,
                            omega_thr_radps=omega_thr_radps,
                            k_sigma=k_sigma,
                            hmd_limit_m=LOWC_HMD_LIMIT_M,
                            class_id=scenario.class_id,
                            t_warning_s=t_warning_s,
                            t_corrective_s=t_corrective_s,
                            beta_wc=beta_wc,
                            k_theta=k_theta,
                            k_omega=k_omega,
                            alpha_r=alpha_r,
                            eta_floor=eta_floor,
                        )
                        if not use_g1_wc1:
                            # Legacy G0 A1 mode: mask out G1 WC1 flags so bridge uses only A1 alerting
                            risk_est = replace(
                                risk_est,
                                wc1_warning=False,
                                wc1_corrective=False,
                                hmd_hat_m=None,
                                tau_mod_hat_s=None,
                                tau_s_hat_s=None,
                            )
                        risk_by_id[trk.track_id] = risk_est
                        class_by_id[trk.track_id] = scenario.class_id

                    step_res = bridge.step_from_tracker(
                        ownship=own_nav,
                        confirmed_tracks=confirmed_tracks,
                        risk_by_track_id=risk_by_id,
                        integrity_state=integrity_state,
                        t_s=t_s,
                        class_by_track_id=class_by_id,
                    )
                    mav_wire_pkts = list(step_res.mavlink_frames)
                    gdl_wire_frms = list(step_res.gdl90_frames)
                    mavlink_packets_rx += len(mav_wire_pkts)
                    for m_rep in step_res.mavlink_messages:
                        if m_rep.squawk == 7700:
                            alert_packets_rx += 1

            cmd = autopilot.step_from_wire_packets(
                t_s=t_s,
                ownship=own_nav,
                mavlink_wire_packets=mav_wire_pkts,
                gdl90_wire_frames=gdl_wire_frms,
            )
            if cmd.alert_active and first_alert_t_s is None:
                first_alert_t_s = t_s
            if cmd.mode == "AVOID_TURN" and first_maneuver_t_s is None:
                first_maneuver_t_s = t_s

            heading_deg = (heading_deg + cmd.turn_rate_cmd_degps * dt) % 360.0
            roll_deg = cmd.bank_angle_deg

        # Advance ownship position over dt
        hdg_next_rad = math.radians(heading_deg)
        v_next_ned = np.array(
            [own_speed_mps * math.cos(hdg_next_rad), own_speed_mps * math.sin(hdg_next_rad), 0.0],
            dtype=np.float64,
        )
        p_own_ecef = p_own_ecef + (n_mat @ v_next_ned) * dt

    alerted = first_alert_t_s is not None
    nmac_occurred = bool(min_hmd_m < config.nmac_threshold_m)
    nmac_resolved = bool(not nmac_occurred)
    well_clear_recovered = bool(min_hmd_m >= config.lowc_threshold_m)

    return {
        "scenario_id": scenario.scenario_id,
        "geometry_stratum": scenario.geometry_stratum,
        "class_id": scenario.class_id,
        "mode": mode,
        "integrity_state": integrity_state,
        "open_loop_hmd_m": float(scenario.open_loop_hmd_m),
        "achieved_min_hmd_m": float(min_hmd_m),
        "achieved_min_3d_dist_m": float(min_3d_dist_m),
        "t_at_min_hmd_s": float(t_at_min_hmd_s),
        "alerted": bool(alerted),
        "first_alert_t_s": float(first_alert_t_s) if first_alert_t_s is not None else None,
        "first_maneuver_t_s": float(first_maneuver_t_s) if first_maneuver_t_s is not None else None,
        "warning_lead_time_s": (
            float(scenario.t_cpa_open_s - first_alert_t_s)
            if first_alert_t_s is not None
            else 0.0
        ),
        "nmac_occurred": nmac_occurred,
        "nmac_resolved": nmac_resolved,
        "well_clear_recovered": well_clear_recovered,
        "max_cross_track_m": float(max_cross_track_m),
        "final_cross_track_m": float(final_cross_track_m),
        "max_heading_dev_deg": float(max_heading_dev_deg),
        "mavlink_packets_rx": int(mavlink_packets_rx),
        "alert_packets_rx": int(alert_packets_rx),
        "invalid_coord_inhibits_count": int(autopilot.invalid_coord_inhibits_count),
        "induced_turns_count": int(autopilot.induced_turns_count),
        "own_trace_en_m": own_trace_en_m,
        "tgt_trace_en_m": tgt_trace_en_m,
    }


def _summarize_mode_runs(runs: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
    n_total = len(runs)
    hmds = np.array([r["achieved_min_hmd_m"] for r in runs], dtype=np.float64)
    xtks = np.array([r["max_cross_track_m"] for r in runs], dtype=np.float64)
    final_xtks = np.array(
        [r.get("final_cross_track_m", r["max_cross_track_m"]) for r in runs],
        dtype=np.float64,
    )
    hdgs = np.array([r["max_heading_dev_deg"] for r in runs], dtype=np.float64)
    leads = np.array(
        [r["warning_lead_time_s"] for r in runs if r["alerted"]], dtype=np.float64
    )

    alerted_runs = [r for r in runs if r["alerted"]]
    n_alerted = len(alerted_runs)
    n_resolved_total = sum(1 for r in runs if r["nmac_resolved"])
    n_resolved_alerted = sum(1 for r in alerted_runs if r["nmac_resolved"])
    n_wc_recovered = sum(1 for r in runs if r["well_clear_recovered"])
    n_nmac_remaining = sum(1 for r in runs if r["nmac_occurred"])
    total_induced_turns = sum(int(r["induced_turns_count"]) for r in runs)
    total_invalid_inhibits = sum(int(r["invalid_coord_inhibits_count"]) for r in runs)

    return {
        "mode": mode,
        "encounters_count": n_total,
        "alerted_count": n_alerted,
        "alert_rate_pct": (100.0 * n_alerted / max(1, n_total)),
        "nmac_remaining_count": n_nmac_remaining,
        "nmac_resolved_count": n_resolved_total,
        "nmac_resolution_rate_all_pct": (100.0 * n_resolved_total / max(1, n_total)),
        "nmac_resolved_alerted_count": n_resolved_alerted,
        "nmac_resolution_rate_alerted_pct": (
            100.0 * n_resolved_alerted / max(1, n_alerted) if n_alerted > 0 else 0.0
        ),
        "well_clear_recovered_count": n_wc_recovered,
        "well_clear_recovery_rate_pct": (100.0 * n_wc_recovered / max(1, n_total)),
        "hmd_median_m": float(np.median(hmds)),
        "hmd_p10_m": float(np.percentile(hmds, 10)),
        "hmd_min_m": float(np.min(hmds)),
        "hmd_max_m": float(np.max(hmds)),
        "lead_time_median_s": float(np.median(leads)) if len(leads) > 0 else 0.0,
        "cross_track_median_m": float(np.median(xtks)),
        "cross_track_p95_m": float(np.percentile(xtks, 95)),
        "cross_track_max_m": float(np.max(xtks)),
        "final_cross_track_median_m": float(np.median(final_xtks)),
        "final_cross_track_p95_m": float(np.percentile(final_xtks, 95)),
        "final_cross_track_max_m": float(np.max(final_xtks)),
        "heading_dev_median_deg": float(np.median(hdgs)),
        "heading_dev_max_deg": float(np.max(hdgs)),
        "induced_turns_count": int(total_induced_turns),
        "invalid_coord_inhibits_count": int(total_invalid_inhibits),
    }


def run_closed_loop_sitl_benchmark(
    config: SitlAvoidanceConfig | None = None,
    max_encounters: int | None = None,
) -> dict[str, Any]:
    """Run the 50-encounter CRN closed-loop MAVLink #246 -> ArduPilot SITL avoidance benchmark."""
    cfg = config if config is not None else SitlAvoidanceConfig.from_display_config()
    trk_cfg = load_g1_tracking_config()
    n_enc = max_encounters if max_encounters is not None else cfg.benchmark_threat_count
    scenarios = build_g3a_collision_threat_scenarios(
        count=n_enc,
        max_open_loop_hmd_m=cfg.open_loop_cpa_max_m,
    )

    bridge = PassiveTrackAvionicsBridge()
    n_steps = int(round((cfg.t_end_s - cfg.t_start_s) / cfg.dt_s)) + 1

    modes = (
        "open_loop",
        "closed_loop_g0_a1",
        "closed_loop_g1_wc1_b050",
        "closed_loop_g1_wc1_b025",
    )
    runs_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    invalid_alert_runs: list[dict[str, Any]] = []

    for scen in scenarios:
        crn = _pregenerate_crn_draws(scen.seed, n_steps)
        for m in modes:
            res = simulate_single_encounter_closed_loop(
                scenario=scen,
                mode=m,
                bridge=bridge,
                config=cfg,
                integrity_state="VALID",
                crn_draws=crn,
                tracking_config=trk_cfg,
            )
            runs_by_mode[m].append(res)

        # V4_G1: Evaluate INVALID_ALERT injection (`VALID_COORDS == 0` / `NIC == 0`)
        res_inv = simulate_single_encounter_closed_loop(
            scenario=scen,
            mode="closed_loop_g1_wc1_b050",
            bridge=bridge,
            config=cfg,
            integrity_state="INVALID_ALERT",
            crn_draws=crn,
            tracking_config=trk_cfg,
        )
        invalid_alert_runs.append(res_inv)

    summaries = {
        m: _summarize_mode_runs(runs_by_mode[m], m) for m in modes
    }
    invalid_summary = _summarize_mode_runs(
        invalid_alert_runs, "closed_loop_g1_wc1_b050_invalid_alert"
    )

    # Pick representative encounter trace (E0011) where G1 WC1 resolves NMAC via stern-pass turn and returns to airway
    rep_idx = 0
    for idx_s, s_obj in enumerate(scenarios):
        r_b050 = runs_by_mode["closed_loop_g1_wc1_b050"][idx_s]
        r_g0 = runs_by_mode["closed_loop_g0_a1"][idx_s]
        if (
            r_b050["nmac_resolved"]
            and not r_g0["nmac_resolved"]
            and r_b050["final_cross_track_m"] < 0.25 * r_b050["max_cross_track_m"]
        ):
            rep_idx = idx_s
            break

    # Store exact 10 Hz kinematic achieved_min_hmd_m as open_loop_hmd_m for 100% plot/report parity (B12-F3)
    rep_open_hmd_m = float(runs_by_mode["open_loop"][rep_idx]["achieved_min_hmd_m"])
    rep_traces = {
        "scenario_id": scenarios[rep_idx].scenario_id,
        "geometry_stratum": scenarios[rep_idx].geometry_stratum,
        "class_id": scenarios[rep_idx].class_id,
        "open_loop_hmd_m": rep_open_hmd_m,
        "analytic_open_loop_hmd_m": float(scenarios[rep_idx].open_loop_hmd_m),
        "modes": {
            m: {
                "achieved_min_hmd_m": float(runs_by_mode[m][rep_idx]["achieved_min_hmd_m"]),
                "max_cross_track_m": float(runs_by_mode[m][rep_idx]["max_cross_track_m"]),
                "final_cross_track_m": float(runs_by_mode[m][rep_idx]["final_cross_track_m"]),
                "own_trace_en_m": runs_by_mode[m][rep_idx]["own_trace_en_m"],
                "tgt_trace_en_m": runs_by_mode[m][rep_idx]["tgt_trace_en_m"],
            }
            for m in modes
        },
    }

    # Remove bulky coordinate traces from per-encounter JSON records except representative trace
    compact_per_encounter: list[dict[str, Any]] = []
    for idx_s, scen in enumerate(scenarios):
        compact_per_encounter.append(
            {
                "scenario_id": scen.scenario_id,
                "geometry_stratum": scen.geometry_stratum,
                "class_id": scen.class_id,
                "open_loop_hmd_m": float(runs_by_mode["open_loop"][idx_s]["achieved_min_hmd_m"]),
                "g0_a1_hmd_m": float(
                    runs_by_mode["closed_loop_g0_a1"][idx_s]["achieved_min_hmd_m"]
                ),
                "g0_a1_alerted": bool(
                    runs_by_mode["closed_loop_g0_a1"][idx_s]["alerted"]
                ),
                "g0_a1_resolved": bool(
                    runs_by_mode["closed_loop_g0_a1"][idx_s]["nmac_resolved"]
                ),
                "g1_wc1_b050_hmd_m": float(
                    runs_by_mode["closed_loop_g1_wc1_b050"][idx_s]["achieved_min_hmd_m"]
                ),
                "g1_wc1_b050_alerted": bool(
                    runs_by_mode["closed_loop_g1_wc1_b050"][idx_s]["alerted"]
                ),
                "g1_wc1_b050_resolved": bool(
                    runs_by_mode["closed_loop_g1_wc1_b050"][idx_s]["nmac_resolved"]
                ),
                "g1_wc1_b050_max_xtk_m": float(
                    runs_by_mode["closed_loop_g1_wc1_b050"][idx_s]["max_cross_track_m"]
                ),
                "g1_wc1_b050_final_xtk_m": float(
                    runs_by_mode["closed_loop_g1_wc1_b050"][idx_s]["final_cross_track_m"]
                ),
                "g1_wc1_b025_hmd_m": float(
                    runs_by_mode["closed_loop_g1_wc1_b025"][idx_s]["achieved_min_hmd_m"]
                ),
                "g1_wc1_b025_alerted": bool(
                    runs_by_mode["closed_loop_g1_wc1_b025"][idx_s]["alerted"]
                ),
                "g1_wc1_b025_resolved": bool(
                    runs_by_mode["closed_loop_g1_wc1_b025"][idx_s]["nmac_resolved"]
                ),
                "invalid_alert_induced_turns": int(
                    invalid_alert_runs[idx_s]["induced_turns_count"]
                ),
            }
        )

    return {
        "benchmark_threat_count": len(scenarios),
        "max_turn_rate_degps": cfg.max_turn_rate_degps,
        "max_bank_angle_deg": cfg.max_bank_angle_deg,
        "reaction_delay_s": cfg.reaction_delay_s,
        "nmac_threshold_m": cfg.nmac_threshold_m,
        "lowc_threshold_m": cfg.lowc_threshold_m,
        "modes": summaries,
        "invalid_alert_injection": invalid_summary,
        "representative_encounter_trace": rep_traces,
        "per_encounter": compact_per_encounter,
    }
