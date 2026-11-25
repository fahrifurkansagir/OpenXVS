# Copyright 2026 OpenXVS Contributors
# SPDX-License-Identifier: Apache-2.0
"""DR-18 clean-room deterministic forced-landing planner & passive LEO-Doppler SoOP guard.

All spatial coordinates strictly follow the OpenXVS NED convention:
  - ``north_m``: North displacement (m)
  - ``east_m``: East displacement (m)
  - ``down_m``: Down displacement (m; altitude MSL/local = -down_m)
  - ``wind_ned_mps``: Ambient air-mass velocity vector ``(w_n, w_e, w_d)`` in NED,
    where ``w_d > 0`` denotes downdraft / sinking air.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

G_STANDARD_MPS2: float = 9.80665
RHO_SL_KG_M3: float = 1.225
MAX_NOMINAL_BANK_DEG: float = 35.0

_VLM_RANK_STRUCT = struct.Struct("<4BHQ")
_VLM_WIRE_STRUCT = struct.Struct("<4BHQI")
_UNUSED_FLZ_SLOT: int = 0xFF

_DOF1_HUBER_THRESHOLD: float = 6.63
_DOF1_HARD_REJECT_THRESHOLD: float = 10.83
_DOF3_PITOT_HARD_REJECT_THRESHOLD: float = 16.27

_BANNED_OVERCLOUD_NAV_SOURCES: frozenset[str] = frozenset(
    {
        "magnav",
        "magnetic_nav",
        "magnetic_anomaly",
        "magnetometer_map",
        "active_dme",
        "dme",
        "dme_interrogator",
        "dme_tx",
    }
)

_ALLOWED_PASSIVE_NAV_SOURCES: frozenset[str] = frozenset(
    {
        "leo_doppler_soop",
        "leo_starlink_ku",
        "leo_iridium_l",
        "ebs_star_tracker",
        "optical_trnav",
    }
)


class ForcedLandPolicyError(PermissionError):
    """Fail-closed policy violation in DR-18 navigation or forced-landing safety gates."""


@dataclass(frozen=True)
class FixedWingPolar:
    """Parabolic drag polar for a civil fixed-wing UAS (`C_D = C_D0 + k * C_L^2`)."""

    name: str
    mass_kg: float
    wingspan_m: float
    wing_area_m2: float
    cd0: float
    induced_k: float
    v_stall_mps: float
    v_max_mps: float
    min_strip_len_m: float
    min_strip_width_m: float
    v_touchdown_mps: float
    rho_kg_m3: float = RHO_SL_KG_M3

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0 or self.wing_area_m2 <= 0.0:
            raise ValueError("mass_kg and wing_area_m2 must be strictly positive")
        if self.cd0 <= 0.0 or self.induced_k <= 0.0:
            raise ValueError("cd0 and induced_k must be strictly positive")
        if self.v_stall_mps <= 0.0 or self.v_max_mps <= 1.15 * self.v_stall_mps:
            raise ValueError("v_max_mps must exceed 1.15 * v_stall_mps")
        if self.rho_kg_m3 <= 0.0:
            raise ValueError("rho_kg_m3 must be strictly positive")

    @property
    def weight_n(self) -> float:
        return self.mass_kg * G_STANDARD_MPS2

    def sink_rate_mps(self, v_tas_mps: float) -> float:
        """Parabolic level-flight sink rate `V_s(V)` in still air (m/s, positive down)."""
        if v_tas_mps <= 0.0:
            raise ValueError("v_tas_mps must be strictly positive")
        q_s = 0.5 * self.rho_kg_m3 * self.wing_area_m2
        w_n = self.weight_n
        parasitic = (q_s * self.cd0 * (v_tas_mps**3)) / w_n
        induced = (self.induced_k * w_n) / (q_s * v_tas_mps)
        return parasitic + induced


POLAR_PRESETS: dict[str, FixedWingPolar] = {
    "light_3kg": FixedWingPolar(
        name="light_3kg",
        mass_kg=3.0,
        wingspan_m=1.24,
        wing_area_m2=(2.0 * 3.0 * G_STANDARD_MPS2) / (RHO_SL_KG_M3 * 0.70 * (15.0**2)),
        cd0=0.035,
        induced_k=1.0 / (4.0 * (10.0**2) * 0.035),
        v_stall_mps=11.0,
        v_max_mps=28.0,
        min_strip_len_m=110.0,
        min_strip_width_m=15.0,
        v_touchdown_mps=13.7,
    ),
    "nominal_12kg": FixedWingPolar(
        name="nominal_12kg",
        mass_kg=12.0,
        wingspan_m=2.35,
        wing_area_m2=(2.0 * 12.0 * G_STANDARD_MPS2) / (RHO_SL_KG_M3 * 0.70 * (19.7**2)),
        cd0=0.025,
        induced_k=1.0 / (4.0 * (14.0**2) * 0.025),
        v_stall_mps=14.5,
        v_max_mps=35.0,
        min_strip_len_m=150.0,
        min_strip_width_m=15.0,
        v_touchdown_mps=16.0,
    ),
    "heavy_25kg": FixedWingPolar(
        name="heavy_25kg",
        mass_kg=25.0,
        wingspan_m=3.37,
        wing_area_m2=(2.0 * 25.0 * G_STANDARD_MPS2) / (RHO_SL_KG_M3 * 0.72 * (23.0**2)),
        cd0=0.020,
        induced_k=1.0 / (4.0 * (18.0**2) * 0.020),
        v_stall_mps=16.5,
        v_max_mps=42.0,
        min_strip_len_m=180.0,
        min_strip_width_m=20.0,
        v_touchdown_mps=18.6,
    ),
}


@dataclass(frozen=True)
class BestGlideParams:
    """Still-air best-glide operating point derived from a `FixedWingPolar`."""

    ld_max: float
    cl_bg: float
    v_bg_mps: float
    v_sink_bg_mps: float
    glide_angle_rad: float


@dataclass(frozen=True)
class BankedTurnSchedule:
    """Coordinated gliding turn schedule at bank angle ``phi_rad``."""

    phi_rad: float
    phi_deg: float
    load_factor_g: float
    v_opt_mps: float
    v_sink_mps: float
    v_stall_banked_mps: float
    turn_radius_m: float
    delta_h_360_m: float
    delta_h_180_m: float


@dataclass(frozen=True)
class MacCreadySolution:
    """Optimal MacCready speed-to-fly under head/tail/crosswind and vertical air motion."""

    v_opt_mps: float
    v_ground_mps: float
    v_sink_total_mps: float
    effective_glide_ratio: float
    crab_angle_rad: float


@dataclass(frozen=True)
class AircraftGlideState:
    """Current aircraft kinematic & ambient wind state in strictly NED coordinates."""

    north_m: float
    east_m: float
    down_m: float
    heading_rad: float
    wind_ned_mps: tuple[float, float, float]


@dataclass(frozen=True)
class FlzCandidate:
    """Candidate Forced Landing Zone derived from 3DEP S1M terrain & exclusion layers."""

    flz_id: int
    north_m: float
    east_m: float
    down_m: float
    final_approach_hdg_rad: float
    slope_deg: float
    cross_slope_deg: float
    rms_roughness_m: float
    ndsm_m: float
    exclusion_overlap_frac: float
    lateral_setback_m: float
    approach_corridor_clear_10_to_1: bool
    strip_length_m: float
    strip_width_m: float
    obstacle_clearance_m: float = 15.0


@dataclass(frozen=True)
class FlzEvaluationResult:
    """Deterministic geometric reachability & 5-gate 3DEP screening output."""

    reachable_flz_mask: int
    geom_ranked_ids: tuple[int, ...]
    energy_margins_m: dict[int, float]
    rejection_reasons: dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class VlmRankMsg:
    """Fixed-schema integer-only VLM advisory message (`<4BHQI`, 18 bytes, zero floats)."""

    ranked_ids: tuple[int, int, int, int]
    hazard_veto_mask: int
    timestamp_ms: int
    crc32: int

    @classmethod
    def build(
        cls,
        ranked_ids: Sequence[int],
        hazard_veto_mask: int,
        timestamp_ms: int,
    ) -> VlmRankMsg:
        if len(ranked_ids) != 4:
            raise ValueError("ranked_ids must contain exactly 4 integer slots (use 0xFF for unused)")
        slots = tuple(ranked_ids)
        for slot in slots:
            if not _is_exact_int(slot) or slot < 0 or slot > 0xFF:
                raise ValueError("ranked_ids slots must be uint8 integers")
        if not _is_exact_int(hazard_veto_mask) or not (0 <= hazard_veto_mask <= 0xFFFF):
            raise ValueError("hazard_veto_mask must be uint16")
        if not _is_exact_int(timestamp_ms) or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative uint64")
        body = _VLM_RANK_STRUCT.pack(
            slots[0],
            slots[1],
            slots[2],
            slots[3],
            hazard_veto_mask,
            timestamp_ms,
        )
        crc = zlib.crc32(body) & 0xFFFFFFFF
        return cls(
            ranked_ids=(slots[0], slots[1], slots[2], slots[3]),
            hazard_veto_mask=hazard_veto_mask,
            timestamp_ms=timestamp_ms,
            crc32=crc,
        )

    def to_bytes(self) -> bytes:
        return _VLM_WIRE_STRUCT.pack(
            self.ranked_ids[0],
            self.ranked_ids[1],
            self.ranked_ids[2],
            self.ranked_ids[3],
            self.hazard_veto_mask,
            self.timestamp_ms,
            self.crc32,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> VlmRankMsg:
        if len(payload) != _VLM_WIRE_STRUCT.size:
            raise ValueError(
                f"VlmRankMsg requires exactly {_VLM_WIRE_STRUCT.size} bytes, got {len(payload)}"
            )
        r0, r1, r2, r3, veto_mask, ts_ms, crc = _VLM_WIRE_STRUCT.unpack(payload)
        return cls(
            ranked_ids=(r0, r1, r2, r3),
            hazard_veto_mask=veto_mask,
            timestamp_ms=ts_ms,
            crc32=crc,
        )


@dataclass(frozen=True)
class ForcedLandDecision:
    """Auditable output of the 6-stage `ForcedLandVlmGate`."""

    selected_flz_id: int | None
    source: str
    reason: str
    latched: bool
    h_commit_m: float
    veto_cleared_deadlock: bool = False


@dataclass(frozen=True)
class LeoOmmNisResult:
    """Validation result for passive LEO-Doppler SoOP OMM & Chi-Squared NIS Guard."""

    nav_source: str
    norad_cat_id: int
    post_rollover_norad_id: bool
    leo_inhibited: bool
    doppler_sigma_mps: float
    leo_nis_action: str
    leo_weight: float
    pitot_fault: bool
    pitot_action: str
    airdata_optical_synthesis_blocked: bool


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def best_glide_params(polar: FixedWingPolar) -> BestGlideParams:
    """Compute exact best-glide speed, lift coefficient, L/D max, and sink rate."""
    ld_max = 1.0 / (2.0 * math.sqrt(polar.cd0 * polar.induced_k))
    cl_bg = math.sqrt(polar.cd0 / polar.induced_k)
    v_bg = math.sqrt((2.0 * polar.weight_n) / (polar.rho_kg_m3 * polar.wing_area_m2 * cl_bg))
    v_sink_bg = v_bg / ld_max
    return BestGlideParams(
        ld_max=ld_max,
        cl_bg=cl_bg,
        v_bg_mps=v_bg,
        v_sink_bg_mps=v_sink_bg,
        glide_angle_rad=math.atan(1.0 / ld_max),
    )


def banked_turn_schedule(
    polar: FixedWingPolar,
    phi_rad: float,
    *,
    enforce_nominal_cap: bool = True,
) -> BankedTurnSchedule:
    """Compute coordinated banked-glide speed, sink penalty, turn radius, and 360° altitude loss."""
    if not math.isfinite(phi_rad) or phi_rad <= 0.0 or phi_rad >= (0.5 * math.pi):
        raise ValueError("phi_rad must be strictly in (0, pi/2)")
    phi_deg = math.degrees(phi_rad)
    if enforce_nominal_cap and phi_deg > (MAX_NOMINAL_BANK_DEG + 1e-9):
        raise ValueError(
            f"Bank angle {phi_deg:.2f} deg exceeds nominal DR-18 cap of {MAX_NOMINAL_BANK_DEG:.1f} deg"
        )
    bg = best_glide_params(polar)
    cos_phi = math.cos(phi_rad)
    sin_phi = math.sin(phi_rad)
    sec_phi = 1.0 / cos_phi
    v_opt = bg.v_bg_mps * math.sqrt(sec_phi)
    v_sink = bg.v_sink_bg_mps * (sec_phi**1.5)
    v_stall_banked = polar.v_stall_mps * math.sqrt(sec_phi)
    turn_radius_m = (bg.v_bg_mps**2) / (G_STANDARD_MPS2 * sin_phi)
    delta_h_360_m = (2.0 * math.pi * (bg.v_bg_mps**2)) / (
        G_STANDARD_MPS2 * bg.ld_max * sin_phi * cos_phi
    )
    return BankedTurnSchedule(
        phi_rad=phi_rad,
        phi_deg=phi_deg,
        load_factor_g=sec_phi,
        v_opt_mps=v_opt,
        v_sink_mps=v_sink,
        v_stall_banked_mps=v_stall_banked,
        turn_radius_m=turn_radius_m,
        delta_h_360_m=delta_h_360_m,
        delta_h_180_m=0.5 * delta_h_360_m,
    )


def maccready_speed_to_fly(
    polar: FixedWingPolar,
    w_par: float,
    w_perp: float,
    w_down: float,
) -> MacCreadySolution:
    """Maximize ``(sqrt(V^2 - w_perp^2) + w_par) / (V_s(V) + w_down)`` over ``[1.15 * V_stall, V_max]``.

    Sign convention:
      - ``w_par``: wind along track (``> 0`` tailwind, ``< 0`` headwind).
      - ``w_perp``: crosswind perpendicular to track (m/s).
      - ``w_down``: ambient downdraft (``> 0`` sink, ``< 0`` updraft).
    """
    v_lo = max(1.15 * polar.v_stall_mps, abs(w_perp) + 1e-3)
    v_hi = polar.v_max_mps
    if v_lo > v_hi:
        raise ValueError(
            f"Crosswind |w_perp|={abs(w_perp):.2f} m/s exceeds controllable airspeed envelope"
        )

    def _objective(v_val: float) -> float:
        v_fwd = math.sqrt(max(v_val * v_val - w_perp * w_perp, 0.0)) + w_par
        v_sink_tot = polar.sink_rate_mps(v_val) + w_down
        if v_sink_tot <= 1e-6:
            return v_fwd / 1e-6
        return v_fwd / v_sink_tot

    grid_n = 256
    step = (v_hi - v_lo) / grid_n
    best_idx = 0
    best_val = -math.inf
    for i in range(grid_n + 1):
        v_cand = v_lo + i * step
        val = _objective(v_cand)
        if val > best_val:
            best_val = val
            best_idx = i

    a = max(v_lo, v_lo + (best_idx - 1) * step)
    b = min(v_hi, v_lo + (best_idx + 1) * step)
    inv_phi = (math.sqrt(5.0) - 1.0) * 0.5
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = _objective(c)
    fd = _objective(d)
    for _ in range(48):
        if (b - a) < 1e-9:
            break
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - inv_phi * (b - a)
            fc = _objective(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + inv_phi * (b - a)
            fd = _objective(d)

    v_opt = 0.5 * (a + b)
    v_ground = math.sqrt(max(v_opt * v_opt - w_perp * w_perp, 0.0)) + w_par
    v_sink_total = polar.sink_rate_mps(v_opt) + w_down
    eff_ld = v_ground / v_sink_total if v_sink_total > 1e-6 else math.inf
    crab_ratio = max(-1.0, min(1.0, w_perp / v_opt))
    return MacCreadySolution(
        v_opt_mps=v_opt,
        v_ground_mps=v_ground,
        v_sink_total_mps=v_sink_total,
        effective_glide_ratio=eff_ld,
        crab_angle_rad=math.asin(crab_ratio),
    )


def abl_wind_profile(
    w_ref: float,
    z_m: float,
    z_ref_m: float,
    alpha: float = 0.14,
) -> float:
    """Power-law atmospheric boundary-layer wind shear profile `W(z) = W_ref * (z / z_ref)^alpha`."""
    if z_ref_m <= 0.0:
        raise ValueError("z_ref_m must be strictly positive")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    z_eff = max(float(z_m), 1.0)
    return float(w_ref) * ((z_eff / float(z_ref_m)) ** float(alpha))


def evaluate_flz_candidates(
    polar: FixedWingPolar,
    state: AircraftGlideState,
    candidates: Sequence[FlzCandidate],
    *,
    emergency_tier: bool = False,
) -> FlzEvaluationResult:
    """Run 5-gate 3DEP FLZ filter, into-wind alignment, and 3D wind-corrected trochoidal reachability."""
    slope_limit_deg = 8.0 if emergency_tier else 6.0
    sched_30 = banked_turn_schedule(polar, math.radians(30.0), enforce_nominal_cap=True)
    w_n, w_e, w_d = state.wind_ned_mps
    wind_horiz_mps = math.hypot(w_n, w_e)
    into_wind_hdg_rad = math.atan2(-w_e, -w_n)

    reachable_mask = 0
    energy_margins: dict[int, float] = {}
    rejections: dict[int, tuple[str, ...]] = {}
    passing_rows: list[tuple[float, float, float, int]] = []

    for cand in candidates:
        reasons: list[str] = []
        if not _is_exact_int(cand.flz_id) or not (0 <= cand.flz_id <= 15):
            raise ValueError(f"flz_id must be integer in 0..=15, got {cand.flz_id!r}")

        # Gate 1: Terrain slope
        if cand.slope_deg > slope_limit_deg or cand.cross_slope_deg > 5.0:
            reasons.append("GATE1_SLOPE_EXCEEDED")
        # Gate 2: Surface roughness
        if cand.rms_roughness_m > 0.30:
            reasons.append("GATE2_ROUGHNESS_EXCEEDED")
        # Gate 3: nDSM canopy / clutter height
        if cand.ndsm_m > 0.50:
            reasons.append("GATE3_NDSM_EXCEEDED")
        # Gate 4: Exclusion mask overlap & lateral setback
        if cand.exclusion_overlap_frac > 0.0 or cand.lateral_setback_m < 25.0:
            reasons.append("GATE4_EXCLUSION_OR_SETBACK")
        # Gate 5: 10:1 obstacle-free corridor & strip dimensions
        if (
            not cand.approach_corridor_clear_10_to_1
            or cand.strip_length_m < polar.min_strip_len_m
            or cand.strip_width_m < polar.min_strip_width_m
        ):
            reasons.append("GATE5_CORRIDOR_OR_STRIP_SIZE")
        # Into-wind final alignment gate when ||W_NE|| >= 3.0 m/s
        if wind_horiz_mps >= 3.0:
            align_err_deg = abs(
                math.degrees(_wrap_pi(cand.final_approach_hdg_rad - into_wind_hdg_rad))
            )
            if align_err_deg > 45.0:
                reasons.append("GATE_INTO_WIND_ALIGNMENT")

        # 3D wind-corrected trochoidal reachability
        dh_available = cand.down_m - state.down_m
        if dh_available <= 0.0:
            reasons.append("UNREACHABLE_BELOW_FLZ")
        else:
            dn_raw = cand.north_m - state.north_m
            de_raw = cand.east_m - state.east_m
            bearing_raw = math.atan2(de_raw, dn_raw) if (dn_raw != 0.0 or de_raw != 0.0) else cand.final_approach_hdg_rad
            dpsi_entry = abs(_wrap_pi(bearing_raw - state.heading_rad))
            dpsi_final = abs(_wrap_pi(cand.final_approach_hdg_rad - bearing_raw))
            total_turn_rad = dpsi_entry + dpsi_final
            turn_frac = total_turn_rad / (2.0 * math.pi)
            t_turn_s = (total_turn_rad * sched_30.turn_radius_m) / sched_30.v_opt_mps
            dh_turn = turn_frac * sched_30.delta_h_360_m

            dn_eff = dn_raw - w_n * t_turn_s
            de_eff = de_raw - w_e * t_turn_s
            straight_dist_m = math.hypot(dn_eff, de_eff)
            if straight_dist_m > 1e-6:
                u_n = dn_eff / straight_dist_m
                u_e = de_eff / straight_dist_m
            else:
                u_n = math.cos(cand.final_approach_hdg_rad)
                u_e = math.sin(cand.final_approach_hdg_rad)
            w_par = w_n * u_n + w_e * u_e
            w_perp = -w_n * u_e + w_e * u_n

            try:
                mc = maccready_speed_to_fly(polar, w_par=w_par, w_perp=w_perp, w_down=w_d)
            except ValueError:
                reasons.append("UNREACHABLE_CROSSWIND_ENVELOPE")
            else:
                if mc.v_ground_mps <= 0.5:
                    reasons.append("UNREACHABLE_HEADWIND_PENETRATION")
                else:
                    dh_straight = (straight_dist_m / mc.v_ground_mps) * max(mc.v_sink_total_mps, 0.0)
                    dh_reserve = max(0.25 * dh_available, sched_30.delta_h_180_m + 30.0)
                    dh_required = cand.obstacle_clearance_m + dh_turn + dh_straight + dh_reserve
                    margin_m = dh_available - dh_required
                    energy_margins[cand.flz_id] = margin_m
                    if margin_m < 0.0:
                        reasons.append("UNREACHABLE_ENERGY_MARGIN")

        if reasons:
            rejections[cand.flz_id] = tuple(reasons)
        else:
            reachable_mask |= 1 << cand.flz_id
            passing_rows.append(
                (
                    -energy_margins[cand.flz_id],
                    cand.slope_deg,
                    cand.rms_roughness_m,
                    cand.flz_id,
                )
            )

    passing_rows.sort()
    geom_ranked = tuple(row[3] for row in passing_rows)
    return FlzEvaluationResult(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        energy_margins_m=energy_margins,
        rejection_reasons=rejections,
    )


class ForcedLandVlmGate:
    """6-stage deterministic safety gate wrapping advisory VLM FLZ ranking (`DR-18` §C)."""

    def __init__(self, polar: FixedWingPolar, *, watchdog_ms: int = 500) -> None:
        if watchdog_ms <= 0:
            raise ValueError("watchdog_ms must be strictly positive")
        self._polar = polar
        self._watchdog_ms = int(watchdog_ms)
        sched_30 = banked_turn_schedule(polar, math.radians(30.0), enforce_nominal_cap=True)
        self._h_commit_m: float = sched_30.delta_h_180_m + 50.0
        self._latched_flz_id: int | None = None

    @property
    def h_commit_m(self) -> float:
        return self._h_commit_m

    @property
    def latched_flz_id(self) -> int | None:
        return self._latched_flz_id

    def reset_latch(self) -> None:
        self._latched_flz_id = None

    def _finalize(
        self,
        selected_id: int,
        *,
        agl_m: float,
        source: str,
        reason: str,
        veto_cleared_deadlock: bool = False,
    ) -> ForcedLandDecision:
        if agl_m <= self._h_commit_m:
            self._latched_flz_id = selected_id
            return ForcedLandDecision(
                selected_flz_id=selected_id,
                source=source,
                reason=reason,
                latched=True,
                h_commit_m=self._h_commit_m,
                veto_cleared_deadlock=veto_cleared_deadlock,
            )
        return ForcedLandDecision(
            selected_flz_id=selected_id,
            source=source,
            reason=reason,
            latched=False,
            h_commit_m=self._h_commit_m,
            veto_cleared_deadlock=veto_cleared_deadlock,
        )

    def select_flz(
        self,
        *,
        reachable_flz_mask: int,
        geom_ranked_ids: Sequence[int],
        now_ms: int,
        agl_m: float,
        optical_odd_valid: bool,
        vlm_msg: VlmRankMsg | bytes | None,
    ) -> ForcedLandDecision:
        """Execute all 6 deterministic verification stages with instant geometric fallback."""
        if self._latched_flz_id is not None:
            return ForcedLandDecision(
                selected_flz_id=self._latched_flz_id,
                source="COMMIT_LATCH",
                reason="STAGE5_COMMIT_ALTITUDE_LATCH_LOCKED",
                latched=True,
                h_commit_m=self._h_commit_m,
                veto_cleared_deadlock=False,
            )

        if reachable_flz_mask == 0 or len(geom_ranked_ids) == 0:
            return ForcedLandDecision(
                selected_flz_id=None,
                source="NONE",
                reason="NO_REACHABLE_FLZ",
                latched=False,
                h_commit_m=self._h_commit_m,
                veto_cleared_deadlock=False,
            )

        fallback_id = int(geom_ranked_ids[0])

        if vlm_msg is None:
            return self._finalize(
                fallback_id,
                agl_m=agl_m,
                source="GEOM_FALLBACK",
                reason="STAGE1_NO_VLM_MESSAGE",
            )

        # Stage 1: CRC32 + strictly integer-only unique schema (0..=15 or 0xFF)
        if isinstance(vlm_msg, bytes):
            try:
                parsed = VlmRankMsg.from_bytes(vlm_msg)
            except ValueError:
                return self._finalize(
                    fallback_id,
                    agl_m=agl_m,
                    source="GEOM_FALLBACK",
                    reason="STAGE1_WIRE_LENGTH_REJECT",
                )
        elif isinstance(vlm_msg, VlmRankMsg):
            parsed = vlm_msg
        else:
            return self._finalize(
                fallback_id,
                agl_m=agl_m,
                source="GEOM_FALLBACK",
                reason="STAGE1_INVALID_MSG_TYPE",
            )

        if not _validate_vlm_schema_and_crc(parsed):
            return self._finalize(
                fallback_id,
                agl_m=agl_m,
                source="GEOM_FALLBACK",
                reason="STAGE1_CRC_OR_SCHEMA_REJECT",
            )

        # Stage 2: 500 ms Watchdog freshness check
        age_ms = now_ms - parsed.timestamp_ms
        if age_ms < 0 or age_ms > self._watchdog_ms:
            return self._finalize(
                fallback_id,
                agl_m=agl_m,
                source="GEOM_FALLBACK",
                reason="STAGE2_WATCHDOG_TIMEOUT",
            )

        # Stage 3: Optical-ODD abstain gate
        if not optical_odd_valid:
            return self._finalize(
                fallback_id,
                agl_m=agl_m,
                source="GEOM_FALLBACK",
                reason="STAGE3_OPTICAL_ODD_ABSTAIN",
            )

        # Stage 4: Strict u16 bitmask containment for every active VLM ID
        active_ids = [flz_id for flz_id in parsed.ranked_ids if flz_id != _UNUSED_FLZ_SLOT]
        for flz_id in active_ids:
            if (reachable_flz_mask & (1 << flz_id)) == 0:
                return self._finalize(
                    fallback_id,
                    agl_m=agl_m,
                    source="GEOM_FALLBACK",
                    reason="STAGE4_BITMASK_CONTAINMENT_REJECT",
                )

        # Stage 5: Anti-all-veto deadlock guard
        surviving_mask = reachable_flz_mask & (~parsed.hazard_veto_mask & 0xFFFF)
        if surviving_mask == 0:
            return self._finalize(
                fallback_id,
                agl_m=agl_m,
                source="GEOM_FALLBACK",
                reason="STAGE5_ANTI_ALL_VETO_DEADLOCK_GUARD",
                veto_cleared_deadlock=True,
            )

        # Stage 6: Monotonic selection from VLM ranking, else surviving geometric fallback
        for flz_id in active_ids:
            if (surviving_mask & (1 << flz_id)) != 0:
                return self._finalize(
                    flz_id,
                    agl_m=agl_m,
                    source="VLM_ACCEPTED",
                    reason="STAGE6_VLM_RANK_SELECTED",
                )

        for flz_id in geom_ranked_ids:
            if (surviving_mask & (1 << flz_id)) != 0:
                return self._finalize(
                    int(flz_id),
                    agl_m=agl_m,
                    source="GEOM_FALLBACK",
                    reason="STAGE6_SURVIVING_GEOM_FALLBACK",
                )

        return self._finalize(
            fallback_id,
            agl_m=agl_m,
            source="GEOM_FALLBACK",
            reason="STAGE6_DEFAULT_GEOM_FALLBACK",
        )


def _validate_vlm_schema_and_crc(msg: VlmRankMsg) -> bool:
    if len(msg.ranked_ids) != 4:
        return False
    if not _is_exact_int(msg.hazard_veto_mask) or not (0 <= msg.hazard_veto_mask <= 0xFFFF):
        return False
    if not _is_exact_int(msg.timestamp_ms) or msg.timestamp_ms < 0:
        return False
    if not _is_exact_int(msg.crc32) or not (0 <= msg.crc32 <= 0xFFFFFFFF):
        return False

    seen: set[int] = set()
    for slot in msg.ranked_ids:
        if not _is_exact_int(slot):
            return False
        if slot == _UNUSED_FLZ_SLOT:
            continue
        if not (0 <= slot <= 15):
            return False
        if slot in seen:
            return False
        seen.add(slot)

    body = _VLM_RANK_STRUCT.pack(
        msg.ranked_ids[0],
        msg.ranked_ids[1],
        msg.ranked_ids[2],
        msg.ranked_ids[3],
        msg.hazard_veto_mask,
        msg.timestamp_ms,
    )
    expected_crc = zlib.crc32(body) & 0xFFFFFFFF
    return msg.crc32 == expected_crc


def validate_leo_omm_and_nis(
    *,
    nav_source: str,
    norad_cat_id: int,
    tau_omm_h: float,
    nis_leo_dof1: float,
    nis_pitot_dof3: float,
    stan_active: bool = False,
) -> LeoOmmNisResult:
    """Validate passive LEO-Doppler SoOP OMM catalog ID, ephemeris age, and Chi-Squared NIS gates.

    Enforces DR-18 fail-closed civil physics & regulatory locks:
      - Hard-rejects ``MagNav`` and ``active DME``.
      - Supports 6-digit / 9-digit ``NORAD_CAT_ID >= 100000`` (post-July-2026 CCSDS OMM rollover).
      - Inhibits LEO-Doppler when ``tau_omm_h > 24.0`` without ``stan_active``.
      - Applies ``DOF=1`` Huber downweighting at ``6.63`` and hard rejection at ``10.83``.
      - Applies ``DOF=3`` Pitot hard rejection at ``16.27`` with permanent ``airdata_optical_synthesis_blocked=True``.
    """
    src_norm = nav_source.strip().lower()
    if src_norm in _BANNED_OVERCLOUD_NAV_SOURCES:
        raise ForcedLandPolicyError(
            f"Navigation source '{nav_source}' is hard-rejected under DR-18 civil passive policy"
        )
    if src_norm not in _ALLOWED_PASSIVE_NAV_SOURCES:
        raise ForcedLandPolicyError(
            f"Unrecognized navigation source '{nav_source}'; fail-closed rejection"
        )

    if not _is_exact_int(norad_cat_id) or norad_cat_id <= 0:
        raise ValueError("norad_cat_id must be a strictly positive integer")
    if tau_omm_h < 0.0 or not math.isfinite(tau_omm_h):
        raise ValueError("tau_omm_h must be finite and non-negative")
    if nis_leo_dof1 < 0.0 or not math.isfinite(nis_leo_dof1):
        raise ValueError("nis_leo_dof1 must be finite and non-negative")
    if nis_pitot_dof3 < 0.0 or not math.isfinite(nis_pitot_dof3):
        raise ValueError("nis_pitot_dof3 must be finite and non-negative")

    if stan_active:
        leo_inhibited = False
        doppler_sigma = 0.20
    elif tau_omm_h < 6.0:
        leo_inhibited = False
        doppler_sigma = 0.35
    elif tau_omm_h <= 24.0:
        leo_inhibited = False
        doppler_sigma = 0.80
    else:
        leo_inhibited = True
        doppler_sigma = math.inf

    if leo_inhibited:
        leo_nis_action = "INHIBITED_STALE_OMM"
        leo_weight = 0.0
    elif nis_leo_dof1 <= _DOF1_HUBER_THRESHOLD:
        leo_nis_action = "ACCEPT"
        leo_weight = 1.0
    elif nis_leo_dof1 <= _DOF1_HARD_REJECT_THRESHOLD:
        leo_nis_action = "HUBER_DOWNWEIGHT"
        leo_weight = math.sqrt(_DOF1_HUBER_THRESHOLD / nis_leo_dof1)
    else:
        leo_nis_action = "HARD_REJECT"
        leo_weight = 0.0

    pitot_fault = nis_pitot_dof3 > _DOF3_PITOT_HARD_REJECT_THRESHOLD
    pitot_action = "HARD_REJECT_ALPHA_IAS_FREEZE" if pitot_fault else "ACCEPT"

    return LeoOmmNisResult(
        nav_source=src_norm,
        norad_cat_id=norad_cat_id,
        post_rollover_norad_id=norad_cat_id >= 100000,
        leo_inhibited=leo_inhibited,
        doppler_sigma_mps=doppler_sigma,
        leo_nis_action=leo_nis_action,
        leo_weight=leo_weight,
        pitot_fault=pitot_fault,
        pitot_action=pitot_action,
        airdata_optical_synthesis_blocked=True,
    )
