"""Exact TSO-C151d / RTCA DO-367 Class B Forward-Looking Terrain Avoidance (FLTA) evaluator.

Implements DR-03:116-117 & SP2_M2_MODELS_DESIGN.md §3.4 strictly in SI metres/seconds:
  - Alert inequality:
      h_DEM(x, y) + sigma_DEM > z_pred(s) - RTC(phase, vz) - VPL_t
  - Exact Class B SI RTC table (0.3048 m/ft):
      EN_ROUTE:  213.36 m (700 ft) level / 152.40 m (500 ft) descending
      TERMINAL:  106.68 m (350 ft) level /  91.44 m (300 ft) descending
      APPROACH:   45.72 m (150 ft) level /  30.48 m (100 ft) descending
      DEPARTURE:  30.48 m (100 ft) level /  30.48 m (100 ft) descending
  - Class B Approach Desensitization (DR-03:117):
      Inhibits nuisance FLTA alerts below 60.96 m (200 ft) AGL within 926.0 m (0.5 NM) of
      threshold OR below 38.10 m (125 ft) AGL within 1,852.0 m (1.0 NM) of threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Literal

from xvs.v2.wp01b.mipmap import MinMaxMipmapPyramid, sample_height_bilinear_m

FT_TO_M: float = 0.3048
NM_TO_M: float = 1852.0

# RTCA DO-367 descending threshold: 500 fpm = 2.54 m/s (positive NED Down velocity)
DESCENT_RATE_THRESHOLD_MPS: float = 500.0 * FT_TO_M / 60.0  # 2.54 m/s

# Class B Approach Desensitization envelope (DR-03:117)
CLASS_B_DESENS_INNER_AGL_M: float = 200.0 * FT_TO_M  # 60.96 m
CLASS_B_DESENS_INNER_DIST_M: float = 0.5 * NM_TO_M   # 926.0 m
CLASS_B_DESENS_OUTER_AGL_M: float = 125.0 * FT_TO_M  # 38.10 m
CLASS_B_DESENS_OUTER_DIST_M: float = 1.0 * NM_TO_M   # 1852.0 m

CAUTION_LOOKAHEAD_S: float = 60.0
WARNING_LOOKAHEAD_S: float = 30.0


class FlightPhase(str, Enum):
    EN_ROUTE = "EN_ROUTE"
    TERMINAL = "TERMINAL"
    APPROACH = "APPROACH"
    DEPARTURE = "DEPARTURE"


# Exact Class B Required Terrain Clearance (RTC) table in SI metres: (level_m, descending_m)
CLASS_B_RTC_SI_M: dict[FlightPhase, tuple[float, float]] = {
    FlightPhase.EN_ROUTE: (700.0 * FT_TO_M, 500.0 * FT_TO_M),     # (213.36, 152.40)
    FlightPhase.TERMINAL: (350.0 * FT_TO_M, 300.0 * FT_TO_M),     # (106.68,  91.44)
    FlightPhase.APPROACH: (150.0 * FT_TO_M, 100.0 * FT_TO_M),     # ( 45.72,  30.48)
    FlightPhase.DEPARTURE: (100.0 * FT_TO_M, 100.0 * FT_TO_M),    # ( 30.48,  30.48)
}


@dataclass(frozen=True)
class FltaResult:
    """TSO-C151d / DO-367 FLTA evaluation output."""

    alert_level: Literal["NONE", "CAUTION", "WARNING"]
    time_to_conflict_s: float | None
    min_clearance_margin_m: float
    rtc_m: float
    desensitized: bool
    conflict_north_m: float | None
    conflict_east_m: float | None
    conflict_dem_h_m: float | None


def required_terrain_clearance_m(phase: FlightPhase, vz_ned_mps: float) -> float:
    """Return exact TSO-C151d Class B RTC (m) for flight phase and NED vertical velocity `vz` (+Down)."""
    level_rtc, desc_rtc = CLASS_B_RTC_SI_M[phase]
    if vz_ned_mps >= DESCENT_RATE_THRESHOLD_MPS:
        return desc_rtc
    return level_rtc


def is_approach_desensitized(
    phase: FlightPhase,
    agl_m: float,
    dist_to_threshold_m: float | None,
) -> bool:
    """Class B Approach Desensitization (DR-03:117) suppressing final-approach nuisance alerts."""
    if phase != FlightPhase.APPROACH or dist_to_threshold_m is None:
        return False
    if not (math.isfinite(agl_m) and math.isfinite(dist_to_threshold_m)):
        return False
    if dist_to_threshold_m < 0.0:
        return False
    in_inner = agl_m < CLASS_B_DESENS_INNER_AGL_M and dist_to_threshold_m <= CLASS_B_DESENS_INNER_DIST_M
    in_outer = agl_m < CLASS_B_DESENS_OUTER_AGL_M and dist_to_threshold_m <= CLASS_B_DESENS_OUTER_DIST_M
    return bool(in_inner or in_outer)


def evaluate_taws_flta(
    pyramid: MinMaxMipmapPyramid,
    north_m: float,
    east_m: float,
    alt_wgs84_m: float,
    vn_mps: float,
    ve_mps: float,
    vz_ned_mps: float,
    phase: FlightPhase,
    *,
    sigma_dem_m: float = 3.0,
    vpl_m: float = 5.0,
    threshold_ned_m: tuple[float, float, float] | None = None,
    dt_step_s: float = 1.0,
    caution_lookahead_s: float = CAUTION_LOOKAHEAD_S,
    warning_lookahead_s: float = WARNING_LOOKAHEAD_S,
) -> FltaResult:
    """Evaluate exact TSO-C151d Class B FLTA along predicted flight path `[0, caution_lookahead_s]`.

    Checks `h_DEM(x, y) + sigma_DEM > z_pred(s) - RTC(phase, vz) - VPL_t` fail-closed.
    """
    rtc_m = required_terrain_clearance_m(phase, vz_ned_mps)

    # Check fail-closed on non-finite inputs
    inputs = (north_m, east_m, alt_wgs84_m, vn_mps, ve_mps, vz_ned_mps, sigma_dem_m, vpl_m)
    if not all(math.isfinite(x) for x in inputs):
        return FltaResult("WARNING", 0.0, -math.inf, rtc_m, False, north_m, east_m, math.nan)

    # Check ownship Class B Approach Desensitization relative to threshold
    own_dist_thr: float | None = None
    own_agl_m: float = alt_wgs84_m - sample_height_bilinear_m(pyramid, north_m, east_m)
    if threshold_ned_m is not None:
        own_dist_thr = math.hypot(north_m - threshold_ned_m[0], east_m - threshold_ned_m[1])
        own_agl_m = alt_wgs84_m - threshold_ned_m[2]
    if is_approach_desensitized(phase, own_agl_m, own_dist_thr):
        return FltaResult("NONE", None, max(0.0, own_agl_m), rtc_m, True, None, None, None)

    n_steps = max(1, int(math.ceil(caution_lookahead_s / dt_step_s)))
    min_margin = math.inf
    first_conflict_s: float | None = None
    first_conflict_pos: tuple[float, float, float] | None = None
    any_sample_desensitized = False

    for k in range(1, n_steps + 1):
        s = min(caution_lookahead_s, float(k) * dt_step_s)
        pn = north_m + vn_mps * s
        pe = east_m + ve_mps * s
        # NED vz > 0 is Down, so predicted WGS-84 altitude decreases by vz * s
        z_pred = alt_wgs84_m - vz_ned_mps * s

        # Skip samples outside DEM grid bounds cleanly
        if not (
            pyramid.north_min_m <= pn <= pyramid.north_max_m
            and pyramid.east_min_m <= pe <= pyramid.east_max_m
        ):
            continue

        # Class B Approach Desensitization along predicted final approach path near threshold
        if threshold_ned_m is not None and phase == FlightPhase.APPROACH:
            d_thr = math.hypot(pn - threshold_ned_m[0], pe - threshold_ned_m[1])
            agl_pred = z_pred - threshold_ned_m[2]
            # Stop lookahead once past runway threshold on nominal landing glide
            dot_to_thr = (threshold_ned_m[0] - pn) * vn_mps + (threshold_ned_m[1] - pe) * ve_mps
            if dot_to_thr < 0.0 and d_thr <= CLASS_B_DESENS_OUTER_DIST_M:
                any_sample_desensitized = True
                break
            if is_approach_desensitized(phase, agl_pred, d_thr):
                any_sample_desensitized = True
                continue

        h_dem = sample_height_bilinear_m(pyramid, pn, pe)
        # Exact TSO-C151d / DO-367 inequality margin:
        # positive = clear, negative = alert (`h_DEM + sigma_DEM > z_pred - RTC - VPL`)
        margin = (z_pred - rtc_m - vpl_m) - (h_dem + sigma_dem_m)
        if margin < min_margin:
            min_margin = margin
        if not (margin >= 0.0) and first_conflict_s is None:
            first_conflict_s = s
            first_conflict_pos = (pn, pe, h_dem)

    if first_conflict_s is None:
        return FltaResult(
            alert_level="NONE",
            time_to_conflict_s=None,
            min_clearance_margin_m=float(min_margin if math.isfinite(min_margin) else 0.0),
            rtc_m=rtc_m,
            desensitized=any_sample_desensitized,
            conflict_north_m=None,
            conflict_east_m=None,
            conflict_dem_h_m=None,
        )

    level: Literal["CAUTION", "WARNING"] = (
        "WARNING" if first_conflict_s <= warning_lookahead_s else "CAUTION"
    )
    assert first_conflict_pos is not None
    return FltaResult(
        alert_level=level,
        time_to_conflict_s=float(first_conflict_s),
        min_clearance_margin_m=float(min_margin),
        rtc_m=rtc_m,
        desensitized=False,
        conflict_north_m=float(first_conflict_pos[0]),
        conflict_east_m=float(first_conflict_pos[1]),
        conflict_dem_h_m=float(first_conflict_pos[2]),
    )
