"""Sampled intruder tracks (WP-02 G3b and later): an intruder given as time-stamped ECEF positions and velocities
instead of the parametric straight-line construction of §2.5.

`TrackTrajectory` exposes the same interface as `encounter.EncounterTrajectory` (`params`, `position_at`,
`velocity_at`, `attitude_at`), so the GT (`gt.compute_intruder_analytic_gt`), CPA and event functions and the G3a
renderer accept it unchanged. Between samples the position is a cubic Hermite interpolation of the sampled
positions and velocities (exact for a straight constant-velocity segment); the attitude follows §2.5: yaw along
the NED ground track, pitch = flight-path angle + alpha_trim, roll 0. Outside [t_first, t_last] the track is
undefined (`covers(t)` is False) and every accessor raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xvs.geo import rotations, wgs84


@dataclass
class TrackParams:
    """Encounter metadata of a sampled track (the fields the GT and the renderer read)."""
    id: str
    class_id: str
    model_id: str
    alpha_trim_deg: float = 0.0
    weight: float = 1.0
    stratum: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    # Requested CPA of a parametric encounter; unknown for a sampled track (NaN, written as null). encounter's
    # evaluate_actual_cpa reads them to report the achieved-minus-requested differences.
    t_cpa: float = float("nan")
    hmd_m: float = float("nan")
    vmd_m: float = float("nan")


class TrackTrajectory:
    def __init__(self, params: TrackParams, t_s: np.ndarray, p_ecef_m: np.ndarray, v_ecef_mps: np.ndarray) -> None:
        t = np.asarray(t_s, dtype=np.float64)
        p = np.asarray(p_ecef_m, dtype=np.float64)
        v = np.asarray(v_ecef_mps, dtype=np.float64)
        if t.ndim != 1 or len(t) < 2 or p.shape != (len(t), 3) or v.shape != (len(t), 3):
            raise ValueError("track needs >= 2 samples: t (N,), p (N, 3), v (N, 3)")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("track times must be strictly increasing")
        self.params = params
        self.t = t
        self.p = p
        self.v = v

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "TrackTrajectory":
        tr = rec["track"]
        params = TrackParams(id=rec["id"], class_id=rec["class_id"], model_id=rec["model_id"],
                             alpha_trim_deg=float(rec.get("alpha_trim_deg", 0.0)), weight=float(rec.get("weight", 1.0)),  # default-ok: optional fallback default
                             stratum=dict(rec.get("stratum", {})), source=dict(rec.get("source", {})))
        return cls(params, np.array(tr["t_s"]), np.array(tr["p_ecef_m"]), np.array(tr["v_ecef_mps"]))

    @property
    def t_first(self) -> float:
        return float(self.t[0])

    @property
    def t_last(self) -> float:
        return float(self.t[-1])

    def covers(self, t: float) -> bool:
        return self.t_first <= t <= self.t_last

    def _segment(self, t: float) -> tuple[int, float, float]:
        if not self.covers(t):
            raise ValueError(f"track {self.params.id}: t = {t} outside [{self.t_first}, {self.t_last}]")
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        i = min(max(i, 0), len(self.t) - 2)
        h = float(self.t[i + 1] - self.t[i])
        return i, h, (t - float(self.t[i])) / h

    def position_at(self, t: float) -> np.ndarray:
        i, h, s = self._segment(t)
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * self.p[i] + h10 * h * self.v[i] + h01 * self.p[i + 1] + h11 * h * self.v[i + 1]

    def velocity_at(self, t: float) -> np.ndarray:
        i, h, s = self._segment(t)
        d00 = (6 * s**2 - 6 * s) / h
        d10 = 3 * s**2 - 4 * s + 1
        d01 = (-6 * s**2 + 6 * s) / h
        d11 = 3 * s**2 - 2 * s
        return d00 * self.p[i] + d10 * self.v[i] + d01 * self.p[i + 1] + d11 * self.v[i + 1]

    def attitude_at(self, t: float) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Coordinated, zero sideslip (§2.5), from the interpolated velocity."""
        p_ecef = self.position_at(t)
        lat_deg, lon_deg, _ = wgs84.ecef_to_geodetic(*p_ecef)
        n_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
        v_ned = n_mat.T @ self.velocity_at(t)
        gs = math.hypot(float(v_ned[0]), float(v_ned[1]))
        yaw_rad = math.atan2(float(v_ned[1]), float(v_ned[0]))
        pitch_rad = math.atan2(-float(v_ned[2]), gs) + math.radians(self.params.alpha_trim_deg)
        r_ecef_body = n_mat @ rotations.euler_to_r(yaw_rad, pitch_rad, 0.0)
        return r_ecef_body, (math.degrees(yaw_rad), math.degrees(pitch_rad), 0.0)
