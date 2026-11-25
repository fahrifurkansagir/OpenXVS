"""Encounter generation, exact ECEF kinematics, CPA, and stratified sampling (WP-02 §2.5).

Conforms strictly to docs/xvs/WP02_TRAFFIC.md §2.5 and §3 G1.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable
import numpy as np

from xvs.geo import wgs84, rotations
from xvs.traffic.models import CLASS_SPECS


@dataclass
class EncounterParams:
    id: str
    class_id: str
    model_id: str
    t_cpa: float
    hmd_m: float
    side: str  # "left" or "right"
    vmd_m: float  # signed; intruder above = positive
    approach_deg: float  # in (-180, 180]
    gs_mps: float
    vs_mps: float
    alpha_trim_deg: float = 0.0
    weight: float = 1.0
    stratum: dict[str, str] = field(default_factory=dict)
    seed: int = 0
    redraws: int = 0


@dataclass
class EncounterTrajectory:
    params: EncounterParams
    p_cpa_ecef: np.ndarray  # (3,)
    v_ecef: np.ndarray      # (3,)

    def position_at(self, t: float) -> np.ndarray:
        """Intruder position in ECEF (m) at time t."""
        return self.p_cpa_ecef + self.v_ecef * (t - self.params.t_cpa)

    def velocity_at(self, t: float) -> np.ndarray:
        """Intruder velocity in ECEF (m/s) at time t (constant in ECEF)."""
        return self.v_ecef

    def attitude_at(self, t: float) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Intruder attitude at time t (coordinated, zero sideslip, spec §2.5).

        Returns: (R_ecef_body, (yaw_deg, pitch_deg, roll_deg)).
        """
        p_ecef = self.position_at(t)
        lat_deg, lon_deg, _ = wgs84.ecef_to_geodetic(*p_ecef)
        n_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
        v_ned = n_mat.T @ self.v_ecef

        v_n = float(v_ned[0])
        v_e = float(v_ned[1])
        v_d = float(v_ned[2])
        gs = math.hypot(v_n, v_e)
        vs = -v_d

        yaw_rad = math.atan2(v_e, v_n)
        gamma_rad = math.atan2(vs, gs)
        pitch_rad = gamma_rad + math.radians(self.params.alpha_trim_deg)
        roll_rad = 0.0

        r_ned_body = rotations.euler_to_r(yaw_rad, pitch_rad, roll_rad)
        r_ecef_body = n_mat @ r_ned_body
        ypr_deg = (math.degrees(yaw_rad), math.degrees(pitch_rad), 0.0)
        return r_ecef_body, ypr_deg


class TestOwnshipO1:
    """Test ownship O1: Straight constant-velocity ECEF line (spec §2.5).

    - start P0 = (37.30°, -121.00°, 1500 m)
    - P(t) = P0 + 100 m/s * t * e_hat, where e_hat is local east at P0
    - constant attitude: R_ecef_body = R_ecef_ned(P0) * R_ned_body(psi=90°, theta=2°, phi=0°)
    """

    def __init__(self) -> None:
        self.lat0_deg = 37.30
        self.lon0_deg = -121.00
        self.h0_m = 1500.0
        self.p0_ecef = wgs84.geodetic_to_ecef(self.lat0_deg, self.lon0_deg, self.h0_m)
        self.n0_mat = wgs84.r_ecef_ned(self.lat0_deg, self.lon0_deg)
        self.e_hat = self.n0_mat[:, 1]
        self.speed_mps = 100.0
        self.v_ecef = self.speed_mps * self.e_hat

        r_ned_body = rotations.euler_to_r(math.radians(90.0), math.radians(2.0), 0.0)
        self.r_ecef_body = self.n0_mat @ r_ned_body
        self.ypr_deg = (90.0, 2.0, 0.0)

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
        """Returns (p_ecef, v_ecef, r_ecef_body, ypr_deg)."""
        p_ecef = self.p0_ecef + self.v_ecef * t
        return p_ecef, self.v_ecef, self.r_ecef_body, self.ypr_deg


def construct_encounter(ownship: Any, params: EncounterParams) -> EncounterTrajectory:
    """Exact ECEF encounter construction at t_cpa (spec §2.5)."""
    # 1. At t_cpa take ownship position P_o and velocity V_o (ECEF), and N = R_ecef_ned at P_o
    if isinstance(ownship, TestOwnshipO1):
        p_o, v_o, _, _ = ownship.state_at(params.t_cpa)
        lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o)
        n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
    elif callable(ownship):
        state = ownship(params.t_cpa)
        p_o = state.p_ecef_m
        lat_o, lon_o = state.lat_deg, state.lon_deg
        n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
        v_o = n_mat @ state.v_ned_mps
    elif hasattr(ownship, "state_at"):
        p_o, v_o, _, _ = ownship.state_at(params.t_cpa)
        lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o)
        n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
    else:
        raise TypeError(f"Unsupported ownship trajectory representation: {type(ownship)}")

    # Ownship track at t_cpa in NED
    v_o_ned = n_mat.T @ v_o
    chi_o = math.atan2(v_o_ned[1], v_o_ned[0])

    # 2. Intruder velocity in NED: V_i^N = (gs*cos chi, gs*sin chi, -vs), chi = chi_o + approach_deg
    chi = chi_o + math.radians(params.approach_deg)
    v_i_ned = np.array([
        params.gs_mps * math.cos(chi),
        params.gs_mps * math.sin(chi),
        -params.vs_mps,
    ], dtype=np.float64)
    v_i = n_mat @ v_i_ned

    # 3. V_rel = V_i - V_o
    v_rel = v_i - v_o
    v_rel_ned = n_mat.T @ v_rel
    v_h_ned = np.array([v_rel_ned[0], v_rel_ned[1], 0.0], dtype=np.float64)
    v_h_norm = float(np.linalg.norm(v_h_ned[:2]))
    if v_h_norm < 1e-9:
        raise ValueError(f"Horizontal relative speed is undefined (|V_h| = {v_h_norm:.3e} m/s < 1e-9)")

    # d_hat = local down in ECEF
    d_hat = n_mat[:, 2]

    # h_hat = s * (d_hat x V_h) / |d_hat x V_h|
    # In NED: d_hat_ned = [0, 0, 1]
    # cross(d_hat_ned, v_h_ned) = [-v_h_ned[1], v_h_ned[0], 0]
    cross_ned = np.array([-v_h_ned[1], v_h_ned[0], 0.0], dtype=np.float64)
    cross_norm = float(np.linalg.norm(cross_ned))
    s = +1.0 if params.side == "right" else -1.0
    h_hat_ned = s * (cross_ned / cross_norm)
    h_hat = n_mat @ h_hat_ned

    # 4. P_i(t_cpa) = P_o(t_cpa) + hmd * h_hat + vmd * (-d_hat)
    p_i_cpa = p_o + params.hmd_m * h_hat - params.vmd_m * d_hat

    return EncounterTrajectory(params=params, p_cpa_ecef=p_i_cpa, v_ecef=v_i)


def evaluate_actual_cpa(
    ownship: Any,
    encounter: EncounterTrajectory,
    t_start: float = 0.0,
    t_end: float = 60.0,
    dt: float = 0.01,
) -> dict[str, float]:
    """Dense evaluation of achieved CPA (HMD, VMD, t_cpa) against ownship trajectory."""
    n_steps = int(math.ceil((t_end - t_start) / dt)) + 1
    times = [t_start + i * dt for i in range(n_steps)]

    d_h_vals = []
    v_sep_vals = []

    for t in times:
        p_i = encounter.position_at(t)
        if isinstance(ownship, TestOwnshipO1):
            p_o, _, _, _ = ownship.state_at(t)
            lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o)
            n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
        elif callable(ownship):
            st = ownship(t)
            p_o = st.p_ecef_m
            n_mat = wgs84.r_ecef_ned(st.lat_deg, st.lon_deg)
        elif hasattr(ownship, "state_at"):
            p_o, _, _, _ = ownship.state_at(t)
            lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o)
            n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
        else:
            raise TypeError("Unsupported ownship")

        r_ecef = p_i - p_o
        r_ned = n_mat.T @ r_ecef
        dh = math.hypot(float(r_ned[0]), float(r_ned[1]))
        vsep = -float(r_ned[2])
        d_h_vals.append(dh)
        v_sep_vals.append(vsep)

    min_idx = int(np.argmin(d_h_vals))

    # Refine with parabolic fit around minimum if not at boundary
    if 0 < min_idx < len(times) - 1:
        y0 = d_h_vals[min_idx - 1]
        y1 = d_h_vals[min_idx]
        y2 = d_h_vals[min_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            delta = max(-1.0, min(1.0, delta))
            t_achieved = times[min_idx] + delta * dt
            # Recompute at exact sub-step
            p_i = encounter.position_at(t_achieved)
            if isinstance(ownship, TestOwnshipO1):
                p_o, _, _, _ = ownship.state_at(t_achieved)
                lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o)
                n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
            elif callable(ownship):
                st = ownship(t_achieved)
                p_o = st.p_ecef_m
                n_mat = wgs84.r_ecef_ned(st.lat_deg, st.lon_deg)
            else:
                p_o, _, _, _ = ownship.state_at(t_achieved)
                lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o)
                n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
            r_ned = n_mat.T @ (p_i - p_o)
            hmd_achieved = math.hypot(float(r_ned[0]), float(r_ned[1]))
            vmd_achieved = -float(r_ned[2])
        else:
            t_achieved = times[min_idx]
            hmd_achieved = d_h_vals[min_idx]
            vmd_achieved = v_sep_vals[min_idx]
    else:
        t_achieved = times[min_idx]
        hmd_achieved = d_h_vals[min_idx]
        vmd_achieved = v_sep_vals[min_idx]

    return {
        "t_cpa_s": t_achieved,
        "hmd_m": hmd_achieved,
        "vmd_m": vmd_achieved,
        "diff_t_cpa_s": t_achieved - encounter.params.t_cpa,
        "diff_hmd_m": hmd_achieved - encounter.params.hmd_m,
        "diff_vmd_m": vmd_achieved - encounter.params.vmd_m,
    }


def evaluate_cpa_o1(params: EncounterParams, o1: TestOwnshipO1) -> tuple[float, float, float]:
    """Exact CPA evaluation for O1 in the local NED frame at P_o(t_cpa) (spec §2.5)."""
    enc = construct_encounter(o1, params)
    # Relative position in fixed N0 frame at t_cpa
    p_o_cpa, v_o, _, _ = o1.state_at(params.t_cpa)
    r_cpa = enc.p_cpa_ecef - p_o_cpa
    v_rel = enc.v_ecef - v_o

    lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o_cpa)
    n0 = wgs84.r_ecef_ned(lat_o, lon_o)
    r_cpa_ned = n0.T @ r_cpa
    v_rel_ned = n0.T @ v_rel

    r_h = r_cpa_ned[:2]
    v_h = v_rel_ned[:2]

    # Time of min horizontal distance: d( |r_h + v_h * dt|^2 )/dt = 0
    # => 2 * (r_h . v_h) + 2 * |v_h|^2 * dt = 0
    # => dt = - (r_h . v_h) / |v_h|^2
    v_h_sq = float(np.dot(v_h, v_h))
    dt_star = -float(np.dot(r_h, v_h)) / v_h_sq if v_h_sq > 0 else 0.0

    t_achieved = params.t_cpa + dt_star
    r_star_ned = r_cpa_ned + v_rel_ned * dt_star
    hmd_achieved = float(np.linalg.norm(r_star_ned[:2]))
    vmd_achieved = -float(r_star_ned[2])

    return t_achieved, hmd_achieved, vmd_achieved


def evaluate_events(
    ownship: Any,
    encounter: EncounterTrajectory,
    t_start: float = 0.0,
    t_end: float = 60.0,
    dt: float = 0.01,
) -> dict[str, float | None]:
    """Evaluate first time of NMAC and LoWC events (spec §2.5, ASTM F3442-23).

    NMAC: horizontal < 152.4 m (500 ft) and vertical < 30.48 m (100 ft).
    LoWC: horizontal < 609.6 m (2000 ft) and vertical < 76.2 m (250 ft).
    """
    n_steps = int(math.ceil((t_end - t_start) / dt)) + 1
    times = [t_start + i * dt for i in range(n_steps)]

    first_nmac_t = None
    first_lowc_t = None

    for t in times:
        p_i = encounter.position_at(t)
        if isinstance(ownship, TestOwnshipO1):
            p_o, _, _, _ = ownship.state_at(t)
            lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o)
            n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
        elif callable(ownship):
            st = ownship(t)
            p_o = st.p_ecef_m
            n_mat = wgs84.r_ecef_ned(st.lat_deg, st.lon_deg)
        elif hasattr(ownship, "state_at"):
            p_o, _, _, _ = ownship.state_at(t)
            lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o)
            n_mat = wgs84.r_ecef_ned(lat_o, lon_o)
        else:
            raise TypeError("Unsupported ownship")

        r_ned = n_mat.T @ (p_i - p_o)
        dh = math.hypot(float(r_ned[0]), float(r_ned[1]))
        dv = abs(float(r_ned[2]))

        if first_lowc_t is None and (dh < 609.6) and (dv < 76.2):
            first_lowc_t = t

        if first_nmac_t is None and (dh < 152.4) and (dv < 30.48):
            first_nmac_t = t

    return {
        "nmac_t": first_nmac_t,
        "lowc_t": first_lowc_t,
    }


# --- Sampler v1 (spec §2.5) ---

GEOMETRY_STRATA = [
    "head_on",
    "crossing_left",
    "crossing_right",
    "same_direction",
    "converging",
]

CLASS_STRATA = [
    "C1_UAS",
    "C2_SEP",
    "C3_TWIN",
    "C4_BIZJET",
    "C5_TRANSPORT",
    "C6_HELI",
]

VERTICAL_STRATA = [
    "above",
    "co_altitude",
    "below",
]

SPEED_RANGES_MPS = {
    "C1_UAS": {"gs": (10.0, 30.0), "vs": (-3.0, 3.0)},
    "C2_SEP": {"gs": (45.0, 75.0), "vs": (-8.0, 8.0)},
    "C3_TWIN": {"gs": (60.0, 110.0), "vs": (-8.0, 8.0)},
    "C4_BIZJET": {"gs": (110.0, 230.0), "vs": (-8.0, 8.0)},
    "C5_TRANSPORT": {"gs": (120.0, 250.0), "vs": (-8.0, 8.0)},
    "C6_HELI": {"gs": (30.0, 75.0), "vs": (-8.0, 8.0)},
}


class EncounterSamplerV1:
    """Stratified encounter sampler v1 generating 900 encounters (spec §2.5)."""

    def __init__(self, master_seed: int = 20260923, count_per_stratum: int = 10) -> None:
        self.master_seed = master_seed
        self.count_per_stratum = count_per_stratum
        self.version = "1.0"

    def _sample_approach_deg(self, geom_stratum: str, rng: np.random.Generator) -> float:
        if geom_stratum == "head_on":
            return float(rng.uniform(150.0, 210.0))
        elif geom_stratum == "crossing_left":
            return float(rng.uniform(60.0, 120.0))
        elif geom_stratum == "crossing_right":
            return float(rng.uniform(-120.0, -60.0))
        elif geom_stratum == "same_direction":
            return float(rng.uniform(-30.0, 30.0))
        elif geom_stratum == "converging":
            if rng.uniform(0.0, 1.0) < 0.5:
                return float(rng.uniform(30.0, 60.0))
            else:
                return float(rng.uniform(-60.0, -30.0))
        else:
            raise ValueError(f"Unknown geom stratum: {geom_stratum}")

    def _sample_vmd_m(self, vert_stratum: str, rng: np.random.Generator) -> float:
        if vert_stratum == "above":
            return float(rng.uniform(100.0, 300.0))
        elif vert_stratum == "co_altitude":
            return float(rng.uniform(-30.0, 30.0))
        elif vert_stratum == "below":
            return float(rng.uniform(-300.0, -100.0))
        else:
            raise ValueError(f"Unknown vert stratum: {vert_stratum}")

    def _sample_hmd_m(self, rng: np.random.Generator) -> float:
        # Half under 500 ft (152.4 m), half between 152.4 m and 1500 m
        if rng.uniform(0.0, 1.0) < 0.5:
            return float(rng.uniform(0.0, 152.4))
        else:
            return float(rng.uniform(152.4, 1500.0))

    def generate(self, ownship_fn: Callable[[float], Any]) -> list[EncounterParams]:
        """Generate 900 encounters (5 x 6 x 3 x 10) deterministically."""
        rng = np.random.default_rng(self.master_seed)
        encounters: list[EncounterParams] = []
        enc_idx = 1

        for g in GEOMETRY_STRATA:
            for c in CLASS_STRATA:
                sp_cfg = SPEED_RANGES_MPS[c]
                trim_deg = CLASS_SPECS[c]["alpha_trim_deg"]
                for v in VERTICAL_STRATA:
                    for _ in range(self.count_per_stratum):
                        redraws = 0
                        while True:
                            sub_seed = int(rng.integers(0, 2**31 - 1))
                            sub_rng = np.random.default_rng(sub_seed)

                            t_cpa = float(sub_rng.uniform(30.0, 60.0))
                            hmd = self._sample_hmd_m(sub_rng)
                            side = "right" if sub_rng.uniform(0.0, 1.0) < 0.5 else "left"
                            vmd = self._sample_vmd_m(v, sub_rng)
                            app_deg = self._sample_approach_deg(g, sub_rng)
                            if app_deg > 180.0:
                                app_deg -= 360.0

                            gs = float(sub_rng.uniform(sp_cfg["gs"][0], sp_cfg["gs"][1]))
                            vs = float(sub_rng.uniform(sp_cfg["vs"][0], sp_cfg["vs"][1]))

                            # Model variant 1-3
                            variant_seed = int(sub_rng.integers(1, 4))
                            model_id = f"{c}_{variant_seed:04d}"

                            # Check horizontal relative speed constraint (|V_h| >= 10 m/s)
                            st_o = ownship_fn(t_cpa)
                            n_mat = wgs84.r_ecef_ned(st_o.lat_deg, st_o.lon_deg)
                            v_o_ned = st_o.v_ned_mps
                            chi_o = math.atan2(v_o_ned[1], v_o_ned[0])
                            chi = chi_o + math.radians(app_deg)
                            v_i_ned = np.array([gs * math.cos(chi), gs * math.sin(chi), -vs])

                            v_rel_h = math.hypot(v_i_ned[0] - v_o_ned[0], v_i_ned[1] - v_o_ned[1])
                            if v_rel_h >= 10.0:
                                enc = EncounterParams(
                                    id=f"E{enc_idx:04d}",
                                    class_id=c,
                                    model_id=model_id,
                                    t_cpa=t_cpa,
                                    hmd_m=hmd,
                                    side=side,
                                    vmd_m=vmd,
                                    approach_deg=app_deg,
                                    gs_mps=gs,
                                    vs_mps=vs,
                                    alpha_trim_deg=trim_deg,
                                    weight=1.0,
                                    stratum={"geometry": g, "class": c, "vertical": v},
                                    seed=sub_seed,
                                    redraws=redraws,
                                )
                                encounters.append(enc)
                                enc_idx += 1
                                break
                            redraws += 1

        return encounters


def export_encounters_jsonl(encounters: list[EncounterParams], out_path: str | Path) -> str:
    """Export encounters to encounters.jsonl and return file SHA-256."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": identical bytes (and SHA-256) on Windows and Linux (G1 finding: text mode wrote CRLF).
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for e in encounters:
            f.write(json.dumps(asdict(e)) + "\n")

    return hashlib.sha256(p.read_bytes()).hexdigest()
