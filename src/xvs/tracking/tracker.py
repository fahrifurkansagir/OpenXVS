"""Angles-first multi-camera tracker and track management (WP-05 §2 & Batch B7 Steps 2-3).

Conforms strictly to docs/xvs/WP05_TRACKING.md §2, GEMINI_G-B3.md §1 items 2 and 3, and BATCH_B7_ARGON.md §§2-3.
Key architectural properties:
- Detections converted to Earth-fixed LOS unit vectors in NED via camera -> body -> NED;
- Ring camera overlap duplicate merging (< 3 * sigma_ang);
- State vector per track: [az, az_rate, el, el_rate, s, s_dot];
- Nearly-constant-rate linear Kalman filter with discrete process noise;
- Gated Global Nearest Neighbour (GNN) association with chi^2_2 gate at 99.7%;
- Track management: M-of-N initiation (3 of 5) and 4.0 s non-cooperative track memory timeout;
- Seam persistence: tracks live in Earth-fixed LOS space, preserving track ID across camera seams;
- G1 extensions:
  * Second-moment scale s_mom and sliding-window Theil-Sen/Huber log-size TTC estimation;
  * 6-DoF WGS-84 ground-clutter rejection at M-of-N track initiation using ownship navigation ECEF state
    and INS-estimated attitude (zero ground-truth access).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats

from xvs.geo import wgs84
from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.ring import TrafficRing
from xvs.tracking.ground_rejection import (
    GroundObservation,
    compute_ground_consistency_residual,
)
from xvs.tracking.measurements import DetectionRecord, rebuild_los_ned
from xvs.tracking.ttc import TTCEstimate, estimate_sliding_window_log_size_ttc

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0.json"


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    CLUTTER_HOLD = "clutter_hold"
    DELETED = "deleted"


@dataclass
class Track:
    """State and history of a single target track in Earth-fixed LOS space."""
    track_id: str
    t_init: float
    status: TrackStatus = TrackStatus.TENTATIVE

    # State vector: [az, az_rate, el, el_rate, s, s_dot]
    x: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    P: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64))

    hits: int = 1
    total_frames: int = 1
    hit_history: list[bool] = field(default_factory=lambda: [True])
    time_since_update: float = 0.0
    confirmed_time: float | None = None
    last_associated_truth_id: str | None = None
    associated_truth_ids: list[str] = field(default_factory=list)

    # History records for analysis
    history_t: list[float] = field(default_factory=list)
    history_az_el: list[tuple[float, float]] = field(default_factory=list)
    history_rates: list[tuple[float, float]] = field(default_factory=list)
    history_s: list[tuple[float, float]] = field(default_factory=list)
    history_cams: list[str] = field(default_factory=list)

    # G1 second-moment scale, sliding-window TTC, and 6-DoF ground-rejection records
    history_s_mom: list[float] = field(default_factory=list)
    ground_obs_history: list[GroundObservation] = field(default_factory=list)
    ttc_estimate: TTCEstimate | None = None
    ground_residual_px: float | None = None
    is_below_horizon: bool = False
    ground_clutter_rejected: bool = False

    @property
    def az(self) -> float:
        return float(self.x[0])

    @property
    def az_rate(self) -> float:
        return float(self.x[1])

    @property
    def el(self) -> float:
        return float(self.x[2])

    @property
    def el_rate(self) -> float:
        return float(self.x[3])

    @property
    def s(self) -> float:
        return float(self.x[4])

    @property
    def s_dot(self) -> float:
        return float(self.x[5])

    @property
    def s_mom(self) -> float:
        """Latest second-moment angular scale s_mom (rad) or filter scale s."""
        if self.history_s_mom:
            return max(float(self.x[4]), float(self.history_s_mom[-1]))
        return float(self.x[4])

    @property
    def omega_los(self) -> float:
        """Total LOS angular rate magnitude |omega_LOS| = sqrt(az_rate^2 + el_rate^2)."""
        return math.hypot(self.az_rate, self.el_rate)

    @property
    def sigma_omega(self) -> float:
        """Filter standard deviation of LOS angular rate magnitude."""
        return math.sqrt(max(0.0, float(self.P[1, 1]) + float(self.P[3, 3])))

    @property
    def sigma_s(self) -> float:
        """Filter standard deviation of angular scale s."""
        return math.sqrt(max(0.0, float(self.P[4, 4])))

    @property
    def tau_s(self) -> float | None:
        """Time-to-contact from angular expansion.

        In G1 (when sliding-window TTC is active), returns ttc_estimate.tau_s_hat_s if valid;
        otherwise falls back to legacy s / s_dot while s > 0 and s_dot > 3 * sigma_s_dot (WP-05 §3).
        """
        if self.ttc_estimate is not None:
            return self.ttc_estimate.tau_s_hat_s if self.ttc_estimate.valid else None
        if self.s > 0.0 and self.s_dot > (3.0 * self.sigma_s_dot) and self.s_dot > 1e-9:
            return self.s / self.s_dot
        return None

    @property
    def tau_s_legacy(self) -> float | None:
        """Legacy G0 single-filter time-to-contact tau_s = s / s_dot."""
        if self.s > 0.0 and self.s_dot > (3.0 * self.sigma_s_dot) and self.s_dot > 1e-9:
            return self.s / self.s_dot
        return None

    @property
    def sigma_s_dot(self) -> float:
        """Standard deviation of expansion rate estimate."""
        return math.sqrt(max(0.0, float(self.P[5, 5])))


@dataclass
class MergedDetection:
    """Pre-processed and merged detection ready for association."""
    u: float
    v: float
    camera: str
    s_px: float
    score: float
    los_ned: np.ndarray
    az: float
    el: float
    s_rad: float
    is_false_alarm: bool
    truth_id: str | None
    s_mom_px: float = 0.0
    s_mom_rad: float = 0.0


class MultiCameraTracker:
    """Angles-first multi-camera tracker across all 8 ring cameras."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        ring: TrafficRing | None = None,
    ) -> None:
        if config is not None:
            self.config = dict(config)
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        self.ring = ring if ring is not None else TrafficRing()

        self.rate_hz: float = float(self.config["rate_hz"])
        self.dt: float = 1.0 / self.rate_hz
        self.sigma_ang_rad: float = float(self.config["sigma_ang_rad"])
        self.sigma_px: float = float(self.config["sigma_px"])
        self.ins_sigma_rad: float = math.radians(float(self.config["ins_sigma_deg"]))
        self.duplicate_gate_sigma: float = float(self.config["duplicate_gate_sigma"])
        self.association_gate_prob: float = float(self.config["association_gate_prob"])
        self.initiation_m: int = int(self.config["initiation_m"])
        self.initiation_n: int = int(self.config["initiation_n"])
        self.deletion_timeout_s: float = float(self.config["deletion_timeout_s"])

        # G1 second-moment scale and sliding-window log-size TTC settings
        self.use_second_moment_scale: bool = bool(self.config.get("use_second_moment_scale", False))
        self.sigma_s_mom_px: float = float(self.config.get("sigma_s_mom_px", 0.08))  # default-ok: G1 sub-pixel moment noise
        self.use_sliding_window_ttc: bool = bool(self.config.get("use_sliding_window_ttc", False))
        self.ttc_window_s: float = float(self.config.get("ttc_window_s", 4.0))  # default-ok: G1 sliding window duration
        self.ttc_min_hits: int = int(self.config.get("ttc_min_hits", 5))  # default-ok: G1 min hits in window
        self.ttc_min_span_s: float = float(self.config.get("ttc_min_span_s", 1.0))  # default-ok: G1 min window span
        self.ttc_min_pair_dt_s: float = float(self.config.get("ttc_min_pair_dt_s", 0.5))  # default-ok: G1 min pair separation
        self.ttc_k_eta: float = float(self.config.get("ttc_k_eta", 1.5))  # default-ok: G1 expansion rate gate multiplier
        self.ttc_eta_min_close: float = float(self.config.get("ttc_eta_min_close", 0.005))  # default-ok: G1 min closing rate
        self.ttc_use_huber: bool = bool(self.config.get("ttc_use_huber", True))

        # G1 6-DoF WGS-84 ground-clutter rejection settings
        self.enable_ground_clutter_rejection: bool = bool(self.config.get("enable_ground_clutter_rejection", False))
        self.r_gnd_thr_px: float = float(self.config.get("r_gnd_thr_px", 2.5))  # default-ok: G1 ground residual threshold
        self.eta_gnd_thr_inv_s: float = float(self.config.get("eta_gnd_thr_inv_s", 0.015))  # default-ok: G1 ground expansion threshold
        self.clutter_hold_timeout_s: float = float(self.config.get("clutter_hold_timeout_s", 1.2))  # default-ok: G1 clutter hold timeout

        # Process noise spectral densities
        self.q_omega: float = float(self.config["q_omega"])
        self.q_s: float = float(self.config["q_s"])

        # Measurement variances
        sigma_centroid_ang = self.sigma_px / self.ring.fx
        self.var_ang = (self.sigma_ang_rad**2 + sigma_centroid_ang**2 + self.ins_sigma_rad**2)
        eff_sigma_s_px = self.sigma_s_mom_px if self.use_second_moment_scale else self.sigma_px
        self.var_s = max(1e-12, (eff_sigma_s_px / self.ring.fx)**2)

        # Measurement covariance R (3x3): [az, el, s]
        self.R = np.diag([self.var_ang, self.var_ang, self.var_s])

        # Association chi^2 gate threshold (2 degrees of freedom: az, el)
        self.chi2_gate_2d = float(stats.chi2.ppf(self.association_gate_prob, df=2))

        # Filter matrices for time step dt
        self.F, self.Q = self._build_transition_and_process_noise(self.dt)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0  # az
        self.H[1, 2] = 1.0  # el
        self.H[2, 4] = 1.0  # s

        self.tracks: list[Track] = []
        self.active_tracks: list[Track] = []
        self._next_track_idx: int = 1
        self.ground_clutter_rejected_count: int = 0

    def _build_transition_and_process_noise(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Build F and Q for nearly-constant-rate decoupled 2D blocks."""
        f_blk = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)

        def q_blk(q: float) -> np.ndarray:
            return q * np.array([
                [dt**3 / 3.0, dt**2 / 2.0],
                [dt**2 / 2.0, dt],
            ], dtype=np.float64)

        F = np.zeros((6, 6), dtype=np.float64)
        Q = np.zeros((6, 6), dtype=np.float64)

        for i, q in enumerate([self.q_omega, self.q_omega, self.q_s]):
            idx = 2 * i
            F[idx:idx+2, idx:idx+2] = f_blk
            Q[idx:idx+2, idx:idx+2] = q_blk(q)

        return F, Q

    def reset(self) -> None:
        """Reset tracker state and active tracks."""
        self.tracks.clear()
        self.active_tracks.clear()
        self._next_track_idx = 1
        self.ground_clutter_rejected_count = 0

    def _build_ground_obs(
        self,
        t_s: float,
        det: MergedDetection,
        p_own_ecef: np.ndarray,
        r_ecef_ned: np.ndarray,
        r_ned_body_ins: np.ndarray,
    ) -> GroundObservation:
        primary_cam_id = det.camera.split("+")[0]
        cam = self.ring.get_camera(primary_cam_id)
        r_ecef_body_ins = r_ecef_ned @ r_ned_body_ins
        p_cam_ecef_ins = p_own_ecef + r_ecef_body_ins @ cam.lever_arm_m
        return GroundObservation(
            t_s=t_s,
            cam_id=primary_cam_id,
            u_px=det.u,
            v_px=det.v,
            s_mom_rad=det.s_mom_rad,
            los_ned_ins=det.los_ned.copy(),
            p_cam_ecef_ins=p_cam_ecef_ins,
            r_ecef_ned=r_ecef_ned.copy(),
        )

    def merge_duplicates(
        self,
        raw_detections: Sequence[DetectionRecord],
        r_ned_body_ins: np.ndarray,
    ) -> list[MergedDetection]:
        """Merge detections from overlapping ring cameras if angular separation < 3 * sigma_ang."""
        if not raw_detections:
            return []

        preprocessed: list[MergedDetection] = []
        for det in raw_detections:
            los = rebuild_los_ned(det.u, det.v, det.camera, self.ring, r_ned_body_ins)
            r_xy = math.hypot(float(los[0]), float(los[1]))
            az = math.atan2(float(los[1]), float(los[0]))
            el = math.atan2(-float(los[2]), max(1e-12, r_xy))
            s_mom_px = float(det.s_mom_px) if det.s_mom_px is not None else float(det.s_px)
            s_mom_rad = s_mom_px / self.ring.fx
            s_px_eff = s_mom_px if self.use_second_moment_scale else float(det.s_px)
            s_rad = s_px_eff / self.ring.fx

            preprocessed.append(MergedDetection(
                u=det.u,
                v=det.v,
                camera=det.camera,
                s_px=det.s_px,
                score=det.score,
                los_ned=los,
                az=az,
                el=el,
                s_rad=s_rad,
                is_false_alarm=det.is_false_alarm,
                truth_id=det.truth_id,
                s_mom_px=s_mom_px,
                s_mom_rad=s_mom_rad,
            ))

        dup_threshold = self.duplicate_gate_sigma * self.sigma_ang_rad
        cos_threshold = math.cos(dup_threshold)
        merged: list[MergedDetection] = []
        n_prep = len(preprocessed)
        used = [False] * n_prep

        los_list = [(float(p.los_ned[0]), float(p.los_ned[1]), float(p.los_ned[2])) for p in preprocessed]

        for i in range(n_prep):
            if used[i]:
                continue
            curr = preprocessed[i]
            group = [curr]
            used[i] = True
            a0, a1, a2 = los_list[i]

            for j in range(i + 1, n_prep):
                if used[j]:
                    continue
                b0, b1, b2 = los_list[j]
                dot_val = a0 * b0 + a1 * b1 + a2 * b2
                if dot_val < cos_threshold:
                    continue

                c0 = a1 * b2 - a2 * b1
                c1 = a2 * b0 - a0 * b2
                c2 = a0 * b1 - a1 * b0
                cross_norm = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
                sep_rad = math.atan2(cross_norm, dot_val)

                if sep_rad < dup_threshold:
                    group.append(preprocessed[j])
                    used[j] = True

            if len(group) == 1:
                merged.append(curr)
            else:
                avg_los = np.mean([g.los_ned for g in group], axis=0)
                avg_los /= np.linalg.norm(avg_los)
                r_xy = math.hypot(float(avg_los[0]), float(avg_los[1]))
                avg_az = math.atan2(float(avg_los[1]), float(avg_los[0]))
                avg_el = math.atan2(-float(avg_los[2]), max(1e-12, r_xy))
                avg_s_px = float(np.mean([g.s_px for g in group]))
                avg_s_mom_px = float(np.mean([g.s_mom_px for g in group]))
                avg_s_mom_rad = avg_s_mom_px / self.ring.fx
                avg_s_rad = avg_s_mom_rad if self.use_second_moment_scale else (avg_s_px / self.ring.fx)
                max_score = float(max(g.score for g in group))
                cams = "+".join(g.camera for g in group)
                truth_id = next((g.truth_id for g in group if g.truth_id is not None), None)
                is_fa = all(g.is_false_alarm for g in group)

                merged.append(MergedDetection(
                    u=group[0].u,
                    v=group[0].v,
                    camera=cams,
                    s_px=avg_s_px,
                    score=max_score,
                    los_ned=avg_los,
                    az=avg_az,
                    el=avg_el,
                    s_rad=avg_s_rad,
                    is_false_alarm=is_fa,
                    truth_id=truth_id,
                    s_mom_px=avg_s_mom_px,
                    s_mom_rad=avg_s_mom_rad,
                ))

        return merged

    def predict_tracks(self, dt: float) -> None:
        """Propagate state and covariance of all active tracks by time dt."""
        F, Q = self._build_transition_and_process_noise(dt)

        if not self.active_tracks and self.tracks:
            self.active_tracks = [trk for trk in self.tracks if trk.status != TrackStatus.DELETED]
        tracks_to_predict = self.active_tracks if self.active_tracks else self.tracks
        for trk in tracks_to_predict:
            if trk.status == TrackStatus.DELETED:
                continue

            x_pred = F @ trk.x
            x_pred[0] = wrap_to_pi(float(x_pred[0]))

            P_pred = F @ trk.P @ F.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)

            trk.x = x_pred
            trk.P = P_pred
            trk.time_since_update += dt

    def update_track(
        self,
        trk: Track,
        det: MergedDetection,
        t_s: float,
        p_own_ecef: np.ndarray | None = None,
        r_ecef_ned: np.ndarray | None = None,
        r_ned_body_ins: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Update track with an associated measurement. Returns (nees_proxy, nis)."""
        z = np.array([det.az, det.el, det.s_rad], dtype=np.float64)

        hx = np.array([trk.x[0], trk.x[2], trk.x[4]], dtype=np.float64)
        y = z - hx
        y[0] = wrap_to_pi(float(y[0]))

        S = self.H @ trk.P @ self.H.T + self.R
        S = 0.5 * (S + S.T)

        S_inv = np.linalg.inv(S)
        K = trk.P @ self.H.T @ S_inv

        x_up = trk.x + K @ y
        x_up[0] = wrap_to_pi(float(x_up[0]))

        I_KH = np.eye(6, dtype=np.float64) - K @ self.H
        P_up = I_KH @ trk.P @ I_KH.T + K @ self.R @ K.T
        P_up = 0.5 * (P_up + P_up.T)

        trk.x = x_up
        trk.P = P_up

        trk.hits += 1
        trk.total_frames += 1
        trk.hit_history.append(True)
        trk.time_since_update = 0.0

        if det.truth_id is not None:
            trk.last_associated_truth_id = det.truth_id
            trk.associated_truth_ids.append(det.truth_id)

        trk.history_t.append(t_s)
        trk.history_az_el.append((trk.az, trk.el))
        trk.history_rates.append((trk.az_rate, trk.el_rate))
        trk.history_s.append((trk.s, trk.s_dot))
        trk.history_cams.append(det.camera)
        trk.history_s_mom.append(det.s_mom_rad)

        if p_own_ecef is not None and r_ecef_ned is not None and r_ned_body_ins is not None:
            trk.ground_obs_history.append(
                self._build_ground_obs(t_s, det, p_own_ecef, r_ecef_ned, r_ned_body_ins)
            )

        if self.use_sliding_window_ttc:
            trk.ttc_estimate = estimate_sliding_window_log_size_ttc(
                times_s=trk.history_t,
                s_mom_rad=trk.history_s_mom,
                window_s=self.ttc_window_s,
                min_hits=self.ttc_min_hits,
                min_span_s=self.ttc_min_span_s,
                min_pair_dt_s=self.ttc_min_pair_dt_s,
                k_eta=self.ttc_k_eta,
                eta_min_close=self.ttc_eta_min_close,
                use_huber=self.ttc_use_huber,
            )

        if trk.status == TrackStatus.TENTATIVE:
            if sum(trk.hit_history[-self.initiation_n:]) >= self.initiation_m:
                if self.enable_ground_clutter_rejection and len(trk.ground_obs_history) >= 2:
                    gnd_res = compute_ground_consistency_residual(
                        trk.ground_obs_history,
                        fx=self.ring.fx,
                        max_window_obs=10,
                    )
                    trk.ground_residual_px = gnd_res.r_gnd_px
                    trk.is_below_horizon = gnd_res.is_below_horizon

                    has_strong_expansion = (
                        trk.ttc_estimate is not None
                        and trk.ttc_estimate.valid
                        and trk.ttc_estimate.eta_hat_inv_s > self.eta_gnd_thr_inv_s
                        and trk.ttc_estimate.eta_hat_inv_s > 3.5 * trk.ttc_estimate.sigma_eta_inv_s
                    )
                    if (
                        gnd_res.is_below_horizon
                        and gnd_res.r_gnd_px <= self.r_gnd_thr_px
                        and not has_strong_expansion
                    ):
                        trk.status = TrackStatus.CLUTTER_HOLD
                        trk.ground_clutter_rejected = True
                        self.ground_clutter_rejected_count += 1
                    else:
                        trk.status = TrackStatus.CONFIRMED
                        trk.confirmed_time = t_s
                else:
                    trk.status = TrackStatus.CONFIRMED
                    trk.confirmed_time = t_s
        elif trk.status == TrackStatus.CLUTTER_HOLD:
            # Re-evaluate held below-horizon track: promote to CONFIRMED if it diverges from stationary ground
            # or exhibits sustained closing scale expansion
            if len(trk.ground_obs_history) >= 3:
                gnd_res = compute_ground_consistency_residual(
                    trk.ground_obs_history,
                    fx=self.ring.fx,
                    max_window_obs=0,
                )
                trk.ground_residual_px = gnd_res.r_gnd_px
                trk.is_below_horizon = gnd_res.is_below_horizon

                has_sustained_expansion = False
                if (
                    trk.ttc_estimate is not None
                    and trk.ttc_estimate.valid
                    and len(trk.history_s_mom) >= 6
                    and trk.ttc_estimate.eta_hat_inv_s > self.eta_gnd_thr_inv_s
                    and trk.ttc_estimate.eta_hat_inv_s > 4.0 * trk.ttc_estimate.sigma_eta_inv_s
                ):
                    s_early = float(np.median(trk.history_s_mom[:3]))
                    s_late = float(np.median(trk.history_s_mom[-3:]))
                    if s_early > 0.0 and (s_late / s_early) >= 1.15:
                        has_sustained_expansion = True

                if (
                    (not gnd_res.is_below_horizon)
                    or (gnd_res.r_gnd_px > 2.0 * self.r_gnd_thr_px)
                    or has_sustained_expansion
                ):
                    trk.status = TrackStatus.CONFIRMED
                    trk.confirmed_time = t_s
                    if trk.ground_clutter_rejected:
                        trk.ground_clutter_rejected = False
                        self.ground_clutter_rejected_count = max(0, self.ground_clutter_rejected_count - 1)

        nis = float(y @ S_inv @ y)
        return 0.0, nis

    def initiate_track(
        self,
        det: MergedDetection,
        t_s: float,
        p_own_ecef: np.ndarray | None = None,
        r_ecef_ned: np.ndarray | None = None,
        r_ned_body_ins: np.ndarray | None = None,
    ) -> Track:
        """Create a new tentative track from an unassociated detection."""
        tid = f"TRK_{self._next_track_idx:04d}"
        self._next_track_idx += 1

        x = np.zeros(6, dtype=np.float64)
        x[0] = det.az
        x[1] = 0.0
        x[2] = det.el
        x[3] = 0.0
        x[4] = det.s_rad
        x[5] = 0.0

        var_rate = (math.radians(5.0))**2
        var_s_dot = (self.var_s / (self.dt**2))
        P = np.diag([
            self.var_ang, var_rate,
            self.var_ang, var_rate,
            self.var_s, var_s_dot,
        ])

        gnd_obs: list[GroundObservation] = []
        if p_own_ecef is not None and r_ecef_ned is not None and r_ned_body_ins is not None:
            gnd_obs.append(self._build_ground_obs(t_s, det, p_own_ecef, r_ecef_ned, r_ned_body_ins))

        trk = Track(
            track_id=tid,
            t_init=t_s,
            status=TrackStatus.TENTATIVE,
            x=x,
            P=P,
            hits=1,
            total_frames=1,
            hit_history=[True],
            time_since_update=0.0,
            confirmed_time=None,
            last_associated_truth_id=det.truth_id,
            associated_truth_ids=[det.truth_id] if det.truth_id else [],
            history_t=[t_s],
            history_az_el=[(det.az, det.el)],
            history_rates=[(0.0, 0.0)],
            history_s=[(det.s_rad, 0.0)],
            history_cams=[det.camera],
            history_s_mom=[det.s_mom_rad],
            ground_obs_history=gnd_obs,
        )
        return trk

    def step(
        self,
        t_s: float,
        raw_detections: Sequence[DetectionRecord],
        r_ned_body_ins: np.ndarray,
        p_own_ecef: np.ndarray | None = None,
        r_ecef_ned: np.ndarray | None = None,
    ) -> list[Track]:
        """Execute one tracking cycle: predict, associate, update, and manage tracks."""
        if self.enable_ground_clutter_rejection and (p_own_ecef is None or r_ecef_ned is None):
            st_o = t1_cruise_state(t_s)
            p_own_ecef = st_o.p_ecef_m
            r_ecef_ned = wgs84.r_ecef_ned(st_o.lat_deg, st_o.lon_deg)

        # 1. Predict active tracks
        self.predict_tracks(self.dt)

        # 2. Merge duplicate detections in camera overlap zones
        detections = self.merge_duplicates(raw_detections, r_ned_body_ins)

        # 3. Gated GNN Association
        if not self.active_tracks and self.tracks:
            self.active_tracks = [trk for trk in self.tracks if trk.status != TrackStatus.DELETED]
        active_tracks = self.active_tracks
        n_tracks = len(active_tracks)
        n_dets = len(detections)

        cost_matrix = np.full((n_tracks, n_dets), 1e9, dtype=np.float64)

        if n_tracks > 0 and n_dets > 0:
            det_azs = np.fromiter((d.az for d in detections), dtype=np.float64, count=n_dets)
            det_els = np.fromiter((d.el for d in detections), dtype=np.float64, count=n_dets)

            two_pi = 2.0 * math.pi
            pi = math.pi
            gate = self.chi2_gate_2d

            for i, trk in enumerate(active_tracks):
                s00_cov = trk.P[0, 0] + self.var_ang
                s01_cov = trk.P[0, 2]
                s11_cov = trk.P[2, 2] + self.var_ang
                det_S = s00_cov * s11_cov - s01_cov * s01_cov

                if det_S > 1e-15:
                    inv_det = 1.0 / det_S
                    s00_inv = s11_cov * inv_det
                    s01_inv = -s01_cov * inv_det
                    s11_inv = s00_cov * inv_det

                    d_az = (det_azs - trk.az + pi) % two_pi - pi
                    d_el = det_els - trk.el
                    d2 = s00_inv * (d_az * d_az) + (2.0 * s01_inv) * (d_az * d_el) + s11_inv * (d_el * d_el)

                    mask = d2 <= gate
                    cost_matrix[i, mask] = d2[mask]

        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        if n_tracks > 0 and n_dets > 0:
            row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.chi2_gate_2d:
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)
                    self.update_track(
                        active_tracks[r],
                        detections[c],
                        t_s,
                        p_own_ecef=p_own_ecef,
                        r_ecef_ned=r_ecef_ned,
                        r_ned_body_ins=r_ned_body_ins,
                    )

        # 4. Handle unmatched tracks
        for i, trk in enumerate(active_tracks):
            if i not in matched_track_indices:
                trk.total_frames += 1
                trk.hit_history.append(False)

                if trk.status == TrackStatus.CONFIRMED:
                    if trk.time_since_update > self.deletion_timeout_s:
                        trk.status = TrackStatus.DELETED
                elif trk.status == TrackStatus.CLUTTER_HOLD:
                    if trk.time_since_update > self.clutter_hold_timeout_s:
                        trk.status = TrackStatus.DELETED
                elif trk.status == TrackStatus.TENTATIVE:
                    window = trk.hit_history[-self.initiation_n:]
                    if (len(trk.hit_history) >= self.initiation_n and sum(window) < self.initiation_m) or (trk.time_since_update > 1.0):
                        trk.status = TrackStatus.DELETED

        # 5. Handle unmatched detections -> initiate tentative tracks
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                new_trk = self.initiate_track(
                    det,
                    t_s,
                    p_own_ecef=p_own_ecef,
                    r_ecef_ned=r_ecef_ned,
                    r_ned_body_ins=r_ned_body_ins,
                )
                self.tracks.append(new_trk)
                self.active_tracks.append(new_trk)

        self.active_tracks = [trk for trk in self.active_tracks if trk.status != TrackStatus.DELETED]
        return [trk for trk in self.tracks if trk.status == TrackStatus.CONFIRMED]
