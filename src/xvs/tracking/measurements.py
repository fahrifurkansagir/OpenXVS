"""Synthetic measurement generator for WP-05 G0 multi-camera tracking.

Conforms strictly to docs/xvs/WP05_TRACKING.md §1, GEMINI_G-B3.md §1, and D29 (Common Random Numbers).
Generates detections from ground truth with:
- exact pinhole projection and analytic vertex bounding box;
- surrogate detection probability P_d(s_px);
- centroid noise (sigma_px) and camera angular frame noise (sigma_ang);
- INS attitude Gauss-Markov drift (roll, pitch, yaw);
- Poisson false alarms with 3x terrain clutter below the analytic horizon on a dedicated clutter RNG stream
  (self.rng_clutter) so true-target measurements and INS drift are byte-identical across lambda (D29);
- independent or first-order Markov miss models.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np

from xvs.geo import wgs84, rotations
from xvs.rig.camera import PinholeCamera
from xvs.sim import horizon
from xvs.traffic.encounter import EncounterTrajectory, EncounterParams
from xvs.traffic.ring import TrafficRing
from xvs.traffic import models
from xvs.tracking.ttc import compute_vertex_second_moment_scale

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "tracking" / "g0.json"

# Cache for 3D model vertices to avoid rebuilding repeatedly
_MODEL_VERTICES_CACHE: dict[str, np.ndarray] = {}


def get_model_vertices(model_id: str) -> np.ndarray:
    """Retrieve concatenated 3D vertices (N, 3) for a traffic model variant."""
    if model_id in _MODEL_VERTICES_CACHE:
        return _MODEL_VERTICES_CACHE[model_id]

    parts = model_id.split("_")
    cls_id = f"{parts[0]}_{parts[1]}"
    variant_seed = int(parts[2])

    builder = getattr(models, f"build_{cls_id.lower()}")
    res = builder(seed=variant_seed)
    verts = np.concatenate([p.vertices for p in res.parts], axis=0).astype(np.float64)
    _MODEL_VERTICES_CACHE[model_id] = verts
    return verts


@dataclass
class DetectionRecord:
    """Per-camera detection record output by the detector (WP-05 §1)."""
    t: float
    camera: str
    u: float
    v: float
    s_px: float
    score: float
    is_false_alarm: bool
    truth_id: str | None = None
    s_mom_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rebuild_los_ned(
    u: float,
    v: float,
    cam_id: str,
    ring: TrafficRing,
    r_ned_body_ins: np.ndarray,
) -> np.ndarray:
    """Convert detection pixel to Earth-fixed LOS unit vector: camera -> body -> NED (WP-05 §2)."""
    cam = ring.get_camera(cam_id)
    ray_cam = ring.camera_model.unproject(u, v)
    ray_body = cam.r_body_cam @ ray_cam
    ray_ned = r_ned_body_ins @ ray_body
    norm = float(np.linalg.norm(ray_ned))
    return ray_ned / norm if norm > 1e-12 else ray_ned


class SyntheticMeasurementGenerator:
    """Generates synthetic multi-camera detections from exact encounter ground truth."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        seed: int = 42,
        ring: TrafficRing | None = None,
    ) -> None:
        if config is not None:
            self.config = dict(config)
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        self.ring = ring if ring is not None else TrafficRing()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        # D29 Common Random Numbers: separate RNG stream for false-alarm clutter
        self.rng_clutter = np.random.default_rng([self.seed, 20261020])

        self.rate_hz: float = float(self.config["rate_hz"])
        self.dt: float = 1.0 / self.rate_hz
        self.s50_px: float = float(self.config["s50_px"])
        self.w_px: float = float(self.config["w_px"])
        self.w_ln: float = float(self.config.get("w_ln", 0.5))  # default-ok: optional fallback default
        self.pd_mode: str = str(self.config.get("pd_mode", "linear_logistic"))
        self.pd_empirical_table: list[tuple[float, float]] | None = self.config.get("pd_empirical_table", None)
        if self.pd_empirical_table is not None:
            self.pd_empirical_table = sorted([(float(pt[0]), float(pt[1])) for pt in self.pd_empirical_table])
        self.sigma_px: float = float(self.config["sigma_px"])
        self.sigma_s_px: float = float(self.config.get("sigma_s_px", self.config["sigma_px"]))
        self.use_second_moment_scale: bool = bool(self.config.get("use_second_moment_scale", False))
        self.sigma_psf_px: float = float(self.config.get("sigma_psf_px", 0.55))  # default-ok: optical PSF default
        self.lambda_min_px: float = float(self.config.get("lambda_min_px", 0.10))  # default-ok: PSF deconvolution floor
        self.sigma_s_mom_px: float = float(self.config.get("sigma_s_mom_px", 0.08))  # default-ok: sub-pixel second-moment scale noise
        self.sigma_ang_rad: float = float(self.config["sigma_ang_rad"])
        self.ins_sigma_rad: float = math.radians(float(self.config["ins_sigma_deg"]))
        self.ins_tau_s: float = float(self.config["ins_tau_s"])
        self.fppi_lambda: float = float(self.config["fppi_lambda"])
        self.terrain_clutter_factor: float = float(self.config.get("terrain_clutter_factor", 3.0))  # default-ok: optional fallback default
        self.fa_size_min_px: float = float(self.config.get("fa_size_min_px", 1.0))  # default-ok: geometry dimension fallback
        self.fa_size_max_px: float = float(self.config.get("fa_size_max_px", 4.0))  # default-ok: geometry dimension fallback
        self.markov_misses: bool = bool(self.config.get("markov_misses", False))
        self.p_stay: float = float(self.config.get("p_stay", 0.8))  # default-ok: optional fallback default

        # Gauss-Markov INS attitude error state (roll, pitch, yaw)
        self.ins_beta = math.exp(-self.dt / self.ins_tau_s) if self.ins_tau_s > 0 else 0.0
        self.ins_err_rpy = self.rng.normal(0.0, self.ins_sigma_rad, size=3)

        # Markov miss state per intruder
        self.prev_detected: dict[str, bool] = {}

    def reset_state(self, seed: int | None = None) -> None:
        """Reset internal RNG and stateful processes (INS drift, Markov misses)."""
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.rng_clutter = np.random.default_rng([self.seed, 20261020])
        self.ins_err_rpy = self.rng.normal(0.0, self.ins_sigma_rad, size=3)
        self.prev_detected.clear()

    def step_ins(self, disable_noise: bool = False) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Advance INS Gauss-Markov attitude drift by one time step."""
        if disable_noise or self.ins_sigma_rad <= 0.0:
            self.ins_err_rpy = np.zeros(3, dtype=np.float64)
        else:
            w = self.rng.normal(0.0, self.ins_sigma_rad, size=3)
            self.ins_err_rpy = (
                self.ins_beta * self.ins_err_rpy + math.sqrt(max(0.0, 1.0 - self.ins_beta**2)) * w
            )
        roll_err = float(self.ins_err_rpy[0])
        pitch_err = float(self.ins_err_rpy[1])
        yaw_err = float(self.ins_err_rpy[2])
        return self.ins_err_rpy, (yaw_err, pitch_err, roll_err)

    def detection_probability(self, s_px: float) -> float:
        """Compute surrogate detector detection probability P_d(s_px)."""
        if self.pd_mode == "empirical_table" and self.pd_empirical_table:
            tbl = self.pd_empirical_table
            if s_px <= tbl[0][0]:
                return float(tbl[0][1])
            if s_px >= tbl[-1][0]:
                return float(tbl[-1][1])
            for i in range(len(tbl) - 1):
                s0, p0 = tbl[i]
                s1, p1 = tbl[i + 1]
                if s0 <= s_px <= s1:
                    alpha = (s_px - s0) / (s1 - s0) if s1 > s0 else 0.0
                    return float(p0 + alpha * (p1 - p0))
            return float(tbl[-1][1])
        elif self.pd_mode == "log_logistic":
            if s_px <= 1e-6:
                return 0.0
            u = math.log(max(1e-4, s_px))
            u50 = math.log(max(1e-4, self.s50_px))
            w = max(1e-4, self.w_ln)
            x = -(u - u50) / w
            x = max(-50.0, min(50.0, x))
            return 1.0 / (1.0 + math.exp(x))
        else:
            x = -(s_px - self.s50_px) / max(1e-4, self.w_px)
            x = max(-50.0, min(50.0, x))
            return 1.0 / (1.0 + math.exp(x))

    def _sample_false_alarm_position(
        self,
        p_cam_ecef: np.ndarray,
        r_ecef_cam: np.ndarray,
        cam_model: PinholeCamera,
    ) -> tuple[float, float]:
        """Sample false alarm (u, v) with 3x higher density below the analytic horizon using rng_clutter."""
        max_tries = 50
        w, h = cam_model.width, cam_model.height
        for _ in range(max_tries):
            u_cand = float(self.rng_clutter.uniform(0.0, w - 1.0))
            v_cand = float(self.rng_clutter.uniform(0.0, h - 1.0))

            roots = horizon.analytic_horizon_v_at_u(p_cam_ecef, r_ecef_cam, cam_model, u_cand)
            if len(roots) == 1:
                is_below = v_cand > roots[0]
            else:
                is_below = horizon.is_earth_at_pixel(p_cam_ecef, r_ecef_cam, cam_model, u_cand, v_cand)

            if is_below:
                return u_cand, v_cand
            else:
                if self.rng_clutter.uniform(0.0, 1.0) < (1.0 / self.terrain_clutter_factor):
                    return u_cand, v_cand

        return float(self.rng_clutter.uniform(0.0, w - 1.0)), float(self.rng_clutter.uniform(0.0, h - 1.0))

    def generate_frame(
        self,
        t_s: float,
        ownship_state: Any,
        encounters: Sequence[EncounterTrajectory],
        disable_noise: bool = False,
        pd_override: float | None = None,
    ) -> tuple[list[DetectionRecord], dict[str, Any]]:
        """Generate all ring-camera detection records and ground truth for time t_s."""
        # 1. Ownship state and INS attitude
        p_arp_ecef = ownship_state.p_ecef_m
        lat_o, lon_o = ownship_state.lat_deg, ownship_state.lon_deg
        r_ecef_ned_mat = wgs84.r_ecef_ned(lat_o, lon_o)

        yaw_true, pitch_true, roll_true = ownship_state.yaw_rad, ownship_state.pitch_rad, ownship_state.roll_rad
        r_ned_body_true = rotations.euler_to_r(yaw_true, pitch_true, roll_true)
        r_ecef_body_true = r_ecef_ned_mat @ r_ned_body_true

        # Step INS drift (uses self.rng)
        _, (yaw_err, pitch_err, roll_err) = self.step_ins(disable_noise=disable_noise)
        yaw_ins = yaw_true + yaw_err
        pitch_ins = pitch_true + pitch_err
        roll_ins = roll_true + roll_err
        r_ned_body_ins = rotations.euler_to_r(yaw_ins, pitch_ins, roll_ins)

        # 2. Camera poses
        cam_poses = self.ring.compute_poses(
            lat_deg=lat_o,
            lon_deg=lon_o,
            h_m=ownship_state.h_m,
            yaw_rad=yaw_true,
            pitch_rad=pitch_true,
            roll_rad=roll_true,
        )

        detections: list[DetectionRecord] = []
        gt_by_intruder: dict[str, dict[str, Any]] = {}

        # 3. Process each intruder (uses self.rng exclusively)
        for enc in encounters:
            p_i = enc.position_at(t_s)
            v_i = enc.velocity_at(t_s)
            r_ecef_body_i, ypr_i = enc.attitude_at(t_s)

            # Ground truth relative vector
            r_ecef = p_i - p_arp_ecef
            v_rel = v_i - (r_ecef_ned_mat @ ownship_state.v_ned_mps)
            range_m = float(np.linalg.norm(r_ecef))

            r_ned = r_ecef_ned_mat.T @ r_ecef
            r_norm = float(np.linalg.norm(r_ned))
            los_ned_gt = (r_ned / r_norm) if r_norm > 1e-12 else np.zeros(3)

            verts_model = get_model_vertices(enc.params.model_id)

            gt_by_intruder[enc.params.id] = {
                "p_ecef": p_i,
                "v_ecef": v_i,
                "range_m": range_m,
                "los_ned": los_ned_gt,
                "cams": {},
            }

            # Check each ring camera
            for cam in self.ring.cameras:
                c_pose = cam_poses[cam.id]
                p_cam = np.array(c_pose["position"], dtype=np.float64)
                r_cam = np.array(c_pose["r_ecef_cam"], dtype=np.float64)

                p_cam_rel = r_cam.T @ (p_i - p_cam)
                z_cam = float(p_cam_rel[2])

                if z_cam <= 1e-3:
                    continue

                u_true, v_true, in_front = self.ring.camera_model.project_cam(p_cam_rel)
                if not in_front or not self.ring.camera_model.in_image(u_true, v_true):
                    continue

                # Projected vertex bounding box
                verts_ecef = p_i[None, :] + (r_ecef_body_i @ verts_model.T).T
                verts_cam = (r_cam.T @ (verts_ecef - p_cam).T).T
                if not np.all(verts_cam[:, 2] > 1e-3):
                    continue

                u_coords = self.ring.camera_model.fx * (verts_cam[:, 0] / verts_cam[:, 2]) + self.ring.camera_model.cx
                v_coords = self.ring.camera_model.fy * (verts_cam[:, 1] / verts_cam[:, 2]) + self.ring.camera_model.cy
                u_min, u_max = float(np.min(u_coords)), float(np.max(u_coords))
                v_min, v_max = float(np.min(v_coords)), float(np.max(v_coords))

                box_w = max(0.0, u_max - u_min)
                box_h = max(0.0, v_max - v_min)
                s_px = max(box_w, box_h)

                mom_res = compute_vertex_second_moment_scale(
                    u_coords=u_coords,
                    v_coords=v_coords,
                    sigma_psf_px=self.sigma_psf_px,
                    lambda_min_px=self.lambda_min_px,
                    fx=self.ring.fx,
                )
                s_mom_true_px = mom_res.s_mom_px

                gt_cam_los = (p_i - p_cam) / float(np.linalg.norm(p_i - p_cam))
                gt_cam_los_ned = r_ecef_ned_mat.T @ gt_cam_los

                gt_by_intruder[enc.params.id]["cams"][cam.id] = {
                    "u_true": u_true,
                    "v_true": v_true,
                    "box": [u_min, v_min, u_max, v_max],
                    "s_px": s_px,
                    "s_mom_px": s_mom_true_px,
                    "gt_cam_los_ned": gt_cam_los_ned,
                }

                # Determine detection
                p_d = self.detection_probability(s_px) if pd_override is None else pd_override

                if disable_noise or pd_override == 1.0:
                    detected = True
                else:
                    if self.markov_misses:
                        prev = self.prev_detected.get(enc.params.id)
                        if prev is not None:
                            if self.rng.uniform(0.0, 1.0) < self.p_stay:
                                detected = prev
                            else:
                                detected = (self.rng.uniform(0.0, 1.0) < p_d)
                        else:
                            detected = (self.rng.uniform(0.0, 1.0) < p_d)
                    else:
                        detected = (self.rng.uniform(0.0, 1.0) < p_d)

                self.prev_detected[enc.params.id] = detected

                if not detected:
                    continue

                # Add noise
                if disable_noise or (self.sigma_px <= 0.0 and self.sigma_ang_rad <= 0.0):
                    u_final, v_final = u_true, v_true
                else:
                    # Centroid noise
                    u_noisy = u_true + self.rng.normal(0.0, self.sigma_px) if self.sigma_px > 0.0 else u_true
                    v_noisy = v_true + self.rng.normal(0.0, self.sigma_px) if self.sigma_px > 0.0 else v_true

                    # Camera angular frame noise
                    if self.sigma_ang_rad > 0.0:
                        xc = (u_noisy - self.ring.cx) / self.ring.fx
                        yc = (v_noisy - self.ring.cy) / self.ring.fy
                        az_c = math.atan2(xc, 1.0)
                        el_c = math.atan2(-yc, math.hypot(xc, 1.0))

                        az_c_noisy = az_c + self.rng.normal(0.0, self.sigma_ang_rad)
                        el_c_noisy = el_c + self.rng.normal(0.0, self.sigma_ang_rad)

                        cos_el = math.cos(el_c_noisy)
                        rc = np.array([
                            cos_el * math.sin(az_c_noisy),
                            -math.sin(el_c_noisy),
                            cos_el * math.cos(az_c_noisy),
                        ], dtype=np.float64)

                        if rc[2] > 1e-6:
                            u_final = self.ring.fx * (rc[0] / rc[2]) + self.ring.cx
                            v_final = self.ring.fy * (rc[1] / rc[2]) + self.ring.cy
                        else:
                            u_final, v_final = u_noisy, v_noisy
                    else:
                        u_final, v_final = u_noisy, v_noisy

                s_mom_min_px = math.sqrt(12.0) * self.lambda_min_px
                if not disable_noise and self.sigma_s_px > 0.0:
                    eps_s = float(self.rng.normal(0.0, self.sigma_s_px))
                    z_s = eps_s / self.sigma_s_px
                    s_px_meas = max(0.05, float(s_px + eps_s))
                    s_mom_px_meas = max(s_mom_min_px, float(s_mom_true_px + self.sigma_s_mom_px * z_s))
                else:
                    s_px_meas = float(s_px)
                    s_mom_px_meas = max(s_mom_min_px, float(s_mom_true_px))

                score = float(p_d)
                detections.append(DetectionRecord(
                    t=t_s,
                    camera=cam.id,
                    u=float(u_final),
                    v=float(v_final),
                    s_px=float(s_px_meas),
                    score=score,
                    is_false_alarm=False,
                    truth_id=enc.params.id,
                    s_mom_px=float(s_mom_px_meas),
                ))

        # 4. False alarms per camera (uses self.rng_clutter exclusively for D29 CRN)
        if not disable_noise and self.fppi_lambda > 0.0:
            for cam in self.ring.cameras:
                c_pose = cam_poses[cam.id]
                p_cam = np.array(c_pose["position"], dtype=np.float64)
                r_cam = np.array(c_pose["r_ecef_cam"], dtype=np.float64)

                n_fa = int(self.rng_clutter.poisson(self.fppi_lambda))
                for _ in range(n_fa):
                    u_fa, v_fa = self._sample_false_alarm_position(p_cam, r_cam, self.ring.camera_model)
                    s_fa = float(self.rng_clutter.uniform(self.fa_size_min_px, self.fa_size_max_px))
                    fa_score = float(self.detection_probability(s_fa))

                    detections.append(DetectionRecord(
                        t=t_s,
                        camera=cam.id,
                        u=u_fa,
                        v=v_fa,
                        s_px=s_fa,
                        score=fa_score,
                        is_false_alarm=True,
                        truth_id=None,
                        s_mom_px=s_fa,
                    ))

        metadata = {
            "t_s": t_s,
            "p_own_ecef": p_arp_ecef,
            "r_ecef_ned": r_ecef_ned_mat,
            "lat_deg": lat_o,
            "lon_deg": lon_o,
            "h_m": ownship_state.h_m,
            "r_ned_body_true": r_ned_body_true,
            "r_ned_body_ins": r_ned_body_ins,
            "ins_err_rpy_rad": [roll_err, pitch_err, yaw_err],
            "gt_by_intruder": gt_by_intruder,
        }

        return detections, metadata
