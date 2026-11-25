"""Analytic traffic ground truth computation per frame, per intruder, per camera (WP-02 §2.6).

Conforms strictly to docs/xvs/WP02_TRAFFIC.md §2.6 and §3 G1.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from xvs.geo import wgs84, rotations
from xvs.rig.camera import PinholeCamera
from xvs.sim import horizon
from xvs.traffic.encounter import EncounterTrajectory


@dataclass
class CameraTrafficGT:
    camera_id: str
    u: float
    v: float
    z_m: float
    in_image: bool
    box_analytic: list[float] | None
    limb_dv_px: float | None
    background: str  # "earth" or "sky"


@dataclass
class TrafficIntruderGT:
    id: str
    class_id: str
    model_id: str
    p_ecef_m: list[float]
    v_ecef_mps: list[float]
    ypr_deg: list[float]
    range_m: float
    range_rate_mps: float
    closing_speed_mps: float
    los_ned: list[float]
    az_el_ned_deg: list[float]
    az_el_body_deg: list[float]
    los_rate_ned_radps: list[float]
    az_el_rates_ned_degps: list[float]
    cpa_extrapolated: dict[str, float]  # {"tcpa_s": ..., "range_cpa_m": ...}
    angular_span_rad: float
    cams: dict[str, CameraTrafficGT]


def compute_intruder_analytic_gt(
    t_s: float,
    encounter: EncounterTrajectory,
    p_o_ecef: np.ndarray,
    v_o_ecef: np.ndarray,
    r_ecef_body_own: np.ndarray,
    cameras: dict[str, tuple[np.ndarray, np.ndarray, PinholeCamera]],
    model_vertices: np.ndarray | None = None,
    max_body_dim_m: float = 10.0,
) -> TrafficIntruderGT:
    """Compute exact per-frame analytic ground truth for an intruder.

    cameras: dict mapping cam_id -> (p_cam_ecef, r_ecef_cam, pinhole_camera).
    """
    p_i = encounter.position_at(t_s)
    v_i = encounter.velocity_at(t_s)
    r_ecef_body_i, ypr_i = encounter.attitude_at(t_s)

    r_ecef = p_i - p_o_ecef
    v_rel = v_i - v_o_ecef
    range_m = float(np.linalg.norm(r_ecef))

    # Range rate & closing speed
    if range_m > 1e-9:
        range_rate = float(np.dot(r_ecef, v_rel) / range_m)
        closing_speed = -range_rate
    else:
        range_rate = 0.0
        closing_speed = 0.0

    # Local NED at ownship
    lat_o, lon_o, _ = wgs84.ecef_to_geodetic(*p_o_ecef)
    n_o = wgs84.r_ecef_ned(lat_o, lon_o)

    # LOS in NED
    r_ned = n_o.T @ r_ecef
    r_norm = float(np.linalg.norm(r_ned))
    los_ned = (r_ned / r_norm).tolist() if r_norm > 1e-9 else [0.0, 0.0, 0.0]

    r_xy = math.hypot(float(r_ned[0]), float(r_ned[1]))
    az_ned = math.atan2(float(r_ned[1]), float(r_ned[0]))
    el_ned = math.atan2(-float(r_ned[2]), r_xy)
    az_el_ned_deg = [math.degrees(az_ned), math.degrees(el_ned)]

    # LOS in BODY (ownship)
    r_body = r_ecef_body_own.T @ r_ecef
    r_body_norm = float(np.linalg.norm(r_body))
    az_body = math.atan2(float(r_body[1]), float(r_body[0]))
    el_body = math.atan2(-float(r_body[2]), math.hypot(float(r_body[0]), float(r_body[1])))
    az_el_body_deg = [math.degrees(az_body), math.degrees(el_body)]

    # LOS rate vector omega = (r x v_rel) / |r|^2 in ECEF, expressed in NED
    if range_m > 1e-9:
        omega_ecef = np.cross(r_ecef, v_rel) / (range_m * range_m)
        omega_ned = n_o.T @ omega_ecef
    else:
        omega_ned = np.zeros(3, dtype=np.float64)

    # Az/El rates in NED
    v_rel_ned = n_o.T @ v_rel
    r_N = float(r_ned[0])
    r_E = float(r_ned[1])
    r_D = float(r_ned[2])
    v_N = float(v_rel_ned[0])
    v_E = float(v_rel_ned[1])
    v_D = float(v_rel_ned[2])

    if r_xy > 1e-6:
        az_rate_rad = (r_N * v_E - r_E * v_N) / (r_xy * r_xy)
        el_rate_rad = (-r_xy * v_D + r_D * (r_N * v_N + r_E * v_E) / r_xy) / (range_m * range_m)
    else:
        az_rate_rad = 0.0
        el_rate_rad = 0.0
    az_el_rates_degps = [math.degrees(az_rate_rad), math.degrees(el_rate_rad)]

    # Constant-velocity CPA/TCPA extrapolation from time t
    v_rel_sq = float(np.dot(v_rel, v_rel))
    if v_rel_sq > 1e-9:
        tcpa_extrap = -float(np.dot(r_ecef, v_rel)) / v_rel_sq
        r_cpa_extrap = float(np.linalg.norm(r_ecef + v_rel * tcpa_extrap))
    else:
        tcpa_extrap = 0.0
        r_cpa_extrap = range_m

    # Angular span
    angular_span_rad = max_body_dim_m / range_m if range_m > 1e-9 else 0.0

    # Per Camera Ground Truth
    cam_gts: dict[str, CameraTrafficGT] = {}
    for cam_id, (p_cam, r_cam, cam_model) in cameras.items():
        p_cam_rel = r_cam.T @ (p_i - p_cam)
        z_m = float(p_cam_rel[2])

        if z_m > 1e-3:
            u, v, in_bounds = cam_model.project_cam(p_cam_rel)
            # Image spans [-0.5, W-0.5] x [-0.5, H-0.5] (CONVENTIONS §5; PinholeCamera.in_image).
            in_image = in_bounds and cam_model.in_image(u, v)
        else:
            u, v = 0.0, 0.0
            in_image = False

        # Analytic box from 3D model vertices
        box_analytic = None
        if model_vertices is not None and len(model_vertices) > 0 and z_m > 1e-3:
            # Transform vertices: P_vert_ecef = P_i + R_body @ V
            verts_ecef = p_i[None, :] + (r_ecef_body_i @ model_vertices.T).T
            verts_cam = (r_cam.T @ (verts_ecef - p_cam).T).T
            if np.all(verts_cam[:, 2] > 1e-3):
                u_coords = cam_model.fx * (verts_cam[:, 0] / verts_cam[:, 2]) + cam_model.cx
                v_coords = cam_model.fy * (verts_cam[:, 1] / verts_cam[:, 2]) + cam_model.cy
                box_analytic = [
                    float(np.min(u_coords)),
                    float(np.min(v_coords)),
                    float(np.max(u_coords)),
                    float(np.max(v_coords)),
                ]

        # Horizon relation (signed pixel distance from centre to limb in column u)
        limb_dv_px = None
        if in_image:
            roots = horizon.analytic_horizon_v_at_u(p_cam, r_cam, cam_model, u)
            if len(roots) == 1:
                limb_dv_px = float(v - roots[0])

        # Background: earth if centre ray intersects ellipsoid, else sky
        background = "sky"
        if z_m > 1e-3:
            if horizon.is_earth_at_pixel(p_cam, r_cam, cam_model, u, v):
                background = "earth"

        cam_gts[cam_id] = CameraTrafficGT(
            camera_id=cam_id,
            u=float(u),
            v=float(v),
            z_m=z_m,
            in_image=bool(in_image),
            box_analytic=box_analytic,
            limb_dv_px=limb_dv_px,
            background=background,
        )

    return TrafficIntruderGT(
        id=encounter.params.id,
        class_id=encounter.params.class_id,
        model_id=encounter.params.model_id,
        p_ecef_m=p_i.tolist(),
        v_ecef_mps=v_i.tolist(),
        ypr_deg=list(ypr_i),
        range_m=range_m,
        range_rate_mps=range_rate,
        closing_speed_mps=closing_speed,
        los_ned=los_ned,
        az_el_ned_deg=az_el_ned_deg,
        az_el_body_deg=az_el_body_deg,
        los_rate_ned_radps=omega_ned.tolist(),
        az_el_rates_ned_degps=az_el_rates_degps,
        cpa_extrapolated={"tcpa_s": tcpa_extrap, "range_cpa_m": r_cpa_extrap},
        angular_span_rad=angular_span_rad,
        cams=cam_gts,
    )
