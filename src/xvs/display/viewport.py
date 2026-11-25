"""Virtual head-tracked viewport camera and rendering engine.

Conforms strictly to docs/xvs/WP08_DISPLAY.md v0.1 §2.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from xvs.geo import wgs84, rotations
from xvs.rig.rig import Rig, CAMERA_ORDER
from xvs.rig.camera import PinholeCamera
from xvs.display.fusion import (
    load_display_config,
    get_eye_point_body,
    compute_world_points,
    sample_camera_bilinear,
)

# Camera forward nominal mapping in body axes:
# col 0: right (body y = [0, 1, 0])
# col 1: down (body z = [0, 0, 1])
# col 2: forward (body x = [1, 0, 0])
_R_BODY_CAM_FORWARD = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def compute_head_pose(t: float, config: dict[str, Any] | None = None) -> tuple[float, float, float]:
    """Compute scripted head track (yaw, pitch, roll) in body frame (radians) at time t.

    Formula (WP08 §2):
      yaw = 110 deg * sin(2*pi*t / 20 s)
      pitch = 15 deg * sin(2*pi*t / 13 s)
      roll = 0.0 rad
    """
    if config is None:
        config = load_display_config()

    ht = config.get("viewport", {}).get("head_track", {})
    yaw_amp = float(ht.get("yaw_amplitude_deg", 110.0))  # default-ok: optional fallback default
    yaw_period = float(ht.get("yaw_period_s", 20.0))  # default-ok: optional fallback default
    pitch_amp = float(ht.get("pitch_amplitude_deg", 15.0))  # default-ok: optional fallback default
    pitch_period = float(ht.get("pitch_period_s", 13.0))  # default-ok: optional fallback default

    yaw_deg = yaw_amp * math.sin(2.0 * math.pi * t / yaw_period)
    pitch_deg = pitch_amp * math.sin(2.0 * math.pi * t / pitch_period)

    return math.radians(yaw_deg), math.radians(pitch_deg), 0.0


def get_viewport_camera(config: dict[str, Any] | None = None) -> PinholeCamera:
    """Return virtual pinhole camera for the viewport (1280x720, 90x60 deg FOV)."""
    if config is None:
        config = load_display_config()

    vp = config.get("viewport", {})
    width = int(vp.get("width", 1280))  # default-ok: geometry dimension fallback
    height = int(vp.get("height", 720))  # default-ok: geometry dimension fallback
    fov_h = float(vp.get("fov_h_deg", 90.0))  # default-ok: geometry dimension fallback
    fov_v = float(vp.get("fov_v_deg", 60.0))  # default-ok: geometry dimension fallback

    fx = (width / 2.0) / math.tan(math.radians(fov_h) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(fov_v) / 2.0)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    return PinholeCamera(width=width, height=height, fov_h_deg=fov_h, fx=fx, fy=fy, cx=cx, cy=cy)


def compute_viewport_pose(
    aircraft_pose: dict[str, float],
    head_euler_rad: tuple[float, float, float],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute ECEF position and rotation matrix for the viewport camera.

    Args:
        aircraft_pose: dict with lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad.
        head_euler_rad: (yaw_rad, pitch_rad, roll_rad) in body axes.
        config: display configuration.

    Returns:
        dict with:
          p_ecef_view: (3,) eye position in ECEF.
          r_ecef_view: (3, 3) matrix mapping camera coords to ECEF: v_ecef = R_ecef_view @ v_cam.
          r_body_view: (3, 3) matrix mapping camera coords to body: v_body = R_body_view @ v_cam.
    """
    if config is None:
        config = load_display_config()

    lat = aircraft_pose["lat_deg"]
    lon = aircraft_pose["lon_deg"]
    h = aircraft_pose["h_m"]
    yaw = aircraft_pose["yaw_rad"]
    pitch = aircraft_pose["pitch_rad"]
    roll = aircraft_pose["roll_rad"]

    r_ecef_ned_mat = wgs84.r_ecef_ned(lat, lon)
    r_ned_body_mat = rotations.euler_to_r(yaw, pitch, roll)
    r_ecef_body_mat = r_ecef_ned_mat @ r_ned_body_mat
    p_ecef_body = wgs84.geodetic_to_ecef(lat, lon, h)

    eye_body = get_eye_point_body(config)
    p_ecef_view = p_ecef_body + r_ecef_body_mat @ eye_body

    head_yaw, head_pitch, head_roll = head_euler_rad
    r_body_head = rotations.euler_to_r(head_yaw, head_pitch, head_roll)

    # v_body = R_body_head @ R_body_cam_forward @ v_cam
    r_body_view = r_body_head @ _R_BODY_CAM_FORWARD
    r_ecef_view = r_ecef_body_mat @ r_body_view

    return {
        "p_ecef_view": p_ecef_view,
        "r_ecef_view": r_ecef_view,
        "r_body_view": r_body_view,
    }


def render_viewport_frame(
    camera_images: dict[str, np.ndarray],
    rig: Rig,
    aircraft_pose: dict[str, float],
    head_euler_rad: tuple[float, float, float],
    depth_model: str = "ellipsoid",
    feather_width_px: float = 64.0,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Render a single 1280x720 viewport frame from the pilot's eye point.

    Args:
        camera_images: Dict mapping cam_id -> image array (H, W, 3).
        rig: Rig object (rig_true).
        aircraft_pose: Dict with lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad.
        head_euler_rad: (yaw_rad, pitch_rad, roll_rad) in body frame.
        depth_model: "ellipsoid" | "infinity" | "sphere_R".
        feather_width_px: Feathered blending edge width.
        config: Display configuration dict.

    Returns:
        (height, width, 3) uint8 RGB image.
    """
    if config is None:
        config = load_display_config()

    vp_cam = get_viewport_camera(config)
    vp_pose = compute_viewport_pose(aircraft_pose, head_euler_rad, config)

    eye_ecef = vp_pose["p_ecef_view"]
    r_ecef_view = vp_pose["r_ecef_view"]

    # Generate view rays for viewport camera
    w = vp_cam.width
    h = vp_cam.height

    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()

    # Unproject to camera frame
    rx = (u_flat - vp_cam.cx) / vp_cam.fx
    ry = (v_flat - vp_cam.cy) / vp_cam.fy
    rz = np.ones_like(rx)
    rays_cam = np.stack([rx, ry, rz], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

    # Rays in ECEF: d_ecef = R_ecef_view @ d_cam
    rays_ecef = rays_cam @ r_ecef_view.T

    # World points via depth model
    sphere_r = float(config.get("sphere_r_m", 1000.0))  # default-ok: optional fallback default
    world_pts, hit_mask, is_finite = compute_world_points(
        eye_ecef, rays_ecef, depth_model=depth_model, sphere_r_m=sphere_r
    )

    n_pixels = len(u_flat)
    first_img = next(iter(camera_images.values()))
    channels = first_img.shape[2] if first_img.ndim == 3 else 1

    weight_sum = np.zeros(n_pixels, dtype=np.float64)
    color_accum = np.zeros((n_pixels, channels), dtype=np.float64)

    # Compute camera poses in ECEF
    cam_poses = rig.compute_poses(
        aircraft_pose["lat_deg"],
        aircraft_pose["lon_deg"],
        aircraft_pose["h_m"],
        aircraft_pose["yaw_rad"],
        aircraft_pose["pitch_rad"],
        aircraft_pose["roll_rad"],
    )

    w_cam_img = rig.width
    h_cam_img = rig.height

    for cam_id in CAMERA_ORDER:
        if cam_id not in camera_images:
            continue
        c_pose = cam_poses[cam_id]
        c_ecef = c_pose["p_ecef_cam"]
        r_ecef_cam = c_pose["r_ecef_cam"]
        img = camera_images[cam_id]

        diff_finite = world_pts - c_ecef
        diff_inf = world_pts
        p_diff = np.where(is_finite[:, None], diff_finite, diff_inf)

        p_cam = p_diff @ r_ecef_cam
        z_cam = p_cam[:, 2]
        x_cam = p_cam[:, 0]
        y_cam = p_cam[:, 1]

        front_mask = z_cam > 1e-6
        safe_z = np.where(front_mask, z_cam, 1.0)

        fx = rig.camera_model.fx
        fy = rig.camera_model.fy
        cx = rig.camera_model.cx
        cy = rig.camera_model.cy

        u = np.where(front_mask, fx * (x_cam / safe_z) + cx, -999.0)
        v = np.where(front_mask, fy * (y_cam / safe_z) + cy, -999.0)

        in_img = (
            front_mask
            & (u >= -0.5)
            & (u <= w_cam_img - 0.5)
            & (v >= -0.5)
            & (v <= h_cam_img - 0.5)
        )

        d_u = np.minimum(u + 0.5, w_cam_img - 0.5 - u)
        d_v = np.minimum(v + 0.5, h_cam_img - 0.5 - v)
        d_border = np.minimum(d_u, d_v)
        w_cam = np.where(in_img, np.clip(d_border, 1e-3, feather_width_px), 0.0)

        sampled = sample_camera_bilinear(img, u, v, in_img)
        if sampled.ndim == 1:
            sampled = sampled[:, None]

        color_accum += w_cam[:, None] * sampled
        weight_sum += w_cam

    # Fallback to infinity rays for any uncovered parallax shadow gaps near ground
    uncovered = weight_sum < 1e-9
    if np.any(uncovered):
        for cam_id in CAMERA_ORDER:
            if cam_id not in camera_images:
                continue
            r_ecef_cam = cam_poses[cam_id]["r_ecef_cam"]
            img = camera_images[cam_id]
            p_cam = rays_ecef[uncovered] @ r_ecef_cam
            z_cam = p_cam[:, 2]
            x_cam = p_cam[:, 0]
            y_cam = p_cam[:, 1]
            front_mask = z_cam > 1e-6
            safe_z = np.where(front_mask, z_cam, 1.0)
            u = np.where(front_mask, rig.camera_model.fx * (x_cam / safe_z) + rig.camera_model.cx, -999.0)
            v = np.where(front_mask, rig.camera_model.fy * (y_cam / safe_z) + rig.camera_model.cy, -999.0)
            in_img = front_mask & (u >= -0.5) & (u <= w_cam_img - 0.5) & (v >= -0.5) & (v <= h_cam_img - 0.5)
            d_u = np.minimum(u + 0.5, w_cam_img - 0.5 - u)
            d_v = np.minimum(v + 0.5, h_cam_img - 0.5 - v)
            d_border = np.minimum(d_u, d_v)
            w_cam = np.where(in_img, np.clip(d_border, 1e-3, feather_width_px), 0.0)
            sampled = sample_camera_bilinear(img, u, v, in_img)
            if sampled.ndim == 1:
                sampled = sampled[:, None]
            color_accum[uncovered] += w_cam[:, None] * sampled
            weight_sum[uncovered] += w_cam

    has_weight = weight_sum > 1e-9
    color_accum = np.where(has_weight[:, None], color_accum / weight_sum[:, None], 0.0)

    fused_img = np.clip(color_accum, 0.0, 255.0).astype(np.uint8)
    return fused_img.reshape(h, w, channels)



def extract_viewport(
    camera_images: dict[str, np.ndarray],
    rig: Rig,
    aircraft_pose: dict[str, float],
    head_euler_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    depth_model: str = "ellipsoid",
    feather_width_px: float = 64.0,
    excluded_cameras: list[str] | set[str] | tuple[str, ...] | None = None,
    config: dict[str, Any] | None = None,
    head_euler: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Extract a 1280x720 see-through viewport frame from 6-camera imagery using the ellipsoid depth model.

    Supports optional single-camera fault exclusion (`excluded_cameras`) when `F_isolated` is asserted.
    Returns (height, width, 3) uint8 RGB image.
    """
    if head_euler is not None:
        head_euler_rad = head_euler
    excluded = set(excluded_cameras or ())
    active_images = {
        cam_id: img for cam_id, img in camera_images.items() if cam_id not in excluded
    }
    return render_viewport_frame(
        camera_images=active_images,
        rig=rig,
        aircraft_pose=aircraft_pose,
        head_euler_rad=head_euler_rad,
        depth_model=depth_model,
        feather_width_px=feather_width_px,
        config=config,
    )


def compute_viewport_pose_from_estimated_attitude(
    ownship_lat_deg: float,
    ownship_lon_deg: float,
    ownship_h_m: float,
    r_ned_body_est: np.ndarray,
    head_euler_rad: tuple[float, float, float],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute pilot eye ECEF position and viewport camera rotations from WP-06/07 estimated attitude.

    Enforces V5_G1: uses only estimated attitude `r_ned_body_est` (never ground-truth pose).
    """
    if config is None:
        config = load_display_config()

    r_ecef_ned_mat = wgs84.r_ecef_ned(ownship_lat_deg, ownship_lon_deg)
    r_ned_body_mat = np.asarray(r_ned_body_est, dtype=np.float64)
    r_ecef_body_est = r_ecef_ned_mat @ r_ned_body_mat
    p_ecef_body = wgs84.geodetic_to_ecef(ownship_lat_deg, ownship_lon_deg, ownship_h_m)

    eye_body = get_eye_point_body(config)
    p_ecef_view = p_ecef_body + r_ecef_body_est @ eye_body

    head_yaw, head_pitch, head_roll = head_euler_rad
    r_body_head = rotations.euler_to_r(head_yaw, head_pitch, head_roll)
    r_body_view = r_body_head @ _R_BODY_CAM_FORWARD
    r_ecef_view_est = r_ecef_body_est @ r_body_view

    return {
        "p_ecef_view": p_ecef_view,
        "r_ecef_view_est": r_ecef_view_est,
        "r_body_view": r_body_view,
        "r_ned_body_est": r_ned_body_mat,
    }
