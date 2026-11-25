"""Display fusion machinery: 360-degree equirectangular panoramas, depth models, and feathered blending.

Conforms strictly to docs/xvs/WP08_DISPLAY.md v0.1 §1, §4, §5.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from xvs.geo import wgs84, rotations
from xvs.rig.rig import Rig, CAMERA_ORDER
from xvs.rig.camera import PinholeCamera

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DISPLAY_CONFIG = REPO_ROOT / "configs" / "xvs" / "display" / "g0.json"

# WGS84 scaling matrix S = diag(1/a, 1/a, 1/b) mapping ellipsoid to unit sphere
_S_MAT = np.diag([1.0 / wgs84.WGS84_A, 1.0 / wgs84.WGS84_A, 1.0 / wgs84.WGS84_B])


def load_display_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load display configuration JSON."""
    p = Path(config_path) if config_path is not None else DEFAULT_DISPLAY_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_eye_point_body(config: dict[str, Any] | None = None) -> np.ndarray:
    """Return eye point in body FRD coordinates (metres)."""
    if config is None:
        config = load_display_config()
    return np.asarray(config["eye_point_body_frd_m"], dtype=np.float64)


def intersect_ray_ellipsoid(
    origin_ecef: np.ndarray,
    direction_ecef: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """Find first forward intersection of a single unit ray with the WGS84 ellipsoid.

    Returns (p_ecef, s_m) where s_m > 0 is distance along ray in metres.
    Returns (None, None) if ray does not intersect (misses Earth into space).
    """
    pts, dists, hits = intersect_rays_ellipsoid(
        origin_ecef.reshape(1, 3), direction_ecef.reshape(1, 3)
    )
    if hits[0]:
        return pts[0], float(dists[0])
    return None, None


def intersect_rays_ellipsoid(
    origins_ecef: np.ndarray,
    directions_ecef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized intersection of unit rays with the WGS84 ellipsoid.

    Args:
        origins_ecef: (3,) or (N, 3) ray origins in ECEF.
        directions_ecef: (N, 3) unit ray directions in ECEF.

    Returns:
        points_ecef: (N, 3) intersection points (or NaN if miss).
        distances: (N,) distance s along ray in metres (or NaN if miss).
        hit_mask: (N,) boolean array indicating forward intersection.
    """
    orig = np.asarray(origins_ecef, dtype=np.float64)
    dirs = np.asarray(directions_ecef, dtype=np.float64)
    if orig.ndim == 1:
        orig = np.broadcast_to(orig, dirs.shape)

    # Scale to unit sphere space: p' = S @ p
    c_prime = orig @ _S_MAT  # (N, 3)
    d_prime = dirs @ _S_MAT  # (N, 3)

    # Solve ||c' + s * d'||^2 = 1 => A*s^2 + 2*B*s + K = 0
    # A = ||d'||^2, B = c' . d', K = ||c'||^2 - 1
    a_coef = np.sum(d_prime * d_prime, axis=-1)  # (N,)
    b_coef = np.sum(c_prime * d_prime, axis=-1)  # (N,)
    k_coef = np.sum(c_prime * c_prime, axis=-1) - 1.0  # (N,)

    disc = b_coef * b_coef - a_coef * k_coef

    hit_mask = (disc >= 0.0) & (b_coef < 0.0) & (k_coef > 0.0)

    # Safe square root
    safe_disc = np.where(hit_mask, disc, 0.0)
    sqrt_disc = np.sqrt(safe_disc)

    # First forward intersection: s1 = (-B - sqrt(disc)) / A
    s1 = (-b_coef - sqrt_disc) / np.maximum(a_coef, 1e-15)
    hit_mask = hit_mask & (s1 > 0.0)

    distances = np.where(hit_mask, s1, np.nan)
    pts = orig + np.where(hit_mask[:, None], distances[:, None] * dirs, np.nan)

    return pts, distances, hit_mask


def compute_world_points(
    eye_ecef: np.ndarray,
    directions_ecef: np.ndarray,
    depth_model: str = "ellipsoid",
    sphere_r_m: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute world points for a batch of view rays under the chosen depth model.

    Args:
        eye_ecef: (3,) eye position in ECEF.
        directions_ecef: (N, 3) unit view directions in ECEF.
        depth_model: "ellipsoid" | "infinity" | "sphere_R".
        sphere_r_m: Sphere radius in metres for "sphere_R" model (default: 1000 m).

    Returns:
        points_ecef: (N, 3) world points. For infinity points, stores unit direction.
        hit_mask: (N,) boolean array indicating ground intersection (for ellipsoid).
        is_finite: (N,) boolean array indicating whether point is at finite 3D distance.
    """
    n_rays = len(directions_ecef)
    pts_ell, _, ground_hit_mask = intersect_rays_ellipsoid(eye_ecef, directions_ecef)

    if depth_model == "ellipsoid":
        finite_pts = np.where(ground_hit_mask[:, None], pts_ell, directions_ecef)
        return finite_pts, ground_hit_mask, ground_hit_mask

    elif depth_model == "infinity":
        is_finite = np.zeros(n_rays, dtype=bool)
        return directions_ecef.copy(), ground_hit_mask, is_finite

    elif depth_model == "sphere_R":
        pts = eye_ecef + sphere_r_m * directions_ecef
        is_finite = np.ones(n_rays, dtype=bool)
        return pts, ground_hit_mask, is_finite

    else:
        raise ValueError(f"Unknown depth model: {depth_model}")


def compute_parallax_angle(
    p_world: np.ndarray,
    eye_ecef: np.ndarray,
    cam_ecef: np.ndarray,
) -> np.ndarray:
    """Compute parallax angular discrepancy (radians) between eye and camera for 3D ground points.

    Angle between (P - C_c) and (P - E).
    """
    p = np.asarray(p_world, dtype=np.float64)
    v_e = p - eye_ecef
    v_c = p - cam_ecef

    norm_e = np.linalg.norm(v_e, axis=-1, keepdims=True)
    norm_c = np.linalg.norm(v_c, axis=-1, keepdims=True)

    u_e = v_e / np.maximum(norm_e, 1e-15)
    u_c = v_c / np.maximum(norm_c, 1e-15)

    dot = np.sum(u_e * u_c, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    cross = np.cross(u_e, u_c)
    sin_ang = np.linalg.norm(cross, axis=-1)
    return np.arctan2(sin_ang, dot)


def generate_panorama_rays(
    width: int = 2048,
    height: int = 1024,
    frame: str = "ned",
    r_ned_body: np.ndarray | None = None,
) -> np.ndarray:
    """Generate equirectangular 360x180 degree unit ray directions.

    Convention (WP08 §1):
      - 360 deg azimuth x 180 deg elevation.
      - Centre column is Azimuth = 0 (North in NED, Nose/Forward in Body).
      - Top row (v=0) is Zenith (+90 deg elevation).
      - Mid row (v=height/2) is Horizon (0 deg elevation).
      - Bottom row (v=height-1) is Nadir (-90 deg elevation).

    Args:
        width: Panorama width in pixels.
        height: Panorama height in pixels.
        frame: "ned" or "body".
        r_ned_body: (3, 3) rotation matrix if converting body to ned or vice versa.

    Returns:
        (height, width, 3) unit direction vectors.
    """
    # u in [0, width - 1], center at width / 2.0 -> az in [-pi, pi]
    u = np.arange(width, dtype=np.float64)
    az = (u - (width / 2.0)) * (2.0 * math.pi / width)  # rad

    # v in [0, height - 1], center at height / 2.0 -> el in [+pi/2, -pi/2]
    v = np.arange(height, dtype=np.float64)
    el = ((height / 2.0) - v) * (math.pi / height)  # rad

    cos_el = np.cos(el)[:, None]  # (H, 1)
    sin_el = np.sin(el)[:, None]  # (H, 1)
    cos_az = np.cos(az)[None, :]  # (1, W)
    sin_az = np.sin(az)[None, :]  # (1, W)

    # Unit vector in local reference frame (x=Forward/North, y=Right/East, z=Down):
    # x = cos(el) * cos(az)
    # y = cos(el) * sin(az)
    # z = -sin(el)  (since Down is positive z)
    dir_x = cos_el * cos_az  # (H, W)
    dir_y = cos_el * sin_az  # (H, W)
    dir_z = -np.broadcast_to(sin_el, (height, width))  # (H, W)

    rays = np.stack([dir_x, dir_y, dir_z], axis=-1)  # (H, W, 3)

    if frame == "body":
        return rays
    elif frame == "ned":
        return rays
    else:
        raise ValueError(f"Unknown frame: {frame}")


def sample_camera_bilinear(
    image: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Vectorized bilinear sampling of an image (H, W) or (H, W, C) at sub-pixel coordinates.

    Out-of-bounds coordinates return 0.
    """
    h_img, w_img = image.shape[:2]
    is_color = (image.ndim == 3)
    c_channels = image.shape[2] if is_color else 1

    out_shape = u_coords.shape
    if is_color:
        out = np.zeros((*out_shape, c_channels), dtype=np.float64)
    else:
        out = np.zeros(out_shape, dtype=np.float64)

    # Valid bounding filter
    m = (
        valid_mask
        & (u_coords >= 0.0)
        & (u_coords <= w_img - 1.0)
        & (v_coords >= 0.0)
        & (v_coords <= h_img - 1.0)
    )

    if not np.any(m):
        return out

    u_sub = u_coords[m]
    v_sub = v_coords[m]

    u0 = np.floor(u_sub).astype(np.int64)
    v0 = np.floor(v_sub).astype(np.int64)
    u1 = np.minimum(u0 + 1, w_img - 1)
    v1 = np.minimum(v0 + 1, h_img - 1)

    du = u_sub - u0
    dv = v_sub - v0

    w00 = (1.0 - du) * (1.0 - dv)
    w10 = du * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w11 = du * dv

    if is_color:
        val00 = image[v0, u0, :].astype(np.float64)
        val10 = image[v0, u1, :].astype(np.float64)
        val01 = image[v1, u0, :].astype(np.float64)
        val11 = image[v1, u1, :].astype(np.float64)

        w00_c = w00[:, None]
        w10_c = w10[:, None]
        w01_c = w01[:, None]
        w11_c = w11[:, None]

        sampled = w00_c * val00 + w10_c * val10 + w01_c * val01 + w11_c * val11
        out[m] = sampled
    else:
        val00 = image[v0, u0].astype(np.float64)
        val10 = image[v0, u1].astype(np.float64)
        val01 = image[v1, u0].astype(np.float64)
        val11 = image[v1, u1].astype(np.float64)

        sampled = w00 * val00 + w10 * val10 + w01 * val01 + w11 * val11
        out[m] = sampled

    return out


def render_display_panorama(
    camera_images: dict[str, np.ndarray],
    rig: Rig,
    aircraft_pose: dict[str, float],
    depth_model: str = "ellipsoid",
    frame: str = "ned",
    width: int = 2048,
    height: int = 1024,
    feather_width_px: float = 64.0,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render 360-degree equirectangular panorama from the pilot's eye point.

    Args:
        camera_images: Dict mapping cam_id -> image array (H, W, 3) uint8 or float.
        rig: Rig object (typically rig_true).
        aircraft_pose: Dict with lat_deg, lon_deg, h_m, yaw_rad, pitch_rad, roll_rad.
        depth_model: "ellipsoid" | "infinity" | "sphere_R".
        frame: "ned" or "body".
        width, height: Panorama output dimensions.
        feather_width_px: Feathered blending edge width (default: 64 px).
        config: Optional display config dict.

    Returns:
        panorama_img: (height, width, 3) uint8 RGB image.
        stats: Coverage, seam photometry, and parallax statistics dict.
    """
    if config is None:
        config = load_display_config()

    lat = aircraft_pose["lat_deg"]
    lon = aircraft_pose["lon_deg"]
    h = aircraft_pose["h_m"]
    yaw = aircraft_pose["yaw_rad"]
    pitch = aircraft_pose["pitch_rad"]
    roll = aircraft_pose["roll_rad"]

    # Compute camera poses and aircraft matrices
    cam_poses = rig.compute_poses(lat, lon, h, yaw, pitch, roll)

    r_ecef_ned_mat = wgs84.r_ecef_ned(lat, lon)
    r_ned_body_mat = rotations.euler_to_r(yaw, pitch, roll)
    r_ecef_body_mat = r_ecef_ned_mat @ r_ned_body_mat
    p_ecef_body = wgs84.geodetic_to_ecef(lat, lon, h)

    # Eye point in ECEF
    eye_body = get_eye_point_body(config)
    eye_ecef = p_ecef_body + r_ecef_body_mat @ eye_body

    # Generate view rays in chosen reference frame
    rays_frame = generate_panorama_rays(width=width, height=height, frame=frame)
    flat_rays = rays_frame.reshape(-1, 3)

    if frame == "ned":
        # Convert NED to ECEF: d_ecef = R_ecef_ned @ d_ned
        rays_ecef = flat_rays @ r_ecef_ned_mat.T
    else:  # "body"
        # Convert Body to ECEF: d_ecef = R_ecef_body @ d_body
        rays_ecef = flat_rays @ r_ecef_body_mat.T

    # World points via depth model
    sphere_r = float(config.get("sphere_r_m", 1000.0))  # default-ok: optional fallback default
    world_pts, hit_mask, is_finite = compute_world_points(
        eye_ecef, rays_ecef, depth_model=depth_model, sphere_r_m=sphere_r
    )

    n_pixels = len(flat_rays)
    first_img = next(iter(camera_images.values()))
    is_color = (first_img.ndim == 3)
    channels = first_img.shape[2] if is_color else 1

    weight_sum = np.zeros(n_pixels, dtype=np.float64)
    color_accum = np.zeros((n_pixels, channels), dtype=np.float64)
    cam_count = np.zeros(n_pixels, dtype=np.int32)

    # Per-camera valid samples for seam analysis
    cam_samples: dict[str, np.ndarray] = {}
    cam_masks: dict[str, np.ndarray] = {}

    w_cam_img = rig.width
    h_cam_img = rig.height

    for cam_id in CAMERA_ORDER:
        if cam_id not in camera_images:
            continue
        c_pose = cam_poses[cam_id]
        c_ecef = c_pose["p_ecef_cam"]
        r_ecef_cam = c_pose["r_ecef_cam"]
        img = camera_images[cam_id]

        # Project world points into camera
        # If finite: p_cam = R_ecef_cam.T @ (P - C_c)
        # If infinity: p_cam = R_ecef_cam.T @ d_ecef
        diff_finite = world_pts - c_ecef
        diff_inf = world_pts  # direction only
        p_diff = np.where(is_finite[:, None], diff_finite, diff_inf)

        p_cam = p_diff @ r_ecef_cam  # (N, 3)
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

        # Feathered blending weight: min(dist to border, feather_width)
        d_u = np.minimum(u, w_cam_img - 1.0 - u)
        d_v = np.minimum(v, h_cam_img - 1.0 - v)
        d_border = np.minimum(d_u, d_v)
        w_cam = np.where(in_img, np.clip(d_border, 0.0, feather_width_px), 0.0)

        # Bilinear sampling
        sampled = sample_camera_bilinear(img, u, v, in_img)
        if sampled.ndim == 1:
            sampled = sampled[:, None]

        cam_samples[cam_id] = sampled
        cam_masks[cam_id] = in_img

        color_accum += w_cam[:, None] * sampled
        weight_sum += w_cam
        cam_count += in_img.astype(np.int32)

    # Normalize accumulated color
    has_weight = weight_sum > 1e-9
    color_accum = np.where(has_weight[:, None], color_accum / weight_sum[:, None], 0.0)

    # Coverage statistics (WP08 §4)
    unseen_count = int(np.sum(cam_count == 0))
    single_count = int(np.sum(cam_count == 1))
    overlap_count = int(np.sum(cam_count >= 2))

    coverage_stats = {
        "unseen_fraction": float(unseen_count / n_pixels),
        "single_camera_fraction": float(single_count / n_pixels),
        "overlap_fraction": float(overlap_count / n_pixels),
    }

    # Seam photometry (mean absolute difference in overlap bands)
    seam_ground_diffs = []
    seam_sky_diffs = []
    cam_keys = list(cam_samples.keys())
    for i in range(len(cam_keys)):
        for j in range(i + 1, len(cam_keys)):
            k1, k2 = cam_keys[i], cam_keys[j]
            overlap_pair = cam_masks[k1] & cam_masks[k2]
            if not np.any(overlap_pair):
                continue
            diff = np.abs(cam_samples[k1][overlap_pair] - cam_samples[k2][overlap_pair])
            mean_diff = float(np.mean(diff))

            # Ground vs Sky split
            m_ground = overlap_pair & hit_mask
            m_sky = overlap_pair & (~hit_mask)
            if np.any(m_ground):
                seam_ground_diffs.append(float(np.mean(np.abs(cam_samples[k1][m_ground] - cam_samples[k2][m_ground]))))
            if np.any(m_sky):
                seam_sky_diffs.append(float(np.mean(np.abs(cam_samples[k1][m_sky] - cam_samples[k2][m_sky]))))

    seam_stats = {
        "ground_overlap_mad": float(np.mean(seam_ground_diffs)) if seam_ground_diffs else 0.0,
        "sky_overlap_mad": float(np.mean(seam_sky_diffs)) if seam_sky_diffs else 0.0,
    }

    # Format output image
    fused_img = np.clip(color_accum, 0.0, 255.0).astype(np.uint8)
    if is_color:
        fused_img = fused_img.reshape(height, width, channels)
    else:
        fused_img = fused_img.reshape(height, width)

    stats = {
        "depth_model": depth_model,
        "frame": frame,
        "coverage": coverage_stats,
        "seam_photometry": seam_stats,
        "ground_hits_fraction": float(np.mean(hit_mask)),
    }

    return fused_img, stats
