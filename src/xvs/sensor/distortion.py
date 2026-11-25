"""Geometric lens distortion (Brown-Conrady model in OpenCV convention).

Spec reference: docs/xvs/WP03_SENSOR.md §1 item 3 and §3 item V3.
- OpenCV convention:
  x = (u - cx) / fx, y = (v - cy) / fy
  r^2 = x^2 + y^2
  x_dist = x * (1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2 + 2*x^2)
  y_dist = y * (1 + k1*r^2 + k2*r^4 + k3*r^6) + p1*(r^2 + 2*y^2) + 2*p2*x*y
- Inversion by Newton-Raphson.
- Bicubic warping and validity masking.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def distort_normalized(
    x: np.ndarray | float,
    y: np.ndarray | float,
    k1: float,
    k2: float,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Brown-Conrady forward distortion in normalized camera coordinates."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)

    r2 = xa * xa + ya * ya
    r4 = r2 * r2
    r6 = r4 * r2

    rad = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xd = xa * rad + 2.0 * p1 * xa * ya + p2 * (r2 + 2.0 * xa * xa)
    yd = ya * rad + p1 * (r2 + 2.0 * ya * ya) + 2.0 * p2 * xa * ya
    return xd, yd


def undistort_normalized(
    xd: np.ndarray | float,
    yd: np.ndarray | float,
    k1: float,
    k2: float,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    max_iter: int = 15,
    tol: float = 1e-13,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert Brown-Conrady distortion using Newton-Raphson iteration.

    Finds (x, y) such that distort_normalized(x, y) == (xd, yd).
    """
    xda = np.asarray(xd, dtype=np.float64)
    yda = np.asarray(yd, dtype=np.float64)

    # Initial guess
    x = np.copy(xda)
    y = np.copy(yda)

    for _ in range(max_iter):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        rad = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        drad_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4

        # Residuals
        f_x = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xda
        f_y = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yda

        # Jacobian components
        j11 = rad + 2.0 * x * x * drad_dr2 + 2.0 * p1 * y + 6.0 * p2 * x
        j12 = 2.0 * x * y * drad_dr2 + 2.0 * p1 * x + 2.0 * p2 * y
        j21 = j12
        j22 = rad + 2.0 * y * y * drad_dr2 + 6.0 * p1 * y + 2.0 * p2 * x

        det = j11 * j22 - j12 * j21
        # Invert 2x2 system
        dx = (j22 * f_x - j12 * f_y) / det
        dy = (-j21 * f_x + j11 * f_y) / det

        x -= dx
        y -= dy

        max_step = float(np.max(np.hypot(dx, dy)))
        if max_step < tol:
            break

    return x, y


def distort_points(
    points_uv: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> np.ndarray:
    """Forward-distort pixel coordinates (u, v) -> (u_dist, v_dist)."""
    pts = np.asarray(points_uv, dtype=np.float64)
    u = pts[..., 0]
    v = pts[..., 1]

    x = (u - cx) / fx
    y = (v - cy) / fy

    xd, yd = distort_normalized(x, y, k1, k2, k3, p1, p2)
    ud = fx * xd + cx
    vd = fy * yd + cy

    return np.stack([ud, vd], axis=-1)


def undistort_points(
    points_uv_dist: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    max_iter: int = 15,
    tol: float = 1e-13,
) -> np.ndarray:
    """Undistort pixel coordinates (u_dist, v_dist) -> (u, v)."""
    pts = np.asarray(points_uv_dist, dtype=np.float64)
    ud = pts[..., 0]
    vd = pts[..., 1]

    xd = (ud - cx) / fx
    yd = (vd - cy) / fy

    x, y = undistort_normalized(xd, yd, k1, k2, k3, p1, p2, max_iter=max_iter, tol=tol)
    u = fx * x + cx
    v = fy * y + cy

    return np.stack([u, v], axis=-1)


def compute_distortion_backward_map(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute backward coordinate map for image warping and its validity mask.

    For every pixel in output (distorted) image, determines the undistorted source (map_u, map_v).

    Returns
    -------
    map_u : np.ndarray, shape (height, width), float64
        Undistorted horizontal source coordinates.
    map_v : np.ndarray, shape (height, width), float64
        Undistorted vertical source coordinates.
    validity_mask : np.ndarray, shape (height, width), bool
        True where source lies inside [-0.5, width - 0.5] x [-0.5, height - 0.5].
    """
    u_out, v_out = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    xd = (u_out - cx) / fx
    yd = (v_out - cy) / fy

    x_in, y_in = undistort_normalized(xd, yd, k1, k2, k3, p1, p2)
    map_u = fx * x_in + cx
    map_v = fy * y_in + cy

    # CONVENTIONS.md §5: image area spans [-0.5, W-0.5] x [-0.5, H-0.5]
    validity_mask = (
        (map_u >= -0.5)
        & (map_u <= width - 0.5)
        & (map_v >= -0.5)
        & (map_v <= height - 0.5)
    )

    return map_u, map_v, validity_mask


def warp_image_distort(
    image: np.ndarray,
    map_u: np.ndarray,
    map_v: np.ndarray,
    validity_mask: np.ndarray | None = None,
    method: str = "scipy",
) -> np.ndarray:
    """Warp linear image from undistorted to distorted using bicubic interpolation.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D linear input image.
    map_u : np.ndarray
        Horizontal source coordinate map.
    map_v : np.ndarray
        Vertical source coordinate map.
    validity_mask : np.ndarray or None
        Mask of valid source pixels. If provided, invalid pixels are set to 0.0.
    method : str
        'scipy' for float64 spline interpolation (highest subpixel accuracy),
        'cv2' for fast C++ fixed-point table remap.
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = map_u.shape

    if method == "scipy":
        coords = np.array([map_v.ravel(), map_u.ravel()])
        if img.ndim == 2:
            warped = map_coordinates(img, coords, order=3, mode="constant", cval=0.0).reshape((h, w))
        else:
            warped = np.empty((h, w, img.shape[2]), dtype=np.float64)
            for c in range(img.shape[2]):
                warped[..., c] = map_coordinates(img[..., c], coords, order=3, mode="constant", cval=0.0).reshape((h, w))
    else:
        # OpenCV remap (float32 maps)
        mu32 = map_u.astype(np.float32)
        mv32 = map_v.astype(np.float32)
        warped = cv2.remap(
            img,
            mu32,
            mv32,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    if validity_mask is not None:
        if warped.ndim == 2:
            warped[~validity_mask] = 0.0
        else:
            warped[~validity_mask] = 0.0

    return warped


def warp_mask_distort(
    mask: np.ndarray,
    map_u: np.ndarray,
    map_v: np.ndarray,
    validity_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Warp ground-truth mask with nearest-neighbour inverse mapping.

    Preserves label integer values and applies validity mask.
    """
    mu32 = map_u.astype(np.float32)
    mv32 = map_v.astype(np.float32)
    warped = cv2.remap(
        mask,
        mu32,
        mv32,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if validity_mask is not None:
        warped[~validity_mask] = 0
    return warped


def compute_max_displacement(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    grid_res: int = 100,
) -> float:
    """Compute maximum distortion displacement |distort(p) - p| over the image area."""
    u_grid = np.linspace(0.0, width - 1.0, grid_res)
    v_grid = np.linspace(0.0, height - 1.0, grid_res)
    u, v = np.meshgrid(u_grid, v_grid, indexing="xy")

    x = (u - cx) / fx
    y = (v - cy) / fy
    xd, yd = distort_normalized(x, y, k1, k2, k3, p1, p2)
    ud = fx * xd + cx
    vd = fy * yd + cy

    disp = np.hypot(ud - u, vd - v)
    return float(np.max(disp))


def compute_roi_grid_displacements(
    width: int,
    height: int,
    roi_w: int,
    roi_h: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    roi_step: int = 64,
) -> dict[str, float]:
    """Compute maximum displacement over each possible ROI position in the grid."""
    u0_vals = list(range(0, width - roi_w + 1, roi_step))
    if (width - roi_w) not in u0_vals:
        u0_vals.append(width - roi_w)

    v0_vals = list(range(0, height - roi_h + 1, roi_step))
    if (height - roi_h) not in v0_vals:
        v0_vals.append(height - roi_h)

    max_roi_disp = 0.0
    for v0 in v0_vals:
        for u0 in u0_vals:
            # Check corners and center of ROI
            u_roi = np.array([u0, u0 + roi_w - 1, u0, u0 + roi_w - 1, u0 + roi_w / 2.0])
            v_roi = np.array([v0, v0, v0 + roi_h - 1, v0 + roi_h - 1, v0 + roi_h / 2.0])
            x = (u_roi - cx) / fx
            y = (v_roi - cy) / fy
            xd, yd = distort_normalized(x, y, k1, k2, k3, p1, p2)
            ud = fx * xd + cx
            vd = fy * yd + cy
            disp = np.max(np.hypot(ud - u_roi, vd - v_roi))
            if disp > max_roi_disp:
                max_roi_disp = float(disp)

    return {
        "max_roi_displacement_px": max_roi_disp,
        "roi_count_evaluated": len(u0_vals) * len(v0_vals),
    }
