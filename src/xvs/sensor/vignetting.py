"""Optical vignetting model.

Spec reference: docs/xvs/WP03_SENSOR.md §1 item 4 and §3 item V4.
- V = cos^4(theta) * (1 + v1*r^2 + v2*r^4)
- theta is the field angle of the pixel ray.
- r is the normalized radius.
"""

from __future__ import annotations

import numpy as np


def compute_vignetting_map(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    v1: float = 0.0,
    v2: float = 0.0,
) -> np.ndarray:
    """Compute 2D optical vignetting transmission map V(u, v).

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point coordinates in pixels.
    v1, v2 : float
        Radial polynomial vignetting coefficients.

    Returns
    -------
    np.ndarray, shape (height, width), dtype float64
        Vignetting transmission factor V in (0, 1].
    """
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    x = (u_grid - cx) / fx
    y = (v_grid - cy) / fy

    r2 = x * x + y * y
    cos4_theta = 1.0 / ((1.0 + r2) ** 2)
    poly = 1.0 + v1 * r2 + v2 * (r2 ** 2)

    vignetting = cos4_theta * poly
    return np.maximum(0.0, vignetting)


def apply_vignetting(image: np.ndarray, vignetting_map: np.ndarray) -> np.ndarray:
    """Apply vignetting transmission to a 2D or 3D linear radiance image.

    Parameters
    ----------
    image : np.ndarray
        Linear radiance image.
    vignetting_map : np.ndarray
        2D vignetting map of matching (height, width).

    Returns
    -------
    np.ndarray
        Vignetted linear radiance image.
    """
    img = np.asarray(image, dtype=np.float64)
    v_map = np.asarray(vignetting_map, dtype=np.float64)

    if img.ndim == 2:
        return img * v_map
    elif img.ndim == 3:
        return img * v_map[..., None]
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")
