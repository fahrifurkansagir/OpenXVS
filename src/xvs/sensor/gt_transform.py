"""Ground-truth geometric transforms for labels under lens distortion.

Spec reference: docs/xvs/WP03_SENSOR.md §1 item 7 and §3 item V3(c).
- Points and polylines (projected centres, horizon GT) go through analytic forward distortion.
- Analytic boxes are recomputed from distorted projections of mesh vertices.
- Masks are warped with nearest-neighbour inverse mapping, and validity mask is kept.
"""

from __future__ import annotations

import numpy as np

from xvs.sensor.distortion import distort_points, warp_mask_distort


def transform_gt_points(
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
    """Forward-distort a set of ground-truth points (e.g. object centres)."""
    return distort_points(points_uv, fx, fy, cx, cy, k1, k2, k3, p1, p2)


def transform_gt_polyline(
    polyline_uv: np.ndarray,
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
    """Forward-distort a polyline (e.g. horizon GT points)."""
    poly = np.asarray(polyline_uv, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"Expected polyline of shape (N, 2), got {poly.shape}")
    return distort_points(poly, fx, fy, cx, cy, k1, k2, k3, p1, p2)


def transform_mesh_box(
    mesh_vertices_uv: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> tuple[np.ndarray, list[float]]:
    """Recompute analytic 2D bounding box from distorted projections of 3D mesh vertices.

    Parameters
    ----------
    mesh_vertices_uv : np.ndarray, shape (N, 2)
        Projected undistorted mesh vertices in pixel coordinates.

    Returns
    -------
    distorted_vertices : np.ndarray, shape (N, 2)
        Distorted vertex pixel coordinates.
    box_edges : list of float [u_min, v_min, u_max, v_max]
        Exact bounding box of the distorted vertices.
    """
    verts = np.asarray(mesh_vertices_uv, dtype=np.float64)
    if verts.ndim == 1 and verts.shape[0] == 2:
        verts = verts.reshape((1, 2))
    elif verts.ndim != 2 or (verts.size > 0 and verts.shape[1] != 2):
        raise ValueError(f"Expected mesh_vertices_uv of shape (N, 2), got {verts.shape}")
    if verts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), []
    dist_verts = distort_points(verts, fx, fy, cx, cy, k1, k2, k3, p1, p2)
    u_min = float(np.min(dist_verts[:, 0]))
    v_min = float(np.min(dist_verts[:, 1]))
    u_max = float(np.max(dist_verts[:, 0]))
    v_max = float(np.max(dist_verts[:, 1]))
    return dist_verts, [u_min, v_min, u_max, v_max]


def transform_gt_mask(
    mask: np.ndarray,
    map_u: np.ndarray,
    map_v: np.ndarray,
    validity_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int] | None]:
    """Warp ground-truth mask with nearest-neighbour inverse mapping.

    Returns
    -------
    warped_mask : np.ndarray
        Distorted binary/label mask.
    mask_box : list of int [u_min, v_min, u_max, v_max] or None if mask is empty
    """
    warped = warp_mask_distort(mask, map_u, map_v, validity_mask=validity_mask)
    y_idx, x_idx = np.where(warped > 0)
    if len(y_idx) == 0:
        return warped, None
    box = [int(x_idx.min()), int(y_idx.min()), int(x_idx.max()), int(y_idx.max())]
    return warped, box
