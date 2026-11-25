"""Tests for lens distortion, inversion, image warping and GT transforms (V3).

Spec reference: docs/xvs/WP03_SENSOR.md §3 item V3 (a-d).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from xvs.sensor.distortion import (
    distort_normalized,
    undistort_normalized,
    distort_points,
    undistort_points,
    compute_distortion_backward_map,
    warp_image_distort,
    warp_mask_distort,
    compute_max_displacement,
    compute_roi_grid_displacements,
)
from xvs.sensor.gt_transform import (
    transform_gt_points,
    transform_gt_polyline,
    transform_mesh_box,
    transform_gt_mask,
)


PRESETS = [
    {
        "name": "ring_mono_v0",
        "W": 2448,
        "H": 2048,
        "fx": 2319.0,
        "fy": 2319.0,
        "cx": 1223.5,
        "cy": 1023.5,
        "k1": -0.05,
        "k2": 0.01,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    },
    {
        "name": "rig_mono_v0",
        "W": 1024,
        "H": 1024,
        "fx": 429.6190111627674,
        "fy": 429.6190111627674,
        "cx": 511.5,
        "cy": 511.5,
        "k1": -0.10,
        "k2": 0.02,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    },
]


@pytest.mark.parametrize("p", PRESETS)
def test_v3a_roundtrip_precision(p):
    """V3(a): Forward o inverse round trip <= 1e-6 px on a 64x64 grid over full image."""
    u_grid = np.linspace(0.0, p["W"] - 1.0, 64)
    v_grid = np.linspace(0.0, p["H"] - 1.0, 64)
    u, v = np.meshgrid(u_grid, v_grid, indexing="xy")
    pts_undist = np.stack([u.ravel(), v.ravel()], axis=-1)

    pts_dist = distort_points(
        pts_undist, p["fx"], p["fy"], p["cx"], p["cy"],
        p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
    )
    pts_rec = undistort_points(
        pts_dist, p["fx"], p["fy"], p["cx"], p["cy"],
        p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
    )

    err = np.hypot(pts_rec[:, 0] - pts_undist[:, 0], pts_rec[:, 1] - pts_undist[:, 1])
    max_err = float(np.max(err))
    assert max_err <= 1e-6, f"[{p['name']}] Round-trip error {max_err:.2e} px exceeds 1e-6 px"


@pytest.mark.parametrize("p", PRESETS)
def test_v3b_gaussian_blobs_centroid_accuracy(p):
    """V3(b): Gaussian blobs (sigma = 1.5 px) centroids vs analytic distorted positions.

    Median <= 0.02 px, max <= 0.05 px. Measure only blobs whose 4*sigma neighbourhood is fully valid.
    """
    # For fast testing, use 512x512 domain centered around optical center
    domain_w, domain_h = min(512, p["W"]), min(512, p["H"])
    offset_u = (p["W"] - domain_w) // 2
    offset_v = (p["H"] - domain_h) // 2

    # Compute backward map for this domain
    u_out, v_out = np.meshgrid(np.arange(domain_w) + offset_u, np.arange(domain_h) + offset_v, indexing="xy")
    xd = (u_out - p["cx"]) / p["fx"]
    yd = (v_out - p["cy"]) / p["fy"]
    x_in, y_in = undistort_normalized(xd, yd, p["k1"], p["k2"], p["k3"], p["p1"], p["p2"])
    map_u = p["fx"] * x_in + p["cx"]
    map_v = p["fy"] * y_in + p["cy"]

    validity_mask = (
        (map_u >= -0.5) & (map_u <= p["W"] - 0.5) &
        (map_v >= -0.5) & (map_v <= p["H"] - 0.5)
    )

    # Place Gaussian blobs
    sigma = 1.5
    grid_u = np.arange(offset_u + 60, offset_u + domain_w - 60, 50)
    grid_v = np.arange(offset_v + 60, offset_v + domain_h - 60, 50)

    img_undist = np.zeros((p["H"], p["W"]), dtype=np.float64)
    blob_centers = []
    rad = int(np.ceil(6 * sigma))
    for v0 in grid_v:
        for u0 in grid_u:
            uu, vv = np.meshgrid(np.arange(u0 - rad, u0 + rad + 1), np.arange(v0 - rad, v0 + rad + 1))
            blob = np.exp(-((uu - u0) ** 2 + (vv - v0) ** 2) / (2.0 * sigma ** 2))
            img_undist[vv, uu] += blob
            blob_centers.append((u0, v0))

    # Warp image (bicubic float64)
    warped = warp_image_distort(img_undist, map_u, map_v, validity_mask=validity_mask, method="scipy")

    errors = []
    win_r = 7
    for u0, v0 in blob_centers:
        pt_dist = distort_points(
            np.array([[u0, v0]]), p["fx"], p["fy"], p["cx"], p["cy"],
            p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
        )[0]
        ud_true, vd_true = pt_dist[0], pt_dist[1]

        # Domain relative coordinates
        ud_rel = ud_true - offset_u
        vd_rel = vd_true - offset_v

        # Check 4*sigma neighborhood validity
        r_check = int(np.ceil(4 * sigma))
        u_min = int(np.round(ud_rel)) - r_check
        u_max = int(np.round(ud_rel)) + r_check
        v_min = int(np.round(ud_rel)) - r_check
        v_max = int(np.round(ud_rel)) + r_check

        if u_min < 0 or u_max >= domain_w or v_min < 0 or v_max >= domain_h:
            continue
        if not np.all(validity_mask[v_min : v_max + 1, u_min : u_max + 1]):
            continue

        cu_idx = int(np.round(ud_rel))
        cv_idx = int(np.round(vd_rel))
        sub_u, sub_v = np.meshgrid(
            np.arange(cu_idx - win_r, cu_idx + win_r + 1) + offset_u,
            np.arange(cv_idx - win_r, cv_idx + win_r + 1) + offset_v,
            indexing="xy",
        )
        weights = np.maximum(0.0, warped[cv_idx - win_r : cv_idx + win_r + 1, cu_idx - win_r : cu_idx + win_r + 1])
        total_w = np.sum(weights)
        if total_w > 0:
            c_u = np.sum(sub_u * weights) / total_w
            c_v = np.sum(sub_v * weights) / total_w
            err = np.hypot(c_u - ud_true, c_v - vd_true)
            errors.append(err)

    assert len(errors) > 0, "No valid blobs tested"
    errors = np.array(errors)
    med_err = float(np.median(errors))
    max_err = float(np.max(errors))

    assert med_err <= 0.02, f"[{p['name']}] Median centroid error {med_err:.4f} px exceeds 0.02 px"
    assert max_err <= 0.05, f"[{p['name']}] Max centroid error {max_err:.4f} px exceeds 0.05 px"


def test_v3c_synthetic_polygon_mask_warp_against_analytic_vertices():
    """V3(c): Warp a synthetic polygon mask.

    The box of the warped mask against the box of the analytically distorted polygon vertices:
    each edge within 1 px.
    """
    p = PRESETS[1]  # rig_mono_v0
    map_u, map_v, valid_mask = compute_distortion_backward_map(
        p["W"], p["H"], p["fx"], p["fy"], p["cx"], p["cy"],
        p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
    )

    # Polygon vertices in undistorted coordinates
    poly = np.array([[400.0, 350.0], [550.0, 320.0], [600.0, 480.0], [520.0, 560.0], [380.0, 500.0]])
    mask = np.zeros((p["H"], p["W"]), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

    warped_mask, mask_box = transform_gt_mask(mask, map_u, map_v, validity_mask=valid_mask)
    assert mask_box is not None

    _, analytic_box = transform_mesh_box(
        poly, p["fx"], p["fy"], p["cx"], p["cy"],
        p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
    )

    # Compare edge bounds: [u_min, v_min, u_max, v_max]
    edge_diffs = [
        abs(mask_box[0] - analytic_box[0]),
        abs(mask_box[1] - analytic_box[1]),
        abs(mask_box[2] - analytic_box[2]),
        abs(mask_box[3] - analytic_box[3]),
    ]
    for i, diff in enumerate(edge_diffs):
        assert diff <= 1.0, f"Edge {i} difference {diff:.3f} px exceeds 1 px tolerance"


def test_v3d_displacement_reporting():
    """V3(d): Compute largest displacement over full ring image and ROI grid."""
    p_ring = PRESETS[0]
    max_disp_ring = compute_max_displacement(
        p_ring["W"], p_ring["H"], p_ring["fx"], p_ring["fy"], p_ring["cx"], p_ring["cy"],
        p_ring["k1"], p_ring["k2"], p_ring["k3"], p_ring["p1"], p_ring["p2"]
    )
    assert max_disp_ring > 20.0  # Around ~34 px
    assert max_disp_ring < 50.0

    # 1024^2 ROI displacement grid
    roi_info = compute_roi_grid_displacements(
        p_ring["W"], p_ring["H"], 1024, 1024,
        p_ring["fx"], p_ring["fy"], p_ring["cx"], p_ring["cy"],
        p_ring["k1"], p_ring["k2"], p_ring["k3"], p_ring["p1"], p_ring["p2"],
        roi_step=128,
    )
    assert roi_info["max_roi_displacement_px"] > 0.0
    assert roi_info["roi_count_evaluated"] > 0


def test_gt_transform_edge_cases():
    """Verify robust handling of 1D, empty and invalid vertex inputs in gt_transform."""
    p = PRESETS[1]

    # 1D single vertex: shape (2,)
    single_vert = np.array([500.0, 500.0])
    dist_v, box = transform_mesh_box(
        single_vert, p["fx"], p["fy"], p["cx"], p["cy"],
        p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
    )
    assert dist_v.shape == (1, 2)
    assert len(box) == 4
    assert box[0] == box[2] and box[1] == box[3]

    # Empty vertices: shape (0, 2)
    empty_verts = np.empty((0, 2))
    dist_empty, box_empty = transform_mesh_box(
        empty_verts, p["fx"], p["fy"], p["cx"], p["cy"],
        p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
    )
    assert dist_empty.shape == (0, 2)
    assert box_empty == []

    # Invalid vertex shape
    with pytest.raises(ValueError, match="Expected mesh_vertices_uv of shape"):
        transform_mesh_box(
            np.ones((5, 3)), p["fx"], p["fy"], p["cx"], p["cy"],
            p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
        )

    # Invalid polyline shape
    with pytest.raises(ValueError, match="Expected polyline of shape"):
        transform_gt_polyline(
            np.ones((5, 3)), p["fx"], p["fy"], p["cx"], p["cy"],
            p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
        )

    # Empty mask
    map_u, map_v, valid_mask = compute_distortion_backward_map(
        100, 100, p["fx"], p["fy"], p["cx"], p["cy"],
        p["k1"], p["k2"], p["k3"], p["p1"], p["p2"]
    )
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    warped, m_box = transform_gt_mask(empty_mask, map_u, map_v, validity_mask=valid_mask)
    assert m_box is None
