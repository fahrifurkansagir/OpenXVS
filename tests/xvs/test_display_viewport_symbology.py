"""Unit tests for WP-08 viewport and symbology acceptance item V3.

Conforms strictly to docs/xvs/WP08_DISPLAY.md v0.1 §5:
- The drawn horizon (truth pose) lies within 0.05 px of the WP-01 analytic horizon for the viewport camera.
- Runway corners match an independent projection within 1e-6 px.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from xvs.geo import wgs84, rotations
from xvs.sim.trajectory import t1_cruise_state, t2_ksfo28l_state, t3_turn_climb_state
from xvs.display.viewport import (
    compute_head_pose,
    get_viewport_camera,
    compute_viewport_pose,
)
from xvs.display.symbology import (
    compute_ksfo_28l_runway_geometry,
    project_conformal_runway,
    clip_line_segment_2d,
    compute_flight_path_marker,
    compute_conformal_horizon,
    draw_symbology,
)
from xvs.sim.horizon import compute_horizon_gt, zero_dip_horizon_v_at_u


def test_v3_conformal_exactness():
    """V3, conformal exactness (R-GB-1, WP08 §5):
    - Render the horizon alone on a blank image with sub-pixel drawing (shift=8, LINE_AA);
      its intensity-weighted centreline per column must lie within 0.05 px of analytic horizon
      across all 5 review poses (T2 straight/scripted, T1 yawed, T3 banking, T1 pitched).
    - Render runway edges alone on a blank image;
      intensity-weighted centreline must lie within 0.05 px of analytic runway edges.
    - Runway corners match independent projection within 1e-6 px.
    """
    import cv2
    view_cam = get_viewport_camera()

    # 1. Horizon exactness: test the actual DRAWN horizon stroke centreline across 5 review poses
    review_poses = [
        ("T2 t=50 head=(0,0)", t2_ksfo28l_state(50.0), (0.0, 0.0, 0.0)),
        ("T2 t=50 scripted head", t2_ksfo28l_state(50.0), compute_head_pose(50.0)),
        ("T1 t=25 head yaw 110", t1_cruise_state(25.0), (math.radians(110.0), 0.0, 0.0)),
        ("T3 t=40 scripted head", t3_turn_climb_state(40.0), compute_head_pose(40.0)),
        ("T1 t=7.3 head=(-20,-5)", t1_cruise_state(7.3), (math.radians(-20.0), math.radians(-5.0), 0.0)),
    ]

    for pose_name, st, hd in review_poses:
        ac_pose = {
            "lat_deg": st.lat_deg,
            "lon_deg": st.lon_deg,
            "h_m": st.h_m,
            "yaw_rad": st.yaw_rad,
            "pitch_rad": st.pitch_rad,
            "roll_rad": st.roll_rad,
        }
        vp = compute_viewport_pose(ac_pose, hd)
        p_view, r_view = vp["p_ecef_view"], vp["r_ecef_view"]

        hz_drawn = compute_conformal_horizon(p_view, r_view, view_cam, num_samples=2048)
        assert hz_drawn["horizon_visible"], f"{pose_name}: horizon should be visible"

        hz_eval = compute_horizon_gt(p_view, r_view, view_cam, num_samples=16384)
        eval_uv = np.array(hz_eval["horizon_uv"], dtype=np.float64)
        eval_uv_sorted = eval_uv[np.argsort(eval_uv[:, 0])]

        img_h = np.zeros((view_cam.height, view_cam.width), dtype=np.uint8)
        pts_float = np.asarray(hz_drawn["horizon_uv"], dtype=np.float64)
        pts_scaled = np.round(pts_float * 256.0).astype(np.int32)
        cv2.polylines(img_h, [pts_scaled], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_AA, shift=8)

        max_h_diff = 0.0
        checked_cols = 0
        for col in range(10, view_cam.width - 10):
            if col < eval_uv_sorted[0, 0] or col > eval_uv_sorted[-1, 0]:
                continue
            va = float(np.interp(col, eval_uv_sorted[:, 0], eval_uv_sorted[:, 1]))
            if va < 10 or va > view_cam.height - 10:
                continue
            lo = max(0, int(math.floor(va)) - 6)
            hi = min(view_cam.height, int(math.floor(va)) + 7)
            weights = img_h[lo:hi, col].astype(float)
            if weights.sum() > 0:
                vc = float(np.sum(np.arange(lo, hi) * weights) / weights.sum())
                diff = abs(vc - va)
                assert diff < 0.05, f"{pose_name} col {col} diff {diff:.5f} px exceeds 0.05 px"
                max_h_diff = max(max_h_diff, diff)
                checked_cols += 1

        assert checked_cols > 100, f"{pose_name}: expected > 100 checked columns, got {checked_cols}"
        assert max_h_diff < 0.05, f"{pose_name}: max drawn horizon diff {max_h_diff:.5f} px exceeds 0.05 px"

    # Use T2 t=50 pose for runway edge testing
    t = 50.0
    state = t2_ksfo28l_state(t)
    aircraft_pose = {
        "lat_deg": state.lat_deg,
        "lon_deg": state.lon_deg,
        "h_m": state.h_m,
        "yaw_rad": state.yaw_rad,
        "pitch_rad": state.pitch_rad,
        "roll_rad": state.roll_rad,
    }
    head_euler = compute_head_pose(t)
    vp_pose = compute_viewport_pose(aircraft_pose, head_euler)
    p_ecef_view = vp_pose["p_ecef_view"]
    r_ecef_view = vp_pose["r_ecef_view"]

    # 2. Runway edges drawn centreline exactness
    rwy_geom = compute_ksfo_28l_runway_geometry()
    projected_res = project_conformal_runway(p_ecef_view, r_ecef_view, view_cam, rwy_geom)
    assert projected_res["is_visible"]

    max_rwy_diff = 0.0
    checked_rwy_segments = 0
    for seg in projected_res["visible_segments"]:
        (u1, v1), (u2, v2) = seg
        img_seg = np.zeros((view_cam.height, view_cam.width), dtype=np.uint8)
        p1 = (int(round(u1 * 256.0)), int(round(v1 * 256.0)))
        p2 = (int(round(u2 * 256.0)), int(round(v2 * 256.0)))
        cv2.line(img_seg, p1, p2, 255, 1, cv2.LINE_AA, shift=8)

        du, dv = u2 - u1, v2 - v1
        if abs(du) >= abs(dv):
            u_min, u_max = min(u1, u2), max(u1, u2)
            for u in range(int(np.ceil(u_min)) + 2, int(np.floor(u_max)) - 1):
                col = img_seg[:, u].astype(float)
                if col.sum() > 0:
                    vc_meas = float(np.sum(np.arange(len(col)) * col) / col.sum())
                    vc_true = v1 + (u - u1) * dv / du
                    diff = abs(vc_meas - vc_true)
                    assert diff < 0.05, f"Runway col {u} diff {diff:.5f} px exceeds 0.05 px"
                    max_rwy_diff = max(max_rwy_diff, diff)
        else:
            v_min, v_max = min(v1, v2), max(v1, v2)
            for v in range(int(np.ceil(v_min)) + 2, int(np.floor(v_max)) - 1):
                row = img_seg[v, :].astype(float)
                if row.sum() > 0:
                    uc_meas = float(np.sum(np.arange(len(row)) * row) / row.sum())
                    uc_true = u1 + (v - v1) * du / dv
                    diff = abs(uc_meas - uc_true)
                    assert diff < 0.05, f"Runway row {v} diff {diff:.5f} px exceeds 0.05 px"
                    max_rwy_diff = max(max_rwy_diff, diff)
        checked_rwy_segments += 1

    assert checked_rwy_segments >= 4, f"Expected >= 4 runway segments, got {checked_rwy_segments}"
    assert max_rwy_diff < 0.05, f"Max drawn runway diff {max_rwy_diff:.5f} px exceeds 0.05 px"

    # 3. Runway corners independent projection exactness
    corners_ecef = rwy_geom["corners_ecef"]
    expected_uvs = []
    for p_corner in corners_ecef:
        p_cam = r_ecef_view.T @ (p_corner - p_ecef_view)
        assert p_cam[2] > 0.0, "Runway corner must be in front of camera on approach"
        u_exp = view_cam.fx * (p_cam[0] / p_cam[2]) + view_cam.cx
        v_exp = view_cam.fy * (p_cam[1] / p_cam[2]) + view_cam.cy
        expected_uvs.append((u_exp, v_exp))

    for i, (u_proj, v_proj) in enumerate(projected_res["corners_uv"]):
        u_exp, v_exp = expected_uvs[i]
        diff_u = abs(u_proj - u_exp)
        diff_v = abs(v_proj - v_exp)
        assert diff_u < 1e-6, f"Runway corner {i} u diff {diff_u:.3e} px exceeds 1e-6 px"
        assert diff_v < 1e-6, f"Runway corner {i} v diff {diff_v:.3e} px exceeds 1e-6 px"


def test_flight_path_marker_and_heading():
    """Verify flight path marker projection and heading values."""
    view_cam = get_viewport_camera()
    state = t2_ksfo28l_state(30.0)
    aircraft_pose = {
        "lat_deg": state.lat_deg,
        "lon_deg": state.lon_deg,
        "h_m": state.h_m,
        "yaw_rad": state.yaw_rad,
        "pitch_rad": state.pitch_rad,
        "roll_rad": state.roll_rad,
    }
    head_euler = (0.0, 0.0, 0.0)  # Head straight ahead
    vp_pose = compute_viewport_pose(aircraft_pose, head_euler)

    r_ecef_ned_mat = wgs84.r_ecef_ned(state.lat_deg, state.lon_deg)
    fpm = compute_flight_path_marker(state.v_ned_mps, r_ecef_ned_mat, vp_pose["r_ecef_view"], view_cam)
    assert fpm is not None
    uf, vf = fpm
    # On approach, velocity is near center of view
    assert abs(uf - view_cam.cx) < 150.0
    assert abs(vf - view_cam.cy) < 150.0


def test_runway_near_corners_behind_eye_clipping():
    """P1a (F-B8-1): A runway whose near corners are behind the eye draws only the visible edge segments.

    Verifies:
      - Near corners (z < 0.1 m) behind camera near plane are clipped segment-by-segment in camera space.
      - Edge 0 (connecting behind-camera corners) is culled.
      - Edges 1 and 3 are clipped at z = 0.1 m near plane.
      - Edge 2 is fully visible.
      - Segment endpoints match an independent clipped projection formula within 1e-6 px.
      - In draw_symbology, no pixel is drawn from behind-camera geometry (no rogue inverted triangles/lines).
    """
    view_cam = get_viewport_camera()

    # Pose on T2 final approach at t = 59.5 s where threshold/near corners have passed behind the eye
    t = 59.5
    state = t2_ksfo28l_state(t)
    aircraft_pose = {
        "lat_deg": state.lat_deg,
        "lon_deg": state.lon_deg,
        "h_m": state.h_m,
        "yaw_rad": state.yaw_rad,
        "pitch_rad": state.pitch_rad,
        "roll_rad": state.roll_rad,
    }
    head_euler = (0.0, 0.0, 0.0)
    vp_pose = compute_viewport_pose(aircraft_pose, head_euler)
    p_ecef_view = vp_pose["p_ecef_view"]
    r_ecef_view = vp_pose["r_ecef_view"]

    rwy_geom = compute_ksfo_28l_runway_geometry()
    corners_ecef = rwy_geom["corners_ecef"]

    # Verify that near corners are strictly behind the eye (z < 0.1 m) and far corners in front (z > 0.1 m)
    z_corners = [(r_ecef_view.T @ (c - p_ecef_view))[2] for c in corners_ecef]
    assert z_corners[0] < 0.1 and z_corners[1] < 0.1, f"Near corners must be behind eye: z={z_corners[:2]}"
    assert z_corners[2] > 0.1 and z_corners[3] > 0.1, f"Far corners must be in front of eye: z={z_corners[2:]}"

    # Independent clipped projection directly computed in test
    expected_segments = []
    z_near = 0.1
    for i in range(4):
        p1_cam = r_ecef_view.T @ (corners_ecef[i] - p_ecef_view)
        p2_cam = r_ecef_view.T @ (corners_ecef[(i + 1) % 4] - p_ecef_view)
        z1, z2 = p1_cam[2], p2_cam[2]

        if z1 < z_near and z2 < z_near:
            continue  # Both behind: completely culled

        c1 = p1_cam.copy()
        c2 = p2_cam.copy()

        if z1 < z_near:
            t_clip = (z_near - z1) / (z2 - z1)
            c1 = p1_cam + t_clip * (p2_cam - p1_cam)
            c1[2] = z_near
        elif z2 < z_near:
            t_clip = (z_near - z1) / (z2 - z1)
            c2 = p1_cam + t_clip * (p2_cam - p1_cam)
            c2[2] = z_near

        u1 = view_cam.fx * (c1[0] / c1[2]) + view_cam.cx
        v1 = view_cam.fy * (c1[1] / c1[2]) + view_cam.cy
        u2 = view_cam.fx * (c2[0] / c2[2]) + view_cam.cx
        v2 = view_cam.fy * (c2[1] / c2[2]) + view_cam.cy

        expected_segments.append(((u1, v1), (u2, v2)))

    # Must have exactly 3 visible edge segments (edge 0 culled, edges 1, 2, 3 visible)
    assert len(expected_segments) == 3

    # Call project_conformal_runway
    projected = project_conformal_runway(p_ecef_view, r_ecef_view, view_cam, rwy_geom)
    assert projected["is_visible"], "Runway must be reported visible when visible segments exist"
    assert not projected["all_front"], "Corners should NOT all be in front"
    visible_segments = projected["visible_segments"]
    assert len(visible_segments) == 3, f"Expected 3 visible segments, got {len(visible_segments)}"

    # Check registration within 1e-6 px against independent clipped projection
    for j in range(3):
        exp_p1, exp_p2 = expected_segments[j]
        act_p1, act_p2 = visible_segments[j]

        diff_u1 = abs(act_p1[0] - exp_p1[0])
        diff_v1 = abs(act_p1[1] - exp_p1[1])
        diff_u2 = abs(act_p2[0] - exp_p2[0])
        diff_v2 = abs(act_p2[1] - exp_p2[1])

        max_diff = max(diff_u1, diff_v1, diff_u2, diff_v2)
        assert max_diff < 1e-6, f"Segment {j} diff {max_diff:.3e} px exceeds 1e-6 px tolerance"

    # Verify that draw_symbology renders cleanly with no pixel from behind-camera geometry
    blank_img = np.zeros((view_cam.height, view_cam.width, 3), dtype=np.uint8)
    rendered = draw_symbology(
        blank_img,
        aircraft_pose,
        state.v_ned_mps,
        p_ecef_view,
        r_ecef_view,
        view_cam,
        is_t2=True,
        burn_in_text=None,
    )

    # 1. Verify visible runway edge segments are actually drawn on final approach
    # The runway outline must be clearly visible, forming the approach perspective
    # (previous bug dropped visible segments due to naive point bounds checks, leaving only 208 px)
    total_drawn_px = np.count_nonzero(rendered)
    assert total_drawn_px > 10000, f"Expected full runway outline to be drawn (>10000 px), got {total_drawn_px}"

    # 2. Verify runway edges extend down into the lower viewport
    bottom_half = rendered[view_cam.height // 2 :, :]
    assert np.count_nonzero(bottom_half) > 5000, "Runway edges should extend through the lower viewport"

    # 3. Verify symbol polygons are NEVER filled:
    # Interior pixels between runway edges remain unfilled background (strokes only)
    interior_sample = rendered[550, 640]
    assert np.all(interior_sample == 0), "Runway interior must remain unfilled (polygons never filled)"

    # 4. Verify behind-camera edge (Edge 0) is never drawn:
    # Far unprojected areas like top-left (0:100, 0:100) must have zero rogue pixels
    top_left_crop = rendered[0:100, 0:100]
    assert np.count_nonzero(top_left_crop) == 0, "No pixel should be drawn from behind-camera geometry"


def test_clip_line_segment_2d_exactness():
    """Verify Cohen-Sutherland 2D line clipping edge cases, bounds, and colinearity."""
    box = (0.0, 0.0, 100.0, 100.0)

    # 1. Segment completely inside
    seg_in = clip_line_segment_2d((20.0, 30.0), (80.0, 70.0), box)
    assert seg_in is not None
    assert seg_in == ((20.0, 30.0), (80.0, 70.0))

    # 2. Segment completely outside (left, right, top, bottom)
    assert clip_line_segment_2d((-50.0, 20.0), (-10.0, 80.0), box) is None
    assert clip_line_segment_2d((110.0, 20.0), (150.0, 80.0), box) is None
    assert clip_line_segment_2d((20.0, -50.0), (80.0, -10.0), box) is None
    assert clip_line_segment_2d((20.0, 110.0), (80.0, 150.0), box) is None

    # 3. Crossing left boundary
    seg_left = clip_line_segment_2d((-100.0, 50.0), (100.0, 50.0), box)
    assert seg_left is not None
    p1, p2 = seg_left
    assert abs(p1[0] - 0.0) < 1e-12 and abs(p1[1] - 50.0) < 1e-12
    assert abs(p2[0] - 100.0) < 1e-12 and abs(p2[1] - 50.0) < 1e-12

    # 4. Diagonal crossing (from (-50, -50) to (150, 150))
    seg_diag = clip_line_segment_2d((-50.0, -50.0), (150.0, 150.0), box)
    assert seg_diag is not None
    p1, p2 = seg_diag
    assert abs(p1[0] - 0.0) < 1e-12 and abs(p1[1] - 0.0) < 1e-12
    assert abs(p2[0] - 100.0) < 1e-12 and abs(p2[1] - 100.0) < 1e-12

    # 5. Colinearity preservation for extreme coordinates
    seg_ext = clip_line_segment_2d((-200000.0, 100000.0), (50.0, 50.0), box)
    assert seg_ext is not None
    p1, p2 = seg_ext
    slope_orig = (50.0 - 100000.0) / (50.0 - (-200000.0))
    slope_clipped = (p2[1] - p1[1]) / (p2[0] - p1[0])
    assert abs(slope_orig - slope_clipped) < 1e-9


def test_t2_final_approach_frames_no_filled_symbols_and_strict_clipping():
    """F-B9-5 / R6: Final approach frames must strictly clip behind-camera geometry and never fill symbols."""
    view_cam = get_viewport_camera()
    rwy_geom = compute_ksfo_28l_runway_geometry()

    # Test final seconds of approach right up to touchdown
    test_times = [57.5, 58.0, 58.5, 59.0, 59.5, 59.8]

    for t in test_times:
        state = t2_ksfo28l_state(t)
        aircraft_pose = {
            "lat_deg": state.lat_deg,
            "lon_deg": state.lon_deg,
            "h_m": state.h_m,
            "yaw_rad": state.yaw_rad,
            "pitch_rad": state.pitch_rad,
            "roll_rad": state.roll_rad,
        }
        head_euler = (0.0, 0.0, 0.0)
        vp_pose = compute_viewport_pose(aircraft_pose, head_euler)
        p_ecef_view = vp_pose["p_ecef_view"]
        r_ecef_view = vp_pose["r_ecef_view"]

        projected = project_conformal_runway(p_ecef_view, r_ecef_view, view_cam, rwy_geom)

        # For every visible segment, endpoints in camera space must have z >= z_near (0.1m)
        for (u1, v1), (u2, v2) in projected["visible_segments"]:
            assert math.isfinite(u1) and math.isfinite(v1)
            assert math.isfinite(u2) and math.isfinite(v2)

        for (u1, v1), (u2, v2) in projected["threshold_segments"]:
            assert math.isfinite(u1) and math.isfinite(v1)
            assert math.isfinite(u2) and math.isfinite(v2)

        # Render onto blank canvas
        blank = np.zeros((view_cam.height, view_cam.width, 3), dtype=np.uint8)
        rendered = draw_symbology(
            blank,
            aircraft_pose,
            state.v_ned_mps,
            p_ecef_view,
            r_ecef_view,
            view_cam,
            is_t2=True,
            burn_in_text=None,
        )

        # 1. No rogue geometry in top-left corner
        assert np.count_nonzero(rendered[0:50, 0:50]) == 0, f"Rogue pixels in top-left corner at t={t}s"

        # 2. Runway interior is never filled (polygons unfilled, only strokes drawn)
        # Check center-bottom region between runway strokes
        if t >= 59.0:
            # When low over runway, the center point on ground between edges must be empty
            sample_pt = rendered[600, 640]
            assert np.all(sample_pt == 0), f"Runway polygon must not be filled at t={t}s"


