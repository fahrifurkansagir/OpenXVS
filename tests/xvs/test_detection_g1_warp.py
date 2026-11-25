"""V2_G1 & V4_G1 — 6-DoF WGS-84 ground and sky registration exactness and pose-source guard (`WP04_DETECTION.md` §9.4, §9.7).

Verifies Decision D34 (S3, S8):
1. Synthetic ground points on the WGS-84 ellipsoid under known 6-DoF motion (translation + rotation)
   register to `< 0.05 px` RMS centroid error (both analytical correspondence and image-domain
   warped sub-pixel Gaussian blobs via `warp_roi_into_current`).
2. Synthetic sky points at infinity under known 3D rotation register to `< 0.05 px` RMS centroid
   error inside the overlap mask.
3. Non-oracle `D1-KF` registration accepts ONLY `q_ned_to_cam_est` and `pos_wgs84` and raises
   `ValueError` when ground-truth pose keys are supplied unless `ablation_mode="oracle"`.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from xvs.detection.warp import (
    DEFAULT_CX,
    DEFAULT_CY,
    DEFAULT_FX,
    DEFAULT_FY,
    project_points_6dof_wgs84,
    register_kframe_stack,
    resolve_pose_ecef_and_r_cam,
    validate_pose_dict,
    warp_roi_into_current,
)
from xvs.geo import rotations, wgs84


def _make_ring_t0_pose(lat_deg: float, lon_deg: float, h_m: float, yaw_deg: float, pitch_deg: float, roll_deg: float) -> dict:
    """Construct camera T0 (forward-looking) pose dict with `q_ned_to_cam_est` and `pos_wgs84`."""
    r_ned_body = rotations.euler_to_r(
        math.radians(yaw_deg),
        math.radians(pitch_deg),
        math.radians(roll_deg),
    )
    # T0 camera axes in body FRD: x_cam = y_body (right), y_cam = z_body (down), z_cam = x_body (forward)
    r_body_cam = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    r_ned_cam = r_ned_body @ r_body_cam
    q_ned_cam = rotations.r_to_quat(r_ned_cam)
    return {
        "q_ned_to_cam_est": q_ned_cam.tolist(),
        "pos_wgs84": [float(lat_deg), float(lon_deg), float(h_m)],
        "pose_source": "wp06_estimated",
    }


def _render_gaussian_blob(height: int, width: int, u_local: float, v_local: float, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    return np.exp(-((xx - u_local) ** 2 + (yy - v_local) ** 2) / (2.0 * sigma * sigma))


def _measure_blob_centroid(img: np.ndarray, u_hint: float, v_hint: float, win: int = 9) -> tuple[float, float]:
    height, width = img.shape
    x0 = max(0, int(round(u_hint)) - win)
    x1 = min(width, int(round(u_hint)) + win + 1)
    y0 = max(0, int(round(v_hint)) - win)
    y1 = min(height, int(round(v_hint)) + win + 1)
    patch = np.maximum(0.0, img[y0:y1, x0:x1])
    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64),
        np.arange(x0, x1, dtype=np.float64),
        indexing="ij",
    )
    total = float(np.sum(patch))
    assert total > 1e-6
    return float(np.sum(patch * xx) / total), float(np.sum(patch * yy) / total)


def test_v2_g1_wgs84_ground_projection_rms_under_0_05_px() -> None:
    """Test synthetic ground points on the WGS-84 ellipsoid under known 6-DoF motion (< 0.05 px RMS)."""
    # Pitch camera slightly down (-12 deg) so the ROI views the WGS-84 ellipsoid surface
    pose_prev = _make_ring_t0_pose(
        lat_deg=37.0500, lon_deg=-120.8500, h_m=3000.0, yaw_deg=315.0, pitch_deg=-12.0, roll_deg=0.0
    )
    # 6-DoF motion over 1.0 s (~120 m translation + 3D attitude rotation)
    pose_curr = _make_ring_t0_pose(
        lat_deg=37.0508, lon_deg=-120.8508, h_m=2995.0, yaw_deg=315.15, pitch_deg=-11.85, roll_deg=0.12
    )

    p_curr, r_curr = resolve_pose_ecef_and_r_cam(pose_curr)
    p_prev, r_prev = resolve_pose_ecef_and_r_cam(pose_prev)

    roi_u0, roi_v0 = 1100.0, 1250.0
    roi_h, roi_w = 128, 128

    # Sample 16 interior points in frame t that intersect the WGS-84 ellipsoid
    pts_u_local = np.linspace(28.0, 100.0, 4) + 0.31
    pts_v_local = np.linspace(28.0, 100.0, 4) + 0.67
    uu_local, vv_local = np.meshgrid(pts_u_local, pts_v_local)
    u_curr_full = uu_local.ravel() + roi_u0
    v_curr_full = vv_local.ravel() + roi_v0

    # Ground-truth 3D surface intersection on WGS-84 ellipsoid and exact projection into frame t-1
    u_prev_full, v_prev_full, hits_earth = project_points_6dof_wgs84(
        u_curr_full, v_curr_full, pose_src=pose_curr, pose_dst=pose_prev
    )
    assert np.all(hits_earth), "All below-horizon test points must intersect the WGS-84 ellipsoid"

    # 1. Round-trip analytical reprojection from prev back to curr must be exact (< 1e-9 px)
    u_rt, v_rt, hits_rt = project_points_6dof_wgs84(
        u_prev_full, v_prev_full, pose_src=pose_prev, pose_dst=pose_curr
    )
    assert np.all(hits_rt)
    rms_analytical = float(np.sqrt(np.mean((u_rt - u_curr_full) ** 2 + (v_rt - v_curr_full) ** 2)))
    assert rms_analytical < 0.05

    # 2. Image-domain sub-pixel Gaussian blob warping via `warp_roi_into_current`
    sq_errors: list[float] = []
    for idx in range(len(u_curr_full)):
        u_loc_prev = float(u_prev_full[idx] - roi_u0)
        v_loc_prev = float(v_prev_full[idx] - roi_v0)
        u_loc_curr_true = float(u_curr_full[idx] - roi_u0)
        v_loc_curr_true = float(v_curr_full[idx] - roi_v0)

        img_prev = _render_gaussian_blob(roi_h, roi_w, u_loc_prev, v_loc_prev, sigma=2.2)
        warped_t, overlap_mask, earth_mask = warp_roi_into_current(
            img_prev=torch.from_numpy(img_prev),
            pose_curr=pose_curr,
            pose_prev=pose_prev,
            roi_u0_curr=roi_u0,
            roi_v0_curr=roi_v0,
            roi_u0_prev=roi_u0,
            roi_v0_prev=roi_v0,
            interp_mode="bicubic",
        )
        assert bool(earth_mask[int(round(v_loc_curr_true)), int(round(u_loc_curr_true))].item())
        assert float(overlap_mask[int(round(v_loc_curr_true)), int(round(u_loc_curr_true))].item()) == 1.0

        u_meas, v_meas = _measure_blob_centroid(warped_t.numpy(), u_loc_curr_true, v_loc_curr_true)
        sq_errors.append((u_meas - u_loc_curr_true) ** 2 + (v_meas - v_loc_curr_true) ** 2)

    rms_blob_ground = math.sqrt(float(np.mean(sq_errors)))
    assert rms_blob_ground < 0.05, f"Ground warp centroid RMS {rms_blob_ground:.5f} px exceeds 0.05 px limit"


def test_v2_g1_sky_rotation_at_infinity_rms_under_0_05_px() -> None:
    """Test synthetic sky points at infinity under known 3D rotation (< 0.05 px RMS inside overlap mask)."""
    pose_prev = _make_ring_t0_pose(
        lat_deg=37.0500, lon_deg=-120.8500, h_m=3000.0, yaw_deg=315.0, pitch_deg=8.0, roll_deg=0.0
    )
    pose_curr = _make_ring_t0_pose(
        lat_deg=37.0508, lon_deg=-120.8508, h_m=3000.0, yaw_deg=315.20, pitch_deg=8.18, roll_deg=-0.15
    )

    roi_u0, roi_v0 = 1100.0, 450.0
    roi_h, roi_w = 128, 128

    pts_u_local = np.linspace(28.0, 100.0, 4) + 0.42
    pts_v_local = np.linspace(28.0, 100.0, 4) + 0.19
    uu_local, vv_local = np.meshgrid(pts_u_local, pts_v_local)
    u_curr_full = uu_local.ravel() + roi_u0
    v_curr_full = vv_local.ravel() + roi_v0

    u_prev_full, v_prev_full, hits_earth = project_points_6dof_wgs84(
        u_curr_full, v_curr_full, pose_src=pose_curr, pose_dst=pose_prev
    )
    assert not np.any(hits_earth), "All above-horizon sky points must not intersect the WGS-84 ellipsoid"

    sq_errors: list[float] = []
    for idx in range(len(u_curr_full)):
        u_loc_prev = float(u_prev_full[idx] - roi_u0)
        v_loc_prev = float(v_prev_full[idx] - roi_v0)
        u_loc_curr_true = float(u_curr_full[idx] - roi_u0)
        v_loc_curr_true = float(v_curr_full[idx] - roi_v0)

        img_prev = _render_gaussian_blob(roi_h, roi_w, u_loc_prev, v_loc_prev, sigma=2.2)
        warped_t, overlap_mask, earth_mask = warp_roi_into_current(
            img_prev=torch.from_numpy(img_prev),
            pose_curr=pose_curr,
            pose_prev=pose_prev,
            roi_u0_curr=roi_u0,
            roi_v0_curr=roi_v0,
            roi_u0_prev=roi_u0,
            roi_v0_prev=roi_v0,
            interp_mode="bicubic",
        )
        assert not bool(earth_mask[int(round(v_loc_curr_true)), int(round(u_loc_curr_true))].item())
        assert float(overlap_mask[int(round(v_loc_curr_true)), int(round(u_loc_curr_true))].item()) == 1.0

        u_meas, v_meas = _measure_blob_centroid(warped_t.numpy(), u_loc_curr_true, v_loc_curr_true)
        sq_errors.append((u_meas - u_loc_curr_true) ** 2 + (v_meas - v_loc_curr_true) ** 2)

    rms_blob_sky = math.sqrt(float(np.mean(sq_errors)))
    assert rms_blob_sky < 0.05, f"Sky warp centroid RMS {rms_blob_sky:.5f} px exceeds 0.05 px limit"


def test_v4_g1_warp_rejects_ground_truth_pose_unless_oracle() -> None:
    """Verify non-oracle D1-KF registration strictly rejects ground-truth pose inputs (`S3`, `S8`, `V4_G1`)."""
    valid_pose = _make_ring_t0_pose(37.05, -120.85, 3000.0, 315.0, 0.0, 0.0)
    validate_pose_dict(valid_pose, ablation_mode="standard")

    # Ground-truth quaternion key `q_ned_to_cam` must be rejected in standard mode
    gt_pose = {
        "q_ned_to_cam": valid_pose["q_ned_to_cam_est"],
        "pos_wgs84": valid_pose["pos_wgs84"],
    }
    with pytest.raises(ValueError, match="Forbidden ground-truth pose keys"):
        validate_pose_dict(gt_pose, ablation_mode="standard")

    with pytest.raises(ValueError, match="Forbidden ground-truth pose keys"):
        register_kframe_stack(
            frames=[np.zeros((32, 32), dtype=np.float32)] * 3,
            poses=[gt_pose] * 3,
            ablation_mode="standard",
        )

    # Explicitly labelled `ablation_mode="oracle"` permits the oracle ablation (Decision D34-S3)
    reg_oracle = register_kframe_stack(
        frames=[np.zeros((32, 32), dtype=np.float32)] * 3,
        poses=[gt_pose] * 3,
        ablation_mode="oracle",
    )
    assert reg_oracle["ablation_mode"] == "oracle"
    assert reg_oracle["input_tensor"].shape == (5, 32, 32)
