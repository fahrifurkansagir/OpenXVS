"""Two-layer 6-DoF WGS-84 ground-projection and sky-rotation registration (WP-04 §9.4, D34-S3).

Registers preceding kept frames I_{t-delta} (at Delta_t = 1.0 s step, D34-S2) into the current
frame I_t image plane using ONLY WP-06 attitude (`q_ned_to_cam_est`) and navigation position
(`pos_wgs84`):
1. Ground projection on the WGS-84 ellipsoid: rays that intersect the WGS-84 ellipsoid surface
   from p_{cam,t} are intersected in 3D ECEF and reprojected into frame t-delta from p_{cam,t-delta}.
2. Sky rotation at infinity: above-horizon rays that do not intersect the WGS-84 ellipsoid are
   registered at infinity via pure rotational compensation K R_{cam,t-delta}^T R_{cam,t} K^{-1}.
3. Strict pose-source guard (D34-S3, D34-S8, V4_G1): rejects ground-truth pose inputs unless
   `ablation_mode="oracle"` is explicitly specified.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from xvs.geo import rotations, wgs84

# Default ring camera intrinsics (WP-02 §2.4 / sequence.json)
DEFAULT_FX: float = 2319.0
DEFAULT_FY: float = 2319.0
DEFAULT_CX: float = 1223.5
DEFAULT_CY: float = 1023.5

FORBIDDEN_GT_POSE_KEYS: frozenset[str] = frozenset(
    {
        "q_ned_to_cam",
        "q_ned_to_cam_gt",
        "pos_wgs84_gt",
        "p_cam_ecef_gt",
        "R_cam_ecef_gt",
        "r_ecef_cam_gt",
        "pose_gt",
        "ground_truth_pose",
    }
)


def validate_pose_dict(pose: Mapping[str, Any], ablation_mode: str = "standard") -> None:
    """Enforce Decision D34-S3 / S8 / V4_G1 on warp pose inputs.

    In standard (non-oracle) mode:
    - Any ground-truth pose key or `pose_source == "ground_truth"` raises `ValueError`.
    - Requires `q_ned_to_cam_est` and `pos_wgs84` to be present.
    """
    if ablation_mode not in ("standard", "oracle"):
        raise ValueError(f"Unsupported ablation_mode={ablation_mode!r}; expected 'standard' or 'oracle'.")

    if ablation_mode != "oracle":
        present_forbidden = FORBIDDEN_GT_POSE_KEYS.intersection(pose.keys())
        if present_forbidden:
            raise ValueError(
                f"Forbidden ground-truth pose keys {sorted(present_forbidden)} passed to D1-KF warp "
                "with ablation_mode='standard' (Decision D34-S3 / S8 / V4_G1). "
                "Ground-truth pose is permitted only when ablation_mode='oracle'."
            )
        if str(pose.get("pose_source", "estimated")).lower() in ("gt", "ground_truth", "truth"):
            raise ValueError(
                "Ground-truth pose_source is forbidden in non-oracle D1-KF registration (D34-S3 / V4_G1)."
            )
        if "q_ned_to_cam_est" not in pose or "pos_wgs84" not in pose:
            raise ValueError(
                "Standard D1-KF warp requires both 'q_ned_to_cam_est' and 'pos_wgs84' in pose metadata (D34-S3 / S8)."
            )
    else:
        has_est = "q_ned_to_cam_est" in pose and "pos_wgs84" in pose
        has_gt = ("q_ned_to_cam" in pose or "q_ned_to_cam_gt" in pose) and (
            "pos_wgs84" in pose or "pos_wgs84_gt" in pose
        )
        if not (has_est or has_gt):
            raise ValueError(
                "Oracle D1-KF warp requires ('q_ned_to_cam' / 'q_ned_to_cam_est') and ('pos_wgs84' / 'pos_wgs84_gt')."
            )


def resolve_pose_ecef_and_r_cam(
    pose: Mapping[str, Any],
    ablation_mode: str = "standard",
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve `(p_cam_ecef, R_ecef_cam)` from a validated pose dictionary.

    - `pos_wgs84` (or `pos_wgs84_gt` in oracle mode) may be ECEF metres `[x, y, z]`
      (identified by Euclidean norm > 1e6 m) or geodetic `[lat_deg, lon_deg, h_m]`.
    - `q_ned_to_cam_est` (or `q_ned_to_cam` in oracle mode) is a Hamilton quaternion
      `[w, x, y, z]` mapping camera-frame vectors to local NED (`v_ned = R_ned_cam @ v_cam`),
      or a 3x3 rotation matrix `R_ned_cam`.
    """
    validate_pose_dict(pose, ablation_mode=ablation_mode)

    if ablation_mode == "oracle" and ("pos_wgs84_gt" in pose or "q_ned_to_cam" in pose or "q_ned_to_cam_gt" in pose):
        raw_pos = np.asarray(pose.get("pos_wgs84_gt", pose.get("pos_wgs84")), dtype=np.float64)
        raw_q = np.asarray(
            pose.get("q_ned_to_cam_gt", pose.get("q_ned_to_cam", pose.get("q_ned_to_cam_est"))),
            dtype=np.float64,
        )
    else:
        raw_pos = np.asarray(pose["pos_wgs84"], dtype=np.float64)
        raw_q = np.asarray(pose["q_ned_to_cam_est"], dtype=np.float64)

    if raw_pos.shape != (3,):
        raise ValueError(f"Expected 3-element pos_wgs84, got shape {raw_pos.shape}")

    if float(np.linalg.norm(raw_pos)) > 1.0e6:
        p_cam_ecef = raw_pos.astype(np.float64)
        lat_deg, lon_deg, _ = wgs84.ecef_to_geodetic(
            float(p_cam_ecef[0]), float(p_cam_ecef[1]), float(p_cam_ecef[2])
        )
    else:
        lat_deg, lon_deg, h_m = float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2])
        p_cam_ecef = wgs84.geodetic_to_ecef(lat_deg, lon_deg, h_m)

    r_ecef_ned_mat = wgs84.r_ecef_ned(lat_deg, lon_deg)
    if raw_q.shape == (4,):
        r_ned_cam = rotations.quat_to_r(raw_q)
    elif raw_q.shape == (3, 3):
        r_ned_cam = raw_q.astype(np.float64)
    else:
        raise ValueError(f"Expected quaternion (4,) or rotation matrix (3,3), got shape {raw_q.shape}")

    r_ecef_cam = r_ecef_ned_mat @ r_ned_cam
    return p_cam_ecef, r_ecef_cam


def intersect_rays_wgs84_and_reproject(
    u_src: np.ndarray,
    v_src: np.ndarray,
    p_cam_src: np.ndarray,
    r_ecef_cam_src: np.ndarray,
    p_cam_dst: np.ndarray,
    r_ecef_cam_dst: np.ndarray,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map full-image pixels `(u_src, v_src)` from frame `src` into frame `dst` under 6-DoF WGS-84 + sky model.

    Returns:
        u_dst: projected horizontal full-image pixel coordinates in frame `dst`.
        v_dst: projected vertical full-image pixel coordinates in frame `dst`.
        hits_earth: boolean mask indicating which rays intersected the WGS-84 ellipsoid.
    """
    u_flat = np.asarray(u_src, dtype=np.float64).ravel()
    v_flat = np.asarray(v_src, dtype=np.float64).ravel()
    orig_shape = np.asarray(u_src).shape

    if u_flat.size == 0:
        empty = np.empty(orig_shape, dtype=np.float64)
        return empty, empty.copy(), np.empty(orig_shape, dtype=bool)

    d_cam_src = np.vstack(
        [(u_flat - cx) / fx, (v_flat - cy) / fy, np.ones_like(u_flat)]
    )  # (3, N)
    d_ecef = r_ecef_cam_src @ d_cam_src  # (3, N)

    # WGS-84 ellipsoid ray intersection: ||S (p + t d)||^2 = 1, with t > 0
    a = wgs84.WGS84_A
    b = wgs84.WGS84_B
    s_vec = np.array([1.0 / a, 1.0 / a, 1.0 / b], dtype=np.float64)
    c_prime = p_cam_src * s_vec  # (3,)
    big_k = float(np.dot(c_prime, c_prime)) - 1.0

    d_prime = d_ecef * s_vec[:, None]  # (3, N)
    c_dot_d = c_prime @ d_prime  # (N,)
    d2 = np.sum(d_prime * d_prime, axis=0)  # (N,)
    disc = c_dot_d * c_dot_d - d2 * big_k
    hits_earth = (disc > 0.0) & (c_dot_d < 0.0)

    # Default to pure rotational compensation at infinity (sky rays)
    v_cam_dst = r_ecef_cam_dst.T @ d_ecef  # (3, N)

    if np.any(hits_earth):
        idx = np.where(hits_earth)[0]
        t_hit = (-c_dot_d[idx] - np.sqrt(np.maximum(0.0, disc[idx]))) / d2[idx]
        x_ecef = p_cam_src[:, None] + d_ecef[:, idx] * t_hit[None, :]  # (3, M)
        rel_ecef = x_ecef - p_cam_dst[:, None]
        v_cam_dst[:, idx] = r_ecef_cam_dst.T @ rel_ecef

    z_dst = np.where(np.abs(v_cam_dst[2]) > 1e-12, v_cam_dst[2], 1e-12)
    u_dst = (fx * (v_cam_dst[0] / z_dst) + cx).reshape(orig_shape)
    v_dst = (fy * (v_cam_dst[1] / z_dst) + cy).reshape(orig_shape)
    return u_dst, v_dst, hits_earth.reshape(orig_shape)


def project_points_6dof_wgs84(
    u_src: np.ndarray,
    v_src: np.ndarray,
    pose_src: Mapping[str, Any],
    pose_dst: Mapping[str, Any],
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    ablation_mode: str = "standard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project full-image pixel coordinates `(u_src, v_src)` from `pose_src` into `pose_dst`."""
    p_src, r_src = resolve_pose_ecef_and_r_cam(pose_src, ablation_mode=ablation_mode)
    p_dst, r_dst = resolve_pose_ecef_and_r_cam(pose_dst, ablation_mode=ablation_mode)
    return intersect_rays_wgs84_and_reproject(
        u_src=u_src,
        v_src=v_src,
        p_cam_src=p_src,
        r_ecef_cam_src=r_src,
        p_cam_dst=p_dst,
        r_ecef_cam_dst=r_dst,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def compute_6dof_correspondence_grid(
    pose_curr: Mapping[str, Any],
    pose_prev: Mapping[str, Any],
    roi_u0_curr: float = 0.0,
    roi_v0_curr: float = 0.0,
    roi_u0_prev: float = 0.0,
    roi_v0_prev: float = 0.0,
    height: int = 256,
    width: int = 256,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    ablation_mode: str = "standard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute backward-sampling grid mapping current ROI pixels `(u_t, v_t)` to previous ROI `(u_{t-d}, v_{t-d})`.

    Returns:
        u_local_prev: `(H, W)` float64 local horizontal coordinates in `I_{t-delta}` (`[0, W-1]` inside ROI).
        v_local_prev: `(H, W)` float64 local vertical coordinates in `I_{t-delta}` (`[0, H-1]` inside ROI).
        overlap_mask: `(H, W)` bool mask where `(u_local_prev, v_local_prev)` lies inside `[0, W-1] x [0, H-1]`.
        earth_mask:   `(H, W)` bool mask where the pixel ray in frame `t` intersects the WGS-84 ellipsoid.
    """
    p_curr, r_curr = resolve_pose_ecef_and_r_cam(pose_curr, ablation_mode=ablation_mode)
    p_prev, r_prev = resolve_pose_ecef_and_r_cam(pose_prev, ablation_mode=ablation_mode)

    u_local = np.arange(width, dtype=np.float64) + float(roi_u0_curr)
    v_local = np.arange(height, dtype=np.float64) + float(roi_v0_curr)
    uu_curr, vv_curr = np.meshgrid(u_local, v_local)

    u_full_prev, v_full_prev, earth_mask = intersect_rays_wgs84_and_reproject(
        u_src=uu_curr,
        v_src=vv_curr,
        p_cam_src=p_curr,
        r_ecef_cam_src=r_curr,
        p_cam_dst=p_prev,
        r_ecef_cam_dst=r_prev,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    u_local_prev = u_full_prev - float(roi_u0_prev)
    v_local_prev = v_full_prev - float(roi_v0_prev)
    overlap_mask = (
        (u_local_prev >= 0.0)
        & (u_local_prev <= float(width - 1))
        & (v_local_prev >= 0.0)
        & (v_local_prev <= float(height - 1))
    )
    return u_local_prev, v_local_prev, overlap_mask, earth_mask


def warp_roi_into_current(
    img_prev: torch.Tensor | np.ndarray,
    pose_curr: Mapping[str, Any],
    pose_prev: Mapping[str, Any],
    roi_u0_curr: float = 0.0,
    roi_v0_curr: float = 0.0,
    roi_u0_prev: float = 0.0,
    roi_v0_prev: float = 0.0,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    ablation_mode: str = "standard",
    interp_mode: str = "bicubic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample preceding ROI `img_prev` into current ROI frame using `torch.nn.functional.grid_sample`.

    Args:
        img_prev: `(H, W)`, `(C, H, W)`, or `(B, C, H, W)` tensor or ndarray.
        pose_curr: frame `t` pose dict (`q_ned_to_cam_est`, `pos_wgs84`).
        pose_prev: frame `t - delta` pose dict (`q_ned_to_cam_est`, `pos_wgs84`).

    Returns:
        warped_prev: resampled image in frame `t` coordinates (same shape/type as tensor input, `(H, W)` if 2D).
        overlap_mask: `(H, W)` float32 mask (`1.0` inside valid overlap, `0.0` outside).
        earth_mask: `(H, W)` bool tensor (`True` for ground rays, `False` for sky rays).
    """
    if isinstance(img_prev, np.ndarray):
        t_prev = torch.from_numpy(img_prev)
    else:
        t_prev = img_prev

    orig_ndim = t_prev.ndim
    if orig_ndim == 2:
        height, width = int(t_prev.shape[0]), int(t_prev.shape[1])
        t_4d = t_prev.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        height, width = int(t_prev.shape[1]), int(t_prev.shape[2])
        t_4d = t_prev.unsqueeze(0)
    elif orig_ndim == 4:
        height, width = int(t_prev.shape[2]), int(t_prev.shape[3])
        t_4d = t_prev
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D image tensor, got shape {tuple(t_prev.shape)}")

    device = t_4d.device
    if (
        float(roi_u0_prev) == float(roi_u0_curr)
        and float(roi_v0_prev) == float(roi_v0_curr)
        and pose_prev == pose_curr
    ):
        validate_pose_dict(pose_curr, ablation_mode=ablation_mode)
        p_c, r_c = resolve_pose_ecef_and_r_cam(pose_curr, ablation_mode=ablation_mode)
        u_mid = np.array([float(roi_u0_curr) + 0.5 * float(width)], dtype=np.float64)
        v_mid = np.array([float(roi_v0_curr) + 0.5 * float(height)], dtype=np.float64)
        _, _, hit_mid = intersect_rays_wgs84_and_reproject(u_mid, v_mid, p_c, r_c, p_c, r_c, fx=fx, fy=fy, cx=cx, cy=cy)
        overlap_ones = torch.ones((height, width), dtype=torch.float32, device=device)
        earth_const = torch.full((height, width), bool(hit_mid[0]), dtype=torch.bool, device=device)
        return t_prev.clone(), overlap_ones, earth_const

    u_local_prev, v_local_prev, overlap_np, earth_np = compute_6dof_correspondence_grid(
        pose_curr=pose_curr,
        pose_prev=pose_prev,
        roi_u0_curr=roi_u0_curr,
        roi_v0_curr=roi_v0_curr,
        roi_u0_prev=roi_u0_prev,
        roi_v0_prev=roi_v0_prev,
        height=height,
        width=width,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        ablation_mode=ablation_mode,
    )

    device = t_4d.device
    work_dtype = torch.float64 if t_4d.dtype == torch.float64 else torch.float32
    denom_x = max(1.0, float(width - 1))
    denom_y = max(1.0, float(height - 1))
    grid_x = 2.0 * (u_local_prev / denom_x) - 1.0
    grid_y = 2.0 * (v_local_prev / denom_y) - 1.0
    grid_np = np.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)

    grid_t = torch.from_numpy(grid_np).to(device=device, dtype=work_dtype).unsqueeze(0)
    if t_4d.shape[0] > 1:
        grid_t = grid_t.expand(t_4d.shape[0], -1, -1, -1)

    warped_4d = F.grid_sample(
        t_4d.to(dtype=work_dtype),
        grid_t,
        mode=interp_mode,
        padding_mode="zeros",
        align_corners=True,
    ).to(dtype=t_prev.dtype)

    overlap_t = torch.from_numpy(overlap_np.astype(np.float32)).to(device=device)
    earth_t = torch.from_numpy(earth_np).to(device=device)

    # Zero out outside overlap mask
    warped_4d = warped_4d * overlap_t.to(dtype=warped_4d.dtype).unsqueeze(0).unsqueeze(0)

    if orig_ndim == 2:
        warped_out = warped_4d.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        warped_out = warped_4d.squeeze(0)
    else:
        warped_out = warped_4d

    return warped_out, overlap_t, earth_t


def register_kframe_stack(
    frames: list[torch.Tensor | np.ndarray],
    poses: list[Mapping[str, Any]],
    roi_origins: list[tuple[float, float]] | None = None,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    ablation_mode: str = "standard",
    interp_mode: str = "bilinear",
) -> dict[str, Any]:
    """Register a K-frame stack `{I_{t-K+1}, ..., I_t}` (Delta_t = 1.0 s) into frame `t` (`WP04_DETECTION.md` §9.4).

    Concatenates `[I_t, tilde{I}_{t-1}, M_{t-1}*(I_t - tilde{I}_{t-1}), ..., tilde{I}_{t-K+1}, M_{t-K+1}*(I_t - tilde{I}_{t-K+1})]`
    producing `2*K - 1` channels (`5` channels for `K=3`, `9` channels for `K=5`).
    """
    k_frames = len(frames)
    if k_frames not in (1, 3, 5):
        raise ValueError(f"D1-KF supports K in {{1, 3, 5}} frames, got K={k_frames}")
    if len(poses) != k_frames:
        raise ValueError(f"Number of poses ({len(poses)}) must match number of frames ({k_frames})")

    if roi_origins is None:
        roi_origins = [(0.0, 0.0)] * k_frames
    if len(roi_origins) != k_frames:
        raise ValueError("roi_origins length must match frames length")

    # Validate current frame pose even when K == 1
    validate_pose_dict(poses[-1], ablation_mode=ablation_mode)

    curr_raw = frames[-1]
    curr_t = torch.from_numpy(curr_raw) if isinstance(curr_raw, np.ndarray) else curr_raw
    if curr_t.ndim != 2:
        raise ValueError(f"Each frame in stack must be 2D (H, W), got shape {tuple(curr_t.shape)}")
    curr_f32 = curr_t.to(dtype=torch.float32)

    u0_curr, v0_curr = roi_origins[-1]
    channels: list[torch.Tensor] = [curr_f32]
    warped_frames: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    overlap_masks: list[torch.Tensor] = []

    # Iterate backwards from t-1 down to t-K+1
    for idx in range(k_frames - 2, -1, -1):
        prev_raw = frames[idx]
        prev_t = torch.from_numpy(prev_raw) if isinstance(prev_raw, np.ndarray) else prev_raw
        prev_f32 = prev_t.to(device=curr_f32.device, dtype=torch.float32)
        u0_prev, v0_prev = roi_origins[idx]

        warped_prev, overlap_mask, _ = warp_roi_into_current(
            img_prev=prev_f32,
            pose_curr=poses[-1],
            pose_prev=poses[idx],
            roi_u0_curr=u0_curr,
            roi_v0_curr=v0_curr,
            roi_u0_prev=u0_prev,
            roi_v0_prev=v0_prev,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            ablation_mode=ablation_mode,
            interp_mode=interp_mode,
        )
        diff_res = overlap_mask * (curr_f32 - warped_prev)
        channels.append(warped_prev)
        channels.append(diff_res)
        warped_frames.append(warped_prev)
        residuals.append(diff_res)
        overlap_masks.append(overlap_mask)

    stacked = torch.stack(channels, dim=0)  # (2*K - 1, H, W)
    return {
        "input_tensor": stacked,
        "current_frame": curr_f32,
        "warped_frames": warped_frames,
        "residuals": residuals,
        "overlap_masks": overlap_masks,
        "ablation_mode": ablation_mode,
        "num_frames_k": k_frames,
    }


def _to_tracking_u8(img: np.ndarray) -> np.ndarray:
    """Normalize a 2D grayscale array to `uint8` strictly for keypoint detection and pyramidal LK tracking."""
    arr = np.asarray(img, dtype=np.float32)
    lo = float(np.percentile(arr, 1.0))
    hi = max(float(np.percentile(arr, 99.9)), float(lo + 0.85 * (float(np.max(arr)) - lo)))
    if hi - lo < 1e-3:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0)
    return scaled.astype(np.uint8)


def estimate_2d_background_homography(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    max_corners: int = 600,
    quality_level: float = 0.01,
    min_distance: float = 12.0,
    ransac_reproj_threshold: float = 3.0,
) -> tuple[np.ndarray, str]:
    """Estimate 2D background homography mapping `img_prev` (`I_{t-d}`) coordinates onto `img_curr` (`I_t`).

    Uses `cv2.goodFeaturesToTrack` + forward-backward pyramidal `cv2.calcOpticalFlowPyrLK` +
    `cv2.findHomography(..., cv2.RANSAC, 3.0)`, falling back cleanly to global phase-correlation
    translation or identity when fewer than 8 background keypoints exist (e.g., pure blue sky).
    """
    prev_f32 = np.asarray(img_prev, dtype=np.float32)
    curr_f32 = np.asarray(img_curr, dtype=np.float32)
    height, width = curr_f32.shape

    u8_prev = _to_tracking_u8(prev_f32)
    u8_curr = _to_tracking_u8(curr_f32)

    pts_prev = cv2.goodFeaturesToTrack(
        u8_prev,
        maxCorners=int(max_corners),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance),
        blockSize=7,
    )
    if pts_prev is not None and len(pts_prev) >= 8:
        lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        pts_curr, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            u8_prev,
            u8_curr,
            pts_prev,
            None,
            winSize=(21, 21),
            maxLevel=4,
            criteria=lk_criteria,
        )
        if pts_curr is not None and st_fwd is not None:
            pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                u8_curr,
                u8_prev,
                pts_curr,
                None,
                winSize=(21, 21),
                maxLevel=4,
                criteria=lk_criteria,
            )
            if pts_back is not None and st_bwd is not None:
                fb_err = np.linalg.norm(pts_prev.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
                valid = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_err < 1.5)
                good_prev = pts_prev.reshape(-1, 2)[valid]
                good_curr = pts_curr.reshape(-1, 2)[valid]
                if len(good_prev) >= 8:
                    h_cand, inliers = cv2.findHomography(
                        good_prev,
                        good_curr,
                        cv2.RANSAC,
                        float(ransac_reproj_threshold),
                    )
                    if (
                        h_cand is not None
                        and h_cand.shape == (3, 3)
                        and np.all(np.isfinite(h_cand))
                        and inliers is not None
                        and int(np.sum(inliers)) >= 6
                    ):
                        det_2x2 = abs(float(np.linalg.det(h_cand[:2, :2])))
                        if 0.25 < det_2x2 < 4.0 and abs(float(h_cand[2, 0])) < 2.0e-3 and abs(float(h_cand[2, 1])) < 2.0e-3:
                            return h_cand.astype(np.float64), "homography_ransac"

    # Fallback 1: Global 2D phase-correlation translation when < 8 keypoints exist on low-texture sky
    if height >= 16 and width >= 16:
        p_zero = prev_f32 - float(np.mean(prev_f32))
        c_zero = curr_f32 - float(np.mean(curr_f32))
        if float(np.std(p_zero)) > 1e-4 and float(np.std(c_zero)) > 1e-4:
            hann = cv2.createHanningWindow((width, height), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(p_zero, c_zero, window=hann)
            max_shift = 0.25 * float(min(height, width))
            if math.isfinite(dx) and math.isfinite(dy) and float(response) >= 0.12 and math.hypot(dx, dy) <= max_shift:
                h_trans = np.array(
                    [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                return h_trans, "phase_correlate"

    # Fallback 2: Clean identity on pure featureless sky
    return np.eye(3, dtype=np.float64), "identity"


def align_aot_kframe_stack_2d(
    frames: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Register a single-camera AOT `K`-frame stack `{I_{t-K+1}, ..., I_t}` onto `I_t` via 2D homography (`B8-Colab-F2`).

    Returns:
        aligned_frames: list of `K` float32 `(H, W)` arrays registered into `I_t` coordinates.
        valid_masks: list of `K` float32 `(H, W)` valid border masks `M_{t-d}` (`1.0` inside valid overlap, `0.0` outside).
    """
    n_f = len(frames)
    if n_f == 0:
        return [], []
    curr_f32 = np.asarray(frames[-1], dtype=np.float32)
    if curr_f32.ndim != 2:
        raise ValueError(f"Expected 2D grayscale frames in align_aot_kframe_stack_2d, got shape {curr_f32.shape}")
    height, width = curr_f32.shape
    if n_f == 1:
        return [curr_f32.copy()], [np.ones((height, width), dtype=np.float32)]

    aligned: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ones_f32 = np.ones((height, width), dtype=np.float32)
    erode_kernel = np.ones((3, 3), dtype=np.uint8)

    for idx in range(n_f - 1):
        prev_f32 = np.asarray(frames[idx], dtype=np.float32)
        if prev_f32.shape != (height, width):
            raise ValueError(
                f"Frame shape mismatch in align_aot_kframe_stack_2d: {prev_f32.shape} vs {(height, width)}"
            )
        if np.array_equal(prev_f32, curr_f32):
            aligned.append(curr_f32.copy())
            masks.append(ones_f32.copy())
            continue

        h_mat, mode = estimate_2d_background_homography(prev_f32, curr_f32)
        if mode == "identity" or np.allclose(h_mat, np.eye(3), atol=1e-6):
            aligned.append(prev_f32.copy())
            masks.append(ones_f32.copy())
        else:
            warped_prev = cv2.warpPerspective(
                prev_f32,
                h_mat,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            raw_mask = cv2.warpPerspective(
                ones_f32,
                h_mat,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            bin_mask = (raw_mask >= 0.999).astype(np.uint8)
            eroded = cv2.erode(bin_mask, erode_kernel, iterations=1).astype(np.float32)
            warped_prev = warped_prev * eroded
            aligned.append(warped_prev.astype(np.float32))
            masks.append(eroded)

    aligned.append(curr_f32.copy())
    masks.append(ones_f32.copy())
    return aligned, masks


def build_prealigned_kframe_tensor(
    aligned_frames: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
) -> torch.Tensor:
    """Build `(2*K - 1, H, W)` `D1-KF` input tensor from 2D-aligned frames `{tilde{I}_{t-K+1}, ..., I_t}` and masks."""
    k_frames = len(aligned_frames)
    if k_frames not in (1, 3, 5):
        raise ValueError(f"D1-KF supports K in {{1, 3, 5}} frames, got K={k_frames}")
    if len(valid_masks) != k_frames:
        raise ValueError("valid_masks length must match aligned_frames length")

    curr_t = torch.from_numpy(np.asarray(aligned_frames[-1], dtype=np.float32))
    if k_frames == 1:
        return curr_t.unsqueeze(0)

    channels: list[torch.Tensor] = [curr_t]
    for idx in range(k_frames - 2, -1, -1):
        w_prev = torch.from_numpy(np.asarray(aligned_frames[idx], dtype=np.float32))
        m_prev = torch.from_numpy(np.asarray(valid_masks[idx], dtype=np.float32))
        diff_res = m_prev * (curr_t - w_prev)
        channels.append(w_prev)
        channels.append(diff_res)
    return torch.stack(channels, dim=0)

