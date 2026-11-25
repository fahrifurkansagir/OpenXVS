"""Training loop and train-split-only on-the-fly target synthesis for WP-04 G1 (WP04_DETECTION.md §9.3–§9.4).

Key invariants:
- Target synthesis (`synthesize_target_on_background` / `synthesize_target_on_kframe_stack`) is
  restricted strictly to `split == "train"` (raises `ValueError` if called on `"val"` or `"test"`).
- Never applies a second pass of whole-image WP-03 shot/read noise when the background crop is already
  in `"degraded"` sensor state (`B8-R2-F6`); applies only local Poisson photon noise on the rendered
  aircraft sprite pixels.
- Preserves real positive `kframe_stack` samples intact with random spatial shifts, and injects
  multi-frame synthetic targets across ALL K frames (`t = 0 .. K-1`) along a constant-velocity
  trajectory mapped through the inverse 6-DoF WGS-84 + sky registration warp (`B8-R2-F6`).
- Respects single-process CPU rules (`num_workers=0`, `torch.set_num_threads(2)`).
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import numpy as np
import torch

from xvs.detection.d1_model import (
    D1HeatmapDetector,
    compute_d1_losses,
    splat_gaussian_targets,
)
from xvs.detection.warp import (
    intersect_rays_wgs84_and_reproject,
    register_kframe_stack,
    resolve_pose_ecef_and_r_cam,
)


def synthesize_target_on_background(
    bg_linear: np.ndarray,
    rng: np.random.Generator,
    split: str = "train",
    s_px_range: tuple[float, float] = (2.0, 12.0),
    contrast_range: tuple[float, float] = (-0.45, 0.35),
    margin_px: int = 16,
    apply_shot_read_noise: bool = False,
    sensor_state: str = "degraded",
    read_noise_e: float = 2.4,
    center_uv: tuple[float, float] | None = None,
    s_px_override: float | None = None,
    aspect_override: float | None = None,
    contrast_override: float | None = None,
) -> dict[str, Any]:
    """Composite a synthetic sub-pixel aircraft silhouette onto a train-split linear background crop.

    Strictly enforces `split == "train"` (WP04_DETECTION.md §9.2 & §9.3).
    When `sensor_state == "degraded"` (or `apply_shot_read_noise=False`), applies ONLY local Poisson
    photon noise on the rendered aircraft sprite pixels rather than re-noising the background (`B8-R2-F6`).
    """
    if split != "train":
        raise ValueError(
            f"On-the-fly target synthesis is permitted ONLY on the 'train' split, got split={split!r}"
        )

    img = np.asarray(bg_linear, dtype=np.float64).copy()
    if img.ndim != 2:
        raise ValueError(f"Expected 2D linear luminance background crop, got shape {img.shape}")
    height, width = img.shape

    if center_uv is not None:
        u_c = float(np.clip(center_uv[0], 4.0, max(5.0, width - 5.0)))
        v_c = float(np.clip(center_uv[1], 4.0, max(5.0, height - 5.0)))
    else:
        eff_margin = min(int(margin_px), max(4, min(height, width) // 4))
        u_c = float(rng.uniform(eff_margin, max(eff_margin + 1, width - eff_margin)))
        v_c = float(rng.uniform(eff_margin, max(eff_margin + 1, height - eff_margin)))

    if s_px_override is not None:
        s_px = float(s_px_override)
    else:
        s_px = float(np.exp(rng.uniform(math.log(max(0.5, s_px_range[0])), math.log(s_px_range[1]))))
    aspect = float(aspect_override) if aspect_override is not None else float(rng.uniform(1.2, 2.2))
    w_px = s_px
    h_px = max(0.5, s_px / aspect)

    if contrast_override is not None:
        c_val = float(contrast_override)
    else:
        c_val = float(rng.uniform(contrast_range[0], contrast_range[1]))
        if abs(c_val) < 0.12:
            c_val = 0.22 if c_val >= 0.0 else -0.22

    half_win = int(math.ceil(max(4.0, 2.0 * s_px)))
    x0 = max(0, int(math.floor(u_c)) - half_win)
    x1 = min(width, int(math.floor(u_c)) + half_win + 1)
    y0 = max(0, int(math.floor(v_c)) - half_win)
    y1 = min(height, int(math.floor(v_c)) + half_win + 1)

    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64),
        np.arange(x0, x1, dtype=np.float64),
        indexing="ij",
    )
    sigma_x = max(0.45, 0.28 * w_px)
    sigma_y = max(0.45, 0.28 * h_px)

    # Fuselage + wings elliptical coverage profile alpha in [0, 1]
    alpha_wing = np.exp(-0.5 * (((xx - u_c) / sigma_x) ** 2 + ((yy - v_c) / (0.6 * sigma_y)) ** 2))
    alpha_body = np.exp(-0.5 * (((xx - u_c) / (0.5 * sigma_x)) ** 2 + ((yy - v_c) / sigma_y) ** 2))
    alpha = np.clip(np.maximum(alpha_wing, alpha_body), 0.0, 1.0)

    patch_bg = np.maximum(1.0, img[y0:y1, x0:x1])
    delta_signal = patch_bg * c_val * alpha
    new_patch = np.clip(patch_bg + delta_signal, 0.0, None)

    if sensor_state == "clean" and apply_shot_read_noise:
        # Whole-patch shot + read noise applied ONLY when input background is clean
        shot_std = np.sqrt(np.maximum(1.0, new_patch))
        noise = rng.normal(0.0, 1.0, size=new_patch.shape) * np.sqrt(shot_std**2 + read_noise_e**2)
        img[y0:y1, x0:x1] = np.clip(new_patch + noise, 0.0, None)
    else:
        # B8-R2-F6: Background is already WP-03 degraded; apply local Poisson photon noise ONLY on sprite pixels
        sprite_mask = (alpha > 1e-3).astype(np.float64)
        sprite_photon_std = np.sqrt(np.maximum(0.0, np.abs(delta_signal))) * sprite_mask
        sprite_noise = rng.normal(0.0, 1.0, size=new_patch.shape) * sprite_photon_std
        img[y0:y1, x0:x1] = np.clip(new_patch + sprite_noise, 0.0, None)

    return {
        "image": img.astype(np.float32),
        "center_uv": (u_c, v_c),
        "box_wh": (w_px, h_px),
        "s_px": s_px,
        "contrast": c_val,
    }


def synthesize_target_on_kframe_stack(
    frames: Sequence[np.ndarray],
    poses: Sequence[Mapping[str, Any]],
    roi_origins: Sequence[tuple[float, float]],
    rng: np.random.Generator,
    split: str = "train",
    sensor_state: str = "degraded",
    ablation_mode: str = "standard",
) -> dict[str, Any]:
    """Inject a synthetic target across ALL K frames (`t = 0 .. K-1`) along a constant-velocity trajectory (`B8-R2-F6`).

    Maps `(u_reg(t), v_reg(t))` in registered frame `K-1` coordinates into each preceding frame `t`'s
    unwarped pixel coordinates via the inverse 6-DoF WGS-84 + sky warp so that after `register_kframe_stack`,
    the synthetic target aligns coherently across all `K` frames.
    """
    if split != "train":
        raise ValueError(
            f"Multi-frame target synthesis is permitted ONLY on the 'train' split, got split={split!r}"
        )

    k_len = len(frames)
    f_last = np.asarray(frames[-1], dtype=np.float32)
    height, width = f_last.shape
    eff_margin = min(16, max(4, min(height, width) // 4))

    u_c = float(rng.uniform(eff_margin, max(eff_margin + 1, width - eff_margin)))
    v_c = float(rng.uniform(eff_margin, max(eff_margin + 1, height - eff_margin)))
    s_px = float(np.exp(rng.uniform(math.log(2.2), math.log(10.0))))
    aspect = float(rng.uniform(1.2, 2.0))
    c_val = float(rng.choice([-0.35, -0.28, 0.25, 0.35]))

    # Constant angular velocity (du_dt, dv_dt) in registered frame (px/frame) so temporal difference channels fire
    du_step = float(rng.uniform(-1.5, 1.5))
    dv_step = float(rng.uniform(-1.5, 1.5))
    if math.hypot(du_step, dv_step) < 0.6:
        du_step = 1.0 if du_step >= 0.0 else -1.0

    u0_last, v0_last = roi_origins[-1] if roi_origins else (0.0, 0.0)
    p_curr, r_curr = (
        resolve_pose_ecef_and_r_cam(poses[-1], ablation_mode=ablation_mode)
        if poses
        else (np.zeros(3), np.eye(3))
    )

    out_frames: list[np.ndarray] = []
    for t_idx in range(k_len):
        dt_back = float(k_len - 1 - t_idx)
        u_reg_t = u_c - dt_back * du_step
        v_reg_t = v_c - dt_back * dv_step

        if t_idx == k_len - 1 or not poses or len(poses) != k_len:
            u_t_local, v_t_local = u_reg_t, v_reg_t
        else:
            p_prev, r_prev = resolve_pose_ecef_and_r_cam(poses[t_idx], ablation_mode=ablation_mode)
            u_full_reg = np.array([u0_last + u_reg_t], dtype=np.float64)
            v_full_reg = np.array([v0_last + v_reg_t], dtype=np.float64)
            u_g, v_g, _ = intersect_rays_wgs84_and_reproject(
                u_full_reg, v_full_reg, p_curr, r_curr, p_prev, r_prev
            )
            u_prev_full, v_prev_full = float(u_g[0]), float(v_g[0])
            u0_t, v0_t = roi_origins[t_idx] if roi_origins else (u0_last, v0_last)
            u_t_local = u_prev_full - float(u0_t)
            v_t_local = v_prev_full - float(v0_t)

        syn_t = synthesize_target_on_background(
            bg_linear=np.asarray(frames[t_idx], dtype=np.float32),
            rng=rng,
            split="train",
            apply_shot_read_noise=False,
            sensor_state=sensor_state,
            center_uv=(u_t_local, v_t_local),
            s_px_override=s_px,
            aspect_override=aspect,
            contrast_override=c_val,
        )
        out_frames.append(syn_t["image"])

    w_px = s_px
    h_px = max(0.5, s_px / aspect)
    return {
        "frames": out_frames,
        "center_uv": (u_c, v_c),
        "box_wh": (w_px, h_px),
        "s_px": s_px,
        "contrast": c_val,
    }


def _apply_random_spatial_shift_to_positive_sample(
    frames_raw: list[np.ndarray],
    roi_origins: list[tuple[float, float]],
    centers_uv: list[tuple[float, float]],
    rng: np.random.Generator,
    train_patch_size: int = 96,
) -> tuple[list[np.ndarray], list[tuple[float, float]], list[tuple[float, float]]]:
    """Randomly shift/crop a positive training stack so the target is not fixed at `(H/2, W/2)`."""
    if not centers_uv:
        return frames_raw, roi_origins, centers_uv
    h, w = frames_raw[-1].shape
    if h <= train_patch_size and w <= train_patch_size:
        if h < 32 or w < 32:
            return frames_raw, roi_origins, centers_uv
        u0_c, v0_c = centers_uv[0]
        max_dx_neg = int(max(0.0, min(16.0, u0_c - 8.0)))
        max_dx_pos = int(max(0.0, min(16.0, (w - 8.0) - u0_c)))
        max_dy_neg = int(max(0.0, min(16.0, v0_c - 8.0)))
        max_dy_pos = int(max(0.0, min(16.0, (h - 8.0) - v0_c)))
        dx = int(rng.integers(-max_dx_neg, max_dx_pos + 1)) if (max_dx_neg + max_dx_pos) > 0 else 0
        dy = int(rng.integers(-max_dy_neg, max_dy_pos + 1)) if (max_dy_neg + max_dy_pos) > 0 else 0
        if dx == 0 and dy == 0:
            return frames_raw, roi_origins, centers_uv
        shifted_frames: list[np.ndarray] = []
        for fr in frames_raw:
            arr = np.asarray(fr, dtype=np.float32)
            pad_arr = np.pad(arr, ((abs(dy), abs(dy)), (abs(dx), abs(dx))), mode="edge")
            y_start = abs(dy) - dy
            x_start = abs(dx) - dx
            shifted_frames.append(pad_arr[y_start : y_start + h, x_start : x_start + w].copy())
        shifted_origins = [(o[0] - float(dx), o[1] - float(dy)) for o in roi_origins]
        shifted_centers = [(u + float(dx), v + float(dy)) for (u, v) in centers_uv]
        return shifted_frames, shifted_origins, shifted_centers

    # Crop a train_patch_size x train_patch_size window containing the target at a random local offset
    ps = min(train_patch_size, h, w)
    u0_c, v0_c = centers_uv[0]
    margin = min(24, ps // 3)
    x_min = max(0, int(math.ceil(u0_c - (ps - margin))))
    x_max = min(w - ps, int(math.floor(u0_c - margin)))
    y_min = max(0, int(math.ceil(v0_c - (ps - margin))))
    y_max = min(h - ps, int(math.floor(v0_c - margin)))
    x0 = int(rng.integers(x_min, x_max + 1)) if x_max >= x_min else int(np.clip(round(u0_c - ps / 2), 0, w - ps))
    y0 = int(rng.integers(y_min, y_max + 1)) if y_max >= y_min else int(np.clip(round(v0_c - ps / 2), 0, h - ps))

    cropped_frames = [np.asarray(fr[y0 : y0 + ps, x0 : x0 + ps], dtype=np.float32).copy() for fr in frames_raw]
    cropped_origins = [(o[0] + float(x0), o[1] + float(y0)) for o in roi_origins]
    cropped_centers = [(u - float(x0), v - float(y0)) for (u, v) in centers_uv]
    return cropped_frames, cropped_origins, cropped_centers


def _crop_negative_training_window(
    frames_raw: list[np.ndarray],
    roi_origins: list[tuple[float, float]],
    rng: np.random.Generator,
    train_patch_size: int = 96,
    full_frame_uniform: bool = False,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Extract a `train_patch_size x train_patch_size` window from a negative `horizon_strip`, `hard_negative_d0`, or `negative_roi` stack."""
    h, w = frames_raw[-1].shape
    if h <= train_patch_size and w <= train_patch_size:
        return [np.asarray(fr, dtype=np.float32).copy() for fr in frames_raw], roi_origins
    ps = min(train_patch_size, h, w)
    if full_frame_uniform:
        x0 = int(rng.integers(0, w - ps + 1)) if w > ps else 0
        y0 = int(rng.integers(0, h - ps + 1)) if h > ps else 0
    else:
        # Center-biased random crop so D0 hard-negative centers (at W/2, H/2) and horizon curves (at H/2) stay inside window
        cx, cy = w // 2, h // 2
        x_min = max(0, cx - ps + 24)
        x_max = min(w - ps, cx - 24)
        y_min = max(0, cy - ps + 24)
        y_max = min(h - ps, cy - 24)
        x0 = int(rng.integers(x_min, x_max + 1)) if x_max >= x_min else max(0, (w - ps) // 2)
        y0 = int(rng.integers(y_min, y_max + 1)) if y_max >= y_min else max(0, (h - ps) // 2)
    cropped_frames = [np.asarray(fr[y0 : y0 + ps, x0 : x0 + ps], dtype=np.float32).copy() for fr in frames_raw]
    cropped_origins = [(o[0] + float(x0), o[1] + float(y0)) for o in roi_origins]
    return cropped_frames, cropped_origins


def build_training_sample_tensors(
    sample: Mapping[str, Any],
    num_frames_k: int = 1,
    use_synthesis: bool = True,
    rng: np.random.Generator | None = None,
    ablation_mode: str = "standard",
    inject_synthetic_on_negative: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Convert one pack sample dict into `(input_tensor, gt_heatmap, gt_offset, gt_log_scale, center_mask)`.

    Honors `B8-R2-F6`:
    - Real positive `kframe_stack` samples (`len(centers_uv) > 0`) are NEVER overwritten by single-frame
      synthetic targets; they retain their real multi-frame trajectory (with random spatial shifts).
    - When `use_synthesis=True` on a negative sample (`len(centers_uv) == 0`), injects a synthetic target
      across ALL `K` frames (`t = 0 .. K-1`) along a constant-velocity trajectory mapped via the inverse
      registration warp (when `inject_synthetic_on_negative` is True) or keeps it as a pure negative
      clutter sample (`horizon_strip` / `hard_negative_d0`) to train false-alarm suppression.
    """
    split = str(sample.get("split", "train"))
    raw_in_frames = list(sample["frames"])
    poses = [dict(p) for p in sample.get("poses", [])]
    roi_origins = [tuple(float(x) for x in o) for o in sample.get("roi_origins", [(0.0, 0.0)] * len(raw_in_frames))]
    sensor_state = str(sample.get("sensor_state", "degraded"))

    if num_frames_k > 1:
        while len(raw_in_frames) < num_frames_k:
            raw_in_frames.insert(0, raw_in_frames[0])
            if poses:
                poses.insert(0, dict(poses[0]))
            roi_origins.insert(0, roi_origins[0])
        if len(raw_in_frames) > num_frames_k:
            raw_in_frames = raw_in_frames[-num_frames_k:]
            if poses:
                poses = poses[-num_frames_k:]
            roi_origins = roi_origins[-num_frames_k:]
    else:
        raw_in_frames = [raw_in_frames[-1]]
        if poses:
            poses = [poses[-1]]
        roi_origins = [roi_origins[-1]]

    centers_uv = list(sample.get("centers_uv", []))
    boxes_wh = list(sample.get("boxes_wh", []))

    if split == "train" and rng is not None:
        if len(centers_uv) > 0:
            frames_raw = [np.asarray(f, dtype=np.float32).copy() for f in raw_in_frames]
            # Real positive kframe_stack: keep real multi-frame trajectory intact and apply random spatial shift
            frames_raw, roi_origins, centers_uv = _apply_random_spatial_shift_to_positive_sample(
                frames_raw, roi_origins, centers_uv, rng
            )
        else:
            is_full_neg = str(sample.get("kind", "")) == "negative_roi"
            frames_raw, roi_origins = _crop_negative_training_window(
                raw_in_frames,
                roi_origins,
                rng,
                full_frame_uniform=is_full_neg,
            )
            if use_synthesis:
                do_inject = (
                    bool(inject_synthetic_on_negative)
                    if inject_synthetic_on_negative is not None
                    else bool(rng.uniform(0.0, 1.0) < 0.5)
                )
                if do_inject:
                    syn_stack = synthesize_target_on_kframe_stack(
                        frames=frames_raw,
                        poses=poses,
                        roi_origins=roi_origins,
                        rng=rng,
                        split="train",
                        sensor_state=sensor_state,
                        ablation_mode=ablation_mode,
                    )
                    frames_raw = syn_stack["frames"]
                    centers_uv = [syn_stack["center_uv"]]
                    boxes_wh = [syn_stack["box_wh"]]
    else:
        frames_raw = [np.asarray(f, dtype=np.float32).copy() for f in raw_in_frames]

    if num_frames_k == 1:
        f_curr = np.asarray(frames_raw[-1], dtype=np.float32)
        input_tensor = torch.from_numpy(f_curr).unsqueeze(0)
        h, w = f_curr.shape
    else:
        reg = register_kframe_stack(
            frames=frames_raw[-num_frames_k:],
            poses=poses[-num_frames_k:],
            roi_origins=roi_origins[-num_frames_k:],
            ablation_mode=ablation_mode,
        )
        input_tensor = reg["input_tensor"]
        _, h, w = input_tensor.shape

    targets = splat_gaussian_targets(height=h, width=w, centers_uv=centers_uv, boxes_wh=boxes_wh)
    return {
        "input": input_tensor,
        **targets,
    }


def _crop_cached_window(
    cached: dict[str, torch.Tensor],
    rng: np.random.Generator,
    crop_size: int = 64,
    hard_yx: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Slice a `crop_size x crop_size` window from a pre-registered tensor dict."""
    inp = cached["input"]
    _, h, w = inp.shape
    if h <= crop_size and w <= crop_size:
        win_inp = inp
        y0, x0, ps_h, ps_w = 0, 0, h, w
    else:
        ps_h = min(crop_size, h)
        ps_w = min(crop_size, w)
        cm = cached["center_mask"][0]
        pos_yx = torch.nonzero(cm > 0.5, as_tuple=False)
        if pos_yx.numel() > 0:
            cy = int(pos_yx[0, 0].item())
            cx = int(pos_yx[0, 1].item())
            margin = 8
            x_min = max(0, cx - (ps_w - margin))
            x_max = min(w - ps_w, cx - margin)
            y_min = max(0, cy - (ps_h - margin))
            y_max = min(h - ps_h, cy - margin)
            x0 = int(rng.integers(x_min, x_max + 1)) if x_max >= x_min else int(np.clip(cx - ps_w // 2, 0, w - ps_w))
            y0 = int(rng.integers(y_min, y_max + 1)) if y_max >= y_min else int(np.clip(cy - ps_h // 2, 0, h - ps_h))
        elif hard_yx is not None:
            hy, hx = hard_yx
            x0 = int(np.clip(hx - ps_w // 2 + int(rng.integers(-6, 7)), 0, max(0, w - ps_w)))
            y0 = int(np.clip(hy - ps_h // 2 + int(rng.integers(-6, 7)), 0, max(0, h - ps_h)))
        else:
            x0 = int(rng.integers(0, w - ps_w + 1)) if w > ps_w else 0
            y0 = int(rng.integers(0, h - ps_h + 1)) if h > ps_h else 0
        win_inp = inp[:, y0 : y0 + ps_h, x0 : x0 + ps_w]

    return {
        "input": win_inp,
        "gt_heatmap": cached["gt_heatmap"][:, y0 : y0 + ps_h, x0 : x0 + ps_w],
        "gt_offset": cached["gt_offset"][:, y0 : y0 + ps_h, x0 : x0 + ps_w],
        "gt_log_scale": cached["gt_log_scale"][:, y0 : y0 + ps_h, x0 : x0 + ps_w],
        "center_mask": cached["center_mask"][:, y0 : y0 + ps_h, x0 : x0 + ps_w],
    }


def train_d1_steps(
    model: D1HeatmapDetector,
    samples: list[Mapping[str, Any]],
    epochs: int = 16,
    lr: float = 2e-3,
    use_synthesis: bool = True,
    seed: int = 20261107,
    device: str = "cpu",
    checkpoint_dir: Path | None = None,
    checkpoint_interval_minutes: float = 30.0,
) -> dict[str, Any]:
    """Run deterministic single-process pure-neural training with FocalLoss + hard-negative mining (`B8-R3-F1`, `B8-R3-F2`)."""
    torch.set_num_threads(2)
    dev = torch.device(device)
    model = model.to(device=dev)
    model.train()

    stem_params = list(model.enc1.parameters())
    stem_ids = {id(p) for p in stem_params}
    rest_params = [p for p in model.parameters() if id(p) not in stem_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": stem_params, "lr": float(lr) * 0.02, "weight_decay": 0.0},
            {"params": rest_params, "lr": float(lr) * 0.25, "weight_decay": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=float(lr) * 0.1
    )
    rng = np.random.default_rng(int(seed))
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    pos_samples = [s for s in samples if len(s.get("centers_uv", [])) > 0]
    neg_samples = [s for s in samples if len(s.get("centers_uv", [])) == 0]
    if not pos_samples or not neg_samples:
        balanced_schedule: list[tuple[Mapping[str, Any], bool]] = [(s, True) for s in samples]
    else:
        # Balance across real positives, pure background clutter negatives, and synthetic-injected multi-scale targets
        balanced_schedule = []
        n_pos = min(len(pos_samples), 64) if dev.type == "cpu" else len(pos_samples)
        for i in range(n_pos):
            balanced_schedule.append((pos_samples[i], False))
            neg_s = neg_samples[i % len(neg_samples)]
            balanced_schedule.append((neg_s, False))
            if use_synthesis:
                neg_syn = neg_samples[(i + 1) % len(neg_samples)]
                balanced_schedule.append((neg_syn, True))

    t_start = time.perf_counter()
    last_ckpt_t = t_start
    history: list[dict[str, float]] = []

    # Pre-register input frame stacks once before the epoch loop
    cached_items: list[dict[str, Any]] = []
    with torch.no_grad():
        for sched_sample, inj_flag in balanced_schedule:
            raw_item = build_training_sample_tensors(
                sample=sched_sample,
                num_frames_k=model.num_frames_k,
                use_synthesis=use_synthesis,
                rng=rng if sched_sample.get("split", "train") == "train" else None,
                ablation_mode=model.ablation_mode,
                inject_synthetic_on_negative=inj_flag,
            )
            is_pure_neg = float(raw_item["center_mask"].sum().item()) == 0.0
            cached_items.append(
                {
                    "input": raw_item["input"].cpu(),
                    "gt_heatmap": raw_item["gt_heatmap"],
                    "gt_offset": raw_item["gt_offset"],
                    "gt_log_scale": raw_item["gt_log_scale"],
                    "center_mask": raw_item["center_mask"],
                    "is_pure_neg": is_pure_neg,
                    "hard_yx": None,
                }
            )

    for epoch in range(int(epochs)):
        # Online hard-negative mining on pure background clutter stacks every 3 epochs
        if epoch > 0 and epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                for ci in cached_items:
                    if ci["is_pure_neg"]:
                        inp_full = ci["input"].unsqueeze(0).to(device=dev)
                        hm_full = model(inp_full)["heatmap"][0, 0]
                        flat_idx = int(torch.argmax(hm_full).item())
                        w_f = int(hm_full.shape[-1])
                        ci["hard_yx"] = (flat_idx // w_f, flat_idx % w_f)
            model.train()

        epoch_loss = 0.0
        n_steps = 0
        order = rng.permutation(len(cached_items)).tolist()
        batch_size = 8
        for b_start in range(0, len(order), batch_size):
            chunk_indices = order[b_start : b_start + batch_size]
            batch_items = [
                _crop_cached_window(
                    cached_items[idx],
                    rng,
                    crop_size=64,
                    hard_yx=cached_items[idx]["hard_yx"],
                )
                for idx in chunk_indices
            ]
            by_shape: dict[tuple[int, int], list[dict[str, torch.Tensor]]] = {}
            for bi in batch_items:
                sh = (int(bi["input"].shape[-2]), int(bi["input"].shape[-1]))
                by_shape.setdefault(sh, []).append(bi)

            for sub_items in by_shape.values():
                inp_batch = torch.stack([it["input"] for it in sub_items], dim=0).to(device=dev)
                targets = {
                    k: torch.stack([it[k] for it in sub_items], dim=0).to(device=dev)
                    for k in ("gt_heatmap", "gt_offset", "gt_log_scale", "center_mask")
                }

                optimizer.zero_grad(set_to_none=True)
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(inp_batch)
                        losses = compute_d1_losses(out, targets)
                else:
                    out = model(inp_batch)
                    losses = compute_d1_losses(out, targets)

                loss = losses["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                n_steps += 1

                now = time.perf_counter()
                if checkpoint_dir is not None and (now - last_ckpt_t) >= (checkpoint_interval_minutes * 60.0):
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = checkpoint_dir / f"ckpt_epoch_{epoch:03d}.pt"
                    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
                    last_ckpt_t = now

        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "mean_loss": epoch_loss / max(1, n_steps),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "steps": float(n_steps),
            }
        )

    wall_s = time.perf_counter() - t_start
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "epochs": epochs, "history": history},
            checkpoint_dir / "ckpt_final.pt",
        )

    return {
        "model_name": model.model_name,
        "num_frames_k": model.num_frames_k,
        "ablation_mode": model.ablation_mode,
        "use_synthesis": use_synthesis,
        "epochs": epochs,
        "wall_s": wall_s,
        "history": history,
    }
