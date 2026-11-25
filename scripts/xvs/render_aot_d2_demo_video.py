"""Standalone ONNX inference & avionics HUD video renderer for `XVS-D2` (AOT / KSFO).

Ready for live Colab G4 `v2_m2_01_xvs_d2_aot_g4_results.tar.zst` weights as well as
direct `.onnx` models (`xvs_d2_web_fp16.onnx`, `xvs_d2_edge_fp16.onnx`).

Features:
- Accepts `--onnx <path>` OR `--results-archive <tar.zst>` (safe Python 3.14
  `compression.zstd` + `tarfile` extraction with strict path-traversal & link guard).
- Accepts `--aot-dir <dir>`, `--images-glob <pattern>`, OR `--smoke-synthetic`
  (60-frame deterministic KSFO aerial flyby with a 6-12 px moving target & GT boxes).
- Runs genuine `onnxruntime` 6-channel dual-interval (`x_t [1, 6, H, W]`) + persistent
  ConvGRU (`h_prev_warped [1, C, H/4, W/4]` -> `h_next`) inference per frame, decoding
  stride-2 `hm_logits`, `offset`, `log_scale`, `log_var`, and `peak_mask` via exact
  local 3x3 windowed soft-argmax matching `xvs.v2.xvs_d2.heads`.
- Draws a crisp 1280x720 avionics HUD overlay: full frame, 4x zoomed target ROI PiP,
  GT (cyan) vs Pred (emerald) boxes, confidence score, frame index, and latency ms.
- Writes MP4 via `cv2.VideoWriter` and a 2x3 keyframe PNG contact sheet.
"""

from __future__ import annotations

import argparse
from compression import zstd
from dataclasses import dataclass
import glob
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KSFO_ORTHO = (
    REPO_ROOT / "apps" / "livedeck" / "public" / "assets" / "ksfo_ortho_inner_v1.jpg"
)
CANVAS_W = 1280
CANVAS_H = 720
HUD_TOP_H = 44
HUD_BOT_H = 36

# BGR colors for crisp avionics HUD
COLOR_BG = (14, 18, 24)
COLOR_PANEL = (22, 28, 36)
COLOR_BORDER = (66, 82, 102)
COLOR_TEXT = (232, 238, 246)
COLOR_MUTED = (148, 164, 184)
COLOR_GT_CYAN = (255, 224, 32)       # Cyan in BGR
COLOR_PRED_EMERALD = (118, 238, 64)  # Emerald in BGR
COLOR_AMBER = (56, 189, 248)         # Amber in BGR

PREFERRED_ARCHIVE_ONNX_NAMES = (
    "xvs_d2_web_fp16.onnx",
    "xvs_d2_edge_fp16.onnx",
)


@dataclass(frozen=True)
class Box2D:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    score: float = 1.0
    sigma_px: float = 0.0


@dataclass(frozen=True)
class FrameSample:
    frame_idx: int
    bgr: np.ndarray
    gt_boxes: tuple[Box2D, ...]
    source_name: str


def _is_safe_tar_member(member: tarfile.TarInfo) -> bool:
    """Reject absolute paths, parent traversals, symlinks, hardlinks, and device nodes."""
    if not member.isfile():
        return False
    if member.issym() or member.islnk():
        return False
    raw_name = member.name.replace("\\", "/")
    if raw_name.startswith("/") or ":" in raw_name:
        return False
    parts = [p for p in raw_name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return False
    return True


def extract_onnx_from_results_archive(archive_path: Path, dest_dir: Path) -> Path:
    """Safely extract `xvs_d2_web_fp16.onnx` or `xvs_d2_edge_fp16.onnx` from a `.tar.zst`."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Results archive not found: {archive_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_zst = archive_path.read_bytes()
    tar_bytes = zstd.decompress(raw_zst)

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        safe_onnx: dict[str, tarfile.TarInfo] = {}
        fallback_onnx: list[tarfile.TarInfo] = []
        for member in tf.getmembers():
            if not _is_safe_tar_member(member):
                continue
            basename = Path(member.name.replace("\\", "/")).name
            if basename in PREFERRED_ARCHIVE_ONNX_NAMES:
                safe_onnx[basename] = member
            elif basename.endswith(".onnx"):
                fallback_onnx.append(member)

        chosen: tarfile.TarInfo | None = None
        for preferred in PREFERRED_ARCHIVE_ONNX_NAMES:
            if preferred in safe_onnx:
                chosen = safe_onnx[preferred]
                break
        if chosen is None and fallback_onnx:
            fallback_onnx.sort(key=lambda m: m.name)
            chosen = fallback_onnx[0]

        if chosen is None:
            raise FileNotFoundError(
                f"No safe .onnx model found inside archive {archive_path}"
            )

        out_name = Path(chosen.name.replace("\\", "/")).name
        extracted = tf.extractfile(chosen)
        if extracted is None:
            raise RuntimeError(f"Could not read member '{chosen.name}' from {archive_path}")
        payload = extracted.read()
        out_path = dest_dir / out_name
        out_path.write_bytes(payload)
        return out_path


def scene_normalize_gray_np(
    gray_f32: np.ndarray,
    *,
    min_std: float = 8.0,
    clamp_val: float = 5.0,
) -> np.ndarray:
    """Exact NumPy equivalent of `xvs.v2.xvs_d2.stem.scene_normalize_frame`."""
    mean = float(gray_f32.mean())
    std = max(float(gray_f32.std()), float(min_std))
    normed = (gray_f32 - mean) / std
    return np.clip(normed, -clamp_val, clamp_val).astype(np.float32)


def estimate_translation_warp(
    prev_gray_f32: np.ndarray,
    curr_gray_f32: np.ndarray,
) -> tuple[float, float]:
    """Estimate smooth global translation `(dx, dy)` aligning `prev` onto `curr` via phase correlation."""
    h, w = curr_gray_f32.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(prev_gray_f32, curr_gray_f32, hann)
    if not (math.isfinite(dx) and math.isfinite(dy)) or response < 0.08:
        return 0.0, 0.0
    max_shift = 0.15 * float(min(h, w))
    dx = max(-max_shift, min(max_shift, float(dx)))
    dy = max(-max_shift, min(max_shift, float(dy)))
    return dx, dy


def warp_gray_and_mask(
    gray_f32: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp 2D grayscale image by `(dx, dy)` and emit binary valid-pixel mask `{0, 1}`."""
    h, w = gray_f32.shape
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(
        gray_f32,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    ones = np.ones((h, w), dtype=np.float32)
    mask = cv2.warpAffine(
        ones,
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    mask = (mask > 0.5).astype(np.float32)
    return warped, mask


def warp_hidden_state(h_prev: np.ndarray, dx_full: float, dy_full: float, stride: float = 4.0) -> np.ndarray:
    """Warp ConvGRU hidden state `[1, C, H/4, W/4]` by `(dx_full / 4, dy_full / 4)`."""
    if abs(dx_full) < 1e-4 and abs(dy_full) < 1e-4:
        return h_prev
    _, channels, hh, hw = h_prev.shape
    mat = np.array(
        [[1.0, 0.0, dx_full / stride], [0.0, 1.0, dy_full / stride]],
        dtype=np.float32,
    )
    out = np.empty_like(h_prev)
    for ch in range(channels):
        out[0, ch] = cv2.warpAffine(
            h_prev[0, ch],
            mat,
            (hw, hh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    return out


def build_dual_interval_tensor_np(
    curr_gray: np.ndarray,
    prev1_gray: np.ndarray | None,
    prev5_gray: np.ndarray | None,
) -> tuple[np.ndarray, float, float]:
    """Construct exact 6-channel `x_t [1, 6, H, W]` + `(dx1, dy1)` GMC translation."""
    c0 = scene_normalize_gray_np(curr_gray)
    h, w = c0.shape

    if prev1_gray is None:
        dx1, dy1 = 0.0, 0.0
        c1 = np.zeros((h, w), dtype=np.float32)
        r1 = np.zeros((h, w), dtype=np.float32)
    else:
        dx1, dy1 = estimate_translation_warp(prev1_gray, curr_gray)
        w1, r1 = warp_gray_and_mask(prev1_gray, dx1, dy1)
        c1 = scene_normalize_gray_np(w1) * r1

    if prev5_gray is None:
        c2 = np.zeros((h, w), dtype=np.float32)
        r5 = np.zeros((h, w), dtype=np.float32)
    else:
        dx5, dy5 = estimate_translation_warp(prev5_gray, curr_gray)
        w5, r5 = warp_gray_and_mask(prev5_gray, dx5, dy5)
        c2 = scene_normalize_gray_np(w5) * r5

    c3 = (c0 - c1) * r1
    c4 = (c0 - c2) * r5
    c5 = r1 * r5
    x_t = np.stack([c0, c1, c2, c3, c4, c5], axis=0)[np.newaxis, :, :, :].astype(np.float32)
    return x_t, dx1, dy1


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x.astype(np.float32), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def decode_xvs_d2_outputs_np(
    hm_logits: np.ndarray,
    offset: np.ndarray,
    log_scale: np.ndarray,
    log_var: np.ndarray,
    peak_mask: np.ndarray,
    *,
    score_thr: float = 0.25,
    max_k: int = 16,
    stride: float = 2.0,
    offset_blend: float = 0.5,
) -> tuple[list[Box2D], tuple[float, float, float]]:
    """Exact NumPy local 3x3 soft-argmax stride-2 decoder matching `xvs.v2.xvs_d2.heads`.

    Returns `(detections_in_model_input_px, (top1_u_px, top1_v_px, top1_prob))`.
    """
    hm = hm_logits[0, 0].astype(np.float32)
    off = offset[0].astype(np.float32)
    lsc = log_scale[0].astype(np.float32)
    lvr = log_var[0].astype(np.float32)
    pmk = peak_mask[0, 0].astype(np.float32)

    probs = _sigmoid_np(hm)
    h_sz, w_sz = probs.shape

    flat_top = int(np.argmax(probs))
    top_cy, top_cx = divmod(flat_top, w_sz)
    top_prob = float(probs[top_cy, top_cx])
    top_u = (float(top_cx) + 0.5) * stride
    top_v = (float(top_cy) + 0.5) * stride

    keep = (probs >= float(score_thr)) & (pmk > 0.5)
    ys, xs = np.where(keep)
    if ys.size == 0:
        return [], (top_u, top_v, top_prob)

    scores = probs[ys, xs]
    order = np.argsort(-scores)[: int(max_k)]
    ys = ys[order]
    xs = xs[order]

    boxes: list[Box2D] = []
    for idx in range(int(ys.size)):
        cy = int(ys[idx])
        cx = int(xs[idx])
        score = float(probs[cy, cx])

        y0 = max(0, cy - 1)
        y1 = min(h_sz, cy + 2)
        x0 = max(0, cx - 1)
        x1 = min(w_sz, cx + 2)
        patch = hm[y0:y1, x0:x1]
        patch_shift = patch - float(patch.max())
        exp_p = np.exp(patch_shift)
        weights = exp_p / max(float(exp_p.sum()), 1e-12)

        gy = np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]
        gx = np.arange(x0, x1, dtype=np.float32)[np.newaxis, :]
        soft_u = float((weights * gx).sum())
        soft_v = float((weights * gy).sum())

        pred_u = float(cx) + float(off[0, cy, cx])
        pred_v = float(cy) + float(off[1, cy, cx])
        cell_u = offset_blend * pred_u + (1.0 - offset_blend) * soft_u
        cell_v = offset_blend * pred_v + (1.0 - offset_blend) * soft_v

        u_px = (cell_u + 0.5) * stride
        v_px = (cell_v + 0.5) * stride

        ln_s = float(lsc[0, cy, cx])
        ln_r = float(lsc[1, cy, cx])
        s_mom = math.exp(max(-4.0, min(8.0, ln_s)))
        r_wh = math.exp(max(-3.0, min(3.0, ln_r)))
        bw = max(6.0, min(96.0, s_mom * math.sqrt(r_wh)))
        bh = max(6.0, min(96.0, s_mom / math.sqrt(r_wh)))

        var_u = math.exp(max(-6.0, min(4.0, float(lvr[0, cy, cx]))))
        var_v = math.exp(max(-6.0, min(4.0, float(lvr[1, cy, cx]))))
        sigma_px = math.sqrt(0.5 * (var_u + var_v))

        boxes.append(
            Box2D(
                x1=u_px - 0.5 * bw,
                y1=v_px - 0.5 * bh,
                x2=u_px + 0.5 * bw,
                y2=v_px + 0.5 * bh,
                label="AIRCRAFT",
                score=score,
                sigma_px=sigma_px,
            )
        )
    return boxes, (top_u, top_v, top_prob)


def generate_synthetic_ksfo_flyby(
    *,
    num_frames: int = 60,
    width: int = 960,
    height: int = 540,
    ortho_path: Path = DEFAULT_KSFO_ORTHO,
) -> list[FrameSample]:
    """Generate a deterministic 60-frame KSFO aerial flyby clip with a 6-12 px moving target."""
    rng = np.random.default_rng(20261121)

    ground_tex: np.ndarray | None = None
    if ortho_path.is_file():
        loaded = cv2.imread(str(ortho_path), cv2.IMREAD_COLOR)
        if loaded is not None and loaded.shape[0] >= 256 and loaded.shape[1] >= 256:
            ground_tex = loaded

    if ground_tex is None:
        # Fallback deterministic SF Bay / peninsula texture if ortho file is absent.
        yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float32)
        water_mask = (xx > (480.0 + 90.0 * np.sin(yy / 140.0))).astype(np.float32)
        base_b = 115.0 * water_mask + 92.0 * (1.0 - water_mask) + 12.0 * np.sin(xx / 28.0)
        base_g = 124.0 * water_mask + 118.0 * (1.0 - water_mask) + 10.0 * np.cos(yy / 31.0)
        base_r = 98.0 * water_mask + 112.0 * (1.0 - water_mask) + 8.0 * np.sin((xx + yy) / 45.0)
        ground_tex = np.clip(np.stack([base_b, base_g, base_r], axis=-1), 0, 255).astype(np.uint8)

    # Precompute smooth sky + horizon backdrop
    horizon_y = int(round(height * 0.44))
    sky_grad = np.linspace(0.0, 1.0, horizon_y, dtype=np.float32)[:, np.newaxis, np.newaxis]
    top_sky = np.array([[[192.0, 146.0, 102.0]]], dtype=np.float32)   # Crisp deep sky BGR
    hor_sky = np.array([[[226.0, 214.0, 198.0]]], dtype=np.float32)   # Marine haze horizon BGR
    sky_band = (top_sky * (1.0 - sky_grad) + hor_sky * sky_grad)
    sky_band = np.broadcast_to(sky_band, (horizon_y, width, 3)).copy()

    # Deterministic soft marine stratocumulus wisps
    cloud_noise = rng.normal(0.0, 1.0, size=(horizon_y // 8, width // 8)).astype(np.float32)
    cloud_noise = cv2.GaussianBlur(cloud_noise, (9, 9), 2.5)
    cloud_full = cv2.resize(cloud_noise, (width, horizon_y), interpolation=cv2.INTER_CUBIC)
    cloud_alpha = np.clip((cloud_full - 0.15) * 0.55, 0.0, 0.45)[:, :, np.newaxis]
    sky_band = np.clip(sky_band * (1.0 - cloud_alpha) + 244.0 * cloud_alpha, 0, 255)

    ground_h = height - horizon_y
    gh_tex, gw_tex = ground_tex.shape[:2]

    samples: list[FrameSample] = []
    for t in range(num_frames):
        frac = float(t) / float(max(1, num_frames - 1))
        frame = np.empty((height, width, 3), dtype=np.float32)

        # Gentle camera pan across sky
        sky_shift = int(round(frac * 26.0))
        frame[:horizon_y] = np.roll(sky_band, -sky_shift, axis=1)

        # Perspective-warped scrolling KSFO ortho ground plane
        scroll_y = 220.0 + frac * 170.0
        scroll_x = 170.0 + frac * 45.0
        src_pts = np.array(
            [
                [scroll_x + 120.0, scroll_y],
                [scroll_x + 620.0, scroll_y],
                [scroll_x + 740.0, min(gh_tex - 2.0, scroll_y + 420.0)],
                [scroll_x, min(gh_tex - 2.0, scroll_y + 420.0)],
            ],
            dtype=np.float32,
        )
        dst_pts = np.array(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(ground_h - 1)],
                [0.0, float(ground_h - 1)],
            ],
            dtype=np.float32,
        )
        persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        ground_slice = cv2.warpPerspective(
            ground_tex,
            persp,
            (width, ground_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)

        # Distance atmospheric haze near horizon
        haze_t = np.exp(-np.linspace(0.0, 4.2, ground_h, dtype=np.float32))[:, np.newaxis, np.newaxis]
        ground_slice = ground_slice * (1.0 - 0.62 * haze_t) + hor_sky * (0.62 * haze_t)
        frame[horizon_y:] = ground_slice

        # Subtle sensor grain
        grain = rng.normal(0.0, 1.8, size=(height, width, 1)).astype(np.float32)
        frame_u8 = np.clip(frame + grain, 0, 255).astype(np.uint8)

        # Moving airborne civil aircraft target (6 px -> 12 px apparent wingspan)
        cx = width * (0.20 + 0.58 * frac)
        cy = height * (0.31 + 0.06 * math.sin(frac * math.pi * 1.4) + 0.05 * frac)
        wingspan_px = 6.5 + 5.2 * frac
        body_h_px = max(4.0, wingspan_px * 0.52)

        ix = int(round(cx))
        iy = int(round(cy))
        half_w = max(3, int(round(wingspan_px * 0.5)))
        half_h = max(2, int(round(body_h_px * 0.5)))

        # Crisp dark fuselage + wings + bright specular strobe glint
        cv2.line(frame_u8, (ix - half_w, iy), (ix + half_w, iy), (28, 30, 36), 2, cv2.LINE_AA)
        cv2.line(
            frame_u8,
            (ix, iy - half_h),
            (ix, iy + max(1, half_h - 1)),
            (22, 24, 30),
            2,
            cv2.LINE_AA,
        )
        if t % 6 == 0:
            cv2.circle(frame_u8, (ix + max(1, half_w // 2), iy), 1, (245, 250, 255), -1, cv2.LINE_AA)

        gt_pad = 3.0
        gt_box = Box2D(
            x1=max(0.0, cx - 0.5 * wingspan_px - gt_pad),
            y1=max(0.0, cy - 0.5 * body_h_px - gt_pad),
            x2=min(float(width - 1), cx + 0.5 * wingspan_px + gt_pad),
            y2=min(float(height - 1), cy + 0.5 * body_h_px + gt_pad),
            label="INTRUDER",
            score=1.0,
        )
        samples.append(
            FrameSample(
                frame_idx=t,
                bgr=frame_u8,
                gt_boxes=(gt_box,),
                source_name=f"ksfo_synth_{t:04d}",
            )
        )
    return samples


def _load_optional_aot_gt(aot_dir: Path) -> dict[str, list[Box2D]]:
    """Load optional per-frame GT boxes from `annotations.json` / `gt.json` if present."""
    gt_map: dict[str, list[Box2D]] = {}
    for candidate in ("annotations.json", "gt.json", "labels.json"):
        path = aot_dir / candidate
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "frames" in payload and isinstance(payload["frames"], dict):
            frames_dict = payload["frames"]
        elif isinstance(payload, dict):
            frames_dict = payload
        else:
            continue
        for fname, items in frames_dict.items():
            if not isinstance(items, list):
                continue
            boxes: list[Box2D] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if "bbox_xyxy" in item:
                    x1, y1, x2, y2 = [float(v) for v in item["bbox_xyxy"]]
                elif "bbox_ltwh" in item:
                    l, t, w, h = [float(v) for v in item["bbox_ltwh"]]
                    x1, y1, x2, y2 = l, t, l + w, t + h
                elif "x1" in item and "y1" in item and "x2" in item and "y2" in item:
                    x1 = float(item["x1"])
                    y1 = float(item["y1"])
                    x2 = float(item["x2"])
                    y2 = float(item["y2"])
                else:
                    continue
                label = str(item["label"]) if "label" in item else "GT"
                boxes.append(Box2D(x1=x1, y1=y1, x2=x2, y2=y2, label=label, score=1.0))
            gt_map[str(fname)] = boxes
        break
    return gt_map


def load_frames_from_paths(
    paths: list[Path],
    *,
    gt_map: dict[str, list[Box2D]] | None = None,
    max_frames: int | None = None,
) -> list[FrameSample]:
    """Load image sequence from sorted filesystem paths."""
    if not paths:
        raise FileNotFoundError("No input frames matched the requested source.")
    if max_frames is not None and max_frames > 0:
        paths = paths[: int(max_frames)]

    samples: list[FrameSample] = []
    for idx, img_path in enumerate(paths):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to decode image frame: {img_path}")
        gt_boxes: tuple[Box2D, ...] = ()
        if gt_map is not None:
            if img_path.name in gt_map:
                gt_boxes = tuple(gt_map[img_path.name])
            elif img_path.stem in gt_map:
                gt_boxes = tuple(gt_map[img_path.stem])
        samples.append(
            FrameSample(
                frame_idx=idx,
                bgr=bgr,
                gt_boxes=gt_boxes,
                source_name=img_path.name,
            )
        )
    return samples


def _draw_corner_box(
    img: np.ndarray,
    box: Box2D,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    tag: str = "",
) -> None:
    x1 = int(round(box.x1))
    y1 = int(round(box.y1))
    x2 = int(round(box.x2))
    y2 = int(round(box.y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    corner = max(5, min(14, (x2 - x1) // 3, (y2 - y1) // 3))
    for (px, py, sx, sy) in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(img, (px, py), (px + sx * corner, py), color, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + sy * corner), color, thickness, cv2.LINE_AA)
    if tag:
        ty = max(16, y1 - 6)
        cv2.putText(
            img,
            tag,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )


def render_hud_frame(
    sample: FrameSample,
    *,
    pred_boxes_src: list[Box2D],
    top1_src_xy: tuple[float, float],
    top1_prob: float,
    hm_logits_2d: np.ndarray,
    latency_ms: float,
    hidden_l2: float,
    model_name: str,
    total_frames: int,
    score_thr: float,
) -> np.ndarray:
    """Compose a crisp 1280x720 avionics HUD frame with 4x target ROI PiP & telemetry."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), COLOR_BG, dtype=np.uint8)

    # Viewport area: full width below top bar and above bottom bar
    vp_x = 12
    vp_y = HUD_TOP_H + 8
    vp_w = CANVAS_W - 24
    vp_h = CANVAS_H - HUD_TOP_H - HUD_BOT_H - 16

    src_h, src_w = sample.bgr.shape[:2]
    scale = min(float(vp_w) / float(src_w), float(vp_h) / float(src_h))
    draw_w = max(1, int(round(src_w * scale)))
    draw_h = max(1, int(round(src_h * scale)))
    off_x = vp_x + (vp_w - draw_w) // 2
    off_y = vp_y + (vp_h - draw_h) // 2

    resized = cv2.resize(sample.bgr, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
    canvas[off_y : off_y + draw_h, off_x : off_x + draw_w] = resized
    cv2.rectangle(
        canvas,
        (off_x - 1, off_y - 1),
        (off_x + draw_w, off_y + draw_h),
        COLOR_BORDER,
        1,
    )

    def map_box(b: Box2D) -> Box2D:
        return Box2D(
            x1=off_x + b.x1 * scale,
            y1=off_y + b.y1 * scale,
            x2=off_x + b.x2 * scale,
            y2=off_y + b.y2 * scale,
            label=b.label,
            score=b.score,
            sigma_px=b.sigma_px,
        )

    for gt in sample.gt_boxes:
        _draw_corner_box(canvas, map_box(gt), COLOR_GT_CYAN, thickness=2, tag="GT")

    for pred in pred_boxes_src:
        tag = f"D2 {pred.score:.2f}"
        _draw_corner_box(canvas, map_box(pred), COLOR_PRED_EMERALD, thickness=2, tag=tag)

    # Choose 4x Target ROI PiP focus center: top pred -> first GT -> top-1 heatmap peak
    if pred_boxes_src:
        focus = pred_boxes_src[0]
        fx = 0.5 * (focus.x1 + focus.x2)
        fy = 0.5 * (focus.y1 + focus.y2)
        roi_tag = f"4X ROI PIP (PRED P={focus.score:.2f})"
    elif sample.gt_boxes:
        focus = sample.gt_boxes[0]
        fx = 0.5 * (focus.x1 + focus.x2)
        fy = 0.5 * (focus.y1 + focus.y2)
        roi_tag = "4X ROI PIP (GT LOCK)"
    else:
        fx, fy = top1_src_xy
        roi_tag = f"4X ROI PIP (PEAK P={top1_prob:.2e})"

    # Extract 64x64 source patch around (fx, fy) and zoom 4x -> 256x256 PiP
    pip_crop = 64
    pip_size = 256
    half_crop = pip_crop // 2
    cx_i = int(round(fx))
    cy_i = int(round(fy))

    padded = cv2.copyMakeBorder(
        sample.bgr,
        half_crop,
        half_crop,
        half_crop,
        half_crop,
        borderType=cv2.BORDER_REPLICATE,
    )
    px0 = max(0, min(src_w, cx_i))
    py0 = max(0, min(src_h, cy_i))
    crop_patch = padded[py0 : py0 + pip_crop, px0 : px0 + pip_crop]
    pip_zoom = cv2.resize(crop_patch, (pip_size, pip_size), interpolation=cv2.INTER_NEAREST)

    # Overlay GT and Pred boxes inside 4x PiP coordinates
    zoom_f = float(pip_size) / float(pip_crop)
    crop_origin_x = float(px0 - half_crop)
    crop_origin_y = float(py0 - half_crop)
    for gt in sample.gt_boxes:
        gx1 = (gt.x1 - crop_origin_x) * zoom_f
        gy1 = (gt.y1 - crop_origin_y) * zoom_f
        gx2 = (gt.x2 - crop_origin_x) * zoom_f
        gy2 = (gt.y2 - crop_origin_y) * zoom_f
        if gx2 > 0 and gy2 > 0 and gx1 < pip_size and gy1 < pip_size:
            cv2.rectangle(
                pip_zoom,
                (int(round(gx1)), int(round(gy1))),
                (int(round(gx2)), int(round(gy2))),
                COLOR_GT_CYAN,
                2,
                cv2.LINE_AA,
            )
    for pred in pred_boxes_src:
        px1 = (pred.x1 - crop_origin_x) * zoom_f
        py1 = (pred.y1 - crop_origin_y) * zoom_f
        px2 = (pred.x2 - crop_origin_x) * zoom_f
        py2 = (pred.y2 - crop_origin_y) * zoom_f
        if px2 > 0 and py2 > 0 and px1 < pip_size and py1 < pip_size:
            cv2.rectangle(
                pip_zoom,
                (int(round(px1)), int(round(py1))),
                (int(round(px2)), int(round(py2))),
                COLOR_PRED_EMERALD,
                2,
                cv2.LINE_AA,
            )

    # Subtle reticle crosshair inside PiP
    mid = pip_size // 2
    cv2.line(pip_zoom, (mid - 14, mid), (mid - 4, mid), COLOR_AMBER, 1, cv2.LINE_AA)
    cv2.line(pip_zoom, (mid + 4, mid), (mid + 14, mid), COLOR_AMBER, 1, cv2.LINE_AA)
    cv2.line(pip_zoom, (mid, mid - 14), (mid, mid - 4), COLOR_AMBER, 1, cv2.LINE_AA)
    cv2.line(pip_zoom, (mid, mid + 4), (mid, mid + 14), COLOR_AMBER, 1, cv2.LINE_AA)

    pip_x = CANVAS_W - pip_size - 24
    pip_y = HUD_TOP_H + 38
    cv2.rectangle(canvas, (pip_x - 2, pip_y - 22), (pip_x + pip_size + 2, pip_y + pip_size + 2), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (pip_x - 2, pip_y - 22), (pip_x + pip_size + 2, pip_y + pip_size + 2), COLOR_BORDER, 1)
    cv2.putText(
        canvas,
        roi_tag,
        (pip_x + 4, pip_y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        COLOR_AMBER,
        1,
        cv2.LINE_AA,
    )
    canvas[pip_y : pip_y + pip_size, pip_x : pip_x + pip_size] = pip_zoom

    # Compact stride-2 neural heatmap inset below the 4x PiP
    hm_min = float(hm_logits_2d.min())
    hm_max = float(hm_logits_2d.max())
    hm_norm = (hm_logits_2d - hm_min) / max(hm_max - hm_min, 1e-6)
    hm_u8 = np.clip(hm_norm * 255.0, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)
    hm_disp_w, hm_disp_h = 128, 128
    hm_resized = cv2.resize(hm_color, (hm_disp_w, hm_disp_h), interpolation=cv2.INTER_NEAREST)
    hm_x = CANVAS_W - hm_disp_w - 24
    hm_y = pip_y + pip_size + 32
    if hm_y + hm_disp_h < CANVAS_H - HUD_BOT_H - 6:
        cv2.rectangle(canvas, (hm_x - 2, hm_y - 20), (hm_x + hm_disp_w + 2, hm_y + hm_disp_h + 2), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (hm_x - 2, hm_y - 20), (hm_x + hm_disp_w + 2, hm_y + hm_disp_h + 2), COLOR_BORDER, 1)
        cv2.putText(
            canvas,
            "S2 HM LOGITS",
            (hm_x + 4, hm_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )
        canvas[hm_y : hm_y + hm_disp_h, hm_x : hm_x + hm_disp_w] = hm_resized

    # Top Avionics Telemetry Bar
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, HUD_TOP_H), COLOR_PANEL, -1)
    cv2.line(canvas, (0, HUD_TOP_H), (CANVAS_W, HUD_TOP_H), COLOR_BORDER, 1)
    top_left = f"OPENXVS // XVS-D2 ONNX DEMO  |  MODEL: {model_name}"
    top_right = (
        f"FRM {sample.frame_idx + 1:03d}/{total_frames:03d}  |  "
        f"ORT {latency_ms:5.1f} ms  |  "
        f"PREDS {len(pred_boxes_src)}  |  "
        f"MAX_P {top1_prob:.3g}"
    )
    cv2.putText(canvas, top_left, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, top_right, (660, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_PRED_EMERALD, 1, cv2.LINE_AA)

    # Bottom Legend & ConvGRU State Bar
    cv2.rectangle(canvas, (0, CANVAS_H - HUD_BOT_H), (CANVAS_W, CANVAS_H), COLOR_PANEL, -1)
    cv2.line(canvas, (0, CANVAS_H - HUD_BOT_H), (CANVAS_W, CANVAS_H - HUD_BOT_H), COLOR_BORDER, 1)
    bot_left = f"SRC: {sample.source_name}   |   CONVGRU ||h||2={hidden_l2:6.2f}   |   SCORE_THR={score_thr:.2f}"
    cv2.putText(canvas, bot_left, (16, CANVAS_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[CYAN] GT BOX", (970, CANVAS_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_GT_CYAN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[EMERALD] XVS-D2 PRED", (1095, CANVAS_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_PRED_EMERALD, 1, cv2.LINE_AA)

    return canvas


def build_contact_sheet_2x3(hud_frames: list[np.ndarray]) -> np.ndarray:
    """Assemble a crisp 2x3 keyframe contact sheet (1920x720) from rendered HUD frames."""
    if not hud_frames:
        raise ValueError("Cannot build contact sheet from empty frame list.")
    n = len(hud_frames)
    indices = [int(round(i * (n - 1) / 5.0)) for i in range(6)]
    tile_w, tile_h = 640, 360
    sheet = np.full((tile_h * 2, tile_w * 3, 3), COLOR_BG, dtype=np.uint8)
    for slot, fidx in enumerate(indices):
        row = slot // 3
        col = slot % 3
        tile = cv2.resize(hud_frames[fidx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return sheet


def _open_video_writer(out_mp4: Path, fps: float) -> tuple[cv2.VideoWriter, str]:
    """Open an MP4 `cv2.VideoWriter`, preferring `mp4v` with safe fallback to H.264 (`avc1`)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (CANVAS_W, CANVAS_H))
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Failed to initialize OpenCV VideoWriter for {out_mp4}")


def filter_boxes_by_max_frac(
    boxes: list[Box2D],
    *,
    frame_w: int,
    frame_h: int,
    max_box_frac: float | None,
) -> list[Box2D]:
    """Optionally suppress oversized ground-clutter boxes exceeding `max_box_frac` of frame dims."""
    if max_box_frac is None:
        return boxes
    frac = float(max_box_frac)
    if frac <= 0.0:
        raise ValueError(f"max_box_frac must be > 0.0 when provided, got {max_box_frac!r}")
    max_w = frac * float(frame_w)
    max_h = frac * float(frame_h)
    return [b for b in boxes if (b.x2 - b.x1) <= max_w and (b.y2 - b.y1) <= max_h]


def run_demo_video(
    *,
    onnx_path: Path | None = None,
    results_archive: Path | None = None,
    aot_dir: Path | None = None,
    images_glob: str | None = None,
    smoke_synthetic: bool = False,
    out_mp4: Path,
    out_contact_sheet: Path,
    score_thr: float = 0.25,
    fps: float = 15.0,
    num_synthetic_frames: int = 60,
    max_frames: int | None = None,
    max_box_frac: float | None = None,
) -> dict[str, object]:
    """Execute full `XVS-D2` ONNX inference loop and emit MP4 + 2x3 PNG contact sheet."""
    temp_dir: Path | None = None
    try:
        if results_archive is not None:
            temp_dir = Path(tempfile.mkdtemp(prefix="xvs_d2_archive_"))
            resolved_onnx = extract_onnx_from_results_archive(results_archive, temp_dir)
        elif onnx_path is not None:
            resolved_onnx = onnx_path
        else:
            raise ValueError("Provide either --onnx or --results-archive.")

        if not resolved_onnx.is_file():
            raise FileNotFoundError(f"ONNX model not found: {resolved_onnx}")

        if smoke_synthetic:
            samples = generate_synthetic_ksfo_flyby(num_frames=int(num_synthetic_frames))
        elif aot_dir is not None:
            if not aot_dir.is_dir():
                raise FileNotFoundError(f"AOT directory not found: {aot_dir}")
            exts = {".png", ".jpg", ".jpeg", ".bmp"}
            frame_paths = sorted(
                [p for p in aot_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
            )
            gt_map = _load_optional_aot_gt(aot_dir)
            samples = load_frames_from_paths(frame_paths, gt_map=gt_map, max_frames=max_frames)
        elif images_glob is not None:
            frame_paths = sorted(Path(p) for p in glob.glob(images_glob))
            samples = load_frames_from_paths(frame_paths, max_frames=max_frames)
        else:
            raise ValueError("Provide one of --smoke-synthetic, --aot-dir, or --images-glob.")

        sess = ort.InferenceSession(str(resolved_onnx), providers=["CPUExecutionProvider"])
        inputs_by_name = {inp.name: inp for inp in sess.get_inputs()}
        if "x_t" not in inputs_by_name or "h_prev_warped" not in inputs_by_name:
            raise ValueError(
                f"Expected ONNX inputs ('x_t', 'h_prev_warped'), got {sorted(inputs_by_name.keys())}"
            )

        x_shape = inputs_by_name["x_t"].shape
        h_shape = inputs_by_name["h_prev_warped"].shape
        in_h = int(x_shape[2]) if isinstance(x_shape[2], int) else 256
        in_w = int(x_shape[3]) if isinstance(x_shape[3], int) else 256
        hid_c = int(h_shape[1]) if isinstance(h_shape[1], int) else 24
        hid_h = int(h_shape[2]) if isinstance(h_shape[2], int) else in_h // 4
        hid_w = int(h_shape[3]) if isinstance(h_shape[3], int) else in_w // 4

        output_names = [out.name for out in sess.get_outputs()]
        out_idx = {name: i for i, name in enumerate(output_names)}
        for req in ("hm_logits", "offset", "log_scale", "log_var", "peak_mask", "h_next"):
            if req not in out_idx:
                raise ValueError(f"Missing expected ONNX output '{req}' in {output_names}")

        gray_history: list[np.ndarray] = []
        h_state = np.zeros((1, hid_c, hid_h, hid_w), dtype=np.float32)

        writer, used_codec = _open_video_writer(out_mp4, fps=fps)
        hud_frames: list[np.ndarray] = []
        latencies_ms: list[float] = []
        total_preds = 0
        max_prob_seen = 0.0

        try:
            for sample in samples:
                src_h, src_w = sample.bgr.shape[:2]
                gray_src = cv2.cvtColor(sample.bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gray_in = cv2.resize(gray_src, (in_w, in_h), interpolation=cv2.INTER_AREA)

                prev1 = gray_history[-1] if len(gray_history) >= 1 else None
                prev5 = gray_history[-5] if len(gray_history) >= 5 else (gray_history[0] if gray_history else None)

                x_t, dx1, dy1 = build_dual_interval_tensor_np(gray_in, prev1, prev5)
                h_warped = warp_hidden_state(h_state, dx1, dy1, stride=4.0)

                t0 = time.perf_counter()
                ort_outs = sess.run(
                    output_names,
                    {"x_t": x_t, "h_prev_warped": h_warped},
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(latency_ms)

                hm_logits = np.asarray(ort_outs[out_idx["hm_logits"]], dtype=np.float32)
                offset = np.asarray(ort_outs[out_idx["offset"]], dtype=np.float32)
                log_scale = np.asarray(ort_outs[out_idx["log_scale"]], dtype=np.float32)
                log_var = np.asarray(ort_outs[out_idx["log_var"]], dtype=np.float32)
                peak_mask = np.asarray(ort_outs[out_idx["peak_mask"]], dtype=np.float32)
                h_state = np.asarray(ort_outs[out_idx["h_next"]], dtype=np.float32)

                preds_in, (top_u_in, top_v_in, top_prob) = decode_xvs_d2_outputs_np(
                    hm_logits,
                    offset,
                    log_scale,
                    log_var,
                    peak_mask,
                    score_thr=score_thr,
                )
                max_prob_seen = max(max_prob_seen, top_prob)

                sx = float(src_w) / float(in_w)
                sy = float(src_h) / float(in_h)
                preds_src = [
                    Box2D(
                        x1=b.x1 * sx,
                        y1=b.y1 * sy,
                        x2=b.x2 * sx,
                        y2=b.y2 * sy,
                        label=b.label,
                        score=b.score,
                        sigma_px=b.sigma_px * 0.5 * (sx + sy),
                    )
                    for b in preds_in
                ]
                preds_src = filter_boxes_by_max_frac(
                    preds_src,
                    frame_w=src_w,
                    frame_h=src_h,
                    max_box_frac=max_box_frac,
                )
                total_preds += len(preds_src)

                hud = render_hud_frame(
                    sample,
                    pred_boxes_src=preds_src,
                    top1_src_xy=(top_u_in * sx, top_v_in * sy),
                    top1_prob=top_prob,
                    hm_logits_2d=hm_logits[0, 0],
                    latency_ms=latency_ms,
                    hidden_l2=float(np.linalg.norm(h_state)),
                    model_name=resolved_onnx.name,
                    total_frames=len(samples),
                    score_thr=score_thr,
                )
                writer.write(hud)
                hud_frames.append(hud)

                gray_history.append(gray_in)
                if len(gray_history) > 6:
                    gray_history.pop(0)
        finally:
            writer.release()

        sheet = build_contact_sheet_2x3(hud_frames)
        out_contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        ok_png = cv2.imwrite(str(out_contact_sheet), sheet)
        if not ok_png:
            raise RuntimeError(f"Failed to write contact sheet PNG: {out_contact_sheet}")

        if not out_mp4.is_file() or out_mp4.stat().st_size == 0:
            raise RuntimeError(f"Output MP4 is missing or empty: {out_mp4}")

        return {
            "model_filename": resolved_onnx.name,
            "input_hw": [in_h, in_w],
            "hidden_shape": [1, hid_c, hid_h, hid_w],
            "frames_rendered": len(hud_frames),
            "codec": used_codec,
            "fps": float(fps),
            "out_mp4": str(out_mp4),
            "out_mp4_bytes": int(out_mp4.stat().st_size),
            "out_contact_sheet": str(out_contact_sheet),
            "out_contact_sheet_bytes": int(out_contact_sheet.stat().st_size),
            "contact_sheet_hw": [int(sheet.shape[0]), int(sheet.shape[1])],
            "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
            "total_preds_above_thr": int(total_preds),
            "max_prob_seen": round(float(max_prob_seen), 8),
        }
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 1280x720 avionics HUD demo video + 2x3 contact sheet for XVS-D2 ONNX."
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--onnx", type=Path, default=None, help="Path to XVS-D2 .onnx model")
    model_group.add_argument(
        "--results-archive",
        type=Path,
        default=None,
        help="Path to Colab G4 v2_m2_01_xvs_d2_aot_g4_results.tar.zst",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--smoke-synthetic",
        action="store_true",
        help="Generate deterministic 60-frame KSFO aerial flyby with 6-12 px moving target & GT boxes",
    )
    source_group.add_argument("--aot-dir", type=Path, default=None, help="Directory of AOT frame images")
    source_group.add_argument("--images-glob", type=str, default=None, help="Glob pattern for input frames")

    parser.add_argument(
        "--out-mp4",
        type=Path,
        default=REPO_ROOT / "output" / "xvs" / "demos" / "xvs_d2_demo.mp4",
        help="Output 1280x720 MP4 path",
    )
    parser.add_argument(
        "--out-contact-sheet",
        type=Path,
        default=REPO_ROOT / "output" / "xvs" / "demos" / "xvs_d2_contact.png",
        help="Output 2x3 keyframe PNG contact sheet path",
    )
    parser.add_argument("--score-thr", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--fps", type=float, default=15.0, help="Output MP4 framerate")
    parser.add_argument("--num-frames", type=int, default=60, help="Frame count for --smoke-synthetic")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for directory/glob inputs")
    parser.add_argument(
        "--max-box-frac",
        type=float,
        default=None,
        help="Optional spatial clutter filter dropping pred boxes wider/taller than frac * frame size (default: off)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_demo_video(
        onnx_path=args.onnx,
        results_archive=args.results_archive,
        aot_dir=args.aot_dir,
        images_glob=args.images_glob,
        smoke_synthetic=bool(args.smoke_synthetic),
        out_mp4=args.out_mp4,
        out_contact_sheet=args.out_contact_sheet,
        score_thr=float(args.score_thr),
        fps=float(args.fps),
        num_synthetic_frames=int(args.num_frames),
        max_frames=args.max_frames,
        max_box_frac=args.max_box_frac,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
