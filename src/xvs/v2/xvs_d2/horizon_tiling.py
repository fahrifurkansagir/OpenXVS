"""Native-Resolution `3x1` Overlapping Horizon-Band Tiling for `XVS-D2` (DO-365B / DO-387).

Recovering `6-12 px` distant aircraft (`1.5-3.0 km`) on `1920x1080` or `2448x2048` sensors
requires avoiding sub-Nyquist full-frame downscaling (which crushes `9 px` targets to `~2.3 px`).
This module extracts `3` overlapping native-resolution `640x640` horizontal tiles centered on
the IMU-leveled horizon band (`extract_horizon_tiles`), runs `XVS-D2` at `1:1` pixel fidelity,
re-projects tile detections back to full-frame pixel coordinates, and merges seam duplicates
via greedy centroid-distance / IoU NMS (`merge_tile_detections`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

__all__ = [
    "HorizonTileSpec",
    "compute_horizon_tile_boxes",
    "extract_horizon_tiles",
    "reproject_tile_detection",
    "merge_tile_detections",
    "run_horizon_tiled_d2_inference",
]


@dataclass(frozen=True)
class HorizonTileSpec:
    """Exact native-resolution crop window and re-projection scale for one horizon tile."""

    tile_index: int
    x0: int
    y0: int
    x1: int
    y1: int
    tile_w: int
    tile_h: int
    scale_x: float
    scale_y: float

    @property
    def crop_w(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def crop_h(self) -> int:
        return int(self.y1 - self.y0)


def compute_horizon_tile_boxes(
    img_w: int,
    img_h: int,
    *,
    tile_hw: tuple[int, int] = (640, 640),
    num_tiles: int = 3,
    overlap_frac: float = 0.10,
    horizon_y_px: float | None = None,
) -> list[HorizonTileSpec]:
    """Compute `num_tiles` (`3x1` default) overlapping native-res horizon-band tile windows.

    - Centers the vertical band on `horizon_y_px` (defaults to `0.5 * img_h`), clamped to `[0, img_h]`.
    - When `img_w >= tile_w`, preserves strict `1:1` native pixel resolution (`scale_x == 1.0`)
      with at least `overlap_frac` horizontal overlap between adjacent tiles.
    - When `img_w < tile_w` or `img_h < tile_h` (e.g. small smoke frames), partitions cleanly
      and records exact `(scale_x, scale_y)` so re-projection back to full frame stays exact.
    """
    w = int(img_w)
    h = int(img_h)
    th, tw = int(tile_hw[0]), int(tile_hw[1])
    n_t = max(1, int(num_tiles))
    ov = float(max(0.0, min(0.5, overlap_frac)))

    if w <= 0 or h <= 0 or tw <= 0 or th <= 0:
        raise ValueError(f"Invalid frame/tile dims: frame=({w},{h}), tile=({tw},{th})")

    hor_y = (0.5 * float(h)) if horizon_y_px is None else float(horizon_y_px)
    if not math.isfinite(hor_y):
        hor_y = 0.5 * float(h)

    crop_h = min(h, th)
    y0 = int(round(hor_y - 0.5 * float(crop_h)))
    y0 = max(0, min(h - crop_h, y0))
    y1 = y0 + crop_h
    scale_y = float(th) / float(crop_h)

    crop_w = min(w, tw)
    scale_x = float(tw) / float(crop_w)

    if n_t == 1 or w <= crop_w:
        x0 = max(0, (w - crop_w) // 2)
        return [
            HorizonTileSpec(
                tile_index=0,
                x0=x0,
                y0=y0,
                x1=x0 + crop_w,
                y1=y1,
                tile_w=tw,
                tile_h=th,
                scale_x=scale_x,
                scale_y=scale_y,
            )
        ]

    # Ideal step giving exact `overlap_frac` overlap between adjacent tiles at 1:1 native resolution
    step_ideal = max(1, int(round(float(crop_w) * (1.0 - ov))))
    ideal_span = crop_w + (n_t - 1) * step_ideal

    if w <= ideal_span:
        # Frame fits completely across `n_t` tiles with >= overlap_frac overlap (e.g., 1920px / 3x640)
        xs = np.linspace(0.0, float(w - crop_w), num=n_t)
        x0_list = [max(0, min(w - crop_w, int(round(float(x))))) for x in xs]
    else:
        # Wide sensor (e.g. 2448px): center the 1:1 native-res `n_t`-tile band around frame center
        start_x = max(0, (w - ideal_span) // 2)
        x0_list = [max(0, min(w - crop_w, start_x + i * step_ideal)) for i in range(n_t)]

    specs: list[HorizonTileSpec] = []
    for idx, x0 in enumerate(x0_list):
        specs.append(
            HorizonTileSpec(
                tile_index=idx,
                x0=x0,
                y0=y0,
                x1=x0 + crop_w,
                y1=y1,
                tile_w=tw,
                tile_h=th,
                scale_x=scale_x,
                scale_y=scale_y,
            )
        )
    return specs


def extract_horizon_tiles(
    frame: np.ndarray,
    *,
    tile_hw: tuple[int, int] = (640, 640),
    num_tiles: int = 3,
    overlap_frac: float = 0.10,
    horizon_y_px: float | None = None,
) -> tuple[list[np.ndarray], list[HorizonTileSpec]]:
    """Extract `3x1` native-resolution overlapping horizon-band tiles from a 2D or 3D image."""
    arr = np.asarray(frame)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D grayscale or 3D HWC frame, got shape {arr.shape}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    specs = compute_horizon_tile_boxes(
        w,
        h,
        tile_hw=tile_hw,
        num_tiles=num_tiles,
        overlap_frac=overlap_frac,
        horizon_y_px=horizon_y_px,
    )
    tiles: list[np.ndarray] = []
    for spec in specs:
        crop = arr[spec.y0 : spec.y1, spec.x0 : spec.x1]
        if crop.shape[0] != spec.tile_h or crop.shape[1] != spec.tile_w:
            crop = cv2.resize(crop, (spec.tile_w, spec.tile_h), interpolation=cv2.INTER_LINEAR)
        tiles.append(np.ascontiguousarray(crop))
    return tiles, specs


def reproject_tile_detection(
    det: Mapping[str, Any],
    spec: HorizonTileSpec,
) -> dict[str, Any]:
    """Re-project a single tile-local detection dict back to full-frame pixel coordinates."""
    if "u_px" not in det or "v_px" not in det or "score" not in det:
        raise KeyError("Detection dict must contain 'u_px', 'v_px', and 'score'")
    tile_u = float(det["u_px"])
    tile_v = float(det["v_px"])
    full_u = float(spec.x0) + tile_u / float(spec.scale_x)
    full_v = float(spec.y0) + tile_v / float(spec.scale_y)

    out = dict(det)
    out["u_px"] = full_u
    out["v_px"] = full_v
    out["tile_u_px"] = tile_u
    out["tile_v_px"] = tile_v
    out["tile_index"] = int(spec.tile_index)

    inv_s = 1.0 / math.sqrt(max(1.0e-12, float(spec.scale_x) * float(spec.scale_y)))
    if "s_mom_px" in det:
        out["s_mom_px"] = float(det["s_mom_px"]) * inv_s
    if "w" in det:
        out["w"] = float(det["w"]) / float(spec.scale_x)
    if "h" in det:
        out["h"] = float(det["h"]) / float(spec.scale_y)
    if "var_u" in det:
        out["var_u"] = float(det["var_u"]) / (float(spec.scale_x) ** 2)
    if "var_v" in det:
        out["var_v"] = float(det["var_v"]) / (float(spec.scale_y) ** 2)
    return out


def _det_box_xyxy(det: Mapping[str, Any]) -> tuple[float, float, float, float]:
    u = float(det["u_px"])
    v = float(det["v_px"])
    if "w" in det and "h" in det:
        bw = max(2.0, float(det["w"]))
        bh = max(2.0, float(det["h"]))
    elif "s_mom_px" in det:
        s = max(2.0, float(det["s_mom_px"]))
        bw, bh = s, s
    else:
        bw, bh = 8.0, 8.0
    return (u - 0.5 * bw, v - 0.5 * bh, u + 0.5 * bw, v + 0.5 * bh)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 1.0e-12 else 0.0


def merge_tile_detections(
    tile_detections: Sequence[Sequence[Mapping[str, Any]]],
    tile_specs: Sequence[HorizonTileSpec],
    *,
    nms_radius_px: float = 12.0,
    nms_iou_thr: float = 0.35,
    max_k: int = 64,
) -> list[dict[str, Any]]:
    """Re-project detections from all horizon tiles to full frame and deduplicate via greedy NMS."""
    if len(tile_detections) != len(tile_specs):
        raise ValueError(
            f"tile_detections length ({len(tile_detections)}) != tile_specs length ({len(tile_specs)})"
        )
    projected: list[dict[str, Any]] = []
    for dets, spec in zip(tile_detections, tile_specs):
        for det in dets:
            projected.append(reproject_tile_detection(det, spec))

    projected.sort(key=lambda d: float(d["score"]), reverse=True)

    kept: list[dict[str, Any]] = []
    kept_boxes: list[tuple[float, float, float, float]] = []
    rad_thr = float(max(0.0, nms_radius_px))
    iou_thr = float(max(0.0, nms_iou_thr))

    for cand in projected:
        if len(kept) >= int(max_k):
            break
        cu = float(cand["u_px"])
        cv_px = float(cand["v_px"])
        cbox = _det_box_xyxy(cand)
        suppressed = False
        for k_det, k_box in zip(kept, kept_boxes):
            dist = math.hypot(cu - float(k_det["u_px"]), cv_px - float(k_det["v_px"]))
            if dist <= rad_thr or _box_iou(cbox, k_box) >= iou_thr:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
            kept_boxes.append(cbox)
    return kept


def run_horizon_tiled_d2_inference(
    frames_gray: Sequence[np.ndarray],
    tile_infer_fn: Callable[[Sequence[np.ndarray], int], list[dict[str, Any]]],
    *,
    tile_hw: tuple[int, int] = (640, 640),
    num_tiles: int = 3,
    overlap_frac: float = 0.10,
    horizon_y_px: float | None = None,
    nms_radius_px: float = 12.0,
    nms_iou_thr: float = 0.35,
    max_k: int = 64,
) -> tuple[list[dict[str, Any]], list[HorizonTileSpec]]:
    """Extract `3x1` native-res temporal tile stacks across `frames_gray`, run `tile_infer_fn`, and merge via NMS."""
    if not frames_gray:
        raise ValueError("frames_gray must be non-empty")
    first = np.asarray(frames_gray[0])
    h, w = int(first.shape[0]), int(first.shape[1])
    specs = compute_horizon_tile_boxes(
        w,
        h,
        tile_hw=tile_hw,
        num_tiles=num_tiles,
        overlap_frac=overlap_frac,
        horizon_y_px=horizon_y_px,
    )
    tile_stacks: list[list[np.ndarray]] = [[] for _ in specs]
    for frame in frames_gray:
        tiles_t, _ = extract_horizon_tiles(
            frame,
            tile_hw=tile_hw,
            num_tiles=num_tiles,
            overlap_frac=overlap_frac,
            horizon_y_px=horizon_y_px,
        )
        for ti, tile_img in enumerate(tiles_t):
            tile_stacks[ti].append(tile_img)

    per_tile_dets = [tile_infer_fn(stack, ti) for ti, stack in enumerate(tile_stacks)]
    merged = merge_tile_detections(
        per_tile_dets,
        specs,
        nms_radius_px=nms_radius_px,
        nms_iou_thr=nms_iou_thr,
        max_k=max_k,
    )
    return merged, specs
