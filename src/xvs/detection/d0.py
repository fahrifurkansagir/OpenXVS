"""Baseline classical morphological and local-normalized traffic detector D0.

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §1 and PROJECT_CHARTER §4-5.
Acceptance V7: imports nothing from xvs.traffic, xvs.metrics, xvs.sim.horizon, or mask I/O.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "xvs" / "detection" / "d0.json"


def load_d0_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load D0 detector configuration from JSON."""
    p = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"D0 config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = data.get("parameters", {})
    return {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in params.items()}


def compute_cmo(image: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute Close-Minus-Open (CMO) filter with a flat kxk square structuring element.

    CMO_k = closing_k(L) - opening_k(L) >= 0.
    Responds to both bright and dark features narrower than k pixels.
    """
    img = np.asarray(image, dtype=np.float64)
    closing = ndimage.grey_closing(img, size=(k, k), mode="nearest")
    opening = ndimage.grey_opening(img, size=(k, k), mode="nearest")
    cmo = np.maximum(0.0, closing - opening)
    return cmo


def compute_cmo_multi(image: np.ndarray, k_values: Sequence[int] = (3, 5, 7)) -> dict[int, np.ndarray]:
    """Compute CMO for multiple k values efficiently from the same image.
    
    Bit-identical to calling compute_cmo for each k individually.
    """
    img = np.asarray(image, dtype=np.float64)
    if set(k_values) == {3, 5, 7}:
        d3 = ndimage.grey_dilation(img, size=(3, 3), mode="nearest")
        e3 = ndimage.grey_erosion(img, size=(3, 3), mode="nearest")
        d5 = ndimage.grey_dilation(d3, size=(3, 3), mode="nearest")
        e5 = ndimage.grey_erosion(e3, size=(3, 3), mode="nearest")
        d7 = ndimage.grey_dilation(d5, size=(3, 3), mode="nearest")
        e7 = ndimage.grey_erosion(e5, size=(3, 3), mode="nearest")

        cl3 = ndimage.grey_erosion(d3, size=(3, 3), mode="nearest")
        op3 = ndimage.grey_dilation(e3, size=(3, 3), mode="nearest")

        cl5 = ndimage.grey_erosion(d5, size=(5, 5), mode="nearest")
        op5 = ndimage.grey_dilation(e5, size=(5, 5), mode="nearest")

        cl7 = ndimage.grey_erosion(d7, size=(7, 7), mode="nearest")
        op7 = ndimage.grey_dilation(e7, size=(7, 7), mode="nearest")

        return {
            3: np.maximum(0.0, cl3 - op3),
            5: np.maximum(0.0, cl5 - op5),
            7: np.maximum(0.0, cl7 - op7),
        }
    else:
        return {k: compute_cmo(img, k=k) for k in k_values}



def interpolate_bilinear_grid(
    grid: np.ndarray,
    h: int,
    w: int,
    stride: int = 32,
) -> np.ndarray:
    """Bilinearly interpolate a 2D sample grid (sampled at tile centres) to full (h, w) resolution.
    
    Optimized separable formulation, bit-identical to standard formulation.
    """
    ny, nx = grid.shape
    y0 = stride // 2
    x0 = stride // 2

    y = np.arange(h, dtype=np.float64)
    x = np.arange(w, dtype=np.float64)

    v = np.clip((y - y0) / stride, 0.0, float(ny - 1))
    u = np.clip((x - x0) / stride, 0.0, float(nx - 1))

    j0 = np.floor(v).astype(np.int64)
    j1 = np.minimum(j0 + 1, ny - 1)
    fy = (v - j0)[:, None]

    i0 = np.floor(u).astype(np.int64)
    i1 = np.minimum(i0 + 1, nx - 1)
    fx = (u - i0)[None, :]

    row_interp = (1.0 - fx) * grid[:, i0] + fx * grid[:, i1]
    out = (1.0 - fy) * row_interp[j0, :] + fy * row_interp[j1, :]
    return out


def compute_local_stats(
    cmo: np.ndarray,
    tile_size: int = 64,
    tile_stride: int = 32,
    mad_scale: float = 1.4826,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local median and robust noise scale (1.4826 * MAD) over tiles with bilinear interpolation.

    Tile centres are placed on a tile_stride grid (e.g. 16, 48, 80, ...).
    Each tile spans up to tile_size x tile_size, centered at the grid point.
    Vectorized sliding-window formulation with partition.
    """
    h, w = cmo.shape
    y_centres = np.arange(tile_stride // 2, h, tile_stride)
    x_centres = np.arange(tile_stride // 2, w, tile_stride)

    ny = len(y_centres)
    nx = len(x_centres)
    half = tile_size // 2

    y_starts = [max(0, min(h - tile_size, yc - half)) if h >= tile_size else 0 for yc in y_centres]
    x_starts = [max(0, min(w - tile_size, xc - half)) if w >= tile_size else 0 for xc in x_centres]

    if h >= tile_size and w >= tile_size:
        patches = np.empty((ny, nx, tile_size * tile_size), dtype=np.float64)
        for j, ys in enumerate(y_starts):
            for i, xs in enumerate(x_starts):
                patches[j, i] = cmo[ys:ys + tile_size, xs:xs + tile_size].ravel()
        N = tile_size * tile_size
        mid = N // 2
        if N % 2 == 1:
            p1 = np.partition(patches, mid, axis=-1)
            med_flat = p1[..., mid]
            dev = np.abs(patches - med_flat[..., None])
            p2 = np.partition(dev, mid, axis=-1)
            mad_flat = p2[..., mid]
        else:
            p1 = np.partition(patches, (mid - 1, mid), axis=-1)
            med_flat = 0.5 * (p1[..., mid - 1] + p1[..., mid])
            dev = np.abs(patches - med_flat[..., None])
            p2 = np.partition(dev, (mid - 1, mid), axis=-1)
            mad_flat = 0.5 * (p2[..., mid - 1] + p2[..., mid])
        m_grid = med_flat.reshape(ny, nx)
        sig_grid = (mad_scale * mad_flat).reshape(ny, nx)
    else:
        m_grid = np.zeros((ny, nx), dtype=np.float64)
        sig_grid = np.zeros((ny, nx), dtype=np.float64)
        for j, y_start in enumerate(y_starts):
            y_end = min(h, y_start + tile_size)
            for i, x_start in enumerate(x_starts):
                x_end = min(w, x_start + tile_size)
                patch = cmo[y_start:y_end, x_start:x_end]
                med = float(np.median(patch))
                mad = float(np.median(np.abs(patch - med)))
                m_grid[j, i] = med
                sig_grid[j, i] = mad_scale * mad

    m_loc = interpolate_bilinear_grid(m_grid, h, w, stride=tile_stride)
    sig_loc = interpolate_bilinear_grid(sig_grid, h, w, stride=tile_stride)
    return m_loc, sig_loc


def compute_srgb_quantization_sigma(linear_luminance: float | np.ndarray) -> float | np.ndarray:
    """Compute 8-bit quantization sigma through the sRGB EOTF curve at linear luminance L.

    Conforms to WP-04 spec v0.1 §1 and GEMINI_G-B8 A4 (F-B7-5).
    In 8-bit sRGB, sigma_s = 1 / (255 * sqrt(12)).
    Through linear EOTF:
      L <= 0.0031308: dL/ds = 1 / 12.92
      L > 0.0031308: dL/ds = (2.4 / 1.055) * L^(7/12)
    sigma_quant(L) = sigma_s * |dL/ds|.
    """
    l_arr = np.asarray(linear_luminance, dtype=np.float64)
    sigma_s = 1.0 / (255.0 * math.sqrt(12.0))
    c_linear = 1.0 / 12.92
    c_gamma = 2.4 / 1.055

    threshold = 0.0031308
    dL_ds = np.where(
        l_arr <= threshold,
        c_linear,
        c_gamma * np.power(np.maximum(0.0, l_arr), 7.0 / 12.0),
    )
    res = sigma_s * dL_ds
    if np.ndim(linear_luminance) == 0:
        return float(res)
    return res


def compute_clean_floor_grid(
    image: np.ndarray,
    tile_size: int = 64,
    tile_stride: int = 32,
) -> np.ndarray:
    """Compute bilinearly interpolated clean image floor sigma_min from tile median luminance.
    
    Vectorized sliding-window formulation.
    """
    h, w = image.shape
    y_centres = np.arange(tile_stride // 2, h, tile_stride)
    x_centres = np.arange(tile_stride // 2, w, tile_stride)

    ny = len(y_centres)
    nx = len(x_centres)
    half = tile_size // 2

    y_starts = [max(0, min(h - tile_size, yc - half)) if h >= tile_size else 0 for yc in y_centres]
    x_starts = [max(0, min(w - tile_size, xc - half)) if w >= tile_size else 0 for xc in x_centres]

    if h >= tile_size and w >= tile_size:
        patches = np.empty((ny, nx, tile_size * tile_size), dtype=np.float64)
        for j, ys in enumerate(y_starts):
            for i, xs in enumerate(x_starts):
                patches[j, i] = image[ys:ys + tile_size, xs:xs + tile_size].ravel()
        N = tile_size * tile_size
        mid = N // 2
        if N % 2 == 1:
            med_flat = np.partition(patches, mid, axis=-1)[..., mid]
        else:
            p1 = np.partition(patches, (mid - 1, mid), axis=-1)
            med_flat = 0.5 * (p1[..., mid - 1] + p1[..., mid])
        sig_min_grid = compute_srgb_quantization_sigma(med_flat).reshape(ny, nx)
    else:
        sig_min_grid = np.zeros((ny, nx), dtype=np.float64)
        for j, y_start in enumerate(y_starts):
            y_end = min(h, y_start + tile_size)
            for i, x_start in enumerate(x_starts):
                x_end = min(w, x_start + tile_size)
                patch = image[y_start:y_end, x_start:x_end]
                med_l = float(np.median(patch))
                sig_min_grid[j, i] = compute_srgb_quantization_sigma(med_l)

    sig_min_loc = interpolate_bilinear_grid(sig_min_grid, h, w, stride=tile_stride)
    return sig_min_loc


def compute_z_score(
    cmo: np.ndarray,
    m_loc: np.ndarray,
    sig_loc: np.ndarray,
    sigma_min: float | np.ndarray = 0.001,
) -> np.ndarray:
    """Compute normalized detection score z = (CMO - m_loc) / max(sig_loc, sigma_min)."""
    denom = np.maximum(sig_loc, sigma_min)
    z = (cmo - m_loc) / denom
    return z


def extract_components(
    z: np.ndarray,
    cmo: np.ndarray,
    tau: float = 4.0,
    u0: float = 0.0,
    v0: float = 0.0,
    image_id: str = "",
    camera: str = "",
    dilation_size: int = 3,
) -> list[dict[str, Any]]:
    """Extract candidate target detections from thresholded z-score field via connected components."""
    binary_mask = (z >= tau)
    if not np.any(binary_mask):
        return []

    struct = np.ones((dilation_size, dilation_size), dtype=bool)
    dilated_mask = ndimage.binary_dilation(binary_mask, structure=struct)
    labels, num_features = ndimage.label(dilated_mask, structure=struct)

    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return []

    pix_labels = labels[ys, xs]
    order = np.argsort(pix_labels)
    sorted_labels = pix_labels[order]
    sorted_ys = ys[order]
    sorted_xs = xs[order]
    sorted_weights = cmo[sorted_ys, sorted_xs]
    sorted_z = z[sorted_ys, sorted_xs]

    uq_labels, split_indices = np.unique(sorted_labels, return_index=True)
    split_indices = np.append(split_indices, len(order))

    detections = []
    for k in range(len(uq_labels)):
        s_start = split_indices[k]
        s_end = split_indices[k + 1]
        sl_xs = sorted_xs[s_start:s_end]
        sl_ys = sorted_ys[s_start:s_end]
        sl_w = sorted_weights[s_start:s_end]
        sl_z = sorted_z[s_start:s_end]

        sum_w = float(np.sum(sl_w))
        if sum_w > 1e-12:
            u_roi = float(np.sum(sl_w * sl_xs) / sum_w)
            v_roi = float(np.sum(sl_w * sl_ys) / sum_w)
        else:
            u_roi = float(np.mean(sl_xs))
            v_roi = float(np.mean(sl_ys))

        x_min = int(np.min(sl_xs))
        x_max = int(np.max(sl_xs))
        y_min = int(np.min(sl_ys))
        y_max = int(np.max(sl_ys))

        box_edges = [
            float(x_min - 0.5 + u0),
            float(y_min - 0.5 + v0),
            float(x_max + 0.5 + u0),
            float(y_max + 0.5 + v0),
        ]
        s_det = float(max(x_max - x_min + 1, y_max - y_min + 1))
        score = float(np.max(sl_z))
        n_px = int(len(sl_xs))

        detections.append({
            "image_id": image_id,
            "camera": camera,
            "u": float(u_roi + u0),
            "v": float(v_roi + v0),
            "box_edges": box_edges,
            "s_det": s_det,
            "score": score,
            "n_px": n_px,
        })

    return detections


def detect_d0(
    image: np.ndarray,
    k: int = 5,
    tau: float = 4.0,
    is_clean: bool = True,
    config: dict[str, Any] | None = None,
    u0: float = 0.0,
    v0: float = 0.0,
    image_id: str = "",
    camera: str = "",
    clean_floor: np.ndarray | float | None = None,
    cmo: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Execute complete D0 detection pipeline on a single 2D linear image/ROI."""
    if config is None:
        config = load_d0_config()

    tile_size = config.get("tile_size", 64)  # default-ok: algorithm parameter default
    tile_stride = config.get("tile_stride", 32)  # default-ok: algorithm parameter default
    mad_scale = config.get("mad_scale", 1.4826)  # default-ok: normal consistency scale default
    dilation_size = config.get("dilation_kernel_size", 3)  # default-ok: morphological kernel size default

    if is_clean:
        if clean_floor is not None:
            sigma_min = clean_floor
        else:
            cfg_sig = config.get("sigma_min_clean")
            if isinstance(cfg_sig, (int, float)):
                sigma_min = float(cfg_sig)
            else:
                sigma_min = compute_clean_floor_grid(image, tile_size=tile_size, tile_stride=tile_stride)
    else:
        sigma_min = float(config["sigma_min_degraded"])

    if cmo is None:
        cmo = compute_cmo(image, k=k)
    m_loc, sig_loc = compute_local_stats(cmo, tile_size=tile_size, tile_stride=tile_stride, mad_scale=mad_scale)
    z = compute_z_score(cmo, m_loc, sig_loc, sigma_min=sigma_min)
    dets = extract_components(
        z=z,
        cmo=cmo,
        tau=tau,
        u0=u0,
        v0=v0,
        image_id=image_id,
        camera=camera,
        dilation_size=dilation_size,
    )
    return dets
