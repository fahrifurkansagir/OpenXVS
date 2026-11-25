"""Fault injection models for WP-07 G0 integrity evaluation.

Implements the four fault models specified in docs/xvs/WP07_INTEGRITY.md §1:
1. Occlusion: 30% of one horizontal camera's columns set to a flat grey.
2. Spurious edge: bright or dark straight edge 5-40 px from the true horizon,
   across 40-100% of the width of one camera.
3. Altitude bias: h_meas = h_true + 200 m.
4. Glare: saturated disc of radius 150 px near the horizon.

All injections are strictly seeded and logged for V3 exact reproducibility.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence
import numpy as np


def compute_sha256(arr: np.ndarray) -> str:
    """Compute SHA-256 hex digest of a numpy array in C-contiguous order."""
    c_arr = np.ascontiguousarray(arr)
    return hashlib.sha256(c_arr.tobytes()).hexdigest()


def inject_occlusion(
    image: np.ndarray,
    col_fraction: float = 0.30,
    col_start: int | None = None,
    grey_value: int | float | None = None,
    rng: np.random.Generator | None = None,
    cam_id: str = "F",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inject occlusion fault: 30% of horizontal camera columns set to flat grey.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D monochrome or 3D RGB).
    col_fraction : float
        Fraction of width to occlude (default 0.30 = 30%).
    col_start : int, optional
        Starting column. If None, chosen uniformly via rng.
    grey_value : int or float, optional
        Grey pixel value. Default 128 for uint8, 0.5 for float.
    rng : np.random.Generator, optional
        Random number generator for deterministic draws.
    cam_id : str
        Camera identifier (e.g. 'F', 'R', 'B', 'L').

    Returns
    -------
    tuple of (modified_image, fault_log)
    """
    img = np.copy(image)
    h, w = img.shape[:2]
    n_cols = max(1, int(round(col_fraction * w)))

    if col_start is None:
        if rng is not None:
            max_start = max(0, w - n_cols)
            col_start = int(rng.integers(0, max_start + 1))
        else:
            col_start = int(max(0, (w - n_cols) // 2))

    col_end = min(w, col_start + n_cols)

    if grey_value is None:
        if np.issubdtype(img.dtype, np.integer):
            grey_val = 128 if img.dtype == np.uint8 else 32768
        else:
            grey_val = 0.5
    else:
        grey_val = grey_value

    if img.ndim == 2:
        img[:, col_start:col_end] = grey_val
    else:
        img[:, col_start:col_end, :] = grey_val

    log: dict[str, Any] = {
        "fault_type": "occlusion",
        "cam_id": cam_id,
        "col_fraction": float(col_fraction),
        "col_start": int(col_start),
        "col_end": int(col_end),
        "n_cols": int(n_cols),
        "grey_value": float(grey_val),
        "sha256": compute_sha256(img),
    }
    return img, log


def _fit_horizon_line(
    horizon_uv: Sequence[Sequence[float]] | np.ndarray,
    h_img: int,
    w_img: int,
) -> tuple[float, float]:
    """Fit a linear line v = m * u + c from horizon UV points or fallback."""
    pts = np.asarray(horizon_uv, dtype=np.float64)
    if len(pts) >= 2:
        u = pts[:, 0]
        v = pts[:, 1]
        # Robust linear fit: v = m*u + c
        if np.max(u) - np.min(u) > 10.0:
            p = np.polyfit(u, v, 1)
            return float(p[0]), float(p[1])
        return 0.0, float(np.mean(v))
    return 0.0, float(h_img / 2.0)


def inject_spurious_edge(
    image: np.ndarray,
    horizon_uv: Sequence[Sequence[float]] | np.ndarray,
    offset_px: float | None = None,
    width_fraction: float | None = None,
    u_start_fraction: float | None = None,
    polarity: str | None = None,
    rng: np.random.Generator | None = None,
    cam_id: str = "F",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inject spurious edge: bright or dark straight edge 5-40 px from true horizon.

    Across 40-100% of the width of one camera.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    horizon_uv : sequence of [u, v]
        Ground-truth horizon polyline for this camera.
    offset_px : float, optional
        Signed distance in pixels from true horizon (magnitude in [5, 40]).
    width_fraction : float, optional
        Fraction of image width spanned by edge in [0.40, 1.00].
    u_start_fraction : float, optional
        Fractional start position across image width.
    polarity : str, optional
        'bright' or 'dark' edge transition.
    rng : np.random.Generator, optional
        Random generator for deterministic parameter selection.
    cam_id : str
        Camera identifier.

    Returns
    -------
    tuple of (modified_image, fault_log)
    """
    img = np.copy(image)
    h, w = img.shape[:2]

    # Parameter selection
    if offset_px is None:
        if rng is not None:
            # 5-40 px either above (-) or below (+) true horizon
            sign = 1.0 if rng.random() > 0.5 else -1.0
            mag = float(rng.uniform(5.0, 40.0))
            offset = sign * mag
        else:
            offset = 15.0
    else:
        offset = float(offset_px)

    if width_fraction is None:
        if rng is not None:
            w_frac = float(rng.uniform(0.40, 1.00))
        else:
            w_frac = 0.70
    else:
        w_frac = float(width_fraction)

    n_cols = max(10, int(round(w_frac * w)))

    if u_start_fraction is None:
        if rng is not None:
            max_start = max(0, w - n_cols)
            u_start = int(rng.integers(0, max_start + 1))
        else:
            u_start = int(max(0, (w - n_cols) // 2))
    else:
        u_start = int(round(u_start_fraction * (w - n_cols)))

    u_end = min(w, u_start + n_cols)

    if polarity is None:
        if rng is not None:
            pol = "bright" if rng.random() > 0.5 else "dark"
        else:
            pol = "bright"
    else:
        pol = str(polarity).lower()

    # Horizon line approximation
    m, c = _fit_horizon_line(horizon_uv, h, w)

    # Inject sharp straight edge across u in [u_start, u_end]
    # For sub-pixel edge detector find_subpixel_edge_1d, a step discontinuity
    # creates a sharp gradient peak at v_edge.
    is_uint8 = (img.dtype == np.uint8)
    high_val = 255 if is_uint8 else 1.0
    low_val = 0 if is_uint8 else 0.0

    for u_idx in range(u_start, u_end):
        v_h = m * u_idx + c
        v_edge = int(round(v_h + offset))
        v_edge = max(2, min(h - 3, v_edge))

        # Create step transition across v_edge
        band = 25
        v_top = max(0, v_edge - band)
        v_bot = min(h, v_edge + band)

        if pol == "bright":
            # Bright edge: lower half is high intensity, upper half is low intensity
            if img.ndim == 2:
                img[v_top:v_edge, u_idx] = low_val
                img[v_edge:v_bot, u_idx] = high_val
            else:
                img[v_top:v_edge, u_idx, :] = low_val
                img[v_edge:v_bot, u_idx, :] = high_val
        else:
            # Dark edge: lower half is low intensity, upper half is high intensity
            if img.ndim == 2:
                img[v_top:v_edge, u_idx] = high_val
                img[v_edge:v_bot, u_idx] = low_val
            else:
                img[v_top:v_edge, u_idx, :] = high_val
                img[v_edge:v_bot, u_idx, :] = low_val

    log: dict[str, Any] = {
        "fault_type": "spurious_edge",
        "cam_id": cam_id,
        "offset_px": float(offset),
        "width_fraction": float(w_frac),
        "u_start": int(u_start),
        "u_end": int(u_end),
        "n_cols": int(u_end - u_start),
        "polarity": pol,
        "line_slope": float(m),
        "line_intercept": float(c),
        "sha256": compute_sha256(img),
    }
    return img, log


def inject_altitude_bias(
    h_true_m: float,
    bias_m: float = 200.0,
) -> tuple[float, dict[str, Any]]:
    """Inject altitude bias fault: h_meas = h_true + 200 m.

    Parameters
    ----------
    h_true_m : float
        True ellipsoidal altitude in meters.
    bias_m : float
        Injected bias in meters (default 200.0 m per WP07 spec).

    Returns
    -------
    tuple of (h_meas_m, fault_log)
    """
    h_meas = float(h_true_m + bias_m)
    log: dict[str, Any] = {
        "fault_type": "altitude_bias",
        "h_true_m": float(h_true_m),
        "bias_m": float(bias_m),
        "h_meas_m": h_meas,
    }
    return h_meas, log


def inject_glare(
    image: np.ndarray,
    horizon_uv: Sequence[Sequence[float]] | np.ndarray,
    radius_px: float = 150.0,
    u_center: float | None = None,
    v_offset: float | None = None,
    sat_value: int | float | None = None,
    rng: np.random.Generator | None = None,
    cam_id: str = "F",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inject glare fault: saturated disc of radius 150 px near horizon.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    horizon_uv : sequence of [u, v]
        Ground-truth horizon polyline.
    radius_px : float
        Radius of saturated disc in pixels (default 150.0 px per spec).
    u_center : float, optional
        Horizontal position of disc center.
    v_offset : float, optional
        Vertical offset of center from horizon in pixels.
    sat_value : int or float, optional
        Saturation pixel value.
    rng : np.random.Generator, optional
        Random generator for deterministic parameter selection.
    cam_id : str
        Camera identifier.

    Returns
    -------
    tuple of (modified_image, fault_log)
    """
    img = np.copy(image)
    h, w = img.shape[:2]
    r = float(radius_px)

    m, c = _fit_horizon_line(horizon_uv, h, w)

    if u_center is None:
        if rng is not None:
            u_c = float(rng.uniform(w * 0.25, w * 0.75))
        else:
            u_c = float(w / 2.0)
    else:
        u_c = float(u_center)

    v_h = m * u_c + c

    if v_offset is None:
        if rng is not None:
            # Near horizon: within -50 to +50 px
            v_off = float(rng.uniform(-50.0, 50.0))
        else:
            v_off = 0.0
    else:
        v_off = float(v_offset)

    v_c = float(v_h + v_off)

    if sat_value is None:
        if np.issubdtype(img.dtype, np.integer):
            sat = 255 if img.dtype == np.uint8 else 65535
        else:
            sat = 1.0
    else:
        sat = sat_value

    # Vectorized circular mask
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sq = (x_coords - u_c) ** 2 + (y_coords - v_c) ** 2
    mask = dist_sq <= (r ** 2)

    if img.ndim == 2:
        img[mask] = sat
    else:
        img[mask, :] = sat

    log: dict[str, Any] = {
        "fault_type": "glare",
        "cam_id": cam_id,
        "radius_px": r,
        "u_center": float(u_c),
        "v_center": float(v_c),
        "v_offset": float(v_off),
        "sat_value": float(sat),
        "pixel_count": int(np.sum(mask)),
        "sha256": compute_sha256(img),
    }
    return img, log


class SeededFaultInjector:
    """Deterministic, reproducible fault injector managing sequence faults."""

    def __init__(self, master_seed: int):
        self.master_seed = master_seed

    def get_rng(self, seq_name: str, fault_type: str, seed_idx: int) -> np.random.Generator:
        """Derive an independent deterministic RNG for a (seq, fault, seed) combination."""
        key = f"{self.master_seed}|{seq_name}|{fault_type}|{seed_idx}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="big")
        return np.random.Generator(np.random.PCG64(seed_int))

    def choose_onset_frame(
        self,
        n_frames: int,
        seq_name: str,
        fault_type: str,
        seed_idx: int,
    ) -> int:
        """Deterministically choose a random onset frame in [10, n_frames - 20]."""
        rng = self.get_rng(seq_name, f"onset_{fault_type}", seed_idx)
        min_onset = min(10, max(1, n_frames // 4))
        max_onset = max(min_onset + 1, int(n_frames * 0.75))
        return int(rng.integers(min_onset, max_onset))

    def choose_target_camera(
        self,
        seq_name: str,
        fault_type: str,
        seed_idx: int,
        candidates: Sequence[str] = ("F", "R", "B", "L"),
    ) -> str:
        """Deterministically choose one horizontal camera for the fault."""
        rng = self.get_rng(seq_name, f"cam_{fault_type}", seed_idx)
        idx = int(rng.integers(0, len(candidates)))
        return candidates[idx]


def __getattr__(name: str) -> Any:
    if name in ("inject_single_cam_misalignment", "perturb_rig_boresight"):
        from xvs.integrity import g1_extensions
        return getattr(g1_extensions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

