"""sRGB transfer functions (IEC 61966-2-1) and linear Rec. 709 monochrome weighting.

Spec reference: docs/xvs/WP03_SENSOR.md §1 item 1 and §3 item V1.
"""

from __future__ import annotations

import numpy as np


SRGB_LINEAR_THRESHOLD: float = 0.04045 / 12.92
REC709_R_WEIGHT: float = 0.2126
REC709_G_WEIGHT: float = 0.7152
REC709_B_WEIGHT: float = 0.0722


def srgb_eotf(c: np.ndarray | float | int) -> np.ndarray:
    """Electro-Optical Transfer Function (EOTF) converting sRGB to linear radiance.

    Standard: IEC 61966-2-1.
    If input is integer/uint8 in [0, 255], it is first normalized to [0, 1].

    Parameters
    ----------
    c : array_like
        Encoded sRGB values in [0, 1] (or uint8 in [0, 255]).

    Returns
    -------
    np.ndarray
        Linear relative radiance in [0, 1], dtype float64.
    """
    arr = np.asarray(c)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = arr.astype(np.float64)

    arr = np.maximum(arr, 0.0)

    linear = np.where(
        arr <= 0.04045,
        arr / 12.92,
        ((arr + 0.055) / 1.055) ** 2.4,
    )
    return linear


def srgb_oetf(linear: np.ndarray | float) -> np.ndarray:
    """Opto-Electronic Transfer Function (OETF) converting linear radiance to sRGB.

    Standard: IEC 61966-2-1.
    Exact mathematical inverse of srgb_eotf in float64.

    Parameters
    ----------
    linear : array_like
        Linear relative radiance values in [0, 1].

    Returns
    -------
    np.ndarray
        Encoded sRGB values in [0, 1], dtype float64.
    """
    arr = np.asarray(linear, dtype=np.float64)

    srgb = np.where(
        arr <= SRGB_LINEAR_THRESHOLD,
        arr * 12.92,
        1.055 * (np.maximum(arr, 0.0) ** (1.0 / 2.4)) - 0.055,
    )
    return srgb


def srgb_to_8bit(srgb: np.ndarray | float) -> np.ndarray:
    """Convert normalized sRGB in [0, 1] to 8-bit integer codes in [0, 255]."""
    arr = np.asarray(srgb, dtype=np.float64)
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def linear_rgb_to_monochrome(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB radiance to linear monochrome radiance using Rec. 709 weights.

    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B.

    Parameters
    ----------
    linear_rgb : np.ndarray
        Array with last dimension 3 corresponding to (R, G, B).

    Returns
    -------
    np.ndarray
        Monochrome linear radiance Y with last dimension reduced.
    """
    arr = np.asarray(linear_rgb, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dimension 3 for RGB, got shape {arr.shape}")
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    return REC709_R_WEIGHT * r + REC709_G_WEIGHT * g + REC709_B_WEIGHT * b
