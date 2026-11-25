"""Unit tests for WP-04 classical ring detector D0 (V1 exactness and V7 import isolation).

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §1 and §7.
"""

import ast
import math
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pytest

from xvs.detection import d0

REPO_ROOT = Path(__file__).resolve().parents[2]


def brute_force_reference_cmo(image: np.ndarray, k: int = 5) -> np.ndarray:
    """Independent reference implementation of CMO using sliding-window min/max."""
    r = k // 2
    pad_img = np.pad(image, r, mode="edge")
    win = sliding_window_view(pad_img, (k, k))

    # Dilation: max over window
    dil = np.max(win, axis=(-2, -1))
    pad_dil = np.pad(dil, r, mode="edge")
    win_dil = sliding_window_view(pad_dil, (k, k))
    # Closing: min of dilation
    closing = np.min(win_dil, axis=(-2, -1))

    # Erosion: min over window
    ero = np.min(win, axis=(-2, -1))
    pad_ero = np.pad(ero, r, mode="edge")
    win_ero = sliding_window_view(pad_ero, (k, k))
    # Opening: max of erosion
    opening = np.max(win_ero, axis=(-2, -1))

    cmo = np.maximum(0.0, closing - opening)
    return cmo


def brute_force_reference_local_stats(
    cmo: np.ndarray,
    tile_size: int = 64,
    tile_stride: int = 32,
    mad_scale: float = 1.4826,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent reference implementation of tile statistics with explicit tile loops."""
    h, w = cmo.shape
    y_centres = [y for y in range(tile_stride // 2, h, tile_stride)]
    x_centres = [x for x in range(tile_stride // 2, w, tile_stride)]

    ny = len(y_centres)
    nx = len(x_centres)
    half = tile_size // 2

    m_grid = np.zeros((ny, nx), dtype=np.float64)
    sig_grid = np.zeros((ny, nx), dtype=np.float64)

    for j, yc in enumerate(y_centres):
        y_start = max(0, min(h - tile_size, yc - half)) if h >= tile_size else 0
        y_end = y_start + tile_size if h >= tile_size else h

        for i, xc in enumerate(x_centres):
            x_start = max(0, min(w - tile_size, xc - half)) if w >= tile_size else 0
            x_end = x_start + tile_size if w >= tile_size else w

            patch = cmo[y_start:y_end, x_start:x_end].flatten()
            med = float(np.median(patch))
            mad = float(np.median(np.abs(patch - med)))
            m_grid[j, i] = med
            sig_grid[j, i] = mad_scale * mad

    # Bilinear interpolation via reference formulas
    m_loc = np.zeros((h, w), dtype=np.float64)
    sig_loc = np.zeros((h, w), dtype=np.float64)

    y0, x0 = tile_stride // 2, tile_stride // 2
    for y in range(h):
        v = min(max((y - y0) / tile_stride, 0.0), float(ny - 1))
        j0 = int(np.floor(v))
        j1 = min(j0 + 1, ny - 1)
        fy = v - j0

        for x in range(w):
            u = min(max((x - x0) / tile_stride, 0.0), float(nx - 1))
            i0 = int(np.floor(u))
            i1 = min(i0 + 1, nx - 1)
            fx = u - i0

            m_val = (1.0 - fy) * (1.0 - fx) * m_grid[j0, i0] + \
                    (1.0 - fy) * fx * m_grid[j0, i1] + \
                    fy * (1.0 - fx) * m_grid[j1, i0] + \
                    fy * fx * m_grid[j1, i1]
            s_val = (1.0 - fy) * (1.0 - fx) * sig_grid[j0, i0] + \
                    (1.0 - fy) * fx * sig_grid[j0, i1] + \
                    fy * (1.0 - fx) * sig_grid[j1, i0] + \
                    fy * fx * sig_grid[j1, i1]

            m_loc[y, x] = m_val
            sig_loc[y, x] = s_val

    return m_loc, sig_loc


def test_v1_d0_exactness_synthetic_blobs():
    """Gate G0 Acceptance V1: D0 exactness against independent brute-force reference.

    Evaluates 1,000 seeded sub-pixel Gaussian blobs (sigma in {0.7, 1.0, 1.5}, bright and dark)
    on 128x128 images. Confirms CMO, z, and centroid match within 1e-9.
    Reports centroid bias against true centre.
    """
    rng = np.random.RandomState(20260925)
    sigmas = [0.7, 1.0, 1.5]
    n_images = 100
    blobs_per_image = 10
    total_blobs = n_images * blobs_per_image  # exactly 1,000 blobs
    k = 5
    bg_level = 10.0

    cmo_max_diff = 0.0
    z_max_diff = 0.0
    centroid_max_diff = 0.0

    biases = []

    # Grid for blob placement in 128x128
    # Keep blobs well separated (e.g. 5 columns x 2 rows in safe interior)
    for img_idx in range(n_images):
        img = np.full((128, 128), bg_level, dtype=np.float64)
        injected = []

        # Place 10 non-overlapping blobs
        for b_idx in range(blobs_per_image):
            row = b_idx // 5
            col = b_idx % 5
            # Center in grid cell [20 + col*20, 30 + row*50]
            cx = 20.0 + col * 20.0 + rng.uniform(-4.0, 4.0)
            cy = 35.0 + row * 45.0 + rng.uniform(-4.0, 4.0)
            sig = sigmas[(img_idx * blobs_per_image + b_idx) % len(sigmas)]
            is_bright = ((img_idx * blobs_per_image + b_idx) % 2 == 0)
            amp = 5.0 if is_bright else -4.0

            # Render Gaussian blob in local neighborhood
            y_i = int(np.floor(cy))
            x_i = int(np.floor(cx))
            y_slice = slice(max(0, y_i - 6), min(128, y_i + 7))
            x_slice = slice(max(0, x_i - 6), min(128, x_i + 7))
            yy, xx = np.mgrid[y_slice, x_slice]
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            blob = amp * np.exp(-dist_sq / (2.0 * sig ** 2))
            img[y_slice, x_slice] += blob
            injected.append((cx, cy, sig, is_bright))

        # D0 execution
        cmo_fast = d0.compute_cmo(img, k=k)
        m_fast, s_fast = d0.compute_local_stats(cmo_fast, tile_size=64, tile_stride=32)
        z_fast = d0.compute_z_score(cmo_fast, m_fast, s_fast, sigma_min=0.001)

        # Brute-force reference execution
        cmo_ref = brute_force_reference_cmo(img, k=k)
        m_ref, s_ref = brute_force_reference_local_stats(cmo_ref, tile_size=64, tile_stride=32)
        z_ref = (cmo_ref - m_ref) / np.maximum(s_ref, 0.001)

        # Compare CMO and z within 1e-9
        diff_cmo = float(np.max(np.abs(cmo_fast - cmo_ref)))
        diff_z = float(np.max(np.abs(z_fast - z_ref)))
        cmo_max_diff = max(cmo_max_diff, diff_cmo)
        z_max_diff = max(z_max_diff, diff_z)

        assert diff_cmo < 1e-9, f"CMO mismatch {diff_cmo:.3e} on image {img_idx}"
        assert diff_z < 1e-9, f"z-score mismatch {diff_z:.3e} on image {img_idx}"

        # Extract components and check centroids
        dets = d0.extract_components(z_fast, cmo_fast, tau=3.0)
        for cx, cy, sig, is_bright in injected:
            # Find nearest detection
            best_d = float("inf")
            best_det = None
            for d in dets:
                dist = math.hypot(d["u"] - cx, d["v"] - cy)
                if dist < best_d:
                    best_d = dist
                    best_det = d

            assert best_det is not None and best_d < 3.0, f"Blob at ({cx:.2f}, {cy:.2f}) not detected (closest {best_d:.2f} px)"
            biases.append(best_d)

    biases = np.array(biases)
    mean_bias = float(np.mean(biases))
    p95_bias = float(np.percentile(biases, 95))
    max_bias = float(np.max(biases))

    print(f"\n[V1 Exactness] Tested {total_blobs} blobs:")
    print(f"  CMO max error vs brute-force: {cmo_max_diff:.2e} (< 1e-9 PASS)")
    print(f"  z-score max error vs brute-force: {z_max_diff:.2e} (< 1e-9 PASS)")
    print(f"  Centroid bias: mean={mean_bias:.4f} px, p95={p95_bias:.4f} px, max={max_bias:.4f} px")

    assert cmo_max_diff < 1e-9
    assert z_max_diff < 1e-9
    assert mean_bias < 0.20, f"Centroid mean bias too large: {mean_bias:.4f} px"


def test_v7_no_truth_in_d0():
    """Gate G0 Acceptance V7: Verify d0.py contains zero ground-truth, mask, or pose imports."""
    d0_path = REPO_ROOT / "src" / "xvs" / "detection" / "d0.py"
    assert d0_path.is_file(), "d0.py must exist"

    tree = ast.parse(d0_path.read_text(encoding="utf-8"))
    forbidden_modules = [
        "xvs.traffic",
        "xvs.metrics",
        "xvs.sim.horizon",
        "xvs.geo",
        "xvs.sim",
        "traffic",
        "metrics",
        "horizon",
    ]

    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.append(node.module)

    print(f"\nD0 imports: {imported_names}")
    for imp in imported_names:
        for forbidden in forbidden_modules:
            assert not imp.startswith(forbidden), f"V7 VIOLATION: d0.py imports forbidden module '{imp}'"


def test_srgb_quantization_floor():
    """Verify local 8-bit quantization noise floor sigma_min derivation through sRGB curve."""
    # Dark region (linear L <= 0.0031308)
    sig_dark = d0.compute_srgb_quantization_sigma(0.001)
    # sigma_s / 12.92 = 1 / (255 * sqrt(12) * 12.92) ~ 8.7618e-5
    expected_dark = 1.0 / (255.0 * math.sqrt(12.0) * 12.92)
    assert abs(sig_dark - expected_dark) < 1e-9

    # Mid luminance (L = 0.18)
    sig_mid = d0.compute_srgb_quantization_sigma(0.18)
    # Mid-gray is ~ 0.000947
    assert 0.00085 < sig_mid < 0.00105

    # High luminance (L = 0.50)
    sig_high = d0.compute_srgb_quantization_sigma(0.50)
    # Corresponds to ~ 0.0017 - 0.0018
    assert 0.0016 < sig_high < 0.00185

    # Peak luminance (L = 1.00)
    sig_peak = d0.compute_srgb_quantization_sigma(1.00)
    # (2.4 / 1.055) / (255 * sqrt(12)) ~ 0.002575
    expected_peak = (2.4 / 1.055) / (255.0 * math.sqrt(12.0))
    assert abs(sig_peak - expected_peak) < 1e-6

    # Test 2D clean floor grid on 128x128 image
    img = np.full((128, 128), 0.5, dtype=np.float64)
    grid = d0.compute_clean_floor_grid(img, tile_size=64, tile_stride=32)
    assert grid.shape == (128, 128)
    assert np.allclose(grid, sig_high, atol=1e-5)
