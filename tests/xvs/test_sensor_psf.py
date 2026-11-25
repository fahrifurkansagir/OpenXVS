"""Tests for PSF kernel generation, energy conservation and MTF DTFT match (V2).

Spec reference: docs/xvs/WP03_SENSOR.md §3 item V2.
- (a) The kernel sums to 1 within 1e-12.
- (b) Its discrete-time Fourier transform, sampled on a 64x64 frequency grid,
  matches the aliased analytic transfer function within 0.005 absolute.
"""

from __future__ import annotations

import numpy as np
import pytest

from xvs.sensor.psf import (
    compute_psf_kernel,
    compute_dtft,
    aliased_analytic_transfer_function,
    apply_psf,
)


@pytest.mark.parametrize(
    "preset_name, N, wavelength_m, pixel_pitch_m, sigma_px",
    [
        ("ring_mono_v0", 2.8, 550e-9, 3.45e-6, 0.30),
        ("rig_mono_v0", 2.0, 550e-9, 3.45e-6, 0.40),
    ],
)
def test_v2a_kernel_sum_to_one(preset_name, N, wavelength_m, pixel_pitch_m, sigma_px):
    """V2(a): The kernel sums to 1 within 1e-12."""
    kernel, radius, trunc_energy = compute_psf_kernel(
        f_number=N,
        wavelength_m=wavelength_m,
        pixel_pitch_m=pixel_pitch_m,
        sigma_px=sigma_px,
        max_radius=25,
    )
    sum_err = abs(float(np.sum(kernel)) - 1.0)
    assert sum_err <= 1e-12, f"[{preset_name}] Kernel sum error {sum_err} exceeds 1e-12"
    assert radius > 0


@pytest.mark.parametrize(
    "preset_name, N, wavelength_m, pixel_pitch_m, sigma_px",
    [
        ("ring_mono_v0", 2.8, 550e-9, 3.45e-6, 0.30),
        ("rig_mono_v0", 2.0, 550e-9, 3.45e-6, 0.40),
    ],
)
def test_v2b_dtft_matches_aliased_analytic_h(preset_name, N, wavelength_m, pixel_pitch_m, sigma_px):
    """V2(b): DTFT on 64x64 frequency grid matches aliased analytic transfer function within 0.005."""
    kernel, radius, _ = compute_psf_kernel(
        f_number=N,
        wavelength_m=wavelength_m,
        pixel_pitch_m=pixel_pitch_m,
        sigma_px=sigma_px,
        max_radius=25,
    )

    # 64x64 frequency grid in [-0.5, 0.5) cycles/pixel
    nu_grid = np.linspace(-0.5, 0.5, 64, endpoint=False)
    nu_x, nu_y = np.meshgrid(nu_grid, nu_grid, indexing="xy")

    # Analytic aliased H
    h_aliased = aliased_analytic_transfer_function(
        nu_x=nu_x,
        nu_y=nu_y,
        f_number=N,
        wavelength_m=wavelength_m,
        pixel_pitch_m=pixel_pitch_m,
        sigma_px=sigma_px,
        m_range=(-3, 3),
        n_range=(-3, 3),
    )

    # Discrete-time Fourier transform of the kernel
    dtft_h = compute_dtft(kernel, nu_x, nu_y)

    abs_err = np.abs(dtft_h - h_aliased)
    max_err = float(np.max(abs_err))
    assert max_err <= 0.005, f"[{preset_name}] Max DTFT error {max_err:.6f} exceeds tolerance 0.005"


def test_apply_psf_conservation_of_energy():
    """Verify that applying the PSF kernel conserves total image energy."""
    kernel, _, _ = compute_psf_kernel(2.8, 550e-9, 3.45e-6, 0.30, max_radius=10)
    img = np.zeros((128, 128), dtype=np.float64)
    img[40:80, 40:80] = 1.0
    initial_energy = np.sum(img)

    blurred = apply_psf(img, kernel)
    # Energy in central region where boundary effects do not reach
    center_energy_initial = np.sum(img[50:70, 50:70])
    center_energy_blurred = np.sum(blurred[50:70, 50:70])
    np.testing.assert_allclose(center_energy_initial, center_energy_blurred, rtol=1e-3)
    np.testing.assert_allclose(np.sum(blurred), initial_energy, rtol=1e-5)


def test_apply_psf_multichannel():
    """Verify applying PSF to a 3D (multi-channel) image."""
    kernel, _, _ = compute_psf_kernel(2.8, 550e-9, 3.45e-6, 0.30, max_radius=10)
    img_3d = np.ones((64, 64, 3), dtype=np.float64)
    blurred_3d = apply_psf(img_3d, kernel)
    assert blurred_3d.shape == (64, 64, 3)
    # Center should remain 1.0 since kernel sums to 1.0
    np.testing.assert_allclose(blurred_3d[32, 32, :], [1.0, 1.0, 1.0], atol=1e-5)

    # Invalid dimension check
    with pytest.raises(ValueError, match="Expected 2D or 3D image"):
        apply_psf(np.ones((10, 10, 2, 2)), kernel)
