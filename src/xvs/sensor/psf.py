"""Optics and Point Spread Function (PSF) / MTF modeling.

Spec reference: docs/xvs/WP03_SENSOR.md §1 item 2 and §3 item V2.
- Continuous PSF: Airy diffraction convolved with isotropic Gaussian (sigma_px).
- Discretized by 8x supersampling, integrated over square pixel aperture (fill factor 1).
- Truncated energy <= 1e-4 recorded, normalized to sum 1.
"""

from __future__ import annotations

import numpy as np
import scipy.signal


def mtf_diffraction(rho: np.ndarray | float, nu_c: float) -> np.ndarray:
    """Circular aperture diffraction MTF.

    Cutoff frequency nu_c = p / (lambda * N) in cycles/pixel.

    Parameters
    ----------
    rho : array_like
        Spatial frequency magnitude in cycles/pixel.
    nu_c : float
        Cutoff spatial frequency in cycles/pixel.

    Returns
    -------
    np.ndarray
        Diffraction MTF values in [0, 1].
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    s = np.clip(rho_arr / nu_c, 0.0, 1.0)
    # Analytical autocorrelation of a circular pupil:
    # MTF = (2 / pi) * (arccos(s) - s * sqrt(1 - s^2)) for s <= 1, 0 otherwise
    val = (2.0 / np.pi) * (np.arccos(s) - s * np.sqrt(np.maximum(0.0, 1.0 - s * s)))
    return np.where(rho_arr < nu_c, val, 0.0)


def mtf_gaussian(rho: np.ndarray | float, sigma_px: float) -> np.ndarray:
    """Fourier transform of isotropic Gaussian blur of standard deviation sigma_px.

    H_gauss(rho) = exp(-2 * pi^2 * sigma_px^2 * rho^2).
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    return np.exp(-2.0 * (np.pi ** 2) * (sigma_px ** 2) * (rho_arr ** 2))


def sinc_2d(fx: np.ndarray | float, fy: np.ndarray | float) -> np.ndarray:
    """Fourier transform of square pixel aperture (1 px x 1 px): sinc(fx) * sinc(fy).

    Note: np.sinc(x) is sin(pi*x)/(pi*x).
    """
    return np.sinc(fx) * np.sinc(fy)


def analytic_transfer_function(
    nu_x: np.ndarray | float,
    nu_y: np.ndarray | float,
    f_number: float,
    wavelength_m: float,
    pixel_pitch_m: float,
    sigma_px: float,
) -> np.ndarray:
    """Analytic continuous transfer function H(nu_x, nu_y) of optics + pixel aperture.

    H = MTF_diffraction * MTF_gauss * sinc(nu_x) * sinc(nu_y).
    """
    fx = np.asarray(nu_x, dtype=np.float64)
    fy = np.asarray(nu_y, dtype=np.float64)
    rho = np.sqrt(fx * fx + fy * fy)
    nu_c = pixel_pitch_m / (wavelength_m * f_number)

    h_diff = mtf_diffraction(rho, nu_c)
    h_gauss = mtf_gaussian(rho, sigma_px)
    h_pix = sinc_2d(fx, fy)
    return h_diff * h_gauss * h_pix


def aliased_analytic_transfer_function(
    nu_x: np.ndarray,
    nu_y: np.ndarray,
    f_number: float,
    wavelength_m: float,
    pixel_pitch_m: float,
    sigma_px: float,
    m_range: tuple[int, int] = (-3, 3),
    n_range: tuple[int, int] = (-3, 3),
) -> np.ndarray:
    """Aliased analytic transfer function on a discrete grid: sum_{m, n} H(nu_x + m, nu_y + n).

    Spec V2(b): m, n = -3..3.
    """
    fx = np.asarray(nu_x, dtype=np.float64)
    fy = np.asarray(nu_y, dtype=np.float64)
    nu_c = pixel_pitch_m / (wavelength_m * f_number)

    h_aliased = np.zeros(fx.shape, dtype=np.float64)
    for m in range(m_range[0], m_range[1] + 1):
        f_x_m = fx + m
        for n in range(n_range[0], n_range[1] + 1):
            f_y_n = fy + n
            rho = np.sqrt(f_x_m * f_x_m + f_y_n * f_y_n)
            h_diff = mtf_diffraction(rho, nu_c)
            h_gauss = mtf_gaussian(rho, sigma_px)
            h_pix = sinc_2d(f_x_m, f_y_n)
            h_aliased += h_diff * h_gauss * h_pix
    return h_aliased


def compute_psf_kernel(
    f_number: float,
    wavelength_m: float,
    pixel_pitch_m: float,
    sigma_px: float,
    energy_truncation_threshold: float = 1e-4,
    supersampling: int = 8,
    domain_size_px: int = 256,
    max_radius: int = 40,
) -> tuple[np.ndarray, int, float]:
    """Compute discrete PSF kernel via 8x supersampling and square pixel aperture integration.

    Parameters
    ----------
    f_number : float
        Aperture f-number N.
    wavelength_m : float
        Wavelength lambda in metres.
    pixel_pitch_m : float
        Pixel pitch p in metres.
    sigma_px : float
        Isotropic Gaussian blur standard deviation in pixels.
    energy_truncation_threshold : float
        Target maximum truncated energy (e.g. 1e-4).
    supersampling : int
        Subpixel sampling factor (8x).
    domain_size_px : int
        FFT domain width in pixels.
    max_radius : int
        Upper bound on kernel radius in pixels.

    Returns
    -------
    kernel : np.ndarray
        2D discrete kernel of shape (2*r + 1, 2*r + 1), normalized to sum 1 within 1e-12.
    radius : int
        Recorded kernel radius in pixels.
    truncated_energy : float
        Truncated energy outside the kernel before normalization.
    """
    n_tot = domain_size_px * supersampling
    dx = 1.0 / supersampling
    fx_1d = np.fft.fftfreq(n_tot, d=dx)
    fx_grid, fy_grid = np.meshgrid(fx_1d, fx_1d, indexing="xy")
    rho = np.sqrt(fx_grid * fx_grid + fy_grid * fy_grid)

    nu_c = pixel_pitch_m / (wavelength_m * f_number)
    h_opt = mtf_diffraction(rho, nu_c) * mtf_gaussian(rho, sigma_px) * np.sinc(fx_grid) * np.sinc(fy_grid)

    # IFFT to obtain the continuous pixel-integrated PSF sampled on the subpixel grid
    psf_cont = np.fft.fftshift(np.fft.ifft2(h_opt).real) / (dx * dx)
    center = n_tot // 2

    # Search for radius where truncated energy <= threshold
    chosen_radius = max_radius
    truncated_energy = 0.0

    # Test radii from 1 to max_radius
    for r in range(1, max_radius + 1):
        patch = psf_cont[
            center - r * supersampling : center + r * supersampling + 1 : supersampling,
            center - r * supersampling : center + r * supersampling + 1 : supersampling,
        ]
        e_inside = float(np.sum(patch))
        t_energy = 1.0 - e_inside
        if t_energy <= energy_truncation_threshold:
            chosen_radius = r
            truncated_energy = t_energy
            break
    else:
        # If threshold not met by max_radius, use max_radius and record truncated energy
        patch = psf_cont[
            center - max_radius * supersampling : center + max_radius * supersampling + 1 : supersampling,
            center - max_radius * supersampling : center + max_radius * supersampling + 1 : supersampling,
        ]
        truncated_energy = 1.0 - float(np.sum(patch))
        chosen_radius = max_radius

    # Normalize to exact sum 1
    raw_kernel = psf_cont[
        center - chosen_radius * supersampling : center + chosen_radius * supersampling + 1 : supersampling,
        center - chosen_radius * supersampling : center + chosen_radius * supersampling + 1 : supersampling,
    ]
    kernel = raw_kernel / np.sum(raw_kernel)

    return kernel, chosen_radius, truncated_energy


def compute_dtft(kernel: np.ndarray, nu_x: np.ndarray, nu_y: np.ndarray) -> np.ndarray:
    """Compute Discrete-Time Fourier Transform (DTFT) of a 2D discrete kernel.

    H(nu_x, nu_y) = sum_{u, v} kernel[v, u] * exp(-2*pi*i*(nu_x*u + nu_y*v)).
    """
    kh, kw = kernel.shape
    r_v = (kh - 1) // 2
    r_u = (kw - 1) // 2
    u_idx = np.arange(-r_u, r_u + 1, dtype=np.float64)
    v_idx = np.arange(-r_v, r_v + 1, dtype=np.float64)

    # Vectorized computation using complex phases
    phase_u = np.exp(-2j * np.pi * nu_x[..., None] * u_idx)  # shape (..., kw)
    phase_v = np.exp(-2j * np.pi * nu_y[..., None] * v_idx)  # shape (..., kh)

    dtft = np.einsum("vu,...u,...v->...", kernel, phase_u, phase_v)
    return dtft


def apply_psf(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve 2D (or multichannel) image with discrete PSF kernel.

    Uses fftconvolve for O(N log N) speed.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        return scipy.signal.fftconvolve(img, kernel, mode="same")
    elif img.ndim == 3:
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[..., c] = scipy.signal.fftconvolve(img[..., c], kernel, mode="same")
        return out
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")


def measure_slanted_edge_mtf(
    edge_image: np.ndarray,
    angle_rad: float,
    center_pos: tuple[float, float],
    max_freq: float = 0.5,
    num_freqs: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure ISO 12233-style simplified slanted edge MTF on a synthetic edge image.

    Parameters
    ----------
    edge_image : np.ndarray
        2D image containing a slanted edge.
    angle_rad : float
        Edge orientation angle in radians (relative to vertical).
    center_pos : tuple[float, float]
        (u0, v0) reference point on edge line.
    max_freq : float
        Maximum spatial frequency to evaluate.
    num_freqs : int
        Number of frequency points.

    Returns
    -------
    freqs : np.ndarray
        Spatial frequencies in cycles/pixel in [0, max_freq].
    mtf : np.ndarray
        Measured normalized MTF values.
    """
    h, w = edge_image.shape
    u0, v0 = center_pos
    normal_x = np.cos(angle_rad)
    normal_y = np.sin(angle_rad)

    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    dist = (u_grid - u0) * normal_x + (v_grid - v0) * normal_y

    # Bin into oversampled 1D Edge Spread Function (ESF)
    bin_width = 0.125  # 8x subpixel bins
    bin_edges = np.arange(-15.0, 15.0 + bin_width, bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_indices = np.digitize(dist.ravel(), bin_edges) - 1
    valid = (bin_indices >= 0) & (bin_indices < len(bin_centers))

    esf = np.zeros(len(bin_centers), dtype=np.float64)
    counts = np.bincount(bin_indices[valid], minlength=len(bin_centers))
    sums = np.bincount(bin_indices[valid], weights=edge_image.ravel()[valid], minlength=len(bin_centers))

    non_empty = counts > 0
    esf[non_empty] = sums[non_empty] / counts[non_empty]

    # Fill any empty bins with linear interpolation
    empty = ~non_empty
    if np.any(empty) and np.any(non_empty):
        esf[empty] = np.interp(bin_centers[empty], bin_centers[non_empty], esf[non_empty])

    # Line Spread Function (LSF) is derivative of ESF
    lsf = np.diff(esf) / bin_width
    lsf_centers = 0.5 * (bin_centers[:-1] + bin_centers[1:])

    # Window LSF with Hann window to reduce edge truncation noise
    hann = np.hanning(len(lsf))
    lsf_win = lsf * hann

    # Compute continuous FT of LSF at frequencies freqs
    freqs = np.linspace(0.0, max_freq, num_freqs)
    # LSF FT: sum lsf_win * exp(-2pi i nu x) dx
    phases = np.exp(-2j * np.pi * freqs[:, None] * lsf_centers[None, :])
    mtf_raw = np.abs(np.sum(lsf_win[None, :] * phases, axis=1))

    if mtf_raw[0] > 0:
        mtf = mtf_raw / mtf_raw[0]
    else:
        mtf = mtf_raw

    return freqs, mtf
