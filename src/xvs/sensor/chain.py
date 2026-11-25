"""Full sensor and image-formation pipeline (WP-03 G0).

Spec reference: docs/xvs/WP03_SENSOR.md §1 (items 1-6) and §3 item V6.
Order of execution:
1. Linearise (sRGB EOTF + Rec. 709 monochrome).
2. Optics (discrete PSF convolution).
3. Geometric distortion (bicubic backward warping).
4. Vignetting (cos^4 theta * radial polynomial).
5. Sensor (EMVA 1288 linear model: shot noise, dark current, read noise, FPN, ADC).
6. Outputs (16-bit raw PNG, 8-bit display PNG).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from xvs.sensor.srgb import srgb_eotf, srgb_oetf, srgb_to_8bit, linear_rgb_to_monochrome
from xvs.sensor.psf import compute_psf_kernel, apply_psf
from xvs.sensor.distortion import compute_distortion_backward_map, warp_image_distort
from xvs.sensor.vignetting import compute_vignetting_map, apply_vignetting
from xvs.sensor.noise import (
    compute_seed_temporal,
    generate_fpn_maps,
    compute_auto_exposure,
    apply_sensor_noise,
)


class SensorConfig:
    """Sensor and optics configuration loaded from schema xvs.sensor/1 JSON."""

    def __init__(self, data: dict[str, Any]):
        self.raw_data = data
        self.schema = data.get("schema", "xvs.sensor/1")
        self.name = data.get("name", "unnamed_sensor")
        self.description = data.get("description", "")
        params = data.get("parameters", {})

        def get_val(key: str, default: Any = None) -> Any:
            p = params.get(key)
            if p is None:
                return default
            if isinstance(p, dict) and "value" in p:
                return p["value"]
            return p

        self.image_width: int = int(get_val("image_width", 1024))
        self.image_height: int = int(get_val("image_height", 1024))
        self.fx: float = float(get_val("fx", 1000.0))
        self.fy: float = float(get_val("fy", 1000.0))
        self.cx: float = float(get_val("cx", 511.5))
        self.cy: float = float(get_val("cy", 511.5))
        self.pixel_pitch_m: float = float(get_val("pixel_pitch_m", 3.45e-6))
        self.f_number: float = float(get_val("f_number", 2.8))
        self.wavelength_m: float = float(get_val("wavelength_m", 550e-9))
        self.sigma_px: float = float(get_val("sigma_px", 0.30))
        self.k1: float = float(get_val("k1", 0.0))
        self.k2: float = float(get_val("k2", 0.0))
        self.k3: float = float(get_val("k3", 0.0))
        self.p1: float = float(get_val("p1", 0.0))
        self.p2: float = float(get_val("p2", 0.0))
        self.v1: float = float(get_val("v1", 0.0))
        self.v2: float = float(get_val("v2", 0.0))
        self.full_well_e: float = float(get_val("full_well_e", 10500.0))
        self.sigma_r_e: float = float(get_val("sigma_r_e", 2.4))
        self.dark_current_e_per_s: float = float(get_val("dark_current_e_per_s", 5.0))
        self.exposure_time_s: float = float(get_val("exposure_time_s", 0.001))
        self.prnu_rms: float = float(get_val("prnu_rms", 0.005))
        self.dsnu_rms_e: float = float(get_val("dsnu_rms_e", 1.0))
        self.k_dn_per_e: float = float(get_val("k_dn_per_e", 0.35))
        self.bits: int = int(get_val("bits", 12))
        self.black_level_dn: float = float(get_val("black_level_dn", 64.0))
        self.ae_target: float = float(get_val("ae_target", 0.6))

        # Lazy cached precomputations
        self._psf_kernel: np.ndarray | None = None
        self._psf_radius: int = 0
        self._psf_truncated_energy: float = 0.0
        self._backward_map: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._vignetting_map: np.ndarray | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> SensorConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def get_psf_kernel(self, max_radius: int = 25) -> tuple[np.ndarray, int, float]:
        if self._psf_kernel is None:
            self._psf_kernel, self._psf_radius, self._psf_truncated_energy = compute_psf_kernel(
                f_number=self.f_number,
                wavelength_m=self.wavelength_m,
                pixel_pitch_m=self.pixel_pitch_m,
                sigma_px=self.sigma_px,
                max_radius=max_radius,
            )
        return self._psf_kernel, self._psf_radius, self._psf_truncated_energy

    def get_backward_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._backward_map is None:
            self._backward_map = compute_distortion_backward_map(
                width=self.image_width,
                height=self.image_height,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                k1=self.k1,
                k2=self.k2,
                k3=self.k3,
                p1=self.p1,
                p2=self.p2,
            )
        return self._backward_map

    def get_vignetting_map(self) -> np.ndarray:
        if self._vignetting_map is None:
            self._vignetting_map = compute_vignetting_map(
                width=self.image_width,
                height=self.image_height,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
                v1=self.v1,
                v2=self.v2,
            )
        return self._vignetting_map


class CameraSimulator:
    """End-to-end camera simulator for a specific sensor instance."""

    def __init__(self, config: SensorConfig, sensor_id: str):
        self.config = config
        self.sensor_id = sensor_id

        # Precompute and cache fixed-pattern noise maps
        self.prnu_map, self.dsnu_map = generate_fpn_maps(
            sensor_id=sensor_id,
            height=config.image_height,
            width=config.image_width,
            prnu_rms=config.prnu_rms,
            dsnu_rms=config.dsnu_rms_e,
        )

    def process_frame(
        self,
        image_input: np.ndarray,
        image_id: str,
        frame_idx: int,
        auto_exposure: bool = False,
        E_override: float | None = None,
        warp_method: str = "cv2",
    ) -> dict[str, Any]:
        """Run the full G0 imaging chain on a single input frame.

        Parameters
        ----------
        image_input : np.ndarray
            Input 8-bit sRGB render (RGB or mono) or linear float.
        image_id : str
            Identifier of the image sequence / scene.
        frame_idx : int
            Frame index in sequence.
        auto_exposure : bool
            Whether to adaptively set photo-electron scaling E.
        E_override : float, optional
            Directly set photo-electrons at L=1 (defaults to full_well_e).
        warp_method : str
            'cv2' or 'scipy'.

        Returns
        -------
        dict
            {"raw": uint16 array, "display": uint8 array, "validity_mask": bool array, "E": float}
        """
        arr = np.asarray(image_input)
        if arr.shape[0] != self.config.image_height or arr.shape[1] != self.config.image_width:
            raise ValueError(
                f"Input image dimensions {arr.shape[:2]} do not match sensor config "
                f"({self.config.image_height}, {self.config.image_width})"
            )

        # 1. Linearise
        if np.issubdtype(arr.dtype, np.integer):
            if arr.ndim == 3 and arr.shape[2] == 3:
                linear_rgb = srgb_eotf(arr)
                linear_mono = linear_rgb_to_monochrome(linear_rgb)
            elif arr.ndim == 2:
                linear_mono = srgb_eotf(arr)
            else:
                raise ValueError(f"Unsupported image shape {arr.shape}")
        else:
            # Already float
            if arr.ndim == 3 and arr.shape[2] == 3:
                linear_mono = linear_rgb_to_monochrome(arr)
            else:
                linear_mono = np.asarray(arr, dtype=np.float64)

        # 2. Optics (discrete PSF)
        kernel, _, _ = self.config.get_psf_kernel()
        blurred = apply_psf(linear_mono, kernel)

        # 3. Geometric distortion
        map_u, map_v, validity_mask = self.config.get_backward_map()
        distorted = warp_image_distort(
            blurred, map_u, map_v, validity_mask=validity_mask, method=warp_method
        )

        # 4. Vignetting
        vign_map = self.config.get_vignetting_map()
        vignetted = apply_vignetting(distorted, vign_map)

        # 5. Sensor noise & ADC
        if auto_exposure:
            e_electrons = compute_auto_exposure(
                vignetted, self.config.full_well_e, self.config.ae_target
            )
        elif E_override is not None:
            e_electrons = E_override
        else:
            e_electrons = self.config.full_well_e

        seed_temporal = compute_seed_temporal(self.sensor_id, image_id, frame_idx)
        rng_temp = np.random.Generator(np.random.PCG64(seed_temporal))

        raw_dn = apply_sensor_noise(
            linear_radiance=vignetted,
            prnu_map=self.prnu_map,
            dsnu_map=self.dsnu_map,
            E=e_electrons,
            full_well_e=self.config.full_well_e,
            sigma_r_e=self.config.sigma_r_e,
            dark_current_e_per_s=self.config.dark_current_e_per_s,
            exposure_time_s=self.config.exposure_time_s,
            k_dn_per_e=self.config.k_dn_per_e,
            bits=self.config.bits,
            black_level_dn=self.config.black_level_dn,
            rng_temporal=rng_temp,
        )

        # 6. Display pipeline
        # Black level subtraction
        dn_sub = np.maximum(0.0, raw_dn.astype(np.float64) - self.config.black_level_dn)
        # Gain: normalize by maximum linear full-well signal
        max_signal_dn = self.config.k_dn_per_e * self.config.full_well_e
        linear_disp = np.clip(dn_sub / max_signal_dn, 0.0, 1.0)
        srgb_disp = srgb_oetf(linear_disp)
        display_8bit = srgb_to_8bit(srgb_disp)

        return {
            "raw": raw_dn,
            "display": display_8bit,
            "validity_mask": validity_mask,
            "E": e_electrons,
            "linear_radiance": vignetted,
        }


def save_raw_png(raw_dn: np.ndarray, output_path: str | Path) -> None:
    """Save 16-bit raw DN image as PNG."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr16 = np.asarray(raw_dn, dtype=np.uint16)
    img = Image.fromarray(arr16)
    img.save(str(path))


def save_display_png(display_8bit: np.ndarray, output_path: str | Path) -> None:
    """Save 8-bit display image as PNG."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr8 = np.asarray(display_8bit, dtype=np.uint8)
    img = Image.fromarray(arr8, mode="L")
    img.save(str(path))
