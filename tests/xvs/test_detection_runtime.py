"""Gate G0 Acceptance V6: Runtime benchmarking for classical detector D0.

Conforms strictly to docs/xvs/WP04_DETECTION.md v0.1 §7 (V6):
"Report ms per 1024² ROI (clean and degraded, each k) and the total CPU time."
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pytest

from xvs.detection.d0 import detect_d0, load_d0_config
from xvs.sensor.chain import SensorConfig
from xvs.sensor.psf import apply_psf
from xvs.sensor.noise import apply_sensor_noise, generate_fpn_maps

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v6_runtime_benchmarks():
    """Gate G0 Acceptance V6: Benchmark D0 runtime on 1024x1024 ROI for clean & degraded across k in [3, 5, 7]."""
    # 1024x1024 test image
    rng = np.random.default_rng(20260925)
    img_1024 = rng.uniform(0.1, 0.4, (1024, 1024)).astype(np.float64)

    d0_cfg = load_d0_config()
    sensor_cfg = SensorConfig.from_file(REPO_ROOT / "configs" / "xvs" / "sensors" / "ring_mono_v0_nodist.json")
    psf_kernel, _, _ = sensor_cfg.get_psf_kernel()
    prnu, dsnu = generate_fpn_maps("bench_cam", 1024, 1024, sensor_cfg.prnu_rms, sensor_cfg.dsnu_rms_e)

    # Pre-render degraded frame
    t_sensor_start = time.perf_counter()
    blurred = apply_psf(img_1024, psf_kernel)
    raw_dn = apply_sensor_noise(
        linear_radiance=blurred,
        prnu_map=prnu,
        dsnu_map=dsnu,
        E=sensor_cfg.full_well_e,
        full_well_e=sensor_cfg.full_well_e,
        sigma_r_e=sensor_cfg.sigma_r_e,
        dark_current_e_per_s=sensor_cfg.dark_current_e_per_s,
        exposure_time_s=sensor_cfg.exposure_time_s,
        k_dn_per_e=sensor_cfg.k_dn_per_e,
        bits=sensor_cfg.bits,
        black_level_dn=sensor_cfg.black_level_dn,
        rng_temporal=np.random.Generator(np.random.PCG64(999)),
    )
    img_deg_1024 = (raw_dn.astype(np.float64) - sensor_cfg.black_level_dn) / sensor_cfg.k_dn_per_e
    sensor_chain_ms = (time.perf_counter() - t_sensor_start) * 1000.0

    timings = {}
    k_values = [3, 5, 7]

    # Benchmark Warmup
    _ = detect_d0(img_1024, k=3, is_clean=True, config=d0_cfg)

    print("\n" + "=" * 72)
    print("  WP-04 D0 RUNTIME BENCHMARK (1024x1024 ROI, CPU Only)")
    print("=" * 72)
    print(f" Sensor degradation chain (PSF + EMVA noise): {sensor_chain_ms:.1f} ms")
    print("-" * 72)
    print(f" {'Sensor State':<15} | {'k':<5} | {'D0 Time (ms)':<15} | {'Rate (fps)':<12}")
    print("-" * 72)

    total_cpu_time_s = 0.0

    # Clean benchmarks
    for k in k_values:
        n_repeats = 1
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = detect_d0(img_1024, k=k, tau=4.0, is_clean=True, config=d0_cfg)
        elapsed_ms = ((time.perf_counter() - t0) / n_repeats) * 1000.0
        total_cpu_time_s += (elapsed_ms * n_repeats) / 1000.0
        fps = 1000.0 / elapsed_ms
        timings[f"clean_k{k}"] = elapsed_ms
        print(f" {'Clean':<15} | {k:<5} | {elapsed_ms:<15.2f} | {fps:<12.2f}")

    # Degraded benchmarks
    for k in k_values:
        n_repeats = 1
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = detect_d0(img_deg_1024, k=k, tau=4.0, is_clean=False, config=d0_cfg)
        elapsed_ms = ((time.perf_counter() - t0) / n_repeats) * 1000.0
        total_cpu_time_s += (elapsed_ms * n_repeats) / 1000.0
        fps = 1000.0 / elapsed_ms
        timings[f"degraded_k{k}"] = elapsed_ms
        print(f" {'Degraded':<15} | {k:<5} | {elapsed_ms:<15.2f} | {fps:<12.2f}")

    print("-" * 72)
    print(f" Total Benchmark CPU Time: {total_cpu_time_s:.2f} s")
    print("=" * 72)

    # Sanity checks
    for key, ms in timings.items():
        assert ms > 0.0, f"Benchmark {key} reported non-positive time"
        assert ms < 5000.0, f"Benchmark {key} took unexpectedly long: {ms:.1f} ms"
