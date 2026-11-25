"""Local CPU Verification Suite for Milestone M2 Subsystem M2-B (`V2-M2L-3`).

Covers SP2_M2_MODELS_DESIGN.md §§3.3, 4:
  1. `src/xvs/sensor/lwir.py`: 5-pt Gauss-Legendre Planck (< 1e-5% rel err vs scipy.integrate.quad),
     microphysics atmosphere (`clear`, `radiation_fog` > 10x LWIR/EO SNR advantage, `advection_fog`,
     `heavy_rain` honest tau_LWIR < tau_EO), area-averaged sub-pixel downsampling, Airy PSF,
     microbolometer inertia (`tau_th = 0.010 s`), NETD/FPN/NUC, and all 4 thermal faults.
  2. `src/xvs/v2/xvs_thermal/`: Asymmetric EO + 2-ch LWIR fusion, WP-07 deterministic physical
     health hard-gating, dual-blind guarantee (`f_fused == 0.0` & `vacuity == 1.0`), auxiliary
     uni-modal heads, `IRSTD-1k`/`HIT-UAV` loaders + license/denylist guard, and `40/20/20/10/10`
     fault curriculum sampler.
  3. `src/xvs/v2/xvs_world/`: `WorldTwinConfig` 1x1 km / 15M splat / 0.35 GB SPZ /
     `gsplat_3dgut_2dgs` / `VISUAL_STRESS_ONLY` hard guards and offline Colab G4 job manifest.
  4. Persists real measured machine-readable evidence to
     `output/xvs/v2_m2/v3_lwir_thermal_world_evidence.json`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import pytest
from scipy import integrate
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from xvs.sensor.lwir import (
    DLDT_300K,
    LWIR_BAND_MAX_M,
    LWIR_BAND_MIN_M,
    LwirSensorConfig,
    ThermalFaultConfig,
    compute_eo_lwir_regime_metrics,
    lwir_extinction_km_inv,
    planck_lwir_8_14,
    planck_lwir_8_14_dldt_scalar,
    planck_lwir_8_14_scalar,
    planck_spectral_radiance_scalar,
    simulate_lwir_frame,
)
from xvs.v2.xvs_thermal import (
    CURRICULUM_PROBS,
    FaultCurriculumSampler,
    HitUavDataset,
    Irstd1kDataset,
    ThermalSimplexMode,
    XvsThermalNet,
    assert_thermal_dataset_allowed,
    compute_thermal_multitask_loss,
    select_thermal_simplex_mode,
    summarize_curriculum_distribution,
)
from xvs.v2.xvs_world import (
    WorldTwinConfig,
    build_colab_g4_job_spec,
    estimate_gsplat_vram_gb,
    ksfo_28l_default_config,
    validate_frame_metadata_visual_stress_only,
    write_dry_run_manifest,
)

EVIDENCE_PATH = REPO_ROOT / "output" / "xvs" / "v2_m2" / "v3_lwir_thermal_world_evidence.json"

_EVIDENCE: dict[str, Any] = {
    "schema": "xvs.v2_m2.v3_lwir_thermal_world_evidence/1",
    "verified_utc": "",
    "lwir_physics": {},
    "xvs_thermal": {},
    "xvs_world": {},
    "pass": False,
}


def _persist_evidence(out_path: Path) -> None:
    _EVIDENCE["verified_utc"] = datetime.now(timezone.utc).isoformat()
    lw = _EVIDENCE["lwir_physics"]
    th = _EVIDENCE["xvs_thermal"]
    wd = _EVIDENCE["xvs_world"]
    _EVIDENCE["pass"] = bool(
        lw
        and th
        and wd
        and lw["pass"]
        and th["pass"]
        and wd["pass"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_EVIDENCE, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def isolated_evidence_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Isolated tmp_path target so pytest never clobbers GSN-pinned output/xvs/v2_m2/ evidence."""
    return tmp_path_factory.mktemp("v2_m2_lwir_evidence") / "v3_lwir_thermal_world_evidence.json"


def test_lwir_planck_quadrature_atmosphere_sensor_and_four_faults(isolated_evidence_path: Path) -> None:
    """Verify V1_THM & V2_THM: Planck < 1e-5% err, fog/rain microphysics, sensor chain & 4 faults."""
    # 1. 5-pt Gauss-Legendre Planck accuracy vs scipy.integrate.quad across 250 K..360 K & 200 K..1200 K
    temps_core = np.linspace(250.0, 360.0, 23)
    max_rel_err_pct_core = 0.0
    for t_k in temps_core:
        ref_val, _ = integrate.quad(
            lambda lam: planck_spectral_radiance_scalar(lam, float(t_k)),
            LWIR_BAND_MIN_M,
            LWIR_BAND_MAX_M,
            epsabs=1e-12,
            epsrel=1e-13,
        )
        gl_scalar = planck_lwir_8_14_scalar(float(t_k))
        gl_tensor = float(planck_lwir_8_14(torch.tensor(float(t_k), dtype=torch.float64)).item())
        rel_pct = abs(gl_scalar - ref_val) / ref_val * 100.0
        rel_pct_t = abs(gl_tensor - ref_val) / ref_val * 100.0
        max_rel_err_pct_core = max(max_rel_err_pct_core, rel_pct, rel_pct_t)

    assert max_rel_err_pct_core < 1e-5, (
        f"5-pt Gauss-Legendre error {max_rel_err_pct_core:.3e}% >= 1e-5% on 250..360 K"
    )

    # Check DR-04 §3.1 physical control points & extended 200..1200 K range
    temps_ext = [200.0, 233.15, 288.15, 300.0, 423.15, 650.0, 1200.0]
    max_rel_err_pct_ext = 0.0
    for t_k in temps_ext:
        ref_val, _ = integrate.quad(
            lambda lam: planck_spectral_radiance_scalar(lam, t_k),
            LWIR_BAND_MIN_M,
            LWIR_BAND_MAX_M,
            epsabs=1e-12,
            epsrel=1e-13,
        )
        gl_val = planck_lwir_8_14_scalar(t_k)
        max_rel_err_pct_ext = max(max_rel_err_pct_ext, abs(gl_val - ref_val) / ref_val * 100.0)
    assert max_rel_err_pct_ext < 1e-5
    assert abs(planck_lwir_8_14_scalar(233.15) - 15.1893) < 0.01
    assert abs(planck_lwir_8_14_scalar(288.15) - 45.5383) < 0.01
    assert abs(planck_lwir_8_14_scalar(300.0) - 54.9335) < 0.01
    assert abs(planck_lwir_8_14_scalar(650.0) - 756.5958) < 0.05
    assert abs(planck_lwir_8_14_dldt_scalar(300.0) - DLDT_300K) < 1e-4

    # 2. Microphysics atmospheric extinction: clear, radiation_fog (>10x SNR), advection_fog, heavy_rain
    m_clear = compute_eo_lwir_regime_metrics(rvr_km=15.0, range_km=1.0, regime="clear")
    m_rad = compute_eo_lwir_regime_metrics(rvr_km=0.5, range_km=1.0, regime="radiation_fog")
    m_adv = compute_eo_lwir_regime_metrics(rvr_km=0.5, range_km=1.0, regime="advection_fog")
    m_rain = compute_eo_lwir_regime_metrics(
        rvr_km=1.5, range_km=1.0, regime="heavy_rain", rain_mm_h=25.0
    )

    assert m_clear["tau_lwir"] > 0.75
    assert m_rad["snr_lwir_over_eo"] > 10.0, (
        f"Expected >10x LWIR/EO SNR advantage in radiation fog, got {m_rad['snr_lwir_over_eo']:.2f}x"
    )
    assert m_rad["tau_lwir_over_eo"] > 10.0
    assert m_adv["tau_lwir"] < m_rad["tau_lwir"]
    assert m_rain["tau_lwir"] < m_rain["tau_eo"], (
        f"Heavy rain must honestly degrade LWIR below EO: tau_lwir={m_rain['tau_lwir']:.4f} >= tau_eo={m_rain['tau_eo']:.4f}"
    )
    assert lwir_extinction_km_inv(0.5, "radiation_fog") < lwir_extinction_km_inv(0.5, "advection_fog")

    # 3. Sub-pixel area-averaged EO -> 640x512 energy conservation + Airy PSF + microbolometer inertia
    # Use 2x super-resolved grid (1024x1280 -> 512x640) for fast deterministic CPU check
    hr_h, hr_w = 1024, 1280
    base_temp = torch.full((hr_h, hr_w), 288.15, dtype=torch.float32)
    emiss = torch.full((hr_h, hr_w), 0.92, dtype=torch.float32)
    rng_km = torch.full((hr_h, hr_w), 1.0, dtype=torch.float32)
    elev_deg = torch.full((hr_h, hr_w), 0.0, dtype=torch.float32)

    cfg = LwirSensorConfig(out_hw=(512, 640), netd_mk=40.0, tau_th_s=0.010, dt_s=1.0 / 30.0)
    bg_out = simulate_lwir_frame(
        base_temp, emiss, rng_km, elev_deg, t_air_k=288.15, rvr_km=15.0,
        regime="clear", sensor_cfg=cfg, noise_off=True,
    )
    bg_sum = float(bg_out["rad_lwir"].sum().item())  # type: ignore[union-attr]

    # Single sub-pixel hot engine (1x1 px on HR grid = 0.25 native LWIR pixel) shifted across 4 sub-pixel offsets
    hot_energies: list[float] = []
    for dy in (0, 1):
        for dx in (0, 1):
            t_hot = base_temp.clone()
            t_hot[512 + dy, 640 + dx] = 410.0
            hot_out = simulate_lwir_frame(
                t_hot, emiss, rng_km, elev_deg, t_air_k=288.15, rvr_km=15.0,
                regime="clear", sensor_cfg=cfg, noise_off=True,
            )
            excess = float(hot_out["rad_lwir"].sum().item()) - bg_sum  # type: ignore[union-attr]
            hot_energies.append(excess)
    energy_cons_min = min(hot_energies) / max(hot_energies)
    assert energy_cons_min > 0.99, f"Sub-pixel energy conservation dropped to {energy_cons_min:.4f}"

    # Verify microbolometer thermal inertia (tau_th = 0.010 s -> alpha = 1 - exp(-(1/30)/0.010) ~ 0.9643)
    prev_bol = bg_out["L_bol"]
    assert isinstance(prev_bol, torch.Tensor)
    warm_temp = torch.full((hr_h, hr_w), 305.0, dtype=torch.float32)
    out_bol = simulate_lwir_frame(
        warm_temp, emiss, rng_km, elev_deg, t_air_k=288.15, rvr_km=15.0,
        regime="clear", sensor_cfg=cfg, prev_bol_rad=prev_bol, noise_off=True,
    )
    out_inst = simulate_lwir_frame(
        warm_temp, emiss, rng_km, elev_deg, t_air_k=288.15, rvr_km=15.0,
        regime="clear", sensor_cfg=cfg, prev_bol_rad=None, noise_off=True,
    )
    expected_alpha = 1.0 - math.exp(-(1.0 / 30.0) / 0.010)
    expected_l_bol = (1.0 - expected_alpha) * prev_bol + expected_alpha * out_inst["L_bol"]  # type: ignore[operator]
    assert torch.allclose(out_bol["L_bol"], expected_l_bol, atol=1e-4)  # type: ignore[arg-type]

    # Verify NETD temporal noise calibration on uniform 300 K blackbody (PRNU=0, FPN=0 -> pure NETD)
    cfg_netd = LwirSensorConfig(
        out_hw=(512, 640), netd_mk=40.0, prnu_std=0.0, fpn_col_ratio=0.0, nuc_drift_k=0.0
    )
    gen = torch.Generator().manual_seed(20261121)
    t_300 = torch.full((512, 640), 300.0, dtype=torch.float32)
    e_1 = torch.ones((512, 640), dtype=torch.float32)
    r_0 = torch.zeros((512, 640), dtype=torch.float32)
    el_0 = torch.zeros((512, 640), dtype=torch.float32)
    bb_out = simulate_lwir_frame(
        t_300, e_1, r_0, el_0, t_air_k=300.0, rvr_km=20.0,
        regime="clear", sensor_cfg=cfg_netd, generator=gen, noise_off=False,
    )
    target_ner = 0.040 * DLDT_300K
    emp_ner = float(bb_out["rad_lwir"].std(unbiased=False).item())  # type: ignore[union-attr]
    netd_rel_err_pct = abs(emp_ner - target_ner) / target_ner * 100.0
    assert netd_rel_err_pct < 5.0, f"NETD empirical noise off by {netd_rel_err_pct:.2f}%"

    # 4. Verify all 4 explicit thermal fault injectors on a structured sky+ground+runway scene
    yy = torch.linspace(0.0, 1.0, 512).unsqueeze(1).expand(512, 640)
    scene_t = torch.where(yy < 0.5, 288.0, 295.0 + 12.0 * (yy - 0.5))
    scene_r = torch.where(yy < 0.5, 100.0, 1.2)
    scene_el = torch.where(yy < 0.5, (0.5 - yy) * 40.0, 0.0)
    scene_e = torch.full((512, 640), 0.92, dtype=torch.float32)

    nom = simulate_lwir_frame(
        scene_t, scene_e, scene_r, scene_el, t_air_k=288.15, rvr_km=12.0,
        regime="clear", sensor_cfg=cfg, generator=torch.Generator().manual_seed(1),
    )
    nom_wp07 = nom["wp07_metrics"]
    assert isinstance(nom_wp07, dict)
    assert nom_wp07["q_lwir_phys"] == 1.0
    assert tuple(nom["rad_lwir"].shape) == (512, 640)  # type: ignore[union-attr]
    assert tuple(nom["two_channel"].shape) == (2, 512, 640)  # type: ignore[union-attr]

    # Fault 1: thermal_crossover
    f_cross = simulate_lwir_frame(
        scene_t, scene_e, torch.full_like(scene_r, 0.5), torch.zeros_like(scene_el),
        t_air_k=288.15, rvr_km=12.0, regime="clear", sensor_cfg=cfg,
        fault_cfg=ThermalFaultConfig(thermal_crossover=True, crossover_residual_k=0.01),
        noise_off=True,
    )
    wp_cross = f_cross["wp07_metrics"]
    assert isinstance(wp_cross, dict)
    assert wp_cross["crossover_fault"] == 1.0 and wp_cross["q_lwir_phys"] == 0.0

    # Fault 2: cloud_washout
    f_cloud = simulate_lwir_frame(
        scene_t, scene_e, scene_r, scene_el,
        t_air_k=288.15, rvr_km=0.2, regime="stratus_cloud", sensor_cfg=cfg,
        fault_cfg=ThermalFaultConfig(cloud_washout=True),
        noise_off=True,
    )
    wp_cloud = f_cloud["wp07_metrics"]
    assert isinstance(wp_cloud, dict)
    assert wp_cloud["washout_fault"] == 1.0 and wp_cloud["dynamic_range_k"] < 1.5
    assert wp_cloud["q_lwir_phys"] == 0.0

    # Fault 3: sun_in_fov_bloom
    f_sun = simulate_lwir_frame(
        scene_t, scene_e, scene_r, scene_el,
        t_air_k=288.15, rvr_km=12.0, regime="clear", sensor_cfg=cfg,
        fault_cfg=ThermalFaultConfig(sun_in_fov_bloom=True, sun_uv_px=(320.0, 120.0)),
        noise_off=True,
    )
    wp_sun = f_sun["wp07_metrics"]
    assert isinstance(wp_sun, dict)
    assert float(f_sun["dn14"].max().item()) == 16383.0  # type: ignore[union-attr]
    assert wp_sun["bloom_fault"] == 1.0 and wp_sun["saturated_pixels"] >= 12.0

    # Fault 4: ffc_freeze
    f_ffc = simulate_lwir_frame(
        scene_t + 15.0, scene_e, scene_r, scene_el,
        t_air_k=288.15, rvr_km=12.0, regime="clear", sensor_cfg=cfg,
        fault_cfg=ThermalFaultConfig(ffc_freeze=True),
        prev_bol_rad=nom["L_bol"],  # type: ignore[arg-type]
        prev_meas_rad=nom["rad_lwir"],  # type: ignore[arg-type]
    )
    wp_ffc = f_ffc["wp07_metrics"]
    assert isinstance(wp_ffc, dict)
    assert wp_ffc["ffc_freeze_fault"] == 1.0
    assert wp_ffc["interframe_diff_var"] == 0.0
    assert wp_ffc["q_lwir_phys"] == 0.0

    _EVIDENCE["lwir_physics"] = {
        "planck_max_rel_err_pct_250_360k": max_rel_err_pct_core,
        "planck_max_rel_err_pct_200_1200k": max_rel_err_pct_ext,
        "planck_300k_w_m2_sr": planck_lwir_8_14_scalar(300.0),
        "radiation_fog_lwir_over_eo_snr_ratio": m_rad["snr_lwir_over_eo"],
        "radiation_fog_lwir_over_eo_tau_ratio": m_rad["tau_lwir_over_eo"],
        "advection_fog_tau_lwir": m_adv["tau_lwir"],
        "heavy_rain_tau_lwir": m_rain["tau_lwir"],
        "heavy_rain_tau_eo": m_rain["tau_eo"],
        "subpixel_energy_conservation_min": energy_cons_min,
        "netd_empirical_rel_err_pct": netd_rel_err_pct,
        "tau_th_s": cfg.tau_th_s,
        "four_faults_verified": {
            "thermal_crossover": True,
            "cloud_washout": True,
            "sun_in_fov_bloom": True,
            "ffc_freeze": True,
        },
        "pass": True,
    }
    _persist_evidence(isolated_evidence_path)


def test_xvs_thermal_net_hard_gating_dual_blind_datasets_and_curriculum(
    tmp_path: Path, isolated_evidence_path: Path
) -> None:
    """Verify XVS-ThermalNet hard-gating, dual-blind vacuity=1.0, datasets guard & 40/20/20/10/10 curriculum."""
    torch.manual_seed(20261121)
    model = XvsThermalNet(eo_in_ch=3, lwir_in_ch=2, feat_ch=32, num_evidential_classes=3)
    model.eval()

    x_eo = torch.randn(2, 3, 128, 128)
    x_lwir = torch.randn(2, 2, 64, 80)

    # 1. Nominal forward check (asymmetric shapes align cleanly, all heads finite)
    q_one = torch.ones(2)
    q_zero = torch.zeros(2)
    out_nom = model(x_eo, x_lwir, q_eo=q_one, q_lwir=q_one)
    assert out_nom["f_fused"].shape == (2, 32, 32, 32)
    assert out_nom["eo_aux_hm_logits"].shape == (2, 1, 32, 32)
    assert out_nom["lwir_aux_hm_logits"].shape == (2, 1, 32, 32)
    assert out_nom["vacuity"].shape == (2, 1, 32, 32)
    assert torch.all(torch.isfinite(out_nom["f_fused"]))
    assert float(out_nom["vacuity"].max().item()) < 1.0

    # 2. EO blind (q_eo == 0.0, q_lwir == 1.0) -> w_eo_eff == 0.0 exact & ZERO EO leakage
    out_eo_blind_a = model(x_eo, x_lwir, q_eo=q_zero, q_lwir=q_one)
    x_eo_corrupt = torch.full_like(x_eo, 1e5)
    out_eo_blind_b = model(x_eo_corrupt, x_lwir, q_eo=q_zero, q_lwir=q_one)
    assert float(out_eo_blind_a["w_eo_eff"].abs().max().item()) == 0.0
    assert float((out_eo_blind_a["w_lwir_eff"] - 1.0).abs().max().item()) == 0.0
    eo_leak_diff = float((out_eo_blind_a["f_fused"] - out_eo_blind_b["f_fused"]).abs().max().item())
    assert eo_leak_diff == 0.0, f"Blind EO leaked {eo_leak_diff:.3e} into f_fused"

    # 3. LWIR blind (q_lwir == 0.0, q_eo == 1.0) -> w_lwir_eff == 0.0 exact & ZERO LWIR leakage
    out_lwir_blind_a = model(x_eo, x_lwir, q_eo=q_one, q_lwir=q_zero)
    x_lwir_corrupt = torch.full_like(x_lwir, -1e5)
    out_lwir_blind_b = model(x_eo, x_lwir_corrupt, q_eo=q_one, q_lwir=q_zero)
    assert float(out_lwir_blind_a["w_lwir_eff"].abs().max().item()) == 0.0
    assert float((out_lwir_blind_a["w_eo_eff"] - 1.0).abs().max().item()) == 0.0
    lwir_leak_diff = float(
        (out_lwir_blind_a["f_fused"] - out_lwir_blind_b["f_fused"]).abs().max().item()
    )
    assert lwir_leak_diff == 0.0, f"Blind LWIR leaked {lwir_leak_diff:.3e} into f_fused"

    # 4. Dual-Blind Guarantee (BC-M2-m2): q_eo == 0.0 AND q_lwir == 0.0 -> fused == 0.0 & vacuity == 1.0
    out_dual = model(x_eo, x_lwir, q_eo=q_zero, q_lwir=q_zero)
    dual_fused_max = float(out_dual["f_fused"].abs().max().item())
    dual_w_eo_max = float(out_dual["w_eo_eff"].abs().max().item())
    dual_w_lwir_max = float(out_dual["w_lwir_eff"].abs().max().item())
    dual_ev_max = float(out_dual["evidence"].abs().max().item())
    dual_vac_min = float(out_dual["vacuity"].min().item())
    dual_vac_max = float(out_dual["vacuity"].max().item())

    assert dual_w_eo_max == 0.0 and dual_w_lwir_max == 0.0
    assert dual_fused_max == 0.0, f"Dual-blind fused feature not zero: {dual_fused_max}"
    assert dual_ev_max == 0.0
    assert dual_vac_min == 1.0 and dual_vac_max == 1.0, (
        f"Dual-blind vacuity must be exactly 1.0, got [{dual_vac_min}, {dual_vac_max}]"
    )

    # Simplex RTA mode switch check (including exhaustive D16-M2-M2 adversarial NaN/±Inf/OOR fail-closed)
    assert select_thermal_simplex_mode(1.0, 1.0, 0.2) == ThermalSimplexMode.NOMINAL_FUSION
    assert select_thermal_simplex_mode(0.0, 1.0, 0.2) == ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY
    assert select_thermal_simplex_mode(1.0, 0.0, 0.2) == ThermalSimplexMode.LWIR_DEGRADED_EO_PRIMARY
    assert select_thermal_simplex_mode(0.0, 0.0, dual_vac_min) == ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK

    bad_vals = (float("nan"), float("inf"), float("-inf"), -0.5, 2.5)
    adversarial_checks = 0
    for bad in bad_vals:
        # Invalid q_eo rejected -> falls back to LWIR primary (never NOMINAL)
        assert select_thermal_simplex_mode(bad, 1.0, 0.2) == ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY
        # Invalid q_lwir rejected -> falls back to EO primary (never NOMINAL)
        assert select_thermal_simplex_mode(1.0, bad, 0.2) == ThermalSimplexMode.LWIR_DEGRADED_EO_PRIMARY
        # Both invalid -> DUAL_DVE_SVS_FALLBACK
        assert select_thermal_simplex_mode(bad, bad, 0.2) == ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK
        # Invalid vacuity -> ALWAYS DUAL_DVE_SVS_FALLBACK
        assert select_thermal_simplex_mode(1.0, 1.0, bad) == ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK
        # Invalid thresholds tau_ok / tau_vac -> ALWAYS DUAL_DVE_SVS_FALLBACK
        assert select_thermal_simplex_mode(1.0, 1.0, 0.2, tau_ok=bad) == ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK
        assert select_thermal_simplex_mode(1.0, 1.0, 0.2, tau_vac=bad) == ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK
        adversarial_checks += 6
    # Boundary threshold checks (tau_ok=0.0 or tau_vac=0.0 must also fail closed)
    assert select_thermal_simplex_mode(1.0, 1.0, 0.2, tau_ok=0.0) == ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK
    assert select_thermal_simplex_mode(1.0, 1.0, 0.2, tau_vac=0.0) == ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK
    adversarial_checks += 2

    # Tensor NaN / ±Inf / OOR health inputs on XvsThermalNet MUST zero the corrupted stream (fail-closed!)
    for bad_q in (float("nan"), float("inf"), float("-inf"), -0.5, 2.5):
        q_bad = torch.full((2,), bad_q)
        out_bad_both = model(x_eo, x_lwir, q_eo=q_bad, q_lwir=q_bad)
        assert float(out_bad_both["w_eo_eff"].abs().max().item()) == 0.0
        assert float(out_bad_both["w_lwir_eff"].abs().max().item()) == 0.0
        assert float(out_bad_both["f_fused"].abs().max().item()) == 0.0
        assert float(out_bad_both["vacuity"].min().item()) == 1.0
        assert float(out_bad_both["vacuity"].max().item()) == 1.0
        out_bad_eo = model(x_eo, x_lwir, q_eo=q_bad, q_lwir=q_one)
        assert float(out_bad_eo["w_eo_eff"].abs().max().item()) == 0.0
        assert float((out_bad_eo["w_lwir_eff"] - 1.0).abs().max().item()) == 0.0
        out_bad_lwir = model(x_eo, x_lwir, q_eo=q_one, q_lwir=q_bad)
        assert float(out_bad_lwir["w_lwir_eff"].abs().max().item()) == 0.0
        assert float((out_bad_lwir["w_eo_eff"] - 1.0).abs().max().item()) == 0.0
        adversarial_checks += 3

    # Multi-task loss backward pass check
    model.train()
    out_tr = model(x_eo, x_lwir, q_eo=q_one, q_lwir=q_one)
    tgt_hm = torch.zeros(2, 1, 32, 32)
    tgt_hm[:, :, 16, 16] = 1.0
    tgt_cls = torch.zeros(2, 32, 32, dtype=torch.long)
    losses = compute_thermal_multitask_loss(out_tr, tgt_hm, tgt_cls)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()

    # 5. License-clean dataset loaders (IRSTD-1k MIT + HIT-UAV CC-BY-4.0) & denylist guard
    irstd_root = tmp_path / "irstd1k"
    (irstd_root / "IRSTD1k_Img").mkdir(parents=True)
    (irstd_root / "IRSTD1k_Label").mkdir(parents=True)
    Image.fromarray(np.full((64, 64), 120, dtype=np.uint8)).save(irstd_root / "IRSTD1k_Img" / "s0.png")
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_arr[28:34, 30:36] = 255
    Image.fromarray(mask_arr).save(irstd_root / "IRSTD1k_Label" / "s0.png")
    ds_irstd = Irstd1kDataset(irstd_root, split_purpose="train", target_hw=(64, 64))
    assert len(ds_irstd) == 1
    s_ir = ds_irstd[0]
    assert tuple(s_ir.lwir_2ch.shape) == (2, 64, 64) and s_ir.boxes_xyxy.shape == (1, 4)

    hituav_root = tmp_path / "hit_uav"
    (hituav_root / "images").mkdir(parents=True)
    (hituav_root / "labels").mkdir(parents=True)
    Image.fromarray(np.full((64, 80), 90, dtype=np.uint8)).save(hituav_root / "images" / "u0.png")
    (hituav_root / "labels" / "u0.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    ds_hit = HitUavDataset(hituav_root, split_purpose="train", target_hw=(64, 80))
    assert len(ds_hit) == 1
    s_hu = ds_hit[0]
    assert tuple(s_hu.lwir_2ch.shape) == (2, 64, 80) and s_hu.boxes_xyxy.shape == (1, 4)

    banned_ids = [
        "nudt_sirst",
        "rgbt_tiny",
        "anti_uav",
        "vedai",
        "pst900",
        "aerial_uav_thermal_inferred_unified_kaggle",
        "thermal_uav_scc_loc",
    ]
    for bid in banned_ids:
        with pytest.raises(PermissionError):
            assert_thermal_dataset_allowed(bid, split_purpose="train")
        with pytest.raises(PermissionError):
            assert_thermal_dataset_allowed(bid, split_purpose="eval")

    eval_only_ids = ["cart", "massmind", "uav_cb", "flir_adas_v2"]
    for eid in eval_only_ids:
        with pytest.raises(PermissionError):
            assert_thermal_dataset_allowed(eid, split_purpose="train")
        assert_thermal_dataset_allowed(eid, split_purpose="eval")

    # 6. Seeded DR-04:205 40/20/20/10/10 Fault Curriculum Sampler empirical distribution
    sampler = FaultCurriculumSampler(seed=20261121)
    batch = sampler.sample_batch(10_000)
    dist = summarize_curriculum_distribution(batch)
    for bucket, expected_p in CURRICULUM_PROBS.items():
        assert abs(dist[bucket] - expected_p) < 0.015, (
            f"Curriculum bucket {bucket} drifted: got {dist[bucket]:.4f} vs {expected_p:.4f}"
        )
    dve_samples = [s for s in batch if s.bucket == "simultaneous_dve"]
    assert all(s.q_eo == 0.0 and s.q_lwir == 0.0 and s.target_vacuity == 1.0 for s in dve_samples)

    _EVIDENCE["xvs_thermal"] = {
        "eo_blind_zero_leak_max_diff": eo_leak_diff,
        "lwir_blind_zero_leak_max_diff": lwir_leak_diff,
        "dual_blind_fused_max_abs": dual_fused_max,
        "dual_blind_vacuity": dual_vac_min,
        "simplex_adversarial_cases_verified": adversarial_checks,
        "aux_heads_verified": True,
        "multitask_loss_finite": True,
        "irstd1k_and_hituav_loaders_verified": True,
        "banned_datasets_blocked_count": len(banned_ids),
        "eval_only_train_blocked_count": len(eval_only_ids),
        "curriculum_n_samples": len(batch),
        "curriculum_empirical_probs": dist,
        "pass": True,
    }
    _persist_evidence(isolated_evidence_path)


def test_xvs_world_scoped_1km_contract_and_dry_run_manifest(
    tmp_path: Path, isolated_evidence_path: Path
) -> None:
    """Verify WorldTwinConfig hard caps (<= 1 km, <= 15M splats, <= 0.35 GB SPZ, VISUAL_STRESS_ONLY)."""
    cfg = ksfo_28l_default_config()
    assert cfg.extent_x_m == 1000.0 and cfg.extent_y_m == 1000.0
    assert cfg.max_splats == 15_000_000
    assert cfg.spz_max_gb == 0.35
    assert cfg.backend == "gsplat_3dgut_2dgs"
    assert cfg.gt_tier == "VISUAL_STRESS_ONLY"

    vram_1km_gb = estimate_gsplat_vram_gb(cfg.max_splats)
    vram_5km_gb = estimate_gsplat_vram_gb(300_000_000)
    assert vram_1km_gb < 25.0 and vram_5km_gb > 96.0

    # Every scope violation must raise hard ValueError
    bad_cases = [
        {"extent_x_m": 1000.1},
        {"extent_y_m": 5000.0},
        {"extent_x_m": 0.0},
        {"max_splats": 15_000_001},
        {"spz_max_gb": 0.50},
        {"backend": "vanilla_3dgs"},
        {"gt_tier": "WP01_CERTIFICATION"},
        {"sh_degree": 3},
        {"min_faa_adip_gcps": 4},
        {"packed_memory_mode": False},
    ]
    for overrides in bad_cases:
        with pytest.raises(ValueError):
            WorldTwinConfig(**overrides)  # type: ignore[arg-type]

    # Offline dry-run Colab G4 job spec + manifest round-trip
    manifest_path = tmp_path / "ksfo_28l_world_dry_run.json"
    spec = write_dry_run_manifest(cfg, manifest_path)
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["gt_tier"] == "VISUAL_STRESS_ONLY"
    assert loaded["wp01_certification_eligible"] is False
    assert loaded["fits_single_g4_96gb"] is True
    assert loaded["config_sha256"] == cfg.config_sha256()
    assert "--with_ut" in spec["gsplat_command"] and "--packed" in spec["gsplat_command"]

    # Frame metadata CI guard rejects any attempt to mark 3DGS as WP-01 eligible
    with pytest.raises(ValueError):
        validate_frame_metadata_visual_stress_only(
            {"gt_tier": "WP01_CERTIFICATION", "wp01_certification_eligible": False, "gcp_reproj_uncertainty_m": 0.2}
        )
    with pytest.raises(ValueError):
        validate_frame_metadata_visual_stress_only(
            {"gt_tier": "VISUAL_STRESS_ONLY", "wp01_certification_eligible": True, "gcp_reproj_uncertainty_m": 0.2}
        )

    _EVIDENCE["xvs_world"] = {
        "ksfo_28l_config_sha256": cfg.config_sha256(),
        "extent_x_m": cfg.extent_x_m,
        "extent_y_m": cfg.extent_y_m,
        "max_splats": cfg.max_splats,
        "spz_max_gb": cfg.spz_max_gb,
        "backend": cfg.backend,
        "gt_tier": cfg.gt_tier,
        "estimated_vram_1km_gb": round(vram_1km_gb, 3),
        "estimated_vram_5km_oom_gb": round(vram_5km_gb, 3),
        "guards_enforced_count": len(bad_cases) + 2,
        "dry_run_manifest_verified": True,
        "pass": True,
    }
    _persist_evidence(isolated_evidence_path)
    assert isolated_evidence_path.is_file()
