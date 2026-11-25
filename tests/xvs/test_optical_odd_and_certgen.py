"""Verification Suite for DR-13 5-Mode Optical-ODD Monitor & DR-14 xvs-certgen Synthesizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from xvs.v2.xvs_optical_odd import (
    DeterministicOpticalOddMonitor,
    MltlPersistenceSpec,
    OpticalOddAction,
    OpticalOddConfig,
    OpticalOddDecision,
    OpticalOddFrameInput,
    OpticalOddMode,
    compute_buck_dew_frost_point_c,
)
from xvs.v2.xvs_vlm_gate import (
    AsymmetricSimplexVlmGate,
    GateStatus,
    VlmAdvisoryPacket,
    VlmHazard,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTGEN_PATH = REPO_ROOT / "scripts" / "xvs" / "xvs_certgen.py"


def _load_certgen_module():
    spec = importlib.util.spec_from_file_location("xvs_certgen", CERTGEN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["xvs_certgen"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fast_config() -> OpticalOddConfig:
    """Compact deterministic MLTL windows (2-of-3, 1-tick debounce) for fast unit verification."""
    fast_spec = MltlPersistenceSpec(confirm_m=2, window_n=3, debounce_ticks=1)
    return OpticalOddConfig(
        raindrop_tile_persist_frames=2,
        mltl_specs={mode: fast_spec for mode in OpticalOddMode},
    )


def _clean_eo_vfr(h: int = 272, w: int = 364) -> np.ndarray:
    """Sharp high-contrast horizon scene with crisp sky/terrain boundary and fine HF texture."""
    yy, xx = np.indices((h, w), dtype=np.float32)
    img = np.zeros((h, w), dtype=np.float32)
    # Bright clean sky above horizon, rich textured terrain below horizon
    sky = yy < (h // 2)
    img[sky] = 620.0 + 3.0 * np.sin(xx[sky] * 0.25)
    terrain = ~sky
    # Multi-frequency checkerboard + ridges (strong Tenengrad, MLAP, and HF MTF energy)
    hf_tex = (
        110.0 * np.sin(xx[terrain] * 1.45)
        + 95.0 * np.cos(yy[terrain] * 1.65)
        + 60.0 * np.sin((xx[terrain] + yy[terrain]) * 1.1)
    )
    img[terrain] = 280.0 + hf_tex
    return np.clip(img, 0.0, 1023.0).astype(np.uint16)


def _clean_lwir(h: int = 128, w: int = 160) -> np.ndarray:
    """Healthy 14-bit Pre-AGC Boson+ frame with rich scene contrast (std >> 6*NETD_DN)."""
    yy, xx = np.indices((h, w), dtype=np.float32)
    scene = 8192.0 + 180.0 * np.sin(yy * 0.08) + 140.0 * np.cos(xx * 0.07)
    return np.clip(scene, 0.0, 16383.0).astype(np.uint16)


# ---------------------------------------------------------------------------
# DR-13: 5-Mode Deterministic Optical-ODD Monitor Tests
# ---------------------------------------------------------------------------


def test_clean_vfr_stays_nominal_with_af447_lock() -> None:
    mon = DeterministicOpticalOddMonitor(_fast_config())
    eo = _clean_eo_vfr()
    lwir = _clean_lwir()

    for _ in range(4):
        dec = mon.evaluate(
            OpticalOddFrameInput(
                eo_frame_u16=eo,
                lwir_frame_u16=lwir,
                sun_boresight_deg=95.0,
                t_win_c=22.0,
                t_amb_c=20.0,
                rh_pct=40.0,
                boson_dt_ffc_s=45.0,
                boson_t_cam_c=25.0,
                boson_t_last_ffc_c=24.8,
                beta_ext_km_inv=0.20,
            )
        )
    assert dec.action == OpticalOddAction.NOMINAL
    assert dec.health_bitmask == 0
    assert dec.vlm_gate_reject is False
    assert dec.airdata_export_blocked is True
    assert dec.metrics.estimated_mor_m > 10_000.0
    assert dec.metrics.horizon_snr_db > 3.0


def test_mode1_raindrop_occlusion_triggers_and_hands_off_to_thermal() -> None:
    mon = DeterministicOpticalOddMonitor(_fast_config())
    eo = _clean_eo_vfr().copy()
    # Blur a large persistent raindrop patch covering ~40% of frame (flat water-film patch)
    eo[80:240, 40:300] = 350
    lwir = _clean_lwir()

    decs = [
        mon.evaluate(
            OpticalOddFrameInput(
                eo_frame_u16=eo,
                lwir_frame_u16=lwir,
                sun_boresight_deg=110.0,
                beta_ext_km_inv=0.30,
            )
        )
        for _ in range(4)
    ]
    assert decs[-1].raw_mode_flags[OpticalOddMode.RAINDROP_OCCLUSION] is True
    assert decs[-1].persisted_mode_flags[OpticalOddMode.RAINDROP_OCCLUSION] is True
    assert decs[-1].action == OpticalOddAction.THERMAL_PRIMARY
    assert decs[-1].vlm_gate_reject is True
    assert decs[-1].metrics.droplet_occlusion_frac > 0.12


def test_mode2_window_icing_buck_svp_and_hard_af447_zero_airdata_ban() -> None:
    # Verify Buck SVP dew/frost physics: at 0 °C & 95% RH, frost point is near -0.6 °C
    t_frost = compute_buck_dew_frost_point_c(0.0, 95.0)
    assert -1.5 < t_frost < 0.0

    # AF447 invariant cannot be disabled even by direct construction
    with pytest.raises(ValueError, match="AF447"):
        OpticalOddDecision(
            action=OpticalOddAction.NOMINAL,
            raw_mode_flags={m: False for m in OpticalOddMode},
            persisted_mode_flags={m: False for m in OpticalOddMode},
            mltl_window_counts={m: 0 for m in OpticalOddMode},
            health_bitmask=0,
            mask_lwir_frame=False,
            vlm_gate_reject=False,
            ito_heater_cmd=False,
            attitude_in_odd=True,
            metrics=DeterministicOpticalOddMonitor(_fast_config())
            .evaluate(OpticalOddFrameInput(_clean_eo_vfr(), _clean_lwir(), beta_ext_km_inv=0.2))
            .metrics,
            airdata_export_blocked=False,
        )

    mon = DeterministicOpticalOddMonitor(_fast_config())
    # Frosted EO: only ultra-low-frequency gradient (HF MTF collapses << 0.35 * eta0)
    yy, xx = np.indices((272, 364), dtype=np.float32)
    iced_eo = np.clip(450.0 + 40.0 * np.sin(yy * 0.03), 0, 1023).astype(np.uint16)
    # Liquid water / ice film >= 70 um is opaque in 8-14 um LWIR -> uniform T_win (std < 3.5*NETD_DN)
    opaque_lwir = np.full((128, 160), 8192, dtype=np.uint16)

    for _ in range(3):
        dec = mon.evaluate(
            OpticalOddFrameInput(
                eo_frame_u16=iced_eo,
                lwir_frame_u16=opaque_lwir,
                sun_boresight_deg=90.0,
                t_win_c=-0.5,
                t_amb_c=0.0,
                rh_pct=96.0,
                beta_ext_km_inv=0.5,
            )
        )
    assert dec.raw_mode_flags[OpticalOddMode.WINDOW_ICING] is True
    assert dec.persisted_mode_flags[OpticalOddMode.WINDOW_ICING] is True
    assert dec.action == OpticalOddAction.DEGRADED_FALLBACK
    assert dec.ito_heater_cmd is True
    assert dec.vlm_gate_reject is True
    assert dec.airdata_export_blocked is True
    assert dec.metrics.window_dew_margin_c <= 2.0


def test_mode3_solar_glare_spa_gated_vs_out_of_fov_ungated() -> None:
    mon = DeterministicOpticalOddMonitor(_fast_config())
    glared_eo = _clean_eo_vfr().copy()
    # Saturate > 10% of EO frame above 1008 DN
    glared_eo[10:120, 100:260] = 1020
    lwir = _clean_lwir()

    # 1. Sun OUT of SPA cone (theta_sun = 80 deg >= 60/2 + 12 = 42 deg) -> MUST NOT trip glare
    for _ in range(3):
        dec_out = mon.evaluate(
            OpticalOddFrameInput(
                eo_frame_u16=glared_eo,
                lwir_frame_u16=lwir,
                sun_boresight_deg=80.0,
                eo_hfov_deg=60.0,
                beta_ext_km_inv=0.2,
            )
        )
    assert dec_out.metrics.spa_in_glare_cone is False
    assert dec_out.raw_mode_flags[OpticalOddMode.SOLAR_GLARE] is False
    assert dec_out.persisted_mode_flags[OpticalOddMode.SOLAR_GLARE] is False

    # 2. Sun INSIDE SPA cone (theta_sun = 18 deg < 42 deg) -> trips glare & elevates EVS+LWIR
    mon.reset()
    for _ in range(3):
        dec_in = mon.evaluate(
            OpticalOddFrameInput(
                eo_frame_u16=glared_eo,
                lwir_frame_u16=lwir,
                sun_boresight_deg=18.0,
                eo_hfov_deg=60.0,
                beta_ext_km_inv=0.2,
            )
        )
    assert dec_in.metrics.spa_in_glare_cone is True
    assert dec_in.raw_mode_flags[OpticalOddMode.SOLAR_GLARE] is True
    assert dec_in.persisted_mode_flags[OpticalOddMode.SOLAR_GLARE] is True
    assert dec_in.action == OpticalOddAction.EVS_LWIR_ELEVATED
    assert dec_in.vlm_gate_reject is True


def test_mode4_fog_mor_under_1609m_triggers_thermal_primary_and_masks_vlm_gate() -> None:
    mon = DeterministicOpticalOddMonitor(_fast_config())
    # Dense uniform fog veil -> high airlight floor, flat horizon
    fog_eo = np.full((272, 364), 890, dtype=np.uint16)
    healthy_lwir = _clean_lwir()

    for _ in range(3):
        dec = mon.evaluate(
            OpticalOddFrameInput(
                eo_frame_u16=fog_eo,
                lwir_frame_u16=healthy_lwir,
                sun_boresight_deg=100.0,
                beta_ext_km_inv=2.5,  # MOR = 2.996 / 2.5 = 1198.4 m < 1609 m
            )
        )
    assert dec.metrics.estimated_mor_m < 1609.0
    assert dec.persisted_mode_flags[OpticalOddMode.FOG_VISIBILITY] is True
    assert dec.action == OpticalOddAction.THERMAL_PRIMARY

    # Direct integration with DR-09 AsymmetricSimplexVlmGate Stage 3!
    vlm_odd = dec.to_vlm_optical_odd_health()
    vlm_gate = AsymmetricSimplexVlmGate()
    pkt = VlmAdvisoryPacket(
        seq=1,
        capture_ts_ms=5000,
        hazard_enum=int(VlmHazard.CRANE_PROXY),
        azimuth_sector=2,
    ).pack()
    vlm_dec = vlm_gate.evaluate(
        packet_or_bytes=pkt,
        now_ms=5040,
        optical_odd=vlm_odd,
        certified_geometric_hazards=frozenset({"LWIR:THERMAL_TRACK_01"}),
    )
    assert vlm_dec.status == GateStatus.OPTICAL_ODD_MASKED
    assert vlm_dec.stage_reached == 3
    assert vlm_dec.final_hazards == frozenset({"LWIR:THERMAL_TRACK_01"})


def test_mode5_lwir_stale_ffc_and_bad_pixel_surge_plus_ffc_freeze_mask() -> None:
    mon = DeterministicOpticalOddMonitor(_fast_config())
    eo = _clean_eo_vfr()
    bad_lwir = _clean_lwir().copy()
    # Inject > 2% bad-pixel spikes (above 1.0% Industrial threshold)
    bad_lwir[10:110:4, 10:150:4] = 16000

    # 1. During active 1-tick Boson+ FFC freeze -> masked without tripping Mode 5
    dec_freeze = mon.evaluate(
        OpticalOddFrameInput(
            eo_frame_u16=eo,
            lwir_frame_u16=bad_lwir,
            boson_dt_ffc_s=350.0,
            boson_t_cam_c=28.0,
            boson_t_last_ffc_c=25.0,
            boson_ffc_freeze=True,
            beta_ext_km_inv=0.2,
        )
    )
    assert dec_freeze.mask_lwir_frame is True
    assert dec_freeze.raw_mode_flags[OpticalOddMode.LWIR_BOLOMETER_DRIFT] is False

    # 2. Normal operation with stale FFC (> 300 s) + bad-pixel surge -> Mode 5 trips!
    for _ in range(3):
        dec = mon.evaluate(
            OpticalOddFrameInput(
                eo_frame_u16=eo,
                lwir_frame_u16=bad_lwir,
                boson_dt_ffc_s=360.0,
                boson_t_cam_c=27.5,
                boson_t_last_ffc_c=25.0,
                boson_ffc_freeze=False,
                beta_ext_km_inv=0.2,
            )
        )
    assert dec.metrics.lwir_bad_pixel_frac > 0.010
    assert dec.persisted_mode_flags[OpticalOddMode.LWIR_BOLOMETER_DRIFT] is True


def test_mltl_persistence_blocks_single_frame_glitch() -> None:
    # Require 3-of-5 with 2-tick G[0,1] debounce -> 1 glitch tick must NOT trip persisted mode
    cfg = OpticalOddConfig(
        mltl_specs={
            mode: MltlPersistenceSpec(confirm_m=3, window_n=5, debounce_ticks=2)
            for mode in OpticalOddMode
        }
    )
    mon = DeterministicOpticalOddMonitor(cfg)
    fog_eo = np.full((272, 364), 890, dtype=np.uint16)
    clean_eo = _clean_eo_vfr()
    lwir = _clean_lwir()

    d1 = mon.evaluate(OpticalOddFrameInput(fog_eo, lwir, beta_ext_km_inv=2.5))
    assert d1.raw_mode_flags[OpticalOddMode.FOG_VISIBILITY] is True
    assert d1.persisted_mode_flags[OpticalOddMode.FOG_VISIBILITY] is False
    assert d1.mltl_window_counts[OpticalOddMode.FOG_VISIBILITY] == 1

    d2 = mon.evaluate(OpticalOddFrameInput(clean_eo, lwir, beta_ext_km_inv=0.2))
    assert d2.persisted_mode_flags[OpticalOddMode.FOG_VISIBILITY] is False
    assert d2.action == OpticalOddAction.NOMINAL


# ---------------------------------------------------------------------------
# DR-14: xvs-certgen GSN v3 Automated Evidence Synthesizer Tests
# ---------------------------------------------------------------------------


def test_xvs_certgen_live_repo_passes_and_emits_valid_bundle(tmp_path: Path) -> None:
    certgen = _load_certgen_module()
    out_file = tmp_path / "xvs_certgen_bundle.json"
    rc = certgen.main(["--check", "--out", str(out_file)])
    assert rc == 0
    assert out_file.is_file()

    bundle = certgen.synthesize_assurance_bundle()
    assert bundle["status"] == "PASS"
    assert bundle["errors"] == []
    assert bundle["forbidden_claims_audit"]["clean"] is True
    assert "G5_GAP" in bundle["gsn_source"]["undeveloped_nodes"]
    assert "G2_TLA_DEFERRED" in bundle["gsn_source"]["undeveloped_nodes"]
    assert len(bundle["bundle_sha256"]) == 64


def test_xvs_certgen_fails_closed_when_g5_gap_falsely_cleared_or_forbidden_claim(
    tmp_path: Path,
) -> None:
    certgen = _load_certgen_module()
    live_gsn = (REPO_ROOT / "assurance" / "openxvs_v2.gsn.yaml").read_text(encoding="utf-8")

    # 1. Falsely clear G5_GAP undeveloped: true -> undeveloped: false => MUST FAIL CLOSED
    tampered_gap = live_gsn.replace(
        'G5_GAP:\n  text: "EXPLICIT',
        'G5_GAP:\n  text: "EXPLICIT',
    )
    # Replace final G5_GAP undeveloped: true with false
    idx = tampered_gap.rfind("undeveloped: true")
    assert idx != -1
    tampered_gap = tampered_gap[:idx] + "undeveloped: false" + tampered_gap[idx + len("undeveloped: true") :]
    bad_gsn_path = tmp_path / "bad_gap.gsn.yaml"
    bad_gsn_path.write_text(tampered_gap, encoding="utf-8")

    out_bad = tmp_path / "bad_bundle.json"
    rc = certgen.main(["--gsn", str(bad_gsn_path), "--out", str(out_bad), "--check"])
    assert rc == 1
    bundle = certgen.synthesize_assurance_bundle(gsn_path=bad_gsn_path)
    assert bundle["status"] == "FAIL"
    assert any("GAP_BOUNDARY_BREACH" in e for e in bundle["errors"])

    # 2. Inject forbidden over-claim "DO-178C certified" => MUST FAIL CLOSED
    bad_claim_gsn = tmp_path / "bad_claim.gsn.yaml"
    bad_claim_gsn.write_text(
        live_gsn.replace("bounded runtime safety margins", "DO-178C certified flight approval"),
        encoding="utf-8",
    )
    bundle_claim = certgen.synthesize_assurance_bundle(gsn_path=bad_claim_gsn)
    assert bundle_claim["status"] == "FAIL"
    assert bundle_claim["forbidden_claims_audit"]["clean"] is False
    assert any("FORBIDDEN_CLAIM" in e for e in bundle_claim["errors"])
