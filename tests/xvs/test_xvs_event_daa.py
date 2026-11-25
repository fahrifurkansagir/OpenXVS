"""Verification Suite for DR-10 IMX636+EO Asymmetric DAA Fusion & FRIES Rotor Discriminator."""

from __future__ import annotations

import numpy as np
import pytest

from xvs.v2.xvs_event_daa import (
    IMX636_DEFAULT_IFOV_RAD,
    KNOTS_TO_MPS,
    AfkNotchBand,
    FriesRotorDiscriminator,
    Imx636EspController,
    Imx636Interface,
    RotorClass,
    SensorAnchorMode,
    compute_cbdr_edge_expansion_px_s,
    compute_colocated_parallax_px,
    select_asymmetric_sensor_anchor,
)


# ---------------------------------------------------------------------------
# 1. Physical Optics, CBDR Paradox & Co-Located Baseline Parallax
# ---------------------------------------------------------------------------


def test_cbdr_paradox_c172_3000m_250kt_matches_dr10_and_routes_to_eo_anchor() -> None:
    v_close_250kt = 250.0 * KNOTS_TO_MPS
    cbdr = compute_cbdr_edge_expansion_px_s(
        target_width_m=11.0,
        closing_speed_mps=v_close_250kt,
        range_m=3000.0,
        ifov_rad=IMX636_DEFAULT_IFOV_RAD,
    )
    assert cbdr.edge_expansion_px_s == pytest.approx(0.194, abs=0.002)
    assert cbdr.seconds_per_px == pytest.approx(5.15, abs=0.02)
    assert cbdr.blind_gap_10hz_m == pytest.approx(12.86, abs=0.05)
    assert cbdr.blind_gap_1khz_m == pytest.approx(0.1286, abs=0.001)

    # Fuselage (1.8 m) at 3,000 m -> ~31.49 s/px
    cbdr_fus = compute_cbdr_edge_expansion_px_s(
        target_width_m=1.8,
        closing_speed_mps=v_close_250kt,
        range_m=3000.0,
    )
    assert cbdr_fus.seconds_per_px == pytest.approx(31.49, abs=0.05)

    # Long-range low-bearing-rate CBDR intruder MUST route to Global-Shutter EO Anchor
    dec = select_asymmetric_sensor_anchor(
        range_m=3000.0,
        closing_speed_mps=v_close_250kt,
        bearing_rate_deg_s=0.05,
        target_width_m=11.0,
    )
    assert dec.mode == SensorAnchorMode.EO_ANCHOR
    assert dec.pure_rotational_homography_valid is True


def test_close_or_crossing_intruder_routes_to_evs_fast_track() -> None:
    # 1. Terminal closure (< 1,000 m) -> 1 kHz EVS Fast-Track
    dec_close = select_asymmetric_sensor_anchor(
        range_m=750.0,
        closing_speed_mps=250.0 * KNOTS_TO_MPS,
        bearing_rate_deg_s=0.10,
    )
    assert dec_close.mode == SensorAnchorMode.EVS_FAST_TRACK

    # 2. High-bearing-rate crossing target at 2,000 m (> 0.5 deg/s) -> 1 kHz EVS Fast-Track
    dec_cross = select_asymmetric_sensor_anchor(
        range_m=2000.0,
        closing_speed_mps=150.0 * KNOTS_TO_MPS,
        bearing_rate_deg_s=2.4,
    )
    assert dec_cross.mode == SensorAnchorMode.EVS_FAST_TRACK


def test_colocated_35_to_50mm_parallax_under_half_and_quarter_pixel() -> None:
    # 35 mm and 50 mm rigid co-located baselines on 12 mm IMX636 (zero -3 dB beam-splitter loss)
    for baseline_m in (0.035, 0.050):
        p_250 = compute_colocated_parallax_px(baseline_m=baseline_m, range_m=250.0)
        p_500 = compute_colocated_parallax_px(baseline_m=baseline_m, range_m=500.0)
        p_1000 = compute_colocated_parallax_px(baseline_m=baseline_m, range_m=1000.0)
        assert p_250 < 0.50
        assert p_500 < 0.25
        assert p_1000 < 0.13


# ---------------------------------------------------------------------------
# 2. IMX636 Hardware ESP Guard (EVT 3.0 Ceilings, ERC & Narrow-Notch AFK)
# ---------------------------------------------------------------------------


def test_imx636_esp_bus_ceilings_erc_formula_and_selective_afk() -> None:
    evk4 = Imx636EspController(Imx636Interface.EVK4_USB3)
    triton2 = Imx636EspController(Imx636Interface.TRITON2_2_5GIGE)
    ids_xcp = Imx636EspController(Imx636Interface.IDS_XCP_E_USB3)

    assert evk4.bus_ceiling_eps == 100_000_000
    assert triton2.bus_ceiling_eps == 138_000_000
    assert ids_xcp.bus_ceiling_eps == 200_000_000

    # Exact ERC formula: 100 Meps -> 20,000; 10 Meps -> 2,000
    assert Imx636EspController.compute_erc_cd_event_count(100_000_000) == 20_000
    assert Imx636EspController.compute_erc_cd_event_count(10_000_000) == 2_000

    # Exceeding EVK4 100 Meps bus ceiling raises ValueError in strict mode
    with pytest.raises(ValueError, match="bus ceiling"):
        evk4.configure(target_rate_eps=130_000_000, strict=True)

    # Broadband 50–500 Hz AFK MUST be rejected (erases 66.7–450 Hz propeller BPF!)
    broadband = [AfkNotchBand(50.0, 500.0, label="broadband")]
    with pytest.raises(ValueError, match="broadband AFK"):
        triton2.configure(target_rate_eps=100_000_000, afk_notches=broadband, strict=True)
    res_rej = triton2.configure(target_rate_eps=100_000_000, afk_notches=broadband, strict=False)
    assert res_rej.accepted is False

    # Selective narrow mains notches (100 Hz +/- 5 Hz, 120 Hz +/- 5 Hz) MUST be accepted
    narrow = [
        AfkNotchBand(95.0, 105.0, label="50Hz_mains_2x"),
        AfkNotchBand(115.0, 125.0, label="60Hz_mains_2x"),
    ]
    res_ok = triton2.configure(target_rate_eps=100_000_000, afk_notches=narrow, strict=True)
    assert res_ok.accepted is True
    assert res_ok.afk_enabled is True
    assert res_ok.erc_cd_event_count == 20_000


# ---------------------------------------------------------------------------
# 3. FRIES Propeller-Harmonic Spectral Discriminator (66.7–450 Hz)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_bpf_hz", [120.0, 240.0])
def test_fries_rotor_bpf_estimation_under_3_percent_error(true_bpf_hz: float) -> None:
    fs = 1000.0
    t = np.arange(512, dtype=np.float64) / fs
    rng = np.random.default_rng(2026)
    # Synthetic ROI event count modulated by blade-pass fundamental + 2nd harmonic + mild noise
    sig = (
        40.0
        + 18.0 * np.sin(2.0 * np.pi * true_bpf_hz * t)
        + 6.0 * np.sin(2.0 * np.pi * (2.0 * true_bpf_hz) * t + 0.4)
        + rng.normal(0.0, 2.5, size=t.size)
    )

    disc = FriesRotorDiscriminator()
    res = disc.classify(sig, sample_rate_hz=fs)
    assert res.classification == RotorClass.ROTORCRAFT_OR_PROP_AIRCRAFT
    rel_err = abs(res.estimated_bpf_hz - true_bpf_hz) / true_bpf_hz
    assert rel_err < 0.03
    assert res.peak_snr_db > 15.0


def test_fries_rejects_5hz_bird_flap_and_white_noise_cloud_clutter() -> None:
    fs = 1000.0
    t = np.arange(512, dtype=np.float64) / fs
    rng = np.random.default_rng(42)
    disc = FriesRotorDiscriminator()

    # 1. 5 Hz bird wing-flap + mild background noise -> NON_ROTATING_CLUTTER
    bird_sig = 25.0 + 20.0 * np.sin(2.0 * np.pi * 5.0 * t) + rng.normal(0.0, 1.0, size=t.size)
    res_bird = disc.classify(bird_sig, sample_rate_hz=fs)
    assert res_bird.classification == RotorClass.NON_ROTATING_CLUTTER

    # 2. Pure white-noise cloud edge clutter -> NON_ROTATING_CLUTTER
    cloud_sig = 30.0 + rng.normal(0.0, 8.0, size=t.size)
    res_cloud = disc.classify(cloud_sig, sample_rate_hz=fs)
    assert res_cloud.classification == RotorClass.NON_ROTATING_CLUTTER

    # 3. Sub-1000 Hz sample rate rejected cleanly (violates Nyquist for 450 Hz BPF)
    with pytest.raises(ValueError, match=">= 1000.0"):
        disc.classify(bird_sig, sample_rate_hz=500.0)
