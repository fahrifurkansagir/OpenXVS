# Copyright 2026 OpenXVS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for DR-18 (`xvs_forcedland`) and DR-17 (`xvs_fleet`)."""

from __future__ import annotations

import math

import pytest

from xvs.v2.xvs_fleet import (
    AlertSeverity,
    FleetPolicyError,
    HcdAlert,
    HcdTriageQueue,
    InterlockState,
    StrategicTacticalInterlock,
    compute_fleet_datalink_plan,
    compute_tactical_evasion_buffer_m,
    parse_operator_fleet_ratio,
)
from xvs.v2.xvs_forcedland import (
    G_STANDARD_MPS2,
    POLAR_PRESETS,
    AircraftGlideState,
    FlzCandidate,
    ForcedLandPolicyError,
    ForcedLandVlmGate,
    VlmRankMsg,
    abl_wind_profile,
    banked_turn_schedule,
    best_glide_params,
    evaluate_flz_candidates,
    maccready_speed_to_fly,
    validate_leo_omm_and_nis,
)


# ---------------------------------------------------------------------------
# DR-18: Aerodynamic Polar, Banked Turn Schedule, MacCready & ABL Wind Shear
# ---------------------------------------------------------------------------


def test_dr18_polar_presets_and_banked_turn_schedule() -> None:
    expected_ld = {"light_3kg": 10.0, "nominal_12kg": 14.0, "heavy_25kg": 18.0}
    for name, ld_target in expected_ld.items():
        polar = POLAR_PRESETS[name]
        bg = best_glide_params(polar)
        assert math.isclose(bg.ld_max, ld_target, rel_tol=1e-6)
        assert math.isclose(polar.sink_rate_mps(bg.v_bg_mps), bg.v_sink_bg_mps, rel_tol=1e-6)

    nom = POLAR_PRESETS["nominal_12kg"]
    bg_nom = best_glide_params(nom)
    assert math.isclose(bg_nom.v_bg_mps, 19.7, abs_tol=1e-3)

    phi_30 = math.radians(30.0)
    sched_30 = banked_turn_schedule(nom, phi_30)
    sec_30 = 1.0 / math.cos(phi_30)
    assert math.isclose(sched_30.v_opt_mps, bg_nom.v_bg_mps * math.sqrt(sec_30), rel_tol=1e-9)
    assert math.isclose(sched_30.v_sink_mps, bg_nom.v_sink_bg_mps * (sec_30**1.5), rel_tol=1e-9)
    assert math.isclose(
        sched_30.turn_radius_m,
        (bg_nom.v_bg_mps**2) / (G_STANDARD_MPS2 * math.sin(phi_30)),
        rel_tol=1e-9,
    )
    assert math.isclose(sched_30.delta_h_360_m, 40.98, abs_tol=0.15)
    assert math.isclose(sched_30.delta_h_180_m, 0.5 * sched_30.delta_h_360_m, rel_tol=1e-9)

    # Nominal bank cap <= 35 deg must reject steeper bank angles by default
    banked_turn_schedule(nom, math.radians(35.0), enforce_nominal_cap=True)
    with pytest.raises(ValueError, match="nominal DR-18 cap"):
        banked_turn_schedule(nom, math.radians(36.0), enforce_nominal_cap=True)

    # Uncapped mode allows analytical 45-deg check (DR-18: delta_h_360 ~ 35.5 m)
    sched_45 = banked_turn_schedule(nom, math.radians(45.0), enforce_nominal_cap=False)
    assert math.isclose(sched_45.delta_h_360_m, 35.5, abs_tol=0.15)


def test_dr18_maccready_speed_to_fly_and_abl_wind_profile() -> None:
    nom = POLAR_PRESETS["nominal_12kg"]
    bg = best_glide_params(nom)

    still = maccready_speed_to_fly(nom, w_par=0.0, w_perp=0.0, w_down=0.0)
    assert math.isclose(still.v_opt_mps, bg.v_bg_mps, abs_tol=1e-2)
    assert math.isclose(still.effective_glide_ratio, bg.ld_max, rel_tol=1e-3)

    headwind = maccready_speed_to_fly(nom, w_par=-6.0, w_perp=0.0, w_down=0.0)
    downdraft = maccready_speed_to_fly(nom, w_par=0.0, w_perp=0.0, w_down=1.5)
    tailwind = maccready_speed_to_fly(nom, w_par=6.0, w_perp=0.0, w_down=0.0)
    crosswind = maccready_speed_to_fly(nom, w_par=0.0, w_perp=5.0, w_down=0.0)

    assert headwind.v_opt_mps > still.v_opt_mps
    assert downdraft.v_opt_mps > still.v_opt_mps
    assert tailwind.v_opt_mps < still.v_opt_mps
    assert tailwind.v_opt_mps >= 1.15 * nom.v_stall_mps - 1e-6
    assert crosswind.crab_angle_rad > 0.0
    assert crosswind.v_ground_mps < crosswind.v_opt_mps

    with pytest.raises(ValueError, match="Crosswind"):
        maccready_speed_to_fly(nom, w_par=0.0, w_perp=nom.v_max_mps + 2.0, w_down=0.0)

    # DR-18 ABL power-law shear numbers: 12 m/s at 300 m -> 7.45 m/s (0.14) and 5.68 m/s (0.22) at 10 m
    assert math.isclose(abl_wind_profile(12.0, 10.0, 300.0, alpha=0.14), 7.45, abs_tol=0.02)
    assert math.isclose(abl_wind_profile(12.0, 10.0, 300.0, alpha=0.22), 5.68, abs_tol=0.02)


# ---------------------------------------------------------------------------
# DR-18: 3D Trochoidal Reachability + 5-Gate 3DEP Filter + Into-Wind Final
# ---------------------------------------------------------------------------


def _make_flz(
    flz_id: int,
    *,
    north_m: float = 900.0,
    east_m: float = 0.0,
    down_m: float = 0.0,
    final_approach_hdg_rad: float = 0.0,
    slope_deg: float = 3.0,
    cross_slope_deg: float = 2.0,
    rms_roughness_m: float = 0.12,
    ndsm_m: float = 0.20,
    exclusion_overlap_frac: float = 0.0,
    lateral_setback_m: float = 40.0,
    approach_corridor_clear_10_to_1: bool = True,
    strip_length_m: float = 200.0,
    strip_width_m: float = 25.0,
) -> FlzCandidate:
    return FlzCandidate(
        flz_id=flz_id,
        north_m=north_m,
        east_m=east_m,
        down_m=down_m,
        final_approach_hdg_rad=final_approach_hdg_rad,
        slope_deg=slope_deg,
        cross_slope_deg=cross_slope_deg,
        rms_roughness_m=rms_roughness_m,
        ndsm_m=ndsm_m,
        exclusion_overlap_frac=exclusion_overlap_frac,
        lateral_setback_m=lateral_setback_m,
        approach_corridor_clear_10_to_1=approach_corridor_clear_10_to_1,
        strip_length_m=strip_length_m,
        strip_width_m=strip_width_m,
    )


def test_dr18_evaluate_flz_candidates_5gate_3dep_and_into_wind_alignment() -> None:
    nom = POLAR_PRESETS["nominal_12kg"]
    # Wind blowing FROM North toward South: w_n = -5.0 m/s -> into-wind heading is North (0.0 rad)
    state = AircraftGlideState(
        north_m=0.0,
        east_m=0.0,
        down_m=-300.0,
        heading_rad=0.0,
        wind_ned_mps=(-5.0, 0.0, 0.0),
    )

    candidates = [
        _make_flz(0, north_m=700.0, final_approach_hdg_rad=0.0),  # Best margin, valid into-wind
        _make_flz(1, north_m=1100.0, final_approach_hdg_rad=math.radians(30.0)),  # Valid, lower margin
        _make_flz(2, north_m=700.0, slope_deg=7.2),  # Fails nominal 6 deg, passes emergency 8 deg
        _make_flz(3, north_m=700.0, rms_roughness_m=0.35),  # Gate 2 fail
        _make_flz(4, north_m=700.0, ndsm_m=0.65),  # Gate 3 fail
        _make_flz(5, north_m=700.0, exclusion_overlap_frac=0.01),  # Gate 4 fail
        _make_flz(6, north_m=700.0, approach_corridor_clear_10_to_1=False),  # Gate 5 fail
        _make_flz(7, north_m=700.0, final_approach_hdg_rad=math.radians(180.0)),  # Downwind fail (>45 deg)
        _make_flz(8, north_m=4500.0, final_approach_hdg_rad=0.0),  # Out of glide energy reserve
    ]

    res_nom = evaluate_flz_candidates(nom, state, candidates, emergency_tier=False)
    assert res_nom.reachable_flz_mask == ((1 << 0) | (1 << 1))
    assert res_nom.geom_ranked_ids == (0, 1)
    assert "GATE1_SLOPE_EXCEEDED" in res_nom.rejection_reasons[2]
    assert "GATE2_ROUGHNESS_EXCEEDED" in res_nom.rejection_reasons[3]
    assert "GATE3_NDSM_EXCEEDED" in res_nom.rejection_reasons[4]
    assert "GATE4_EXCLUSION_OR_SETBACK" in res_nom.rejection_reasons[5]
    assert "GATE5_CORRIDOR_OR_STRIP_SIZE" in res_nom.rejection_reasons[6]
    assert "GATE_INTO_WIND_ALIGNMENT" in res_nom.rejection_reasons[7]
    assert "UNREACHABLE_ENERGY_MARGIN" in res_nom.rejection_reasons[8]

    # Emergency tier raises slope gate from 6.0 deg to 8.0 deg -> FLZ 2 now passes
    res_emg = evaluate_flz_candidates(nom, state, candidates, emergency_tier=True)
    assert (res_emg.reachable_flz_mask & (1 << 2)) != 0

    # Calm wind (< 3.0 m/s) waives into-wind alignment restriction
    calm_state = AircraftGlideState(
        north_m=0.0,
        east_m=0.0,
        down_m=-300.0,
        heading_rad=0.0,
        wind_ned_mps=(-1.5, 0.0, 0.0),
    )
    res_calm = evaluate_flz_candidates(nom, calm_state, [_make_flz(7, north_m=700.0, final_approach_hdg_rad=math.pi)])
    assert res_calm.reachable_flz_mask == (1 << 7)


# ---------------------------------------------------------------------------
# DR-18: 6-Stage ForcedLandVlmGate (CRC, Watchdog, ODD, Bitmask, Anti-Veto, Commit Latch)
# ---------------------------------------------------------------------------


def test_dr18_forcedland_vlm_gate_all_six_stages() -> None:
    nom = POLAR_PRESETS["nominal_12kg"]
    gate = ForcedLandVlmGate(nom, watchdog_ms=500)
    assert math.isclose(gate.h_commit_m, 70.49, abs_tol=0.2)

    reachable_mask = (1 << 1) | (1 << 3) | (1 << 5)
    geom_ranked = (1, 3, 5)

    # Valid VLM message prefers FLZ 3 over geometric #1 (wire round-trip verified)
    good_msg = VlmRankMsg.build((3, 5, 1, 0xFF), hazard_veto_mask=0, timestamp_ms=10_000)
    d_ok = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_200,
        agl_m=180.0,
        optical_odd_valid=True,
        vlm_msg=good_msg.to_bytes(),
    )
    assert d_ok.selected_flz_id == 3
    assert d_ok.source == "VLM_ACCEPTED"
    assert not d_ok.latched

    # Stage 1A: Corrupted CRC32 -> instant geometric fallback
    bad_crc = VlmRankMsg(
        ranked_ids=(3, 5, 1, 0xFF),
        hazard_veto_mask=0,
        timestamp_ms=10_000,
        crc32=good_msg.crc32 ^ 0x1,
    )
    d_crc = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_100,
        agl_m=180.0,
        optical_odd_valid=True,
        vlm_msg=bad_crc,
    )
    assert d_crc.selected_flz_id == 1
    assert d_crc.reason == "STAGE1_CRC_OR_SCHEMA_REJECT"

    # Stage 1B: Duplicate FLZ IDs or float injection -> rejected
    dup_msg = VlmRankMsg.build((3, 3, 1, 0xFF), hazard_veto_mask=0, timestamp_ms=10_000)
    d_dup = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_100,
        agl_m=180.0,
        optical_odd_valid=True,
        vlm_msg=dup_msg,
    )
    assert d_dup.selected_flz_id == 1
    assert d_dup.reason == "STAGE1_CRC_OR_SCHEMA_REJECT"

    # Stage 2: Stale watchdog (> 500 ms) -> instant geometric fallback
    d_stale = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_501,
        agl_m=180.0,
        optical_odd_valid=True,
        vlm_msg=good_msg,
    )
    assert d_stale.selected_flz_id == 1
    assert d_stale.reason == "STAGE2_WATCHDOG_TIMEOUT"

    # Stage 3: Optical-ODD invalid -> abstain & geometric fallback
    d_odd = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_100,
        agl_m=180.0,
        optical_odd_valid=False,
        vlm_msg=good_msg,
    )
    assert d_odd.selected_flz_id == 1
    assert d_odd.reason == "STAGE3_OPTICAL_ODD_ABSTAIN"

    # Stage 4: Bitmask containment rejects hallucinated unreachable FLZ ID (bit 9 not in mask)
    halluc_msg = VlmRankMsg.build((9, 3, 1, 0xFF), hazard_veto_mask=0, timestamp_ms=10_000)
    d_mask = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_100,
        agl_m=180.0,
        optical_odd_valid=True,
        vlm_msg=halluc_msg,
    )
    assert d_mask.selected_flz_id == 1
    assert d_mask.reason == "STAGE4_BITMASK_CONTAINMENT_REJECT"

    # Stage 5A: Anti-all-veto deadlock guard clears veto mask if ALL reachable FLZs are vetoed
    all_veto_msg = VlmRankMsg.build((3, 5, 1, 0xFF), hazard_veto_mask=reachable_mask, timestamp_ms=10_000)
    d_veto = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_100,
        agl_m=180.0,
        optical_odd_valid=True,
        vlm_msg=all_veto_msg,
    )
    assert d_veto.selected_flz_id == 1
    assert d_veto.veto_cleared_deadlock is True
    assert d_veto.reason == "STAGE5_ANTI_ALL_VETO_DEADLOCK_GUARD"

    # Stage 5B: Non-chattering commit-altitude latch once agl_m <= h_commit_m
    d_latch = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_100,
        agl_m=gate.h_commit_m - 1.0,
        optical_odd_valid=True,
        vlm_msg=good_msg,
    )
    assert d_latch.selected_flz_id == 3
    assert d_latch.latched is True

    # Subsequent conflicting VLM message MUST NOT chatter away from latched FLZ 3
    flip_msg = VlmRankMsg.build((5, 1, 3, 0xFF), hazard_veto_mask=(1 << 3), timestamp_ms=10_150)
    d_locked = gate.select_flz(
        reachable_flz_mask=reachable_mask,
        geom_ranked_ids=geom_ranked,
        now_ms=10_200,
        agl_m=50.0,
        optical_odd_valid=True,
        vlm_msg=flip_msg,
    )
    assert d_locked.selected_flz_id == 3
    assert d_locked.source == "COMMIT_LATCH"
    assert d_locked.latched is True


# ---------------------------------------------------------------------------
# DR-18: Passive LEO-Doppler SoOP OMM & Chi-Squared NIS Guard
# ---------------------------------------------------------------------------


def test_dr18_leo_omm_post_rollover_nis_and_policy_locks() -> None:
    # Post-July-2026 6-digit NORAD_CAT_ID >= 100000 accepted cleanly
    res_nom = validate_leo_omm_and_nis(
        nav_source="leo_starlink_ku",
        norad_cat_id=104821,
        tau_omm_h=3.5,
        nis_leo_dof1=2.1,
        nis_pitot_dof3=4.0,
    )
    assert res_nom.post_rollover_norad_id is True
    assert not res_nom.leo_inhibited
    assert math.isclose(res_nom.doppler_sigma_mps, 0.35)
    assert res_nom.leo_nis_action == "ACCEPT"
    assert math.isclose(res_nom.leo_weight, 1.0)
    assert not res_nom.pitot_fault
    assert res_nom.airdata_optical_synthesis_blocked is True

    # Huber downweight band (6.63 < NIS <= 10.83) and hard reject (> 10.83)
    res_huber = validate_leo_omm_and_nis(
        nav_source="leo_doppler_soop",
        norad_cat_id=250000,
        tau_omm_h=12.0,
        nis_leo_dof1=9.0,
        nis_pitot_dof3=5.0,
    )
    assert math.isclose(res_huber.doppler_sigma_mps, 0.80)
    assert res_huber.leo_nis_action == "HUBER_DOWNWEIGHT"
    assert math.isclose(res_huber.leo_weight, math.sqrt(6.63 / 9.0), rel_tol=1e-9)

    res_rej = validate_leo_omm_and_nis(
        nav_source="leo_iridium_l",
        norad_cat_id=100000,
        tau_omm_h=2.0,
        nis_leo_dof1=12.5,
        nis_pitot_dof3=18.4,  # Exceeds DOF=3 Pitot threshold 16.27
    )
    assert res_rej.leo_nis_action == "HARD_REJECT"
    assert res_rej.leo_weight == 0.0
    assert res_rej.pitot_fault is True
    assert res_rej.airdata_optical_synthesis_blocked is True

    # Stale OMM (> 24 h) without STAN -> hard inhibited; with STAN -> active at 0.20 m/s
    res_stale = validate_leo_omm_and_nis(
        nav_source="leo_doppler_soop",
        norad_cat_id=123456,
        tau_omm_h=24.1,
        nis_leo_dof1=1.0,
        nis_pitot_dof3=2.0,
        stan_active=False,
    )
    assert res_stale.leo_inhibited is True
    assert math.isinf(res_stale.doppler_sigma_mps)
    assert res_stale.leo_weight == 0.0

    res_stan = validate_leo_omm_and_nis(
        nav_source="leo_doppler_soop",
        norad_cat_id=123456,
        tau_omm_h=36.0,
        nis_leo_dof1=1.0,
        nis_pitot_dof3=2.0,
        stan_active=True,
    )
    assert not res_stan.leo_inhibited
    assert math.isclose(res_stan.doppler_sigma_mps, 0.20)

    # Hard fail-closed policy rejection of MagNav and active DME
    for banned in ("MagNav", "magnetic_anomaly", "active_dme", "DME_TX"):
        with pytest.raises(ForcedLandPolicyError):
            validate_leo_omm_and_nis(
                nav_source=banned,
                norad_cat_id=100001,
                tau_omm_h=1.0,
                nis_leo_dof1=1.0,
                nis_pitot_dof3=1.0,
            )


# ---------------------------------------------------------------------------
# DR-17: Adaptive MeshDAA RF Governor + Dynamic ECBF-QP Barrier Expansion
# ---------------------------------------------------------------------------


def test_dr17_fleet_datalink_plan_primary_failover_and_banned_rf() -> None:
    # Primary Dual-SIM 5G/LTE + MEC Zenoh: 10 / 25 / 50 AC nominal & < 2 km burst
    p10_nom = compute_fleet_datalink_plan(fleet_size=10, min_peer_distance_m=3000.0)
    assert p10_nom.nominal_rate_hz == 10.0
    assert p10_nom.active_publish_rate_hz == 10.0
    assert p10_nom.geo_culled_peers_5km == 9

    p10_burst = compute_fleet_datalink_plan(fleet_size=10, min_peer_distance_m=1500.0, r_min_m=15.0, v_max_mps=50.0)
    assert p10_burst.active_daa_burst is True
    assert p10_burst.active_publish_rate_hz == 20.0
    assert math.isclose(p10_burst.ecbf_effective_r_min_m, 15.0 + 50.0 * 0.05, rel_tol=1e-9)

    p25_nom = compute_fleet_datalink_plan(fleet_size=25, min_peer_distance_m=4000.0)
    assert p25_nom.nominal_rate_hz == 5.0
    assert p25_nom.geo_culled_peers_5km == 12  # Strictly capped at k <= 12 within 5 km

    p50_burst = compute_fleet_datalink_plan(fleet_size=50, min_peer_distance_m=1200.0)
    assert p50_burst.nominal_rate_hz == 5.0
    assert p50_burst.active_publish_rate_hz == 10.0
    assert p50_burst.geo_culled_peers_5km == 12

    # SX1262 GFSK GPS-TDMA lifeboat: US 200k vs EU g3 100k + dynamic ECBF barrier expansion
    us_25 = compute_fleet_datalink_plan(
        fleet_size=25,
        region="US",
        transport="sx1262_gfsk_gps_tdma",
        r_min_m=15.0,
        v_max_mps=50.0,
    )
    assert us_25.phy_bitrate_kbps == 200.0
    assert us_25.active_publish_rate_hz == 5.0
    assert math.isclose(us_25.ecbf_effective_r_min_m, 25.0, rel_tol=1e-9)

    eu_25 = compute_fleet_datalink_plan(
        fleet_size=25,
        region="EU",
        transport="sx1262_gfsk_gps_tdma",
        r_min_m=15.0,
        v_max_mps=50.0,
    )
    assert eu_25.phy_bitrate_kbps == 100.0
    assert eu_25.duty_cycle_cap_pct == 10.0
    assert eu_25.active_publish_rate_hz == 2.0
    assert math.isclose(eu_25.ecbf_barrier_expansion_m, 25.0, rel_tol=1e-9)
    assert math.isclose(eu_25.ecbf_effective_r_min_m, 40.0, rel_tol=1e-9)

    # EU g3 100k strictly forbids > 25 AC in a single RF sub-cluster
    with pytest.raises(FleetPolicyError, match="<= 25 AC"):
        compute_fleet_datalink_plan(fleet_size=26, region="EU", transport="sx1262_gfsk_gps_tdma")

    # Fail-closed rejection of ALL DR-17 banned RF links
    banned_links = (
        "lora_css_sf7",
        "lora_sf10",
        "ExpressLRS",
        "elrs_900",
        "Meshtastic",
        "starlink_roam_airborne",
        "direct_to_cell",
        "starlink_d2c",
    )
    for bad_link in banned_links:
        with pytest.raises(FleetPolicyError):
            compute_fleet_datalink_plan(fleet_size=10, transport=bad_link)


# ---------------------------------------------------------------------------
# DR-17: ASTM F3548-21 <-> ASTM F3269-21 StrategicTacticalInterlock
# ---------------------------------------------------------------------------


def test_dr17_strategic_tactical_interlock_hysteresis_and_contingent() -> None:
    buf_m = compute_tactical_evasion_buffer_m(
        lateral_daa_envelope_m=60.0,
        r_min_m=15.0,
        v_max_mps=40.0,
        telemetry_age_s=0.25,
    )
    assert math.isclose(buf_m, 85.0, rel_tol=1e-9)

    interlock = StrategicTacticalInterlock(
        d_dwc_m=150.0,
        d_hyst_m=40.0,  # spatial release > 150 + 1.25 * 40 = 200 m
        h_reentry=1.0,
        dwell_required_s=8.0,
        contingent_timeout_s=30.0,
    )
    assert math.isclose(interlock.spatial_release_distance_m, 200.0)

    # Nominal accepts strategic waypoints
    r0 = interlock.step(
        dt_s=1.0,
        tactical_override_active=False,
        min_intruder_dist_m=500.0,
        min_ecbf_h=5.0,
        incoming_strategic_waypoint=True,
    )
    assert r0.state == InterlockState.NOMINAL_STRATEGIC
    assert not r0.strategic_freeze
    assert r0.strategic_cmd_accepted is True

    # Tactical override freezes strategic channel & omits OVN key on NONCONFORMING broadcast
    r1 = interlock.step(
        dt_s=1.0,
        tactical_override_active=True,
        min_intruder_dist_m=120.0,
        min_ecbf_h=0.2,
        incoming_strategic_waypoint=True,
    )
    assert r1.state == InterlockState.TACTICAL_OVERRIDE_FREEZE
    assert r1.strategic_freeze is True
    assert r1.strategic_cmd_accepted is False
    assert r1.omit_key_ovn is True
    assert r1.dss_operational_intent_state == "NONCONFORMING"

    # Tactical clears, but distance is inside spatial hysteresis (190 m <= 200 m) -> MUST stay frozen
    r2 = interlock.step(
        dt_s=5.0,
        tactical_override_active=False,
        min_intruder_dist_m=190.0,
        min_ecbf_h=2.0,
        incoming_strategic_waypoint=True,
    )
    assert r2.strategic_freeze is True
    assert r2.clear_dwell_s == 0.0

    # Spatially clear (> 200 m, h > 1.0) for 5 s (< 8 s dwell) -> HYSTERESIS_RECOVERY, still frozen
    r3 = interlock.step(
        dt_s=5.0,
        tactical_override_active=False,
        min_intruder_dist_m=240.0,
        min_ecbf_h=2.5,
        incoming_strategic_waypoint=True,
    )
    assert r3.state == InterlockState.HYSTERESIS_RECOVERY
    assert r3.strategic_freeze is True
    assert math.isclose(r3.clear_dwell_s, 5.0)

    # Additional 3.0 s clear -> reaches 8.0 s dwell -> unlocks strategic replanning!
    r4 = interlock.step(
        dt_s=3.0,
        tactical_override_active=False,
        min_intruder_dist_m=260.0,
        min_ecbf_h=3.0,
        incoming_strategic_waypoint=True,
    )
    assert r4.state == InterlockState.NOMINAL_STRATEGIC
    assert r4.strategic_freeze is False
    assert r4.replan_unlocked is True
    assert r4.strategic_cmd_accepted is True
    assert r4.omit_key_ovn is False

    # Prolonged nonconforming >= 30 s escalates irreversibly to CONTINGENT
    r_trig = interlock.step(
        dt_s=30.0,
        tactical_override_active=True,
        min_intruder_dist_m=100.0,
        min_ecbf_h=0.1,
    )
    assert r_trig.state == InterlockState.CONTINGENT
    assert r_trig.dss_operational_intent_state == "CONTINGENT"
    r_after = interlock.step(
        dt_s=10.0,
        tactical_override_active=False,
        min_intruder_dist_m=1000.0,
        min_ecbf_h=10.0,
        incoming_strategic_waypoint=True,
    )
    assert r_after.state == InterlockState.CONTINGENT
    assert r_after.strategic_freeze is True


# ---------------------------------------------------------------------------
# DR-17: 1-to-Many EASA HCD / NASA Ames Triage Queue
# ---------------------------------------------------------------------------


def test_dr17_hcd_triage_queue_ratio_cap_preemption_coalescing_and_offload() -> None:
    # Hard rejection of "1:100" and any fleet_size > 20
    with pytest.raises(FleetPolicyError):
        parse_operator_fleet_ratio("1:100")
    with pytest.raises(FleetPolicyError):
        HcdTriageQueue(fleet_size="1:100")
    with pytest.raises(FleetPolicyError):
        HcdTriageQueue(fleet_size=21)
    with pytest.raises(FleetPolicyError):
        HcdTriageQueue(fleet_size=10, max_active_focus_cards=3)

    queue = HcdTriageQueue(fleet_size="1:12")
    alerts = [
        HcdAlert("a-p2", "AC-09", AlertSeverity.P2_ADVISORY, 1.0, "VLM advisory"),
        HcdAlert("a-p1-1", "AC-03", AlertSeverity.P1_HIGH, 2.0, "GNSS degrade", region_cluster_id="WEST_SECTOR"),
        HcdAlert("a-p1-2", "AC-04", AlertSeverity.P1_HIGH, 2.5, "GNSS degrade", region_cluster_id="WEST_SECTOR"),
        HcdAlert("a-p1-3", "AC-05", AlertSeverity.P1_HIGH, 3.0, "Battery low"),
        HcdAlert("a-p0-1", "AC-01", AlertSeverity.P0_EMERGENCY, 4.0, "Forced landing commit"),
        HcdAlert("a-p0-2", "AC-02", AlertSeverity.P0_EMERGENCY, 4.5, "Motor loss"),
    ]
    snap = queue.triage(alerts)

    # Strictly <= 2 active Focus Cards; both P0 emergencies preempt P1 cards
    assert len(snap.active_focus_cards) == 2
    assert [c.card_id for c in snap.active_focus_cards] == ["a-p0-1", "a-p0-2"]
    assert all(c.severity == AlertSeverity.P0_EMERGENCY for c in snap.active_focus_cards)

    # Overflow P1 cards (coalesced WEST_SECTOR + AC-05) trigger NASA Ames peer offload signaling
    assert snap.offload_required is True
    assert len(snap.preempted_overflow_cards) == 2
    assert any(c.coalesced and c.aircraft_ids == ("AC-03", "AC-04") for c in snap.preempted_overflow_cards)
    assert set(snap.offload_recommended_ac_ids) == {"AC-03", "AC-04", "AC-05"}

    # P2 alert auto-deferred & chime suppressed while P0/P1 Focus Cards active
    assert [a.alert_id for a in snap.deferred_alerts] == ["a-p2"]
    assert snap.p2_aural_chime_suppressed is True
