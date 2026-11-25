// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0
//! Machine-checked `#![no_std]` Kani CBMC formal proofs for `xvs-multiship-coord` (`Kani #19..#26`).
//!
//! Proves across symbolic inputs:
//!   - Kani #19 (`kani_f1_optical_only_non_reversing_far91113`): Tier B optical-only FAR 91.113(e)
//!     right-turn precedence never reverses sign across arbitrary tick sequences, eliminating
//!     symmetric 12-tick cancellation.
//!   - Kani #20 (`kani_f1_cooperative_peer_stagger_asymmetry`): Tier A cooperative staggering over
//!     any two distinct aircraft IDs (`own_id != peer_id`) with fresh `PeerCoordHint` enforces
//!     deterministic asymmetric phase/sign separation (zero mutual cancellation) and bounded bank
//!     angles `|cmd_bank_rad| <= EVADE_BANK_RAD` with zero NaN/Inf.
//!   - Kani #21 (`kani_f4_mesh_staleness_replay_and_sticky_ghost`): `StatefulMeshDaaGuard` rejects
//!     any observation with `age > 200 ms`, future skew `> 50 ms`, or non-monotonic replay timestamp,
//!     and preserves sticky triple-ray ghost suppression across transient `3 -> 2` drops within the
//!     `500 ms` hysteresis window.
//!   - Kani #22 (`kani_f3_ring0_p0_zeroes_all_avoidance`): `Ring0AuthorityGate` strictly outputs
//!     `cmd_bank_rad == 0.0` whenever `rta_inhibit_avoidance_turns == true` OR
//!     `hal_out_mode_wire == 0`, regardless of arbitrary `ActiveRange` command (including
//!     `ImmediateEvade` or arbitrary IEEE-754 `NaN`/`Inf` inputs).
//!   - Kani #23 (`kani_clock_sync_overflow_and_bounds_free`): `PeerClockOffsetEstimator` is
//!     strictly panic-free and overflow-free over arbitrary `u64` timestamps, rejects invalid/high
//!     RTT fail-closed, verifies the 5-element median sorting network, and emits bounded offsets.
//!   - Kani #24 (`kani_clock_sync_gnss_denied_drift_no_false_reject`): Arbitrary bounded TCXO
//!     drift (`[-500 ms, +500 ms]`) + small RF jitter + single multipath outlier calibrates via
//!     median-of-5, accepts fresh monotonic peer packets with 0 false rejections, and still
//!     rejects genuine `> 200 ms` stale and replay attacks.
//!   - Kani #25 (`kani_clock_sync_uncalibrated_forces_tier_b_fallback`): Uncalibrated, excessive-RTT,
//!     or holdover-expired peer during GNSS-denied mode (and `> 500 ms` 3-ship dropout) NEVER admits
//!     unguarded 2-ship coplanar ghosts and strictly asserts `fallback_to_tier_b_optical == true`.
//!   - Kani #26 (`kani_clock_sync_slew_preserves_monotonic_replay_guard`): Bounded offset slewing
//!     can NEVER cause a previously rejected old replay timestamp to be re-accepted, while fresh
//!     monotonic peer packets continue to be accepted cleanly.

#![cfg(kani)]

use xvs_active_range::sc3bf_fim::select_active_swerve_from_evals;
use xvs_active_range::{
    ActiveRangeMode, SwerveDecision, EVADE_BANK_RAD, NUM_PHI_CANDS, PHI_CAND_RAD,
};
use xvs_math_core::mesh_daa::{FusedMeshTrack, MeshFusionResult, MAX_FUSED_TRACKS};

use crate::clock_sync::{
    sort5_i64, sort5_u64, translate_ts_with_estimate, validate_two_way_exchange,
    ClockSyncRejectReason, PeerClockOffsetEstimator, PeerSyncSlot, TwoWaySyncExchange,
    MAX_ABS_CLOCK_OFFSET_US, MAX_OFFSET_UNCERT_US, MAX_SLEW_STEP_US, MAX_SYNC_RTT_US,
    MIN_SYNC_SAMPLES, SLEW_RATE_SHIFT,
};
use crate::mesh_guard::{
    empty_bearing_track, evaluate_obs_clock_synced, ObsClockSyncVerdict, StatefulMeshDaaGuard,
    MAX_FUTURE_SKEW_US, MAX_MESH_AGE_US, TRIPLE_RAY_HYSTERESIS_US,
};
use crate::swerve_coord::{
    apply_coordination_and_authority, compute_preferred_swerve_sign, tier_a_own_active_slot,
    PeerCoordHint, Ring0AuthorityGate, SwerveCoordRole, HEAD_ON_ALPHA_BAND_RAD, PEER_LINK_FRESH_US,
    STAGGER_SLOT_TICKS,
};

/// Kani #19 (DR-22 Finding F1 Tier B): Optical-only FAR 91.113(e) right-turn precedence NEVER
/// reverses the preferred swerve sign across arbitrary `u32` tick sequences and eliminates
/// opposite-sign reciprocal cancellation between two independently phased head-on aircraft.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::unwind(20)]
fn kani_f1_optical_only_non_reversing_far91113() {
    // Part A: Across ALL 2^32 tick values and arbitrary IEEE-754 head-on bearings inside
    // (-HEAD_ON_ALPHA_BAND_RAD, +HEAD_ON_ALPHA_BAND_RAD), `compute_preferred_swerve_sign`
    // is strictly +1.0 and NEVER reverses across any 16-tick trajectory spanning the 12-tick period.
    let alpha1 = f64::from_bits(kani::any::<u64>());
    let alpha2 = f64::from_bits(kani::any::<u64>());
    kani::assume(alpha1.is_finite() && libm::fabs(alpha1) < HEAD_ON_ALPHA_BAND_RAD);
    kani::assume(alpha2.is_finite() && libm::fabs(alpha2) < HEAD_ON_ALPHA_BAND_RAD);

    let swerve_ticks_1: u32 = kani::any();
    let swerve_ticks_2: u32 = kani::any();

    let pref1 = compute_preferred_swerve_sign(alpha1, swerve_ticks_1);
    let pref2 = compute_preferred_swerve_sign(alpha2, swerve_ticks_2);
    assert!(pref1 == 1.0);
    assert!(pref2 == 1.0);

    // Explicitly prove invariance across a 16-tick sequence starting from any `u32` phase offset
    let mut step = 0u32;
    while step < 16 {
        let t = swerve_ticks_1.wrapping_add(step);
        assert!(compute_preferred_swerve_sign(alpha1, t) == 1.0);
        step += 1;
    }

    // Part B: End-to-end Tier B (peer_hint = None) decision proof through frozen
    // `select_active_swerve_from_evals` + `apply_coordination_and_authority` on symmetric head-on
    // candidate evaluations with arbitrary independent ship phase offsets (swerve_ticks_1, swerve_ticks_2).
    let mut cand_margins = [0.0f64; NUM_PHI_CANDS];
    let mut cand_all_safe = [false; NUM_PHI_CANDS];
    let mut cand_fov_safe = [false; NUM_PHI_CANDS];
    let mut cand_fim = [0.0f64; NUM_PHI_CANDS];

    // Straight flight k = 4 (phi = 0): zero FIM gain, symmetric safety/FOV
    cand_margins[4] = 64.0;
    cand_all_safe[4] = true;
    cand_fov_safe[4] = true;
    cand_fim[4] = 0.0;

    // Symmetric left/right pairs (k vs 8-k) for k in 0..4 (|phi| in {12, 9, 6, 3} deg)
    let mut any_nonzero_safe = false;
    let mut k = 0usize;
    while k < 4 {
        let mirror = (NUM_PHI_CANDS - 1) - k;
        let pair_safe: bool = kani::any();
        let pair_fov: bool = kani::any();
        let hi_fim: bool = kani::any();
        let m_val = if pair_safe { 64.0 } else { -25.0 };
        let f_val = if hi_fim { 16.0 } else { 4.0 };

        cand_margins[k] = m_val;
        cand_margins[mirror] = m_val;
        cand_all_safe[k] = pair_safe;
        cand_all_safe[mirror] = pair_safe;
        cand_fov_safe[k] = pair_fov;
        cand_fov_safe[mirror] = pair_fov;
        cand_fim[k] = f_val;
        cand_fim[mirror] = f_val;

        if pair_safe && pair_fov {
            any_nonzero_safe = true;
        }
        k += 1;
    }

    // Concrete head-on bearing choice inside (-0.07, +0.07) to keep CBMC algebraic evaluation fast
    let head_sel: u8 = kani::any();
    kani::assume(head_sel < 3);
    let head_alpha = match head_sel {
        0 => -0.03,
        1 => 0.0,
        _ => 0.03,
    };
    let head_pref_1 = compute_preferred_swerve_sign(head_alpha, swerve_ticks_1);
    let head_pref_2 = compute_preferred_swerve_sign(head_alpha, swerve_ticks_2);
    assert!(head_pref_1 == 1.0 && head_pref_2 == 1.0);

    let mut dec1 = select_active_swerve_from_evals(
        1000.0,
        400.0,
        500.0,
        -0.08,
        head_alpha,
        100.0,
        45.0,
        14.5,
        true,
        false,
        head_pref_1,
        &cand_margins,
        &cand_all_safe,
        &cand_fov_safe,
        &cand_fim,
    );
    let mut dec2 = select_active_swerve_from_evals(
        1000.0,
        400.0,
        500.0,
        -0.08,
        head_alpha,
        100.0,
        45.0,
        14.5,
        true,
        false,
        head_pref_2,
        &cand_margins,
        &cand_all_safe,
        &cand_fov_safe,
        &cand_fim,
    );

    let role1 = apply_coordination_and_authority(
        1,
        swerve_ticks_1,
        true,
        None,
        64.0,
        true,
        Ring0AuthorityGate::NOMINAL,
        &mut dec1,
    );
    let role2 = apply_coordination_and_authority(
        2,
        swerve_ticks_2,
        true,
        None,
        64.0,
        true,
        Ring0AuthorityGate::NOMINAL,
        &mut dec2,
    );

    assert!(role1 == SwerveCoordRole::OpticalFar91113Right);
    assert!(role2 == SwerveCoordRole::OpticalFar91113Right);
    assert!(dec1.cmd_bank_rad.is_finite() && dec2.cmd_bank_rad.is_finite());
    assert!(dec1.cmd_bank_rad >= -EVADE_BANK_RAD && dec1.cmd_bank_rad <= EVADE_BANK_RAD);
    assert!(dec2.cmd_bank_rad >= -EVADE_BANK_RAD && dec2.cmd_bank_rad <= EVADE_BANK_RAD);

    if any_nonzero_safe {
        assert!(dec1.mode == ActiveRangeMode::ActiveSwerve);
        assert!(dec2.mode == ActiveRangeMode::ActiveSwerve);
        assert!(dec1.cmd_bank_rad > 0.0 && dec1.cmd_bank_rad <= PHI_CAND_RAD[8]);
        assert!(dec2.cmd_bank_rad > 0.0 && dec2.cmd_bank_rad <= PHI_CAND_RAD[8]);
    }
    // Zero opposite-sign reciprocal cancellation under all symmetric head-on states
    assert!(!(dec1.cmd_bank_rad > 0.0 && dec2.cmd_bank_rad < 0.0));
    assert!(!(dec1.cmd_bank_rad < 0.0 && dec2.cmd_bank_rad > 0.0));
}

/// Kani #20 (DR-22 Finding F1 Tier A): For ANY two distinct aircraft IDs (`own_id != peer_id`)
/// with valid fresh `PeerCoordHint`, Tier A coordination enforces deterministic asymmetric
/// phase/sign separation (zero mutual cancellation) and strictly bounded bank angles
/// `|cmd_bank_rad| <= EVADE_BANK_RAD` with zero NaN/Inf.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::unwind(12)]
fn kani_f1_cooperative_peer_stagger_asymmetry() {
    let id_a: u8 = kani::any();
    let id_b: u8 = kani::any();
    kani::assume(id_a != id_b);

    let coord_ticks: u32 = kani::any();
    let now_utc_us: u64 = kani::any();
    let age_a: u64 = kani::any();
    let age_b: u64 = kani::any();
    kani::assume(age_a <= PEER_LINK_FRESH_US && age_b <= PEER_LINK_FRESH_US);
    kani::assume(now_utc_us >= age_a && now_utc_us >= age_b);

    let hint_a = PeerCoordHint {
        peer_sender_id: id_b,
        peer_ts_utc_us: now_utc_us - age_a,
        now_utc_us,
        peer_a_ned: None,
    };
    let hint_b = PeerCoordHint {
        peer_sender_id: id_a,
        peer_ts_utc_us: now_utc_us - age_b,
        now_utc_us,
        peer_a_ned: None,
    };
    assert!(hint_a.is_fresh() && hint_b.is_fresh());

    // Part A: Strict deterministic slot exclusivity & STAGGER_SLOT_TICKS alternation
    let slot_a = tier_a_own_active_slot(id_a, id_b, coord_ticks);
    let slot_b = tier_a_own_active_slot(id_b, id_a, coord_ticks);
    assert!(slot_a != slot_b);

    if coord_ticks <= u32::MAX - STAGGER_SLOT_TICKS {
        let next_slot_a = tier_a_own_active_slot(id_a, id_b, coord_ticks + STAGGER_SLOT_TICKS);
        let next_slot_b = tier_a_own_active_slot(id_b, id_a, coord_ticks + STAGGER_SLOT_TICKS);
        assert!(next_slot_a == !slot_a);
        assert!(next_slot_b == !slot_b);
        assert!(next_slot_a != next_slot_b);
    }

    // Part B: Head-on Tier A arbitration across all positive FAR 91.113 swerve candidates and
    // symbolic straight-flight SC3BF safety flags
    let cand_a_idx: usize = kani::any();
    let cand_b_idx: usize = kani::any();
    kani::assume(cand_a_idx >= 5 && cand_a_idx < NUM_PHI_CANDS);
    kani::assume(cand_b_idx >= 5 && cand_b_idx < NUM_PHI_CANDS);

    let mut dec_a = SwerveDecision {
        mode: ActiveRangeMode::ActiveSwerve,
        cmd_bank_rad: PHI_CAND_RAD[cand_a_idx],
        min_sc3bf_margin: 40.0,
        fim_gain: 12.0,
        sc3bf_all_safe: true,
        fov_safe: true,
        now_locked: false,
    };
    let mut dec_b = SwerveDecision {
        mode: ActiveRangeMode::ActiveSwerve,
        cmd_bank_rad: PHI_CAND_RAD[cand_b_idx],
        min_sc3bf_margin: 40.0,
        fim_gain: 12.0,
        sc3bf_all_safe: true,
        fov_safe: true,
        now_locked: false,
    };

    let safe0_a: bool = kani::any();
    let safe0_b: bool = kani::any();
    let m0_a = if safe0_a { 32.0 } else { -16.0 };
    let m0_b = if safe0_b { 32.0 } else { -16.0 };

    let role_a = apply_coordination_and_authority(
        id_a,
        coord_ticks,
        true,
        Some(hint_a),
        m0_a,
        safe0_a,
        Ring0AuthorityGate::NOMINAL,
        &mut dec_a,
    );
    let role_b = apply_coordination_and_authority(
        id_b,
        coord_ticks,
        true,
        Some(hint_b),
        m0_b,
        safe0_b,
        Ring0AuthorityGate::NOMINAL,
        &mut dec_b,
    );

    // When straight-flight hold is SC3BF-safe for both aircraft, EXACTLY ONE probes right while
    // the other holds straight (phi = 0, YieldHoldForPeerProbe) — strict asymmetric separation!
    if safe0_a && safe0_b {
        if slot_a {
            assert!(role_a == SwerveCoordRole::CooperativeRightProbe);
            assert!(dec_a.mode == ActiveRangeMode::ActiveSwerve && dec_a.cmd_bank_rad > 0.0);
            assert!(role_b == SwerveCoordRole::YieldHoldForPeerProbe);
            assert!(dec_b.mode == ActiveRangeMode::NominalHold && dec_b.cmd_bank_rad == 0.0);
        } else {
            assert!(role_b == SwerveCoordRole::CooperativeRightProbe);
            assert!(dec_b.mode == ActiveRangeMode::ActiveSwerve && dec_b.cmd_bank_rad > 0.0);
            assert!(role_a == SwerveCoordRole::YieldHoldForPeerProbe);
            assert!(dec_a.mode == ActiveRangeMode::NominalHold && dec_a.cmd_bank_rad == 0.0);
        }
    }

    // Across ALL straight-flight safety combinations: zero mutual cancellation, finite & bounded!
    assert!(dec_a.cmd_bank_rad.is_finite() && dec_b.cmd_bank_rad.is_finite());
    assert!(dec_a.cmd_bank_rad >= 0.0 && dec_a.cmd_bank_rad <= EVADE_BANK_RAD);
    assert!(dec_b.cmd_bank_rad >= 0.0 && dec_b.cmd_bank_rad <= EVADE_BANK_RAD);
    assert!(!(dec_a.cmd_bank_rad > 0.0 && dec_b.cmd_bank_rad < 0.0));
    assert!(!(dec_a.cmd_bank_rad < 0.0 && dec_b.cmd_bank_rad > 0.0));

    // Part C: Arbitrary IEEE-754 bank angle (including NaN/Inf/out-of-range) & arbitrary
    // Ring0AuthorityGate states remain strictly finite and bounded |cmd_bank_rad| <= EVADE_BANK_RAD.
    let mode_idx: u8 = kani::any();
    kani::assume(mode_idx <= 3);
    let arb_mode = match mode_idx {
        0 => ActiveRangeMode::NominalHold,
        1 => ActiveRangeMode::ActiveSwerve,
        2 => ActiveRangeMode::Locked,
        _ => ActiveRangeMode::ImmediateEvade,
    };
    let mut dec_arb = SwerveDecision {
        mode: arb_mode,
        cmd_bank_rad: f64::from_bits(kani::any::<u64>()),
        min_sc3bf_margin: f64::from_bits(kani::any::<u64>()),
        fim_gain: f64::from_bits(kani::any::<u64>()),
        sc3bf_all_safe: kani::any(),
        fov_safe: kani::any(),
        now_locked: kani::any(),
    };
    let arb_gate = Ring0AuthorityGate {
        rta_inhibit_avoidance_turns: kani::any(),
        hal_out_mode_wire: kani::any(),
    };
    let _arb_role = apply_coordination_and_authority(
        id_a,
        coord_ticks,
        kani::any(),
        Some(hint_a),
        m0_a,
        safe0_a,
        arb_gate,
        &mut dec_arb,
    );
    assert!(dec_arb.cmd_bank_rad.is_finite());
    assert!(dec_arb.cmd_bank_rad >= -EVADE_BANK_RAD && dec_arb.cmd_bank_rad <= EVADE_BANK_RAD);
}

/// Kani #21 (DR-22 Finding F4): `StatefulMeshDaaGuard` rejects ANY observation with
/// `age > 200 ms`, future skew `> 50 ms`, or per-sender non-monotonic replay timestamp, and
/// preserves sticky triple-ray ghost suppression across transient `3 -> 2` drops within the
/// `500 ms` hysteresis window.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::unwind(20)]
fn kani_f4_mesh_staleness_replay_and_sticky_ghost() {
    // Part A: Universal 2^128 staleness (> 200 ms) & future-skew (> 50 ms) gate
    let now_utc_us: u64 = kani::any();
    let obs_ts: u64 = kani::any();
    let sender_id: u8 = kani::any();

    let mut obs = empty_bearing_track();
    obs.time_utc_us = obs_ts;
    obs.sender_id = sender_id;

    let mut guard_a = StatefulMeshDaaGuard::new();
    let (_buf, n_fresh, stale_rej, replay_rej, n_senders) =
        guard_a.filter_fresh_observations(&[obs], now_utc_us);

    let is_stale = now_utc_us > obs_ts.saturating_add(MAX_MESH_AGE_US);
    let is_future = obs_ts > now_utc_us.saturating_add(MAX_FUTURE_SKEW_US);
    if is_stale || is_future {
        assert!(n_fresh == 0);
        assert!(stale_rej == 1);
        assert!(replay_rej == 0);
        assert!(n_senders == 0);

        // Also confirm full `fuse_guarded` rejects the stale/future observation with 0 tracks
        let mut guard_full = StatefulMeshDaaGuard::new();
        let res_full = guard_full.fuse_guarded(&[obs], now_utc_us);
        assert!(res_full.stale_obs_rejected == 1);
        assert!(res_full.fresh_sender_count == 0);
        assert!(res_full.num_fused == 0);
    } else {
        assert!(n_fresh == 1);
        assert!(stale_rej == 0);
        assert!(replay_rej == 0);
        assert!(n_senders == 1);
    }

    // Part B: Universal per-sender monotonic replay rejection (ts_2 < ts_1)
    let now_1: u64 = kani::any();
    let ts_1: u64 = kani::any();
    let now_2: u64 = kani::any();
    let ts_2: u64 = kani::any();
    let rep_sender: u8 = kani::any();

    // Tick 1 fresh; Tick 2 also within [now_2 - 200ms, now_2 + 50ms] freshness window,
    // but strictly older than committed ts_1 (non-monotonic replay!)
    kani::assume(now_1 <= ts_1.saturating_add(MAX_MESH_AGE_US));
    kani::assume(ts_1 <= now_1.saturating_add(MAX_FUTURE_SKEW_US));
    kani::assume(now_2 <= ts_2.saturating_add(MAX_MESH_AGE_US));
    kani::assume(ts_2 <= now_2.saturating_add(MAX_FUTURE_SKEW_US));
    kani::assume(ts_2 < ts_1);

    let mut obs_1 = empty_bearing_track();
    obs_1.time_utc_us = ts_1;
    obs_1.sender_id = rep_sender;

    let mut obs_2 = empty_bearing_track();
    obs_2.time_utc_us = ts_2;
    obs_2.sender_id = rep_sender;

    let mut guard_b = StatefulMeshDaaGuard::new();
    let (_b1, n1, s1, r1, ns1) = guard_b.filter_fresh_observations(&[obs_1], now_1);
    assert!(n1 == 1 && s1 == 0 && r1 == 0 && ns1 == 1);

    let res_rep = guard_b.fuse_guarded(&[obs_2], now_2);
    assert!(res_rep.replay_obs_rejected == 1);
    assert!(res_rep.stale_obs_rejected == 0);
    assert!(res_rep.fresh_sender_count == 0);
    assert!(res_rep.num_fused == 0);

    // Part C: Sticky triple-ray ghost suppression across transient 3 -> 2 peer drop
    // within TRIPLE_RAY_HYSTERESIS_US (500 ms), plus expiry after 500 ms.
    // C1: Universal 2^128 proof of exact 500 ms sticky hysteresis activation/deactivation window
    let t3_univ: u64 = kani::any();
    let now_univ: u64 = kani::any();
    let drop_senders_univ: usize = kani::any();
    kani::assume(drop_senders_univ < 3);

    let mut guard_win = StatefulMeshDaaGuard::new();
    let empty_raw = MeshFusionResult {
        tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
        num_fused: 0,
        ghosts_rejected: 0,
    };
    let _ = guard_win.apply_sticky_hysteresis(&empty_raw, 3, 0, 0, t3_univ);
    let res_win = guard_win.apply_sticky_hysteresis(&empty_raw, drop_senders_univ, 0, 0, now_univ);
    let expected_sticky =
        now_univ >= t3_univ && (now_univ - t3_univ) <= TRIPLE_RAY_HYSTERESIS_US;
    assert!(res_win.sticky_triple_ray_active == expected_sticky);

    // C2: Symbolic 3-ship anchor lock + 3 -> 2 dropout ghost suppression across boundary & interior
    // dropout durations (0 us, 1 us, 100 ms, 200 ms, 250 ms, 499.999 ms, exact 500 ms) + expiry.
    // (Translation invariance over all 2^128 (t3_univ, now_univ) pairs is already proved in C1.)
    let t3: u64 = 1_700_000_000_000;
    let dt_sel: u8 = kani::any();
    kani::assume(dt_sel < 7);
    let dt_drop_us: u64 = match dt_sel {
        0 => 0,
        1 => 1,
        2 => 100_000,
        3 => 200_000,
        4 => 250_000,
        5 => TRIPLE_RAY_HYSTERESIS_US - 1,
        _ => TRIPLE_RAY_HYSTERESIS_US,
    };
    let now_drop = t3 + dt_drop_us;

    let mut raw_3ship = MeshFusionResult {
        tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
        num_fused: 1,
        ghosts_rejected: 2,
    };
    raw_3ship.tracks[0] = FusedMeshTrack {
        pos_ecef: [6_378_137.0, 200.0, 0.0],
        cov_ecef: [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        triple_ray_confirmed: true,
        valid: true,
    };

    let mut guard_c = StatefulMeshDaaGuard::new();
    let res3 = guard_c.apply_sticky_hysteresis(&raw_3ship, 3, 0, 0, t3);
    assert!(!res3.sticky_triple_ray_active);
    assert!(res3.num_fused == 1);

    // Construct 2-ship dropout result containing 1 real track within merge_radius_m (150 m)
    // and 1 coplanar ghost outside maximum 500 ms kinematic gate (150 m + 120 m/s * 0.5 s = 210 m).
    let real_sel: u8 = kani::any();
    kani::assume(real_sel < 3);
    let real_dy = match real_sel {
        0 => -100.0,
        1 => 0.0,
        _ => 100.0,
    };
    let ghost_sel: u8 = kani::any();
    kani::assume(ghost_sel < 3);
    let ghost_dy = match ghost_sel {
        0 => -400.0,
        1 => 350.0,
        _ => 600.0,
    };

    let mut raw_2ship = MeshFusionResult {
        tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
        num_fused: 2,
        ghosts_rejected: 0,
    };
    raw_2ship.tracks[0] = FusedMeshTrack {
        pos_ecef: [6_378_137.0, 200.0 + real_dy, 0.0],
        cov_ecef: [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        triple_ray_confirmed: false,
        valid: true,
    };
    raw_2ship.tracks[1] = FusedMeshTrack {
        pos_ecef: [6_378_137.0, 200.0 + ghost_dy, 0.0],
        cov_ecef: [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        triple_ray_confirmed: false,
        valid: true,
    };

    let drop_senders: usize = kani::any();
    kani::assume(drop_senders < 3);

    let res2 = guard_c.apply_sticky_hysteresis(&raw_2ship, drop_senders, 0, 0, now_drop);
    assert!(res2.sticky_triple_ray_active);
    assert!(res2.num_fused == 1);
    assert!(res2.sticky_ghosts_suppressed == 1);
    assert!(res2.tracks[0].valid && res2.tracks[0].triple_ray_confirmed);
    assert!(res2.tracks[0].pos_ecef[1] == 200.0 + real_dy);
    assert!(!res2.tracks[1].valid);

    // After hysteresis expiry (> 500 ms), sticky gate deactivates cleanly
    let res_exp = guard_c.apply_sticky_hysteresis(
        &raw_2ship,
        drop_senders,
        0,
        0,
        t3 + TRIPLE_RAY_HYSTERESIS_US + 1,
    );
    assert!(!res_exp.sticky_triple_ray_active);
}

/// Kani #22 (DR-22 Finding F3): `Ring0AuthorityGate` strictly outputs `cmd_bank_rad == 0.0`
/// (exact `+0.0`, `NominalHold`, `InhibitedByP0`) whenever `rta_inhibit_avoidance_turns == true`
/// OR `hal_out_mode_wire == 0`, regardless of arbitrary `ActiveRange` command (including
/// `ImmediateEvade` or arbitrary IEEE-754 `NaN`/`Inf` bit patterns).
#[kani::proof]
#[kani::solver(cadical)]
#[kani::unwind(12)]
fn kani_f3_ring0_p0_zeroes_all_avoidance() {
    let rta_inhibit_avoidance_turns: bool = kani::any();
    let hal_out_mode_wire: u8 = kani::any();
    let gate = Ring0AuthorityGate {
        rta_inhibit_avoidance_turns,
        hal_out_mode_wire,
    };

    let mode_idx: u8 = kani::any();
    kani::assume(mode_idx <= 3);
    let init_mode = match mode_idx {
        0 => ActiveRangeMode::NominalHold,
        1 => ActiveRangeMode::ActiveSwerve,
        2 => ActiveRangeMode::Locked,
        _ => ActiveRangeMode::ImmediateEvade,
    };
    let role_idx: u8 = kani::any();
    kani::assume(role_idx <= 3);
    let mut role = match role_idx {
        0 => SwerveCoordRole::InhibitedByP0,
        1 => SwerveCoordRole::YieldHoldForPeerProbe,
        2 => SwerveCoordRole::CooperativeRightProbe,
        _ => SwerveCoordRole::OpticalFar91113Right,
    };

    let raw_bank = f64::from_bits(kani::any::<u64>());
    let mut dec = SwerveDecision {
        mode: init_mode,
        cmd_bank_rad: raw_bank,
        min_sc3bf_margin: f64::from_bits(kani::any::<u64>()),
        fim_gain: f64::from_bits(kani::any::<u64>()),
        sc3bf_all_safe: kani::any(),
        fov_safe: kani::any(),
        now_locked: kani::any(),
    };

    // Part A: Universal Ring0AuthorityGate::apply_to_decision proof over ALL 2^64 bank bits
    gate.apply_to_decision(&mut dec, &mut role);

    if rta_inhibit_avoidance_turns || hal_out_mode_wire == 0 {
        assert!(gate.inhibits_turns());
        assert!(dec.cmd_bank_rad == 0.0);
        assert!(dec.cmd_bank_rad.to_bits() == 0.0_f64.to_bits());
        assert!(dec.mode == ActiveRangeMode::NominalHold);
        assert!(dec.fim_gain == 0.0);
        assert!(role == SwerveCoordRole::InhibitedByP0);
    } else {
        assert!(!gate.inhibits_turns());
    }

    // Unconditional finiteness & ±EVADE_BANK_RAD envelope guarantee even under NaN/Inf input
    assert!(dec.cmd_bank_rad.is_finite());
    assert!(dec.cmd_bank_rad >= -EVADE_BANK_RAD && dec.cmd_bank_rad <= EVADE_BANK_RAD);
    if !raw_bank.is_finite() {
        assert!(dec.cmd_bank_rad == 0.0);
        assert!(dec.mode == ActiveRangeMode::NominalHold);
    }

    // Part B: End-to-end frozen `select_active_swerve_from_evals` producing genuine terminal
    // `ImmediateEvade` (±25 deg) piped through `apply_coordination_and_authority` with symbolic
    // `Ring0AuthorityGate` + symbolic `PeerCoordHint`: whenever `inhibits_turns()`, bank is 0.0!
    let cand_margins = [-50.0f64; NUM_PHI_CANDS];
    let cand_all_safe = [false; NUM_PHI_CANDS];
    let cand_fov_safe = [false; NUM_PHI_CANDS];
    let cand_fim = [0.0f64; NUM_PHI_CANDS];
    let alpha_sel: u8 = kani::any();
    kani::assume(alpha_sel < 3);
    let alpha_body = match alpha_sel {
        0 => -0.05,
        1 => 0.0,
        _ => 0.05,
    };
    let mut evade_dec = select_active_swerve_from_evals(
        180.0,
        20.0,
        140.0,
        -0.30, // tau = 3.33 s < 4.5 s terminal evade
        alpha_body,
        -120.0, // inside violated collision cone
        40.0,
        14.5,
        false,
        false,
        1.0,
        &cand_margins,
        &cand_all_safe,
        &cand_fov_safe,
        &cand_fim,
    );
    assert!(evade_dec.mode == ActiveRangeMode::ImmediateEvade);
    assert!(evade_dec.cmd_bank_rad != 0.0);

    let evade_role = apply_coordination_and_authority(
        kani::any(),
        kani::any(),
        kani::any(),
        None,
        -50.0,
        false,
        gate,
        &mut evade_dec,
    );
    if rta_inhibit_avoidance_turns || hal_out_mode_wire == 0 {
        assert!(evade_dec.cmd_bank_rad == 0.0);
        assert!(evade_dec.cmd_bank_rad.to_bits() == 0.0_f64.to_bits());
        assert!(evade_dec.mode == ActiveRangeMode::NominalHold);
        assert!(evade_role == SwerveCoordRole::InhibitedByP0);
    } else {
        assert!(evade_dec.mode == ActiveRangeMode::ImmediateEvade);
        assert!(evade_dec.cmd_bank_rad.is_finite());
        assert!(
            evade_dec.cmd_bank_rad >= -EVADE_BANK_RAD && evade_dec.cmd_bank_rad <= EVADE_BANK_RAD
        );
    }
}

/// Kani #23 (V5.2 Clock Sync): Arbitrary `u64` two-way RF timestamps NEVER panic or overflow;
/// invalid causality, negative RTT, `rtt > 150 ms`, and out-of-bounds offsets are rejected
/// fail-closed; the 5-element median sorting network is universally sorted; and emitted offsets
/// and synchronized uncertainty bounds are strictly respected across the full 5-sample window.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::unwind(20)]
fn kani_clock_sync_overflow_and_bounds_free() {
    // Part A: Universal sorting network correctness over arbitrary 5-tuples
    let raw_i: [i64; 5] = [
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
    ];
    let si = sort5_i64(raw_i);
    assert!(si[0] <= si[1] && si[1] <= si[2] && si[2] <= si[3] && si[3] <= si[4]);

    let raw_u: [u64; 5] = [
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
    ];
    let su = sort5_u64(raw_u);
    assert!(su[0] <= su[1] && su[1] <= su[2] && su[2] <= su[3] && su[3] <= su[4]);

    // Part B: Universal single-exchange proof over entire 2^256 (t1, t2, t3, t4) u64 space
    let sender_id: u8 = kani::any();
    let t1: u64 = kani::any();
    let t2: u64 = kani::any();
    let t3: u64 = kani::any();
    let t4: u64 = kani::any();

    let mut est_single = PeerClockOffsetEstimator::new();
    let out = est_single.record_exchange(TwoWaySyncExchange {
        sender_id,
        t1_local_tx_us: t1,
        t2_peer_rx_us: t2,
        t3_peer_tx_us: t3,
        t4_local_rx_us: t4,
    });

    if t4 < t1 || t3 < t2 {
        assert!(!out.accepted);
        assert!(matches!(
            out.reject_reason,
            Some(ClockSyncRejectReason::NonCausalTimestamps)
        ));
    } else {
        let local_span = t4 - t1;
        let peer_turn = t3 - t2;
        if local_span < peer_turn {
            assert!(!out.accepted);
            assert!(matches!(
                out.reject_reason,
                Some(ClockSyncRejectReason::NegativeRtt)
            ));
        } else {
            let rtt = local_span - peer_turn;
            if rtt > MAX_SYNC_RTT_US {
                assert!(!out.accepted);
                assert!(matches!(
                    out.reject_reason,
                    Some(ClockSyncRejectReason::RttExceeded)
                ));
            } else {
                let d1 = (t2 as i128) - (t1 as i128);
                let d2 = (t3 as i128) - (t4 as i128);
                let sum = d1 + d2;
                let off = if sum >= 0 {
                    sum >> 1
                } else {
                    -((sum.wrapping_neg() as u128 >> 1) as i128)
                };
                let bound = MAX_ABS_CLOCK_OFFSET_US as i128;
                if off < -bound || off > bound {
                    assert!(!out.accepted);
                    assert!(matches!(
                        out.reject_reason,
                        Some(ClockSyncRejectReason::OffsetOutOfBounds)
                    ));
                } else {
                    assert!(out.accepted);
                    assert!(out.reject_reason.is_none());
                    assert!(out.rtt_us <= MAX_SYNC_RTT_US);
                    assert!(out.raw_offset_us >= -MAX_ABS_CLOCK_OFFSET_US);
                    assert!(out.raw_offset_us <= MAX_ABS_CLOCK_OFFSET_US);
                }
            }
        }
    }

    // Part C: Full 5-sample rolling window + arbitrary evaluation & translation timestamps.
    // Part B already proved `validate_two_way_exchange` over all 2^256 u64 tuples emits only
    // `|raw_off| <= MAX_ABS_CLOCK_OFFSET_US` and `rtt <= MAX_SYNC_RTT_US`; here we prove all
    // 5-sample rolling window updates, median-of-5, bounded slewing, and translation are panic-,
    // overflow-, and bounds-free across arbitrary valid samples + arbitrary local timestamps.
    let mut slot = PeerSyncSlot::EMPTY;
    let mut step = 0usize;
    while step < 5 {
        let raw_off: i64 = kani::any();
        let rtt: u64 = kani::any();
        let t4_s: u64 = kani::any();
        kani::assume(raw_off >= -MAX_ABS_CLOCK_OFFSET_US && raw_off <= MAX_ABS_CLOCK_OFFSET_US);
        kani::assume(rtt <= MAX_SYNC_RTT_US);
        let _ = slot.ingest_validated(raw_off, rtt, t4_s, MAX_OFFSET_UNCERT_US, MAX_SLEW_STEP_US);
        let est_step = slot.estimate(t4_s, MAX_OFFSET_UNCERT_US);
        assert!(est_step.offset_us >= -MAX_ABS_CLOCK_OFFSET_US);
        assert!(est_step.offset_us <= MAX_ABS_CLOCK_OFFSET_US);
        step += 1;
    }

    let now_eval: u64 = kani::any();
    let peer_pkt_ts: u64 = kani::any();
    let snap = slot.estimate(now_eval, MAX_OFFSET_UNCERT_US);
    assert!(snap.offset_us >= -MAX_ABS_CLOCK_OFFSET_US);
    assert!(snap.offset_us <= MAX_ABS_CLOCK_OFFSET_US);
    if snap.synchronized {
        assert!((snap.sample_count as usize) >= MIN_SYNC_SAMPLES);
        assert!(snap.offset_uncert_us <= MAX_OFFSET_UNCERT_US);
        assert!(snap.median_rtt_us <= MAX_SYNC_RTT_US);
        assert!(now_eval >= snap.last_sync_local_us);
    }

    if let Some((aligned_ts, trans_est)) = translate_ts_with_estimate(peer_pkt_ts, snap) {
        assert!(trans_est.synchronized);
        assert!(trans_est.offset_uncert_us <= MAX_OFFSET_UNCERT_US);
        if trans_est.offset_us >= 0 {
            assert!(aligned_ts == peer_pkt_ts - (trans_est.offset_us as u64));
        } else {
            assert!(aligned_ts == peer_pkt_ts + trans_est.offset_us.unsigned_abs());
        }
    }
}

/// Kani #24 (V5.2 Clock Sync): Under arbitrary GNSS-denied TCXO offset in `[-500 ms, +500 ms]`
/// plus small RF link jitter and a single multipath outlier, median-of-5 calibration achieves
/// `offset_uncert_us <= 20 ms`, accepts fresh monotonic `#52001` peer packets with ZERO false
/// rejections, and STILL rejects genuine `> 200 ms` stale and replay attacks.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::unwind(12)]
fn kani_clock_sync_gnss_denied_drift_no_false_reject() {
    let own_id: u8 = kani::any();
    let peer_id: u8 = kani::any();
    kani::assume(peer_id != own_id);

    // Arbitrary GNSS-denied TCXO clock offset in [-500 ms, +500 ms] (10x past 50 ms future skew!)
    let true_offset_us: i64 = kani::any();
    kani::assume(true_offset_us >= -500_000 && true_offset_us <= 500_000);

    // Small bounded RF flight delays per nominal exchange (RTT <= 16 ms, |asymmetry| <= 4 ms)
    let d_fwd_nom: u64 = kani::any();
    let d_rev_nom: u64 = kani::any();
    let turn_nom: u64 = kani::any();
    kani::assume(d_fwd_nom <= 8_000 && d_rev_nom <= 8_000 && turn_nom <= 5_000);

    // Single wild multipath outlier slot in 0..5 with ±200 ms corrupted offset and 120 ms RTT
    let outlier_slot: usize = kani::any();
    kani::assume(outlier_slot < 5);
    let outlier_pos: bool = kani::any();
    let outlier_bias: i64 = if outlier_pos { 200_000 } else { -200_000 };

    let base_local: u64 = 1_700_000_000_000;
    let mut slot = PeerSyncSlot::EMPTY;
    slot.used = true;
    slot.sender_id = peer_id;

    let mut k = 0usize;
    while k < 5 {
        let t1 = base_local + (k as u64) * 200_000;
        let (off_k, df, dr, turn) = if k == outlier_slot {
            (true_offset_us + outlier_bias, 60_000u64, 60_000u64, 5_000u64)
        } else {
            (true_offset_us, d_fwd_nom, d_rev_nom, turn_nom)
        };
        let t2 = t1.wrapping_add_signed(off_k) + df;
        let t3 = t2 + turn;
        let t4 = t1 + df + turn + dr;
        let ex = TwoWaySyncExchange {
            sender_id: peer_id,
            t1_local_tx_us: t1,
            t2_peer_rx_us: t2,
            t3_peer_tx_us: t3,
            t4_local_rx_us: t4,
        };
        let validated = validate_two_way_exchange(&ex, MAX_SYNC_RTT_US);
        assert!(validated.is_ok());
        if let Ok((raw_off, rtt)) = validated {
            let ing = slot.ingest_validated(
                raw_off,
                rtt,
                t4,
                MAX_OFFSET_UNCERT_US,
                MAX_SLEW_STEP_US,
            );
            assert!(ing.is_ok());
        }
        k += 1;
    }

    // Evaluate shortly after sync (within 200 ms)
    let now_local = base_local + 1_000_000;
    let est = slot.estimate(now_local, MAX_OFFSET_UNCERT_US);
    assert!(est.synchronized);
    assert!(est.offset_uncert_us <= MAX_OFFSET_UNCERT_US);
    let err_us = est.offset_us - true_offset_us;
    assert!(err_us >= -4_000 && err_us <= 4_000);

    // 1. Fresh monotonic peer packet (transit delay 0..=150 ms) MUST be accepted (0 false rejects!)
    let transit_us: u64 = kani::any();
    kani::assume(transit_us <= 150_000);
    let peer_fresh_ts = (now_local - transit_us).wrapping_add_signed(true_offset_us);

    let v_fresh = evaluate_obs_clock_synced(
        peer_id,
        peer_fresh_ts,
        now_local,
        own_id,
        true,
        est,
        None,
        MAX_MESH_AGE_US,
        MAX_FUTURE_SKEW_US,
    );
    assert!(matches!(v_fresh, ObsClockSyncVerdict::Accept(_)));

    // 2. Genuine stale packet (> 200 ms + uncertainty margin, e.g. 225..=1_000 ms old) MUST be rejected!
    let stale_age_us: u64 = kani::any();
    kani::assume(stale_age_us >= MAX_MESH_AGE_US + 25_000 && stale_age_us <= 1_000_000);
    let peer_stale_ts = (now_local - stale_age_us).wrapping_add_signed(true_offset_us);

    let v_stale = evaluate_obs_clock_synced(
        peer_id,
        peer_stale_ts,
        now_local,
        own_id,
        true,
        est,
        None,
        MAX_MESH_AGE_US,
        MAX_FUTURE_SKEW_US,
    );
    assert!(matches!(v_stale, ObsClockSyncVerdict::RejectStaleOrFuture));

    // 3. Genuine non-monotonic replay packet (raw_ts < peer_fresh_ts, yet aligned within freshness
    //    window) MUST be rejected!
    let replay_back_us: u64 = kani::any();
    kani::assume(replay_back_us >= 1 && replay_back_us <= 20_000);
    let est_next = slot.estimate(now_local + 1_000, MAX_OFFSET_UNCERT_US);
    let v_replay = evaluate_obs_clock_synced(
        peer_id,
        peer_fresh_ts - replay_back_us,
        now_local + 1_000,
        own_id,
        true,
        est_next,
        Some(peer_fresh_ts),
        MAX_MESH_AGE_US,
        MAX_FUTURE_SKEW_US,
    );
    assert!(matches!(v_replay, ObsClockSyncVerdict::RejectReplay));
}

/// Kani #25 (V5.2 Clock Sync): Any uncalibrated, excessive-RTT (`> 150 ms`), degraded-RTT
/// (`rtt/2 > 20 ms`), or holdover-expired peer during GNSS-denied mode — as well as any 3-ship
/// dropout exceeding `TRIPLE_RAY_HYSTERESIS_US` (`500 ms`) — NEVER admits an unguarded 2-ship
/// coplanar ghost and strictly asserts `fallback_to_tier_b_optical == true`.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::unwind(12)]
fn kani_clock_sync_uncalibrated_forces_tier_b_fallback() {
    let own_id: u8 = 1;
    let peer_bad: u8 = 3;
    let base_local: u64 = 1_700_000_000_000;

    // Symbolic fault mode for peer_bad during GNSS-denied mode:
    //   0: never calibrated (0 exchanges)
    //   1: excessive RTT (> MAX_SYNC_RTT_US = 150 ms) rejected fail-closed
    //   2: degraded sustained RTT (42..=150 ms -> rtt/2 > 20 ms MAX_OFFSET_UNCERT_US)
    //   3: holdover expired (> 500 s since last sync -> TCXO drift > 20 ms MAX_OFFSET_UNCERT_US)
    let fault_mode: u8 = kani::any();
    kani::assume(fault_mode < 4);

    let mut bad_slot = PeerSyncSlot::EMPTY;
    bad_slot.used = true;
    bad_slot.sender_id = peer_bad;
    let mut now_local = base_local + 700_000;

    if fault_mode == 1 {
        let bad_rtt_us: u64 = kani::any();
        kani::assume(bad_rtt_us > MAX_SYNC_RTT_US && bad_rtt_us <= 2_000_000);
        let t1 = base_local;
        let ex = TwoWaySyncExchange {
            sender_id: peer_bad,
            t1_local_tx_us: t1,
            t2_peer_rx_us: t1 + (bad_rtt_us >> 1),
            t3_peer_tx_us: t1 + (bad_rtt_us >> 1) + 1_000,
            t4_local_rx_us: t1 + bad_rtt_us + 1_000,
        };
        let res = validate_two_way_exchange(&ex, MAX_SYNC_RTT_US);
        assert!(matches!(res, Err((ClockSyncRejectReason::RttExceeded, _, _))));
    } else if fault_mode == 2 {
        let high_rtt_us: u64 = kani::any();
        kani::assume(high_rtt_us >= 42_000 && high_rtt_us <= MAX_SYNC_RTT_US);
        let half = high_rtt_us >> 1;
        let rem = high_rtt_us - half;
        let mut j = 0u64;
        while j < 3 {
            let t1 = base_local + j * 200_000;
            let ex = TwoWaySyncExchange {
                sender_id: peer_bad,
                t1_local_tx_us: t1,
                t2_peer_rx_us: t1 + half,
                t3_peer_tx_us: t1 + half + 1_000,
                t4_local_rx_us: t1 + half + 1_000 + rem,
            };
            let validated = validate_two_way_exchange(&ex, MAX_SYNC_RTT_US);
            assert!(validated.is_ok());
            if let Ok((raw_off, rtt)) = validated {
                let ing = bad_slot.ingest_validated(
                    raw_off,
                    rtt,
                    ex.t4_local_rx_us,
                    MAX_OFFSET_UNCERT_US,
                    MAX_SLEW_STEP_US,
                );
                assert!(ing.is_ok());
            }
            j += 1;
        }
    } else if fault_mode == 3 {
        let mut j = 0u64;
        while j < 3 {
            let t1 = base_local + j * 100_000;
            let ing = bad_slot.ingest_validated(
                100_000,
                10_000,
                t1 + 12_000,
                MAX_OFFSET_UNCERT_US,
                MAX_SLEW_STEP_US,
            );
            assert!(ing.is_ok());
            j += 1;
        }
        let holdover_extra: u64 = kani::any();
        kani::assume(holdover_extra >= 500_000_000 && holdover_extra <= 3_600_000_000);
        now_local = base_local + 212_000 + holdover_extra;
    }

    let est_bad = bad_slot.estimate(now_local, MAX_OFFSET_UNCERT_US);
    assert!(!est_bad.synchronized);

    // GNSS-denied gate MUST reject peer_bad's arbitrary observation as RejectUncalibrated!
    let arb_peer_ts: u64 = kani::any();
    let verdict = evaluate_obs_clock_synced(
        peer_bad,
        arb_peer_ts,
        now_local,
        own_id,
        true,
        est_bad,
        None,
        MAX_MESH_AGE_US,
        MAX_FUTURE_SKEW_US,
    );
    assert!(matches!(verdict, ObsClockSyncVerdict::RejectUncalibrated));

    // Pipe 2-ship raw fusion result containing an unconfirmed coplanar ghost through
    // StatefulMeshDaaGuard::apply_sticky_hysteresis_clock_synced (with any_peer_uncalibrated = true)
    let mut raw_2ship_ghost = MeshFusionResult {
        tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
        num_fused: 1,
        ghosts_rejected: 0,
    };
    raw_2ship_ghost.tracks[0] = FusedMeshTrack {
        pos_ecef: [6_378_137.0, 600.0, 0.0],
        cov_ecef: [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        triple_ray_confirmed: false,
        valid: true,
    };

    let mut guard = StatefulMeshDaaGuard::new();
    let res_uncal = guard.apply_sticky_hysteresis_clock_synced(
        &raw_2ship_ghost,
        2,
        0,
        0,
        1,
        true,
        now_local,
    );
    assert!(res_uncal.fallback_to_tier_b_optical);
    assert!(res_uncal.num_fused == 0);
    assert!(!res_uncal.tracks[0].valid);
    assert!(res_uncal.sticky_ghosts_suppressed == 1);

    // Part B: After locking 3-ship anchors, once 3-ship dropout exceeds TRIPLE_RAY_HYSTERESIS_US
    // (500 ms), unguarded 2-ship ghosts are NEVER admitted and fallback_to_tier_b_optical == true!
    let mut raw_3ship = MeshFusionResult {
        tracks: [FusedMeshTrack::default(); MAX_FUSED_TRACKS],
        num_fused: 1,
        ghosts_rejected: 1,
    };
    raw_3ship.tracks[0] = FusedMeshTrack {
        pos_ecef: [6_378_137.0, 200.0, 0.0],
        cov_ecef: [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        triple_ray_confirmed: true,
        valid: true,
    };
    let mut guard_exp = StatefulMeshDaaGuard::new();
    let res_3 =
        guard_exp.apply_sticky_hysteresis_clock_synced(&raw_3ship, 3, 0, 0, 0, false, now_local);
    assert!(!res_3.fallback_to_tier_b_optical);
    assert!(res_3.num_fused == 1);

    let extra_us: u64 = kani::any();
    kani::assume(extra_us >= 1 && extra_us <= 10_000_000);
    let now_expired = now_local + TRIPLE_RAY_HYSTERESIS_US + extra_us;
    let drop_senders: usize = kani::any();
    kani::assume(drop_senders < 3);

    let res_exp = guard_exp.apply_sticky_hysteresis_clock_synced(
        &raw_2ship_ghost,
        drop_senders,
        0,
        0,
        0,
        false,
        now_expired,
    );
    assert!(res_exp.fallback_to_tier_b_optical);
    assert!(!res_exp.sticky_triple_ray_active);
    assert!(res_exp.num_fused == 0);
    assert!(!res_exp.tracks[0].valid);
    assert!(res_exp.sticky_ghosts_suppressed == 1);
}

/// Kani #26 (V5.2 Clock Sync): Bounded clock-offset slewing (upward OR downward) can NEVER cause
/// a previously rejected old replay timestamp to be re-accepted, while genuinely newer monotonic
/// peer packets continue to pass cleanly.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::unwind(12)]
fn kani_clock_sync_slew_preserves_monotonic_replay_guard() {
    let own_id: u8 = kani::any();
    let peer_id: u8 = kani::any();
    kani::assume(peer_id != own_id);

    let offset_1: i64 = kani::any();
    kani::assume(offset_1 >= -500_000 && offset_1 <= 500_000);

    // Non-zero symbolic shift (upward OR downward) exercised via bounded slewing!
    let target_shift: i64 = kani::any();
    kani::assume(target_shift >= -8_000 && target_shift <= 8_000);
    let target_off = offset_1 + target_shift;

    let base_local: u64 = 1_700_000_000_000;
    let mut slot = PeerSyncSlot::EMPTY;
    slot.used = true;
    slot.sender_id = peer_id;

    // Calibrate peer at exact initial median `offset_1` using 3 exchanges: [offset_1, offset_1, target_off].
    let ex_offs = [offset_1, offset_1, target_off];
    let mut k = 0usize;
    while k < 3 {
        let t1 = base_local + (k as u64) * 100_000;
        let ex = TwoWaySyncExchange {
            sender_id: peer_id,
            t1_local_tx_us: t1,
            t2_peer_rx_us: t1.wrapping_add_signed(ex_offs[k]) + 5_000,
            t3_peer_tx_us: t1.wrapping_add_signed(ex_offs[k]) + 7_000,
            t4_local_rx_us: t1 + 12_000,
        };
        let validated = validate_two_way_exchange(&ex, MAX_SYNC_RTT_US);
        assert!(validated.is_ok());
        if let Ok((raw_off, rtt)) = validated {
            let ing = slot.ingest_validated(
                raw_off,
                rtt,
                ex.t4_local_rx_us,
                MAX_OFFSET_UNCERT_US,
                MAX_SLEW_STEP_US,
            );
            assert!(ing.is_ok());
        }
        k += 1;
    }

    let now_1 = base_local + 220_000;
    let est_1 = slot.estimate(now_1, MAX_OFFSET_UNCERT_US);
    assert!(est_1.synchronized);
    assert!(est_1.offset_us == offset_1);

    // Tick 1: Accept fresh packet P1 and commit watermark `prev_raw_ts = Some(raw_1)`
    let transit_1: u64 = kani::any();
    kani::assume(transit_1 <= 40_000);
    let raw_1 = (now_1 - transit_1).wrapping_add_signed(offset_1);
    let v1 = evaluate_obs_clock_synced(
        peer_id,
        raw_1,
        now_1,
        own_id,
        true,
        est_1,
        None,
        MAX_MESH_AGE_US,
        MAX_FUTURE_SKEW_US,
    );
    assert!(matches!(v1, ObsClockSyncVerdict::Accept(_)));
    let committed_watermark = Some(raw_1);

    // Old replay packet P_old (strictly older raw timestamp, yet inside freshness window) -> REJECT!
    let replay_delta: u64 = kani::any();
    kani::assume(replay_delta >= 1 && replay_delta <= 20_000);
    let raw_old = raw_1 - replay_delta;
    let v_old_before = evaluate_obs_clock_synced(
        peer_id,
        raw_old,
        now_1 + 1_000,
        own_id,
        true,
        slot.estimate(now_1 + 1_000, MAX_OFFSET_UNCERT_US),
        committed_watermark,
        MAX_MESH_AGE_US,
        MAX_FUTURE_SKEW_US,
    );
    assert!(matches!(v_old_before, ObsClockSyncVerdict::RejectReplay));

    // Exchange 4 at `target_off`: rolling 4-sample window `[o1, o1, target, target]` shifts median
    // to `offset_1 + target_shift / 2`, triggering REAL non-zero bounded slewing (upward OR downward)!
    let dt_sync: u64 = kani::any();
    kani::assume(dt_sync >= 20_000 && dt_sync <= 60_000);
    let t1_s = est_1.last_sync_local_us + dt_sync;
    let ex_s = TwoWaySyncExchange {
        sender_id: peer_id,
        t1_local_tx_us: t1_s,
        t2_peer_rx_us: t1_s.wrapping_add_signed(target_off) + 5_000,
        t3_peer_tx_us: t1_s.wrapping_add_signed(target_off) + 7_000,
        t4_local_rx_us: t1_s + 12_000,
    };
    let val_s = validate_two_way_exchange(&ex_s, MAX_SYNC_RTT_US);
    assert!(val_s.is_ok());
    if let Ok((raw_off, rtt)) = val_s {
        let ing = slot.ingest_validated(
            raw_off,
            rtt,
            ex_s.t4_local_rx_us,
            MAX_OFFSET_UNCERT_US,
            MAX_SLEW_STEP_US,
        );
        assert!(ing.is_ok());
    }

    let now_2 = t1_s + 14_000;
    let est_2 = slot.estimate(now_2, MAX_OFFSET_UNCERT_US);
    assert!(est_2.synchronized);
    assert!(est_2.offset_uncert_us <= MAX_OFFSET_UNCERT_US);

    let actual_dt = est_2.last_sync_local_us - est_1.last_sync_local_us;
    let max_allowed_slew = (actual_dt >> SLEW_RATE_SHIFT).min(MAX_SLEW_STEP_US);
    let slew_mag = (est_2.offset_us - offset_1).unsigned_abs();
    assert!(slew_mag <= max_allowed_slew);
    if target_shift >= 2 {
        assert!(est_2.offset_us > offset_1);
    } else if target_shift <= -2 {
        assert!(est_2.offset_us < offset_1);
    }

    // Re-submit exact previously rejected replay packet P_old after bounded slewing!
    let v_old_after = evaluate_obs_clock_synced(
        peer_id,
        raw_old,
        now_2,
        own_id,
        true,
        est_2,
        committed_watermark,
        MAX_MESH_AGE_US,
        MAX_FUTURE_SKEW_US,
    );
    assert!(!matches!(v_old_after, ObsClockSyncVerdict::Accept(_)));
    assert!(matches!(
        v_old_after,
        ObsClockSyncVerdict::RejectReplay | ObsClockSyncVerdict::RejectStaleOrFuture
    ));

    // Whenever P_old falls inside the post-slew freshness window (including downward slew pushing
    // aligned(P_old) >= aligned(P1)!), replay gate explicitly rejects it!
    let aligned_old = raw_old.wrapping_add_signed(-est_2.offset_us);
    let in_window = now_2 <= aligned_old.saturating_add(MAX_MESH_AGE_US)
        && aligned_old <= now_2.saturating_add(MAX_FUTURE_SKEW_US);
    if in_window {
        assert!(matches!(v_old_after, ObsClockSyncVerdict::RejectReplay));
    }

    // Meanwhile a genuinely newer monotonic peer packet P2 IS accepted cleanly!
    let raw_2 = (now_2 - 5_000).wrapping_add_signed(est_2.offset_us);
    if raw_2 >= raw_1 {
        let v2 = evaluate_obs_clock_synced(
            peer_id,
            raw_2,
            now_2,
            own_id,
            true,
            est_2,
            committed_watermark,
            MAX_MESH_AGE_US,
            MAX_FUTURE_SKEW_US,
        );
        assert!(matches!(v2, ObsClockSyncVerdict::Accept(_)));
    }
}
