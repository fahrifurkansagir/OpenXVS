// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0
//! DR-22 Finding F1 Fix: FAR 91.113(e) Non-Reversing Right-of-Way & Deterministic Aircraft-ID
//! Phase-Staggered Active-Range Coordination + Ring-0 `P0` Authority Arbiter.
//!
//! Wraps the unmodified frozen `xvs-active-range` filtering (`MscRpekfBank`), split-conformal
//! (`compute_conformal_lower_bound`), universal `SC3BF-FIM` (`select_active_swerve`), and
//! 64-byte IEEE-802.3 CRC32 wire codec (`encode_wire`) from above without modifying a single
//! line of any frozen crate.

use xvs_active_range::{
    clamp_f64, compute_conformal_lower_bound, encode_wire, eval_universal_sc3bf,
    select_active_swerve, wrap_pi, ActiveRangeConfig, ActiveRangeMeasurement, ActiveRangeMode,
    ActiveRangeTelemetry, LosTriadNed, MscRpekfBank, OwnshipStateNed, SwerveDecision,
    SwerveInputs, EVADE_BANK_RAD, FLAG_FOV_SAFE, FLAG_SC3BF_ALL_SAFE, NUM_BANKS, R_CEIL_M,
    R_FLOOR_M, TAU_MIN_SIZE_PX, TAU_MIN_TRACK_AGE,
};

/// Head-on azimuth symmetry band (`|alpha_body_rad| < 0.07 rad`, matching `sc3bf_fim.rs:440`).
pub const HEAD_ON_ALPHA_BAND_RAD: f64 = 0.07;

/// Deterministic alternating probe window length in ticks (`20 ticks = 2.0 s @ 10 Hz`)
/// when a fresh cooperative `XVS-MeshDAA` peer link is active.
pub const STAGGER_SLOT_TICKS: u32 = 20;

/// Maximum acceptable age of a `XVS-MeshDAA` peer hint (`200 ms`, matching `xvs-os-hal`).
pub const PEER_LINK_FRESH_US: u64 = 200_000;

/// Optional fresh cooperative peer identity & acceleration from `XVS-MeshDAA`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeerCoordHint {
    pub peer_sender_id: u8,
    pub peer_ts_utc_us: u64,
    pub now_utc_us: u64,
    /// Optional peer lateral/inertial acceleration in NED (`m/s^2`) for exact relative
    /// acceleration compensation (`a_eff = a_own - a_peer`) in frozen `MscRpekfBank`.
    pub peer_a_ned: Option<[f64; 3]>,
}

impl PeerCoordHint {
    #[inline]
    pub const fn is_fresh(&self) -> bool {
        self.now_utc_us >= self.peer_ts_utc_us
            && (self.now_utc_us - self.peer_ts_utc_us) <= PEER_LINK_FRESH_US
    }
}

/// Ring-0 cross-layer authority gate from frozen `xvs-rta-core` + `xvs-os-hal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ring0AuthorityGate {
    /// `RtaTickOutput::inhibit_avoidance_turns` (`true` iff `xvs-rta-core` is in `P0`).
    pub rta_inhibit_avoidance_turns: bool,
    /// `xvs-os-hal` `SupervisorDecision::out_mode.to_rta_wire_u8()` (`0` iff `HAL-P0`).
    pub hal_out_mode_wire: u8,
}

impl Ring0AuthorityGate {
    pub const NOMINAL: Self = Self {
        rta_inhibit_avoidance_turns: false,
        hal_out_mode_wire: 4,
    };

    #[inline]
    pub const fn inhibits_turns(&self) -> bool {
        self.rta_inhibit_avoidance_turns || self.hal_out_mode_wire == 0
    }

    /// Enforce Ring-0 `P0` / `HAL-P0` dominance and finite `±EVADE_BANK_RAD` envelope bounds
    /// over any `ActiveRange` decision (including `ImmediateEvade` or non-finite `NaN`/`Inf`).
    #[inline]
    pub(crate) fn apply_to_decision(
        &self,
        decision: &mut SwerveDecision,
        role: &mut SwerveCoordRole,
    ) {
        if self.inhibits_turns() {
            decision.mode = ActiveRangeMode::NominalHold;
            decision.cmd_bank_rad = 0.0;
            decision.fim_gain = 0.0;
            if !decision.min_sc3bf_margin.is_finite() {
                decision.min_sc3bf_margin = 0.0;
            }
            *role = SwerveCoordRole::InhibitedByP0;
        } else if !decision.cmd_bank_rad.is_finite() {
            decision.mode = ActiveRangeMode::NominalHold;
            decision.cmd_bank_rad = 0.0;
            decision.fim_gain = 0.0;
            if !decision.min_sc3bf_margin.is_finite() {
                decision.min_sc3bf_margin = 0.0;
            }
        } else {
            decision.cmd_bank_rad =
                clamp_f64(decision.cmd_bank_rad, -EVADE_BANK_RAD, EVADE_BANK_RAD);
        }
    }
}

/// DR-22 Finding F1 Tier B helper: FAR 91.113(e) head-on non-reversing preferred swerve sign.
///
/// Strictly returns `+1.0` (right turn) inside the head-on symmetry band
/// `|alpha_body_rad| < HEAD_ON_ALPHA_BAND_RAD` (`0.07 rad`) across ALL `swerve_ticks: u32`,
/// eliminating asynchronous 12-tick reciprocal sign reversal.
#[inline]
pub(crate) fn compute_preferred_swerve_sign(alpha_body_rad: f64, swerve_ticks: u32) -> f64 {
    let head_on =
        alpha_body_rad.is_finite() && libm::fabs(alpha_body_rad) < HEAD_ON_ALPHA_BAND_RAD;
    if head_on || (swerve_ticks / 12).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// DR-22 Finding F1 Tier A helper: deterministic aircraft-ID stagger slot ownership.
///
/// For any two distinct aircraft IDs (`own_sender_id != peer_sender_id`) and any `coord_ticks`,
/// strictly enforces asymmetric phase separation:
/// `tier_a_own_active_slot(a, b, t) != tier_a_own_active_slot(b, a, t)`.
#[inline]
pub(crate) const fn tier_a_own_active_slot(
    own_sender_id: u8,
    peer_sender_id: u8,
    coord_ticks: u32,
) -> bool {
    let slot = (coord_ticks / STAGGER_SLOT_TICKS) % 2;
    let own_is_lower = own_sender_id < peer_sender_id;
    (slot == 0 && own_is_lower) || (slot == 1 && !own_is_lower)
}

/// Pure internal Tier A cooperative stagger + Ring-0 `P0` arbitration kernel executed on every
/// `CoordinatedActiveRange::step`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_coordination_and_authority(
    own_sender_id: u8,
    coord_ticks: u32,
    head_on: bool,
    peer_hint: Option<PeerCoordHint>,
    straight_margin_0: f64,
    straight_safe_0: bool,
    authority: Ring0AuthorityGate,
    decision: &mut SwerveDecision,
) -> SwerveCoordRole {
    let mut role = SwerveCoordRole::OpticalFar91113Right;
    if let Some(hint) = peer_hint {
        if hint.is_fresh()
            && hint.peer_sender_id != own_sender_id
            && head_on
            && decision.mode == ActiveRangeMode::ActiveSwerve
        {
            if tier_a_own_active_slot(own_sender_id, hint.peer_sender_id, coord_ticks) {
                role = SwerveCoordRole::CooperativeRightProbe;
            } else if straight_safe_0 && straight_margin_0.is_finite() && straight_margin_0 >= 0.0
            {
                decision.mode = ActiveRangeMode::NominalHold;
                decision.cmd_bank_rad = 0.0;
                decision.min_sc3bf_margin = straight_margin_0;
                decision.fim_gain = 0.0;
                role = SwerveCoordRole::YieldHoldForPeerProbe;
            } else {
                role = SwerveCoordRole::CooperativeRightProbe;
            }
        }
    }
    authority.apply_to_decision(decision, &mut role);
    role
}

/// Active multi-ship coordination role selected on the current tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwerveCoordRole {
    /// Inhibited by Ring-0 `P0` / `HAL-P0` (strict zero bank).
    InhibitedByP0,
    /// Cooperative `XVS-MeshDAA` peer is in its staggered probe slot; ownship holds straight
    /// (`phi = 0` verified SC3BF-safe) so peer sees exact `a_int = 0` observability.
    YieldHoldForPeerProbe,
    /// Cooperative `XVS-MeshDAA` active probe slot (`+1` FAR 91.113 right turn).
    CooperativeRightProbe,
    /// Uncooperative / optical-only FAR 91.113 non-reversing `+1` right-turn precedence
    /// (matches `USE_COORD_FIX = TRUE` in `OpenXvsMultiShipRta.tla`).
    OpticalFar91113Right,
}

/// Additive `#![no_std]` coordinated active-range wrapper closing DR-22 Finding F1.
#[derive(Debug, Clone, Copy)]
pub struct CoordinatedActiveRange {
    own_sender_id: u8,
    cfg: ActiveRangeConfig,
    filter: MscRpekfBank,
    seq: u32,
    swerve_ticks: u32,
    coord_ticks: u32,
    was_locked: bool,
    last_role: SwerveCoordRole,
}

impl CoordinatedActiveRange {
    pub fn new(own_sender_id: u8, cfg: ActiveRangeConfig) -> Self {
        Self {
            own_sender_id,
            cfg,
            filter: MscRpekfBank::new(),
            seq: 0,
            swerve_ticks: 0,
            coord_ticks: 0,
            was_locked: false,
            last_role: SwerveCoordRole::OpticalFar91113Right,
        }
    }

    #[inline]
    pub const fn own_sender_id(&self) -> u8 {
        self.own_sender_id
    }

    #[inline]
    pub const fn swerve_ticks(&self) -> u32 {
        self.swerve_ticks
    }

    #[inline]
    pub const fn last_role(&self) -> SwerveCoordRole {
        self.last_role
    }

    /// Reset MSC-RPEKF filter moments while preserving `swerve_ticks` phase offset (for tests/re-acquisition).
    pub fn reset_filter_keep_phase(&mut self) {
        self.filter = MscRpekfBank::new();
        self.was_locked = false;
    }

    /// Execute one deterministic coordinated tick using unmodified frozen `xvs-active-range`
    /// filter, conformal bound, universal `SC3BF-FIM`, and 64-byte IEEE-CRC32 wire encoder.
    pub fn step(
        &mut self,
        own: &OwnshipStateNed,
        meas: &ActiveRangeMeasurement,
        peer_hint: Option<PeerCoordHint>,
        authority: Ring0AuthorityGate,
    ) -> ActiveRangeTelemetry {
        self.seq = self.seq.wrapping_add(1);
        let own_safe = sanitize_ownship(own);
        let inputs_valid = meas.alpha_rad.is_finite()
            && meas.beta_rad.is_finite()
            && meas.dt_s.is_finite()
            && meas.dt_s > 0.0;

        if !inputs_valid {
            if authority.inhibits_turns() {
                self.last_role = SwerveCoordRole::InhibitedByP0;
            }
            return self.fail_safe_hold_packet();
        }

        // If a fresh cooperative peer reports its NED acceleration, pass exact relative
        // acceleration `a_eff = a_own - a_peer` into frozen `MscRpekfBank::update` so both
        // aircraft can execute simultaneous FAR 91.113 right turns with zero 2x range bias!
        let mut own_for_filter = own_safe;
        if let Some(hint) = peer_hint {
            if hint.is_fresh() && hint.peer_sender_id != self.own_sender_id {
                if let Some(pa) = hint.peer_a_ned {
                    if pa[0].is_finite() && pa[1].is_finite() && pa[2].is_finite() {
                        own_for_filter.a_ned = [
                            clamp_f64(own_safe.a_ned[0] - clamp_f64(pa[0], -25.0, 25.0), -50.0, 50.0),
                            clamp_f64(own_safe.a_ned[1] - clamp_f64(pa[1], -25.0, 25.0), -50.0, 50.0),
                            clamp_f64(own_safe.a_ned[2] - clamp_f64(pa[2], -25.0, 25.0), -50.0, 50.0),
                        ];
                    }
                }
            }
        }

        let summary = self.filter.update(&self.cfg, &own_for_filter, meas);
        let r_safe_lower_m = compute_conformal_lower_bound(
            summary.r_hat_m,
            summary.sigma_r_m,
            self.cfg.q_hat_conformal,
        );

        let triad = LosTriadNed::from_angles(summary.alpha_rad, summary.beta_rad);
        let sh = libm::sin(own_safe.heading_rad);
        let ch = libm::cos(own_safe.heading_rad);
        let own_fwd_ned = [ch, sh, 0.0];
        let own_right_ned = [-sh, ch, 0.0];
        let alpha_body_rad = wrap_pi(summary.alpha_rad - own_safe.heading_rad);

        let cbdr_probe_active = meas.track_age >= TAU_MIN_TRACK_AGE
            && meas.size_px.is_finite()
            && meas.size_px < TAU_MIN_SIZE_PX
            && meas.size_rate_px_s.is_finite()
            && meas.size_rate_px_s > 0.0
            && libm::fabs(summary.alpha_dot) < 0.004
            && libm::fabs(summary.beta_dot) < 0.004;

        // DR-22 F1 Fix: NEVER alternate preferred_swerve_sign to -1.0 inside head-on band!
        // FAR 91.113(e) right-of-way locks head-on baseline swerve strictly to +1.0 (right turn),
        // eliminating asynchronous 12-tick reciprocal cancellation (a_rel_az == 0).
        let head_on =
            alpha_body_rad.is_finite() && libm::fabs(alpha_body_rad) < HEAD_ON_ALPHA_BAND_RAD;
        let preferred_swerve_sign =
            compute_preferred_swerve_sign(alpha_body_rad, self.swerve_ticks);

        let swerve_inp = SwerveInputs {
            r_banks: summary.r_banks,
            weights: summary.weights,
            r_hat_m: summary.r_hat_m,
            sigma_r_m: summary.sigma_r_m,
            r_safe_lower_m,
            rho_dot_inv_s: summary.rho_dot_inv_s,
            alpha_dot: summary.alpha_dot,
            beta_dot: summary.beta_dot,
            alpha_body_rad,
            triad,
            own_fwd_ned,
            own_right_ned,
            tas_mps: own_safe.tas_mps,
            v_stall_mps: own_safe.v_stall_mps,
            sigma_bearing_rad: self.cfg.sigma_bearing_rad,
            cbdr_probe_active,
            was_locked: self.was_locked,
            preferred_swerve_sign,
        };

        let mut decision = select_active_swerve(&swerve_inp);
        self.was_locked = decision.now_locked;

        // Evaluate straight-flight SC3BF safety only when Tier A yielding is eligible
        let need_straight_check = match peer_hint {
            Some(hint) => {
                hint.is_fresh()
                    && hint.peer_sender_id != self.own_sender_id
                    && head_on
                    && decision.mode == ActiveRangeMode::ActiveSwerve
                    && !tier_a_own_active_slot(
                        self.own_sender_id,
                        hint.peer_sender_id,
                        self.coord_ticks,
                    )
            }
            None => false,
        };
        let (m0, safe0) = if need_straight_check {
            eval_universal_sc3bf(&swerve_inp, 0.0)
        } else {
            (0.0, false)
        };

        let role = apply_coordination_and_authority(
            self.own_sender_id,
            self.coord_ticks,
            head_on,
            peer_hint,
            m0,
            safe0,
            authority,
            &mut decision,
        );
        self.coord_ticks = self.coord_ticks.wrapping_add(1);
        self.last_role = role;

        if decision.mode == ActiveRangeMode::ActiveSwerve {
            self.swerve_ticks = self.swerve_ticks.wrapping_add(1);
        } else {
            self.swerve_ticks = 0;
        }

        let mut flags = summary.flags;
        if decision.sc3bf_all_safe {
            flags |= FLAG_SC3BF_ALL_SAFE;
        }
        if decision.fov_safe {
            flags |= FLAG_FOV_SAFE;
        }

        let mut weights_f32 = [0.0f32; NUM_BANKS];
        let mut m = 0usize;
        while m < NUM_BANKS {
            weights_f32[m] = summary.weights[m] as f32;
            m += 1;
        }

        let mut pkt = ActiveRangeTelemetry {
            seq: self.seq,
            mode: decision.mode as u8,
            active_bank_idx: summary.active_bank_idx,
            flags,
            r_hat_m: summary.r_hat_m as f32,
            sigma_r_m: summary.sigma_r_m as f32,
            r_safe_lower_m: r_safe_lower_m as f32,
            rho_dot_inv_s: summary.rho_dot_inv_s as f32,
            cmd_bank_rad: decision.cmd_bank_rad as f32,
            min_sc3bf_margin: decision.min_sc3bf_margin as f32,
            fim_gain: decision.fim_gain as f32,
            weights: weights_f32,
            crc32: 0,
        };
        let wire = encode_wire(&pkt);
        pkt.crc32 = u32::from_le_bytes([wire[60], wire[61], wire[62], wire[63]]);
        pkt
    }

    fn fail_safe_hold_packet(&self) -> ActiveRangeTelemetry {
        let mut r_banks = [0.0f64; NUM_BANKS];
        let mut v_banks = [0.0f64; NUM_BANKS];
        let mut r_hat = 0.0f64;
        let mut rho_dot = 0.0f64;
        let mut active_idx = 0usize;
        let mut max_w = -1.0f64;
        let mut weights_f32 = [0.0f32; NUM_BANKS];
        let mut m = 0usize;
        while m < NUM_BANKS {
            let (rm, vm) = self.filter.banks[m].range_and_variance();
            r_banks[m] = rm;
            v_banks[m] = vm;
            let wm = self.filter.weights[m];
            weights_f32[m] = wm as f32;
            if wm > max_w {
                max_w = wm;
                active_idx = m;
            }
            r_hat += wm * rm;
            rho_dot += wm * self.filter.banks[m].x[5];
            m += 1;
        }
        let r_hat = clamp_f64(r_hat, R_FLOOR_M, R_CEIL_M);
        let mut var_tot = 0.0f64;
        let mut m = 0usize;
        while m < NUM_BANKS {
            let dr = r_banks[m] - r_hat;
            var_tot += self.filter.weights[m] * (v_banks[m] + dr * dr);
            m += 1;
        }
        let sigma_r = if var_tot.is_finite() && var_tot > 0.0 {
            libm::sqrt(var_tot)
        } else {
            0.0
        };
        let r_safe_lower_m =
            compute_conformal_lower_bound(r_hat, sigma_r, self.cfg.q_hat_conformal);
        let mut pkt = ActiveRangeTelemetry {
            seq: self.seq,
            mode: ActiveRangeMode::NominalHold as u8,
            active_bank_idx: active_idx as u8,
            flags: 0,
            r_hat_m: r_hat as f32,
            sigma_r_m: sigma_r as f32,
            r_safe_lower_m: r_safe_lower_m as f32,
            rho_dot_inv_s: clamp_f64(rho_dot, -2.0, 2.0) as f32,
            cmd_bank_rad: 0.0f32,
            min_sc3bf_margin: 0.0f32,
            fim_gain: 0.0f32,
            weights: weights_f32,
            crc32: 0,
        };
        let wire = encode_wire(&pkt);
        pkt.crc32 = u32::from_le_bytes([wire[60], wire[61], wire[62], wire[63]]);
        pkt
    }
}

fn sanitize_ownship(own: &OwnshipStateNed) -> OwnshipStateNed {
    let clean = |v: f64, fallback: f64| -> f64 { if v.is_finite() { v } else { fallback } };
    OwnshipStateNed {
        v_ned: [
            clean(own.v_ned[0], 0.0),
            clean(own.v_ned[1], 0.0),
            clean(own.v_ned[2], 0.0),
        ],
        a_ned: [
            clamp_f64(clean(own.a_ned[0], 0.0), -25.0, 25.0),
            clamp_f64(clean(own.a_ned[1], 0.0), -25.0, 25.0),
            clamp_f64(clean(own.a_ned[2], 0.0), -25.0, 25.0),
        ],
        heading_rad: wrap_pi(clean(own.heading_rad, 0.0)),
        tas_mps: clamp_f64(clean(own.tas_mps, 25.0), 1.0, 250.0),
        v_stall_mps: clamp_f64(clean(own.v_stall_mps, 14.5), 1.0, 100.0),
    }
}
