//! Canonical 5-state ASTM F3269-21 Simplex RTA (`SP1_LIVEDECK_DESIGN.md` §4.1.3).
//!
//! Total-function integer state machine in SI units (`i64` `urad` and `mm`), `#![forbid(unsafe_code)]`,
//! checked/saturating arithmetic only (`#![deny(clippy::arithmetic_side_effects)]`).

use crate::mltl::OddMltlMonitor;
use crate::odd_generated::{
    CONSEC_DROPOUT_MAX_TICKS, DWELL_TICKS, HORIZON_AL_URAD, MIN_HEALTHY_CAMS, P0_RECOVERY_TICKS,
    V_HAL_MM, V_VAL_MM, WINDOW_DROPOUT_20_MAX_TICKS,
};

/// Canonical 5-state priority ordering (`P0 < P1 < P2 < P3 < P4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SimplexMode {
    InvalidInhibit = 0,
    FallbackD0Safe = 1,
    DegradedCoast = 2,
    Isolated5Cam = 3,
    NominalMl = 4,
}

/// Pure SI integer (`i64 urad` / `mm`) input contract for a single 10 Hz Simplex RTA tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct RtaTickInput {
    pub cam_health_mask: u8,
    pub ood_exceeded_mask: u8,
    pub horizon_pl_urad: i64,
    pub horizon_al_urad: i64,
    pub v_vpl_mm: i64,
    pub v_val_mm: i64,
    pub v_hpl_mm: i64,
    pub v_hal_mm: i64,
    pub epipolar_max_urad: [i64; 6],
    pub signed_horizon_residual_urad: i64,
    pub runway_width_delta_mm: i64,
    pub spoof_alarm: bool,
    pub ml_detector_fault: bool,
    pub ood_global_exceeded: bool,
    pub runway_width_fault: bool,
    pub transient_dropout: bool,
    pub approach_active: bool,
    pub go_around_ack_reset: bool,
    pub quantize_fault: bool,
}

impl Default for RtaTickInput {
    fn default() -> Self {
        Self {
            cam_health_mask: 0x3F,
            ood_exceeded_mask: 0x00,
            horizon_pl_urad: 0,
            horizon_al_urad: HORIZON_AL_URAD,
            v_vpl_mm: 0,
            v_val_mm: V_VAL_MM,
            v_hpl_mm: 0,
            v_hal_mm: V_HAL_MM,
            epipolar_max_urad: [0; 6],
            signed_horizon_residual_urad: 0,
            runway_width_delta_mm: 0,
            spoof_alarm: false,
            ml_detector_fault: false,
            ood_global_exceeded: false,
            runway_width_fault: false,
            transient_dropout: false,
            approach_active: false,
            go_around_ack_reset: false,
            quantize_fault: false,
        }
    }
}

/// Deterministic C-ABI output of a single 10 Hz Simplex RTA tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct RtaTickOutput {
    pub mode: SimplexMode,
    pub raw_target: SimplexMode,
    pub healthy_cam_mask: u8,
    pub active_healthy_cams: u8,
    pub consec_dropout_ticks: u32,
    pub window_dropout_20: u8,
    pub recovery_ticks: u32,
    pub dwell_ticks: u32,
    pub mandatory_go_around: bool,
    pub mltl_critical_violation: bool,
    pub blank_conformal_hud: bool,
    pub inhibit_avoidance_turns: bool,
    pub tick_index: u64,
}

/// Stateful deterministic ASTM F3269-21 Simplex RTA + past-time MLTL Visual-ODD monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplexRta {
    mode: SimplexMode,
    consec_dropout_ticks: u32,
    recovery_ticks: u32,
    dwell_ticks: u32,
    mandatory_go_around: bool,
    tick_index: u64,
    mltl: OddMltlMonitor,
}

impl Default for SimplexRta {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplexRta {
    pub const fn new() -> Self {
        Self {
            mode: SimplexMode::NominalMl,
            consec_dropout_ticks: 0,
            recovery_ticks: 0,
            dwell_ticks: 0,
            mandatory_go_around: false,
            tick_index: 0,
            mltl: OddMltlMonitor::new(),
        }
    }

    pub const fn mode(&self) -> SimplexMode {
        self.mode
    }

    pub const fn consec_dropout_ticks(&self) -> u32 {
        self.consec_dropout_ticks
    }

    pub const fn recovery_ticks(&self) -> u32 {
        self.recovery_ticks
    }

    pub const fn dwell_ticks(&self) -> u32 {
        self.dwell_ticks
    }

    pub const fn mandatory_go_around(&self) -> bool {
        self.mandatory_go_around
    }

    pub const fn tick_index(&self) -> u64 {
        self.tick_index
    }

    /// Construct a `SimplexRta` at an explicit internal state for formal verification (`Kani`).
    #[cfg(kani)]
    pub(crate) const fn from_parts_for_proof(
        mode: SimplexMode,
        consec_dropout_ticks: u32,
        recovery_ticks: u32,
        dwell_ticks: u32,
        mandatory_go_around: bool,
    ) -> Self {
        Self {
            mode,
            consec_dropout_ticks,
            recovery_ticks,
            dwell_ticks,
            mandatory_go_around,
            tick_index: 0,
            mltl: OddMltlMonitor::new(),
        }
    }

    /// Execute one total-function 10 Hz Simplex RTA transition (`SP1_LIVEDECK_DESIGN.md` §4.1.3).
    pub fn step(&mut self, inp: &RtaTickInput) -> RtaTickOutput {
        // 1. Evaluate past-time MLTL rules & effective 6-bit camera mask
        let mltl_out = self.mltl.eval_tick(
            inp.cam_health_mask,
            inp.ood_exceeded_mask,
            &inp.epipolar_max_urad,
            inp.spoof_alarm,
            inp.transient_dropout,
        );

        // 2. Mode-independent dropout duration & non-resettable Mandatory Go-Around latch
        let next_consec_dropout_ticks = if inp.transient_dropout {
            self.consec_dropout_ticks.saturating_add(1)
        } else {
            0
        };

        let mut go_around_latched = self.mandatory_go_around;
        if inp.approach_active {
            if inp.quantize_fault
                || inp.v_vpl_mm > inp.v_val_mm
                || inp.v_hpl_mm > inp.v_hal_mm
                || inp.runway_width_fault
            {
                go_around_latched = true;
            }
        } else if inp.go_around_ack_reset {
            go_around_latched = false;
        }

        // 3. Total Raw Target Function `raw_target` covering EVERY n_cams in 0..=6
        let raw_target = if inp.quantize_fault
            || mltl_out.n_cams < MIN_HEALTHY_CAMS
            || inp.horizon_pl_urad > inp.horizon_al_urad
            || go_around_latched
            || mltl_out.critical_violation
            || next_consec_dropout_ticks > CONSEC_DROPOUT_MAX_TICKS
            || mltl_out.window_dropout_20 > WINDOW_DROPOUT_20_MAX_TICKS
        {
            SimplexMode::InvalidInhibit
        } else if inp.spoof_alarm
            || mltl_out.spoof_tripped
            || inp.ml_detector_fault
            || inp.ood_global_exceeded
            || mltl_out.n_cams == 4
        {
            SimplexMode::FallbackD0Safe
        } else if inp.transient_dropout {
            SimplexMode::DegradedCoast
        } else if mltl_out.n_cams == 5 {
            SimplexMode::Isolated5Cam
        } else {
            SimplexMode::NominalMl
        };

        // 4. Explicit P0 healthy-recovery predicate
        let is_p0_healthy = matches!(
            raw_target,
            SimplexMode::FallbackD0Safe | SimplexMode::Isolated5Cam | SimplexMode::NominalMl
        ) && !inp.transient_dropout;

        // 5. Total Next-State & Counter Transition Table `(self.mode, raw_target) -> next_state`
        let (next_mode, next_recovery, next_dwell) = if (raw_target as u8) < (self.mode as u8) {
            // Immediate 0-tick downward preemption
            (raw_target, 0_u32, 0_u32)
        } else {
            match self.mode {
                SimplexMode::InvalidInhibit => {
                    if !is_p0_healthy {
                        (SimplexMode::InvalidInhibit, 0_u32, 0_u32)
                    } else {
                        let r = self.recovery_ticks.saturating_add(1);
                        if r >= P0_RECOVERY_TICKS {
                            // Strictly exits P0 into P1 (never directly into P2, P3, or P4)
                            (SimplexMode::FallbackD0Safe, 0_u32, 0_u32)
                        } else {
                            (SimplexMode::InvalidInhibit, r, 0_u32)
                        }
                    }
                }
                SimplexMode::FallbackD0Safe => {
                    if matches!(
                        raw_target,
                        SimplexMode::FallbackD0Safe | SimplexMode::DegradedCoast
                    ) {
                        // NEVER climbs upward into transient P2 (DegradedCoast)
                        (SimplexMode::FallbackD0Safe, 0_u32, 0_u32)
                    } else {
                        // raw_target in {Isolated5Cam (P3), NominalMl (P4)}
                        let d = self.dwell_ticks.saturating_add(1);
                        if d >= DWELL_TICKS {
                            (raw_target, 0_u32, 0_u32)
                        } else {
                            (SimplexMode::FallbackD0Safe, 0_u32, d)
                        }
                    }
                }
                SimplexMode::DegradedCoast => {
                    if raw_target == SimplexMode::DegradedCoast {
                        (SimplexMode::DegradedCoast, 0_u32, 0_u32)
                    } else {
                        // Dropout cleared before tick 11 -> enter P1 with dwell_ticks = 1 (0 extra ticks in P2)
                        (SimplexMode::FallbackD0Safe, 0_u32, 1_u32)
                    }
                }
                SimplexMode::Isolated5Cam => {
                    if raw_target == SimplexMode::Isolated5Cam {
                        (SimplexMode::Isolated5Cam, 0_u32, 0_u32)
                    } else {
                        // raw_target == NominalMl (P4)
                        let d = self.dwell_ticks.saturating_add(1);
                        if d >= DWELL_TICKS {
                            (SimplexMode::NominalMl, 0_u32, 0_u32)
                        } else {
                            (SimplexMode::Isolated5Cam, 0_u32, d)
                        }
                    }
                }
                SimplexMode::NominalMl => (SimplexMode::NominalMl, 0_u32, 0_u32),
            }
        };

        self.mode = next_mode;
        self.consec_dropout_ticks = next_consec_dropout_ticks;
        self.recovery_ticks = next_recovery;
        self.dwell_ticks = next_dwell;
        self.mandatory_go_around = go_around_latched;
        self.tick_index = self.tick_index.saturating_add(1);

        let is_p0 = next_mode == SimplexMode::InvalidInhibit;
        RtaTickOutput {
            mode: next_mode,
            raw_target,
            healthy_cam_mask: mltl_out.healthy_mask,
            active_healthy_cams: mltl_out.n_cams,
            consec_dropout_ticks: next_consec_dropout_ticks,
            window_dropout_20: mltl_out.window_dropout_20,
            recovery_ticks: next_recovery,
            dwell_ticks: next_dwell,
            mandatory_go_around: go_around_latched,
            mltl_critical_violation: mltl_out.critical_violation,
            blank_conformal_hud: is_p0,
            inhibit_avoidance_turns: is_p0,
            tick_index: self.tick_index,
        }
    }
}
