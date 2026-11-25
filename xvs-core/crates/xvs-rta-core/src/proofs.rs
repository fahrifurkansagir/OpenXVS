//! Five mandatory Kani bounded model-checking proof harnesses (`SP1_LIVEDECK_DESIGN.md` §4.1.4).
//!
//! Executed via local WSL2 Ubuntu `cargo kani -p xvs-rta-core` (`Option A` approved 2026-11-20).

use crate::mltl::BoolRing64;
use crate::odd_generated::{
    CONSEC_DROPOUT_MAX_TICKS, DWELL_TICKS, HORIZON_AL_URAD, P0_RECOVERY_TICKS, V_HAL_MM, V_VAL_MM,
    WINDOW_DROPOUT_20_MAX_TICKS,
};
use crate::quantize::{
    build_rta_input_from_f64, quantize_nonneg_upper_i64, quantize_signed_bounded_i64,
    RtaFloatTelemetry, MAX_VALID_ANGLE_URAD, MAX_VALID_DIST_MM, M_TO_MM_SCALE, RAD_TO_URAD_SCALE,
};
use crate::simplex::{RtaTickInput, SimplexMode, SimplexRta};

fn any_mode() -> SimplexMode {
    match kani::any::<u8>() % 5 {
        0 => SimplexMode::InvalidInhibit,
        1 => SimplexMode::FallbackD0Safe,
        2 => SimplexMode::DegradedCoast,
        3 => SimplexMode::Isolated5Cam,
        _ => SimplexMode::NominalMl,
    }
}

fn any_tick_input() -> RtaTickInput {
    RtaTickInput {
        cam_health_mask: kani::any::<u8>() & 0x3F,
        ood_exceeded_mask: kani::any::<u8>() & 0x3F,
        horizon_pl_urad: kani::any::<i64>(),
        horizon_al_urad: HORIZON_AL_URAD,
        v_vpl_mm: kani::any::<i64>(),
        v_val_mm: V_VAL_MM,
        v_hpl_mm: kani::any::<i64>(),
        v_hal_mm: V_HAL_MM,
        epipolar_max_urad: [
            kani::any::<i64>(),
            kani::any::<i64>(),
            kani::any::<i64>(),
            kani::any::<i64>(),
            kani::any::<i64>(),
            kani::any::<i64>(),
        ],
        signed_horizon_residual_urad: kani::any::<i64>(),
        runway_width_delta_mm: kani::any::<i64>(),
        spoof_alarm: kani::any::<bool>(),
        ml_detector_fault: kani::any::<bool>(),
        ood_global_exceeded: kani::any::<bool>(),
        runway_width_fault: kani::any::<bool>(),
        transient_dropout: kani::any::<bool>(),
        approach_active: kani::any::<bool>(),
        go_around_ack_reset: kani::any::<bool>(),
        quantize_fault: kani::any::<bool>(),
    }
}

/// Harness 1: Arbitrary `f64::from_bits(u64)` on non-negative & signed quantizers AND
/// float-derived alarm stats through `build_rta_input_from_f64` never panics and fails closed to P0.
#[kani::proof]
#[kani::unwind(8)]
fn prove_quantize_fail_closed_no_panic() {
    let raw_nonneg = f64::from_bits(kani::any::<u64>());
    let q_nn = quantize_nonneg_upper_i64(raw_nonneg, RAD_TO_URAD_SCALE, MAX_VALID_ANGLE_URAD);
    if !raw_nonneg.is_finite() || raw_nonneg < 0.0 {
        assert!(q_nn.is_err());
    } else if raw_nonneg == 0.0 {
        assert_eq!(q_nn, Ok(0));
    } else if let Ok(v) = q_nn {
        assert!(v >= 0 && v <= MAX_VALID_ANGLE_URAD);
    }

    let raw_signed = f64::from_bits(kani::any::<u64>());
    let q_sg = quantize_signed_bounded_i64(
        raw_signed,
        M_TO_MM_SCALE,
        -MAX_VALID_DIST_MM,
        MAX_VALID_DIST_MM,
    );
    if !raw_signed.is_finite() {
        assert!(q_sg.is_err());
    } else if raw_signed == 0.0 {
        assert_eq!(q_sg, Ok(0));
    } else if let Ok(v) = q_sg {
        assert!(v >= -MAX_VALID_DIST_MM && v <= MAX_VALID_DIST_MM);
    }

    // Valid negative signed residual passes without false fault
    let neg_res = quantize_signed_bounded_i64(-1.5, M_TO_MM_SCALE, -500_000, 500_000);
    assert_eq!(neg_res, Ok(-1_500));

    // Full boundary conversion with arbitrary float bits in key metrics & alarm stats
    let cusum_stat = f64::from_bits(kani::any::<u64>());
    let ood_stat = f64::from_bits(kani::any::<u64>());
    let telem = RtaFloatTelemetry {
        horizon_pl_rad: raw_nonneg,
        horizon_al_rad: 0.008727,
        v_vpl_m: 4.0,
        v_val_m: 10.0,
        v_hpl_m: 12.0,
        v_hal_m: 40.0,
        epipolar_max_rad: [0.0005; 6],
        signed_horizon_residual_rad: -0.001,
        runway_width_delta_m: raw_signed,
        runway_width_delta_limit_m: 5.0,
        cusum_stat,
        cusum_thr: 10.530,
        ood_global_score: ood_stat,
        ood_global_thr: 1.0,
        ood_cam_scores: [0.1; 6],
        ood_cam_thr: 1.0,
        cam_health_mask: 0x3F,
        ml_detector_fault: false,
        transient_dropout: false,
        approach_active: false,
        go_around_ack_reset: false,
    };
    let inp = build_rta_input_from_f64(&telem);
    let any_nonfinite = !raw_nonneg.is_finite()
        || !raw_signed.is_finite()
        || !cusum_stat.is_finite()
        || !ood_stat.is_finite();
    if any_nonfinite {
        assert!(inp.quantize_fault);
        let mut rta = SimplexRta::new();
        let out = rta.step(&inp);
        assert_eq!(out.mode, SimplexMode::InvalidInhibit);
        assert!(out.blank_conformal_hud);
        assert!(out.inhibit_avoidance_turns);
    }
}

/// Harness 2: Priority dominance & immediate 0-tick downward preemption across all reachable states.
#[kani::proof]
#[kani::unwind(8)]
fn prove_priority_dominance_and_0tick_preemption() {
    let prev_mode = any_mode();
    let consec = kani::any::<u32>();
    let rec = kani::any::<u32>();
    let dwell = kani::any::<u32>();
    let go_around = kani::any::<bool>();
    let mut rta = SimplexRta::from_parts_for_proof(prev_mode, consec, rec, dwell, go_around);

    let inp = any_tick_input();
    let out = rta.step(&inp);

    if (out.raw_target as u8) < (prev_mode as u8) {
        assert_eq!(out.mode, out.raw_target);
        assert_eq!(out.recovery_ticks, 0);
        assert_eq!(out.dwell_ticks, 0);
    }
    if inp.quantize_fault {
        assert_eq!(out.mode, SimplexMode::InvalidInhibit);
    }
    if out.mode == SimplexMode::InvalidInhibit {
        assert!(out.blank_conformal_hud);
        assert!(out.inhibit_avoidance_turns);
    }
}

/// Harness 3: Sustained dropout > 10 ticks AND intermittent >= 11/20 (55%) dropout duty cycle lock P0.
#[kani::proof]
#[kani::unwind(24)]
fn prove_coast_timeout_and_intermittent_dropout_lock() {
    let mut rta = SimplexRta::new();
    let mut drop_hist = [false; 22];
    let mut consec_expected: u32 = 0;

    let mut t: usize = 0;
    while t < 22 {
        let drop_now: bool = kani::any();
        drop_hist[t] = drop_now;
        consec_expected = if drop_now {
            consec_expected.saturating_add(1)
        } else {
            0
        };

        // Compute independent reference 20-tick sliding window count
        let win_start = t.saturating_sub(19);
        let mut win_cnt: u8 = 0;
        let mut j = win_start;
        while j <= t {
            if drop_hist[j] {
                win_cnt = win_cnt.saturating_add(1);
            }
            j = j.saturating_add(1);
        }

        let mut inp = RtaTickInput::default();
        inp.transient_dropout = drop_now;
        let out = rta.step(&inp);

        assert_eq!(out.consec_dropout_ticks, consec_expected);
        assert_eq!(out.window_dropout_20, win_cnt);

        if out.mode == SimplexMode::DegradedCoast {
            assert!(out.consec_dropout_ticks >= 1);
            assert!(out.consec_dropout_ticks <= CONSEC_DROPOUT_MAX_TICKS);
            assert!(out.window_dropout_20 <= WINDOW_DROPOUT_20_MAX_TICKS);
        }
        if consec_expected > CONSEC_DROPOUT_MAX_TICKS || win_cnt > WINDOW_DROPOUT_20_MAX_TICKS {
            assert_eq!(out.mode, SimplexMode::InvalidInhibit);
        }
        t = t.saturating_add(1);
    }
}

/// Harness 4: Staged upward recovery hysteresis (`P0 -> P1` requires >= 30 healthy ticks;
/// `P1 -> P4` requires >= 20 additional dwell ticks -> >= 50 total; never jumps P0 -> P3/P4;
/// never climbs upward from P1 into P2).
#[kani::proof]
#[kani::unwind(54)]
fn prove_staged_recovery_hysteresis() {
    let mut rta = SimplexRta::from_parts_for_proof(SimplexMode::InvalidInhibit, 0, 0, 0, false);

    let mut consec_p0_healthy: u32 = 0;
    let mut consec_p1_dwell: u32 = 0;
    let mut total_healthy_since_p0: u32 = 0;

    let mut t: usize = 0;
    while t < 52 {
        let prev_mode = rta.mode();
        let fault_p0: bool = kani::any();
        let fault_p1: bool = kani::any();
        let dropout: bool = kani::any();

        let mut inp = RtaTickInput::default();
        if fault_p0 {
            inp.horizon_pl_urad = HORIZON_AL_URAD.saturating_add(10);
        }
        inp.ml_detector_fault = fault_p1;
        inp.transient_dropout = dropout;

        let out = rta.step(&inp);

        // From P0, can ONLY remain in P0 or transition to P1 (never P2/P3/P4)
        if prev_mode == SimplexMode::InvalidInhibit {
            assert!(matches!(
                out.mode,
                SimplexMode::InvalidInhibit | SimplexMode::FallbackD0Safe
            ));
            let is_p0_h = matches!(
                out.raw_target,
                SimplexMode::FallbackD0Safe | SimplexMode::Isolated5Cam | SimplexMode::NominalMl
            ) && !inp.transient_dropout;
            if is_p0_h {
                consec_p0_healthy = consec_p0_healthy.saturating_add(1);
                total_healthy_since_p0 = total_healthy_since_p0.saturating_add(1);
            } else {
                consec_p0_healthy = 0;
                total_healthy_since_p0 = 0;
            }
            if out.mode == SimplexMode::FallbackD0Safe {
                assert!(consec_p0_healthy >= P0_RECOVERY_TICKS);
                consec_p0_healthy = 0;
                consec_p1_dwell = 0;
            }
        } else if prev_mode == SimplexMode::FallbackD0Safe {
            // P1 NEVER climbs upward into transient P2 (DegradedCoast)
            assert_ne!(out.mode, SimplexMode::DegradedCoast);
            if out.raw_target == SimplexMode::NominalMl {
                consec_p1_dwell = consec_p1_dwell.saturating_add(1);
                total_healthy_since_p0 = total_healthy_since_p0.saturating_add(1);
            } else if out.mode == SimplexMode::InvalidInhibit {
                consec_p0_healthy = 0;
                consec_p1_dwell = 0;
                total_healthy_since_p0 = 0;
            } else {
                consec_p1_dwell = 0;
            }
            if out.mode == SimplexMode::NominalMl {
                assert!(consec_p1_dwell >= DWELL_TICKS);
                assert!(total_healthy_since_p0 >= P0_RECOVERY_TICKS.saturating_add(DWELL_TICKS));
            }
        } else if out.mode == SimplexMode::InvalidInhibit {
            consec_p0_healthy = 0;
            consec_p1_dwell = 0;
            total_healthy_since_p0 = 0;
        }

        t = t.saturating_add(1);
    }
}

/// Harness 5: Mandatory go-around stays latched during `approach_active` regardless of fault
/// disappearance or `go_around_ack_reset`, and past-time MLTL operators never panic across arbitrary ticks.
#[kani::proof]
#[kani::unwind(70)]
fn prove_mandatory_go_around_and_mltl_safety() {
    let mut rta = SimplexRta::new();

    // Trip mandatory go-around via any approach fault source
    let mut trip = RtaTickInput::default();
    trip.approach_active = true;
    match kani::any::<u8>() % 4 {
        0 => trip.v_vpl_mm = V_VAL_MM.saturating_add(1),
        1 => trip.v_hpl_mm = V_HAL_MM.saturating_add(1),
        2 => trip.runway_width_fault = true,
        _ => trip.quantize_fault = true,
    }
    let out0 = rta.step(&trip);
    assert!(out0.mandatory_go_around);
    assert_eq!(out0.mode, SimplexMode::InvalidInhibit);

    // Even with all faults cleared AND arbitrary go_around_ack_reset, approach_active keeps latch locked
    let mut k: usize = 0;
    while k < 4 {
        let mut clear_in = RtaTickInput::default();
        clear_in.approach_active = true;
        clear_in.go_around_ack_reset = kani::any::<bool>();
        let out_k = rta.step(&clear_in);
        assert!(out_k.mandatory_go_around);
        assert_eq!(out_k.mode, SimplexMode::InvalidInhibit);
        k = k.saturating_add(1);
    }

    // Exercise past-time MLTL operators (H, O, S, count_true_last) for zero panic across bounds
    let mut p_ring = BoolRing64::new();
    let mut q_ring = BoolRing64::new();
    let mut i: usize = 0;
    while i < 6 {
        p_ring.push(kani::any::<bool>());
        q_ring.push(kani::any::<bool>());
        i = i.saturating_add(1);
    }
    let n: usize = kani::any::<usize>() % 64;
    let a: usize = kani::any::<usize>() % 64;
    let b: usize = kani::any::<usize>() % 64;
    let _h = p_ring.historically(n, kani::any::<bool>());
    let _o = q_ring.once(n);
    let _s = BoolRing64::since(&p_ring, &q_ring, a, b);
    let _c = p_ring.count_true_last(20);
}
