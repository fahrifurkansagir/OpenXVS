use proptest::prelude::*;
use xvs_rta_core::{
    build_rta_input_from_f64, quantize_nonneg_upper_i64, quantize_signed_bounded_i64, BoolRing64,
    QuantFault, RtaFloatTelemetry, RtaTickInput, SimplexMode, SimplexRta,
    CONSEC_DROPOUT_MAX_TICKS, DWELL_TICKS, EPIPOLAR_CAM_MAX_URAD, HORIZON_AL_URAD,
    ODD_YAML_SHA256, P0_RECOVERY_TICKS, V_HAL_MM, V_VAL_MM, WINDOW_DROPOUT_20_MAX_TICKS,
};

fn nominal_float_telemetry() -> RtaFloatTelemetry {
    RtaFloatTelemetry {
        horizon_pl_rad: 0.002,
        horizon_al_rad: 0.008727,
        v_vpl_m: 4.0,
        v_val_m: 10.0,
        v_hpl_m: 12.0,
        v_hal_m: 40.0,
        epipolar_max_rad: [0.0005; 6],
        signed_horizon_residual_rad: -0.0012,
        runway_width_delta_m: -1.5,
        runway_width_delta_limit_m: 5.0,
        cusum_stat: 2.0,
        cusum_thr: 10.530,
        ood_global_score: 0.5,
        ood_global_thr: 1.0,
        ood_cam_scores: [0.2; 6],
        ood_cam_thr: 1.0,
        cam_health_mask: 0x3F,
        ml_detector_fault: false,
        transient_dropout: false,
        approach_active: false,
        go_around_ack_reset: false,
    }
}

fn nominal_input() -> RtaTickInput {
    RtaTickInput::default()
}

#[test]
fn test_odd_generated_sha256_and_si_constants() {
    assert_eq!(ODD_YAML_SHA256.len(), 64);
    assert_eq!(EPIPOLAR_CAM_MAX_URAD, 6_109);
    assert_eq!(HORIZON_AL_URAD, 8_727);
    assert_eq!(V_VAL_MM, 10_000);
    assert_eq!(V_HAL_MM, 40_000);
    assert_eq!(CONSEC_DROPOUT_MAX_TICKS, 10);
    assert_eq!(WINDOW_DROPOUT_20_MAX_TICKS, 10);
    assert_eq!(P0_RECOVERY_TICKS, 30);
    assert_eq!(DWELL_TICKS, 20);
}

#[test]
fn test_quantize_nan_inf_neg_zero_and_signed_residuals() {
    // Review Focus #2: -0.0 must quantize to 0 (never false-fault val < 0.0)
    assert_eq!(quantize_nonneg_upper_i64(-0.0, 1_000_000.0, 100_000_000), Ok(0));
    assert_eq!(quantize_nonneg_upper_i64(0.0, 1_000_000.0, 100_000_000), Ok(0));
    // Conservative ceiling on non-negative bounds
    assert_eq!(quantize_nonneg_upper_i64(10.0001, 1_000.0, 100_000), Ok(10_001));
    // Subnormal positive value rounds up safely to 1 (finite, never panics)
    let subnormal = f64::from_bits(1);
    assert_eq!(quantize_nonneg_upper_i64(subnormal, 1_000_000.0, 100_000), Ok(1));

    // Non-finite, negative, signaling NaN, and out-of-range values MUST fail closed
    let s_nan = f64::from_bits(0x7FF0_0000_0000_0001);
    assert!(matches!(quantize_nonneg_upper_i64(f64::NAN, 1e6, 100_000), Err(QuantFault::NonFinite)));
    assert!(matches!(quantize_nonneg_upper_i64(s_nan, 1e6, 100_000), Err(QuantFault::NonFinite)));
    assert!(matches!(quantize_nonneg_upper_i64(f64::INFINITY, 1e6, 100_000), Err(QuantFault::NonFinite)));
    assert!(matches!(quantize_nonneg_upper_i64(f64::NEG_INFINITY, 1e6, 100_000), Err(QuantFault::NonFinite)));
    assert!(matches!(quantize_nonneg_upper_i64(-1e-9, 1e6, 100_000), Err(QuantFault::NegativeInput)));
    assert!(matches!(quantize_nonneg_upper_i64(200.0, 1e3, 100_000), Err(QuantFault::OutOfRange)));

    // Valid negative physical residuals pass signed bounded quantizer cleanly
    assert_eq!(quantize_signed_bounded_i64(-1.5, 1_000.0, -500_000, 500_000), Ok(-1_500));
    assert_eq!(quantize_signed_bounded_i64(-0.0, 1_000.0, -500_000, 500_000), Ok(0));
    assert!(matches!(
        quantize_signed_bounded_i64(f64::NAN, 1_000.0, -500_000, 500_000),
        Err(QuantFault::NonFinite)
    ));
    assert!(matches!(
        quantize_signed_bounded_i64(-600.0, 1_000.0, -500_000, 500_000),
        Err(QuantFault::OutOfRange)
    ));

    // Nominal float telemetry with valid negative residuals produces clean NominalMl
    let nom_f = nominal_float_telemetry();
    let inp_nom = build_rta_input_from_f64(&nom_f);
    assert!(!inp_nom.quantize_fault);
    assert_eq!(inp_nom.runway_width_delta_mm, -1_500);
    assert_eq!(inp_nom.signed_horizon_residual_urad, -1_200);
    assert!(!inp_nom.spoof_alarm);
    assert!(!inp_nom.ood_global_exceeded);
    let mut rta = SimplexRta::new();
    assert_eq!(rta.step(&inp_nom).mode, SimplexMode::NominalMl);

    // NaN in cusum_stat OR ood_global_score MUST set quantize_fault = true and force P0 at 0 ticks
    let mut nan_cusum = nominal_float_telemetry();
    nan_cusum.cusum_stat = f64::NAN;
    let inp_nan_cusum = build_rta_input_from_f64(&nan_cusum);
    assert!(inp_nan_cusum.quantize_fault);
    assert!(inp_nan_cusum.spoof_alarm);
    let mut rta_c = SimplexRta::new();
    assert_eq!(rta_c.step(&inp_nan_cusum).mode, SimplexMode::InvalidInhibit);

    let mut nan_ood = nominal_float_telemetry();
    nan_ood.ood_global_score = s_nan;
    let inp_nan_ood = build_rta_input_from_f64(&nan_ood);
    assert!(inp_nan_ood.quantize_fault);
    assert!(inp_nan_ood.ood_global_exceeded);
    let mut rta_o = SimplexRta::new();
    assert_eq!(rta_o.step(&inp_nan_ood).mode, SimplexMode::InvalidInhibit);
}

#[test]
fn test_total_transition_table_all_cam_counts_and_recovery_chatter() {
    // 6 cams -> P4, 5 cams -> P3, 4 cams -> P1, 3 cams -> P0 (0-tick downward preemption)
    let mut rta = SimplexRta::new();
    let out6 = rta.step(&nominal_input());
    assert_eq!(out6.mode, SimplexMode::NominalMl);
    assert_eq!(out6.active_healthy_cams, 6);

    let mut in5 = nominal_input();
    in5.cam_health_mask = 0b111101; // Cam R faulted -> 5 healthy cams
    let out5 = rta.step(&in5);
    assert_eq!(out5.mode, SimplexMode::Isolated5Cam);
    assert_eq!(out5.active_healthy_cams, 5);

    let mut in4 = nominal_input();
    in4.cam_health_mask = 0b111100; // 4 healthy cams -> P1 (6->5->4 degradation)
    let out4 = rta.step(&in4);
    assert_eq!(out4.mode, SimplexMode::FallbackD0Safe);
    assert_eq!(out4.active_healthy_cams, 4);

    let mut in3 = nominal_input();
    in3.cam_health_mask = 0b000111; // <4 healthy cams -> P0 at 0 ticks
    let out3 = rta.step(&in3);
    assert_eq!(out3.mode, SimplexMode::InvalidInhibit);
    assert!(out3.blank_conformal_hud);
    assert!(out3.inhibit_avoidance_turns);

    // Review Focus #1: Test exact 30-tick P0->P1 and 20-tick P1->P4 with single-tick chatter at 29 & 19!
    let mut rta2 = SimplexRta::new();
    let mut pl_fault = nominal_input();
    pl_fault.horizon_pl_urad = HORIZON_AL_URAD + 1;
    assert_eq!(rta2.step(&pl_fault).mode, SimplexMode::InvalidInhibit);

    // 29 healthy ticks -> still P0 with recovery_ticks == 29
    for t in 1..=29 {
        let out = rta2.step(&nominal_input());
        assert_eq!(out.mode, SimplexMode::InvalidInhibit);
        assert_eq!(out.recovery_ticks, t);
    }
    // Single fault tick at recovery_ticks == 29 MUST reset recovery_ticks cleanly to 0
    let out_reset = rta2.step(&pl_fault);
    assert_eq!(out_reset.mode, SimplexMode::InvalidInhibit);
    assert_eq!(out_reset.recovery_ticks, 0);

    // Now 29 healthy ticks stay P0, 30th transitions strictly P0 -> P1
    for t in 1..=29 {
        let out = rta2.step(&nominal_input());
        assert_eq!(out.mode, SimplexMode::InvalidInhibit);
        assert_eq!(out.recovery_ticks, t);
    }
    let out_p1 = rta2.step(&nominal_input());
    assert_eq!(out_p1.mode, SimplexMode::FallbackD0Safe);
    assert_eq!(out_p1.recovery_ticks, 0);
    assert_eq!(out_p1.dwell_ticks, 0);

    // 19 healthy ticks in P1 -> dwell_ticks == 19, still P1
    for t in 1..=19 {
        let out = rta2.step(&nominal_input());
        assert_eq!(out.mode, SimplexMode::FallbackD0Safe);
        assert_eq!(out.dwell_ticks, t);
    }
    // Single P1 fault tick (e.g. ml_detector_fault) at dwell_ticks == 19 MUST reset dwell_ticks to 0
    let mut ml_fault = nominal_input();
    ml_fault.ml_detector_fault = true;
    let out_dwell_reset = rta2.step(&ml_fault);
    assert_eq!(out_dwell_reset.mode, SimplexMode::FallbackD0Safe);
    assert_eq!(out_dwell_reset.dwell_ticks, 0);

    // Now 19 healthy ticks stay P1, 20th ascends cleanly P1 -> P4
    for t in 1..=19 {
        let out = rta2.step(&nominal_input());
        assert_eq!(out.mode, SimplexMode::FallbackD0Safe);
        assert_eq!(out.dwell_ticks, t);
    }
    let out_p4 = rta2.step(&nominal_input());
    assert_eq!(out_p4.mode, SimplexMode::NominalMl);
    assert_eq!(out_p4.dwell_ticks, 0);
}

#[test]
fn test_sustained_and_intermittent_dropout_50_vs_55_pct() {
    let mut drop_in = nominal_input();
    drop_in.transient_dropout = true;

    // 1. Sustained dropout: ticks 1..=10 stay P2; tick 11..=15 lock P0 continuously
    let mut rta_sus = SimplexRta::new();
    for t in 1..=10 {
        let out = rta_sus.step(&drop_in);
        assert_eq!(out.mode, SimplexMode::DegradedCoast);
        assert_eq!(out.consec_dropout_ticks, t);
    }
    for t in 11..=15 {
        let out = rta_sus.step(&drop_in);
        assert_eq!(out.mode, SimplexMode::InvalidInhibit);
        assert_eq!(out.consec_dropout_ticks, t);
    }

    // 2. Review Focus #1: 50% duty cycle (10-on / 10-off repeating) stays bounded in P1 (never enters P0)
    let mut rta_50 = SimplexRta::new();
    for cycle in 0..5 {
        for _ in 0..10 {
            let out = rta_50.step(&drop_in);
            assert!(out.window_dropout_20 <= 10);
            if cycle == 0 {
                assert_eq!(out.mode, SimplexMode::DegradedCoast);
            } else {
                // P1 NEVER climbs up into transient P2
                assert_eq!(out.mode, SimplexMode::FallbackD0Safe);
            }
        }
        for _ in 0..10 {
            let out = rta_50.step(&nominal_input());
            assert_eq!(out.mode, SimplexMode::FallbackD0Safe);
        }
    }

    // 3. Review Focus #1: 10-on / 1-off / 1-on (11/20 = 55% duty cycle) forces P0 immediately at 0 ticks!
    let mut rta_55 = SimplexRta::new();
    for _ in 0..10 {
        assert_eq!(rta_55.step(&drop_in).mode, SimplexMode::DegradedCoast);
    }
    // 1 tick clear -> transitions P2 -> P1 with dwell_ticks = 1
    let clear1 = rta_55.step(&nominal_input());
    assert_eq!(clear1.mode, SimplexMode::FallbackD0Safe);
    assert_eq!(clear1.dwell_ticks, 1);
    // Next tick drops again -> consec=1, but window_dropout_20 == 11 > 10 -> P0 immediately!
    let flicker_trip = rta_55.step(&drop_in);
    assert_eq!(flicker_trip.window_dropout_20, 11);
    assert_eq!(flicker_trip.mode, SimplexMode::InvalidInhibit);
}

#[test]
fn test_mandatory_go_around_non_resettable_during_approach() {
    let mut rta = SimplexRta::new();
    let mut approach_breach = nominal_input();
    approach_breach.approach_active = true;
    approach_breach.v_vpl_mm = V_VAL_MM + 1;
    let out1 = rta.step(&approach_breach);
    assert!(out1.mandatory_go_around);
    assert_eq!(out1.mode, SimplexMode::InvalidInhibit);

    // Fault clears AND go_around_ack_reset = true, but approach_active = true -> MUST stay latched!
    let mut ack_during_approach = nominal_input();
    ack_during_approach.approach_active = true;
    ack_during_approach.go_around_ack_reset = true;
    for _ in 0..40 {
        let out = rta.step(&ack_during_approach);
        assert!(out.mandatory_go_around);
        assert_eq!(out.mode, SimplexMode::InvalidInhibit);
    }

    // Only when approach_active == false && go_around_ack_reset == true does latch clear
    let mut ack_climbout = nominal_input();
    ack_climbout.approach_active = false;
    ack_climbout.go_around_ack_reset = true;
    let out_clear = rta.step(&ack_climbout);
    assert!(!out_clear.mandatory_go_around);
    assert_eq!(out_clear.mode, SimplexMode::InvalidInhibit); // still requires 30 healthy ticks to exit P0
    assert_eq!(out_clear.recovery_ticks, 1);
}

#[test]
fn test_mltl_primitives_and_single_cam_epipolar_isolation() {
    // Verify BoolRing64 H[0,n], O[0,n], and S[a,b] directly
    let mut p = BoolRing64::new();
    let mut q = BoolRing64::new();
    // Push history: t=0 (q=true, p=false), t=1 (q=false, p=true), t=2 (q=false, p=true)
    q.push(true);
    p.push(false);
    q.push(false);
    p.push(true);
    q.push(false);
    p.push(true);
    assert!(p.historically(1, true));
    assert!(!p.historically(2, true));
    assert!(q.once(2));
    assert!(!q.once(1));
    assert!(BoolRing64::since(&p, &q, 1, 2));
    assert!(!BoolRing64::since(&p, &q, 0, 1));

    // Verify R_EPI_SINGLE_CAM: single camera epipolar > 6109 urad excludes cam for 2 ticks (O[0,1])
    let mut rta = SimplexRta::new();
    let mut epi_fault = nominal_input();
    epi_fault.epipolar_max_urad[2] = EPIPOLAR_CAM_MAX_URAD + 1;
    let out_t0 = rta.step(&epi_fault);
    assert_eq!(out_t0.healthy_cam_mask, 0b111011);
    assert_eq!(out_t0.active_healthy_cams, 5);
    assert_eq!(out_t0.mode, SimplexMode::Isolated5Cam);

    // Next tick epipolar clears, but O[0,1] keeps Cam 2 excluded for 1 more tick
    let out_t1 = rta.step(&nominal_input());
    assert_eq!(out_t1.healthy_cam_mask, 0b111011);
    assert_eq!(out_t1.active_healthy_cams, 5);
    assert_eq!(out_t1.mode, SimplexMode::Isolated5Cam);

    // Tick after that: O[0,1] clears, Cam 2 restored -> 6 healthy cams, dwells 20 ticks before P3->P4
    let out_t2 = rta.step(&nominal_input());
    assert_eq!(out_t2.healthy_cam_mask, 0b111111);
    assert_eq!(out_t2.active_healthy_cams, 6);
    assert_eq!(out_t2.mode, SimplexMode::Isolated5Cam);
    assert_eq!(out_t2.dwell_ticks, 1);
}

fn arb_tick_input() -> impl Strategy<Value = RtaTickInput> {
    let part_a = (
        0u8..=0x3F,
        0u8..=0x3F,
        prop_oneof![0i64..=HORIZON_AL_URAD, (HORIZON_AL_URAD + 1)..=30_000],
        prop_oneof![0i64..=V_VAL_MM, (V_VAL_MM + 1)..=25_000],
        prop_oneof![0i64..=V_HAL_MM, (V_HAL_MM + 1)..=80_000],
        prop::array::uniform6(prop_oneof![
            0i64..=EPIPOLAR_CAM_MAX_URAD,
            (EPIPOLAR_CAM_MAX_URAD + 1)..=25_000
        ]),
    );
    let part_b = (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        prop_oneof![9 => Just(false), 1 => Just(true)],
    );
    (part_a, part_b).prop_map(
        |(
            (
                cam_health_mask,
                ood_exceeded_mask,
                horizon_pl_urad,
                v_vpl_mm,
                v_hpl_mm,
                epipolar_max_urad,
            ),
            (
                spoof_alarm,
                ml_detector_fault,
                ood_global_exceeded,
                runway_width_fault,
                transient_dropout,
                approach_active,
                go_around_ack_reset,
                quantize_fault,
            ),
        )| RtaTickInput {
            cam_health_mask,
            ood_exceeded_mask,
            horizon_pl_urad,
            horizon_al_urad: HORIZON_AL_URAD,
            v_vpl_mm,
            v_val_mm: V_VAL_MM,
            v_hpl_mm,
            v_hal_mm: V_HAL_MM,
            epipolar_max_urad,
            signed_horizon_residual_urad: 0,
            runway_width_delta_mm: 0,
            spoof_alarm,
            ml_detector_fault,
            ood_global_exceeded,
            runway_width_fault,
            transient_dropout,
            approach_active,
            go_around_ack_reset,
            quantize_fault,
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    #[test]
    fn proptest_100k_rta_safety_invariants(trace in prop::collection::vec(arb_tick_input(), 1..=25)) {
        let mut rta = SimplexRta::new();
        let mut prev_mode = rta.mode();
        let mut prev_go_around = false;

        for inp in &trace {
            let out = rta.step(inp);

            // Invariant 1: Quantize fault always forces P0 at 0 ticks
            if inp.quantize_fault {
                prop_assert_eq!(out.mode, SimplexMode::InvalidInhibit);
            }

            // Invariant 2: 0-tick downward preemption (raw_target < prev_mode => mode == raw_target)
            if (out.raw_target as u8) < (prev_mode as u8) {
                prop_assert_eq!(out.mode, out.raw_target);
            }

            // Invariant 3: P2 implies 1 <= consec_dropout <= 10 and window_dropout_20 <= 10;
            // sustained > 10 or window_dropout_20 > 10 forces P0
            if out.mode == SimplexMode::DegradedCoast {
                prop_assert!(out.consec_dropout_ticks >= 1 && out.consec_dropout_ticks <= CONSEC_DROPOUT_MAX_TICKS);
                prop_assert!(out.window_dropout_20 <= WINDOW_DROPOUT_20_MAX_TICKS);
            }
            if out.consec_dropout_ticks > CONSEC_DROPOUT_MAX_TICKS || out.window_dropout_20 > WINDOW_DROPOUT_20_MAX_TICKS {
                prop_assert_eq!(out.mode, SimplexMode::InvalidInhibit);
            }

            // Invariant 4: Never climb upward into P2, and never jump P0 -> P3/P4 directly
            if prev_mode == SimplexMode::InvalidInhibit {
                prop_assert!(matches!(out.mode, SimplexMode::InvalidInhibit | SimplexMode::FallbackD0Safe));
            }
            if prev_mode == SimplexMode::FallbackD0Safe {
                prop_assert_ne!(out.mode, SimplexMode::DegradedCoast);
            }

            // Invariant 5: Mandatory go-around non-resettable while approach_active
            if prev_go_around && inp.approach_active {
                prop_assert!(out.mandatory_go_around);
                prop_assert_eq!(out.mode, SimplexMode::InvalidInhibit);
            }

            // Invariant 6: P0 always blanks conformal HUD and inhibits avoidance turns
            if out.mode == SimplexMode::InvalidInhibit {
                prop_assert!(out.blank_conformal_hud);
                prop_assert!(out.inhibit_avoidance_turns);
            }

            prev_mode = out.mode;
            prev_go_around = out.mandatory_go_around;
        }
    }
}
