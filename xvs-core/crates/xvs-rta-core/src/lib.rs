#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(clippy::arithmetic_side_effects)]

pub mod mltl;
pub mod odd_generated;
pub mod quantize;
pub mod simplex;

#[cfg(kani)]
mod proofs;

pub use mltl::{BoolRing64, OddMltlMonitor, OddMltlOutcome, PastMltlMonitor};
pub use odd_generated::{
    CAM_COUNT_HISTORY_WINDOW, CONSEC_DROPOUT_MAX_TICKS, DWELL_TICKS, EPIPOLAR_CAM_MAX_URAD,
    EPI_ONCE_WINDOW, HORIZON_AL_URAD, MIN_HEALTHY_CAMS, MLTL_RING_CAPACITY, ODD_YAML_SHA256,
    P0_RECOVERY_TICKS, RTA_TICK_RATE_HZ, SPOOF_ONCE_WINDOW, V_HAL_MM, V_VAL_MM,
    WINDOW_DROPOUT_20_MAX_TICKS, WINDOW_DROPOUT_LEN_TICKS,
};
pub use quantize::{
    build_rta_input_from_f64, quantize_nonneg_upper_i64, quantize_signed_bounded_i64, QuantFault,
    RtaFloatTelemetry,
};
pub use simplex::{RtaTickInput, RtaTickOutput, SimplexMode, SimplexRta};
