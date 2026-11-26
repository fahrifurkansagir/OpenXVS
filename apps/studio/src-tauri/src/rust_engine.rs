// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Native Pure-Rust `ort` (`load-dynamic`) + `xvs-active-range` + `xvs-multiship-coord`
//! + `xvs-rta-core` + `SensorHealthMonitor` Inference Engine for OpenXVS Studio v5.0 Tauri v2 IPC.
//!
//! Security & Regulatory Invariants:
//! - `#![forbid(unsafe_code)]` (enforced at crate root)
//! - Under `#[cfg(feature = "non_flight_sim_only")]`, ALL flight actuation / outbound MAVLink /
//!   FlightGear TX paths are compiled out (`ACTUATOR_AUTHORITY == false` compile-time invariant).
//! - Explicitly defeats Windows 11 `System32\onnxruntime.dll` v1.10.0 hijack by blocking all
//!   `System32`/`SysWOW64`/`WinSxS` paths and loading bundled relative `ORT_DYLIB_PATH`.
//! - Fail-closed SHA-256 pinning enforced on ALL 6 ONNX models (`XVS-D2`, `XVS-ThermalNet`,
//!   `XVS-Land`, `XVS-Nav v5.0 Faz-B`, `XFeat Stage-2A FP16`, `LighterGlue Stage-2B FP16`).
//!   Tampered files are strictly rejected (`SHA256_MISMATCH_REJECTED`) before ORT session creation.
//! - Zero compile-time host path strings (`CARGO_MANIFEST_DIR` is never embedded in binary).
//! - Zero host paths serialized into IPC payloads or exported evaluation reports (REL1 clean).
//! - Zero Python required: runs real ONNX sessions + `xvs-multiship-coord` (F1/F3/F4) +
//!   `SensorHealthMonitor` (LWIR NUC freeze, EO blur/glare/occlusion, timestamp skew guard) +
//!   frozen Rust `xvs-active-range` & `xvs-rta-core` Simplex RTA.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::STANDARD as BASE64_STD;
use base64::Engine as _;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{ImageReader, Limits, Rgb, RgbImage};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xvs_active_range::{
    decode_wire, encode_wire, ActiveRangeConfig, ActiveRangeMeasurement, ActiveRangeMode,
    ActiveRangeTelemetry, OwnshipStateNed, ACTIVE_RANGE_WIRE_BYTES,
};
use xvs_math_core::mesh_daa::OpenXvsBearingTrack;
use xvs_multiship_coord::{
    CoordinatedActiveRange, Ring0AuthorityGate, StatefulMeshDaaGuard, SwerveCoordRole,
};
use crate::nav_integrity::{
    DualBankConfig, DualBankEsEkfCusum, DualBankStepInput, SequenceVprConfig, SequenceVprFilter,
    CUSUM_H_LOCKED, CUSUM_REINTEGRATION_HYSTERESIS_EPOCHS,
};
use xvs_rta_core::{
    build_rta_input_from_f64, RtaFloatTelemetry, RtaTickInput, SimplexMode, SimplexRta,
    DWELL_TICKS, ODD_YAML_SHA256, P0_RECOVERY_TICKS,
};

pub const STUDIO_VERSION: &str = "1.1.1";
pub const NON_FLIGHT_SIM_ONLY: bool = cfg!(feature = "non_flight_sim_only");
pub const ACTUATOR_AUTHORITY: bool = false;
pub const NON_FLIGHT_WATERMARK: &str =
    "SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT";
pub const EU_PLD_ART4_7_LOCKOUT_MSG: &str =
    "Outbound MAVLink/FlightGear actuation is compiled out under non_flight_sim_only (EU PLD Directive (EU) 2024/2853 Art. 4(7) — zero actuation authority).";

pub const D2_CROP_SIZE: usize = 256;
pub const D2_HID_C: usize = 24;
pub const D2_HID_HW: usize = D2_CROP_SIZE / 4; // 64
pub const AOT_FOCAL_PX: f64 = 2690.0;
pub const AOT_REF_WIDTH_PX: f64 = 2448.0;
pub const LWIR_L_SAT_PROXY: f32 = 215.7;
pub const LAND_RUNWAY_PRESENT_MIN_PROB: f64 = 0.65;
pub const MAX_ANALYZE_IMAGE_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_ANALYZE_IMAGE_B64_LEN: usize = MAX_ANALYZE_IMAGE_BYTES.div_ceil(3) * 4 + 256;
pub const MAX_ANALYZE_IMAGE_DIM: u32 = 4096;
pub const DEMO_SLOW_RATE_GROUP_STRIDE: u32 = 5;
const MEDIA_LRU_CAPACITY: usize = 64;
const SAT_REF_XFEAT_CACHE_CAPACITY: usize = 32;

pub const XFEAT_IN_W: usize = 320;
pub const XFEAT_IN_H: usize = 256;
pub const XFEAT_CELL_W: usize = 40;
pub const XFEAT_CELL_H: usize = 32;
pub const XFEAT_DESC_DIM: usize = 64;
pub const XFEAT_MAX_KEYPOINTS: usize = 256;
pub const LIGHTERGLUE_STATIC_K: usize = 256;
pub const LIGHTERGLUE_MATCH_MIN_PROB: f32 = 0.20;

// ---- Fail-Closed SHA-256 Pins for ALL 6 Production ONNX Models (v5.2 Land+Thermal + Faz-B Nav Triad) ----

pub const EXPECTED_D2_ONNX_SHA256: &str =
    "2c9578f362f6bbbcc0f9ce1de925bf5dbce5459d75dfb387c79c810023750d0d";
pub const EXPECTED_THERMAL_V52_FP16_ONNX_SHA256: &str =
    "bc6ceac6252f48dc52305c4e7955cb7841bc5adc7c191a59ce9b0e36ef10fcac";
pub const EXPECTED_THERMAL_V52_FP32_ONNX_SHA256: &str =
    "87fbd639591012ec5918c95a1915de9c0fa2911606823afb4e5c6f2bd0c82dbd";
pub const EXPECTED_THERMAL_FP32_ONNX_SHA256: &str =
    "db06e32fb44dd8a949ed8972e06b362c10557c608ed36809d0e9626eeb0a93c3";
pub const EXPECTED_THERMAL_ONNX_SHA256: &str = EXPECTED_THERMAL_V52_FP16_ONNX_SHA256;
pub const EXPECTED_THERMAL_FP16_ONNX_SHA256: &str =
    "85a6c9100846b00810889c6c1d97b227cd495057b34c98923abd3b04acf6bd98";
pub const EXPECTED_LAND_V52_FP16_SHA256: &str =
    "834c8e270f237f044d3920c98d5c0818363ab781be78071884e9365a221ac799";
pub const EXPECTED_LAND_V52_FP32_SHA256: &str =
    "fc8e808394b7e20c0d80e5fe357306ba332ee6f57018666e941a8d55caee6407";
pub const EXPECTED_LAND_FP16_ONNX_SHA256: &str =
    "254a6e49bafb9cedf01edaf9650ff98b98ad42ad2d219a29dc8725bb280997e5";
pub const EXPECTED_LAND_FP32_ONNX_SHA256: &str =
    "4245c0d1cdfea8d5065f56d9d3a7dd9a3d6cc29ff8dea1b1a7c173cd39e2920d";
// XVS-Nav v5.2 Faz-C is explicitly HELD (DO NOT PIN as active default: -10.54 pp all-512 /
// -16.09 pp land-320 regression from 92.6% S2 / 7.4% NAIP ratio). Keep Faz-B pinned.
pub const EXPECTED_NAV_V5_FAZB_FP16_ONNX_SHA256: &str =
    "e48f0ed5a234dcf1661031b80db5028d8c5b926b7b54e06575b4ed20bd6fc46f";
pub const EXPECTED_NAV_V5_FAZB_FP32_ONNX_SHA256: &str =
    "a43c3ccf5e6f776f49dc7661001a3cddc844cc2e1706367f3beb5ba0a6e187bc";
pub const EXPECTED_NAV_V5_FP16_ONNX_SHA256: &str =
    "c9b78a98f51ff294b2eeffb12c65902647a9ad08b28a06dec6b870b39fdf6bd1";
pub const EXPECTED_NAV_V5_FP32_ONNX_SHA256: &str =
    "5dc2a12bf1774b5ebc3bf1612d7a08da8eced63aa6b3aa78ebb6e9bf51b5f22e";
pub const EXPECTED_XFEAT_FP16_ONNX_SHA256: &str =
    "e28a2f3afd7e23fe9818ecea34b57185708ff2e2b45f24cb514628f49c3d8182";
pub const EXPECTED_XFEAT_FP32_ONNX_SHA256: &str =
    "04a153209ec1bd494fea9489b92552d05c91339c86990d2492edbf104deb37c6";
pub const EXPECTED_LIGHTERGLUE_FP16_ONNX_SHA256: &str =
    "e63121ce24b840dba04acd490133eba4175a2ae43b087db3dac6e9d2f3220f3b";
pub const EXPECTED_LIGHTERGLUE_FP32_ONNX_SHA256: &str =
    "fd43db4f25a7d714c414526dd387e0c7f1397a313175994f5eba79389a143512";

pub const D2_ONNX_PINS: &[(&str, &str)] = &[("xvs_d2_web_fp16.onnx", EXPECTED_D2_ONNX_SHA256)];
pub const THERMAL_ONNX_PINS: &[(&str, &str)] = &[
    ("xvs_thermal_v52_fp16.onnx", EXPECTED_THERMAL_V52_FP16_ONNX_SHA256),
    ("xvs_thermal_v52_fp32.onnx", EXPECTED_THERMAL_V52_FP32_ONNX_SHA256),
    ("xvs_thermal_v45_fp32.onnx", EXPECTED_THERMAL_FP32_ONNX_SHA256),
    ("xvs_thermal_v45_fp16.onnx", EXPECTED_THERMAL_FP16_ONNX_SHA256),
];
pub const LAND_ONNX_PINS: &[(&str, &str)] = &[
    ("xvs_land_v52_fp16.onnx", EXPECTED_LAND_V52_FP16_SHA256),
    ("xvs_land_v52_fp32.onnx", EXPECTED_LAND_V52_FP32_SHA256),
    ("xvs_land_v45_fp16.onnx", EXPECTED_LAND_FP16_ONNX_SHA256),
    ("xvs_land_v45.onnx", EXPECTED_LAND_FP32_ONNX_SHA256),
];
pub const NAV_ONNX_PINS: &[(&str, &str)] = &[
    ("xvs_nav_v5_fazb_fp16.onnx", EXPECTED_NAV_V5_FAZB_FP16_ONNX_SHA256),
    ("xvs_nav_v5_fp16.onnx", EXPECTED_NAV_V5_FP16_ONNX_SHA256),
    ("xvs_nav_v5_fazb_fp32.onnx", EXPECTED_NAV_V5_FAZB_FP32_ONNX_SHA256),
    ("xvs_nav_v5_fp32.onnx", EXPECTED_NAV_V5_FP32_ONNX_SHA256),
];
pub const XFEAT_ONNX_PINS: &[(&str, &str)] = &[
    ("xfeat_320x256_fp16.onnx", EXPECTED_XFEAT_FP16_ONNX_SHA256),
    ("xfeat_320x256_fp32.onnx", EXPECTED_XFEAT_FP32_ONNX_SHA256),
];
pub const LIGHTERGLUE_ONNX_PINS: &[(&str, &str)] = &[
    ("lighterglue_k256_fp16.onnx", EXPECTED_LIGHTERGLUE_FP16_ONNX_SHA256),
    ("lighterglue_k256_fp32.onnx", EXPECTED_LIGHTERGLUE_FP32_ONNX_SHA256),
];

pub const V52_LAND_STUDENT_T_NU: f64 = 6.0;
pub const V52_LAND_STUDENT_T_SCALE: f64 = 1.048_075_542_704_791_2;
pub const WAVE_C_LAND_STUDENT_T_NU: f64 = 6.0;
pub const WAVE_C_LAND_STUDENT_T_SCALE: f64 = 4.214_477_485_437_598;
pub const LAND_NUM_KEYPOINTS: usize = 12;
pub const LAND_NUM_KEYLINES: usize = 5;
pub const LAND_DEFAULT_KEYLINE_SIGMA_PX: f64 = 1.5;
pub const MIN_KEYLINE_SIGMA_PX: f64 = 0.25;
/// Promoted Wave 3A SHA-256 pin for v5.2 6-output `xvs_land_v52_fp16.onnx` (`2,899,997 B`, `384x384`).
pub const EXPECTED_LAND_V52_FP16_ONNX_SHA256: Option<&str> = Some(EXPECTED_LAND_V52_FP16_SHA256);

pub const LWIR_NUC_FREEZE_MAD_THR: f64 = 0.25;
pub const LWIR_NUC_FREEZE_MIN_TICKS: u32 = 3;
pub const LWIR_CONTRAST_COLLAPSE_P95_P05_THR: f64 = 10.0;
pub const LWIR_CONTRAST_REF_RANGE: f64 = 30.0;
pub const LWIR_LOCAL_PEAK_MIN_DN: f64 = 40.0;
pub const LWIR_LOCAL_PEAK_DELTA_DN: f64 = 25.0;
pub const LWIR_LOCAL_PEAK_SNR_MIN: f64 = 6.0;
pub const LWIR_FFC_BIT_IMMINENT: u8 = 1 << 0;
pub const LWIR_FFC_BIT_IN_PROGRESS: u8 = 1 << 1;
pub const EO_SECTOR_DEGRADED_THR: f64 = 0.50;
pub const FPN_DEAD_LINE_FLAG_FRAC: f64 = 0.03;
pub const FPN_DEAD_LINE_FAIL_FRAC: f64 = 0.10;
pub const NOISE_STD_MIN: f64 = 15.0;
pub const NOISE_AUTOCORR_MIN: f64 = 0.25;
pub const MAX_TIMESTAMP_SKEW_MS: f64 = 200.0;
pub const FSYNC_PRECISION_MAX_SKEW_MS: f64 = 1.0;
pub const BEARING_TBC_CONFIRM_M: u8 = 3;
pub const BEARING_TBC_WINDOW_N: usize = 5;
pub const BEARING_TBC_MAX_GATE_RAD: f64 = 0.035;
pub const BEARING_TBC_MAX_MISS_STREAK: u32 = 5;

#[cfg(feature = "non_flight_sim_only")]
const _: () = {
    assert!(!ACTUATOR_AUTHORITY);
};

// ---- Exact Rust StudentTCalibrator (`src/xvs/v2/xvs_land/model.py` parity) ----

#[derive(Debug, Clone, Copy)]
pub struct StudentTCalibrator {
    pub nu: f64,
    pub scale: f64,
}

impl StudentTCalibrator {
    pub fn new(nu: f64, scale: f64) -> Result<Self, String> {
        if !(nu.is_finite() && nu > 2.0) {
            return Err(format!("Student-t nu must be finite > 2.0, got {nu}"));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(format!("Student-t scale must be finite > 0.0, got {scale}"));
        }
        Ok(Self { nu, scale })
    }

    #[inline]
    pub fn variance_factor(&self) -> f64 {
        (self.scale * self.scale) * (self.nu / (self.nu - 2.0))
    }

    #[inline]
    pub fn calibrate_cov2x2(&self, cov: [f64; 4]) -> [f64; 4] {
        let factor = self.variance_factor();
        [
            cov[0] * factor,
            cov[1] * factor,
            cov[2] * factor,
            cov[3] * factor,
        ]
    }
}

// ---- Multi-Contract XVS-Land 5-Output (v5.1), 6-Output (v5.2) & 7-Output (extended `runway_present_logit`) Parser ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LandOutputContract {
    V51FiveOutput,
    V52SixOutputKeylineSigma,
    V53SevenOutputRunwayPresence,
}

impl LandOutputContract {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::V51FiveOutput => "v51_5output",
            Self::V52SixOutputKeylineSigma => "v52_6output",
            Self::V53SevenOutputRunwayPresence => "v53_7output",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ParsedLandOutputs {
    pub contract: LandOutputContract,
    pub keypoints_xy: Vec<[f64; 2]>,
    pub keylines_hesse: Vec<[f64; 3]>,
    pub calibrated_kp_sigmas_px: [f64; LAND_NUM_KEYPOINTS],
    pub mean_keypoint_sigma_px: f64,
    pub keyline_sigma_px: Option<[f64; LAND_NUM_KEYLINES]>,
    pub keyline_precision_weights: [f64; LAND_NUM_KEYLINES],
    pub effective_keyline_sigma_px: f64,
    pub runway_present_logit: Option<f64>,
    pub runway_present_prob: Option<f64>,
    pub runway_present_admitted: bool,
    pub runway_gate_mode: &'static str,
}

/// Inspect XVS-Land ONNX session input shape `[1, 3, H, W]` dynamically (defaults to `256x256`).
pub fn resolve_land_input_hw(sess: &Session) -> (u32, u32) {
    if let Some(inp) = sess.inputs().first() {
        if let Some(shape) = inp.dtype().tensor_shape() {
            if shape.len() == 4 && shape[2] > 0 && shape[3] > 0 {
                return (shape[2] as u32, shape[3] as u32);
            }
        }
    }
    (256, 256)
}

/// Inspect XVS-D2 ONNX session input shapes dynamically:
/// `x_t` `[1, 6, H, W]` (`256x256` v5.2 OR `640x640` extended) and `h_prev_warped` `[1, C, H/4, W/4]`.
pub fn resolve_d2_input_shapes(sess: &Session) -> (usize, usize, usize, usize, usize) {
    let mut in_h = D2_CROP_SIZE;
    let mut in_w = D2_CROP_SIZE;
    let mut hid_c = D2_HID_C;
    let mut hid_h = D2_HID_HW;
    let mut hid_w = D2_HID_HW;
    let mut saw_x = false;
    let mut saw_h = false;

    for inp in sess.inputs() {
        if let Some(shape) = inp.dtype().tensor_shape() {
            if shape.len() == 4 {
                if inp.name() == "x_t" && shape[2] > 0 && shape[3] > 0 {
                    in_h = shape[2] as usize;
                    in_w = shape[3] as usize;
                    saw_x = true;
                } else if inp.name() == "h_prev_warped"
                    && shape[1] > 0
                    && shape[2] > 0
                    && shape[3] > 0
                {
                    hid_c = shape[1] as usize;
                    hid_h = shape[2] as usize;
                    hid_w = shape[3] as usize;
                    saw_h = true;
                }
            }
        }
    }
    if saw_x && !saw_h {
        hid_h = (in_h / 4).max(1);
        hid_w = (in_w / 4).max(1);
    }
    (in_h, in_w, hid_c, hid_h, hid_w)
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + (-x.clamp(-60.0, 60.0)).exp())
}

/// Backward-compatible 5/6-output parser delegating to [`parse_xvs_land_outputs_with_presence`].
pub fn parse_xvs_land_outputs(
    kpts: &[f32],
    keylines: &[f32],
    cov: &[f32],
    keyline_sigma_opt: Option<&[f32]>,
    calibrator: &StudentTCalibrator,
) -> Result<ParsedLandOutputs, String> {
    parse_xvs_land_outputs_with_presence(kpts, keylines, cov, keyline_sigma_opt, None, calibrator)
}

/// Parse and fail-closed validate XVS-Land outputs for 5-output (v4.5/v5.1), 6-output
/// (v5.2 `keyline_sigma_px`), and 7-output (extended `runway_present_logit`, shape `[1, 1]`) ONNX contracts.
///
/// - When `runway_present_logit_opt` is `None`: 100% backward-compatible with v5.1/v5.2 (`admitted = true`).
/// - When `runway_present_logit_opt` is `Some(raw)`: fails closed (`P0_INVALID_INHIBIT`) whenever
///   `sigmoid(logit) < LAND_RUNWAY_PRESENT_MIN_PROB` (`0.65`) or `logit` is non-finite.
pub fn parse_xvs_land_outputs_with_presence(
    kpts: &[f32],
    keylines: &[f32],
    cov: &[f32],
    keyline_sigma_opt: Option<&[f32]>,
    runway_present_logit_opt: Option<&[f32]>,
    calibrator: &StudentTCalibrator,
) -> Result<ParsedLandOutputs, String> {
    if kpts.len() < LAND_NUM_KEYPOINTS * 2 {
        return Err(format!(
            "Land keypoints_xy length {} < {}",
            kpts.len(),
            LAND_NUM_KEYPOINTS * 2
        ));
    }
    if keylines.len() < LAND_NUM_KEYLINES * 3 {
        return Err(format!(
            "Land keylines_hesse length {} < {}",
            keylines.len(),
            LAND_NUM_KEYLINES * 3
        ));
    }
    if cov.len() < LAND_NUM_KEYPOINTS * 4 {
        return Err(format!(
            "Land cov_2x2 length {} < {}",
            cov.len(),
            LAND_NUM_KEYPOINTS * 4
        ));
    }

    let mut keypoints_xy: Vec<[f64; 2]> = Vec::with_capacity(LAND_NUM_KEYPOINTS);
    let mut calibrated_kp_sigmas_px = [0.0_f64; LAND_NUM_KEYPOINTS];
    let mut sigma_sum = 0.0_f64;
    for (i, kp_sig_slot) in calibrated_kp_sigmas_px.iter_mut().enumerate() {
        let u = kpts[2 * i] as f64;
        let v = kpts[2 * i + 1] as f64;
        if !(u.is_finite() && v.is_finite()) {
            return Err(format!("Land keypoint[{i}] contains non-finite value"));
        }
        keypoints_xy.push([u, v]);

        let raw_c = [
            cov[4 * i] as f64,
            cov[4 * i + 1] as f64,
            cov[4 * i + 2] as f64,
            cov[4 * i + 3] as f64,
        ];
        if !raw_c.iter().all(|x| x.is_finite()) {
            return Err(format!("Land cov_2x2[{i}] contains non-finite value"));
        }
        let cal_c = calibrator.calibrate_cov2x2(raw_c);
        let sig = (cal_c[0] + cal_c[3]).max(1e-6).sqrt();
        if !sig.is_finite() {
            return Err(format!("Land calibrated sigma[{i}] is non-finite"));
        }
        *kp_sig_slot = sig;
        sigma_sum += sig;
    }

    let mut keylines_hesse: Vec<[f64; 3]> = Vec::with_capacity(LAND_NUM_KEYLINES);
    for j in 0..LAND_NUM_KEYLINES {
        let kl = [
            keylines[3 * j] as f64,
            keylines[3 * j + 1] as f64,
            keylines[3 * j + 2] as f64,
        ];
        if !kl.iter().all(|x| x.is_finite()) {
            return Err(format!("Land keyline[{j}] contains non-finite value"));
        }
        keylines_hesse.push(kl);
    }

    let (mut contract, keyline_sigma_px, keyline_precision_weights, effective_keyline_sigma_px) =
        match keyline_sigma_opt {
            None => {
                let w_unif = 1.0 / (LAND_DEFAULT_KEYLINE_SIGMA_PX * LAND_DEFAULT_KEYLINE_SIGMA_PX);
                (
                    LandOutputContract::V51FiveOutput,
                    None,
                    [w_unif; LAND_NUM_KEYLINES],
                    LAND_DEFAULT_KEYLINE_SIGMA_PX,
                )
            }
            Some(raw_sig) => {
                if raw_sig.len() < LAND_NUM_KEYLINES {
                    return Err(format!(
                        "Land keyline_sigma_px length {} < {}",
                        raw_sig.len(),
                        LAND_NUM_KEYLINES
                    ));
                }
                let mut sigs = [0.0_f64; LAND_NUM_KEYLINES];
                let mut weights = [0.0_f64; LAND_NUM_KEYLINES];
                let mut prec_sum = 0.0_f64;
                for k in 0..LAND_NUM_KEYLINES {
                    let raw_s = raw_sig[k] as f64;
                    if !raw_s.is_finite() || raw_s <= 0.0 {
                        return Err(format!(
                            "Land keyline_sigma_px[{k}] invalid (must be finite and > 0): {raw_s}"
                        ));
                    }
                    let s = raw_s.max(MIN_KEYLINE_SIGMA_PX);
                    let w = 1.0 / (s * s);
                    if !w.is_finite() || w <= 0.0 {
                        return Err(format!(
                            "Land keyline_sigma_px[{k}] produced non-finite precision weight: {s}"
                        ));
                    }
                    sigs[k] = s;
                    weights[k] = w;
                    prec_sum += w;
                }
                let mean_prec = prec_sum / (LAND_NUM_KEYLINES as f64);
                let eff_sig = (1.0 / mean_prec).sqrt();
                if !eff_sig.is_finite() || eff_sig <= 0.0 {
                    return Err("Land effective_keyline_sigma_px is non-finite".to_string());
                }
                (
                    LandOutputContract::V52SixOutputKeylineSigma,
                    Some(sigs),
                    weights,
                    eff_sig,
                )
            }
        };

    let (runway_present_logit, runway_present_prob, runway_present_admitted, runway_gate_mode) =
        match runway_present_logit_opt {
            None => (None, None, true, "NOMINAL"),
            Some(raw_pres) => {
                if raw_pres.is_empty() {
                    return Err("Land runway_present_logit tensor is empty".to_string());
                }
                contract = LandOutputContract::V53SevenOutputRunwayPresence;
                let logit = raw_pres[0] as f64;
                if !logit.is_finite() {
                    (None, Some(0.0), false, "P0_INVALID_INHIBIT")
                } else {
                    let prob = sigmoid_f64(logit);
                    let admitted = prob.is_finite() && prob >= LAND_RUNWAY_PRESENT_MIN_PROB;
                    let gate_mode = if admitted {
                        "NOMINAL"
                    } else {
                        "P0_INVALID_INHIBIT"
                    };
                    (Some(logit), Some(prob), admitted, gate_mode)
                }
            }
        };

    Ok(ParsedLandOutputs {
        contract,
        keypoints_xy,
        keylines_hesse,
        calibrated_kp_sigmas_px,
        mean_keypoint_sigma_px: sigma_sum / (LAND_NUM_KEYPOINTS as f64),
        keyline_sigma_px,
        keyline_precision_weights,
        effective_keyline_sigma_px,
        runway_present_logit,
        runway_present_prob,
        runway_present_admitted,
        runway_gate_mode,
    })
}

// ---- Exact Thermal Simplex Mode (`src/xvs/v2/xvs_thermal/model.py` parity) ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum ThermalSimplexMode {
    NominalFusion = 0,
    EoDegradedLwirPrimary = 1,
    LwirDegradedEoPrimary = 2,
    DualDveSvsFallback = 3,
}

impl ThermalSimplexMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NominalFusion => "NOMINAL_FUSION",
            Self::EoDegradedLwirPrimary => "EO_DEGRADED_LWIR_PRIMARY",
            Self::LwirDegradedEoPrimary => "LWIR_DEGRADED_EO_PRIMARY",
            Self::DualDveSvsFallback => "DUAL_DVE_SVS_FALLBACK",
        }
    }
}

pub fn select_thermal_simplex_mode(q_eo: f64, q_lwir: f64, vacuity: f64) -> ThermalSimplexMode {
    let tau_ok = 0.5_f64;
    let tau_vac = 0.7_f64;
    if !(vacuity.is_finite() && (0.0..tau_vac).contains(&vacuity)) {
        return ThermalSimplexMode::DualDveSvsFallback;
    }
    let eo_ok = q_eo.is_finite() && (tau_ok..=1.0).contains(&q_eo);
    let lwir_ok = q_lwir.is_finite() && (tau_ok..=1.0).contains(&q_lwir);
    match (eo_ok, lwir_ok) {
        (false, false) => ThermalSimplexMode::DualDveSvsFallback,
        (true, true) => ThermalSimplexMode::NominalFusion,
        (false, true) => ThermalSimplexMode::EoDegradedLwirPrimary,
        (true, false) => ThermalSimplexMode::LwirDegradedEoPrimary,
    }
}

// ---- Pure-Rust Real-World Optics: Brown-Conrady Pinhole & Kannala-Brandt Fisheye (Camera RDF) ----

/// Exact OpenCV / Brown-Conrady `(k1, k2, p1, p2, k3)` pinhole camera intrinsics in Camera RDF `(u, v)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PinholeIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

impl PinholeIntrinsics {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    ) -> Result<Self, String> {
        let inst = Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
            k3,
        };
        inst.validate()?;
        Ok(inst)
    }

    pub fn validate(&self) -> Result<(), String> {
        let vals = [
            self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2, self.k3,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("All PinholeIntrinsics parameters must be finite".to_string());
        }
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(format!(
                "Focal lengths fx, fy must be > 0, got ({}, {})",
                self.fx, self.fy
            ));
        }
        Ok(())
    }

    /// Apply exact Brown-Conrady forward distortion `(u, v) -> (u_dist, v_dist)` in Camera RDF.
    pub fn distort_pixel(&self, uv: [f64; 2]) -> Result<[f64; 2], String> {
        self.validate()?;
        if !(uv[0].is_finite() && uv[1].is_finite()) {
            return Err("Pixel coordinate contains non-finite value".to_string());
        }
        let x = (uv[0] - self.cx) / self.fx;
        let y = (uv[1] - self.cy) / self.fy;
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let rad = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let xd = x * rad + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let yd = y * rad + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        Ok([self.fx * xd + self.cx, self.fy * yd + self.cy])
    }

    /// Invert Brown-Conrady distortion to normalized Camera RDF coordinates `(x, y)` via damped 2D Newton-Raphson.
    pub fn undistort_to_normalized(
        &self,
        uv_dist: [f64; 2],
        max_iter: usize,
        tol: f64,
    ) -> Result<[f64; 2], String> {
        self.validate()?;
        if !(uv_dist[0].is_finite() && uv_dist[1].is_finite()) {
            return Err("Distorted pixel coordinate contains non-finite value".to_string());
        }
        if max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
            return Err("Invalid max_iter or tol for Brown-Conrady undistortion".to_string());
        }
        let xd = (uv_dist[0] - self.cx) / self.fx;
        let yd = (uv_dist[1] - self.cy) / self.fy;
        if self.k1 == 0.0 && self.k2 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0 && self.k3 == 0.0 {
            return Ok([xd, yd]);
        }
        let mut x = xd;
        let mut y = yd;
        for _ in 0..max_iter {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let rad = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
            let drad_dr2 = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;

            let fx_res = x * rad + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x) - xd;
            let fy_res = y * rad + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y - yd;

            let j11 = rad + 2.0 * x * x * drad_dr2 + 2.0 * self.p1 * y + 6.0 * self.p2 * x;
            let j12 = 2.0 * x * y * drad_dr2 + 2.0 * self.p1 * x + 2.0 * self.p2 * y;
            let j22 = rad + 2.0 * y * y * drad_dr2 + 6.0 * self.p1 * y + 2.0 * self.p2 * x;

            let det = j11 * j22 - j12 * j12;
            if !det.is_finite() || det.abs() < 1e-15 {
                return Err("Singular Jacobian in Brown-Conrady undistortion".to_string());
            }
            let mut dx = (j22 * fx_res - j12 * fy_res) / det;
            let mut dy = (-j12 * fx_res + j11 * fy_res) / det;
            let step_norm = dx.hypot(dy);
            if step_norm > 0.5 {
                let s = 0.5 / step_norm;
                dx *= s;
                dy *= s;
            }
            x -= dx;
            y -= dy;
            if dx.hypot(dy) < tol && fx_res.hypot(fy_res) < tol {
                break;
            }
        }
        if !(x.is_finite() && y.is_finite()) {
            return Err("Brown-Conrady undistortion diverged to non-finite coordinates".to_string());
        }
        Ok([x, y])
    }

    /// Invert Brown-Conrady distortion `(u_dist, v_dist) -> (u_rect, v_rect)` (`< 1e-12 px` round-trip).
    pub fn undistort_pixel(
        &self,
        uv_dist: [f64; 2],
        max_iter: usize,
        tol: f64,
    ) -> Result<[f64; 2], String> {
        let [x, y] = self.undistort_to_normalized(uv_dist, max_iter, tol)?;
        Ok([self.fx * x + self.cx, self.fy * y + self.cy])
    }

    /// Convenience default undistortion using `max_iter = 30`, `tol = 1e-15` (`< 1e-12 px` round-trip).
    pub fn undistort_pixel_exact(&self, uv_dist: [f64; 2]) -> Result<[f64; 2], String> {
        self.undistort_pixel(uv_dist, 30, 1e-15)
    }

    /// Undistort pixel directly into a unit bearing ray `[rx, ry, rz]` in Camera RDF (`+X` right, `+Y` down, `+Z` forward).
    pub fn undistort_to_unit_ray_rdf(&self, uv_dist: [f64; 2]) -> Result<[f64; 3], String> {
        let [x, y] = self.undistort_to_normalized(uv_dist, 30, 1e-15)?;
        let inv_n = 1.0 / (x * x + y * y + 1.0).sqrt();
        Ok([x * inv_n, y * inv_n, inv_n])
    }
}

/// Exact Kannala-Brandt equidistant fisheye camera model (`k1..k4`) for wide-angle `110-140 deg` lenses in Camera RDF.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KannalaBrandtFisheye {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
}

impl KannalaBrandtFisheye {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
    ) -> Result<Self, String> {
        let inst = Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            k3,
            k4,
        };
        inst.validate()?;
        Ok(inst)
    }

    pub fn validate(&self) -> Result<(), String> {
        let vals = [
            self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.k3, self.k4,
        ];
        if !vals.iter().all(|v| v.is_finite()) {
            return Err("All KannalaBrandtFisheye parameters must be finite".to_string());
        }
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(format!(
                "Fisheye focal lengths fx, fy must be > 0, got ({}, {})",
                self.fx, self.fy
            ));
        }
        Ok(())
    }

    /// Project a 3D Camera RDF ray `[X, Y, Z]` to distorted fisheye pixel `(u_d, v_d)`.
    pub fn project_ray_rdf(&self, ray_rdf: [f64; 3]) -> Result<[f64; 2], String> {
        self.validate()?;
        if !ray_rdf.iter().all(|v| v.is_finite()) {
            return Err("Camera RDF ray contains non-finite component".to_string());
        }
        let r_xy = ray_rdf[0].hypot(ray_rdf[1]);
        let norm = r_xy.hypot(ray_rdf[2]);
        if norm < 1e-15 {
            return Err("Zero-length Camera RDF ray".to_string());
        }
        if r_xy < 1e-15 {
            if ray_rdf[2] <= 0.0 {
                return Err("Antipodal ray behind fisheye optical axis".to_string());
            }
            return Ok([self.cx, self.cy]);
        }
        let theta = r_xy.atan2(ray_rdf[2]);
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        let theta_d = theta * (1.0 + self.k1 * t2 + self.k2 * t4 + self.k3 * t6 + self.k4 * t8);
        let scale = theta_d / r_xy;
        let xd = scale * ray_rdf[0];
        let yd = scale * ray_rdf[1];
        Ok([self.fx * xd + self.cx, self.fy * yd + self.cy])
    }

    /// Project an undistorted pinhole pixel `(u, v)` through the Kannala-Brandt fisheye lens.
    pub fn distort_pixel(&self, uv_pinhole: [f64; 2]) -> Result<[f64; 2], String> {
        self.validate()?;
        if !(uv_pinhole[0].is_finite() && uv_pinhole[1].is_finite()) {
            return Err("Pinhole pixel contains non-finite coordinate".to_string());
        }
        let x = (uv_pinhole[0] - self.cx) / self.fx;
        let y = (uv_pinhole[1] - self.cy) / self.fy;
        self.project_ray_rdf([x, y, 1.0])
    }

    /// Undistort a fisheye pixel `(u_d, v_d)` to a unit Camera RDF bearing ray `[rx, ry, rz]` via damped 1D Newton-Raphson.
    pub fn undistort_to_unit_ray_rdf(
        &self,
        uv_dist: [f64; 2],
        max_iter: usize,
        tol: f64,
    ) -> Result<[f64; 3], String> {
        self.validate()?;
        if !(uv_dist[0].is_finite() && uv_dist[1].is_finite()) {
            return Err("Distorted fisheye pixel contains non-finite coordinate".to_string());
        }
        if max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
            return Err("Invalid max_iter or tol for Kannala-Brandt undistortion".to_string());
        }
        let xd = (uv_dist[0] - self.cx) / self.fx;
        let yd = (uv_dist[1] - self.cy) / self.fy;
        let theta_d = xd.hypot(yd);
        if theta_d < 1e-15 {
            return Ok([0.0, 0.0, 1.0]);
        }

        let max_theta = core::f64::consts::PI - 1e-6;
        let mut theta = theta_d.clamp(0.0, 1.50);
        for _ in 0..max_iter {
            let t2 = theta * theta;
            let t4 = t2 * t2;
            let t6 = t4 * t2;
            let t8 = t4 * t4;
            let poly = 1.0 + self.k1 * t2 + self.k2 * t4 + self.k3 * t6 + self.k4 * t8;
            let f = theta * poly - theta_d;
            let df = 1.0
                + 3.0 * self.k1 * t2
                + 5.0 * self.k2 * t4
                + 7.0 * self.k3 * t6
                + 9.0 * self.k4 * t8;
            if !df.is_finite() || df.abs() < 1e-12 {
                return Err("Singular derivative in Kannala-Brandt polynomial inversion".to_string());
            }
            let delta = (f / df).clamp(-0.25, 0.25);
            theta = (theta - delta).clamp(0.0, max_theta);
            if delta.abs() < tol && f.abs() < tol {
                break;
            }
        }
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let inv_td = 1.0 / theta_d;
        Ok([xd * inv_td * sin_t, yd * inv_td * sin_t, cos_t])
    }

    /// Undistort a fisheye pixel into rectified pinhole coordinates `(u, v)` (`< 180 deg` FOV).
    pub fn undistort_pixel(
        &self,
        uv_dist: [f64; 2],
        max_iter: usize,
        tol: f64,
    ) -> Result<[f64; 2], String> {
        let [rx, ry, rz] = self.undistort_to_unit_ray_rdf(uv_dist, max_iter, tol)?;
        if rz <= 1e-6 {
            return Err("Fisheye ray exceeds pinhole reprojection hemisphere (rz <= 0)".to_string());
        }
        Ok([self.fx * (rx / rz) + self.cx, self.fy * (ry / rz) + self.cy])
    }

    /// Convenience default fisheye undistortion to unit Camera RDF ray (`max_iter = 30`, `tol = 1e-15`).
    pub fn undistort_to_unit_ray_exact(&self, uv_dist: [f64; 2]) -> Result<[f64; 3], String> {
        self.undistort_to_unit_ray_rdf(uv_dist, 30, 1e-15)
    }
}

// ---- Deterministic Unit-Quaternion SLERP & Exposure-Midpoint IMU Interpolation ----

fn normalize_unit_quat(q: [f64; 4]) -> Result<[f64; 4], String> {
    if !q.iter().all(|v| v.is_finite()) {
        return Err("IMU quaternion contains non-finite component".to_string());
    }
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err("IMU quaternion norm is degenerate".to_string());
    }
    let inv = 1.0 / norm;
    Ok([q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv])
}

/// Deterministic shortest-arc Hamilton unit-quaternion `[w, x, y, z]` SLERP (`alpha in [0.0, 1.0]`).
pub fn slerp_imu_quaternion(q0: [f64; 4], q1: [f64; 4], alpha: f64) -> Result<[f64; 4], String> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(format!(
            "SLERP interpolation alpha must be finite in [0.0, 1.0], got {alpha}"
        ));
    }
    let a = normalize_unit_quat(q0)?;
    let mut b = normalize_unit_quat(q1)?;
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }
    dot = dot.clamp(0.0, 1.0);
    let raw = if dot > 0.999_999_5 {
        let w0 = 1.0 - alpha;
        let w1 = alpha;
        [
            w0 * a[0] + w1 * b[0],
            w0 * a[1] + w1 * b[1],
            w0 * a[2] + w1 * b[2],
            w0 * a[3] + w1 * b[3],
        ]
    } else {
        let theta = dot.acos();
        let inv_sin = 1.0 / theta.sin();
        let w0 = ((1.0 - alpha) * theta).sin() * inv_sin;
        let w1 = (alpha * theta).sin() * inv_sin;
        [
            w0 * a[0] + w1 * b[0],
            w0 * a[1] + w1 * b[1],
            w0 * a[2] + w1 * b[2],
            w0 * a[3] + w1 * b[3],
        ]
    };
    normalize_unit_quat(raw)
}

/// Compute exact camera exposure midpoint `t_sof + 0.5 * t_exp` (ms), failing closed on invalid inputs.
pub fn exposure_midpoint_ms(t_sof_ms: f64, exposure_ms: f64) -> Result<f64, String> {
    if !(t_sof_ms.is_finite() && exposure_ms.is_finite()) || exposure_ms < 0.0 {
        return Err("Exposure SOF timestamp and duration must be finite with t_exp >= 0".to_string());
    }
    Ok(t_sof_ms + 0.5 * exposure_ms)
}

/// Interpolate bracketed IMU unit quaternions `(t0_ms, q0) .. (t1_ms, q1)` to exact exposure midpoint `(t_sof + 0.5 * t_exp)`.
pub fn interpolate_imu_attitude_at_exposure_midpoint(
    t0_ms: f64,
    q0: [f64; 4],
    t1_ms: f64,
    q1: [f64; 4],
    t_sof_ms: f64,
    exposure_ms: f64,
) -> Result<([f64; 4], f64), String> {
    let t_mid = exposure_midpoint_ms(t_sof_ms, exposure_ms)?;
    if !(t0_ms.is_finite() && t1_ms.is_finite()) || t1_ms <= t0_ms {
        return Err("IMU bracket timestamps must be finite and strictly increasing".to_string());
    }
    if t_mid < t0_ms || t_mid > t1_ms {
        return Err(format!(
            "Exposure midpoint {t_mid:.3} ms outside IMU bracket [{t0_ms:.3}, {t1_ms:.3}] ms"
        ));
    }
    let alpha = ((t_mid - t0_ms) / (t1_ms - t0_ms)).clamp(0.0, 1.0);
    let q_mid = slerp_imu_quaternion(q0, q1, alpha)?;
    Ok((q_mid, t_mid))
}

// ---- Dual-Tier TimestampSkewGuard (`<= 1.0 ms` FSYNC precision warning + `> 200 ms` hard breaker) ----

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TimestampCheckResult {
    pub skew_ms: f64,
    pub monotonic_ok: bool,
    pub skew_fault: bool,
    pub fsync_precision_ok: bool,
}

#[derive(Debug, Clone)]
pub struct TimestampSkewGuard {
    last_cam_ts_ms: Option<f64>,
    pub hard_max_skew_ms: f64,
    pub fsync_precision_ms: f64,
    pub last_skew_ms: f64,
    pub monotonic_ok: bool,
    pub skew_fault: bool,
    pub fsync_precision_ok: bool,
}

impl Default for TimestampSkewGuard {
    fn default() -> Self {
        Self::new(MAX_TIMESTAMP_SKEW_MS, FSYNC_PRECISION_MAX_SKEW_MS)
    }
}

impl TimestampSkewGuard {
    pub fn new(hard_max_skew_ms: f64, fsync_precision_ms: f64) -> Self {
        Self {
            last_cam_ts_ms: None,
            hard_max_skew_ms,
            fsync_precision_ms,
            last_skew_ms: 0.0,
            monotonic_ok: true,
            skew_fault: false,
            fsync_precision_ok: true,
        }
    }

    pub fn check(&mut self, cam_ts_ms: f64, imu_ts_ms: f64) -> TimestampCheckResult {
        if !(cam_ts_ms.is_finite() && imu_ts_ms.is_finite()) {
            self.last_skew_ms = f64::INFINITY;
            self.monotonic_ok = false;
            self.skew_fault = true;
            self.fsync_precision_ok = false;
            return TimestampCheckResult {
                skew_ms: f64::INFINITY,
                monotonic_ok: false,
                skew_fault: true,
                fsync_precision_ok: false,
            };
        }
        let monotonic_ok = match self.last_cam_ts_ms {
            Some(prev) => cam_ts_ms > prev,
            None => true,
        };
        let skew_ms = (cam_ts_ms - imu_ts_ms).abs();
        let skew_fault = !monotonic_ok || skew_ms > self.hard_max_skew_ms;
        let fsync_precision_ok = monotonic_ok && !skew_fault && skew_ms <= self.fsync_precision_ms;
        if monotonic_ok {
            self.last_cam_ts_ms = Some(cam_ts_ms);
        }
        self.last_skew_ms = round2(skew_ms);
        self.monotonic_ok = monotonic_ok;
        self.skew_fault = skew_fault;
        self.fsync_precision_ok = fsync_precision_ok;
        TimestampCheckResult {
            skew_ms: self.last_skew_ms,
            monotonic_ok,
            skew_fault,
            fsync_precision_ok,
        }
    }

    pub fn check_exposure_midpoint(
        &mut self,
        t_sof_ms: f64,
        exposure_ms: f64,
        imu_ts_ms: f64,
    ) -> TimestampCheckResult {
        match exposure_midpoint_ms(t_sof_ms, exposure_ms) {
            Ok(t_mid) => self.check(t_mid, imu_ts_ms),
            Err(_) => self.check(f64::NAN, imu_ts_ms),
        }
    }
}

// ---- Lightweight Zero-Allocation Rust `BearingTrackBeforeConfirm` (`3-of-5` Confirm) ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum BearingTrackStatus {
    Idle = 0,
    Tentative = 1,
    Confirmed = 2,
    Dropped = 3,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BearingTrackSnapshot {
    pub status: BearingTrackStatus,
    pub confirmed: bool,
    pub hits_in_window: u8,
    pub window_len: u8,
    pub total_hits: u32,
    pub miss_streak: u32,
    pub alpha_rad: f64,
    pub beta_rad: f64,
}

#[derive(Debug, Clone)]
pub struct BearingTrackBeforeConfirm {
    confirm_m: u8,
    window_n: usize,
    max_gate_rad: f64,
    max_miss_streak: u32,
    ring: [bool; 16],
    head: usize,
    filled: usize,
    hits_in_window: u8,
    total_hits: u32,
    miss_streak: u32,
    confirmed: bool,
    has_bearing: bool,
    last_alpha_rad: f64,
    last_beta_rad: f64,
}

impl Default for BearingTrackBeforeConfirm {
    fn default() -> Self {
        Self::new_3_of_5()
    }
}

impl BearingTrackBeforeConfirm {
    pub fn new_3_of_5() -> Self {
        Self {
            confirm_m: BEARING_TBC_CONFIRM_M,
            window_n: BEARING_TBC_WINDOW_N,
            max_gate_rad: BEARING_TBC_MAX_GATE_RAD,
            max_miss_streak: BEARING_TBC_MAX_MISS_STREAK,
            ring: [false; 16],
            head: 0,
            filled: 0,
            hits_in_window: 0,
            total_hits: 0,
            miss_streak: 0,
            confirmed: false,
            has_bearing: false,
            last_alpha_rad: 0.0,
            last_beta_rad: 0.0,
        }
    }

    pub fn new(
        confirm_m: u8,
        window_n: usize,
        max_gate_rad: f64,
        max_miss_streak: u32,
    ) -> Result<Self, String> {
        if window_n == 0 || window_n > 16 || confirm_m == 0 || (confirm_m as usize) > window_n {
            return Err("Invalid M-of-N parameters (require 1 <= M <= N <= 16)".to_string());
        }
        if !max_gate_rad.is_finite() || max_gate_rad <= 0.0 {
            return Err("max_gate_rad must be finite > 0".to_string());
        }
        if max_miss_streak == 0 {
            return Err("max_miss_streak must be >= 1".to_string());
        }
        Ok(Self {
            confirm_m,
            window_n,
            max_gate_rad,
            max_miss_streak,
            ring: [false; 16],
            head: 0,
            filled: 0,
            hits_in_window: 0,
            total_hits: 0,
            miss_streak: 0,
            confirmed: false,
            has_bearing: false,
            last_alpha_rad: 0.0,
            last_beta_rad: 0.0,
        })
    }

    pub fn reset(&mut self) {
        self.ring = [false; 16];
        self.head = 0;
        self.filled = 0;
        self.hits_in_window = 0;
        self.total_hits = 0;
        self.miss_streak = 0;
        self.confirmed = false;
        self.has_bearing = false;
        self.last_alpha_rad = 0.0;
        self.last_beta_rad = 0.0;
    }

    pub fn is_confirmed(&self) -> bool {
        self.confirmed
    }

    pub fn hits_in_window(&self) -> u8 {
        self.hits_in_window
    }

    /// Observe optional `(alpha_rad, beta_rad)` bearing measurement gated by EO optical health.
    /// Fails closed (treats tick as miss) whenever `q_eo_eff` is non-finite / below threshold or EO is degraded.
    pub fn observe_with_health(
        &mut self,
        det: Option<(f64, f64)>,
        q_eo_eff: f64,
        eo_degraded: bool,
    ) -> BearingTrackSnapshot {
        let eo_healthy =
            q_eo_eff.is_finite() && q_eo_eff >= EO_SECTOR_DEGRADED_THR && !eo_degraded;
        if eo_healthy {
            self.observe(det)
        } else {
            self.observe(None)
        }
    }

    /// Observe optional `(alpha_rad, beta_rad)` bearing measurement and advance `M-of-N` (`3-of-5`) confirmation.
    pub fn observe(&mut self, det: Option<(f64, f64)>) -> BearingTrackSnapshot {
        let hit = match det {
            Some((alpha, beta)) if alpha.is_finite() && beta.is_finite() => {
                if !self.has_bearing {
                    self.has_bearing = true;
                    self.last_alpha_rad = alpha;
                    self.last_beta_rad = beta;
                    true
                } else {
                    let two_pi = 2.0 * core::f64::consts::PI;
                    let d_alpha = ((alpha - self.last_alpha_rad + core::f64::consts::PI)
                        .rem_euclid(two_pi))
                        - core::f64::consts::PI;
                    let d_beta = beta - self.last_beta_rad;
                    if d_alpha.hypot(d_beta) <= self.max_gate_rad {
                        self.last_alpha_rad = alpha;
                        self.last_beta_rad = beta;
                        true
                    } else {
                        false
                    }
                }
            }
            _ => false,
        };

        if self.filled == self.window_n {
            if self.ring[self.head] {
                self.hits_in_window = self.hits_in_window.saturating_sub(1);
            }
        } else {
            self.filled += 1;
        }
        self.ring[self.head] = hit;
        if hit {
            self.hits_in_window = self.hits_in_window.saturating_add(1);
            self.total_hits = self.total_hits.saturating_add(1);
            self.miss_streak = 0;
        } else {
            self.miss_streak = self.miss_streak.saturating_add(1);
        }
        self.head += 1;
        if self.head == self.window_n {
            self.head = 0;
        }

        if self.hits_in_window >= self.confirm_m {
            self.confirmed = true;
        }

        if self.miss_streak >= self.max_miss_streak && (self.has_bearing || self.confirmed) {
            let snap = BearingTrackSnapshot {
                status: BearingTrackStatus::Dropped,
                confirmed: false,
                hits_in_window: 0,
                window_len: self.window_n as u8,
                total_hits: self.total_hits,
                miss_streak: self.miss_streak,
                alpha_rad: self.last_alpha_rad,
                beta_rad: self.last_beta_rad,
            };
            self.reset();
            return snap;
        }

        let status = if self.confirmed {
            BearingTrackStatus::Confirmed
        } else if self.hits_in_window > 0 {
            BearingTrackStatus::Tentative
        } else {
            BearingTrackStatus::Idle
        };

        BearingTrackSnapshot {
            status,
            confirmed: self.confirmed,
            hits_in_window: self.hits_in_window,
            window_len: self.window_n as u8,
            total_hits: self.total_hits,
            miss_streak: self.miss_streak,
            alpha_rad: self.last_alpha_rad,
            beta_rad: self.last_beta_rad,
        }
    }
}

// ---- Real-World Avionics SensorHealthMonitor (`#![forbid(unsafe_code)]`) ----

#[derive(Debug, Clone, Copy)]
pub struct EoOpticalQuality {
    pub blur_score: f64,
    pub focus_quality: f64,
    pub glare_ratio: f64,
    pub occlusion_ratio: f64,
    pub q_eo_optical: f64,
    pub grid_4x4_q_eo: [f64; 16],
    pub sector_q_eo: [f64; 12],
    pub min_sector_q_eo: f64,
    pub degraded_sector_mask: u16,
    pub dead_line_fraction: f64,
    pub fpn_score: f64,
    pub spatial_autocorr: f64,
    pub noise_score: f64,
    pub dead_pixel_fpn_fault: bool,
    pub sensor_noise_fault: bool,
    pub degraded: bool,
}

impl EoOpticalQuality {
    pub fn flags(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.focus_quality < EO_SECTOR_DEGRADED_THR {
            out.push("MOTION_OR_FOCUS_BLUR");
        }
        if self.glare_ratio > 0.375 {
            out.push("SUN_GLARE_OVEREXPOSED");
        }
        if self.occlusion_ratio > 0.675 {
            out.push("LENS_CAP_OR_UNDEREXPOSED");
        }
        if self.dead_pixel_fpn_fault {
            out.push("DEAD_PIXEL_FPN_FAULT");
        }
        if self.sensor_noise_fault {
            out.push("SENSOR_NOISE_FAULT");
        }
        out
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct LwirHealthStatus {
    pub nuc_frozen: bool,
    pub freeze_ticks: u32,
    pub last_mad: f64,
    pub ffc_imminent: bool,
    pub ffc_in_progress: bool,
    pub dynamic_range_p95_p05: f64,
    pub effective_dynamic_range: f64,
    pub contrast_collapsed: bool,
    pub localized_hot_target_rescue: bool,
    pub glare_ratio: f64,
    pub occlusion_ratio: f64,
    pub dead_line_fraction: f64,
    pub fpn_score: f64,
    pub spatial_autocorr: f64,
    pub noise_score: f64,
    pub dead_pixel_fpn_fault: bool,
    pub sensor_noise_fault: bool,
    pub q_lwir_optical: f64,
    pub degraded: bool,
    pub mask_lwir_frame: bool,
}

impl LwirHealthStatus {
    pub fn flags(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.nuc_frozen {
            out.push("LWIR_NUC_SHUTTER_FROZEN");
        }
        if self.ffc_imminent {
            out.push("LWIR_FFC_IMMINENT");
        }
        if self.ffc_in_progress {
            out.push("LWIR_FFC_IN_PROGRESS");
        }
        let glare_score = (1.0 - (self.glare_ratio - 0.02).max(0.0) / 0.48).clamp(0.0, 1.0);
        if glare_score < EO_SECTOR_DEGRADED_THR {
            out.push("SUN_GLARE_OVEREXPOSED");
        }
        let occ_score = if self.localized_hot_target_rescue {
            1.0
        } else {
            (1.0 - (self.occlusion_ratio - 0.10).max(0.0) / 0.65).clamp(0.0, 1.0)
        };
        if occ_score < EO_SECTOR_DEGRADED_THR {
            out.push("LENS_CAP_OR_UNDEREXPOSED");
        }
        let contrast_score =
            (self.effective_dynamic_range / LWIR_CONTRAST_REF_RANGE).clamp(0.0, 1.0);
        if self.contrast_collapsed || contrast_score < EO_SECTOR_DEGRADED_THR {
            out.push("LOW_CONTRAST_DYNAMIC_RANGE");
        }
        if self.dead_pixel_fpn_fault {
            out.push("DEAD_PIXEL_FPN_FAULT");
        }
        if self.sensor_noise_fault {
            out.push("SENSOR_NOISE_FAULT");
        }
        out
    }
}

/// Compute exact `p95 - p05` dynamic range of an 8-bit LWIR plane via O(N) 256-bin histogram (zero alloc).
pub fn lwir_dynamic_range_p95_p05(lwir_u8: &[u8]) -> f64 {
    if lwir_u8.is_empty() {
        return 0.0;
    }
    let mut hist = [0usize; 256];
    for &px in lwir_u8 {
        hist[px as usize] += 1;
    }
    let n = lwir_u8.len();
    let idx_p05 = ((n.saturating_sub(1)) * 5) / 100;
    let idx_p95 = ((n.saturating_sub(1)) * 95) / 100;
    let mut cum = 0usize;
    let mut p05 = 0u8;
    let mut p95 = 255u8;
    let mut found_05 = false;
    for (val, &cnt) in hist.iter().enumerate() {
        cum += cnt;
        if !found_05 && cum > idx_p05 {
            p05 = val as u8;
            found_05 = true;
        }
        if cum > idx_p95 {
            p95 = val as u8;
            break;
        }
    }
    (p95 as f64) - (p05 as f64)
}

#[inline]
fn hist_percentile_u8(hist: &[usize; 256], n: usize, pct_num: usize, pct_den: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let idx = (n.saturating_sub(1).saturating_mul(pct_num)) / pct_den.max(1);
    let mut cum = 0usize;
    for (val, &cnt) in hist.iter().enumerate() {
        cum += cnt;
        if cum > idx {
            return val as f64;
        }
    }
    255.0
}

#[inline]
fn hist_median_u32(hist: &[u32; 256], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let lo_idx = (n - 1) / 2;
    let hi_idx = n / 2;
    let mut cum = 0usize;
    let mut lo_v = 0u8;
    let mut hi_v = 0u8;
    let mut found_lo = false;
    for (val, &cnt) in hist.iter().enumerate() {
        cum += cnt as usize;
        if !found_lo && cum > lo_idx {
            lo_v = val as u8;
            found_lo = true;
        }
        if cum > hi_idx {
            hi_v = val as u8;
            break;
        }
    }
    0.5 * ((lo_v as f64) + (hi_v as f64))
}

/// Derive `(median, is_stuck)` directly from a 256-bin line histogram in O(256) with zero second scan.
#[inline]
fn hist_line_median_and_stuck(hist: &[u32; 256], len: usize) -> (f64, bool) {
    let med = hist_median_u32(hist, len);
    let mut sum_u64 = 0u64;
    let mut sq_u64 = 0u64;
    for (v, &cnt) in hist.iter().enumerate() {
        if cnt != 0 {
            let c = cnt as u64;
            let vu = v as u64;
            sum_u64 += c * vu;
            sq_u64 += c * vu * vu;
        }
    }
    let lf = len as f64;
    let mean = (sum_u64 as f64) / lf;
    let std = (((sq_u64 as f64) / lf) - mean * mean).max(0.0).sqrt();
    if std < 1.0 {
        return (med, true);
    }
    let lo_b = ((med - 1.5).floor().max(0.0) as usize).min(255);
    let hi_b = ((med + 1.5).ceil().min(255.0) as usize).min(255);
    let mut near = 0usize;
    for (v, &cnt) in hist.iter().enumerate().take(hi_b + 1).skip(lo_b) {
        if ((v as f64) - med).abs() <= 1.5 {
            near += cnt as usize;
        }
    }
    (med, ((near as f64) / lf) >= 0.70)
}

/// Detect stuck/dead sensor readout rows & columns (1-2 px FPN stripes, mirroring Wave 1 `sensor_health.py`).
///
/// Uses per-row/column O(256) histogram median + variance + near-median stuck fraction (`>= 70%`
/// within `±1.5 DN` of line median, OR within-line std `< 1.0 DN`) with L1-tiled row-major column
/// passes (zero column-major cache thrashing, exact bit-parity with reference).
pub fn detect_dead_line_fpn_u8(plane: &[u8], w: usize, h: usize) -> (f64, f64) {
    if w < 6 || h < 6 || plane.len() != w * h {
        return (0.0, 1.0);
    }

    let mut frame_sum = 0u64;
    let mut frame_sq = 0u64;
    let mut row_med = vec![0.0_f64; h];
    let mut row_stuck = vec![false; h];
    for y in 0..h {
        let mut hist = [0u32; 256];
        let row = &plane[y * w..(y + 1) * w];
        for &px in row {
            hist[px as usize] += 1;
        }
        for (v, &cnt) in hist.iter().enumerate() {
            if cnt != 0 {
                let c = cnt as u64;
                let vu = v as u64;
                frame_sum += c * vu;
                frame_sq += c * vu * vu;
            }
        }
        let (med, stuck) = hist_line_median_and_stuck(&hist, w);
        row_med[y] = med;
        row_stuck[y] = stuck;
    }

    let n = (w * h) as f64;
    let mean = (frame_sum as f64) / n;
    let frame_std = (((frame_sq as f64) / n) - mean * mean).max(0.0).sqrt();
    if !frame_std.is_finite() || frame_std < 1.0 {
        return (0.0, 1.0);
    }

    let mut dead_rows = 0usize;
    for y in 2..(h - 2) {
        if !row_stuck[y] {
            continue;
        }
        let m = row_med[y];
        let r_up = (m - row_med[y - 1]).abs().max((m - row_med[y - 2]).abs());
        let r_dn = (m - row_med[y + 1]).abs().max((m - row_med[y + 2]).abs());
        let jump = r_up.min(r_dn);
        let pinned_rail = m <= 2.0 || m >= 253.0;
        if jump >= 15.0 || (pinned_rail && jump >= 8.0) {
            dead_rows += 1;
        }
    }

    // L1-cache-friendly row-major tiled column histograms (16 columns = 16 KB stack tile)
    let mut col_med = vec![0.0_f64; w];
    let mut col_stuck = vec![false; w];
    const COL_TILE: usize = 16;
    let mut col_hists = [[0u32; 256]; COL_TILE];
    let mut x0 = 0usize;
    while x0 < w {
        let tile_w = (w - x0).min(COL_TILE);
        for hist in col_hists.iter_mut().take(tile_w) {
            hist.fill(0);
        }
        for y in 0..h {
            let slice = &plane[y * w + x0..y * w + x0 + tile_w];
            for (k, &px) in slice.iter().enumerate() {
                col_hists[k][px as usize] += 1;
            }
        }
        for (k, hist) in col_hists.iter().take(tile_w).enumerate() {
            let (med, stuck) = hist_line_median_and_stuck(hist, h);
            col_med[x0 + k] = med;
            col_stuck[x0 + k] = stuck;
        }
        x0 += tile_w;
    }

    let mut dead_cols = 0usize;
    for x in 2..(w - 2) {
        if !col_stuck[x] {
            continue;
        }
        let m = col_med[x];
        let c_lt = (m - col_med[x - 1]).abs().max((m - col_med[x - 2]).abs());
        let c_rt = (m - col_med[x + 1]).abs().max((m - col_med[x + 2]).abs());
        let jump = c_lt.min(c_rt);
        let pinned_rail = m <= 2.0 || m >= 253.0;
        if jump >= 15.0 || (pinned_rail && jump >= 8.0) {
            dead_cols += 1;
        }
    }

    let f_rows = (dead_rows as f64) / (h as f64);
    let f_cols = (dead_cols as f64) / (w as f64);
    let dead_frac = (f_rows + f_cols).min(1.0);
    if dead_frac < FPN_DEAD_LINE_FLAG_FRAC {
        return (dead_frac, 1.0);
    }
    let span = (FPN_DEAD_LINE_FAIL_FRAC - FPN_DEAD_LINE_FLAG_FRAC).max(1e-6);
    let fpn_score = (0.45 * (1.0 - (dead_frac - FPN_DEAD_LINE_FLAG_FRAC) / span)).clamp(0.0, 1.0);
    (dead_frac, fpn_score)
}

/// Detect spatially uncorrelated white/binary noise bypassing Laplacian focus checks
/// (mirroring Wave 1 `sensor_health.py`).
///
/// Computes horizontal & vertical lag-1/lag-2 spatial autocorrelation via single integer stats pass
/// + single fused row-major LUT pass (exact mathematical parity, ~3-4x faster).
pub fn detect_white_noise_fault_u8(plane: &[u8], w: usize, h: usize) -> (f64, f64) {
    if w < 4 || h < 4 || plane.len() != w * h {
        return (1.0, 1.0);
    }
    let n = (w * h) as f64;
    let mut sum_u64 = 0u64;
    let mut sq_u64 = 0u64;
    for &px in plane {
        let v = px as u64;
        sum_u64 += v;
        sq_u64 += v * v;
    }
    let mean = (sum_u64 as f64) / n;
    let var = (((sq_u64 as f64) / n) - mean * mean).max(0.0);
    let std_val = var.sqrt();
    if !std_val.is_finite() || std_val <= NOISE_STD_MIN || var <= 1e-6 {
        return (1.0, 1.0);
    }

    let mut lut = [0.0_f64; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        *slot = (v as f64) - mean;
    }

    let mut cov_h1 = 0.0_f64;
    let mut cov_h2 = 0.0_f64;
    let mut cov_v1 = 0.0_f64;
    let mut cov_v2 = 0.0_f64;

    for y in 0..h {
        let row0 = &plane[y * w..(y + 1) * w];
        for x in 0..(w - 2) {
            let a = lut[row0[x] as usize];
            cov_h1 += a * lut[row0[x + 1] as usize];
            cov_h2 += a * lut[row0[x + 2] as usize];
        }
        cov_h1 += lut[row0[w - 2] as usize] * lut[row0[w - 1] as usize];

        if y + 1 < h {
            let row1 = &plane[(y + 1) * w..(y + 2) * w];
            for (&p0, &p1) in row0.iter().zip(row1.iter()) {
                cov_v1 += lut[p0 as usize] * lut[p1 as usize];
            }
        }
        if y + 2 < h {
            let row2 = &plane[(y + 2) * w..(y + 3) * w];
            for (&p0, &p2) in row0.iter().zip(row2.iter()) {
                cov_v2 += lut[p0 as usize] * lut[p2 as usize];
            }
        }
    }

    let cnt_h1 = (h * (w - 1)).max(1) as f64;
    let cnt_h2 = (h * (w - 2)).max(1) as f64;
    let cnt_v1 = ((h - 1) * w).max(1) as f64;
    let cnt_v2 = ((h - 2) * w).max(1) as f64;

    let ac_h1 = (cov_h1 / cnt_h1) / var;
    let ac_v1 = (cov_v1 / cnt_v1) / var;
    let ac_h2 = (cov_h2 / cnt_h2) / var;
    let ac_v2 = (cov_v2 / cnt_v2) / var;
    let ac = ac_h1.max(ac_v1);
    if !ac.is_finite() {
        return (0.0, 0.0);
    }
    let periodic_checker = ac_h2.is_finite()
        && ac_v2.is_finite()
        && ac_h2.abs().max(ac_v2.abs()) >= 0.70;
    if periodic_checker {
        return (ac_h2.abs().max(ac_v2.abs()), 1.0);
    }
    if ac < NOISE_AUTOCORR_MIN {
        let noise_score = (0.05 * ac.max(0.0) / NOISE_AUTOCORR_MIN).clamp(0.0, 0.05);
        return (ac, noise_score);
    }
    (ac, 1.0)
}

fn extract_native_luma_u8_window(
    img: &RgbImage,
    max_w: usize,
    max_h: usize,
) -> (Vec<u8>, usize, usize) {
    let iw = img.width() as usize;
    let ih = img.height() as usize;
    if iw == 0 || ih == 0 {
        return (Vec::new(), 0, 0);
    }
    let w = iw.min(max_w);
    let h = ih.min(max_h);
    let x0 = (iw - w) / 2;
    let y0 = (ih - h) / 2;
    let raw = img.as_raw();
    let stride = iw * 3;
    let mut out = Vec::with_capacity(w * h);
    for y in y0..(y0 + h) {
        let row_bytes = &raw[y * stride + x0 * 3..y * stride + (x0 + w) * 3];
        for rgb in row_bytes.as_chunks::<3>().0 {
            let y_val =
                (0.299 * (rgb[0] as f64) + 0.587 * (rgb[1] as f64) + 0.114 * (rgb[2] as f64))
                    .round()
                    .clamp(0.0, 255.0) as u8;
            out.push(y_val);
        }
    }
    (out, w, h)
}

/// Evaluate native-resolution FPN & white-noise once on `img` (`<= 640x480` center window).
fn compute_native_rgb_fpn_noise(img: &RgbImage) -> Option<(f64, f64, f64, f64)> {
    let (nat_u8, nw, nh) = extract_native_luma_u8_window(img, 640, 480);
    if nw >= 6 && nh >= 6 {
        let (dlf, fpn) = detect_dead_line_fpn_u8(&nat_u8, nw, nh);
        let (ac, noise) = detect_white_noise_fault_u8(&nat_u8, nw, nh);
        Some((dlf, fpn, ac, noise))
    } else {
        None
    }
}

fn evaluate_luma_patch_quality(
    luma: &[f64],
    stride: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> (f64, f64, f64, f64, f64) {
    if x1 <= x0 + 2 || y1 <= y0 + 2 {
        return (0.0, 0.0, 0.0, 1.0, 0.0);
    }
    let mut bright_cnt = 0usize;
    let mut dark_cnt = 0usize;
    let mut total = 0usize;
    for y in y0..y1 {
        let row = y * stride;
        for x in x0..x1 {
            let v = luma[row + x];
            if v > 250.0 {
                bright_cnt += 1;
            } else if v < 5.0 {
                dark_cnt += 1;
            }
            total += 1;
        }
    }

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut cnt = 0usize;
    for y in (y0 + 1)..(y1 - 1) {
        let row = y * stride;
        let row_up = (y - 1) * stride;
        let row_dn = (y + 1) * stride;
        for x in (x0 + 1)..(x1 - 1) {
            let lap = -4.0 * luma[row + x]
                + luma[row_up + x]
                + luma[row_dn + x]
                + luma[row + x - 1]
                + luma[row + x + 1];
            sum += lap;
            sum_sq += lap * lap;
            cnt += 1;
        }
    }
    let n_int = cnt.max(1) as f64;
    let mean = sum / n_int;
    let lap_var = ((sum_sq / n_int) - mean * mean).max(0.0);

    let focus_quality = if !lap_var.is_finite() || lap_var <= 0.0 {
        0.0
    } else if lap_var >= 50.0 {
        1.0
    } else {
        ((lap_var / (lap_var + 6.0)) * (56.0 / 50.0)).clamp(0.0, 1.0)
    };

    let glare_ratio = (bright_cnt as f64) / (total.max(1) as f64);
    let occlusion_ratio = (dark_cnt as f64) / (total.max(1) as f64);

    let q_glare = if glare_ratio <= 0.15 {
        1.0
    } else {
        (1.0 - (glare_ratio - 0.15) / 0.45).clamp(0.0, 1.0)
    };
    let q_occ = if occlusion_ratio <= 0.50 {
        1.0
    } else {
        (1.0 - (occlusion_ratio - 0.50) / 0.35).clamp(0.0, 1.0)
    };
    let q_opt = focus_quality.min(q_glare).min(q_occ).clamp(0.0, 1.0);
    (lap_var, focus_quality, glare_ratio, occlusion_ratio, q_opt)
}

#[derive(Debug, Clone)]
pub struct SensorHealthMonitor {
    prev_lwir_u8: Option<Vec<u8>>,
    pub lwir_freeze_ticks: u32,
    pub lwir_nuc_frozen: bool,
    pub lwir_last_mad: f64,
    pub lwir_ffc_imminent: bool,
    pub lwir_ffc_in_progress: bool,
    pub lwir_dynamic_range_p95_p05: f64,
    pub lwir_effective_dynamic_range: f64,
    pub lwir_contrast_collapsed: bool,
    pub lwir_localized_hot_target_rescue: bool,
    pub lwir_glare_ratio: f64,
    pub lwir_occlusion_ratio: f64,
    pub lwir_dead_line_fraction: f64,
    pub lwir_fpn_score: f64,
    pub lwir_spatial_autocorr: f64,
    pub lwir_noise_score: f64,
    pub lwir_dead_pixel_fpn_fault: bool,
    pub lwir_sensor_noise_fault: bool,
    pub lwir_q_optical: f64,
    pub lwir_degraded: bool,
    pub eo_blur_score: f64,
    pub eo_focus_quality: f64,
    pub eo_glare_ratio: f64,
    pub eo_occlusion_ratio: f64,
    pub eo_q_optical: f64,
    pub eo_dead_line_fraction: f64,
    pub eo_fpn_score: f64,
    pub eo_spatial_autocorr: f64,
    pub eo_noise_score: f64,
    pub eo_dead_pixel_fpn_fault: bool,
    pub eo_sensor_noise_fault: bool,
    pub eo_degraded: bool,
    pub grid_4x4_q_eo: [f64; 16],
    pub sector_q_eo: [f64; 12],
    pub min_sector_q_eo: f64,
    pub degraded_sector_mask: u16,
    timestamp_guard: TimestampSkewGuard,
    pub timestamp_skew_ms: f64,
    pub timestamp_monotonic_ok: bool,
    pub timestamp_skew_fault: bool,
    pub fsync_precision_ok: bool,
    pub q_eo_effective: f64,
    pub q_lwir_effective: f64,
    pub thermal_mode: ThermalSimplexMode,
}

impl Default for SensorHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SensorHealthMonitor {
    pub fn new() -> Self {
        Self {
            prev_lwir_u8: None,
            lwir_freeze_ticks: 0,
            lwir_nuc_frozen: false,
            lwir_last_mad: -1.0,
            lwir_ffc_imminent: false,
            lwir_ffc_in_progress: false,
            lwir_dynamic_range_p95_p05: 255.0,
            lwir_effective_dynamic_range: 255.0,
            lwir_contrast_collapsed: false,
            lwir_localized_hot_target_rescue: false,
            lwir_glare_ratio: 0.0,
            lwir_occlusion_ratio: 0.0,
            lwir_dead_line_fraction: 0.0,
            lwir_fpn_score: 1.0,
            lwir_spatial_autocorr: 1.0,
            lwir_noise_score: 1.0,
            lwir_dead_pixel_fpn_fault: false,
            lwir_sensor_noise_fault: false,
            lwir_q_optical: 1.0,
            lwir_degraded: false,
            eo_blur_score: 100.0,
            eo_focus_quality: 1.0,
            eo_glare_ratio: 0.0,
            eo_occlusion_ratio: 0.0,
            eo_q_optical: 1.0,
            eo_dead_line_fraction: 0.0,
            eo_fpn_score: 1.0,
            eo_spatial_autocorr: 1.0,
            eo_noise_score: 1.0,
            eo_dead_pixel_fpn_fault: false,
            eo_sensor_noise_fault: false,
            eo_degraded: false,
            grid_4x4_q_eo: [1.0; 16],
            sector_q_eo: [1.0; 12],
            min_sector_q_eo: 1.0,
            degraded_sector_mask: 0,
            timestamp_guard: TimestampSkewGuard::default(),
            timestamp_skew_ms: 0.0,
            timestamp_monotonic_ok: true,
            timestamp_skew_fault: false,
            fsync_precision_ok: true,
            q_eo_effective: 1.0,
            q_lwir_effective: 1.0,
            thermal_mode: ThermalSimplexMode::NominalFusion,
        }
    }

    pub fn eo_flags(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.eo_focus_quality < EO_SECTOR_DEGRADED_THR {
            out.push("MOTION_OR_FOCUS_BLUR");
        }
        if self.eo_glare_ratio > 0.375 {
            out.push("SUN_GLARE_OVEREXPOSED");
        }
        if self.eo_occlusion_ratio > 0.675 {
            out.push("LENS_CAP_OR_UNDEREXPOSED");
        }
        if self.eo_dead_pixel_fpn_fault {
            out.push("DEAD_PIXEL_FPN_FAULT");
        }
        if self.eo_sensor_noise_fault {
            out.push("SENSOR_NOISE_FAULT");
        }
        out
    }

    pub fn lwir_flags(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.lwir_nuc_frozen {
            out.push("LWIR_NUC_SHUTTER_FROZEN");
        }
        if self.lwir_ffc_imminent {
            out.push("LWIR_FFC_IMMINENT");
        }
        if self.lwir_ffc_in_progress {
            out.push("LWIR_FFC_IN_PROGRESS");
        }
        let glare_score = (1.0 - (self.lwir_glare_ratio - 0.02).max(0.0) / 0.48).clamp(0.0, 1.0);
        if glare_score < EO_SECTOR_DEGRADED_THR {
            out.push("SUN_GLARE_OVEREXPOSED");
        }
        let occ_score = if self.lwir_localized_hot_target_rescue {
            1.0
        } else {
            (1.0 - (self.lwir_occlusion_ratio - 0.10).max(0.0) / 0.65).clamp(0.0, 1.0)
        };
        if occ_score < EO_SECTOR_DEGRADED_THR {
            out.push("LENS_CAP_OR_UNDEREXPOSED");
        }
        let contrast_score =
            (self.lwir_effective_dynamic_range / LWIR_CONTRAST_REF_RANGE).clamp(0.0, 1.0);
        if self.lwir_contrast_collapsed || contrast_score < EO_SECTOR_DEGRADED_THR {
            out.push("LOW_CONTRAST_DYNAMIC_RANGE");
        }
        if self.lwir_dead_pixel_fpn_fault {
            out.push("DEAD_PIXEL_FPN_FAULT");
        }
        if self.lwir_sensor_noise_fault {
            out.push("SENSOR_NOISE_FAULT");
        }
        out
    }

    /// Evaluate EO focus/blur (3x3 Laplacian variance), sun-glare (>250), lens-cap/dark occlusion (<5),
    /// dead-pixel row/column FPN, spatially uncorrelated white/binary noise, plus `4x4` spatial grid
    /// and `12`-azimuth-sector localized quality map on a luminance plane.
    pub fn evaluate_eo_luma_plane(luma: &[f64], w: usize, h: usize) -> EoOpticalQuality {
        Self::evaluate_eo_luma_plane_with_fpn_noise(luma, w, h, None)
    }

    fn evaluate_eo_luma_plane_with_fpn_noise(
        luma: &[f64],
        w: usize,
        h: usize,
        precomputed_fpn_noise: Option<(f64, f64, f64, f64)>,
    ) -> EoOpticalQuality {
        if w < 24 || h < 16 || luma.len() != w * h {
            return EoOpticalQuality {
                blur_score: 0.0,
                focus_quality: 0.0,
                glare_ratio: 0.0,
                occlusion_ratio: 1.0,
                q_eo_optical: 0.0,
                grid_4x4_q_eo: [0.0; 16],
                sector_q_eo: [0.0; 12],
                min_sector_q_eo: 0.0,
                degraded_sector_mask: 0x0FFF,
                dead_line_fraction: 0.0,
                fpn_score: 0.0,
                spatial_autocorr: 0.0,
                noise_score: 0.0,
                dead_pixel_fpn_fault: false,
                sensor_noise_fault: false,
                degraded: true,
            };
        }

        let (dead_line_frac, fpn_score, spatial_autocorr, noise_score) =
            if let Some(pre) = precomputed_fpn_noise {
                pre
            } else {
                let plane_u8: Vec<u8> = luma
                    .iter()
                    .map(|&v| {
                        if v.is_finite() {
                            v.round().clamp(0.0, 255.0) as u8
                        } else {
                            0u8
                        }
                    })
                    .collect();
                let (dlf, fs) = detect_dead_line_fpn_u8(&plane_u8, w, h);
                let (ac, ns) = detect_white_noise_fault_u8(&plane_u8, w, h);
                (dlf, fs, ac, ns)
            };
        let fault_mult = (fpn_score * noise_score).clamp(0.0, 1.0);

        let (lap_var, focus_quality, glare_ratio, occlusion_ratio, q_opt_base) =
            evaluate_luma_patch_quality(luma, w, 0, w, 0, h);
        let q_eo_optical = (q_opt_base * fault_mult).clamp(0.0, 1.0);

        // 4x4 Spatial Grid (`16` tiles)
        let mut grid_4x4_q_eo = [1.0_f64; 16];
        for r in 0..4usize {
            let y0 = (r * h) / 4;
            let y1 = ((r + 1) * h) / 4;
            for c in 0..4usize {
                let x0 = (c * w) / 4;
                let x1 = ((c + 1) * w) / 4;
                let (_, _, _, _, tile_q) = evaluate_luma_patch_quality(luma, w, x0, x1, y0, y1);
                grid_4x4_q_eo[r * 4 + c] = round4((tile_q * fault_mult).clamp(0.0, 1.0));
            }
        }

        // 12-Azimuth-Sector x 4-Elevation-Band worst-case quality map (catches localized raindrop or sun-glint)
        let mut sector_q_eo = [1.0_f64; 12];
        let mut min_sector_q_eo = 1.0_f64;
        let mut degraded_sector_mask: u16 = 0;
        for (s, slot) in sector_q_eo.iter_mut().enumerate() {
            let x0 = (s * w) / 12;
            let x1 = ((s + 1) * w) / 12;
            let mut worst_band_q = 1.0_f64;
            for band in 0..4usize {
                let y0 = (band * h) / 4;
                let y1 = ((band + 1) * h) / 4;
                let (_, _, _, _, band_q) = evaluate_luma_patch_quality(luma, w, x0, x1, y0, y1);
                worst_band_q = worst_band_q.min(band_q);
            }
            let q_s = round4((worst_band_q * fault_mult).clamp(0.0, 1.0));
            *slot = q_s;
            min_sector_q_eo = min_sector_q_eo.min(q_s);
            if q_s < EO_SECTOR_DEGRADED_THR {
                degraded_sector_mask |= 1u16 << s;
            }
        }

        let dead_pixel_fpn_fault = dead_line_frac >= FPN_DEAD_LINE_FLAG_FRAC;
        let sensor_noise_fault = noise_score < EO_SECTOR_DEGRADED_THR;
        let q_rounded = round4(q_eo_optical);

        EoOpticalQuality {
            blur_score: round2(lap_var),
            focus_quality: round4(focus_quality),
            glare_ratio: round4(glare_ratio),
            occlusion_ratio: round4(occlusion_ratio),
            q_eo_optical: q_rounded,
            grid_4x4_q_eo,
            sector_q_eo,
            min_sector_q_eo: round4(min_sector_q_eo),
            degraded_sector_mask,
            dead_line_fraction: round4(dead_line_frac),
            fpn_score: round4(fpn_score),
            spatial_autocorr: round4(spatial_autocorr),
            noise_score: round4(noise_score),
            dead_pixel_fpn_fault,
            sensor_noise_fault,
            degraded: q_rounded < EO_SECTOR_DEGRADED_THR
                || dead_pixel_fpn_fault
                || sensor_noise_fault,
        }
    }

    pub fn evaluate_eo_optical_quality(img: &RgbImage) -> EoOpticalQuality {
        let pre = compute_native_rgb_fpn_noise(img);
        Self::evaluate_eo_optical_quality_with_fpn_noise(img, pre)
    }

    fn evaluate_eo_optical_quality_with_fpn_noise(
        img: &RgbImage,
        precomputed_fpn_noise: Option<(f64, f64, f64, f64)>,
    ) -> EoOpticalQuality {
        let (w, h) = (160_usize, 128_usize);
        let small = image::imageops::resize(img, w as u32, h as u32, FilterType::Triangle);
        let mut luma = Vec::with_capacity(w * h);
        for rgb in small.as_raw().as_chunks::<3>().0 {
            let y = 0.299 * (rgb[0] as f64) + 0.587 * (rgb[1] as f64) + 0.114 * (rgb[2] as f64);
            luma.push(y);
        }
        Self::evaluate_eo_luma_plane_with_fpn_noise(&luma, w, h, precomputed_fpn_noise)
    }

    pub fn observe_eo_frame(&mut self, img: &RgbImage) -> EoOpticalQuality {
        let q = Self::evaluate_eo_optical_quality(img);
        self.record_eo_quality(q);
        q
    }

    fn record_eo_quality(&mut self, q: EoOpticalQuality) {
        self.eo_blur_score = q.blur_score;
        self.eo_focus_quality = q.focus_quality;
        self.eo_glare_ratio = q.glare_ratio;
        self.eo_occlusion_ratio = q.occlusion_ratio;
        self.eo_q_optical = q.q_eo_optical;
        self.eo_dead_line_fraction = q.dead_line_fraction;
        self.eo_fpn_score = q.fpn_score;
        self.eo_spatial_autocorr = q.spatial_autocorr;
        self.eo_noise_score = q.noise_score;
        self.eo_dead_pixel_fpn_fault = q.dead_pixel_fpn_fault;
        self.eo_sensor_noise_fault = q.sensor_noise_fault;
        self.eo_degraded = q.degraded;
        self.grid_4x4_q_eo = q.grid_4x4_q_eo;
        self.sector_q_eo = q.sector_q_eo;
        self.min_sector_q_eo = q.min_sector_q_eo;
        self.degraded_sector_mask = q.degraded_sector_mask;
    }

    /// Observe a 2D LWIR plane with hardware `FFC_IMMINENT`/`FFC_IN_PROGRESS` bitmask,
    /// `p95-p05` contrast-collapse backstop, cold-sky + `>=2x2` hot-target preservation
    /// (rejecting isolated 1-pixel hot noise spikes), dead-pixel FPN, white-noise autocorr,
    /// saturation/glare, and consecutive `MAD < 0.25` DMA-freeze detector.
    pub fn observe_lwir_plane_with_ffc(
        &mut self,
        lwir_u8: &[u8],
        w: usize,
        h: usize,
        ffc_bits: u8,
    ) -> LwirHealthStatus {
        self.observe_lwir_plane_with_ffc_precomputed(lwir_u8, w, h, ffc_bits, None)
    }

    fn observe_lwir_plane_with_ffc_precomputed(
        &mut self,
        lwir_u8: &[u8],
        w: usize,
        h: usize,
        ffc_bits: u8,
        precomputed_fpn_noise: Option<(f64, f64, f64, f64)>,
    ) -> LwirHealthStatus {
        let ffc_imminent = (ffc_bits & LWIR_FFC_BIT_IMMINENT) != 0;
        let ffc_in_progress = (ffc_bits & LWIR_FFC_BIT_IN_PROGRESS) != 0;
        self.lwir_ffc_imminent = ffc_imminent;
        self.lwir_ffc_in_progress = ffc_in_progress;

        let n = lwir_u8.len();
        let has_2d = w >= 2 && h >= 2 && w.saturating_mul(h) == n;

        let (
            dyn_range,
            eff_dyn_range,
            contrast_collapsed,
            localized_hot_target_rescue,
            glare_ratio,
            occlusion_ratio,
            dead_line_frac,
            fpn_score,
            spatial_autocorr,
            noise_score,
            q_lwir_optical,
        ) = if n == 0 {
            (0.0, 0.0, true, false, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        } else {
            let mut hist = [0usize; 256];
            let mut over_cnt = 0usize;
            let mut under_cnt = 0usize;
            for &px in lwir_u8 {
                hist[px as usize] += 1;
                if px >= 250 {
                    over_cnt += 1;
                }
                if px <= 5 {
                    under_cnt += 1;
                }
            }
            let p05 = hist_percentile_u8(&hist, n, 5, 100);
            let p50 = hist_percentile_u8(&hist, n, 50, 100);
            let p95 = hist_percentile_u8(&hist, n, 95, 100);
            let p999 = hist_percentile_u8(&hist, n, 999, 1000);
            let dyn_range = (p95 - p05).max(0.0);

            let over_ratio = (over_cnt as f64) / (n as f64);
            let under_ratio = (under_cnt as f64) / (n as f64);

            let (dlf, fs, ac, ns) = if let Some(pre) = precomputed_fpn_noise {
                pre
            } else if has_2d {
                let (dlf, fs) = detect_dead_line_fpn_u8(lwir_u8, w, h);
                let (ac, ns) = detect_white_noise_fault_u8(lwir_u8, w, h);
                (dlf, fs, ac, ns)
            } else {
                (0.0, 1.0, 1.0, 1.0)
            };

            let sensor_clean_for_peak =
                dlf < FPN_DEAD_LINE_FLAG_FRAC && ns >= EO_SECTOR_DEGRADED_THR;

            // 2x2 morphological min-filter rejects isolated 1-pixel hot noise spikes while
            // preserving genuine >=2x2 hot targets on cold night/overcast skies.
            let min2x2_max = if has_2d {
                let mut best = 0u8;
                for y in 0..(h - 1) {
                    let r0 = y * w;
                    let r1 = (y + 1) * w;
                    for x in 0..(w - 1) {
                        let m = lwir_u8[r0 + x]
                            .min(lwir_u8[r0 + x + 1])
                            .min(lwir_u8[r1 + x])
                            .min(lwir_u8[r1 + x + 1]);
                        if m > best {
                            best = m;
                        }
                    }
                }
                best as f64
            } else {
                p999
            };

            let mut mad_hist = [0usize; 256];
            let p50_i = p50.round() as i32;
            for (val, &cnt) in hist.iter().enumerate() {
                if cnt > 0 {
                    let d = ((val as i32) - p50_i).unsigned_abs() as usize;
                    mad_hist[d.min(255)] += cnt;
                }
            }
            let bg_mad = hist_percentile_u8(&mad_hist, n, 50, 100);
            let bg_sigma = (1.4826 * bg_mad).max(1.0);

            let peak_val = min2x2_max;
            let peak_above_bg = (peak_val - p50).max(0.0);
            let peak_snr = peak_above_bg / bg_sigma;

            let has_localized = sensor_clean_for_peak
                && peak_val >= LWIR_LOCAL_PEAK_MIN_DN
                && peak_above_bg >= LWIR_LOCAL_PEAK_DELTA_DN
                && (peak_snr >= LWIR_LOCAL_PEAK_SNR_MIN || peak_above_bg >= 35.0);

            let eff_dyn = if has_localized {
                dyn_range.max(peak_above_bg)
            } else {
                dyn_range
            };
            let glare_score = (1.0 - (over_ratio - 0.02).max(0.0) / 0.48).clamp(0.0, 1.0);
            let occ_score = if has_localized {
                1.0
            } else {
                (1.0 - (under_ratio - 0.10).max(0.0) / 0.65).clamp(0.0, 1.0)
            };
            let contrast_score = (eff_dyn / LWIR_CONTRAST_REF_RANGE).clamp(0.0, 1.0);
            let collapsed = eff_dyn < LWIR_CONTRAST_COLLAPSE_P95_P05_THR;
            let q_opt = (contrast_score * glare_score * occ_score * fs * ns).clamp(0.0, 1.0);

            (
                dyn_range,
                eff_dyn,
                collapsed,
                has_localized,
                over_ratio,
                under_ratio,
                dlf,
                fs,
                ac,
                ns,
                q_opt,
            )
        };

        self.lwir_dynamic_range_p95_p05 = round2(dyn_range);
        self.lwir_effective_dynamic_range = round2(eff_dyn_range);
        self.lwir_contrast_collapsed = contrast_collapsed;
        self.lwir_localized_hot_target_rescue = localized_hot_target_rescue;
        self.lwir_glare_ratio = round4(glare_ratio);
        self.lwir_occlusion_ratio = round4(occlusion_ratio);
        self.lwir_dead_line_fraction = round4(dead_line_frac);
        self.lwir_fpn_score = round4(fpn_score);
        self.lwir_spatial_autocorr = round4(spatial_autocorr);
        self.lwir_noise_score = round4(noise_score);
        self.lwir_dead_pixel_fpn_fault = dead_line_frac >= FPN_DEAD_LINE_FLAG_FRAC;
        self.lwir_sensor_noise_fault = noise_score < EO_SECTOR_DEGRADED_THR;
        self.lwir_q_optical = round4(q_lwir_optical);

        let mad = if let Some(prev) = self.prev_lwir_u8.as_ref() {
            if prev.len() == lwir_u8.len() && !lwir_u8.is_empty() {
                let mut acc = 0.0_f64;
                for (&a, &b) in lwir_u8.iter().zip(prev.iter()) {
                    acc += ((a as i32) - (b as i32)).abs() as f64;
                }
                let m = acc / (lwir_u8.len() as f64);
                if m < LWIR_NUC_FREEZE_MAD_THR {
                    self.lwir_freeze_ticks = self.lwir_freeze_ticks.saturating_add(1);
                } else {
                    self.lwir_freeze_ticks = 0;
                }
                self.lwir_nuc_frozen = self.lwir_freeze_ticks >= LWIR_NUC_FREEZE_MIN_TICKS;
                m
            } else {
                self.lwir_freeze_ticks = 0;
                self.lwir_nuc_frozen = false;
                -1.0
            }
        } else {
            self.lwir_freeze_ticks = 0;
            self.lwir_nuc_frozen = false;
            -1.0
        };
        self.lwir_last_mad = mad;
        self.prev_lwir_u8 = Some(lwir_u8.to_vec());

        let mask_lwir_frame = self.lwir_nuc_frozen
            || ffc_imminent
            || ffc_in_progress
            || contrast_collapsed
            || self.lwir_dead_pixel_fpn_fault
            || self.lwir_sensor_noise_fault;
        let degraded = mask_lwir_frame || self.lwir_q_optical < EO_SECTOR_DEGRADED_THR;
        self.lwir_degraded = degraded;

        LwirHealthStatus {
            nuc_frozen: self.lwir_nuc_frozen,
            freeze_ticks: self.lwir_freeze_ticks,
            last_mad: mad,
            ffc_imminent,
            ffc_in_progress,
            dynamic_range_p95_p05: self.lwir_dynamic_range_p95_p05,
            effective_dynamic_range: self.lwir_effective_dynamic_range,
            contrast_collapsed,
            localized_hot_target_rescue,
            glare_ratio: self.lwir_glare_ratio,
            occlusion_ratio: self.lwir_occlusion_ratio,
            dead_line_fraction: self.lwir_dead_line_fraction,
            fpn_score: self.lwir_fpn_score,
            spatial_autocorr: self.lwir_spatial_autocorr,
            noise_score: self.lwir_noise_score,
            dead_pixel_fpn_fault: self.lwir_dead_pixel_fpn_fault,
            sensor_noise_fault: self.lwir_sensor_noise_fault,
            q_lwir_optical: self.lwir_q_optical,
            degraded,
            mask_lwir_frame,
        }
    }

    /// Merge native-resolution FPN and white-noise guards from an RGB LWIR source image before
    /// Triangle downsampling can smear 1px dead-pixel grids or correlate white noise.
    pub fn apply_lwir_native_rgb_fpn_noise_guard(&mut self, img: &RgbImage) {
        let (nat_u8, nw, nh) = extract_native_luma_u8_window(img, 640, 480);
        if nw < 6 || nh < 6 {
            return;
        }
        let (nat_dlf, nat_fpn) = detect_dead_line_fpn_u8(&nat_u8, nw, nh);
        let (nat_ac, nat_noise) = detect_white_noise_fault_u8(&nat_u8, nw, nh);

        let prev_mult = (self.lwir_fpn_score * self.lwir_noise_score).max(1e-9);
        let new_dlf = self.lwir_dead_line_fraction.max(nat_dlf);
        let new_fpn = self.lwir_fpn_score.min(nat_fpn).clamp(0.0, 1.0);
        let new_ac = self.lwir_spatial_autocorr.min(nat_ac);
        let new_noise = self.lwir_noise_score.min(nat_noise).clamp(0.0, 1.0);
        let new_mult = (new_fpn * new_noise).clamp(0.0, 1.0);
        let scale = if self.lwir_fpn_score * self.lwir_noise_score <= 0.0 {
            0.0
        } else {
            (new_mult / prev_mult).min(1.0)
        };

        self.lwir_dead_line_fraction = round4(new_dlf);
        self.lwir_fpn_score = round4(new_fpn);
        self.lwir_spatial_autocorr = round4(new_ac);
        self.lwir_noise_score = round4(new_noise);
        self.lwir_dead_pixel_fpn_fault = new_dlf >= FPN_DEAD_LINE_FLAG_FRAC;
        self.lwir_sensor_noise_fault = new_noise < EO_SECTOR_DEGRADED_THR;
        self.lwir_q_optical = round4((self.lwir_q_optical * scale).clamp(0.0, 1.0));
        if self.lwir_dead_pixel_fpn_fault || self.lwir_sensor_noise_fault {
            self.lwir_localized_hot_target_rescue = false;
            self.lwir_degraded = true;
        }
    }

    /// Observe LWIR frame with hardware `FFC_IMMINENT`/`FFC_IN_PROGRESS` bitmask + `p95-p05` contrast-collapse backstop
    /// + consecutive `MAD < 0.25` DMA-freeze detector (infers `160x128` 2D geometry when `lwir_u8.len() == 160 * 128`).
    pub fn observe_lwir_frame_with_ffc(
        &mut self,
        lwir_u8: &[u8],
        ffc_bits: u8,
    ) -> LwirHealthStatus {
        let (w, h) = if lwir_u8.len() == 160 * 128 {
            (160_usize, 128_usize)
        } else {
            (0_usize, 0_usize)
        };
        self.observe_lwir_plane_with_ffc(lwir_u8, w, h, ffc_bits)
    }

    /// Detect LWIR microbolometer NUC shutter freeze (`MAD < 0.25` for `>= 3` consecutive ticks)
    /// while updating the `p95 - p05` contrast-collapse backstop (backward-compatible signature).
    pub fn observe_lwir_frame(&mut self, lwir_u8: &[u8]) -> (bool, u32, f64) {
        let st = self.observe_lwir_frame_with_ffc(lwir_u8, 0);
        (st.nuc_frozen, st.freeze_ticks, st.last_mad)
    }

    /// Guard strict camera timestamp monotonicity, `<= 1.0 ms` FSYNC warning tier, and `<= 200.0 ms` hard breaker.
    pub fn observe_timestamps(&mut self, cam_ts_ms: f64, imu_ts_ms: f64) -> (f64, bool, bool) {
        let res = self.timestamp_guard.check(cam_ts_ms, imu_ts_ms);
        self.timestamp_skew_ms = res.skew_ms;
        self.timestamp_monotonic_ok = res.monotonic_ok;
        self.timestamp_skew_fault = res.skew_fault;
        self.fsync_precision_ok = res.fsync_precision_ok;
        (res.skew_ms, res.monotonic_ok, res.skew_fault)
    }

    /// Validate camera exposure midpoint `(t_sof + 0.5 * t_exp)` against IMU timestamp with dual-tier FSYNC guard.
    pub fn observe_exposure_timestamps(
        &mut self,
        t_sof_ms: f64,
        exposure_ms: f64,
        imu_ts_ms: f64,
    ) -> TimestampCheckResult {
        let res = self
            .timestamp_guard
            .check_exposure_midpoint(t_sof_ms, exposure_ms, imu_ts_ms);
        self.timestamp_skew_ms = res.skew_ms;
        self.timestamp_monotonic_ok = res.monotonic_ok;
        self.timestamp_skew_fault = res.skew_fault;
        self.fsync_precision_ok = res.fsync_precision_ok;
        res
    }

    /// Compute fail-closed effective sensor qualities coupling optical blur/glare/occlusion/FPN/noise,
    /// hardware LWIR FFC bits + contrast-collapse backstop + NUC shutter freeze, and timestamp skew guards.
    pub fn compute_effective_qualities(&self, raw_q_eo: f64, raw_q_lwir: f64) -> (f64, f64) {
        if self.timestamp_skew_fault {
            return (0.0, 0.0);
        }
        let q_eo_base = if raw_q_eo.is_finite() {
            raw_q_eo.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q_lwir_base = if raw_q_lwir.is_finite() {
            raw_q_lwir.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q_eo_eff = (q_eo_base * self.eo_q_optical).clamp(0.0, 1.0);
        let q_lwir_eff = if self.lwir_nuc_frozen
            || self.lwir_ffc_imminent
            || self.lwir_ffc_in_progress
            || self.lwir_contrast_collapsed
        {
            0.0
        } else {
            (q_lwir_base * self.lwir_q_optical).clamp(0.0, 1.0)
        };
        (round4(q_eo_eff), round4(q_lwir_eff))
    }
}

#[inline]
fn round2(v: f64) -> f64 {
    if v.is_finite() {
        (v * 100.0).round() / 100.0
    } else {
        0.0
    }
}

#[inline]
fn round4(v: f64) -> f64 {
    if v.is_finite() {
        (v * 10_000.0).round() / 10_000.0
    } else {
        0.0
    }
}

// ---- Safe ORT Dynamic Library Resolver (Defeats Win11 System32\onnxruntime.dll Hijack) ----

pub fn is_forbidden_system_ort_path(path: &Path) -> bool {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    lower.contains("\\windows\\system32\\")
        || lower.contains("/windows/system32/")
        || lower.contains("\\windows\\syswow64\\")
        || lower.contains("/windows/syswow64/")
        || lower.contains("\\windows\\winsxs\\")
        || lower.contains("/windows/winsxs/")
        || lower.ends_with("\\windows\\system32")
        || lower.ends_with("/windows/system32")
}

pub fn resolve_safe_ort_dylib_path_with_roots(
    explicit_env: Option<&str>,
    exe_dir: Option<&Path>,
    cwd: Option<&Path>,
) -> Result<PathBuf, String> {
    if let Some(raw) = explicit_env {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let p = PathBuf::from(trimmed);
            if is_forbidden_system_ort_path(&p) {
                return Err(
                    "Blocked forbidden System32/SysWOW64 onnxruntime.dll path (Win11 hijack guard)"
                        .to_string(),
                );
            }
            if p.is_file() {
                return Ok(p);
            }
            return Err("Explicit ORT_DYLIB_PATH does not point to an existing library".to_string());
        }
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    let lib_names: &[&str] = if cfg!(target_os = "windows") {
        &["onnxruntime.dll"]
    } else if cfg!(target_os = "macos") {
        &["libonnxruntime.dylib"]
    } else {
        &["libonnxruntime.so"]
    };

    if let Some(dir) = exe_dir {
        for name in lib_names {
            candidates.push(dir.join(name));
            candidates.push(dir.join("lib").join(name));
            candidates.push(dir.join("resources").join(name));
        }
    }

    // Walk up runtime directories to discover dev `.venv` ORT DLL without baking host paths
    for start in [cwd, exe_dir].into_iter().flatten() {
        let mut cur = Some(start);
        for _ in 0..7 {
            if let Some(base) = cur {
                if cfg!(target_os = "windows") {
                    candidates.push(
                        base.join(".venv")
                            .join("Lib")
                            .join("site-packages")
                            .join("onnxruntime")
                            .join("capi")
                            .join("onnxruntime.dll"),
                    );
                } else {
                    candidates.push(base.join("lib").join("libonnxruntime.so"));
                }
                cur = base.parent();
            }
        }
    }

    for p in candidates {
        if !is_forbidden_system_ort_path(&p) && p.is_file() {
            return Ok(p);
        }
    }

    Err("Bundled ONNX Runtime shared library not found; place onnxruntime.dll beside executable or set ORT_DYLIB_PATH".to_string())
}

pub fn resolve_safe_ort_dylib_path() -> Result<PathBuf, String> {
    let env_val = std::env::var("ORT_DYLIB_PATH").ok();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();
    resolve_safe_ort_dylib_path_with_roots(
        env_val.as_deref(),
        exe_dir.as_deref(),
        cwd.as_deref(),
    )
}

pub fn find_relative_asset(rel_candidates: &[&str]) -> Option<PathBuf> {
    if let Ok(model_dir) = std::env::var("OPENXVS_MODEL_DIR") {
        let base = PathBuf::from(model_dir.trim());
        for rel in rel_candidates {
            if let Some(fname) = Path::new(rel).file_name() {
                let p = base.join(fname);
                if p.is_file() {
                    return Some(p);
                }
            }
            let p = base.join(rel);
            if p.is_file() {
                return Some(p);
            }
        }
    }

    if let Ok(media_dir) = std::env::var("OPENXVS_MEDIA_DIR") {
        let base = PathBuf::from(media_dir.trim());
        for rel in rel_candidates {
            let trimmed = rel
                .strip_prefix("media/")
                .or_else(|| rel.strip_prefix("data/studio_media/"))
                .unwrap_or(rel);
            let p = base.join(trimmed);
            if p.is_file() {
                return Some(p);
            }
        }
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();

    if let Some(dir) = exe_dir.as_deref() {
        for rel in rel_candidates {
            // Direct relative path beside executable (e.g. `media/eo_sky/eo_sky_f000.jpg` or `models/...`)
            let direct = dir.join(rel);
            if direct.is_file() {
                return Some(direct);
            }
            let trimmed_media = rel
                .strip_prefix("data/studio_media/")
                .or_else(|| rel.strip_prefix("media/"));
            if let Some(sub_media) = trimmed_media {
                for media_prefix in ["media", "resources/media"] {
                    let p = dir.join(media_prefix).join(sub_media);
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
            if let Some(fname) = Path::new(rel).file_name() {
                for sub in ["models", "resources/models", "media", "resources/media", "."] {
                    let p = dir.join(sub).join(fname);
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
        }
    }

    for start in [cwd.as_deref(), exe_dir.as_deref()].into_iter().flatten() {
        let mut cur = Some(start);
        for _ in 0..7 {
            if let Some(base) = cur {
                for rel in rel_candidates {
                    let p = base.join(rel);
                    if p.is_file() {
                        return Some(p);
                    }
                    if let Some(sub_media) = rel.strip_prefix("media/") {
                        let p2 = base.join("data").join("studio_media").join(sub_media);
                        if p2.is_file() {
                            return Some(p2);
                        }
                    }
                }
                cur = base.parent();
            }
        }
    }
    None
}

pub fn sha256_bytes_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

pub fn sha256_file_hex(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read error: {e}"))?;
    Ok(sha256_bytes_hex(&bytes))
}

/// Fail-closed SHA-256 verification against an exact expected digest.
pub fn verify_onnx_sha256_pin(path: &Path, expected_sha256: &str) -> Result<String, String> {
    let actual = sha256_file_hex(path)?;
    if actual.eq_ignore_ascii_case(expected_sha256.trim()) {
        Ok(actual)
    } else {
        Err(format!(
            "SHA256_MISMATCH_REJECTED: actual={actual} expected={expected_sha256}"
        ))
    }
}

/// Fail-closed SHA-256 verification against a model's allowed `(filename, expected_sha256)` pin table.
pub fn verify_pinned_onnx_artifact(
    path: &Path,
    allowed_pins: &[(&str, &str)],
) -> Result<(String, String), (Option<String>, Option<String>)> {
    let fname = path.file_name().map(|s| s.to_string_lossy().to_string());
    let actual_sha = sha256_file_hex(path).ok();
    let Some(ref name) = fname else {
        return Err((None, actual_sha));
    };
    let Some(ref sha) = actual_sha else {
        return Err((fname, None));
    };
    for &(pinned_name, pinned_sha) in allowed_pins {
        if name == pinned_name && sha.eq_ignore_ascii_case(pinned_sha) {
            return Ok((name.clone(), sha.clone()));
        }
    }
    Err((fname, actual_sha))
}

static GLOBAL_ORT_ENV: OnceLock<Result<(Environment, String), String>> = OnceLock::new();

fn acquire_global_ort_env() -> Result<(Environment, String), String> {
    GLOBAL_ORT_ENV
        .get_or_init(|| {
            let dll_path = resolve_safe_ort_dylib_path().map_err(|e| format!("NOT_FOUND: {e}"))?;
            let fname = dll_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "onnxruntime".to_string());
            let builder = ort::init_from(&dll_path).map_err(|e| format!("LOAD_ERROR: {e}"))?;
            let env = builder.build().map_err(|e| format!("ENV_BUILD_ERROR: {e}"))?;
            Ok((env, format!("{fname} (load-dynamic, System32-blocked)")))
        })
        .clone()
}

fn resolve_ort_thread_budget() -> usize {
    std::env::var("OMP_NUM_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .map(|n| n.clamp(1, 6))
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| (n.get() / 2).clamp(1, 2))
                .unwrap_or(2)
        })
}

fn build_optimized_ort_session(
    env: &Environment,
    path: &Path,
    max_intra_threads: usize,
) -> ort::Result<Session> {
    let intra_threads = resolve_ort_thread_budget().min(max_intra_threads.max(1));
    let mut builder = Session::builder(env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra_threads)?
        .with_inter_threads(1)?
        .with_intra_op_spinning(false)?
        .with_inter_op_spinning(false)?
        .with_dynamic_block_base(4)?
        .with_memory_pattern(true)?;
    builder.commit_from_file(path)
}

#[allow(clippy::too_many_arguments)]
fn load_pinned_onnx_session(
    ort_env: Option<&Environment>,
    display_name: &str,
    default_filename: &str,
    ready_status: &str,
    missing_status: &str,
    rel_candidates: &[&str],
    allowed_pins: &[(&str, &str)],
    max_intra_threads: usize,
) -> (Option<Session>, ModelStatusEntry) {
    let Some(env) = ort_env else {
        return (
            None,
            ModelStatusEntry {
                name: display_name.to_string(),
                loaded: false,
                real_weights: false,
                device: "cpu-ort-rust".to_string(),
                backend: "ort-load-dynamic".to_string(),
                status: missing_status.to_string(),
                sha256: None,
                expected_sha256: allowed_pins.first().map(|(_, s)| (*s).to_string()),
                filename: Some(default_filename.to_string()),
            },
        );
    };

    let Some(path) = find_relative_asset(rel_candidates) else {
        return (
            None,
            ModelStatusEntry {
                name: display_name.to_string(),
                loaded: false,
                real_weights: false,
                device: "cpu-ort-rust".to_string(),
                backend: "ort-load-dynamic".to_string(),
                status: missing_status.to_string(),
                sha256: None,
                expected_sha256: allowed_pins.first().map(|(_, s)| (*s).to_string()),
                filename: Some(default_filename.to_string()),
            },
        );
    };

    match verify_pinned_onnx_artifact(&path, allowed_pins) {
        Err((fname, actual_sha)) => {
            let expected = fname
                .as_deref()
                .and_then(|n| allowed_pins.iter().find(|(pn, _)| *pn == n).map(|(_, s)| *s))
                .or_else(|| allowed_pins.first().map(|(_, s)| *s))
                .map(str::to_string);
            (
                None,
                ModelStatusEntry {
                    name: display_name.to_string(),
                    loaded: false,
                    real_weights: false,
                    device: "cpu-ort-rust".to_string(),
                    backend: "ort-load-dynamic".to_string(),
                    status: "SHA256_MISMATCH_REJECTED".to_string(),
                    sha256: actual_sha,
                    expected_sha256: expected,
                    filename: fname,
                },
            )
        }
        Ok((fname, verified_sha)) => {
            match build_optimized_ort_session(env, &path, max_intra_threads) {
                Ok(sess) => (
                    Some(sess),
                    ModelStatusEntry {
                        name: display_name.to_string(),
                        loaded: true,
                        real_weights: true,
                        device: "cpu-ort-rust".to_string(),
                        backend: "ort-load-dynamic".to_string(),
                        status: ready_status.to_string(),
                        sha256: Some(verified_sha.clone()),
                        expected_sha256: Some(verified_sha),
                        filename: Some(fname),
                    },
                ),
                Err(e) => (
                    None,
                    ModelStatusEntry {
                        name: display_name.to_string(),
                        loaded: false,
                        real_weights: false,
                        device: "cpu-ort-rust".to_string(),
                        backend: "ort-load-dynamic".to_string(),
                        status: format!("SESSION_ERROR: {e}"),
                        sha256: Some(verified_sha.clone()),
                        expected_sha256: Some(verified_sha),
                        filename: Some(fname),
                    },
                ),
            }
        }
    }
}

// ---- Real Bundled Media Library Discovery (`media/` beside `.exe` OR `data/studio_media/`) ----

#[derive(Debug, Clone)]
struct MediaFrameEntry {
    rel_file: String,
    path: PathBuf,
    secondary_path: Option<PathBuf>,
    regime: Option<String>,
    q_eo_expected: Option<f32>,
    q_lwir_expected: Option<f32>,
    spoof_offset_m: Option<f64>,
}

struct DecodedThermalFrameArc {
    eo: Arc<RgbImage>,
    lwir: Option<Arc<RgbImage>>,
    rel_file: String,
    total_frames: usize,
    q_eo_expected: Option<f32>,
    q_lwir_expected: Option<f32>,
}

struct DecodedNavFrameArc {
    aerial: Arc<RgbImage>,
    sat_ref: Option<Arc<RgbImage>>,
    sat_ref_path: Option<PathBuf>,
    rel_file: String,
    total_frames: usize,
    spoof_offset_m: Option<f64>,
}

#[cfg(test)]
struct DecodedThermalFrame {
    eo: RgbImage,
    lwir: Option<RgbImage>,
}

#[cfg(test)]
struct DecodedNavFrame {
    aerial: RgbImage,
    sat_ref: Option<RgbImage>,
    rel_file: String,
    total_frames: usize,
    spoof_offset_m: Option<f64>,
}

#[derive(Debug, Default)]
struct LruImageCache {
    cap: usize,
    map: HashMap<PathBuf, Arc<RgbImage>>,
    order: VecDeque<PathBuf>,
}

impl LruImageCache {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            map: HashMap::with_capacity(cap.max(1)),
            order: VecDeque::with_capacity(cap.max(1)),
        }
    }

    fn get(&mut self, key: &Path) -> Option<Arc<RgbImage>> {
        let img = self.map.get(key).map(Arc::clone)?;
        if let Some(pos) = self.order.iter().position(|p| p == key) {
            if let Some(k) = self.order.remove(pos) {
                self.order.push_back(k);
            }
        }
        Some(img)
    }

    fn insert(&mut self, key: PathBuf, img: Arc<RgbImage>) {
        if self.map.contains_key(&key) {
            if let Some(pos) = self.order.iter().position(|p| p == &key) {
                self.order.remove(pos);
            }
        } else if self.map.len() >= self.cap {
            if let Some(evict) = self.order.pop_front() {
                self.map.remove(&evict);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, img);
    }
}

#[derive(Debug)]
struct StudioMediaLibrary {
    eo_sky: Vec<MediaFrameEntry>,
    runway_land: Vec<MediaFrameEntry>,
    thermal_dve: Vec<MediaFrameEntry>,
    crossview_nav: Vec<MediaFrameEntry>,
    decode_cache: Mutex<LruImageCache>,
}

impl Default for StudioMediaLibrary {
    fn default() -> Self {
        Self {
            eo_sky: Vec::new(),
            runway_land: Vec::new(),
            thermal_dve: Vec::new(),
            crossview_nav: Vec::new(),
            decode_cache: Mutex::new(LruImageCache::new(MEDIA_LRU_CAPACITY)),
        }
    }
}

impl StudioMediaLibrary {
    fn discover() -> Self {
        let Some(root) = discover_media_root_dir() else {
            return Self::default();
        };
        let manifest_json: Option<serde_json::Value> = std::fs::read_to_string(root.join("manifest.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());

        let eo_sky = Self::load_channel(&root, manifest_json.as_ref(), "eo_sky", None);
        let runway_land = Self::load_channel(&root, manifest_json.as_ref(), "runway_land", None);
        let thermal_dve = Self::load_thermal_channel(&root, manifest_json.as_ref());
        let crossview_nav = Self::load_nav_channel(&root, manifest_json.as_ref());

        Self {
            eo_sky,
            runway_land,
            thermal_dve,
            crossview_nav,
            decode_cache: Mutex::new(LruImageCache::new(MEDIA_LRU_CAPACITY)),
        }
    }

    fn load_channel(
        root: &Path,
        manifest: Option<&serde_json::Value>,
        channel: &str,
        prefix_filter: Option<&str>,
    ) -> Vec<MediaFrameEntry> {
        let mut out = Vec::new();
        if let Some(frames) = manifest
            .and_then(|m| m.get("channels"))
            .and_then(|c| c.get(channel))
            .and_then(|ch| ch.get("frames"))
            .and_then(|f| f.as_array())
        {
            for entry in frames {
                if let Some(rel) = entry.get("file").and_then(|v| v.as_str()) {
                    let p = root.join(rel);
                    if p.is_file() {
                        out.push(MediaFrameEntry {
                            rel_file: rel.to_string(),
                            path: p,
                            secondary_path: None,
                            regime: None,
                            q_eo_expected: None,
                            q_lwir_expected: None,
                            spoof_offset_m: None,
                        });
                    }
                }
            }
        }
        if !out.is_empty() {
            return out;
        }
        // Fallback directory scan if manifest absent or compact subset without manifest entries
        let subdir = root.join(channel);
        if let Ok(rd) = std::fs::read_dir(&subdir) {
            let mut files: Vec<PathBuf> = rd
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.is_file()
                        && matches!(
                            p.extension()
                                .and_then(|e| e.to_str())
                                .map(|s| s.to_ascii_lowercase())
                                .as_deref(),
                            Some("jpg" | "jpeg" | "png")
                        )
                })
                .filter(|p| match prefix_filter {
                    Some(pref) => p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with(pref))
                        .unwrap_or(false),
                    None => true,
                })
                .collect();
            files.sort();
            for p in files {
                let fname = p
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                out.push(MediaFrameEntry {
                    rel_file: format!("{channel}/{fname}"),
                    path: p,
                    secondary_path: None,
                    regime: None,
                    q_eo_expected: None,
                    q_lwir_expected: None,
                    spoof_offset_m: None,
                });
            }
        }
        out
    }

    fn load_thermal_channel(
        root: &Path,
        manifest: Option<&serde_json::Value>,
    ) -> Vec<MediaFrameEntry> {
        let mut out = Vec::new();
        if let Some(frames) = manifest
            .and_then(|m| m.get("channels"))
            .and_then(|c| c.get("thermal_dve"))
            .and_then(|ch| ch.get("frames"))
            .and_then(|f| f.as_array())
        {
            for entry in frames {
                let eo_rel = entry
                    .get("eo_file")
                    .or_else(|| entry.get("file"))
                    .and_then(|v| v.as_str());
                let lwir_rel = entry.get("lwir_file").and_then(|v| v.as_str());
                if let Some(rel) = eo_rel {
                    let eo_path = root.join(rel);
                    if eo_path.is_file() {
                        let lwir_path = lwir_rel
                            .map(|lr| root.join(lr))
                            .filter(|p| p.is_file());
                        out.push(MediaFrameEntry {
                            rel_file: rel.to_string(),
                            path: eo_path,
                            secondary_path: lwir_path,
                            regime: entry
                                .get("regime")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            q_eo_expected: entry
                                .get("q_eo_expected")
                                .and_then(|v| v.as_f64())
                                .map(|v| v as f32),
                            q_lwir_expected: entry
                                .get("q_lwir_expected")
                                .and_then(|v| v.as_f64())
                                .map(|v| v as f32),
                            spoof_offset_m: None,
                        });
                    }
                }
            }
        }
        if !out.is_empty() {
            return out;
        }
        let eo_list = Self::load_channel(root, None, "thermal_dve", Some("eo_fog_"));
        if !eo_list.is_empty() {
            return eo_list
                .into_iter()
                .map(|mut e| {
                    if let Some(fname) = e.path.file_name().and_then(|n| n.to_str()) {
                        let lwir_name = fname.replacen("eo_fog_", "lwir_", 1);
                        let cand = e.path.with_file_name(lwir_name);
                        if cand.is_file() {
                            e.secondary_path = Some(cand);
                        }
                    }
                    e
                })
                .collect();
        }
        Self::load_channel(root, None, "thermal_dve", None)
    }

    fn load_nav_channel(root: &Path, manifest: Option<&serde_json::Value>) -> Vec<MediaFrameEntry> {
        let mut out = Vec::new();
        if let Some(frames) = manifest
            .and_then(|m| m.get("channels"))
            .and_then(|c| c.get("crossview_nav"))
            .and_then(|ch| ch.get("frames"))
            .and_then(|f| f.as_array())
        {
            for entry in frames {
                let rel_opt = entry
                    .get("aerial_file")
                    .or_else(|| entry.get("file"))
                    .and_then(|v| v.as_str());
                if let Some(rel) = rel_opt {
                    let p = root.join(rel);
                    if p.is_file() {
                        let sat_p = entry
                            .get("sat_ref_file")
                            .and_then(|v| v.as_str())
                            .map(|s| root.join(s))
                            .filter(|sp| sp.is_file());
                        out.push(MediaFrameEntry {
                            rel_file: rel.to_string(),
                            path: p,
                            secondary_path: sat_p,
                            regime: None,
                            q_eo_expected: None,
                            q_lwir_expected: None,
                            spoof_offset_m: entry.get("spoof_offset_m").and_then(|v| v.as_f64()),
                        });
                    }
                }
            }
        }
        if !out.is_empty() {
            return out;
        }
        let aerial = Self::load_channel(root, None, "crossview_nav", Some("aerial_"));
        if !aerial.is_empty() {
            return aerial;
        }
        Self::load_channel(root, None, "crossview_nav", None)
    }

    fn is_available(&self) -> bool {
        !self.eo_sky.is_empty()
            || !self.runway_land.is_empty()
            || !self.thermal_dve.is_empty()
            || !self.crossview_nav.is_empty()
    }

    fn channel_counts(&self) -> BTreeMap<String, usize> {
        let mut m = BTreeMap::new();
        m.insert("eo_sky".to_string(), self.eo_sky.len());
        m.insert("runway_land".to_string(), self.runway_land.len());
        m.insert("thermal_dve".to_string(), self.thermal_dve.len());
        m.insert("crossview_nav".to_string(), self.crossview_nav.len());
        m
    }

    fn decode_rgb_arc(&self, path: &Path) -> Option<Arc<RgbImage>> {
        if let Ok(mut cache) = self.decode_cache.lock() {
            if let Some(hit) = cache.get(path) {
                return Some(hit);
            }
        }
        let raw = std::fs::read(path).ok()?;
        let img = image::load_from_memory(&raw).ok()?;
        let arc = Arc::new(img.to_rgb8());
        if let Ok(mut cache) = self.decode_cache.lock() {
            cache.insert(path.to_path_buf(), Arc::clone(&arc));
        }
        Some(arc)
    }

    fn pick_eo_sky_arc(&self, frame: u32) -> Option<(Arc<RgbImage>, String, usize)> {
        if self.eo_sky.is_empty() {
            return None;
        }
        let idx = (frame as usize) % self.eo_sky.len();
        let entry = &self.eo_sky[idx];
        let img = self.decode_rgb_arc(&entry.path)?;
        Some((img, entry.rel_file.clone(), self.eo_sky.len()))
    }

    fn pick_runway_land_arc(&self, frame: u32) -> Option<(Arc<RgbImage>, String, usize)> {
        if self.runway_land.is_empty() {
            return None;
        }
        let idx = (frame as usize) % self.runway_land.len();
        let entry = &self.runway_land[idx];
        let img = self.decode_rgb_arc(&entry.path)?;
        Some((img, entry.rel_file.clone(), self.runway_land.len()))
    }

    fn pick_thermal_for_scenario_arc(
        &self,
        scenario: &str,
        frame: u32,
    ) -> Option<DecodedThermalFrameArc> {
        if self.thermal_dve.is_empty() {
            return None;
        }
        let matching_indices: Vec<usize> = if scenario == "thermal_fog" {
            self.thermal_dve
                .iter()
                .enumerate()
                .filter(|(_, e)| e.regime.as_deref() == Some("radiation_fog"))
                .map(|(i, _)| i)
                .collect()
        } else {
            self.thermal_dve
                .iter()
                .enumerate()
                .filter(|(_, e)| e.regime.as_deref() == Some("clear"))
                .map(|(i, _)| i)
                .collect()
        };
        let idx = if !matching_indices.is_empty() {
            matching_indices[(frame as usize) % matching_indices.len()]
        } else {
            (frame as usize) % self.thermal_dve.len()
        };
        self.decode_thermal_arc_at(idx)
    }

    fn pick_thermal_seq_arc(&self, frame: u32) -> Option<DecodedThermalFrameArc> {
        if self.thermal_dve.is_empty() {
            return None;
        }
        let idx = (frame as usize) % self.thermal_dve.len();
        self.decode_thermal_arc_at(idx)
    }

    fn decode_thermal_arc_at(&self, idx: usize) -> Option<DecodedThermalFrameArc> {
        let entry = self.thermal_dve.get(idx)?;
        let eo = self.decode_rgb_arc(&entry.path)?;
        let lwir = entry
            .secondary_path
            .as_deref()
            .and_then(|p| self.decode_rgb_arc(p));
        Some(DecodedThermalFrameArc {
            eo,
            lwir,
            rel_file: entry.rel_file.clone(),
            total_frames: self.thermal_dve.len(),
            q_eo_expected: entry.q_eo_expected,
            q_lwir_expected: entry.q_lwir_expected,
        })
    }

    fn pick_nav_pair_arc(&self, frame: u32) -> Option<DecodedNavFrameArc> {
        if self.crossview_nav.is_empty() {
            return None;
        }
        let idx = (frame as usize) % self.crossview_nav.len();
        let entry = &self.crossview_nav[idx];
        let aerial = self.decode_rgb_arc(&entry.path)?;
        let sat_ref = entry
            .secondary_path
            .as_deref()
            .and_then(|p| self.decode_rgb_arc(p));
        Some(DecodedNavFrameArc {
            aerial,
            sat_ref,
            sat_ref_path: entry.secondary_path.clone(),
            rel_file: entry.rel_file.clone(),
            total_frames: self.crossview_nav.len(),
            spoof_offset_m: entry.spoof_offset_m,
        })
    }

    #[cfg(test)]
    fn pick_eo_sky(&self, frame: u32) -> Option<(RgbImage, String, usize)> {
        self.pick_eo_sky_arc(frame)
            .map(|(arc, rel, tot)| ((*arc).clone(), rel, tot))
    }

    #[cfg(test)]
    fn pick_runway_land(&self, frame: u32) -> Option<(RgbImage, String, usize)> {
        self.pick_runway_land_arc(frame)
            .map(|(arc, rel, tot)| ((*arc).clone(), rel, tot))
    }

    #[cfg(test)]
    fn pick_thermal_seq(&self, frame: u32) -> Option<DecodedThermalFrame> {
        self.pick_thermal_seq_arc(frame).map(|df| DecodedThermalFrame {
            eo: (*df.eo).clone(),
            lwir: df.lwir.map(|a| (*a).clone()),
        })
    }

    #[cfg(test)]
    fn pick_nav_pair(&self, frame: u32) -> Option<DecodedNavFrame> {
        self.pick_nav_pair_arc(frame).map(|df| DecodedNavFrame {
            aerial: (*df.aerial).clone(),
            sat_ref: df.sat_ref.map(|a| (*a).clone()),
            rel_file: df.rel_file,
            total_frames: df.total_frames,
            spoof_offset_m: df.spoof_offset_m,
        })
    }

    #[cfg(test)]
    fn pick_nav(&self, frame: u32) -> Option<(RgbImage, String, usize, Option<f64>)> {
        self.pick_nav_pair(frame)
            .map(|df| (df.aerial, df.rel_file, df.total_frames, df.spoof_offset_m))
    }
}

fn dir_has_studio_media(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    if dir.join("manifest.json").is_file() {
        return true;
    }
    for sub in ["eo_sky", "runway_land", "thermal_dve", "crossview_nav"] {
        let p = dir.join(sub);
        if p.is_dir() {
            if let Ok(mut rd) = std::fs::read_dir(&p) {
                if rd.any(|e| e.ok().map(|x| x.path().is_file()).unwrap_or(false)) {
                    return true;
                }
            }
        }
    }
    false
}

fn discover_media_root_dir() -> Option<PathBuf> {
    if let Ok(env_dir) = std::env::var("OPENXVS_MEDIA_DIR") {
        let p = PathBuf::from(env_dir.trim());
        if dir_has_studio_media(&p) {
            return Some(p);
        }
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();

    if let Some(dir) = exe_dir.as_deref() {
        for sub in ["media", "resources/media"] {
            let p = dir.join(sub);
            if dir_has_studio_media(&p) {
                return Some(p);
            }
        }
    }

    for start in [cwd.as_deref(), exe_dir.as_deref()].into_iter().flatten() {
        let mut cur = Some(start);
        for _ in 0..7 {
            if let Some(base) = cur {
                for rel in ["media", "data/studio_media"] {
                    let p = base.join(rel);
                    if dir_has_studio_media(&p) {
                        return Some(p);
                    }
                }
                cur = base.parent();
            }
        }
    }
    None
}

// ---- JPEG Encoding, Overlay Drawing & Fallback Synthetic Frame Generators ----

fn encode_rgb_jpeg_b64(img: &RgbImage, quality: u8) -> Result<String, String> {
    let mut buf: Vec<u8> = Vec::with_capacity(16 * 1024);
    let mut enc = JpegEncoder::new_with_quality(&mut buf, quality);
    enc.encode_image(img)
        .map_err(|e| format!("JPEG encode failed: {e}"))?;
    Ok(BASE64_STD.encode(&buf))
}

fn draw_rect_outline(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, col: Rgb<u8>) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    if w <= 0 || h <= 0 {
        return;
    }
    let mut xa = x0.clamp(0, w - 1);
    let mut xb = x1.clamp(0, w - 1);
    let mut ya = y0.clamp(0, h - 1);
    let mut yb = y1.clamp(0, h - 1);
    if xa > xb {
        std::mem::swap(&mut xa, &mut xb);
    }
    if ya > yb {
        std::mem::swap(&mut ya, &mut yb);
    }
    for x in xa..=xb {
        img.put_pixel(x as u32, ya as u32, col);
        img.put_pixel(x as u32, yb as u32, col);
    }
    for y in ya..=yb {
        img.put_pixel(xa as u32, y as u32, col);
        img.put_pixel(xb as u32, y as u32, col);
    }
}

fn draw_dot(img: &mut RgbImage, cx: i32, cy: i32, radius: i32, col: Rgb<u8>) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x = cx + dx;
                let y = cy + dy;
                if x >= 0 && x < w && y >= 0 && y < h {
                    img.put_pixel(x as u32, y as u32, col);
                }
            }
        }
    }
}

fn stamp_bottom_sim_only_strip(img: &mut RgbImage) {
    // Non-removable amber/dark safety border on bottom 6px of every exported/rendered preview
    let w = img.width();
    let h = img.height();
    if h < 10 || w < 10 {
        return;
    }
    for y in (h - 6)..h {
        for x in 0..w {
            let stripe = ((x / 8) % 2) == 0;
            let col = if y == h - 6 {
                Rgb([251, 191, 36])
            } else if stripe {
                Rgb([24, 18, 8])
            } else {
                Rgb([52, 34, 10])
            };
            img.put_pixel(x, y, col);
        }
    }
}

fn build_synthetic_sky_frame(frame_idx: u32) -> RgbImage {
    let (w, h) = (384_u32, 256_u32);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        let t = (y as f32) / (h as f32);
        let (r, g, b) = if t < 0.72 {
            (
                (140.0 + 60.0 * t) as u8,
                (165.0 + 48.0 * t) as u8,
                (205.0 + 30.0 * t) as u8,
            )
        } else {
            (58, 72, 64)
        };
        for x in 0..w {
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    // Moving intruder dot in upper sky for D2 ConvGRU temporal response
    let cx = 140 + ((frame_idx * 7) % 104) as i32;
    let cy = 88 + (((frame_idx * 3) % 20) as i32 - 10);
    draw_dot(&mut img, cx, cy, 4, Rgb([28, 32, 38]));
    img
}

fn build_synthetic_runway_frame_at(frame_idx: u32) -> RgbImage {
    let (w, h) = (384_u32, 256_u32);
    let mut img = RgbImage::new(w, h);
    let phase = (frame_idx * 3) % 16;
    for y in 0..h {
        let sky = y < 110;
        for x in 0..w {
            let col = if sky {
                Rgb([135, 165, 200])
            } else {
                Rgb([48, 64, 52])
            };
            img.put_pixel(x, y, col);
        }
    }
    // Perspective runway trapezoid with dynamic approach centerline phase
    for y in 118..244 {
        let frac = ((y - 118) as f32) / 126.0;
        let half_w = (12.0 + 68.0 * frac) as i32;
        let cx = 192_i32;
        for x in (cx - half_w)..=(cx + half_w) {
            if x >= 0 && x < w as i32 {
                let is_center = (x - cx).abs() <= 1 && ((y + phase) / 8).is_multiple_of(2);
                let is_edge = (x - (cx - half_w)).abs() <= 1 || (x - (cx + half_w)).abs() <= 1;
                let col = if is_center || is_edge {
                    Rgb([238, 242, 246])
                } else {
                    Rgb([62, 68, 74])
                };
                img.put_pixel(x as u32, y, col);
            }
        }
    }
    img
}

#[cfg(test)]
fn build_synthetic_runway_frame() -> RgbImage {
    build_synthetic_runway_frame_at(0)
}

// ---- D2 Scene Normalization & Decoding ----

fn scene_normalize_gray_into(gray: &[f32], out: &mut Vec<f32>) {
    let n = gray.len().max(1) as f64;
    let mean = gray.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = gray
        .iter()
        .map(|&v| {
            let d = (v as f64) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = var.sqrt().max(8.0);
    let inv_std = 1.0 / std;
    out.clear();
    out.reserve(gray.len());
    for &v in gray {
        out.push((((v as f64) - mean) * inv_std).clamp(-5.0, 5.0) as f32);
    }
}

fn sigmoid_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + (-x.clamp(-40.0, 40.0)).exp())
}

/// Compute honest signal-derived visual navigation lock confidence combining `XVS-Nav` unit-norm
/// embedding health, `XFeat` keypoint richness, `LighterGlue` mutual match count & mean match
/// probability, and optical integrity (FPN/white-noise gate).
///
/// Yields exact `0.0` on degenerate dimensions (`< 16x16`), non-unit/non-finite embeddings,
/// zero keypoints (when XFeat is loaded), zero mutual matches (when LighterGlue is loaded),
/// or zero optical integrity; yields `0.75..=0.98` on genuine high-inlier aerial/satellite pairs.
#[allow(clippy::too_many_arguments)]
pub fn compute_nav_confidence(
    img_w: u32,
    img_h: u32,
    emb_l2_norm: f64,
    xfeat_loaded: bool,
    xfeat_kpt_count: usize,
    xfeat_mean_score: f64,
    lighterglue_loaded: bool,
    lg_matches: usize,
    lg_mean_prob: f64,
    optical_integrity_gate: f64,
) -> f64 {
    if img_w < 16 || img_h < 16 {
        return 0.0;
    }
    if !emb_l2_norm.is_finite() || (emb_l2_norm - 1.0).abs() > 1e-2 {
        return 0.0;
    }
    if !optical_integrity_gate.is_finite() || optical_integrity_gate <= 0.0 {
        return 0.0;
    }
    if xfeat_loaded && xfeat_kpt_count == 0 {
        return 0.0;
    }
    if lighterglue_loaded && lg_matches == 0 {
        return 0.0;
    }

    let emb_health = (1.0 - (emb_l2_norm - 1.0).abs() / 1e-2).clamp(0.0, 1.0);
    let gate = optical_integrity_gate.clamp(0.0, 1.0);
    let xfeat_q = if xfeat_mean_score.is_finite() {
        (xfeat_mean_score / 0.25).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let raw = if xfeat_loaded && lighterglue_loaded {
        let kpt_cov = ((xfeat_kpt_count as f64) / 64.0).clamp(0.0, 1.0);
        let match_cov = ((lg_matches as f64) / 32.0).clamp(0.0, 1.0);
        let prob_score = if lg_mean_prob.is_finite() {
            lg_mean_prob.clamp(0.0, 1.0)
        } else {
            0.0
        };
        (0.15 * emb_health
            + 0.25 * kpt_cov
            + 0.35 * match_cov
            + 0.20 * prob_score
            + 0.05 * xfeat_q)
            .clamp(0.0, 0.98)
    } else if xfeat_loaded {
        let kpt_cov = ((xfeat_kpt_count as f64) / 64.0).clamp(0.0, 1.0);
        (0.30 * emb_health + 0.50 * kpt_cov + 0.20 * xfeat_q).clamp(0.0, 0.85)
    } else {
        (0.60 * emb_health).clamp(0.0, 0.60)
    };

    round4((raw * gate).clamp(0.0, 0.98))
}

fn decode_d2_top(
    hm: &[f32],
    offset: &[f32],
    log_scale: &[f32],
    peak_mask: &[f32],
    hm_h: usize,
    hm_w: usize,
    score_thr: f32,
) -> (Option<[f32; 5]>, f32) {
    let Some(area) = hm_h.checked_mul(hm_w) else {
        return (None, 0.0);
    };
    if hm_h == 0
        || hm_w == 0
        || hm.len() < area
        || peak_mask.len() < area
        || offset.len() < area.saturating_mul(2)
        || log_scale.len() < area.saturating_mul(2)
    {
        return (None, 0.0);
    }
    let stride = 2.0_f32;
    let mut max_prob = 0.0_f32;
    let mut best_score = -1.0_f32;
    let mut best_xy: Option<(usize, usize)> = None;

    for y in 0..hm_h {
        let row = y * hm_w;
        for x in 0..hm_w {
            let idx = row + x;
            let p = sigmoid_f32(hm[idx]);
            if p > max_prob {
                max_prob = p;
            }
            if p >= score_thr && peak_mask[idx].is_finite() && peak_mask[idx] > 0.5 && p > best_score
            {
                best_score = p;
                best_xy = Some((x, y));
            }
        }
    }

    let Some((cx, cy)) = best_xy else {
        return (None, max_prob);
    };

    let y0 = cy.saturating_sub(1);
    let y1 = (cy + 2).min(hm_h);
    let x0 = cx.saturating_sub(1);
    let x1 = (cx + 2).min(hm_w);
    let mut p_max = f32::NEG_INFINITY;
    for y in y0..y1 {
        let row = y * hm_w;
        for x in x0..x1 {
            let v = hm[row + x];
            if v.is_finite() && v > p_max {
                p_max = v;
            }
        }
    }
    if !p_max.is_finite() {
        return (None, max_prob);
    }
    let mut w_sum = 0.0_f32;
    let mut soft_u = 0.0_f32;
    let mut soft_v = 0.0_f32;
    for y in y0..y1 {
        let row = y * hm_w;
        for x in x0..x1 {
            let v = hm[row + x];
            if !v.is_finite() {
                continue;
            }
            let w = (v - p_max).exp();
            w_sum += w;
            soft_u += w * (x as f32);
            soft_v += w * (y as f32);
        }
    }
    let inv = 1.0 / w_sum.max(1e-12);
    soft_u *= inv;
    soft_v *= inv;

    let cell = cy * hm_w + cx;
    let off_u = offset[cell];
    let off_v = offset[area + cell];
    let ls0 = log_scale[cell];
    let ls1 = log_scale[area + cell];
    if !(off_u.is_finite() && off_v.is_finite() && ls0.is_finite() && ls1.is_finite()) {
        return (None, max_prob);
    }
    let cell_u = 0.5 * ((cx as f32) + off_u) + 0.5 * soft_u;
    let cell_v = 0.5 * ((cy as f32) + off_v) + 0.5 * soft_v;
    let u_px = (cell_u + 0.5) * stride;
    let v_px = (cell_v + 0.5) * stride;

    let s_mom = ls0.clamp(-4.0, 8.0).exp();
    let r_wh = ls1.clamp(-3.0, 3.0).exp();
    let bw = (s_mom * r_wh.sqrt()).clamp(6.0, 96.0);
    let bh = (s_mom / r_wh.sqrt()).clamp(6.0, 96.0);
    (
        Some([
            u_px - 0.5 * bw,
            v_px - 0.5 * bh,
            u_px + 0.5 * bw,
            v_px + 0.5 * bh,
            best_score,
        ]),
        max_prob,
    )
}

// ---- Exact Apache-2.0 XFeat Dense Head Decoder (`stage2_xfeat.py` parity) ----

#[derive(Debug, Clone)]
pub struct XFeatKeypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub desc: [f32; XFEAT_DESC_DIM],
}

pub fn decode_xfeat_dense_heads(
    feats: &[f32],
    keypoints_logits: &[f32],
    heatmap: &[f32],
    max_keypoints: usize,
    nms_threshold: f32,
    nms_radius: usize,
    border_px: usize,
) -> Vec<XFeatKeypoint> {
    let hc = XFEAT_CELL_H;
    let wc = XFEAT_CELL_W;
    let cell_area = hc * wc;
    if feats.len() < XFEAT_DESC_DIM * cell_area
        || keypoints_logits.len() < 65 * cell_area
        || heatmap.len() < cell_area
    {
        return Vec::new();
    }

    let h_img = hc * 8;
    let w_img = wc * 8;
    let mut kmap = vec![0.0_f32; h_img * w_img];

    // 65-way softmax over 8x8 sub-pixel cells + 1 dustbin channel
    let mut probs = [0.0_f32; 65];
    for yc in 0..hc {
        for xc in 0..wc {
            let cell_idx = yc * wc + xc;
            let mut max_l = f32::NEG_INFINITY;
            for c in 0..65 {
                let v = keypoints_logits[c * cell_area + cell_idx];
                if v > max_l {
                    max_l = v;
                }
            }
            let mut sum_exp = 0.0_f32;
            for c in 0..65 {
                let e = (keypoints_logits[c * cell_area + cell_idx] - max_l).exp();
                probs[c] = e;
                sum_exp += e;
            }
            let inv_sum = 1.0_f32 / sum_exp.max(1e-12);
            for sy in 0..8 {
                let y = yc * 8 + sy;
                let row_off = y * w_img;
                for sx in 0..8 {
                    let x = xc * 8 + sx;
                    kmap[row_off + x] = probs[sy * 8 + sx] * inv_sum;
                }
            }
        }
    }

    // Bilinear upsampling of heatmap [32, 40] -> [256, 320] + 5x5 NMS
    let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
    let y_start = border_px.max(nms_radius);
    let y_end = h_img.saturating_sub(border_px.max(nms_radius));
    let x_start = border_px.max(nms_radius);
    let x_end = w_img.saturating_sub(border_px.max(nms_radius));

    for y in y_start..y_end {
        let fy = (((y as f32) + 0.5) * (hc as f32) / (h_img as f32) - 0.5)
            .clamp(0.0, (hc - 1) as f32);
        let y0 = (fy.floor() as usize).min(hc - 1);
        let y1 = (y0 + 1).min(hc - 1);
        let wy = fy - (y0 as f32);

        for x in x_start..x_end {
            let kv = kmap[y * w_img + x];
            if kv <= nms_threshold {
                continue;
            }
            let mut is_max = true;
            'nms: for ny in (y - nms_radius)..=(y + nms_radius) {
                let row = ny * w_img;
                for nx in (x - nms_radius)..=(x + nms_radius) {
                    if kmap[row + nx] > kv {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if !is_max {
                continue;
            }

            let fx = (((x as f32) + 0.5) * (wc as f32) / (w_img as f32) - 0.5)
                .clamp(0.0, (wc - 1) as f32);
            let x0 = (fx.floor() as usize).min(wc - 1);
            let x1 = (x0 + 1).min(wc - 1);
            let wx = fx - (x0 as f32);

            let hm_val = (1.0 - wy) * (1.0 - wx) * heatmap[y0 * wc + x0]
                + (1.0 - wy) * wx * heatmap[y0 * wc + x1]
                + wy * (1.0 - wx) * heatmap[y1 * wc + x0]
                + wy * wx * heatmap[y1 * wc + x1];
            let score = kv * hm_val.clamp(0.0, 1.0);
            candidates.push((x, y, score));
        }
    }

    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
    candidates.truncate(max_keypoints);

    // Bilinearly sample and L2-normalize 64-D descriptors at stride-8 cell coordinates
    let mut out = Vec::with_capacity(candidates.len());
    for (x, y, score) in candidates {
        let fx = (((x as f32) + 0.5) / 8.0 - 0.5).clamp(0.0, ((wc - 1) as f32) - 1e-4);
        let fy = (((y as f32) + 0.5) / 8.0 - 0.5).clamp(0.0, ((hc - 1) as f32) - 1e-4);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(wc - 1);
        let y1 = (y0 + 1).min(hc - 1);
        let wx = fx - (x0 as f32);
        let wy = fy - (y0 as f32);
        let w00 = (1.0 - wy) * (1.0 - wx);
        let w01 = (1.0 - wy) * wx;
        let w10 = wy * (1.0 - wx);
        let w11 = wy * wx;
        let i00 = y0 * wc + x0;
        let i01 = y0 * wc + x1;
        let i10 = y1 * wc + x0;
        let i11 = y1 * wc + x1;

        let mut desc = [0.0_f32; XFEAT_DESC_DIM];
        let mut norm_sq = 0.0_f32;
        for (c, slot) in desc.iter_mut().enumerate() {
            let base = c * cell_area;
            let v = w00 * feats[base + i00]
                + w01 * feats[base + i01]
                + w10 * feats[base + i10]
                + w11 * feats[base + i11];
            *slot = v;
            norm_sq += v * v;
        }
        let inv_norm = 1.0_f32 / norm_sq.sqrt().max(1e-12);
        for v in &mut desc {
            *v *= inv_norm;
        }
        out.push(XFeatKeypoint {
            x: x as f32,
            y: y as f32,
            score,
            desc,
        });
    }
    out
}

// ---- Exact Apache-2.0 LighterGlue Stage-2B Static K=256 Packer & Mutual-Match Decoder ----

/// Pack up to `LIGHTERGLUE_STATIC_K` (`256`) XFeat keypoints into preallocated normalized `[-1, 1]`
/// `[1, 256, 2]` coordinates and `[1, 256, 64]` L2-normalized descriptors (zero-padded, zero alloc).
fn pack_lighterglue_k256_into(
    kpts: &[XFeatKeypoint],
    k_buf: &mut Vec<f32>,
    d_buf: &mut Vec<f32>,
) -> usize {
    k_buf.clear();
    k_buf.resize(LIGHTERGLUE_STATIC_K * 2, 0.0_f32);
    d_buf.clear();
    d_buf.resize(LIGHTERGLUE_STATIC_K * XFEAT_DESC_DIM, 0.0_f32);
    let n = kpts.len().min(LIGHTERGLUE_STATIC_K);
    let inv_w = 2.0_f32 / (XFEAT_IN_W as f32);
    let inv_h = 2.0_f32 / (XFEAT_IN_H as f32);
    for (i, kp) in kpts.iter().take(n).enumerate() {
        k_buf[i * 2] = (kp.x * inv_w - 1.0).clamp(-1.0, 1.0);
        k_buf[i * 2 + 1] = (kp.y * inv_h - 1.0).clamp(-1.0, 1.0);
        d_buf[i * XFEAT_DESC_DIM..(i + 1) * XFEAT_DESC_DIM].copy_from_slice(&kp.desc);
    }
    n
}

/// Pack up to `LIGHTERGLUE_STATIC_K` (`256`) XFeat keypoints into normalized `[-1, 1]`
/// `[1, 256, 2]` coordinates and `[1, 256, 64]` L2-normalized descriptors (zero-padded).
pub fn pack_lighterglue_k256(kpts: &[XFeatKeypoint]) -> (Vec<f32>, Vec<f32>, usize) {
    let mut k_buf = Vec::with_capacity(LIGHTERGLUE_STATIC_K * 2);
    let mut d_buf = Vec::with_capacity(LIGHTERGLUE_STATIC_K * XFEAT_DESC_DIM);
    let n = pack_lighterglue_k256_into(kpts, &mut k_buf, &mut d_buf);
    (k_buf, d_buf, n)
}

/// Decode mutual nearest-neighbor matches from LighterGlue log-assignment `[1, 256, 256]` scores.
pub fn decode_lighterglue_mutual_matches(
    scores: &[f32],
    n0: usize,
    n1: usize,
    min_prob: f32,
) -> (usize, f64) {
    let k = LIGHTERGLUE_STATIC_K;
    if scores.len() != k * k || n0 == 0 || n1 == 0 {
        return (0, 0.0);
    }
    let lim0 = n0.min(k);
    let lim1 = n1.min(k);

    let mut row_best_j = vec![0usize; lim0];
    let mut row_best_s = vec![f32::NEG_INFINITY; lim0];
    let mut col_best_i = vec![0usize; lim1];
    let mut col_best_s = vec![f32::NEG_INFINITY; lim1];

    for i in 0..lim0 {
        let row_off = i * k;
        for j in 0..lim1 {
            let s = scores[row_off + j];
            if s > row_best_s[i] {
                row_best_s[i] = s;
                row_best_j[i] = j;
            }
            if s > col_best_s[j] {
                col_best_s[j] = s;
                col_best_i[j] = i;
            }
        }
    }

    let mut match_count = 0usize;
    let mut prob_sum = 0.0_f64;
    for i in 0..lim0 {
        let j = row_best_j[i];
        if col_best_i[j] == i {
            let p = row_best_s[i].clamp(-80.0, 0.0).exp();
            if p >= min_prob {
                match_count += 1;
                prob_sum += p as f64;
            }
        }
    }
    let mean_prob = if match_count > 0 {
        round4(prob_sum / (match_count as f64))
    } else {
        0.0
    };
    (match_count, mean_prob)
}

// ---- Exact DR-04 §4.2 2-Channel LWIR Input Builder (`src/xvs/sensor/lwir.py` parity) ----
// Uses O(1)-per-pixel sliding-window separable 7x7 box filter + preallocated scratch buffers.

struct LwirScratch {
    rad: Vec<f32>,
    horiz: Vec<f32>,
    col_win: Vec<f32>,
    hp: Vec<f32>,
    out: Vec<f32>,
    lwir_u8: Vec<u8>,
}

impl LwirScratch {
    fn new() -> Self {
        Self {
            rad: Vec::new(),
            horiz: Vec::new(),
            col_win: Vec::new(),
            hp: Vec::new(),
            out: Vec::new(),
            lwir_u8: Vec::new(),
        }
    }
}

fn build_lwir_2ch_from_rgb_into(img: &RgbImage, w: usize, h: usize, scratch: &mut LwirScratch) {
    let resized = image::imageops::resize(img, w as u32, h as u32, FilterType::Triangle);
    let n = w * h;
    scratch.rad.resize(n, 0.0_f32);
    scratch.lwir_u8.resize(n, 0_u8);
    scratch.horiz.resize(n, 0.0_f32);
    scratch.col_win.resize(w, 0.0_f32);
    scratch.hp.resize(n, 0.0_f32);
    scratch.out.resize(2 * n, 0.0_f32);

    let inv255 = 1.0_f32 / 255.0_f32;
    let denom_log = (1.0_f32 + LWIR_L_SAT_PROXY.max(1.0)).ln();
    let inv_denom_log = 1.0_f32 / denom_log;
    for (idx, rgb) in resized.as_raw().as_chunks::<3>().0.iter().enumerate().take(n) {
        let r = (rgb[0] as f32) * inv255;
        let g = (rgb[1] as f32) * inv255;
        let b = (rgb[2] as f32) * inv255;
        let gray = (0.55 * r + 0.30 * g + 0.15 * (1.0 - b)).clamp(0.0, 1.0);
        let rv = gray * LWIR_L_SAT_PROXY;
        scratch.rad[idx] = rv;
        scratch.lwir_u8[idx] = (gray * 255.0).clamp(0.0, 255.0) as u8;
        scratch.out[idx] = (1.0_f32 + rv.max(0.0)).ln() * inv_denom_log;
    }

    // O(1)/pixel sliding-window separable 7x7 average pool (padding=3, zero-padded to match PyTorch F.avg_pool2d(x, 7, 1, 3))
    let rad = &scratch.rad;
    let horiz = &mut scratch.horiz;
    for y in 0..h {
        let row = y * w;
        let init_end = 4usize.min(w);
        let mut win = 0.0_f32;
        for &v in &rad[row..row + init_end] {
            win += v;
        }
        for x in 0..w {
            horiz[row + x] = win;
            if x + 4 < w {
                win += rad[row + x + 4];
            }
            if x >= 3 {
                win -= rad[row + x - 3];
            }
        }
    }

    let col_win = &mut scratch.col_win;
    col_win.fill(0.0_f32);
    let init_rows = 4usize.min(h);
    for y in 0..init_rows {
        let row = &horiz[y * w..(y + 1) * w];
        for (cw, &hv) in col_win.iter_mut().zip(row.iter()) {
            *cw += hv;
        }
    }
    let inv49 = 1.0_f32 / 49.0_f32;
    let hp = &mut scratch.hp;
    let mut sum_hp = 0.0_f64;
    let mut sq_hp = 0.0_f64;
    for y in 0..h {
        let row_off = y * w;
        for x in 0..w {
            let v = rad[row_off + x] - col_win[x] * inv49;
            hp[row_off + x] = v;
            let vf = v as f64;
            sum_hp += vf;
            sq_hp += vf * vf;
        }
        if y + 4 < h {
            let add_row = &horiz[(y + 4) * w..(y + 5) * w];
            for (cw, &hv) in col_win.iter_mut().zip(add_row.iter()) {
                *cw += hv;
            }
        }
        if y >= 3 {
            let sub_row = &horiz[(y - 3) * w..(y - 2) * w];
            for (cw, &hv) in col_win.iter_mut().zip(sub_row.iter()) {
                *cw -= hv;
            }
        }
    }

    let nf = n.max(1) as f64;
    let mean_hp = sum_hp / nf;
    let var_hp = ((sq_hp / nf) - mean_hp * mean_hp).max(0.0);
    let std_hp = (var_hp.sqrt() as f32).max(1e-3);
    let inv_std = 1.0_f32 / std_hp;

    for (dst, &v) in scratch.out[n..2 * n].iter_mut().zip(hp.iter()) {
        *dst = (v * inv_std).clamp(-5.0, 5.0);
    }
}

// ---- Public Serializable IPC Contracts ----

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatusEntry {
    pub name: String,
    pub loaded: bool,
    pub real_weights: bool,
    pub device: String,
    pub backend: String,
    pub status: String,
    pub sha256: Option<String>,
    pub expected_sha256: Option<String>,
    pub filename: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeHealthPayload {
    pub version: &'static str,
    pub status: &'static str,
    pub transport: &'static str,
    pub kani_proofs: &'static str,
    pub odd_yaml_sha256: &'static str,
    pub host_gpu_name: String,
    pub compute_device: String,
    pub ort_loaded: bool,
    pub ort_dylib_label: String,
    pub vram_mb: f64,
    pub media_available: bool,
    pub multiship_coord_enabled: bool,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub watermark: &'static str,
    pub models: BTreeMap<String, ModelStatusEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeSensorHealthPayload {
    pub eo_blur_score: f64,
    pub eo_glare_ratio: f64,
    pub lwir_nuc_frozen: bool,
    pub lwir_freeze_ticks: u32,
    pub timestamp_skew_ms: f64,
    pub q_eo_effective: f64,
    pub q_lwir_effective: f64,
    pub thermal_mode: String,
    pub multiship_coord_enabled: bool,
    pub f1_far91113_active: bool,
    pub f4_ghost_hysteresis_active: bool,
    pub eo_occlusion_ratio: f64,
    pub eo_focus_quality: f64,
    pub eo_dead_line_fraction: f64,
    pub eo_spatial_autocorr: f64,
    pub eo_dead_pixel_fpn_fault: bool,
    pub eo_sensor_noise_fault: bool,
    pub eo_degraded: bool,
    pub eo_flags: Vec<&'static str>,
    pub grid_4x4_q_eo: [f64; 16],
    pub sector_q_eo: [f64; 12],
    pub min_sector_q_eo: f64,
    pub degraded_sector_mask: u16,
    pub lwir_ffc_imminent: bool,
    pub lwir_ffc_in_progress: bool,
    pub lwir_dynamic_range_p95_p05: f64,
    pub lwir_effective_dynamic_range: f64,
    pub lwir_contrast_collapsed: bool,
    pub lwir_localized_hot_target_rescue: bool,
    pub lwir_dead_line_fraction: f64,
    pub lwir_spatial_autocorr: f64,
    pub lwir_dead_pixel_fpn_fault: bool,
    pub lwir_sensor_noise_fault: bool,
    pub lwir_q_optical: f64,
    pub lwir_degraded: bool,
    pub lwir_flags: Vec<&'static str>,
    pub timestamp_monotonic_ok: bool,
    pub timestamp_skew_fault: bool,
    pub fsync_precision_ok: bool,
    pub bearing_track_confirmed: bool,
    pub bearing_track_hits_in_window: u8,
    pub ring0_authority_inhibited: bool,
    pub last_swerve_role: String,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub watermark: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeEvalReportPayload {
    pub filename: String,
    pub json_content: String,
    pub sha256: String,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub watermark: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveRangeReport {
    pub mode: String,
    pub mode_id: u8,
    pub r_hat_m: f32,
    pub sigma_r_m: f32,
    pub r_safe_lower_m: f32,
    pub rho_dot_inv_s: f32,
    pub cmd_bank_deg: f64,
    pub min_sc3bf_margin: f32,
    pub active_bank_idx: u8,
    pub swerve_coord_role: String,
    pub f1_far91113_active: bool,
    pub ring0_inhibited: bool,
    pub wire_bytes: usize,
    pub wire_crc_ok: bool,
    pub wire_hex: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RtaReport {
    pub mode: String,
    pub mode_id: u8,
    pub raw_target: String,
    pub active_healthy_cams: u8,
    pub mandatory_go_around: bool,
    pub inhibit_avoidance_turns: bool,
    pub tick_index: u64,
    pub odd_yaml_sha256: &'static str,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub watermark: &'static str,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct NativeRtaInput {
    pub cam_health_mask: Option<u8>,
    pub ood_exceeded_mask: Option<u8>,
    pub horizon_pl_rad: Option<f64>,
    pub v_vpl_m: Option<f64>,
    pub v_hpl_m: Option<f64>,
    pub spoof_alarm: Option<bool>,
    pub ml_detector_fault: Option<bool>,
    pub approach_active: Option<bool>,
    #[serde(default)]
    pub go_around_ack_reset: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeChannelPayload {
    pub channel: String,
    pub model: String,
    pub backend: String,
    pub real_weights: bool,
    pub checkpoint_sha256: Option<String>,
    pub latency_ms: f64,
    pub confidence: f64,
    pub summary: String,
    pub annotated_jpeg_b64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_b64: Option<String>,
    pub metrics: serde_json::Value,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub watermark: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeDemoTickPayload {
    pub scenario: String,
    pub frame: u32,
    pub transport: &'static str,
    pub total_latency_ms: f64,
    pub rta: RtaReport,
    pub sensor_health: NativeSensorHealthPayload,
    pub channels: BTreeMap<String, NativeChannelPayload>,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub watermark: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeHilStatusPayload {
    pub mode: &'static str,
    pub bound_host: &'static str,
    pub rx_port: u16,
    pub tx_port: u16,
    pub rx_packets: u64,
    pub tx_packets: u64,
    pub tx_enabled: bool,
    pub actuator_authority: bool,
    pub non_flight_sim_only: bool,
    pub eu_pld_notice: &'static str,
    pub watermark: &'static str,
    pub gpl_free: bool,
    pub last_telemetry_ned: Option<serde_json::Value>,
    pub last_loopback: Option<NativeLoopbackReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeLoopbackCheck {
    pub name: String,
    pub pass: bool,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeLoopbackReport {
    pub pass: bool,
    pub timestamp_utc: String,
    pub elapsed_ms: f64,
    pub packets_verified: u32,
    pub tamper_rejected: bool,
    pub gpl_free: bool,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub watermark: &'static str,
    pub checks: Vec<NativeLoopbackCheck>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NavIntegritySnapshot {
    pub active_bank: &'static str,
    pub gnss_quarantined: bool,
    pub spoof_alarm: bool,
    pub vertical_spoof_alarm: bool,
    pub max_cusum_stat: f64,
    pub vertical_cusum_stat: f64,
    pub cusum_h_threshold: f64,
    pub dominant_arm: usize,
    pub nominal_streak: usize,
    pub seq_vpr_window_len: usize,
    pub seq_vpr_admitted: bool,
    pub seq_vpr_top1_idx: i64,
    pub seq_vpr_posterior_prob: f64,
    pub seq_vpr_psr_cal: f64,
    pub seq_vpr_reason_codes: Vec<String>,
}

impl Default for NavIntegritySnapshot {
    fn default() -> Self {
        Self {
            active_bank: "bank_a_gnss",
            gnss_quarantined: false,
            spoof_alarm: false,
            vertical_spoof_alarm: false,
            max_cusum_stat: 0.0,
            vertical_cusum_stat: 0.0,
            cusum_h_threshold: CUSUM_H_LOCKED,
            dominant_arm: 0,
            nominal_streak: 0,
            seq_vpr_window_len: 0,
            seq_vpr_admitted: false,
            seq_vpr_top1_idx: -1,
            seq_vpr_posterior_prob: 0.0,
            seq_vpr_psr_cal: 0.0,
            seq_vpr_reason_codes: Vec::new(),
        }
    }
}

#[inline]
fn sanitize_json_f64(v: f64, non_finite_fallback: f64) -> f64 {
    if v.is_finite() {
        round4(v)
    } else {
        non_finite_fallback
    }
}

pub fn default_sensor_health_payload() -> NativeSensorHealthPayload {
    NativeSensorHealthPayload {
        eo_blur_score: 100.0,
        eo_glare_ratio: 0.0,
        lwir_nuc_frozen: false,
        lwir_freeze_ticks: 0,
        timestamp_skew_ms: 0.0,
        q_eo_effective: 1.0,
        q_lwir_effective: 1.0,
        thermal_mode: ThermalSimplexMode::NominalFusion.as_str().to_string(),
        multiship_coord_enabled: true,
        f1_far91113_active: true,
        f4_ghost_hysteresis_active: false,
        eo_occlusion_ratio: 0.0,
        eo_focus_quality: 1.0,
        eo_dead_line_fraction: 0.0,
        eo_spatial_autocorr: 1.0,
        eo_dead_pixel_fpn_fault: false,
        eo_sensor_noise_fault: false,
        eo_degraded: false,
        eo_flags: Vec::new(),
        grid_4x4_q_eo: [1.0; 16],
        sector_q_eo: [1.0; 12],
        min_sector_q_eo: 1.0,
        degraded_sector_mask: 0,
        lwir_ffc_imminent: false,
        lwir_ffc_in_progress: false,
        lwir_dynamic_range_p95_p05: 255.0,
        lwir_effective_dynamic_range: 255.0,
        lwir_contrast_collapsed: false,
        lwir_localized_hot_target_rescue: false,
        lwir_dead_line_fraction: 0.0,
        lwir_spatial_autocorr: 1.0,
        lwir_dead_pixel_fpn_fault: false,
        lwir_sensor_noise_fault: false,
        lwir_q_optical: 1.0,
        lwir_degraded: false,
        lwir_flags: Vec::new(),
        timestamp_monotonic_ok: true,
        timestamp_skew_fault: false,
        fsync_precision_ok: true,
        bearing_track_confirmed: false,
        bearing_track_hits_in_window: 0,
        ring0_authority_inhibited: false,
        last_swerve_role: "OPTICAL_FAR91113_RIGHT".to_string(),
        non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
        actuator_authority: ACTUATOR_AUTHORITY,
        watermark: NON_FLIGHT_WATERMARK,
    }
}

pub fn default_hil_status_payload() -> NativeHilStatusPayload {
    NativeHilStatusPayload {
        mode: "locked_non_flight_sim_only",
        bound_host: "none",
        rx_port: 0,
        tx_port: 0,
        rx_packets: 0,
        tx_packets: 0,
        tx_enabled: false,
        actuator_authority: ACTUATOR_AUTHORITY,
        non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
        eu_pld_notice: EU_PLD_ART4_7_LOCKOUT_MSG,
        watermark: NON_FLIGHT_WATERMARK,
        gpl_free: true,
        last_telemetry_ned: None,
        last_loopback: None,
    }
}

pub fn evaluate_rta_step(rta: &mut SimplexRta, req: &NativeRtaInput) -> RtaReport {
    let ood_mask = req.ood_exceeded_mask.unwrap_or(0x00);
    let mut ood_cam_scores = [0.1_f64; 6];
    for (idx, score) in ood_cam_scores.iter_mut().enumerate() {
        if ((ood_mask >> idx) & 1) != 0 {
            *score = 0.95;
        }
    }
    let spoof_alarm = req.spoof_alarm.unwrap_or(false);
    let float_tel = RtaFloatTelemetry {
        horizon_pl_rad: req.horizon_pl_rad.unwrap_or(0.002),
        horizon_al_rad: 0.017_453_3,
        v_vpl_m: req.v_vpl_m.unwrap_or(5.0),
        v_val_m: 10.0,
        v_hpl_m: req.v_hpl_m.unwrap_or(14.0),
        v_hal_m: 30.0,
        epipolar_max_rad: [0.0002; 6],
        signed_horizon_residual_rad: 0.0,
        runway_width_delta_m: 0.0,
        runway_width_delta_limit_m: 5.0,
        cusum_stat: if spoof_alarm { 12.0 } else { 0.5 },
        cusum_thr: 5.0,
        ood_global_score: 0.1,
        ood_global_thr: 0.7,
        ood_cam_scores,
        ood_cam_thr: 0.7,
        cam_health_mask: req.cam_health_mask.unwrap_or(0x3F),
        ml_detector_fault: req.ml_detector_fault.unwrap_or(false),
        transient_dropout: false,
        approach_active: req.approach_active.unwrap_or(false),
        go_around_ack_reset: req.go_around_ack_reset,
    };
    let inp = build_rta_input_from_f64(&float_tel);
    let out = rta.step(&inp);
    RtaReport {
        mode: format!("{:?}", out.mode),
        mode_id: out.mode as u8,
        raw_target: format!("{:?}", out.raw_target),
        active_healthy_cams: out.active_healthy_cams,
        mandatory_go_around: out.mandatory_go_around,
        inhibit_avoidance_turns: out.inhibit_avoidance_turns,
        tick_index: out.tick_index,
        odd_yaml_sha256: ODD_YAML_SHA256,
        non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
        actuator_authority: ACTUATOR_AUTHORITY,
        watermark: NON_FLIGHT_WATERMARK,
    }
}

pub fn build_demo_rta_input(scenario: &str) -> NativeRtaInput {
    NativeRtaInput {
        cam_health_mask: Some(0x3F),
        ood_exceeded_mask: Some(0x00),
        horizon_pl_rad: Some(0.002),
        v_vpl_m: Some(5.5),
        v_hpl_m: Some(15.0),
        spoof_alarm: Some(scenario == "gps_spoof"),
        ml_detector_fault: Some(false),
        approach_active: Some(scenario == "forcedland" || scenario == "thermal_fog"),
        go_around_ack_reset: false,
    }
}

fn build_nav_corridor_grid_and_gallery() -> (Vec<[f64; 2]>, Vec<[f32; 512]>) {
    let mut coords = Vec::with_capacity(16);
    for r in 0..4usize {
        for c in 0..4usize {
            coords.push([-(r as f64) * 150.0, (c as f64) * 150.0]);
        }
    }
    let mut gallery = Vec::with_capacity(16);
    for tile in 0..16usize {
        let mut vec = [0.0_f32; 512];
        let mut norm_sq = 0.0_f64;
        for (d, slot) in vec.iter_mut().enumerate() {
            let phase = ((tile as f64) * 1.73205 + (d as f64) * 0.6180339).sin()
                + 0.35 * (((tile * 7 + d * 3) as f64) * 0.271828).cos();
            *slot = phase as f32;
            norm_sq += phase * phase;
        }
        let inv = (1.0 / norm_sq.sqrt().max(1e-12)) as f32;
        for v in &mut vec {
            *v *= inv;
        }
        gallery.push(vec);
    }
    (coords, gallery)
}

const NAV_CORRIDOR_SERPENTINE: [usize; 16] = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12];

// ---- Core Stateful Pure-Rust Inference Engine ----

fn swerve_role_label(role: SwerveCoordRole) -> &'static str {
    match role {
        SwerveCoordRole::InhibitedByP0 => "INHIBITED_BY_P0",
        SwerveCoordRole::YieldHoldForPeerProbe => "YIELD_HOLD_FOR_PEER_PROBE",
        SwerveCoordRole::CooperativeRightProbe => "COOPERATIVE_RIGHT_PROBE",
        SwerveCoordRole::OpticalFar91113Right => "OPTICAL_FAR91113_RIGHT",
    }
}

fn make_ecef_bearing_obs(
    sender: u8,
    tid: u16,
    origin: [f64; 3],
    target: [f64; 3],
    ts_us: u64,
) -> OpenXvsBearingTrack {
    let dir = [
        target[0] - origin[0],
        target[1] - origin[1],
        target[2] - origin[2],
    ];
    let raw = OpenXvsBearingTrack::from_ecef_ray(
        ts_us, origin, dir, [0.0; 3], 1500, 100, tid, sender, 0,
    );
    OpenXvsBearingTrack::from_le_bytes(&raw.to_le_bytes())
}

/// Format current wall-clock UTC time as ISO-8601 `YYYY-MM-DDTHH:MM:SSZ` (zero external crates).
pub fn format_utc_now_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3_600;
    let min = (rem % 3_600) / 60;
    let sec = rem % 60;

    // Howard Hinnant's exact civil-from-days algorithm (Gregorian calendar, proleptic UTC)
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

struct D2TemporalState {
    in_h: usize,
    in_w: usize,
    hid_c: usize,
    hid_h: usize,
    hid_w: usize,
    h_prev: Vec<f32>,
    norm_gray_hist: VecDeque<Vec<f32>>,
    coord_active_range: CoordinatedActiveRange,
    mesh_daa_guard: StatefulMeshDaaGuard,
    bearing_tbc: BearingTrackBeforeConfirm,
    mesh_tick_utc_us: u64,
    mesh_cycle: u64,
    f1_far91113_active: bool,
    f4_ghost_hysteresis_active: bool,
    ring0_inhibited: bool,
    mesh_sticky_suppressed_total: u32,
    mesh_stale_rejected_total: u32,
    mesh_replay_rejected_total: u32,
    track_age: u32,
    last_size_px: Option<f64>,
}

impl D2TemporalState {
    fn new(in_h: usize, in_w: usize, hid_c: usize, hid_h: usize, hid_w: usize) -> Self {
        Self {
            in_h,
            in_w,
            hid_c,
            hid_h,
            hid_w,
            h_prev: vec![0.0_f32; hid_c * hid_h * hid_w],
            norm_gray_hist: VecDeque::with_capacity(8),
            coord_active_range: CoordinatedActiveRange::new(
                1,
                ActiveRangeConfig {
                    q_hat_conformal: 0.955,
                    sigma_bearing_rad: 2.86 / AOT_FOCAL_PX,
                    ..ActiveRangeConfig::default()
                },
            ),
            mesh_daa_guard: StatefulMeshDaaGuard::new(),
            bearing_tbc: BearingTrackBeforeConfirm::new_3_of_5(),
            mesh_tick_utc_us: 1_700_000_000_000,
            mesh_cycle: 0,
            f1_far91113_active: true,
            f4_ghost_hysteresis_active: false,
            ring0_inhibited: false,
            mesh_sticky_suppressed_total: 0,
            mesh_stale_rejected_total: 0,
            mesh_replay_rejected_total: 0,
            track_age: 0,
            last_size_px: None,
        }
    }

    /// Run one deterministic `StatefulMeshDaaGuard` (DR-22 F4) step alternating 3-ship anchor
    /// confirmation with a transient 200 ms peer-#3 dropout (2-ship sticky ghost suppression).
    fn step_multiship_mesh_guard(&mut self) {
        let base_ecef = [6_378_137.0_f64, 0.0, 0.0];
        let p_s1 = [base_ecef[0], 0.0, 0.0];
        let p_s2 = [base_ecef[0], 1000.0, 0.0];
        let p_s3 = [base_ecef[0] + 1000.0, 500.0, 0.0];
        let t1 = [base_ecef[0] + 500.0, 200.0, 0.0];
        let t2 = [base_ecef[0] + 500.0, 800.0, 0.0];

        let ts_obs = self.mesh_tick_utc_us;
        let now_us = ts_obs + 20_000;
        let res = if self.mesh_cycle.is_multiple_of(2) {
            let obs_3 = [
                make_ecef_bearing_obs(1, 1, p_s1, t1, ts_obs),
                make_ecef_bearing_obs(1, 2, p_s1, t2, ts_obs),
                make_ecef_bearing_obs(2, 1, p_s2, t1, ts_obs),
                make_ecef_bearing_obs(2, 2, p_s2, t2, ts_obs),
                make_ecef_bearing_obs(3, 1, p_s3, t1, ts_obs),
                make_ecef_bearing_obs(3, 2, p_s3, t2, ts_obs),
            ];
            self.mesh_daa_guard.fuse_guarded(&obs_3, now_us)
        } else {
            let obs_2 = [
                make_ecef_bearing_obs(1, 1, p_s1, t1, ts_obs),
                make_ecef_bearing_obs(1, 2, p_s1, t2, ts_obs),
                make_ecef_bearing_obs(2, 1, p_s2, t1, ts_obs),
                make_ecef_bearing_obs(2, 2, p_s2, t2, ts_obs),
            ];
            self.mesh_daa_guard.fuse_guarded(&obs_2, now_us)
        };

        self.mesh_sticky_suppressed_total = self
            .mesh_sticky_suppressed_total
            .saturating_add(res.sticky_ghosts_suppressed);
        self.mesh_stale_rejected_total = self
            .mesh_stale_rejected_total
            .saturating_add(res.stale_obs_rejected);
        self.mesh_replay_rejected_total = self
            .mesh_replay_rejected_total
            .saturating_add(res.replay_obs_rejected);
        self.f4_ghost_hysteresis_active =
            res.sticky_triple_ray_active || (res.fresh_sender_count >= 3 && res.num_fused > 0);
        self.mesh_tick_utc_us = self.mesh_tick_utc_us.saturating_add(200_000);
        self.mesh_cycle = self.mesh_cycle.saturating_add(1);
    }
}

pub struct RustStudioEngine {
    _ort_env: Option<Environment>,
    ort_loaded: bool,
    ort_dylib_label: String,
    d2_session: Option<Session>,
    thermal_session: Option<Session>,
    land_session: Option<Session>,
    nav_session: Option<Session>,
    xfeat_session: Option<Session>,
    lighterglue_session: Option<Session>,
    land_calibrator: StudentTCalibrator,
    d2_state: D2TemporalState,
    rta: SimplexRta,
    last_rta_mode_id: u8,
    last_rta_inhibit_turns: bool,
    sensor_health: SensorHealthMonitor,
    nav_cusum: DualBankEsEkfCusum,
    seq_vpr: SequenceVprFilter,
    nav_corridor_coords: Vec<[f64; 2]>,
    nav_corridor_gallery: Vec<[f32; 512]>,
    nav_corridor_step: usize,
    last_nav_integrity: NavIntegritySnapshot,
    sim_clock_ms: f64,
    media_lib: StudioMediaLibrary,
    last_channel_latencies_ms: BTreeMap<String, f64>,
    models_meta: BTreeMap<String, ModelStatusEntry>,
    last_loopback: Option<NativeLoopbackReport>,
    // Preallocated scratch buffers (zero per-tick multi-MB tensor heap churn)
    scratch_d2_gray: Vec<f32>,
    scratch_d2_x: Vec<f32>,
    scratch_land_inp: Vec<f32>,
    scratch_thermal_eo: Vec<f32>,
    lwir_scratch: LwirScratch,
    scratch_nav_inp: Vec<f32>,
    scratch_xfeat_inp: Vec<f32>,
    scratch_lg_k0: Vec<f32>,
    scratch_lg_d0: Vec<f32>,
    scratch_lg_k1: Vec<f32>,
    scratch_lg_d1: Vec<f32>,
    // Bounded static satellite reference XFeat keypoint cache (avoids 2x XFeat per tick)
    sat_ref_xfeat_cache: HashMap<String, Vec<XFeatKeypoint>>,
    sat_ref_xfeat_order: VecDeque<String>,
    // Slow rate-group (1-2 Hz) cached demo channel outputs
    last_slow_scenario: Option<String>,
    last_slow_land: Option<NativeChannelPayload>,
    last_slow_nav: Option<NativeChannelPayload>,
}

impl RustStudioEngine {
    pub fn initialize() -> Self {
        let mut models_meta: BTreeMap<String, ModelStatusEntry> = BTreeMap::new();
        let (ort_env, ort_loaded, ort_dylib_label) = match acquire_global_ort_env() {
            Ok((env, label)) => (Some(env), true, label),
            Err(err_label) => (None, false, err_label),
        };

        // Smart ORT threading: pin lightweight models (`d2`, `thermal`, `land`, `xfeat`,
        // `lighterglue`) to 1 intra-op thread (zero ORT worker-pool wake jitter; verified in
        // 10s soak benchmarks) and allocate up to 2 intra-op threads ONLY to heavy `xvs_nav_v5`
        // when `OMP_NUM_THREADS >= 2`.
        // 1. XVS-D2 (0.53 MB FP16 — SHA-256 fail-closed pinned)
        let (d2_session, d2_meta) = load_pinned_onnx_session(
            ort_env.as_ref(),
            "XVS-D2 (ConvGRU ONNX)",
            "xvs_d2_web_fp16.onnx",
            "LOADED",
            "MISSING_ONNX_OR_ORT",
            &[
                "apps/livedeck/public/models/xvs_d2_web_fp16.onnx",
                "models/xvs_d2_web_fp16.onnx",
                "xvs_d2_web_fp16.onnx",
            ],
            D2_ONNX_PINS,
            1,
        );
        models_meta.insert("xvs_d2".to_string(), d2_meta);

        // 2. XVS-ThermalNet v5.2 (1.756 MiB FP16 promoted -> v4.5 fallback — SHA-256 fail-closed pinned)
        let (thermal_session, thermal_meta) = load_pinned_onnx_session(
            ort_env.as_ref(),
            "XVS-ThermalNet v5.2 (EO+LWIR FPN ONNX)",
            "xvs_thermal_v52_fp16.onnx",
            "LOADED",
            "MISSING_ONNX_OR_ORT",
            &[
                "models/xvs_thermal_v52_fp16.onnx",
                "outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp16.onnx",
                "xvs_thermal_v52_fp16.onnx",
                "models/xvs_thermal_v52_fp32.onnx",
                "outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp32.onnx",
                "outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp32.onnx",
                "outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp16.onnx",
                "models/xvs_thermal_v45_fp32.onnx",
                "models/xvs_thermal_v45_fp16.onnx",
                "xvs_thermal_v45_fp32.onnx",
                "xvs_thermal_v45_fp16.onnx",
            ],
            THERMAL_ONNX_PINS,
            1,
        );
        models_meta.insert("xvs_thermal".to_string(), thermal_meta);

        // 3. XVS-Land v5.2 (2.766 MiB FP16 6-output promoted -> v4.5 fallback — SHA-256 fail-closed pinned)
        let (land_session, land_meta) = load_pinned_onnx_session(
            ort_env.as_ref(),
            "XVS-Land v5.2 (12-KPT SimCC 384 + Keyline Sigma ONNX)",
            "xvs_land_v52_fp16.onnx",
            "LOADED",
            "MISSING_ONNX_OR_ORT",
            &[
                "models/xvs_land_v52_fp16.onnx",
                "outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp16.onnx",
                "xvs_land_v52_fp16.onnx",
                "models/xvs_land_v52_fp32.onnx",
                "outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp32.onnx",
                "outputs/colab_l4_v45/harvest_land/xvs_land_v45_fp16.onnx",
                "outputs/colab_l4_v45/harvest_land/xvs_land_v45.onnx",
                "models/xvs_land_v45_fp16.onnx",
                "models/xvs_land_v45.onnx",
                "xvs_land_v45_fp16.onnx",
                "xvs_land_v45.onnx",
            ],
            LAND_ONNX_PINS,
            1,
        );
        let is_land_v52 = matches!(
            land_meta.sha256.as_deref(),
            Some(EXPECTED_LAND_V52_FP16_SHA256) | Some(EXPECTED_LAND_V52_FP32_SHA256)
        );
        let (land_nu, land_scale) = if is_land_v52 {
            (V52_LAND_STUDENT_T_NU, V52_LAND_STUDENT_T_SCALE)
        } else {
            (WAVE_C_LAND_STUDENT_T_NU, WAVE_C_LAND_STUDENT_T_SCALE)
        };
        let land_calibrator = StudentTCalibrator::new(land_nu, land_scale)
            .expect("canonical Student-t calibrator params are valid");
        models_meta.insert("xvs_land".to_string(), land_meta);

        // 4. XVS-Nav v5.0 Stage-1 (Faz-B 21.525 MB FP16 prioritized -> Faz-A fallback; SHA-256 fail-closed)
        let (nav_session, nav_meta) = load_pinned_onnx_session(
            ort_env.as_ref(),
            "XVS-Nav v5.0 Faz-B (MobileNetV4-Hybrid-M + MixVPR FP16 ONNX)",
            "xvs_nav_v5_fazb_fp16.onnx",
            "READY_RUST_ORT_V5",
            "AWAITING_V5_ONNX_HARVEST",
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/xvs_nav_v5_fazb_fp16.onnx",
                "models/xvs_nav_v5_fazb_fp16.onnx",
                "xvs_nav_v5_fazb_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest/xvs_nav_v5_fp16.onnx",
                "models/xvs_nav_v5_fp16.onnx",
                "outputs/xvs_nav_v5/xvs_nav_v5_fp16.onnx",
                "outputs/colab_l4_v50/xvs_nav_v5_fp16.onnx",
                "xvs_nav_v5_fp16.onnx",
            ],
            NAV_ONNX_PINS,
            2,
        );
        models_meta.insert("xvs_nav".to_string(), nav_meta);

        // 5. Stage-2A XFeat (1.303 MB FP16 prioritized -> 2.530 MB FP32 fallback; SHA-256 fail-closed)
        let (xfeat_session, xfeat_meta) = load_pinned_onnx_session(
            ort_env.as_ref(),
            "XVS-XFeat Stage-2A (320x256 Keypoints + 64D FP16 ONNX)",
            "xfeat_320x256_fp16.onnx",
            "READY_RUST_ORT_V5",
            "MISSING_XFEAT_ONNX",
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp16.onnx",
                "models/xfeat_320x256_fp16.onnx",
                "xfeat_320x256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp32.onnx",
                "outputs/colab_l4_v5_nav/harvest/xfeat_320x256_fp32.onnx",
                "models/xfeat_320x256_fp32.onnx",
                "outputs/xvs_nav_v5/xfeat_320x256_fp32.onnx",
                "xfeat_320x256_fp32.onnx",
            ],
            XFEAT_ONNX_PINS,
            1,
        );
        models_meta.insert("xvs_xfeat".to_string(), xfeat_meta);

        // 6. Stage-2B LighterGlue (2.455 MB FP16 static K=256 prioritized -> FP32 fallback; SHA-256 fail-closed)
        let (lighterglue_session, lighterglue_meta) = load_pinned_onnx_session(
            ort_env.as_ref(),
            "XVS-LighterGlue Stage-2B (K=256 Static Matcher FP16 ONNX)",
            "lighterglue_k256_fp16.onnx",
            "READY_RUST_ORT_V5",
            "MISSING_LIGHTERGLUE_ONNX",
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp16.onnx",
                "models/lighterglue_k256_fp16.onnx",
                "lighterglue_k256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp32.onnx",
                "models/lighterglue_k256_fp32.onnx",
                "lighterglue_k256_fp32.onnx",
            ],
            LIGHTERGLUE_ONNX_PINS,
            1,
        );
        models_meta.insert("xvs_lighterglue".to_string(), lighterglue_meta);

        // 7. Direct Frozen Rust `xvs-active-range` + `xvs-rta-core` + additive `xvs-multiship-coord`
        models_meta.insert(
            "xvs_active_range".to_string(),
            ModelStatusEntry {
                name: "XVS-ActiveRange + MultiShipCoord (F1/F3/F4) + SimplexRta (no_std Rust)"
                    .to_string(),
                loaded: true,
                real_weights: true,
                device: "native-rust".to_string(),
                backend: "rust-direct".to_string(),
                status: "LOADED_26_26_KANI_MULTISHIP_COORD".to_string(),
                sha256: Some(ODD_YAML_SHA256.to_string()),
                expected_sha256: Some(ODD_YAML_SHA256.to_string()),
                filename: Some("xvs-active-range + xvs-multiship-coord + xvs-rta-core".to_string()),
            },
        );

        let (d2_h, d2_w, d2_hc, d2_hh, d2_hw) = d2_session
            .as_ref()
            .map(resolve_d2_input_shapes)
            .unwrap_or((D2_CROP_SIZE, D2_CROP_SIZE, D2_HID_C, D2_HID_HW, D2_HID_HW));
        let media_lib = StudioMediaLibrary::discover();
        let mut d2_state = D2TemporalState::new(d2_h, d2_w, d2_hc, d2_hh, d2_hw);
        // Prime initial 3-ship anchor tick so F4 ghost-suppression hysteresis is armed cleanly
        d2_state.step_multiship_mesh_guard();
        let (nav_corridor_coords, nav_corridor_gallery) = build_nav_corridor_grid_and_gallery();
        let seq_vpr = SequenceVprFilter::new(
            nav_corridor_coords.clone(),
            SequenceVprConfig::default(),
        )
        .expect("default SequenceVprFilter config valid");
        let nav_cusum = DualBankEsEkfCusum::new(DualBankConfig::default())
            .expect("default DualBankEsEkfCusum config valid");

        Self {
            _ort_env: ort_env,
            ort_loaded,
            ort_dylib_label,
            d2_session,
            thermal_session,
            land_session,
            nav_session,
            xfeat_session,
            lighterglue_session,
            land_calibrator,
            d2_state,
            rta: SimplexRta::new(),
            last_rta_mode_id: SimplexMode::NominalMl as u8,
            last_rta_inhibit_turns: false,
            sensor_health: SensorHealthMonitor::new(),
            nav_cusum,
            seq_vpr,
            nav_corridor_coords,
            nav_corridor_gallery,
            nav_corridor_step: 0,
            last_nav_integrity: NavIntegritySnapshot::default(),
            sim_clock_ms: 1_000.0,
            media_lib,
            last_channel_latencies_ms: BTreeMap::new(),
            models_meta,
            last_loopback: None,
            scratch_d2_gray: Vec::with_capacity(d2_h * d2_w),
            scratch_d2_x: Vec::with_capacity(6 * d2_h * d2_w),
            scratch_land_inp: Vec::with_capacity(3 * 384 * 384),
            scratch_thermal_eo: Vec::with_capacity(3 * 256 * 320),
            lwir_scratch: LwirScratch::new(),
            scratch_nav_inp: Vec::with_capacity(3 * 224 * 224),
            scratch_xfeat_inp: Vec::with_capacity(3 * XFEAT_IN_H * XFEAT_IN_W),
            scratch_lg_k0: Vec::with_capacity(LIGHTERGLUE_STATIC_K * 2),
            scratch_lg_d0: Vec::with_capacity(LIGHTERGLUE_STATIC_K * XFEAT_DESC_DIM),
            scratch_lg_k1: Vec::with_capacity(LIGHTERGLUE_STATIC_K * 2),
            scratch_lg_d1: Vec::with_capacity(LIGHTERGLUE_STATIC_K * XFEAT_DESC_DIM),
            sat_ref_xfeat_cache: HashMap::with_capacity(SAT_REF_XFEAT_CACHE_CAPACITY),
            sat_ref_xfeat_order: VecDeque::with_capacity(SAT_REF_XFEAT_CACHE_CAPACITY),
            last_slow_scenario: None,
            last_slow_land: None,
            last_slow_nav: None,
        }
    }

    pub fn health(&self) -> NativeHealthPayload {
        NativeHealthPayload {
            version: STUDIO_VERSION,
            status: "ok",
            transport: "tauri-ipc-rust-ort",
            kani_proofs: "26/26",
            odd_yaml_sha256: ODD_YAML_SHA256,
            host_gpu_name: "Pure-Rust ORT (Zero-Python Native IPC)".to_string(),
            compute_device: "cpu-ort-rust".to_string(),
            ort_loaded: self.ort_loaded,
            ort_dylib_label: self.ort_dylib_label.clone(),
            vram_mb: 0.0,
            media_available: self.media_lib.is_available(),
            multiship_coord_enabled: true,
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
            models: self.models_meta.clone(),
        }
    }

    pub fn sensor_health(&self) -> NativeSensorHealthPayload {
        let skew_sanitized = if self.sensor_health.timestamp_skew_ms.is_finite() {
            self.sensor_health.timestamp_skew_ms
        } else {
            9999.0
        };
        NativeSensorHealthPayload {
            eo_blur_score: round2(self.sensor_health.eo_blur_score),
            eo_glare_ratio: round4(self.sensor_health.eo_glare_ratio),
            lwir_nuc_frozen: self.sensor_health.lwir_nuc_frozen,
            lwir_freeze_ticks: self.sensor_health.lwir_freeze_ticks,
            timestamp_skew_ms: skew_sanitized,
            q_eo_effective: round4(self.sensor_health.q_eo_effective),
            q_lwir_effective: round4(self.sensor_health.q_lwir_effective),
            thermal_mode: self.sensor_health.thermal_mode.as_str().to_string(),
            multiship_coord_enabled: true,
            f1_far91113_active: self.d2_state.f1_far91113_active,
            f4_ghost_hysteresis_active: self.d2_state.f4_ghost_hysteresis_active,
            eo_occlusion_ratio: round4(self.sensor_health.eo_occlusion_ratio),
            eo_focus_quality: round4(self.sensor_health.eo_focus_quality),
            eo_dead_line_fraction: round4(self.sensor_health.eo_dead_line_fraction),
            eo_spatial_autocorr: round4(self.sensor_health.eo_spatial_autocorr),
            eo_dead_pixel_fpn_fault: self.sensor_health.eo_dead_pixel_fpn_fault,
            eo_sensor_noise_fault: self.sensor_health.eo_sensor_noise_fault,
            eo_degraded: self.sensor_health.eo_degraded,
            eo_flags: self.sensor_health.eo_flags(),
            grid_4x4_q_eo: self.sensor_health.grid_4x4_q_eo,
            sector_q_eo: self.sensor_health.sector_q_eo,
            min_sector_q_eo: round4(self.sensor_health.min_sector_q_eo),
            degraded_sector_mask: self.sensor_health.degraded_sector_mask,
            lwir_ffc_imminent: self.sensor_health.lwir_ffc_imminent,
            lwir_ffc_in_progress: self.sensor_health.lwir_ffc_in_progress,
            lwir_dynamic_range_p95_p05: round2(self.sensor_health.lwir_dynamic_range_p95_p05),
            lwir_effective_dynamic_range: round2(self.sensor_health.lwir_effective_dynamic_range),
            lwir_contrast_collapsed: self.sensor_health.lwir_contrast_collapsed,
            lwir_localized_hot_target_rescue: self.sensor_health.lwir_localized_hot_target_rescue,
            lwir_dead_line_fraction: round4(self.sensor_health.lwir_dead_line_fraction),
            lwir_spatial_autocorr: round4(self.sensor_health.lwir_spatial_autocorr),
            lwir_dead_pixel_fpn_fault: self.sensor_health.lwir_dead_pixel_fpn_fault,
            lwir_sensor_noise_fault: self.sensor_health.lwir_sensor_noise_fault,
            lwir_q_optical: round4(self.sensor_health.lwir_q_optical),
            lwir_degraded: self.sensor_health.lwir_degraded,
            lwir_flags: self.sensor_health.lwir_flags(),
            timestamp_monotonic_ok: self.sensor_health.timestamp_monotonic_ok,
            timestamp_skew_fault: self.sensor_health.timestamp_skew_fault,
            fsync_precision_ok: self.sensor_health.fsync_precision_ok,
            bearing_track_confirmed: self.d2_state.bearing_tbc.is_confirmed(),
            bearing_track_hits_in_window: self.d2_state.bearing_tbc.hits_in_window(),
            ring0_authority_inhibited: self.d2_state.ring0_inhibited,
            last_swerve_role: swerve_role_label(self.d2_state.coord_active_range.last_role())
                .to_string(),
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
        }
    }

    pub fn sync_external_rta_state(&mut self, mode_id: u8, inhibit_turns: bool) {
        self.last_rta_mode_id = mode_id;
        self.last_rta_inhibit_turns = inhibit_turns;
    }

    pub fn nav_integrity_snapshot(&self) -> NavIntegritySnapshot {
        self.last_nav_integrity.clone()
    }

    pub fn step_rta(&mut self, req: &NativeRtaInput) -> RtaReport {
        let rep = evaluate_rta_step(&mut self.rta, req);
        self.last_rta_mode_id = rep.mode_id;
        self.last_rta_inhibit_turns = rep.inhibit_avoidance_turns;
        rep
    }

    pub fn infer_d2_rgb(
        &mut self,
        img: &RgbImage,
        return_jpeg: bool,
    ) -> Result<NativeChannelPayload, String> {
        let t0 = Instant::now();
        let eo_q = self.sensor_health.observe_eo_frame(img);
        let (q_eo_eff, _) = self.sensor_health.compute_effective_qualities(1.0, 1.0);
        self.sensor_health.q_eo_effective = q_eo_eff;
        let eo_optical_fault = eo_q.dead_pixel_fpn_fault
            || eo_q.sensor_noise_fault
            || eo_q.degraded
            || q_eo_eff < EO_SECTOR_DEGRADED_THR
            || self.sensor_health.timestamp_skew_fault;

        let (w_in, h_in) = (img.width() as f64, img.height() as f64);
        let d2_h = self.d2_state.in_h;
        let d2_w = self.d2_state.in_w;
        let hid_c = self.d2_state.hid_c;
        let hid_h = self.d2_state.hid_h;
        let hid_w = self.d2_state.hid_w;
        let n_px = d2_h * d2_w;

        let resized = image::imageops::resize(img, d2_w as u32, d2_h as u32, FilterType::Triangle);
        self.scratch_d2_gray.clear();
        self.scratch_d2_gray.reserve(n_px);
        for rgb in resized.as_raw().as_chunks::<3>().0 {
            let y = 0.299 * (rgb[0] as f32) + 0.587 * (rgb[1] as f32) + 0.114 * (rgb[2] as f32);
            self.scratch_d2_gray.push(y);
        }

        // Reuse evicted VecDeque buffer when history is full -> zero per-tick c0 allocation
        let mut c0 = if self.d2_state.norm_gray_hist.len() >= 8 {
            self.d2_state.norm_gray_hist.pop_front().unwrap_or_default()
        } else {
            Vec::with_capacity(n_px)
        };
        scene_normalize_gray_into(&self.scratch_d2_gray, &mut c0);

        let prev1_opt = self.d2_state.norm_gray_hist.back();
        let r1_val = if prev1_opt.is_some() { 1.0_f32 } else { 0.0_f32 };
        let prev5_opt = if self.d2_state.norm_gray_hist.len() >= 5 {
            self.d2_state
                .norm_gray_hist
                .get(self.d2_state.norm_gray_hist.len() - 5)
        } else {
            None
        };
        let r5_val = if prev5_opt.is_some() { 1.0_f32 } else { 0.0_f32 };

        self.scratch_d2_x.resize(6 * n_px, 0.0_f32);
        let (ch0, rest) = self.scratch_d2_x.split_at_mut(n_px);
        let (ch1, rest) = rest.split_at_mut(n_px);
        let (ch2, rest) = rest.split_at_mut(n_px);
        let (ch3, rest) = rest.split_at_mut(n_px);
        let (ch4, ch5) = rest.split_at_mut(n_px);

        ch0.copy_from_slice(&c0);
        if let Some(c1) = prev1_opt {
            ch1.copy_from_slice(c1);
            for ((dst, &v0), &v1) in ch3.iter_mut().zip(c0.iter()).zip(c1.iter()) {
                *dst = v0 - v1;
            }
        } else {
            ch1.fill(0.0);
            ch3.fill(0.0);
        }
        if let Some(c2) = prev5_opt {
            ch2.copy_from_slice(c2);
            for ((dst, &v0), &v2) in ch4.iter_mut().zip(c0.iter()).zip(c2.iter()) {
                *dst = v0 - v2;
            }
        } else {
            ch2.fill(0.0);
            ch4.fill(0.0);
        }
        ch5.fill(r1_val * r5_val);

        self.d2_state.norm_gray_hist.push_back(c0);

        let mut det: Option<[f32; 5]> = None;
        let mut max_prob: f32 = 0.0;
        let real_weights = self.d2_session.is_some();

        if let Some(sess) = self.d2_session.as_mut() {
            let tx = TensorRef::from_array_view(([1usize, 6, d2_h, d2_w], self.scratch_d2_x.as_slice()))
                .map_err(|e| format!("D2 x_t tensor error: {e}"))?;
            let th = TensorRef::from_array_view((
                [1usize, hid_c, hid_h, hid_w],
                self.d2_state.h_prev.as_slice(),
            ))
            .map_err(|e| format!("D2 h_prev tensor error: {e}"))?;
            let outs = sess
                .run(ort::inputs!["x_t" => tx, "h_prev_warped" => th])
                .map_err(|e| format!("D2 ORT run error: {e}"))?;
            let (hm_shape, hm) = outs["hm_logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("D2 hm_logits extract: {e}"))?;
            let (_, off) = outs["offset"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("D2 offset extract: {e}"))?;
            let (_, lsc) = outs["log_scale"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("D2 log_scale extract: {e}"))?;
            let (_, pmk) = outs["peak_mask"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("D2 peak_mask extract: {e}"))?;
            let (_, h_next) = outs["h_next"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("D2 h_next extract: {e}"))?;
            let (hm_h, hm_w) = if hm_shape.len() == 4 && hm_shape[2] > 0 && hm_shape[3] > 0 {
                (hm_shape[2] as usize, hm_shape[3] as usize)
            } else {
                ((d2_h / 2).max(1), (d2_w / 2).max(1))
            };
            let (d, mp) = decode_d2_top(hm, off, lsc, pmk, hm_h, hm_w, 0.25);
            det = d;
            max_prob = mp;
            if h_next.len() == self.d2_state.h_prev.len() {
                self.d2_state.h_prev.copy_from_slice(h_next);
            }
        }

        let raw_detected = det.is_some();
        let safe_max_prob = if max_prob.is_finite() {
            max_prob.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let det = if eo_optical_fault { None } else { det };

        let (cx_px, cy_px, size_px, size_rate, score) = if let Some(b) = det {
            let sx = w_in / (d2_w as f64);
            let sy = h_in / (d2_h as f64);
            let fx1 = (b[0] as f64) * sx;
            let fy1 = (b[1] as f64) * sy;
            let fx2 = (b[2] as f64) * sx;
            let fy2 = (b[3] as f64) * sy;
            let sz = ((fx2 - fx1) * (fy2 - fy1)).max(1.0).sqrt();
            let sr = self
                .d2_state
                .last_size_px
                .map(|prev| (sz - prev) / 0.1)
                .unwrap_or(0.0);
            self.d2_state.last_size_px = Some(sz);
            self.d2_state.track_age += 1;
            (0.5 * (fx1 + fx2), 0.5 * (fy1 + fy2), sz, sr, b[4] as f64)
        } else {
            (
                0.5 * w_in,
                0.5 * h_in,
                self.d2_state.last_size_px.unwrap_or(10.0),
                0.0,
                0.0,
            )
        };

        let focal_px = (AOT_FOCAL_PX * (w_in / AOT_REF_WIDTH_PX)).max(1.0);
        let alpha_rad = (cx_px - 0.5 * w_in).atan2(focal_px);
        let beta_rad = (0.5 * h_in - cy_px).atan2(focal_px);
        let tbc_snap = self.d2_state.bearing_tbc.observe_with_health(
            det.map(|_| (alpha_rad, beta_rad)),
            q_eo_eff,
            eo_optical_fault,
        );
        let own = OwnshipStateNed {
            v_ned: [35.0, 0.0, 0.0],
            a_ned: [0.0, 0.0, 0.0],
            heading_rad: 0.0,
            tas_mps: 35.0,
            v_stall_mps: 14.5,
        };
        let meas = ActiveRangeMeasurement {
            alpha_rad,
            beta_rad,
            size_px,
            size_rate_px_s: size_rate,
            contrast_ratio: 0.65,
            sky_beta_ext: 2.0e-4,
            daytime_clear: true,
            track_age: self.d2_state.track_age.max(1),
            dt_s: 0.1,
        };
        let authority = Ring0AuthorityGate {
            rta_inhibit_avoidance_turns: self.last_rta_inhibit_turns,
            hal_out_mode_wire: self.last_rta_mode_id,
        };
        let telem = self
            .d2_state
            .coord_active_range
            .step(&own, &meas, None, authority);
        let role = self.d2_state.coord_active_range.last_role();
        self.d2_state.ring0_inhibited = authority.inhibits_turns();
        self.d2_state.f1_far91113_active = role != SwerveCoordRole::InhibitedByP0;

        let wire = encode_wire(&telem);
        let wire_ok = decode_wire(&wire).is_some();
        let mut wire_hex = String::with_capacity(ACTIVE_RANGE_WIRE_BYTES * 2);
        for byte in wire {
            wire_hex.push_str(&format!("{byte:02x}"));
        }
        let mode_name = match telem.mode {
            0 => "NOMINAL_HOLD",
            1 => "ACTIVE_SWERVE",
            2 => "LOCKED",
            _ => "IMMEDIATE_EVADE",
        };
        let ar_report = ActiveRangeReport {
            mode: mode_name.to_string(),
            mode_id: telem.mode,
            r_hat_m: telem.r_hat_m,
            sigma_r_m: telem.sigma_r_m,
            r_safe_lower_m: telem.r_safe_lower_m,
            rho_dot_inv_s: telem.rho_dot_inv_s,
            cmd_bank_deg: (telem.cmd_bank_rad as f64).to_degrees(),
            min_sc3bf_margin: telem.min_sc3bf_margin,
            active_bank_idx: telem.active_bank_idx,
            swerve_coord_role: swerve_role_label(role).to_string(),
            f1_far91113_active: self.d2_state.f1_far91113_active,
            ring0_inhibited: self.d2_state.ring0_inhibited,
            wire_bytes: wire.len(),
            wire_crc_ok: wire_ok,
            wire_hex,
        };

        let latency_ms = round2(t0.elapsed().as_secs_f64() * 1000.0);
        self.last_channel_latencies_ms
            .insert("eo_sky".to_string(), latency_ms);
        let eo_flags = eo_q.flags();
        let summary = if eo_optical_fault {
            format!(
                "RUST-ORT D2 EO FAULT GATED ({}) | R_low {:.0}m | {mode_name}",
                eo_flags.join(","),
                telem.r_safe_lower_m
            )
        } else if det.is_some() {
            format!(
                "RUST-ORT D2 {score:.2} | R_low {:.0}m | {mode_name}",
                telem.r_safe_lower_m
            )
        } else {
            format!(
                "RUST-ORT SCAN {safe_max_prob:.2} | R_low {:.0}m | {mode_name}",
                telem.r_safe_lower_m
            )
        };

        let (image_b64, annotated_jpeg_b64) = if return_jpeg {
            let mut vis = image::imageops::resize(img, 384, 256, FilterType::Triangle);
            if let Some(b) = det {
                let x0 = (b[0] * (384.0 / (d2_w as f32))).round() as i32;
                let y0 = (b[1] * (256.0 / (d2_h as f32))).round() as i32;
                let x1 = (b[2] * (384.0 / (d2_w as f32))).round() as i32;
                let y1 = (b[3] * (256.0 / (d2_h as f32))).round() as i32;
                draw_rect_outline(&mut vis, x0, y0, x1, y1, Rgb([52, 211, 153]));
            }
            stamp_bottom_sim_only_strip(&mut vis);
            let b64 = encode_rgb_jpeg_b64(&vis, 84)?;
            let uri = format!("data:image/jpeg;base64,{b64}");
            (None, Some(uri))
        } else {
            (None, None)
        };

        let d2_confidence = if eo_optical_fault {
            0.0
        } else if det.is_some() && score.is_finite() {
            round4(score.clamp(0.0, 1.0))
        } else {
            round4(safe_max_prob as f64)
        };

        let sha = self.models_meta.get("xvs_d2").and_then(|m| m.sha256.clone());
        Ok(NativeChannelPayload {
            channel: "eo_sky".to_string(),
            model: "XVS-D2 + MultiShipCoord ActiveRange (Rust ORT)".to_string(),
            backend: "ort-load-dynamic".to_string(),
            real_weights,
            checkpoint_sha256: sha,
            latency_ms,
            confidence: d2_confidence,
            summary,
            annotated_jpeg_b64,
            image_b64,
            metrics: serde_json::json!({
                "detected": det.is_some(),
                "raw_detected": raw_detected,
                "max_prob": safe_max_prob,
                "eo_optical_fault": eo_optical_fault,
                "q_eo_effective": q_eo_eff,
                "eo_dead_line_fraction": eo_q.dead_line_fraction,
                "eo_spatial_autocorr": eo_q.spatial_autocorr,
                "eo_flags": eo_flags,
                "bearing_tbc": tbc_snap,
                "active_range": ar_report,
                "multiship_coord": {
                    "enabled": true,
                    "f1_far91113_active": self.d2_state.f1_far91113_active,
                    "f4_ghost_hysteresis_active": self.d2_state.f4_ghost_hysteresis_active,
                    "swerve_role": swerve_role_label(role),
                    "ring0_inhibited": self.d2_state.ring0_inhibited,
                },
            }),
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
        })
    }

    pub fn infer_land_rgb(
        &mut self,
        img: &RgbImage,
        return_jpeg: bool,
    ) -> Result<NativeChannelPayload, String> {
        let t0 = Instant::now();
        let (roi_h, roi_w) = self
            .land_session
            .as_ref()
            .map(resolve_land_input_hw)
            .unwrap_or((256, 256));
        let resized = image::imageops::resize(img, roi_w, roi_h, FilterType::Triangle);
        let h_sz = roi_h as usize;
        let w_sz = roi_w as usize;
        let n_px = h_sz * w_sz;
        self.scratch_land_inp.resize(3 * n_px, 0.0_f32);
        let inv255 = 1.0_f32 / 255.0_f32;
        let (r_plane, gb_planes) = self.scratch_land_inp.split_at_mut(n_px);
        let (g_plane, b_plane) = gb_planes.split_at_mut(n_px);
        for (idx, rgb) in resized.as_raw().as_chunks::<3>().0.iter().enumerate().take(n_px) {
            r_plane[idx] = (rgb[0] as f32) * inv255;
            g_plane[idx] = (rgb[1] as f32) * inv255;
            b_plane[idx] = (rgb[2] as f32) * inv255;
        }

        let real_weights = self.land_session.is_some();
        let parsed = if let Some(sess) = self.land_session.as_mut() {
            let troi = TensorRef::from_array_view((
                [1usize, 3, h_sz, w_sz],
                self.scratch_land_inp.as_slice(),
            ))
            .map_err(|e| format!("Land tensor error: {e}"))?;
            let outs = sess
                .run(ort::inputs!["roi_crop" => troi])
                .map_err(|e| format!("Land ORT run error: {e}"))?;
            let (_, kpts) = outs["keypoints_xy"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Land keypoints extract: {e}"))?;
            let (_, kl) = outs["keylines_hesse"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Land keylines extract: {e}"))?;
            let (_, cov) = outs["cov_2x2"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Land cov_2x2 extract: {e}"))?;
            let kl_sig_opt = if let Some(sig_val) = outs.get("keyline_sigma_px") {
                let (_, sig_slice) = sig_val
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("Land keyline_sigma_px extract: {e}"))?;
                Some(sig_slice)
            } else {
                None
            };
            let rwy_pres_opt = if let Some(pres_val) = outs.get("runway_present_logit") {
                let (_, pres_slice) = pres_val
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("Land runway_present_logit extract: {e}"))?;
                Some(pres_slice)
            } else {
                None
            };
            parse_xvs_land_outputs_with_presence(
                kpts,
                kl,
                cov,
                kl_sig_opt,
                rwy_pres_opt,
                &self.land_calibrator,
            )?
        } else {
            let mut kpts_fb = Vec::with_capacity(LAND_NUM_KEYPOINTS);
            for i in 0..LAND_NUM_KEYPOINTS {
                kpts_fb.push([128.0, 96.0 + (i as f64) * 10.0]);
            }
            let w_unif = 1.0 / (LAND_DEFAULT_KEYLINE_SIGMA_PX * LAND_DEFAULT_KEYLINE_SIGMA_PX);
            ParsedLandOutputs {
                contract: LandOutputContract::V51FiveOutput,
                keypoints_xy: kpts_fb,
                keylines_hesse: Vec::new(),
                calibrated_kp_sigmas_px: [3.5; LAND_NUM_KEYPOINTS],
                mean_keypoint_sigma_px: 3.5,
                keyline_sigma_px: None,
                keyline_precision_weights: [w_unif; LAND_NUM_KEYLINES],
                effective_keyline_sigma_px: LAND_DEFAULT_KEYLINE_SIGMA_PX,
                runway_present_logit: None,
                runway_present_prob: None,
                runway_present_admitted: true,
                runway_gate_mode: "NOMINAL",
            }
        };

        let scale_u_256 = 256.0 / (roi_w as f64);
        let scale_v_256 = 256.0 / (roi_h as f64);
        let kpts_xy: Vec<[f64; 2]> = parsed
            .keypoints_xy
            .iter()
            .map(|p| [p[0] * scale_u_256, p[1] * scale_v_256])
            .collect();
        let keylines = parsed.keylines_hesse;
        let mean_sigma_px = parsed.mean_keypoint_sigma_px;
        let center_u = kpts_xy.iter().map(|p| p[0]).sum::<f64>() / (LAND_NUM_KEYPOINTS as f64);
        let center_v = kpts_xy.iter().map(|p| p[1]).sum::<f64>() / (LAND_NUM_KEYPOINTS as f64);
        let lat_dev_deg = round2((center_u - 128.0) / 256.0 * 18.0);
        let glideslope_deg = round2(3.0 + (128.0 - center_v) / 256.0 * 2.5);
        let hpl_m = round2((mean_sigma_px * 14.5).max(12.0));
        let vpl_m = round2((mean_sigma_px * 5.2).max(6.0));
        let latency_ms = round2(t0.elapsed().as_secs_f64() * 1000.0);
        self.last_channel_latencies_ms
            .insert("runway_land".to_string(), latency_ms);
        let runway_admitted = parsed.runway_present_admitted;
        let summary = if runway_admitted {
            format!("RUST-ORT 12-KPT | GS {glideslope_deg:.1}deg | VPL {vpl_m:.0}m")
        } else {
            format!(
                "RUST-ORT {} | P_rwy={:.3} < {:.2}",
                parsed.runway_gate_mode,
                parsed.runway_present_prob.unwrap_or(0.0),
                LAND_RUNWAY_PRESENT_MIN_PROB
            )
        };

        let (image_b64, annotated_jpeg_b64) = if return_jpeg {
            let mut vis = image::imageops::resize(img, 384, 256, FilterType::Triangle);
            if runway_admitted {
                for (idx, pt) in kpts_xy.iter().enumerate() {
                    let px = (pt[0] / 256.0 * 384.0).round() as i32;
                    let py = pt[1].round() as i32;
                    let col = if idx % 2 == 0 {
                        Rgb([56, 189, 248])
                    } else {
                        Rgb([52, 211, 153])
                    };
                    draw_dot(&mut vis, px, py, 3, col);
                }
            }
            stamp_bottom_sim_only_strip(&mut vis);
            let b64 = encode_rgb_jpeg_b64(&vis, 84)?;
            let uri = format!("data:image/jpeg;base64,{b64}");
            (None, Some(uri))
        } else {
            (None, None)
        };

        let land_confidence = if runway_admitted {
            (1.0 - mean_sigma_px / 60.0).clamp(0.1, 0.99)
        } else {
            0.0
        };
        let sha = self
            .models_meta
            .get("xvs_land")
            .and_then(|m| m.sha256.clone());
        Ok(NativeChannelPayload {
            channel: "runway_land".to_string(),
            model: "XVS-Land (Rust ORT + StudentTCalibrator)".to_string(),
            backend: "ort-load-dynamic".to_string(),
            real_weights,
            checkpoint_sha256: sha,
            latency_ms,
            confidence: land_confidence,
            summary,
            annotated_jpeg_b64,
            image_b64,
            metrics: serde_json::json!({
                "onnx_output_contract": parsed.contract.as_str(),
                "roi_hw": [roi_h, roi_w],
                "keypoints_xy_256": kpts_xy,
                "keylines_hesse": keylines,
                "keyline_sigma_px": parsed.keyline_sigma_px,
                "keyline_precision_weights": parsed.keyline_precision_weights,
                "effective_keyline_sigma_px": round4(parsed.effective_keyline_sigma_px),
                "runway_present_prob": parsed.runway_present_prob.map(round4),
                "runway_present_admitted": parsed.runway_present_admitted,
                "runway_gate_mode": parsed.runway_gate_mode,
                "mean_keypoint_sigma_px": (mean_sigma_px * 1000.0).round() / 1000.0,
                "student_t_nu": self.land_calibrator.nu,
                "student_t_scale": self.land_calibrator.scale,
                "glideslope_deg": glideslope_deg,
                "lateral_dev_deg": lat_dev_deg,
                "hpl_m": hpl_m,
                "vpl_m": vpl_m,
                "hmi_count": 0,
            }),
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
        })
    }

    pub fn infer_thermal_rgb(
        &mut self,
        img: &RgbImage,
        q_eo: f32,
        q_lwir: f32,
        fog_density: f32,
        return_jpeg: bool,
    ) -> Result<NativeChannelPayload, String> {
        self.infer_thermal_pair_rgb(img, None, q_eo, q_lwir, fog_density, return_jpeg)
    }

    pub fn infer_thermal_pair_rgb(
        &mut self,
        eo_img: &RgbImage,
        lwir_img: Option<&RgbImage>,
        q_eo: f32,
        q_lwir: f32,
        fog_density: f32,
        return_jpeg: bool,
    ) -> Result<NativeChannelPayload, String> {
        let t0 = Instant::now();
        let fog_c = if fog_density.is_finite() {
            fog_density.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q_eo_input_valid = q_eo.is_finite();
        let q_lwir_input_valid = q_lwir.is_finite();
        let q_eo_sanitized = if q_eo_input_valid {
            round4((q_eo as f64).clamp(0.0, 1.0))
        } else {
            0.0
        };
        let q_lwir_sanitized = if q_lwir_input_valid {
            round4((q_lwir as f64).clamp(0.0, 1.0))
        } else {
            0.0
        };

        let mut eo_vis = image::imageops::resize(eo_img, 320, 256, FilterType::Triangle);
        if fog_c > 0.01 {
            let a = 1.0 - 0.88 * fog_c;
            let b = 220.0 * 0.88 * fog_c;
            for p in eo_vis.pixels_mut() {
                p[0] = ((p[0] as f32) * a + b).clamp(0.0, 255.0) as u8;
                p[1] = ((p[1] as f32) * a + b).clamp(0.0, 255.0) as u8;
                p[2] = ((p[2] as f32) * a + b).clamp(0.0, 255.0) as u8;
            }
        }

        // Real-world SensorHealthMonitor: run native-resolution FPN & white-noise ONCE per distinct
        // source image (eliminating 2x-3x redundant scans while preserving exact bit-parity).
        let eo_native_fpn_noise = compute_native_rgb_fpn_noise(eo_img);
        let eo_q = if fog_c > 0.01 {
            let q = SensorHealthMonitor::evaluate_eo_optical_quality_with_fpn_noise(
                &eo_vis,
                eo_native_fpn_noise,
            );
            self.sensor_health.record_eo_quality(q);
            q
        } else {
            let q = SensorHealthMonitor::evaluate_eo_optical_quality_with_fpn_noise(
                eo_img,
                eo_native_fpn_noise,
            );
            self.sensor_health.record_eo_quality(q);
            q
        };
        let lwir_native_fpn_noise = match lwir_img {
            Some(lw_im) => compute_native_rgb_fpn_noise(lw_im),
            None => eo_native_fpn_noise,
        };
        let lwir_src = lwir_img.unwrap_or(eo_img);
        build_lwir_2ch_from_rgb_into(lwir_src, 160, 128, &mut self.lwir_scratch);
        let lwir_st = self.sensor_health.observe_lwir_plane_with_ffc_precomputed(
            &self.lwir_scratch.lwir_u8,
            160,
            128,
            0,
            lwir_native_fpn_noise,
        );
        let lwir_frozen = lwir_st.nuc_frozen;
        let lwir_freeze_ticks = lwir_st.freeze_ticks;
        let (q_eo_eff, q_lwir_eff) = self
            .sensor_health
            .compute_effective_qualities(q_eo as f64, q_lwir as f64);

        let n_eo = 320 * 256;
        self.scratch_thermal_eo.resize(3 * n_eo, 0.0_f32);
        let inv255 = 1.0_f32 / 255.0_f32;
        let (eo_r, eo_gb) = self.scratch_thermal_eo.split_at_mut(n_eo);
        let (eo_g, eo_b) = eo_gb.split_at_mut(n_eo);
        for (idx, rgb) in eo_vis.as_raw().as_chunks::<3>().0.iter().enumerate().take(n_eo) {
            eo_r[idx] = (rgb[0] as f32) * inv255;
            eo_g[idx] = (rgb[1] as f32) * inv255;
            eo_b[idx] = (rgb[2] as f32) * inv255;
        }

        let q_eo_arr = [q_eo_eff as f32];
        let q_lwir_arr = [q_lwir_eff as f32];

        let mut mean_vacuity = 1.0_f64;
        let mut max_hm_prob = 0.0_f64;
        let mut w_eo_eff = 0.0_f64;
        let mut w_lwir_eff = 0.0_f64;
        let mut peak_xy = (40_usize, 32_usize);
        let real_weights = self.thermal_session.is_some();

        if let Some(sess) = self.thermal_session.as_mut() {
            let teo = TensorRef::from_array_view((
                [1usize, 3, 256, 320],
                self.scratch_thermal_eo.as_slice(),
            ))
            .map_err(|e| format!("Thermal x_eo tensor: {e}"))?;
            let tlw = TensorRef::from_array_view((
                [1usize, 2, 128, 160],
                self.lwir_scratch.out.as_slice(),
            ))
            .map_err(|e| format!("Thermal x_lwir tensor: {e}"))?;
            let tqe = TensorRef::from_array_view(([1usize], q_eo_arr.as_slice()))
                .map_err(|e| format!("Thermal q_eo tensor: {e}"))?;
            let tql = TensorRef::from_array_view(([1usize], q_lwir_arr.as_slice()))
                .map_err(|e| format!("Thermal q_lwir tensor: {e}"))?;
            let outs = sess
                .run(ort::inputs![
                    "x_eo" => teo,
                    "x_lwir" => tlw,
                    "q_eo" => tqe,
                    "q_lwir" => tql
                ])
                .map_err(|e| format!("Thermal ORT run: {e}"))?;

            let (_, hm) = outs["fused_hm_logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Thermal hm extract: {e}"))?;
            let (_, vac) = outs["vacuity"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Thermal vacuity extract: {e}"))?;
            let (_, weo) = outs["w_eo_eff"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Thermal w_eo_eff extract: {e}"))?;
            let (_, wlw) = outs["w_lwir_eff"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Thermal w_lwir_eff extract: {e}"))?;

            mean_vacuity = vac.iter().map(|&v| v as f64).sum::<f64>() / (vac.len().max(1) as f64);
            w_eo_eff = weo.iter().map(|&v| v as f64).sum::<f64>() / (weo.len().max(1) as f64);
            w_lwir_eff = wlw.iter().map(|&v| v as f64).sum::<f64>() / (wlw.len().max(1) as f64);
            let mut best_idx = 0usize;
            for (i, &logit) in hm.iter().enumerate() {
                let p = sigmoid_f32(logit) as f64;
                if p > max_hm_prob {
                    max_hm_prob = p;
                    best_idx = i;
                }
            }
            peak_xy = (best_idx % 80, best_idx / 80);
        }

        if !mean_vacuity.is_finite() {
            mean_vacuity = 1.0;
        } else {
            mean_vacuity = mean_vacuity.clamp(0.0, 1.0);
        }
        if !w_eo_eff.is_finite() {
            w_eo_eff = 0.0;
        }
        if !w_lwir_eff.is_finite() {
            w_lwir_eff = 0.0;
        }
        if !max_hm_prob.is_finite() {
            max_hm_prob = 0.0;
        }

        let simplex = select_thermal_simplex_mode(q_eo_eff, q_lwir_eff, mean_vacuity);
        self.sensor_health.q_eo_effective = q_eo_eff;
        self.sensor_health.q_lwir_effective = q_lwir_eff;
        self.sensor_health.thermal_mode = simplex;

        let latency_ms = round2(t0.elapsed().as_secs_f64() * 1000.0);
        self.last_channel_latencies_ms
            .insert("thermal_dve".to_string(), latency_ms);
        let summary = if lwir_frozen {
            format!("RUST-ORT LWIR NUC FROZEN -> {} | Vac {mean_vacuity:.2}", simplex.as_str())
        } else {
            format!("RUST-ORT {} | Vac {mean_vacuity:.2}", simplex.as_str())
        };

        let (image_b64, annotated_jpeg_b64) = if return_jpeg {
            // Direct split-screen composite into 384x256 with ZERO redundant full-frame resizes
            let eo_raw = eo_vis.as_raw();
            let lwir_u8 = &self.lwir_scratch.lwir_u8;
            let mut vis = RgbImage::new(384, 256);
            let vis_raw = vis.as_mut();
            let real_lwir_info = lwir_img.and_then(|im| {
                let lw = im.width() as usize;
                let lh = im.height() as usize;
                if lw > 0 && lh > 0 {
                    Some((im.as_raw(), lw, lh))
                } else {
                    None
                }
            });
            for y in 0..256_usize {
                let ly = (y * 128 / 256).min(127);
                let eo_row = y * 320 * 3;
                let dst_row = y * 384 * 3;
                // Left half (x in 0..192): sample left half of eo_vis (320x256 -> 384x256)
                for x in 0..192_usize {
                    let sx = (x * 320 / 384).min(319);
                    let s = eo_row + sx * 3;
                    let d = dst_row + x * 3;
                    vis_raw[d] = eo_raw[s];
                    vis_raw[d + 1] = eo_raw[s + 1];
                    vis_raw[d + 2] = eo_raw[s + 2];
                }
                // Center split divider (x == 192)
                let mid = dst_row + 192 * 3;
                vis_raw[mid] = 251;
                vis_raw[mid + 1] = 191;
                vis_raw[mid + 2] = 36;
                // Right half (x in 193..384): real LWIR or false-color LWIR
                if let Some((lw_raw, lw_w, lw_h)) = real_lwir_info {
                    let sy = (y * lw_h / 256).min(lw_h - 1);
                    let lw_row = sy * lw_w * 3;
                    for x in 193..384_usize {
                        let sx = (x * lw_w / 384).min(lw_w - 1);
                        let s = lw_row + sx * 3;
                        let d = dst_row + x * 3;
                        vis_raw[d] = lw_raw[s];
                        vis_raw[d + 1] = lw_raw[s + 1];
                        vis_raw[d + 2] = lw_raw[s + 2];
                    }
                } else {
                    let lw_row = ly * 160;
                    for x in 193..384_usize {
                        let lx = (x * 160 / 384).min(159);
                        let v = lwir_u8[lw_row + lx];
                        let d = dst_row + x * 3;
                        vis_raw[d] = v.saturating_add(40);
                        vis_raw[d + 1] = ((v as u16) * 2 / 3) as u8;
                        vis_raw[d + 2] = 255u8.saturating_sub(v) / 2;
                    }
                }
            }
            let px = (((peak_xy.0 as f64) + 0.5) / 80.0 * 384.0).round() as i32;
            let py = (((peak_xy.1 as f64) + 0.5) / 64.0 * 256.0).round() as i32;
            draw_dot(&mut vis, px, py, 5, Rgb([52, 211, 153]));
            stamp_bottom_sim_only_strip(&mut vis);
            let b64 = encode_rgb_jpeg_b64(&vis, 84)?;
            let uri = format!("data:image/jpeg;base64,{b64}");
            (None, Some(uri))
        } else {
            (None, None)
        };

        let sha = self
            .models_meta
            .get("xvs_thermal")
            .and_then(|m| m.sha256.clone());
        Ok(NativeChannelPayload {
            channel: "thermal_dve".to_string(),
            model: "XVS-ThermalNet + SensorHealthMonitor (Rust ORT)".to_string(),
            backend: "ort-load-dynamic".to_string(),
            real_weights,
            checkpoint_sha256: sha,
            latency_ms,
            confidence: round4((1.0 - mean_vacuity).clamp(0.0, 1.0)),
            summary,
            annotated_jpeg_b64,
            image_b64,
            metrics: serde_json::json!({
                "q_eo": q_eo_sanitized,
                "q_lwir": q_lwir_sanitized,
                "q_eo_input_non_finite": !q_eo_input_valid,
                "q_lwir_input_non_finite": !q_lwir_input_valid,
                "q_eo_effective": q_eo_eff,
                "q_lwir_effective": q_lwir_eff,
                "eo_blur_score": eo_q.blur_score,
                "eo_glare_ratio": eo_q.glare_ratio,
                "eo_occlusion_ratio": eo_q.occlusion_ratio,
                "eo_dead_line_fraction": eo_q.dead_line_fraction,
                "eo_fpn_score": eo_q.fpn_score,
                "eo_spatial_autocorr": eo_q.spatial_autocorr,
                "eo_noise_score": eo_q.noise_score,
                "eo_degraded": eo_q.degraded,
                "eo_flags": eo_q.flags(),
                "sector_q_eo": eo_q.sector_q_eo,
                "min_sector_q_eo": eo_q.min_sector_q_eo,
                "degraded_sector_mask": eo_q.degraded_sector_mask,
                "lwir_nuc_frozen": lwir_frozen,
                "lwir_freeze_ticks": lwir_freeze_ticks,
                "lwir_dynamic_range_p95_p05": self.sensor_health.lwir_dynamic_range_p95_p05,
                "lwir_effective_dynamic_range": self.sensor_health.lwir_effective_dynamic_range,
                "lwir_contrast_collapsed": self.sensor_health.lwir_contrast_collapsed,
                "lwir_localized_hot_target_rescue": self.sensor_health.lwir_localized_hot_target_rescue,
                "lwir_dead_line_fraction": self.sensor_health.lwir_dead_line_fraction,
                "lwir_fpn_score": self.sensor_health.lwir_fpn_score,
                "lwir_spatial_autocorr": self.sensor_health.lwir_spatial_autocorr,
                "lwir_noise_score": self.sensor_health.lwir_noise_score,
                "lwir_q_optical": self.sensor_health.lwir_q_optical,
                "lwir_degraded": self.sensor_health.lwir_degraded,
                "lwir_flags": self.sensor_health.lwir_flags(),
                "lwir_ffc_imminent": self.sensor_health.lwir_ffc_imminent,
                "lwir_ffc_in_progress": self.sensor_health.lwir_ffc_in_progress,
                "fsync_precision_ok": self.sensor_health.fsync_precision_ok,
                "w_eo_eff": round4(w_eo_eff),
                "w_lwir_eff": round4(w_lwir_eff),
                "mean_vacuity": round4(mean_vacuity),
                "max_heatmap_prob": round4(max_hm_prob),
                "peak_grid_xy": [peak_xy.0, peak_xy.1],
                "simplex_mode": simplex.as_str(),
                "simplex_mode_id": simplex as u8,
            }),
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
        })
    }

    /// Run Stage-2A `xfeat_320x256_fp16.onnx` (or FP32 fallback) real ONNX inference + dense head decode.
    pub fn infer_xfeat_rgb(&mut self, img: &RgbImage) -> Result<(Vec<XFeatKeypoint>, f64), String> {
        if self.xfeat_session.is_none() {
            return Ok((Vec::new(), 0.0));
        }
        let resized = image::imageops::resize(
            img,
            XFEAT_IN_W as u32,
            XFEAT_IN_H as u32,
            FilterType::Triangle,
        );
        let n_px = XFEAT_IN_W * XFEAT_IN_H;
        self.scratch_xfeat_inp.resize(3 * n_px, 0.0_f32);
        let inv255 = 1.0_f32 / 255.0_f32;
        let (r_p, gb_p) = self.scratch_xfeat_inp.split_at_mut(n_px);
        let (g_p, b_p) = gb_p.split_at_mut(n_px);
        for (idx, rgb) in resized.as_raw().as_chunks::<3>().0.iter().enumerate().take(n_px) {
            r_p[idx] = (rgb[0] as f32) * inv255;
            g_p[idx] = (rgb[1] as f32) * inv255;
            b_p[idx] = (rgb[2] as f32) * inv255;
        }
        let sess = self.xfeat_session.as_mut().expect("checked xfeat_session");
        let t_xf = Instant::now();
        let tin = TensorRef::from_array_view((
            [1usize, 3, XFEAT_IN_H, XFEAT_IN_W],
            self.scratch_xfeat_inp.as_slice(),
        ))
        .map_err(|e| format!("XFeat tensor: {e}"))?;
        let outs = sess
            .run(ort::inputs!["img_s2" => tin])
            .map_err(|e| format!("XFeat ORT run: {e}"))?;
        let (_, feats) = outs["feats"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("XFeat feats extract: {e}"))?;
        let (_, kpts_logits) = outs["keypoints"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("XFeat keypoints extract: {e}"))?;
        let (_, heatmap) = outs["heatmap"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("XFeat heatmap extract: {e}"))?;
        let kpts = decode_xfeat_dense_heads(
            feats,
            kpts_logits,
            heatmap,
            XFEAT_MAX_KEYPOINTS,
            0.05,
            2,
            4,
        );
        let xfeat_ms = round2(t_xf.elapsed().as_secs_f64() * 1000.0);
        self.last_channel_latencies_ms
            .insert("xvs_xfeat".to_string(), xfeat_ms);
        Ok((kpts, xfeat_ms))
    }

    fn infer_xfeat_sat_ref_cached(
        &mut self,
        ref_img: &RgbImage,
        sat_ref_path: Option<&Path>,
    ) -> Result<Vec<XFeatKeypoint>, String> {
        let key = match sat_ref_path {
            Some(p) => format!("path:{}", p.to_string_lossy()),
            None => format!(
                "sha:{}x{}:{}",
                ref_img.width(),
                ref_img.height(),
                sha256_bytes_hex(ref_img.as_raw())
            ),
        };
        if let Some(cached) = self.sat_ref_xfeat_cache.get(&key) {
            return Ok(cached.clone());
        }
        let (kpts, _) = self.infer_xfeat_rgb(ref_img)?;
        if self.sat_ref_xfeat_cache.len() >= SAT_REF_XFEAT_CACHE_CAPACITY {
            if let Some(evict) = self.sat_ref_xfeat_order.pop_front() {
                self.sat_ref_xfeat_cache.remove(&evict);
            }
        }
        self.sat_ref_xfeat_order.push_back(key.clone());
        self.sat_ref_xfeat_cache.insert(key, kpts.clone());
        Ok(kpts)
    }

    /// Run Stage-2B `lighterglue_k256_fp16.onnx` static K=256 `[1,256,2]/[1,256,64] -> [1,256,256]`
    /// real ONNX matching on XFeat keypoints and 64-D descriptors.
    pub fn infer_lighterglue_match(
        &mut self,
        kpts0: &[XFeatKeypoint],
        kpts1: &[XFeatKeypoint],
    ) -> Result<(usize, f64, f64), String> {
        if self.lighterglue_session.is_none() {
            return Ok((0, 0.0, 0.0));
        }
        let n0 = pack_lighterglue_k256_into(kpts0, &mut self.scratch_lg_k0, &mut self.scratch_lg_d0);
        let n1 = pack_lighterglue_k256_into(kpts1, &mut self.scratch_lg_k1, &mut self.scratch_lg_d1);
        let k = LIGHTERGLUE_STATIC_K;
        let sess = self
            .lighterglue_session
            .as_mut()
            .expect("checked lighterglue_session");
        let t_lg = Instant::now();
        let tk0 = TensorRef::from_array_view(([1usize, k, 2], self.scratch_lg_k0.as_slice()))
            .map_err(|e| format!("LighterGlue kpts0 tensor: {e}"))?;
        let tk1 = TensorRef::from_array_view(([1usize, k, 2], self.scratch_lg_k1.as_slice()))
            .map_err(|e| format!("LighterGlue kpts1 tensor: {e}"))?;
        let td0 = TensorRef::from_array_view((
            [1usize, k, XFEAT_DESC_DIM],
            self.scratch_lg_d0.as_slice(),
        ))
        .map_err(|e| format!("LighterGlue desc0 tensor: {e}"))?;
        let td1 = TensorRef::from_array_view((
            [1usize, k, XFEAT_DESC_DIM],
            self.scratch_lg_d1.as_slice(),
        ))
        .map_err(|e| format!("LighterGlue desc1 tensor: {e}"))?;
        let outs = sess
            .run(ort::inputs![
                "kpts0" => tk0,
                "kpts1" => tk1,
                "desc0" => td0,
                "desc1" => td1
            ])
            .map_err(|e| format!("LighterGlue ORT run: {e}"))?;
        let (shape, scores) = outs["scores"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("LighterGlue scores extract: {e}"))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims != [1, k as i64, k as i64] || scores.len() != k * k {
            return Err(format!(
                "Unexpected LighterGlue scores shape: {dims:?} (len={})",
                scores.len()
            ));
        }
        if !scores.iter().all(|v| v.is_finite()) {
            return Err("Non-finite value in LighterGlue scores tensor".to_string());
        }
        let (matches, mean_prob) =
            decode_lighterglue_mutual_matches(scores, n0, n1, LIGHTERGLUE_MATCH_MIN_PROB);
        let lg_ms = round2(t_lg.elapsed().as_secs_f64() * 1000.0);
        self.last_channel_latencies_ms
            .insert("xvs_lighterglue".to_string(), lg_ms);
        Ok((matches, mean_prob, lg_ms))
    }

    pub fn infer_nav_rgb(
        &mut self,
        img: &RgbImage,
        gps_offset_m: f64,
        return_jpeg: bool,
    ) -> Result<NativeChannelPayload, String> {
        self.infer_nav_pair_rgb_with_key(img, None, None, gps_offset_m, return_jpeg)
    }

    pub fn infer_nav_pair_rgb(
        &mut self,
        img: &RgbImage,
        sat_ref: Option<&RgbImage>,
        gps_offset_m: f64,
        return_jpeg: bool,
    ) -> Result<NativeChannelPayload, String> {
        self.infer_nav_pair_rgb_with_key(img, sat_ref, None, gps_offset_m, return_jpeg)
    }

    fn infer_nav_pair_rgb_with_key(
        &mut self,
        img: &RgbImage,
        sat_ref: Option<&RgbImage>,
        sat_ref_path: Option<&Path>,
        gps_offset_m: f64,
        return_jpeg: bool,
    ) -> Result<NativeChannelPayload, String> {
        let t0 = Instant::now();
        let sha = self
            .models_meta
            .get("xvs_nav")
            .and_then(|m| m.sha256.clone());
        let xfeat_sha = self
            .models_meta
            .get("xvs_xfeat")
            .and_then(|m| m.sha256.clone());
        let lighterglue_sha = self
            .models_meta
            .get("xvs_lighterglue")
            .and_then(|m| m.sha256.clone());
        let xfeat_loaded = self.xfeat_session.is_some();
        let lighterglue_loaded = self.lighterglue_session.is_some();

        // Run Stage-2A XFeat FP16 ONNX keypoint extraction whenever loaded
        let (xfeat_kpts, xfeat_ort_ms) = if xfeat_loaded {
            self.infer_xfeat_rgb(img)?
        } else {
            (Vec::new(), 0.0)
        };
        let xfeat_kpt_count = xfeat_kpts.len();
        let xfeat_mean_score = if xfeat_kpt_count > 0 {
            round4(
                xfeat_kpts.iter().map(|k| k.score as f64).sum::<f64>()
                    / (xfeat_kpt_count as f64),
            )
        } else {
            0.0
        };

        // Run Stage-2B LighterGlue K=256 FP16 ONNX matching whenever loaded (caching static sat_ref XFeat!)
        let (lg_matches, lg_mean_prob, lg_ort_ms, lg_mode) =
            if lighterglue_loaded && !xfeat_kpts.is_empty() {
                if let Some(ref_img) = sat_ref {
                    let ref_kpts = if xfeat_loaded {
                        self.infer_xfeat_sat_ref_cached(ref_img, sat_ref_path)?
                    } else {
                        Vec::new()
                    };
                    if !ref_kpts.is_empty() {
                        let (m, p, ms) = self.infer_lighterglue_match(&xfeat_kpts, &ref_kpts)?;
                        (m, p, ms, "crossview_aerial_vs_sat")
                    } else {
                        let (m, p, ms) = self.infer_lighterglue_match(&xfeat_kpts, &xfeat_kpts)?;
                        (m, p, ms, "self_consistency_probe")
                    }
                } else {
                    let (m, p, ms) = self.infer_lighterglue_match(&xfeat_kpts, &xfeat_kpts)?;
                    (m, p, ms, "self_consistency_probe")
                }
            } else {
                (0, 0.0, 0.0, "inactive")
            };

        let gps_offset_valid = gps_offset_m.is_finite();
        let gps_offset_sanitized = if gps_offset_valid {
            gps_offset_m
        } else {
            9999.0
        };
        // Real Pure-Rust DualBankEsEkfCusum (10D multipath H0 observer + 8-arm Cholesky Page's CUSUM
        // + 2-sided vertical Down CUSUM) — strictly fail-closed on NaN/Inf without corrupting EKF banks.
        let cusum_out = if gps_offset_valid {
            self.nav_cusum.step(&DualBankStepInput {
                ins_accel_ne_mps2: [0.0, 0.0],
                baro_down_m: -120.0,
                gnss_ne_m: Some([gps_offset_m, 0.0]),
                vis_ne_m: Some([0.0, 0.0]),
                vis_gate_admitted: true,
                r_gnss: None,
                r_vis: None,
                gnss_down_m: None,
            })
        } else {
            self.nav_cusum.step(&DualBankStepInput {
                ins_accel_ne_mps2: [0.0, 0.0],
                baro_down_m: -120.0,
                gnss_ne_m: None,
                vis_ne_m: Some([0.0, 0.0]),
                vis_gate_admitted: true,
                r_gnss: None,
                r_vis: None,
                gnss_down_m: Some(gps_offset_m),
            })
        };
        let spoof_detected = !gps_offset_valid || cusum_out.spoof_alarm;
        let cusum_max_sanitized = sanitize_json_f64(cusum_out.max_cusum_stat, 9999.0);
        let cusum_vert_sanitized = sanitize_json_f64(cusum_out.vertical_cusum_stat, 9999.0);
        let nav_opt_q = SensorHealthMonitor::evaluate_eo_optical_quality(img);
        let nav_optical_gate = if nav_opt_q.blur_score < 0.5 && !xfeat_loaded {
            0.0
        } else {
            (nav_opt_q.fpn_score * nav_opt_q.noise_score).clamp(0.0, 1.0)
        };

        if let Some(sess) = self.nav_session.as_mut() {
            let resized = image::imageops::resize(img, 224, 224, FilterType::Triangle);
            let n_px = 224 * 224;
            self.scratch_nav_inp.resize(3 * n_px, 0.0_f32);
            let inv255 = 1.0_f32 / 255.0_f32;
            let (nr, ngb) = self.scratch_nav_inp.split_at_mut(n_px);
            let (ng, nb) = ngb.split_at_mut(n_px);
            for (idx, rgb) in resized.as_raw().as_chunks::<3>().0.iter().enumerate().take(n_px) {
                nr[idx] = (rgb[0] as f32) * inv255;
                ng[idx] = (rgb[1] as f32) * inv255;
                nb[idx] = (rgb[2] as f32) * inv255;
            }
            let tin = TensorRef::from_array_view((
                [1usize, 3, 224, 224],
                self.scratch_nav_inp.as_slice(),
            ))
            .map_err(|e| format!("Nav tensor: {e}"))?;
            let t_infer = Instant::now();
            let outs = sess
                .run(ort::inputs!["img_s1" => tin])
                .map_err(|e| format!("Nav ORT run: {e}"))?;
            let ort_infer_ms = round2(t_infer.elapsed().as_secs_f64() * 1000.0);
            let (_, emb) = outs["desc_global"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Nav desc_global extract: {e}"))?;
            if emb.len() != 512 {
                return Err(format!("Expected 512-D desc_global, got {}", emb.len()));
            }
            let l2 = emb
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                .sqrt();
            if !l2.is_finite() || (l2 - 1.0).abs() > 1e-2 {
                return Err(format!("Non-unit desc_global L2 norm: {l2}"));
            }
            let nav_confidence = compute_nav_confidence(
                img.width(),
                img.height(),
                l2,
                xfeat_loaded,
                xfeat_kpt_count,
                xfeat_mean_score,
                lighterglue_loaded,
                lg_matches,
                lg_mean_prob,
                nav_optical_gate,
            );
            // Step Pure-Rust W=7 Bayesian SequenceVprFilter (fail-closed zero vector on blind/degraded frames)
            let seq_dec = if nav_confidence > 0.0 {
                let step_i = self.nav_corridor_step;
                let cur_tile = NAV_CORRIDOR_SERPENTINE[step_i % NAV_CORRIDOR_SERPENTINE.len()];
                let delta_ne = if step_i == 0 {
                    [0.0, 0.0]
                } else {
                    let prev_tile =
                        NAV_CORRIDOR_SERPENTINE[(step_i - 1) % NAV_CORRIDOR_SERPENTINE.len()];
                    [
                        self.nav_corridor_coords[cur_tile][0]
                            - self.nav_corridor_coords[prev_tile][0],
                        self.nav_corridor_coords[cur_tile][1]
                            - self.nav_corridor_coords[prev_tile][1],
                    ]
                };
                self.nav_corridor_gallery[cur_tile].copy_from_slice(emb);
                let mut sims = vec![0.0_f64; self.nav_corridor_gallery.len()];
                for (t_idx, ref_emb) in self.nav_corridor_gallery.iter().enumerate() {
                    let mut dot = 0.0_f64;
                    for d in 0..512 {
                        dot += (emb[d] as f64) * (ref_emb[d] as f64);
                    }
                    sims[t_idx] = dot.clamp(-1.0, 1.0);
                }
                self.nav_corridor_step = self.nav_corridor_step.saturating_add(1);
                self.seq_vpr.step(&sims, delta_ne, 20.0, None, None)
            } else {
                let zeros = vec![0.0_f64; self.seq_vpr.num_tiles()];
                self.seq_vpr.step(&zeros, [0.0, 0.0], 20.0, None, None)
            };
            let seq_psr_sanitized = sanitize_json_f64(seq_dec.psr_result.psr_cal, 0.0);
            let seq_post_sanitized = sanitize_json_f64(seq_dec.posterior_prob_top1, 0.0);
            self.last_nav_integrity = NavIntegritySnapshot {
                active_bank: cusum_out.active_bank,
                gnss_quarantined: cusum_out.gnss_quarantined,
                spoof_alarm: spoof_detected,
                vertical_spoof_alarm: cusum_out.vertical_spoof_alarm,
                max_cusum_stat: cusum_max_sanitized,
                vertical_cusum_stat: cusum_vert_sanitized,
                cusum_h_threshold: CUSUM_H_LOCKED,
                dominant_arm: cusum_out.dominant_arm,
                nominal_streak: cusum_out.nominal_streak,
                seq_vpr_window_len: seq_dec.window_len,
                seq_vpr_admitted: seq_dec.admitted,
                seq_vpr_top1_idx: seq_dec.final_top1_idx,
                seq_vpr_posterior_prob: seq_post_sanitized,
                seq_vpr_psr_cal: seq_psr_sanitized,
                seq_vpr_reason_codes: seq_dec.reason_codes.clone(),
            };
            let latency_ms = round2(t0.elapsed().as_secs_f64() * 1000.0);
            self.last_channel_latencies_ms
                .insert("crossview_nav".to_string(), latency_ms);
            let summary = if spoof_detected {
                if !gps_offset_valid {
                    format!(
                        "RUST-ORT NAV-V5 FAZ-B CUSUM SPOOF NON-FINITE | XF {xfeat_kpt_count} · LG {lg_matches} | {ort_infer_ms:.1}+{xfeat_ort_ms:.1}+{lg_ort_ms:.1}ms"
                    )
                } else {
                    format!(
                        "RUST-ORT NAV-V5 FAZ-B CUSUM SPOOF +{gps_offset_sanitized:.0}m (S={cusum_max_sanitized:.1}) | XF {xfeat_kpt_count} · LG {lg_matches} | {ort_infer_ms:.1}+{xfeat_ort_ms:.1}+{lg_ort_ms:.1}ms"
                    )
                }
            } else {
                format!(
                    "RUST-ORT NAV-V5 FAZ-B LOCK (W={}) | 512D · XF {xfeat_kpt_count} · LG {lg_matches} | {ort_infer_ms:.1}+{xfeat_ort_ms:.1}+{lg_ort_ms:.1}ms",
                    seq_dec.window_len
                )
            };
            let (image_b64, annotated_jpeg_b64) = if return_jpeg {
                let mut vis = image::imageops::resize(img, 384, 256, FilterType::Triangle);
                for kp in xfeat_kpts.iter().take(40) {
                    let px = (kp.x / (XFEAT_IN_W as f32) * 384.0).round() as i32;
                    let py = (kp.y / (XFEAT_IN_H as f32) * 256.0).round() as i32;
                    draw_dot(&mut vis, px, py, 2, Rgb([34, 211, 238]));
                }
                stamp_bottom_sim_only_strip(&mut vis);
                let b64 = encode_rgb_jpeg_b64(&vis, 84)?;
                (None, Some(format!("data:image/jpeg;base64,{b64}")))
            } else {
                (None, None)
            };
            return Ok(NativeChannelPayload {
                channel: "crossview_nav".to_string(),
                model: "XVS-Nav v5.0 Faz-B + XFeat + LighterGlue (Rust ORT)".to_string(),
                backend: "ort-load-dynamic".to_string(),
                real_weights: true,
                checkpoint_sha256: sha,
                latency_ms,
                confidence: nav_confidence,
                summary,
                annotated_jpeg_b64,
                image_b64,
                metrics: serde_json::json!({
                    "status": "READY_RUST_ORT_V5",
                    "embedding_dim": emb.len(),
                    "l2_norm": (l2 * 1_000_000.0).round() / 1_000_000.0,
                    "ort_infer_ms": ort_infer_ms,
                    "xfeat_loaded": xfeat_loaded,
                    "xfeat_sha256": xfeat_sha.as_deref().unwrap_or(""),
                    "xfeat_keypoints_count": xfeat_kpt_count,
                    "xfeat_ort_ms": xfeat_ort_ms,
                    "xfeat_mean_score": xfeat_mean_score,
                    "lighterglue_loaded": lighterglue_loaded,
                    "lighterglue_sha256": lighterglue_sha.as_deref().unwrap_or(""),
                    "lighterglue_matches_count": lg_matches,
                    "lighterglue_mean_prob": lg_mean_prob,
                    "lighterglue_ort_ms": lg_ort_ms,
                    "lighterglue_mode": lg_mode,
                    "lighterglue_scores_shape": [1, LIGHTERGLUE_STATIC_K, LIGHTERGLUE_STATIC_K],
                    "nav_optical_gate": round4(nav_optical_gate),
                    "gps_offset_m": gps_offset_sanitized,
                    "gps_offset_non_finite": !gps_offset_valid,
                    "spoof_detected": spoof_detected,
                    "cusum_active_bank": cusum_out.active_bank,
                    "cusum_gnss_quarantined": cusum_out.gnss_quarantined,
                    "cusum_max_stat": cusum_max_sanitized,
                    "cusum_vertical_stat": cusum_vert_sanitized,
                    "cusum_h_threshold": CUSUM_H_LOCKED,
                    "cusum_dominant_arm": cusum_out.dominant_arm,
                    "cusum_nominal_streak": cusum_out.nominal_streak,
                    "cusum_vertical_spoof_alarm": cusum_out.vertical_spoof_alarm,
                    "seq_vpr_window_len": seq_dec.window_len,
                    "seq_vpr_admitted": seq_dec.admitted,
                    "seq_vpr_top1_idx": seq_dec.final_top1_idx,
                    "seq_vpr_posterior_prob": seq_post_sanitized,
                    "seq_vpr_psr_cal": seq_psr_sanitized,
                    "seq_vpr_reason_codes": seq_dec.reason_codes,
                }),
                non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
                actuator_authority: ACTUATOR_AUTHORITY,
                watermark: NON_FLIGHT_WATERMARK,
            });
        }

        let zeros = vec![0.0_f64; self.seq_vpr.num_tiles()];
        let seq_dec = self.seq_vpr.step(&zeros, [0.0, 0.0], 20.0, None, None);
        self.last_nav_integrity = NavIntegritySnapshot {
            active_bank: cusum_out.active_bank,
            gnss_quarantined: cusum_out.gnss_quarantined,
            spoof_alarm: spoof_detected,
            vertical_spoof_alarm: cusum_out.vertical_spoof_alarm,
            max_cusum_stat: cusum_max_sanitized,
            vertical_cusum_stat: cusum_vert_sanitized,
            cusum_h_threshold: CUSUM_H_LOCKED,
            dominant_arm: cusum_out.dominant_arm,
            nominal_streak: cusum_out.nominal_streak,
            seq_vpr_window_len: seq_dec.window_len,
            seq_vpr_admitted: seq_dec.admitted,
            seq_vpr_top1_idx: seq_dec.final_top1_idx,
            seq_vpr_posterior_prob: 0.0,
            seq_vpr_psr_cal: 0.0,
            seq_vpr_reason_codes: seq_dec.reason_codes.clone(),
        };

        // Clean, honest graceful status if XVS-Nav v5.0 ONNX is absent or rejected
        let latency_ms = round2(t0.elapsed().as_secs_f64() * 1000.0);
        let (image_b64, annotated_jpeg_b64) = if return_jpeg {
            let mut vis = image::imageops::resize(img, 384, 256, FilterType::Triangle);
            for kp in xfeat_kpts.iter().take(40) {
                let px = (kp.x / (XFEAT_IN_W as f32) * 384.0).round() as i32;
                let py = (kp.y / (XFEAT_IN_H as f32) * 256.0).round() as i32;
                draw_dot(&mut vis, px, py, 2, Rgb([34, 211, 238]));
            }
            stamp_bottom_sim_only_strip(&mut vis);
            let b64 = encode_rgb_jpeg_b64(&vis, 84)?;
            (None, Some(format!("data:image/jpeg;base64,{b64}")))
        } else {
            (None, None)
        };
        let nav_status = self
            .models_meta
            .get("xvs_nav")
            .map(|m| m.status.clone())
            .unwrap_or_else(|| "AWAITING_V5_ONNX_HARVEST".to_string());
        Ok(NativeChannelPayload {
            channel: "crossview_nav".to_string(),
            model: "XVS-Nav v5.0 Faz-B (Plug-In Slot)".to_string(),
            backend: "ort-load-dynamic".to_string(),
            real_weights: false,
            checkpoint_sha256: None,
            latency_ms,
            confidence: 0.0,
            summary: format!("{nav_status} | XF {xfeat_kpt_count} · LG {lg_matches}"),
            annotated_jpeg_b64,
            image_b64,
            metrics: serde_json::json!({
                "status": nav_status,
                "expected_filename": "xvs_nav_v5_fazb_fp16.onnx",
                "xfeat_loaded": xfeat_loaded,
                "xfeat_sha256": xfeat_sha.as_deref().unwrap_or(""),
                "xfeat_keypoints_count": xfeat_kpt_count,
                "xfeat_ort_ms": xfeat_ort_ms,
                "xfeat_mean_score": xfeat_mean_score,
                "lighterglue_loaded": lighterglue_loaded,
                "lighterglue_sha256": lighterglue_sha.as_deref().unwrap_or(""),
                "lighterglue_matches_count": lg_matches,
                "lighterglue_mean_prob": lg_mean_prob,
                "lighterglue_ort_ms": lg_ort_ms,
                "lighterglue_mode": lg_mode,
                "lighterglue_scores_shape": [1, LIGHTERGLUE_STATIC_K, LIGHTERGLUE_STATIC_K],
                "nav_optical_gate": round4(nav_optical_gate),
                "gps_offset_m": gps_offset_sanitized,
                "gps_offset_non_finite": !gps_offset_valid,
                "spoof_detected": spoof_detected,
                "cusum_active_bank": cusum_out.active_bank,
                "cusum_gnss_quarantined": cusum_out.gnss_quarantined,
                "cusum_max_stat": cusum_max_sanitized,
                "cusum_vertical_stat": cusum_vert_sanitized,
                "cusum_h_threshold": CUSUM_H_LOCKED,
                "cusum_dominant_arm": cusum_out.dominant_arm,
                "cusum_nominal_streak": cusum_out.nominal_streak,
                "cusum_vertical_spoof_alarm": cusum_out.vertical_spoof_alarm,
                "seq_vpr_window_len": seq_dec.window_len,
                "seq_vpr_admitted": seq_dec.admitted,
                "seq_vpr_top1_idx": seq_dec.final_top1_idx,
                "seq_vpr_posterior_prob": 0.0,
                "seq_vpr_psr_cal": 0.0,
                "seq_vpr_reason_codes": seq_dec.reason_codes,
            }),
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
        })
    }

    pub fn run_demo_tick(
        &mut self,
        scenario: &str,
        frame: u32,
        include_images: bool,
    ) -> Result<NativeDemoTickPayload, String> {
        let rta = self.step_rta(&build_demo_rta_input(scenario));
        self.run_demo_tick_with_rta(scenario, frame, include_images, rta)
    }

    pub fn run_demo_tick_with_rta(
        &mut self,
        scenario: &str,
        frame: u32,
        include_images: bool,
        rta: RtaReport,
    ) -> Result<NativeDemoTickPayload, String> {
        let t0 = Instant::now();

        // Advance monotonic camera & IMU clock (100 ms / 10 Hz cadence, 0.5 ms hardware FSYNC nominal skew)
        self.sim_clock_ms += 100.0;
        self.sensor_health
            .observe_timestamps(self.sim_clock_ms, self.sim_clock_ms - 0.5);

        self.sync_external_rta_state(rta.mode_id, rta.inhibit_avoidance_turns);

        // Step live xvs-multiship-coord StatefulMeshDaaGuard (DR-22 F4)
        self.d2_state.step_multiship_mesh_guard();

        // Discover real bundled JPEGs via LRU-cached `Arc<RgbImage>` (zero disk/decode per tick),
        // falling back to synthetic frames ONLY if media is absent!
        let (sky_img, sky_src, sky_file) = match self.media_lib.pick_eo_sky_arc(frame) {
            Some((img, rel, _)) => (img, "real_bundled_jpeg", Some(rel)),
            None => (
                Arc::new(build_synthetic_sky_frame(frame)),
                "synthetic_fallback",
                None,
            ),
        };
        let (th_eo_img, th_lwir_opt, th_src, th_file, q_eo, q_lwir, fog) =
            match self.media_lib.pick_thermal_for_scenario_arc(scenario, frame) {
                Some(df) => {
                    let (qe, ql, fg) = if scenario == "thermal_fog" {
                        (
                            df.q_eo_expected.unwrap_or(0.08),
                            df.q_lwir_expected.unwrap_or(0.96),
                            0.0_f32,
                        )
                    } else {
                        (
                            df.q_eo_expected.unwrap_or(0.88),
                            df.q_lwir_expected.unwrap_or(0.95),
                            0.0_f32,
                        )
                    };
                    (
                        df.eo,
                        df.lwir,
                        "real_bundled_jpeg",
                        Some(df.rel_file),
                        qe,
                        ql,
                        fg,
                    )
                }
                None => {
                    let (qe, ql, fg) = if scenario == "thermal_fog" {
                        (0.08_f32, 0.96_f32, 0.92_f32)
                    } else {
                        (0.88_f32, 0.95_f32, 0.0_f32)
                    };
                    (
                        Arc::new(build_synthetic_runway_frame_at(frame)),
                        None,
                        "synthetic_fallback",
                        None,
                        qe,
                        ql,
                        fg,
                    )
                }
            };

        // Fast rate group (10 Hz, every tick): XVS-D2 + XVS-ThermalNet + MultiShipCoord + SensorHealth
        let mut d2 = self.infer_d2_rgb(&sky_img, include_images)?;
        let mut thermal = self.infer_thermal_pair_rgb(
            &th_eo_img,
            th_lwir_opt.as_deref(),
            q_eo,
            q_lwir,
            fog,
            include_images,
        )?;
        attach_media_provenance(
            &mut d2,
            sky_src,
            sky_file.as_deref(),
            frame,
            self.media_lib.eo_sky.len(),
        );
        attach_media_provenance(
            &mut thermal,
            th_src,
            th_file.as_deref(),
            frame,
            self.media_lib.thermal_dve.len(),
        );

        // Slow rate group (1-2 Hz, every 5th tick or on scenario/image-mode change): Land + Nav + XFeat + LighterGlue
        let scenario_changed = self.last_slow_scenario.as_deref() != Some(scenario);
        let need_images_refresh = include_images
            && (self
                .last_slow_land
                .as_ref()
                .map(|p| p.annotated_jpeg_b64.is_none())
                .unwrap_or(true)
                || self
                    .last_slow_nav
                    .as_ref()
                    .map(|p| p.annotated_jpeg_b64.is_none())
                    .unwrap_or(true));
        let run_slow_now = frame.is_multiple_of(DEMO_SLOW_RATE_GROUP_STRIDE)
            || scenario_changed
            || need_images_refresh
            || self.last_slow_land.is_none()
            || self.last_slow_nav.is_none();

        let (land, nav) = if run_slow_now {
            let (land_img, land_src, land_file) = match self.media_lib.pick_runway_land_arc(frame) {
                Some((img, rel, _)) => (img, "real_bundled_jpeg", Some(rel)),
                None => (
                    Arc::new(build_synthetic_runway_frame_at(frame)),
                    "synthetic_fallback",
                    None,
                ),
            };
            let (nav_img, nav_sat_ref, nav_sat_path, nav_src, nav_file) =
                match self.media_lib.pick_nav_pair_arc(frame) {
                    Some(df) => (
                        df.aerial,
                        df.sat_ref,
                        df.sat_ref_path,
                        "real_bundled_jpeg",
                        Some(df.rel_file),
                    ),
                    None => (
                        Arc::new(build_synthetic_runway_frame_at(frame)),
                        None,
                        None,
                        "synthetic_fallback",
                        None,
                    ),
                };
            let gps_off = if scenario == "gps_spoof" { 165.0 } else { 0.0 };

            let mut fresh_land = self.infer_land_rgb(&land_img, include_images)?;
            let mut fresh_nav = self.infer_nav_pair_rgb_with_key(
                &nav_img,
                nav_sat_ref.as_deref(),
                nav_sat_path.as_deref(),
                gps_off,
                include_images,
            )?;
            attach_media_provenance(
                &mut fresh_land,
                land_src,
                land_file.as_deref(),
                frame,
                self.media_lib.runway_land.len(),
            );
            attach_media_provenance(
                &mut fresh_nav,
                nav_src,
                nav_file.as_deref(),
                frame,
                self.media_lib.crossview_nav.len(),
            );
            self.last_slow_scenario = Some(scenario.to_string());
            self.last_slow_land = Some(fresh_land.clone());
            self.last_slow_nav = Some(fresh_nav.clone());
            (fresh_land, fresh_nav)
        } else {
            let mut cached_land = self
                .last_slow_land
                .clone()
                .expect("cached land present");
            let mut cached_nav = self
                .last_slow_nav
                .clone()
                .expect("cached nav present");
            if !include_images {
                cached_land.annotated_jpeg_b64 = None;
                cached_land.image_b64 = None;
                cached_nav.annotated_jpeg_b64 = None;
                cached_nav.image_b64 = None;
            }
            (cached_land, cached_nav)
        };

        let mut channels = BTreeMap::new();
        channels.insert("eo_sky".to_string(), d2);
        channels.insert("runway_land".to_string(), land);
        channels.insert("thermal_dve".to_string(), thermal);
        channels.insert("crossview_nav".to_string(), nav);

        let total_ms = round2(t0.elapsed().as_secs_f64() * 1000.0);
        let sensor_health = self.sensor_health();
        Ok(NativeDemoTickPayload {
            scenario: scenario.to_string(),
            frame,
            transport: "tauri-ipc-rust-ort",
            total_latency_ms: total_ms,
            rta,
            sensor_health,
            channels,
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
        })
    }

    /// Run sequential Pure-Rust `ort` inference over real frames from `media/` or `data/studio_media/`
    /// so Video Lab works 100% standalone inside `openxvs-studio.exe` with ZERO Python!
    pub fn run_video_sequence_tick(
        &mut self,
        channel: &str,
        frame_idx: u32,
    ) -> Result<NativeChannelPayload, String> {
        self.sim_clock_ms += 100.0;
        self.sensor_health
            .observe_timestamps(self.sim_clock_ms, self.sim_clock_ms - 0.5);

        match channel.trim().to_ascii_lowercase().as_str() {
            "d2" | "eo_sky" | "xvs_d2" => {
                self.d2_state.step_multiship_mesh_guard();
                let (img, src, file, total) = match self.media_lib.pick_eo_sky_arc(frame_idx) {
                    Some((im, rel, tot)) => (im, "real_bundled_jpeg", Some(rel), tot),
                    None => (
                        Arc::new(build_synthetic_sky_frame(frame_idx)),
                        "synthetic_fallback",
                        None,
                        60,
                    ),
                };
                let mut out = self.infer_d2_rgb(&img, true)?;
                attach_media_provenance(&mut out, src, file.as_deref(), frame_idx, total);
                Ok(out)
            }
            "land" | "runway_land" | "xvs_land" => {
                let (img, src, file, total) = match self.media_lib.pick_runway_land_arc(frame_idx) {
                    Some((im, rel, tot)) => (im, "real_bundled_jpeg", Some(rel), tot),
                    None => (
                        Arc::new(build_synthetic_runway_frame_at(frame_idx)),
                        "synthetic_fallback",
                        None,
                        36,
                    ),
                };
                let mut out = self.infer_land_rgb(&img, true)?;
                attach_media_provenance(&mut out, src, file.as_deref(), frame_idx, total);
                Ok(out)
            }
            "thermal" | "thermal_dve" | "xvs_thermal" => {
                let (eo, lwir, src, file, total, q_eo, q_lwir, fog) =
                    match self.media_lib.pick_thermal_seq_arc(frame_idx) {
                        Some(df) => (
                            df.eo,
                            df.lwir,
                            "real_bundled_jpeg",
                            Some(df.rel_file),
                            df.total_frames,
                            df.q_eo_expected.unwrap_or(0.88),
                            df.q_lwir_expected.unwrap_or(0.95),
                            0.0_f32,
                        ),
                        None => (
                            Arc::new(build_synthetic_runway_frame_at(frame_idx)),
                            None,
                            "synthetic_fallback",
                            None,
                            24,
                            0.88_f32,
                            0.95_f32,
                            0.0_f32,
                        ),
                    };
                let mut out =
                    self.infer_thermal_pair_rgb(&eo, lwir.as_deref(), q_eo, q_lwir, fog, true)?;
                attach_media_provenance(&mut out, src, file.as_deref(), frame_idx, total);
                Ok(out)
            }
            "nav" | "crossview_nav" | "xvs_nav" => {
                let (img, sat_ref, sat_path, src, file, total, gps_off) =
                    match self.media_lib.pick_nav_pair_arc(frame_idx) {
                        Some(df) => (
                            df.aerial,
                            df.sat_ref,
                            df.sat_ref_path,
                            "real_bundled_jpeg",
                            Some(df.rel_file),
                            df.total_frames,
                            df.spoof_offset_m.unwrap_or(0.0),
                        ),
                        None => (
                            Arc::new(build_synthetic_runway_frame_at(frame_idx)),
                            None,
                            None,
                            "synthetic_fallback",
                            None,
                            24,
                            if frame_idx >= 12 { 165.0 } else { 0.0 },
                        ),
                    };
                let mut out = self.infer_nav_pair_rgb_with_key(
                    &img,
                    sat_ref.as_deref(),
                    sat_path.as_deref(),
                    gps_off,
                    true,
                )?;
                attach_media_provenance(&mut out, src, file.as_deref(), frame_idx, total);
                Ok(out)
            }
            other => Err(format!(
                "Unsupported video sequence channel '{other}'; expected d2, land, thermal, or nav"
            )),
        }
    }

    /// 1-click standalone evaluation snapshot report with all 5+1 ONNX SHA-256s, live latencies,
    /// `xvs-multiship-coord` F1/F3/F4 status, `SensorHealthMonitor`, and immutable `NOT FOR FLIGHT` stamp.
    pub fn export_eval_report(&mut self) -> Result<NativeEvalReportPayload, String> {
        let self_test = self.run_native_self_test();
        let sensor_health = self.sensor_health();
        let health = self.health();

        let report_doc = serde_json::json!({
            "schema": "openxvs.studio_eval_report/5.0",
            "product": "OpenXVS Studio v5.0",
            "version": STUDIO_VERSION,
            "transport": health.transport,
            "non_flight_sim_only": NON_FLIGHT_SIM_ONLY,
            "actuator_authority": ACTUATOR_AUTHORITY,
            "watermark": NON_FLIGHT_WATERMARK,
            "eu_pld_notice": EU_PLD_ART4_7_LOCKOUT_MSG,
            "odd_yaml_sha256": ODD_YAML_SHA256,
            "kani_proofs": health.kani_proofs,
            "ort_loaded": self.ort_loaded,
            "ort_dylib_label": self.ort_dylib_label,
            "models": self.models_meta,
            "last_channel_latencies_ms": self.last_channel_latencies_ms,
            "sensor_health": sensor_health,
            "multiship_coord": {
                "enabled": true,
                "f1_far91113_active": self.d2_state.f1_far91113_active,
                "f4_ghost_hysteresis_active": self.d2_state.f4_ghost_hysteresis_active,
                "ring0_authority_inhibited": self.d2_state.ring0_inhibited,
                "last_swerve_role": swerve_role_label(self.d2_state.coord_active_range.last_role()),
                "mesh_sticky_suppressed_total": self.d2_state.mesh_sticky_suppressed_total,
                "mesh_stale_rejected_total": self.d2_state.mesh_stale_rejected_total,
                "mesh_replay_rejected_total": self.d2_state.mesh_replay_rejected_total,
            },
            "media_library": {
                "available": self.media_lib.is_available(),
                "channel_frame_counts": self.media_lib.channel_counts(),
            },
            "self_test": self_test,
        });

        let json_content = serde_json::to_string_pretty(&report_doc)
            .map_err(|e| format!("Failed to serialize eval report: {e}"))?;
        let sha256 = sha256_bytes_hex(json_content.as_bytes());
        let filename = format!("openxvs_v5_eval_report_{}.json", &sha256[..8]);

        Ok(NativeEvalReportPayload {
            filename,
            json_content,
            sha256,
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
        })
    }

    pub fn analyze_image_b64(
        &mut self,
        channel: &str,
        image_b64: &str,
        q_eo: f32,
        q_lwir: f32,
        fog_density: f32,
        gps_offset_m: f64,
    ) -> Result<NativeChannelPayload, String> {
        let stripped = match image_b64.split_once(',') {
            Some((prefix, rest)) if prefix.starts_with("data:") => rest,
            _ => image_b64,
        }
        .trim();
        if stripped.len() > MAX_ANALYZE_IMAGE_B64_LEN {
            return Err(format!(
                "Image base64 payload exceeds 16 MiB limit ({} chars > {MAX_ANALYZE_IMAGE_B64_LEN})",
                stripped.len()
            ));
        }
        let raw = BASE64_STD
            .decode(stripped)
            .map_err(|e| format!("Invalid base64 image payload: {e}"))?;
        if raw.len() > MAX_ANALYZE_IMAGE_BYTES {
            return Err(format!(
                "Decoded image payload exceeds 16 MiB limit ({} bytes > {MAX_ANALYZE_IMAGE_BYTES})",
                raw.len()
            ));
        }

        // Pre-decode header check + strict allocation limits defeat PNG/JPEG decompression bombs
        let mut limits = Limits::default();
        limits.max_image_width = Some(MAX_ANALYZE_IMAGE_DIM);
        limits.max_image_height = Some(MAX_ANALYZE_IMAGE_DIM);
        limits.max_alloc = Some(64 * 1024 * 1024);

        let mut dim_reader = ImageReader::new(Cursor::new(&raw))
            .with_guessed_format()
            .map_err(|e| format!("Failed to inspect image header: {e}"))?;
        dim_reader.limits(limits.clone());
        let (w, h) = dim_reader
            .into_dimensions()
            .map_err(|e| format!("Failed to read JPEG/PNG dimensions: {e}"))?;
        if w == 0 || h == 0 || w > MAX_ANALYZE_IMAGE_DIM || h > MAX_ANALYZE_IMAGE_DIM {
            return Err(format!(
                "Image dimensions {w}x{h} outside allowed [1..={MAX_ANALYZE_IMAGE_DIM}]^2 range"
            ));
        }

        let mut reader = ImageReader::new(Cursor::new(&raw))
            .with_guessed_format()
            .map_err(|e| format!("Failed to inspect JPEG/PNG image bytes: {e}"))?;
        reader.limits(limits);
        let dyn_img = reader
            .decode()
            .map_err(|e| format!("Failed to decode JPEG/PNG image bytes: {e}"))?;
        let rgb = dyn_img.to_rgb8();

        match channel.trim().to_ascii_lowercase().as_str() {
            "d2" | "eo_sky" => self.infer_d2_rgb(&rgb, true),
            "land" | "runway_land" => self.infer_land_rgb(&rgb, true),
            "thermal" | "thermal_dve" => {
                self.infer_thermal_rgb(&rgb, q_eo, q_lwir, fog_density, true)
            }
            "nav" | "crossview_nav" => self.infer_nav_rgb(&rgb, gps_offset_m, true),
            other => Err(format!(
                "Unsupported channel '{other}'; expected one of d2, land, thermal, nav"
            )),
        }
    }

    pub fn hil_status(&self) -> NativeHilStatusPayload {
        NativeHilStatusPayload {
            mode: "locked_non_flight_sim_only",
            bound_host: "none",
            rx_port: 0,
            tx_port: 0,
            rx_packets: 0,
            tx_packets: 0,
            tx_enabled: false,
            actuator_authority: ACTUATOR_AUTHORITY,
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            eu_pld_notice: EU_PLD_ART4_7_LOCKOUT_MSG,
            watermark: NON_FLIGHT_WATERMARK,
            gpl_free: true,
            last_telemetry_ned: None,
            last_loopback: self.last_loopback.clone(),
        }
    }

    #[cfg(feature = "non_flight_sim_only")]
    pub fn hil_connect_blocked(&self, _mode: &str) -> Result<NativeHilStatusPayload, String> {
        Err(EU_PLD_ART4_7_LOCKOUT_MSG.to_string())
    }

    pub fn run_native_self_test(&mut self) -> NativeLoopbackReport {
        let t0 = Instant::now();

        // 1. CoordinatedActiveRange + frozen 64-byte IEEE-802.3 CRC32 wire + all-64-byte tamper test
        let mut coord = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
        let own = OwnshipStateNed {
            v_ned: [35.0, 0.0, 0.0],
            a_ned: [0.0, 0.0, 0.0],
            heading_rad: 0.0,
            tas_mps: 35.0,
            v_stall_mps: 14.5,
        };
        let meas = ActiveRangeMeasurement {
            alpha_rad: 0.0,
            beta_rad: 0.0,
            size_px: 14.0,
            size_rate_px_s: 2.0,
            contrast_ratio: 0.65,
            sky_beta_ext: 2.0e-4,
            daytime_clear: true,
            track_age: 6,
            dt_s: 0.1,
        };
        let pkt: ActiveRangeTelemetry = coord.step(&own, &meas, None, Ring0AuthorityGate::NOMINAL);
        let wire = encode_wire(&pkt);
        let roundtrip_ok = decode_wire(&wire) == Some(pkt);
        let mut tamper_rejected = 0u32;
        for byte_idx in 0..ACTIVE_RANGE_WIRE_BYTES {
            let mut bad = wire;
            bad[byte_idx] ^= 0x01;
            if decode_wire(&bad).is_none() {
                tamper_rejected += 1;
            }
        }
        let ar_pass = roundtrip_ok && tamper_rejected == (ACTIVE_RANGE_WIRE_BYTES as u32);

        // 2. Frozen `xvs-rta-core` Simplex P4 -> P0 immediate preemption + staged P0 -> P1 -> P4 recovery
        let mut rta_test = SimplexRta::new();
        let nom = rta_test.step(&RtaTickInput::default());
        let fault_inp = RtaTickInput {
            quantize_fault: true,
            ..RtaTickInput::default()
        };
        let p0 = rta_test.step(&fault_inp);
        let mut after_p0_recovery = p0;
        for _ in 0..P0_RECOVERY_TICKS {
            after_p0_recovery = rta_test.step(&RtaTickInput::default());
        }
        let mut after_dwell = after_p0_recovery;
        for _ in 0..DWELL_TICKS {
            after_dwell = rta_test.step(&RtaTickInput::default());
        }
        let rta_pass = nom.mode == SimplexMode::NominalMl
            && p0.mode == SimplexMode::InvalidInhibit
            && p0.inhibit_avoidance_turns
            && after_p0_recovery.mode == SimplexMode::FallbackD0Safe
            && after_dwell.mode == SimplexMode::NominalMl;

        // 3. `xvs-multiship-coord` F1 (FAR 91.113 right turn) + F3 (Ring-0 P0 zero bank) + F4 (200ms staleness + sticky 3->2 ghost suppression)
        let f1_ok = pkt.cmd_bank_rad >= 0.0
            && coord.last_role() == SwerveCoordRole::OpticalFar91113Right;
        let close_meas = ActiveRangeMeasurement {
            alpha_rad: 0.0,
            beta_rad: 0.0,
            size_px: 140.0,
            size_rate_px_s: 45.0,
            contrast_ratio: 0.9,
            sky_beta_ext: 2.0e-4,
            daytime_clear: true,
            track_age: 15,
            dt_s: 0.1,
        };
        let p0_pkt = coord.step(
            &own,
            &close_meas,
            None,
            Ring0AuthorityGate {
                rta_inhibit_avoidance_turns: true,
                hal_out_mode_wire: 4,
            },
        );
        let hal_p0_pkt = coord.step(
            &own,
            &close_meas,
            None,
            Ring0AuthorityGate {
                rta_inhibit_avoidance_turns: false,
                hal_out_mode_wire: 0,
            },
        );
        let f3_ok = p0_pkt.cmd_bank_rad == 0.0
            && p0_pkt.mode == ActiveRangeMode::NominalHold as u8
            && hal_p0_pkt.cmd_bank_rad == 0.0
            && hal_p0_pkt.mode == ActiveRangeMode::NominalHold as u8;

        let base_ecef = [6_378_137.0_f64, 0.0, 0.0];
        let p_s1 = [base_ecef[0], 0.0, 0.0];
        let p_s2 = [base_ecef[0], 1000.0, 0.0];
        let p_s3 = [base_ecef[0] + 1000.0, 500.0, 0.0];
        let t1 = [base_ecef[0] + 500.0, 200.0, 0.0];
        let t2 = [base_ecef[0] + 500.0, 800.0, 0.0];
        let mut guard = StatefulMeshDaaGuard::new();
        let ts3 = 1_700_000_000_000u64;
        let obs3 = [
            make_ecef_bearing_obs(1, 1, p_s1, t1, ts3),
            make_ecef_bearing_obs(1, 2, p_s1, t2, ts3),
            make_ecef_bearing_obs(2, 1, p_s2, t1, ts3),
            make_ecef_bearing_obs(2, 2, p_s2, t2, ts3),
            make_ecef_bearing_obs(3, 1, p_s3, t1, ts3),
            make_ecef_bearing_obs(3, 2, p_s3, t2, ts3),
        ];
        let g3 = guard.fuse_guarded(&obs3, ts3 + 20_000);
        let ts2 = ts3 + 200_000;
        let obs2 = [
            make_ecef_bearing_obs(1, 1, p_s1, t1, ts2),
            make_ecef_bearing_obs(1, 2, p_s1, t2, ts2),
            make_ecef_bearing_obs(2, 1, p_s2, t1, ts2),
            make_ecef_bearing_obs(2, 2, p_s2, t2, ts2),
        ];
        let g2 = guard.fuse_guarded(&obs2, ts2 + 20_000);
        let stale_obs = [make_ecef_bearing_obs(1, 1, p_s1, t1, ts2 - 250_000)];
        let g_stale = guard.fuse_guarded(&stale_obs, ts2 + 50_000);
        let f4_ok = g3.num_fused == 2
            && g2.num_fused == 2
            && g2.sticky_triple_ray_active
            && g2.sticky_ghosts_suppressed >= 2
            && g_stale.stale_obs_rejected == 1;
        let multiship_pass = f1_ok && f3_ok && f4_ok;

        // 4. SensorHealthMonitor: LWIR NUC freeze + hardware FFC + p95-p05 + 4x4/12-sector + dual-tier FSYNC + SLERP + Optics + 3-of-5 TBC
        let mut shm = SensorHealthMonitor::new();
        let flat_lwir = vec![128u8; 160 * 128];
        for _ in 0..4 {
            shm.observe_lwir_frame(&flat_lwir);
        }
        let nuc_detected = shm.lwir_nuc_frozen && shm.lwir_freeze_ticks >= 3;
        let (_, q_lwir_after_nuc) = shm.compute_effective_qualities(0.95, 0.95);
        let nuc_mode = select_thermal_simplex_mode(0.95, q_lwir_after_nuc, 0.15);
        let glare_img = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]));
        let eo_glare = SensorHealthMonitor::evaluate_eo_optical_quality(&glare_img);
        let (_, _, skew_fault) = shm.observe_timestamps(1000.0, 750.0);

        let pin = PinholeIntrinsics::new(
            1280.0, 1280.0, 640.0, 360.0, -0.24, 0.08, 0.0012, -0.0009, -0.015,
        );
        let bc_err_px = pin
            .and_then(|cam| {
                let pt = [1120.0, 620.0];
                let d = cam.distort_pixel(pt)?;
                let u = cam.undistort_pixel_exact(d)?;
                Ok((u[0] - pt[0]).hypot(u[1] - pt[1]))
            })
            .unwrap_or(9999.0);
        let kb = KannalaBrandtFisheye::new(
            420.0, 420.0, 640.0, 360.0, -0.025, 0.008, -0.002, 0.0004,
        );
        let kb_err_px = kb
            .and_then(|cam| {
                // 67.5 deg half-angle => 135 deg wide-angle FOV
                let theta = 67.5_f64.to_radians();
                let ray = [theta.sin() * 0.8, theta.sin() * 0.6, theta.cos()];
                let d = cam.project_ray_rdf(ray)?;
                let u_ray = cam.undistort_to_unit_ray_exact(d)?;
                let re = cam.project_ray_rdf(u_ray)?;
                Ok((re[0] - d[0]).hypot(re[1] - d[1]))
            })
            .unwrap_or(9999.0);
        let q_mid_ok = interpolate_imu_attitude_at_exposure_midpoint(
            1000.0,
            [1.0, 0.0, 0.0, 0.0],
            1010.0,
            [
                (0.25_f64 * core::f64::consts::FRAC_PI_2).cos(),
                0.0,
                0.0,
                (0.25_f64 * core::f64::consts::FRAC_PI_2).sin(),
            ],
            1002.0,
            6.0,
        )
        .map(|(q, tm)| (tm - 1005.0).abs() < 1e-12 && (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] - 1.0).abs() < 1e-12)
        .unwrap_or(false);
        let mut tbc = BearingTrackBeforeConfirm::new_3_of_5();
        let s1 = tbc.observe(Some((0.10, 0.02)));
        let s2 = tbc.observe(None);
        let s3 = tbc.observe(Some((0.105, 0.021)));
        let s4 = tbc.observe(None);
        let s5 = tbc.observe(Some((0.108, 0.022)));
        let tbc_ok = !s1.confirmed && !s2.confirmed && !s3.confirmed && !s4.confirmed && s5.confirmed;

        let sensor_health_pass = nuc_detected
            && q_lwir_after_nuc == 0.0
            && nuc_mode == ThermalSimplexMode::LwirDegradedEoPrimary
            && eo_glare.q_eo_optical == 0.0
            && eo_glare.degraded_sector_mask == 0x0FFF
            && skew_fault
            && bc_err_px < 1e-12
            && kb_err_px < 1e-12
            && q_mid_ok
            && tbc_ok;

        // 5. Compile-time non-flight lockout & System32 hijack guard
        let lock_pass = NON_FLIGHT_SIM_ONLY
            && !ACTUATOR_AUTHORITY
            && is_forbidden_system_ort_path(Path::new(r"C:\Windows\System32\onnxruntime.dll"));

        // 6. StudentTCalibrator exact factor + ORT load-dynamic status + pinned SHA-256 check
        let is_land_v52 = matches!(
            self.models_meta
                .get("xvs_land")
                .and_then(|m| m.sha256.as_deref()),
            Some(EXPECTED_LAND_V52_FP16_SHA256) | Some(EXPECTED_LAND_V52_FP32_SHA256)
        );
        let expected_scale = if is_land_v52 {
            V52_LAND_STUDENT_T_SCALE
        } else {
            WAVE_C_LAND_STUDENT_T_SCALE
        };
        let expected_factor = expected_scale * expected_scale * (6.0 / 4.0);
        let cal_pass = (self.land_calibrator.variance_factor() - expected_factor).abs() < 1e-12;
        let all_loaded_pins_ok = self.models_meta.values().all(|m| {
            if !m.loaded {
                return true;
            }
            m.sha256.is_some() && m.sha256 == m.expected_sha256
        });
        let ort_pass = self.ort_loaded && cal_pass && all_loaded_pins_ok;

        // 7. Pure-Rust Nav Integrity self-check (W=7 SequenceVprFilter blind/flat guard + DualBankEsEkfCusum ramp/vertical-NaN)
        let mut svpr_test = SequenceVprFilter::new(
            self.nav_corridor_coords.clone(),
            SequenceVprConfig::default(),
        )
        .expect("valid SequenceVprFilter");
        let mut svpr_w7_ok = true;
        for step_i in 0..7usize {
            let cur = NAV_CORRIDOR_SERPENTINE[step_i];
            let delta = if step_i == 0 {
                [0.0, 0.0]
            } else {
                let prev = NAV_CORRIDOR_SERPENTINE[step_i - 1];
                [
                    self.nav_corridor_coords[cur][0] - self.nav_corridor_coords[prev][0],
                    self.nav_corridor_coords[cur][1] - self.nav_corridor_coords[prev][1],
                ]
            };
            let mut sims = vec![0.15_f64; 16];
            sims[cur] = 0.88;
            let dec = svpr_test.step(&sims, delta, 20.0, None, None);
            svpr_w7_ok &= dec.admitted && dec.final_top1_idx == (cur as i64);
        }
        let svpr_zero_rejected = !svpr_test
            .step(&[0.0_f64; 16], [0.0, 150.0], 20.0, None, None)
            .admitted
            && svpr_test.window_len() == 7;
        let mut cusum_test =
            DualBankEsEkfCusum::new(DualBankConfig::default()).expect("valid DualBankEsEkfCusum");
        let ang = 22.5_f64.to_radians();
        let mut cusum_ramp_ttd = None;
        for step_k in 1..=30usize {
            let off = 2.0 * (step_k as f64);
            let out = cusum_test.step(&DualBankStepInput {
                gnss_ne_m: Some([off * ang.cos(), off * ang.sin()]),
                vis_ne_m: Some([0.0, 0.0]),
                vis_gate_admitted: true,
                ..DualBankStepInput::default()
            });
            if out.quarantine_triggered_this_epoch {
                cusum_ramp_ttd = Some(step_k);
                break;
            }
        }
        let mut cusum_reint_ep = None;
        for ep in 1..=80usize {
            let out = cusum_test.step(&DualBankStepInput {
                gnss_ne_m: Some([0.0, 0.0]),
                vis_ne_m: Some([0.0, 0.0]),
                vis_gate_admitted: true,
                ..DualBankStepInput::default()
            });
            if out.reintegrated_this_epoch {
                cusum_reint_ep = Some(ep);
                break;
            }
        }
        let cusum_vert_nan_ok = cusum_test
            .step(&DualBankStepInput {
                baro_down_m: -120.0,
                gnss_ne_m: Some([0.0, 0.0]),
                vis_ne_m: Some([0.0, 0.0]),
                vis_gate_admitted: true,
                gnss_down_m: Some(f64::NAN),
                ..DualBankStepInput::default()
            })
            .vertical_spoof_alarm;
        let nav_integrity_pass = svpr_w7_ok
            && svpr_zero_rejected
            && cusum_ramp_ttd.map(|t| t <= 12).unwrap_or(false)
            && cusum_reint_ep
                .map(|e| e >= CUSUM_REINTEGRATION_HYSTERESIS_EPOCHS)
                .unwrap_or(false)
            && cusum_vert_nan_ok;

        let checks = vec![
            NativeLoopbackCheck {
                name: "active_range_64b_wire_crc32".to_string(),
                pass: ar_pass,
                details: serde_json::json!({
                    "wire_bytes": ACTIVE_RANGE_WIRE_BYTES,
                    "tamper_cases_tested": ACTIVE_RANGE_WIRE_BYTES,
                    "tamper_cases_rejected": tamper_rejected,
                }),
            },
            NativeLoopbackCheck {
                name: "xvs_rta_core_simplex_preemption".to_string(),
                pass: rta_pass,
                details: serde_json::json!({
                    "odd_yaml_sha256": ODD_YAML_SHA256,
                    "p4_to_p0_immediate": p0.mode == SimplexMode::InvalidInhibit,
                    "p0_to_p1_recovery": after_p0_recovery.mode == SimplexMode::FallbackD0Safe,
                    "p1_to_p4_dwell": after_dwell.mode == SimplexMode::NominalMl,
                }),
            },
            NativeLoopbackCheck {
                name: "xvs_multiship_coord_f1_f3_f4".to_string(),
                pass: multiship_pass,
                details: serde_json::json!({
                    "f1_far91113_right_turn": f1_ok,
                    "f3_ring0_p0_zero_bank": f3_ok,
                    "f4_sticky_triple_ray_zero_ghosts": f4_ok,
                    "f4_sticky_ghosts_suppressed": g2.sticky_ghosts_suppressed,
                    "f4_stale_obs_rejected": g_stale.stale_obs_rejected,
                }),
            },
            NativeLoopbackCheck {
                name: "sensor_health_monitor_nuc_blur_skew".to_string(),
                pass: sensor_health_pass,
                details: serde_json::json!({
                    "lwir_nuc_freeze_detected": nuc_detected,
                    "q_lwir_after_nuc": q_lwir_after_nuc,
                    "nuc_simplex_mode": nuc_mode.as_str(),
                    "eo_glare_q_optical": eo_glare.q_eo_optical,
                    "eo_glare_degraded_sector_mask": eo_glare.degraded_sector_mask,
                    "timestamp_skew_250ms_rejected": skew_fault,
                    "brown_conrady_err_px": bc_err_px,
                    "kannala_brandt_err_px": kb_err_px,
                    "slerp_exposure_midpoint_ok": q_mid_ok,
                    "bearing_tbc_3_of_5_ok": tbc_ok,
                }),
            },
            NativeLoopbackCheck {
                name: "non_flight_sim_only_eu_pld_lock".to_string(),
                pass: lock_pass,
                details: serde_json::json!({
                    "non_flight_sim_only": NON_FLIGHT_SIM_ONLY,
                    "actuator_authority": ACTUATOR_AUTHORITY,
                    "system32_hijack_blocked": true,
                }),
            },
            NativeLoopbackCheck {
                name: "ort_load_dynamic_sha256_pins_and_student_t".to_string(),
                pass: ort_pass,
                details: serde_json::json!({
                    "ort_loaded": self.ort_loaded,
                    "all_loaded_pins_verified": all_loaded_pins_ok,
                    "student_t_factor": self.land_calibrator.variance_factor(),
                }),
            },
            NativeLoopbackCheck {
                name: "pure_rust_nav_integrity_sequence_vpr_and_dual_bank_cusum".to_string(),
                pass: nav_integrity_pass,
                details: serde_json::json!({
                    "seq_vpr_w7_nominal_ok": svpr_w7_ok,
                    "seq_vpr_blind_zero_rejected": svpr_zero_rejected,
                    "cusum_2mps_ramp_ttd_s": cusum_ramp_ttd,
                    "cusum_reintegration_epochs": cusum_reint_ep,
                    "cusum_vertical_nan_fail_closed": cusum_vert_nan_ok,
                }),
            },
        ];

        let all_pass = checks.iter().all(|c| c.pass);
        let report = NativeLoopbackReport {
            pass: all_pass,
            timestamp_utc: format_utc_now_iso8601(),
            elapsed_ms: round2(t0.elapsed().as_secs_f64() * 1000.0),
            packets_verified: (ACTIVE_RANGE_WIRE_BYTES as u32) + 64,
            tamper_rejected: tamper_rejected == (ACTIVE_RANGE_WIRE_BYTES as u32),
            gpl_free: true,
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
            checks,
        };
        self.last_loopback = Some(report.clone());
        report
    }
}

fn attach_media_provenance(
    payload: &mut NativeChannelPayload,
    source: &str,
    rel_file: Option<&str>,
    frame_idx: u32,
    total_frames: usize,
) {
    if let Some(obj) = payload.metrics.as_object_mut() {
        obj.insert("media_source".to_string(), serde_json::json!(source));
        obj.insert("media_file".to_string(), serde_json::json!(rel_file));
        obj.insert("frame_idx".to_string(), serde_json::json!(frame_idx));
        obj.insert("total_frames".to_string(), serde_json::json!(total_frames));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_time_non_flight_and_system32_hijack_guard() {
        const { assert!(NON_FLIGHT_SIM_ONLY) };
        const { assert!(!ACTUATOR_AUTHORITY) };
        assert!(is_forbidden_system_ort_path(Path::new(
            r"C:\Windows\System32\onnxruntime.dll"
        )));
        assert!(is_forbidden_system_ort_path(Path::new(
            r"C:\WINDOWS\SysWOW64\onnxruntime.dll"
        )));
        assert!(resolve_safe_ort_dylib_path_with_roots(
            Some(r"C:\Windows\System32\onnxruntime.dll"),
            None,
            None
        )
        .is_err());
    }

    #[test]
    fn test_student_t_calibrator_matches_wave_c_python_exact() {
        let cal = StudentTCalibrator::new(WAVE_C_LAND_STUDENT_T_NU, WAVE_C_LAND_STUDENT_T_SCALE)
            .expect("valid calibrator");
        let expected = WAVE_C_LAND_STUDENT_T_SCALE * WAVE_C_LAND_STUDENT_T_SCALE * (6.0 / 4.0);
        assert!((cal.variance_factor() - expected).abs() < 1e-12);
        let out = cal.calibrate_cov2x2([2.0, 0.1, 0.1, 3.0]);
        assert!((out[0] - 2.0 * expected).abs() < 1e-10);
        assert!((out[3] - 3.0 * expected).abs() < 1e-10);
    }

    #[test]
    fn test_sha256_fail_closed_pinning_rejects_tampered_models() {
        let tmp = std::env::temp_dir().join("openxvs_sha256_tamper_test");
        let _ = std::fs::create_dir_all(&tmp);
        let cases: &[(&str, &[(&str, &str)])] = &[
            ("xvs_d2_web_fp16.onnx", D2_ONNX_PINS),
            ("xvs_thermal_v52_fp16.onnx", THERMAL_ONNX_PINS),
            ("xvs_thermal_v52_fp32.onnx", THERMAL_ONNX_PINS),
            ("xvs_thermal_v45_fp32.onnx", THERMAL_ONNX_PINS),
            ("xvs_thermal_v45_fp16.onnx", THERMAL_ONNX_PINS),
            ("xvs_land_v52_fp16.onnx", LAND_ONNX_PINS),
            ("xvs_land_v52_fp32.onnx", LAND_ONNX_PINS),
            ("xvs_land_v45_fp16.onnx", LAND_ONNX_PINS),
            ("xvs_land_v45.onnx", LAND_ONNX_PINS),
            ("xvs_nav_v5_fazb_fp16.onnx", NAV_ONNX_PINS),
            ("xvs_nav_v5_fp16.onnx", NAV_ONNX_PINS),
            ("xfeat_320x256_fp16.onnx", XFEAT_ONNX_PINS),
            ("xfeat_320x256_fp32.onnx", XFEAT_ONNX_PINS),
            ("lighterglue_k256_fp16.onnx", LIGHTERGLUE_ONNX_PINS),
            ("lighterglue_k256_fp32.onnx", LIGHTERGLUE_ONNX_PINS),
        ];

        for &(fname, pins) in cases {
            let p = tmp.join(fname);
            std::fs::write(&p, b"TAMPERED_ONNX_PAYLOAD_BYTES").expect("write temp tamper file");
            assert!(
                verify_pinned_onnx_artifact(&p, pins).is_err(),
                "Expected fail-closed SHA256_MISMATCH_REJECTED for tampered {fname}"
            );
            let _ = std::fs::remove_file(&p);
        }

        // Unrecognized filename must also fail closed
        let unknown = tmp.join("rogue_model.onnx");
        std::fs::write(&unknown, b"ROGUE").expect("write rogue file");
        assert!(verify_pinned_onnx_artifact(&unknown, D2_ONNX_PINS).is_err());
        assert!(verify_pinned_onnx_artifact(&unknown, LIGHTERGLUE_ONNX_PINS).is_err());
        let _ = std::fs::remove_file(&unknown);
        let _ = std::fs::remove_dir(&tmp);
    }

    #[test]
    fn test_sensor_health_monitor_nuc_freeze_blur_glare_occlusion_and_skew() {
        let mut mon = SensorHealthMonitor::new();

        // 1. LWIR NUC shutter freeze detector (MAD < 0.25 for >= 3 consecutive ticks)
        let frame_a = vec![110u8; 160 * 128];
        let (f0, t0, _) = mon.observe_lwir_frame(&frame_a);
        assert!(!f0 && t0 == 0);
        let (f1, t1, mad1) = mon.observe_lwir_frame(&frame_a);
        assert!(!f1 && t1 == 1 && mad1 < 0.01);
        let (f2, t2, _) = mon.observe_lwir_frame(&frame_a);
        assert!(!f2 && t2 == 2);
        let (f3, t3, _) = mon.observe_lwir_frame(&frame_a);
        assert!(f3 && t3 == 3, "3rd consecutive frozen frame must trip NUC freeze");
        let (q_eo_eff, q_lwir_eff) = mon.compute_effective_qualities(0.92, 0.96);
        assert_eq!(q_lwir_eff, 0.0, "NUC freeze must force q_lwir_effective = 0.0");
        assert_eq!(
            select_thermal_simplex_mode(q_eo_eff, q_lwir_eff, 0.12),
            ThermalSimplexMode::LwirDegradedEoPrimary
        );
        // Recovery when fresh thermal motion resumes (MAD >= 0.25)
        let frame_b = vec![145u8; 160 * 128];
        let (f_rec, t_rec, mad_rec) = mon.observe_lwir_frame(&frame_b);
        assert!(!f_rec && t_rec == 0 && mad_rec >= 30.0);

        // 2. EO Blur / Glare / Occlusion quality gate
        let sharp = build_synthetic_runway_frame();
        let q_sharp = SensorHealthMonitor::evaluate_eo_optical_quality(&sharp);
        assert!(
            q_sharp.blur_score >= 50.0 && q_sharp.q_eo_optical >= 0.95,
            "Sharp frame must pass optical quality gate: {q_sharp:?}"
        );

        let blurred = RgbImage::from_pixel(160, 128, Rgb([128, 128, 128]));
        let q_blur = SensorHealthMonitor::evaluate_eo_optical_quality(&blurred);
        assert!(
            q_blur.blur_score < 1.0 && q_blur.q_eo_optical < 0.05,
            "Uniform blurred/fogged frame must degrade q_eo_optical: {q_blur:?}"
        );

        let glared = RgbImage::from_pixel(160, 128, Rgb([254, 255, 254]));
        let q_glare = SensorHealthMonitor::evaluate_eo_optical_quality(&glared);
        assert!(q_glare.glare_ratio > 0.99 && q_glare.q_eo_optical == 0.0);

        let occluded = RgbImage::from_pixel(160, 128, Rgb([1, 2, 1]));
        let q_occ = SensorHealthMonitor::evaluate_eo_optical_quality(&occluded);
        assert!(q_occ.occlusion_ratio > 0.99 && q_occ.q_eo_optical == 0.0);

        // 3. Timestamp monotonicity & IMU-Camera skew guard (>200 ms & backward jump)
        let (sk0, mono0, fault0) = mon.observe_timestamps(1000.0, 995.0);
        assert!((sk0 - 5.0).abs() < 1e-6 && mono0 && !fault0);
        let (sk_hi, mono_hi, fault_hi) = mon.observe_timestamps(1200.0, 950.0);
        assert!(sk_hi > 200.0 && mono_hi && fault_hi);
        let (qe_sk, ql_sk) = mon.compute_effective_qualities(0.95, 0.95);
        assert_eq!((qe_sk, ql_sk), (0.0, 0.0));
        // Backward non-monotonic timestamp replay
        let (_, mono_back, fault_back) = mon.observe_timestamps(1150.0, 1148.0);
        assert!(!mono_back && fault_back);
    }

    #[test]
    fn test_real_ort_load_dynamic_all_5plus1_onnx_models_ipc_and_rta() {
        let mut engine = RustStudioEngine::initialize();
        let health = engine.health();
        assert_eq!(health.version, STUDIO_VERSION);
        assert!(health.non_flight_sim_only);
        assert!(!health.actuator_authority);
        assert!(health.multiship_coord_enabled);
        assert_eq!(health.watermark, NON_FLIGHT_WATERMARK);
        if !health.ort_loaded {
            eprintln!(
                "[skip] test_real_ort_load_dynamic_all_5plus1_onnx_models_ipc_and_rta: ORT dylib absent in source-only checkout ({})",
                health.ort_dylib_label
            );
            return;
        }
        assert!(
            health.ort_loaded,
            "ORT load-dynamic must load local onnxruntime.dll cleanly: {}",
            health.ort_dylib_label
        );

        // ALL 5+1 ONNX models + frozen Rust ActiveRange/MultiShipCoord/RTA verified with exact SHA-256 pins!
        assert!(health.models["xvs_d2"].loaded, "XVS-D2 ONNX failed to load");
        assert_eq!(
            health.models["xvs_d2"].sha256.as_deref(),
            Some(EXPECTED_D2_ONNX_SHA256)
        );
        assert!(
            health.models["xvs_thermal"].loaded,
            "XVS-ThermalNet v5.2 ONNX failed to load"
        );
        assert_eq!(
            health.models["xvs_thermal"].sha256.as_deref(),
            Some(EXPECTED_THERMAL_V52_FP16_ONNX_SHA256)
        );
        assert!(
            health.models["xvs_land"].loaded,
            "XVS-Land v5.2 ONNX failed to load"
        );
        assert_eq!(
            health.models["xvs_land"].sha256.as_deref(),
            Some(EXPECTED_LAND_V52_FP16_SHA256),
            "Unexpected XVS-Land SHA-256: {:?}",
            health.models["xvs_land"].sha256
        );
        assert!(
            health.models["xvs_active_range"].loaded,
            "Frozen xvs-active-range + xvs-multiship-coord + xvs-rta-core must be loaded"
        );

        // XVS-Nav v5.0 Faz-B Stage-1 + XFeat Stage-2A FP16 + LighterGlue Stage-2B FP16 ONNX verification
        assert!(health.models.contains_key("xvs_nav"));
        assert!(health.models.contains_key("xvs_xfeat"));
        assert!(health.models.contains_key("xvs_lighterglue"));
        let has_fazb_nav = find_relative_asset(&[
            "outputs/colab_l4_v5_nav/harvest_faz_b/xvs_nav_v5_fazb_fp16.onnx",
        ])
        .is_some();
        if has_fazb_nav {
            assert!(health.models["xvs_nav"].loaded);
            assert_eq!(health.models["xvs_nav"].status, "READY_RUST_ORT_V5");
            assert_eq!(
                health.models["xvs_nav"].sha256.as_deref(),
                Some(EXPECTED_NAV_V5_FAZB_FP16_ONNX_SHA256)
            );
        } else if find_relative_asset(&["outputs/colab_l4_v5_nav/harvest/xvs_nav_v5_fp16.onnx"])
            .is_some()
        {
            assert!(health.models["xvs_nav"].loaded);
            assert_eq!(health.models["xvs_nav"].status, "READY_RUST_ORT_V5");
            assert_eq!(
                health.models["xvs_nav"].sha256.as_deref(),
                Some(EXPECTED_NAV_V5_FP16_ONNX_SHA256)
            );
        }
        let has_fazb_xfeat = find_relative_asset(&[
            "outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp16.onnx",
        ])
        .is_some();
        if has_fazb_xfeat {
            assert!(health.models["xvs_xfeat"].loaded);
            assert_eq!(health.models["xvs_xfeat"].status, "READY_RUST_ORT_V5");
            assert_eq!(
                health.models["xvs_xfeat"].sha256.as_deref(),
                Some(EXPECTED_XFEAT_FP16_ONNX_SHA256)
            );
        } else if find_relative_asset(&["outputs/colab_l4_v5_nav/harvest/xfeat_320x256_fp32.onnx"])
            .is_some()
        {
            assert!(health.models["xvs_xfeat"].loaded);
            assert_eq!(health.models["xvs_xfeat"].status, "READY_RUST_ORT_V5");
            assert_eq!(
                health.models["xvs_xfeat"].sha256.as_deref(),
                Some(EXPECTED_XFEAT_FP32_ONNX_SHA256)
            );
        }
        let has_fazb_lg = find_relative_asset(&[
            "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp16.onnx",
        ])
        .is_some();
        if has_fazb_lg {
            assert!(health.models["xvs_lighterglue"].loaded);
            assert_eq!(health.models["xvs_lighterglue"].status, "READY_RUST_ORT_V5");
            assert_eq!(
                health.models["xvs_lighterglue"].sha256.as_deref(),
                Some(EXPECTED_LIGHTERGLUE_FP16_ONNX_SHA256)
            );
        }

        // Run full 4-channel demo tick using REAL bundled media JPEGs + MultiShipCoord + SensorHealthMonitor
        let tick = engine
            .run_demo_tick("thermal_fog", 0, true)
            .expect("native demo tick must succeed");
        assert!(tick.non_flight_sim_only);
        assert!(!tick.actuator_authority);
        assert_eq!(tick.rta.mode_id, SimplexMode::NominalMl as u8);
        assert!(tick.sensor_health.multiship_coord_enabled);
        assert!(tick.sensor_health.f1_far91113_active);
        assert!(tick.sensor_health.f4_ghost_hysteresis_active);
        assert!(tick.channels["eo_sky"].annotated_jpeg_b64.is_some());
        assert!(tick.channels["runway_land"].annotated_jpeg_b64.is_some());
        assert_eq!(
            tick.channels["runway_land"].metrics["onnx_output_contract"],
            "v52_6output"
        );
        assert_eq!(
            tick.channels["runway_land"].metrics["roi_hw"],
            serde_json::json!([384, 384])
        );
        assert!(
            tick.channels["runway_land"].metrics["keyline_sigma_px"]
                .as_array()
                .map(|a| a.len() == LAND_NUM_KEYLINES)
                .unwrap_or(false),
            "Expected 5 keyline_sigma_px values from promoted XVS-Land v5.2"
        );
        assert!(tick.channels["thermal_dve"].annotated_jpeg_b64.is_some());
        assert!(tick.channels["crossview_nav"].annotated_jpeg_b64.is_some());
        if health.media_available {
            assert_eq!(tick.channels["eo_sky"].metrics["media_source"], "real_bundled_jpeg");
            assert_eq!(
                tick.channels["runway_land"].metrics["media_source"],
                "real_bundled_jpeg"
            );
            assert_eq!(
                tick.channels["thermal_dve"].metrics["media_source"],
                "real_bundled_jpeg"
            );
            assert_eq!(
                tick.channels["crossview_nav"].metrics["media_source"],
                "real_bundled_jpeg"
            );
        }

        // Verify XFeat Stage-2A keypoints + LighterGlue Stage-2B [1,256,2]/[1,256,64] -> [1,256,256]
        if health.models["xvs_xfeat"].loaded {
            let kpt_cnt = tick.channels["crossview_nav"].metrics["xfeat_keypoints_count"]
                .as_u64()
                .unwrap();
            let xf_ms = tick.channels["crossview_nav"].metrics["xfeat_ort_ms"]
                .as_f64()
                .unwrap();
            assert!(kpt_cnt > 0, "Expected >0 real XFeat keypoints on studio aerial image");
            assert!(xf_ms > 0.0 && xf_ms.is_finite());
        }
        if health.models["xvs_lighterglue"].loaded {
            assert_eq!(
                tick.channels["crossview_nav"].metrics["lighterglue_loaded"],
                true
            );
            assert_eq!(
                tick.channels["crossview_nav"].metrics["lighterglue_scores_shape"],
                serde_json::json!([1, 256, 256])
            );
            let lg_ms = tick.channels["crossview_nav"].metrics["lighterglue_ort_ms"]
                .as_f64()
                .unwrap();
            let lg_matches = tick.channels["crossview_nav"].metrics["lighterglue_matches_count"]
                .as_u64()
                .unwrap();
            assert!(lg_ms > 0.0 && lg_ms.is_finite());
            assert!(
                lg_matches > 0,
                "Expected >0 real LighterGlue mutual matches on studio nav frame"
            );
            // Direct self-match probe on real XFeat keypoints verifying [1,256,2]/[1,256,64] -> [1,256,256]
            let nav_im = engine
                .media_lib
                .pick_nav(0)
                .map(|(im, _, _, _)| im)
                .unwrap_or_else(build_synthetic_runway_frame);
            let (kpts, _) = engine.infer_xfeat_rgb(&nav_im).expect("xfeat self probe");
            let (self_m, self_p, self_ms) = engine
                .infer_lighterglue_match(&kpts, &kpts)
                .expect("lighterglue self-match probe must pass");
            assert!(
                self_m >= kpts.len().saturating_sub(4) && self_m > 0,
                "Expected near-complete self-match on LighterGlue, got {self_m}/{}",
                kpts.len()
            );
            assert!(self_p > 0.70, "Expected high self-match prob, got {self_p}");
            assert!(self_ms > 0.0 && self_ms.is_finite());
        }

        // Verify all 3 new Native Tauri IPC engine methods:
        // 1) native_sensor_health
        let sh = engine.sensor_health();
        assert!(sh.multiship_coord_enabled);
        assert!(sh.f1_far91113_active);
        assert!(sh.f4_ghost_hysteresis_active);
        assert!(!sh.lwir_nuc_frozen);
        assert!(sh.timestamp_monotonic_ok);
        assert!(!sh.timestamp_skew_fault);

        // 2) native_video_sequence_tick across all 4 channels
        for ch in ["d2", "land", "thermal", "nav"] {
            let seq_out = engine
                .run_video_sequence_tick(ch, 2)
                .expect("native_video_sequence_tick must succeed");
            assert!(seq_out.annotated_jpeg_b64.is_some());
            assert!(seq_out.real_weights);
            assert_eq!(seq_out.metrics["frame_idx"], 2);
        }

        // 3) native_export_eval_report (zero host path leaks, exact SHA-256, NOT FOR FLIGHT stamp)
        let rep = engine
            .export_eval_report()
            .expect("native_export_eval_report must succeed");
        assert!(rep.non_flight_sim_only);
        assert!(!rep.actuator_authority);
        assert_eq!(rep.watermark, NON_FLIGHT_WATERMARK);
        assert_eq!(rep.sha256, sha256_bytes_hex(rep.json_content.as_bytes()));
        assert!(rep.json_content.contains(EXPECTED_D2_ONNX_SHA256));
        if has_fazb_nav {
            assert!(rep.json_content.contains(EXPECTED_NAV_V5_FAZB_FP16_ONNX_SHA256));
        }
        if has_fazb_xfeat {
            assert!(rep.json_content.contains(EXPECTED_XFEAT_FP16_ONNX_SHA256));
        } else {
            assert!(rep.json_content.contains(EXPECTED_XFEAT_FP32_ONNX_SHA256));
        }
        if has_fazb_lg {
            assert!(rep.json_content.contains(EXPECTED_LIGHTERGLUE_FP16_ONNX_SHA256));
        }
        assert!(rep.json_content.contains("NOT FOR FLIGHT"));
        let rep_lower = rep.json_content.to_ascii_lowercase();
        if let Ok(user) = std::env::var("USERNAME").or_else(|_| std::env::var("USER")) {
            let u = user.trim().to_ascii_lowercase();
            if !u.is_empty() {
                assert!(!rep_lower.contains(&format!("users\\{u}")));
                assert!(!rep_lower.contains(&format!("users/{u}")));
                assert!(!rep_lower.contains(&format!("home/{u}")));
            }
        }
        assert!(!rep_lower.contains("projects\\openxvs"));
        assert!(!rep_lower.contains("antigravity"));

        // Exact dual-blind guarantee on real XVS-ThermalNet ONNX session (q_eo=0, q_lwir=0 => vacuity==1.0)
        let rwy = build_synthetic_runway_frame_at(99);
        let dual = engine
            .infer_thermal_rgb(&rwy, 0.0, 0.0, 0.0, false)
            .expect("dual-blind thermal inference must succeed");
        let vac = dual.metrics["mean_vacuity"].as_f64().unwrap();
        assert!(
            (vac - 1.0).abs() < 1e-4,
            "expected dual-blind vacuity == 1.0, got {vac}"
        );
        assert_eq!(dual.metrics["simplex_mode"], "DUAL_DVE_SVS_FALLBACK");

        // Compile-time HIL lockout check
        assert!(engine.hil_connect_blocked("mavlink2_udp").is_err());

        // Self-test check (all 6 checks including MultiShipCoord F1/F3/F4 & SensorHealthMonitor)
        let self_test = engine.run_native_self_test();
        assert!(self_test.pass, "native self-test failed: {self_test:?}");
        assert!(self_test.tamper_rejected);
        assert!(self_test
            .checks
            .iter()
            .any(|c| c.name == "xvs_multiship_coord_f1_f3_f4" && c.pass));
        assert!(self_test
            .checks
            .iter()
            .any(|c| c.name == "sensor_health_monitor_nuc_blur_skew" && c.pass));
    }

    #[test]
    fn test_combined_multisensor_realtime_scheduling_benchmark() {
        let mut engine = RustStudioEngine::initialize();
        let health = engine.health();
        if !health.ort_loaded || !health.models["xvs_nav"].loaded {
            eprintln!("[skip] test_combined_multisensor_realtime_scheduling_benchmark: ORT/ONNX assets absent in source-only checkout");
            return;
        }
        assert!(health.ort_loaded);
        assert!(health.models["xvs_d2"].loaded);
        assert!(health.models["xvs_thermal"].loaded);
        assert!(health.models["xvs_land"].loaded);
        assert!(health.models["xvs_nav"].loaded);
        assert!(health.models["xvs_xfeat"].loaded);
        assert!(health.models["xvs_lighterglue"].loaded);

        // Pre-decode real frames outside timed loop to measure pure inference + control scheduling
        let sky_frames: Vec<RgbImage> = (0..12u32)
            .map(|i| {
                engine
                    .media_lib
                    .pick_eo_sky(i)
                    .map(|(im, _, _)| im)
                    .unwrap_or_else(|| build_synthetic_sky_frame(i))
            })
            .collect();
        let th_frames: Vec<(RgbImage, Option<RgbImage>)> = (0..12u32)
            .map(|i| {
                engine
                    .media_lib
                    .pick_thermal_seq(i)
                    .map(|df| (df.eo, df.lwir))
                    .unwrap_or_else(|| (build_synthetic_runway_frame_at(i), None))
            })
            .collect();
        let nav_frame = engine
            .media_lib
            .pick_nav(0)
            .map(|(im, _, _, _)| im)
            .unwrap_or_else(build_synthetic_runway_frame);
        let land_frame = engine
            .media_lib
            .pick_runway_land(0)
            .map(|(im, _, _)| im)
            .unwrap_or_else(build_synthetic_runway_frame);

        let rta_req = NativeRtaInput {
            cam_health_mask: Some(0x3F),
            ood_exceeded_mask: Some(0x00),
            horizon_pl_rad: Some(0.002),
            v_vpl_m: Some(5.5),
            v_hpl_m: Some(15.0),
            spoof_alarm: Some(false),
            ml_detector_fault: Some(false),
            approach_active: Some(true),
            go_around_ack_reset: false,
        };

        // Warmup both fast (10 Hz) and slow (1-2 Hz) loops
        for i in 0..2usize {
            let _ = engine.step_rta(&rta_req);
            engine.d2_state.step_multiship_mesh_guard();
            let _ = engine.infer_d2_rgb(&sky_frames[i], false).unwrap();
            let _ = engine
                .infer_thermal_pair_rgb(&th_frames[i].0, th_frames[i].1.as_ref(), 0.9, 0.95, 0.0, false)
                .unwrap();
            let _ = engine.infer_nav_rgb(&nav_frame, 0.0, false).unwrap();
            let _ = engine.infer_land_rgb(&land_frame, false).unwrap();
        }

        // Measure 10 Hz Fast Loop: D2 + Thermal + MultiShipCoord ActiveRange/MeshGuard + SensorHealth + RTA
        let mut fast_times_ms: Vec<f64> = Vec::with_capacity(10);
        let mut d2_times_ms: Vec<f64> = Vec::with_capacity(10);
        let mut th_times_ms: Vec<f64> = Vec::with_capacity(10);
        for i in 0..10usize {
            let t_loop = Instant::now();
            engine.sim_clock_ms += 100.0;
            engine
                .sensor_health
                .observe_timestamps(engine.sim_clock_ms, engine.sim_clock_ms - 0.5);
            let _ = engine.step_rta(&rta_req);
            engine.d2_state.step_multiship_mesh_guard();
            let d2_out = engine.infer_d2_rgb(&sky_frames[i + 2], false).unwrap();
            let th_out = engine
                .infer_thermal_pair_rgb(
                    &th_frames[i + 2].0,
                    th_frames[i + 2].1.as_ref(),
                    0.9,
                    0.95,
                    0.0,
                    false,
                )
                .unwrap();
            let dt_ms = t_loop.elapsed().as_secs_f64() * 1000.0;
            fast_times_ms.push(dt_ms);
            d2_times_ms.push(d2_out.latency_ms);
            th_times_ms.push(th_out.latency_ms);
        }

        // Measure 1-2 Hz Slow Loop: Nav v5.0 Faz-B (S1) + XFeat FP16 (S2A) + LighterGlue FP16 (S2B) + Land
        let mut slow_times_ms: Vec<f64> = Vec::with_capacity(5);
        let mut nav_s1_times_ms: Vec<f64> = Vec::with_capacity(5);
        let mut xfeat_times_ms: Vec<f64> = Vec::with_capacity(5);
        let mut lg_times_ms: Vec<f64> = Vec::with_capacity(5);
        let mut land_times_ms: Vec<f64> = Vec::with_capacity(5);
        let mut last_xfeat_kpts = 0u64;
        let mut last_lg_matches = 0u64;
        for _ in 0..5usize {
            let t_loop = Instant::now();
            let nav_out = engine.infer_nav_rgb(&nav_frame, 0.0, false).unwrap();
            let land_out = engine.infer_land_rgb(&land_frame, false).unwrap();
            let dt_ms = t_loop.elapsed().as_secs_f64() * 1000.0;
            slow_times_ms.push(dt_ms);
            nav_s1_times_ms.push(nav_out.metrics["ort_infer_ms"].as_f64().unwrap());
            xfeat_times_ms.push(nav_out.metrics["xfeat_ort_ms"].as_f64().unwrap());
            lg_times_ms.push(nav_out.metrics["lighterglue_ort_ms"].as_f64().unwrap());
            last_xfeat_kpts = nav_out.metrics["xfeat_keypoints_count"].as_u64().unwrap();
            last_lg_matches = nav_out.metrics["lighterglue_matches_count"].as_u64().unwrap();
            land_times_ms.push(land_out.latency_ms);
        }

        let median = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let p95 = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(|a, b| a.total_cmp(b));
            let idx = ((v.len() as f64) * 0.95).ceil() as usize;
            v[idx.saturating_sub(1).min(v.len() - 1)]
        };

        let fast_med = median(&mut fast_times_ms);
        let fast_p95 = p95(&mut fast_times_ms);
        let d2_med = median(&mut d2_times_ms);
        let th_med = median(&mut th_times_ms);

        let slow_med = median(&mut slow_times_ms);
        let slow_p95 = p95(&mut slow_times_ms);
        let nav_s1_med = median(&mut nav_s1_times_ms);
        let xf_med = median(&mut xfeat_times_ms);
        let lg_med = median(&mut lg_times_ms);
        let land_med = median(&mut land_times_ms);

        println!(
            "[SCHED-BENCH] profile={} | 10Hz_FAST_LOOP(D2+Thermal+MultiShipCoord+SensorHealth+RTA): med={fast_med:.2}ms p95={fast_p95:.2}ms (d2={d2_med:.2}ms thermal={th_med:.2}ms) | 1-2Hz_SLOW_LOOP(NavFazB+XFeatFP16+LighterGlueFP16+Land): med={slow_med:.2}ms p95={slow_p95:.2}ms (nav_s1={nav_s1_med:.2}ms xfeat={xf_med:.2}ms[{last_xfeat_kpts}kpts] lg={lg_med:.2}ms[{last_lg_matches}m] land={land_med:.2}ms)",
            if cfg!(debug_assertions) { "debug" } else { "release" }
        );

        assert!(fast_med > 0.0 && fast_med.is_finite());
        assert!(slow_med > 0.0 && slow_med.is_finite());
        assert!(last_xfeat_kpts > 0);
        assert!(last_lg_matches > 0);
        // Release-mode scheduling envelope sanity (10 Hz nominal target 100 ms; allow 250 ms margin
        // under concurrent multi-process host contention; 1 Hz slow loop < 1000 ms)
        if !cfg!(debug_assertions) {
            assert!(
                fast_med < 250.0,
                "Release fast loop exceeded 250 ms sanity envelope: {fast_med:.2} ms"
            );
            assert!(
                slow_med < 1000.0,
                "Release 1 Hz slow loop exceeded 1000 ms budget: {slow_med:.2} ms"
            );
        }
    }

    #[test]
    fn test_pinhole_brown_conrady_and_kannala_brandt_fisheye_sub_picopixel() {
        let pin = PinholeIntrinsics::new(
            1280.0, 1280.0, 640.0, 360.0, -0.24, 0.08, 0.0012, -0.0009, -0.015,
        )
        .expect("valid Brown-Conrady pinhole");
        let mut max_bc_err = 0.0_f64;
        for gy in 0..15 {
            for gx in 0..15 {
                let u = 80.0 + (gx as f64) * 80.0;
                let v = 40.0 + (gy as f64) * 45.0;
                let d = pin.distort_pixel([u, v]).expect("distort");
                let r = pin.undistort_pixel_exact(d).expect("undistort");
                let err = (r[0] - u).hypot(r[1] - v);
                if err > max_bc_err {
                    max_bc_err = err;
                }
                let ray = pin.undistort_to_unit_ray_rdf(d).expect("unit ray");
                let rn = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
                assert!((rn - 1.0).abs() < 1e-14 && ray[2] > 0.0);
            }
        }
        assert!(
            max_bc_err < 1e-12,
            "Brown-Conrady max round-trip error {max_bc_err:e} px exceeded 1e-12 px"
        );

        let kb = KannalaBrandtFisheye::new(
            420.0, 420.0, 640.0, 360.0, -0.025, 0.008, -0.002, 0.0004,
        )
        .expect("valid Kannala-Brandt fisheye");
        let mut max_kb_px_err = 0.0_f64;
        let mut max_kb_rad_err = 0.0_f64;
        // Sweep full 0..140 deg FOV (half-angle 0..70 deg) across azimuths
        for deg in [0.0_f64, 15.0, 35.0, 55.0, 62.5, 70.0] {
            let theta = deg.to_radians();
            for az_idx in 0..8 {
                let phi = (az_idx as f64) * core::f64::consts::FRAC_PI_4;
                let ray = [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()];
                let d = kb.project_ray_rdf(ray).expect("kb project");
                let u_ray = kb.undistort_to_unit_ray_exact(d).expect("kb undistort");
                let dot = (ray[0] * u_ray[0] + ray[1] * u_ray[1] + ray[2] * u_ray[2]).clamp(-1.0, 1.0);
                let cx = ray[1] * u_ray[2] - ray[2] * u_ray[1];
                let cy = ray[2] * u_ray[0] - ray[0] * u_ray[2];
                let cz = ray[0] * u_ray[1] - ray[1] * u_ray[0];
                let ang_err = (cx * cx + cy * cy + cz * cz).sqrt().atan2(dot);
                let re_d = kb.project_ray_rdf(u_ray).expect("kb reproject");
                let px_err = (re_d[0] - d[0]).hypot(re_d[1] - d[1]);
                if px_err > max_kb_px_err {
                    max_kb_px_err = px_err;
                }
                if ang_err > max_kb_rad_err {
                    max_kb_rad_err = ang_err;
                }
                let pinhole_uv = kb.undistort_pixel(d, 30, 1e-15).expect("kb pinhole uv");
                let re_pin = kb.distort_pixel(pinhole_uv).expect("kb distort_pixel");
                assert!((re_pin[0] - d[0]).hypot(re_pin[1] - d[1]) < 1e-12);
            }
        }
        assert!(
            max_kb_px_err < 1e-12,
            "Kannala-Brandt 140deg max round-trip error {max_kb_px_err:e} px exceeded 1e-12 px"
        );
        assert!(max_kb_rad_err < 1e-13);
        println!(
            "[OPTICS-V52] Brown-Conrady max_err={max_bc_err:.3e}px | Kannala-Brandt(140deg) max_err={max_kb_px_err:.3e}px"
        );
    }

    #[test]
    fn test_spatial_4x4_12sector_ffc_contrast_collapse_fsync_slerp_and_tbc() {
        // 1. Build a crisp globally textured 160x128 RGB frame -> all 16 tiles & 12 sectors clean
        let make_textured = || {
            let mut img = RgbImage::new(160, 128);
            for y in 0..128u32 {
                for x in 0..160u32 {
                    let checker = (((x / 2) + (y / 2)) % 2) == 0;
                    let v = if checker { 190u8 } else { 65u8 };
                    img.put_pixel(x, y, Rgb([v, v, v]));
                }
            }
            img
        };
        let clean = make_textured();
        let q_clean = SensorHealthMonitor::evaluate_eo_optical_quality(&clean);
        assert!(q_clean.q_eo_optical >= 0.99);
        assert!(q_clean.min_sector_q_eo >= 0.99);
        assert_eq!(q_clean.degraded_sector_mask, 0);

        // Localized sun-glint inside sector 9 (x in 120..133, y in 32..64 => < 2.1% of whole frame!)
        let mut with_glint = make_textured();
        for y in 32..64u32 {
            for x in 120..133u32 {
                with_glint.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        let q_glint = SensorHealthMonitor::evaluate_eo_optical_quality(&with_glint);
        assert!(
            q_glint.q_eo_optical >= 0.95,
            "Global frame average must still appear nominal: {q_glint:?}"
        );
        assert!(
            q_glint.sector_q_eo[9] < 0.05 && q_glint.min_sector_q_eo < 0.05,
            "Sector 9 localized sun-glint must be caught: {:?}",
            q_glint.sector_q_eo
        );
        assert_ne!(q_glint.degraded_sector_mask & (1u16 << 9), 0);

        // Localized raindrop blur inside tile (r=2, c=1) => x in 40..80, y in 64..96 (1/16 of frame)
        let mut with_drop = make_textured();
        for y in 64..96u32 {
            for x in 40..80u32 {
                with_drop.put_pixel(x, y, Rgb([128, 128, 128]));
            }
        }
        let q_drop = SensorHealthMonitor::evaluate_eo_optical_quality(&with_drop);
        assert!(q_drop.q_eo_optical >= 0.95);
        assert!(q_drop.grid_4x4_q_eo[2 * 4 + 1] < 0.10);
        assert!(q_drop.min_sector_q_eo < 0.10);
        assert_ne!(q_drop.degraded_sector_mask & (1u16 << 3), 0);
        assert_ne!(q_drop.degraded_sector_mask & (1u16 << 4), 0);
        assert_ne!(q_drop.degraded_sector_mask & (1u16 << 5), 0);

        // 2. LWIR Hardware FFC_IMMINENT / FFC_IN_PROGRESS zero-tick preemptive mask
        let mut mon = SensorHealthMonitor::new();
        let high_contrast_lwir: Vec<u8> = (0..(160 * 128)).map(|i| ((i * 7) % 200 + 20) as u8).collect();
        let st_nom = mon.observe_lwir_frame_with_ffc(&high_contrast_lwir, 0);
        assert!(!st_nom.mask_lwir_frame && st_nom.dynamic_range_p95_p05 >= 150.0);
        let (_, ql_nom) = mon.compute_effective_qualities(0.9, 0.95);
        assert!((ql_nom - 0.95).abs() < 1e-4);

        let st_imm = mon.observe_lwir_frame_with_ffc(&high_contrast_lwir, LWIR_FFC_BIT_IMMINENT);
        assert!(st_imm.ffc_imminent && st_imm.mask_lwir_frame);
        let (_, ql_imm) = mon.compute_effective_qualities(0.9, 0.95);
        assert_eq!(ql_imm, 0.0, "FFC_IMMINENT must mask LWIR on tick 0");

        let st_prog = mon.observe_lwir_frame_with_ffc(&high_contrast_lwir, LWIR_FFC_BIT_IN_PROGRESS);
        assert!(st_prog.ffc_in_progress && st_prog.mask_lwir_frame);
        let (_, ql_prog) = mon.compute_effective_qualities(0.9, 0.95);
        assert_eq!(ql_prog, 0.0, "FFC_IN_PROGRESS must mask LWIR on tick 0");

        // 3. Noisy closed-shutter AGC frame: MAD >= 0.25 (beats MAD freeze check) yet p95 - p05 < 10 DN!
        let noisy_shutter_a: Vec<u8> = (0..(160 * 128)).map(|i| 128 + ((i % 5) as u8)).collect();
        let noisy_shutter_b: Vec<u8> = (0..(160 * 128)).map(|i| 131 + (((i + 2) % 5) as u8)).collect();
        let _ = mon.observe_lwir_frame_with_ffc(&noisy_shutter_a, 0);
        let st_shutter = mon.observe_lwir_frame_with_ffc(&noisy_shutter_b, 0);
        assert!(st_shutter.last_mad >= 1.5 && !st_shutter.nuc_frozen);
        assert!(st_shutter.dynamic_range_p95_p05 < LWIR_CONTRAST_COLLAPSE_P95_P05_THR);
        assert!(st_shutter.contrast_collapsed && st_shutter.mask_lwir_frame);
        let (_, ql_shutter) = mon.compute_effective_qualities(0.9, 0.95);
        assert_eq!(ql_shutter, 0.0, "p95-p05 contrast-collapse backstop must mask noisy closed shutter");

        // 4. Dual-tier TimestampSkewGuard + Exposure Midpoint + Unit-Quaternion SLERP
        let mut tsg = TimestampSkewGuard::default();
        let r_fsync_ok = tsg.check_exposure_midpoint(1000.0, 4.0, 1001.4); // mid=1002.0, skew=0.6ms <= 1.0ms
        assert!(r_fsync_ok.monotonic_ok && !r_fsync_ok.skew_fault && r_fsync_ok.fsync_precision_ok);
        let r_fsync_warn = tsg.check_exposure_midpoint(1100.0, 4.0, 1098.0); // mid=1102.0, skew=4.0ms > 1.0ms, <= 200ms
        assert!(r_fsync_warn.monotonic_ok && !r_fsync_warn.skew_fault && !r_fsync_warn.fsync_precision_ok);
        let r_hard_fault = tsg.check_exposure_midpoint(1200.0, 4.0, 950.0); // skew=252.0ms > 200ms
        assert!(r_hard_fault.skew_fault && !r_hard_fault.fsync_precision_ok);

        // SLERP half-way between 0 deg and 90 deg yaw (and verify double-cover antipodal sign flip!)
        let q0 = [1.0, 0.0, 0.0, 0.0];
        let half_90 = core::f64::consts::FRAC_PI_4;
        let q1 = [half_90.cos(), 0.0, 0.0, half_90.sin()];
        let q1_neg = [-q1[0], -q1[1], -q1[2], -q1[3]];
        let (qm_pos, t_mid) =
            interpolate_imu_attitude_at_exposure_midpoint(1000.0, q0, 1010.0, q1, 1002.0, 6.0)
                .expect("slerp pos");
        let (qm_neg, _) =
            interpolate_imu_attitude_at_exposure_midpoint(1000.0, q0, 1010.0, q1_neg, 1002.0, 6.0)
                .expect("slerp neg hemisphere");
        assert!((t_mid - 1005.0).abs() < 1e-12);
        let half_45 = core::f64::consts::FRAC_PI_8;
        assert!((qm_pos[0] - half_45.cos()).abs() < 1e-14);
        assert!((qm_pos[3] - half_45.sin()).abs() < 1e-14);
        assert!((qm_neg[0] - qm_pos[0]).abs() < 1e-14 && (qm_neg[3] - qm_pos[3]).abs() < 1e-14);
        assert!(interpolate_imu_attitude_at_exposure_midpoint(1000.0, q0, 1010.0, q1, 1012.0, 2.0).is_err());

        // 5. BearingTrackBeforeConfirm (3-of-5 confirm + clutter gate + miss drop)
        let mut tbc = BearingTrackBeforeConfirm::new_3_of_5();
        // Transient 2-hit clutter does NOT confirm
        assert!(!tbc.observe(Some((0.20, 0.05))).confirmed);
        assert!(!tbc.observe(Some((0.205, 0.051))).confirmed);
        // Far outlier rejected by 0.035 rad angular gate
        let out_snap = tbc.observe(Some((0.90, -0.40)));
        assert!(!out_snap.confirmed && out_snap.hits_in_window == 2);
        // 3rd gated hit within 5-frame window confirms!
        let conf_snap = tbc.observe(Some((0.210, 0.052)));
        assert!(conf_snap.confirmed && conf_snap.status == BearingTrackStatus::Confirmed && conf_snap.hits_in_window == 3);
        // 5 consecutive misses drop track cleanly
        for _ in 0..4 {
            assert_eq!(tbc.observe(None).status, BearingTrackStatus::Confirmed);
        }
        let dropped = tbc.observe(None);
        assert_eq!(dropped.status, BearingTrackStatus::Dropped);
        assert!(!tbc.is_confirmed());
    }

    #[test]
    fn test_xvs_land_dual_contract_5_and_6_output_precision_and_fail_closed() {
        let cal = StudentTCalibrator::new(WAVE_C_LAND_STUDENT_T_NU, WAVE_C_LAND_STUDENT_T_SCALE)
            .expect("valid calibrator");
        let mut kpts = vec![0.0_f32; LAND_NUM_KEYPOINTS * 2];
        for i in 0..LAND_NUM_KEYPOINTS {
            kpts[2 * i] = 80.0 + (i as f32) * 8.0;
            kpts[2 * i + 1] = 64.0 + (i as f32) * 10.0;
        }
        let mut keylines = vec![0.0_f32; LAND_NUM_KEYLINES * 3];
        for j in 0..LAND_NUM_KEYLINES {
            keylines[3 * j] = 100.0 + (j as f32) * 12.0;
            keylines[3 * j + 1] = 0.6;
            keylines[3 * j + 2] = 0.8;
        }
        let mut cov = vec![0.0_f32; LAND_NUM_KEYPOINTS * 4];
        for i in 0..LAND_NUM_KEYPOINTS {
            cov[4 * i] = 1.0;
            cov[4 * i + 3] = 1.0;
        }

        // 1. 5-output (v4.5/v5.1) backward compatibility
        let p5 = parse_xvs_land_outputs(&kpts, &keylines, &cov, None, &cal)
            .expect("5-output v5.1 parse must succeed");
        assert_eq!(p5.contract, LandOutputContract::V51FiveOutput);
        assert_eq!(p5.contract.as_str(), "v51_5output");
        assert!(p5.keyline_sigma_px.is_none());
        assert_eq!(p5.keypoints_xy.len(), LAND_NUM_KEYPOINTS);
        assert_eq!(p5.keylines_hesse.len(), LAND_NUM_KEYLINES);
        let expected_w_unif = 1.0 / (LAND_DEFAULT_KEYLINE_SIGMA_PX * LAND_DEFAULT_KEYLINE_SIGMA_PX);
        for w in p5.keyline_precision_weights {
            assert!((w - expected_w_unif).abs() < 1e-12);
        }
        assert!((p5.effective_keyline_sigma_px - LAND_DEFAULT_KEYLINE_SIGMA_PX).abs() < 1e-12);
        let expected_kp_sig = (2.0 * cal.variance_factor()).sqrt();
        assert!((p5.mean_keypoint_sigma_px - expected_kp_sig).abs() < 1e-10);

        // 2. 6-output (v5.2) per-line keyline_sigma_px inverse-variance precision weighting
        let sigmas_v52 = [1.0_f32, 1.0, 2.0, 4.0, 50.0];
        let p6 = parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(&sigmas_v52), &cal)
            .expect("6-output v5.2 parse must succeed");
        assert_eq!(p6.contract, LandOutputContract::V52SixOutputKeylineSigma);
        assert_eq!(p6.contract.as_str(), "v52_6output");
        let parsed_sigs = p6.keyline_sigma_px.expect("keyline_sigma_px must be Some");
        assert!((parsed_sigs[0] - 1.0).abs() < 1e-6);
        assert!((parsed_sigs[4] - 50.0).abs() < 1e-6);
        // Sharp line (sigma=1.0) carries 2500x precision weight of degraded line (sigma=50.0)
        assert!((p6.keyline_precision_weights[0] - 1.0).abs() < 1e-12);
        assert!((p6.keyline_precision_weights[2] - 0.25).abs() < 1e-12);
        assert!((p6.keyline_precision_weights[3] - 0.0625).abs() < 1e-12);
        assert!((p6.keyline_precision_weights[4] - 0.0004).abs() < 1e-12);
        assert!(p6.keyline_precision_weights[0] / p6.keyline_precision_weights[4] > 2499.0);
        let mean_prec: f64 = (1.0 + 1.0 + 0.25 + 0.0625 + 0.0004) / 5.0;
        let expected_eff = (1.0_f64 / mean_prec).sqrt();
        assert!((p6.effective_keyline_sigma_px - expected_eff).abs() < 1e-10);
        assert!((p6.mean_keypoint_sigma_px - p5.mean_keypoint_sigma_px).abs() < 1e-12);

        // 3. Fail-closed rejection of NaN, +Inf, -Inf, 0.0, negative, and truncated sigma
        let bad_cases: &[&[f32]] = &[
            &[1.0, f32::NAN, 2.0, 2.0, 2.0],
            &[1.0, f32::INFINITY, 2.0, 2.0, 2.0],
            &[1.0, f32::NEG_INFINITY, 2.0, 2.0, 2.0],
            &[1.0, 0.0, 2.0, 2.0, 2.0],
            &[1.0, -1.5, 2.0, 2.0, 2.0],
            &[1.0, 2.0, 3.0],
        ];
        for bad in bad_cases {
            assert!(
                parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(bad), &cal).is_err(),
                "Expected fail-closed rejection for bad keyline_sigma_px={bad:?}"
            );
        }
    }

    fn json_contains_null(v: &serde_json::Value) -> bool {
        match v {
            serde_json::Value::Null => true,
            serde_json::Value::Array(arr) => arr.iter().any(json_contains_null),
            serde_json::Value::Object(map) => map.values().any(json_contains_null),
            _ => false,
        }
    }

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_u8(&mut self) -> u8 {
            (self.next_u64() & 0xFF) as u8
        }
    }

    #[test]
    fn test_r1_fail_closed_nan_gps_spoof_and_zero_json_null_leaks() {
        let mut engine = RustStudioEngine::initialize();
        let img = build_synthetic_runway_frame();

        // 1. NaN / +Inf / -Inf GPS offset MUST fail CLOSED (spoof_detected = true) with ZERO JSON null
        for bad_off in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let nav = engine
                .infer_nav_rgb(&img, bad_off, false)
                .expect("nav inference must survive non-finite gps_offset_m");
            assert_eq!(
                nav.metrics["spoof_detected"], true,
                "Non-finite gps_offset_m={bad_off} failed OPEN!"
            );
            assert_eq!(nav.metrics["gps_offset_non_finite"], true);
            let off_val = nav.metrics["gps_offset_m"]
                .as_f64()
                .expect("gps_offset_m must serialize as finite number, never null");
            assert!(off_val.is_finite() && off_val >= 9999.0);
            assert!(
                !json_contains_null(&nav.metrics),
                "Nav metrics leaked JSON null: {}",
                nav.metrics
            );
        }

        // Nominal finite offsets remain exact
        let nav_clean = engine.infer_nav_rgb(&img, 12.5, false).expect("nav clean");
        assert_eq!(nav_clean.metrics["spoof_detected"], false);
        assert_eq!(nav_clean.metrics["gps_offset_non_finite"], false);
        assert!((nav_clean.metrics["gps_offset_m"].as_f64().unwrap() - 12.5).abs() < 1e-9);
        assert!(!json_contains_null(&nav_clean.metrics));

        let nav_spoof = engine.infer_nav_rgb(&img, 165.0, false).expect("nav spoof");
        assert_eq!(nav_spoof.metrics["spoof_detected"], true);
        assert_eq!(nav_spoof.metrics["gps_offset_non_finite"], false);
        assert!((nav_spoof.metrics["gps_offset_m"].as_f64().unwrap() - 165.0).abs() < 1e-9);

        // 2. Non-finite q_eo / q_lwir / fog_density into thermal MUST fail closed with ZERO JSON null
        let th_bad = engine
            .infer_thermal_rgb(&img, f32::NAN, f32::INFINITY, f32::NAN, false)
            .expect("thermal inference must survive NaN/Inf inputs");
        assert_eq!(th_bad.metrics["q_eo"].as_f64(), Some(0.0));
        assert_eq!(th_bad.metrics["q_lwir"].as_f64(), Some(0.0));
        assert_eq!(th_bad.metrics["q_eo_input_non_finite"], true);
        assert_eq!(th_bad.metrics["q_lwir_input_non_finite"], true);
        assert_eq!(th_bad.metrics["q_eo_effective"].as_f64(), Some(0.0));
        assert_eq!(th_bad.metrics["q_lwir_effective"].as_f64(), Some(0.0));
        assert_eq!(th_bad.metrics["simplex_mode"], "DUAL_DVE_SVS_FALLBACK");
        let vac = th_bad.metrics["mean_vacuity"].as_f64().unwrap();
        assert!((vac - 1.0).abs() < 1e-4, "Expected vacuity=1.0, got {vac}");
        assert!(
            !json_contains_null(&th_bad.metrics),
            "Thermal metrics leaked JSON null: {}",
            th_bad.metrics
        );

        // 3. Non-finite timestamp skew into sensor_health() MUST NOT leak JSON null
        engine.sensor_health.observe_timestamps(f64::NAN, 1000.0);
        let sh_val = serde_json::to_value(engine.sensor_health()).expect("serialize sensor_health");
        assert_eq!(sh_val["timestamp_skew_fault"], true);
        assert!(
            !json_contains_null(&sh_val),
            "NativeSensorHealthPayload leaked JSON null: {sh_val}"
        );
    }

    #[test]
    fn test_r2_simplex_rta_p0_latch_and_go_around_ack_reset_recovery() {
        // Backward-compatible JSON deserialization when go_around_ack_reset is omitted
        let omitted: NativeRtaInput =
            serde_json::from_str(r#"{"approach_active": true, "v_vpl_m": 5.0}"#)
                .expect("NativeRtaInput without go_around_ack_reset must deserialize");
        assert!(!omitted.go_around_ack_reset);
        let explicit: NativeRtaInput =
            serde_json::from_str(r#"{"approach_active": false, "go_around_ack_reset": true}"#)
                .expect("NativeRtaInput with go_around_ack_reset=true must deserialize");
        assert!(explicit.go_around_ack_reset);

        let mut engine = RustStudioEngine::initialize();

        // 1. Fault (NaN v_vpl_m) while approach_active = true latches P0 InvalidInhibit + mandatory_go_around
        let fault = engine.step_rta(&NativeRtaInput {
            v_vpl_m: Some(f64::NAN),
            approach_active: Some(true),
            go_around_ack_reset: false,
            ..NativeRtaInput::default()
        });
        assert_eq!(fault.mode_id, SimplexMode::InvalidInhibit as u8);
        assert!(fault.mandatory_go_around);
        assert!(fault.inhibit_avoidance_turns);

        // 2. Stays latched across clean ticks when go_around_ack_reset = false (even after approach_active = false)
        let clean_no_ack = NativeRtaInput {
            approach_active: Some(false),
            go_around_ack_reset: false,
            ..NativeRtaInput::default()
        };
        for _ in 0..25 {
            let st = engine.step_rta(&clean_no_ack);
            assert_eq!(st.mode_id, SimplexMode::InvalidInhibit as u8);
            assert!(st.mandatory_go_around);
            assert!(st.inhibit_avoidance_turns);
        }

        // 3. Cannot clear latch while approach_active == true even if go_around_ack_reset == true
        let still_on_approach = engine.step_rta(&NativeRtaInput {
            approach_active: Some(true),
            go_around_ack_reset: true,
            ..NativeRtaInput::default()
        });
        assert_eq!(still_on_approach.mode_id, SimplexMode::InvalidInhibit as u8);
        assert!(still_on_approach.mandatory_go_around);

        // 4. Clean recovery when !approach_active && go_around_ack_reset == true
        let ack_req = NativeRtaInput {
            approach_active: Some(false),
            go_around_ack_reset: true,
            ..NativeRtaInput::default()
        };
        let mut st = engine.step_rta(&ack_req);
        assert!(
            !st.mandatory_go_around,
            "go_around_ack_reset=true with approach_active=false must clear latch"
        );
        // Remaining P0_RECOVERY_TICKS - 1 clean ticks exit P0 -> P1 FallbackD0Safe
        for _ in 1..P0_RECOVERY_TICKS {
            st = engine.step_rta(&clean_no_ack);
        }
        assert_eq!(
            st.mode_id,
            SimplexMode::FallbackD0Safe as u8,
            "Expected P0 -> P1 staged recovery after P0_RECOVERY_TICKS, got {st:?}"
        );
        assert!(!st.inhibit_avoidance_turns);
        assert!(!st.mandatory_go_around);

        // DWELL_TICKS more clean ticks recover P1 -> P4 NominalMl
        for _ in 0..DWELL_TICKS {
            st = engine.step_rta(&clean_no_ack);
        }
        assert_eq!(
            st.mode_id,
            SimplexMode::NominalMl as u8,
            "Expected P1 -> P4 full nominal recovery after DWELL_TICKS, got {st:?}"
        );
        assert!(!st.inhibit_avoidance_turns);
        assert!(!st.mandatory_go_around);
    }

    #[test]
    fn test_r3_sensor_health_fpn_white_noise_cold_sky_and_d2_tbc_gate() {
        let mut rng = SplitMix64(0x00C3_2026_5233);

        // Helper: realistic crisp textured 640x480 aerial scene (horizon + structures + terrain)
        let make_scene_640x480 = || {
            let mut img = RgbImage::new(640, 480);
            for y in 0..480u32 {
                for x in 0..640u32 {
                    let xf = x as f64;
                    let yf = y as f64;
                    let base = 115.0
                        + 35.0 * (xf * 0.045).sin() * (yf * 0.038).cos()
                        + 25.0 * ((xf + yf) * 0.025).cos()
                        + if y > 260 { -35.0 } else { 30.0 };
                    // Crisp building/runway blocks (high focus_quality, high spatial autocorrelation)
                    let block = if ((x / 16) + (y / 16)) % 3 == 0 && y > 180 {
                        55.0
                    } else {
                        0.0
                    };
                    let v = (base + block).clamp(15.0, 235.0) as u8;
                    img.put_pixel(x, y, Rgb([v, v.saturating_add(4), v.saturating_sub(4)]));
                }
            }
            img
        };

        // Helper: stamp dead-pixel FPN row/col grid onto RGB image at given fraction
        let stamp_fpn_rgb = |base: &RgbImage, frac: f64| -> RgbImage {
            let mut out = base.clone();
            let (w, h) = (out.width() as usize, out.height() as usize);
            let n_rows = ((h as f64) * frac).round().max(1.0) as usize;
            let n_cols = ((w as f64) * frac).round().max(1.0) as usize;
            for i in 0..n_rows {
                let r = 4 + (i * (h - 9)) / n_rows.max(1);
                let col = if i % 2 == 0 {
                    Rgb([0, 0, 0])
                } else {
                    Rgb([255, 255, 255])
                };
                for x in 0..w {
                    out.put_pixel(x as u32, r as u32, col);
                }
            }
            for i in 0..n_cols {
                let c = 4 + (i * (w - 9)) / n_cols.max(1);
                let col = if i % 2 == 0 {
                    Rgb([255, 255, 255])
                } else {
                    Rgb([0, 0, 0])
                };
                for y in 0..h {
                    out.put_pixel(c as u32, y as u32, col);
                }
            }
            out
        };

        // 1. Clean 640x480 scene has zero FPN and high spatial autocorrelation
        let clean_eo = make_scene_640x480();
        let q_clean = SensorHealthMonitor::evaluate_eo_optical_quality(&clean_eo);
        assert!(
            !q_clean.dead_pixel_fpn_fault
                && !q_clean.sensor_noise_fault
                && !q_clean.degraded
                && q_clean.dead_line_fraction < 0.03
                && q_clean.spatial_autocorr > 0.60,
            "Clean scene falsely degraded: {q_clean:?}"
        );

        // 2. 5% and 15% dead-pixel row/col FPN grids caught on EO
        let fpn_5 = stamp_fpn_rgb(&clean_eo, 0.05);
        let q_fpn_5 = SensorHealthMonitor::evaluate_eo_optical_quality(&fpn_5);
        assert!(q_fpn_5.dead_pixel_fpn_fault);
        assert!(q_fpn_5.flags().contains(&"DEAD_PIXEL_FPN_FAULT"));
        assert!(q_fpn_5.degraded && q_fpn_5.q_eo_optical <= 0.45);

        let fpn_15 = stamp_fpn_rgb(&clean_eo, 0.15);
        let q_fpn_15 = SensorHealthMonitor::evaluate_eo_optical_quality(&fpn_15);
        assert!(q_fpn_15.dead_pixel_fpn_fault);
        assert!(q_fpn_15.dead_line_fraction >= 0.15);
        assert!(q_fpn_15.flags().contains(&"DEAD_PIXEL_FPN_FAULT"));
        assert!(
            q_fpn_15.degraded && q_fpn_15.q_eo_optical <= 0.10,
            "15% FPN bypassed EO gate: {q_fpn_15:?}"
        );

        // 3. Uncorrelated uniform white noise and binary 0/255 noise caught on EO (q <= 0.05)
        let make_uniform_noise = |r: &mut SplitMix64| {
            let mut im = RgbImage::new(640, 480);
            for p in im.pixels_mut() {
                p[0] = r.next_u8();
                p[1] = r.next_u8();
                p[2] = r.next_u8();
            }
            im
        };
        let make_binary_noise = |r: &mut SplitMix64| {
            let mut im = RgbImage::new(640, 480);
            for p in im.pixels_mut() {
                let v = if (r.next_u8() & 1) == 0 { 0u8 } else { 255u8 };
                *p = Rgb([v, v, v]);
            }
            im
        };
        let uni_noise = make_uniform_noise(&mut rng);
        let bin_noise = make_binary_noise(&mut rng);
        for (label, n_img) in [("uniform", &uni_noise), ("binary", &bin_noise)] {
            let q_n = SensorHealthMonitor::evaluate_eo_optical_quality(n_img);
            assert!(
                q_n.sensor_noise_fault
                    && q_n.spatial_autocorr < NOISE_AUTOCORR_MIN
                    && q_n.degraded
                    && q_n.q_eo_optical <= 0.05
                    && q_n.flags().contains(&"SENSOR_NOISE_FAULT"),
                "{label} white noise bypassed EO gate: {q_n:?}"
            );
        }

        // 4. Cold-sky LWIR + compact hot-target preservation vs single-pixel spike / blackout / saturation
        let mut lw_mon = SensorHealthMonitor::new();
        let cold_probes: &[(i32, i32, u8, usize)] = &[
            (3, 1, 210, 4),
            (8, 2, 185, 5),
            (16, 3, 165, 6),
            (28, 3, 195, 4),
            (40, 4, 220, 5),
        ];
        for (idx, &(bg_dn, jitter, tgt_dn, box_sz)) in cold_probes.iter().enumerate() {
            let mut frame = vec![0u8; 160 * 128];
            for px in &mut frame {
                let d = ((rng.next_u8() as i32) % (2 * jitter + 1)) - jitter;
                *px = (bg_dn + d).clamp(0, 255) as u8;
            }
            // Shift target slightly per probe so NUC freeze never triggers across probes
            let ox = 72 + idx * 3;
            let oy = 56 + idx * 2;
            for dy in 0..box_sz {
                for dx in 0..box_sz {
                    frame[(oy + dy) * 160 + (ox + dx)] = tgt_dn;
                }
            }
            let st = lw_mon.observe_lwir_plane_with_ffc(&frame, 160, 128, 0);
            let flags = st.flags();
            assert!(
                st.localized_hot_target_rescue
                    && !st.contrast_collapsed
                    && !st.degraded
                    && st.q_lwir_optical >= 0.85
                    && !flags.contains(&"LOW_CONTRAST_DYNAMIC_RANGE")
                    && !flags.contains(&"LENS_CAP_OR_UNDEREXPOSED"),
                "Clean cold-sky LWIR hot-target probe bg={bg_dn} falsely killed: {st:?} flags={flags:?}"
            );
        }

        // Isolated 1-pixel hot spikes on black lens-cap MUST fail closed (2x2 min-filter rejects them)
        let mut dead_1px = vec![0u8; 160 * 128];
        dead_1px[64 * 160 + 80] = 250;
        for k in 0..25usize {
            dead_1px[(10 + k * 4) * 160 + (10 + k * 5)] = 245;
        }
        let st_1px = lw_mon.observe_lwir_plane_with_ffc(&dead_1px, 160, 128, 0);
        assert!(
            !st_1px.localized_hot_target_rescue && st_1px.degraded && st_1px.q_lwir_optical == 0.0,
            "Isolated 1px spikes bypassed 2x2 min-filter: {st_1px:?}"
        );

        // All-zero black, all-255 white, 40% clip, 60% saturation MUST fail closed
        let st_blk = lw_mon.observe_lwir_plane_with_ffc(&vec![0u8; 160 * 128], 160, 128, 0);
        assert!(st_blk.degraded && st_blk.q_lwir_optical == 0.0);
        let st_wht = lw_mon.observe_lwir_plane_with_ffc(&vec![255u8; 160 * 128], 160, 128, 0);
        assert!(st_wht.degraded && st_wht.q_lwir_optical == 0.0);
        let mut lw_sat60: Vec<u8> = (0..(160 * 128)).map(|i| ((i * 7) % 160 + 30) as u8).collect();
        for px in lw_sat60.iter_mut().take(160 * 78) {
            *px = 254;
        }
        let st_sat60 = lw_mon.observe_lwir_plane_with_ffc(&lw_sat60, 160, 128, 0);
        assert!(st_sat60.degraded && st_sat60.q_lwir_optical == 0.0);

        // 5. Real ONNX Thermal + D2 / 3-of-5 TBC optical fault gate
        let mut engine = RustStudioEngine::initialize();

        // 15% dead-pixel grid -> DUAL_DVE_SVS_FALLBACK & vacuity == 1.0000
        let th_fpn = engine
            .infer_thermal_pair_rgb(&fpn_15, Some(&fpn_15), 0.90, 0.95, 0.0, false)
            .expect("thermal 15% FPN");
        assert!(th_fpn.metrics["q_eo_effective"].as_f64().unwrap() <= 0.10);
        assert!(th_fpn.metrics["q_lwir_effective"].as_f64().unwrap() <= 0.10);
        assert_eq!(th_fpn.metrics["simplex_mode"], "DUAL_DVE_SVS_FALLBACK");
        assert!((th_fpn.metrics["mean_vacuity"].as_f64().unwrap() - 1.0).abs() < 1e-4);

        // Pure white noise -> DUAL_DVE_SVS_FALLBACK & q_eff <= 0.05
        let th_noise = engine
            .infer_thermal_pair_rgb(&uni_noise, Some(&uni_noise), 0.90, 0.95, 0.0, false)
            .expect("thermal noise");
        assert!(th_noise.metrics["q_eo_effective"].as_f64().unwrap() <= 0.05);
        assert!(th_noise.metrics["q_lwir_effective"].as_f64().unwrap() <= 0.05);
        assert_eq!(th_noise.metrics["simplex_mode"], "DUAL_DVE_SVS_FALLBACK");

        // 20/20 pure random noise frames (10 uniform + 10 binary) -> 0/20 false TBC confirms!
        let mut noise_tbc_confirms = 0usize;
        let mut noise_dets = 0usize;
        for i in 0..20usize {
            let n_frame = if i < 10 {
                make_uniform_noise(&mut rng)
            } else {
                make_binary_noise(&mut rng)
            };
            let d2_out = engine
                .infer_d2_rgb(&n_frame, false)
                .expect("d2 noise inference");
            assert_eq!(d2_out.metrics["eo_optical_fault"], true);
            if d2_out.metrics["detected"] == true {
                noise_dets += 1;
            }
            if d2_out.metrics["bearing_tbc"]["confirmed"] == true {
                noise_tbc_confirms += 1;
            }
        }
        assert_eq!(noise_dets, 0, "Pure noise must produce 0/20 D2 detections");
        assert_eq!(
            noise_tbc_confirms, 0,
            "Pure noise must produce 0/20 false 3-of-5 TBC confirmations"
        );

        // 10/10 15% dead-pixel grid frames -> 0/10 false TBC confirms!
        let mut fpn_tbc_confirms = 0usize;
        for _ in 0..10usize {
            let d2_fpn = engine
                .infer_d2_rgb(&fpn_15, false)
                .expect("d2 15% FPN inference");
            assert_eq!(d2_fpn.metrics["eo_optical_fault"], true);
            assert_eq!(d2_fpn.metrics["detected"], false);
            if d2_fpn.metrics["bearing_tbc"]["confirmed"] == true {
                fpn_tbc_confirms += 1;
            }
        }
        assert_eq!(
            fpn_tbc_confirms, 0,
            "15% dead-pixel FPN grid must produce 0/10 false TBC confirmations"
        );

        // Real bundled eo_sky frames must remain healthy (zero false EO optical faults)
        let n_sky = engine.media_lib.eo_sky.len().min(60);
        for f_idx in 0..n_sky {
            if let Some((sky_im, rel, _)) = engine.media_lib.pick_eo_sky(f_idx as u32) {
                let q = SensorHealthMonitor::evaluate_eo_optical_quality(&sky_im);
                assert!(
                    !q.dead_pixel_fpn_fault && !q.sensor_noise_fault && !q.degraded,
                    "Real bundled eo_sky frame {rel} falsely flagged: {q:?}"
                );
            }
        }
    }

    #[test]
    fn test_r4_xvs_land_keyline_sigma_physical_floor_clamps_subnormal() {
        let cal = StudentTCalibrator::new(V52_LAND_STUDENT_T_NU, V52_LAND_STUDENT_T_SCALE)
            .expect("valid v5.2 calibrator");
        let kpts = vec![120.0_f32; LAND_NUM_KEYPOINTS * 2];
        let keylines = vec![1.0_f32; LAND_NUM_KEYLINES * 3];
        let mut cov = vec![0.0_f32; LAND_NUM_KEYPOINTS * 4];
        for i in 0..LAND_NUM_KEYPOINTS {
            cov[4 * i] = 1.0;
            cov[4 * i + 3] = 1.0;
        }

        // Subnormal f32::from_bits(1) (~1.4e-45) MUST clamp cleanly to MIN_KEYLINE_SIGMA_PX (0.25 px)
        let sub = f32::from_bits(1);
        assert!(sub > 0.0 && sub.is_finite() && (sub as f64) < 1e-40);
        let parsed_all =
            parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(&[sub; LAND_NUM_KEYLINES]), &cal)
                .expect("subnormal positive sigmas must clamp to floor");
        assert!(
            parsed_all.effective_keyline_sigma_px >= MIN_KEYLINE_SIGMA_PX - 1e-12,
            "effective_keyline_sigma_px collapsed: {}",
            parsed_all.effective_keyline_sigma_px
        );
        assert!((parsed_all.effective_keyline_sigma_px - MIN_KEYLINE_SIGMA_PX).abs() < 1e-12);
        let sigs = parsed_all.keyline_sigma_px.expect("Some sigmas");
        for k in 0..LAND_NUM_KEYLINES {
            assert!((sigs[k] - MIN_KEYLINE_SIGMA_PX).abs() < 1e-12);
            assert!((parsed_all.keyline_precision_weights[k] - 16.0).abs() < 1e-12);
        }

        // Single subnormal mixed with nominal sigmas must NOT collapse effective_keyline_sigma_px
        let mixed = [sub, 1.0_f32, 1.0, 2.0, 4.0];
        let parsed_mix = parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(&mixed), &cal)
            .expect("mixed subnormal sigmas must parse cleanly");
        assert!(
            parsed_mix.effective_keyline_sigma_px >= MIN_KEYLINE_SIGMA_PX,
            "mixed effective_keyline_sigma_px collapsed: {}",
            parsed_mix.effective_keyline_sigma_px
        );
        let expected_mean_prec = (16.0 + 1.0 + 1.0 + 0.25 + 0.0625) / 5.0;
        let expected_eff = (1.0_f64 / expected_mean_prec).sqrt();
        assert!((parsed_mix.effective_keyline_sigma_px - expected_eff).abs() < 1e-10);
    }

    #[test]
    fn test_r6_honest_signal_derived_nav_confidence() {
        // 1. Pure formula boundary checks
        assert_eq!(
            compute_nav_confidence(1, 1, 1.0, true, 100, 0.2, true, 80, 0.8, 1.0),
            0.0
        );
        assert_eq!(
            compute_nav_confidence(320, 256, 1.0, true, 0, 0.0, true, 0, 0.0, 1.0),
            0.0
        );
        assert_eq!(
            compute_nav_confidence(320, 256, 1.0, true, 128, 0.2, true, 0, 0.0, 1.0),
            0.0
        );
        assert_eq!(
            compute_nav_confidence(320, 256, f64::NAN, true, 128, 0.2, true, 64, 0.8, 1.0),
            0.0
        );
        assert_eq!(
            compute_nav_confidence(320, 256, 1.0, true, 200, 0.2, true, 150, 0.85, 0.0),
            0.0
        );
        let hi = compute_nav_confidence(320, 256, 1.0, true, 200, 0.22, true, 80, 0.82, 1.0);
        let lo = compute_nav_confidence(320, 256, 1.0, true, 16, 0.08, true, 4, 0.25, 1.0);
        assert!((0.75..=0.98).contains(&hi), "Expected high-inlier in 0.75..=0.98, got {hi}");
        assert!(lo > 0.0 && lo < 0.50 && lo < hi, "Expected weak match honestly low, got {lo}");

        // 2. End-to-end real ONNX Nav triad verification (all-black, all-white, 1x1, noise vs real pair)
        let mut engine = RustStudioEngine::initialize();

        let black = RgbImage::from_pixel(640, 480, Rgb([0, 0, 0]));
        let nav_black = engine.infer_nav_rgb(&black, 0.0, false).expect("nav black");
        assert_eq!(
            nav_black.confidence, 0.0,
            "All-black frame must yield 0.0 nav confidence, got {}",
            nav_black.confidence
        );

        let white = RgbImage::from_pixel(640, 480, Rgb([255, 255, 255]));
        let nav_white = engine.infer_nav_rgb(&white, 0.0, false).expect("nav white");
        assert_eq!(
            nav_white.confidence, 0.0,
            "All-white frame must yield 0.0 nav confidence, got {}",
            nav_white.confidence
        );

        let tiny = RgbImage::from_pixel(1, 1, Rgb([128, 128, 128]));
        let nav_tiny = engine.infer_nav_rgb(&tiny, 0.0, false).expect("nav 1x1");
        assert_eq!(
            nav_tiny.confidence, 0.0,
            "1x1 frame must yield 0.0 nav confidence, got {}",
            nav_tiny.confidence
        );

        let mut rng = SplitMix64(0x00C3_9999_1234);
        let mut noise = RgbImage::new(640, 480);
        for p in noise.pixels_mut() {
            *p = Rgb([rng.next_u8(), rng.next_u8(), rng.next_u8()]);
        }
        let nav_noise = engine.infer_nav_rgb(&noise, 0.0, false).expect("nav noise");
        assert!(
            nav_noise.confidence <= 0.05,
            "Pure white noise must collapse nav confidence <= 0.05, got {}",
            nav_noise.confidence
        );

        // Genuine aerial/satellite pair from bundled studio_media (or textured runway self-probe)
        if engine.health().models["xvs_nav"].loaded {
            let nav_real = if let Some(df) = engine.media_lib.pick_nav_pair(0) {
                engine
                    .infer_nav_pair_rgb(&df.aerial, df.sat_ref.as_ref(), 0.0, false)
                    .expect("nav real aerial/sat pair")
            } else {
                let rwy = build_synthetic_runway_frame();
                engine.infer_nav_rgb(&rwy, 0.0, false).expect("nav runway")
            };
            assert!(
                (0.75..=0.98).contains(&nav_real.confidence),
                "Expected genuine high-inlier aerial/sat nav confidence in [0.75, 0.98], got {} (metrics={})",
                nav_real.confidence,
                nav_real.metrics
            );
        }
    }

    #[test]
    fn test_faz1_real_dual_bank_cusum_catches_sub_45m_stealth_ramp_and_seq_vpr_wired() {
        let mut engine = RustStudioEngine::initialize();
        let img = build_synthetic_runway_frame();
        // Slow 2.0 m/s stealth ramp: EVERY single offset is <= 28 m (< old 45 m dummy threshold!),
        // proving RustStudioEngine now runs real DualBankEsEkfCusum instead of dummy abs() > 45.0.
        let mut detected_at_m: Option<f64> = None;
        for step_k in 1..=14usize {
            let off_m = 2.0 * (step_k as f64);
            let nav = engine
                .infer_nav_rgb(&img, off_m, false)
                .expect("nav stealth ramp step");
            assert!(!json_contains_null(&nav.metrics));
            if nav.real_weights {
                assert!(nav.metrics["seq_vpr_window_len"].as_u64().unwrap() >= 1);
            }
            if nav.metrics["spoof_detected"] == true {
                detected_at_m = Some(off_m);
                assert_eq!(nav.metrics["cusum_gnss_quarantined"], true);
                assert_eq!(nav.metrics["cusum_active_bank"], "bank_b_visual");
                break;
            }
        }
        assert!(
            detected_at_m.map(|m| m < 30.0).unwrap_or(false),
            "Real DualBankEsEkfCusum must catch 2.0 m/s stealth ramp < 30 m HAL (got {detected_at_m:?})"
        );
    }

    #[test]
    fn test_v53_land_7th_output_presence_gate_d2_dynamic_640_and_safety_guards() {
        let cal = StudentTCalibrator::new(V52_LAND_STUDENT_T_NU, V52_LAND_STUDENT_T_SCALE)
            .expect("valid calibrator");
        let kpts = vec![120.0_f32; LAND_NUM_KEYPOINTS * 2];
        let keylines = vec![1.0_f32; LAND_NUM_KEYLINES * 3];
        let mut cov = vec![0.0_f32; LAND_NUM_KEYPOINTS * 4];
        for i in 0..LAND_NUM_KEYPOINTS {
            cov[4 * i] = 1.0;
            cov[4 * i + 3] = 1.0;
        }
        let sigs = [1.0_f32, 1.2, 1.5, 2.0, 2.5];

        // 1. XVS-Land extended 7th output: sigmoid(logit) >= 0.65 -> NOMINAL
        let p_nom = parse_xvs_land_outputs_with_presence(
            &kpts,
            &keylines,
            &cov,
            Some(&sigs),
            Some(&[2.0_f32]), // sigmoid(2.0) ~= 0.8808 >= 0.65
            &cal,
        )
        .expect("extended nominal parse");
        assert_eq!(p_nom.contract, LandOutputContract::V53SevenOutputRunwayPresence);
        assert_eq!(p_nom.contract.as_str(), "v53_7output");
        assert!(p_nom.runway_present_admitted);
        assert_eq!(p_nom.runway_gate_mode, "NOMINAL");
        assert!(p_nom.runway_present_prob.unwrap() > 0.85);

        // 2. XVS-Land extended 7th output: sigmoid(logit) < 0.65 -> P0_INVALID_INHIBIT (fail-closed)
        let p_rej = parse_xvs_land_outputs_with_presence(
            &kpts,
            &keylines,
            &cov,
            Some(&sigs),
            Some(&[0.0_f32]), // sigmoid(0.0) = 0.50 < 0.65
            &cal,
        )
        .expect("extended low-presence parse");
        assert_eq!(p_rej.contract, LandOutputContract::V53SevenOutputRunwayPresence);
        assert!(!p_rej.runway_present_admitted);
        assert_eq!(p_rej.runway_gate_mode, "P0_INVALID_INHIBIT");
        assert!((p_rej.runway_present_prob.unwrap() - 0.5).abs() < 1e-6);

        // 3. Non-finite runway_present_logit fails closed to P0_INVALID_INHIBIT
        let p_nan = parse_xvs_land_outputs_with_presence(
            &kpts,
            &keylines,
            &cov,
            Some(&sigs),
            Some(&[f32::NAN]),
            &cal,
        )
        .expect("extended NaN presence parse");
        assert!(!p_nan.runway_present_admitted);
        assert_eq!(p_nan.runway_gate_mode, "P0_INVALID_INHIBIT");
        assert_eq!(p_nan.runway_present_prob, Some(0.0));
        assert!(p_nan.runway_present_logit.is_none());

        // 4. Empty runway_present_logit fails closed with Err
        assert!(
            parse_xvs_land_outputs_with_presence(&kpts, &keylines, &cov, Some(&sigs), Some(&[]), &cal)
                .is_err()
        );

        // 5. Dynamic 640x640 XVS-D2 decoder (hm 320x320) + truncated/empty bounds safety
        let hm_h = 320usize;
        let hm_w = 320usize;
        let area = hm_h * hm_w;
        let mut hm = vec![-10.0_f32; area];
        let mut pmk = vec![0.0_f32; area];
        let off = vec![0.0_f32; 2 * area];
        let lsc = vec![2.3_f32; 2 * area];
        let cell = 150 * hm_w + 200;
        hm[cell] = 3.0;
        pmk[cell] = 1.0;
        let (det640, mp640) = decode_d2_top(&hm, &off, &lsc, &pmk, hm_h, hm_w, 0.25);
        assert!(det640.is_some() && mp640 > 0.90);
        let b = det640.unwrap();
        let cx = 0.5 * (b[0] + b[2]);
        let cy = 0.5 * (b[1] + b[3]);
        assert!((cx - 401.0).abs() < 2.0 && (cy - 301.0).abs() < 2.0);
        // Truncated slices must return (None, 0.0) without panicking
        assert_eq!(decode_d2_top(&hm[..10], &off, &lsc, &pmk, hm_h, hm_w, 0.25), (None, 0.0));

        // 6. draw_rect_outline on 0x0 image must not panic
        let mut empty = RgbImage::new(0, 0);
        draw_rect_outline(&mut empty, 0, 0, 10, 10, Rgb([255, 0, 0]));

        // 7. analyze_image_b64 rejects oversized (>16 MiB) base64 payloads without decoding
        let mut engine = RustStudioEngine::initialize();
        let huge_b64 = "A".repeat(MAX_ANALYZE_IMAGE_B64_LEN + 8);
        assert!(engine
            .analyze_image_b64("d2", &huge_b64, 0.9, 0.9, 0.0, 0.0)
            .is_err());

        // 8. IPC optimization: image_b64 is omitted (None) when annotated_jpeg_b64 is populated,
        //    and run_demo_tick rate-group decimation + real UTC timestamp work cleanly
        let t0 = engine.run_demo_tick("thermal_fog", 0, true).expect("tick 0");
        for ch in t0.channels.values() {
            assert!(ch.annotated_jpeg_b64.is_some());
            assert!(ch.image_b64.is_none(), "duplicate image_b64 must be None");
        }
        let t1 = engine.run_demo_tick("thermal_fog", 1, true).expect("tick 1");
        assert!(t1.channels["runway_land"].annotated_jpeg_b64.is_some());
        assert!(t1.channels["crossview_nav"].annotated_jpeg_b64.is_some());

        let ts = format_utc_now_iso8601();
        assert!(ts.len() == 20 && ts.ends_with('Z') && ts != "2026-11-23T00:00:00Z");

        // 9. Video sequence tick skew is 0.5 ms (fsync_precision_ok == true)
        let _ = engine.run_video_sequence_tick("d2", 0).expect("video tick");
        assert!(engine.sensor_health().fsync_precision_ok);
    }
}
