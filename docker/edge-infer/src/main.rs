// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! OpenXVS v5.0 Headless Pure-Rust `ort` (`load-dynamic`) + `xvs-active-range`
//! + `xvs-multiship-coord` + `xvs-rta-core` edge evaluator.
//!
//! Security & Regulatory Invariants:
//! - `#![forbid(unsafe_code)]`
//! - `#[cfg(feature = "non_flight_sim_only")]` compiles out any actuation path (`ACTUATOR_AUTHORITY = false`)
//! - Explicitly blocks Windows `System32\onnxruntime.dll` hijack; loads bundled/explicit `ORT_DYLIB_PATH`
//! - Fail-closed SHA-256 pinning enforced on ALL 6 ONNX models (`SHA256_MISMATCH_REJECTED`)
//! - Zero TLS/AES confidentiality crypto crates (5D002 private-first guardrail)
//! - Zero GPL/AGPL dependencies

#![forbid(unsafe_code)]

pub mod eval_frames;
pub mod nav_integrity;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use nav_integrity::{
    DualBankConfig, DualBankEsEkfCusum, DualBankStepInput, SequenceVprConfig, SequenceVprFilter,
    VisualAliasingThresholds, VisualFixMetrics, CUSUM_H_LOCKED,
};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xvs_active_range::{
    decode_wire, encode_wire, ActiveRangeConfig, ActiveRangeMeasurement, ActiveRangeMode,
    OwnshipStateNed,
};
use xvs_math_core::mesh_daa::OpenXvsBearingTrack;
use xvs_multiship_coord::{
    CoordinatedActiveRange, Ring0AuthorityGate, StatefulMeshDaaGuard, SwerveCoordRole,
};
use xvs_rta_core::{
    build_rta_input_from_f64, RtaFloatTelemetry, RtaTickInput, SimplexRta, ODD_YAML_SHA256,
};

pub const EDGE_INFER_VERSION: &str = "1.1.1";
pub const NON_FLIGHT_SIM_ONLY: bool = cfg!(feature = "non_flight_sim_only");
pub const ACTUATOR_AUTHORITY: bool = false;
pub const NON_FLIGHT_WATERMARK: &str =
    "SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT";

pub const D2_CROP_SIZE: usize = 256;
pub const D2_HID_C: usize = 24;
pub const D2_HID_HW: usize = D2_CROP_SIZE / 4; // 64
pub const LAND_RUNWAY_PRESENT_MIN_PROB: f64 = 0.65;

pub const V52_LAND_STUDENT_T_NU: f64 = 6.0;
pub const V52_LAND_STUDENT_T_SCALE: f64 = 1.048_075_542_704_791_2;
pub const LAND_STUDENT_T_NU: f64 = 6.0;
pub const LAND_STUDENT_T_SCALE: f64 = 4.214_477_485_437_598;
pub const LAND_NUM_KEYPOINTS: usize = 12;
pub const LAND_NUM_KEYLINES: usize = 5;
pub const LAND_DEFAULT_KEYLINE_SIGMA_PX: f64 = 1.5;
pub const MIN_KEYLINE_SIGMA_PX: f64 = 0.25;
/// Promoted Wave 3A SHA-256 pin for v5.2 6-output `xvs_land_v52_fp16.onnx` (`2,899,997 B`, `384x384`).
pub const EXPECTED_LAND_V52_FP16_SHA256: &str =
    "834c8e270f237f044d3920c98d5c0818363ab781be78071884e9365a221ac799";
pub const EXPECTED_LAND_V52_FP32_SHA256: &str =
    "fc8e808394b7e20c0d80e5fe357306ba332ee6f57018666e941a8d55caee6407";
pub const EXPECTED_LAND_V52_FP16_ONNX_SHA256: Option<&str> = Some(EXPECTED_LAND_V52_FP16_SHA256);

pub const LIGHTERGLUE_STATIC_K: usize = 256;
pub const LIGHTERGLUE_DESC_DIM: usize = 64;
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
pub const EXPECTED_THERMAL_FP16_ONNX_SHA256: &str =
    "85a6c9100846b00810889c6c1d97b227cd495057b34c98923abd3b04acf6bd98";
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

#[cfg(feature = "non_flight_sim_only")]
const _: () = {
    assert!(!ACTUATOR_AUTHORITY);
};

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

    pub fn variance_factor(&self) -> f64 {
        (self.scale * self.scale) * (self.nu / (self.nu - 2.0))
    }

    pub fn calibrate_cov2x2(&self, cov: [f64; 4]) -> [f64; 4] {
        let f = self.variance_factor();
        [cov[0] * f, cov[1] * f, cov[2] * f, cov[3] * f]
    }
}

// ---- Multi-Contract XVS-Land 5-Output, 6-Output & 7-Output (extended `runway_present_logit`) Parser ----

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

pub fn resolve_land_input_hw(sess: &Session) -> (usize, usize) {
    if let Some(inp) = sess.inputs().first() {
        if let Some(shape) = inp.dtype().tensor_shape() {
            if shape.len() == 4 && shape[2] > 0 && shape[3] > 0 {
                return (shape[2] as usize, shape[3] as usize);
            }
        }
    }
    (256, 256)
}

/// Inspect XVS-D2 ONNX session input shapes dynamically:
/// `x_t` `[1, 6, H, W]` (`256x256` OR `640x640` high-res) and `h_prev_warped` `[1, C, H/4, W/4]`.
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

pub fn parse_xvs_land_outputs(
    kpts: &[f32],
    keylines: &[f32],
    cov: &[f32],
    keyline_sigma_opt: Option<&[f32]>,
    calibrator: &StudentTCalibrator,
) -> Result<ParsedLandOutputs, String> {
    parse_xvs_land_outputs_with_presence(kpts, keylines, cov, keyline_sigma_opt, None, calibrator)
}

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

// ---- Wave-1/2 Synced Pure-Rust Sensor & Optics Hardening Mirror (`#![forbid(unsafe_code)]`) ----

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

#[inline]
pub(crate) fn round2(v: f64) -> f64 {
    if v.is_finite() {
        (v * 100.0).round() / 100.0
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn round4(v: f64) -> f64 {
    if v.is_finite() {
        (v * 10_000.0).round() / 10_000.0
    } else {
        0.0
    }
}

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

    pub fn undistort_pixel(
        &self,
        uv_dist: [f64; 2],
        max_iter: usize,
        tol: f64,
    ) -> Result<[f64; 2], String> {
        let [x, y] = self.undistort_to_normalized(uv_dist, max_iter, tol)?;
        Ok([self.fx * x + self.cx, self.fy * y + self.cy])
    }

    pub fn undistort_pixel_exact(&self, uv_dist: [f64; 2]) -> Result<[f64; 2], String> {
        self.undistort_pixel(uv_dist, 30, 1e-15)
    }

    pub fn undistort_to_unit_ray_rdf(&self, uv_dist: [f64; 2]) -> Result<[f64; 3], String> {
        let [x, y] = self.undistort_to_normalized(uv_dist, 30, 1e-15)?;
        let inv_n = 1.0 / (x * x + y * y + 1.0).sqrt();
        Ok([x * inv_n, y * inv_n, inv_n])
    }
}

/// Exact Kannala-Brandt equidistant fisheye camera model (`k1..k4`) for wide-angle `110-140 deg` lenses.
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
        Ok([
            self.fx * (scale * ray_rdf[0]) + self.cx,
            self.fy * (scale * ray_rdf[1]) + self.cy,
        ])
    }

    pub fn distort_pixel(&self, uv_pinhole: [f64; 2]) -> Result<[f64; 2], String> {
        self.validate()?;
        if !(uv_pinhole[0].is_finite() && uv_pinhole[1].is_finite()) {
            return Err("Pinhole pixel contains non-finite coordinate".to_string());
        }
        let x = (uv_pinhole[0] - self.cx) / self.fx;
        let y = (uv_pinhole[1] - self.cy) / self.fy;
        self.project_ray_rdf([x, y, 1.0])
    }

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

    pub fn undistort_to_unit_ray_exact(&self, uv_dist: [f64; 2]) -> Result<[f64; 3], String> {
        self.undistort_to_unit_ray_rdf(uv_dist, 30, 1e-15)
    }
}

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

pub fn exposure_midpoint_ms(t_sof_ms: f64, exposure_ms: f64) -> Result<f64, String> {
    if !(t_sof_ms.is_finite() && exposure_ms.is_finite()) || exposure_ms < 0.0 {
        return Err("Exposure SOF timestamp and duration must be finite with t_exp >= 0".to_string());
    }
    Ok(t_sof_ms + 0.5 * exposure_ms)
}

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

#[inline]
fn hist_line_median_and_stuck(hist: &[u32; 256], len: usize) -> (f64, bool, u64, u64) {
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
        return (med, true, sum_u64, sq_u64);
    }
    let lo_b = ((med - 1.5).floor().max(0.0) as usize).min(255);
    let hi_b = ((med + 1.5).ceil().min(255.0) as usize).min(255);
    let mut near = 0usize;
    for (v, &cnt) in hist.iter().enumerate().take(hi_b + 1).skip(lo_b) {
        if ((v as f64) - med).abs() <= 1.5 {
            near += cnt as usize;
        }
    }
    (med, ((near as f64) / lf) >= 0.70, sum_u64, sq_u64)
}

#[inline]
fn dot_u8_slice(a: &[u8], b: &[u8]) -> u64 {
    let mut total = 0u64;
    for (ca, cb) in a.chunks(4096).zip(b.chunks(4096)) {
        let mut acc = 0u32;
        for (&x, &y) in ca.iter().zip(cb.iter()) {
            acc += (x as u32) * (y as u32);
        }
        total += acc as u64;
    }
    total
}

#[inline]
fn sum_u8_slice(a: &[u8]) -> u64 {
    let mut total = 0u64;
    for chunk in a.chunks(4096) {
        let mut acc = 0u32;
        for &x in chunk {
            acc += x as u32;
        }
        total += acc as u64;
    }
    total
}

/// Detect stuck/dead sensor readout rows & columns (exact Wave-1/2 parity with Studio `rust_engine.rs`).
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
        let (med, stuck, r_sum, r_sq) = hist_line_median_and_stuck(&hist, w);
        frame_sum += r_sum;
        frame_sq += r_sq;
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
            let (med, stuck, _, _) = hist_line_median_and_stuck(hist, h);
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

/// Detect spatially uncorrelated white/binary noise (exact Wave-1/2 parity with Studio `rust_engine.rs`,
/// using auto-vectorizable integer dot-product covariance identities).
pub fn detect_white_noise_fault_u8(plane: &[u8], w: usize, h: usize) -> (f64, f64) {
    if w < 4 || h < 4 || plane.len() != w * h {
        return (1.0, 1.0);
    }
    let n = (w * h) as f64;
    let mut sum_u64 = 0u64;
    let mut sq_u64 = 0u64;
    for chunk in plane.chunks(4096) {
        let mut s = 0u32;
        let mut q = 0u32;
        for &px in chunk {
            let v = px as u32;
            s += v;
            q += v * v;
        }
        sum_u64 += s as u64;
        sq_u64 += q as u64;
    }
    let mean = (sum_u64 as f64) / n;
    let var = (((sq_u64 as f64) / n) - mean * mean).max(0.0);
    let std_val = var.sqrt();
    if !std_val.is_finite() || std_val <= NOISE_STD_MIN || var <= 1e-6 {
        return (1.0, 1.0);
    }

    let mut dot_h1 = 0u64;
    let mut dot_h2 = 0u64;
    let mut dot_v1 = 0u64;
    let mut dot_v2 = 0u64;
    let mut col0_sum = 0u64;
    let mut col1_sum = 0u64;
    let mut col_m2_sum = 0u64;
    let mut col_m1_sum = 0u64;

    for y in 0..h {
        let row0 = &plane[y * w..(y + 1) * w];
        col0_sum += row0[0] as u64;
        col1_sum += row0[1] as u64;
        col_m2_sum += row0[w - 2] as u64;
        col_m1_sum += row0[w - 1] as u64;

        dot_h1 += dot_u8_slice(&row0[..w - 1], &row0[1..]);
        dot_h2 += dot_u8_slice(&row0[..w - 2], &row0[2..]);

        if y + 1 < h {
            let row1 = &plane[(y + 1) * w..(y + 2) * w];
            dot_v1 += dot_u8_slice(row0, row1);
        }
        if y + 2 < h {
            let row2 = &plane[(y + 2) * w..(y + 3) * w];
            dot_v2 += dot_u8_slice(row0, row2);
        }
    }

    let row0_sum = sum_u8_slice(&plane[..w]);
    let row1_sum = sum_u8_slice(&plane[w..2 * w]);
    let row_m2_sum = sum_u8_slice(&plane[(h - 2) * w..(h - 1) * w]);
    let row_m1_sum = sum_u8_slice(&plane[(h - 1) * w..h * w]);

    let cnt_h1 = (h * (w - 1)).max(1) as f64;
    let cnt_h2 = (h * (w - 2)).max(1) as f64;
    let cnt_v1 = ((h - 1) * w).max(1) as f64;
    let cnt_v2 = ((h - 2) * w).max(1) as f64;

    let mean_sq = mean * mean;
    let sum_ab_h1 = ((sum_u64 - col_m1_sum) + (sum_u64 - col0_sum)) as f64;
    let sum_ab_h2 = ((sum_u64 - col_m1_sum - col_m2_sum) + (sum_u64 - col0_sum - col1_sum)) as f64;
    let sum_ab_v1 = ((sum_u64 - row_m1_sum) + (sum_u64 - row0_sum)) as f64;
    let sum_ab_v2 = ((sum_u64 - row_m1_sum - row_m2_sum) + (sum_u64 - row0_sum - row1_sum)) as f64;

    let cov_h1 = (dot_h1 as f64) - mean * sum_ab_h1 + cnt_h1 * mean_sq;
    let cov_h2 = (dot_h2 as f64) - mean * sum_ab_h2 + cnt_h2 * mean_sq;
    let cov_v1 = (dot_v1 as f64) - mean * sum_ab_v1 + cnt_v1 * mean_sq;
    let cov_v2 = (dot_v2 as f64) - mean * sum_ab_v2 + cnt_v2 * mean_sq;

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

    pub fn evaluate_eo_luma_plane(luma: &[f64], w: usize, h: usize) -> EoOpticalQuality {
        Self::evaluate_eo_luma_plane_with_fpn_noise(luma, w, h, None)
    }

    pub(crate) fn evaluate_eo_luma_plane_with_fpn_noise(
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

    pub(crate) fn record_eo_quality(&mut self, q: EoOpticalQuality) {
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

    pub fn observe_eo_luma_u8(&mut self, luma_u8: &[u8], w: usize, h: usize) -> EoOpticalQuality {
        let luma: Vec<f64> = luma_u8.iter().map(|&v| v as f64).collect();
        let q = Self::evaluate_eo_luma_plane(&luma, w, h);
        self.record_eo_quality(q);
        q
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

    pub(crate) fn observe_lwir_plane_with_ffc_precomputed(
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
            } else {
                let (d, f) = if has_2d {
                    detect_dead_line_fpn_u8(lwir_u8, w, h)
                } else {
                    (0.0, 1.0)
                };
                let (a, s) = if has_2d {
                    detect_white_noise_fault_u8(lwir_u8, w, h)
                } else {
                    (1.0, 1.0)
                };
                (d, f, a, s)
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
                let mut acc_u64 = 0u64;
                for (&a, &b) in lwir_u8.iter().zip(prev.iter()) {
                    acc_u64 += a.abs_diff(b) as u64;
                }
                let m = (acc_u64 as f64) / (lwir_u8.len() as f64);
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
        if let Some(prev) = self.prev_lwir_u8.as_mut() {
            prev.clear();
            prev.extend_from_slice(lwir_u8);
        } else {
            self.prev_lwir_u8 = Some(lwir_u8.to_vec());
        }

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

    pub fn observe_timestamps(&mut self, cam_ts_ms: f64, imu_ts_ms: f64) -> TimestampCheckResult {
        let res = self.timestamp_guard.check(cam_ts_ms, imu_ts_ms);
        self.timestamp_skew_ms = res.skew_ms;
        self.timestamp_monotonic_ok = res.monotonic_ok;
        self.timestamp_skew_fault = res.skew_fault;
        self.fsync_precision_ok = res.fsync_precision_ok;
        res
    }

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

pub fn resolve_ort_dylib_path() -> Result<PathBuf, String> {
    resolve_ort_dylib_path_with_hint(None)
}

pub(crate) fn resolve_ort_dylib_path_with_hint(hint_dir: Option<&Path>) -> Result<PathBuf, String> {
    if let Ok(raw) = std::env::var("ORT_DYLIB_PATH") {
        let candidate = PathBuf::from(raw.trim());
        if is_forbidden_system_ort_path(&candidate) {
            return Err(
                "Blocked forbidden Windows System32/SysWOW64 onnxruntime.dll path (hijack guard)"
                    .to_string(),
            );
        }
        if candidate.is_file() {
            return Ok(candidate);
        }
        return Err("ORT_DYLIB_PATH does not point to an existing shared library".to_string());
    }

    let mut search_paths: Vec<PathBuf> = Vec::new();
    let lib_names: &[&str] = if cfg!(target_os = "windows") {
        &["onnxruntime.dll"]
    } else if cfg!(target_os = "macos") {
        &["libonnxruntime.dylib"]
    } else {
        &["libonnxruntime.so"]
    };
    if let Some(hint) = hint_dir {
        for name in lib_names {
            search_paths.push(hint.join(name));
            search_paths.push(hint.join("lib").join(name));
            if let Some(parent) = hint.parent() {
                search_paths.push(parent.join(name));
                search_paths.push(parent.join("lib").join(name));
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for name in lib_names {
                search_paths.push(dir.join(name));
                search_paths.push(dir.join("lib").join(name));
                search_paths.push(dir.join("resources").join(name));
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for name in lib_names {
            search_paths.push(cwd.join(name));
            search_paths.push(cwd.join("lib").join(name));
        }
    }
    for name in lib_names {
        search_paths.push(PathBuf::from("/opt/openxvs/lib").join(name));
        search_paths.push(PathBuf::from("/usr/local/lib").join(name));
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();
    for start in [cwd.as_deref(), exe_dir.as_deref()].into_iter().flatten() {
        let mut cur = Some(start);
        for _ in 0..7 {
            if let Some(base) = cur {
                if cfg!(target_os = "windows") {
                    search_paths.push(
                        base.join(".venv")
                            .join("Lib")
                            .join("site-packages")
                            .join("onnxruntime")
                            .join("capi")
                            .join("onnxruntime.dll"),
                    );
                } else {
                    for name in lib_names {
                        search_paths.push(base.join("lib").join(name));
                    }
                }
                cur = base.parent();
            }
        }
    }

    for p in search_paths {
        if !is_forbidden_system_ort_path(&p) && p.is_file() {
            return Ok(p);
        }
    }
    Err("Could not locate bundled ONNX Runtime shared library; set ORT_DYLIB_PATH".to_string())
}

pub(crate) fn find_model_path(
    model_dir: &Path,
    explicit_model_dir: bool,
    direct_names: &[&str],
    repo_rel_candidates: &[&str],
) -> Option<PathBuf> {
    if explicit_model_dir {
        for name in direct_names {
            let p = model_dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
        let nested_models = model_dir.join("models");
        for name in direct_names {
            let p = nested_models.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
        return None;
    }

    // Auto-discovery order: exe_dir -> exe_dir/models -> cwd -> cwd/models -> model_dir (/opt/openxvs/models) -> repo parents
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();

    let mut direct_dirs: Vec<PathBuf> = Vec::with_capacity(6);
    if let Some(ref d) = exe_dir {
        direct_dirs.push(d.clone());
        direct_dirs.push(d.join("models"));
    }
    if let Some(ref d) = cwd {
        direct_dirs.push(d.clone());
        direct_dirs.push(d.join("models"));
    }
    direct_dirs.push(model_dir.to_path_buf());

    for dir in &direct_dirs {
        for name in direct_names {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }

    for start in [cwd.as_deref(), exe_dir.as_deref()].into_iter().flatten() {
        let mut cur = Some(start);
        for _ in 0..7 {
            if let Some(base) = cur {
                for rel in repo_rel_candidates {
                    let p = base.join(rel);
                    if p.is_file() {
                        return Some(p);
                    }
                }
                cur = base.parent();
            }
        }
    }
    None
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("failed to read model: {e}"))?;
    let digest = Sha256::digest(&bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    Ok(out)
}

/// Fail-closed SHA-256 verification against a model's allowed `(filename, expected_sha256)` pin table.
pub fn verify_pinned_onnx_artifact(
    path: &Path,
    allowed_pins: &[(&str, &str)],
) -> Result<(String, String), (Option<String>, Option<String>)> {
    let fname = path.file_name().map(|s| s.to_string_lossy().to_string());
    let actual_sha = sha256_file(path).ok();
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

/// Count top-K Apache-2.0 XFeat keypoints from dense `[1,65,32,40]` logits + `[1,1,32,40]` heatmap.
fn count_xfeat_keypoints(
    keypoints_logits: &[f32],
    heatmap: &[f32],
    max_keypoints: usize,
    nms_threshold: f32,
) -> usize {
    let hc = 32_usize;
    let wc = 40_usize;
    let cell_area = hc * wc;
    if keypoints_logits.len() < 65 * cell_area || heatmap.len() < cell_area {
        return 0;
    }
    let h_img = hc * 8;
    let w_img = wc * 8;
    let mut kmap = vec![0.0_f32; h_img * w_img];
    let mut probs = [0.0_f32; 65];
    for yc in 0..hc {
        for xc in 0..wc {
            let ci = yc * wc + xc;
            let mut max_l = f32::NEG_INFINITY;
            for c in 0..65 {
                max_l = max_l.max(keypoints_logits[c * cell_area + ci]);
            }
            let mut sum_e = 0.0_f32;
            for c in 0..65 {
                let e = (keypoints_logits[c * cell_area + ci] - max_l).exp();
                probs[c] = e;
                sum_e += e;
            }
            let inv = 1.0_f32 / sum_e.max(1e-12);
            for sy in 0..8 {
                let row = (yc * 8 + sy) * w_img;
                for sx in 0..8 {
                    kmap[row + xc * 8 + sx] = probs[sy * 8 + sx] * inv;
                }
            }
        }
    }
    let mut count = 0usize;
    for y in 4..(h_img - 4) {
        for x in 4..(w_img - 4) {
            let kv = kmap[y * w_img + x];
            if kv <= nms_threshold {
                continue;
            }
            let mut is_max = true;
            'nms: for ny in (y - 2)..=(y + 2) {
                let row = ny * w_img;
                for nx in (x - 2)..=(x + 2) {
                    if kmap[row + nx] > kv {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if is_max {
                count += 1;
            }
        }
    }
    count.min(max_keypoints)
}

/// Count mutual nearest-neighbor matches from LighterGlue log-assignment `[1, 256, 256]` scores.
fn count_lighterglue_mutual_matches(scores: &[f32], min_prob: f32) -> (usize, f64) {
    let k = LIGHTERGLUE_STATIC_K;
    if scores.len() != k * k {
        return (0, 0.0);
    }
    let mut row_best_j = vec![0usize; k];
    let mut row_best_s = vec![f32::NEG_INFINITY; k];
    let mut col_best_i = vec![0usize; k];
    let mut col_best_s = vec![f32::NEG_INFINITY; k];

    for i in 0..k {
        let row_off = i * k;
        for j in 0..k {
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

    let mut matches = 0usize;
    let mut prob_sum = 0.0_f64;
    for i in 0..k {
        let j = row_best_j[i];
        if col_best_i[j] == i {
            let p = row_best_s[i].clamp(-80.0, 0.0).exp();
            if p >= min_prob {
                matches += 1;
                prob_sum += p as f64;
            }
        }
    }
    let mean_p = if matches > 0 {
        prob_sum / (matches as f64)
    } else {
        0.0
    };
    (matches, mean_p)
}

fn make_ecef_obs(
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

#[derive(Debug, Serialize)]
struct ModelBenchResult {
    name: String,
    loaded: bool,
    status: String,
    sha256: Option<String>,
    load_ms: Option<f64>,
    latency_ms: Option<f64>,
    output_summary: String,
}

#[derive(Debug, Serialize)]
struct MultiShipCoordCheck {
    enabled: bool,
    f1_far91113_right_turn_ok: bool,
    f3_ring0_p0_zero_bank_ok: bool,
    f4_sticky_triple_ray_ok: bool,
    f4_sticky_ghosts_suppressed: u32,
    f4_stale_obs_rejected: u32,
}

#[derive(Debug, Serialize)]
struct SensorOpticsHardeningCheck {
    brown_conrady_max_err_px: f64,
    kannala_brandt_140deg_max_err_px: f64,
    spatial_4x4_12sector_localized_fault_ok: bool,
    lwir_ffc_zero_tick_and_contrast_collapse_ok: bool,
    dual_tier_fsync_and_slerp_ok: bool,
    bearing_tbc_3_of_5_ok: bool,
}

#[derive(Debug, Serialize)]
struct EdgeSelfCheckReport {
    version: &'static str,
    non_flight_sim_only: bool,
    actuator_authority: bool,
    watermark: &'static str,
    target_arch: &'static str,
    target_os: &'static str,
    ort_dylib_loaded: bool,
    odd_yaml_sha256: &'static str,
    rta_mode: u8,
    active_range_wire_crc_ok: bool,
    active_range_r_hat_m: f32,
    multiship_coord: MultiShipCoordCheck,
    sensor_optics_hardening: SensorOpticsHardeningCheck,
    models: Vec<ModelBenchResult>,
}

static GLOBAL_EDGE_ORT_ENV: OnceLock<Result<Environment, String>> = OnceLock::new();

fn acquire_edge_ort_env() -> Result<Environment, String> {
    acquire_edge_ort_env_with_hint(None)
}

pub(crate) fn acquire_edge_ort_env_with_hint(
    hint_dir: Option<&Path>,
) -> Result<Environment, String> {
    GLOBAL_EDGE_ORT_ENV
        .get_or_init(|| {
            let dll_path = resolve_ort_dylib_path_with_hint(hint_dir)?;
            let builder = ort::init_from(&dll_path).map_err(|e| format!("ort init error: {e}"))?;
            builder
                .build()
                .map_err(|e| format!("ort env build error: {e}"))
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

pub(crate) fn build_optimized_ort_session(
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

fn run_self_check_with_model_dir(model_dir: &Path, explicit_model_dir: bool) -> EdgeSelfCheckReport {
    // 1. Direct Rust xvs-multiship-coord (F1/F3) + xvs-active-range 64B CRC32 wire check
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
        size_rate_px_s: 2.2,
        contrast_ratio: 0.65,
        sky_beta_ext: 2.0e-4,
        daytime_clear: true,
        track_age: 6,
        dt_s: 0.1,
    };
    let ar_pkt = coord.step(&own, &meas, None, Ring0AuthorityGate::NOMINAL);
    let wire = encode_wire(&ar_pkt);
    let ar_wire_ok = decode_wire(&wire).is_some();
    let f1_ok =
        ar_pkt.cmd_bank_rad >= 0.0 && coord.last_role() == SwerveCoordRole::OpticalFar91113Right;

    // F3 Ring-0 P0 arbiter check
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
    let f3_ok = p0_pkt.cmd_bank_rad == 0.0 && p0_pkt.mode == ActiveRangeMode::NominalHold as u8;

    // F4 StatefulMeshDaaGuard 3-ship -> 2-ship sticky ghost suppression + >200 ms staleness check
    let base_ecef = [6_378_137.0_f64, 0.0, 0.0];
    let p_s1 = [base_ecef[0], 0.0, 0.0];
    let p_s2 = [base_ecef[0], 1000.0, 0.0];
    let p_s3 = [base_ecef[0] + 1000.0, 500.0, 0.0];
    let t1 = [base_ecef[0] + 500.0, 200.0, 0.0];
    let t2 = [base_ecef[0] + 500.0, 800.0, 0.0];
    let mut guard = StatefulMeshDaaGuard::new();
    let ts3 = 1_700_000_000_000u64;
    let obs3 = [
        make_ecef_obs(1, 1, p_s1, t1, ts3),
        make_ecef_obs(1, 2, p_s1, t2, ts3),
        make_ecef_obs(2, 1, p_s2, t1, ts3),
        make_ecef_obs(2, 2, p_s2, t2, ts3),
        make_ecef_obs(3, 1, p_s3, t1, ts3),
        make_ecef_obs(3, 2, p_s3, t2, ts3),
    ];
    let g3 = guard.fuse_guarded(&obs3, ts3 + 20_000);
    let ts2 = ts3 + 200_000;
    let obs2 = [
        make_ecef_obs(1, 1, p_s1, t1, ts2),
        make_ecef_obs(1, 2, p_s1, t2, ts2),
        make_ecef_obs(2, 1, p_s2, t1, ts2),
        make_ecef_obs(2, 2, p_s2, t2, ts2),
    ];
    let g2 = guard.fuse_guarded(&obs2, ts2 + 20_000);
    let stale_obs = [make_ecef_obs(1, 1, p_s1, t1, ts2 - 250_000)];
    let g_stale = guard.fuse_guarded(&stale_obs, ts2 + 50_000);
    let f4_ok = g3.num_fused == 2
        && g2.num_fused == 2
        && g2.sticky_triple_ray_active
        && g2.sticky_ghosts_suppressed >= 2
        && g_stale.stale_obs_rejected == 1;

    // 2. Direct Rust xvs-rta-core ASTM F3269-21 Simplex RTA check
    let mut rta = SimplexRta::new();
    let rta_out = rta.step(&RtaTickInput::default());

    // 3. ORT load-dynamic + fail-closed SHA-256-pinned ONNX probes on ALL 5+1 models
    let mut ort_loaded = false;
    let mut models: Vec<ModelBenchResult> = Vec::new();

    if let Ok(env) = acquire_edge_ort_env() {
        ort_loaded = true;

        // 3.1 XVS-D2 (SHA-256 pinned fail-closed, dynamic 256x256 / 640x640 high-res input shape, 1T pinned)
        if let Some(d2_path) = find_model_path(
            model_dir,
            explicit_model_dir,
            &["xvs_d2_web_fp16.onnx"],
            &[
                "apps/livedeck/public/models/xvs_d2_web_fp16.onnx",
                "models/xvs_d2_web_fp16.onnx",
            ],
        ) {
            match verify_pinned_onnx_artifact(&d2_path, D2_ONNX_PINS) {
                Err((_, bad_sha)) => models.push(ModelBenchResult {
                    name: "xvs_d2".to_string(),
                    loaded: false,
                    status: "SHA256_MISMATCH_REJECTED".to_string(),
                    sha256: bad_sha,
                    load_ms: None,
                    latency_ms: None,
                    output_summary: "SHA-256 pin mismatch; rejected fail-closed".to_string(),
                }),
                Ok((_, sha)) => {
                    let t_load = Instant::now();
                    match build_optimized_ort_session(&env, &d2_path, 1) {
                        Err(e) => models.push(ModelBenchResult {
                            name: "xvs_d2".to_string(),
                            loaded: false,
                            status: format!("ERROR: {e}"),
                            sha256: Some(sha),
                            load_ms: None,
                            latency_ms: None,
                            output_summary: "failed".to_string(),
                        }),
                        Ok(mut s) => {
                            let load_ms = round2(t_load.elapsed().as_secs_f64() * 1000.0);
                            let (in_h, in_w, hid_c, hid_h, hid_w) = resolve_d2_input_shapes(&s);
                            let x_t = vec![0.0_f32; 6 * in_h * in_w];
                            let h_prev = vec![0.0_f32; hid_c * hid_h * hid_w];
                            let run_res = (|| -> ort::Result<(usize, f64)> {
                                // 1 warm-up pass + 1 timed warm pass (separates ORT session init from steady infer_ms)
                                let tx0 = TensorRef::from_array_view((
                                    [1usize, 6, in_h, in_w],
                                    x_t.as_slice(),
                                ))?;
                                let th0 = TensorRef::from_array_view((
                                    [1usize, hid_c, hid_h, hid_w],
                                    h_prev.as_slice(),
                                ))?;
                                let _ = s.run(ort::inputs!["x_t" => tx0, "h_prev_warped" => th0])?;

                                let tx = TensorRef::from_array_view((
                                    [1usize, 6, in_h, in_w],
                                    x_t.as_slice(),
                                ))?;
                                let th = TensorRef::from_array_view((
                                    [1usize, hid_c, hid_h, hid_w],
                                    h_prev.as_slice(),
                                ))?;
                                let t0 = Instant::now();
                                let outs =
                                    s.run(ort::inputs!["x_t" => tx, "h_prev_warped" => th])?;
                                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                                let (_, hm) = outs["hm_logits"].try_extract_tensor::<f32>()?;
                                Ok((hm.len(), dt))
                            })();
                            match run_res {
                                Ok((n, dt)) => models.push(ModelBenchResult {
                                    name: "xvs_d2".to_string(),
                                    loaded: true,
                                    status: "OK".to_string(),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: Some(round2(dt)),
                                    output_summary: format!("hm_logits_len={n} in={in_h}x{in_w}"),
                                }),
                                Err(e) => models.push(ModelBenchResult {
                                    name: "xvs_d2".to_string(),
                                    loaded: false,
                                    status: format!("ERROR: {e}"),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: None,
                                    output_summary: "failed".to_string(),
                                }),
                            }
                        }
                    }
                }
            }
        } else {
            models.push(ModelBenchResult {
                name: "xvs_d2".to_string(),
                loaded: false,
                status: "MODEL_NOT_MOUNTED".to_string(),
                sha256: None,
                load_ms: None,
                latency_ms: None,
                output_summary: "mount xvs_d2_web_fp16.onnx in OPENXVS_MODEL_DIR".to_string(),
            });
        }

        // 3.2 XVS-ThermalNet v5.2 (1.756 MiB FP16 promoted -> v4.5 fallback, SHA-256 pinned fail-closed, 1T pinned)
        if let Some(th_path) = find_model_path(
            model_dir,
            explicit_model_dir,
            &[
                "xvs_thermal_v52_fp16.onnx",
                "xvs_thermal_v52_fp32.onnx",
                "xvs_thermal_v45_fp32.onnx",
                "xvs_thermal_v45_fp16.onnx",
            ],
            &[
                "models/xvs_thermal_v52_fp16.onnx",
                "outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp16.onnx",
                "models/xvs_thermal_v52_fp32.onnx",
                "outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp32.onnx",
                "outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp32.onnx",
                "outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp16.onnx",
                "models/xvs_thermal_v45_fp32.onnx",
                "models/xvs_thermal_v45_fp16.onnx",
            ],
        ) {
            match verify_pinned_onnx_artifact(&th_path, THERMAL_ONNX_PINS) {
                Err((_, bad_sha)) => models.push(ModelBenchResult {
                    name: "xvs_thermal".to_string(),
                    loaded: false,
                    status: "SHA256_MISMATCH_REJECTED".to_string(),
                    sha256: bad_sha,
                    load_ms: None,
                    latency_ms: None,
                    output_summary: "SHA-256 pin mismatch; rejected fail-closed".to_string(),
                }),
                Ok((_, sha)) => {
                    let t_load = Instant::now();
                    match build_optimized_ort_session(&env, &th_path, 1) {
                        Err(e) => models.push(ModelBenchResult {
                            name: "xvs_thermal".to_string(),
                            loaded: false,
                            status: format!("ERROR: {e}"),
                            sha256: Some(sha),
                            load_ms: None,
                            latency_ms: None,
                            output_summary: "failed".to_string(),
                        }),
                        Ok(mut s) => {
                            let load_ms = round2(t_load.elapsed().as_secs_f64() * 1000.0);
                            let x_eo = vec![0.0_f32; 3 * 256 * 320];
                            let x_lw = vec![0.0_f32; 2 * 128 * 160];
                            let q_eo = [0.0_f32];
                            let q_lw = [0.0_f32];
                            let run_res = (|| -> ort::Result<(f64, f64)> {
                                // 1 warm-up pass + 1 timed warm pass
                                let teo0 = TensorRef::from_array_view((
                                    [1usize, 3, 256, 320],
                                    x_eo.as_slice(),
                                ))?;
                                let tlw0 = TensorRef::from_array_view((
                                    [1usize, 2, 128, 160],
                                    x_lw.as_slice(),
                                ))?;
                                let tqe0 = TensorRef::from_array_view(([1usize], q_eo.as_slice()))?;
                                let tql0 = TensorRef::from_array_view(([1usize], q_lw.as_slice()))?;
                                let _ = s.run(ort::inputs![
                                    "x_eo" => teo0,
                                    "x_lwir" => tlw0,
                                    "q_eo" => tqe0,
                                    "q_lwir" => tql0
                                ])?;

                                let teo = TensorRef::from_array_view((
                                    [1usize, 3, 256, 320],
                                    x_eo.as_slice(),
                                ))?;
                                let tlw = TensorRef::from_array_view((
                                    [1usize, 2, 128, 160],
                                    x_lw.as_slice(),
                                ))?;
                                let tqe = TensorRef::from_array_view(([1usize], q_eo.as_slice()))?;
                                let tql = TensorRef::from_array_view(([1usize], q_lw.as_slice()))?;
                                let t0 = Instant::now();
                                let outs = s.run(ort::inputs![
                                    "x_eo" => teo,
                                    "x_lwir" => tlw,
                                    "q_eo" => tqe,
                                    "q_lwir" => tql
                                ])?;
                                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                                let (_, vac) = outs["vacuity"].try_extract_tensor::<f32>()?;
                                let mean_v = vac.iter().map(|&v| v as f64).sum::<f64>()
                                    / (vac.len().max(1) as f64);
                                Ok((mean_v, dt))
                            })();
                            match run_res {
                                Ok((mv, dt)) => models.push(ModelBenchResult {
                                    name: "xvs_thermal".to_string(),
                                    loaded: true,
                                    status: "OK".to_string(),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: Some(round2(dt)),
                                    output_summary: format!("dual_blind_vacuity={mv:.4}"),
                                }),
                                Err(e) => models.push(ModelBenchResult {
                                    name: "xvs_thermal".to_string(),
                                    loaded: false,
                                    status: format!("ERROR: {e}"),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: None,
                                    output_summary: "failed".to_string(),
                                }),
                            }
                        }
                    }
                }
            }
        } else {
            models.push(ModelBenchResult {
                name: "xvs_thermal".to_string(),
                loaded: false,
                status: "MODEL_NOT_MOUNTED".to_string(),
                sha256: None,
                load_ms: None,
                latency_ms: None,
                output_summary: "mount xvs_thermal_v52_fp16.onnx in OPENXVS_MODEL_DIR".to_string(),
            });
        }

        // 3.3 XVS-Land v1.1.2 + StudentTCalibrator (6-output promoted + optional 7th output runway_present_logit, 1T pinned)
        if let Some(land_path) = find_model_path(
            model_dir,
            explicit_model_dir,
            &[
                "xvs_land_v52_fp16.onnx",
                "xvs_land_v52_fp32.onnx",
                "xvs_land_v45_fp16.onnx",
                "xvs_land_v45.onnx",
            ],
            &[
                "models/xvs_land_v52_fp16.onnx",
                "outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp16.onnx",
                "models/xvs_land_v52_fp32.onnx",
                "outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp32.onnx",
                "outputs/colab_l4_v45/harvest_land/xvs_land_v45_fp16.onnx",
                "outputs/colab_l4_v45/harvest_land/xvs_land_v45.onnx",
                "models/xvs_land_v45_fp16.onnx",
                "models/xvs_land_v45.onnx",
            ],
        ) {
            match verify_pinned_onnx_artifact(&land_path, LAND_ONNX_PINS) {
                Err((_, bad_sha)) => models.push(ModelBenchResult {
                    name: "xvs_land".to_string(),
                    loaded: false,
                    status: "SHA256_MISMATCH_REJECTED".to_string(),
                    sha256: bad_sha,
                    load_ms: None,
                    latency_ms: None,
                    output_summary: "SHA-256 pin mismatch; rejected fail-closed".to_string(),
                }),
                Ok((_, sha)) => {
                    let is_v52 = sha.eq_ignore_ascii_case(EXPECTED_LAND_V52_FP16_SHA256)
                        || sha.eq_ignore_ascii_case(EXPECTED_LAND_V52_FP32_SHA256);
                    let (nu, scale) = if is_v52 {
                        (V52_LAND_STUDENT_T_NU, V52_LAND_STUDENT_T_SCALE)
                    } else {
                        (LAND_STUDENT_T_NU, LAND_STUDENT_T_SCALE)
                    };
                    let cal = StudentTCalibrator::new(nu, scale)
                        .expect("valid default StudentTCalibrator");
                    let t_load = Instant::now();
                    match build_optimized_ort_session(&env, &land_path, 1) {
                        Err(e) => models.push(ModelBenchResult {
                            name: "xvs_land".to_string(),
                            loaded: false,
                            status: format!("ERROR: {e}"),
                            sha256: Some(sha),
                            load_ms: None,
                            latency_ms: None,
                            output_summary: "failed".to_string(),
                        }),
                        Ok(mut s) => {
                            let load_ms = round2(t_load.elapsed().as_secs_f64() * 1000.0);
                            let (roi_h, roi_w) = resolve_land_input_hw(&s);
                            let roi = vec![0.5_f32; 3 * roi_h * roi_w];
                            let res: Result<(ParsedLandOutputs, f64), String> = (|| {
                                // 1 warm-up pass + 1 timed warm pass
                                let troi0 = TensorRef::from_array_view((
                                    [1usize, 3, roi_h, roi_w],
                                    roi.as_slice(),
                                ))
                                .map_err(|e| format!("{e}"))?;
                                let _ = s
                                    .run(ort::inputs!["roi_crop" => troi0])
                                    .map_err(|e| format!("{e}"))?;

                                let troi = TensorRef::from_array_view((
                                    [1usize, 3, roi_h, roi_w],
                                    roi.as_slice(),
                                ))
                                .map_err(|e| format!("{e}"))?;
                                let t0 = Instant::now();
                                let outs = s
                                    .run(ort::inputs!["roi_crop" => troi])
                                    .map_err(|e| format!("{e}"))?;
                                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                                let (_, kpts) = outs["keypoints_xy"]
                                    .try_extract_tensor::<f32>()
                                    .map_err(|e| format!("{e}"))?;
                                let (_, kl) = outs["keylines_hesse"]
                                    .try_extract_tensor::<f32>()
                                    .map_err(|e| format!("{e}"))?;
                                let (_, cov) = outs["cov_2x2"]
                                    .try_extract_tensor::<f32>()
                                    .map_err(|e| format!("{e}"))?;
                                let kl_sig_opt = if let Some(sv) = outs.get("keyline_sigma_px") {
                                    let (_, sl) = sv
                                        .try_extract_tensor::<f32>()
                                        .map_err(|e| format!("{e}"))?;
                                    Some(sl)
                                } else {
                                    None
                                };
                                let rwy_pres_opt =
                                    if let Some(rv) = outs.get("runway_present_logit") {
                                        let (_, rl) = rv
                                            .try_extract_tensor::<f32>()
                                            .map_err(|e| format!("{e}"))?;
                                        Some(rl)
                                    } else {
                                        None
                                    };
                                let parsed = parse_xvs_land_outputs_with_presence(
                                    kpts,
                                    kl,
                                    cov,
                                    kl_sig_opt,
                                    rwy_pres_opt,
                                    &cal,
                                )?;
                                Ok((parsed, dt))
                            })();
                            match res {
                                Ok((parsed, dt)) => {
                                    let mut summary = format!(
                                        "keypoints={} cal_sigma0_px={:.2} contract={} eff_kl_sigma_px={:.2}",
                                        parsed.keypoints_xy.len(),
                                        parsed.calibrated_kp_sigmas_px[0],
                                        parsed.contract.as_str(),
                                        parsed.effective_keyline_sigma_px,
                                    );
                                    if let Some(prob) = parsed.runway_present_prob {
                                        summary.push_str(&format!(
                                            " rwy_prob={prob:.3} rwy_gate={}",
                                            parsed.runway_gate_mode
                                        ));
                                    }
                                    models.push(ModelBenchResult {
                                        name: "xvs_land".to_string(),
                                        loaded: true,
                                        status: "OK".to_string(),
                                        sha256: Some(sha),
                                        load_ms: Some(load_ms),
                                        latency_ms: Some(round2(dt)),
                                        output_summary: summary,
                                    });
                                }
                                Err(e) => models.push(ModelBenchResult {
                                    name: "xvs_land".to_string(),
                                    loaded: false,
                                    status: format!("ERROR: {e}"),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: None,
                                    output_summary: "failed".to_string(),
                                }),
                            }
                        }
                    }
                }
            }
        } else {
            models.push(ModelBenchResult {
                name: "xvs_land".to_string(),
                loaded: false,
                status: "MODEL_NOT_MOUNTED".to_string(),
                sha256: None,
                load_ms: None,
                latency_ms: None,
                output_summary: "mount xvs_land_v52_fp16.onnx in OPENXVS_MODEL_DIR".to_string(),
            });
        }

        // 3.4 XVS-Nav v5.0 Stage-1 (Faz-B 21.525 MB FP16 prioritized -> Faz-A fallback; SHA-256 fail-closed, up to 2T)
        if let Some(nav_path) = find_model_path(
            model_dir,
            explicit_model_dir,
            &[
                "xvs_nav_v5_fazb_fp16.onnx",
                "xvs_nav_v5_fp16.onnx",
                "xvs_nav_v5_fazb_fp32.onnx",
                "xvs_nav_v5_fp32.onnx",
            ],
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/xvs_nav_v5_fazb_fp16.onnx",
                "models/xvs_nav_v5_fazb_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest/xvs_nav_v5_fp16.onnx",
                "models/xvs_nav_v5_fp16.onnx",
            ],
        ) {
            match verify_pinned_onnx_artifact(&nav_path, NAV_ONNX_PINS) {
                Err((_, bad_sha)) => models.push(ModelBenchResult {
                    name: "xvs_nav_v5".to_string(),
                    loaded: false,
                    status: "SHA256_MISMATCH_REJECTED".to_string(),
                    sha256: bad_sha,
                    load_ms: None,
                    latency_ms: None,
                    output_summary: "SHA-256 pin mismatch; rejected fail-closed".to_string(),
                }),
                Ok((_, sha)) => {
                    let t_load = Instant::now();
                    match build_optimized_ort_session(&env, &nav_path, 2) {
                        Err(e) => models.push(ModelBenchResult {
                            name: "xvs_nav_v5".to_string(),
                            loaded: false,
                            status: format!("ERROR: {e}"),
                            sha256: Some(sha),
                            load_ms: None,
                            latency_ms: None,
                            output_summary: "failed".to_string(),
                        }),
                        Ok(mut s) => {
                            let load_ms = round2(t_load.elapsed().as_secs_f64() * 1000.0);
                            let img = vec![0.5_f32; 3 * 224 * 224];
                            let res = (|| -> ort::Result<(usize, f64, f64)> {
                                // 1 warm-up pass + 1 timed warm pass
                                let tin0 = TensorRef::from_array_view((
                                    [1usize, 3, 224, 224],
                                    img.as_slice(),
                                ))?;
                                let _ = s.run(ort::inputs!["img_s1" => tin0])?;

                                let tin = TensorRef::from_array_view((
                                    [1usize, 3, 224, 224],
                                    img.as_slice(),
                                ))?;
                                let t0 = Instant::now();
                                let outs = s.run(ort::inputs!["img_s1" => tin])?;
                                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                                let (_, desc) = outs["desc_global"].try_extract_tensor::<f32>()?;
                                let l2 = desc
                                    .iter()
                                    .map(|&v| (v as f64) * (v as f64))
                                    .sum::<f64>()
                                    .sqrt();
                                Ok((desc.len(), l2, dt))
                            })();
                            match res {
                                Ok((dim, l2, dt)) => models.push(ModelBenchResult {
                                    name: "xvs_nav_v5".to_string(),
                                    loaded: true,
                                    status: "READY_RUST_ORT_V5".to_string(),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: Some(round2(dt)),
                                    output_summary: format!("dim={dim} l2_norm={l2:.6}"),
                                }),
                                Err(e) => models.push(ModelBenchResult {
                                    name: "xvs_nav_v5".to_string(),
                                    loaded: false,
                                    status: format!("ERROR: {e}"),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: None,
                                    output_summary: "failed".to_string(),
                                }),
                            }
                        }
                    }
                }
            }
        } else {
            models.push(ModelBenchResult {
                name: "xvs_nav_v5".to_string(),
                loaded: false,
                status: "AWAITING_V5_ONNX_HARVEST".to_string(),
                sha256: None,
                load_ms: None,
                latency_ms: None,
                output_summary: "Plug-in slot ready for xvs_nav_v5_fazb_fp16.onnx".to_string(),
            });
        }

        // 3.5 Stage-2A XFeat (1.303 MB FP16 prioritized -> 2.530 MB FP32 fallback; SHA-256 fail-closed, 1T pinned)
        if let Some(xfeat_path) = find_model_path(
            model_dir,
            explicit_model_dir,
            &["xfeat_320x256_fp16.onnx", "xfeat_320x256_fp32.onnx"],
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp16.onnx",
                "models/xfeat_320x256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp32.onnx",
                "outputs/colab_l4_v5_nav/harvest/xfeat_320x256_fp32.onnx",
                "models/xfeat_320x256_fp32.onnx",
            ],
        ) {
            match verify_pinned_onnx_artifact(&xfeat_path, XFEAT_ONNX_PINS) {
                Err((_, bad_sha)) => models.push(ModelBenchResult {
                    name: "xvs_xfeat".to_string(),
                    loaded: false,
                    status: "SHA256_MISMATCH_REJECTED".to_string(),
                    sha256: bad_sha,
                    load_ms: None,
                    latency_ms: None,
                    output_summary: "SHA-256 pin mismatch; rejected fail-closed".to_string(),
                }),
                Ok((_, sha)) => {
                    let t_load = Instant::now();
                    match build_optimized_ort_session(&env, &xfeat_path, 1) {
                        Err(e) => models.push(ModelBenchResult {
                            name: "xvs_xfeat".to_string(),
                            loaded: false,
                            status: format!("ERROR: {e}"),
                            sha256: Some(sha),
                            load_ms: None,
                            latency_ms: None,
                            output_summary: "failed".to_string(),
                        }),
                        Ok(mut s) => {
                            let load_ms = round2(t_load.elapsed().as_secs_f64() * 1000.0);
                            // Multi-scale textured terrain + sharp corners probe for authentic XFeat keypoints
                            let mut img = vec![0.0_f32; 3 * 256 * 320];
                            let mut pat = vec![0.0_f32; 256 * 320];
                            for y in 0..256usize {
                                let yf = y as f32;
                                for x in 0..320usize {
                                    let xf = x as f32;
                                    let v = (0.5
                                        + 0.25 * (xf * 0.19).sin() * (yf * 0.23).cos()
                                        + 0.20
                                            * ((xf + yf) * 0.37).sin()
                                            * ((xf - yf) * 0.11).cos())
                                    .clamp(0.0, 1.0);
                                    pat[y * 320 + x] = v;
                                }
                            }
                            for r in (20..236usize).step_by(28) {
                                for c in (20..300usize).step_by(32) {
                                    for dy in 0..10usize {
                                        for dx in 0..10usize {
                                            let inner =
                                                (2..8).contains(&dy) && (2..8).contains(&dx);
                                            pat[(r + dy) * 320 + (c + dx)] =
                                                if inner { 0.05 } else { 0.95 };
                                        }
                                    }
                                }
                            }
                            for ch in 0..3usize {
                                img[ch * 256 * 320..(ch + 1) * 256 * 320].copy_from_slice(&pat);
                            }
                            let res = (|| -> ort::Result<(usize, f64)> {
                                // 1 warm-up pass + 1 timed warm pass
                                let tin0 = TensorRef::from_array_view((
                                    [1usize, 3, 256, 320],
                                    img.as_slice(),
                                ))?;
                                let _ = s.run(ort::inputs!["img_s2" => tin0])?;

                                let tin = TensorRef::from_array_view((
                                    [1usize, 3, 256, 320],
                                    img.as_slice(),
                                ))?;
                                let t0 = Instant::now();
                                let outs = s.run(ort::inputs!["img_s2" => tin])?;
                                let (_, kpts) = outs["keypoints"].try_extract_tensor::<f32>()?;
                                let (_, hm) = outs["heatmap"].try_extract_tensor::<f32>()?;
                                let n_kpts = count_xfeat_keypoints(kpts, hm, 256, 0.05);
                                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                                Ok((n_kpts, dt))
                            })();
                            match res {
                                Ok((nk, dt)) => models.push(ModelBenchResult {
                                    name: "xvs_xfeat".to_string(),
                                    loaded: true,
                                    status: "READY_RUST_ORT_V5".to_string(),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: Some(round2(dt)),
                                    output_summary: format!("xfeat_keypoints_count={nk}"),
                                }),
                                Err(e) => models.push(ModelBenchResult {
                                    name: "xvs_xfeat".to_string(),
                                    loaded: false,
                                    status: format!("ERROR: {e}"),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: None,
                                    output_summary: "failed".to_string(),
                                }),
                            }
                        }
                    }
                }
            }
        } else {
            models.push(ModelBenchResult {
                name: "xvs_xfeat".to_string(),
                loaded: false,
                status: "MODEL_NOT_MOUNTED".to_string(),
                sha256: None,
                load_ms: None,
                latency_ms: None,
                output_summary: "mount xfeat_320x256_fp16.onnx in OPENXVS_MODEL_DIR".to_string(),
            });
        }

        // 3.6 Stage-2B LighterGlue (2.455 MB FP16 static K=256 prioritized -> FP32 fallback; SHA-256 fail-closed, 1T pinned)
        if let Some(lg_path) = find_model_path(
            model_dir,
            explicit_model_dir,
            &["lighterglue_k256_fp16.onnx", "lighterglue_k256_fp32.onnx"],
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp16.onnx",
                "models/lighterglue_k256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp32.onnx",
                "models/lighterglue_k256_fp32.onnx",
            ],
        ) {
            match verify_pinned_onnx_artifact(&lg_path, LIGHTERGLUE_ONNX_PINS) {
                Err((_, bad_sha)) => models.push(ModelBenchResult {
                    name: "xvs_lighterglue".to_string(),
                    loaded: false,
                    status: "SHA256_MISMATCH_REJECTED".to_string(),
                    sha256: bad_sha,
                    load_ms: None,
                    latency_ms: None,
                    output_summary: "SHA-256 pin mismatch; rejected fail-closed".to_string(),
                }),
                Ok((_, sha)) => {
                    let t_load = Instant::now();
                    match build_optimized_ort_session(&env, &lg_path, 1) {
                        Err(e) => models.push(ModelBenchResult {
                            name: "xvs_lighterglue".to_string(),
                            loaded: false,
                            status: format!("ERROR: {e}"),
                            sha256: Some(sha),
                            load_ms: None,
                            latency_ms: None,
                            output_summary: "failed".to_string(),
                        }),
                        Ok(mut s) => {
                            let load_ms = round2(t_load.elapsed().as_secs_f64() * 1000.0);
                            // Deterministic 256-keypoint geometric grid + distinct L2-normalized 64-D descriptors
                            let k = LIGHTERGLUE_STATIC_K;
                            let d = LIGHTERGLUE_DESC_DIM;
                            let mut kpts = vec![0.0_f32; k * 2];
                            let mut desc = vec![0.0_f32; k * d];
                            for i in 0..k {
                                let gx = (i % 16) as f32;
                                let gy = (i / 16) as f32;
                                kpts[i * 2] = -0.9 + (gx / 15.0) * 1.8;
                                kpts[i * 2 + 1] = -0.9 + (gy / 15.0) * 1.8;
                                let mut nsq = 0.0_f32;
                                for c in 0..d {
                                    let phase = ((i as f32) * 0.37 + (c as f32) * 0.71).sin()
                                        + 0.5 * ((i as f32) * 1.13 - (c as f32) * 0.29).cos();
                                    desc[i * d + c] = phase;
                                    nsq += phase * phase;
                                }
                                let inv = 1.0_f32 / nsq.sqrt().max(1e-12);
                                for c in 0..d {
                                    desc[i * d + c] *= inv;
                                }
                            }
                            let res = (|| -> ort::Result<(Vec<i64>, bool, usize, f64, f64)> {
                                // 1 warm-up pass + 1 timed warm pass
                                let tk0_w = TensorRef::from_array_view(([1usize, k, 2], kpts.as_slice()))?;
                                let tk1_w = TensorRef::from_array_view(([1usize, k, 2], kpts.as_slice()))?;
                                let td0_w = TensorRef::from_array_view(([1usize, k, d], desc.as_slice()))?;
                                let td1_w = TensorRef::from_array_view(([1usize, k, d], desc.as_slice()))?;
                                let _ = s.run(ort::inputs![
                                    "kpts0" => tk0_w,
                                    "kpts1" => tk1_w,
                                    "desc0" => td0_w,
                                    "desc1" => td1_w
                                ])?;

                                let tk0 = TensorRef::from_array_view(([1usize, k, 2], kpts.as_slice()))?;
                                let tk1 = TensorRef::from_array_view(([1usize, k, 2], kpts.as_slice()))?;
                                let td0 = TensorRef::from_array_view(([1usize, k, d], desc.as_slice()))?;
                                let td1 = TensorRef::from_array_view(([1usize, k, d], desc.as_slice()))?;
                                let t0 = Instant::now();
                                let outs = s.run(ort::inputs![
                                    "kpts0" => tk0,
                                    "kpts1" => tk1,
                                    "desc0" => td0,
                                    "desc1" => td1
                                ])?;
                                let (shape, scores) = outs["scores"].try_extract_tensor::<f32>()?;
                                let dims: Vec<i64> = shape.iter().copied().collect();
                                let all_finite = scores.iter().all(|v| v.is_finite());
                                let (n_matches, mean_p) = count_lighterglue_mutual_matches(
                                    scores,
                                    LIGHTERGLUE_MATCH_MIN_PROB,
                                );
                                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                                Ok((dims, all_finite, n_matches, mean_p, dt))
                            })();
                            match res {
                                Ok((dims, true, nm, mp, dt))
                                    if dims
                                        == [
                                            1,
                                            LIGHTERGLUE_STATIC_K as i64,
                                            LIGHTERGLUE_STATIC_K as i64,
                                        ] =>
                                {
                                    models.push(ModelBenchResult {
                                        name: "xvs_lighterglue".to_string(),
                                        loaded: true,
                                        status: "READY_RUST_ORT_V5".to_string(),
                                        sha256: Some(sha),
                                        load_ms: Some(load_ms),
                                        latency_ms: Some(round2(dt)),
                                        output_summary: format!(
                                            "scores_shape=1x256x256 mutual_matches={nm} mean_prob={mp:.4}"
                                        ),
                                    });
                                }
                                Ok((dims, finite, _, _, _)) => models.push(ModelBenchResult {
                                    name: "xvs_lighterglue".to_string(),
                                    loaded: false,
                                    status: format!(
                                        "INVALID_OUTPUT_SHAPE_OR_NAN: dims={dims:?} finite={finite}"
                                    ),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: None,
                                    output_summary: "failed".to_string(),
                                }),
                                Err(e) => models.push(ModelBenchResult {
                                    name: "xvs_lighterglue".to_string(),
                                    loaded: false,
                                    status: format!("ERROR: {e}"),
                                    sha256: Some(sha),
                                    load_ms: Some(load_ms),
                                    latency_ms: None,
                                    output_summary: "failed".to_string(),
                                }),
                            }
                        }
                    }
                }
            }
        } else {
            models.push(ModelBenchResult {
                name: "xvs_lighterglue".to_string(),
                loaded: false,
                status: "MODEL_NOT_MOUNTED".to_string(),
                sha256: None,
                load_ms: None,
                latency_ms: None,
                output_summary: "mount lighterglue_k256_fp16.onnx in OPENXVS_MODEL_DIR"
                    .to_string(),
            });
        }
    }

    // 2.5 Pure-Rust v5.2.0 Optics, 4x4/12-Sector SensorHealth, FFC/Contrast-Collapse, FSYNC/SLERP, 3-of-5 TBC check
    let bc_err = PinholeIntrinsics::new(
        1280.0, 1280.0, 640.0, 360.0, -0.24, 0.08, 0.0012, -0.0009, -0.015,
    )
    .and_then(|c| {
        let p = [1120.0, 620.0];
        let d = c.distort_pixel(p)?;
        let u = c.undistort_pixel_exact(d)?;
        Ok((u[0] - p[0]).hypot(u[1] - p[1]))
    })
    .unwrap_or(f64::INFINITY);

    let kb_err = KannalaBrandtFisheye::new(
        420.0, 420.0, 640.0, 360.0, -0.025, 0.008, -0.002, 0.0004,
    )
    .and_then(|c| {
        let theta = 70.0_f64.to_radians();
        let ray = [theta.sin() * 0.8, theta.sin() * 0.6, theta.cos()];
        let d = c.project_ray_rdf(ray)?;
        let ur = c.undistort_to_unit_ray_exact(d)?;
        let re = c.project_ray_rdf(ur)?;
        Ok((re[0] - d[0]).hypot(re[1] - d[1]))
    })
    .unwrap_or(f64::INFINITY);

    let mut shm = SensorHealthMonitor::new();
    let mut luma_glint = vec![0u8; 160 * 128];
    for y in 0..128usize {
        for x in 0..160usize {
            let v = if (((x / 2) + (y / 2)) % 2) == 0 { 190u8 } else { 65u8 };
            luma_glint[y * 160 + x] = v;
        }
    }
    for y in 32..64usize {
        for x in 120..133usize {
            luma_glint[y * 160 + x] = 255;
        }
    }
    let eo_q = shm.observe_eo_luma_u8(&luma_glint, 160, 128);
    let spatial_ok = eo_q.q_eo_optical >= 0.95
        && eo_q.sector_q_eo[9] < 0.05
        && (eo_q.degraded_sector_mask & (1u16 << 9)) != 0;

    let noisy_a: Vec<u8> = (0..(160 * 128)).map(|i| 128 + ((i % 5) as u8)).collect();
    let noisy_b: Vec<u8> = (0..(160 * 128)).map(|i| 131 + (((i + 2) % 5) as u8)).collect();
    let st_ffc = shm.observe_lwir_frame_with_ffc(&noisy_a, LWIR_FFC_BIT_IMMINENT);
    let (_, ql_ffc) = shm.compute_effective_qualities(0.95, 0.95);
    let st_cc = shm.observe_lwir_frame_with_ffc(&noisy_b, 0);
    let (_, ql_cc) = shm.compute_effective_qualities(0.95, 0.95);
    let ffc_cc_ok =
        st_ffc.mask_lwir_frame && ql_ffc == 0.0 && st_cc.contrast_collapsed && ql_cc == 0.0;

    let t_ok = shm.observe_timestamps(1000.0, 999.5);
    let t_warn = shm.observe_timestamps(1100.0, 1096.0);
    let t_fault = shm.observe_timestamps(1200.0, 950.0);
    let slerp_ok = interpolate_imu_attitude_at_exposure_midpoint(
        1000.0,
        [1.0, 0.0, 0.0, 0.0],
        1010.0,
        [
            core::f64::consts::FRAC_PI_4.cos(),
            0.0,
            0.0,
            core::f64::consts::FRAC_PI_4.sin(),
        ],
        1002.0,
        6.0,
    )
    .map(|(q, tm)| {
        (tm - 1005.0).abs() < 1e-12
            && (q[0] - core::f64::consts::FRAC_PI_8.cos()).abs() < 1e-12
            && (q[3] - core::f64::consts::FRAC_PI_8.sin()).abs() < 1e-12
    })
    .unwrap_or(false);
    let fsync_slerp_ok = t_ok.fsync_precision_ok
        && !t_warn.skew_fault
        && !t_warn.fsync_precision_ok
        && t_fault.skew_fault
        && slerp_ok;

    let mut tbc = BearingTrackBeforeConfirm::new_3_of_5();
    let _ = tbc.observe(Some((0.10, 0.01)));
    let _ = tbc.observe(None);
    let _ = tbc.observe(Some((0.105, 0.011)));
    let _ = tbc.observe(None);
    let tbc_confirmed = tbc.observe(Some((0.109, 0.012))).confirmed;

    EdgeSelfCheckReport {
        version: EDGE_INFER_VERSION,
        non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
        actuator_authority: ACTUATOR_AUTHORITY,
        watermark: NON_FLIGHT_WATERMARK,
        target_arch: std::env::consts::ARCH,
        target_os: std::env::consts::OS,
        ort_dylib_loaded: ort_loaded,
        odd_yaml_sha256: ODD_YAML_SHA256,
        rta_mode: rta_out.mode as u8,
        active_range_wire_crc_ok: ar_wire_ok,
        active_range_r_hat_m: ar_pkt.r_hat_m,
        multiship_coord: MultiShipCoordCheck {
            enabled: true,
            f1_far91113_right_turn_ok: f1_ok,
            f3_ring0_p0_zero_bank_ok: f3_ok,
            f4_sticky_triple_ray_ok: f4_ok,
            f4_sticky_ghosts_suppressed: g2.sticky_ghosts_suppressed,
            f4_stale_obs_rejected: g_stale.stale_obs_rejected,
        },
        sensor_optics_hardening: SensorOpticsHardeningCheck {
            brown_conrady_max_err_px: bc_err,
            kannala_brandt_140deg_max_err_px: kb_err,
            spatial_4x4_12sector_localized_fault_ok: spatial_ok,
            lwir_ffc_zero_tick_and_contrast_collapse_ok: ffc_cc_ok,
            dual_tier_fsync_and_slerp_ok: fsync_slerp_ok,
            bearing_tbc_3_of_5_ok: tbc_confirmed,
        },
        models,
    }
}

pub(crate) fn resolve_default_model_dir() -> (PathBuf, bool) {
    match std::env::var("OPENXVS_MODEL_DIR") {
        Ok(raw) if !raw.trim().is_empty() => (PathBuf::from(raw.trim()), true),
        _ => (PathBuf::from("/opt/openxvs/models"), false),
    }
}

#[cfg(test)]
fn run_self_check() -> EdgeSelfCheckReport {
    let (model_dir, explicit) = resolve_default_model_dir();
    run_self_check_with_model_dir(&model_dir, explicit)
}

// ============================================================================
// Headless CLI Mode 1: `--soak --seconds <N> --hz <F>` (Thermal-Safe Paced ORT Soak)
// ============================================================================

#[derive(Debug, Serialize)]
struct EdgeSoakReport {
    mode: &'static str,
    version: &'static str,
    non_flight_sim_only: bool,
    actuator_authority: bool,
    watermark: &'static str,
    duration_requested_s: f64,
    duration_actual_s: f64,
    target_hz: f64,
    models_loaded: Vec<String>,
    fast_ticks: u64,
    slow_ticks: u64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    slow_p50_ms: f64,
    slow_p95_ms: f64,
    slow_max_ms: f64,
    non_finite_count: u64,
    rss_start_mib: f64,
    rss_end_mib: f64,
    rss_delta_mib: f64,
    deadline_misses: u64,
    pass: bool,
}

fn read_process_rss_mib() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
                    if let Ok(kib) = digits.parse::<u64>() {
                        return round2((kib as f64) / 1024.0);
                    }
                }
            }
        }
        0.0
    }
    #[cfg(target_os = "windows")]
    {
        let pid = std::process::id();
        let filter = format!("PID eq {pid}");
        if let Ok(out) = std::process::Command::new("tasklist")
            .args(["/FI", &filter, "/FO", "CSV", "/NH"])
            .output()
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Some(line) = text.lines().find(|l| !l.trim().is_empty()) {
                    // Last CSV field is memory usage in KiB (e.g. `"65,780 K"` or `"65.780 K"`)
                    if let Some(mem_col) = line.rsplit(',').next() {
                        let digits: String =
                            mem_col.chars().filter(|c| c.is_ascii_digit()).collect();
                        if let Ok(kib) = digits.parse::<u64>() {
                            return round2((kib as f64) / 1024.0);
                        }
                    }
                }
            }
        }
        0.0
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        0.0
    }
}

pub(crate) fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64) * q).ceil() as usize;
    sorted[idx.saturating_sub(1).min(sorted.len() - 1)]
}

fn load_pinned_session_for_soak(
    env: &Environment,
    model_dir: &Path,
    explicit: bool,
    direct_names: &[&str],
    repo_rel_candidates: &[&str],
    pins: &[(&str, &str)],
    max_intra_threads: usize,
) -> Option<(Session, String)> {
    let path = find_model_path(model_dir, explicit, direct_names, repo_rel_candidates)?;
    let (_, sha) = verify_pinned_onnx_artifact(&path, pins).ok()?;
    let sess = build_optimized_ort_session(env, &path, max_intra_threads).ok()?;
    Some((sess, sha))
}

fn run_soak_mode(seconds: f64, hz: f64, model_dir_opt: Option<PathBuf>) -> Result<EdgeSoakReport, String> {
    if !(seconds.is_finite() && seconds > 0.0) {
        return Err(format!("--seconds must be finite and > 0, got {seconds}"));
    }
    if !(hz.is_finite() && hz > 0.0 && hz <= 200.0) {
        return Err(format!("--hz must be finite in (0, 200], got {hz}"));
    }
    let (model_dir, explicit) = match model_dir_opt {
        Some(p) => (p, true),
        None => resolve_default_model_dir(),
    };

    let mut models_loaded: Vec<String> = Vec::new();
    let mut d2_sess: Option<Session> = None;
    let mut d2_shapes = (D2_CROP_SIZE, D2_CROP_SIZE, D2_HID_C, D2_HID_HW, D2_HID_HW);
    let mut thermal_sess: Option<Session> = None;
    let mut land_sess: Option<(Session, usize, usize, StudentTCalibrator)> = None;
    let mut nav_sess: Option<Session> = None;
    let mut xfeat_sess: Option<Session> = None;
    let mut lg_sess: Option<Session> = None;

    if let Ok(env) = acquire_edge_ort_env() {
        if let Some((s, _)) = load_pinned_session_for_soak(
            &env,
            &model_dir,
            explicit,
            &["xvs_d2_web_fp16.onnx"],
            &[
                "apps/livedeck/public/models/xvs_d2_web_fp16.onnx",
                "models/xvs_d2_web_fp16.onnx",
            ],
            D2_ONNX_PINS,
            1,
        ) {
            d2_shapes = resolve_d2_input_shapes(&s);
            d2_sess = Some(s);
            models_loaded.push("xvs_d2".to_string());
        }
        if let Some((s, _)) = load_pinned_session_for_soak(
            &env,
            &model_dir,
            explicit,
            &[
                "xvs_thermal_v52_fp16.onnx",
                "xvs_thermal_v52_fp32.onnx",
                "xvs_thermal_v45_fp32.onnx",
                "xvs_thermal_v45_fp16.onnx",
            ],
            &[
                "models/xvs_thermal_v52_fp16.onnx",
                "outputs/colab_g4_v52_mega/harvest/thermal_v52/xvs_thermal_v52_fp16.onnx",
                "outputs/colab_l4_v45/harvest_thermal/xvs_thermal_v45_fp32.onnx",
                "models/xvs_thermal_v45_fp32.onnx",
            ],
            THERMAL_ONNX_PINS,
            1,
        ) {
            thermal_sess = Some(s);
            models_loaded.push("xvs_thermal".to_string());
        }
        if let Some((s, sha)) = load_pinned_session_for_soak(
            &env,
            &model_dir,
            explicit,
            &[
                "xvs_land_v52_fp16.onnx",
                "xvs_land_v52_fp32.onnx",
                "xvs_land_v45_fp16.onnx",
                "xvs_land_v45.onnx",
            ],
            &[
                "models/xvs_land_v52_fp16.onnx",
                "outputs/colab_g4_v52_mega/harvest/land_v52/xvs_land_v52_fp16.onnx",
                "outputs/colab_l4_v45/harvest_land/xvs_land_v45_fp16.onnx",
                "models/xvs_land_v45_fp16.onnx",
            ],
            LAND_ONNX_PINS,
            1,
        ) {
            let (rh, rw) = resolve_land_input_hw(&s);
            let is_v52 = sha.eq_ignore_ascii_case(EXPECTED_LAND_V52_FP16_SHA256)
                || sha.eq_ignore_ascii_case(EXPECTED_LAND_V52_FP32_SHA256);
            let (nu, scale) = if is_v52 {
                (V52_LAND_STUDENT_T_NU, V52_LAND_STUDENT_T_SCALE)
            } else {
                (LAND_STUDENT_T_NU, LAND_STUDENT_T_SCALE)
            };
            let cal = StudentTCalibrator::new(nu, scale).expect("valid StudentTCalibrator");
            land_sess = Some((s, rh, rw, cal));
            models_loaded.push("xvs_land".to_string());
        }
        if let Some((s, _)) = load_pinned_session_for_soak(
            &env,
            &model_dir,
            explicit,
            &[
                "xvs_nav_v5_fazb_fp16.onnx",
                "xvs_nav_v5_fp16.onnx",
                "xvs_nav_v5_fazb_fp32.onnx",
                "xvs_nav_v5_fp32.onnx",
            ],
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/xvs_nav_v5_fazb_fp16.onnx",
                "models/xvs_nav_v5_fazb_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest/xvs_nav_v5_fp16.onnx",
                "models/xvs_nav_v5_fp16.onnx",
            ],
            NAV_ONNX_PINS,
            2,
        ) {
            nav_sess = Some(s);
            models_loaded.push("xvs_nav_v5".to_string());
        }
        if let Some((s, _)) = load_pinned_session_for_soak(
            &env,
            &model_dir,
            explicit,
            &["xfeat_320x256_fp16.onnx", "xfeat_320x256_fp32.onnx"],
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp16.onnx",
                "models/xfeat_320x256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/xfeat_320x256_fp32.onnx",
                "models/xfeat_320x256_fp32.onnx",
            ],
            XFEAT_ONNX_PINS,
            1,
        ) {
            xfeat_sess = Some(s);
            models_loaded.push("xvs_xfeat".to_string());
        }
        if let Some((s, _)) = load_pinned_session_for_soak(
            &env,
            &model_dir,
            explicit,
            &["lighterglue_k256_fp16.onnx", "lighterglue_k256_fp32.onnx"],
            &[
                "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp16.onnx",
                "models/lighterglue_k256_fp16.onnx",
                "outputs/colab_l4_v5_nav/harvest_faz_b/lighterglue_k256_fp32.onnx",
                "models/lighterglue_k256_fp32.onnx",
            ],
            LIGHTERGLUE_ONNX_PINS,
            1,
        ) {
            lg_sess = Some(s);
            models_loaded.push("xvs_lighterglue".to_string());
        }
    }

    // Pre-allocate reusable input tensors outside soak loop (zero per-tick heap churn)
    let (d2_in_h, d2_in_w, d2_hid_c, d2_hid_h, d2_hid_w) = d2_shapes;
    let d2_x = vec![0.15_f32; 6 * d2_in_h * d2_in_w];
    let mut d2_h = vec![0.0_f32; d2_hid_c * d2_hid_h * d2_hid_w];
    let th_eo = vec![0.35_f32; 3 * 256 * 320];
    let th_lw = vec![0.40_f32; 2 * 128 * 160];
    let th_qe = [0.85_f32];
    let th_ql = [0.95_f32];
    let nav_img = vec![0.45_f32; 3 * 224 * 224];
    let mut xfeat_img = vec![0.0_f32; 3 * 256 * 320];
    for y in 0..256usize {
        for x in 0..320usize {
            let v = (0.5 + 0.3 * ((x as f32) * 0.21).sin() * ((y as f32) * 0.19).cos()).clamp(0.0, 1.0);
            for ch in 0..3usize {
                xfeat_img[ch * 256 * 320 + y * 320 + x] = v;
            }
        }
    }
    let k = LIGHTERGLUE_STATIC_K;
    let d = LIGHTERGLUE_DESC_DIM;
    let mut lg_kpts = vec![0.0_f32; k * 2];
    let mut lg_desc = vec![0.0_f32; k * d];
    for i in 0..k {
        lg_kpts[i * 2] = -0.9 + ((i % 16) as f32 / 15.0) * 1.8;
        lg_kpts[i * 2 + 1] = -0.9 + ((i / 16) as f32 / 15.0) * 1.8;
        let mut nsq = 0.0_f32;
        for c in 0..d {
            let val = ((i as f32) * 0.37 + (c as f32) * 0.71).sin();
            lg_desc[i * d + c] = val;
            nsq += val * val;
        }
        let inv = 1.0_f32 / nsq.sqrt().max(1e-12);
        for c in 0..d {
            lg_desc[i * d + c] *= inv;
        }
    }
    let land_roi: Vec<f32> = land_sess
        .as_ref()
        .map(|(_, rh, rw, _)| vec![0.5_f32; 3 * (*rh) * (*rw)])
        .unwrap_or_default();

    // Stateful pure-Rust avionics modules
    let mut coord = CoordinatedActiveRange::new(1, ActiveRangeConfig::default());
    let mut rta = SimplexRta::new();
    let mut cusum = DualBankEsEkfCusum::new(DualBankConfig {
        sample_rate_hz: hz,
        ..DualBankConfig::default()
    })?;
    let mut gallery_coords = Vec::with_capacity(16);
    for r in 0..4usize {
        for c in 0..4usize {
            gallery_coords.push([-(r as f64) * 150.0, (c as f64) * 150.0]);
        }
    }
    let mut seq_vpr = SequenceVprFilter::new(gallery_coords.clone(), SequenceVprConfig::default())?;
    let own = OwnshipStateNed {
        v_ned: [35.0, 0.0, 0.0],
        a_ned: [0.0, 0.0, 0.0],
        heading_rad: 0.0,
        tas_mps: 35.0,
        v_stall_mps: 14.5,
    };

    // Warm up ALL 6 loaded sessions for 2 passes BEFORE RSS start snapshot:
    // Pass 1 records ORT MemoryPattern; Pass 2 commits the contiguous BFC arena so steady-state
    // soak ticks run with ~0 MiB RSS growth and zero cold allocation spikes on tick 1.
    for _ in 0..2 {
        if let Some(s) = d2_sess.as_mut() {
            if let (Ok(tx), Ok(th)) = (
                TensorRef::from_array_view(([1usize, 6, d2_in_h, d2_in_w], d2_x.as_slice())),
                TensorRef::from_array_view((
                    [1usize, d2_hid_c, d2_hid_h, d2_hid_w],
                    d2_h.as_slice(),
                )),
            ) {
                let _ = s.run(ort::inputs!["x_t" => tx, "h_prev_warped" => th]);
            }
        }
        if let Some(s) = thermal_sess.as_mut() {
            if let (Ok(teo), Ok(tlw), Ok(tqe), Ok(tql)) = (
                TensorRef::from_array_view(([1usize, 3, 256, 320], th_eo.as_slice())),
                TensorRef::from_array_view(([1usize, 2, 128, 160], th_lw.as_slice())),
                TensorRef::from_array_view(([1usize], th_qe.as_slice())),
                TensorRef::from_array_view(([1usize], th_ql.as_slice())),
            ) {
                let _ = s.run(
                    ort::inputs!["x_eo" => teo, "x_lwir" => tlw, "q_eo" => tqe, "q_lwir" => tql],
                );
            }
        }
        if let Some((ref mut s, rh, rw, _)) = land_sess {
            if let Ok(troi) = TensorRef::from_array_view(([1usize, 3, rh, rw], land_roi.as_slice()))
            {
                let _ = s.run(ort::inputs!["roi_crop" => troi]);
            }
        }
        if let Some(s) = nav_sess.as_mut() {
            if let Ok(tin) =
                TensorRef::from_array_view(([1usize, 3, 224, 224], nav_img.as_slice()))
            {
                let _ = s.run(ort::inputs!["img_s1" => tin]);
            }
        }
        if let Some(s) = xfeat_sess.as_mut() {
            if let Ok(tin) =
                TensorRef::from_array_view(([1usize, 3, 256, 320], xfeat_img.as_slice()))
            {
                let _ = s.run(ort::inputs!["img_s2" => tin]);
            }
        }
        if let Some(s) = lg_sess.as_mut() {
            if let (Ok(tk0), Ok(tk1), Ok(td0), Ok(td1)) = (
                TensorRef::from_array_view(([1usize, k, 2], lg_kpts.as_slice())),
                TensorRef::from_array_view(([1usize, k, 2], lg_kpts.as_slice())),
                TensorRef::from_array_view(([1usize, k, d], lg_desc.as_slice())),
                TensorRef::from_array_view(([1usize, k, d], lg_desc.as_slice())),
            ) {
                let _ = s.run(
                    ort::inputs!["kpts0" => tk0, "kpts1" => tk1, "desc0" => td0, "desc1" => td1],
                );
            }
        }
    }

    let rss_start_mib = read_process_rss_mib();
    let period = Duration::from_secs_f64(1.0 / hz);
    let period_ms = 1000.0 / hz;
    let slow_stride = (hz.round() as u64).max(1);
    let target_dur = Duration::from_secs_f64(seconds);

    let mut fast_ticks: u64 = 0;
    let mut slow_ticks: u64 = 0;
    let mut non_finite_count: u64 = 0;
    let mut deadline_misses: u64 = 0;
    let mut fast_ms_samples: Vec<f64> = Vec::with_capacity((seconds * hz) as usize + 16);
    let mut slow_ms_samples: Vec<f64> = Vec::with_capacity((seconds as usize) + 8);

    let soak_start = Instant::now();
    while soak_start.elapsed() < target_dur {
        let tick_t0 = Instant::now();
        fast_ticks += 1;

        // Fast path: D2 + Thermal + CoordinatedActiveRange + SimplexRta + DualBankEsEkfCusum
        if let Some(s) = d2_sess.as_mut() {
            let tx = TensorRef::from_array_view(([1usize, 6, d2_in_h, d2_in_w], d2_x.as_slice()))
                .map_err(|e| format!("soak d2 tx: {e}"))?;
            let th = TensorRef::from_array_view((
                [1usize, d2_hid_c, d2_hid_h, d2_hid_w],
                d2_h.as_slice(),
            ))
            .map_err(|e| format!("soak d2 th: {e}"))?;
            let outs = s
                .run(ort::inputs!["x_t" => tx, "h_prev_warped" => th])
                .map_err(|e| format!("soak d2 run: {e}"))?;
            let (_, hm) = outs["hm_logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("soak d2 hm: {e}"))?;
            if !hm.iter().all(|v| v.is_finite()) {
                non_finite_count += 1;
            }
            if let Some(hv) = outs.get("h_next") {
                if let Ok((_, h_next)) = hv.try_extract_tensor::<f32>() {
                    if h_next.len() == d2_h.len() && h_next.iter().all(|v| v.is_finite()) {
                        d2_h.copy_from_slice(h_next);
                    } else {
                        non_finite_count += 1;
                    }
                }
            }
        }
        if let Some(s) = thermal_sess.as_mut() {
            let teo = TensorRef::from_array_view(([1usize, 3, 256, 320], th_eo.as_slice()))
                .map_err(|e| format!("soak th eo: {e}"))?;
            let tlw = TensorRef::from_array_view(([1usize, 2, 128, 160], th_lw.as_slice()))
                .map_err(|e| format!("soak th lw: {e}"))?;
            let tqe = TensorRef::from_array_view(([1usize], th_qe.as_slice()))
                .map_err(|e| format!("soak th qe: {e}"))?;
            let tql = TensorRef::from_array_view(([1usize], th_ql.as_slice()))
                .map_err(|e| format!("soak th ql: {e}"))?;
            let outs = s
                .run(ort::inputs!["x_eo" => teo, "x_lwir" => tlw, "q_eo" => tqe, "q_lwir" => tql])
                .map_err(|e| format!("soak th run: {e}"))?;
            let (_, vac) = outs["vacuity"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("soak th vac: {e}"))?;
            if !vac.iter().all(|v| v.is_finite()) {
                non_finite_count += 1;
            }
        }

        let meas = ActiveRangeMeasurement {
            alpha_rad: 0.0,
            beta_rad: 0.0,
            size_px: 14.0,
            size_rate_px_s: 1.5,
            contrast_ratio: 0.65,
            sky_beta_ext: 2.0e-4,
            daytime_clear: true,
            track_age: (fast_ticks as u32).max(1),
            dt_s: 1.0 / hz,
        };
        let ar_pkt = coord.step(&own, &meas, None, Ring0AuthorityGate::NOMINAL);
        if !ar_pkt.r_hat_m.is_finite() || decode_wire(&encode_wire(&ar_pkt)).is_none() {
            non_finite_count += 1;
        }
        let t_s = (fast_ticks as f64) / hz;
        let c_out = cusum.step(&DualBankStepInput {
            baro_down_m: -120.0,
            gnss_ne_m: Some([15.0 * t_s, 0.0]),
            vis_ne_m: Some([15.0 * t_s, 0.0]),
            vis_gate_admitted: true,
            gnss_down_m: Some(-120.0),
            ..DualBankStepInput::default()
        });
        if !(c_out.max_cusum_stat.is_finite()
            && c_out.vertical_cusum_stat.is_finite()
            && c_out.bank_a_p_ne_m.iter().all(|v| v.is_finite())
            && c_out.bank_b_p_ne_m.iter().all(|v| v.is_finite()))
        {
            non_finite_count += 1;
        }
        let _rta_out = rta.step(&RtaTickInput::default());

        let fast_dt_ms = tick_t0.elapsed().as_secs_f64() * 1000.0;
        fast_ms_samples.push(fast_dt_ms);

        // Slow 1 Hz path: Land + Nav S1 + XFeat + LighterGlue + SequenceVprFilter
        if (fast_ticks - 1).is_multiple_of(slow_stride) {
            let slow_t0 = Instant::now();
            slow_ticks += 1;
            if let Some((ref mut s, rh, rw, ref cal)) = land_sess {
                let troi = TensorRef::from_array_view(([1usize, 3, rh, rw], land_roi.as_slice()))
                    .map_err(|e| format!("soak land troi: {e}"))?;
                let outs = s
                    .run(ort::inputs!["roi_crop" => troi])
                    .map_err(|e| format!("soak land run: {e}"))?;
                let (_, kpts) = outs["keypoints_xy"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("soak land kpts: {e}"))?;
                let (_, kl) = outs["keylines_hesse"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("soak land kl: {e}"))?;
                let (_, cov) = outs["cov_2x2"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("soak land cov: {e}"))?;
                let kl_sig_opt = if let Some(sv) = outs.get("keyline_sigma_px") {
                    Some(
                        sv.try_extract_tensor::<f32>()
                            .map_err(|e| format!("soak land sig: {e}"))?
                            .1,
                    )
                } else {
                    None
                };
                let rwy_pres_opt = if let Some(rv) = outs.get("runway_present_logit") {
                    Some(
                        rv.try_extract_tensor::<f32>()
                            .map_err(|e| format!("soak land rwy_present: {e}"))?
                            .1,
                    )
                } else {
                    None
                };
                if parse_xvs_land_outputs_with_presence(
                    kpts,
                    kl,
                    cov,
                    kl_sig_opt,
                    rwy_pres_opt,
                    cal,
                )
                .is_err()
                {
                    non_finite_count += 1;
                }
            }
            if let Some(s) = nav_sess.as_mut() {
                let tin = TensorRef::from_array_view(([1usize, 3, 224, 224], nav_img.as_slice()))
                    .map_err(|e| format!("soak nav tin: {e}"))?;
                let outs = s
                    .run(ort::inputs!["img_s1" => tin])
                    .map_err(|e| format!("soak nav run: {e}"))?;
                let (_, desc) = outs["desc_global"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("soak nav desc: {e}"))?;
                if !desc.iter().all(|v| v.is_finite()) {
                    non_finite_count += 1;
                }
            }
            if let Some(s) = xfeat_sess.as_mut() {
                let tin = TensorRef::from_array_view(([1usize, 3, 256, 320], xfeat_img.as_slice()))
                    .map_err(|e| format!("soak xf tin: {e}"))?;
                let outs = s
                    .run(ort::inputs!["img_s2" => tin])
                    .map_err(|e| format!("soak xf run: {e}"))?;
                let (_, hm) = outs["heatmap"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("soak xf hm: {e}"))?;
                if !hm.iter().all(|v| v.is_finite()) {
                    non_finite_count += 1;
                }
            }
            if let Some(s) = lg_sess.as_mut() {
                let tk0 = TensorRef::from_array_view(([1usize, k, 2], lg_kpts.as_slice()))
                    .map_err(|e| format!("soak lg tk0: {e}"))?;
                let tk1 = TensorRef::from_array_view(([1usize, k, 2], lg_kpts.as_slice()))
                    .map_err(|e| format!("soak lg tk1: {e}"))?;
                let td0 = TensorRef::from_array_view(([1usize, k, d], lg_desc.as_slice()))
                    .map_err(|e| format!("soak lg td0: {e}"))?;
                let td1 = TensorRef::from_array_view(([1usize, k, d], lg_desc.as_slice()))
                    .map_err(|e| format!("soak lg td1: {e}"))?;
                let outs = s
                    .run(ort::inputs!["kpts0" => tk0, "kpts1" => tk1, "desc0" => td0, "desc1" => td1])
                    .map_err(|e| format!("soak lg run: {e}"))?;
                let (_, scores) = outs["scores"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("soak lg scores: {e}"))?;
                if !scores.iter().all(|v| v.is_finite()) {
                    non_finite_count += 1;
                }
            }
            let cur_tile = ((slow_ticks - 1) as usize) % 16;
            let mut sims = vec![0.15_f64; 16];
            sims[cur_tile] = 0.88;
            let delta = if slow_ticks == 1 {
                [0.0, 0.0]
            } else {
                let prev_tile = ((slow_ticks - 2) as usize) % 16;
                [
                    gallery_coords[cur_tile][0] - gallery_coords[prev_tile][0],
                    gallery_coords[cur_tile][1] - gallery_coords[prev_tile][1],
                ]
            };
            let dec = seq_vpr.step(&sims, delta, 20.0, None, None);
            if !dec.posterior_prob_top1.is_finite() {
                non_finite_count += 1;
            }
            slow_ms_samples.push(slow_t0.elapsed().as_secs_f64() * 1000.0);
        }

        let total_tick_elapsed = tick_t0.elapsed();
        if total_tick_elapsed.as_secs_f64() * 1000.0 > period_ms {
            deadline_misses += 1;
        }
        if let Some(rem) = period.checked_sub(total_tick_elapsed) {
            std::thread::sleep(rem);
        }
    }

    let duration_actual_s = round2(soak_start.elapsed().as_secs_f64());
    let rss_end_mib = read_process_rss_mib();
    let rss_delta_mib = round2(rss_end_mib - rss_start_mib);

    let mean_ms = if fast_ms_samples.is_empty() {
        0.0
    } else {
        fast_ms_samples.iter().sum::<f64>() / (fast_ms_samples.len() as f64)
    };
    fast_ms_samples.sort_by(|a, b| a.total_cmp(b));
    slow_ms_samples.sort_by(|a, b| a.total_cmp(b));

    let pass = non_finite_count == 0 && fast_ticks > 0;
    Ok(EdgeSoakReport {
        mode: "soak",
        version: EDGE_INFER_VERSION,
        non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
        actuator_authority: ACTUATOR_AUTHORITY,
        watermark: NON_FLIGHT_WATERMARK,
        duration_requested_s: seconds,
        duration_actual_s,
        target_hz: hz,
        models_loaded,
        fast_ticks,
        slow_ticks,
        p50_ms: round2(percentile_sorted(&fast_ms_samples, 0.50)),
        p95_ms: round2(percentile_sorted(&fast_ms_samples, 0.95)),
        p99_ms: round2(percentile_sorted(&fast_ms_samples, 0.99)),
        max_ms: round2(*fast_ms_samples.last().unwrap_or(&0.0)),
        mean_ms: round2(mean_ms),
        slow_p50_ms: round2(percentile_sorted(&slow_ms_samples, 0.50)),
        slow_p95_ms: round2(percentile_sorted(&slow_ms_samples, 0.95)),
        slow_max_ms: round2(*slow_ms_samples.last().unwrap_or(&0.0)),
        non_finite_count,
        rss_start_mib,
        rss_end_mib,
        rss_delta_mib,
        deadline_misses,
        pass,
    })
}

// ============================================================================
// Headless CLI Mode 2: `--replay <shadow_ndjson_path>` (Pure-Rust SHA-256 Chain + Replay)
// ============================================================================

pub(crate) const SHADOW_MODE_LOG_SCHEMA: &str = "openxvs.shadow_mode_log/1.0";
pub(crate) const SHADOW_GENESIS_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Format an IEEE-754 finite `f64` identically to CPython `json.dumps(..., allow_nan=False)`.
fn format_python_json_float(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let raw = serde_json::to_string(&v).unwrap_or_else(|_| format!("{v}"));
    let neg = raw.starts_with('-');
    let body = if neg { &raw[1..] } else { raw.as_str() };

    let (mantissa_str, exp_part): (&str, i32) = match body.find(['e', 'E']) {
        Some(pos) => (&body[..pos], body[(pos + 1)..].parse::<i32>().unwrap_or(0)),
        None => (body, 0),
    };

    let (int_part, frac_part) = match mantissa_str.find('.') {
        Some(dot) => (&mantissa_str[..dot], &mantissa_str[(dot + 1)..]),
        None => (mantissa_str, ""),
    };

    let mut digits = String::with_capacity(int_part.len() + frac_part.len());
    digits.push_str(int_part);
    digits.push_str(frac_part);
    let mut exp10 = exp_part - (frac_part.len() as i32);

    // Strip leading zeros
    let first_nz = digits.find(|c: char| c != '0').unwrap_or(digits.len());
    let digits_nz = &digits[first_nz..];
    if digits_nz.is_empty() {
        return if neg { "-0.0".to_string() } else { "0.0".to_string() };
    }
    // Strip trailing zeros
    let trimmed_nz = digits_nz.trim_end_matches('0');
    let stripped_trailing = digits_nz.len() - trimmed_nz.len();
    exp10 += stripped_trailing as i32;

    // Scientific exponent of most significant digit: value = d0.d1d2... * 10^sci_exp
    let sci_exp = exp10 + (trimmed_nz.len() as i32) - 1;
    let sign_prefix = if neg { "-" } else { "" };

    if !(-4..16).contains(&sci_exp) {
        let mant = if trimmed_nz.len() == 1 {
            trimmed_nz.to_string()
        } else {
            format!("{}.{}", &trimmed_nz[..1], &trimmed_nz[1..])
        };
        if sci_exp >= 0 {
            format!("{sign_prefix}{mant}e+{sci_exp:02}")
        } else {
            format!("{sign_prefix}{mant}e-{:02}", -sci_exp)
        }
    } else if sci_exp >= 0 {
        let int_len = (sci_exp + 1) as usize;
        if int_len >= trimmed_nz.len() {
            let zeros = "0".repeat(int_len - trimmed_nz.len());
            format!("{sign_prefix}{trimmed_nz}{zeros}.0")
        } else {
            format!(
                "{sign_prefix}{}.{}",
                &trimmed_nz[..int_len],
                &trimmed_nz[int_len..]
            )
        }
    } else {
        let lead_zeros = "0".repeat((-sci_exp - 1) as usize);
        format!("{sign_prefix}0.{lead_zeros}{trimmed_nz}")
    }
}

fn write_python_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if (c as u32) <= 0x7E => out.push(c),
            c if (c as u32) <= 0xFFFF => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => {
                let code = (c as u32) - 0x10000;
                let hi = 0xD800 + (code >> 10);
                let lo = 0xDC00 + (code & 0x3FF);
                out.push_str(&format!("\\u{:04x}\\u{:04x}", hi, lo));
            }
        }
    }
    out.push('"');
}

/// Exact byte-for-byte Rust mirror of Python `json.dumps(val, sort_keys=True, separators=(",", ":"), allow_nan=False)`.
pub fn canonical_json_python_compat(val: &serde_json::Value) -> Result<String, String> {
    let mut out = String::new();
    fn rec(out: &mut String, v: &serde_json::Value) -> Result<(), String> {
        match v {
            serde_json::Value::Null => out.push_str("null"),
            serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    out.push_str(&i.to_string());
                } else if let Some(u) = n.as_u64() {
                    out.push_str(&u.to_string());
                } else if let Some(f) = n.as_f64() {
                    if !f.is_finite() {
                        return Err("Non-finite float in canonical JSON".to_string());
                    }
                    out.push_str(&format_python_json_float(f));
                } else {
                    return Err("Unsupported JSON number".to_string());
                }
            }
            serde_json::Value::String(s) => write_python_json_string(out, s),
            serde_json::Value::Array(arr) => {
                out.push('[');
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    rec(out, item)?;
                }
                out.push(']');
            }
            serde_json::Value::Object(map) => {
                out.push('{');
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_python_json_string(out, k);
                    out.push(':');
                    rec(out, &map[k])?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
    rec(&mut out, val)?;
    Ok(out)
}

#[derive(Debug, Serialize)]
pub struct EdgeReplayReport {
    mode: &'static str,
    version: &'static str,
    non_flight_sim_only: bool,
    actuator_authority: bool,
    watermark: &'static str,
    log_filename: String,
    valid: bool,
    record_count: u64,
    first_seq: Option<u64>,
    last_seq: Option<u64>,
    head_hash: String,
    tx_inhibited_all: bool,
    actuator_authority_all_false: bool,
    total_outbound_mavlink_frames: u64,
    monotonic_timestamps_ok: bool,
    first_failed_line: Option<usize>,
    cusum_epochs_replayed: u64,
    cusum_alarms: u64,
    cusum_max_stat: f64,
    cusum_vertical_max_stat: f64,
    cusum_quarantined_final: bool,
    rta_ticks_replayed: u64,
    rta_final_mode: String,
    rta_final_mode_id: u8,
    rta_mandatory_go_around_count: u64,
    errors: Vec<String>,
}

fn extract_f64_field(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<f64> {
    obj.get(key).and_then(|v| v.as_f64()).filter(|v| v.is_finite())
}

fn extract_ne_pair(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<[f64; 2]> {
    let arr = obj.get(key)?.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    let a = arr[0].as_f64()?;
    let b = arr[1].as_f64()?;
    if a.is_finite() && b.is_finite() {
        Some([a, b])
    } else {
        None
    }
}

pub fn verify_and_replay_shadow_ndjson(log_path: &Path) -> EdgeReplayReport {
    let log_filename = log_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "shadow.ndjson".to_string());

    let mut errors: Vec<String> = Vec::new();
    let mut first_failed_line: Option<usize> = None;
    let mut record_err = |line_no: usize, msg: String| {
        if first_failed_line.is_none() {
            first_failed_line = Some(line_no);
        }
        errors.push(format!("line {line_no}: {msg}"));
    };

    let raw_text = match std::fs::read_to_string(log_path) {
        Ok(t) => t,
        Err(e) => {
            return EdgeReplayReport {
                mode: "replay",
                version: EDGE_INFER_VERSION,
                non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
                actuator_authority: ACTUATOR_AUTHORITY,
                watermark: NON_FLIGHT_WATERMARK,
                log_filename,
                valid: false,
                record_count: 0,
                first_seq: None,
                last_seq: None,
                head_hash: SHADOW_GENESIS_HASH.to_string(),
                tx_inhibited_all: false,
                actuator_authority_all_false: false,
                total_outbound_mavlink_frames: 0,
                monotonic_timestamps_ok: false,
                first_failed_line: Some(1),
                cusum_epochs_replayed: 0,
                cusum_alarms: 0,
                cusum_max_stat: 0.0,
                cusum_vertical_max_stat: 0.0,
                cusum_quarantined_final: false,
                rta_ticks_replayed: 0,
                rta_final_mode: "InvalidInhibit".to_string(),
                rta_final_mode_id: 0,
                rta_mandatory_go_around_count: 0,
                errors: vec![format!("Shadow-mode NDJSON log file unreadable: {e}")],
            };
        }
    };

    let lines: Vec<&str> = raw_text.lines().collect();
    if lines.is_empty() {
        return EdgeReplayReport {
            mode: "replay",
            version: EDGE_INFER_VERSION,
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            watermark: NON_FLIGHT_WATERMARK,
            log_filename,
            valid: false,
            record_count: 0,
            first_seq: None,
            last_seq: None,
            head_hash: SHADOW_GENESIS_HASH.to_string(),
            tx_inhibited_all: false,
            actuator_authority_all_false: false,
            total_outbound_mavlink_frames: 0,
            monotonic_timestamps_ok: false,
            first_failed_line: Some(1),
            cusum_epochs_replayed: 0,
            cusum_alarms: 0,
            cusum_max_stat: 0.0,
            cusum_vertical_max_stat: 0.0,
            cusum_quarantined_final: false,
            rta_ticks_replayed: 0,
            rta_final_mode: "InvalidInhibit".to_string(),
            rta_final_mode_id: 0,
            rta_mandatory_go_around_count: 0,
            errors: vec!["Empty shadow-mode NDJSON log (0 records)".to_string()],
        };
    }

    let mut record_count: u64 = 0;
    let mut first_seq: Option<u64> = None;
    let mut last_seq: Option<u64> = None;
    let mut expected_prev_hash = SHADOW_GENESIS_HASH.to_string();
    let mut prev_ts: Option<u64> = None;
    let mut tx_inhibited_all = true;
    let mut actuator_authority_all_false = true;
    let mut total_outbound: u64 = 0;
    let mut monotonic_ok = true;

    let mut cusum =
        DualBankEsEkfCusum::new(DualBankConfig::default()).expect("default DualBankEsEkfCusum");
    let mut rta = SimplexRta::new();
    let mut cusum_epochs_replayed: u64 = 0;
    let mut cusum_alarms: u64 = 0;
    let mut cusum_max_stat: f64 = 0.0;
    let mut cusum_vertical_max_stat: f64 = 0.0;
    let mut rta_ticks_replayed: u64 = 0;
    let mut rta_final_mode = "NominalMl".to_string();
    let mut rta_final_mode_id: u8 = 4;
    let mut rta_mandatory_go_around_count: u64 = 0;

    for (idx0, raw_line) in lines.iter().enumerate() {
        let line_no = idx0 + 1;
        if raw_line.trim().is_empty() {
            record_err(
                line_no,
                "Blank line forbidden inside hash-chained NDJSON log".to_string(),
            );
            continue;
        }
        let parsed: serde_json::Value = match serde_json::from_str(raw_line) {
            Ok(v) => v,
            Err(e) => {
                record_err(line_no, format!("Malformed or non-finite JSON: {e}"));
                continue;
            }
        };
        let Some(rec) = parsed.as_object() else {
            record_err(line_no, "Record is not a JSON object".to_string());
            continue;
        };

        let expected_seq = record_count;
        record_count += 1;

        match rec.get("schema").and_then(|v| v.as_str()) {
            Some(s) if s == SHADOW_MODE_LOG_SCHEMA => {}
            other => record_err(
                line_no,
                format!("Invalid schema {other:?}, expected {SHADOW_MODE_LOG_SCHEMA:?}"),
            ),
        }

        match rec.get("seq").and_then(|v| v.as_u64()) {
            Some(seq) if seq == expected_seq => {
                if first_seq.is_none() {
                    first_seq = Some(seq);
                }
                last_seq = Some(seq);
            }
            other => record_err(
                line_no,
                format!("Sequence mismatch: got {other:?}, expected {expected_seq}"),
            ),
        }

        match rec.get("timestamp_mono_us").and_then(|v| v.as_u64()) {
            Some(ts) => {
                if let Some(p) = prev_ts {
                    if ts <= p {
                        monotonic_ok = false;
                        record_err(
                            line_no,
                            format!("Non-monotonic timestamp_mono_us: {ts} <= {p}"),
                        );
                    }
                }
                prev_ts = Some(ts);
            }
            None => {
                monotonic_ok = false;
                record_err(line_no, "Invalid timestamp_mono_us".to_string());
            }
        }

        if rec.get("tx_inhibited").and_then(|v| v.as_bool()) != Some(true) {
            tx_inhibited_all = false;
            record_err(
                line_no,
                "Hard invariant violated: tx_inhibited must be true".to_string(),
            );
        }

        if rec.get("actuator_authority").and_then(|v| v.as_bool()) != Some(false) {
            actuator_authority_all_false = false;
            record_err(
                line_no,
                "Hard invariant violated: actuator_authority must be false".to_string(),
            );
        }

        match rec
            .get("outbound_mavlink_frames_emitted")
            .and_then(|v| v.as_u64())
        {
            Some(0) => {}
            Some(n) => {
                total_outbound = total_outbound.saturating_add(n);
                record_err(
                    line_no,
                    format!("Hard invariant violated: outbound_mavlink_frames_emitted={n} != 0"),
                );
            }
            None => {
                record_err(
                    line_no,
                    "Hard invariant violated: invalid outbound_mavlink_frames_emitted".to_string(),
                );
            }
        }

        match rec.get("direction").and_then(|v| v.as_str()) {
            Some("rx") | Some("internal") => {}
            other => record_err(
                line_no,
                format!("Forbidden shadow-mode direction {other:?}"),
            ),
        }

        match rec.get("prev_hash").and_then(|v| v.as_str()) {
            Some(ph) if ph == expected_prev_hash => {}
            other => record_err(
                line_no,
                format!("Hash chain break: prev_hash={other:?} != expected {expected_prev_hash:?}"),
            ),
        }

        let entry_hash = match rec.get("entry_hash").and_then(|v| v.as_str()) {
            Some(eh) if eh.len() == 64 && eh.chars().all(|c| c.is_ascii_hexdigit()) => eh,
            other => {
                record_err(line_no, format!("Invalid entry_hash field: {other:?}"));
                expected_prev_hash = other.unwrap_or("").to_string();
                continue;
            }
        };

        let mut body_map = serde_json::Map::new();
        for (k, v) in rec {
            if k != "entry_hash" {
                body_map.insert(k.clone(), v.clone());
            }
        }
        let body_val = serde_json::Value::Object(body_map);
        match canonical_json_python_compat(&body_val) {
            Ok(canonical) => {
                let digest = Sha256::digest(canonical.as_bytes());
                let mut calc_hex = String::with_capacity(64);
                for b in digest {
                    calc_hex.push_str(&format!("{b:02x}"));
                }
                if calc_hex != entry_hash {
                    record_err(
                        line_no,
                        format!(
                            "SHA-256 tamper detected: entry_hash={}... != calc={}...",
                            &entry_hash[..16],
                            &calc_hex[..16]
                        ),
                    );
                }
            }
            Err(e) => record_err(line_no, format!("Canonical JSON serialization error: {e}")),
        }
        expected_prev_hash = entry_hash.to_string();

        // Replay record telemetry through Pure-Rust DualBankEsEkfCusum & SimplexRta
        if let Some(payload) = rec.get("payload").and_then(|v| v.as_object()) {
            let src = payload
                .get("telemetry_ned")
                .and_then(|v| v.as_object())
                .unwrap_or(payload);
            let baro_down = extract_f64_field(src, "baro_down_m")
                .or_else(|| extract_f64_field(src, "down_m"))
                .or_else(|| extract_f64_field(src, "alt_msl_m").map(|h| -h))
                .unwrap_or(0.0);
            let gnss_ne = extract_ne_pair(src, "gnss_ne_m").or_else(|| {
                let n = extract_f64_field(src, "north_m")?;
                let e = extract_f64_field(src, "east_m")?;
                Some([n, e])
            });
            // Fail-closed: NEVER copy `gnss_ne` into `vis_ne` when `vis_ne_m` is absent!
            // Feeding GNSS into visual Bank-B would mask GNSS spoofing in replay.
            let vis_ne = extract_ne_pair(src, "vis_ne_m");
            let gnss_down = extract_f64_field(src, "gnss_down_m");
            let ins_accel = extract_ne_pair(src, "ins_accel_ne_mps2").unwrap_or([0.0, 0.0]);
            let vis_gate = src
                .get("vis_gate_admitted")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let c_out = cusum.step(&DualBankStepInput {
                ins_accel_ne_mps2: ins_accel,
                baro_down_m: baro_down,
                gnss_ne_m: gnss_ne,
                vis_ne_m: vis_ne,
                vis_gate_admitted: vis_gate,
                r_gnss: None,
                r_vis: None,
                gnss_down_m: gnss_down,
            });
            cusum_epochs_replayed += 1;
            if c_out.spoof_alarm {
                cusum_alarms += 1;
            }
            if c_out.max_cusum_stat.is_finite() {
                cusum_max_stat = cusum_max_stat.max(c_out.max_cusum_stat);
            }
            if c_out.vertical_cusum_stat.is_finite() {
                cusum_vertical_max_stat = cusum_vertical_max_stat.max(c_out.vertical_cusum_stat);
            }

            let float_tel = RtaFloatTelemetry {
                horizon_pl_rad: extract_f64_field(src, "horizon_pl_rad").unwrap_or(0.002),
                horizon_al_rad: 0.017_453_3,
                v_vpl_m: extract_f64_field(src, "v_vpl_m").unwrap_or(5.0),
                v_val_m: 10.0,
                v_hpl_m: extract_f64_field(src, "v_hpl_m").unwrap_or(14.0),
                v_hal_m: 30.0,
                epipolar_max_rad: [0.0002; 6],
                signed_horizon_residual_rad: 0.0,
                runway_width_delta_m: 0.0,
                runway_width_delta_limit_m: 5.0,
                cusum_stat: if c_out.spoof_alarm {
                    CUSUM_H_LOCKED + 1.0
                } else {
                    c_out.max_cusum_stat.min(CUSUM_H_LOCKED - 0.1)
                },
                cusum_thr: CUSUM_H_LOCKED,
                ood_global_score: 0.1,
                ood_global_thr: 0.7,
                ood_cam_scores: [0.1; 6],
                ood_cam_thr: 0.7,
                cam_health_mask: src
                    .get("cam_health_mask")
                    .and_then(|v| v.as_u64())
                    .map(|m| m as u8)
                    .unwrap_or(0x3F),
                ml_detector_fault: src
                    .get("ml_detector_fault")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                transient_dropout: false,
                approach_active: src
                    .get("approach_active")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                go_around_ack_reset: false,
            };
            let rta_out = rta.step(&build_rta_input_from_f64(&float_tel));
            rta_ticks_replayed += 1;
            rta_final_mode = format!("{:?}", rta_out.mode);
            rta_final_mode_id = rta_out.mode as u8;
            if rta_out.mandatory_go_around {
                rta_mandatory_go_around_count += 1;
            }
        }
    }

    let valid = errors.is_empty()
        && record_count > 0
        && tx_inhibited_all
        && actuator_authority_all_false
        && total_outbound == 0
        && monotonic_ok;

    EdgeReplayReport {
        mode: "replay",
        version: EDGE_INFER_VERSION,
        non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
        actuator_authority: ACTUATOR_AUTHORITY,
        watermark: NON_FLIGHT_WATERMARK,
        log_filename,
        valid,
        record_count,
        first_seq,
        last_seq,
        head_hash: expected_prev_hash,
        tx_inhibited_all,
        actuator_authority_all_false,
        total_outbound_mavlink_frames: total_outbound,
        monotonic_timestamps_ok: monotonic_ok,
        first_failed_line,
        cusum_epochs_replayed,
        cusum_alarms,
        cusum_max_stat: round4(cusum_max_stat),
        cusum_vertical_max_stat: round4(cusum_vertical_max_stat),
        cusum_quarantined_final: cusum.gnss_quarantined,
        rta_ticks_replayed,
        rta_final_mode,
        rta_final_mode_id,
        rta_mandatory_go_around_count,
        errors,
    }
}

// ============================================================================
// Headless CLI Mode 3: `--parity-eval <trace_json_path | ->` (Python<->Rust Parity Harness)
// ============================================================================

fn parse_parity_f64(v: &serde_json::Value) -> Result<f64, String> {
    if let Some(f) = v.as_f64() {
        return Ok(f);
    }
    if let Some(s) = v.as_str() {
        return match s {
            "NaN" | "nan" => Ok(f64::NAN),
            "Inf" | "+Inf" | "Infinity" | "inf" => Ok(f64::INFINITY),
            "-Inf" | "-Infinity" | "-inf" => Ok(f64::NEG_INFINITY),
            other => Err(format!("Unsupported float token: {other:?}")),
        };
    }
    Err(format!("Expected float or NaN/Inf token, got {v}"))
}

fn encode_parity_f64(v: f64) -> serde_json::Value {
    if v.is_nan() {
        serde_json::Value::String("NaN".to_string())
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            serde_json::Value::String("Inf".to_string())
        } else {
            serde_json::Value::String("-Inf".to_string())
        }
    } else {
        serde_json::json!(v)
    }
}

fn eval_parity_case(case: &serde_json::Value) -> Result<serde_json::Value, String> {
    let kind = case
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Missing parity case 'kind'".to_string())?;
    match kind {
        "sequence_vpr" => {
            let gallery_arr = case
                .get("gallery_ne_m")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "Missing gallery_ne_m".to_string())?;
            let mut gallery = Vec::with_capacity(gallery_arr.len());
            for row in gallery_arr {
                let pair = row
                    .as_array()
                    .ok_or_else(|| "gallery row must be [N, E]".to_string())?;
                if pair.len() != 2 {
                    return Err(format!(
                        "gallery row must have exactly 2 elements [N, E], got {}",
                        pair.len()
                    ));
                }
                gallery.push([parse_parity_f64(&pair[0])?, parse_parity_f64(&pair[1])?]);
            }
            let cfg: SequenceVprConfig = match case.get("config") {
                Some(c) if !c.is_null() => {
                    serde_json::from_value(c.clone()).map_err(|e| format!("bad seq config: {e}"))?
                }
                _ => SequenceVprConfig::default(),
            };
            let mut flt = SequenceVprFilter::new(gallery, cfg)?;
            let steps = case
                .get("steps")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "Missing steps array".to_string())?;
            let mut outputs = Vec::with_capacity(steps.len());
            for step in steps {
                if step.get("reset").and_then(|v| v.as_bool()) == Some(true) {
                    flt.reset();
                    continue;
                }
                let sims_arr = step
                    .get("similarities")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| "Missing similarities".to_string())?;
                let mut sims = Vec::with_capacity(sims_arr.len());
                for x in sims_arr {
                    sims.push(parse_parity_f64(x)?);
                }
                let delta_ne = match step.get("delta_ne_odom_m").and_then(|v| v.as_array()) {
                    Some(d) if d.len() == 2 => [parse_parity_f64(&d[0])?, parse_parity_f64(&d[1])?],
                    _ => [0.0, 0.0],
                };
                let odom_sigma = match step.get("odom_sigma_m") {
                    Some(s) if !s.is_null() => parse_parity_f64(s)?,
                    _ => 25.0,
                };
                let vfm: Option<VisualFixMetrics> = match step.get("visual_fix_metrics") {
                    Some(v) if !v.is_null() => {
                        Some(serde_json::from_value(v.clone()).map_err(|e| format!("bad vfm: {e}"))?)
                    }
                    _ => None,
                };
                let thr: Option<VisualAliasingThresholds> = match step.get("aliasing_thresholds") {
                    Some(v) if !v.is_null() => {
                        Some(serde_json::from_value(v.clone()).map_err(|e| format!("bad thr: {e}"))?)
                    }
                    _ => None,
                };
                let dec = flt.step(&sims, delta_ne, odom_sigma, vfm.as_ref(), thr.as_ref());
                outputs.push(serde_json::json!({
                    "frame_idx": dec.frame_idx,
                    "window_len": dec.window_len,
                    "raw_top1_idx": dec.raw_top1_idx,
                    "seq_top1_idx": dec.seq_top1_idx,
                    "final_top1_idx": dec.final_top1_idx,
                    "final_top5_indices": dec.final_top5_indices,
                    "viterbi_path_indices": dec.viterbi_path_indices,
                    "posterior_prob_top1": encode_parity_f64(dec.posterior_prob_top1),
                    "posterior_margin_non_adj": encode_parity_f64(dec.posterior_margin_non_adj),
                    "psr_cal": encode_parity_f64(dec.psr_result.psr_cal),
                    "psr_margin": encode_parity_f64(dec.psr_result.margin),
                    "psr_admitted": dec.psr_result.admitted,
                    "admitted": dec.admitted,
                    "reason_codes": dec.reason_codes,
                    "aliasing_admitted": dec.aliasing_decision.as_ref().map(|a| a.admitted),
                    "aliasing_integrity_alert": dec.aliasing_decision.as_ref().map(|a| a.integrity_alert),
                }));
            }
            Ok(serde_json::json!({ "kind": "sequence_vpr", "outputs": outputs }))
        }
        "dual_bank_cusum" => {
            let cfg: DualBankConfig = match case.get("config") {
                Some(c) if !c.is_null() => {
                    serde_json::from_value(c.clone()).map_err(|e| format!("bad cusum config: {e}"))?
                }
                _ => DualBankConfig::default(),
            };
            let mut flt = DualBankEsEkfCusum::new(cfg)?;
            if let Some(rst) = case.get("reset").and_then(|v| v.as_object()) {
                let p0 = rst
                    .get("init_ned_m")
                    .and_then(|v| v.as_array())
                    .map(|a| -> Result<[f64; 3], String> {
                        if a.len() != 3 {
                            return Err(format!(
                                "init_ned_m must have exactly 3 elements [N, E, D], got {}",
                                a.len()
                            ));
                        }
                        Ok([
                            parse_parity_f64(&a[0])?,
                            parse_parity_f64(&a[1])?,
                            parse_parity_f64(&a[2])?,
                        ])
                    })
                    .transpose()?
                    .unwrap_or([0.0, 0.0, 0.0]);
                let v0 = rst
                    .get("init_vel_ne_mps")
                    .and_then(|v| v.as_array())
                    .map(|a| -> Result<[f64; 2], String> {
                        if a.len() != 2 {
                            return Err(format!(
                                "init_vel_ne_mps must have exactly 2 elements [vN, vE], got {}",
                                a.len()
                            ));
                        }
                        Ok([parse_parity_f64(&a[0])?, parse_parity_f64(&a[1])?])
                    })
                    .transpose()?
                    .unwrap_or([0.0, 0.0]);
                flt.reset(p0, v0);
            }
            let steps = case
                .get("steps")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "Missing steps array".to_string())?;
            let mut outputs = Vec::with_capacity(steps.len());
            for step in steps {
                let ins_accel = match step.get("ins_accel_ne_mps2").and_then(|v| v.as_array()) {
                    Some(a) if a.len() == 2 => [parse_parity_f64(&a[0])?, parse_parity_f64(&a[1])?],
                    _ => [0.0, 0.0],
                };
                let baro_down_m = match step.get("baro_down_m") {
                    Some(v) if !v.is_null() => parse_parity_f64(v)?,
                    _ => 0.0,
                };
                let gnss_ne_m = match step.get("gnss_ne_m") {
                    Some(v) if !v.is_null() => {
                        let a = v
                            .as_array()
                            .ok_or_else(|| "gnss_ne_m must be [N, E]".to_string())?;
                        if a.len() != 2 {
                            return Err(format!(
                                "gnss_ne_m must have exactly 2 elements [N, E], got {}",
                                a.len()
                            ));
                        }
                        Some([parse_parity_f64(&a[0])?, parse_parity_f64(&a[1])?])
                    }
                    _ => None,
                };
                let vis_ne_m = match step.get("vis_ne_m") {
                    Some(v) if !v.is_null() => {
                        let a = v
                            .as_array()
                            .ok_or_else(|| "vis_ne_m must be [N, E]".to_string())?;
                        if a.len() != 2 {
                            return Err(format!(
                                "vis_ne_m must have exactly 2 elements [N, E], got {}",
                                a.len()
                            ));
                        }
                        Some([parse_parity_f64(&a[0])?, parse_parity_f64(&a[1])?])
                    }
                    _ => None,
                };
                let vis_gate_admitted = step
                    .get("vis_gate_admitted")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let gnss_down_m = match step.get("gnss_down_m") {
                    Some(v) if !v.is_null() => Some(parse_parity_f64(v)?),
                    _ => None,
                };
                let out = flt.step(&DualBankStepInput {
                    ins_accel_ne_mps2: ins_accel,
                    baro_down_m,
                    gnss_ne_m,
                    vis_ne_m,
                    vis_gate_admitted,
                    r_gnss: None,
                    r_vis: None,
                    gnss_down_m,
                });
                outputs.push(serde_json::json!({
                    "epoch_idx": out.epoch_idx,
                    "timestamp_s": encode_parity_f64(out.timestamp_s),
                    "active_bank": out.active_bank,
                    "gnss_quarantined": out.gnss_quarantined,
                    "quarantine_triggered_this_epoch": out.quarantine_triggered_this_epoch,
                    "reintegrated_this_epoch": out.reintegrated_this_epoch,
                    "active_p_ned_m": [
                        encode_parity_f64(out.active_p_ned_m[0]),
                        encode_parity_f64(out.active_p_ned_m[1]),
                        encode_parity_f64(out.active_p_ned_m[2]),
                    ],
                    "bank_a_p_ne_m": [
                        encode_parity_f64(out.bank_a_p_ne_m[0]),
                        encode_parity_f64(out.bank_a_p_ne_m[1]),
                    ],
                    "bank_b_p_ne_m": [
                        encode_parity_f64(out.bank_b_p_ne_m[0]),
                        encode_parity_f64(out.bank_b_p_ne_m[1]),
                    ],
                    "d_ne_m": [
                        encode_parity_f64(out.d_ne_m[0]),
                        encode_parity_f64(out.d_ne_m[1]),
                    ],
                    "max_cusum_stat": encode_parity_f64(out.max_cusum_stat),
                    "dominant_arm": out.dominant_arm,
                    "q_vis": encode_parity_f64(out.q_vis),
                    "nominal_streak": out.nominal_streak,
                    "vertical_cusum_stat": encode_parity_f64(out.vertical_cusum_stat),
                    "vertical_spoof_alarm": out.vertical_spoof_alarm,
                    "spoof_alarm": out.spoof_alarm,
                }));
            }
            Ok(serde_json::json!({ "kind": "dual_bank_cusum", "outputs": outputs }))
        }
        "batch" => {
            let cases = case
                .get("cases")
                .and_then(|v| v.as_object())
                .ok_or_else(|| "Missing batch 'cases' object".to_string())?;
            let mut results = serde_json::Map::new();
            for (name, sub) in cases {
                results.insert(name.clone(), eval_parity_case(sub)?);
            }
            Ok(serde_json::json!({ "kind": "batch", "results": results }))
        }
        other => Err(format!("Unsupported parity case kind: {other:?}")),
    }
}

fn run_parity_eval_mode(path_arg: &str) -> Result<serde_json::Value, String> {
    let raw = if path_arg == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        buf
    } else {
        std::fs::read_to_string(path_arg)
            .map_err(|e| format!("failed to read parity trace {path_arg}: {e}"))?
    };
    let doc: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("invalid parity trace JSON: {e}"))?;
    eval_parity_case(&doc)
}

pub fn cli_help_text() -> String {
    format!(
        "openxvs-edge-infer v{EDGE_INFER_VERSION} — Headless Pure-Rust Edge Evaluation Runner\n\
         {NON_FLIGHT_WATERMARK}\n\
         NON_FLIGHT_SIM_ONLY={NON_FLIGHT_SIM_ONLY} | ACTUATOR_AUTHORITY={ACTUATOR_AUTHORITY} | OUTBOUND_MAVLINK_FRAMES=0\n\n\
         USAGE:\n\
           openxvs-edge-infer [--self-check] [--require-models] [--model-dir <DIR>]\n\
           openxvs-edge-infer --eval-frames <DIR> [--mode all|eo|thermal|land] [--out <REPORT.JSON>]\n\
                                                  [--tape <TAPE.NDJSON>] [--max-frames <N>] [--model-dir <DIR>]\n\
           openxvs-edge-infer --soak [--seconds <N>] [--hz <F>] [--require-models] [--model-dir <DIR>]\n\
           openxvs-edge-infer --replay <SHADOW.NDJSON>\n\
           openxvs-edge-infer --parity-eval <TRACE.JSON | ->\n\
           openxvs-edge-infer --help | -h | --version | -V\n\n\
         EVAL-FRAMES OPTIONS:\n\
           --eval-frames <DIR>       Directory of sorted .png/.jpg/.jpeg images to evaluate offline\n\
           --mode <MODE>             Model pipeline: all (default), eo, thermal, land\n\
           --out <REPORT.JSON>       Write full JSON evaluation report to file (default: stdout)\n\
           --tape <TAPE.NDJSON>      Write SHA-256 hash-chained shadow flight tape verifiable by --replay\n\
           --max-frames <N>          Evaluate at most N sorted frames (positive integer >= 1)\n\
           --model-dir <DIR>         Explicit ONNX model directory (overrides OPENXVS_MODEL_DIR auto-discovery)\n\n\
         SAFETY LIMITS:\n\
           - Rejects corrupt, 0-byte, >16 MiB, or >4096x4096 images fail-closed without panicking\n\
           - Enforces fail-closed SHA-256 pinning on all required ONNX models\n\
           - Zero flight actuation, zero outbound MAVLink frames\n"
    )
}

#[derive(Debug)]
enum CliCommand {
    Help,
    Version,
    SelfCheck {
        require_models: bool,
        model_dir: Option<PathBuf>,
    },
    Soak {
        seconds: f64,
        hz: f64,
        model_dir: Option<PathBuf>,
        require_models: bool,
    },
    Replay {
        path: PathBuf,
    },
    ParityEval {
        path_arg: String,
    },
    EvalFrames(eval_frames::EvalFramesConfig),
}

#[derive(Debug)]
enum CliParseError {
    General { message: String, exit_code: i32 },
    EvalFrames(eval_frames::EvalFramesFailure),
}

fn require_flag_value<'a>(
    args: &'a [String],
    idx: usize,
    flag: &str,
) -> Result<&'a str, String> {
    let Some(val) = args.get(idx + 1) else {
        return Err(format!("Missing value for {flag}"));
    };
    if val.trim().is_empty() {
        return Err(format!("Empty value for {flag}"));
    }
    Ok(val.as_str())
}

fn parse_cli_command(args: &[String]) -> Result<CliCommand, CliParseError> {
    if args.is_empty() {
        return Ok(CliCommand::SelfCheck {
            require_models: false,
            model_dir: None,
        });
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return Ok(CliCommand::Help);
    }
    if args.len() == 1 && (args[0] == "--version" || args[0] == "-V") {
        return Ok(CliCommand::Version);
    }

    let first = args[0].as_str();
    if first == "--self-check" || first == "--require-models" || first == "--model-dir" {
        let mut require_models = false;
        let mut model_dir: Option<PathBuf> = None;
        let mut i = usize::from(first == "--self-check");
        while i < args.len() {
            match args[i].as_str() {
                "--require-models" => {
                    require_models = true;
                    i += 1;
                }
                "--model-dir" => {
                    let v = require_flag_value(args, i, "--model-dir").map_err(|m| {
                        CliParseError::General {
                            message: m,
                            exit_code: 2,
                        }
                    })?;
                    model_dir = Some(PathBuf::from(v));
                    i += 2;
                }
                other => {
                    return Err(CliParseError::General {
                        message: format!("Unknown --self-check argument: {other}"),
                        exit_code: 2,
                    });
                }
            }
        }
        return Ok(CliCommand::SelfCheck {
            require_models,
            model_dir,
        });
    }

    match first {
        "--eval-frames" => {
            let dir_str = require_flag_value(args, 0, "--eval-frames").map_err(|m| {
                CliParseError::EvalFrames(eval_frames::EvalFramesFailure::new(
                    "INVALID_CLI_ARGS",
                    m,
                ))
            })?;
            if dir_str.starts_with("--") {
                return Err(CliParseError::EvalFrames(
                    eval_frames::EvalFramesFailure::new(
                        "INVALID_CLI_ARGS",
                        format!("Expected input directory after --eval-frames, got flag '{dir_str}'"),
                    ),
                ));
            }
            let input_dir = PathBuf::from(dir_str);
            let mut mode = eval_frames::EvalMode::All;
            let mut out_path: Option<PathBuf> = None;
            let mut tape_path: Option<PathBuf> = None;
            let mut max_frames: Option<usize> = None;
            let mut model_dir: Option<PathBuf> = None;

            let mut i = 2usize;
            while i < args.len() {
                match args[i].as_str() {
                    "--mode" => {
                        let v = require_flag_value(args, i, "--mode").map_err(|m| {
                            CliParseError::EvalFrames(eval_frames::EvalFramesFailure::new(
                                "INVALID_CLI_ARGS",
                                m,
                            ))
                        })?;
                        mode = eval_frames::EvalMode::parse(v).map_err(|m| {
                            CliParseError::EvalFrames(eval_frames::EvalFramesFailure::new(
                                "INVALID_MODE",
                                m,
                            ))
                        })?;
                        i += 2;
                    }
                    "--out" => {
                        let v = require_flag_value(args, i, "--out").map_err(|m| {
                            CliParseError::EvalFrames(eval_frames::EvalFramesFailure::new(
                                "INVALID_CLI_ARGS",
                                m,
                            ))
                        })?;
                        out_path = Some(PathBuf::from(v));
                        i += 2;
                    }
                    "--tape" => {
                        let v = require_flag_value(args, i, "--tape").map_err(|m| {
                            CliParseError::EvalFrames(eval_frames::EvalFramesFailure::new(
                                "INVALID_CLI_ARGS",
                                m,
                            ))
                        })?;
                        tape_path = Some(PathBuf::from(v));
                        i += 2;
                    }
                    "--max-frames" => {
                        let v = require_flag_value(args, i, "--max-frames").map_err(|m| {
                            CliParseError::EvalFrames(eval_frames::EvalFramesFailure::new(
                                "INVALID_MAX_FRAMES",
                                m,
                            ))
                        })?;
                        let parsed = v.trim().parse::<usize>().map_err(|_| {
                            CliParseError::EvalFrames(eval_frames::EvalFramesFailure::new(
                                "INVALID_MAX_FRAMES",
                                format!("--max-frames must be a positive integer >= 1, got '{v}'"),
                            ))
                        })?;
                        if parsed == 0 {
                            return Err(CliParseError::EvalFrames(
                                eval_frames::EvalFramesFailure::new(
                                    "INVALID_MAX_FRAMES",
                                    "--max-frames must be a positive integer >= 1, got 0",
                                ),
                            ));
                        }
                        max_frames = Some(parsed);
                        i += 2;
                    }
                    "--model-dir" => {
                        let v = require_flag_value(args, i, "--model-dir").map_err(|m| {
                            CliParseError::EvalFrames(eval_frames::EvalFramesFailure::new(
                                "INVALID_CLI_ARGS",
                                m,
                            ))
                        })?;
                        model_dir = Some(PathBuf::from(v));
                        i += 2;
                    }
                    other => {
                        return Err(CliParseError::EvalFrames(
                            eval_frames::EvalFramesFailure::new(
                                "UNKNOWN_CLI_FLAG",
                                format!("Unknown --eval-frames argument: {other}"),
                            ),
                        ));
                    }
                }
            }

            Ok(CliCommand::EvalFrames(eval_frames::EvalFramesConfig {
                input_dir,
                mode,
                out_path,
                tape_path,
                max_frames,
                model_dir,
            }))
        }
        "--soak" => {
            let mut seconds = 30.0_f64;
            let mut hz = 5.0_f64;
            let mut model_dir: Option<PathBuf> = None;
            let mut require_models = false;
            let mut i = 1usize;
            while i < args.len() {
                match args[i].as_str() {
                    "--seconds" => {
                        let v = require_flag_value(args, i, "--seconds").map_err(|m| {
                            CliParseError::General {
                                message: m,
                                exit_code: 2,
                            }
                        })?;
                        seconds = v.parse::<f64>().unwrap_or(f64::NAN);
                        i += 2;
                    }
                    "--hz" => {
                        let v = require_flag_value(args, i, "--hz").map_err(|m| {
                            CliParseError::General {
                                message: m,
                                exit_code: 2,
                            }
                        })?;
                        hz = v.parse::<f64>().unwrap_or(f64::NAN);
                        i += 2;
                    }
                    "--model-dir" => {
                        let v = require_flag_value(args, i, "--model-dir").map_err(|m| {
                            CliParseError::General {
                                message: m,
                                exit_code: 2,
                            }
                        })?;
                        model_dir = Some(PathBuf::from(v));
                        i += 2;
                    }
                    "--require-models" => {
                        require_models = true;
                        i += 1;
                    }
                    other => {
                        return Err(CliParseError::General {
                            message: format!("Unknown --soak argument: {other}"),
                            exit_code: 2,
                        });
                    }
                }
            }
            Ok(CliCommand::Soak {
                seconds,
                hz,
                model_dir,
                require_models,
            })
        }
        "--replay" => {
            if args.len() != 2 || args[1].trim().is_empty() {
                return Err(CliParseError::General {
                    message: "Usage: openxvs-edge-infer --replay <shadow_ndjson_path>".to_string(),
                    exit_code: 2,
                });
            }
            Ok(CliCommand::Replay {
                path: PathBuf::from(&args[1]),
            })
        }
        "--parity-eval" => {
            if args.len() != 2 || args[1].trim().is_empty() {
                return Err(CliParseError::General {
                    message: "Usage: openxvs-edge-infer --parity-eval <trace_json_path | ->"
                        .to_string(),
                    exit_code: 2,
                });
            }
            Ok(CliCommand::ParityEval {
                path_arg: args[1].clone(),
            })
        }
        other => Err(CliParseError::General {
            message: format!(
                "Unknown CLI flag '{other}'. Run with --help for usage. Supported modes: --self-check, --eval-frames, --soak, --replay, --parity-eval"
            ),
            exit_code: 2,
        }),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = match parse_cli_command(&args) {
        Ok(c) => c,
        Err(CliParseError::General { message, exit_code }) => {
            eprintln!("[OPENXVS-EDGE CLI ERROR] {message}");
            std::process::exit(exit_code);
        }
        Err(CliParseError::EvalFrames(fail)) => {
            eprintln!("[OPENXVS-EDGE EVAL-FRAMES ERROR] {}", fail.message);
            let err_rep = fail.into_error_report();
            println!(
                "{}",
                serde_json::to_string_pretty(&err_rep).expect("serialize eval_frames error report")
            );
            std::process::exit(1);
        }
    };

    match cmd {
        CliCommand::Help => {
            print!("{}", cli_help_text());
        }
        CliCommand::Version => {
            println!(
                "openxvs-edge-infer {EDGE_INFER_VERSION} (NON_FLIGHT_SIM_ONLY={NON_FLIGHT_SIM_ONLY}, ACTUATOR_AUTHORITY={ACTUATOR_AUTHORITY})"
            );
        }
        CliCommand::SelfCheck {
            require_models,
            model_dir: model_dir_opt,
        } => {
            eprintln!("[OPENXVS-EDGE v{EDGE_INFER_VERSION}] {NON_FLIGHT_WATERMARK}");
            eprintln!(
                "[OPENXVS-EDGE] NON_FLIGHT_SIM_ONLY={NON_FLIGHT_SIM_ONLY} ACTUATOR_AUTHORITY={ACTUATOR_AUTHORITY}"
            );
            let (model_dir, explicit_model_dir) = match model_dir_opt {
                Some(p) => (p, true),
                None => resolve_default_model_dir(),
            };
            let report = run_self_check_with_model_dir(&model_dir, explicit_model_dir);
            let json = serde_json::to_string_pretty(&report).expect("serialize report");
            println!("{json}");

            let any_hard_error = report.models.iter().any(|m| {
                m.status == "SHA256_MISMATCH_REJECTED"
                    || m.status.starts_with("ERROR:")
                    || m.status.starts_with("INVALID_OUTPUT")
            });
            let all_models_loaded = report.ort_dylib_loaded
                && report.models.len() >= 6
                && report.models.iter().all(|m| m.loaded);
            if any_hard_error || ((require_models || explicit_model_dir) && !all_models_loaded) {
                std::process::exit(1);
            }
        }
        CliCommand::EvalFrames(cfg) => {
            eprintln!("[OPENXVS-EDGE v{EDGE_INFER_VERSION}] {NON_FLIGHT_WATERMARK}");
            eprintln!(
                "[OPENXVS-EDGE] NON_FLIGHT_SIM_ONLY={NON_FLIGHT_SIM_ONLY} ACTUATOR_AUTHORITY={ACTUATOR_AUTHORITY}"
            );
            match eval_frames::run_eval_frames(&cfg) {
                Ok(rep) => {
                    if cfg.out_path.is_some() {
                        let brief_out = serde_json::json!({
                            "mode": rep.mode,
                            "ok": rep.ok,
                            "eval_mode": rep.eval_mode,
                            "version": rep.version,
                            "non_flight_sim_only": rep.non_flight_sim_only,
                            "actuator_authority": rep.actuator_authority,
                            "outbound_mavlink_frames": rep.outbound_mavlink_frames,
                            "watermark": rep.watermark,
                            "report_written": true,
                            "tape_written": rep.tape_written,
                            "tape_sha256_head": rep.tape_sha256_head,
                            "summary": rep.summary,
                        });
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&brief_out)
                                .expect("serialize eval_frames summary")
                        );
                    } else {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&rep)
                                .expect("serialize eval_frames report")
                        );
                    }
                }
                Err(fail) => {
                    eprintln!("[OPENXVS-EDGE EVAL-FRAMES ERROR] {}", fail.message);
                    let err_rep = fail.into_error_report();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&err_rep)
                            .expect("serialize eval_frames error report")
                    );
                    std::process::exit(1);
                }
            }
        }
        CliCommand::Soak {
            seconds,
            hz,
            model_dir,
            require_models,
        } => {
            eprintln!("[OPENXVS-EDGE v{EDGE_INFER_VERSION}] {NON_FLIGHT_WATERMARK}");
            eprintln!(
                "[OPENXVS-EDGE] NON_FLIGHT_SIM_ONLY={NON_FLIGHT_SIM_ONLY} ACTUATOR_AUTHORITY={ACTUATOR_AUTHORITY}"
            );
            match run_soak_mode(seconds, hz, model_dir) {
                Ok(rep) => {
                    let ok = rep.pass && (!require_models || rep.models_loaded.len() >= 6);
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&rep).expect("serialize soak report")
                    );
                    if !ok {
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("[OPENXVS-EDGE SOAK ERROR] {e}");
                    std::process::exit(2);
                }
            }
        }
        CliCommand::Replay { path } => {
            eprintln!("[OPENXVS-EDGE v{EDGE_INFER_VERSION}] {NON_FLIGHT_WATERMARK}");
            let rep = verify_and_replay_shadow_ndjson(&path);
            let ok = rep.valid;
            println!(
                "{}",
                serde_json::to_string_pretty(&rep).expect("serialize replay report")
            );
            if !ok {
                std::process::exit(1);
            }
        }
        CliCommand::ParityEval { path_arg } => match run_parity_eval_mode(&path_arg) {
            Ok(out) => {
                println!(
                    "{}",
                    serde_json::to_string(&out).expect("serialize parity output")
                );
            }
            Err(e) => {
                eprintln!("[OPENXVS-EDGE PARITY-EVAL ERROR] {e}");
                std::process::exit(2);
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_flight_invariants_and_system32_guard() {
        const { assert!(NON_FLIGHT_SIM_ONLY) };
        const { assert!(!ACTUATOR_AUTHORITY) };
        assert!(is_forbidden_system_ort_path(Path::new(
            r"C:\Windows\System32\onnxruntime.dll"
        )));
        assert!(is_forbidden_system_ort_path(Path::new(
            r"C:\WINDOWS\SysWOW64\onnxruntime.dll"
        )));
        assert!(!is_forbidden_system_ort_path(Path::new(
            "/opt/openxvs/lib/libonnxruntime.so"
        )));
    }

    #[test]
    fn test_student_t_calibrator_exact_factor() {
        let cal = StudentTCalibrator::new(LAND_STUDENT_T_NU, LAND_STUDENT_T_SCALE).unwrap();
        let expected = LAND_STUDENT_T_SCALE * LAND_STUDENT_T_SCALE * (6.0 / 4.0);
        assert!((cal.variance_factor() - expected).abs() < 1e-12);
        let out = cal.calibrate_cov2x2([1.0, 0.2, 0.2, 1.5]);
        assert!((out[0] - expected).abs() < 1e-12);
        assert!((out[3] - 1.5 * expected).abs() < 1e-12);
    }

    #[test]
    fn test_edge_sha256_fail_closed_tamper_rejection_all_6_models() {
        let tmp = std::env::temp_dir().join("openxvs_edge_tamper_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);
        for fname in [
            "xvs_d2_web_fp16.onnx",
            "xvs_thermal_v52_fp16.onnx",
            "xvs_land_v52_fp16.onnx",
            "xvs_nav_v5_fazb_fp16.onnx",
            "xfeat_320x256_fp16.onnx",
            "lighterglue_k256_fp16.onnx",
        ] {
            std::fs::write(tmp.join(fname), b"TAMPERED_EDGE_MODEL").expect("write tamper file");
        }
        let rep = run_self_check_with_model_dir(&tmp, true);
        if rep.ort_dylib_loaded {
            assert_eq!(rep.models.len(), 6);
            for m in &rep.models {
                assert!(!m.loaded, "Tampered model {} must NOT load!", m.name);
                assert_eq!(
                    m.status, "SHA256_MISMATCH_REJECTED",
                    "Tampered model {} must fail closed",
                    m.name
                );
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_edge_self_check_runs_rta_multiship_coord_and_all_6_models() {
        let rep = run_self_check();
        assert_eq!(rep.version, EDGE_INFER_VERSION);
        assert!(rep.non_flight_sim_only);
        assert!(!rep.actuator_authority);
        assert!(rep.active_range_wire_crc_ok);
        assert_eq!(rep.rta_mode, 4);
        assert_eq!(rep.watermark, NON_FLIGHT_WATERMARK);
        assert!(rep.multiship_coord.enabled);
        assert!(rep.multiship_coord.f1_far91113_right_turn_ok);
        assert!(rep.multiship_coord.f3_ring0_p0_zero_bank_ok);
        assert!(rep.multiship_coord.f4_sticky_triple_ray_ok);
        assert!(rep.multiship_coord.f4_sticky_ghosts_suppressed >= 2);
        assert_eq!(rep.multiship_coord.f4_stale_obs_rejected, 1);

        if rep.ort_dylib_loaded {
            let find_m = |n: &str| rep.models.iter().find(|m| m.name == n);
            let d2 = find_m("xvs_d2").expect("xvs_d2 entry");
            assert!(d2.loaded && d2.sha256.as_deref() == Some(EXPECTED_D2_ONNX_SHA256));
            let th = find_m("xvs_thermal").expect("xvs_thermal entry");
            assert!(
                th.loaded && th.sha256.as_deref() == Some(EXPECTED_THERMAL_V52_FP16_ONNX_SHA256),
                "Expected promoted v5.2 thermal ONNX: {th:?}"
            );
            assert!(
                th.output_summary.contains("dual_blind_vacuity=1.0000"),
                "Expected bit-exact dual-blind vacuity == 1.0000: {}",
                th.output_summary
            );
            let land = find_m("xvs_land").expect("xvs_land entry");
            assert!(land.loaded, "XVS-Land v5.2 failed to load: {land:?}");
            assert_eq!(
                land.sha256.as_deref(),
                Some(EXPECTED_LAND_V52_FP16_SHA256)
            );
            assert!(
                land.output_summary.contains("contract=v52_6output"),
                "Expected v5.2 6-output land contract: {}",
                land.output_summary
            );
            let nav = find_m("xvs_nav_v5").expect("xvs_nav_v5 entry");
            assert!(nav.loaded);
            assert_eq!(nav.status, "READY_RUST_ORT_V5");
            assert_eq!(
                nav.sha256.as_deref(),
                Some(EXPECTED_NAV_V5_FAZB_FP16_ONNX_SHA256)
            );
            assert!(nav.output_summary.contains("dim=512"));
            assert!(nav.output_summary.contains("l2_norm=1.000"));
            let xf = find_m("xvs_xfeat").expect("xvs_xfeat entry");
            assert!(xf.loaded);
            assert_eq!(xf.status, "READY_RUST_ORT_V5");
            assert_eq!(xf.sha256.as_deref(), Some(EXPECTED_XFEAT_FP16_ONNX_SHA256));
            let nk: usize = xf
                .output_summary
                .strip_prefix("xfeat_keypoints_count=")
                .expect("expected xfeat_keypoints_count= prefix")
                .parse()
                .expect("numeric xfeat keypoint count");
            assert!(nk > 0, "Expected >0 XFeat keypoints on textured probe, got {nk}");
            let lg = find_m("xvs_lighterglue").expect("xvs_lighterglue entry");
            assert!(lg.loaded, "LighterGlue FP16 failed to load: {lg:?}");
            assert_eq!(lg.status, "READY_RUST_ORT_V5");
            assert_eq!(
                lg.sha256.as_deref(),
                Some(EXPECTED_LIGHTERGLUE_FP16_ONNX_SHA256)
            );
            assert!(lg.output_summary.contains("scores_shape=1x256x256"));
            assert!(
                !lg.output_summary.contains("mutual_matches=0 "),
                "Expected >0 LighterGlue mutual matches: {}",
                lg.output_summary
            );
            println!(
                "[EDGE-FAZB-TRIAD] nav_s1={:?}ms ({}) | xfeat_fp16={:?}ms ({}) | lighterglue_fp16={:?}ms ({})",
                nav.latency_ms,
                nav.output_summary,
                xf.latency_ms,
                xf.output_summary,
                lg.latency_ms,
                lg.output_summary
            );
        }
        assert!(rep.sensor_optics_hardening.brown_conrady_max_err_px < 1e-12);
        assert!(rep.sensor_optics_hardening.kannala_brandt_140deg_max_err_px < 1e-12);
        assert!(rep.sensor_optics_hardening.spatial_4x4_12sector_localized_fault_ok);
        assert!(rep.sensor_optics_hardening.lwir_ffc_zero_tick_and_contrast_collapse_ok);
        assert!(rep.sensor_optics_hardening.dual_tier_fsync_and_slerp_ok);
        assert!(rep.sensor_optics_hardening.bearing_tbc_3_of_5_ok);
    }

    #[test]
    fn test_edge_v52_optics_spatial_ffc_fsync_slerp_and_tbc() {
        let pin = PinholeIntrinsics::new(
            1280.0, 1280.0, 640.0, 360.0, -0.24, 0.08, 0.0012, -0.0009, -0.015,
        )
        .expect("valid pinhole");
        let mut max_bc = 0.0_f64;
        for gy in 0..10 {
            for gx in 0..10 {
                let u = 100.0 + (gx as f64) * 110.0;
                let v = 50.0 + (gy as f64) * 65.0;
                let d = pin.distort_pixel([u, v]).unwrap();
                let r = pin.undistort_pixel_exact(d).unwrap();
                max_bc = max_bc.max((r[0] - u).hypot(r[1] - v));
                let ray = pin.undistort_to_unit_ray_rdf(d).unwrap();
                assert!((ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2] - 1.0).abs() < 1e-14);
            }
        }
        assert!(max_bc < 1e-12, "Brown-Conrady max err {max_bc:e}");

        let kb = KannalaBrandtFisheye::new(
            420.0, 420.0, 640.0, 360.0, -0.025, 0.008, -0.002, 0.0004,
        )
        .expect("valid fisheye");
        let mut max_kb = 0.0_f64;
        for deg in [0.0_f64, 25.0, 55.0, 70.0] {
            let t = deg.to_radians();
            for az in 0..8 {
                let phi = (az as f64) * core::f64::consts::FRAC_PI_4;
                let ray = [t.sin() * phi.cos(), t.sin() * phi.sin(), t.cos()];
                let d = kb.project_ray_rdf(ray).unwrap();
                let u_ray = kb.undistort_to_unit_ray_exact(d).unwrap();
                let re = kb.project_ray_rdf(u_ray).unwrap();
                max_kb = max_kb.max((re[0] - d[0]).hypot(re[1] - d[1]));
                let uv = kb.undistort_pixel(d, 30, 1e-15).unwrap();
                let re2 = kb.distort_pixel(uv).unwrap();
                assert!((re2[0] - d[0]).hypot(re2[1] - d[1]) < 1e-12);
            }
        }
        assert!(max_kb < 1e-12, "Kannala-Brandt max err {max_kb:e}");

        let mut tbc = BearingTrackBeforeConfirm::new_3_of_5();
        assert!(!tbc.observe(Some((0.0, 0.0))).confirmed);
        assert!(!tbc.observe(Some((0.5, 0.5))).confirmed); // outside gate
        assert!(!tbc.observe(Some((0.01, 0.01))).confirmed);
        assert!(tbc.observe(Some((0.015, 0.012))).confirmed);
        for _ in 0..5 {
            let _ = tbc.observe(None);
        }
        assert!(!tbc.is_confirmed());
    }

    #[test]
    fn test_edge_xvs_land_dual_contract_5_and_6_output_and_fail_closed() {
        let cal = StudentTCalibrator::new(LAND_STUDENT_T_NU, LAND_STUDENT_T_SCALE).unwrap();
        let mut kpts = vec![0.0_f32; LAND_NUM_KEYPOINTS * 2];
        for i in 0..LAND_NUM_KEYPOINTS {
            kpts[2 * i] = 60.0 + (i as f32) * 10.0;
            kpts[2 * i + 1] = 48.0 + (i as f32) * 12.0;
        }
        let mut keylines = vec![0.0_f32; LAND_NUM_KEYLINES * 3];
        for j in 0..LAND_NUM_KEYLINES {
            keylines[3 * j] = 90.0 + (j as f32) * 15.0;
            keylines[3 * j + 1] = 0.8;
            keylines[3 * j + 2] = 0.6;
        }
        let mut cov = vec![0.0_f32; LAND_NUM_KEYPOINTS * 4];
        for i in 0..LAND_NUM_KEYPOINTS {
            cov[4 * i] = 1.5;
            cov[4 * i + 3] = 0.5;
        }

        // 1. 5-output (v4.5/v5.1) backward compatibility
        let p5 = parse_xvs_land_outputs(&kpts, &keylines, &cov, None, &cal)
            .expect("5-output parse must succeed");
        assert_eq!(p5.contract, LandOutputContract::V51FiveOutput);
        assert!(p5.keyline_sigma_px.is_none());
        let expected_w = 1.0 / (LAND_DEFAULT_KEYLINE_SIGMA_PX * LAND_DEFAULT_KEYLINE_SIGMA_PX);
        for w in p5.keyline_precision_weights {
            assert!((w - expected_w).abs() < 1e-12);
        }
        assert!((p5.effective_keyline_sigma_px - LAND_DEFAULT_KEYLINE_SIGMA_PX).abs() < 1e-12);

        // 2. 6-output (v5.2) per-line keyline_sigma_px inverse-variance precision weighting
        let sigmas = [0.75_f32, 1.5, 2.0, 5.0, 40.0];
        let p6 = parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(&sigmas), &cal)
            .expect("6-output parse must succeed");
        assert_eq!(p6.contract, LandOutputContract::V52SixOutputKeylineSigma);
        let parsed_sigs = p6.keyline_sigma_px.expect("keyline_sigma_px must be Some");
        assert!((parsed_sigs[0] - 0.75).abs() < 1e-6);
        assert!((parsed_sigs[4] - 40.0).abs() < 1e-6);
        assert!((p6.keyline_precision_weights[0] - (1.0 / (0.75 * 0.75))).abs() < 1e-12);
        assert!((p6.keyline_precision_weights[4] - (1.0 / 1600.0)).abs() < 1e-12);
        assert!(p6.keyline_precision_weights[0] > p6.keyline_precision_weights[4] * 2500.0);

        // 3. Fail-closed rejection on NaN / Inf / <= 0 / truncated
        let bad_cases: &[&[f32]] = &[
            &[f32::NAN, 1.0, 1.0, 1.0, 1.0],
            &[1.0, f32::INFINITY, 1.0, 1.0, 1.0],
            &[1.0, f32::NEG_INFINITY, 1.0, 1.0, 1.0],
            &[1.0, 1.0, 0.0, 1.0, 1.0],
            &[1.0, 1.0, -2.0, 1.0, 1.0],
            &[1.0, 2.0, 3.0, 4.0],
        ];
        for bad in bad_cases {
            assert!(
                parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(bad), &cal).is_err(),
                "Expected fail-closed rejection for bad keyline_sigma_px={bad:?}"
            );
        }
    }

    #[test]
    fn test_edge_r4_keyline_sigma_physical_floor_clamps_subnormal() {
        let cal = StudentTCalibrator::new(LAND_STUDENT_T_NU, LAND_STUDENT_T_SCALE).unwrap();
        let kpts = vec![100.0_f32; LAND_NUM_KEYPOINTS * 2];
        let keylines = vec![1.0_f32; LAND_NUM_KEYLINES * 3];
        let mut cov = vec![0.0_f32; LAND_NUM_KEYPOINTS * 4];
        for i in 0..LAND_NUM_KEYPOINTS {
            cov[4 * i] = 1.0;
            cov[4 * i + 3] = 1.0;
        }

        // Subnormal f32::from_bits(1) (~1.4e-45) MUST clamp to MIN_KEYLINE_SIGMA_PX (0.25 px)
        let sub = f32::from_bits(1);
        assert!(sub > 0.0 && sub.is_finite() && (sub as f64) < 1e-40);
        let all_sub = [sub; LAND_NUM_KEYLINES];
        let parsed_all = parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(&all_sub), &cal)
            .expect("positive subnormal keyline_sigma_px must clamp cleanly");
        assert!(
            parsed_all.effective_keyline_sigma_px >= MIN_KEYLINE_SIGMA_PX - 1e-12,
            "effective_keyline_sigma_px collapsed below floor: {}",
            parsed_all.effective_keyline_sigma_px
        );
        assert!((parsed_all.effective_keyline_sigma_px - MIN_KEYLINE_SIGMA_PX).abs() < 1e-12);
        let sigs = parsed_all.keyline_sigma_px.expect("Some sigmas");
        for k in 0..LAND_NUM_KEYLINES {
            assert!((sigs[k] - MIN_KEYLINE_SIGMA_PX).abs() < 1e-12);
            assert!((parsed_all.keyline_precision_weights[k] - 16.0).abs() < 1e-12);
        }

        // Mixed subnormal + sub-floor + nominal sigmas clamp only values < 0.25 px
        let mixed = [sub, 0.10_f32, 1.0, 2.0, 4.0];
        let parsed_mix = parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(&mixed), &cal)
            .expect("mixed subnormal sigmas must parse cleanly");
        let mix_sigs = parsed_mix.keyline_sigma_px.expect("Some mixed sigmas");
        assert!((mix_sigs[0] - MIN_KEYLINE_SIGMA_PX).abs() < 1e-12);
        assert!((mix_sigs[1] - MIN_KEYLINE_SIGMA_PX).abs() < 1e-12);
        assert!((mix_sigs[2] - 1.0).abs() < 1e-6);
        assert!(parsed_mix.effective_keyline_sigma_px >= MIN_KEYLINE_SIGMA_PX);
        assert!(parsed_mix.effective_keyline_sigma_px.is_finite());

        // Zero, negative, NaN, and Inf MUST STILL fail closed
        for bad in [
            [0.0_f32, 1.0, 1.0, 1.0, 1.0],
            [-0.25_f32, 1.0, 1.0, 1.0, 1.0],
            [f32::NAN, 1.0, 1.0, 1.0, 1.0],
            [f32::INFINITY, 1.0, 1.0, 1.0, 1.0],
        ] {
            assert!(parse_xvs_land_outputs(&kpts, &keylines, &cov, Some(&bad), &cal).is_err());
        }
    }

    #[test]
    fn test_edge_canonical_json_python_compat_and_replay_tamper_detection() {
        // Verify em-dash U+2014 ASCII escaping + Python float formatting parity
        let v = serde_json::json!({
            "watermark": NON_FLIGHT_WATERMARK,
            "down_m": -180.0,
            "tiny": 1.0e-5,
            "zero": 0.0,
            "a_first": true,
        });
        let canon = canonical_json_python_compat(&v).expect("canonicalize");
        assert!(canon.contains("\\u2014"), "em-dash must escape to \\u2014: {canon}");
        assert!(canon.contains("\"down_m\":-180.0"), "float formatting: {canon}");
        assert!(canon.contains("\"tiny\":1e-05"), "sci float formatting: {canon}");
        assert!(canon.starts_with("{\"a_first\":true,"), "keys must be sorted: {canon}");

        // Build a valid 3-record shadow NDJSON file and prove verify + 1-byte tamper rejection
        let tmp = std::env::temp_dir().join("openxvs_edge_replay_unit_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir");
        let valid_path = tmp.join("valid_shadow.ndjson");
        let mut lines: Vec<String> = Vec::new();
        let mut prev_h = SHADOW_GENESIS_HASH.to_string();
        for seq in 0u64..3 {
            let mut body = serde_json::json!({
                "schema": SHADOW_MODE_LOG_SCHEMA,
                "seq": seq,
                "timestamp_mono_us": (seq + 1) * 100_000,
                "trial_phase": "phase3_university_shadow_tx_pin_removed",
                "hardware_tx_pin_removed": true,
                "event_type": "rx_mavlink2",
                "direction": "rx",
                "tx_inhibited": true,
                "actuator_authority": false,
                "outbound_mavlink_frames_emitted": 0,
                "non_flight_sim_only": true,
                "watermark": NON_FLIGHT_WATERMARK,
                "payload": {
                    "down_m": -180.0 - (seq as f64),
                    "gnss_ne_m": [(seq as f64) * 15.0, 0.0],
                    "vis_ne_m": [(seq as f64) * 15.0, 0.0],
                    "gnss_down_m": -180.0 - (seq as f64),
                },
                "prev_hash": prev_h,
            });
            let canon_body = canonical_json_python_compat(&body).expect("canon body");
            let dig = Sha256::digest(canon_body.as_bytes());
            let mut hex = String::with_capacity(64);
            for b in dig {
                hex.push_str(&format!("{b:02x}"));
            }
            body.as_object_mut()
                .unwrap()
                .insert("entry_hash".to_string(), serde_json::Value::String(hex.clone()));
            lines.push(canonical_json_python_compat(&body).expect("canon line"));
            prev_h = hex;
        }
        std::fs::write(&valid_path, format!("{}\n", lines.join("\n"))).expect("write valid");

        let rep_ok = verify_and_replay_shadow_ndjson(&valid_path);
        assert!(rep_ok.valid, "valid chain rejected: {:?}", rep_ok.errors);
        assert_eq!(rep_ok.record_count, 3);
        assert_eq!(rep_ok.cusum_epochs_replayed, 3);
        assert_eq!(rep_ok.rta_ticks_replayed, 3);
        assert!(rep_ok.tx_inhibited_all && rep_ok.actuator_authority_all_false);

        // 1-byte tamper in line 2 payload -> MUST fail closed
        let tamper_path = tmp.join("tampered_shadow.ndjson");
        let mut bad_lines = lines.clone();
        bad_lines[1] = bad_lines[1].replacen("-181.0", "-182.0", 1);
        std::fs::write(&tamper_path, format!("{}\n", bad_lines.join("\n"))).expect("write tamper");
        let rep_bad = verify_and_replay_shadow_ndjson(&tamper_path);
        assert!(!rep_bad.valid, "1-byte payload tamper must fail closed!");
        assert_eq!(rep_bad.first_failed_line, Some(2));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_edge_wave12_sensor_health_parity_and_no_json_null() {
        // 1. Fresh SensorHealthMonitor must NOT leak NaN / JSON `null` in lwir_last_mad
        let mut shm = SensorHealthMonitor::new();
        assert_eq!(shm.lwir_last_mad, -1.0);
        assert!(shm.lwir_last_mad.is_finite());

        // 2. Cold-sky 2x2 min-filter rescues 3x3 real hot intruder while rejecting 1-px hot spike
        let w = 160usize;
        let h = 128usize;
        let mut cold_sky_target = vec![30u8; w * h];
        for y in 50..53 {
            for x in 70..73 {
                cold_sky_target[y * w + x] = 220;
            }
        }
        let st_target = shm.observe_lwir_plane_with_ffc(&cold_sky_target, w, h, 0);
        let lwir_json = serde_json::to_string(&st_target).expect("serialize LwirHealthStatus");
        assert!(
            !lwir_json.contains("null"),
            "LwirHealthStatus must never serialize NaN as JSON null: {lwir_json}"
        );
        assert_eq!(st_target.last_mad, -1.0);
        assert!(
            !st_target.contrast_collapsed && !st_target.mask_lwir_frame,
            "3x3 hot target over cold sky must NOT falsely collapse contrast: {st_target:?}"
        );

        let mut shm_spike = SensorHealthMonitor::new();
        let mut cold_sky_spike = vec![30u8; w * h];
        cold_sky_spike[51 * w + 71] = 240; // single isolated dead/hot pixel
        let st_spike = shm_spike.observe_lwir_plane_with_ffc(&cold_sky_spike, w, h, 0);
        assert!(
            st_spike.contrast_collapsed && st_spike.mask_lwir_frame,
            "1-px hot spike on flat cold sky must collapse contrast: {st_spike:?}"
        );

        // 3. Dead-line FPN detection (>= 10% spaced stuck readout columns -> fail)
        let mut fpn_frame = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                fpn_frame[y * w + x] = (((x * 7 + y * 13) % 140) + 40) as u8;
            }
        }
        for col in (4..(w - 4)).step_by(6) {
            for y in 0..h {
                fpn_frame[y * w + col] = 255;
            }
        }
        let eo_fpn = shm.observe_eo_luma_u8(&fpn_frame, w, h);
        assert!(
            eo_fpn.dead_pixel_fpn_fault && eo_fpn.dead_line_fraction >= FPN_DEAD_LINE_FAIL_FRAC,
            "Spaced dead columns must trip FPN fault: {eo_fpn:?}"
        );

        // 4. White-noise autocorrelation fault on synthetic LCG white noise
        let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut noise_frame = vec![0u8; w * h];
        for px in &mut noise_frame {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *px = ((lcg >> 33) & 0xFF) as u8;
        }
        let eo_noise = shm.observe_eo_luma_u8(&noise_frame, w, h);
        assert!(
            eo_noise.sensor_noise_fault,
            "Uncorrelated full-scale white noise must trip sensor_noise_fault: {eo_noise:?}"
        );

        // 5. BearingTrackBeforeConfirm::observe_with_health fails closed on degraded EO
        let mut tbc = BearingTrackBeforeConfirm::new_3_of_5();
        assert!(!tbc.observe_with_health(Some((0.1, 0.0)), 0.95, false).confirmed);
        assert!(!tbc.observe_with_health(Some((0.101, 0.001)), 0.95, false).confirmed);
        let degraded = tbc.observe_with_health(Some((0.102, 0.002)), 0.20, true);
        assert_eq!(degraded.hits_in_window, 2);
        assert!(!degraded.confirmed);
        assert!(tbc.observe_with_health(Some((0.102, 0.002)), 0.95, false).confirmed);

        // 6. system32 guard only blocks actual Windows system32/syswow64 directories
        assert!(!is_forbidden_system_ort_path(Path::new(
            r"C:\my_system32_backup\onnxruntime.dll"
        )));
    }

    #[test]
    fn test_edge_xvs_land_v53_7th_output_presence_gate() {
        let cal = StudentTCalibrator::new(V52_LAND_STUDENT_T_NU, V52_LAND_STUDENT_T_SCALE).unwrap();
        let kpts = vec![100.0_f32; LAND_NUM_KEYPOINTS * 2];
        let keylines = vec![1.0_f32; LAND_NUM_KEYLINES * 3];
        let mut cov = vec![0.0_f32; LAND_NUM_KEYPOINTS * 4];
        for i in 0..LAND_NUM_KEYPOINTS {
            cov[4 * i] = 1.0;
            cov[4 * i + 3] = 1.0;
        }
        let sigmas = [1.0_f32; LAND_NUM_KEYLINES];

        // Runway strongly present: sigmoid(2.0) ~= 0.8808 >= 0.65 -> NOMINAL
        let p_ok = parse_xvs_land_outputs_with_presence(
            &kpts,
            &keylines,
            &cov,
            Some(&sigmas),
            Some(&[2.0_f32]),
            &cal,
        )
        .expect("extended 7-output present parse");
        assert_eq!(p_ok.contract, LandOutputContract::V53SevenOutputRunwayPresence);
        assert!(p_ok.runway_present_admitted);
        assert_eq!(p_ok.runway_gate_mode, "NOMINAL");
        assert!(p_ok.runway_present_prob.unwrap() > 0.85);

        // Runway absent: sigmoid(-1.0) ~= 0.2689 < 0.65 -> fail-closed P0_INVALID_INHIBIT
        let p_no = parse_xvs_land_outputs_with_presence(
            &kpts,
            &keylines,
            &cov,
            Some(&sigmas),
            Some(&[-1.0_f32]),
            &cal,
        )
        .expect("extended 7-output absent parse");
        assert_eq!(p_no.contract, LandOutputContract::V53SevenOutputRunwayPresence);
        assert!(!p_no.runway_present_admitted);
        assert_eq!(p_no.runway_gate_mode, "P0_INVALID_INHIBIT");
        assert!(p_no.runway_present_prob.unwrap() < LAND_RUNWAY_PRESENT_MIN_PROB);

        // Non-finite logit -> fail-closed P0_INVALID_INHIBIT
        let p_nan = parse_xvs_land_outputs_with_presence(
            &kpts,
            &keylines,
            &cov,
            Some(&sigmas),
            Some(&[f32::NAN]),
            &cal,
        )
        .expect("extended 7-output nan logit fails closed");
        assert!(!p_nan.runway_present_admitted);
        assert_eq!(p_nan.runway_gate_mode, "P0_INVALID_INHIBIT");

        // Empty 7th tensor -> Err
        assert!(parse_xvs_land_outputs_with_presence(
            &kpts,
            &keylines,
            &cov,
            Some(&sigmas),
            Some(&[]),
            &cal,
        )
        .is_err());
    }

    #[test]
    fn test_edge_replay_missing_vis_ne_does_not_fallback_to_gnss() {
        let tmp = std::env::temp_dir().join("openxvs_edge_replay_no_vis_fallback_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir");
        let path = tmp.join("no_vis_spoof.ndjson");
        let mut lines: Vec<String> = Vec::new();
        let mut prev_h = SHADOW_GENESIS_HASH.to_string();
        for seq in 0u64..12 {
            // GNSS jumps +250 m North while INS accel stays 0 and `vis_ne_m` is OMITTED.
            // With fail-closed `vis_ne = None`, Bank-B dead-reckons near 0 while Bank-A tracks
            // spoofed GNSS -> CUSUM diverges and alarms! (Old `.or(gnss_ne)` hid the spoof.)
            let gnss_n = if seq >= 2 { 250.0 } else { 0.0 };
            let mut body = serde_json::json!({
                "schema": SHADOW_MODE_LOG_SCHEMA,
                "seq": seq,
                "timestamp_mono_us": (seq + 1) * 100_000,
                "trial_phase": "phase3_university_shadow_tx_pin_removed",
                "hardware_tx_pin_removed": true,
                "event_type": "rx_mavlink2",
                "direction": "rx",
                "tx_inhibited": true,
                "actuator_authority": false,
                "outbound_mavlink_frames_emitted": 0,
                "non_flight_sim_only": true,
                "watermark": NON_FLIGHT_WATERMARK,
                "payload": {
                    "down_m": -120.0,
                    "gnss_ne_m": [gnss_n, 0.0],
                    "gnss_down_m": -120.0,
                },
                "prev_hash": prev_h,
            });
            let canon_body = canonical_json_python_compat(&body).expect("canon body");
            let dig = Sha256::digest(canon_body.as_bytes());
            let mut hex = String::with_capacity(64);
            for b in dig {
                hex.push_str(&format!("{b:02x}"));
            }
            body.as_object_mut()
                .unwrap()
                .insert("entry_hash".to_string(), serde_json::Value::String(hex.clone()));
            lines.push(canonical_json_python_compat(&body).expect("canon line"));
            prev_h = hex;
        }
        std::fs::write(&path, format!("{}\n", lines.join("\n"))).expect("write log");
        let rep = verify_and_replay_shadow_ndjson(&path);
        assert!(rep.valid, "chain must be valid: {:?}", rep.errors);
        assert!(
            rep.cusum_max_stat > 1.0 && rep.cusum_alarms > 0,
            "Missing vis_ne_m must NOT fall back to gnss_ne_m; expected Bank-A/B divergence & alarm: {rep:?}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_edge_parity_eval_malformed_short_arrays_return_err() {
        let bad_gallery = serde_json::json!({
            "kind": "sequence_vpr",
            "gallery_ne_m": [[10.0]],
            "steps": [],
        });
        assert!(eval_parity_case(&bad_gallery).is_err());

        let bad_init_ned = serde_json::json!({
            "kind": "dual_bank_cusum",
            "reset": { "init_ned_m": [1.0, 2.0] },
            "steps": [],
        });
        assert!(eval_parity_case(&bad_init_ned).is_err());

        let bad_init_vel = serde_json::json!({
            "kind": "dual_bank_cusum",
            "reset": { "init_vel_ne_mps": [1.0] },
            "steps": [],
        });
        assert!(eval_parity_case(&bad_init_vel).is_err());

        let bad_gnss = serde_json::json!({
            "kind": "dual_bank_cusum",
            "steps": [{ "gnss_ne_m": [10.0] }],
        });
        assert!(eval_parity_case(&bad_gnss).is_err());

        let bad_vis = serde_json::json!({
            "kind": "dual_bank_cusum",
            "steps": [{ "vis_ne_m": [10.0] }],
        });
        assert!(eval_parity_case(&bad_vis).is_err());
    }

    #[test]
    fn test_edge_cli_help_and_eval_frames_argument_hardening() {
        let help = cli_help_text();
        assert!(help.contains("--eval-frames <DIR>"));
        assert!(help.contains("--mode all|eo|thermal|land"));
        assert!(help.contains("--out <REPORT.JSON>"));
        assert!(help.contains("--tape <TAPE.NDJSON>"));
        assert!(help.contains("--max-frames <N>"));
        assert!(help.contains("--model-dir <DIR>"));
        assert!(help.contains("--replay <SHADOW.NDJSON>"));
        assert!(help.contains(NON_FLIGHT_WATERMARK));

        let s = |vals: &[&str]| vals.iter().map(|v| (*v).to_string()).collect::<Vec<_>>();

        assert!(matches!(parse_cli_command(&s(&["--help"])), Ok(CliCommand::Help)));
        assert!(matches!(parse_cli_command(&s(&["-h"])), Ok(CliCommand::Help)));
        assert!(matches!(
            parse_cli_command(&s(&["--eval-frames", "--help"])),
            Ok(CliCommand::Help)
        ));
        assert!(matches!(
            parse_cli_command(&s(&["--version"])),
            Ok(CliCommand::Version)
        ));

        // Valid full --eval-frames invocation
        match parse_cli_command(&s(&[
            "--eval-frames",
            "frames_dir",
            "--mode",
            "thermal",
            "--out",
            "report.json",
            "--tape",
            "tape.ndjson",
            "--max-frames",
            "25",
            "--model-dir",
            "models",
        ])) {
            Ok(CliCommand::EvalFrames(cfg)) => {
                assert_eq!(cfg.input_dir, PathBuf::from("frames_dir"));
                assert_eq!(cfg.mode, eval_frames::EvalMode::Thermal);
                assert_eq!(cfg.out_path, Some(PathBuf::from("report.json")));
                assert_eq!(cfg.tape_path, Some(PathBuf::from("tape.ndjson")));
                assert_eq!(cfg.max_frames, Some(25));
                assert_eq!(cfg.model_dir, Some(PathBuf::from("models")));
            }
            other => panic!("Expected EvalFrames command, got {other:?}"),
        }

        // Edge cases must fail cleanly (no panics!)
        for bad_args in [
            s(&["--eval-frames"]),
            s(&["--eval-frames", "--mode", "eo"]),
            s(&["--eval-frames", "dir", "--mode"]),
            s(&["--eval-frames", "dir", "--mode", "bogus"]),
            s(&["--eval-frames", "dir", "--out"]),
            s(&["--eval-frames", "dir", "--tape"]),
            s(&["--eval-frames", "dir", "--model-dir"]),
            s(&["--eval-frames", "dir", "--max-frames"]),
            s(&["--eval-frames", "dir", "--max-frames", "0"]),
            s(&["--eval-frames", "dir", "--max-frames", "-3"]),
            s(&["--eval-frames", "dir", "--max-frames", "abc"]),
            s(&["--eval-frames", "dir", "--unknown-flag"]),
            s(&["--self-check", "--model-dir"]),
            s(&["--soak", "--seconds"]),
            s(&["--soak", "--hz"]),
            s(&["--replay"]),
            s(&["--replay", "a.ndjson", "extra.ndjson"]),
            s(&["--parity-eval"]),
            s(&["--unknown-top-level"]),
        ] {
            assert!(
                parse_cli_command(&bad_args).is_err(),
                "Expected CLI parse error for {bad_args:?}"
            );
        }
    }
}
