// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Pure-Rust offline custom-footage evaluation mode (`--eval-frames`) for `openxvs-edge-infer`.
//!
//! Security & Safety Invariants:
//! - `#![forbid(unsafe_code)]` (inherited from crate root)
//! - Non-flight simulation & offline evaluation only (`NON_FLIGHT_SIM_ONLY`, `ACTUATOR_AUTHORITY = false`, `0` outbound MAVLink frames)
//! - Strict decompression-bomb & corrupt-image guards (`<= 16 MiB` file size, `<= 4096x4096` dimensions, `<= 64 MiB` decode alloc, PNG/JPEG only)
//! - Fail-closed SHA-256 pinning on every ONNX model required by `--mode`
//! - Exact Studio Wave-1/2 parity for `SensorHealthMonitor`, `BearingTrackBeforeConfirm`, `CoordinatedActiveRange`, `SimplexRta`, and SHA-256 hash-chained `tape.ndjson`

use std::collections::VecDeque;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{ImageFormat, ImageReader, Limits, RgbImage};
use ort::session::Session;
use ort::value::TensorRef;
use serde::Serialize;
use sha2::{Digest, Sha256};
use xvs_active_range::{ActiveRangeConfig, ActiveRangeMeasurement, OwnshipStateNed};
use xvs_multiship_coord::{CoordinatedActiveRange, Ring0AuthorityGate};
use xvs_rta_core::{build_rta_input_from_f64, RtaFloatTelemetry, SimplexMode, SimplexRta};

use crate::nav_integrity::CUSUM_H_LOCKED;
use crate::{
    acquire_edge_ort_env_with_hint, build_optimized_ort_session, canonical_json_python_compat,
    detect_dead_line_fpn_u8, detect_white_noise_fault_u8, find_model_path,
    parse_xvs_land_outputs_with_presence, percentile_sorted, resolve_d2_input_shapes,
    resolve_default_model_dir, resolve_land_input_hw, round2, round4, verify_pinned_onnx_artifact,
    BearingTrackBeforeConfirm, BearingTrackStatus, EoOpticalQuality, LwirHealthStatus,
    SensorHealthMonitor, StudentTCalibrator, ACTUATOR_AUTHORITY, D2_ONNX_PINS, EDGE_INFER_VERSION,
    EO_SECTOR_DEGRADED_THR, EXPECTED_LAND_V52_FP16_SHA256, EXPECTED_LAND_V52_FP32_SHA256,
    LAND_NUM_KEYPOINTS, LAND_ONNX_PINS, LAND_STUDENT_T_NU, LAND_STUDENT_T_SCALE,
    NON_FLIGHT_SIM_ONLY, NON_FLIGHT_WATERMARK, SHADOW_GENESIS_HASH, SHADOW_MODE_LOG_SCHEMA,
    THERMAL_ONNX_PINS, V52_LAND_STUDENT_T_NU, V52_LAND_STUDENT_T_SCALE,
};

pub const MAX_EVAL_IMAGE_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_EVAL_IMAGE_DIM: u32 = 4096;
pub const MAX_EVAL_DECODE_ALLOC_BYTES: u64 = 64 * 1024 * 1024;
pub const AOT_FOCAL_PX: f64 = 2690.0;
pub const AOT_REF_WIDTH_PX: f64 = 2448.0;
pub const LWIR_L_SAT_PROXY: f32 = 215.7;
pub const D2_SCORE_THRESHOLD: f32 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalMode {
    All,
    Eo,
    Thermal,
    Land,
}

impl EvalMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "all" => Ok(Self::All),
            "eo" => Ok(Self::Eo),
            "thermal" => Ok(Self::Thermal),
            "land" => Ok(Self::Land),
            other => Err(format!(
                "Invalid --mode '{other}'; expected one of: all, eo, thermal, land"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Eo => "eo",
            Self::Thermal => "thermal",
            Self::Land => "land",
        }
    }

    pub fn needs_eo(&self) -> bool {
        matches!(self, Self::All | Self::Eo)
    }

    pub fn needs_thermal(&self) -> bool {
        matches!(self, Self::All | Self::Thermal)
    }

    pub fn needs_land(&self) -> bool {
        matches!(self, Self::All | Self::Land)
    }
}

#[derive(Debug, Clone)]
pub struct EvalFramesConfig {
    pub input_dir: PathBuf,
    pub mode: EvalMode,
    pub out_path: Option<PathBuf>,
    pub tape_path: Option<PathBuf>,
    pub max_frames: Option<usize>,
    pub model_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct EvalFramesFailure {
    pub error_code: &'static str,
    pub message: String,
}

impl EvalFramesFailure {
    pub fn new(error_code: &'static str, message: impl Into<String>) -> Self {
        Self {
            error_code,
            message: message.into(),
        }
    }

    pub fn into_error_report(self) -> EvalFramesErrorReport {
        EvalFramesErrorReport {
            mode: "eval_frames",
            ok: false,
            version: EDGE_INFER_VERSION,
            non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
            actuator_authority: ACTUATOR_AUTHORITY,
            outbound_mavlink_frames: 0,
            watermark: NON_FLIGHT_WATERMARK,
            error_code: self.error_code,
            error: self.message,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EvalFramesErrorReport {
    pub mode: &'static str,
    pub ok: bool,
    pub version: &'static str,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub outbound_mavlink_frames: u64,
    pub watermark: &'static str,
    pub error_code: &'static str,
    pub error: String,
}

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

#[derive(Debug, Serialize)]
pub struct LoadedModelInfo {
    pub name: &'static str,
    pub filename: String,
    pub sha256: String,
    pub load_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct FrameSensorHealth {
    pub q_eo_effective: f64,
    pub q_lwir_effective: f64,
    pub eo_blur_score: f64,
    pub eo_focus_quality: f64,
    pub eo_glare_ratio: f64,
    pub eo_occlusion_ratio: f64,
    pub eo_dead_line_fraction: f64,
    pub eo_spatial_autocorr: f64,
    pub eo_dead_pixel_fpn_fault: bool,
    pub eo_sensor_noise_fault: bool,
    pub eo_degraded: bool,
    pub eo_flags: Vec<&'static str>,
    pub lwir_nuc_frozen: bool,
    pub lwir_freeze_ticks: u32,
    pub lwir_last_mad: f64,
    pub lwir_dynamic_range_p95_p05: f64,
    pub lwir_contrast_collapsed: bool,
    pub lwir_dead_pixel_fpn_fault: bool,
    pub lwir_sensor_noise_fault: bool,
    pub lwir_q_optical: f64,
    pub lwir_degraded: bool,
    pub lwir_flags: Vec<&'static str>,
    pub dual_blind_detected: bool,
}

#[derive(Debug, Serialize)]
pub struct FrameD2Result {
    pub latency_ms: f64,
    pub detected: bool,
    pub raw_detected: bool,
    pub max_prob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox_xyxy_px: Option<[f64; 4]>,
    pub score: f64,
    pub alpha_rad: f64,
    pub beta_rad: f64,
    pub tbc_status: &'static str,
    pub tbc_confirmed: bool,
    pub tbc_hits_in_window: u8,
    pub active_range_mode: &'static str,
    pub r_hat_m: f64,
    pub r_safe_lower_m: f64,
    pub cmd_bank_deg: f64,
}

#[derive(Debug, Serialize)]
pub struct FrameThermalResult {
    pub latency_ms: f64,
    pub thermal_mode: &'static str,
    pub dual_blind: bool,
    pub mean_vacuity: f64,
    pub max_hm_prob: f64,
    pub w_eo_eff: f64,
    pub w_lwir_eff: f64,
    pub peak_xy: [usize; 2],
}

#[derive(Debug, Serialize)]
pub struct FrameLandResult {
    pub latency_ms: f64,
    pub contract: &'static str,
    pub roi_hw: [usize; 2],
    pub mean_keypoint_sigma_px: f64,
    pub effective_keyline_sigma_px: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runway_present_logit: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runway_present_prob: Option<f64>,
    pub runway_present_admitted: bool,
    pub runway_gate_mode: &'static str,
    pub glideslope_deg: f64,
    pub lateral_dev_deg: f64,
    pub hpl_m: f64,
    pub vpl_m: f64,
}

#[derive(Debug, Serialize)]
pub struct EvalFrameRecord {
    pub frame_index: usize,
    pub filename: String,
    pub status: &'static str,
    pub corrupt: bool,
    pub fail_closed_rejected: bool,
    pub fail_closed_reasons: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    pub latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensor_health: Option<FrameSensorHealth>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eo_d2: Option<FrameD2Result>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thermal: Option<FrameThermalResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub land: Option<FrameLandResult>,
    pub rta_mode: String,
    pub rta_mode_id: u8,
    pub rta_raw_target: String,
    pub rta_active_healthy_cams: u8,
    pub rta_inhibit_avoidance_turns: bool,
    pub rta_mandatory_go_around: bool,
}

#[derive(Debug, Serialize)]
pub struct EvalFramesSummary {
    pub frames_total: usize,
    pub frames_processed: usize,
    pub frames_corrupt: usize,
    pub models_run: Vec<String>,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_max_ms: f64,
    pub detections_total: usize,
    pub tbc_confirmed_tracks: usize,
    pub tbc_confirmed_frames: usize,
    pub fail_closed_rejections: usize,
    pub sensor_faults_detected: usize,
    pub dual_blind_frames: usize,
    pub rta_final_mode: String,
    pub rta_final_mode_id: u8,
    pub verdict_plain_english: String,
}

#[derive(Debug, Serialize)]
pub struct EvalFramesReport {
    pub mode: &'static str,
    pub ok: bool,
    pub eval_mode: &'static str,
    pub version: &'static str,
    pub non_flight_sim_only: bool,
    pub actuator_authority: bool,
    pub outbound_mavlink_frames: u64,
    pub watermark: &'static str,
    pub input_dir_name: String,
    pub models_loaded: Vec<LoadedModelInfo>,
    pub tape_written: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tape_sha256_head: Option<String>,
    pub summary: EvalFramesSummary,
    pub frames: Vec<EvalFrameRecord>,
}

// ---- Pre-allocated Scratch Buffers & Fast Pure-Rust Fixed-Point RGB Resizer ----

const RESIZE_FP_SHIFT: u32 = 14;
const RESIZE_FP_ONE: u32 = 1 << RESIZE_FP_SHIFT;
const RESIZE_FP_HALF: u32 = 1 << (RESIZE_FP_SHIFT - 1);

#[derive(Debug, Clone, Copy)]
struct AxisSpan {
    start: usize,
    end: usize,
    w_first: u32,
    w_mid: u32,
    w_last: u32,
}

struct ResizeScratch {
    spans_x: Vec<AxisSpan>,
    spans_y: Vec<AxisSpan>,
    horiz_rgb: Vec<u8>,
    v_acc: Vec<u32>,
    v_mid: Vec<u32>,
}

impl ResizeScratch {
    fn new() -> Self {
        Self {
            spans_x: Vec::with_capacity(512),
            spans_y: Vec::with_capacity(512),
            horiz_rgb: Vec::with_capacity(384 * 1080 * 3),
            v_acc: Vec::with_capacity(512 * 3),
            v_mid: Vec::with_capacity(512 * 3),
        }
    }
}

struct EvalFrameScratch {
    raw_file_buf: Vec<u8>,
    native_luma: Vec<u8>,
    eo_luma: Vec<f64>,
    resize: ResizeScratch,
    pyramid_a: Vec<u8>,
    pyramid_b: Vec<u8>,
    rgb_320x256: Vec<u8>,
    rgb_160x128: Vec<u8>,
    rgb_d2: Vec<u8>,
    rgb_land: Vec<u8>,
}

impl EvalFrameScratch {
    fn new() -> Self {
        Self {
            raw_file_buf: Vec::with_capacity(512 * 1024),
            native_luma: Vec::with_capacity(640 * 480),
            eo_luma: Vec::with_capacity(160 * 128),
            resize: ResizeScratch::new(),
            pyramid_a: Vec::new(),
            pyramid_b: Vec::new(),
            rgb_320x256: Vec::with_capacity(320 * 256 * 3),
            rgb_160x128: Vec::with_capacity(160 * 128 * 3),
            rgb_d2: Vec::with_capacity(256 * 256 * 3),
            rgb_land: Vec::with_capacity(384 * 384 * 3),
        }
    }
}

fn build_axis_spans(src_len: usize, dst_len: usize, out: &mut Vec<AxisSpan>) {
    out.clear();
    out.reserve(dst_len);
    if src_len == 0 || dst_len == 0 {
        return;
    }
    if src_len >= dst_len {
        // Exact fractional box-area downsampling (INTER_AREA) with DC-exact fixed-point weights.
        let scale = (src_len as f64) / (dst_len as f64);
        for d in 0..dst_len {
            let x0 = (d as f64) * scale;
            let x1 = ((d + 1) as f64) * scale;
            let span = (x1 - x0).max(1e-12);
            let start = (x0.floor() as usize).min(src_len - 1);
            let end = (x1.ceil() as usize).clamp(start + 1, src_len);
            let len = end - start;
            if len == 1 {
                out.push(AxisSpan {
                    start,
                    end,
                    w_first: RESIZE_FP_ONE,
                    w_mid: 0,
                    w_last: 0,
                });
            } else if len == 2 {
                let first_ov = (((start + 1) as f64) - x0).clamp(0.0, span);
                let w_first = ((first_ov / span) * (RESIZE_FP_ONE as f64))
                    .round()
                    .clamp(0.0, RESIZE_FP_ONE as f64) as u32;
                out.push(AxisSpan {
                    start,
                    end,
                    w_first,
                    w_mid: 0,
                    w_last: RESIZE_FP_ONE - w_first,
                });
            } else {
                let mid_cnt = (len - 2) as u32;
                let w_mid = ((RESIZE_FP_ONE as f64) / span).floor() as u32;
                let used_mid = w_mid.saturating_mul(mid_cnt).min(RESIZE_FP_ONE);
                let rem = RESIZE_FP_ONE - used_mid;
                let first_ov = (((start + 1) as f64) - x0).clamp(0.0, 1.0);
                let last_ov = (x1 - ((end - 1) as f64)).clamp(0.0, 1.0);
                let edge_sum = (first_ov + last_ov).max(1e-12);
                let w_first = ((rem as f64) * (first_ov / edge_sum))
                    .round()
                    .clamp(0.0, rem as f64) as u32;
                let w_last = rem - w_first;
                out.push(AxisSpan {
                    start,
                    end,
                    w_first,
                    w_mid,
                    w_last,
                });
            }
        }
    } else {
        // Exact center-aligned 2-tap bilinear upsampling with DC-exact fixed-point weights.
        let scale = (src_len as f64) / (dst_len as f64);
        let max_idx = (src_len - 1) as f64;
        for d in 0..dst_len {
            let src_f = (((d as f64) + 0.5) * scale - 0.5).clamp(0.0, max_idx);
            let i0 = src_f.floor() as usize;
            let i1 = (i0 + 1).min(src_len - 1);
            if i0 == i1 {
                out.push(AxisSpan {
                    start: i0,
                    end: i0 + 1,
                    w_first: RESIZE_FP_ONE,
                    w_mid: 0,
                    w_last: 0,
                });
            } else {
                let frac = src_f - (i0 as f64);
                let w_last = (frac * (RESIZE_FP_ONE as f64))
                    .round()
                    .clamp(0.0, RESIZE_FP_ONE as f64) as u32;
                out.push(AxisSpan {
                    start: i0,
                    end: i0 + 2,
                    w_first: RESIZE_FP_ONE - w_last,
                    w_mid: 0,
                    w_last,
                });
            }
        }
    }
}

/// Exact 2x2 integer box-average downsampler for even-dimension RGB buffers (`src_w = 2*dw, src_h = 2*dh`).
fn downsample_2x2_rgb_into(src: &[u8], src_w: usize, src_h: usize, dst: &mut Vec<u8>) {
    let dst_w = src_w / 2;
    let dst_h = src_h / 2;
    dst.resize(dst_w * dst_h * 3, 0u8);
    let src_stride = src_w * 3;
    let dst_stride = dst_w * 3;
    for dy in 0..dst_h {
        let r0 = &src[(2 * dy) * src_stride..(2 * dy + 1) * src_stride];
        let r1 = &src[(2 * dy + 1) * src_stride..(2 * dy + 2) * src_stride];
        let out_row = &mut dst[dy * dst_stride..(dy + 1) * dst_stride];
        for ((a, b), o) in r0
            .as_chunks::<6>()
            .0
            .iter()
            .zip(r1.as_chunks::<6>().0.iter())
            .zip(out_row.as_chunks_mut::<3>().0.iter_mut())
        {
            o[0] = ((a[0] as u32 + a[3] as u32 + b[0] as u32 + b[3] as u32 + 2) >> 2) as u8;
            o[1] = ((a[1] as u32 + a[4] as u32 + b[1] as u32 + b[4] as u32 + 2) >> 2) as u8;
            o[2] = ((a[2] as u32 + a[5] as u32 + b[2] as u32 + b[5] as u32 + 2) >> 2) as u8;
        }
    }
}

/// Pure-Rust `#![forbid(unsafe_code)]`, zero-allocation separable fixed-point RGB resizer
/// (exact fractional `INTER_AREA` downscale + center-aligned bilinear upscale + 2x2 fast path).
fn resize_rgb_fast_into(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    scratch: &mut ResizeScratch,
    dst: &mut Vec<u8>,
) {
    let dst_bytes = dst_w.saturating_mul(dst_h).saturating_mul(3);
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 || src.len() < src_w * src_h * 3 {
        dst.clear();
        return;
    }
    if src_w == dst_w && src_h == dst_h {
        dst.resize(dst_bytes, 0u8);
        dst.copy_from_slice(&src[..dst_bytes]);
        return;
    }
    if src_w == dst_w * 2 && src_h == dst_h * 2 {
        downsample_2x2_rgb_into(src, src_w, src_h, dst);
        return;
    }

    build_axis_spans(src_w, dst_w, &mut scratch.spans_x);
    build_axis_spans(src_h, dst_h, &mut scratch.spans_y);

    // Pass 1 (horizontal): src_w x src_h -> dst_w x src_h
    let h_stride = dst_w * 3;
    let src_stride = src_w * 3;
    let horiz_bytes = h_stride * src_h;
    scratch.horiz_rgb.resize(horiz_bytes, 0u8);
    let spans_x = &scratch.spans_x;

    for sy in 0..src_h {
        let srow = &src[sy * src_stride..(sy + 1) * src_stride];
        let drow = &mut scratch.horiz_rgb[sy * h_stride..(sy + 1) * h_stride];
        for (span, out_px) in spans_x.iter().zip(drow.as_chunks_mut::<3>().0.iter_mut()) {
            let p0 = span.start * 3;
            let len = span.end - span.start;
            if len == 1 {
                out_px[0] = srow[p0];
                out_px[1] = srow[p0 + 1];
                out_px[2] = srow[p0 + 2];
            } else if len == 2 {
                let p1 = p0 + 3;
                let wf = span.w_first;
                let wl = span.w_last;
                out_px[0] = (((srow[p0] as u32) * wf + (srow[p1] as u32) * wl + RESIZE_FP_HALF)
                    >> RESIZE_FP_SHIFT) as u8;
                out_px[1] =
                    (((srow[p0 + 1] as u32) * wf + (srow[p1 + 1] as u32) * wl + RESIZE_FP_HALF)
                        >> RESIZE_FP_SHIFT) as u8;
                out_px[2] =
                    (((srow[p0 + 2] as u32) * wf + (srow[p1 + 2] as u32) * wl + RESIZE_FP_HALF)
                        >> RESIZE_FP_SHIFT) as u8;
            } else {
                let wf = span.w_first;
                let wm = span.w_mid;
                let wl = span.w_last;
                let p_last = (span.end - 1) * 3;
                let mut mr = 0u32;
                let mut mg = 0u32;
                let mut mb = 0u32;
                for px in srow[(span.start + 1) * 3..p_last].as_chunks::<3>().0 {
                    mr += px[0] as u32;
                    mg += px[1] as u32;
                    mb += px[2] as u32;
                }
                let r = (srow[p0] as u32) * wf + mr * wm + (srow[p_last] as u32) * wl;
                let g = (srow[p0 + 1] as u32) * wf + mg * wm + (srow[p_last + 1] as u32) * wl;
                let b = (srow[p0 + 2] as u32) * wf + mb * wm + (srow[p_last + 2] as u32) * wl;
                out_px[0] = ((r + RESIZE_FP_HALF) >> RESIZE_FP_SHIFT) as u8;
                out_px[1] = ((g + RESIZE_FP_HALF) >> RESIZE_FP_SHIFT) as u8;
                out_px[2] = ((b + RESIZE_FP_HALF) >> RESIZE_FP_SHIFT) as u8;
            }
        }
    }

    // Pass 2 (vertical, row-major linear scan): dst_w x src_h -> dst_w x dst_h
    dst.resize(dst_bytes, 0u8);
    scratch.v_acc.resize(h_stride, 0u32);
    scratch.v_mid.resize(h_stride, 0u32);
    let spans_y = &scratch.spans_y;
    let horiz = &scratch.horiz_rgb;

    for (dy, span) in spans_y.iter().enumerate() {
        let out_row = &mut dst[dy * h_stride..(dy + 1) * h_stride];
        let len = span.end - span.start;
        let r0 = &horiz[span.start * h_stride..(span.start + 1) * h_stride];
        if len == 1 {
            out_row.copy_from_slice(r0);
        } else if len == 2 {
            let r1 = &horiz[(span.start + 1) * h_stride..(span.start + 2) * h_stride];
            let wf = span.w_first;
            let wl = span.w_last;
            for ((dst_b, &a), &b) in out_row.iter_mut().zip(r0.iter()).zip(r1.iter()) {
                *dst_b = (((a as u32) * wf + (b as u32) * wl + RESIZE_FP_HALF) >> RESIZE_FP_SHIFT)
                    as u8;
            }
        } else {
            let wf = span.w_first;
            let wm = span.w_mid;
            let wl = span.w_last;
            let r_last = &horiz[(span.end - 1) * h_stride..span.end * h_stride];
            for (acc, &a) in scratch.v_acc.iter_mut().zip(r0.iter()) {
                *acc = (a as u32) * wf;
            }
            scratch.v_mid.fill(0);
            for sy in (span.start + 1)..(span.end - 1) {
                let r_mid = &horiz[sy * h_stride..(sy + 1) * h_stride];
                for (m, &v) in scratch.v_mid.iter_mut().zip(r_mid.iter()) {
                    *m += v as u32;
                }
            }
            for (((dst_b, &acc), &mid), &b) in out_row
                .iter_mut()
                .zip(scratch.v_acc.iter())
                .zip(scratch.v_mid.iter())
                .zip(r_last.iter())
            {
                let sum = acc + mid * wm + (b as u32) * wl;
                *dst_b = ((sum + RESIZE_FP_HALF) >> RESIZE_FP_SHIFT) as u8;
            }
        }
    }
}

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
        let n = 160 * 128;
        Self {
            rad: Vec::with_capacity(n),
            horiz: Vec::with_capacity(n),
            col_win: Vec::with_capacity(160),
            hp: Vec::with_capacity(n),
            out: Vec::with_capacity(2 * n),
            lwir_u8: Vec::with_capacity(n),
        }
    }
}

struct D2EvalState {
    in_h: usize,
    in_w: usize,
    hid_c: usize,
    hid_h: usize,
    hid_w: usize,
    h_prev: Vec<f32>,
    norm_gray_hist: VecDeque<Vec<f32>>,
    scratch_gray: Vec<f32>,
    scratch_x: Vec<f32>,
    last_size_px: Option<f64>,
    track_age: u32,
    bearing_tbc: BearingTrackBeforeConfirm,
    prev_tbc_confirmed: bool,
    coord_active_range: CoordinatedActiveRange,
}

impl D2EvalState {
    fn new(in_h: usize, in_w: usize, hid_c: usize, hid_h: usize, hid_w: usize) -> Self {
        let n_px = in_h * in_w;
        Self {
            in_h,
            in_w,
            hid_c,
            hid_h,
            hid_w,
            h_prev: vec![0.0_f32; hid_c * hid_h * hid_w],
            norm_gray_hist: VecDeque::with_capacity(8),
            scratch_gray: vec![0.0_f32; n_px],
            scratch_x: vec![0.0_f32; 6 * n_px],
            last_size_px: None,
            track_age: 0,
            bearing_tbc: BearingTrackBeforeConfirm::new_3_of_5(),
            prev_tbc_confirmed: false,
            coord_active_range: CoordinatedActiveRange::new(
                1,
                ActiveRangeConfig {
                    sigma_bearing_rad: 2.86 / AOT_FOCAL_PX,
                    ..ActiveRangeConfig::default()
                },
            ),
        }
    }
}

// ---- Strict Safe Image Loading & Frame Directory Discovery ----

fn is_supported_image_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
        .unwrap_or(false)
}

pub fn discover_sorted_image_frames(
    dir: &Path,
    max_frames: Option<usize>,
) -> Result<Vec<PathBuf>, EvalFramesFailure> {
    if let Some(0) = max_frames {
        return Err(EvalFramesFailure::new(
            "INVALID_MAX_FRAMES",
            "--max-frames must be a positive integer >= 1",
        ));
    }
    if !dir.exists() {
        return Err(EvalFramesFailure::new(
            "DIR_NOT_FOUND",
            "Input frame directory does not exist",
        ));
    }
    if !dir.is_dir() {
        return Err(EvalFramesFailure::new(
            "NOT_A_DIRECTORY",
            "Input path for --eval-frames must be a directory",
        ));
    }

    let entries = std::fs::read_dir(dir).map_err(|e| {
        EvalFramesFailure::new(
            "DIR_UNREADABLE",
            format!("Failed to read input frame directory: {e}"),
        )
    })?;

    let mut files: Vec<(String, String, PathBuf)> = Vec::new();
    for entry_res in entries {
        let entry = match entry_res {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.is_file() && is_supported_image_extension(&path) {
            let exact = entry.file_name().to_string_lossy().to_string();
            let lower = exact.to_ascii_lowercase();
            files.push((lower, exact, path));
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut paths: Vec<PathBuf> = files.into_iter().map(|(_, _, p)| p).collect();
    if paths.is_empty() {
        return Err(EvalFramesFailure::new(
            "EMPTY_FRAME_DIR",
            "Input directory contains 0 supported image files (.png, .jpg, .jpeg)",
        ));
    }
    if let Some(limit) = max_frames {
        paths.truncate(limit);
    }
    Ok(paths)
}

/// Strictly bounded PNG/JPEG loader that defeats >16 MiB payloads, >4096x4096 decompression bombs,
/// mismatched formats, empty files, and corrupt byte streams without panicking.
pub fn load_rgb_image_safe(path: &Path) -> Result<RgbImage, String> {
    let mut raw_buf = Vec::new();
    load_rgb_image_safe_with_buf(path, &mut raw_buf)
}

fn load_rgb_image_safe_with_buf(path: &Path, raw: &mut Vec<u8>) -> Result<RgbImage, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("Failed to stat image file: {e}"))?;
    if !meta.is_file() {
        return Err("Image path is not a regular file".to_string());
    }
    let file_len = meta.len();
    if file_len == 0 {
        return Err("Empty image file (0 bytes)".to_string());
    }
    if file_len > MAX_EVAL_IMAGE_BYTES as u64 {
        return Err(format!(
            "Image file exceeds 16 MiB safety limit ({file_len} bytes > {MAX_EVAL_IMAGE_BYTES})"
        ));
    }

    raw.clear();
    let expected = file_len as usize;
    if raw.capacity() < expected {
        raw.reserve(expected - raw.len());
    }
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Failed to open image file: {e}"))?;
    file.by_ref()
        .take((MAX_EVAL_IMAGE_BYTES as u64) + 1)
        .read_to_end(raw)
        .map_err(|e| format!("Failed to read image bytes: {e}"))?;
    if raw.is_empty() {
        return Err("Empty image payload (0 bytes)".to_string());
    }
    if raw.len() > MAX_EVAL_IMAGE_BYTES {
        return Err(format!(
            "Image payload exceeds 16 MiB safety limit ({} bytes > {MAX_EVAL_IMAGE_BYTES})",
            raw.len()
        ));
    }

    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_EVAL_IMAGE_DIM);
    limits.max_image_height = Some(MAX_EVAL_IMAGE_DIM);
    limits.max_alloc = Some(MAX_EVAL_DECODE_ALLOC_BYTES);

    let mut dim_reader = ImageReader::new(Cursor::new(raw.as_slice()))
        .with_guessed_format()
        .map_err(|e| format!("Failed to inspect image header: {e}"))?;
    let format = match dim_reader.format() {
        Some(f @ (ImageFormat::Png | ImageFormat::Jpeg)) => f,
        other => {
            return Err(format!(
                "Unsupported or corrupt image signature (detected={other:?}; expected PNG or JPEG)"
            ));
        }
    };
    dim_reader.limits(limits.clone());
    let (w, h) = dim_reader
        .into_dimensions()
        .map_err(|e| format!("Failed to read PNG/JPEG header dimensions: {e}"))?;
    if w == 0 || h == 0 || w > MAX_EVAL_IMAGE_DIM || h > MAX_EVAL_IMAGE_DIM {
        return Err(format!(
            "Image dimensions {w}x{h} outside allowed [1..={MAX_EVAL_IMAGE_DIM}]^2 decompression-bomb limit"
        ));
    }

    let mut reader = ImageReader::with_format(Cursor::new(raw.as_slice()), format);
    reader.limits(limits);
    let dyn_img = reader
        .decode()
        .map_err(|e| format!("Corrupt or truncated PNG/JPEG image data: {e}"))?;
    // Zero-copy move when decoder already produced 8-bit RGB (avoids full-frame clone!)
    let rgb = dyn_img.into_rgb8();
    if rgb.width() != w || rgb.height() != h || rgb.as_raw().is_empty() {
        return Err("Decoded RGB image dimensions mismatch header".to_string());
    }
    Ok(rgb)
}

// ---- Exact Studio Parity Preprocessing & Signal Helpers ----

fn extract_native_luma_u8_window_into(
    img: &RgbImage,
    max_w: usize,
    max_h: usize,
    out: &mut Vec<u8>,
) -> (usize, usize) {
    let iw = img.width() as usize;
    let ih = img.height() as usize;
    out.clear();
    if iw == 0 || ih == 0 {
        return (0, 0);
    }
    let w = iw.min(max_w);
    let h = ih.min(max_h);
    let x0 = (iw - w) / 2;
    let y0 = (ih - h) / 2;
    let raw = img.as_raw();
    let stride = iw * 3;
    out.resize(w * h, 0u8);
    // Exact 16-bit fixed-point BT.601 luma: 19595 + 38470 + 7471 = 65536 (0.299, 0.587, 0.114).
    for row_idx in 0..h {
        let y = y0 + row_idx;
        let row_bytes = &raw[y * stride + x0 * 3..y * stride + (x0 + w) * 3];
        let dst_row = &mut out[row_idx * w..(row_idx + 1) * w];
        for (dst, rgb) in dst_row.iter_mut().zip(row_bytes.as_chunks::<3>().0.iter()) {
            let y_val = (19_595u32 * (rgb[0] as u32)
                + 38_470u32 * (rgb[1] as u32)
                + 7_471u32 * (rgb[2] as u32)
                + 32_768u32)
                >> 16;
            *dst = y_val as u8;
        }
    }
    (w, h)
}

fn compute_native_rgb_fpn_noise_with_buf(
    img: &RgbImage,
    nat_buf: &mut Vec<u8>,
) -> Option<(f64, f64, f64, f64)> {
    let (nw, nh) = extract_native_luma_u8_window_into(img, 640, 480, nat_buf);
    if nw >= 6 && nh >= 6 {
        let (dlf, fpn) = detect_dead_line_fpn_u8(nat_buf, nw, nh);
        let (ac, noise) = detect_white_noise_fault_u8(nat_buf, nw, nh);
        Some((dlf, fpn, ac, noise))
    } else {
        None
    }
}

fn observe_eo_rgb_with_fpn_from_160x128(
    shm: &mut SensorHealthMonitor,
    rgb_160x128: &[u8],
    native_fpn: Option<(f64, f64, f64, f64)>,
    luma_buf: &mut Vec<f64>,
) -> EoOpticalQuality {
    let (w, h) = (160_usize, 128_usize);
    let n = w * h;
    luma_buf.resize(n, 0.0_f64);
    for (dst, rgb) in luma_buf
        .iter_mut()
        .zip(rgb_160x128.as_chunks::<3>().0.iter().take(n))
    {
        *dst = 0.299 * (rgb[0] as f64) + 0.587 * (rgb[1] as f64) + 0.114 * (rgb[2] as f64);
    }
    let q = SensorHealthMonitor::evaluate_eo_luma_plane_with_fpn_noise(luma_buf, w, h, native_fpn);
    shm.record_eo_quality(q);
    q
}

fn build_lwir_2ch_from_rgb_slice_into(
    rgb_slice: &[u8],
    w: usize,
    h: usize,
    compute_2ch_tensor: bool,
    scratch: &mut LwirScratch,
) {
    let n = w * h;
    scratch.lwir_u8.resize(n, 0_u8);
    let inv255 = 1.0_f32 / 255.0_f32;

    if !compute_2ch_tensor {
        for (dst, rgb) in scratch
            .lwir_u8
            .iter_mut()
            .zip(rgb_slice.as_chunks::<3>().0.iter().take(n))
        {
            let r = (rgb[0] as f32) * inv255;
            let g = (rgb[1] as f32) * inv255;
            let b = (rgb[2] as f32) * inv255;
            let gray = (0.55 * r + 0.30 * g + 0.15 * (1.0 - b)).clamp(0.0, 1.0);
            *dst = (gray * 255.0).clamp(0.0, 255.0) as u8;
        }
        return;
    }

    scratch.rad.resize(n, 0.0_f32);
    scratch.horiz.resize(n, 0.0_f32);
    scratch.col_win.resize(w, 0.0_f32);
    scratch.hp.resize(n, 0.0_f32);
    scratch.out.resize(2 * n, 0.0_f32);

    let denom_log = (1.0_f32 + LWIR_L_SAT_PROXY.max(1.0)).ln();
    let inv_denom_log = 1.0_f32 / denom_log;
    for (idx, rgb) in rgb_slice.as_chunks::<3>().0.iter().enumerate().take(n) {
        let r = (rgb[0] as f32) * inv255;
        let g = (rgb[1] as f32) * inv255;
        let b = (rgb[2] as f32) * inv255;
        let gray = (0.55 * r + 0.30 * g + 0.15 * (1.0 - b)).clamp(0.0, 1.0);
        let rv = gray * LWIR_L_SAT_PROXY;
        scratch.rad[idx] = rv;
        scratch.lwir_u8[idx] = (gray * 255.0).clamp(0.0, 255.0) as u8;
        scratch.out[idx] = (1.0_f32 + rv.max(0.0)).ln() * inv_denom_log;
    }

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

fn rgb_to_scene_norm_gray_into(
    rgb: &[u8],
    n_px: usize,
    gray_scratch: &mut Vec<f32>,
    out: &mut Vec<f32>,
) {
    gray_scratch.resize(n_px, 0.0_f32);
    let mut sum = 0.0_f64;
    let mut sq = 0.0_f64;
    for (dst, px) in gray_scratch
        .iter_mut()
        .zip(rgb.as_chunks::<3>().0.iter().take(n_px))
    {
        let y = 0.299_f32 * (px[0] as f32)
            + 0.587_f32 * (px[1] as f32)
            + 0.114_f32 * (px[2] as f32);
        *dst = y;
        let yf = y as f64;
        sum += yf;
        sq += yf * yf;
    }
    let nf = n_px.max(1) as f64;
    let mean = sum / nf;
    let var = ((sq / nf) - mean * mean).max(0.0);
    let std = var.sqrt().max(8.0);
    let mean_f32 = mean as f32;
    let inv_std_f32 = (1.0 / std) as f32;
    out.resize(n_px, 0.0_f32);
    for (dst, &v) in out.iter_mut().zip(gray_scratch.iter()) {
        *dst = ((v - mean_f32) * inv_std_f32).clamp(-5.0, 5.0);
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    1.0 / (1.0 + (-x.clamp(-40.0, 40.0)).exp())
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
    let score_logit_thr = if score_thr <= 0.0 {
        f32::NEG_INFINITY
    } else if score_thr >= 1.0 {
        f32::INFINITY
    } else {
        (score_thr / (1.0 - score_thr)).ln() - 1e-4
    };
    let mut max_logit = f32::NEG_INFINITY;
    let mut best_logit = f32::NEG_INFINITY;
    let mut best_xy: Option<(usize, usize)> = None;

    for y in 0..hm_h {
        let row = y * hm_w;
        for x in 0..hm_w {
            let idx = row + x;
            let v = hm[idx];
            if !v.is_finite() {
                continue;
            }
            if v > max_logit {
                max_logit = v;
            }
            if v >= score_logit_thr && v > best_logit {
                let pm = peak_mask[idx];
                if pm.is_finite() && pm > 0.5 && sigmoid_f32(v) >= score_thr {
                    best_logit = v;
                    best_xy = Some((x, y));
                }
            }
        }
    }

    let max_prob = if max_logit.is_finite() {
        sigmoid_f32(max_logit)
    } else {
        0.0
    };
    let best_score = if best_logit.is_finite() {
        sigmoid_f32(best_logit)
    } else {
        -1.0
    };

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

fn tbc_status_label(st: BearingTrackStatus) -> &'static str {
    match st {
        BearingTrackStatus::Idle => "IDLE",
        BearingTrackStatus::Tentative => "TENTATIVE",
        BearingTrackStatus::Confirmed => "CONFIRMED",
        BearingTrackStatus::Dropped => "DROPPED",
    }
}

fn active_range_mode_label(mode: u8) -> &'static str {
    match mode {
        0 => "NOMINAL_HOLD",
        1 => "ACTIVE_SWERVE",
        2 => "LOCKED",
        _ => "IMMEDIATE_EVADE",
    }
}

fn load_required_pinned_session(
    env: &ort::environment::Environment,
    model_name: &'static str,
    model_dir: &Path,
    explicit_model_dir: bool,
    direct_names: &[&str],
    repo_rel_candidates: &[&str],
    pins: &[(&str, &str)],
) -> Result<(Session, LoadedModelInfo), EvalFramesFailure> {
    let path = find_model_path(model_dir, explicit_model_dir, direct_names, repo_rel_candidates)
        .ok_or_else(|| {
            EvalFramesFailure::new(
                "MODEL_NOT_FOUND",
                format!(
                    "Required ONNX model '{model_name}' not found (searched OPENXVS_MODEL_DIR / --model-dir / exe_dir / cwd / /opt/openxvs/models)"
                ),
            )
        })?;
    let (filename, sha256) = verify_pinned_onnx_artifact(&path, pins).map_err(|(fname, _)| {
        EvalFramesFailure::new(
            "SHA256_MISMATCH_REJECTED",
            format!(
                "Fail-closed SHA-256 verification rejected model '{model_name}' (file={})",
                fname.unwrap_or_else(|| "unknown".to_string())
            ),
        )
    })?;
    let t_load = Instant::now();
    let sess = build_optimized_ort_session(env, &path, 1).map_err(|e| {
        EvalFramesFailure::new(
            "ORT_SESSION_ERROR",
            format!("Failed to create ONNX Runtime session for '{model_name}': {e}"),
        )
    })?;
    let load_ms = round2(t_load.elapsed().as_secs_f64() * 1000.0);
    Ok((
        sess,
        LoadedModelInfo {
            name: model_name,
            filename,
            sha256,
            load_ms,
        },
    ))
}

fn append_shadow_tape_record(
    tape_lines: &mut Vec<String>,
    prev_hash: &mut String,
    seq: u64,
    payload: serde_json::Value,
) -> Result<(), EvalFramesFailure> {
    let mut body = serde_json::json!({
        "schema": SHADOW_MODE_LOG_SCHEMA,
        "seq": seq,
        "timestamp_mono_us": (seq + 1) * 100_000,
        "trial_phase": "offline_eval_frames_shadow_non_flight",
        "hardware_tx_pin_removed": true,
        "event_type": "eval_frame",
        "direction": "internal",
        "tx_inhibited": true,
        "actuator_authority": ACTUATOR_AUTHORITY,
        "outbound_mavlink_frames_emitted": 0u64,
        "outbound_mavlink_frames": 0u64,
        "non_flight_sim_only": NON_FLIGHT_SIM_ONLY,
        "watermark": NON_FLIGHT_WATERMARK,
        "payload": payload,
        "prev_hash": prev_hash.as_str(),
    });
    let canon_body = canonical_json_python_compat(&body).map_err(|e| {
        EvalFramesFailure::new(
            "TAPE_SERIALIZATION_ERROR",
            format!("Failed to canonicalize tape record seq={seq}: {e}"),
        )
    })?;
    let digest = Sha256::digest(canon_body.as_bytes());
    let mut entry_hash = String::with_capacity(64);
    for b in digest {
        entry_hash.push_str(&format!("{b:02x}"));
    }
    if let Some(map) = body.as_object_mut() {
        map.insert(
            "entry_hash".to_string(),
            serde_json::Value::String(entry_hash.clone()),
        );
    }
    let canon_line = canonical_json_python_compat(&body).map_err(|e| {
        EvalFramesFailure::new(
            "TAPE_SERIALIZATION_ERROR",
            format!("Failed to serialize finalized tape record seq={seq}: {e}"),
        )
    })?;
    tape_lines.push(canon_line);
    *prev_hash = entry_hash;
    Ok(())
}

struct VerdictStats<'a> {
    frames_processed: usize,
    frames_corrupt: usize,
    models_run: &'a [String],
    p50_ms: f64,
    p95_ms: f64,
    tbc_confirmed_tracks: usize,
    fail_closed_rejections: usize,
    sensor_faults_detected: usize,
}

fn build_verdict_plain_english(s: &VerdictStats<'_>) -> String {
    let models_label = if s.models_run.is_empty() {
        "none".to_string()
    } else {
        s.models_run.join(", ")
    };
    if s.frames_processed == 0 && s.frames_corrupt > 0 {
        return format!(
            "FAIL-CLOSED: All {} input image(s) were corrupt or exceeded safety limits (0 valid frames evaluated, {} fail-closed rejection(s)). Zero panics, simulation and offline evaluation only — zero flight actuation.",
            s.frames_corrupt, s.fail_closed_rejections
        );
    }
    if s.frames_corrupt > 0 || s.fail_closed_rejections > 0 || s.sensor_faults_detected > 0 {
        format!(
            "PASS (FAIL-CLOSED GUARD ACTIVE): Evaluated {} valid frame(s) across {models_label} (median {:.2} ms, p95 {:.2} ms); rejected {} corrupt/oversized image(s), flagged {} sensor fault(s) and {} fail-closed rejection(s), confirmed {} airborne track(s). Offline evaluation only — zero flight actuation.",
            s.frames_processed,
            s.p50_ms,
            s.p95_ms,
            s.frames_corrupt,
            s.sensor_faults_detected,
            s.fail_closed_rejections,
            s.tbc_confirmed_tracks
        )
    } else {
        format!(
            "PASS: Evaluated {} frame(s) cleanly across {models_label} (median {:.2} ms, p95 {:.2} ms, 0 corrupt images, 0 sensor faults, 0 fail-closed rejections, {} confirmed airborne track(s)). Offline evaluation only — zero flight actuation.",
            s.frames_processed, s.p50_ms, s.p95_ms, s.tbc_confirmed_tracks
        )
    }
}

fn ensure_parent_dir(path: &Path) -> Result<(), EvalFramesFailure> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| {
                EvalFramesFailure::new(
                    "OUTPUT_DIR_ERROR",
                    format!("Failed to create output parent directory: {e}"),
                )
            })?;
        }
    }
    Ok(())
}

/// Execute Pure-Rust offline frame evaluation (`--eval-frames`).
pub fn run_eval_frames(cfg: &EvalFramesConfig) -> Result<EvalFramesReport, EvalFramesFailure> {
    // 1. Discover and sort valid frame paths first (fail cleanly on empty/missing dir)
    let frame_paths = discover_sorted_image_frames(&cfg.input_dir, cfg.max_frames)?;

    // 2. Acquire ORT runtime and fail-closed load ALL required pinned ONNX models for --mode
    let (model_dir, explicit_model_dir) = match cfg.model_dir.as_ref() {
        Some(p) => (p.clone(), true),
        None => resolve_default_model_dir(),
    };
    let env = acquire_edge_ort_env_with_hint(Some(&model_dir)).map_err(|e| {
        EvalFramesFailure::new(
            "ORT_DYLIB_UNAVAILABLE",
            format!("ONNX Runtime shared library could not be loaded: {e}"),
        )
    })?;

    let mut models_loaded: Vec<LoadedModelInfo> = Vec::new();
    let mut models_run: Vec<String> = Vec::new();

    let mut d2_sess: Option<Session> = None;
    let mut d2_state: Option<D2EvalState> = None;
    if cfg.mode.needs_eo() {
        let (mut sess, info) = load_required_pinned_session(
            &env,
            "xvs_d2",
            &model_dir,
            explicit_model_dir,
            &["xvs_d2_web_fp16.onnx"],
            &[
                "apps/livedeck/public/models/xvs_d2_web_fp16.onnx",
                "models/xvs_d2_web_fp16.onnx",
            ],
            D2_ONNX_PINS,
        )?;
        let (in_h, in_w, hid_c, hid_h, hid_w) = resolve_d2_input_shapes(&sess);
        let st = D2EvalState::new(in_h, in_w, hid_c, hid_h, hid_w);
        // 2 warm-up passes on preallocated zero buffers to stabilize ORT arena/kernels before frame 0
        for _ in 0..2 {
            let tx = TensorRef::from_array_view(([1usize, 6, in_h, in_w], st.scratch_x.as_slice()))
                .map_err(|e| {
                    EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("D2 warm tx: {e}"))
                })?;
            let th = TensorRef::from_array_view((
                [1usize, hid_c, hid_h, hid_w],
                st.h_prev.as_slice(),
            ))
            .map_err(|e| EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("D2 warm th: {e}")))?;
            let _ = sess
                .run(ort::inputs!["x_t" => tx, "h_prev_warped" => th])
                .map_err(|e| {
                    EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("D2 warm run: {e}"))
                })?;
        }
        d2_state = Some(st);
        d2_sess = Some(sess);
        models_loaded.push(info);
        models_run.push("xvs_d2".to_string());
    }

    let mut lwir_scratch = LwirScratch::new();
    let mut thermal_sess: Option<Session> = None;
    let mut scratch_thermal_eo: Vec<f32> = Vec::new();
    if cfg.mode.needs_thermal() {
        let (mut sess, info) = load_required_pinned_session(
            &env,
            "xvs_thermal",
            &model_dir,
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
            THERMAL_ONNX_PINS,
        )?;
        scratch_thermal_eo = vec![0.0_f32; 3 * 256 * 320];
        lwir_scratch.out.resize(2 * 128 * 160, 0.0_f32);
        let q0 = [1.0_f32];
        for _ in 0..2 {
            let teo = TensorRef::from_array_view((
                [1usize, 3, 256, 320],
                scratch_thermal_eo.as_slice(),
            ))
            .map_err(|e| {
                EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("Thermal warm eo: {e}"))
            })?;
            let tlw = TensorRef::from_array_view((
                [1usize, 2, 128, 160],
                lwir_scratch.out.as_slice(),
            ))
            .map_err(|e| {
                EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("Thermal warm lw: {e}"))
            })?;
            let tqe = TensorRef::from_array_view(([1usize], q0.as_slice())).map_err(|e| {
                EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("Thermal warm q_eo: {e}"))
            })?;
            let tql = TensorRef::from_array_view(([1usize], q0.as_slice())).map_err(|e| {
                EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("Thermal warm q_lw: {e}"))
            })?;
            let _ = sess
                .run(ort::inputs!["x_eo" => teo, "x_lwir" => tlw, "q_eo" => tqe, "q_lwir" => tql])
                .map_err(|e| {
                    EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("Thermal warm run: {e}"))
                })?;
        }
        thermal_sess = Some(sess);
        models_loaded.push(info);
        models_run.push("xvs_thermal".to_string());
    }

    let mut land_sess: Option<(Session, usize, usize, StudentTCalibrator)> = None;
    let mut scratch_land_inp: Vec<f32> = Vec::new();
    if cfg.mode.needs_land() {
        let (mut sess, info) = load_required_pinned_session(
            &env,
            "xvs_land",
            &model_dir,
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
            LAND_ONNX_PINS,
        )?;
        let (roi_h, roi_w) = resolve_land_input_hw(&sess);
        let is_v52 = info.sha256.eq_ignore_ascii_case(EXPECTED_LAND_V52_FP16_SHA256)
            || info.sha256.eq_ignore_ascii_case(EXPECTED_LAND_V52_FP32_SHA256);
        let (nu, scale) = if is_v52 {
            (V52_LAND_STUDENT_T_NU, V52_LAND_STUDENT_T_SCALE)
        } else {
            (LAND_STUDENT_T_NU, LAND_STUDENT_T_SCALE)
        };
        let cal = StudentTCalibrator::new(nu, scale).map_err(|e| {
            EvalFramesFailure::new("CALIBRATOR_ERROR", format!("Land calibrator error: {e}"))
        })?;
        scratch_land_inp = vec![0.5_f32; 3 * roi_h * roi_w];
        for _ in 0..2 {
            let troi =
                TensorRef::from_array_view(([1usize, 3, roi_h, roi_w], scratch_land_inp.as_slice()))
                    .map_err(|e| {
                        EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("Land warm tensor: {e}"))
                    })?;
            let _ = sess.run(ort::inputs!["roi_crop" => troi]).map_err(|e| {
                EvalFramesFailure::new("ORT_WARMUP_ERROR", format!("Land warm run: {e}"))
            })?;
        }
        land_sess = Some((sess, roi_h, roi_w, cal));
        models_loaded.push(info);
        models_run.push("xvs_land".to_string());
    }

    // 3. Per-frame stateful evaluation pipeline (reusing preallocated scratch across frames)
    let mut shm = SensorHealthMonitor::new();
    let mut frame_scratch = EvalFrameScratch::new();
    let mut rta = SimplexRta::new();
    let mut last_rta_mode = SimplexMode::NominalMl;
    let mut last_rta_inhibit_turns = false;

    let mut max_target_w: usize = 160;
    let mut max_target_h: usize = 128;
    if let Some(ref st) = d2_state {
        max_target_w = max_target_w.max(st.in_w);
        max_target_h = max_target_h.max(st.in_h);
    }
    if thermal_sess.is_some() {
        max_target_w = max_target_w.max(320);
        max_target_h = max_target_h.max(256);
    }
    if let Some((_, roi_h, roi_w, _)) = land_sess.as_ref() {
        max_target_w = max_target_w.max(*roi_w);
        max_target_h = max_target_h.max(*roi_h);
    }

    let mut frames_processed: usize = 0;
    let mut frames_corrupt: usize = 0;
    let mut detections_total: usize = 0;
    let mut tbc_confirmed_tracks: usize = 0;
    let mut tbc_confirmed_frames: usize = 0;
    let mut fail_closed_rejections: usize = 0;
    let mut sensor_faults_detected: usize = 0;
    let mut dual_blind_frames: usize = 0;

    let mut latencies_ms: Vec<f64> = Vec::with_capacity(frame_paths.len());
    let mut frame_records: Vec<EvalFrameRecord> = Vec::with_capacity(frame_paths.len());

    let record_tape = cfg.tape_path.is_some();
    let mut tape_lines: Vec<String> = if record_tape {
        Vec::with_capacity(frame_paths.len())
    } else {
        Vec::new()
    };
    let mut tape_prev_hash = SHADOW_GENESIS_HASH.to_string();

    let own = OwnshipStateNed {
        v_ned: [35.0, 0.0, 0.0],
        a_ned: [0.0, 0.0, 0.0],
        heading_rad: 0.0,
        tas_mps: 35.0,
        v_stall_mps: 14.5,
    };

    for (frame_index, frame_path) in frame_paths.iter().enumerate() {
        let filename = frame_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("frame_{frame_index:04}"));
        let t_frame = Instant::now();

        let rgb = match load_rgb_image_safe_with_buf(frame_path, &mut frame_scratch.raw_file_buf) {
            Ok(img) => img,
            Err(err_msg) => {
                frames_corrupt += 1;
                fail_closed_rejections += 1;
                if let Some(st) = d2_state.as_mut() {
                    let snap = st.bearing_tbc.observe_with_health(None, 0.0, true);
                    st.prev_tbc_confirmed = snap.confirmed;
                }
                let fault_tel = RtaFloatTelemetry {
                    horizon_pl_rad: 0.002,
                    horizon_al_rad: 0.017_453_3,
                    v_vpl_m: 5.0,
                    v_val_m: 10.0,
                    v_hpl_m: 14.0,
                    v_hal_m: 30.0,
                    epipolar_max_rad: [0.0002; 6],
                    signed_horizon_residual_rad: 0.0,
                    runway_width_delta_m: 0.0,
                    runway_width_delta_limit_m: 5.0,
                    cusum_stat: 0.0,
                    cusum_thr: CUSUM_H_LOCKED,
                    ood_global_score: 1.0,
                    ood_global_thr: 0.7,
                    ood_cam_scores: [1.0; 6],
                    ood_cam_thr: 0.7,
                    cam_health_mask: 0x00,
                    ml_detector_fault: true,
                    transient_dropout: false,
                    approach_active: false,
                    go_around_ack_reset: false,
                };
                let rta_out = rta.step(&build_rta_input_from_f64(&fault_tel));
                last_rta_mode = rta_out.mode;
                last_rta_inhibit_turns = rta_out.inhibit_avoidance_turns;
                let latency_ms = round2(t_frame.elapsed().as_secs_f64() * 1000.0);

                if record_tape {
                    let seq = frame_index as u64;
                    let payload = serde_json::json!({
                        "frame_index": frame_index,
                        "filename": filename,
                        "eval_mode": cfg.mode.as_str(),
                        "corrupt": true,
                        "fail_closed_rejected": true,
                        "latency_ms": latency_ms,
                        "down_m": -120.0,
                        "gnss_ne_m": [round4((seq as f64) * 3.5), 0.0],
                        "vis_ne_m": [round4((seq as f64) * 3.5), 0.0],
                        "gnss_down_m": -120.0,
                        "vis_gate_admitted": false,
                        "cam_health_mask": 0u8,
                        "ml_detector_fault": true,
                        "approach_active": false,
                        "rta_mode": format!("{:?}", rta_out.mode),
                        "rta_mode_id": rta_out.mode as u8,
                        "q_eo_effective": 0.0,
                        "q_lwir_effective": 0.0,
                        "tbc_confirmed": false,
                        "d2_detected": false,
                        "thermal_mode": "DUAL_DVE_SVS_FALLBACK",
                        "land_runway_admitted": false,
                    });
                    append_shadow_tape_record(&mut tape_lines, &mut tape_prev_hash, seq, payload)?;
                }

                frame_records.push(EvalFrameRecord {
                    frame_index,
                    filename,
                    status: "CORRUPT_REJECTED_FAIL_CLOSED",
                    corrupt: true,
                    fail_closed_rejected: true,
                    fail_closed_reasons: vec!["CORRUPT_OR_OVERSIZED_IMAGE".to_string()],
                    width: None,
                    height: None,
                    latency_ms,
                    error: Some(err_msg),
                    sensor_health: None,
                    eo_d2: None,
                    thermal: None,
                    land: None,
                    rta_mode: format!("{:?}", rta_out.mode),
                    rta_mode_id: rta_out.mode as u8,
                    rta_raw_target: format!("{:?}", rta_out.raw_target),
                    rta_active_healthy_cams: rta_out.active_healthy_cams,
                    rta_inhibit_avoidance_turns: rta_out.inhibit_avoidance_turns,
                    rta_mandatory_go_around: rta_out.mandatory_go_around,
                });
                continue;
            }
        };

        frames_processed += 1;
        let img_w = rgb.width();
        let img_h = rgb.height();
        let mut fail_closed_reasons: Vec<String> = Vec::new();

        // 1. Native-resolution Wave-1/2 FPN & white-noise scan ALWAYS runs on full-res un-decimated &rgb!
        let native_fpn =
            compute_native_rgb_fpn_noise_with_buf(&rgb, &mut frame_scratch.native_luma);

        // 2. Single-pass 2x2 pre-pyramid when high-res frame is >= 2x the largest required model target
        //    (e.g., 1920x1080 -> 960x540 once, cutting pixel work 4x across all downstream resizes).
        let (work_w, work_h, work_sel) = {
            let mut cur_w = img_w as usize;
            let mut cur_h = img_h as usize;
            let mut sel = 0u8;
            while cur_w >= max_target_w.saturating_mul(2)
                && cur_h >= max_target_h.saturating_mul(2)
                && cur_w.is_multiple_of(2)
                && cur_h.is_multiple_of(2)
            {
                match sel {
                    0 => {
                        downsample_2x2_rgb_into(
                            rgb.as_raw(),
                            cur_w,
                            cur_h,
                            &mut frame_scratch.pyramid_a,
                        );
                        sel = 1;
                    }
                    1 => {
                        downsample_2x2_rgb_into(
                            &frame_scratch.pyramid_a,
                            cur_w,
                            cur_h,
                            &mut frame_scratch.pyramid_b,
                        );
                        sel = 2;
                    }
                    _ => {
                        downsample_2x2_rgb_into(
                            &frame_scratch.pyramid_b,
                            cur_w,
                            cur_h,
                            &mut frame_scratch.pyramid_a,
                        );
                        sel = 1;
                    }
                }
                cur_w /= 2;
                cur_h /= 2;
            }
            (cur_w, cur_h, sel)
        };
        let work_rgb: &[u8] = match work_sel {
            1 => &frame_scratch.pyramid_a,
            2 => &frame_scratch.pyramid_b,
            _ => rgb.as_raw(),
        };

        // 3. Build shared 320x256 (when Thermal active) + 160x128 RGB ONCE for EO health, LWIR health & Thermal EO!
        let thermal_eo_rgb: &[u8] = if thermal_sess.is_some() {
            if work_w == 320 && work_h == 256 {
                downsample_2x2_rgb_into(work_rgb, 320, 256, &mut frame_scratch.rgb_160x128);
                work_rgb
            } else {
                resize_rgb_fast_into(
                    work_rgb,
                    work_w,
                    work_h,
                    320,
                    256,
                    &mut frame_scratch.resize,
                    &mut frame_scratch.rgb_320x256,
                );
                downsample_2x2_rgb_into(
                    &frame_scratch.rgb_320x256,
                    320,
                    256,
                    &mut frame_scratch.rgb_160x128,
                );
                &frame_scratch.rgb_320x256
            }
        } else {
            &[]
        };
        let eo_160x128_rgb: &[u8] = if thermal_sess.is_some() {
            &frame_scratch.rgb_160x128
        } else if work_w == 160 && work_h == 128 {
            work_rgb
        } else {
            resize_rgb_fast_into(
                work_rgb,
                work_w,
                work_h,
                160,
                128,
                &mut frame_scratch.resize,
                &mut frame_scratch.rgb_160x128,
            );
            &frame_scratch.rgb_160x128
        };

        let eo_q = observe_eo_rgb_with_fpn_from_160x128(
            &mut shm,
            eo_160x128_rgb,
            native_fpn,
            &mut frame_scratch.eo_luma,
        );
        build_lwir_2ch_from_rgb_slice_into(
            eo_160x128_rgb,
            160,
            128,
            cfg.mode.needs_thermal(),
            &mut lwir_scratch,
        );
        let lwir_st: LwirHealthStatus = shm.observe_lwir_plane_with_ffc_precomputed(
            &lwir_scratch.lwir_u8,
            160,
            128,
            0,
            native_fpn,
        );
        let (q_eo_eff, q_lwir_eff) = shm.compute_effective_qualities(1.0, 1.0);
        let eo_flags = eo_q.flags();
        let lwir_flags = lwir_st.flags();
        let eo_fault = eo_q.degraded
            || eo_q.dead_pixel_fpn_fault
            || eo_q.sensor_noise_fault
            || q_eo_eff < EO_SECTOR_DEGRADED_THR;
        let lwir_fault = lwir_st.degraded
            || lwir_st.mask_lwir_frame
            || lwir_st.dead_pixel_fpn_fault
            || lwir_st.sensor_noise_fault
            || q_lwir_eff < EO_SECTOR_DEGRADED_THR;
        let mut dual_blind_detected = eo_fault && lwir_fault;

        let sensor_fault_this_frame = eo_fault || (cfg.mode.needs_thermal() && lwir_fault);
        if sensor_fault_this_frame {
            sensor_faults_detected += 1;
        }
        if eo_fault && !cfg.mode.needs_thermal() {
            fail_closed_reasons.push(if eo_flags.is_empty() {
                "EO_OPTICAL_DEGRADED".to_string()
            } else {
                format!("EO_FAULT:{}", eo_flags.join(","))
            });
        }

        // ---- Model 1: XVS-D2 + BearingTrackBeforeConfirm + CoordinatedActiveRange ----
        let mut d2_result: Option<FrameD2Result> = None;
        let mut d2_non_finite = false;
        if let (Some(sess), Some(st)) = (d2_sess.as_mut(), d2_state.as_mut()) {
            let t_d2 = Instant::now();
            let d2_h = st.in_h;
            let d2_w = st.in_w;
            let n_px = d2_h * d2_w;
            let d2_rgb: &[u8] = if work_w == d2_w && work_h == d2_h {
                work_rgb
            } else {
                resize_rgb_fast_into(
                    work_rgb,
                    work_w,
                    work_h,
                    d2_w,
                    d2_h,
                    &mut frame_scratch.resize,
                    &mut frame_scratch.rgb_d2,
                );
                &frame_scratch.rgb_d2
            };
            let mut c0 = if st.norm_gray_hist.len() >= 8 {
                st.norm_gray_hist.pop_front().unwrap_or_default()
            } else {
                Vec::with_capacity(n_px)
            };
            rgb_to_scene_norm_gray_into(d2_rgb, n_px, &mut st.scratch_gray, &mut c0);

            let prev1_opt = st.norm_gray_hist.back();
            let r1_val = if prev1_opt.is_some() { 1.0_f32 } else { 0.0_f32 };
            let prev5_opt = if st.norm_gray_hist.len() >= 5 {
                st.norm_gray_hist.get(st.norm_gray_hist.len() - 5)
            } else {
                None
            };
            let r5_val = if prev5_opt.is_some() { 1.0_f32 } else { 0.0_f32 };

            st.scratch_x.resize(6 * n_px, 0.0_f32);
            let (ch0, rest) = st.scratch_x.split_at_mut(n_px);
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
            st.norm_gray_hist.push_back(c0);

            let tx = TensorRef::from_array_view(([1usize, 6, d2_h, d2_w], st.scratch_x.as_slice()))
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("D2 tx: {e}")))?;
            let th = TensorRef::from_array_view((
                [1usize, st.hid_c, st.hid_h, st.hid_w],
                st.h_prev.as_slice(),
            ))
            .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("D2 th: {e}")))?;
            let outs = sess
                .run(ort::inputs!["x_t" => tx, "h_prev_warped" => th])
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("D2 run: {e}")))?;
            let (hm_shape, hm) = outs["hm_logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("D2 hm: {e}")))?;
            let (_, off) = outs["offset"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("D2 off: {e}")))?;
            let (_, lsc) = outs["log_scale"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("D2 lsc: {e}")))?;
            let (_, pmk) = outs["peak_mask"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("D2 pmk: {e}")))?;
            let (_, h_next) = outs["h_next"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("D2 h_next: {e}")))?;

            let outputs_finite = hm.iter().all(|v| v.is_finite())
                && off.iter().all(|v| v.is_finite())
                && lsc.iter().all(|v| v.is_finite())
                && pmk.iter().all(|v| v.is_finite())
                && h_next.iter().all(|v| v.is_finite());
            if !outputs_finite {
                d2_non_finite = true;
                fail_closed_reasons.push("D2_NON_FINITE_OUTPUT".to_string());
            } else if h_next.len() == st.h_prev.len() {
                st.h_prev.copy_from_slice(h_next);
            }

            let (hm_h, hm_w) = if hm_shape.len() == 4 && hm_shape[2] > 0 && hm_shape[3] > 0 {
                (hm_shape[2] as usize, hm_shape[3] as usize)
            } else {
                ((d2_h / 2).max(1), (d2_w / 2).max(1))
            };
            let (raw_det, max_prob) = if outputs_finite {
                decode_d2_top(hm, off, lsc, pmk, hm_h, hm_w, D2_SCORE_THRESHOLD)
            } else {
                (None, 0.0)
            };
            let raw_detected = raw_det.is_some();
            let det = if eo_fault || d2_non_finite {
                None
            } else {
                raw_det
            };
            if det.is_some() {
                detections_total += 1;
            }

            let w_in = img_w as f64;
            let h_in = img_h as f64;
            let (cx_px, cy_px, size_px, size_rate, score, bbox_px) = if let Some(b) = det {
                let sx = w_in / (d2_w as f64);
                let sy = h_in / (d2_h as f64);
                let fx1 = (b[0] as f64) * sx;
                let fy1 = (b[1] as f64) * sy;
                let fx2 = (b[2] as f64) * sx;
                let fy2 = (b[3] as f64) * sy;
                let sz = ((fx2 - fx1) * (fy2 - fy1)).max(1.0).sqrt();
                let sr = st.last_size_px.map(|p| (sz - p) / 0.1).unwrap_or(0.0);
                st.last_size_px = Some(sz);
                st.track_age = st.track_age.saturating_add(1);
                (
                    0.5 * (fx1 + fx2),
                    0.5 * (fy1 + fy2),
                    sz,
                    sr,
                    b[4] as f64,
                    Some([round2(fx1), round2(fy1), round2(fx2), round2(fy2)]),
                )
            } else {
                (
                    0.5 * w_in,
                    0.5 * h_in,
                    st.last_size_px.unwrap_or(10.0),
                    0.0,
                    0.0,
                    None,
                )
            };

            let focal_px = (AOT_FOCAL_PX * (w_in / AOT_REF_WIDTH_PX)).max(1.0);
            let alpha_rad = (cx_px - 0.5 * w_in).atan2(focal_px);
            let beta_rad = (0.5 * h_in - cy_px).atan2(focal_px);
            let tbc_snap = st.bearing_tbc.observe_with_health(
                det.map(|_| (alpha_rad, beta_rad)),
                q_eo_eff,
                eo_fault || d2_non_finite,
            );
            if tbc_snap.confirmed {
                tbc_confirmed_frames += 1;
                if !st.prev_tbc_confirmed {
                    tbc_confirmed_tracks += 1;
                }
            }
            st.prev_tbc_confirmed = tbc_snap.confirmed;

            let meas = ActiveRangeMeasurement {
                alpha_rad,
                beta_rad,
                size_px,
                size_rate_px_s: size_rate,
                contrast_ratio: 0.65,
                sky_beta_ext: 2.0e-4,
                daytime_clear: true,
                track_age: st.track_age.max(1),
                dt_s: 0.1,
            };
            let authority = Ring0AuthorityGate {
                rta_inhibit_avoidance_turns: last_rta_inhibit_turns,
                hal_out_mode_wire: last_rta_mode as u8,
            };
            let telem = st.coord_active_range.step(&own, &meas, None, authority);

            d2_result = Some(FrameD2Result {
                latency_ms: round2(t_d2.elapsed().as_secs_f64() * 1000.0),
                detected: det.is_some(),
                raw_detected,
                max_prob: round4(if max_prob.is_finite() {
                    (max_prob as f64).clamp(0.0, 1.0)
                } else {
                    0.0
                }),
                bbox_xyxy_px: bbox_px,
                score: round4(score.clamp(0.0, 1.0)),
                alpha_rad: round4(alpha_rad),
                beta_rad: round4(beta_rad),
                tbc_status: tbc_status_label(tbc_snap.status),
                tbc_confirmed: tbc_snap.confirmed,
                tbc_hits_in_window: tbc_snap.hits_in_window,
                active_range_mode: active_range_mode_label(telem.mode),
                r_hat_m: round2(telem.r_hat_m as f64),
                r_safe_lower_m: round2(telem.r_safe_lower_m as f64),
                cmd_bank_deg: round2((telem.cmd_bank_rad as f64).to_degrees()),
            });
        }

        // ---- Model 2: XVS-ThermalNet (EO + LWIR Evidential Fusion + Dual-Blind Fail-Closed) ----
        let mut thermal_result: Option<FrameThermalResult> = None;
        let mut thermal_non_finite = false;
        let mut thermal_vacuity_for_rta: Option<f64> = None;
        if let Some(sess) = thermal_sess.as_mut() {
            let t_th = Instant::now();
            let n_eo = 320 * 256;
            scratch_thermal_eo.resize(3 * n_eo, 0.0_f32);
            let inv255 = 1.0_f32 / 255.0_f32;
            let (eo_r, eo_gb) = scratch_thermal_eo.split_at_mut(n_eo);
            let (eo_g, eo_b) = eo_gb.split_at_mut(n_eo);
            for (idx, px) in thermal_eo_rgb.as_chunks::<3>().0.iter().enumerate().take(n_eo) {
                eo_r[idx] = (px[0] as f32) * inv255;
                eo_g[idx] = (px[1] as f32) * inv255;
                eo_b[idx] = (px[2] as f32) * inv255;
            }
            let q_eo_arr = [q_eo_eff as f32];
            let q_lw_arr = [q_lwir_eff as f32];

            let teo = TensorRef::from_array_view((
                [1usize, 3, 256, 320],
                scratch_thermal_eo.as_slice(),
            ))
            .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal eo: {e}")))?;
            let tlw = TensorRef::from_array_view((
                [1usize, 2, 128, 160],
                lwir_scratch.out.as_slice(),
            ))
            .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal lw: {e}")))?;
            let tqe = TensorRef::from_array_view(([1usize], q_eo_arr.as_slice())).map_err(|e| {
                EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal qe: {e}"))
            })?;
            let tql = TensorRef::from_array_view(([1usize], q_lw_arr.as_slice())).map_err(|e| {
                EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal ql: {e}"))
            })?;
            let outs = sess
                .run(ort::inputs!["x_eo" => teo, "x_lwir" => tlw, "q_eo" => tqe, "q_lwir" => tql])
                .map_err(|e| {
                    EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal run: {e}"))
                })?;

            let (_, hm) = outs["fused_hm_logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal hm: {e}"))
                })?;
            let (_, vac) = outs["vacuity"].try_extract_tensor::<f32>().map_err(|e| {
                EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal vac: {e}"))
            })?;
            let (_, weo) = outs["w_eo_eff"].try_extract_tensor::<f32>().map_err(|e| {
                EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal weo: {e}"))
            })?;
            let (_, wlw) = outs["w_lwir_eff"]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    EvalFramesFailure::new("ORT_INFER_ERROR", format!("Thermal wlw: {e}"))
                })?;

            let all_fin = hm.iter().all(|v| v.is_finite())
                && vac.iter().all(|v| v.is_finite())
                && weo.iter().all(|v| v.is_finite())
                && wlw.iter().all(|v| v.is_finite());
            if !all_fin {
                thermal_non_finite = true;
                fail_closed_reasons.push("THERMAL_NON_FINITE_OUTPUT".to_string());
            }

            let mean_vacuity = if all_fin && !vac.is_empty() {
                (vac.iter().map(|&v| v as f64).sum::<f64>() / (vac.len() as f64)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let w_eo_eff = if all_fin && !weo.is_empty() {
                (weo.iter().map(|&v| v as f64).sum::<f64>() / (weo.len() as f64)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let w_lwir_eff = if all_fin && !wlw.is_empty() {
                (wlw.iter().map(|&v| v as f64).sum::<f64>() / (wlw.len() as f64)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut max_hm_logit = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            if all_fin {
                for (idx, &logit) in hm.iter().enumerate() {
                    if logit > max_hm_logit {
                        max_hm_logit = logit;
                        best_idx = idx;
                    }
                }
            }
            let max_hm_prob = if all_fin && max_hm_logit.is_finite() {
                sigmoid_f32(max_hm_logit) as f64
            } else {
                0.0
            };
            let simplex_mode = select_thermal_simplex_mode(q_eo_eff, q_lwir_eff, mean_vacuity);
            let dual_blind = simplex_mode == ThermalSimplexMode::DualDveSvsFallback;
            if dual_blind {
                dual_blind_detected = true;
                fail_closed_reasons.push("THERMAL_DUAL_BLIND_SVS_FALLBACK".to_string());
            }
            thermal_vacuity_for_rta = Some(mean_vacuity);

            thermal_result = Some(FrameThermalResult {
                latency_ms: round2(t_th.elapsed().as_secs_f64() * 1000.0),
                thermal_mode: simplex_mode.as_str(),
                dual_blind,
                mean_vacuity: round4(mean_vacuity),
                max_hm_prob: round4(max_hm_prob),
                w_eo_eff: round4(w_eo_eff),
                w_lwir_eff: round4(w_lwir_eff),
                peak_xy: [best_idx % 80, best_idx / 80],
            });
        }

        // ---- Model 3: XVS-Land (5/6/7-Output Runway Presence Gate + Student-t Calibration) ----
        let mut land_result: Option<FrameLandResult> = None;
        let mut land_rejected = false;
        let mut land_hpl_m = 14.0_f64;
        let mut land_vpl_m = 5.0_f64;
        if let Some((ref mut sess, roi_h, roi_w, ref cal)) = land_sess {
            let t_land = Instant::now();
            let land_rgb: &[u8] = if work_w == roi_w && work_h == roi_h {
                work_rgb
            } else {
                resize_rgb_fast_into(
                    work_rgb,
                    work_w,
                    work_h,
                    roi_w,
                    roi_h,
                    &mut frame_scratch.resize,
                    &mut frame_scratch.rgb_land,
                );
                &frame_scratch.rgb_land
            };
            let n_px = roi_h * roi_w;
            scratch_land_inp.resize(3 * n_px, 0.0_f32);
            let inv255 = 1.0_f32 / 255.0_f32;
            let (r_p, gb_p) = scratch_land_inp.split_at_mut(n_px);
            let (g_p, b_p) = gb_p.split_at_mut(n_px);
            for (idx, px) in land_rgb.as_chunks::<3>().0.iter().enumerate().take(n_px) {
                r_p[idx] = (px[0] as f32) * inv255;
                g_p[idx] = (px[1] as f32) * inv255;
                b_p[idx] = (px[2] as f32) * inv255;
            }
            let troi =
                TensorRef::from_array_view(([1usize, 3, roi_h, roi_w], scratch_land_inp.as_slice()))
                    .map_err(|e| {
                        EvalFramesFailure::new("ORT_INFER_ERROR", format!("Land troi: {e}"))
                    })?;
            let outs = sess
                .run(ort::inputs!["roi_crop" => troi])
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("Land run: {e}")))?;
            let (_, kpts) = outs["keypoints_xy"]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    EvalFramesFailure::new("ORT_INFER_ERROR", format!("Land kpts: {e}"))
                })?;
            let (_, kl) = outs["keylines_hesse"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("Land kl: {e}")))?;
            let (_, cov) = outs["cov_2x2"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EvalFramesFailure::new("ORT_INFER_ERROR", format!("Land cov: {e}")))?;
            let kl_sig_opt = if let Some(sv) = outs.get("keyline_sigma_px") {
                Some(
                    sv.try_extract_tensor::<f32>()
                        .map_err(|e| {
                            EvalFramesFailure::new("ORT_INFER_ERROR", format!("Land kl_sig: {e}"))
                        })?
                        .1,
                )
            } else {
                None
            };
            let rwy_pres_opt = if let Some(rv) = outs.get("runway_present_logit") {
                Some(
                    rv.try_extract_tensor::<f32>()
                        .map_err(|e| {
                            EvalFramesFailure::new(
                                "ORT_INFER_ERROR",
                                format!("Land runway_present_logit: {e}"),
                            )
                        })?
                        .1,
                )
            } else {
                None
            };

            match parse_xvs_land_outputs_with_presence(
                kpts,
                kl,
                cov,
                kl_sig_opt,
                rwy_pres_opt,
                cal,
            ) {
                Ok(parsed) => {
                    let scale_u = 256.0 / (roi_w as f64);
                    let scale_v = 256.0 / (roi_h as f64);
                    let center_u = parsed
                        .keypoints_xy
                        .iter()
                        .map(|p| p[0] * scale_u)
                        .sum::<f64>()
                        / (LAND_NUM_KEYPOINTS as f64);
                    let center_v = parsed
                        .keypoints_xy
                        .iter()
                        .map(|p| p[1] * scale_v)
                        .sum::<f64>()
                        / (LAND_NUM_KEYPOINTS as f64);
                    let lat_dev_deg = round2((center_u - 128.0) / 256.0 * 18.0);
                    let glideslope_deg = round2(3.0 + (128.0 - center_v) / 256.0 * 2.5);
                    let hpl_m = round2((parsed.mean_keypoint_sigma_px * 14.5).max(12.0));
                    let vpl_m = round2((parsed.mean_keypoint_sigma_px * 5.2).max(6.0));
                    let admitted = parsed.runway_present_admitted && !eo_fault;
                    if admitted {
                        land_hpl_m = hpl_m;
                        land_vpl_m = vpl_m;
                    } else {
                        land_rejected = true;
                        fail_closed_reasons.push(if !parsed.runway_present_admitted {
                            "LAND_RUNWAY_ABSENT_P0_INHIBIT".to_string()
                        } else {
                            "LAND_EO_OPTICAL_FAULT_INHIBIT".to_string()
                        });
                    }
                    land_result = Some(FrameLandResult {
                        latency_ms: round2(t_land.elapsed().as_secs_f64() * 1000.0),
                        contract: parsed.contract.as_str(),
                        roi_hw: [roi_h, roi_w],
                        mean_keypoint_sigma_px: round4(parsed.mean_keypoint_sigma_px),
                        effective_keyline_sigma_px: round4(parsed.effective_keyline_sigma_px),
                        runway_present_logit: parsed.runway_present_logit.map(round4),
                        runway_present_prob: parsed.runway_present_prob.map(round4),
                        runway_present_admitted: admitted,
                        runway_gate_mode: if admitted {
                            parsed.runway_gate_mode
                        } else {
                            "P0_INVALID_INHIBIT"
                        },
                        glideslope_deg,
                        lateral_dev_deg: lat_dev_deg,
                        hpl_m,
                        vpl_m,
                    });
                }
                Err(parse_err) => {
                    land_rejected = true;
                    fail_closed_reasons.push(format!("LAND_PARSE_REJECTED:{parse_err}"));
                    land_result = Some(FrameLandResult {
                        latency_ms: round2(t_land.elapsed().as_secs_f64() * 1000.0),
                        contract: "invalid_fail_closed",
                        roi_hw: [roi_h, roi_w],
                        mean_keypoint_sigma_px: 0.0,
                        effective_keyline_sigma_px: 0.0,
                        runway_present_logit: None,
                        runway_present_prob: Some(0.0),
                        runway_present_admitted: false,
                        runway_gate_mode: "P0_INVALID_INHIBIT",
                        glideslope_deg: 0.0,
                        lateral_dev_deg: 0.0,
                        hpl_m: 999.0,
                        vpl_m: 999.0,
                    });
                }
            }
        }

        if dual_blind_detected {
            dual_blind_frames += 1;
        }
        let fail_closed_rejected = !fail_closed_reasons.is_empty();
        if fail_closed_rejected {
            fail_closed_rejections += 1;
        }

        // ---- Step ASTM F3269-21 SimplexRta ----
        let cam_health_mask: u8 = if dual_blind_detected || (!cfg.mode.needs_thermal() && eo_fault)
        {
            0x00
        } else if eo_fault || (cfg.mode.needs_thermal() && lwir_fault) {
            0x1F
        } else {
            0x3F
        };
        let ml_detector_fault = d2_non_finite
            || thermal_non_finite
            || dual_blind_detected
            || (cfg.mode == EvalMode::Land && land_rejected)
            || (cfg.mode == EvalMode::Eo && eo_fault);
        let ood_global_score = if dual_blind_detected {
            1.0
        } else if let Some(vac) = thermal_vacuity_for_rta {
            vac
        } else if eo_fault {
            0.85
        } else {
            0.10
        };
        let ood_cam_scores = if cam_health_mask == 0x00 {
            [1.0; 6]
        } else if cam_health_mask == 0x1F {
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.85]
        } else {
            [0.10; 6]
        };
        let float_tel = RtaFloatTelemetry {
            horizon_pl_rad: 0.002,
            horizon_al_rad: 0.017_453_3,
            v_vpl_m: land_vpl_m.min(9.5),
            v_val_m: 10.0,
            v_hpl_m: land_hpl_m.min(28.0),
            v_hal_m: 30.0,
            epipolar_max_rad: [0.0002; 6],
            signed_horizon_residual_rad: 0.0,
            runway_width_delta_m: 0.0,
            runway_width_delta_limit_m: 5.0,
            cusum_stat: 0.0,
            cusum_thr: CUSUM_H_LOCKED,
            ood_global_score,
            ood_global_thr: 0.7,
            ood_cam_scores,
            ood_cam_thr: 0.7,
            cam_health_mask,
            ml_detector_fault,
            transient_dropout: false,
            approach_active: false,
            go_around_ack_reset: false,
        };
        let rta_out = rta.step(&build_rta_input_from_f64(&float_tel));
        last_rta_mode = rta_out.mode;
        last_rta_inhibit_turns = rta_out.inhibit_avoidance_turns;

        let latency_ms = round2(t_frame.elapsed().as_secs_f64() * 1000.0);
        latencies_ms.push(latency_ms);

        let sensor_health = FrameSensorHealth {
            q_eo_effective: q_eo_eff,
            q_lwir_effective: q_lwir_eff,
            eo_blur_score: eo_q.blur_score,
            eo_focus_quality: eo_q.focus_quality,
            eo_glare_ratio: eo_q.glare_ratio,
            eo_occlusion_ratio: eo_q.occlusion_ratio,
            eo_dead_line_fraction: eo_q.dead_line_fraction,
            eo_spatial_autocorr: eo_q.spatial_autocorr,
            eo_dead_pixel_fpn_fault: eo_q.dead_pixel_fpn_fault,
            eo_sensor_noise_fault: eo_q.sensor_noise_fault,
            eo_degraded: eo_q.degraded,
            eo_flags,
            lwir_nuc_frozen: lwir_st.nuc_frozen,
            lwir_freeze_ticks: lwir_st.freeze_ticks,
            lwir_last_mad: round4(lwir_st.last_mad),
            lwir_dynamic_range_p95_p05: lwir_st.dynamic_range_p95_p05,
            lwir_contrast_collapsed: lwir_st.contrast_collapsed,
            lwir_dead_pixel_fpn_fault: lwir_st.dead_pixel_fpn_fault,
            lwir_sensor_noise_fault: lwir_st.sensor_noise_fault,
            lwir_q_optical: lwir_st.q_lwir_optical,
            lwir_degraded: lwir_st.degraded,
            lwir_flags,
            dual_blind_detected,
        };

        if record_tape {
            let seq = frame_index as u64;
            let payload = serde_json::json!({
                "frame_index": frame_index,
                "filename": filename,
                "eval_mode": cfg.mode.as_str(),
                "corrupt": false,
                "fail_closed_rejected": fail_closed_rejected,
                "latency_ms": latency_ms,
                "down_m": -120.0,
                "gnss_ne_m": [round4((seq as f64) * 3.5), 0.0],
                "vis_ne_m": [round4((seq as f64) * 3.5), 0.0],
                "gnss_down_m": -120.0,
                "vis_gate_admitted": !fail_closed_rejected,
                "cam_health_mask": cam_health_mask,
                "ml_detector_fault": ml_detector_fault,
                "approach_active": false,
                "rta_mode": format!("{:?}", rta_out.mode),
                "rta_mode_id": rta_out.mode as u8,
                "q_eo_effective": q_eo_eff,
                "q_lwir_effective": q_lwir_eff,
                "tbc_confirmed": d2_result.as_ref().is_some_and(|d| d.tbc_confirmed),
                "d2_detected": d2_result.as_ref().is_some_and(|d| d.detected),
                "thermal_mode": thermal_result.as_ref().map(|t| t.thermal_mode).unwrap_or("NOT_RUN"),
                "land_runway_admitted": land_result.as_ref().is_some_and(|l| l.runway_present_admitted),
            });
            append_shadow_tape_record(&mut tape_lines, &mut tape_prev_hash, seq, payload)?;
        }

        frame_records.push(EvalFrameRecord {
            frame_index,
            filename,
            status: if fail_closed_rejected {
                "FAIL_CLOSED"
            } else {
                "OK"
            },
            corrupt: false,
            fail_closed_rejected,
            fail_closed_reasons,
            width: Some(img_w),
            height: Some(img_h),
            latency_ms,
            error: None,
            sensor_health: Some(sensor_health),
            eo_d2: d2_result,
            thermal: thermal_result,
            land: land_result,
            rta_mode: format!("{:?}", rta_out.mode),
            rta_mode_id: rta_out.mode as u8,
            rta_raw_target: format!("{:?}", rta_out.raw_target),
            rta_active_healthy_cams: rta_out.active_healthy_cams,
            rta_inhibit_avoidance_turns: rta_out.inhibit_avoidance_turns,
            rta_mandatory_go_around: rta_out.mandatory_go_around,
        });
    }

    // 4. Write optional SHA-256 hash-chained flight tape (--tape)
    let mut tape_written = false;
    let mut tape_sha256_head = None;
    if let Some(ref tape_path) = cfg.tape_path {
        ensure_parent_dir(tape_path)?;
        let content = if tape_lines.is_empty() {
            String::new()
        } else {
            format!("{}\n", tape_lines.join("\n"))
        };
        std::fs::write(tape_path, content).map_err(|e| {
            EvalFramesFailure::new(
                "TAPE_WRITE_ERROR",
                format!("Failed to write SHA-256 flight tape: {e}"),
            )
        })?;
        tape_written = true;
        tape_sha256_head = Some(tape_prev_hash);
    }

    // 5. Compute latency percentiles and assemble final report
    latencies_ms.sort_by(|a, b| a.total_cmp(b));
    let latency_p50_ms = round2(percentile_sorted(&latencies_ms, 0.50));
    let latency_p95_ms = round2(percentile_sorted(&latencies_ms, 0.95));
    let latency_max_ms = round2(*latencies_ms.last().unwrap_or(&0.0));

    let verdict_plain_english = build_verdict_plain_english(&VerdictStats {
        frames_processed,
        frames_corrupt,
        models_run: &models_run,
        p50_ms: latency_p50_ms,
        p95_ms: latency_p95_ms,
        tbc_confirmed_tracks,
        fail_closed_rejections,
        sensor_faults_detected,
    });

    let input_dir_name = cfg
        .input_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "frames".to_string());

    let report = EvalFramesReport {
        mode: "eval_frames",
        ok: true,
        eval_mode: cfg.mode.as_str(),
        version: EDGE_INFER_VERSION,
        non_flight_sim_only: NON_FLIGHT_SIM_ONLY,
        actuator_authority: ACTUATOR_AUTHORITY,
        outbound_mavlink_frames: 0,
        watermark: NON_FLIGHT_WATERMARK,
        input_dir_name,
        models_loaded,
        tape_written,
        tape_sha256_head,
        summary: EvalFramesSummary {
            frames_total: frame_records.len(),
            frames_processed,
            frames_corrupt,
            models_run,
            latency_p50_ms,
            latency_p95_ms,
            latency_max_ms,
            detections_total,
            tbc_confirmed_tracks,
            tbc_confirmed_frames,
            fail_closed_rejections,
            sensor_faults_detected,
            dual_blind_frames,
            rta_final_mode: format!("{last_rta_mode:?}"),
            rta_final_mode_id: last_rta_mode as u8,
            verdict_plain_english,
        },
        frames: frame_records,
    };

    // 6. Write optional report.json (--out)
    if let Some(ref out_path) = cfg.out_path {
        ensure_parent_dir(out_path)?;
        let json = serde_json::to_string_pretty(&report).map_err(|e| {
            EvalFramesFailure::new(
                "REPORT_SERIALIZATION_ERROR",
                format!("Failed to serialize evaluation report: {e}"),
            )
        })?;
        std::fs::write(out_path, format!("{json}\n")).map_err(|e| {
            EvalFramesFailure::new(
                "REPORT_WRITE_ERROR",
                format!("Failed to write evaluation report JSON: {e}"),
            )
        })?;
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify_and_replay_shadow_ndjson;
    use image::{ImageBuffer, Rgb};

    fn make_textured_rgb_frame(w: u32, h: u32, shift: u32) -> RgbImage {
        ImageBuffer::from_fn(w, h, |x, y| {
            let sx = x.wrapping_add(shift * 7);
            let sy = y.wrapping_add(shift * 5);
            let checker = (((sx / 4) + (sy / 4)) % 2) == 0;
            let grad = ((sx.wrapping_mul(3) + sy.wrapping_mul(5)) % 90) as u8;
            let base = if checker { 155u8 } else { 55u8 };
            let r = base.saturating_add(grad / 2);
            let g = base.saturating_add(grad / 3);
            let b = base.saturating_sub(grad / 4);
            Rgb([r, g, b])
        })
    }

    #[test]
    fn test_eval_mode_parser_and_cli_validation() {
        assert_eq!(EvalMode::parse("all").unwrap(), EvalMode::All);
        assert_eq!(EvalMode::parse("EO").unwrap(), EvalMode::Eo);
        assert_eq!(EvalMode::parse("Thermal").unwrap(), EvalMode::Thermal);
        assert_eq!(EvalMode::parse("land").unwrap(), EvalMode::Land);
        assert!(EvalMode::parse("invalid_mode").is_err());
        assert!(EvalMode::parse("").is_err());
    }

    #[test]
    fn test_eval_frames_missing_and_empty_dir_fail_closed() {
        let missing = std::env::temp_dir().join("openxvs_eval_frames_does_not_exist_9999");
        let _ = std::fs::remove_dir_all(&missing);
        let err_missing = discover_sorted_image_frames(&missing, None).unwrap_err();
        assert_eq!(err_missing.error_code, "DIR_NOT_FOUND");

        let empty_dir = std::env::temp_dir().join("openxvs_eval_frames_empty_dir_test");
        let _ = std::fs::remove_dir_all(&empty_dir);
        std::fs::create_dir_all(&empty_dir).expect("mkdir empty");
        // Put a non-image file and a subdirectory named fake.png (must be ignored!)
        std::fs::write(empty_dir.join("notes.txt"), b"not an image").expect("write txt");
        std::fs::create_dir_all(empty_dir.join("subdir.png")).expect("mkdir fake.png");
        let err_empty = discover_sorted_image_frames(&empty_dir, None).unwrap_err();
        assert_eq!(err_empty.error_code, "EMPTY_FRAME_DIR");

        // Passing a file instead of a directory must fail cleanly
        let err_file =
            discover_sorted_image_frames(&empty_dir.join("notes.txt"), None).unwrap_err();
        assert_eq!(err_file.error_code, "NOT_A_DIRECTORY");

        // --max-frames 0 must fail cleanly
        let err_zero = discover_sorted_image_frames(&empty_dir, Some(0)).unwrap_err();
        assert_eq!(err_zero.error_code, "INVALID_MAX_FRAMES");

        let _ = std::fs::remove_dir_all(&empty_dir);
    }

    #[test]
    fn test_load_rgb_image_safe_rejects_corrupt_empty_and_dimension_bomb() {
        let tmp = std::env::temp_dir().join("openxvs_eval_frames_safe_loader_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir");

        // 1. Corrupt random bytes disguised as .png
        let corrupt_png = tmp.join("corrupt.png");
        std::fs::write(&corrupt_png, b"NOT_A_VALID_PNG_STREAM_1234567890").expect("write corrupt");
        assert!(load_rgb_image_safe(&corrupt_png).is_err());

        // 2. Empty 0-byte file
        let empty_jpg = tmp.join("empty.jpg");
        std::fs::write(&empty_jpg, b"").expect("write empty");
        assert!(load_rgb_image_safe(&empty_jpg).is_err());

        // 3. Minimal PNG header claiming 8192x8192 (decompression bomb > 4096x4096)
        let mut bomb_png: Vec<u8> = Vec::new();
        bomb_png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]); // PNG signature
        bomb_png.extend_from_slice(&13u32.to_be_bytes()); // IHDR length
        bomb_png.extend_from_slice(b"IHDR");
        bomb_png.extend_from_slice(&8192u32.to_be_bytes()); // width = 8192 (> 4096)
        bomb_png.extend_from_slice(&8192u32.to_be_bytes()); // height = 8192 (> 4096)
        bomb_png.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
        bomb_png.extend_from_slice(&[0, 0, 0, 0]); // dummy CRC
        let bomb_path = tmp.join("bomb_8192.png");
        std::fs::write(&bomb_path, &bomb_png).expect("write bomb");
        let bomb_err = load_rgb_image_safe(&bomb_path).unwrap_err();
        assert!(
            bomb_err.contains("4096") || bomb_err.contains("limit") || bomb_err.contains("header"),
            "Unexpected bomb rejection error: {bomb_err}"
        );

        // 4. Valid PNG and valid JPEG decode cleanly
        let img = make_textured_rgb_frame(160, 128, 0);
        let ok_png = tmp.join("valid.png");
        let ok_jpg = tmp.join("valid.jpg");
        img.save_with_format(&ok_png, ImageFormat::Png)
            .expect("save png");
        img.save_with_format(&ok_jpg, ImageFormat::Jpeg)
            .expect("save jpg");
        let dec_png = load_rgb_image_safe(&ok_png).expect("decode valid png");
        let dec_jpg = load_rgb_image_safe(&ok_jpg).expect("decode valid jpg");
        assert_eq!((dec_png.width(), dec_png.height()), (160, 128));
        assert_eq!((dec_jpg.width(), dec_jpg.height()), (160, 128));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_eval_frames_missing_model_dir_fails_closed() {
        let tmp = std::env::temp_dir().join("openxvs_eval_frames_bad_models_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let frames_dir = tmp.join("frames");
        let empty_models = tmp.join("empty_models");
        std::fs::create_dir_all(&frames_dir).expect("mkdir frames");
        std::fs::create_dir_all(&empty_models).expect("mkdir models");
        make_textured_rgb_frame(160, 128, 0)
            .save_with_format(frames_dir.join("0001.png"), ImageFormat::Png)
            .expect("save frame");

        let cfg = EvalFramesConfig {
            input_dir: frames_dir.clone(),
            mode: EvalMode::Eo,
            out_path: None,
            tape_path: None,
            max_frames: None,
            model_dir: Some(empty_models.clone()),
        };
        let err = run_eval_frames(&cfg).unwrap_err();
        assert!(
            err.error_code == "MODEL_NOT_FOUND" || err.error_code == "ORT_DYLIB_UNAVAILABLE",
            "Expected fail-closed model/ort error, got: {err:?}"
        );

        // Tampered model file -> must reject SHA256_MISMATCH_REJECTED
        std::fs::write(empty_models.join("xvs_d2_web_fp16.onnx"), b"TAMPERED").expect("write");
        let err_tamper = run_eval_frames(&cfg).unwrap_err();
        assert!(
            err_tamper.error_code == "SHA256_MISMATCH_REJECTED"
                || err_tamper.error_code == "ORT_DYLIB_UNAVAILABLE",
            "Expected SHA256_MISMATCH_REJECTED, got: {err_tamper:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_eval_frames_synthetic_png_jpg_corrupt_tape_replay_and_tamper() {
        // Skip only if ORT runtime isn't present in environment; on dev/CI with models, runs full end-to-end!
        if crate::resolve_ort_dylib_path().is_err() {
            return;
        }
        let tmp = std::env::temp_dir().join("openxvs_eval_frames_e2e_unit_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let frames_dir = tmp.join("frames");
        std::fs::create_dir_all(&frames_dir).expect("mkdir frames");

        // 3 valid moving frames (mix of PNG + JPEG) + 1 corrupt image in the middle
        make_textured_rgb_frame(320, 240, 0)
            .save_with_format(frames_dir.join("frame_0001.png"), ImageFormat::Png)
            .expect("save f1");
        make_textured_rgb_frame(320, 240, 1)
            .save_with_format(frames_dir.join("frame_0002.jpg"), ImageFormat::Jpeg)
            .expect("save f2");
        std::fs::write(
            frames_dir.join("frame_0003.png"),
            b"\x89PNG\r\n\x1a\nCORRUPT_GARBAGE_BYTES_NOT_A_REAL_IMAGE",
        )
        .expect("write corrupt");
        make_textured_rgb_frame(320, 240, 2)
            .save_with_format(frames_dir.join("frame_0004.jpeg"), ImageFormat::Jpeg)
            .expect("save f4");

        let out_report = tmp.join("out").join("report.json");
        let out_tape = tmp.join("out").join("tape.ndjson");

        let cfg = EvalFramesConfig {
            input_dir: frames_dir.clone(),
            mode: EvalMode::All,
            out_path: Some(out_report.clone()),
            tape_path: Some(out_tape.clone()),
            max_frames: None,
            model_dir: None,
        };
        let rep = run_eval_frames(&cfg).expect("end-to-end --eval-frames all must pass");
        assert!(rep.ok);
        assert!(rep.non_flight_sim_only);
        assert!(!rep.actuator_authority);
        assert_eq!(rep.outbound_mavlink_frames, 0);
        assert_eq!(rep.summary.frames_total, 4);
        assert_eq!(rep.summary.frames_processed, 3);
        assert_eq!(rep.summary.frames_corrupt, 1);
        assert!(rep.summary.fail_closed_rejections >= 1);
        assert_eq!(
            rep.summary.models_run,
            vec!["xvs_d2", "xvs_thermal", "xvs_land"]
        );
        assert!(rep.summary.latency_p50_ms >= 0.0);
        assert!(rep.summary.latency_p95_ms >= rep.summary.latency_p50_ms);
        assert!(rep.summary.latency_max_ms >= rep.summary.latency_p95_ms);
        assert!(!rep.summary.verdict_plain_english.is_empty());
        assert!(out_report.is_file());
        assert!(out_tape.is_file());
        assert!(rep.tape_written && rep.tape_sha256_head.is_some());

        // Verify frame 2 (index 2 -> frame_0003.png) was gracefully rejected fail-closed without panic
        assert!(rep.frames[2].corrupt);
        assert!(rep.frames[2].fail_closed_rejected);
        assert_eq!(rep.frames[2].status, "CORRUPT_REJECTED_FAIL_CLOSED");
        assert!(rep.frames[2].error.is_some());
        assert_eq!(rep.frames[2].rta_active_healthy_cams, 0);

        // Verify report.json parses back as clean JSON with zero null/NaN leaks
        let report_text = std::fs::read_to_string(&out_report).expect("read report.json");
        let parsed_report: serde_json::Value =
            serde_json::from_str(&report_text).expect("valid report JSON");
        assert_eq!(parsed_report["summary"]["frames_processed"], 3);
        assert_eq!(parsed_report["summary"]["frames_corrupt"], 1);

        // Verify generated tape.ndjson passes --replay SHA-256 chain check!
        let replay_ok = verify_and_replay_shadow_ndjson(&out_tape);
        assert!(
            replay_ok.valid,
            "Generated eval_frames tape failed --replay: {:?}",
            replay_ok.errors
        );
        assert_eq!(replay_ok.record_count, 4);
        assert!(replay_ok.tx_inhibited_all);
        assert!(replay_ok.actuator_authority_all_false);
        assert_eq!(replay_ok.total_outbound_mavlink_frames, 0);
        assert!(replay_ok.monotonic_timestamps_ok);

        // 1-byte tamper in tape.ndjson -> MUST fail --replay fail-closed!
        let mut tape_bytes = std::fs::read(&out_tape).expect("read tape bytes");
        let target_idx = tape_bytes
            .windows(10)
            .position(|w| w == b"frame_0002")
            .expect("locate frame_0002 in tape");
        tape_bytes[target_idx + 9] = b'9'; // 1-byte flip: frame_0002 -> frame_0009
        let tampered_tape = tmp.join("out").join("tampered_tape.ndjson");
        std::fs::write(&tampered_tape, &tape_bytes).expect("write tampered tape");
        let replay_bad = verify_and_replay_shadow_ndjson(&tampered_tape);
        assert!(
            !replay_bad.valid,
            "1-byte tampered eval_frames tape MUST fail closed!"
        );
        assert_eq!(replay_bad.first_failed_line, Some(2));

        // Verify --max-frames truncation works cleanly
        let cfg_max = EvalFramesConfig {
            input_dir: frames_dir.clone(),
            mode: EvalMode::Eo,
            out_path: None,
            tape_path: None,
            max_frames: Some(2),
            model_dir: None,
        };
        let rep_max = run_eval_frames(&cfg_max).expect("--max-frames 2 eo mode");
        assert_eq!(rep_max.summary.frames_total, 2);
        assert_eq!(rep_max.summary.frames_processed, 2);
        assert_eq!(rep_max.summary.frames_corrupt, 0);
        assert_eq!(rep_max.summary.models_run, vec!["xvs_d2"]);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_eval_frames_thermal_shutter_freeze_and_dual_blind_detection() {
        if crate::resolve_ort_dylib_path().is_err() {
            return;
        }
        let tmp = std::env::temp_dir().join("openxvs_eval_frames_freeze_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let frames_dir = tmp.join("frames");
        std::fs::create_dir_all(&frames_dir).expect("mkdir");

        // 4 identical frames -> LWIR NUC shutter freeze triggers on tick 3 (frame index 3, 3rd consecutive MAD < 0.25)
        let static_img = make_textured_rgb_frame(160, 128, 0);
        for i in 0..4usize {
            static_img
                .save_with_format(frames_dir.join(format!("frozen_{i:02}.png")), ImageFormat::Png)
                .expect("save static");
        }
        // Frame 4: pure flat black (lens cap + contrast collapse -> EO degraded + LWIR degraded -> dual-blind!)
        let black_img = ImageBuffer::from_pixel(160, 128, Rgb([0u8, 0u8, 0u8]));
        black_img
            .save_with_format(frames_dir.join("frozen_04_black.png"), ImageFormat::Png)
            .expect("save black");

        let cfg = EvalFramesConfig {
            input_dir: frames_dir.clone(),
            mode: EvalMode::Thermal,
            out_path: None,
            tape_path: None,
            max_frames: None,
            model_dir: None,
        };
        let rep = run_eval_frames(&cfg).expect("thermal freeze test");
        assert_eq!(rep.summary.frames_processed, 5);
        assert!(
            rep.frames[3]
                .sensor_health
                .as_ref()
                .is_some_and(|sh| sh.lwir_nuc_frozen && sh.q_lwir_effective == 0.0),
            "4 identical frames must trigger LWIR NUC shutter freeze with q_lwir_effective == 0.0: {:?}",
            rep.frames[3].sensor_health
        );
        assert!(
            rep.frames[4]
                .thermal
                .as_ref()
                .is_some_and(|th| th.dual_blind && th.thermal_mode == "DUAL_DVE_SVS_FALLBACK"),
            "Flat black frame must trigger dual-blind DUAL_DVE_SVS_FALLBACK: {:?}",
            rep.frames[4].thermal
        );
        assert!(rep.summary.sensor_faults_detected >= 2);
        assert!(rep.summary.dual_blind_frames >= 1);
        assert!(rep.summary.fail_closed_rejections >= 1);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_eval_frames_fpn_dead_pixels_white_noise_and_land_mode() {
        if crate::resolve_ort_dylib_path().is_err() {
            return;
        }
        let tmp = std::env::temp_dir().join("openxvs_eval_frames_fpn_noise_land_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let frames_dir = tmp.join("frames");
        std::fs::create_dir_all(&frames_dir).expect("mkdir");

        // Frame 0: clean textured scene
        let clean = make_textured_rgb_frame(160, 128, 0);
        clean
            .save_with_format(frames_dir.join("00_clean.png"), ImageFormat::Png)
            .expect("save clean");

        // Frame 1: spaced dead readout columns (>= 10% dead lines -> FPN fault)
        let mut fpn = make_textured_rgb_frame(160, 128, 1);
        for col in (4..156u32).step_by(6) {
            for y in 0..128u32 {
                fpn.put_pixel(col, y, Rgb([255, 255, 255]));
            }
        }
        fpn.save_with_format(frames_dir.join("01_fpn.png"), ImageFormat::Png)
            .expect("save fpn");

        // Frame 2: spatially uncorrelated white noise (LCG PRNG -> sensor_noise_fault)
        let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
        let noise = ImageBuffer::from_fn(160, 128, |_, _| {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((lcg >> 33) & 0xFF) as u8;
            Rgb([v, v, v])
        });
        noise
            .save_with_format(frames_dir.join("02_noise.png"), ImageFormat::Png)
            .expect("save noise");

        let cfg = EvalFramesConfig {
            input_dir: frames_dir.clone(),
            mode: EvalMode::Land,
            out_path: None,
            tape_path: None,
            max_frames: None,
            model_dir: None,
        };
        let rep = run_eval_frames(&cfg).expect("land mode fpn/noise evaluation");
        assert_eq!(rep.summary.frames_processed, 3);
        assert_eq!(rep.summary.models_run, vec!["xvs_land"]);
        assert!(
            rep.frames[0]
                .land
                .as_ref()
                .is_some_and(|l| l.contract == "v52_6output" && l.roi_hw == [384, 384]),
            "Expected v5.2 land output on clean frame: {:?}",
            rep.frames[0].land
        );
        assert!(
            rep.frames[1]
                .sensor_health
                .as_ref()
                .is_some_and(|sh| sh.eo_dead_pixel_fpn_fault),
            "Frame 1 must detect DEAD_PIXEL_FPN_FAULT: {:?}",
            rep.frames[1].sensor_health
        );
        assert!(
            rep.frames[2]
                .sensor_health
                .as_ref()
                .is_some_and(|sh| sh.eo_sensor_noise_fault),
            "Frame 2 must detect SENSOR_NOISE_FAULT: {:?}",
            rep.frames[2].sensor_health
        );
        assert!(rep.frames[1].fail_closed_rejected);
        assert!(rep.frames[2].fail_closed_rejected);
        assert!(rep.summary.sensor_faults_detected >= 2);
        assert!(rep.summary.fail_closed_rejections >= 2);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_fast_rgb_resizer_dc_exactness_odd_dims_and_triangle_parity() {
        use image::imageops::FilterType;
        let mut scratch = ResizeScratch::new();
        let mut dst = Vec::new();

        // 1. Solid-color DC exactness across downscale, upscale, identity, 2x2, mixed & odd dimensions
        let cases = [
            (1920usize, 1080usize, 384usize, 384usize),
            (1920, 1004, 320, 256),
            (320, 256, 160, 128),
            (160, 128, 384, 384),
            (256, 256, 256, 256),
            (191, 103, 73, 59),
            (640, 120, 256, 256),
            (1, 1, 16, 12),
        ];
        for (sw, sh, dw, dh) in cases {
            let mut src = Vec::with_capacity(sw * sh * 3);
            for _ in 0..(sw * sh) {
                src.extend_from_slice(&[37u8, 149u8, 231u8]);
            }
            resize_rgb_fast_into(&src, sw, sh, dw, dh, &mut scratch, &mut dst);
            assert_eq!(dst.len(), dw * dh * 3, "size mismatch for {sw}x{sh}->{dw}x{dh}");
            for px in dst.as_chunks::<3>().0 {
                assert_eq!(
                    px,
                    &[37u8, 149u8, 231u8],
                    "DC drift on solid color {sw}x{sh}->{dw}x{dh}"
                );
            }
        }

        // 2. Smooth gradient image parity vs image::imageops::resize(..., FilterType::Triangle)
        let grad = ImageBuffer::from_fn(640, 480, |x, y| {
            Rgb([
                ((x * 255) / 639) as u8,
                ((y * 255) / 479) as u8,
                (((x + y) * 255) / (639 + 479)) as u8,
            ])
        });
        for (dw, dh) in [(384usize, 384usize), (320, 256), (256, 256), (160, 128)] {
            resize_rgb_fast_into(grad.as_raw(), 640, 480, dw, dh, &mut scratch, &mut dst);
            let ref_img = image::imageops::resize(&grad, dw as u32, dh as u32, FilterType::Triangle);
            let mae: f64 = dst
                .iter()
                .zip(ref_img.as_raw().iter())
                .map(|(&a, &b)| a.abs_diff(b) as f64)
                .sum::<f64>()
                / (dst.len() as f64);
            assert!(
                mae < 1.5,
                "Fast resizer MAE vs Triangle too high on 640x480->{dw}x{dh}: {mae:.4}"
            );
        }
    }

    #[test]
    fn test_white_noise_integer_covariance_exact_parity() {
        let (w, h) = (160usize, 128usize);
        let mut lcg: u64 = 0x1234_5678_9ABC_DEF0;
        let mut noisy = vec![0u8; w * h];
        for px in &mut noisy {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *px = ((lcg >> 33) & 0xFF) as u8;
        }
        let mut smooth = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                smooth[y * w + x] = (((x * 2 + y * 3) / 4) & 0xFF) as u8;
            }
        }
        // Moderate correlation (2-pixel correlated blocks + noise -> ac in ~0.4..0.6, bypasses periodic_checker)
        let mut moderate = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let bx = x / 2;
                let base = ((bx * 37 + y * 13) % 180) as u8;
                let jitter = noisy[y * w + x] / 3;
                moderate[y * w + x] = base.saturating_add(jitter);
            }
        }
        for plane in [&noisy, &smooth, &moderate] {
            let (ac_fast, score_fast) = detect_white_noise_fault_u8(plane, w, h);
            // Reference f64 direct covariance formula (including lag-2 periodic_checker branch)
            let n = (w * h) as f64;
            let mean = plane.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = plane
                .iter()
                .map(|&v| {
                    let d = (v as f64) - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            let mut cov_h1 = 0.0_f64;
            let mut cov_h2 = 0.0_f64;
            let mut cov_v1 = 0.0_f64;
            let mut cov_v2 = 0.0_f64;
            for y in 0..h {
                for x in 0..(w - 1) {
                    cov_h1 += ((plane[y * w + x] as f64) - mean)
                        * ((plane[y * w + x + 1] as f64) - mean);
                }
                for x in 0..(w - 2) {
                    cov_h2 += ((plane[y * w + x] as f64) - mean)
                        * ((plane[y * w + x + 2] as f64) - mean);
                }
                if y + 1 < h {
                    for x in 0..w {
                        cov_v1 += ((plane[y * w + x] as f64) - mean)
                            * ((plane[(y + 1) * w + x] as f64) - mean);
                    }
                }
                if y + 2 < h {
                    for x in 0..w {
                        cov_v2 += ((plane[y * w + x] as f64) - mean)
                            * ((plane[(y + 2) * w + x] as f64) - mean);
                    }
                }
            }
            let ac_h1 = (cov_h1 / ((h * (w - 1)) as f64)) / var;
            let ac_h2 = (cov_h2 / ((h * (w - 2)) as f64)) / var;
            let ac_v1 = (cov_v1 / (((h - 1) * w) as f64)) / var;
            let ac_v2 = (cov_v2 / (((h - 2) * w) as f64)) / var;
            let ac_ref = if ac_h2.abs().max(ac_v2.abs()) >= 0.70 {
                ac_h2.abs().max(ac_v2.abs())
            } else {
                ac_h1.max(ac_v1)
            };
            assert!(
                (ac_fast - ac_ref).abs() < 1e-10,
                "Integer covariance parity mismatch: fast={ac_fast} ref={ac_ref} score={score_fast}"
            );
        }
    }
}
