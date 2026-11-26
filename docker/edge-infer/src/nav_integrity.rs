// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Pure-Rust Avionics Navigation Integrity Modules (`#![forbid(unsafe_code)]`, zero external math crates):
//!   1. `compute_psr_calibrated` & `evaluate_visual_aliasing_gate` (`src/xvs/v2/xvs_nav/aliasing_gate.py` parity)
//!   2. `SequenceVprFilter` (`W=7` Bayesian/Viterbi sliding-window IMU-odometry filter + blind/flat guard;
//!      `src/xvs/v2/xvs_nav/sequence_vpr.py` parity)
//!   3. `EightArmCholeskyCusum` & `DualBankEsEkfCusum` (10D Gauss-Markov multipath-aware H0 observer +
//!      orthogonal-eigenvalue multipath estimator + `dt`-invariant 8-arm Cholesky Page's CUSUM +
//!      persistent constant-step monitor + 2-sided vertical `Down` GNSS-vs-Baro CUSUM;
//!      `src/xvs/v2/xvs_nav/dual_bank_cusum.py` parity)
//!
//! Strictly local NED (`[north_m, east_m, down_m]`, zero internal ENU) and strictly fail-closed on `NaN`/`Inf`.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

// ---- Locked DR-02 / DR-21 / NAV6 Constants ----

pub const DEFAULT_SEQ_WINDOW: usize = 7;
pub const DEFAULT_NON_ADJ_RADIUS_M: f64 = 300.0;
pub const MIN_PSR_LOCKED: f64 = 1.15;
pub const MAX_CHI2_MC_LOCKED: f64 = 13.82;
pub const MAX_Q_VIS_LOCKED: f64 = 18.42;
pub const DEFAULT_HAL_NAV_M: f64 = 30.0;

pub const CUSUM_NUM_ARMS: usize = 8;
pub const CUSUM_K_D_LOCKED: f64 = 0.50;
pub const CUSUM_H_LOCKED: f64 = 17.619;
pub const CUSUM_REINTEGRATION_HYSTERESIS_EPOCHS: usize = 15;
pub const CUSUM_REINTEGRATION_STAT_FRAC: f64 = 0.25;

#[inline]
fn cusum_arm_unit_vectors() -> [[f64; 2]; CUSUM_NUM_ARMS] {
    let mut out = [[0.0_f64; 2]; CUSUM_NUM_ARMS];
    for (m, slot) in out.iter_mut().enumerate() {
        let ang = (m as f64) * core::f64::consts::FRAC_PI_4;
        *slot = [ang.cos(), ang.sin()];
    }
    out
}

#[inline]
fn is_finite_ge(val: f64, thr: f64) -> bool {
    val.is_finite() && thr.is_finite() && val >= thr
}

#[inline]
fn is_finite_le(val: f64, thr: f64) -> bool {
    val.is_finite() && thr.is_finite() && val <= thr
}

// ============================================================================
// Part 1: Calibrated PSR & 6-Stage Visual Aliasing Admission Gate
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisualAliasingThresholds {
    pub min_inliers: i64,
    pub min_n_eff: f64,
    pub min_spatial_entropy: f64,
    pub max_kappa_geom: f64,
    pub max_kappa_fim: f64,
    pub min_psr: f64,
    pub max_chi2_mc: f64,
    pub max_q_vis: f64,
    pub min_psr_delta: f64,
    pub default_hal_nav_m: f64,
}

impl Default for VisualAliasingThresholds {
    fn default() -> Self {
        Self {
            min_inliers: 30,
            min_n_eff: 4.0,
            min_spatial_entropy: 0.55,
            max_kappa_geom: 12.0,
            max_kappa_fim: 10.0,
            min_psr: MIN_PSR_LOCKED,
            max_chi2_mc: MAX_CHI2_MC_LOCKED,
            max_q_vis: MAX_Q_VIS_LOCKED,
            min_psr_delta: 0.0,
            default_hal_nav_m: DEFAULT_HAL_NAV_M,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualFixMetrics {
    pub n_inliers: i64,
    pub n_eff: f64,
    pub h_norm: f64,
    pub kappa_geom: f64,
    pub kappa_fim: f64,
    pub psr: f64,
    pub chi2_mc: f64,
    pub q_vis: f64,
    pub hpl_nav_m: f64,
    pub hal_nav_m: f64,
    #[serde(default)]
    pub psr_margin: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VisualAliasingDecision {
    pub admitted: bool,
    pub integrity_alert: bool,
    pub failed_stages: Vec<String>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct CalibratedPsrResult {
    pub psr_cal: f64,
    pub margin: f64,
    pub s1: f64,
    pub s2_non_adj: f64,
    pub mu_bg: f64,
    pub delta_min: f64,
    pub min_psr: f64,
    pub admitted: bool,
}

pub fn compute_psr_calibrated(
    similarities: &[f64],
    gallery_ne_m: &[[f64; 2]],
    non_adj_radius_m: f64,
    mu_bg: Option<f64>,
    delta_min: f64,
    min_psr: f64,
    eps: f64,
) -> CalibratedPsrResult {
    let fail = |s1: f64, s2: f64, bg: f64| CalibratedPsrResult {
        psr_cal: f64::NAN,
        margin: f64::NAN,
        s1,
        s2_non_adj: s2,
        mu_bg: bg,
        delta_min,
        min_psr,
        admitted: false,
    };

    if similarities.len() < 2
        || gallery_ne_m.len() != similarities.len()
        || !similarities.iter().all(|v| v.is_finite())
        || !gallery_ne_m
            .iter()
            .all(|p| p[0].is_finite() && p[1].is_finite())
        || !(non_adj_radius_m.is_finite()
            && delta_min.is_finite()
            && min_psr.is_finite()
            && eps.is_finite())
    {
        return fail(f64::NAN, f64::NAN, f64::NAN);
    }

    // Strict `>` matches numpy.argmax tie-breaking (lowest index wins)
    let mut best_idx = 0usize;
    let mut s1 = similarities[0];
    for (i, &val) in similarities.iter().enumerate().skip(1) {
        if val > s1 {
            s1 = val;
            best_idx = i;
        }
    }

    let ref_p = gallery_ne_m[best_idx];
    let mut non_adj: Vec<f64> = Vec::with_capacity(similarities.len());
    for (i, &sim) in similarities.iter().enumerate() {
        let dist = (gallery_ne_m[i][0] - ref_p[0]).hypot(gallery_ne_m[i][1] - ref_p[1]);
        if dist > non_adj_radius_m {
            non_adj.push(sim);
        }
    }
    if non_adj.is_empty() {
        return fail(s1, f64::NAN, f64::NAN);
    }

    let mut s2_non_adj = f64::NEG_INFINITY;
    for &v in &non_adj {
        if v > s2_non_adj {
            s2_non_adj = v;
        }
    }

    let bg_val = match mu_bg {
        Some(v) => v,
        None => {
            non_adj.sort_by(|a, b| a.total_cmp(b));
            let n = non_adj.len();
            if n % 2 == 1 {
                non_adj[n / 2]
            } else {
                0.5 * (non_adj[n / 2 - 1] + non_adj[n / 2])
            }
        }
    };
    if !bg_val.is_finite() {
        return fail(s1, s2_non_adj, bg_val);
    }

    let num = (s1 - bg_val).max(eps);
    let den = (s2_non_adj - bg_val).max(eps);
    let psr_cal = num / den;
    let margin = s1 - s2_non_adj;
    let admitted = psr_cal.is_finite()
        && margin.is_finite()
        && psr_cal >= min_psr
        && margin >= delta_min;

    CalibratedPsrResult {
        psr_cal,
        margin,
        s1,
        s2_non_adj,
        mu_bg: bg_val,
        delta_min,
        min_psr,
        admitted,
    }
}

#[inline]
fn fmt_py_f64(val: f64) -> String {
    if val.is_nan() {
        return "nan".to_string();
    }
    if val.is_infinite() {
        return if val.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    let s = format!("{val}");
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        format!("{s}.0")
    } else {
        s
    }
}

pub fn evaluate_visual_aliasing_gate(
    metrics: &VisualFixMetrics,
    thresholds: &VisualAliasingThresholds,
) -> VisualAliasingDecision {
    let mut failed_stages: Vec<String> = Vec::new();
    let mut reasons: Vec<String> = Vec::new();

    // Stage 1: Masked Support (N >= 30 AND N_eff >= 4.0)
    let n_ok = metrics.n_inliers >= thresholds.min_inliers;
    let neff_ok = is_finite_ge(metrics.n_eff, thresholds.min_n_eff);
    let s1_ok = n_ok && neff_ok;
    if !s1_ok {
        failed_stages.push("stage_1_support".to_string());
        if !n_ok {
            reasons.push(format!(
                "S1_MIN_INLIERS_FAIL(N={}<{})",
                metrics.n_inliers, thresholds.min_inliers
            ));
        }
        if !neff_ok {
            reasons.push(format!(
                "S1_MIN_NEFF_FAIL(N_eff={}<{})",
                fmt_py_f64(metrics.n_eff),
                fmt_py_f64(thresholds.min_n_eff)
            ));
        }
    }

    // Stage 2: Spatial Grid Shannon Entropy (H_norm >= 0.55)
    let s2_ok = is_finite_ge(metrics.h_norm, thresholds.min_spatial_entropy);
    if !s2_ok {
        failed_stages.push("stage_2_entropy".to_string());
        reasons.push(format!(
            "S2_ENTROPY_FAIL(H_norm={}<{})",
            fmt_py_f64(metrics.h_norm),
            fmt_py_f64(thresholds.min_spatial_entropy)
        ));
    }

    // Stage 3: Geometric & FIM Conditioning
    let kg_ok = is_finite_le(metrics.kappa_geom, thresholds.max_kappa_geom);
    let kf_ok = is_finite_le(metrics.kappa_fim, thresholds.max_kappa_fim);
    let s3_ok = kg_ok && kf_ok;
    if !s3_ok {
        failed_stages.push("stage_3_conditioning".to_string());
        if !kg_ok {
            reasons.push(format!(
                "S3_KAPPA_GEOM_FAIL(kappa_geom={}>{})",
                fmt_py_f64(metrics.kappa_geom),
                fmt_py_f64(thresholds.max_kappa_geom)
            ));
        }
        if !kf_ok {
            reasons.push(format!(
                "S3_KAPPA_FIM_FAIL(kappa_fim={}>{})",
                fmt_py_f64(metrics.kappa_fim),
                fmt_py_f64(thresholds.max_kappa_fim)
            ));
        }
    }

    // Stage 4: Retrieval Distinctiveness (PSR_cal >= 1.15 AND optional margin >= delta_min)
    let psr_ok = is_finite_ge(metrics.psr, thresholds.min_psr);
    let margin_ok = match metrics.psr_margin {
        Some(m) => is_finite_ge(m, thresholds.min_psr_delta),
        None => true,
    };
    let s4_ok = psr_ok && margin_ok;
    if !s4_ok {
        failed_stages.push("stage_4_psr".to_string());
        if !psr_ok {
            reasons.push(format!(
                "S4_PSR_FAIL(PSR={}<{})",
                fmt_py_f64(metrics.psr),
                fmt_py_f64(thresholds.min_psr)
            ));
        }
        if !margin_ok {
            let m_str = metrics
                .psr_margin
                .map(fmt_py_f64)
                .unwrap_or_else(|| "None".to_string());
            reasons.push(format!(
                "S4_PSR_MARGIN_FAIL(margin={}<{})",
                m_str,
                fmt_py_f64(thresholds.min_psr_delta)
            ));
        }
    }

    // Stage 5: Multi-Camera Consistency
    let s5_ok = is_finite_le(metrics.chi2_mc, thresholds.max_chi2_mc);
    if !s5_ok {
        failed_stages.push("stage_5_multicam".to_string());
        reasons.push(format!(
            "S5_MULTICAM_CHI2_FAIL(chi2_MC={}>{})",
            fmt_py_f64(metrics.chi2_mc),
            fmt_py_f64(thresholds.max_chi2_mc)
        ));
    }

    // Stage 6: Temporal Prior Innovation Gate
    let s6_ok = is_finite_le(metrics.q_vis, thresholds.max_q_vis);
    if !s6_ok {
        failed_stages.push("stage_6_innovation".to_string());
        reasons.push(format!(
            "S6_INNOVATION_QVIS_FAIL(q_vis={}>{})",
            fmt_py_f64(metrics.q_vis),
            fmt_py_f64(thresholds.max_q_vis)
        ));
    }

    // Continuous Integrity Alert: HPL_nav <= HAL_nav
    let hpl_ok = metrics.hpl_nav_m.is_finite()
        && metrics.hal_nav_m.is_finite()
        && metrics.hpl_nav_m >= 0.0
        && metrics.hal_nav_m > 0.0
        && metrics.hpl_nav_m <= metrics.hal_nav_m;
    let integrity_alert = !hpl_ok;
    if integrity_alert {
        failed_stages.push("integrity_hpl_nav".to_string());
        reasons.push(format!(
            "INTEGRITY_ALERT_HPL_EXCEEDS_HAL(HPL_nav={},HAL_nav={})",
            fmt_py_f64(metrics.hpl_nav_m),
            fmt_py_f64(metrics.hal_nav_m)
        ));
    }

    let admitted = s1_ok && s2_ok && s3_ok && s4_ok && s5_ok && s6_ok && hpl_ok;
    VisualAliasingDecision {
        admitted,
        integrity_alert,
        failed_stages,
        reason_codes: reasons,
    }
}

// ============================================================================
// Part 2: Pure-Rust SequenceVprFilter (W=7 Bayesian / Viterbi + Blind/Flat Guard)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SequenceMode {
    Bayesian,
    Viterbi,
}

impl SequenceMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bayesian => "bayesian",
            Self::Viterbi => "viterbi",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SequenceVprConfig {
    pub window_size: usize,
    pub mode: SequenceMode,
    pub emission_temperature: f64,
    pub odom_sigma_floor_m: f64,
    pub odom_sigma_scale: f64,
    pub max_step_residual_m: f64,
    pub teleport_log_prob: f64,
    pub non_adj_radius_m: f64,
    pub min_psr_cal: f64,
    pub min_psr_delta: f64,
    pub min_posterior_prob: f64,
    pub min_posterior_margin: f64,
    pub min_raw_sim_floor: f64,
    pub min_raw_sim_std: f64,
}

impl Default for SequenceVprConfig {
    fn default() -> Self {
        Self {
            window_size: DEFAULT_SEQ_WINDOW,
            mode: SequenceMode::Bayesian,
            emission_temperature: 0.05,
            odom_sigma_floor_m: 20.0,
            odom_sigma_scale: 1.0,
            max_step_residual_m: 450.0,
            teleport_log_prob: -18.0,
            non_adj_radius_m: DEFAULT_NON_ADJ_RADIUS_M,
            min_psr_cal: MIN_PSR_LOCKED,
            min_psr_delta: 0.0,
            min_posterior_prob: 0.25,
            min_posterior_margin: 0.05,
            min_raw_sim_floor: 0.05,
            min_raw_sim_std: 1.0e-6,
        }
    }
}

impl SequenceVprConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size < 1 {
            return Err(format!("window_size must be >= 1, got {}", self.window_size));
        }
        if !(self.emission_temperature.is_finite() && self.emission_temperature > 0.0) {
            return Err(format!(
                "emission_temperature must be finite and > 0, got {}",
                self.emission_temperature
            ));
        }
        if !(self.odom_sigma_floor_m.is_finite() && self.odom_sigma_floor_m > 0.0) {
            return Err(format!(
                "odom_sigma_floor_m must be finite and > 0, got {}",
                self.odom_sigma_floor_m
            ));
        }
        if !(self.odom_sigma_scale.is_finite() && self.odom_sigma_scale > 0.0) {
            return Err(format!(
                "odom_sigma_scale must be finite and > 0, got {}",
                self.odom_sigma_scale
            ));
        }
        if !(self.max_step_residual_m.is_finite() && self.max_step_residual_m > 0.0) {
            return Err(format!(
                "max_step_residual_m must be finite and > 0, got {}",
                self.max_step_residual_m
            ));
        }
        if !(self.teleport_log_prob.is_finite() && self.teleport_log_prob < 0.0) {
            return Err(format!(
                "teleport_log_prob must be finite and < 0, got {}",
                self.teleport_log_prob
            ));
        }
        if !(self.non_adj_radius_m.is_finite() && self.non_adj_radius_m > 0.0) {
            return Err(format!(
                "non_adj_radius_m must be finite and > 0, got {}",
                self.non_adj_radius_m
            ));
        }
        if !(self.min_psr_cal.is_finite() && self.min_psr_cal >= 1.0) {
            return Err(format!(
                "min_psr_cal must be finite and >= 1.0, got {}",
                self.min_psr_cal
            ));
        }
        if !(self.min_psr_delta.is_finite() && self.min_psr_delta >= 0.0) {
            return Err(format!(
                "min_psr_delta must be finite and >= 0.0, got {}",
                self.min_psr_delta
            ));
        }
        if !(self.min_posterior_prob.is_finite() && (0.0..=1.0).contains(&self.min_posterior_prob))
        {
            return Err(format!(
                "min_posterior_prob must be in [0, 1], got {}",
                self.min_posterior_prob
            ));
        }
        if !(self.min_posterior_margin.is_finite()
            && (0.0..=1.0).contains(&self.min_posterior_margin))
        {
            return Err(format!(
                "min_posterior_margin must be in [0, 1], got {}",
                self.min_posterior_margin
            ));
        }
        if !(self.min_raw_sim_floor.is_finite() && (-1.0..=1.0).contains(&self.min_raw_sim_floor))
        {
            return Err(format!(
                "min_raw_sim_floor must be finite in [-1, 1], got {}",
                self.min_raw_sim_floor
            ));
        }
        if !(self.min_raw_sim_std.is_finite() && self.min_raw_sim_std >= 0.0) {
            return Err(format!(
                "min_raw_sim_std must be finite and >= 0, got {}",
                self.min_raw_sim_std
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SequenceVprDecision {
    pub frame_idx: usize,
    pub window_len: usize,
    pub mode: &'static str,
    pub raw_top1_idx: i64,
    pub raw_top5_indices: Vec<usize>,
    pub seq_top1_idx: i64,
    pub seq_top5_indices: Vec<usize>,
    pub final_top1_idx: i64,
    pub final_top5_indices: Vec<usize>,
    pub viterbi_path_indices: Vec<usize>,
    pub posterior_prob_top1: f64,
    pub posterior_margin_non_adj: f64,
    pub fused_scores: Vec<f64>,
    pub psr_result: CalibratedPsrResult,
    pub admitted: bool,
    pub reason_codes: Vec<String>,
    pub aliasing_decision: Option<VisualAliasingDecision>,
}

#[derive(Debug, Clone)]
struct WindowObservation {
    emit_prob: Vec<f64>,
    raw_sims: Vec<f64>,
    log_trans: Option<Vec<f64>>,  // row-major N x N
    trans_prob: Option<Vec<f64>>, // row-major N x N
    delta_ne_odom_m: [f64; 2],
    odom_sigma_m: f64,
}

#[derive(Debug, Clone)]
pub struct SequenceVprFilter {
    gallery_ne_m: Vec<[f64; 2]>,
    num_tiles: usize,
    config: SequenceVprConfig,
    dn_ij: Vec<f64>, // row-major N x N: gallery[j][0] - gallery[i][0]
    de_ij: Vec<f64>, // row-major N x N: gallery[j][1] - gallery[i][1]
    window: VecDeque<WindowObservation>,
    frame_counter: usize,
}

fn argsort_desc_stable(vals: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..vals.len()).collect();
    order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]).then_with(|| a.cmp(&b)));
    order
}

impl SequenceVprFilter {
    pub fn new(gallery_ne_m: Vec<[f64; 2]>, config: SequenceVprConfig) -> Result<Self, String> {
        if gallery_ne_m.len() < 2 {
            return Err(format!(
                "Expected gallery_ne_m of length >= 2 in NED meters, got {}",
                gallery_ne_m.len()
            ));
        }
        if !gallery_ne_m
            .iter()
            .all(|p| p[0].is_finite() && p[1].is_finite())
        {
            return Err("gallery_ne_m contains non-finite coordinates".to_string());
        }
        config.validate()?;

        let n = gallery_ne_m.len();
        let mut dn_ij = vec![0.0_f64; n * n];
        let mut de_ij = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                dn_ij[i * n + j] = gallery_ne_m[j][0] - gallery_ne_m[i][0];
                de_ij[i * n + j] = gallery_ne_m[j][1] - gallery_ne_m[i][1];
            }
        }
        let cap = config.window_size;
        Ok(Self {
            gallery_ne_m,
            num_tiles: n,
            config,
            dn_ij,
            de_ij,
            window: VecDeque::with_capacity(cap),
            frame_counter: 0,
        })
    }

    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.frame_counter = 0;
    }

    fn compute_log_transition(&self, delta_ne_odom_m: [f64; 2], odom_sigma_m: f64) -> Vec<f64> {
        let cfg = &self.config;
        let sigma_eff = cfg
            .odom_sigma_floor_m
            .max(cfg.odom_sigma_scale * odom_sigma_m);
        let inv_2var = 0.5 / (sigma_eff * sigma_eff);
        let max_r2 = cfg.max_step_residual_m * cfg.max_step_residual_m;
        let nn = self.num_tiles * self.num_tiles;
        let mut log_t = vec![cfg.teleport_log_prob; nn];
        for k in 0..nn {
            let err_n = self.dn_ij[k] - delta_ne_odom_m[0];
            let err_e = self.de_ij[k] - delta_ne_odom_m[1];
            let r2 = err_n * err_n + err_e * err_e;
            if r2 <= max_r2 {
                log_t[k] = (-r2 * inv_2var).max(cfg.teleport_log_prob);
            }
        }
        log_t
    }

    fn row_stochastic_from_log_trans(n: usize, log_trans: &[f64]) -> Vec<f64> {
        let mut trans_prob = vec![0.0_f64; n * n];
        for i in 0..n {
            let row_off = i * n;
            let mut row_max = f64::NEG_INFINITY;
            for j in 0..n {
                let v = log_trans[row_off + j];
                if v > row_max {
                    row_max = v;
                }
            }
            let mut row_sum = 0.0_f64;
            for j in 0..n {
                let e = (log_trans[row_off + j] - row_max).clamp(-80.0, 0.0).exp();
                trans_prob[row_off + j] = e;
                row_sum += e;
            }
            let denom = row_sum.max(1e-300);
            for j in 0..n {
                trans_prob[row_off + j] /= denom;
            }
        }
        trans_prob
    }

    fn make_fail_closed_decision(
        &self,
        frame_idx: usize,
        reason_codes: Vec<String>,
        raw_top1_idx: i64,
        raw_top5_indices: Vec<usize>,
    ) -> SequenceVprDecision {
        SequenceVprDecision {
            frame_idx,
            window_len: self.window.len(),
            mode: self.config.mode.as_str(),
            raw_top1_idx,
            raw_top5_indices,
            seq_top1_idx: -1,
            seq_top5_indices: Vec::new(),
            final_top1_idx: -1,
            final_top5_indices: Vec::new(),
            viterbi_path_indices: Vec::new(),
            posterior_prob_top1: 0.0,
            posterior_margin_non_adj: 0.0,
            fused_scores: vec![f64::NAN; self.num_tiles],
            psr_result: CalibratedPsrResult {
                psr_cal: f64::NAN,
                margin: f64::NAN,
                s1: f64::NAN,
                s2_non_adj: f64::NAN,
                mu_bg: f64::NAN,
                delta_min: self.config.min_psr_delta,
                min_psr: self.config.min_psr_cal,
                admitted: false,
            },
            admitted: false,
            reason_codes,
            aliasing_decision: None,
        }
    }

    pub fn step(
        &mut self,
        similarities: &[f64],
        delta_ne_odom_m: [f64; 2],
        odom_sigma_m: f64,
        visual_fix_metrics: Option<&VisualFixMetrics>,
        aliasing_thresholds: Option<&VisualAliasingThresholds>,
    ) -> SequenceVprDecision {
        let frame_idx = self.frame_counter;
        self.frame_counter += 1;

        // 1. Fail-closed validation (do NOT pollute sliding window on corrupted/non-finite frame)
        if similarities.len() != self.num_tiles || !similarities.iter().all(|v| v.is_finite()) {
            return self.make_fail_closed_decision(
                frame_idx,
                vec!["NONFINITE_OR_INVALID_SIMILARITIES".to_string()],
                -1,
                Vec::new(),
            );
        }
        if !(delta_ne_odom_m[0].is_finite() && delta_ne_odom_m[1].is_finite()) {
            return self.make_fail_closed_decision(
                frame_idx,
                vec!["NONFINITE_OR_INVALID_ODOM_DELTA".to_string()],
                -1,
                Vec::new(),
            );
        }
        if !(odom_sigma_m.is_finite() && odom_sigma_m > 0.0) {
            return self.make_fail_closed_decision(
                frame_idx,
                vec!["NONFINITE_OR_NONPOSITIVE_ODOM_SIGMA".to_string()],
                -1,
                Vec::new(),
            );
        }

        let n = self.num_tiles;
        let k_top = 5.min(n);
        let raw_order = argsort_desc_stable(similarities);
        let raw_top1 = raw_order[0] as i64;
        let raw_top5: Vec<usize> = raw_order[..k_top].to_vec();

        // 1b. Fail-closed blind/zero/flat similarity guard (population std matches numpy.std)
        let mut raw_max = f64::NEG_INFINITY;
        let mut raw_sum = 0.0_f64;
        for &s in similarities {
            if s > raw_max {
                raw_max = s;
            }
            raw_sum += s;
        }
        let raw_mean = raw_sum / (n as f64);
        let mut var_sum = 0.0_f64;
        for &s in similarities {
            let d = s - raw_mean;
            var_sum += d * d;
        }
        let raw_std = (var_sum / (n as f64)).sqrt();

        let mut signal_reasons: Vec<String> = Vec::new();
        if raw_max < self.config.min_raw_sim_floor {
            signal_reasons.push("LOW_SIMILARITY_SIGNAL".to_string());
        }
        if raw_std <= self.config.min_raw_sim_std {
            signal_reasons.push("FLAT_SIMILARITY_VECTOR".to_string());
        }
        if !signal_reasons.is_empty() {
            return self.make_fail_closed_decision(frame_idx, signal_reasons, raw_top1, raw_top5);
        }

        // 2. Temperature-scaled emission likelihood & odometry transition matrices
        let inv_temp = 1.0 / self.config.emission_temperature;
        let mut scaled = vec![0.0_f64; n];
        let mut max_scaled = f64::NEG_INFINITY;
        for i in 0..n {
            let v = similarities[i] * inv_temp;
            scaled[i] = v;
            if v > max_scaled {
                max_scaled = v;
            }
        }
        let mut emit_prob = vec![0.0_f64; n];
        for i in 0..n {
            emit_prob[i] = (scaled[i] - max_scaled).clamp(-80.0, 0.0).exp();
        }

        let (log_trans, trans_prob) = if !self.window.is_empty() {
            let lt = self.compute_log_transition(delta_ne_odom_m, odom_sigma_m);
            let tp = Self::row_stochastic_from_log_trans(n, &lt);
            (Some(lt), Some(tp))
        } else {
            (None, None)
        };

        if self.window.len() == self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(WindowObservation {
            emit_prob,
            raw_sims: similarities.to_vec(),
            log_trans,
            trans_prob,
            delta_ne_odom_m,
            odom_sigma_m,
        });

        // Ensure transition matrices exist for any step_k >= 1
        let win_len = self.window.len();
        for step_k in 1..win_len {
            let needs_comp = self.window[step_k].log_trans.is_none()
                || self.window[step_k].trans_prob.is_none();
            if needs_comp {
                let d_ne = self.window[step_k].delta_ne_odom_m;
                let sig = self.window[step_k].odom_sigma_m;
                let lt = self.compute_log_transition(d_ne, sig);
                let tp = Self::row_stochastic_from_log_trans(n, &lt);
                self.window[step_k].log_trans = Some(lt);
                self.window[step_k].trans_prob = Some(tp);
            }
        }

        // 3. Exact sliding-window Bayesian Forward + Viterbi MAP inference
        let mut v_raw: Vec<f64> = self.window[0]
            .raw_sims
            .iter()
            .map(|&s| s / self.config.emission_temperature)
            .collect();
        let mut alpha = self.window[0].emit_prob.clone();
        let a0_sum = alpha.iter().sum::<f64>().max(1e-300);
        for a in &mut alpha {
            *a /= a0_sum;
        }
        let mut backpointers: Vec<Vec<usize>> = Vec::with_capacity(win_len.saturating_sub(1));

        for step_k in 1..win_len {
            let obs = &self.window[step_k];
            let lt = obs.log_trans.as_ref().expect("log_trans populated");
            let tp = obs.trans_prob.as_ref().expect("trans_prob populated");

            // Viterbi max-plus recurrence
            let mut bp = vec![0usize; n];
            let mut v_next = vec![0.0_f64; n];
            for j in 0..n {
                let mut best_i = 0usize;
                let mut best_val = v_raw[0] + lt[j];
                for i in 1..n {
                    let cand = v_raw[i] + lt[i * n + j];
                    if cand > best_val {
                        best_val = cand;
                        best_i = i;
                    }
                }
                bp[j] = best_i;
                v_next[j] = (obs.raw_sims[j] / self.config.emission_temperature) + best_val;
            }
            backpointers.push(bp);
            v_raw = v_next;

            // Exact Bayesian Forward sum-product recursion: pred[j] = sum_i alpha[i] * tp[i, j]
            let mut pred = vec![0.0_f64; n];
            for (i, &ai) in alpha.iter().enumerate() {
                let row = &tp[i * n..(i + 1) * n];
                for (p, &t) in pred.iter_mut().zip(row.iter()) {
                    *p += ai * t;
                }
            }
            let mut a_sum = 0.0_f64;
            for j in 0..n {
                let val = obs.emit_prob[j] * pred[j];
                alpha[j] = val;
                a_sum += val;
            }
            if !(a_sum.is_finite() && a_sum > 0.0) {
                return self.make_fail_closed_decision(
                    frame_idx,
                    vec!["NONFINITE_POSTERIOR".to_string()],
                    raw_top1,
                    raw_top5,
                );
            }
            for a in &mut alpha {
                *a /= a_sum;
            }
        }

        // Reconstruct exact Viterbi MAP path across the current window
        let mut best_v_end = 0usize;
        let mut max_v_raw = v_raw[0];
        for (i, &val) in v_raw.iter().enumerate().skip(1) {
            if val > max_v_raw {
                max_v_raw = val;
                best_v_end = i;
            }
        }
        let mut path_rev = Vec::with_capacity(win_len);
        path_rev.push(best_v_end);
        let mut cur = best_v_end;
        for bp in backpointers.iter().rev() {
            cur = bp[cur];
            path_rev.push(cur);
        }
        path_rev.reverse();
        let viterbi_path = path_rev;

        if !alpha.iter().all(|v| v.is_finite()) {
            return self.make_fail_closed_decision(
                frame_idx,
                vec!["NONFINITE_POSTERIOR".to_string()],
                raw_top1,
                raw_top5,
            );
        }

        // Cosine-equivalent fused trajectory score vector `(N,)`
        let scale_to_cos = self.config.emission_temperature / (win_len as f64);
        let fused_scores: Vec<f64> = match self.config.mode {
            SequenceMode::Viterbi => v_raw.iter().map(|&v| v * scale_to_cos).collect(),
            SequenceMode::Bayesian => {
                let mut log_post = vec![0.0_f64; n];
                let mut max_lp = f64::NEG_INFINITY;
                for i in 0..n {
                    let lp = alpha[i].max(1e-300).ln();
                    log_post[i] = lp;
                    if lp > max_lp {
                        max_lp = lp;
                    }
                }
                let peak_cos = max_v_raw * scale_to_cos;
                log_post
                    .iter()
                    .map(|&lp| peak_cos + (lp - max_lp) * scale_to_cos)
                    .collect()
            }
        };

        let seq_order = argsort_desc_stable(&fused_scores);
        let seq_top1 = seq_order[0];
        let seq_top5: Vec<usize> = seq_order[..k_top].to_vec();
        let final_top1 = seq_top1;
        let final_top5 = seq_top5.clone();

        // Posterior probability & non-adjacent posterior margin around final_top1
        let p_top1 = alpha[final_top1];
        let top_coord = self.gallery_ne_m[final_top1];
        let mut has_non_adj = false;
        let mut p_second_non_adj = f64::NEG_INFINITY;
        for i in 0..n {
            let d = (self.gallery_ne_m[i][0] - top_coord[0])
                .hypot(self.gallery_ne_m[i][1] - top_coord[1]);
            if d > self.config.non_adj_radius_m {
                has_non_adj = true;
                if alpha[i] > p_second_non_adj {
                    p_second_non_adj = alpha[i];
                }
            }
        }
        let p_margin_non_adj = if has_non_adj {
            p_top1 - p_second_non_adj
        } else {
            f64::NAN
        };

        // 5. Stage-4 PSR_cal + posterior margin + optional 6-stage aliasing gate
        let psr_res = compute_psr_calibrated(
            &fused_scores,
            &self.gallery_ne_m,
            self.config.non_adj_radius_m,
            None,
            self.config.min_psr_delta,
            self.config.min_psr_cal,
            1e-4,
        );

        let mut reasons: Vec<String> = Vec::new();
        if !psr_res.admitted {
            reasons.push(format!(
                "SEQ_PSR_GATE_REJECT(PSR_cal={:.3},margin={:.4})",
                psr_res.psr_cal, psr_res.margin
            ));
        }
        if !(p_top1.is_finite() && p_top1 >= self.config.min_posterior_prob) {
            reasons.push(format!(
                "SEQ_LOW_POSTERIOR(p_top1={:.3}<{:.3})",
                p_top1, self.config.min_posterior_prob
            ));
        }
        if !(p_margin_non_adj.is_finite() && p_margin_non_adj >= self.config.min_posterior_margin) {
            reasons.push(format!(
                "SEQ_LOW_POSTERIOR_MARGIN(margin={:.3}<{:.3})",
                p_margin_non_adj, self.config.min_posterior_margin
            ));
        }

        let mut aliasing_dec: Option<VisualAliasingDecision> = None;
        if let Some(vfm) = visual_fix_metrics {
            let default_thr = VisualAliasingThresholds::default();
            let thr = aliasing_thresholds.unwrap_or(&default_thr);
            let mut updated = vfm.clone();
            updated.psr = psr_res.psr_cal;
            updated.psr_margin = Some(psr_res.margin);
            let dec = evaluate_visual_aliasing_gate(&updated, thr);
            if !dec.admitted {
                reasons.extend(dec.reason_codes.iter().cloned());
            }
            aliasing_dec = Some(dec);
        }

        let admitted = reasons.is_empty();
        SequenceVprDecision {
            frame_idx,
            window_len: win_len,
            mode: self.config.mode.as_str(),
            raw_top1_idx: raw_top1,
            raw_top5_indices: raw_top5,
            seq_top1_idx: seq_top1 as i64,
            seq_top5_indices: seq_top5,
            final_top1_idx: final_top1 as i64,
            final_top5_indices: final_top5,
            viterbi_path_indices: viterbi_path,
            posterior_prob_top1: p_top1,
            posterior_margin_non_adj: p_margin_non_adj,
            fused_scores,
            psr_result: psr_res,
            admitted,
            reason_codes: reasons,
            aliasing_decision: aliasing_dec,
        }
    }
}

// ============================================================================
// Part 3: Stack-Allocated Exact Matrix Algebra (Zero External Crates)
// ============================================================================

#[inline]
fn eigvalsh_min_2x2(m: [[f64; 2]; 2]) -> f64 {
    let a = m[0][0];
    let b = 0.5 * (m[0][1] + m[1][0]);
    let d = m[1][1];
    let tr = a + d;
    let disc = (a - d).hypot(2.0 * b);
    0.5 * (tr - disc)
}

fn symmetrize_spd_2x2(m: [[f64; 2]; 2], jitter: f64) -> [[f64; 2]; 2] {
    let a = m[0][0];
    let b = 0.5 * (m[0][1] + m[1][0]);
    let d = m[1][1];
    let safe_j = if jitter.is_finite() && jitter > 0.0 {
        jitter
    } else {
        1e-9
    };
    if !(a.is_finite() && b.is_finite() && d.is_finite()) {
        return [[safe_j, 0.0], [0.0, safe_j]];
    }
    if b.abs() <= 1e-15 * (a.abs() + d.abs() + 1.0) {
        return [[a.max(safe_j), 0.0], [0.0, d.max(safe_j)]];
    }
    let tr = a + d;
    let diff = a - d;
    let disc = diff.hypot(2.0 * b);
    let lam1 = (0.5 * (tr + disc)).max(safe_j);
    let lam2 = (0.5 * (tr - disc)).max(safe_j);
    let theta = 0.5 * (2.0 * b).atan2(diff);
    let c = theta.cos();
    let s = theta.sin();
    let r00 = c * c * lam1 + s * s * lam2;
    let r01 = c * s * (lam1 - lam2);
    let r11 = s * s * lam1 + c * c * lam2;
    [[r00, r01], [r01, r11]]
}

/// Exact Cyclic Jacobi symmetric eigendecomposition + PSD eigenvalue floor `>= jitter`.
fn symmetrize_spd_nxn<const N: usize>(m: [[f64; N]; N], jitter: f64) -> [[f64; N]; N] {
    let mut a = [[0.0_f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            a[i][j] = 0.5 * (m[i][j] + m[j][i]);
        }
    }
    let mut v = [[0.0_f64; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for _sweep in 0..64 {
        let mut max_off = 0.0_f64;
        let mut diag_scale = 0.0_f64;
        for (i, row) in a.iter().enumerate() {
            diag_scale = diag_scale.max(row[i].abs());
            for &val in row.iter().skip(i + 1) {
                max_off = max_off.max(val.abs());
            }
        }
        if max_off <= 1e-15 * diag_scale.max(1e-30) {
            break;
        }
        for p in 0..N {
            for q in (p + 1)..N {
                let apq = a[p][q];
                let scale = a[p][p].abs() + a[q][q].abs() + 1e-30;
                if apq.abs() <= 1e-16 * scale {
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + tau.hypot(1.0))
                } else {
                    -1.0 / (-tau + tau.hypot(1.0))
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for k in 0..N {
                    if k != p && k != q {
                        let akp = a[k][p];
                        let akq = a[k][q];
                        let new_kp = c * akp - s * akq;
                        let new_kq = s * akp + c * akq;
                        a[k][p] = new_kp;
                        a[p][k] = new_kp;
                        a[k][q] = new_kq;
                        a[q][k] = new_kq;
                    }
                }
                for row in &mut v {
                    let vkp = row[p];
                    let vkq = row[q];
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut d = [0.0_f64; N];
    for i in 0..N {
        d[i] = a[i][i].max(jitter);
    }

    let mut out = [[0.0_f64; N]; N];
    for i in 0..N {
        for j in i..N {
            let mut sum = 0.0_f64;
            for k in 0..N {
                sum += v[i][k] * d[k] * v[j][k];
            }
            out[i][j] = sum;
            out[j][i] = sum;
        }
    }
    out
}

#[inline]
fn cholesky_2x2(c: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let l00 = c[0][0].max(1e-18).sqrt();
    let l10 = c[1][0] / l00;
    let l11 = (c[1][1] - l10 * l10).max(1e-18).sqrt();
    [[l00, 0.0], [l10, l11]]
}

#[inline]
fn solve_lower_2x2(l: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    let z0 = v[0] / l[0][0];
    let z1 = (v[1] - l[1][0] * z0) / l[1][1];
    [z0, z1]
}

#[inline]
fn inv_2x2(s: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
    let inv_det = 1.0 / det;
    [
        [s[1][1] * inv_det, -s[0][1] * inv_det],
        [-s[1][0] * inv_det, s[0][0] * inv_det],
    ]
}

#[inline]
fn solve_2x2(s: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    let inv = inv_2x2(s);
    [
        inv[0][0] * v[0] + inv[0][1] * v[1],
        inv[1][0] * v[0] + inv[1][1] * v[1],
    ]
}

fn matmul<const R: usize, const K: usize, const C: usize>(
    a: &[[f64; K]; R],
    b: &[[f64; C]; K],
) -> [[f64; C]; R] {
    let mut out = [[0.0_f64; C]; R];
    for i in 0..R {
        for k in 0..K {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..C {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn matmul_t<const R: usize, const K: usize, const C: usize>(
    a: &[[f64; K]; R],
    b: &[[f64; K]; C],
) -> [[f64; C]; R] {
    let mut out = [[0.0_f64; C]; R];
    for i in 0..R {
        for j in 0..C {
            let mut acc = 0.0_f64;
            for k in 0..K {
                acc += a[i][k] * b[j][k];
            }
            out[i][j] = acc;
        }
    }
    out
}

fn matvec<const R: usize, const C: usize>(a: &[[f64; C]; R], x: &[f64; C]) -> [f64; R] {
    let mut out = [0.0_f64; R];
    for i in 0..R {
        let mut acc = 0.0_f64;
        for j in 0..C {
            acc += a[i][j] * x[j];
        }
        out[i] = acc;
    }
    out
}

fn matadd<const R: usize, const C: usize>(
    a: &[[f64; C]; R],
    b: &[[f64; C]; R],
) -> [[f64; C]; R] {
    let mut out = [[0.0_f64; C]; R];
    for i in 0..R {
        for j in 0..C {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

fn eye<const N: usize>() -> [[f64; N]; N] {
    let mut out = [[0.0_f64; N]; N];
    for (i, row) in out.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    out
}

// ============================================================================
// Part 4: Pure-Rust DualBankEsEkfCusum & EightArmCholeskyCusum
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DualBankConfig {
    pub sample_rate_hz: f64,
    pub sigma_ins_accel_mps2: f64,
    pub sigma_gnss_ne_m: f64,
    pub sigma_vis_ne_m: f64,
    pub sigma_baro_d_m: f64,
    pub k_d: f64,
    pub h_threshold: f64,
    pub reintegration_hysteresis_epochs: usize,
    pub reintegration_stat_frac: f64,
    pub max_q_vis: f64,
    pub hal_nav_m: f64,
    pub sigma_mp_ne_m: f64,
    pub sigma_mp_ne_floor_m: f64,
    pub tau_mp_ne_s: f64,
    pub sigma_gnss_obs_white_m: f64,
    pub sigma_vis_obs_white_m: f64,
    pub persist_window_s: usize,
    pub persist_min_samples: usize,
    pub persist_scatter_scale: f64,
    pub sigma_gnss_d_m: f64,
    pub sigma_mp_d_m: f64,
    pub tau_mp_d_s: f64,
    pub vertical_k_d: f64,
    pub vertical_h_threshold: f64,
}

impl Default for DualBankConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 1.0,
            sigma_ins_accel_mps2: 0.35,
            sigma_gnss_ne_m: 1.5,
            sigma_vis_ne_m: 2.8,
            sigma_baro_d_m: 1.2,
            k_d: CUSUM_K_D_LOCKED,
            h_threshold: CUSUM_H_LOCKED,
            reintegration_hysteresis_epochs: CUSUM_REINTEGRATION_HYSTERESIS_EPOCHS,
            reintegration_stat_frac: CUSUM_REINTEGRATION_STAT_FRAC,
            max_q_vis: MAX_Q_VIS_LOCKED,
            hal_nav_m: DEFAULT_HAL_NAV_M,
            sigma_mp_ne_m: 2.60,
            sigma_mp_ne_floor_m: 0.0,
            tau_mp_ne_s: 16.0,
            sigma_gnss_obs_white_m: 1.50,
            sigma_vis_obs_white_m: 2.80,
            persist_window_s: 60,
            persist_min_samples: 52,
            persist_scatter_scale: 3.40,
            sigma_gnss_d_m: 2.5,
            sigma_mp_d_m: 3.5,
            tau_mp_d_s: 12.0,
            vertical_k_d: CUSUM_K_D_LOCKED,
            vertical_h_threshold: CUSUM_H_LOCKED,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CusumStepOutput {
    pub epoch_idx: usize,
    pub timestamp_s: f64,
    pub active_bank: &'static str,
    pub gnss_quarantined: bool,
    pub quarantine_triggered_this_epoch: bool,
    pub quarantine_latency_s: Option<f64>,
    pub reintegrated_this_epoch: bool,
    pub active_p_ned_m: [f64; 3],
    pub bank_a_p_ne_m: [f64; 2],
    pub bank_b_p_ne_m: [f64; 2],
    pub d_ne_m: [f64; 2],
    pub z_whitened: [f64; 2],
    pub arm_stats: [f64; CUSUM_NUM_ARMS],
    pub max_cusum_stat: f64,
    pub dominant_arm: usize,
    pub q_vis: f64,
    pub nominal_streak: usize,
    pub p_diff_ne: [[f64; 2]; 2],
    pub vertical_cusum_stat: f64,
    pub vertical_spoof_alarm: bool,
    pub spoof_alarm: bool,
}

#[derive(Debug, Clone)]
pub struct EightArmCholeskyCusum {
    pub k_d: f64,
    pub h_threshold: f64,
    pub arm_stats: [f64; CUSUM_NUM_ARMS],
    arm_unit_vectors: [[f64; 2]; CUSUM_NUM_ARMS],
}

impl EightArmCholeskyCusum {
    pub fn new(k_d: f64, h_threshold: f64) -> Result<Self, String> {
        if !(k_d.is_finite() && k_d > 0.0 && h_threshold.is_finite() && h_threshold > 0.0) {
            return Err(format!(
                "Invalid CUSUM parameters: k_d={k_d}, h={h_threshold}"
            ));
        }
        Ok(Self {
            k_d,
            h_threshold,
            arm_stats: [0.0; CUSUM_NUM_ARMS],
            arm_unit_vectors: cusum_arm_unit_vectors(),
        })
    }

    pub fn reset(&mut self) {
        self.arm_stats = [0.0; CUSUM_NUM_ARMS];
    }

    pub fn update(
        &mut self,
        vec_ne_m: [f64; 2],
        cov_ne: [[f64; 2]; 2],
        dt_s: f64,
    ) -> ([f64; 2], f64, usize, bool) {
        let v_fin = vec_ne_m[0].is_finite() && vec_ne_m[1].is_finite();
        let c_fin = cov_ne[0][0].is_finite()
            && cov_ne[0][1].is_finite()
            && cov_ne[1][0].is_finite()
            && cov_ne[1][1].is_finite();
        if !(v_fin && c_fin && dt_s.is_finite() && dt_s > 0.0) {
            return ([f64::NAN, f64::NAN], f64::INFINITY, 0, true);
        }

        let spd = symmetrize_spd_2x2(cov_ne, 1e-9);
        let l_mat = cholesky_2x2(spd);
        let z = solve_lower_2x2(l_mat, vec_ne_m);
        let sqrt_dt = dt_s.sqrt();
        let drift = self.k_d * dt_s;

        let mut dom = 0usize;
        let mut max_s = -1.0_f64;
        for m in 0..CUSUM_NUM_ARMS {
            let u = self.arm_unit_vectors[m];
            let y_m = u[0] * z[0] + u[1] * z[1];
            let next_s = (self.arm_stats[m] + sqrt_dt * y_m - drift).max(0.0);
            self.arm_stats[m] = next_s;
            if next_s > max_s {
                max_s = next_s;
                dom = m;
            }
        }
        let alarm = max_s >= self.h_threshold;
        (z, max_s, dom, alarm)
    }
}

#[derive(Debug, Clone)]
pub struct DualBankStepInput {
    pub ins_accel_ne_mps2: [f64; 2],
    pub baro_down_m: f64,
    pub gnss_ne_m: Option<[f64; 2]>,
    pub vis_ne_m: Option<[f64; 2]>,
    pub vis_gate_admitted: bool,
    pub r_gnss: Option<[[f64; 2]; 2]>,
    pub r_vis: Option<[[f64; 2]; 2]>,
    pub gnss_down_m: Option<f64>,
}

impl Default for DualBankStepInput {
    fn default() -> Self {
        Self {
            ins_accel_ne_mps2: [0.0, 0.0],
            baro_down_m: 0.0,
            gnss_ne_m: None,
            vis_ne_m: None,
            vis_gate_admitted: true,
            r_gnss: None,
            r_vis: None,
            gnss_down_m: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DualBankEsEkfCusum {
    pub cfg: DualBankConfig,
    pub dt: f64,
    f_mat: [[f64; 4]; 4],
    h_mat: [[f64; 4]; 2],
    i4: [[f64; 4]; 4],
    i10: [[f64; 10]; 10],
    q_ins: [[f64; 4]; 4],
    r_mult: f64,
    r_gnss_default: [[f64; 2]; 2],
    r_vis_default: [[f64; 2]; 2],
    r_gnss_obs_default: [[f64; 2]; 2],
    r_vis_obs_default: [[f64; 2]; 2],

    sigma_mp_ne_ceil: f64,
    sigma_mp_ne_floor: f64,
    sigma_mp_ne: f64,
    phi_mp_ne: f64,
    q_mp_ne: f64,
    c_xi: [[f64; 10]; 2],

    sigma_mp_d: f64,
    phi_mp_d: f64,
    q_mp_d: f64,
    r_vert_obs: f64,

    pub cusum: EightArmCholeskyCusum,
    arm_unit_vectors: [[f64; 2]; CUSUM_NUM_ARMS],

    pub x_a: [f64; 4],
    pub x_b: [f64; 4],
    pub baro_down_m: f64,
    pub p_a: [[f64; 4]; 4],
    pub p_b: [[f64; 4]; 4],
    pub p_ab: [[f64; 4]; 4],

    xi_hat: [f64; 10],
    p_xi: [[f64; 10]; 10],

    sec_d_accum: [f64; 2],
    sec_d_cnt: usize,
    sec_elapsed_s: f64,
    d_1hz_window: VecDeque<[f64; 2]>,
    d_1hz_window_cap: usize,
    persist_arms: [f64; CUSUM_NUM_ARMS],

    vert_b_hat: f64,
    vert_p: f64,
    vert_s_pos: f64,
    vert_s_neg: f64,

    pub gnss_quarantined: bool,
    pub nominal_streak: usize,
    pub epoch_idx: usize,
    pub timestamp_s: f64,
}

impl DualBankEsEkfCusum {
    pub fn new(cfg: DualBankConfig) -> Result<Self, String> {
        if !(cfg.sample_rate_hz.is_finite() && cfg.sample_rate_hz > 0.0) {
            return Err(format!("Invalid sample_rate_hz: {}", cfg.sample_rate_hz));
        }
        let dt = 1.0 / cfg.sample_rate_hz;
        let f_mat = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let h_mat = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let i4 = eye::<4>();
        let i10 = eye::<10>();

        let q_a = cfg.sigma_ins_accel_mps2 * cfg.sigma_ins_accel_mps2;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let q_ins = [
            [q_a * dt3 / 3.0, 0.0, q_a * dt2 / 2.0, 0.0],
            [0.0, q_a * dt3 / 3.0, 0.0, q_a * dt2 / 2.0],
            [q_a * dt2 / 2.0, 0.0, q_a * dt, 0.0],
            [0.0, q_a * dt2 / 2.0, 0.0, q_a * dt],
        ];

        let r_mult = cfg.sample_rate_hz.max(1.0);
        let r_g = (cfg.sigma_gnss_ne_m * cfg.sigma_gnss_ne_m) * r_mult;
        let r_v = (cfg.sigma_vis_ne_m * cfg.sigma_vis_ne_m) * r_mult;
        let r_go = (cfg.sigma_gnss_obs_white_m * cfg.sigma_gnss_obs_white_m) * r_mult;
        let r_vo = (cfg.sigma_vis_obs_white_m * cfg.sigma_vis_obs_white_m) * r_mult;

        let sigma_mp_ne_ceil = cfg.sigma_mp_ne_m.max(0.0);
        let sigma_mp_ne_floor = cfg.sigma_mp_ne_floor_m.max(0.0).min(sigma_mp_ne_ceil);
        let phi_mp_ne = if sigma_mp_ne_ceil > 0.0 && cfg.tau_mp_ne_s > 0.0 {
            (-dt / cfg.tau_mp_ne_s).exp()
        } else {
            0.0
        };
        let q_mp_ne = (sigma_mp_ne_ceil * sigma_mp_ne_ceil) * (1.0 - phi_mp_ne * phi_mp_ne).max(0.0);

        let mut c_xi = [[0.0_f64; 10]; 2];
        c_xi[0][0] = 1.0;
        c_xi[1][1] = 1.0;
        c_xi[0][4] = -1.0;
        c_xi[1][5] = -1.0;

        let sigma_mp_d = cfg.sigma_mp_d_m.max(0.0);
        let tau_mp_d = cfg.tau_mp_d_s.max(1e-3);
        let phi_mp_d = if sigma_mp_d > 0.0 {
            (-dt / tau_mp_d).exp()
        } else {
            0.0
        };
        let q_mp_d = (sigma_mp_d * sigma_mp_d) * (1.0 - phi_mp_d * phi_mp_d).max(0.0);
        let r_vert_obs = (cfg.sigma_gnss_d_m * cfg.sigma_gnss_d_m
            + cfg.sigma_baro_d_m * cfg.sigma_baro_d_m)
            * r_mult;

        let cusum = EightArmCholeskyCusum::new(cfg.k_d, cfg.h_threshold)?;
        let d_1hz_window_cap = cfg.persist_window_s.max(10);

        let mut inst = Self {
            cfg,
            dt,
            f_mat,
            h_mat,
            i4,
            i10,
            q_ins,
            r_mult,
            r_gnss_default: [[r_g, 0.0], [0.0, r_g]],
            r_vis_default: [[r_v, 0.0], [0.0, r_v]],
            r_gnss_obs_default: [[r_go, 0.0], [0.0, r_go]],
            r_vis_obs_default: [[r_vo, 0.0], [0.0, r_vo]],
            sigma_mp_ne_ceil,
            sigma_mp_ne_floor,
            sigma_mp_ne: sigma_mp_ne_ceil,
            phi_mp_ne,
            q_mp_ne,
            c_xi,
            sigma_mp_d,
            phi_mp_d,
            q_mp_d,
            r_vert_obs,
            cusum,
            arm_unit_vectors: cusum_arm_unit_vectors(),
            x_a: [0.0; 4],
            x_b: [0.0; 4],
            baro_down_m: 0.0,
            p_a: [[0.0; 4]; 4],
            p_b: [[0.0; 4]; 4],
            p_ab: [[0.0; 4]; 4],
            xi_hat: [0.0; 10],
            p_xi: [[0.0; 10]; 10],
            sec_d_accum: [0.0; 2],
            sec_d_cnt: 0,
            sec_elapsed_s: 0.0,
            d_1hz_window: VecDeque::with_capacity(d_1hz_window_cap),
            d_1hz_window_cap,
            persist_arms: [0.0; CUSUM_NUM_ARMS],
            vert_b_hat: 0.0,
            vert_p: sigma_mp_d * sigma_mp_d,
            vert_s_pos: 0.0,
            vert_s_neg: 0.0,
            gnss_quarantined: false,
            nominal_streak: 0,
            epoch_idx: 0,
            timestamp_s: 0.0,
        };
        inst.reset([0.0, 0.0, 0.0], [0.0, 0.0]);
        Ok(inst)
    }

    pub fn reset(&mut self, init_ned_m: [f64; 3], init_vel_ne_mps: [f64; 2]) {
        let state0 = [
            init_ned_m[0],
            init_ned_m[1],
            init_vel_ne_mps[0],
            init_vel_ne_mps[1],
        ];
        self.x_a = state0;
        self.x_b = state0;
        self.baro_down_m = init_ned_m[2];

        self.p_a = [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.p_b = [
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.p_ab = [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        self.sigma_mp_ne = self.sigma_mp_ne_ceil;
        self.q_mp_ne = (self.sigma_mp_ne * self.sigma_mp_ne)
            * (1.0 - self.phi_mp_ne * self.phi_mp_ne).max(0.0);
        self.xi_hat = [0.0; 10];
        self.p_xi = [[0.0; 10]; 10];
        for i in 0..4 {
            for j in 0..4 {
                self.p_xi[i][j] = self.p_a[i][j];
                self.p_xi[4 + i][4 + j] = self.p_b[i][j];
                self.p_xi[i][4 + j] = self.p_ab[i][j];
                self.p_xi[4 + i][j] = self.p_ab[j][i];
            }
        }
        let mp_var = self.sigma_mp_ne * self.sigma_mp_ne;
        self.p_xi[8][8] = mp_var;
        self.p_xi[9][9] = mp_var;

        self.sec_d_accum = [0.0; 2];
        self.sec_d_cnt = 0;
        self.sec_elapsed_s = 0.0;
        self.d_1hz_window.clear();
        self.persist_arms = [0.0; CUSUM_NUM_ARMS];

        self.vert_b_hat = 0.0;
        self.vert_p = self.sigma_mp_d * self.sigma_mp_d;
        self.vert_s_pos = 0.0;
        self.vert_s_neg = 0.0;

        self.cusum.reset();
        self.gnss_quarantined = false;
        self.nominal_streak = 0;
        self.epoch_idx = 0;
        self.timestamp_s = 0.0;
    }

    fn reset_detectors_post_reintegration(&mut self) {
        self.cusum.reset();
        self.persist_arms = [0.0; CUSUM_NUM_ARMS];
        self.d_1hz_window.clear();
        self.sec_d_accum = [0.0; 2];
        self.sec_d_cnt = 0;
        self.sec_elapsed_s = 0.0;
        self.xi_hat = [0.0; 10];
        self.vert_b_hat = 0.0;
        self.vert_p = self.sigma_mp_d * self.sigma_mp_d;
        self.vert_s_pos = 0.0;
        self.vert_s_neg = 0.0;
    }

    fn step_vertical_cusum(&mut self, gnss_down_m: Option<f64>) -> (f64, bool) {
        let Some(gd) = gnss_down_m else {
            return (0.0, false);
        };
        if !(gd.is_finite() && self.baro_down_m.is_finite()) {
            return (f64::INFINITY, true);
        }

        let d_down = gd - self.baro_down_m;
        let b_pred = self.phi_mp_d * self.vert_b_hat;
        let p_pred = (self.phi_mp_d * self.phi_mp_d) * self.vert_p + self.q_mp_d;
        let v_d = d_down - b_pred;
        let s_d = (p_pred + self.r_vert_obs).max(1e-6);
        let k_d_gain = p_pred / s_d;
        self.vert_b_hat = b_pred + k_d_gain * v_d;
        self.vert_p = ((1.0 - k_d_gain) * p_pred).max(1e-9);

        let z_d = v_d / s_d.sqrt();
        let sqrt_dt = self.dt.sqrt();
        let k_v = self.cfg.vertical_k_d;
        self.vert_s_pos = (self.vert_s_pos + sqrt_dt * z_d - k_v * self.dt).max(0.0);
        self.vert_s_neg = (self.vert_s_neg + sqrt_dt * (-z_d) - k_v * self.dt).max(0.0);
        let stat = self.vert_s_pos.max(self.vert_s_neg);
        (stat, stat >= self.cfg.vertical_h_threshold)
    }

    pub fn step(&mut self, input: &DualBankStepInput) -> CusumStepOutput {
        self.epoch_idx += 1;
        self.timestamp_s += self.dt;
        if input.baro_down_m.is_finite() {
            self.baro_down_m = input.baro_down_m;
        }

        // 1. Shared INS propagation for both Bank-A and Bank-B
        let a_ne = if input.ins_accel_ne_mps2[0].is_finite()
            && input.ins_accel_ne_mps2[1].is_finite()
        {
            input.ins_accel_ne_mps2
        } else {
            [0.0, 0.0]
        };
        let b_ctrl = [
            0.5 * self.dt * self.dt * a_ne[0],
            0.5 * self.dt * self.dt * a_ne[1],
            self.dt * a_ne[0],
            self.dt * a_ne[1],
        ];
        let fx_a = matvec(&self.f_mat, &self.x_a);
        let fx_b = matvec(&self.f_mat, &self.x_b);
        for i in 0..4 {
            self.x_a[i] = fx_a[i] + b_ctrl[i];
            self.x_b[i] = fx_b[i] + b_ctrl[i];
        }

        let fp_a = matmul(&self.f_mat, &self.p_a);
        self.p_a = symmetrize_spd_nxn(
            matadd(&matmul_t(&fp_a, &self.f_mat), &self.q_ins),
            1e-9,
        );
        let fp_b = matmul(&self.f_mat, &self.p_b);
        self.p_b = symmetrize_spd_nxn(
            matadd(&matmul_t(&fp_b, &self.f_mat), &self.q_ins),
            1e-9,
        );
        let fp_ab = matmul(&self.f_mat, &self.p_ab);
        self.p_ab = matadd(&matmul_t(&fp_ab, &self.f_mat), &self.q_ins);

        // 2. Bank-A GNSS update (runs in shadow monitor mode even during quarantine)
        let mut i_a = self.i4;
        let mut k_a = [[0.0_f64; 2]; 4];
        let mut gnss_used = false;
        let mut rg_obs = self.r_gnss_obs_default;
        if let Some(yg) = input.gnss_ne_m {
            if yg[0].is_finite() && yg[1].is_finite() {
                let rg_pair = match input.r_gnss {
                    None => Some((self.r_gnss_default, self.r_gnss_obs_default)),
                    Some(raw_rg) => {
                        if raw_rg.iter().flatten().all(|v| v.is_finite()) {
                            let scaled = [
                                [raw_rg[0][0] * self.r_mult, raw_rg[0][1] * self.r_mult],
                                [raw_rg[1][0] * self.r_mult, raw_rg[1][1] * self.r_mult],
                            ];
                            let ratio = self.cfg.sigma_gnss_obs_white_m
                                / self.cfg.sigma_gnss_ne_m.max(1e-6);
                            let r2 = ratio * ratio;
                            let obs = [
                                [scaled[0][0] * r2, scaled[0][1] * r2],
                                [scaled[1][0] * r2, scaled[1][1] * r2],
                            ];
                            if scaled.iter().flatten().all(|v| v.is_finite())
                                && obs.iter().flatten().all(|v| v.is_finite())
                            {
                                Some((scaled, obs))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                };
                if let Some((rg, obs_mat)) = rg_pair {
                    rg_obs = obs_mat;
                    let hp_a = matmul(&self.h_mat, &self.p_a);
                    let s_a = symmetrize_spd_2x2(
                        matadd(&matmul_t(&hp_a, &self.h_mat), &rg),
                        1e-9,
                    );
                    let p_ht = matmul_t(&self.p_a, &self.h_mat);
                    k_a = matmul(&p_ht, &inv_2x2(s_a));
                    let kh = matmul(&k_a, &self.h_mat);
                    for r in 0..4 {
                        for c in 0..4 {
                            i_a[r][c] = self.i4[r][c] - kh[r][c];
                        }
                    }
                    let hx_a = matvec(&self.h_mat, &self.x_a);
                    let innov_g = [yg[0] - hx_a[0], yg[1] - hx_a[1]];
                    let dx_a = matvec(&k_a, &innov_g);
                    for r in 0..4 {
                        self.x_a[r] += dx_a[r];
                    }
                    let ia_pa = matmul(&i_a, &self.p_a);
                    let ia_pa_iat = matmul_t(&ia_pa, &i_a);
                    let ka_rg = matmul(&k_a, &rg);
                    let ka_rg_kat = matmul_t(&ka_rg, &k_a);
                    self.p_a = symmetrize_spd_nxn(matadd(&ia_pa_iat, &ka_rg_kat), 1e-9);
                    self.p_ab = matmul(&i_a, &self.p_ab);
                    gnss_used = true;
                }
            }
        }

        // 3. Bank-B Gate-Verified Visual update (STRICTLY immune to GNSS spoofing)
        let mut i_b = self.i4;
        let mut k_b = [[0.0_f64; 2]; 4];
        let mut q_vis = 0.0_f64;
        let mut vis_used = false;
        let mut rv_obs = self.r_vis_obs_default;
        if input.vis_gate_admitted {
            if let Some(yv) = input.vis_ne_m {
                if yv[0].is_finite() && yv[1].is_finite() {
                    let rv_pair = match input.r_vis {
                        None => Some((self.r_vis_default, self.r_vis_obs_default)),
                        Some(raw_rv) => {
                            if raw_rv.iter().flatten().all(|v| v.is_finite()) {
                                let scaled = [
                                    [raw_rv[0][0] * self.r_mult, raw_rv[0][1] * self.r_mult],
                                    [raw_rv[1][0] * self.r_mult, raw_rv[1][1] * self.r_mult],
                                ];
                                let ratio = self.cfg.sigma_vis_obs_white_m
                                    / self.cfg.sigma_vis_ne_m.max(1e-6);
                                let r2 = ratio * ratio;
                                let obs = [
                                    [scaled[0][0] * r2, scaled[0][1] * r2],
                                    [scaled[1][0] * r2, scaled[1][1] * r2],
                                ];
                                if scaled.iter().flatten().all(|v| v.is_finite())
                                    && obs.iter().flatten().all(|v| v.is_finite())
                                {
                                    Some((scaled, obs))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                    };
                    if let Some((rv, obs_mat)) = rv_pair {
                        rv_obs = obs_mat;
                        let hp_b = matmul(&self.h_mat, &self.p_b);
                        let s_b = symmetrize_spd_2x2(
                            matadd(&matmul_t(&hp_b, &self.h_mat), &rv),
                            1e-9,
                        );
                        let hx_b = matvec(&self.h_mat, &self.x_b);
                        let innov_v = [yv[0] - hx_b[0], yv[1] - hx_b[1]];
                        let sol_v = solve_2x2(s_b, innov_v);
                        q_vis = innov_v[0] * sol_v[0] + innov_v[1] * sol_v[1];
                        if q_vis.is_finite() && q_vis <= self.cfg.max_q_vis {
                            let p_ht = matmul_t(&self.p_b, &self.h_mat);
                            k_b = matmul(&p_ht, &inv_2x2(s_b));
                            let kh = matmul(&k_b, &self.h_mat);
                            for r in 0..4 {
                                for c in 0..4 {
                                    i_b[r][c] = self.i4[r][c] - kh[r][c];
                                }
                            }
                            let dx_b = matvec(&k_b, &innov_v);
                            for r in 0..4 {
                                self.x_b[r] += dx_b[r];
                            }
                            let ib_pb = matmul(&i_b, &self.p_b);
                            let ib_pb_ibt = matmul_t(&ib_pb, &i_b);
                            let kb_rv = matmul(&k_b, &rv);
                            let kb_rv_kbt = matmul_t(&kb_rv, &k_b);
                            self.p_b = symmetrize_spd_nxn(matadd(&ib_pb_ibt, &kb_rv_kbt), 1e-9);
                            self.p_ab = matmul_t(&self.p_ab, &i_b);
                            vis_used = true;
                        }
                    }
                }
            }
        }

        // 4. Exact Bank-A vs Bank-B separation & exact difference covariance P_diff = P_A + P_B - P_AB - P_AB^T
        let d_k = [self.x_a[0] - self.x_b[0], self.x_a[1] - self.x_b[1]];
        let mut p_diff_raw = [[0.0_f64; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                p_diff_raw[r][c] =
                    self.p_a[r][c] + self.p_b[r][c] - self.p_ab[r][c] - self.p_ab[c][r];
            }
        }
        let p_diff_cur = symmetrize_spd_2x2(p_diff_raw, 1e-6);

        // 5. Exact 10D Gauss-Markov H0 innovations whitening (`xi = [e_A(4), e_B(4), b_mp(2)]`)
        let mut a_xi = [[0.0_f64; 10]; 10];
        let ia_f = matmul(&i_a, &self.f_mat);
        let ib_f = matmul(&i_b, &self.f_mat);
        for r in 0..4 {
            for c in 0..4 {
                a_xi[r][c] = ia_f[r][c];
                a_xi[4 + r][4 + c] = ib_f[r][c];
            }
        }
        a_xi[8][8] = self.phi_mp_ne;
        a_xi[9][9] = self.phi_mp_ne;
        if gnss_used {
            for r in 0..4 {
                for c in 0..2 {
                    a_xi[r][8 + c] = k_a[r][c] * self.phi_mp_ne;
                }
            }
        }

        let mut q_xi = [[0.0_f64; 10]; 10];
        let ia_q = matmul(&i_a, &self.q_ins);
        let q_aa = matmul_t(&ia_q, &i_a);
        let ib_q = matmul(&i_b, &self.q_ins);
        let q_bb = matmul_t(&ib_q, &i_b);
        let q_ab_ins = matmul_t(&ia_q, &i_b);
        for r in 0..4 {
            for c in 0..4 {
                q_xi[r][c] += q_aa[r][c];
                q_xi[4 + r][4 + c] += q_bb[r][c];
                q_xi[r][4 + c] += q_ab_ins[r][c];
                q_xi[4 + r][c] += q_ab_ins[c][r];
            }
        }
        let q_eta = [[self.q_mp_ne, 0.0], [0.0, self.q_mp_ne]];
        q_xi[8][8] += self.q_mp_ne;
        q_xi[9][9] += self.q_mp_ne;
        if gnss_used {
            let rg_plus_eta = matadd(&rg_obs, &q_eta);
            let ka_rge = matmul(&k_a, &rg_plus_eta);
            let ka_rge_kat = matmul_t(&ka_rge, &k_a);
            for r in 0..4 {
                for c in 0..4 {
                    q_xi[r][c] += ka_rge_kat[r][c];
                }
                for c in 0..2 {
                    let v = k_a[r][c] * self.q_mp_ne;
                    q_xi[r][8 + c] += v;
                    q_xi[8 + c][r] += v;
                }
            }
        }
        if vis_used {
            let kb_rv = matmul(&k_b, &rv_obs);
            let kb_rv_kbt = matmul_t(&kb_rv, &k_b);
            for r in 0..4 {
                for c in 0..4 {
                    q_xi[4 + r][4 + c] += kb_rv_kbt[r][c];
                }
            }
        }

        let xi_pred = matvec(&a_xi, &self.xi_hat);
        let a_pxi = matmul(&a_xi, &self.p_xi);
        let p_xi_pred = symmetrize_spd_nxn(
            matadd(&matmul_t(&a_pxi, &a_xi), &q_xi),
            1e-9,
        );
        let c_xi_pred = matvec(&self.c_xi, &xi_pred);
        let v_k = [d_k[0] - c_xi_pred[0], d_k[1] - c_xi_pred[1]];
        let c_p = matmul(&self.c_xi, &p_xi_pred);
        let s_innov = symmetrize_spd_2x2(matmul_t(&c_p, &self.c_xi), 1e-5);

        let p_ct = matmul_t(&p_xi_pred, &self.c_xi);
        let k_xi = matmul(&p_ct, &inv_2x2(s_innov));
        let k_v = matvec(&k_xi, &v_k);
        for i in 0..10 {
            self.xi_hat[i] = xi_pred[i] + k_v[i];
        }
        let kc = matmul(&k_xi, &self.c_xi);
        let mut i_kc = [[0.0_f64; 10]; 10];
        for r in 0..10 {
            for c in 0..10 {
                i_kc[r][c] = self.i10[r][c] - kc[r][c];
            }
        }
        let ikc_p = matmul(&i_kc, &p_xi_pred);
        self.p_xi = symmetrize_spd_nxn(matmul_t(&ikc_p, &i_kc), 1e-9);

        let mut quarantine_triggered = false;
        let mut quarantine_latency_s: Option<f64> = None;
        let mut reintegrated = false;

        let (z_k, combined_arms, dom_arm, max_s, horiz_alarm) = if gnss_used || vis_used {
            let (z_k, innov_max_s, _innov_dom, innov_alarm) =
                self.cusum.update(v_k, s_innov, self.dt);

            // 1 Hz downsampled persistent constant-step monitor & spoof-invariant orthogonal multipath estimator
            self.sec_d_accum[0] += d_k[0];
            self.sec_d_accum[1] += d_k[1];
            self.sec_d_cnt += 1;
            self.sec_elapsed_s += self.dt;
            if self.sec_elapsed_s + 1e-9 >= 1.0 {
                let cnt_f = self.sec_d_cnt as f64;
                let d_1s = [self.sec_d_accum[0] / cnt_f, self.sec_d_accum[1] / cnt_f];
                self.sec_d_accum = [0.0, 0.0];
                self.sec_d_cnt = 0;
                self.sec_elapsed_s = 0.0;
                if self.d_1hz_window.len() == self.d_1hz_window_cap {
                    self.d_1hz_window.pop_front();
                }
                self.d_1hz_window.push_back(d_1s);

                let win_len = self.d_1hz_window.len();
                let win_f = win_len as f64;
                if win_len >= 8 && self.sigma_mp_ne_ceil > self.sigma_mp_ne_floor {
                    let mut m00 = 0.0_f64;
                    let mut m01 = 0.0_f64;
                    let mut m11 = 0.0_f64;
                    for p in &self.d_1hz_window {
                        m00 += p[0] * p[0];
                        m01 += p[0] * p[1];
                        m11 += p[1] * p[1];
                    }
                    let m2 = [[m00 / win_f, m01 / win_f], [m01 / win_f, m11 / win_f]];
                    let lam_min = eigvalsh_min_2x2(m2);
                    let p_white_axis = 0.5 * (p_diff_cur[0][0] + p_diff_cur[1][1]);
                    let base_v =
                        (if self.cfg.sample_rate_hz <= 1.5 { 0.88 } else { 0.35 }) * p_white_axis;
                    let excess = (lam_min - base_v).max(0.0);
                    let target_mp = self.sigma_mp_ne_ceil.min(
                        self.sigma_mp_ne_floor
                            .max(self.sigma_mp_ne_floor + 1.95 * excess.sqrt()),
                    );
                    let alpha_mp = if target_mp > self.sigma_mp_ne { 0.50 } else { 0.40 };
                    self.sigma_mp_ne = (1.0 - alpha_mp) * self.sigma_mp_ne + alpha_mp * target_mp;
                    self.q_mp_ne = (self.sigma_mp_ne * self.sigma_mp_ne)
                        * (1.0 - self.phi_mp_ne * self.phi_mp_ne).max(0.0);
                }

                if win_len >= self.cfg.persist_min_samples {
                    let mut sum_n = 0.0_f64;
                    let mut sum_e = 0.0_f64;
                    for p in &self.d_1hz_window {
                        sum_n += p[0];
                        sum_e += p[1];
                    }
                    let mu_n = sum_n / win_f;
                    let mu_e = sum_e / win_f;
                    let mut var_n = 0.0_f64;
                    let mut var_e = 0.0_f64;
                    for p in &self.d_1hz_window {
                        let dn = p[0] - mu_n;
                        let de = p[1] - mu_e;
                        var_n += dn * dn;
                        var_e += de * de;
                    }
                    let mean_var = 0.5 * ((var_n / win_f) + (var_e / win_f));
                    let sd_axis = mean_var.max(0.0).sqrt();
                    let sd_eff = sd_axis.max(1.35);
                    let denom = sd_eff * self.cfg.persist_scatter_scale;
                    let z_p = [mu_n / denom, mu_e / denom];
                    for m in 0..CUSUM_NUM_ARMS {
                        let u = self.arm_unit_vectors[m];
                        let y_p = u[0] * z_p[0] + u[1] * z_p[1];
                        self.persist_arms[m] =
                            (self.persist_arms[m] + y_p - self.cfg.k_d).max(0.0);
                    }
                }
            }

            let mut combined = [0.0_f64; CUSUM_NUM_ARMS];
            let mut dom = 0usize;
            let mut best_c = -1.0_f64;
            for m in 0..CUSUM_NUM_ARMS {
                let c = self.cusum.arm_stats[m].max(self.persist_arms[m]);
                combined[m] = c;
                if c > best_c {
                    best_c = c;
                    dom = m;
                }
            }
            let max_s = if !innov_max_s.is_finite() {
                f64::INFINITY
            } else {
                innov_max_s.max(combined[dom])
            };
            let horiz_alarm = innov_alarm || max_s >= self.cfg.h_threshold;
            (z_k, combined, dom, max_s, horiz_alarm)
        } else {
            let mut combined = [0.0_f64; CUSUM_NUM_ARMS];
            let mut dom = 0usize;
            let mut best_c = -1.0_f64;
            for m in 0..CUSUM_NUM_ARMS {
                let c = self.cusum.arm_stats[m].max(self.persist_arms[m]);
                combined[m] = c;
                if c > best_c {
                    best_c = c;
                    dom = m;
                }
            }
            ([0.0, 0.0], combined, dom, combined[dom], false)
        };

        // 5b. Optional 2-sided vertical (`Down`) GNSS-vs-Baro CUSUM monitor
        let (vert_stat, vert_alarm) = self.step_vertical_cusum(input.gnss_down_m);
        let spoof_alarm = horiz_alarm || vert_alarm;

        // 6. Immediate (<= 1.0 s, same-epoch) GNSS quarantine & >=15-epoch hysteresis reintegration
        if spoof_alarm {
            if !self.gnss_quarantined {
                self.gnss_quarantined = true;
                quarantine_triggered = true;
                quarantine_latency_s = Some(0.0);
                self.xi_hat[8] = 0.0;
                self.xi_hat[9] = 0.0;
            }
            self.nominal_streak = 0;
        } else if self.gnss_quarantined {
            let sol_d = solve_2x2(p_diff_cur, d_k);
            let d_chi2 = d_k[0] * sol_d[0] + d_k[1] * sol_d[1];
            let consist_ok = d_chi2.is_finite() && d_chi2 <= 13.82;
            if consist_ok {
                self.xi_hat[8] *= 0.5;
                self.xi_hat[9] *= 0.5;
            }
            let stat_ok = max_s < (self.cfg.reintegration_stat_frac * self.cfg.h_threshold);
            let vert_ok =
                vert_stat < (self.cfg.reintegration_stat_frac * self.cfg.vertical_h_threshold);
            let vis_ok = vis_used && (q_vis <= self.cfg.max_q_vis);
            if stat_ok && vert_ok && vis_ok && consist_ok {
                self.nominal_streak += 1;
                if self.nominal_streak >= self.cfg.reintegration_hysteresis_epochs {
                    self.gnss_quarantined = false;
                    reintegrated = true;
                    self.nominal_streak = 0;
                    self.reset_detectors_post_reintegration();
                }
            } else {
                self.nominal_streak = 0;
            }
        } else {
            self.nominal_streak = 0;
        }

        let active_bank = if self.gnss_quarantined {
            "bank_b_visual"
        } else {
            "bank_a_gnss"
        };
        let active_ne = if self.gnss_quarantined {
            [self.x_b[0], self.x_b[1]]
        } else {
            [self.x_a[0], self.x_a[1]]
        };

        CusumStepOutput {
            epoch_idx: self.epoch_idx,
            timestamp_s: self.timestamp_s,
            active_bank,
            gnss_quarantined: self.gnss_quarantined,
            quarantine_triggered_this_epoch: quarantine_triggered,
            quarantine_latency_s,
            reintegrated_this_epoch: reintegrated,
            active_p_ned_m: [active_ne[0], active_ne[1], self.baro_down_m],
            bank_a_p_ne_m: [self.x_a[0], self.x_a[1]],
            bank_b_p_ne_m: [self.x_b[0], self.x_b[1]],
            d_ne_m: d_k,
            z_whitened: z_k,
            arm_stats: combined_arms,
            max_cusum_stat: max_s,
            dominant_arm: dom_arm,
            q_vis,
            nominal_streak: self.nominal_streak,
            p_diff_ne: p_diff_cur,
            vertical_cusum_stat: vert_stat,
            vertical_spoof_alarm: vert_alarm,
            spoof_alarm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_symmetrize_spd_reconstruction_and_floor() {
        let mut m = [[0.0_f64; 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = if i == j {
                    5.0 + (i as f64)
                } else {
                    0.35 / ((i.abs_diff(j) + 1) as f64)
                };
            }
        }
        let spd = symmetrize_spd_nxn(m, 1e-9);
        for (i, (spd_row, m_row)) in spd.iter().zip(m.iter()).enumerate() {
            for (j, (&s_val, &m_val)) in spd_row.iter().zip(m_row.iter()).enumerate() {
                assert!(
                    (s_val - m_val).abs() < 1e-12,
                    "Jacobi SPD reconstruction error at ({i},{j}): {s_val} vs {m_val}"
                );
            }
        }
        // Matrix with a negative eigenvalue must be floored cleanly to jitter
        let neg = [[1.0, 2.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]];
        let clipped = symmetrize_spd_nxn(neg, 1e-6);
        let lam_min_sub = eigvalsh_min_2x2([[clipped[0][0], clipped[0][1]], [clipped[1][0], clipped[1][1]]]);
        assert!((lam_min_sub - 1e-6).abs() < 1e-11);
    }

    #[test]
    fn test_sequence_vpr_w7_blind_zero_flat_and_recovery() {
        let mut coords = Vec::new();
        for r in 0..4 {
            for c in 0..4 {
                coords.push([-(r as f64) * 150.0, (c as f64) * 150.0]);
            }
        }
        let path: [usize; 8] = [0, 1, 2, 3, 7, 6, 5, 4];
        let mut flt = SequenceVprFilter::new(coords.clone(), SequenceVprConfig::default())
            .expect("valid SequenceVprFilter");

        for step_i in 0..7 {
            let cur = path[step_i];
            let delta = if step_i == 0 {
                [0.0, 0.0]
            } else {
                let prev = path[step_i - 1];
                [coords[cur][0] - coords[prev][0], coords[cur][1] - coords[prev][1]]
            };
            let mut sims = vec![0.15; 16];
            sims[cur] = 0.88;
            let dec = flt.step(&sims, delta, 20.0, None, None);
            assert!(dec.admitted, "step {step_i} rejected: {:?}", dec.reason_codes);
            assert_eq!(dec.final_top1_idx, cur as i64);
        }
        assert_eq!(flt.window_len(), 7);

        let delta_next = [
            coords[path[7]][0] - coords[path[6]][0],
            coords[path[7]][1] - coords[path[6]][1],
        ];
        // Blind all-zero fails closed without polluting window
        let dec_zero = flt.step(&[0.0; 16], delta_next, 20.0, None, None);
        assert!(!dec_zero.admitted && dec_zero.final_top1_idx == -1 && dec_zero.window_len == 7);
        assert!(dec_zero.reason_codes.iter().any(|r| r == "LOW_SIMILARITY_SIGNAL"));
        assert!(dec_zero.reason_codes.iter().any(|r| r == "FLAT_SIMILARITY_VECTOR"));

        // Constant 0.50 fails closed
        let dec_flat = flt.step(&[0.50; 16], delta_next, 20.0, None, None);
        assert!(!dec_flat.admitted && dec_flat.window_len == 7);
        assert!(dec_flat.reason_codes.iter().any(|r| r == "FLAT_SIMILARITY_VECTOR"));

        // NaN fails closed
        let mut nan_sims = vec![0.15; 16];
        nan_sims[2] = f64::NAN;
        let dec_nan = flt.step(&nan_sims, delta_next, 20.0, None, None);
        assert!(!dec_nan.admitted && dec_nan.window_len == 7);

        // Recovery on valid frame
        let mut valid_sims = vec![0.15; 16];
        valid_sims[path[7]] = 0.86;
        let dec_rec = flt.step(&valid_sims, delta_next, 20.0, None, None);
        assert!(dec_rec.admitted && dec_rec.final_top1_idx == path[7] as i64);
    }

    #[test]
    fn test_dual_bank_cusum_ramp_quarantine_vertical_nan_and_hysteresis() {
        let mut flt = DualBankEsEkfCusum::new(DualBankConfig::default()).expect("valid cusum");
        let ang = 22.5_f64.to_radians();
        let u = [ang.cos(), ang.sin()];

        // 2.0 m/s ramp triggers quarantine <= 12 s
        let mut ttd_s = None;
        for step_k in 1..=40 {
            let off = 2.0 * (step_k as f64);
            let out = flt.step(&DualBankStepInput {
                gnss_ne_m: Some([off * u[0], off * u[1]]),
                vis_ne_m: Some([0.0, 0.0]),
                vis_gate_admitted: true,
                ..DualBankStepInput::default()
            });
            if out.quarantine_triggered_this_epoch {
                ttd_s = Some(step_k as f64);
                break;
            }
        }
        assert!(ttd_s.is_some() && ttd_s.unwrap() <= 12.0, "ttd={ttd_s:?}");
        assert!(flt.gnss_quarantined);

        // Clean recovery requires >= 15 nominal epochs
        let mut rec_ep = None;
        for ep in 1..=80 {
            let out = flt.step(&DualBankStepInput {
                gnss_ne_m: Some([0.0, 0.0]),
                vis_ne_m: Some([0.0, 0.0]),
                vis_gate_admitted: true,
                ..DualBankStepInput::default()
            });
            if out.reintegrated_this_epoch {
                rec_ep = Some(ep);
                break;
            }
        }
        assert!(
            rec_ep.is_some() && rec_ep.unwrap() >= CUSUM_REINTEGRATION_HYSTERESIS_EPOCHS,
            "rec_ep={rec_ep:?}"
        );

        // NaN vertical gnss_down_m fails closed immediately
        let out_nan = flt.step(&DualBankStepInput {
            baro_down_m: -100.0,
            gnss_ne_m: Some([0.0, 0.0]),
            vis_ne_m: Some([0.0, 0.0]),
            vis_gate_admitted: true,
            gnss_down_m: Some(f64::NAN),
            ..DualBankStepInput::default()
        });
        assert!(out_nan.vertical_spoof_alarm && out_nan.spoof_alarm && out_nan.gnss_quarantined);
    }

    #[test]
    fn test_dual_bank_cusum_nonfinite_r_gnss_and_r_vis_fail_safe() {
        let mut flt = DualBankEsEkfCusum::new(DualBankConfig::default()).expect("valid cusum");
        // Non-finite measurement covariances must NEVER poison Bank-A/Bank-B state or covariances
        let out1 = flt.step(&DualBankStepInput {
            gnss_ne_m: Some([10.0, -5.0]),
            vis_ne_m: Some([10.0, -5.0]),
            vis_gate_admitted: true,
            r_gnss: Some([[f64::NAN, 0.0], [0.0, 4.0]]),
            r_vis: Some([[f64::INFINITY, 0.0], [0.0, 9.0]]),
            ..DualBankStepInput::default()
        });
        assert!(out1.bank_a_p_ne_m.iter().all(|v| v.is_finite()));
        assert!(out1.bank_b_p_ne_m.iter().all(|v| v.is_finite()));
        assert!(out1.p_diff_ne.iter().flatten().all(|v| v.is_finite()));
        assert!(out1.max_cusum_stat.is_finite());

        // Subsequent nominal step remains finite and tracks cleanly
        let out2 = flt.step(&DualBankStepInput {
            gnss_ne_m: Some([12.0, -4.0]),
            vis_ne_m: Some([12.0, -4.0]),
            vis_gate_admitted: true,
            ..DualBankStepInput::default()
        });
        assert!(out2.bank_a_p_ne_m.iter().all(|v| v.is_finite()));
        assert!(out2.bank_b_p_ne_m.iter().all(|v| v.is_finite()));
        assert!(out2.p_diff_ne.iter().flatten().all(|v| v.is_finite()));
    }
}
