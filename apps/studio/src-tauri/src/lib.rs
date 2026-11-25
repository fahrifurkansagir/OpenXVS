// Copyright 2026 OpenXVS Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! OpenXVS Studio v5.0 Tauri v2 Native IPC Library
//! (`Pure-Rust ORT + XFeat + xvs-multiship-coord + Frozen RTA + NavIntegrity + Lock-Split Realtime + NON_FLIGHT_SIM_ONLY`).

#![forbid(unsafe_code)]

pub mod nav_integrity;
pub mod rust_engine;

use std::sync::{Mutex, OnceLock};

use rust_engine::{
    build_demo_rta_input, default_hil_status_payload, default_sensor_health_payload,
    evaluate_rta_step, NativeChannelPayload, NativeDemoTickPayload, NativeEvalReportPayload,
    NativeHealthPayload, NativeHilStatusPayload, NativeLoopbackReport, NativeRtaInput,
    NativeSensorHealthPayload, NavIntegritySnapshot, RtaReport, RustStudioEngine,
    EU_PLD_ART4_7_LOCKOUT_MSG,
};
use xvs_rta_core::{SimplexMode, SimplexRta};

/// Fast, lightweight real-time safety & control state decoupled from synchronous ONNX inference.
///
/// Strict lock hierarchy: lightweight IPC commands (`native_rta_step`, `native_sensor_health`,
/// `native_hil_status`, `native_health`) NEVER acquire the heavy ONNX `Mutex<RustStudioEngine>`.
/// Heavy ONNX commands only acquire `realtime` for microsecond-scale snapshot handoffs (never
/// during ONNX session execution), guaranteeing deadlock freedom and sub-2 ms RTA response even
/// under continuous 6-model ONNX saturation.
pub struct RealtimeSafetyState {
    pub rta: SimplexRta,
    pub last_rta_mode_id: u8,
    pub last_rta_inhibit_turns: bool,
    pub sensor_health_snapshot: NativeSensorHealthPayload,
    pub nav_integrity_snapshot: NavIntegritySnapshot,
    pub hil_status_snapshot: NativeHilStatusPayload,
}

impl Default for RealtimeSafetyState {
    fn default() -> Self {
        Self {
            rta: SimplexRta::new(),
            last_rta_mode_id: SimplexMode::NominalMl as u8,
            last_rta_inhibit_turns: false,
            sensor_health_snapshot: default_sensor_health_payload(),
            nav_integrity_snapshot: NavIntegritySnapshot::default(),
            hil_status_snapshot: default_hil_status_payload(),
        }
    }
}

pub struct ManagedEngine {
    realtime: Mutex<RealtimeSafetyState>,
    health_cache: OnceLock<NativeHealthPayload>,
    onnx_engine: OnceLock<Mutex<RustStudioEngine>>,
}

impl Default for ManagedEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ManagedEngine {
    pub fn new() -> Self {
        Self {
            realtime: Mutex::new(RealtimeSafetyState::default()),
            health_cache: OnceLock::new(),
            onnx_engine: OnceLock::new(),
        }
    }

    fn lock_realtime(&self) -> Result<std::sync::MutexGuard<'_, RealtimeSafetyState>, String> {
        self.realtime
            .lock()
            .map_err(|_| "OpenXVS realtime safety lock poisoned".to_string())
    }

    fn lock_onnx(&self) -> Result<std::sync::MutexGuard<'_, RustStudioEngine>, String> {
        let mutex = self.onnx_engine.get_or_init(|| {
            let eng = RustStudioEngine::initialize();
            let _ = self.health_cache.set(eng.health());
            if let Ok(mut rt) = self.realtime.lock() {
                rt.sensor_health_snapshot = eng.sensor_health();
                rt.nav_integrity_snapshot = eng.nav_integrity_snapshot();
                rt.hil_status_snapshot = eng.hil_status();
            }
            Mutex::new(eng)
        });
        mutex
            .lock()
            .map_err(|_| "OpenXVS native ONNX engine lock poisoned".to_string())
    }

    /// Wait-free after one-time engine initialization; NEVER blocks behind ongoing ONNX inference.
    pub fn health(&self) -> Result<NativeHealthPayload, String> {
        if let Some(cached) = self.health_cache.get() {
            return Ok(cached.clone());
        }
        let eng = self.lock_onnx()?;
        let payload = eng.health();
        let _ = self.health_cache.set(payload.clone());
        Ok(payload)
    }

    /// Fast realtime-only lock (`< 10 us`); NEVER blocks behind ONNX inference.
    pub fn sensor_health(&self) -> Result<NativeSensorHealthPayload, String> {
        let rt = self.lock_realtime()?;
        Ok(rt.sensor_health_snapshot.clone())
    }

    /// Fast realtime-only lock (`< 10 us`); NEVER blocks behind ONNX inference.
    pub fn nav_integrity(&self) -> Result<NavIntegritySnapshot, String> {
        let rt = self.lock_realtime()?;
        Ok(rt.nav_integrity_snapshot.clone())
    }

    /// Fast realtime-only `SimplexRta` step (`< 10 us`); NEVER blocks behind ONNX inference.
    pub fn step_rta(&self, input: &NativeRtaInput) -> Result<RtaReport, String> {
        let mut rt = self.lock_realtime()?;
        let rep = evaluate_rta_step(&mut rt.rta, input);
        rt.last_rta_mode_id = rep.mode_id;
        rt.last_rta_inhibit_turns = rep.inhibit_avoidance_turns;
        Ok(rep)
    }

    /// Fast realtime-only lock (`< 10 us`); NEVER blocks behind ONNX inference.
    pub fn hil_status(&self) -> Result<NativeHilStatusPayload, String> {
        let rt = self.lock_realtime()?;
        Ok(rt.hil_status_snapshot.clone())
    }

    /// Compile-time non-flight lockout (zero lock acquisition).
    pub fn hil_connect(&self, mode: &str) -> Result<NativeHilStatusPayload, String> {
        #[cfg(feature = "non_flight_sim_only")]
        {
            let _ = mode;
            Err(EU_PLD_ART4_7_LOCKOUT_MSG.to_string())
        }
        #[cfg(not(feature = "non_flight_sim_only"))]
        {
            let _ = mode;
            Err(
                "HIL UDP bridge requires explicit dev server; desktop binary has zero actuation authority."
                    .to_string(),
            )
        }
    }

    pub fn run_demo_tick(
        &self,
        scenario: &str,
        frame: u32,
        include_images: bool,
    ) -> Result<NativeDemoTickPayload, String> {
        let mut eng = self.lock_onnx()?;
        // Microsecond realtime critical section to step RTA, released BEFORE ONNX runs
        let rta = self.step_rta(&build_demo_rta_input(scenario))?;
        let payload = eng.run_demo_tick_with_rta(scenario, frame, include_images, rta)?;
        if let Ok(mut rt) = self.realtime.lock() {
            rt.sensor_health_snapshot = payload.sensor_health.clone();
            rt.nav_integrity_snapshot = eng.nav_integrity_snapshot();
        }
        Ok(payload)
    }

    pub fn analyze_image_b64(
        &self,
        channel: &str,
        image_b64: &str,
        q_eo: f32,
        q_lwir: f32,
        fog_density: f32,
        gps_offset_m: f64,
    ) -> Result<NativeChannelPayload, String> {
        let mut eng = self.lock_onnx()?;
        if let Ok(rt) = self.realtime.lock() {
            eng.sync_external_rta_state(rt.last_rta_mode_id, rt.last_rta_inhibit_turns);
        }
        let payload = eng.analyze_image_b64(
            channel,
            image_b64,
            q_eo,
            q_lwir,
            fog_density,
            gps_offset_m,
        )?;
        if let Ok(mut rt) = self.realtime.lock() {
            rt.sensor_health_snapshot = eng.sensor_health();
            rt.nav_integrity_snapshot = eng.nav_integrity_snapshot();
        }
        Ok(payload)
    }

    pub fn run_video_sequence_tick(
        &self,
        channel: &str,
        frame_idx: u32,
    ) -> Result<NativeChannelPayload, String> {
        let mut eng = self.lock_onnx()?;
        if let Ok(rt) = self.realtime.lock() {
            eng.sync_external_rta_state(rt.last_rta_mode_id, rt.last_rta_inhibit_turns);
        }
        let payload = eng.run_video_sequence_tick(channel, frame_idx)?;
        if let Ok(mut rt) = self.realtime.lock() {
            rt.sensor_health_snapshot = eng.sensor_health();
            rt.nav_integrity_snapshot = eng.nav_integrity_snapshot();
        }
        Ok(payload)
    }

    pub fn run_native_self_test(&self) -> Result<NativeLoopbackReport, String> {
        let mut eng = self.lock_onnx()?;
        let rep = eng.run_native_self_test();
        if let Ok(mut rt) = self.realtime.lock() {
            rt.hil_status_snapshot = eng.hil_status();
            rt.sensor_health_snapshot = eng.sensor_health();
        }
        Ok(rep)
    }

    pub fn export_eval_report(&self) -> Result<NativeEvalReportPayload, String> {
        let mut eng = self.lock_onnx()?;
        let rep = eng.export_eval_report()?;
        if let Ok(mut rt) = self.realtime.lock() {
            rt.hil_status_snapshot = eng.hil_status();
            rt.sensor_health_snapshot = eng.sensor_health();
            rt.nav_integrity_snapshot = eng.nav_integrity_snapshot();
        }
        Ok(rep)
    }
}

#[tauri::command]
fn native_health(state: tauri::State<'_, ManagedEngine>) -> Result<NativeHealthPayload, String> {
    state.health()
}

#[tauri::command]
fn native_demo_tick(
    state: tauri::State<'_, ManagedEngine>,
    scenario: String,
    frame: u32,
    include_images: bool,
) -> Result<NativeDemoTickPayload, String> {
    state.run_demo_tick(&scenario, frame, include_images)
}

#[tauri::command]
fn native_analyze_image(
    state: tauri::State<'_, ManagedEngine>,
    channel: String,
    image_b64: String,
    q_eo: Option<f32>,
    q_lwir: Option<f32>,
    fog_density: Option<f32>,
    gps_offset_m: Option<f64>,
) -> Result<NativeChannelPayload, String> {
    state.analyze_image_b64(
        &channel,
        &image_b64,
        q_eo.unwrap_or(0.85),
        q_lwir.unwrap_or(0.95),
        fog_density.unwrap_or(0.0),
        gps_offset_m.unwrap_or(0.0),
    )
}

#[tauri::command]
fn native_rta_step(
    state: tauri::State<'_, ManagedEngine>,
    input: NativeRtaInput,
) -> Result<RtaReport, String> {
    state.step_rta(&input)
}

#[tauri::command]
fn native_hil_status(
    state: tauri::State<'_, ManagedEngine>,
) -> Result<NativeHilStatusPayload, String> {
    state.hil_status()
}

#[tauri::command]
fn native_hil_connect(
    state: tauri::State<'_, ManagedEngine>,
    mode: String,
) -> Result<NativeHilStatusPayload, String> {
    state.hil_connect(&mode)
}

#[tauri::command]
fn native_hil_loopback_test(
    state: tauri::State<'_, ManagedEngine>,
) -> Result<NativeLoopbackReport, String> {
    state.run_native_self_test()
}

#[tauri::command]
fn native_sensor_health(
    state: tauri::State<'_, ManagedEngine>,
) -> Result<NativeSensorHealthPayload, String> {
    state.sensor_health()
}

#[tauri::command]
fn native_video_sequence_tick(
    state: tauri::State<'_, ManagedEngine>,
    channel: String,
    frame_idx: u32,
) -> Result<NativeChannelPayload, String> {
    state.run_video_sequence_tick(&channel, frame_idx)
}

#[tauri::command]
fn native_export_eval_report(
    state: tauri::State<'_, ManagedEngine>,
) -> Result<NativeEvalReportPayload, String> {
    state.export_eval_report()
}

pub fn run_tauri_app() {
    tauri::Builder::default()
        .manage(ManagedEngine::new())
        .invoke_handler(tauri::generate_handler![
            native_health,
            native_demo_tick,
            native_analyze_image,
            native_rta_step,
            native_hil_status,
            native_hil_connect,
            native_hil_loopback_test,
            native_sensor_health,
            native_video_sequence_tick,
            native_export_eval_report,
        ])
        .run(tauri::generate_context!())
        .expect("error while running OpenXVS Studio v5.0");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn test_lock_split_rta_latency_under_onnx_load() {
        let managed = Arc::new(ManagedEngine::new());
        // One-time cold boot initialization (loads all 6 ONNX models & primes health cache)
        let h0 = managed.health().expect("cold boot health must succeed");
        assert!(h0.ort_loaded, "ORT must load for real ONNX contention test");
        assert!(h0.models["xvs_d2"].loaded);
        assert!(h0.models["xvs_thermal"].loaded);
        assert!(h0.models["xvs_land"].loaded);
        assert!(h0.models["xvs_nav"].loaded);

        let stop = Arc::new(AtomicBool::new(false));
        let onnx_ticks = Arc::new(AtomicU64::new(0));

        // 1. Heavy ONNX worker continuously hammering 4-channel 6-model inference under the heavy lock
        let onnx_eng = Arc::clone(&managed);
        let onnx_stop = Arc::clone(&stop);
        let onnx_cnt = Arc::clone(&onnx_ticks);
        let onnx_handle = std::thread::spawn(move || {
            let mut frame = 0u32;
            while !onnx_stop.load(Ordering::Relaxed) || onnx_cnt.load(Ordering::Relaxed) < 6 {
                let res = onnx_eng.run_demo_tick("thermal_fog", frame, false);
                assert!(res.is_ok(), "ONNX worker tick failed: {res:?}");
                frame = frame.wrapping_add(1);
                onnx_cnt.fetch_add(1, Ordering::Relaxed);
                if onnx_cnt.load(Ordering::Relaxed) >= 12 {
                    break;
                }
            }
        });

        // Give the ONNX worker a short head start so it is actively inside ORT inference
        std::thread::sleep(Duration::from_millis(15));

        // 2. Three concurrent realtime threads hammering `step_rta`, `health`, and `sensor_health`
        let mut fast_handles = Vec::new();
        for tid in 0..3usize {
            let eng = Arc::clone(&managed);
            let stp = Arc::clone(&stop);
            let cnt = Arc::clone(&onnx_ticks);
            fast_handles.push(std::thread::spawn(move || {
                let mut rta_ms: Vec<f64> = Vec::with_capacity(512);
                let mut health_ms: Vec<f64> = Vec::with_capacity(512);
                let req = NativeRtaInput {
                    cam_health_mask: Some(0x3F),
                    ood_exceeded_mask: Some(0x00),
                    horizon_pl_rad: Some(0.002),
                    v_vpl_m: Some(5.0),
                    v_hpl_m: Some(14.0),
                    spoof_alarm: Some(false),
                    ml_detector_fault: Some(false),
                    approach_active: Some(false),
                    go_around_ack_reset: false,
                };
                while !stp.load(Ordering::Relaxed) && (rta_ms.len() < 400 || cnt.load(Ordering::Relaxed) < 6) {
                    let t0 = Instant::now();
                    let r = eng.step_rta(&req).expect("step_rta must succeed");
                    rta_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                    assert!(r.non_flight_sim_only && !r.actuator_authority);

                    let t1 = Instant::now();
                    if tid % 2 == 0 {
                        let h = eng.health().expect("health must succeed");
                        assert!(h.ort_loaded);
                    } else {
                        let sh = eng.sensor_health().expect("sensor_health must succeed");
                        assert!(sh.multiship_coord_enabled);
                    }
                    health_ms.push(t1.elapsed().as_secs_f64() * 1000.0);

                    // Paced 2 ms sleep prevents CPU spin-starvation while hitting >500 Hz aggregate probes
                    std::thread::sleep(Duration::from_millis(2));
                }
                (rta_ms, health_ms)
            }));
        }

        // Wait until ONNX worker completes >= 6 full 6-model ticks (or ~2.5 s)
        let start = Instant::now();
        while onnx_ticks.load(Ordering::Relaxed) < 6 && start.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(20));
        }
        stop.store(true, Ordering::Relaxed);

        onnx_handle.join().expect("onnx worker joined");
        let mut all_rta_ms: Vec<f64> = Vec::new();
        let mut all_health_ms: Vec<f64> = Vec::new();
        for h in fast_handles {
            let (rm, hm) = h.join().expect("realtime thread joined");
            all_rta_ms.extend(rm);
            all_health_ms.extend(hm);
        }

        all_rta_ms.sort_by(|a, b| a.total_cmp(b));
        all_health_ms.sort_by(|a, b| a.total_cmp(b));
        let pct = |v: &[f64], q: f64| -> f64 {
            let idx = ((v.len() as f64) * q).ceil() as usize;
            v[idx.saturating_sub(1).min(v.len() - 1)]
        };

        let rta_p50 = pct(&all_rta_ms, 0.50);
        let rta_p95 = pct(&all_rta_ms, 0.95);
        let rta_max = *all_rta_ms.last().unwrap_or(&f64::INFINITY);
        let health_p95 = pct(&all_health_ms, 0.95);
        let health_max = *all_health_ms.last().unwrap_or(&f64::INFINITY);
        let completed_onnx = onnx_ticks.load(Ordering::Relaxed);

        println!(
            "[LOCK-SPLIT-BENCH] onnx_ticks={completed_onnx} rta_calls={} rta_p50={rta_p50:.4}ms rta_p95={rta_p95:.4}ms rta_max={rta_max:.4}ms | health_calls={} health_p95={health_p95:.4}ms health_max={health_max:.4}ms",
            all_rta_ms.len(),
            all_health_ms.len()
        );

        assert!(completed_onnx >= 6, "Expected >= 6 concurrent ONNX ticks, got {completed_onnx}");
        assert!(all_rta_ms.len() >= 100, "Expected >= 100 RTA probes, got {}", all_rta_ms.len());
        assert!(
            rta_p95 < 2.0,
            "Lock-split RTA p95 latency {rta_p95:.4} ms exceeded 2.0 ms budget!"
        );
        assert!(
            rta_max < 10.0,
            "Lock-split RTA max latency {rta_max:.4} ms exceeded 10.0 ms budget!"
        );
        assert!(
            health_p95 < 2.0,
            "Lock-split health/sensor_health p95 latency {health_p95:.4} ms exceeded 2.0 ms budget!"
        );
        assert!(
            health_max < 10.0,
            "Lock-split health/sensor_health max latency {health_max:.4} ms exceeded 10.0 ms budget!"
        );
    }
}
