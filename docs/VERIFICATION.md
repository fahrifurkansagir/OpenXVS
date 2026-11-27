# OpenXVS Verification Evidence (`v1.1.2`)

Every metric reported below is backed by deterministic tests, machine-readable verification logs, and reproducible offline or software-in-the-loop (SIL) runs. All benchmarks enforce non-flight locks (`NON_FLIGHT_SIM_ONLY=1`, `ACTUATOR_AUTHORITY=false`) and zero GPL runtime dependencies.

---

## 1. Formal Methods, Full-Stack CI & Bare-Metal Emulation (Ring-0)

| Verification Layer | Tooling & Scope | Verified Result |
| :--- | :--- | :--- |
| **Bounded Model Checking (Live in CI)** | Kani CBMC `0.68.0` across all 5 `#![no_std]` Ring-0 crates (`xvs-rta-core`, `xvs-active-range`, `xvs-odd-vlm`, `xvs-os-hal`, `xvs-multiship-coord`) in `.github/workflows/xvs_core_ci.yml` | **`26 / 26` proofs PASS** (`5,228` multiship CBMC checks, `0` failures) |
| **Exhaustive State Exploration** | `tlc2.TLC` on multi-ship coordination & safety protocol (`N=2`, `N=3`) | `4,823,201` states (`289,000` distinct, `0` deadlocks) for `N=2`; `1,291,732` states for `N=3` |
| **Unbounded Symbolic Safety** | `Apalache v0.63.0` (Z3) inductive state + action invariants (`N=2`, `N=3`) | **`NoError`** (unbounded safety + BMC `length=7` PASS) |
| **Implementation Trace Refinement** | Rust-to-TLA+ deterministic state-trace replay | **`14,000 / 14,000` ticks PASS** (`0` violations) |
| **ARM Cortex-M7 Bare-Metal Emulation** | `xPack QEMU 9.2.4` (`-cpu cortex-m7 -M mps2-an500`, `thumbv7em-none-eabi`) | **`10,000 / 10,000` ticks PASS**, `77,917` invariants checked (`0` violations), `88,028 B` Flash, `0 B` static RAM & heap, FNV-1a64 `0x8ca234e6ff7ab48d` |
| **Full-Stack GitHub Actions CI** | `xvs_core_ci.yml` (5-crate `thumbv7em` + `26/26` Kani + `edge-infer-rust` + `studio-frontend` + clean-room `pytest`) & `xvs_edge_arm64_oci.yml` (native `ubuntu-24.04-arm` + SPDX SBOM) | Automated on every push & PR; `0` GPL crates, `0` 5D002 TLS/AES crates |

> **Hardware Honesty Boundary:** The Cortex-M7 execution proof runs on QEMU ISA emulation (`physical_stm32_hardware_verified = false`) and verifies machine-code semantics, stack discipline, and zero-heap invariants — not physical STM32H7 silicon timing or EMC.

---

## 2. Pure-Rust `--eval-frames` Performance & Real-Footage Cross-Test (`v1.1.2`)

`openxvs-edge-infer.exe` (`1.58 MiB`, `#![forbid(unsafe_code)]`, `0` host-path leaks) evaluates real multi-scenario aviation frames end-to-end across `xvs_d2`, `xvs_thermal`, and `xvs_land` (`--mode all`, `OMP_NUM_THREADS=1`, single-threaded CPU ORT floor `~104.1 ms`):

- **16 Real Multi-Scenario Aviation Frames (`640x480` to `1920x1080`):**
  - Pre-optimization baseline: `p50 = 160.23 ms`, `p95 = 193.22 ms`, `max = 193.22 ms`.
  - **Optimized `v1.1.2` (`--eval-frames --mode all`):** **`p50 = 108.06 ms`** (`-32.6%`), **`p95 = 121.69 ms`**, **`max = 121.69 ms`** (`frames_processed = 16/16`, `frames_corrupt = 0`, `outbound_mavlink_frames = 0`, non-ORT overhead `3.63 ms` at `640x480` and `12.00–12.89 ms` at `1920x1080`).
  - **Flight-Recorder Chain & Tamper Proof:** Emitted `16`-record SHA-256 hash-chained tape verified via `--replay` (`exit 0`, `valid = true`, `tx_inhibited_all = true`, `actuator_authority_all_false = true`); a 1-byte mutation fails closed with **`exit 1`** (`valid = false`, `first_failed_line = 2`).
- **8-Frame Air-to-Air Apples-to-Apples Check (`640x480` PNG):**
  - Reduced from `610.19 ms` p50 / `1105.56 ms` p95 down to **`112.39 ms` p50 / `133.96 ms` p95** (`-81.6%` p50, `-87.9%` p95) with identical detection and fail-closed parity.
- **Self-Contained Zero-Env-Var Unpack Verification:** Unpacked `OpenXVS-Edge-CLI-Win64-v1.1.2.zip` in a clean directory with `ORT_DYLIB_PATH` and `OPENXVS_MODEL_DIR` unset passes `--self-check --require-models` (`6/6` models, `exit 0`), `--eval-frames samples\frames --mode all` (`4/4` real frames, `exit 0`), `--replay eval_tape.ndjson` (`valid = true`, `exit 0`), and `--replay samples\mission_flight_tape.ndjson` (`24/24` valid, `exit 0`).

---

## 3. 9-Clip Open-Licensed Real-World Aviation Video Benchmark (`v1.1.2`)

Evaluated via `scripts/xvs/run_real_video_benchmark.py` across **`9` genuine open-licensed aviation videos** (Public Domain / CC0 / CC-BY / CC-BY-SA, zero synthetic video) with plain-English proof reel `openxvs_v1.1.2_real_video_benchmark.mp4` (`44.0 s`, `1280x720 @ 30 fps`, `3.89 MiB`):

| Scenario | Real Footage Evaluated | Verified Outcome |
| :--- | :--- | :--- |
| **1. Moving-Camera Air-to-Air** | Budapest glider (`CC BY-SA 3.0`, unseen) + NASA SR-71 (`Public domain`) | `18.5–18.6 FPS` single-pass (`97–100/100` det); `4.6 FPS` `4x1` horizon tiled (`98–100/100` det); `94–98` confirmed tracks; **`0` HMI** |
| **2. Unseen Cockpit Landing** | Eastsound day (`CC BY-SA 2.0`) + Teterboro night (`CC BY 3.0`, unseen) | `12.4–12.8 FPS` (`78.4–80.9 ms`), **`100%` safe fail-closed rejection** + Ring-0 Simplex RTA go-around latch on uncalibrated unseen approaches; **`0` HMI** |
| **3. Aerial Drone Visual Nav** | Eastern Oregon BLM (`Public domain`, unseen) + Wales drone (`CC0`) | `138.8–139.4 ms`, **`2.15x–2.50x` XFeat inlier lift** from BGR-to-RGB fix (`107.6 vs 50.0`, `73.1 vs 29.2`), `+30 deg` & `+90 deg` heading rescue `6/6`, `100%` off-map reject, **`3.4 s` CUSUM spoof catch**; **`0` HMI** |
| **4. Airborne Thermal IR & Faults** | NASA WB-57 IR + KSC X-37B IR (`Public domain`) + unseen FLIR hot clutter | Shutter freeze caught in `4` frames (`q_lwir = 0.00`), 1-px FPN caught in **`1` frame**, white noise in **`1` frame**, dual-blind in **`1` frame (`vacuity = 1.000`)**; **`0` HMI** |

---

## 4. Simultaneous 4-Fault Compound `ArduPlane 4.7.3` SIL (`8 / 8`) & PX4 MAVLink 2.0

Executed continuously (`30.99 s`) via `scripts/xvs/run_compound_fault_sil_gate.py` against an official `ArduPlane 4.7.3` SITL binary over UDP (`pymavlink_imported = false`, strictly NED, `0` leftover WSL processes):

| Compound Check | Simultaneous Stimulus | Verified Outcome |
| :--- | :--- | :--- |
| **0. Crosswind Baseline** | `15 m/s` crosswind + `5.0` turbulence (`#74 VFR_HUD`) | `max\|IAS - GS\| = 14.36 m/s`, `0` false CUSUM alarms, `min q_lwir = 0.9931` |
| **1. Live GNSS Spoof + Switch** | Live `SIM_GPS1_GLTCH` ramp under crosswind + `DatumAlignmentGuard` | **`TTD = 9.0 s`**, `Bank-A = 19.13 m < 30 m HAL`, `Bank-B = 2.10 m`, `\|innov\| = 0.000 m`, live `ACK(42007, 0)` |
| **2. Simultaneous `90 deg` Spin** | Live `90 deg` visual heading perturbation during spoof | Naive `0 deg`: `6` inliers -> `4x90` XFeat sweep: **`191` inliers** (`0.666 px` center err, `0` distractor HMI) |
| **3. Simultaneous LWIR Freeze** | Injected microbolometer shutter freeze during spoof + DAA | Caught at step `3`, **`q_lwir = 0.0000`** (`LWIR_DEGRADED_EO_PRIMARY`), clean recovery to `0.9931` |
| **4. Simultaneous Fixed-Wing DAA** | Intruder avoidance (`#75 COMMAND_INT` & `#82 SET_ATTITUDE_TARGET`) | Stall-free **`min IAS = 20.75 m/s`** (`>= 17.0 m/s` floor), **`0` `#84` frames** on wire |
| **5. Staged Recovery** | `>= 2.0 s` GNSS dwell reversion + Ring-0 RTA go-around ack | `EKF3 2 -> 1` (`2x ACK 42007=0`), clean **`P0 -> P1 -> P4`** recovery, `0` spurious datum errors |
| **6. Dual Replay & 1-B Tamper** | `30`-record SHA-256 hash-chained compound tape | Python & Pure-Rust `--replay` match (`head_hash` exact, `0` outbound TX); **1-byte tamper rejected (`exit 1`)** |
| **7. Teardown Hygiene** | Post-flight WSL process audit | **`0` leftover `arduplane` PIDs**, `pymavlink_imported = false` |

- **Sequential 7-Phase SIL Baseline (`v1.1.1`):** Continues to pass `7/7` phases (`15 m/s` crosswind, `10.0 s` TTD, `min IAS = 19.54 m/s`, `24`-record `samples/mission_flight_tape.ndjson`).
- **Clean-Room Zero-GPL PX4 MAVLink 2.0 (`src/xvs/v2/studio_bridge/px4_mavlink.py`):** Verified canonical CRC_EXTRA derivation (`#102 VISION_POSITION_ESTIMATE -> 158`, `#331 ODOMETRY -> 91`), SHA-256 MAVLink 2.0 signing/tamper rejection, strict `LOCAL_NED (1)` + `BODY_FRD (12)` enforcement (rejecting `MAV_FRAME_LOCAL_ENU` on encode and decode), `NaN`/`Inf`/negative-variance rejection, and `NON_FLIGHT_SIM_ONLY=1` lockouts.
- **Jetson Orin Day-1 TensorRT Kit (`docker/edge-infer/build_trt_engines.sh`):** Verified Unix LF hygiene, `bash -n` syntax, fail-closed SHA-256 check, all 6 models (`v1.1` and extended high-res profiles), `trtexec` FP16/INT8 + p50/p95/p99 + `tegrastats` trap cleanup.

---

## 5. Python-to-Pure-Rust Parity, Async RTA & 8 Runtime Fixes

- **Numerical Parity (`openxvs-edge-infer --parity-eval`):** `SequenceVprFilter` (`81` steps, `max_abs_diff = 1.82e-12`) and `DualBankEsEkfCusum` (`10,701` epochs, `max_abs_diff = 1.36e-12`) achieve `100%` exact discrete/alarm parity across Python, Studio, and Edge-Infer (`nav_integrity.rs` byte-identical).
- **Async Lock-Split RTA & Soak:** Wait-free Studio RTA (`0.0385 ms` p95) and health snapshots (`0.1724 ms` max) under concurrent 6-model ONNX load; `0` deadline misses in `5 Hz` soak (`+0.17 MiB` RSS) and `305 s` sustained soak (`+0.15 MiB` RSS, `-3` OS handles).
- **8 Wave-1 Runtime, Safety & Thermal Fixes Verified (`tests/test_v112_bug_fixes.py` + `eval_frames.rs`):**
  1. Wired pinned LighterGlue ONNX into `StudioInferenceEngine` with graceful Sinkhorn fallback (`BUG-01`).
  2. Forwarded `img_hw_m=(h_ref, w_ref)` in `infer_crossview_nav` for mismatched 720p/1080p vs `224x224` tiles (`BUG-02`).
  3. Moved `infer_thermal_dve` `SensorHealthMonitor` checks to native-resolution EO/LWIR before `INTER_AREA` downscaling, catching 720p white-noise and 1080p 1-px FPN stripes in `1` frame (`BUG-03`).
  4. Added `color_order` (`"rgb"` default / `"bgr"` OpenCV) forwarding across `coarse_heading.py` & `sequence_vpr.py` (`BUG-04`).
  5. Defaulted `D2TrackerState` TBC `reject_static_clutter=True` and preserved per-tile ConvGRU temporal history across HD horizon sweeps (`BUG-05`).
  6. Clamped CPU ORT `intra_op_num_threads <= OMP_NUM_THREADS` when set (`BUG-06`).
  7. Prevented `StudioHilBridgeManager` from re-gating `DatumAlignmentGuard` once `EKF3=2` is already latched (`BUG-07`).
  8. Eliminated redundant resizes, per-frame clones, and `expf` heatmap scans in Pure-Rust `--eval-frames` (`BUG-08`, `108.06 ms` p50).

---

## 6. Automated Regression Suites & Release Demonstrations

- **Automated Test Suites:** **`717+` Python tests** (`pytest` in full dev checkout; `0` errors / `0` failures in clean-room `pytest -q`), **`47 / 47` Rust tests** (`26` in `openxvs-edge-infer`, `21` in `openxvs_studio_lib`), and **`15 / 15` Studio Vitest specs** pass with **`0` Clippy warnings** (`-D warnings`), `0` GPL dependencies, `0` host build-path leaks, and `0` confidentiality TLS/AES crates in `Cargo.lock`.
- **Release Videos:** `openxvs_v1.1.2_demo.mp4` (`64 s`, `4.16 MiB` HD Studio & avionics story demo) and `openxvs_v1.1.2_real_video_benchmark.mp4` (`44 s`, `3.89 MiB` plain-English 9-clip real-video proof reel).
