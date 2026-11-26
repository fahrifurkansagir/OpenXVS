# OpenXVS Verification Evidence (`v1.1.1`)

Every metric reported below is backed by deterministic tests, machine-readable verification logs, and reproducible offline or software-in-the-loop (SIL) runs. All benchmarks enforce non-flight locks (`NON_FLIGHT_SIM_ONLY=1`, `ACTUATOR_AUTHORITY=false`) and zero GPL runtime dependencies.

---

## 1. Formal Methods & Bare-Metal Emulation (Ring-0)

| Verification Layer | Tooling & Scope | Verified Result |
| :--- | :--- | :--- |
| **Bounded Model Checking** | Kani CBMC across all 5 `#![no_std]` Ring-0 crates (`xvs-rta-core`, `xvs-active-range`, `xvs-odd-vlm`, `xvs-os-hal`, `xvs-multiship-coord`) | **`26 / 26` proofs PASS** (`5,228` multiship CBMC checks, `0` failures) |
| **Exhaustive State Exploration** | `tlc2.TLC` on multi-ship coordination & safety protocol (`N=2`, `N=3`) | `4,823,201` states (`289,000` distinct, `0` deadlocks) for `N=2`; `1,291,732` states for `N=3` |
| **Unbounded Symbolic Safety** | `Apalache v0.63.0` (Z3) inductive state + action invariants (`N=2`, `N=3`) | **`NoError`** (unbounded safety + BMC `length=7` PASS) |
| **Implementation Trace Refinement** | Rust-to-TLA+ deterministic state-trace replay | **`14,000 / 14,000` ticks PASS** (`0` violations) |
| **ARM Cortex-M7 Bare-Metal Emulation** | `xPack QEMU 9.2.4` (`-cpu cortex-m7 -M mps2-an500`, `thumbv7em-none-eabi`) | **`10,000 / 10,000` ticks PASS**, `77,917` invariants checked (`0` violations), `88,028 B` Flash, `0 B` static RAM & heap, FNV-1a64 `0x8ca234e6ff7ab48d` |

> **Hardware Honesty Boundary:** The Cortex-M7 execution proof runs on QEMU ISA emulation (`physical_stm32_hardware_verified = false`) and verifies machine-code semantics, stack discipline, and zero-heap invariants — not physical STM32H7 silicon timing or EMC.

---

## 2. Python-to-Pure-Rust Numerical Parity

Both `openxvs-studio.exe` and `openxvs-edge-infer.exe` compile byte-identical copies of `nav_integrity.rs`, audited directly against the Python reference implementations via `openxvs-edge-infer --parity-eval`:

- **`SequenceVprFilter` (`81` trajectory + edge steps):** Bayesian (`W=7`), Viterbi (`W=7`), all-zero blind input, constant-vector flat input, `NaN` injection, and aliasing threshold gating achieve **`max_abs_diff = 1.82e-12`** (`1.819e-12`) with **`100%` exact discrete/boolean decision parity**.
- **`DualBankEsEkfCusum` (`10,701` epochs):** Correlated multipath (`1 Hz` and `10 Hz`, `0.0 / fh` false alarms), horizontal slow-ramp spoofing (`0.05`–`5.0 m/s` at `22.5 deg`, `100%` caught below `30 m` HAL), step faults, and two-sided vertical `Down` ramps/`NaN` checks achieve **`max_abs_diff = 1.36e-12`** (worst-case envelope `< 3.0e-11`) with **`100%` exact alarm and bank-state parity**.

---

## 3. Pure-Rust ONNX Performance, Async Lock-Split RTA & Soak (`v1.1.1`)

- **Async Lock-Split Simplex RTA & Health Isolation:** Heavy Studio ONNX commands run via `tauri::async_runtime::spawn_blocking` while `RealtimeSafetyState` and `ArcJson` health snapshots stay wait-free under concurrent 6-model inference load:
  - **RTA Latency:** `p50 = 0.0097 ms`, **`p95 = 0.0385 ms`**, `p99 = 0.0655 ms`, `max = 0.5561 ms` (isolated baseline `0.0061 ms` p95 vs. `104.58 ms` before lock splitting).
  - **Health Snapshot Latency:** `p50 = 0.0006 ms`, `p95 = 0.0136 ms`, **`max = 0.1724 ms`** (`125x` lower max latency than synchronous mutex contention).
- **Warm CPU ONNX Inference (`openxvs-edge-infer --self-check --require-models`, `load_ms` separated):**
  - `xvs_xfeat`: `10.14 ms` (`1T`) &bull; `xvs_land`: `20.09 ms` (`1T`) &bull; `xvs_lighterglue`: `22.21 ms` (`1T`) &bull; `xvs_nav_v5`: `26.50 ms` (`2T`) / `37.68 ms` (`1T`) &bull; `xvs_d2`: `28.70 ms` (`1T`) &bull; `xvs_thermal`: `36.59 ms` (`1T`).
  - All `6 / 6` SHA-256 pinned ONNX weights verify fail-closed with `dual_blind_vacuity = 1.0000`; missing models or runtime exit `1` under `--require-models`.
- **Paced 6-Model Headless Soak (`5 Hz`, all 6 models, 2-pass ORT BFC warmup):**
  - `10 s @ 5 Hz` (`OMP_NUM_THREADS=2`, smart `nav=2T` / lightweight `1T` pinning): **`0` deadline misses**, steady-state RSS growth **`+0.17 MiB`** (`265.19 -> 265.36 MiB`), `0` non-finite outputs.
  - `305 s` sustained 6-model soak (`305.29 s`): **`+0.15 MiB` RSS**, **`-3` OS handles**, `0` panics, `0` non-finite outputs.

---

## 4. Live `ArduPlane 4.7.3` Continuous Mission SIL (`7 / 7` Phases)

Executed continuously against an official `ArduPlane 4.7.3` SITL binary over UDP (`0` digital-twin substitution, `pymavlink_imported = false`, `0` leftover background processes):

| Mission Phase | Stimulus & Test Condition | Verified Outcome |
| :--- | :--- | :--- |
| **1. Crosswind Flight** | `15 m/s` crosswind + turbulence, `#74 VFR_HUD` active | `max\|IAS - GS\| = 11.63 m/s`, `0` false CUSUM alarms |
| **2. Live GNSS Spoof** | Live `PARAM_SET(SIM_GPS1_GLTCH)` horizontal ramp | **`TTD = 10.0 s`**, `Bank-A = 20.20 m < 30 m HAL`, `Bank-B = 1.82 m` |
| **3. Datum-Guarded Switch** | `DatumAlignmentGuard` + live `EKF3 42007` switch | `\|innov\| = 0.67 m`, **`2x` live `COMMAND_ACK(42007, 0)`** |
| **4. Fixed-Wing DAA** | Dual-intruder `#75 COMMAND_INT` & `#82 SET_ATTITUDE_TARGET` | Stall-free **`min IAS = 19.54 m/s`**, **`0` `#84` frames** |
| **5. Simplex RTA Recovery** | Injected fault latch followed by go-around ack reset | Clean **`P0 -> P1 -> P4`** staged transition |
| **6. Mission Continuity** | Single uninterrupted flight (`32.8 s`) | `7 / 7` phases PASS with `0` link drop or crash |
| **7. Cryptographic Replay** | `24`-record SHA-256 hash-chained shadow flight tape (`samples/mission_flight_tape.ndjson`) | Python & Pure-Rust `--replay` match (`0` outbound TX); **exact 1-byte payload tamper rejected (`exit 1`)** |

---

## 5. Geometric Navigation Hardening & Security Fixes

- **Coarse Heading Geometric Rescue:** Multi-hypothesis XFeat yaw search recovers `>= 12`-inlier geometric matches under severe heading error: **`45 deg`: `18.8% -> 81.2%`**, **`90 deg`: `1.6% -> 85.9%`**, and **`180 deg`: `3.1% -> 92.2%`** (`0 / 64` far-distractor false positives, **`0` HMI**).
- **Oblique Rectification & Baro Guard:** Horizon-singularity-guarded homography (`w <= 1e-6`, depression `>= 15 deg`) eliminates sky wrap-around at `45 deg` and `60 deg` tilt (`0` wrap artifacts). `BaroThermalDriftGuard` maintains `0` false alarms under `10 min` thermal pressure drift while catching vertical spoofing ramps.
- **Hardened Security & Runtime Controls:**
  - **Optical Health, CUSUM & SeqVPR:** Cold-sky LWIR recovery (`q_lwir 0.287 -> 0.9353`), dead-pixel FPN (`>= 10%` fail-closed) and white-noise rejection (`q = 0.0`), 12-sector overcast EO scoring, 10D Gauss-Markov multipath whitener + vertical `Down` CUSUM, and `SequenceVprFilter` zero/flat blind-vector fail-closed gating — synchronized across Python, Studio, and Edge-Infer.
  - **MAVLink 2.0 & HIL Security:** `65,535 B` UDP RX DoS survival, mandatory MAVLink 2.0 signature verification (`10/10` forged unsigned frames rejected), fail-closed `DatumAlignmentGuard`, `#82` climb thrust floor (`type_mask = 0x07`), per-msgid `time_boot_ms` anti-replay, non-shadowed `StaleTelemetryError`, `#74 VFR_HUD` IAS/GS separation, and `MavGlobalPositionInt` bounds checks.
  - **Rust Runtime Hygiene:** Fail-closed `NaN`/`Inf` GPS spoof detection (`0` JSON `null` leaks), `--replay` fail-closed missing-visual handling, SPD covariance guards, `MIN_KEYLINE_SIGMA_PX = 0.25` clamp, `<= 16 MiB` / `<= 4096x4096` input bounds, and recursive SPDX `AND`/`OR` Zero-GPL + Zero-5D002 build audits.

---

## 6. Automated Regression Suites & Story Demonstration

- **Automated Test Suites:** **`696+` Python tests** (`pytest`), **`37 / 37` Rust tests** (`16` in `openxvs-edge-infer`, `21` in `openxvs_studio_lib`), and **`15 / 15` Studio Vitest specs** pass with `0` Clippy warnings (`-D warnings`), `0` GPL dependencies, `0` host build-path leaks, and `0` confidentiality TLS/AES crates across `453` Cargo packages.
- **Story Demonstration (`openxvs_v1.1.1_demo.mp4`):** Full `64 s` HD (`1280x720`, `30 fps`, H.264 `+faststart`, `4.16 MiB`) demonstration covering 3D Studio collision avoidance, unobstructed G1000 cockpit, 360-degree 6-camera dome & spoof shield, aerial-to-satellite RANSAC navigation, live `ArduPlane 4.7.3` spoof quarantine, real thermal IR fault fail-closed guarding, and live Pure-Rust `--self-check` + 1-byte tamper rejection (`exit 1`).
