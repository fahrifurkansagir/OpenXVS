# OpenXVS Verification Evidence (`v1.1.0`)

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
- **`DualBankEsEkfCusum` (`10,701` epochs):** Correlated multipath (`1 Hz` and `10 Hz`, `0.0 / fh` false alarms), horizontal slow-ramp spoofing (`0.05`–`5.0 m/s` at `22.5 deg`, `100%` caught below `30 m` HAL), step faults, and two-sided vertical `Down` ramps/`NaN` checks achieve **`max_abs_diff = 2.91e-11`** (`2.910e-11`) with **`100%` exact alarm and bank-state parity**.

---

## 3. Real-Time Contention, Sustained Soak & Edge CLI

- **Lock-Split Simplex RTA Isolation:** Under 4-thread concurrent 6-model ONNX inference load (`2,257` safety calls), decoupling `RealtimeSafetyState` and wait-free `OnceLock` health snapshots reduced RTA latency from `104.58 ms` p95 (`163.80 ms` max) to **`0.0061 ms` p95** (`2.54 ms` max) — a **`17,144x` p95 latency reduction**.
- **`305 s` Sustained 6-Model Headless Soak (`305.29 s`):** Steady-state memory growth **`+0.15 MiB` RSS** (`+0.149 MiB`, `420 MiB` peak = `6.84%` of a `6,144 MiB` Jetson Orin Nano memory budget), **`-3` OS handles**, `0` panics, and `0` non-finite outputs.
- **Paced Headless CLI Soak (`openxvs-edge-infer --soak --seconds 30 --hz 5`):** `0` deadline misses with `p95 = 46.86 ms` on CPU-only ONNX Runtime; `--self-check` verifies `6 / 6` SHA-256 pinned ONNX models and exact `dual_blind_vacuity = 1.0000`.

---

## 4. Live `ArduPlane 4.7.3` Continuous Mission SIL (`7 / 7` Phases)

Executed continuously against an official `ArduPlane 4.7.3` SITL binary over UDP (`0` digital-twin substitution, `pymavlink_imported = false`, `0` leftover background processes):

| Mission Phase | Stimulus & Test Condition | Verified Outcome |
| :--- | :--- | :--- |
| **1. Crosswind Flight** | `15 m/s` crosswind + turbulence, `#74 VFR_HUD` active | `max\|IAS - GS\| = 11.63 m/s`, `0` false CUSUM alarms |
| **2. Live GNSS Spoof** | Live `PARAM_SET(SIM_GPS1_GLTCH)` horizontal ramp | **`TTD = 10.0 s`**, `Bank-A = 20.08 m < 30 m HAL` |
| **3. Datum-Guarded Switch** | `DatumAlignmentGuard` + live `EKF3 42007` switch | `|innov| = 0.67 m`, **`2x` live `COMMAND_ACK(42007, 0)`** |
| **4. Fixed-Wing DAA** | Dual-intruder `#75 COMMAND_INT` & `#82 SET_ATTITUDE_TARGET` | Stall-free **`min IAS = 19.47 m/s`**, **`0` `#84` frames** |
| **5. Simplex RTA Recovery** | Injected fault latch followed by go-around ack reset | Clean **`P0 -> P1 -> P4`** staged transition |
| **6. Mission Continuity** | Single uninterrupted flight (`33.45 s`) | `7 / 7` phases PASS with `0` link drop or crash |
| **7. Cryptographic Replay** | `24`-record SHA-256 hash-chained shadow flight tape | Python & Pure-Rust `--replay` match (`0` outbound TX); **exact 1-byte payload tamper rejected (`exit 1`)** |

---

## 5. Geometric Navigation Hardening & Adversarial Security Fixes

- **Coarse Heading Geometric Rescue:** Multi-hypothesis XFeat yaw search recovers `>= 12`-inlier geometric matches under severe heading error: **`45 deg`: `18.8% -> 81.2%`**, **`90 deg`: `1.6% -> 85.9%`**, and **`180 deg`: `3.1% -> 92.2%`** (`0 / 64` far-distractor false positives, **`0` HMI**).
- **Oblique Rectification & Baro Guard:** `rectify.py` horizon-singularity guard (`w <= 1e-6`, depression `>= 15 deg`) eliminates sky wrap-around at `45 deg` and `60 deg` tilt (`0` wrap artifacts). `BaroThermalDriftGuard` maintains `0` false alarms under `10 min` thermal pressure drift while catching vertical spoofing ramps.
- **`18` Hardened Security & Runtime Fixes:**
  - **5 Optical Health, CUSUM & SeqVPR fixes:** Cold-sky LWIR recovery (`q_lwir 0.287 -> 0.9353`, `0%` false degradation on NASA WB-57 IR), dead-pixel FPN & white-noise Laplacian-bypass rejection (`q = 0.0`), 12-sector overcast EO quality fix, 10D Gauss-Markov multipath whitener + vertical `Down` CUSUM (`0.0/fh` FA), and `SequenceVprFilter` zero/flat blind-vector fail-closed gating.
  - **8 MAVLink 2.0 & HIL Bridge Security fixes:** `65,535 B` UDP RX DoS survival, mandatory MAVLink 2.0 signature verification (`10/10` forged unsigned frames rejected), fail-closed `DatumAlignmentGuard` when `visual_alt_m` is missing, `#82` climb thrust floor (`type_mask = 0x07`, fixing a `6.60 m/s` stall to `17.76–19.47 m/s`), per-msgid `time_boot_ms` anti-replay, non-shadowed `StaleTelemetryError`, clean-room `#74 VFR_HUD` IAS/GS separation, and `MavGlobalPositionInt` bounds validation.
  - **5 Rust Runtime Hardening fixes:** Fail-closed `NaN`/`Inf` GPS spoof detection (`0` JSON `null` leaks), recoverable `SimplexRta` `P0` latch via `go_around_ack_reset`, native-res FPN/noise/cold-sky sensor-health parity in Rust, `MIN_KEYLINE_SIGMA_PX = 0.25` subnormal clamp across Studio and Edge-Infer, and signal-derived navigation confidence (eliminating constant fallback confidence on blind frames).

---

## 6. Automated Regression Suites & Real-Footage Demonstration

- **Automated Test Suites:** **`696+` Python tests** (`pytest`) and **`30 / 30` Rust tests** (`11` in `openxvs-edge-infer`, `19` in `openxvs_studio_lib`) pass with zero GPL imports and clean-room binary hygiene (`0` host-path leaks, `0` confidentiality TLS/AES crates across `453` Cargo packages).
- **Real-Footage Verification Showcase (`openxvs_v1.1.0_demo.mp4`):** Full `86 s` HD (`1280x720`, 30 fps, H.264) demonstration compiled from real open-licensed footage and live verification telemetry covering air-to-air DAA (`269/270` detected, `253` TBC-confirmed frames, `max_prob = 0.9408`), nadir visual matching (`81` keyframes, `49.6` mean / `77` max inliers), thermal FFC-freeze / FPN / dual-blind vacuity (`1.0000`), `24`-epoch `ArduPlane 4.7.3` SIL replay, and Pure-Rust `--self-check` + 1-byte tamper rejection (`exit 1`).
