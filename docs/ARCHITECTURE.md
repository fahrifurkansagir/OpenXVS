# OpenXVS Architecture (`v1.1.1`)

OpenXVS is a civil and dual-use autonomous aviation perception, GNSS-denied navigation integrity, and deterministic Runtime Assurance (RTA) architecture. The software stack enforces strict privilege separation across a bare-metal safety monitor (**Ring-0**), a Pure-Rust inference and avionics runtime (**Ring-1**), and a clean-room zero-GPL autopilot bridge (**SIL / HIL**).

All evaluation binaries compile with `NON_FLIGHT_SIM_ONLY=1` and `ACTUATOR_AUTHORITY=false`.

---

## 1. Layered Privilege Separation

### Ring-0: Bare-Metal `#![no_std]` Safety Kernel (`ARM Cortex-M7`)
Five `#![no_std]`, `#![forbid(unsafe_code)]` Rust crates enforce bounded runtime safety invariants with zero dynamic allocation and zero external OS dependencies:

- **`xvs-rta-core`**: Deterministic five-mode Simplex Runtime Assurance (`P0 InvalidInhibit` through `P4 NominalMl`), enforcing `P0` turn dominance and staged `P0 -> P1 -> P4` recovery gated by explicit go-around acknowledgement.
- **`xvs-active-range`**: Passive monocular bearing-rate ranging with CRC-32 wire-frame integrity checks.
- **`xvs-odd-vlm`**: Optical Operational Design Domain (ODD) gating across 12 azimuth sectors and CRC-validated advisory veto arbitration.
- **`xvs-os-hal`**: Absorbing `PluginOffline` crash-latch supervisor and firmware/UART bandwidth gate (`< 20%` of `921.6 kbps`).
- **`xvs-multiship-coord`**: `FAR 91.113` non-reversing turn arbitration, stateful MeshDAA staleness/replay/coplanar-ghost suppression (`500 ms` hysteresis), and GNSS-denied peer clock-offset estimation (`+120 ms` TCXO drift tolerance).

**Bare-Metal Footprint & Formal Boundary:** Compiled for `thumbv7em-none-eabi` (`opt-level=s`, `panic=abort`), the combined five-crate Ring-0 binary occupies `88 KB` (`88,028 B`) of Flash (`4.20%` of a `2 MiB` Cortex-M7 MCU) with `0 B` static RAM (`.data + .bss = 0 B`) and `0 B` heap allocation (core per-controller state fits within a `4 KB` SRAM budget; total combined five-crate test harness structs measure `8,256 B` on the stack). Ring-0 is verified via `26/26` Kani CBMC formal proofs, Apalache symbolic TLA+ model checking, and `10,000` deterministic execution ticks on QEMU Cortex-M7 (`mps2-an500`, emulation only).

### Ring-1: Pure-Rust ONNX Runtime & Async Lock-Split Simplex RTA
Ring-1 executes ML perception and avionics integrity natively in Rust without Python runtime coupling (`openxvs-studio.exe` and headless `openxvs-edge-infer.exe`):

- **Pinned ONNX Model Suite & Smart Threading:** Dynamically loads ONNX Runtime via explicit relative `ORT_DYLIB_PATH` (blocking Windows `System32\onnxruntime.dll` hijack) and verifies `6` SHA-256 pinned FP16 models (`30.33 MiB` total weights) covering EO traffic detection, EO/LWIR thermal fusion, runway keypoint/keyline landing guidance, visual place recognition, and geometric keypoint matching. Lightweight sub-30 ms sessions pin `1` intra-op thread while `xvs_nav_v5` scales to `2` intra-op threads when permitted, preventing tail-latency spikes.
- **Async Lock-Split Rate-Group Isolation:** Heavy Studio ONNX commands run asynchronously on blocking worker threads (`tauri::async_runtime::spawn_blocking`) while real-time safety state (`RealtimeSafetyState` wrapping `SimplexRta`) and `ArcJson` cached health snapshots execute wait-free without touching the inference mutex (`0.038 ms` p95 RTA latency and `0.172 ms` max health snapshot under concurrent 6-model load). Studio demo ticks apply fast/slow rate-group decimation, cached reference keypoints, LRU frame decoding, and single-copy JPEG IPC serialization.
- **Byte-Identical Avionics Integrity (`nav_integrity.rs`):** Shared directly between Studio and Edge-Infer, providing bit-for-bit aligned Rust implementations of `SequenceVprFilter` and `DualBankEsEkfCusum` with SPD covariance symmetrization and finiteness guards.

### Autopilot Telemetry & Audit Plane (Zero-GPL MAVLink 2.0 + `ArduPlane 4.7.3` SIL)
- **Zero-GPL Wire Codec:** Clean-room MAVLink 2.0 serializer/deserializer with zero GPL/AGPL dependencies (`pymavlink` excluded). Enforces mandatory signature verification when a key is provisioned, per-message-ID `time_boot_ms` monotonic anti-replay filtering, `65,535 B` UDP datagram DoS resilience, altitude/heading bounds checking, and fail-closed `StaleTelemetryError` blocking on telemetry blackout.
- **Fixed-Wing Command Safety:** Decodes `#74 VFR_HUD` to separate indicated airspeed (IAS) from ground speed (GS) under crosswind, drives avoidance strictly via `#75 COMMAND_INT` and `#82 SET_ATTITUDE_TARGET` with an active pitch-scaled thrust floor (`type_mask = 0x07`), emits `0` forbidden `#84 SET_POSITION_TARGET_LOCAL_NED` frames, enforces `>= 2.0 s` dwell hysteresis on `EKF3 42007` source switches, and gates altitude handoffs via `DatumAlignmentGuard` (WGS-84 ellipsoid vs. EGM96 MSL) and `BaroThermalDriftGuard`.
- **Tamper-Evident Flight Recorder:** Every telemetry epoch is recorded to a SHA-256 hash-chained NDJSON shadow tape verifiable identically in Python and Pure-Rust (`openxvs-edge-infer --replay`).

---

## 2. Strict Coordinate & Attitude Contract

All internal kinematics, camera projections, tracking, and autopilot bridges obey a single unambiguous coordinate standard with **`0` internal ENU**:

| Frame | Axes | Origin / Convention |
| :--- | :--- | :--- |
| **Geodetic / ECEF** | WGS-84 `(X, Y, Z)` | Earth-centered, Earth-fixed; orthometric heights converted via GEOID18 to ellipsoidal height |
| **Local Navigation** | **NED** `(North, East, Down)` | Local tangent plane; negative `Down` indicates positive altitude |
| **Aircraft Body** | **FRD** `(Forward, Right, Down)` | Aircraft reference point / IMU; camera-to-CG lever arm and `VISO_DELAY_MS` compensated in NED/FRD |
| **Camera Optical** | **RDF** `(Right, Down, Forward)` | Optical center; Hamilton quaternions `(w, x, y, z)` normalized with `w >= 0` |

Any third-party boundary requiring alternate conventions is strictly confined behind an isolated C-ABI adapter layer.

---

## 3. Navigation Integrity & Fail-Closed Sensor Health

### Dual-Bank 8-Arm Cholesky CUSUM & Sequence-VPR
- **Dual-Bank Isolation:** Bank A fuses INS, barometric altitude, and GNSS; Bank B maintains a GNSS-free INS + visual-position estimate.
- **8-Arm Cholesky CUSUM:** A `dt`-invariant 8-directional Page's CUSUM paired with a 10D Gauss-Markov correlated multipath whitener rejects sustained horizontal multipath (`0.0/fh` false alarms at `1 Hz` and `10 Hz`) while isolating slow horizontal spoofing ramps below the `30 m` Horizontal Alert Limit (HAL). A companion two-sided vertical `Down` CUSUM monitors barometric-vs-GNSS altitude divergence and fails closed on non-finite inputs.
- **Sequence-VPR & Geometric Rescue:** `SequenceVprFilter` (`W=7` Bayesian/Viterbi temporal smoothing) rejects all-zero, flat, non-finite, and aliased similarity vectors fail-closed. Oblique aerial imagery uses a horizon-singularity-guarded homography (`w <= 1e-6`, depression `>= 15 deg`) and multi-hypothesis coarse-heading search to recover large yaw offsets before fine geometric matching.

### Fail-Closed EO / LWIR Degradation Monitor
`SensorHealthMonitor` (verified in parity across Python, Studio, and Edge-Infer) continuously scores EO and LWIR streams before perception admission:

- **LWIR NUC / FFC Freeze Guard:** Detects microbolometer shutter freezes (`MAD < 0.25` for `>= 3` consecutive ticks or `> 250 ms`) and drives `q_lwir = 0.0`, transitioning cleanly to EO-primary or synthetic-vision fallback.
- **Dead-Pixel FPN & Noise Guards:** Flags fixed-pattern dead rows/columns (`>= 3%` fault, `>= 10%` fail-closed `q = 0.0`, computed via `O(256)` histogram-derived statistics) and lag-1 autocorrelation white-noise faults (`q <= 0.05`) that otherwise bypass Laplacian blur checks, blocking false track-before-confirm locks.
- **Cold-Sky Preservation & Dual-Blind Fail-Closed:** Preserves valid hot airborne targets against cold dark skies (`2x2` min-filter contrast check) while enforcing exact `dual_blind vacuity = 1.0000` -> `DUAL_DVE_SVS_FALLBACK` when both optical channels are degraded.
