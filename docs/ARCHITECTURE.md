# OpenXVS Architecture (`v1.1.2`)

OpenXVS is a civil and dual-use autonomous aviation perception, GNSS-denied navigation integrity, and deterministic Runtime Assurance (RTA) architecture. The software stack enforces strict privilege separation across a bare-metal safety monitor (**Ring-0**), a Pure-Rust inference and avionics runtime (**Ring-1**), and clean-room zero-GPL ArduPilot & PX4 autopilot bridges (**SIL / HIL**).

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

**Bare-Metal Footprint & Formal Boundary:** Compiled for `thumbv7em-none-eabi` (`opt-level=s`, `panic=abort`), the combined five-crate Ring-0 binary occupies `88 KB` (`88,028 B`) of Flash (`4.20%` of a `2 MiB` Cortex-M7 MCU) with `0 B` static RAM (`.data + .bss = 0 B`) and `0 B` heap allocation (core per-controller state fits within a `4 KB` SRAM budget; total combined five-crate test harness structs measure `8,256 B` on the stack). Ring-0 is verified via **`26/26` Kani CBMC formal proofs running live in GitHub Actions CI**, Apalache symbolic TLA+ model checking, and `10,000` deterministic execution ticks on QEMU Cortex-M7 (`mps2-an500`, emulation only).

### Ring-1: Pure-Rust ONNX Runtime, `--eval-frames` & Async Lock-Split Simplex RTA
Ring-1 executes ML perception and avionics integrity natively in Rust without Python runtime coupling (`openxvs-studio.exe` and headless `openxvs-edge-infer.exe`):

- **Pinned ONNX Model Suite & Smart Threading:** Dynamically loads ONNX Runtime via explicit relative `ORT_DYLIB_PATH` (blocking Windows `System32\onnxruntime.dll` hijack) and verifies `6` SHA-256 pinned FP16 models (`30.33 MiB` total weights) covering EO traffic detection, EO/LWIR thermal fusion, runway keypoint/keyline landing guidance, visual place recognition, and geometric keypoint matching. Lightweight sub-30 ms sessions pin `1` intra-op thread while `xvs_nav_v5` scales to `2` intra-op threads when permitted; Python CPU ORT factories clamp `intra_op_num_threads <= OMP_NUM_THREADS` when explicitly set.
- **Pure-Rust Real-Footage Evaluation CLI (`--eval-frames`):** Evaluates directories of real JPEG/PNG frames directly in `#![forbid(unsafe_code)]` Rust across `xvs_d2`, `xvs_thermal`, and `xvs_land` (`--mode all|eo|thermal|land`) using a zero-allocation fixed-point separable resizer, pre-loop ORT session warmup, logit-space peak scans, native-resolution FPN/noise scans, and fail-closed `SimplexRta` gating — emitting a structured JSON report and SHA-256 hash-chained replay tape verifiable via `--replay`.
- **Jetson Orin Day-1 TensorRT Builder (`docker/edge-infer/build_trt_engines.sh`):** Fail-closed `trtexec` compilation script supporting FP16 and optional INT8 calibration across all 6 ONNX models (both `v1.1` `256x256` D2 / 6-output Land and extended `640x640` D2 / 7-output Land profiles), enforcing SHA-256 digest verification before build and logging p50/p95/p99 latency, VRAM, and `tegrastats` power envelopes strictly under `NON_FLIGHT_SIM_ONLY=1` and the NVIDIA TensorRT SLA boundary.
- **Async Lock-Split Rate-Group Isolation:** Heavy Studio ONNX commands run asynchronously on blocking worker threads (`tauri::async_runtime::spawn_blocking`) while real-time safety state (`RealtimeSafetyState` wrapping `SimplexRta`) and `ArcJson` cached health snapshots execute wait-free without touching the inference mutex (`0.038 ms` p95 RTA latency and `0.172 ms` max health snapshot under concurrent 6-model load).
- **Byte-Identical Avionics Integrity (`nav_integrity.rs`):** Shared directly between Studio and Edge-Infer, providing bit-for-bit aligned Rust implementations of `SequenceVprFilter` and `DualBankEsEkfCusum` with SPD covariance symmetrization and finiteness guards.

### Autopilot Telemetry & Audit Plane (Zero-GPL ArduPilot + PX4 MAVLink 2.0)
- **Zero-GPL Wire Codec (`hil_bridge.py` & `px4_mavlink.py`):** Clean-room MAVLink 2.0 serializer/deserializer with zero GPL/AGPL dependencies (`pymavlink` excluded). Enforces mandatory signature verification when a key is provisioned, per-message-ID `time_boot_ms` monotonic anti-replay filtering, `65,535 B` UDP datagram DoS resilience, altitude/heading bounds checking, and fail-closed `StaleTelemetryError` blocking on telemetry blackout.
- **Fixed-Wing ArduPilot Command Safety (`ArduPlane 4.7.3` SIL):** Decodes `#74 VFR_HUD` to separate indicated airspeed (IAS) from ground speed (GS) under crosswind, drives avoidance strictly via `#75 COMMAND_INT` and `#82 SET_ATTITUDE_TARGET` with an active pitch-scaled thrust floor (`type_mask = 0x07`), emits `0` forbidden `#84 SET_POSITION_TARGET_LOCAL_NED` frames, enforces `>= 2.0 s` dwell hysteresis on `EKF3 42007` source switches, and gates only the transition into `EKF3=2` via `DatumAlignmentGuard` (preventing spurious re-gating during post-switch DAA climbs) and `BaroThermalDriftGuard`.
- **Clean-Room PX4 Visual Odometry Dialect (`src/xvs/v2/studio_bridge/px4_mavlink.py`):** Encodes and decodes `#102 VISION_POSITION_ESTIMATE` (`CRC_EXTRA = 158`) and `#331 ODOMETRY` (`CRC_EXTRA = 91`) strictly in `MAV_FRAME_LOCAL_NED (1)` + `MAV_FRAME_BODY_FRD (12)` with upper-triangular 21-float covariance validation, fail-closed rejection of `MAV_FRAME_LOCAL_ENU` / unknown frames / `NaN` / `Inf` / non-unit quaternions, and `NON_FLIGHT_SIM_ONLY=1` outbound lockouts.
- **Tamper-Evident Flight Recorder:** Every telemetry and `--eval-frames` epoch is recorded to a SHA-256 hash-chained NDJSON shadow tape verifiable identically in Python and Pure-Rust (`openxvs-edge-infer --replay`).

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
- **Sequence-VPR, Geometric Rescue & Matched Color Order:** `SequenceVprFilter` (`W=7` Bayesian/Viterbi temporal smoothing) rejects all-zero, flat, non-finite, and aliased similarity vectors fail-closed. Oblique aerial imagery uses a horizon-singularity-guarded homography (`w <= 1e-6`, depression `>= 15 deg`), explicit `color_order` (`"bgr"`/`"rgb"`) forwarding into 3-channel XFeat, pinned LighterGlue ONNX matching with independent query/map resolution normalization (`img_hw`, `img_hw_m`), and multi-hypothesis coarse-heading search to recover large yaw offsets.

### Fail-Closed Native-Resolution EO / LWIR Degradation Monitor
`SensorHealthMonitor` (verified in parity across Python, Studio, and Edge-Infer) evaluates **native-resolution** EO and LWIR frames before any `INTER_AREA` perception downscaling:

- **LWIR NUC / FFC Freeze Guard:** Detects microbolometer shutter freezes (`MAD < 0.25` for `>= 3` consecutive ticks or `> 250 ms`) and drives `q_lwir = 0.0`, transitioning cleanly to EO-primary or synthetic-vision fallback.
- **Dead-Pixel FPN & Noise Guards:** Evaluates native 720p/1080p imagery before downscaling so 1-pixel hot FPN stripes (`>= 3%` fault, `>= 10%` fail-closed `q = 0.0`, `O(256)` histogram stats) and uncorrelated white-noise faults (`q <= 0.05`) are caught in `1` frame instead of being smoothed out by resizing.
- **Stateful Horizon Tiling & Static-Clutter TBC:** `D2TrackerState` enables static-clutter track-before-confirm rejection by default (`reject_static_clutter=True`, preserving closing head-on targets) and retains per-tile ConvGRU temporal history (`c1..c5`, `h_prev`) across multi-crop horizon sweeps.
- **Cold-Sky Preservation & Dual-Blind Fail-Closed:** Preserves valid hot airborne targets against cold dark skies (`2x2` min-filter contrast check) while enforcing exact `dual_blind vacuity = 1.0000` -> `DUAL_DVE_SVS_FALLBACK` when both optical channels are degraded.
