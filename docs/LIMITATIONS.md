# OpenXVS Known Limitations & Engineering Boundaries (`v1.1.2`)

**Non-Flight Research & Simulation Scope:** OpenXVS `v1.1.2` is an offline verification, perception research, and software-in-the-loop (SIL) engineering suite (`NON_FLIGHT_SIM_ONLY=1`, `ACTUATOR_AUTHORITY=false`). It is **not certified** by the FAA, EASA, or any civil aviation authority and must never be coupled to live flight-control actuators.

This document records current perception boundaries and physical hardware domains that remain unverified on desktop and SIL hosts.

---

## 1. Current Perception Model Boundaries on Unseen Real-World Imagery

Deterministic safety monitors (`SensorHealthMonitor`, `SequenceVprFilter`, `DualBankEsEkfCusum`, and `SimplexRta`) hold **`0` Hazardously Misleading Information (HMI)** across all `9` real-world aviation benchmark clips by rejecting out-of-distribution inputs fail-closed. However, fail-closed safety reduces availability where current perception weights exhibit domain gaps:

- **Unseen Real-World Runway Approaches (`XVS-Land`):**
  - On unseen real cockpit approaches with uncalibrated intrinsics and off-center or nighttime geometry (verified on Eastsound day and Teterboro night cockpit footage), current 6-output weights exhibit center-template collapse (`100%` safe fail-closed rejection + Ring-0 go-around latch; `0%` CAT-I landing availability due to `5 Hz` horizontal protection limits exceeding `40 m` HAL).
  - Non-runway linear structures (highways, bridges, avenues, coastlines) trigger raw false detections (`480 / 480` frames on stress clips) and glare/occlusion uncertainty estimates are under-calibrated (`0.72x`–`0.93x` shrinkage under glare/rain). Zero HMI currently depends on strict downstream geometric and RTA rejection until multi-airport 7-output `runway_present_logit` weights are trained and promoted.
- **Airborne Sky Clutter & Sub-Nyquist Distant Targets (`XVS-D2`):**
  - Across zero-aircraft sky, cumulus cloud, bird, sunset glare, coastline, and urban-horizon clips, raw false detections fire on `85.5%` of frames (`1,119 / 1,309`), producing `91` false track-before-confirm locks with static-clutter rejection enabled (`104` without it) once overcast optical quality is accurately scored.
  - Detection recall drops sharply below `14 px` apparent target span, reaching `10%` raw (`0%` confirmed) at `9 px` and `0%` at `6 px` without higher-resolution `640x640` horizon tiling weights and hard-negative clutter retraining.
- **Thermal Ground Clutter & EO-LWIR Optical Parallax (`XVS-ThermalNet`):**
  - On zero-aircraft real thermal imagery (industrial plants, hot rooftops, wildfire/volcanic scenes), current weights act as a hot-blob detector (`86%`–`100%` false-positive frames).
  - Under cross-modal EO-to-LWIR spatial misalignment (`>= (6, 6)` LWIR pixels), LWIR cues dominate fusion while Dirichlet vacuity remains near `0.180`–`0.182` rather than crossing the `0.70` fault threshold.
- **Optical Weather Wall, Uncalibrated Field Cameras & Seasonal Shifts:**
  - Passive EO and 8–14 um LWIR sensors remain fundamentally bounded by dense advection fog and heavy precipitation (forcing `DUAL_DVE_SVS_FALLBACK`); they are not an all-weather IMC radar replacement.
  - Field videos lacking calibrated camera intrinsics degrade metric range and pose projections; strong cross-season albedo inversions (heavy snow, leaf-off canopy) reduce single-frame visual place recognition recall and force fail-closed aliasing rejections.

---

## 2. Physical Hardware & Flight-Test Domains Pending Qualification

Desktop simulation, recorded footage, QEMU ISA emulation, and UDP `ArduPlane 4.7.3` SITL cannot substitute for physical avionics qualification. The following hardware domains are explicitly unverified and never extrapolated or simulated as hardware truth:

1. **Target Embedded AI Silicon (`Jetson Orin` / ARM64 TensorRT):** Although `docker/edge-infer/build_trt_engines.sh` and ARM64 OCI recipes are verified syntactically and structurally for Day-1 bring-up, all local perception timing and soak measurements executed `100%` on x86-64 desktop CPU (`ONNX Runtime CPUExecutionProvider`). Physical Jetson Orin TensorRT FP16/INT8 latency, DLA/GPU memory contention, and sustained `7–25 W` thermal throttling at `+50 C` ambient are pending target-hardware bench evaluation.
2. **Physical Safety Microcontroller HIL (`Cortex-M7` / `STM32H7`):** QEMU `mps2-an500` proves bare-metal ARMv7E-M machine-code semantics, `0 B` heap/static RAM, and invariant logic, but does **not** verify physical silicon worst-case execution time (WCET), interrupt/DMA jitter on CAN-FD or UART links, hardware windowed watchdog (WWDT) brownout holdup, or `RTCA DO-160G` EMI/EMC and vibration endurance.
3. **Physical EO / LWIR Optics, Synchronization & Airframe Vibration:** Co-mounted EO/LWIR thermal-mechanical boresight drift across `-20 C` to `+50 C`, passive athermalized lens calibration, `< 1 ms` hardware global-shutter `FSYNC` camera-to-IMU synchronization, `50–120 Hz` propeller vibration isolation, hydrophobic/heated optics, and raw 14-bit radiometric microbolometer hardware FFC status flags require physical pod bench and shaker-table testing.
4. **Live RF GNSS Spoofing & Airborne RF Links:** Simulator glitch injection (`SIM_GPS1_GLTCH`) validates filter dynamics and alert latency, not physical L1/L2/L5 baseband correlation, C/N0 and AGC anomalies, gradual code-phase pull-off in an RF chamber, or airframe-shadowed half-duplex RF packet dropouts (`> 500 ms`).
5. **PX4 Physical Flight-Controller HIL:** `src/xvs/v2/studio_bridge/px4_mavlink.py` is verified via wire-codec, CRC_EXTRA, frame-guard, and UDP loopback tests; closed-loop live PX4-Autopilot EKF2 flight-controller HIL remains pending target bench testing.
6. **Airworthiness Certification & Open-Air Flight Test:** Formal `RTCA DO-178C` software assurance, `DO-254` complex electronic hardware assurance, `DO-330` compiler/toolchain qualification, certified `ARINC 653` partitioning, airframe-level `SAE ARP4754B` safety assessments, independent human DER/CVE review, and open-air flight testing remain outside this release's scope.

---

## 3. Scope, Export & Distribution Boundaries

- **Strictly Civil & Dual-Use Scope:** OpenXVS is restricted to civil aviation safety, passive detect-and-avoid, degraded visual environment awareness, and GNSS-resilient navigation.
- **Navigation Source Boundary:** Hybrid database-referenced navigation Python source is excluded from this clean-room snapshot under applicable export-control guardrails ([`docs/legal/EXPORTS.md`](legal/EXPORTS.md)).
- **Third-Party Runtime Licensing:** Standard NVIDIA TensorRT SDK licenses prohibit use in life-critical flight control; OpenXVS therefore isolates all safety-critical arbitration inside bare-metal `#![no_std]` Rust crates with zero GPU/TensorRT dependency.
