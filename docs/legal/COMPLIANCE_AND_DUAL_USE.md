# OpenXVS `v1.1.1` — Civil Aviation Scope, Regulatory Honesty & Release Governance (`COMPLIANCE_AND_DUAL_USE.md`)

- **Document Scope:** OpenXVS `v1.1.1` (`xvs-core`, `xvs-active-range`, `xvs-odd-vlm`, `xvs-os-hal`, `xvs-multiship-coord`, `xvs-rt`, `apps/livedeck`, `apps/studio`, `docker/edge-infer`, `src/xvs/v2`, `assurance/`)
- **Architecture & Limitations:** [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) & [`docs/LIMITATIONS.md`](../LIMITATIONS.md)
- **Coordinate Standard:** `WGS-84 ECEF` / `Local NED` / `Body FRD` / `Camera RDF`, Hamilton `(w, x, y, z)` with `w >= 0`, zero internal ENU
- **Regulatory Baseline:** November 2026 (`Nov 19` BIS `4E091` repeal, `Aug 13` `91 FR 52501` `3-hr / <300-km` UAV & Part 744 Supp. 2 rule, `Nov 20` Reg. `(EU) 2026/2288`, `Nov 30` EASA SORA 2.5 cutover, `Dec 9` EU PLD `Directive (EU) 2024/2853`)

---

## 1. Strictly Civil & Dual-Use Governance Boundary (`Charter §2`)

OpenXVS is developed exclusively for **civil aviation safety, Detect-and-Avoid (DAA), degraded-visual-environment (DVE) situational awareness, precision runway/forced-landing safety, and GNSS-resilient navigation** supporting:
- Civil General Aviation (`14 CFR Part 91 / Part 135`) and Civil Unmanned Aircraft Systems (`14 CFR Part 107 / Part 108` BVLOS, `< 3-hour` endurance and `< 300-km` range civil airframes during Private-First).
- **EASA SORA 2.5** (`SAIL II–III` civil inspection, medical logistics, island cargo, wildfire/forestry monitoring, and maritime SAR), **ASTM F3442-26** mixed cooperative/non-cooperative DAA, and **EUROCAE ED-324 (`SAE ARP6983`)** offline ML assurance.

### Explicitly Prohibited Uses & Non-Goals
OpenXVS MUST NOT include, support, or be adapted for:
- Weapons integration, munitions guidance, defense fire-control, or lethal targeting (`USML Cat. VIII/XII`, `EAR 7D611`, EU Common Military List, Turkish `Law No. 5201`).
- Ordnance approach warning, military IFF/transponder interrogation, electronic warfare, electronic countermeasures, RF jamming, spoofing transmission, or military combat operations.
- Unlicensed commercial flight-simulator capture or scraping of proprietary map tiles (`Google`, `Bing`, `Mapbox`, `*.arcgis.com`).

---

## 2. Two-Phase Release Governance (Model 2: Staged Split Open-Core)

To eliminate licensing and liability contradictions, OpenXVS strictly separates its **current Private-First phase** from **future Timed Public Open-Core releases**:

1. **Current Phase — Private-First Non-Flight Evaluation (`private_eval`, Nov 2026 – Sep 2027):**
   - Strictly **non-public / confidential** desktop & bench simulation evaluation ONLY (`ACTUATOR_AUTHORITY = false`, `NON_FLIGHT_SIM_ONLY=1`, zero flight-controller actuation, immutable `SIMULATION & BENCHMARK EVALUATION ONLY` watermark).
   - **No** `Apache-2.0` / `CDLA-Permissive-2.0` / `CERN-OHL-P-2.0` grant is active; **no** `15 CFR § 734.7` public-availability claim and **no** `Directive (EU) 2024/2853 Art. 2(2)` FOSS exemption are asserted (`Art. 4(1)`, `Art. 4(7)`, `Art. 15`, `Art. 17` govern closed binaries even at zero cost).
   - Governed by [`EXPORTS.md`](EXPORTS.md) and bilateral non-flight evaluation agreements. Private `XVS-Nav` source code is withheld from cross-border transfer (`7D003.b.3` guard), `.stl`/`.gcode`/`.kicad*` manufacturing files are excluded (`Art. 4(1)`), and private desktop kits carry zero TLS/AES confidentiality crypto (`5D002` — pure Ed25519 signing/integrity only).
2. **Planned Wave-1 Public Release (`wave1_formal_core`, Q1 2027 — Brand-Decoupled):**
   - Clean-room public `Apache-2.0` release of deterministic formal assurance artifacts ONLY (`xvs-certgen`, SORA 2.5 / F3442-26 / ED-324 traceability & SPDX 2.3 SBOM generator, and `#![no_std]` Rust/Kani crates) under a **brand-decoupled** identity preserving May 2027 NeurIPS E&D double-blind anonymity.
3. **Planned Wave-2 Public Open-Core Flip (`public_open_core`, October 2027 — Post-NeurIPS Camera-Ready):**
   - Clean-room unrestricted public release under **`Apache-2.0`** (code & pure-OSS Community Studio), **`CDLA-Permissive-2.0`** (datasets & model weights), **`CERN-OHL-P-2.0`** (reference CAD/PCB/BOM), and **`CC-BY-4.0`** (docs & GSN) — activating `15 CFR § 734.7` and `Directive (EU) 2024/2853 Art. 2(2)` for the non-commercial public community core.
4. **Commercial Revenue & Live-Flight Gate (`REL6`, 2028+):**
   - Hard stop against accepting commercial revenue or distributing flight-actuating software prior to corporate incorporation, formal export/legal counsel ECCN/CCATS & Turkish Law 5201 / Ministry of Trade clearance, and Aviation Products Liability + Tech E&O coverage aligned with EU PLD 10-year `Art. 17` exposure.

---

## 3. Regulatory Honesty, Certification Independence & `G5_GAP`

OpenXVS is a **research, verification, and assurance-engineering platform**, NOT a type-certificated avionics product or TSO-approved flight article:
1. **No Unqualified Certification Claims:** ML perception modules (`XVS-D2`, `XVS-Nav`, `XVS-Land`, `XVS-ThermalNet`) are bounded at runtime by `#![no_std]` Rust Simplex RTA (`xvs-rta-core`), `xvs-active-range`, `xvs-odd-vlm`, `xvs-os-hal`, and `xvs-multiship-coord` (`26/26` Kani CBMC proofs), plus offline `ED-324`, `ASTM F3442-26`, and `EASA SORA 2.5` traceability evidence (`src/xvs/v2/assurance/sora25_f3442_matrix.py`).
2. **Sensor Emission Precision:** Passive EO and 8–14 µm LWIR cameras operate with zero active radar emissions and emit zero active RF energy. Cooperative `XVS-MeshDAA` (`#52001`) and ADS-B RX transceivers are standard civil RF avionics links and are never described as RF-silent.
3. **ArduPlane Fixed-Wing Command Safety:** Fixed-wing avoidance strictly prohibits MAVLink `#84` (`SET_POSITION_TARGET_LOCAL_NED`) and enforces `#75 COMMAND_INT` / `#82 SET_ATTITUDE_TARGET` plus `EKF3 42007` dwell (`>= 2.0 s`), with all outbound actuation physically compiled out in `private_eval` (`NON_FLIGHT_SIM_ONLY=1`).

| Internal OpenXVS Control | Official Civil Aviation Standard | Scope Covered Locally | Remaining Certification Gap (`G5_GAP`) |
| :--- | :--- | :--- | :--- |
| Independent `D16` Adversarial Audit | `RTCA DO-178C Table A-7` (Verification Independence) | Automated read-only code/test/provenance verification | Independent human DER / CVE verification (`QA-01`) |
| `ED-324` GSN v3 + SORA 2.5 / F3442-26 Matrix | `EUROCAE ED-324`, `EASA SORA 2.5` (17 OSOs, M1A/B/C), `ASTM F3442-26` | Machine-checkable GSN DAG, SHA-256 evidence, SPDX 2.3 SBOM, Ed25519 receipts | Formal CAA/NAA review, airframe `ARP4754B` FHA/PSSA/SSA, flight-test qualification |
| `XVS-VRAIM` Offline OOD / SAE / AST Audits | `EASA NPA 2025-07 (Issue 03) SU-DA-04 / QA-01` | Seeded `R-LM-07`–`R-LM-11` JSON audit + Mahalanobis/SAE/AST | Qualified production training data governance & human `QA-01` sign-off |

### Explicit Undeveloped Certification Gaps (`G5_GAP` & `G2_TLA_DEFERRED`)
In frozen [assurance/openxvs_v2.gsn.yaml](../../assurance/openxvs_v2.gsn.yaml), **`G5_GAP`** remains permanently **`undeveloped: true`** (zero synthetic solutions) covering:
- `RTCA DO-254` complex electronic hardware assurance;
- `RTCA DO-330` compiler/toolchain qualification (`rustc`, `g++`, `clang`, ONNX/TensorRT compilers, verification scripts);
- Certified `ARINC 653` RTOS partitioning;
- Airframe-level `SAE ARP4754B` FHA/PSSA/SSA and physical environmental/EMI (`DO-160G/H`) qualification;
- Independent human `DER/CVE` (`EASA QA-01` / `DO-178C Table A-7`) sign-off.

**`G2_TLA_DEFERRED`** likewise remains explicitly **`undeveloped: true`** in frozen `assurance/openxvs_v2.gsn.yaml` until formally promoted via a verified multi-ship TLA+ trace-refinement release.

---

## 4. Third-Party Software, Dataset & Zero-GPL License Register

All dependencies and datasets enforce **0 GPL / AGPL / LGPL-static** runtime linkage (audited deterministically in SPDX 2.3 SBOMs and `xvs-core/deny.toml`) and obey `configs/xvs/data_allowlist.json` & `src/xvs/v2/license_guard.py`:

| Component / Dataset | License | Isolation / Usage Rule in OpenXVS |
| :--- | :--- | :--- |
| **`NASA DAIDALUS 2.0.4`** | `NOSA 1.3` (NOT Apache-2.0) | Isolated behind pure C-ABI boundary (`third_party/daidalus_boundary/`); zero upstream NOSA source bundled; `NED <-> ENU` confined inside `xvs_daidalus_bridge.cpp`. |
| **`DAQP` Solver Interface** | `MIT` | Pure C-ABI active-set QP contract; standalone zero-malloc fixed-size solver linked. |
| **ONNX Runtime (`ort` `load-dynamic`)** | `MIT` (`ort` Rust crate `MIT OR Apache-2.0`) | Loaded dynamically via explicit relative `ORT_DYLIB_PATH` (blocks Win11 `System32\onnxruntime.dll` hijack). NVIDIA TensorRT recipes (Jetson Orin NX) remain OEM-side non-flight evaluation recipes subject to NVIDIA TensorRT SLA Sec. 1.2 & 2 (never redistributed into public OSS registries). |
| **Amazon `AOT`** | `CDLA-Permissive-1.0` | Primary EO small-target DAA dataset (`XVS-D2`), strict `capture_day` split & `m_box=0`. |
| **Copernicus `GLO-30` & `Sentinel-2 L2A`** | `ESA / Copernicus Free & Open Attribution` | Bare-earth DEM & multi-season optical training/eval (attribution preserved). |
| **USGS `3DEP` / `HRO` / `NHD`, NOAA `GEOID18`, FAA `NASR`, USDA `NAIP`, FEMA `USA Structures`** | `U.S. Public Domain` (`ORNL CC-BY-4.0` attribution noted on FEMA structures) | Authoritative terrain, geoid, runway geometry, and orthophoto corpora. |
| **`ALTO`**, **`IRSTD-1k`**, **`HIT-UAV`**, **`TartanAir`**, **`ECCV2026-LTC`** | `BSD-3-Clause` / `MIT` / `CC-BY-4.0` | Permissive attribution EO/LWIR/navigation corpora. |
| **`MobileNetV4-Hybrid-M`, `MixVPR-Clean`, `XFeat`, `LighterGlue`** | `Apache-2.0` | Clean-room / verified permissive `XVS-Nav` Stage-1/Stage-2 stack. |
| **`LARD` Real Airports** | `Research Permission (Eval-Only, Never Redistribute; Zero Training)` | Eval-only real-airport runway validation (`>= 36` ICAO manifest, zero `KSFO` overlap). |

### Permanently Banned Dependencies, Weights & Datasets
- **Copyleft / Non-Permissive Code:** `pymavlink` (GPL/LGPL), `PyInstaller` (GPLv2 bootloader + `.pyc` extraction leak), `SALAD` (GPL-3.0), `dump1090`/`readsb`/`v4l2loopback`/`DFReader.py` (GPL), `JSBSim`/`ardupilot_gazebo` (LGPL/GPL direct linkage; ArduPlane SITL is strictly process-isolated over UDP).
- **Non-Commercial / Custom Weights & Datasets:** `ConvNeXt-V2` weights (`CC-BY-NC-4.0`), `DINOv3` / `RoMa v2` (custom Meta license), `FastViT` (`apple-ascl`), `UFM-DINOv2` (`CC-BY-NC-SA-4.0`), `SuperPoint` (Magic Leap NC), `C-RADIOv4` (NVIDIA proprietary), `MobileGeo` / `MixVPR` upstream (no license), `MegaLoc` (GPL SALAD + GSV-Cities), `university_1652`, `sues_200`, `lard_synthetic`, `msfs`, `xplane`, `*.arcgis.com`, and scraped commercial map tiles.

---

## 5. Honest Deferral & Environment Transparency

Every unexecuted hardware, cloud-training, or open-air flight metric remains explicitly marked unverified (`docs/LIMITATIONS.md`); zero hardware benchmarks or flight captures are ever fabricated.
