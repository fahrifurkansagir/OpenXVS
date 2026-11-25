# OpenXVS Aviation Safety Disclaimer & Product-Liability Governance Notice (`SAFETY_DISCLAIMER.md`)

> **CRITICAL AVIATION SAFETY & PRE-DEC 9, 2026 EU PLD NOTICE — READ BEFORE BUILDING, EVALUATING, OR OPERATING ANY OPENXVS ARTIFACT.**

---

## 1. Not Certified for Safety-Critical or Human-Life-Controlling Flight (`G5_GAP`)

OpenXVS (including all software crates, ONNX neural network weights, simulation bridges, documentation, and `xvs-certgen` Goal Structuring Notation templates) is a **civil aviation research, simulation, and assurance-engineering architecture** — **NOT** a certified flight-control or civil airworthiness product.

- OpenXVS is **NOT certified, approved, or authorized** by the FAA, EASA, ICAO, or any other civil aviation authority (CAA) for safety-critical, human-life-controlling, passenger-carrying, or unmitigated autonomous flight operations.
- Formal verification artifacts (including `18/18` Rust `Kani` CBMC harnesses across `xvs-rta-core`, `xvs-active-range`, `xvs-odd-vlm`, and `xvs-os-hal`) verify bounded mathematical properties of isolated software modules **only**. They do **NOT** constitute `RTCA DO-178C`, `DO-254`, `DO-160G/H`, `DO-330`, `EUROCAE ED-324` (`SAE ARP6983`), `ASTM F3442-26`, or `FAA Part 108` / `EASA SORA 2.5` airworthiness certification.
- Every `xvs-certgen` and SORA 2.5 / F3442-26 traceability artifact explicitly keeps target-hardware `DO-254` qualification, `DO-330` compiler/RTOS qualification, `ARP4754B` airframe FHA/PSSA/SSA, physical flight-test validation, and independent human `DER/CVE` (`QA-01`) sign-off open as **UNDEVELOPED (`G5_GAP`)** boundary nodes, alongside **`G2_TLA_DEFERRED`** for full distributed temporal model checking.

---

## 2. Strict Civil-Only Purpose (`PROJECT_CHARTER.md` §2)

OpenXVS is engineered exclusively for **civil aviation safety** (`14 CFR Part 91/107/108/135`, **EASA SORA 2.5** effective November 30, 2026, `ASTM F3442-26`, and `EUROCAE ED-324`): non-cooperative Detect-and-Avoid (DAA), GNSS-denied visual navigation resilience, precision runway/forced-landing safety, wildfire/SAR/infrastructure perception, and open runtime assurance. No component is designed, intended, or authorized for weaponization, lethal targeting, defense fire-control, electronic countermeasures, or military combat operations.

---

## 3. Directive (EU) 2024/2853 (Effective December 9, 2026) — Release-Mode Liability Firewall

The revised **EU Product Liability Directive (Directive (EU) 2024/2853**, aligned by the May 7, 2026 Corrigendum `OJ L 2026/90364`) applies to software, AI systems, and digital manufacturing files placed on the EU market or put into service **after December 8, 2026** (effective **December 9, 2026**). OpenXVS enforces an explicit structural distinction between its two distribution postures:

### 3.1 Current Phase: Private-First Confidential Non-Flight Evaluation (`private_eval`, Nov 2026 – Sep 2027)
- **NO `Art. 2(2)` Open-Source Exemption Claimed:** Because `Directive (EU) 2024/2853 Art. 2(2)` excludes only *"free and open-source software that is developed or supplied outside the course of a commercial activity"*, and `Art. 4(7)` defines making available on the market as supply **"whether in return for payment or free of charge"**, closed/NDA-gated evaluation binaries do **NOT** rely on `Art. 2(2)` immunity.
- **Physical Non-Flight Isolation:** Every Private-First evaluation binary and edge benchmark kit is compiled with `NON_FLIGHT_SIM_ONLY=1` (`ACTUATOR_AUTHORITY = false`, zero outbound MAVLink/PWM/CAN flight-controller actuation, immutable `SIMULATION & BENCHMARK EVALUATION ONLY — ZERO ACTUATION — NOT FOR FLIGHT` watermark), collects zero personal data, restricts writes to sandboxed workspace directories (`Art. 6`), and excludes all `.stl`/`.gcode`/`.kicad*` digital manufacturing files (`Art. 4(1)`).
- **Mandatory Legal & Operational Gate:** Private-First evaluation kits are supplied strictly under a countersigned bilateral evaluation agreement and Non-Flight Evaluation License. Any unauthorized flight-controller coupling or airborne deployment by a recipient constitutes an unforeseeable **substantial modification** (`Art. 4(18)` / `Art. 8(2)`) under the recipient's exclusive liability. University aero-lab testing during Private-First is limited strictly to **physically air-gapped passive shadow-mode data logging** under the university's own permit and aviation insurance.
- **Hard Commercial & Live-Flight Stop:** Zero commercial revenue and zero flight-actuating software distribution occur prior to limited-liability corporate incorporation, formal export/legal counsel clearance, and binding **Aviation Products Liability + Tech E&O** coverage aligned with the **10-year `Art. 17` expiry period** (25 years for latent injury, restarting upon substantial AI updates under `Art. 4(18)` / `Art. 11(2)`) and negotiated B2B `Art. 12(2)` micro/small-enterprise recourse frameworks.

### 3.2 Future Timed Public Open-Core Releases (Model 2: Wave-1 Q1 2027 & Wave-2 October 2027)
- Only **after** clean-room unrestricted public release — **Wave-1 (Q1 2027, brand-decoupled formal Rust/Kani crates & `xvs-certgen` under `Apache-2.0`)** and **Wave-2 (October 2027, Community Core & Studio under `Apache-2.0`, datasets/weights under `CDLA-Permissive-2.0`, reference CAD under `CERN-OHL-P-2.0`, docs under `CC-BY-4.0`)** — will the public non-commercial community repositories be supplied free of charge outside commercial activity within the meaning of **`Directive (EU) 2024/2853 Art. 2(2)` & Recital 15**.
- Under `Directive (EU) 2024/2853 Art. 8`, any commercial manufacturer, OEM, or operator that integrates public OpenXVS software, model weights, or CAD files into a commercial aircraft, sensor pod, or service bears **sole manufacturer product-liability, regulatory qualification, and lifecycle security-update responsibility** (including EU Cyber Resilience Act `Regulation (EU) 2024/2847` compliance).

---

## 4. Operator & Integrator Responsibility

Any organization evaluating, integrating, or (in future public/commercial phases) flying hardware or software derived from OpenXVS assumes **100% sole legal, regulatory, and operational responsibility** for:
1. **Civil Regulatory Authorization:** Complying with `14 CFR Part 91/107/89/108/135`, `14 CFR § 107.53` (ADS-B Out transmit prohibition), `EASA Regulations (EU) 2019/945`, `2019/947`, **EASA SORA 2.5** (effective Nov 30, 2026), and `U-Space (EU) 2021/664`.
2. **Independent Human Override & Shadow Isolation:** Maintaining physical air-gapping during Private-First evaluation and independent hardware manual RC safety-pilot override during any future authorized flight testing.
3. **RF, Spectrum & Privacy Compliance:** Complying with FCC Part 15 / ETSI EN 300 220 passive RX-only rules, sky-only elevation masking, and ADS-B identifier scrubbing.

---

## 5. Statutory Limit on Disclaimers (`Art. 15`) & AS-IS Provision

**Mandatory Statutory Notice (`Directive (EU) 2024/2853 Art. 15`):** Contractual warranty disclaimers and liability exclusions cannot limit or exclude strict statutory liability toward an injured natural person where Directive (EU) 2024/2853 applies; for that reason, OpenXVS enforces physical non-flight isolation rather than relying on disclaimers alone.

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, ALL SOFTWARE, MODELS, DATASETS, AND DOCUMENTATION ARE PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AIRWORTHINESS, OR NON-INFRINGEMENT.
