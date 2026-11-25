# OpenXVS Export-Control & Regulatory Compliance Notice (`EXPORTS.md`)

> **Governance Status (Updated November 2026 — Model 2 Staged Split Open-Core):**
> OpenXVS operates under a **two-phase distribution architecture** that strictly separates the
> current **Private-First Maturation Phase** (`private_eval`, non-public) from future **Timed
> Public Open-Core Releases** (`wave1_formal_core` in Q1 2027 and `public_open_core` Wave-2 in
> October 2027). Engineering regulatory-risk guidance only — **NOT** formal legal advice.
> Before cross-border transfer of private navigation source code, commercial incorporation, or
> hardware export, obtain binding classification from qualified export-control counsel.

---

## 1. Release-Mode Jurisdiction Split (`Private-First` vs. `Timed Public Open-Core`)

| Release Posture | Active Window | Public Availability (`15 CFR § 734.7` & EU GSN) | License Grant Status | Mandatory Airlock Controls |
| :--- | :--- | :--- | :--- | :--- |
| **Current: Private-First (`private_eval`)** | Nov 2026 – Sep 2027 | **NON-PUBLIC / RESTRICTED.** Access-restricted repos, NDA-gated OEM kits, and private cloud copies do **NOT** qualify for `15 CFR § 734.7` or EU Dual-Use General Software Note (GSN) public-domain exclusions. | **Confidential Non-Flight Evaluation Only** (no active `Apache-2.0` / `CDLA-Permissive-2.0` / `CERN-OHL-P-2.0` public grant). | Clean-room `git archive` genesis export; exclude `XVS-Nav` source (`7D003.b.3`); zero TLS/AES confidentiality crypto (`5D002`); zero `.stl`/`.gcode`/`.kicad*`; counterparty screening; `docs/xvs/governance/EXPORTS_PRIVATE_EVAL.md`. |
| **Planned Wave-1 (`wave1_formal_core`, Model 2)** | Q1 2027 | **PUBLIC** upon unrestricted Internet publication (`15 CFR § 734.7` & EU GSN apply to published formal core only). | `Apache-2.0` (`xvs-certgen` + `#![no_std]` Rust/Kani formal crates, **brand-decoupled** from NeurIPS 2027 E&D double-blind ML benchmark). | Zero ML weights, zero training pipelines, zero `XVS-Nav` DBRN fusion source, zero legacy history. |
| **Planned Wave-2 (`public_open_core`, Model 2)** | Oct 2027 (post-NeurIPS camera-ready) | **PUBLIC** upon unrestricted Internet publication (`15 CFR § 734.7` & EU GSN apply to published community tier; academic submissions with intent to publish fall under `15 CFR § 734.7(a)(5)`). | `Apache-2.0` (code & Community Studio), `CDLA-Permissive-2.0` (datasets & model weights), `CERN-OHL-P-2.0` (reference CAD/BOM), `CC-BY-4.0` (docs & GSN). | Full clean-room audit; BIS/NSA `§ 742.15(b)` URL email if any `5D002` confidentiality crypto is introduced; physical hardware remains outside `§ 734.7`. |

- **Fundamental Research Boundary (`15 CFR § 734.8`):** `§ 734.8` covers basic/applied research ordinarily intended for broad publication and never shields pre-existing proprietary EAR-controlled software transferred to a university lab (Note 1 to `§ 734.8(a)`).
- **Cloud Backup Safe Harbor (`15 CFR § 734.18(a)(5)`):** Unclassified private cloud backups exit EAR export/transfer triggers only when protected by true end-to-end encryption (`FIPS 140-2` / `FIPS 140-3` AES-256) with keys held exclusively by the originator and zero intentional storage in Country Group `D:5`. Standard unencrypted cloud storage and ephemeral GPU RAM do not satisfy `§ 734.18(a)(5)` — limit cloud GPU training strictly to clean civil EAR99 perception scripts/datasets (zero legacy defense artifacts and zero hybrid DBRN fusion source).

---

## 2. November 2026 Multi-Jurisdictional Regulatory Baseline

1. **U.S. BIS Repeal of `ECCN 4E091` (Effective November 19, 2026):** The November 19, 2026 BIS Interim Final Rule formally rescinded the January 2025 AI Diffusion Rule and deleted `ECCN 4E091` (closed-weight general AI model controls). Deleting `4E091` removes general closed-weight AI export controls, **but does NOT relax domain-specific aviation, navigation, UAV, thermal, or military end-use controls**.
2. **U.S. BIS UAV Threshold & Part 744 Military End-Use Update (`91 FR 52501`, Effective August 13, 2026):** BIS updated civil UAV national-security endurance thresholds (`9A012.a`) to **3 hours** (eliminating wind-gust tolerance while preserving `MT1` `>= 300 km` range controls) and added `9A012`, `9D001`, `9D002`, `9D004`, and `9E001` to **Supplement No. 2 to Part 744** (`15 CFR § 744.21` military end-use/end-user controls). All Private-First evaluation kits are restricted strictly to **civil `< 3-hour` endurance, `< 300-km` range** airframes and screened recipients (U.S. Consolidated Screening Lists, EU Annex IV, UN embargoes, zero `D:5`/`E:1`/`E:2` destinations).
3. **EU Dual-Use Annex I Update (`Commission Delegated Regulation (EU) 2026/2288`, Effective November 20, 2026):** Replaces the prior Annex I control list under `Regulation (EU) 2021/821`. Private access-restricted transfers do not benefit from the EU General Software Note public-domain exclusion.
4. **GNSS-Denied Navigation Source-Code Warning (`ECCN 7D003.a` & `7D003.b.3` — Highest Exposure for `XVS-Nav`):** Both U.S. CCL and EU Annex I (`(EU) 2026/2288`) control software improving inertial navigation error to `7A003.b` thresholds (`< 10 m` CEP for up to 4 minutes after loss of positioning aids), and **`7D003.b.3` explicitly controls "source code"** for hybrid integrated systems continuously combining heading data with **Database Referenced Navigation (DBRN)**. Pure relative visual odometry on COTS MEMS hardware below `7A003` thresholds typically maps to `7D994` / `EAR99`. **Mandatory Guardrail:** Never transmit private `XVS-Nav` / DBRN / Dual-Bank ES-EKF *source code* cross-border without formal attorney/BIS CCATS classification; clean-room `private_eval` exports enable `--exclude-nav-source` by default, and OEM evaluation relies strictly on compiled non-flight desktop simulation binaries and rendered evaluation video.
5. **Türkiye Strategic Trade Controls (`Law No. 5201` & Ministry of Trade):** Development and cross-border transfers originating in Türkiye are governed in parallel by **Law No. 5201** (*Ministry of National Defence / MSB* & *SSB* defense controlled list — strictly avoided via zero defense/Stratum-A scope) and civil dual-use Wassenaar Category 6/7/9 and catch-all controls administered by the **Turkish Ministry of Trade** (*Ticaret Bakanlığı*). Prior to any cross-border transfer of private navigation source or commercial export (`REL6`), obtain Turkish export-control clearance alongside U.S./EU counsel review.
6. **EU Civil Aviation & Product Liability Cutoffs (Nov 30 & Dec 9, 2026):** EASA **SORA 2.5** transition cutover (`November 30, 2026`, 17 OSOs + `M1A`/`M1B`/`M1C`) and **EU Product Liability Directive (Directive (EU) 2024/2853)** (`December 9, 2026`, `Art. 2(2)`, `Art. 4(1)`, `Art. 4(7)`, `Art. 15`, `Art. 17`) govern every post-cutoff evaluation pack; see `SAFETY_DISCLAIMER.md` and `docs/xvs/governance/NON_FLIGHT_EVAL_LICENSE_TEMPLATE.md`.

---

## 3. Cryptographic Integrity vs. Confidentiality (`ECCN 5D002` & `15 CFR § 742.15(b)`)

Under Category 5, Part 2 (`5A002.a` Technical Notes), *"cryptography for data confidentiality"* **expressly excludes cryptography limited to authentication, digital signatures, and data integrity**:

| Component | Cryptographic Function | Private-First (`private_eval`) Rule | Timed Public Open-Core (`Wave-1` / `Wave-2`) Rule |
| :--- | :--- | :--- | :--- |
| **`xvs-certgen`, Buyer Pack, `XVS-Bench` Ledger** | Pure-stdlib / `ed25519-dalek` **Ed25519** signatures & **SHA-256** digests | **PERMITTED (Outside `5D002`).** Authentication & integrity only; zero data-confidentiality encryption. | **PERMITTED (Outside `5D002`).** Standard authentication/integrity primitives only. |
| **MAVLink 2.0 & `#52001` MeshDAA** | SHA-256 packet signing & Ed25519 verification | **PERMITTED (Read-Only Sim/Loopback Only; Outside `5D002`).** Zero outbound flight actuation (`NON_FLIGHT_SIM_ONLY=1`). | **PERMITTED (Outside `5D002`).** Standard authentication primitives; zero custom crypto. |
| **TLS / QUIC / AES Confidentiality (`rustls`, `openssl`, `quinn`, `wtransport`, AES-256 weight encryption)** | Data-confidentiality encryption (`5D002` trigger) | **PROHIBITED IN PRIVATE BINARIES.** Calling an upstream OSS TLS/AES library inside a private binary creates a new controlled item (`5D002` / `5D992.c`) that does NOT inherit upstream OSS exemptions. Private-First Studio kits MUST stay offline localhost/IPC with zero TLS/AES crates. | **CONTROLLED PUBLIC NOTICE REQUIRED.** If future public Wave-2 bridges add `5D002` confidentiality libraries, follow conservative BIS/ASF practice by filing the one-time public URL email to `crypt@bis.doc.gov` and `enc@nsa.gov` (`15 CFR § 742.15(b)`). Custom/non-standard cryptography remains strictly banned. |

---

## 4. Physical Sensor Hardware & Digital Manufacturing Quarantine

`15 CFR § 734.7` never covers tangible physical hardware. Furthermore, under **Directive (EU) 2024/2853 Art. 4(1)**, automated CAD/CAM **digital manufacturing files** (`.stl`, `.gcode`, `.kicad*`) are strict-liability products and are **hard-banned from Private-First exports** (reserved exclusively for Wave-2 `CERN-OHL-P-2.0` public open-hardware release).

| Subsystem / Reference Sensor | Export / Sanctions Posture | OpenXVS Repository Policy |
| :--- | :--- | :--- |
| **Tier-0B Maker Pod (`DR-15`)** (`RPi 5`, mono `IMX296LLR`, `CIL062`, `SAM-M10Q`, `BNO085`, `Pro Stick Plus`) | Civil COTS (`EAR99`, `5A992.c` NLR, `7A994` NLR; RX-only ADS-B). Zero thermal camera included. | **APPROVED Civil Baseline** (2D specification docs only during Private-First; CAD files deferred to Wave-2 `CERN-OHL-P-2.0`). |
| **Tier-4 Civil LWIR Thermal (`FLIR Boson+ 640×512`)** (`DR-11` / `DR-14`) | Governed by **EAR (NOT ITAR)**. Cores `> 9 Hz` fall under **ECCN `6A003.b.4.b`** (`> 60 Hz`, `> 111,000` elements, or embedded export trigger **RS Column 1**, `§ 743.3` reporting, `§ 744.9` bans, `0A919` zero *de minimis*). Note: `XVS-ThermalNet` (`1.07 MB` ONNX) runs civil inference only and does NOT unlock FPA frame rates (`6D003.c`). | **HARDWARE QUARANTINED.** OpenXVS never ships thermal hardware; builders source `<= 9 Hz` or locally compliant civil cores independently. |
| **OFAC-Sanctioned LWIR (`InfiRay` / Yantai IRay Tech.)** (`Tiny1-C`, `P2 Pro`, `Xinfrared`) | **U.S. OFAC SDN List** (May 1, 2024, E.O. 14024; 50% ownership rule). | **HARD BANNED** across all BOMs, adapters, and code (`assert_thermal_sensor_sanctions_clean`). |
| **Event Sensor (`Sony / Prophesee IMX636`)** (`DR-10` / `DR-18`) | Commercial robotics/automotive sensor; public ECCN sheet unverified (`6A003` high-speed imaging review). | **INTEGRATOR VERIFICATION REQUIRED** prior to physical cross-border pod export. |
| **Geophysical Atomic Magnetometers (`QuSpin QTFM`)** (`DR-18`) | **ECCN `6A006.a.2`** export-controlled sensor. | **REJECTED / EXCLUDED** (`MagNav` architecturally rejected). |
| **Cooperative Traffic RX (`uAvionix pingRX Pro`)** (`DR-11` / `DR-14`) | Verified civil **EAR99**, physically receive-only (`14 CFR § 107.53` compliant). | **APPROVED Civil RX-Only** (zero ADS-B Out transmit paths exist in OpenXVS). |

---

## 5. Strict ITAR / USML & Military End-Use Prohibition (`22 CFR Part 121`, EU CML, TR Law 5201)

OpenXVS **strictly prohibits** any code, weight, dataset, CAD file, or documentation specially designed for defense articles on the **U.S. Munitions List (`22 CFR Part 121` Categories VIII & XII)**, the **EAR 600 Series (`7D611`)**, the **EU Common Military List**, or **Turkish Law No. 5201** defense controlled lists. All external exports must pass the fail-closed clean-room airlock checks.
