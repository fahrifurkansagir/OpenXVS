# OpenXVS Trademark & Brand-Decoupling Policy (`TRADEMARK.md`)

> *Adapted from Dronecode Foundation and Rust Foundation trademark governance guidelines. Engineering governance policy — NOT formal legal advice.*

---

## 1. Purpose, Release-Phase Scope & Brand Decoupling

OpenXVS follows **Model 2 (Staged Split Open-Core)** across distinct release phases:

1. **Current Phase — Private-First Maturation (`private_eval`, Nov 2026 – Sep 2027):** All evaluation kits, compiled binaries, weights, and documentation are distributed confidentially on a **non-public, non-flight evaluation basis** under a bilateral evaluation agreement and Non-Flight Evaluation License. No public open-source license grant is active in Private-First distribution packages.
2. **Planned Wave-1 Formal Core (`wave1_formal_core`, Q1 2027 — Brand-Decoupled):** Formal `#![no_std]` Rust/Kani assurance crates (`xvs-rta-core`, `xvs-active-range`, `xvs-odd-vlm`, `xvs-os-hal`) and `xvs-certgen` are scheduled for public `Apache-2.0` release under a **brand-decoupled repository/package name** to preserve strict double-blind peer-review anonymity for the May 2027 NeurIPS Evaluations & Datasets submission.
3. **Planned Wave-2 Public Open-Core (`public_open_core`, October 2027):** Following NeurIPS 2027 E&D camera-ready publication, the community core will transition to permissive open licenses: `Apache-2.0` (software & pure-OSS Community Studio), `CDLA-Permissive-2.0` (datasets & model weights), `CERN-OHL-P-2.0` (reference hardware CAD/PCB/BOM), and `CC-BY-4.0` (documentation & GSN artifacts).

Neither a Private-First Evaluation License nor future permissive copyright/patent licenses (`Apache-2.0` §6, `CERN-OHL-P-2.0` §7.2) grant trademark rights. In civil aviation, misbranding unverified forks, modified pods, or uncertified avionics under the **OpenXVS** name creates direct flight-safety hazards. This policy governs all use of **OpenXVS**, **OpenXVS-OS**, **OpenXVS Studio**, **XVS-Bench**, and **OpenXVS Live Deck** marks and logos (the **"OpenXVS Marks"**).

---

## 2. Permitted Factual (Nominative) Use

You do not need written permission for accurate, non-misleading factual statements that do not imply official endorsement, partnership, or airworthiness certification (subject to confidentiality obligations if bound by a Private-First NDA):

- Accurate factual attribution in public technical papers, regulatory comments (FAA, EASA, ASTM F38.01, RTCA, EUROCAE WG-114), or build logs once the referenced release is public.
- Forking and redistributing future publicly released open-core modules under your own distinct brand name while retaining required attribution notices (e.g., *"Acme DAA Suite, derived from OpenXVS public core"*).

---

## 3. Prohibited Uses (Strict Safety, Confidentiality & Anti-Misbranding Rules)

Without prior written authorization from the OpenXVS project stewards, you **may not**:

1. **Make False Certification or Airworthiness Claims:** Use **"OpenXVS Certified"**, **"OpenXVS Approved"**, **"Airworthiness Verified by OpenXVS"**, **"DO-178C / ED-324 / SORA 2.5 Certified by OpenXVS"**, or any phrase implying CAA approval or flight-safety certification (`G5_GAP` remains explicitly undeveloped).
2. **Misrepresent Private-First Evaluation Kits as Public Open Source:** Redistribute confidential `private_eval` binaries, documentation, or Buyer Packs under `Apache-2.0` or imply public open-source release prior to the official Wave-1 / Wave-2 public flip.
3. **Brand Modified Derivatives as Official OpenXVS:** Incorporate any OpenXVS Mark into the primary name of a commercial product, company, hardware SKU, application, domain name, or social-media handle suggesting official release status.
4. **Compromise Double-Blind Review Anonymity:** Link brand-decoupled Wave-1 formal crates to anonymized NeurIPS 2027 E&D review artifacts during the May–September 2027 double-blind window.
5. **Display Altered Logos or Unverified Badges:** Modify OpenXVS logos or display benchmark/conformance badges lacking a valid cryptographic Ed25519 receipt.
6. **Apply Marks to Prohibited Domains:** Use any OpenXVS Mark in connection with military weaponization, targeting, defense fire-control, electronic countermeasures, or non-civil systems prohibited by `PROJECT_CHARTER.md` §2.

---

## 4. Verifiable `XVS-Bench` & Buyer-Pack Ed25519 Receipts

Conformance or evaluation verification receipts (`MANIFEST.sha256.sig` and future public `XVS-Bench` ledger badges) are valid **exclusively** when accompanied by an unmodified, verifiable **RFC 8032 Ed25519 signature** binding exact artifact SHA-256 digests, SPDX 2.3 SBOM hashes, and Croissant 1.1 + RAI provenance. Any modification to weights, thresholds, or runtime code invalidates the receipt.

---

## 5. Future Legal Stewardship Note (`REL6`)

Upon post-2027 commercial entity incorporation (`REL6`), any trademark registration by an operating company will be filed strictly as a standard word/service mark rather than a U.S. Certification Mark (respecting the Lanham Act anti-use rule, `15 U.S.C. § 1064(5)`). Any future multi-vendor conformance certification mark will be vested solely in a neutral non-profit foundation independent of commercial product sales.
