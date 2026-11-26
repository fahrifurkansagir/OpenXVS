# Security Policy (`SECURITY.md`)

## 1. Non-Flight & Zero-Actuation Safety Boundary

**OpenXVS `v1.1.1` is a civil and dual-use autonomous aviation research, verification, and software-in-the-loop (SIL) suite — NOT a certified flight-control or airworthiness product.**

- All evaluation binaries (`openxvs-studio.exe`, `openxvs-edge-infer.exe`) compile with **`NON_FLIGHT_SIM_ONLY=1`** and **`ACTUATOR_AUTHORITY=false`**.
- Target hardware qualification (`RTCA DO-254`), environmental qualification (`RTCA DO-160G`), compiler/RTOS qualification (`RTCA DO-330` / `ARINC 653`), airframe safety assessment (`SAE ARP4754B`), and independent `DER/CVE` verification remain outside this release's scope ([`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)).
- **Never couple evaluation builds to a live aircraft, autopilot actuator, or flight-critical bus.**

---

## 2. Supported Releases

| Release | Status | Notes |
| :--- | :--- | :--- |
| **`v1.1.1`** | Supported | Active clean-room release (async lock-split Studio & Edge CLI, 3D tactical symbology polish, `6/6` self-contained bundles, verified `ArduPlane 4.7.3` SIL) |
| `v1.1.0` | Prior | Superseded by `v1.1.1` for async RTA isolation, ONNX performance, and self-contained CLI verification |
| `< v1.1.0` | Unsupported | Upgrade to `v1.1.1` for MAVLink 2.0 / HIL security hardening and Rust runtime safety fixes |

---

## 3. Reporting a Vulnerability

Please report security vulnerabilities or safety-invariant regressions **privately** rather than opening a public issue:

1. **Preferred Channel:** Open a private **GitHub Security Advisory** on `fahrifurkansagir/OpenXVS` (`Security -> Advisories -> Report a vulnerability`), or contact the repository maintainer **`@fahrifurkansagir`** directly.
2. **What to Include:**
   - Affected component (`xvs-rta-core`, `xvs-active-range`, `xvs-multiship-coord`, `apps/studio`, `docker/edge-infer`, `src/xvs/v2/xvs_mavlink`, `src/xvs/v2/studio_bridge`, or release binary SHA-256).
   - Minimal deterministic reproducer (input vector, NDJSON flight tape, MAVLink 2.0 frame bytes, or malformed ONNX/sensor payload).
   - Observed vs. expected fail-closed behavior.
3. **Response Target:** Initial acknowledgement within **72 hours** and a root-cause assessment with a regression test before any public disclosure.

---

## 4. Security & Cryptographic Integrity Baseline

- **Zero-GPL Clean-Room Stack:** Strictly `0` GPL / AGPL / LGPL-static runtime dependencies (`pymavlink` and `PyInstaller` excluded).
- **MAVLink 2.0 & Telemetry Hardening:** Mandatory MAVLink 2.0 signature verification when a key is configured (`require_signature=True`), per-message-ID `time_boot_ms` anti-replay monotonicity, fail-closed `DatumAlignmentGuard` and `StaleTelemetryError` guards, altitude/heading bounds validation, and `65,535 B` UDP RX DoS protection.
- **Tamper-Evident Flight Recorder:** SHA-256 hash-chained NDJSON shadow tapes verified identically in Python and Pure-Rust (`openxvs-edge-infer --replay`, failing closed with exit code `1` on any 1-byte payload mutation).
- **Authentication-Only Cryptography (`ECCN 5D002`):** Pure **RFC 8032 Ed25519** signatures and **SHA-256** integrity digests only (`0` TLS/QUIC/OpenSSL/AES confidentiality crypto crates in `Cargo.lock`).
- **Release Artifact Verification:** Verify every downloaded release bundle and demonstration video against `SHA256SUMS.txt` attached to GitHub Release `v1.1.1`.
