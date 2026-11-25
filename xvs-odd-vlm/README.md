# `xvs-odd-vlm` — `#![no_std]` `#![forbid(unsafe_code)]` Optical-ODD & VLM Safety Gate

Standalone, zero-dependency `#![no_std]` Rust safety crate implementing:
- **DR-13 §1.3 (`OpticalOddMonitor`):** Zero-ML 5-mode deterministic optical-ODD health monitor with `O(1)` correlated-block `M-of-N` MLTL ring counters and hard compile-time AF447 airdata isolation (`AIRDATA_EXPORT_BLOCKED = true`).
- **DR-09 §4 (`AsymmetricSimplexVlmGate`):** 6-stage DO-178C DAL B/C verification gate isolating an IDAL-D/E VLM peripheral (`ACTUATOR_AUTHORITY = false`), 1:1 wire/enum-aligned with `configs/mavlink/pod_daa.xml` and `src/xvs/v2/xvs_vlm_gate/gate.py` (32-byte little-endian frame + IEEE 802.3 CRC32).

## Verification Commands (WSL2 Ubuntu)

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd "<OPENXVS_WORKSPACE>/xvs-odd-vlm"
cargo test
cargo kani
```

## Formal Proof Ledger (`Kani Rust Verifier 0.68.0` / `CBMC 6.11.0`, WSL2 Ubuntu)

| # | Kani Proof Harness | Result | Formal Property Proved |
| :-: | :--- | :---: | :--- |
| 1 | `proof_vlm_monotonic_pessimistic_union_and_zero_actuator_auth` | `VERIFICATION:- SUCCESSFUL` | For all symbolic `cert_hazards_mask`, packet fields & `now_ms`: `(final_hazards_mask & cert_hazards_mask) == cert_hazards_mask` and `!actuator_authority`. |
| 2 | `proof_vlm_watchdog_enforces_offline_on_stale_or_non_monotonic` | `VERIFICATION:- SUCCESSFUL` | Any stale (`age > 200 ms`), future, or non-monotonic `seq` packet strictly yields `GateStatus::VlmOffline` and `accepted_vlm_mask == 0`. |
| 3 | `proof_vlm_optical_odd_or_conformal_abstain_preserves_cert_hazards` | `VERIFICATION:- SUCCESSFUL` | Optical-ODD fault (`!optical_odd_ok`) or `conformal_valid == 0` suppresses unverified VLM veto while preserving every certified geometric hazard. |
| 4 | `proof_optical_odd_af447_and_fog_thermal_handoff` | `VERIFICATION:- SUCCESSFUL` | Unconstrained symbolic 5-mode readings preserve `airdata_export_blocked == true` with zero panic/overflow, and persistent severe fog (`MOR < 1609 m`, healthy LWIR) routes to `OddAction::ThermalPrimary`. |
| 5 | `proof_flz_relative_ranking_subset_and_unique` | `VERIFICATION:- SUCCESSFUL` | For all symbolic `reachable_mask: u16` and `ranked_ids: [u8; 4]`, every returned FLZ ID is `< 16`, strictly in `reachable_mask`, and duplicate-free. |

- **Verified Summary:** `cargo test` -> `3 passed; 0 failed` (including bit-exact Python `zlib.crc32` & 32-byte wire vector parity); `cargo kani` -> `Complete - 5 successfully verified harnesses, 0 failures, 5 total`.
