# OpenXVS v2.0 — `ED-324` (`SAE ARP6983`) GSN v3 Evidence Register

- **Standard:** `SCSC-141C / EUROCAE ED-324 (SAE ARP6983) GSN v3`
- **Root Goal:** `G0`
- **Traceability Graph:** `assurance/openxvs_v2.gsn.yaml` | `assurance/openxvs_v2_gsn.mmd` | `assurance/evidence_trace.json`

## 1. Bound Solution Nodes (`Sn*` — 100% Real On-Disk SHA-256 Artifacts)

| Solution ID | Artifact Path | Size (B) | SHA-256 |
| :--- | :--- | ---: | :--- |
| `Sn1_MHVIT` | `output/xvs/v2_m3/v1_xvs_rt_evidence.json` | 3121 | `7ec4638bb82dfb82f6f0dd478f4197864a863d1d560459cdec3d16bd1eb2e472` |
| `Sn1_MHVIT` | `output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json` | 1054 | `70f632ae626dd7721c90b1a2e36fb10720637de826b902e0802f8f63a790a384` |
| `Sn2_HYBRID_CBF` | `output/xvs/v2_m3/v1_xvs_rt_evidence.json` | 3121 | `7ec4638bb82dfb82f6f0dd478f4197864a863d1d560459cdec3d16bd1eb2e472` |
| `Sn2_HYBRID_CBF` | `output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json` | 1054 | `70f632ae626dd7721c90b1a2e36fb10720637de826b902e0802f8f63a790a384` |
| `Sn3_DAQP_WCET` | `output/xvs/v2_m3/v1_xvs_rt_evidence.json` | 3121 | `7ec4638bb82dfb82f6f0dd478f4197864a863d1d560459cdec3d16bd1eb2e472` |
| `Sn5_KANI` | `output/xvs/v2_m1/kani_proofs.log` | 1347680 | `99ed76777e33c80535b3f420d2b4ed1370d0f3a5a98fb81b790640c00987c967` |
| `Sn6_PARITY` | `output/xvs/v2_m1/parity_results.json` | 1245 | `76e032b0ebc7b46fd6e88de36fee3a2a0ed2a01e5ae4acb71ace1d7320f8b710` |
| `Sn6_PARITY` | `output/xvs/v2_m1/e2e_results.json` | 21694 | `fe5b071fee472d8d30f6d1921b377c1f603c661bc95c2ffe64c856b83908f49c` |
| `Sn7_DTACI_OOD` | `output/xvs/v2_m2/v3_lwir_thermal_world_evidence.json` | 2076 | `8a54a1c397a6b95693084dae0940d5326c2c5ab71badae0551566061dffb1bf2` |
| `Sn7_DTACI_OOD` | `output/xvs/v2_m2/v5_nav_vraim_evidence.json` | 1373 | `1c3c1cb87e8368cd2c3fabc614aeecab3354814932ee234343e32de37c3e2d0b` |
| `Sn8_SPLIT_H3` | `output/xvs/v2_m2/v4_wp01b_land_evidence.json` | 1685 | `20eb12a8b9a00eb07d576d817ba0e782644d4f0d830e9d0e4d2091d8949f8e8a` |
| `Sn8_SPLIT_H3` | `output/xvs/v2_m2/v5_nav_vraim_evidence.json` | 1373 | `1c3c1cb87e8368cd2c3fabc614aeecab3354814932ee234343e32de37c3e2d0b` |
| `Sn9_PROVENANCE` | `docs/xvs/reports/V2_M1_livedeck_report.md` | 7377 | `584fb9eb992904c20268a1615ba1ece36371f1e065f4f3ecbe902a76bd278f0b` |
| `Sn9_PROVENANCE` | `docs/xvs/reports/V2_M2_local_prep_report.md` | 8069 | `a865806505568f1a5afb0dd2d903df7d8095561ad25582ecc038b39ad4f1b27c` |
| `Sn9_PROVENANCE` | `output/xvs/v2_m3/xvs_bench_chain.json` | 7101 | `8954130b24317bd6c819c9ed36a50246c8794c9353ff9378af3f45e17b94ce60` |

## 2. Explicit Undeveloped Goals (`undeveloped: true` — Zero Synthetic Solutions)

| Goal ID | Status | Scope Statement |
| :--- | :--- | :--- |
| `G2_TLA_DEFERRED` | `UNDEVELOPED` | Full TLA+ temporal model checking of distributed multi-ship Simplex RTA liveness properties is explicitly deferred (undeveloped); local safety invariants are currently bounded by 5/5 Kani/CBMC proofs (Sn5_KANI) and 100k proptest cases. |
| `G5_GAP` | `UNDEVELOPED` | EXPLICIT UNDEVELOPED CERTIFICATION GAP (NOT CLAIMED BY OPENXVS): RTCA DO-254 complex electronic hardware assurance, DAL-A/B qualified compiler & RTCA DO-330 tool qualification, ARINC 653 certified RTOS partitioning, airframe-level SAE ARP4754B FHA/PSSA/SSA, and independent human DER/CVE verification sign-off (EASA QA-01 / RTCA DO-178C Table A-7). |
