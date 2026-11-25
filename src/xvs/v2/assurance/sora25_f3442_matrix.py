"""Additive EASA SORA 2.5, ASTM F3442-26 & EUROCAE ED-324 Traceability Matrix + SPDX 2.3 SBOM.

Strictly additive module — performs ZERO edits to frozen `assurance/openxvs_v2.gsn.yaml` or
any frozen Rust crate. Reads the frozen GSN v3 graph read-only, preserves `G5_GAP` and
`G2_TLA_DEFERRED` as explicitly `UNDEVELOPED`, maps all 17 EASA SORA 2.5 OSOs + M1(A)/M1(B)/M1(C)
ground-risk mitigations, ASTM F3442-26 mixed DAA requirements, and EUROCAE ED-324 objectives,
and generates a deterministic SPDX 2.3 JSON SBOM auditing Cargo + Python dependencies with
strict 0 GPL/AGPL/LGPL-static enforcement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
FROZEN_GSN_PATH = REPO_ROOT / "assurance" / "openxvs_v2.gsn.yaml"
CERTGEN_SCRIPT_PATH = REPO_ROOT / "scripts" / "xvs" / "xvs_certgen.py"
REQUIREMENTS_LOCK_PATH = REPO_ROOT / "requirements-xvs.lock.txt"

RUST_CRATE_MANIFESTS: tuple[str, ...] = (
    "xvs-core/Cargo.toml",
    "xvs-active-range/Cargo.toml",
    "xvs-odd-vlm/Cargo.toml",
    "xvs-os-hal/Cargo.toml",
)

# Canonical permissive license registry for locked Python packages whose wheel metadata
# uses free-text banners or is omitted in minimal offline environments.
_PYTHON_KNOWN_SPDX_FALLBACK: dict[str, str] = {
    "colorama": "BSD-3-Clause",
    "contourpy": "BSD-3-Clause",
    "cycler": "BSD-3-Clause",
    "dataclasses-json": "MIT",
    "deprecated": "MIT",
    "flatbuffers": "Apache-2.0",
    "fonttools": "MIT",
    "greenlet": "MIT",
    "iniconfig": "MIT",
    "kiwisolver": "BSD-3-Clause",
    "marshmallow": "MIT",
    "matplotlib": "PSF-2.0",
    "mypy-extensions": "MIT",
    "mypy_extensions": "MIT",
    "numpy": "BSD-3-Clause",
    "onnxruntime": "MIT",
    "opencv-python-headless": "Apache-2.0",
    "packaging": "Apache-2.0 OR BSD-2-Clause",
    "pillow": "MIT-CMU",
    "playwright": "Apache-2.0",
    "pluggy": "MIT",
    "protobuf": "BSD-3-Clause",
    "pyee": "MIT",
    "pygltflib": "MIT",
    "pygments": "BSD-2-Clause",
    "pyparsing": "MIT",
    "pytest": "MIT",
    "python-dateutil": "Apache-2.0 OR BSD-3-Clause",
    "scipy": "BSD-3-Clause",
    "six": "MIT",
    "typing-inspect": "MIT",
    "typing-extensions": "PSF-2.0",
    "typing_extensions": "PSF-2.0",
    "wrapt": "BSD-2-Clause",
}

FORBIDDEN_PACKAGE_NAMES: frozenset[str] = frozenset(
    {
        "pymavlink",
        "pyinstaller",
        "pyinstaller-hooks-contrib",
        "salad",
        "jsbsim",
        "ardupilot-gazebo",
        "v4l2loopback",
    }
)

# Forbidden copyleft / non-commercial / proprietary tokens inside a required SPDX branch
_FORBIDDEN_LICENSE_TOKEN_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:A?GPL|LGPL|SSPL|EUPL|CPAL|OSL|CC-BY-NC|CC-BY-SA|APPLE-ASCL|PROPRIETARY)(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# EASA SORA 2.5 (All 17 OSOs + M1(A)/M1(B)/M1(C)), ASTM F3442-26 & ED-324
# ---------------------------------------------------------------------------

EASA_SORA_25_OSOS: tuple[dict[str, Any], ...] = (
    {
        "oso_id": "OSO#03",
        "title": "UAS maintained by competent and/or proven entity",
        "responsibility": "OPERATOR_OEM",
        "openxvs_coverage": "UNDEVELOPED_GAP",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G5_GAP"],
        "openxvs_artifacts": ["SAFETY_DISCLAIMER.md"],
        "gap_status": "G5_GAP: Organizational maintenance/CAMO procedures are strictly operator/OEM owned.",
    },
    {
        "oso_id": "OSO#05",
        "title": "UAS is designed considering system safety and reliability",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_FORMAL",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G0", "G1", "G2", "Sn5_KANI", "G5_GAP"],
        "openxvs_artifacts": [
            "xvs-core/crates/xvs-rta-core/src/proofs.rs",
            "xvs-active-range/src/kani_proofs.rs",
            "xvs-odd-vlm/src/lib.rs",
            "xvs-os-hal/src/lib.rs",
        ],
        "gap_status": "G5_GAP: 18/18 Kani proofs verify software invariants only; DO-254/ARP4754B/DO-330 remain open.",
    },
    {
        "oso_id": "OSO#06",
        "title": "C3 link performance is appropriate for the operation",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_DESIGN",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G1", "Sn1_MHVIT", "G5_GAP"],
        "openxvs_artifacts": [
            "configs/mavlink/pod_daa.xml",
            "src/xvs/v2/xvs_fleet/governor.py",
        ],
        "gap_status": "G5_GAP: Adaptive #52001 bandwidth governor (<20% UART); physical RF/C2 link qualification is operator/OEM owned.",
    },
    {
        "oso_id": "OSO#07",
        "title": "Inspection of the UAS (product inspection) to ensure consistency with the ConOps",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_PROVENANCE",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G4", "Sn9_PROVENANCE", "G5_GAP"],
        "openxvs_artifacts": [
            "scripts/xvs/xvs_certgen.py",
            "src/xvs/v2/assurance/sora25_f3442_matrix.py",
        ],
        "gap_status": "G5_GAP: Software SHA-256/Ed25519/SPDX provenance verified; physical airframe inspection is operator owned.",
    },
    {
        "oso_id": "OSO#08",
        "title": "Operational procedures are defined, validated and adhered to",
        "responsibility": "OPERATOR_OEM",
        "openxvs_coverage": "UNDEVELOPED_GAP",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["C1", "G5_GAP"],
        "openxvs_artifacts": ["SAFETY_DISCLAIMER.md", "COMPLIANCE_AND_DUAL_USE.md"],
        "gap_status": "G5_GAP: Flight operations manual and airspace procedures are exclusively operator owned.",
    },
    {
        "oso_id": "OSO#09",
        "title": "Remote crew trained and current and able to control the abnormal situation",
        "responsibility": "OPERATOR_OEM",
        "openxvs_coverage": "UNDEVELOPED_GAP",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G5_GAP"],
        "openxvs_artifacts": ["SAFETY_DISCLAIMER.md"],
        "gap_status": "G5_GAP: Crew licensing, currency, and abnormal-situation training are strictly operator owned.",
    },
    {
        "oso_id": "OSO#11",
        "title": "Procedures are in place to handle the deterioration of external systems supporting UAS operations",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_RESILIENCE",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G3", "Sn8_SPLIT_H3", "G5_GAP"],
        "openxvs_artifacts": [
            "output/xvs/v2_m2/v5_nav_vraim_evidence.json",
            "src/xvs/v2/xvs_forcedland/planner.py",
        ],
        "gap_status": "G5_GAP: GNSS-denied navigation & 6-stage aliasing gate reduce GNSS dependency; operational contingency approval remains open.",
    },
    {
        "oso_id": "OSO#13",
        "title": "External services supporting UAS operations are adequate for the operation",
        "responsibility": "OPERATOR_OEM",
        "openxvs_coverage": "PARTIAL_DATA_PROVENANCE",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G4", "G5_GAP"],
        "openxvs_artifacts": ["configs/xvs/data_allowlist.json"],
        "gap_status": "G5_GAP: Terrain/DEM/ADS-B data provenance validated; live UTM/U-Space/ANSP SLAs are operator owned.",
    },
    {
        "oso_id": "OSO#16",
        "title": "Multi-crew coordination",
        "responsibility": "OPERATOR_OEM",
        "openxvs_coverage": "UNDEVELOPED_GAP",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G5_GAP"],
        "openxvs_artifacts": ["docs/xvs/research_v2/reports/DR-17-civil-fleet-utm-uspace-and-1tomany-gcs-hcd.md"],
        "gap_status": "G5_GAP: Multi-crew coordination SOPs are strictly operator owned.",
    },
    {
        "oso_id": "OSO#17",
        "title": "Remote crew is fit to operate",
        "responsibility": "OPERATOR_OEM",
        "openxvs_coverage": "UNDEVELOPED_GAP",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G5_GAP"],
        "openxvs_artifacts": ["SAFETY_DISCLAIMER.md"],
        "gap_status": "G5_GAP: Crew fitness and fatigue management are strictly operator owned.",
    },
    {
        "oso_id": "OSO#18",
        "title": "Automatic protection of the flight envelope from human error",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_FORMAL",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G1", "G2", "Sn2_HYBRID_CBF", "Sn5_KANI", "G2_TLA_DEFERRED", "G5_GAP"],
        "openxvs_artifacts": [
            "xvs-core/crates/xvs-rta-core/src/proofs.rs",
            "xvs-active-range/src/kani_proofs.rs",
            "xvs-os-hal/src/lib.rs",
        ],
        "gap_status": "G2_TLA_DEFERRED & G5_GAP: Simplex RTA & SC3BF bounds formally checked by Kani; multi-ship TLA+ liveness & certified autopilot integration remain open.",
    },
    {
        "oso_id": "OSO#19",
        "title": "Safe recovery from human error",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_FORMAL",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G2", "G3", "Sn5_KANI", "Sn8_SPLIT_H3", "G5_GAP"],
        "openxvs_artifacts": [
            "xvs-core/crates/xvs-rta-core/src/proofs.rs",
            "output/xvs/v2_m2/v4_wp01b_land_evidence.json",
        ],
        "gap_status": "G5_GAP: Non-resettable mandatory Go-Around latch & staged recovery dwell verified in software; airborne override/pilot recovery procedures remain operator owned.",
    },
    {
        "oso_id": "OSO#20",
        "title": "A Human Factors evaluation has been performed and the HMI found appropriate for the mission",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_BENCH_PROTOTYPE",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["Sn6_PARITY", "G5_GAP"],
        "openxvs_artifacts": ["docs/xvs/reports/V2_M1_livedeck_report.md"],
        "gap_status": "G5_GAP: Deterministic HUD & Studio eval UI provided; formal CAA Human Factors evaluation is OEM/operator owned.",
    },
    {
        "oso_id": "OSO#21",
        "title": "Operational procedures are validated against standards",
        "responsibility": "OPERATOR_OEM",
        "openxvs_coverage": "UNDEVELOPED_GAP",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G5_GAP"],
        "openxvs_artifacts": ["SAFETY_DISCLAIMER.md"],
        "gap_status": "G5_GAP: Operational validation against NAA/CAA standards is strictly operator owned.",
    },
    {
        "oso_id": "OSO#22",
        "title": "The remote crew is trained to identify critical environmental conditions and to avoid them",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_ODD_ADVISORY",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G3", "Sn7_DTACI_OOD", "G5_GAP"],
        "openxvs_artifacts": [
            "xvs-odd-vlm/src/lib.rs",
            "src/xvs/v2/xvs_optical_odd/monitor.py",
        ],
        "gap_status": "G5_GAP: 5-mode Optical-ODD monitor flags degraded optical/thermal conditions; crew weather training is operator owned.",
    },
    {
        "oso_id": "OSO#23",
        "title": "Environmental conditions for safe operations are defined, measurable and adhered to",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_ODD_MONITOR",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["C1", "G3", "Sn7_DTACI_OOD", "G5_GAP"],
        "openxvs_artifacts": [
            "xvs-odd-vlm/src/lib.rs",
            "src/xvs/v2/xvs_optical_odd/monitor.py",
            "output/xvs/v2_m2/v3_lwir_thermal_world_evidence.json",
        ],
        "gap_status": "G5_GAP: Zero-ML Optical-ODD monitor (<1 ms WCET) bounds optical ODD; physical DO-160G/H environmental qualification remains open.",
    },
    {
        "oso_id": "OSO#24",
        "title": "UAS is designed and qualified for adverse environmental conditions",
        "responsibility": "SHARED_OPENXVS_SUBSTRATE",
        "openxvs_coverage": "PARTIAL_SOFTWARE_DVE",
        "sail_target": "SAIL II-III",
        "gsn_nodes": ["G3", "Sn7_DTACI_OOD", "G5_GAP"],
        "openxvs_artifacts": [
            "output/xvs/v2_m2/v3_lwir_thermal_world_evidence.json",
            "src/xvs/v2/xvs_ast/falsifier.py",
        ],
        "gap_status": "G5_GAP: EO/LWIR physical-health gating & AST weather falsification verified in software; physical DO-160G/H & DO-254 qualification remain open.",
    },
)

EASA_SORA_25_MITIGATIONS: tuple[dict[str, Any], ...] = (
    {
        "mitigation_id": "M1(A)",
        "category": "Ground Risk Mitigation (SORA 2.5)",
        "title": "Sheltering",
        "openxvs_coverage": "OPERATOR_OWNED",
        "gsn_nodes": ["G5_GAP"],
        "notes": "Operational population sheltering credit is solely established by the UAS operator/NAA under SORA 2.5.",
    },
    {
        "mitigation_id": "M1(B)",
        "category": "Ground Risk Mitigation (SORA 2.5)",
        "title": "Operational Restrictions (Temporal/Spatial Population Reduction)",
        "openxvs_coverage": "OPERATOR_OWNED",
        "gsn_nodes": ["C1", "G5_GAP"],
        "notes": "Time-of-day / corridor population restrictions are operator owned; OpenXVS enforces civil ODD bounds only.",
    },
    {
        "mitigation_id": "M1(C)",
        "category": "Ground Risk Mitigation (SORA 2.5)",
        "title": "Ground Observation",
        "openxvs_coverage": "PARTIAL_ADVISORY_ONLY",
        "gsn_nodes": ["G3", "Sn8_SPLIT_H3", "G5_GAP"],
        "notes": "XVS-Land / XVS-ForcedLand / XVS-ThermalNet supply optical/thermal surface hazard perception; NAA operational credit requires full airframe validation (G5_GAP).",
    },
)

ASTM_F3442_26_REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "req_id": "F3442-26-DAA-WELL-CLEAR",
        "title": "Mixed Cooperative/Non-Cooperative DAA Well Clear (DWC) & NMAC Prevention (ARC-a/b/c)",
        "status": "PARTIAL_SIM_AND_FORMAL",
        "gsn_nodes": ["G1", "Sn1_MHVIT", "Sn2_HYBRID_CBF", "G5_GAP"],
        "evidence_artifacts": [
            "output/xvs/v2_m3/v1_xvs_rt_evidence.json",
            "output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json",
        ],
        "gap_status": "0 NMAC in 500 seeded civil encounters; physical flight-test validation & NAA approval remain in G5_GAP.",
    },
    {
        "req_id": "F3442-26-NON-COOP-RANGE",
        "title": "Non-Cooperative Monocular/EO/LWIR Detection & Active Range Observability",
        "status": "PARTIAL_REALSKY_AND_FORMAL",
        "gsn_nodes": ["G1", "G3", "Sn7_DTACI_OOD", "G5_GAP"],
        "evidence_artifacts": [
            "xvs-active-range/src/kani_proofs.rs",
            "docs/xvs/reports/V3_M1_active_range_benchmark.md",
            "docs/xvs/reports/V3_M1_aot_realsky_proof.md",
            "tests/golden/active_range_vectors.json",
        ],
        "gap_status": "4/4 Kani proofs + real Amazon AOT proof (97.5% coverage); physical pod DO-254 & flight capture remain in G5_GAP.",
    },
    {
        "req_id": "F3442-26-COOP-MIXED",
        "title": "Cooperative ADS-B RX + XVS-MeshDAA Bearing Track Fusion & Advisory Interface",
        "status": "PARTIAL_LOOPBACK_AND_FORMAL",
        "gsn_nodes": ["G1", "Sn1_MHVIT", "G5_GAP"],
        "evidence_artifacts": [
            "configs/mavlink/pod_daa.xml",
            "xvs-os-hal/src/lib.rs",
        ],
        "gap_status": "4/4 Kani proofs on xvs-os-hal & signed #52001 loopback; RF hardware certification remains in G5_GAP.",
    },
    {
        "req_id": "F3442-26-RTA-MANEUVER",
        "title": "Bounded Fixed-Wing Avoidance Authority & Fail-Closed Simplex RTA Containment",
        "status": "PARTIAL_FORMAL_WITH_TLA_GAP",
        "gsn_nodes": ["G2", "Sn5_KANI", "Sn6_PARITY", "G2_TLA_DEFERRED", "G5_GAP"],
        "evidence_artifacts": [
            "xvs-core/crates/xvs-rta-core/src/proofs.rs",
            "output/xvs/v2_m1/kani_proofs.log",
            "output/xvs/v2_m1/parity_results.json",
        ],
        "gap_status": "5/5 Kani proofs + 10k parity verified; distributed multi-ship TLA+ liveness open in G2_TLA_DEFERRED; autopilot cert in G5_GAP.",
    },
    {
        "req_id": "F3442-26-ODD-HEALTH",
        "title": "Continuous Optical/Thermal Sensor Health Monitoring & Degraded-Mode Fail-Closed Alerting",
        "status": "PARTIAL_FORMAL",
        "gsn_nodes": ["G3", "Sn7_DTACI_OOD", "G5_GAP"],
        "evidence_artifacts": [
            "xvs-odd-vlm/src/lib.rs",
            "src/xvs/v2/xvs_optical_odd/monitor.py",
            "src/xvs/v2/xvs_vlm_gate/gate.py",
        ],
        "gap_status": "5/5 Kani proofs on xvs-odd-vlm; physical environmental DO-160G/H qualification remains in G5_GAP.",
    },
    {
        "req_id": "F3442-26-F3705-VERIFICATION",
        "title": "ASTM F3705-25 Aligned Simulation, Reproducibility & Cryptographic Evidence Chain",
        "status": "PARTIAL_SIM_AND_PROVENANCE",
        "gsn_nodes": ["G4", "Sn9_PROVENANCE", "G5_GAP"],
        "evidence_artifacts": [
            "output/xvs/v2_m3/xvs_bench_chain.json",
            "scripts/xvs/xvs_certgen.py",
        ],
        "gap_status": "Ed25519 hash chain & deterministic SITL proofs verified; physical flight-test campaign remains in G5_GAP.",
    },
)

ED324_OBJECTIVES: tuple[dict[str, Any], ...] = (
    {
        "objective_id": "ED-324-OBJ-1",
        "title": "AI/ML Operational Design Domain (ODD) & ConOps Specification",
        "dal_ceiling": "IDAL C (ML offline supervised) / IDAL D (VLM advisory veto)",
        "status": "PARTIAL_LOCAL_SPEC",
        "gsn_nodes": ["G0", "C1", "C2", "G5_GAP"],
        "gap_status": "G5_GAP: Authority ConOps review & airframe-level ARP4754B FHA/PSSA remain open.",
    },
    {
        "objective_id": "ED-324-OBJ-2",
        "title": "Data Management, License Hygiene & Split Independence",
        "dal_ceiling": "IDAL C",
        "status": "PARTIAL_LOCAL_AUTOMATED",
        "gsn_nodes": ["G4", "Sn9_PROVENANCE", "G5_GAP"],
        "gap_status": "G5_GAP: Permissive allowlist & provenance enforced; qualified production dataset governance & QA-01 sign-off remain open.",
    },
    {
        "objective_id": "ED-324-OBJ-3",
        "title": "Learning Process Verification, OOD Bounds & Conformal Calibration (R-LM-07..11)",
        "dal_ceiling": "IDAL C",
        "status": "PARTIAL_LOCAL_AUTOMATED",
        "gsn_nodes": ["G3", "Sn7_DTACI_OOD", "Sn8_SPLIT_H3", "G5_GAP"],
        "gap_status": "G5_GAP: Offline Mahalanobis/SAE/AST & split-conformal bounds verified; DO-330 tool qualification remains open.",
    },
    {
        "objective_id": "ED-324-OBJ-4",
        "title": "Machine-Checkable GSN v3 Assurance Case with Explicit Undeveloped Boundaries",
        "dal_ceiling": "IDAL C",
        "status": "VERIFIED_WITH_EXPLICIT_GAPS",
        "gsn_nodes": ["G0", "S1", "G5_GAP", "G2_TLA_DEFERRED"],
        "gap_status": "G5_GAP & G2_TLA_DEFERRED enforced fail-closed by xvs_certgen.py (zero synthetic gap closure).",
    },
    {
        "objective_id": "ED-324-OBJ-5",
        "title": "Deterministic Runtime Assurance (RTA) Containment & Zero-Allocation RT Execution",
        "dal_ceiling": "DAL B/C Software Substrate (Non-Certified)",
        "status": "PARTIAL_FORMAL_18_KANI",
        "gsn_nodes": ["G1", "G2", "Sn3_DAQP_WCET", "Sn5_KANI", "Sn6_PARITY", "G2_TLA_DEFERRED", "G5_GAP"],
        "gap_status": "G2_TLA_DEFERRED & G5_GAP: 18/18 Kani proofs + 0 B heap allocation; qualified compiler (DO-330) & ARINC 653 RTOS remain open.",
    },
    {
        "objective_id": "ED-324-OBJ-6",
        "title": "Verification Independence (DO-178C Table A-7 / EASA QA-01 Precursor)",
        "dal_ceiling": "Precursor Only",
        "status": "UNDEVELOPED_GAP",
        "gsn_nodes": ["G5_GAP"],
        "gap_status": "G5_GAP: Automated D16 audits are strictly precursors; independent human DER/CVE / QA-01 sign-off remains open.",
    },
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_certgen_bundle(repo_root: Path) -> dict[str, Any]:
    certgen_path = repo_root / "scripts" / "xvs" / "xvs_certgen.py"
    gsn_path = repo_root / "assurance" / "openxvs_v2.gsn.yaml"
    spec = importlib.util.spec_from_file_location("openxvs_certgen_for_matrix", str(certgen_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load xvs_certgen.py from {certgen_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.synthesize_assurance_bundle(gsn_path=gsn_path, repo_root=repo_root)


def build_sora25_f3442_ed324_matrix(
    *,
    repo_root: Path = REPO_ROOT,
    release_mode: str = "private_eval",
) -> dict[str, Any]:
    """Build deterministic machine-readable EASA SORA 2.5 + ASTM F3442-26 + ED-324 matrix.

    Strictly read-only on frozen `assurance/openxvs_v2.gsn.yaml`. Fails closed if either
    `G5_GAP` or `G2_TLA_DEFERRED` is not present as an undeveloped node in the live GSN bundle.
    """
    root = repo_root.resolve()
    certgen = _load_certgen_bundle(root)
    undeveloped_nodes = list(certgen.get("gsn_source", {}).get("undeveloped_nodes", []))
    errors: list[str] = list(certgen.get("errors", []))

    for mandatory_gap in ("G5_GAP", "G2_TLA_DEFERRED"):
        if mandatory_gap not in undeveloped_nodes:
            errors.append(f"MANDATORY_GAP_NOT_UNDEVELOPED: {mandatory_gap}")

    if len(EASA_SORA_25_OSOS) != 17:
        errors.append(f"Expected 17 EASA SORA 2.5 OSOs, got {len(EASA_SORA_25_OSOS)}")

    raw_kani = certgen["kani_proofs_total"] if "kani_proofs_total" in certgen else None
    if isinstance(raw_kani, int) and not isinstance(raw_kani, bool):
        kani_proofs_total = raw_kani
    else:
        errors.append("MISSING_OR_INVALID_KANI_PROOFS_TOTAL")
        kani_proofs_total = 0

    gsn_path = root / "assurance" / "openxvs_v2.gsn.yaml"
    gsn_sha256 = _sha256_file(gsn_path) if gsn_path.is_file() else None

    matrix: dict[str, Any] = {
        "schema": "openxvs.sora25_f3442_ed324_matrix/1.0",
        "release_mode": release_mode,
        "regulatory_baseline": {
            "easa_sora_version": "SORA 2.5 (Transition Cutover: 2026-11-30)",
            "astm_daa_standard": "ASTM F3442-26 (Published 2026-10-15) + ASTM F3705-25",
            "eurocae_ml_standard": "EUROCAE ED-324 / SAE ARP6983 + EASA NPA 2025-07 (Issue 03)",
            "eu_pld_directive": "Directive (EU) 2024/2853 (Effective 2026-12-09)",
            "non_certification_notice": (
                "Traceability evidence for civil engineering & OEM evaluation ONLY. "
                "OpenXVS is NOT FAA/EASA certified; G5_GAP and G2_TLA_DEFERRED remain explicitly undeveloped."
            ),
        },
        "status": "PASS" if not errors else "FAIL",
        "frozen_gsn_reference": {
            "path": "assurance/openxvs_v2.gsn.yaml",
            "sha256": gsn_sha256,
            "certgen_status": certgen.get("status", "UNKNOWN"),
            "certgen_bundle_sha256": certgen.get("bundle_sha256"),
            "kani_proofs_total": kani_proofs_total,
            "explicit_undeveloped_gaps": sorted(set(undeveloped_nodes)),
            "g5_gap_honestly_undeveloped": "G5_GAP" in undeveloped_nodes,
            "g2_tla_deferred_honestly_undeveloped": "G2_TLA_DEFERRED" in undeveloped_nodes,
        },
        "easa_sora_25": {
            "oso_count": len(EASA_SORA_25_OSOS),
            "osos": [dict(row) for row in EASA_SORA_25_OSOS],
            "ground_risk_mitigations_m1": [dict(row) for row in EASA_SORA_25_MITIGATIONS],
        },
        "astm_f3442_26": {
            "requirement_count": len(ASTM_F3442_26_REQUIREMENTS),
            "requirements": [dict(row) for row in ASTM_F3442_26_REQUIREMENTS],
        },
        "eurocae_ed324": {
            "objective_count": len(ED324_OBJECTIVES),
            "objectives": [dict(row) for row in ED324_OBJECTIVES],
        },
        "errors": errors,
    }
    canonical = json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode("utf-8")
    matrix["matrix_sha256"] = hashlib.sha256(canonical).hexdigest()
    return matrix


# ---------------------------------------------------------------------------
# SPDX 2.3 Deterministic JSON SBOM & Zero-GPL/AGPL/LGPL-Static Auditor
# ---------------------------------------------------------------------------


def _normalize_spdx_expression(raw: str | None) -> str:
    if not raw or not raw.strip():
        return "NOASSERTION"
    cleaned = raw.strip()
    # Normalize Cargo legacy slash syntax (`Apache-2.0 / MIT` -> `Apache-2.0 OR MIT`)
    if "/" in cleaned and " OR " not in cleaned and " AND " not in cleaned:
        parts = [p.strip() for p in cleaned.split("/") if p.strip()]
        if len(parts) >= 2:
            cleaned = " OR ".join(parts)
    # Normalize common aliases
    cleaned = re.sub(r"\bApache\s+2\.0\b", "Apache-2.0", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bMIT\s+License\b", "MIT", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b3-Clause\s+BSD\s+License\b", "BSD-3-Clause", cleaned, flags=re.IGNORECASE)
    return cleaned


def _split_top_level(expr: str, op: str) -> list[str]:
    """Split an SPDX expression by top-level boolean operator (`AND` / `OR`) outside parentheses."""
    tokens = re.findall(r"\(|\)|\bAND\b|\bOR\b|[^()\s]+", expr)
    parts: list[list[str]] = [[]]
    depth = 0
    for tok in tokens:
        if tok == "(":
            depth += 1
            parts[-1].append(tok)
        elif tok == ")":
            depth = max(0, depth - 1)
            parts[-1].append(tok)
        elif depth == 0 and tok == op:
            parts.append([])
        else:
            parts[-1].append(tok)
    return [" ".join(p).strip() for p in parts if p]


def _strip_outer_parens(expr: str) -> str:
    s = expr.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        wraps = True
        for idx, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx < len(s) - 1:
                    wraps = False
                    break
        if not wraps:
            break
        s = s[1:-1].strip()
    return s


def evaluate_spdx_expression_for_copyleft(expr: str) -> tuple[bool, str]:
    """Return `(is_permissive_clean, reason)` for an SPDX license expression.

    Correctly handles SPDX `OR` / `AND` precedence:
      - `MIT OR Apache-2.0 OR LGPL-2.1-or-later` -> CLEAN (permissive `OR` branch selectable)
      - `(MIT OR Apache-2.0) AND Unicode-3.0`    -> CLEAN (every `AND` clause is clean)
      - `GPL-3.0-only`, `AGPL-3.0-or-later`, `LGPL-2.1-only`, `MIT AND GPL-2.0` -> REJECTED!
    """
    norm = _strip_outer_parens(_normalize_spdx_expression(expr))
    if not norm or norm == "NOASSERTION":
        return False, "missing_or_noassertion_license"

    or_branches = _split_top_level(norm, "OR")
    if len(or_branches) > 1:
        reasons: list[str] = []
        for branch in or_branches:
            ok, reason = evaluate_spdx_expression_for_copyleft(branch)
            if ok:
                return True, f"permissive_or_branch:{_strip_outer_parens(branch)}"
            reasons.append(reason)
        return False, f"all_or_branches_rejected({';'.join(reasons)})"

    and_clauses = _split_top_level(norm, "AND")
    if len(and_clauses) > 1:
        for clause in and_clauses:
            ok, reason = evaluate_spdx_expression_for_copyleft(clause)
            if not ok:
                return False, f"and_clause_rejected:{reason}"
        return True, "all_and_clauses_permissive"

    if _FORBIDDEN_LICENSE_TOKEN_RE.search(norm) is not None:
        return False, f"forbidden_copyleft_or_restricted_license:{norm}"
    return True, f"permissive:{norm}"


def _resolve_python_pkg_license(pkg_name: str) -> str:
    low = pkg_name.lower()
    try:
        meta = importlib_metadata.metadata(pkg_name)
    except importlib_metadata.PackageNotFoundError:
        return _PYTHON_KNOWN_SPDX_FALLBACK.get(low, "NOASSERTION")

    expr = meta.get("License-Expression")
    if expr and expr.strip():
        return _normalize_spdx_expression(expr)

    classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    detected: list[str] = []
    for cls in classifiers:
        cl = cls.lower()
        if "apache" in cl:
            detected.append("Apache-2.0")
        elif "mit" in cl:
            detected.append("MIT")
        elif "bsd" in cl:
            detected.append("BSD-3-Clause")
        elif "python software foundation" in cl:
            detected.append("PSF-2.0")
        elif "lgpl" in cl:
            detected.append("LGPL-3.0-only")
        elif "agpl" in cl:
            detected.append("AGPL-3.0-only")
        elif "gpl" in cl or "gnu general public" in cl:
            detected.append("GPL-3.0-only")
    if detected:
        return " OR ".join(sorted(set(detected)))

    raw_lic = meta.get("License")
    if raw_lic and len(raw_lic.strip()) <= 64 and "\n" not in raw_lic.strip():
        norm = _normalize_spdx_expression(raw_lic)
        if norm != "Dual License":
            return norm

    return _PYTHON_KNOWN_SPDX_FALLBACK.get(low, "NOASSERTION")


def _collect_cargo_packages(
    repo_root: Path,
    manifest_rels: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Collect Cargo packages across workspaces using offline `cargo metadata` (fallback `Cargo.lock`)."""
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for rel_manifest in manifest_rels:
        manifest_path = repo_root / rel_manifest
        if not manifest_path.is_file():
            continue
        workspace_label = str(Path(rel_manifest).parent).replace("\\", "/")
        used_cargo_metadata = False
        try:
            proc = subprocess.run(
                [
                    "cargo",
                    "metadata",
                    "--format-version",
                    "1",
                    "--manifest-path",
                    str(manifest_path),
                    "--offline",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                meta = json.loads(proc.stdout)
                for pkg in meta.get("packages", []):
                    name = str(pkg["name"])
                    version = str(pkg["version"])
                    lic = _normalize_spdx_expression(pkg.get("license"))
                    key = (name, version)
                    if key not in dedup:
                        dedup[key] = {
                            "ecosystem": "cargo",
                            "name": name,
                            "version": version,
                            "licenseConcluded": lic,
                            "licenseDeclared": lic,
                            "source": str(pkg.get("source") or f"workspace:{workspace_label}"),
                            "workspaces": [workspace_label],
                        }
                    elif workspace_label not in dedup[key]["workspaces"]:
                        dedup[key]["workspaces"].append(workspace_label)
                used_cargo_metadata = True
        except Exception:
            used_cargo_metadata = False

        if not used_cargo_metadata:
            lock_path = manifest_path.parent / "Cargo.lock"
            if lock_path.is_file():
                lock_doc = tomllib.loads(lock_path.read_text(encoding="utf-8"))
                for pkg in lock_doc.get("package", []):
                    name = str(pkg["name"])
                    version = str(pkg["version"])
                    key = (name, version)
                    # Local OpenXVS crates declare Apache-2.0 OR MIT in Cargo.toml
                    is_local = "source" not in pkg
                    lic = "Apache-2.0 OR MIT" if is_local else "NOASSERTION"
                    if key not in dedup:
                        dedup[key] = {
                            "ecosystem": "cargo",
                            "name": name,
                            "version": version,
                            "licenseConcluded": lic,
                            "licenseDeclared": lic,
                            "source": str(pkg.get("source") or f"workspace:{workspace_label}"),
                            "workspaces": [workspace_label],
                        }
    return [dedup[k] for k in sorted(dedup)]


def _collect_python_packages(requirements_path: Path) -> list[dict[str, Any]]:
    if not requirements_path.is_file():
        return []
    pkgs: list[dict[str, Any]] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            name = name.strip()
            version = version.strip().split()[0]
        else:
            name = line.split()[0]
            version = "UNPINNED"
        lic = _resolve_python_pkg_license(name)
        pkgs.append(
            {
                "ecosystem": "pypi",
                "name": name,
                "version": version,
                "licenseConcluded": lic,
                "licenseDeclared": lic,
                "source": "requirements-xvs.lock.txt",
            }
        )
    return sorted(pkgs, key=lambda p: (p["name"].lower(), p["version"]))


def _spdx_safe_id(prefix: str, name: str, version: str) -> str:
    raw = f"{prefix}-{name}-{version}"
    return "SPDXRef-" + re.sub(r"[^A-Za-z0-9.-]", "-", raw)


def build_spdx_2_3_sbom(
    *,
    repo_root: Path = REPO_ROOT,
    release_mode: str = "private_eval",
    include_studio_cargo: bool = False,
    requirements_lock_path: Path | None = None,
    deterministic_timestamp: str = "2026-11-23T00:00:00Z",
) -> dict[str, Any]:
    """Generate a deterministic SPDX 2.3 JSON SBOM auditing Cargo + Python dependencies.

    Enforces strictly `0` GPL, `0` AGPL, `0` LGPL-static, and `0` forbidden packages
    (`pymavlink`, `PyInstaller`, `SALAD`, etc.).
    """
    root = repo_root.resolve()
    manifests = list(RUST_CRATE_MANIFESTS)
    if include_studio_cargo and (root / "apps" / "studio" / "src-tauri" / "Cargo.toml").is_file():
        manifests.append("apps/studio/src-tauri/Cargo.toml")

    cargo_pkgs = _collect_cargo_packages(root, tuple(manifests))
    req_path = (root / "requirements-xvs.lock.txt") if requirements_lock_path is None else Path(requirements_lock_path)
    python_pkgs = _collect_python_packages(req_path)

    root_spdx_id = "SPDXRef-OpenXVS-v5-0"
    root_license = (
        "NOASSERTION"
        if release_mode == "private_eval"
        else "Apache-2.0"
    )
    spdx_packages: list[dict[str, Any]] = [
        {
            "SPDXID": root_spdx_id,
            "name": "openxvs",
            "versionInfo": "5.0.0",
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": root_license,
            "licenseDeclared": root_license,
            "supplier": "Organization: OpenXVS Contributors",
            "comment": (
                "Confidential Non-Flight Evaluation package (no public OSS grant active)"
                if release_mode == "private_eval"
                else "OpenXVS Civil Autonomous Aviation Open-Core release"
            ),
        }
    ]
    relationships: list[dict[str, str]] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_spdx_id,
        }
    ]

    copyleft_violations: list[dict[str, str]] = []
    forbidden_pkg_violations: list[dict[str, str]] = []

    for pkg in cargo_pkgs + python_pkgs:
        eco = pkg["ecosystem"]
        name = pkg["name"]
        version = pkg["version"]
        lic = pkg["licenseConcluded"]
        spdx_id = _spdx_safe_id(eco, name, version)
        purl = (
            f"pkg:cargo/{name}@{version}"
            if eco == "cargo"
            else f"pkg:pypi/{name.lower()}@{version}"
        )
        spdx_packages.append(
            {
                "SPDXID": spdx_id,
                "name": name,
                "versionInfo": version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": lic,
                "licenseDeclared": pkg["licenseDeclared"],
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": purl,
                    }
                ],
            }
        )
        relationships.append(
            {
                "spdxElementId": root_spdx_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": spdx_id,
            }
        )

        if name.lower().replace("_", "-") in FORBIDDEN_PACKAGE_NAMES:
            forbidden_pkg_violations.append({"ecosystem": eco, "name": name, "version": version, "license": lic})

        clean_ok, reason = evaluate_spdx_expression_for_copyleft(lic)
        if not clean_ok:
            copyleft_violations.append(
                {
                    "ecosystem": eco,
                    "name": name,
                    "version": version,
                    "license": lic,
                    "reason": reason,
                }
            )

    sbom_body: dict[str, Any] = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"openxvs-v5.0-{release_mode}-sbom",
        "documentNamespace": f"https://openxvs.org/spdx/v5.0/{release_mode}",
        "creationInfo": {
            "created": deterministic_timestamp,
            "creators": ["Tool: openxvs-sora25-f3442-sbom-generator-1.0"],
            "licenseListVersion": "3.25",
        },
        "packages": spdx_packages,
        "relationships": relationships,
        "openxvsLicenseAudit": {
            "release_mode": release_mode,
            "cargo_packages_audited": len(cargo_pkgs),
            "python_packages_audited": len(python_pkgs),
            "total_dependencies_audited": len(cargo_pkgs) + len(python_pkgs),
            "gpl_agpl_lgpl_static_violations": copyleft_violations,
            "forbidden_package_violations": forbidden_pkg_violations,
            "zero_copyleft_pass": (
                len(copyleft_violations) == 0 and len(forbidden_pkg_violations) == 0
            ),
        },
    }
    canonical = json.dumps(sbom_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sbom_body["openxvsLicenseAudit"]["sbom_sha256"] = hashlib.sha256(canonical).hexdigest()
    return sbom_body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenXVS v5.0 EASA SORA 2.5 / ASTM F3442-26 / ED-324 Matrix & SPDX 2.3 SBOM Generator"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--release-mode",
        choices=("private_eval", "wave1_formal_core", "public_open_core"),
        default="private_eval",
    )
    parser.add_argument("--out-matrix", type=Path, default=None)
    parser.add_argument("--out-sbom", type=Path, default=None)
    parser.add_argument("--include-studio-cargo", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    matrix = build_sora25_f3442_ed324_matrix(repo_root=args.repo_root, release_mode=args.release_mode)
    sbom = build_spdx_2_3_sbom(
        repo_root=args.repo_root,
        release_mode=args.release_mode,
        include_studio_cargo=args.include_studio_cargo,
    )

    if args.out_matrix is not None:
        args.out_matrix.parent.mkdir(parents=True, exist_ok=True)
        args.out_matrix.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_sbom is not None:
        args.out_sbom.parent.mkdir(parents=True, exist_ok=True)
        args.out_sbom.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = sbom["openxvsLicenseAudit"]
    ok = matrix["status"] == "PASS" and audit["zero_copyleft_pass"]
    print(
        f"sora25-f3442-sbom: status={'PASS' if ok else 'FAIL'} "
        f"osos={matrix['easa_sora_25']['oso_count']} "
        f"m1={len(matrix['easa_sora_25']['ground_risk_mitigations_m1'])} "
        f"f3442={matrix['astm_f3442_26']['requirement_count']} "
        f"ed324={matrix['eurocae_ed324']['objective_count']} "
        f"sbom_deps={audit['total_dependencies_audited']} "
        f"copyleft_violations={len(audit['gpl_agpl_lgpl_static_violations'])} "
        f"g5_gap={matrix['frozen_gsn_reference']['g5_gap_honestly_undeveloped']} "
        f"g2_tla_deferred={matrix['frozen_gsn_reference']['g2_tla_deferred_honestly_undeveloped']}"
    )
    if args.check and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
