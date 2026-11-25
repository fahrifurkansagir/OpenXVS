"""ED-324 (SAE ARP6983) Goal Structuring Notation (GSN v3 / SCSC-141C) data model (`V2-M3L-4`).

Implements the strict OpenXVS v2.0 assurance case topology (`SP3_M3_XVS_RT_BENCH_DESIGN.md` §3.3):
- Root `G0` -> `inContextOf: [C1, C2]`, `supportedBy: [S1]`
- Strategy `S1` -> `supportedBy: [G1, G2, G3, G4, G5_GAP]`
- `G1` -> `[Sn1_MHVIT, Sn2_HYBRID_CBF, Sn3_DAQP_WCET]` (Stage-1 `xvs-rt` & `XVS-MeshDAA` evidence)
- `G2` -> `[G2_TLA_DEFERRED, Sn5_KANI, Sn6_PARITY]` (`G2_TLA_DEFERRED` is `undeveloped: true` Goal!)
- `G3` -> `[Sn7_DTACI_OOD, Sn8_SPLIT_H3]` (M2 LWIR, `XVS-Land`, `XVS-Nav`, `XVS-VRAIM` evidence)
- `G4` -> `[Sn9_PROVENANCE]` (M1/M2 provenance reports + Stage-2 `xvs_bench_chain.json` — strictly acyclic!)
- `G5_GAP` -> `undeveloped: true` Goal (`DO-254`, qualified compiler, `DO-330`, RTOS, `ARP4754B`, human `QA-01`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any


BANNED_REGULATORY_OVERCLAIMS: tuple[str, ...] = (
    "certifiable ai",
    "do-178c ml",
    "zero rf emissions",
)

FORBIDDEN_CIRCULAR_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "assurance/",
    "docs/xvs/reports/V2_M3_local_core_report.md",
    "output/xvs/v2_m3/run_manifest.json",
    "output/xvs/v2_m3/v4_ed324_gsn_evidence.json",
    "output/xvs/v2_m3/v6_baseline_evidence.json",
)


class GsnNodeType(str, Enum):
    GOAL = "Goal"
    STRATEGY = "Strategy"
    CONTEXT = "Context"
    SOLUTION = "Solution"


@dataclass(frozen=True)
class EvidenceBinding:
    """Cryptographic SHA-256 binding between a GSN `Sn*` Solution node and a real on-disk artifact."""

    rel_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class GsnNode:
    """Single SCSC-141C / gsn2x-compatible GSN v3 node."""

    node_id: str
    node_type: GsnNodeType
    text: str
    supported_by: tuple[str, ...] = ()
    in_context_of: tuple[str, ...] = ()
    undeveloped: bool = False
    evidence_files: tuple[EvidenceBinding, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GsnGraph:
    """Immutable ED-324 GSN v3 assurance graph."""

    schema: str
    standard: str
    root_id: str
    nodes: dict[str, GsnNode]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def bind_evidence_paths(repo_root: Path, rel_paths: tuple[str, ...]) -> tuple[EvidenceBinding, ...]:
    """Resolve and SHA-256 hash real on-disk artifact paths relative to `repo_root`."""
    bindings: list[EvidenceBinding] = []
    root = Path(repo_root).resolve()
    for rel in rel_paths:
        norm_rel = rel.replace("\\", "/")
        fpath = (root / norm_rel).resolve()
        if not fpath.is_file():
            raise FileNotFoundError(f"GSN Solution evidence file missing on disk: {norm_rel} ({fpath})")
        bindings.append(
            EvidenceBinding(
                rel_path=norm_rel,
                sha256=sha256_file(fpath),
                size_bytes=int(fpath.stat().st_size),
            )
        )
    return tuple(bindings)


# Authoritative evidence bindings per Solution node (`SP3_M3_XVS_RT_BENCH_DESIGN.md` §2.1 & §3.3)
OPENXVS_V2_SOLUTION_PATHS: dict[str, tuple[str, ...]] = {
    "Sn1_MHVIT": (
        "output/xvs/v2_m3/v1_xvs_rt_evidence.json",
        "output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json",
    ),
    "Sn2_HYBRID_CBF": (
        "output/xvs/v2_m3/v1_xvs_rt_evidence.json",
        "output/xvs/v2_m3/v2_meshdaa_500enc_evidence.json",
    ),
    "Sn3_DAQP_WCET": (
        "output/xvs/v2_m3/v1_xvs_rt_evidence.json",
    ),
    "Sn5_KANI": (
        "output/xvs/v2_m1/kani_proofs.log",
    ),
    "Sn6_PARITY": (
        "output/xvs/v2_m1/parity_results.json",
        "output/xvs/v2_m1/e2e_results.json",
    ),
    "Sn7_DTACI_OOD": (
        "output/xvs/v2_m2/v3_lwir_thermal_world_evidence.json",
        "output/xvs/v2_m2/v5_nav_vraim_evidence.json",
    ),
    "Sn8_SPLIT_H3": (
        "output/xvs/v2_m2/v4_wp01b_land_evidence.json",
        "output/xvs/v2_m2/v5_nav_vraim_evidence.json",
    ),
    "Sn9_PROVENANCE": (
        "docs/xvs/reports/V2_M1_livedeck_report.md",
        "docs/xvs/reports/V2_M2_local_prep_report.md",
        "output/xvs/v2_m3/xvs_bench_chain.json",
    ),
}


def build_openxvs_v2_gsn_graph(repo_root: Path) -> GsnGraph:
    """Construct the canonical OpenXVS v2.0 ED-324 GSN v3 graph bound to real SHA-256 files."""
    root = Path(repo_root).resolve()

    def _sn(node_id: str, text: str) -> GsnNode:
        rel_paths = OPENXVS_V2_SOLUTION_PATHS[node_id]
        bindings = bind_evidence_paths(root, rel_paths)
        return GsnNode(
            node_id=node_id,
            node_type=GsnNodeType.SOLUTION,
            text=text,
            undeveloped=False,
            evidence_files=bindings,
        )

    nodes_list: list[GsnNode] = [
        GsnNode(
            node_id="G0",
            node_type=GsnNodeType.GOAL,
            text=(
                "OpenXVS v2.0 civil Detect-and-Avoid and GNSS-resilient perception software "
                "maintains bounded runtime safety margins within its defined civil ODD, with "
                "unverified hardware/toolchain certification gaps explicitly isolated in G5_GAP."
            ),
            in_context_of=("C1", "C2"),
            supported_by=("S1",),
        ),
        GsnNode(
            node_id="C1",
            node_type=GsnNodeType.CONTEXT,
            text=(
                "Strictly civil aviation ODD (14 CFR Part 91/135 GA & Part 107/108 UAS; Charter §2 "
                "civil-only scope, zero active radar emissions on passive EO/LWIR sensors) under "
                "known deferral tags ([PENDING_COLAB_G4_TRAINING], [PENDING_NATIVE_LINUX_GPU_WCET], "
                "[LOCAL_LOOPBACK_SITL_EMULATOR; FULL_ARDUPILOT_BINARY_DEFERRED_TO_PHASE_4], "
                "[PREWIRED_PHASE4_PENDING_CAPTURE])."
            ),
        ),
        GsnNode(
            node_id="C2",
            node_type=GsnNodeType.CONTEXT,
            text=(
                "EUROCAE ED-324 / SAE ARP6983 Objective 4, EASA NPA 2025-07 (Proposed Issue 03) "
                "R-LM-07 to R-LM-11, RTCA DO-365A/B, TSO-C151d / DO-367, and CONVENTIONS.md v1.0 "
                "(ECEF/NED/BODY/CAM, Hamilton quaternions w>=0, zero internal ENU)."
            ),
        ),
        GsnNode(
            node_id="S1",
            node_type=GsnNodeType.STRATEGY,
            text=(
                "Argue bounded civil runtime safety via five decomposed pillars: (G1) deterministic "
                "conflict avoidance & zero-alloc RT execution, (G2) formal Simplex RTA safety monitor "
                "& bit-exact cross-language parity, (G3) calibrated ML OOD/conformal/Vision-RAIM "
                "bounds, (G4) cryptographic provenance & hash-chained reproducibility, and (G5_GAP) "
                "explicit undeveloped boundary for external avionics certification artifacts."
            ),
            supported_by=("G1", "G2", "G3", "G4", "G5_GAP"),
        ),
        GsnNode(
            node_id="G1",
            node_type=GsnNodeType.GOAL,
            text=(
                "Deterministic fixed-wing conflict avoidance (K=5 NED MH-VIT, 3D NED ECBF-QP, "
                "isolated DAIDALUS C-ABI, and 3-ship XVS-MeshDAA) executes with zero hot-loop heap "
                "allocations and achieves zero NMAC across 500 seeded civil encounters."
            ),
            supported_by=("Sn1_MHVIT", "Sn2_HYBRID_CBF", "Sn3_DAQP_WCET"),
        ),
        _sn(
            "Sn1_MHVIT",
            "WSL2 C++20 + libxvs_core.a K=5 log-spaced NED MH-VIT (0% inward overshoot) & 500-encounter XVS-MeshDAA multi-intruder evidence.",
        ),
        _sn(
            "Sn2_HYBRID_CBF",
            "Strict NED 3D fixed-wing ECBF-QP safety/viability & 3-ship 0-NMAC / ghost-suppression closed-loop evidence.",
        ),
        _sn(
            "Sn3_DAQP_WCET",
            "WSL2 g++ -Wl,--wrap=malloc zero-allocation hot-loop audit (0 B under both -O0 and -O2) & DAQP-ABI active-set solver timing evidence.",
        ),
        GsnNode(
            node_id="G2",
            node_type=GsnNodeType.GOAL,
            text=(
                "Deterministic no_std Simplex RTA monitor (xvs-rta-core) enforces fail-closed mode "
                "transitions, non-finite rejection, and cross-language bit-exact parity across Rust, "
                "Python, WASM, and C++20 FFI."
            ),
            supported_by=("G2_TLA_DEFERRED", "Sn5_KANI", "Sn6_PARITY"),
        ),
        GsnNode(
            node_id="G2_TLA_DEFERRED",
            node_type=GsnNodeType.GOAL,
            text=(
                "Full TLA+ temporal model checking of distributed multi-ship Simplex RTA liveness "
                "properties is explicitly deferred (undeveloped); local safety invariants are "
                "currently bounded by 5/5 Kani/CBMC proofs (Sn5_KANI) and 100k proptest cases."
            ),
            undeveloped=True,
            supported_by=(),
        ),
        _sn(
            "Sn5_KANI",
            "WSL2 Kani/CBMC 5/5 formal verification harness log on frozen no_std xvs-rta-core.",
        ),
        _sn(
            "Sn6_PARITY",
            "10,000-tick bit-exact Simplex RTA & codec differential parity plus WebGPU/WebGL2 browser E2E verification.",
        ),
        GsnNode(
            node_id="G3",
            node_type=GsnNodeType.GOAL,
            text=(
                "ML perception modules are bounded by fail-closed sensor health gating, split-conformal "
                "Vision-RAIM protection levels, 6-stage visual-aliasing rejection, and EASA R-LM-07 to "
                "R-LM-11 offline OOD/SAE/AST assurance audits."
            ),
            supported_by=("Sn7_DTACI_OOD", "Sn8_SPLIT_H3"),
        ),
        _sn(
            "Sn7_DTACI_OOD",
            "LWIR physics & XVS-ThermalNet 47-case fail-closed gating plus XVS-VRAIM Mahalanobis OOD, Top-K SAE, and seeded AST EASA R-LM-07 to R-LM-11 audit.",
        ),
        _sn(
            "Sn8_SPLIT_H3",
            "WP-01b curved ray-DEM / TSO-C151d TAWS FLTA, XVS-Land hybrid Conformal-RAIM (KSFO 28L vs Taxiway C/28R), and XVS-Nav 6-stage aliasing gate evidence.",
        ),
        GsnNode(
            node_id="G4",
            node_type=GsnNodeType.GOAL,
            text=(
                "All verification metrics and benchmark artifacts are reproducibly bound to authoritative "
                "on-disk JSON producers via rv() provenance tracking and an Ed25519-signed SHA-256 hash chain."
            ),
            supported_by=("Sn9_PROVENANCE",),
        ),
        _sn(
            "Sn9_PROVENANCE",
            "Provenance-verified M1 & M2 reports plus Stage-2 XVS-Bench Ed25519-signed SHA-256 hash chain (strictly acyclic lower-stage inputs).",
        ),
        GsnNode(
            node_id="G5_GAP",
            node_type=GsnNodeType.GOAL,
            text=(
                "EXPLICIT UNDEVELOPED CERTIFICATION GAP (NOT CLAIMED BY OPENXVS): RTCA DO-254 "
                "complex electronic hardware assurance, DAL-A/B qualified compiler & RTCA DO-330 "
                "tool qualification, ARINC 653 certified RTOS partitioning, airframe-level SAE "
                "ARP4754B FHA/PSSA/SSA, and independent human DER/CVE verification sign-off "
                "(EASA QA-01 / RTCA DO-178C Table A-7)."
            ),
            undeveloped=True,
            supported_by=(),
        ),
    ]

    nodes = {n.node_id: n for n in nodes_list}
    return GsnGraph(
        schema="xvs.ed324_gsn/1",
        standard="SCSC-141C / EUROCAE ED-324 (SAE ARP6983) GSN v3",
        root_id="G0",
        nodes=nodes,
    )
