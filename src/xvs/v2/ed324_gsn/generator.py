"""Deterministic generator for `assurance/{openxvs_v2.gsn.yaml, openxvs_v2_gsn.mmd, evidence.md, evidence_trace.json}` (`V2-M3L-4`).

Emits SCSC-141C / gsn2x-compatible YAML (zero external PyYAML dependency in Windows `.venv`),
Mermaid `graph TD` with dashed red styling on BOTH `G5_GAP` and `G2_TLA_DEFERRED`, human-readable
evidence register, and machine-readable SHA-256 traceability ledger.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xvs.v2.ed324_gsn.model import GsnGraph, GsnNode, GsnNodeType, build_openxvs_v2_gsn_graph, sha256_file
from xvs.v2.ed324_gsn.validator import find_banned_overclaims, validate_gsn_graph

# Canonical top-down deterministic ordering for YAML / Mermaid / Markdown
CANONICAL_NODE_ORDER: tuple[str, ...] = (
    "G0",
    "C1",
    "C2",
    "S1",
    "G1",
    "Sn1_MHVIT",
    "Sn2_HYBRID_CBF",
    "Sn3_DAQP_WCET",
    "G2",
    "G2_TLA_DEFERRED",
    "Sn5_KANI",
    "Sn6_PARITY",
    "G3",
    "Sn7_DTACI_OOD",
    "Sn8_SPLIT_H3",
    "G4",
    "Sn9_PROVENANCE",
    "G5_GAP",
)


def _ordered_nodes(graph: GsnGraph) -> list[GsnNode]:
    seen: set[str] = set()
    ordered: list[GsnNode] = []
    for nid in CANONICAL_NODE_ORDER:
        if nid in graph.nodes:
            ordered.append(graph.nodes[nid])
            seen.add(nid)
    for nid in sorted(graph.nodes.keys()):
        if nid not in seen:
            ordered.append(graph.nodes[nid])
    return ordered


def _yaml_quote(s: str) -> str:
    """Emit a YAML-safe double-quoted scalar using JSON string escaping."""
    return json.dumps(s, ensure_ascii=False)


def render_gsn_yaml(graph: GsnGraph) -> str:
    """Render SCSC-141C / gsn2x-compatible YAML (`yaml.safe_load` clean)."""
    lines: list[str] = [
        "# OpenXVS v2.0 EUROCAE ED-324 (SAE ARP6983) GSN v3 Assurance Case",
        "# Format: SCSC-141C / gsn2x-compatible top-level node mapping",
        f"# Schema: {graph.schema} | Standard: {graph.standard}",
        "",
    ]
    for node in _ordered_nodes(graph):
        lines.append(f"{node.node_id}:")
        lines.append(f"  text: {_yaml_quote(node.text)}")
        if node.in_context_of:
            lines.append("  inContextOf:")
            for cid in node.in_context_of:
                lines.append(f"    - {cid}")
        if node.supported_by:
            lines.append("  supportedBy:")
            for sid in node.supported_by:
                lines.append(f"    - {sid}")
        if node.undeveloped:
            lines.append("  undeveloped: true")
        if node.evidence_files:
            lines.append(f"  url: {_yaml_quote(node.evidence_files[0].rel_path)}")
            lines.append("  evidenceFiles:")
            for b in node.evidence_files:
                lines.append(f"    - path: {_yaml_quote(b.rel_path)}")
                lines.append(f"      sha256: {_yaml_quote(b.sha256)}")
                lines.append(f"      size_bytes: {b.size_bytes}")
        lines.append("")
    return "\n".join(lines)


def _mermaid_label(node: GsnNode) -> str:
    short = node.text
    if len(short) > 78:
        short = short[:75].rstrip() + "..."
    safe = short.replace('"', "'").replace("[", "(").replace("]", ")")
    suffix = " [UNDEVELOPED]" if node.undeveloped else ""
    return f"{node.node_id}: {safe}{suffix}"


def render_gsn_mermaid(graph: GsnGraph) -> str:
    """Render deterministic Mermaid `graph TD` with dashed red styling on undeveloped Goals."""
    lines: list[str] = [
        "graph TD",
        "    classDef goal fill:#eef6ff,stroke:#1565c0,stroke-width:1.5px,color:#0d1b2a;",
        "    classDef strategy fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1.5px,color:#1a0d2a;",
        "    classDef context fill:#f5f5f5,stroke:#616161,stroke-width:1.2px,color:#212121;",
        "    classDef solution fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#0d260f;",
        "    classDef undeveloped fill:#fff0f0,stroke:#c62828,stroke-width:2.5px,stroke-dasharray: 6 4,color:#7f0000;",
        "",
    ]

    for node in _ordered_nodes(graph):
        lbl = _mermaid_label(node)
        if node.node_type == GsnNodeType.GOAL:
            lines.append(f'    {node.node_id}["{lbl}"]')
        elif node.node_type == GsnNodeType.STRATEGY:
            lines.append(f'    {node.node_id}[/"{lbl}"/]')
        elif node.node_type == GsnNodeType.CONTEXT:
            lines.append(f'    {node.node_id}(["{lbl}"])')
        else:
            lines.append(f'    {node.node_id}(("{lbl}"))')

    lines.append("")
    for node in _ordered_nodes(graph):
        for cid in node.in_context_of:
            lines.append(f"    {node.node_id} -.-> {cid}")
        for sid in node.supported_by:
            lines.append(f"    {node.node_id} --> {sid}")

    lines.append("")
    for node in _ordered_nodes(graph):
        if node.undeveloped:
            cls = "undeveloped"
        elif node.node_type == GsnNodeType.GOAL:
            cls = "goal"
        elif node.node_type == GsnNodeType.STRATEGY:
            cls = "strategy"
        elif node.node_type == GsnNodeType.CONTEXT:
            cls = "context"
        else:
            cls = "solution"
        lines.append(f"    class {node.node_id} {cls};")

    lines.append("")
    return "\n".join(lines)


def render_evidence_markdown(graph: GsnGraph) -> str:
    """Render human-readable `assurance/evidence.md` register of all Solution SHA-256 bindings."""
    lines: list[str] = [
        "# OpenXVS v2.0 — `ED-324` (`SAE ARP6983`) GSN v3 Evidence Register",
        "",
        f"- **Standard:** `{graph.standard}`",
        f"- **Root Goal:** `{graph.root_id}`",
        "- **Traceability Graph:** `assurance/openxvs_v2.gsn.yaml` | `assurance/openxvs_v2_gsn.mmd` | `assurance/evidence_trace.json`",
        "",
        "## 1. Bound Solution Nodes (`Sn*` — 100% Real On-Disk SHA-256 Artifacts)",
        "",
        "| Solution ID | Artifact Path | Size (B) | SHA-256 |",
        "| :--- | :--- | ---: | :--- |",
    ]

    for node in _ordered_nodes(graph):
        if node.node_type == GsnNodeType.SOLUTION:
            for b in node.evidence_files:
                lines.append(f"| `{node.node_id}` | `{b.rel_path}` | {b.size_bytes} | `{b.sha256}` |")

    lines.extend(
        [
            "",
            "## 2. Explicit Undeveloped Goals (`undeveloped: true` — Zero Synthetic Solutions)",
            "",
            "| Goal ID | Status | Scope Statement |",
            "| :--- | :--- | :--- |",
        ]
    )
    for node in _ordered_nodes(graph):
        if node.undeveloped:
            lines.append(f"| `{node.node_id}` | `UNDEVELOPED` | {node.text} |")

    lines.append("")
    return "\n".join(lines)


def build_evidence_trace(graph: GsnGraph) -> dict[str, Any]:
    """Build machine-readable `assurance/evidence_trace.json` traceability ledger."""
    solutions: dict[str, Any] = {}
    undeveloped_goals: dict[str, str] = {}
    for node in _ordered_nodes(graph):
        if node.node_type == GsnNodeType.SOLUTION:
            solutions[node.node_id] = {
                "text": node.text,
                "undeveloped": False,
                "evidence_files": [
                    {
                        "path": b.rel_path,
                        "sha256": b.sha256,
                        "size_bytes": b.size_bytes,
                    }
                    for b in node.evidence_files
                ],
            }
        elif node.undeveloped:
            undeveloped_goals[node.node_id] = node.text

    return {
        "schema": "xvs.ed324_evidence_trace/1",
        "standard": graph.standard,
        "root_id": graph.root_id,
        "node_count": len(graph.nodes),
        "solution_count": len(solutions),
        "undeveloped_goals": undeveloped_goals,
        "solutions": solutions,
    }


def generate_assurance_bundle(
    repo_root: Path,
    out_assurance_dir: Path | None = None,
) -> dict[str, Any]:
    """Build, validate, and emit the full `assurance/` bundle for OpenXVS v2.0."""
    root = Path(repo_root).resolve()
    out_dir = (root / "assurance") if out_assurance_dir is None else Path(out_assurance_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = build_openxvs_v2_gsn_graph(root)
    val = validate_gsn_graph(graph, root)
    if not val.valid:
        raise RuntimeError(f"GSN validation failed before emission: {list(val.errors)}")

    yaml_text = render_gsn_yaml(graph)
    mmd_text = render_gsn_mermaid(graph)
    md_text = render_evidence_markdown(graph)
    trace_doc = build_evidence_trace(graph)
    trace_text = json.dumps(trace_doc, indent=2) + "\n"

    for label, txt in (
        ("openxvs_v2.gsn.yaml", yaml_text),
        ("openxvs_v2_gsn.mmd", mmd_text),
        ("evidence.md", md_text),
        ("evidence_trace.json", trace_text),
    ):
        hits = find_banned_overclaims(txt)
        if hits:
            raise RuntimeError(f"Banned regulatory overclaim(s) in emitted {label}: {hits}")

    yaml_path = out_dir / "openxvs_v2.gsn.yaml"
    mmd_path = out_dir / "openxvs_v2_gsn.mmd"
    md_path = out_dir / "evidence.md"
    trace_path = out_dir / "evidence_trace.json"

    yaml_path.write_text(yaml_text, encoding="utf-8")
    mmd_path.write_text(mmd_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    trace_path.write_text(trace_text, encoding="utf-8")

    return {
        "graph": graph,
        "validation": val,
        "files": {
            "gsn_yaml": str(yaml_path),
            "gsn_mermaid": str(mmd_path),
            "evidence_md": str(md_path),
            "evidence_trace_json": str(trace_path),
        },
        "sha256": {
            "openxvs_v2.gsn.yaml": sha256_file(yaml_path),
            "openxvs_v2_gsn.mmd": sha256_file(mmd_path),
            "evidence.md": sha256_file(md_path),
            "evidence_trace.json": sha256_file(trace_path),
        },
    }
