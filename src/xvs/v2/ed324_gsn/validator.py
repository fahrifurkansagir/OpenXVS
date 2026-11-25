"""Pure-Python + WSL2 PyYAML validator for OpenXVS v2.0 `ED-324` GSN v3 (`V2-M3L-4`).

Enforces `BC-M3-M1` & `BC-M3-M2` strictly:
1. Acyclic DAG (cycle detection on `supportedBy` + `inContextOf`).
2. Single root `G0` reaching 100% of nodes (zero orphan nodes).
3. 100% of `Sn*` (`Solution`) nodes bind to real on-disk files with matching SHA-256 digests
   and NEVER set `undeveloped: true`.
4. Strict acyclic evidence boundary: zero `Sn*` nodes may bind Stage-4 reports, `run_manifest.json`,
   `v4_ed324_gsn_evidence.json`, or `assurance/*`.
5. Every `undeveloped: true` node (`G5_GAP`, `G2_TLA_DEFERRED`) MUST be a `Goal` with `0`
   `supportedBy` children or solutions.
6. Zero banned regulatory overclaims (`"certifiable ai"`, `"do-178c ml"`, `"zero rf emissions"`).
7. Cross-validation of emitted `openxvs_v2.gsn.yaml` via WSL2 `python3` `yaml` (`yaml.safe_load`).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

from xvs.v2.ed324_gsn.model import (
    BANNED_REGULATORY_OVERCLAIMS,
    FORBIDDEN_CIRCULAR_EVIDENCE_PREFIXES,
    GsnGraph,
    GsnNodeType,
    sha256_file,
)


@dataclass(frozen=True)
class GsnValidationResult:
    valid: bool
    node_count: int
    solution_count: int
    undeveloped_goal_ids: tuple[str, ...]
    evidence_bindings_verified: int
    banned_overclaim_hits: int
    errors: tuple[str, ...]


def find_banned_overclaims(text: str) -> list[str]:
    """Return any banned regulatory overclaim phrases present in `text` (case-insensitive)."""
    lowered = text.lower()
    return [phrase for phrase in BANNED_REGULATORY_OVERCLAIMS if phrase in lowered]


def validate_gsn_graph(graph: GsnGraph, repo_root: Path) -> GsnValidationResult:
    """Validate structural DAG, root reachability, SHA-256 evidence bindings, and honesty rules."""
    errors: list[str] = []
    root = Path(repo_root).resolve()
    nodes = graph.nodes

    if graph.root_id != "G0" or "G0" not in nodes:
        errors.append(f"Root node must be 'G0', got root_id={graph.root_id!r}")

    # 1. Verify all referenced edges exist & detect cycles (DFS 3-color)
    color: dict[str, int] = {nid: 0 for nid in nodes}  # 0=unvisited, 1=visiting, 2=done

    def _dfs(nid: str) -> None:
        color[nid] = 1
        node = nodes[nid]
        for child in (*node.supported_by, *node.in_context_of):
            if child not in nodes:
                errors.append(f"Dangling edge: {nid} -> {child} (target node missing)")
                continue
            if color[child] == 1:
                errors.append(f"Cycle detected involving edge {nid} -> {child}")
            elif color[child] == 0:
                _dfs(child)
        color[nid] = 2

    if "G0" in nodes:
        _dfs("G0")

    # 2. Single root reachability (every node must be reachable from G0)
    unreachable = sorted(nid for nid, c in color.items() if c == 0)
    if unreachable:
        errors.append(f"Orphan/unreachable nodes not reachable from G0: {unreachable}")

    # Also check G0 has in-degree 0
    incoming_to_g0 = [
        nid for nid, n in nodes.items() if "G0" in n.supported_by or "G0" in n.in_context_of
    ]
    if incoming_to_g0:
        errors.append(f"Root G0 must have in-degree 0, found incoming from {incoming_to_g0}")

    # 3. Node-type, Solution SHA-256, Undeveloped Goal, and Overclaim checks
    solution_count = 0
    evidence_bindings_verified = 0
    undeveloped_goals: list[str] = []
    banned_hits_total = 0

    for nid in sorted(nodes.keys()):
        node = nodes[nid]
        overclaims = find_banned_overclaims(node.text)
        if overclaims:
            banned_hits_total += len(overclaims)
            errors.append(f"Node {nid} contains banned regulatory overclaim(s): {overclaims}")

        if node.undeveloped:
            if node.node_type != GsnNodeType.GOAL or not nid.startswith("G"):
                errors.append(
                    f"Node {nid} has undeveloped=True but is not a Goal (type={node.node_type.value})"
                )
            if len(node.supported_by) > 0:
                errors.append(
                    f"Undeveloped Goal {nid} must have 0 supportedBy children/solutions, got {list(node.supported_by)}"
                )
            if len(node.evidence_files) > 0:
                errors.append(
                    f"Undeveloped Goal {nid} must have 0 fake evidence bindings, got {len(node.evidence_files)}"
                )
            undeveloped_goals.append(nid)

        if node.node_type == GsnNodeType.SOLUTION or nid.startswith("Sn"):
            solution_count += 1
            if node.node_type != GsnNodeType.SOLUTION or not nid.startswith("Sn"):
                errors.append(f"Solution naming/type mismatch on {nid} (type={node.node_type.value})")
            if node.undeveloped:
                errors.append(f"BC-M3-M1 violation: Solution {nid} MUST NOT have undeveloped=True")
            if len(node.supported_by) > 0:
                errors.append(f"Solution {nid} must be a leaf node (0 supportedBy children)")
            if len(node.evidence_files) == 0:
                errors.append(f"Solution {nid} has 0 bound evidence files")

            for binding in node.evidence_files:
                norm_rel = binding.rel_path.replace("\\", "/")
                for forbidden in FORBIDDEN_CIRCULAR_EVIDENCE_PREFIXES:
                    if norm_rel == forbidden or norm_rel.startswith(forbidden):
                        errors.append(
                            f"BC-M3-M2 circular evidence violation in {nid}: forbidden path {norm_rel!r}"
                        )
                fpath = (root / norm_rel).resolve()
                if not fpath.is_file():
                    errors.append(f"Solution {nid} bound evidence file missing: {norm_rel}")
                    continue
                actual_sha = sha256_file(fpath)
                actual_size = int(fpath.stat().st_size)
                if actual_sha != binding.sha256:
                    errors.append(
                        f"Solution {nid} SHA-256 mismatch for {norm_rel}: "
                        f"expected {binding.sha256}, actual {actual_sha}"
                    )
                elif actual_size != binding.size_bytes:
                    errors.append(
                        f"Solution {nid} size mismatch for {norm_rel}: "
                        f"expected {binding.size_bytes}, actual {actual_size}"
                    )
                else:
                    evidence_bindings_verified += 1

        elif node.node_type in (GsnNodeType.GOAL, GsnNodeType.STRATEGY):
            if not node.undeveloped and len(node.supported_by) == 0:
                errors.append(
                    f"Developed {node.node_type.value} {nid} has 0 supportedBy children (must be developed or undeveloped=True Goal)"
                )

    for required_undeveloped in ("G5_GAP", "G2_TLA_DEFERRED"):
        if required_undeveloped not in undeveloped_goals:
            errors.append(f"Required undeveloped Goal {required_undeveloped!r} missing or not marked undeveloped=True")

    return GsnValidationResult(
        valid=len(errors) == 0,
        node_count=len(nodes),
        solution_count=solution_count,
        undeveloped_goal_ids=tuple(sorted(undeveloped_goals)),
        evidence_bindings_verified=evidence_bindings_verified,
        banned_overclaim_hits=banned_hits_total,
        errors=tuple(errors),
    )


def _windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def validate_yaml_via_wsl2(yaml_path: Path, expected_node_ids: set[str] | None = None) -> dict[str, Any]:
    """Cross-validate emitted `openxvs_v2.gsn.yaml` using WSL2 Ubuntu `python3` `yaml.safe_load`."""
    wsl_yaml_path = _windows_to_wsl_path(yaml_path)
    py_code = (
        "import json, sys, yaml\n"
        "p = sys.argv[1]\n"
        "with open(p, 'r', encoding='utf-8') as f:\n"
        "    doc = yaml.safe_load(f)\n"
        "assert isinstance(doc, dict), 'YAML root must be mapping'\n"
        "undeveloped = sorted(k for k, v in doc.items() if isinstance(v, dict) and 'undeveloped' in v and v['undeveloped'] is True)\n"
        "solutions = sorted(k for k in doc.keys() if k.startswith('Sn'))\n"
        "print(json.dumps({'yaml_version': yaml.__version__, 'node_count': len(doc), "
        "'node_ids': sorted(doc.keys()), 'undeveloped': undeveloped, 'solutions': solutions}))\n"
    )
    res = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "python3", "-c", py_code, wsl_yaml_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"WSL2 yaml.safe_load validation failed (rc={res.returncode}):\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    parsed = json.loads(res.stdout.strip())
    if expected_node_ids is not None:
        got_ids = set(parsed["node_ids"])
        if got_ids != expected_node_ids:
            raise AssertionError(
                f"WSL2 parsed YAML node IDs mismatch: missing={sorted(expected_node_ids - got_ids)}, "
                f"extra={sorted(got_ids - expected_node_ids)}"
            )
    return parsed
