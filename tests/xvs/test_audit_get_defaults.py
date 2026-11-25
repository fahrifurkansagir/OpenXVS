"""Test for G-B14 W5: Assert no unannotated .get() calls with numeric or 'N/A' defaults exist."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PATTERN_NUM = re.compile(r'\.get\s*\(\s*[^,\)]+,\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*\)')
PATTERN_NA = re.compile(r'\.get\s*\(\s*[^,\)]+,\s*[\'"]N/A[\'"]\s*\)')


def test_no_unannotated_get_defaults() -> None:
    violating = []
    for root_name in ["src/xvs", "scripts/xvs"]:
        search_dir = REPO_ROOT / root_name
        for py_path in search_dir.rglob("*.py"):
            content = py_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            # 1. Regex line-by-line check
            for idx, line in enumerate(lines, 1):
                if "# default-ok:" in line:
                    continue
                if PATTERN_NUM.search(line) or PATTERN_NA.search(line):
                    violating.append(f"{py_path.relative_to(REPO_ROOT)}:{idx}: {line.strip()}")

            # 2. AST-based check (catches multi-line .get() calls)
            try:
                tree = ast.parse(content, filename=str(py_path))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                    if len(node.args) >= 2:
                        default_arg = node.args[1]
                        is_numeric = False
                        is_na = False
                        if isinstance(default_arg, ast.Constant):
                            if isinstance(default_arg.value, (int, float)) and not isinstance(default_arg.value, bool):
                                is_numeric = True
                            elif default_arg.value == "N/A":
                                is_na = True
                        elif isinstance(default_arg, ast.UnaryOp) and isinstance(default_arg.operand, ast.Constant):
                            if isinstance(default_arg.operand.value, (int, float)):
                                is_numeric = True

                        if is_numeric or is_na:
                            call_lines = lines[node.lineno - 1 : (node.end_lineno or node.lineno)]
                            has_annotation = any("# default-ok:" in l for l in call_lines)
                            if not has_annotation:
                                entry = f"{py_path.relative_to(REPO_ROOT)}:{node.lineno}: {call_lines[0].strip()}"
                                if entry not in violating:
                                    violating.append(entry)

    assert not violating, f"Found {len(violating)} unannotated .get() calls with numeric or N/A defaults:\n" + "\n".join(violating[:25])

