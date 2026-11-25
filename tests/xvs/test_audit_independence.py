"""D15: the Director's audit oracle under audit/xvs/ must stay independent of the implementation.

Fails if any file under audit/xvs/ contains an import of the implementation package, a path insertion of the
repo's `src` folder, a dynamic import of the package, or a sandbox path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "audit" / "xvs"
FORBIDDEN = {
    "import of the implementation package": re.compile(r"\bimport\s+xvs\b|\bfrom\s+xvs\b"),
    "dynamic import of the implementation package": re.compile(r"import_module\(\s*['\"]xvs"),
    "path insertion of the repo src folder": re.compile(r"['\"]src['\"]"),
    "sandbox path": re.compile(r"/mnt/|user-data|g2_work"),
}
FILES = sorted(p for p in AUDIT.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def test_audit_folder_exists_with_oracle():
    names = {p.name for p in FILES}
    assert {"oracle.py", "camera_model.py", "audit_g2.py", "raster_xor.py", "README.md"} <= names


@pytest.mark.parametrize("path", FILES, ids=[p.relative_to(AUDIT).as_posix() for p in FILES])
def test_no_implementation_imports_or_sandbox_paths(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    for what, rx in FORBIDDEN.items():
        m = rx.search(text)
        assert m is None, f"{path.relative_to(REPO_ROOT).as_posix()}: {what} ({m.group(0)!r})"
