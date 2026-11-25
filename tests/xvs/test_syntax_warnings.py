"""Test that all Python scripts compile cleanly with SyntaxWarning treated as an error (W6 / R-GB-24)."""

from pathlib import Path
import warnings
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_syntax_warnings_on_compile():
    """Verify compile() triggers zero SyntaxWarning across src, scripts, and tests."""
    py_files = []
    for rel_dir in ["src", "scripts", "tests"]:
        dir_path = REPO_ROOT / rel_dir
        if dir_path.is_dir():
            py_files.extend(dir_path.rglob("*.py"))

    assert len(py_files) > 0, "No Python files found"

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for py_path in py_files:
            source = py_path.read_text(encoding="utf-8")
            try:
                compile(source, str(py_path), "exec")
            except SyntaxWarning as e:
                pytest.fail(f"SyntaxWarning in {py_path}: {e}")
