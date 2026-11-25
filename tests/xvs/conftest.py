"""Pytest configuration and fixtures for OpenXVS tests.

Scoped strictly to tests/xvs/ to avoid affecting existing repo tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from xvs.paths import get_data_root, set_data_root, data_root_source


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--data-root",
        action="store",
        default=None,
        help="Root data directory containing xvs/ output hierarchy (defaults to XVS_DATA_ROOT or repo output/)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "render: Browser-based rendering tests using Playwright & CesiumJS",
    )
    config.addinivalue_line(
        "markers",
        "data: Tests requiring external or pre-generated datasets in XVS_DATA_ROOT or output/",
    )
    config.addinivalue_line(
        "markers",
        "bench: Benchmark tests measuring runtime, excluded from fast suite",
    )

    data_root_opt = config.getoption("--data-root", default=None)
    if data_root_opt:
        p = Path(data_root_opt).resolve()
        set_data_root(p)
        os.environ["XVS_DATA_ROOT"] = str(p)

    # Protect against Windows temp directory permission issue with pytest-of-<user>
    if getattr(config.option, "basetemp", None) is None:
        try:
            import getpass, tempfile
            user = getpass.getuser()
            test_dir = Path(tempfile.gettempdir()) / f"pytest-of-{user}"
            if test_dir.exists():
                list(os.scandir(test_dir))
        except (PermissionError, OSError):
            import uuid
            basetemp = Path(tempfile.gettempdir()) / f"pytest_temp_{uuid.uuid4().hex[:8]}"
            basetemp.mkdir(parents=True, exist_ok=True)
            config.option.basetemp = str(basetemp)
            print(f"[conftest] Windows basetemp workaround activated: using {basetemp}")


def is_explicit_data_root(config: pytest.Config | None = None) -> bool:
    """Return True if data root was explicitly set via --data-root CLI option or XVS_DATA_ROOT env var."""
    if config is not None:
        try:
            if config.getoption("--data-root", default=None):
                return True
        except (ValueError, AttributeError):
            pass
    if os.environ.get("XVS_DATA_ROOT"):
        return True
    return data_root_source() in ("arg", "set", "env")


def require_data_path(
    path: Path | str,
    desc: str = "Dataset",
    request: pytest.FixtureRequest | None = None,
) -> Path:
    """Verify that a required dataset path exists.

    Public API rule: if data root is explicit, missing data FAILs; otherwise SKIPs.
    """
    p = Path(path)
    if not p.exists():
        cfg = request.config if request is not None else None
        if is_explicit_data_root(cfg):
            pytest.fail(f"Required data not found under explicit data root: {desc} ({p})")
        else:
            pytest.skip(f"Required data not found (default data root): {desc} ({p})")
    return p


@pytest.fixture
def check_data(request: pytest.FixtureRequest):
    """Fixture providing data path checking: FAIL if explicit data root, SKIP otherwise."""
    def _check(path: Path | str, desc: str = "Dataset") -> Path:
        return require_data_path(path, desc=desc, request=request)
    return _check


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Exclude bench tests from the fast suite unless explicitly requested with -m bench."""
    mark_expr = config.getoption("-m", default="")
    if "bench" not in mark_expr:
        deselected = []
        remaining = []
        for item in items:
            if item.get_closest_marker("bench"):
                deselected.append(item)
            else:
                remaining.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = remaining
