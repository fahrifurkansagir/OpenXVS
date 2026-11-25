"""Path resolution and data directory utilities for OpenXVS.

Conforms to OpenXVS project conventions:
- REPO_ROOT dynamically resolved via Path(__file__) traversal.
- Data directory resolved with precedence:
  1. Explicit argument to get_data_root(arg) -> "arg"
  2. Explicitly configured data root via set_data_root(path) -> "set"
  3. XVS_DATA_ROOT environment variable -> "env"
  4. Default: REPO_ROOT / "output" -> "default"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# src/xvs/paths.py -> parents[0] = src/xvs -> parents[1] = src -> parents[2] = repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]

_EXPLICIT_DATA_ROOT: Path | None = None
_LAST_SOURCE: Literal["arg", "set", "env", "default"] = "default"


def get_repo_root() -> Path:
    """Return the repository root directory resolved via Path(__file__)."""
    return REPO_ROOT


def set_data_root(path: Path | str | None) -> None:
    """Explicitly set the data root for the current process."""
    global _EXPLICIT_DATA_ROOT, _LAST_SOURCE
    if path is not None:
        _EXPLICIT_DATA_ROOT = Path(path).resolve()
        _LAST_SOURCE = "set"
    else:
        _EXPLICIT_DATA_ROOT = None
        _LAST_SOURCE = "default"


def data_root_source(data_root_arg: Path | str | None = None) -> Literal["arg", "set", "env", "default"]:
    """Return the source of the data root: 'arg' | 'set' | 'env' | 'default'."""
    if data_root_arg is not None:
        return "arg"
    if _EXPLICIT_DATA_ROOT is not None:
        return "set"
    if os.environ.get("XVS_DATA_ROOT"):
        return "env"
    return "default"


def get_data_root(data_root_arg: Path | str | None = None) -> Path:
    """Return the data root directory.

    Precedence:
    1. Explicit data_root_arg (e.g. from --data-root CLI flag)
    2. Explicitly configured data root via set_data_root()
    3. XVS_DATA_ROOT environment variable
    4. Default: REPO_ROOT / "output"
    """
    global _LAST_SOURCE
    if data_root_arg is not None:
        _LAST_SOURCE = "arg"
        return Path(data_root_arg).resolve()
    if _EXPLICIT_DATA_ROOT is not None:
        _LAST_SOURCE = "set"
        return _EXPLICIT_DATA_ROOT
    env_root = os.environ.get("XVS_DATA_ROOT")
    if env_root:
        _LAST_SOURCE = "env"
        return Path(env_root).resolve()
    _LAST_SOURCE = "default"
    return REPO_ROOT / "output"
