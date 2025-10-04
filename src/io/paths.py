"""
Utilities for centralized path resolution across local (Windows) and Google Colab
environments. All notebooks should rely on these helpers instead of hardcoding
`/content/drive/...` or `C:/Users/...`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

__all__ = ["get_base_path", "get_data_paths", "get_project_paths"]

_DATA_DIRS = ("data_raw", "data_processed", "data_interim")
_PROJECT_DIRS = ("notebooks", "configs", "reports", "src")
_COLAB_DRIVE_CANDIDATES = (
    Path("/content/drive/MyDrive/TCC_USP"),
    Path("/content/drive/MeuDrive/TCC_USP"),
)
_LOCAL_DEFAULTS = (
    Path("C:/Users/ander/OneDrive/TCC_USP"),
    Path.home() / "OneDrive" / "TCC_USP",
)


def _is_colab_runtime() -> bool:
    """Infer whether the current interpreter is running inside Google Colab."""
    return any(
        env_var in os.environ
        for env_var in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_BACKEND_VERSION")
    )


def _has_data_dirs(path: Path) -> bool:
    """Check whether all expected data directories live inside `path`."""
    return all((path / directory).exists() for directory in _DATA_DIRS)


def _env_override() -> Path | None:
    """Allow power users to specify the base path manually via env var."""
    env_value = os.environ.get("TCC_USP_BASE_PATH")
    if not env_value:
        return None
    override = Path(env_value).expanduser()
    return override if override.exists() else None


def _detect_colab_base() -> Path:
    """Resolve the Drive mount used inside Colab."""
    drive_root = Path("/content/drive")
    if not drive_root.exists():
        return Path.cwd()
    for candidate in _COLAB_DRIVE_CANDIDATES:
        if candidate.exists():
            return candidate
    # Default to MyDrive even if the folder is not there yet.
    return _COLAB_DRIVE_CANDIDATES[0]


def _candidate_local_roots() -> list[Path]:
    """Generate a list of plausible base folders on the local machine."""
    module_path = Path(__file__).resolve()
    parents = list(module_path.parents)
    cwd = Path.cwd()
    candidates = [cwd] + parents
    candidates.extend(_LOCAL_DEFAULTS)
    return candidates


def _detect_local_base() -> Path:
    """Find the best local folder that should contain the shared data dirs."""
    candidates = _candidate_local_roots()
    for candidate in candidates:
        if candidate.exists() and _has_data_dirs(candidate):
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fall back to the repository root (two levels above src/io).
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_base_path() -> Path:
    """
    Detect the canonical BASE_PATH.

    - Colab → `/content/drive/{MyDrive|MeuDrive}/TCC_USP`
    - Local → parent directory that holds `data_raw`, `data_processed`, `data_interim`
      (defaults to the OneDrive workspace if nothing else matches)
    - Manual override → set `TCC_USP_BASE_PATH` before importing this module.
    """
    override = _env_override()
    if override:
        return override.resolve()

    if _is_colab_runtime():
        return _detect_colab_base().resolve()
    return _detect_local_base().resolve()


@lru_cache(maxsize=1)
def get_data_paths() -> Dict[str, Path]:
    """
    Return the three canonical data directories relative to BASE_PATH.

    The directories are created lazily if they do not exist yet to keep the
    pipeline idempotent when run on a fresh machine.
    """
    base_path = get_base_path()
    mapping = {name: (base_path / name) for name in _DATA_DIRS}
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping


@lru_cache(maxsize=1)
def get_project_paths() -> Dict[str, Path]:
    """
    Return important project folders (repo root + notebooks/configs/reports/src).

    These paths are derived from the repository location to keep code + data
    references decoupled: data lives in BASE_PATH, code lives inside the repo.
    """
    repo_root = Path(__file__).resolve().parents[2]
    mapping = {"repo_root": repo_root}
    for directory in _PROJECT_DIRS:
        mapping[directory] = repo_root / directory
    return mapping
