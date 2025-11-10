"""
Centralized path helpers for the TCC USP project.

This module standardizes how notebooks/scripts refer to the shared folders both
locally (Windows) and on Google Colab.  Always import these helpers instead of
hardcoding `/content/drive/...` or `C:/Users/...`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

__all__ = ["detect_env", "get_base_path", "get_data_paths", "get_project_paths"]

_COLAB_SENTINELS = ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_BACKEND_VERSION")
_COLAB_DRIVE_ROOT = Path("/content/drive")
_COLAB_DRIVE_FOLDERS = ("MyDrive", "MeuDrive")
_ENV_OVERRIDE_VAR = "TCC_USP_BASE"

_DATA_DIRS = ("data_raw", "data_processed", "data_interim")
_PROJECT_DIRS = ("data", "notebooks", "configs", "reports", "src")


def detect_env() -> str:
    """
    Return the current runtime environment identifier.

    Returns
    -------
    str
        `"colab"` when running inside Google Colab, otherwise `"local"`.
    """
    if any(marker in os.environ for marker in _COLAB_SENTINELS):
        return "colab"
    if _COLAB_DRIVE_ROOT.exists():
        # When Drive is mounted but COLAB_* vars are absent (rare but possible).
        return "colab"
    return "local"


def _repo_root() -> Path:
    """Return the repository root (two levels above src/io)."""
    return Path(__file__).resolve().parents[2]


def _colab_base() -> Path:
    """Locate `/content/drive/<Drive>/TCC_USP` on Colab."""
    if not _COLAB_DRIVE_ROOT.exists():
        raise RuntimeError(
            "Google Drive não está montado. Execute `drive.mount('/content/drive')`."
        )

    for folder in _COLAB_DRIVE_FOLDERS:
        candidate = _COLAB_DRIVE_ROOT / folder / "TCC_USP"
        if candidate.exists():
            return candidate
    # Default to MyDrive even if the folder has not been created yet.
    return _COLAB_DRIVE_ROOT / _COLAB_DRIVE_FOLDERS[0] / "TCC_USP"


def _local_base() -> Path:
    """Infer the local base folder `.../OneDrive/TCC_USP`."""
    repo_parent = _repo_root().parent
    explicit = Path("C:/Users/ander/OneDrive/TCC_USP")
    onedrive_default = Path.home() / "OneDrive" / "TCC_USP"

    candidates = []
    for candidate in (repo_parent, onedrive_default, explicit):
        resolved = candidate.expanduser()
        key = resolved.as_posix().lower()
        if key not in {c.as_posix().lower() for c in candidates}:
            candidates.append(resolved)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_base_path() -> Path:
    """
    Resolve the canonical `BASE_PATH` where shared data lives.

    Precedence:
    1. Environment variable `TCC_USP_BASE`
    2. Colab detection  → `/content/drive/{MyDrive|MeuDrive}/TCC_USP`
    3. Local detection  → parent folder that contains the repository (fallback to
       `%USERPROFILE%/OneDrive/TCC_USP`)
    """
    override = os.environ.get(_ENV_OVERRIDE_VAR)
    if override:
        return Path(override).expanduser().resolve()

    env = detect_env()
    base = _colab_base() if env == "colab" else _local_base()
    return base.resolve()


def get_data_paths(create: bool = True) -> Dict[str, Path]:
    """
    Return a dictionary with the shared data folders.

    Parameters
    ----------
    create : bool, default True
        Create the directories when they do not exist.

    Returns
    -------
    dict
        Keys: `base`, `data_raw`, `data_processed`, `data_interim`.
    """
    base = get_base_path()
    mapping: Dict[str, Path] = {"base": base}

    for name in _DATA_DIRS:
        path = base / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        mapping[name] = path
    return mapping


def get_project_paths() -> Dict[str, Path]:
    """
    Return a dictionary with important folders inside the repository.

    Returns
    -------
    dict
        Keys: `repo_root`, `data`, `notebooks`, `configs`, `reports`, `src`.
    """
    repo_root = _repo_root()
    mapping: Dict[str, Path] = {"repo_root": repo_root}
    for name in _PROJECT_DIRS:
        mapping[name] = repo_root / name
    return mapping
