"""
Config loader utilities for the TCC USP project.

Provides a single entry-point to read `configs/config_tcc.yaml` and expose
typed helpers for downstream notebooks / scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_FILE = Path(__file__).resolve().parents[2] / "configs" / "config_tcc.yaml"

_CONFIG_CACHE: Dict[str, Any] | None = None


def load_config(refresh: bool = False) -> Dict[str, Any]:
    """
    Load the YAML configuration as a dictionary (cached in-memory).

    Parameters
    ----------
    refresh : bool, default False
        When True, forces re-reading the YAML from disk.
    """
    global _CONFIG_CACHE
    if refresh or _CONFIG_CACHE is None:
        if not CONFIG_FILE.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {CONFIG_FILE}")
        with CONFIG_FILE.open("r", encoding="utf-8") as fh:
            _CONFIG_CACHE = yaml.safe_load(fh) or {}
    return _CONFIG_CACHE


def get_periodo_estudo() -> Dict[str, str]:
    """Return the configured study period (`start`, `end`)."""
    cfg = load_config()
    periodo = cfg.get("periodo_estudo")
    if not periodo:
        raise KeyError("Configuração 'periodo_estudo' não encontrada no YAML.")
    return periodo


def get_colunas_data() -> Dict[str, str]:
    """Return the mapping of logical dataset names to date column names."""
    cfg = load_config()
    mapping = cfg.get("colunas_data")
    if not mapping:
        raise KeyError("Configuração 'colunas_data' não encontrada no YAML.")
    return mapping


def get_arquivo(nome_logico: str, base_path: Path | None = None) -> Path:
    """
    Resolve a logical file name to an absolute path.

    Parameters
    ----------
    nome_logico : str
        Key inside `arquivos_chave` (e.g., `ibov_clean`, `tfidf_daily_matrix`).
    base_path : pathlib.Path, optional
        Base folder used to resolve relative paths. When omitted, defaults to
        the shared data root detected by `src.io.paths`.
    """
    cfg = load_config()
    mapping = cfg.get("arquivos_chave") or {}
    if nome_logico not in mapping:
        raise KeyError(f"Arquivo lógico '{nome_logico}' não definido no YAML.")

    rel_path = Path(mapping[nome_logico])
    if rel_path.is_absolute():
        return rel_path

    if base_path is None:
        try:
            from src.io import paths as path_utils

            base_path = path_utils.get_data_paths()["base"]
        except Exception:
            base_path = Path.cwd()
    return (base_path / rel_path).resolve()


def get_config_path() -> Path:
    """Return the absolute path to `config_tcc.yaml`."""
    return CONFIG_FILE
