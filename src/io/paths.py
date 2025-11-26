"""
Centralized path helpers for the TCC USP project.
=================================================

Este módulo padroniza como notebooks e scripts referenciam as pastas compartilhadas
tanto localmente (Windows) quanto no Google Colab. Sempre importe estes helpers
ao invés de hardcoding de caminhos como `/content/drive/...` ou `C:/Users/...`.

Estrutura de diretórios esperada:
---------------------------------
    C:/TCC_USP/                     <- BASE_PATH (Windows local)
    ├── data_raw/                   <- Dados brutos (downloads, CSVs originais)
    ├── data_processed/             <- Dados processados (limpos, transformados)
    ├── data_interim/               <- Dados intermediários (cache, temporários)
    └── reports/                    <- Relatórios, logs, outputs finais

Uso típico:
-----------
    from src.io.paths import BASE_PATH, DATA_RAW, DATA_PROCESSED, get_data_paths

    # Usando constantes diretamente
    df = pd.read_csv(DATA_RAW / "ibovespa.csv")

    # Ou via função (garante criação das pastas)
    paths = get_data_paths()
    df.to_parquet(paths["data_processed"] / "output.parquet")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Exportações públicas do módulo
# ---------------------------------------------------------------------------
__all__ = [
    # Constantes de caminho (acesso direto)
    "BASE_PATH",
    "DATA_RAW",
    "DATA_PROCESSED",
    "DATA_INTERIM",
    "REPORTS",
    # Funções auxiliares
    "detect_env",
    "get_base_path",
    "get_data_paths",
    "get_project_paths",
]

# ---------------------------------------------------------------------------
# Configurações internas para detecção de ambiente
# ---------------------------------------------------------------------------
_COLAB_SENTINELS = ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_BACKEND_VERSION")
_COLAB_DRIVE_ROOT = Path("/content/drive")
_COLAB_DRIVE_FOLDERS = ("MyDrive", "MeuDrive")
_ENV_OVERRIDE_VAR = "TCC_USP_BASE"

# Caminho base local fixo (Windows)
_LOCAL_BASE_PATH = Path("C:/TCC_USP")

_DATA_DIRS = ("data_raw", "data_processed", "data_interim")
_PROJECT_DIRS = ("data", "notebooks", "configs", "reports", "src")

# ---------------------------------------------------------------------------
# Funções de detecção de ambiente
# ---------------------------------------------------------------------------


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
    """
    Retorna o caminho base local fixo: C:/TCC_USP.
    
    Prioridade:
    1. Caminho fixo C:/TCC_USP (padrão do projeto)
    2. Fallback para o diretório pai do repositório se C:/TCC_USP não existir
    """
    # Caminho padrão fixo do projeto
    if _LOCAL_BASE_PATH.exists():
        return _LOCAL_BASE_PATH
    
    # Fallback: diretório pai do repositório
    repo_parent = _repo_root().parent
    if repo_parent.exists():
        return repo_parent
    
    # Retorna o padrão mesmo se não existir (será criado depois)
    return _LOCAL_BASE_PATH


def get_base_path() -> Path:
    """
    Resolve o caminho base canônico (BASE_PATH) onde os dados compartilhados residem.

    Precedência:
    1. Variável de ambiente `TCC_USP_BASE` (override manual)
    2. Google Colab → `/content/drive/{MyDrive|MeuDrive}/TCC_USP`
    3. Local (Windows) → `C:/TCC_USP`

    Returns
    -------
    Path
        Caminho absoluto resolvido para a pasta base do projeto.
    """
    override = os.environ.get(_ENV_OVERRIDE_VAR)
    if override:
        return Path(override).expanduser().resolve()

    env = detect_env()
    base = _colab_base() if env == "colab" else _local_base()
    return base.resolve()


def get_data_paths(create: bool = True) -> Dict[str, Path]:
    """
    Retorna um dicionário com as pastas de dados compartilhadas.

    Parameters
    ----------
    create : bool, default True
        Se True, cria os diretórios caso não existam.

    Returns
    -------
    dict
        Chaves: `base`, `data_raw`, `data_processed`, `data_interim`.
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
    Retorna um dicionário com as pastas importantes dentro do repositório.

    Returns
    -------
    dict
        Chaves: `repo_root`, `data`, `notebooks`, `configs`, `reports`, `src`.
    """
    repo_root = _repo_root()
    mapping: Dict[str, Path] = {"repo_root": repo_root}
    for name in _PROJECT_DIRS:
        mapping[name] = repo_root / name
    return mapping


# ---------------------------------------------------------------------------
# Constantes de caminho para acesso direto (conveniência)
# ---------------------------------------------------------------------------

# Caminho base raiz do projeto
BASE_PATH: Path = get_base_path()

# Pasta para dados brutos (downloads, CSVs originais, APIs)
DATA_RAW: Path = BASE_PATH / "data_raw"

# Pasta para dados processados (limpos, transformados, features)
DATA_PROCESSED: Path = BASE_PATH / "data_processed"

# Pasta para dados intermediários (cache, checkpoints, temporários)
DATA_INTERIM: Path = BASE_PATH / "data_interim"

# Pasta para relatórios, logs e outputs finais
REPORTS: Path = BASE_PATH / "reports"
