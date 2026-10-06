"""Configuração central: onde estão os dados e os parâmetros fixos do estudo.

O único valor que muda de máquina para máquina é a pasta dos dados, lida da variável de
ambiente `TCC_USP_BASE` (ou do arquivo `.env` na raiz do repositório; ver `.env.example`).
Todos os demais caminhos e parâmetros são derivados daqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR_ENV = "TCC_USP_BASE"

# Período analisado no artigo (Tabelas 1, 3 e 4; Figuras 1 a 8).
ARTICLE_START = pd.Timestamp("2018-01-02")
ARTICLE_END = pd.Timestamp("2024-12-31")

RANDOM_SEED = 42
MODEL_LABELS = {"logreg_l2": "Regressão Logística", "rf_200": "Random Forest"}


@dataclass(frozen=True)
class Settings:
    """Pastas de entrada (dados) e de saída (relatórios) de uma execução."""

    base_dir: Path
    reports_dir: Path = field(default_factory=lambda: Path("reports"))

    @property
    def raw_dir(self) -> Path:
        """Dados brutos: coleta do GDELT e base consolidada de notícias."""
        return self.base_dir / "data_raw"

    @property
    def interim_dir(self) -> Path:
        """Notícias deduplicadas, entrada do pré-processamento."""
        return self.base_dir / "data_interim"

    @property
    def processed_dir(self) -> Path:
        """Ibovespa e matriz TF-IDF, entradas da reprodução."""
        return self.base_dir / "data_processed"

    @property
    def figures_dir(self) -> Path:
        """Tabelas e figuras do artigo."""
        return self.reports_dir / "figures"

    @property
    def verification_dir(self) -> Path:
        """Resultados das verificações pós-submissão."""
        return self.reports_dir / "verificacao"

    @property
    def presentation_dir(self) -> Path:
        """Figuras dos slides da apresentação."""
        return self.reports_dir / "apresentacao"


def load_settings(base_dir: Path | None = None, reports_dir: Path | None = None) -> Settings:
    """Monta a configuração a partir dos argumentos, do ambiente ou do `.env`.

    Raises:
        RuntimeError: se a pasta dos dados não foi informada.
    """
    load_dotenv(Path.cwd() / ".env", override=False)
    base = base_dir or os.environ.get(BASE_DIR_ENV)
    if not base:
        raise RuntimeError(
            f"Informe a pasta dos dados: defina {BASE_DIR_ENV} no ambiente ou no arquivo .env "
            "(ver .env.example), ou use a opção --data-dir."
        )
    reports = reports_dir if reports_dir is not None else Path("reports")
    return Settings(base_dir=Path(base).expanduser().resolve(), reports_dir=reports)
