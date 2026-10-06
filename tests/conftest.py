"""Fixtures comuns: os testes que dependem dos dados do artigo são pulados sem eles."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tcc.config import BASE_DIR_ENV, Settings, load_settings
from tcc.datasets import IBOVESPA_FILE, TFIDF_INDEX_FILE, TFIDF_MATRIX_FILE, missing_inputs

# Código antigo (scripts/ e src/ fora do pacote) ainda importado por alguns testes.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPRODUCTION_INPUTS = (IBOVESPA_FILE, TFIDF_MATRIX_FILE, TFIDF_INDEX_FILE)


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Configuração apontando para os dados do artigo; saídas numa pasta temporária."""
    base = os.environ.get(BASE_DIR_ENV)
    if not base:
        pytest.skip(f"{BASE_DIR_ENV} não definida: dados do artigo indisponíveis.")
    config = load_settings(Path(base), tmp_path_factory.mktemp("reports"))
    missing = missing_inputs(config, REPRODUCTION_INPUTS)
    if missing:
        pytest.skip(f"Arquivos de entrada ausentes: {[str(path) for path in missing]}")
    return config
