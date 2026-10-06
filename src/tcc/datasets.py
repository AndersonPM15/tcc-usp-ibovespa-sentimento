"""Leitura dos arquivos de entrada e recorte do período do artigo.

As entradas da reprodução são três arquivos em `data_processed/` (ver `data/MANIFEST.md`):
o Ibovespa diário e a matriz TF-IDF diária com o seu índice de dias.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, load_npz

from tcc.config import ARTICLE_END, ARTICLE_START, Settings

IBOVESPA_FILE = "ibovespa_clean.csv"
TFIDF_MATRIX_FILE = "tfidf_daily_matrix.npz"
TFIDF_INDEX_FILE = "tfidf_daily_index.csv"
NEWS_CLEAN_FILE = "news_clean_multisource.parquet"
NEWS_RAW_FILE = "news_multisource.parquet"


def clamp_period(
    frame: pd.DataFrame,
    column: str,
    start: pd.Timestamp = ARTICLE_START,
    end: pd.Timestamp = ARTICLE_END,
) -> pd.DataFrame:
    """Mantém as linhas com `column` entre `start` e `end` (inclusive)."""
    dates = pd.to_datetime(frame[column], errors="coerce")
    return frame.loc[(dates >= start) & (dates <= end)].assign(**{column: dates})


def read_ibovespa(settings: Settings) -> pd.DataFrame:
    """Ibovespa diário completo (02/01/2018 a 18/11/2025), ordenado por data."""
    frame = pd.read_csv(settings.processed_dir / IBOVESPA_FILE, parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)


def read_tfidf(settings: Settings) -> tuple[csr_matrix, pd.DataFrame]:
    """Matriz TF-IDF diária (um documento por dia) e o índice `day`/`row_id` das linhas."""
    matrix = load_npz(settings.processed_dir / TFIDF_MATRIX_FILE).tocsr()
    index = pd.read_csv(settings.processed_dir / TFIDF_INDEX_FILE, parse_dates=["day"])
    return matrix, index


def missing_inputs(settings: Settings, names: tuple[str, ...]) -> list[Path]:
    """Arquivos de `data_processed/` que não existem."""
    paths = [settings.processed_dir / name for name in names]
    return [path for path in paths if not path.exists()]
