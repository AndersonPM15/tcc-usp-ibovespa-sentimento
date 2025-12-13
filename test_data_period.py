"""Teste que garante que a série do Ibovespa respeita o intervalo oficial
2018-01-02 a 2024-12-31 definido para o TCC."""

from pathlib import Path

import pandas as pd

from src.config.constants import START_DATE, END_DATE
from src.io.paths import DATA_PROCESSED


def test_ibovespa_clean_periodo_oficial():
    csv_path: Path = DATA_PROCESSED / "ibovespa_clean.csv"
    if not csv_path.exists():
        raise AssertionError(f"Arquivo não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise AssertionError("Coluna 'date' não encontrada em ibovespa_clean.csv")

    df["date"] = pd.to_datetime(df["date"])
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    if not (min_date >= START_DATE and max_date <= END_DATE):
        raise AssertionError(
            f"Período de ibovespa_clean.csv fora dos limites oficiais do TCC: "
            f"{min_date} -> {max_date}"
        )
