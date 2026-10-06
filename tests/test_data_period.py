"""Garante que a série do Ibovespa respeita o período oficial do estudo (2018-01-02 a 2024-12-31)."""

from pathlib import Path

import pandas as pd
import pytest

from src.config.constants import END_DATE, START_DATE
from src.io.paths import DATA_PROCESSED


def test_ibovespa_clean_within_official_period() -> None:
    csv_path: Path = DATA_PROCESSED / "ibovespa_clean.csv"
    if not csv_path.exists():
        pytest.skip(f"Arquivo não encontrado: {csv_path} (defina TCC_USP_BASE).")

    dates = pd.to_datetime(pd.read_csv(csv_path)["date"])

    assert dates.min().date() >= START_DATE
    assert dates.max().date() <= END_DATE
