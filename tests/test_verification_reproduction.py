"""
Regressão das verificações pós-submissão: recalculadas, devem sair iguais às tabelas
versionadas em `reports/verificacao/` (tolerância de 1e-9, ruído de ponto flutuante).

Leva cerca de 2 minutos (três walk-forwards e os bootstraps). Sem os dados, é pulado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcc import reproduce
from tcc.config import Settings

COMMITTED = Path(__file__).resolve().parents[1] / "reports" / "verificacao"
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def tables(
    settings: Settings, article_results: reproduce.ArticleResults
) -> dict[str, pd.DataFrame]:
    produced = reproduce.write_verifications(article_results, settings)
    return {name: pd.read_csv(settings.verification_dir / f"{name}.csv") for name in produced}


def test_every_committed_table_is_reproduced(tables: dict[str, pd.DataFrame]) -> None:
    assert sorted(tables) == sorted(path.stem for path in COMMITTED.glob("*.csv"))


@pytest.mark.parametrize("name", sorted(path.stem for path in COMMITTED.glob("*.csv")))
def test_verification_table_unchanged(tables: dict[str, pd.DataFrame], name: str) -> None:
    committed = pd.read_csv(COMMITTED / f"{name}.csv")
    produced = tables[name]
    assert list(produced.columns) == list(committed.columns)
    assert produced.shape == committed.shape
    for column in committed.columns:
        if pd.api.types.is_numeric_dtype(committed[column]):
            np.testing.assert_allclose(
                produced[column], committed[column], rtol=0, atol=TOLERANCE, err_msg=column
            )
        else:
            assert (produced[column].astype(str) == committed[column].astype(str)).all(), column
