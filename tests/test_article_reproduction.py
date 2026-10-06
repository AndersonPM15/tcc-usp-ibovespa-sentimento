"""
Teste de regressão: a reprodução deve sair igual aos números publicados no artigo
(XXIX SemeAd, 2026, artigo 160).

Cobertura: Tabelas 1 a 4, Figuras 3, 4, 7B e 8 e os 270 eventos do estudo de eventos,
recalculados a partir das três entradas (Ibovespa e matriz TF-IDF com índice). As
referências e tolerâncias estão em `tcc.article`. Sem os dados (TCC_USP_BASE), os testes
são pulados com aviso.
"""

from __future__ import annotations

import pytest

from tcc import reproduce
from tcc.article import Check

ARTICLE_ITEMS = [
    "Tabela 1",
    "Tabela 2",
    "Tabela 3",
    "Tabela 4",
    "Figuras 3 e 8",
    "Figura 4",
    "Eventos",
    "Figura 7B",
]


@pytest.fixture(scope="module")
def checks(article_results: reproduce.ArticleResults) -> dict[str, Check]:
    return {check.item: check for check in reproduce.compare_with_article(article_results)}


def test_every_article_item_is_compared(checks: dict[str, Check]) -> None:
    assert sorted(checks) == sorted(ARTICLE_ITEMS)


@pytest.mark.parametrize("item", ARTICLE_ITEMS)
def test_article_number_reproduced(checks: dict[str, Check], item: str) -> None:
    assert checks[item].ok, checks[item].detail


def test_sample_sizes(article_results: reproduce.ArticleResults) -> None:
    assert article_results.table1["Dias"].tolist() == [1737, 1341, 1341]
    assert len(article_results.oof_full) == 2 * 1552
    assert len(article_results.events) == 270
