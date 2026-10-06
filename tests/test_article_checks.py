"""Comparação com os números publicados (tcc.article)."""

from __future__ import annotations

import pandas as pd

from tcc import article


def test_formatted_tables_are_compared_as_published_text() -> None:
    # Regressão: ler a referência como número transformava "-0.080" em "-0.08" e acusava
    # diferença numa tabela idêntica ao artigo.
    published = article.published_table("tabela_3", as_text=True)
    assert "-0.080" in published["sharpe"].tolist()
    assert article.check_formatted_table("Tabela 3", published.copy(), "tabela_3").ok


def test_a_changed_cell_is_detected() -> None:
    changed = article.published_table("tabela_4", as_text=True)
    changed.loc[0, "sharpe"] = "-0.081"
    assert not article.check_formatted_table("Tabela 4", changed, "tabela_4").ok


def test_published_numbers_cover_the_article_items() -> None:
    numbers = article.published_numbers()
    assert numbers["tabela_1"] == {
        "pregoes_ibovespa": 1737,
        "dias_com_sentimento": 1341,
        "intersecao": 1341,
    }
    assert set(numbers["tabela_2"]) == {"logreg_l2", "rf_200"}
    assert article.check_figure4(numbers["figura_4_pearson"]).ok
    assert not article.check_figure4(numbers["figura_4_pearson"] + 1e-6).ok


def test_table2_check_uses_the_published_values() -> None:
    rows = [
        {"model": model, **values}
        for model, values in article.published_numbers()["tabela_2"].items()
    ]
    assert article.check_table2(pd.DataFrame(rows)).ok
    rows[0]["auc"] += 1e-6
    assert not article.check_table2(pd.DataFrame(rows)).ok
