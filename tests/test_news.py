"""Coleta GDELT, deduplicação, pré-processamento e TF-IDF (tcc.news)."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz

from tcc.config import Settings
from tcc.datasets import NEWS_CLEAN_FILE, TFIDF_MATRIX_FILE
from tcc.news import etl, gdelt, text

STOPWORDS = frozenset({"de", "a", "o", "em"})


@pytest.fixture
def small_stopwords(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita baixar o corpus do NLTK nos testes unitários."""
    monkeypatch.setattr(text, "portuguese_stopwords", lambda: STOPWORDS)


# --------------------------------------------------------------------------- GDELT


class FakeResponse:
    def __init__(self, articles: list[dict[str, Any]] | None) -> None:
        self.articles = articles

    def raise_for_status(self) -> None:
        if self.articles is None:
            raise RuntimeError("HTTP 500")

    def json(self) -> dict[str, Any]:
        return {"articles": self.articles}


class FakeSession:
    """Responde por dia; `None` simula erro HTTP."""

    def __init__(self, answers: dict[str, list[list[dict[str, Any]] | None]]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def get(self, _url: str, params: dict[str, Any], timeout: float) -> FakeResponse:
        day = params["startdatetime"][:8]
        self.calls.append(day)
        return FakeResponse(self.answers[day].pop(0))


def _article(
    day: str, url: str, title: str = "Ibovespa fecha em alta com bancos"
) -> dict[str, Any]:
    return {
        "seendate": f"{day}T120000Z".replace("T", "").replace("Z", ""),
        "url": url,
        "title": title,
        "domain": "site.com",
    }


def test_collect_retries_failed_days_and_reports_the_ones_that_never_answer() -> None:
    # Regressão: o coletor original pulava em silêncio o dia com erro (coleta não reproduzível).
    session = FakeSession(
        {
            "20240102": [None, [_article("20240102", "https://a.com/1")]],
            "20240103": [None, None, None],
        }
    )
    articles, failed = gdelt.collect(
        date(2024, 1, 2), date(2024, 1, 3), session, sleep=lambda _: None
    )
    assert len(articles) == 1
    assert failed == [date(2024, 1, 3)]
    assert session.calls.count("20240103") == 3  # número limitado de tentativas


def test_normalize_articles_filters_and_tolerates_missing_domain() -> None:
    raw = pd.DataFrame(
        {
            "seendate": ["20240102120000", "20240102130000", "20240103090000", "lixo"],
            "url": ["https://a.com/1", "https://a.com/1", "https://a.com/2", "https://a.com/3"],
            "title": [
                "Ibovespa fecha em alta",
                "Ibovespa fecha em alta",
                "curto",
                "Título válido aqui",
            ],
        }
    )
    # Regressão: sem a coluna `domain`, o coletor original quebrava.
    normalized = gdelt.normalize_articles(raw)
    assert normalized["url"].tolist() == ["https://a.com/1"]
    assert normalized["source"].tolist() == ["gdelt_unknown"]
    assert normalized["date"].iloc[0] == pd.Timestamp("2024-01-02")
    assert (normalized["text_full"] == normalized["title"]).all()


def test_consolidate_requires_enough_days() -> None:
    days = pd.date_range("2024-01-01", periods=10)
    small = pd.DataFrame(
        {"date": days, "title": "Manchete longa o suficiente", "url": [f"u{i}" for i in range(10)]}
    )
    with pytest.raises(RuntimeError, match="insuficiente"):
        gdelt.consolidate([small])


# --------------------------------------------------------------------------- ETL


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://Site.COM/a/b/?utm_source=x&id=3&fbclid=9", "https://site.com/a/b?id=3"),
        ("https://site.com/a/", "https://site.com/a"),
        ("None", ""),
        (None, ""),
        ("  ", ""),
    ],
)
def test_normalize_url(url: object, expected: str) -> None:
    assert etl.normalize_url(url) == expected


def test_deduplicate_by_normalized_url_or_date_and_title() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "source": "site.com",
            "title": ["Alta da bolsa hoje", "Alta da bolsa hoje", "Sem link", "Sem link"],
            "url": ["https://a.com/x/", "https://A.com/x?utm_medium=y", "", ""],
        }
    )
    clean = etl.deduplicate(raw)
    assert clean["url"].tolist() == ["https://a.com/x", ""]
    # comportamento do artigo preservado: texto e origem ausentes viram "None"
    assert (clean["text"] == "None").all() and (clean["origin"] == "None").all()


def test_deduplicate_drops_empty_titles() -> None:
    # Regressão: no ETL original, um título vazio virava "None" e a linha era mantida.
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 2),
            "source": "s",
            "title": ["", "Válido"],
            "url": ["u1", "u2"],
        }
    )
    assert etl.deduplicate(raw)["title"].tolist() == ["Válido"]


# --------------------------------------------------------------------------- texto e TF-IDF


def test_headline_text_removes_urls_emails_and_numbers() -> None:
    assert text.headline_text("Bolsa sobe 2,5% www.x.com", "contato@x.com 2024") == "Bolsa sobe %"


def test_clean_text_keeps_letters_without_stopwords() -> None:
    assert text.clean_text("A Bolsa de São Paulo em alta: x %", STOPWORDS) == "bolsa são paulo alta"


def test_keep_valid_tokens_drops_html_residue() -> None:
    assert text.keep_valid_tokens("alta href=x nbsp queda pré-sal") == "alta queda pré-sal"


def test_daily_documents_join_headlines_in_order(small_stopwords: None) -> None:
    news = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"]),
            "title": ["Terceira manchete", "Primeira manchete", "Segunda manchete"],
            "text": "None",
        }
    )
    documents = text.daily_documents(news)
    assert documents["day"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert documents["doc"].iloc[0] == "primeira manchete none segunda manchete none"
    fixed = text.daily_documents(news, missing_text="")
    assert fixed["doc"].iloc[0] == "primeira manchete segunda manchete"


def test_tfidf_matrix_shape_and_index() -> None:
    documents = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=3),
            "doc": ["alta bolsa", "alta dólar", "bolsa dólar"],
        }
    )
    matrix, index, vocabulary = text.tfidf_matrix(documents)
    assert matrix.shape == (3, len(vocabulary))
    assert index["row_id"].tolist() == [0, 1, 2]


def test_tfidf_reproduces_article_matrix(settings: Settings) -> None:
    news_path = settings.interim_dir / NEWS_CLEAN_FILE
    if not news_path.exists():
        pytest.skip("base de notícias (data_interim) indisponível")
    try:
        text.portuguese_stopwords()
    except LookupError:
        pytest.skip("stopwords do NLTK indisponíveis (sem internet)")
    matrix, _, vocabulary = text.tfidf_matrix(text.daily_documents(pd.read_parquet(news_path)))
    article = load_npz(settings.processed_dir / TFIDF_MATRIX_FILE).tocsr()
    assert matrix.shape == article.shape == (2771, 45473)
    assert len(vocabulary) == 45473
    assert matrix.nnz == article.nnz
    assert np.max(np.abs((matrix - article).toarray())) == 0
