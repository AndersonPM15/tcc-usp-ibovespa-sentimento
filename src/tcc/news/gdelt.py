"""Coleta de manchetes em português na API DOC 2.0 do GDELT (notebook 12 da versão do artigo).

A base do artigo vem 100% do GDELT (91.941 manchetes de 508 domínios, 02/01/2018 a
19/11/2025). A API devolve até 250 artigos por consulta, então a coleta é feita dia a dia.
Atenção: o GDELT não garante a mesma resposta em coletas diferentes; por isso o artigo é
reproduzido a partir da base já coletada (ver `data/MANIFEST.md`), não de uma coleta nova.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = "(Ibovespa OR Bovespa OR B3 OR 'Bolsa de valores' OR ações OR mercado) sourcelang:por"
MAX_RECORDS = 250
MIN_TITLE_LENGTH_COLLECTOR = 15
MIN_TITLE_LENGTH_BASE = 20
MIN_DISTINCT_DAYS = 200
COLUMNS = ["date", "source", "title", "url", "text_full"]


def fetch_day(
    session: Any, day: date, query: str = QUERY, timeout: float = 30
) -> list[dict[str, Any]]:
    """Artigos de um dia (00:00:00 a 23:59:59). Levanta exceção em erro HTTP."""
    stamp = day.strftime("%Y%m%d")
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": MAX_RECORDS,
        "format": "json",
        "startdatetime": f"{stamp}000000",
        "enddatetime": f"{stamp}235959",
        "sort": "hybridrel",
    }
    response = session.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    articles: list[dict[str, Any]] = response.json().get("articles", [])
    return articles


def collect(
    start: date,
    end: date,
    session: Any,
    query: str = QUERY,
    max_attempts: int = 3,
    pause_seconds: float = 1.5,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, list[date]]:
    """Coleta dia a dia de `start` a `end`; devolve os artigos e os dias que falharam.

    Correção em relação ao coletor original: um dia com erro (inclusive limite de
    requisições) é tentado até `max_attempts` vezes e, se ainda falhar, é informado em vez
    de ser descartado em silêncio.
    """
    articles: list[dict[str, Any]] = []
    failed: list[date] = []
    day = start
    while day <= end:
        for attempt in range(1, max_attempts + 1):
            try:
                articles.extend(fetch_day(session, day, query))
                break
            except Exception:  # rede, HTTP ou JSON inválido: tenta de novo e depois registra
                if attempt == max_attempts:
                    failed.append(day)
                sleep(pause_seconds * attempt)
        sleep(pause_seconds)
        day += timedelta(days=1)
    return normalize_articles(pd.DataFrame(articles)), failed


def normalize_articles(raw: pd.DataFrame) -> pd.DataFrame:
    """Converte a resposta do GDELT nas colunas da base (data sem hora, domínio como fonte).

    Remove artigos sem data ou título, títulos com menos de 15 caracteres, sem URL e URLs
    repetidas. O texto completo não é fornecido pela API: `text_full` repete o título.
    """
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    seen = raw["seendate"].astype(str).str.strip()
    dates = pd.to_datetime(seen, format="%Y%m%d%H%M%S", errors="coerce")
    fallback = pd.to_datetime(seen[dates.isna()], errors="coerce", utc=True)
    dates = dates.fillna(fallback.dt.tz_localize(None))
    title = _column(raw, "title", "")
    frame = pd.DataFrame(
        {
            "date": dates.dt.normalize(),
            "source": _column(raw, "domain", "gdelt_unknown"),
            "title": title,
            "url": _column(raw, "url", ""),
            "text_full": title,
        }
    ).dropna(subset=["date", "title"])
    frame = frame[frame["title"].str.len() >= MIN_TITLE_LENGTH_COLLECTOR]
    frame = frame[frame["url"].str.len() > 0]
    return frame.drop_duplicates(subset=["url"]).reset_index(drop=True)


def _column(raw: pd.DataFrame, name: str, default: str) -> pd.Series:
    """Coluna da resposta com valores ausentes trocados por `default`.

    Correção: o coletor original quebrava (`str.fillna`) quando a coluna não vinha.
    """
    if name not in raw:
        return pd.Series(default, index=raw.index)
    return raw[name].fillna(default)


def consolidate(collected: list[pd.DataFrame]) -> pd.DataFrame:
    """Base de notícias (`news_multisource.parquet`).

    Sem URL repetida, ordenada por data e com títulos de pelo menos 20 caracteres.

    Raises:
        RuntimeError: se a base cobrir menos de 200 dias distintos.
    """
    base = pd.concat(collected, ignore_index=True).drop_duplicates(subset=["url"])
    base = base.sort_values("date").reset_index(drop=True).dropna(subset=["date", "title"])
    base = base[base["title"].str.len() >= MIN_TITLE_LENGTH_BASE]
    if base["date"].nunique() < MIN_DISTINCT_DAYS:
        raise RuntimeError(
            f"Base insuficiente: {base['date'].nunique()} dias distintos (mínimo {MIN_DISTINCT_DAYS})."
        )
    return base
