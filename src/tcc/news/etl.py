"""Limpeza e deduplicação da base de notícias (notebook 13 da versão do artigo).

Entrada: `data_raw/news_multisource.parquet` (colunas do coletor GDELT). Saída:
`data_interim/news_clean_multisource.parquet` (`date`, `title`, `text`, `source`, `url`,
`origin`), com URLs normalizadas e sem duplicatas.

Comportamento preservado do artigo (bug conhecido): o coletor grava o texto na coluna
`text_full`, mas a limpeza procurava `text` e `origin`; sem encontrá-las, gravava a palavra
"None" nas duas colunas de todas as linhas. O "None" chega ao TF-IDF (ver `tcc.news.text`).
Corrigir isso mudaria a matriz TF-IDF e os números do artigo, então o padrão o mantém.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import pandas as pd

MISSING_FIELD = "None"  # str(None): valor gravado em `text` e `origin` na base do artigo
TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
UID_TITLE_CHARS = 80


def normalize_url(url: object) -> str:
    """URL sem parâmetros de rastreamento (utm_*, fbclid…), com host minúsculo e sem "/" final."""
    if not isinstance(url, str) or not url.strip() or url.strip().lower() in {"nan", "none"}:
        return ""
    try:
        parts = urlparse(url.strip())
        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=False)
            if not (key.lower().startswith("utm_") or key.lower() in TRACKING_PARAMETERS)
        ]
        return urlunparse(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path.rstrip("/"),
                "",
                urlencode(query, doseq=True),
                "",
            )
        )
    except ValueError:  # URL malformada (ex.: IPv6 inválido): mantém como veio
        return url.strip()


def deduplicate(raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza as colunas e remove duplicatas pela URL normalizada (ou data + título).

    Linhas sem data válida ou sem título saem antes da deduplicação, que mantém a 1ª
    ocorrência de cada chave.
    """
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"].astype(str), errors="coerce"),
            "title": raw["title"].astype(str).str.strip(),
            "text": MISSING_FIELD,
            "source": raw["source"].astype(str).str.strip(),
            "url": raw["url"].astype(str).str.strip(),
            "origin": MISSING_FIELD,
        }
    )
    frame = frame.dropna(subset=["date"])
    frame = frame[frame["title"].str.len() > 0].copy()
    frame["url"] = frame["url"].map(normalize_url)
    title_key = frame["title"].str[:UID_TITLE_CHARS].str.replace(r"\W+", "_", regex=True)
    frame["uid"] = np.where(
        frame["url"].str.len() > 0,
        frame["url"],
        frame["date"].dt.strftime("%Y-%m-%d") + "_" + title_key,
    )
    deduplicated = frame.drop_duplicates(subset=["uid"]).reset_index(drop=True)
    return deduplicated[["date", "title", "text", "source", "url", "origin"]]
