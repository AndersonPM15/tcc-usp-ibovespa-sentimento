"""Pré-processamento das manchetes e TF-IDF diário (notebooks 14 e 15 da versão do artigo).

Pré-processamento usado no artigo: remoção de URLs, e-mails e números, minúsculas, só letras
(inclusive acentuadas), stopwords do NLTK em português e tokens de 1 caractere. Não há
lematização: na rodada do artigo o spaCy não estava instalado e o notebook caiu nesse
caminho, que é o único mantido aqui.

Cada dia vira um documento (as manchetes concatenadas) e o TF-IDF usa min_df=2,
max_df=0,95 e n-gramas 1–2, sem limite de termos. Aplicado a
`data_interim/news_clean_multisource.parquet`, reproduz a matriz do artigo sem diferença.
"""

from __future__ import annotations

import re
from functools import cache

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from tcc.news.etl import MISSING_FIELD

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
SPACES_RE = re.compile(r"\s+")
NON_LETTERS_RE = re.compile(r"[^a-zà-úç\s]", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"^[a-z0-9áàâãéêíóôõúç\-]+$", flags=re.IGNORECASE)
HTML_RESIDUE = ("href", "font", "color", "nbsp")
TOKEN_PATTERN = r"(?u)\b\w[\w\-áàâãéêíóôõúç]+\b"
MIN_DF = 2
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)


@cache
def portuguese_stopwords() -> frozenset[str]:
    """Stopwords do NLTK em português (baixa o corpus na 1ª vez; requer o extra `pipeline`)."""
    import nltk
    from nltk.corpus import stopwords

    try:
        words = stopwords.words("portuguese")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("portuguese")
    return frozenset(words)


def _squeeze_spaces(text: str) -> str:
    return SPACES_RE.sub(" ", text).strip()


def headline_text(title: str, text: str) -> str:
    """Título + texto, sem URLs, e-mails e números."""
    combined = f"{title} {text}".strip()
    for pattern in (URL_RE, EMAIL_RE, NUMBER_RE):
        combined = pattern.sub(" ", combined)
    return _squeeze_spaces(combined)


def clean_text(text: str, stopwords: frozenset[str]) -> str:
    """Minúsculas, só letras, sem stopwords e sem tokens de 1 caractere."""
    lowered = EMAIL_RE.sub(" ", URL_RE.sub(" ", text.lower()))
    letters = _squeeze_spaces(NON_LETTERS_RE.sub(" ", lowered))
    return " ".join(word for word in letters.split() if word not in stopwords and len(word) > 1)


def keep_valid_tokens(text: str) -> str:
    """Tokens só com letras, dígitos ou hífen e sem resíduos de HTML."""
    tokens = (token.strip() for token in str(text).split())
    return " ".join(
        token.lower()
        for token in tokens
        if token
        and not any(residue in token for residue in HTML_RESIDUE)
        and TOKEN_RE.fullmatch(token)
    )


def daily_documents(news: pd.DataFrame, missing_text: str = MISSING_FIELD) -> pd.DataFrame:
    """Um documento por dia: as manchetes limpas do dia concatenadas, em ordem de data.

    `missing_text` é o texto que acompanha cada manchete. O padrão ("None") reproduz o
    artigo; `""` mostra o efeito de corrigir o bug do "None" (verificação pós-submissão).
    """
    stopwords = portuguese_stopwords()
    texts = news["text"].where(news["text"] != MISSING_FIELD, missing_text)
    cleaned = [
        keep_valid_tokens(clean_text(headline_text(str(title), str(text)), stopwords))
        for title, text in zip(news["title"], texts, strict=True)
    ]
    frame = pd.DataFrame({"day": pd.to_datetime(news["date"]).dt.floor("D"), "doc": cleaned})
    documents = frame.groupby("day")["doc"].agg(lambda docs: " ".join(docs.dropna()))
    return documents.reset_index().sort_values("day").reset_index(drop=True)


def tfidf_matrix(documents: pd.DataFrame) -> tuple[csr_matrix, pd.DataFrame, list[str]]:
    """Matriz TF-IDF (dias × termos), índice `day`/`row_id` e vocabulário em ordem de coluna."""
    vectorizer = TfidfVectorizer(
        min_df=MIN_DF, max_df=MAX_DF, ngram_range=NGRAM_RANGE, token_pattern=TOKEN_PATTERN
    )
    matrix = vectorizer.fit_transform(documents["doc"].fillna(""))
    index = documents[["day"]].assign(row_id=range(len(documents)))
    return matrix.tocsr(), index, list(vectorizer.get_feature_names_out())
