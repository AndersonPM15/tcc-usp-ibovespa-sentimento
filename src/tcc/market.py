"""Dados de mercado: Ibovespa diário, retornos, rótulo do classificador e variáveis técnicas.

O download segue o script que gerou os dados do artigo (`download_ibovespa_full.py`,
novembro de 2025, preservado no histórico do git): `^BVSP` no Yahoo Finance, sem ajuste,
de 02/01/2018 a 18/11/2025. Um download feito em 06/10/2026 reproduziu o arquivo do artigo
sem nenhuma diferença nas cotações.
"""

from __future__ import annotations

import pandas as pd

TICKER = "^BVSP"
DOWNLOAD_START = "2018-01-02"
DOWNLOAD_END_EXCLUSIVE = "2025-11-19"  # o yfinance não inclui o dia final
CLEAN_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "return",
    "direction",
]
RETURN_LAGS = 5
VOLATILITY_WINDOWS = (5, 20)


def download_ibovespa(
    start: str = DOWNLOAD_START, end: str | None = DOWNLOAD_END_EXCLUSIVE
) -> pd.DataFrame:
    """Baixa as cotações diárias sem ajuste (requer o extra `pipeline` e internet)."""
    import yfinance as yf  # dependência opcional: só quem baixa dados precisa dela

    raw = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f"O Yahoo Finance não retornou dados para {TICKER} ({start} a {end}).")
    return raw


def prepare_ibovespa(raw: pd.DataFrame) -> pd.DataFrame:
    """Converte a tabela do yfinance nas colunas de `ibovespa_clean.csv`.

    `return` é o retorno simples de fechamento a fechamento (vazio no 1º dia) e
    `direction` vale 1 quando esse retorno é positivo.
    """
    frame = raw.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):  # yfinance recente: (campo, ticker)
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    frame = frame.rename(columns={"index": "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["return"] = frame["close"].pct_change()
    frame["direction"] = (frame["return"] > 0).astype(int)
    return frame[CLEAN_COLUMNS]


def daily_returns(ibovespa: pd.DataFrame) -> pd.DataFrame:
    """Fechamento e retorno do dia (de D−1 para D), com a data na coluna `day`."""
    frame = ibovespa.sort_values("date")
    return pd.DataFrame(
        {
            "day": pd.to_datetime(frame["date"]).to_numpy(),
            "close": frame["close"].to_numpy(),
            "ret": frame["close"].pct_change().to_numpy(),
        }
    )


def build_labels(tfidf_index: pd.DataFrame, ibovespa: pd.DataFrame) -> pd.DataFrame:
    """Alvo do classificador para cada dia da matriz TF-IDF (lógica do notebook 15).

    `ret_next` é o retorno do fechamento de D para o do pregão seguinte e `y` vale 1 quando
    ele é positivo. Dias sem pregão (ou o último pregão) ficam sem rótulo.
    """
    market = ibovespa.sort_values("date").reset_index(drop=True)
    ret_next = market["close"].pct_change().shift(-1)
    labels = pd.DataFrame(
        {
            "day": pd.to_datetime(market["date"]).dt.floor("D"),
            "y": (ret_next > 0).astype(int),
            "ret_next": ret_next,
            "close": market["close"],
        }
    )
    labels = labels.dropna().drop_duplicates("day")
    return tfidf_index.merge(labels, how="left", on="day")


def technical_features(
    close: pd.Series,
    n_lags: int = RETURN_LAGS,
    volatility_windows: tuple[int, ...] = VOLATILITY_WINDOWS,
) -> pd.DataFrame:
    """Retornos defasados e volatilidade histórica, indexados pela data D.

    Só usam informação disponível no fechamento de D (o alvo é a direção de D para D+1):
    `ret_lag_k` é o retorno de D−k−1 para D−k e `vol_w` o desvio-padrão dos últimos `w`
    retornos (mínimo de 2). Valores iniciais indisponíveis viram 0; só ocorrem no 1º bloco
    de treino.
    """
    returns = close.sort_index().pct_change()
    columns = {f"ret_lag_{lag}": returns.shift(lag) for lag in range(n_lags)}
    for window in volatility_windows:
        columns[f"vol_{window}"] = returns.rolling(window, min_periods=2).std()
    return pd.DataFrame(columns).fillna(0.0)
