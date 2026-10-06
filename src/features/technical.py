"""
Variáveis técnicas do Ibovespa para a verificação pós-submissão (e).

Todas usam apenas informação disponível no fechamento do dia D, já que o alvo é a
direção do retorno de D para D+1.
"""

from __future__ import annotations

import pandas as pd

RETURN_LAGS = 5
VOLATILITY_WINDOWS = (5, 20)


def technical_features(
    close: pd.Series,
    n_lags: int = RETURN_LAGS,
    volatility_windows: tuple[int, ...] = VOLATILITY_WINDOWS,
) -> pd.DataFrame:
    """Retornos defasados e volatilidade histórica, indexados pela data D.

    - `ret_lag_k`: retorno de D−k−1 para D−k (k = 0 é o retorno do próprio dia D);
    - `vol_w`: desvio-padrão dos últimos w retornos até D (mínimo de 2 observações).
    Valores iniciais indisponíveis viram 0; eles só ocorrem no primeiro bloco de treino.
    """
    returns = close.sort_index().pct_change()
    columns = {f"ret_lag_{k}": returns.shift(k) for k in range(n_lags)}
    for window in volatility_windows:
        columns[f"vol_{window}"] = returns.rolling(window, min_periods=2).std()
    return pd.DataFrame(columns).fillna(0.0)
