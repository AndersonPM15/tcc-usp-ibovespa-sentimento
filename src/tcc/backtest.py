"""Backtests das Figuras 3 e 8 e das Tabelas 3 e 4, e o buy-and-hold do Ibovespa.

Duas regras de negociação, ambas só compradas, custo de 0,0005 por unidade de turnover:

- **limiar fixo** (Figuras 3 e 8): compra se p ≥ 0,60, vende se p ≤ 0,40, mantém entre os dois;
  a posição do dia D é aplicada ao retorno do dia seguinte da série (`shift(1)`);
- **quantil** (Tabelas 3 e 4): compra se p ≥ quantil *q* de p, vende se p ≤ mediana; a
  posição é aplicada com `lag + 1` dias de defasagem. No artigo os quantis vêm do período
  inteiro (`threshold_mode="full_sample"`, com informação futura); a verificação (c) usa os
  60 pregões anteriores (`"rolling"`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tcc.config import MODEL_LABELS

TRADING_DAYS_PER_YEAR = 252
MIN_DISTINCT_EQUITY_VALUES = 200  # a curva diária precisa variar (proteção contra dados vazios)


@dataclass(frozen=True)
class Strategy:
    """Regra só comprada do artigo (`long_only_60`)."""

    name: str = "long_only_60"
    long_threshold: float = 0.60
    exit_threshold: float = 0.40
    cost: float = 0.0005


ARTICLE_STRATEGY = Strategy()


def _hold_positions(enter: np.ndarray, leave: np.ndarray) -> tuple[list[int], list[int]]:
    """Posições 0/1 que entram em `enter`, saem em `leave` e mantêm o estado entre os dois."""
    positions, turnovers, previous = [], [], 0
    for go_long, go_flat in zip(enter, leave, strict=True):
        position = 1 if go_long else 0 if go_flat else previous
        turnovers.append(abs(position - previous))
        positions.append(position)
        previous = position
    return positions, turnovers


def run_threshold_strategy(oof_model: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
    """Regra de limiar fixo sobre a série diária de p de um modelo (Figuras 3 e 8)."""
    frame = oof_model.reset_index(drop=True).copy()
    proba = frame["proba"].to_numpy()
    positions, turnovers = _hold_positions(
        proba >= strategy.long_threshold, proba <= strategy.exit_threshold
    )
    frame["signal"] = positions
    frame["turnover"] = turnovers
    frame["cost"] = frame["turnover"] * strategy.cost
    effective = frame["signal"].shift(1, fill_value=0)
    frame["strategy_ret"] = effective * frame["ret_next"].fillna(0) - frame["cost"]
    frame["equity"] = (1 + frame["strategy_ret"]).cumprod()
    return frame


def threshold_backtest_curves(
    oof: pd.DataFrame, returns: pd.DataFrame, strategy: Strategy = ARTICLE_STRATEGY
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curvas da Figura 8 (normalizadas em 1 no 1º dia) e Sharpe/CAGR da Figura 3.

    O Ibovespa entra com o retorno do mesmo dia (D−1 → D) nas datas das estratégias.
    """
    common_days = sorted(set(oof["day"]) & set(returns["day"]))
    benchmark = returns.set_index("day").loc[common_days, "ret"].fillna(0)
    benchmark_equity = (1 + benchmark).cumprod()
    curves = pd.DataFrame(
        {"date": common_days, "equity_ibov": (benchmark_equity / benchmark_equity.iloc[0]).values}
    )
    summary = []
    for model in MODEL_LABELS:
        run = run_threshold_strategy(oof[oof["model"] == model].sort_values("day"), strategy)
        run = run[run["day"].isin(common_days)].sort_values("day")
        ret = run["strategy_ret"].fillna(0)
        equity = (1 + ret).cumprod()
        equity = equity / equity.iloc[0]
        if equity.nunique() <= MIN_DISTINCT_EQUITY_VALUES:
            raise RuntimeError(f"Curva diária sem variação suficiente para {model}.")
        curves[f"equity_{model}"] = equity.to_numpy()
        summary.append(
            {
                "model": model,
                "strategy": strategy.name,
                "cagr": annualized_growth(equity),
                "sharpe": sharpe_ratio(ret),
            }
        )
    return curves, pd.DataFrame(summary)


def quantile_thresholds(
    proba: pd.Series, event_q: float, threshold_mode: str, threshold_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Limiares diários de entrada (quantil `event_q`) e de saída (mediana).

    `"full_sample"` usa os quantis do período inteiro (artigo); `"rolling"`, os dos
    `threshold_window` pregões anteriores, sem o dia corrente (antes disso, sem limiar).
    """
    if threshold_mode == "full_sample":
        n_days = len(proba)
        return np.full(n_days, proba.quantile(event_q)), np.full(n_days, proba.quantile(0.50))
    if threshold_mode == "rolling":
        past = proba.shift(1).rolling(threshold_window, min_periods=threshold_window)
        return past.quantile(event_q).to_numpy(), past.quantile(0.50).to_numpy()
    raise ValueError(
        f"threshold_mode inválido: {threshold_mode!r} (use 'full_sample' ou 'rolling')"
    )


def run_quantile_strategy(
    oof_model: pd.DataFrame,
    event_q: float,
    lag: int,
    threshold_mode: str = "full_sample",
    threshold_window: int = 60,
    strategy: Strategy = ARTICLE_STRATEGY,
) -> pd.DataFrame:
    """Regra de quantil das Tabelas 3 e 4 sobre a série diária de p de um modelo."""
    frame = oof_model.sort_values("day").copy()
    long_th, exit_th = quantile_thresholds(
        frame["proba"], event_q, threshold_mode, threshold_window
    )
    proba = frame["proba"].to_numpy()
    # comparações com NaN (sem limiar) são falsas: a posição anterior é mantida
    positions, turnovers = _hold_positions(proba >= long_th, proba <= exit_th)
    frame["signal"] = positions
    frame["turnover"] = turnovers
    frame["cost"] = frame["turnover"] * strategy.cost
    effective = frame["signal"].shift(lag + 1, fill_value=0)
    frame["strategy_ret"] = effective * frame["ret_next"].fillna(0) - frame["cost"]
    frame["equity"] = (1 + frame["strategy_ret"]).cumprod()
    return frame


def annualized_growth(equity: pd.Series) -> float:
    """CAGR com 252 pregões por ano."""
    if len(equity) <= 1:
        return np.nan
    return float(equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(equity)) - 1)


def sharpe_ratio(returns: pd.Series) -> float:
    """Sharpe anualizado sem taxa livre de risco (desvio-padrão populacional)."""
    deviation = returns.std(ddof=0)
    if deviation == 0:
        return np.nan
    return float(returns.mean() / deviation * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """Maior queda relativa a partir de um pico."""
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1).min())


def performance_metrics(
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    cost: pd.Series,
    signal: pd.Series,
) -> dict[str, float]:
    """Métricas das Tabelas 3 e 4."""
    returns = returns.fillna(0)
    deviation = returns.std(ddof=0)
    return {
        "cagr": annualized_growth(equity.ffill()),
        "sharpe": sharpe_ratio(returns),
        "vol_anual": float(deviation * np.sqrt(TRADING_DAYS_PER_YEAR))
        if deviation != 0
        else np.nan,
        "max_drawdown": max_drawdown(equity.ffill()),
        "turnover": float(turnover.sum()),
        "n_trades": int((signal.diff().fillna(0) != 0).sum()),
        "hit_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "exposure": float((signal != 0).mean()) if len(signal) else np.nan,
        "total_cost": float(cost.sum()),
    }


def strategy_metrics(run: pd.DataFrame) -> dict[str, float]:
    """Métricas de uma execução de `run_quantile_strategy`."""
    return performance_metrics(
        run["strategy_ret"], run["equity"], run["turnover"], run["cost"], run["signal"]
    )


def buy_and_hold_metrics(returns: pd.Series, normalize_to_first: bool = False) -> dict[str, float]:
    """Métricas do Ibovespa sempre comprado e sem custos.

    `normalize_to_first` repete a Tabela 3 do artigo, que divide a curva pelo 1º valor
    (o CAGR então ignora o retorno do 1º dia; o Sharpe o inclui).
    """
    equity = (1 + returns).cumprod()
    if normalize_to_first:
        equity = equity / equity.iloc[0]
    zeros = pd.Series([0.0])
    return performance_metrics(
        returns.reset_index(drop=True),
        equity.reset_index(drop=True),
        zeros,
        zeros,
        pd.Series(np.ones(len(returns))),
    )


def robustness_grid(
    oof: pd.DataFrame,
    lags: tuple[int, ...] = (0, 1, 2),
    quantiles: tuple[float, ...] = (0.90, 0.95),
    threshold_mode: str = "full_sample",
    threshold_window: int = 60,
) -> pd.DataFrame:
    """Tabela 4: regra de quantil para cada lag × quantil × modelo (lag 0 e q 0,90 = Tabela 3)."""
    rows = []
    for lag in lags:
        for event_q in quantiles:
            for model in MODEL_LABELS:
                run = run_quantile_strategy(
                    oof[oof["model"] == model], event_q, lag, threshold_mode, threshold_window
                )
                rows.append(
                    {
                        "model": model,
                        "strategy": ARTICLE_STRATEGY.name,
                        "lag": lag,
                        "event_q": event_q,
                        **strategy_metrics(run),
                    }
                )
    return pd.DataFrame(rows)
