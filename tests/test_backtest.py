"""Regras de negociação, métricas e buy-and-hold (tcc.backtest)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcc import backtest


def _oof(proba: list[float], ret_next: float = 0.01) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=len(proba))
    return pd.DataFrame({"day": days, "proba": proba, "ret_next": ret_next, "model": "logreg_l2"})


def test_threshold_rule_enters_exits_and_holds() -> None:
    run = backtest.run_threshold_strategy(_oof([0.5, 0.65, 0.5, 0.35, 0.5]), backtest.Strategy())
    assert run["signal"].tolist() == [0, 1, 1, 0, 0]
    assert run["turnover"].tolist() == [0, 1, 0, 1, 0]
    # a posição de D vale para o retorno da linha seguinte; o custo sai no dia da troca
    assert run["strategy_ret"].tolist() == pytest.approx([0, -0.0005, 0.01, 0.01 - 0.0005, 0])


def test_rolling_thresholds_use_only_previous_days() -> None:
    proba = pd.Series(np.random.default_rng(0).uniform(size=50))
    long_th, exit_th = backtest.quantile_thresholds(proba, 0.90, "rolling", 10)
    assert np.isnan(long_th[:10]).all() and np.isnan(exit_th[:10]).all()
    for day in range(10, 50):
        past = proba.iloc[day - 10 : day]
        assert long_th[day] == pytest.approx(past.quantile(0.90))
        assert exit_th[day] == pytest.approx(past.quantile(0.50))
    changed = proba.copy()
    changed.iloc[30:] = 0.99  # mudar o dia corrente e os seguintes não altera o limiar de hoje
    np.testing.assert_array_equal(
        backtest.quantile_thresholds(changed, 0.90, "rolling", 10)[0][:31], long_th[:31]
    )


def test_full_sample_thresholds_follow_the_article() -> None:
    proba = pd.Series(np.linspace(0.3, 0.7, 41))
    long_th, exit_th = backtest.quantile_thresholds(proba, 0.95, "full_sample", 60)
    assert (long_th == proba.quantile(0.95)).all()
    assert (exit_th == proba.quantile(0.50)).all()


def test_invalid_threshold_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold_mode"):
        backtest.quantile_thresholds(pd.Series([0.5, 0.6]), 0.9, "futuro", 60)


def test_rolling_strategy_stays_out_without_history() -> None:
    run = backtest.run_quantile_strategy(
        _oof(list(np.linspace(0.4, 0.6, 80))), 0.90, 0, threshold_mode="rolling"
    )
    assert (run["signal"].iloc[:60] == 0).all()


def test_lag_delays_the_position() -> None:
    oof = _oof([0.9, 0.1, 0.1, 0.1, 0.1])
    lag0 = backtest.run_quantile_strategy(oof, 0.80, lag=0)
    lag2 = backtest.run_quantile_strategy(oof, 0.80, lag=2)
    assert lag0["signal"].shift(1, fill_value=0).tolist() == [0, 1, 0, 0, 0]
    assert lag2["strategy_ret"].iloc[3] == pytest.approx(0.01)


def test_metrics() -> None:
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert backtest.max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1)
    assert backtest.annualized_growth(pd.Series([1.0, 1.0])) == 0
    assert np.isnan(backtest.sharpe_ratio(pd.Series([0.01, 0.01])))


def test_buy_and_hold_normalization_drops_first_return_only_from_cagr() -> None:
    returns = pd.Series([0.10, 0.0, 0.0])
    raw = backtest.buy_and_hold_metrics(returns)
    normalized = backtest.buy_and_hold_metrics(returns, normalize_to_first=True)
    assert raw["cagr"] > normalized["cagr"] == pytest.approx(0)
    assert raw["sharpe"] == normalized["sharpe"]
    assert raw["exposure"] == 1.0


def test_robustness_grid_covers_lags_quantiles_and_models() -> None:
    proba = list(np.random.default_rng(1).uniform(size=300))
    oof = pd.concat([_oof(proba).assign(model=model) for model in ("logreg_l2", "rf_200")])
    grid = backtest.robustness_grid(oof)
    assert len(grid) == 3 * 2 * 2
    assert grid[["lag", "event_q", "model"]].drop_duplicates().shape[0] == 12
