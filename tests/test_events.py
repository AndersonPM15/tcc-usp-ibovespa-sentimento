"""Detecção de eventos, CAR e CAAR (tcc.events)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcc import events


def _business_day_returns(value: float | None = 0.0, periods: int = 120) -> pd.Series:
    """Retornos em dias úteis (sem fins de semana): constantes ou aleatórios (value=None)."""
    days = pd.bdate_range("2021-01-04", periods=periods)
    rng = np.random.default_rng(3)
    data = rng.normal(scale=0.01, size=periods) if value is None else np.full(periods, value)
    return pd.Series(data, index=days)


def _friday_events(returns: pd.Series) -> pd.DataFrame:
    fridays = [day for day in returns.index[70:110] if day.weekday() == 4]
    polarities = ["pos" if i % 2 else "neg" for i in range(len(fridays))]
    return pd.DataFrame({"event_day": fridays, "polarity": polarities})


def test_trading_day_window_keeps_every_event() -> None:
    returns = _business_day_returns(None)
    friday = returns.index[74]
    assert friday.weekday() == 4
    # artigo (dias corridos): o fim de semana encurta a janela e a sexta só tem τ = 0
    assert list(events.event_cars(returns, friday, 4)) == [0]
    trading = events.event_cars(returns, friday, 4, window_unit="trading_days")
    assert list(trading) == [0, 1, 2, 3, 4]
    assert trading[4] == pytest.approx(returns.iloc[74:79].sum())
    caar = events.caar_by_event_time(
        _friday_events(returns), returns, tau_max=4, window_unit="trading_days", n_boot=50
    )
    assert ((caar["n_events_neg"] + caar["n_events_pos"]) == len(_friday_events(returns))).all()


def test_abnormal_returns_are_zero_when_returns_are_constant() -> None:
    returns = _business_day_returns(0.002)
    caar = events.caar_by_event_time(
        _friday_events(returns), returns, 4, window_unit="trading_days", abnormal=True, n_boot=50
    )
    for column in ("caar_neg_mean", "caar_pos_mean", "caar_neg_ci_low", "caar_pos_ci_high"):
        np.testing.assert_allclose(caar[column], 0.0, atol=1e-15)


def test_first_tau_excludes_event_day() -> None:
    returns = _business_day_returns(0.0)
    event_day = returns.index[80]
    returns.loc[event_day] = 0.05
    assert events.event_cars(returns, event_day, 3, "trading_days", first_tau=1) == {
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }
    assert events.event_cars(returns, event_day, 3, "trading_days")[0] == pytest.approx(0.05)


def test_invalid_window_unit_is_rejected() -> None:
    returns = _business_day_returns()
    with pytest.raises(ValueError, match="window_unit"):
        events.event_cars(returns, returns.index[80], 2, "semanas")


def test_detect_events_marks_extreme_deciles_and_sums_raw_returns() -> None:
    days = pd.bdate_range("2024-01-01", periods=40)
    proba = np.linspace(0.3, 0.7, 40)
    oof = pd.DataFrame({"day": np.tile(days, 2), "proba": np.tile(proba, 2)})
    returns = pd.DataFrame({"day": days, "ret": 0.01})
    detected = events.detect_events(oof, returns)
    assert set(detected["polarity"]) == {"pos", "neg"}
    assert (detected["n_news"] == 2).all()
    first = detected.iloc[0]
    window = returns[(returns["day"] >= first["event_day"])]
    window = window[window["day"] <= first["event_day"] + pd.Timedelta(days=5)]
    assert first["car_value"] == pytest.approx(window["ret"].sum())
