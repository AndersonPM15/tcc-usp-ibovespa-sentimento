"""Ibovespa, retornos, rótulos e variáveis técnicas (tcc.market)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcc import datasets, market
from tcc.config import Settings


def _yfinance_like(closes: list[float], multiindex: bool) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=len(closes), name="Date")
    fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    data = dict.fromkeys(fields, closes)
    frame = pd.DataFrame(data, index=days)
    if multiindex:
        frame.columns = pd.MultiIndex.from_product([fields, ["^BVSP"]], names=["Price", "Ticker"])
    return frame


@pytest.mark.parametrize("multiindex", [True, False])
def test_prepare_ibovespa_builds_clean_columns(multiindex: bool) -> None:
    prepared = market.prepare_ibovespa(_yfinance_like([100.0, 110.0, 99.0], multiindex))
    assert list(prepared.columns) == market.CLEAN_COLUMNS
    assert np.isnan(prepared["return"].iloc[0])
    np.testing.assert_allclose(prepared["return"].iloc[1:], [0.10, -0.10])
    assert prepared["direction"].tolist() == [0, 1, 0]


def test_build_labels_uses_next_trading_day_and_skips_days_without_prices() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),  # sexta, seg, ter
            "close": [100.0, 102.0, 101.0],
        }
    )
    index = pd.DataFrame(
        {
            "day": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-08", "2024-01-09"]),
            "row_id": [0, 1, 2, 3],
        }
    )
    labels = market.build_labels(index, prices)
    assert labels["y"].tolist()[0] == 1  # sexta → segunda: +2%
    assert np.isnan(labels["y"].iloc[1])  # sábado: sem pregão
    assert labels["y"].iloc[2] == 0  # segunda → terça: queda
    assert np.isnan(labels["y"].iloc[3])  # último pregão: sem retorno seguinte
    assert labels["ret_next"].iloc[0] == pytest.approx(0.02)


def test_daily_returns_are_close_to_close() -> None:
    prices = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-03", "2024-01-02"]), "close": [110.0, 100.0]}
    )
    returns = market.daily_returns(prices)
    assert returns["day"].is_monotonic_increasing
    assert returns["ret"].iloc[1] == pytest.approx(0.10)


def test_technical_features_do_not_use_future_prices() -> None:
    close = pd.Series(np.linspace(100, 130, 40), index=pd.bdate_range("2022-01-03", periods=40))
    shocked = close.copy()
    shocked.iloc[25:] *= 1.5
    pd.testing.assert_frame_equal(
        market.technical_features(close).iloc[:25], market.technical_features(shocked).iloc[:25]
    )
    assert list(market.technical_features(close).columns) == [
        "ret_lag_0",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_4",
        "vol_5",
        "vol_20",
    ]


def test_article_ibovespa_period(settings: Settings) -> None:
    ibovespa = datasets.read_ibovespa(settings)
    assert len(ibovespa) == 1960
    assert ibovespa["date"].min() == pd.Timestamp("2018-01-02")
    assert ibovespa["date"].max() == pd.Timestamp("2025-11-18")
    assert len(datasets.clamp_period(ibovespa, "date")) == 1737  # Tabela 1


def test_labels_match_the_article_file(settings: Settings) -> None:
    reference_path = settings.processed_dir / "labels_y_daily.csv"
    if not reference_path.exists():
        pytest.skip("labels_y_daily.csv do artigo indisponível")
    _, index = datasets.read_tfidf(settings)
    labels = market.build_labels(index, datasets.read_ibovespa(settings))
    reference = pd.read_csv(reference_path, parse_dates=["day"])
    pd.testing.assert_frame_equal(labels, reference, check_exact=False, rtol=0, atol=1e-15)
    assert labels["y"].notna().sum() == 1942


def test_build_labels_never_invents_labels_without_prices() -> None:
    # Regressão: sem interseção com o Ibovespa, o notebook 15 gerava rótulos falsos alternados
    # (0, 1, 0, 1…) e seguia em frente; agora os dias ficam sem rótulo.
    index = pd.DataFrame({"day": pd.date_range("2030-01-01", periods=4), "row_id": range(4)})
    prices = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "close": [1.0, 2.0]}
    )
    assert market.build_labels(index, prices)["y"].isna().all()
