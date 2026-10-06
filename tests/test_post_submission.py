"""
Testes das opções usadas nas verificações pós-submissão (não constam do artigo).

Os testes unitários usam dados sintéticos e rodam sempre. O último refaz o walk-forward
só-texto a partir da matriz TF-IDF e confere a Tabela 2; sem os dados, é pulado.
O padrão de cada opção nova reproduz o artigo, o que é coberto por
`tests/test_article_reproduction.py`.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.analysis import h1_tests
from src.features.technical import technical_features
from src.models import walk_forward

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def export() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_tcc_figures", REPO_ROOT / "scripts" / "export_tcc_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- (c) backtest


def test_rolling_thresholds_use_only_previous_days(export: ModuleType) -> None:
    proba = pd.Series(np.random.default_rng(0).uniform(size=50))
    long_th, exit_th = export._quantile_thresholds(proba, 0.90, False, "rolling", 10)

    assert np.isnan(long_th[:10]).all() and np.isnan(exit_th[:10]).all()
    for day in range(10, 50):
        past = proba.iloc[day - 10 : day]
        assert long_th[day] == pytest.approx(past.quantile(0.90))
        assert exit_th[day] == pytest.approx(past.quantile(0.50))

    # Alterar p de hoje e dos dias seguintes não muda o limiar de hoje.
    changed = proba.copy()
    changed.iloc[30:] = 0.99
    long_changed, _ = export._quantile_thresholds(changed, 0.90, False, "rolling", 10)
    np.testing.assert_array_equal(long_changed[:31], long_th[:31])


def test_full_sample_thresholds_follow_article_rule(export: ModuleType) -> None:
    proba = pd.Series(np.linspace(0.3, 0.7, 41))
    long_th, exit_th = export._quantile_thresholds(proba, 0.95, True, "full_sample", 60)
    assert (long_th == proba.quantile(0.95)).all()
    assert (exit_th == proba.quantile(0.05)).all()


def test_invalid_threshold_mode_is_rejected(export: ModuleType) -> None:
    with pytest.raises(ValueError):
        export._quantile_thresholds(pd.Series([0.5, 0.6]), 0.9, False, "futuro", 60)


def test_rolling_strategy_stays_out_without_history(export: ModuleType) -> None:
    days = pd.bdate_range("2020-01-01", periods=80)
    oof = pd.DataFrame({"day": days, "proba": np.linspace(0.4, 0.6, 80), "ret_next": 0.01})
    cfg = {"allow_short": False, "cost": 0.0005}
    strategy = export._run_strategy_quantile(oof, cfg, 0.90, 0, threshold_mode="rolling")
    assert (strategy["signal"].iloc[:60] == 0).all()


# --------------------------------------------------------------------------- (b) H1


def test_moving_block_bootstrap_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = 0.3 * x + rng.normal(size=300)
    first = h1_tests.moving_block_bootstrap_ci(x, y, h1_tests.pearson, 10, n_boot=200)
    second = h1_tests.moving_block_bootstrap_ci(x, y, h1_tests.pearson, 10, n_boot=200)
    other_seed = h1_tests.moving_block_bootstrap_ci(x, y, h1_tests.pearson, 10, 200, seed=7)
    assert first == second
    assert first != other_seed
    assert first[0] < h1_tests.pearson(x, y) < first[1]


def test_block_size_and_newey_west_lag_rules() -> None:
    assert h1_tests.cube_root_block_size(1341) == 11
    assert h1_tests.cube_root_block_size(1552) == 12
    assert h1_tests.newey_west_default_lags(1341) == 7
    assert h1_tests.newey_west_default_lags(1552) == 7
    assert h1_tests.newey_west_default_lags(100) == 4


def test_newey_west_recovers_known_slope() -> None:
    rng = np.random.default_rng(2)
    sentiment = rng.normal(size=2000)
    returns = 0.01 + 0.5 * sentiment + rng.normal(scale=0.1, size=2000)
    result = h1_tests.newey_west_regression(returns, sentiment, maxlags=7)
    assert result["beta"] == pytest.approx(0.5, abs=0.01)
    assert result["beta_p_hac"] < 1e-6


def test_demean_by_group_removes_level_shifts() -> None:
    values = np.array([1.0, 3.0, 10.0, 12.0])
    groups = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(h1_tests.demean_by_group(values, groups), [-1, 1, -1, 1])


def test_fold_ids_match_article_walk_forward() -> None:
    ids = walk_forward.fold_ids(1942)
    assert (ids[:390] == -1).all()
    assert [int((ids == fold).sum()) for fold in range(4)] == [388, 388, 388, 388]


# --------------------------------------------------------------------------- (d) eventos


def _business_day_returns(value: float | None = 0.0, periods: int = 120) -> pd.Series:
    """Retornos em dias úteis (sem fins de semana): constantes ou aleatórios (value=None)."""
    days = pd.bdate_range("2021-01-04", periods=periods)
    rng = np.random.default_rng(3)
    returns = rng.normal(scale=0.01, size=periods) if value is None else np.full(periods, value)
    return pd.Series(returns, index=days)


def _friday_events(returns: pd.Series) -> pd.DataFrame:
    fridays = [day for day in returns.index[70:110] if day.weekday() == 4]
    polarities = ["pos" if i % 2 else "neg" for i in range(len(fridays))]
    return pd.DataFrame({"event_day": fridays, "polarity": polarities})


def test_trading_day_window_keeps_every_event(export: ModuleType) -> None:
    returns = _business_day_returns(None)
    friday = returns.index[74]
    assert friday.weekday() == 4

    # Artigo (dias corridos): o fim de semana encurta a janela e o evento de sexta só tem τ=0.
    calendar = export._event_cars(returns, friday, 4, "calendar_days", False, 60, 0)
    assert list(calendar) == [0]
    # Verificação (pregões): o mesmo evento entra em todo τ.
    trading = export._event_cars(returns, friday, 4, "trading_days", False, 60, 0)
    assert list(trading) == [0, 1, 2, 3, 4]
    assert trading[4] == pytest.approx(returns.iloc[74:79].sum())

    events = _friday_events(returns)
    caar = export.compute_caar_by_event_time(
        events, returns, tau_max=4, window_unit="trading_days", n_boot=50
    )
    assert ((caar["n_events_neg"] + caar["n_events_pos"]) == len(events)).all()


def test_abnormal_returns_are_zero_when_returns_are_constant(export: ModuleType) -> None:
    returns = _business_day_returns(0.002)
    events = _friday_events(returns)
    caar = export.compute_caar_by_event_time(
        events, returns, tau_max=4, window_unit="trading_days", abnormal=True, n_boot=50
    )
    for column in ("caar_neg_mean", "caar_pos_mean", "caar_neg_ci_low", "caar_pos_ci_high"):
        np.testing.assert_allclose(caar[column], 0.0, atol=1e-15)


def test_first_tau_excludes_event_day(export: ModuleType) -> None:
    returns = _business_day_returns(0.0)
    event_day = returns.index[80]
    returns.loc[event_day] = 0.05  # só o dia do evento tem retorno
    cars = export._event_cars(returns, event_day, 3, "trading_days", False, 60, first_tau=1)
    assert cars == {1: 0.0, 2: 0.0, 3: 0.0}
    cars_with_day0 = export._event_cars(returns, event_day, 3, "trading_days", False, 60, 0)
    assert cars_with_day0[0] == pytest.approx(0.05)


# --------------------------------------------------------------------------- (e) modelos


def test_technical_features_do_not_use_future_prices() -> None:
    close = pd.Series(np.linspace(100, 130, 40), index=pd.bdate_range("2022-01-03", periods=40))
    features = technical_features(close)
    shocked = close.copy()
    shocked.iloc[25:] *= 1.5
    features_shocked = technical_features(shocked)
    pd.testing.assert_frame_equal(features.iloc[:25], features_shocked.iloc[:25])
    assert list(features.columns) == [
        "ret_lag_0",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_4",
        "vol_5",
        "vol_20",
    ]


def test_text_only_walk_forward_reproduces_table2() -> None:
    base = os.environ.get("TCC_USP_BASE")
    data_dir = Path(base) / "data_processed" if base else None
    needed = ("tfidf_daily_matrix.npz", "tfidf_daily_index.csv", "labels_y_daily.csv")
    if data_dir is None or not all((data_dir / name).exists() for name in needed):
        pytest.skip("Dados do artigo indisponíveis (TCC_USP_BASE).")
    from scipy.sparse import load_npz

    index = pd.read_csv(data_dir / "tfidf_daily_index.csv", parse_dates=["day"])
    labels = pd.read_csv(data_dir / "labels_y_daily.csv", parse_dates=["day"])
    base_rows = index.merge(labels[["day", "y"]], how="left", on="day")
    labelled = base_rows["y"].notna().to_numpy()
    features = load_npz(data_dir / "tfidf_daily_matrix.npz").tocsr()[labelled]
    y = base_rows.loc[labelled, "y"].astype(int).to_numpy()

    predictions = walk_forward.walk_forward_predictions(features, y, walk_forward.baseline_models())
    summary = walk_forward.summarize_predictions(y, predictions).set_index("model")

    assert summary.loc["logreg_l2", "auc"] == pytest.approx(0.5015378221113882, abs=1e-12)
    assert summary.loc["logreg_l2", "mda"] == pytest.approx(0.5025773195876289, abs=1e-12)
    assert summary.loc["rf_200", "auc"] == pytest.approx(0.4912635078969243, abs=1e-12)
    assert summary.loc["rf_200", "mda"] == pytest.approx(0.5115979381443299, abs=1e-12)
    assert summary.loc["logreg_l2", "auc_low"] == pytest.approx(0.4739878216337317, abs=1e-12)
    assert (summary["n_obs"] == 1552).all()
