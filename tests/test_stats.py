"""Bootstraps, correlações, Newey-West e estacionariedade (tcc.stats)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcc import stats


def test_moving_block_bootstrap_is_deterministic_and_covers_the_estimate() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = 0.3 * x + rng.normal(size=300)
    first = stats.moving_block_bootstrap_ci(x, y, stats.pearson, 10, n_boot=200)
    second = stats.moving_block_bootstrap_ci(x, y, stats.pearson, 10, n_boot=200)
    other_seed = stats.moving_block_bootstrap_ci(x, y, stats.pearson, 10, 200, seed=7)
    assert first == second
    assert first != other_seed
    assert first[0] < stats.pearson(x, y) < first[1]


def test_rolling_correlation_uses_only_the_window() -> None:
    rng = np.random.default_rng(2)
    x, y = pd.Series(rng.normal(size=40)), pd.Series(rng.normal(size=40))
    rolling = stats.rolling_correlation(x, y, 10)
    assert rolling.iloc[:9].isna().all()
    assert rolling.iloc[25] == pytest.approx(
        stats.pearson(x.iloc[16:26].to_numpy(), y.iloc[16:26].to_numpy())
    )


def test_block_size_and_newey_west_lag_rules() -> None:
    assert stats.cube_root_block_size(1341) == 11
    assert stats.cube_root_block_size(1552) == 12
    assert stats.newey_west_default_lags(1341) == 7
    assert stats.newey_west_default_lags(100) == 4


def test_newey_west_recovers_known_slope() -> None:
    rng = np.random.default_rng(2)
    sentiment = rng.normal(size=2000)
    returns = 0.01 + 0.5 * sentiment + rng.normal(scale=0.1, size=2000)
    result = stats.newey_west_regression(returns, sentiment, maxlags=7)
    assert result["beta"] == pytest.approx(0.5, abs=0.01)
    assert result["beta_p_hac"] < 1e-6


def test_demean_by_group_removes_level_shifts() -> None:
    values = np.array([1.0, 3.0, 10.0, 12.0])
    np.testing.assert_array_equal(
        stats.demean_by_group(values, np.array([0, 0, 1, 1])), [-1, 1, -1, 1]
    )


def test_stationarity_flags_random_walk_but_not_noise() -> None:
    rng = np.random.default_rng(3)
    noise = rng.normal(size=500)
    assert stats.stationarity_tests(noise)["adf_p"] < 0.01
    assert stats.stationarity_tests(np.cumsum(noise))["adf_p"] > 0.05


def test_bootstrap_mean_ci_is_seeded_and_brackets_the_mean() -> None:
    values = np.random.default_rng(4).normal(loc=1.0, size=100)
    mean, low, high = stats.bootstrap_mean_ci(values)
    assert (mean, low, high) == stats.bootstrap_mean_ci(values)
    assert low < values.mean() < high
    assert np.isnan(stats.bootstrap_mean_ci(np.array([]))[0])


def test_iid_bootstrap_ci_skips_single_class_samples() -> None:
    y = np.array([0, 1] * 20)
    scores = np.linspace(0, 1, 40)
    low, high = stats.iid_bootstrap_ci(
        y, scores, lambda t, s: float(np.mean(t == (s > 0.5))), n_boot=50
    )
    assert 0 <= low <= high <= 1
