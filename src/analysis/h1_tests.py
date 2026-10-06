"""
Testes estatísticos da hipótese H1 (sentimento × retorno) para a verificação pós-submissão (b).

- correlações de Pearson e Spearman;
- IC 95% por bootstrap em blocos móveis (moving block bootstrap), que preserva a
  dependência serial dentro de cada bloco;
- regressão r_t = α + β·s_t + ε_t com erros-padrão HAC (Newey-West);
- testes de estacionariedade ADF e KPSS;
- remoção da média por grupo, para isolar mudanças de nível entre blocos do walk-forward.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

RANDOM_SEED = 42
N_BOOTSTRAP = 2000


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.pearsonr(x, y)[0])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y)[0])


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson e Spearman com p-valores assintóticos (supõem observações independentes)."""
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n": len(x),
    }


def cube_root_block_size(n: int) -> int:
    """Regra n^(1/3) para o tamanho do bloco."""
    return max(1, round(n ** (1 / 3)))


def moving_block_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    block_size: int,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """IC percentil da estatística por bootstrap em blocos móveis de pares (x_t, y_t).

    Cada réplica concatena blocos contíguos de `block_size` dias, com início sorteado,
    até completar n observações.
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts_max = n - block_size + 1
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel()[:n]
        estimates[b] = statistic(x[idx], y[idx])
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def newey_west_default_lags(n: int) -> int:
    """Regra usual de Newey-West (1994): floor(4·(n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_regression(
    returns: np.ndarray, sentiment: np.ndarray, maxlags: int
) -> dict[str, float]:
    """OLS de r_t = α + β·s_t + ε_t com erros-padrão HAC (Newey-West, núcleo de Bartlett)."""
    model = sm.OLS(returns, sm.add_constant(sentiment)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}
    )
    return {
        "alpha": float(model.params[0]),
        "beta": float(model.params[1]),
        "beta_se_hac": float(model.bse[1]),
        "beta_t_hac": float(model.tvalues[1]),
        "beta_p_hac": float(model.pvalues[1]),
        "r2": float(model.rsquared),
        "maxlags": maxlags,
        "n": int(model.nobs),
    }


def demean_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtrai de cada valor a média do seu grupo (ex.: bloco do walk-forward)."""
    group_means = pd.Series(values).groupby(groups).transform("mean").to_numpy()
    return values - group_means


def stationarity_tests(series: np.ndarray) -> dict[str, float]:
    """ADF (H0: raiz unitária) e KPSS (H0: estacionária), ambos com constante.

    O p-valor do KPSS vem de tabela e é truncado em [0,01; 0,10].
    """
    adf_stat, adf_p, adf_lags, _, _, _ = adfuller(series, autolag="AIC", result_object=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_p, kpss_lags, _ = kpss(
            series, regression="c", nlags="auto", result_object=False
        )
    return {
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "adf_lags": int(adf_lags),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "kpss_lags": int(kpss_lags),
    }
