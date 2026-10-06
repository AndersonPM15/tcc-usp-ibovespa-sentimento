"""Verificações pós-submissão (a)–(e), gravadas em `reports/verificacao/`.

Não constam do artigo. Usam as mesmas previsões fora da amostra e as opções novas do
mesmo código (padrão = artigo):

a) benchmark buy-and-hold no mesmo período e convenção das estratégias;
b) H1: correlações com IC por bootstrap em blocos, Newey-West e ADF/KPSS;
c) backtest sem look-ahead (limiares nos 60 pregões anteriores);
d) estudo de eventos com retorno anormal e janela em pregões;
e) modelo técnico e modelo texto + técnico, comparados ao só-texto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tcc import backtest, events, market, models, stats
from tcc.config import ARTICLE_END, MODEL_LABELS, RANDOM_SEED

LABEL = "pós-submissão; não consta do artigo"
BLOCK_SIZES_EXTRA = (5, 10, 20)
NEWEY_WEST_EXTRA_LAGS = (5, 10)
RETURN_TIMINGS = {"mesmo dia (D−1→D)": "ret_same_day", "dia seguinte (D→D+1)": "ret_next_day"}
DEMEANED = "menos a média do seu bloco do walk-forward"
TEXT_ONLY = "só texto (TF-IDF)"
TECHNICAL_ONLY = "técnico (5 retornos defasados + volatilidade de 5 e 20 dias)"
TEXT_AND_TECHNICAL = "texto + técnico"
SAME_PERIOD_BENCHMARK = "Ibovespa buy-and-hold — mesmas linhas e retorno das estratégias (D→D+1)"


@dataclass(frozen=True)
class VerificationInputs:
    """Tudo o que as verificações usam, já calculado pela reprodução."""

    oof_full: pd.DataFrame  # p diário fora da amostra até 17/11/2025 (com o bloco)
    oof_article: pd.DataFrame  # o mesmo, recortado em 2024
    returns_full: pd.DataFrame  # Ibovespa até 18/11/2025 (`day`, `close`, `ret`)
    returns_article: pd.DataFrame  # o mesmo, recortado em 2018–2024
    events: pd.DataFrame  # eventos do artigo
    labelled: pd.DataFrame  # dias com rótulo, na ordem da matriz
    text_features: Any  # linhas da matriz TF-IDF dos dias com rótulo
    text_predictions: dict[str, np.ndarray]


def run_verifications(inputs: VerificationInputs) -> dict[str, pd.DataFrame]:
    """Executa (a)–(e) e devolve as tabelas pelo nome do arquivo (sem extensão)."""
    tables = {"a_benchmark_mesmo_periodo": check_benchmark(inputs)}
    tables.update(check_h1(_sentiment_table(inputs.oof_full, inputs.returns_full)))
    tables["c_backtest_sem_lookahead"] = check_backtest(
        inputs.oof_article, tables["a_benchmark_mesmo_periodo"]
    )
    tables["d_eventos_retorno_anormal"] = check_event_study(inputs.events, inputs.returns_full)
    tables.update(check_models(inputs))
    return {name: table.assign(rotulo=LABEL) for name, table in tables.items()}


# --------------------------------------------------------------------------- (a)


def check_benchmark(inputs: VerificationInputs) -> pd.DataFrame:
    """(a) Ibovespa buy-and-hold no período da Tabela 3 e no mesmo período das estratégias.

    As estratégias têm 1.341 linhas: os 1.346 pregões de 05/08/2019 a 30/12/2024, menos 5
    dias sem notícias. Por isso há três versões do benchmark no mesmo período: mesmas linhas
    e retorno das estratégias (comparação direta com a Tabela 3), todos os pregões
    (desempenho real do índice) e a convenção da Figura 8 (retorno D−1→D nas datas das
    estratégias, que pula outros 5 retornos).
    """
    market_returns = inputs.returns_article.set_index("day")["ret"]
    oof = inputs.oof_article
    strategy_days = oof[oof["model"] == "logreg_l2"].sort_values("day")
    first_day, last_day = strategy_days["day"].min(), strategy_days["day"].max()
    same_rows = strategy_days["ret_next"].fillna(0).reset_index(drop=True)
    all_days = market_returns.loc[first_day:last_day].copy()
    all_days.iloc[0] = 0.0  # a curva começa em 1 no fechamento de 05/08/2019
    figure8 = market_returns.reindex(strategy_days["day"]).reset_index(drop=True)
    figure8.iloc[0] = 0.0
    rows = [
        _buy_and_hold_row(
            "Ibovespa buy-and-hold — período da Tabela 3 do artigo",
            market_returns.dropna(),
            normalize_to_first=True,
        ),
        _buy_and_hold_row(SAME_PERIOD_BENCHMARK, same_rows, strategy_days["day"]),
        _buy_and_hold_row(
            "Ibovespa buy-and-hold — todos os pregões de 05/08/2019 a 30/12/2024", all_days
        ),
        _buy_and_hold_row(
            "Ibovespa buy-and-hold — convenção da Figura 8 (D−1→D nas datas das estratégias)",
            figure8,
            strategy_days["day"],
        ),
    ]
    for model, name in MODEL_LABELS.items():
        run = backtest.run_quantile_strategy(oof[oof["model"] == model], event_q=0.90, lag=0)
        rows.append(
            {
                "cenario": f"{name} — Tabela 3 do artigo (lag 0, quantil 0,90)",
                "inicio": run["day"].min().date(),
                "fim": run["day"].max().date(),
                "n": len(run),
                **backtest.strategy_metrics(run),
            }
        )
    return pd.DataFrame(rows)


def _buy_and_hold_row(
    name: str, returns: pd.Series, days: pd.Series | None = None, normalize_to_first: bool = False
) -> dict[str, object]:
    days = pd.Series(returns.index) if days is None else days
    return {
        "cenario": name,
        "inicio": pd.Timestamp(days.min()).date(),
        "fim": pd.Timestamp(days.max()).date(),
        "n": len(returns),
        **backtest.buy_and_hold_metrics(returns, normalize_to_first),
    }


# --------------------------------------------------------------------------- (b)


def _sentiment_table(oof: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Sentimento 2p − 1 por modelo (colunas), retornos de D−1→D e D→D+1 e bloco do dia."""
    wide = oof.pivot(index="day", columns="model", values="proba").mul(2).sub(1)
    per_day = oof.drop_duplicates("day").set_index("day")
    per_day = per_day[["ret_next", "fold"]].rename(columns={"ret_next": "ret_next_day"})
    same_day = returns.set_index("day")["ret"].rename("ret_same_day")
    return wide.join(per_day).join(same_day).reset_index()


def check_h1(sentiment: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(b) Correlações, IC por bootstrap em blocos, Newey-West e estacionariedade."""
    correlations: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    newey_west: list[dict[str, Any]] = []
    stationarity: list[dict[str, Any]] = []
    periods = {
        "artigo (05/08/2019–30/12/2024)": sentiment[sentiment["day"] <= ARTICLE_END],
        "completo (05/08/2019–17/11/2025)": sentiment,
    }
    for period, frame in periods.items():
        stationarity.extend(_stationarity_rows(period, frame))
        for model, name in MODEL_LABELS.items():
            for timing, column in RETURN_TIMINGS.items():
                pair = frame[[model, column, "fold"]].dropna()
                x, y = pair[model].to_numpy(), pair[column].to_numpy()
                keys = {"periodo": period, "modelo": name, "retorno": timing}
                correlations.append({**keys, **stats.correlations(x, y)})
                bootstrap.extend(_block_bootstrap_rows(keys, x, y))
                newey_west.extend(_newey_west_rows(keys, x, y, pair["fold"].to_numpy()))
    return {
        "b_h1_correlacoes": pd.DataFrame(correlations),
        "b_h1_bootstrap_blocos": pd.DataFrame(bootstrap),
        "b_h1_newey_west": pd.DataFrame(newey_west),
        "b_h1_estacionariedade": pd.DataFrame(stationarity),
    }


def _stationarity_rows(period: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """ADF/KPSS do sentimento (bruto e sem a média de cada bloco) e do retorno diário."""
    folds = frame["fold"].to_numpy()
    rows: list[dict[str, Any]] = []
    for model, name in MODEL_LABELS.items():
        series = frame[model].to_numpy()
        variants = {
            f"sentimento — {name}": series,
            f"sentimento — {name}, {DEMEANED}": stats.demean_by_group(series, folds),
        }
        rows.extend(
            {"periodo": period, "serie": label, **stats.stationarity_tests(values)}
            for label, values in variants.items()
        )
    returns = frame["ret_same_day"].dropna().to_numpy()
    rows.append(
        {
            "periodo": period,
            "serie": "retorno diário do Ibovespa",
            **stats.stationarity_tests(returns),
        }
    )
    return rows


def _newey_west_rows(
    keys: dict[str, str], x: np.ndarray, y: np.ndarray, folds: np.ndarray
) -> list[dict[str, Any]]:
    """Regressões com erros HAC (Newey-West).

    Defasagens pela regra usual e de sensibilidade; e, com as usuais, o sentimento sem a
    média do bloco (equivale a um efeito fixo por bloco).
    """
    default_lags = stats.newey_west_default_lags(len(x))
    rows: list[dict[str, Any]] = [
        {**keys, "sentimento": "bruto", **stats.newey_west_regression(y, x, lags)}
        for lags in (default_lags, *NEWEY_WEST_EXTRA_LAGS)
    ]
    demeaned = stats.demean_by_group(x, folds)
    rows.append(
        {**keys, "sentimento": DEMEANED, **stats.newey_west_regression(y, demeaned, default_lags)}
    )
    return rows


def _block_bootstrap_rows(
    keys: dict[str, str], x: np.ndarray, y: np.ndarray
) -> list[dict[str, Any]]:
    main_block = stats.cube_root_block_size(len(x))
    rows = []
    for block in (main_block, *BLOCK_SIZES_EXTRA):
        for stat_name, statistic in (("pearson", stats.pearson), ("spearman", stats.spearman)):
            low, high = stats.moving_block_bootstrap_ci(x, y, statistic, block)
            rows.append(
                {
                    **keys,
                    "estatistica": stat_name,
                    "estimativa": statistic(x, y),
                    "bloco": block,
                    "regra_bloco": "n^(1/3) (principal)"
                    if block == main_block
                    else "sensibilidade",
                    "ic95_inferior": low,
                    "ic95_superior": high,
                    "ic_inclui_zero": low <= 0 <= high,
                    "reamostragens": stats.N_BOOTSTRAP_VERIFICATION,
                    "semente": RANDOM_SEED,
                }
            )
    return rows


# --------------------------------------------------------------------------- (c)


def check_backtest(oof_article: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """(c) Grade da Tabela 4 com limiares do período inteiro (artigo) e dos 60 pregões anteriores."""
    grids = []
    for mode, description in (
        ("full_sample", "artigo: quantis do período inteiro"),
        ("rolling", "sem look-ahead: quantis dos 60 pregões anteriores"),
    ):
        grid = backtest.robustness_grid(oof_article, threshold_mode=mode)
        grids.append(grid.assign(limiar=description, modelo=grid["model"].map(MODEL_LABELS)))
    result = pd.concat(grids, ignore_index=True).drop(columns=["model", "strategy"])
    same_period = benchmark[benchmark["cenario"] == SAME_PERIOD_BENCHMARK]
    same_period = same_period.rename(columns={"cenario": "limiar"})
    result = pd.concat([result, same_period.drop(columns=["inicio", "fim", "n"])])
    return result.reset_index(drop=True)


# --------------------------------------------------------------------------- (d)


def check_event_study(event_table: pd.DataFrame, returns_full: pd.DataFrame) -> pd.DataFrame:
    """(d) CAAR com retorno anormal e janela em pregões, ao lado da replicação do artigo."""
    returns = returns_full.set_index("day")["ret"].dropna()
    scenarios: dict[str, dict[str, Any]] = {
        "artigo: retorno bruto, dias corridos, [0, τ]": {"tau_max": 5},
        "anormal (média t−60..t−1), pregões, [0, τ]": {
            "tau_max": 4,
            "window_unit": "trading_days",
            "abnormal": True,
            "n_boot": stats.N_BOOTSTRAP_VERIFICATION,
        },
        "anormal (média t−60..t−1), pregões, [1, τ] (sem o dia do evento)": {
            "tau_max": 4,
            "window_unit": "trading_days",
            "abnormal": True,
            "first_tau": 1,
            "n_boot": stats.N_BOOTSTRAP_VERIFICATION,
        },
    }
    tables = [
        events.caar_by_event_time(event_table, returns, **options).assign(cenario=name)
        for name, options in scenarios.items()
    ]
    return pd.concat(tables, ignore_index=True)


# --------------------------------------------------------------------------- (e)


def check_models(inputs: VerificationInputs) -> dict[str, pd.DataFrame]:
    """(e) Modelos técnico e texto + técnico com o mesmo walk-forward da Tabela 2."""
    labelled = inputs.labelled
    y = labelled["y"].astype(int).to_numpy()
    close = inputs.returns_full.set_index("day")["close"]
    technical = market.technical_features(close).reindex(labelled["day"]).fillna(0.0).to_numpy()
    n_text, n_tech = inputs.text_features.shape[1], technical.shape[1]
    feature_sets = {
        TECHNICAL_ONLY: (technical, range(n_tech)),
        TEXT_AND_TECHNICAL: (
            hstack([inputs.text_features, technical]).tocsr(),
            range(n_text, n_text + n_tech),
        ),
    }
    oofs = {TEXT_ONLY: inputs.text_predictions}
    for name, (features, technical_columns) in feature_sets.items():
        scaled = _scale_technical_columns(models.baseline_models(), technical_columns)
        oofs[name] = models.walk_forward_predictions(features, y, scaled)
    article_window = (labelled["day"] <= ARTICLE_END).to_numpy()
    summary = pd.concat(
        [
            _summarize_with_article_window(y, predictions, article_window).assign(variaveis=name)
            for name, predictions in oofs.items()
        ],
        ignore_index=True,
    )
    summary["modelo"] = summary["model"].map(MODEL_LABELS)
    valid = ~np.isnan(inputs.text_predictions["logreg_l2"])
    summary["mda_sempre_alta"] = float(y[valid].mean())
    return {
        "e_modelos_texto_tecnico": summary.drop(columns=["model"]),
        "e_diferencas_auc": _auc_differences(y, oofs, valid),
    }


def _scale_technical_columns(base_models: dict[str, Any], columns: range) -> dict[str, Any]:
    """Padroniza só as colunas técnicas, dentro de cada bloco de treino (sem vazamento)."""
    scaler = ColumnTransformer(
        [("technical", StandardScaler(with_mean=False), list(columns))],
        remainder="passthrough",
        sparse_threshold=1.0,
    )
    return {
        name: Pipeline([("scale", scaler), ("model", model)]) for name, model in base_models.items()
    }


def _summarize_with_article_window(
    y: np.ndarray, predictions: dict[str, np.ndarray], article_window: np.ndarray
) -> pd.DataFrame:
    """AUC/MDA em 1.552 dias (como a Tabela 2) e só nos 1.341 dias até 30/12/2024."""
    summary = models.summarize_predictions(y, predictions)
    for model, proba in predictions.items():
        valid = ~np.isnan(proba) & article_window
        is_model = summary["model"] == model
        summary.loc[is_model, "auc_artigo_1341d"] = roc_auc_score(y[valid], proba[valid])
        summary.loc[is_model, "mda_artigo_1341d"] = models.mean_directional_accuracy(
            y[valid], proba[valid]
        )
    return summary


def auc_gap(target: np.ndarray, pair: np.ndarray) -> float:
    """AUC da coluna 1 menos a da coluna 0 (NaN se a amostra tiver uma só classe)."""
    if np.unique(target).size < 2:
        return np.nan
    return float(roc_auc_score(target, pair[:, 1]) - roc_auc_score(target, pair[:, 0]))


def _auc_differences(
    y: np.ndarray, oofs: dict[str, dict[str, np.ndarray]], valid: np.ndarray
) -> pd.DataFrame:
    """Diferença de AUC em relação ao só-texto, com IC por bootstrap em blocos pareado."""
    y_valid = y[valid]
    block = stats.cube_root_block_size(len(y_valid))
    rows = []
    for set_name, predictions in oofs.items():
        if set_name == TEXT_ONLY:
            continue
        for model, name in MODEL_LABELS.items():
            # coluna 0 = só texto, coluna 1 = candidato; os blocos sorteiam os mesmos dias
            pairs = np.column_stack([oofs[TEXT_ONLY][model][valid], predictions[model][valid]])
            low, high = stats.moving_block_bootstrap_ci(y_valid, pairs, auc_gap, block)
            rows.append(
                {
                    "modelo": name,
                    "comparacao": f"{set_name} − {TEXT_ONLY}",
                    "diferenca_auc": auc_gap(y_valid, pairs),
                    "ic95_inferior": low,
                    "ic95_superior": high,
                    "ic_inclui_zero": low <= 0 <= high,
                    "bloco": block,
                    "n": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)
