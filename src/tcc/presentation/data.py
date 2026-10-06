"""Dados de cada figura da apresentação (o conteúdo do CSV gravado ao lado dela).

Nada é calculado de outro jeito aqui: as funções só chamam `tcc.models`, `tcc.backtest`,
`tcc.events`, `tcc.stats` e `tcc.verification`, com as opções do artigo ou da verificação
pós-submissão correspondente, e organizam o resultado numa tabela com colunas em português.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from tcc import backtest, events, figures, models, reproduce, stats, verification
from tcc.config import ARTICLE_END, MODEL_LABELS, RANDOM_SEED
from tcc.news import text
from tcc.reproduce import ArticleResults

TRAINING = "treino inicial"
ARTICLE_TEST = "teste: H1, H2 e H3"
H2_ONLY_TEST = "teste: só H2"
POSITIVE_EVENTS = "Sentimento positivo extremo"
NEGATIVE_EVENTS = "Sentimento negativo extremo"
WITH_NONE = 'com "None" (artigo)'
WITHOUT_NONE = 'sem "None" (corrigido)'
THRESHOLDS = {
    "full_sample": "limiares do artigo (quantis de todo o período)",
    "rolling": "sem look-ahead (quantis dos 60 pregões anteriores)",
}
IBOVESPA = "Ibovespa buy-and-hold"


def walk_forward_schedule(results: ArticleResults) -> pd.DataFrame:
    """F3: datas e tamanhos de treino e teste das 4 etapas do walk-forward."""
    schedule = models.walk_forward_schedule(results.labelled["day"])
    return schedule.rename(
        columns={
            "step": "etapa",
            "train_start": "treino_inicio",
            "train_end": "treino_fim",
            "n_train": "pregoes_treino",
            "test_start": "teste_inicio",
            "test_end": "teste_fim",
            "n_test": "pregoes_teste",
        }
    )


def study_periods(
    results: ArticleResults, schedule: pd.DataFrame
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Treino inicial, teste no período do artigo (H1, H2 e H3) e teste só de H2 (2025)."""
    oof_days, article_days = results.oof_full["day"], results.oof_article["day"]
    return {
        TRAINING: (schedule["treino_inicio"].iloc[0], schedule["treino_fim"].iloc[0]),
        ARTICLE_TEST: (article_days.min(), article_days.max()),
        H2_ONLY_TEST: (oof_days[oof_days > article_days.max()].min(), oof_days.max()),
    }


def ibovespa_and_headlines(
    results: ArticleResults, news: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    """F1: fechamento do Ibovespa, manchetes da base limpa por dia e faixa do estudo."""
    headlines = pd.to_datetime(news["date"]).dt.floor("D").value_counts().rename("manchetes")
    prices = results.returns_full.set_index("day")["close"].rename("ibovespa")
    frame = pd.concat([prices, headlines], axis=1).sort_index().rename_axis("dia").reset_index()
    frame["manchetes"] = frame["manchetes"].fillna(0).astype(int)
    frame["pregoes_treino_inicial"] = int(schedule["pregoes_treino"].iloc[0])
    frame["faixa"] = ""
    for name, (start, end) in study_periods(results, schedule).items():
        frame.loc[frame["dia"].between(start, end), "faixa"] = name
    return frame


def daily_sentiment(results: ArticleResults) -> pd.DataFrame:
    """F2: sentimento diário (2p − 1) dos dois modelos e o bloco do walk-forward de cada dia."""
    table = verification.sentiment_table(results.oof_full, results.returns_full)
    long = table.melt(
        id_vars=["day", "fold"],
        value_vars=list(MODEL_LABELS),
        var_name="modelo",
        value_name="sentimento",
    )
    return pd.DataFrame(
        {
            "dia": long["day"],
            "modelo": long["modelo"].map(MODEL_LABELS),
            "sentimento": long["sentimento"],
            "bloco": long["fold"] + 1,
        }
    )


def sentiment_scatter(results: ArticleResults) -> pd.DataFrame:
    """F4a: sentimento da Regressão Logística × retorno do mesmo dia e do dia seguinte.

    Período do artigo (05/08/2019–30/12/2024). r de Pearson com IC 95% por bootstrap em
    blocos móveis (bloco n^(1/3), 2.000 reamostragens, semente 42) e reta de MQO: os
    mesmos números da verificação (b).
    """
    table = verification.sentiment_table(results.oof_full, results.returns_full)
    article = table[table["day"] <= ARTICLE_END]
    panels = []
    for timing, column in verification.RETURN_TIMINGS.items():
        pair = article[["day", figures.FIGURE_4_MODEL, column]].dropna()
        x, y = pair[figures.FIGURE_4_MODEL].to_numpy(), pair[column].to_numpy()
        block = stats.cube_root_block_size(len(x))
        low, high = stats.moving_block_bootstrap_ci(x, y, stats.pearson, block)
        fit = stats.newey_west_regression(y, x, stats.newey_west_default_lags(len(x)))
        panels.append(
            pd.DataFrame(
                {
                    "retorno_de": timing,
                    "dia": pair["day"].to_numpy(),
                    "sentimento": x,
                    "retorno": y,
                    "r": stats.pearson(x, y),
                    "ic95_inferior": low,
                    "ic95_superior": high,
                    "bloco_bootstrap": block,
                    "reamostragens": stats.N_BOOTSTRAP_VERIFICATION,
                    "intercepto": fit["alpha"],
                    "inclinacao": fit["beta"],
                    "n": len(x),
                }
            )
        )
    return pd.concat(panels, ignore_index=True)


def rolling_correlations(results: ArticleResults) -> pd.DataFrame:
    """F4b: correlação móvel de 60 e 90 pregões com o retorno do mesmo dia (Figura 5 do artigo)."""
    merged = figures.sentiment_and_returns(results.sentiment, results.returns_article)
    frames = [
        pd.DataFrame(
            {
                "dia": merged["day"].to_numpy(),
                "janela_pregoes": window,
                "correlacao": stats.rolling_correlation(
                    merged["sentiment"], merged["ret"], window
                ).to_numpy(),
            }
        )
        for window in figures.ROLLING_WINDOWS
    ]
    rolling = pd.concat(frames, ignore_index=True).dropna(subset=["correlacao"])
    return rolling.assign(
        r_periodo=figures.figure_4_correlation(results.sentiment, results.returns_article)
    )


def _labels(results: ArticleResults) -> np.ndarray:
    return results.labelled["y"].astype(int).to_numpy()


def roc_curves(results: ArticleResults) -> pd.DataFrame:
    """F5a: curvas ROC fora da amostra (1.552 dias) com a AUC e o IC da Tabela 2."""
    y = _labels(results)
    table2 = results.table2.set_index("model")
    frames = []
    for model, proba in results.predictions.items():
        valid = ~np.isnan(proba)
        false_positive, true_positive, _ = roc_curve(y[valid], proba[valid])
        frames.append(
            pd.DataFrame(
                {
                    "modelo": MODEL_LABELS[model],
                    "taxa_falsos_positivos": false_positive,
                    "taxa_verdadeiros_positivos": true_positive,
                    "auc": table2.loc[model, "auc"],
                    "auc_ic95_inferior": table2.loc[model, "auc_low"],
                    "auc_ic95_superior": table2.loc[model, "auc_high"],
                    "n_dias": table2.loc[model, "n_obs"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def auc_by_block(results: ArticleResults, schedule: pd.DataFrame) -> pd.DataFrame:
    """F5b: AUC de cada bloco de teste, com IC 95% pelo bootstrap i.i.d. da Tabela 2."""
    y = _labels(results)
    folds = models.fold_ids(len(y))
    rows = []
    for step in schedule.to_dict("records"):
        in_block = folds == step["etapa"] - 1
        block_predictions = {model: proba[in_block] for model, proba in results.predictions.items()}
        summary = models.summarize_predictions(y[in_block], block_predictions)
        rows.append(
            pd.DataFrame(
                {
                    "bloco": step["etapa"],
                    "inicio": step["teste_inicio"],
                    "fim": step["teste_fim"],
                    "modelo": summary["model"].map(MODEL_LABELS),
                    "auc": summary["auc"],
                    "ic95_inferior": summary["auc_low"],
                    "ic95_superior": summary["auc_high"],
                    "n_dias": summary["n_obs"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def event_study(results: ArticleResults) -> pd.DataFrame:
    """F6: CAAR com retorno anormal, τ = 0 a 4 pregões e IC 95% (verificação (d))."""
    returns = results.returns_full.set_index("day")["ret"].dropna()
    caar = events.caar_by_event_time(results.events, returns, **verification.ABNORMAL_EVENT_STUDY)
    groups = {"pos": POSITIVE_EVENTS, "neg": NEGATIVE_EVENTS}
    frames = [
        pd.DataFrame(
            {
                "tau_pregoes": caar["tau"],
                "grupo": name,
                "caar": caar[f"caar_{polarity}_mean"],
                "ic95_inferior": caar[f"caar_{polarity}_ci_low"],
                "ic95_superior": caar[f"caar_{polarity}_ci_high"],
                "n_eventos": caar[f"n_events_{polarity}"],
                "reamostragens": caar["n_boot"],
                "semente": RANDOM_SEED,
            }
        )
        for polarity, name in groups.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _curve(
    name: str, days: pd.Series, equity: pd.Series, trading_days: pd.Series, metrics: dict[str, Any]
) -> pd.DataFrame:
    curve = backtest.realized_equity_curve(days, equity, trading_days)
    return pd.DataFrame(
        {
            "data": curve["date"],
            "serie": name,
            "patrimonio": curve["equity"],
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
        }
    )


def equity_curves(results: ArticleResults, threshold_mode: str) -> pd.DataFrame:
    """F7: patrimônio das estratégias da Tabela 3 e do Ibovespa nas mesmas linhas.

    Regra de quantil (lag 0, quantil 0,90) com os limiares do artigo (`"full_sample"`) ou
    sem look-ahead (`"rolling"`); o Ibovespa é o benchmark do mesmo período da verificação (a).
    Cada valor é o patrimônio no fechamento do pregão em que o retorno se realiza.
    """
    oof, trading_days = results.oof_article, results.returns_full["day"]
    frames = []
    for model, name in MODEL_LABELS.items():
        run = backtest.run_quantile_strategy(
            oof[oof["model"] == model],
            backtest.TABLE3_QUANTILE,
            backtest.TABLE3_LAG,
            threshold_mode,
        )
        frames.append(
            _curve(name, run["day"], run["equity"], trading_days, backtest.strategy_metrics(run))
        )
    benchmark = verification.same_period_benchmark_returns(oof)
    frames.append(
        _curve(
            IBOVESPA,
            pd.Series(benchmark.index),
            backtest.buy_and_hold_equity(benchmark),
            trading_days,
            backtest.buy_and_hold_metrics(benchmark),
        )
    )
    return pd.concat(frames, ignore_index=True).assign(limiar=THRESHOLDS[threshold_mode])


def sharpe_grid(results: ArticleResults) -> pd.DataFrame:
    """F8: Sharpe das 12 configurações da Tabela 4, com e sem look-ahead (verificação (c))."""
    inputs = reproduce.verification_inputs(results)
    grid = verification.check_backtest(results.oof_article, verification.check_benchmark(inputs))
    is_benchmark = grid["limiar"] == verification.SAME_PERIOD_BENCHMARK
    strategies = grid[~is_benchmark]
    descriptions = {
        "artigo: quantis do período inteiro": THRESHOLDS["full_sample"],
        "sem look-ahead: quantis dos 60 pregões anteriores": THRESHOLDS["rolling"],
    }
    return pd.DataFrame(
        {
            "limiar": strategies["limiar"].map(descriptions),
            "modelo": strategies["modelo"],
            "lag": strategies["lag"].astype(int),
            "quantil": strategies["event_q"],
            "sharpe": strategies["sharpe"],
            "cagr": strategies["cagr"],
            "sharpe_ibovespa": float(grid.loc[is_benchmark, "sharpe"].iloc[0]),
        }
    ).reset_index(drop=True)


def none_token_auc(
    results: ArticleResults, news: pd.DataFrame, ibovespa: pd.DataFrame
) -> pd.DataFrame:
    """F9: AUC com e sem o token espúrio "None" (bug 12), com o IC da Tabela 2.

    Sem o "None", a matriz TF-IDF é refeita a partir da base limpa e passa pelo mesmo
    walk-forward e pela mesma avaliação da Tabela 2.
    """
    matrix, index, vocabulary = text.tfidf_matrix(text.daily_documents(news, missing_text=""))
    labelled, _, predictions = reproduce.text_walk_forward(matrix, index, ibovespa)
    fixed = models.summarize_predictions(labelled["y"].astype(int).to_numpy(), predictions)
    versions = (
        (WITH_NONE, results.table2, results.text_features.shape[1]),
        (WITHOUT_NONE, fixed, len(vocabulary)),
    )
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "modelo": summary["model"].map(MODEL_LABELS),
                    "versao": version,
                    "auc": summary["auc"],
                    "ic95_inferior": summary["auc_low"],
                    "ic95_superior": summary["auc_high"],
                    "mda": summary["mda"],
                    "n_dias": summary["n_obs"],
                    "termos": terms,
                }
            )
            for version, summary, terms in versions
        ],
        ignore_index=True,
    )
