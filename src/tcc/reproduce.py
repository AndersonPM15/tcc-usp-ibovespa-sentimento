"""Reprodução completa: das três entradas às tabelas, figuras e verificações.

Etapas: rótulos → walk-forward (Tabela 2) → série diária de p → eventos → backtests →
tabelas e figuras do artigo → verificações (a)–(e) → comparação com os números publicados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tcc import article, backtest, datasets, events, figures, market, models, verification
from tcc.config import MODEL_LABELS, Settings

TABLE_DECIMAL_COLUMNS = [
    "cagr",
    "sharpe",
    "vol_anual",
    "max_drawdown",
    "turnover",
    "hit_rate",
    "exposure",
    "total_cost",
]


@dataclass(frozen=True)
class ArticleResults:
    """Resultados do artigo calculados a partir das entradas."""

    labelled: pd.DataFrame
    text_features: Any
    predictions: dict[str, np.ndarray]
    table1: pd.DataFrame
    table2: pd.DataFrame
    table3: pd.DataFrame
    table4: pd.DataFrame
    oof_full: pd.DataFrame
    oof_article: pd.DataFrame
    returns_full: pd.DataFrame
    returns_article: pd.DataFrame
    sentiment: pd.DataFrame
    events: pd.DataFrame
    caar: pd.DataFrame
    curves: pd.DataFrame
    threshold_summary: pd.DataFrame
    figure4_correlation: float


def compute_article(settings: Settings) -> ArticleResults:
    """Calcula tudo o que entra nas tabelas e figuras do artigo."""
    matrix, index = datasets.read_tfidf(settings)
    ibovespa = datasets.read_ibovespa(settings)
    labels = market.build_labels(index, ibovespa)
    has_label = labels["y"].notna().to_numpy()
    labelled = labels.loc[has_label].reset_index(drop=True)
    text_features = matrix[has_label]
    y = labelled["y"].astype(int).to_numpy()
    predictions = models.walk_forward_predictions(text_features, y, models.baseline_models())

    oof_full = models.oof_frame(labelled, predictions)
    oof_article = datasets.clamp_period(oof_full, "day")
    returns_full = market.daily_returns(ibovespa)
    returns_article = datasets.clamp_period(returns_full, "day")
    sentiment = oof_article.assign(sentiment=oof_article["proba"] * 2 - 1)
    event_table = events.detect_events(oof_article, returns_article)
    caar = events.caar_by_event_time(event_table, returns_article.set_index("day")["ret"].dropna())
    curves, threshold_summary = backtest.threshold_backtest_curves(oof_article, returns_article)
    return ArticleResults(
        labelled=labelled,
        text_features=text_features,
        predictions=predictions,
        table1=_table1(returns_article, sentiment),
        table2=models.summarize_predictions(y, predictions),
        table3=_table3(oof_article, returns_article),
        table4=figures.format_decimals(
            backtest.robustness_grid(oof_article), TABLE_DECIMAL_COLUMNS
        ),
        oof_full=oof_full,
        oof_article=oof_article,
        returns_full=returns_full,
        returns_article=returns_article,
        sentiment=sentiment,
        events=event_table,
        caar=caar,
        curves=curves,
        threshold_summary=threshold_summary,
        figure4_correlation=figures.figure_4_correlation(sentiment, returns_article),
    )


def _table1(returns: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Tabela 1: pregões do Ibovespa, dias com sentimento e interseção (2018–2024)."""
    trading_days, sentiment_days = set(returns["day"]), set(sentiment["day"])
    return pd.DataFrame(
        {
            "Conjunto": ["Pregões Ibov", "Dias com sentimento", "Interseção"],
            "Dias": [len(trading_days), len(sentiment_days), len(trading_days & sentiment_days)],
        }
    )


def _table3(oof: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Tabela 3: regra de quantil (lag 0, q 0,90) e o buy-and-hold de 2018 a 2024."""
    rows: list[dict[str, Any]] = [
        {
            "modelo": model,
            "dataset": "backtest_daily",
            "strategy": backtest.ARTICLE_STRATEGY.name,
            **backtest.strategy_metrics(
                backtest.run_quantile_strategy(oof[oof["model"] == model], event_q=0.90, lag=0)
            ),
        }
        for model in MODEL_LABELS
    ]
    benchmark = returns.set_index("day")["ret"].dropna()
    rows.append(
        {
            "modelo": "ibov_buyhold",
            "dataset": "benchmark",
            "strategy": "—",
            **backtest.buy_and_hold_metrics(benchmark, normalize_to_first=True),
        }
    )
    return figures.format_decimals(pd.DataFrame(rows), TABLE_DECIMAL_COLUMNS)


def _table2_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Tabela 2 com nomes dos modelos e 3 casas (o CSV numérico vai junto)."""
    display = summary.assign(model=summary["model"].map(MODEL_LABELS))
    columns = ["auc", "auc_low", "auc_high", "mda", "mda_low", "mda_high"]
    return figures.format_decimals(display, columns)


def write_article_outputs(results: ArticleResults, settings: Settings) -> None:
    """Grava as Tabelas 1–4 e as Figuras 1–9."""
    output = settings.figures_dir
    output.mkdir(parents=True, exist_ok=True)
    figures.write_table(
        results.table1, output, "Tabela_1_amostra", "Tabela 1 – Amostra (2018–2024)"
    )
    results.table2.to_csv(output / "Tabela_2_auc_mda_valores.csv", index=False)
    figures.write_table(
        _table2_for_display(results.table2),
        output,
        "Tabela_2_auc_mda",
        "Tabela 2 – AUC e MDA fora da amostra (IC 95%, bootstrap i.i.d.)",
    )
    figures.write_table(
        results.table3, output, "Tabela_3_backtest_padrao", "Tabela 3 – Métricas (cenário padrão)"
    )
    figures.write_table(
        results.table4, output, "Tabela_4_robustez_backtest", "Tabela 4 – Robustez (lag × quantil)"
    )
    figures.figure_1_ibovespa_events(results.returns_article, results.events, output)
    figures.figure_2_daily_sentiment(results.sentiment, output)
    figures.figure_3_model_comparison(results.threshold_summary, output)
    figures.figure_4_scatter(results.sentiment, results.returns_article, output)
    figures.figure_5_rolling_correlation(results.sentiment, results.returns_article, output)
    figures.figure_6_distribution(results.sentiment, output)
    figures.figure_7a_car_boxplot(results.events, output)
    figures.figure_7b_caar(results.caar, output)
    figures.figure_8_backtest(results.curves, backtest.ARTICLE_STRATEGY.name, output)
    figures.figure_9_rolling_robustness(results.sentiment, results.returns_article, output)


def write_verifications(results: ArticleResults, settings: Settings) -> dict[str, pd.DataFrame]:
    """Executa e grava as verificações pós-submissão (a)–(e)."""
    tables = verification.run_verifications(
        verification.VerificationInputs(
            oof_full=results.oof_full,
            oof_article=results.oof_article,
            returns_full=results.returns_full,
            returns_article=results.returns_article,
            events=results.events,
            labelled=results.labelled,
            text_features=results.text_features,
            text_predictions=results.predictions,
        )
    )
    settings.verification_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(settings.verification_dir / f"{name}.csv", index=False)
    return tables


def compare_with_article(results: ArticleResults) -> list[article.Check]:
    """Compara cada tabela e figura reproduzida com o publicado."""
    keys = ["pregoes_ibovespa", "dias_com_sentimento", "intersecao"]
    counts = {key: int(value) for key, value in zip(keys, results.table1["Dias"], strict=True)}
    return [
        article.check_table1(counts),
        article.check_table2(results.table2),
        article.check_formatted_table("Tabela 3", results.table3, "tabela_3"),
        article.check_formatted_table("Tabela 4", results.table4, "tabela_4"),
        article.check_figures_3_and_8(results.curves, results.threshold_summary),
        article.check_figure4(results.figure4_correlation),
        article.check_events(results.events),
        article.check_figure7b(results.caar),
    ]


def run(settings: Settings, with_verifications: bool = True) -> list[article.Check]:
    """Reproduz o artigo (e, por padrão, as verificações) e devolve as comparações."""
    results = compute_article(settings)
    write_article_outputs(results, settings)
    if with_verifications:
        write_verifications(results, settings)
    return compare_with_article(results)
